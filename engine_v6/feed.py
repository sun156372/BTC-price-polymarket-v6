"""Public read-only market data, with depth measured rather than assumed.

    from engine_v6.feed import PolymarketFeed, SpotFeed, DepthReport

NO API KEYS, NO SIGNING, NO ORDER PLACEMENT. Every call is a GET against a public
endpoint, so "paper only" is structural rather than a configuration flag.

WHY THIS FILE EXISTS
--------------------
The V4 live loop called ``GET /book`` and used whatever came back, without knowing how
many levels that was or what notional it represented. The strategy's core gate is a
depth IMBALANCE with a threshold of 0.30, calibrated on the research dataset. The
diagnostic established that the dataset's books are capped at 10 levels per side, with
82% of snapshots exactly at that cap - so the gate was fitted to a TOP-10 view.

If the live endpoint returns a different number of levels, or the same number but
sampled at a different price region, then ``|imbalance| >= 0.30`` means something
different and the entry frequency changes accordingly. That is precisely what the
live session showed: 90% of windows rejected by the imbalance gate against 71% in
backtest.

So this module never assumes. It:

* records how many levels actually arrived, per side, on every fetch
* computes notional over a configurable cap, defaulting to 10 to match the dataset
* exposes ``DepthReport`` so a probe can compare live depth against the dataset
* tries a larger cap and reports whether the endpoint honours it or silently clamps

The last point matters: an endpoint that silently ignores a ``depth`` parameter would
otherwise look like it had been asked for more and given less.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com"
BINANCE = "https://api.binance.com"

SLUG_TEMPLATE = "btc-updown-5m-{epoch}"
WINDOW_SECONDS = 300

DEFAULT_TIMEOUT = 8.0
DEFAULT_RETRIES = 2

#: The research dataset's books hold AT MOST 10 levels per side, and 82% of its
#: snapshots sit exactly at that cap. The 0.30 imbalance threshold was fitted on
#: notional over that top-10 view, so 10 is the definition the threshold means.
DATASET_DEPTH = 10

#: The live endpoint, by contrast, is DOCUMENTED as returning the complete book with
#: all resting levels, and exposes no depth parameter. So the live book is probably
#: DEEPER than the dataset's, not shallower.
#:
#: That direction matters and is easy to get backwards. Imbalance is a ratio, and a
#: deeper book pulls it toward zero: adding levels far from the touch dilutes the
#: top-of-book lopsidedness that the signal is about. Feeding a full-depth live book
#: into a threshold fitted on top-10 therefore measures a smaller quantity and refuses
#: far more windows - which is exactly the live symptom (90% refused by this gate
#: against 71% in backtest).
#:
#: The fix is not to assume either shape. It is to compute the imbalance over a NAMED
#: cap on both sides and record several caps at once, so the choice is measured. This
#: constant is that default name; ``tools_v6/depth_check.py`` settles which is right on
#: the trading machine.
DEFAULT_IMBALANCE_CAP = DATASET_DEPTH

#: Documented rate limits, per 10 seconds, so polling can be budgeted instead of
#: guessed. Cloudflare THROTTLES rather than rejects, which adds latency instead of
#: raising an error - and added latency silently corrupts a latency-sensitive decision.
RATE_LIMITS = {
    "book": 1500,          # GET /book
    "books": 500,          # POST /books, up to 500 tokens per request
    "clob_general": 9000,
    "gamma_markets": 300,
}


class FeedError(RuntimeError):
    """Any failure to obtain usable market data. Never raised as a silent default."""


@dataclass
class FetchStats:
    """What the endpoint actually returned, so depth can be audited afterwards."""

    calls: int = 0
    failures: int = 0
    bid_levels: list[int] = field(default_factory=list)
    ask_levels: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    #: True when a larger ``depth`` was requested but the response was not larger
    clamped: bool = False

    def note(self, level_counts: tuple[int, int], latency_ms: float) -> None:
        self.calls += 1
        self.bid_levels.append(level_counts[0])
        self.ask_levels.append(level_counts[1])
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        def describe(values: list[int]) -> dict:
            if not values:
                return {}
            ordered = sorted(values)
            n = len(ordered)
            return {"min": ordered[0], "median": ordered[n // 2],
                    "p90": ordered[9 * n // 10], "max": ordered[-1],
                    "at_10": sum(1 for v in values if v >= DATASET_DEPTH) / n}
        lat = sorted(self.latencies_ms)
        return {
            "calls": self.calls, "failures": self.failures,
            "bid_levels": describe(self.bid_levels),
            "ask_levels": describe(self.ask_levels),
            "latency_ms_median": (lat[len(lat) // 2] if lat else None),
            "latency_ms_p90": (lat[9 * len(lat) // 10] if lat else None),
            "depth_was_clamped": self.clamped,
        }


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT,
              retries: int = DEFAULT_RETRIES, params: dict | None = None):
    """GET a JSON document with bounded retries. Raises FeedError on failure."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "dsh-v6-paper/1.0",
                              "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
    raise FeedError(f"GET {url} failed after {retries + 1} attempts: {last}")


# ---------------------------------------------------------------- instruments


@dataclass
class MarketInfo:
    slug: str
    condition_id: str
    up_token: str
    down_token: str
    start_ts: float
    end_ts: float
    outcomes: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        now = time.time()
        return self.start_ts - 30 <= now <= self.end_ts + 30


@dataclass
class Book:
    """One side's book, with the depth it actually arrived with.

    Best bid and ask are DERIVED with max/min, never taken from an array index. The
    venue documents ``bids`` as sorted descending and ``asks`` ascending, but that
    ordering is not reliable in practice: a stale long-shot order at 0.99 can sit at
    ``asks[0]`` while the real best ask of 0.53 sits in the middle. Indexing would then
    read the worst price in the book as the touch, which corrupts the mid, the spread,
    and every move measured against them - silently, with no error.

    ``fetch`` does sort the arrays for display, but this class does not depend on it.
    """

    token_id: str
    bids: list[tuple[float, float]]          # (price, size)
    asks: list[tuple[float, float]]
    fetched_at: float
    requested_depth: int | None = None

    @property
    def bid_depth(self) -> int:
        return len(self.bids)

    @property
    def ask_depth(self) -> int:
        return len(self.asks)

    @property
    def best_bid(self) -> float | None:
        return max((price for price, _ in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((price for price, _ in self.asks), default=None)

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def size_at_best(self, side: str) -> float:
        """Size resting AT the touch only.

        A market order can only fill at the best price level, so this is the quantity
        immediately available. Summing every level would count size that needs the
        book walked to reach.
        """
        if side == "bids":
            best = self.best_bid
            return sum(size for price, size in self.bids if price == best)
        best = self.best_ask
        return sum(size for price, size in self.asks if price == best)

    def notional(self, side: str, cap: int | None = None) -> float:
        """Notional on one side, over at most ``cap`` levels.

        ``cap`` defaults to the whole book. Pass ``DATASET_DEPTH`` to reproduce the
        research dataset's top-10 definition exactly, which is what the 0.30 threshold
        was fitted to.
        """
        levels = self.bids if side == "bids" else self.asks
        if cap is not None:
            levels = self._best_n(side, cap)
        return sum(price * size for price, size in levels)

    def _best_n(self, side: str, cap: int) -> list[tuple[float, float]]:
        """The ``cap`` best levels, selected by price rather than by array position."""
        levels = self.bids if side == "bids" else self.asks
        ordered = sorted(levels, key=lambda lv: lv[0],
                         reverse=(side == "bids"))
        return ordered[:cap]

    def imbalance(self, cap: int | None = None) -> float | None:
        """(bid - ask) / (bid + ask) over the ``cap`` best levels per side."""
        b = self.notional("bids", cap)
        a = self.notional("asks", cap)
        if b + a <= 0:
            return None
        return (b - a) / (b + a)

    def cost_to_buy(self, shares: float) -> float | None:
        """Walk the asks in price order for ``shares``. None if it cannot fill."""
        remaining = shares
        cost = 0.0
        for price, size in sorted(self.asks, key=lambda lv: lv[0]):
            if remaining <= 0:
                break
            take = min(remaining, size)
            cost += take * price
            remaining -= take
        return None if remaining > 1e-9 else cost

    def depth_span(self) -> tuple[float, float] | None:
        """How far the recorded levels reach from the touch, per side."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None or not self.bids or not self.asks:
            return None
        return (bid - min(p for p, _ in self.bids), max(p for p, _ in self.asks) - ask)

    def is_crossed(self) -> bool:
        """True when the best bid is at or above the best ask - a data fault."""
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask


@dataclass
class DepthReport:
    """Live depth measured against the dataset's definition."""

    up: Book
    down: Book
    stats: FetchStats

    def lines(self) -> list[str]:
        out = []
        for label, book in (("UP", self.up), ("DOWN", self.down)):
            span = book.depth_span()
            out.append(
                f"  {label:<5} levels {book.bid_depth:>3} bid / {book.ask_depth:>3} ask"
                f"   best {book.best_bid} / {book.best_ask}   spread {book.spread}")
            if span:
                out.append(f"        levels reach {span[0]:.4f} below / "
                           f"{span[1]:.4f} above the touch")
            out.append(
                f"        notional top-{DATASET_DEPTH}: bid "
                f"{book.notional('bids', DATASET_DEPTH):>10,.2f}  ask "
                f"{book.notional('asks', DATASET_DEPTH):>10,.2f}  "
                f"imbalance {book.imbalance(DATASET_DEPTH)}")
            out.append(
                f"        notional whole book: bid "
                f"{book.notional('bids'):>10,.2f}  ask "
                f"{book.notional('asks'):>10,.2f}  "
                f"imbalance {book.imbalance()}")
        return out


# ------------------------------------------------------------------- the feed


class PolymarketFeed:
    """Read-only access to Polymarket's public market data."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT,
                 depth: int = DATASET_DEPTH):
        self.timeout = timeout
        self.depth = depth
        self.stats = FetchStats()

    # ------------------------------------------------------------- discovery
    def market_for_epoch(self, epoch: int) -> MarketInfo | None:
        slug = SLUG_TEMPLATE.format(epoch=int(epoch))
        payload = _get_json(f"{GAMMA}/markets", timeout=self.timeout,
                            params={"slug": slug})
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        return self._to_market_info(row, slug)

    def _to_market_info(self, row: dict, slug: str) -> MarketInfo:
        def maybe_json(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return None
            return value

        tokens = maybe_json(row.get("clobTokenIds")) or []
        outcomes = maybe_json(row.get("outcomes")) or []
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise FeedError(f"market {slug} returned no usable clobTokenIds: "
                            f"{row.get('clobTokenIds')!r}")
        up_token = down_token = None
        for token, label in zip(tokens, outcomes or ["Up", "Down"]):
            text = str(label).strip().lower()
            if text in ("up", "yes"):
                up_token = token
            elif text in ("down", "no"):
                down_token = token
        if up_token is None or down_token is None:
            up_token, down_token = tokens[0], tokens[1]

        start_ts = _parse_ts(row.get("startDate") or row.get("start_date"))
        end_ts = _parse_ts(row.get("endDate") or row.get("end_date"))
        if start_ts is None:
            try:
                start_ts = float(slug.rsplit("-", 1)[-1])
            except (ValueError, IndexError):
                raise FeedError(f"cannot determine window start for {slug}")
        if end_ts is None:
            end_ts = start_ts + WINDOW_SECONDS
        return MarketInfo(slug=slug,
                          condition_id=str(row.get("conditionId")
                                           or row.get("condition_id") or ""),
                          up_token=str(up_token), down_token=str(down_token),
                          start_ts=start_ts, end_ts=end_ts,
                          outcomes=list(outcomes), raw=row)

    # ----------------------------------------------------------------- books
    def book(self, token_id: str, depth: int | None = None) -> Book:
        """Fetch a book, recording how many levels actually arrived.

        ``depth`` is passed to the endpoint as a hint where supported. A response that
        is not larger than the 10-level baseline is recorded as a clamp, so a silently
        ignored parameter cannot be mistaken for a deep book.
        """
        want = self.depth if depth is None else depth
        params = {"token_id": token_id}
        if want and want != DATASET_DEPTH:
            params["depth"] = want
        started = time.time()
        try:
            payload = _get_json(f"{CLOB}/book", timeout=self.timeout, params=params)
            bids = _levels(payload.get("bids"))
            asks = _levels(payload.get("asks"))
        except FeedError:
            self.stats.failures += 1
            raise
        latency = (time.time() - started) * 1000.0
        bids.sort(key=lambda lv: lv[0], reverse=True)
        asks.sort(key=lambda lv: lv[0])
        self.stats.note((len(bids), len(asks)), latency)
        if want and want > DATASET_DEPTH and max(len(bids), len(asks)) <= DATASET_DEPTH:
            self.stats.clamped = True
        return Book(token_id=token_id, bids=bids, asks=asks,
                    fetched_at=time.time(), requested_depth=want)

    def both_books(self, market: MarketInfo,
                   depth: int | None = None) -> tuple[Book, Book]:
        return (self.book(market.up_token, depth),
                self.book(market.down_token, depth))

    def depth_report(self, market: MarketInfo) -> DepthReport:
        up, down = self.both_books(market)
        return DepthReport(up=up, down=down, stats=self.stats)

    # ------------------------------------------------------------ settlement
    def resolved_outcome(self, market: MarketInfo) -> str | None:
        """'UP', 'DOWN', or None if not yet resolved."""
        try:
            payload = _get_json(f"{GAMMA}/markets", timeout=self.timeout,
                                params={"slug": market.slug})
        except FeedError:
            return None
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        if not row.get("closed"):
            return None
        prices = row.get("outcomePrices")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (ValueError, TypeError):
                return None
        if not isinstance(prices, list) or len(prices) < 2:
            return None
        try:
            values = [float(p) for p in prices]
        except (TypeError, ValueError):
            return None
        return "UP" if values[0] >= values[1] else "DOWN"


def _levels(raw) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for level in raw or []:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size > 0:
            out.append((price, size))
    return out


def _parse_ts(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            timezone.utc).timestamp()
    except (ValueError, TypeError):
        pass
    try:
        return float(text)
    except ValueError:
        return None


# ----------------------------------------------------------------------- spot


@dataclass
class SpotQuote:
    ts: float
    price: float
    source: str

    def age(self) -> float:
        return time.time() - self.ts


class SpotFeed:
    """BTC/USD spot from public endpoints, with a fallback chain.

    The research used a 100ms feed. Polling is coarser, and that is not cosmetic: the
    spot gate requires a 4.5bps move, which accrues in about six seconds at the
    sampled volatility. A poll gap of several seconds therefore loses a large part of
    the measured move, and the measured move is what the gate tests.

    So this class records the timestamp of every quote it returns, and callers must
    compare quotes by TIMESTAMP - using the window-open price against the price at the
    decision instant - rather than implicitly using whatever was last fetched.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.stats = FetchStats()

    def quote(self) -> SpotQuote | None:
        for source, fetch in (("coinbase", self._coinbase),
                              ("binance", self._binance)):
            started = time.time()
            try:
                price = fetch()
            except FeedError:
                self.stats.failures += 1
                continue
            if price:
                self.stats.note((0, 0), (time.time() - started) * 1000.0)
                return SpotQuote(ts=time.time(), price=price, source=source)
        return None

    def _coinbase(self) -> float | None:
        payload = _get_json(f"{COINBASE}/products/BTC-USD/ticker",
                            timeout=self.timeout)
        price = payload.get("price")
        return float(price) if price else None

    def _binance(self) -> float | None:
        payload = _get_json(f"{BINANCE}/api/v3/ticker/price",
                            timeout=self.timeout, params={"symbol": "BTCUSDT"})
        price = payload.get("price")
        return float(price) if price else None
