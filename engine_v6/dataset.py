"""Offline dataset access for the V6 backtest.

    from engine_v6 import dataset
    for market in dataset.iter_markets(root, horizons=("5m",)): ...

Two dataset defects are handled here rather than in every caller, because both are the
kind that fail quietly:

1. ``window_key`` is corrupted - it holds the literal string ``"et"``. Joining on it
   yields zero matches and no error. The reliable key is ``market_id``.
2. Tick rows are stored with the countdown ASCENDING, i.e. from the close back to the
   open. Iterating them forward reads the tape backwards, and a first-touch exit model
   then reports the LAST touch instead of the first. Ticks are re-sorted to chronological
   order here, and the backtest asserts the order before trusting it.

The slim cache is used when present (about 160 markets/s against 6.5 for the raw files)
and the raw loader otherwise. Cache correctness is proven by ``tools_v6/parity_check.py``
rather than assumed: a cache is only safe once it is shown to equal what it caches.
"""
from __future__ import annotations

import bisect
import dataclasses
import os
from typing import Iterator, Sequence

import pyarrow.parquet as pq

#: dataset directory names inside the extraction root
FEATURES = "polymarket_features_100ms"
BOOKS = "books_100ms"
EXTERNAL = "external_100ms"
RESOLUTION = "resolution"
CACHE_DIR = "_cache"


# ----------------------------------------------------------------- discovery


def _date_dirs(root: str, dataset: str) -> list[str]:
    out = []
    base = os.path.join(root)
    for dirpath, dirnames, _filenames in os.walk(base):
        if os.path.basename(dirpath) == f"dataset={dataset}":
            out.extend(os.path.join(dirpath, d) for d in sorted(dirnames))
            dirnames[:] = []
    return out


def day_files(root: str, dataset: str) -> list[str]:
    """Every parquet under ``dataset=<dataset>``, in date order."""
    files = []
    for date_dir in _date_dirs(root, dataset):
        for name in sorted(os.listdir(date_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(date_dir, name))
    return files


def feature_files(root: str) -> list[str]:
    return day_files(root, FEATURES)


def book_files(root: str) -> list[str]:
    return day_files(root, BOOKS)


def cache_dir(root: str) -> str:
    """Where the slim cache lives, tolerating a workspace root or a dataset root."""
    direct = os.path.join(root, CACHE_DIR)
    if os.path.isdir(direct):
        return direct
    for name in ("micro_v3", "data"):
        nested = os.path.join(root, name, CACHE_DIR)
        if os.path.isdir(nested):
            return nested
    if os.path.basename(os.path.normpath(root)) != "micro_v3":
        parent = os.path.join(os.path.dirname(os.path.normpath(root)), "micro_v3",
                              CACHE_DIR)
        if os.path.isdir(parent):
            return parent
    return direct


def has_cache(root: str) -> bool:
    directory = cache_dir(root)
    return all(os.path.exists(os.path.join(directory, name))
               for name in ("slim_ticks.parquet", "slim_index.parquet"))


# --------------------------------------------------------------------- model


@dataclasses.dataclass(slots=True)
class Tick:
    """One 100ms observation. ``to_deadline_ms`` is the authoritative countdown."""

    ts: float
    to_deadline_ms: float
    up_bid: float | None
    up_ask: float | None
    up_mid: float | None
    down_bid: float | None
    down_ask: float | None
    down_mid: float | None

    def ask(self, side: str) -> float | None:
        return self.up_ask if side == "UP" else self.down_ask

    def bid(self, side: str) -> float | None:
        return self.up_bid if side == "UP" else self.down_bid

    def mid(self, side: str) -> float | None:
        return self.up_mid if side == "UP" else self.down_mid


@dataclasses.dataclass(slots=True)
class RecordedMarket:
    market_id: str
    slug: str
    day: str
    horizon: str
    start_ts: float
    up_won: bool | None
    ticks: list[Tick]
    _deadlines: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        # ticks are chronological, so countdowns DESCEND. Bisecting needs an ASCENDING
        # list, so keep one - and remember the index must be translated:
        #     _deadlines[i] corresponds to ticks[len(ticks) - 1 - i]
        # Indexing both lists with the same number is a silent ~57s error that still
        # returns a plausible price; tools_v6/parity_check.py guards against it.
        self._deadlines = sorted(t.to_deadline_ms for t in self.ticks)

    def _tick_for(self, i: int) -> Tick:
        return self.ticks[len(self.ticks) - 1 - i]

    def first_tick(self) -> Tick | None:
        for tick in self.ticks:
            if tick.to_deadline_ms > 0:
                return tick
        return None

    def first_mid(self) -> float | None:
        tick = self.first_tick()
        return None if tick is None else tick.up_mid

    def tick_at_t_minus(self, t_minus_s: float,
                        tolerance_s: float = 1.0) -> Tick | None:
        """The tick nearest ``t_minus_s`` seconds before the close.

        Identical semantics to the authoritative loader: bisect to the target, then
        consider the three candidates around the insertion point and keep the closest
        one within tolerance.
        """
        if not self.ticks:
            return None
        target = t_minus_s * 1000.0
        index = bisect.bisect_left(self._deadlines, target)
        best = None
        best_gap = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self._deadlines):
                tick = self._tick_for(candidate)
                gap = abs(tick.to_deadline_ms - target)
                if gap <= tolerance_s * 1000.0 and (best_gap is None or gap < best_gap):
                    best, best_gap = tick, gap
        return best


def horizon_of(slug: str) -> str:
    for token in ("-5m-", "-15m-", "-1h-", "-4h-"):
        if token in (slug or ""):
            return token.strip("-")
    return ""


def start_of(slug: str) -> float | None:
    try:
        return float((slug or "").rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return None


# ------------------------------------------------------------------- loading


def load_resolutions(root: str) -> dict[str, dict]:
    """market_id -> resolution row. Keyed on market_id, never on window_key."""
    out: dict[str, dict] = {}
    for path in day_files(root, RESOLUTION):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
            for row in batch.to_pylist():
                market_id = str(row.get("market_id") or "")
                if market_id:
                    out[market_id] = row
    return out


def _market_from_ticks(market_id: str, slug: str, day: str, start_ts: float,
                       up_won: bool | None, ticks: list[Tick]) -> RecordedMarket:
    ticks.sort(key=lambda t: t.ts)
    return RecordedMarket(market_id=market_id, slug=slug, day=day,
                          horizon=horizon_of(slug), start_ts=start_ts,
                          up_won=up_won, ticks=ticks)


def iter_markets(root: str, horizons: Sequence[str] | None = ("5m",)
                 ) -> Iterator[RecordedMarket]:
    """Stream markets from the slim cache when present, else from the raw files."""
    if has_cache(root):
        yield from _iter_from_cache(root, horizons)
    else:
        yield from _iter_from_raw(root, horizons)


def _iter_from_cache(root: str, horizons: Sequence[str] | None
                     ) -> Iterator[RecordedMarket]:
    directory = cache_dir(root)
    index = pq.read_table(os.path.join(directory, "slim_index.parquet")).to_pylist()
    meta = {str(r["market_id"]): r for r in index
            if horizons is None or r["horizon"] in horizons}
    if not meta:
        return
    path = os.path.join(directory, "slim_ticks.parquet")
    pf = pq.ParquetFile(path)
    current: str | None = None
    bucket: list[Tick] = []

    def emit(market_id: str, ticks: list[Tick]) -> RecordedMarket:
        info = meta[market_id]
        return _market_from_ticks(
            market_id, str(info["slug"]), str(info["day"]),
            float(info["start_ts"] or 0.0),
            None if info["up_won"] is None else bool(info["up_won"]), ticks)

    for batch in pf.iter_batches(batch_size=200_000):
        cols = batch.to_pydict()
        for i, market_id in enumerate(cols["market_id"]):
            if market_id != current:
                if current is not None and current in meta and bucket:
                    yield emit(current, bucket)
                current = market_id
                bucket = []
            if market_id not in meta:
                continue
            bucket.append(Tick(
                ts=cols["ts"][i], to_deadline_ms=cols["to_deadline_ms"][i],
                up_bid=cols["up_bid"][i], up_ask=cols["up_ask"][i],
                up_mid=cols["up_mid"][i], down_bid=cols["down_bid"][i],
                down_ask=cols["down_ask"][i], down_mid=cols["down_mid"][i]))
    if current is not None and current in meta and bucket:
        yield emit(current, bucket)


def _iter_from_raw(root: str, horizons: Sequence[str] | None
                   ) -> Iterator[RecordedMarket]:
    resolutions = load_resolutions(root)
    columns = ("market_id", "slug", "timestamp_ms", "time_to_deadline_ms",
               "up_best_bid", "up_best_ask", "up_midquote",
               "down_best_bid", "down_best_ask", "down_midquote")
    for path in feature_files(root):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in names]
        data = pq.read_table(path, columns=cols).to_pydict()
        rows = len(data.get("market_id") or [])
        if not rows:
            continue
        market_id = str(data["market_id"][0])
        slug = str((data.get("slug") or [""])[0] or "")
        horizon = horizon_of(slug)
        if horizons is not None and horizon not in horizons:
            continue
        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]

        def col(name):
            return data.get(name) or [None] * rows

        ts, deadline = col("timestamp_ms"), col("time_to_deadline_ms")
        up_bid, up_ask, up_mid = (col("up_best_bid"), col("up_best_ask"),
                                  col("up_midquote"))
        dn_bid, dn_ask, dn_mid = (col("down_best_bid"), col("down_best_ask"),
                                  col("down_midquote"))
        ticks = []
        for i in range(rows):
            if ts[i] is None or deadline[i] is None:
                continue
            ticks.append(Tick(ts=float(ts[i]) / 1000.0,
                              to_deadline_ms=float(deadline[i]),
                              up_bid=up_bid[i], up_ask=up_ask[i], up_mid=up_mid[i],
                              down_bid=dn_bid[i], down_ask=dn_ask[i],
                              down_mid=dn_mid[i]))
        if not ticks:
            continue
        resolution = resolutions.get(market_id) or {}
        direction = str(resolution.get("outcome_direction") or "").upper()
        up_won = None if direction not in ("UP", "DOWN") else direction == "UP"
        yield _market_from_ticks(market_id, slug, day, start_of(slug) or 0.0,
                                 up_won, ticks)


# ----------------------------------------------------------------------- spot


def spot_index(root: str) -> dict[str, tuple[list[float], list[float]]]:
    """day -> (sorted timestamps, midquote prices). One small file per day."""
    out: dict[str, tuple[list[float], list[float]]] = {}
    for path in day_files(root, EXTERNAL):
        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]
        names = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in ("timestamp_ms", "midquote_price") if c in names]
        if len(cols) < 2:
            continue
        data = pq.read_table(path, columns=cols).to_pydict()
        pairs = []
        for ts, price in zip(data["timestamp_ms"], data["midquote_price"]):
            if ts is None or price is None:
                continue
            try:
                pairs.append((float(ts) / 1000.0, float(price)))
            except (TypeError, ValueError):
                continue
        pairs.sort()
        out[day] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def spot_at(index, ts: float) -> float | None:
    """Price at or before ``ts``; None when the day has no earlier quote."""
    times, prices = index
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else prices[i]
