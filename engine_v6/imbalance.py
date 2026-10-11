"""Offline book imbalance from the research dataset.

    from engine_v6.imbalance import token_imbalance, book_imbalance, orient

WHAT IT IS
----------
``full_depth_bid_notional`` and ``full_depth_ask_notional`` are ready-made scalar
columns in the dataset: notional resting on each side of the token book. The
normalised difference is the strategy's core feature::

    imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)

The column names say "full depth", but the dataset's books are capped at 10 levels per
side and 82% of snapshots saturate that cap, so this is really TOP-10 notional. That
detail matters, because the live threshold only transfers if the live feed measures the
same thing. Measured on 44 days / 12,670 markets the relation to winning is U-shaped:

    |imbalance| deciles    1     2     3     4     5     6     7     8
    win rate (%)         93.6  93.1  79.5  67.9  70.3  75.1  93.3  94.4

Extremes are strong, the middle is weak, and the middle LOSES money because the entry
sits near 0.79 while the win rate is only about 70%. Hence the gate
``|imbalance| >= 0.30``, which removes the balanced-book trades.

ORIENTATION
-----------
The raw column is in UP-token space. Buying DOWN requires flipping the sign, because a
book that is bid-heavy on the UP token is ask-heavy on the DOWN token. ``orient`` is the
only sanctioned way to do that, and the convention is asserted by the test suite: a gate
on ``|imbalance|`` looks symmetric and would silently accept a sign error.
"""
from __future__ import annotations

import pyarrow.parquet as pq

#: columns needed from the feature files
_NEEDED = ("market_id", "time_to_deadline_ms",
           "full_depth_bid_notional", "full_depth_ask_notional")


def book_imbalance(bid_notional: float | None,
                   ask_notional: float | None) -> float | None:
    """(bid - ask) / (bid + ask); None when the book is empty or unusable."""
    if bid_notional is None or ask_notional is None:
        return None
    total = bid_notional + ask_notional
    if total <= 0:
        return None
    return (bid_notional - ask_notional) / total


def orient(imbalance: float | None, side: str) -> float | None:
    """Flip the sign so it always favours the side being bought."""
    if imbalance is None:
        return None
    return imbalance if side == "UP" else -imbalance


def token_imbalance(root: str, checkpoint_s: float = 180.0,
                    tolerance_ms: float = 1500.0) -> dict[str, float]:
    """Map ``market_id`` -> imbalance at the decision tick.

    Keyed on ``market_id`` because ``window_key`` is corrupted in this dataset (it holds
    the literal string ``"et"``), so joining on the obvious-looking key yields zero
    matches. The decision tick is the one nearest ``checkpoint_s`` seconds before the
    close, matching ``Market.tick_at_t_minus`` in the loader.

    Reads four scalar columns only, so it is far cheaper than a full market load.
    """
    from engine_v6 import dataset

    out: dict[str, float] = {}
    target = checkpoint_s * 1000.0
    for path in dataset.feature_files(root):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in _NEEDED):
            continue
        data = pq.read_table(path, columns=list(_NEEDED)).to_pydict()
        ids = data.get("market_id") or []
        if not ids:
            continue
        market_id = str(ids[0])
        deadline = data["time_to_deadline_ms"]
        bids = data["full_depth_bid_notional"]
        asks = data["full_depth_ask_notional"]
        best_i = None
        best_gap = None
        for i, value in enumerate(deadline):
            if value is None:
                continue
            gap = abs(value - target)
            if gap <= tolerance_ms and (best_gap is None or gap < best_gap):
                best_i, best_gap = i, gap
        if best_i is None:
            continue
        value = book_imbalance(bids[best_i], asks[best_i])
        if value is not None:
            out[market_id] = value
    return out


def depth_reference(root: str, sample_every: int = 400, max_files: int = 40) -> dict:
    """Measure the dataset's own book depth, so live depth has a reference.

    Returns level-count stats and notional percentiles over a sample of book
    snapshots. Used by the live probe comparison and by the docs.
    """
    from engine_v6 import dataset

    files = dataset.book_files(root)
    bid_levels: list[int] = []
    ask_levels: list[int] = []
    for path in files[::sample_every][:max_files]:
        try:
            data = pq.read_table(
                path, columns=["side_bids", "side_asks"]).to_pydict()
        except (OSError, KeyError):
            continue
        for bids, asks in zip(data["side_bids"], data["side_asks"]):
            bid_levels.append(len(bids) if bids else 0)
            ask_levels.append(len(asks) if asks else 0)
    if not ask_levels:
        return {}
    ordered = sorted(ask_levels)
    n = len(ordered)
    return {"snapshots": n, "median_levels": ordered[n // 2],
            "max_levels": ordered[-1],
            "share_at_10": sum(1 for v in ask_levels if v >= 10) / n}
