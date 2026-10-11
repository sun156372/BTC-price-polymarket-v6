"""Compare LIVE order book depth against the dataset the gate was fitted on.

    python tools_v6/depth_check.py              # dataset reference only
    python tools_v6/depth_check.py --live       # also fetch a live book and compare

THE QUESTION
------------
The strategy's core gate is ``|book imbalance| >= 0.30``. It was fitted on
``full_depth_bid_notional`` / ``full_depth_ask_notional`` from the research dataset.
Despite the column names, that dataset stores AT MOST 10 levels per side and 82% of
its snapshots sit exactly at that cap, so the gate is really a TOP-10 measure.

That makes the live measurement a real risk rather than a formality:

* if the live endpoint returns fewer levels, the notional is smaller and |imbalance|
  is compressed toward zero, so the same 0.30 refuses far more windows - which is
  what a live session showed (90% refused, against 71% in backtest)
* if it returns MORE levels, the gate is being applied to a different quantity than
  the one it was fitted on, and the fitted threshold does not apply either

This tool measures both sides and prints the verdict. Run it before trusting a
session.

WHAT IS AND IS NOT KNOWABLE FROM HERE
-------------------------------------
The dataset side is measured locally and is exact. The live side requires network
access, so it can only run where the endpoint is reachable - which is the machine that
will trade. That is why this ships as a tool rather than as a baked-in constant.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from engine_v6 import dataset  # noqa: E402
from engine_v6.feed import (  # noqa: E402
    DATASET_DEPTH, FeedError, PolymarketFeed, SLUG_TEMPLATE, WINDOW_SECONDS,
)
from engine_v6.imbalance import depth_reference  # noqa: E402
from paths import resolve_data_root  # noqa: E402


def dataset_side(root: str) -> dict:
    """Measure the dataset's own book depth and imbalance scale."""
    print("=" * 92)
    print("SIDE A - THE RESEARCH DATASET (the reference the gate was fitted on)")
    print("=" * 92)
    depth = depth_reference(root)
    if not depth:
        print("  no book files found")
        return {}
    print(f"  book snapshots sampled : {depth['snapshots']:,}")
    print(f"  levels per side        : median {depth['median_levels']}  "
          f"max {depth['max_levels']}")
    print(f"  share sitting at the 10-level cap : {depth['share_at_10']*100:.1f}%")
    print()
    print("  => the dataset CANNOT supply more than 10 levels per side. Whatever the")
    print("     live endpoint offers, the fitted threshold applies to a top-10 view.")
    return depth


def imbalance_reference(root: str, checkpoint: float) -> None:
    """The |imbalance| distribution and pass rate the live session must reproduce."""
    import pyarrow.parquet as pq

    needed = ["market_id", "time_to_deadline_ms", "full_depth_bid_notional",
              "full_depth_ask_notional"]
    values: list[float] = []
    bid_n: list[float] = []
    ask_n: list[float] = []
    for path in dataset.feature_files(root):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in needed):
            continue
        data = pq.read_table(path, columns=needed).to_pydict()
        best = None
        for i, value in enumerate(data["time_to_deadline_ms"]):
            if value is None:
                continue
            gap = abs(value - checkpoint * 1000.0)
            if gap <= 1500.0 and (best is None or gap < best[0]):
                best = (gap, i)
        if best is None:
            continue
        i = best[1]
        b, a = data["full_depth_bid_notional"][i], data["full_depth_ask_notional"][i]
        if b is None or a is None or (b + a) <= 0:
            continue
        bid_n.append(b)
        ask_n.append(a)
        values.append(abs((b - a) / (b + a)))
        if len(values) >= 3000:
            break
    if not values:
        print("  no imbalance values measured")
        return
    values.sort()
    bid_n.sort()
    ask_n.sort()
    n = len(values)

    def at(xs, q):
        return xs[min(n - 1, int(q * (n - 1)))]

    print()
    print(f"  markets measured at T-{checkpoint:.0f} : {n:,}")
    print(f"  bid notional : p10 {at(bid_n, .10):>10,.0f}   median "
          f"{at(bid_n, .50):>10,.0f}   p90 {at(bid_n, .90):>10,.0f}")
    print(f"  ask notional : p10 {at(ask_n, .10):>10,.0f}   median "
          f"{at(ask_n, .50):>10,.0f}   p90 {at(ask_n, .90):>10,.0f}")
    print(f"  |imbalance|  : p10 {at(values, .10):.3f}  p25 {at(values, .25):.3f}  "
          f"median {at(values, .50):.3f}  p75 {at(values, .75):.3f}  "
          f"p90 {at(values, .90):.3f}")
    passing = sum(1 for v in values if v >= 0.30) / n * 100
    print(f"  share with |imbalance| >= 0.30 : {passing:.1f}%")
    print()
    print("  THIS is the reference. A live session whose |imbalance| median is well")
    print("  below the value above, or whose pass rate is far under it, has a")
    print("  measurement mismatch rather than a market change.")


def live_side() -> bool:
    """Fetch one live book and compare it, side by side, against the reference."""
    print()
    print("=" * 92)
    print("SIDE B - THE LIVE ENDPOINT")
    print("=" * 92)
    feed = PolymarketFeed()
    epoch = int(__import__("time").time() // WINDOW_SECONDS) * WINDOW_SECONDS
    slug = SLUG_TEMPLATE.format(epoch=epoch)
    try:
        market = feed.market_for_epoch(epoch)
    except FeedError as exc:
        print(f"  cannot reach the market endpoint: {exc}")
        print("  run this tool from the machine that will trade.")
        return False
    if market is None:
        print(f"  no market for {slug}")
        return False

    print(f"  market {market.slug}")
    print()
    ok = True
    for label, token in (("UP", market.up_token), ("DOWN", market.down_token)):
        try:
            book = feed.book(token)
        except FeedError as exc:
            print(f"  {label}: fetch failed - {exc}")
            ok = False
            continue
        span = book.depth_span()
        print(f"  {label} book")
        print(f"    levels              : {book.bid_depth} bid / {book.ask_depth} ask")
        if span:
            print(f"    reach from touch    : {span[0]:.4f} down / {span[1]:.4f} up")
        print(f"    best bid / ask      : {book.best_bid} / {book.best_ask}"
              f"   spread {book.spread}")
        print(f"    notional top-{DATASET_DEPTH:<2}     : bid "
              f"{book.notional('bids', DATASET_DEPTH):>10,.2f}   ask "
              f"{book.notional('asks', DATASET_DEPTH):>10,.2f}")
        print(f"    notional whole book : bid {book.notional('bids'):>10,.2f}   "
              f"ask {book.notional('asks'):>10,.2f}")
        print(f"    imbalance top-{DATASET_DEPTH:<2}     : "
              f"{book.imbalance(DATASET_DEPTH)}")
        print(f"    imbalance whole book: {book.imbalance()}")
        if book.is_crossed():
            print("    WARNING: the book is crossed (best bid >= best ask). That is a")
            print("    data fault; any price read from it is unsafe.")
            ok = False
    print()
    print("  Try a larger depth request, to see whether the endpoint honours it:")
    try:
        deep = feed.book(market.up_token, depth=100)
        print(f"    requested 100 -> {deep.bid_depth} bid / {deep.ask_depth} ask")
        if max(deep.bid_depth, deep.ask_depth) <= DATASET_DEPTH:
            print("    => the parameter is IGNORED or the book is capped. The endpoint")
            print("       returns a top-10 view, which MATCHES the dataset, so the")
            print("       threshold is comparable and no refit is needed for depth.")
        else:
            print("    => the endpoint returns DEEPER books than the dataset. The gate")
            print("       was fitted on a shallower view, so its threshold does NOT")
            print("       transfer as-is; refit it on live-measured imbalance.")
    except FeedError as exc:
        print(f"    depth=100 request failed: {exc}")
    print()
    print("  NOTE: the live CLOB is not documented to expose more than about ten")
    print("  levels per side. Full-book depth via this endpoint is therefore not")
    print("  obtainable, and the dataset is capped at the same ten - which means the")
    print("  gate is comparable. The real risk is not the cap, it is whether the")
    print("  endpoint returns REAL levels or a thin, stale view.")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None)
    parser.add_argument("--live", action="store_true",
                        help="also fetch a live book (needs network)")
    parser.add_argument("--checkpoint", type=float, default=180.0)
    args = parser.parse_args(argv)

    root = resolve_data_root(args.root)
    depth = dataset_side(root)
    imbalance_reference(root, args.checkpoint)
    live_ok = True
    if args.live:
        live_ok = live_side()
    else:
        print()
        print("  (pass --live to compare against the live endpoint; that needs network)")
    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    if depth:
        print(f"  The dataset offers at most {depth['max_levels']} levels per side, and")
        print(f"  {depth['share_at_10']*100:.0f}% of snapshots are at that cap. So the gate is a")
        print("  TOP-10 measure by construction, and 'full depth' is not a thing this")
        print("  data can provide - nor is it what the 0.30 threshold means.")
    print("  Whether the live feed is COMPARABLE is answerable only on the trading")
    print("  machine, with --live.")
    return 0 if live_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
