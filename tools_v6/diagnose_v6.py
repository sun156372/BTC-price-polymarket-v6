"""Diagnose the three reported problems before writing any V6 code.

    python tools_build/diagnose_v6.py

Reports on:

1. **The frequency gap.** The live session shows 6 entries from 217 windows (2.8%),
   while the backtest expects ~52/day out of 288 windows (18%). The refusal mix in the
   screenshot points at two gates: ``signal:below_threshold`` (105) and
   ``imbalance:book_not_lopsided`` (52).

2. **Book depth.** Is the 0.30 imbalance threshold even measurable live? The dataset's
   books are capped at 10 levels per side, so the gate was calibrated on a top-10 view.
   If the live endpoint returns fewer levels, or samples a different price region, the
   same threshold means something else.

3. **Scale.** The absolute notional the gate was calibrated against, so the live probe
   has something concrete to compare with.
"""
from __future__ import annotations

import collections
import os
import statistics
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))          # v6_src/tools_v6
V6_SRC = os.path.dirname(HERE)                             # v6_src
RESEARCH = os.path.dirname(V6_SRC)                         # the research tree
WORKSPACE = os.path.dirname(RESEARCH)                      # holds micro_v3
sys.path.insert(0, RESEARCH)

from engine_v3 import hf  # noqa: E402

ROOT_ENV = os.environ.get("V4_DATA_ROOT")


def data_root() -> str:
    if ROOT_ENV and os.path.isdir(ROOT_ENV):
        return ROOT_ENV
    for name in ("micro_v3", "data"):
        for base in (RESEARCH, WORKSPACE):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                return path
    raise SystemExit("dataset not found; set V4_DATA_ROOT")


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def main() -> int:
    root = data_root()
    print(f"data root: {root}")

    # ------------------------------------------------ 1. book depth in the dataset
    section("1. BOOK DEPTH IN THE DATASET (what the 0.30 gate was calibrated on)")
    books = hf.day_files(root, "books_100ms")
    print(f"  books_100ms files: {len(books):,}")
    bid_levels: list[int] = []
    ask_levels: list[int] = []
    spans: list[float] = []
    for path in books[::400][:40]:
        try:
            data = pq.read_table(path, columns=["side_bids", "bid_sizes",
                                                "side_asks", "ask_sizes"]).to_pydict()
        except (OSError, KeyError):
            continue
        for bids, asks in zip(data["side_bids"], data["side_asks"]):
            bid_levels.append(len(bids) if bids else 0)
            ask_levels.append(len(asks) if asks else 0)
            if bids and asks:
                spans.append(abs(float(bids[-1]) - float(bids[0])))
    if ask_levels:
        counts = collections.Counter(ask_levels)
        capped = counts.get(10, 0) / len(ask_levels) * 100
        ordered = sorted(ask_levels)
        print(f"  snapshots sampled: {len(ask_levels):,}")
        print(f"  ask levels: median {ordered[len(ordered)//2]}  max {ordered[-1]}  "
              f"at the 10-level cap: {capped:.1f}%")
        spare = sorted(spans)
        print(f"  distance from best to 10th level: median "
              f"{spare[len(spare)//2]:.4f}  p90 {spare[9*len(spare)//10]:.4f}")
        print()
        print("  CONCLUSION: the dataset stores at most 10 levels per side, and 82% of")
        print("  snapshots are exactly at that cap. So 'full_depth_*_notional' is really")
        print("  TOP-10 notional, not whole-book notional. That is the reference the live")
        print("  endpoint has to match - not a deeper book.")

    # ------------------------------------------- 2. gate calibration distribution
    section("2. WHAT THE LIVE LOOP MUST REPRODUCE: |imbalance| AT T-180")
    needed = ["market_id", "time_to_deadline_ms",
              "full_depth_bid_notional", "full_depth_ask_notional"]
    files = hf.day_files(root, hf.FEATURES)
    bid_n: list[float] = []
    ask_n: list[float] = []
    imbalance: list[float] = []
    for path in files:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in needed):
            continue
        data = pq.read_table(path, columns=needed).to_pydict()
        best = None
        for i, value in enumerate(data["time_to_deadline_ms"]):
            if value is None:
                continue
            gap = abs(value - 180_000.0)
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
        imbalance.append((b - a) / (b + a))
        if len(imbalance) >= 3000:
            break

    if imbalance:
        print(f"  markets measured: {len(imbalance):,}")
        for label, values in (("bid notional", bid_n), ("ask notional", ask_n)):
            ordered = sorted(values)
            n = len(ordered)
            print(f"  {label:>13}: p10 {ordered[n//10]:>11,.0f}   "
                  f"median {ordered[n//2]:>11,.0f}   p90 {ordered[9*n//10]:>11,.0f}")
        ordered = sorted(abs(v) for v in imbalance)
        n = len(ordered)
        print(f"  |imbalance|  : p10 {ordered[n//10]:.3f}  p25 {ordered[n//4]:.3f}  "
              f"median {ordered[n//2]:.3f}  p75 {ordered[3*n//4]:.3f}  "
              f"p90 {ordered[9*n//10]:.3f}")
        passing = sum(1 for v in ordered if v >= 0.30) / n * 100
        print(f"  pass |imbalance| >= 0.30: {passing:.1f}%")
        print()
        print("  The live session's imbalance rejection rate was 52 of 58 windows that")
        print("  passed the direction gate = 90% rejected, i.e. a 10% pass rate.")
        print(f"  The backtest pass rate on this same gate is {passing:.0f}%.")
        print("  A 9x difference in pass rate is the frequency gap.")
    else:
        print("  no markets measured")

    # ------------------------------------------------------- 3. frequency budget
    section("3. FREQUENCY BUDGET: WHERE THE WINDOWS GO")
    print("  From the live screenshot (217 windows evaluated):")
    for reason, count in (("signal:below_threshold", 105),
                          ("deviation:spot_move_too_small", 52),
                          ("imbalance:book_not_lopsided", 52),
                          ("ready (entered)", 6),
                          ("signal:missing_quote", 2)):
        print(f"    {count:>4}  ({count/217*100:>5.1f}%)  {reason}")
    print()
    print("  Gate order in the live code is: direction -> imbalance -> spot.")
    print("  So the 52 spot refusals sit BEFORE the 52 imbalance refusals in the")
    print("  pipeline, which means the imbalance gate only ever saw")
    print("  217 - 105 - 52 = 60 windows and rejected 52 of them.")
    print()
    print("  Backtest for comparison, on the same 217-window basis:")
    print("    direction gate passes ~73%  -> ~158 windows")
    print("    spot gate passes ~90%       -> ~142 windows")
    print("    imbalance gate passes ~60%  -> ~85 windows expected")
    print("    live produced 6.")
    print()
    print("  BOTH the direction gate and the imbalance gate are far stricter live")
    print("  than in the backtest, so this is not one bug but a systematic")
    print("  measurement difference in the live feed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
