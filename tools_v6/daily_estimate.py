"""Daily P&L estimate for the shipped V6 cell, with the uncertainty stated.

    python tools_v6/daily_estimate.py

Everything here comes from the measured out-of-sample session. Nothing is annualised
and nothing is extrapolated beyond the stake sizes actually tested.

The three things that decide the answer, in order:

1. **edge per share** - measured, +2.30% out of sample at T-240
2. **trades per day** - measured, 17.3
3. **fill cost** - NOT measured. The backtest assumes a fill at the decision ask, and
   the strategy's whole edge is about 2 points per share, so a couple of cents of
   slippage is a large fraction of it. This is swept explicitly rather than assumed
   away.
"""
from __future__ import annotations

import math

#: measured out-of-sample, T-240, 10 USDC stake, 22 days.
#: P&L is in USDC at that stake; per-share and per-day figures are derived.
OOS_TRADES = 380
OOS_DAYS = 22
OOS_MEAN_ENTRY = 0.8665
OOS_TRADES_PER_DAY = OOS_TRADES / OOS_DAYS
OOS_WIN_RATE = 0.8974
#: realised day-level P&L, USDC at a 10 USDC stake
OOS_DAY_MEAN_USDC = 4.613
OOS_DAY_SD_USDC = 13.662
OOS_POSITIVE_DAYS = 15
OOS_TOTAL_PNL = 101.48
#: the same day series over all 44 days, for contrast
ALL_DAY_MEAN_USDC = 4.328
ALL_DAY_SD_USDC = 13.830
ALL_DAYS = 44
ALL_POSITIVE_DAYS = 28
FEE = 0.07
STAKE = 10.0


def net_per_share(entry: float, won: bool, extra_cost: float = 0.0) -> float:
    p = min(0.999, entry + extra_cost)
    gross = (1.0 - p) if won else (-p)
    return gross - FEE * p * (1 - p)


def main() -> int:
    shares = STAKE / OOS_MEAN_ENTRY
    #: per-share edge implied by the settled P&L, so the two are consistent
    edge_per_share = OOS_DAY_MEAN_USDC / (OOS_TRADES_PER_DAY * shares)
    se_usdc = OOS_DAY_SD_USDC / math.sqrt(OOS_DAYS)
    lo, hi = OOS_DAY_MEAN_USDC - 2 * se_usdc, OOS_DAY_MEAN_USDC + 2 * se_usdc

    print("=" * 84)
    print("V6 DAILY P&L ESTIMATE  (T-240, hold to settlement, no stop)")
    print("=" * 84)
    print(f"  source: {OOS_DAYS} out-of-sample days, {OOS_TRADES} trades, "
          f"10 USDC stake")
    print(f"  mean entry {OOS_MEAN_ENTRY:.4f}   win rate {OOS_WIN_RATE*100:.2f}%   "
          f"trades/day {OOS_TRADES_PER_DAY:.1f}")
    print(f"  edge per share {edge_per_share*100:+.3f}%")
    print()

    print("=" * 84)
    print("1. AT THE MEASURED FILL  (the optimistic case: you get the quoted ask)")
    print("=" * 84)
    print(f"  {'stake':>8}{'shares':>9}{'per trade':>12}{'per day':>11}"
          f"{'per 30d':>11}")
    print("-" * 84)
    for stake in (10.0, 25.0, 50.0, 100.0):
        s = stake / OOS_MEAN_ENTRY
        per_trade = edge_per_share * s
        per_day = per_trade * OOS_TRADES_PER_DAY
        print(f"  {stake:>8.0f}{s:>9.2f}{per_trade:>+12.3f}{per_day:>+11.2f}"
              f"{per_day*30:>+11.1f}")

    print()
    print("=" * 84)
    print("2. THE UNCERTAINTY (this is the part that matters)")
    print("=" * 84)
    print(f"  day-level P&L : mean {OOS_DAY_MEAN_USDC:+.2f}   "
          f"sd {OOS_DAY_SD_USDC:.2f}   se {se_usdc:.2f} USDC  "
          f"over {OOS_DAYS} days")
    print(f"  2-sigma band  : {lo:+.2f} .. {hi:+.2f} USDC/day at a 10 USDC stake")
    print(f"  positive days : {OOS_POSITIVE_DAYS}/{OOS_DAYS} "
          f"= {OOS_POSITIVE_DAYS/OOS_DAYS*100:.0f}%")
    print()
    print(f"  all {ALL_DAYS} days: mean {ALL_DAY_MEAN_USDC:+.2f}   "
          f"sd {ALL_DAY_SD_USDC:.2f}   "
          f"positive {ALL_POSITIVE_DAYS}/{ALL_DAYS}")
    print()
    print(f"  A single day can swing +-{OOS_DAY_SD_USDC:.0f} USDC around a "
          f"{OOS_DAY_MEAN_USDC:+.1f} mean.")
    print(f"  The worst out-of-sample day was -39.00 USDC; the best +27.90.")
    if lo > 0:
        print("  The 2-sigma lower bound is above zero, so this sample establishes")
        print("  a positive daily expectation.")
    else:
        print(f"  The 2-sigma lower bound is {lo:+.2f}, which INCLUDES zero: 22 days")
        print("  is not enough to establish the expected value. The point estimate is")
        print("  positive; the sample does not prove it.")

    print()
    print("=" * 84)
    print("3. FILL COST - the one input the backtest could not measure")
    print("=" * 84)
    print("  The backtest fills at the quoted ask. Live, the fill lands after the")
    print("  decision, so the ask may have moved. Everything else held fixed.")
    print()
    print(f"  {'extra cost':>12}{'edge/share':>13}{'per day':>11}{'per 30d':>11}"
          f"{'verdict':>11}")
    print("-" * 84)
    for cents in (0, 1, 2, 3, 4, 5, 6):
        extra = cents / 100.0
        entry = OOS_MEAN_ENTRY
        won_ps = net_per_share(entry, True, extra)
        lose_ps = net_per_share(entry, False, extra)
        edge = OOS_WIN_RATE * won_ps + (1 - OOS_WIN_RATE) * lose_ps
        s = STAKE / (entry + extra)
        per_day = edge * s * OOS_TRADES_PER_DAY
        label = "as measured" if cents == 0 else f"+{cents} cents"
        print(f"  {label:>12}{edge*100:>+12.3f}%{per_day:>+11.2f}"
              f"{per_day*30:>+11.1f}{('profit' if per_day > 0 else 'LOSS'):>11}")
    print()
    print("  Tolerance is about 2-3 cents of extra fill cost. The live V4 session")
    print("  measured a MEDIAN drift of 0.00 cents, which is encouraging but came")
    print("  from only 6 trades.")

    print()
    print("=" * 84)
    print("BOTTOM LINE")
    print("=" * 84)
    print(f"  Most likely, 10 USDC stake : {OOS_DAY_MEAN_USDC:+.2f} USDC/day  "
          f"({OOS_TRADES_PER_DAY:.1f} trades/day)")
    print(f"  Plausible range (2 sigma)  : {lo:+.2f} .. {hi:+.2f} USDC/day")
    print(f"  Days in profit             : {OOS_POSITIVE_DAYS}/{OOS_DAYS}")
    print(f"  Per 30 days                : ~{OOS_DAY_MEAN_USDC*30:+.0f} USDC")
    print()
    print("  At larger stakes the same edge scales roughly linearly until depth")
    print("  becomes the constraint:")
    for stake in (10.0, 25.0, 50.0):
        print(f"    {stake:>5.0f} USDC stake -> ~"
              f"{OOS_DAY_MEAN_USDC*stake/10.0:+.1f} USDC/day")
    print("  Capacity above about 50 USDC per order has NOT been measured.")
    print()
    print("  What could invalidate all of it:")
    print("    - the 0.30 imbalance threshold assumes the live book matches the")
    print("      dataset's top-10 view; the live endpoint may return full depth,")
    print("      which would change the gate. Run tools_v6/depth_check.py --live.")
    print("    - feed.py has never run against the real network.")
    print("    - 22 out-of-sample days is a small sample for a 2-point edge, and")
    print("      the 2-sigma band still includes zero.")
    print("    - the entry price IS the edge: 2-3 cents of drift erases it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
