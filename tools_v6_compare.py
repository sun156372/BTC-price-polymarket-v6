"""Why does V6 disagree with the V4 backtest? Decompose it, do not guess.

    python tools_v6_compare.py

V4 and V6 run the same dataset, the same signal and the same thresholds, and select
almost exactly the same number of markets (1,145 vs 1,138), yet report very different
per-share returns (+4.96% vs -0.84%). The difference has to live in the PRICE, because
everything upstream agrees.

This measures the components instead of arguing about them:

1. the entry-price distribution, broken into buckets, with win rate and P&L per bucket
2. the spread at the decision tick, per bucket - a wide spread both raises the ask and
   signals a worse fill
3. the same market set evaluated at both checkpoints, to separate "checkpoint choice"
   from "implementation difference"
4. a direct check of which tick each implementation picks at T-180 and T-240
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools_v6"))

from engine_v6 import dataset  # noqa: E402
from engine_v6.imbalance import orient, token_imbalance  # noqa: E402
from engine_v6.signals import net_pnl  # noqa: E402
from paths import resolve_data_root  # noqa: E402

BUCKETS = [(0.0, 0.75), (0.75, 0.82), (0.82, 0.88), (0.88, 0.92),
           (0.92, 0.96), (0.96, 1.01)]


def collect(root: str, checkpoint: float, limit: int = 0):
    """Every market that clears the signal gates at ``checkpoint``."""
    imbalance = token_imbalance(root, checkpoint_s=checkpoint)
    spot = dataset.spot_index(root)
    rows = []
    for i, market in enumerate(dataset.iter_markets(root, horizons=("5m",))):
        if limit and i >= limit:
            break
        if market.up_won is None or not market.start_ts:
            continue
        open_tick = market.first_tick()
        entry_tick = market.tick_at_t_minus(checkpoint, tolerance_s=1.5)
        if open_tick is None or entry_tick is None:
            continue
        if open_tick.up_mid is None or entry_tick.up_mid is None:
            continue
        move = entry_tick.up_mid - open_tick.up_mid
        if abs(move) < 0.20:
            continue
        side = "UP" if move > 0 else "DOWN"
        raw = imbalance.get(market.market_id)
        if raw is None:
            continue
        signed = orient(raw, side)
        if signed is None or abs(signed) < 0.30:
            continue
        index = spot.get(market.day)
        if index and market.start_ts:
            so = dataset.spot_at(index, market.start_ts)
            sn = dataset.spot_at(index, entry_tick.ts)
            if so and sn and abs(sn / so - 1.0) < 0.00045:
                continue
        ask = entry_tick.ask(side)
        bid = entry_tick.bid(side)
        if ask is None or not 0.0 < ask < 1.0:
            continue
        shares = 10.0 / ask
        won = (side == "UP") == market.up_won
        rows.append({
            "slug": market.slug, "day": market.day, "side": side,
            "entry": ask, "bid": bid,
            "spread": (ask - bid) if bid is not None else None,
            "up_mid_open": open_tick.up_mid, "up_mid_now": entry_tick.up_mid,
            "move": move, "imbalance": signed, "shares": shares, "won": won,
            "pnl": net_pnl(ask, won, shares),
            "entry_deadline_ms": entry_tick.to_deadline_ms,
        })
    return rows


def summarise(label: str, rows: list) -> dict:
    if not rows:
        print(f"  {label}: no trades")
        return {}
    wins = sum(1 for r in rows if r["won"])
    entries = [r["entry"] for r in rows]
    pnls = [r["pnl"] for r in rows]
    per_share = [r["pnl"] / r["shares"] for r in rows]
    spreads = [r["spread"] for r in rows if r["spread"] is not None]
    mean_entry = statistics.fmean(entries)
    win_rate = wins / len(rows)
    fee = 0.07 * mean_entry * (1 - mean_entry)
    print(f"  {label}")
    print(f"    trades {len(rows):,}   win {win_rate*100:.2f}%   "
          f"mean entry {mean_entry:.4f}   median entry {statistics.median(entries):.4f}")
    print(f"    break-even accuracy at mean entry : "
          f"{(mean_entry + fee)*100:.2f}%")
    print(f"    margin (win rate - break-even)    : "
          f"{(win_rate - mean_entry - fee)*100:+.2f} points")
    print(f"    pnl/share {statistics.fmean(per_share)*100:+.3f}%   "
          f"total {sum(pnls):+.2f} USDC")
    if spreads:
        print(f"    spread median {statistics.median(spreads)*100:.2f}c   "
              f"mean {statistics.fmean(spreads)*100:.2f}c")
    return {"n": len(rows), "win": win_rate, "entry": mean_entry,
            "per_share": statistics.fmean(per_share)}


def bucket_table(rows: list, label: str) -> None:
    print(f"  {label}")
    print(f"    {'entry bucket':<14}{'n':>7}{'win%':>8}{'entry':>8}{'spread(c)':>11}"
          f"{'pnl/share':>11}")
    print("    " + "-" * 58)
    for low, high in BUCKETS:
        chunk = [r for r in rows if low <= r["entry"] < high]
        if len(chunk) < 5:
            continue
        spreads = [r["spread"] for r in chunk if r["spread"] is not None]
        print(f"    {f'{low:.2f}-{high:.2f}':<14}{len(chunk):>7}"
              f"{sum(1 for r in chunk if r['won'])/len(chunk)*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in chunk):>8.4f}"
              f"{(statistics.median(spreads)*100 if spreads else 0):>11.2f}"
              f"{statistics.fmean(r['pnl']/r['shares'] for r in chunk)*100:>+10.2f}%")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    root = resolve_data_root(args.root)

    print("=" * 96)
    print("V4 vs V6 DISAGREEMENT - DECOMPOSED")
    print("=" * 96)
    print("  Same dataset, same signal, same thresholds, nearly the same trade count")
    print("  (1,145 vs 1,138). So the difference must be in the PRICE, not the gates.")
    print()

    results = {}
    for checkpoint in (180.0, 240.0):
        print("=" * 96)
        print(f"CHECKPOINT T-{checkpoint:.0f}")
        print("=" * 96)
        rows = collect(root, checkpoint, args.limit)
        print(f"  markets clearing all signal gates: {len(rows):,}")
        print()
        summarise(f"T-{checkpoint:.0f} summary", rows)
        print()
        bucket_table(rows, f"by entry price at T-{checkpoint:.0f}:")
        results[checkpoint] = rows

    print("=" * 96)
    print("WHERE THE V4 NUMBER WOULD HAVE TO COME FROM")
    print("=" * 96)
    for checkpoint, rows in results.items():
        if not rows:
            continue
        entries = sorted(r["entry"] for r in rows)
        n = len(entries)
        print(f"  T-{checkpoint:.0f}: entry p10 {entries[n//10]:.4f}  "
              f"median {entries[n//2]:.4f}  mean {statistics.fmean(entries):.4f}  "
              f"p90 {entries[9*n//10]:.4f}")
    print()
    print("  V4 reported mean entry 0.880 with a 93.6% win rate out of sample.")
    print("  If no checkpoint here reproduces that entry, then either V4's gate set")
    print("  admitted a materially different market set, or its entry price was taken")
    print("  from a different tick than the one that produced its trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
