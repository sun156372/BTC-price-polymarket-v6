"""V6 offline backtest with complete performance reporting.

    python run_backtest_v6.py --horizons 5m
    python run_backtest_v6.py --horizons 5m --split 2026-04-27

WHAT WAS MISSING IN V4
----------------------
The V4 backtest printed a per-share edge and a day count. It did not print a win rate,
a profit factor, a drawdown or an equity curve, so there was no way to see how the
strategy actually behaved, and two aggregations it did print disagreed in sign on the
same data. Every number here comes from one list of settled trades via
``engine_v6.metrics``, so the conventions are explicit and the arithmetic is traceable.

It also reports an IN-SAMPLE / OUT-OF-SAMPLE split, because a single pooled number is
what let earlier rounds overstate the edge.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tools_v6"))

from engine_v6 import dataset  # noqa: E402
from engine_v6.imbalance import token_imbalance  # noqa: E402
from engine_v6.metrics import (  # noqa: E402
    TradeRecord, bucket_by_market_and_day, format_report, from_trades,
)
from engine_v6.signals import net_pnl, taker_fee_per_share  # noqa: E402
from engine_v6.strategy import StrategyV6, V6Params  # noqa: E402

def _import_paths():
    """Find tools_v6 regardless of whether v6_src has been flattened."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "tools_v6"),
                      os.path.join(os.path.dirname(here), "v6_src", "tools_v6"),
                      here):
        if os.path.exists(os.path.join(candidate, "paths.py")):
            sys.path.insert(0, candidate)
            return
    raise SystemExit("cannot locate tools_v6/paths.py next to this runner")


_import_paths()

from paths import resolve_data_root  # noqa: E402


def simulate_market(strategy: StrategyV6, market, imbalance_map: dict,
                    spot_index: dict, stake: float) -> TradeRecord | None:
    """Run one market through the strategy and return a settled trade, or None.

    Every measured quantity is PASSED TO ``strategy.evaluate`` rather than checked
    here. An earlier version computed the spot move locally and then re-tested the
    decision, which meant the strategy itself never saw the spot value and its spot
    gate silently did nothing - the backtest and the live loop were then running
    different rule sets while appearing to run the same one.
    """
    p = strategy.p
    open_tick = market.first_tick()
    entry_tick = market.tick_at_t_minus(p.entry_t_minus, tolerance_s=1.5)
    if open_tick is None or entry_tick is None:
        return None
    if open_tick.up_mid is None or entry_tick.up_mid is None:
        return None
    if market.up_won is None:
        return None

    # the side the signal points at, needed before the imbalance can be oriented
    move = entry_tick.up_mid - open_tick.up_mid
    side_probe = ""
    if abs(move) >= p.threshold:
        side_probe = "UP" if move > 0 else "DOWN"

    raw_imbalance = imbalance_map.get(market.market_id)
    signed = (strategy.orient(raw_imbalance, side_probe) if side_probe
              else raw_imbalance)

    spot_open = spot_now = None
    day_index = spot_index.get(market.day)
    if day_index and market.start_ts:
        spot_open = dataset.spot_at(day_index, market.start_ts)
        spot_now = dataset.spot_at(day_index, entry_tick.ts)

    ask = entry_tick.ask(side_probe) if side_probe else None
    if ask is None or not 0.0 < ask < 1.0:
        return None
    shares = stake / ask
    bid = entry_tick.bid(side_probe)

    decision = strategy.evaluate(
        slug=market.slug, now=entry_tick.ts,
        secs_to_close=entry_tick.to_deadline_ms / 1000.0,
        up_mid_open=open_tick.up_mid, up_mid_now=entry_tick.up_mid,
        book_imbalance=signed, spot_open=spot_open, spot_now=spot_now,
        ask=ask, spread=(ask - bid if bid else None),
        shares_wanted=shares, depth_cost=ask * shares,
    )
    if not decision.entering:
        return None
    entry = decision.limit_price
    won = (decision.side == "UP") == market.up_won
    pnl = net_pnl(entry, won, shares, p.fee_coefficient)
    return TradeRecord(slug=market.slug, day=market.day, side=decision.side,
                       entry=entry, shares=shares, won=won, pnl=pnl,
                       basis="decision")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None)
    parser.add_argument("--horizons", nargs="+", default=["5m"])
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--split", default="2026-04-27",
                        help="in-sample before this date, out-of-sample on/after")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--entry-t-minus", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--imbalance", type=float, default=None)
    parser.add_argument("--no-imbalance", action="store_true",
                        help="disable the core gate, to show what it contributes")
    parser.add_argument("--csv", default=None, help="write trades to this CSV")
    args = parser.parse_args(argv)

    root = resolve_data_root(args.root)
    params = V6Params()
    if args.entry_t_minus is not None:
        params = replace(params, entry_t_minus=args.entry_t_minus)
    if args.threshold is not None:
        params = replace(params, threshold=args.threshold)
    if args.imbalance is not None:
        params = replace(params, min_book_imbalance=args.imbalance)
    if args.no_imbalance:
        params = replace(params, min_book_imbalance=0.0)

    print("=" * 96)
    print("V6 BACKTEST")
    print("=" * 96)
    print(f"  data root  : {root}")
    print(f"  cache      : {'slim (fast)' if dataset.has_cache(root) else 'raw files'}")
    print(f"  horizons   : {' '.join(args.horizons)}")
    print(f"  stake      : {args.stake:.2f} USDC/trade")
    print(f"  split      : in-sample < {args.split} <= out-of-sample")
    print()
    strategy = StrategyV6(params)
    for line in strategy.describe():
        print(f"  {line}")
    print()

    print("  loading token imbalance at the decision tick...")
    imbalance_map = token_imbalance(root, checkpoint_s=params.entry_t_minus)
    print(f"    {len(imbalance_map):,} markets")
    print("  loading spot index...")
    spot = dataset.spot_index(root)
    print(f"    {len(spot)} days")

    trades: list[TradeRecord] = []
    seen = 0
    for market in dataset.iter_markets(root, horizons=tuple(args.horizons)):
        seen += 1
        if args.limit and seen > args.limit:
            break
        trade = simulate_market(strategy, market, imbalance_map, spot, args.stake)
        if trade is not None:
            trades.append(trade)
        if seen % 2000 == 0:
            print(f"    {seen:,} markets scanned, {len(trades):,} trades")

    print()
    print(f"  markets scanned: {seen:,}")
    if not trades:
        print("  no trades: the gates refused everything. Check the thresholds against")
        print("  the measured distributions in the session summary below.")
        _print_funnel(strategy)
        return 1

    ins = [t for t in trades if t.day < args.split]
    oos = [t for t in trades if t.day >= args.split]

    _print_funnel(strategy)
    print()
    if ins:
        print(format_report(from_trades(ins), "IN-SAMPLE"))
        print()
    if oos:
        print(format_report(from_trades(oos), "OUT-OF-SAMPLE"))
        print()
    print(format_report(from_trades(trades), "ALL DATA"))
    _print_daily(trades)
    if args.csv:
        _write_csv(args.csv, trades)
    return 0


def _print_funnel(strategy: StrategyV6) -> None:
    print("=" * 96)
    print("GATE FUNNEL")
    print("=" * 96)
    total = len(strategy.ledger.readings)
    if not total:
        print("  nothing evaluated")
        return
    for gate, reached, passed in strategy.ledger.funnel():
        share = (passed / reached * 100) if reached else 0.0
        print(f"  {reached:>7,}  {gate:<34} -> {passed:>7,} kept  ({share:5.1f}%)")
    print()
    print("  MEASURED DISTRIBUTIONS (live sessions compare against these):")
    for field_name, label in (("move", "|move|"), ("book_imbalance", "|imbalance|"),
                              ("spot_move", "|spot move|")):
        dist = strategy.ledger.distribution(field_name)
        if not dist:
            continue
        print(f"    {label:<12} p10 {dist['p10']:>9.4f}  p25 {dist['p25']:>9.4f}  "
              f"median {dist['median']:>9.4f}  p75 {dist['p75']:>9.4f}  "
              f"p90 {dist['p90']:>9.4f}")


def _print_daily(trades: list[TradeRecord]) -> None:
    print()
    print("=" * 96)
    print("PER-DAY BREAKDOWN")
    print("=" * 96)
    print(f"  {'day':<12}{'trades':>8}{'win%':>8}{'pnl':>11}{'cumulative':>13}"
          f"{'drawdown':>11}")
    print("-" * 96)
    by_day = bucket_by_market_and_day(trades)
    cumulative = 0.0
    peak = 0.0
    for day in sorted(by_day):
        rows = by_day[day]
        pnl = sum(t.pnl for t in rows)
        cumulative += pnl
        peak = max(peak, cumulative)
        wins = sum(1 for t in rows if t.won)
        print(f"  {day:<12}{len(rows):>8}{wins/len(rows)*100:>8.1f}{pnl:>+11.2f}"
              f"{cumulative:>+13.2f}{peak - cumulative:>11.2f}")
    print("-" * 96)
    total = sum(t.pnl for t in trades)
    print(f"  {'TOTAL':<12}{len(trades):>8}"
          f"{sum(1 for t in trades if t.won)/len(trades)*100:>8.1f}{total:>+11.2f}")


def _write_csv(path: str, trades: list[TradeRecord]) -> None:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slug", "day", "side", "entry", "shares", "won", "pnl",
                         "per_share"])
        for t in trades:
            writer.writerow([t.slug, t.day, t.side, f"{t.entry:.6f}",
                             f"{t.shares:.4f}", t.won, f"{t.pnl:.6f}",
                             f"{t.per_share:.6f}"])
    print(f"\n  trades written to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
