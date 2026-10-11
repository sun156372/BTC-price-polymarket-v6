"""Does the Up-token price lag the spot move, and is that lag tradeable?

    python tools_v6/lag_signal.py

THE HYPOTHESIS
--------------
The strategy currently enters at a FIXED time (T-240) and buys the direction the
Up-token has already moved. The bucket analysis suggested a better idea: the losers
are the CHEAP entries (below 0.82), where a 78.9% win rate meets a 0.774 entry and the
move was small (mean |move| 0.279). Those are not bargains, they are weak signals.

That points at a specific mechanism. The underlying (BTC spot) and the Up-token price
carry the SAME information. If the spot has moved but the token price has not caught up
yet, the token is genuinely cheap relative to what is already known - that is a real
mispricing, not a weak signal. This script measures whether that lag exists and whether
it predicts returns.

THE MEASUREMENT
---------------
For each market:
  spot_move  = spot(T-240) / spot(open) - 1        the information
  token_move = up_mid(T-240) - up_mid(open)        the price's response

Elasticity is fitted as beta in  token_move = beta * spot_move  (through the origin),
because a binary that pays 1 if BTC is up should respond roughly linearly to the
underlying over a small range.

Then per market:
  expected_move = beta * spot_move                 what the price SHOULD have done
  lag           = expected_move - token_move       positive = price has not caught up

If the hypothesis holds, high-lag windows should show a higher win rate and a better
return per share than low-lag windows, and the gate should be on |lag| rather than on
|move|.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from engine_v6 import dataset  # noqa: E402
from engine_v6.imbalance import orient, token_imbalance  # noqa: E402
from engine_v6.signals import net_pnl  # noqa: E402
from paths import resolve_data_root  # noqa: E402

BUCKETS = 8


def collect(root: str, checkpoint: float, min_move: float, min_imb: float):
    """Every market that clears the existing gates, with its lag measured."""
    imbalance = token_imbalance(root, checkpoint_s=checkpoint)
    spot = dataset.spot_index(root)
    rows = []
    for market in dataset.iter_markets(root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        open_tick = market.first_tick()
        entry_tick = market.tick_at_t_minus(checkpoint, tolerance_s=1.5)
        if open_tick is None or entry_tick is None:
            continue
        if open_tick.up_mid is None or entry_tick.up_mid is None:
            continue
        index = spot.get(market.day)
        if not index:
            continue
        spot_open = dataset.spot_at(index, market.start_ts)
        spot_now = dataset.spot_at(index, entry_tick.ts)
        if not spot_open or not spot_now:
            continue
        spot_move = spot_now / spot_open - 1.0
        token_move = entry_tick.up_mid - open_tick.up_mid

        move = token_move
        if abs(move) < min_move:
            continue
        side = "UP" if move > 0 else "DOWN"
        raw = imbalance.get(market.market_id)
        if raw is None:
            continue
        signed = orient(raw, side)
        if signed is None or abs(signed) < min_imb:
            continue
        if abs(spot_move) < 0.00045:
            continue
        ask = entry_tick.ask(side)
        if ask is None or not 0.0 < ask < 1.0:
            continue
        shares = 10.0 / ask
        won = (side == "UP") == market.up_won
        rows.append({
            "slug": market.slug, "day": market.day, "side": side,
            "sign": 1.0 if side == "UP" else -1.0,
            "spot_move": spot_move, "token_move": token_move,
            "entry": ask, "won": won, "shares": shares,
            "pnl": net_pnl(ask, won, shares),
            #: the spot move expressed in the direction we are buying
            "aligned_spot": spot_move * (1.0 if side == "UP" else -1.0),
            #: the token move in the direction we are buying (always positive here)
            "aligned_token": abs(token_move),
        })
    return rows


def fit_elasticity(rows: list) -> float:
    """beta in aligned_token = beta * aligned_spot, through the origin."""
    num = sum(r["aligned_spot"] * r["aligned_token"] for r in rows)
    den = sum(r["aligned_spot"] ** 2 for r in rows)
    return num / den if den > 0 else 0.0


def describe(rows: list) -> dict:
    if not rows:
        return {}
    entry = statistics.fmean(r["entry"] for r in rows)
    win = sum(1 for r in rows if r["won"]) / len(rows)
    fee = 0.07 * entry * (1 - entry)
    per_share = statistics.fmean(r["pnl"] / r["shares"] for r in rows)
    return {"n": len(rows), "entry": entry, "win": win,
            "margin": win - entry - fee, "per_share": per_share,
            "total": sum(r["pnl"] for r in rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None)
    parser.add_argument("--checkpoint", type=float, default=240.0)
    parser.add_argument("--min-move", type=float, default=0.20)
    parser.add_argument("--min-imbalance", type=float, default=0.30)
    args = parser.parse_args(argv)

    root = resolve_data_root(args.root)
    print("=" * 96)
    print("PRICE LAG VS SPOT MOVE - is the token slow to reprice?")
    print("=" * 96)
    print(f"  checkpoint T-{args.checkpoint:.0f}   gates: |move| >= "
          f"{args.min_move:.2f}, |imbalance| >= {args.min_imbalance:.2f}")
    print()

    rows = collect(root, args.checkpoint, args.min_move, args.min_imbalance)
    print(f"  markets passing the existing gates: {len(rows):,}")
    if len(rows) < 50:
        print("  too few to analyse")
        return 1

    beta = fit_elasticity(rows)
    print()
    print("=" * 96)
    print("1. HOW STRONGLY DOES THE TOKEN RESPOND TO THE UNDERLYING?")
    print("=" * 96)
    print(f"  elasticity  beta = {beta:.3f}")
    print(f"    A 10bps spot move is accompanied by a {beta*1000:.4f} token move.")
    corr_num = sum(r["aligned_spot"] * r["aligned_token"] for r in rows)
    corr_a = sum(r["aligned_spot"] ** 2 for r in rows) ** 0.5
    corr_b = sum(r["aligned_token"] ** 2 for r in rows) ** 0.5
    correlation = corr_num / (corr_a * corr_b) if corr_a * corr_b else 0.0
    print(f"  correlation(spot move, token move) = {correlation:.3f}")
    print()
    if correlation < 0.2:
        print("  The two are only weakly related. If the token barely tracks the")
        print("  underlying over this horizon, a 'lag' relative to spot is not a")
        print("  mispricing - it is simply that spot is not what the token is pricing.")
        print("  That would kill the hypothesis, so read the next section carefully.")

    print()
    print("=" * 96)
    print("2. DOES THE LAG PREDICT ANYTHING?")
    print("=" * 96)
    for r in rows:
        r["expected"] = beta * r["aligned_spot"]
        r["lag"] = r["expected"] - r["aligned_token"]
    ordered = sorted(rows, key=lambda r: r["lag"])
    size = len(ordered) // BUCKETS
    print(f"  {'lag bucket':<13}{'n':>6}{'lag':>9}{'entry':>8}{'win%':>7}"
          f"{'margin':>8}{'pnl/share':>11}{'total':>9}")
    print("  " + "-" * 71)
    for i in range(BUCKETS):
        lo = i * size
        hi = len(ordered) if i == BUCKETS - 1 else (i + 1) * size
        chunk = ordered[lo:hi]
        if len(chunk) < 5:
            continue
        d = describe(chunk)
        print(f"  {f'bucket {i+1}':<13}{d['n']:>6}"
              f"{statistics.fmean(r['lag'] for r in chunk):>+9.4f}"
              f"{d['entry']:>8.4f}{d['win']*100:>7.1f}"
              f"{d['margin']*100:>+8.2f}{d['per_share']*100:>+10.2f}%"
              f"{d['total']:>+9.1f}")

    print()
    print("=" * 96)
    print("3. THE DECISIVE TEST: gate on lag instead of on move size")
    print("=" * 96)
    print("  If lag is the real signal, requiring a MINIMUM lag should improve the")
    print("  per-share return without cutting the trade count as hard as raising the")
    print("  move threshold does.")
    print()
    print(f"  {'config':<40}{'n':>6}{'entry':>8}{'win%':>7}{'margin':>8}"
          f"{'pnl/share':>11}")
    print("  " + "-" * 78)
    base = describe(rows)
    print(f"  {'baseline (current gates)':<40}{base['n']:>6}{base['entry']:>8.4f}"
          f"{base['win']*100:>7.1f}{base['margin']*100:>+8.2f}"
          f"{base['per_share']*100:>+10.2f}%")
    for lag_min in (0.02, 0.04, 0.06, 0.08):
        chunk = [r for r in rows if r["lag"] >= lag_min]
        if len(chunk) < 20:
            continue
        d = describe(chunk)
        print(f"  {f'+ lag >= {lag_min:.2f}':<40}{d['n']:>6}{d['entry']:>8.4f}"
              f"{d['win']*100:>7.1f}{d['margin']*100:>+8.2f}"
              f"{d['per_share']*100:>+10.2f}%")

    print()
    print("=" * 96)
    print("4. THE ALTERNATIVE READING: is 'lag' just a proxy for a big spot move?")
    print("=" * 96)
    print("  A large spot move with a modest token move produces a large lag. So lag")
    print("  could be a repackaging of 'the underlying moved a lot', which is already")
    print("  known to help. Controlling for spot move size separates the two.")
    print()
    print(f"  {'spot move band':<18}{'lag':>9}{'n':>6}{'win%':>7}{'entry':>8}"
          f"{'pnl/share':>11}")
    print("  " + "-" * 59)
    bands = [(0.00045, 0.0008), (0.0008, 0.0015), (0.0015, 0.003), (0.003, 1.0)]
    for lo, hi in bands:
        inside = [r for r in rows if lo <= abs(r["aligned_spot"]) < hi]
        if len(inside) < 30:
            continue
        med = statistics.median(r["lag"] for r in inside)
        low = [r for r in inside if r["lag"] <= med]
        high = [r for r in inside if r["lag"] > med]
        for label, chunk in ((f"{lo*1e4:.0f}-{hi*1e4:.0f}bps low lag", low),
                             (f"{lo*1e4:.0f}-{hi*1e4:.0f}bps high lag", high)):
            if len(chunk) < 15:
                continue
            d = describe(chunk)
            print(f"  {label:<18}{statistics.fmean(r['lag'] for r in chunk):>+9.4f}"
                  f"{d['n']:>6}{d['win']*100:>7.1f}{d['entry']:>8.4f}"
                  f"{d['per_share']*100:>+10.2f}%")
    print()
    print("  If the high-lag half beats the low-lag half INSIDE each band, lag carries")
    print("  information beyond move size. If the two halves are the same, lag is just")
    print("  a proxy and adds nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
