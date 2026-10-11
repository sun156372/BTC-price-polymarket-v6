"""Performance metrics: win rate, profit factor, drawdown, and honest caveats.

    from engine_v6.metrics import from_trades, EquityCurve

Issue this solves
-----------------
The V4 reports showed a per-share edge and a day count but no win rate, no profit
factor, no drawdown and no equity curve, so there was no way to see how a session
actually behaved. Worse, the two aggregations that were reported disagreed in sign on
the same data (pooled per-share was -0.54% while the day mean was -0.24%), because
they weight trades differently and nothing made that explicit.

Everything here is computed from a single list of settled trades, so every number is
traceable to the same records and the aggregation convention is stated on the label
rather than left implicit.

DEFINITIONS, so the numbers cannot be misread
---------------------------------------------
``win_rate``           fraction of settled trades with positive net P&L.
``profit_factor``      gross profit / gross loss. 1.0 is break-even. Undefined (None)
                       when there are no losses - reporting 0 or infinity would both
                       be lies.
``expectancy``         mean net P&L per trade, in USDC.
``edge_per_share``     mean net P&L per SHARE, which is the scale-free version and the
                       one comparable to the backtest's per-share figures.
``max_drawdown``       largest peak-to-trough fall of the cumulative P&L curve, in
                       USDC, plus the same as a fraction of the peak.
``sharpe``             mean/std of per-trade net P&L, NOT annualised. Annualising a
                       few hundred independent 5-minute bets would imply a precision
                       the sample does not have.
``sortino``            as Sharpe but only downside deviation.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    """One settled trade. The single input every metric is derived from."""

    slug: str
    day: str
    side: str
    entry: float
    shares: float
    won: bool
    #: net P&L in USDC after fees
    pnl: float
    #: which moment the entry price came from, e.g. "decision" or "fill_2s"
    basis: str = "decision"

    @property
    def per_share(self) -> float:
        return self.pnl / self.shares if self.shares else 0.0

    @property
    def stake(self) -> float:
        return self.entry * self.shares


@dataclass
class Metrics:
    """Everything needed to judge a run. None means "not computable", never 0."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    profit_factor: float | None = None
    expectancy: float | None = None
    edge_per_share: float | None = None
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    mean_entry: float | None = None
    total_staked: float = 0.0
    return_on_stake: float | None = None
    max_drawdown: float = 0.0
    max_drawdown_pct: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    best_trade: float | None = None
    worst_trade: float | None = None
    days: int = 0
    days_positive: int = 0
    trades_per_day: float | None = None
    equity: list[float] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def max_drawdown(curve: list[float]) -> tuple[float, float | None]:
    """Largest peak-to-trough decline of a cumulative curve.

    Returns ``(absolute, fraction_of_peak)``. The fraction is None when the peak was
    never above zero, because dividing by it would be meaningless.
    """
    peak = float("-inf")
    worst = 0.0
    worst_pct: float | None = None
    for value in curve:
        peak = max(peak, value)
        drop = peak - value
        if drop > worst:
            worst = drop
            worst_pct = (drop / peak) if peak > 0 else None
    return worst, worst_pct


def from_trades(trades: list[TradeRecord]) -> Metrics:
    """Compute every metric from one list of settled trades."""
    m = Metrics()
    if not trades:
        m.caveats.append("no settled trades: nothing can be concluded")
        return m

    pnls = [t.pnl for t in trades]
    shares = [t.shares for t in trades]
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]

    m.trades = len(trades)
    m.wins = len(wins)
    m.losses = len(losses)
    m.win_rate = len(wins) / len(trades)
    m.total_pnl = sum(pnls)
    m.gross_profit = sum(t.pnl for t in wins)
    m.gross_loss = abs(sum(t.pnl for t in losses))
    m.expectancy = statistics.fmean(pnls)
    m.edge_per_share = statistics.fmean(t.per_share for t in trades)
    m.mean_entry = statistics.fmean(t.entry for t in trades)
    m.total_staked = sum(t.stake for t in trades)
    m.return_on_stake = (m.total_pnl / m.total_staked) if m.total_staked else None
    m.best_trade = max(pnls)
    m.worst_trade = min(pnls)

    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    else:
        # no losing trades at all: a ratio is undefined, not infinite
        m.caveats.append(
            f"profit factor undefined: {len(wins)} trades, none of them a loss")

    # equity curve, trade by trade, starting from zero
    curve: list[float] = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        curve.append(running)
    m.equity = curve
    m.max_drawdown, m.max_drawdown_pct = max_drawdown([0.0] + curve)

    if len(pnls) > 1:
        sd = statistics.pstdev(pnls)
        if sd > 0:
            m.sharpe = statistics.fmean(pnls) / sd
            downside = [p for p in pnls if p < 0]
            if downside:
                dsd = math.sqrt(sum(p * p for p in downside) / len(pnls))
                m.sortino = statistics.fmean(pnls) / dsd if dsd else None

    days = sorted({t.day for t in trades})
    m.days = len(days)
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t.day] = by_day.get(t.day, 0.0) + t.pnl
    m.days_positive = sum(1 for v in by_day.values() if v > 0)
    m.trades_per_day = len(trades) / len(days) if days else None

    _add_caveats(m, trades)
    return m


def _add_caveats(m: Metrics, trades: list[TradeRecord]) -> None:
    """State the limits of the sample instead of leaving them for the reader."""
    if m.days and m.days < 20:
        m.caveats.append(
            f"only {m.days} distinct days: day-level statistics are unreliable")
    if m.trades < 100:
        m.caveats.append(
            f"only {m.trades} trades: the win rate has a 95% interval of about "
            f"+/-{1.96 * math.sqrt((m.win_rate or 0) * (1 - (m.win_rate or 0)) / max(1, m.trades)) * 100:.1f} points")
    if m.max_drawdown_pct is not None and m.total_pnl > 0:
        if m.max_drawdown_pct > 0.5:
            m.caveats.append(
                f"max drawdown is {m.max_drawdown_pct*100:.0f}% of the peak "
                f"cumulative P&L: the path is far rougher than the total suggests")
    bases = {t.basis for t in trades}
    if len(bases) > 1:
        m.caveats.append(f"mixed price bases in one metric set: {sorted(bases)}")


def format_report(m: Metrics, title: str = "PERFORMANCE", stake: float | None = None,
                  width: int = 78) -> str:
    """A readable block. Every None prints as 'n/a' rather than as a zero."""
    def num(value, fmt="{:.4f}", dash="n/a"):
        return dash if value is None else fmt.format(value)

    lines = []
    add = lines.append
    add("=" * width)
    add(title)
    add("=" * width)
    add(f"  trades                : {m.trades:,}"
        + (f"   over {m.days} days" if m.days else ""))
    if m.trades_per_day is not None:
        add(f"  trades per day        : {m.trades_per_day:.1f}")
    add(f"  win rate              : "
        + ("n/a" if m.win_rate is None else f"{m.win_rate*100:.2f}%")
        + f"   ({m.wins} wins / {m.losses} losses)")
    add(f"  profit factor         : {num(m.profit_factor, '{:.3f}')}")
    add(f"  expectancy per trade  : {num(m.expectancy, '{:+.4f}')} USDC")
    add(f"  edge per share        : "
        + ("n/a" if m.edge_per_share is None else f"{m.edge_per_share*100:+.3f}%"))
    add(f"  mean entry price      : {num(m.mean_entry, '{:.4f}')}")
    add(f"  total P&L             : {m.total_pnl:+,.2f} USDC")
    add(f"  total staked          : {m.total_staked:,.2f} USDC")
    add(f"  return on stake       : "
        + ("n/a" if m.return_on_stake is None else f"{m.return_on_stake*100:+.2f}%"))
    add(f"  gross profit / loss   : {m.gross_profit:+,.2f} / -{m.gross_loss:,.2f}")
    add(f"  best / worst trade    : {num(m.best_trade, '{:+.4f}')} / "
        f"{num(m.worst_trade, '{:+.4f}')}")
    add("  " + "-" * (width - 4))
    add(f"  MAX DRAWDOWN          : {m.max_drawdown:,.2f} USDC"
        + ("" if m.max_drawdown_pct is None
           else f"   ({m.max_drawdown_pct*100:.1f}% of peak)"))
    add(f"  sharpe (per trade)    : {num(m.sharpe, '{:+.4f}')}")
    add(f"  sortino (per trade)   : {num(m.sortino, '{:+.4f}')}")
    add(f"  positive days         : {m.days_positive}/{m.days}")
    if m.equity:
        spark = _sparkline(m.equity)
        add(f"  equity curve          : {spark}")
    if m.caveats:
        add("  " + "-" * (width - 4))
        add("  CAVEATS:")
        for caveat in m.caveats:
            add(f"    - {caveat}")
    add("=" * width)
    return "\n".join(lines)


_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 56) -> str:
    """A compact visual of the equity curve. Helps spot drawdowns at a glance."""
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(len(values) - 1, int(i * step))] for i in range(width)]
    else:
        sampled = values
    low, high = min(sampled), max(sampled)
    if high == low:
        return _SPARK[3] * len(sampled)
    span = high - low
    return "".join(_SPARK[min(7, int((v - low) / span * 7.999))] for v in sampled)


def bucket_by_market_and_day(trades: list[TradeRecord]):
    """Group trades for per-market and per-day breakdowns."""
    by_day: dict[str, list[TradeRecord]] = {}
    for t in trades:
        by_day.setdefault(t.day, []).append(t)
    return by_day
