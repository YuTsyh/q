"""Performance metrics for strategy evaluation.

Computes Sharpe, Sortino, CAGR, Max Drawdown, Calmar Ratio,
Profit Factor, Win Rate, and Expectancy from a series of returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PerformanceMetrics:
    """Aggregate performance report for a backtest."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    expectancy: float
    total_trades: int
    avg_trade_return: float
    max_consecutive_losses: int
    annualised_volatility: float


def compute_metrics(
    equity_curve: list[float],
    trade_returns: list[float],
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: List of portfolio equity values (starting equity first).
        trade_returns: List of per-trade percentage returns.
        periods_per_year: Number of rebalance periods per year.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.

    Returns:
        PerformanceMetrics dataclass.
    """
    if len(equity_curve) < 2:
        raise ValueError("equity_curve must have at least 2 values")

    # --- Period returns ---
    period_returns = [
        (equity_curve[i] / equity_curve[i - 1]) - 1.0
        for i in range(1, len(equity_curve))
    ]

    n_periods = len(period_returns)
    total_return = equity_curve[-1] / equity_curve[0] - 1.0

    # --- CAGR ---
    years = n_periods / periods_per_year
    if years > 0 and equity_curve[-1] > 0 and equity_curve[0] > 0:
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # --- Annualised volatility ---
    mean_return = sum(period_returns) / n_periods if n_periods > 0 else 0.0
    variance = sum((r - mean_return) ** 2 for r in period_returns) / max(n_periods - 1, 1)
    volatility = math.sqrt(variance)
    ann_volatility = volatility * math.sqrt(periods_per_year)

    # --- Sharpe Ratio ---
    period_rf = risk_free_rate / periods_per_year
    excess_returns = [r - period_rf for r in period_returns]
    excess_mean = sum(excess_returns) / len(excess_returns) if excess_returns else 0.0
    if volatility > 1e-12:
        sharpe = (excess_mean / volatility) * math.sqrt(periods_per_year)
    elif excess_mean > 0:
        sharpe = float("inf")
    else:
        sharpe = 0.0

    # --- Sortino Ratio ---
    downside_returns = [min(r - period_rf, 0.0) for r in period_returns]
    downside_var = sum(d ** 2 for d in downside_returns) / max(len(downside_returns), 1)
    downside_std = math.sqrt(downside_var)
    if downside_std > 1e-12:
        sortino = (excess_mean / downside_std) * math.sqrt(periods_per_year)
    elif excess_mean > 0:
        sortino = float("inf")
    else:
        sortino = 0.0

    # --- Max Drawdown ---
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # --- Calmar Ratio ---
    calmar = cagr / max_dd if max_dd > 1e-12 else (float("inf") if cagr > 0 else 0.0)

    # --- Trade-level metrics ---
    total_trades = len(trade_returns)
    if total_trades > 0:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / total_trades
        avg_trade = sum(trade_returns) / total_trades

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else float("inf")

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Max consecutive losses
        max_consec_loss = 0
        current_streak = 0
        for r in trade_returns:
            if r <= 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0
    else:
        win_rate = 0.0
        avg_trade = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        max_consec_loss = 0

    return PerformanceMetrics(
        total_return=round(total_return, 6),
        cagr=round(cagr, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 6),
        calmar_ratio=round(calmar, 4),
        profit_factor=round(profit_factor, 4),
        win_rate=round(win_rate, 4),
        expectancy=round(expectancy, 6),
        total_trades=total_trades,
        avg_trade_return=round(avg_trade, 6),
        max_consecutive_losses=max_consec_loss,
        annualised_volatility=round(ann_volatility, 6),
    )
