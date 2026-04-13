"""Full backtesting engine for multi-factor portfolio strategies.

Supports:
- Walk-forward analysis with expanding/rolling windows
- Out-of-sample testing
- Monte Carlo simulation
- Parameter sensitivity analysis
- Realistic transaction cost and slippage modelling via Almgren-Chriss
  non-linear market impact (:class:`EnhancedExecutionSimulator`)
- Bar-level stop-loss checking (every bar, not just at rebalance)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.market_impact import (
    EnhancedExecutionSimulator,
    MarketImpactConfig,
    compute_adv,
)
from quantbot.research.metrics import PerformanceMetrics, compute_metrics


_ZERO = Decimal("0")


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run."""

    initial_equity: float = 100_000.0
    rebalance_every_n_bars: int = 1
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    partial_fill_ratio: float = 1.0
    periods_per_year: float = 365.0  # daily bars default
    use_market_impact: bool = True
    market_impact_config: MarketImpactConfig | None = None


@dataclass
class BacktestResult:
    """Result of a single backtest run."""

    equity_curve: list[float]
    trade_returns: list[float]
    timestamps: list[datetime]
    weights_history: list[dict[str, float]]
    metrics: PerformanceMetrics


StrategyAllocator = Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]
"""Callable: (bars_by_instrument, funding_by_instrument) -> target_weights"""


class BacktestEngine:
    """Run time-series backtests on portfolio allocation strategies.

    By default the engine uses :class:`EnhancedExecutionSimulator` for
    volume-dependent, non-linear slippage (Almgren-Chriss).  Set
    ``config.use_market_impact = False`` to fall back to the legacy
    fixed-BPS slippage model.

    Stop-losses stored in the allocator's ``_stop_levels`` attribute
    (if present) are checked on **every bar**, not just at rebalance
    intervals.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        if config.use_market_impact:
            mi_config = config.market_impact_config or MarketImpactConfig(
                taker_fee_rate=Decimal(str(config.taker_fee_rate)),
            )
            self._enhanced_sim = EnhancedExecutionSimulator(mi_config)
            self._legacy_sim = None
        else:
            from quantbot.research.simulator import (
                ExecutionAssumptions,
                ReplayExecutionSimulator,
            )
            self._enhanced_sim = None
            self._legacy_sim = ReplayExecutionSimulator(
                ExecutionAssumptions(
                    taker_fee_rate=Decimal(str(config.taker_fee_rate)),
                    slippage_bps=Decimal(str(config.slippage_bps)),
                    partial_fill_ratio=Decimal(str(config.partial_fill_ratio)),
                )
            )

    def run(
        self,
        allocator: StrategyAllocator,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
        min_history: int = 5,
    ) -> BacktestResult:
        """Run a backtest over the provided data.

        Args:
            allocator: Function mapping bars/funding to target weights.
            bars_by_instrument: Historical bars per instrument keyed by inst_id.
            funding_by_instrument: Historical funding per instrument.
            min_history: Minimum bars required before first allocation.

        Returns:
            BacktestResult with equity curve and metrics.
        """
        # Determine common time range
        all_timestamps = _common_timestamps(bars_by_instrument)
        if len(all_timestamps) < min_history + 1:
            raise ValueError("Not enough common bars for backtest")

        equity = self._config.initial_equity
        equity_curve = [equity]
        trade_returns: list[float] = []
        timestamps: list[datetime] = []
        weights_history: list[dict[str, float]] = []
        current_weights: dict[str, Decimal] = {}
        bar_count = 0
        last_rebalance_equity = equity  # Track equity at last rebalance

        # Extract the underlying strategy object (if any) to access
        # stop-levels for bar-level stop checking.
        _strategy_obj = _extract_strategy_object(allocator)

        for t_idx in range(min_history, len(all_timestamps)):
            bar_count += 1
            ts = all_timestamps[t_idx]

            # Slice history up to current bar (inclusive)
            sliced_bars = {
                inst_id: [b for b in bars if b.ts <= ts]
                for inst_id, bars in bars_by_instrument.items()
            }
            sliced_funding = {
                inst_id: [f for f in fr if f.funding_time <= ts]
                for inst_id, fr in funding_by_instrument.items()
            }

            # Get current prices
            prices: dict[str, Decimal] = {}
            for inst_id, bars in sliced_bars.items():
                if bars:
                    prices[inst_id] = bars[-1].close

            # Mark-to-market: update equity based on price changes
            if current_weights and t_idx > min_history:
                prev_ts = all_timestamps[t_idx - 1]
                pnl = Decimal("0")
                for inst_id, weight in current_weights.items():
                    if inst_id in prices:
                        prev_bars = [b for b in bars_by_instrument[inst_id] if b.ts <= prev_ts]
                        if prev_bars:
                            prev_price = prev_bars[-1].close
                            curr_price = prices[inst_id]
                            if prev_price > 0:
                                ret = (curr_price - prev_price) / prev_price
                                pnl += Decimal(str(equity)) * weight * ret
                equity = max(equity + float(pnl), 0.0)

            # ---- Bar-level stop-loss check (Issue 5) --------------------
            # Check stop levels every bar, not just at rebalance.  If any
            # held instrument breaches its stop, zero its weight immediately.
            if _strategy_obj is not None and current_weights:
                stop_levels = getattr(_strategy_obj, "_stop_levels", None)
                if stop_levels is not None:
                    stopped_ids: list[str] = []
                    for inst_id, weight in current_weights.items():
                        if weight > _ZERO and inst_id in stop_levels:
                            price_f = float(prices.get(inst_id, Decimal("0")))
                            if price_f > 0 and price_f < stop_levels[inst_id]:
                                stopped_ids.append(inst_id)
                    for inst_id in stopped_ids:
                        del stop_levels[inst_id]
                        current_weights[inst_id] = _ZERO

            # Rebalance at interval
            if bar_count % self._config.rebalance_every_n_bars == 0:
                # Record rebalance-to-rebalance return (trade-level)
                if last_rebalance_equity > 0 and current_weights:
                    rebal_ret = equity / last_rebalance_equity - 1.0
                    trade_returns.append(rebal_ret)

                try:
                    target_weights = allocator(sliced_bars, sliced_funding)
                except (ValueError, ZeroDivisionError):
                    target_weights = {}

                if target_weights and equity > 0:
                    total_costs = self._execute_rebalance(
                        equity=equity,
                        current_weights=current_weights,
                        target_weights=target_weights,
                        prices=prices,
                        sliced_bars=sliced_bars,
                    )
                    # Cap transaction costs at equity — in live trading the
                    # exchange would reject an order that cannot be funded;
                    # here we skip the rebalance when costs would be ruinous
                    # (> 50 % of equity).
                    if total_costs > equity * 0.5:
                        # Reject the rebalance: costs too high (illiquid)
                        pass
                    else:
                        equity = max(equity - total_costs, 0.0)
                        current_weights = target_weights

                last_rebalance_equity = equity

            equity_curve.append(max(equity, 0.0))
            timestamps.append(ts)
            weights_history.append({k: float(v) for k, v in current_weights.items()})

        # Compute period returns for Sharpe/equity metrics
        period_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                r = equity_curve[i] / equity_curve[i - 1] - 1.0
                period_returns.append(r)

        metrics = compute_metrics(
            equity_curve=equity_curve,
            trade_returns=trade_returns if trade_returns else period_returns,
            periods_per_year=self._config.periods_per_year,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trade_returns=period_returns,
            timestamps=timestamps,
            weights_history=weights_history,
            metrics=metrics,
        )

    # -- internal helpers --------------------------------------------------

    def _execute_rebalance(
        self,
        *,
        equity: float,
        current_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        prices: dict[str, Decimal],
        sliced_bars: dict[str, list[OhlcvBar]],
    ) -> float:
        """Execute a portfolio rebalance and return total transaction costs.

        When ``use_market_impact`` is enabled, includes both slippage and
        fees.  In legacy mode, returns fees only.
        """
        if self._enhanced_sim is not None:
            # Compute volumes and ADV from bar history
            volumes: dict[str, Decimal] = {}
            adv_map: dict[str, Decimal] = {}
            for inst_id in target_weights:
                bars_list = sliced_bars.get(inst_id, [])
                if bars_list:
                    volumes[inst_id] = bars_list[-1].volume
                    adv_map[inst_id] = compute_adv(bars_list)
                else:
                    volumes[inst_id] = _ZERO
                    adv_map[inst_id] = _ZERO

            fills = self._enhanced_sim.rebalance(
                equity=Decimal(str(equity)),
                current_weights=current_weights,
                target_weights=target_weights,
                prices=prices,
                volumes=volumes,
                adv=adv_map,
            )
            return sum(float(f.total_cost) for f in fills)
        else:
            assert self._legacy_sim is not None
            fills = self._legacy_sim.rebalance(
                equity=Decimal(str(equity)),
                current_weights=current_weights,
                target_weights=target_weights,
                prices=prices,
            )
            return sum(float(f.fee) for f in fills)


def _extract_strategy_object(allocator: StrategyAllocator) -> object | None:
    """Try to extract the strategy object from an allocator closure.

    Allocator factories (e.g. ``create_trend_following_allocator``) wrap
    a strategy instance inside a closure.  We inspect ``__closure__`` to
    find an object that owns a ``_stop_levels`` dict so the engine can
    check stops every bar.
    """
    closure = getattr(allocator, "__closure__", None)
    if closure:
        for cell in closure:
            try:
                obj = cell.cell_contents
                if hasattr(obj, "_stop_levels"):
                    return obj
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardWindow:
    """One window in walk-forward analysis."""

    in_sample_start: int
    in_sample_end: int
    out_of_sample_start: int
    out_of_sample_end: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""

    windows: list[WalkForwardWindow]
    in_sample_metrics: list[PerformanceMetrics]
    oos_metrics: list[PerformanceMetrics]
    combined_oos_equity: list[float]
    combined_oos_metrics: PerformanceMetrics


def walk_forward_analysis(
    allocator: StrategyAllocator,
    bars_by_instrument: dict[str, list[OhlcvBar]],
    funding_by_instrument: dict[str, list[FundingRate]],
    config: BacktestConfig,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    min_history: int = 5,
) -> WalkForwardResult:
    """Run walk-forward analysis with expanding windows.

    Splits the data into n_splits windows, trains on train_ratio of each
    window, and tests on the remaining portion.
    """
    all_timestamps = _common_timestamps(bars_by_instrument)
    n_bars = len(all_timestamps)
    window_size = n_bars // n_splits

    if window_size < min_history + 2:
        raise ValueError("Not enough data for walk-forward splits")

    windows: list[WalkForwardWindow] = []
    is_metrics: list[PerformanceMetrics] = []
    oos_metrics: list[PerformanceMetrics] = []
    combined_oos_equity: list[float] = [config.initial_equity]

    engine = BacktestEngine(config)

    for i in range(n_splits):
        start = i * window_size
        end = min(start + window_size, n_bars)
        split_point = start + int((end - start) * train_ratio)

        if split_point - start < min_history + 1 or end - split_point < 2:
            continue

        wf = WalkForwardWindow(
            in_sample_start=start,
            in_sample_end=split_point,
            out_of_sample_start=split_point,
            out_of_sample_end=end,
        )
        windows.append(wf)

        # In-sample
        is_bars = _slice_bars(bars_by_instrument, all_timestamps, start, split_point)
        is_funding = _slice_funding(funding_by_instrument, all_timestamps, start, split_point)
        is_result = engine.run(allocator, is_bars, is_funding, min_history)
        is_metrics.append(is_result.metrics)

        # Out-of-sample
        oos_bars = _slice_bars(bars_by_instrument, all_timestamps, split_point, end)
        oos_funding = _slice_funding(funding_by_instrument, all_timestamps, split_point, end)
        try:
            oos_result = engine.run(allocator, oos_bars, oos_funding, min(min_history, split_point))
            oos_metrics.append(oos_result.metrics)
            # Chain equity curves
            if oos_result.equity_curve[0] > 0:
                scale = combined_oos_equity[-1] / oos_result.equity_curve[0]
            else:
                scale = 1.0
            combined_oos_equity.extend(eq * scale for eq in oos_result.equity_curve[1:])
        except ValueError:
            # Not enough data in OOS window
            continue

    if not oos_metrics:
        raise ValueError("Walk-forward analysis produced no OOS windows")

    combined_returns = [
        combined_oos_equity[i] / combined_oos_equity[i - 1] - 1.0
        for i in range(1, len(combined_oos_equity))
    ]
    combined_metrics = compute_metrics(
        equity_curve=combined_oos_equity,
        trade_returns=combined_returns,
        periods_per_year=config.periods_per_year,
    )

    return WalkForwardResult(
        windows=windows,
        in_sample_metrics=is_metrics,
        oos_metrics=oos_metrics,
        combined_oos_equity=combined_oos_equity,
        combined_oos_metrics=combined_metrics,
    )


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    simulations: int
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive_return: float
    median_total_return: float


def monte_carlo_simulation(
    trade_returns: list[float],
    n_simulations: int = 1000,
    initial_equity: float = 100_000.0,
    periods_per_year: float = 365.0,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation via *sign-randomization test*.

    Instead of merely shuffling the trade-return *sequence* (which
    preserves positive expectancy and almost always produces positive
    results), this implementation uses a randomization test that
    randomly flips the sign of each trade return with 50 % probability.
    This destroys the signal-return relationship while preserving the
    magnitude distribution, providing a proper null-hypothesis test
    for whether the strategy has genuine alpha.

    Note: this is a randomization test (independent random sign flips),
    not an exhaustive permutation test over all 2^N sign combinations.
    With ``n_simulations`` >= 500 the sampling is sufficient for
    reliable p-value estimation.

    The returned ``p5_sharpe`` represents the 5th-percentile Sharpe
    under the null of *no predictive signal*.  If the strategy's
    actual Sharpe is above this value, the strategy is unlikely to be
    a fluke.
    """
    if not trade_returns:
        raise ValueError("No trade returns for Monte Carlo")

    rng = random.Random(seed)
    sharpes: list[float] = []
    max_dds: list[float] = []
    total_returns: list[float] = []

    for _ in range(n_simulations):
        # Permutation test: randomly flip signs to destroy signal-return
        # mapping while preserving the distribution of magnitudes.
        permuted = [r * (1 if rng.random() < 0.5 else -1) for r in trade_returns]

        equity_curve = [initial_equity]
        eq = initial_equity
        for r in permuted:
            eq *= (1.0 + r)
            equity_curve.append(max(eq, 0.0))

        metrics = compute_metrics(
            equity_curve=equity_curve,
            trade_returns=permuted,
            periods_per_year=periods_per_year,
        )
        sharpes.append(metrics.sharpe_ratio)
        max_dds.append(metrics.max_drawdown)
        total_returns.append(metrics.total_return)

    sharpes.sort()
    max_dds.sort()
    total_returns.sort()
    n = len(sharpes)

    return MonteCarloResult(
        simulations=n_simulations,
        median_sharpe=sharpes[n // 2],
        p5_sharpe=sharpes[int(n * 0.05)],
        p95_sharpe=sharpes[int(n * 0.95)],
        median_max_dd=max_dds[n // 2],
        p95_max_dd=max_dds[int(n * 0.95)],
        prob_positive_return=sum(1 for r in total_returns if r > 0) / n,
        median_total_return=total_returns[n // 2],
    )


# ---------------------------------------------------------------------------
# Parameter Sensitivity Analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityPoint:
    """One point in parameter sensitivity analysis."""

    param_name: str
    param_value: float
    sharpe: float
    max_dd: float
    profit_factor: float
    total_return: float


def parameter_sensitivity(
    allocator_factory: Callable[..., StrategyAllocator],
    param_name: str,
    param_values: list[float],
    bars_by_instrument: dict[str, list[OhlcvBar]],
    funding_by_instrument: dict[str, list[FundingRate]],
    config: BacktestConfig,
    min_history: int = 5,
    **fixed_params: float,
) -> list[SensitivityPoint]:
    """Test strategy across a range of parameter values."""
    engine = BacktestEngine(config)
    results: list[SensitivityPoint] = []

    for val in param_values:
        kwargs = {**fixed_params, param_name: val}
        allocator = allocator_factory(**kwargs)
        try:
            bt = engine.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
            results.append(SensitivityPoint(
                param_name=param_name,
                param_value=val,
                sharpe=bt.metrics.sharpe_ratio,
                max_dd=bt.metrics.max_drawdown,
                profit_factor=bt.metrics.profit_factor,
                total_return=bt.metrics.total_return,
            ))
        except (ValueError, ZeroDivisionError):
            continue

    return results


# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressTestResult:
    """Result of stress testing under different market conditions."""

    scenario_name: str
    metrics: PerformanceMetrics


def stress_test(
    allocator: StrategyAllocator,
    bars_by_instrument: dict[str, list[OhlcvBar]],
    funding_by_instrument: dict[str, list[FundingRate]],
    config: BacktestConfig,
    min_history: int = 5,
) -> list[StressTestResult]:
    """Run strategy under normal and stressed conditions.

    Tests with:
    - Normal conditions (base config)
    - High fees (3x)
    - High slippage (5x)
    - Low fill rate (50%)
    - Combined stress (all above)
    """
    engine = BacktestEngine(config)
    results: list[StressTestResult] = []

    # Normal
    bt = engine.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
    results.append(StressTestResult("normal", bt.metrics))

    # High fees
    high_fee_config = BacktestConfig(
        initial_equity=config.initial_equity,
        rebalance_every_n_bars=config.rebalance_every_n_bars,
        taker_fee_rate=config.taker_fee_rate * 3,
        slippage_bps=config.slippage_bps,
        partial_fill_ratio=config.partial_fill_ratio,
        periods_per_year=config.periods_per_year,
    )
    eng = BacktestEngine(high_fee_config)
    bt = eng.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
    results.append(StressTestResult("high_fees_3x", bt.metrics))

    # High slippage
    high_slip_config = BacktestConfig(
        initial_equity=config.initial_equity,
        rebalance_every_n_bars=config.rebalance_every_n_bars,
        taker_fee_rate=config.taker_fee_rate,
        slippage_bps=config.slippage_bps * 5,
        partial_fill_ratio=config.partial_fill_ratio,
        periods_per_year=config.periods_per_year,
    )
    eng = BacktestEngine(high_slip_config)
    bt = eng.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
    results.append(StressTestResult("high_slippage_5x", bt.metrics))

    # Low fill rate
    low_fill_config = BacktestConfig(
        initial_equity=config.initial_equity,
        rebalance_every_n_bars=config.rebalance_every_n_bars,
        taker_fee_rate=config.taker_fee_rate,
        slippage_bps=config.slippage_bps,
        partial_fill_ratio=0.5,
        periods_per_year=config.periods_per_year,
    )
    eng = BacktestEngine(low_fill_config)
    bt = eng.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
    results.append(StressTestResult("low_fill_50pct", bt.metrics))

    # Combined stress
    combined_config = BacktestConfig(
        initial_equity=config.initial_equity,
        rebalance_every_n_bars=config.rebalance_every_n_bars,
        taker_fee_rate=config.taker_fee_rate * 3,
        slippage_bps=config.slippage_bps * 5,
        partial_fill_ratio=0.5,
        periods_per_year=config.periods_per_year,
    )
    eng = BacktestEngine(combined_config)
    bt = eng.run(allocator, bars_by_instrument, funding_by_instrument, min_history)
    results.append(StressTestResult("combined_stress", bt.metrics))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _common_timestamps(
    bars_by_instrument: dict[str, list[OhlcvBar]],
) -> list[datetime]:
    """Get sorted union of all timestamps across instruments."""
    all_ts: set[datetime] = set()
    for bars in bars_by_instrument.values():
        for b in bars:
            all_ts.add(b.ts)
    return sorted(all_ts)


def _slice_bars(
    bars_by_instrument: dict[str, list[OhlcvBar]],
    all_timestamps: list[datetime],
    start_idx: int,
    end_idx: int,
) -> dict[str, list[OhlcvBar]]:
    """Slice bars to a time range."""
    start_ts = all_timestamps[start_idx]
    end_ts = all_timestamps[min(end_idx - 1, len(all_timestamps) - 1)]
    return {
        inst_id: [b for b in bars if start_ts <= b.ts <= end_ts]
        for inst_id, bars in bars_by_instrument.items()
    }


def _slice_funding(
    funding_by_instrument: dict[str, list[FundingRate]],
    all_timestamps: list[datetime],
    start_idx: int,
    end_idx: int,
) -> dict[str, list[FundingRate]]:
    """Slice funding to a time range."""
    start_ts = all_timestamps[start_idx]
    end_ts = all_timestamps[min(end_idx - 1, len(all_timestamps) - 1)]
    return {
        inst_id: [f for f in fr if start_ts <= f.funding_time <= end_ts]
        for inst_id, fr in funding_by_instrument.items()
    }
