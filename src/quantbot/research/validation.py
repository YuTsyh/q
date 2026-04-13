"""Full strategy validation runner.

Runs a complete validation pipeline:
1. In-sample backtest
2. Walk-forward analysis
3. Monte Carlo simulation
4. Stress testing
5. Checks acceptance criteria

Acceptance Criteria (all must pass):
- Sharpe Ratio ≥ 1.5
- Maximum Drawdown ≤ 25%
- Profit Factor ≥ 1.5
- Positive Expectancy
- OOS Sharpe ≥ 1.0
- Monte Carlo P5 Sharpe > 0
"""

from __future__ import annotations

from dataclasses import dataclass

from quantbot.research.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    MonteCarloResult,
    StressTestResult,
    WalkForwardResult,
    monte_carlo_simulation,
    stress_test,
    walk_forward_analysis,
)
from quantbot.research.data import FundingRate, OhlcvBar


StrategyAllocator = callable


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Minimum performance thresholds for strategy acceptance."""

    min_sharpe: float = 1.5
    max_drawdown: float = 0.25
    min_profit_factor: float = 1.5
    min_expectancy: float = 0.0
    min_oos_sharpe: float = 1.0
    min_mc_p5_sharpe: float = 0.0
    max_stress_drawdown: float = 0.35


@dataclass
class ValidationResult:
    """Complete validation results for one strategy."""

    strategy_name: str
    in_sample: BacktestResult
    walk_forward: WalkForwardResult | None
    monte_carlo: MonteCarloResult
    stress_tests: list[StressTestResult]
    acceptance: dict[str, bool]
    all_passed: bool
    summary: str


def validate_strategy(
    strategy_name: str,
    allocator: StrategyAllocator,
    bars_by_instrument: dict[str, list[OhlcvBar]],
    funding_by_instrument: dict[str, list[FundingRate]],
    config: BacktestConfig | None = None,
    criteria: AcceptanceCriteria | None = None,
    n_wf_splits: int = 3,
    n_mc_sims: int = 500,
    min_history: int = 15,
) -> ValidationResult:
    """Run full validation pipeline on a strategy.

    Args:
        strategy_name: Name for reporting.
        allocator: Strategy allocator callable.
        bars_by_instrument: OHLCV data per instrument.
        funding_by_instrument: Funding rate data per instrument.
        config: Backtest configuration.
        criteria: Acceptance criteria thresholds.
        n_wf_splits: Number of walk-forward splits.
        n_mc_sims: Number of Monte Carlo simulations.
        min_history: Minimum bars before first allocation.

    Returns:
        ValidationResult with all test results and pass/fail status.
    """
    if config is None:
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            taker_fee_rate=0.0005,
            slippage_bps=2.0,
            partial_fill_ratio=1.0,
            periods_per_year=365.0,
        )
    if criteria is None:
        criteria = AcceptanceCriteria()

    # 1. In-sample backtest
    engine = BacktestEngine(config)
    is_result = engine.run(allocator, bars_by_instrument, funding_by_instrument, min_history)

    # 2. Walk-forward analysis
    wf_result = None
    try:
        wf_result = walk_forward_analysis(
            allocator,
            bars_by_instrument,
            funding_by_instrument,
            config,
            n_splits=n_wf_splits,
            train_ratio=0.6,
            min_history=min_history,
        )
    except ValueError:
        pass  # Not enough data for walk-forward

    # 3. Monte Carlo simulation
    mc_result = monte_carlo_simulation(
        is_result.trade_returns,
        n_simulations=n_mc_sims,
        initial_equity=config.initial_equity,
        periods_per_year=config.periods_per_year,
        seed=42,
    )

    # 4. Stress testing
    stress_results = stress_test(
        allocator,
        bars_by_instrument,
        funding_by_instrument,
        config,
        min_history,
    )

    # 5. Check acceptance criteria
    acceptance: dict[str, bool] = {}
    is_m = is_result.metrics

    acceptance["sharpe_ratio"] = is_m.sharpe_ratio >= criteria.min_sharpe
    acceptance["max_drawdown"] = is_m.max_drawdown <= criteria.max_drawdown
    acceptance["profit_factor"] = is_m.profit_factor >= criteria.min_profit_factor
    acceptance["expectancy"] = is_m.expectancy > criteria.min_expectancy

    if wf_result is not None:
        acceptance["oos_sharpe"] = (
            wf_result.combined_oos_metrics.sharpe_ratio >= criteria.min_oos_sharpe
        )
    else:
        acceptance["oos_sharpe"] = False

    acceptance["mc_p5_sharpe"] = mc_result.p5_sharpe > criteria.min_mc_p5_sharpe

    # Check no stress scenario exceeds max drawdown
    max_stress_dd = max(sr.metrics.max_drawdown for sr in stress_results)
    acceptance["stress_drawdown"] = max_stress_dd <= criteria.max_stress_drawdown

    all_passed = all(acceptance.values())

    # Build summary
    lines = [
        f"=== Validation Report: {strategy_name} ===",
        "",
        "--- In-Sample Performance ---",
        f"  Total Return:   {is_m.total_return:.4%}",
        f"  CAGR:           {is_m.cagr:.4%}",
        f"  Sharpe Ratio:   {is_m.sharpe_ratio:.4f}",
        f"  Sortino Ratio:  {is_m.sortino_ratio:.4f}",
        f"  Max Drawdown:   {is_m.max_drawdown:.4%}",
        f"  Calmar Ratio:   {is_m.calmar_ratio:.4f}",
        f"  Profit Factor:  {is_m.profit_factor:.4f}",
        f"  Win Rate:       {is_m.win_rate:.4%}",
        f"  Expectancy:     {is_m.expectancy:.6f}",
        f"  Total Trades:   {is_m.total_trades}",
        f"  Ann. Volatility:{is_m.annualised_volatility:.4%}",
        "",
    ]

    if wf_result is not None:
        oos_m = wf_result.combined_oos_metrics
        lines.extend([
            "--- Walk-Forward OOS Performance ---",
            f"  OOS Sharpe:     {oos_m.sharpe_ratio:.4f}",
            f"  OOS Return:     {oos_m.total_return:.4%}",
            f"  OOS Max DD:     {oos_m.max_drawdown:.4%}",
            f"  Windows:        {len(wf_result.oos_metrics)}",
            "",
        ])

    lines.extend([
        "--- Monte Carlo (500 sims) ---",
        f"  Median Sharpe:  {mc_result.median_sharpe:.4f}",
        f"  P5 Sharpe:      {mc_result.p5_sharpe:.4f}",
        f"  P95 Sharpe:     {mc_result.p95_sharpe:.4f}",
        f"  Median Max DD:  {mc_result.median_max_dd:.4%}",
        f"  P95 Max DD:     {mc_result.p95_max_dd:.4%}",
        f"  P(Return>0):    {mc_result.prob_positive_return:.2%}",
        "",
        "--- Stress Tests ---",
    ])
    for sr in stress_results:
        lines.append(
            f"  {sr.scenario_name:20s} Sharpe={sr.metrics.sharpe_ratio:7.4f}  "
            f"DD={sr.metrics.max_drawdown:7.4%}  PF={sr.metrics.profit_factor:7.4f}"
        )

    lines.extend([
        "",
        "--- Acceptance Criteria ---",
    ])
    for criterion, passed in acceptance.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"  {criterion:20s} {status}")

    lines.extend([
        "",
        f"{'=== ALL CRITERIA PASSED ===' if all_passed else '=== SOME CRITERIA FAILED ==='}",
    ])

    summary = "\n".join(lines)

    return ValidationResult(
        strategy_name=strategy_name,
        in_sample=is_result,
        walk_forward=wf_result,
        monte_carlo=mc_result,
        stress_tests=stress_results,
        acceptance=acceptance,
        all_passed=all_passed,
        summary=summary,
    )
