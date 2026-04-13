"""Tests for backtesting engine, walk-forward, Monte Carlo, and stress testing."""

from __future__ import annotations


import pytest

from quantbot.research.backtest import (
    BacktestConfig,
    BacktestEngine,
    MonteCarloResult,
    WalkForwardResult,
    monte_carlo_simulation,
    stress_test,
    walk_forward_analysis,
)
from quantbot.research.synthetic_data import (
    generate_multi_instrument_data,
)
from quantbot.strategy.adaptive_momentum import create_adaptive_dual_momentum_allocator
from quantbot.strategy.trend_following import create_trend_following_allocator


@pytest.fixture
def config():
    return BacktestConfig(
        initial_equity=100_000.0,
        rebalance_every_n_bars=5,
        taker_fee_rate=0.0005,
        slippage_bps=2.0,
        partial_fill_ratio=1.0,
        periods_per_year=365.0,
    )


class TestBacktestEngine:
    def test_basic_run(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.equity_curve) > 10
        assert result.metrics is not None

    def test_trend_following_run(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.equity_curve) > 10

    def test_equity_curve_starts_at_initial(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert result.equity_curve[0] == config.initial_equity

    def test_timestamps_populated(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.timestamps) > 0

    def test_weights_history_populated(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.weights_history) > 0

    def test_not_enough_data_raises(self, config):
        from quantbot.research.synthetic_data import MarketRegime
        tiny_regime = [MarketRegime("tiny", 0.0, 0.01, 3)]
        bars, funding = generate_multi_instrument_data(["X-USDT-SWAP"], tiny_regime)
        allocator = create_adaptive_dual_momentum_allocator(top_n=1)
        engine = BacktestEngine(config)
        with pytest.raises(ValueError, match="Not enough"):
            engine.run(allocator, bars, funding, min_history=10)


class TestWalkForwardAnalysis:
    def test_produces_results(self, multi_cycle_data, config):
        bars, funding = multi_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        result = walk_forward_analysis(
            allocator, bars, funding, config,
            n_splits=3, train_ratio=0.6, min_history=15,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.oos_metrics) > 0
        assert result.combined_oos_metrics is not None

    def test_oos_equity_curve_continuous(self, multi_cycle_data, config):
        bars, funding = multi_cycle_data
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        result = walk_forward_analysis(
            allocator, bars, funding, config,
            n_splits=3, train_ratio=0.6, min_history=15,
        )
        assert result.combined_oos_equity[0] == config.initial_equity
        assert len(result.combined_oos_equity) > 1


class TestMonteCarloSimulation:
    def test_basic_run(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003, 0.01]
        result = monte_carlo_simulation(returns, n_simulations=100, seed=42)
        assert isinstance(result, MonteCarloResult)
        assert result.simulations == 100
        assert 0 <= result.prob_positive_return <= 1

    def test_deterministic_with_seed(self):
        returns = [0.01, -0.005, 0.02, -0.01]
        r1 = monte_carlo_simulation(returns, n_simulations=50, seed=123)
        r2 = monte_carlo_simulation(returns, n_simulations=50, seed=123)
        assert r1.median_sharpe == r2.median_sharpe
        assert r1.median_max_dd == r2.median_max_dd

    def test_no_returns_raises(self):
        with pytest.raises(ValueError, match="No trade returns"):
            monte_carlo_simulation([], n_simulations=10)


class TestStressTest:
    def test_produces_five_scenarios(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        results = stress_test(allocator, bars, funding, config, min_history=15)
        assert len(results) == 5
        scenario_names = {r.scenario_name for r in results}
        assert "normal" in scenario_names
        assert "high_fees_3x" in scenario_names
        assert "high_slippage_5x" in scenario_names
        assert "low_fill_50pct" in scenario_names
        assert "combined_stress" in scenario_names

    def test_normal_better_than_stressed(self, full_cycle_data, config):
        bars, funding = full_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        results = stress_test(allocator, bars, funding, config, min_history=15)
        normal = next(r for r in results if r.scenario_name == "normal")
        combined = next(r for r in results if r.scenario_name == "combined_stress")
        # Combined stress should have more drag
        assert normal.metrics.total_return >= combined.metrics.total_return
