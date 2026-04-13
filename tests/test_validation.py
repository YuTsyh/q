"""Tests for the full strategy validation pipeline."""

from __future__ import annotations

import pytest

from quantbot.research.backtest import BacktestConfig
from quantbot.research.validation import (
    AcceptanceCriteria,
    ValidationResult,
    validate_strategy,
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


class TestValidateStrategy:
    def test_returns_validation_result(self, multi_cycle_data, config):
        bars, funding = multi_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        result = validate_strategy(
            "Adaptive Dual Momentum",
            allocator, bars, funding, config,
            n_wf_splits=3, n_mc_sims=50, min_history=15,
        )
        assert isinstance(result, ValidationResult)
        assert result.strategy_name == "Adaptive Dual Momentum"
        assert result.in_sample is not None
        assert result.monte_carlo is not None
        assert len(result.stress_tests) == 5

    def test_acceptance_dict_populated(self, multi_cycle_data, config):
        bars, funding = multi_cycle_data
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        result = validate_strategy(
            "Trend Following",
            allocator, bars, funding, config,
            n_wf_splits=3, n_mc_sims=50, min_history=15,
        )
        assert "sharpe_ratio" in result.acceptance
        assert "max_drawdown" in result.acceptance
        assert "profit_factor" in result.acceptance
        assert "expectancy" in result.acceptance
        assert "mc_p5_sharpe" in result.acceptance

    def test_summary_generated(self, multi_cycle_data, config):
        bars, funding = multi_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        result = validate_strategy(
            "Test Strategy",
            allocator, bars, funding, config,
            n_wf_splits=3, n_mc_sims=50, min_history=15,
        )
        assert "Validation Report" in result.summary
        assert "Acceptance Criteria" in result.summary

    def test_relaxed_criteria_pass(self, multi_cycle_data, config):
        """With very relaxed criteria, validation should pass."""
        bars, funding = multi_cycle_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        relaxed = AcceptanceCriteria(
            min_sharpe=-100.0,
            max_drawdown=1.0,
            min_profit_factor=0.0,
            min_expectancy=-100.0,
            min_oos_sharpe=-100.0,
            min_mc_p5_sharpe=-100.0,
            max_stress_drawdown=1.0,
        )
        result = validate_strategy(
            "Relaxed Test",
            allocator, bars, funding, config,
            criteria=relaxed,
            n_wf_splits=3, n_mc_sims=50, min_history=15,
        )
        assert result.all_passed is True
