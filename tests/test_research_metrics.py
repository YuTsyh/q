"""Tests for research performance metrics computation."""

from __future__ import annotations

import pytest

from quantbot.research.metrics import compute_metrics


class TestComputeMetrics:
    def test_minimum_equity_curve(self):
        metrics = compute_metrics([100.0, 110.0], [0.1])
        assert metrics.total_return == pytest.approx(0.1, abs=1e-4)

    def test_requires_at_least_two_values(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_metrics([100.0], [])

    def test_flat_equity_returns_zero_sharpe(self):
        equity = [100.0] * 50
        metrics = compute_metrics(equity, [0.0] * 49)
        assert metrics.sharpe_ratio == 0.0
        assert metrics.total_return == 0.0

    def test_steady_growth_positive_sharpe(self):
        equity = [100.0]
        for _ in range(100):
            equity.append(equity[-1] * 1.01)
        returns = [0.01] * 100
        metrics = compute_metrics(equity, returns, periods_per_year=252)
        assert metrics.sharpe_ratio == float("inf")  # Zero vol, positive return
        assert metrics.total_return > 0
        assert metrics.max_drawdown == 0.0

    def test_max_drawdown_calculated(self):
        equity = [100.0, 120.0, 90.0, 110.0]
        returns = [0.2, -0.25, 0.222]
        metrics = compute_metrics(equity, returns)
        assert metrics.max_drawdown == pytest.approx(0.25, abs=1e-4)

    def test_profit_factor(self):
        returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        metrics = compute_metrics(equity, returns)
        assert metrics.profit_factor == pytest.approx(4.0, abs=0.5)
        assert metrics.win_rate == pytest.approx(0.6, abs=0.01)

    def test_consecutive_losses(self):
        returns = [0.01, -0.01, -0.02, -0.03, 0.05, -0.01]
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        metrics = compute_metrics(equity, returns)
        assert metrics.max_consecutive_losses == 3

    def test_sortino_ratio_positive(self):
        returns = [0.02, 0.01, 0.03, -0.005, 0.02, 0.015]
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        metrics = compute_metrics(equity, returns, periods_per_year=252)
        assert metrics.sortino_ratio > 0
        assert metrics.sortino_ratio >= metrics.sharpe_ratio

    def test_expectancy_positive(self):
        returns = [0.05, 0.03, -0.01, 0.04, 0.02]
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        metrics = compute_metrics(equity, returns)
        assert metrics.expectancy > 0

    def test_annualised_volatility(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        metrics = compute_metrics(equity, returns, periods_per_year=252)
        assert metrics.annualised_volatility > 0
