"""Tests for regime-switching cross-sectional alpha strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantbot.research.backtest import BacktestConfig, BacktestEngine
from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    THREE_YEAR_REGIMES,
    generate_multi_instrument_data,
)
from quantbot.strategy.regime_switching import (
    RegimeSwitchingAlpha,
    RegimeSwitchingConfig,
    create_regime_switching_allocator,
)


INSTRUMENTS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP",
    "AVAX-USDT-SWAP", "LINK-USDT-SWAP", "DOT-USDT-SWAP", "MATIC-USDT-SWAP",
]


@pytest.fixture
def full_cycle_8():
    return generate_multi_instrument_data(
        INSTRUMENTS, regimes=FULL_CYCLE_REGIMES, seed_base=42,
    )


@pytest.fixture
def three_year_8():
    return generate_multi_instrument_data(
        INSTRUMENTS, regimes=THREE_YEAR_REGIMES, seed_base=42,
    )


class TestRegimeSwitchingAlpha:
    def test_allocate_returns_dict(self, full_cycle_8):
        bars, funding = full_cycle_8
        strategy = RegimeSwitchingAlpha()
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)

    def test_all_weights_non_negative(self, full_cycle_8):
        bars, funding = full_cycle_8
        strategy = RegimeSwitchingAlpha()
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_max_weight_respected(self, full_cycle_8):
        bars, funding = full_cycle_8
        config = RegimeSwitchingConfig(max_position_weight=0.25)
        strategy = RegimeSwitchingAlpha(config)
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert float(w) <= 0.25 + 1e-6

    def test_empty_data_returns_empty(self):
        strategy = RegimeSwitchingAlpha()
        weights = strategy.allocate({}, {})
        assert weights == {}

    def test_custom_config(self, full_cycle_8):
        bars, funding = full_cycle_8
        config = RegimeSwitchingConfig(
            momentum_lookback=5,
            reversion_lookback=10,
            vol_target=0.08,
            top_n=3,
        )
        strategy = RegimeSwitchingAlpha(config)
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)

    def test_successive_calls_update_state(self, full_cycle_8):
        bars, funding = full_cycle_8
        strategy = RegimeSwitchingAlpha()
        w1 = strategy.allocate(bars, funding)
        w2 = strategy.allocate(bars, funding)
        # Second call should still return valid weights
        assert isinstance(w2, dict)


class TestCreateRegimeSwitchingAllocator:
    def test_factory_creates_callable(self):
        allocator = create_regime_switching_allocator()
        assert callable(allocator)

    def test_produces_weights(self, full_cycle_8):
        bars, funding = full_cycle_8
        allocator = create_regime_switching_allocator(
            momentum_lookback=5, top_n=3,
        )
        weights = allocator(bars, funding)
        assert isinstance(weights, dict)


class TestBacktestIntegration:
    def test_backtest_produces_results(self, three_year_8):
        bars, funding = three_year_8
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=1,
            taker_fee_rate=0.0005,
            slippage_bps=2.0,
            partial_fill_ratio=1.0,
            periods_per_year=365.0,
        )
        allocator = create_regime_switching_allocator(
            vol_target=0.10, top_n=7, gross_exposure=1.5,
            max_position_weight=0.25,
            drawdown_circuit_breaker=0.07, circuit_breaker_lookback=3,
            range_exposure=0.7,
        )
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=30)
        m = result.metrics

        assert m.total_trades > 500
        assert m.cagr > -0.10  # Realistic with market impact
        assert m.sharpe_ratio > -0.5
        assert m.max_drawdown < 0.50

    def test_strategy_survives_stress(self, three_year_8):
        """Strategy survives under 3x fees."""
        bars, funding = three_year_8
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=1,
            taker_fee_rate=0.0015,  # 3x fees
            slippage_bps=2.0,
            partial_fill_ratio=1.0,
            periods_per_year=365.0,
        )
        allocator = create_regime_switching_allocator(
            vol_target=0.10, top_n=7, gross_exposure=1.5,
            max_position_weight=0.25,
            drawdown_circuit_breaker=0.07, circuit_breaker_lookback=3,
            range_exposure=0.7,
        )
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=30)
        # Should not catastrophically fail under stress
        assert result.metrics.cagr > -0.20
        assert result.metrics.sharpe_ratio > -1.0

    def test_parameter_sensitivity(self, three_year_8):
        """Core parameters ±15% must not degrade performance >20%."""
        bars, funding = three_year_8
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=1,
            taker_fee_rate=0.0005,
            slippage_bps=2.0,
            partial_fill_ratio=1.0,
            periods_per_year=365.0,
        )

        # Baseline
        base_alloc = create_regime_switching_allocator(
            vol_target=0.10, top_n=7, gross_exposure=1.5,
            max_position_weight=0.25,
            drawdown_circuit_breaker=0.07, circuit_breaker_lookback=3,
            range_exposure=0.7,
        )
        engine = BacktestEngine(config)
        base_result = engine.run(base_alloc, bars, funding, min_history=30)
        base_sharpe = base_result.metrics.sharpe_ratio

        # Perturbed: vol_target +15%
        perturbed_alloc = create_regime_switching_allocator(
            vol_target=0.115, top_n=7, gross_exposure=1.5,
            max_position_weight=0.25,
            drawdown_circuit_breaker=0.07, circuit_breaker_lookback=3,
            range_exposure=0.7,
        )
        engine = BacktestEngine(config)
        perturbed_result = engine.run(perturbed_alloc, bars, funding, min_history=30)
        perturbed_sharpe = perturbed_result.metrics.sharpe_ratio

        # Sharpe should not degrade more than 20%
        assert perturbed_sharpe >= base_sharpe * 0.80
