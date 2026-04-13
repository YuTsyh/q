"""Tests for ensemble momentum-trend strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    generate_multi_instrument_data,
)
from quantbot.strategy.ensemble import (
    EnsembleConfig,
    EnsembleMomentumTrend,
    create_ensemble_allocator,
)


INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


@pytest.fixture
def three_instrument_data():
    return generate_multi_instrument_data(
        INSTRUMENTS, regimes=FULL_CYCLE_REGIMES, seed_base=42
    )


class TestEnsembleMomentumTrend:
    def test_allocate_returns_dict(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = EnsembleMomentumTrend()
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)

    def test_all_weights_non_negative(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = EnsembleMomentumTrend()
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_max_weight_respected(self, three_instrument_data):
        bars, funding = three_instrument_data
        config = EnsembleConfig(max_position_weight=0.2)
        strategy = EnsembleMomentumTrend(config)
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert float(w) <= 0.2 + 1e-6

    def test_empty_data_returns_empty(self):
        strategy = EnsembleMomentumTrend()
        weights = strategy.allocate({}, {})
        assert weights == {}

    def test_top_n_limits_selections(self, three_instrument_data):
        bars, funding = three_instrument_data
        config = EnsembleConfig(top_n=1)
        strategy = EnsembleMomentumTrend(config)
        weights = strategy.allocate(bars, funding)
        assert len(weights) <= 1

    def test_high_trend_threshold_more_selective(self, three_instrument_data):
        bars, funding = three_instrument_data
        low_thresh = EnsembleMomentumTrend(EnsembleConfig(min_trend_strength=0.0))
        high_thresh = EnsembleMomentumTrend(EnsembleConfig(min_trend_strength=1.0))
        w_low = low_thresh.allocate(bars, funding)
        w_high = high_thresh.allocate(bars, funding)
        assert len(w_low) >= len(w_high)


class TestCreateEnsembleAllocator:
    def test_factory_creates_callable(self):
        allocator = create_ensemble_allocator()
        assert callable(allocator)

    def test_produces_weights(self, three_instrument_data):
        bars, funding = three_instrument_data
        allocator = create_ensemble_allocator(fast_ema=5, slow_ema=20)
        weights = allocator(bars, funding)
        assert isinstance(weights, dict)
