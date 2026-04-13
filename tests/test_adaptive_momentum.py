"""Tests for adaptive dual momentum strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    generate_multi_instrument_data,
)
from quantbot.strategy.adaptive_momentum import (
    AdaptiveDualMomentumStrategy,
    InverseVolWeightConstructor,
    create_adaptive_dual_momentum_allocator,
)


INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


@pytest.fixture
def three_instrument_data():
    return generate_multi_instrument_data(
        INSTRUMENTS, regimes=FULL_CYCLE_REGIMES, seed_base=42
    )


class TestInverseVolWeightConstructor:
    def test_equal_weight_without_vol(self):
        constructor = InverseVolWeightConstructor(
            top_n=2, gross_exposure=Decimal("1.0"), max_symbol_weight=Decimal("1.0")
        )
        scores = {"A": Decimal("0.8"), "B": Decimal("0.6"), "C": Decimal("0.3")}
        weights = constructor.construct(scores)
        assert len(weights) == 2
        assert "A" in weights
        assert "B" in weights
        assert weights["A"] == Decimal("0.5")

    def test_inverse_vol_weighting(self):
        constructor = InverseVolWeightConstructor(
            top_n=2, gross_exposure=Decimal("1.0"), max_symbol_weight=Decimal("1.0")
        )
        scores = {"A": Decimal("0.8"), "B": Decimal("0.6")}
        vols = {"A": 0.1, "B": 0.2}  # A is half the vol → double weight
        weights = constructor.construct(scores, vols)
        assert weights["A"] > weights["B"]

    def test_max_weight_cap(self):
        constructor = InverseVolWeightConstructor(
            top_n=2, gross_exposure=Decimal("1.0"), max_symbol_weight=Decimal("0.3")
        )
        scores = {"A": Decimal("0.9"), "B": Decimal("0.1")}
        vols = {"A": 0.001, "B": 0.5}  # A gets huge inv-vol weight
        weights = constructor.construct(scores, vols)
        assert weights["A"] <= Decimal("0.3")

    def test_top_n_must_be_positive(self):
        constructor = InverseVolWeightConstructor(
            top_n=0, gross_exposure=Decimal("1.0"), max_symbol_weight=Decimal("1.0")
        )
        with pytest.raises(ValueError, match="top_n must be positive"):
            constructor.construct({"A": Decimal("1")})


class TestAdaptiveDualMomentumStrategy:
    def test_default_creation(self):
        strategy = AdaptiveDualMomentumStrategy.default(top_n=3)
        assert strategy is not None
        assert len(strategy.factors) == 4

    def test_allocate_returns_weights(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = AdaptiveDualMomentumStrategy.default(top_n=2)
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)

    def test_allocate_respects_top_n(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = AdaptiveDualMomentumStrategy.default(top_n=1)
        weights = strategy.allocate(bars, funding)
        assert len(weights) <= 1

    def test_empty_data_returns_empty(self):
        strategy = AdaptiveDualMomentumStrategy.default(top_n=2)
        weights = strategy.allocate({}, {})
        assert weights == {}


class TestCreateAllocator:
    def test_factory_returns_callable(self):
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        assert callable(allocator)

    def test_allocator_produces_weights(self, three_instrument_data):
        bars, funding = three_instrument_data
        allocator = create_adaptive_dual_momentum_allocator(top_n=2)
        weights = allocator(bars, funding)
        assert isinstance(weights, dict)
