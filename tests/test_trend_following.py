"""Tests for volatility-adjusted trend following strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    generate_multi_instrument_data,
)
from quantbot.strategy.trend_following import (
    TrendFollowConfig,
    VolatilityAdjustedTrendFollower,
    _atr,
    _ema,
    create_trend_following_allocator,
)


INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


@pytest.fixture
def three_instrument_data():
    return generate_multi_instrument_data(
        INSTRUMENTS, regimes=FULL_CYCLE_REGIMES, seed_base=42
    )


class TestEma:
    def test_single_value(self):
        result = _ema([100.0], 3)
        assert result == [100.0]

    def test_constant_series(self):
        result = _ema([50.0] * 10, 5)
        assert all(abs(v - 50.0) < 1e-10 for v in result)

    def test_trending_series(self):
        values = [float(i) for i in range(1, 11)]
        result = _ema(values, 3)
        assert len(result) == 10
        # EMA should trail the trend
        assert result[-1] < values[-1]
        assert result[-1] > result[-2]

    def test_empty(self):
        assert _ema([], 3) == []


class TestAtr:
    def test_basic_atr(self):
        bars = _make_bars_with_hl([
            (100, 105, 95),
            (102, 108, 98),
            (106, 112, 100),
            (104, 110, 96),
            (108, 115, 102),
        ])
        atr = _atr(bars, 3)
        assert atr > 0

    def test_not_enough_bars(self):
        bars = _make_bars_with_hl([(100, 105, 95)])
        atr = _atr(bars, 5)
        assert atr == 0.0


class TestVolatilityAdjustedTrendFollower:
    def test_allocate_returns_dict(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = VolatilityAdjustedTrendFollower()
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)

    def test_all_weights_non_negative(self, three_instrument_data):
        bars, funding = three_instrument_data
        strategy = VolatilityAdjustedTrendFollower()
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_max_weight_respected(self, three_instrument_data):
        bars, funding = three_instrument_data
        config = TrendFollowConfig(max_position_weight=0.3)
        strategy = VolatilityAdjustedTrendFollower(config)
        weights = strategy.allocate(bars, funding)
        for w in weights.values():
            assert float(w) <= 0.3 + 1e-6

    def test_empty_data_returns_empty(self):
        strategy = VolatilityAdjustedTrendFollower()
        weights = strategy.allocate({}, {})
        assert weights == {}

    def test_custom_config(self, three_instrument_data):
        bars, funding = three_instrument_data
        config = TrendFollowConfig(
            fast_ema_period=3,
            slow_ema_period=10,
            vol_target=0.1,
            stop_loss_atr_multiple=1.5,
        )
        strategy = VolatilityAdjustedTrendFollower(config)
        weights = strategy.allocate(bars, funding)
        assert isinstance(weights, dict)


class TestCreateTrendFollowingAllocator:
    def test_factory_creates_callable(self):
        allocator = create_trend_following_allocator()
        assert callable(allocator)

    def test_produces_weights(self, three_instrument_data):
        bars, funding = three_instrument_data
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        weights = allocator(bars, funding)
        assert isinstance(weights, dict)


def _make_bars_with_hl(
    data: list[tuple[float, float, float]],
    inst_id: str = "BTC-USDT-SWAP",
) -> list[OhlcvBar]:
    """Create bars from (close, high, low) tuples."""
    bars = []
    for i, (close, high, low) in enumerate(data):
        bars.append(OhlcvBar(
            inst_id=inst_id,
            ts=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            open=Decimal(str(close)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(close)),
            volume=Decimal("1000"),
        ))
    return bars
