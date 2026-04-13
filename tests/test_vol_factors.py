"""Tests for volatility-adjusted factors."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.vol_factors import (
    DualMomentumFactor,
    MeanReversionFactor,
    TrendStrengthFactor,
    VolAdjMomentumFactor,
    VolatilityFactor,
)


def _make_bars(prices: list[float], inst_id: str = "BTC-USDT-SWAP") -> list[OhlcvBar]:
    """Create OHLCV bars from close prices."""
    bars = []
    for i, p in enumerate(prices):
        d = Decimal(str(p))
        bars.append(OhlcvBar(
            inst_id=inst_id,
            ts=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            open=d,
            high=d * Decimal("1.01"),
            low=d * Decimal("0.99"),
            close=d,
            volume=Decimal("1000"),
        ))
    return bars


class TestVolatilityFactor:
    def test_returns_negative_for_ranking(self):
        bars = _make_bars([100, 102, 98, 103, 99, 104])
        factor = VolatilityFactor(lookback=4)
        result = factor.compute(bars, [])
        assert result < 0  # Negative for rank-based scoring

    def test_low_vol_ranks_higher(self):
        low_vol = _make_bars([100, 100.5, 101, 100.8, 101.2, 101.5])
        high_vol = _make_bars([100, 110, 90, 115, 85, 100])
        factor = VolatilityFactor(lookback=4)
        low_result = factor.compute(low_vol, [])
        high_result = factor.compute(high_vol, [])
        assert low_result > high_result  # Less negative = higher rank

    def test_not_enough_bars(self):
        bars = _make_bars([100, 101])
        factor = VolatilityFactor(lookback=4)
        with pytest.raises(ValueError, match="not enough bars"):
            factor.compute(bars, [])


class TestVolAdjMomentumFactor:
    def test_positive_momentum_positive_result(self):
        bars = _make_bars([100, 101, 102, 103, 104, 110])
        factor = VolAdjMomentumFactor(lookback=4)
        result = factor.compute(bars, [])
        assert result > 0

    def test_higher_momentum_lower_vol_scores_higher(self):
        # High momentum, low vol
        good = _make_bars([100, 102, 104, 106, 108, 115])
        # Low momentum, high vol
        bad = _make_bars([100, 110, 90, 95, 105, 102])
        factor = VolAdjMomentumFactor(lookback=4)
        good_score = factor.compute(good, [])
        bad_score = factor.compute(bad, [])
        assert good_score > bad_score


class TestMeanReversionFactor:
    def test_oversold_positive_score(self):
        # Price below SMA → positive mean reversion score
        bars = _make_bars([100, 105, 110, 108, 95])
        factor = MeanReversionFactor(lookback=3)
        result = factor.compute(bars, [])
        assert result > 0  # SMA > current price

    def test_overbought_negative_score(self):
        # Price above SMA → negative score
        bars = _make_bars([100, 95, 90, 92, 110])
        factor = MeanReversionFactor(lookback=3)
        result = factor.compute(bars, [])
        assert result < 0


class TestDualMomentumFactor:
    def test_positive_absolute_momentum(self):
        bars = _make_bars([100, 102, 104, 106, 108, 115])
        factor = DualMomentumFactor(lookback=4)
        result = factor.compute(bars, [])
        assert result > 0

    def test_negative_absolute_momentum_penalised(self):
        bars = _make_bars([100, 98, 95, 90, 85, 80])
        factor = DualMomentumFactor(lookback=4)
        result = factor.compute(bars, [])
        assert result < Decimal("-1")  # Heavy penalty


class TestTrendStrengthFactor:
    def test_strong_uptrend(self):
        bars = _make_bars(list(range(100, 130)))
        factor = TrendStrengthFactor(lookbacks=(3, 5, 10))
        result = factor.compute(bars, [])
        assert result == Decimal("1")  # Above all MAs

    def test_strong_downtrend(self):
        bars = _make_bars(list(range(130, 100, -1)))
        factor = TrendStrengthFactor(lookbacks=(3, 5, 10))
        result = factor.compute(bars, [])
        assert result == Decimal("0")  # Below all MAs
