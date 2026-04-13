"""Tests for crypto-native alpha factors."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.crypto_factors import (
    AmihudIlliquidityFactor,
    BollingerBandWidthFactor,
    CTrendAggregateFactor,
    FundingRateSpreadFactor,
    NvtRatioFactor,
    OBVTrendFactor,
    OrderImbalanceFactor,
    RsiFactory,
    VolatilityOfVolatilityFactor,
    VwapDeviationFactor,
)


# ---------------------------------------------------------------------------
# Inline data generators
# ---------------------------------------------------------------------------

def _make_bars(
    n: int,
    drift: float = 0.001,
    vol: float = 0.02,
    start_price: float = 100.0,
    seed: int = 42,
) -> list[OhlcvBar]:
    rng = random.Random(seed)
    bars: list[OhlcvBar] = []
    price = start_price
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        ret = drift + vol * rng.gauss(0, 1)
        new_price = price * math.exp(ret)
        bars.append(
            OhlcvBar(
                inst_id="TEST-USDT",
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(max(price, new_price) * 1.005, 6))),
                low=Decimal(str(round(min(price, new_price) * 0.995, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal(str(round(rng.uniform(1000, 10000), 2))),
            )
        )
        price = new_price
        t += timedelta(days=1)
    return bars


def _make_funding(n: int, seed: int = 42) -> list[FundingRate]:
    rng = random.Random(seed)
    rates: list[FundingRate] = []
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        rates.append(
            FundingRate(
                inst_id="TEST-USDT",
                funding_time=t,
                funding_rate=Decimal(str(round(rng.gauss(0.0001, 0.0003), 8))),
            )
        )
        t += timedelta(hours=8)
    return rates


def _make_constant_bars(n: int, price: float = 100.0) -> list[OhlcvBar]:
    """Bars with identical OHLC (all same price)."""
    bars: list[OhlcvBar] = []
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    p = Decimal(str(price))
    for _ in range(n):
        bars.append(
            OhlcvBar(
                inst_id="TEST-USDT",
                ts=t,
                open=p,
                high=p,
                low=p,
                close=p,
                volume=Decimal("5000"),
            )
        )
        t += timedelta(days=1)
    return bars


def _make_zero_volume_bars(n: int, seed: int = 42) -> list[OhlcvBar]:
    """Bars with zero volume but varying prices."""
    rng = random.Random(seed)
    bars: list[OhlcvBar] = []
    price = 100.0
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        ret = 0.001 + 0.02 * rng.gauss(0, 1)
        new_price = price * math.exp(ret)
        bars.append(
            OhlcvBar(
                inst_id="TEST-USDT",
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(max(price, new_price) * 1.005, 6))),
                low=Decimal(str(round(min(price, new_price) * 0.995, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal("0"),
            )
        )
        price = new_price
        t += timedelta(days=1)
    return bars


# ---------------------------------------------------------------------------
# NvtRatioFactor
# ---------------------------------------------------------------------------


class TestNvtRatioFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(30)
        result = NvtRatioFactor().compute(bars, [])
        assert isinstance(result, Decimal)
        assert result < Decimal("0")  # returns -NVT

    def test_not_enough_data(self) -> None:
        bars = _make_bars(5)
        with pytest.raises(ValueError, match="not enough bars"):
            NvtRatioFactor().compute(bars, [])

    def test_zero_volume_returns_zero(self) -> None:
        bars = _make_zero_volume_bars(30)
        result = NvtRatioFactor().compute(bars, [])
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# OrderImbalanceFactor
# ---------------------------------------------------------------------------


class TestOrderImbalanceFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(20)
        result = OrderImbalanceFactor().compute(bars, [])
        assert isinstance(result, Decimal)
        assert Decimal("-1") <= result <= Decimal("1")

    def test_not_enough_data(self) -> None:
        bars = _make_bars(3)
        with pytest.raises(ValueError, match="not enough bars"):
            OrderImbalanceFactor().compute(bars, [])

    def test_all_same_price_returns_zero(self) -> None:
        bars = _make_constant_bars(20)
        result = OrderImbalanceFactor().compute(bars, [])
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# FundingRateSpreadFactor
# ---------------------------------------------------------------------------


class TestFundingRateSpreadFactor:
    def test_normal_computation(self) -> None:
        funding = _make_funding(30)
        bars = _make_bars(30)
        result = FundingRateSpreadFactor().compute(bars, funding)
        assert isinstance(result, Decimal)

    def test_not_enough_funding(self) -> None:
        funding = _make_funding(3)
        bars = _make_bars(30)
        with pytest.raises(ValueError, match="not enough funding"):
            FundingRateSpreadFactor().compute(bars, funding)


# ---------------------------------------------------------------------------
# AmihudIlliquidityFactor
# ---------------------------------------------------------------------------


class TestAmihudIlliquidityFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(30)
        result = AmihudIlliquidityFactor().compute(bars, [])
        assert isinstance(result, Decimal)
        assert result <= Decimal("0")  # returns -illiquidity

    def test_not_enough_data(self) -> None:
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="not enough bars"):
            AmihudIlliquidityFactor().compute(bars, [])

    def test_zero_volume_returns_zero(self) -> None:
        bars = _make_zero_volume_bars(30)
        result = AmihudIlliquidityFactor().compute(bars, [])
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# VolatilityOfVolatilityFactor
# ---------------------------------------------------------------------------


class TestVolatilityOfVolatilityFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(50)
        result = VolatilityOfVolatilityFactor().compute(bars, [])
        assert isinstance(result, Decimal)
        assert result <= Decimal("0")  # returns -vol_of_vol

    def test_not_enough_data(self) -> None:
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="not enough bars"):
            VolatilityOfVolatilityFactor().compute(bars, [])

    def test_constant_price_returns_zero(self) -> None:
        bars = _make_constant_bars(50)
        result = VolatilityOfVolatilityFactor().compute(bars, [])
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# RsiFactory
# ---------------------------------------------------------------------------


class TestRsiFactory:
    def test_normal_computation(self) -> None:
        bars = _make_bars(30)
        result = RsiFactory().compute(bars, [])
        assert isinstance(result, Decimal)
        assert Decimal("-1") <= result <= Decimal("1")

    def test_not_enough_data(self) -> None:
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="not enough bars"):
            RsiFactory().compute(bars, [])

    def test_all_same_price_returns_one(self) -> None:
        # With no price changes, avg_gain = avg_loss = 0 → RSI = 100 → (100-50)/50 = 1
        bars = _make_constant_bars(30)
        result = RsiFactory().compute(bars, [])
        assert result == Decimal("1.0")


# ---------------------------------------------------------------------------
# BollingerBandWidthFactor
# ---------------------------------------------------------------------------


class TestBollingerBandWidthFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(30)
        result = BollingerBandWidthFactor().compute(bars, [])
        assert isinstance(result, Decimal)
        assert result < Decimal("0")  # returns -width

    def test_not_enough_data(self) -> None:
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="not enough bars"):
            BollingerBandWidthFactor().compute(bars, [])

    def test_constant_price(self) -> None:
        bars = _make_constant_bars(30)
        result = BollingerBandWidthFactor().compute(bars, [])
        # std = 0 → width = 0 → -width = 0
        assert result == Decimal("0") or result == Decimal("-0.0")


# ---------------------------------------------------------------------------
# VwapDeviationFactor
# ---------------------------------------------------------------------------


class TestVwapDeviationFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(20)
        result = VwapDeviationFactor().compute(bars, [])
        assert isinstance(result, Decimal)

    def test_not_enough_data(self) -> None:
        bars = _make_bars(3)
        with pytest.raises(ValueError, match="not enough bars"):
            VwapDeviationFactor().compute(bars, [])

    def test_zero_volume_returns_zero(self) -> None:
        bars = _make_zero_volume_bars(20)
        result = VwapDeviationFactor().compute(bars, [])
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# OBVTrendFactor
# ---------------------------------------------------------------------------


class TestOBVTrendFactor:
    def test_normal_computation(self) -> None:
        bars = _make_bars(30)
        result = OBVTrendFactor().compute(bars, [])
        assert isinstance(result, Decimal)

    def test_not_enough_data(self) -> None:
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="not enough bars"):
            OBVTrendFactor().compute(bars, [])

    def test_constant_price_returns_zero(self) -> None:
        bars = _make_constant_bars(30)
        result = OBVTrendFactor().compute(bars, [])
        # No price change → OBV stays flat → slope ≈ 0
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# CTrendAggregateFactor
# ---------------------------------------------------------------------------


class TestCTrendAggregateFactor:
    def test_no_sub_factors_returns_zero(self) -> None:
        bars = _make_bars(50)
        result = CTrendAggregateFactor().compute(bars, [])
        assert result == Decimal("0")

    def test_with_sub_factors(self) -> None:
        bars = _make_bars(50)
        funding = _make_funding(50)
        factor = CTrendAggregateFactor(
            factors=(
                RsiFactory(),
                BollingerBandWidthFactor(),
                VwapDeviationFactor(),
            )
        )
        result = factor.compute(bars, funding)
        assert isinstance(result, Decimal)

    def test_single_sub_factor(self) -> None:
        bars = _make_bars(50)
        factor = CTrendAggregateFactor(factors=(RsiFactory(),))
        result = factor.compute(bars, [])
        assert isinstance(result, Decimal)
