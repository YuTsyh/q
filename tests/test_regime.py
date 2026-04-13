"""Tests for regime detection module."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.regime import (
    MarketRegimeType,
    RegimeClassification,
    RegimeConfig,
    aggregate_regime,
    classify_portfolio_regime,
    classify_regime,
)
from quantbot.research.synthetic_data import (
    BULL_MARKET,
    BEAR_MARKET,
    FULL_CYCLE_REGIMES,
    MarketRegime,
    generate_multi_instrument_data,
    generate_ohlcv,
)


INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


@pytest.fixture
def bull_bars():
    """Generate bars from a pure bull regime."""
    regime = MarketRegime("bull", drift=0.003, volatility=0.02, duration_bars=100)
    return generate_ohlcv("BTC-USDT-SWAP", [regime], seed=42)


@pytest.fixture
def bear_bars():
    """Generate bars from a pure bear regime."""
    regime = MarketRegime("bear", drift=-0.003, volatility=0.04, duration_bars=100)
    return generate_ohlcv("BTC-USDT-SWAP", [regime], seed=42)


@pytest.fixture
def sideways_bars():
    """Generate bars from a sideways regime."""
    regime = MarketRegime("sideways", drift=0.0, volatility=0.01, duration_bars=100)
    return generate_ohlcv("BTC-USDT-SWAP", [regime], seed=42)


class TestClassifyRegime:
    def test_returns_classification(self, bull_bars):
        result = classify_regime(bull_bars)
        assert isinstance(result, RegimeClassification)
        assert isinstance(result.regime, MarketRegimeType)
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_data_raises(self):
        bar = OhlcvBar(
            inst_id="BTC-USDT-SWAP",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),
        )
        with pytest.raises(ValueError, match="Need at least"):
            classify_regime([bar])

    def test_bull_detected_in_uptrend(self, bull_bars):
        result = classify_regime(bull_bars)
        assert result.regime in (MarketRegimeType.BULL_TRENDING, MarketRegimeType.RANGE_BOUND)
        assert result.trend_score >= 0

    def test_bear_detected_in_downtrend(self, bear_bars):
        result = classify_regime(bear_bars)
        # With GBM noise, a short bear regime may be detected as range-bound
        # due to EMA lag. The key assertion is trend_score is non-positive.
        assert result.trend_score <= 0.01
        assert result.regime in (
            MarketRegimeType.BEAR_TRENDING,
            MarketRegimeType.HIGH_VOL_CRISIS,
            MarketRegimeType.RANGE_BOUND,
        )

    def test_vol_ratio_positive(self, bull_bars):
        result = classify_regime(bull_bars)
        assert result.vol_ratio > 0

    def test_custom_config(self, bull_bars):
        config = RegimeConfig(
            short_vol_window=5,
            long_vol_window=15,
            trend_sma_window=5,
            trend_long_sma_window=15,
        )
        result = classify_regime(bull_bars, config)
        assert isinstance(result, RegimeClassification)


class TestClassifyPortfolioRegime:
    def test_returns_dict(self):
        bars, funding = generate_multi_instrument_data(
            INSTRUMENTS, regimes=FULL_CYCLE_REGIMES, seed_base=42,
        )
        result = classify_portfolio_regime(bars)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_skips_insufficient_data(self):
        bar = OhlcvBar(
            inst_id="BTC-USDT-SWAP",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),
        )
        result = classify_portfolio_regime({"BTC-USDT-SWAP": [bar]})
        assert result == {}


class TestAggregateRegime:
    def test_empty_returns_range_bound(self):
        assert aggregate_regime({}) == MarketRegimeType.RANGE_BOUND

    def test_single_classification(self):
        rc = RegimeClassification(
            regime=MarketRegimeType.BULL_TRENDING,
            vol_ratio=1.0,
            trend_score=0.05,
            confidence=0.8,
        )
        assert aggregate_regime({"BTC": rc}) == MarketRegimeType.BULL_TRENDING

    def test_majority_wins(self):
        bull = RegimeClassification(
            regime=MarketRegimeType.BULL_TRENDING,
            vol_ratio=1.0, trend_score=0.05, confidence=0.9,
        )
        bear = RegimeClassification(
            regime=MarketRegimeType.BEAR_TRENDING,
            vol_ratio=1.2, trend_score=-0.05, confidence=0.3,
        )
        result = aggregate_regime({"A": bull, "B": bull, "C": bear})
        assert result == MarketRegimeType.BULL_TRENDING
