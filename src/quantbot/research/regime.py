"""Market regime detection for adaptive strategy switching.

Classifies market conditions into discrete regimes based on:
- Realized volatility vs long-term average (vol-ratio)
- Price trend direction (SMA slope)
- Funding rate environment

Regimes:
- BULL_TRENDING: Low-to-normal vol, positive trend
- BEAR_TRENDING: Elevated vol, negative trend
- RANGE_BOUND: Low vol, no clear trend
- HIGH_VOL_CRISIS: Extreme vol spike (> 2× long-term average)

References:
- Ang & Bekaert (2002) "Regime Switches in Interest Rates"
- Bulla et al. (2011) "Markov-Switching Asset Allocation"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from quantbot.research.data import FundingRate, OhlcvBar


class MarketRegimeType(Enum):
    """Discrete market regime classifications."""

    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOL_CRISIS = "high_vol_crisis"


@dataclass(frozen=True)
class RegimeClassification:
    """Result of regime detection for one instrument at one point in time."""

    regime: MarketRegimeType
    vol_ratio: float  # current_vol / long_term_vol
    trend_score: float  # normalized trend direction [-1, 1]
    confidence: float  # regime classification confidence [0, 1]


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime detection."""

    short_vol_window: int = 10
    long_vol_window: int = 40
    trend_sma_window: int = 20
    trend_long_sma_window: int = 50
    vol_crisis_threshold: float = 2.0  # vol_ratio > this => crisis
    vol_high_threshold: float = 1.3  # vol_ratio > this => elevated
    trend_threshold: float = 0.02  # |trend_score| < this => range-bound


def classify_regime(
    bars: list[OhlcvBar],
    config: RegimeConfig | None = None,
) -> RegimeClassification:
    """Classify the current market regime for a single instrument.

    Args:
        bars: Historical OHLCV bars (most recent last).
        config: Regime detection parameters.

    Returns:
        RegimeClassification with regime type and supporting metrics.

    Raises:
        ValueError: If insufficient data for classification.
    """
    if config is None:
        config = RegimeConfig()

    min_required = max(config.long_vol_window + 1, config.trend_long_sma_window)
    if len(bars) < min_required:
        raise ValueError(f"Need at least {min_required} bars for regime detection")

    # Compute returns
    closes = [float(b.close) for b in bars]
    returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]

    if len(returns) < config.long_vol_window:
        raise ValueError("Not enough valid returns for regime detection")

    # Short-term realized volatility
    short_returns = returns[-config.short_vol_window:]
    short_vol = _std(short_returns)

    # Long-term realized volatility
    long_returns = returns[-config.long_vol_window:]
    long_vol = _std(long_returns)

    # Vol ratio
    vol_ratio = short_vol / long_vol if long_vol > 1e-12 else 1.0

    # Trend score: use EMA for smoother signal (less noise sensitivity)
    short_ema = _ema_value(closes, config.trend_sma_window)
    long_ema = _ema_value(closes, config.trend_long_sma_window)

    if long_ema > 0:
        trend_score = (short_ema - long_ema) / long_ema
    else:
        trend_score = 0.0

    # Additional momentum confirmation: absolute return over trend window
    momentum_window = min(config.trend_sma_window, len(closes) - 1)
    if momentum_window > 0 and closes[-momentum_window - 1] > 0:
        abs_momentum = closes[-1] / closes[-momentum_window - 1] - 1.0
    else:
        abs_momentum = 0.0

    # Classify with momentum confirmation
    if vol_ratio > config.vol_crisis_threshold:
        regime = MarketRegimeType.HIGH_VOL_CRISIS
        confidence = min(1.0, (vol_ratio - config.vol_crisis_threshold) / 1.0 + 0.6)
    elif abs(trend_score) < config.trend_threshold:
        regime = MarketRegimeType.RANGE_BOUND
        confidence = 1.0 - abs(trend_score) / config.trend_threshold
    elif trend_score > 0 and abs_momentum > -config.trend_threshold:
        # Bull requires positive EMA trend AND non-negative momentum
        regime = MarketRegimeType.BULL_TRENDING
        confidence = min(1.0, trend_score / 0.1)
    elif trend_score < 0 or abs_momentum < -config.trend_threshold:
        # Bear if EMA trend is negative OR strong negative momentum
        regime = MarketRegimeType.BEAR_TRENDING
        confidence = min(1.0, max(abs(trend_score), abs(min(0, abs_momentum))) / 0.1)
    else:
        regime = MarketRegimeType.RANGE_BOUND
        confidence = 0.5

    return RegimeClassification(
        regime=regime,
        vol_ratio=round(vol_ratio, 4),
        trend_score=round(trend_score, 6),
        confidence=round(max(0.0, min(1.0, confidence)), 4),
    )


def classify_portfolio_regime(
    bars_by_instrument: dict[str, list[OhlcvBar]],
    config: RegimeConfig | None = None,
) -> dict[str, RegimeClassification]:
    """Classify regimes for all instruments in the universe.

    Args:
        bars_by_instrument: OHLCV data per instrument.
        config: Regime detection parameters.

    Returns:
        Dict mapping inst_id to its RegimeClassification.
    """
    classifications: dict[str, RegimeClassification] = {}
    for inst_id, bars in bars_by_instrument.items():
        try:
            classifications[inst_id] = classify_regime(bars, config)
        except ValueError:
            continue
    return classifications


def aggregate_regime(
    classifications: dict[str, RegimeClassification],
) -> MarketRegimeType:
    """Determine the dominant market regime across the portfolio.

    Uses a voting mechanism weighted by confidence.

    Args:
        classifications: Per-instrument regime classifications.

    Returns:
        The dominant MarketRegimeType across the portfolio.
    """
    if not classifications:
        return MarketRegimeType.RANGE_BOUND

    votes: dict[MarketRegimeType, float] = {}
    for rc in classifications.values():
        votes[rc.regime] = votes.get(rc.regime, 0.0) + rc.confidence

    return max(votes, key=lambda r: votes[r])


def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _ema_value(values: list[float], period: int) -> float:
    """Compute exponential moving average and return the final value.

    EMA is less sensitive to noise than SMA, making it more suitable
    for regime detection where lag/noise trade-off is critical.
    """
    if not values or period < 1:
        return 0.0
    alpha = 2.0 / (period + 1)
    ema = values[0]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * ema
    return ema
