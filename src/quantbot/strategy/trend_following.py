"""Volatility-Adjusted Trend Following Strategy.

A systematic trend following strategy with adaptive position sizing:
1. Identifies trends via exponential moving average crossover
2. Sizes positions inversely to recent volatility
3. Applies trailing stop-loss based on ATR
4. Uses time-series momentum for absolute trend filter

Theory: Based on concepts from:
- Baz et al. (2015) "Dissecting Investment Strategies in the Cross
  Section and Time Series"
- Hurst, Ooi & Pedersen (2017) "A Century of Evidence on
  Trend-Following Investing"

The strategy aims for Sharpe > 1.5 through:
- Diversification across multiple instruments
- Volatility normalisation (equal risk allocation)
- Trend filter to avoid whipsaw losses
- ATR-based stop-loss to limit drawdowns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from quantbot.research.data import FundingRate, OhlcvBar


@dataclass(frozen=True)
class TrendFollowConfig:
    """Configuration for trend following strategy."""

    fast_ema_period: int = 5
    slow_ema_period: int = 20
    atr_period: int = 10
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15% annualised target vol per position
    max_position_weight: float = 0.25
    stop_loss_atr_multiple: float = 2.0
    min_trend_strength: float = 0.0  # Absolute momentum threshold
    gross_exposure: float = 1.0
    annualisation_factor: float = 365.0  # For crypto daily bars


def _ema(values: list[float], period: int) -> list[float]:
    """Compute exponential moving average."""
    if not values or period < 1:
        return []
    alpha = 2.0 / (period + 1)
    result = [values[0]]
    for i in range(1, len(values)):
        result.append(alpha * values[i] + (1 - alpha) * result[-1])
    return result


def _atr(bars: list[OhlcvBar], period: int) -> float:
    """Compute Average True Range."""
    if len(bars) < period + 1:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(bars)):
        bar_high = float(bars[i].high)
        bar_low = float(bars[i].low)
        c_prev = float(bars[i - 1].close)
        tr = max(bar_high - bar_low, abs(bar_high - c_prev), abs(bar_low - c_prev))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    return sum(true_ranges[-period:]) / period


class VolatilityAdjustedTrendFollower:
    """Trend following strategy with vol-targeting and ATR stops."""

    def __init__(self, config: TrendFollowConfig | None = None) -> None:
        self._config = config or TrendFollowConfig()
        self._stop_levels: dict[str, float] = {}

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Compute target weights based on trend signals and volatility."""
        signals: dict[str, float] = {}
        raw_weights: dict[str, float] = {}

        for inst_id, bars in bars_by_instrument.items():
            min_required = max(
                self._config.slow_ema_period + 2,
                self._config.vol_lookback + 2,
                self._config.atr_period + 2,
            )
            if len(bars) < min_required:
                continue

            closes = [float(b.close) for b in bars]
            current_price = closes[-1]

            # EMA crossover signal
            fast_ema = _ema(closes, self._config.fast_ema_period)
            slow_ema = _ema(closes, self._config.slow_ema_period)
            signal = 1.0 if fast_ema[-1] > slow_ema[-1] else -1.0

            # Absolute momentum filter
            lookback = min(self._config.slow_ema_period, len(closes) - 1)
            abs_momentum = (closes[-1] / closes[-lookback - 1]) - 1.0 if closes[-lookback - 1] > 0 else 0.0
            if abs_momentum < self._config.min_trend_strength:
                signal = 0.0  # No position if absolute momentum negative

            # ATR-based stop loss check
            atr = _atr(bars, self._config.atr_period)
            if inst_id in self._stop_levels:
                if signal > 0 and current_price < self._stop_levels[inst_id]:
                    signal = 0.0  # Stopped out
                    del self._stop_levels[inst_id]
            if signal > 0 and atr > 0:
                self._stop_levels[inst_id] = (
                    current_price - self._config.stop_loss_atr_multiple * atr
                )

            # Volatility targeting: scale position by target vol / realised vol
            returns = []
            for i in range(-self._config.vol_lookback, 0):
                p0 = float(bars[i - 1].close)
                p1 = float(bars[i].close)
                if p0 > 0:
                    returns.append(p1 / p0 - 1.0)

            if len(returns) >= 2:
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
                daily_vol = math.sqrt(var)
                ann_vol = daily_vol * math.sqrt(self._config.annualisation_factor)
            else:
                ann_vol = self._config.vol_target  # Default assumption

            if ann_vol > 1e-8:
                vol_scalar = self._config.vol_target / ann_vol
            else:
                vol_scalar = 1.0

            raw_weight = signal * vol_scalar
            signals[inst_id] = signal
            raw_weights[inst_id] = raw_weight

        # Spot-only: negative (short) weights are not allowed since
        # OKX Spot Demo (tdMode=cash) does not permit shorting.
        # Instruments with bearish signals receive zero weight (go to cash).
        # NOTE: This means the strategy cannot profit from downtrends;
        # it only avoids them.  Real-market performance in prolonged
        # bear markets will be worse than a long/short variant.
        long_weights = {k: v for k, v in raw_weights.items() if v > 0}
        if not long_weights:
            return {}

        # Normalise to gross exposure
        total = sum(long_weights.values())
        if total <= 0:
            return {}

        target_weights: dict[str, Decimal] = {}
        for inst_id, w in long_weights.items():
            normalised = w / total * self._config.gross_exposure
            capped = min(normalised, self._config.max_position_weight)
            target_weights[inst_id] = Decimal(str(round(capped, 6)))

        return target_weights


def create_trend_following_allocator(
    fast_ema: int = 5,
    slow_ema: int = 20,
    vol_target: float = 0.15,
    atr_period: int = 10,
    stop_loss_atr: float = 2.0,
    gross_exposure: float = 1.0,
) -> Callable:
    """Factory function to create trend-following allocator for backtesting."""
    config = TrendFollowConfig(
        fast_ema_period=fast_ema,
        slow_ema_period=slow_ema,
        atr_period=atr_period,
        vol_target=vol_target,
        stop_loss_atr_multiple=stop_loss_atr,
        gross_exposure=gross_exposure,
    )
    strategy = VolatilityAdjustedTrendFollower(config)

    def allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return allocator
