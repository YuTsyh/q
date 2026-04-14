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
from quantbot.research.regime import MarketRegimeType, classify_regime


@dataclass(frozen=True)
class TrendFollowConfig:
    """Configuration for trend following strategy."""

    fast_ema_period: int = 8
    slow_ema_period: int = 21
    atr_period: int = 14
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15% annualised target vol per position
    max_position_weight: float = 0.20
    stop_loss_atr_multiple: float = 2.5
    min_trend_strength: float = 0.005
    gross_exposure: float = 0.8
    annualisation_factor: float = 365.0  # For crypto daily bars
    use_regime_filter: bool = True
    drawdown_threshold: float = 0.04
    crash_lookback: int = 3
    # Cooldown: bars to skip after a stop-out before re-entering
    stop_cooldown_bars: int = 5
    # Progressive drawdown scaling
    dd_scale_start: float = 0.10  # start reducing at 10% DD
    dd_scale_flat: float = 0.25   # go flat at 25% DD


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
        self._cooldown: dict[str, int] = {}  # inst_id -> bars remaining
        # Rolling equity for progressive DD scaling
        self._equity_hist: list[float] = [1.0]
        self._prev_weights: dict[str, float] = {}
        self._prev_prices: dict[str, float] = {}

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Compute target weights based on trend signals and volatility."""
        # --- Regime filter: reduce/eliminate exposure in bear/crisis ---
        regime_scale = 1.0
        if self._config.use_regime_filter:
            regime_scale = self._compute_regime_scale(bars_by_instrument)
        if regime_scale <= 0:
            return {}

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

            # Per-instrument crash guard
            if self._is_recently_crashed(bars):
                continue

            # Cooldown check: skip if recently stopped out
            if inst_id in self._cooldown:
                self._cooldown[inst_id] -= 1
                if self._cooldown[inst_id] > 0:
                    continue
                else:
                    del self._cooldown[inst_id]

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
                    self._cooldown[inst_id] = self._config.stop_cooldown_bars
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

        # Normalise to gross exposure (scaled by regime)
        total = sum(long_weights.values())
        if total <= 0:
            return {}

        # Progressive drawdown scaling
        dd_scale = self._compute_dd_scale(bars_by_instrument)

        target_exposure = self._config.gross_exposure * regime_scale * dd_scale
        if target_exposure <= 0:
            return {}
        target_weights: dict[str, Decimal] = {}
        for inst_id, w in long_weights.items():
            normalised = w / total * target_exposure
            capped = min(normalised, self._config.max_position_weight)
            target_weights[inst_id] = Decimal(str(round(capped, 6)))

        self._prev_weights = {k: float(v) for k, v in target_weights.items()}
        return target_weights

    def _compute_dd_scale(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> float:
        """Progressive drawdown scaling: reduce exposure as DD grows."""
        # Update equity estimate
        current_prices = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                current_prices[inst_id] = float(bars[-1].close)

        if self._prev_weights and self._prev_prices:
            port_ret = 0.0
            for inst_id, w in self._prev_weights.items():
                if inst_id in current_prices and inst_id in self._prev_prices:
                    prev_p = self._prev_prices[inst_id]
                    curr_p = current_prices[inst_id]
                    if prev_p > 0:
                        port_ret += w * (curr_p / prev_p - 1.0)
            self._equity_hist.append(self._equity_hist[-1] * (1.0 + port_ret))
        elif len(self._equity_hist) > 1:
            self._equity_hist.append(self._equity_hist[-1])

        if len(self._equity_hist) > 60:
            self._equity_hist = self._equity_hist[-60:]
        self._prev_prices = current_prices

        if len(self._equity_hist) < 2:
            return 1.0
        peak = max(self._equity_hist)
        current = self._equity_hist[-1]
        dd = 1.0 - current / peak if peak > 0 else 0.0

        cfg = self._config
        if dd <= cfg.dd_scale_start:
            return 1.0
        if dd >= cfg.dd_scale_flat:
            return 0.0
        return max(0.0, 1.0 - (dd - cfg.dd_scale_start) / (cfg.dd_scale_flat - cfg.dd_scale_start))

    def _compute_regime_scale(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> float:
        """Classify regime and return exposure multiplier."""
        best_bars: list[OhlcvBar] = []
        for bars in bars_by_instrument.values():
            if len(bars) > len(best_bars):
                best_bars = bars
        if len(best_bars) < 51:
            return 1.0
        try:
            regime = classify_regime(best_bars)
            regime_map = {
                MarketRegimeType.BULL_TRENDING: 1.0,
                MarketRegimeType.RANGE_BOUND: 0.5,
                MarketRegimeType.BEAR_TRENDING: 0.0,
                MarketRegimeType.HIGH_VOL_CRISIS: 0.0,
            }
            return regime_map.get(regime.regime, 0.5)
        except (ValueError, ZeroDivisionError):
            return 0.5

    def _is_recently_crashed(self, bars: list[OhlcvBar]) -> bool:
        """Check if an instrument had a recent large drawdown."""
        lookback = self._config.crash_lookback
        if len(bars) < lookback + 1:
            return False
        closes = [float(b.close) for b in bars]
        ret = closes[-1] / closes[-lookback - 1] - 1.0
        return ret < -self._config.drawdown_threshold


def create_trend_following_allocator(
    fast_ema: int = 8,
    slow_ema: int = 21,
    vol_target: float = 0.15,
    atr_period: int = 14,
    stop_loss_atr: float = 2.5,
    gross_exposure: float = 0.8,
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
