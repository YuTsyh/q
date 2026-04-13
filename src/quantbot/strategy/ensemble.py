"""Ensemble Momentum-Trend Strategy.

Combines trend following with cross-sectional momentum selection
for more selective entries and higher profit factor.

The strategy requires multiple confirming signals:
1. EMA crossover (trend following)
2. Cross-sectional momentum ranking (top N)
3. Absolute momentum > 0 (dual momentum)
4. Trend strength above threshold

Only instruments passing ALL filters receive allocation.
Position sizing uses inverse volatility for risk parity.

Theory: Signal combination reduces false positives while maintaining
exposure to genuine trends. Based on:
- Harvey et al. (2016) "...and the Cross-Section of Expected Returns"
- Lempérière et al. (2014) "Two Centuries of Trend Following"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.regime import MarketRegimeType, classify_regime


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble strategy."""

    fast_ema: int = 5
    slow_ema: int = 15
    momentum_lookback: int = 10
    vol_lookback: int = 15
    atr_period: int = 10
    trend_ma_lookbacks: tuple[int, ...] = (5, 10, 20)
    min_trend_strength: float = 0.5  # At least half of MAs must confirm
    vol_target: float = 0.12
    max_position_weight: float = 0.15
    gross_exposure: float = 0.5
    stop_loss_atr_multiple: float = 1.5
    top_n: int = 3  # Max instruments
    use_regime_filter: bool = True
    crash_lookback: int = 3
    crash_threshold: float = -0.03


def _ema(values: list[float], period: int) -> list[float]:
    """Compute exponential moving average."""
    if not values or period < 1:
        return []
    alpha = 2.0 / (period + 1)
    result = [values[0]]
    for i in range(1, len(values)):
        result.append(alpha * values[i] + (1 - alpha) * result[-1])
    return result


def _compute_atr(bars: list[OhlcvBar], period: int) -> float:
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


class EnsembleMomentumTrend:
    """Ensemble strategy combining momentum ranking with trend following."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self._config = config or EnsembleConfig()
        self._stop_levels: dict[str, float] = {}

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Compute target weights using ensemble signal filtering."""
        cfg = self._config

        # --- Regime filter: reduce/eliminate exposure in bear/crisis ---
        regime_scale = 1.0
        if cfg.use_regime_filter:
            regime_scale = self._compute_regime_scale(bars_by_instrument)
        if regime_scale <= 0:
            return {}

        candidates: dict[str, float] = {}  # inst_id -> composite_score
        volatilities: dict[str, float] = {}

        min_required = max(
            cfg.slow_ema + 2,
            cfg.momentum_lookback + 2,
            cfg.vol_lookback + 2,
            max(cfg.trend_ma_lookbacks) + 1,
            cfg.atr_period + 2,
        )

        for inst_id, bars in bars_by_instrument.items():
            if len(bars) < min_required:
                continue

            # Per-instrument crash guard
            closes = [float(b.close) for b in bars]
            if len(closes) > cfg.crash_lookback + 1:
                recent_ret = closes[-1] / closes[-cfg.crash_lookback - 1] - 1.0
                if recent_ret < cfg.crash_threshold:
                    continue

            current_price = closes[-1]

            # --- Signal 1: EMA Crossover ---
            fast_ema = _ema(closes, cfg.fast_ema)
            slow_ema = _ema(closes, cfg.slow_ema)
            ema_signal = fast_ema[-1] > slow_ema[-1]
            if not ema_signal:
                continue  # Skip if no trend

            # --- Signal 2: Absolute Momentum ---
            lookback_price = closes[-(cfg.momentum_lookback + 1)]
            if lookback_price <= 0:
                continue
            abs_momentum = (current_price / lookback_price) - 1.0
            if abs_momentum <= 0:
                continue  # Skip if negative absolute momentum

            # --- Signal 3: Trend Strength ---
            above_count = 0
            for lb in cfg.trend_ma_lookbacks:
                if len(closes) >= lb:
                    sma = sum(closes[-lb:]) / lb
                    if current_price > sma:
                        above_count += 1
            trend_strength = above_count / len(cfg.trend_ma_lookbacks)
            if trend_strength < cfg.min_trend_strength:
                continue  # Skip if trend not confirmed

            # --- Signal 4: ATR Stop Check ---
            atr = _compute_atr(bars, cfg.atr_period)
            if inst_id in self._stop_levels:
                if current_price < self._stop_levels[inst_id]:
                    del self._stop_levels[inst_id]
                    continue  # Stopped out
            if atr > 0:
                self._stop_levels[inst_id] = (
                    current_price - cfg.stop_loss_atr_multiple * atr
                )

            # --- Composite Score ---
            # Combine momentum, trend strength, and EMA divergence
            ema_divergence = (fast_ema[-1] - slow_ema[-1]) / slow_ema[-1] if slow_ema[-1] > 0 else 0
            composite = (
                0.4 * abs_momentum
                + 0.3 * trend_strength
                + 0.3 * ema_divergence
            )
            candidates[inst_id] = composite

            # --- Volatility for sizing ---
            returns = []
            for i in range(-cfg.vol_lookback, 0):
                p0 = float(bars[i - 1].close)
                p1 = float(bars[i].close)
                if p0 > 0:
                    returns.append(p1 / p0 - 1.0)
            if len(returns) >= 2:
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
                volatilities[inst_id] = math.sqrt(var)
            else:
                volatilities[inst_id] = 0.01

        if not candidates:
            return {}

        # Select top N by composite score
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_candidates[: cfg.top_n]

        # Inverse volatility weighting
        inv_vols: dict[str, float] = {}
        for inst_id, _ in selected:
            vol = volatilities.get(inst_id, 0.01)
            # Vol-target scaling
            ann_vol = vol * math.sqrt(365.0)
            if ann_vol > 1e-8:
                vol_scalar = cfg.vol_target / ann_vol
            else:
                vol_scalar = 1.0
            inv_vols[inst_id] = vol_scalar

        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return {}

        weights: dict[str, Decimal] = {}
        target_exposure = cfg.gross_exposure * regime_scale
        for inst_id, _ in selected:
            raw_w = inv_vols[inst_id] / total_inv * target_exposure
            capped = min(raw_w, cfg.max_position_weight)
            weights[inst_id] = Decimal(str(round(capped, 6)))

        return weights

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


def create_ensemble_allocator(
    fast_ema: int = 5,
    slow_ema: int = 15,
    momentum_lookback: int = 10,
    vol_target: float = 0.12,
    stop_loss_atr: float = 1.5,
    top_n: int = 3,
    gross_exposure: float = 0.5,
    min_trend_strength: float = 0.5,
) -> Callable:
    """Factory function to create ensemble allocator for backtesting."""
    config = EnsembleConfig(
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        momentum_lookback=momentum_lookback,
        vol_target=vol_target,
        stop_loss_atr_multiple=stop_loss_atr,
        top_n=top_n,
        gross_exposure=gross_exposure,
        min_trend_strength=min_trend_strength,
    )
    strategy = EnsembleMomentumTrend(config)

    def allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return allocator
