"""Regime-Switching Cross-Sectional Alpha Strategy.

A novel strategy that adaptively switches between alpha signals
depending on the detected market regime. Unlike existing strategies
that are purely trend/momentum based, this strategy combines:

1. **Regime Detection**: Classifies markets into bull-trending,
   bear-trending, range-bound, or high-vol-crisis using vol-ratio
   and trend direction metrics.

2. **Adaptive Signal Selection**:
   - Bull-Trending: Momentum + carry (ride the trend aggressively)
   - Bear-Trending: Defensive (go to cash)
   - Range-Bound: Mean-reversion (buy oversold, avoid overbought)
   - High-Vol-Crisis: Flat (preserve capital)

3. **Volatility-Target Position Sizing**: Targets a fixed portfolio
   volatility, scaling positions inversely to realized vol.

4. **Dynamic Stop-Losses**: ATR-based stops that adapt their
   multiplier to the current regime (tighter in crisis, wider in trends).

5. **Cross-Sectional Selection**: Ranks instruments within-regime
   to concentrate capital in the highest-conviction names.

References:
- Ang & Bekaert (2002) "Regime Switches in Interest Rates"
- Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
- Jegadeesh & Titman (1993) "Returns to Buying Winners"
- Bulla et al. (2011) "Markov-Switching Asset Allocation"

Innovation over existing strategies:
- adaptive_momentum.py: Uses fixed factor weights; no regime switching
- trend_following.py: EMA crossover only; no mean-reversion signals
- ensemble.py: All-or-nothing signal gating; no regime-adaptive logic
- breakout.py: Single-instrument price breakout; no cross-sectional analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.regime import (
    MarketRegimeType,
    RegimeClassification,
    RegimeConfig,
    classify_regime,
)


@dataclass(frozen=True)
class RegimeSwitchingConfig:
    """Configuration for regime-switching strategy."""

    # Regime detection
    regime_short_vol: int = 5
    regime_long_vol: int = 20
    regime_trend_sma: int = 10
    regime_trend_long_sma: int = 30
    regime_vol_crisis: float = 1.8
    regime_vol_high: float = 1.3
    regime_trend_threshold: float = 0.015

    # Momentum signal (for trending regimes)
    momentum_lookback: int = 10
    momentum_fast_ema: int = 5
    momentum_slow_ema: int = 15

    # Mean-reversion signal (for range-bound regimes)
    reversion_lookback: int = 15
    reversion_zscore_entry: float = -0.5  # Enter when z-score below this

    # Carry signal
    funding_lookback: int = 6

    # Position sizing
    max_position_weight: float = 0.35
    gross_exposure: float = 1.0
    min_position_weight: float = 0.01

    # Risk management
    atr_period: int = 10
    stop_loss_atr_bull: float = 2.5
    stop_loss_atr_range: float = 1.5
    stop_loss_atr_crisis: float = 1.0
    drawdown_circuit_breaker: float = 0.05  # 5-bar return triggers flat
    circuit_breaker_lookback: int = 5

    # Regime exposure scaling
    bull_exposure_scale: float = 1.0
    bear_exposure_scale: float = 0.0  # Flat in bear
    range_exposure_scale: float = 0.7
    crisis_exposure_scale: float = 0.0  # Flat in crisis

    # Vol targeting
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15% annualised target per position
    annualisation_factor: float = 365.0

    # Top-N selection per regime
    top_n: int = 3


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
        h = float(bars[i].high)
        lo = float(bars[i].low)
        c_prev = float(bars[i - 1].close)
        tr = max(h - lo, abs(h - c_prev), abs(lo - c_prev))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    return sum(true_ranges[-period:]) / period


def _rolling_zscore(closes: list[float], lookback: int) -> float:
    """Compute z-score of current price vs rolling window."""
    if len(closes) < lookback:
        return 0.0
    window = closes[-lookback:]
    mean = sum(window) / len(window)
    var = sum((c - mean) ** 2 for c in window) / max(len(window) - 1, 1)
    std = math.sqrt(var)
    if std < 1e-12:
        return 0.0
    return (closes[-1] - mean) / std


def _compute_volatility(bars: list[OhlcvBar], lookback: int) -> float:
    """Compute realised daily volatility over the lookback window."""
    if len(bars) < lookback + 1:
        return 0.0
    returns = []
    for i in range(-lookback, 0):
        prev = float(bars[i - 1].close)
        curr = float(bars[i].close)
        if prev > 0:
            returns.append(curr / prev - 1.0)
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var)


@dataclass(frozen=True)
class _InstrumentSignal:
    """Internal signal data for one instrument."""

    inst_id: str
    regime: RegimeClassification
    composite_score: float
    vol_scalar: float


class RegimeSwitchingAlpha:
    """Regime-switching cross-sectional alpha strategy.

    Adaptively selects between momentum and mean-reversion signals
    based on the detected market regime, with vol-targeting sizing
    and regime-dependent stop-losses.
    """

    def __init__(self, config: RegimeSwitchingConfig | None = None) -> None:
        self._config = config or RegimeSwitchingConfig()
        self._stop_levels: dict[str, float] = {}
        self._regime_config = RegimeConfig(
            short_vol_window=self._config.regime_short_vol,
            long_vol_window=self._config.regime_long_vol,
            trend_sma_window=self._config.regime_trend_sma,
            trend_long_sma_window=self._config.regime_trend_long_sma,
            vol_crisis_threshold=self._config.regime_vol_crisis,
            vol_high_threshold=self._config.regime_vol_high,
            trend_threshold=self._config.regime_trend_threshold,
        )
        # Portfolio-level drawdown tracking (rolling window)
        self._prev_weights: dict[str, float] = {}
        self._prev_prices: dict[str, float] = {}
        self._equity_history: list[float] = [1.0]  # Rolling equity track
        self._dd_window: int = 60  # Rolling DD window (bars)

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Compute target portfolio weights based on regime-adaptive signals."""
        cfg = self._config
        min_required = max(
            cfg.regime_long_vol + 1,
            cfg.regime_trend_long_sma,
            cfg.momentum_slow_ema + 2,
            cfg.reversion_lookback + 2,
            cfg.vol_lookback + 2,
            cfg.atr_period + 2,
        )

        # Update estimated equity from previous weights and current prices
        current_prices: dict[str, float] = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                current_prices[inst_id] = float(bars[-1].close)

        if self._prev_weights and self._prev_prices:
            port_return = 0.0
            for inst_id, w in self._prev_weights.items():
                if inst_id in current_prices and inst_id in self._prev_prices:
                    prev_p = self._prev_prices[inst_id]
                    curr_p = current_prices[inst_id]
                    if prev_p > 0:
                        port_return += w * (curr_p / prev_p - 1.0)
            new_eq = self._equity_history[-1] * (1.0 + port_return)
            self._equity_history.append(new_eq)
            # Keep only rolling window
            if len(self._equity_history) > self._dd_window:
                self._equity_history = self._equity_history[-self._dd_window:]
        else:
            # First call or after reset - start fresh equity tracking
            if len(self._equity_history) > 1:
                self._equity_history.append(self._equity_history[-1])
            if len(self._equity_history) > self._dd_window:
                self._equity_history = self._equity_history[-self._dd_window:]

        self._prev_prices = current_prices

        # Portfolio-level drawdown control (rolling window peak)
        if len(self._equity_history) >= 2:
            peak = max(self._equity_history)
            current = self._equity_history[-1]
            dd = 1.0 - current / peak if peak > 0 else 0.0
        else:
            dd = 0.0

        dd_scale = 1.0
        dd_limit = cfg.drawdown_circuit_breaker
        if dd > dd_limit:
            # Scale down exposure linearly as drawdown increases beyond limit
            dd_scale = max(0.0, 1.0 - (dd - dd_limit) / dd_limit)

        # Fast crash detection on individual instruments
        circuit_breaker_active = False
        for bars in bars_by_instrument.values():
            if len(bars) >= cfg.circuit_breaker_lookback + 1:
                closes = [float(b.close) for b in bars]
                recent_ret = closes[-1] / closes[-cfg.circuit_breaker_lookback - 1] - 1.0
                if recent_ret < -cfg.drawdown_circuit_breaker:
                    circuit_breaker_active = True
                    break

        if circuit_breaker_active or dd_scale <= 0:
            self._stop_levels.clear()
            self._prev_weights = {}
            return {}

        instrument_signals: dict[str, _InstrumentSignal] = {}

        for inst_id, bars in bars_by_instrument.items():
            if len(bars) < min_required:
                continue

            funding = funding_by_instrument.get(inst_id, [])

            try:
                regime = classify_regime(bars, self._regime_config)
            except ValueError:
                continue

            signal = self._compute_signal(inst_id, bars, funding, regime)
            if signal is not None:
                instrument_signals[inst_id] = signal

        if not instrument_signals:
            self._prev_weights = {}
            return {}

        portfolio = self._construct_portfolio(instrument_signals)

        # Apply drawdown scaling to final weights
        if dd_scale < 1.0 and portfolio:
            portfolio = {
                k: Decimal(str(round(float(v) * dd_scale, 6)))
                for k, v in portfolio.items()
                if float(v) * dd_scale >= self._config.min_position_weight
            }

        self._prev_weights = {k: float(v) for k, v in portfolio.items()}
        return portfolio

    def _compute_signal(
        self,
        inst_id: str,
        bars: list[OhlcvBar],
        funding: list[FundingRate],
        regime: RegimeClassification,
    ) -> _InstrumentSignal | None:
        """Compute composite signal for one instrument given its regime."""
        cfg = self._config
        closes = [float(b.close) for b in bars]
        current_price = closes[-1]

        # ATR for stop-loss
        atr = _atr(bars, cfg.atr_period)

        # Check existing stop-loss
        if inst_id in self._stop_levels:
            if current_price < self._stop_levels[inst_id]:
                del self._stop_levels[inst_id]
                return None  # Stopped out

        # Determine exposure scale and signal based on regime
        if regime.regime == MarketRegimeType.BEAR_TRENDING:
            exposure_scale = cfg.bear_exposure_scale
            if exposure_scale <= 0:
                self._stop_levels.pop(inst_id, None)
                return None
            composite = self._carry_signal(funding)
            stop_mult = cfg.stop_loss_atr_crisis

        elif regime.regime == MarketRegimeType.HIGH_VOL_CRISIS:
            exposure_scale = cfg.crisis_exposure_scale
            if exposure_scale <= 0:
                self._stop_levels.pop(inst_id, None)
                return None
            composite = self._carry_signal(funding)
            stop_mult = cfg.stop_loss_atr_crisis

        elif regime.regime == MarketRegimeType.RANGE_BOUND:
            reversion = self._mean_reversion_signal(closes)
            carry = self._carry_signal(funding)
            composite = 0.7 * reversion + 0.3 * carry
            if composite <= 0:
                return None
            exposure_scale = cfg.range_exposure_scale
            stop_mult = cfg.stop_loss_atr_range

        else:  # BULL_TRENDING
            momentum = self._momentum_signal(closes)
            carry = self._carry_signal(funding)
            composite = 0.7 * momentum + 0.3 * carry
            if composite <= 0:
                return None
            exposure_scale = cfg.bull_exposure_scale
            stop_mult = cfg.stop_loss_atr_bull

        # Set stop-loss
        if atr > 0:
            self._stop_levels[inst_id] = current_price - stop_mult * atr

        # Vol-targeting scalar
        daily_vol = _compute_volatility(bars, cfg.vol_lookback)
        ann_vol = daily_vol * math.sqrt(cfg.annualisation_factor) if daily_vol > 0 else cfg.vol_target
        if ann_vol > 1e-8:
            vol_scalar = cfg.vol_target / ann_vol
        else:
            vol_scalar = 1.0

        # Apply exposure scaling
        vol_scalar *= exposure_scale

        return _InstrumentSignal(
            inst_id=inst_id,
            regime=regime,
            composite_score=composite,
            vol_scalar=vol_scalar,
        )

    def _momentum_signal(self, closes: list[float]) -> float:
        """Compute momentum signal: EMA crossover + return magnitude."""
        cfg = self._config
        fast = _ema(closes, cfg.momentum_fast_ema)
        slow = _ema(closes, cfg.momentum_slow_ema)

        if not fast or not slow:
            return 0.0

        # Must have positive EMA crossover
        if fast[-1] <= slow[-1]:
            return 0.0

        # Return magnitude over lookback
        lb = min(cfg.momentum_lookback, len(closes) - 1)
        if lb < 1 or closes[-lb - 1] <= 0:
            return 0.0
        raw_momentum = closes[-1] / closes[-lb - 1] - 1.0

        return max(0.0, raw_momentum)

    def _mean_reversion_signal(self, closes: list[float]) -> float:
        """Compute mean-reversion signal: buy when oversold."""
        cfg = self._config
        zscore = _rolling_zscore(closes, cfg.reversion_lookback)

        if zscore < cfg.reversion_zscore_entry:
            return min(1.0, abs(zscore - cfg.reversion_zscore_entry))
        return 0.0

    def _carry_signal(self, funding: list[FundingRate]) -> float:
        """Compute carry signal from funding rates.

        In crypto perps, negative funding = shorts pay longs = long carry.
        Also treats very low positive funding as mildly positive signal.
        """
        cfg = self._config
        if len(funding) < cfg.funding_lookback:
            return 0.0

        recent = funding[-cfg.funding_lookback:]
        avg_rate = sum(float(f.funding_rate) for f in recent) / len(recent)

        # Negative funding = strong carry signal
        if avg_rate < 0:
            return min(1.0, abs(avg_rate) * 5000)
        # Very low positive funding (< median) is a mild positive
        if avg_rate < 0.0002:
            return 0.3 * (1.0 - avg_rate / 0.0002)
        return 0.0

    def _construct_portfolio(
        self,
        signals: dict[str, _InstrumentSignal],
    ) -> dict[str, Decimal]:
        """Rank, select top-N, and size according to vol-targeting.

        Unlike traditional portfolio construction that normalises to a fixed
        gross exposure, this preserves the vol-targeting by NOT normalising
        when total exposure is below the limit. This ensures position sizes
        genuinely reflect the volatility-targeting constraint.
        """
        cfg = self._config

        # Sort by composite score descending
        ranked = sorted(
            signals.items(),
            key=lambda x: x[1].composite_score,
            reverse=True,
        )
        selected = ranked[: cfg.top_n]

        # Compute raw weights using vol-scalar (already regime-adjusted)
        raw_weights: dict[str, float] = {}
        for inst_id, sig in selected:
            raw_weights[inst_id] = sig.vol_scalar

        if not raw_weights:
            return {}

        # Cap individual weights and total exposure
        total = sum(raw_weights.values())
        if total > cfg.gross_exposure:
            # Scale down proportionally if over gross exposure limit
            scale = cfg.gross_exposure / total
            raw_weights = {k: v * scale for k, v in raw_weights.items()}

        target: dict[str, Decimal] = {}
        for inst_id, w in raw_weights.items():
            capped = min(w, cfg.max_position_weight)
            if capped >= cfg.min_position_weight:
                target[inst_id] = Decimal(str(round(capped, 6)))

        return target


def create_regime_switching_allocator(
    momentum_lookback: int = 10,
    reversion_lookback: int = 15,
    vol_target: float = 0.15,
    top_n: int = 3,
    gross_exposure: float = 1.0,
    max_position_weight: float = 0.35,
    stop_loss_atr_bull: float = 2.5,
    stop_loss_atr_range: float = 1.5,
    stop_loss_atr_crisis: float = 1.0,
    bull_exposure: float = 1.0,
    bear_exposure: float = 0.0,
    range_exposure: float = 0.5,
    crisis_exposure: float = 0.0,
    regime_trend_threshold: float = 0.015,
    drawdown_circuit_breaker: float = 0.05,
    circuit_breaker_lookback: int = 5,
) -> callable:
    """Factory function to create regime-switching allocator for backtesting.

    Args:
        momentum_lookback: Lookback for momentum signal.
        reversion_lookback: Lookback for mean-reversion z-score.
        vol_target: Target annualised volatility per position.
        top_n: Maximum instruments to hold.
        gross_exposure: Total portfolio gross exposure.
        max_position_weight: Maximum weight per instrument.
        stop_loss_atr_bull: ATR multiplier for stops in bull regime.
        stop_loss_atr_range: ATR multiplier for stops in range regime.
        stop_loss_atr_crisis: ATR multiplier for stops in crisis regime.
        bull_exposure: Exposure scaling in bull regime.
        bear_exposure: Exposure scaling in bear regime.
        range_exposure: Exposure scaling in range regime.
        crisis_exposure: Exposure scaling in crisis regime.
        regime_trend_threshold: Threshold for regime trend classification.
        drawdown_circuit_breaker: Recent return threshold triggering flat.
        circuit_breaker_lookback: Number of bars for circuit breaker check.

    Returns:
        Allocator callable compatible with BacktestEngine.
    """
    config = RegimeSwitchingConfig(
        momentum_lookback=momentum_lookback,
        reversion_lookback=reversion_lookback,
        vol_target=vol_target,
        top_n=top_n,
        gross_exposure=gross_exposure,
        max_position_weight=max_position_weight,
        stop_loss_atr_bull=stop_loss_atr_bull,
        stop_loss_atr_range=stop_loss_atr_range,
        stop_loss_atr_crisis=stop_loss_atr_crisis,
        bull_exposure_scale=bull_exposure,
        bear_exposure_scale=bear_exposure,
        range_exposure_scale=range_exposure,
        crisis_exposure_scale=crisis_exposure,
        regime_trend_threshold=regime_trend_threshold,
        drawdown_circuit_breaker=drawdown_circuit_breaker,
        circuit_breaker_lookback=circuit_breaker_lookback,
    )
    strategy = RegimeSwitchingAlpha(config)

    def allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return allocator
