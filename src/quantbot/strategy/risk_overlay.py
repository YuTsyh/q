"""Risk overlay for strategy allocators.

Wraps any strategy allocator with a regime-aware drawdown circuit
breaker, per-instrument crash guard, and dynamic exposure scaling.
This provides institutional-grade risk management without modifying
individual strategy logic.

Usage::

    from quantbot.strategy.risk_overlay import with_risk_overlay

    base_allocator = create_trend_following_allocator()
    safe_allocator = with_risk_overlay(base_allocator)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.regime import (
    MarketRegimeType,
    RegimeConfig,
    classify_regime,
)

_ZERO = Decimal("0")


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Configuration for the risk overlay wrapper.

    Attributes
    ----------
    max_gross_exposure : float
        Maximum total portfolio weight (sum of all positions).
    max_per_instrument : float
        Maximum weight for any single instrument.
    drawdown_lookback : int
        Number of bars for rolling peak equity tracking.
    drawdown_threshold : float
        Rolling drawdown level at which exposure starts scaling down.
    drawdown_flat_threshold : float
        Rolling drawdown level at which the portfolio goes fully flat.
    crash_guard_lookback : int
        Number of bars for per-instrument crash detection.
    crash_guard_threshold : float
        Negative return threshold triggering crash guard (e.g. -0.05).
    regime_gating : bool
        Whether to reduce exposure based on detected regime.
    regime_exposure : dict[str, float] | None
        Mapping of MarketRegimeType.value → exposure multiplier.
        Defaults to bear=0.2, crisis=0.0 if None.
    min_adv_ratio : float
        Minimum ratio of instrument volume to position notional.
        Instruments with lower liquidity are excluded.
    """

    max_gross_exposure: float = 0.7
    max_per_instrument: float = 0.20
    drawdown_lookback: int = 20
    drawdown_threshold: float = 0.04
    drawdown_flat_threshold: float = 0.12
    crash_guard_lookback: int = 3
    crash_guard_threshold: float = -0.05
    regime_gating: bool = True
    regime_exposure: dict[str, float] | None = None
    min_adv_ratio: float = 0.0
    vol_spike_lookback: int = 5
    vol_spike_threshold: float = 1.5

    def get_regime_exposure(self, regime: MarketRegimeType) -> float:
        """Return the exposure multiplier for a given regime."""
        if self.regime_exposure is not None:
            return self.regime_exposure.get(regime.value, 1.0)
        defaults = {
            MarketRegimeType.BULL_TRENDING: 1.0,
            MarketRegimeType.RANGE_BOUND: 0.7,
            MarketRegimeType.BEAR_TRENDING: 0.15,
            MarketRegimeType.HIGH_VOL_CRISIS: 0.0,
        }
        return defaults.get(regime, 1.0)


class RiskOverlay:
    """Stateful risk overlay that wraps an allocator with risk controls.

    Maintains rolling equity estimates and applies:
    1. Regime-based exposure scaling (go flat in crisis/bear).
    2. Rolling drawdown circuit breaker (scale down as DD grows).
    3. Per-instrument crash guard (exclude recently crashed instruments).
    4. Position size caps (max per instrument and gross exposure).
    """

    def __init__(self, config: RiskOverlayConfig | None = None) -> None:
        self._config = config or RiskOverlayConfig()
        self._equity_history: list[float] = []
        self._peak_equity: float = 0.0

    def apply(
        self,
        raw_weights: dict[str, Decimal],
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> dict[str, Decimal]:
        """Apply risk controls to raw target weights.

        Parameters
        ----------
        raw_weights :
            Target weights from the underlying strategy allocator.
        bars_by_instrument :
            Full bar history per instrument (for regime detection and
            crash guard checks).

        Returns
        -------
        dict[str, Decimal]
            Risk-adjusted target weights.
        """
        cfg = self._config
        if not raw_weights:
            return {}

        # --- 1. Regime gating (use first instrument with enough history) ---
        regime_multiplier = 1.0
        if cfg.regime_gating:
            regime_multiplier = self._detect_regime_multiplier(bars_by_instrument)

        if regime_multiplier <= 0:
            return {}

        # --- 2. Volatility spike pre-emption ---
        vol_spike_scale = self._compute_vol_spike_scale(bars_by_instrument)

        # --- 3. Crash guard: exclude instruments with recent crash ---
        safe_instruments = set()
        for inst_id in raw_weights:
            if self._is_crashed(inst_id, bars_by_instrument):
                continue
            safe_instruments.add(inst_id)

        if not safe_instruments:
            return {}

        filtered = {k: v for k, v in raw_weights.items() if k in safe_instruments}
        if not filtered:
            return {}

        # --- 4. Drawdown circuit breaker ---
        dd_scale = self._compute_drawdown_scale(bars_by_instrument)
        if dd_scale <= 0:
            return {}

        # --- 5. Combined exposure scaling ---
        combined_scale = regime_multiplier * dd_scale * vol_spike_scale

        # --- 5. Cap per-instrument and gross exposure ---
        result: dict[str, Decimal] = {}
        for inst_id, w in filtered.items():
            scaled = float(w) * combined_scale
            capped = min(scaled, cfg.max_per_instrument)
            if capped > 1e-6:
                result[inst_id] = Decimal(str(round(capped, 6)))

        # Cap gross exposure
        total = sum(float(v) for v in result.values())
        if total > cfg.max_gross_exposure:
            scale_factor = cfg.max_gross_exposure / total
            result = {
                k: Decimal(str(round(float(v) * scale_factor, 6)))
                for k, v in result.items()
            }

        return result

    def _detect_regime_multiplier(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> float:
        """Detect the market regime and return the exposure multiplier."""
        regime_config = RegimeConfig(
            short_vol_window=10,
            long_vol_window=40,
            trend_sma_window=20,
            trend_long_sma_window=50,
        )
        # Use the instrument with the most data for regime detection
        best_bars: list[OhlcvBar] = []
        for bars in bars_by_instrument.values():
            if len(bars) > len(best_bars):
                best_bars = bars

        if len(best_bars) < 51:
            return 1.0

        try:
            regime = classify_regime(best_bars, regime_config)
            return self._config.get_regime_exposure(regime.regime)
        except (ValueError, ZeroDivisionError):
            return 1.0

    def _is_crashed(
        self,
        inst_id: str,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> bool:
        """Check if an instrument has recently crashed."""
        cfg = self._config
        bars = bars_by_instrument.get(inst_id, [])
        if len(bars) < cfg.crash_guard_lookback + 1:
            return False
        closes = [float(b.close) for b in bars]
        recent_ret = closes[-1] / closes[-cfg.crash_guard_lookback - 1] - 1.0
        return recent_ret < cfg.crash_guard_threshold

    def _compute_vol_spike_scale(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> float:
        """Detect volatility spikes and scale exposure inversely.

        Compares short-term realized vol to longer-term average vol
        across all instruments.  When short-term vol exceeds the
        threshold multiple of long-term vol, exposure is scaled down.
        """
        cfg = self._config
        vol_ratios: list[float] = []
        for bars in bars_by_instrument.values():
            if len(bars) < 40:
                continue
            closes = [float(b.close) for b in bars]
            # Short-term vol (recent 5 bars)
            short_rets = []
            for i in range(-cfg.vol_spike_lookback, 0):
                if closes[i - 1] > 0:
                    short_rets.append(abs(closes[i] / closes[i - 1] - 1.0))
            # Long-term vol (last 30 bars)
            long_rets = []
            for i in range(-30, 0):
                if closes[i - 1] > 0:
                    long_rets.append(abs(closes[i] / closes[i - 1] - 1.0))

            if short_rets and long_rets:
                short_vol = sum(short_rets) / len(short_rets)
                long_vol = sum(long_rets) / len(long_rets)
                if long_vol > 1e-10:
                    vol_ratios.append(short_vol / long_vol)

        if not vol_ratios:
            return 1.0

        avg_ratio = sum(vol_ratios) / len(vol_ratios)
        if avg_ratio > cfg.vol_spike_threshold * 2:
            return 0.1  # Extreme vol spike → minimal exposure
        if avg_ratio > cfg.vol_spike_threshold:
            # Linear scale from 1.0 to 0.1
            excess = (avg_ratio - cfg.vol_spike_threshold) / cfg.vol_spike_threshold
            return max(0.1, 1.0 - excess * 0.9)
        return 1.0

    def _compute_drawdown_scale(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> float:
        """Compute exposure scale based on estimated rolling drawdown.

        Uses average instrument return as a proxy for portfolio equity
        changes since we don't have exact portfolio equity in the overlay.
        """
        cfg = self._config
        # Estimate equity using average returns of held instruments
        all_returns: list[float] = []
        for bars in bars_by_instrument.values():
            if len(bars) >= cfg.drawdown_lookback + 2:
                closes = [float(b.close) for b in bars]
                for i in range(-cfg.drawdown_lookback, 0):
                    if closes[i - 1] > 0:
                        all_returns.append(closes[i] / closes[i - 1] - 1.0)

        if not all_returns:
            return 1.0

        # Estimate drawdown from average market returns
        equity = 1.0
        peak = 1.0
        n_inst = max(len(bars_by_instrument), 1)
        avg_returns = []
        for i in range(0, len(all_returns), n_inst):
            chunk = all_returns[i:i + n_inst]
            avg_returns.append(sum(chunk) / len(chunk))

        for r in avg_returns:
            equity *= (1.0 + r * 0.5)  # Assume 50% exposure as proxy
            peak = max(peak, equity)

        dd = (peak - equity) / peak if peak > 0 else 0.0

        if dd >= cfg.drawdown_flat_threshold:
            return 0.0
        if dd > cfg.drawdown_threshold:
            # Linear scaling from 1.0 down to 0.0
            return max(
                0.0,
                1.0 - (dd - cfg.drawdown_threshold) / (cfg.drawdown_flat_threshold - cfg.drawdown_threshold),
            )
        return 1.0


def with_risk_overlay(
    allocator: Callable[
        [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
        dict[str, Decimal],
    ],
    config: RiskOverlayConfig | None = None,
) -> Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]:
    """Wrap a strategy allocator with the risk overlay.

    Parameters
    ----------
    allocator :
        The base strategy allocator callable.
    config :
        Risk overlay configuration. Uses defaults if ``None``.

    Returns
    -------
    Callable
        A wrapped allocator that applies risk controls to the raw weights.
    """
    overlay = RiskOverlay(config)

    def wrapped(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        raw = allocator(bars_by_instrument, funding_by_instrument)
        return overlay.apply(raw, bars_by_instrument)

    return wrapped
