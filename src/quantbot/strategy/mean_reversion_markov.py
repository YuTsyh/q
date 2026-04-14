"""Markov-Filtered Cross-Sectional Mean-Reversion Strategy.

A novel spot-only strategy that combines Bayesian Markov regime detection
with crypto-native mean-reversion signals.  Unlike existing strategies in
this repository (breakout, trend-following, adaptive momentum, and the
simpler regime-switching alpha), this strategy:

1. **Bayesian HMM Regime Gating**: Uses the four-state Markov-Switching
   detector (:class:`MarkovRegimeDetector`) with Amihud illiquidity and
   vol-ratio observables.  Mean-reversion signals are only active during
   calm (LOW_VOL_HIGH_LIQ) and normal (MID_VOL_MID_LIQ) regimes where
   mean-reversion edge is empirically strongest.  The strategy moves to
   cash during HIGH_VOL_LOW_LIQ and CRISIS regimes.

2. **Crypto-Native Factor Composite**: Constructs a composite z-score from
   five orthogonal mean-reversion signals:
   - RSI (centered): identifies oversold instruments
   - Bollinger Band width: detects volatility compression (tight bands
     indicate reduced dispersion, a precondition for mean-reversion entry)
   - VWAP deviation: measures displacement from fair value
   - Order-flow imbalance: proxy for order-book pressure
   - OBV trend: accumulation/distribution confirmation

3. **Volatility-Target Position Sizing**: Scales each position inversely
   to realised volatility so that each contributes equal risk, with a
   regime-dependent exposure multiplier.

4. **ATR Stop-Losses**: Mandatory hard stops with multipliers that tighten
   in stressed regimes.

5. **Portfolio Drawdown Circuit Breaker**: Linearly reduces gross exposure
   as the rolling drawdown exceeds a configurable threshold.

Innovation over existing strategies:
- ``regime_switching.py``: uses simpler :func:`classify_regime` (vol-ratio
  + SMA slope), not the Bayesian HMM.  Also primarily momentum-based in
  bull regime.
- ``adaptive_momentum.py``: no regime gating at all; pure momentum/carry.
- ``trend_following.py``: EMA crossover with no mean-reversion signals.
- ``breakout.py``: single-instrument price-breakout shell.

References:
- Amihud (2002) "Illiquidity and stock returns"
- Hamilton (1989) "A new approach to the economic analysis of
  nonstationary time series"
- Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns"
- Poterba & Summers (1988) "Mean Reversion in Stock Prices"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from quantbot.research.crypto_factors import (
    BollingerBandWidthFactor,
    OBVTrendFactor,
    OrderImbalanceFactor,
    RsiFactory,
    VwapDeviationFactor,
)
from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.markov_regime import (
    MarkovRegimeConfig,
    MarkovRegimeDetector,
    MarkovRegimeState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanReversionMarkovConfig:
    """Tunable parameters for the Markov-filtered mean-reversion strategy.

    All windows are measured in *bars* (typically daily candles).
    """

    # -- Markov regime detector params ------------------------------------
    markov_short_vol: int = 10
    markov_long_vol: int = 40
    markov_amihud_window: int = 20

    # -- Factor lookbacks --------------------------------------------------
    rsi_lookback: int = 14
    bb_lookback: int = 20
    bb_num_std: float = 2.0
    vwap_lookback: int = 10
    oi_lookback: int = 10
    obv_lookback: int = 20

    # -- Factor weights (sum to 1 not required; normalised internally) -----
    w_rsi: float = 0.30
    w_bb: float = 0.15
    w_vwap: float = 0.20
    w_oi: float = 0.20
    w_obv: float = 0.15

    # -- Entry / exit thresholds -------------------------------------------
    entry_z_threshold: float = 0.10  # composite > this → enter long
    exit_z_threshold: float = -0.05  # composite < this → exit

    # -- Position sizing ---------------------------------------------------
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15 % annualised per-position vol target
    annualisation_factor: float = 365.0
    max_position_weight: float = 0.35
    min_position_weight: float = 0.01
    gross_exposure: float = 1.0

    # -- Risk management ---------------------------------------------------
    atr_period: int = 14
    stop_loss_atr_calm: float = 2.5
    stop_loss_atr_normal: float = 2.0
    stop_loss_atr_stressed: float = 1.0

    # -- Regime exposure scaling -------------------------------------------
    calm_exposure: float = 1.0  # LOW_VOL_HIGH_LIQ
    normal_exposure: float = 0.8  # MID_VOL_MID_LIQ
    stressed_exposure: float = 0.0  # HIGH_VOL_LOW_LIQ → flat
    crisis_exposure: float = 0.0  # CRISIS → flat

    # -- Drawdown circuit breaker ------------------------------------------
    dd_threshold: float = 0.08  # start scaling at 8% rolling DD
    dd_window: int = 60  # bars for rolling peak
    circuit_breaker_lookback: int = 5
    circuit_breaker_drop: float = 0.08  # single-instrument crash guard

    # -- Top-N selection ---------------------------------------------------
    top_n: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_atr(bars: list[OhlcvBar], period: int) -> float:
    """Average True Range over the trailing *period* bars."""
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


def _realised_vol(bars: list[OhlcvBar], lookback: int) -> float:
    """Realised daily volatility (Bessel-corrected) over *lookback* bars."""
    if len(bars) < lookback + 1:
        return 0.0
    rets: list[float] = []
    for i in range(-lookback, 0):
        prev = float(bars[i - 1].close)
        curr = float(bars[i].close)
        if prev > 0:
            rets.append(curr / prev - 1.0)
    if len(rets) < 2:
        return 0.0
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MarkovMeanReversionAlpha:
    """Markov-filtered cross-sectional mean-reversion alpha.

    Call :meth:`allocate` on each rebalance bar to obtain target weights.
    The strategy is stateful: the internal Markov detector and stop-loss
    map persist across calls.
    """

    def __init__(self, config: MeanReversionMarkovConfig | None = None) -> None:
        self._cfg = config or MeanReversionMarkovConfig()

        # Markov detector (stateful – belief vector persists)
        self._detector = MarkovRegimeDetector(
            MarkovRegimeConfig(
                short_vol_window=self._cfg.markov_short_vol,
                long_vol_window=self._cfg.markov_long_vol,
                amihud_window=self._cfg.markov_amihud_window,
            )
        )

        # Factor instances (stateless)
        self._rsi = RsiFactory(lookback=self._cfg.rsi_lookback)
        self._bb = BollingerBandWidthFactor(
            lookback=self._cfg.bb_lookback, num_std=self._cfg.bb_num_std,
        )
        self._vwap = VwapDeviationFactor(lookback=self._cfg.vwap_lookback)
        self._oi = OrderImbalanceFactor(lookback=self._cfg.oi_lookback)
        self._obv = OBVTrendFactor(lookback=self._cfg.obv_lookback)

        # Per-instrument hard stop-loss levels
        self._stop_levels: dict[str, float] = {}

        # Rolling equity tracker for portfolio drawdown
        self._equity_hist: list[float] = [1.0]
        self._prev_weights: dict[str, float] = {}
        self._prev_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Return target portfolio weights (spot-long only, values ≥ 0)."""
        cfg = self._cfg
        # Minimum bars required across all factor/regime computations
        min_bars = max(
            cfg.markov_long_vol + 1,
            cfg.markov_amihud_window + 1,
            cfg.rsi_lookback + 1,
            cfg.bb_lookback,
            cfg.vwap_lookback,
            cfg.oi_lookback,
            cfg.obv_lookback + 1,
            cfg.vol_lookback + 1,
            cfg.atr_period + 1,
        )

        # --- update rolling equity estimate ---------------------------------
        current_prices: dict[str, float] = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                current_prices[inst_id] = float(bars[-1].close)
        self._update_equity(current_prices)

        # --- portfolio-level drawdown scaling --------------------------------
        dd_scale = self._drawdown_scale()
        if dd_scale <= 0:
            self._stop_levels.clear()
            self._prev_weights = {}
            return {}

        # --- per-instrument crash circuit breaker ----------------------------
        for bars in bars_by_instrument.values():
            if len(bars) >= cfg.circuit_breaker_lookback + 1:
                closes = [float(b.close) for b in bars]
                ret = closes[-1] / closes[-cfg.circuit_breaker_lookback - 1] - 1.0
                if ret < -cfg.circuit_breaker_drop:
                    self._stop_levels.clear()
                    self._prev_weights = {}
                    return {}

        # --- classify aggregate regime via Markov detector -------------------
        # Pick the first instrument with enough data for the portfolio-level
        # regime signal (or aggregate if multiple instruments qualify).
        regime_bars: dict[str, list[OhlcvBar]] = {
            k: v for k, v in bars_by_instrument.items() if len(v) >= min_bars
        }
        if not regime_bars:
            self._prev_weights = {}
            return {}

        regime_result = self._detector.classify_portfolio(regime_bars)
        regime_state = regime_result.state

        exposure_by_regime: dict[MarkovRegimeState, float] = {
            MarkovRegimeState.LOW_VOL_HIGH_LIQ: cfg.calm_exposure,
            MarkovRegimeState.MID_VOL_MID_LIQ: cfg.normal_exposure,
            MarkovRegimeState.HIGH_VOL_LOW_LIQ: cfg.stressed_exposure,
            MarkovRegimeState.CRISIS: cfg.crisis_exposure,
        }
        regime_exposure = exposure_by_regime.get(regime_state, 0.0)
        if regime_exposure <= 0:
            self._stop_levels.clear()
            self._prev_weights = {}
            return {}

        # ATR stop-loss multiplier per regime
        stop_mult_map: dict[MarkovRegimeState, float] = {
            MarkovRegimeState.LOW_VOL_HIGH_LIQ: cfg.stop_loss_atr_calm,
            MarkovRegimeState.MID_VOL_MID_LIQ: cfg.stop_loss_atr_normal,
            MarkovRegimeState.HIGH_VOL_LOW_LIQ: cfg.stop_loss_atr_stressed,
            MarkovRegimeState.CRISIS: cfg.stop_loss_atr_stressed,
        }
        stop_mult = stop_mult_map.get(regime_state, cfg.stop_loss_atr_normal)

        # --- score each instrument ----------------------------------------
        scored: list[tuple[str, float, float]] = []  # (inst_id, composite, vol_scalar)

        for inst_id, bars in bars_by_instrument.items():
            if len(bars) < min_bars:
                continue
            price = float(bars[-1].close)

            # Hard stop-loss check
            if inst_id in self._stop_levels and price < self._stop_levels[inst_id]:
                del self._stop_levels[inst_id]
                continue  # stopped out

            funding = funding_by_instrument.get(inst_id, [])
            composite = self._composite_score(bars, funding)

            if composite is None or composite < cfg.entry_z_threshold:
                # If currently held, check exit threshold
                if inst_id in self._prev_weights and composite is not None:
                    if composite < cfg.exit_z_threshold:
                        self._stop_levels.pop(inst_id, None)
                continue

            # Set / update stop-loss
            atr = _compute_atr(bars, cfg.atr_period)
            if atr > 0:
                self._stop_levels[inst_id] = price - stop_mult * atr

            # Vol-targeting scalar
            daily_vol = _realised_vol(bars, cfg.vol_lookback)
            ann_vol = daily_vol * math.sqrt(cfg.annualisation_factor) if daily_vol > 0 else cfg.vol_target
            vol_scalar = cfg.vol_target / ann_vol if ann_vol > 1e-8 else 1.0
            vol_scalar *= regime_exposure

            scored.append((inst_id, composite, vol_scalar))

        if not scored:
            self._prev_weights = {}
            return {}

        # --- portfolio construction: top-N by composite -------------------
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[: cfg.top_n]

        raw: dict[str, float] = {inst_id: vs for inst_id, _, vs in selected}
        total = sum(raw.values())
        if total > cfg.gross_exposure:
            scale = cfg.gross_exposure / total
            raw = {k: v * scale for k, v in raw.items()}

        # Apply drawdown scale
        weights: dict[str, Decimal] = {}
        for inst_id, w in raw.items():
            capped = min(w * dd_scale, cfg.max_position_weight)
            if capped >= cfg.min_position_weight:
                weights[inst_id] = Decimal(str(round(capped, 6)))

        self._prev_weights = {k: float(v) for k, v in weights.items()}
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite_score(
        self, bars: list[OhlcvBar], funding: list[FundingRate],
    ) -> float | None:
        """Compute the normalised composite mean-reversion score.

        Returns ``None`` when any required factor cannot be computed.
        """
        cfg = self._cfg
        try:
            rsi_val = float(self._rsi.compute(bars, funding))
            bb_val = float(self._bb.compute(bars, funding))
            vwap_val = float(self._vwap.compute(bars, funding))
            oi_val = float(self._oi.compute(bars, funding))
            obv_val = float(self._obv.compute(bars, funding))
        except (ValueError, ZeroDivisionError):
            return None

        wt = cfg.w_rsi + cfg.w_bb + cfg.w_vwap + cfg.w_oi + cfg.w_obv
        if wt <= 0:
            return None
        composite = (
            cfg.w_rsi * rsi_val
            + cfg.w_bb * bb_val
            + cfg.w_vwap * vwap_val
            + cfg.w_oi * oi_val
            + cfg.w_obv * obv_val
        ) / wt
        return composite

    def _update_equity(self, current_prices: dict[str, float]) -> None:
        """Mark-to-market the internal equity tracker."""
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
        # Trim to window
        if len(self._equity_hist) > self._cfg.dd_window:
            self._equity_hist = self._equity_hist[-self._cfg.dd_window:]
        self._prev_prices = current_prices

    def _drawdown_scale(self) -> float:
        """Return an exposure multiplier in [0, 1] based on rolling DD."""
        if len(self._equity_hist) < 2:
            return 1.0
        peak = max(self._equity_hist)
        current = self._equity_hist[-1]
        dd = 1.0 - current / peak if peak > 0 else 0.0
        limit = self._cfg.dd_threshold
        if dd <= limit:
            return 1.0
        return max(0.0, 1.0 - (dd - limit) / limit)


# ---------------------------------------------------------------------------
# Allocator factory (compatible with BacktestEngine)
# ---------------------------------------------------------------------------


def create_mean_reversion_markov_allocator(
    rsi_lookback: float = 14,
    bb_lookback: float = 20,
    vwap_lookback: float = 10,
    oi_lookback: float = 10,
    obv_lookback: float = 20,
    vol_target: float = 0.15,
    top_n: float = 3,
    gross_exposure: float = 1.0,
    max_position_weight: float = 0.35,
    entry_z_threshold: float = 0.15,
    exit_z_threshold: float = -0.05,
    stop_loss_atr_calm: float = 2.5,
    stop_loss_atr_normal: float = 2.0,
    stop_loss_atr_stressed: float = 1.0,
    calm_exposure: float = 1.0,
    normal_exposure: float = 0.8,
    stressed_exposure: float = 0.0,
    crisis_exposure: float = 0.0,
    dd_threshold: float = 0.05,
    circuit_breaker_drop: float = 0.05,
) -> Callable:
    """Factory producing an allocator callable for :class:`BacktestEngine`.

    All numeric arguments are ``float`` so that the perturbation-robustness
    harness can perturb them uniformly.  Integer parameters are rounded
    internally.
    """
    config = MeanReversionMarkovConfig(
        rsi_lookback=max(2, int(round(rsi_lookback))),
        bb_lookback=max(2, int(round(bb_lookback))),
        vwap_lookback=max(2, int(round(vwap_lookback))),
        oi_lookback=max(2, int(round(oi_lookback))),
        obv_lookback=max(2, int(round(obv_lookback))),
        vol_target=vol_target,
        top_n=max(1, int(round(top_n))),
        gross_exposure=gross_exposure,
        max_position_weight=max_position_weight,
        entry_z_threshold=entry_z_threshold,
        exit_z_threshold=exit_z_threshold,
        stop_loss_atr_calm=stop_loss_atr_calm,
        stop_loss_atr_normal=stop_loss_atr_normal,
        stop_loss_atr_stressed=stop_loss_atr_stressed,
        calm_exposure=calm_exposure,
        normal_exposure=normal_exposure,
        stressed_exposure=stressed_exposure,
        crisis_exposure=crisis_exposure,
        dd_threshold=dd_threshold,
        circuit_breaker_drop=circuit_breaker_drop,
    )
    strategy = MarkovMeanReversionAlpha(config)

    def _allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return _allocator
