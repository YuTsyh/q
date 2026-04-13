"""Volatility Mean-Reversion Strategy with Garman-Klass Estimator.

A novel spot-only strategy that exploits the mean-reverting nature of
volatility in crypto markets. Unlike existing strategies in this repository
that focus on price-based signals, this strategy treats volatility itself
as the primary alpha source.

Core thesis (Engle & Patton 2001): realised volatility is strongly
mean-reverting.  After extreme vol events — spikes or crushes — volatility
tends to revert to its long-run level.  The strategy selects instruments
whose volatility is most displaced from equilibrium and combines this
signal with Bollinger / Keltner squeeze detection and price mean-reversion
factors.

Key components:
1. **Garman-Klass (1980) OHLC volatility estimator** — more efficient than
   close-to-close because it uses full bar range information.
2. **Vol z-score** — displacement of short-term GK vol from its long-run
   mean / std, used to gate entries and scale exposure.
3. **Bollinger-inside-Keltner squeeze detection** — identifies compressed
   vol regimes that precede breakout expansion.
4. **Composite mean-reversion signal** — RSI + VWAP deviation + vol-regime
   score, gated by Markov regime.
5. **Bayesian HMM regime gating** — via :class:`MarkovRegimeDetector`.
6. **Volatility-target position sizing** with ATR-based stops, drawdown
   circuit breaker, and single-instrument crash guard.

Innovation over existing strategies:
- ``mean_reversion_markov.py``: uses a composite z-score of 5 *price*
  factors for mean-reversion.  This strategy uses *volatility* as the
  primary alpha and adds squeeze detection + Keltner confirmation.
- ``regime_switching.py``: simple vol-ratio + SMA, no OHLC vol or squeeze.
- ``adaptive_momentum.py``: momentum/carry, no mean-reversion or vol alpha.
- ``trend_following.py``: EMA crossover, no vol signal.

References:
- Garman, M. B. & Klass, M. J. (1980) "On the estimation of security
  price volatilities from historical data", *Journal of Business* 53(1).
- Engle, R. F. & Patton, A. J. (2001) "What good is a volatility model?",
  *Quantitative Finance* 1(2), 237–245.
- Bollinger, J. (2002) *Bollinger on Bollinger Bands*.
- Keltner, C. (1960) *How to Make Money in Commodities*.
- Amihud, Y. (2002) "Illiquidity and stock returns", *Journal of
  Financial Markets* 5(1), 31–56.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from quantbot.research.crypto_factors import (
    BollingerBandWidthFactor,
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
_EPSILON: float = 1e-12

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolMeanReversionConfig:
    """Tunable parameters for the vol mean-reversion strategy.

    All windows are measured in *bars* (typically hourly or daily candles).
    """

    # -- Garman-Klass vol windows -----------------------------------------
    gk_short_window: int = 5
    gk_long_window: int = 30

    # -- Volatility z-score thresholds ------------------------------------
    vol_zscore_high: float = 2.0  # extreme vol → reduce exposure
    vol_zscore_low: float = -1.0  # vol crush → cautious
    vol_zscore_entry_min: float = -0.5  # min z for entry
    vol_zscore_entry_max: float = 1.5  # max z for entry

    # -- Squeeze detection ------------------------------------------------
    bb_lookback: int = 20
    bb_num_std: float = 2.0
    keltner_atr_period: int = 20
    keltner_multiplier: float = 1.5
    squeeze_lookback: int = 10  # bars to confirm squeeze release

    # -- Mean-reversion signals -------------------------------------------
    rsi_lookback: int = 14
    vwap_lookback: int = 10

    # -- Signal weights ---------------------------------------------------
    w_rsi: float = 0.40
    w_vwap: float = 0.30
    w_vol_regime: float = 0.30

    # -- Entry / exit thresholds ------------------------------------------
    entry_threshold: float = 0.15
    exit_threshold: float = -0.05

    # -- Markov regime params ---------------------------------------------
    markov_short_vol: int = 10
    markov_long_vol: int = 40
    markov_amihud_window: int = 20

    # -- Position sizing --------------------------------------------------
    vol_lookback: int = 20
    vol_target: float = 0.15
    annualisation_factor: float = 365.0
    max_position_weight: float = 0.35
    min_position_weight: float = 0.01
    gross_exposure: float = 1.0

    # -- Risk management --------------------------------------------------
    atr_period: int = 14
    stop_loss_atr_calm: float = 2.5
    stop_loss_atr_normal: float = 2.0
    stop_loss_atr_stressed: float = 1.5

    # -- Regime exposure --------------------------------------------------
    calm_exposure: float = 1.0
    normal_exposure: float = 0.7
    stressed_exposure: float = 0.2
    crisis_exposure: float = 0.0

    # -- Drawdown circuit breaker -----------------------------------------
    dd_threshold: float = 0.05
    dd_window: int = 60
    circuit_breaker_lookback: int = 5
    circuit_breaker_drop: float = 0.05

    # -- Top-N selection --------------------------------------------------
    top_n: int = 3


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _garman_klass_var(bars: list[OhlcvBar]) -> list[float]:
    """Per-bar Garman-Klass (1980) variance estimates.

    For each bar:
        GK_var = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Bars with non-positive OHLC values are assigned zero variance.

    Returns:
        A list with one GK variance per bar (same length as *bars*).
    """
    coeff = 2.0 * math.log(2.0) - 1.0
    out: list[float] = []
    for b in bars:
        h = float(b.high)
        lo = float(b.low)
        c = float(b.close)
        o = float(b.open)
        if h <= 0 or lo <= 0 or c <= 0 or o <= 0 or lo > h:
            out.append(0.0)
            continue
        ln_hl = math.log(h / lo)
        ln_co = math.log(c / o)
        gk = 0.5 * ln_hl * ln_hl - coeff * ln_co * ln_co
        out.append(max(gk, 0.0))
    return out


def _rolling_gk_vol(
    gk_vars: list[float],
    window: int,
) -> list[float]:
    """Rolling Garman-Klass volatility (sqrt of mean GK variance).

    Returns one value per position where a full window is available.
    The returned list has length ``max(0, len(gk_vars) - window + 1)``.
    """
    if len(gk_vars) < window or window < 1:
        return []
    out: list[float] = []
    running_sum = sum(gk_vars[:window])
    out.append(math.sqrt(max(running_sum / window, 0.0)))
    for i in range(window, len(gk_vars)):
        running_sum += gk_vars[i] - gk_vars[i - window]
        out.append(math.sqrt(max(running_sum / window, 0.0)))
    return out


def _vol_zscore(
    gk_vars: list[float],
    short_window: int,
    long_window: int,
) -> float | None:
    """Volatility z-score: (short_gk - long_mean) / long_std.

    Returns *None* when there is insufficient data.
    """
    long_vols = _rolling_gk_vol(gk_vars, long_window)
    if len(long_vols) < 2:
        return None
    long_mean = _mean(long_vols)
    long_std = _std(long_vols)
    if long_std < _EPSILON:
        return None
    short_vols = _rolling_gk_vol(gk_vars, short_window)
    if not short_vols:
        return None
    short_latest = short_vols[-1]
    return (short_latest - long_mean) / long_std


def _compute_atr(bars: list[OhlcvBar], period: int) -> float:
    """Average True Range over the trailing *period* bars."""
    if len(bars) < 2 or period < 1:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(bars)):
        h = float(bars[i].high)
        lo = float(bars[i].low)
        c_prev = float(bars[i - 1].close)
        if h <= 0 or lo <= 0 or c_prev <= 0:
            continue
        tr = max(h - lo, abs(h - c_prev), abs(lo - c_prev))
        true_ranges.append(tr)
    if not true_ranges:
        return 0.0
    tail = true_ranges[-period:]
    return sum(tail) / len(tail)


def _bb_width(bars: list[OhlcvBar], lookback: int, num_std: float) -> float:
    """Bollinger Band width: (upper - lower) / middle."""
    if len(bars) < lookback:
        return 0.0
    closes = [float(b.close) for b in bars[-lookback:]]
    middle = _mean(closes)
    if middle < _EPSILON:
        return 0.0
    sd = _std(closes)
    return (2.0 * num_std * sd) / middle


def _keltner_width(bars: list[OhlcvBar], atr_period: int, mult: float) -> float:
    """Keltner Channel width: 2 * mult * ATR / EMA(close)."""
    if len(bars) < atr_period + 1:
        return 0.0
    atr = _compute_atr(bars, atr_period)
    closes = [float(b.close) for b in bars[-atr_period:]]
    ema = _mean(closes)
    if ema < _EPSILON:
        return 0.0
    return (2.0 * mult * atr) / ema


def _detect_squeeze(
    bars: list[OhlcvBar],
    bb_lookback: int,
    bb_num_std: float,
    keltner_atr_period: int,
    keltner_multiplier: float,
    squeeze_lookback: int,
) -> tuple[bool, bool]:
    """Detect Bollinger/Keltner squeeze and squeeze release.

    Returns:
        (is_in_squeeze, is_squeeze_release)
        - is_in_squeeze: BB width < Keltner width right now.
        - is_squeeze_release: was in squeeze during the last
          *squeeze_lookback* bars but no longer.
    """
    required = max(bb_lookback, keltner_atr_period + 1, squeeze_lookback + 1)
    if len(bars) < required:
        return False, False

    current_bb = _bb_width(bars, bb_lookback, bb_num_std)
    current_kelt = _keltner_width(bars, keltner_atr_period, keltner_multiplier)

    in_squeeze = current_bb < current_kelt and current_kelt > _EPSILON

    # Check if any of the previous bars were in a squeeze
    was_in_squeeze = False
    for offset in range(1, min(squeeze_lookback + 1, len(bars))):
        sub_bars = bars[: len(bars) - offset]
        if len(sub_bars) < required:
            break
        prev_bb = _bb_width(sub_bars, bb_lookback, bb_num_std)
        prev_kelt = _keltner_width(
            sub_bars,
            keltner_atr_period,
            keltner_multiplier,
        )
        if prev_kelt > _EPSILON and prev_bb < prev_kelt:
            was_in_squeeze = True
            break

    squeeze_release = was_in_squeeze and not in_squeeze
    return in_squeeze, squeeze_release


def _vol_regime_signal(
    vol_z: float,
    in_squeeze: bool,
    squeeze_release: bool,
    cfg: VolMeanReversionConfig,
) -> float:
    """Score in [-1, 1] from vol z-score and squeeze state.

    * Extreme high vol (z > vol_zscore_high) → negative (expect reversion
      downward in vol, reduce risk).
    * Extreme low vol (z < vol_zscore_low) → slightly negative (vol crush
      precedes expansion, be cautious).
    * Normal zone (entry_min < z < entry_max) → positive, scaled.
    * Squeeze release amplifies the signal toward +1.
    """
    if vol_z > cfg.vol_zscore_high:
        # Extreme vol spike — strong reversion expected.  Reduce exposure
        # but the score itself represents the opportunity: extreme
        # displacement = strong mean-reversion opportunity if direction
        # is handled by the composite signal.
        score = -0.5
    elif vol_z < cfg.vol_zscore_low:
        # Vol crush — expansion likely, cautious
        score = -0.3
    elif cfg.vol_zscore_entry_min <= vol_z <= cfg.vol_zscore_entry_max:
        # Normal zone: linearly scale 0 → 1 across the entry band
        band = cfg.vol_zscore_entry_max - cfg.vol_zscore_entry_min
        score = (vol_z - cfg.vol_zscore_entry_min) / band if band > _EPSILON else 0.5
    else:
        score = 0.0

    # Squeeze release is a strong bullish vol signal
    if squeeze_release:
        score = min(score + 0.4, 1.0)
    elif in_squeeze:
        # Inside a squeeze — vol is coiling; slight positive bias
        score = min(score + 0.15, 1.0)

    return max(-1.0, min(1.0, score))


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
# Strategy class
# ---------------------------------------------------------------------------


class VolMeanReversionAlpha:
    """Volatility mean-reversion alpha with Garman-Klass estimator.

    Call :meth:`allocate` on each rebalance bar to obtain target weights.
    The strategy is stateful: the internal Markov detector, stop-loss map,
    and equity tracker persist across calls.

    Parameters
    ----------
    config : VolMeanReversionConfig | None
        Strategy hyper-parameters.  Defaults to :class:`VolMeanReversionConfig`
        when *None*.
    """

    def __init__(self, config: VolMeanReversionConfig | None = None) -> None:
        self._cfg = config or VolMeanReversionConfig()

        # Markov detector (stateful — belief vector persists)
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
            lookback=self._cfg.bb_lookback,
            num_std=self._cfg.bb_num_std,
        )
        self._vwap = VwapDeviationFactor(lookback=self._cfg.vwap_lookback)

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
        """Return target portfolio weights (spot-long only, values ≥ 0).

        Weights are non-negative :class:`~decimal.Decimal` values that sum
        to at most ``gross_exposure``.
        """
        cfg = self._cfg

        if not bars_by_instrument:
            self._prev_weights = {}
            return {}

        # Minimum bars required across all factor / regime computations
        min_bars = max(
            cfg.gk_long_window + 1,
            cfg.markov_long_vol + 1,
            cfg.markov_amihud_window + 1,
            cfg.rsi_lookback + 1,
            cfg.bb_lookback,
            cfg.vwap_lookback,
            cfg.vol_lookback + 1,
            cfg.atr_period + 1,
            cfg.keltner_atr_period + 1,
            cfg.squeeze_lookback + 1,
        )

        # --- update rolling equity estimate --------------------------------
        current_prices: dict[str, float] = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                price = float(bars[-1].close)
                if price > 0:
                    current_prices[inst_id] = price
        self._update_equity(current_prices)

        # --- portfolio-level drawdown scaling -------------------------------
        dd_scale = self._drawdown_scale()
        if dd_scale <= 0:
            self._stop_levels.clear()
            self._prev_weights = {}
            return {}

        # --- per-instrument crash circuit breaker ---------------------------
        for bars in bars_by_instrument.values():
            if len(bars) >= cfg.circuit_breaker_lookback + 1:
                prev_close = float(
                    bars[-cfg.circuit_breaker_lookback - 1].close,
                )
                curr_close = float(bars[-1].close)
                if prev_close > 0:
                    ret = curr_close / prev_close - 1.0
                    if ret < -cfg.circuit_breaker_drop:
                        self._stop_levels.clear()
                        self._prev_weights = {}
                        return {}

        # --- classify aggregate regime via Markov detector ------------------
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
        stop_mult = stop_mult_map.get(
            regime_state,
            cfg.stop_loss_atr_normal,
        )

        # --- score each instrument ----------------------------------------
        scored: list[tuple[str, float, float]] = []

        for inst_id, bars in bars_by_instrument.items():
            if len(bars) < min_bars:
                continue
            price = float(bars[-1].close)
            if price <= 0:
                continue

            # Hard stop-loss check
            if inst_id in self._stop_levels:
                if price < self._stop_levels[inst_id]:
                    del self._stop_levels[inst_id]
                    continue  # stopped out

            funding = funding_by_instrument.get(inst_id, [])

            # --- Garman-Klass vol z-score ---
            gk_vars = _garman_klass_var(bars)
            vol_z = _vol_zscore(
                gk_vars,
                cfg.gk_short_window,
                cfg.gk_long_window,
            )
            if vol_z is None:
                continue

            # In HIGH_VOL_LOW_LIQ regime, only trade if extreme reversion
            if regime_state == MarkovRegimeState.HIGH_VOL_LOW_LIQ:
                if vol_z < cfg.vol_zscore_high:
                    continue

            # --- Squeeze detection ---
            in_squeeze, squeeze_release = _detect_squeeze(
                bars,
                cfg.bb_lookback,
                cfg.bb_num_std,
                cfg.keltner_atr_period,
                cfg.keltner_multiplier,
                cfg.squeeze_lookback,
            )

            # --- Composite signal ---
            composite = self._composite_score(
                bars,
                funding,
                vol_z,
                in_squeeze,
                squeeze_release,
            )
            if composite is None:
                continue

            if composite < cfg.entry_threshold:
                # Check exit for currently held positions
                if inst_id in self._prev_weights and composite < cfg.exit_threshold:
                    self._stop_levels.pop(inst_id, None)
                continue

            # Set / update stop-loss
            atr = _compute_atr(bars, cfg.atr_period)
            if atr > 0:
                self._stop_levels[inst_id] = price - stop_mult * atr

            # Vol-targeting scalar — skip instrument when vol is zero
            daily_vol = _realised_vol(bars, cfg.vol_lookback)
            if daily_vol <= 0:
                continue
            ann_vol = daily_vol * math.sqrt(cfg.annualisation_factor)
            vol_scalar = cfg.vol_target / ann_vol if ann_vol > _EPSILON else 1.0
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
            scale = cfg.gross_exposure / total if total > _EPSILON else 0.0
            raw = {k: v * scale for k, v in raw.items()}

        _QUANT = Decimal("0.000001")

        # Apply drawdown scale and cap
        weights: dict[str, Decimal] = {}
        for inst_id, w in raw.items():
            capped = min(w * dd_scale, cfg.max_position_weight)
            if capped >= cfg.min_position_weight:
                weights[inst_id] = Decimal(capped).quantize(_QUANT)

        # Final gross exposure cap
        gross_limit = Decimal(str(cfg.gross_exposure))
        gross = sum(weights.values())
        if gross > gross_limit:
            scale_d = gross_limit / gross
            weights = {k: (v * scale_d).quantize(_QUANT) for k, v in weights.items()}

        self._prev_weights = {k: float(v) for k, v in weights.items()}
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite_score(
        self,
        bars: list[OhlcvBar],
        funding: list[FundingRate],
        vol_z: float,
        in_squeeze: bool,
        squeeze_release: bool,
    ) -> float | None:
        """Weighted composite: RSI + VWAP deviation + vol-regime score.

        Returns ``None`` when any required factor cannot be computed.
        """
        cfg = self._cfg
        try:
            rsi_val = float(self._rsi.compute(bars, funding))
            vwap_val = float(self._vwap.compute(bars, funding))
        except (ValueError, ZeroDivisionError):
            return None

        vol_reg = _vol_regime_signal(
            vol_z,
            in_squeeze,
            squeeze_release,
            cfg,
        )

        wt = cfg.w_rsi + cfg.w_vwap + cfg.w_vol_regime
        if wt <= 0:
            return None

        composite = (cfg.w_rsi * rsi_val + cfg.w_vwap * vwap_val + cfg.w_vol_regime * vol_reg) / wt
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
            self._equity_hist.append(
                self._equity_hist[-1] * (1.0 + port_ret),
            )
        elif len(self._equity_hist) > 1:
            self._equity_hist.append(self._equity_hist[-1])
        # Trim to window
        if len(self._equity_hist) > self._cfg.dd_window:
            self._equity_hist = self._equity_hist[-self._cfg.dd_window :]

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


def create_vol_mean_reversion_allocator(
    config: VolMeanReversionConfig | None = None,
) -> Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]:
    """Factory producing an allocator callable for :class:`BacktestEngine`.

    Parameters
    ----------
    config : VolMeanReversionConfig | None
        Strategy hyper-parameters.  When *None* the defaults from
        :class:`VolMeanReversionConfig` are used.

    Returns
    -------
    Callable
        A function ``(bars_by_inst, funding_by_inst) -> dict[str, Decimal]``
        returning non-negative weights that sum to at most ``gross_exposure``.
    """
    strategy = VolMeanReversionAlpha(config)

    def _allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return _allocator
