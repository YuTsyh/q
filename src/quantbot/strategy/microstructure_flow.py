"""Microstructure / Order Flow with Funding Rate Spillover Strategy.

A spot-only strategy that exploits **high-frequency microstructure proxies**
derived from OHLCV data, combined with **perpetual funding-rate spillover**
signals to capture short-term inefficiencies caused by information asymmetry
and supply/demand imbalances.

Unlike every other strategy in this repository—which relies on traditional
momentum, mean-reversion, or cross-sectional ranking signals—this strategy:

1. **Volume-Weighted Buying/Selling Pressure (OHLCV proxy)**: Decomposes
   each bar's volume into a buy-pressure and sell-pressure component using
   the close's position within the [low, high] range, then derives a
   Volume Imbalance Ratio (VIR) and its short/medium acceleration.

2. **Trade Flow Toxicity (VPIN proxy)**: Approximates the
   Volume-Synchronized Probability of Informed Trading by bucketing
   volume and measuring the imbalance within equal-volume chunks.
   High VPIN indicates informed-trader activity and is used as a
   danger gate to avoid adverse selection.

3. **Funding Rate Spillover**: Perpetual funding rates reveal the
   futures premium/discount.  A negative funding rate (shorts paying
   longs) is bullish for spot; positive funding is bearish.  The signal
   is contrarian to the rolling average of funding, enhanced by a
   funding momentum (rate-of-change) term.

4. **Price Impact Asymmetry**: Measures how much volume is required to
   move the price up versus down.  If up-moves need less volume the
   microstructure is bullish; the ratio of up-impact to down-impact
   gives a directional flow signal.

5. **Amihud Illiquidity Acceleration**: Compares short-term and
   long-term Amihud illiquidity ratios.  Improving liquidity
   (short < long) is bullish; deteriorating liquidity is bearish.

6. **Markov Regime Gating**: The Bayesian HMM
   (:class:`MarkovRegimeDetector`) scales gross exposure from full
   in calm markets down to zero in crisis.

7. **Risk Controls**: Per-instrument ATR stop-losses (regime-dependent),
   portfolio drawdown circuit breaker, single-instrument crash guard,
   VPIN danger threshold, and minimum signal-persistence filter.

References
----------
- Kyle, A. S. (1985) "Continuous Auctions and Insider Trading",
  *Econometrica* 53(6).
- Easley, D., López de Prado, M. M. & O'Hara, M. (2012)
  "Flow Toxicity and Liquidity in a High-Frequency World",
  *Review of Financial Studies* 25(5).
- Amihud, Y. (2002) "Illiquidity and stock returns",
  *Journal of Financial Markets* 5(1).
- Bouchaud, J.-P., Farmer, J. D. & Lillo, F. (2009)
  "How markets slowly digest changes in supply and demand",
  in *Handbook of Financial Markets: Dynamics and Evolution*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

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
class MicrostructureFlowConfig:
    """Tunable parameters for the microstructure / order-flow strategy.

    All windows are measured in *bars* (typically daily candles).
    """

    # -- Volume imbalance --------------------------------------------------
    vir_short_window: int = 3
    vir_medium_window: int = 10

    # -- VPIN proxy --------------------------------------------------------
    vpin_window: int = 15
    vpin_danger_threshold: float = 0.7  # don't enter if VPIN above

    # -- Funding rate ------------------------------------------------------
    funding_short_avg: int = 3
    funding_long_avg: int = 8

    # -- Impact asymmetry --------------------------------------------------
    impact_lookback: int = 15

    # -- Amihud liquidity --------------------------------------------------
    amihud_short_window: int = 5
    amihud_long_window: int = 20

    # -- Signal weights (sum normalised internally) ------------------------
    w_vir: float = 0.30
    w_vpin: float = 0.15
    w_funding: float = 0.25
    w_impact: float = 0.15
    w_liquidity: float = 0.15

    # -- Entry / exit thresholds -------------------------------------------
    entry_threshold: float = 0.12
    exit_threshold: float = -0.03
    min_signal_bars: int = 2  # signal persistence before entry

    # -- Markov regime detector params -------------------------------------
    markov_short_vol: int = 10
    markov_long_vol: int = 40
    markov_amihud_window: int = 20

    # -- Regime exposure scaling -------------------------------------------
    calm_exposure: float = 1.0  # LOW_VOL_HIGH_LIQ
    normal_exposure: float = 0.6  # MID_VOL_MID_LIQ
    stressed_exposure: float = 0.15  # HIGH_VOL_LOW_LIQ
    crisis_exposure: float = 0.0  # CRISIS → flat

    # -- Position sizing ---------------------------------------------------
    vol_lookback: int = 20
    vol_target: float = 0.15  # 15% annualised per-position vol target
    annualisation_factor: float = 365.0
    max_position_weight: float = 0.35
    min_position_weight: float = 0.01
    gross_exposure: float = 1.0

    # -- Risk management ---------------------------------------------------
    atr_period: int = 14
    stop_loss_atr_calm: float = 2.0
    stop_loss_atr_normal: float = 1.5
    stop_loss_atr_stressed: float = 1.0

    # -- Drawdown circuit breaker ------------------------------------------
    dd_threshold: float = 0.05  # start scaling at 5% rolling DD
    dd_window: int = 60  # bars for rolling peak
    circuit_breaker_lookback: int = 5
    circuit_breaker_drop: float = 0.05  # single-instrument crash guard

    # -- Top-N selection ---------------------------------------------------
    top_n: int = 3


# ---------------------------------------------------------------------------
# Module-level private helpers
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


def _buy_sell_volume(bar: OhlcvBar) -> tuple[float, float]:
    """Decompose *bar* volume into buy and sell pressure proxies.

    Uses the close's position within the [low, high] range as the proxy
    for the fraction of volume that represents buying vs. selling
    pressure.  Bars with zero range (high == low) are split 50/50.

    Returns ``(buy_vol, sell_vol)`` as floats.
    """
    vol = float(bar.volume)
    if vol <= 0:
        return 0.0, 0.0
    h = float(bar.high)
    lo = float(bar.low)
    c = float(bar.close)
    rng = h - lo
    if rng <= _EPSILON:
        return vol * 0.5, vol * 0.5
    buy_frac = (c - lo) / rng
    sell_frac = (h - c) / rng
    return vol * buy_frac, vol * sell_frac


def _volume_imbalance_ratio(
    bars: list[OhlcvBar],
    window: int,
) -> float | None:
    """Rolling Volume Imbalance Ratio (VIR) over the last *window* bars.

    VIR = (sum_buy - sum_sell) / (sum_buy + sum_sell)
    Returns None when there is insufficient data or zero total volume.
    """
    if len(bars) < window:
        return None
    total_buy = 0.0
    total_sell = 0.0
    for bar in bars[-window:]:
        bv, sv = _buy_sell_volume(bar)
        total_buy += bv
        total_sell += sv
    denom = total_buy + total_sell
    if denom <= _EPSILON:
        return None
    return (total_buy - total_sell) / denom


def _vir_acceleration(bars: list[OhlcvBar], cfg: MicrostructureFlowConfig) -> float | None:
    """VIR acceleration: short-window VIR minus medium-window VIR.

    Captures the *change* in buying pressure — a positive acceleration
    signals increasing buy-side urgency.
    """
    vir_short = _volume_imbalance_ratio(bars, cfg.vir_short_window)
    vir_medium = _volume_imbalance_ratio(bars, cfg.vir_medium_window)
    if vir_short is None or vir_medium is None:
        return None
    return vir_short - vir_medium


def _vpin_proxy(bars: list[OhlcvBar], window: int) -> float | None:
    """VPIN proxy — Volume-Synchronized Probability of Informed Trading.

    For each bar, compute ``|buy_vol - sell_vol| / total_vol`` as a
    per-bar toxicity measure, then average over the trailing *window*
    bars.  Returns a value in [0, 1] where higher values indicate
    greater flow toxicity.

    Returns None when there is insufficient data.
    """
    if len(bars) < window:
        return None
    toxicities: list[float] = []
    for bar in bars[-window:]:
        bv, sv = _buy_sell_volume(bar)
        total = bv + sv
        if total <= _EPSILON:
            toxicities.append(0.0)
        else:
            toxicities.append(abs(bv - sv) / total)
    if not toxicities:
        return None
    return sum(toxicities) / len(toxicities)


def _funding_signal(
    funding: list[FundingRate],
    short_avg: int,
    long_avg: int,
) -> tuple[float | None, float | None]:
    """Compute funding rate signals.

    Returns ``(contrarian_signal, funding_momentum)`` where:
    - contrarian_signal = ``-avg(funding_rate, long_avg)``
    - funding_momentum  = ``avg(short) - avg(long)``

    Both are None when there is insufficient data.
    """
    if not funding:
        return None, None
    rates = [float(f.funding_rate) for f in funding]

    # Contrarian signal: negate the rolling average
    if len(rates) < long_avg:
        return None, None
    avg_long = sum(rates[-long_avg:]) / long_avg

    if len(rates) < short_avg:
        return None, None
    avg_short = sum(rates[-short_avg:]) / short_avg

    contrarian = -avg_long
    momentum = avg_short - avg_long
    return contrarian, momentum


def _price_impact_asymmetry(
    bars: list[OhlcvBar],
    lookback: int,
) -> float | None:
    """Measure directional price-impact asymmetry.

    Computes the ratio:
        sum(positive_return / volume) / sum(|negative_return| / volume)

    Values > 1.0 indicate bullish flow: up-moves require less volume
    (lower cost of moving the price up).  Returns None when data is
    insufficient or no moves in either direction.
    """
    if len(bars) < lookback + 1:
        return None
    up_impact = 0.0
    down_impact = 0.0
    for i in range(-lookback, 0):
        prev_c = float(bars[i - 1].close)
        curr_c = float(bars[i].close)
        vol = float(bars[i].volume)
        if prev_c <= 0 or vol <= _EPSILON:
            continue
        ret = curr_c / prev_c - 1.0
        if ret > _EPSILON:
            up_impact += ret / vol
        elif ret < -_EPSILON:
            down_impact += abs(ret) / vol
    if up_impact <= _EPSILON or down_impact <= _EPSILON:
        return None
    return up_impact / down_impact


def _amihud_illiquidity(bars: list[OhlcvBar], window: int) -> float:
    """Amihud illiquidity ratio over the trailing *window* bars.

    Amihud = mean(|return| / dollar_volume) across the window.
    Returns 0.0 when data is insufficient.
    """
    if len(bars) < window + 1:
        return 0.0
    values: list[float] = []
    for i in range(-window, 0):
        prev_c = float(bars[i - 1].close)
        curr_c = float(bars[i].close)
        vol = float(bars[i].volume)
        if prev_c <= 0 or vol <= _EPSILON:
            continue
        ret = abs(curr_c / prev_c - 1.0)
        dollar_vol = vol * curr_c
        if dollar_vol > _EPSILON:
            values.append(ret / dollar_vol)
    return sum(values) / len(values) if values else 0.0


def _liquidity_improvement(
    bars: list[OhlcvBar],
    short_window: int,
    long_window: int,
) -> float | None:
    """Amihud liquidity improvement signal.

    Signal = ``amihud_long / amihud_short - 1``.
    Positive values mean short-term liquidity is better than the
    long-term baseline (bullish).

    Returns None when short-term Amihud is zero or data is
    insufficient for the long window.
    """
    if len(bars) < long_window + 1:
        return None
    amihud_short = _amihud_illiquidity(bars, short_window)
    amihud_long = _amihud_illiquidity(bars, long_window)
    if amihud_short <= _EPSILON:
        # Perfect liquidity at the short end — treat as maximally bullish
        return 1.0 if amihud_long > _EPSILON else 0.0
    return amihud_long / amihud_short - 1.0


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class MicrostructureFlowAlpha:
    """Microstructure / order-flow alpha with funding-rate spillover.

    Stateful strategy: the Markov regime detector, stop-loss map,
    signal-persistence history, and rolling equity tracker all
    persist across successive calls to :meth:`allocate`.

    Call :meth:`allocate` on each rebalance bar to obtain target
    weights.  Returned weights are non-negative (spot-only) and
    sum to at most ``gross_exposure``.
    """

    def __init__(
        self,
        config: MicrostructureFlowConfig | None = None,
    ) -> None:
        self._cfg = config or MicrostructureFlowConfig()

        # Markov detector (stateful — belief vector persists)
        self._detector = MarkovRegimeDetector(
            MarkovRegimeConfig(
                short_vol_window=self._cfg.markov_short_vol,
                long_vol_window=self._cfg.markov_long_vol,
                amihud_window=self._cfg.markov_amihud_window,
            )
        )

        # Per-instrument hard stop-loss levels
        self._stop_levels: dict[str, float] = {}

        # Signal persistence tracker: how many consecutive bars
        # each instrument's flow score has exceeded entry_threshold
        self._signal_streak: dict[str, int] = {}

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

        Weights sum to at most ``cfg.gross_exposure``.  Each value is
        a :class:`~decimal.Decimal` rounded to six decimal places.
        """
        cfg = self._cfg

        # Minimum bars required across all signal / regime computations
        min_bars = max(
            cfg.markov_long_vol + 1,
            cfg.markov_amihud_window + 1,
            cfg.vir_medium_window,
            cfg.vpin_window,
            cfg.impact_lookback + 1,
            cfg.amihud_long_window + 1,
            cfg.vol_lookback + 1,
            cfg.atr_period + 1,
            cfg.circuit_breaker_lookback + 1,
        )

        # --- update rolling equity estimate -----------------------------
        current_prices: dict[str, float] = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                current_prices[inst_id] = float(bars[-1].close)
        self._update_equity(current_prices)

        # --- portfolio-level drawdown scaling ----------------------------
        dd_scale = self._drawdown_scale()
        if dd_scale <= 0:
            self._stop_levels.clear()
            self._prev_weights = {}
            self._signal_streak.clear()
            return {}

        # --- per-instrument crash circuit breaker ------------------------
        for bars in bars_by_instrument.values():
            if len(bars) >= cfg.circuit_breaker_lookback + 1:
                closes = [float(b.close) for b in bars]
                recent = closes[-cfg.circuit_breaker_lookback - 1]
                if recent > 0:
                    ret = closes[-1] / recent - 1.0
                    if ret < -cfg.circuit_breaker_drop:
                        self._stop_levels.clear()
                        self._prev_weights = {}
                        self._signal_streak.clear()
                        return {}

        # --- classify aggregate regime via Markov detector ---------------
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
            self._signal_streak.clear()
            return {}

        # ATR stop-loss multiplier per regime
        stop_mult_map: dict[MarkovRegimeState, float] = {
            MarkovRegimeState.LOW_VOL_HIGH_LIQ: cfg.stop_loss_atr_calm,
            MarkovRegimeState.MID_VOL_MID_LIQ: cfg.stop_loss_atr_normal,
            MarkovRegimeState.HIGH_VOL_LOW_LIQ: (cfg.stop_loss_atr_stressed),
            MarkovRegimeState.CRISIS: cfg.stop_loss_atr_stressed,
        }
        stop_mult = stop_mult_map.get(
            regime_state,
            cfg.stop_loss_atr_normal,
        )

        # --- score each instrument --------------------------------------
        scored: list[tuple[str, float, float]] = []

        active_instruments: set[str] = set()

        for inst_id, bars in bars_by_instrument.items():
            if len(bars) < min_bars:
                continue
            price = float(bars[-1].close)
            if price <= 0:
                continue

            # Hard stop-loss check
            if inst_id in self._stop_levels and price < self._stop_levels[inst_id]:
                del self._stop_levels[inst_id]
                self._signal_streak.pop(inst_id, None)
                continue  # stopped out

            funding = funding_by_instrument.get(inst_id, [])
            flow_score = self._composite_flow_score(bars, funding)

            if flow_score is None:
                self._signal_streak.pop(inst_id, None)
                continue

            # VPIN danger gate: don't enter new positions when toxic
            vpin = _vpin_proxy(bars, cfg.vpin_window)
            is_held = inst_id in self._prev_weights

            if vpin is not None and vpin > cfg.vpin_danger_threshold and not is_held:
                self._signal_streak.pop(inst_id, None)
                continue

            # Update signal persistence streak
            if flow_score >= cfg.entry_threshold:
                self._signal_streak[inst_id] = self._signal_streak.get(inst_id, 0) + 1
            else:
                self._signal_streak[inst_id] = 0

            # Entry / exit logic
            if flow_score >= cfg.entry_threshold:
                streak = self._signal_streak.get(inst_id, 0)
                if streak < cfg.min_signal_bars and not is_held:
                    # Signal not yet persistent enough for new entry
                    continue
            elif is_held and flow_score < cfg.exit_threshold:
                # Exit existing position
                self._stop_levels.pop(inst_id, None)
                self._signal_streak.pop(inst_id, None)
                continue
            elif not is_held:
                # Below entry threshold and not held — skip
                continue

            # Set / update stop-loss
            atr = _compute_atr(bars, cfg.atr_period)
            if atr > 0:
                self._stop_levels[inst_id] = price - stop_mult * atr

            # Vol-targeting scalar
            daily_vol = _realised_vol(bars, cfg.vol_lookback)
            ann_vol = (
                daily_vol * math.sqrt(cfg.annualisation_factor) if daily_vol > 0 else cfg.vol_target
            )
            vol_scalar = cfg.vol_target / ann_vol if ann_vol > _EPSILON else 1.0
            vol_scalar *= regime_exposure

            active_instruments.add(inst_id)
            scored.append((inst_id, flow_score, vol_scalar))

        # Clean up streaks for instruments no longer in universe
        stale = [k for k in self._signal_streak if k not in bars_by_instrument]
        for k in stale:
            del self._signal_streak[k]

        if not scored:
            self._prev_weights = {}
            return {}

        # --- portfolio construction: top-N by flow score ----------------
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[: cfg.top_n]

        raw: dict[str, float] = {inst_id: vs for inst_id, _, vs in selected}
        total = sum(raw.values())
        if total > cfg.gross_exposure:
            scale = cfg.gross_exposure / total
            raw = {k: v * scale for k, v in raw.items()}

        # Apply drawdown scaling and capping
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

    def _composite_flow_score(
        self,
        bars: list[OhlcvBar],
        funding: list[FundingRate],
    ) -> float | None:
        """Compute the weighted composite flow score.

        Combines five sub-signals:
        - VIR acceleration (volume imbalance momentum)
        - VPIN proxy (negated — low toxicity is good)
        - Funding rate spillover (contrarian + momentum)
        - Price impact asymmetry
        - Amihud liquidity improvement

        Returns ``None`` when any critical sub-signal cannot be computed.
        """
        cfg = self._cfg

        # 1. VIR acceleration
        vir_accel = _vir_acceleration(bars, cfg)
        if vir_accel is None:
            return None

        # 2. VPIN proxy (negated: lower toxicity → higher score)
        vpin = _vpin_proxy(bars, cfg.vpin_window)
        if vpin is None:
            return None
        vpin_score = -vpin  # in [-1, 0] range typically

        # 3. Funding rate spillover
        contrarian, f_momentum = _funding_signal(
            funding,
            cfg.funding_short_avg,
            cfg.funding_long_avg,
        )
        if contrarian is not None and f_momentum is not None:
            # Blend contrarian level and momentum (equal parts)
            funding_sig = 0.5 * contrarian + 0.5 * f_momentum
        else:
            # No funding data — use zero (neutral)
            funding_sig = 0.0

        # 4. Price impact asymmetry
        impact = _price_impact_asymmetry(bars, cfg.impact_lookback)
        if impact is not None:
            # Centre around 1.0 (neutral): >0 bullish, <0 bearish
            impact_score = impact - 1.0
        else:
            impact_score = 0.0

        # 5. Amihud liquidity improvement
        liq = _liquidity_improvement(
            bars,
            cfg.amihud_short_window,
            cfg.amihud_long_window,
        )
        if liq is not None:
            liq_score = liq
        else:
            liq_score = 0.0

        # Weighted composite
        w_total = cfg.w_vir + cfg.w_vpin + cfg.w_funding + cfg.w_impact + cfg.w_liquidity
        if w_total <= 0:
            return None

        composite = (
            cfg.w_vir * vir_accel
            + cfg.w_vpin * vpin_score
            + cfg.w_funding * funding_sig
            + cfg.w_impact * impact_score
            + cfg.w_liquidity * liq_score
        ) / w_total

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
# Allocator factory
# ---------------------------------------------------------------------------


def create_microstructure_flow_allocator(
    config: MicrostructureFlowConfig | None = None,
) -> Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]:
    """Factory producing an allocator callable for :class:`BacktestEngine`.

    Unlike the other factory functions in this package that expose every
    parameter as a ``float`` for perturbation, this factory accepts an
    optional :class:`MicrostructureFlowConfig` directly.  When ``None``
    is passed, sensible defaults are used.

    Returns
    -------
    Callable
        A closure with signature
        ``(bars_by_instrument, funding_by_instrument) -> dict[str, Decimal]``
        compatible with the backtest engine's allocator protocol.
    """
    strategy = MicrostructureFlowAlpha(config)

    def _allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(
            bars_by_instrument,
            funding_by_instrument,
        )

    return _allocator
