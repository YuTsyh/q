"""Cross-Sectional Statistical Arbitrage — Relative Value with Dynamic Clustering.

A pure cross-sectional strategy that ranks **all** instruments against each
other using relative signals, then longs the top-decile assets while
avoiding the bottom-decile.  Because OKX Spot Demo (tdMode=cash) does not
permit shorting, the portfolio is long-only; the statistical-arbitrage
edge derives from *relative* mispricing rather than absolute price levels.

Unlike every other strategy in this repository—which scores each
instrument *independently*—this strategy:

1. **Cross-Sectional Return Decomposition**: Decomposes each instrument's
   return into market-beta and alpha (residual) components via rolling
   OLS against a volume-weighted market return.  The alpha residual is
   the primary signal.

2. **Multi-Factor Rank-Based Scoring**: Six orthogonal factors—relative
   momentum, relative volume surprise, mean-reversion residual, liquidity
   premium, funding-rate carry, and volatility discount—are each ranked
   across the full cross-section.  The composite score is the weighted
   sum of normalised ranks.

3. **Beta-Hedged Long-Only Construction**: Instruments with high market
   beta receive proportionally lower weights so the portfolio's aggregate
   beta stays below a configurable ceiling.

4. **Markov Regime Gating**: The Bayesian HMM
   (:class:`MarkovRegimeDetector`) controls the number of instruments
   selected: full breadth in calm markets, flight-to-quality in stress,
   and flat in crisis.

5. **Cross-Sectional Dispersion Filter**: If the factor scores are too
   clustered (low dispersion), no positions are taken—there is
   insufficient differentiation for a relative-value strategy.

6. **Risk Controls**: Per-instrument weight cap, Herfindahl concentration
   check, ATR stop-losses, portfolio drawdown circuit breaker, and
   minimum-rank-change turnover filter.

References:
- Fama, E. F. & French, K. R. (1993) "Common risk factors in the
  returns on stocks and bonds", *Journal of Financial Economics* 33(1).
- Jegadeesh, N. & Titman, S. (1993) "Returns to Buying Winners and
  Selling Losers", *Journal of Finance* 48(1).
- Amihud, Y. (2002) "Illiquidity and stock returns",
  *Journal of Financial Markets* 5(1).
- Lo, A. W. & MacKinlay, A. C. (1990) "When Are Contrarian Profits
  Due to Stock Market Overreaction?", *Review of Financial Studies* 3(2).
- Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006) "Pairs
  Trading: Performance of a Relative-Value Arbitrage Rule",
  *Review of Financial Studies* 19(3).
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
_MAX_VOL_SCALAR: float = 2.0  # cap vol-target multiplier to avoid leverage
_BETA_HEDGE_MAX_ITER: int = 20  # max iterations for beta-hedge convergence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSectionalArbConfig:
    """Tunable parameters for the cross-sectional stat-arb strategy.

    All windows are measured in *bars* (typically daily candles).
    Factor weights should sum to 1.0 for interpretability, but the
    composite score is computed as a weighted sum of rank-normalised
    values in [0, 1] regardless.
    """

    # -- Factor lookbacks -------------------------------------------------
    momentum_lookback: int = 10
    volume_avg_lookback: int = 20
    residual_lookback: int = 30
    amihud_lookback: int = 20
    funding_lookback: int = 6
    vol_lookback: int = 20

    # -- Factor weights (sum to 1) ----------------------------------------
    w_momentum: float = 0.25
    w_vol_surprise: float = 0.15
    w_mean_reversion: float = 0.20
    w_liquidity: float = 0.15
    w_carry: float = 0.15
    w_vol_discount: float = 0.10

    # -- Portfolio construction -------------------------------------------
    top_n: int = 3
    min_instruments: int = 4
    max_position_weight: float = 0.35
    min_position_weight: float = 0.01
    gross_exposure: float = 1.0
    score_dispersion_min: float = 0.05

    # -- Beta hedging -----------------------------------------------------
    beta_lookback: int = 30
    beta_hedge_enabled: bool = True
    max_portfolio_beta: float = 0.5

    # -- Markov regime detector params ------------------------------------
    markov_short_vol: int = 10
    markov_long_vol: int = 40
    markov_amihud_window: int = 20

    # -- Regime exposure --------------------------------------------------
    calm_top_n: int = 3
    normal_top_n: int = 2
    stressed_top_n: int = 1

    # -- Position sizing --------------------------------------------------
    vol_target: float = 0.15
    annualisation_factor: float = 365.0

    # -- Risk management --------------------------------------------------
    atr_period: int = 14
    stop_loss_atr_multiplier: float = 2.0
    dd_threshold: float = 0.05
    dd_window: int = 60
    circuit_breaker_lookback: int = 5
    circuit_breaker_drop: float = 0.05

    # -- Turnover control -------------------------------------------------
    min_rank_change: int = 2


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _log_returns(bars: list[OhlcvBar], lookback: int) -> list[float]:
    """Compute simple returns over the trailing *lookback* bars.

    Returns a list of length ``min(lookback, len(bars) - 1)``.
    """
    n = min(lookback, len(bars) - 1)
    if n <= 0:
        return []
    rets: list[float] = []
    for i in range(len(bars) - n, len(bars)):
        prev = float(bars[i - 1].close)
        curr = float(bars[i].close)
        if prev > _EPSILON:
            rets.append(curr / prev - 1.0)
        else:
            rets.append(0.0)
    return rets


def _mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Bessel-corrected sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _median(values: list[float]) -> float:
    """Simple median of a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


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
    """Realised daily volatility (Bessel-corrected)."""
    if len(bars) < lookback + 1:
        return 0.0
    rets: list[float] = []
    for i in range(-lookback, 0):
        prev = float(bars[i - 1].close)
        curr = float(bars[i].close)
        if prev > _EPSILON:
            rets.append(curr / prev - 1.0)
    return _std(rets)


def _volume_weighted_market_return(
    bars_by_inst: dict[str, list[OhlcvBar]],
) -> float:
    """Volume-weighted average of the most-recent bar return.

    This serves as the *market return* for cross-sectional
    decomposition.  Each instrument's return is weighted by its
    latest bar volume.
    """
    total_vw_ret = 0.0
    total_vol = 0.0
    for bars in bars_by_inst.values():
        if len(bars) < 2:
            continue
        prev_close = float(bars[-2].close)
        curr_close = float(bars[-1].close)
        vol = float(bars[-1].volume)
        if prev_close > _EPSILON and vol > _EPSILON:
            ret = curr_close / prev_close - 1.0
            total_vw_ret += ret * vol
            total_vol += vol
    if total_vol < _EPSILON:
        return 0.0
    return total_vw_ret / total_vol


def _rolling_ols_beta(
    inst_rets: list[float],
    mkt_rets: list[float],
) -> tuple[float, list[float]]:
    """Rolling OLS beta and residual series: r_i = alpha + beta * r_mkt.

    Returns ``(beta, residuals)``.  Falls back to (1.0, inst_rets)
    when OLS is degenerate.
    """
    n = min(len(inst_rets), len(mkt_rets))
    if n < 3:
        return 1.0, list(inst_rets)

    y = inst_rets[-n:]
    x = mkt_rets[-n:]

    x_mean = _mean(x)
    y_mean = _mean(y)

    cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
    var_x = sum((xi - x_mean) ** 2 for xi in x) / n

    if var_x < _EPSILON:
        return 1.0, list(y)

    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    residuals = [y[i] - alpha - beta * x[i] for i in range(n)]
    return beta, residuals


def _compute_market_returns_series(
    bars_by_inst: dict[str, list[OhlcvBar]],
    lookback: int,
) -> list[float]:
    """Build a market-return time-series of length *lookback*.

    At each bar offset ``t`` within the trailing window, compute the
    volume-weighted cross-sectional average return.
    """
    # Determine the common series length
    max_len = 0
    for bars in bars_by_inst.values():
        if len(bars) > max_len:
            max_len = len(bars)
    usable = min(lookback, max_len - 1)
    if usable <= 0:
        return []

    mkt_rets: list[float] = []
    for offset in range(usable, 0, -1):
        vw_ret = 0.0
        total_vol = 0.0
        for bars in bars_by_inst.values():
            idx = len(bars) - offset
            if idx < 1:
                continue
            prev_c = float(bars[idx - 1].close)
            curr_c = float(bars[idx].close)
            vol = float(bars[idx].volume)
            if prev_c > _EPSILON and vol > _EPSILON:
                ret = curr_c / prev_c - 1.0
                vw_ret += ret * vol
                total_vol += vol
        if total_vol > _EPSILON:
            mkt_rets.append(vw_ret / total_vol)
        else:
            mkt_rets.append(0.0)
    return mkt_rets


def _instrument_returns_series(
    bars: list[OhlcvBar],
    lookback: int,
) -> list[float]:
    """Simple return series for a single instrument over *lookback*."""
    usable = min(lookback, len(bars) - 1)
    if usable <= 0:
        return []
    rets: list[float] = []
    for i in range(len(bars) - usable, len(bars)):
        prev = float(bars[i - 1].close)
        curr = float(bars[i].close)
        if prev > _EPSILON:
            rets.append(curr / prev - 1.0)
        else:
            rets.append(0.0)
    return rets


# ---- Factor computation helpers -----------------------------------------


def _factor_relative_momentum(
    bars: list[OhlcvBar],
    mkt_rets: list[float],
    lookback: int,
) -> float | None:
    """Relative momentum: ``(r_i - r_market) / vol_i``."""
    inst_rets = _instrument_returns_series(bars, lookback)
    n = min(len(inst_rets), len(mkt_rets))
    if n < 2:
        return None
    inst_tail = inst_rets[-n:]
    mkt_tail = mkt_rets[-n:]
    excess = [inst_tail[i] - mkt_tail[i] for i in range(n)]
    vol_i = _std(inst_tail)
    if vol_i < _EPSILON:
        return None
    return _mean(excess) / vol_i


def _factor_volume_surprise(
    bars: list[OhlcvBar],
    avg_lookback: int,
) -> float | None:
    """Volume surprise: ``vol_today / avg_vol - 1``.

    The cross-sectional median is subtracted later in the ranking
    step.  Here we return the raw ratio.
    """
    if len(bars) < avg_lookback + 1:
        return None
    recent_vols = [float(bars[i].volume) for i in range(len(bars) - avg_lookback, len(bars))]
    avg_vol = _mean(recent_vols[:-1]) if len(recent_vols) > 1 else 0.0
    if avg_vol < _EPSILON:
        return None
    today_vol = recent_vols[-1]
    return today_vol / avg_vol


def _factor_mean_reversion_residual(
    inst_rets: list[float],
    mkt_rets: list[float],
    lookback: int,
) -> float | None:
    """Mean-reversion residual: normalised OLS residual."""
    n = min(len(inst_rets), len(mkt_rets), lookback)
    if n < 5:
        return None
    _, residuals = _rolling_ols_beta(inst_rets[-n:], mkt_rets[-n:])
    if len(residuals) < 3:
        return None
    res_vol = _std(residuals)
    if res_vol < _EPSILON:
        return None
    # Negative residual = underperformance → expect reversion up
    return -residuals[-1] / res_vol


def _factor_liquidity_premium(
    bars: list[OhlcvBar],
    lookback: int,
) -> float | None:
    """Amihud illiquidity (negated so higher = more liquid)."""
    if len(bars) < lookback + 1:
        return None
    ratios: list[float] = []
    tail = bars[-(lookback + 1) :]
    for i in range(1, len(tail)):
        prev_c = float(tail[i - 1].close)
        curr_c = float(tail[i].close)
        vol = float(tail[i].volume)
        if prev_c > _EPSILON and vol > _EPSILON:
            ratios.append(abs(curr_c / prev_c - 1.0) / vol)
    if not ratios:
        return None
    return -_mean(ratios)


def _factor_funding_carry(
    funding: list[FundingRate],
    lookback: int,
) -> float | None:
    """Funding-rate carry: ``-avg(funding_rate)`` — long negative carry."""
    if not funding:
        return 0.0  # no funding data → neutral
    recent = funding[-lookback:] if len(funding) >= lookback else funding
    avg_fr = _mean([float(f.funding_rate) for f in recent])
    return -avg_fr


def _factor_vol_discount(
    bars: list[OhlcvBar],
    lookback: int,
) -> float | None:
    """Volatility discount: ``1 / realised_vol``."""
    rv = _realised_vol(bars, lookback)
    if rv < _EPSILON:
        return None
    return 1.0 / rv


# ---- Rank-normalisation -------------------------------------------------


def _rank_normalise(
    raw_scores: dict[str, float],
) -> dict[str, float]:
    """Rank-normalise *raw_scores* to [0, 1] (descending: best → 1.0).

    Ties receive the average rank.
    """
    if not raw_scores:
        return {}
    n = len(raw_scores)
    if n == 1:
        return {k: 0.5 for k in raw_scores}

    sorted_items = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_items[j + 1][1] == sorted_items[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        norm = 1.0 - avg_rank / max(n - 1, 1)
        for k in range(i, j + 1):
            ranked[sorted_items[k][0]] = norm
        i = j + 1
    return ranked


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class CrossSectionalArbAlpha:
    """Cross-sectional statistical-arbitrage alpha model.

    Call :meth:`allocate` on each rebalance bar to obtain target weights.
    The strategy is **stateful**: the internal Markov detector, rolling
    beta estimates, previous ranks, and stop-loss map persist across
    calls.

    This is the *only* pure cross-sectional strategy in the repository.
    All other strategies score each instrument in isolation; this one
    ranks instruments *against each other* and derives edge from
    relative mispricing.
    """

    def __init__(
        self,
        config: CrossSectionalArbConfig | None = None,
    ) -> None:
        self._cfg = config or CrossSectionalArbConfig()

        # Markov detector (stateful – belief vector persists)
        self._detector = MarkovRegimeDetector(
            MarkovRegimeConfig(
                short_vol_window=self._cfg.markov_short_vol,
                long_vol_window=self._cfg.markov_long_vol,
                amihud_window=self._cfg.markov_amihud_window,
            )
        )

        # Per-instrument hard stop-loss levels
        self._stop_levels: dict[str, float] = {}

        # Previous composite ranks for turnover control
        self._prev_ranks: dict[str, int] = {}

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
        """Return target portfolio weights (spot-long only, ≥ 0).

        The returned dict maps instrument ids to non-negative
        :class:`Decimal` weights that sum to at most
        ``gross_exposure``.  Instruments not present in the output
        should be treated as zero-weight.
        """
        cfg = self._cfg

        # Minimum bars required for all computations
        min_bars = max(
            cfg.markov_long_vol + 1,
            cfg.markov_amihud_window + 1,
            cfg.residual_lookback + 1,
            cfg.beta_lookback + 1,
            cfg.volume_avg_lookback + 1,
            cfg.amihud_lookback + 1,
            cfg.vol_lookback + 1,
            cfg.atr_period + 1,
            cfg.momentum_lookback + 1,
            cfg.dd_window,
        )

        # --- update rolling equity estimate -----------------------------
        current_prices: dict[str, float] = {}
        for inst_id, bars in bars_by_instrument.items():
            if bars:
                current_prices[inst_id] = float(bars[-1].close)
        self._update_equity(current_prices)

        # --- portfolio-level drawdown scaling ---------------------------
        dd_scale = self._drawdown_scale()
        if dd_scale <= 0:
            return self._go_flat()

        # --- per-instrument crash circuit breaker -----------------------
        for bars in bars_by_instrument.values():
            n_cb = cfg.circuit_breaker_lookback
            if len(bars) >= n_cb + 1:
                c_now = float(bars[-1].close)
                c_prev = float(bars[-n_cb - 1].close)
                if c_prev > _EPSILON:
                    ret = c_now / c_prev - 1.0
                    if ret < -cfg.circuit_breaker_drop:
                        return self._go_flat()

        # --- filter instruments with enough data ------------------------
        eligible: dict[str, list[OhlcvBar]] = {
            k: v
            for k, v in bars_by_instrument.items()
            if len(v) >= min_bars and float(v[-1].close) > _EPSILON
        }
        if len(eligible) < cfg.min_instruments:
            return self._go_flat()

        # --- Markov regime classification -------------------------------
        regime_result = self._detector.classify_portfolio(eligible)
        regime_state = regime_result.state

        top_n_by_regime: dict[MarkovRegimeState, int] = {
            MarkovRegimeState.LOW_VOL_HIGH_LIQ: cfg.calm_top_n,
            MarkovRegimeState.MID_VOL_MID_LIQ: cfg.normal_top_n,
            MarkovRegimeState.HIGH_VOL_LOW_LIQ: cfg.stressed_top_n,
            MarkovRegimeState.CRISIS: 0,
        }
        active_top_n = top_n_by_regime.get(regime_state, 0)
        if active_top_n <= 0:
            return self._go_flat()

        # --- build market return series ---------------------------------
        ols_lookback = max(cfg.residual_lookback, cfg.beta_lookback)
        mkt_rets = _compute_market_returns_series(eligible, ols_lookback)
        if len(mkt_rets) < 5:
            return self._go_flat()

        # --- compute six factors for every eligible instrument ----------
        raw_momentum: dict[str, float] = {}
        raw_vol_surprise: dict[str, float] = {}
        raw_mean_rev: dict[str, float] = {}
        raw_liquidity: dict[str, float] = {}
        raw_carry: dict[str, float] = {}
        raw_vol_disc: dict[str, float] = {}
        inst_betas: dict[str, float] = {}

        for inst_id, bars in eligible.items():
            inst_rets = _instrument_returns_series(bars, ols_lookback)
            funding = funding_by_instrument.get(inst_id, [])

            # 1. Relative momentum
            mom = _factor_relative_momentum(bars, mkt_rets, cfg.momentum_lookback)
            if mom is not None:
                raw_momentum[inst_id] = mom

            # 2. Volume surprise (raw ratio)
            vs = _factor_volume_surprise(bars, cfg.volume_avg_lookback)
            if vs is not None:
                raw_vol_surprise[inst_id] = vs

            # 3. Mean-reversion residual
            mr = _factor_mean_reversion_residual(inst_rets, mkt_rets, cfg.residual_lookback)
            if mr is not None:
                raw_mean_rev[inst_id] = mr

            # 4. Liquidity premium
            liq = _factor_liquidity_premium(bars, cfg.amihud_lookback)
            if liq is not None:
                raw_liquidity[inst_id] = liq

            # 5. Funding-rate carry
            carry = _factor_funding_carry(funding, cfg.funding_lookback)
            if carry is not None:
                raw_carry[inst_id] = carry

            # 6. Volatility discount
            vd = _factor_vol_discount(bars, cfg.vol_lookback)
            if vd is not None:
                raw_vol_disc[inst_id] = vd

            # Rolling beta (for hedging)
            n_beta = min(len(inst_rets), len(mkt_rets), cfg.beta_lookback)
            if n_beta >= 3:
                beta, _ = _rolling_ols_beta(inst_rets[-n_beta:], mkt_rets[-n_beta:])
                inst_betas[inst_id] = beta

        # Cross-sectional median-adjust volume surprise
        if raw_vol_surprise:
            med = _median(list(raw_vol_surprise.values()))
            raw_vol_surprise = {k: v - med for k, v in raw_vol_surprise.items()}

        # Cross-sectional mean-adjust liquidity (normalise by mean)
        if raw_liquidity:
            liq_mean = _mean(list(raw_liquidity.values()))
            if abs(liq_mean) > _EPSILON:
                raw_liquidity = {k: v / abs(liq_mean) for k, v in raw_liquidity.items()}

        # --- rank-normalise each factor ---------------------------------
        ranked_mom = _rank_normalise(raw_momentum)
        ranked_vs = _rank_normalise(raw_vol_surprise)
        ranked_mr = _rank_normalise(raw_mean_rev)
        ranked_liq = _rank_normalise(raw_liquidity)
        ranked_carry = _rank_normalise(raw_carry)
        ranked_vd = _rank_normalise(raw_vol_disc)

        # --- composite score = weighted sum of ranks --------------------
        composite: dict[str, float] = {}
        for inst_id in eligible:
            score = 0.0
            score += cfg.w_momentum * ranked_mom.get(inst_id, 0.5)
            score += cfg.w_vol_surprise * ranked_vs.get(inst_id, 0.5)
            score += cfg.w_mean_reversion * ranked_mr.get(inst_id, 0.5)
            score += cfg.w_liquidity * ranked_liq.get(inst_id, 0.5)
            score += cfg.w_carry * ranked_carry.get(inst_id, 0.5)
            score += cfg.w_vol_discount * ranked_vd.get(inst_id, 0.5)
            composite[inst_id] = score

        # --- cross-sectional dispersion filter --------------------------
        scores_list = list(composite.values())
        if len(scores_list) >= 2:
            dispersion = _std(scores_list)
            if dispersion < cfg.score_dispersion_min:
                return self._go_flat()

        # --- apply stop-losses (remove stopped-out instruments) ---------
        for inst_id in list(composite.keys()):
            bars = eligible[inst_id]
            price = float(bars[-1].close)
            if inst_id in self._stop_levels and price < self._stop_levels[inst_id]:
                del self._stop_levels[inst_id]
                del composite[inst_id]

        if not composite:
            return self._go_flat()

        # --- select top-N by composite score ----------------------------
        sorted_instruments = sorted(composite.items(), key=lambda kv: kv[1], reverse=True)

        # Build current rank map (1-based)
        current_ranks: dict[str, int] = {
            inst_id: rank + 1 for rank, (inst_id, _) in enumerate(sorted_instruments)
        }

        # Turnover control: keep previous selection if rank changes
        # are all below min_rank_change threshold
        if self._prev_ranks and self._prev_weights:
            max_change = 0
            prev_selected = set(self._prev_weights.keys())
            new_candidates = {inst_id for inst_id, _ in sorted_instruments[:active_top_n]}
            all_involved = prev_selected | new_candidates
            for inst_id in all_involved:
                old_r = self._prev_ranks.get(inst_id, len(sorted_instruments))
                new_r = current_ranks.get(inst_id, len(sorted_instruments))
                max_change = max(max_change, abs(old_r - new_r))
            if max_change < cfg.min_rank_change:
                # Maintain previous weights (no rebalance)
                self._prev_ranks = current_ranks
                return {
                    k: Decimal(str(round(v, 6)))
                    for k, v in self._prev_weights.items()
                    if v >= cfg.min_position_weight
                }

        self._prev_ranks = current_ranks

        selected = sorted_instruments[:active_top_n]
        if not selected:
            return self._go_flat()

        # --- weight by score magnitude ----------------------------------
        total_score = sum(s for _, s in selected)
        if total_score < _EPSILON:
            return self._go_flat()

        raw_weights: dict[str, float] = {
            inst_id: score / total_score for inst_id, score in selected
        }

        # --- beta-hedge: reduce weight of high-beta instruments ---------
        if cfg.beta_hedge_enabled and inst_betas:
            raw_weights = _apply_beta_hedge(raw_weights, inst_betas, cfg.max_portfolio_beta)

        # --- vol-target scaling -----------------------------------------
        for inst_id in list(raw_weights.keys()):
            bars = eligible.get(inst_id)
            if bars is None:
                continue
            daily_vol = _realised_vol(bars, cfg.vol_lookback)
            ann_vol = daily_vol * math.sqrt(cfg.annualisation_factor)
            if ann_vol > _EPSILON:
                vol_scalar = cfg.vol_target / ann_vol
                raw_weights[inst_id] *= min(vol_scalar, _MAX_VOL_SCALAR)

        # --- apply per-instrument weight cap ----------------------------
        for inst_id in raw_weights:
            raw_weights[inst_id] = min(raw_weights[inst_id], cfg.max_position_weight)

        # --- Herfindahl concentration check ----------------------------
        raw_weights = _herfindahl_adjust(raw_weights)

        # --- normalise to gross_exposure --------------------------------
        total_w = sum(raw_weights.values())
        if total_w > cfg.gross_exposure and total_w > _EPSILON:
            scale = cfg.gross_exposure / total_w
            raw_weights = {k: v * scale for k, v in raw_weights.items()}

        # --- apply drawdown scale ---------------------------------------
        raw_weights = {k: v * dd_scale for k, v in raw_weights.items()}

        # --- re-cap after drawdown scaling ------------------------------
        for inst_id in raw_weights:
            raw_weights[inst_id] = min(raw_weights[inst_id], cfg.max_position_weight)

        # --- set stop-losses for selected instruments -------------------
        for inst_id in raw_weights:
            bars = eligible.get(inst_id)
            if bars is None:
                continue
            atr = _compute_atr(bars, cfg.atr_period)
            price = float(bars[-1].close)
            if atr > 0:
                self._stop_levels[inst_id] = price - cfg.stop_loss_atr_multiplier * atr

        # --- prune tiny weights & build final output --------------------
        weights: dict[str, Decimal] = {}
        for inst_id, w in raw_weights.items():
            if w >= cfg.min_position_weight:
                weights[inst_id] = Decimal(str(round(w, 6)))

        self._prev_weights = {k: float(v) for k, v in weights.items()}

        # Clean up stop levels for instruments no longer held
        held = set(weights.keys())
        for inst_id in list(self._stop_levels.keys()):
            if inst_id not in held:
                del self._stop_levels[inst_id]

        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _go_flat(self) -> dict[str, Decimal]:
        """Return an empty allocation and reset stateful accumulators."""
        self._stop_levels.clear()
        self._prev_weights = {}
        return {}

    def _update_equity(self, current_prices: dict[str, float]) -> None:
        """Mark-to-market the internal equity tracker."""
        if self._prev_weights and self._prev_prices:
            port_ret = 0.0
            for inst_id, w in self._prev_weights.items():
                if inst_id in current_prices and inst_id in self._prev_prices:
                    prev_p = self._prev_prices[inst_id]
                    curr_p = current_prices[inst_id]
                    if prev_p > _EPSILON:
                        port_ret += w * (curr_p / prev_p - 1.0)
            self._equity_hist.append(self._equity_hist[-1] * (1.0 + port_ret))
        elif len(self._equity_hist) > 1:
            self._equity_hist.append(self._equity_hist[-1])
        # Trim to window
        if len(self._equity_hist) > self._cfg.dd_window:
            self._equity_hist = self._equity_hist[-self._cfg.dd_window :]
        self._prev_prices = current_prices

    def _drawdown_scale(self) -> float:
        """Exposure multiplier in [0, 1] based on rolling drawdown."""
        if len(self._equity_hist) < 2:
            return 1.0
        peak = max(self._equity_hist)
        current = self._equity_hist[-1]
        dd = 1.0 - current / peak if peak > _EPSILON else 0.0
        limit = self._cfg.dd_threshold
        if dd <= limit:
            return 1.0
        return max(0.0, 1.0 - (dd - limit) / limit)


# ---------------------------------------------------------------------------
# Module-level portfolio construction helpers
# ---------------------------------------------------------------------------


def _apply_beta_hedge(
    weights: dict[str, float],
    betas: dict[str, float],
    max_beta: float,
) -> dict[str, float]:
    """Reduce weights of high-beta instruments to keep portfolio beta low.

    Iteratively scales down the weight of the highest-beta instrument
    until the portfolio beta is ≤ *max_beta* or no further adjustment
    is possible.
    """
    result = dict(weights)
    for _ in range(_BETA_HEDGE_MAX_ITER):
        port_beta = sum(result.get(k, 0.0) * betas.get(k, 1.0) for k in result)
        total_w = sum(result.values())
        if total_w < _EPSILON:
            break
        avg_beta = port_beta / total_w
        if avg_beta <= max_beta:
            break
        # Find the highest-beta instrument in the portfolio
        worst = max(
            result,
            key=lambda k: betas.get(k, 1.0),
        )
        worst_beta = betas.get(worst, 1.0)
        if worst_beta < _EPSILON:
            break
        # Scale down this instrument's weight
        shrink = 0.8
        result[worst] *= shrink
    return result


def _herfindahl_adjust(
    weights: dict[str, float],
) -> dict[str, float]:
    """Redistribute weights if Herfindahl index indicates over-concentration.

    The Herfindahl-Hirschman Index (HHI) = Σ w_i².
    For N positions, a fully equal-weight portfolio has HHI = 1/N.
    If HHI > 2/N (too concentrated), blend weights towards equal-weight.
    """
    n = len(weights)
    if n <= 1:
        return dict(weights)

    total = sum(weights.values())
    if total < _EPSILON:
        return dict(weights)

    normed = {k: v / total for k, v in weights.items()}
    hhi = sum(w * w for w in normed.values())
    threshold = 2.0 / n  # twice the equal-weight HHI

    if hhi <= threshold:
        return dict(weights)

    # Blend towards equal-weight (50/50 blend as a single step)
    eq_w = 1.0 / n
    blended = {k: (normed[k] + eq_w) / 2.0 for k in normed}
    # Re-scale to original total
    blend_total = sum(blended.values())
    if blend_total < _EPSILON:
        return dict(weights)
    return {k: v / blend_total * total for k, v in blended.items()}


# ---------------------------------------------------------------------------
# Allocator factory (compatible with BacktestEngine)
# ---------------------------------------------------------------------------


def create_cross_sectional_arb_allocator(
    config: CrossSectionalArbConfig | None = None,
) -> Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]:
    """Factory producing an allocator callable for :class:`BacktestEngine`.

    Args:
        config: Strategy configuration.  Defaults to
            :class:`CrossSectionalArbConfig` if *None*.

    Returns:
        A callable with signature::

            (bars_by_instrument, funding_by_instrument) -> weights

        where *weights* is a ``dict[str, Decimal]`` of non-negative
        position weights summing to at most ``gross_exposure``.
    """
    strategy = CrossSectionalArbAlpha(config)

    def _allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return _allocator
