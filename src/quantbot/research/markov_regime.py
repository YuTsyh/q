"""Markov-Switching regime detection for crypto markets.

Implements a discrete Hidden Markov Model (HMM) with four market regimes,
using volatility ratio and Amihud illiquidity as observable signals.  The
detector maintains a Bayesian state-belief vector that is updated each time
:meth:`classify` is called, providing temporal persistence across
successive observations.

States:
- LOW_VOL_HIGH_LIQ: Calm / trending — tight spreads, low realised vol.
- MID_VOL_MID_LIQ: Normal — typical market conditions.
- HIGH_VOL_LOW_LIQ: Stressed — elevated vol, widening spreads, thinning books.
- CRISIS: Extreme — vol spike with severe liquidity withdrawal.

Key signals:
- **Vol-ratio**: short-term realised volatility divided by long-term
  realised volatility.  Values near 1.0 indicate stable vol; values >> 1
  indicate a vol expansion (Hamilton 1989).
- **Amihud illiquidity**: mean(|return| / volume) over a rolling window.
  Higher values indicate worse liquidity (Amihud 2002).

References:
- Amihud, Y. (2002) "Illiquidity and stock returns: cross-section and
  time-series effects", *Journal of Financial Markets* 5(1), 31–56.
- Hamilton, J. D. (1989) "A new approach to the economic analysis of
  nonstationary time series and the business cycle", *Econometrica*
  57(2), 357–384.
- Ang, A. & Bekaert, G. (2002) "Regime Switches in Interest Rates",
  *Journal of Business & Economic Statistics* 20(2), 163–182.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from quantbot.research.data import OhlcvBar


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# Guard against division by zero in ratio computations (volatility, weights).
_EPSILON: float = 1e-12
# Stricter floor for probability normalisation to avoid amplifying noise.
_PROB_FLOOR: float = 1e-30

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class MarkovRegimeState(Enum):
    """Discrete market-regime states for the Markov-Switching model."""

    LOW_VOL_HIGH_LIQ = "low_vol_high_liq"
    MID_VOL_MID_LIQ = "mid_vol_mid_liq"
    HIGH_VOL_LOW_LIQ = "high_vol_low_liq"
    CRISIS = "crisis"


# Ordered list matching matrix row/column indices.
_STATES: list[MarkovRegimeState] = list(MarkovRegimeState)
_NUM_STATES: int = len(_STATES)


# ---------------------------------------------------------------------------
# Amihud illiquidity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmihudIlliquidity:
    """Computes the Amihud (2002) illiquidity measure.

    Definition:  ILLIQ = (1/N) * Σ |r_t| / V_t

    where r_t is the log-return and V_t is the bar volume.  Higher values
    indicate worse (more illiquid) conditions.

    Args:
        window: Number of trailing bars used in the rolling mean.
    """

    window: int = 20

    def compute(self, bars: list[OhlcvBar]) -> float:
        """Return the Amihud illiquidity ratio for *bars*.

        Args:
            bars: OHLCV bars (most recent last), must have at least
                  ``window + 1`` elements.

        Returns:
            The mean |return| / volume over the window.  Returns ``0.0``
            when there is insufficient data or all volumes are zero.
        """
        return compute_amihud(bars, self.window)


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkovRegimeConfig:
    """Configuration for the Markov-Switching regime detector.

    All window sizes are in *number of bars*.  Threshold values are
    dimensionless ratios or quantile cut-offs.
    """

    short_vol_window: int = 10
    long_vol_window: int = 40
    amihud_window: int = 20
    amihud_crisis_threshold: float = 0.8  # quantile cutoff for crisis
    vol_low_threshold: float = 0.8  # vol_ratio below => calm
    vol_high_threshold: float = 1.5  # vol_ratio above => stressed
    vol_crisis_threshold: float = 2.5  # vol_ratio above => crisis
    transition_smoothing: float = 0.3  # exponential smoothing for priors


@dataclass(frozen=True)
class MarkovRegimeResult:
    """Output of a single regime classification step.

    Attributes:
        state: Maximum a-posteriori (MAP) regime state.
        vol_ratio: Short-term / long-term realised-vol ratio.
        amihud_score: Normalised Amihud illiquidity in [0, 1].
        state_probabilities: Posterior probability of each state.
        confidence: Posterior probability of the MAP state.
    """

    state: MarkovRegimeState
    vol_ratio: float
    amihud_score: float  # normalised 0-1
    state_probabilities: dict[str, float]
    confidence: float


# ---------------------------------------------------------------------------
# Emission-probability parameters (centre, spread) per state
# ---------------------------------------------------------------------------

# Each state is characterised by (vol_ratio_centre, amihud_centre).
_EMISSION_PARAMS: dict[MarkovRegimeState, tuple[float, float]] = {
    MarkovRegimeState.LOW_VOL_HIGH_LIQ: (0.7, 0.2),
    MarkovRegimeState.MID_VOL_MID_LIQ: (1.0, 0.5),
    MarkovRegimeState.HIGH_VOL_LOW_LIQ: (1.8, 0.7),
    MarkovRegimeState.CRISIS: (3.0, 0.9),
}

# Spread (σ) for the Gaussian kernel per observable.
_VOL_SIGMA: float = 0.5
_AMIHUD_SIGMA: float = 0.25


# ---------------------------------------------------------------------------
# Pre-calibrated transition matrix  (row = from-state, col = to-state)
# ---------------------------------------------------------------------------

# Crypto markets exhibit high state persistence on the diagonal (≈ 0.85-0.95).
# Crisis tends to exit faster (lower self-transition) because extreme vol
# events are short-lived relative to calm / normal regimes.
_TRANSITION_MATRIX: list[list[float]] = [
    # LOW_VOL  MID_VOL  HIGH_VOL  CRISIS
    [0.92, 0.06, 0.015, 0.005],  # from LOW_VOL_HIGH_LIQ
    [0.08, 0.85, 0.05, 0.02],  # from MID_VOL_MID_LIQ
    [0.02, 0.08, 0.85, 0.05],  # from HIGH_VOL_LOW_LIQ
    [0.03, 0.10, 0.15, 0.72],  # from CRISIS (exits faster)
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_amihud(bars: list[OhlcvBar], window: int) -> float:
    """Compute the Amihud (2002) illiquidity ratio.

    Args:
        bars: OHLCV bars (most recent last).
        window: Number of trailing bars for the rolling mean.

    Returns:
        Mean of |return_t| / volume_t over the window.  Returns ``0.0``
        when there is insufficient data.
    """
    if len(bars) < window + 1 or window < 1:
        return 0.0

    tail = bars[-(window + 1) :]
    ratios: list[float] = []
    for i in range(1, len(tail)):
        prev_close = float(tail[i - 1].close)
        cur_close = float(tail[i].close)
        volume = float(tail[i].volume)
        if prev_close <= 0 or volume <= 0:
            continue
        abs_return = abs(cur_close / prev_close - 1.0)
        ratios.append(abs_return / volume)

    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _compute_vol_ratio(
    bars: list[OhlcvBar],
    short_window: int,
    long_window: int,
) -> float:
    """Compute the ratio of short-term to long-term realised volatility.

    Args:
        bars: OHLCV bars (most recent last).
        short_window: Lookback for short-term vol.
        long_window: Lookback for long-term vol.

    Returns:
        short_vol / long_vol.  Defaults to ``1.0`` when long-term vol
        is negligible or when there is insufficient data.
    """
    # We need long_window + 1 closes to compute long_window returns.
    if len(bars) < long_window + 1:
        return 1.0

    closes = [float(b.close) for b in bars]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]

    if len(returns) < long_window:
        return 1.0

    short_vol = _std(returns[-short_window:])
    long_vol = _std(returns[-long_window:])
    return short_vol / long_vol if long_vol > _EPSILON else 1.0


def _emission_probability(
    vol_ratio: float,
    amihud_score: float,
    state: MarkovRegimeState,
) -> float:
    """Compute the emission (observation) probability for *state*.

    Uses an un-normalised Gaussian kernel centred on the characteristic
    (vol_ratio, amihud_score) of the state.  This approximates the
    true emission density and allows the Bayesian update to weight
    states that better match the current observations.

    Args:
        vol_ratio: Current short/long vol ratio.
        amihud_score: Normalised Amihud illiquidity in [0, 1].
        state: The regime state being evaluated.

    Returns:
        Un-normalised likelihood ∈ (0, 1].
    """
    vol_centre, amihud_centre = _EMISSION_PARAMS[state]
    vol_term = ((vol_ratio - vol_centre) / _VOL_SIGMA) ** 2
    amihud_term = ((amihud_score - amihud_centre) / _AMIHUD_SIGMA) ** 2
    return math.exp(-0.5 * (vol_term + amihud_term))


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _normalise(probs: list[float]) -> list[float]:
    """Normalise a probability vector to sum to 1.

    Falls back to a uniform distribution when the total is negligible.
    """
    total = sum(probs)
    if total < _PROB_FLOOR:
        return [1.0 / _NUM_STATES] * _NUM_STATES
    return [p / total for p in probs]


def _normalise_amihud(raw: float, bars: list[OhlcvBar], window: int) -> float:
    """Map a raw Amihud value to [0, 1] via rank within the trailing window.

    We recompute every trailing Amihud value in the window and return the
    empirical quantile of *raw*.  When the window is too short the raw value
    is clamped to [0, 1] using a simple sigmoid-like transform.
    """
    # Need at least window + 2 bars to build a single trailing Amihud series.
    if len(bars) < window + 2:
        return max(0.0, min(1.0, raw / (raw + _EPSILON)))

    # Collect trailing Amihud values for each sub-window.
    amihud_vals: list[float] = []
    for end in range(window + 1, len(bars) + 1):
        sub = bars[end - window - 1 : end]
        a = compute_amihud(sub, window)
        amihud_vals.append(a)

    if not amihud_vals:
        return 0.0

    rank = sum(1 for v in amihud_vals if v <= raw)
    return rank / len(amihud_vals)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class MarkovRegimeDetector:
    """Bayesian Markov-Switching regime detector.

    Maintains an internal state-belief vector (``_state_priors``) that is
    updated on every call to :meth:`classify`.  The belief vector decays
    towards the stationary distribution between observations via the
    transition matrix, providing temporal persistence.

    Example::

        detector = MarkovRegimeDetector()
        result = detector.classify(bars)
        print(result.state, result.confidence)

    Args:
        config: Model hyper-parameters.  Defaults to
            :class:`MarkovRegimeConfig` if *None*.
    """

    def __init__(self, config: MarkovRegimeConfig | None = None) -> None:
        self._config = config or MarkovRegimeConfig()
        self._transition: list[list[float]] = [row[:] for row in _TRANSITION_MATRIX]
        # Start with a uniform prior.
        self._state_priors: list[float] = [1.0 / _NUM_STATES] * _NUM_STATES

    # -- public API ---------------------------------------------------------

    def classify(self, bars: list[OhlcvBar]) -> MarkovRegimeResult:
        """Classify the current regime from OHLCV data.

        Performs a single Bayesian HMM forward step:

        1. Compute observables (vol-ratio, Amihud score).
        2. Compute emission likelihoods per state.
        3. Propagate prior through transition matrix.
        4. Bayesian update: posterior ∝ emission × predicted prior.
        5. Apply exponential smoothing for state persistence.

        Args:
            bars: OHLCV bars sorted chronologically (most recent last).

        Returns:
            :class:`MarkovRegimeResult` with the MAP state and full
            posterior distribution.

        Raises:
            ValueError: If there are fewer bars than required by the
                config windows.
        """
        cfg = self._config
        min_required = max(cfg.long_vol_window + 1, cfg.amihud_window + 1)
        if len(bars) < min_required:
            raise ValueError(f"Need at least {min_required} bars, got {len(bars)}")

        # --- observables ---------------------------------------------------
        vol_ratio = _compute_vol_ratio(bars, cfg.short_vol_window, cfg.long_vol_window)
        raw_amihud = compute_amihud(bars, cfg.amihud_window)
        amihud_score = _normalise_amihud(raw_amihud, bars, cfg.amihud_window)

        # --- emission likelihoods ------------------------------------------
        emissions = [_emission_probability(vol_ratio, amihud_score, s) for s in _STATES]

        # --- transition (predict step): prior_pred = T^T · prior -----------
        predicted: list[float] = [0.0] * _NUM_STATES
        for j in range(_NUM_STATES):
            for i in range(_NUM_STATES):
                predicted[j] += self._transition[i][j] * self._state_priors[i]

        # --- Bayesian update: posterior ∝ emission × predicted -------------
        raw_posterior = [emissions[j] * predicted[j] for j in range(_NUM_STATES)]
        posterior = _normalise(raw_posterior)

        # --- exponential smoothing for state persistence -------------------
        alpha = cfg.transition_smoothing
        smoothed = [
            alpha * posterior[j] + (1.0 - alpha) * self._state_priors[j] for j in range(_NUM_STATES)
        ]
        smoothed = _normalise(smoothed)

        # Update internal belief.
        self._state_priors = smoothed

        # --- MAP state -----------------------------------------------------
        map_idx = max(range(_NUM_STATES), key=lambda k: smoothed[k])
        map_state = _STATES[map_idx]
        confidence = round(max(0.0, min(1.0, smoothed[map_idx])), 4)

        state_probabilities = {s.value: round(smoothed[i], 4) for i, s in enumerate(_STATES)}

        return MarkovRegimeResult(
            state=map_state,
            vol_ratio=round(vol_ratio, 4),
            amihud_score=round(amihud_score, 4),
            state_probabilities=state_probabilities,
            confidence=confidence,
        )

    def classify_portfolio(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> MarkovRegimeResult:
        """Classify the aggregate regime across multiple instruments.

        Each instrument is classified independently and the results are
        combined via confidence-weighted voting over the state posterior.

        Args:
            bars_by_instrument: Mapping of instrument id to its OHLCV bars.

        Returns:
            An aggregated :class:`MarkovRegimeResult`.  If no instrument
            can be classified, returns a default MID_VOL_MID_LIQ result.
        """
        results: list[MarkovRegimeResult] = []
        for bars in bars_by_instrument.values():
            try:
                results.append(self.classify(bars))
            except ValueError:
                continue

        if not results:
            uniform = 1.0 / _NUM_STATES
            return MarkovRegimeResult(
                state=MarkovRegimeState.MID_VOL_MID_LIQ,
                vol_ratio=1.0,
                amihud_score=0.0,
                state_probabilities={s.value: round(uniform, 4) for s in _STATES},
                confidence=round(uniform, 4),
            )

        # Aggregate: confidence-weighted average of state probabilities.
        agg_probs = [0.0] * _NUM_STATES
        total_weight = 0.0
        agg_vol = 0.0
        agg_amihud = 0.0

        for r in results:
            w = r.confidence
            total_weight += w
            agg_vol += r.vol_ratio * w
            agg_amihud += r.amihud_score * w
            for i, s in enumerate(_STATES):
                agg_probs[i] += r.state_probabilities.get(s.value, 0.0) * w

        if total_weight < _EPSILON:
            total_weight = 1.0

        agg_probs = _normalise(agg_probs)
        agg_vol /= total_weight
        agg_amihud /= total_weight

        map_idx = max(range(_NUM_STATES), key=lambda k: agg_probs[k])
        map_state = _STATES[map_idx]

        state_probabilities = {s.value: round(agg_probs[i], 4) for i, s in enumerate(_STATES)}

        return MarkovRegimeResult(
            state=map_state,
            vol_ratio=round(agg_vol, 4),
            amihud_score=round(agg_amihud, 4),
            state_probabilities=state_probabilities,
            confidence=round(max(0.0, min(1.0, agg_probs[map_idx])), 4),
        )
