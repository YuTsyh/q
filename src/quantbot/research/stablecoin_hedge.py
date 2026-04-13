"""Dynamic stablecoin hedging and volatility scaling for crypto portfolios.

Combines two complementary risk-management layers:

1. **Volatility scaling** — adjusts per-instrument weights so that each
   position targets a constant annualised volatility budget.  When realised
   vol rises the position shrinks; when vol falls it grows (Moskowitz,
   Ooi & Pedersen 2012, "Time-Series Momentum").

2. **Stablecoin hedging** — shifts a regime-dependent fraction of capital
   into stablecoins (e.g. USDT / USDC) to reduce drawdowns during
   stressed or crisis regimes identified by the Markov-Switching detector.

The :class:`AdaptivePortfolioConstructor` orchestrates both layers, first
scaling for vol then overlaying the stablecoin hedge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from quantbot.research.data import OhlcvBar
from quantbot.research.markov_regime import MarkovRegimeState

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-12
_ZERO: Decimal = Decimal(0)
_ONE: Decimal = Decimal(1)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityScalingConfig:
    """Parameters for per-instrument volatility targeting.

    Attributes
    ----------
    target_volatility:
        Annualised volatility target (e.g. 0.15 = 15 %).
    vol_lookback:
        Number of recent bars used to estimate realised volatility.
    vol_floor:
        Minimum annualised vol estimate to prevent blow-up.
    vol_cap:
        Maximum annualised vol estimate.
    annualization_factor:
        Trading days per year.  365 for 24/7 crypto markets.
    max_leverage:
        Upper bound on the per-instrument scaling factor.
    """

    target_volatility: float = 0.15
    vol_lookback: int = 20
    vol_floor: float = 0.05
    vol_cap: float = 1.0
    annualization_factor: float = 365.0
    max_leverage: float = 2.0


@dataclass(frozen=True)
class StablecoinHedgeConfig:
    """Parameters for regime-dependent stablecoin allocation.

    Attributes
    ----------
    stablecoin_ids:
        Instrument identifiers treated as stablecoins.
    crisis_stablecoin_weight:
        Target stablecoin allocation in CRISIS regime.
    high_vol_stablecoin_weight:
        Target stablecoin allocation in HIGH_VOL_LOW_LIQ regime.
    mid_vol_stablecoin_weight:
        Target stablecoin allocation in MID_VOL_MID_LIQ regime.
    low_vol_stablecoin_weight:
        Target stablecoin allocation in LOW_VOL_HIGH_LIQ regime.
    rebalance_threshold:
        Minimum absolute weight change required to trigger a rebalance.
    """

    stablecoin_ids: tuple[str, ...] = ("USDT", "USDC")
    crisis_stablecoin_weight: float = 0.80
    high_vol_stablecoin_weight: float = 0.40
    mid_vol_stablecoin_weight: float = 0.10
    low_vol_stablecoin_weight: float = 0.0
    rebalance_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Volatility scaler
# ---------------------------------------------------------------------------


class VolatilityScaler:
    """Scales portfolio weights so each position targets a fixed vol budget."""

    def __init__(self, config: VolatilityScalingConfig | None = None) -> None:
        self.config = config or VolatilityScalingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_weights(
        self,
        weights: dict[str, Decimal],
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> dict[str, Decimal]:
        """Return volatility-scaled weights.

        For every instrument present in *weights* that also has sufficient
        price history in *bars_by_instrument*, the weight is multiplied by
        ``target_vol / realised_vol`` (clamped to ``[1/max_leverage,
        max_leverage]``).  Instruments without bar data are left unchanged.
        The result is then normalised so that the gross exposure equals the
        original gross exposure.

        Parameters
        ----------
        weights:
            Raw instrument weights (e.g. from a factor model).
        bars_by_instrument:
            Recent OHLCV bars keyed by instrument id.

        Returns
        -------
        dict[str, Decimal]
            Volatility-scaled and re-normalised weights.
        """
        if not weights:
            return {}

        cfg = self.config
        scaled: dict[str, Decimal] = {}

        for inst_id, raw_weight in weights.items():
            if raw_weight == _ZERO:
                scaled[inst_id] = _ZERO
                continue

            bars = bars_by_instrument.get(inst_id, [])
            if len(bars) < 2:
                # Not enough data — pass through unchanged.
                scaled[inst_id] = raw_weight
                continue

            realised = self.compute_realized_vol(bars)
            if realised < _EPSILON:
                scaled[inst_id] = raw_weight
                continue

            factor = cfg.target_volatility / realised
            floor = 1.0 / cfg.max_leverage
            factor = max(floor, min(factor, cfg.max_leverage))
            scaled[inst_id] = raw_weight * Decimal(str(factor))

        # Normalise to preserve original gross exposure.
        original_gross = sum(abs(w) for w in weights.values())
        scaled_gross = sum(abs(w) for w in scaled.values())

        if scaled_gross > _ZERO and original_gross > _ZERO:
            ratio = original_gross / scaled_gross
            scaled = {k: v * ratio for k, v in scaled.items()}

        return scaled

    def compute_realized_vol(self, bars: list[OhlcvBar]) -> float:
        """Annualised close-to-close return volatility.

        Parameters
        ----------
        bars:
            OHLCV bars sorted chronologically (oldest first).  At least two
            bars are required; otherwise ``vol_floor`` is returned.

        Returns
        -------
        float
            Annualised volatility clamped to ``[vol_floor, vol_cap]``.
        """
        cfg = self.config

        if len(bars) < 2:
            return cfg.vol_floor

        # Use at most the last `lookback + 1` bars to get `lookback` returns.
        window = bars[-(cfg.vol_lookback + 1) :]

        returns: list[float] = []
        for i in range(1, len(window)):
            prev_close = float(window[i - 1].close)
            cur_close = float(window[i].close)
            if prev_close > _EPSILON:
                returns.append(math.log(cur_close / prev_close))

        if len(returns) < 2:
            return cfg.vol_floor

        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(var)
        annual_vol = daily_vol * math.sqrt(cfg.annualization_factor)

        return max(cfg.vol_floor, min(annual_vol, cfg.vol_cap))


# ---------------------------------------------------------------------------
# Stablecoin hedger
# ---------------------------------------------------------------------------


class StablecoinHedger:
    """Shifts capital into stablecoins based on the current market regime."""

    def __init__(self, config: StablecoinHedgeConfig | None = None) -> None:
        self.config = config or StablecoinHedgeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_hedge(
        self,
        weights: dict[str, Decimal],
        regime: MarkovRegimeState,
    ) -> dict[str, Decimal]:
        """Apply a regime-dependent stablecoin overlay.

        Non-stablecoin positions are scaled down proportionally so that the
        freed capital is redirected to the configured stablecoin basket.
        Gross exposure is preserved.

        If the absolute change in the stablecoin allocation is below
        ``rebalance_threshold`` the original weights are returned unmodified.

        Parameters
        ----------
        weights:
            Current instrument weights.
        regime:
            Most-probable Markov regime state.

        Returns
        -------
        dict[str, Decimal]
            Hedged weights with stablecoin allocation injected.
        """
        if not weights:
            return {}

        cfg = self.config
        target_pct = self._get_stablecoin_target(regime)

        # Partition into stablecoin and risky buckets.
        stable_ids = set(cfg.stablecoin_ids)
        current_stable = sum(
            (abs(w) for inst, w in weights.items() if inst in stable_ids),
            _ZERO,
        )
        gross = sum(abs(w) for w in weights.values())
        if gross == _ZERO:
            return dict(weights)

        current_stable_pct = float(current_stable / gross)

        # Skip rebalance if within threshold.
        if abs(target_pct - current_stable_pct) < cfg.rebalance_threshold:
            return dict(weights)

        target_stable_amount = gross * Decimal(str(target_pct))
        risky_budget = gross - target_stable_amount

        # Scale risky (non-stablecoin) weights proportionally.
        risky_weights: dict[str, Decimal] = {
            k: v for k, v in weights.items() if k not in stable_ids
        }
        risky_gross = sum(abs(w) for w in risky_weights.values())

        result: dict[str, Decimal] = {}

        if risky_gross > _ZERO:
            scale = risky_budget / risky_gross
            for inst_id, w in risky_weights.items():
                result[inst_id] = w * scale
        else:
            # All existing weight is in stablecoins; nothing risky to scale.
            for inst_id in risky_weights:
                result[inst_id] = _ZERO

        # Distribute stablecoin allocation equally.
        n_stables = len(cfg.stablecoin_ids)
        if n_stables > 0 and target_stable_amount > _ZERO:
            per_coin = target_stable_amount / Decimal(n_stables)
            for sid in cfg.stablecoin_ids:
                result[sid] = per_coin
        else:
            # Preserve any stablecoins already present with zero target.
            for sid in cfg.stablecoin_ids:
                if sid in weights:
                    result[sid] = _ZERO

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_stablecoin_target(self, regime: MarkovRegimeState) -> float:
        """Map a :class:`MarkovRegimeState` to a stablecoin weight target.

        Returns
        -------
        float
            Fraction of gross exposure to allocate to stablecoins.
        """
        cfg = self.config
        mapping: dict[MarkovRegimeState, float] = {
            MarkovRegimeState.CRISIS: cfg.crisis_stablecoin_weight,
            MarkovRegimeState.HIGH_VOL_LOW_LIQ: cfg.high_vol_stablecoin_weight,
            MarkovRegimeState.MID_VOL_MID_LIQ: cfg.mid_vol_stablecoin_weight,
            MarkovRegimeState.LOW_VOL_HIGH_LIQ: cfg.low_vol_stablecoin_weight,
        }
        return mapping.get(regime, cfg.mid_vol_stablecoin_weight)


# ---------------------------------------------------------------------------
# Adaptive constructor (vol scaling + stablecoin hedging)
# ---------------------------------------------------------------------------


class AdaptivePortfolioConstructor:
    """Orchestrates volatility scaling followed by stablecoin hedging.

    Usage::

        constructor = AdaptivePortfolioConstructor()
        final = constructor.construct(raw_weights, regime, bars)
    """

    def __init__(
        self,
        vol_config: VolatilityScalingConfig | None = None,
        hedge_config: StablecoinHedgeConfig | None = None,
    ) -> None:
        self._scaler = VolatilityScaler(vol_config)
        self._hedger = StablecoinHedger(hedge_config)

    def construct(
        self,
        raw_weights: dict[str, Decimal],
        regime: MarkovRegimeState,
        bars_by_instrument: dict[str, list[OhlcvBar]],
    ) -> dict[str, Decimal]:
        """Build final portfolio weights.

        Pipeline:
        1. Scale *raw_weights* for per-instrument volatility.
        2. Overlay stablecoin hedge according to *regime*.

        Parameters
        ----------
        raw_weights:
            Initial desired weights (e.g. from a factor model).
        regime:
            Current Markov regime state.
        bars_by_instrument:
            Recent OHLCV bars keyed by instrument id.

        Returns
        -------
        dict[str, Decimal]
            Final, risk-adjusted portfolio weights.
        """
        if not raw_weights:
            return {}

        scaled = self._scaler.scale_weights(raw_weights, bars_by_instrument)
        hedged = self._hedger.apply_hedge(scaled, regime)
        return hedged
