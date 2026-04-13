"""Adaptive Dual Momentum Strategy.

A multi-factor portfolio strategy combining:
1. Volatility-adjusted momentum for cross-sectional selection
2. Trend strength for regime filtering
3. Carry (funding) for perpetual swap alpha
4. Inverse-volatility portfolio weighting for risk parity

Theory: Based on Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
and Antonacci (2014) "Dual Momentum Investing". The strategy:
- Selects assets with strong risk-adjusted momentum
- Filters out assets in downtrends
- Weights inversely to volatility for better risk allocation
- Uses funding rate carry as additional alpha source
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.factors import (
    CarryFundingFactor,
    Factor,
    WeightedFactorScorer,
)
from quantbot.research.vol_factors import (
    DualMomentumFactor,
    TrendStrengthFactor,
    VolAdjMomentumFactor,
)


@dataclass(frozen=True)
class InverseVolWeightConstructor:
    """Inverse-volatility weighted portfolio construction.

    Allocates more capital to less volatile assets for
    approximate risk parity exposure.
    """

    top_n: int
    gross_exposure: Decimal
    max_symbol_weight: Decimal

    def construct(
        self,
        scores: dict[str, Decimal],
        volatilities: dict[str, float] | None = None,
    ) -> dict[str, Decimal]:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        # Select top N by score
        selected = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.top_n]
        if not selected:
            return {}

        # If no volatility data, use equal weight
        if volatilities is None:
            raw_weight = self.gross_exposure / Decimal(len(selected))
            capped = min(raw_weight, self.max_symbol_weight)
            return {inst_id: capped for inst_id, _ in selected}

        # Inverse volatility weighting
        inv_vols: dict[str, float] = {}
        for inst_id, _ in selected:
            vol = volatilities.get(inst_id, 0.01)
            inv_vols[inst_id] = 1.0 / max(vol, 0.001)

        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol <= 0:
            raw_weight = self.gross_exposure / Decimal(len(selected))
            capped = min(raw_weight, self.max_symbol_weight)
            return {inst_id: capped for inst_id, _ in selected}

        weights: dict[str, Decimal] = {}
        for inst_id, _ in selected:
            raw = Decimal(str(inv_vols[inst_id] / total_inv_vol)) * self.gross_exposure
            weights[inst_id] = min(raw, self.max_symbol_weight)

        return weights


@dataclass(frozen=True)
class AdaptiveDualMomentumStrategy:
    """Adaptive dual momentum strategy with volatility adjustment.

    Combines multiple alpha factors and uses inverse-volatility
    weighting for better risk-adjusted returns.
    """

    factors: list[Factor]
    scorer: WeightedFactorScorer
    portfolio_constructor: InverseVolWeightConstructor
    vol_lookback: int = 10
    trend_filter_threshold: Decimal = Decimal("0.4")

    @classmethod
    def default(
        cls,
        *,
        top_n: int = 3,
        gross_exposure: Decimal = Decimal("1.0"),
        momentum_lookback: int = 5,
        vol_lookback: int = 10,
    ) -> AdaptiveDualMomentumStrategy:
        """Create strategy with default parameters."""
        return cls(
            factors=[
                VolAdjMomentumFactor(lookback=momentum_lookback),
                TrendStrengthFactor(lookbacks=(3, 5, 10)),
                DualMomentumFactor(lookback=momentum_lookback),
                CarryFundingFactor(lookback=2),
            ],
            scorer=WeightedFactorScorer(
                weights={
                    "vol_adj_momentum": Decimal("0.35"),
                    "trend_strength": Decimal("0.25"),
                    "dual_momentum": Decimal("0.25"),
                    "carry": Decimal("0.15"),
                }
            ),
            portfolio_constructor=InverseVolWeightConstructor(
                top_n=top_n,
                gross_exposure=gross_exposure,
                max_symbol_weight=Decimal("0.4"),
            ),
            vol_lookback=vol_lookback,
            trend_filter_threshold=Decimal("0.4"),
        )

    def allocate(
        self,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        """Compute target portfolio weights."""
        features: dict[str, dict[str, Decimal]] = {}
        volatilities: dict[str, float] = {}

        for inst_id, bars in bars_by_instrument.items():
            funding = funding_by_instrument.get(inst_id, [])
            try:
                inst_features: dict[str, Decimal] = {}
                for factor in self.factors:
                    inst_features[factor.name] = factor.compute(bars, funding)
                features[inst_id] = inst_features

                # Compute volatility for weighting
                vol = self._compute_volatility(bars)
                volatilities[inst_id] = vol
            except (ValueError, ZeroDivisionError):
                continue

        if not features:
            return {}

        # Apply trend filter: remove assets with weak trend
        if "trend_strength" in next(iter(features.values()), {}):
            features = {
                inst_id: feats
                for inst_id, feats in features.items()
                if feats.get("trend_strength", Decimal("0")) >= self.trend_filter_threshold
            }

        if not features:
            return {}

        scores = self.scorer.score(features)
        return self.portfolio_constructor.construct(scores, volatilities)

    def _compute_volatility(self, bars: list[OhlcvBar]) -> float:
        """Compute annualised volatility from bars."""
        if len(bars) < self.vol_lookback + 1:
            return 0.01  # Default low vol
        returns = []
        for i in range(-self.vol_lookback, 0):
            prev = bars[i - 1].close
            curr = bars[i].close
            if prev > 0:
                returns.append(float(curr / prev - 1))
        if len(returns) < 2:
            return 0.01
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var)


def create_adaptive_dual_momentum_allocator(
    top_n: int = 3,
    gross_exposure: float = 1.0,
    momentum_lookback: int = 5,
    vol_lookback: int = 10,
) -> callable:
    """Factory function to create an allocator callable for backtesting."""
    strategy = AdaptiveDualMomentumStrategy.default(
        top_n=top_n,
        gross_exposure=Decimal(str(gross_exposure)),
        momentum_lookback=momentum_lookback,
        vol_lookback=vol_lookback,
    )

    def allocator(
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        return strategy.allocate(bars_by_instrument, funding_by_instrument)

    return allocator
