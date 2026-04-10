from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.factors import (
    CarryFundingFactor,
    Factor,
    MomentumFactor,
    TrendFactor,
    WeightedFactorScorer,
)
from quantbot.research.portfolio import EqualWeightTopNConstructor


@dataclass(frozen=True)
class MultiFactorPerpStrategy:
    factors: list[Factor]
    scorer: WeightedFactorScorer
    portfolio_constructor: EqualWeightTopNConstructor

    @classmethod
    def default_low_frequency(
        cls,
        *,
        top_n: int,
        gross_exposure: Decimal,
    ) -> MultiFactorPerpStrategy:
        return cls(
            factors=[
                MomentumFactor(lookback=3),
                TrendFactor(lookback=3),
                CarryFundingFactor(lookback=2),
            ],
            scorer=WeightedFactorScorer(
                weights={
                    "momentum": Decimal("0.4"),
                    "trend": Decimal("0.4"),
                    "carry": Decimal("0.2"),
                }
            ),
            portfolio_constructor=EqualWeightTopNConstructor(
                top_n=top_n,
                gross_exposure=gross_exposure,
                max_symbol_weight=gross_exposure,
            ),
        )

    def allocate(
        self,
        *,
        bars_by_instrument: dict[str, list[OhlcvBar]],
        funding_by_instrument: dict[str, list[FundingRate]],
    ) -> dict[str, Decimal]:
        features: dict[str, dict[str, Decimal]] = {}
        for inst_id, bars in bars_by_instrument.items():
            instrument_funding = funding_by_instrument.get(inst_id, [])
            features[inst_id] = {
                factor.name: factor.compute(bars, instrument_funding) for factor in self.factors
            }
        return self.portfolio_constructor.construct(self.scorer.score(features))

