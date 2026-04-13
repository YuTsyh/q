from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from quantbot.research.data import FundingRate, OhlcvBar


class Factor(Protocol):
    name: str

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        """Compute one instrument-level factor value."""


@dataclass(frozen=True)
class MomentumFactor:
    lookback: int
    name: str = "momentum"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for momentum")
        start = bars[-self.lookback - 1].close
        end = bars[-1].close
        if start <= 0:
            return Decimal("0")
        return end / start - Decimal("1")


@dataclass(frozen=True)
class TrendFactor:
    lookback: int
    name: str = "trend"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for trend")
        recent = bars[-self.lookback :]
        average = sum((bar.close for bar in recent), Decimal("0")) / Decimal(len(recent))
        if average <= 0:
            return Decimal("0")
        return bars[-1].close / average - Decimal("1")


@dataclass(frozen=True)
class CarryFundingFactor:
    lookback: int
    name: str = "carry"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(funding) < self.lookback:
            raise ValueError("not enough funding observations for carry")
        recent = funding[-self.lookback :]
        average = sum((item.funding_rate for item in recent), Decimal("0")) / Decimal(len(recent))
        return -average


@dataclass(frozen=True)
class WeightedFactorScorer:
    weights: dict[str, Decimal]

    def score(self, features: dict[str, dict[str, Decimal]]) -> dict[str, Decimal]:
        scores = {inst_id: Decimal("0") for inst_id in features}
        for factor_name, weight in self.weights.items():
            ranked = _rank_desc(
                {
                    inst_id: factor_values[factor_name]
                    for inst_id, factor_values in features.items()
                    if factor_name in factor_values
                }
            )
            for inst_id, rank_score in ranked.items():
                scores[inst_id] += weight * rank_score
        return scores


def _rank_desc(values: dict[str, Decimal]) -> dict[str, Decimal]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) == 1:
        return {ordered[0][0]: Decimal("1")}
    denominator = Decimal(len(ordered) - 1)
    return {
        inst_id: Decimal("1") - Decimal(index) / denominator
        for index, (inst_id, _value) in enumerate(ordered)
    }

