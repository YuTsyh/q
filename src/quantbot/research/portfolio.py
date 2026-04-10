from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class EqualWeightTopNConstructor:
    top_n: int
    gross_exposure: Decimal
    max_symbol_weight: Decimal

    def construct(self, scores: dict[str, Decimal]) -> dict[str, Decimal]:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        selected = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.top_n]
        if not selected:
            return {}
        raw_weight = self.gross_exposure / Decimal(len(selected))
        capped_weight = min(raw_weight, self.max_symbol_weight)
        return {inst_id: capped_weight for inst_id, _score in selected}

