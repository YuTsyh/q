from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ExecutionAssumptions:
    taker_fee_rate: Decimal
    slippage_bps: Decimal
    partial_fill_ratio: Decimal


@dataclass(frozen=True)
class SimulatedFill:
    inst_id: str
    notional: Decimal
    price: Decimal
    fee: Decimal


class ReplayExecutionSimulator:
    def __init__(self, assumptions: ExecutionAssumptions) -> None:
        if not Decimal("0") < assumptions.partial_fill_ratio <= Decimal("1"):
            raise ValueError("partial_fill_ratio must be in (0, 1]")
        self._assumptions = assumptions

    def rebalance(
        self,
        *,
        equity: Decimal,
        current_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        prices: dict[str, Decimal],
    ) -> list[SimulatedFill]:
        fills: list[SimulatedFill] = []
        for inst_id, target_weight in target_weights.items():
            current_weight = current_weights.get(inst_id, Decimal("0"))
            target_delta_notional = equity * (target_weight - current_weight)
            filled_notional = target_delta_notional * self._assumptions.partial_fill_ratio
            if filled_notional == 0:
                continue
            fill_price = prices[inst_id] * (
                Decimal("1") + self._assumptions.slippage_bps / Decimal("10000")
            )
            fee = abs(filled_notional) * self._assumptions.taker_fee_rate
            fills.append(
                SimulatedFill(inst_id=inst_id, notional=filled_notional, price=fill_price, fee=fee)
            )
        return fills

