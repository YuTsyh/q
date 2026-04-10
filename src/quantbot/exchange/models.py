from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class InstrumentMetadata:
    inst_id: str
    inst_type: str
    tick_sz: Decimal
    lot_sz: Decimal
    min_sz: Decimal
    state: str

    def quantize_price(self, price: Decimal) -> Decimal:
        return _floor_to_step(price, self.tick_sz)

    def quantize_size(self, size: Decimal) -> Decimal:
        return _floor_to_step(size, self.lot_sz)

    def validate_size(self, size: Decimal) -> None:
        if size < self.min_sz:
            raise ValueError(f"Order size {size} is below minimum {self.min_sz} for {self.inst_id}")


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("step must be positive")
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step

