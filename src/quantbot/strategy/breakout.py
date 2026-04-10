from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from quantbot.core.models import MarketSnapshot, OrderIntent, OrderSide, OrderType


@dataclass(frozen=True)
class BreakoutConfig:
    inst_id: str
    lookback: int
    order_size: Decimal
    limit_offset_bps: Decimal
    client_order_prefix: str


class BreakoutStrategy:
    def __init__(self, config: BreakoutConfig) -> None:
        if config.lookback < 2:
            raise ValueError("lookback must be at least 2")
        self._config = config
        self._prices: deque[Decimal] = deque(maxlen=config.lookback)
        self._next_sequence = 1

    def on_market(self, snapshot: MarketSnapshot) -> OrderIntent | None:
        if snapshot.inst_id != self._config.inst_id:
            return None
        previous_high = max(self._prices) if len(self._prices) == self._config.lookback else None
        self._prices.append(snapshot.last_price)
        if previous_high is None or snapshot.last_price <= previous_high:
            return None

        limit_price = snapshot.last_price * (
            Decimal("1") + self._config.limit_offset_bps / Decimal("10000")
        )
        intent = OrderIntent(
            client_order_id=f"{self._config.client_order_prefix}{self._next_sequence:06d}",
            inst_id=self._config.inst_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=limit_price,
            size=self._config.order_size,
        )
        self._next_sequence += 1
        return intent

