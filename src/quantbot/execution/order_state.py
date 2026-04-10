from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantbot.core.models import OrderStatus, TERMINAL_ORDER_STATUSES


@dataclass(frozen=True)
class OrderEvent:
    status: OrderStatus
    filled_size: Decimal


class OrderStateMachine:
    def __init__(self, client_order_id: str, inst_id: str) -> None:
        self.client_order_id = client_order_id
        self.inst_id = inst_id
        self.status = OrderStatus.PENDING_SUBMIT
        self.filled_size = Decimal("0")

    def apply(self, event: OrderEvent) -> None:
        if self.status in TERMINAL_ORDER_STATUSES:
            return
        self.status = event.status
        if event.filled_size > self.filled_size:
            self.filled_size = event.filled_size

