from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quantbot.core.models import OrderStatus
from quantbot.execution.order_state import OrderEvent, OrderStateMachine
from quantbot.exchange.okx.rest import OkxOrderState


@dataclass(frozen=True)
class ReconciliationResult:
    unknown_client_order_ids: set[str]


def reconcile_orders(
    local_orders: list[OrderStateMachine],
    exchange_open_client_ids: set[str],
) -> ReconciliationResult:
    unknown_ids: set[str] = set()
    for order in local_orders:
        if order.status is not OrderStatus.LIVE:
            continue
        if order.client_order_id in exchange_open_client_ids:
            continue
        order.apply(OrderEvent(status=OrderStatus.UNKNOWN, filled_size=order.filled_size))
        unknown_ids.add(order.client_order_id)
    return ReconciliationResult(unknown_client_order_ids=unknown_ids)


class OrderStateReader(Protocol):
    async def get_order_state(self, *, inst_id: str, client_order_id: str) -> OkxOrderState:
        """Return current exchange state for one order."""


class ReconciliationService:
    def __init__(self, rest_client: OrderStateReader) -> None:
        self._rest_client = rest_client

    async def reconcile_order(self, local_order: OrderStateMachine) -> OkxOrderState:
        exchange_state = await self._rest_client.get_order_state(
            inst_id=local_order.inst_id,
            client_order_id=local_order.client_order_id,
        )
        local_order.apply(
            OrderEvent(status=exchange_state.status, filled_size=exchange_state.filled_size)
        )
        return exchange_state
