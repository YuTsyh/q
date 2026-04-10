from decimal import Decimal

from quantbot.core.models import OrderStatus
from quantbot.execution.reconciliation import (
    ReconciliationResult,
    ReconciliationService,
    reconcile_orders,
)
from quantbot.execution.order_state import OrderEvent, OrderStateMachine
from quantbot.exchange.okx.rest import OkxOrderState


def test_reconcile_marks_missing_local_open_order_unknown() -> None:
    local = OrderStateMachine(client_order_id="botA0004", inst_id="BTC-USDT")
    local.apply(OrderEvent(status=OrderStatus.LIVE, filled_size=Decimal("0")))

    result = reconcile_orders(local_orders=[local], exchange_open_client_ids=set())

    assert result == ReconciliationResult(unknown_client_order_ids={"botA0004"})
    assert local.status == OrderStatus.UNKNOWN


def test_reconcile_keeps_exchange_visible_open_order_live() -> None:
    local = OrderStateMachine(client_order_id="botA0005", inst_id="BTC-USDT")
    local.apply(OrderEvent(status=OrderStatus.LIVE, filled_size=Decimal("0")))

    result = reconcile_orders(local_orders=[local], exchange_open_client_ids={"botA0005"})

    assert result == ReconciliationResult(unknown_client_order_ids=set())
    assert local.status == OrderStatus.LIVE


def test_reconciliation_service_updates_local_order_from_rest_query() -> None:
    class FakeRestClient:
        async def get_order_state(self, *, inst_id: str, client_order_id: str) -> OkxOrderState:
            assert inst_id == "BTC-USDT"
            assert client_order_id == "botA0006"
            return OkxOrderState(
                order_id="123",
                client_order_id="botA0006",
                status=OrderStatus.FILLED,
                filled_size=Decimal("0.001"),
            )

    local = OrderStateMachine(client_order_id="botA0006", inst_id="BTC-USDT")
    local.apply(OrderEvent(status=OrderStatus.LIVE, filled_size=Decimal("0")))

    service = ReconciliationService(rest_client=FakeRestClient())

    import asyncio

    result = asyncio.run(service.reconcile_order(local))

    assert result.status == OrderStatus.FILLED
    assert local.status == OrderStatus.FILLED
    assert local.filled_size == Decimal("0.001")
