from decimal import Decimal

from quantbot.execution.order_state import OrderEvent, OrderStateMachine
from quantbot.core.models import OrderStatus


def test_order_state_machine_tracks_partial_and_full_fill() -> None:
    machine = OrderStateMachine(client_order_id="botA0001", inst_id="BTC-USDT")

    machine.apply(OrderEvent(status=OrderStatus.LIVE, filled_size=Decimal("0")))
    machine.apply(OrderEvent(status=OrderStatus.PARTIALLY_FILLED, filled_size=Decimal("0.01")))
    machine.apply(OrderEvent(status=OrderStatus.FILLED, filled_size=Decimal("0.02")))

    assert machine.status == OrderStatus.FILLED
    assert machine.filled_size == Decimal("0.02")


def test_order_state_machine_does_not_downgrade_terminal_state() -> None:
    machine = OrderStateMachine(client_order_id="botA0001", inst_id="BTC-USDT")

    machine.apply(OrderEvent(status=OrderStatus.FILLED, filled_size=Decimal("0.02")))
    machine.apply(OrderEvent(status=OrderStatus.LIVE, filled_size=Decimal("0.01")))

    assert machine.status == OrderStatus.FILLED
    assert machine.filled_size == Decimal("0.02")

