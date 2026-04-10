import sqlite3

from quantbot.persistence.event_store import EventStore


def test_event_store_appends_and_reads_execution_events() -> None:
    connection = sqlite3.connect(":memory:")
    store = EventStore(connection)

    store.append_event(
        stream="orders",
        event_type="order_submitted",
        payload={"clOrdId": "botA0007", "instId": "BTC-USDT"},
    )

    events = store.list_events("orders")

    assert len(events) == 1
    assert events[0].event_type == "order_submitted"
    assert events[0].payload == {"clOrdId": "botA0007", "instId": "BTC-USDT"}

