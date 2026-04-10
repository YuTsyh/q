from quantbot.observability.metrics import BotMetrics


def test_bot_metrics_records_reconnects_rejects_and_unknown_orders() -> None:
    metrics = BotMetrics()

    metrics.record_ws_reconnect("okx_private")
    metrics.record_order_reject("risk")
    metrics.record_unknown_order()

    assert metrics.snapshot()["ws_reconnects_total"]["okx_private"] == 1
    assert metrics.snapshot()["order_rejects_total"]["risk"] == 1
    assert metrics.snapshot()["unknown_orders_total"] == 1
