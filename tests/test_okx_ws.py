from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantbot.core.models import OrderStatus
from quantbot.exchange.okx.auth import OkxCredentials, OkxSigner
from quantbot.exchange.okx.ws import (
    OkxOrderChannelParser,
    OkxWsChannel,
    OkxWsHeartbeat,
    OkxWsMessageBuilder,
    OkxWsSubscriptionRegistry,
)


def _builder() -> OkxWsMessageBuilder:
    signer = OkxSigner(OkxCredentials(api_key="key", api_secret="secret", passphrase="pass"))
    return OkxWsMessageBuilder(
        credentials=OkxCredentials(api_key="key", api_secret="secret", passphrase="pass"),
        signer=signer,
        timestamp_factory=lambda: "1700000000",
    )


def test_ws_login_message_uses_okx_verify_path_signature() -> None:
    message = _builder().login_message()

    assert message == {
        "op": "login",
        "args": [
            {
                "apiKey": "key",
                "passphrase": "pass",
                "timestamp": "1700000000",
                "sign": "lhmJXK08fk9SI1ZwFXKFRrPtzfbNOwC+D1xMJJ/1KZg=",
            }
        ],
    }


def test_ws_subscribe_messages_for_public_ticker_and_private_orders() -> None:
    builder = _builder()

    assert builder.subscribe_ticker("BTC-USDT", request_id="pub1") == {
        "id": "pub1",
        "op": "subscribe",
        "args": [{"channel": "tickers", "instId": "BTC-USDT"}],
    }
    assert builder.subscribe_orders("SPOT", request_id="priv1") == {
        "id": "priv1",
        "op": "subscribe",
        "args": [{"channel": "orders", "instType": "SPOT"}],
    }


def test_ws_heartbeat_requests_ping_before_okx_30_second_idle_cutoff() -> None:
    now = datetime.now(UTC)
    heartbeat = OkxWsHeartbeat(idle_timeout=timedelta(seconds=25))

    assert heartbeat.should_ping(last_received_at=now, now=now + timedelta(seconds=24)) is False
    assert heartbeat.should_ping(last_received_at=now, now=now + timedelta(seconds=26)) is True
    assert heartbeat.is_pong("pong") is True


def test_order_channel_parser_maps_order_update() -> None:
    message = {
        "arg": {"channel": "orders", "instType": "SPOT"},
        "data": [
            {
                "ordId": "123",
                "clOrdId": "botA0010",
                "instId": "BTC-USDT",
                "state": "filled",
                "accFillSz": "0.001",
                "fillPx": "65000.1",
                "fillTime": "1710000000000",
            }
        ],
    }

    event = OkxOrderChannelParser.parse(message)

    assert event.order_id == "123"
    assert event.client_order_id == "botA0010"
    assert event.inst_id == "BTC-USDT"
    assert event.status == OrderStatus.FILLED
    assert event.filled_size == Decimal("0.001")
    assert event.fill_price == Decimal("65000.1")
    assert event.fill_time.isoformat() == "2024-03-09T16:00:00+00:00"


def test_subscription_registry_builds_resubscribe_messages_after_reconnect() -> None:
    registry = OkxWsSubscriptionRegistry()
    registry.add(OkxWsChannel(channel="tickers", inst_id="BTC-USDT"))
    registry.add(OkxWsChannel(channel="orders", inst_type="SPOT"))

    messages = registry.resubscribe_messages()

    assert messages == [
        {
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": "BTC-USDT"}],
        },
        {
            "op": "subscribe",
            "args": [{"channel": "orders", "instType": "SPOT"}],
        },
    ]
