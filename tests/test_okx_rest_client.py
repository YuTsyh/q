import asyncio
from decimal import Decimal

import httpx
import pytest

from quantbot.core.models import OrderIntent, OrderSide, OrderType
from quantbot.exchange.okx.auth import OkxCredentials, OkxSigner
from quantbot.core.models import OrderStatus
from quantbot.exchange.okx.rest import AmbiguousOrderError, OkxRestClient


def _client_with_transport(transport: httpx.MockTransport) -> OkxRestClient:
    signer = OkxSigner(OkxCredentials(api_key="key", api_secret="secret", passphrase="pass"))
    return OkxRestClient(
        base_url="https://us.okx.com",
        signer=signer,
        simulated=True,
        transport=transport,
        timestamp_factory=lambda: "2020-12-08T09:08:57.715Z",
    )


def _intent() -> OrderIntent:
    return OrderIntent(
        client_order_id="botA0006",
        inst_id="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("65000.1"),
        size=Decimal("0.001"),
    )


def test_place_order_sends_signed_demo_cash_order_without_real_network() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [{"ordId": "123", "clOrdId": "botA0006", "sCode": "0", "sMsg": ""}],
            },
        )

    client = _client_with_transport(httpx.MockTransport(handler))

    ack = asyncio.run(client.place_order(_intent()))

    assert ack.order_id == "123"
    assert ack.client_order_id == "botA0006"
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v5/trade/order"
    assert request.headers["x-simulated-trading"] == "1"
    assert request.headers["OK-ACCESS-KEY"] == "key"
    assert b'"tdMode":"cash"' in request.content


def test_place_order_does_not_retry_when_order_state_is_ambiguous() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    client = _client_with_transport(httpx.MockTransport(handler))

    with pytest.raises(AmbiguousOrderError):
        asyncio.run(client.place_order(_intent()))

    assert attempts == 1


def test_get_order_state_queries_by_client_order_id() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "ordId": "123",
                        "clOrdId": "botA0006",
                        "instId": "BTC-USDT",
                        "state": "partially_filled",
                        "accFillSz": "0.0005",
                    }
                ],
            },
        )

    client = _client_with_transport(httpx.MockTransport(handler))

    state = asyncio.run(client.get_order_state(inst_id="BTC-USDT", client_order_id="botA0006"))

    assert state.order_id == "123"
    assert state.client_order_id == "botA0006"
    assert state.status == OrderStatus.PARTIALLY_FILLED
    assert state.filled_size == Decimal("0.0005")
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/api/v5/trade/order"
    assert seen_requests[0].url.params["instId"] == "BTC-USDT"
    assert seen_requests[0].url.params["clOrdId"] == "botA0006"


def test_cancel_order_sends_client_order_id_and_requires_reconciliation() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [{"ordId": "123", "clOrdId": "botA0006", "sCode": "0", "sMsg": ""}],
            },
        )

    client = _client_with_transport(httpx.MockTransport(handler))

    ack = asyncio.run(client.cancel_order(inst_id="BTC-USDT", client_order_id="botA0006"))

    assert ack.accepted is True
    assert ack.client_order_id == "botA0006"
    assert seen_requests[0].method == "POST"
    assert seen_requests[0].url.path == "/api/v5/trade/cancel-order"
    assert b'"clOrdId":"botA0006"' in seen_requests[0].content


def test_fetch_spot_instrument_metadata_parses_okx_steps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v5/public/instruments"
        assert request.url.params["instType"] == "SPOT"
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instType": "SPOT",
                        "instId": "BTC-USDT",
                        "tickSz": "0.1",
                        "lotSz": "0.00001",
                        "minSz": "0.0001",
                        "state": "live",
                    }
                ],
            },
        )

    client = _client_with_transport(httpx.MockTransport(handler))

    metadata = asyncio.run(client.fetch_spot_instrument("BTC-USDT"))

    assert metadata.inst_id == "BTC-USDT"
    assert metadata.tick_sz == Decimal("0.1")
    assert metadata.lot_sz == Decimal("0.00001")
    assert metadata.min_sz == Decimal("0.0001")
    assert metadata.state == "live"
