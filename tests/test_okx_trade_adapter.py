from decimal import Decimal

from quantbot.core.models import OrderIntent, OrderSide, OrderStatus, OrderType
from quantbot.exchange.okx.trade import OkxOrderMapper


def test_okx_order_mapper_builds_spot_cash_limit_payload() -> None:
    intent = OrderIntent(
        client_order_id="botA0002",
        inst_id="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("65000.1"),
        size=Decimal("0.001"),
    )

    payload = OkxOrderMapper.to_place_order_payload(intent)

    assert payload == {
        "instId": "BTC-USDT",
        "tdMode": "cash",
        "clOrdId": "botA0002",
        "side": "buy",
        "ordType": "limit",
        "px": "65000.1",
        "sz": "0.001",
    }


def test_okx_order_mapper_rejects_market_order_without_explicit_price_policy() -> None:
    intent = OrderIntent(
        client_order_id="botA0003",
        inst_id="BTC-USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        price=None,
        size=Decimal("0.001"),
    )

    try:
        OkxOrderMapper.to_place_order_payload(intent)
    except ValueError as exc:
        assert "market orders are disabled" in str(exc)
    else:
        raise AssertionError("market order should be rejected")


def test_okx_order_mapper_maps_exchange_statuses() -> None:
    assert OkxOrderMapper.to_domain_status("live") == OrderStatus.LIVE
    assert OkxOrderMapper.to_domain_status("partially_filled") == OrderStatus.PARTIALLY_FILLED
    assert OkxOrderMapper.to_domain_status("filled") == OrderStatus.FILLED
    assert OkxOrderMapper.to_domain_status("canceled") == OrderStatus.CANCELED
    assert OkxOrderMapper.to_domain_status("unknown-new-value") == OrderStatus.UNKNOWN

