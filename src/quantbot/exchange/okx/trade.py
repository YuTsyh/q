from __future__ import annotations

from decimal import Decimal

from quantbot.core.models import OrderIntent, OrderStatus, OrderType


class OkxOrderMapper:
    @staticmethod
    def to_place_order_payload(intent: OrderIntent) -> dict[str, str]:
        if intent.order_type is OrderType.MARKET:
            raise ValueError("OKX Spot Demo market orders are disabled until explicit slippage policy exists")
        if intent.price is None:
            raise ValueError("Limit-style OKX orders require a price")
        return {
            "instId": intent.inst_id,
            "tdMode": "cash",
            "clOrdId": intent.client_order_id,
            "side": intent.side.value,
            "ordType": intent.order_type.value,
            "px": _decimal_to_okx_string(intent.price),
            "sz": _decimal_to_okx_string(intent.size),
        }

    @staticmethod
    def to_domain_status(okx_status: str) -> OrderStatus:
        return _OKX_STATUS_MAP.get(okx_status, OrderStatus.UNKNOWN)


_OKX_STATUS_MAP = {
    "live": OrderStatus.LIVE,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
}


def _decimal_to_okx_string(value: Decimal) -> str:
    return format(value.normalize(), "f")

