from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(StrEnum):
    PENDING_SUBMIT = "pending_submit"
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
}


@dataclass(frozen=True)
class MarketSnapshot:
    inst_id: str
    last_price: Decimal
    received_at: datetime


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    inst_id: str
    side: OrderSide
    order_type: OrderType
    price: Decimal | None
    size: Decimal

    @property
    def notional(self) -> Decimal:
        if self.price is None:
            raise ValueError("Cannot compute notional for an order without a price")
        return self.price * self.size

