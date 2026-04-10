from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from quantbot.core.models import OrderStatus
from quantbot.exchange.okx.auth import OkxCredentials, OkxSigner
from quantbot.exchange.okx.trade import OkxOrderMapper


@dataclass(frozen=True)
class OkxOrderChannelEvent:
    order_id: str
    client_order_id: str
    inst_id: str
    status: OrderStatus
    filled_size: Decimal
    fill_price: Decimal | None
    fill_time: datetime | None


@dataclass(frozen=True)
class OkxWsChannel:
    channel: str
    inst_id: str | None = None
    inst_type: str | None = None

    def to_arg(self) -> dict[str, str]:
        arg = {"channel": self.channel}
        if self.inst_id is not None:
            arg["instId"] = self.inst_id
        if self.inst_type is not None:
            arg["instType"] = self.inst_type
        return arg


class OkxWsSubscriptionRegistry:
    def __init__(self) -> None:
        self._channels: list[OkxWsChannel] = []

    def add(self, channel: OkxWsChannel) -> None:
        if channel not in self._channels:
            self._channels.append(channel)

    def resubscribe_messages(self) -> list[dict[str, Any]]:
        return [{"op": "subscribe", "args": [channel.to_arg()]} for channel in self._channels]


class OkxWsMessageBuilder:
    def __init__(
        self,
        *,
        credentials: OkxCredentials,
        signer: OkxSigner,
        timestamp_factory: Callable[[], str],
    ) -> None:
        self._credentials = credentials
        self._signer = signer
        self._timestamp_factory = timestamp_factory

    def login_message(self) -> dict[str, Any]:
        timestamp = self._timestamp_factory()
        return {
            "op": "login",
            "args": [
                {
                    "apiKey": self._credentials.api_key,
                    "passphrase": self._credentials.passphrase,
                    "timestamp": timestamp,
                    "sign": self._signer.sign(
                        timestamp=timestamp,
                        method="GET",
                        request_path="/users/self/verify",
                    ),
                }
            ],
        }

    def subscribe_ticker(self, inst_id: str, *, request_id: str) -> dict[str, Any]:
        return {
            "id": request_id,
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": inst_id}],
        }

    def subscribe_orders(self, inst_type: str, *, request_id: str) -> dict[str, Any]:
        return {
            "id": request_id,
            "op": "subscribe",
            "args": [{"channel": "orders", "instType": inst_type}],
        }


@dataclass(frozen=True)
class OkxWsHeartbeat:
    idle_timeout: timedelta

    def should_ping(self, *, last_received_at: datetime, now: datetime) -> bool:
        return now - last_received_at > self.idle_timeout

    def is_pong(self, message: str) -> bool:
        return message == "pong"


class OkxOrderChannelParser:
    @staticmethod
    def parse(message: dict[str, Any]) -> OkxOrderChannelEvent:
        arg = message.get("arg", {})
        if arg.get("channel") != "orders":
            raise ValueError("OKX message is not an order channel update")
        data = message.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("OKX order channel update has no data")
        row = data[0]
        return OkxOrderChannelEvent(
            order_id=row.get("ordId", ""),
            client_order_id=row.get("clOrdId", ""),
            inst_id=row["instId"],
            status=OkxOrderMapper.to_domain_status(row.get("state", "")),
            filled_size=Decimal(row.get("accFillSz", "0")),
            fill_price=_optional_decimal(row.get("fillPx", "")),
            fill_time=_optional_millis(row.get("fillTime", "")),
        )


def _optional_decimal(raw: str) -> Decimal | None:
    return Decimal(raw) if raw else None


def _optional_millis(raw: str) -> datetime | None:
    if not raw:
        return None
    return datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
