from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantbot.core.models import MarketSnapshot


class OkxTickerParser:
    @staticmethod
    def parse(message: dict[str, Any]) -> MarketSnapshot:
        if "data" not in message or not message["data"]:
            raise ValueError("OKX message does not contain ticker data")
        arg = message.get("arg", {})
        if arg.get("channel") != "tickers":
            raise ValueError("OKX message is not ticker data")

        row = message["data"][0]
        timestamp_ms = int(row["ts"])
        return MarketSnapshot(
            inst_id=row["instId"],
            last_price=Decimal(row["last"]),
            received_at=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
        )

