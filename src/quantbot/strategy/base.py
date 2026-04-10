from __future__ import annotations

from typing import Protocol

from quantbot.core.models import MarketSnapshot, OrderIntent


class Strategy(Protocol):
    def on_market(self, snapshot: MarketSnapshot) -> OrderIntent | None:
        """Consume market data and optionally return a single order intent."""

