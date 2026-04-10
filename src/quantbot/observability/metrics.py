from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


class BotMetrics:
    def __init__(self) -> None:
        self._ws_reconnects: Counter[str] = Counter()
        self._order_rejects: Counter[str] = Counter()
        self._unknown_orders = 0

    def record_ws_reconnect(self, channel: str) -> None:
        self._ws_reconnects[channel] += 1

    def record_order_reject(self, source: str) -> None:
        self._order_rejects[source] += 1

    def record_unknown_order(self) -> None:
        self._unknown_orders += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "ws_reconnects_total": deepcopy(dict(self._ws_reconnects)),
            "order_rejects_total": deepcopy(dict(self._order_rejects)),
            "unknown_orders_total": self._unknown_orders,
        }

