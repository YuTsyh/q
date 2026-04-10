from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantbot.core.models import MarketSnapshot, OrderIntent


@dataclass(frozen=True)
class RiskConfig:
    max_order_notional: Decimal = Decimal("100")
    max_position_notional: Decimal = Decimal("500")
    max_daily_loss: Decimal = Decimal("50")
    max_market_data_age: timedelta = timedelta(seconds=5)
    price_band_bps: Decimal = Decimal("100")
    kill_switch_enabled: bool = False


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reason: str = "accepted"


@dataclass
class RiskEngine:
    config: RiskConfig
    _seen_client_order_ids: set[str] = field(default_factory=set)

    def evaluate(self, intent: OrderIntent, market: MarketSnapshot) -> RiskDecision:
        if self.config.kill_switch_enabled:
            return RiskDecision(False, "kill_switch_enabled")
        if intent.client_order_id in self._seen_client_order_ids:
            return RiskDecision(False, "duplicate_client_order_id")
        if _is_stale(market, self.config.max_market_data_age):
            return RiskDecision(False, "stale_market_data")
        if intent.notional > self.config.max_order_notional:
            return RiskDecision(False, "max_order_notional_exceeded")
        if _price_band_breached(intent, market, self.config.price_band_bps):
            return RiskDecision(False, "price_band_violation")

        self._seen_client_order_ids.add(intent.client_order_id)
        return RiskDecision(True)


def _is_stale(market: MarketSnapshot, max_age: timedelta) -> bool:
    return datetime.now(UTC) - market.received_at > max_age


def _price_band_breached(
    intent: OrderIntent,
    market: MarketSnapshot,
    price_band_bps: Decimal,
) -> bool:
    if intent.price is None:
        return False
    max_deviation = market.last_price * price_band_bps / Decimal("10000")
    return abs(intent.price - market.last_price) > max_deviation

