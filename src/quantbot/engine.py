from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from quantbot.core.models import MarketSnapshot, OrderIntent
from quantbot.exchange.models import InstrumentMetadata
from quantbot.exchange.okx.rest import OkxOrderAck
from quantbot.risk.engine import RiskEngine
from quantbot.strategy.base import Strategy


class ExecutionClient(Protocol):
    async def place_order(self, intent: OrderIntent) -> OkxOrderAck:
        """Submit one order intent."""


class TradingEngine:
    def __init__(
        self,
        *,
        strategy: Strategy,
        risk_engine: RiskEngine,
        execution: ExecutionClient,
        instrument: InstrumentMetadata,
    ) -> None:
        self._strategy = strategy
        self._risk_engine = risk_engine
        self._execution = execution
        self._instrument = instrument

    async def on_market(self, snapshot: MarketSnapshot) -> OkxOrderAck | None:
        intent = self._strategy.on_market(snapshot)
        if intent is None:
            return None
        quantized = replace(
            intent,
            price=self._instrument.quantize_price(intent.price) if intent.price is not None else None,
            size=self._instrument.quantize_size(intent.size),
        )
        self._instrument.validate_size(quantized.size)
        decision = self._risk_engine.evaluate(quantized, snapshot)
        if not decision.accepted:
            return None
        return await self._execution.place_order(quantized)

