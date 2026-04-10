from datetime import UTC, datetime
from decimal import Decimal

from quantbot.core.models import MarketSnapshot, OrderIntent, OrderSide, OrderType
from quantbot.engine import TradingEngine
from quantbot.exchange.models import InstrumentMetadata
from quantbot.exchange.okx.rest import OkxOrderAck
from quantbot.risk.engine import RiskConfig, RiskEngine


def test_trading_engine_quantizes_and_submits_risk_accepted_intent() -> None:
    submitted: list[OrderIntent] = []

    class StaticStrategy:
        def on_market(self, snapshot: MarketSnapshot) -> OrderIntent:
            return OrderIntent(
                client_order_id="botA0008",
                inst_id=snapshot.inst_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.19"),
                size=Decimal("0.123456"),
            )

    class FakeExecution:
        async def place_order(self, intent: OrderIntent) -> OkxOrderAck:
            submitted.append(intent)
            return OkxOrderAck(order_id="123", client_order_id=intent.client_order_id)

    engine = TradingEngine(
        strategy=StaticStrategy(),
        risk_engine=RiskEngine(RiskConfig(max_order_notional=Decimal("1000"))),
        execution=FakeExecution(),
        instrument=InstrumentMetadata(
            inst_id="BTC-USDT",
            inst_type="SPOT",
            tick_sz=Decimal("0.1"),
            lot_sz=Decimal("0.00001"),
            min_sz=Decimal("0.0001"),
            state="live",
        ),
    )

    import asyncio

    ack = asyncio.run(
        engine.on_market(MarketSnapshot("BTC-USDT", Decimal("100"), datetime.now(UTC)))
    )

    assert ack is not None
    assert submitted[0].price == Decimal("100.1")
    assert submitted[0].size == Decimal("0.12345")


def test_trading_engine_does_not_submit_risk_rejected_intent() -> None:
    class StaticStrategy:
        def on_market(self, snapshot: MarketSnapshot) -> OrderIntent:
            return OrderIntent(
                client_order_id="botA0009",
                inst_id=snapshot.inst_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("200"),
                size=Decimal("1"),
            )

    class FakeExecution:
        async def place_order(self, intent: OrderIntent) -> OkxOrderAck:
            raise AssertionError("risk-rejected order must not be submitted")

    engine = TradingEngine(
        strategy=StaticStrategy(),
        risk_engine=RiskEngine(RiskConfig(max_order_notional=Decimal("10"))),
        execution=FakeExecution(),
        instrument=InstrumentMetadata(
            inst_id="BTC-USDT",
            inst_type="SPOT",
            tick_sz=Decimal("0.1"),
            lot_sz=Decimal("0.00001"),
            min_sz=Decimal("0.0001"),
            state="live",
        ),
    )

    import asyncio

    ack = asyncio.run(
        engine.on_market(MarketSnapshot("BTC-USDT", Decimal("100"), datetime.now(UTC)))
    )

    assert ack is None

