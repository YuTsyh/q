from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantbot.core.models import MarketSnapshot, OrderIntent, OrderSide, OrderType
from quantbot.risk.engine import RiskConfig, RiskEngine


def _intent(client_order_id: str = "botA0001") -> OrderIntent:
    return OrderIntent(
        client_order_id=client_order_id,
        inst_id="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        size=Decimal("0.1"),
    )


def test_risk_accepts_small_fresh_order_once() -> None:
    engine = RiskEngine(RiskConfig(max_order_notional=Decimal("50")))
    snapshot = MarketSnapshot(
        inst_id="BTC-USDT",
        last_price=Decimal("100"),
        received_at=datetime.now(UTC),
    )

    decision = engine.evaluate(_intent(), snapshot)

    assert decision.accepted is True


def test_risk_rejects_duplicate_client_order_id() -> None:
    engine = RiskEngine(RiskConfig(max_order_notional=Decimal("50")))
    snapshot = MarketSnapshot(
        inst_id="BTC-USDT",
        last_price=Decimal("100"),
        received_at=datetime.now(UTC),
    )

    assert engine.evaluate(_intent("dup0001"), snapshot).accepted is True
    duplicate = engine.evaluate(_intent("dup0001"), snapshot)

    assert duplicate.accepted is False
    assert duplicate.reason == "duplicate_client_order_id"


def test_risk_rejects_stale_market_data() -> None:
    engine = RiskEngine(RiskConfig(max_market_data_age=timedelta(seconds=2)))
    snapshot = MarketSnapshot(
        inst_id="BTC-USDT",
        last_price=Decimal("100"),
        received_at=datetime.now(UTC) - timedelta(seconds=5),
    )

    decision = engine.evaluate(_intent(), snapshot)

    assert decision.accepted is False
    assert decision.reason == "stale_market_data"


def test_risk_rejects_price_band_violation() -> None:
    engine = RiskEngine(RiskConfig(price_band_bps=Decimal("100")))
    snapshot = MarketSnapshot(
        inst_id="BTC-USDT",
        last_price=Decimal("100"),
        received_at=datetime.now(UTC),
    )

    decision = engine.evaluate(
        OrderIntent(
            client_order_id="band0001",
            inst_id="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("102"),
            size=Decimal("0.1"),
        ),
        snapshot,
    )

    assert decision.accepted is False
    assert decision.reason == "price_band_violation"

