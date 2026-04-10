from datetime import UTC, datetime
from decimal import Decimal

from quantbot.core.models import MarketSnapshot, OrderSide, OrderType
from quantbot.strategy.breakout import BreakoutConfig, BreakoutStrategy


def test_breakout_strategy_emits_buy_limit_intent_above_lookback_high() -> None:
    strategy = BreakoutStrategy(
        BreakoutConfig(
            inst_id="BTC-USDT",
            lookback=3,
            order_size=Decimal("0.001"),
            limit_offset_bps=Decimal("5"),
            client_order_prefix="botA",
        )
    )
    now = datetime.now(UTC)

    assert strategy.on_market(MarketSnapshot("BTC-USDT", Decimal("100"), now)) is None
    assert strategy.on_market(MarketSnapshot("BTC-USDT", Decimal("101"), now)) is None
    assert strategy.on_market(MarketSnapshot("BTC-USDT", Decimal("102"), now)) is None
    intent = strategy.on_market(MarketSnapshot("BTC-USDT", Decimal("103"), now))

    assert intent is not None
    assert intent.side == OrderSide.BUY
    assert intent.order_type == OrderType.LIMIT
    assert intent.price == Decimal("103.0515")
    assert intent.size == Decimal("0.001")
    assert intent.client_order_id == "botA000001"

