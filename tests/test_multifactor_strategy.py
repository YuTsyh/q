from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.multifactor import MultiFactorPerpStrategy


def _bars(inst_id: str, closes: list[str]) -> list[OhlcvBar]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        OhlcvBar(
            inst_id=inst_id,
            ts=base + timedelta(days=idx),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=Decimal("10"),
        )
        for idx, close in enumerate(closes)
    ]


def _funding(inst_id: str, rates: list[str]) -> list[FundingRate]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        FundingRate(
            inst_id=inst_id,
            funding_time=base + timedelta(hours=8 * idx),
            funding_rate=Decimal(rate),
        )
        for idx, rate in enumerate(rates)
    ]


def test_multifactor_perp_strategy_outputs_top_ranked_target_weights() -> None:
    strategy = MultiFactorPerpStrategy.default_low_frequency(
        top_n=1,
        gross_exposure=Decimal("0.5"),
    )

    weights = strategy.allocate(
        bars_by_instrument={
            "BTC-USDT-SWAP": _bars("BTC-USDT-SWAP", ["100", "105", "110", "120"]),
            "ETH-USDT-SWAP": _bars("ETH-USDT-SWAP", ["100", "99", "98", "97"]),
        },
        funding_by_instrument={
            "BTC-USDT-SWAP": _funding("BTC-USDT-SWAP", ["0.0001", "0.0001"]),
            "ETH-USDT-SWAP": _funding("ETH-USDT-SWAP", ["0.0005", "0.0005"]),
        },
    )

    assert weights == {"BTC-USDT-SWAP": Decimal("0.5")}

