from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.factors import (
    CarryFundingFactor,
    MomentumFactor,
    TrendFactor,
    WeightedFactorScorer,
)


def _bars(closes: list[str]) -> list[OhlcvBar]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        OhlcvBar(
            inst_id="BTC-USDT-SWAP",
            ts=base + timedelta(hours=4 * idx),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=Decimal("10"),
        )
        for idx, close in enumerate(closes)
    ]


def test_price_factors_compute_momentum_and_trend() -> None:
    bars = _bars(["100", "105", "110", "120"])

    assert MomentumFactor(lookback=3).compute(bars, []) == Decimal("0.20")
    assert TrendFactor(lookback=3).compute(bars, []) == Decimal(
        "0.074626865671641791044776119"
    )


def test_carry_factor_prefers_lower_funding_for_long_perp_book() -> None:
    funding = [
        FundingRate(
            inst_id="BTC-USDT-SWAP",
            funding_time=datetime(2026, 1, 1, tzinfo=UTC),
            funding_rate=Decimal("0.0003"),
        ),
        FundingRate(
            inst_id="BTC-USDT-SWAP",
            funding_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
            funding_rate=Decimal("0.0001"),
        ),
    ]

    assert CarryFundingFactor(lookback=2).compute([], funding) == Decimal("-0.0002")


def test_weighted_factor_scorer_combines_cross_sectional_ranks() -> None:
    scorer = WeightedFactorScorer(weights={"momentum": Decimal("0.7"), "carry": Decimal("0.3")})

    scores = scorer.score(
        {
            "BTC-USDT-SWAP": {"momentum": Decimal("0.2"), "carry": Decimal("-0.0001")},
            "ETH-USDT-SWAP": {"momentum": Decimal("0.1"), "carry": Decimal("-0.0005")},
        }
    )

    assert scores["BTC-USDT-SWAP"] > scores["ETH-USDT-SWAP"]
