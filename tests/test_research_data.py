from datetime import UTC, datetime
from decimal import Decimal

from quantbot.research.data import (
    FundingRate,
    OhlcvBar,
    ResearchInstrument,
    UniverseFilter,
)


def test_universe_filter_selects_liquid_live_usdt_perpetuals() -> None:
    instruments = [
        ResearchInstrument(
            inst_id="BTC-USDT-SWAP",
            inst_type="SWAP",
            quote_ccy="USDT",
            state="live",
            tick_sz=Decimal("0.1"),
            lot_sz=Decimal("0.01"),
            min_sz=Decimal("0.01"),
            min_notional=Decimal("5"),
            maker_fee_rate=Decimal("0.0002"),
            taker_fee_rate=Decimal("0.0005"),
            quote_volume_24h=Decimal("100000000"),
        ),
        ResearchInstrument(
            inst_id="ILLQ-USDT-SWAP",
            inst_type="SWAP",
            quote_ccy="USDT",
            state="live",
            tick_sz=Decimal("0.001"),
            lot_sz=Decimal("1"),
            min_sz=Decimal("1"),
            min_notional=Decimal("5"),
            maker_fee_rate=Decimal("0.0002"),
            taker_fee_rate=Decimal("0.0005"),
            quote_volume_24h=Decimal("1000"),
        ),
        ResearchInstrument(
            inst_id="ETH-USDC-SWAP",
            inst_type="SWAP",
            quote_ccy="USDC",
            state="live",
            tick_sz=Decimal("0.01"),
            lot_sz=Decimal("0.01"),
            min_sz=Decimal("0.01"),
            min_notional=Decimal("5"),
            maker_fee_rate=Decimal("0.0002"),
            taker_fee_rate=Decimal("0.0005"),
            quote_volume_24h=Decimal("90000000"),
        ),
    ]

    selected = UniverseFilter(
        quote_ccy="USDT",
        min_quote_volume_24h=Decimal("1000000"),
        max_symbols=10,
    ).select(instruments)

    assert [item.inst_id for item in selected] == ["BTC-USDT-SWAP"]


def test_ohlcv_and_funding_records_are_timestamped_in_utc() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    bar = OhlcvBar(
        inst_id="BTC-USDT-SWAP",
        ts=ts,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10"),
    )
    funding = FundingRate(
        inst_id="BTC-USDT-SWAP",
        funding_time=ts,
        funding_rate=Decimal("0.0001"),
    )

    assert bar.ts.tzinfo is UTC
    assert funding.funding_time.tzinfo is UTC

