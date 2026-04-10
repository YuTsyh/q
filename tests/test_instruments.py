from decimal import Decimal

import pytest

from quantbot.exchange.models import InstrumentMetadata


def test_instrument_quantizes_price_and_size_down_to_okx_steps() -> None:
    instrument = InstrumentMetadata(
        inst_id="BTC-USDT",
        inst_type="SPOT",
        tick_sz=Decimal("0.1"),
        lot_sz=Decimal("0.00001"),
        min_sz=Decimal("0.0001"),
        state="live",
    )

    assert instrument.quantize_price(Decimal("65000.19")) == Decimal("65000.1")
    assert instrument.quantize_size(Decimal("0.12345678")) == Decimal("0.12345")


def test_instrument_rejects_size_below_minimum() -> None:
    instrument = InstrumentMetadata(
        inst_id="BTC-USDT",
        inst_type="SPOT",
        tick_sz=Decimal("0.1"),
        lot_sz=Decimal("0.00001"),
        min_sz=Decimal("0.0001"),
        state="live",
    )

    with pytest.raises(ValueError, match="below minimum"):
        instrument.validate_size(Decimal("0.00009"))

