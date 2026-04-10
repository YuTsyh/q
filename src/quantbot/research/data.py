from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class OhlcvBar:
    inst_id: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class FundingRate:
    inst_id: str
    funding_time: datetime
    funding_rate: Decimal


@dataclass(frozen=True)
class ResearchInstrument:
    inst_id: str
    inst_type: str
    quote_ccy: str
    state: str
    tick_sz: Decimal
    lot_sz: Decimal
    min_sz: Decimal
    min_notional: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    quote_volume_24h: Decimal


@dataclass(frozen=True)
class UniverseFilter:
    quote_ccy: str
    min_quote_volume_24h: Decimal
    max_symbols: int

    def select(self, instruments: list[ResearchInstrument]) -> list[ResearchInstrument]:
        eligible = [
            instrument
            for instrument in instruments
            if instrument.inst_type == "SWAP"
            and instrument.quote_ccy == self.quote_ccy
            and instrument.state == "live"
            and instrument.inst_id.endswith(f"-{self.quote_ccy}-SWAP")
            and instrument.quote_volume_24h >= self.min_quote_volume_24h
        ]
        return sorted(eligible, key=lambda item: item.quote_volume_24h, reverse=True)[
            : self.max_symbols
        ]

