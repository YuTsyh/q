"""Synthetic market data generator for backtesting.

Generates realistic OHLCV and funding rate data with configurable
market regimes (bull, bear, sideways, volatile).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar


@dataclass(frozen=True)
class MarketRegime:
    """Configuration for a market regime segment."""

    name: str
    drift: float  # Daily drift (annualised / 365)
    volatility: float  # Daily volatility
    duration_bars: int


def generate_ohlcv(
    inst_id: str,
    regimes: list[MarketRegime],
    start_price: float = 100.0,
    start_time: datetime | None = None,
    bar_interval: timedelta = timedelta(days=1),
    seed: int | None = 42,
) -> list[OhlcvBar]:
    """Generate synthetic OHLCV bars across market regimes.

    Uses geometric Brownian motion within each regime.
    """
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

    bars: list[OhlcvBar] = []
    price = start_price
    t = start_time

    for regime in regimes:
        for _ in range(regime.duration_bars):
            # GBM: dS = S * (mu*dt + sigma*dW)
            noise = rng.gauss(0, 1)
            ret = regime.drift + regime.volatility * noise
            new_price = price * math.exp(ret)
            new_price = max(new_price, 0.01)

            # Generate OHLCV
            intrabar_vol = regime.volatility * 0.5
            high_mult = 1 + abs(rng.gauss(0, intrabar_vol))
            low_mult = 1 - abs(rng.gauss(0, intrabar_vol))

            bar_open = Decimal(str(round(price, 6)))
            bar_high = Decimal(str(round(max(price, new_price) * high_mult, 6)))
            bar_low = Decimal(str(round(min(price, new_price) * low_mult, 6)))
            bar_close = Decimal(str(round(new_price, 6)))
            bar_vol = Decimal(str(round(rng.uniform(100, 10000), 2)))

            bars.append(OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=bar_close,
                volume=bar_vol,
            ))
            price = new_price
            t += bar_interval

    return bars


def generate_funding_rates(
    inst_id: str,
    n_observations: int,
    start_time: datetime | None = None,
    interval: timedelta = timedelta(hours=8),
    base_rate: float = 0.0001,
    rate_volatility: float = 0.0003,
    seed: int | None = 42,
) -> list[FundingRate]:
    """Generate synthetic funding rate observations."""
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

    rates: list[FundingRate] = []
    t = start_time
    for _ in range(n_observations):
        rate = base_rate + rng.gauss(0, rate_volatility)
        rates.append(FundingRate(
            inst_id=inst_id,
            funding_time=t,
            funding_rate=Decimal(str(round(rate, 8))),
        ))
        t += interval
    return rates


# ---------------------------------------------------------------------------
# Pre-configured regime sets for testing
# ---------------------------------------------------------------------------

BULL_MARKET = MarketRegime("bull", drift=0.001, volatility=0.015, duration_bars=100)
BEAR_MARKET = MarketRegime("bear", drift=-0.001, volatility=0.02, duration_bars=80)
SIDEWAYS_MARKET = MarketRegime("sideways", drift=0.0, volatility=0.01, duration_bars=60)
VOLATILE_MARKET = MarketRegime("volatile", drift=0.0002, volatility=0.04, duration_bars=50)

FULL_CYCLE_REGIMES = [BULL_MARKET, BEAR_MARKET, SIDEWAYS_MARKET, VOLATILE_MARKET, BULL_MARKET]
"""A full market cycle: bull → bear → sideways → volatile → bull."""

MULTI_CYCLE_REGIMES = FULL_CYCLE_REGIMES * 2
"""Two full market cycles for longer backtests (780 bars)."""


def generate_multi_instrument_data(
    inst_ids: list[str],
    regimes: list[MarketRegime] | None = None,
    seed_base: int = 42,
) -> tuple[dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]]:
    """Generate OHLCV and funding data for multiple instruments.

    Each instrument gets slightly different parameters (seeded differently)
    to simulate realistic cross-sectional variation.
    """
    if regimes is None:
        regimes = FULL_CYCLE_REGIMES

    bars_by_inst: dict[str, list[OhlcvBar]] = {}
    funding_by_inst: dict[str, list[FundingRate]] = {}

    total_bars = sum(r.duration_bars for r in regimes)

    for i, inst_id in enumerate(inst_ids):
        seed = seed_base + i * 100
        start_price = 50.0 + i * 30  # Different starting prices

        bars = generate_ohlcv(
            inst_id=inst_id,
            regimes=regimes,
            start_price=start_price,
            seed=seed,
        )
        bars_by_inst[inst_id] = bars

        # 3 funding observations per day (8h interval)
        funding = generate_funding_rates(
            inst_id=inst_id,
            n_observations=total_bars * 3,
            base_rate=0.0001 + i * 0.00005,
            seed=seed + 1,
        )
        funding_by_inst[inst_id] = funding

    return bars_by_inst, funding_by_inst
