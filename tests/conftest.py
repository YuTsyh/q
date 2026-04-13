"""Shared fixtures for strategy tests."""

from __future__ import annotations

import pytest

from quantbot.research.backtest import BacktestConfig
from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    MULTI_CYCLE_REGIMES,
    generate_multi_instrument_data,
)


STRATEGY_TEST_INSTRUMENTS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "DOGE-USDT-SWAP",
]


@pytest.fixture
def full_cycle_data():
    """Generate one full market cycle of data for 4 instruments."""
    return generate_multi_instrument_data(
        STRATEGY_TEST_INSTRUMENTS,
        regimes=FULL_CYCLE_REGIMES,
        seed_base=42,
    )


@pytest.fixture
def multi_cycle_data():
    """Generate two full market cycles of data for 4 instruments."""
    return generate_multi_instrument_data(
        STRATEGY_TEST_INSTRUMENTS,
        regimes=MULTI_CYCLE_REGIMES,
        seed_base=42,
    )


@pytest.fixture
def default_backtest_config():
    """Standard backtest configuration."""
    return BacktestConfig(
        initial_equity=100_000.0,
        rebalance_every_n_bars=5,
        taker_fee_rate=0.0005,
        slippage_bps=2.0,
        partial_fill_ratio=1.0,
        periods_per_year=365.0,
    )
