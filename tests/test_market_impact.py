"""Tests for the non-linear market impact model and enhanced execution simulator."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.market_impact import (
    EnhancedExecutionSimulator,
    EnhancedSimulatedFill,
    MarketImpactConfig,
    MarketImpactResult,
    compute_adv,
    compute_market_impact,
)


# ---------------------------------------------------------------------------
# Inline data generators
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")


def _make_bars(
    n: int,
    drift: float = 0.001,
    vol: float = 0.02,
    start_price: float = 100.0,
    seed: int = 42,
) -> list[OhlcvBar]:
    rng = random.Random(seed)
    bars: list[OhlcvBar] = []
    price = start_price
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        ret = drift + vol * rng.gauss(0, 1)
        new_price = price * math.exp(ret)
        bars.append(
            OhlcvBar(
                inst_id="TEST-USDT",
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(max(price, new_price) * 1.005, 6))),
                low=Decimal(str(round(min(price, new_price) * 0.995, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal(str(round(rng.uniform(1000, 10000), 2))),
            )
        )
        price = new_price
        t += timedelta(days=1)
    return bars


# ---------------------------------------------------------------------------
# Tests for compute_market_impact
# ---------------------------------------------------------------------------


class TestComputeMarketImpact:
    def test_normal_buy(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        assert isinstance(result, MarketImpactResult)
        assert result.temporary_impact > _ZERO
        assert result.permanent_impact > _ZERO
        assert result.total_slippage > _ZERO
        assert result.effective_price > Decimal("100")
        assert result.taker_fee > _ZERO
        assert not result.was_capped

    def test_normal_sell(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("-10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        assert result.temporary_impact < _ZERO
        assert result.effective_price < Decimal("100")

    def test_volume_participation_capping(self) -> None:
        cfg = MarketImpactConfig(max_volume_participation=0.05)
        # Trade that is 10% of volume (above 5% cap)
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("1000"),  # volume_notional = 100 * 1000 = 100k
            adv=Decimal("10000"),
            config=cfg,
        )
        assert result.was_capped
        assert result.volume_participation_rate == pytest.approx(0.05)

    def test_zero_volume_returns_zero_impact(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("0"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        assert result.total_slippage == _ZERO
        assert result.effective_price == Decimal("100")
        assert not result.was_capped

    def test_zero_adv_returns_zero_impact(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("0"),
            config=cfg,
        )
        assert result.total_slippage == _ZERO

    def test_zero_notional_returns_zero_impact(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("0"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        assert result.total_slippage == _ZERO
        assert result.total_cost == _ZERO

    def test_negative_price_raises(self) -> None:
        cfg = MarketImpactConfig()
        with pytest.raises(ValueError, match="price must be positive"):
            compute_market_impact(
                trade_notional=Decimal("10000"),
                price=Decimal("-1"),
                bar_volume=Decimal("500000"),
                adv=Decimal("1000000"),
                config=cfg,
            )

    def test_square_root_law_sublinear(self) -> None:
        """Larger trades have larger but sub-linear temporary impact."""
        cfg = MarketImpactConfig()
        small = compute_market_impact(
            trade_notional=Decimal("1000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        large = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
        )
        # Impact grows but less than linearly (sqrt law)
        assert large.temporary_impact > small.temporary_impact
        ratio_notional = 10000 / 1000
        ratio_impact = float(large.temporary_impact / small.temporary_impact)
        assert ratio_impact < ratio_notional  # sub-linear

    def test_funding_cost_calculation(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
            funding_rate=Decimal("0.0001"),
            hours_held=24.0,
        )
        assert result.funding_cost > _ZERO
        # 24h / 8h = 3 intervals; cost = 10000 * 0.0001 * 3 = 3
        assert result.funding_cost == Decimal("3.00000000")

    def test_no_funding_when_zero_hours(self) -> None:
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("100"),
            bar_volume=Decimal("500000"),
            adv=Decimal("1000000"),
            config=cfg,
            funding_rate=Decimal("0.0001"),
            hours_held=0.0,
        )
        assert result.funding_cost == _ZERO


# ---------------------------------------------------------------------------
# Tests for compute_adv
# ---------------------------------------------------------------------------


class TestComputeAdv:
    def test_normal_bars(self) -> None:
        bars = _make_bars(30)
        adv = compute_adv(bars, lookback=20)
        assert isinstance(adv, Decimal)
        assert adv > _ZERO

    def test_empty_bars(self) -> None:
        assert compute_adv([], lookback=20) == _ZERO

    def test_fewer_bars_than_lookback(self) -> None:
        bars = _make_bars(5)
        adv = compute_adv(bars, lookback=20)
        assert adv > _ZERO  # uses all 5 bars

    def test_single_bar(self) -> None:
        bars = _make_bars(1)
        adv = compute_adv(bars, lookback=20)
        assert adv == bars[0].volume.quantize(Decimal("0.00000001"))


# ---------------------------------------------------------------------------
# Tests for EnhancedExecutionSimulator
# ---------------------------------------------------------------------------


class TestEnhancedExecutionSimulator:
    def test_rebalance_produces_fills(self) -> None:
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={"BTC": Decimal("0.5")},
            target_weights={"BTC": Decimal("0.6"), "ETH": Decimal("0.4")},
            prices={"BTC": Decimal("50000"), "ETH": Decimal("3000")},
            volumes={"BTC": Decimal("1000000"), "ETH": Decimal("500000")},
            adv={"BTC": Decimal("5000000"), "ETH": Decimal("2000000")},
        )
        assert len(fills) == 2
        for f in fills:
            assert isinstance(f, EnhancedSimulatedFill)

    def test_no_change_no_fills(self) -> None:
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={"BTC": Decimal("0.5")},
            target_weights={"BTC": Decimal("0.5")},
            prices={"BTC": Decimal("50000")},
            volumes={"BTC": Decimal("1000000")},
            adv={"BTC": Decimal("5000000")},
        )
        assert len(fills) == 0

    def test_rebalance_with_funding_rates(self) -> None:
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={"BTC": Decimal("0.3")},
            target_weights={"BTC": Decimal("0.5")},
            prices={"BTC": Decimal("50000")},
            volumes={"BTC": Decimal("1000000")},
            adv={"BTC": Decimal("5000000")},
            funding_rates={"BTC": Decimal("0.0001")},
            hours_held=16.0,
        )
        assert len(fills) == 1
        fill = fills[0]
        assert fill.funding_cost > _ZERO  # should have held-position funding

    def test_fill_cost_positive(self) -> None:
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={},
            target_weights={"BTC": Decimal("0.5")},
            prices={"BTC": Decimal("50000")},
            volumes={"BTC": Decimal("1000000")},
            adv={"BTC": Decimal("5000000")},
        )
        assert len(fills) == 1
        assert fills[0].total_cost > _ZERO
        assert fills[0].fee > _ZERO
