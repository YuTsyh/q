"""Phase 1 – Framework Audit & Reality Enforcement.

Validates that the backtesting framework models OKX Spot live-trading
frictions correctly:
 1. Exact OKX Spot Maker/Taker fees deducted.
 2. Non-linear Almgren-Chriss market impact (square-root law).
 3. Volume participation capped at 5 % of bar volume.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantbot.research.market_impact import (
    EnhancedExecutionSimulator,
    MarketImpactConfig,
    compute_market_impact,
)
from quantbot.research.backtest import BacktestConfig, BacktestEngine

_ZERO = Decimal("0")


# ---------------------------------------------------------------------------
# 1. OKX Spot fee rates
# ---------------------------------------------------------------------------

class TestOKXFeeRates:
    """Ensure the default config uses exact OKX Spot fee rates."""

    def test_default_market_impact_config_fees(self):
        cfg = MarketImpactConfig()
        # OKX Spot: Maker 0.02 %, Taker 0.05 %
        assert cfg.maker_fee_rate == Decimal("0.0002")
        assert cfg.taker_fee_rate == Decimal("0.0005")

    def test_backtest_config_default_taker_fee(self):
        cfg = BacktestConfig()
        assert cfg.taker_fee_rate == 0.0005

    def test_fees_deducted_on_trade(self):
        """A round-trip trade must pay taker fees proportional to notional."""
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        assert result.taker_fee > _ZERO
        # Taker fee = |notional| * taker_rate
        expected_taker = Decimal("10000") * Decimal("0.0005")
        assert result.taker_fee == expected_taker.quantize(Decimal("0.00000001"))


# ---------------------------------------------------------------------------
# 2. Non-linear market impact (Almgren-Chriss)
# ---------------------------------------------------------------------------

class TestAlmgrenChrissImpact:
    """Verify square-root impact law and its properties."""

    def test_impact_increases_with_size(self):
        cfg = MarketImpactConfig()
        small = compute_market_impact(
            trade_notional=Decimal("1000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        large = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        assert abs(large.total_slippage) > abs(small.total_slippage)

    def test_impact_exponent_is_half(self):
        """Default impact exponent must be 0.5 (square-root law)."""
        cfg = MarketImpactConfig()
        assert cfg.impact_exponent == 0.5

    def test_impact_sub_linear(self):
        """10× notional should produce < 10× slippage (concavity)."""
        cfg = MarketImpactConfig()
        base = compute_market_impact(
            trade_notional=Decimal("1000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        big = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        ratio = abs(big.total_slippage) / abs(base.total_slippage)
        assert ratio < 10.0, "Impact should be sub-linear (square-root law)"

    def test_zero_volume_returns_zero_impact(self):
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("1000"),
            price=Decimal("50000"),
            bar_volume=_ZERO,
            adv=_ZERO,
            config=cfg,
        )
        assert result.total_slippage == _ZERO

    def test_buy_impact_positive_sell_negative(self):
        """Buys should push price up, sells should push price down."""
        cfg = MarketImpactConfig()
        buy = compute_market_impact(
            trade_notional=Decimal("5000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        sell = compute_market_impact(
            trade_notional=Decimal("-5000"),
            price=Decimal("50000"),
            bar_volume=Decimal("1000000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        assert buy.total_slippage > _ZERO
        assert sell.total_slippage < _ZERO


# ---------------------------------------------------------------------------
# 3. Volume participation cap (5 % of bar volume)
# ---------------------------------------------------------------------------

class TestVolumeCap:
    """Ensure that order volume cannot exceed 5 % of bar volume."""

    def test_default_max_participation_rate(self):
        cfg = MarketImpactConfig()
        assert cfg.max_volume_participation == 0.05

    def test_large_order_is_capped(self):
        """An order consuming > 5 % of bar volume is flagged as capped."""
        cfg = MarketImpactConfig()
        # bar_volume = 100 units, price = 1000  → bar notional = 100_000
        # trade_notional = 10_000 → 10 % participation → must be capped
        result = compute_market_impact(
            trade_notional=Decimal("10000"),
            price=Decimal("1000"),
            bar_volume=Decimal("100"),
            adv=Decimal("1000"),
            config=cfg,
        )
        assert result.was_capped is True
        assert result.volume_participation_rate == 0.05

    def test_small_order_not_capped(self):
        """An order within 5 % should NOT be capped."""
        cfg = MarketImpactConfig()
        result = compute_market_impact(
            trade_notional=Decimal("100"),
            price=Decimal("1000"),
            bar_volume=Decimal("100000"),
            adv=Decimal("500000"),
            config=cfg,
        )
        assert result.was_capped is False


# ---------------------------------------------------------------------------
# 4. BacktestEngine uses enhanced simulator by default
# ---------------------------------------------------------------------------

class TestBacktestEngineDefaults:
    """Ensure the engine defaults to Almgren-Chriss, not flat BPS."""

    def test_default_uses_market_impact(self):
        cfg = BacktestConfig()
        assert cfg.use_market_impact is True

    def test_engine_creates_enhanced_sim(self):
        cfg = BacktestConfig()
        engine = BacktestEngine(cfg)
        assert engine._enhanced_sim is not None
        assert engine._legacy_sim is None

    def test_legacy_mode_available(self):
        cfg = BacktestConfig(use_market_impact=False)
        engine = BacktestEngine(cfg)
        assert engine._enhanced_sim is None
        assert engine._legacy_sim is not None


# ---------------------------------------------------------------------------
# 5. EnhancedExecutionSimulator integration
# ---------------------------------------------------------------------------

class TestEnhancedSimulatorRebalance:
    """Validate the rebalance method on the enhanced simulator."""

    def test_rebalance_deducts_costs(self):
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={},
            target_weights={"BTC-USDT": Decimal("0.5")},
            prices={"BTC-USDT": Decimal("50000")},
            volumes={"BTC-USDT": Decimal("500000")},
            adv={"BTC-USDT": Decimal("1000000")},
        )
        assert len(fills) == 1
        fill = fills[0]
        assert fill.total_cost > _ZERO
        assert fill.fee > _ZERO
        assert fill.slippage != _ZERO

    def test_no_trade_when_weights_unchanged(self):
        cfg = MarketImpactConfig()
        sim = EnhancedExecutionSimulator(cfg)
        fills = sim.rebalance(
            equity=Decimal("100000"),
            current_weights={"BTC-USDT": Decimal("0.5")},
            target_weights={"BTC-USDT": Decimal("0.5")},
            prices={"BTC-USDT": Decimal("50000")},
            volumes={"BTC-USDT": Decimal("500000")},
            adv={"BTC-USDT": Decimal("1000000")},
        )
        assert len(fills) == 0
