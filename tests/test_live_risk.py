"""Tests for live deployment risk manager."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal


from quantbot.risk.live_risk import (
    DailyPnL,
    LiveRiskConfig,
    LiveRiskManager,
)


class TestLiveRiskManager:
    def _make_manager(self, **kwargs) -> LiveRiskManager:
        config = LiveRiskConfig(**kwargs)
        return LiveRiskManager(config=config, _equity=Decimal("100000"))

    def test_position_size_risk_based(self):
        mgr = self._make_manager()
        result = mgr.compute_position_size(
            price=Decimal("50000"),
            stop_loss_pct=Decimal("0.02"),
        )
        # Risk = 100000 * 0.01 = 1000
        # Size = 1000 / (50000 * 0.02) = 1.0
        assert result.size > 0
        assert result.risk_amount == Decimal("1000")

    def test_position_size_respects_max_position(self):
        mgr = self._make_manager(max_position_pct=Decimal("0.1"))
        result = mgr.compute_position_size(
            price=Decimal("100"),
            stop_loss_pct=Decimal("0.001"),  # Tiny stop → huge size
        )
        # Max position = 100000 * 0.1 = 10000 notional
        assert result.notional <= Decimal("10000")

    def test_position_size_zero_price(self):
        mgr = self._make_manager()
        result = mgr.compute_position_size(
            price=Decimal("0"),
            stop_loss_pct=Decimal("0.02"),
        )
        assert result.size == Decimal("0")

    def test_position_size_zero_stop(self):
        mgr = self._make_manager()
        result = mgr.compute_position_size(
            price=Decimal("100"),
            stop_loss_pct=Decimal("0"),
        )
        assert result.size == Decimal("0")

    def test_order_allowed_by_default(self):
        mgr = self._make_manager()
        allowed, reason = mgr.check_order_allowed()
        assert allowed is True
        assert reason == "allowed"

    def test_kill_switch_blocks_orders(self):
        mgr = self._make_manager()
        mgr.activate_kill_switch("test")
        allowed, reason = mgr.check_order_allowed()
        assert allowed is False
        assert reason == "kill_switch_active"

    def test_kill_switch_deactivation(self):
        mgr = self._make_manager()
        mgr.activate_kill_switch("test")
        mgr.deactivate_kill_switch()
        allowed, _ = mgr.check_order_allowed()
        assert allowed is True

    def test_max_open_orders_limit(self):
        mgr = self._make_manager(max_open_orders=2)
        mgr.record_order_opened()
        mgr.record_order_opened()
        allowed, reason = mgr.check_order_allowed()
        assert allowed is False
        assert reason == "max_open_orders_exceeded"

    def test_order_close_decrements_count(self):
        mgr = self._make_manager(max_open_orders=2)
        mgr.record_order_opened()
        mgr.record_order_opened()
        mgr.record_order_closed()
        allowed, _ = mgr.check_order_allowed()
        assert allowed is True

    def test_drawdown_triggers_kill_switch(self):
        mgr = self._make_manager(max_drawdown_pct=Decimal("0.10"))
        mgr._peak_equity = Decimal("100000")
        mgr.update_equity(Decimal("89000"))  # 11% DD
        assert mgr.is_kill_switch_active is True

    def test_drawdown_below_threshold_ok(self):
        mgr = self._make_manager(max_drawdown_pct=Decimal("0.10"))
        mgr._peak_equity = Decimal("100000")
        mgr.update_equity(Decimal("95000"))  # 5% DD
        assert mgr.is_kill_switch_active is False

    def test_peak_equity_updates_on_new_high(self):
        mgr = self._make_manager()
        mgr.update_equity(Decimal("110000"))
        assert mgr._peak_equity == Decimal("110000")

    def test_record_fill_tracks_pnl(self):
        mgr = self._make_manager()
        mgr.record_fill(Decimal("-500"))
        assert mgr._daily_pnl.realised_pnl == Decimal("-500")

    def test_current_drawdown(self):
        mgr = self._make_manager()
        mgr._peak_equity = Decimal("100000")
        mgr._equity = Decimal("90000")
        assert mgr.current_drawdown == Decimal("0.1")

    def test_lot_sz_floors_position(self):
        mgr = self._make_manager()
        result = mgr.compute_position_size(
            price=Decimal("50000"),
            stop_loss_pct=Decimal("0.02"),
            lot_sz=Decimal("0.01"),
        )
        # Should be floored to 0.01 increments
        remainder = result.size % Decimal("0.01")
        assert remainder == Decimal("0")


class TestDailyPnL:
    def test_total_pnl(self):
        pnl = DailyPnL(
            date=datetime.now(UTC),
            realised_pnl=Decimal("100"),
            unrealised_pnl=Decimal("-30"),
        )
        assert pnl.total_pnl == Decimal("70")
