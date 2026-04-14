"""Tests for the risk overlay module."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal


from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.strategy.risk_overlay import (
    RiskOverlay,
    RiskOverlayConfig,
    with_risk_overlay,
)


def _make_bars(
    inst_id: str, n: int, start_price: float = 100.0, daily_ret: float = 0.001
) -> list[OhlcvBar]:
    """Generate simple synthetic bars."""
    bars = []
    price = start_price
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        p = Decimal(str(round(price, 6)))
        bars.append(
            OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=p,
                high=p * Decimal("1.01"),
                low=p * Decimal("0.99"),
                close=p,
                volume=Decimal("1000"),
            )
        )
        price *= 1 + daily_ret
        t += timedelta(days=1)
    return bars


def _make_crash_bars(
    inst_id: str, n: int, start_price: float = 100.0, crash_at: int = 50
) -> list[OhlcvBar]:
    """Generate bars with a crash at a specific point."""
    bars = []
    price = start_price
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        if i == crash_at:
            price *= 0.8  # 20% crash
        p = Decimal(str(round(price, 6)))
        bars.append(
            OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=p,
                high=p * Decimal("1.01"),
                low=p * Decimal("0.99"),
                close=p,
                volume=Decimal("1000"),
            )
        )
        price *= 1.001
        t += timedelta(days=1)
    return bars


class TestRiskOverlay:
    def test_empty_weights_returns_empty(self):
        overlay = RiskOverlay()
        result = overlay.apply({}, {"BTC": _make_bars("BTC", 60)})
        assert result == {}

    def test_caps_per_instrument(self):
        config = RiskOverlayConfig(max_per_instrument=0.10)
        overlay = RiskOverlay(config)
        raw = {"BTC": Decimal("0.5"), "ETH": Decimal("0.3")}
        bars = {
            "BTC": _make_bars("BTC", 60),
            "ETH": _make_bars("ETH", 60),
        }
        result = overlay.apply(raw, bars)
        for v in result.values():
            assert float(v) <= 0.10 + 1e-6

    def test_caps_gross_exposure(self):
        config = RiskOverlayConfig(max_gross_exposure=0.5, max_per_instrument=0.3)
        overlay = RiskOverlay(config)
        raw = {
            "BTC": Decimal("0.3"),
            "ETH": Decimal("0.3"),
            "SOL": Decimal("0.3"),
        }
        bars = {
            "BTC": _make_bars("BTC", 60),
            "ETH": _make_bars("ETH", 60),
            "SOL": _make_bars("SOL", 60),
        }
        result = overlay.apply(raw, bars)
        total = sum(float(v) for v in result.values())
        assert total <= 0.5 + 1e-6

    def test_crash_guard_excludes_crashed(self):
        config = RiskOverlayConfig(
            crash_guard_lookback=5,
            crash_guard_threshold=-0.05,
            regime_gating=False,
        )
        overlay = RiskOverlay(config)
        # BTC has a recent 20% crash
        btc_bars = _make_crash_bars("BTC", 60, crash_at=55)
        eth_bars = _make_bars("ETH", 60)
        raw = {"BTC": Decimal("0.3"), "ETH": Decimal("0.3")}
        result = overlay.apply(raw, {"BTC": btc_bars, "ETH": eth_bars})
        # BTC should be excluded
        assert "BTC" not in result or float(result.get("BTC", Decimal("0"))) == 0

    def test_non_negative_weights(self):
        overlay = RiskOverlay()
        raw = {"BTC": Decimal("0.2"), "ETH": Decimal("0.1")}
        bars = {
            "BTC": _make_bars("BTC", 60),
            "ETH": _make_bars("ETH", 60),
        }
        result = overlay.apply(raw, bars)
        for v in result.values():
            assert float(v) >= 0


class TestWithRiskOverlay:
    def test_wraps_allocator(self):
        def dummy_allocator(bars, funding):
            return {"BTC": Decimal("0.5")}

        wrapped = with_risk_overlay(dummy_allocator)
        bars = {"BTC": _make_bars("BTC", 60)}
        funding: dict[str, list[FundingRate]] = {}
        result = wrapped(bars, funding)
        assert isinstance(result, dict)
        # Weight should be capped by overlay
        if "BTC" in result:
            assert float(result["BTC"]) <= 0.20 + 1e-6

    def test_overlay_preserves_empty_from_strategy(self):
        def empty_allocator(bars, funding):
            return {}

        wrapped = with_risk_overlay(empty_allocator)
        bars = {"BTC": _make_bars("BTC", 60)}
        funding: dict[str, list[FundingRate]] = {}
        result = wrapped(bars, funding)
        assert result == {}
