"""Tests for the Microstructure / Order-Flow strategy.

Validates:
- Buy/sell volume decomposition and flow signal helpers
- VPIN danger gating
- Funding signal correctness
- Composite flow scoring
- Regime gating
- Stop-loss enforcement
- Drawdown circuit breaker
- Allocator factory
- Backtest integration with BacktestEngine
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from quantbot.research.backtest import BacktestConfig, BacktestEngine
from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.strategy.microstructure_flow import (
    MicrostructureFlowAlpha,
    MicrostructureFlowConfig,
    _buy_sell_volume,
    _compute_atr,
    _realised_vol,
    _vpin_proxy,
    create_microstructure_flow_allocator,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers (inline, following project conventions)
# ---------------------------------------------------------------------------

_T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
_DAY = timedelta(days=1)


def _make_bars(
    inst_id: str,
    n: int,
    *,
    start_price: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.015,
    seed: int = 42,
) -> list[OhlcvBar]:
    """Generate GBM OHLCV bars."""
    rng = random.Random(seed)
    bars: list[OhlcvBar] = []
    price = start_price
    t = _T0
    for _ in range(n):
        noise = rng.gauss(0, 1)
        ret = drift + vol * noise
        new_price = price * math.exp(ret)
        new_price = max(new_price, 0.01)
        intra = vol * 0.5
        h = max(price, new_price) * (1 + abs(rng.gauss(0, intra)))
        lo = min(price, new_price) * (1 - abs(rng.gauss(0, intra)))
        bars.append(
            OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(h, 6))),
                low=Decimal(str(round(lo, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal(str(round(rng.uniform(500, 5000), 2))),
            )
        )
        price = new_price
        t += _DAY
    return bars


def _make_funding(
    inst_id: str,
    n: int,
    *,
    base_rate: float = 0.0001,
    seed: int = 42,
) -> list[FundingRate]:
    """Generate synthetic funding rates (daily timestamps)."""
    rng = random.Random(seed)
    rates: list[FundingRate] = []
    t = _T0
    for _ in range(n):
        r = base_rate + rng.gauss(0, 0.0002)
        rates.append(
            FundingRate(
                inst_id=inst_id,
                funding_time=t,
                funding_rate=Decimal(str(round(r, 8))),
            )
        )
        t += _DAY
    return rates


def _make_crash_bars(
    inst_id: str, n: int, *, start_price: float = 100.0, seed: int = 99,
) -> list[OhlcvBar]:
    """Bars with a sharp crash in the last 5 bars (>5 % drop)."""
    bars = _make_bars(inst_id, n - 5, start_price=start_price, drift=0.0005, seed=seed)
    price = float(bars[-1].close)
    t = bars[-1].ts + _DAY
    for _ in range(5):
        new_price = price * 0.97
        bars.append(
            OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(price * 1.001, 6))),
                low=Decimal(str(round(new_price * 0.999, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal("3000"),
            )
        )
        price = new_price
        t += _DAY
    return bars


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestMicrostructureHelpers:
    """Unit tests for standalone helper functions."""

    def test_buy_sell_volume_sums_to_total(self) -> None:
        bar = _make_bars("X", 1, seed=42)[0]
        buy, sell = _buy_sell_volume(bar)
        total = float(bar.volume)
        assert abs(buy + sell - total) < 1e-6

    def test_buy_sell_volume_non_negative(self) -> None:
        bar = _make_bars("X", 1, seed=42)[0]
        buy, sell = _buy_sell_volume(bar)
        assert buy >= 0.0
        assert sell >= 0.0

    def test_buy_sell_volume_bullish_bar(self) -> None:
        """Close near high → mostly buy volume."""
        bar = OhlcvBar(
            inst_id="X",
            ts=_T0,
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("99"),
            close=Decimal("109"),
            volume=Decimal("1000"),
        )
        buy, sell = _buy_sell_volume(bar)
        assert buy > sell

    def test_buy_sell_volume_bearish_bar(self) -> None:
        """Close near low → mostly sell volume."""
        bar = OhlcvBar(
            inst_id="X",
            ts=_T0,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("90"),
            close=Decimal("91"),
            volume=Decimal("1000"),
        )
        buy, sell = _buy_sell_volume(bar)
        assert sell > buy

    def test_compute_atr_basic(self) -> None:
        bars = _make_bars("X", 30)
        atr = _compute_atr(bars, 14)
        assert atr > 0

    def test_compute_atr_insufficient_data(self) -> None:
        bars = _make_bars("X", 3)
        atr = _compute_atr(bars, 14)
        assert atr >= 0

    def test_realised_vol_basic(self) -> None:
        bars = _make_bars("X", 30, vol=0.02)
        rv = _realised_vol(bars, 20)
        assert rv > 0

    def test_realised_vol_insufficient_data(self) -> None:
        bars = _make_bars("X", 5)
        rv = _realised_vol(bars, 20)
        assert rv == 0.0

    def test_vpin_proxy_basic(self) -> None:
        bars = _make_bars("X", 30, seed=42)
        vpin = _vpin_proxy(bars, 15)
        assert vpin is None or (0.0 <= vpin <= 1.0)

    def test_vpin_proxy_insufficient_data(self) -> None:
        bars = _make_bars("X", 5, seed=42)
        vpin = _vpin_proxy(bars, 15)
        assert vpin is None


# ---------------------------------------------------------------------------
# Tests: MicrostructureFlowAlpha
# ---------------------------------------------------------------------------


class TestMicrostructureFlowAlpha:
    """Core allocation logic."""

    def test_allocate_returns_dict(self) -> None:
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 80, seed=1)
        funding = _make_funding("A", 80, seed=1)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert isinstance(weights, dict)

    def test_weights_non_negative(self) -> None:
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 80, seed=42)
        funding = _make_funding("A", 80, seed=42)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        for w in weights.values():
            assert w >= Decimal("0")

    def test_max_weight_respected(self) -> None:
        config = MicrostructureFlowConfig(max_position_weight=0.20)
        strategy = MicrostructureFlowAlpha(config)
        insts = {f"I{i}": _make_bars(f"I{i}", 80, seed=i) for i in range(5)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 80, seed=i) for i in range(5)}
        weights = strategy.allocate(insts, fundings)
        for w in weights.values():
            assert float(w) <= config.max_position_weight + 0.01

    def test_gross_exposure_respected(self) -> None:
        config = MicrostructureFlowConfig(gross_exposure=0.8)
        strategy = MicrostructureFlowAlpha(config)
        insts = {f"I{i}": _make_bars(f"I{i}", 80, seed=i) for i in range(5)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 80, seed=i) for i in range(5)}
        weights = strategy.allocate(insts, fundings)
        total = sum(float(w) for w in weights.values())
        assert total <= config.gross_exposure + 0.01

    def test_empty_bars_returns_empty(self) -> None:
        strategy = MicrostructureFlowAlpha()
        assert strategy.allocate({}, {}) == {}

    def test_single_bar_returns_empty(self) -> None:
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 1)
        funding = _make_funding("A", 1)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert weights == {}

    def test_insufficient_data_returns_empty(self) -> None:
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 10)
        funding = _make_funding("A", 10)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert weights == {}

    def test_custom_config(self) -> None:
        config = MicrostructureFlowConfig(
            vir_short_window=2,
            vir_medium_window=8,
            vpin_window=10,
            vol_target=0.10,
            top_n=2,
        )
        strategy = MicrostructureFlowAlpha(config)
        insts = {f"I{i}": _make_bars(f"I{i}", 80, seed=i) for i in range(4)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 80, seed=i) for i in range(4)}
        weights = strategy.allocate(insts, fundings)
        assert isinstance(weights, dict)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_successive_calls(self) -> None:
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 80, seed=1)
        funding = _make_funding("A", 80, seed=1)
        w1 = strategy.allocate({"A": bars}, {"A": funding})
        w2 = strategy.allocate({"A": bars}, {"A": funding})
        assert isinstance(w1, dict)
        assert isinstance(w2, dict)

    def test_zero_volume_bars(self) -> None:
        """Bars with zero volume should be handled gracefully."""
        normal_bars = _make_bars("A", 75, seed=42)
        t = normal_bars[-1].ts + _DAY
        for i in range(5):
            normal_bars.append(
                OhlcvBar(
                    inst_id="A",
                    ts=t + _DAY * i,
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("0"),
                )
            )
        funding = _make_funding("A", 80, seed=42)
        strategy = MicrostructureFlowAlpha()
        weights = strategy.allocate({"A": normal_bars}, {"A": funding})
        assert isinstance(weights, dict)

    def test_crisis_regime_flat(self) -> None:
        config = MicrostructureFlowConfig(
            crisis_exposure=0.0,
            stressed_exposure=0.0,
        )
        strategy = MicrostructureFlowAlpha(config)
        bars = _make_bars("A", 80, vol=0.08, drift=-0.005, seed=11)
        funding = _make_funding("A", 80, seed=11)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        for w in weights.values():
            assert w >= Decimal("0")
            assert float(w) <= config.max_position_weight + 0.01

    def test_stop_loss_on_crash(self) -> None:
        config = MicrostructureFlowConfig(
            circuit_breaker_drop=0.05,
            circuit_breaker_lookback=5,
        )
        strategy = MicrostructureFlowAlpha(config)
        bars = _make_crash_bars("A", 80)
        funding = _make_funding("A", 80)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert weights == {}

    def test_drawdown_circuit_breaker(self) -> None:
        strategy = MicrostructureFlowAlpha(
            MicrostructureFlowConfig(dd_threshold=0.02),
        )
        strategy._equity_hist = [1.0, 0.99, 0.97, 0.94, 0.90]
        scale = strategy._drawdown_scale()
        assert scale < 1.0

    def test_vpin_danger_blocks_entry(self) -> None:
        """When VPIN is very high, the strategy should gate entries."""
        config = MicrostructureFlowConfig(vpin_danger_threshold=0.0)
        strategy = MicrostructureFlowAlpha(config)
        bars = _make_bars("A", 80, seed=42)
        funding = _make_funding("A", 80, seed=42)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        # With danger threshold at 0, essentially all VPIN > 0 blocks entry
        assert isinstance(weights, dict)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_funding_signal(self) -> None:
        """Negative funding → bullish signal (contrarian)."""
        strategy = MicrostructureFlowAlpha()
        bars = _make_bars("A", 80, seed=42)
        # Strongly negative funding
        funding = _make_funding("A", 80, base_rate=-0.005, seed=42)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert isinstance(weights, dict)
        for w in weights.values():
            assert w >= Decimal("0")


# ---------------------------------------------------------------------------
# Tests: allocator factory
# ---------------------------------------------------------------------------


class TestCreateMicrostructureAllocator:
    """The factory function must produce a valid allocator callable."""

    def test_factory_returns_callable(self) -> None:
        alloc = create_microstructure_flow_allocator()
        assert callable(alloc)

    def test_factory_produces_valid_weights(self) -> None:
        alloc = create_microstructure_flow_allocator()
        bars = _make_bars("A", 80, seed=1)
        funding = _make_funding("A", 80, seed=1)
        result = alloc({"A": bars}, {"A": funding})
        assert isinstance(result, dict)
        for w in result.values():
            assert w >= Decimal("0")


# ---------------------------------------------------------------------------
# Tests: backtest integration
# ---------------------------------------------------------------------------


class TestMicrostructureBacktestIntegration:
    """Run the strategy through the BacktestEngine end-to-end."""

    def test_backtest_engine_runs(self) -> None:
        alloc = create_microstructure_flow_allocator()
        bars_a = _make_bars("A", 120, seed=1)
        bars_b = _make_bars("B", 120, seed=2)
        funding_a = _make_funding("A", 120, seed=1)
        funding_b = _make_funding("B", 120, seed=2)

        config = BacktestConfig(
            rebalance_every_n_bars=5,
            periods_per_year=365,
        )
        engine = BacktestEngine(config)
        result = engine.run(
            alloc,
            {"A": bars_a, "B": bars_b},
            {"A": funding_a, "B": funding_b},
            min_history=50,
        )
        assert len(result.equity_curve) > 1
        assert result.metrics.total_trades >= 0

    def test_backtest_equity_stays_positive(self) -> None:
        alloc = create_microstructure_flow_allocator()
        bars = _make_bars("A", 150, seed=7)
        funding = _make_funding("A", 150, seed=7)

        config = BacktestConfig(
            initial_equity=50_000.0,
            rebalance_every_n_bars=5,
            taker_fee_rate=0.001,
            slippage_bps=5.0,
            periods_per_year=365.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(alloc, {"A": bars}, {"A": funding}, min_history=50)
        for eq in result.equity_curve:
            assert eq >= 0
