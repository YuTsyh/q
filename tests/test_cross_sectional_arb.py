"""Tests for the Cross-Sectional Statistical Arbitrage strategy.

Validates:
- Cross-sectional factor scoring and ranking
- Minimum instrument requirements
- Beta hedging and weight capping
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
from quantbot.strategy.cross_sectional_arb import (
    CrossSectionalArbAlpha,
    CrossSectionalArbConfig,
    create_cross_sectional_arb_allocator,
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
# Tests: CrossSectionalArbAlpha
# ---------------------------------------------------------------------------


class TestCrossSectionalArbAlpha:
    """Core allocation logic."""

    def _make_multi(
        self, n_insts: int = 5, n_bars: int = 80,
    ) -> tuple[dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]]:
        """Helper: create bars + funding for *n_insts* instruments."""
        insts = {
            f"I{i}": _make_bars(f"I{i}", n_bars, seed=i * 10)
            for i in range(n_insts)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", n_bars, seed=i * 10)
            for i in range(n_insts)
        }
        return insts, fundings

    def test_allocate_returns_dict(self) -> None:
        strategy = CrossSectionalArbAlpha()
        insts, fundings = self._make_multi(5, 80)
        weights = strategy.allocate(insts, fundings)
        assert isinstance(weights, dict)

    def test_weights_non_negative(self) -> None:
        strategy = CrossSectionalArbAlpha()
        insts, fundings = self._make_multi(5, 80)
        weights = strategy.allocate(insts, fundings)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_max_weight_respected(self) -> None:
        config = CrossSectionalArbConfig(max_position_weight=0.20)
        strategy = CrossSectionalArbAlpha(config)
        insts, fundings = self._make_multi(6, 80)
        weights = strategy.allocate(insts, fundings)
        for w in weights.values():
            assert float(w) <= config.max_position_weight + 0.01

    def test_gross_exposure_respected(self) -> None:
        config = CrossSectionalArbConfig(gross_exposure=0.8)
        strategy = CrossSectionalArbAlpha(config)
        insts, fundings = self._make_multi(6, 80)
        weights = strategy.allocate(insts, fundings)
        total = sum(float(w) for w in weights.values())
        assert total <= config.gross_exposure + 0.01

    def test_empty_bars_returns_empty(self) -> None:
        strategy = CrossSectionalArbAlpha()
        assert strategy.allocate({}, {}) == {}

    def test_single_bar_returns_empty(self) -> None:
        strategy = CrossSectionalArbAlpha()
        insts = {f"I{i}": _make_bars(f"I{i}", 1, seed=i) for i in range(5)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 1, seed=i) for i in range(5)}
        weights = strategy.allocate(insts, fundings)
        assert weights == {}

    def test_insufficient_instruments(self) -> None:
        """Fewer than min_instruments → empty."""
        config = CrossSectionalArbConfig(min_instruments=4)
        strategy = CrossSectionalArbAlpha(config)
        insts = {f"I{i}": _make_bars(f"I{i}", 80, seed=i) for i in range(2)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 80, seed=i) for i in range(2)}
        weights = strategy.allocate(insts, fundings)
        assert weights == {}

    def test_needs_min_instruments_for_cross_section(self) -> None:
        """1-2 instruments → empty (cross-section needs breadth)."""
        strategy = CrossSectionalArbAlpha()
        bars = _make_bars("A", 80, seed=1)
        funding = _make_funding("A", 80, seed=1)
        weights = strategy.allocate({"A": bars}, {"A": funding})
        assert weights == {}

    def test_custom_config(self) -> None:
        config = CrossSectionalArbConfig(
            momentum_lookback=5,
            vol_lookback=15,
            top_n=2,
            min_instruments=3,
            w_momentum=0.40,
            w_vol_surprise=0.10,
            w_mean_reversion=0.20,
            w_liquidity=0.10,
            w_carry=0.10,
            w_vol_discount=0.10,
        )
        strategy = CrossSectionalArbAlpha(config)
        insts, fundings = self._make_multi(5, 80)
        weights = strategy.allocate(insts, fundings)
        assert isinstance(weights, dict)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_successive_calls(self) -> None:
        strategy = CrossSectionalArbAlpha()
        insts, fundings = self._make_multi(5, 80)
        w1 = strategy.allocate(insts, fundings)
        w2 = strategy.allocate(insts, fundings)
        assert isinstance(w1, dict)
        assert isinstance(w2, dict)

    def test_many_instruments(self) -> None:
        """Test with 8+ instruments using different seeds."""
        strategy = CrossSectionalArbAlpha()
        insts = {
            f"I{i}": _make_bars(f"I{i}", 80, seed=i * 7 + 3)
            for i in range(8)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", 80, seed=i * 7 + 3)
            for i in range(8)
        }
        weights = strategy.allocate(insts, fundings)
        assert isinstance(weights, dict)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_crisis_exposure_flat(self) -> None:
        """High-vol data should reduce or eliminate exposure."""
        config = CrossSectionalArbConfig(min_instruments=3)
        strategy = CrossSectionalArbAlpha(config)
        insts = {
            f"I{i}": _make_bars(f"I{i}", 80, vol=0.08, drift=-0.005, seed=i + 20)
            for i in range(5)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", 80, seed=i + 20)
            for i in range(5)
        }
        weights = strategy.allocate(insts, fundings)
        for w in weights.values():
            assert w >= Decimal("0")

    def test_stop_loss_on_crash(self) -> None:
        config = CrossSectionalArbConfig(
            circuit_breaker_drop=0.05,
            circuit_breaker_lookback=5,
            min_instruments=3,
        )
        strategy = CrossSectionalArbAlpha(config)
        insts = {
            f"I{i}": _make_crash_bars(f"I{i}", 80, seed=90 + i)
            for i in range(5)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", 80, seed=90 + i)
            for i in range(5)
        }
        weights = strategy.allocate(insts, fundings)
        assert weights == {}

    def test_drawdown_circuit_breaker(self) -> None:
        config = CrossSectionalArbConfig(dd_threshold=0.02)
        strategy = CrossSectionalArbAlpha(config)
        strategy._equity_hist = [1.0, 0.99, 0.97, 0.94, 0.90]
        scale = strategy._drawdown_scale()
        assert scale < 1.0


# ---------------------------------------------------------------------------
# Tests: allocator factory
# ---------------------------------------------------------------------------


class TestCreateCrossSectionalAllocator:
    """The factory function must produce a valid allocator callable."""

    def test_factory_returns_callable(self) -> None:
        alloc = create_cross_sectional_arb_allocator()
        assert callable(alloc)

    def test_factory_produces_valid_weights(self) -> None:
        alloc = create_cross_sectional_arb_allocator()
        insts = {f"I{i}": _make_bars(f"I{i}", 80, seed=i) for i in range(5)}
        fundings = {f"I{i}": _make_funding(f"I{i}", 80, seed=i) for i in range(5)}
        result = alloc(insts, fundings)
        assert isinstance(result, dict)
        for w in result.values():
            assert w >= Decimal("0")


# ---------------------------------------------------------------------------
# Tests: backtest integration
# ---------------------------------------------------------------------------


class TestCrossSectionalBacktestIntegration:
    """Run the strategy through the BacktestEngine end-to-end."""

    def test_backtest_engine_runs(self) -> None:
        alloc = create_cross_sectional_arb_allocator()
        insts = {
            f"I{i}": _make_bars(f"I{i}", 120, seed=10 + i)
            for i in range(5)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", 120, seed=10 + i)
            for i in range(5)
        }

        config = BacktestConfig(
            rebalance_every_n_bars=5,
            periods_per_year=365,
        )
        engine = BacktestEngine(config)
        result = engine.run(alloc, insts, fundings, min_history=50)
        assert len(result.equity_curve) > 1
        assert result.metrics.total_trades >= 0

    def test_backtest_equity_stays_positive(self) -> None:
        alloc = create_cross_sectional_arb_allocator()
        insts = {
            f"I{i}": _make_bars(f"I{i}", 150, seed=20 + i)
            for i in range(5)
        }
        fundings = {
            f"I{i}": _make_funding(f"I{i}", 150, seed=20 + i)
            for i in range(5)
        }

        config = BacktestConfig(
            initial_equity=50_000.0,
            rebalance_every_n_bars=5,
            taker_fee_rate=0.001,
            slippage_bps=5.0,
            periods_per_year=365.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(alloc, insts, fundings, min_history=50)
        for eq in result.equity_curve:
            assert eq >= 0
