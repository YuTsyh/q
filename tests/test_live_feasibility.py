"""Tests verifying live-trading feasibility for all strategies.

Covers:
- Division-by-zero guards in factors, metrics, and backtest engine
- Zero/negative price handling
- Empty data edge cases
- Equity wipeout scenarios
- Strategy allocation with minimal / degenerate data
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantbot.research.backtest import BacktestConfig, BacktestEngine, walk_forward_analysis
from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.factors import MomentumFactor, TrendFactor
from quantbot.research.metrics import compute_metrics
from quantbot.research.vol_factors import MeanReversionFactor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _bars(closes: list[str], *, volumes: list[str] | None = None) -> list[OhlcvBar]:
    """Build OhlcvBar list from string prices."""
    vols = volumes or ["100"] * len(closes)
    return [
        OhlcvBar(
            inst_id="BTC-USDT",
            ts=_BASE_TS + timedelta(hours=i),
            open=Decimal(c),
            high=Decimal(c),
            low=Decimal(c),
            close=Decimal(c),
            volume=Decimal(vols[i]),
        )
        for i, c in enumerate(closes)
    ]


def _funding(n: int, rate: str = "0.0001") -> list[FundingRate]:
    return [
        FundingRate(
            inst_id="BTC-USDT",
            funding_time=_BASE_TS + timedelta(hours=i * 8),
            funding_rate=Decimal(rate),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Factor division-by-zero tests
# ---------------------------------------------------------------------------


class TestMomentumFactorZeroPrice:
    """MomentumFactor must not crash when start price is zero."""

    def test_zero_start_price_returns_zero(self):
        bars = _bars(["0", "10", "20", "30"])
        result = MomentumFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_negative_start_price_returns_zero(self):
        bars = _bars(["-1", "10", "20", "30"])
        result = MomentumFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_normal_case_still_works(self):
        bars = _bars(["100", "105", "110", "120"])
        result = MomentumFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0.20")


class TestTrendFactorZeroAverage:
    """TrendFactor must not crash when average price is zero."""

    def test_all_zero_closes_returns_zero(self):
        bars = _bars(["0", "0", "0"])
        result = TrendFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_negative_average_returns_zero(self):
        bars = _bars(["-1", "-2", "-3"])
        result = TrendFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_normal_case_still_works(self):
        bars = _bars(["100", "110", "120"])
        result = TrendFactor(lookback=3).compute(bars, [])
        assert result > Decimal("0")


class TestMeanReversionFactorZeroClose:
    """MeanReversionFactor must not crash when current close is zero."""

    def test_zero_close_returns_zero(self):
        bars = _bars(["100", "110", "0"])
        result = MeanReversionFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_negative_close_returns_zero(self):
        bars = _bars(["100", "110", "-5"])
        result = MeanReversionFactor(lookback=3).compute(bars, [])
        assert result == Decimal("0")

    def test_normal_oversold_positive(self):
        bars = _bars(["100", "110", "90"])
        result = MeanReversionFactor(lookback=3).compute(bars, [])
        assert result > Decimal("0")


# ---------------------------------------------------------------------------
# Metrics edge cases
# ---------------------------------------------------------------------------


class TestMetricsZeroEquity:
    """compute_metrics must handle zero equity in the curve."""

    def test_zero_equity_in_curve_no_crash(self):
        # Equity goes to zero mid-stream
        equity = [100.0, 50.0, 0.0, 0.0]
        returns = [-0.5, -1.0, 0.0]
        metrics = compute_metrics(equity, returns)
        assert metrics.total_return == pytest.approx(-1.0, abs=1e-6)
        assert metrics.max_drawdown == pytest.approx(1.0, abs=1e-6)

    def test_equity_wipeout_sharpe_not_nan(self):
        equity = [100.0, 80.0, 20.0, 0.0]
        returns = [-0.2, -0.75, -1.0]
        metrics = compute_metrics(equity, returns)
        assert not (metrics.sharpe_ratio != metrics.sharpe_ratio)  # not NaN


# ---------------------------------------------------------------------------
# Backtest engine equity clamping
# ---------------------------------------------------------------------------


class TestBacktestEquityClamping:
    """BacktestEngine must clamp equity to zero; never go negative."""

    def test_catastrophic_loss_equity_stays_non_negative(self):
        """If price drops to near-zero, equity should not go negative."""
        # Build data: sharp 99% crash
        closes = ["100"] * 10 + ["1"]  # 99% crash at the end
        bars_data = _bars(closes)
        funding_data = _funding(11)

        def always_long(bars_by_inst, fund_by_inst):
            return {"BTC-USDT": Decimal("1.0")}

        config = BacktestConfig(initial_equity=100_000.0, rebalance_every_n_bars=1)
        engine = BacktestEngine(config)
        result = engine.run(
            always_long,
            {"BTC-USDT": bars_data},
            {"BTC-USDT": funding_data},
            min_history=2,
        )
        # Every point in equity curve must be >= 0
        assert all(e >= 0.0 for e in result.equity_curve)


# ---------------------------------------------------------------------------
# Walk-forward zero-equity safety
# ---------------------------------------------------------------------------


class TestWalkForwardZeroEquity:
    """walk_forward_analysis must not crash on zero equity windows."""

    def test_walk_forward_with_crash_data(self):
        """Walk-forward analysis should handle windows where equity drops to zero."""
        # Build enough data for walk-forward with a crash in the middle
        n = 120
        closes = []
        for i in range(n):
            if 50 <= i < 55:
                closes.append("1")  # crash
            else:
                closes.append(str(100 + i % 10))
        bars_data = _bars(closes)
        funding_data = _funding(n)

        def simple_allocator(bars_by_inst, fund_by_inst):
            return {"BTC-USDT": Decimal("0.5")}

        config = BacktestConfig(initial_equity=100_000.0)
        # Should not crash
        try:
            walk_forward_analysis(
                simple_allocator,
                {"BTC-USDT": bars_data},
                {"BTC-USDT": funding_data},
                config=config,
                n_splits=3,
            )
        except ValueError:
            # May raise "Not enough data" which is acceptable
            pass


# ---------------------------------------------------------------------------
# Strategy allocation: empty data
# ---------------------------------------------------------------------------


class TestStrategyEmptyData:
    """All strategies must return empty dict when given no instruments."""

    def test_trend_following_empty(self):
        from quantbot.strategy.trend_following import VolatilityAdjustedTrendFollower

        s = VolatilityAdjustedTrendFollower()
        assert s.allocate({}, {}) == {}

    def test_ensemble_empty(self):
        from quantbot.strategy.ensemble import EnsembleMomentumTrend

        s = EnsembleMomentumTrend()
        assert s.allocate({}, {}) == {}

    def test_adaptive_momentum_empty(self):
        from quantbot.strategy.adaptive_momentum import AdaptiveDualMomentumStrategy

        s = AdaptiveDualMomentumStrategy.default()
        assert s.allocate({}, {}) == {}

    def test_regime_switching_empty(self):
        from quantbot.strategy.regime_switching import RegimeSwitchingAlpha

        s = RegimeSwitchingAlpha()
        assert s.allocate({}, {}) == {}

    def test_mean_reversion_markov_empty(self):
        from quantbot.strategy.mean_reversion_markov import MarkovMeanReversionAlpha

        s = MarkovMeanReversionAlpha()
        assert s.allocate({}, {}) == {}


# ---------------------------------------------------------------------------
# Strategy allocation: insufficient bars
# ---------------------------------------------------------------------------


class TestStrategyInsufficientBars:
    """Strategies must return empty dict with too few bars (no crash)."""

    def test_trend_following_single_bar(self):
        from quantbot.strategy.trend_following import VolatilityAdjustedTrendFollower

        s = VolatilityAdjustedTrendFollower()
        bars_data = _bars(["100"])
        result = s.allocate({"BTC": bars_data}, {})
        assert result == {}

    def test_ensemble_single_bar(self):
        from quantbot.strategy.ensemble import EnsembleMomentumTrend

        s = EnsembleMomentumTrend()
        bars_data = _bars(["100"])
        result = s.allocate({"BTC": bars_data}, {})
        assert result == {}

    def test_regime_switching_single_bar(self):
        from quantbot.strategy.regime_switching import RegimeSwitchingAlpha

        s = RegimeSwitchingAlpha()
        bars_data = _bars(["100"])
        result = s.allocate({"BTC": bars_data}, {})
        assert result == {}

    def test_mean_reversion_single_bar(self):
        from quantbot.strategy.mean_reversion_markov import MarkovMeanReversionAlpha

        s = MarkovMeanReversionAlpha()
        bars_data = _bars(["100"])
        result = s.allocate({"BTC": bars_data}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# Strategy weights are always non-negative and bounded
# ---------------------------------------------------------------------------


class TestStrategyWeightBounds:
    """All strategies must produce non-negative weights ≤ max_position_weight."""

    @staticmethod
    def _make_uptrend_bars(n: int = 80) -> list[OhlcvBar]:
        closes = [str(100 + i * 2) for i in range(n)]
        return _bars(closes)

    def test_trend_following_weights_bounded(self):
        from quantbot.strategy.trend_following import VolatilityAdjustedTrendFollower

        s = VolatilityAdjustedTrendFollower()
        bars_data = self._make_uptrend_bars()
        result = s.allocate({"BTC": bars_data, "ETH": bars_data}, {})
        for w in result.values():
            assert w >= Decimal("0")
            assert w <= Decimal("0.25")  # max_position_weight default

    def test_ensemble_weights_bounded(self):
        from quantbot.strategy.ensemble import EnsembleMomentumTrend

        s = EnsembleMomentumTrend()
        bars_data = self._make_uptrend_bars()
        result = s.allocate({"BTC": bars_data, "ETH": bars_data}, {})
        for w in result.values():
            assert w >= Decimal("0")
            assert w <= Decimal("0.30")  # max_position_weight default

    def test_regime_switching_weights_bounded(self):
        from quantbot.strategy.regime_switching import RegimeSwitchingAlpha

        s = RegimeSwitchingAlpha()
        bars_data = self._make_uptrend_bars()
        result = s.allocate({"BTC": bars_data, "ETH": bars_data}, {})
        for w in result.values():
            assert w >= Decimal("0")
            assert w <= Decimal("0.35")  # max_position_weight default


# ---------------------------------------------------------------------------
# Factory functions return Callable
# ---------------------------------------------------------------------------


class TestFactoryReturnCallable:
    """All factory functions must return a callable."""

    def test_trend_following_factory(self):
        from quantbot.strategy.trend_following import create_trend_following_allocator

        alloc = create_trend_following_allocator()
        assert callable(alloc)

    def test_ensemble_factory(self):
        from quantbot.strategy.ensemble import create_ensemble_allocator

        alloc = create_ensemble_allocator()
        assert callable(alloc)

    def test_adaptive_momentum_factory(self):
        from quantbot.strategy.adaptive_momentum import create_adaptive_dual_momentum_allocator

        alloc = create_adaptive_dual_momentum_allocator()
        assert callable(alloc)

    def test_regime_switching_factory(self):
        from quantbot.strategy.regime_switching import create_regime_switching_allocator

        alloc = create_regime_switching_allocator()
        assert callable(alloc)

    def test_mean_reversion_markov_factory(self):
        from quantbot.strategy.mean_reversion_markov import create_mean_reversion_markov_allocator

        alloc = create_mean_reversion_markov_allocator()
        assert callable(alloc)
