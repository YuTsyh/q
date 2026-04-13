"""Phase 2 – Comprehensive OOS backtesting of ALL strategies.

Runs every strategy in the repository through a 3-year OOS backtest
with realistic market frictions (Almgren-Chriss impact, OKX fees,
volume caps) and compiles a performance matrix.

The test collects results but does NOT assert pass/fail on performance
thresholds — that diagnostic is handled by the report generation.
Instead, each test validates that:
 • The strategy runs without error on 3-year data.
 • Produces valid equity curves and trade returns.
 • Returns a PerformanceMetrics object.
"""

from __future__ import annotations


import pytest

from quantbot.research.backtest import BacktestConfig, BacktestEngine
from quantbot.research.synthetic_data import (
    RealisticNoiseConfig,
    THREE_YEAR_REGIMES,
    generate_multi_instrument_data,
)
from quantbot.strategy.risk_overlay import with_risk_overlay
from quantbot.strategy.trend_following import create_trend_following_allocator
from quantbot.strategy.adaptive_momentum import create_adaptive_dual_momentum_allocator
from quantbot.strategy.ensemble import create_ensemble_allocator
from quantbot.strategy.regime_switching import create_regime_switching_allocator
from quantbot.strategy.mean_reversion_markov import create_mean_reversion_markov_allocator
from quantbot.strategy.vol_mean_reversion import create_vol_mean_reversion_allocator
from quantbot.strategy.cross_sectional_arb import create_cross_sectional_arb_allocator
from quantbot.strategy.microstructure_flow import create_microstructure_flow_allocator

# Calibrated noise config for OOS backtesting.  Balances realism with
# testability: produces worst-case single-bar returns around ±30% (matching
# extreme crypto daily moves like BTC March 2020) while preserving
# fat-tail and vol-clustering characteristics.
_OOS_NOISE = RealisticNoiseConfig(
    df=5.0,
    garch_omega=1e-6,
    garch_alpha=0.09,
    garch_beta=0.86,
    jump_intensity=0.015,
    jump_mean=-0.02,
    jump_std=0.03,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INSTRUMENTS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT"]


@pytest.fixture(scope="module")
def three_year_data():
    """Generate 3-year multi-instrument data with realistic noise."""
    bars, funding = generate_multi_instrument_data(
        inst_ids=INSTRUMENTS,
        regimes=THREE_YEAR_REGIMES,
        seed_base=42,
        noise_config=_OOS_NOISE,
        correlation=0.6,
    )
    return bars, funding


@pytest.fixture(scope="module")
def strict_config():
    """Strict OOS backtest config with realistic frictions."""
    return BacktestConfig(
        initial_equity=100_000.0,
        rebalance_every_n_bars=1,
        taker_fee_rate=0.0005,
        slippage_bps=2.0,
        partial_fill_ratio=1.0,
        periods_per_year=365.0,
        use_market_impact=True,
    )


def _make_allocator(factory, **kwargs):
    """Create a strategy allocator wrapped with risk overlay.

    Strategies that already have built-in regime gating (regime_switching,
    mean_reversion_markov, vol_mean_reversion, cross_sectional_arb,
    microstructure_flow) get a lighter overlay to avoid double-penalising.
    Simpler strategies (trend_following, adaptive_momentum, ensemble) get
    a tighter overlay.
    """
    from quantbot.strategy.risk_overlay import RiskOverlayConfig

    base = factory(**kwargs) if kwargs else factory()

    # Strategies with built-in regime management need lighter overlay
    light_strategies = {
        "create_regime_switching_allocator",
        "create_mean_reversion_markov_allocator",
        "create_vol_mean_reversion_allocator",
        "create_cross_sectional_arb_allocator",
        "create_microstructure_flow_allocator",
    }
    is_light = factory.__name__ in light_strategies

    if is_light:
        config = RiskOverlayConfig(
            max_gross_exposure=0.8,
            max_per_instrument=0.25,
            drawdown_lookback=20,
            drawdown_threshold=0.05,
            drawdown_flat_threshold=0.12,
            crash_guard_lookback=3,
            crash_guard_threshold=-0.08,
            regime_gating=False,  # Already handled internally
            vol_spike_lookback=5,
            vol_spike_threshold=2.0,
        )
    else:
        config = None  # Use default (strict) overlay

    return with_risk_overlay(base, config)


def _run_strategy(allocator, bars, funding, config, min_hist=20):
    """Helper to run a strategy and return its BacktestResult."""
    engine = BacktestEngine(config)
    return engine.run(allocator, bars, funding, min_history=min_hist)


# ---------------------------------------------------------------------------
# Strategy backtest tests
# ---------------------------------------------------------------------------

class TestTrendFollowingOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_trend_following_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100
        assert result.metrics.total_trades >= 0

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_trend_following_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestAdaptiveMomentumOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_adaptive_dual_momentum_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_adaptive_dual_momentum_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestEnsembleOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_ensemble_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_ensemble_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestRegimeSwitchingOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_regime_switching_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_regime_switching_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestMeanReversionMarkovOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_mean_reversion_markov_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_mean_reversion_markov_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestVolMeanReversionOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_vol_mean_reversion_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_vol_mean_reversion_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestCrossSectionalArbOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_cross_sectional_arb_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_cross_sectional_arb_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


class TestMicrostructureFlowOOS:
    def test_runs_without_error(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_microstructure_flow_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert len(result.equity_curve) > 100

    def test_equity_curve_positive(self, three_year_data, strict_config):
        bars, funding = three_year_data
        allocator = _make_allocator(create_microstructure_flow_allocator)
        result = _run_strategy(allocator, bars, funding, strict_config)
        assert all(e >= 0 for e in result.equity_curve)


# ---------------------------------------------------------------------------
# Performance matrix report
# ---------------------------------------------------------------------------

class TestPerformanceReport:
    """Generate a combined performance matrix for all strategies."""

    STRATEGY_FACTORIES = {
        "trend_following": create_trend_following_allocator,
        "adaptive_momentum": create_adaptive_dual_momentum_allocator,
        "ensemble": create_ensemble_allocator,
        "regime_switching": create_regime_switching_allocator,
        "mean_reversion_markov": create_mean_reversion_markov_allocator,
        "vol_mean_reversion": create_vol_mean_reversion_allocator,
        "cross_sectional_arb": create_cross_sectional_arb_allocator,
        "microstructure_flow": create_microstructure_flow_allocator,
    }

    def test_all_strategies_produce_metrics(self, three_year_data, strict_config):
        """Run all strategies and ensure each returns valid metrics."""
        bars, funding = three_year_data
        engine = BacktestEngine(strict_config)

        results = {}
        for name, factory in self.STRATEGY_FACTORIES.items():
            allocator = _make_allocator(factory)
            result = engine.run(allocator, bars, funding, min_history=20)
            results[name] = result.metrics
            # Must have valid numeric metrics
            assert result.metrics.sharpe_ratio is not None
            assert result.metrics.max_drawdown is not None
            assert result.metrics.cagr is not None

        # Print the performance matrix for diagnostic
        header = (
            f"{'Strategy':<25} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>8} | "
            f"{'Sortino':>8} | {'WinRate':>8} | {'PF':>8} | {'Trades':>8}"
        )
        print("\n" + "=" * len(header))
        print("PERFORMANCE MATRIX – 3-Year OOS (Strict Frictions + Risk Overlay)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for name, m in results.items():
            print(
                f"{name:<25} | {m.cagr:>8.2%} | {m.max_drawdown:>8.2%} | "
                f"{m.sharpe_ratio:>8.2f} | {m.sortino_ratio:>8.2f} | "
                f"{m.win_rate:>8.2%} | {m.profit_factor:>8.2f} | "
                f"{m.total_trades:>8}"
            )
        print("=" * len(header))
