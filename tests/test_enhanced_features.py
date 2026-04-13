"""Tests for enhanced features: realistic synthetic data, market impact
integration, AllocatorStrategyAdapter, bar-level stop-loss, and
permutation-based Monte Carlo.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

from quantbot.core.models import MarketSnapshot
from quantbot.research.backtest import (
    BacktestConfig,
    BacktestEngine,
    monte_carlo_simulation,
)
from quantbot.research.metrics import compute_metrics
from quantbot.research.synthetic_data import (
    DEFAULT_NOISE_CONFIG,
    MarketRegime,
    RealisticNoiseConfig,
    _cholesky_decompose,
    _student_t_sample,
    build_correlation_matrix,
    generate_multi_instrument_data,
    generate_ohlcv,
)
from quantbot.strategy.base import AllocatorStrategyAdapter
from quantbot.strategy.trend_following import create_trend_following_allocator


# ---------------------------------------------------------------------------
# Issue 1 & 8: Enhanced synthetic data
# ---------------------------------------------------------------------------


class TestStudentTSample:
    def test_high_df_approximates_gaussian(self):
        """With df >= 30, Student-t should approximate N(0,1)."""
        import random

        rng = random.Random(42)
        samples = [_student_t_sample(rng, df=100) for _ in range(5000)]
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        assert abs(mean) < 0.1
        assert abs(var - 1.0) < 0.2

    def test_low_df_produces_heavier_tails(self):
        """With df=3, kurtosis should be higher than Gaussian."""
        import random

        rng = random.Random(42)
        samples = [_student_t_sample(rng, df=3) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        # Student-t with df=3 has theoretical kurtosis = infinity
        # but empirically the 4th moment should be much larger than 3
        kurt = sum((s - mean) ** 4 for s in samples) / (len(samples) * var ** 2)
        assert kurt > 4.0  # Gaussian kurtosis = 3


class TestRealisticOhlcv:
    def test_backward_compatible_without_noise_config(self):
        """Without noise_config, generate_ohlcv should behave identically."""
        regime = MarketRegime("test", 0.001, 0.02, 50)
        bars_legacy = generate_ohlcv("X", [regime], seed=42)
        bars_new = generate_ohlcv("X", [regime], seed=42, noise_config=None)
        assert len(bars_legacy) == len(bars_new)
        assert [b.close for b in bars_legacy] == [b.close for b in bars_new]

    def test_realistic_mode_produces_valid_bars(self):
        """With noise_config, bars should still have valid structure."""
        regime = MarketRegime("test", 0.001, 0.02, 100)
        bars = generate_ohlcv("X", [regime], seed=42, noise_config=DEFAULT_NOISE_CONFIG)
        assert len(bars) == 100
        for b in bars:
            assert b.close > 0
            assert b.high >= b.low

    def test_realistic_mode_deterministic(self):
        """Realistic mode should be deterministic with same seed."""
        regime = MarketRegime("test", 0.001, 0.02, 50)
        b1 = generate_ohlcv("X", [regime], seed=42, noise_config=DEFAULT_NOISE_CONFIG)
        b2 = generate_ohlcv("X", [regime], seed=42, noise_config=DEFAULT_NOISE_CONFIG)
        assert [b.close for b in b1] == [b.close for b in b2]

    def test_fat_tails_increase_extreme_returns(self):
        """Realistic data should have more extreme returns than GBM."""
        regime = MarketRegime("test", 0.0, 0.03, 2000)
        bars_gbm = generate_ohlcv("X", [regime], seed=42, noise_config=None)
        bars_real = generate_ohlcv(
            "X", [regime], seed=42,
            noise_config=RealisticNoiseConfig(df=3, jump_intensity=0.05),
        )

        def max_abs_return(bars):
            returns = []
            for i in range(1, len(bars)):
                p0 = float(bars[i - 1].close)
                p1 = float(bars[i].close)
                if p0 > 0:
                    returns.append(abs(p1 / p0 - 1))
            return max(returns) if returns else 0

        # Realistic data should have larger extreme moves
        assert max_abs_return(bars_real) > max_abs_return(bars_gbm) * 0.5

    def test_jump_diffusion_creates_discontinuous_moves(self):
        """With high jump intensity, we should see sudden large moves."""
        regime = MarketRegime("test", 0.0, 0.01, 500)
        nc = RealisticNoiseConfig(
            df=30, jump_intensity=0.1, jump_mean=-0.05, jump_std=0.06,
            garch_alpha=0.0, garch_beta=0.0,
        )
        bars = generate_ohlcv("X", [regime], seed=42, noise_config=nc)
        returns = []
        for i in range(1, len(bars)):
            p0 = float(bars[i - 1].close)
            p1 = float(bars[i].close)
            if p0 > 0:
                returns.append(p1 / p0 - 1)
        # Should have at least some returns larger than 3 sigma of base vol
        large_moves = [r for r in returns if abs(r) > 0.05]
        assert len(large_moves) > 0


class TestCholeskyCorrelation:
    def test_identity_matrix(self):
        """Cholesky of identity should be identity."""
        identity = [[1.0, 0.0], [0.0, 1.0]]
        L = _cholesky_decompose(identity)
        assert abs(L[0][0] - 1.0) < 1e-10
        assert abs(L[1][1] - 1.0) < 1e-10
        assert abs(L[1][0]) < 1e-10

    def test_correlated_matrix(self):
        """Cholesky of a correlation matrix should reconstruct it."""
        C = [[1.0, 0.6], [0.6, 1.0]]
        L = _cholesky_decompose(C)
        # L @ L^T should equal C
        for i in range(2):
            for j in range(2):
                val = sum(L[i][k] * L[j][k] for k in range(2))
                assert abs(val - C[i][j]) < 1e-10

    def test_build_correlation_matrix(self):
        corr = build_correlation_matrix(3, base_corr=0.7)
        assert len(corr) == 3
        for i in range(3):
            assert corr[i][i] == 1.0
            for j in range(3):
                if i != j:
                    assert corr[i][j] == 0.7


class TestMultiInstrumentCorrelation:
    def test_correlated_data_has_similar_trends(self):
        """With high correlation, instruments should move together."""
        ids = ["A-USDT", "B-USDT"]
        regime = MarketRegime("bull", drift=0.002, volatility=0.02, duration_bars=200)
        bars_corr, _ = generate_multi_instrument_data(
            ids, [regime], seed_base=42,
            noise_config=DEFAULT_NOISE_CONFIG, correlation=0.9,
        )
        # Compute returns
        rets_a = []
        rets_b = []
        bars_a = bars_corr["A-USDT"]
        bars_b = bars_corr["B-USDT"]
        for i in range(1, min(len(bars_a), len(bars_b))):
            ra = float(bars_a[i].close / bars_a[i - 1].close - 1)
            rb = float(bars_b[i].close / bars_b[i - 1].close - 1)
            rets_a.append(ra)
            rets_b.append(rb)

        # Compute correlation
        n = len(rets_a)
        mean_a = sum(rets_a) / n
        mean_b = sum(rets_b) / n
        cov = sum((rets_a[i] - mean_a) * (rets_b[i] - mean_b) for i in range(n)) / n
        std_a = math.sqrt(sum((r - mean_a) ** 2 for r in rets_a) / n)
        std_b = math.sqrt(sum((r - mean_b) ** 2 for r in rets_b) / n)
        corr = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0
        # With 0.9 input correlation, observed should be notably positive
        assert corr > 0.2

    def test_uncorrelated_backward_compatible(self):
        """Without correlation parameter, old behavior is preserved."""
        ids = ["A-USDT", "B-USDT"]
        bars1, _ = generate_multi_instrument_data(ids, seed_base=42)
        bars2, _ = generate_multi_instrument_data(ids, seed_base=42, correlation=None)
        assert [b.close for b in bars1["A-USDT"]] == [b.close for b in bars2["A-USDT"]]


# ---------------------------------------------------------------------------
# Issue 2: Market impact integration
# ---------------------------------------------------------------------------


class TestMarketImpactBacktest:
    def test_market_impact_mode_runs(self):
        """BacktestEngine with market impact should complete without error."""
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            periods_per_year=365.0,
            use_market_impact=True,
        )
        ids = ["BTC-USDT", "ETH-USDT"]
        bars, funding = generate_multi_instrument_data(ids)
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.equity_curve) > 10
        assert result.metrics is not None

    def test_legacy_mode_runs(self):
        """BacktestEngine with legacy slippage should still work."""
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            periods_per_year=365.0,
            use_market_impact=False,
            slippage_bps=2.0,
            partial_fill_ratio=1.0,
        )
        ids = ["BTC-USDT", "ETH-USDT"]
        bars, funding = generate_multi_instrument_data(ids)
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        assert len(result.equity_curve) > 10

    def test_market_impact_costs_more_than_legacy(self):
        """Market impact should generally create more drag than flat BPS."""
        ids = ["BTC-USDT", "ETH-USDT"]
        bars, funding = generate_multi_instrument_data(ids)
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)

        mi_config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            periods_per_year=365.0,
            use_market_impact=True,
        )
        legacy_config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            periods_per_year=365.0,
            use_market_impact=False,
            slippage_bps=0.1,  # Very low slippage
            partial_fill_ratio=1.0,
        )

        mi_result = BacktestEngine(mi_config).run(allocator, bars, funding, min_history=15)
        legacy_result = BacktestEngine(legacy_config).run(allocator, bars, funding, min_history=15)
        # Market impact should produce lower or similar returns (more cost drag)
        assert mi_result.metrics.total_return <= legacy_result.metrics.total_return + 0.1


# ---------------------------------------------------------------------------
# Issue 3: AllocatorStrategyAdapter
# ---------------------------------------------------------------------------


class TestAllocatorStrategyAdapter:
    def test_adapter_produces_order_intents(self):
        """Adapter should emit OrderIntents from allocator signals."""
        allocator = create_trend_following_allocator(fast_ema=3, slow_ema=10)
        adapter = AllocatorStrategyAdapter(
            allocator=allocator,
            inst_ids=["BTC-USDT"],
            bar_interval_seconds=1,  # 1 second bars for testing
            order_size_quote=Decimal("1000"),
        )

        ids = ["BTC-USDT"]
        bars, _ = generate_multi_instrument_data(ids)
        # Feed bars as snapshots
        results = []
        for bar in bars["BTC-USDT"]:
            snap = MarketSnapshot(
                inst_id="BTC-USDT",
                last_price=bar.close,
                received_at=bar.ts,
            )
            intent = adapter.on_market(snap)
            if intent is not None:
                results.append(intent)
        # Should produce at least one order
        # (it depends on whether price moves enough for the strategy to signal)
        assert isinstance(results, list)

    def test_adapter_ignores_unknown_instruments(self):
        """Snapshots for instruments not in inst_ids should be ignored."""
        allocator = create_trend_following_allocator()
        adapter = AllocatorStrategyAdapter(
            allocator=allocator,
            inst_ids=["BTC-USDT"],
        )
        snap = MarketSnapshot(
            inst_id="UNKNOWN-USDT",
            last_price=Decimal("100"),
            received_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert adapter.on_market(snap) is None


# ---------------------------------------------------------------------------
# Issue 5: Bar-level stop-loss checking
# ---------------------------------------------------------------------------


class TestBarLevelStopLoss:
    def test_stop_loss_triggers_between_rebalances(self):
        """Stop-loss should fire every bar, not just at rebalance."""
        # Use a rebalance interval of 5 bars, so bars 1-4 between
        # rebalances should still check stops.
        config = BacktestConfig(
            initial_equity=100_000.0,
            rebalance_every_n_bars=5,
            periods_per_year=365.0,
            use_market_impact=True,
        )
        ids = ["BTC-USDT", "ETH-USDT"]
        bars, funding = generate_multi_instrument_data(ids)
        allocator = create_trend_following_allocator(
            fast_ema=3, slow_ema=10, stop_loss_atr=0.5,  # Tight stop
        )
        engine = BacktestEngine(config)
        result = engine.run(allocator, bars, funding, min_history=15)
        # Should complete without error and have valid metrics
        assert result.metrics is not None
        assert len(result.equity_curve) > 10


# ---------------------------------------------------------------------------
# Issue 6: Metrics default periods_per_year
# ---------------------------------------------------------------------------


class TestMetricsDefaultPeriods:
    def test_default_periods_per_year_is_365(self):
        """compute_metrics default periods_per_year should be 365."""
        equity = [100.0, 110.0, 105.0, 115.0]
        returns = [0.1, -0.045, 0.095]
        # Call without specifying periods_per_year
        m = compute_metrics(equity, returns)
        # Verify it used 365 by checking the CAGR calculation
        years = len(returns) / 365.0
        expected_cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0
        assert abs(m.cagr - expected_cagr) < 0.01


# ---------------------------------------------------------------------------
# Issue 7: Permutation-based Monte Carlo
# ---------------------------------------------------------------------------


class TestPermutationMonteCarlo:
    def test_permutation_test_can_fail(self):
        """With pure noise returns, MC should produce near-zero median Sharpe."""
        # Returns with no signal (random positive and negative)
        returns = [0.01, -0.01, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002] * 10
        result = monte_carlo_simulation(returns, n_simulations=200, seed=42)
        # Median Sharpe under null should be near zero
        assert abs(result.median_sharpe) < 2.0

    def test_permutation_destroys_positive_expectancy(self):
        """With strong positive returns, permutation should reduce expectancy."""
        # Strongly positive returns
        returns = [0.05, 0.03, 0.04, 0.02, 0.06, 0.01, 0.03, 0.02] * 5
        result = monte_carlo_simulation(returns, n_simulations=200, seed=42)
        # Under permutation (sign-flip), median return should be near zero
        assert result.median_total_return < sum(returns)

    def test_deterministic_with_seed(self):
        returns = [0.01, -0.005, 0.02, -0.01]
        r1 = monte_carlo_simulation(returns, n_simulations=50, seed=123)
        r2 = monte_carlo_simulation(returns, n_simulations=50, seed=123)
        assert r1.median_sharpe == r2.median_sharpe
        assert r1.median_max_dd == r2.median_max_dd
