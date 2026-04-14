"""Synthetic OHLCV and Funding Rate data generator for **unit tests only**.

.. warning::
   **DO NOT use this module for strategy performance evaluation or
   validation.**  Synthetic data generators produce idealised price
   dynamics (GBM with regime overlays) that do not reproduce the
   distributional properties of real crypto markets: fat tails,
   volatility clustering, microstructure noise, and liquidity gaps.

   Any "Sharpe" or "CAGR" computed on synthetic data is meaningless
   for predicting live trading performance.

   For strategy validation, use :mod:`quantbot.research.real_data`
   to download actual historical OHLCV and funding rate data from
   exchanges such as OKX.

This module is **appropriate** for:
- Unit testing strategy logic (does allocate() return correct types?)
- Integration testing the backtest engine pipeline
- Verifying risk overlay and market impact model mechanics

References:
- Bollerslev (1986) "Generalized Autoregressive Conditional
  Heteroskedasticity", *Journal of Econometrics* 31(3).
- Merton (1976) "Option Pricing When Underlying Stock Returns Are
  Discontinuous", *Journal of Financial Economics* 3(1-2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar


@dataclass(frozen=True)
class MarketRegime:
    """Configuration for a market regime segment."""

    name: str
    drift: float  # Daily drift (annualised / 365)
    volatility: float  # Daily volatility
    duration_bars: int


@dataclass(frozen=True)
class RealisticNoiseConfig:
    """Parameters controlling realistic noise features.

    Attributes
    ----------
    df : float
        Degrees of freedom for Student-t innovations.  Lower values
        produce heavier tails.  ``df >= 30`` approximates Gaussian.
        Typical crypto: 3–5.
    garch_omega : float
        GARCH(1,1) constant term (long-run variance baseline).
    garch_alpha : float
        GARCH(1,1) ARCH coefficient (reaction to recent shocks).
    garch_beta : float
        GARCH(1,1) GARCH coefficient (persistence of variance).
        Constraint: ``garch_alpha + garch_beta < 1``.
    jump_intensity : float
        Expected number of jumps per bar (Poisson lambda).  0 disables.
    jump_mean : float
        Mean of log-normal jump size (in log-return space).
    jump_std : float
        Std of log-normal jump size.
    """

    df: float = 4.0
    garch_omega: float = 1e-6
    garch_alpha: float = 0.10
    garch_beta: float = 0.85
    jump_intensity: float = 0.02
    jump_mean: float = -0.03
    jump_std: float = 0.04


#: Default realistic noise config calibrated to crypto markets.
DEFAULT_NOISE_CONFIG = RealisticNoiseConfig()


def _student_t_sample(rng: random.Random, df: float) -> float:
    """Sample from a Student-t distribution using the ratio method.

    For ``df >= 30`` the result is nearly Gaussian.
    """
    if df >= 30:
        return rng.gauss(0, 1)
    # Use the fact that T = Z / sqrt(V/df) where Z ~ N(0,1), V ~ chi2(df)
    z = rng.gauss(0, 1)
    # chi2(df) = sum of df standard normals squared; approximate via gamma
    # Gamma(shape=df/2, scale=2) = chi2(df)
    v = rng.gammavariate(df / 2.0, 2.0)
    if v <= 0:
        return z  # degenerate guard
    return z / math.sqrt(v / df)


def generate_ohlcv(
    inst_id: str,
    regimes: list[MarketRegime],
    start_price: float = 100.0,
    start_time: datetime | None = None,
    bar_interval: timedelta = timedelta(days=1),
    seed: int | None = 42,
    noise_config: RealisticNoiseConfig | None = None,
    *,
    _correlated_shocks: list[float] | None = None,
) -> list[OhlcvBar]:
    """Generate synthetic OHLCV bars across market regimes.

    When *noise_config* is ``None`` the classic GBM generator is used
    (backward compatible).  Pass :data:`DEFAULT_NOISE_CONFIG` (or a
    custom instance) to enable fat tails, volatility clustering and
    jump diffusion.

    Parameters
    ----------
    _correlated_shocks :
        **Internal.**  Pre-generated correlated standard-normal shocks
        (one per bar) produced by :func:`generate_multi_instrument_data`
        via Cholesky decomposition.  When provided, these replace the
        idiosyncratic Gaussian draw so that cross-asset correlation is
        injected *before* the Student-t / GARCH / jump transforms.
    """
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

    bars: list[OhlcvBar] = []
    price = start_price
    t = start_time

    use_realistic = noise_config is not None
    if use_realistic:
        nc = noise_config
        # Initialise GARCH conditional variance to regime base vol²
        h_t = regimes[0].volatility ** 2 if regimes else 1e-4
    else:
        nc = None  # keep linter happy
        h_t = 0.0

    bar_idx = 0
    for regime in regimes:
        for _ in range(regime.duration_bars):
            if use_realistic:
                assert nc is not None
                # --- correlated or idiosyncratic innovation ----------
                if _correlated_shocks is not None and bar_idx < len(_correlated_shocks):
                    z_base = _correlated_shocks[bar_idx]
                else:
                    z_base = rng.gauss(0, 1)

                # Fat tails via Student-t scaling
                z = _student_t_sample(rng, nc.df) if nc.df < 30 else z_base
                # Preserve correlation direction from Cholesky but apply
                # fat-tail magnitude: use sign of correlated shock, abs of t.
                # This ensures cross-asset correlation structure (from
                # Cholesky decomposition) survives the Student-t transform
                # while still producing heavier tails.
                if _correlated_shocks is not None and bar_idx < len(_correlated_shocks):
                    z = math.copysign(abs(z), z_base) if z_base != 0 else z

                # GARCH(1,1) conditional variance update
                h_t = nc.garch_omega + nc.garch_alpha * (z ** 2) * h_t + nc.garch_beta * h_t
                h_t = max(h_t, 1e-10)
                cond_vol = math.sqrt(h_t)
                # Scale so long-run average ≈ regime.volatility
                vol = regime.volatility * cond_vol / max(
                    math.sqrt(nc.garch_omega / max(1 - nc.garch_alpha - nc.garch_beta, 0.01)),
                    1e-8,
                )

                ret = regime.drift + vol * z

                # Jump component (Merton 1976)
                if nc.jump_intensity > 0:
                    n_jumps = 0
                    # Poisson sampling via inverse CDF
                    p = math.exp(-nc.jump_intensity)
                    u = rng.random()
                    cum = p
                    while u > cum:
                        n_jumps += 1
                        p *= nc.jump_intensity / n_jumps
                        cum += p
                    if n_jumps > 0:
                        for _ in range(n_jumps):
                            ret += rng.gauss(nc.jump_mean, nc.jump_std)

                # Clamp return to avoid overflow in exp()
                ret = max(min(ret, 1.0), -1.0)
                new_price = price * math.exp(ret)
            else:
                # Legacy GBM path (backward compatible)
                noise = rng.gauss(0, 1)
                ret = regime.drift + regime.volatility * noise
                new_price = price * math.exp(ret)

            new_price = max(new_price, 0.01)

            # Generate OHLCV
            intrabar_vol = regime.volatility * 0.5
            high_mult = 1 + abs(rng.gauss(0, intrabar_vol))
            low_mult = 1 - abs(rng.gauss(0, intrabar_vol))

            bar_open = Decimal(str(round(price, 6)))
            bar_high = Decimal(str(round(max(price, new_price) * high_mult, 6)))
            bar_low = Decimal(str(round(min(price, new_price) * low_mult, 6)))
            bar_close = Decimal(str(round(new_price, 6)))
            bar_vol = Decimal(str(round(rng.uniform(100, 10000), 2)))

            bars.append(OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=bar_close,
                volume=bar_vol,
            ))
            price = new_price
            t += bar_interval
            bar_idx += 1

    return bars


def generate_funding_rates(
    inst_id: str,
    n_observations: int,
    start_time: datetime | None = None,
    interval: timedelta = timedelta(hours=8),
    base_rate: float = 0.0001,
    rate_volatility: float = 0.0003,
    seed: int | None = 42,
) -> list[FundingRate]:
    """Generate synthetic funding rate observations."""
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

    rates: list[FundingRate] = []
    t = start_time
    for _ in range(n_observations):
        rate = base_rate + rng.gauss(0, rate_volatility)
        rates.append(FundingRate(
            inst_id=inst_id,
            funding_time=t,
            funding_rate=Decimal(str(round(rate, 8))),
        ))
        t += interval
    return rates


# ---------------------------------------------------------------------------
# Pre-configured regime sets for testing
# ---------------------------------------------------------------------------

BULL_MARKET = MarketRegime("bull", drift=0.001, volatility=0.015, duration_bars=100)
BEAR_MARKET = MarketRegime("bear", drift=-0.001, volatility=0.02, duration_bars=80)
SIDEWAYS_MARKET = MarketRegime("sideways", drift=0.0, volatility=0.01, duration_bars=60)
VOLATILE_MARKET = MarketRegime("volatile", drift=0.0002, volatility=0.04, duration_bars=50)

FULL_CYCLE_REGIMES = [BULL_MARKET, BEAR_MARKET, SIDEWAYS_MARKET, VOLATILE_MARKET, BULL_MARKET]
"""A full market cycle: bull → bear → sideways → volatile → bull."""

MULTI_CYCLE_REGIMES = FULL_CYCLE_REGIMES * 2
"""Two full market cycles for longer backtests (780 bars)."""

# ---------------------------------------------------------------------------
# Extended regime sets for 3+ year OOS validation (1095+ bars)
# ---------------------------------------------------------------------------

# More realistic crypto market regime durations and magnitudes.
# Calibrated to approximate BTC/ETH historical characteristics:
# - Bull markets: annualised drift ~100-200%, daily vol ~3-4%
# - Bear markets: annualised drift ~-50-80%, daily vol ~4-5%
# - Crashes: sharp drawdowns with extreme vol
# - Sideways: low drift, compressed volatility
DEEP_BEAR = MarketRegime("deep_bear", drift=-0.002, volatility=0.04, duration_bars=120)
ACCUMULATION = MarketRegime("accumulation", drift=0.0005, volatility=0.02, duration_bars=90)
STRONG_BULL = MarketRegime("strong_bull", drift=0.003, volatility=0.03, duration_bars=130)
DISTRIBUTION = MarketRegime("distribution", drift=-0.0003, volatility=0.035, duration_bars=70)
CRASH = MarketRegime("crash", drift=-0.005, volatility=0.06, duration_bars=30)
RECOVERY = MarketRegime("recovery", drift=0.004, volatility=0.035, duration_bars=50)

THREE_YEAR_REGIMES = [
    STRONG_BULL,        # 130 bars - early bull market
    DISTRIBUTION,       # 70 bars  - topping pattern
    DEEP_BEAR,          # 120 bars - bear market
    CRASH,              # 30 bars  - capitulation event
    ACCUMULATION,       # 90 bars  - bottoming phase
    RECOVERY,           # 50 bars  - recovery rally
    BULL_MARKET,        # 100 bars - moderate bull
    SIDEWAYS_MARKET,    # 60 bars  - consolidation
    VOLATILE_MARKET,    # 50 bars  - high vol expansion
    STRONG_BULL,        # 130 bars - late bull
    DISTRIBUTION,       # 70 bars  - second distribution
    BEAR_MARKET,        # 80 bars  - correction
    RECOVERY,           # 50 bars  - bounce
    ACCUMULATION,       # 90 bars  - re-accumulation
]
"""3+ year regime sequence (1120 bars) covering bull, deep bear, crash, recovery."""


def _cholesky_decompose(matrix: list[list[float]]) -> list[list[float]]:
    """Cholesky decomposition of a positive-definite matrix.

    Returns lower-triangular matrix L such that ``matrix = L @ L^T``.
    Pure-Python implementation (no numpy dependency).
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                L[i][j] = math.sqrt(max(val, 0.0))
            else:
                L[i][j] = (matrix[i][j] - s) / L[j][j] if L[j][j] != 0 else 0.0
    return L


def build_correlation_matrix(
    n: int,
    *,
    base_corr: float = 0.6,
    stress_corr: float = 0.90,
) -> list[list[float]]:
    """Build an *n × n* correlation matrix for crypto assets.

    Uses *base_corr* for off-diagonal elements (normal-market pairwise
    correlation).  The caller may further adjust during stress regimes
    via *stress_corr*.
    """
    return [
        [1.0 if i == j else base_corr for j in range(n)]
        for i in range(n)
    ]


def generate_multi_instrument_data(
    inst_ids: list[str],
    regimes: list[MarketRegime] | None = None,
    seed_base: int = 42,
    noise_config: RealisticNoiseConfig | None = None,
    correlation: float | list[list[float]] | None = None,
) -> tuple[dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]]:
    """Generate OHLCV and funding data for multiple instruments.

    Each instrument gets slightly different parameters (seeded
    differently) to simulate realistic cross-sectional variation.

    Parameters
    ----------
    noise_config :
        Pass :data:`DEFAULT_NOISE_CONFIG` to enable fat tails,
        volatility clustering and jump diffusion.
    correlation :
        Cross-asset correlation.  Pass a ``float`` for uniform pairwise
        correlation or a full correlation matrix.  When set, Cholesky-
        decomposed correlated shocks are injected into every instrument.
    """
    if regimes is None:
        regimes = FULL_CYCLE_REGIMES

    bars_by_inst: dict[str, list[OhlcvBar]] = {}
    funding_by_inst: dict[str, list[FundingRate]] = {}

    total_bars = sum(r.duration_bars for r in regimes)
    n = len(inst_ids)

    # --- build correlated shocks via Cholesky ---------------------------
    correlated_shocks: list[list[float]] | None = None
    if correlation is not None and n > 1:
        if isinstance(correlation, (int, float)):
            corr_matrix = build_correlation_matrix(n, base_corr=float(correlation))
        else:
            corr_matrix = correlation
        L = _cholesky_decompose(corr_matrix)
        rng_corr = random.Random(seed_base + 999)
        # Generate independent standard normals and multiply by L
        correlated_shocks = [[] for _ in range(n)]
        for _ in range(total_bars):
            z_indep = [rng_corr.gauss(0, 1) for _ in range(n)]
            for i in range(n):
                corr_z = sum(L[i][j] * z_indep[j] for j in range(i + 1))
                correlated_shocks[i].append(corr_z)

    for i, inst_id in enumerate(inst_ids):
        seed = seed_base + i * 100
        start_price = 50.0 + i * 30  # Different starting prices

        shocks_i = correlated_shocks[i] if correlated_shocks is not None else None

        bars = generate_ohlcv(
            inst_id=inst_id,
            regimes=regimes,
            start_price=start_price,
            seed=seed,
            noise_config=noise_config,
            _correlated_shocks=shocks_i,
        )
        bars_by_inst[inst_id] = bars

        # 3 funding observations per day (8h interval)
        funding = generate_funding_rates(
            inst_id=inst_id,
            n_observations=total_bars * 3,
            base_rate=0.0001 + i * 0.00005,
            seed=seed + 1,
        )
        funding_by_inst[inst_id] = funding

    return bars_by_inst, funding_by_inst
