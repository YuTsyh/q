"""Parameter perturbation robustness validation (Monte Carlo sensitivity).

Perturbs strategy parameters by small random amounts across many simulations
and checks that performance does not degrade beyond an acceptable threshold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from quantbot.research.backtest import BacktestConfig, BacktestEngine, StrategyAllocator
from quantbot.research.data import FundingRate, OhlcvBar
from quantbot.research.metrics import PerformanceMetrics  # noqa: F401


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for parameter perturbation analysis."""

    n_simulations: int = 100
    perturbation_pct: float = 0.10
    max_degradation_pct: float = 0.20
    seed: int = 42


@dataclass(frozen=True)
class PerturbationResult:
    """Result of parameter perturbation robustness analysis."""

    base_sharpe: float
    base_cagr: float
    base_max_dd: float
    perturbed_sharpes: list[float]
    perturbed_cagrs: list[float]
    perturbed_max_dds: list[float]
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    sharpe_degradation_pct: float
    cagr_degradation_pct: float
    max_dd_degradation_pct: float
    passed: bool
    n_simulations: int


def perturb_parameters(
    params: dict[str, float],
    pct: float,
    rng: random.Random,
) -> dict[str, float]:
    """Return a copy of *params* with each value randomly perturbed by ±*pct*.

    Positive base values are clamped so the perturbed value stays positive.
    """
    perturbed: dict[str, float] = {}
    for key, value in params.items():
        factor = 1.0 + rng.uniform(-pct, pct)
        new_value = value * factor
        if value > 0.0:
            new_value = max(new_value, 1e-12)
        perturbed[key] = new_value
    return perturbed


def run_perturbation_analysis(
    allocator_factory: Callable[..., StrategyAllocator],
    base_params: dict[str, float],
    bars_by_instrument: dict[str, list[OhlcvBar]],
    funding_by_instrument: dict[str, list[FundingRate]],
    backtest_config: BacktestConfig,
    perturbation_config: PerturbationConfig | None = None,
    min_history: int = 5,
) -> PerturbationResult:
    """Run parameter perturbation robustness analysis.

    1. Run a base backtest with *base_params*.
    2. For each simulation, randomly perturb every parameter by
       ±perturbation_pct, create an allocator, and run a backtest.
    3. Compute distribution statistics and check degradation thresholds.

    Raises:
        ValueError: If the base backtest fails.
    """
    cfg = perturbation_config or PerturbationConfig()
    engine = BacktestEngine(backtest_config)

    # --- Base run ---
    base_allocator = allocator_factory(**base_params)
    base_result = engine.run(
        base_allocator, bars_by_instrument, funding_by_instrument, min_history,
    )
    base_sharpe = base_result.metrics.sharpe_ratio
    base_cagr = base_result.metrics.cagr
    base_max_dd = base_result.metrics.max_drawdown

    # --- Perturbation runs ---
    rng = random.Random(cfg.seed)
    perturbed_sharpes: list[float] = []
    perturbed_cagrs: list[float] = []
    perturbed_max_dds: list[float] = []

    for _ in range(cfg.n_simulations):
        tweaked = perturb_parameters(base_params, cfg.perturbation_pct, rng)
        try:
            allocator = allocator_factory(**tweaked)
            result = engine.run(
                allocator, bars_by_instrument, funding_by_instrument, min_history,
            )
        except Exception:  # noqa: BLE001
            continue
        perturbed_sharpes.append(result.metrics.sharpe_ratio)
        perturbed_cagrs.append(result.metrics.cagr)
        perturbed_max_dds.append(result.metrics.max_drawdown)

    if not perturbed_sharpes:
        raise ValueError(
            "All perturbation simulations failed; cannot compute statistics"
        )

    # --- Statistics ---
    perturbed_sharpes.sort()
    perturbed_cagrs.sort()
    perturbed_max_dds.sort()
    n = len(perturbed_sharpes)

    median_sharpe = perturbed_sharpes[n // 2]
    p5_sharpe = perturbed_sharpes[int(n * 0.05)]
    p95_sharpe = perturbed_sharpes[int(min(n * 0.95, n - 1))]

    median_cagr = perturbed_cagrs[n // 2]
    median_max_dd = perturbed_max_dds[n // 2]

    # --- Degradation checks ---
    sharpe_deg = _degradation_pct(base_sharpe, median_sharpe)
    cagr_deg = _degradation_pct(base_cagr, median_cagr)
    # For max_dd a *higher* value is worse, so invert the check.
    max_dd_deg = _dd_degradation_pct(base_max_dd, median_max_dd)

    passed = (
        sharpe_deg <= cfg.max_degradation_pct
        and cagr_deg <= cfg.max_degradation_pct
        and max_dd_deg <= cfg.max_degradation_pct
    )

    return PerturbationResult(
        base_sharpe=base_sharpe,
        base_cagr=base_cagr,
        base_max_dd=base_max_dd,
        perturbed_sharpes=perturbed_sharpes,
        perturbed_cagrs=perturbed_cagrs,
        perturbed_max_dds=perturbed_max_dds,
        median_sharpe=median_sharpe,
        p5_sharpe=p5_sharpe,
        p95_sharpe=p95_sharpe,
        sharpe_degradation_pct=sharpe_deg,
        cagr_degradation_pct=cagr_deg,
        max_dd_degradation_pct=max_dd_deg,
        passed=passed,
        n_simulations=n,
    )


def format_perturbation_report(result: PerturbationResult) -> str:
    """Return a human-readable perturbation robustness report."""
    status = "PASSED" if result.passed else "FAILED"
    lines = [
        "=" * 60,
        f"  Parameter Perturbation Robustness Report  [{status}]",
        "=" * 60,
        f"  Simulations completed: {result.n_simulations}",
        "",
        "  Metric          Base      Median-Perturbed  Degradation",
        "  " + "-" * 56,
        (
            f"  Sharpe     {result.base_sharpe:>10.4f}"
            f"  {result.median_sharpe:>10.4f}"
            f"       {result.sharpe_degradation_pct:>6.2%}"
        ),
        (
            f"  CAGR       {result.base_cagr:>10.4f}"
            f"  {_median(result.perturbed_cagrs):>10.4f}"
            f"       {result.cagr_degradation_pct:>6.2%}"
        ),
        (
            f"  Max DD     {result.base_max_dd:>10.4f}"
            f"  {_median(result.perturbed_max_dds):>10.4f}"
            f"       {result.max_dd_degradation_pct:>6.2%}"
        ),
        "",
        f"  Sharpe distribution: p5={result.p5_sharpe:.4f}  "
        f"median={result.median_sharpe:.4f}  p95={result.p95_sharpe:.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _degradation_pct(base: float, perturbed: float) -> float:
    """Fraction by which *perturbed* is worse than *base* (higher-is-better).

    Returns 0.0 when performance improved or when the base is non-positive.
    """
    if base <= 0.0:
        return 0.0
    return max(0.0, (base - perturbed) / base)


def _dd_degradation_pct(base_dd: float, perturbed_dd: float) -> float:
    """Fraction by which drawdown worsened (larger dd is worse).

    Returns 0.0 when drawdown improved or the base is effectively zero.
    """
    if base_dd <= 1e-12:
        return 0.0 if perturbed_dd <= 1e-12 else perturbed_dd
    return max(0.0, (perturbed_dd - base_dd) / base_dd)


def _median(values: list[float]) -> float:
    return values[len(values) // 2] if values else 0.0
