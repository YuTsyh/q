"""Tests for parameter perturbation robustness analysis."""

from __future__ import annotations

import random

import pytest

from quantbot.research.robustness import (
    PerturbationResult,
    format_perturbation_report,
    perturb_parameters,
)


# ---------------------------------------------------------------------------
# Tests for perturb_parameters
# ---------------------------------------------------------------------------


class TestPerturbParameters:
    def test_values_change(self) -> None:
        params = {"lookback": 20.0, "threshold": 0.5, "alpha": 1.0}
        rng = random.Random(42)
        perturbed = perturb_parameters(params, pct=0.10, rng=rng)
        assert set(perturbed.keys()) == set(params.keys())
        # At least one value should differ (extremely unlikely all are identical)
        assert any(perturbed[k] != params[k] for k in params)

    def test_values_within_range(self) -> None:
        params = {"x": 100.0, "y": 50.0}
        rng = random.Random(123)
        for _ in range(50):
            perturbed = perturb_parameters(params, pct=0.10, rng=rng)
            for key, base in params.items():
                assert perturbed[key] >= base * 0.9 - 1e-9
                assert perturbed[key] <= base * 1.1 + 1e-9

    def test_positive_values_stay_positive(self) -> None:
        params = {"small": 0.001, "big": 1000.0}
        rng = random.Random(99)
        for _ in range(100):
            perturbed = perturb_parameters(params, pct=0.10, rng=rng)
            assert perturbed["small"] > 0
            assert perturbed["big"] > 0

    def test_zero_value_stays_zero(self) -> None:
        params = {"zero_param": 0.0}
        rng = random.Random(42)
        perturbed = perturb_parameters(params, pct=0.10, rng=rng)
        assert perturbed["zero_param"] == 0.0

    def test_negative_value_perturbed(self) -> None:
        params = {"neg": -10.0}
        rng = random.Random(42)
        perturbed = perturb_parameters(params, pct=0.10, rng=rng)
        assert perturbed["neg"] != -10.0
        # Should still be in the neighbourhood
        assert abs(perturbed["neg"] - (-10.0)) < 2.0

    def test_deterministic_with_same_seed(self) -> None:
        params = {"a": 1.0, "b": 2.0}
        r1 = perturb_parameters(params, pct=0.10, rng=random.Random(42))
        r2 = perturb_parameters(params, pct=0.10, rng=random.Random(42))
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests for format_perturbation_report
# ---------------------------------------------------------------------------


class TestFormatPerturbationReport:
    def _make_result(self, passed: bool = True) -> PerturbationResult:
        return PerturbationResult(
            base_sharpe=2.5,
            base_cagr=0.35,
            base_max_dd=0.10,
            perturbed_sharpes=[2.0, 2.2, 2.4, 2.5, 2.6],
            perturbed_cagrs=[0.30, 0.32, 0.34, 0.35, 0.36],
            perturbed_max_dds=[0.08, 0.09, 0.10, 0.11, 0.12],
            median_sharpe=2.4,
            p5_sharpe=2.0,
            p95_sharpe=2.6,
            sharpe_degradation_pct=0.04,
            cagr_degradation_pct=0.03,
            max_dd_degradation_pct=0.10,
            passed=passed,
            n_simulations=5,
        )

    def test_returns_string(self) -> None:
        report = format_perturbation_report(self._make_result())
        assert isinstance(report, str)

    def test_contains_sections(self) -> None:
        report = format_perturbation_report(self._make_result())
        assert "Robustness Report" in report
        assert "Sharpe" in report
        assert "CAGR" in report
        assert "Max DD" in report
        assert "Simulations completed" in report

    def test_passed_status_shown(self) -> None:
        report = format_perturbation_report(self._make_result(passed=True))
        assert "PASSED" in report

    def test_failed_status_shown(self) -> None:
        report = format_perturbation_report(self._make_result(passed=False))
        assert "FAILED" in report

    def test_distribution_info_present(self) -> None:
        report = format_perturbation_report(self._make_result())
        assert "p5=" in report
        assert "median=" in report
        assert "p95=" in report
