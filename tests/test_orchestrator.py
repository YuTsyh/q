"""Tests for the multi-agent orchestration framework."""

from __future__ import annotations

import pytest

from quantbot.research.metrics import PerformanceMetrics
from quantbot.research.orchestrator import (
    AuditAgent,
    DAGPlanner,
    IterationResult,
    MemoryAgent,
    MemoryEntry,
    Orchestrator,
    TaskNode,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_metrics(
    sharpe: float = 2.5,
    cagr: float = 0.35,
    max_dd: float = 0.10,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=1.0,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=3.0,
        max_drawdown=max_dd,
        calmar_ratio=3.5,
        profit_factor=2.0,
        win_rate=0.6,
        expectancy=0.01,
        total_trades=100,
        avg_trade_return=0.001,
        max_consecutive_losses=5,
        annualised_volatility=0.15,
    )


def _make_entry(
    iteration: int = 1,
    success: bool = True,
    sharpe: float = 2.5,
    failure_reason: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        iteration=iteration,
        strategy_name="test_strategy",
        parameters={"lookback": 20.0, "threshold": 0.5},
        metrics={"sharpe": sharpe, "cagr": 0.35},
        success=success,
        failure_reason=failure_reason,
        timestamp="2024-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# MemoryAgent tests
# ---------------------------------------------------------------------------


class TestMemoryAgent:
    def test_record_and_summary(self) -> None:
        mem = MemoryAgent()
        mem.record(_make_entry(iteration=1, success=True, sharpe=2.0))
        mem.record(_make_entry(iteration=2, success=False, sharpe=1.0, failure_reason="low Sharpe"))
        summary = mem.summary()
        assert summary["total_iterations"] == 2
        assert summary["successes"] == 1
        assert summary["failures"] == 1

    def test_has_been_tried_after_failure(self) -> None:
        mem = MemoryAgent()
        entry = _make_entry(success=False, failure_reason="failed")
        mem.record(entry)
        sig = mem._make_signature(entry.strategy_name, entry.parameters)
        assert mem.has_been_tried(sig)

    def test_has_been_tried_false_for_new(self) -> None:
        mem = MemoryAgent()
        assert not mem.has_been_tried("unknown|x=1")

    def test_get_best_result(self) -> None:
        mem = MemoryAgent()
        mem.record(_make_entry(iteration=1, sharpe=1.5))
        mem.record(_make_entry(iteration=2, sharpe=3.0))
        mem.record(_make_entry(iteration=3, sharpe=2.0))
        best = mem.get_best_result()
        assert best is not None
        assert best.metrics["sharpe"] == 3.0

    def test_get_best_result_empty(self) -> None:
        mem = MemoryAgent()
        assert mem.get_best_result() is None

    def test_get_failure_reasons(self) -> None:
        mem = MemoryAgent()
        mem.record(_make_entry(failure_reason="too slow"))
        mem.record(_make_entry(failure_reason="low Sharpe"))
        mem.record(_make_entry(failure_reason="too slow"))  # duplicate
        reasons = mem.get_failure_reasons()
        assert "too slow" in reasons
        assert "low Sharpe" in reasons
        assert len(reasons) == 2  # no duplicates

    def test_summary_best_metrics(self) -> None:
        mem = MemoryAgent()
        mem.record(_make_entry(sharpe=2.5))
        summary = mem.summary()
        assert summary["best_metrics"]["sharpe"] == 2.5


# ---------------------------------------------------------------------------
# AuditAgent tests
# ---------------------------------------------------------------------------


class TestAuditAgent:
    def test_all_pass(self) -> None:
        audit = AuditAgent(target_sharpe=2.0, target_cagr=0.30, target_max_dd=0.15)
        metrics = _make_metrics(sharpe=2.5, cagr=0.35, max_dd=0.10)
        passed, issues = audit.diagnose(metrics)
        assert passed is True
        assert issues == []

    def test_sharpe_too_low(self) -> None:
        audit = AuditAgent(target_sharpe=2.0)
        metrics = _make_metrics(sharpe=1.0)
        passed, issues = audit.diagnose(metrics)
        assert passed is False
        assert any("Sharpe" in i for i in issues)

    def test_cagr_too_low(self) -> None:
        audit = AuditAgent(target_cagr=0.50)
        metrics = _make_metrics(cagr=0.20)
        passed, issues = audit.diagnose(metrics)
        assert passed is False
        assert any("CAGR" in i for i in issues)

    def test_max_dd_too_high(self) -> None:
        audit = AuditAgent(target_max_dd=0.05)
        metrics = _make_metrics(max_dd=0.20)
        passed, issues = audit.diagnose(metrics)
        assert passed is False
        assert any("drawdown" in i.lower() for i in issues)

    def test_multiple_failures(self) -> None:
        audit = AuditAgent(target_sharpe=5.0, target_cagr=0.90, target_max_dd=0.01)
        metrics = _make_metrics(sharpe=1.0, cagr=0.10, max_dd=0.30)
        passed, issues = audit.diagnose(metrics)
        assert passed is False
        assert len(issues) == 3

    def test_suggest_adjustments_for_failing(self) -> None:
        audit = AuditAgent(target_sharpe=5.0, target_cagr=0.90, target_max_dd=0.01)
        metrics = _make_metrics(sharpe=1.0, cagr=0.10, max_dd=0.30)
        suggestions = audit.suggest_adjustments(metrics)
        assert isinstance(suggestions, dict)
        assert len(suggestions) > 0
        assert "risk" in suggestions or "factors" in suggestions

    def test_suggest_adjustments_empty_when_passing(self) -> None:
        audit = AuditAgent(target_sharpe=1.0, target_cagr=0.10, target_max_dd=0.50)
        metrics = _make_metrics(sharpe=2.5, cagr=0.35, max_dd=0.10)
        suggestions = audit.suggest_adjustments(metrics)
        assert suggestions == {}


# ---------------------------------------------------------------------------
# DAGPlanner tests
# ---------------------------------------------------------------------------


class TestDAGPlanner:
    def test_create_initial_dag_returns_seven_tasks(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        assert len(dag) == 7
        ids = [t.task_id for t in dag]
        assert "data_load" in ids
        assert "audit" in ids

    def test_initial_dag_correct_dependencies(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        task_map = {t.task_id: t for t in dag}
        assert task_map["data_load"].dependencies == []
        assert "data_load" in task_map["factor_mining"].dependencies
        assert "data_load" in task_map["regime_detection"].dependencies
        assert "factor_mining" in task_map["portfolio_construction"].dependencies
        assert "regime_detection" in task_map["portfolio_construction"].dependencies
        assert "portfolio_construction" in task_map["backtest"].dependencies
        assert "backtest" in task_map["validation"].dependencies
        assert "validation" in task_map["audit"].dependencies

    def test_get_ready_tasks_only_data_load_initially(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        ready = planner.get_ready_tasks(dag)
        assert len(ready) == 1
        assert ready[0].task_id == "data_load"

    def test_mark_completed_advances_ready_tasks(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        planner.mark_completed(dag, "data_load", result="data loaded")
        ready = planner.get_ready_tasks(dag)
        ready_ids = {t.task_id for t in ready}
        assert "factor_mining" in ready_ids
        assert "regime_detection" in ready_ids

    def test_mark_failed_retry_logic(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        task_map = {t.task_id: t for t in dag}

        # First failure → reset to PENDING for retry
        planner.mark_failed(dag, "data_load", "timeout")
        assert task_map["data_load"].status == TaskStatus.PENDING
        assert task_map["data_load"].attempts == 1

        # Second failure → still PENDING
        planner.mark_failed(dag, "data_load", "timeout again")
        assert task_map["data_load"].status == TaskStatus.PENDING
        assert task_map["data_load"].attempts == 2

        # Third failure → max_attempts (3) reached → FAILED
        planner.mark_failed(dag, "data_load", "final timeout")
        assert task_map["data_load"].status == TaskStatus.FAILED
        assert task_map["data_load"].attempts == 3

    def test_mark_completed_stores_result(self) -> None:
        planner = DAGPlanner(MemoryAgent(), AuditAgent())
        dag = planner.create_initial_dag()
        planner.mark_completed(dag, "data_load", result={"bars": 100})
        task_map = {t.task_id: t for t in dag}
        assert task_map["data_load"].result == {"bars": 100}
        assert task_map["data_load"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_plan_next_iteration_increments_counter(self) -> None:
        orch = Orchestrator(max_iterations=5)
        dag1 = orch.plan_next_iteration()
        assert len(dag1) == 7
        dag2 = orch.plan_next_iteration()
        assert len(dag2) == 7

    def test_record_iteration_stores_result(self) -> None:
        orch = Orchestrator(max_iterations=5)
        orch.plan_next_iteration()
        metrics = _make_metrics(sharpe=2.5, cagr=0.35, max_dd=0.10)
        orch.record_iteration(metrics, "test_strat", {"lookback": 20.0})
        summary = orch.summary()
        assert summary["total_results"] == 1

    def test_should_continue_false_after_pass(self) -> None:
        orch = Orchestrator(max_iterations=10)
        orch.plan_next_iteration()
        # Passing metrics
        metrics = _make_metrics(sharpe=2.5, cagr=0.35, max_dd=0.10)
        orch.record_iteration(metrics, "good_strat", {})
        assert orch.should_continue() is False

    def test_should_continue_false_after_max_iterations(self) -> None:
        orch = Orchestrator(max_iterations=2)
        for _ in range(2):
            orch.plan_next_iteration()
            # Failing metrics
            metrics = _make_metrics(sharpe=0.5, cagr=0.05, max_dd=0.50)
            orch.record_iteration(metrics, "bad_strat", {})
        assert orch.should_continue() is False

    def test_should_continue_true_when_no_pass(self) -> None:
        orch = Orchestrator(max_iterations=10)
        orch.plan_next_iteration()
        metrics = _make_metrics(sharpe=0.5, cagr=0.05, max_dd=0.50)
        orch.record_iteration(metrics, "bad_strat", {})
        assert orch.should_continue() is True

    def test_get_best_result(self) -> None:
        orch = Orchestrator(max_iterations=5)
        orch.plan_next_iteration()
        orch.record_iteration(_make_metrics(sharpe=1.0), "s1", {})
        orch.plan_next_iteration()
        orch.record_iteration(_make_metrics(sharpe=3.0), "s2", {})
        best = orch.get_best_result()
        assert best is not None
        assert best.metrics.sharpe_ratio == 3.0

    def test_get_best_result_none_when_empty(self) -> None:
        orch = Orchestrator()
        assert orch.get_best_result() is None

    def test_summary_structure(self) -> None:
        orch = Orchestrator(max_iterations=5)
        orch.plan_next_iteration()
        orch.record_iteration(_make_metrics(), "strat", {})
        summary = orch.summary()
        assert "iterations_completed" in summary
        assert "max_iterations" in summary
        assert "passed" in summary
        assert "memory" in summary
        assert "best_iteration" in summary

    def test_record_iteration_with_none_metrics(self) -> None:
        orch = Orchestrator(max_iterations=5)
        orch.plan_next_iteration()
        orch.record_iteration(None, "failed_strat", {})
        summary = orch.summary()
        assert summary["total_results"] == 1
        assert orch.should_continue() is True
