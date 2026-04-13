"""Multi-agent orchestration framework for iterative strategy development.

Coordinates a directed-acyclic-graph (DAG) of research tasks across
iterations.  A *MemoryAgent* tracks what has been tried, an *AuditAgent*
diagnoses backtest shortcomings, and a *DAGPlanner* builds the next
iteration's work-plan.  The top-level *Orchestrator* drives the loop
until performance targets are met or the iteration budget is exhausted.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

from quantbot.research.metrics import PerformanceMetrics


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(enum.Enum):
    """Lifecycle state of a single DAG task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Task DAG node
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    """A single unit of work inside an iteration DAG.

    Parameters
    ----------
    task_id:
        Unique identifier within the DAG.
    name:
        Human-readable task name.
    description:
        What the task does.
    dependencies:
        ``task_id`` values that must reach COMPLETED before this task
        may start.
    """

    task_id: str
    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: object | None = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 3


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryEntry:
    """Immutable record of one strategy evaluation attempt."""

    iteration: int
    strategy_name: str
    parameters: dict[str, float]
    metrics: dict[str, float]
    success: bool
    failure_reason: str
    timestamp: str


class MemoryAgent:
    """Persistent memory across iterations.

    Tracks every strategy attempt so the planner can avoid repeating
    configurations that already failed.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._failed_strategies: set[str] = set()

    # -- public API ---------------------------------------------------------

    def record(self, entry: MemoryEntry) -> None:
        """Append *entry* and update the failure index."""
        self._entries.append(entry)
        if not entry.success:
            sig = self._make_signature(entry.strategy_name, entry.parameters)
            self._failed_strategies.add(sig)

    def has_been_tried(self, strategy_signature: str) -> bool:
        """Return ``True`` if *strategy_signature* was already tried and failed."""
        return strategy_signature in self._failed_strategies

    def get_best_result(self) -> MemoryEntry | None:
        """Return the entry with the highest Sharpe ratio, or ``None``."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.metrics.get("sharpe", float("-inf")))

    def get_failure_reasons(self) -> list[str]:
        """Return all unique failure reasons across every entry."""
        seen: set[str] = set()
        reasons: list[str] = []
        for entry in self._entries:
            if entry.failure_reason and entry.failure_reason not in seen:
                seen.add(entry.failure_reason)
                reasons.append(entry.failure_reason)
        return reasons

    def summary(self) -> dict[str, object]:
        """Return high-level statistics about the run so far."""
        successes = [e for e in self._entries if e.success]
        failures = [e for e in self._entries if not e.success]
        best = self.get_best_result()
        return {
            "total_iterations": len(self._entries),
            "successes": len(successes),
            "failures": len(failures),
            "best_metrics": best.metrics if best is not None else {},
        }

    # -- internal -----------------------------------------------------------

    def _make_signature(self, strategy_name: str, parameters: dict[str, float]) -> str:
        """Create a deterministic string key for a strategy configuration."""
        sorted_params = sorted(parameters.items())
        return f"{strategy_name}|{'|'.join(f'{k}={v}' for k, v in sorted_params)}"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditAgent:
    """Evaluates backtest metrics against hard performance targets.

    Parameters
    ----------
    target_sharpe:
        Minimum acceptable annualised Sharpe ratio.
    target_cagr:
        Minimum acceptable compound annual growth rate (as a fraction).
    target_max_dd:
        Maximum acceptable peak-to-trough drawdown (as a positive fraction).
    """

    def __init__(
        self,
        target_sharpe: float = 2.0,
        target_cagr: float = 0.30,
        target_max_dd: float = 0.15,
    ) -> None:
        self.target_sharpe = target_sharpe
        self.target_cagr = target_cagr
        self.target_max_dd = target_max_dd

    def diagnose(self, metrics: PerformanceMetrics) -> tuple[bool, list[str]]:
        """Check *metrics* against targets.

        Returns
        -------
        tuple[bool, list[str]]
            ``(passed, issues)`` where *passed* is ``True`` only when
            every target is satisfied and *issues* lists human-readable
            descriptions of any shortcomings.
        """
        issues: list[str] = []

        if metrics.sharpe_ratio < self.target_sharpe:
            issues.append(
                f"Sharpe {metrics.sharpe_ratio:.4f} below target {self.target_sharpe}"
            )
        if metrics.cagr < self.target_cagr:
            issues.append(
                f"CAGR {metrics.cagr:.4f} below target {self.target_cagr}"
            )
        if metrics.max_drawdown > self.target_max_dd:
            issues.append(
                f"Max drawdown {metrics.max_drawdown:.4f} above target {self.target_max_dd}"
            )

        return (len(issues) == 0, issues)

    def suggest_adjustments(self, metrics: PerformanceMetrics) -> dict[str, str]:
        """Produce actionable improvement hints based on failing metrics.

        Returns
        -------
        dict[str, str]
            Mapping from category (e.g. ``"risk"``, ``"factors"``) to a
            plain-English suggestion.
        """
        suggestions: dict[str, str] = {}

        if metrics.sharpe_ratio < self.target_sharpe:
            suggestions["factors"] = "add mean-reversion"
            suggestions["signal"] = "increase factor diversity"

        if metrics.cagr < self.target_cagr:
            suggestions["leverage"] = "increase position sizing"
            suggestions["alpha"] = "mine additional alpha factors"

        if metrics.max_drawdown > self.target_max_dd:
            suggestions["risk"] = "reduce exposure"
            suggestions["hedging"] = "strengthen stablecoin hedge"

        return suggestions


# ---------------------------------------------------------------------------
# DAG planner
# ---------------------------------------------------------------------------

_INITIAL_TASKS: list[tuple[str, str, str, list[str]]] = [
    ("data_load", "data_load", "Load/generate multi-instrument data", []),
    ("factor_mining", "factor_mining", "Compute crypto-native factors", ["data_load"]),
    ("regime_detection", "regime_detection", "Run Markov regime detection", ["data_load"]),
    (
        "portfolio_construction",
        "portfolio_construction",
        "Build portfolio with vol scaling + stablecoin hedge",
        ["factor_mining", "regime_detection"],
    ),
    ("backtest", "backtest", "Run backtest with enhanced simulator", ["portfolio_construction"]),
    (
        "validation",
        "validation",
        "Walk-forward + Monte Carlo + parameter perturbation",
        ["backtest"],
    ),
    ("audit", "audit", "Evaluate results against targets", ["validation"]),
]


class DAGPlanner:
    """Builds and manages task DAGs for each research iteration.

    Parameters
    ----------
    memory:
        Shared memory agent for cross-iteration learning.
    audit:
        Audit agent used to inform DAG adjustments.
    """

    def __init__(self, memory: MemoryAgent, audit: AuditAgent) -> None:
        self.memory = memory
        self.audit = audit

    # -- DAG creation -------------------------------------------------------

    def create_initial_dag(self) -> list[TaskNode]:
        """Return the canonical seven-step research DAG."""
        return [
            TaskNode(
                task_id=tid,
                name=name,
                description=desc,
                dependencies=list(deps),
            )
            for tid, name, desc, deps in _INITIAL_TASKS
        ]

    def create_iteration_dag(self, iteration: int) -> list[TaskNode]:
        """Build a DAG for *iteration*, adjusting for past failures.

        On the first iteration the canonical DAG is returned unchanged.
        Subsequent iterations annotate task descriptions with audit
        suggestions so downstream executors can adapt their behaviour.
        """
        dag = self.create_initial_dag()

        if iteration <= 1:
            return dag

        best = self.memory.get_best_result()
        if best is None:
            return dag

        # Construct a lightweight PerformanceMetrics from the stored dict
        # so the audit agent can produce suggestions.
        metrics_dict = best.metrics
        perf = PerformanceMetrics(
            total_return=metrics_dict.get("total_return", 0.0),
            cagr=metrics_dict.get("cagr", 0.0),
            sharpe_ratio=metrics_dict.get("sharpe", 0.0),
            sortino_ratio=metrics_dict.get("sortino", 0.0),
            max_drawdown=metrics_dict.get("max_dd", 0.0),
            calmar_ratio=metrics_dict.get("calmar", 0.0),
            profit_factor=metrics_dict.get("profit_factor", 0.0),
            win_rate=metrics_dict.get("win_rate", 0.0),
            expectancy=metrics_dict.get("expectancy", 0.0),
            total_trades=int(metrics_dict.get("total_trades", 0)),
            avg_trade_return=metrics_dict.get("avg_trade_return", 0.0),
            max_consecutive_losses=int(metrics_dict.get("max_consecutive_losses", 0)),
            annualised_volatility=metrics_dict.get("annualised_volatility", 0.0),
        )

        suggestions = self.audit.suggest_adjustments(perf)
        if not suggestions:
            return dag

        hint = "; ".join(f"{k}: {v}" for k, v in suggestions.items())
        node_map = {t.task_id: t for t in dag}

        for target in ("factor_mining", "portfolio_construction"):
            node = node_map.get(target)
            if node is not None:
                node.description = f"{node.description} [adjust: {hint}]"

        return dag

    # -- DAG helpers --------------------------------------------------------

    def get_ready_tasks(self, dag: list[TaskNode]) -> list[TaskNode]:
        """Return tasks whose dependencies are all COMPLETED."""
        completed_ids = {t.task_id for t in dag if t.status is TaskStatus.COMPLETED}
        return [
            t
            for t in dag
            if t.status is TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def mark_completed(self, dag: list[TaskNode], task_id: str, result: object) -> None:
        """Set *task_id* to COMPLETED and attach *result*."""
        for task in dag:
            if task.task_id == task_id:
                task.status = TaskStatus.COMPLETED
                task.result = result
                return

    def mark_failed(self, dag: list[TaskNode], task_id: str, error: str) -> None:
        """Increment attempts for *task_id*.

        If ``max_attempts`` is exceeded the task moves to FAILED;
        otherwise it is reset to PENDING for retry.
        """
        for task in dag:
            if task.task_id == task_id:
                task.attempts += 1
                task.error = error
                if task.attempts >= task.max_attempts:
                    task.status = TaskStatus.FAILED
                else:
                    task.status = TaskStatus.PENDING
                return


# ---------------------------------------------------------------------------
# Iteration result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IterationResult:
    """Snapshot of one complete research iteration."""

    iteration: int
    dag: list[TaskNode]
    metrics: PerformanceMetrics | None
    passed: bool
    issues: list[str]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Central coordinator for the iterative strategy development loop.

    Drives repeated plan → execute → audit cycles, recording outcomes
    in shared memory so that each new iteration can learn from the last.

    Parameters
    ----------
    max_iterations:
        Hard cap on the number of research iterations.
    """

    def __init__(self, max_iterations: int = 10) -> None:
        self.memory = MemoryAgent()
        self.audit = AuditAgent()
        self.planner = DAGPlanner(self.memory, self.audit)
        self._iteration: int = 0
        self._max_iterations: int = max_iterations
        self._results: list[IterationResult] = []

    # -- iteration lifecycle ------------------------------------------------

    def plan_next_iteration(self) -> list[TaskNode]:
        """Advance the iteration counter and return the next DAG."""
        self._iteration += 1
        if self._iteration == 1:
            return self.planner.create_initial_dag()
        return self.planner.create_iteration_dag(self._iteration)

    def record_iteration(
        self,
        metrics: PerformanceMetrics | None,
        strategy_name: str,
        parameters: dict[str, float],
    ) -> None:
        """Diagnose *metrics*, record in memory, and store the result."""
        if metrics is not None:
            passed, issues = self.audit.diagnose(metrics)
            metrics_dict: dict[str, float] = {
                "sharpe": metrics.sharpe_ratio,
                "cagr": metrics.cagr,
                "max_dd": metrics.max_drawdown,
                "sortino": metrics.sortino_ratio,
                "calmar": metrics.calmar_ratio,
                "total_return": metrics.total_return,
            }
        else:
            passed = False
            issues = ["No metrics produced"]
            metrics_dict = {}

        failure_reason = "; ".join(issues) if issues else ""

        entry = MemoryEntry(
            iteration=self._iteration,
            strategy_name=strategy_name,
            parameters=parameters,
            metrics=metrics_dict,
            success=passed,
            failure_reason=failure_reason,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        self.memory.record(entry)

        dag = self.planner.create_iteration_dag(self._iteration)
        result = IterationResult(
            iteration=self._iteration,
            dag=dag,
            metrics=metrics,
            passed=passed,
            issues=issues,
        )
        self._results.append(result)

    # -- control flow -------------------------------------------------------

    def should_continue(self) -> bool:
        """Return ``False`` if a passing iteration exists or budget is spent."""
        if any(r.passed for r in self._results):
            return False
        if self._iteration >= self._max_iterations:
            return False
        return True

    # -- queries ------------------------------------------------------------

    def get_best_result(self) -> IterationResult | None:
        """Return the iteration whose Sharpe ratio is highest."""
        candidates = [r for r in self._results if r.metrics is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics.sharpe_ratio)  # type: ignore[union-attr]

    def summary(self) -> dict[str, object]:
        """Return a comprehensive summary of the entire run."""
        best = self.get_best_result()
        return {
            "iterations_completed": self._iteration,
            "max_iterations": self._max_iterations,
            "passed": any(r.passed for r in self._results),
            "total_results": len(self._results),
            "memory": self.memory.summary(),
            "best_iteration": best.iteration if best is not None else None,
            "best_metrics": (
                {
                    "sharpe": best.metrics.sharpe_ratio,
                    "cagr": best.metrics.cagr,
                    "max_dd": best.metrics.max_drawdown,
                }
                if best is not None and best.metrics is not None
                else None
            ),
        }
