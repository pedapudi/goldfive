"""Core dataclasses and enums for goldfive.

Pinned by ``docs/design/PROTOCOLS.md`` (v0.1). Types in this module are
pure data — mutation of live state happens only through a ``Steerer``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


# Terminal statuses — a task in any of these cannot transition further
# and must not be re-invoked. This set is the **single source of truth**
# used by the steerer (state-transition guards), the tool-dispatch layer
# (terminal-task rejection), and the ADK adapter (invoke-loop early
# break). Do not duplicate this set; import from here. If ``TaskStatus``
# gains a new terminal member, add it here and every consumer sees it.
# See ``docs/design/TASK-LIFECYCLE.md`` §7.1 for the rationale.
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class DriftKind(StrEnum):
    TOOL_ERROR = "tool_error"
    AGENT_REFUSAL = "agent_refusal"
    NEW_WORK_DISCOVERED = "new_work_discovered"
    PLAN_DIVERGENCE = "plan_divergence"
    USER_STEER = "user_steer"
    USER_CANCEL = "user_cancel"
    USER_PAUSE = "user_pause"
    TASK_FAILED_RECOVERABLE = "task_failed_recoverable"
    TASK_FAILED_FATAL = "task_failed_fatal"
    CONTEXT_PRESSURE = "context_pressure"
    BLOCKED = "blocked"
    WRONG_AGENT = "wrong_agent"
    AGENT_TRANSFER = "agent_transfer"
    MODEL_REFUSAL = "model_refusal"
    STOPPED_EARLY = "stopped_early"
    TOO_MANY_STEPS = "too_many_steps"
    GOAL_UNREACHABLE = "goal_unreachable"
    TASK_TIMEOUT = "task_timeout"
    REPEATED_FAILURE = "repeated_failure"
    UNEXPECTED_OUTPUT = "unexpected_output"
    SCHEMA_VIOLATION = "schema_violation"
    HALLUCINATION_SUSPECTED = "hallucination_suspected"
    SAFETY_CONCERN = "safety_concern"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    CUSTOM = "custom"
    LOOPING_TOOL_CALL = "looping_tool_call"
    LOOPING_REASONING = "looping_reasoning"
    CONFUSION = "confusion"
    OFF_TOPIC = "off_topic"
    INTENT_DIVERGENCE = "intent_divergence"


class DriftSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_RANK: dict[str, int] = {
    DriftSeverity.INFO.value: 0,
    DriftSeverity.WARNING.value: 1,
    DriftSeverity.CRITICAL.value: 2,
}


def severity_rank(sev: DriftSeverity | str) -> int:
    """Ordinal rank for comparing ``DriftSeverity`` values."""
    v = sev.value if isinstance(sev, DriftSeverity) else str(sev)
    return _SEVERITY_RANK.get(v, -1)


@dataclasses.dataclass
class Task:
    id: str
    title: str
    description: str = ""
    assignee_agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    predicted_start_ms: int = 0
    predicted_duration_ms: int = 0
    bound_span_id: str = ""


@dataclasses.dataclass
class TaskEdge:
    from_task_id: str
    to_task_id: str


@dataclasses.dataclass
class Plan:
    id: str
    run_id: str
    goal_ids: list[str]
    tasks: list[Task]
    edges: list[TaskEdge]
    summary: str = ""
    revision_reason: str = ""
    revision_kind: str = ""  # DriftKind value (str) or ""
    revision_severity: str = ""  # DriftSeverity value (str) or ""
    revision_index: int = 0

    def validate(self, for_revision: bool = False, *, prior: Plan | None = None) -> None:
        """Structurally validate this plan. Raise ``ValueError`` on failure.

        Checks, in order:

        1. Every task has a non-empty ``id``.
        2. Task ids are unique within ``tasks``.
        3. Every edge's ``from_task_id`` and ``to_task_id`` reference a
           task that exists in ``tasks``.
        4. The task graph is acyclic (every task must be placeable by
           ``topological_stages``; any leftover is a cycle member).
        5. When ``for_revision`` is ``False`` (the default — the
           creation path), every task's ``status`` must be
           ``TaskStatus.PENDING``. Revised plans legitimately carry
           COMPLETED / FAILED / CANCELLED tasks preserved from the prior
           plan, so this check is skipped when ``for_revision`` is
           ``True``.
        6. When ``for_revision`` is ``True`` and a ``prior`` plan is
           supplied, enforce the cross-revision preservation contract
           from ``docs/design/PLAN-LIFECYCLE.md`` §3.1 and §3.2:

           - **Terminal task preservation (§3.1).** Every task in
             ``prior.tasks`` whose status is terminal must appear in
             ``self.tasks`` with the same id AND the same terminal
             status — no regression from COMPLETED back to PENDING is
             allowed, and dropping a terminal task is forbidden.
           - **Terminal→terminal edge preservation (§3.2).** Every
             edge in ``prior.edges`` where both endpoints were terminal
             in ``prior`` must appear in ``self.edges``. Historical
             topology between frozen tasks is frozen.

        This is a pure-data validator: it does not mutate the plan. It
        is intended to be called at plan creation (``LLMPlanner.generate``)
        and at plan revision (``LLMPlanner.refine`` /
        ``DefaultSteerer._apply_revision``) so malformed plans are
        rejected before they are installed on a ``Session``.
        """
        # 1. & 2. ids present and unique.
        seen: set[str] = set()
        for t in self.tasks:
            if not t.id:
                raise ValueError("plan contains a task with an empty id")
            if t.id in seen:
                raise ValueError(f"duplicate task id in plan: {t.id!r}")
            seen.add(t.id)

        # 3. edges reference known tasks.
        for e in self.edges:
            if e.from_task_id not in seen:
                raise ValueError(
                    f"edge references unknown task id (from_task_id={e.from_task_id!r})"
                )
            if e.to_task_id not in seen:
                raise ValueError(f"edge references unknown task id (to_task_id={e.to_task_id!r})")

        # 4. no cycles. topological_stages places every non-cycle task;
        # any task left over is part of a cycle. We compute placement
        # locally (rather than calling topological_stages) so we can
        # avoid its edge-tolerance behaviour — validation wants a clean
        # signal.
        indeg: dict[str, int] = {tid: 0 for tid in seen}
        children: dict[str, list[str]] = {tid: [] for tid in seen}
        for e in self.edges:
            children[e.from_task_id].append(e.to_task_id)
            indeg[e.to_task_id] += 1
        ready = [tid for tid, d in indeg.items() if d == 0]
        placed: set[str] = set()
        while ready:
            tid = ready.pop()
            if tid in placed:
                continue
            placed.add(tid)
            for child in children[tid]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
        unplaced = seen - placed
        if unplaced:
            raise ValueError(f"plan contains a cycle among tasks: {sorted(unplaced)!r}")

        # 5. creation-time: all tasks must be PENDING.
        if not for_revision:
            for t in self.tasks:
                if t.status is not TaskStatus.PENDING:
                    raise ValueError(
                        f"task {t.id!r} has non-PENDING status {t.status.value!r} at plan creation"
                    )

        # 6. revision-time with a prior plan: enforce terminal-task and
        # terminal->terminal-edge preservation (PLAN-LIFECYCLE.md §3.1,
        # §3.2). Skipped when ``prior`` is None — callers that do not
        # supply the outgoing plan get the legacy structural checks only.
        if for_revision and prior is not None:
            new_by_id: dict[str, Task] = {t.id: t for t in self.tasks}
            prior_terminal_ids: set[str] = set()
            for t in prior.tasks:
                if t.status not in TERMINAL_TASK_STATUSES:
                    continue
                prior_terminal_ids.add(t.id)
                new_t = new_by_id.get(t.id)
                if new_t is None:
                    raise ValueError(f"terminal task {t.id!r} missing in revision")
                if new_t.status is not t.status:
                    raise ValueError(f"terminal task {t.id!r} regressed to {new_t.status.value!r}")
            # Every terminal->terminal edge in the outgoing plan must
            # appear verbatim in the revision.
            new_edges: set[tuple[str, str]] = {(e.from_task_id, e.to_task_id) for e in self.edges}
            for e in prior.edges:
                if (
                    e.from_task_id in prior_terminal_ids
                    and e.to_task_id in prior_terminal_ids
                    and (e.from_task_id, e.to_task_id) not in new_edges
                ):
                    raise ValueError(
                        "terminal->terminal edge "
                        f"{e.from_task_id!r} -> {e.to_task_id!r} missing in revision"
                    )

    def topological_stages(self) -> list[list[Task]]:
        """Return tasks grouped into topological stages (Kahn's algorithm).

        Each stage contains tasks whose dependencies are all satisfied by
        tasks in earlier stages. Tasks with no deps live in stage 0.
        Cycles or edges referencing unknown task ids are tolerated — any
        task that can never be placed is appended to a final trailing
        stage so the full set is always returned.
        """
        tasks_by_id = {t.id: t for t in self.tasks if t.id}
        indeg: dict[str, int] = {tid: 0 for tid in tasks_by_id}
        children: dict[str, list[str]] = {tid: [] for tid in tasks_by_id}
        for e in self.edges:
            if e.from_task_id in tasks_by_id and e.to_task_id in tasks_by_id:
                children[e.from_task_id].append(e.to_task_id)
                indeg[e.to_task_id] += 1

        stages: list[list[Task]] = []
        ready = [tid for tid, d in indeg.items() if d == 0]
        placed: set[str] = set()
        while ready:
            stage_ids = sorted(ready)
            stages.append([tasks_by_id[tid] for tid in stage_ids])
            placed.update(stage_ids)
            next_ready: list[str] = []
            for tid in stage_ids:
                for child in children[tid]:
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        leftover = [t for tid, t in tasks_by_id.items() if tid not in placed]
        if leftover:
            stages.append(leftover)
        return stages


@dataclasses.dataclass
class Goal:
    id: str
    summary: str
    success_predicate: Callable[[Session], bool] | None = None
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None  # original event that triggered detection


@dataclasses.dataclass
class Session:
    """Live state for one Runner.run() invocation.

    ``conversation_id`` links this turn to the owning :class:`~goldfive.conversation.Conversation`
    and is stable across successive turns on the same Runner. It
    defaults to ``""`` for legacy callers that build Sessions directly
    without going through a Conversation.
    """

    run_id: str
    conversation_id: str = ""
    goals: list[Goal] = dataclasses.field(default_factory=list)
    plan: Plan | None = None
    current_task_id: str = ""
    completed_results: dict[str, str] = dataclasses.field(default_factory=dict)
    # task_id -> progress fraction 0-1
    task_progress: dict[str, float] = dataclasses.field(default_factory=dict)
    # task_id -> last agent note
    agent_notes: dict[str, str] = dataclasses.field(default_factory=dict)
    divergence_flag: bool = False
    history: list[Any] = dataclasses.field(default_factory=list)
    started_at_ms: int = 0
    # Waiters for outstanding human-in-the-loop approvals. Keyed by
    # ``target_id``: task_id for Flow A (report_awaiting_approval) and the
    # ADK function_call_id for Flow B (ADK require_confirmation). The event
    # is set by the control dispatcher when APPROVE / REJECT arrives.
    pending_approvals: dict[str, asyncio.Event] = dataclasses.field(default_factory=dict)
    # Per-approval metadata. Populated when the waiter is registered; the
    # dispatcher adds ``decision`` ("approve" | "reject") and optional
    # ``detail`` before setting the event.
    pending_approvals_meta: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # Recent reasoning-content blocks emitted by the adapter's
    # ``emit_reasoning`` hook. Bounded to the last ``reasoning_history_max``
    # entries so long runs do not accumulate chain-of-thought forever.
    # Consumed by the reasoning-drift detectors (see ``goldfive.drift_reasoning``).
    reasoning_history: list[str] = dataclasses.field(default_factory=list)
    reasoning_history_max: int = 20
    # Per-(drift_kind_value, task_id) consecutive refine-failure counter.
    # Incremented each time ``planner.refine`` raises or returns ``None``
    # for the given (kind, task) tuple; reset on a successful refine.
    # Consumed by :class:`~goldfive.steerer.DefaultSteerer` to back off
    # and mark the task FAILED after N consecutive failures, preventing
    # the same drift from looping until ``max_plan_reinvocations`` trips.
    refine_failure_counts: dict[tuple[str, str], int] = dataclasses.field(
        default_factory=dict
    )
    # monotonic event sequence counter for sinks
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        s = self._next_sequence
        self._next_sequence = s + 1
        return s
