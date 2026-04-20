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
    revision_kind: str = ""           # DriftKind value (str) or ""
    revision_severity: str = ""       # DriftSeverity value (str) or ""
    revision_index: int = 0

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
    raw: Any = None   # original event that triggered detection


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
    pending_approvals: dict[str, asyncio.Event] = dataclasses.field(
        default_factory=dict
    )
    # Per-approval metadata. Populated when the waiter is registered; the
    # dispatcher adds ``decision`` ("approve" | "reject") and optional
    # ``detail`` before setting the event.
    pending_approvals_meta: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    # monotonic event sequence counter for sinks
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        s = self._next_sequence
        self._next_sequence = s + 1
        return s
