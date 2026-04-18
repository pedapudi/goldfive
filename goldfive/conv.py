"""Round-trip converters between goldfive dataclasses and protobuf messages.

The generated proto stubs live under ``goldfive.pb.goldfive.v1`` (issue #3).
Until those stubs are available this module defers imports so that importing
``goldfive.conv`` does not hard-require the ``proto`` optional-dependency
group. Callers that invoke the ``to_pb_*`` / ``from_pb_*`` helpers will
receive a clear ``ModuleNotFoundError`` at call time if the stubs are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)

if TYPE_CHECKING:
    # These names only exist once issue #3 generates the pb stubs. Guarding
    # them under TYPE_CHECKING keeps this module importable regardless.
    from goldfive.pb.goldfive.v1 import types_pb2  # noqa: F401


def _pb_module() -> Any:
    """Import and return ``goldfive.pb.goldfive.v1.types_pb2`` lazily."""
    try:
        from goldfive.pb.goldfive.v1 import types_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by tests
        raise ModuleNotFoundError(
            "goldfive protobuf stubs not available; generate them via "
            "`make proto` (requires the `proto` optional-dependency group) "
            "or install the package with the `proto` extra. See issue #3."
        ) from exc
    return types_pb2


# ---------------------------------------------------------------------------
# Enum helpers
# ---------------------------------------------------------------------------


_TASK_STATUS_TO_PB: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "TASK_STATUS_PENDING",
    TaskStatus.RUNNING: "TASK_STATUS_RUNNING",
    TaskStatus.COMPLETED: "TASK_STATUS_COMPLETED",
    TaskStatus.FAILED: "TASK_STATUS_FAILED",
    TaskStatus.CANCELLED: "TASK_STATUS_CANCELLED",
    TaskStatus.BLOCKED: "TASK_STATUS_BLOCKED",
}
_PB_TO_TASK_STATUS: dict[str, TaskStatus] = {v: k for k, v in _TASK_STATUS_TO_PB.items()}


def _task_status_to_pb(status: TaskStatus, pb: Any) -> int:
    name = _TASK_STATUS_TO_PB[status]
    return getattr(pb, name, 0)


def _task_status_from_pb(value: int, pb: Any) -> TaskStatus:
    # Resolve enum name from value and map back; unknown / unspecified falls
    # back to PENDING so callers never crash on forward-compatible inputs.
    try:
        name = pb.TaskStatus.Name(value)
    except (ValueError, AttributeError):
        return TaskStatus.PENDING
    return _PB_TO_TASK_STATUS.get(name, TaskStatus.PENDING)


_DRIFT_SEVERITY_TO_PB: dict[DriftSeverity, str] = {
    DriftSeverity.INFO: "DRIFT_SEVERITY_INFO",
    DriftSeverity.WARNING: "DRIFT_SEVERITY_WARNING",
    DriftSeverity.CRITICAL: "DRIFT_SEVERITY_CRITICAL",
}
_PB_TO_DRIFT_SEVERITY: dict[str, DriftSeverity] = {
    v: k for k, v in _DRIFT_SEVERITY_TO_PB.items()
}


def _drift_severity_to_pb(severity: DriftSeverity, pb: Any) -> int:
    name = _DRIFT_SEVERITY_TO_PB[severity]
    return getattr(pb, name, 0)


def _drift_severity_from_pb(value: int, pb: Any) -> DriftSeverity:
    try:
        name = pb.DriftSeverity.Name(value)
    except (ValueError, AttributeError):
        return DriftSeverity.INFO
    return _PB_TO_DRIFT_SEVERITY.get(name, DriftSeverity.INFO)


def _drift_kind_to_pb(kind: DriftKind, pb: Any) -> int:
    # Proto convention: DRIFT_KIND_<UPPER_SNAKE> values mirror the enum
    # member name (not the wire value). Fall back to CUSTOM when absent.
    name = f"DRIFT_KIND_{kind.name}"
    return getattr(pb, name, getattr(pb, "DRIFT_KIND_CUSTOM", 0))


def _drift_kind_from_pb(value: int, pb: Any) -> DriftKind:
    try:
        name = pb.DriftKind.Name(value)
    except (ValueError, AttributeError):
        return DriftKind.CUSTOM
    # Strip the "DRIFT_KIND_" prefix and look up by enum member name.
    if name.startswith("DRIFT_KIND_"):
        member = name[len("DRIFT_KIND_"):]
        try:
            return DriftKind[member]
        except KeyError:
            return DriftKind.CUSTOM
    return DriftKind.CUSTOM


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def to_pb_task(task: Task) -> Any:
    pb = _pb_module()
    msg = pb.Task(
        id=task.id,
        title=task.title,
        description=task.description,
        assignee_agent_id=task.assignee_agent_id,
        status=_task_status_to_pb(task.status, pb),
        predicted_start_ms=task.predicted_start_ms,
        predicted_duration_ms=task.predicted_duration_ms,
        bound_span_id=task.bound_span_id,
    )
    return msg


def from_pb_task(msg: Any) -> Task:
    pb = _pb_module()
    return Task(
        id=msg.id,
        title=msg.title,
        description=msg.description,
        assignee_agent_id=msg.assignee_agent_id,
        status=_task_status_from_pb(msg.status, pb),
        predicted_start_ms=msg.predicted_start_ms,
        predicted_duration_ms=msg.predicted_duration_ms,
        bound_span_id=msg.bound_span_id,
    )


# ---------------------------------------------------------------------------
# TaskEdge
# ---------------------------------------------------------------------------


def to_pb_task_edge(edge: TaskEdge) -> Any:
    pb = _pb_module()
    return pb.TaskEdge(from_task_id=edge.from_task_id, to_task_id=edge.to_task_id)


def from_pb_task_edge(msg: Any) -> TaskEdge:
    return TaskEdge(from_task_id=msg.from_task_id, to_task_id=msg.to_task_id)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def to_pb_plan(plan: Plan) -> Any:
    pb = _pb_module()
    msg = pb.Plan(
        id=plan.id,
        run_id=plan.run_id,
        summary=plan.summary,
        revision_reason=plan.revision_reason,
        revision_kind=plan.revision_kind,
        revision_severity=plan.revision_severity,
        revision_index=plan.revision_index,
    )
    msg.goal_ids.extend(plan.goal_ids)
    for t in plan.tasks:
        msg.tasks.append(to_pb_task(t))
    for e in plan.edges:
        msg.edges.append(to_pb_task_edge(e))
    return msg


def from_pb_plan(msg: Any) -> Plan:
    return Plan(
        id=msg.id,
        run_id=msg.run_id,
        goal_ids=list(msg.goal_ids),
        tasks=[from_pb_task(t) for t in msg.tasks],
        edges=[from_pb_task_edge(e) for e in msg.edges],
        summary=msg.summary,
        revision_reason=msg.revision_reason,
        revision_kind=msg.revision_kind,
        revision_severity=msg.revision_severity,
        revision_index=msg.revision_index,
    )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


def to_pb_goal(goal: Goal) -> Any:
    pb = _pb_module()
    msg = pb.Goal(id=goal.id, summary=goal.summary)
    for k, v in goal.metadata.items():
        msg.metadata[k] = v
    return msg


def from_pb_goal(msg: Any) -> Goal:
    # success_predicate is a live Python callable and never survives the
    # wire; the round-trip for it is not preserved by design.
    return Goal(
        id=msg.id,
        summary=msg.summary,
        success_predicate=None,
        metadata=dict(msg.metadata),
    )


# ---------------------------------------------------------------------------
# DriftEvent
# ---------------------------------------------------------------------------


def to_pb_drift_event(evt: DriftEvent) -> Any:
    pb = _pb_module()
    return pb.DriftEvent(
        kind=_drift_kind_to_pb(evt.kind, pb),
        severity=_drift_severity_to_pb(evt.severity, pb),
        detail=evt.detail,
        current_task_id=evt.current_task_id,
        current_agent_id=evt.current_agent_id,
    )


def from_pb_drift_event(msg: Any) -> DriftEvent:
    pb = _pb_module()
    return DriftEvent(
        kind=_drift_kind_from_pb(msg.kind, pb),
        severity=_drift_severity_from_pb(msg.severity, pb),
        detail=msg.detail,
        current_task_id=msg.current_task_id,
        current_agent_id=msg.current_agent_id,
        raw=None,
    )
