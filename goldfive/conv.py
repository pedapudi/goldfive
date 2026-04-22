"""Round-trip converters between goldfive dataclasses and protobuf messages.

The generated proto stubs live under ``goldfive.pb.goldfive.v1`` (issue #3).
Until those stubs are available this module defers imports so that importing
``goldfive.conv`` does not hard-require the ``proto`` optional-dependency
group. Callers that invoke the ``to_pb_*`` / ``from_pb_*`` helpers will
receive a clear ``ModuleNotFoundError`` at call time if the stubs are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldfive.control import AckResult, ControlAck, ControlKind, ControlMessage
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
    from goldfive.pb.goldfive.v1 import control_pb2, types_pb2  # noqa: F401


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
    # Overlay-model (goldfive#141): PlanReconciler marks PENDING tasks
    # the tree legitimately skipped as NOT_NEEDED so sinks can tell
    # "chose not to run" apart from a user/system-initiated CANCELLED.
    TaskStatus.NOT_NEEDED: "TASK_STATUS_NOT_NEEDED",
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
_PB_TO_DRIFT_SEVERITY: dict[str, DriftSeverity] = {v: k for k, v in _DRIFT_SEVERITY_TO_PB.items()}


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
        member = name[len("DRIFT_KIND_") :]
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


def _plan_revision_kind_to_pb(value: str, pb: Any) -> int:
    """Convert a dataclass ``Plan.revision_kind`` (a ``DriftKind`` string
    value or ``""``) into the matching proto enum int. Empty string →
    ``DRIFT_KIND_UNSPECIFIED``; unknown value → ``DRIFT_KIND_CUSTOM``.
    """
    if not value:
        return getattr(pb, "DRIFT_KIND_UNSPECIFIED", 0)
    try:
        kind = DriftKind(value)
    except ValueError:
        return getattr(pb, "DRIFT_KIND_CUSTOM", 0)
    return _drift_kind_to_pb(kind, pb)


def _plan_revision_severity_to_pb(value: str, pb: Any) -> int:
    if not value:
        return getattr(pb, "DRIFT_SEVERITY_UNSPECIFIED", 0)
    try:
        severity = DriftSeverity(value)
    except ValueError:
        return getattr(pb, "DRIFT_SEVERITY_UNSPECIFIED", 0)
    return _drift_severity_to_pb(severity, pb)


def _plan_revision_kind_from_pb(value: int, pb: Any) -> str:
    try:
        name = pb.DriftKind.Name(value)
    except (ValueError, AttributeError):
        return ""
    if name == "DRIFT_KIND_UNSPECIFIED":
        return ""
    return _drift_kind_from_pb(value, pb).value


def _plan_revision_severity_from_pb(value: int, pb: Any) -> str:
    try:
        name = pb.DriftSeverity.Name(value)
    except (ValueError, AttributeError):
        return ""
    if name == "DRIFT_SEVERITY_UNSPECIFIED":
        return ""
    return _drift_severity_from_pb(value, pb).value


def to_pb_plan(plan: Plan) -> Any:
    pb = _pb_module()
    msg = pb.Plan(
        id=plan.id,
        run_id=plan.run_id,
        summary=plan.summary,
        revision_reason=plan.revision_reason,
        revision_kind=_plan_revision_kind_to_pb(plan.revision_kind, pb),
        revision_severity=_plan_revision_severity_to_pb(plan.revision_severity, pb),
        revision_index=plan.revision_index,
    )
    msg.goal_ids.extend(plan.goal_ids)
    for t in plan.tasks:
        msg.tasks.append(to_pb_task(t))
    for e in plan.edges:
        msg.edges.append(to_pb_task_edge(e))
    return msg


def from_pb_plan(msg: Any) -> Plan:
    pb = _pb_module()
    return Plan(
        id=msg.id,
        run_id=msg.run_id,
        goal_ids=list(msg.goal_ids),
        tasks=[from_pb_task(t) for t in msg.tasks],
        edges=[from_pb_task_edge(e) for e in msg.edges],
        summary=msg.summary,
        revision_reason=msg.revision_reason,
        revision_kind=_plan_revision_kind_from_pb(msg.revision_kind, pb),
        revision_severity=_plan_revision_severity_from_pb(msg.revision_severity, pb),
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


def _events_pb_module() -> Any:
    """Import ``goldfive.pb.goldfive.v1.events_pb2`` lazily.

    ``DriftEvent`` is expressed on the wire as the ``DriftDetected``
    payload of an ``Event`` envelope (see ``proto/goldfive/v1/events.proto``),
    so the drift-event round-trip lives in the events module, not the
    types module.
    """
    try:
        from goldfive.pb.goldfive.v1 import events_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "goldfive protobuf stubs not available; generate them via "
            "`make proto` (requires the `proto` optional-dependency group) "
            "or install the package with the `proto` extra. See issue #3."
        ) from exc
    return events_pb2


def to_pb_drift_event(evt: DriftEvent) -> Any:
    """Convert a :class:`DriftEvent` to the wire ``DriftDetected`` proto."""
    types_pb = _pb_module()
    events_pb = _events_pb_module()
    return events_pb.DriftDetected(
        kind=_drift_kind_to_pb(evt.kind, types_pb),
        severity=_drift_severity_to_pb(evt.severity, types_pb),
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


# ---------------------------------------------------------------------------
# Control events / acks
# ---------------------------------------------------------------------------
#
# The wire format lives in proto/goldfive/v1/control.proto — goldfive is
# the single source of truth for the control plane, and harmonograf (or
# any other bridge) imports those messages rather than mirroring them.
# See docs/design/CONTROL.md for the motivation.


def _control_pb_module() -> Any:
    """Import ``goldfive.pb.goldfive.v1.control_pb2`` lazily."""
    try:
        from goldfive.pb.goldfive.v1 import control_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "goldfive protobuf stubs not available; generate them via "
            "`make proto` (requires the `proto` optional-dependency group) "
            "or install the package with the `proto` extra. See issue #3."
        ) from exc
    return control_pb2


_CONTROL_KIND_TO_PB: dict[ControlKind, str] = {
    ControlKind.PAUSE: "CONTROL_KIND_PAUSE",
    ControlKind.RESUME: "CONTROL_KIND_RESUME",
    ControlKind.CANCEL: "CONTROL_KIND_CANCEL",
    ControlKind.REWIND_TO: "CONTROL_KIND_REWIND_TO",
    ControlKind.STEER: "CONTROL_KIND_STEER",
    ControlKind.APPROVE: "CONTROL_KIND_APPROVE",
    ControlKind.REJECT: "CONTROL_KIND_REJECT",
    ControlKind.STATUS_QUERY: "CONTROL_KIND_STATUS_QUERY",
    ControlKind.INTERCEPT_TRANSFER: "CONTROL_KIND_INTERCEPT_TRANSFER",
    ControlKind.INJECT_MESSAGE: "CONTROL_KIND_INJECT_MESSAGE",
}
_PB_TO_CONTROL_KIND: dict[str, ControlKind] = {v: k for k, v in _CONTROL_KIND_TO_PB.items()}


def _control_kind_to_pb(kind: ControlKind, pb: Any) -> int:
    name = _CONTROL_KIND_TO_PB[kind]
    return getattr(pb, name, 0)


def _control_kind_from_pb(value: int, pb: Any) -> ControlKind:
    try:
        name = pb.ControlKind.Name(value)
    except (ValueError, AttributeError):
        return ControlKind.PAUSE
    # UNSPECIFIED has no dataclass equivalent; the dataclass requires a
    # ControlKind so we fall back to PAUSE rather than crashing. Callers
    # that care check the proto ``kind`` field directly.
    if name == "CONTROL_KIND_UNSPECIFIED":
        return ControlKind.PAUSE
    return _PB_TO_CONTROL_KIND.get(name, ControlKind.PAUSE)


_ACK_RESULT_TO_PB: dict[AckResult, str] = {
    AckResult.SUCCESS: "CONTROL_ACK_RESULT_SUCCESS",
    AckResult.FAILURE: "CONTROL_ACK_RESULT_FAILURE",
    AckResult.UNSUPPORTED: "CONTROL_ACK_RESULT_UNSUPPORTED",
}
_PB_TO_ACK_RESULT: dict[str, AckResult] = {v: k for k, v in _ACK_RESULT_TO_PB.items()}


def _ack_result_to_pb(result: AckResult, pb: Any) -> int:
    name = _ACK_RESULT_TO_PB[result]
    return getattr(pb, name, 0)


def _ack_result_from_pb(value: int, pb: Any) -> AckResult:
    try:
        name = pb.ControlAckResult.Name(value)
    except (ValueError, AttributeError):
        return AckResult.SUCCESS
    if name == "CONTROL_ACK_RESULT_UNSPECIFIED":
        return AckResult.SUCCESS
    return _PB_TO_ACK_RESULT.get(name, AckResult.SUCCESS)


# Mapping from ControlKind to (oneof-field-name, payload-builder). Only
# kinds with structured payloads appear here; the others leave the
# oneof unset.
def _build_steer(payload: dict[str, Any], pb: Any) -> Any:
    return pb.SteerPayload(
        note=str(payload.get("note", "")),
        suggested_action=str(payload.get("suggested_action", "")),
        author=str(payload.get("author", "")),
        annotation_id=str(payload.get("annotation_id", "")),
    )


def _build_rewind(payload: dict[str, Any], pb: Any) -> Any:
    return pb.RewindPayload(task_id=str(payload.get("task_id", "")))


def _build_approve(payload: dict[str, Any], pb: Any) -> Any:
    return pb.ApprovePayload(
        target_id=str(payload.get("target_id", "")),
        detail=str(payload.get("detail", "")),
    )


def _build_reject(payload: dict[str, Any], pb: Any) -> Any:
    return pb.RejectPayload(
        target_id=str(payload.get("target_id", "")),
        detail=str(payload.get("detail", "")),
    )


def _build_inject_message(payload: dict[str, Any], pb: Any) -> Any:
    return pb.InjectMessagePayload(
        role=str(payload.get("role", "")),
        text=str(payload.get("text", "")),
    )


_PAYLOAD_BUILDERS: dict[ControlKind, tuple[str, Any]] = {
    ControlKind.STEER: ("steer", _build_steer),
    ControlKind.REWIND_TO: ("rewind", _build_rewind),
    ControlKind.APPROVE: ("approve", _build_approve),
    ControlKind.REJECT: ("reject", _build_reject),
    ControlKind.INJECT_MESSAGE: ("inject_message", _build_inject_message),
}


def to_pb_control_event(msg: ControlMessage) -> Any:
    """Convert a :class:`ControlMessage` to a ``ControlEvent`` proto.

    The dataclass' opaque ``payload: dict`` is translated to the typed
    oneof branch matching ``msg.kind``. Kinds without a structured
    payload (``PAUSE`` / ``RESUME`` / ``CANCEL`` / ``STATUS_QUERY`` /
    ``INTERCEPT_TRANSFER``) leave the oneof unset.
    """
    pb = _control_pb_module()
    kwargs: dict[str, Any] = {
        "id": msg.id,
        "kind": _control_kind_to_pb(msg.kind, pb),
    }
    builder = _PAYLOAD_BUILDERS.get(msg.kind)
    if builder is not None:
        field_name, build = builder
        kwargs[field_name] = build(msg.payload, pb)
    event = pb.ControlEvent(**kwargs)
    if msg.issued_at_ms:
        event.issued_at.FromMilliseconds(msg.issued_at_ms)
    return event


def from_pb_control_event(pb_msg: Any) -> ControlMessage:
    """Convert a ``ControlEvent`` proto back to a :class:`ControlMessage`."""
    pb = _control_pb_module()
    kind = _control_kind_from_pb(pb_msg.kind, pb)
    payload: dict[str, Any] = {}
    which = pb_msg.WhichOneof("payload")
    if which == "steer":
        payload = {
            "note": pb_msg.steer.note,
            "suggested_action": pb_msg.steer.suggested_action,
            "author": pb_msg.steer.author,
            "annotation_id": pb_msg.steer.annotation_id,
        }
    elif which == "rewind":
        payload = {"task_id": pb_msg.rewind.task_id}
    elif which == "approve":
        payload = {
            "target_id": pb_msg.approve.target_id,
            "detail": pb_msg.approve.detail,
        }
    elif which == "reject":
        payload = {
            "target_id": pb_msg.reject.target_id,
            "detail": pb_msg.reject.detail,
        }
    elif which == "inject_message":
        payload = {
            "role": pb_msg.inject_message.role,
            "text": pb_msg.inject_message.text,
        }
    issued_at_ms = 0
    if pb_msg.HasField("issued_at"):
        issued_at_ms = pb_msg.issued_at.ToMilliseconds()
    return ControlMessage(
        kind=kind,
        id=pb_msg.id,
        payload=payload,
        issued_at_ms=issued_at_ms,
    )


def to_pb_control_ack(ack: ControlAck) -> Any:
    """Convert a :class:`ControlAck` to a ``ControlAck`` proto."""
    pb = _control_pb_module()
    # StrEnum round-trip: handles both AckResult instances and raw strings
    # ("SUCCESS" / "FAILURE" / "UNSUPPORTED") from loosely-typed callers.
    result = AckResult(ack.result)
    pb_ack = pb.ControlAck(
        control_id=ack.control_id,
        result=_ack_result_to_pb(result, pb),
        detail=ack.detail,
    )
    if ack.acked_at_ms:
        pb_ack.acked_at.FromMilliseconds(ack.acked_at_ms)
    return pb_ack


def from_pb_control_ack(pb_msg: Any) -> ControlAck:
    """Convert a ``ControlAck`` proto back to a :class:`ControlAck`."""
    pb = _control_pb_module()
    acked_at_ms = 0
    if pb_msg.HasField("acked_at"):
        acked_at_ms = pb_msg.acked_at.ToMilliseconds()
    return ControlAck(
        control_id=pb_msg.control_id,
        result=_ack_result_from_pb(pb_msg.result, pb),
        detail=pb_msg.detail,
        acked_at_ms=acked_at_ms,
    )
