"""Reporting-tool specs and handlers.

The eight canonical reporting tools — the agent-facing contract for
driving the plan's task state machine and signalling plan mutations.
Each :class:`ReportingToolSpec` pairs a stable tool name with a JSON-schema
parameters block and an async handler. Handlers receive the decoded
arguments, the live :class:`Session`, and the bound :class:`Steerer`, and
route the call into the steerer's transition / drift pipeline.

Adapters materialise these specs into whatever native tool shape their
framework wants (ADK ``FunctionTool``, Claude Agent SDK tool blocks, …).

The eighth tool, ``report_awaiting_approval``, is the task-level half of
the human-in-the-loop approval flow described in
``docs/design/APPROVAL.md``. Its handler blocks the calling tool-call
until the control dispatcher lands an ``APPROVE`` or ``REJECT``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.types import Session

log = logging.getLogger(__name__)


# Framework-agnostic async handler signature.
# Handlers receive the tool-call arguments (already decoded to a dict),
# the live Session, and the Steerer, and return a JSON-serializable dict.
ReportingHandler = Callable[
    [dict[str, Any], "Session", "Steerer"],
    Awaitable[dict[str, Any]],
]


@dataclasses.dataclass
class ReportingToolSpec:
    """Framework-agnostic spec for a reporting tool.

    Adapters translate one of these into whatever tool representation their
    underlying framework expects (e.g., an ADK ``FunctionTool`` or a Claude
    Agent SDK tool definition). The canonical set of tool names is pinned in
    :data:`REPORTING_TOOL_NAMES`.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters
    handler: ReportingHandler


# The eight canonical reporting tool names. These are a stable contract: the
# adapter must surface tools with exactly these names so that the Steerer can
# interpret them uniformly across frameworks. Do not rename.
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
    "report_awaiting_approval",
)


# ---------------------------------------------------------------------------
# Handler shims
# ---------------------------------------------------------------------------


_ACK: dict[str, Any] = {"acknowledged": True}

# Orchestration-state key the adapter stamps at delegation time
# (goldfive#191). Handlers fall back to this value when the model's
# tool call omits ``task_id``. Re-declared here rather than imported
# from :mod:`goldfive.orchestration_state` to avoid a circular import
# — the string is a stable contract shared between the adapter's
# :mod:`._adk_state_protocol`, :mod:`orchestration_state`, and this
# handler module.
_STATE_KEY_CURRENT_TASK_ID = "goldfive.current_task_id"


def _resolve_task_id(args: dict[str, Any], session: Session) -> str:
    """Return the task_id to act on, falling back to session state.

    Order of precedence (goldfive#191):

    1. ``args["task_id"]`` — explicit model-provided id always wins.
    2. ``session.state["goldfive.current_task_id"]`` — the id pinned
       by the adapter's ``before_agent_callback`` when the current
       sub-agent has exactly one PENDING/RUNNING task assigned to
       it. Closes the loop where the LLM's tool call omits the
       arg but the orchestration layer knew the answer.

    Empty string when neither source supplies a value — caller
    should short-circuit with the canonical ``missing_task_id``
    error in that case.
    """
    raw = args.get("task_id")
    if raw is not None:
        task_id = str(raw).strip()
        if task_id:
            return task_id
    state = getattr(session, "state", None)
    if isinstance(state, dict):
        fallback = state.get(_STATE_KEY_CURRENT_TASK_ID, "")
        if isinstance(fallback, str):
            return fallback.strip()
        if fallback is not None:
            return str(fallback).strip()
    return ""


def _missing_task_id_response(tool_name: str) -> dict[str, Any]:
    """Return the canonical ``missing_task_id`` rejection shape.

    Mirrors the shape :mod:`goldfive.adapters._tool_invocation`
    returns so adapters that call the handler directly (legacy paths,
    custom adapters) surface the same structured error the
    ``invoke_tool`` dispatcher would.
    """
    return {
        "acknowledged": False,
        "error": "missing_task_id",
        "tool": tool_name,
        "message": (
            f"Tool {tool_name!r} requires a task_id; call it with the id "
            "of the task you're reporting on, or ensure the adapter "
            "has pinned goldfive.current_task_id on session state."
        ),
    }


def _str(args: dict[str, Any], key: str, default: str = "") -> str:
    v = args.get(key, default)
    if v is None:
        return default
    return str(v)


def _float(args: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = args.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(args: dict[str, Any], key: str, default: bool = True) -> bool:
    v = args.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes"}
    return bool(v) if v is not None else default


def _int(args: dict[str, Any], key: str, default: int = 0) -> int:
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


async def _handle_task_started(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _resolve_task_id(args, session)
    detail = _str(args, "detail")
    if not task_id:
        return _missing_task_id_response("report_task_started")
    await steerer.mark_task_running(task_id, session=session, detail=detail)
    return dict(_ACK)


async def _handle_task_progress(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _resolve_task_id(args, session)
    fraction = _float(args, "fraction")
    detail = _str(args, "detail")
    if not task_id:
        return _missing_task_id_response("report_task_progress")
    await steerer.mark_task_progress(task_id, session=session, fraction=fraction, detail=detail)
    return dict(_ACK)


async def _handle_task_completed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _resolve_task_id(args, session)
    summary = _str(args, "summary")
    artifacts_raw = args.get("artifacts")
    artifacts = (
        {str(k): str(v) for k, v in (artifacts_raw or {}).items()}
        if isinstance(artifacts_raw, dict)
        else {}
    )
    if not task_id:
        return _missing_task_id_response("report_task_completed")
    await steerer.mark_task_completed(
        task_id, session=session, summary=summary, artifacts=artifacts
    )
    return dict(_ACK)


async def _handle_task_failed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _resolve_task_id(args, session)
    reason = _str(args, "reason")
    recoverable = _bool(args, "recoverable", default=True)
    if not task_id:
        return _missing_task_id_response("report_task_failed")
    await steerer.mark_task_failed(
        task_id,
        session=session,
        reason=reason,
        recoverable=recoverable,
    )
    return dict(_ACK)


async def _handle_task_blocked(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id = _resolve_task_id(args, session)
    blocker = _str(args, "blocker")
    needed = _str(args, "needed")
    if not task_id:
        return _missing_task_id_response("report_task_blocked")
    await steerer.mark_task_blocked(task_id, session=session, blocker=blocker, needed=needed)
    return dict(_ACK)


async def _handle_new_work_discovered(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    parent_task_id = _str(args, "parent_task_id")
    title = _str(args, "title")
    description = _str(args, "description")
    assignee = _str(args, "assignee")
    await steerer.report_new_work_discovered(
        session=session,
        parent_task_id=parent_task_id,
        title=title,
        description=description,
        assignee=assignee,
    )
    return dict(_ACK)


async def _handle_plan_divergence(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    note = _str(args, "note")
    suggested_action = _str(args, "suggested_action")
    await steerer.report_plan_divergence(
        session=session,
        note=note,
        suggested_action=suggested_action,
    )
    return dict(_ACK)


async def _handle_awaiting_approval(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    """Block the current task until APPROVE / REJECT arrives on the control channel.

    Transitions the task to ``BLOCKED`` (so sinks see a concrete status
    for the "awaiting approval" card), emits ``ApprovalRequested``, and
    awaits the per-task ``asyncio.Event`` the control dispatcher sets
    when the matching ``ControlMessage(APPROVE|REJECT)`` lands.

    Returns ``{"decision": "approve" | "reject", "detail": ...}`` so the
    agent can decide whether to proceed or transition the task to
    ``FAILED`` itself. A ``timeout_ms > 0`` that elapses before a
    decision lands returns ``{"decision": "timeout", "detail": ...}``
    and leaves the task blocked (the caller may re-prompt or fail).
    """
    task_id = _resolve_task_id(args, session)
    prompt = _str(args, "prompt")
    timeout_ms = _int(args, "timeout_ms", 0)
    if not task_id:
        return _missing_task_id_response("report_awaiting_approval")

    # Idempotency: reuse an existing waiter if one is already pending.
    waiter = session.pending_approvals.get(task_id)
    if waiter is None:
        waiter = asyncio.Event()
        session.pending_approvals[task_id] = waiter
    meta = session.pending_approvals_meta.setdefault(
        task_id,
        {"kind": "task", "prompt": prompt, "task_id": task_id},
    )
    # Refresh prompt in case it changed — the dispatcher only reads
    # ``decision`` and ``detail`` so this is safe.
    meta["prompt"] = prompt

    await steerer.mark_task_blocked(
        task_id,
        session=session,
        blocker="awaiting_approval",
        needed=prompt,
    )
    await _emit_approval_requested(
        session=session,
        steerer=steerer,
        target_id=task_id,
        kind="task",
        prompt=prompt,
        task_id=task_id,
        metadata={},
    )

    try:
        if timeout_ms > 0:
            await asyncio.wait_for(waiter.wait(), timeout=timeout_ms / 1000.0)
        else:
            await waiter.wait()
    except TimeoutError:
        return {
            "acknowledged": True,
            "decision": "timeout",
            "detail": f"no decision after {timeout_ms}ms",
        }

    decision = str(meta.get("decision", "")) or "approve"
    detail = str(meta.get("detail", ""))
    return {"acknowledged": True, "decision": decision, "detail": detail}


async def _emit_approval_requested(
    *,
    session: Session,
    steerer: Steerer,
    target_id: str,
    kind: str,
    prompt: str,
    task_id: str,
    metadata: dict[str, str],
) -> None:
    """Emit an ``ApprovalRequested`` through the steerer's bound sinks.

    Falls back to a no-op if the steerer lacks a sinks list (test stubs
    may drop the ``bind`` attribute). Proto-build errors are logged and
    swallowed — losing the event is better than failing the tool call.
    """
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    from goldfive.events import approval_requested_event, emit

    try:
        evt = approval_requested_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            target_id=target_id,
            kind=kind,
            prompt=prompt,
            task_id=task_id,
            metadata=metadata,
            session_id=session.id,
        )
    except Exception as exc:  # noqa: BLE001 — proto stubs may be missing in unit tests
        log.debug("approval_requested: proto event build failed: %s", exc)
        return
    try:
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("approval_requested: sink emit raised: %s", exc)


# ---------------------------------------------------------------------------
# Parameter schemas (JSON Schema draft-07 style)
# ---------------------------------------------------------------------------


def _object_schema(*, required: list[str], properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# NOTE: ``task_id`` is intentionally omitted from every schema's
# ``required`` list (goldfive#191). The adapter stamps
# ``goldfive.current_task_id`` onto session state at delegation time
# so the handler can default from state when the model doesn't supply
# the arg. Handlers still reject with the canonical
# ``missing_task_id`` shape when neither source resolves a value —
# so strictness is enforced at the handler layer, not the schema.

_SCHEMA_TASK_STARTED = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_PROGRESS = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_COMPLETED = _object_schema(
    required=["summary"],
    properties={
        "task_id": {"type": "string"},
        "summary": {"type": "string"},
        "artifacts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
)

_SCHEMA_TASK_FAILED = _object_schema(
    required=["reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
        "recoverable": {"type": "boolean"},
    },
)

_SCHEMA_TASK_BLOCKED = _object_schema(
    required=["blocker"],
    properties={
        "task_id": {"type": "string"},
        "blocker": {"type": "string"},
        "needed": {"type": "string"},
    },
)

_SCHEMA_NEW_WORK_DISCOVERED = _object_schema(
    required=["parent_task_id", "title", "description"],
    properties={
        "parent_task_id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "assignee": {"type": "string"},
    },
)

_SCHEMA_PLAN_DIVERGENCE = _object_schema(
    required=["note"],
    properties={
        "note": {"type": "string"},
        "suggested_action": {"type": "string"},
    },
)

_SCHEMA_AWAITING_APPROVAL = _object_schema(
    required=["prompt"],
    properties={
        "task_id": {"type": "string"},
        "prompt": {"type": "string"},
        "timeout_ms": {"type": "integer", "minimum": 0},
    },
)


# ---------------------------------------------------------------------------
# Built-in tool specs
# ---------------------------------------------------------------------------


BUILTIN_REPORTING_TOOLS: list[ReportingToolSpec] = [
    ReportingToolSpec(
        name="report_task_started",
        description=(
            "Report that you are beginning work on a planned task. Call this "
            "BEFORE doing the actual work so the framework knows which task "
            "is currently in progress."
        ),
        parameters=_SCHEMA_TASK_STARTED,
        handler=_handle_task_started,
    ),
    ReportingToolSpec(
        name="report_task_progress",
        description=(
            "Report mid-task progress. Optional — only call if the task has "
            "meaningful sub-steps. 'fraction' is a 0.0-1.0 hint of how far "
            "through the task you are."
        ),
        parameters=_SCHEMA_TASK_PROGRESS,
        handler=_handle_task_progress,
    ),
    ReportingToolSpec(
        name="report_task_completed",
        description=(
            "Report that you have completed a planned task successfully. "
            "Call this AFTER producing the final output. 'summary' describes "
            "the result in one or two sentences."
        ),
        parameters=_SCHEMA_TASK_COMPLETED,
        handler=_handle_task_completed,
    ),
    ReportingToolSpec(
        name="report_task_failed",
        description=(
            "Report that you were unable to complete a planned task. "
            "'recoverable=True' lets the plan route around this failure; "
            "'recoverable=False' means the whole workflow should probably stop."
        ),
        parameters=_SCHEMA_TASK_FAILED,
        handler=_handle_task_failed,
    ),
    ReportingToolSpec(
        name="report_task_blocked",
        description=(
            "Report that you cannot currently proceed with a task. Use this "
            "when an external blocker prevents progress. 'blocker' describes "
            "what is in the way; 'needed' describes what would unblock you."
        ),
        parameters=_SCHEMA_TASK_BLOCKED,
        handler=_handle_task_blocked,
    ),
    ReportingToolSpec(
        name="report_new_work_discovered",
        description=(
            "Report that you've discovered additional work the plan doesn't "
            "know about. The framework will ask the planner to add this task "
            "as a child of 'parent_task_id'."
        ),
        parameters=_SCHEMA_NEW_WORK_DISCOVERED,
        handler=_handle_new_work_discovered,
    ),
    ReportingToolSpec(
        name="report_plan_divergence",
        description=(
            "Report that the current plan no longer matches what needs to "
            "happen. The framework will trigger an explicit replan."
        ),
        parameters=_SCHEMA_PLAN_DIVERGENCE,
        handler=_handle_plan_divergence,
    ),
    ReportingToolSpec(
        name="report_awaiting_approval",
        description=(
            "Block the current task until a human approves or rejects via "
            "the control channel. Use this when the task has a side effect "
            "that needs sign-off (spending money, writing to a shared "
            "system, sending a message). The call blocks until the UI "
            "dispatches an APPROVE or REJECT and returns "
            "{'decision': 'approve' | 'reject' | 'timeout', 'detail': ...}. "
            "The agent decides what to do with the decision: on approve, "
            "proceed; on reject, typically report_task_failed with a "
            "user-rejection reason."
        ),
        parameters=_SCHEMA_AWAITING_APPROVAL,
        handler=_handle_awaiting_approval,
    ),
]


__all__ = [
    "ReportingHandler",
    "ReportingToolSpec",
    "REPORTING_TOOL_NAMES",
    "BUILTIN_REPORTING_TOOLS",
]
