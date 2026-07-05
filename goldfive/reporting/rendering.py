"""Presentation helpers for reporting-tool responses.

The functions here all return payload dicts that the agent's LLM will
read on the next turn — they're the contract surface between
:mod:`goldfive.reporting.handlers` and the model. Three families:

* **Directive acks** (:func:`_build_plan_state`, :func:`_directive_ack`)
  — successful transitions return ``plan_state`` so the coordinator
  sees the next pending hand-off instead of an information-free ack.
  This is the F1 / Tier 1 loop-prevention pattern.
* **Idempotent / invalid / refused** responses — branch shapes for
  retries, contract violations, and stale-pin refusals. Distinct
  shapes so loop-detector / observability layers can tell them apart.
* **Missing-arg rejections** — the canonical
  ``missing_task_id`` / ``missing_required_field`` rejection bodies
  the handlers emit before driving the steerer.

The output of :func:`_build_plan_state` is byte-identical to the
pre-split renderer; the LLM sees it inside directive responses and
any drift would shift coordinator behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldfive.reporting._internal import _ACK, _TERMINAL_STATUSES
from goldfive.types import TaskStatus

if TYPE_CHECKING:
    from goldfive.types import Session


# ---------------------------------------------------------------------------
# Tier 1 / F1 — directive tool responses (loop prevention)
# ---------------------------------------------------------------------------
#
# Every report_task_* handler returns a richer payload than the historical
# ``{"acknowledged": True}`` ack. The payload anchors the LLM's "what do
# I do next?" reasoning by embedding the live plan_state, so a coordinator
# that just completed a task sees the next pending hand-off instead of an
# information-free ack and looping back onto the just-finished work.
#
# Shape:
#   {
#     "acknowledged": True,
#     "task": {"id": <task_id>, "status": <new_status_str>},
#     "plan_state": {
#        "completed_task_ids": [...sorted ids of COMPLETED tasks...],
#        "next_pending": {
#            "id": ..., "title": ..., "assigned_to": <bare agent name>,
#            "predecessors_completed": True,
#        } | None,
#     },
#   }
#
# Idempotent / invalid / refused responses keep their existing shapes —
# they signal "do nothing" to the LLM and don't need the directive surface.
# This is the pre-dispatch loop-source closure pattern (see
# ``docs/design/`` for the loop-prevention strategy).


def _next_pending_with_completed_predecessors(plan: Any) -> Any | None:
    """Return the first PENDING task whose incoming edges all have terminal predecessors.

    Walks ``plan.tasks`` in declared order (topological-ish — refine
    preserves the original ordering for unmutated stages) and returns
    the first ``Task`` whose every incoming-edge predecessor is in a
    terminal status (``TERMINAL_TASK_STATUSES``). Returns ``None`` when
    no such task exists.

    Note: a task with NO incoming edges trivially passes the
    "predecessors_completed" gate and is returned if PENDING.
    """
    if plan is None:
        return None
    tasks = list(getattr(plan, "tasks", None) or ())
    if not tasks:
        return None
    edges = list(getattr(plan, "edges", None) or ())
    by_id = {str(getattr(t, "id", "") or ""): t for t in tasks}
    incoming: dict[str, list[str]] = {tid: [] for tid in by_id}
    for e in edges:
        to_id = str(getattr(e, "to_task_id", "") or "")
        from_id = str(getattr(e, "from_task_id", "") or "")
        if to_id in incoming and from_id:
            incoming[to_id].append(from_id)
    for task in tasks:
        if getattr(task, "status", None) is not TaskStatus.PENDING:
            continue
        tid = str(getattr(task, "id", "") or "")
        if not tid:
            continue
        preds = incoming.get(tid, [])
        if all(
            (by_id.get(p) is not None)
            and (getattr(by_id[p], "status", None) in _TERMINAL_STATUSES)
            for p in preds
        ):
            return task
    return None


def _bare_agent_name(name: str) -> str:
    """Return the bare agent name (last dot-separated segment).

    Display-only: fully-qualified ADK agent paths like
    ``coordinator.research_agent`` collapse to ``research_agent`` so
    the LLM sees a name it can pass back as the AgentTool target.
    (Correction keys, by contrast, use the verbatim assignee id — see
    ``_correction_injection._normalize_agent_name``.)
    """
    s = (name or "").strip()
    if not s:
        return ""
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def _build_plan_state(plan: Any) -> dict[str, Any]:
    """Return the F1 ``plan_state`` block for a directive tool response."""
    if plan is None:
        return {"completed_task_ids": [], "next_pending": None}
    tasks = list(getattr(plan, "tasks", None) or ())
    completed_ids = sorted(
        str(getattr(t, "id", "") or "")
        for t in tasks
        if getattr(t, "status", None) is TaskStatus.COMPLETED
        and getattr(t, "id", "")
    )
    next_pending = _next_pending_with_completed_predecessors(plan)
    next_pending_payload: dict[str, Any] | None = None
    if next_pending is not None:
        next_pending_payload = {
            "id": str(getattr(next_pending, "id", "") or ""),
            "title": str(getattr(next_pending, "title", "") or ""),
            "assigned_to": _bare_agent_name(
                str(getattr(next_pending, "assignee_agent_id", "") or "")
            ),
            # The selector only returns tasks whose every predecessor is
            # terminal; expose the assertion so the LLM doesn't have to
            # re-derive it.
            "predecessors_completed": True,
        }
    return {
        "completed_task_ids": completed_ids,
        "next_pending": next_pending_payload,
    }


def _directive_ack(
    *,
    session: Session,
    task_id: str,
    new_status: TaskStatus,
) -> dict[str, Any]:
    """Build the F1 directive payload for a report_task_* handler.

    ``new_status`` is the status the call moved (or would move) the task
    INTO. Idempotent / invalid / refused branches use their own shapes
    and do not call this helper.
    """
    return {
        "acknowledged": True,
        "task": {"id": task_id, "status": new_status.value},
        "plan_state": _build_plan_state(getattr(session, "plan", None)),
    }


def _idempotent_response(
    current_status: TaskStatus,
    *,
    session: Session | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    """Idempotent ack for a re-report on a task already in ``current_status``.

    F1 (loop prevention): when the LLM re-reports a task that's already
    terminal, the response still carries the live ``plan_state`` so the
    coordinator sees the next pending hand-off instead of an
    information-free ack — the same anchor the directive ack provides
    on real transitions. Without ``session`` the helper degrades to the
    pre-F1 shape (legacy callers, test stubs).
    """
    response: dict[str, Any] = {
        "acknowledged": True,
        "idempotent": True,
        "current_status": current_status.value,
    }
    if session is not None:
        response["task"] = {"id": task_id, "status": current_status.value}
        response["plan_state"] = _build_plan_state(getattr(session, "plan", None))
    return response


def _invalid_transition_response(
    *,
    tool_name: str,
    current_status: TaskStatus,
    attempted: TaskStatus,
    task_id: str,
) -> dict[str, Any]:
    return {
        "acknowledged": False,
        "error": "invalid_transition",
        "tool": tool_name,
        "task_id": task_id,
        "current_status": current_status.value,
        "attempted": attempted.value,
        "message": (
            f"Cannot {tool_name!r} task {task_id!r} from {current_status.value} "
            f"to {attempted.value}. The task is already in a terminal or "
            "otherwise-incompatible state; do not retry."
        ),
    }


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


def _missing_required_field_response(
    *,
    tool_name: str,
    field: str,
    reason: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Canonical rejection shape for a missing/empty required field.

    Mirrors the structure of :func:`_missing_task_id_response` —
    ``{"acknowledged": False, "error": ..., "tool": ..., "message":
    ...}`` plus a ``field`` and ``schema`` so the LLM can self-correct
    on the next turn. The ``schema`` echo is the parameters block the
    handler is enforcing; tool dispatchers / observability layers can
    use it to render a hint.
    """
    properties = schema.get("properties") or {}
    expected = properties.get(field) or {}
    return {
        "acknowledged": False,
        "error": "missing_required_field",
        "tool": tool_name,
        "field": field,
        "reason": reason,
        "expected": expected,
        "required": list(schema.get("required") or []),
        "message": (
            f"Tool {tool_name!r} requires field {field!r} to be a non-empty "
            f"value; received {reason}. Call the tool again with all "
            f"required fields populated."
        ),
    }


def _refused_response() -> dict[str, Any]:
    """Return the ack-only response for a refused stale-pin transition.

    The LLM still sees ``{"acknowledged": True}`` rather than an error
    payload — surfacing the refusal as a structured error would create a
    prompt-injection surface (the LLM might reason against the rejection
    and bypass the contract). Operators see the refusal via the
    ``task_transition_refused`` sink event.
    """
    return dict(_ACK)


__all__ = [
    "_bare_agent_name",
    "_build_plan_state",
    "_directive_ack",
    "_idempotent_response",
    "_invalid_transition_response",
    "_missing_required_field_response",
    "_missing_task_id_response",
    "_next_pending_with_completed_predecessors",
    "_refused_response",
]
