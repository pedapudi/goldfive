"""Shared helper for invoking reporting tool handlers from adapters.

Adapters typically receive a tool call from their SDK as ``(name, args)``
and need to locate the matching :class:`ReportingToolSpec` and call its
handler with ``(args, session, steerer)``. This helper centralises that
lookup so that every adapter (CallableAdapter, ADK, Claude) routes through
the same code path — keeping behaviour and error messages consistent.

The dispatcher also threads every reporting-tool call through a per-task
guard (see :mod:`goldfive.adapters._tool_loop_guard`) so that:

* a duplicate call (same task, tool, args) returns a cheap ACK without
  re-invoking the underlying handler / Steerer transition; and
* a sustained burst of identical calls within a short window emits a
  ``LOOPING_TOOL_CALL`` drift so the planner can intervene.

The guard is applied to **all** reporting-tool calls — the eight
canonical reporting tools all carry an explicit ``task_id`` argument
(except ``report_plan_divergence``, which we exempt below). Plan-level
calls without a task scope are passed through unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from goldfive.adapters._tool_loop_guard import (
    args_signature,
    detect_loop,
    emit_loop_drift,
    guard_for,
)
from goldfive.reporting import ReportingToolSpec
from goldfive.types import TERMINAL_TASK_STATUSES

if TYPE_CHECKING:  # pragma: no cover - type-only
    from goldfive.protocols import Steerer
    from goldfive.types import Session, Task


# Tools that carry no task_id and represent plan-level signals — they
# are exempt from the per-task idempotency guard, but still feed the
# loop detector under a synthetic "(plan)" task key so a runaway
# divergence-spam loop is still caught.
_PLAN_LEVEL_TOOLS: frozenset[str] = frozenset({"report_plan_divergence"})

# Tools that are intentionally allowed to be called multiple times with
# the same args (the dispatch is the entire point — e.g. blocking on an
# approval decision). These bypass the idempotency table but still
# count toward the loop detector window.
_NON_IDEMPOTENT_TOOLS: frozenset[str] = frozenset({"report_awaiting_approval"})


def find_tool(
    tools: Iterable[ReportingToolSpec],
    name: str,
) -> ReportingToolSpec | None:
    """Return the first tool whose ``name`` matches, else ``None``."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def invoke_tool(
    tools: Iterable[ReportingToolSpec],
    name: str,
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
) -> dict[str, Any]:
    """Look up ``name`` in ``tools`` and invoke its handler.

    Raises :class:`KeyError` if no tool with that name is registered — this
    mirrors the behaviour a real SDK would exhibit when an agent hallucinates
    a tool name, and lets adapters surface a clean error to the agent.

    Idempotency: if this ``(task_id, name, args)`` has already been
    dispatched for this Session, the handler is skipped and the dispatcher
    returns ``{"acknowledged": true, "duplicate": true}`` instead of
    re-running the underlying Steerer transition. The first call still
    drives the UI; subsequent ones are a no-op + ACK.

    Loop detection: every call updates a per-task sliding window. Once
    enough recent calls match the same ``(name, args)`` signature, a
    one-shot ``LOOPING_TOOL_CALL`` drift is emitted (CRITICAL severity)
    so the planner can either fail the looping task or replan around it.
    """
    tool = find_tool(tools, name)
    if tool is None:
        raise KeyError(f"unknown reporting tool: {name!r}")

    if not isinstance(args, dict):
        # Defensive: handlers expect a dict-shaped args payload. Pass
        # through to preserve the SDK's KeyError behaviour rather than
        # silently coercing.
        return await tool.handler(args, session, steerer)

    task_id = str(args.get("task_id") or "").strip()
    guard_key = task_id or "__plan__"
    is_plan_level = name in _PLAN_LEVEL_TOOLS
    sig = (name, args_signature(args))

    # Terminal-task rejection (root-cause prevention). Models that have
    # already reported a task as COMPLETED/FAILED/CANCELLED sometimes
    # keep calling reporting tools for that task — the first call moved
    # the task to a terminal status (the Steerer's ``mark_task_*``
    # guards make subsequent transitions no-ops), but a bland ``ACK``
    # back to the agent doesn't communicate "stop." Without this check,
    # a model can burn dozens of reporting calls after the task is
    # already done. We return a structured error response so the model
    # has clear, actionable feedback to stop reporting against that
    # task. Plan-level tools (no ``task_id``) bypass this path.
    if not is_plan_level and task_id:
        task = _find_task(session, task_id)
        if task is not None and task.status in TERMINAL_TASK_STATUSES:
            return {
                "acknowledged": False,
                "error": "task_already_terminal",
                "task_id": task_id,
                "current_status": task.status.value,
                "message": (
                    f"Task {task_id!r} is already {task.status.value}. Do not "
                    "call further reporting tools for this task; wait for the "
                    "orchestrator to route you to the next task."
                ),
            }

    guard = guard_for(session)
    state = guard.state_for(guard_key)
    state.window.append(sig)

    if detect_loop(state, sig):
        await emit_loop_drift(
            session=session,
            steerer=steerer,
            task_id=task_id,
            tool_name=name,
        )

    if not is_plan_level and name not in _NON_IDEMPOTENT_TOOLS and task_id and sig in state.seen:
        return {"acknowledged": True, "duplicate": True}

    state.seen.add(sig)
    return await tool.handler(args, session, steerer)


def _find_task(session: Session, task_id: str) -> Task | None:
    """Return the task with ``task_id`` from ``session.plan``, or ``None``."""
    plan = session.plan
    if plan is None:
        return None
    for task in plan.tasks:
        if task.id == task_id:
            return task
    return None


__all__ = ["find_tool", "invoke_tool"]
