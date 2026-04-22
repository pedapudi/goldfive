"""Shared helper for invoking reporting tool handlers from adapters.

Adapters typically receive a tool call from their SDK as ``(name, args)``
and need to locate the matching :class:`ReportingToolSpec` and call its
handler with ``(args, session, steerer)``. This helper centralises that
lookup so that every adapter (CallableAdapter, ADK, Claude) routes through
the same code path — keeping behaviour and error messages consistent.

The dispatcher threads every reporting-tool call through a four-layer
guard (see :mod:`goldfive.adapters._tool_loop_guard`) so that:

* **Layer 1 — schema rejections.** A call with no ``task_id`` for a
  task-scoped tool, or an unknown ``task_id`` not present in the
  current plan, is rejected with a structured error **before** any
  other layer runs (so malformed calls can't poison the session
  counter).
* **Layer 2 — terminal-task rejection.** A call on a task already in
  ``COMPLETED`` / ``FAILED`` / ``CANCELLED`` is rejected with
  ``task_already_terminal`` so the agent sees a clear stop signal.
* **Layer 3 — per-task loop guard.** Duplicate-args calls return a
  cheap ``duplicate`` ACK; a sustained burst (same signature) or
  volume cap (same tool name, varying args) emits a
  ``LOOPING_TOOL_CALL`` drift and flips that ``(task, tool)`` bucket
  into a hard-reject state so subsequent spam gets
  ``loop_detected`` instead of pass-through.
* **Layer 4 — session-wide volume cap.** A final safety net against
  adversarial callers that invent a fresh ``task_id`` every call,
  which would distribute one call per per-task bucket and defeat the
  per-task cap. Once a tool is called > 50 times across ALL tasks in
  a session, it is flagged session-wide and every subsequent call is
  hard-rejected.

See ``docs/design/TASK-LIFECYCLE.md`` §5 for the layering rationale.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from goldfive.adapters._tool_loop_guard import (
    args_signature,
    detect_loop,
    detect_session_loop,
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


# Orchestration-state key the adapter stamps at delegation time
# (goldfive#191). Duplicated here rather than imported from
# :mod:`goldfive.orchestration_state` to avoid forcing every adapter
# to pull in the orchestration module just to resolve a fallback.
_STATE_KEY_CURRENT_TASK_ID = "goldfive.current_task_id"


def _resolve_state_task_id(session: Session) -> str:
    """Return ``session.state["goldfive.current_task_id"]`` or ``""``.

    Tolerant of a missing ``state`` attribute and a non-mapping /
    non-string value — any degenerate shape yields the empty string
    so the caller's existing ``missing_task_id`` rejection still
    fires. Strips whitespace to match the ``task_id`` normalisation
    :func:`invoke_tool` already applies to the arg.
    """
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return ""
    raw = state.get(_STATE_KEY_CURRENT_TASK_ID, "")
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


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

    See the module docstring for the layer ordering. In brief, the layers
    fire in this order and any rejection short-circuits the rest:

    1. Schema validation (missing / unknown ``task_id``) — malformed
       calls never reach the counters so an adversarial flood can't
       poison session-wide state.
    2. Terminal-task rejection — structured ``task_already_terminal``.
    3. Per-task loop guard — duplicate ACK, or ``loop_detected`` once
       the bucket is flagged.
    4. Session-wide volume cap — ``loop_detected`` once the tool is
       flagged session-wide.
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
    # goldfive#191 Layer 2: fall back to the orchestration-state pin
    # when the model's tool call omits ``task_id``. The adapter's
    # ``before_agent_callback`` stamps ``goldfive.current_task_id`` at
    # delegation time when the starting sub-agent has an unambiguous
    # plan task assignment. Mutating ``args`` here (rather than just
    # using a local) makes the resolved id visible to the handler too,
    # so all downstream paths (idempotency signature, terminal-task
    # lookup, handler body) see a single consistent value.
    if not task_id:
        fallback = _resolve_state_task_id(session)
        if fallback:
            task_id = fallback
            args["task_id"] = fallback
    guard_key = task_id or "__plan__"
    is_plan_level = name in _PLAN_LEVEL_TOOLS
    sig = (name, args_signature(args))

    # ------------------------------------------------------------------
    # Layer 1 — schema rejections. These MUST run before any counter
    # updates so a malformed-call spam can't poison session state.
    # ------------------------------------------------------------------
    if not is_plan_level:
        if not task_id:
            return {
                "acknowledged": False,
                "error": "missing_task_id",
                "tool": name,
                "message": (
                    f"Tool {name!r} requires a task_id; call it with the id "
                    "of the task you're reporting on."
                ),
            }
        # The unknown-task and terminal-task checks are only meaningful
        # when the session has an installed plan to look the id up in.
        # Plan-less sessions do occur (early bootstrap, minimal test
        # harnesses) and we don't want to spuriously reject a call
        # that the adapter has legitimately dispatched.
        if session.plan is not None:
            task = _find_task(session, task_id)
            if task is None:
                return {
                    "acknowledged": False,
                    "error": "unknown_task_id",
                    "tool": name,
                    "task_id": task_id,
                    "message": (
                        f"Task with id {task_id!r} does not exist in the "
                        "current plan. Re-check the current plan via session "
                        "state and use an id that appears in it."
                    ),
                }

            # Layer 2 — terminal-task rejection. Models that have
            # already reported a task as COMPLETED/FAILED/CANCELLED
            # sometimes keep calling reporting tools for that task —
            # the first call moved the task to a terminal status (the
            # Steerer's ``mark_task_*`` guards make subsequent
            # transitions no-ops), but a bland ``ACK`` back to the
            # agent doesn't communicate "stop." Without this check, a
            # model can burn dozens of reporting calls after the task
            # is already done. Structured error response so the model
            # has clear, actionable feedback to stop reporting.
            if task.status in TERMINAL_TASK_STATUSES:
                return {
                    "acknowledged": False,
                    "error": "task_already_terminal",
                    "task_id": task_id,
                    "current_status": task.status.value,
                    "message": (
                        f"Task {task_id!r} is already {task.status.value}. "
                        "Do not call further reporting tools for this task; "
                        "wait for the orchestrator to route you to the next "
                        "task."
                    ),
                }

    # ------------------------------------------------------------------
    # Layer 3 — per-task loop guard.
    # ------------------------------------------------------------------
    guard = guard_for(session)
    state = guard.state_for(guard_key)

    # If this (task, tool) bucket is already flagged, hard-reject
    # without updating the sliding window or incrementing counters
    # further. This is the fix for "one-shot loop flag lets subsequent
    # calls pass through": after the drift fires once, every future
    # call to that tool on that task gets a structured
    # ``loop_detected`` error instead of silently falling through to
    # the handler (which was the live-run failure).
    if state.loop_flagged and state.loop_tool == name:
        return _loop_rejection(
            task_id=task_id,
            tool_name=name,
            reason=state.loop_reason or "per_task_loop",
            scope="per_task",
        )

    state.window.append(sig)

    if detect_loop(state, sig):
        await emit_loop_drift(
            session=session,
            steerer=steerer,
            task_id=task_id,
            tool_name=name,
            reason=_human_reason(state.loop_reason),
        )
        # The call that trips the guard is itself rejected — there's
        # no forward progress to preserve (the agent is already
        # spamming) and letting it through would be our 16th+ handler
        # invocation on a confirmed-looping task.
        return _loop_rejection(
            task_id=task_id,
            tool_name=name,
            reason=state.loop_reason,
            scope="per_task",
        )

    if not is_plan_level and name not in _NON_IDEMPOTENT_TOOLS and task_id and sig in state.seen:
        return {"acknowledged": True, "duplicate": True}

    # ------------------------------------------------------------------
    # Layer 4 — session-wide volume cap. Runs AFTER Layer 1 (so
    # malformed calls don't count) but before we invoke the handler,
    # so the flood is cut off at its 51st call, not its 500th.
    # ------------------------------------------------------------------
    if detect_session_loop(guard, name):
        await emit_loop_drift(
            session=session,
            steerer=steerer,
            task_id=task_id or "(session)",
            tool_name=name,
            reason=(
                f"session-wide volume cap ({guard.session_tool_count[name]} "
                "calls across all tasks, no forward progress)"
            ),
        )
        return _loop_rejection(
            task_id=task_id,
            tool_name=name,
            reason="session_volume_cap",
            scope="session",
        )
    if name in guard.session_tool_flagged:
        # Tool was already session-flagged on a prior call; keep
        # rejecting without re-firing drift.
        return _loop_rejection(
            task_id=task_id,
            tool_name=name,
            reason="session_volume_cap",
            scope="session",
        )

    state.seen.add(sig)
    return await tool.handler(args, session, steerer)


def _loop_rejection(*, task_id: str, tool_name: str, reason: str, scope: str) -> dict[str, Any]:
    """Build the structured ``loop_detected`` response.

    All rejection paths funnel through this helper so the response
    shape stays consistent and the agent sees the same error key
    (``loop_detected``) regardless of which trigger fired. ``scope``
    is ``"per_task"`` or ``"session"``; ``reason`` is a machine-
    readable classifier (``exact_signature_burst``,
    ``per_task_volume_cap``, ``session_volume_cap``).
    """
    message: str
    if scope == "session":
        message = (
            f"Tool {tool_name!r} has been called too many times across all "
            "tasks in this session without forward progress. Do not retry; "
            "wait for the orchestrator to route you to the next task."
        )
    else:
        message = (
            f"Repeated calls to {tool_name!r} without forward progress "
            f"detected (reason: {reason}). Do not retry; wait for the "
            "orchestrator to route you to the next task."
        )
    payload: dict[str, Any] = {
        "acknowledged": False,
        "error": "loop_detected",
        "tool": tool_name,
        "reason": reason,
        "scope": scope,
        "message": message,
    }
    if task_id:
        payload["task_id"] = task_id
    return payload


def _human_reason(machine_reason: str) -> str:
    """Map a machine classifier to a human-readable drift detail fragment."""
    if machine_reason == "exact_signature_burst":
        return (
            "exact-signature burst: same (tool, args) seen repeatedly in the recent sliding window"
        )
    if machine_reason == "per_task_volume_cap":
        return "per-task volume cap: too many total calls to this tool on this task"
    return machine_reason or "repeated calls without forward progress"


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
