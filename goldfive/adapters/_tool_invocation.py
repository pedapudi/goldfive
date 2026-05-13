"""Shared helper for invoking reporting tool handlers from adapters.

Adapters typically receive a tool call from their SDK as ``(name, args)``
and need to locate the matching :class:`ReportingToolSpec` and call its
handler with ``(args, session, steerer)``. This helper centralises that
lookup so that every adapter (CallableAdapter, ADK, Claude) routes through
the same code path — keeping behaviour and error messages consistent.

Schema validation (Layer 1) is the only gate the dispatcher enforces on
its own. Terminal-task / idempotency / invalid-transition semantics all
live inside the reporting handlers themselves (goldfive#201); tool-loop
detection is covered by :class:`goldfive.drift.tool_loops.ToolLoopTracker`
which observes every tool call the ADK plugin sees (goldfive#181, #204)
and emits graduated-severity ``LOOPING_REASONING`` drifts.

Historically this module also hosted a per-task + session-wide
``ToolLoopGuard`` (goldfive#109). goldfive#206 retired it: the guard's
unconditional CRITICAL + hard-reject behaviour pre-dated both the
idempotent-handler layer (#203, benign retries no longer look like
loops) and the graduated tool-loop detector (#204, INFO/WARNING/CRITICAL
tiers), so it was actively firing CRITICAL drifts on benign idempotent
retries and aborting runs that the newer stack would have absorbed.

See ``docs/design/TASK-LIFECYCLE.md`` §5 for the current dispatch
contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from goldfive.reporting import ReportingToolSpec

if TYPE_CHECKING:  # pragma: no cover - type-only
    from goldfive.protocols import Steerer
    from goldfive.types import Session, Task


# Tools that carry no task_id and represent plan-level signals — they
# skip the schema-level task-id validation because they don't target a
# specific task.
_PLAN_LEVEL_TOOLS: frozenset[str] = frozenset({"report_plan_divergence"})


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
# :mod:`goldfive.state_store` to avoid forcing every adapter
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

    Schema validation short-circuits the handler: a task-scoped tool
    with no / unknown ``task_id`` returns a structured
    ``missing_task_id`` / ``unknown_task_id`` error without invoking
    the handler. Every other decision (terminal-state idempotency,
    invalid transitions, progress dispatch) lives in the handler
    itself (goldfive#201). Tool-loop detection is handled separately
    by :class:`goldfive.drift.tool_loops.ToolLoopTracker` at the ADK
    plugin's ``after_tool_callback`` — not gated inline on this
    dispatch path.
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
    # so all downstream paths see a single consistent value.
    if not task_id:
        fallback = _resolve_state_task_id(session)
        if fallback:
            task_id = fallback
            args["task_id"] = fallback
    is_plan_level = name in _PLAN_LEVEL_TOOLS

    # ------------------------------------------------------------------
    # Schema rejections. Task-scoped tools require a task_id that
    # exists in the current plan; anything else is a malformed call
    # the handler cannot serve.
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
        # The unknown-task check is only meaningful when the session has
        # an installed plan to look the id up in. Plan-less sessions do
        # occur (early bootstrap, minimal test harnesses) and we don't
        # want to spuriously reject a call that the adapter has
        # legitimately dispatched.
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
