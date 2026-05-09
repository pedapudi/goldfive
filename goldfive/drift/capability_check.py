"""Structural capability-mismatch detector for ``delegation_observed`` (goldfive#253).

Fires :class:`~goldfive.types.DriftKind.CAPABILITY_MISMATCH` when the
agent the coordinator delegated to *structurally* cannot perform the
bound task. Two narrow rules; false positives are worse than false
negatives at this layer because every fire cancels the in-flight
invocation and triggers a planner refine.

Replaces the planner-LLM ``PLAN_DIVERGENCE`` comparison for the
"wrong-assignee" case: instead of comparing the planner's *predicted*
assignee against the runtime delegation (and getting it wrong when the
planner LLM hallucinated the assignee), we ground the comparison in
*actual* tool capability surfaced by the ADK agent object.

Detection rules (intentionally surgical):

* **Rule A — coordinator-style leaf-assignment.** If every tool on the
  invoked agent is an ``AgentTool`` instance (i.e. its only capability
  is to delegate further) AND the bound task reads as a leaf authoring
  task ("draft", "write", "review", "research", "patch", "locate" …
  rather than "coordinate", "delegate", "orchestrate"), the agent
  cannot actually do the work. Fires CRITICAL.

* **Rule B — required-tools advisory.** If
  :attr:`~goldfive.types.Task.required_tools` is non-empty and the
  invoked agent's tool names do not cover every required name, fire
  CRITICAL. Skipped entirely when the advisory is empty (legacy plans
  and planners that don't populate it are a no-op).

Both rules return :class:`~goldfive.types.DriftEvent` carrying the
agent name + bound task id + the structural gap, suitable for the
goldfive intervention ladder. The detector is framework-neutral: it
takes ADK ``Tool`` objects but only reads their attributes (``.agent``
for AgentTool detection, ``.name`` for the required-tools cover).
"""

from __future__ import annotations

import logging
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Task

log = logging.getLogger(__name__)


__all__ = [
    "DELEGATION_VERB_MARKERS",
    "detect_capability_mismatch",
    "is_agent_tool",
]


#: Phrases that mark a task as *delegation/coordination* shaped rather
#: than a leaf authoring task. When a task description matches any of
#: these (case-insensitive substring), Rule A is suppressed even if the
#: invoked agent has only ``AgentTool`` wrappers — coordinating IS what
#: a coordinator does, and that is the agent's actual capability.
#:
#: Conservative by design: false positives here would suppress real
#: capability mismatches. Only include phrases that strongly imply the
#: task itself is orchestrational. Verbs like "review", "research",
#: "patch", "locate" are deliberately NOT here — they are leaf-task
#: verbs that a coordinator structurally cannot perform.
DELEGATION_VERB_MARKERS: tuple[str, ...] = (
    "coordinate",
    "delegate",
    "orchestrate",
    "dispatch",
    "route to",
    "hand off",
    "handoff",
)


def is_agent_tool(tool: Any) -> bool:
    """Return True if ``tool`` is an ADK ``AgentTool`` (sub-agent wrapper).

    Mirrors the discriminator in :mod:`goldfive.adapters._adk_plugin`:
    prefers ``isinstance(tool, AgentTool)`` when the optional ``adk``
    extra is importable, with a duck-typed fallback (``.agent``
    attribute) for test stubs and forward-compatibility. A plain
    ``FunctionTool`` carries ``.func`` instead, so the absence of
    ``.agent`` is a robust no-AgentTool signal.
    """
    if tool is None:
        return False
    try:
        from google.adk.tools import AgentTool  # type: ignore  # noqa: PLC0415

        if isinstance(tool, AgentTool):
            return True
    except Exception:  # noqa: BLE001 — adk extra not installed / import edge
        pass
    return getattr(tool, "agent", None) is not None


def _tool_name(tool: Any) -> str:
    """Best-effort extraction of an ADK ``Tool``'s public name."""
    if tool is None:
        return ""
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    func = getattr(tool, "func", None)
    if func is not None:
        func_name = getattr(func, "__name__", None)
        if isinstance(func_name, str) and func_name:
            return func_name
    return ""


def _looks_like_delegation_task(task: Task) -> bool:
    """Return True when the task title/description reads as orchestrational.

    Substring scan against :data:`DELEGATION_VERB_MARKERS`,
    case-insensitive. Empty title + description is treated as
    *non-delegation* (the conservative default for Rule A: if we can't
    tell, prefer to fire).
    """
    title = str(getattr(task, "title", "") or "")
    description = str(getattr(task, "description", "") or "")
    haystack = f"{title}\n{description}".lower()
    if not haystack.strip():
        return False
    return any(marker in haystack for marker in DELEGATION_VERB_MARKERS)


def detect_capability_mismatch(
    *,
    invoked_agent_name: str,
    invoked_agent_tools: list[Any],
    task: Task,
) -> DriftEvent | None:
    """Return a CAPABILITY_MISMATCH drift if ``invoked_agent_name`` cannot perform ``task``.

    Parameters
    ----------
    invoked_agent_name:
        Display name of the agent the coordinator delegated to. Used
        only for the human-readable detail string; an empty value still
        produces a usable event.
    invoked_agent_tools:
        Live ADK ``Tool`` objects from the invoked agent. The detector
        introspects each via :func:`is_agent_tool` and ``_tool_name``;
        no calls are made.
    task:
        The :class:`~goldfive.types.Task` the coordinator bound to the
        delegation. ``required_tools`` powers Rule B; ``title`` /
        ``description`` power Rule A's leaf-task heuristic.

    Returns
    -------
    DriftEvent | None
        ``None`` when neither rule trips OR when ``invoked_agent_tools``
        is empty AND ``required_tools`` is empty (no signal). A
        ``DriftEvent`` with ``severity=CRITICAL`` otherwise.
    """
    if task is None:
        return None

    tools = list(invoked_agent_tools or [])
    task_id = str(getattr(task, "id", "") or "")
    required_tools = tuple(getattr(task, "required_tools", ()) or ())

    # Rule B first — it consults explicit planner output, so it is the
    # higher-confidence signal. Fires only when populated; an empty
    # advisory is not a no-op miss, it's "no opinion".
    if required_tools:
        agent_tool_names = {n for n in (_tool_name(t) for t in tools) if n}
        missing = tuple(name for name in required_tools if name not in agent_tool_names)
        if missing:
            detail = (
                f"agent {invoked_agent_name!r} delegated for task "
                f"{task_id!r} is missing required tool(s) "
                f"{list(missing)!r}; available tools: "
                f"{sorted(agent_tool_names)!r}"
            )
            return DriftEvent(
                kind=DriftKind.CAPABILITY_MISMATCH,
                severity=DriftSeverity.CRITICAL,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=invoked_agent_name,
            )

    # Rule A — coordinator-style leaf-assignment. Empty tool list does
    # not trip Rule A: we cannot distinguish "agent has no tools" from
    # "test stub / introspection failure", and the cost of a false
    # positive (cancelling + refine) is high.
    if tools and all(is_agent_tool(t) for t in tools):
        if not _looks_like_delegation_task(task):
            detail = (
                f"agent {invoked_agent_name!r} has only AgentTool "
                f"wrappers ({len(tools)} delegation tools, no leaf "
                f"capability) but was delegated leaf task "
                f"{task_id!r} ({(task.title or '')[:80]!r}); "
                f"the agent structurally cannot perform this task"
            )
            return DriftEvent(
                kind=DriftKind.CAPABILITY_MISMATCH,
                severity=DriftSeverity.CRITICAL,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=invoked_agent_name,
            )

    return None
