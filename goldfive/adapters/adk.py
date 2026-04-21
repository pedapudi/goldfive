"""ADK :class:`~goldfive.protocols.AgentAdapter` for Google's Agent Development Kit.

Wraps an ADK ``BaseAgent`` (or an existing ``Runner``) and conforms it
to goldfive's :class:`~goldfive.protocols.AgentAdapter` protocol so
goldfive's runner / executor / steerer can drive it uniformly.

Single-Runner model (goldfive#130)
----------------------------------

``wrap(root)`` produces ONE :class:`~google.adk.runners.InMemoryRunner`
around the caller-supplied root agent. ``ADKAdapter.invoke(task, ...)``
drives that one runner. Delegation within the tree happens via ADK's
native mechanisms (``AgentTool``, ``transfer_to_agent``, ``sub_agents``)
— goldfive does not route tasks to per-agent runners.

This is the revert of goldfive#120's registry-dispatch model. The
dispatch-by-assignee approach broke the "one tree, one Runner" invariant
that adk-web and harmonograf rely on, and the cascade of integration
fixes (#121-#126, harmonograf#55/#57/#58) never fully closed the seam.
The real root cause of the coordinator-looping regression that prompted
#120 was the coordinator's **prompt** describing a pipeline — which is
outside goldfive's control because users bring their own trees. The
backstop is now an AgentTool-per-invoke cap enforced by the plugin
(:class:`_GoldfiveADKPlugin`) — see ``agent_tool_cap`` below.

Responsibilities:

* **Reporting tool registration** — :meth:`ADKAdapter.register_reporting_tools`
  accepts goldfive :class:`~goldfive.reporting.ReportingToolSpec` values,
  wraps each as a ``google.adk.tools.FunctionTool`` around a thin sync
  shim, and attaches the tools to the root agent plus every sub-agent
  reachable via ``sub_agents`` / ``inner_agent`` / nested ``AgentTool.agent``.
  The plugin intercepts the calls in ``before_tool_callback`` and routes
  their arguments to the spec's real handler.
* **Invocation** — :meth:`ADKAdapter.invoke` hands a per-invocation
  :class:`~._adk_plugin.SessionContext` to the plugin, drives one turn
  via ``runner.run_async(...)``, consumes the event stream, and
  returns an :class:`~goldfive.results.InvocationResult`.

Optional dependency
-------------------

``google.adk`` is an optional install. The top-level import in this
module is gated so ``import goldfive.adapters.adk`` raises a clear
``ImportError`` with install instructions when ADK is missing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters._adk_plugin import (
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.results import InvocationResult
from goldfive.types import TERMINAL_TASK_STATUSES

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.reporting import ReportingToolSpec
    from goldfive.types import Session, Task, TaskStatus  # noqa: F401

log = logging.getLogger("goldfive.adapters.adk")


try:  # noqa: SIM105 — explicit import-time guard with install hint
    import google.adk  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover — covered in tests via importorskip
    raise ImportError("goldfive.adapters.adk requires 'pip install goldfive[adk]'") from None


DEFAULT_AGENT_TOOL_CAP = 16
"""Default per-invocation AgentTool-spawn cap for a single ``invoke``.

The cap is enforced by :class:`~goldfive.adapters._adk_plugin._GoldfiveADKPlugin`
and short-circuits an invocation that delegates more than this many
times in a single turn. Picked to be comfortably higher than any
legitimate coordinator pattern but well below ADK's 500-LLM-call
ceiling — a runaway coordinator hits the cap in a handful of loops, not
minutes of wasted LLM calls.
"""


def _build_ack_shim(name: str, description: str):
    """Return a sync function named ``name`` that returns an ACK dict.

    Used as the body of each ``FunctionTool`` so ADK has a callable
    with a proper ``__name__`` / docstring for the model to see. The
    real handler routing happens in the plugin's
    ``before_tool_callback`` — this shim is never actually executed.
    """

    def _shim(**kwargs: Any) -> dict[str, Any]:
        return {"acknowledged": True}

    _shim.__name__ = name
    _shim.__qualname__ = name
    _shim.__doc__ = description or f"Reporting tool: {name}"
    return _shim


def _build_function_tool(spec: ReportingToolSpec) -> Any:
    """Wrap a :class:`ReportingToolSpec` as a ``google.adk.tools.FunctionTool``."""
    from google.adk.tools import FunctionTool  # type: ignore

    shim = _build_ack_shim(spec.name, spec.description)
    return FunctionTool(shim)


def _augment_subtree_with_reporting(root_agent: Any, tools: list[Any], tool_names: set[str]) -> int:
    """Append reporting ``tools`` to every agent reachable from ``root_agent``.

    Traverses three edges to cover the shapes goldfive must support:

    * ``sub_agents`` — native ADK agent tree.
    * ``inner_agent`` — wrapper agents (e.g. HarmonografAgent-style).
    * ``AgentTool.agent`` — agents exposed to a parent as tools.

    Idempotent: agents that already carry any of the canonical reporting
    tool names are skipped. Returns the number of agents touched.

    This survives the goldfive#130 single-Runner revert because the
    reporting-tool coverage contract is independent of runner topology —
    every reachable agent needs to be able to call ``report_task_*``
    regardless of who drives it. The augmentation is also what makes
    early-exit on ``_task_is_terminal`` work inside an AgentTool
    sub-invocation.
    """
    if root_agent is None or not tools:
        return 0

    touched = 0
    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        children = list(getattr(cur, "sub_agents", None) or ())
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            children.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                children.append(nested)
        for child in children:
            stack.append(child)

        existing = getattr(cur, "tools", None)
        if existing is None:
            # Agent doesn't carry a tool list — nothing to augment.
            continue
        existing_names: set[str] = set()
        for t in existing:
            n = getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
            if n:
                existing_names.add(str(n))
        if any(n in existing_names for n in tool_names):
            continue
        new_list = list(existing) + list(tools)
        try:
            cur.tools = new_list
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "could not augment tools on %s: %s",
                getattr(cur, "name", "?"),
                exc,
            )
            continue
        touched += 1

    if touched:
        log.info("goldfive: registered reporting tools on %d sub-agents", touched)
    return touched


def _collect_reachable_agent_names(root_agent: Any) -> list[str]:
    """Return the sorted unique names of every agent reachable from ``root_agent``.

    Follows ``sub_agents`` / ``inner_agent`` / ``AgentTool.agent`` edges
    — the same traversal :func:`_augment_subtree_with_reporting` uses.
    Used only to populate :attr:`ADKAdapter.available_agents` so the
    planner can see the delegation targets available in the tree.

    Does NOT build a dispatch registry — under the single-Runner model
    (goldfive#130) the adapter drives one runner and ADK handles
    delegation. This is a shallow observation, not a routing seam.
    """
    if root_agent is None:
        return []

    names: set[str] = set()
    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        name = getattr(cur, "name", None)
        if isinstance(name, str) and name:
            names.add(name)

        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                stack.append(nested)
    return sorted(names)


GOLDFIVE_PLANNER_OPT_OUT_ATTR = "_goldfive_planner_opt_out"
"""Attribute users can set on an agent to skip GoldfivePlanner attachment.

When ``getattr(agent, "_goldfive_planner_opt_out", False)`` is truthy
the auto-attachment pass leaves that agent's ``planner`` untouched.
Applies per-agent — a sibling in the same tree without the marker
still gets a GoldfivePlanner. Rarely needed; the planner is designed
to be compose-safe with any user-supplied ``BasePlanner``, so the
opt-out is reserved for cases where even the additive injection is
undesired (e.g. a highly constrained research agent that must emit
only tool calls in a specific shape).
"""


def _attach_goldfive_planner_to_tree(root_agent: Any) -> int:
    """Attach :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner`
    to every ``LlmAgent`` reachable from ``root_agent``.

    Traverses the same three edges :func:`_augment_subtree_with_reporting`
    uses (``sub_agents`` / ``inner_agent`` / ``AgentTool.agent``). For
    every node that looks like an ``LlmAgent`` (duck-typed via the
    presence of a ``planner`` attribute — LlmAgents have it as an
    optional pydantic field):

    * If :data:`GOLDFIVE_PLANNER_OPT_OUT_ATTR` is set and truthy on
      the agent, skip.
    * If ``agent.planner`` is ``None``: attach a fresh
      :class:`GoldfivePlanner`.
    * If ``agent.planner`` is already a :class:`GoldfivePlanner`: skip
      (idempotent — re-wrap on a re-binded adapter should not stack).
    * Otherwise compose: replace with
      ``GoldfivePlanner(user_planner=agent.planner)``.

    Non-LlmAgents (``SequentialAgent``, ``ParallelAgent``, custom
    ``BaseAgent`` subclasses without a ``planner`` field) have no
    ``planner`` attribute and are skipped silently.

    Returns the number of agents where an attachment happened (not
    counting opt-outs / already-goldfive cases).

    Deliberately silent on individual failures: assigning to
    ``agent.planner`` may raise if the agent's pydantic model has
    frozen configuration; in that case we log and continue so one
    bad agent doesn't block the rest of the tree.
    """
    if root_agent is None:
        return 0
    try:
        from goldfive.planners.goldfive_planner import GoldfivePlanner
    except ImportError:  # pragma: no cover
        return 0

    touched = 0
    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        # Push children BEFORE the skip-checks so opted-out / non-LlmAgent
        # nodes still propagate the walk to their children.
        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                stack.append(nested)

        if getattr(cur, GOLDFIVE_PLANNER_OPT_OUT_ATTR, False):
            continue
        # LlmAgent carries a ``planner: Optional[BasePlanner]`` field;
        # other BaseAgent subclasses (Sequential/Parallel/custom) do
        # not expose one. Duck-type by presence + settability.
        if not hasattr(cur, "planner"):
            continue
        existing = getattr(cur, "planner", None)
        if isinstance(existing, GoldfivePlanner):
            # Idempotent: a re-wrap on the same tree (e.g. a test that
            # constructs two adapters over the same agent) should not
            # stack GoldfivePlanners on top of each other.
            continue
        try:
            if existing is None:
                cur.planner = GoldfivePlanner()
            else:
                cur.planner = GoldfivePlanner(user_planner=existing)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "could not attach GoldfivePlanner to %s: %s",
                getattr(cur, "name", "?"),
                exc,
            )
            continue
        touched += 1

    if touched:
        log.debug("goldfive: attached GoldfivePlanner to %d agent(s)", touched)
    return touched


def _rebind_goldfive_planners(
    root_agent: Any,
    *,
    agent_registry: list[str],
    steerer: Any,
    session: Any,
) -> None:
    """Rebind every :class:`GoldfivePlanner` in the tree to the live collaborators.

    Called from :meth:`ADKAdapter._invoke_internal` right before
    ``runner.run_async`` so the planner sees the current steerer,
    session, and the authoritative agent registry. Cheap walk — the
    tree is small and this runs once per invocation.
    """
    if root_agent is None:
        return
    try:
        from goldfive.planners.goldfive_planner import GoldfivePlanner
    except ImportError:  # pragma: no cover
        return

    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                stack.append(nested)

        planner = getattr(cur, "planner", None)
        if isinstance(planner, GoldfivePlanner):
            try:
                planner.bind(
                    agent_registry=agent_registry,
                    steerer=steerer,
                    session=session,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "could not rebind GoldfivePlanner on %s: %s",
                    getattr(cur, "name", "?"),
                    exc,
                )
def _collect_reachable_agent_tree(root_agent: Any) -> list[dict[str, Any]]:
    """Return structured metadata for every agent reachable from ``root_agent``.

    Each entry is a dict with ``name``, ``depth``, ``parent``, ``role``
    and ``kind``:

    * ``depth``: 0 for the root agent, N for depth-N descendants.
    * ``parent``: the name of the parent agent. Empty string for root.
    * ``role``: ``"root"`` for the root agent, ``"intermediate"`` for
      agents that have children, ``"leaf"`` for agents with no
      children.
    * ``kind``: the ADK class name — ``LlmAgent``, ``SequentialAgent``,
      ``ParallelAgent``, ``BaseAgent``, or any custom subclass name.
      Tree-agnostic: the function does not interpret or rename kinds.

    Follows the same ``sub_agents`` / ``inner_agent`` / ``AgentTool.agent``
    edges as :func:`_collect_reachable_agent_names` and
    :func:`_augment_subtree_with_reporting`. First BFS visit wins so
    depth / parent are minimal; duplicate reachable edges are collapsed
    by ``id(agent)`` identity to avoid double-counting shared agents.

    Used to populate :attr:`ADKAdapter.available_agents_tree`, which the
    LLM planner consumes to constrain ``assignee_agent_id`` selection.
    Under the single-Runner model (goldfive#130) this remains advisory —
    goldfive drives one runner and ADK handles delegation.
    """
    if root_agent is None:
        return []

    # BFS so depth is minimal (first reachable edge wins). We record
    # entries keyed by id() so shared sub-tree references don't get
    # double-appended with a different depth.
    from collections import deque

    visited: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    queue: deque[tuple[Any, int, str]] = deque()
    queue.append((root_agent, 0, ""))
    while queue:
        cur, depth, parent = queue.popleft()
        if cur is None:
            continue
        key = id(cur)
        if key in visited:
            continue
        name = str(getattr(cur, "name", "") or "")
        if not name:
            # Agents without a usable name are not routable; skip them
            # so the planner only ever sees named targets. Children
            # are still walked so named descendants are not lost.
            pass
        kind = type(cur).__name__
        children: list[Any] = []
        for sub in getattr(cur, "sub_agents", None) or ():
            children.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            children.append(inner)
        for t in getattr(cur, "tools", None) or ():
            nested = getattr(t, "agent", None)
            if nested is not None:
                children.append(nested)

        if name:
            role = "root" if depth == 0 else ("intermediate" if children else "leaf")
            visited[key] = {
                "name": name,
                "depth": depth,
                "parent": parent,
                "role": role,
                "kind": kind,
            }
            order.append(key)

        for child in children:
            child_key = id(child) if child is not None else 0
            if child_key and child_key not in visited:
                queue.append((child, depth + 1, name))

    return [visited[k] for k in order]


def _register_plugin_on_runner(runner: Any, plugin: Any) -> bool:
    """Install ``plugin`` on ``runner``'s plugin manager if one exists.

    Tolerates both ``runner.plugin_manager.register(plugin)`` and the
    newer ``runner.plugins.append(plugin)`` shapes. Returns True if the
    plugin was installed.
    """
    if runner is None:
        return False
    pm = getattr(runner, "plugin_manager", None)
    if pm is not None:
        for meth in ("register", "register_plugin", "add"):
            fn = getattr(pm, meth, None)
            if callable(fn):
                try:
                    fn(plugin)
                    return True
                except Exception as exc:  # noqa: BLE001
                    log.debug("runner.plugin_manager.%s raised: %s", meth, exc)
        plugins = getattr(pm, "plugins", None)
        if isinstance(plugins, list):
            plugins.append(plugin)
            return True
    plugins_attr = getattr(runner, "plugins", None)
    if isinstance(plugins_attr, list):
        plugins_attr.append(plugin)
        return True
    return False


def _build_runner(agent: Any, plugins: list[Any] | None = None) -> Any:
    """Construct an ADK ``InMemoryRunner`` around a ``BaseAgent``.

    Used by :class:`ADKAdapter` when the caller passes an agent rather
    than a runner. ``app_name`` defaults to the agent's name for tidy
    session-service bookkeeping.

    When ``plugins`` is a non-empty iterable, the plugin list is forwarded
    to :class:`InMemoryRunner` via its ``plugins=`` kwarg. Under the
    single-Runner model there is exactly one runner, and ADK propagates
    its plugin manager into any AgentTool-spawned sub-Runners
    automatically — so installing the plugin here is sufficient for the
    whole tree.
    """
    from google.adk.runners import InMemoryRunner  # type: ignore

    app_name = str(getattr(agent, "name", "") or "goldfive")
    kwargs: dict[str, Any] = {"agent": agent, "app_name": app_name}
    if plugins:
        kwargs["plugins"] = list(plugins)
    return InMemoryRunner(**kwargs)


def _looks_like_runner(obj: Any) -> bool:
    """Duck-type check for an ADK Runner.

    Runners expose ``run_async`` / ``agent`` / ``session_service``;
    agents may expose ``run_async_impl`` but not the session service.
    """
    return callable(getattr(obj, "run_async", None)) and getattr(obj, "agent", None) is not None


def _extract_text_from_event(event: Any) -> str:
    """Return the non-thought text from an ADK ``Event``, if any."""
    content = getattr(event, "content", None)
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or ()
    out: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", "") or ""
        if text:
            out.append(str(text))
    return "\n".join(out).strip()


def _is_final_event(event: Any) -> bool:
    """Duck-type check for a terminal ADK event."""
    attr = getattr(event, "is_final_response", None)
    if callable(attr):
        try:
            return bool(attr())
        except Exception:
            return False
    return bool(attr)


def _task_is_terminal(task: Task, session: Session) -> bool:
    """Return True if ``task``'s status in the session's plan is terminal.

    Reads status off the live ``session.plan`` entry (not the snapshot
    ``task`` object passed into ``invoke``) so transitions driven by
    reporting-tool handlers during the in-flight invocation are seen.
    Used as an early-exit optimization inside :meth:`ADKAdapter.invoke`
    — generator-end on the runner is the authoritative termination
    signal under the single-Runner model, but breaking early when the
    agent reports terminal avoids letting an otherwise-chatty agent
    keep driving LLM turns for a task it has already marked done.
    """
    task_id = getattr(task, "id", "") or ""
    if not task_id:
        return False
    plan = getattr(session, "plan", None)
    if plan is None:
        return False
    for live in getattr(plan, "tasks", ()) or ():
        if getattr(live, "id", None) == task_id:
            return getattr(live, "status", None) in TERMINAL_TASK_STATUSES
    return False


def _function_call_ids_in_event(event: Any) -> list[tuple[str, str]]:
    """Return ``(id, name)`` pairs for every function call in ``event``.

    Tolerates events that don't expose the helper method (fake events in
    tests) by walking ``event.content.parts`` directly.
    """
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        try:
            calls = getter() or []
        except Exception:  # noqa: BLE001
            calls = []
    else:
        calls = []
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or ():
            fc = getattr(part, "function_call", None)
            if fc is not None:
                calls.append(fc)
    out: list[tuple[str, str]] = []
    for fc in calls:
        fc_id = getattr(fc, "id", None)
        if fc_id:
            out.append((str(fc_id), str(getattr(fc, "name", "") or "")))
    return out


def _function_response_ids_in_event(event: Any) -> list[str]:
    """Return the ``function_response.id`` values in ``event``.

    Mirrors :func:`_function_call_ids_in_event` but for responses.
    """
    getter = getattr(event, "get_function_responses", None)
    if callable(getter):
        try:
            responses = getter() or []
        except Exception:  # noqa: BLE001
            responses = []
    else:
        responses = []
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or ():
            fr = getattr(part, "function_response", None)
            if fr is not None:
                responses.append(fr)
    out: list[str] = []
    for fr in responses:
        fr_id = getattr(fr, "id", None)
        if fr_id:
            out.append(str(fr_id))
    return out


# ---------------------------------------------------------------------------
# Symbolic reasons for mid-invocation tool-call cancellation.
#
# These are the semantic tags the synthetic ``function_response`` content is
# differentiated on. The content below each variant is written for the LLM
# that will read this history on the next turn — NOT for a human reading
# goldfive's logs. Qwen and similar small/quantized models choke on the
# prior "goldfive_cancelled / preserve session history well-formedness"
# phrasing and either retry the cancelled call or enter a reasoning loop
# trying to reconcile the jargon. See goldfive#139.
# ---------------------------------------------------------------------------

SYMBOLIC_REASON_USER_STEER = "user_steer"
"""Cancel was caused by a user steering command (USER_STEER drift)."""

SYMBOLIC_REASON_REPLAN = "replan"
"""Cancel was caused by goldfive updating the plan mid-flight."""

SYMBOLIC_REASON_ERROR = "error"
"""Cancel / heal was caused by an unexpected error / disconnect."""


_USER_STEER_RESPONSE_CONTENT: dict[str, str] = {
    "status": "cancelled_by_user_steering",
    "instruction": (
        "The user issued a steering command during this tool call. "
        "The prior task has been ABANDONED. Do NOT retry this call. "
        "Do NOT continue the prior plan. The next user message "
        "contains your new task — respond to it fresh."
    ),
}

_REPLAN_RESPONSE_CONTENT: dict[str, str] = {
    "status": "cancelled_by_replan",
    "instruction": (
        "The plan was updated by goldfive. This tool call is no longer "
        "part of the plan. Do NOT retry. Await the next task message."
    ),
}

_GENERIC_RESPONSE_CONTENT: dict[str, str] = {
    "status": "cancelled",
    "instruction": "This call was cancelled. Do NOT retry. Await next instruction.",
}


_USER_STEER_PRIMER_TEXT = (
    "\u26a0 STEERING NOTICE: Your prior task was cancelled by user steering. "
    "The task below supersedes all prior work. Do not retry or reference "
    "the cancelled tool call."
)


def _resolve_response_content(reason: str) -> dict[str, str]:
    """Map ``reason`` onto one of the three content variants.

    Unknown or legacy reasons (``"cancelled_mid_invocation"``,
    ``"error:<ExceptionName>"``, ``"unexpected_orphan_on_normal_exit"``)
    fall through to the generic bucket — the earlier callers didn't
    differentiate, and the content there is neutral and LLM-actionable.
    """
    if reason == SYMBOLIC_REASON_USER_STEER:
        return dict(_USER_STEER_RESPONSE_CONTENT)
    if reason == SYMBOLIC_REASON_REPLAN:
        return dict(_REPLAN_RESPONSE_CONTENT)
    return dict(_GENERIC_RESPONSE_CONTENT)


def _build_cancelled_response_event(
    *,
    function_call_id: str,
    tool_name: str,
    author: str,
    invocation_id: str,
    reason: str,
) -> Any:
    """Synthesize a ``function_response`` ADK :class:`Event` for ``function_call_id``.

    Used on mid-invocation cancel: ADK's session conversation history would
    otherwise contain an assistant event with a ``function_call`` that never
    receives a matching ``function_response``, which confuses subsequent LLM
    turns (the "Missing tool results for tool_call_id(s): [...]" symptom in
    driver logs). This builder produces a minimally-shaped response event
    that the next turn's request assembler will pair with the orphan call.

    The response payload is reason-differentiated (see goldfive#139): the
    prior generic "goldfive_cancelled" jargon caused Qwen and similar
    smaller models to either retry the cancelled call or enter a
    reasoning loop. The new content is short, model-actionable, and
    selected based on the symbolic ``reason`` (USER_STEER / REPLAN /
    generic) passed from the cancel trigger.
    """
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part.from_function_response(
        name=tool_name or "unknown_tool",
        response=_resolve_response_content(reason),
    )
    # Preserve the pairing so ADK's find_event_by_function_call_id
    # correctly matches this response to the orphan function_call.
    if part.function_response is not None:
        part.function_response.id = function_call_id

    content = types.Content(role="user", parts=[part])
    return Event(
        invocation_id=invocation_id or "",
        author=author or "user",
        content=content,
    )


def _build_user_steer_primer_event(
    *,
    invocation_id: str,
) -> Any:
    """Synthesize a user-role primer event reinforcing the USER_STEER pivot.

    Appended after the per-tool ``function_response`` heals when the
    cancel reason is ``user_steer``. Belt-and-braces: even if the LLM
    skims the function_response content, the subsequent user-role
    message with the ⚠ STEERING NOTICE is impossible to miss on the
    next turn. See goldfive#139.
    """
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    content = types.Content(
        role="user",
        parts=[types.Part(text=_USER_STEER_PRIMER_TEXT)],
    )
    return Event(
        invocation_id=invocation_id or "",
        author="user",
        content=content,
    )


def _new_message_parts(task: Task) -> Any:
    """DEPRECATED — kept for back-compat with the pre-overlay path.

    Equivalent to :func:`_follow_up_message_parts`. Under the
    overlay model (goldfive#141) primary dispatch happens via
    :func:`_passthrough_message_parts` against the user's original
    request; per-task nudges only fire for PENDING tasks the tree
    missed and use the gentler "Also, please" phrasing below.

    Callers that need the old jargon-heavy shape must retire —
    "Task: X. Use the goldfive.* session-state keys..." messages
    cause coordinator agents with flow-oriented prompts to treat
    every plan task as a new user request and run their full
    pipeline for each one. See goldfive#141 for the root cause.
    """
    return _follow_up_message_parts(task)


def _passthrough_message_parts(user_input: str) -> Any:
    """Build the ADK user ``Content`` for an overlay-model passthrough.

    Used by :meth:`ADKAdapter.invoke_passthrough` — sends the
    user's original request verbatim so the agent tree sees the
    same input plain ADK would. No goldfive jargon; no task
    framing. The tree runs naturally and goldfive observes via
    the plugin callbacks.
    """
    from google.genai.types import Content, Part  # type: ignore

    text = user_input or ""
    return Content(role="user", parts=[Part(text=text)])


def _follow_up_message_parts(task: Task) -> Any:
    """Build the gentle follow-up user turn for a missed plan task.

    Used by :meth:`ADKAdapter.invoke_follow_up` — fires only when
    the PlanReconciler determined the tree legitimately missed
    ``task`` during the passthrough invocation. Phrased as a
    natural follow-up ("Also, please: ...") so coordinator agents
    don't re-run their full pipeline; the agent treats it as a
    small additional request on top of the conversation history.

    No mention of "goldfive.*" session-state keys — users bring
    their own trees and the overlay model lets those trees run
    with their native prompts intact.
    """
    from google.genai.types import Content, Part  # type: ignore

    title = getattr(task, "title", "") or ""
    description = getattr(task, "description", "") or ""
    if title and description:
        body = f"Also, please: {title}. {description}"
    elif title:
        body = f"Also, please: {title}."
    else:
        body = f"Also, please: {description}" if description else "Also, please continue."
    return Content(role="user", parts=[Part(text=body)])


class ADKAdapter:
    """``AgentAdapter`` for Google's Agent Development Kit.

    Single-Runner model (goldfive#130). One ADK ``InMemoryRunner`` is
    built around the caller-supplied root agent (or the caller's own
    pre-built ``Runner`` in degraded mode). :meth:`invoke` drives that
    one runner for every task; delegation within the tree happens via
    ADK's native ``AgentTool`` / ``transfer_to_agent`` / ``sub_agents``
    mechanisms.

    Parameters
    ----------
    agent_or_runner:
        Either an ADK ``BaseAgent`` (in which case an ``InMemoryRunner``
        is constructed around it) or an already-built ``Runner``.
    user_id:
        Stable user id for ADK session lookup. Defaults to
        ``"goldfive_user"`` which is fine for local / single-user runs.
    session_id:
        Optional stable session id. When omitted the adapter mints a
        fresh session id on first :meth:`invoke`.
    app_name:
        Optional ADK app_name override. Defaults to the runner's own
        ``app_name`` or the agent's ``name``.
    plugins:
        Optional list of ADK ``BasePlugin`` instances to install on the
        runner. Observability plugins (e.g. ``HarmonografTelemetryPlugin``)
        are installed on the one runner; ADK propagates the plugin
        manager to any ``AgentTool``-spawned sub-Runner so delegation
        inherits the same plugin surface automatically.
    agent_tool_cap:
        Optional cap on the number of ``AgentTool`` spawns allowed in
        a single top-level invocation. Defaults to
        :data:`DEFAULT_AGENT_TOOL_CAP` (16). On exceed, the plugin
        emits a ``RUNAWAY_DELEGATION`` drift and cancels the
        invocation — belt-and-braces against runaway coordinators when
        a user-supplied prompt describes a pipeline rather than a
        task. Pass ``0`` or a negative value to disable.
    """

    def __init__(
        self,
        agent_or_runner: Any,
        *,
        user_id: str = "goldfive_user",
        session_id: str | None = None,
        app_name: str | None = None,
        plugins: list[Any] | None = None,
        agent_tool_cap: int | None = None,
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id
        self._degraded_prebuilt_runner = False
        # Caller-supplied ADK plugins (e.g. HarmonografTelemetryPlugin).
        # ADK's InMemoryRunner forwards its plugin manager into
        # AgentTool sub-Runners, so installing here covers the whole
        # tree for the single-Runner model.
        self._plugins: list[Any] = list(plugins) if plugins else []

        if _looks_like_runner(agent_or_runner):
            # Caller handed us an already-built Runner — use it verbatim.
            self._runner = agent_or_runner
            self._agent = getattr(agent_or_runner, "agent", None)
            if self._agent is None:
                raise ValueError("ADKAdapter: could not resolve an inner agent")
            self._degraded_prebuilt_runner = True
            log.debug(
                "ADKAdapter: caller passed a pre-built Runner; using it verbatim "
                "(degraded mode — goldfive-specific runner construction skipped)",
            )
        else:
            self._agent = agent_or_runner
            self._runner = _build_runner(agent_or_runner, plugins=self._plugins)

        self._app_name = (
            app_name
            or getattr(self._runner, "app_name", None)
            or getattr(self._agent, "name", None)
            or "goldfive"
        )

        host_agent_name = str(getattr(self._agent, "name", "") or "")
        cap = DEFAULT_AGENT_TOOL_CAP if agent_tool_cap is None else int(agent_tool_cap)
        self._plugin = make_adk_plugin(
            host_agent_name=host_agent_name,
            agent_tool_cap=cap,
        )
        self._agent_tool_cap = cap

        # Install the goldfive plugin on the one runner. ADK propagates
        # the plugin manager into any AgentTool-spawned sub-Runner so the
        # same ``_active_ctx`` is readable from nested invocations and
        # ``before_run_callback`` fires against every sub-session's live
        # state — which is what the state-protocol fix from #120 relies
        # on (and is preserved here).
        if not _register_plugin_on_runner(self._runner, self._plugin):
            log.warning(
                "ADKAdapter: could not attach plugin to runner — "
                "reporting callbacks will be inactive",
            )

        # Collect the names of every reachable agent (via sub_agents /
        # inner_agent / AgentTool.agent edges). Exposed as
        # :attr:`available_agents` so the planner can see the
        # delegation targets available in the tree. Not used for
        # dispatch — the adapter drives the root runner and ADK handles
        # delegation.
        if self._degraded_prebuilt_runner:
            # With a pre-built runner we only know about the root agent.
            root_name = str(getattr(self._agent, "name", "") or host_agent_name or "goldfive")
            self._available_agents: list[str] = [root_name]
            self._available_agents_tree: list[dict[str, Any]] = [
                {
                    "name": root_name,
                    "depth": 0,
                    "parent": "",
                    "role": "root",
                    "kind": type(self._agent).__name__ if self._agent is not None else "BaseAgent",
                }
            ]
        else:
            self._available_agents = _collect_reachable_agent_names(self._agent)
            self._available_agents_tree = _collect_reachable_agent_tree(self._agent)

        # Auto-attach GoldfivePlanner to every LlmAgent in the tree
        # (goldfive#153). Idempotent: skipped for agents already
        # carrying a GoldfivePlanner, and for agents opting out via
        # the ``_goldfive_planner_opt_out = True`` marker. A pre-existing
        # user-supplied planner is COMPOSED (wrapped as
        # ``user_planner=...``) rather than replaced. Skipped in
        # degraded mode because the caller has taken over runner
        # construction and may have its own planner conventions we
        # don't want to stomp on.
        if not self._degraded_prebuilt_runner:
            _attach_goldfive_planner_to_tree(self._agent)

        # tool_name -> handler. Populated by register_reporting_tools.
        self._tool_handlers: dict[str, Any] = {}
        # Full reporting-tool specs, in registration order. The plugin's
        # ``before_tool_callback`` routes each call through
        # :func:`goldfive.adapters._tool_invocation.invoke_tool` so the
        # terminal-rejection / idempotency / loop-guard layers fire.
        self._tool_specs: list[ReportingToolSpec] = []
        # The current Steerer. Set by bind_steerer() before invoke().
        self._steerer: Steerer | None = None
        # Pending ADK function_call ids observed in the current invoke()'s
        # event stream that have not yet received a matching
        # function_response. Maintained so _heal_pending_tool_calls can
        # synthesize "cancelled" responses on mid-invocation cancel /
        # unexpected exception and keep the ADK session's conversation
        # history well-formed. Paired with _pending_tool_call_names so
        # the synthetic response can name its tool.
        self._pending_tool_call_ids: set[str] = set()
        self._pending_tool_call_names: dict[str, str] = {}
        # Short-lived tag for the NEXT mid-invocation cancel. Set by the
        # Steerer (on USER_STEER drift) or by refine-triggered paths
        # BEFORE they cancel the in-flight invoke so
        # _heal_pending_tool_calls knows which content variant to emit
        # in the synthetic function_response. Cleared after consumption
        # so a stale tag can't bleed into the next cancel. See
        # goldfive#139 and
        # :func:`_build_cancelled_response_event` for the content map.
        self._next_cancel_reason: str = ""
        # Outer session id pinned by :class:`GoldfiveADKAgent` when the
        # adapter runs inside adk-web. ``None`` for programmatic callers
        # and test harnesses; the adapter falls back to the lazy-uuid
        # mint in :meth:`_ensure_session`. Live tests in
        # tests/test_live_steering_e2e.py set this explicitly.
        self._outer_session_id: str | None = None

        # Wrap-time integrity check: the one runner must carry the
        # goldfive plugin. In degraded mode we skip because the caller
        # may have constructed a runner shape we don't fully control.
        if not self._degraded_prebuilt_runner:
            plugin_name = getattr(self._plugin, "name", "")
            installed = list(getattr(getattr(self._runner, "plugin_manager", None), "plugins", []))
            if not any(getattr(p, "name", "") == plugin_name for p in installed):
                raise RuntimeError(
                    f"ADKAdapter: goldfive plugin {plugin_name!r} failed to "
                    f"install on the runner — reporting callbacks, "
                    f"state-protocol writes, and drift observation would all "
                    f"be broken"
                )

    # ------------------------------------------------------------------
    # Post-construction plugin install
    # ------------------------------------------------------------------

    def add_plugin(self, plugin: Any) -> None:
        """Install an ADK ``BasePlugin`` on the runner.

        Under the single-Runner model there is one runner; caller
        plugins install once and ADK propagates the plugin manager into
        any ``AgentTool``-spawned sub-Runner automatically. Tolerant of
        runners that don't expose a plugin manager — logs at DEBUG and
        continues.
        """
        if not _register_plugin_on_runner(self._runner, plugin):
            log.debug(
                "ADKAdapter.add_plugin: runner has no plugin manager; plugin %r not installed",
                type(plugin).__name__,
            )

    # ------------------------------------------------------------------
    # AgentAdapter protocol
    # ------------------------------------------------------------------

    @property
    def available_agents(self) -> list[str]:
        """Return the sorted names of agents reachable from the root agent.

        Reports the names the planner may reference in
        ``task.assignee_agent_id`` for observability and for the
        planner's own delegation hints. Under the single-Runner model
        the field is advisory — goldfive drives the root runner and
        ADK handles delegation through its native mechanisms
        (AgentTool, transfer_to_agent, sub_agents).
        """
        return list(self._available_agents)

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        """Return structured metadata for every reachable agent in the tree.

        Each entry is a dict with the keys ``name`` (str), ``depth``
        (int; 0 = root), ``parent`` (str; empty for root), ``role``
        (``"root"`` | ``"intermediate"`` | ``"leaf"``), and ``kind``
        (the ADK class name — ``LlmAgent``, ``SequentialAgent``,
        ``ParallelAgent``, ``BaseAgent``, or any custom subclass name;
        tree-agnostic — no semantic interpretation).

        Used by :class:`goldfive.planner.LLMPlanner` (goldfive#151) to
        render an "AGENT TREE" section in planner prompts and to
        validate that every task's ``assignee_agent_id`` is actually
        reachable in the wrapped tree. A fresh shallow copy is returned
        on each access so callers cannot mutate the cached list in
        place.
        """
        return [dict(entry) for entry in self._available_agents_tree]

    async def register_reporting_tools(self, tools: list[ReportingToolSpec]) -> None:
        """Register goldfive reporting tools with the wrapped agent tree.

        Each spec is wrapped as a ``google.adk.tools.FunctionTool`` and
        attached to the root agent plus every sub-agent reachable via
        :func:`_augment_subtree_with_reporting`. Coverage across the
        whole tree matters because an AgentTool sub-invocation can
        itself report terminal status for the outer task — so every
        reachable agent needs the reporting tools available regardless
        of which one drives each turn.
        """
        if not tools:
            return
        function_tools: list[Any] = []
        names: set[str] = set()
        for spec in tools:
            name = getattr(spec, "name", "")
            if not isinstance(name, str) or not name:
                raise ValueError(f"ReportingToolSpec has no name: {spec!r}")
            handler = getattr(spec, "handler", None)
            if handler is None:
                raise ValueError(f"ReportingToolSpec '{name}' missing handler")
            self._tool_handlers[name] = handler
            function_tools.append(_build_function_tool(spec))
            names.add(name)
        # Replace the spec list on every call so repeated registrations
        # (re-bind on a new run) stay consistent with the handler map.
        self._tool_specs = list(tools)

        # Attach to the root agent itself, then the whole subtree.
        root_tools = getattr(self._agent, "tools", None)
        if root_tools is None:
            try:
                self._agent.tools = list(function_tools)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "could not set tools on root agent %s: %s",
                    getattr(self._agent, "name", "?"),
                    exc,
                )
        else:
            existing_names: set[str] = set()
            for t in root_tools:
                n = getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", None)
                if n:
                    existing_names.add(str(n))
            if not any(n in existing_names for n in names):
                try:
                    self._agent.tools = list(root_tools) + list(function_tools)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "could not augment tools on root agent %s: %s",
                        getattr(self._agent, "name", "?"),
                        exc,
                    )

        _augment_subtree_with_reporting(self._agent, function_tools, names)

        # Integrity: every reachable agent must carry the reporting tool
        # names we just registered. A partial augmentation leaves a
        # sub-agent with no way to report terminal status back — if a
        # coordinator delegates to it via AgentTool, the outer task
        # cannot finish via the early-exit optimization. Degraded mode
        # (pre-built Runner) skips this because we only see the root.
        if not self._degraded_prebuilt_runner and names:
            missing: list[tuple[str, set[str]]] = []
            seen: set[int] = set()
            stack: list[Any] = [self._agent]
            while stack:
                cur = stack.pop()
                if cur is None or id(cur) in seen:
                    continue
                seen.add(id(cur))
                agent_name = str(getattr(cur, "name", "") or "")
                agent_tools = list(getattr(cur, "tools", None) or [])
                if agent_tools:
                    tool_names_on_agent: set[str] = set()
                    for t in agent_tools:
                        n = getattr(t, "name", None) or getattr(
                            getattr(t, "func", None), "__name__", None
                        )
                        if n:
                            tool_names_on_agent.add(str(n))
                    gap = names - tool_names_on_agent
                    if gap and agent_name:
                        missing.append((agent_name, gap))
                for sub in getattr(cur, "sub_agents", None) or ():
                    stack.append(sub)
                inner = getattr(cur, "inner_agent", None)
                if inner is not None:
                    stack.append(inner)
                for t in getattr(cur, "tools", None) or ():
                    nested = getattr(t, "agent", None)
                    if nested is not None:
                        stack.append(nested)
            if missing:
                details = ", ".join(
                    f"{name}: missing {sorted(gap)}" for name, gap in sorted(missing)
                )
                raise RuntimeError(
                    f"ADKAdapter: reporting-tool set did not land on "
                    f"{len(missing)} reachable agent(s): {details}. "
                    f"Expected every reachable agent to carry the "
                    f"reporting tools so terminal status can flow back "
                    f"from an AgentTool sub-invocation. See "
                    f"_augment_subtree_with_reporting."
                )

    def bind_steerer(self, steerer: Steerer | None) -> None:
        """Attach the active :class:`~goldfive.protocols.Steerer`.

        Called by the executor before :meth:`invoke` so plugin callbacks
        can route drift observations. Safe to call with ``None`` to
        unbind.
        """
        self._steerer = steerer

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",  # noqa: ARG002 -- part of the protocol
    ) -> None:
        """Forward an extracted reasoning block to the bound steerer."""
        steerer = self._steerer
        if steerer is None or not text:
            return
        observe = getattr(steerer, "observe_reasoning", None)
        if observe is None:
            return
        await observe(text, task=task, session=session, provider=provider)

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        """DEPRECATED — drive one ADK turn for a single ``task``.

        Retained for back-compat with legacy executors and tests
        that predate the goldfive#141 overlay refactor. New code
        should call :meth:`invoke_passthrough` for the initial
        invocation and :meth:`invoke_follow_up` for soft per-task
        nudges when the reconciler detects missed work.

        This method now uses the gentler "Also, please: {title}."
        phrasing (see :func:`_follow_up_message_parts`) rather than
        the jargon-heavy "Task: X. Use the goldfive.* session-state
        keys ..." shape that caused coordinator agents with flow
        prompts to treat every plan task as a full new pipeline
        trigger. See goldfive#141 for the root cause.
        """
        return await self._invoke_internal(
            task=task,
            session=session,
            new_message=_follow_up_message_parts(task),
            reconciler=None,
        )

    async def invoke_passthrough(
        self,
        user_message: str,
        *,
        session: Session,
        reconciler: Any = None,
        ctx: Any = None,  # noqa: ARG002 -- reserved for future invocation-context plumbing
    ) -> InvocationResult:
        """Drive ONE ADK turn with the user's original request (overlay path).

        Primary overlay-model dispatch (goldfive#141). Sends
        ``user_message`` verbatim — no task framing, no goldfive
        jargon — so coordinator agents with flow-oriented prompts
        run their natural pipeline exactly as they would under
        plain ADK. Observation happens via the plugin's callback
        surface and, when supplied, a :class:`PlanReconciler` maps
        observed agent turns back to plan-task transitions.

        The ``task`` field on :class:`SessionContext` is ``None``
        for this path — there is no single "current task" during
        a passthrough invocation. The plugin's state-protocol writes
        still run (``before_run_callback``) but write no current-task
        metadata; reporting tools, if called, route through the
        reconciler's observation pipeline instead of per-task
        attribution.

        Returns an :class:`InvocationResult` whose ``task_id`` is
        empty (no single task owns the whole invocation) and
        ``text`` is the final assistant text.
        """
        return await self._invoke_internal(
            task=None,
            session=session,
            new_message=_passthrough_message_parts(user_message),
            reconciler=reconciler,
        )

    async def invoke_follow_up(self, task: Task, session: Session) -> InvocationResult:
        """Drive a gentle follow-up for a plan ``task`` missed during passthrough.

        Used by the overlay-model executor (goldfive#141) after
        :meth:`invoke_passthrough` finishes and the
        :class:`PlanReconciler` reports PENDING tasks the tree did
        not exercise. Sends a natural-language "Also, please: ..."
        user turn on top of the existing conversation so the tree
        picks up the missed work without re-running its full
        pipeline.
        """
        return await self._invoke_internal(
            task=task,
            session=session,
            new_message=_follow_up_message_parts(task),
            reconciler=None,
        )

    async def _invoke_internal(
        self,
        *,
        task: Task | None,
        session: Session,
        new_message: Any,
        reconciler: Any = None,
    ) -> InvocationResult:
        """Shared driver behind :meth:`invoke`, :meth:`invoke_passthrough`,
        and :meth:`invoke_follow_up`.

        Handles session creation, plugin context install, the
        ``runner.run_async`` event loop, pending-tool-call healing,
        and the plugin-context release. The only inputs that differ
        between the three public entry points are:

        * ``task`` — the current-task pin or ``None`` for passthrough
        * ``new_message`` — the pre-built ADK ``Content`` body
        * ``reconciler`` — overlay-mode reconciler or ``None``

        Preserves every behaviour the legacy :meth:`invoke` exposed:

        * ``_task_is_terminal(task, session)`` early-break (skipped
          when ``task`` is ``None``)
        * ``_is_final_event(event)`` check
        * runaway-delegation cap short-circuit
        * ``_heal_pending_tool_calls`` on cancel / error / orphaned
          normal exit
        """
        task_id = getattr(task, "id", "") if task is not None else ""

        session_id = await self._ensure_session()
        state = await self._get_session_state(session_id)

        ctx = SessionContext(
            session=session,
            steerer=self._steerer,
            task=task,
            tool_handlers=self._tool_handlers,
            tools=self._tool_specs,
            host_agent_name=str(getattr(self._agent, "name", "") or ""),
        )

        # Hand the per-invocation context to the goldfive plugin. This
        # is the AUTHORITATIVE handoff — the plugin's callbacks read the
        # active context off its own instance, not from ADK session
        # state. The state-protocol write (run_id, plan, current task,
        # tools) happens inside the plugin's ``before_run_callback``
        # against the LIVE invocation session — which is the #120
        # state-protocol fix, preserved here.
        if self._plugin is not None:
            self._plugin.set_active_context(ctx)
            # Overlay path: attach the PlanReconciler so the plugin
            # forwards before/after_agent observations. Cleared in
            # the ``finally`` block via clear_active_context.
            if reconciler is not None:
                set_rec = getattr(self._plugin, "set_reconciler", None)
                if callable(set_rec):
                    set_rec(reconciler)

        # Rebind every GoldfivePlanner in the tree to the live
        # collaborators for this invocation (goldfive#153). Cheap walk
        # — the tree is small and this lets the structural filters
        # emit PLAN_DIVERGENCE drifts through the bound steerer.
        _rebind_goldfive_planners(
            self._agent,
            agent_registry=list(self._available_agents),
            steerer=self._steerer,
            session=session,
        )

        # Mirror into ADK state as a best-effort fallback for legacy
        # unit tests that construct a plain ``tool_context`` holding a
        # populated state dict and drive the plugin directly. Live-run
        # path does not depend on this write succeeding and will re-seed
        # via before_run_callback regardless.
        try:
            state[SESSION_CONTEXT_STATE_KEY] = ctx  # type: ignore[index]
        except Exception:
            log.debug("ADKAdapter.invoke: could not stash session context")

        final_text = ""
        stop_reason = "completed"
        err: Exception | None = None
        last_event: Any = None
        last_invocation_id = ""
        # Reset per-invocation pending-id bookkeeping.
        self._pending_tool_call_ids.clear()
        self._pending_tool_call_names.clear()
        was_cancelled = False
        try:
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                last_event = event
                inv_id = getattr(event, "invocation_id", "") or ""
                if inv_id:
                    last_invocation_id = inv_id
                # Track outstanding function_call ids so we can heal
                # history on mid-invocation cancel (see _heal_pending_tool_calls).
                for fc_id, fc_name in _function_call_ids_in_event(event):
                    self._pending_tool_call_ids.add(fc_id)
                    if fc_name:
                        self._pending_tool_call_names[fc_id] = fc_name
                for fr_id in _function_response_ids_in_event(event):
                    self._pending_tool_call_ids.discard(fr_id)
                    self._pending_tool_call_names.pop(fr_id, None)
                # Runaway-delegation cap: the plugin counts AgentTool
                # spawns on the current top-level invocation. On exceed
                # it requests a cancel; detect that signal here so we
                # break cleanly (the plugin has already emitted the
                # RUNAWAY_DELEGATION drift).
                if self._plugin is not None and self._plugin.runaway_delegation_tripped:
                    stop_reason = "runaway_delegation"
                    break
                text = _extract_text_from_event(event)
                if text:
                    final_text = text
                if _is_final_event(event):
                    stop_reason = "final_response"
                # Early termination when the agent has reported this
                # task as terminal via a reporting tool. Skipped when
                # ``task`` is None (passthrough path has no single
                # "current task" to break on).
                if task is not None and _task_is_terminal(task, session):
                    stop_reason = "task_terminal"
                    break
        except asyncio.CancelledError:
            was_cancelled = True
            stop_reason = "cancelled"
            # Consume the tag the steerer / executor stashed on us before
            # triggering the cancel (see goldfive#139). An unset tag
            # falls through to the legacy generic reason so existing
            # heal paths keep the neutral content variant.
            reason = self._next_cancel_reason or "cancelled_mid_invocation"
            self._next_cancel_reason = ""
            await self._heal_pending_tool_calls(
                runner=self._runner,
                session_id=session_id,
                invocation_id=last_invocation_id,
                reason=reason,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            err = exc
            stop_reason = f"error:{type(exc).__name__}"
            log.debug("ADKAdapter.invoke: runner.run_async raised: %s", exc)
            await self._heal_pending_tool_calls(
                runner=self._runner,
                session_id=session_id,
                invocation_id=last_invocation_id,
                reason=f"error:{type(exc).__name__}",
            )
        finally:
            if not was_cancelled:
                if self._pending_tool_call_ids:
                    log.warning(
                        "ADKAdapter.invoke: %d function_call id(s) ended without "
                        "responses on normal exit; healing session history",
                        len(self._pending_tool_call_ids),
                    )
                    await self._heal_pending_tool_calls(
                        runner=self._runner,
                        session_id=session_id,
                        invocation_id=last_invocation_id,
                        reason="unexpected_orphan_on_normal_exit",
                    )
                # No cancel fired — drop any stale tag so the NEXT
                # invoke's cancel (if any) doesn't pick up leftover state.
                self._next_cancel_reason = ""
            if self._plugin is not None:
                # ``clear_active_context`` also clears any attached
                # reconciler — overlay-mode is strictly per-invocation.
                self._plugin.clear_active_context()
            if isinstance(state, Mapping):
                try:
                    state.pop(SESSION_CONTEXT_STATE_KEY, None)  # type: ignore[attr-defined]
                except Exception:
                    pass

        return InvocationResult(
            task_id=task_id,
            text=final_text,
            stop_reason=stop_reason,
            error=err,
            raw=last_event,
        )

    # ------------------------------------------------------------------
    # Session plumbing
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> str:
        """Return the ADK session id for the runner.

        Mints one lazily on first call when the caller didn't pin one
        via ``session_id=``. Cached on ``self._session_id`` after the
        first call so every subsequent :meth:`invoke` reuses it — the
        whole run rolls up under one logical session.
        """
        if self._session_id:
            # Ensure the session exists on the runner's service. Safe to
            # call repeatedly; create_session is typically idempotent
            # for a given (app_name, user_id, session_id) triple and we
            # swallow any conflict so tests that pre-create the session
            # keep working.
            await self._touch_session(self._session_id)
            return self._session_id

        self._session_id = str(uuid.uuid4())
        await self._touch_session(self._session_id)
        return self._session_id

    async def _touch_session(self, session_id: str) -> None:
        """Best-effort ``create_session`` on the runner's session service.

        Idempotent: swallow exceptions so repeated calls against a
        pre-existing session don't raise. Session services that don't
        expose ``create_session`` (e.g. some test stubs) are skipped.
        """
        session_service = getattr(self._runner, "session_service", None)
        if session_service is None:
            return
        create = getattr(session_service, "create_session", None)
        if not callable(create):
            return
        app_name = str(getattr(self._runner, "app_name", "") or "") or self._app_name
        try:
            coro = create(
                app_name=app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            if hasattr(coro, "__await__"):
                await coro
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "ADKAdapter._touch_session: create_session raised: %s",
                exc,
            )

    async def _heal_pending_tool_calls(
        self,
        *,
        runner: Any,
        session_id: str,
        invocation_id: str,
        reason: str,
    ) -> None:
        """Append synthetic ``function_response`` events for orphan tool calls.

        Called from :meth:`invoke`'s cancel / exception paths. For every
        ``function_call`` id still pending in :attr:`_pending_tool_call_ids`,
        build a matching response event (see
        :func:`_build_cancelled_response_event`) and append it to the ADK
        session via ``session_service.append_event``. Best-effort: logs and
        swallows individual failures so healing one orphan doesn't block
        the others.
        """
        if not self._pending_tool_call_ids:
            return

        session_service = getattr(runner, "session_service", None)
        if session_service is None:
            log.debug(
                "ADKAdapter._heal_pending_tool_calls: runner has no "
                "session_service; cannot heal %d orphan tool call(s)",
                len(self._pending_tool_call_ids),
            )
            self._pending_tool_call_ids.clear()
            self._pending_tool_call_names.clear()
            return

        append = getattr(session_service, "append_event", None)
        get = getattr(session_service, "get_session", None)
        if not callable(append) or not callable(get):
            log.debug(
                "ADKAdapter._heal_pending_tool_calls: session_service lacks "
                "append_event/get_session; cannot heal %d orphan tool call(s)",
                len(self._pending_tool_call_ids),
            )
            self._pending_tool_call_ids.clear()
            self._pending_tool_call_names.clear()
            return

        app_name = str(getattr(runner, "app_name", "") or "") or self._app_name
        try:
            coro = get(
                app_name=app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            adk_session = await coro if hasattr(coro, "__await__") else coro
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "ADKAdapter._heal_pending_tool_calls: get_session raised: %s",
                exc,
            )
            self._pending_tool_call_ids.clear()
            self._pending_tool_call_names.clear()
            return

        if adk_session is None:
            self._pending_tool_call_ids.clear()
            self._pending_tool_call_names.clear()
            return

        host_author = str(getattr(self._agent, "name", "") or "") or "user"
        pending_ids = sorted(self._pending_tool_call_ids)
        healed = 0
        for fc_id in pending_ids:
            tool_name = self._pending_tool_call_names.get(fc_id, "")
            try:
                synth = _build_cancelled_response_event(
                    function_call_id=fc_id,
                    tool_name=tool_name,
                    author=host_author,
                    invocation_id=invocation_id,
                    reason=reason,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "ADKAdapter._heal_pending_tool_calls: could not build "
                    "synthetic event for %s: %s",
                    fc_id,
                    exc,
                )
                continue
            try:
                coro = append(session=adk_session, event=synth)
                if hasattr(coro, "__await__"):
                    await coro
                healed += 1
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "ADKAdapter._heal_pending_tool_calls: append_event for %s raised: %s",
                    fc_id,
                    exc,
                )

        # After the per-call function_response heals, append a user-role
        # primer event for USER_STEER cancels only. Belt-and-braces so
        # the next LLM turn sees the pivot reinforced even if the model
        # skims the function_response content. REPLAN and generic
        # cancels keep the function_response content as the sole signal
        # — those pivots are less jolting and don't need the extra
        # framing (see goldfive#139).
        if healed and reason == SYMBOLIC_REASON_USER_STEER:
            try:
                primer = _build_user_steer_primer_event(invocation_id=invocation_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "ADKAdapter._heal_pending_tool_calls: could not build "
                    "user_steer primer event: %s",
                    exc,
                )
            else:
                try:
                    coro = append(session=adk_session, event=primer)
                    if hasattr(coro, "__await__"):
                        await coro
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "ADKAdapter._heal_pending_tool_calls: append_event "
                        "for user_steer primer raised: %s",
                        exc,
                    )

        if healed:
            log.info(
                "goldfive ADKAdapter: healed %d orphan tool_call_id(s) after %s (pending=%s)",
                healed,
                reason,
                pending_ids,
            )

        self._pending_tool_call_ids.clear()
        self._pending_tool_call_names.clear()

    async def _get_session_state(self, session_id: str) -> Any:
        """Fetch the ADK session's state dict for the runner."""
        session_service = getattr(self._runner, "session_service", None)
        if session_service is None:
            return {}
        get = getattr(session_service, "get_session", None)
        if not callable(get):
            return {}
        app_name = str(getattr(self._runner, "app_name", "") or "") or self._app_name
        try:
            coro = get(
                app_name=app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            if hasattr(coro, "__await__"):
                session = await coro
            else:
                session = coro
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "ADKAdapter._get_session_state: get_session raised: %s",
                exc,
            )
            return {}
        state = getattr(session, "state", None)
        if state is None:
            return {}
        return state
