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
import inspect
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

    The returned function's ``__signature__`` is set explicitly so ADK's
    :class:`FunctionTool` introspection generates a declaration with
    only the LLM-author parameters — ``task_id`` is deliberately
    omitted (goldfive#241). The plugin's ``before_tool_callback`` fills
    ``task_id`` in from session state before the handler runs; exposing
    it in the schema led LLMs to either hallucinate bad values or
    abandon the reporting protocol entirely (live evidence: "the
    function is still trying to use a task_id parameter even though
    the schema says it doesn't require any parameters").
    """

    def _shim(**kwargs: Any) -> dict[str, Any]:
        return {"acknowledged": True}

    _shim.__name__ = name
    _shim.__qualname__ = name
    _shim.__doc__ = description or f"Reporting tool: {name}"
    return _shim


# Declared-for-LLM signatures for the built-in reporting tools
# (goldfive#241 Item 3). Keys are the canonical tool names from
# :mod:`goldfive.reporting`; values are the list of :class:`inspect.Parameter`
# entries the ADK FunctionTool will expose to the model. ``task_id`` is
# deliberately absent from every entry — the plugin's
# ``before_tool_callback`` resolves it from session state, so the
# model never sees the field in the tool declaration and cannot
# abandon the protocol over optional-arg confusion.
#
# Callers that register custom reporting tools via
# :meth:`ADKAdapter.register_reporting_tools` fall back to the legacy
# ``**kwargs`` signature if their tool name isn't in this map; the
# injection path in the plugin still hides ``task_id`` from them by
# stamping it from state before the handler runs.
def _reporting_tool_signatures() -> dict[str, list[inspect.Parameter]]:
    P = inspect.Parameter
    return {
        "report_task_started": [
            P("detail", P.KEYWORD_ONLY, default="", annotation=str),
        ],
        "report_task_progress": [
            P("detail", P.KEYWORD_ONLY, default="", annotation=str),
            P("fraction", P.KEYWORD_ONLY, default=None, annotation=float | None),
        ],
        "report_task_completed": [
            P("summary", P.KEYWORD_ONLY, default="", annotation=str),
        ],
        "report_task_failed": [
            P("reason", P.KEYWORD_ONLY, default="", annotation=str),
            P("recoverable", P.KEYWORD_ONLY, default=None, annotation=bool | None),
        ],
        "report_task_blocked": [
            P("blocker", P.KEYWORD_ONLY, default="", annotation=str),
            P("needed", P.KEYWORD_ONLY, default="", annotation=str),
        ],
    }


def _apply_llm_signature(shim: Any, tool_name: str) -> None:
    """Attach a restricted ``__signature__`` to ``shim`` so ADK's
    FunctionTool builds a declaration that hides ``task_id``.

    No-op when ``tool_name`` isn't one of the built-ins — custom
    reporting tools keep the permissive ``**kwargs`` signature they
    had pre-#241 (the plugin's task_id injection still runs for them).
    """
    params_map = _reporting_tool_signatures()
    params = params_map.get(tool_name)
    if params is None:
        return
    try:
        shim.__signature__ = inspect.Signature(parameters=params)
    except (TypeError, ValueError) as exc:
        log.debug(
            "_apply_llm_signature: could not attach signature for %s: %s",
            tool_name,
            exc,
        )


def _build_function_tool(spec: ReportingToolSpec) -> Any:
    """Wrap a :class:`ReportingToolSpec` as a ``google.adk.tools.FunctionTool``."""
    from google.adk.tools import FunctionTool  # type: ignore

    shim = _build_ack_shim(spec.name, spec.description)
    _apply_llm_signature(shim, spec.name)
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


def _plugin_already_installed(runner: Any, plugin_name: str) -> bool:
    """Return True when a plugin of ``plugin_name`` is already on ``runner``.

    Under ``App(plugins=[...])`` the ADK runner already carries the
    caller-supplied plugin list by the time goldfive's wrap path tries
    to install anything. Asking the runner before appending is cheap
    (a walk over a short list) and prevents the duplicate-registration
    cascade that caused goldfive #166 (every harmonograf span appearing
    twice).

    Tolerates missing plugin managers and unusual ``plugins`` shapes —
    returns ``False`` on any lookup error so the caller falls through to
    the normal install path without masking real errors.
    """
    if runner is None or not plugin_name:
        return False
    pm = getattr(runner, "plugin_manager", None)
    plugins = getattr(pm, "plugins", None) if pm is not None else None
    if plugins is None:
        plugins = getattr(runner, "plugins", None)
    if not isinstance(plugins, list):
        return False
    return any(getattr(p, "name", None) == plugin_name for p in plugins)


def _register_plugin_on_runner(runner: Any, plugin: Any) -> bool:
    """Install ``plugin`` on ``runner``'s plugin manager if one exists.

    Tolerates both ``runner.plugin_manager.register(plugin)`` and the
    newer ``runner.plugins.append(plugin)`` shapes. Returns True if the
    plugin was installed (or was already present under the same name).

    Idempotent on plugin ``name`` (goldfive #166). When the runner
    already carries a plugin with the same ``name``, returns ``True``
    without appending a second instance — ADK's own ``register_plugin``
    raises on duplicate names and the silent-append fallback path we
    used to keep as a last resort was precisely what let the duplicate
    slip through. The dedup is primary; the fallback is belt-and-braces
    and preserved only for runner shapes without a working
    ``register_plugin``.
    """
    if runner is None:
        return False
    plugin_name = getattr(plugin, "name", None)
    if plugin_name and _plugin_already_installed(runner, plugin_name):
        log.info(
            "goldfive.adapters.adk: plugin %r already installed on runner; "
            "skipping duplicate install (see goldfive #166)",
            plugin_name,
        )
        return True
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


def _dedupe_plugins_by_name(plugins: list[Any]) -> list[Any]:
    """Return ``plugins`` with later same-``name`` instances removed.

    Caller-supplied plugin lists can land duplicates in subtle ways
    (``App(plugins=[p])`` followed by ``observe()`` or ``add_plugin(p)``
    both referencing the same plugin class). ADK's own
    :class:`PluginManager.register_plugin` raises on same-name collisions
    with a terse ``ValueError`` that is easy to miss in a stack trace —
    we'd rather normalise the list up front so every downstream install
    path sees a single instance per name.

    Preserves order: the first plugin at each name wins. Plugins without
    a ``name`` attribute pass through unfiltered (they can't collide by
    name). See goldfive #166.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for plugin in plugins:
        name = getattr(plugin, "name", None)
        if isinstance(name, str) and name:
            if name in seen:
                log.info(
                    "goldfive.adapters.adk: dropping duplicate plugin %r from "
                    "caller-supplied plugins (first instance wins; see goldfive #166)",
                    name,
                )
                continue
            seen.add(name)
        out.append(plugin)
    return out


def _build_runner(agent: Any, plugins: list[Any] | None = None) -> Any:
    """Construct an ADK ``InMemoryRunner`` around a ``BaseAgent``.

    Used by :class:`ADKAdapter` when the caller passes an agent rather
    than a runner. ``app_name`` defaults to the agent's name for tidy
    session-service bookkeeping.

    When ``plugins`` is a non-empty iterable, the plugin list is forwarded
    to :class:`InMemoryRunner` via its ``plugins=`` kwarg. Duplicates by
    ``name`` are collapsed via :func:`_dedupe_plugins_by_name` so ADK's
    ``PluginManager.register_plugin`` (which raises on same-name
    collisions) never sees them. Under the single-Runner model there is
    exactly one runner, and ADK propagates its plugin manager into any
    AgentTool-spawned sub-Runners automatically — so installing the
    plugin here is sufficient for the whole tree.
    """
    from google.adk.runners import InMemoryRunner  # type: ignore

    app_name = str(getattr(agent, "name", "") or "goldfive")
    kwargs: dict[str, Any] = {"agent": agent, "app_name": app_name}
    if plugins:
        kwargs["plugins"] = _dedupe_plugins_by_name(list(plugins))
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
        llm_call_timeout_ms: int | None = None,
    ) -> None:
        self._user_id = user_id
        # Per-(goldfive session.conversation_id) cached ADK session id
        # for routing under concurrent goldfive sessions on one adapter
        # (PR #294 audit / goldfive#271 follow-up). The legacy single
        # ``_session_id`` field stayed shared across every goldfive
        # session that ever ran on this adapter, so a second concurrent
        # invocation could pick up the first's cached id and target the
        # wrong ADK session history. The dict is keyed by
        # ``Session.conversation_id`` (stable across turns of one
        # Conversation, unique across concurrent Conversations) so each
        # logical conversation gets its own ADK session id while
        # multi-turn continuity within one conversation still works.
        # The legacy ``_session_id`` attribute survives as a property
        # backed by ``__legacy_session_id`` so callers that pin a
        # constructor-supplied id (and the back-compat
        # ``_pin_outer_session_on_adapter`` write) still observe the
        # historical single-session shape.
        self._adk_session_ids_by_conv: dict[str, str] = {}
        self.__legacy_session_id: str | None = session_id
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
        # Per-LLM-call wall-clock budget (goldfive#271 follow-up). When
        # the caller leaves ``llm_call_timeout_ms`` unset, the plugin
        # uses its module-level default
        # (:data:`goldfive.adapters._adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS`).
        # Pass ``0`` or a negative int to disable the watcher.
        plugin_kwargs: dict[str, Any] = {
            "host_agent_name": host_agent_name,
            "agent_tool_cap": cap,
        }
        if llm_call_timeout_ms is not None:
            plugin_kwargs["llm_call_timeout_ms"] = int(llm_call_timeout_ms)
        self._plugin = make_adk_plugin(**plugin_kwargs)
        self._agent_tool_cap = cap
        self._llm_call_timeout_ms = llm_call_timeout_ms

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
        #
        # PR #301 follow-up (goldfive#271): keyed by goldfive
        # ``Session.id`` because one adapter is shared across every
        # goldfive session driven by a :class:`Runner` — same hazard
        # PR #301 fixed for ``_next_cancel_reason`` /
        # ``_session_id`` / ``_outer_session_id``. Two concurrent
        # invocations on different sessions used to share the same
        # bare ``set`` / ``dict``, so session A's mid-invocation
        # cancel could heal session B's still-pending function_call
        # ids (and vice versa), corrupting both ADK sessions'
        # function-call/response pairing. The legacy bare attributes
        # survive as ``@property`` shims over the empty-key (``""``)
        # bucket so single-session callers and tests that read /
        # write ``adapter._pending_tool_call_ids`` directly keep
        # working.
        self._pending_tool_call_ids_by_session: dict[str, set[str]] = {}
        self._pending_tool_call_names_by_session: dict[str, dict[str, str]] = {}
        # Short-lived tag for the NEXT mid-invocation cancel. Set by the
        # Steerer (on USER_STEER drift) or by refine-triggered paths
        # BEFORE they cancel the in-flight invoke so
        # _heal_pending_tool_calls knows which content variant to emit
        # in the synthetic function_response. Cleared after consumption
        # so a stale tag can't bleed into the next cancel. See
        # goldfive#139 and
        # :func:`_build_cancelled_response_event` for the content map.
        #
        # Per-session keyed dict (PR #294 audit / goldfive#271 follow-up):
        # the legacy single attribute let session A's USER_STEER tag bleed
        # into session B's cancel emission when one adapter drove two
        # concurrent goldfive sessions. Production writers
        # (:class:`SequentialExecutor`, :class:`ParallelExecutor`,
        # :class:`DefaultSteerer`) now route through
        # :meth:`set_next_cancel_reason` which keys by the goldfive
        # ``Session.id``. The bare ``_next_cancel_reason`` attribute
        # survives as a property backed by ``__legacy_next_cancel_reason``
        # for tests and external callers that drive ``invoke`` on a
        # single session at a time — its read is a fallback consulted
        # only when the per-session dict has no entry for the current
        # session id, so cross-session leak through the legacy slot is
        # impossible in production paths.
        self._next_cancel_reasons: dict[str, str] = {}
        self.__legacy_next_cancel_reason: str = ""
        # Handle to the asyncio.Task currently executing
        # :meth:`_invoke_internal` — captured via ``asyncio.current_task()``
        # at entry and cleared in the ``finally`` block. Used by
        # :meth:`request_cancel` (goldfive#241) to fire ``task.cancel()``
        # on the in-flight invocation from a goldfive-promoted steer so
        # the contaminated LLM call terminates early instead of running
        # to completion while the steerer queues a restart for the next
        # turn.
        #
        # PR #301 follow-up (goldfive#271): keyed by goldfive
        # ``Session.id`` because one adapter is shared across every
        # goldfive session driven by a :class:`Runner`. The pre-fix
        # single-handle slot meant a second concurrent invocation on
        # session B would clobber session A's pinned task at entry,
        # so a USER_STEER cancel intended for A's stream would target
        # B's task instead — wrong-session attribution, B's stream
        # cancelled while A's contaminated stream kept running. The
        # legacy bare ``_inflight_invoke_task`` attribute survives as
        # a ``@property`` shim over the empty-key (``""``) bucket so
        # tests that drive ``adapter._inflight_invoke_task = task`` /
        # ``adapter.request_cancel(...)`` directly keep working
        # (single-session use case). Empty when no invocation is
        # in-flight on any session.
        self._inflight_invoke_tasks: dict[str, asyncio.Task[Any]] = {}
        # Outer session id pinned by :class:`GoldfiveADKAgent` when the
        # adapter runs inside adk-web. ``None`` for programmatic callers
        # and test harnesses; the adapter falls back to the lazy-uuid
        # mint in :meth:`_ensure_session`. Live tests in
        # tests/test_live_steering_e2e.py set this explicitly.
        #
        # PR #294 audit / goldfive#271 follow-up: a single shared field
        # only carried the FIRST pin's id, hiding subsequent pins from
        # forensic / log paths when one adapter served multiple outer
        # adk-web sessions. We keep the legacy ``_outer_session_id``
        # property for back-compat (tests assert the single-session
        # shape directly) and additionally maintain
        # ``_pinned_outer_session_ids`` so structural consumers that
        # iterate over every pinned outer session see them all.
        self.__legacy_outer_session_id: str | None = None
        self._pinned_outer_session_ids: set[str] = set()

        # Optional fan-out listeners for raw inner-Runner ADK events.
        # :meth:`Runner.run_streamed` (goldfive: stream-inner-adk-events)
        # registers a listener here so it can forward every
        # ``google.adk.events.Event`` out to an outer ADK consumer
        # (:class:`GoldfiveADKAgent._run_async_impl`) in real time while
        # the adapter continues its own bookkeeping (pending-tool heal,
        # reconciler, plugin callbacks). Empty by default — no overhead
        # for programmatic callers that don't use ``run_streamed``.
        # Listeners are plain sync callables; they MUST NOT raise and
        # MUST NOT block on I/O. The adapter swallows listener
        # exceptions so a faulty subscriber cannot break the real run.
        self._adk_event_listeners: list[Any] = []

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
    # Per-session state accessors (PR #294 audit / goldfive#271 follow-up)
    # ------------------------------------------------------------------
    #
    # The ADKAdapter is shared across every goldfive session driven by
    # one :class:`Runner`. Three pieces of per-invocation state used to
    # live as bare instance attributes — they leaked across concurrent
    # sessions because there was no key. The accessors below preserve
    # the historical single-attribute API (so single-session callers
    # and tests keep working) while routing production writers through
    # session-keyed helpers.

    @property
    def _next_cancel_reason(self) -> str:
        """Legacy view of the most-recent bare cancel-reason write.
        See :meth:`set_next_cancel_reason` for the session-aware setter
        that production code uses.
        """
        return self.__legacy_next_cancel_reason

    @_next_cancel_reason.setter
    def _next_cancel_reason(self, value: str) -> None:
        self.__legacy_next_cancel_reason = value

    @property
    def _session_id(self) -> str | None:
        """Legacy view of the constructor-pinned / outer-pinned ADK session id."""
        return self.__legacy_session_id

    @_session_id.setter
    def _session_id(self, value: str | None) -> None:
        self.__legacy_session_id = value

    @property
    def _outer_session_id(self) -> str | None:
        """Legacy view of the most-recent outer adk-web session pin."""
        return self.__legacy_outer_session_id

    @_outer_session_id.setter
    def _outer_session_id(self, value: str | None) -> None:
        self.__legacy_outer_session_id = value
        if value:
            self._pinned_outer_session_ids.add(value)

    def set_next_cancel_reason(self, session: Session, reason: str) -> None:
        """Stamp the next cancel reason for ``session``.

        Replaces the bare ``adapter._next_cancel_reason = X`` write so
        two concurrent goldfive sessions on one adapter cannot bleed
        a USER_STEER tag into each other's cancel emission. The reader
        in :meth:`_invoke_internal` consumes the entry keyed by the
        active session id; absent that, falls back to the legacy slot
        so single-session callers (tests, simple scripts) keep their
        bare-attribute write semantics.
        """
        sid = getattr(session, "id", "") or ""
        if sid:
            self._next_cancel_reasons[sid] = reason
        else:
            self.__legacy_next_cancel_reason = reason

    def _consume_next_cancel_reason(self, session: Session) -> str:
        """Pop ``session``'s pending cancel reason, falling back to legacy.

        Production writers route through :meth:`set_next_cancel_reason`
        so the per-session entry wins. Legacy bare-attribute writers
        (tests, external callers) are still observed via the legacy
        slot; the legacy slot is only consulted when the per-session
        dict has no entry for ``session.id`` so cross-session leak
        through the shared slot is impossible in production paths.
        """
        sid = getattr(session, "id", "") or ""
        if sid and sid in self._next_cancel_reasons:
            reason = self._next_cancel_reasons.pop(sid)
            if reason:
                return reason
        legacy = self.__legacy_next_cancel_reason
        self.__legacy_next_cancel_reason = ""
        return legacy

    # ------------------------------------------------------------------
    # Per-session pending-tool-call accessors (PR #301 follow-up)
    # ------------------------------------------------------------------
    #
    # Two concurrent invocations on different goldfive sessions used to
    # share one ``set`` / ``dict`` for tool-call pairing — session A's
    # mid-invocation cancel could heal session B's still-pending ids
    # (and vice versa), corrupting both ADK sessions' history. Now keyed
    # by ``Session.id``; the legacy bare attributes survive as property
    # shims over the empty-key (``""``) bucket so single-session callers
    # and tests that read / write ``adapter._pending_tool_call_ids``
    # directly keep working.

    def _ensure_per_session_dicts(self) -> None:
        """Lazy-init per-session bucket dicts.

        Most callers go through ``__init__`` which sets these dicts
        directly, but a handful of unit tests bypass construction via
        ``ADKAdapter.__new__(ADKAdapter)`` and then set bare attributes.
        The property shims and helpers below tolerate that idiom by
        creating the dicts on first access.
        """
        if not hasattr(self, "_pending_tool_call_ids_by_session"):
            self._pending_tool_call_ids_by_session = {}
        if not hasattr(self, "_pending_tool_call_names_by_session"):
            self._pending_tool_call_names_by_session = {}
        if not hasattr(self, "_inflight_invoke_tasks"):
            self._inflight_invoke_tasks = {}

    def _pending_ids_for(self, session_id: str) -> set[str]:
        """Return (creating if needed) the pending function_call-id set
        for ``session_id``. Always the same object across calls so the
        adapter's ``add`` / ``discard`` / ``clear`` mutations land on the
        right session's bucket.
        """
        self._ensure_per_session_dicts()
        bucket = self._pending_tool_call_ids_by_session.get(session_id)
        if bucket is None:
            bucket = set()
            self._pending_tool_call_ids_by_session[session_id] = bucket
        return bucket

    def _pending_names_for(self, session_id: str) -> dict[str, str]:
        """Return (creating if needed) the function_call-id -> tool-name
        map for ``session_id``. Companion to :meth:`_pending_ids_for`.
        """
        self._ensure_per_session_dicts()
        bucket = self._pending_tool_call_names_by_session.get(session_id)
        if bucket is None:
            bucket = {}
            self._pending_tool_call_names_by_session[session_id] = bucket
        return bucket

    def _clear_pending_for(self, session_id: str) -> None:
        """Empty both pending-id buckets for ``session_id``.

        Drops the per-session entry entirely on empty state so a long-
        lived adapter doesn't accumulate dict entries forever — every
        re-invocation re-creates the bucket on first
        :meth:`_pending_ids_for` access.
        """
        self._ensure_per_session_dicts()
        bucket_ids = self._pending_tool_call_ids_by_session.get(session_id)
        if bucket_ids is not None:
            bucket_ids.clear()
        bucket_names = self._pending_tool_call_names_by_session.get(session_id)
        if bucket_names is not None:
            bucket_names.clear()

    @property
    def _pending_tool_call_ids(self) -> set[str]:
        """Legacy view — the empty-key bucket.

        Single-session callers and tests that read / write
        ``adapter._pending_tool_call_ids`` directly land on the
        ``""`` bucket so their bare-attribute semantics keep working.
        Production code routes through
        :meth:`_pending_ids_for(session_id)` so concurrent sessions
        cannot bleed into each other.
        """
        return self._pending_ids_for("")

    @_pending_tool_call_ids.setter
    def _pending_tool_call_ids(self, value: set[str]) -> None:
        # Replace the empty-key bucket wholesale; tests do this when
        # they pre-populate the set before driving _heal_pending_tool_calls
        # directly (see tests/test_orchestration_state.py).
        self._ensure_per_session_dicts()
        self._pending_tool_call_ids_by_session[""] = set(value)

    @property
    def _pending_tool_call_names(self) -> dict[str, str]:
        """Legacy view — the empty-key bucket.

        See :attr:`_pending_tool_call_ids` for the shim's rationale.
        """
        return self._pending_names_for("")

    @_pending_tool_call_names.setter
    def _pending_tool_call_names(self, value: dict[str, str]) -> None:
        self._ensure_per_session_dicts()
        self._pending_tool_call_names_by_session[""] = dict(value)

    # ------------------------------------------------------------------
    # Per-session in-flight invoke task accessors (PR #301 follow-up)
    # ------------------------------------------------------------------
    #
    # ``_invoke_internal`` pins ``asyncio.current_task()`` so a
    # goldfive-promoted steer can fire ``task.cancel()`` mid-stream
    # (goldfive#241). When two concurrent invocations on different
    # sessions share one slot, the second invocation's pin clobbers
    # the first — a steer for session A would target session B's
    # task instead. Now keyed by ``Session.id``; the legacy bare
    # ``_inflight_invoke_task`` attribute is a property shim over
    # the empty-key (``""``) bucket so single-session tests that
    # drive ``adapter._inflight_invoke_task = task`` /
    # ``adapter.request_cancel(...)`` keep working.

    def _set_inflight_invoke_task(self, session_id: str, task: asyncio.Task[Any] | None) -> None:
        """Pin (or unpin) the in-flight ``invoke`` task for ``session_id``.

        Passing ``None`` removes the entry entirely so a stale handle
        cannot target a completed invocation and ``request_cancel``
        with no ``session=`` kwarg sees an accurate set of live tasks.
        """
        self._ensure_per_session_dicts()
        if task is None:
            self._inflight_invoke_tasks.pop(session_id, None)
        else:
            self._inflight_invoke_tasks[session_id] = task

    def _get_inflight_invoke_task(self, session_id: str) -> asyncio.Task[Any] | None:
        """Return the pinned in-flight task for ``session_id`` or ``None``."""
        self._ensure_per_session_dicts()
        return self._inflight_invoke_tasks.get(session_id)

    @property
    def _inflight_invoke_task(self) -> asyncio.Task[Any] | None:
        """Legacy view — the empty-key bucket's pinned task (or ``None``).

        Tests that drive ``adapter._inflight_invoke_task = task`` see
        their write through this shim's setter and a subsequent
        ``adapter.request_cancel(...)`` (no ``session=`` kwarg) finds it
        in ``_inflight_invoke_tasks[""]`` via :meth:`request_cancel`'s
        cancel-all path.
        """
        self._ensure_per_session_dicts()
        return self._inflight_invoke_tasks.get("")

    @_inflight_invoke_task.setter
    def _inflight_invoke_task(self, value: asyncio.Task[Any] | None) -> None:
        self._ensure_per_session_dicts()
        if value is None:
            self._inflight_invoke_tasks.pop("", None)
        else:
            self._inflight_invoke_tasks[""] = value

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

    # ------------------------------------------------------------------
    # Inner ADK event fan-out (goldfive: stream-inner-adk-events)
    # ------------------------------------------------------------------

    def subscribe_adk_events(self, listener: Any) -> None:
        """Register a sync callable to receive every inner-Runner ADK Event.

        The listener is invoked once per event the adapter consumes from
        ``runner.run_async(...)``, IN ORDER, BEFORE the adapter's own
        bookkeeping (pending-tool tracking, final-event detection,
        runaway-delegation cap) runs. This guarantees the outer
        consumer sees the exact same event stream ADK delivers, without
        the adapter filtering or transforming anything.

        Listeners MUST be sync and MUST NOT block on I/O — a
        ``queue.put_nowait`` or ``list.append`` is the expected shape.
        Any exception raised by the listener is swallowed with a DEBUG
        log so a faulty subscriber cannot break the real run.
        """
        if listener is None:
            return
        if listener in self._adk_event_listeners:
            return
        self._adk_event_listeners.append(listener)

    def unsubscribe_adk_events(self, listener: Any) -> None:
        """Remove a previously-registered ADK Event listener. Idempotent."""
        try:
            self._adk_event_listeners.remove(listener)
        except ValueError:
            return

    def _dispatch_adk_event(self, event: Any) -> None:
        """Fan ``event`` out to every registered listener. Swallows raises."""
        if not self._adk_event_listeners:
            return
        for listener in list(self._adk_event_listeners):
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001 — defensive
                log.debug(
                    "ADKAdapter: adk-event listener %r raised (swallowed): %s",
                    listener,
                    exc,
                )

    def bind_steerer(self, steerer: Steerer | None) -> None:
        """Attach the active :class:`~goldfive.protocols.Steerer`.

        Called by the executor before :meth:`invoke` so plugin callbacks
        can route drift observations. Safe to call with ``None`` to
        unbind.
        """
        self._steerer = steerer

    async def request_cancel(self, reason: str, *, session: Session | None = None) -> None:
        """Cancel the in-flight ADK invocation so a goldfive-promoted
        steer takes effect immediately rather than on the next turn.

        goldfive#241 — the pre-unification goldfive-steer path (see
        :meth:`goldfive.steerer.DefaultSteerer._promote_drift_to_steer`)
        tagged ``_next_cancel_reason`` and queued a restart message but
        left the in-flight ``runner.run_async`` stream running to
        completion. Observed consequence: the coordinator kept writing
        the contaminated reasoning / tool calls into the session for
        tens of seconds after the drift fired, and the restart landed
        only on the next turn. This method fires ``task.cancel()`` on
        the asyncio task currently inside :meth:`_invoke_internal` so
        the ``generate_content_async`` stream raises ``CancelledError``
        and the adapter's standard heal path runs with the tag we just
        stamped on ``_next_cancel_reason``.

        ``reason`` is informational — the ``_next_cancel_reason`` tag
        is already set by the steerer before this call; we log it here
        purely for the operator log.

        PR #301 follow-up (goldfive#271): the in-flight task is now
        keyed by ``Session.id``. When the caller supplies ``session``,
        only that session's task is cancelled — production steerers
        target a specific drift's session and must not collaterally
        cancel a sibling session sharing the same adapter. When
        ``session`` is omitted, every currently-in-flight task is
        cancelled (back-compat with single-session callers and the
        existing :class:`DefaultSteerer._request_adapter_cancel` shape;
        in single-session use there's only one task to cancel anyway).

        No-op when no invocation is in-flight (e.g. the drift fires
        between turns) or when the pinned task is already finished —
        the steerer's restart message still arrives on the next turn.
        Never raises: adapters that ignore cancel still get the queued
        restart via the pre-existing pathway.
        """
        # Resolve the candidate task(s) to cancel.
        self._ensure_per_session_dicts()
        if session is not None:
            sid = getattr(session, "id", "") or ""
            tasks: list[asyncio.Task[Any]] = []
            t = self._inflight_invoke_tasks.get(sid)
            if t is not None:
                tasks.append(t)
        else:
            # Snapshot the values so a concurrent ``finally`` clearing
            # an entry can't mutate the dict mid-iteration.
            tasks = list(self._inflight_invoke_tasks.values())

        live = [t for t in tasks if not t.done()]
        if not live:
            log.debug(
                "ADKAdapter.request_cancel(reason=%r, session=%r): no in-flight invocation; no-op",
                reason,
                getattr(session, "id", None),
            )
            return
        log.info(
            "adapter.request_cancel(reason=%r, session=%r): cancelling %d in-flight invocation(s)",
            reason,
            getattr(session, "id", None),
            len(live),
        )
        for t in live:
            t.cancel()

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",  # noqa: ARG002 -- part of the protocol
        agent_name: str = "",
    ) -> None:
        """Forward an extracted reasoning block to the bound steerer."""
        steerer = self._steerer
        if steerer is None or not text:
            return
        observe = getattr(steerer, "observe_reasoning", None)
        if observe is None:
            return
        try:
            await observe(
                text,
                task=task,
                session=session,
                provider=provider,
                agent_name=agent_name,
            )
        except TypeError:
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

        .. note::
           As of goldfive#163 this method is **no longer called by the
           overlay-mode** :class:`~goldfive.executors.sequential.SequentialExecutor`.
           The overlay now marks PENDING tasks ``NOT_NEEDED`` at the
           end of the passthrough invocation instead of dispatching
           follow-ups — flow-prompted coordinators were re-running
           their full pipeline on every follow-up user message,
           amplifying a ~10 min run into 40+ min. STEER is the
           supported user-driven path for exercising uncovered work.

        The method is retained for external callers that want to
        manually nudge a single task on top of an existing
        conversation (e.g. custom executors, interactive tooling).
        It sends a natural-language "Also, please: ..." user turn
        on top of the existing conversation so the tree picks up
        the missed work without re-running its full pipeline.
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

        session_id = await self._ensure_session(session)
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
        # PR #301 follow-up (goldfive#271): pending-id buckets and
        # the in-flight-task pin are keyed by ``Session.id`` so a
        # second concurrent invocation on a different session cannot
        # see (or clobber) ours.
        gf_session_id = getattr(session, "id", "") or ""
        # Reset per-invocation pending-id bookkeeping (this session only).
        self._clear_pending_for(gf_session_id)
        pending_ids = self._pending_ids_for(gf_session_id)
        pending_names = self._pending_names_for(gf_session_id)
        was_cancelled = False
        # Pin the task driving this invocation so request_cancel() can
        # fire ``task.cancel()`` mid-stream (goldfive#241). Cleared in
        # the ``finally`` block below so a stale handle cannot target
        # the next invocation. Keyed by goldfive ``Session.id`` so
        # concurrent invocations on different sessions don't clobber
        # each other (PR #301 follow-up).
        self._set_inflight_invoke_task(gf_session_id, asyncio.current_task())
        try:
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                # Fan the raw event out to any registered listeners
                # (e.g. :meth:`Runner.run_streamed` forwarding to
                # :class:`GoldfiveADKAgent._run_async_impl` so adk-web
                # sees real per-agent activity in its UI) BEFORE we
                # run any adapter bookkeeping. Dispatch is best-effort
                # sync and swallows raises — the adapter's own
                # observation pipeline must not depend on it.
                self._dispatch_adk_event(event)
                last_event = event
                inv_id = getattr(event, "invocation_id", "") or ""
                if inv_id:
                    last_invocation_id = inv_id
                # Track outstanding function_call ids so we can heal
                # history on mid-invocation cancel (see _heal_pending_tool_calls).
                # The buckets are session-keyed (PR #301 follow-up); we
                # captured the per-session aliases above the loop so
                # the hot path stays a plain set/dict mutation.
                for fc_id, fc_name in _function_call_ids_in_event(event):
                    pending_ids.add(fc_id)
                    if fc_name:
                        pending_names[fc_id] = fc_name
                for fr_id in _function_response_ids_in_event(event):
                    pending_ids.discard(fr_id)
                    pending_names.pop(fr_id, None)
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
            # heal paths keep the neutral content variant. Routed
            # through :meth:`_consume_next_cancel_reason` so the
            # per-session-keyed entry wins over the legacy shared slot
            # — eliminates the cross-session leak flagged by PR #294's
            # audit when one adapter drives multiple goldfive sessions.
            reason = self._consume_next_cancel_reason(session) or "cancelled_mid_invocation"
            await self._heal_pending_tool_calls(
                runner=self._runner,
                session_id=session_id,
                invocation_id=last_invocation_id,
                reason=reason,
                session=session,
            )
            # Notify every caller-supplied plugin that implements the
            # ``on_cancellation(invocation_id)`` hook so they can flush
            # per-invocation state that would otherwise leak.
            #
            # Why this is needed: ADK's
            # :meth:`Runner._exec_with_plugin` places
            # ``after_run_callback`` (and the sub-call after-hooks) AFTER
            # its ``async with Aclosing(execute_fn(...))`` block — NOT
            # inside a ``finally``. On ``CancelledError`` the generator
            # is closed but the after-callbacks never fire. Observability
            # plugins like :class:`HarmonografTelemetryPlugin` that open
            # spans on the before-callbacks would then leave those spans
            # in ``status=RUNNING`` forever (goldfive#167).
            #
            # Best-effort, fire-and-forget: any plugin exception is
            # swallowed so we still re-raise ``CancelledError`` with the
            # expected cancel semantics (no extra exception chaining).
            self._notify_plugins_on_cancellation(last_invocation_id)
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
                session=session,
            )
        finally:
            if not was_cancelled:
                if pending_ids:
                    log.warning(
                        "ADKAdapter.invoke: %d function_call id(s) ended without "
                        "responses on normal exit; healing session history",
                        len(pending_ids),
                    )
                    await self._heal_pending_tool_calls(
                        runner=self._runner,
                        session_id=session_id,
                        invocation_id=last_invocation_id,
                        reason="unexpected_orphan_on_normal_exit",
                        session=session,
                    )
                # No cancel fired — drop any stale tag so the NEXT
                # invoke's cancel (if any) doesn't pick up leftover state.
                # Clear BOTH the per-session entry (the production
                # writers' lane) and the legacy single-slot fallback
                # so neither path can bleed into the next invocation.
                sid_for_clear = getattr(session, "id", "") or ""
                if sid_for_clear:
                    self._next_cancel_reasons.pop(sid_for_clear, None)
                self.__legacy_next_cancel_reason = ""
            if self._plugin is not None:
                # ``clear_active_context`` also clears any attached
                # reconciler — overlay-mode is strictly per-invocation.
                self._plugin.clear_active_context()
            if isinstance(state, Mapping):
                try:
                    state.pop(SESSION_CONTEXT_STATE_KEY, None)  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Release the in-flight task handle so a later
            # request_cancel() cannot target a completed invocation.
            # Keyed by goldfive ``Session.id`` (PR #301 follow-up) so
            # we don't accidentally clear another concurrent
            # invocation's pin on a different session.
            self._set_inflight_invoke_task(gf_session_id, None)

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

    async def _ensure_session(self, session: Session | None = None) -> str:
        """Return the ADK session id for ``session``.

        Per-conversation cache (PR #294 audit / goldfive#271 follow-up):
        the ADK session id is keyed by ``session.conversation_id`` so
        two concurrent goldfive Conversations driven by the same
        adapter target distinct ADK session histories. Within one
        Conversation every turn shares the same id — preserving the
        multi-turn ADK history continuity the legacy single-attribute
        cache provided.

        Lookup order:

        1. ``self._adk_session_ids_by_conv[conversation_id]`` if
           cached — reuse so multi-turn callers stay on one ADK
           session.
        2. Constructor / outer-pin ``self._session_id`` legacy slot —
           seeds the cache so legacy single-session callers (no
           Conversation context, ``Session.conversation_id == ""``)
           keep working unchanged.
        3. ``session.id`` (the goldfive ``run_id``, possibly already
           pinned to the outer adk-web session id by
           :meth:`Runner.run`) — adopting it here lets the harmonograf
           plugin co-locate plan + spans on one session id without
           the older :meth:`_pin_outer_session_on_adapter` shared-slot
           write.
        4. Fresh uuid4 mint — programmatic callers with no pin.
        """
        conv_key = ""
        if session is not None:
            conv_key = getattr(session, "conversation_id", "") or ""

        # 1. Per-conversation cache hit — multi-turn callers reuse the
        #    same ADK session id within one logical Conversation.
        cached = self._adk_session_ids_by_conv.get(conv_key)
        if cached:
            await self._touch_session(cached)
            return cached

        # 2. Legacy single-attribute pin (constructor ``session_id=``
        #    or :meth:`_pin_outer_session_on_adapter` write). Applies
        #    ONLY to the empty-conv-id legacy bucket so concurrent
        #    goldfive Conversations cannot collide on the shared slot.
        if conv_key == "" and self.__legacy_session_id:
            sid = self.__legacy_session_id
            self._adk_session_ids_by_conv[""] = sid
            await self._touch_session(sid)
            return sid

        # 3. Inherit ``session.id`` when present — typically the outer
        #    adk-web session id pinned by :meth:`Runner.run(session_id=)`.
        #    Adopting it here lets the harmonograf plugin co-locate plan
        #    + spans on one session id without the older shared-slot
        #    pin-on-adapter dance.
        if session is not None:
            sid_from_session = getattr(session, "id", "") or ""
            if sid_from_session:
                self._adk_session_ids_by_conv[conv_key] = sid_from_session
                # Mirror to the legacy slot ONLY when the slot is the
                # legacy empty-conv-id bucket and currently empty.
                # Mirroring per-conversation derivations would
                # reintroduce the cross-session leak we just fixed.
                if conv_key == "" and not self.__legacy_session_id:
                    self.__legacy_session_id = sid_from_session
                await self._touch_session(sid_from_session)
                return sid_from_session

        # 4. Lazy uuid4 mint — programmatic callers with no pin and no
        #    Runner-overridden session.run_id.
        minted = str(uuid.uuid4())
        self._adk_session_ids_by_conv[conv_key] = minted
        if conv_key == "" and not self.__legacy_session_id:
            self.__legacy_session_id = minted
        await self._touch_session(minted)
        return minted

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

    def _notify_plugins_on_cancellation(self, invocation_id: str) -> None:
        """Fire-and-forget ``on_cancellation`` notification to every plugin.

        Walks the caller-supplied plugin list (``self._plugins``) and
        invokes ``plugin.on_cancellation(invocation_id)`` on every plugin
        that defines the method. This is the canonical handoff from
        goldfive's ``except asyncio.CancelledError:`` branch to
        observability plugins that must flush per-invocation state —
        most notably :class:`HarmonografTelemetryPlugin`, which closes
        its open spans with ``status=CANCELLED`` to prevent the stale
        ``RUNNING`` spans reported in goldfive#167.

        The hook is an opt-in duck-typed contract: plugins that don't
        need cancellation cleanup simply don't define the method. ADK's
        ``BasePlugin`` doesn't require it.

        Swallows every exception: we must NOT replace the
        ``CancelledError`` about to be re-raised with a plugin-side
        error — that would change the cancel semantics for every caller
        above us in the stack.
        """
        if not invocation_id:
            return
        for plugin in self._plugins:
            hook = getattr(plugin, "on_cancellation", None)
            if not callable(hook):
                continue
            try:
                hook(invocation_id)
            except Exception:  # noqa: BLE001 — observability must not break cancel
                log.debug(
                    "ADKAdapter: plugin %r on_cancellation raised (swallowed)",
                    type(plugin).__name__,
                    exc_info=True,
                )

    async def _heal_pending_tool_calls(
        self,
        *,
        runner: Any,
        session_id: str,
        invocation_id: str,
        reason: str,
        session: Session | None = None,
    ) -> None:
        """Append synthetic ``function_response`` events for orphan tool calls.

        Called from :meth:`invoke`'s cancel / exception paths. For every
        ``function_call`` id still pending in :attr:`_pending_tool_call_ids`,
        build a matching response event (see
        :func:`_build_cancelled_response_event`) and append it to the ADK
        session via ``session_service.append_event``. Best-effort: logs and
        swallows individual failures so healing one orphan doesn't block
        the others.

        goldfive#152: when ``session`` is provided, the orchestration-state
        dict at ``session.state`` is stamped with the healed function_call
        ids under ``goldfive.cancelled_function_call_ids`` (append-only,
        de-duplicated) so downstream planners / prompt templates can see
        which tool-call ids were cancelled mid-invocation without poking
        adapter internals.

        PR #301 follow-up (goldfive#271): the pending-id buckets are
        keyed by goldfive ``Session.id``. The ``_invoke_internal``
        production path populates the per-session bucket directly so
        the heal resolves to ``session.id``'s entries. When the
        per-session bucket is empty AND the legacy empty-key bucket
        is non-empty (a unit-test idiom: pre-populate
        ``adapter._pending_tool_call_ids`` then call this method
        with a constructed ``session``), the heal transparently
        falls back to the legacy bucket so existing tests keep
        working. The session-keyed path always wins when present,
        so the cross-session leak the refactor closes is unaffected.
        """
        gf_session_id = getattr(session, "id", "") or "" if session is not None else ""
        pending_ids_bucket = self._pending_ids_for(gf_session_id)
        pending_names_bucket = self._pending_names_for(gf_session_id)
        # Back-compat fallback for unit tests that pre-populate the
        # legacy bare ``_pending_tool_call_ids`` attribute (lands in
        # the empty-key bucket via the property shim) and then drive
        # this method with a non-empty ``session``. The session-keyed
        # path still wins when present.
        if not pending_ids_bucket and gf_session_id != "":
            legacy_ids = self._pending_ids_for("")
            legacy_names = self._pending_names_for("")
            if legacy_ids:
                pending_ids_bucket = legacy_ids
                pending_names_bucket = legacy_names
                gf_session_id = ""
        if not pending_ids_bucket:
            return
        # goldfive#152: snapshot the ids we're about to heal BEFORE we
        # clear the set so the orchestration-state stamp reflects the
        # full heal even if an early return fires below.
        snapshot_ids = sorted(pending_ids_bucket)
        if session is not None and snapshot_ids:
            try:
                from goldfive import orchestration_state as _ostate

                _ostate.append_cancelled_function_call_ids(session.state, snapshot_ids)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "ADKAdapter._heal_pending_tool_calls: could not stamp "
                    "goldfive.cancelled_function_call_ids: %s",
                    exc,
                )

        session_service = getattr(runner, "session_service", None)
        if session_service is None:
            log.debug(
                "ADKAdapter._heal_pending_tool_calls: runner has no "
                "session_service; cannot heal %d orphan tool call(s)",
                len(pending_ids_bucket),
            )
            self._clear_pending_for(gf_session_id)
            return

        append = getattr(session_service, "append_event", None)
        get = getattr(session_service, "get_session", None)
        if not callable(append) or not callable(get):
            log.debug(
                "ADKAdapter._heal_pending_tool_calls: session_service lacks "
                "append_event/get_session; cannot heal %d orphan tool call(s)",
                len(pending_ids_bucket),
            )
            self._clear_pending_for(gf_session_id)
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
            self._clear_pending_for(gf_session_id)
            return

        if adk_session is None:
            self._clear_pending_for(gf_session_id)
            return

        host_author = str(getattr(self._agent, "name", "") or "") or "user"
        pending_ids = sorted(pending_ids_bucket)
        healed = 0
        for fc_id in pending_ids:
            tool_name = pending_names_bucket.get(fc_id, "")
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

        self._clear_pending_for(gf_session_id)

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
