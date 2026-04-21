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
    """
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part.from_function_response(
        name=tool_name or "unknown_tool",
        response={
            "goldfive_cancelled": True,
            "reason": reason,
            "detail": (
                "Tool call was cancelled mid-invocation by goldfive "
                "(USER_STEER / USER_CANCEL control). This synthetic "
                "response was appended to preserve session history "
                "well-formedness."
            ),
        },
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


def _new_message_parts(task: Task) -> Any:
    """Build the ADK user ``Content`` the adapter sends for a task turn.

    The agent already reads the richer task metadata from
    ``session.state`` under the ``goldfive.*`` keys; the message body
    is just a short imperative nudge so the model has a user turn to
    respond to.
    """
    from google.genai.types import Content, Part  # type: ignore

    title = getattr(task, "title", "") or ""
    description = getattr(task, "description", "") or ""
    body_lines = [f"Task: {title}"]
    if description:
        body_lines.append(description)
    body_lines.append(
        "Use the goldfive.* session-state keys for plan context and call "
        "the report_task_* tools to report outcome."
    )
    return Content(role="user", parts=[Part(text="\n".join(body_lines))])


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
        else:
            self._available_agents = _collect_reachable_agent_names(self._agent)

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
        """Drive one ADK turn for ``task`` and return the result.

        Single-Runner model: the one runner around the root agent is
        driven for every task. Delegation within the tree happens via
        ADK's native ``AgentTool`` / ``transfer_to_agent`` / ``sub_agents``
        mechanisms. ``task.assignee_agent_id`` is carried on the task
        but not used for routing — the planner may populate it for
        observability and for delegation hints inside the agent's
        prompt, but the adapter does not route.

        Three break conditions preserved:

        * ``_task_is_terminal(task, session)`` — the agent reported
          terminal via a reporting tool. Early-exit optimization.
        * ``_is_final_event(event)`` — ADK's final-response flag.
        * Generator end on ``runner.run_async`` natural completion —
          the authoritative termination signal.

        Cancellation: the asyncio task running ``invoke()`` has its
        cancellation propagate naturally through ``runner.run_async()``
        generator awaits, INCLUDING nested AgentTool sub-Runner awaits.
        """
        task_id = getattr(task, "id", "") or ""

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
        # against the LIVE invocation session — which is the #120 state-
        # protocol fix, preserved here.
        if self._plugin is not None:
            self._plugin.set_active_context(ctx)

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
            new_message = _new_message_parts(task)
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
                # Early termination when the agent has reported this task
                # as terminal via a reporting tool.
                if _task_is_terminal(task, session):
                    stop_reason = "task_terminal"
                    break
        except asyncio.CancelledError:
            was_cancelled = True
            stop_reason = "cancelled"
            await self._heal_pending_tool_calls(
                runner=self._runner,
                session_id=session_id,
                invocation_id=last_invocation_id,
                reason="cancelled_mid_invocation",
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
            if self._plugin is not None:
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
