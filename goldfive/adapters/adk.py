"""ADK :class:`~goldfive.protocols.AgentAdapter` for Google's Agent Development Kit.

Wraps an ADK ``BaseAgent`` (or an existing ``Runner``) and conforms it
to goldfive's :class:`~goldfive.protocols.AgentAdapter` protocol so
goldfive's runner / executor / steerer can drive it uniformly.

Responsibilities:

* **Reporting tool registration** — :meth:`ADKAdapter.register_reporting_tools`
  accepts goldfive :class:`~goldfive.reporting.ReportingToolSpec` values,
  wraps each as a ``google.adk.tools.FunctionTool`` around a thin sync
  shim that just returns an acknowledgment, and attaches the tools to
  the inner agent (plus every sub-agent reachable via ``sub_agents`` /
  ``inner_agent`` / nested ``AgentTool.agent``). The plugin intercepts
  calls to these tools in ``before_tool_callback`` and routes their
  arguments to the spec's real handler — the in-process shim is never
  actually executed by ADK.
* **Invocation** — :meth:`ADKAdapter.invoke` stashes a
  :class:`~._adk_plugin.SessionContext` onto ADK session state,
  drives one turn via ``runner.run_async(...)``, consumes the event
  stream to assemble an :class:`~goldfive.results.InvocationResult`,
  and cleans the context back off.

Optional dependency
-------------------

``google.adk`` is an optional install. The top-level import in this
module is gated so ``import goldfive.adapters.adk`` raises a clear
``ImportError`` with install instructions when ADK is missing.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters import _adk_state_protocol as _sp
from goldfive.adapters._adk_plugin import (
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.results import InvocationResult

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.reporting import ReportingToolSpec
    from goldfive.types import Session, Task

log = logging.getLogger("goldfive.adapters.adk")


try:  # noqa: SIM105 — explicit import-time guard with install hint
    import google.adk  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover — covered in tests via importorskip
    raise ImportError(
        "goldfive.adapters.adk requires 'pip install goldfive[adk]'"
    ) from None


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


def _augment_subtree_with_reporting(
    root_agent: Any, tools: list[Any], tool_names: set[str]
) -> int:
    """Append reporting ``tools`` to every agent reachable from ``root_agent``.

    Ported from harmonograf's ``_register_harmonograf_reporting_tools_for_test``.
    Traverses three edges to cover the shapes goldfive must support:

    * ``sub_agents`` — native ADK agent tree.
    * ``inner_agent`` — wrapper agents (e.g. HarmonografAgent-style).
    * ``AgentTool.agent`` — agents exposed to a parent as tools.

    Idempotent: agents that already carry any of the canonical reporting
    tool names are skipped. Returns the number of agents touched.
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
            n = getattr(t, "name", None) or getattr(
                getattr(t, "func", None), "__name__", None
            )
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


def _build_runner(agent: Any) -> Any:
    """Construct an ADK ``InMemoryRunner`` around a ``BaseAgent``.

    Used by :class:`ADKAdapter` when the caller passes an agent rather
    than a runner. ``app_name`` defaults to the agent's name for tidy
    session-service bookkeeping.
    """
    from google.adk.runners import InMemoryRunner  # type: ignore

    app_name = str(getattr(agent, "name", "") or "goldfive")
    return InMemoryRunner(agent=agent, app_name=app_name)


def _looks_like_runner(obj: Any) -> bool:
    """Duck-type check for an ADK Runner.

    Runners expose ``run_async`` / ``agent`` / ``session_service``;
    agents may expose ``run_async_impl`` but not the session service.
    """
    return (
        callable(getattr(obj, "run_async", None))
        and getattr(obj, "agent", None) is not None
    )


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

    Parameters
    ----------
    agent_or_runner:
        Either an ADK ``BaseAgent`` (in which case an ``InMemoryRunner``
        is constructed and used) or an already-built ``Runner``.
    user_id:
        Stable user id for ADK session lookup. Defaults to
        ``"goldfive_user"`` which is fine for local / single-user runs.
    session_id:
        Optional stable session id. When omitted the adapter mints a
        fresh session id on first :meth:`invoke`.
    app_name:
        Optional ADK app_name override. Defaults to the runner's own
        ``app_name`` or the agent's ``name``.
    """

    def __init__(
        self,
        agent_or_runner: Any,
        *,
        user_id: str = "goldfive_user",
        session_id: str | None = None,
        app_name: str | None = None,
    ) -> None:
        if _looks_like_runner(agent_or_runner):
            self._runner = agent_or_runner
            self._agent = getattr(agent_or_runner, "agent", None)
        else:
            self._agent = agent_or_runner
            self._runner = _build_runner(agent_or_runner)

        if self._agent is None:
            raise ValueError("ADKAdapter: could not resolve an inner agent")

        self._user_id = user_id
        self._session_id = session_id
        self._app_name = (
            app_name
            or getattr(self._runner, "app_name", None)
            or getattr(self._agent, "name", None)
            or "goldfive"
        )

        host_agent_name = str(getattr(self._agent, "name", "") or "")
        self._plugin = make_adk_plugin(host_agent_name=host_agent_name)
        if not _register_plugin_on_runner(self._runner, self._plugin):
            log.warning(
                "ADKAdapter: could not attach plugin to runner %r — "
                "reporting callbacks will be inactive",
                type(self._runner).__name__,
            )

        # tool_name -> handler. Populated by register_reporting_tools.
        self._tool_handlers: dict[str, Any] = {}
        # The current Steerer. Set by bind_steerer() before invoke().
        self._steerer: Steerer | None = None

    # ------------------------------------------------------------------
    # Post-construction plugin install
    # ------------------------------------------------------------------

    def add_plugin(self, plugin: Any) -> None:
        """Install an ADK ``BasePlugin`` on the inner ``Runner``.

        Tolerant of runners that don't expose a plugin manager (e.g. a
        custom ``Runner`` shape) — logs at DEBUG and returns. Used by
        observability integrations that need to attach a plugin after
        the adapter has already been built.
        """
        if not _register_plugin_on_runner(self._runner, plugin):
            log.debug(
                "ADKAdapter.add_plugin: runner %r has no plugin manager; "
                "plugin %r not installed",
                type(self._runner).__name__,
                type(plugin).__name__,
            )

    # ------------------------------------------------------------------
    # AgentAdapter protocol
    # ------------------------------------------------------------------

    @property
    def available_agents(self) -> list[str]:
        """Return the names of agents reachable from the root agent."""
        names: list[str] = []
        seen: set[int] = set()
        stack: list[Any] = [self._agent]
        while stack:
            cur = stack.pop()
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            name = getattr(cur, "name", None)
            if isinstance(name, str) and name:
                names.append(name)
            for sub in getattr(cur, "sub_agents", None) or ():
                stack.append(sub)
            inner = getattr(cur, "inner_agent", None)
            if inner is not None:
                stack.append(inner)
            for t in getattr(cur, "tools", None) or ():
                nested = getattr(t, "agent", None)
                if nested is not None:
                    stack.append(nested)
        return names

    async def register_reporting_tools(
        self, tools: list[ReportingToolSpec]
    ) -> None:
        """Register goldfive reporting tools with the wrapped agent tree.

        Each spec is wrapped as a ``google.adk.tools.FunctionTool`` (via
        a stub ACK shim) and attached to the inner agent and every
        sub-agent reachable via :func:`_augment_subtree_with_reporting`.
        The spec's handler is stored in :attr:`_tool_handlers`; the
        plugin's ``before_tool_callback`` routes real calls to it.
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
                raise ValueError(
                    f"ReportingToolSpec '{name}' missing handler"
                )
            self._tool_handlers[name] = handler
            function_tools.append(_build_function_tool(spec))
            names.add(name)

        # Attach to the inner agent itself (root), then the whole subtree.
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
                n = getattr(t, "name", None) or getattr(
                    getattr(t, "func", None), "__name__", None
                )
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
        """Forward an extracted reasoning block to the bound steerer.

        Normally the ADK plugin's ``after_model_callback`` extracts
        reasoning and calls ``steerer.observe_reasoning`` directly; this
        method is the public protocol-level entry point so callers that
        wire reasoning through the adapter (tests, custom executors)
        don't need to reach into the plugin internals.
        """
        steerer = self._steerer
        if steerer is None or not text:
            return
        observe = getattr(steerer, "observe_reasoning", None)
        if observe is None:
            return
        await observe(text, task=task, session=session, provider=provider)

    async def invoke(
        self, task: Task, session: Session
    ) -> InvocationResult:
        """Drive one ADK turn for ``task`` and return the result."""
        task_id = getattr(task, "id", "") or ""
        session_id = await self._ensure_session()
        state = await self._get_session_state(session_id)

        ctx = SessionContext(
            session=session,
            steerer=self._steerer,
            task=task,
            tool_handlers=self._tool_handlers,
            host_agent_name=str(getattr(self._agent, "name", "") or ""),
        )

        # Stash the per-invocation context on ADK state so plugin
        # callbacks can pick it up. Written directly rather than via
        # the state protocol because this is an adapter-internal handoff.
        try:
            state[SESSION_CONTEXT_STATE_KEY] = ctx  # type: ignore[index]
        except Exception:
            log.debug("ADKAdapter.invoke: could not stash session context")

        # Pre-seed the goldfive.* state keys so agents reading session
        # state before the first before_model_callback see context.
        if _is_mutable_mapping(state):
            try:
                _sp.write_run_id(state, getattr(session, "run_id", "") or "")
                _sp.write_plan_context(
                    state,
                    getattr(session, "plan", None),
                    getattr(session, "completed_results", {}) or {},
                    ctx.host_agent_name,
                )
                _sp.write_current_task(state, task)
                _sp.write_tools_available(state, list(self._tool_handlers))
            except Exception as exc:  # noqa: BLE001
                log.debug("ADKAdapter.invoke: state pre-seed failed: %s", exc)

        final_text = ""
        stop_reason = "completed"
        err: Exception | None = None
        last_event: Any = None
        try:
            new_message = _new_message_parts(task)
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                last_event = event
                text = _extract_text_from_event(event)
                if text:
                    final_text = text
                if _is_final_event(event):
                    stop_reason = "final_response"
        except Exception as exc:  # noqa: BLE001
            err = exc
            stop_reason = f"error:{type(exc).__name__}"
            log.debug("ADKAdapter.invoke: runner.run_async raised: %s", exc)
        finally:
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
        """Return the ADK session id, creating one if none has been set."""
        if self._session_id:
            return self._session_id

        session_service = getattr(self._runner, "session_service", None)
        if session_service is None:
            self._session_id = str(uuid.uuid4())
            return self._session_id

        new_id = str(uuid.uuid4())
        try:
            create = getattr(session_service, "create_session", None)
            if callable(create):
                coro = create(
                    app_name=self._app_name,
                    user_id=self._user_id,
                    session_id=new_id,
                )
                if hasattr(coro, "__await__"):
                    await coro
        except Exception as exc:  # noqa: BLE001
            log.debug("ADKAdapter._ensure_session: create_session raised: %s", exc)
        self._session_id = new_id
        return new_id

    async def _get_session_state(self, session_id: str) -> Any:
        """Fetch the ADK session's mutable state dict for ``session_id``."""
        session_service = getattr(self._runner, "session_service", None)
        if session_service is None:
            return {}
        get = getattr(session_service, "get_session", None)
        if not callable(get):
            return {}
        try:
            coro = get(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            if hasattr(coro, "__await__"):
                session = await coro
            else:
                session = coro
        except Exception as exc:  # noqa: BLE001
            log.debug("ADKAdapter._get_session_state: get_session raised: %s", exc)
            return {}
        state = getattr(session, "state", None)
        if state is None:
            return {}
        return state


def _is_mutable_mapping(obj: Any) -> bool:
    try:
        obj[_sp.KEY_RUN_ID] = obj.get(_sp.KEY_RUN_ID, "")  # type: ignore[index]
        return True
    except Exception:
        return False
