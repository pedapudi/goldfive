"""Internal ADK ``BasePlugin`` used by :class:`goldfive.adapters.adk.ADKAdapter`.

The plugin is the routing layer between ADK's callback lifecycle and
goldfive's :class:`~goldfive.protocols.Steerer`. It does four jobs:

1. **State protocol** — ``before_model_callback`` writes the current
   task and plan context into the ADK session state under the
   ``goldfive.*`` keys (see :mod:`._adk_state_protocol`) so agents can
   read them during their turn.
2. **Reporting-tool interception** — ``before_tool_callback`` watches
   for the eight canonical reporting tools. When one fires the plugin
   routes the call's arguments to the corresponding
   :class:`~goldfive.reporting.ReportingToolSpec` handler and returns
   a short-circuit acknowledgment so ADK doesn't execute the stub
   shim the :class:`FunctionTool` actually wraps.
3. **Tool confirmation bridge** — the same ``before_tool_callback``
   intercepts any ADK tool flagged ``require_confirmation=True``
   (Flow B in ``docs/design/APPROVAL.md``), registers a waiter on
   ``session.pending_approvals`` keyed by the ADK ``function_call_id``,
   emits ``ApprovalRequested``, and suspends the tool call until the
   goldfive control dispatcher lands ``APPROVE`` or ``REJECT``. On
   reject, returns a "skipped" dict so ADK does not run the tool body.
4. **Drift observation** — ``after_model_callback``,
   ``on_event_callback`` (transfer/escalation), and
   ``on_tool_error_callback`` feed raw signals into
   ``steerer.observe(...)`` so the steerer can classify drift.

The plugin never imports ``google.adk`` at module load. It imports the
ADK ``BasePlugin`` base class lazily inside :func:`make_adk_plugin`,
which is only called from the adapter's ``__init__``. That keeps this
module importable from non-ADK code for type-checking and allows unit
tests to patch the base class with a stub when ADK is not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters import _adk_state_protocol as _sp

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.types import Session


log = logging.getLogger("goldfive.adapters.adk")


# ``SessionContext`` is stashed on ADK ``session.state`` under this key
# so the plugin callbacks can reach back to the goldfive session (and
# its steerer + current task) without threading them through every
# callback signature. The adapter writes this before ``runner.run_async``
# and deletes it after.
SESSION_CONTEXT_STATE_KEY = "goldfive._session_context"


class SessionContext:
    """Per-invocation context the adapter stashes on ADK state.

    Carries the goldfive :class:`~goldfive.types.Session`, the active
    :class:`~goldfive.protocols.Steerer`, the task the adapter is about
    to invoke, and the registered reporting-tool handler map. The
    plugin picks it up from the ADK callback's ``callback_context``
    via :func:`_session_context_from_callback`.
    """

    __slots__ = (
        "session",
        "steerer",
        "task",
        "tool_handlers",
        "host_agent_name",
    )

    def __init__(
        self,
        *,
        session: Session,
        steerer: Steerer | None,
        task: Any,
        tool_handlers: Mapping[str, Any],
        host_agent_name: str,
    ) -> None:
        self.session = session
        self.steerer = steerer
        self.task = task
        self.tool_handlers = dict(tool_handlers)
        self.host_agent_name = host_agent_name


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return value if value is not None else default


def _session_state_from_callback(ctx: Any) -> Any:
    """Return the ADK session.state mutable mapping for a callback ctx.

    Tolerates the several shapes ADK has used across versions:
    ``ctx.session.state``, ``ctx._invocation_context.session.state``,
    ``ctx.state``. Returns an empty dict if none match.
    """
    for attr_chain in (
        ("session", "state"),
        ("_invocation_context", "session", "state"),
        ("invocation_context", "session", "state"),
    ):
        cur: Any = ctx
        ok = True
        for part in attr_chain:
            cur = _safe_attr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            return cur
    direct = _safe_attr(ctx, "state", None)
    if direct is not None:
        return direct
    return {}


def _session_context_from_callback(ctx: Any) -> SessionContext | None:
    state = _session_state_from_callback(ctx)
    if not isinstance(state, Mapping):
        return None
    value = state.get(SESSION_CONTEXT_STATE_KEY)
    if isinstance(value, SessionContext):
        return value
    return None


def _extract_text_parts(llm_response: Any) -> list[str]:
    content = _safe_attr(llm_response, "content", None)
    if content is None:
        return []
    parts = _safe_attr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        if _safe_attr(part, "thought", False):
            continue
        text = _safe_attr(part, "text", "") or ""
        if text:
            texts.append(str(text))
    return texts


def _extract_function_calls(llm_response: Any) -> list[dict]:
    content = _safe_attr(llm_response, "content", None)
    if content is None:
        return []
    parts = _safe_attr(content, "parts", None) or []
    calls: list[dict] = []
    for part in parts:
        fc = _safe_attr(part, "function_call", None)
        if fc is None:
            continue
        calls.append(
            {
                "name": str(_safe_attr(fc, "name", "") or ""),
                "args": _safe_attr(fc, "args", None),
            }
        )
    return calls


async def _invoke_handler(
    handler: Any,
    args: Mapping[str, Any],
    session: Session,
    steerer: Steerer | None,
) -> dict[str, Any]:
    """Invoke a reporting-tool handler, tolerating sync or async shapes."""
    import asyncio

    try:
        call = handler(dict(args), session, steerer)
    except TypeError:
        # Some handler implementations accept keyword-style signatures.
        call = handler(args=dict(args), session=session, steerer=steerer)
    if asyncio.iscoroutine(call):
        result = await call
    else:
        result = call
    if isinstance(result, dict):
        return result
    return {"acknowledged": True}


def _as_observation(
    *,
    kind: str,
    detail: str = "",
    raw: Any = None,
    task: Any = None,
    agent_id: str = "",
) -> dict[str, Any]:
    """Build the lightweight observation dict handed to ``steerer.observe``.

    The steerer is responsible for classifying this into a
    :class:`~goldfive.types.DriftEvent` via ``detect_drift``. This
    adapter just translates ADK-native events into a stable shape.
    """
    return {
        "source": "adk",
        "kind": kind,
        "detail": detail,
        "task_id": _safe_attr(task, "id", "") or "",
        "agent_id": agent_id or "",
        "raw": raw,
    }


def _tool_requires_confirmation(tool: Any, tool_args: Any) -> bool:
    """Return True if ``tool`` opts into ADK's require_confirmation flag.

    ``FunctionTool`` stores the flag on ``_require_confirmation``; we also
    accept a public ``require_confirmation`` attribute so tests can
    supply minimal stub tools without subclassing. The value may be a
    bool or a callable that receives the tool args; per ADK semantics
    the callable resolves the decision per-call.
    """
    flag = getattr(tool, "_require_confirmation", None)
    if flag is None:
        flag = getattr(tool, "require_confirmation", None)
    if flag is None:
        return False
    if callable(flag):
        try:
            args = dict(tool_args) if isinstance(tool_args, Mapping) else {}
            return bool(flag(**args))
        except TypeError:
            try:
                return bool(flag(tool_args))
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001
            return False
    return bool(flag)


def _function_call_id(tool_context: Any) -> str:
    """Best-effort pull of the ADK ``function_call_id`` for a tool invocation.

    ADK sets this on ``ToolContext._function_call_id``; exposes it via
    a public property in recent versions. Falls back to generating a
    fresh ``adk-<uuid>`` so correlation still works when tests pass a
    minimal stub context.
    """
    for attr in ("function_call_id", "_function_call_id"):
        value = _safe_attr(tool_context, attr, None)
        if value:
            return str(value)
    import uuid as _uuid

    return f"adk-{_uuid.uuid4().hex}"


async def _emit_approval_requested_from_plugin(
    *,
    session: Any,
    steerer: Any,
    target_id: str,
    prompt: str,
    tool_name: str,
    tool_args: Mapping[str, Any] | dict[str, Any],
    task_id: str,
) -> None:
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    try:
        args_json = json.dumps(
            {k: _jsonable(v) for k, v in dict(tool_args).items()},
            sort_keys=True,
        )
    except Exception:  # noqa: BLE001
        args_json = "{}"
    try:
        from goldfive.events import approval_requested_event, emit

        evt = approval_requested_event(
            run_id=getattr(session, "run_id", ""),
            sequence=session.next_sequence(),
            target_id=target_id,
            kind="tool",
            prompt=prompt,
            task_id=task_id,
            metadata={"tool_name": tool_name, "args_json": args_json},
        )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "_emit_approval_requested_from_plugin: sink emit failed: %s", exc
        )


def _jsonable(v: Any) -> Any:
    """Best-effort coerce ``v`` to a JSON-serializable shape for metadata."""
    if isinstance(v, str | int | float | bool) or v is None:
        return v
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return repr(v)


async def _await_tool_approval(
    *,
    tool: Any,
    tool_name: str,
    tool_args: Any,
    tool_context: Any,
    session_ctx: Any,
) -> dict[str, Any] | None:
    """Gate ``tool`` on a goldfive control-channel APPROVE / REJECT.

    Registers an ``asyncio.Event`` on
    ``session.pending_approvals[function_call_id]``, emits
    ``ApprovalRequested``, and awaits the waiter. On REJECT returns a
    skipped dict (which ADK treats as the tool's response to the model,
    bypassing ``tool.run_async``). On APPROVE returns ``None`` so ADK
    proceeds with the original args.

    No timeout on the wait by design: the control channel is the
    authoritative signal. Callers wanting a timeout should layer it via
    a CANCEL control.
    """
    session = session_ctx.session
    steerer = session_ctx.steerer
    target_id = _function_call_id(tool_context)

    prompt = _tool_approval_prompt(tool, tool_name, tool_args)
    waiter = session.pending_approvals.get(target_id)
    if waiter is None:
        waiter = asyncio.Event()
        session.pending_approvals[target_id] = waiter
    session.pending_approvals_meta.setdefault(
        target_id,
        {
            "kind": "tool",
            "tool_name": tool_name,
            "args": dict(tool_args) if isinstance(tool_args, Mapping) else {},
            "task_id": session_ctx.task.id if session_ctx.task is not None else "",
            "prompt": prompt,
        },
    )

    await _emit_approval_requested_from_plugin(
        session=session,
        steerer=steerer,
        target_id=target_id,
        prompt=prompt,
        tool_name=tool_name,
        tool_args=tool_args if isinstance(tool_args, Mapping) else {},
        task_id=(
            session_ctx.task.id if session_ctx.task is not None else ""
        ),
    )

    await waiter.wait()
    meta = session.pending_approvals_meta.get(target_id, {})
    decision = str(meta.get("decision", "")) or "approve"
    detail = str(meta.get("detail", ""))

    if decision == "reject":
        return {
            "skipped": True,
            "reason": "user_rejected",
            "tool_name": tool_name,
            "detail": detail,
        }
    # APPROVE: fall through so ADK runs the tool normally.
    return None


def _tool_approval_prompt(
    tool: Any, tool_name: str, tool_args: Any
) -> str:
    """Human-readable prompt the UI shows to the human.

    Prefers an explicit ``approval_prompt`` attribute on the tool so
    tool authors can own the copy; otherwise synthesises a short form
    of ``tool_name(arg=value, ...)``.
    """
    explicit = _safe_attr(tool, "approval_prompt", "")
    if explicit:
        return str(explicit)
    if isinstance(tool_args, Mapping) and tool_args:
        parts = [f"{k}={v!r}" for k, v in tool_args.items()]
        return f"Run {tool_name}({', '.join(parts)})?"
    return f"Run {tool_name}()?"


def make_adk_plugin(
    *,
    name: str = "goldfive_adk_plugin",
    host_agent_name: str = "",
) -> Any:
    """Build the ADK plugin class bound to goldfive's protocol.

    The class is built lazily so this module can be imported without
    ``google.adk`` installed. The plugin routes the five callbacks we
    care about (``before_model``, ``before_tool``, ``after_model``,
    ``on_event``, ``on_tool_error``) through the
    :class:`SessionContext` stashed on ADK state.

    ``host_agent_name`` is the fallback name rendered into
    ``goldfive.available_tasks`` entries whose task has no explicit
    assignee — typically the wrapped root agent's name.
    """
    try:
        from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    except ImportError as exc:  # pragma: no cover — tested via importorskip
        raise ImportError(
            "goldfive.adapters.adk requires 'pip install goldfive[adk]'"
        ) from exc

    class _GoldfiveADKPlugin(BasePlugin):  # type: ignore[misc, valid-type]
        """Routes ADK callbacks into the goldfive steerer + state protocol."""

        def __init__(self) -> None:
            super().__init__(name=name)
            self._host_agent_name = host_agent_name

        # --- Plan + current-task context -------------------------------

        async def before_model_callback(
            self, *, callback_context: Any, llm_request: Any
        ) -> None:
            ctx = _session_context_from_callback(callback_context)
            if ctx is None:
                return None
            state = _session_state_from_callback(callback_context)
            if not isinstance(state, dict):
                # Some ADK session state objects are dict-likes. We only
                # write when we can mutate — otherwise the agent just
                # sees no goldfive.* context, which is a survivable
                # degraded mode.
                try:
                    state[_sp.KEY_RUN_ID] = state.get(_sp.KEY_RUN_ID, "")  # type: ignore[index]
                except Exception:
                    return None

            session = ctx.session
            try:
                _sp.write_run_id(state, _safe_attr(session, "run_id", "") or "")
                _sp.write_plan_context(
                    state,
                    _safe_attr(session, "plan", None),
                    _safe_attr(session, "completed_results", {}) or {},
                    self._host_agent_name,
                )
                _sp.write_current_task(state, ctx.task)
                _sp.write_tools_available(state, ctx.tool_handlers.keys())
            except Exception as exc:  # noqa: BLE001
                log.debug("before_model_callback: state write failed: %s", exc)
            return None

        # --- Reporting-tool interception + tool-confirmation bridge ---

        async def before_tool_callback(
            self, *, tool: Any, tool_args: Any, tool_context: Any
        ) -> dict[str, Any] | None:
            ctx = _session_context_from_callback(tool_context)
            if ctx is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            if not tool_name:
                func = _safe_attr(tool, "func", None)
                tool_name = str(_safe_attr(func, "__name__", "") or "")

            # Reporting-tool short-circuit takes precedence: a tool named
            # e.g. report_task_started should never also be gated by
            # confirmation — the protocol handlers are control-plane
            # calls, not side-effects.
            handler = ctx.tool_handlers.get(tool_name)
            if handler is not None:
                args_map: Mapping[str, Any]
                if isinstance(tool_args, Mapping):
                    args_map = tool_args
                else:
                    args_map = {}
                try:
                    result = await _invoke_handler(
                        handler, args_map, ctx.session, ctx.steerer
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_tool_callback: handler for %s raised: %s",
                        tool_name,
                        exc,
                    )
                    # Fall back to the canonical acknowledgment so the
                    # agent doesn't see a tool error for a protocol call.
                    return {"acknowledged": True, "error": str(exc)}
                # Return a non-None result to short-circuit ADK tool dispatch.
                return result or {"acknowledged": True}

            # Tool-level approval (Flow B). If the tool opts into
            # confirmation via ADK's native `require_confirmation` flag,
            # bridge the gate onto goldfive's control channel.
            if _tool_requires_confirmation(tool, tool_args):
                return await _await_tool_approval(
                    tool=tool,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_context=tool_context,
                    session_ctx=ctx,
                )
            return None

        # --- Drift observation -----------------------------------------

        async def after_model_callback(
            self, *, callback_context: Any, llm_response: Any
        ) -> None:
            ctx = _session_context_from_callback(callback_context)
            if ctx is None or ctx.steerer is None:
                return None
            texts = _extract_text_parts(llm_response)
            calls = _extract_function_calls(llm_response)
            finish = _safe_attr(llm_response, "finish_reason", None)
            observation = _as_observation(
                kind="llm_response",
                detail=" ".join(texts)[:500],
                raw={
                    "texts": texts,
                    "function_calls": calls,
                    "finish_reason": str(finish) if finish is not None else "",
                },
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("after_model_callback: steerer.observe raised: %s", exc)
            return None

        async def on_event_callback(
            self, *, invocation_context: Any, event: Any
        ) -> None:
            ctx = _session_context_from_callback(invocation_context)
            if ctx is None or ctx.steerer is None:
                return None
            # Detect transfer / escalation actions on the event payload.
            actions = _safe_attr(event, "actions", None)
            transfer_to = _safe_attr(actions, "transfer_to_agent", "") or ""
            escalate = bool(_safe_attr(actions, "escalate", False))
            if not transfer_to and not escalate:
                return None
            kind = "agent_transfer" if transfer_to else "agent_escalation"
            detail = (
                f"transfer -> {transfer_to}" if transfer_to else "escalate"
            )
            observation = _as_observation(
                kind=kind,
                detail=detail,
                raw=event,
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("on_event_callback: steerer.observe raised: %s", exc)
            return None

        async def on_tool_error_callback(
            self,
            *,
            tool: Any,
            tool_args: Any,
            tool_context: Any,
            error: Any,
        ) -> None:
            ctx = _session_context_from_callback(tool_context)
            if ctx is None or ctx.steerer is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            observation = _as_observation(
                kind="tool_error",
                detail=f"{tool_name}: {error}",
                raw={"tool": tool_name, "error": repr(error)},
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "on_tool_error_callback: steerer.observe raised: %s", exc
                )
            return None

    return _GoldfiveADKPlugin()
