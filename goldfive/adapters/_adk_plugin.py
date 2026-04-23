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
import time
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters import _adk_state_protocol as _sp
from goldfive.adapters._tool_invocation import invoke_tool

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.reporting import ReportingToolSpec
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
    to invoke, the registered reporting-tool handler map, and the full
    :class:`~goldfive.reporting.ReportingToolSpec` list. The plugin
    picks it up from the ADK callback's ``callback_context`` via
    :func:`_session_context_from_callback`.

    The ``tools`` field is the authoritative source the plugin's
    ``before_tool_callback`` uses to route calls through
    :func:`~goldfive.adapters._tool_invocation.invoke_tool` — that
    routing is what picks up the terminal-task rejection, idempotency,
    and loop-guard layers. ``tool_handlers`` is kept for backward
    compatibility with callers that construct a ``SessionContext``
    directly (examples + test stubs) without a full spec list; when
    only ``tool_handlers`` is supplied the plugin synthesizes minimal
    ``ReportingToolSpec`` shims so the dispatch path still flows
    through ``invoke_tool``.
    """

    __slots__ = (
        "session",
        "steerer",
        "task",
        "tool_handlers",
        "tools",
        "host_agent_name",
    )

    def __init__(
        self,
        *,
        session: Session,
        steerer: Steerer | None,
        task: Any,
        tool_handlers: Mapping[str, Any] | None = None,
        tools: list[ReportingToolSpec] | None = None,
        host_agent_name: str,
    ) -> None:
        self.session = session
        self.steerer = steerer
        self.task = task
        self.host_agent_name = host_agent_name

        # Prefer an explicit ``tools`` list (the adapter constructs
        # ``SessionContext`` that way). Fall back to materialising
        # lightweight specs from ``tool_handlers`` so existing external
        # callers (approval_gated_agent example, legacy tests) still
        # route through ``invoke_tool`` and get the protection layers.
        if tools is not None:
            self.tools = list(tools)
            # Keep ``tool_handlers`` populated as a legacy read-only
            # view for any code that introspected it (e.g.
            # ``before_model_callback`` used to surface the list of
            # reporting-tool names to the state protocol).
            self.tool_handlers = {spec.name: spec.handler for spec in tools}
        else:
            self.tool_handlers = dict(tool_handlers or {})
            self.tools = _tools_from_handlers(self.tool_handlers)


def _tools_from_handlers(
    tool_handlers: Mapping[str, Any],
) -> list[ReportingToolSpec]:
    """Materialise a list of ``ReportingToolSpec`` from a name→handler map.

    Used when a caller builds :class:`SessionContext` with just a
    handler map (legacy path — examples / test stubs). We fabricate
    minimal specs so the dispatch path still routes through
    :func:`invoke_tool` and picks up the terminal-task rejection,
    idempotency, and loop-guard layers.

    The synthesized specs carry empty ``description`` / ``parameters``
    — the plugin only needs ``name`` + ``handler`` at dispatch time;
    the rich schema is surfaced through the native SDK tool wrapping
    done earlier in the adapter's ``register_reporting_tools``.
    """
    from goldfive.reporting import ReportingToolSpec

    specs: list[ReportingToolSpec] = []
    for name, handler in tool_handlers.items():
        specs.append(
            ReportingToolSpec(
                name=str(name),
                description="",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
    return specs


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


# --------------------------------------------------------------------------
# goldfive#191 — Layer 3: task_id arg injection for reporting tools
# --------------------------------------------------------------------------
#
# When an LLM emits a report_task_* / report_awaiting_approval call with a
# missing or obviously-placeholder ``task_id``, we fall back to the pinned
# ``goldfive.current_task_id`` that ``before_agent_callback`` (Layer 1)
# stamps into session.state at the start of every agent turn. The reporting-
# tool handler itself (Layer 2) also defaults from state, so this layer is
# the outermost safety net: the goal is to have the handler see a valid
# task_id no matter which dispatch path runs it.
#
# We ONLY rewrite when the arg is missing or is a well-known placeholder
# string. A real-looking task_id (even if it's for the wrong task) is left
# alone so the handler surfaces the mismatch as a proper terminal-task /
# not-found failure rather than silently re-targeting the call. Wrong
# task_ids are better surfaced as failures than masked.

# Reporting-tool names that must always target a specific task. The match
# is an exact-set test for ``report_awaiting_approval`` plus a prefix test
# for ``report_task_*`` (see goldfive.reporting for the canonical list).
_REPORT_AWAITING_APPROVAL = "report_awaiting_approval"

# Case-insensitive set of strings we treat as "the LLM didn't supply a real
# task_id". Whitespace is stripped before comparison. Keep this list
# conservative — any real-looking slug should NOT be on it.
_PLACEHOLDER_TASK_IDS: frozenset[str] = frozenset(
    {"", "placeholder", "unknown", "todo", "none", "null", "n/a"}
)


def _is_placeholder_task_id(value: Any) -> bool:
    """Return True if ``value`` is missing or an obvious placeholder.

    Used by :meth:`_GoldfiveADKPlugin.before_tool_callback` to decide
    whether to overwrite ``tool_args["task_id"]`` with the pinned
    ``goldfive.current_task_id`` from session.state. Case-insensitive and
    whitespace-tolerant. Non-string values are treated as placeholders
    (an int task_id isn't real either).
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return True
    return value.strip().lower() in _PLACEHOLDER_TASK_IDS


def _is_reporting_tool_name(tool_name: str) -> bool:
    """Return True if ``tool_name`` is a reporting-tool that needs a task_id."""
    if not tool_name:
        return False
    return tool_name.startswith("report_task_") or tool_name == _REPORT_AWAITING_APPROVAL


def _inject_task_id_from_state(
    *,
    tool_name: str,
    tool_args: Any,
    tool_context: Any,
) -> None:
    """Best-effort: fill in ``tool_args['task_id']`` from pinned state.

    This is the goldfive#191 Layer-3 safety net. Mutates ``tool_args`` in
    place when:

      * ``tool_name`` is a reporting tool (report_task_* or
        report_awaiting_approval),
      * ``tool_args`` is a mutable mapping,
      * the current ``task_id`` arg is missing / blank / a known
        placeholder,
      * session.state has a non-empty ``goldfive.current_task_id``.

    NEVER raises — injection is advisory. If anything goes wrong we log
    at DEBUG and return, letting Layer 2 (handler default-from-state)
    handle the fallback or surface the error.
    """
    try:
        if not _is_reporting_tool_name(tool_name):
            return
        if not isinstance(tool_args, MutableMapping):
            return
        existing = tool_args.get("task_id", "")
        if not _is_placeholder_task_id(existing):
            return
        state = _session_state_from_callback(tool_context)
        if not isinstance(state, Mapping):
            return
        state_tid = state.get(_sp.KEY_CURRENT_TASK_ID, "")
        if not isinstance(state_tid, str) or not state_tid.strip():
            return
        tool_args["task_id"] = state_tid
    except Exception:  # noqa: BLE001
        log.debug(
            "before_tool_callback: task_id injection failed for tool=%s",
            tool_name,
            exc_info=True,
        )


def _measure_request_chars(llm_request: Any) -> tuple[int, int]:
    """Return ``(total_chars, messages_count)`` for an ADK ``LlmRequest``.

    Used by the goldfive#172 per-LLM-call instrumentation in
    :meth:`_GoldfiveADKPlugin.before_model_callback`. We walk
    ``llm_request.contents`` (a list of ``Content`` whose ``parts`` hold
    ``text`` / ``function_call`` / ``function_response`` leaves) and
    sum the character count of each serialised part. The three leaf
    shapes we care about:

    * ``part.text`` -- plain assistant / user / system text.
    * ``part.function_call`` -- a model-emitted tool call. We serialise
      as ``name + json(args)`` because both contribute to the prompt
      tokens the model pays for.
    * ``part.function_response`` -- a tool's return payload. Serialised
      as ``name + json(response)`` for the same reason.

    Unknown part shapes fall through silently (count zero) so a novel
    ADK part type doesn't break instrumentation. The system instruction
    on ``llm_request.config.system_instruction`` is counted separately
    when present, since it's a prompt-prefix the model must process on
    every call — often the dominant contributor right after a
    GoldfivePlanner injection.

    Returns ``(0, 0)`` on any failure — instrumentation must never
    raise into the caller path.
    """
    try:
        contents = _safe_attr(llm_request, "contents", None) or []
        total_chars = 0
        messages_count = 0
        for content in contents:
            messages_count += 1
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                text = _safe_attr(part, "text", "") or ""
                if text:
                    total_chars += len(str(text))
                    continue
                fc = _safe_attr(part, "function_call", None)
                if fc is not None:
                    name = str(_safe_attr(fc, "name", "") or "")
                    args = _safe_attr(fc, "args", None)
                    total_chars += len(name)
                    if args is not None:
                        try:
                            total_chars += len(json.dumps(args, default=repr))
                        except Exception:  # noqa: BLE001
                            total_chars += len(repr(args))
                    continue
                fr = _safe_attr(part, "function_response", None)
                if fr is not None:
                    name = str(_safe_attr(fr, "name", "") or "")
                    resp = _safe_attr(fr, "response", None)
                    total_chars += len(name)
                    if resp is not None:
                        try:
                            total_chars += len(json.dumps(resp, default=repr))
                        except Exception:  # noqa: BLE001
                            total_chars += len(repr(resp))
                    continue
        # Include the system instruction — a GoldfivePlanner injection
        # lands here and typically dominates the prompt prefix.
        config = _safe_attr(llm_request, "config", None)
        sys_inst = _safe_attr(config, "system_instruction", "") or ""
        if isinstance(sys_inst, str):
            total_chars += len(sys_inst)
        elif sys_inst is not None:
            try:
                total_chars += len(str(sys_inst))
            except Exception:  # noqa: BLE001
                pass
        return total_chars, messages_count
    except Exception:  # noqa: BLE001 — instrumentation must never raise
        return 0, 0


def _extract_usage_metadata(llm_response: Any) -> dict[str, int]:
    """Pull prompt / completion / total token counts off an ADK ``LlmResponse``.

    ADK normalises per-provider usage onto
    ``llm_response.usage_metadata`` (a
    ``google.genai.types.GenerateContentResponseUsageMetadata`` with
    ``prompt_token_count`` / ``candidates_token_count`` /
    ``total_token_count``). Returns an empty dict when the backend
    didn't report usage (some LiteLLM-fronted providers skip it, some
    streaming responses defer it to a trailing chunk).

    Returns only fields that are present and non-zero — callers treat
    missing keys as "not reported".
    """
    out: dict[str, int] = {}
    usage = _safe_attr(llm_response, "usage_metadata", None)
    if usage is None:
        return out
    for src_attr, dst_key in (
        ("prompt_token_count", "prompt_tokens"),
        ("candidates_token_count", "completion_tokens"),
        ("total_token_count", "total_tokens"),
    ):
        value = _safe_attr(usage, src_attr, None)
        if isinstance(value, int) and value > 0:
            out[dst_key] = value
    return out


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


def _infer_provider(llm_response: Any) -> str:
    """Guess which backend produced ``llm_response`` for observability.

    Returns one of ``"openai"`` / ``"anthropic"`` / ``"google"`` /
    ``""``. Used only to tag reasoning-drift events, so an approximate
    answer is fine -- misclassification does not change detector
    behavior.
    """
    module = type(llm_response).__module__ if llm_response is not None else ""
    module_lower = module.lower()
    if "anthropic" in module_lower:
        return "anthropic"
    if "openai" in module_lower or "litellm" in module_lower:
        return "openai"
    if "google" in module_lower or "genai" in module_lower:
        return "google"
    # ADK's LlmResponse carries a Content with parts that include
    # thought-flagged entries: treat as google when no better signal.
    content = _safe_attr(llm_response, "content", None)
    if content is not None and _safe_attr(content, "parts", None) is not None:
        return "google"
    if _safe_attr(llm_response, "choices", None) is not None:
        return "openai"
    return ""


def _extract_reasoning(llm_response: Any) -> str:
    """Best-effort per-provider reasoning extraction.

    Reasoning-content / thinking blocks live in different places on
    different providers. This helper walks the known shapes in
    priority order and returns the first non-empty reasoning text it
    finds, or ``""`` when the response carries none.

    Shapes handled:

    * ADK ``content.parts[i]`` with ``thought=True`` -- Google's
      Gemini surface (thought blocks are standard parts flagged as
      such).
    * OpenAI-compat ``response.choices[0].message.reasoning_content``
      -- Qwen3.5 via LiteLLM, some o1-series models, Deepseek.
    * Anthropic ``response.content[i].type == "thinking"`` blocks
      -- Claude extended thinking.
    * Plain string fields ``reasoning`` / ``reasoning_content`` on
      the response itself (tolerant fallback).

    Returns the concatenated reasoning text. Callers downstream treat
    empty strings as "no reasoning available".
    """
    # ADK thought parts: parts with .thought=True carry the
    # chain-of-thought when Google's Gemini returns one.
    content = _safe_attr(llm_response, "content", None)
    if content is not None:
        parts = _safe_attr(content, "parts", None) or []
        thoughts: list[str] = []
        for part in parts:
            if not _safe_attr(part, "thought", False):
                continue
            text = _safe_attr(part, "text", "") or ""
            if text:
                thoughts.append(str(text))
        if thoughts:
            return "\n".join(thoughts)

    # OpenAI-compat: response.choices[0].message.reasoning_content.
    # Used by LiteLLM-fronted Qwen3.5 and some o1 / Deepseek models.
    try:
        choices = _safe_attr(llm_response, "choices", None) or []
        if choices:
            msg = _safe_attr(choices[0], "message", None)
            if msg is not None:
                rc = _safe_attr(msg, "reasoning_content", None) or _safe_attr(
                    msg, "reasoning", None
                )
                if rc:
                    return str(rc)
    except Exception:  # noqa: BLE001 -- best-effort extraction
        pass

    # Anthropic: content blocks with type="thinking".
    try:
        blocks = _safe_attr(llm_response, "content", None)
        if isinstance(blocks, list):
            for block in blocks:
                if _safe_attr(block, "type", "") == "thinking":
                    t = _safe_attr(block, "thinking", "") or ""
                    if t:
                        return str(t)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: a flat attribute on the response itself.
    for attr in ("reasoning_content", "reasoning", "thinking"):
        v = _safe_attr(llm_response, attr, None)
        if isinstance(v, str) and v:
            return v

    return ""


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


def _is_progress_report_success(response: Any) -> bool:
    """Return ``True`` iff ``response`` indicates a successful progress report.

    Used by the ADK plugin's ``after_tool_callback`` to decide whether a
    ``report_task_*`` / ``report_awaiting_approval`` call should reset
    the tool-loop tracker's per-(invocation, agent) window (goldfive#192).

    Previously the exemption triggered on the **call** regardless of
    outcome: an agent stuck retrying ``report_task_started`` with a
    bad ``task_id`` would keep getting ``{"acknowledged": false,
    "error": "missing_task_id"}`` and every one of those errored calls
    reset the detector window -- masking an obvious tool-loop. This
    helper tightens the gate so only acknowledged-success responses
    reset. Everything else (errored, missing field, unknown shape) is
    conservatively treated as NOT a legitimate progress report so the
    loop detector gets to count the call.

    Shape contract: the reporting tools return a dict with
    ``acknowledged`` set to ``True`` on success and ``False`` on
    failure (with an ``error`` key describing what went wrong); see
    :mod:`goldfive.adapters._tool_invocation`. Any other shape
    (``None``, a string, a bare list, etc.) is treated as "unknown"
    and does NOT reset the window.
    """
    if response is None:
        return False
    if isinstance(response, Mapping):
        # Explicit failure wins: even if ``acknowledged`` is somehow
        # True alongside an ``error`` key, treat it as errored so we
        # don't silently reset on half-broken shapes.
        if "error" in response:
            return False
        if response.get("acknowledged") is True:
            return True
        return False
    return False  # unknown shape (str, list, bare value) -- conservative no-reset


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
            session_id=getattr(session, "id", ""),
        )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("_emit_approval_requested_from_plugin: sink emit failed: %s", exc)


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
        task_id=(session_ctx.task.id if session_ctx.task is not None else ""),
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


def _tool_approval_prompt(tool: Any, tool_name: str, tool_args: Any) -> str:
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


async def _inject_goldfive_planner_instruction(
    *,
    callback_context: Any,
    llm_request: Any,
) -> None:
    """Append :class:`GoldfivePlanner` output to ``llm_request.config.system_instruction``.

    ADK's ``flows/llm_flows/_nl_planning.py`` request-side processor
    gates instruction injection on ``isinstance(planner,
    PlanReActPlanner)`` — so a ``BasePlanner`` subclass that is NOT a
    PlanReAct subclass gets its ``build_planning_instruction`` called
    on the RESPONSE side (via ``process_planning_response``) but never
    on the REQUEST side. That's fine for response filtering but it
    starves the model of goldfive's orchestration context block on
    the turn that matters.

    This helper is the workaround: it detects when the running
    agent's ``.planner`` is a :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner`
    (NOT a ``PlanReActPlanner`` or ``BuiltInPlanner`` — ADK handles
    those on its own via ``_nl_planning``) and appends the planner's
    :meth:`build_planning_instruction` output to the request's system
    instruction using ADK's own ``append_instructions`` method.

    Silent fall-throughs in priority order:

    * ADK not installed or ``BasePlanner`` import fails → skip.
    * Running agent has no ``planner`` attribute or planner is None
      → skip (plain ADK LlmAgent with no goldfive attachment).
    * Planner is a ``PlanReActPlanner`` / ``BuiltInPlanner`` subclass
      → skip (ADK will handle it natively).
    * ``build_planning_instruction`` returns ``None`` / empty →
      skip (planner opted out for this turn).
    * ``llm_request`` lacks ``append_instructions`` → skip (unit-test
      request stubs).
    """
    try:
        from goldfive.planners.goldfive_planner import GoldfivePlanner
    except ImportError:  # pragma: no cover — ADK not installed
        return
    try:
        from google.adk.planners.base_planner import BasePlanner  # type: ignore
        from google.adk.planners.built_in_planner import BuiltInPlanner  # type: ignore
        from google.adk.planners.plan_re_act_planner import (  # type: ignore
            PlanReActPlanner,
        )
    except ImportError:  # pragma: no cover — ADK not installed
        return

    # Find the running agent on the callback_context. ADK exposes it
    # through the invocation context; tests may supply a context that
    # carries ``.agent`` directly.
    inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
        callback_context, "invocation_context", None
    )
    agent = _safe_attr(inv_ctx, "agent", None)
    if agent is None:
        agent = _safe_attr(callback_context, "agent", None)
    if agent is None:
        return

    planner = _safe_attr(agent, "planner", None)
    if planner is None:
        return
    # If ADK itself will inject for this planner type, skip. ``BuiltInPlanner``
    # never emits a text instruction (it configures thinking on the
    # request instead), ``PlanReActPlanner`` is the one ADK gates on.
    if isinstance(planner, (PlanReActPlanner, BuiltInPlanner)):
        return
    if not isinstance(planner, BasePlanner):
        return
    # Narrow further: the adapter attaches GoldfivePlanner specifically.
    # A custom BasePlanner subclass that is not GoldfivePlanner should
    # be respected by ADK's own (response-side) dispatch only, not
    # re-injected here. This keeps the hook behaviour predictable for
    # users who subclass BasePlanner themselves.
    if not isinstance(planner, GoldfivePlanner):
        return

    # Build the ReadonlyContext ADK expects. When invocation_context
    # is available we use ADK's real class; otherwise we fall back to
    # ``callback_context`` itself (test stubs carrying ``.state`` work
    # through GoldfivePlanner's tolerant _extract_state).
    readonly = callback_context
    try:
        from google.adk.agents.readonly_context import ReadonlyContext  # type: ignore

        if inv_ctx is not None:
            readonly = ReadonlyContext(inv_ctx)
    except Exception as exc:  # noqa: BLE001 — use fallback
        log.debug(
            "_inject_goldfive_planner_instruction: ReadonlyContext unavailable: %s",
            exc,
        )

    instruction = planner.build_planning_instruction(readonly, llm_request)
    if not instruction:
        return

    append = getattr(llm_request, "append_instructions", None)
    if not callable(append):
        # Best-effort write directly into ``config.system_instruction``
        # when the test stub doesn't carry the helper. Preserves the
        # existing value if it's a string.
        config = getattr(llm_request, "config", None)
        if config is None:
            return
        existing = getattr(config, "system_instruction", None)
        if not existing:
            try:
                config.system_instruction = instruction
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_inject_goldfive_planner_instruction: could not set system_instruction: %s",
                    exc,
                )
        elif isinstance(existing, str):
            try:
                config.system_instruction = existing + "\n\n" + instruction
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_inject_goldfive_planner_instruction: could not append system_instruction: %s",
                    exc,
                )
        return

    try:
        append([instruction])
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "_inject_goldfive_planner_instruction: append_instructions raised: %s",
            exc,
        )


def make_adk_plugin(
    *,
    name: str = "goldfive_adk_plugin",
    host_agent_name: str = "",
    agent_tool_cap: int = 16,
) -> Any:
    """Build the ADK plugin class bound to goldfive's protocol.

    The class is built lazily so this module can be imported without
    ``google.adk`` installed. The plugin routes the callbacks we care
    about (``before_run``, ``after_run``, ``before_model``,
    ``before_tool``, ``after_model``, ``on_event``, ``on_tool_error``)
    through the :class:`SessionContext` stashed on ADK state.

    ``host_agent_name`` is the fallback name rendered into
    ``goldfive.available_tasks`` entries whose task has no explicit
    assignee — typically the wrapped root agent's name.

    ``agent_tool_cap`` is the maximum number of ``AgentTool`` spawns
    the plugin will tolerate in a single top-level invocation before
    emitting a ``RUNAWAY_DELEGATION`` drift and signalling cancel.
    Set to ``0`` or a negative value to disable. See goldfive#130 —
    the cap is the backstop against a coordinator whose prompt
    describes a pipeline and keeps delegating forever.
    """
    try:
        from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    except ImportError as exc:  # pragma: no cover — tested via importorskip
        raise ImportError("goldfive.adapters.adk requires 'pip install goldfive[adk]'") from exc

    class _GoldfiveADKPlugin(BasePlugin):  # type: ignore[misc, valid-type]
        """Routes ADK callbacks into the goldfive steerer + state protocol."""

        def __init__(self) -> None:
            super().__init__(name=name)
            self._host_agent_name = host_agent_name
            self._agent_tool_cap = agent_tool_cap
            # Active :class:`SessionContext` for the invocation that is
            # currently driving this plugin's runner. Set by
            # :meth:`ADKAdapter.invoke` before ``run_async`` and cleared
            # in its ``finally`` block. Callbacks prefer this field over
            # the ADK-state lookup because ADK's
            # :class:`~google.adk.sessions.in_memory_session_service.InMemorySessionService`
            # returns a **shallow copy** of the stored session on every
            # ``get_session`` call (see ``_light_copy`` /
            # ``copy.copy(session.state)``) — so a SessionContext written
            # into the adapter's own ``get_session`` copy never reaches
            # the fresh copy that ``runner.run_async`` materialises for
            # the invocation, and the callbacks would see an empty state
            # and silently fall through to the ACK shim.
            #
            # A module-level fallback path (``_session_context_from_callback``)
            # is kept so unit tests that stash a ``SessionContext`` in a
            # plain dict they control still work — the state-based lookup
            # there is authoritative for those synthetic harnesses.
            self._active_ctx: SessionContext | None = None
            # Overlay-model reconciler (goldfive#141). Attached by
            # :meth:`ADKAdapter.invoke_passthrough` before ``run_async``;
            # cleared in its ``finally``. When present, the plugin
            # forwards ``before_agent_callback`` / ``after_agent_callback``
            # and delegation observations to the reconciler so it can
            # transition plan tasks based on observed agent activity.
            # None outside the overlay path — ``invoke(task)`` and
            # ``invoke_follow_up`` keep the per-task model and do NOT
            # attach a reconciler.
            self._reconciler: Any = None
            # Track the top-level invocation_id on the current dispatch so
            # AgentTool-spawned sub-Runners' before_run_callbacks can
            # attribute themselves with a ``parent_invocation_id``.
            self._top_invocation_id: str = ""
            # Per-invocation tool-call counters and last-text buffers
            # keyed by ADK ``invocation_id``. Feeds the
            # CONFABULATION_RISK classifier in ``after_run_callback``:
            # a research-shaped task that completes with zero tool calls
            # and non-empty text is the suspicious pattern worth
            # surfacing. Reset per invocation so nested AgentTool
            # sub-Runners get their own counters.
            self._invocation_tool_calls: dict[str, int] = {}
            self._invocation_last_text: dict[str, str] = {}
            # AgentTool-per-invoke counter (goldfive#130). Scoped to
            # the current top-level invocation; reset in
            # :meth:`clear_active_context`. When the counter exceeds
            # :attr:`_agent_tool_cap` the plugin sets
            # :attr:`runaway_delegation_tripped`, emits a
            # ``RUNAWAY_DELEGATION`` drift, and short-circuits
            # subsequent AgentTool calls in the same invocation.
            self._agent_tool_spawn_count: int = 0
            # One-shot flag: True once the cap has been exceeded in the
            # current invocation. The adapter's ``invoke`` loop reads
            # this to break out of ``run_async`` cleanly — the drift
            # event has already been emitted.
            self.runaway_delegation_tripped: bool = False
            # Per-LLM-call instrumentation (goldfive#172). Keyed by
            # ADK ``invocation_id`` — ``before_model_callback`` stashes
            # the start time + request chars + message count here and
            # ``after_model_callback`` pops it to compute
            # ``llm.call.duration_ms``. Since ADK fires before/after
            # pairs synchronously for a single invocation, a single
            # slot per invocation_id is sufficient (nested AgentTool
            # sub-Runners get their own invocation_id). Each entry is
            # a small dict to keep the payload auditable in tests.
            self._invocation_llm_pending: dict[str, dict[str, Any]] = {}
            # Tool-loop drift detector (goldfive#181). Observes every
            # tool call the plugin sees in ``after_tool_callback`` and
            # fires a ``LOOPING_REASONING`` drift when any of the
            # three configured patterns (exact / name / alternating)
            # trips. Thresholds read from ``GOLDFIVE_TOOL_LOOP_*`` env
            # vars, falling back to the defaults documented in
            # :mod:`goldfive.drift.tool_loops`. Lazy import so the
            # plugin module stays importable without the drift helpers
            # materialised — matches the pattern used for the
            # confabulation classifier.
            from goldfive.drift import tool_loops as _tool_loops

            self._tool_loop_tracker = _tool_loops.ToolLoopTracker(
                **_tool_loops.load_thresholds_from_env()
            )
            # Reporting-tool names that indicate forward task progress
            # and therefore can clear the tool-loop tracker's window
            # for the current (invocation, agent) key — SUBJECT to the
            # acknowledged-success gate in ``after_tool_callback``
            # (goldfive#192). Matches the set exposed by the adapter's
            # state-transition protocol, plus the approval-gate
            # reporter which is the other call an agent uses to signal
            # forward progress on a running task.
            self._progress_reporting_tools: frozenset[str] = frozenset(
                {
                    "report_task_started",
                    "report_task_progress",
                    "report_task_completed",
                    "report_task_failed",
                    "report_task_blocked",
                    "report_awaiting_approval",
                }
            )

        def set_active_context(self, ctx: SessionContext) -> None:
            """Attach the ``SessionContext`` for the running invocation.

            Called once per :meth:`ADKAdapter.invoke` before
            ``runner.run_async``. The plugin's callback methods prefer
            this context over any value stashed in ADK session state
            (which is an unreliable channel because InMemorySessionService
            copies state on every get). Overwriting a non-``None`` value
            is accepted — sequential invocations reuse the adapter.
            """
            self._active_ctx = ctx
            # Reset the runaway-delegation bookkeeping for the new
            # invocation so a prior trip doesn't leak into this one.
            self._agent_tool_spawn_count = 0
            self.runaway_delegation_tripped = False

        def clear_active_context(self) -> None:
            """Release the active ``SessionContext`` reference.

            Called from ``ADKAdapter.invoke``'s ``finally`` block. Safe
            to call when no context is active.
            """
            self._active_ctx = None
            self._top_invocation_id = ""
            self._agent_tool_spawn_count = 0
            self.runaway_delegation_tripped = False
            self._reconciler = None
            # Drop any straggling per-LLM-call metrics entries
            # (goldfive#172). Normal operation pops each entry in
            # ``after_model_callback``; this catches the case where a
            # model turn errored between before/after and never paired.
            self._invocation_llm_pending.clear()
            # Drop per-(invocation, agent) tool-loop ring buffers so
            # state from the just-finished dispatch doesn't leak into
            # the next one on the same plugin instance (goldfive#181).
            self._tool_loop_tracker.clear()

        def set_reconciler(self, reconciler: Any) -> None:
            """Attach a :class:`~goldfive.reconciler.PlanReconciler`.

            Set by :meth:`ADKAdapter.invoke_passthrough` for the
            duration of a single overlay-model invocation. The plugin
            will forward before/after-agent + delegation observations
            to the reconciler's hooks. Pass ``None`` to detach.
            """
            self._reconciler = reconciler

        def _resolve_ctx(self, adk_ctx: Any) -> SessionContext | None:
            """Return the live ``SessionContext`` or ``None`` if unbound.

            Prefers the plugin-local ``_active_ctx`` (authoritative for
            real ADK runs) but falls through to the state-dict lookup
            for unit tests that drive the plugin with a hand-built
            ``tool_context`` holding a populated state mapping. The two
            paths are never inconsistent in production — only one is
            populated at a time.
            """
            if self._active_ctx is not None:
                return self._active_ctx
            return _session_context_from_callback(adk_ctx)

        # --- Invocation lifecycle --------------------------------------

        async def before_run_callback(self, *, invocation_context: Any) -> None:
            """Seed ``goldfive.*`` state on the LIVE invocation session and
            emit :class:`AgentInvocationStarted`.

            This is the RELIABILITY-CRITICAL state-protocol write path.
            ``invocation_context.session`` is the session ADK is actually
            running the invocation against — writes here are visible to
            every subsequent callback and tool on the same session,
            including AgentTool-spawned sub-Runners (whose own
            ``before_run_callback`` fires with their own session and
            therefore gets its own authoritative seed).

            Previously the adapter wrote these keys against a session
            fetched via ``session_service.get_session`` — which
            ``InMemorySessionService`` returns as a shallow copy, so the
            writes landed on a stranded dict the runner never saw. That
            was flagged "best-effort" and left the state-protocol keys
            unreliable, which is unacceptable for correctness — see
            docs/design/TASK-LIFECYCLE.md §5.
            """
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None:
                return None

            # Determine parent_invocation_id: if a top-level one is
            # already pinned on this plugin, we're in a nested AgentTool
            # sub-Runner. The top-level adapter invocation is the first
            # one to fire before_run_callback.
            inv_id = str(_safe_attr(invocation_context, "invocation_id", "") or "")
            parent_inv_id = ""
            if self._top_invocation_id:
                parent_inv_id = self._top_invocation_id
            else:
                # First before_run for this dispatch — pin it so nested
                # AgentTool sub-Runners can attribute themselves below.
                self._top_invocation_id = inv_id

            # Reset the per-invocation counters so the CONFABULATION_RISK
            # check in ``after_run_callback`` sees only the tool calls
            # and final text produced by THIS invocation. Nested
            # AgentTool sub-Runners fire their own before_run_callback
            # with a fresh ``invocation_id`` so they get their own slot.
            if inv_id:
                self._invocation_tool_calls[inv_id] = 0
                self._invocation_last_text[inv_id] = ""

            # Write state-protocol keys onto the LIVE session the
            # invocation is actually running against — not a copy.
            session_obj = _safe_attr(invocation_context, "session", None)
            state = _safe_attr(session_obj, "state", None)
            if state is not None:
                try:
                    _sp.write_run_id(state, _safe_attr(ctx.session, "run_id", "") or "")
                    _sp.write_plan_context(
                        state,
                        _safe_attr(ctx.session, "plan", None),
                        _safe_attr(ctx.session, "completed_results", {}) or {},
                        self._host_agent_name,
                    )
                    _sp.write_current_task(state, ctx.task)
                    _sp.write_tools_available(state, list(ctx.tool_handlers.keys()))
                except Exception as exc:  # noqa: BLE001
                    log.debug("before_run_callback: state write failed: %s", exc)

                # goldfive#170: bridge orchestration-level state
                # (goldfive.Session.state — written by DefaultSteerer,
                # PlanReconciler, and the heal path) onto the live ADK
                # session.state so GoldfivePlanner's request-side
                # injection renders real values instead of ``(none)``.
                # Tree-agnostic: fires on every invocation, including
                # AgentTool-spawned sub-Runners whose own
                # ``before_run_callback`` will repeat this bridge on
                # their own live session. No separate propagation path
                # needed.
                try:
                    self._bridge_orchestration_state(ctx.session, state)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_run_callback: orchestration-state bridge failed: %s",
                        exc,
                    )

            # Emit AgentInvocationStarted. Best-effort: observability
            # only, so a sink / proto issue must not block the run.
            agent_name = str(_safe_attr(ctx, "host_agent_name", "") or "") or self._host_agent_name
            running_agent = _safe_attr(invocation_context, "agent", None)
            running_agent_name = str(_safe_attr(running_agent, "name", "") or "")
            if running_agent_name:
                agent_name = running_agent_name
            await self._emit_observability(
                "agent_invocation_started",
                agent_name=agent_name,
                task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                invocation_id=inv_id,
                parent_invocation_id=parent_inv_id,
            )
            # Feed the trajectory-level activity buffer used by the
            # GOAL_DRIFT judge (goldfive#143). Duck-typed: custom
            # steerers without ``note_agent_activity`` fall through to
            # a no-op. Always safe — the recorder does not itself
            # trigger an LLM call.
            note_activity = getattr(ctx.steerer, "note_agent_activity", None)
            if note_activity is not None:
                try:
                    note_activity(
                        ctx.session,
                        kind="agent_invocation_started",
                        agent_name=agent_name,
                        task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("before_run_callback: note_agent_activity raised: %s", exc)
            return None

        async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            """Pin ``goldfive.current_task_id`` for the starting sub-agent and
            forward an agent-turn start to the overlay reconciler.

            Fires once per agent invocation (including sub-agents
            inside AgentTool sub-Runners). Two jobs:

            1. **Task-id pinning (goldfive#191 Layer 1).** At delegation
               time, find the unique plan task whose
               ``assignee_agent_id == agent.name`` and whose status is
               PENDING or RUNNING, and stamp its id onto both the live
               ADK ``session.state`` and the goldfive orchestration
               ``session.state`` under the ``goldfive.current_task_id``
               key. Sub-agents' reporting-tool handlers read this key
               as a fallback when the model's tool call omits
               ``task_id``, so delegated work doesn't retry-loop on
               the structured ``missing_task_id`` error.

               Zero matches (off-plan agent) and multiple matches
               (ambiguous — a coordinator with two pending siblings
               for the same assignee) intentionally leave the state
               unset. The ``missing_task_id`` error path still fires
               in those cases — better an explicit rejection than a
               mis-attributed report.

            2. **Overlay reconciler forward.** When a
               :class:`~goldfive.reconciler.PlanReconciler` is
               attached, we forward ``agent.name``, the invocation
               id, and the parent_invocation_id (the outer runner's
               id, when the current invocation is nested inside an
               ``AgentTool`` sub-Runner) so the reconciler can
               resolve parent chains for contextual matching
               (goldfive#151).
            """
            agent_name = str(_safe_attr(agent, "name", "") or "")
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            parent_inv_id = ""
            if inv_id and self._top_invocation_id and inv_id != self._top_invocation_id:
                parent_inv_id = self._top_invocation_id

            # Layer 1: pin the starting sub-agent's task_id so its
            # reporting-tool calls can default the arg from state
            # (goldfive#191). Best-effort: a raise here must never
            # break the invocation.
            try:
                self._pin_current_task_id_for_agent(
                    agent_name=agent_name,
                    callback_context=callback_context,
                )
            except Exception as exc:  # noqa: BLE001 — pinning must never raise
                log.debug(
                    "before_agent_callback: current_task_id pin raised: %s",
                    exc,
                )

            reconciler = self._reconciler
            if reconciler is None:
                return None
            try:
                await reconciler.on_before_agent(
                    agent_name=agent_name,
                    invocation_id=inv_id,
                    parent_invocation_id=parent_inv_id,
                )
            except TypeError:
                # Custom reconciler without the #151 kwarg — fall back
                # to the pre-#151 signature. Keeps back-compat.
                try:
                    await reconciler.on_before_agent(
                        agent_name=agent_name,
                        invocation_id=inv_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: reconciler.on_before_agent raised: %s",
                        exc,
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_agent_callback: reconciler.on_before_agent raised: %s",
                    exc,
                )
            return None

        def _pin_current_task_id_for_agent(
            self,
            *,
            agent_name: str,
            callback_context: Any,
        ) -> None:
            """Stamp ``goldfive.current_task_id`` for ``agent_name`` if unambiguous.

            Matching rule: the plan task whose ``assignee_agent_id``
            equals ``agent_name`` and whose status is PENDING or
            RUNNING. Exactly-one matches stamp the id onto both the
            live ADK ``session.state`` (agent-side reads via
            ``tool_ctx.state``) and the goldfive orchestration
            ``session.state`` (handler fallback in
            :mod:`goldfive.reporting` + :mod:`goldfive.adapters._tool_invocation`).

            Zero / multiple matches leave state unset — the handler
            path will surface the existing ``missing_task_id`` error
            rather than guess, which is the correct signal for an
            off-plan agent or an ambiguous coordinator assignment.

            Silent on every failure mode (no agent name, no ctx, no
            plan, state not a mapping). Instrumentation-class path:
            a mistake here degrades to the pre-#191 behaviour where
            the model sees ``missing_task_id`` and retry-loops — bad,
            but strictly no worse than the baseline.
            """
            if not agent_name:
                return
            ctx = self._resolve_ctx(callback_context)
            if ctx is None:
                return
            plan = _safe_attr(ctx.session, "plan", None)
            if plan is None:
                return
            tasks = _safe_attr(plan, "tasks", None) or ()
            # Import here so the type is available without forcing a
            # top-level import for a rarely-hot-path enum compare.
            from goldfive.types import TaskStatus

            matches: list[Any] = []
            for task in tasks:
                assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                if assignee != agent_name:
                    continue
                status = _safe_attr(task, "status", None)
                if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                    matches.append(task)
                    if len(matches) > 1:
                        break  # ambiguous — no need to count further

            if len(matches) != 1:
                # Zero matches (off-plan) or >1 matches (ambiguous).
                # Leave state unset; the existing ``missing_task_id``
                # error path remains the explicit signal.
                log.debug(
                    "before_agent_callback: not pinning current_task_id for %s "
                    "(%d PENDING/RUNNING task matches)",
                    agent_name,
                    len(matches),
                )
                return
            task = matches[0]
            task_id = str(_safe_attr(task, "id", "") or "")
            if not task_id:
                return

            # Stamp the goldfive orchestration-state key first — the
            # reporting-tool handler fallback in ``invoke_tool`` +
            # ``reporting.py`` reads from here.
            gf_state = _safe_attr(ctx.session, "state", None)
            if isinstance(gf_state, dict):
                try:
                    gf_state[_sp.KEY_CURRENT_TASK_ID] = task_id
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: goldfive session.state pin failed: %s",
                        exc,
                    )

            # Stamp the ADK session.state key — the sibling's Layer 3
            # (before_tool_callback arg-injection) reads from here and
            # any agent that inspects ``tool_ctx.state`` directly picks
            # up the same value.
            adk_state = _session_state_from_callback(callback_context)
            if isinstance(adk_state, Mapping):
                try:
                    _sp.write_current_task_id(adk_state, task_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: ADK session.state pin failed: %s",
                        exc,
                    )

            log.info(
                "goldfive: pinned current_task_id=%s for sub-agent %s",
                task_id,
                agent_name,
            )

        async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            """Forward an agent-turn end to the overlay reconciler."""
            reconciler = self._reconciler
            if reconciler is None:
                return None
            agent_name = str(_safe_attr(agent, "name", "") or "")
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            parent_inv_id = ""
            if inv_id and self._top_invocation_id and inv_id != self._top_invocation_id:
                parent_inv_id = self._top_invocation_id
            summary = self._invocation_last_text.get(inv_id, "") if inv_id else ""
            try:
                await reconciler.on_after_agent(
                    agent_name=agent_name,
                    invocation_id=inv_id,
                    error=None,
                    summary=summary,
                    parent_invocation_id=parent_inv_id,
                )
            except TypeError:
                try:
                    await reconciler.on_after_agent(
                        agent_name=agent_name,
                        invocation_id=inv_id,
                        error=None,
                        summary=summary,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "after_agent_callback: reconciler.on_after_agent raised: %s",
                        exc,
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "after_agent_callback: reconciler.on_after_agent raised: %s",
                    exc,
                )
            return None

        async def after_run_callback(self, *, invocation_context: Any) -> None:
            """Emit :class:`AgentInvocationCompleted` when an invocation ends.

            Fires once per runner invocation: top-level (goldfive
            dispatch) and per-AgentTool sub-Runner.

            Also runs the cheap structural CONFABULATION_RISK classifier
            (issue #128) before cleanup: if the current task's
            description reads like external-data work but the
            invocation produced non-empty text with zero tool calls, a
            goldfive.drift.classify_confabulation_risk call surfaces the
            suspicious pattern as an INFO drift through the same path
            AGENT_REFUSAL uses.
            """
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None:
                return None
            inv_id = str(_safe_attr(invocation_context, "invocation_id", "") or "")
            agent_name = str(_safe_attr(ctx, "host_agent_name", "") or "") or self._host_agent_name
            running_agent = _safe_attr(invocation_context, "agent", None)
            running_agent_name = str(_safe_attr(running_agent, "name", "") or "")
            if running_agent_name:
                agent_name = running_agent_name

            # Confabulation-risk check. We run this BEFORE emitting
            # "agent_invocation_completed" so the drift lands in the
            # event stream adjacent to the invocation it describes,
            # matching the AGENT_REFUSAL ordering. Gated on:
            #   * a live steerer + task,
            #   * assignee on the task matching the agent that just
            #     finished (so nested AgentTool sub-Runners do not
            #     misattribute their inner text to the outer task),
            #   * the counters we tracked per invocation_id.
            await self._maybe_emit_confabulation_risk(
                ctx=ctx,
                inv_id=inv_id,
                finishing_agent_name=agent_name,
            )

            # If the finishing invocation is the top-level one, release
            # the pin so a subsequent invoke() on the same plugin gets a
            # fresh dispatch.
            if self._top_invocation_id and self._top_invocation_id == inv_id:
                self._top_invocation_id = ""
            # Drop the per-invocation counters now that the check has
            # run — keeps the dict bounded across long-lived plugins.
            if inv_id:
                self._invocation_tool_calls.pop(inv_id, None)
                self._invocation_last_text.pop(inv_id, None)
            await self._emit_observability(
                "agent_invocation_completed",
                agent_name=agent_name,
                task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                invocation_id=inv_id,
                summary="",
            )
            # Feed the GOAL_DRIFT activity buffer + counter
            # (goldfive#143). Duck-typed: custom steerers without
            # these hooks fall through cleanly. The counter is
            # trajectory-level and persists across task transitions.
            if ctx.steerer is not None:
                note_activity = getattr(ctx.steerer, "note_agent_activity", None)
                if note_activity is not None:
                    try:
                        note_activity(
                            ctx.session,
                            kind="agent_invocation_completed",
                            agent_name=agent_name,
                            task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_run_callback: note_agent_activity raised: %s",
                            exc,
                        )
                note_agent_turn = getattr(ctx.steerer, "note_agent_turn", None)
                if note_agent_turn is not None:
                    try:
                        await note_agent_turn(ctx.session)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("after_run_callback: note_agent_turn raised: %s", exc)
            return None

        def _bridge_orchestration_state(
            self,
            gf_session: Any,
            adk_state: Any,
        ) -> None:
            """Copy ``goldfive.Session.state`` orchestration keys onto the
            ADK session.state (goldfive#170).

            The orchestration dict is the framework-agnostic source of
            truth for active-steer body / turn, formatted goals summary,
            and the list of cancelled function-call ids (written by the
            DefaultSteerer USER_STEER path, PlanReconciler goals-refresh
            path, and the adapter's heal path respectively).
            :class:`GoldfivePlanner.build_planning_instruction` reads
            the same logical keys off the ADK session.state — so this
            bridge is the missing data path between the orchestration
            writes and the per-turn instruction injection.

            Called from :meth:`before_run_callback` against the live
            invocation session so the bridge runs on every root invoke
            AND every AgentTool-spawned sub-Runner invoke (which has
            its own ``before_run_callback`` firing against its own
            live session). Sub-Runner propagation is therefore
            automatic — no separate handoff path required.

            Silent on any individual key that can't be read: the
            planner's placeholders default to ``(none)`` so a degraded
            bridge never breaks the run.
            """
            # Lazy import: orchestration_state is framework-agnostic
            # and cheap, but the adapter module shouldn't assume at
            # import time that the orchestration module is loaded.
            from goldfive import orchestration_state as _ostate

            gf_state = _safe_attr(gf_session, "state", None)
            if gf_state is None:
                return
            # Active steer body + turn. The ADK-side helper clears
            # both keys when body is empty, so a steer-then-clear
            # sequence correctly renders as ``(none)`` on the next
            # turn instead of retaining the old body.
            body = _ostate.read(gf_state, _ostate.KEY_ACTIVE_STEER_BODY, "")
            at_turn = _ostate.read(gf_state, _ostate.KEY_ACTIVE_STEER_AT_TURN, None)
            try:
                _sp.set_active_steer_on_adk_state(
                    adk_state,
                    body=str(body) if body else "",
                    at_turn=at_turn,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_bridge_orchestration_state: active_steer bridge failed: %s",
                    exc,
                )
            # Goals summary.
            summary = _ostate.read(gf_state, _ostate.KEY_GOALS_SUMMARY, "")
            try:
                _sp.set_goals_summary_on_adk_state(
                    adk_state,
                    str(summary) if summary else "",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_bridge_orchestration_state: goals_summary bridge failed: %s",
                    exc,
                )
            # Cancelled function-call ids. Use the orchestration-state
            # reader so the list-shape guard (non-list → []) is
            # centralised in one place.
            cancelled = _ostate.read_cancelled_function_call_ids(gf_state)
            try:
                _sp.set_cancelled_function_call_ids_on_adk_state(
                    adk_state,
                    cancelled,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_bridge_orchestration_state: cancelled_ids bridge failed: %s",
                    exc,
                )

        async def _maybe_emit_confabulation_risk(
            self,
            *,
            ctx: SessionContext,
            inv_id: str,
            finishing_agent_name: str,
        ) -> None:
            """Fire ``CONFABULATION_RISK`` if the invocation shape is suspicious.

            See :func:`goldfive.drift.classify_confabulation_risk` for
            the exact trigger conditions. Silent when any precondition
            fails — tasks without a clear assignee, sub-agent
            invocations whose assignee does not match, or invocations
            with no tracked state all fall through to no-op so we never
            over-report.
            """
            if ctx.steerer is None or ctx.task is None:
                return
            task = ctx.task
            task_id = str(_safe_attr(task, "id", "") or "")
            if not task_id:
                # Out-of-scope per issue #128: tasks without a clear id
                # / assignee fall through to no-op.
                return
            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
            if assignee and finishing_agent_name and assignee != finishing_agent_name:
                # Nested AgentTool sub-Runner whose agent is not the
                # task's owner — let the outer runner's after_run fire
                # the check against the outer text.
                return
            tool_calls = self._invocation_tool_calls.get(inv_id, 0)
            final_text = self._invocation_last_text.get(inv_id, "")
            from goldfive.drift import classify_confabulation_risk

            drift = classify_confabulation_risk(
                task=task,
                tool_call_count=tool_calls,
                output_text=final_text,
            )
            if drift is None:
                return
            observation = _as_observation(
                kind="confabulation_risk",
                detail=drift.detail,
                raw={
                    "tool_call_count": tool_calls,
                    "output_text": final_text[:500],
                },
                task=task,
                agent_id=finishing_agent_name or self._host_agent_name,
            )
            # Route through steerer.observe so the drift hits the same
            # pipeline AGENT_REFUSAL uses (DriftDetected sink event,
            # severity-based refine decision). We pre-classified above
            # so the steerer's classify_* cascade will no-op on the
            # observation dict — we still fire it explicitly by calling
            # _handle_drift directly when the steerer exposes it.
            handle = getattr(ctx.steerer, "_handle_drift", None)
            if handle is not None:
                try:
                    await handle(drift, ctx.session)
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_maybe_emit_confabulation_risk: _handle_drift raised: %s",
                        exc,
                    )
            # Fallback for steerer stubs without _handle_drift: feed the
            # observation through observe() so custom steerers still
            # see the signal.
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_maybe_emit_confabulation_risk: steerer.observe raised: %s",
                    exc,
                )

        async def _emit_runaway_delegation_drift(
            self,
            *,
            ctx: SessionContext,
            from_agent: str,
            to_agent: str,
            task_id: str,
            invocation_id: str,
            spawn_count: int,
        ) -> None:
            """Emit a ``RUNAWAY_DELEGATION`` drift at CRITICAL severity.

            Built and dispatched directly (not through
            ``steerer.observe`` → ``detect_drift``) because the cap is
            an observed invariant violation, not a heuristic. Routes
            through ``steerer._handle_drift`` when available so the
            planner gets a refine hook; falls back to a direct
            ``_emit_drift_detected`` if the steerer doesn't expose
            ``_handle_drift``. Failures swallowed — observability
            cannot block the invocation, and the adapter's invoke loop
            will break out on ``runaway_delegation_tripped`` regardless.
            """
            steerer = ctx.steerer
            if steerer is None:
                return
            try:
                from goldfive.types import (  # noqa: PLC0415 — lazy
                    DriftEvent,
                    DriftKind,
                    DriftSeverity,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_runaway_delegation_drift: cannot import types: %s",
                    exc,
                )
                return

            detail = (
                f"AgentTool-per-invoke cap of {self._agent_tool_cap} "
                f"exceeded (spawn #{spawn_count}); last delegation "
                f"{from_agent or '?'} -> {to_agent or '?'} at invocation "
                f"{invocation_id or '?'}"
            )
            drift = DriftEvent(
                kind=DriftKind.RUNAWAY_DELEGATION,
                severity=DriftSeverity.CRITICAL,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=from_agent or self._host_agent_name,
            )
            # Prefer _handle_drift so the full refine/emit path fires.
            handle = getattr(steerer, "_handle_drift", None)
            if callable(handle):
                try:
                    await handle(drift, ctx.session)
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_emit_runaway_delegation_drift: _handle_drift raised: %s",
                        exc,
                    )
            # Fallback: direct sink emission.
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    drift_detected_event,
                    emit,
                )

                run_id = str(_safe_attr(ctx.session, "run_id", "") or "")
                session_id = str(_safe_attr(ctx.session, "id", "") or "") or run_id
                try:
                    seq = ctx.session.next_sequence()
                except Exception:  # noqa: BLE001
                    seq = 0
                evt = drift_detected_event(run_id, seq, drift, session_id=session_id)
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_runaway_delegation_drift: direct sink emit failed: %s",
                    exc,
                )

        async def _emit_observability(self, kind: str, **fields: Any) -> None:
            """Fan out an observability event to the session's sinks.

            Sinks live on the steerer (``steerer._sinks``) — same
            channel the approval-requested path uses. Failures are
            swallowed: observability cannot block an invocation.
            """
            ctx = self._active_ctx
            if ctx is None:
                return
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    agent_invocation_completed_event,
                    agent_invocation_started_event,
                    delegation_observed_event,
                    emit,
                )

                # Only thread session_id when caller hasn't supplied it
                # explicitly, so the plugin's stamping stays back-compat
                # with callers that set the field themselves.
                fields.setdefault("session_id", session_id)
                if kind == "agent_invocation_started":
                    evt = agent_invocation_started_event(run_id, seq, **fields)
                elif kind == "agent_invocation_completed":
                    evt = agent_invocation_completed_event(run_id, seq, **fields)
                elif kind == "delegation_observed":
                    evt = delegation_observed_event(run_id, seq, **fields)
                else:
                    return
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug("_emit_observability: failed to emit %s: %s", kind, exc)

        # --- Plan + current-task context -------------------------------

        async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
            """Best-effort re-seed goldfive.* state for legacy test harnesses
            + GoldfivePlanner request-side instruction injection
            + per-LLM-call instrumentation (goldfive#172).

            The authoritative state write happens in
            :meth:`before_run_callback` against the live invocation
            session. This callback retained its historic write-through
            for unit tests that drive the plugin with a minimal
            callback-context stub without a matching ``before_run``
            lifecycle (see ``tests/test_adk_adapter.py``).

            It ALSO performs GoldfivePlanner request-side instruction
            injection (goldfive#153): ADK's ``_nl_planning.py``
            request-side gate fires only for ``PlanReActPlanner``
            subclasses, not every ``BasePlanner``. Since goldfive
            deliberately subclasses ``BasePlanner`` directly (we don't
            want PlanReAct's response filtering), we invoke
            :meth:`GoldfivePlanner.build_planning_instruction` ourselves
            here and append the returned string to
            ``llm_request.config.system_instruction`` via the same
            ``append_instructions`` helper ADK uses internally.

            Finally, it stamps per-LLM-call metrics (goldfive#172):
            counts ``llm_request.contents`` chars and message count,
            stashes a start-time on the plugin keyed by invocation_id
            so the paired ``after_model_callback`` can compute call
            duration, and logs the measurements at INFO with
            structured fields so operators running the live kikuchi
            endpoint can correlate post-steer slowdowns to context
            growth (see issue #172, hypothesis 1).
            """
            ctx = self._resolve_ctx(callback_context)
            if ctx is None:
                return None
            state = _session_state_from_callback(callback_context)
            if not isinstance(state, dict):
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

            # GoldfivePlanner request-side injection (goldfive#153).
            # Best-effort: never raise from this path; injection failure
            # degrades to "LLM runs without goldfive's orchestration
            # context block" which is safe — ADK state still carries
            # the same keys via the write above.
            try:
                await _inject_goldfive_planner_instruction(
                    callback_context=callback_context,
                    llm_request=llm_request,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_model_callback: goldfive planner injection raised: %s",
                    exc,
                )

            # Per-LLM-call instrumentation (goldfive#172). Measure the
            # request AFTER GoldfivePlanner has appended its
            # instruction so the reported chars reflect what the
            # model actually sees. Stash a start-time so the paired
            # after_model_callback can compute duration.
            try:
                chars, messages_count = _measure_request_chars(llm_request)
                inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                    callback_context, "invocation_context", None
                )
                inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
                start_mono = time.monotonic()
                if inv_id:
                    self._invocation_llm_pending[inv_id] = {
                        "start_mono": start_mono,
                        "chars": chars,
                        "messages_count": messages_count,
                    }
                # INFO log so an operator running a live e2e (kikuchi)
                # can tail stderr and correlate context growth against
                # the subsequent duration line.
                log.info(
                    "goldfive.llm.request invocation_id=%s "
                    "llm.request.chars=%d llm.request.messages_count=%d "
                    "task_id=%s agent=%s",
                    inv_id or "?",
                    chars,
                    messages_count,
                    str(_safe_attr(ctx.task, "id", "") or "") or "?",
                    self._host_agent_name or "?",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_model_callback: LLM-call instrumentation raised: %s",
                    exc,
                )
            return None

        # --- Reporting-tool interception + tool-confirmation bridge ---

        async def before_tool_callback(
            self, *, tool: Any, tool_args: Any, tool_context: Any
        ) -> dict[str, Any] | None:
            ctx = self._resolve_ctx(tool_context)
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
            #
            # Route through ``invoke_tool`` (NOT a direct handler call)
            # so every reporting-tool dispatch picks up schema validation
            # (missing / unknown ``task_id`` → structured error) and
            # then reaches the reporting handlers, which own the rest of
            # the protection stack:
            #
            #   * idempotency — same-transition retries return
            #     ``{"acknowledged": True, "idempotent": True, ...}``
            #     (goldfive#201, #203),
            #   * invalid-transition — cross-transitions on terminal
            #     tasks return ``{"acknowledged": False,
            #     "error": "invalid_transition", ...}``,
            #   * tool-loop detection — covered independently by
            #     :class:`goldfive.drift.tool_loops.ToolLoopTracker`
            #     at ``after_tool_callback`` (goldfive#181, #204).
            #
            # See ``docs/design/TASK-LIFECYCLE.md`` §5 for the contract.
            #
            # goldfive#191 Layer 3 — task_id injection. Before we dispatch
            # a reporting tool, if the LLM omitted ``task_id`` (or passed
            # an obvious placeholder like "" / "placeholder" / "unknown"
            # / "TODO") we fall back to the ``goldfive.current_task_id``
            # pinned into session.state by ``before_agent_callback`` at
            # the start of the agent turn. The state write fires first in
            # ADK's callback order (agent-turn → tool-calls-within-turn)
            # so by the time we reach here the pin is already visible.
            # Real-looking task_ids are preserved verbatim — we don't
            # silently rewrite them even if they look wrong (see #191).
            _inject_task_id_from_state(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_context=tool_context,
            )

            tool_names_registered = {spec.name for spec in ctx.tools}
            if tool_name in tool_names_registered:
                args_map: dict[str, Any]
                if isinstance(tool_args, Mapping):
                    args_map = dict(tool_args)
                else:
                    args_map = {}
                try:
                    result = await invoke_tool(
                        ctx.tools,
                        tool_name,
                        args_map,
                        ctx.session,
                        ctx.steerer,
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
                # ``invoke_tool`` always returns a dict; preserve it verbatim
                # so the terminal-rejection / duplicate-ACK payloads reach
                # the agent unchanged.
                if isinstance(result, dict):
                    return result
                return {"acknowledged": True}

            # AgentTool detection. We look both at the tool's class
            # name (avoids an unconditional ADK import on module load)
            # and its ``agent`` attribute so we catch wrapper / subclass
            # shapes. Two jobs here:
            #   1. Emit DelegationObserved (observability).
            #   2. Count toward the per-invocation AgentTool cap
            #      (goldfive#130 runaway-delegation backstop).
            nested_agent = _safe_attr(tool, "agent", None)
            tool_cls_name = type(tool).__name__
            if nested_agent is not None or tool_cls_name == "AgentTool":
                to_agent = str(_safe_attr(nested_agent, "name", "") or "") or tool_name
                from_agent = str(_safe_attr(ctx, "host_agent_name", "") or "")
                task_id = str(_safe_attr(ctx.task, "id", "") or "")
                inv_id = ""
                inv_ctx = _safe_attr(tool_context, "_invocation_context", None)
                if inv_ctx is not None:
                    inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
                await self._emit_observability(
                    "delegation_observed",
                    from_agent=from_agent,
                    to_agent=to_agent,
                    task_id=task_id,
                    invocation_id=inv_id,
                )
                if self._reconciler is not None:
                    try:
                        await self._reconciler.on_delegation_observed(
                            from_agent=from_agent,
                            to_agent=to_agent,
                            invocation_id=inv_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "before_tool_callback: reconciler.on_delegation_observed raised: %s",
                            exc,
                        )

                # Runaway-delegation cap. Count BEFORE short-circuiting
                # so the drift fires exactly once at the threshold
                # crossing; subsequent AgentTool calls in the same
                # invocation return a short-circuit skipped dict so
                # the runner wraps up quickly.
                if self._agent_tool_cap > 0:
                    self._agent_tool_spawn_count += 1
                    if (
                        self._agent_tool_spawn_count > self._agent_tool_cap
                        and not self.runaway_delegation_tripped
                    ):
                        self.runaway_delegation_tripped = True
                        await self._emit_runaway_delegation_drift(
                            ctx=ctx,
                            from_agent=from_agent,
                            to_agent=to_agent,
                            task_id=task_id,
                            invocation_id=inv_id,
                            spawn_count=self._agent_tool_spawn_count,
                        )
                    if self.runaway_delegation_tripped:
                        # Short-circuit the spawn: return a skipped dict
                        # so ADK does not drive the sub-agent. The
                        # adapter's invoke loop notices the tripped
                        # flag between events and breaks out.
                        return {
                            "skipped": True,
                            "reason": "goldfive_runaway_delegation_cap",
                            "tool_name": tool_name,
                            "detail": (
                                f"AgentTool-per-invoke cap of "
                                f"{self._agent_tool_cap} exceeded "
                                f"(spawn #{self._agent_tool_spawn_count})"
                            ),
                        }
                # Fall through: AgentTool still runs, we're just observing.

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

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
            ctx = self._resolve_ctx(callback_context)
            if ctx is None or ctx.steerer is None:
                return None
            texts = _extract_text_parts(llm_response)
            calls = _extract_function_calls(llm_response)
            reasoning = _extract_reasoning(llm_response)
            finish = _safe_attr(llm_response, "finish_reason", None)
            # Feed the per-invocation counters used by the
            # CONFABULATION_RISK check in after_run_callback. We track:
            #   * tool-call count: cumulative across the invocation's
            #     LLM turns so we only fire if NO tool was ever used,
            #   * last non-empty text: used as the output signal — an
            #     invocation that ended with empty text is not
            #     suspicious regardless of tool calls.
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if inv_id:
                if calls:
                    self._invocation_tool_calls[inv_id] = self._invocation_tool_calls.get(
                        inv_id, 0
                    ) + len(calls)
                if texts:
                    joined = " ".join(texts).strip()
                    if joined:
                        self._invocation_last_text[inv_id] = joined
            # Per-LLM-call instrumentation (goldfive#172). Pair with the
            # before_model_callback stash to compute duration, extract
            # token usage, log the result, and enrich the observation
            # raw dict so custom steerer sinks can surface the metrics
            # alongside each LLM turn. Any failure in this block is
            # swallowed — instrumentation must not shadow a real LLM
            # response from the steerer.
            metrics: dict[str, Any] = {}
            try:
                pending = self._invocation_llm_pending.pop(inv_id, None) if inv_id else None
                if pending is not None:
                    duration_ms = int((time.monotonic() - pending["start_mono"]) * 1000)
                    metrics["llm.call.duration_ms"] = duration_ms
                    metrics["llm.request.chars"] = int(pending.get("chars", 0))
                    metrics["llm.request.messages_count"] = int(pending.get("messages_count", 0))
                usage = _extract_usage_metadata(llm_response)
                for key, value in usage.items():
                    metrics[f"llm.usage.{key}"] = value
                if metrics:
                    log.info(
                        "goldfive.llm.response invocation_id=%s "
                        "llm.call.duration_ms=%s llm.request.chars=%s "
                        "llm.request.messages_count=%s "
                        "llm.usage.prompt_tokens=%s "
                        "llm.usage.completion_tokens=%s "
                        "llm.usage.total_tokens=%s "
                        "task_id=%s agent=%s",
                        inv_id or "?",
                        metrics.get("llm.call.duration_ms", "?"),
                        metrics.get("llm.request.chars", "?"),
                        metrics.get("llm.request.messages_count", "?"),
                        metrics.get("llm.usage.prompt_tokens", "?"),
                        metrics.get("llm.usage.completion_tokens", "?"),
                        metrics.get("llm.usage.total_tokens", "?"),
                        str(_safe_attr(ctx.task, "id", "") or "") or "?",
                        self._host_agent_name or "?",
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "after_model_callback: LLM-call instrumentation raised: %s",
                    exc,
                )
            raw: dict[str, Any] = {
                "texts": texts,
                "function_calls": calls,
                "reasoning": reasoning,
                "finish_reason": str(finish) if finish is not None else "",
            }
            if metrics:
                raw["metrics"] = metrics
            observation = _as_observation(
                kind="llm_response",
                detail=" ".join(texts)[:500],
                raw=raw,
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("after_model_callback: steerer.observe raised: %s", exc)
            if reasoning:
                observe_reasoning = getattr(ctx.steerer, "observe_reasoning", None)
                if observe_reasoning is not None:
                    try:
                        await observe_reasoning(
                            reasoning,
                            task=ctx.task,
                            session=ctx.session,
                            provider=_infer_provider(llm_response),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_model_callback: observe_reasoning raised: %s",
                            exc,
                        )
            # Note this turn for the opt-in reflective self-progress check.
            # ``note_llm_call`` is a no-op unless the steerer was
            # constructed with ``reflective_call_llm``; adapters that
            # don't ship this hook simply skip the counter.
            note_llm_call = getattr(ctx.steerer, "note_llm_call", None)
            if note_llm_call is not None:
                try:
                    await note_llm_call(ctx.session)
                except Exception as exc:  # noqa: BLE001
                    log.debug("after_model_callback: note_llm_call raised: %s", exc)
            return None

        async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None or ctx.steerer is None:
                return None
            # Detect transfer / escalation actions on the event payload.
            actions = _safe_attr(event, "actions", None)
            transfer_to = _safe_attr(actions, "transfer_to_agent", "") or ""
            escalate = bool(_safe_attr(actions, "escalate", False))
            if not transfer_to and not escalate:
                return None
            kind = "agent_transfer" if transfer_to else "agent_escalation"
            detail = f"transfer -> {transfer_to}" if transfer_to else "escalate"
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
            ctx = self._resolve_ctx(tool_context)
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
                log.debug("on_tool_error_callback: steerer.observe raised: %s", exc)
            return None

        # --- Tool-loop drift detection (goldfive#181) ------------------

        async def after_tool_callback(
            self,
            *,
            tool: Any,
            tool_args: Any,
            tool_context: Any,
            result: Any,
        ) -> None:
            """Feed the tool-loop tracker and emit any drifts it raises.

            Runs after every tool ADK dispatched (reporting tools,
            AgentTool delegations, MCP/custom tools) so the detector
            sees the real function_call stream the agent is emitting.
            A reporting-tool progress call (``report_task_started`` /
            ``_progress`` / ``_completed`` / ``_failed`` / ``_blocked`` /
            ``report_awaiting_approval``) additionally clears the
            per-(invocation, agent) window — but ONLY when the call
            was acknowledged (``result == {"acknowledged": True, ...}``)
            so mode 2's "no task progress" gate is correct.

            The acknowledged-success gate (goldfive#192) is the
            tightening over goldfive#181's original behaviour: errored
            progress reports (``acknowledged=False`` or responses with
            an ``error`` key) do NOT reset the window, so an agent
            stuck retrying a failing ``report_task_*`` with a bad
            ``task_id`` gets caught as a tool-loop at the normal
            thresholds.

            The detector is deterministic and O(1) per call modulo the
            tracker's ``window`` length; any failure is swallowed so a
            buggy classifier never breaks tool dispatch. Drifts are
            routed through ``steerer._handle_drift`` when available so
            the intervention ladder sees them; falls back to
            ``steerer.observe`` for stubs that don't expose
            ``_handle_drift``.
            """
            ctx = self._resolve_ctx(tool_context)
            if ctx is None or ctx.steerer is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            if not tool_name:
                func = _safe_attr(tool, "func", None)
                tool_name = str(_safe_attr(func, "__name__", "") or "")
            # Resolve invocation_id + agent_name so the tracker's
            # per-(invocation, agent) buckets match the reconciler's
            # isolation model. Missing fields fall back to "" so the
            # tracker still keys consistently on an ephemeral "unknown"
            # bucket (tests exercise this path).
            inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
                tool_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            running_agent = _safe_attr(inv_ctx, "agent", None)
            agent_name = str(_safe_attr(running_agent, "name", "") or "") or self._host_agent_name
            task_id = str(_safe_attr(ctx.task, "id", "") or "")

            # Every tool call is observed — regardless of kind. For a
            # progress-reporting tool (``report_task_*`` /
            # ``report_awaiting_approval``) we THEN look at the
            # ``result`` payload and reset the per-(invocation, agent)
            # window only when the call was *acknowledged* successfully
            # (goldfive#192). Previously the exemption triggered on the
            # call alone, so an agent stuck retrying a failing
            # ``report_task_started`` kept resetting the window and the
            # loop detector never fired. By observing first and resetting
            # only on acknowledged success, errored report_* calls
            # accumulate in the ring buffer and trigger loop detection
            # at the normal thresholds.
            try:
                # tool_args may be None / missing on adapter edge cases;
                # the tracker's hash helper copes with both.
                args_payload = tool_args if isinstance(tool_args, Mapping) else {}
                drifts = self._tool_loop_tracker.observe_tool_call(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    args=dict(args_payload),
                    task_id=task_id,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "after_tool_callback: tool-loop tracker raised: %s",
                    exc,
                )
                return None

            # Post-observation: reset the window only on acknowledged
            # success for progress-reporting tools. An errored report
            # (``acknowledged=False`` or a response containing an
            # ``error`` key) falls through to the regular drift
            # dispatch path so a stuck retry-loop still lights up.
            if tool_name in self._progress_reporting_tools:
                if _is_progress_report_success(result):
                    try:
                        self._tool_loop_tracker.on_task_progress(
                            invocation_id=inv_id,
                            agent_name=agent_name,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_tool_callback: on_task_progress raised: %s",
                            exc,
                        )

            if not drifts:
                return None

            # Prefer ``_handle_drift`` so the intervention ladder
            # sees the signal. Fall back to ``observe`` for stubs.
            handle = getattr(ctx.steerer, "_handle_drift", None)
            for drift in drifts:
                if handle is not None:
                    try:
                        await handle(drift, ctx.session)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_tool_callback: _handle_drift raised: %s",
                            exc,
                        )
                # Fallback path for steerer stubs that don't expose
                # _handle_drift.
                observation = _as_observation(
                    kind="tool_loop_detected",
                    detail=drift.detail,
                    raw=drift.raw if isinstance(drift.raw, dict) else {"drift": repr(drift.raw)},
                    task=ctx.task,
                    agent_id=agent_name or self._host_agent_name,
                )
                try:
                    await ctx.steerer.observe(observation, ctx.session)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "after_tool_callback: steerer.observe raised: %s",
                        exc,
                    )
            return None

    return _GoldfiveADKPlugin()
