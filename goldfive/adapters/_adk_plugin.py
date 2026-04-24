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


def _agent_has_pending_candidates(ctx: Any, agent_name: str) -> bool:
    """Return True if the plan has any PENDING/RUNNING task for ``agent_name``.

    Used to distinguish the two failure modes when the reporting-tool
    ``task_id`` pin cannot be resolved (goldfive#250 follow-up):

    * **No candidates** — the agent's work was moved to other agents by
      a plan refine; a silent-ack no-op is correct.
    * **Has candidates** — the pin SHOULD have worked (single match, or
      delegation-site pin, or fallback). The tool response is still
      ``{"acknowledged": True}`` so the LLM can't pattern-match on an
      error shape (observed live: models read ``"error": "pin_unresolved"``
      as a reasoning cue and bypass the reporting contract). Operator
      visibility is preserved via a WARNING log + a ``DriftDetected``
      sink event; see :meth:`_emit_pin_unresolved_drift`.

    NOTE: the DAG-readiness gate from
    :meth:`_pin_current_task_id_for_agent` is intentionally NOT applied
    here. Even a DAG-gated candidate means the agent's turn shouldn't
    be happening yet, which is also a stall worth surfacing rather than
    silencing.

    Silent on every failure (missing ctx / plan / non-iterable tasks);
    the caller falls back to the conservative silent-ack path when this
    returns ``False``.
    """
    if not agent_name:
        return False
    try:
        plan = _safe_attr(ctx, "session", None)
        plan = _safe_attr(plan, "plan", None) if plan is not None else None
        if plan is None:
            return False
        tasks = _safe_attr(plan, "tasks", None) or ()
        from goldfive.types import TaskStatus

        for task in tasks:
            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
            if assignee != agent_name:
                continue
            status = _safe_attr(task, "status", None)
            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                return True
    except Exception:  # noqa: BLE001 — diagnostic-only
        return False
    return False


def _inject_task_id_from_state(
    *,
    tool_name: str,
    tool_args: Any,
    tool_context: Any,
) -> bool:
    """Populate ``tool_args['task_id']`` from state for reporting tools.

    goldfive#241 — ``task_id`` is hidden from the LLM-facing reporting-
    tool schema (see :func:`goldfive.adapters.adk._apply_llm_signature`),
    so the model cannot supply it. Every reporting-tool call lands here
    with no ``task_id`` arg; this function is the authoritative
    resolution layer.

    Resolution order, keyed by the invocation's ``function_call_id``:

    1. ``goldfive.pending_delegations[<function_call_id>]`` — the
       delegation-site pin stamped by :meth:`before_tool_callback` for
       the AgentTool dispatch that spawned the current sub-invocation.
       This path handles coordinators that fire multiple parallel
       AgentTool calls to the same sub-agent on the same turn — each
       parallel dispatch gets its own pin rather than racing on the
       single ``goldfive.current_task_id`` slot.
    2. ``session.state[goldfive.current_task_id]`` — the agent-turn
       pin written by ``before_agent_callback`` at the start of every
       agent invocation (goldfive#191/#195).

    Returns ``True`` when ``tool_args`` now carries a usable ``task_id``
    (either pre-existing non-placeholder, or freshly populated from
    state) and ``False`` when no pin is available — in that case the
    caller short-circuits with a bare ``{"acknowledged": True}`` and
    emits operator observability (WARNING log + ``DriftDetected`` sink
    event) rather than letting the handler run on an unpinned call.
    Pre-#241 the call would have fallen through to the handler's
    ``missing_task_id`` error, but that path goes back to the LLM via a
    successful tool response shape and confused the model into retry
    loops. Post-#252-followup, even the structured-error shape is gone
    from the LLM-visible response — see :meth:`_emit_pin_unresolved_drift`.

    NEVER raises — any internal failure degrades to "no pin".
    """
    try:
        if not _is_reporting_tool_name(tool_name):
            return True
        if not isinstance(tool_args, MutableMapping):
            return True
        existing = tool_args.get("task_id", "")
        if not _is_placeholder_task_id(existing):
            # A real-looking id was supplied (e.g. legacy caller,
            # custom tool that didn't opt into the hidden schema).
            # Leave it alone so the handler surfaces mismatches as
            # terminal-task / not-found failures rather than silently
            # re-targeting the call.
            return True
        resolved = _resolve_pinned_task_id(
            tool_context=tool_context,
        )
        if not resolved:
            return False
        tool_args["task_id"] = resolved
        return True
    except Exception:  # noqa: BLE001
        log.debug(
            "before_tool_callback: task_id injection failed for tool=%s",
            tool_name,
            exc_info=True,
        )
        return False


# State-key used to stash per-function_call_id task pins at the
# delegation site (goldfive#241 Item 3-bis). The plugin's
# ``before_tool_callback`` writes an entry here when it dispatches an
# AgentTool call and the reporting-tool callback reads it back when
# the sub-invocation's tool call arrives. Keyed by the ADK
# ``function_call_id`` of the AgentTool dispatch so parallel calls
# don't race.
_PENDING_DELEGATIONS_KEY = "goldfive.pending_delegations"


def _resolve_pinned_task_id(*, tool_context: Any) -> str:
    """Return the task_id pinned for this tool invocation, or ``""``.

    Consults the delegation-site map first (``pending_delegations``
    keyed by the current invocation's ``function_call_id``), then
    falls back to the agent-turn pin (``goldfive.current_task_id``).
    Returns ``""`` when neither path yields a value — the caller
    should then short-circuit with a bare ``{"acknowledged": True}``
    (plus a ``DriftDetected`` sink event for operator visibility on
    the has-candidates branch) rather than invoking the handler on an
    unpinned arg. Pre-#252-followup the has-candidates branch emitted
    an ``error: pin_unresolved`` payload, which the LLM read as a
    reasoning cue and used to bypass the reporting contract — see
    the pin-leak fix in that PR.
    """
    state = _session_state_from_callback(tool_context)
    if not isinstance(state, Mapping):
        return ""
    fc_id = _function_call_id_from_tool_context(tool_context)
    if fc_id:
        pend = state.get(_PENDING_DELEGATIONS_KEY)
        if isinstance(pend, Mapping):
            raw = pend.get(fc_id, "")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    state_tid = state.get(_sp.KEY_CURRENT_TASK_ID, "")
    if isinstance(state_tid, str) and state_tid.strip():
        return state_tid.strip()
    return ""


def _function_call_id_from_tool_context(tool_context: Any) -> str:
    """Best-effort extraction of the current ``function_call_id``.

    ADK's ToolContext carries the active ``function_call_id`` directly;
    legacy / test stubs may not have the attribute, in which case we
    degrade to ``""`` (falls back to the agent-turn pin).
    """
    fc_id = _safe_attr(tool_context, "function_call_id", "")
    if isinstance(fc_id, str) and fc_id.strip():
        return fc_id.strip()
    return ""


def _tokenize_for_matching(text: Any) -> set[str]:
    """Return the set of lowercase alphanumeric tokens of length ≥4.

    Used by :func:`_score_candidates_by_args` to score candidate tasks
    against AgentTool args. The ≥4 threshold filters out noisy
    short-word matches ("in", "of", "the") that would otherwise
    saturate every candidate's score.
    """
    if not isinstance(text, str):
        text = str(text or "")
    tokens: set[str] = set()
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tok = "".join(buf)
                if len(tok) >= 4:
                    tokens.add(tok)
                buf.clear()
    if buf:
        tok = "".join(buf)
        if len(tok) >= 4:
            tokens.add(tok)
    return tokens


def _score_candidates_by_args(candidates: list[Any], tool_args: Any) -> Any:
    """Return the best-scoring candidate, or ``None`` on tie / zero match.

    Scoring: tokenise ``tool_args`` and each candidate's
    ``title + description``. Candidate with the highest overlap wins.
    Ties (two or more candidates with the same top non-zero score)
    return ``None`` so the caller falls through to the no-pin path;
    guessing would be worse than letting the sub-agent path handle
    the ambiguity.
    """
    if not candidates:
        return None
    # Serialise args into a single token bag. Keys contribute too
    # (``topic=solar`` contributes "topic" and "solar") because the
    # key names often hint at which task the call is about.
    args_text = ""
    if isinstance(tool_args, Mapping):
        parts: list[str] = []
        for k, v in tool_args.items():
            parts.append(str(k))
            parts.append(str(v))
        args_text = " ".join(parts)
    elif isinstance(tool_args, str):
        args_text = tool_args
    arg_tokens = _tokenize_for_matching(args_text)
    if not arg_tokens:
        return None
    best_score = 0
    best: Any = None
    tied = False
    for cand in candidates:
        title = str(_safe_attr(cand, "title", "") or "")
        desc = str(_safe_attr(cand, "description", "") or "")
        cand_tokens = _tokenize_for_matching(f"{title} {desc}")
        score = len(arg_tokens & cand_tokens)
        if score > best_score:
            best_score = score
            best = cand
            tied = False
        elif score == best_score and score > 0:
            tied = True
    if best_score == 0 or tied:
        return None
    return best


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


def _bridge_pending_corrections(gf_state: Any, adk_state: Any) -> None:
    """Mirror every ``goldfive.pending_corrections.*`` key from ``gf_state`` onto ``adk_state``.

    goldfive#251 Stream D. Structural (prefix-scoped) bridge rather than
    a per-key copy because the correction family is ``(agent,
    task)``-expanded — goldfive doesn't know the set of keys at
    plugin-init time. Idempotent: re-running on the same pair of states
    restores whatever ``gf_state`` has NOW and evicts ADK entries
    ``gf_state`` no longer carries. That eviction is the mechanism by
    which a :func:`goldfive._correction_injection.clear_correction`
    call on the orchestration state reaches the dynamic instruction
    resolver on the ADK side.

    Called from :meth:`make_adk_plugin`'s inner
    ``_bridge_orchestration_state`` once per ``before_run_callback``;
    also directly unit-testable as a module-level helper with any pair
    of dicts.

    Silent on non-mapping inputs so one malformed state never blocks
    the rest of the bridge.
    """
    if not isinstance(adk_state, MutableMapping):
        return
    prefix = _sp.KEY_PENDING_CORRECTIONS + "."
    # Snapshot the gf-side keys first so mutation during the loop is
    # impossible.
    gf_keys: dict[str, Any] = {}
    if isinstance(gf_state, Mapping):
        for k, v in gf_state.items():
            if isinstance(k, str) and k.startswith(prefix):
                gf_keys[k] = v
    # Evict any ADK-side correction key that no longer exists on the
    # goldfive side. This is the mechanism by which a clear on
    # gf_state reaches the resolver (the resolver reads the ADK copy
    # via ``readonly_context.state``).
    for k in list(adk_state.keys()):
        if isinstance(k, str) and k.startswith(prefix) and k not in gf_keys:
            adk_state.pop(k, None)
    # Copy fresh values through.
    for k, v in gf_keys.items():
        adk_state[k] = v


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
            # trips. Thresholds are sourced from
            # :func:`~goldfive.drift.tool_loops.resolve_thresholds`,
            # which prefers an installed
            # :class:`~goldfive.config.ToolLoopConfig` (goldfive#225,
            # wired by :func:`goldfive.wrap`) and falls back to
            # ``GOLDFIVE_TOOL_LOOP_*`` env vars and then the module
            # defaults. Lazy import so the plugin module stays
            # importable without the drift helpers materialised —
            # matches the pattern used for the confabulation classifier.
            from goldfive.drift import tool_loops as _tool_loops

            self._tool_loop_tracker = _tool_loops.ToolLoopTracker(
                **_tool_loops.resolve_thresholds()
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
            # Cooperative cancellation state (goldfive#251 Stream C / 7a).
            # Authoritative source for the cancel-requested flag:
            # ``dict[str, CancellationRequest]`` keyed by ``invocation_id``.
            # The steerer writes entries here via
            # :meth:`request_invocation_cancel` on the adapter; every
            # adapter callback checks the dict at the top of its body
            # and, when an entry matches the current invocation_id,
            # consumes it (read + clear — same-invocation re-entry won't
            # re-cancel) and short-circuits the dispatch.
            #
            # Stored on the plugin instance rather than ADK session.state
            # because ``InMemorySessionService`` shallow-copies state on
            # every ``get_session`` (see the same rationale that drove
            # goldfive#170 for the _active_ctx field), which would make
            # cross-callback reads unreliable. The state-protocol module
            # (:data:`_adk_state_protocol.KEY_CANCEL_REQUESTED`) documents
            # the key semantics so external consumers see a stable
            # contract; the plugin's dict is the live source of truth.
            self._cancel_state: dict[str, Any] = {}
            # Parent/child invocation map for cancel propagation.
            # ``dict[str, str]`` mapping ``invocation_id ->
            # parent_invocation_id``. Populated on every
            # ``before_run_callback`` that observes a fresh invocation_id
            # with a known parent (the top-level invocation_id pinned on
            # the first ``before_run``). Consumed by
            # :meth:`request_invocation_cancel` so that cancelling a
            # parent also flags its spawned sub-invocations — this is
            # how a cancelled coordinator's mid-flight AgentTool child
            # short-circuits cleanly instead of emitting its turn and
            # poisoning the parent's history.
            self._invocation_parents: dict[str, str] = {}
            # Per-invocation pinned task_id (goldfive#264 — aggressive
            # pin resolution). ``dict[str, str]`` mapping
            # ``invocation_id -> task_id`` populated by
            # :meth:`_stamp_current_task_id` whenever a pin lands.
            # Consumed by signal 5 of :meth:`_pin_current_task_id_for_agent`
            # so a child invocation can read its parent's pinned task
            # without racing on the single ``goldfive.current_task_id``
            # slot. Cleared on ``clear_active_context``.
            self._invocation_pinned_task_id: dict[str, str] = {}

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
            # Drop any lingering cancellation state / parent map so the
            # next dispatch starts clean (goldfive#251). A request that
            # was never consumed means the callback path never ran —
            # still safe to drop because the invocation it targeted is
            # gone, and keeping it would misfire on an unrelated future
            # invocation_id collision.
            self._cancel_state.clear()
            self._invocation_parents.clear()
            # goldfive#264 — drop per-invocation pin map.
            self._invocation_pinned_task_id.clear()

        # --- Cooperative cancellation (goldfive#251 Stream C / 7a) -----

        def request_invocation_cancel(
            self,
            *,
            invocation_id: str,
            request: Any,
            propagate_to_children: bool = True,
        ) -> list[str]:
            """Flag ``invocation_id`` (and optionally its descendants)
            for cooperative cancellation.

            Called by :class:`~goldfive.steerer.DefaultSteerer` when a
            drift at CRITICAL severity (or a user-initiated cancel)
            warrants aborting an in-flight adapter dispatch. Writes an
            entry to the plugin's ``_cancel_state`` dict keyed by the
            invocation id; every adapter callback consults the dict at
            the top of its body and short-circuits when its own id
            matches.

            When ``propagate_to_children`` is True (the default), the
            recorded parent/child map is walked breadth-first and an
            entry is added for every transitive descendant of
            ``invocation_id`` so an in-flight AgentTool sub-invocation
            is also flagged. The returned list contains every id that
            was flagged — the target itself plus any descendants — so
            callers can sink-report the full set if they want.

            Tree-agnostic: the parent/child map is per-invocation, the
            plugin has no notion of "coordinator" vs "sub-agent", and
            every level in the tree is flagged the same way.
            """
            if not invocation_id:
                return []
            flagged: list[str] = [str(invocation_id)]
            # Walk the parent/child map when propagation is enabled.
            # Order is unspecified; deduplication happens inside
            # ``descendants_of_invocation`` via a seen-set.
            if propagate_to_children:
                try:
                    from goldfive.adapters import _adk_state_protocol as _sp_local

                    descendants = _sp_local.descendants_of_invocation(
                        {_sp_local.KEY_INVOCATION_PARENTS: self._invocation_parents},
                        invocation_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_GoldfiveADKPlugin.request_invocation_cancel: "
                        "descendant walk raised: %s",
                        exc,
                    )
                    descendants = []
                flagged.extend(descendants)
            for flagged_id in flagged:
                # Preserve the first-writer's request for each id so a
                # parent cancel with reason="user_steer" doesn't get
                # silently overwritten by a descendant-propagation pass
                # reusing the parent's request object. When a descendant
                # already has a distinct request pending (uncommon but
                # possible), keep the earlier one — the more-recent
                # overwrite semantics are only for same-id re-writes
                # from the steerer itself.
                self._cancel_state.setdefault(flagged_id, request)
            return flagged

        def consume_cancel_for_invocation(self, invocation_id: str) -> Any | None:
            """Read the pending cancel for ``invocation_id`` and clear it.

            Callback-facing helper. Returns the
            :class:`~goldfive.types.CancellationRequest` when one was
            pending, or ``None`` otherwise. Clearing before returning
            gives the "cancel fires once" semantic: a re-entry into the
            same callback (e.g. after the LLM call was already skipped
            and a follow-up tool call fires) doesn't re-emit the
            cancelled marker.
            """
            if not invocation_id:
                return None
            return self._cancel_state.pop(str(invocation_id), None)

        def peek_cancel_for_invocation(self, invocation_id: str) -> Any | None:
            """Return the pending cancel for ``invocation_id`` without
            clearing it.

            Diagnostic / test helper. Production callback paths use
            :meth:`consume_cancel_for_invocation`; this method exists so
            the adapter's ``invoke`` loop can check whether a cancel was
            flagged on the current dispatch without side-effecting the
            consume-once semantic.
            """
            if not invocation_id:
                return None
            return self._cancel_state.get(str(invocation_id))

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

            # Register the parent/child relationship for cooperative
            # cancellation propagation (goldfive#251). A cancel targeting
            # the parent id can then flag this child id without the
            # steerer having to know the tree shape.
            if inv_id and parent_inv_id:
                self._invocation_parents[inv_id] = parent_inv_id

            # Cooperative-cancellation check (goldfive#251 Stream C / 7a).
            # If the adapter / steerer flagged this invocation before
            # ``run_async`` actually yielded to ADK, short-circuit the
            # whole dispatch: skip the state-protocol write, skip the
            # AgentInvocationStarted emit, and emit an
            # InvocationCancelled sink event so operators see the
            # cancel in harmonograf. The outer ``ADKAdapter.invoke``
            # loop honours the short-circuit via
            # :meth:`peek_cancel_for_invocation` (below).
            if inv_id and self._cancel_state.get(inv_id) is not None:
                request = self.consume_cancel_for_invocation(inv_id)
                await self._emit_invocation_cancelled(
                    invocation_id=inv_id,
                    agent_name="",
                    request=request,
                )
                return None

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

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a).
            # When a cancel was flagged for this invocation_id (by the
            # steerer's CRITICAL-severity ladder path, or by a programmatic
            # caller), consume the request, emit an InvocationCancelled
            # sink event, and short-circuit the callback — the agent's
            # turn is skipped entirely. Done BEFORE the pinning /
            # reconciler forward work so a cancelled turn leaves no
            # side-effects on orchestration state.
            if inv_id and self._cancel_state.get(inv_id) is not None:
                request = self.consume_cancel_for_invocation(inv_id)
                await self._emit_invocation_cancelled(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    request=request,
                )
                return None

            # Layer 1: pin the starting sub-agent's task_id so its
            # reporting-tool calls can default the arg from state
            # (goldfive#191). Best-effort: a raise here must never
            # break the invocation. goldfive#264 — multi-signal
            # resolution, async to allow the PinResolved sink emit.
            try:
                await self._pin_current_task_id_for_agent(
                    agent_name=agent_name,
                    callback_context=callback_context,
                    invocation_id=inv_id,
                    parent_invocation_id=parent_inv_id,
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

        async def _pin_current_task_id_for_agent(
            self,
            *,
            agent_name: str,
            callback_context: Any,
            invocation_id: str = "",
            parent_invocation_id: str = "",
        ) -> None:
            """Aggressive multi-signal pin resolution (goldfive#264).

            Reframed from the original "exactly-1 DAG-ready single
            match" gate after live operator feedback: *if an agent was
            invoked, something precipitated the call*. The previous
            implementation gave up silently on zero/multiple matches,
            so the agent ran without a pin, every reporting-tool call
            short-circuited as a no-op, and the orchestration loop
            stagnated.

            Replaced with an 8-signal resolution ladder, picking the
            first signal that yields a single best candidate:

            1. **Delegation-site pin** — the parent's ``before_tool_callback``
               already stamped a per-function_call_id pin on
               ``pending_delegations``. Authoritative.
            2. **DAG-ready exactly-1** — assignee match, status PENDING /
               RUNNING, all upstream predecessors COMPLETED. The pre-
               existing happy path; preserved as the fast short-circuit.
            3. **Tool-arg scoring over DAG-ready candidates** — when (2)
               returns 2+ matches, score each against the parent
               AgentTool's args via :func:`_score_candidates_by_args`
               and pick the winner. Falls through on tie.
            4. **DAG gate relaxed** — drop the upstream-completion check
               and retry the assignee+status filter. The agent was
               invoked so something precipitated it; surface the pin
               and emit a WARNING + low-confidence sink event so
               operators see the anomaly. Tool-arg scoring breaks ties.
            5. **Parent-pin downstream** — if a parent invocation has a
               pinned task on this plugin, prefer candidates whose id
               is a downstream of the parent's pinned task in
               ``plan.edges``.
            6. **Recent drift / correction targeting** — if a
               ``goldfive.pending_corrections.<agent>.<task_id>`` entry
               exists in session state, pin the named task. The plan-
               revision pipeline writes these for CORRECT-kind
               supersedes; pinning to one is a strong signal that the
               agent was invoked specifically to act on the correction.
            7. **Assignee normalization fallback** — re-run signals
               2-4 with bare/compound forms of ``agent_name`` swapped
               in. PR #215 fixed planner-side normalisation; this is
               defence-in-depth for transcripts that retain a compound
               assignee.
            8. **Low-confidence best-guess** — if every prior signal
               failed, pick the highest-scoring candidate from the
               full assignee+status set (or, lacking any, the highest-
               scoring PENDING/RUNNING task in the plan) and emit a
               ``pin_resolved_low_confidence`` sink event so the LLM
               gets to continue and the operator sees the weakening.

            Every successful pin emits a single ``pin_resolved`` (dict-
            envelope) sink event labelled with ``via_signal`` so
            harmonograf and operators can chart how often the happy
            path is short-circuiting vs. how often the relaxed signals
            are firing — a leading indicator that pin invariants are
            weakening.

            Silent on every failure mode (no agent name, no ctx, no
            plan, state not a mapping). Never raises.
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
            from goldfive.types import TaskStatus, task_upstream_ready

            tasks_list = list(tasks)

            # ---- Signal 1: delegation-site pin -------------------------
            # goldfive#241 Item 3-bis — delegation-site pin takes
            # precedence. If the parent coordinator's
            # ``before_tool_callback`` stamped a task_id for THIS
            # AgentTool dispatch (keyed by function_call_id), trust it.
            gf_state_early = _safe_attr(ctx.session, "state", None)
            if isinstance(gf_state_early, Mapping):
                pend = gf_state_early.get(_PENDING_DELEGATIONS_KEY)
                if isinstance(pend, Mapping) and pend:
                    for tid in pend.values():
                        if not isinstance(tid, str) or not tid:
                            continue
                        for task in tasks_list:
                            if str(_safe_attr(task, "id", "") or "") != tid:
                                continue
                            assignee = str(
                                _safe_attr(task, "assignee_agent_id", "") or ""
                            )
                            if assignee != agent_name:
                                continue
                            status = _safe_attr(task, "status", None)
                            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                                self._stamp_current_task_id(
                                    ctx=ctx,
                                    callback_context=callback_context,
                                    task_id=tid,
                                    agent_name=agent_name,
                                    source="delegation_pin",
                                    task=task,
                                    invocation_id=invocation_id,
                                )
                                await self._emit_pin_resolved(
                                    ctx=ctx,
                                    agent_name=agent_name,
                                    task_id=tid,
                                    via_signal="delegation_pin",
                                    score=1.0,
                                    invocation_id=invocation_id,
                                    candidate_count=1,
                                )
                                return

            # Build the assignee+status candidate set once; signals 2-4
            # all reuse it. We also keep the parent's tool args (best-
            # effort) so signals 3 / 4 / 8 can score candidates.
            assignee_candidates = self._candidates_for_agent(
                tasks_list, agent_name
            )
            scoring_args = self._scoring_args_for(
                ctx=ctx,
                callback_context=callback_context,
                parent_invocation_id=parent_invocation_id,
            )

            # ---- Signal 2: DAG-ready exactly-1 -------------------------
            dag_ready = self._filter_dag_ready(
                plan, assignee_candidates, task_upstream_ready
            )
            if len(dag_ready) == 1:
                task = dag_ready[0]
                task_id = str(_safe_attr(task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        callback_context=callback_context,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="single_match",
                        task=task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="dag_ready_single",
                        score=1.0,
                        invocation_id=invocation_id,
                        candidate_count=1,
                    )
                    return

            # ---- Signal 3: tool-arg scoring over DAG-ready -------------
            if len(dag_ready) > 1 and scoring_args is not None:
                chosen = _score_candidates_by_args(dag_ready, scoring_args)
                if chosen is not None:
                    task_id = str(_safe_attr(chosen, "id", "") or "")
                    if task_id:
                        self._stamp_current_task_id(
                            ctx=ctx,
                            callback_context=callback_context,
                            task_id=task_id,
                            agent_name=agent_name,
                            source="arg_scored",
                            task=chosen,
                            invocation_id=invocation_id,
                        )
                        await self._emit_pin_resolved(
                            ctx=ctx,
                            agent_name=agent_name,
                            task_id=task_id,
                            via_signal="arg_scored",
                            score=1.0,
                            invocation_id=invocation_id,
                            candidate_count=len(dag_ready),
                        )
                        return

            # ---- Signal 4: DAG gate relaxed ----------------------------
            # The user's reframe: "if an agent was invoked, something
            # precipitated it." We've lost ground truth already; bind
            # to the most-plausible task and surface the anomaly to
            # operators rather than silent-no-op.
            if assignee_candidates:
                relaxed = assignee_candidates
                if len(relaxed) > 1 and scoring_args is not None:
                    chosen = _score_candidates_by_args(relaxed, scoring_args)
                    if chosen is None:
                        # Tie / no overlap — fall through.
                        chosen = None
                else:
                    chosen = relaxed[0] if len(relaxed) == 1 else None
                if chosen is not None:
                    task_id = str(_safe_attr(chosen, "id", "") or "")
                    if task_id:
                        log.warning(
                            "pin: DAG-gate relaxed, bound %s for agent %s "
                            "(upstreams not yet complete; %d assignee candidates)",
                            task_id,
                            agent_name,
                            len(relaxed),
                        )
                        self._stamp_current_task_id(
                            ctx=ctx,
                            callback_context=callback_context,
                            task_id=task_id,
                            agent_name=agent_name,
                            source="dag_relaxed",
                            task=chosen,
                            invocation_id=invocation_id,
                        )
                        await self._emit_pin_resolved(
                            ctx=ctx,
                            agent_name=agent_name,
                            task_id=task_id,
                            via_signal="dag_relaxed",
                            score=0.7,
                            invocation_id=invocation_id,
                            candidate_count=len(relaxed),
                        )
                        return

            # ---- Signal 5: parent-pin downstream -----------------------
            parent_pin_task = self._task_from_parent_pin_downstream(
                plan=plan,
                tasks=tasks_list,
                parent_invocation_id=parent_invocation_id,
                agent_name=agent_name,
                scoring_args=scoring_args,
            )
            if parent_pin_task is not None:
                task_id = str(_safe_attr(parent_pin_task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        callback_context=callback_context,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="parent_pin_downstream",
                        task=parent_pin_task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="parent_pin_downstream",
                        score=0.6,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # ---- Signal 6: recent drift / correction targeting --------
            correction_task = self._task_from_pending_correction(
                ctx=ctx,
                tasks=tasks_list,
                agent_name=agent_name,
            )
            if correction_task is not None:
                task_id = str(_safe_attr(correction_task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        callback_context=callback_context,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="correction_target",
                        task=correction_task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="correction_target",
                        score=0.9,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # ---- Signal 7: assignee bare/compound normalisation -------
            normalised_alt = self._alternate_agent_name_form(agent_name)
            if normalised_alt and normalised_alt != agent_name:
                alt_assignee = self._candidates_for_agent(
                    tasks_list, normalised_alt
                )
                if alt_assignee:
                    alt_dag_ready = self._filter_dag_ready(
                        plan, alt_assignee, task_upstream_ready
                    )
                    chosen = None
                    if len(alt_dag_ready) == 1:
                        chosen = alt_dag_ready[0]
                    elif len(alt_dag_ready) > 1 and scoring_args is not None:
                        chosen = _score_candidates_by_args(alt_dag_ready, scoring_args)
                    elif len(alt_dag_ready) == 0 and len(alt_assignee) == 1:
                        chosen = alt_assignee[0]
                    elif len(alt_assignee) > 1 and scoring_args is not None:
                        chosen = _score_candidates_by_args(alt_assignee, scoring_args)
                    if chosen is not None:
                        task_id = str(_safe_attr(chosen, "id", "") or "")
                        if task_id:
                            log.warning(
                                "pin: assignee normalisation %r->%r "
                                "found candidate %s",
                                agent_name,
                                normalised_alt,
                                task_id,
                            )
                            self._stamp_current_task_id(
                                ctx=ctx,
                                callback_context=callback_context,
                                task_id=task_id,
                                agent_name=agent_name,
                                source="assignee_normalised",
                                task=chosen,
                                invocation_id=invocation_id,
                            )
                            await self._emit_pin_resolved(
                                ctx=ctx,
                                agent_name=agent_name,
                                task_id=task_id,
                                via_signal="assignee_normalised",
                                score=0.5,
                                invocation_id=invocation_id,
                                candidate_count=len(alt_assignee),
                            )
                            return

            # ---- Signal 8: low-confidence best-guess -------------------
            best_guess = self._low_confidence_best_guess(
                tasks=tasks_list,
                agent_name=agent_name,
                scoring_args=scoring_args,
            )
            if best_guess is not None:
                task, score = best_guess
                task_id = str(_safe_attr(task, "id", "") or "")
                if task_id:
                    log.warning(
                        "pin: low-confidence best-guess %s for agent %s "
                        "(score=%.2f); every prior signal failed",
                        task_id,
                        agent_name,
                        score,
                    )
                    self._stamp_current_task_id(
                        ctx=ctx,
                        callback_context=callback_context,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="low_confidence",
                        task=task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="low_confidence",
                        score=score,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # All signals failed AND no best-guess candidate at all
            # (empty plan / agent has nothing remotely matching). Leave
            # state unset — there's nothing better than the existing
            # ``missing_task_id`` error path here.
            log.debug(
                "before_agent_callback: pin resolution exhausted all "
                "signals for agent %s; leaving state unset",
                agent_name,
            )

        # ---- Pin resolution helpers (goldfive#264) --------------------

        @staticmethod
        def _candidates_for_agent(tasks: list[Any], agent_name: str) -> list[Any]:
            """Return PENDING/RUNNING tasks whose assignee matches ``agent_name``.

            Pre-DAG candidate set used by signals 2/3/4/8. Pure helper,
            no side effects.
            """
            from goldfive.types import TaskStatus

            out: list[Any] = []
            for task in tasks:
                assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                if assignee != agent_name:
                    continue
                status = _safe_attr(task, "status", None)
                if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                    out.append(task)
            return out

        @staticmethod
        def _filter_dag_ready(
            plan: Any,
            candidates: list[Any],
            task_upstream_ready: Any,
        ) -> list[Any]:
            """Filter ``candidates`` to those whose upstream is COMPLETED."""
            ready: list[Any] = []
            for task in candidates:
                task_id = str(_safe_attr(task, "id", "") or "")
                if not task_id:
                    continue
                try:
                    if task_upstream_ready(plan, task_id):
                        ready.append(task)
                except Exception as exc:  # noqa: BLE001 — never raise from pin
                    log.debug(
                        "_filter_dag_ready: task_upstream_ready raised "
                        "for %s: %s — treating as not-ready",
                        task_id,
                        exc,
                    )
            return ready

        def _scoring_args_for(
            self,
            *,
            ctx: SessionContext,
            callback_context: Any,
            parent_invocation_id: str,
        ) -> Any:
            """Return a token-bag string for tool-arg scoring, or ``None``.

            Signal 3/4/8 score candidates by token overlap with whatever
            we have for "what was this agent invoked for". In priority
            order:

            1. Parent invocation's last AgentTool args, if we have a
               record (best signal — exact dispatch payload).
            2. The active steer body — operators usually phrase the
               steer with task-named tokens.
            3. The session's goal summary — broad fallback that at
               least disambiguates by domain vocabulary.

            Returns whatever non-empty string-or-mapping we found, or
            ``None`` when there's nothing to score against (in which
            case the score-based signals fall through silently).
            """
            # 1) Parent invocation's last AgentTool args. The plugin's
            # before_tool_callback already tracks pending_delegations
            # keyed by function_call_id — that's a per-dispatch pin,
            # not a per-invocation tool-args record. Without a richer
            # record we synthesise the best we can: the steer body or
            # the active-steer-targeting drift detail tends to carry
            # the same vocabulary as the dispatch.
            session_state = _safe_attr(ctx.session, "state", None)
            if isinstance(session_state, Mapping):
                # Active steer body is a strong signal when present.
                steer_body = session_state.get("goldfive.active_steer.body", "")
                if isinstance(steer_body, str) and steer_body.strip():
                    return steer_body
                goals_summary = session_state.get("goldfive.goals_summary", "")
                if isinstance(goals_summary, str) and goals_summary.strip():
                    return goals_summary
            # 2) Goals on the session itself (some test harnesses don't
            # populate the orchestration-state mirror).
            goals = _safe_attr(ctx.session, "goals", None) or []
            if goals:
                summaries = [
                    str(_safe_attr(g, "summary", "") or "") for g in goals
                ]
                joined = " ".join(s for s in summaries if s).strip()
                if joined:
                    return joined
            # 3) Nothing to score against.
            _ = parent_invocation_id  # intentionally unused; kept for future plumbing
            return None

        def _task_from_parent_pin_downstream(
            self,
            *,
            plan: Any,
            tasks: list[Any],
            parent_invocation_id: str,
            agent_name: str,
            scoring_args: Any,
        ) -> Any:
            """Signal 5 — pick a candidate downstream of the parent's pin.

            Reads ``self._invocation_pinned_task_id[parent_invocation_id]``
            to find the parent's pin, then scans ``plan.edges`` for
            tasks whose id is a downstream of the parent pin. Among
            those, restrict to assignee-matching PENDING/RUNNING tasks
            (re-using :meth:`_candidates_for_agent`). If multiple,
            fall back to tool-arg scoring; if zero, return ``None``.
            """
            if not parent_invocation_id:
                return None
            parent_pin = self._invocation_pinned_task_id.get(parent_invocation_id, "")
            if not parent_pin:
                return None
            edges = _safe_attr(plan, "edges", None) or ()
            downstream_ids: set[str] = set()
            for e in edges:
                from_id = str(_safe_attr(e, "from_task_id", "") or "")
                if from_id != parent_pin:
                    continue
                to_id = str(_safe_attr(e, "to_task_id", "") or "")
                if to_id:
                    downstream_ids.add(to_id)
            if not downstream_ids:
                return None
            assignee_candidates = self._candidates_for_agent(tasks, agent_name)
            preferred: list[Any] = [
                t for t in assignee_candidates
                if str(_safe_attr(t, "id", "") or "") in downstream_ids
            ]
            if not preferred:
                return None
            if len(preferred) == 1:
                return preferred[0]
            if scoring_args is not None:
                return _score_candidates_by_args(preferred, scoring_args)
            return None

        @staticmethod
        def _task_from_pending_correction(
            *,
            ctx: SessionContext,
            tasks: list[Any],
            agent_name: str,
        ) -> Any:
            """Signal 6 — pin to a task targeted by a pending correction.

            Reads ``goldfive.pending_corrections.<agent>.<task_id>``
            keys off the orchestration session state (written by
            :mod:`goldfive._correction_injection` for CORRECT-kind
            supersedes). When at least one entry exists for the bare
            form of ``agent_name``, returns the first matching plan
            task that is PENDING or RUNNING.
            """
            from goldfive.types import TaskStatus

            state = _safe_attr(ctx.session, "state", None)
            if not isinstance(state, Mapping):
                return None
            # Strip a compound prefix on agent_name to match the
            # bare-form keys the writer uses.
            bare_agent = agent_name.rsplit(":", 1)[-1]
            prefix = f"goldfive.pending_corrections.{bare_agent}."
            target_task_ids: list[str] = []
            for key in state:
                if not isinstance(key, str):
                    continue
                if not key.startswith(prefix):
                    continue
                tid = key[len(prefix):]
                if tid:
                    target_task_ids.append(tid)
            if not target_task_ids:
                return None
            # Resolve to the first PENDING/RUNNING plan task with a
            # matching id. We do not require assignee-equality here —
            # the writer keyed on the agent already, and we trust that
            # the correction is for this agent's turn.
            tasks_by_id = {
                str(_safe_attr(t, "id", "") or ""): t for t in tasks
            }
            for tid in target_task_ids:
                task = tasks_by_id.get(tid)
                if task is None:
                    continue
                status = _safe_attr(task, "status", None)
                if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                    return task
            return None

        @staticmethod
        def _alternate_agent_name_form(agent_name: str) -> str:
            """Return the bare/compound alternate of ``agent_name``.

            ``"compound:foo"`` -> ``"foo"``; ``"foo"`` -> ``""`` (no
            compound prefix to add — there's no convention for which
            prefix to try without context). Signal 7 only exercises
            the strip direction, since the planner-side normalisation
            (PR #215) already strips on the way in.
            """
            if not agent_name:
                return ""
            if ":" in agent_name:
                return agent_name.rsplit(":", 1)[-1]
            return ""

        def _low_confidence_best_guess(
            self,
            *,
            tasks: list[Any],
            agent_name: str,
            scoring_args: Any,
        ) -> tuple[Any, float] | None:
            """Signal 8 — return a best-guess (task, confidence) pair.

            Last-resort: if every prior signal failed but there's
            something resembling work for this agent in the plan, pick
            the most-plausible task and tag the resolution as
            low-confidence so the operator-visible event makes the
            uncertainty explicit.

            Strategy: assignee-match candidates (any status that's
            PENDING/RUNNING) scored against tool args. If empty, no
            pin — there's nothing better than ``missing_task_id``.
            """
            assignee_candidates = self._candidates_for_agent(tasks, agent_name)
            if not assignee_candidates:
                return None
            if len(assignee_candidates) == 1:
                # Single assignee match but DAG-relaxed already would
                # have caught this — getting here means the relaxed
                # path didn't run (e.g. signals 5/6 fell through with
                # parent or correction context but neither matched).
                # Pin with low confidence.
                return assignee_candidates[0], 0.4
            if scoring_args is not None:
                chosen = _score_candidates_by_args(
                    assignee_candidates, scoring_args
                )
                if chosen is not None:
                    return chosen, 0.4
            # Tie / no scoring available — pick the first deterministically
            # so behaviour is reproducible across runs. The low-
            # confidence event makes the uncertainty visible.
            return assignee_candidates[0], 0.2

        async def _emit_pin_resolved(
            self,
            *,
            ctx: SessionContext,
            agent_name: str,
            task_id: str,
            via_signal: str,
            score: float,
            invocation_id: str,
            candidate_count: int,
        ) -> None:
            """Emit a ``pin_resolved`` (or ``pin_resolved_low_confidence``)
            sink event so operators see which signal landed the pin.

            Uses :func:`goldfive.events.make_event` (dict envelope)
            because the proto schema doesn't yet carry a PinResolved
            slot — adding one would expand scope. Best-effort: every
            failure is logged and swallowed.
            """
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            kind = (
                "pin_resolved_low_confidence"
                if via_signal == "low_confidence"
                else "pin_resolved"
            )
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            payload: dict[str, Any] = {
                "agent_name": str(agent_name or ""),
                "task_id": str(task_id or ""),
                "via_signal": str(via_signal or ""),
                "score": float(score),
                "invocation_id": str(invocation_id or ""),
                "candidate_count": int(candidate_count),
            }
            try:
                from goldfive.events import emit, make_event  # noqa: PLC0415

                evt = make_event(
                    run_id, seq, kind, payload, session_id=session_id
                )
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_pin_resolved: failed to emit %s: %s", kind, exc
                )

        def _stamp_current_task_id(
            self,
            *,
            ctx: SessionContext,
            callback_context: Any,
            task_id: str,
            agent_name: str,
            source: str,
            task: Any = None,
            invocation_id: str = "",
        ) -> None:
            """Write ``task_id`` into both state surfaces for the sub-agent.

            Shared by every signal in :meth:`_pin_current_task_id_for_agent`
            (goldfive#264). The ``source`` label threads into the log
            line so operators see which signal landed the pin
            (delegation_pin / single_match / arg_scored / dag_relaxed /
            parent_pin_downstream / correction_target /
            assignee_normalised / low_confidence).

            ``invocation_id`` (when non-empty) is also recorded onto
            ``self._invocation_pinned_task_id`` so signal 5 of a
            child invocation's pin can read this invocation's pin
            without racing on the single ``goldfive.current_task_id``
            slot.

            When ``task`` is provided, the ADK side also stamps
            ``goldfive.current_task_title`` /
            ``goldfive.current_task_description`` so the dynamic
            instruction resolver (goldfive#251) can render plan-causal
            prompts without re-walking the plan. Legacy callers that
            don't pass the task object keep working — the resolver
            falls back to placeholders.
            """
            gf_state = _safe_attr(ctx.session, "state", None)
            if isinstance(gf_state, dict):
                try:
                    gf_state[_sp.KEY_CURRENT_TASK_ID] = task_id
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: goldfive session.state pin failed: %s",
                        exc,
                    )
            adk_state = _session_state_from_callback(callback_context)
            if isinstance(adk_state, Mapping):
                try:
                    if task is not None:
                        # write_current_task stamps all four fields
                        # (id / title / description / assignee) from the
                        # Task. Prefer this over the narrow id-only write
                        # so the dynamic-instruction resolver (#251)
                        # sees the title + description it needs.
                        _sp.write_current_task(adk_state, task)
                    else:
                        _sp.write_current_task_id(adk_state, task_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: ADK session.state pin failed: %s",
                        exc,
                    )
            log.info(
                "goldfive: pinned current_task_id=%s for sub-agent %s "
                "(source=%s)",
                task_id,
                agent_name,
                source,
            )
            # goldfive#264 — record per-invocation pin so child
            # invocations can resolve their parent's pin (signal 5).
            if invocation_id and task_id:
                self._invocation_pinned_task_id[invocation_id] = task_id

        def _pin_delegation_task_id(
            self,
            *,
            ctx: SessionContext,
            tool_context: Any,
            to_agent: str,
            tool_args: Any,
        ) -> None:
            """Stamp a per-``function_call_id`` task pin for an AgentTool
            dispatch so parallel same-agent invocations don't race on the
            single ``goldfive.current_task_id`` slot.

            goldfive#241 Item 3-bis. Resolution algorithm:

            1. Collect PENDING/RUNNING tasks whose ``assignee_agent_id``
               matches ``to_agent``.
            2. Keep only the tasks whose upstream edges all point at
               COMPLETED predecessors (DAG-aware; a task whose
               dependency isn't done yet cannot be the target of THIS
               dispatch).
            3. If exactly one candidate, that's the pin.
            4. If multiple candidates, score each against ``tool_args``
               by keyword overlap with its ``title + description`` and
               pick the top. Ties or zero-overlap fall through to "no
               pin" — the sub-agent's ``before_agent_callback`` takes
               over via the legacy single-match path.
            5. If zero candidates, no pin.

            Writes to ``ctx.session.state[goldfive.pending_delegations]``
            (a dict keyed by function_call_id) when a pin resolves.
            Silent on every failure mode — the worst case is we fall
            through to the legacy behaviour.
            """
            if not to_agent:
                return
            fc_id = _function_call_id_from_tool_context(tool_context)
            if not fc_id:
                return
            plan = _safe_attr(ctx.session, "plan", None)
            if plan is None:
                return
            tasks = _safe_attr(plan, "tasks", None) or ()
            edges = _safe_attr(plan, "edges", None) or ()
            from goldfive.types import TaskStatus

            completed_ids: set[str] = set()
            for t in tasks:
                if _safe_attr(t, "status", None) is TaskStatus.COMPLETED:
                    tid = str(_safe_attr(t, "id", "") or "")
                    if tid:
                        completed_ids.add(tid)

            def _upstream_ok(task_id: str) -> bool:
                for e in edges:
                    to_id = str(_safe_attr(e, "to_task_id", "") or "")
                    if to_id != task_id:
                        continue
                    from_id = str(_safe_attr(e, "from_task_id", "") or "")
                    if from_id and from_id not in completed_ids:
                        return False
                return True

            candidates: list[Any] = []
            for task in tasks:
                assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                if assignee != to_agent:
                    continue
                status = _safe_attr(task, "status", None)
                if status is not TaskStatus.PENDING and status is not TaskStatus.RUNNING:
                    continue
                tid = str(_safe_attr(task, "id", "") or "")
                if not tid or not _upstream_ok(tid):
                    continue
                candidates.append(task)

            if not candidates:
                return
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = _score_candidates_by_args(candidates, tool_args)
                if chosen is None:
                    log.debug(
                        "before_tool_callback: %d candidates for %s; "
                        "args did not disambiguate — no pin",
                        len(candidates),
                        to_agent,
                    )
                    return
            task_id = str(_safe_attr(chosen, "id", "") or "")
            if not task_id:
                return
            # Stamp the pin onto BOTH the goldfive orchestration
            # session.state (so reporting-tool handlers that read it
            # directly see it) AND the ADK tool_context session.state
            # (so the plugin's before_tool_callback → _resolve_pinned_task_id
            # sees it on the sub-invocation's ToolContext). The two
            # dicts are distinct in live ADK — the goldfive Session
            # is our orchestration state, the ADK session.state is
            # the live ADK session the Runner drives.
            for target in (
                _safe_attr(ctx.session, "state", None),
                _session_state_from_callback(tool_context),
            ):
                if not isinstance(target, dict):
                    continue
                pend = target.get(_PENDING_DELEGATIONS_KEY)
                if not isinstance(pend, dict):
                    pend = {}
                    target[_PENDING_DELEGATIONS_KEY] = pend
                pend[fc_id] = task_id
            log.info(
                "goldfive: pinned delegation task_id=%s for fc_id=%s "
                "(sub-agent=%s, %d candidates)",
                task_id,
                fc_id,
                to_agent,
                len(candidates),
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
            # goldfive#251 Stream D: pending-corrections bridge. Keys
            # under ``goldfive.pending_corrections.<agent>.<task>`` are
            # written by :mod:`goldfive._correction_injection` on refine
            # landing. They're prefix-scoped and per-(agent, task), so
            # the bridge is structural — copy anything present under
            # the family prefix, evict anything the orchestration side
            # has cleared. The dynamic instruction resolver reads the
            # ADK-side copy per turn.
            try:
                _bridge_pending_corrections(gf_state, adk_state)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_bridge_orchestration_state: pending_corrections bridge failed: %s",
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

        async def _emit_pin_unresolved_drift(
            self,
            *,
            ctx: SessionContext,
            agent_name: str,
            tool_name: str,
            candidate_ids: list[str],
        ) -> None:
            """Emit a ``DriftDetected`` for an unresolvable reporting-tool pin.

            Used when ``before_tool_callback`` can't resolve a pin on a
            reporting tool and the current agent has PENDING / RUNNING
            candidates in the plan (so the pin SHOULD have worked — not
            an orchestration-only turn). The tool response to the LLM is
            a bare ``{"acknowledged": True}``; this drift event is the
            operator-visible signal that a stall occurred.

            Uses ``DriftKind.OFF_TOPIC`` with a ``reason=pin_unresolved: …``
            prefix (not a new ``PIN_UNRESOLVED`` proto kind) because the
            invariant is observer visibility, not wire-level classification.
            Sink dispatch fails-safe — observability must not block an
            invocation.
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
                    "_emit_pin_unresolved_drift: cannot import types: %s",
                    exc,
                )
                return

            detail = (
                f"pin_unresolved: {tool_name} for agent={agent_name or '?'}; "
                f"candidates=[{', '.join(candidate_ids)}]"
            )
            drift = DriftEvent(
                kind=DriftKind.OFF_TOPIC,
                severity=DriftSeverity.WARNING,
                detail=detail,
                current_task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                current_agent_id=agent_name or self._host_agent_name,
            )
            # Direct sink emission (bypass _handle_drift): this signal is
            # purely observability — no refine needed. A pin_unresolved
            # stall gets resolved by the LLM retrying or the orchestrator
            # intervening, not by a plan revision.
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
                    "_emit_pin_unresolved_drift: direct sink emit failed: %s",
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

        async def _emit_invocation_cancelled(
            self,
            *,
            invocation_id: str,
            agent_name: str = "",
            request: Any = None,
            tool_name: str = "",
        ) -> None:
            """Emit an ``InvocationCancelled`` sink event (goldfive#251).

            Operator-visible only — does NOT propagate to the LLM
            (that's what the minimal ``{"status": "cancelled"}`` tool
            response is for). Rich context: the invocation id, agent
            name, triggering reason / drift kind / severity / drift
            id, and an optional tool name when the cancel fired at a
            tool-dispatch checkpoint.

            Uses :func:`goldfive.events.make_event` (dict envelope)
            rather than a proto envelope because the proto schema
            doesn't yet carry an ``InvocationCancelled`` message —
            adding a new proto + regen would expand the scope of this
            change beyond Stream C. Dict events round-trip through the
            same sink fan-out (``goldfive.events.emit``) as proto
            events, so harmonograf's ingest path can handle both. A
            follow-up proto slot can be minted if / when sink-side
            strict typing is needed.

            Best-effort: every failure is logged and swallowed —
            observability must never block a callback.
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
            # Extract fields from the CancellationRequest dataclass if
            # provided. Duck-typed so a plain dict or an unfamiliar
            # shape still rounds-trips without raising.
            reason = ""
            severity = ""
            drift_id = ""
            drift_kind = ""
            detail = ""
            if request is not None:
                reason = str(_safe_attr(request, "reason", "") or "")
                sev_val = _safe_attr(request, "severity", None)
                severity = str(getattr(sev_val, "value", sev_val) or "")
                drift_id = str(_safe_attr(request, "drift_id", "") or "")
                drift_kind = str(_safe_attr(request, "drift_kind", "") or "")
                detail = str(_safe_attr(request, "detail", "") or "")
            payload: dict[str, Any] = {
                "invocation_id": str(invocation_id or ""),
                "agent_name": str(agent_name or ""),
                "reason": reason,
                "severity": severity,
                "drift_id": drift_id,
                "drift_kind": drift_kind,
                "detail": detail,
            }
            if tool_name:
                payload["tool_name"] = str(tool_name)
            try:
                from goldfive.events import emit, make_event  # noqa: PLC0415 — lazy

                evt = make_event(
                    run_id,
                    seq,
                    "invocation_cancelled",
                    payload,
                    session_id=session_id,
                )
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_invocation_cancelled: failed to emit: %s",
                    exc,
                )

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

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a).
            # Skip the LLM call when this invocation is flagged for
            # cancel. This is the checkpoint that matters most in
            # practice: a mid-flight LLM call is the expensive work
            # whose output would contaminate the parent transcript.
            # The ``before_agent_callback`` checkpoint above normally
            # fires first, but ADK may reach ``before_model_callback``
            # without ``before_agent_callback`` on some dispatch
            # shapes (e.g. direct model invocations in tests); this
            # check is the backstop.
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id_check = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if inv_id_check and self._cancel_state.get(inv_id_check) is not None:
                request = self.consume_cancel_for_invocation(inv_id_check)
                await self._emit_invocation_cancelled(
                    invocation_id=inv_id_check,
                    agent_name="",
                    request=request,
                )
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

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a).
            # When this invocation was flagged for cancel (either the
            # steerer at CRITICAL severity or a user-initiated cancel),
            # skip tool dispatch and return a MINIMAL LLM-visible tool
            # response: ``{"status": "cancelled"}``. The minimal shape
            # is deliberate — richer shapes (``reason``, ``detail``,
            # ``drift_kind``) become prompt-injection vectors (see
            # lessons from goldfive#250 / #252 / #253 where LLMs
            # pattern-matched on error strings and invented workarounds).
            # Rich context for operators lives on the
            # InvocationCancelled sink event emitted by
            # :meth:`_emit_invocation_cancelled`.
            inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
                tool_context, "invocation_context", None
            )
            inv_id_check = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if inv_id_check and self._cancel_state.get(inv_id_check) is not None:
                request = self.consume_cancel_for_invocation(inv_id_check)
                await self._emit_invocation_cancelled(
                    invocation_id=inv_id_check,
                    agent_name="",
                    request=request,
                    tool_name=tool_name,
                )
                # MINIMAL LLM-visible response — single-key dict, no
                # ``reason`` / ``detail`` / ``drift_kind``. The parent
                # LLM that receives this as an AgentTool response can
                # pattern-match only on the word "cancelled" and
                # should defer to the plan-revised context it sees on
                # its next turn to decide whether to re-dispatch.
                return {"status": "cancelled"}

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
            # goldfive#241 — task_id is hidden from the LLM-facing
            # reporting-tool schema so the model never supplies it.
            # Resolve from state (delegation-site pin first, then the
            # agent-turn pin). If neither resolves, the response branch
            # depends on whether the current agent actually has work in
            # the plan (goldfive#250 follow-up):
            #
            # * **No PENDING/RUNNING candidates for this agent** — a
            #   legit orchestration-only turn (e.g. coordinator whose
            #   tasks were superseded by a plan refine into tasks
            #   assigned to other agents). Return a bare silent
            #   acknowledgment so the agent's reporting protocol does
            #   NOT crash on plan-revision boundaries. A loud error
            #   here makes the LLM bypass the reporting protocol —
            #   observed live.
            # * **Has candidates** — the pin SHOULD have worked; a
            #   silent ack could mask a real stall. To keep the stall
            #   visible WITHOUT leaking a prompt-injection surface to
            #   the LLM (observed live: research_agent read an
            #   ``error: pin_unresolved`` payload and reasoned "This
            #   might be related to the plan/task system. Let me try
            #   a different approach — I'll just compile the research
            #   and create the presentation content directly", bypassing
            #   the reporting contract), the tool response is the same
            #   bare ``{"acknowledged": True}``. Operator visibility is
            #   preserved via a WARNING log AND a ``DriftDetected`` sink
            #   event (``DriftKind.OFF_TOPIC`` with a ``pin_unresolved:``
            #   reason prefix). See goldfive#252 follow-up + PR notes.
            #
            # The silent-ack response carries NO ``detail`` / ``error`` /
            # ``no_task_pinned`` / ``pin_unresolved`` keys — tool responses
            # go back to the LLM verbatim and any editorialising string
            # (or error-shaped payload) is treated as actionable context
            # (observed live: research_agent paraphrased a detail string
            # into its reasoning and proceeded with stale pre-refine
            # instructions, ignoring the refined scope).
            pinned = _inject_task_id_from_state(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_context=tool_context,
            )
            if _is_reporting_tool_name(tool_name) and not pinned:
                # Resolve the current agent name — prefer the live
                # invocation's agent (tool_context._invocation_context.
                # agent.name), fall back to the host agent from
                # SessionContext. Any resolution failure degrades to
                # the silent-ack path (conservative — avoid breaking
                # runs on edge-cases).
                agent_name = ""
                try:
                    inv_ctx = _safe_attr(
                        tool_context, "_invocation_context", None
                    ) or _safe_attr(tool_context, "invocation_context", None)
                    running_agent = _safe_attr(inv_ctx, "agent", None)
                    agent_name = str(_safe_attr(running_agent, "name", "") or "")
                    if not agent_name:
                        agent_name = str(
                            _safe_attr(ctx, "host_agent_name", "") or ""
                        )
                except Exception:  # noqa: BLE001
                    agent_name = ""

                has_candidates = False
                try:
                    has_candidates = _agent_has_pending_candidates(
                        ctx, agent_name
                    )
                except Exception:  # noqa: BLE001 — conservative fall-through
                    has_candidates = False

                if has_candidates:
                    # Gather candidate ids for the diagnostic WARNING +
                    # drift event (nice-to-have; swallow any resolution
                    # errors).
                    candidate_ids: list[str] = []
                    try:
                        from goldfive.types import TaskStatus

                        plan = _safe_attr(ctx.session, "plan", None)
                        tasks = _safe_attr(plan, "tasks", None) or ()
                        for task in tasks:
                            assignee = str(
                                _safe_attr(task, "assignee_agent_id", "") or ""
                            )
                            if assignee != agent_name:
                                continue
                            status = _safe_attr(task, "status", None)
                            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                                candidate_ids.append(
                                    str(_safe_attr(task, "id", "") or "")
                                )
                    except Exception:  # noqa: BLE001
                        pass
                    log.warning(
                        "before_tool_callback: pin_unresolved for %s "
                        "(agent=%s, candidates=[%s]); emitting "
                        "DriftDetected(pin_unresolved) and returning silent "
                        "ack so the LLM cannot pattern-match on the error",
                        tool_name,
                        agent_name or "?",
                        ", ".join(candidate_ids),
                    )
                    # Surface the stall to operators via a sink event so
                    # it's visible in harmonograf without being visible
                    # to the LLM. Reuse OFF_TOPIC with a reason prefix
                    # rather than adding a new proto DriftKind (heavier
                    # change; the invariant here is operator observability,
                    # not a new wire-level classification).
                    try:
                        await self._emit_pin_unresolved_drift(
                            ctx=ctx,
                            agent_name=agent_name,
                            tool_name=tool_name,
                            candidate_ids=candidate_ids,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "before_tool_callback: pin_unresolved drift emit "
                            "raised: %s",
                            exc,
                        )
                    return {"acknowledged": True}

                log.info(
                    "before_tool_callback: no task pinned for %s; "
                    "returning no-op acknowledgment (orchestration-only turn)",
                    tool_name,
                )
                return {"acknowledged": True}

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

                # goldfive#241 Item 3-bis — delegation-site task_id
                # pinning. When the coordinator fires multiple parallel
                # AgentTool calls to the same sub-agent in one turn,
                # each dispatch spawns its own sub-invocation; the
                # sub-agent's ``before_agent_callback`` cannot
                # disambiguate because all N parallel calls share the
                # same (agent_name, session.state) pair. We resolve
                # the candidate task for THIS AgentTool dispatch and
                # stash it on ``pending_delegations[<function_call_id>]``
                # so the reporting-tool callback (which DOES see the
                # function_call_id) can read the correct pin back.
                try:
                    self._pin_delegation_task_id(
                        ctx=ctx,
                        tool_context=tool_context,
                        to_agent=to_agent,
                        tool_args=tool_args,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    log.debug(
                        "before_tool_callback: delegation pin raised: %s",
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
                    # Resolve the live agent name so the steerer's
                    # per-(agent, task) reasoning-judge rate-limit
                    # bucket isolates agents (goldfive#252 follow-up).
                    # Prefer the invocation's running agent, fall back
                    # to the host agent so single-agent runs keep their
                    # historical bucketing.
                    reasoning_agent_name = ""
                    try:
                        running_agent = _safe_attr(inv_ctx, "agent", None)
                        reasoning_agent_name = str(
                            _safe_attr(running_agent, "name", "") or ""
                        )
                    except Exception:  # noqa: BLE001
                        reasoning_agent_name = ""
                    if not reasoning_agent_name:
                        reasoning_agent_name = self._host_agent_name or ""
                    try:
                        await observe_reasoning(
                            reasoning,
                            task=ctx.task,
                            session=ctx.session,
                            provider=_infer_provider(llm_response),
                            agent_name=reasoning_agent_name,
                        )
                    except TypeError:
                        # Back-compat: custom steerer without the
                        # ``agent_name`` kwarg. Fall back silently.
                        try:
                            await observe_reasoning(
                                reasoning,
                                task=ctx.task,
                                session=ctx.session,
                                provider=_infer_provider(llm_response),
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "after_model_callback: observe_reasoning "
                                "(fallback) raised: %s",
                                exc,
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
