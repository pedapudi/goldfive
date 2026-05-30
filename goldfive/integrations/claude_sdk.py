"""Claude Agent SDK adapter for goldfive's two LLM call-site shapes.

Two shapes covered:

1. ``make_call_llm`` — an async ``(system, prompt, model) -> str`` that
   plugs into goldfive's ``CallLLM`` contract: ``LLMPlanner.call_llm``,
   ``LLMGoalDeriver.call_llm``, judge fallback via
   ``goldfive.wrap(call_llm=...)``.

2. ``ClaudeAgentSDKLlm`` — an ADK ``BaseLlm`` subclass for use as an
   agent ``model=``. Translates ADK ``LlmRequest`` → fresh
   ``ClaudeSDKClient`` per call → ADK ``LlmResponse``. ADK tools from
   ``LlmRequest.tools_dict`` are exposed as MCP-prefixed schemas
   (``mcp__adk__<name>``); a ``PreToolUse`` hook returns
   ``permissionDecision: "defer"`` so Claude's tool calls are
   *captured* and returned to ADK as ``function_call`` parts rather
   than executed inside the adapter. ADK then runs the tool through
   its normal pipeline and the goldfive plugin observes every call
   (``CONFABULATION_RISK`` / ``CAPABILITY_MISMATCH`` detectors fire
   correctly, tool args / results land in the harmonograf inspector).

Both routes go through claude-agent-sdk which uses the local
``claude`` CLI's login (Max plan, API key, etc.) — no separate
``ANTHROPIC_API_KEY`` needed when the user is already authenticated via
``claude /login``.

Usage::

    from goldfive import wrap, LLMPlanner, LLMGoalDeriver
    from goldfive.integrations.claude_sdk import (
        ClaudeAgentSDKLlm,
        make_call_llm,
    )

    call_llm = make_call_llm("claude-haiku-4-5")
    agent_model = ClaudeAgentSDKLlm(model="claude-haiku-4-5")
    # build agent tree using ``agent_model`` as the ``model=`` for each Agent
    runner = wrap(
        agent_tree,
        planner=LLMPlanner(call_llm=call_llm, model="claude-haiku-4-5"),
        goal_deriver=LLMGoalDeriver(call_llm=call_llm, model="claude-haiku-4-5"),
        call_llm=call_llm,  # judges inherit this fallback
    )

Short, structured prompts (plan generation, goal extraction, drift
verdicts) run cleanly: claude-agent-sdk's internal agent loop does not
engage for one-shot structured outputs.

Quota / billing visibility
--------------------------

The callable bills against whichever auth the local ``claude`` CLI is
logged into — a Max plan in the typical setup. A single steered
goldfive run can issue multiple calls through this factory:

* one ``LLMPlanner`` call for the initial plan;
* one ``LLMPlanner.refine`` call per drift that produces a plan
  revision (rate-limited, but bursty runs can fire several);
* one ``LLMGoalDeriver`` call per ``run_started``;
* one ``judge_goal_drift`` call per task-boundary judge fire and per
  ``goal_drift_check_interval`` agent turn (rate-limited to ~10s
  between task-boundary fires);
* one ``judge_reasoning`` call per reasoning-judge fire.

Anecdotal: the ``presentation_agent`` example issues ~5–15 calls
through this factory on a clean run; failing runs (drift cascades) can
double that. Operators on Max should expect modest per-run quota burn
on Haiku; bumping to Sonnet/Opus or running tight loops in
production should plan for paid-tier billing instead.

The integration is gated on the optional ``claude-agent-sdk``
dependency. Install via ``uv pip install claude-agent-sdk`` (or add to
your project's extras) before importing this module.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)


# Module-level so the test suite can assert on it without reaching
# into ``_call``'s closure.
_DEFAULT_MAX_TURNS = 5


def make_call_llm(
    default_model: str = "claude-haiku-4-5",
) -> Callable[[str, str, str], Awaitable[str]]:
    """Build an async ``(system, prompt, model) -> str`` callable.

    Goldfive plumbs the configured model through on every call. Falsy
    ``model`` (``""`` or ``None``) falls back to ``default_model`` —
    matches the rest of goldfive's call-site convention where an empty
    model string means "use whatever the integration picked." Callers
    that need to bypass the default (e.g. to assert a specific model
    string in tests) should pass a sentinel non-empty value instead.

    Raises :class:`ImportError` (with an install hint) on first call if
    ``claude_agent_sdk`` is not importable. We import lazily so simply
    importing this module does not require the SDK to be installed.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "goldfive.integrations.claude_sdk requires `claude-agent-sdk`. "
            "Install it with: uv pip install claude-agent-sdk"
        ) from e

    async def _call(system: str, prompt: str, model: str) -> str:
        # Two SDK-isolation knobs (must agree with the BaseLlm adapter
        # in a follow-up PR):
        #
        # * ``setting_sources=[]`` — disables the SDK's default
        #   discovery chain (``~/.claude/settings.json``, project
        #   ``.claude/settings.json``, project ``CLAUDE.md``). Without
        #   it, operator-local Claude config leaks into every
        #   planner / judge prompt — e.g. a personal CLAUDE.md
        #   ("respond in YAML", "be terse") would silently corrupt the
        #   structured-JSON output goldfive parsers expect, surfacing
        #   downstream as "unparseable verdict" with no obvious cause.
        # * ``tools=[]`` — strips Claude Code's built-in tools
        #   (TodoWrite, Task, Read, Bash, …) from Claude's view, which
        #   keeps the internal agent loop dormant for these short
        #   structured prompts. ``allowed_tools=[]`` is NOT the right
        #   knob here: it only suppresses permission prompts; the tools
        #   stay visible to the model.
        opts = ClaudeAgentOptions(
            system_prompt=system or None,
            model=model or default_model,
            tools=[],
            setting_sources=[],
            max_turns=_DEFAULT_MAX_TURNS,
        )
        chunks: list[str] = []
        last_stop_reason: Any = None
        async for msg in query(prompt=prompt, options=opts):
            stop_reason = getattr(msg, "stop_reason", None)
            if stop_reason is not None:
                last_stop_reason = stop_reason
            for block in getattr(msg, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)

        if not chunks:
            # Silent zero-output is the classic "unparseable verdict"
            # diagnostic dead-end — downstream JSON parsers
            # (``LLMPlanner._parse_plan_response``,
            # ``judge_goal_drift._parse_verdict``) see ``""`` with no
            # breadcrumb pointing at the cause. Log loudly so operators
            # can correlate empty returns with model / turn settings.
            log.warning(
                "goldfive.integrations.claude_sdk.make_call_llm: "
                "claude-agent-sdk produced no text "
                "(model=%s, stop_reason=%s, max_turns=%d); "
                "downstream parsers will see an empty string",
                model or default_model,
                last_stop_reason,
                _DEFAULT_MAX_TURNS,
            )
        return "".join(chunks)

    return _call


# ---------------------------------------------------------------------------
# ADK BaseLlm adapter
# ---------------------------------------------------------------------------


def _extract_system_instruction(config: Any) -> str:
    """Pull a plain-string system prompt out of ADK's GenerateContentConfig.

    ``GenerateContentConfig.system_instruction`` may be ``None``, a bare
    string, or a ``Content`` proto (parts list). Reduce to a single
    string by concatenating any text parts.
    """
    if config is None:
        return ""
    si = getattr(config, "system_instruction", None)
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _render_contents_as_transcript(contents: list[Any]) -> str:
    """Render ADK ``contents`` into a single descriptive prompt string.

    Used by the BaseLlm adapter's text-replay strategy: each
    ``generate_content_async`` call spawns a fresh ``query()`` and we
    can't pass prior assistant messages with real ``tool_use`` blocks,
    so we describe the prior turns as a text transcript instead.

    Translation rules per ``Content``:

    * ``role="user"`` + text part → ``"User: <text>"``
    * ``role="user"`` + ``function_response`` part →
      ``"Tool result (<name>): <response>"``
    * ``role="model"`` + text part → ``"Assistant: <text>"``
    * ``role="model"`` + ``function_call`` part →
      ``"Assistant called tool <name> with args <args>"``

    Adjacent same-role text parts are concatenated. Empty contents
    returns an empty string. The output is intended to be wrapped as
    *the* user message to a fresh Claude call; the actual tools are
    registered separately via MCP.
    """
    import json as _json

    lines: list[str] = []
    for content in contents or ():
        role = getattr(content, "role", "") or "user"
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            fcall = getattr(part, "function_call", None)
            fresp = getattr(part, "function_response", None)
            if text:
                if role == "model":
                    lines.append(f"Assistant: {text}")
                else:
                    lines.append(f"User: {text}")
            elif fcall is not None and getattr(fcall, "name", None):
                args = getattr(fcall, "args", None) or {}
                try:
                    args_repr = _json.dumps(dict(args), default=str)
                except (TypeError, ValueError):
                    args_repr = str(args)
                lines.append(
                    f"Assistant called tool `{fcall.name}` with args {args_repr}"
                )
            elif fresp is not None and getattr(fresp, "name", None):
                resp = getattr(fresp, "response", None)
                try:
                    resp_repr = _json.dumps(resp, default=str)
                except (TypeError, ValueError):
                    resp_repr = str(resp)
                lines.append(f"Tool result (`{fresp.name}`): {resp_repr}")
    return "\n".join(lines)


# Back-compat alias — older import sites may still reach for the prior name.
_flatten_contents_to_prompt = _render_contents_as_transcript


_PY_TYPE_TO_JSON: dict[type, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
}


def _genai_schema_to_json_schema(schema: Any) -> dict[str, Any]:
    """Convert a ``google.genai.types.Schema`` into a JSON-Schema dict.

    Used to translate ADK's ``FunctionDeclaration.parameters`` (a Schema
    proto) into the dict shape claude-agent-sdk's ``@tool`` decorator
    accepts. Pass-through fields:

    * ``type``, ``properties``, ``items``, ``required``,
      ``description``, ``enum`` — the structural backbone.
    * ``nullable``, ``format``, ``default``, ``pattern`` —
      constraint hints that downstream consumers (Claude when
      constructing tool calls, ADK when validating returns) actually
      use. Dropping these produces malformed tool calls that ADK then
      rejects, triggering a retry loop noisy in logs and chewing into
      the per-call ``max_turns`` budget. Pass them through.

    Other Schema fields are dropped (intentional — we only translate
    the surface ADK tools populate today).
    """
    if schema is None:
        return {"type": "object", "properties": {}, "required": []}

    type_str: str = ""
    raw_type = getattr(schema, "type", None) or getattr(schema, "type_", None)
    if raw_type is not None:
        type_str = str(getattr(raw_type, "name", raw_type)).lower()

    out: dict[str, Any] = {}
    if type_str:
        out["type"] = type_str
    description = getattr(schema, "description", None)
    if description:
        out["description"] = description
    enum_vals = getattr(schema, "enum", None)
    if enum_vals:
        out["enum"] = list(enum_vals)
    properties = getattr(schema, "properties", None)
    if properties:
        out["properties"] = {
            k: _genai_schema_to_json_schema(v) for k, v in dict(properties).items()
        }
    items = getattr(schema, "items", None)
    if items is not None:
        out["items"] = _genai_schema_to_json_schema(items)
    required = getattr(schema, "required", None)
    if required:
        out["required"] = list(required)

    # Constraint hints — pass through when ADK populated them.
    for attr in ("nullable", "format", "default", "pattern"):
        val = getattr(schema, attr, None)
        if val is not None:
            out[attr] = val

    if "type" not in out:
        out["type"] = "object"
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def _build_input_schema_from_signature(func: Any) -> dict[str, Any]:
    """Build a JSON-schema dict from a Python callable's signature.

    Fallback path for tools where :func:`_extract_input_schema_from_adk_tool`
    cannot recover a declaration (e.g. plain callables without ADK
    wrapping). We map basic Python annotations to JSON Schema types;
    unknown annotations default to ``string``.
    """
    import inspect

    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return schema
    for pname, param in sig.parameters.items():
        if pname in ("self", "tool_context"):
            continue
        ann = param.annotation
        json_type = _PY_TYPE_TO_JSON.get(ann, "string")
        schema["properties"][pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            schema["required"].append(pname)
    return schema


def _extract_input_schema_from_adk_tool(adk_tool: Any) -> dict[str, Any]:
    """Recover a JSON-Schema dict for any ADK ``BaseTool``.

    Strategy: prefer ``_get_declaration()`` since it works uniformly for
    ``FunctionTool``, ``AgentTool``, and anything else conforming to
    ``BaseTool``. For ``AgentTool`` this is the only path that yields
    the required ``request`` parameter — using
    :func:`_build_input_schema_from_signature` on ``tool.func`` returns
    an empty schema because ``AgentTool`` has no ``func`` attribute,
    which then makes Claude call the tool with empty args and ADK
    raises ``KeyError('request')``.

    Falls back to the signature-based path when the declaration isn't
    available (some tool wrappers may not implement it).
    """
    declaration = None
    getter = getattr(adk_tool, "_get_declaration", None)
    if callable(getter):
        try:
            declaration = getter()
        except Exception:  # noqa: BLE001 — defensive; fall back below
            declaration = None
    if declaration is None:
        declaration = getattr(adk_tool, "declaration", None)
    if declaration is not None:
        # Check ``parameters_json_schema`` first. ADK gates which field
        # is populated via ``FeatureName.JSON_SCHEMA_FOR_FUNC_DECL``;
        # today ``parameters`` is the populated one, but if that flag
        # flips in a future ADK upgrade and we don't read both, this
        # extractor silently returns ``{"type": "object", "properties": {}}``
        # — the same failure mode the PR's ``AgentTool`` fix addresses.
        parameters_json = getattr(declaration, "parameters_json_schema", None)
        if parameters_json:
            return dict(parameters_json)
        parameters = getattr(declaration, "parameters", None)
        if parameters is not None:
            return _genai_schema_to_json_schema(parameters)

    func = getattr(adk_tool, "func", None)
    if func is not None:
        return _build_input_schema_from_signature(func)
    return {"type": "object", "properties": {}, "required": []}


def _adk_tool_to_sdk_tool_schema(adk_name: str, adk_tool: Any) -> Any:
    """Declare an ADK tool's schema to claude-agent-sdk for tool-discovery.

    The SDK requires every MCP tool to have a handler, but in the
    text-replay BaseLlm strategy we never want the handler to run —
    the ``PreToolUse`` hook defers each call so ADK can execute the
    real tool through its normal pipeline (preserving goldfive
    observability). The handler here is a safety stub that returns an
    explicit "intercepted, should not run" sentinel; if it ever does
    fire, the surrounding integration has a bug worth surfacing.
    """
    from claude_agent_sdk import tool as sdk_tool_decorator

    description = (
        getattr(adk_tool, "description", None) or f"ADK tool {adk_name}"
    )
    input_schema = _extract_input_schema_from_adk_tool(adk_tool)

    async def _stub_handler(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "internal error: ClaudeAgentSDKLlm stub handler "
                        f"reached for tool {adk_name!r}. The PreToolUse "
                        "hook should have deferred this call to ADK."
                    ),
                }
            ],
            "is_error": True,
        }

    return sdk_tool_decorator(adk_name, description, input_schema)(_stub_handler)


# Back-compat alias.
_adk_tool_to_sdk_tool = _adk_tool_to_sdk_tool_schema


_MCP_TOOL_PREFIX = "mcp__adk__"
_TRANSCRIPT_PREAMBLE = (
    "The following is the conversation so far with this agent and the "
    "tools it has access to. Continue from the latest turn. If you "
    "need to use a tool to make progress, call it via the normal tool "
    "interface; otherwise produce a final answer for the user.\n\n"
    "--- BEGIN PRIOR TRANSCRIPT ---\n"
)
_TRANSCRIPT_EPILOGUE = (
    "\n--- END PRIOR TRANSCRIPT ---\n\n"
    "Please produce your next turn."
)


def _defer_in_permission_decision(sdk_types: Any) -> bool | None:
    """Whether ``"defer"`` is a permitted ``permissionDecision`` value.

    Returns ``True``/``False`` when the ``permissionDecision`` ``Literal``
    can be located and inspected on any TypedDict in
    ``claude_agent_sdk.types``; returns ``None`` when the field or its
    ``Literal`` can't be found/parsed (caller treats ``None`` as
    "cannot verify — don't fail", to avoid false alarms on SDK layout
    changes that don't actually drop the feature).
    """
    import typing

    for obj in vars(sdk_types).values():
        ann = getattr(obj, "__annotations__", None)
        if not isinstance(ann, dict) or "permissionDecision" not in ann:
            continue
        # Walk the (possibly ``NotRequired[...]``-wrapped) hint looking
        # for a nested ``Literal[...]`` and collect its string members.
        found_literal = False
        seen: set[str] = set()
        stack = [ann["permissionDecision"]]
        while stack:
            cur = stack.pop()
            if typing.get_origin(cur) is typing.Literal:
                found_literal = True
                seen.update(a for a in typing.get_args(cur) if isinstance(a, str))
            else:
                stack.extend(typing.get_args(cur))
        return "defer" in seen if found_literal else None
    return None


def _verify_defer_contract() -> None:
    """Fail loudly at the seam if the installed ``claude-agent-sdk`` no
    longer exposes the ``defer`` PreToolUse contract this adapter is
    built on.

    The adapter pins ``claude-agent-sdk>=0.1.80`` with no upper bound and
    keys its runtime behaviour on two SDK affordances that aren't visible
    to the test suite (the SDK isn't a test dependency): the
    ``permissionDecision="defer"`` value and the ``DeferredToolUse`` /
    ``ResultMessage.deferred_tool_use`` envelope (``stop_reason ==
    "tool_deferred"``). A future release that renames or drops these
    would otherwise turn into a silent mid-run hang. This check converts
    that into a clear error at class-build time.
    """
    from claude_agent_sdk import types as sdk_types

    missing: list[str] = []
    if not hasattr(sdk_types, "DeferredToolUse"):
        missing.append("the DeferredToolUse type")
    if _defer_in_permission_decision(sdk_types) is False:
        missing.append('"defer" in the PreToolUse permissionDecision literal')
    if missing:
        raise RuntimeError(
            "Installed claude-agent-sdk no longer exposes the defer "
            "contract ClaudeAgentSDKLlm depends on (missing: "
            + "; ".join(missing)
            + "). The adapter pins claude-agent-sdk>=0.1.80 with no "
            "upper bound; a newer release appears to have changed the "
            "PreToolUse defer API. Pin a compatible version or update "
            "the adapter."
        )


def make_claude_agent_sdk_llm_class() -> type:
    """Build the ADK ``BaseLlm`` subclass on demand.

    Constructed at call time so ``import goldfive.integrations.claude_sdk``
    does not require ADK to be installed when only :func:`make_call_llm`
    is needed.
    """
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ClaudeSDKClient,
            create_sdk_mcp_server,
        )
        from claude_agent_sdk.types import HookMatcher
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "goldfive.integrations.claude_sdk requires `claude-agent-sdk`. "
            "Install it with: uv pip install claude-agent-sdk"
        ) from e
    # Fail loudly now (not mid-run) if the installed SDK dropped the
    # defer contract this adapter keys on.
    _verify_defer_contract()
    try:
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_request import LlmRequest
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types as genai_types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ClaudeAgentSDKLlm requires `google-adk`. Install via the "
            "goldfive[adk] extra."
        ) from e

    class ClaudeAgentSDKLlm(BaseLlm):  # type: ignore[misc]
        """ADK ``BaseLlm`` routing generation through claude-agent-sdk.

        Implementation strategy (text-encoded replay): on every
        ``generate_content_async`` call we spawn a fresh Claude run via
        ``ClaudeSDKClient``, render the entire prior ADK conversation as
        a descriptive text transcript prepended to the user prompt, and
        register the ADK tool schemas via the SDK's MCP transport. A
        ``PreToolUse`` hook returns ``defer`` so each tool request is
        *captured* (not executed) and surfaced back to ADK as a
        ``function_call`` ``Part``. ADK then runs the tool through its
        normal pipeline (goldfive plugin observes), and the next ADK
        invocation re-enters this method with the
        ``function_response`` appended to ``contents``.

        One tool call per turn (parallel fan-out is serialised). Per the
        ``DeferredToolUse`` contract below, returning ``defer`` *stops
        the run* and the terminating ``ResultMessage`` carries a single
        deferred tool call, so even when Claude emits several
        ``tool_use`` blocks in one assistant message the SDK halts at
        the first defer and the hook captures only one call before
        ``generate_content_async`` breaks. The capture is kept as a
        ``list`` so that *if* a future SDK fires several ``PreToolUse``
        hooks before halting we surface them all (one ``function_call``
        ``Part`` each; append-order == emission-order is best-effort),
        but we do **not** claim that happens on ``>=0.1.80`` — it does
        not, and a live two-tools-in-one-turn trace proving simultaneous
        N-capture was not reproduced. When the model attempts to fan out
        (more ADK ``tool_use`` blocks observed in the stream than were
        captured) we ``log.warning`` so the serialisation is visible
        rather than silent; ADK loops back and Claude re-issues the
        remaining call(s) next turn, so the end state matches the native
        Gemini path even though intra-turn parallelism is flattened.

        The ``defer`` mechanism is SDK-supported as of
        ``claude-agent-sdk>=0.1.80`` — see ``claude_agent_sdk/types.py``:

        * ``PreToolUseHookSpecificOutput.permissionDecision``: the
          literal includes ``"defer"`` (alongside ``allow``, ``deny``,
          ``ask``);
        * ``DeferredToolUse`` dataclass: docstring reads *"Tool use
          that was deferred by a PreToolUse hook returning ``defer``.
          The run stops and the result message carries the deferred
          tool call here so the caller can inspect it and decide
          whether to resume."*;
        * ``ResultMessage.deferred_tool_use`` field + ``stop_reason ==
          "tool_deferred"`` on the terminating message.

        We rely on the ``stop_reason`` + the hook-captured calls, not
        on ``ResultMessage.deferred_tool_use`` itself, because the hook
        sees the call earlier in the stream (before SDK synthesises the
        deferred-tool envelope on the closing ``ResultMessage``). Pin
        is enforced via the ``goldfive[claude_sdk]`` extra.

        Why text replay instead of stateful client reuse: the SDK has
        no clean API for "given this transcript + tool history, give me
        the next message." ``query()`` and ``ClaudeSDKClient`` accept
        only user-message streams. Replaying prior assistant tool_use
        blocks via a same-client ``query()`` doesn't continue cleanly
        after a deferred run, and ``resume=<session_id>`` spawns a
        fresh subprocess anyway. So we accept the quadratic token cost
        of resending the transcript each turn in exchange for a much
        simpler implementation and predictable per-turn behaviour.

        Two coercion knobs (same as :func:`make_call_llm`) keep
        Claude Code's bundled CLI in thin behaviour:

        * ``setting_sources=[]`` — SDK isolation mode; no filesystem
          settings or CLAUDE.md auto-detection.
        * ``tools=[allowlist]`` — only our MCP-prefixed ADK tools are
          visible to Claude. Claude Code's built-ins (TodoWrite, Task,
          Read, Bash, …) are excluded, which keeps the internal agent
          loop dormant.

        Observability: tool calls flow through ADK's normal pipeline,
        so goldfive's plugin sees each one. ``CONFABULATION_RISK`` and
        ``CAPABILITY_MISMATCH`` drift detectors fire correctly.
        """

        @classmethod
        def supported_models(cls) -> list[str]:
            return [r"claude-.*"]

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            system = _extract_system_instruction(llm_request.config)
            transcript = _render_contents_as_transcript(llm_request.contents)
            user_prompt = (
                _TRANSCRIPT_PREAMBLE + transcript + _TRANSCRIPT_EPILOGUE
                if transcript
                else ""
            )

            adk_tools_dict = getattr(llm_request, "tools_dict", None) or {}
            mcp_tool_names: list[str] = []
            mcp_servers: dict[str, Any] = {}
            if adk_tools_dict:
                sdk_tools = [
                    _adk_tool_to_sdk_tool_schema(name, t)
                    for name, t in adk_tools_dict.items()
                ]
                mcp_servers["adk"] = create_sdk_mcp_server(
                    name="adk", tools=sdk_tools
                )
                mcp_tool_names = [
                    f"{_MCP_TOOL_PREFIX}{name}" for name in adk_tools_dict.keys()
                ]

            # Append (don't overwrite) each deferred ADK tool call. The
            # SDK halts the run on the *first* ``defer`` (see the class
            # docstring + the ``DeferredToolUse`` contract), so in
            # practice this holds at most one entry per turn; the list
            # shape is forward-compatible if a future SDK fires several
            # ``PreToolUse`` hooks before halting. ``attempted_adk_calls``
            # counts ADK ``tool_use`` blocks seen in the stream so we can
            # warn when the model tried to fan out but only one call was
            # captured (the rest serialise across ADK turns).
            captured_tool_calls: list[dict[str, Any]] = []
            attempted_adk_calls = 0
            valid_tool_names = set(mcp_tool_names)

            async def _defer_hook(
                input_data: dict[str, Any],
                tool_use_id: str | None,
                context: Any,
            ) -> dict[str, Any]:
                name = input_data.get("tool_name", "")
                # Only defer ADK tools (registered via our ``adk`` MCP
                # server). Tools outside this allowlist are usually
                # user-account-bound MCP servers (Gmail, Drive, etc.)
                # leaking through Claude Code's CLI process at the
                # network level — these aren't disabled by
                # ``setting_sources=[]`` because they come from the
                # cloud account, not local filesystem. Deny them so
                # Claude retries with a tool that ADK actually has.
                if name not in valid_tool_names:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Tool {name!r} is not available in this "
                                "run. Only the tools declared on the "
                                "current agent may be used. Choose one of: "
                                + ", ".join(sorted(valid_tool_names)) + "."
                            ),
                        }
                    }
                captured_tool_calls.append(
                    {
                        "name": name,
                        "args": input_data.get("tool_input", {}) or {},
                        "tool_use_id": tool_use_id or "",
                    }
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "defer",
                        "permissionDecisionReason": (
                            "ADK will execute this tool through its normal "
                            "pipeline and feed the result back."
                        ),
                    }
                }

            # ``tools`` and ``allowed_tools`` are intentionally *both*
            # the ADK allowlist, and they do different jobs:
            #   * ``tools`` is the visibility allowlist — only our
            #     MCP-prefixed ADK tools are shown to the model, which
            #     strips Claude Code's built-ins (TodoWrite/Task/…) so
            #     the internal agent loop stays dormant.
            #   * ``allowed_tools`` suppresses the interactive permission
            #     ``ask`` the SDK would otherwise raise on each ADK tool
            #     before our ``PreToolUse`` hook gets to ``defer`` it.
            # ``setting_sources=[]`` is the SDK isolation knob (no
            # CLAUDE.md / settings discovery). ``max_turns`` reuses the
            # module constant so this path can't drift from
            # ``make_call_llm``.
            opts = ClaudeAgentOptions(
                system_prompt=system or None,
                model=llm_request.model or self.model,
                tools=mcp_tool_names if mcp_tool_names else [],
                mcp_servers=mcp_servers,
                allowed_tools=mcp_tool_names,
                setting_sources=[],
                max_turns=_DEFAULT_MAX_TURNS,
                hooks={"PreToolUse": [HookMatcher(hooks=[_defer_hook])]},
            )

            # Text bookkeeping: we keep the *preamble* (text emitted
            # before any ``tool_use`` block) as a real ``Part`` because
            # it's Claude's "let me check the weather first" framing.
            # Text emitted *after* the first ``tool_use`` is the SDK's
            # synthetic apology / "tool was deferred" artifact — discard.
            preamble_chunks: list[str] = []
            saw_tool_use_inline = False
            last_usage: dict[str, Any] | None = None
            last_stop_reason: Any = None
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(user_prompt or "Please continue.")
                async for msg in client.receive_response():
                    # Surface usage from whichever ``AssistantMessage``
                    # most recently reported it (per
                    # ``claude_agent_sdk.types.AssistantMessage.usage``).
                    msg_usage = getattr(msg, "usage", None)
                    if msg_usage:
                        last_usage = msg_usage
                    stop_reason = getattr(msg, "stop_reason", None)
                    if stop_reason is not None:
                        last_stop_reason = stop_reason
                    for block in getattr(msg, "content", []) or []:
                        btype = type(block).__name__
                        if btype == "TextBlock" and not saw_tool_use_inline:
                            text = getattr(block, "text", None)
                            if text:
                                preamble_chunks.append(text)
                        elif btype == "ToolUseBlock":
                            # Flip the gate so subsequent text in *this*
                            # stream goes to the discard bucket.
                            saw_tool_use_inline = True
                            # Count ADK tool_use blocks the model emitted
                            # so we can detect attempted fan-out the SDK
                            # serialised by halting on the first defer.
                            if getattr(block, "name", None) in valid_tool_names:
                                attempted_adk_calls += 1
                    if stop_reason == "tool_deferred":
                        break

            # The model emitted more ADK tool calls than the SDK let us
            # capture before halting on the first defer. ADK will loop
            # back and Claude re-issues the rest next turn, but flag the
            # serialisation so it's not a silent divergence from the
            # native (parallel) Gemini path.
            if attempted_adk_calls > len(captured_tool_calls):
                log.warning(
                    "goldfive.integrations.claude_sdk.ClaudeAgentSDKLlm: "
                    "model emitted %d ADK tool calls in one turn but the "
                    "SDK deferred only %d before halting (model=%s); the "
                    "remaining %d will be re-issued on subsequent ADK "
                    "turns (intra-turn parallelism is serialised)",
                    attempted_adk_calls,
                    len(captured_tool_calls),
                    llm_request.model or self.model,
                    attempted_adk_calls - len(captured_tool_calls),
                )

            parts: list[Any] = []
            preamble = "".join(preamble_chunks).strip()
            if preamble:
                parts.append(genai_types.Part(text=preamble))
            for call in captured_tool_calls:
                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=call["name"].removeprefix(_MCP_TOOL_PREFIX),
                            args=call["args"],
                        )
                    )
                )
            if not parts:
                # Neither preamble text nor a captured tool call — a bare
                # empty model turn. To ADK this is indistinguishable from
                # a legitimately empty response and is a prime suspect
                # when a subagent stalls, so log loudly (mirrors the
                # zero-output WARNING in ``make_call_llm``). We still
                # emit one empty-text ``Part`` so the response shape
                # stays well-formed for ADK to walk.
                log.warning(
                    "goldfive.integrations.claude_sdk.ClaudeAgentSDKLlm: "
                    "claude-agent-sdk produced no text and no tool call "
                    "(model=%s, stop_reason=%s, max_turns=%d); ADK will "
                    "see an empty model turn",
                    llm_request.model or self.model,
                    last_stop_reason,
                    _DEFAULT_MAX_TURNS,
                )
                parts.append(genai_types.Part(text=""))

            # Translate the SDK's ``usage`` (Anthropic's
            # ``{input_tokens, output_tokens, ...}`` dict) into ADK's
            # ``GenerateContentResponseUsageMetadata`` so the
            # ``goldfive#172`` per-call instrumentation can log
            # ``llm.usage.*`` instead of ``?``. Operators on Max care
            # about this metric — without it, quota burn per goldfive
            # run is invisible until the invoice arrives.
            usage_metadata: Any = None
            if last_usage:
                input_tokens = last_usage.get("input_tokens")
                output_tokens = last_usage.get("output_tokens")
                total = (input_tokens or 0) + (output_tokens or 0)
                usage_metadata = genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=input_tokens,
                    candidates_token_count=output_tokens,
                    total_token_count=total or None,
                )

            yield LlmResponse(
                content=genai_types.Content(role="model", parts=parts),
                partial=False,
                turn_complete=True,
                usage_metadata=usage_metadata,
            )

    return ClaudeAgentSDKLlm


# Module-level alias for convenience: ``ClaudeAgentSDKLlm = ...``. Built
# lazily on first attribute access so that ``import
# goldfive.integrations.claude_sdk`` stays cheap and dependency-light.
#
# Error contract:
#
# * Unknown attribute names → :class:`AttributeError` (standard
#   Python protocol; lets ``hasattr`` answer ``False`` cleanly).
# * Attribute is ``"ClaudeAgentSDKLlm"`` but ``claude-agent-sdk`` (or
#   ``google-adk``) is not importable → the underlying
#   :class:`ImportError` propagates with the install hint baked in by
#   :func:`make_claude_agent_sdk_llm_class`. Surfacing the more
#   informative ImportError is intentional: a generic
#   ``AttributeError`` would point the caller at the *symbol* when the
#   actual problem is a *missing dependency*. The trade-off is that
#   ``hasattr(module, "ClaudeAgentSDKLlm")`` raises rather than
#   returning ``False`` when the SDK is uninstalled — callers who need
#   the soft-check semantics should catch ``ImportError`` explicitly.
def __getattr__(name: str) -> Any:
    if name == "ClaudeAgentSDKLlm":
        cls = make_claude_agent_sdk_llm_class()
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
