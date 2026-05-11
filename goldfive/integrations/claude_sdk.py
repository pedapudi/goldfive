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
    accepts. Handles the subset of Schema fields ADK tools actually
    populate: ``type``, ``properties``, ``items``, ``required``,
    ``description``, ``enum``. Unknown fields are dropped.
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
        ``PreToolUse`` hook returns ``defer`` so Claude's first tool
        request is *captured* (not executed) and surfaced back to ADK as
        a ``function_call`` ``Part``. ADK then runs the tool through its
        normal pipeline (goldfive plugin observes), and the next ADK
        invocation re-enters this method with the ``function_response``
        appended to ``contents``.

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

            captured_tool_call: dict[str, Any] = {}
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
                # Capture first ADK tool defer; ignore any subsequent
                # calls in the same run (we'll only return the first to
                # ADK and let it loop back for the next one).
                if not captured_tool_call:
                    captured_tool_call.update(
                        name=name,
                        args=input_data.get("tool_input", {}) or {},
                        tool_use_id=tool_use_id or "",
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

            opts = ClaudeAgentOptions(
                system_prompt=system or None,
                model=llm_request.model or self.model,
                tools=mcp_tool_names if mcp_tool_names else [],
                mcp_servers=mcp_servers,
                allowed_tools=mcp_tool_names,
                setting_sources=[],
                max_turns=5,
                hooks={"PreToolUse": [HookMatcher(hooks=[_defer_hook])]},
            )

            text_chunks: list[str] = []
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(user_prompt or "Please continue.")
                async for msg in client.receive_response():
                    # Pull plain text content; track tool_use via the
                    # hook (captured_tool_call) rather than mid-stream
                    # so we don't double-handle.
                    for block in getattr(msg, "content", []) or []:
                        btype = type(block).__name__
                        if btype == "TextBlock":
                            text = getattr(block, "text", None)
                            if text:
                                text_chunks.append(text)
                    # Stop iterating once Claude has acknowledged the
                    # deferred call (stop_reason=tool_deferred) — the
                    # rest of the stream is the SDK winding down and
                    # may include Claude's pre-defer apology text we
                    # want to discard.
                    stop_reason = getattr(msg, "stop_reason", None)
                    if stop_reason == "tool_deferred":
                        break

            parts: list[Any] = []
            # When we defer a tool, Claude often emits an apology text
            # block before the defer registers (e.g. "I apologize, the
            # tool was unavailable…"). That's an artifact of the defer
            # mechanism — discard it and surface only the function_call.
            if not captured_tool_call:
                joined = "".join(text_chunks).strip()
                if joined:
                    parts.append(genai_types.Part(text=joined))
            else:
                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=captured_tool_call["name"].removeprefix(
                                _MCP_TOOL_PREFIX
                            ),
                            args=captured_tool_call["args"],
                        )
                    )
                )

            if not parts:
                parts.append(genai_types.Part(text=""))

            yield LlmResponse(
                content=genai_types.Content(role="model", parts=parts),
                partial=False,
                turn_complete=True,
            )

    return ClaudeAgentSDKLlm


# Module-level alias for convenience: ``ClaudeAgentSDKLlm = ...``. Built
# lazily on first attribute access so that ``import
# goldfive.integrations.claude_sdk`` stays cheap and dependency-light.
def __getattr__(name: str) -> Any:
    if name == "ClaudeAgentSDKLlm":
        cls = make_claude_agent_sdk_llm_class()
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
