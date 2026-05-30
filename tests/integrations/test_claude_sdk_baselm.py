"""Unit tests for the ``ClaudeAgentSDKLlm`` adapter helpers.

These exercise the three non-trivial pure-function translators —
``_render_contents_as_transcript``, ``_extract_input_schema_from_adk_tool``,
``_genai_schema_to_json_schema`` — and the lazy ``__getattr__`` build
path for ``ClaudeAgentSDKLlm``.

None of these tests need a live ``claude-agent-sdk`` subprocess. The
SDK and (where relevant) ADK pieces are stubbed via ``sys.modules``
injection so we can assert on the actual translator outputs without
spinning up Claude Code.
"""
from __future__ import annotations

import logging
import sys
import types
import typing
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared genai-types stub. The real ``google.genai.types`` module is a thin
# pydantic-wrapping shim; the helpers under test only use it via the
# ``type``/``properties``/``items``/``required`` attribute surface, so a
# dataclass façade is plenty.
# ---------------------------------------------------------------------------


@dataclass
class _StubGenaiSchema:
    """Minimal ``google.genai.types.Schema`` look-alike."""

    type: str | None = None
    type_: str | None = None
    description: str | None = None
    properties: dict[str, Any] | None = None
    items: Any = None
    required: list[str] | None = None
    enum: list[Any] | None = None
    nullable: bool | None = None
    format: str | None = None
    default: Any = None
    pattern: str | None = None


@dataclass
class _StubFunctionDeclaration:
    parameters: Any = None
    parameters_json_schema: Any = None


@dataclass
class _StubFunctionCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubFunctionResponse:
    name: str
    response: Any = None


@dataclass
class _StubPart:
    text: str | None = None
    function_call: _StubFunctionCall | None = None
    function_response: _StubFunctionResponse | None = None


@dataclass
class _StubContent:
    role: str
    parts: list[_StubPart] = field(default_factory=list)


# ---------------------------------------------------------------------------
# _render_contents_as_transcript
# ---------------------------------------------------------------------------


class TestRenderContentsAsTranscript:
    """Branch coverage on the conversation-to-transcript translator.

    Every shape this function emits feeds Claude as its prior-history
    description; if any branch silently drops content the agent loses
    state across turns.
    """

    def setup_method(self) -> None:
        # Lazy import so a missing ``claude_agent_sdk`` doesn't block
        # tests that only touch the pure-function translators.
        from goldfive.integrations.claude_sdk import _render_contents_as_transcript
        self.render = _render_contents_as_transcript

    def test_user_text_renders_with_user_prefix(self) -> None:
        contents = [_StubContent(role="user", parts=[_StubPart(text="hello")])]
        assert self.render(contents) == "User: hello"

    def test_model_text_renders_with_assistant_prefix(self) -> None:
        contents = [_StubContent(role="model", parts=[_StubPart(text="hi back")])]
        assert self.render(contents) == "Assistant: hi back"

    def test_function_call_renders_as_descriptive_line(self) -> None:
        contents = [
            _StubContent(
                role="model",
                parts=[
                    _StubPart(
                        function_call=_StubFunctionCall(
                            name="get_weather", args={"city": "SF"}
                        )
                    )
                ],
            )
        ]
        rendered = self.render(contents)
        # JSON dict args, named tool — operator can read it in logs.
        assert "Assistant called tool `get_weather`" in rendered
        assert '"city": "SF"' in rendered

    def test_function_response_renders_as_tool_result_line(self) -> None:
        contents = [
            _StubContent(
                role="user",
                parts=[
                    _StubPart(
                        function_response=_StubFunctionResponse(
                            name="get_weather", response={"output": "sunny"}
                        )
                    )
                ],
            )
        ]
        rendered = self.render(contents)
        assert "Tool result (`get_weather`):" in rendered
        assert '"output": "sunny"' in rendered

    def test_empty_contents_returns_empty_string(self) -> None:
        """A first-turn ``LlmRequest`` has no prior history yet — must
        not emit a spurious transcript line."""
        assert self.render([]) == ""
        assert self.render(None) == ""  # type: ignore[arg-type]

    def test_mixed_sequence_preserves_order(self) -> None:
        """User → model+tool_call → user+tool_response — Claude has to
        see them in chronological order to continue the conversation."""
        contents = [
            _StubContent(role="user", parts=[_StubPart(text="ask")]),
            _StubContent(
                role="model",
                parts=[
                    _StubPart(text="thinking..."),
                    _StubPart(
                        function_call=_StubFunctionCall(name="t", args={"a": 1})
                    ),
                ],
            ),
            _StubContent(
                role="user",
                parts=[
                    _StubPart(
                        function_response=_StubFunctionResponse(
                            name="t", response={"ok": True}
                        )
                    )
                ],
            ),
        ]
        rendered = self.render(contents)
        lines = rendered.splitlines()
        assert lines[0] == "User: ask"
        assert lines[1] == "Assistant: thinking..."
        assert "Assistant called tool `t`" in lines[2]
        assert "Tool result (`t`):" in lines[3]


# ---------------------------------------------------------------------------
# _genai_schema_to_json_schema
# ---------------------------------------------------------------------------


class TestGenaiSchemaToJsonSchema:
    """Schema-proto → JSON-Schema-dict conversion.

    The fidelity here matters: dropped fields surface as Claude
    constructing tool calls that ADK's real schema then rejects,
    triggering retry loops noisy in logs and chewing into the
    per-call ``max_turns`` budget.
    """

    def setup_method(self) -> None:
        from goldfive.integrations.claude_sdk import _genai_schema_to_json_schema
        self.convert = _genai_schema_to_json_schema

    def test_nested_object_properties(self) -> None:
        inner = _StubGenaiSchema(type="STRING", description="inner field")
        outer = _StubGenaiSchema(
            type="OBJECT",
            properties={"inner": inner},
            required=["inner"],
        )
        result = self.convert(outer)
        assert result["type"] == "object"
        assert result["required"] == ["inner"]
        assert result["properties"]["inner"]["type"] == "string"
        assert result["properties"]["inner"]["description"] == "inner field"

    def test_array_of_objects(self) -> None:
        item = _StubGenaiSchema(
            type="OBJECT",
            properties={"name": _StubGenaiSchema(type="STRING")},
            required=["name"],
        )
        arr = _StubGenaiSchema(type="ARRAY", items=item)
        result = self.convert(arr)
        assert result["type"] == "array"
        assert result["items"]["type"] == "object"
        assert result["items"]["required"] == ["name"]
        assert result["items"]["properties"]["name"]["type"] == "string"

    def test_enum_pass_through(self) -> None:
        s = _StubGenaiSchema(type="STRING", enum=["low", "medium", "high"])
        result = self.convert(s)
        assert result["enum"] == ["low", "medium", "high"]

    def test_type_underscore_attr_is_honored(self) -> None:
        """Some genai proto bindings expose ``type_`` (trailing
        underscore) instead of ``type``. The translator should handle
        both — Schema bindings vary across protobuf versions."""
        s = _StubGenaiSchema(type=None, type_="STRING")
        result = self.convert(s)
        assert result["type"] == "string"

    def test_constraint_hints_pass_through(self) -> None:
        """``nullable``, ``format``, ``default``, ``pattern`` are
        constraint hints Claude and ADK both use. Dropping them
        produces malformed values ADK rejects."""
        s = _StubGenaiSchema(
            type="STRING",
            nullable=True,
            format="uri",
            default="https://example.com",
            pattern=r"^https?://",
        )
        result = self.convert(s)
        assert result["nullable"] is True
        assert result["format"] == "uri"
        assert result["default"] == "https://example.com"
        assert result["pattern"] == r"^https?://"

    def test_none_schema_returns_empty_object(self) -> None:
        """Defensive: ADK occasionally hands us ``parameters=None``."""
        result = self.convert(None)
        assert result == {"type": "object", "properties": {}, "required": []}

    def test_object_type_always_has_properties_key(self) -> None:
        """JSON-Schema convention: an ``object`` always has a
        ``properties`` key even when empty. Keeps downstream consumers
        from having to special-case missing keys."""
        s = _StubGenaiSchema(type="OBJECT")
        result = self.convert(s)
        assert result["type"] == "object"
        assert result["properties"] == {}


# ---------------------------------------------------------------------------
# _extract_input_schema_from_adk_tool
# ---------------------------------------------------------------------------


class _FakeFunctionTool:
    """``FunctionTool`` look-alike: has ``.func`` and
    ``_get_declaration`` returning a Schema."""

    def __init__(self, func: Any, declaration: Any | None = None) -> None:
        self.func = func
        self._declaration = declaration

    def _get_declaration(self) -> Any:
        return self._declaration


class _FakeAgentTool:
    """``AgentTool`` look-alike: NO ``.func`` attribute (this is the
    regression the PR's schema extractor fixes — falling back to
    signature would produce an empty schema)."""

    def __init__(self, declaration: Any) -> None:
        self._declaration = declaration

    def _get_declaration(self) -> Any:
        return self._declaration


class _FakeRaisingTool:
    """Tool whose ``_get_declaration`` raises — the extractor must
    fall back to the signature path instead of bubbling the error."""

    def __init__(self, func: Any) -> None:
        self.func = func

    def _get_declaration(self) -> Any:
        raise RuntimeError("declaration unavailable")


class TestExtractInputSchemaFromAdkTool:
    """Schema extraction for both common ADK ``BaseTool`` shapes.

    The ``AgentTool`` branch is the regression pin for PR #383's fix —
    without the declaration-first path, Claude calls the AgentTool
    with empty args and ADK raises ``KeyError('request')``.
    """

    def setup_method(self) -> None:
        from goldfive.integrations.claude_sdk import (
            _extract_input_schema_from_adk_tool,
        )
        self.extract = _extract_input_schema_from_adk_tool

    def test_function_tool_with_declaration_uses_schema_path(self) -> None:
        def my_func(topic: str) -> str:
            return ""

        declaration = _StubFunctionDeclaration(
            parameters=_StubGenaiSchema(
                type="OBJECT",
                properties={"topic": _StubGenaiSchema(type="STRING")},
                required=["topic"],
            )
        )
        result = self.extract(_FakeFunctionTool(my_func, declaration))
        assert result["type"] == "object"
        assert result["properties"]["topic"]["type"] == "string"
        assert result["required"] == ["topic"]

    def test_agent_tool_uses_declaration_request_parameter(self) -> None:
        """The exact regression that motivated this PR's fix — without
        going through ``_get_declaration`` we'd return ``{}`` and ADK
        would raise ``KeyError('request')``."""
        declaration = _StubFunctionDeclaration(
            parameters=_StubGenaiSchema(
                type="OBJECT",
                properties={"request": _StubGenaiSchema(type="STRING")},
                required=["request"],
            )
        )
        result = self.extract(_FakeAgentTool(declaration))
        assert result["required"] == ["request"]
        assert result["properties"]["request"]["type"] == "string"

    def test_parameters_json_schema_preferred_over_proto_schema(self) -> None:
        """If ADK populates ``parameters_json_schema`` instead of the
        proto Schema (gated on ``FeatureName.JSON_SCHEMA_FOR_FUNC_DECL``),
        the extractor must read it directly. Otherwise a future flag
        flip silently breaks every AgentTool — same failure mode as
        the bug we just fixed."""
        json_schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        declaration = _StubFunctionDeclaration(parameters_json_schema=json_schema)
        result = self.extract(_FakeAgentTool(declaration))
        assert result == json_schema

    def test_raising_declaration_falls_back_to_signature(self) -> None:
        def takes_count(count=5):  # type: ignore[no-untyped-def]
            return ""

        result = self.extract(_FakeRaisingTool(takes_count))
        # Signature-path schema: ``count`` is optional (default=5), so
        # not in ``required``. Type defaults to ``string`` since the
        # signature has no concrete annotation (the signature
        # extractor's pre-existing behaviour for non-annotated params).
        assert "count" in result["properties"]
        assert "count" not in result.get("required", [])

    def test_bare_callable_without_declaration_uses_signature(self) -> None:
        """Some operator-supplied tools may be raw callables — the
        extractor should still build a usable schema from inspection."""

        class _BareTool:
            def __init__(self, func: Any) -> None:
                self.func = func

            # No _get_declaration / declaration / parameters at all.

        def echo(message: str) -> str:
            return message

        result = self.extract(_BareTool(echo))
        assert result["properties"]["message"]["type"] == "string"
        assert result["required"] == ["message"]

    def test_completely_opaque_tool_returns_empty_object(self) -> None:
        """Last-resort fallback. Should never silently corrupt the
        downstream call — but if a caller hands us a tool we can't
        introspect at all, we return a well-formed empty object
        schema rather than raising."""

        class _Opaque:
            pass

        result = self.extract(_Opaque())
        assert result == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# __getattr__ lazy build
# ---------------------------------------------------------------------------


class TestLazyClassBuild:
    """The module's ``__getattr__`` builds ``ClaudeAgentSDKLlm`` on
    first request. Two failure modes worth pinning:

    1. Unknown attribute → ``AttributeError`` (so ``hasattr`` works
       cleanly for any unrelated probing).
    2. Known attribute but the SDK is missing → ``ImportError``
       propagates with the install hint. This is the documented
       trade-off; ``hasattr`` callers who want soft-check semantics
       have to catch ``ImportError`` explicitly.
    """

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import goldfive.integrations.claude_sdk as mod

        with pytest.raises(AttributeError):
            mod.SomeNonExistentName  # type: ignore[attr-defined]

    def test_known_attribute_with_sdk_missing_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``claude-agent-sdk`` isn't importable, the lazy build
        of ``ClaudeAgentSDKLlm`` raises ``ImportError`` (not
        ``AttributeError``) — the more informative error pointing the
        operator at the actual missing dependency."""
        import builtins

        # Force the lazy build to run again on this access (any prior
        # successful build would have cached the class onto the module
        # globals, hiding the import path).
        sys.modules.pop("goldfive.integrations.claude_sdk", None)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                raise ImportError("No module named 'claude_agent_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        import goldfive.integrations.claude_sdk as mod

        with pytest.raises(ImportError) as excinfo:
            mod.ClaudeAgentSDKLlm  # type: ignore[attr-defined]
        assert "claude-agent-sdk" in str(excinfo.value)


# ---------------------------------------------------------------------------
# SDK ``defer`` contract guard (_defer_in_permission_decision /
# _verify_defer_contract)
# ---------------------------------------------------------------------------
#
# These pin the "fail loudly at the seam" check: the adapter pins
# ``claude-agent-sdk>=0.1.80`` with no ceiling and keys runtime behaviour
# on the ``permissionDecision="defer"`` literal + the ``DeferredToolUse``
# envelope. A future SDK that drops either should fail at class-build
# time, not as a silent mid-run hang.


def _ns_with_permission_decision(literal: Any) -> Any:
    """Build a namespace holding one TypedDict-shaped object whose
    ``permissionDecision`` annotation is ``literal`` (a real typing
    object, not a stringized annotation — this module uses
    ``from __future__ import annotations`` so we assign ``__annotations__``
    directly to dodge stringization)."""
    hook_out = type("PreToolUseHookSpecificOutput", (), {})
    hook_out.__annotations__ = {"permissionDecision": literal}
    return types.SimpleNamespace(PreToolUseHookSpecificOutput=hook_out)


class TestDeferContractGuard:
    def setup_method(self) -> None:
        from goldfive.integrations.claude_sdk import (
            _defer_in_permission_decision,
            _verify_defer_contract,
        )
        self.detect = _defer_in_permission_decision
        self.verify = _verify_defer_contract

    def test_literal_with_defer_detected(self) -> None:
        ns = _ns_with_permission_decision(
            typing.Literal["allow", "deny", "ask", "defer"]
        )
        assert self.detect(ns) is True

    def test_literal_wrapped_in_not_required_detected(self) -> None:
        ns = _ns_with_permission_decision(
            typing.Optional[typing.Literal["allow", "deny", "defer"]]
        )
        assert self.detect(ns) is True

    def test_literal_without_defer_is_false(self) -> None:
        ns = _ns_with_permission_decision(typing.Literal["allow", "deny", "ask"])
        assert self.detect(ns) is False

    def test_no_permission_decision_field_returns_none(self) -> None:
        """Can't locate the field → ``None`` (cannot verify), so the
        guard stays lenient rather than false-failing on layout drift."""
        assert self.detect(types.SimpleNamespace(Unrelated=object())) is None

    def test_verify_raises_when_deferredtooluse_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_types = types.ModuleType("claude_agent_sdk.types")
        fake_types.PreToolUseHookSpecificOutput = (  # type: ignore[attr-defined]
            _ns_with_permission_decision(
                typing.Literal["allow", "deny", "ask", "defer"]
            ).PreToolUseHookSpecificOutput
        )
        # No DeferredToolUse attribute.
        fake_pkg = types.ModuleType("claude_agent_sdk")
        fake_pkg.types = fake_types  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_pkg)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", fake_types)

        with pytest.raises(RuntimeError) as excinfo:
            self.verify()
        assert "DeferredToolUse" in str(excinfo.value)

    def test_verify_raises_when_defer_dropped_from_literal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_types = types.ModuleType("claude_agent_sdk.types")
        fake_types.PreToolUseHookSpecificOutput = (  # type: ignore[attr-defined]
            _ns_with_permission_decision(
                typing.Literal["allow", "deny", "ask"]
            ).PreToolUseHookSpecificOutput
        )
        fake_types.DeferredToolUse = type("DeferredToolUse", (), {})  # type: ignore[attr-defined]
        fake_pkg = types.ModuleType("claude_agent_sdk")
        fake_pkg.types = fake_types  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_pkg)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", fake_types)

        with pytest.raises(RuntimeError) as excinfo:
            self.verify()
        assert "permissionDecision" in str(excinfo.value)

    def test_verify_passes_with_full_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_types = types.ModuleType("claude_agent_sdk.types")
        fake_types.PreToolUseHookSpecificOutput = (  # type: ignore[attr-defined]
            _ns_with_permission_decision(
                typing.Literal["allow", "deny", "ask", "defer"]
            ).PreToolUseHookSpecificOutput
        )
        fake_types.DeferredToolUse = type("DeferredToolUse", (), {})  # type: ignore[attr-defined]
        fake_pkg = types.ModuleType("claude_agent_sdk")
        fake_pkg.types = fake_types  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_pkg)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", fake_types)

        # Should not raise.
        self.verify()


# ---------------------------------------------------------------------------
# generate_content_async — the high-risk runtime path. Fakes both the SDK
# (ClaudeSDKClient + the PreToolUse hook dispatch) and the ADK / genai
# surface via sys.modules injection so we can drive real message streams
# through the adapter and assert on the emitted ``LlmResponse`` parts,
# the defer-vs-deny hook branch, the preamble/post-tool-use text split,
# the empty-output WARNING, the parallel-fan-out WARNING, and
# usage_metadata translation.
# ---------------------------------------------------------------------------


# Block stubs — the adapter dispatches on ``type(block).__name__`` so the
# class names here MUST be exactly ``TextBlock`` / ``ToolUseBlock``.
class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    def __init__(self, name: str, input: dict[str, Any] | None = None) -> None:
        self.name = name
        self.input = input or {}


class _SdkMessage:
    """Stand-in for AssistantMessage / ResultMessage. ``usage`` is read
    off whichever message reports it; ``stop_reason == 'tool_deferred'``
    terminates the adapter's receive loop."""

    def __init__(
        self,
        content: list[Any] | None = None,
        stop_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.content = content or []
        self.stop_reason = stop_reason
        self.usage = usage


def _install_baselm_env(
    monkeypatch: pytest.MonkeyPatch, *, messages: list[_SdkMessage]
) -> dict[str, Any]:
    """Inject fake ``claude_agent_sdk`` (+ ``.types``) and the ADK / genai
    modules the BaseLlm adapter imports. The fake ``ClaudeSDKClient``
    replays ``messages`` and, for each ``ToolUseBlock``, invokes the
    registered ``PreToolUse`` hook until one returns ``defer`` (mirroring
    the SDK halting the run on the first defer). Returns a recorder."""
    recorder: dict[str, Any] = {"hook_decisions": [], "options": None}

    class _FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class _FakeHookMatcher:
        def __init__(self, hooks: list[Any]) -> None:
            self.hooks = hooks

    def _fake_create_sdk_mcp_server(name: str, tools: list[Any]) -> Any:
        return {"name": name, "tools": tools}

    def _fake_tool(name: str, description: str, input_schema: Any) -> Any:
        def _wrap(handler: Any) -> Any:
            return handler
        return _wrap

    class _FakeClient:
        def __init__(self, options: Any) -> None:
            self.options = options
            recorder["options"] = options

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(self, prompt: str) -> None:
            recorder["prompt"] = prompt

        async def receive_response(self) -> Any:
            hook = None
            hooks_cfg = getattr(self.options, "hooks", {}) or {}
            matchers = hooks_cfg.get("PreToolUse", [])
            if matchers:
                hook = matchers[0].hooks[0]
            for msg in messages:
                halted = False
                if hook is not None:
                    for block in msg.content:
                        if type(block).__name__ != "ToolUseBlock":
                            continue
                        decision = await hook(
                            {
                                "tool_name": block.name,
                                "tool_input": getattr(block, "input", {}),
                            },
                            f"tu_{block.name}",
                            None,
                        )
                        recorder["hook_decisions"].append(decision)
                        pd = (
                            decision.get("hookSpecificOutput", {})
                            .get("permissionDecision")
                        )
                        if pd == "defer":
                            # SDK halts the run on the first defer; do not
                            # fire hooks for subsequent blocks this turn.
                            halted = True
                            break
                yield msg
                if halted:
                    break

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = _FakeOptions  # type: ignore[attr-defined]
    fake_sdk.ClaudeSDKClient = _FakeClient  # type: ignore[attr-defined]
    fake_sdk.create_sdk_mcp_server = _fake_create_sdk_mcp_server  # type: ignore[attr-defined]
    fake_sdk.tool = _fake_tool  # type: ignore[attr-defined]

    fake_sdk_types = types.ModuleType("claude_agent_sdk.types")
    fake_sdk_types.HookMatcher = _FakeHookMatcher  # type: ignore[attr-defined]
    fake_sdk_types.DeferredToolUse = type("DeferredToolUse", (), {})  # type: ignore[attr-defined]
    _hook_out = type("PreToolUseHookSpecificOutput", (), {})
    _hook_out.__annotations__ = {
        "permissionDecision": typing.Literal["allow", "deny", "ask", "defer"]
    }
    fake_sdk_types.PreToolUseHookSpecificOutput = _hook_out  # type: ignore[attr-defined]
    fake_sdk.types = fake_sdk_types  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", fake_sdk_types)

    # --- ADK + genai surface ------------------------------------------------
    class _FakeBaseLlm:
        def __init__(self, model: str | None = None, **kwargs: Any) -> None:
            self.model = model

    class _FakeLlmResponse:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class _FakeLlmRequest:  # only used as a type symbol in the adapter
        pass

    class _GFunctionCall:
        def __init__(self, name: str, args: dict[str, Any]) -> None:
            self.name = name
            self.args = args

    class _GPart:
        def __init__(
            self, text: str | None = None, function_call: Any = None
        ) -> None:
            self.text = text
            self.function_call = function_call

    class _GContent:
        def __init__(self, role: str, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    class _GUsage:
        def __init__(
            self,
            prompt_token_count: Any = None,
            candidates_token_count: Any = None,
            total_token_count: Any = None,
        ) -> None:
            self.prompt_token_count = prompt_token_count
            self.candidates_token_count = candidates_token_count
            self.total_token_count = total_token_count

    genai_types = types.ModuleType("google.genai.types")
    genai_types.Part = _GPart  # type: ignore[attr-defined]
    genai_types.FunctionCall = _GFunctionCall  # type: ignore[attr-defined]
    genai_types.Content = _GContent  # type: ignore[attr-defined]
    genai_types.GenerateContentResponseUsageMetadata = _GUsage  # type: ignore[attr-defined]

    google_mod = types.ModuleType("google")
    google_genai = types.ModuleType("google.genai")
    google_genai.types = genai_types  # type: ignore[attr-defined]
    google_adk = types.ModuleType("google.adk")
    google_adk_models = types.ModuleType("google.adk.models")
    base_llm_mod = types.ModuleType("google.adk.models.base_llm")
    base_llm_mod.BaseLlm = _FakeBaseLlm  # type: ignore[attr-defined]
    llm_request_mod = types.ModuleType("google.adk.models.llm_request")
    llm_request_mod.LlmRequest = _FakeLlmRequest  # type: ignore[attr-defined]
    llm_response_mod = types.ModuleType("google.adk.models.llm_response")
    llm_response_mod.LlmResponse = _FakeLlmResponse  # type: ignore[attr-defined]

    for name, mod in {
        "google": google_mod,
        "google.genai": google_genai,
        "google.genai.types": genai_types,
        "google.adk": google_adk,
        "google.adk.models": google_adk_models,
        "google.adk.models.base_llm": base_llm_mod,
        "google.adk.models.llm_request": llm_request_mod,
        "google.adk.models.llm_response": llm_response_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    recorder["genai_types"] = genai_types
    return recorder


@dataclass
class _FakeTool:
    """Minimal ADK tool: a description + a declaration so schema
    extraction succeeds. Schema content is irrelevant to these tests."""

    description: str = "a tool"

    def _get_declaration(self) -> Any:
        return _StubFunctionDeclaration(
            parameters=_StubGenaiSchema(
                type="OBJECT",
                properties={"city": _StubGenaiSchema(type="STRING")},
                required=["city"],
            )
        )


@dataclass
class _FakeLlmReq:
    """Duck-typed ``LlmRequest`` the adapter reads from."""

    contents: list[Any] = field(default_factory=list)
    tools_dict: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    config: Any = None


async def _run_adapter(
    recorder_env: Any, req: _FakeLlmReq, *, model: str = "claude-haiku-4-5"
) -> list[Any]:
    from goldfive.integrations.claude_sdk import make_claude_agent_sdk_llm_class

    cls = make_claude_agent_sdk_llm_class()
    llm = cls(model=model)
    return [r async for r in llm.generate_content_async(req)]


class TestGenerateContentAsync:
    """Drive real message streams through ``generate_content_async``."""

    @pytest.mark.asyncio
    async def test_single_tool_call_becomes_function_call_part(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[
                        ToolUseBlock("mcp__adk__get_weather", {"city": "SF"})
                    ],
                    stop_reason="tool_deferred",
                )
            ],
        )
        req = _FakeLlmReq(tools_dict={"get_weather": _FakeTool()})
        responses = await _run_adapter(env, req)
        assert len(responses) == 1
        parts = responses[0].content.parts
        fcs = [p for p in parts if p.function_call is not None]
        assert len(fcs) == 1
        # ``mcp__adk__`` prefix stripped back to the ADK tool name.
        assert fcs[0].function_call.name == "get_weather"
        assert fcs[0].function_call.args == {"city": "SF"}
        # Hook deferred exactly once.
        decisions = env["hook_decisions"]
        assert len(decisions) == 1
        assert (
            decisions[0]["hookSpecificOutput"]["permissionDecision"] == "defer"
        )

    @pytest.mark.asyncio
    async def test_preamble_text_kept_post_tool_text_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[
                        TextBlock("Let me check the weather."),
                        ToolUseBlock("mcp__adk__get_weather", {"city": "SF"}),
                        TextBlock("(tool was deferred)"),  # post-tool: discard
                    ],
                    stop_reason="tool_deferred",
                )
            ],
        )
        req = _FakeLlmReq(tools_dict={"get_weather": _FakeTool()})
        responses = await _run_adapter(env, req)
        parts = responses[0].content.parts
        texts = [p.text for p in parts if p.text]
        assert texts == ["Let me check the weather."]
        assert any(p.function_call is not None for p in parts)

    @pytest.mark.asyncio
    async def test_deny_branch_for_non_adk_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool outside the ADK allowlist (e.g. a leaked cloud-account
        MCP server) is denied, not captured; the run continues and the
        real ADK tool is deferred."""
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[ToolUseBlock("mcp__claude_ai_Drive__create_file")]
                ),
                _SdkMessage(
                    content=[
                        ToolUseBlock("mcp__adk__get_weather", {"city": "SF"})
                    ],
                    stop_reason="tool_deferred",
                ),
            ],
        )
        req = _FakeLlmReq(tools_dict={"get_weather": _FakeTool()})
        responses = await _run_adapter(env, req)
        decisions = env["hook_decisions"]
        kinds = [
            d["hookSpecificOutput"]["permissionDecision"] for d in decisions
        ]
        assert kinds == ["deny", "defer"]
        # Denied tool names the valid options in its reason.
        assert "get_weather" in decisions[0]["hookSpecificOutput"][
            "permissionDecisionReason"
        ]
        parts = responses[0].content.parts
        fcs = [p for p in parts if p.function_call is not None]
        assert len(fcs) == 1
        assert fcs[0].function_call.name == "get_weather"

    @pytest.mark.asyncio
    async def test_parallel_attempt_warns_and_serialises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two ADK tool calls in one assistant message: the SDK halts at
        the first defer so only one is captured. The adapter still counts
        both ``tool_use`` blocks and WARNs that the model fanned out and
        the rest serialise across ADK turns."""
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[
                        ToolUseBlock("mcp__adk__get_weather", {"city": "Tokyo"}),
                        ToolUseBlock("mcp__adk__get_weather", {"city": "NYC"}),
                    ],
                    stop_reason="tool_deferred",
                )
            ],
        )
        req = _FakeLlmReq(tools_dict={"get_weather": _FakeTool()})
        with caplog.at_level(
            logging.WARNING, logger="goldfive.integrations.claude_sdk"
        ):
            responses = await _run_adapter(env, req)
        # Only the first call captured (one defer before halt).
        fcs = [p for p in responses[0].content.parts if p.function_call]
        assert len(fcs) == 1
        assert fcs[0].function_call.args == {"city": "Tokyo"}
        assert any(
            "emitted 2 ADK tool calls in one turn" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_empty_output_warns_and_emits_empty_part(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No preamble, no tool call → one empty-text Part AND a WARNING
        (parity with ``make_call_llm``'s zero-output diagnostic)."""
        env = _install_baselm_env(
            monkeypatch,
            messages=[_SdkMessage(content=[], stop_reason="end_turn")],
        )
        req = _FakeLlmReq(tools_dict={})
        with caplog.at_level(
            logging.WARNING, logger="goldfive.integrations.claude_sdk"
        ):
            responses = await _run_adapter(env, req)
        parts = responses[0].content.parts
        assert len(parts) == 1
        assert parts[0].text == ""
        assert any(
            "no text and no tool call" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_usage_metadata_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[TextBlock("final answer")],
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )
            ],
        )
        req = _FakeLlmReq(tools_dict={})
        responses = await _run_adapter(env, req)
        um = responses[0].usage_metadata
        assert um is not None
        assert um.prompt_token_count == 10
        assert um.candidates_token_count == 5
        assert um.total_token_count == 15

    @pytest.mark.asyncio
    async def test_plain_text_answer_no_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _install_baselm_env(
            monkeypatch,
            messages=[
                _SdkMessage(
                    content=[TextBlock("the answer is 42")],
                    stop_reason="end_turn",
                )
            ],
        )
        req = _FakeLlmReq(tools_dict={})
        responses = await _run_adapter(env, req)
        parts = responses[0].content.parts
        assert [p.text for p in parts if p.text] == ["the answer is 42"]
        assert all(p.function_call is None for p in parts)


# Module-level fixture: a clean import slate per test, like the
# companion ``test_claude_sdk_call_llm.py`` file.
@pytest.fixture(autouse=True)
def _reset_module_cache() -> None:
    sys.modules.pop("goldfive.integrations.claude_sdk", None)
