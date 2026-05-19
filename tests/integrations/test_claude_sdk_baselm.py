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

import sys
import types
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


# Module-level fixture: a clean import slate per test, like the
# companion ``test_claude_sdk_call_llm.py`` file.
@pytest.fixture(autouse=True)
def _reset_module_cache() -> None:
    sys.modules.pop("goldfive.integrations.claude_sdk", None)
