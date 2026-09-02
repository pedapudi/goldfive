"""Unit tests for :func:`goldfive.integrations.claude_sdk.make_call_llm`.

The factory feeds three goldfive call-sites (``LLMPlanner.call_llm``,
``LLMGoalDeriver.call_llm``, judge fallback via
``goldfive.wrap(call_llm=...)``) whose downstream parsers all assume
a well-formed string return — silent failures here surface much later
as "unparseable verdict" with no obvious cause. These tests pin the
contract:

1. ``ImportError`` propagates with the install hint when
   ``claude_agent_sdk`` is missing.
2. Multiple text content blocks across multiple yielded messages
   concatenate in stream order (the goldfive parser is
   position-sensitive when JSON wrappers span chunks).
3. Empty ``model`` (``""``) falls back to ``default_model`` —
   matches the convention used elsewhere in goldfive.
4. Empty ``system`` produces ``system_prompt=None`` (not ``""``) on
   the SDK options object.
5. An empty SDK stream returns ``""`` AND logs a WARNING so the
   silent-empty diagnostic case is observable.
6. ``setting_sources=[]`` and ``tools=[]`` are passed on every call —
   prevents operator-local Claude config (``CLAUDE.md``,
   ``~/.claude/settings.json``) from leaking into planner / judge
   prompts.
"""
from __future__ import annotations

import builtins
import logging
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake ``claude_agent_sdk`` module — injected via ``monkeypatch.setitem`` so
# the lazy import inside ``make_call_llm`` resolves to our stub, recording
# the exact ``ClaudeAgentOptions`` that goldfive constructs and feeding back
# whatever async-iterable of messages the test parameterised.
# ---------------------------------------------------------------------------


@dataclass
class _FakeOptions:
    """Recorder for the ``ClaudeAgentOptions`` ctor — captures everything
    goldfive passed so tests can assert on the actual ctor arguments."""

    system_prompt: Any = None
    model: Any = None
    tools: Any = None
    allowed_tools: Any = None
    setting_sources: Any = None
    max_turns: Any = None
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __init__(self, **kwargs: Any) -> None:
        # ``ClaudeAgentOptions`` in the real SDK is a dataclass with many
        # fields; capture the ones we care about and stash the rest so
        # tests can introspect.
        self.system_prompt = kwargs.pop("system_prompt", None)
        self.model = kwargs.pop("model", None)
        self.tools = kwargs.pop("tools", None)
        self.allowed_tools = kwargs.pop("allowed_tools", None)
        self.setting_sources = kwargs.pop("setting_sources", None)
        self.max_turns = kwargs.pop("max_turns", None)
        self.extra = kwargs


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeMessage:
    content: list[_FakeTextBlock]
    stop_reason: str | None = None


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    messages: list[_FakeMessage] | None = None,
) -> dict[str, Any]:
    """Inject a stub ``claude_agent_sdk`` module that records the last
    ``ClaudeAgentOptions`` ctor and yields ``messages`` (or none) from
    ``query``. Returns a recorder dict the test can inspect.

    Recorder keys:
        ``"options"``   — the most recent :class:`_FakeOptions` ctor call
        ``"prompt"``    — the most recent ``query()`` prompt argument
        ``"call_count"`` — number of times ``query()`` was iterated
    """
    recorder: dict[str, Any] = {"options": None, "prompt": None, "call_count": 0}

    async def _query(*, prompt: Any, options: Any) -> Any:  # type: ignore[no-redef]
        recorder["prompt"] = prompt
        recorder["options"] = options
        recorder["call_count"] += 1
        for msg in messages or ():
            yield msg

    fake_module = types.ModuleType("claude_agent_sdk")
    fake_module.ClaudeAgentOptions = _FakeOptions  # type: ignore[attr-defined]
    fake_module.query = _query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)
    return recorder


# Drop any stale ``goldfive.integrations.claude_sdk`` import so each test
# re-runs the lazy import from a clean slate. Otherwise ``make_call_llm``'s
# closure captures whichever ``claude_agent_sdk`` was first imported.
@pytest.fixture(autouse=True)
def _reset_integration_module() -> None:
    sys.modules.pop("goldfive.integrations.claude_sdk", None)


# ---------------------------------------------------------------------------
# 1. ImportError on missing SDK
# ---------------------------------------------------------------------------


def test_make_call_llm_raises_import_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``claude_agent_sdk`` is not installed, ``make_call_llm``
    raises :class:`ImportError` with the install-hint message. The hint
    is what tells users which package to pip install."""
    # Ensure no stale import — and also block any real import attempt.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # forces ImportError
    real_import = builtins.__import__

    def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
            raise ImportError("No module named 'claude_agent_sdk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    from goldfive.integrations.claude_sdk import make_call_llm

    with pytest.raises(ImportError) as excinfo:
        make_call_llm()
    assert "uv pip install claude-agent-sdk" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. Chunks concatenate in stream order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_call_llm_concatenates_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text content across multiple yielded messages and multiple
    blocks-per-message concatenates in stream order. Important because
    goldfive's JSON parsers sometimes see brace-balanced JSON spanning
    several content blocks."""
    _install_fake_sdk(
        monkeypatch,
        messages=[
            _FakeMessage(content=[_FakeTextBlock("alpha-"), _FakeTextBlock("beta-")]),
            _FakeMessage(content=[_FakeTextBlock("gamma")]),
        ],
    )

    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm()
    result = await call_llm("sys", "user prompt", "claude-haiku-4-5")
    assert result == "alpha-beta-gamma"


# ---------------------------------------------------------------------------
# 3. Empty model falls back to default_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_model_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model=""`` (or ``None``) → ``default_model`` is passed to the
    SDK. Goldfive's call-sites pass the configured model verbatim;
    this fallback is the integration's contract for "unspecified."""
    recorder = _install_fake_sdk(
        monkeypatch,
        messages=[_FakeMessage(content=[_FakeTextBlock("ok")])],
    )

    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm(default_model="claude-haiku-4-5")
    await call_llm("sys", "prompt", "")
    assert recorder["options"].model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 4. Empty system → system_prompt=None on the options
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_system_produces_none_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``system=""`` → ``system_prompt=None`` on the SDK options. The
    distinction matters: the SDK treats ``""`` as "use my empty system
    prompt" (which the bundled CLI may not honor) versus ``None`` which
    means "no override, use the SDK default" — and we explicitly want
    the latter for short structured prompts."""
    recorder = _install_fake_sdk(
        monkeypatch,
        messages=[_FakeMessage(content=[_FakeTextBlock("ok")])],
    )

    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm()
    await call_llm("", "prompt", "claude-haiku-4-5")
    assert recorder["options"].system_prompt is None


# ---------------------------------------------------------------------------
# 5. Empty stream returns "" AND logs WARNING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_stream_returns_empty_string_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An SDK stream that yields no text blocks returns ``""`` from the
    callable AND emits a WARNING log. The empty return is the classic
    "unparseable verdict" diagnostic dead-end pre-fix; the WARNING makes
    it observable so operators can correlate with model / turn config."""
    _install_fake_sdk(monkeypatch, messages=[])  # no messages at all

    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm()

    with caplog.at_level(logging.WARNING, logger="goldfive.integrations.claude_sdk"):
        result = await call_llm("sys", "prompt", "claude-haiku-4-5")
    assert result == ""
    warning_records = [
        r for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        "claude-agent-sdk produced no text" in r.getMessage()
        for r in warning_records
    ), f"expected zero-output WARNING, saw: {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# 6. setting_sources=[] and tools=[] passed on every call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_isolation_knobs_passed_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``setting_sources=[]`` and ``tools=[]`` must be on the
    ``ClaudeAgentOptions`` ctor for every invocation. Together they
    prevent operator-local Claude config (``CLAUDE.md`` / settings
    files / Claude Code built-in tools) from leaking into planner /
    judge prompts. Regression test: PR #378 originally shipped without
    ``setting_sources=[]`` and with ``allowed_tools=[]`` (the wrong
    knob); reviewer flagged it and that's now part of the contract."""
    recorder = _install_fake_sdk(
        monkeypatch,
        messages=[_FakeMessage(content=[_FakeTextBlock("ok")])],
    )

    from goldfive.integrations.claude_sdk import make_call_llm

    call_llm = make_call_llm()
    await call_llm("sys", "prompt", "claude-haiku-4-5")
    opts = recorder["options"]
    assert opts.setting_sources == [], (
        f"setting_sources must be [] (SDK isolation), got {opts.setting_sources!r}"
    )
    assert opts.tools == [], (
        f"tools must be [] (strips Claude Code built-ins), got {opts.tools!r}"
    )
