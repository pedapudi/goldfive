"""Claude-adapter observer-note surface: fail-passive kill-switch +
truthful delivery under factory-attached client options.

Two fixes under test (agency-preservation fix wave):

1. ``ClaudeAgentSDKAdapter._consume_observer_note`` reads the kill-switch
   via :func:`goldfive.steerer.steering_is_active` — the documented,
   FAIL-PASSIVE accessor. The previous private ``_should_inject`` read
   defaulted to ``True`` when the predicate was missing/raising, i.e. a
   stub steerer started INJECTING — the inverse of the contract every
   other surface follows.

2. When the client factory attaches its own ``options``, goldfive's
   options object (system prompt + hooks) never reaches the client. The
   adapter must NOT consume the observer note on that path (that would
   mark a render that never happened); instead it merges the
   ``PostToolUse`` note hook into a COPY of the operator options so
   pending notes still deliver truthfully.

These tests run without ``claude_agent_sdk`` installed: the SDK-touching
paths are exercised against a minimal fake SDK monkeypatched onto the
adapter module (the adapter only needs ``HookMatcher`` /
``ClaudeAgentOptions`` / ``ResultMessage`` shapes here).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

import goldfive.adapters.claude as claude_mod
from goldfive.adapters.claude import ClaudeAgentSDKAdapter
from goldfive.config import SteeringConfig
from goldfive.observer_note_queue import ObserverNoteQueue
from goldfive.steerer import DefaultSteerer
from goldfive.types import Session, Task

# ---------------------------------------------------------------------------
# Fake SDK (only the shapes the adapter touches on these paths)
# ---------------------------------------------------------------------------


class _FakeHookMatcher:
    def __init__(self, matcher: str | None = None, hooks: list[Any] | None = None) -> None:
        self.matcher = matcher
        self.hooks = list(hooks or [])


class _FakeOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.hooks: Any = None
        self.system_prompt: str = ""
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResultMessage:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.stop_reason = stop_reason
        self.result = "final"


class _FakeSdk:
    HookMatcher = _FakeHookMatcher
    ClaudeAgentOptions = _FakeOptions
    ResultMessage = _FakeResultMessage


class _StubClient:
    def __init__(self, options: Any = None) -> None:
        self.options: Any = options
        self.queries: list[str] = []

    async def connect(self, prompt: Any = None) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.queries.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        yield _FakeResultMessage()


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
    sdk = _FakeSdk()
    monkeypatch.setattr(claude_mod, "_sdk", sdk)
    monkeypatch.setattr(claude_mod, "_SDK_IMPORT_ERROR", None)
    return sdk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_with_note(*, agent_id: str = "") -> Session:
    session = Session(run_id="r-claude-truth")
    ObserverNoteQueue.for_session(session).enqueue(
        body="Observation: repeated identical tool call.",
        observation="repeated identical tool call",
        severity="warning",
        drift_id="d-claude-truth",
        kind="looping_tool_call",
        task_id="t1",
        agent_id=agent_id,
    )
    return session


def _adapter(steerer: Any) -> ClaudeAgentSDKAdapter:
    # Bypass __init__'s _require_sdk (the fake is installed by the
    # fixture for invoke-path tests; the _consume_observer_note tests
    # never touch the SDK at all).
    adapter = object.__new__(ClaudeAgentSDKAdapter)
    adapter._client_factory = lambda: _StubClient()
    adapter._steerer = steerer
    adapter._template = None
    adapter._model = None
    adapter._available_agents = []
    adapter._reporting_specs = {}
    adapter._mcp_server_config = None
    adapter._mcp_tool_names = []
    return adapter


def _steerer(*, observation_only: bool) -> DefaultSteerer:
    return DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=observation_only,
            signal_channel="request_context",
        )
    )


def _task() -> Task:
    return Task(id="t1", title="Draft", assignee_agent_id="writer")


# ---------------------------------------------------------------------------
# Finding 1 — fail-passive kill-switch on the note surfaces
# ---------------------------------------------------------------------------


async def test_note_renders_under_active_steering() -> None:
    adapter = _adapter(_steerer(observation_only=False))
    session = _session_with_note()
    block = await adapter._consume_observer_note(
        session, surface="claude_system_prompt", current_agent_id="writer"
    )
    assert block is not None and "repeated identical tool call" in block
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and not notes[0].delivered_dry_run


async def test_note_not_rendered_under_observation_only() -> None:
    adapter = _adapter(_steerer(observation_only=True))
    session = _session_with_note()
    block = await adapter._consume_observer_note(
        session, surface="claude_system_prompt", current_agent_id="writer"
    )
    assert block is None
    # Consumed as a DRY-RUN delivery (decision parity), never shown.
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and notes[0].delivered_dry_run


async def test_note_not_rendered_when_steerer_lacks_predicate() -> None:
    """FAIL-PASSIVE: a stub steerer missing ``is_active_steering`` must
    NOT render (the retired fail-open ``_should_inject`` default-True
    read rendered here)."""

    class _StubSteerer:
        _signal_channel = "request_context"

    adapter = _adapter(_StubSteerer())
    session = _session_with_note()
    block = await adapter._consume_observer_note(
        session, surface="claude_system_prompt", current_agent_id="writer"
    )
    assert block is None
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and notes[0].delivered_dry_run


async def test_note_not_rendered_when_predicate_raises() -> None:
    class _RaisingSteerer:
        _signal_channel = "request_context"

        def is_active_steering(self) -> bool:
            raise RuntimeError("boom")

    adapter = _adapter(_RaisingSteerer())
    session = _session_with_note()
    block = await adapter._consume_observer_note(
        session, surface="claude_posttooluse", current_agent_id="writer"
    )
    assert block is None
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and notes[0].delivered_dry_run


async def test_posttooluse_hook_fail_passive_with_stub_steerer() -> None:
    """The 3b surface shares the gate: stub steerer → hook returns {}."""

    class _StubSteerer:
        _signal_channel = "request_context"

    adapter = _adapter(_StubSteerer())
    session = _session_with_note()
    hook = adapter._make_posttooluse_hook(session, current_agent_id="writer")
    out = await hook({}, None, None)
    assert out == {}


# ---------------------------------------------------------------------------
# Finding 2 — factory-attached options: truthful delivery
# ---------------------------------------------------------------------------


async def test_default_path_note_rides_system_prompt(fake_sdk: _FakeSdk) -> None:
    """Baseline: factory leaves options=None → goldfive options attach and
    the note is consumed into the system prompt (a real render)."""
    adapter = _adapter(_steerer(observation_only=False))
    client = _StubClient(options=None)
    adapter._client_factory = lambda: client
    session = _session_with_note()

    result = await adapter.invoke(_task(), session)
    assert result.error is None
    assert client.options is not None
    assert "repeated identical tool call" in client.options.system_prompt
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and not notes[0].delivered_dry_run
    assert notes[0].delivered_surface == "claude_system_prompt"


async def test_operator_options_do_not_consume_note(fake_sdk: _FakeSdk) -> None:
    """Factory-attached options: the note must NOT be consumed at invoke
    time (goldfive's system prompt never reaches the client — consuming
    would record a render that did not happen)."""
    adapter = _adapter(_steerer(observation_only=False))
    operator_options = _FakeOptions(system_prompt="OPERATOR PROMPT")
    client = _StubClient(options=operator_options)
    adapter._client_factory = lambda: client
    session = _session_with_note()

    result = await adapter.invoke(_task(), session)
    assert result.error is None
    # No consumed-but-unrendered note: it is still pending (or was
    # delivered truthfully by the merged hook — which this stub stream
    # never fires).
    pending = ObserverNoteQueue.for_session(session).pending()
    assert [n.drift_id for n in pending] == ["d-claude-truth"]
    # The operator's original options object was not mutated.
    assert operator_options.hooks is None
    assert operator_options.system_prompt == "OPERATOR PROMPT"


async def test_operator_options_get_note_hook_merged_and_deliver_truthfully(
    fake_sdk: _FakeSdk,
) -> None:
    """The PostToolUse note hook is merged onto a COPY of the operator
    options; firing it delivers the pending note and only then marks it."""
    adapter = _adapter(_steerer(observation_only=False))
    sentinel_matcher = _FakeHookMatcher(matcher="OperatorTool", hooks=[object()])
    operator_options = _FakeOptions(
        system_prompt="OPERATOR PROMPT",
        hooks={"PostToolUse": [sentinel_matcher]},
    )
    client = _StubClient(options=operator_options)
    adapter._client_factory = lambda: client
    session = _session_with_note()

    await adapter.invoke(_task(), session)

    # Merged copy: operator's matcher kept, goldfive's appended; the
    # operator's own options object untouched (no accumulation when a
    # factory reuses one shared options instance).
    assert client.options is not operator_options
    merged = client.options.hooks["PostToolUse"]
    assert merged[0] is sentinel_matcher
    assert len(merged) == 2
    assert operator_options.hooks == {"PostToolUse": [sentinel_matcher]}

    # Firing the merged hook renders the note and marks it delivered —
    # telemetry now matches what the agent actually saw.
    note_hook = merged[1].hooks[0]
    out = await note_hook({}, None, None)
    block = out["hookSpecificOutput"]["additionalContext"]
    assert "repeated identical tool call" in block
    notes = ObserverNoteQueue.for_session(session).notes()
    assert notes[0].delivered and not notes[0].delivered_dry_run
    assert notes[0].delivered_surface == "claude_posttooluse"
    assert ObserverNoteQueue.for_session(session).pending() == []


async def test_operator_options_legacy_channel_left_untouched(
    fake_sdk: _FakeSdk,
) -> None:
    """Legacy channel: no note queue, no merge — the operator's options
    object is installed on the client exactly as the factory attached it."""
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    adapter = _adapter(steerer)
    operator_options = _FakeOptions(system_prompt="OPERATOR PROMPT")
    client = _StubClient(options=operator_options)
    adapter._client_factory = lambda: client
    session = Session(run_id="r-claude-legacy")

    result = await adapter.invoke(_task(), session)
    assert result.error is None
    assert client.options is operator_options
    assert operator_options.hooks is None
