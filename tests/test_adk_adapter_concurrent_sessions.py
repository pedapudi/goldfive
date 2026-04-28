"""Regression tests for per-(adapter, session) state isolation.

PR #294's audit flagged three :class:`ADKAdapter` instance attributes
as concurrent-runs hazards because one adapter is shared across every
goldfive session driven by a :class:`Runner`:

* ``_next_cancel_reason`` — the symbolic tag the steerer / executor
  stamps before triggering a mid-invocation cancel so the synthetic
  ``function_response`` carries LLM-actionable content. Cross-session
  leak: a USER_STEER tag stamped for session A's invocation could be
  consumed by session B's cancel emission, mislabelling B's cancel.
* ``_session_id`` — the cached ADK session id chosen by
  :meth:`ADKAdapter._ensure_session`. Cross-session leak: the first
  invocation's mint sticks across every concurrent invocation, so two
  goldfive Conversations target the same ADK session history (history
  bleed; corrupts tool-id pairing).
* ``_outer_session_id`` — the outer adk-web session pinned by
  :meth:`GoldfiveADKAgent._pin_outer_session_on_adapter`. Forensic
  field, but the single-attribute store hid every pin after the first.

These tests exercise the per-session helpers
(:meth:`ADKAdapter.set_next_cancel_reason`,
:meth:`ADKAdapter._consume_next_cancel_reason`,
:meth:`ADKAdapter._ensure_session`, and the
``_pinned_outer_session_ids`` set) plus an end-to-end mid-invocation
cancel path. They fail on origin/main and pass with the per-session
dict refactor.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.types import Session, Task


# ---------------------------------------------------------------------------
# Minimal ADK fakes (mirrors helpers in tests/test_adk_adapter.py).
# ---------------------------------------------------------------------------


def _make_agent() -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name="concurrent_agent",
        model="fake-model",
        description="Test agent",
        instruction="Test.",
    )


class _FakeADKSession:
    def __init__(self, sid: str = "") -> None:
        self.id = sid
        self.events: list = []
        self.state: dict = {}


class _FakeSessionService:
    """Tracks append_event calls per ADK session id.

    ``create_session`` is keyed by ``session_id`` so two distinct
    goldfive Conversations get distinct ADK sessions, making
    cross-session leak observable.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _FakeADKSession] = {}
        self.appended_by_sid: dict[str, list] = {}

    async def create_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> _FakeADKSession:
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = _FakeADKSession(sid=session_id)
            self._sessions[session_id] = sess
            self.appended_by_sid.setdefault(session_id, [])
        return sess

    async def get_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> _FakeADKSession:
        return self._sessions.setdefault(
            session_id, _FakeADKSession(sid=session_id)
        )

    async def append_event(self, *, session, event):
        self.appended_by_sid.setdefault(session.id, []).append(event)
        session.events.append(event)
        return event


class _FakeRunner:
    """Runner stub whose ``run_async`` consults a per-session script."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.app_name = "concurrent-app"
        self.session_service = _FakeSessionService()
        self.plugin_manager = None
        self.plugins: list = []
        self._scripts: dict[str, Any] = {}

    def script_for(self, session_id: str, factory) -> None:
        self._scripts[session_id] = factory

    async def run_async(self, *, user_id: str, session_id: str, new_message=None):
        factory = self._scripts.get(session_id)
        if factory is None:

            async def _empty():
                if False:
                    yield None

            factory = _empty
        async for event in factory():
            yield event


def _mk_function_call_event(
    *, call_id: str, name: str, invocation_id: str = "inv-c"
):
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part(function_call=types.FunctionCall(id=call_id, name=name))
    return Event(
        invocation_id=invocation_id,
        author="concurrent_agent",
        content=types.Content(role="model", parts=[part]),
    )


def _make_session(*, conversation_id: str, run_id: str) -> Session:
    """Build a goldfive Session with explicit conversation + run ids.

    Mirrors what :meth:`Runner._run_locked` produces for outer-pinned
    invocations: ``conversation_id`` from a Conversation lookup keyed
    on the outer adk-web session id, ``run_id`` overridden to that
    same outer id (so :attr:`Session.id` returns the outer id).
    """
    return Session(run_id=run_id, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Regression tests — per-session state isolation
# ---------------------------------------------------------------------------


def test_set_next_cancel_reason_does_not_leak_across_sessions() -> None:
    """A reason set for session A is invisible to session B's consumer.

    Direct unit-level check on
    :meth:`ADKAdapter.set_next_cancel_reason` /
    :meth:`ADKAdapter._consume_next_cancel_reason`. On origin/main both
    sessions read / write the same ``_next_cancel_reason`` string
    attribute, so session B's consume returns ``"user_steer"`` even
    though no tag was ever set for B.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    session_a = _make_session(conversation_id="conv-A", run_id="adk-outer-A")
    session_b = _make_session(conversation_id="conv-B", run_id="adk-outer-B")

    adapter.set_next_cancel_reason(session_a, "user_steer")

    # Session B's consume must NOT pick up A's tag.
    b_reason = adapter._consume_next_cancel_reason(session_b)
    assert b_reason == "", (
        "session B leaked session A's user_steer tag through the "
        f"shared cancel-reason slot; got {b_reason!r}"
    )

    # Session A's consume DOES return its tag, then clears it.
    a_reason = adapter._consume_next_cancel_reason(session_a)
    assert a_reason == "user_steer"
    a_reason_again = adapter._consume_next_cancel_reason(session_a)
    assert a_reason_again == ""


def test_legacy_bare_attribute_write_falls_through_to_consume() -> None:
    """Single-session callers / tests using ``adapter._next_cancel_reason = X``
    still see the tag consumed on the next cancel.

    Back-compat: legacy bare writes go into a fallback slot the
    consumer reads only when the per-session entry for the active
    session id is empty. Without this, every test in the codebase
    that drives ``adapter._next_cancel_reason = ...`` directly would
    break.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    # Single-session legacy use: bare-attribute write, then consume.
    session = _make_session(conversation_id="conv-solo", run_id="adk-outer-solo")
    adapter._next_cancel_reason = "user_steer"
    assert adapter._consume_next_cancel_reason(session) == "user_steer"
    # Cleared after consume.
    assert adapter._next_cancel_reason == ""


async def test_ensure_session_caches_per_conversation() -> None:
    """Two concurrent goldfive Conversations get distinct ADK session ids.

    On origin/main ``_ensure_session`` mints once and caches on the
    shared ``_session_id`` attribute, so the second concurrent
    Conversation's invocation observes the FIRST Conversation's id and
    targets the wrong ADK session history. With the per-conversation
    cache, each Conversation gets its own id keyed by
    ``Session.conversation_id``.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    session_a = _make_session(conversation_id="conv-A", run_id="adk-outer-A")
    session_b = _make_session(conversation_id="conv-B", run_id="adk-outer-B")

    sid_a = await adapter._ensure_session(session_a)
    sid_b = await adapter._ensure_session(session_b)

    assert sid_a != sid_b, (
        "concurrent goldfive Conversations must resolve to distinct "
        f"ADK session ids; got {sid_a!r} == {sid_b!r}"
    )
    # The chosen ids inherit ``session.id`` (Runner.run(session_id=)
    # outer pin) for adk-web flows, giving harmonograf one id across
    # plan + spans without the older shared-slot pin-on-adapter dance.
    assert sid_a == "adk-outer-A"
    assert sid_b == "adk-outer-B"

    # Re-resolution within the same Conversation reuses the cached id
    # (multi-turn ADK history continuity is preserved).
    sid_a_again = await adapter._ensure_session(session_a)
    sid_b_again = await adapter._ensure_session(session_b)
    assert sid_a_again == sid_a
    assert sid_b_again == sid_b


async def test_ensure_session_legacy_pin_only_applies_to_empty_conv_bucket() -> None:
    """A constructor / outer-pin ``_session_id`` never leaks into other Conversations.

    Tests like ``tests/test_adk_adapter_overlay.py`` set
    ``adapter._session_id = "stub-session"`` for single-session use.
    The legacy pin must only seed the empty-conversation-id bucket so
    a subsequent concurrent Conversation does not adopt the pinned id
    by accident.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)
    adapter._session_id = "stub-session"

    # Empty conversation_id (legacy single-session caller) → adopts the pin.
    legacy_session = Session(run_id="r1")
    sid_legacy = await adapter._ensure_session(legacy_session)
    assert sid_legacy == "stub-session"

    # Non-empty conversation_id (concurrent Conversation) → ignores
    # the legacy pin, derives from session.id.
    other_session = _make_session(
        conversation_id="conv-other", run_id="adk-outer-other"
    )
    sid_other = await adapter._ensure_session(other_session)
    assert sid_other == "adk-outer-other", (
        "legacy ``_session_id`` pin leaked into a concurrent "
        f"Conversation; got {sid_other!r}"
    )


def test_outer_session_id_pin_records_every_outer_session() -> None:
    """Pinning the same adapter from two outer sessions retains both ids.

    On origin/main ``_outer_session_id`` is a single string, so the
    second pin overwrites (or, with the existing
    ``not adapter._outer_session_id`` guard in
    :meth:`GoldfiveADKAgent._pin_outer_session_on_adapter`, silently
    drops) the second outer session id — forensic logging sees only
    the first pinned outer id forever after.

    With the fix, every distinct pinned outer sid is recorded on
    ``_pinned_outer_session_ids`` (a set), so log / forensic
    consumers can see the full set of outer adk-web sessions the
    adapter has served.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    adapter._outer_session_id = "adk-outer-A"
    adapter._outer_session_id = "adk-outer-B"

    pinned = adapter._pinned_outer_session_ids
    assert "adk-outer-A" in pinned
    assert "adk-outer-B" in pinned


async def test_invoke_cancel_uses_session_keyed_reason_not_legacy_slot() -> None:
    """End-to-end check: an in-flight invoke's cancel consumes the
    session-keyed reason, ignoring a stale value left in the legacy
    slot from a different session's flow.

    Scenario:

    1. Session B sets a USER_STEER reason via the bare-attribute
       legacy path (simulating a stale tag left behind by a
       pre-#294 caller or a test that doesn't use the helper).
    2. Session A starts an invoke, then we set A's reason via the
       per-session helper.
    3. A's task is cancelled.

    With the fix, A's cancel reads its session-keyed entry first
    and the synthetic function_response carries the user_steer
    variant — the stale legacy slot is NOT consulted. On origin/main
    both writes targeted the same shared attribute so whichever
    write landed last won, hiding the cross-session hazard behind a
    last-writer-wins flake.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    session_a = _make_session(conversation_id="conv-A", run_id="adk-outer-A")
    session_b = _make_session(conversation_id="conv-B", run_id="adk-outer-B")

    a_started = asyncio.Event()
    a_block = asyncio.Event()

    async def _script_a():
        yield _mk_function_call_event(call_id="call-a", name="search")
        a_started.set()
        await a_block.wait()
        yield None  # pragma: no cover

    runner.script_for("adk-outer-A", _script_a)

    # Stash a value via the legacy bare-attribute path "for session B".
    # In the per-session-keyed design this lands in the shared fallback
    # slot only — it never gets routed to session A's heal path.
    adapter._next_cancel_reason = "stale_from_session_b"

    invoke_a = asyncio.create_task(
        adapter.invoke(Task(id="ta", title="A"), session_a),
        name="invoke-A",
    )
    await asyncio.wait_for(a_started.wait(), timeout=2.0)

    # Now set the proper per-session reason for A via the helper.
    adapter.set_next_cancel_reason(session_a, "user_steer")

    invoke_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_a

    appended_a = runner.session_service.appended_by_sid.get("adk-outer-A", [])
    assert appended_a, "session A must have a healed function_response"
    statuses: list[str] = []
    for ev in appended_a:
        for fr in ev.get_function_responses():
            payload = fr.response or {}
            if "status" in payload:
                statuses.append(str(payload.get("status", "")))
    assert "cancelled_by_user_steering" in statuses, (
        "session A's per-session cancel reason did NOT win over the "
        f"legacy slot; got statuses={statuses!r}"
    )

    # Session B never invoked, so it has no healed events. The legacy
    # slot's stale ``stale_from_session_b`` value did not bleed into
    # any session-keyed lookup.
    assert "adk-outer-B" not in runner.session_service.appended_by_sid
    # Direct per-session consume from B confirms isolation.
    b_reason = adapter._consume_next_cancel_reason(session_b)
    # On origin/main this would be the user_steer tag set for A
    # (single shared attribute, last write wins). With the fix it's
    # only the legacy fallback that survived (cleared after A's
    # invocation consumed it).
    assert b_reason != "user_steer", (
        "session B leaked session A's user_steer tag: "
        f"got {b_reason!r}"
    )
