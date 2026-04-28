"""Regression tests for per-(adapter, session) pending-tool / in-flight-task isolation.

PR #301 fixed three :class:`ADKAdapter` per-adapter singletons that
cross-wired across concurrent goldfive sessions sharing one adapter
(``_next_cancel_reason`` / ``_session_id`` / ``_outer_session_id``).
That audit also surfaced two more singletons with the same hazard
shape that PR #301 did not address; this module pins them down:

* ``_pending_tool_call_ids`` / ``_pending_tool_call_names`` — track
  outstanding ADK ``function_call`` ids in the current invoke()'s
  event stream so :meth:`_heal_pending_tool_calls` can synthesise
  matching ``function_response`` events on mid-invocation cancel /
  exception. Pre-fix: a single ``set`` / ``dict`` was shared across
  every invocation regardless of session, so session A's heal could
  pick up session B's still-pending ids (and vice versa), corrupting
  both ADK sessions' function-call/response pairing.

* ``_inflight_invoke_task`` — handle to the asyncio task currently
  inside :meth:`_invoke_internal`; :meth:`request_cancel` fires
  ``task.cancel()`` on it from a goldfive-promoted steer
  (goldfive#241). Pre-fix: a single slot meant a second concurrent
  invocation on session B clobbered session A's pin at entry, so a
  steer for A would target B's task.

Each test instantiates one adapter, drives two interleaved goldfive
sessions through it, and asserts neither session sees the other's
state. The tests fail on origin/main and pass with the per-session
dict refactor.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("google.adk")
from goldfive.types import Session, Task

# ---------------------------------------------------------------------------
# Minimal ADK fakes (mirrors helpers in tests/test_adk_adapter_concurrent_sessions.py).
# ---------------------------------------------------------------------------


def _make_agent() -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name="iso_agent",
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
    """Tracks append_event calls per ADK session id."""

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

    async def get_session(self, *, app_name: str, user_id: str, session_id: str) -> _FakeADKSession:
        return self._sessions.setdefault(session_id, _FakeADKSession(sid=session_id))

    async def append_event(self, *, session, event):
        self.appended_by_sid.setdefault(session.id, []).append(event)
        session.events.append(event)
        return event


class _FakeRunner:
    """Runner stub whose ``run_async`` consults a per-session script."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.app_name = "iso-app"
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


def _mk_function_call_event(*, call_id: str, name: str, invocation_id: str = "inv-iso"):
    from google.adk.events.event import Event  # type: ignore
    from google.genai import types  # type: ignore

    part = types.Part(function_call=types.FunctionCall(id=call_id, name=name))
    return Event(
        invocation_id=invocation_id,
        author="iso_agent",
        content=types.Content(role="model", parts=[part]),
    )


def _make_session(*, conversation_id: str, run_id: str) -> Session:
    return Session(run_id=run_id, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Per-session pending-tool-call isolation
# ---------------------------------------------------------------------------


async def test_pending_tool_call_ids_isolated_across_concurrent_sessions() -> None:
    """Concurrent invocations on different sessions don't share a pending-id set.

    Drives two invocations on one adapter — session A emits two
    ``function_call`` ids and is then cancelled mid-stream; session B
    emits one ``function_call`` id of its own and is then cancelled.
    With the fix, A's heal touches only A's two ids and B's heal
    touches only B's single id. On origin/main both invocations share
    one ``set`` so A's heal would also synthesise responses for B's
    pending id (and vice versa) — observable as cross-session
    function_response events on the wrong ADK session.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    session_a = _make_session(conversation_id="conv-A", run_id="adk-iso-A")
    session_b = _make_session(conversation_id="conv-B", run_id="adk-iso-B")

    a_emitted_two = asyncio.Event()
    a_block = asyncio.Event()
    b_emitted_one = asyncio.Event()
    b_block = asyncio.Event()

    async def _script_a():
        yield _mk_function_call_event(call_id="A-call-1", name="search")
        yield _mk_function_call_event(call_id="A-call-2", name="search")
        a_emitted_two.set()
        await a_block.wait()
        yield None  # pragma: no cover

    async def _script_b():
        yield _mk_function_call_event(call_id="B-call-1", name="search")
        b_emitted_one.set()
        await b_block.wait()
        yield None  # pragma: no cover

    runner.script_for("adk-iso-A", _script_a)
    runner.script_for("adk-iso-B", _script_b)

    invoke_a = asyncio.create_task(
        adapter.invoke(Task(id="ta", title="A"), session_a),
        name="invoke-A",
    )
    invoke_b = asyncio.create_task(
        adapter.invoke(Task(id="tb", title="B"), session_b),
        name="invoke-B",
    )

    # Wait until BOTH streams have populated their pending-id buckets.
    await asyncio.wait_for(a_emitted_two.wait(), timeout=2.0)
    await asyncio.wait_for(b_emitted_one.wait(), timeout=2.0)

    # Cancel A first; the heal path on origin/main reads the SHARED
    # ``_pending_tool_call_ids`` set so it synthesises function_responses
    # for ALL three ids (both A's and B's) — a wrong-session bleed.
    # With the fix, A's heal touches only A's two ids.
    invoke_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_a

    appended_a = runner.session_service.appended_by_sid.get("adk-iso-A", [])
    healed_a_ids: set[str] = set()
    for ev in appended_a:
        for fr in ev.get_function_responses():
            if fr.id:
                healed_a_ids.add(fr.id)
    assert healed_a_ids == {"A-call-1", "A-call-2"}, (
        "session A's heal must synthesise responses for ONLY A's "
        f"pending ids; got {healed_a_ids!r} — cross-session leak "
        "from B's still-in-flight pending id"
    )

    # Session B's stream is still in-flight — A's cancel must NOT
    # have appended any synthetic events to B's ADK session.
    appended_b_after_a = runner.session_service.appended_by_sid.get("adk-iso-B", [])
    assert appended_b_after_a == [], (
        "session A's heal appended events to session B's ADK session — "
        "cross-session function_response leak"
    )

    # Now cancel B and verify B's heal still finds its own pending id
    # (on origin/main A's clear nuked the shared set so B has nothing
    # to heal — wrong-session clear).
    invoke_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_b
    appended_b = runner.session_service.appended_by_sid.get("adk-iso-B", [])
    healed_b_ids: set[str] = set()
    for ev in appended_b:
        for fr in ev.get_function_responses():
            if fr.id:
                healed_b_ids.add(fr.id)
    assert healed_b_ids == {"B-call-1"}, (
        "session B's heal must synthesise responses for B's pending "
        f"id; got {healed_b_ids!r} — A's heal cleared the shared "
        "bucket on cancel so B's id was lost (cross-session clear)"
    )


async def test_pending_tool_call_names_isolated_across_concurrent_sessions() -> None:
    """The companion call_id -> tool_name map is also per-session.

    Same hazard, paired field. The synthetic function_response carries
    ``name=tool_name`` looked up from this dict; if A's tool_name map
    were shared with B, B's heal could mislabel A's outstanding id (or
    vice versa). Direct unit check that the per-session helpers return
    distinct dicts and writes to one don't show up in the other.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    a_names = adapter._pending_names_for("sid-A")
    b_names = adapter._pending_names_for("sid-B")
    assert a_names is not b_names

    a_names["call-A1"] = "search"
    b_names["call-B1"] = "fetch"

    assert adapter._pending_names_for("sid-A") == {"call-A1": "search"}
    assert adapter._pending_names_for("sid-B") == {"call-B1": "fetch"}


def test_legacy_pending_tool_call_attributes_back_compat() -> None:
    """Bare-attribute writes / reads still work for single-session callers.

    Single-session unit tests (notably tests/test_orchestration_state.py
    and tests/test_cancel_propagation.py) drive
    ``adapter._pending_tool_call_ids = {...}`` and
    ``adapter._pending_tool_call_names = {...}`` directly. The
    property shims must keep their bare-attribute semantics: writes
    land on the empty-key bucket and reads return the same set/dict
    so subsequent mutations are observable.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    adapter._pending_tool_call_ids = {"call-a", "call-b"}
    adapter._pending_tool_call_names = {"call-a": "tool1", "call-b": "tool2"}

    assert adapter._pending_tool_call_ids == {"call-a", "call-b"}
    assert adapter._pending_tool_call_names == {
        "call-a": "tool1",
        "call-b": "tool2",
    }
    # Empty-key bucket is what the shim uses; per-session helpers
    # confirm.
    assert adapter._pending_ids_for("") == {"call-a", "call-b"}
    assert adapter._pending_names_for("") == {
        "call-a": "tool1",
        "call-b": "tool2",
    }
    # A different session's bucket must be unaffected.
    assert adapter._pending_ids_for("other-session") == set()
    assert adapter._pending_names_for("other-session") == {}


# ---------------------------------------------------------------------------
# Per-session in-flight-invoke-task isolation
# ---------------------------------------------------------------------------


async def test_inflight_invoke_task_isolated_across_concurrent_sessions() -> None:
    """Two concurrent invocations pin their own asyncio.Task per session.

    Pre-fix: ``self._inflight_invoke_task = asyncio.current_task()``
    inside :meth:`_invoke_internal` runs on entry of EACH invocation.
    The second invocation overwrites the first's pin, so
    :meth:`request_cancel` would target the second invocation's task
    even when the steerer asked to cancel the first.

    With the fix the pin is keyed by ``Session.id`` and
    :meth:`request_cancel(reason, session=...)` cancels only that
    session's task. We assert directly via the per-session lookup
    helpers (no ``request_cancel`` plumbing needed).
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    session_a = _make_session(conversation_id="conv-A", run_id="adk-iso-A")
    session_b = _make_session(conversation_id="conv-B", run_id="adk-iso-B")

    a_started = asyncio.Event()
    a_block = asyncio.Event()
    b_started = asyncio.Event()
    b_block = asyncio.Event()

    async def _script_a():
        a_started.set()
        await a_block.wait()
        yield None  # pragma: no cover

    async def _script_b():
        b_started.set()
        await b_block.wait()
        yield None  # pragma: no cover

    runner.script_for("adk-iso-A", _script_a)
    runner.script_for("adk-iso-B", _script_b)

    invoke_a = asyncio.create_task(
        adapter.invoke(Task(id="ta", title="A"), session_a),
        name="invoke-A",
    )
    invoke_b = asyncio.create_task(
        adapter.invoke(Task(id="tb", title="B"), session_b),
        name="invoke-B",
    )

    await asyncio.wait_for(a_started.wait(), timeout=2.0)
    await asyncio.wait_for(b_started.wait(), timeout=2.0)

    # request_cancel(session=A) cancels ONLY A's task. B's task is
    # still alive after the await.
    #
    # On origin/main ``_inflight_invoke_task`` is a single slot — B's
    # invocation entry overwrote A's pin, so request_cancel either
    # cancels B (wrong-session attribution) or no-ops if A's slot was
    # already cleared. Either way A doesn't observe its own
    # CancelledError on the right side of the assertion.
    await adapter.request_cancel("goldfive_off_topic", session=session_a)
    with pytest.raises(asyncio.CancelledError):
        await invoke_a
    assert not invoke_b.done(), (
        "request_cancel(session=A) collaterally cancelled session B's "
        "in-flight task — wrong-session attribution leaked through "
        "the shared singleton slot"
    )

    # Cleanup: cancel B too.
    invoke_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_b


async def test_legacy_inflight_invoke_task_back_compat() -> None:
    """Bare-attribute write / read for ``_inflight_invoke_task`` still work.

    tests/test_goldfive_steer_request_cancel.py and friends drive
    ``adapter._inflight_invoke_task = task`` directly then call
    ``adapter.request_cancel(...)``. The property shim must store
    on the empty-key bucket so the request_cancel cancel-all path
    still finds the task.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    async def _noop() -> None:
        return

    sentinel = asyncio.create_task(_noop())
    try:
        adapter._inflight_invoke_task = sentinel
        assert adapter._inflight_invoke_task is sentinel
        assert adapter._get_inflight_invoke_task("") is sentinel

        adapter._inflight_invoke_task = None
        assert adapter._inflight_invoke_task is None
        assert adapter._get_inflight_invoke_task("") is None
    finally:
        await sentinel


async def test_request_cancel_no_session_cancels_all_inflight() -> None:
    """``request_cancel(reason)`` (no session=) cancels every in-flight task.

    Single-session callers and the existing
    :class:`DefaultSteerer._request_adapter_cancel` shape pass only
    ``reason``. We must not break them — when no session is supplied
    we cancel every currently-pinned task. In single-session use
    there's only one entry, so the observable behavior matches the
    pre-fix path.
    """
    from goldfive.adapters.adk import ADKAdapter

    agent = _make_agent()
    runner = _FakeRunner(agent)
    adapter = ADKAdapter(runner)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _body() -> None:
        started.set()
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(_body())
    await started.wait()
    adapter._inflight_invoke_task = task

    await adapter.request_cancel("goldfive_off_topic")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
