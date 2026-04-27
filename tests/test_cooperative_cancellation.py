"""Cooperative-cancellation tests (goldfive#251 Stream C / 7a).

Ships the adapter callback + steerer wiring for cancellable
in-flight invocations. Each test pins one invariant from the
design brief:

1. Cancel flag set -> ``before_agent_callback`` skips turn.
2. Flag is CONSUMED (read + clear) — re-entry doesn't re-cancel.
3. ``before_tool_callback`` returns ``{"status": "cancelled"}`` for
   a cancelled invocation.
4. Cancellation emits an ``InvocationCancelled`` sink event with the
   rich context (operator-visible; not LLM-visible).
5. Parent-child propagation: cancelling an invocation also flags its
   spawned sub-invocations.
6. Severity ladder — INFO drift does NOT cancel.
7. Severity ladder — CRITICAL drift DOES cancel the active
   invocation.
8. Empty invocation-id guard — a drift with no resolvable active
   invocation doesn't crash and doesn't fabricate a target.
9. LLM-visible response is MINIMAL — no ``reason`` / ``detail`` /
   ``drift_kind`` leaks. Invariant.
10. No auto re-dispatch — after cancel fires, the framework does NOT
    re-call the agent. The parent's next turn handles the cancelled
    marker itself.

The ADK plugin tests use a minimal stub that mimics the attribute
shape of the live ADK ``callback_context`` / ``tool_context`` /
``invocation_context`` types without pulling in ADK as a test
dependency — keeps the file usable in environments that don't have
the ``adk`` extra installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")

from goldfive.adapters import _adk_state_protocol as _sp  # noqa: E402
from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    make_adk_plugin,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    CancellationRequest,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


def _cancelled_events(sink: ListSink) -> list[Any]:
    """Filter proto envelope events by ``WhichOneof('payload') ==
    'invocation_cancelled'`` (goldfive A5: dict envelope promoted to
    proto)."""
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == "invocation_cancelled":
                out.append(evt)
        except Exception:
            continue
    return out


class _FakeInvocationContext:
    """Mimics ADK ``InvocationContext`` enough for the plugin's
    callback paths: exposes ``invocation_id``, ``session`` (with
    ``state`` + ``run_id`` + ``id``), and ``agent`` (with ``name``).
    """

    def __init__(
        self,
        *,
        invocation_id: str,
        session_state: dict[str, Any],
        agent_name: str = "sub_agent",
    ) -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession(state=session_state)
        self.agent = _FakeAgent(name=agent_name)


class _FakeADKSession:
    def __init__(self, *, state: dict[str, Any]) -> None:
        self.state = state
        self.run_id = "run-test"
        self.id = "session-test"


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeCallbackContext:
    """Mimics ADK ``CallbackContext`` — holds ``_invocation_context``."""

    def __init__(self, *, invocation_context: _FakeInvocationContext) -> None:
        self._invocation_context = invocation_context
        # A minimal ``state`` attribute that points at the same mapping
        # ADK state-protocol callbacks read. The real ADK wrapper is
        # richer (``state.to_dict()`` / ``state.delta``), but the plugin
        # code paths we exercise only dereference a Mapping.
        self.state = invocation_context.session.state


class _FakeToolContext(_FakeCallbackContext):
    """``ToolContext`` extends ``CallbackContext`` with a function_call_id."""

    def __init__(
        self,
        *,
        invocation_context: _FakeInvocationContext,
        function_call_id: str = "fc-1",
    ) -> None:
        super().__init__(invocation_context=invocation_context)
        self.function_call_id = function_call_id


class _FakeTool:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _StubAdapter:
    """Adapter stub exposing just the plugin slot the steerer needs."""

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin
        self._next_cancel_reason: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plugin(*, with_sinks: bool = True) -> tuple[Any, ListSink, Session]:
    """Build a fresh plugin instance + an attached SessionContext so
    the plugin's ``_emit_invocation_cancelled`` path can fan out."""
    plugin = make_adk_plugin(host_agent_name="coordinator")
    sink = ListSink()
    steerer = DefaultSteerer()
    if with_sinks:
        steerer.bind(sinks=[sink], planner=None)
    session = Session(
        run_id="run-test",
        goals=[Goal(id="g1", summary="ship")],
        plan=_make_plan(),
        current_task_id="t1",
    )
    ctx = SessionContext(
        session=session,
        steerer=steerer,
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    return plugin, sink, session


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="run-test",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_request(
    *,
    invocation_id: str = "inv-1",
    severity: DriftSeverity = DriftSeverity.CRITICAL,
    reason: str = "drift",
    drift_kind: str = "off_topic",
) -> CancellationRequest:
    return CancellationRequest(
        invocation_id=invocation_id,
        reason=reason,
        severity=severity,
        drift_kind=drift_kind,
        detail="cancel test",
    )


# ---------------------------------------------------------------------------
# 1. Flag set -> before_agent_callback skips turn
# ---------------------------------------------------------------------------


async def test_before_agent_skips_turn_when_cancelled() -> None:
    plugin, sink, _session = _make_plugin()
    request = _make_request(invocation_id="inv-1")
    plugin.request_invocation_cancel(invocation_id="inv-1", request=request)
    assert plugin.peek_cancel_for_invocation("inv-1") is request

    # Attach a reconciler spy so we can prove the pinning / reconciler
    # forward work did NOT run — the callback short-circuited.
    class _Spy:
        def __init__(self) -> None:
            self.before_agent_calls = 0

        async def on_before_agent(self, **_kwargs: Any) -> None:
            self.before_agent_calls += 1

    spy = _Spy()
    plugin.set_reconciler(spy)

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
        agent_name="sub_agent",
    )
    await plugin.before_agent_callback(
        agent=_FakeAgent(name="sub_agent"),
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
    )

    # Reconciler forward did NOT fire — agent turn was skipped.
    assert spy.before_agent_calls == 0
    # Sink received an InvocationCancelled event.
    assert len(_cancelled_events(sink)) == 1


# ---------------------------------------------------------------------------
# 2. Flag is CONSUMED (read + clear) -- re-entry doesn't re-emit, but
#    the cancellation is sticky for the rest of the invocation
#    (goldfive#271 follow-up; see /tmp/demo-v12.log regression).
# ---------------------------------------------------------------------------


async def test_cancel_flag_consumed_on_first_callback() -> None:
    plugin, sink, _session = _make_plugin()
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request())

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )

    # Spy reconciler so we can assert no forward work runs on either
    # the first or second callback (both must short-circuit).
    class _Spy:
        def __init__(self) -> None:
            self.calls = 0

        async def on_before_agent(self, **_kwargs: Any) -> None:
            self.calls += 1

    spy = _Spy()
    plugin.set_reconciler(spy)

    # First callback consumes the cancel-state entry, emits
    # InvocationCancelled, marks the invocation sticky-cancelled, and
    # short-circuits.
    await plugin.before_agent_callback(
        agent=_FakeAgent(name="sub_agent"),
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
    )
    # The popped flag is gone from cancel_state...
    assert plugin.peek_cancel_for_invocation("inv-1") is None
    # ...but the sticky bit is set so subsequent callbacks see the
    # invocation as still cancelled.
    assert plugin.is_invocation_cancelled("inv-1") is True
    # A second callback on the same invocation must ALSO short-circuit
    # (without re-emitting the sink event). Pre-fix: the reconciler's
    # ``on_before_agent`` ran and the agent turn proceeded.
    await plugin.before_agent_callback(
        agent=_FakeAgent(name="sub_agent"),
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
    )
    assert spy.calls == 0  # neither callback ran the reconciler
    # Only ONE InvocationCancelled event in total — the second
    # callback short-circuited silently (it knows the cancel was
    # already announced).
    assert len(_cancelled_events(sink)) == 1


# ---------------------------------------------------------------------------
# 3. before_tool_callback returns {"status": "cancelled"} when cancelled
# ---------------------------------------------------------------------------


async def test_before_tool_returns_cancelled_status() -> None:
    plugin, _sink, _session = _make_plugin()
    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(drift_kind="off_topic", reason="drift"),
    )

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )
    tool_ctx = _FakeToolContext(invocation_context=inv_ctx)
    response = await plugin.before_tool_callback(
        tool=_FakeTool(name="search"),
        tool_args={"query": "test"},
        tool_context=tool_ctx,
    )
    assert response == {"status": "cancelled"}


# ---------------------------------------------------------------------------
# 4. InvocationCancelled sink event carries rich context
# ---------------------------------------------------------------------------


async def test_invocation_cancelled_sink_event_rich_fields() -> None:
    plugin, sink, _session = _make_plugin()
    request = _make_request(
        invocation_id="inv-1",
        reason="user_steer",
        severity=DriftSeverity.CRITICAL,
        drift_kind="off_topic",
    )
    request.drift_id = "drift-abc"
    plugin.request_invocation_cancel(invocation_id="inv-1", request=request)
    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
        agent_name="planner_agent",
    )
    await plugin.before_agent_callback(
        agent=_FakeAgent(name="planner_agent"),
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
    )
    cancelled = _cancelled_events(sink)
    assert len(cancelled) == 1
    payload = cancelled[0].invocation_cancelled
    assert payload.invocation_id == "inv-1"
    assert payload.agent_name == "planner_agent"
    assert payload.reason == "user_steer"
    assert payload.severity == "critical"
    assert payload.drift_id == "drift-abc"
    assert payload.drift_kind == "off_topic"


# ---------------------------------------------------------------------------
# 5. Parent-child propagation
# ---------------------------------------------------------------------------


def test_parent_cancel_propagates_to_children() -> None:
    plugin, _sink, _session = _make_plugin()
    # Record a small tree: A -> B, B -> C
    plugin._invocation_parents["inv-B"] = "inv-A"
    plugin._invocation_parents["inv-C"] = "inv-B"
    # Independent sibling that must NOT be flagged.
    plugin._invocation_parents["inv-D"] = "inv-X"

    flagged = plugin.request_invocation_cancel(
        invocation_id="inv-A",
        request=_make_request(invocation_id="inv-A"),
    )
    assert set(flagged) == {"inv-A", "inv-B", "inv-C"}
    # The state dict really carries entries for every flagged id.
    assert plugin.peek_cancel_for_invocation("inv-A") is not None
    assert plugin.peek_cancel_for_invocation("inv-B") is not None
    assert plugin.peek_cancel_for_invocation("inv-C") is not None
    # Unrelated sibling is clear.
    assert plugin.peek_cancel_for_invocation("inv-D") is None


# ---------------------------------------------------------------------------
# 6. Severity ladder -- INFO drift does NOT cancel
# ---------------------------------------------------------------------------


async def test_info_drift_does_not_request_cancel() -> None:
    plugin, _sink, session = _make_plugin()
    adapter = _StubAdapter(plugin)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)
    steerer.bind_adapter(adapter)
    # Tell the plugin what the top-level invocation_id is so
    # ``_resolve_active_invocation_ids`` has something to target
    # (without this the empty-invocation-id guard would short-circuit
    # cancel independently and we couldn't distinguish the severity
    # gate from the empty-guard).
    plugin._top_invocation_id = "inv-A"

    drift = DriftEvent(
        kind=DriftKind.CONFUSION,
        severity=DriftSeverity.INFO,
        current_task_id="t1",
        current_agent_id="sub_agent",
    )
    await steerer._handle_drift(drift, session)
    # No cancel was flagged for any invocation.
    assert plugin.peek_cancel_for_invocation("inv-A") is None


# ---------------------------------------------------------------------------
# 7. Severity ladder -- CRITICAL drift DOES cancel
# ---------------------------------------------------------------------------


async def test_critical_drift_requests_cancel() -> None:
    plugin, _sink, session = _make_plugin()
    adapter = _StubAdapter(plugin)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)
    steerer.bind_adapter(adapter)
    plugin._top_invocation_id = "inv-A"

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        current_task_id="t1",
        current_agent_id="sub_agent",
    )
    await steerer._handle_drift(drift, session)
    # Cancel was flagged for the resolved invocation id.
    req = plugin.peek_cancel_for_invocation("inv-A")
    assert req is not None
    assert req.severity is DriftSeverity.CRITICAL
    assert req.drift_kind == DriftKind.OFF_TOPIC.value


# ---------------------------------------------------------------------------
# 8. Empty invocation-id guard
# ---------------------------------------------------------------------------


async def test_cancel_noop_when_no_active_invocation() -> None:
    plugin, _sink, session = _make_plugin()
    adapter = _StubAdapter(plugin)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)
    steerer.bind_adapter(adapter)
    # DELIBERATELY do NOT pin a top invocation id.
    plugin._top_invocation_id = ""
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        # No current_agent_id / current_task_id either.
    )
    # Does not raise.
    flagged = await steerer.request_invocation_cancel(drift=drift, session=session)
    assert flagged == []


# ---------------------------------------------------------------------------
# 9. LLM-visible response is MINIMAL -- invariant
# ---------------------------------------------------------------------------


async def test_llm_visible_cancelled_response_is_minimal() -> None:
    """The tool response returned to the LLM must be ``{"status":
    "cancelled"}`` — bare shape. No ``reason`` / ``detail`` /
    ``drift_kind`` leak that the parent LLM could pattern-match on
    and invent workarounds (lesson from goldfive#250 / #252 / #253).
    """
    plugin, _sink, _session = _make_plugin()
    request = _make_request(
        reason="user_steer",
        drift_kind="off_topic",
    )
    request.detail = "the agent wandered into raccoons"
    plugin.request_invocation_cancel(invocation_id="inv-1", request=request)

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )
    response = await plugin.before_tool_callback(
        tool=_FakeTool(name="search"),
        tool_args={},
        tool_context=_FakeToolContext(invocation_context=inv_ctx),
    )
    # Bare shape. No reason, no detail, no drift_kind — the LLM can
    # only pattern-match on the single word "cancelled".
    assert response == {"status": "cancelled"}
    assert set(response.keys()) == {"status"}
    # Specifically forbidden keys (prompt-injection risk):
    for forbidden in ("reason", "detail", "drift_kind", "severity", "drift_id"):
        assert forbidden not in response


# ---------------------------------------------------------------------------
# 10. No auto re-dispatch -- cancel stops the invocation; parent decides next
# ---------------------------------------------------------------------------


async def test_no_auto_redispatch_after_cancel() -> None:
    """After a cancel fires, the framework does NOT re-call the
    agent. The plugin's job is to skip the in-flight invocation
    cleanly; re-dispatch (if any) is the parent's decision, driven
    by its own next-turn prompt (Stream B plan-causal prompting).
    """
    plugin, sink, _session = _make_plugin()

    class _ReconcilerSpy:
        def __init__(self) -> None:
            self.before_agent_calls = 0
            self.delegation_calls = 0

        async def on_before_agent(self, **_kwargs: Any) -> None:
            self.before_agent_calls += 1

        async def on_delegation_observed(self, **_kwargs: Any) -> None:
            self.delegation_calls += 1

        async def on_after_agent(self, **_kwargs: Any) -> None:
            pass

    spy = _ReconcilerSpy()
    plugin.set_reconciler(spy)

    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request())
    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )
    # One "attempt" at before_agent — gets cancelled.
    await plugin.before_agent_callback(
        agent=_FakeAgent(name="sub_agent"),
        callback_context=_FakeCallbackContext(invocation_context=inv_ctx),
    )
    # The framework did NOT make a second call on our behalf.
    # (If it had, ``before_agent_calls`` would be > 0 because the
    # second call would have found the flag consumed and proceeded
    # through the reconciler path.)
    assert spy.before_agent_calls == 0
    # A single cancelled event landed; no follow-up
    # "re-dispatched" marker of any kind.
    assert len(_cancelled_events(sink)) == 1


# ---------------------------------------------------------------------------
# Extra: user-initiated cancel bypasses the severity gate
# ---------------------------------------------------------------------------


async def test_user_steer_drift_bypasses_severity_gate() -> None:
    plugin, _sink, session = _make_plugin()
    adapter = _StubAdapter(plugin)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)
    steerer.bind_adapter(adapter)
    plugin._top_invocation_id = "inv-A"

    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,  # below CRITICAL
        detail="please pivot",
    )
    # The user-authored drift goes through _handle_drift's promotion
    # path before cancel is considered; ensure the directly-reachable
    # cancel API honours the bypass.
    flagged = await steerer.request_invocation_cancel(drift=drift, session=session)
    assert "inv-A" in flagged


# ---------------------------------------------------------------------------
# Extra: state-protocol helpers round-trip
# ---------------------------------------------------------------------------


def test_state_protocol_cancel_helpers_roundtrip() -> None:
    state: dict[str, Any] = {}
    req = _make_request(invocation_id="inv-1")
    _sp.write_cancel_request(state, invocation_id="inv-1", request=req)
    assert _sp.read_cancel_request(state, "inv-1") is req
    # Consume clears the entry.
    assert _sp.consume_cancel_request(state, "inv-1") is req
    assert _sp.read_cancel_request(state, "inv-1") is None
    assert _sp.KEY_CANCEL_REQUESTED not in state


def test_state_protocol_descendants_walk() -> None:
    state: dict[str, Any] = {}
    _sp.register_invocation_parent(state, invocation_id="B", parent_invocation_id="A")
    _sp.register_invocation_parent(state, invocation_id="C", parent_invocation_id="B")
    _sp.register_invocation_parent(state, invocation_id="D", parent_invocation_id="A")
    _sp.register_invocation_parent(state, invocation_id="X", parent_invocation_id="Y")
    descendants = _sp.descendants_of_invocation(state, "A")
    assert set(descendants) == {"B", "C", "D"}
    # Unrelated chain not touched.
    assert "X" not in descendants
    assert "Y" not in descendants
