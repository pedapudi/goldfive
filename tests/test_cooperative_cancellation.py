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
    """FunctionTool-shaped stub: ``.name`` + ``.func`` (a callable)."""

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.func = lambda **_kw: {"ok": True}


class _FakeAgentTool:
    """AgentTool-shaped stub: ``.name`` + ``.agent`` (a sub-agent).

    Mirrors ADK's :class:`google.adk.tools.AgentTool` discriminator —
    the cooperative-cancel short-circuit in
    :meth:`~goldfive.adapters._adk_plugin._GoldfiveADKPlugin.before_tool_callback`
    distinguishes AgentTool dispatches from plain FunctionTool dispatches
    via :func:`_is_agent_tool_dispatch`. The duck-typed branch keys on
    ``getattr(tool, "agent", None) is not None``, so this stub trips
    the AgentTool path even when the optional ``adk`` extra is not
    importable in the test environment.
    """

    def __init__(self, *, name: str, agent_name: str = "sub_agent") -> None:
        self.name = name
        self.agent = _FakeAgent(name=agent_name)


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
# 3. before_tool_callback returns {"status": "cancelled"} for AgentTool
#    dispatches when the invocation is cancelled. FunctionTool dispatches
#    are NOT short-circuited (Bug C from v23 validation, goldfive#211610):
#    short-circuiting them silently strands committed side-effects (e.g.
#    write_webpage / patch_file file writes) on every supersede-cancel.
# ---------------------------------------------------------------------------


async def test_before_tool_returns_cancelled_status_for_agent_tool() -> None:
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
        tool=_FakeAgentTool(name="research_agent"),
        tool_args={"query": "test"},
        tool_context=tool_ctx,
    )
    assert response == {"status": "cancelled"}


async def test_before_tool_does_not_short_circuit_function_tool_on_cancel() -> None:
    """Bug C from v23 validation (goldfive#211610). Cancelling the
    in-flight invocation must NOT cause a plain FunctionTool dispatch
    (write_webpage, patch_file, …) to be skipped — the LLM has already
    committed to the call (args chosen, function_call event emitted)
    and the work is typically a side-effect (file write, DB row patch)
    we want to land. The next ``before_model_callback`` short-circuits
    the LLM call regardless, so the dispatch still ends cleanly.

    A short-circuit here would synthesize ``{"status": "cancelled"}``
    in place of the real tool result — exactly the v23 regression where
    ``write_webpage`` / ``patch_file`` returned ``cancelled`` after a
    GOAL_DRIFT-triggered supersede and no presentation file landed.
    """
    plugin, _sink, _session = _make_plugin()
    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(drift_kind="goal_drift", reason="drift"),
    )

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )
    tool_ctx = _FakeToolContext(invocation_context=inv_ctx)
    response = await plugin.before_tool_callback(
        tool=_FakeTool(name="write_webpage"),
        tool_args={"topic": "solar_flares", "html_content": "<html/>"},
        tool_context=tool_ctx,
    )
    # No short-circuit — the callback returns ``None`` so ADK proceeds
    # to dispatch the real FunctionTool. ``{"status": "cancelled"}``
    # would be the buggy response.
    assert response is None
    # The cancel-state flag is still pending — the next
    # ``before_model_callback`` will consume it and emit the
    # ``InvocationCancelled`` sink event there. Pre-fix this method
    # consumed it, blocking the LLM-call short-circuit from firing.
    assert plugin.peek_cancel_for_invocation("inv-1") is not None


async def test_before_tool_real_adk_function_tool_runs_during_supersede(tmp_path: Any) -> None:
    """End-to-end shape of Bug C: a real ADK ``FunctionTool`` whose
    ``func`` writes a file must actually run when the in-flight
    invocation has been flagged for a supersede-cancel. Pre-fix the
    callback returned ``{"status": "cancelled"}`` and the file was
    never written.

    We don't drive a full ADK ``Runner`` here (the cooperative-cancel
    contract is per-plugin-callback) — instead we assert the callback
    contract: ``before_tool_callback(...)`` returns ``None`` so ADK
    proceeds to dispatch the wrapped function. Then we manually invoke
    the function to model what ADK would do post-callback, and assert
    the side-effect landed.
    """
    pytest.importorskip("google.adk")
    from google.adk.tools import FunctionTool

    output_path = tmp_path / "solar_flares.html"

    def write_webpage(topic: str, html_content: str) -> str:
        output_path.write_text(html_content)
        return f"Wrote {topic}"

    tool = FunctionTool(write_webpage)

    plugin, _sink, _session = _make_plugin()
    plugin.request_invocation_cancel(
        invocation_id="inv-1",
        request=_make_request(drift_kind="goal_drift", reason="drift"),
    )

    inv_ctx = _FakeInvocationContext(
        invocation_id="inv-1",
        session_state={},
    )
    tool_ctx = _FakeToolContext(invocation_context=inv_ctx)
    response = await plugin.before_tool_callback(
        tool=tool,
        tool_args={"topic": "solar_flares", "html_content": "<html>flares</html>"},
        tool_context=tool_ctx,
    )
    # Callback returned ``None`` — ADK is told "proceed normally".
    assert response is None, (
        f"FunctionTool dispatch was short-circuited (response={response!r}); "
        "Bug C regression — write_webpage / patch_file must land their work "
        "even on supersede-cancel."
    )

    # Model what ADK does next: invoke the wrapped function. The
    # side-effect must land.
    write_webpage(topic="solar_flares", html_content="<html>flares</html>")
    assert output_path.exists()
    assert output_path.read_text() == "<html>flares</html>"


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
        kind=DriftKind.REASONING_CLUSTER_TIGHTENING,
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
    """The AgentTool response returned to the LLM must be ``{"status":
    "cancelled"}`` — bare shape. No ``reason`` / ``detail`` /
    ``drift_kind`` leak that the parent LLM could pattern-match on
    and invent workarounds (lesson from goldfive#250 / #252 / #253).

    Asserted on AgentTool dispatch only — FunctionTool dispatches are
    no longer short-circuited at this callback (Bug C from v23
    validation, goldfive#211610). The minimal-shape invariant still
    matters for the AgentTool path because the parent LLM reads that
    response verbatim as the sub-agent's "result".
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
        tool=_FakeAgentTool(name="researcher"),
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


# ---------------------------------------------------------------------------
# iter-11D: after_model_callback skips observe_reasoning on cancelled invocations
# ---------------------------------------------------------------------------
#
# Live e2e log (raccoon-research replay): after a CRITICAL drift cancels an
# invocation, the reasoning judges keep firing on the cancelled invocation's
# still-buffered thought blocks, producing spurious drifts on zombie
# reasoning and burning LLM-judge calls. The fix gates ``observe_reasoning``
# (and the opt-in ``note_llm_call`` reflective check) on the same sticky
# cancel flag the rest of the cancel-aware callbacks consult.


class _ReasoningPart:
    """ADK ``content.parts[i]`` shape with ``thought=True`` so
    :func:`goldfive.adapters._adk_plugin._extract_reasoning` returns a
    non-empty string for the test response.
    """

    def __init__(self, text: str, *, thought: bool = True) -> None:
        self.text = text
        self.thought = thought
        self.function_call = None


class _ReasoningContent:
    def __init__(self, parts: list[_ReasoningPart]) -> None:
        self.parts = parts


class _ReasoningLlmResponse:
    """LlmResponse-shaped stub carrying both a regular text part and a
    thought part. ``_extract_reasoning`` picks up the thought; the
    after_model_callback path also runs ``_extract_text_parts`` which
    reads the non-thought text for the regular ``steerer.observe``
    fan-out.
    """

    def __init__(self, *, text: str, reasoning: str) -> None:
        self.content = _ReasoningContent(
            [
                _ReasoningPart(text=text, thought=False),
                _ReasoningPart(text=reasoning, thought=True),
            ]
        )
        self.finish_reason = None


class _SteererSpy:
    """Wraps a real :class:`DefaultSteerer` so a test can assert which
    of ``observe`` / ``observe_reasoning`` / ``note_llm_call`` ran on
    a given ``after_model_callback`` invocation.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.observe_calls = 0
        self.observe_reasoning_calls = 0
        self.note_llm_calls = 0

    async def observe(self, observation: Any, session: Any) -> None:
        self.observe_calls += 1
        await self._inner.observe(observation, session)

    async def observe_reasoning(self, text: str, **kwargs: Any) -> None:
        self.observe_reasoning_calls += 1
        await self._inner.observe_reasoning(text, **kwargs)

    async def note_llm_call(self, session: Any) -> None:
        self.note_llm_calls += 1
        # Forward to inner so the counter on the real steerer advances
        # only when we actually delegate (no-op anyway when the inner
        # steerer wasn't given a reflective_call_llm).
        await self._inner.note_llm_call(session)

    # Plugin reads other attrs (e.g. ``request_invocation_cancel``,
    # ``_background_judges``) directly off the steerer; forward those.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _install_steerer_spy(plugin: Any) -> _SteererSpy:
    """Replace the ``SessionContext.steerer`` on the plugin's active
    context with a spy wrapping the real steerer. Returns the spy so
    the test can read its call counters.
    """
    ctx = plugin._active_ctx  # noqa: SLF001 -- test fixture poke
    spy = _SteererSpy(ctx.steerer)
    # ``SessionContext`` uses ``__slots__`` so direct attribute
    # assignment is supported and avoids a rebuild.
    ctx.steerer = spy
    return spy


def _make_after_model_callback_context(
    *,
    invocation_id: str,
    agent_name: str = "coordinator",
) -> _FakeCallbackContext:
    inv_ctx = _FakeInvocationContext(
        invocation_id=invocation_id,
        session_state={},
        agent_name=agent_name,
    )
    return _FakeCallbackContext(invocation_context=inv_ctx)


async def test_observe_reasoning_skipped_after_cancel(caplog: Any) -> None:
    """When the invocation is flagged cancelled, the
    after_model_callback path:

    * Does NOT call ``steerer.observe_reasoning`` (the reasoning judge
      / pattern detectors don't run on zombie reasoning).
    * Does NOT spawn a background judge task — ``_background_judges``
      is unchanged.
    * Does NOT bump the reflective-check counter via ``note_llm_call``.
    * Logs the "skipping observe_reasoning for cancelled invocation"
      debug line.
    * Still runs the regular ``steerer.observe`` LLM-response fan-out
      (operators need the cancelled turn's text observable; the gate
      is local to reasoning).
    """
    plugin, _sink, _session = _make_plugin()
    spy = _install_steerer_spy(plugin)
    bg_judges_before = set(spy._background_judges)  # noqa: SLF001 -- test asserts

    # Mark inv-1 cancelled BEFORE the after_model_callback fires —
    # mimics the real flow where a critical drift on an earlier
    # callback cancelled the invocation, and we're now looking at
    # already-buffered reasoning streaming back from the LLM.
    plugin.request_invocation_cancel(invocation_id="inv-1", request=_make_request())

    cb_ctx = _make_after_model_callback_context(invocation_id="inv-1")
    response = _ReasoningLlmResponse(
        text="ok let me think...",
        reasoning="I am uncertain. Maybe? Possibly? Perhaps? I don't know.",
    )

    import logging

    with caplog.at_level(logging.DEBUG, logger="goldfive.adapters.adk"):
        await plugin.after_model_callback(
            callback_context=cb_ctx,
            llm_response=response,
        )

    # Reasoning observation was skipped.
    assert spy.observe_reasoning_calls == 0
    # Reflective-check counter was skipped (zombie work shouldn't
    # consume the next reflective-check window).
    assert spy.note_llm_calls == 0
    # No background judge was spawned.
    assert set(spy._background_judges) == bg_judges_before  # noqa: SLF001
    # The regular llm_response observation still ran (the gate is
    # local to reasoning + reflective).
    assert spy.observe_calls == 1
    # Debug log records the skip with the inv id.
    skip_logs = [
        r
        for r in caplog.records
        if "skipping observe_reasoning for cancelled invocation" in r.getMessage()
    ]
    assert len(skip_logs) == 1, (
        f"expected exactly one skip log, got {[r.getMessage() for r in caplog.records]!r}"
    )
    assert "inv-1" in skip_logs[0].getMessage()


async def test_observe_reasoning_runs_when_not_cancelled() -> None:
    """Regression: when the invocation is NOT cancelled, the
    after_model_callback path runs ``observe_reasoning`` and
    ``note_llm_call`` as before. Guards against an over-broad gate
    that accidentally short-circuits every turn.
    """
    plugin, _sink, _session = _make_plugin()
    spy = _install_steerer_spy(plugin)

    cb_ctx = _make_after_model_callback_context(invocation_id="inv-2")
    response = _ReasoningLlmResponse(
        text="here is my answer",
        reasoning="the user asked X, so I will Y.",
    )

    await plugin.after_model_callback(
        callback_context=cb_ctx,
        llm_response=response,
    )

    # Both observation paths ran.
    assert spy.observe_calls == 1
    assert spy.observe_reasoning_calls == 1
    # And the reflective-check counter advanced (no-op inside the
    # inner steerer because reflective_call_llm wasn't configured,
    # but the spy still records the delegation).
    assert spy.note_llm_calls == 1


async def test_observe_reasoning_skip_does_not_break_other_observation() -> None:
    """The cancel gate must be local to the reasoning + reflective
    paths. The plain ``steerer.observe`` LLM-response fan-out (kind=
    'llm_response') should fire even on cancelled invocations so
    operators retain visibility into the cancelled turn's text.
    """
    plugin, _sink, _session = _make_plugin()
    spy = _install_steerer_spy(plugin)

    plugin.request_invocation_cancel(invocation_id="inv-3", request=_make_request())

    # Same cancelled invocation; this turn carries text + reasoning.
    cb_ctx = _make_after_model_callback_context(invocation_id="inv-3")
    response = _ReasoningLlmResponse(
        text="post-cancel buffered text",
        reasoning="post-cancel buffered reasoning",
    )

    # Capture observations the steerer received via the regular
    # observation path so we can confirm it was the llm_response fan-out.
    observed: list[Any] = []
    real_observe = spy._inner.observe  # noqa: SLF001

    async def _capturing_observe(observation: Any, session: Any) -> None:
        observed.append(observation)
        await real_observe(observation, session)

    spy._inner.observe = _capturing_observe  # type: ignore[method-assign] # noqa: SLF001

    await plugin.after_model_callback(
        callback_context=cb_ctx,
        llm_response=response,
    )

    # The reasoning + reflective gates short-circuited.
    assert spy.observe_reasoning_calls == 0
    assert spy.note_llm_calls == 0
    # But the llm_response observation ran exactly once.
    assert spy.observe_calls == 1
    assert len(observed) == 1
    obs = observed[0]
    # The observation kind is the llm_response fan-out; the cancelled
    # turn's text is still surfaced to sinks. ``_as_observation``
    # builds a plain dict in this adapter.
    obs_kind = obs.get("kind") if isinstance(obs, dict) else getattr(obs, "kind", "")
    assert obs_kind == "llm_response"
