"""Terminal-drift boundary cleanup (goldfive#271 follow-up to #307).

v15 evidence: when ``tool_loop_detector`` fired ``LOOPING_REASONING``
and the run escalated through the ladder, the coordinator_agent +
research_agent ``InvocationBoundaryEntered`` events were never paired
with an ``InvocationBoundaryExited`` because the executor paused the
run BEFORE the canonical ``except CancelledError`` arc in
:meth:`ADKAdapter._invoke_internal` could fire
``close_open_boundaries(reason="cancelled")``. Sinks (and the
harmonograf Gantt) rendered those spans as ``dur=(open)`` forever.

The fix: :meth:`DefaultSteerer._emit_drift_detected` consults
:meth:`_is_terminal_drift` after sending the drift on the wire and,
for the unconditionally-terminal kinds
(``HUMAN_INTERVENTION_REQUIRED`` and ``REPEATED_FAILURE``), calls the
plugin's :meth:`close_open_boundaries` helper with reason
``terminal_drift:<kind>`` so every still-open boundary gets a paired
``Exited`` event before the pause / teardown lands.

``LOOPING_REASONING`` is NOT in the terminal set — it is graduated
and CRITICAL-first is still recoverable via NUDGE. The v15 stuck-
spans symptom is still cleaned up because the ladder's eventual
``PAUSE_ESCALATE`` step emits a fresh ``HUMAN_INTERVENTION_REQUIRED``
drift, which IS terminal and triggers the close.

These tests pin the contract: cleanup fires for the terminal kinds,
the reason carries the drift kind, ``LOOPING_REASONING`` alone does
NOT fire cleanup (any severity), and the helper short-circuits
silently on missing-adapter / legacy-plugin paths.
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

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SessionContext,
    make_adk_plugin,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
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


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _FakeADKSession:
    def __init__(self, *, state: dict[str, Any], run_id: str = "run-test") -> None:
        self.state = state
        self.run_id = run_id
        self.id = "session-test"


class _FakeCallbackContext:
    def __init__(self, *, invocation_context: Any) -> None:
        self._invocation_context = invocation_context


class _FakeInvocationContext:
    def __init__(
        self,
        *,
        invocation_id: str,
        session_state: dict[str, Any] | None = None,
        agent_name: str = "coordinator",
    ) -> None:
        self.invocation_id = invocation_id
        self.session = _FakeADKSession(state=session_state or {})
        self.agent = _FakeAgent(name=agent_name)


class _FakeAdapter:
    """Minimal adapter shim that exposes ``_plugin`` (the only attribute
    :meth:`DefaultSteerer._close_open_boundaries_for_terminal_drift`
    needs to walk)."""

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin


def _make_session() -> Session:
    return Session(
        run_id="run-test",
        goals=[Goal(id="g1", summary="ship")],
        plan=Plan(
            id="p1",
            run_id="run-test",
            goal_ids=["g1"],
            tasks=[
                Task(id="t1", title="T1", status=TaskStatus.RUNNING),
                Task(id="t2", title="T2", status=TaskStatus.PENDING),
            ],
            edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        ),
        current_task_id="t1",
    )


def _events_of_kind(sink: _ListSink, kind: str) -> list[Any]:
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == kind:
                out.append(evt)
        except Exception:
            continue
    return out


def _build_steerer_with_open_boundaries(
    invocation_ids: tuple[str, ...] = ("inv-coordinator", "inv-research"),
) -> tuple[DefaultSteerer, Session, _ListSink, Any]:
    """Wire a steerer + plugin + sink, then open ``invocation_ids`` boundaries.

    Returns ``(steerer, session, sink, plugin)`` so the caller can
    drive the steerer's emit path and inspect emitted events.
    """
    plugin = make_adk_plugin(host_agent_name="coordinator")
    session = _make_session()
    sink = _ListSink()
    steerer = DefaultSteerer()
    # Sinks attached directly — the bind() helper requires a planner,
    # but for the cleanup-on-emit contract we only need ``_sinks``.
    steerer._sinks = [sink]
    steerer.bind_adapter(_FakeAdapter(plugin))
    ctx = SessionContext(
        session=session,
        steerer=steerer,
        tools=(),
        tool_handlers={},
        host_agent_name="coordinator",
        task=session.plan.tasks[0],
    )
    plugin.set_active_context(ctx)
    return steerer, session, sink, plugin


async def _open_boundaries(plugin: Any, sink: _ListSink, invocation_ids: tuple[str, ...]) -> None:
    for inv_id in invocation_ids:
        cb_ctx = _FakeCallbackContext(
            invocation_context=_FakeInvocationContext(invocation_id=inv_id),
        )
        agent = _FakeAgent(name="coordinator")
        await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    entered = _events_of_kind(sink, "invocation_boundary_entered")
    assert len(entered) == len(invocation_ids), (
        "boundary setup failed — expected one Entered per invocation"
    )


# ---------------------------------------------------------------------------
# 1. HUMAN_INTERVENTION_REQUIRED closes every still-open boundary
# ---------------------------------------------------------------------------


async def test_human_intervention_required_closes_open_boundaries() -> None:
    """A HUMAN_INTERVENTION_REQUIRED drift must fire the paired
    ``InvocationBoundaryExited(reason="terminal_drift:human_intervention_required")``
    for every still-open boundary."""
    steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
        ("inv-coordinator", "inv-research"),
    )
    await _open_boundaries(plugin, sink, ("inv-coordinator", "inv-research"))

    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="escalated for test",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer.drift._emit_drift_detected(session, drift)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 2, (
        "expected one Exited per still-open boundary; "
        f"got {[e.invocation_boundary_exited.reason for e in exited]}"
    )
    reasons = {e.invocation_boundary_exited.reason for e in exited}
    assert reasons == {"terminal_drift:human_intervention_required"}
    inv_ids = {e.invocation_boundary_exited.invocation_id for e in exited}
    assert inv_ids == {"inv-coordinator", "inv-research"}


# ---------------------------------------------------------------------------
# 2. LOOPING_REASONING -> escalation path (v15 scenario)
# ---------------------------------------------------------------------------


async def test_looping_reasoning_escalation_closes_open_boundaries() -> None:
    """Tool-loop detector firing ``LOOPING_REASONING`` does NOT itself
    close boundaries (CRITICAL-first is still recoverable via NUDGE).
    The escalation path that emits ``HUMAN_INTERVENTION_REQUIRED`` is
    what closes them — modelled here by emitting both drifts in
    sequence (the steerer's ladder dispatch in
    ``_dispatch_pause_escalate`` does this directly)."""
    steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
        ("inv-coordinator", "inv-research"),
    )
    await _open_boundaries(plugin, sink, ("inv-coordinator", "inv-research"))

    looping = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.CRITICAL,
        detail="reasoning loop count=5",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    # Step 1: LOOPING_REASONING fires -- by itself it does NOT close
    # boundaries because the run might still recover through the
    # nudge / refine ladder.
    await steerer.drift._emit_drift_detected(session, looping)
    assert _events_of_kind(sink, "invocation_boundary_exited") == [], (
        "LOOPING_REASONING alone must not close boundaries — CRITICAL-first is recoverable"
    )

    # Step 2: ladder escalates to HUMAN_INTERVENTION_REQUIRED. This is
    # the actual terminal signal and closes every open boundary.
    escalation = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="escalated from looping_reasoning",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer.drift._emit_drift_detected(session, escalation)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 2, (
        "expected one Exited per still-open boundary; "
        f"got {[e.invocation_boundary_exited.reason for e in exited]}"
    )
    reasons = {e.invocation_boundary_exited.reason for e in exited}
    assert reasons == {"terminal_drift:human_intervention_required"}


# ---------------------------------------------------------------------------
# 3. REPEATED_FAILURE closes still-open boundaries
# ---------------------------------------------------------------------------


async def test_repeated_failure_closes_open_boundaries() -> None:
    """Refine-failure-cap drift (CRITICAL ``REPEATED_FAILURE``) must
    close every still-open boundary."""
    steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
        ("inv-coordinator",),
    )
    await _open_boundaries(plugin, sink, ("inv-coordinator",))

    drift = DriftEvent(
        kind=DriftKind.REPEATED_FAILURE,
        severity=DriftSeverity.CRITICAL,
        detail="refine failed 3 consecutive times",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer.drift._emit_drift_detected(session, drift)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 1
    assert exited[0].invocation_boundary_exited.reason == "terminal_drift:repeated_failure"


# ---------------------------------------------------------------------------
# 4. LOOPING_REASONING by itself does NOT close boundaries (any severity)
# ---------------------------------------------------------------------------


async def test_looping_reasoning_alone_does_not_close_boundaries() -> None:
    """``LOOPING_REASONING`` is graduated and CRITICAL-first maps to
    ``NUDGE`` (recoverable). Closing on the LOOPING_REASONING emission
    itself would corrupt the boundary pair when the run actually
    recovers via refine + corrective follow-up. The ladder's eventual
    ``PAUSE_ESCALATE`` step emits ``HUMAN_INTERVENTION_REQUIRED``,
    which is what triggers the close."""
    for severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
        steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
            ("inv-coordinator",),
        )
        await _open_boundaries(plugin, sink, ("inv-coordinator",))

        drift = DriftEvent(
            kind=DriftKind.LOOPING_REASONING,
            severity=severity,
            detail=f"loop at severity={severity.value}",
            current_task_id="t1",
            current_agent_id="coordinator",
        )
        await steerer.drift._emit_drift_detected(session, drift)

        exited = _events_of_kind(sink, "invocation_boundary_exited")
        assert exited == [], (
            f"LOOPING_REASONING at {severity.value} must NOT close boundaries "
            "— recovery via refine / nudge is still possible"
        )


# ---------------------------------------------------------------------------
# 5. Non-terminal drift kinds do NOT close boundaries
# ---------------------------------------------------------------------------


async def test_non_terminal_drift_does_not_close_boundaries() -> None:
    """A garden-variety drift (e.g. AGENT_REFUSAL at WARNING) must not
    invoke the boundary-cleanup helper."""
    steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
        ("inv-coordinator",),
    )
    await _open_boundaries(plugin, sink, ("inv-coordinator",))

    drift = DriftEvent(
        kind=DriftKind.AGENT_REFUSAL,
        severity=DriftSeverity.WARNING,
        detail="model declined",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer.drift._emit_drift_detected(session, drift)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert exited == []


# ---------------------------------------------------------------------------
# 6. Cleanup is no-op when there are no still-open boundaries
# ---------------------------------------------------------------------------


async def test_terminal_drift_with_no_open_boundaries_is_noop() -> None:
    """A terminal drift fired AFTER the normal arc already closed every
    boundary must not emit a duplicate Exited event."""
    steerer, session, sink, plugin = _build_steerer_with_open_boundaries(
        ("inv-coordinator",),
    )
    # Open and immediately close via the normal arc.
    cb_ctx = _FakeCallbackContext(
        invocation_context=_FakeInvocationContext(invocation_id="inv-coordinator"),
    )
    agent = _FakeAgent(name="coordinator")
    await plugin.before_agent_callback(agent=agent, callback_context=cb_ctx)
    await plugin.after_agent_callback(agent=agent, callback_context=cb_ctx)
    # Sanity: one Exited(completed) on the wire from the normal arc.
    exited = _events_of_kind(sink, "invocation_boundary_exited")
    assert len(exited) == 1
    assert exited[0].invocation_boundary_exited.reason == "completed"

    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="late escalation",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    await steerer.drift._emit_drift_detected(session, drift)

    exited = _events_of_kind(sink, "invocation_boundary_exited")
    # No additional Exited event — cleanup walks an empty registry.
    assert len(exited) == 1


# ---------------------------------------------------------------------------
# 7. No adapter bound: cleanup is a silent no-op (does not raise)
# ---------------------------------------------------------------------------


async def test_terminal_drift_without_adapter_does_not_raise() -> None:
    """When the steerer has no adapter bound (test scaffold, early
    setup), the cleanup helper must short-circuit silently."""
    sink = _ListSink()
    session = _make_session()
    steerer = DefaultSteerer()
    steerer._sinks = [sink]
    # NOTE: no bind_adapter() — _adapter stays None.

    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="no adapter",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    # Must not raise.
    await steerer.drift._emit_drift_detected(session, drift)
    # DriftDetected still landed on the wire.
    assert _events_of_kind(sink, "drift_detected"), (
        "the drift itself must still be emitted even without an adapter"
    )


# ---------------------------------------------------------------------------
# 8. Plugin without close_open_boundaries: cleanup is a silent no-op
# ---------------------------------------------------------------------------


async def test_terminal_drift_with_legacy_plugin_does_not_raise() -> None:
    """A third-party / pre-#307 plugin that lacks
    ``close_open_boundaries`` must not break the steerer; the drift
    still lands on the wire."""

    class _LegacyPlugin:
        # Deliberately no close_open_boundaries attribute.
        pass

    sink = _ListSink()
    session = _make_session()
    steerer = DefaultSteerer()
    steerer._sinks = [sink]
    steerer.bind_adapter(_FakeAdapter(_LegacyPlugin()))

    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="legacy plugin",
        current_task_id="t1",
        current_agent_id="coordinator",
    )
    # Must not raise.
    await steerer.drift._emit_drift_detected(session, drift)
    assert _events_of_kind(sink, "drift_detected")
