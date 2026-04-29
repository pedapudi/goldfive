"""Tests for the drift-as-stateful-condition refactor (goldfive#271 PR1).

Pin the additive surface introduced by the first PR of the refactor:

* ``orchestration_state.compute_condition_id`` — stable for same
  kind+task+agent within a turn, distinct across turns.
* ``orchestration_state.open_or_escalate_drift`` — opens then escalates;
  severity bumps are monotonic; ``prev_severity`` carries the prior
  severity on escalation.
* ``orchestration_state.resolve_drift`` /
  ``escalate_drift_to_human_intervention`` — terminal lifecycle
  transitions; remove the entry from the active set; idempotent.
* :class:`OrchestrationStore` exposes the same surface as a typed
  veneer for callers that prefer the store handle.
* Wire integration: ``DefaultSteerer._emit_drift_detected`` stamps
  ``condition_id`` / ``lifecycle`` / ``prev_severity`` on
  ``DriftDetected`` and the existing fields (kind / severity / detail
  / synthetic / id) are unchanged — back-compat regression.

All tests are wire-compatible: a sink that doesn't know the new fields
sees one row per emit and renders identically. No harmonograf changes
are exercised here (out of scope for PR1).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import orchestration_state as _ostate  # noqa: E402
from goldfive.orchestration_store import OrchestrationStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Session,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ListSink:
    """Minimal ``EventSink`` capturing emitted Event protos for assertions."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    """Minimal Planner stub — never called by the wire-emit path under test."""

    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, **_: Any) -> Any:
        return None


# ---------------------------------------------------------------------------
# compute_condition_id — identity rules
# ---------------------------------------------------------------------------


def test_condition_id_is_stable_within_turn() -> None:
    """Same kind+task+agent+turn -> same condition_id."""
    cid1 = _ostate.compute_condition_id(
        kind=DriftKind.USER_STEER,
        task_id="t1",
        agent_id="a1",
        turn_id="run-001",
    )
    cid2 = _ostate.compute_condition_id(
        kind=DriftKind.USER_STEER,
        task_id="t1",
        agent_id="a1",
        turn_id="run-001",
    )
    assert cid1 == cid2
    # 16 chars (sha1 prefix).
    assert len(cid1) == 16


def test_condition_id_changes_across_turns() -> None:
    """A new turn opens a fresh condition for the same kind+task+agent."""
    cid_run1 = _ostate.compute_condition_id(
        kind=DriftKind.USER_STEER,
        task_id="t1",
        agent_id="a1",
        turn_id="run-001",
    )
    cid_run2 = _ostate.compute_condition_id(
        kind=DriftKind.USER_STEER,
        task_id="t1",
        agent_id="a1",
        turn_id="run-002",
    )
    assert cid_run1 != cid_run2


def test_condition_id_changes_across_kinds() -> None:
    """Different kinds within the same turn -> different condition_ids."""
    cid_steer = _ostate.compute_condition_id(
        kind=DriftKind.USER_STEER,
        task_id="t1",
        agent_id="a1",
        turn_id="run-001",
    )
    cid_loop = _ostate.compute_condition_id(
        kind=DriftKind.LOOPING_TOOL_CALL,
        task_id="t1",
        agent_id="a1",
        turn_id="run-001",
    )
    assert cid_steer != cid_loop


def test_condition_id_changes_across_tasks_and_agents() -> None:
    """Differ on either task_id or agent_id."""
    cid_t1 = _ostate.compute_condition_id(
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t1",
        agent_id="a1",
        turn_id="r",
    )
    cid_t2 = _ostate.compute_condition_id(
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t2",
        agent_id="a1",
        turn_id="r",
    )
    cid_a2 = _ostate.compute_condition_id(
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t1",
        agent_id="a2",
        turn_id="r",
    )
    assert cid_t1 != cid_t2
    assert cid_t1 != cid_a2
    assert cid_t2 != cid_a2


# ---------------------------------------------------------------------------
# open_or_escalate_drift — lifecycle progression
# ---------------------------------------------------------------------------


def test_open_or_escalate_opens_first_emit() -> None:
    """First emit: lifecycle=opened, occurrences=1, prev_severity=None."""
    state: dict[str, Any] = {}
    drift = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.LOOPING_TOOL_CALL,
        task_id="t1",
        agent_id="a1",
        turn_id="r",
        severity=DriftSeverity.WARNING,
    )
    assert drift.lifecycle == _ostate.LIFECYCLE_OPENED
    assert drift.occurrences == 1
    assert drift.prev_severity is None
    assert drift.severity is DriftSeverity.WARNING
    # Stored under condition_id.
    assert _ostate.get_active_drift(state, drift.condition_id) is not None


def test_open_or_escalate_re_emit_escalates() -> None:
    """Second emit on same kind+task+agent+turn: lifecycle=escalating."""
    state: dict[str, Any] = {}
    first = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.LOOPING_TOOL_CALL,
        task_id="t1",
        agent_id="a1",
        turn_id="r",
        severity=DriftSeverity.WARNING,
    )
    second = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.LOOPING_TOOL_CALL,
        task_id="t1",
        agent_id="a1",
        turn_id="r",
        severity=DriftSeverity.CRITICAL,
    )
    assert first.condition_id == second.condition_id
    assert second.lifecycle == _ostate.LIFECYCLE_ESCALATING
    assert second.prev_severity is DriftSeverity.WARNING
    assert second.severity is DriftSeverity.CRITICAL
    assert second.occurrences == 2


def test_open_or_escalate_severity_is_monotonic() -> None:
    """A lower-severity re-emit preserves the higher recorded severity."""
    state: dict[str, Any] = {}
    _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t",
        agent_id="a",
        turn_id="r",
        severity=DriftSeverity.CRITICAL,
    )
    drift = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t",
        agent_id="a",
        turn_id="r",
        severity=DriftSeverity.INFO,
    )
    # Severity stays at CRITICAL even though the re-emit reported INFO.
    assert drift.severity is DriftSeverity.CRITICAL
    # prev_severity is the *recorded* severity before the bump (CRITICAL).
    assert drift.prev_severity is DriftSeverity.CRITICAL
    assert drift.lifecycle == _ostate.LIFECYCLE_ESCALATING


def test_open_or_escalate_distinct_turns_are_independent() -> None:
    """Two emits in different turns produce two distinct conditions."""
    state: dict[str, Any] = {}
    d1 = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.USER_STEER,
        task_id="t",
        agent_id="a",
        turn_id="run-1",
        severity=DriftSeverity.WARNING,
    )
    d2 = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.USER_STEER,
        task_id="t",
        agent_id="a",
        turn_id="run-2",
        severity=DriftSeverity.WARNING,
    )
    assert d1.condition_id != d2.condition_id
    # Both opened (each is the first emit on its own condition).
    assert d1.lifecycle == _ostate.LIFECYCLE_OPENED
    assert d2.lifecycle == _ostate.LIFECYCLE_OPENED
    # Both still in the active set.
    assert len(_ostate.list_active_drifts(state)) == 2


# ---------------------------------------------------------------------------
# resolve_drift / escalate_drift_to_human_intervention
# ---------------------------------------------------------------------------


def test_resolve_drift_terminal_lifecycle_and_removed() -> None:
    """Resolve marks the condition resolved and removes from active set."""
    state: dict[str, Any] = {}
    opened = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.LOOPING_REASONING,
        task_id="t",
        agent_id="a",
        turn_id="r",
        severity=DriftSeverity.WARNING,
    )
    resolved = _ostate.resolve_drift(state, opened.condition_id)
    assert resolved is not None
    assert resolved.lifecycle == _ostate.LIFECYCLE_RESOLVED
    assert resolved.condition_id == opened.condition_id
    # Removed from active set.
    assert _ostate.get_active_drift(state, opened.condition_id) is None
    assert _ostate.list_active_drifts(state) == []


def test_resolve_drift_unknown_id_is_noop() -> None:
    """Resolving an unknown condition is a quiet no-op."""
    state: dict[str, Any] = {}
    assert _ostate.resolve_drift(state, "nope") is None
    assert _ostate.resolve_drift(state, "") is None


def test_escalate_to_human_intervention_terminal() -> None:
    """Escalation marks lifecycle=human_intervention_required and removes."""
    state: dict[str, Any] = {}
    opened = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.GOAL_DRIFT,
        task_id="t",
        agent_id="a",
        turn_id="r",
        severity=DriftSeverity.WARNING,
    )
    final = _ostate.escalate_drift_to_human_intervention(state, opened.condition_id)
    assert final is not None
    assert final.lifecycle == _ostate.LIFECYCLE_HUMAN_INTERVENTION_REQUIRED
    # Severity bumped to CRITICAL (level-4 contract).
    assert final.severity is DriftSeverity.CRITICAL
    # Removed from active set.
    assert _ostate.get_active_drift(state, opened.condition_id) is None


# ---------------------------------------------------------------------------
# OrchestrationStore — same surface, typed veneer
# ---------------------------------------------------------------------------


def test_orchestration_store_open_resolve_round_trip() -> None:
    """``OrchestrationStore`` exposes the same lifecycle progression."""
    session = Session(run_id="r1")
    store = OrchestrationStore.for_session(session)
    opened = store.open_or_escalate_drift(
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t1",
        agent_id="a1",
        turn_id="r1",
        severity=DriftSeverity.WARNING,
    )
    assert opened.lifecycle == _ostate.LIFECYCLE_OPENED
    # Lookup via the store.
    assert store.get_active_drift(opened.condition_id) is not None
    # Re-emit escalates.
    escalated = store.open_or_escalate_drift(
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t1",
        agent_id="a1",
        turn_id="r1",
        severity=DriftSeverity.CRITICAL,
    )
    assert escalated.condition_id == opened.condition_id
    assert escalated.lifecycle == _ostate.LIFECYCLE_ESCALATING
    # Resolve via the store.
    resolved = store.resolve_drift(opened.condition_id)
    assert resolved is not None
    assert resolved.lifecycle == _ostate.LIFECYCLE_RESOLVED
    assert store.active_drifts() == []


def test_orchestration_store_escalate_to_human() -> None:
    """``escalate_to_human_intervention`` is exposed on the store."""
    session = Session(run_id="r1")
    store = OrchestrationStore.for_session(session)
    opened = store.open_or_escalate_drift(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        task_id="t",
        agent_id="a",
        turn_id="r1",
        severity=DriftSeverity.CRITICAL,
    )
    final = store.escalate_to_human_intervention(opened.condition_id)
    assert final is not None
    assert final.lifecycle == _ostate.LIFECYCLE_HUMAN_INTERVENTION_REQUIRED
    assert store.active_drifts() == []


# ---------------------------------------------------------------------------
# Round-tripping through state.dict — Drift.to_dict / from_dict
# ---------------------------------------------------------------------------


def test_drift_dict_round_trip() -> None:
    """``Drift`` is JSON-friendly: to_dict / from_dict round-trips."""
    state: dict[str, Any] = {}
    original = _ostate.open_or_escalate_drift(
        state,
        kind=DriftKind.PLAN_DIVERGENCE,
        task_id="t",
        agent_id="a",
        turn_id="r",
        severity=DriftSeverity.WARNING,
    )
    payload = original.to_dict()
    restored = _ostate.Drift.from_dict(payload)
    assert restored.condition_id == original.condition_id
    assert restored.kind is original.kind
    assert restored.severity is original.severity
    assert restored.lifecycle == original.lifecycle
    assert restored.occurrences == original.occurrences


def test_drift_from_dict_tolerates_unknown_enum_values() -> None:
    """Garbage enum values fall back to None rather than raising."""
    drift = _ostate.Drift.from_dict(
        {
            "condition_id": "abc",
            "kind": "not_a_real_kind",
            "task_id": "t",
            "agent_id": "a",
            "turn_id": "r",
            "severity": "no_such_severity",
            "prev_severity": "",
            "lifecycle": _ostate.LIFECYCLE_OPENED,
            "occurrences": 1,
        }
    )
    assert drift.kind is None
    assert drift.severity is None


# ---------------------------------------------------------------------------
# Wire integration — DefaultSteerer._emit_drift_detected stamps the fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_drift_detected_stamps_lifecycle_opened() -> None:
    """A first-emit drift carries condition_id + DRIFT_LIFECYCLE_OPENED."""
    from goldfive.pb.goldfive.v1 import types_pb2

    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    session = Session(run_id="run-001")

    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="repeated tool call",
        current_task_id="t1",
        current_agent_id="a1",
    )
    await steerer._emit_drift_detected(session, drift)

    assert len(sink.events) == 1
    evt = sink.events[0]
    assert evt.HasField("drift_detected")
    payload = evt.drift_detected
    # Existing fields unchanged (back-compat regression).
    assert payload.detail == "repeated tool call"
    assert payload.current_task_id == "t1"
    assert payload.current_agent_id == "a1"
    # New fields stamped.
    assert payload.condition_id != ""
    assert len(payload.condition_id) == 16
    assert payload.lifecycle == types_pb2.DRIFT_LIFECYCLE_OPENED
    # prev_severity is UNSPECIFIED on the opening emit.
    assert payload.prev_severity == types_pb2.DRIFT_SEVERITY_UNSPECIFIED


@pytest.mark.asyncio
async def test_emit_drift_detected_escalates_within_turn() -> None:
    """A second emit on the same kind+task+agent in a turn -> ESCALATING."""
    from goldfive.pb.goldfive.v1 import types_pb2

    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    session = Session(run_id="run-001")

    drift_a = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        current_task_id="t1",
        current_agent_id="a1",
    )
    drift_b = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.CRITICAL,
        current_task_id="t1",
        current_agent_id="a1",
    )
    await steerer._emit_drift_detected(session, drift_a)
    await steerer._emit_drift_detected(session, drift_b)

    first, second = sink.events
    # Same condition_id across both emits.
    assert first.drift_detected.condition_id == second.drift_detected.condition_id
    assert second.drift_detected.lifecycle == types_pb2.DRIFT_LIFECYCLE_ESCALATING
    assert second.drift_detected.prev_severity == types_pb2.DRIFT_SEVERITY_WARNING
    assert second.drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL


@pytest.mark.asyncio
async def test_emit_drift_detected_distinct_turns_are_distinct_conditions() -> None:
    """Two sessions with different run_id -> distinct condition_ids."""
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())

    s1 = Session(run_id="run-001")
    s2 = Session(run_id="run-002")
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        current_task_id="t1",
        current_agent_id="a1",
    )
    await steerer._emit_drift_detected(s1, drift)
    await steerer._emit_drift_detected(s2, drift)
    assert len(sink.events) == 2
    cid1 = sink.events[0].drift_detected.condition_id
    cid2 = sink.events[1].drift_detected.condition_id
    assert cid1 != cid2


@pytest.mark.asyncio
async def test_emit_drift_detected_preserves_legacy_fields() -> None:
    """Regression: the existing wire fields are unchanged (no breaking move)."""
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    session = Session(run_id="run-001")

    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="overlay diverged",
        current_task_id="t1",
        current_agent_id="a1",
    )
    await steerer._emit_drift_detected(session, drift)
    payload = sink.events[0].drift_detected
    # Existing fields branch on these — none should be perturbed.
    assert payload.detail == "overlay diverged"
    assert payload.current_task_id == "t1"
    assert payload.current_agent_id == "a1"
    # Per-event id (#199) is still populated and DISTINCT from condition_id.
    assert payload.id != ""
    assert payload.id != payload.condition_id


# ---------------------------------------------------------------------------
# I5 — per-condition refine gate (lifecycle-driven)
# ---------------------------------------------------------------------------
#
# The drift-as-stateful-condition refactor (#271 PR1 / #318) gives every
# emit a stable ``condition_id`` and a ``lifecycle``. Without a gate,
# every re-emit of the same condition (``ESCALATING`` at the same
# severity) re-fired ``planner.refine`` — empirically dominating the
# LLM-call budget on long V24 runs. The gate skips refine when:
#
# * ``ESCALATING`` at the SAME severity as the prior emit (re-detection
#   without change — refine would be a no-op replay).
# * ``RESOLVED`` / ``HUMAN_INTERVENTION_REQUIRED`` (terminal lifecycle
#   for the condition; refine has either already happened or has been
#   lifted to the operator).
#
# ``OPENED`` and ``ESCALATING`` with a real severity bump still refine
# (baseline behaviour and genuine escalation).


class _RecordingPlanner:
    """Planner stub that records every ``refine`` call for assertions.

    Returns a structurally-modified plan so the steerer's
    refine-returned-None cascade (``_emit_refine_failure`` ->
    CRITICAL re-emit of the source drift) doesn't perturb the lifecycle
    sequence the gate tests assert on.
    """

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, *, plan: Any, drift: Any, goals: Any, **_: Any) -> Any:
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        # Return a copy of the plan with revision_index bumped — passes
        # the post-refine validator and keeps the lifecycle counter
        # cleanly at OPENED -> ESCALATING (no failure-cascade re-emit).
        import dataclasses as _dc

        revised = _dc.replace(
            plan,
            revision_index=plan.revision_index + 1,
            revision_kind=drift.kind,
            revision_severity=drift.severity,
            revision_reason="test-refine",
            revision_trigger_event_id=drift.id,
        )
        return revised


def _drift_event() -> DriftEvent:
    """Build a fresh WARNING-severity TOOL_ERROR drift on (t1, a1)."""
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="simulated tool error",
        current_task_id="t1",
        current_agent_id="a1",
    )


def _make_session_with_plan() -> Session:
    from goldfive.types import Goal, Plan, Task

    return Session(
        run_id="run-001",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=Plan(
            id="p1",
            run_id="run-001",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
            edges=[],
        ),
    )


@pytest.mark.asyncio
async def test_refine_gate_opened_lifecycle_fires_refine() -> None:
    """First emit of a condition (OPENED) -> refine runs (baseline)."""
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session_with_plan()

    await steerer._handle_drift(_drift_event(), session)
    assert len(planner.refine_calls) == 1


@pytest.mark.asyncio
async def test_refine_gate_escalating_same_severity_skips_refine() -> None:
    """Re-emit at the same severity (ESCALATING, prev==cur) -> NO refine."""
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session_with_plan()

    # First emit: OPENED -> refine runs.
    await steerer._handle_drift(_drift_event(), session)
    assert len(planner.refine_calls) == 1
    # Second emit at the SAME severity: ESCALATING with prev == cur.
    # Gate skips refine.
    await steerer._handle_drift(_drift_event(), session)
    assert len(planner.refine_calls) == 1, (
        "same-severity ESCALATING re-emit should NOT trigger another refine"
    )
    # But the wire emits still landed — observability is preserved.
    # The first emit is OPENED, the second is ESCALATING (same severity).
    # (Additional TOOL_ERROR emits may follow as the steerer's refine-
    # failure cascade re-emits at CRITICAL — those are ESCALATING with
    # a real bump, which is fine; we only assert on the OPENED+ESCALATING
    # prefix the test exercises.)
    from goldfive.pb.goldfive.v1 import types_pb2

    tool_error_emits = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof")
        and e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_TOOL_ERROR
    ]
    assert len(tool_error_emits) >= 2
    assert tool_error_emits[0].drift_detected.lifecycle == (types_pb2.DRIFT_LIFECYCLE_OPENED)
    assert tool_error_emits[1].drift_detected.lifecycle == (types_pb2.DRIFT_LIFECYCLE_ESCALATING)
    # Same-severity re-emit: severity stays WARNING and prev_severity
    # is also WARNING — this is the exact pattern the gate suppresses.
    assert tool_error_emits[1].drift_detected.severity == (types_pb2.DRIFT_SEVERITY_WARNING)
    assert tool_error_emits[1].drift_detected.prev_severity == (types_pb2.DRIFT_SEVERITY_WARNING)


@pytest.mark.asyncio
async def test_refine_gate_escalating_severity_bump_fires_refine() -> None:
    """Re-emit with prev_severity != severity (real escalation) -> refine fires."""
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session_with_plan()

    # First emit at WARNING.
    await steerer._handle_drift(_drift_event(), session)
    assert len(planner.refine_calls) == 1
    # Bump to CRITICAL: this is a real escalation. The TOOL_ERROR ladder
    # entry routes CRITICAL-first to CANCEL_REINVOKE which still calls
    # refine, so the gate must NOT block it.
    bumped = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.CRITICAL,
        detail="now critical",
        current_task_id="t1",
        current_agent_id="a1",
    )
    await steerer._handle_drift(bumped, session)
    assert len(planner.refine_calls) == 2, (
        "real severity escalation (WARNING -> CRITICAL) must still trigger refine"
    )


@pytest.mark.asyncio
async def test_refine_gate_resolved_condition_skips_refine() -> None:
    """A drift whose condition was previously RESOLVED -> NO refine.

    ``resolve_drift`` removes the entry from the active set; the gate
    treats a missing active entry as "condition closed" and skips
    refine — protecting against an out-of-band resolution between
    detection and dispatch.
    """
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session_with_plan()

    # First emit opens the condition.
    drift = _drift_event()
    await steerer._handle_drift(drift, session)
    assert len(planner.refine_calls) == 1
    # Resolve the condition out-of-band (sink callback / external
    # observer marks it cleared).
    cid = _ostate.compute_condition_id(
        kind=drift.kind,
        task_id=drift.current_task_id,
        agent_id=drift.current_agent_id,
        turn_id=session.run_id,
    )
    resolved = _ostate.resolve_drift(session.state, cid)
    assert resolved is not None
    assert resolved.lifecycle == _ostate.LIFECYCLE_RESOLVED
    # Now construct a NEW drift event that would re-fire on the same
    # condition. ``_stamp_drift_lifecycle`` will re-open the condition
    # (active set was empty), so this case naturally re-runs refine.
    # The gate's RESOLVED-protection is exercised via the get-then-act
    # race: stamp lifecycle, then resolve before _is_refine_gated_by_lifecycle
    # reads. We simulate that by directly invoking the gate after a
    # resolve.
    next_drift = _drift_event()
    # Manually stamp lifecycle (opens a fresh condition since the prior
    # was resolved+removed) then immediately resolve before the gate
    # check sees it -- this models the race the gate guards against.
    from goldfive.pb.goldfive.v1 import types_pb2  # noqa: F401

    # Run the gate WITHOUT going through _emit_drift_detected first:
    # the active set is empty (we just resolved), so the gate should
    # short-circuit at "tracked is None" -> True (skip).
    assert steerer._is_refine_gated_by_lifecycle(next_drift, session) is True


@pytest.mark.asyncio
async def test_refine_gate_human_intervention_lifecycle_skips_refine() -> None:
    """A condition escalated to HUMAN_INTERVENTION_REQUIRED -> NO refine."""
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session_with_plan()

    drift = _drift_event()
    # Open the condition then escalate to human intervention.
    cid = _ostate.compute_condition_id(
        kind=drift.kind,
        task_id=drift.current_task_id,
        agent_id=drift.current_agent_id,
        turn_id=session.run_id,
    )
    _ostate.open_or_escalate_drift(
        session.state,
        kind=drift.kind,
        task_id=drift.current_task_id,
        agent_id=drift.current_agent_id,
        turn_id=session.run_id,
        severity=drift.severity,
    )
    final = _ostate.escalate_drift_to_human_intervention(session.state, cid)
    assert final is not None
    assert final.lifecycle == _ostate.LIFECYCLE_HUMAN_INTERVENTION_REQUIRED
    # The active entry is gone — the gate treats a missing active drift
    # as "condition closed" and skips refine.
    assert steerer._is_refine_gated_by_lifecycle(drift, session) is True


@pytest.mark.asyncio
async def test_refine_gate_user_steer_bypasses_lifecycle_gate() -> None:
    """USER_STEER bypasses the gate — operator intent always honoured.

    Mirrors the existing ``_is_plan_revision_gated`` exemption for
    user / trajectory-level drifts. A re-emit of a USER_STEER condition
    must still flow through the refine path.
    """
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_RecordingPlanner())
    session = _make_session_with_plan()

    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please pivot",
        current_task_id="t1",
        current_agent_id="a1",
    )
    # Pre-open the condition so a fresh _is_refine_gated_by_lifecycle
    # would otherwise see an OPENED active entry, then pretend it was
    # already escalating at the same severity.
    _ostate.open_or_escalate_drift(
        session.state,
        kind=drift.kind,
        task_id=drift.current_task_id,
        agent_id=drift.current_agent_id,
        turn_id=session.run_id,
        severity=drift.severity,
    )
    _ostate.open_or_escalate_drift(
        session.state,
        kind=drift.kind,
        task_id=drift.current_task_id,
        agent_id=drift.current_agent_id,
        turn_id=session.run_id,
        severity=drift.severity,
    )
    # Even at ESCALATING-same-severity, USER_STEER must NOT be gated.
    assert steerer._is_refine_gated_by_lifecycle(drift, session) is False
