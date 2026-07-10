"""Observability + side-effect contract for the PR-3 ladder demotions.

AGENCY-PRESERVATION.md PR 3 binding correctness requirements:

* **Observability preserved** — DriftDetected still emits for the
  demoted forecast-mismatch kinds; only the ladder *action* is removed.
  Sinks (harmonograf / zicato) are unaffected.
* **§5.3 side-effect check** — a demoted kind that no longer refines
  must also stop writing ``session.refine_outcomes`` (the dict other
  gates read for occurrence counts / the refine-skip gate). OBSERVE
  short-circuits ``_handle_drift_dispatch`` before the refine path, so
  the write never happens.

Per demoted kind, the observability path differs:

* ``CAPABILITY_MISMATCH`` (CRITICAL → OBSERVE) and the framework INFO
  ``NEW_WORK_DISCOVERED`` flow through ``handle_drift``, which emits
  ``DriftDetected`` and then OBSERVEs.
* ``PLAN_DIVERGENCE`` is dropped at the TOP of ``handle_drift`` (#252),
  so its observability comes from the executor reachability-audit
  emitter (``_plan_divergence_drift_event``), which builds a
  ``DriftDetected`` envelope directly. PR 3 leaves that emitter
  untouched; this test pins that it still carries the real
  ``PLAN_DIVERGENCE`` enum value.
* Agent-authored ``NEW_WORK_DISCOVERED`` reroutes to descriptive
  growth — its ``DriftDetected`` (INFO) is asserted in
  ``test_steerer`` / ``test_reporting``; here we pin the no-refine
  side-effect.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
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
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)


class _RecordingPlanner:
    """Planner stub that records every ``refine`` call so the test can
    assert a demoted kind never reaches the refine path.
    """

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def refine(self, **kwargs: Any) -> None:
        self.refine_calls.append(kwargs)
        return None


def _fresh() -> tuple[DefaultSteerer, Session, _ListSink, _RecordingPlanner]:
    steerer = DefaultSteerer()
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship it")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="A")],
            edges=[],
        ),
    )
    return steerer, session, sink, planner


def _drift_detected(sink: _ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]


@pytest.mark.parametrize("kind", [DriftKind.CAPABILITY_MISMATCH])
async def test_demoted_critical_kind_observes_without_refine_or_outcome(
    kind: DriftKind,
) -> None:
    """A CRITICAL drift of a demoted kind routed through ``handle_drift``
    emits DriftDetected (observability preserved) but takes NO ladder
    action: no ``planner.refine`` call and no ``refine_outcomes`` write
    (§5.3 side-effect check).
    """
    steerer, session, sink, planner = _fresh()
    drift = DriftEvent(
        kind=kind,
        severity=DriftSeverity.CRITICAL,
        detail="demoted",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    await steerer.drift._wait_background_drifts_idle()

    # Observability preserved.
    assert len(_drift_detected(sink)) == 1
    # No ladder action (the demotion).
    assert planner.refine_calls == []
    # §5.3: no refine_outcomes side effect for the demoted kind.
    assert dict(session.refine_outcomes) == {}


async def test_demoted_kind_still_records_signal_ledger_fire_at_observe() -> None:
    """PR #456's SignalLedger fire-recording survives the PR-3 demotion.

    The fire-recording hook (``_note_signal_drift_fire``) lives at the
    END of ``_emit_drift_detected`` — the single DriftDetected
    chokepoint, which runs in the dispatch BEFORE the ladder routing.
    So a CAPABILITY_MISMATCH routed to OBSERVE still records a fire on
    the ledger: the demotion removes the ladder *action* (no signal
    delivered), not the observability telemetry. This guards the lead's
    binding concern that demotions "must not silently stop telemetry for
    demoted kinds (DriftDetected + fire-recording still happen at
    OBSERVE)".
    """
    from goldfive.config import SteeringConfig
    from goldfive.signal_ledger import SignalLedger

    steerer = DefaultSteerer(
        steering_config=SteeringConfig(observation_only=False, signal_telemetry=True)
    )
    sink = _ListSink()
    planner = _RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship it")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="A")],
            edges=[],
        ),
    )
    session.current_task_id = "t1"
    drift = DriftEvent(
        kind=DriftKind.CAPABILITY_MISMATCH,
        severity=DriftSeverity.CRITICAL,
        detail="demoted",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    await steerer.drift._wait_background_drifts_idle()

    # OBSERVE: no signal delivered, no refine.
    assert planner.refine_calls == []
    assert len(_drift_detected(sink)) == 1
    # PR #456 fire-recording preserved: the ledger saw the fire, but no
    # delivery (OBSERVE delivers no signal).
    entry = SignalLedger.for_session(session).entry(
        DriftKind.CAPABILITY_MISMATCH.value, "t1"
    )
    assert entry is not None
    assert entry.fire_count >= 1
    assert not entry.has_delivery


async def test_plan_divergence_dropped_in_handle_drift_no_outcome() -> None:
    """PLAN_DIVERGENCE is dropped at the top of ``handle_drift`` (#252):
    no DriftDetected via this path, no refine, no ``refine_outcomes``
    write. Its observability comes from the executor audit emitter
    (asserted separately below).
    """
    steerer, session, sink, planner = _fresh()
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="diverged",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    await steerer.drift._wait_background_drifts_idle()

    assert _drift_detected(sink) == []
    assert planner.refine_calls == []
    assert dict(session.refine_outcomes) == {}


def test_plan_divergence_executor_audit_emitter_preserves_observability() -> None:
    """The executor reachability-audit emitter
    (``_plan_divergence_drift_event``) — PLAN_DIVERGENCE's live
    observability path — still builds a DriftDetected carrying the real
    PLAN_DIVERGENCE enum value. PR 3 leaves this emitter untouched.
    """
    from goldfive.executors.sequential import _plan_divergence_drift_event
    from goldfive.pb.goldfive.v1 import types_pb2

    session = Session(run_id="r1", goals=[], plan=None)
    session.current_task_id = "t1"
    evt = _plan_divergence_drift_event(session, "two PENDING tasks unreachable")

    assert evt.WhichOneof("payload") == "drift_detected"
    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_PLAN_DIVERGENCE
    assert evt.drift_detected.current_task_id == "t1"
