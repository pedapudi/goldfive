"""Tests for the ``SteeringDecisionMade`` event.

Pairs with ``DriftDetected``: every positive-fire detector emit
produces a paired ``SteeringDecisionMade`` envelope. Every silent-path
detector decision (judge ran, decided on-task) produces a
``SteeringDecisionMade`` envelope alone.

The optimizer downstream consumes both classes; without the silent
path, the only training signal for tuning thresholds is the firing
detector positive class.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.events import steering_decision_made_event  # noqa: E402
from goldfive.pb.goldfive.v1 import events_pb2 as pb  # noqa: E402
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
    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, **_: Any) -> Any:
        return None


def _build_steerer() -> tuple[DefaultSteerer, _ListSink, Session]:
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    session = Session(run_id="run-test")
    return steerer, sink, session


def _steering_decisions(sink: _ListSink) -> list[Any]:
    return [e for e in sink.events if e.WhichOneof("payload") == "steering_decision_made"]


def _drift_detected(sink: _ListSink) -> list[Any]:
    return [e for e in sink.events if e.WhichOneof("payload") == "drift_detected"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_steering_decision_made_event_factory_basic_fields() -> None:
    evt = steering_decision_made_event(
        "run-1",
        7,
        detector_name="reasoning_judge",
        outcome="no_drift",
        reason="judge verdict: on_task",
        score=0.95,
        task_id="task_x",
        agent_name="research_agent",
    )
    assert evt.run_id == "run-1"
    assert evt.sequence == 7
    payload = evt.steering_decision_made
    assert payload.detector_name == "reasoning_judge"
    assert payload.outcome == "no_drift"
    assert payload.reason == "judge verdict: on_task"
    assert payload.score == pytest.approx(0.95)
    assert payload.task_id == "task_x"
    assert payload.agent_name == "research_agent"
    # decided_at always populated by the factory.
    assert payload.decided_at.seconds > 0 or payload.decided_at.nanos > 0


def test_steering_decision_made_event_factory_invalid_score_is_coerced() -> None:
    # NaN is a valid float; we don't coerce it. But non-numeric coerces to 0.
    evt2 = steering_decision_made_event(
        "run-1",
        1,
        detector_name="x",
        outcome="no_drift",
        score="not a number",  # type: ignore[arg-type]
    )
    assert evt2.steering_decision_made.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Proto envelope shape
# ---------------------------------------------------------------------------


def test_steering_decision_made_is_in_event_payload_oneof() -> None:
    """``Event.payload`` must accept the new SteeringDecisionMade variant."""
    e = pb.Event()
    e.steering_decision_made.detector_name = "tool_loops"
    e.steering_decision_made.outcome = "drift_emitted"
    assert e.WhichOneof("payload") == "steering_decision_made"


# ---------------------------------------------------------------------------
# DriftObserver pairs every DriftDetected with a SteeringDecisionMade
# ---------------------------------------------------------------------------


async def test_emit_drift_detected_pairs_with_steering_decision() -> None:
    steerer, sink, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="reasoning departed from the bound task",
        current_task_id="t1",
        current_agent_id="researcher",
    )
    await steerer.drift._emit_drift_detected(session, drift)
    decisions = _steering_decisions(sink)
    drifts = _drift_detected(sink)
    assert len(drifts) == 1
    assert len(decisions) == 1
    decision = decisions[0].steering_decision_made
    assert decision.detector_name == "reasoning_judge"
    assert decision.outcome == "drift_emitted"
    assert decision.considered_severity == "warning"
    assert decision.chosen_severity == "warning"
    assert decision.drift_id == drift.id
    assert decision.task_id == "t1"
    assert decision.agent_name == "researcher"


async def test_emit_drift_detected_under_user_steer_suppression() -> None:
    steerer, sink, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail="tree wandered",
        suppressed_by_user_steer=True,
    )
    await steerer.drift._emit_drift_detected(session, drift)
    decisions = _steering_decisions(sink)
    assert len(decisions) == 1
    decision = decisions[0].steering_decision_made
    assert decision.outcome == "drift_suppressed"
    assert decision.considered_severity == "critical"
    # chosen_severity is empty on suppression — the steerer didn't apply it.
    assert decision.chosen_severity == ""
    assert "suppressed" in decision.reason


# ---------------------------------------------------------------------------
# Silent-path emit helper
# ---------------------------------------------------------------------------


async def test_emit_no_drift_decision_fires_with_outcome_no_drift() -> None:
    steerer, sink, session = _build_steerer()
    await steerer.drift.emit_no_drift_decision(
        session=session,
        detector_name="goal_drift_judge",
        reason="judge verdict: progressing",
        task_id="t2",
    )
    decisions = _steering_decisions(sink)
    assert len(decisions) == 1
    assert _drift_detected(sink) == []
    decision = decisions[0].steering_decision_made
    assert decision.outcome == "no_drift"
    assert decision.detector_name == "goal_drift_judge"
    assert decision.drift_id == ""
    assert decision.task_id == "t2"


async def test_emit_no_drift_decision_threads_score() -> None:
    steerer, sink, session = _build_steerer()
    await steerer.drift.emit_no_drift_decision(
        session=session,
        detector_name="reflective_check",
        reason="agent self-reported healthy",
        score=0.82,
    )
    decisions = _steering_decisions(sink)
    assert decisions[0].steering_decision_made.score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Detector-name resolution
# ---------------------------------------------------------------------------


def test_detector_name_for_drift_known_kinds() -> None:
    cases = [
        (DriftKind.OFF_TOPIC, "reasoning_judge"),
        (DriftKind.JUSTIFIED_DEVIATION, "reasoning_judge"),
        (DriftKind.GOAL_DRIFT, "goal_drift_judge"),
        (DriftKind.LOOPING_REASONING, "reasoning_loop_embedding"),
        (DriftKind.CAPABILITY_MISMATCH, "capability_check"),
        (DriftKind.CONFABULATION_RISK, "confabulation_risk"),
        (DriftKind.USER_STEER, "user_control"),
        (DriftKind.SELF_REPORTED_STUCK, "reflective_check"),
    ]
    from goldfive.drift_observer import DriftObserver

    for kind, expected in cases:
        drift = DriftEvent(kind=kind, severity=DriftSeverity.INFO)
        assert DriftObserver._detector_name_for_drift(drift) == expected


def test_detector_name_for_drift_unknown_falls_back_to_kind_value() -> None:
    from goldfive.drift_observer import DriftObserver

    drift = DriftEvent(kind=DriftKind.CUSTOM, severity=DriftSeverity.INFO)
    name = DriftObserver._detector_name_for_drift(drift)
    # CUSTOM isn't in the map; falls back to the bare kind value.
    assert name == "custom"


# ---------------------------------------------------------------------------
# Sequence ordering: DriftDetected first, paired SteeringDecisionMade second
# ---------------------------------------------------------------------------


async def test_drift_detected_emits_before_paired_decision() -> None:
    steerer, sink, session = _build_steerer()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="x",
    )
    await steerer.drift._emit_drift_detected(session, drift)
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert kinds == ["drift_detected", "steering_decision_made"]
    # Sequence numbers strictly increasing.
    seqs = [e.sequence for e in sink.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
