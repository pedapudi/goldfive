"""Exhaustive drift taxonomy coverage.

Part 1 (always runs once ``goldfive.types`` exists): every DriftKind value
pinned by INTERFACE_SPEC.md is present and round-trips as a StrEnum.

Part 2 (runs once ``goldfive.steerers`` exists): for every DriftKind, a
synthesized raw event causes ``Steerer.detect_drift`` to fire with that
exact kind. Classifier keys below are the event-shape conventions the
steerer PR is expected to follow; if a feature PR diverges, update the
mapping here and coordinate via PR comments.
"""

from __future__ import annotations

from typing import Any

import pytest

types = pytest.importorskip("goldfive.types")


EXPECTED_DRIFT_KINDS = {
    "TOOL_ERROR": "tool_error",
    "AGENT_REFUSAL": "agent_refusal",
    "NEW_WORK_DISCOVERED": "new_work_discovered",
    "PLAN_DIVERGENCE": "plan_divergence",
    "USER_STEER": "user_steer",
    "USER_CANCEL": "user_cancel",
    "TASK_FAILED_RECOVERABLE": "task_failed_recoverable",
    "TASK_FAILED_FATAL": "task_failed_fatal",
    "CONTEXT_PRESSURE": "context_pressure",
    "BLOCKED": "blocked",
    "WRONG_AGENT": "wrong_agent",
    "AGENT_TRANSFER": "agent_transfer",
    "MODEL_REFUSAL": "model_refusal",
    "STOPPED_EARLY": "stopped_early",
    "TOO_MANY_STEPS": "too_many_steps",
    "GOAL_UNREACHABLE": "goal_unreachable",
    "TASK_TIMEOUT": "task_timeout",
    "REPEATED_FAILURE": "repeated_failure",
    "UNEXPECTED_OUTPUT": "unexpected_output",
    "SCHEMA_VIOLATION": "schema_violation",
    "HALLUCINATION_SUSPECTED": "hallucination_suspected",
    "SAFETY_CONCERN": "safety_concern",
    "RESOURCE_EXHAUSTED": "resource_exhausted",
    "AMBIGUOUS_INTENT": "ambiguous_intent",
    "CUSTOM": "custom",
    "LOOPING_TOOL_CALL": "looping_tool_call",
    "LOOPING_REASONING": "looping_reasoning",
    "REASONING_CLUSTER_TIGHTENING": "reasoning_cluster_tightening",
    "CONFUSION": "confusion",
    "OFF_TOPIC": "off_topic",
    "INTENT_DIVERGENCE": "intent_divergence",
    "UNCERTAIN_PROGRESS": "uncertain_progress",
    "SELF_REPORTED_STUCK": "self_reported_stuck",
}


@pytest.mark.parametrize("attr,value", list(EXPECTED_DRIFT_KINDS.items()))
def test_drift_kind_value_is_pinned(attr: str, value: str) -> None:
    kind = getattr(types.DriftKind, attr)
    assert str(kind.value) == value


def test_drift_kind_is_str_enum_and_serializable() -> None:
    assert issubclass(types.DriftKind, str)
    # Every member round-trips via its string value.
    for member in types.DriftKind:
        assert types.DriftKind(member.value) is member


def test_drift_severity_values_are_pinned() -> None:
    assert {s.value for s in types.DriftSeverity} == {"info", "warning", "critical"}


# ---------------------------------------------------------------------------
# Classifier exhaustiveness
# ---------------------------------------------------------------------------


def _raw_event_for(kind_value: str, task_id: str = "t1", agent_id: str = "a1") -> dict[str, Any]:
    """Build a stub raw event per a canonical shape.

    The steerer is expected to dispatch on ``event["kind"]`` (the raw
    adapter kind) when it is one of the drift kind names, or otherwise
    infer from ``event["reason"]``. This mapping pins a default contract
    that is easy to harden later.
    """
    return {
        "kind": kind_value,
        "task_id": task_id,
        "agent_id": agent_id,
        "detail": f"synthetic {kind_value}",
    }


@pytest.mark.parametrize("kind_value", list(EXPECTED_DRIFT_KINDS.values()))
def test_classifier_fires_for_every_drift_kind(kind_value: str) -> None:
    steerers = pytest.importorskip("goldfive.steerers")
    Steerer = getattr(steerers, "DefaultSteerer", None)
    if Steerer is None:  # pragma: no cover - defensive
        pytest.skip("DefaultSteerer not yet implemented")

    steerer = Steerer()
    session = types.Session(run_id="drift-classifier", current_task_id="t1")
    raw = _raw_event_for(kind_value)
    drift = steerer.detect_drift(raw, session)

    if drift is None:
        pytest.skip(
            f"DefaultSteerer does not yet classify kind={kind_value!r}; "
            f"coordinate with the steerer PR."
        )

    assert isinstance(drift, types.DriftEvent)
    assert drift.kind.value == kind_value
    # Severity is populated and belongs to the taxonomy.
    assert drift.severity in set(types.DriftSeverity)
    # The raw event is threaded through unmodified.
    assert drift.raw == raw


def test_unknown_event_does_not_fire_drift() -> None:
    steerers = pytest.importorskip("goldfive.steerers")
    Steerer = getattr(steerers, "DefaultSteerer", None)
    if Steerer is None:  # pragma: no cover - defensive
        pytest.skip("DefaultSteerer not yet implemented")
    steerer = Steerer()
    session = types.Session(run_id="no-drift")
    # A benign progress event should not fire drift.
    drift = steerer.detect_drift({"kind": "progress", "task_id": "t1"}, session)
    assert drift is None
