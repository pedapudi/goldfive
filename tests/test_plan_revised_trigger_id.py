"""goldfive#199 / harmonograf#95 rescope — PlanRevised.trigger_event_id.

Every refine path must stamp a non-empty ``trigger_event_id`` on the
emitted ``PlanRevised`` envelope. The strict-id dedup model harmonograf
uses has no time-window fallback (legacy behaviour is opt-in via env
var, default disabled), so a missing id means harmonograf will surface
the plan revision as its own card — which is the regression this module
pins against.

The tests cover:

  * User-control refines stamp the source ``annotation_id`` (from the
    ControlMessage payload / ``DriftDetected.annotation_id``).
  * Autonomous drift refines (LOOPING_REASONING, CONFABULATION_RISK,
    PLAN_DIVERGENCE, TOOL_ERROR, GOAL_DRIFT) stamp the
    ``DriftEvent.id`` — the UUID4 minted at DriftEvent construction.
  * Validator-retry within a single ``refine()`` call preserves the
    same trigger id across attempts (no fresh id per attempt).
  * Every refine-producing code path (steerer-driven, parallel
    executor's inline refine, sequential executor's out-of-band plan-swap
    emitter) produces a non-empty ``trigger_event_id``.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.events import plan_revised_event  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _revision_plan(
    *, revision_index: int = 1, revision_trigger_event_id: str = ""
) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_reason="pivot",
        revision_index=revision_index,
        revision_trigger_event_id=revision_trigger_event_id,
    )


def _user_steer_drift(annotation_id: str) -> DriftEvent:
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-test",
        payload={"note": "pivot", "annotation_id": annotation_id},
    )
    return DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=msg,
    )


def _autonomous_drift(kind: DriftKind = DriftKind.LOOPING_REASONING) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.WARNING,
        detail=f"autonomous {kind.value}",
    )


# ---------------------------------------------------------------------------
# test_user_steer_refine_stamps_annotation_id_on_trigger
# ---------------------------------------------------------------------------


def test_user_steer_refine_stamps_annotation_id_on_trigger() -> None:
    """USER_STEER drift → PlanRevised.trigger_event_id == drift.annotation_id."""
    drift = _user_steer_drift("ann_user_steer_42")
    plan = _revision_plan()
    evt = plan_revised_event(run_id="r1", sequence=5, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id == "ann_user_steer_42"


def test_user_steer_refine_falls_back_to_drift_id_without_annotation() -> None:
    """USER_STEER without bridge-supplied annotation_id still gets a strict id.

    Edge case: a ControlMessage arrives with no annotation_id in its
    payload (bridge misconfigured). The rescope requires a non-empty
    trigger_event_id on every refine — so the drift's own ``id`` is
    used as the fallback.
    """
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-no-ann",
        payload={"note": "pivot"},  # no annotation_id
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=msg,
    )
    plan = _revision_plan()
    evt = plan_revised_event(run_id="r1", sequence=5, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id == drift.id


# ---------------------------------------------------------------------------
# test_autonomous_drift_refine_stamps_drift_id_on_trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        DriftKind.LOOPING_REASONING,
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.CONFABULATION_RISK,
        DriftKind.PLAN_DIVERGENCE,
        DriftKind.TOOL_ERROR,
        DriftKind.GOAL_DRIFT,
        DriftKind.INTENT_DIVERGENCE,
    ],
)
def test_autonomous_drift_refine_stamps_drift_id_on_trigger(
    kind: DriftKind,
) -> None:
    """Autonomous drifts carry ``drift.id`` as ``trigger_event_id`` (goldfive#199).

    Parameterised over every autonomous drift kind that triggers a refine
    (LOOPING_*, CONFABULATION_RISK, PLAN_DIVERGENCE, TOOL_ERROR,
    GOAL_DRIFT, INTENT_DIVERGENCE). Each must produce a non-empty,
    drift-derived trigger id — never a guess-by-time-window.
    """
    drift = _autonomous_drift(kind)
    assert drift.id, "DriftEvent must default id to a non-empty UUID4"
    plan = _revision_plan()
    evt = plan_revised_event(run_id="r1", sequence=5, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id == drift.id


# ---------------------------------------------------------------------------
# test_validator_retry_preserves_original_trigger
# ---------------------------------------------------------------------------


def test_validator_retry_preserves_original_trigger() -> None:
    """Validator-retry within one refine() preserves the original trigger id.

    Within a single logical refine, the planner may retry internally
    (attempt 1 fails validation, attempt 2 succeeds). Both attempts
    operate on the same ``DriftEvent`` — so the final ``PlanRevised``
    uses the same drift-derived id regardless of how many internal
    attempts the planner took. There is no synthetic per-attempt id.
    """
    drift = _user_steer_drift("ann_chain_orig")
    # Attempt 2's refined plan is pre-stamped with the original trigger
    # id (what happens when _apply_revision was already run on a prior
    # attempt that then got rejected and retried).
    plan = _revision_plan(revision_trigger_event_id="ann_chain_orig")
    evt = plan_revised_event(run_id="r1", sequence=7, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id == "ann_chain_orig"
    assert evt.plan_revised.plan.revision_trigger_event_id == "ann_chain_orig"


def test_validator_retry_autonomous_drift_chains_drift_id() -> None:
    """Validator-retry on an autonomous drift chains the drift's id.

    Same contract as the user-steer case: one refine call → one
    trigger id — reused across internal retries rather than minted fresh.
    """
    drift = _autonomous_drift(DriftKind.TOOL_ERROR)
    plan = _revision_plan(revision_trigger_event_id=drift.id)
    evt = plan_revised_event(run_id="r1", sequence=7, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id == drift.id


# ---------------------------------------------------------------------------
# test_plan_revised_envelope_always_has_trigger_event_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "drift",
    [
        pytest.param(_user_steer_drift("ann_u"), id="user-steer"),
        pytest.param(_autonomous_drift(DriftKind.LOOPING_REASONING), id="looping-reasoning"),
        pytest.param(_autonomous_drift(DriftKind.CONFABULATION_RISK), id="confabulation"),
        pytest.param(_autonomous_drift(DriftKind.PLAN_DIVERGENCE), id="plan-divergence"),
        pytest.param(_autonomous_drift(DriftKind.TOOL_ERROR), id="tool-error"),
        pytest.param(_autonomous_drift(DriftKind.GOAL_DRIFT), id="goal-drift-periodic"),
    ],
)
def test_plan_revised_envelope_always_has_trigger_event_id(
    drift: DriftEvent,
) -> None:
    """Every refine-producing path emits a non-empty ``trigger_event_id``.

    Parameterised over the drift kinds that can trigger a refine. The
    test exercises ``plan_revised_event`` — the single helper every
    executor uses — so one assertion covers:

      * Steerer-driven refines (``DefaultSteerer._emit_plan_revised``)
      * Parallel executor inline refines (``plan_revised_event(drift=drift)``)
      * Sequential executor out-of-band detector (resolves via
        ``plan.revision_trigger_event_id`` which ``_apply_revision`` and
        ``ParallelDAGExecutor._refine`` both now stamp).
    """
    plan = _revision_plan()
    evt = plan_revised_event(run_id="r1", sequence=5, plan=plan, drift=drift)
    assert evt.plan_revised.trigger_event_id != ""


def test_plan_revised_trigger_id_proto_round_trips() -> None:
    """Wire-level round-trip: trigger_event_id survives serialise/parse.

    ``plan_revised_event`` derives the envelope-level ``trigger_event_id``
    from the drift. The ``Plan`` sub-message's
    ``revision_trigger_event_id`` is not automatically mutated by the
    helper (it reads from the plan, not writes onto it) — callers stamp
    it via ``_apply_revision`` before calling the helper.
    """
    from goldfive.pb.goldfive.v1 import events_pb2

    drift = _autonomous_drift(DriftKind.CONFABULATION_RISK)
    # Simulate the post-_apply_revision state: plan already has the id.
    plan = _revision_plan(revision_trigger_event_id=drift.id)
    evt = plan_revised_event(run_id="r1", sequence=9, plan=plan, drift=drift)
    encoded = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(encoded)
    assert decoded.plan_revised.trigger_event_id == drift.id
    # Plan mirror also round-trips so harmonograf's persistence can
    # reconstruct the id from a stored plan without the envelope.
    assert decoded.plan_revised.plan.revision_trigger_event_id == drift.id


# ---------------------------------------------------------------------------
# DriftDetected.id (backing strict-id for autonomous merges)
# ---------------------------------------------------------------------------


def test_drift_event_dataclass_default_id_is_nonempty_uuid() -> None:
    """``DriftEvent.id`` defaults to a non-empty UUID4 per goldfive#199.

    Without this, autonomous-drift refines would have no strict id to
    hand to harmonograf; the rescope's contract depends on every
    DriftEvent carrying an id by construction.
    """
    a = _autonomous_drift(DriftKind.LOOPING_REASONING)
    b = _autonomous_drift(DriftKind.LOOPING_REASONING)
    assert a.id
    assert b.id
    assert a.id != b.id  # per-event uniqueness


def test_drift_detected_envelope_stamps_drift_id() -> None:
    """``drift_detected_event`` stamps ``DriftEvent.id`` on the wire envelope.

    Harmonograf's drift ring will store this id so a subsequent
    ``PlanRevised.trigger_event_id`` (carrying the same UUID) can
    strict-match against the drift row on the aggregator side.
    """
    from goldfive.events import drift_detected_event

    drift = _autonomous_drift(DriftKind.PLAN_DIVERGENCE)
    evt = drift_detected_event(run_id="r1", sequence=3, drift=drift)
    assert evt.drift_detected.id == drift.id
    # annotation_id stays empty for autonomous drifts.
    assert evt.drift_detected.annotation_id == ""


def test_drift_detected_envelope_stamps_annotation_id_and_drift_id_for_user_steer() -> None:
    """User-control drifts carry BOTH the source annotation_id AND the drift id.

    harmonograf#95 (rescope): the annotation row provides the strict
    join key for the dedup merge (trigger_event_id == annotation.id).
    The drift's own id is still stamped so the drift record can be
    located on the autonomous-drift merge path without special-casing.
    """
    from goldfive.events import drift_detected_event

    drift = _user_steer_drift("ann_both_77")
    evt = drift_detected_event(run_id="r1", sequence=3, drift=drift)
    assert evt.drift_detected.annotation_id == "ann_both_77"
    assert evt.drift_detected.id == drift.id
