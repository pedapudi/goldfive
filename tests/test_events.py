"""Unit tests for goldfive.events helpers.

Focused on :func:`goldfive.events.build_plan_revision_diff`, the helper
that powers the ``PlanRevisionDiff`` sidecar attached to every
``PlanRevised`` event (closes PLAN-LIFECYCLE.md §8 gap #4). The helper
is deliberately small and pure so these tests pin its identity rules
(PLAN-LIFECYCLE.md §3.0) in isolation — full end-to-end wiring is
exercised in ``tests/test_steerer.py``.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.events import build_plan_revision_diff, plan_revised_event  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)


def _plan(
    tasks: list[Task] | None = None,
    edges: list[TaskEdge] | None = None,
    *,
    plan_id: str = "p1",
    run_id: str = "r1",
) -> Plan:
    return Plan(
        id=plan_id,
        run_id=run_id,
        goal_ids=["g1"],
        tasks=list(tasks or []),
        edges=list(edges or []),
    )


def test_plan_revision_diff_detects_added_task() -> None:
    """Tasks present in new but not old land in ``added_task_ids``."""
    old = _plan(tasks=[Task(id="t1", title="T1")])
    new = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_task_ids) == ["t2"]
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_edges) == []
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_detects_removed_task() -> None:
    """Tasks present in old but not new land in ``removed_task_ids``."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
        ]
    )
    new = _plan(tasks=[Task(id="t1", title="T1")])
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == ["t2"]
    assert list(diff.modified_task_ids) == []


def test_plan_revision_diff_detects_modified_task() -> None:
    """Same id, differing tracked metadata → ``modified_task_ids``.

    Tracked fields are ``title``, ``description``, ``assignee_agent_id``,
    ``status``. A change to any one of them is enough to surface the
    task as modified.
    """
    old = _plan(
        tasks=[
            Task(id="t1", title="old title", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ]
    )
    new = _plan(
        tasks=[
            # title changed
            Task(id="t1", title="new title", description="d", assignee_agent_id="a1"),
            # assignee changed
            Task(id="t2", title="T2", assignee_agent_id="a2"),
            # status changed
            Task(id="t3", title="T3", status=TaskStatus.COMPLETED),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    # All three modified; preserve new-plan order for deterministic UI.
    assert list(diff.modified_task_ids) == ["t1", "t2", "t3"]
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []


def test_plan_revision_diff_unchanged_task_not_modified() -> None:
    """Identical metadata → task must NOT appear in ``modified_task_ids``."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
        ]
    )
    new = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []


def test_plan_revision_diff_detects_added_edge() -> None:
    """An edge present in new but not old lands in ``added_edges``."""
    old = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[],
    )
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(old, new)
    assert len(diff.added_edges) == 1
    assert diff.added_edges[0].from_task_id == "t1"
    assert diff.added_edges[0].to_task_id == "t2"
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_detects_removed_edge() -> None:
    """An edge present in old but not new lands in ``removed_edges``."""
    old = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[],
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_edges) == []
    assert len(diff.removed_edges) == 1
    assert diff.removed_edges[0].from_task_id == "t1"
    assert diff.removed_edges[0].to_task_id == "t2"


def test_plan_revision_diff_edge_re_target_surfaces_both_added_and_removed() -> None:
    """Re-targeting an edge is modelled as (removed old, added new)."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    new = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t3")],
    )
    diff = build_plan_revision_diff(old, new)
    added = [(e.from_task_id, e.to_task_id) for e in diff.added_edges]
    removed = [(e.from_task_id, e.to_task_id) for e in diff.removed_edges]
    assert added == [("t1", "t3")]
    assert removed == [("t1", "t2")]


def test_plan_revision_diff_old_plan_none_treats_all_as_added() -> None:
    """An absent old plan means every task/edge in new is a fresh add."""
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(None, new)
    assert list(diff.added_task_ids) == ["t1", "t2"]
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert [(e.from_task_id, e.to_task_id) for e in diff.added_edges] == [("t1", "t2")]
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_identity_revision_is_empty() -> None:
    """Pathological case: new plan deep-equal to old plan → empty diff."""
    plan = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a"),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    # Build a second plan with the same contents (distinct objects; the
    # helper must not short-circuit on identity).
    other = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a"),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(plan, other)
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_edges) == []
    assert list(diff.removed_edges) == []


# ---------------------------------------------------------------------------
# plan_revised_event — trigger_event_id propagation (goldfive#199)
# ---------------------------------------------------------------------------


def _revised_plan(*, trigger_event_id: str = "") -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_reason="pivot",
        revision_index=1,
        revision_trigger_event_id=trigger_event_id,
    )


def test_plan_revised_event_reads_trigger_event_id_from_plan() -> None:
    """When ``plan.revision_trigger_event_id`` is set, the event copies it."""
    plan = _revised_plan(trigger_event_id="ann_plan_src")
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift_kind=plan.revision_kind,
        severity=plan.revision_severity,
        reason=plan.revision_reason,
        revision_index=plan.revision_index,
    )
    assert evt.plan_revised.trigger_event_id == "ann_plan_src"
    # Plan sub-message also carries it so persisted plans round-trip.
    assert evt.plan_revised.plan.revision_trigger_event_id == "ann_plan_src"


def test_plan_revised_event_user_steer_uses_annotation_id_from_drift() -> None:
    """User-control drift: trigger_event_id resolves to ControlMessage.annotation_id."""
    plan = _revised_plan(trigger_event_id="")  # not stamped on plan
    raw = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-x",
        payload={"note": "pivot", "annotation_id": "ann_from_drift"},
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=raw,
    )
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
    )
    assert evt.plan_revised.trigger_event_id == "ann_from_drift"


def test_plan_revised_event_explicit_kwarg_wins() -> None:
    """Explicit ``trigger_event_id`` kwarg overrides both plan and drift sources."""
    plan = _revised_plan(trigger_event_id="ann_plan_src")
    raw = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-y",
        payload={"note": "pivot", "annotation_id": "ann_from_drift"},
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=raw,
    )
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
        trigger_event_id="ann_override",
    )
    assert evt.plan_revised.trigger_event_id == "ann_override"


def test_plan_revised_event_autonomous_refine_stamps_drift_id() -> None:
    """Rescope: autonomous drift → trigger_event_id == drift.id (goldfive#199).

    Previously the field was left empty; harmonograf then relied on a
    time-window fallback. The rescope removes that guesswork — every
    refine carries a strict id.
    """
    plan = _revised_plan(trigger_event_id="")
    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="loop detected",
        raw={"event": "tool_error"},  # dict, not ControlMessage
    )
    assert drift.id  # DriftEvent defaults id to a UUID4 at construction
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
    )
    assert evt.plan_revised.trigger_event_id == drift.id


def test_plan_revised_event_proto_round_trips_trigger_event_id() -> None:
    """Serialize + deserialize the envelope — the field survives the wire."""
    from goldfive.pb.goldfive.v1 import events_pb2

    plan = _revised_plan(trigger_event_id="ann_roundtrip")
    evt = plan_revised_event(
        run_id="r1",
        sequence=9,
        plan=plan,
        drift_kind=plan.revision_kind,
        revision_index=plan.revision_index,
    )
    encoded = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(encoded)
    assert decoded.plan_revised.trigger_event_id == "ann_roundtrip"
    assert decoded.plan_revised.plan.revision_trigger_event_id == "ann_roundtrip"
