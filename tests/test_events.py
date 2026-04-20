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

from goldfive.events import build_plan_revision_diff  # noqa: E402
from goldfive.types import Plan, Task, TaskEdge, TaskStatus  # noqa: E402


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
