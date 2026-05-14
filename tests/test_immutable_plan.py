"""Coverage for the goldfive#247 frozen :class:`Plan` / :class:`Task` refactor.

These tests are the new-test surface for Phase 3 of the structural fix
documented at the top of ``goldfive/types.py``:

* :class:`Plan` and :class:`Task` are ``frozen=True``; in-place mutation
  raises :class:`dataclasses.FrozenInstanceError`.
* The blessed mutation primitives — :func:`replace_task`,
  :func:`with_task_status`, :func:`add_tasks`, :func:`replace_edges`,
  :func:`bump_revision` — return NEW Plan instances and never mutate
  the input.
* Multiple readers see independent snapshots: capturing a
  :class:`Plan` reference and then triggering a revision via the
  helpers leaves the captured reference at the old shape.
* :func:`set_session_plan` enforces the single-writer invariant via
  :data:`_CHANNEL_PROCESSOR_ACTIVE`. Outside the contextvar it warns
  (production) or raises :class:`PlanOwnershipViolation` (under
  ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from unittest import mock

import pytest

from goldfive.types import (
    Plan,
    PlanOwnershipViolation,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
    add_tasks,
    bump_revision,
    channel_processor_active,
    replace_edges,
    replace_task,
    set_session_plan,
    with_task_status,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan(*task_ids: str) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=("g1",),
        tasks=tuple(Task(id=tid, title=tid.upper()) for tid in task_ids),
        edges=tuple(
            TaskEdge(from_task_id=task_ids[i], to_task_id=task_ids[i + 1])
            for i in range(len(task_ids) - 1)
        ),
    )


# ---------------------------------------------------------------------------
# Frozen-dataclass enforcement
# ---------------------------------------------------------------------------


def test_task_is_frozen() -> None:
    """``task.status = X`` raises :class:`FrozenInstanceError`."""
    t = Task(id="t1", title="T1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.status = TaskStatus.RUNNING  # type: ignore[misc]


def test_plan_is_frozen() -> None:
    """``plan.tasks = ...`` and ``plan.id = ...`` both raise."""
    p = _plan("t1", "t2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.id = "p2"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tasks = ()  # type: ignore[misc]


def test_taskedge_is_frozen() -> None:
    e = TaskEdge(from_task_id="a", to_task_id="b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.from_task_id = "c"  # type: ignore[misc]


def test_plan_coerces_list_inputs_to_tuples() -> None:
    """Back-compat: callers may pass lists; ``__post_init__`` coerces.

    Test fixtures and the planner's JSON parser construct Plans with
    list-typed ``goal_ids`` / ``tasks`` / ``edges``. The frozen
    refactor stores them as tuples, so the dataclass coerces in
    ``__post_init__``.
    """
    p = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1", "g2"],  # type: ignore[arg-type]
        tasks=[Task(id="t1", title="T1")],  # type: ignore[arg-type]
        edges=[],  # type: ignore[arg-type]
    )
    assert isinstance(p.goal_ids, tuple)
    assert isinstance(p.tasks, tuple)
    assert isinstance(p.edges, tuple)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_replace_task_returns_new_plan_with_input_unchanged() -> None:
    """``replace_task`` is pure: input plan and tasks are not mutated."""
    p = _plan("t1", "t2")
    new = replace_task(p, "t1", title="renamed", description="x")
    assert new is not p
    # Input plan unchanged.
    assert p.tasks[0].title == "T1"
    assert p.tasks[0].description == ""
    # New plan has the change.
    new_t1 = next(t for t in new.tasks if t.id == "t1")
    assert new_t1.title == "renamed"
    assert new_t1.description == "x"
    # Other tasks preserved by reference (sharing is safe — they're
    # frozen).
    assert new.tasks[1] is p.tasks[1]


def test_replace_task_raises_when_id_missing() -> None:
    p = _plan("t1")
    with pytest.raises(KeyError):
        replace_task(p, "nope", status=TaskStatus.RUNNING)


def test_with_task_status_preserves_other_fields() -> None:
    """``with_task_status`` only flips the status; everything else is preserved."""
    p = Plan(
        id="p1",
        run_id="r1",
        goal_ids=("g1",),
        tasks=(
            Task(
                id="t1",
                title="The task",
                description="long description",
                assignee_agent_id="agent_a",
                predicted_duration_ms=1234,
                cancel_reason="prior",
                supersedes="t0",
            ),
        ),
        edges=(),
    )
    new = with_task_status(p, "t1", TaskStatus.RUNNING)
    new_t1 = new.tasks[0]
    assert new_t1.status is TaskStatus.RUNNING
    assert new_t1.title == "The task"
    assert new_t1.description == "long description"
    assert new_t1.assignee_agent_id == "agent_a"
    assert new_t1.predicted_duration_ms == 1234
    assert new_t1.cancel_reason == "prior"
    assert new_t1.supersedes == "t0"
    # Plan id / run_id / revision metadata preserved.
    assert new.id == p.id
    assert new.run_id == p.run_id
    assert new.revision_index == p.revision_index


def test_add_tasks_preserves_order_and_input() -> None:
    """``add_tasks`` appends; original order is preserved; input untouched."""
    p = _plan("t1", "t2")
    new = add_tasks(p, [Task(id="t3", title="T3"), Task(id="t4", title="T4")])
    # Input unchanged.
    assert [t.id for t in p.tasks] == ["t1", "t2"]
    # Output is original + new in order.
    assert [t.id for t in new.tasks] == ["t1", "t2", "t3", "t4"]
    # Existing tasks shared by reference.
    for old, new_t in zip(p.tasks, new.tasks[: len(p.tasks)], strict=True):
        assert old is new_t


def test_replace_edges_accepts_tuples_and_taskedges() -> None:
    """``replace_edges`` normalises both shapes."""
    p = _plan("t1", "t2", "t3")
    new = replace_edges(p, [("t1", "t3"), TaskEdge(from_task_id="t2", to_task_id="t3")])
    assert new is not p
    assert len(new.edges) == 2
    assert all(isinstance(e, TaskEdge) for e in new.edges)
    assert (new.edges[0].from_task_id, new.edges[0].to_task_id) == ("t1", "t3")
    assert (new.edges[1].from_task_id, new.edges[1].to_task_id) == ("t2", "t3")
    # Input edges preserved.
    assert len(p.edges) == 2
    assert (p.edges[0].from_task_id, p.edges[0].to_task_id) == ("t1", "t2")


def test_bump_revision_default_increments_by_one() -> None:
    p = _plan("t1")
    new = bump_revision(p)
    assert new.revision_index == p.revision_index + 1
    # Other revision metadata preserved (no override given).
    assert new.revision_kind == p.revision_kind
    assert new.revision_severity == p.revision_severity


def test_bump_revision_with_explicit_index_and_metadata() -> None:
    p = _plan("t1")
    new = bump_revision(
        p,
        revision_index=5,
        revision_kind="tool_error",
        revision_severity="warning",
        revision_reason="api 500",
        revision_trigger_event_id="evt-42",
    )
    assert new.revision_index == 5
    assert new.revision_kind == "tool_error"
    assert new.revision_severity == "warning"
    assert new.revision_reason == "api 500"
    assert new.revision_trigger_event_id == "evt-42"
    # Input untouched.
    assert p.revision_index == 0
    assert p.revision_kind == ""


# ---------------------------------------------------------------------------
# Snapshot semantics — readers see independent views
# ---------------------------------------------------------------------------


def test_multiple_readers_see_independent_snapshots() -> None:
    """A reader who captured ``plan_v1`` keeps seeing v1 after the swap.

    The bug class goldfive#247 targets: a judge / sink reads
    ``session.plan`` (call this ``plan_v1``), awaits an LLM, and
    produces a verdict. While the judge was awaiting, the live state
    swapped to ``plan_v2``. Pre-#247 — when Plan was mutable — the
    judge's ``plan_v1`` reference saw the v2 mutations. With frozen
    Plan, the reader's reference still points at v1 and only the
    pointer on the session has moved.
    """
    plan_v1 = _plan("t1", "t2")
    session = Session(run_id="r1", plan=plan_v1)
    captured = session.plan  # reader's snapshot reference

    # Live state advances: t1 marked COMPLETED.
    with channel_processor_active():
        set_session_plan(session, with_task_status(session.plan, "t1", TaskStatus.COMPLETED))

    # Captured snapshot is unchanged.
    assert captured is plan_v1
    assert captured.tasks[0].status is TaskStatus.PENDING
    # Session.plan moved.
    assert session.plan is not None
    assert session.plan is not plan_v1
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Channel-processor enforcement
# ---------------------------------------------------------------------------


def test_set_session_plan_inside_contextvar_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The blessed path: swap inside the contextvar is silent."""
    plan = _plan("t1")
    session = Session(run_id="r1")
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        with channel_processor_active():
            set_session_plan(session, plan)
    assert session.plan is plan
    assert all(
        "channel_processor_active" not in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_set_session_plan_outside_contextvar_warns_in_non_strict_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calling :func:`set_session_plan` outside the contextvar logs a WARNING.

    Production-defaults behaviour: the runtime stays defensive — we
    log and proceed. Strict mode (separate test) escalates to an
    exception. The check is the structural enforcement of the
    "single writer onto session.plan" invariant from the goldfive#247
    architectural fix.
    """
    plan = _plan("t1")
    session = Session(run_id="r1")
    # Force non-strict mode regardless of the test runner's pytest
    # auto-on default.
    with mock.patch.dict(os.environ, {"GOLDFIVE_STRICT_STATE_OWNERSHIP": "0"}):
        with caplog.at_level(logging.WARNING, logger="goldfive"):
            set_session_plan(session, plan)
    # Plan still installed — defensive, not blocking.
    assert session.plan is plan
    assert any(
        "channel_processor_active" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_set_session_plan_outside_contextvar_raises_in_strict_mode() -> None:
    """Strict mode (``GOLDFIVE_STRICT_STATE_OWNERSHIP=1``) raises.

    The CI / dev tripwire — same env gate the existing
    :class:`StateOwnershipViolation` audit uses. Production deploys
    with the variable unset never pay the penalty; tests pay because
    pytest auto-enables strict mode (mirrored from
    :mod:`goldfive._state_audit`).
    """
    plan = _plan("t1")
    session = Session(run_id="r1")
    with mock.patch.dict(os.environ, {"GOLDFIVE_STRICT_STATE_OWNERSHIP": "1"}):
        with pytest.raises(PlanOwnershipViolation):
            set_session_plan(session, plan)


def test_channel_processor_active_is_per_contextvar_token() -> None:
    """Nested entries / exits restore the prior value cleanly."""
    plan = _plan("t1")
    session = Session(run_id="r1", plan=plan)
    # Outer CM enters; inner CM enters; exits; outer is still active.
    with channel_processor_active():
        with channel_processor_active():
            set_session_plan(session, plan)  # inner: silent
        # Outer still active — this should also be silent.
        set_session_plan(session, plan)
    # Outside both: now strict-mode would raise.
    with mock.patch.dict(os.environ, {"GOLDFIVE_STRICT_STATE_OWNERSHIP": "1"}):
        with pytest.raises(PlanOwnershipViolation):
            set_session_plan(session, plan)


# ---------------------------------------------------------------------------
# Helper interaction with the channel-processor primitive
# ---------------------------------------------------------------------------


def test_helpers_do_not_swap_session_plan_on_their_own() -> None:
    """The derivation helpers are pure — they never touch a Session.

    A regression here would mean the helpers are entangled with the
    single-writer enforcement, which would re-couple the structural
    fix to a global. They must stay pure functions over Plan/Task.
    """
    plan = _plan("t1")
    session = Session(run_id="r1", plan=plan)
    # None of these touch session.
    _ = with_task_status(plan, "t1", TaskStatus.RUNNING)
    _ = add_tasks(plan, [Task(id="t2", title="T2")])
    _ = replace_edges(plan, [("t1", "t2")])
    _ = bump_revision(plan)
    _ = replace_task(plan, "t1", title="x")
    # session.plan still the original — the helpers didn't swap it.
    assert session.plan is plan


def test_post_helper_install_keeps_old_snapshot_intact() -> None:
    """End-to-end snapshot proof: derive + install via the helper +
    contextvar pair; the captured reference remains pointed at the
    old shape.

    This is the architectural goal of #247 demonstrated as a single
    test: torn-read elimination via type-system-enforced
    immutability.
    """
    plan_v1 = _plan("t1", "t2", "t3")
    session = Session(run_id="r1", plan=plan_v1)
    snapshot_a = session.plan  # judge captures here
    snapshot_b = session.plan  # second judge captures the same

    # While both judges await an LLM, the steerer-equivalent path
    # marks t1 COMPLETED, then add_tasks(t4), then bump_revision.
    with channel_processor_active():
        new = with_task_status(session.plan, "t1", TaskStatus.COMPLETED)
        new = add_tasks(new, [Task(id="t4", title="T4")])
        new = bump_revision(new, revision_index=1, revision_kind="tool_error")
        set_session_plan(session, new)

    # Snapshots A and B see plan_v1 — torn-read is structurally
    # impossible.
    assert snapshot_a is plan_v1
    assert snapshot_b is plan_v1
    assert snapshot_a.tasks[0].status is TaskStatus.PENDING
    assert {t.id for t in snapshot_a.tasks} == {"t1", "t2", "t3"}
    # Session has the new plan with the new shape.
    assert session.plan is not None
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    assert {t.id for t in session.plan.tasks} == {"t1", "t2", "t3", "t4"}
    assert session.plan.revision_index == 1
    assert session.plan.revision_kind == "tool_error"


# ---------------------------------------------------------------------------
# Phase 4 (#248) compatibility: Task.supersedes default is in place.
# ---------------------------------------------------------------------------


def test_task_supersedes_default_is_empty_string() -> None:
    """Phase 4 (#248) ships ``Task.supersedes`` validation; the field's
    default must be in place from Phase 3 so the wiring lands cleanly.
    """
    t = Task(id="t1", title="T1")
    assert t.supersedes == ""
