"""Unit tests for goldfive.types."""

from __future__ import annotations

import pytest

from goldfive.types import (
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
# Defaults
# ---------------------------------------------------------------------------


class TestTaskDefaults:
    def test_task_minimum_fields(self) -> None:
        t = Task(id="a", title="A")
        assert t.description == ""
        assert t.assignee_agent_id == ""
        assert t.status == TaskStatus.PENDING
        assert t.predicted_start_ms == 0
        assert t.predicted_duration_ms == 0
        assert t.bound_span_id == ""

    def test_task_edge_fields(self) -> None:
        e = TaskEdge(from_task_id="a", to_task_id="b")
        assert e.from_task_id == "a"
        assert e.to_task_id == "b"

    def test_plan_defaults(self) -> None:
        p = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[], edges=[])
        assert p.summary == ""
        assert p.revision_reason == ""
        assert p.revision_kind == ""
        assert p.revision_severity == ""
        assert p.revision_index == 0

    def test_goal_defaults(self) -> None:
        g = Goal(id="g1", summary="build it")
        assert g.success_predicate is None
        assert g.metadata == {}

    def test_drift_event_defaults(self) -> None:
        d = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
        assert d.detail == ""
        assert d.current_task_id == ""
        assert d.current_agent_id == ""
        assert d.raw is None

    def test_session_defaults(self) -> None:
        s = Session(run_id="r1")
        assert s.goals == []
        assert s.plan is None
        assert s.current_task_id == ""
        assert s.completed_results == {}
        assert s.task_progress == {}
        assert s.agent_notes == {}
        assert s.divergence_flag is False
        assert s.history == []
        assert s.started_at_ms == 0


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------


class TestEnumValues:
    def test_task_status_values(self) -> None:
        assert TaskStatus.PENDING.value == "PENDING"
        assert TaskStatus.RUNNING.value == "RUNNING"
        assert TaskStatus.COMPLETED.value == "COMPLETED"
        assert TaskStatus.FAILED.value == "FAILED"
        assert TaskStatus.CANCELLED.value == "CANCELLED"
        assert TaskStatus.BLOCKED.value == "BLOCKED"

    def test_drift_severity_values(self) -> None:
        assert DriftSeverity.INFO.value == "info"
        assert DriftSeverity.WARNING.value == "warning"
        assert DriftSeverity.CRITICAL.value == "critical"

    def test_drift_kind_has_custom(self) -> None:
        assert DriftKind.CUSTOM.value == "custom"

    def test_enums_are_strings(self) -> None:
        # StrEnum members should compare equal to their string value — this
        # is the contract relied on by the revision_* fields on Plan.
        assert TaskStatus.PENDING == "PENDING"
        assert DriftSeverity.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# Session sequence counter
# ---------------------------------------------------------------------------


class TestSessionSequence:
    def test_next_sequence_is_monotonic(self) -> None:
        s = Session(run_id="r1")
        assert s.next_sequence() == 0
        assert s.next_sequence() == 1
        assert s.next_sequence() == 2

    def test_sequence_counter_is_per_session(self) -> None:
        a = Session(run_id="a")
        b = Session(run_id="b")
        assert a.next_sequence() == 0
        assert a.next_sequence() == 1
        # b starts fresh regardless of a's state.
        assert b.next_sequence() == 0


# ---------------------------------------------------------------------------
# Goal predicate is a live callable
# ---------------------------------------------------------------------------


class TestGoalPredicate:
    def test_predicate_receives_session(self) -> None:
        calls: list[Session] = []

        def pred(s: Session) -> bool:
            calls.append(s)
            return s.current_task_id == "t1"

        g = Goal(id="g", summary="s", success_predicate=pred)
        sess = Session(run_id="r", current_task_id="t1")
        assert g.success_predicate is not None
        assert g.success_predicate(sess) is True
        assert calls == [sess]


# ---------------------------------------------------------------------------
# Plan.topological_stages() — edge cases
# ---------------------------------------------------------------------------


def _mk_plan(tasks: list[Task], edges: list[TaskEdge]) -> Plan:
    return Plan(id="p", run_id="r", goal_ids=[], tasks=tasks, edges=edges)


class TestTopologicalStages:
    def test_empty_plan(self) -> None:
        assert _mk_plan([], []).topological_stages() == []

    def test_single_task_no_edges(self) -> None:
        t = Task(id="t1", title="A")
        stages = _mk_plan([t], []).topological_stages()
        assert stages == [[t]]

    def test_independent_tasks_land_in_same_stage_sorted(self) -> None:
        tb = Task(id="b", title="B")
        ta = Task(id="a", title="A")
        tc = Task(id="c", title="C")
        stages = _mk_plan([tb, ta, tc], []).topological_stages()
        # Sorted by id within a stage.
        assert len(stages) == 1
        assert [t.id for t in stages[0]] == ["a", "b", "c"]

    def test_linear_chain(self) -> None:
        t1 = Task(id="t1", title="1")
        t2 = Task(id="t2", title="2")
        t3 = Task(id="t3", title="3")
        stages = _mk_plan(
            [t1, t2, t3],
            [TaskEdge("t1", "t2"), TaskEdge("t2", "t3")],
        ).topological_stages()
        assert [[t.id for t in s] for s in stages] == [["t1"], ["t2"], ["t3"]]

    def test_diamond(self) -> None:
        # t1 -> t2, t1 -> t3, t2 -> t4, t3 -> t4
        tasks = [Task(id=f"t{i}", title=str(i)) for i in range(1, 5)]
        edges = [
            TaskEdge("t1", "t2"),
            TaskEdge("t1", "t3"),
            TaskEdge("t2", "t4"),
            TaskEdge("t3", "t4"),
        ]
        stages = _mk_plan(tasks, edges).topological_stages()
        assert [[t.id for t in s] for s in stages] == [["t1"], ["t2", "t3"], ["t4"]]

    def test_edge_to_unknown_task_is_ignored(self) -> None:
        t1 = Task(id="t1", title="1")
        stages = _mk_plan([t1], [TaskEdge("t1", "ghost")]).topological_stages()
        # Only t1 is known, so it's stage 0 with nothing else.
        assert [[t.id for t in s] for s in stages] == [["t1"]]

    def test_edge_from_unknown_task_is_ignored(self) -> None:
        t1 = Task(id="t1", title="1")
        stages = _mk_plan([t1], [TaskEdge("ghost", "t1")]).topological_stages()
        # t1 still has in-degree 0 because the unknown edge is dropped.
        assert [[t.id for t in s] for s in stages] == [["t1"]]

    def test_cycle_tasks_land_in_trailing_stage(self) -> None:
        # t1 <-> t2 form a cycle; t3 is independent.
        t1 = Task(id="t1", title="1")
        t2 = Task(id="t2", title="2")
        t3 = Task(id="t3", title="3")
        stages = _mk_plan(
            [t1, t2, t3],
            [TaskEdge("t1", "t2"), TaskEdge("t2", "t1")],
        ).topological_stages()
        # t3 is ready immediately; t1/t2 can never drain so they're leftover.
        assert stages[0] == [t3]
        leftover_ids = {t.id for t in stages[-1]}
        assert leftover_ids == {"t1", "t2"}

    def test_empty_id_tasks_are_dropped(self) -> None:
        good = Task(id="t1", title="ok")
        bad = Task(id="", title="bad")
        stages = _mk_plan([good, bad], []).topological_stages()
        # Empty-id task is filtered out, not surfaced as leftover.
        assert [[t.id for t in s] for s in stages] == [["t1"]]

    def test_all_tasks_in_cycle_are_returned_as_leftover(self) -> None:
        # Pure cycle with no entry point — every task must still appear.
        t1 = Task(id="t1", title="1")
        t2 = Task(id="t2", title="2")
        stages = _mk_plan(
            [t1, t2],
            [TaskEdge("t1", "t2"), TaskEdge("t2", "t1")],
        ).topological_stages()
        # No stage 0 because no task has in-degree 0.
        assert len(stages) == 1
        assert {t.id for t in stages[0]} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Plan.validate() — structural validation at creation and revision.
# ---------------------------------------------------------------------------


class TestPlanValidate:
    def test_empty_plan_is_valid(self) -> None:
        # No tasks and no edges is structurally fine (the planner treats
        # an empty plan as "no plan", but validate() itself should not
        # raise).
        _mk_plan([], []).validate()
        _mk_plan([], []).validate(for_revision=True)

    def test_well_formed_pending_plan_is_valid(self) -> None:
        tasks = [
            Task(id="research", title="R"),
            Task(id="draft", title="D"),
            Task(id="review", title="V"),
        ]
        edges = [TaskEdge("research", "draft"), TaskEdge("draft", "review")]
        plan = _mk_plan(tasks, edges)
        plan.validate()
        plan.validate(for_revision=True)

    def test_duplicate_task_ids_rejected(self) -> None:
        tasks = [
            Task(id="a", title="A"),
            Task(id="a", title="A2"),
        ]
        plan = _mk_plan(tasks, [])
        with pytest.raises(ValueError, match="duplicate task id"):
            plan.validate()
        with pytest.raises(ValueError, match="duplicate task id"):
            plan.validate(for_revision=True)

    def test_empty_task_id_rejected(self) -> None:
        plan = _mk_plan([Task(id="", title="no id")], [])
        with pytest.raises(ValueError, match="empty id"):
            plan.validate()

    def test_edge_from_unknown_task_rejected(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1")],
            [TaskEdge("ghost", "t1")],
        )
        with pytest.raises(ValueError, match="unknown task id"):
            plan.validate()

    def test_edge_to_unknown_task_rejected(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1")],
            [TaskEdge("t1", "ghost")],
        )
        with pytest.raises(ValueError, match="unknown task id"):
            plan.validate()

    def test_simple_two_cycle_rejected(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1"), Task(id="t2", title="2")],
            [TaskEdge("t1", "t2"), TaskEdge("t2", "t1")],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.validate()

    def test_self_loop_rejected(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1")],
            [TaskEdge("t1", "t1")],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.validate()

    def test_three_cycle_rejected(self) -> None:
        plan = _mk_plan(
            [
                Task(id="t1", title="1"),
                Task(id="t2", title="2"),
                Task(id="t3", title="3"),
            ],
            [
                TaskEdge("t1", "t2"),
                TaskEdge("t2", "t3"),
                TaskEdge("t3", "t1"),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.validate()

    def test_partial_cycle_with_independent_task_rejected(self) -> None:
        # t3 is independent; t1 <-> t2 form a cycle. The cycle members
        # must still be flagged even though t3 is placeable.
        plan = _mk_plan(
            [
                Task(id="t1", title="1"),
                Task(id="t2", title="2"),
                Task(id="t3", title="3"),
            ],
            [TaskEdge("t1", "t2"), TaskEdge("t2", "t1")],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.validate()

    def test_creation_rejects_non_pending_task(self) -> None:
        plan = _mk_plan(
            [
                Task(id="t1", title="1", status=TaskStatus.COMPLETED),
            ],
            [],
        )
        with pytest.raises(ValueError, match="non-PENDING"):
            plan.validate()

    def test_creation_rejects_running_task(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1", status=TaskStatus.RUNNING)],
            [],
        )
        with pytest.raises(ValueError, match="non-PENDING"):
            plan.validate()

    def test_revision_allows_completed_tasks(self) -> None:
        plan = _mk_plan(
            [
                Task(id="done", title="D", status=TaskStatus.COMPLETED),
                Task(id="next", title="N", status=TaskStatus.PENDING),
            ],
            [TaskEdge("done", "next")],
        )
        # Raises at creation.
        with pytest.raises(ValueError, match="non-PENDING"):
            plan.validate(for_revision=False)
        # Allowed at revision.
        plan.validate(for_revision=True)

    def test_revision_allows_failed_and_cancelled(self) -> None:
        plan = _mk_plan(
            [
                Task(id="f", title="F", status=TaskStatus.FAILED),
                Task(id="c", title="C", status=TaskStatus.CANCELLED),
                Task(id="p", title="P", status=TaskStatus.PENDING),
            ],
            [],
        )
        plan.validate(for_revision=True)

    def test_revision_still_rejects_duplicate_ids(self) -> None:
        plan = _mk_plan(
            [
                Task(id="t", title="1", status=TaskStatus.COMPLETED),
                Task(id="t", title="2", status=TaskStatus.PENDING),
            ],
            [],
        )
        with pytest.raises(ValueError, match="duplicate task id"):
            plan.validate(for_revision=True)

    def test_revision_still_rejects_cycles(self) -> None:
        plan = _mk_plan(
            [
                Task(id="a", title="A", status=TaskStatus.PENDING),
                Task(id="b", title="B", status=TaskStatus.PENDING),
            ],
            [TaskEdge("a", "b"), TaskEdge("b", "a")],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.validate(for_revision=True)

    def test_revision_still_rejects_unknown_edges(self) -> None:
        plan = _mk_plan(
            [Task(id="t1", title="1", status=TaskStatus.COMPLETED)],
            [TaskEdge("t1", "ghost")],
        )
        with pytest.raises(ValueError, match="unknown task id"):
            plan.validate(for_revision=True)
