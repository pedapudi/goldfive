"""Regression tests for the I4 phantom-state fix.

Covers ``DefaultSteerer._fold_runtime_terminal_statuses`` and the four
install paths that wire it in:

* :meth:`DefaultSteerer._handle_drift` (autonomous refine)
* :meth:`DefaultSteerer._promote_drift_to_steer` (goldfive-promoted
  steer)
* :meth:`DefaultSteerer._install_with_drift` (NEW_WORK_DISCOVERED +
  USER_STEER ControlMessage)
* :meth:`DefaultSteerer.install_user_steer` (LLM-driven user-steer)

The v24 phantom-state escalation (iter_1 §I4, session
``2a324f78-8c73-416f-8a45-55314b39bda7``) showed that runtime
terminal transitions emitted out-of-band between revisions — the
overlay reaper's NOT_NEEDED reap, the SequentialExecutor reachability-
audit cancels, and an explicit coordinator ``mark_task_*`` — were not
folded into the next plan-revision snapshot. The persisted
``task_plan_revisions`` row therefore showed ``correct_research`` at
PENDING despite a ``task_cancelled`` event reaping it minutes earlier.

The fold is the persistence-boundary fix: at every install path,
before ``Plan.validate(for_revision=True, prior=...)`` runs, terminal
statuses on the prior plan are copied onto matching task ids in the
revised plan. Validators and downstream sinks see the runtime-correct
shape; the LLM's stale view is silently corrected.
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
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Local stubs (mirrored from test_steerer.py for self-containment)
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed: bool = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True

    @property
    def proto_events(self) -> list[Any]:
        out: list[Any] = []
        for e in self.events:
            which = getattr(e, "WhichOneof", None)
            if which is None:
                continue
            try:
                if which("payload") == "task_transitioned":
                    continue
            except Exception:
                pass
            out.append(e)
        return out


class StubPlanner:
    def __init__(self, *, revised: Plan | None = None) -> None:
        self.revised = revised

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return self.revised

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        return self.revised


def _make_session(plan: Plan) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


# ---------------------------------------------------------------------------
# Direct unit tests for the fold helper.
# ---------------------------------------------------------------------------


def test_fold_overwrites_pending_with_prior_not_needed() -> None:
    """The v24 archetype: prior has NOT_NEEDED, revised has PENDING.

    The fold must overwrite the revised entry with NOT_NEEDED so the
    persisted snapshot reflects the runtime ``task_cancelled`` event
    that fired between revisions.
    """
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.NOT_NEEDED,
                cancel_reason=(
                    "not_needed: overlay: tree did not exercise; "
                    "no follow-up dispatched (goldfive#163)"
                ),
            ),
        ],
        edges=[TaskEdge("research", "correct_research")],
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.PENDING,
            ),
            Task(id="next_step", title="next", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge("research", "correct_research"),
            TaskEdge("correct_research", "next_step"),
        ],
    )

    folded = DefaultSteerer._fold_runtime_terminal_statuses(revised, prior)

    assert folded == ["correct_research"], folded
    new_by_id = {t.id: t for t in revised.tasks}
    assert new_by_id["correct_research"].status is TaskStatus.NOT_NEEDED
    assert new_by_id["correct_research"].cancel_reason.startswith("not_needed: overlay:")
    # Untouched terminal stays as it was.
    assert new_by_id["research"].status is TaskStatus.COMPLETED
    # Newly-introduced PENDING is left alone.
    assert new_by_id["next_step"].status is TaskStatus.PENDING


def test_fold_no_op_when_revised_already_matches_prior_terminal() -> None:
    """Idempotent: when revised already has the prior terminal, no rewrite."""
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.CANCELLED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.CANCELLED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )

    folded = DefaultSteerer._fold_runtime_terminal_statuses(revised, prior)

    assert folded == []
    assert revised.tasks[0].status is TaskStatus.CANCELLED


def test_fold_preserves_genuine_terminal_to_terminal_regression() -> None:
    """A revised plan that flips COMPLETED → CANCELLED is a true regression.

    The fold must NOT silently rewrite it — let the validator surface
    the SCHEMA_VIOLATION. (Prior is COMPLETED, revised is CANCELLED;
    both terminal, so no fold should happen.)
    """
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.COMPLETED),
        ],
        edges=[],
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.CANCELLED),
        ],
        edges=[],
    )

    folded = DefaultSteerer._fold_runtime_terminal_statuses(revised, prior)

    # No fold — the validator owns this rejection.
    assert folded == []
    assert revised.tasks[0].status is TaskStatus.CANCELLED


def test_fold_handles_none_or_empty_prior() -> None:
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="t1")],
        edges=[],
    )

    assert DefaultSteerer._fold_runtime_terminal_statuses(revised, None) == []
    empty = Plan(id="p1", run_id="r1", goal_ids=["g1"], tasks=[], edges=[])
    assert DefaultSteerer._fold_runtime_terminal_statuses(revised, empty) == []


def test_fold_skips_tasks_present_only_in_prior() -> None:
    """A prior-only task is silently dropped from the revision (the
    LLM trimmed it). The fold operates on intersection — it does not
    re-insert; the validator's "terminal task missing in revision"
    check is the right enforcement point for that case."""
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.COMPLETED),
            Task(id="dropped", title="dropped", status=TaskStatus.NOT_NEEDED),
        ],
        edges=[],
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )

    folded = DefaultSteerer._fold_runtime_terminal_statuses(revised, prior)

    assert folded == []
    assert {t.id for t in revised.tasks} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Integration: the fold lands at every install path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_with_drift_folds_runtime_not_needed_into_snapshot() -> None:
    """End-to-end I4 reproducer.

    Mirrors the v24 trace shape: prior plan has ``correct_research``
    at NOT_NEEDED (the overlay reaper fired between revisions). The
    LLM's NEW_WORK_DISCOVERED revision still lists ``correct_research``
    at PENDING (the LLM saw the prior plan rendered without runtime
    state, or rebuilt the task from scratch). After install:

    * Validation passes (the fold corrected the regression before
      ``Plan.validate``).
    * ``session.plan`` carries ``correct_research`` at NOT_NEEDED, not
      PENDING — the persisted snapshot reflects runtime reality.
    * ``session.plan.revision_index`` advanced.
    """
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.NOT_NEEDED,
                cancel_reason=(
                    "not_needed: overlay: tree did not exercise; "
                    "no follow-up dispatched (goldfive#163)"
                ),
            ),
        ],
        edges=[TaskEdge("research", "correct_research")],
    )
    # Bump the prior's revision_index so we can see monotonic advance.
    prior.revision_index = 3
    session = _make_session(plan=prior)

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            # Phantom: LLM re-emits correct_research as PENDING.
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.PENDING,
            ),
            # Plus genuinely new work the LLM added this turn — note we
            # deliberately do NOT make ``finalize`` depend on
            # ``correct_research``: that would form an unreachable
            # NOT_NEEDED → PENDING edge after the fold, which the
            # validator must (and does) reject. The v24 archetype had
            # ``correct_research`` as a leaf task.
            Task(id="finalize", title="finalize", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge("research", "correct_research"),
            TaskEdge("research", "finalize"),
        ],
    )

    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="user added new follow-up",
        authored_by="goldfive",
    )

    installed = await steerer.install_revision_for_drift(
        session=session,
        drift=drift,
        revised_plan=revised,
    )

    assert installed is True, (
        "install must succeed: the fold corrects the regression before "
        "validation runs"
    )
    assert session.plan is revised
    assert session.plan.revision_index == 4
    new_by_id = {t.id: t for t in session.plan.tasks}
    assert new_by_id["correct_research"].status is TaskStatus.NOT_NEEDED, (
        "fold must overwrite the LLM's PENDING regression with the "
        "prior runtime NOT_NEEDED status"
    )
    assert new_by_id["correct_research"].cancel_reason.startswith(
        "not_needed: overlay:"
    )
    assert new_by_id["finalize"].status is TaskStatus.PENDING

    # No SCHEMA_VIOLATION drift was emitted — the fold corrected the
    # regression before validation ran.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert "plan_revised" in kinds, kinds
    schema_violations = [
        e
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
        and "plan validation failed" in e.drift_detected.detail
    ]
    assert not schema_violations, schema_violations


@pytest.mark.asyncio
async def test_install_with_drift_genuine_terminal_regression_still_rejected() -> None:
    """Validator still rejects a true terminal→different-terminal regression.

    Anti-pattern check: the fold must NOT relax the validator. A
    revised plan that flips a prior CANCELLED task to COMPLETED is a
    real bug — the steerer must surface SCHEMA_VIOLATION.
    """
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.CANCELLED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    session = _make_session(plan=prior)
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )

    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="genuine regression",
        authored_by="goldfive",
    )

    installed = await steerer.install_revision_for_drift(
        session=session,
        drift=drift,
        revised_plan=revised,
    )

    assert installed is False
    # Plan unchanged.
    assert session.plan is prior
    assert prior.tasks[0].status is TaskStatus.CANCELLED
    # SCHEMA_VIOLATION drift surfaced.
    schema_violations = [
        e
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
        and "plan validation failed" in e.drift_detected.detail
    ]
    assert schema_violations, "validator must still reject terminal regressions"


@pytest.mark.asyncio
async def test_install_user_steer_folds_before_validating_llm_revision() -> None:
    """:meth:`install_user_steer` folds the LLM revision before
    validating it, so a stale-prior-state LLM output still lands as
    the LLM revision (not the deterministic minimum)."""
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.NOT_NEEDED,
            ),
        ],
        edges=[TaskEdge("research", "correct_research")],
    )
    session = _make_session(plan=prior)

    llm_revision = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research",
                title="correct research",
                status=TaskStatus.PENDING,
            ),
            Task(id="finalize", title="finalize", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge("research", "correct_research"),
            TaskEdge("research", "finalize"),
        ],
    )

    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="user wants finalize",
        raw="finalize the work",
        authored_by="user",
    )

    returned = await steerer.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=llm_revision,
        session=session,
    )

    # The LLM revision was used (not the deterministic minimum) because
    # the fold corrected the regression before validation.
    assert returned is llm_revision
    new_by_id = {t.id: t for t in session.plan.tasks}
    assert new_by_id["correct_research"].status is TaskStatus.NOT_NEEDED
    assert new_by_id["finalize"].status is TaskStatus.PENDING


@pytest.mark.asyncio
async def test_handle_drift_refine_path_folds_before_validation() -> None:
    """:meth:`_handle_drift` (autonomous refine) also folds runtime
    terminals from session.plan before validating the planner's
    output."""
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_done", title="done", status=TaskStatus.COMPLETED),
            Task(
                id="t_reaped",
                title="reaped",
                status=TaskStatus.NOT_NEEDED,
            ),
            Task(id="t_pending", title="pending", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    session = _make_session(plan=prior)

    # Planner returns a revised plan that regressed t_reaped to PENDING.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_done", title="done", status=TaskStatus.COMPLETED),
            Task(id="t_reaped", title="reaped", status=TaskStatus.PENDING),
            Task(id="t_new", title="new", status=TaskStatus.PENDING),
        ],
        edges=[],
    )

    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=StubPlanner(revised=revised))

    # Trigger an autonomous refine via observe (any error context fires
    # the drift -> refine -> apply pipeline).
    await steerer.observe({"error": "trigger refine"}, session)

    # Plan was installed; t_reaped landed as NOT_NEEDED (folded), not
    # PENDING.
    new_by_id = {t.id: t for t in session.plan.tasks}
    assert new_by_id["t_reaped"].status is TaskStatus.NOT_NEEDED, (
        f"fold must apply during _handle_drift; got {new_by_id['t_reaped'].status!r}"
    )
    # No SCHEMA_VIOLATION emitted.
    schema_violations = [
        e
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
        and "plan validation failed" in e.drift_detected.detail
    ]
    assert not schema_violations, schema_violations


@pytest.mark.asyncio
async def test_apply_revision_defensive_fold_idempotent_when_caller_already_folded() -> None:
    """The defensive fold inside ``_apply_revision`` is idempotent —
    calling it after the caller already folded must not double-mutate
    or raise."""
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", status=TaskStatus.NOT_NEEDED),
        ],
        edges=[],
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            # Caller already folded.
            Task(id="t1", title="t1", status=TaskStatus.NOT_NEEDED),
            Task(id="t2", title="t2", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    session = _make_session(plan=prior)

    # First fold (caller-equivalent): no-op.
    folded_first = DefaultSteerer._fold_runtime_terminal_statuses(revised, prior)
    assert folded_first == []

    # _apply_revision's internal defensive fold: also no-op.
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="x",
    )
    DefaultSteerer._apply_revision(session, revised, drift)

    assert session.plan is revised
    new_by_id = {t.id: t for t in session.plan.tasks}
    assert new_by_id["t1"].status is TaskStatus.NOT_NEEDED
    assert new_by_id["t2"].status is TaskStatus.PENDING
