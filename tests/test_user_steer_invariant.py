"""Property-style tests for the L2 user-steer contract.

Pin: ``DefaultSteerer.install_user_steer`` ALWAYS returns a valid
:class:`Plan`. User-steer rejection is structurally impossible per
``docs/design/PLAN-LIFECYCLE.md`` §4.2.1.

The invariants under test:

1. The return value is a ``Plan`` instance, never ``None``.
2. The returned plan validates against ``prior`` via
   ``Plan.validate(for_revision=True, prior=prior)`` — no
   ``ValueError`` from the steerer.
3. ``returned.id == prior.id`` (plan-id stable across revisions —
   goldfive#271 Phase 4 invariant).
4. ``returned.revision_index == prior.revision_index + 1``.
5. Every prior-terminal task appears verbatim in the returned plan
   (status preserved — §3.1).
6. When the LLM revision is ``None``, every PENDING / RUNNING / BLOCKED
   prior task transitions to ``CANCELLED`` in the deterministic
   minimum (§4.2 delete-and-replan shape).

Inputs varied across the matrix:

* Plan shape: empty, single-task, multi-task with mixed terminals,
  multi-task all-PENDING, multi-task all-terminal.
* ``llm_revision``: ``None``, structurally valid revision, invalid
  revision dropping a terminal, invalid revision with a malformed
  edge.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402, I001
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Sink stub
# ---------------------------------------------------------------------------


class _Sink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Plan-shape fixtures
# ---------------------------------------------------------------------------


def _plan_empty(*, run_id: str = "run") -> Plan:
    return Plan(
        id="plan-empty",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[],
        edges=[],
        summary="empty",
    )


def _plan_single_pending(*, run_id: str = "run") -> Plan:
    return Plan(
        id="plan-single",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="single-pending",
    )


def _plan_mixed_terminals(*, run_id: str = "run") -> Plan:
    """One COMPLETED, one CANCELLED, one PENDING with edges."""
    return Plan(
        id="plan-mixed",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[
            Task(
                id="t1",
                title="T1",
                assignee_agent_id="w",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2",
                title="T2",
                assignee_agent_id="w",
                status=TaskStatus.CANCELLED,
            ),
            Task(id="t3", title="T3", assignee_agent_id="w"),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
        ],
        summary="mixed-terminals",
    )


def _plan_all_pending(*, run_id: str = "run") -> Plan:
    return Plan(
        id="plan-all-pending",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[
            Task(id="a", title="A", assignee_agent_id="w"),
            Task(id="b", title="B", assignee_agent_id="w"),
            Task(id="c", title="C", assignee_agent_id="w"),
        ],
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="b", to_task_id="c"),
        ],
        summary="all-pending-chain",
    )


def _plan_all_completed(*, run_id: str = "run") -> Plan:
    return Plan(
        id="plan-all-done",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[
            Task(
                id="x",
                title="X",
                assignee_agent_id="w",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="y",
                title="Y",
                assignee_agent_id="w",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[TaskEdge(from_task_id="x", to_task_id="y")],
        summary="all-completed",
    )


def _plan_blocked_running(*, run_id: str = "run") -> Plan:
    return Plan(
        id="plan-blocked-running",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[
            Task(
                id="r",
                title="R",
                assignee_agent_id="w",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="b",
                title="B",
                assignee_agent_id="w",
                status=TaskStatus.BLOCKED,
            ),
        ],
        edges=[],
        summary="running-and-blocked",
    )


_PLAN_FIXTURES = [
    _plan_empty,
    _plan_single_pending,
    _plan_mixed_terminals,
    _plan_all_pending,
    _plan_all_completed,
    _plan_blocked_running,
]


# ---------------------------------------------------------------------------
# llm_revision fixtures (parameterised against a chosen prior)
# ---------------------------------------------------------------------------


def _llm_revision_valid(prior: Plan) -> Plan:
    """A trivially valid revision: preserve terminals, drop mutables,
    add one fresh PENDING root.

    Mirrors what a well-behaved planner would emit. Distinguishable from
    the deterministic minimum (which adds NO fresh tasks) so tests can
    tell which branch ran.
    """
    new_tasks: list[Task] = []
    for t in prior.tasks:
        if t.status.is_terminal:
            new_tasks.append(dataclasses.replace(t))
    new_tasks.append(
        Task(id="llm-new", title="LLM-added pivot work", assignee_agent_id="w")
    )
    new_edges: list[TaskEdge] = []
    for e in prior.edges:
        ids = {t.id for t in new_tasks}
        if e.from_task_id in ids and e.to_task_id in ids:
            new_edges.append(dataclasses.replace(e))
    return Plan(
        id=prior.id,
        run_id=prior.run_id,
        goal_ids=list(prior.goal_ids),
        tasks=new_tasks,
        edges=new_edges,
        summary="llm-valid-revision",
    )


def _llm_revision_drops_terminal(prior: Plan) -> Plan | None:
    """Invalid: drops a prior-terminal task. Validator must reject (§3.1).

    Returns ``None`` when prior has no terminals — the input is then
    "no terminals to drop", which is structurally valid; the test
    matrix drops this row.
    """
    if not any(t.status.is_terminal for t in prior.tasks):
        return None
    new_tasks = [t for t in prior.tasks if not t.status.is_terminal]
    # Strip edges touching the dropped terminals.
    surviving = {t.id for t in new_tasks}
    new_edges = [
        e
        for e in prior.edges
        if e.from_task_id in surviving and e.to_task_id in surviving
    ]
    return Plan(
        id=prior.id,
        run_id=prior.run_id,
        goal_ids=list(prior.goal_ids),
        tasks=new_tasks,
        edges=new_edges,
        summary="llm-drops-terminal",
    )


def _llm_revision_malformed_edge(prior: Plan) -> Plan:
    """Invalid: edge references a task that does not exist."""
    new_tasks = [dataclasses.replace(t) for t in prior.tasks]
    bad_edge = TaskEdge(from_task_id="ghost", to_task_id="phantom")
    return Plan(
        id=prior.id,
        run_id=prior.run_id,
        goal_ids=list(prior.goal_ids),
        tasks=new_tasks,
        edges=[*[dataclasses.replace(e) for e in prior.edges], bad_edge],
        summary="llm-malformed-edge",
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _new_steerer() -> tuple[DefaultSteerer, _Sink]:
    sink = _Sink()
    steerer = DefaultSteerer()
    # ``planner=None`` is fine — install_user_steer never calls the
    # planner; it only consults ``llm_revision`` / falls back to the
    # deterministic minimum.
    steerer.bind(sinks=[sink], planner=None)
    return steerer, sink


def _new_session(prior: Plan) -> Session:
    session = Session(run_id=prior.run_id)
    session.plan = prior
    return session


def _drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="user pivot",
        authored_by="user",
    )


# ---------------------------------------------------------------------------
# Invariant 1-5: returned plan is a Plan that validates + identity holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prior_factory", _PLAN_FIXTURES, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    "llm_factory_name",
    ["none", "valid", "drops_terminal", "malformed_edge"],
)
async def test_install_user_steer_always_returns_valid_plan(
    prior_factory: Any, llm_factory_name: str
) -> None:
    prior = prior_factory()
    if llm_factory_name == "none":
        llm_revision: Plan | None = None
    elif llm_factory_name == "valid":
        llm_revision = _llm_revision_valid(prior)
    elif llm_factory_name == "drops_terminal":
        llm_revision = _llm_revision_drops_terminal(prior)
        if llm_revision is None:
            pytest.skip("prior has no terminals — drops-terminal row N/A")
    else:
        llm_revision = _llm_revision_malformed_edge(prior)

    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    drift = _drift()
    prior_index = prior.revision_index

    returned = await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=llm_revision,
        session=session,
    )

    # Invariant 1: returned IS a Plan.
    assert isinstance(returned, Plan), (
        f"install_user_steer returned non-Plan: {type(returned).__name__}"
    )

    # Invariant 2: validates without raising.
    try:
        returned.validate(for_revision=True, prior=prior)
    except ValueError as exc:
        raise AssertionError(
            f"install_user_steer returned a plan that does NOT validate "
            f"against prior: {exc}. prior tasks="
            f"{[(t.id, t.status.value) for t in prior.tasks]} "
            f"returned tasks={[(t.id, t.status.value) for t in returned.tasks]}"
        ) from exc

    # Invariant 3: plan_id stable.
    assert returned.id == prior.id, (
        f"plan_id changed across user-steer revision: prior={prior.id!r} "
        f"returned={returned.id!r}"
    )

    # Invariant 4: revision_index bumped.
    # Exception: when the deterministic minimum equals the prior plan
    # structurally (no PENDING/RUNNING/BLOCKED to cancel) AND the
    # LLM revision wasn't usable, install_user_steer short-circuits
    # (no PlanRevised emitted) and returns ``prior`` as-is. That's
    # acceptable degradation: the contract is "always a valid Plan",
    # not "always a fresh revision". This row covers all-terminal
    # priors and empty priors when the LLM input is unusable.
    used_fallback = llm_revision is None or llm_factory_name in (
        "drops_terminal",
        "malformed_edge",
    )
    no_mutables = not any(not t.status.is_terminal for t in prior.tasks)
    deterministic_no_op = used_fallback and no_mutables
    if not deterministic_no_op:
        assert returned.revision_index == prior_index + 1, (
            f"revision_index not bumped: prior={prior_index} "
            f"returned={returned.revision_index}"
        )

    # Invariant 5: every prior-terminal task is preserved verbatim.
    returned_by_id = {t.id: t for t in returned.tasks}
    for t in prior.tasks:
        if not t.status.is_terminal:
            continue
        rt = returned_by_id.get(t.id)
        assert rt is not None, (
            f"terminal task {t.id!r} dropped from user-steer revision"
        )
        assert rt.status is t.status, (
            f"terminal task {t.id!r} status regressed: "
            f"{t.status.value} -> {rt.status.value}"
        )


# ---------------------------------------------------------------------------
# Invariant 6: llm_revision=None on a plan with mutables -> mutables
# transition to CANCELLED in the deterministic minimum
# ---------------------------------------------------------------------------


async def test_deterministic_minimum_cancels_pending_running_blocked() -> None:
    """When the LLM revision is unavailable, every PENDING / RUNNING /
    BLOCKED task lands as CANCELLED — that is the §4.2 delete-and-replan
    shape, deterministic and provably valid."""
    prior = Plan(
        id="plan-mix-mutables",
        run_id="run",
        goal_ids=["g"],
        tasks=[
            Task(
                id="done",
                title="done",
                assignee_agent_id="w",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="pending",
                title="pending",
                assignee_agent_id="w",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="running",
                title="running",
                assignee_agent_id="w",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="blocked",
                title="blocked",
                assignee_agent_id="w",
                status=TaskStatus.BLOCKED,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="done", to_task_id="pending"),
            TaskEdge(from_task_id="running", to_task_id="blocked"),
        ],
        summary="mix",
    )
    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    drift = _drift()

    returned = await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=None,
        session=session,
    )

    statuses = {t.id: t.status for t in returned.tasks}
    assert statuses["done"] is TaskStatus.COMPLETED
    assert statuses["pending"] is TaskStatus.CANCELLED
    assert statuses["running"] is TaskStatus.CANCELLED
    assert statuses["blocked"] is TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# When the LLM revision IS valid, install_user_steer prefers it (the
# deterministic minimum is the FALLBACK, not the default).
# ---------------------------------------------------------------------------


async def test_install_user_steer_prefers_valid_llm_revision() -> None:
    prior = _plan_mixed_terminals()
    llm_revision = _llm_revision_valid(prior)
    # The LLM revision adds a task id "llm-new"; the deterministic
    # minimum does NOT. The returned plan having "llm-new" proves the
    # LLM branch fired.
    assert any(t.id == "llm-new" for t in llm_revision.tasks)

    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    drift = _drift()

    returned = await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=llm_revision,
        session=session,
    )
    assert any(t.id == "llm-new" for t in returned.tasks), (
        "valid LLM revision should be preferred over the deterministic "
        "minimum"
    )


# ---------------------------------------------------------------------------
# When the LLM revision is INVALID, fallback fires AND the steerer logs
# a warning (the operator must see WHY the deterministic shape ran).
# ---------------------------------------------------------------------------


async def test_install_user_steer_falls_back_on_invalid_llm_revision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    prior = _plan_mixed_terminals()
    invalid = _llm_revision_drops_terminal(prior)
    assert invalid is not None  # mixed_terminals has terminals to drop

    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    drift = _drift()

    caplog.set_level(logging.WARNING, logger="goldfive.steerer")
    returned = await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=invalid,
        session=session,
    )
    # Deterministic minimum: no LLM-added task; mutables cancelled.
    assert not any(t.id == "llm-new" for t in returned.tasks)
    # WARNING log emitted with the validator's error.
    assert any(
        "rejected by validator" in rec.message
        and "deterministic minimum" in rec.message
        for rec in caplog.records
    ), f"expected fallback WARNING; got {[r.message for r in caplog.records]!r}"


# ---------------------------------------------------------------------------
# Side-effect contract: ``session.refine_outcomes`` is NOT touched
# by the deterministic-fallback path. That table is for goldfive-
# authored autonomous refines (§4.5), not user-driven changes.
# ---------------------------------------------------------------------------


async def test_install_user_steer_does_not_touch_refine_outcomes() -> None:
    from goldfive.types import RefineOutcome

    prior = _plan_mixed_terminals()
    invalid = _llm_revision_drops_terminal(prior)
    assert invalid is not None

    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    # Pre-seed unrelated outcomes — invariant: they must NOT be cleared
    # OR mutated by install_user_steer.
    session.refine_outcomes[("user_steer", "")] = RefineOutcome(
        state="failed", fail_count=7
    )
    session.refine_outcomes[("plan_divergence", "t3")] = RefineOutcome(
        state="failed", fail_count=3
    )

    drift = _drift()
    await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=invalid,
        session=session,
    )
    assert session.refine_outcomes[("user_steer", "")].fail_count == 7
    assert session.refine_outcomes[("plan_divergence", "t3")].fail_count == 3


# ---------------------------------------------------------------------------
# session.plan is swapped on success.
# ---------------------------------------------------------------------------


async def test_install_user_steer_swaps_session_plan() -> None:
    prior = _plan_mixed_terminals()
    steerer, _sink = _new_steerer()
    session = _new_session(prior)
    drift = _drift()

    returned = await steerer.plans.install_user_steer(
        drift=drift,
        prior=prior,
        llm_revision=None,
        session=session,
    )
    # Mixed-terminals has a PENDING task; deterministic minimum cancels
    # it -> structural change -> session.plan is swapped.
    assert session.plan is returned
    assert session.plan.revision_index == prior.revision_index + 1
