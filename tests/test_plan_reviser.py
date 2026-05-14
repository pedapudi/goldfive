"""Regression tests for :mod:`goldfive.plan_reviser`.

This module hosts focused regressions for the plan-revision install
pipeline that lives in :class:`goldfive.plan_reviser.PlanReviser` (the
post-#408 home of the steerer's revision logic).

Currently covered:

* **goldfive#403** — partial-apply window between
  :meth:`PlanReviser._apply_revision` and
  :meth:`PlanReviser._emit_plan_revised`. Pre-fix the first call swapped
  ``session.plan`` and stamped
  ``session.last_addressed_revision_by_drift_key`` OUTSIDE the
  per-session plan lock; the caller then ``await``-ed
  ``_cancel_inflight_for_revision`` (which yields the event loop) before
  ``_emit_plan_revised`` acquired the lock. Readers calling
  ``_wait_plan_stable`` during the yield observed ``session.plan`` with
  the bumped ``revision_index`` but with the un-rewired edge DAG and
  un-repinned ``current_task_id``. The fix defers every session-
  mutation site into ``_emit_plan_revised`` under the lock.
"""

from __future__ import annotations

import asyncio
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
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ListSink:
    """Records every emitted event (proto + dict envelopes)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    """Planner whose ``refine`` returns the configured ``revised`` plan.

    Each successful ``refine`` call returns a *fresh* clone of
    ``revised`` with a per-call sentinel task appended so the
    steerer's no-op revision rejection (goldfive#271) sees a real
    structural diff. Mirrors the StubPlanner used by
    :mod:`tests.test_refine_atomicity_events`.
    """

    def __init__(self, *, revised: Plan) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append({"plan": plan, "drift": drift})
        sentinel_id = f"stub-refine-{len(self.refine_calls)}"
        return Plan(
            id=self.revised.id,
            run_id=self.revised.run_id,
            goal_ids=list(self.revised.goal_ids),
            tasks=[
                Task(
                    id=t.id,
                    title=t.title,
                    description=t.description,
                    assignee_agent_id=t.assignee_agent_id,
                    status=t.status,
                    supersedes=t.supersedes,
                    supersedes_kind=t.supersedes_kind,
                )
                for t in self.revised.tasks
            ]
            + [Task(id=sentinel_id, title=sentinel_id)],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in self.revised.edges
            ],
            summary=self.revised.summary,
            revision_index=self.revised.revision_index,
        )


def _initial_plan() -> Plan:
    return Plan(
        id="p-403",
        run_id="r-403",
        goal_ids=["g-403"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t2", to_task_id="t3"),
        ],
    )


def _supersedes_revised(prior: Plan) -> Plan:
    """A revision that supersedes ``t1`` with a corrective ``t1_corr``.

    The supersedes link is intentionally ``CORRECT``-kind so the DAG
    rewrite inside :meth:`PlanReviser._integrate_correction_supersedes`
    fires (an edge ``t1 -> t2`` is rewritten to ``t1_corr -> t2``).
    This is the second ``set_session_plan`` site pre-#403, and the
    partial-apply detector keys off the un-rewired ``t1 -> t2`` edge
    being visible together with the bumped ``revision_index`` — the
    exact intermediate state the audit observed.

    The revised plan deliberately ships the **pre-integration** edge
    shape (``t1 -> t2`` retained alongside the new ``t1 -> t1_corr``
    introduction edge); the executor expects
    ``_integrate_correction_supersedes`` to rewrite ``t1 -> t2`` into
    ``t1_corr -> t2``. Pre-#403 the first ``set_session_plan`` (inside
    ``_apply_revision``) wrote this pre-integration shape onto
    ``session.plan``; the integration ran later inside
    ``_emit_plan_revised``'s lock, producing the second swap with the
    rewired shape. The polling reader in the test catches frames
    between those two swaps.

    Uses ``t1=COMPLETED`` (terminal but not absorbing) so the
    revised plan satisfies the reachability invariant
    (``Plan.validate`` step 7): edges from CANCELLED/FAILED to PENDING
    are rejected, but COMPLETED→PENDING is allowed.
    """
    return Plan(
        id=prior.id,
        run_id=prior.run_id,
        goal_ids=list(prior.goal_ids),
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(
                id="t1_corr",
                title="T1 corrected",
                status=TaskStatus.PENDING,
                supersedes="t1",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ],
        # PRE-INTEGRATION shape: t1 -> t2 retained, plus the new
        # introduction edge t1 -> t1_corr. The integration step is
        # what rewrites t1 -> t2 into t1_corr -> t2.
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t2", to_task_id="t3"),
            TaskEdge(from_task_id="t1", to_task_id="t1_corr"),
        ],
        revision_index=prior.revision_index + 1,
    )


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r-403",
        goals=[Goal(id="g-403", summary="exercise #403")],
        plan=plan if plan is not None else _initial_plan(),
    )


def _drift(
    *,
    kind: DriftKind = DriftKind.TOOL_ERROR,
    severity: DriftSeverity = DriftSeverity.WARNING,
    task_id: str = "t1",
    detail: str = "trigger #403 refine",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail,
        current_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# goldfive#403 — partial-apply window between _apply_revision and
# _emit_plan_revised must be invisible to readers.
# ---------------------------------------------------------------------------


async def test_apply_revision_no_partial_state_visible_during_cancel_inflight_yield() -> None:
    """During a refine, no observer sees ``session.plan`` with a bumped
    ``revision_index`` AND ``edges`` still equal to the prior edges.

    Pre-#403 fix the flow was::

        _apply_revision:
            set_session_plan(revised)              # FIRST swap, outside lock
            session.last_addressed_revision_by_drift_key[...] = idx
        await _cancel_inflight_for_revision(...)   # yields the event loop
        _emit_plan_revised:                        # lock acquired HERE
            integrated = integrate_correction_supersedes(revised)
            set_session_plan(integrated)           # SECOND swap (rewired DAG)
            _repin_current_task_on_supersedes(...)
            ...

    A reader scheduled during the ``_cancel_inflight_for_revision`` yield
    observed ``session.plan.revision_index = N+1`` with the
    pre-integration edges (no ``t1->t1_corr`` edge yet). The docstring on
    ``_emit_plan_revised`` claimed atomicity (pre- or post-revision
    state, never partial); this test asserts that claim.

    Post-fix all session mutations happen inside ``_emit_plan_revised``'s
    lock — and ``_wait_plan_stable`` (used by report handlers) and a
    raw poll during the yield BOTH see either the pre- or post-revision
    state, never partial.
    """
    prior = _initial_plan()
    session = _make_session(prior)
    session.current_task_id = "t1"
    prior_revision = prior.revision_index

    revised = _supersedes_revised(prior)
    # The signature of the partial-apply window we're trying to catch:
    # session.plan.revision_index has advanced to ``prior_revision + 1``
    # (because _apply_revision pre-#403 already called set_session_plan
    # outside the lock), but the edge DAG still contains ``t1 -> t2``
    # — the pre-integration edge that _integrate_correction_supersedes
    # should rewrite to ``t1_corr -> t2``. If a poll snapshot has the
    # bumped index AND the un-rewired t1->t2 edge, that's the bug.
    un_rewired_edge = ("t1", "t2")

    planner = _StubPlanner(revised=revised)
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    drift = _drift(kind=DriftKind.TOOL_ERROR, task_id="t1")

    # Force ``_cancel_inflight_for_revision`` to yield the event loop
    # for long enough that the polling reader scheduled below can
    # interleave and observe whatever ``session.plan`` looks like at
    # that moment. Pre-#403 this is the partial-apply window; post-#403
    # the lock in ``_emit_plan_revised`` is held across this yield so
    # the reader cannot enter the critical section until the install
    # completes (it can still snapshot ``session.plan`` raw, which is
    # either the prior pointer or the fully-integrated revised
    # pointer).
    original_cancel = steerer.drift._cancel_inflight_for_revision

    async def slow_cancel(d: DriftEvent, s: Session) -> list[str]:
        await asyncio.sleep(0.05)
        return await original_cancel(d, s)

    steerer.drift._cancel_inflight_for_revision = slow_cancel  # type: ignore[method-assign]

    partial_snapshots: list[dict[str, Any]] = []

    async def partial_state_poller() -> None:
        """Poll ``session.plan`` repeatedly across the refine and flag
        any frame where ``revision_index`` has advanced AND the
        un-rewired ``t1 -> t2`` edge is still present. That's the
        partial-apply window we're regressing against — pre-#403
        ``_apply_revision`` swapped the LLM-emitted plan onto
        ``session.plan`` (with the pre-integration edge shape) and
        the integration / rewire didn't run until much later, inside
        ``_emit_plan_revised``'s lock — across the
        ``_cancel_inflight_for_revision`` yield.
        """
        deadline = asyncio.get_event_loop().time() + 0.2
        while asyncio.get_event_loop().time() < deadline:
            plan = session.plan
            if plan is not None:
                rev = plan.revision_index
                edges = tuple(
                    (e.from_task_id, e.to_task_id) for e in plan.edges
                )
                if rev > prior_revision and un_rewired_edge in edges:
                    # PARTIAL APPLY DETECTED — record it; the test will
                    # assert there were zero such frames.
                    partial_snapshots.append(
                        {
                            "revision_index": rev,
                            "edges": edges,
                            "current_task_id": session.current_task_id,
                        }
                    )
            await asyncio.sleep(0)  # yield to the refine task

    async def refine_driver() -> None:
        await steerer.drift.handle_drift(drift, session)

    await asyncio.gather(refine_driver(), partial_state_poller())

    # Post-condition sanity: the refine actually landed (otherwise the
    # poller would trivially observe no partial state because there was
    # no install at all).
    assert session.plan is not None
    assert session.plan.revision_index == prior_revision + 1, (
        "the refine must have installed the revision for this test to "
        "exercise the install pipeline; got "
        f"revision_index={session.plan.revision_index}"
    )
    post_edges = tuple(
        (e.from_task_id, e.to_task_id) for e in session.plan.edges
    )
    # Post-integration the rewired DAG flows t1 -> t1_corr -> t2 -> t3
    # (t1->t2 was rewritten to t1_corr->t2 by
    # _integrate_correction_supersedes).
    assert ("t1", "t1_corr") in post_edges, (
        "the rewired DAG must include t1->t1_corr post-install; got "
        f"edges={post_edges}"
    )
    assert ("t1_corr", "t2") in post_edges, (
        "the rewired DAG must include t1_corr->t2 (integrate rewrites "
        f"t1->t2); got edges={post_edges}"
    )
    assert ("t1", "t2") not in post_edges, (
        "the un-rewired t1->t2 edge must NOT be present post-install — "
        f"that's the partial-apply signature; got edges={post_edges}"
    )

    # The actual regression assertion: zero partial-apply frames seen.
    # Pre-#403 the poller would observe one or more frames with
    # revision_index=prior+1 AND edges==prior_edges (the first
    # set_session_plan inside _apply_revision wrote the bumped index
    # but with the un-rewired DAG).
    assert partial_snapshots == [], (
        "goldfive#403 regression: observed partial-apply window where "
        "session.plan.revision_index advanced but edges still match "
        f"prior. Snapshots: {partial_snapshots}"
    )


async def test_apply_revision_does_not_mutate_session_before_emit() -> None:
    """``_apply_revision`` returns a stamped Plan without mutating session state.

    Unit-level companion to the racy regression above: assert the
    contract change directly. Pre-#403 ``_apply_revision`` wrote
    ``session.plan``, ``session.last_addressed_revision_by_drift_key``,
    and ``session.state`` (orchestration current-plan pointer) before
    returning. Post-#403 all three live inside the
    ``_emit_plan_revised`` lock.
    """
    prior = _initial_plan()
    session = _make_session(prior)
    revised = _supersedes_revised(prior)

    drift = _drift(kind=DriftKind.OFF_TOPIC, task_id="t1")

    steerer = DefaultSteerer()
    returned, was_installed = steerer.plans._apply_revision(session, revised, drift)

    # The decision is "install" (no observation-only carve-out fires).
    assert was_installed is True, (
        "TOOL_ERROR/OFF_TOPIC drift in active-steering mode must decide "
        "to install — observation_only is False by default"
    )
    # The stamped plan reflects the bumped revision metadata.
    assert returned.revision_index > prior.revision_index, (
        "_apply_revision must stamp revision_index on the returned Plan "
        "even though the session pointer is unchanged"
    )
    # But the session has NOT yet been mutated. This is the post-#403
    # contract: _apply_revision computes, _emit_plan_revised installs.
    assert session.plan is prior, (
        "goldfive#403: _apply_revision must NOT swap session.plan — "
        "the install moved into _emit_plan_revised's lock"
    )
    assert session.last_addressed_revision_by_drift_key == {}, (
        "goldfive#403: _apply_revision must NOT stamp the per-(kind, "
        "target) watermark — the stamp moved into _emit_plan_revised's "
        "lock so it never appears without the matching session.plan swap"
    )
