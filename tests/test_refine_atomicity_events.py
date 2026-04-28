"""Tests for goldfive a4: refine atomicity barrier + RefineAttempted/Failed events.

Two coordinated changes exercised here, both touching
:meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised`:

1. **Atomicity barrier.** A per-session :class:`asyncio.Lock` serialises
   the consistency-critical region of plan mutation (revision_index
   bump → supersedes integration → repin → PlanRevised emit).
   :meth:`DefaultSteerer._wait_plan_stable` lets reports acquire +
   immediately release the lock so they observe either pre-revision or
   post-revision state, never a partial apply. Fixes the race between
   fire-and-forget judge-triggered refines (#254) and imperative
   ``report_task_*`` handlers.

2. **Refine observability events.** Three dict-envelope events (paired
   by ``attempt_id``):

   * ``refine_attempted`` — emitted at refine start.
   * ``refine_failed`` — emitted on parse / validator / LLM failure.
     NO ``revision_index`` bump.
   * ``plan_revised`` — proto stays unchanged; a sidecar dict envelope
     stamped with ``attempt_id`` is emitted alongside for correlation.

The dict-event surface is intentionally lightweight — promote to proto
when the Stream C (#256) follow-up gets prioritised.
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
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """Records every emitted event (proto + dict envelopes)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    @property
    def proto_events(self) -> list[Any]:
        return [e for e in self.events if hasattr(e, "WhichOneof")]

    @property
    def dict_events(self) -> list[dict[str, Any]]:
        return [e for e in self.events if isinstance(e, dict)]

    def by_kind(self, kind: str) -> list[dict[str, Any]]:
        """Return dict events whose ``kind`` matches."""
        return [e for e in self.dict_events if e.get("kind") == kind]


class StubPlanner:
    """Planner whose ``refine`` is configurable per call (raise / None / plan).

    Each successful ``refine`` call returns a *fresh* plan instance with
    a per-call unique sentinel task appended so the steerer's no-op
    revision rejection (goldfive#271) sees a real structural diff. The
    caller-supplied ``revised`` template is preserved verbatim apart
    from the appended sentinel.
    """

    def __init__(
        self,
        *,
        revised: Plan | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.revised = revised
        self.raise_exc = raise_exc
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
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.revised is None:
            return None
        # Fresh structural mutation per call so successive refines don't
        # appear as no-op revisions to the steerer's structural check.
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
            revision_reason=self.revised.revision_reason,
            revision_kind=self.revised.revision_kind,
            revision_severity=self.revised.revision_severity,
            revision_trigger_event_id=self.revised.revision_trigger_event_id,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(task_ids: tuple[str, ...] = ("t1", "t2")) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=tid, title=tid.upper()) for tid in task_ids],
        edges=[
            TaskEdge(from_task_id=task_ids[i], to_task_id=task_ids[i + 1])
            for i in range(len(task_ids) - 1)
        ],
    )


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r-a4",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan if plan is not None else _make_plan(),
    )


def _drift(
    *,
    kind: DriftKind = DriftKind.TOOL_ERROR,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    detail: str = "drift",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail,
        current_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Observability — refine_attempted / refine_failed / correlation plan_revised
# ---------------------------------------------------------------------------


async def test_successful_refine_emits_attempted_and_correlation_with_same_attempt_id() -> None:
    """A successful refine produces:

      * one ``refine_attempted`` dict event,
      * one proto ``PlanRevised`` event,
      * one correlation ``plan_revised`` dict event,
      * NO ``refine_failed`` event,
      * the correlation event carries the same ``attempt_id`` as the
        ``refine_attempted`` event.
    """
    # Structurally-distinct revised plan so the steerer's no-op
    # revision rejection (goldfive#271) doesn't short-circuit refine
    # success — this test exercises the success path.
    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer._handle_drift(_drift(), session)

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    revised_dicts = sink.by_kind("plan_revised")
    proto_revised = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]

    assert len(attempted) == 1, "exactly one refine_attempted on success"
    assert len(failed) == 0, "no refine_failed on success"
    assert len(revised_dicts) == 1, "one correlation sidecar on success"
    assert len(proto_revised) == 1, "one proto PlanRevised on success"

    # attempt_id correlates the pair.
    aid_attempted = attempted[0]["payload"]["attempt_id"]
    aid_correlation = revised_dicts[0]["payload"]["attempt_id"]
    assert aid_attempted
    assert aid_attempted == aid_correlation
    # The correlation event also carries the revision_index from the
    # actually-installed plan.
    assert revised_dicts[0]["payload"]["revision_index"] == revised.revision_index


async def test_planner_exception_emits_attempted_and_failed_no_plan_revised() -> None:
    """When ``planner.refine`` raises, the steerer emits:

      * one ``refine_attempted``
      * one ``refine_failed`` with ``failure_kind == "llm_error"``
      * NO proto ``PlanRevised`` and NO correlation ``plan_revised``
      * NO ``revision_index`` bump

    Both dict events carry the same ``attempt_id``.
    """
    planner = StubPlanner(raise_exc=RuntimeError("planner boom"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    initial_revision = session.plan.revision_index

    await steerer._handle_drift(_drift(), session)

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    proto_revised = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]
    correlation_revised = sink.by_kind("plan_revised")

    assert len(attempted) == 1
    assert len(failed) == 1
    assert len(proto_revised) == 0
    assert len(correlation_revised) == 0

    assert attempted[0]["payload"]["attempt_id"] == failed[0]["payload"]["attempt_id"]
    assert failed[0]["payload"]["failure_kind"] == "llm_error"
    assert "planner boom" in failed[0]["payload"]["reason"]
    # Plan untouched: no revision bump on failure.
    assert session.plan.revision_index == initial_revision


async def test_planner_returns_none_emits_failed_with_parse_error_kind() -> None:
    """``revised is None`` is treated as a ``parse_error`` failure_kind."""
    planner = StubPlanner(revised=None)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer._handle_drift(_drift(), session)

    failed = sink.by_kind("refine_failed")
    assert len(failed) == 1
    assert failed[0]["payload"]["failure_kind"] == "parse_error"
    assert "no revised plan" in failed[0]["payload"]["reason"]


async def test_validator_rejection_emits_failed_with_validator_kind() -> None:
    """A revised plan that fails ``Plan.validate`` produces
    ``failure_kind == "validator_rejected"``.
    """
    # Revised plan with a duplicate task id triggers Plan.validate to raise.
    bad_revised = Plan(
        id="p1",
        run_id="r-a4",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", status=TaskStatus.PENDING),
            Task(id="t1", title="A-DUP", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer._handle_drift(_drift(), session)

    failed = sink.by_kind("refine_failed")
    assert len(failed) == 1
    assert failed[0]["payload"]["failure_kind"] == "validator_rejected"
    assert "validation failed" in failed[0]["payload"]["reason"]


async def test_attempt_id_is_unique_per_attempt() -> None:
    """Two consecutive refines mint two distinct ``attempt_id`` values."""
    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    # Disable cooldown gate so back-to-back drifts both refine.
    steerer = DefaultSteerer(plan_revision_cooldown_seconds=0.0)
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer._handle_drift(_drift(kind=DriftKind.TOOL_ERROR), session)
    # Use a different drift kind so the cooldown / counter machinery
    # doesn't gate the second attempt.
    await steerer._handle_drift(_drift(kind=DriftKind.LOOPING_REASONING), session)

    attempted = sink.by_kind("refine_attempted")
    assert len(attempted) == 2
    aids = {a["payload"]["attempt_id"] for a in attempted}
    assert len(aids) == 2, "attempt_id must be unique per attempt"


async def test_refine_attempted_carries_drift_kind_and_severity() -> None:
    """Sanity check: the ``refine_attempted`` payload exposes the
    triggering drift's kind / severity / drift_id so harmonograf can
    render an intervention timeline without re-fetching the drift.
    """
    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    drift = _drift(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
    )
    await steerer._handle_drift(drift, session)

    attempted = sink.by_kind("refine_attempted")
    assert len(attempted) == 1
    payload = attempted[0]["payload"]
    assert payload["trigger_kind"] == DriftKind.LOOPING_REASONING.value
    assert payload["trigger_severity"] == DriftSeverity.WARNING.value
    assert payload["drift_id"] == drift.id


# ---------------------------------------------------------------------------
# Atomicity barrier — _wait_plan_stable
# ---------------------------------------------------------------------------


async def test_wait_plan_stable_returns_immediately_when_unlocked() -> None:
    """Calling ``_wait_plan_stable`` outside any refine returns ASAP."""
    steerer = DefaultSteerer()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    # Fast-path: lock not even instantiated yet.
    ok = await steerer._wait_plan_stable(session, timeout=0.5)
    assert ok is True


async def test_wait_plan_stable_blocks_until_emit_plan_revised_completes() -> None:
    """A concurrent ``_wait_plan_stable`` blocks while
    ``_emit_plan_revised`` is mid-mutation, then returns once the
    mutation region releases. The barrier observes a stable plan: the
    revised one (post-mutation), never a partial state.
    """
    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # Hold the per-session lock manually to simulate an in-flight mutation.
    lock = steerer._get_plan_lock(session)
    initial_revision = session.plan.revision_index
    barrier_observed: list[int] = []

    async def reader() -> None:
        # Wait for stability, then snapshot what the read sees.
        await steerer._wait_plan_stable(session, timeout=2.0)
        assert session.plan is not None
        barrier_observed.append(session.plan.revision_index)

    async with lock:
        # Start the reader: it MUST block while we hold the lock.
        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        assert not reader_task.done(), "reader should be blocked on the plan lock"
        # Mutate the plan in-place to simulate the part of
        # _emit_plan_revised that runs inside the lock.
        session.plan = revised
    # Lock released — reader wakes up.
    await asyncio.wait_for(reader_task, timeout=1.0)
    assert barrier_observed == [
        revised.revision_index
    ], f"reader saw partial state: {barrier_observed} vs {revised.revision_index}"
    assert session.plan.revision_index != initial_revision


async def test_concurrent_emit_plan_revised_and_report_observe_consistent_state() -> None:
    """Atomicity test (per the PR spec):

    Two coroutines race — a refine that mutates the plan, and a
    "report" coroutine that calls ``_wait_plan_stable`` before reading
    plan state. The report MUST observe either the pre-revision or
    post-revision plan, never a half-applied mix.

    The mutation region also re-pins ``current_task_id`` via
    ``_repin_current_task_on_supersedes``; we use the supersedes path
    so the partial-vs-complete distinction has an observable signal:
    the report's plan_revision_index must match its current_task_id
    pin (either both pre or both post).
    """
    initial_plan = _make_plan(("t1", "t2"))
    session = _make_session(initial_plan)
    session.current_task_id = "t1"
    initial_revision = session.plan.revision_index

    # Build a revised plan that supersedes t1 with t1_corr.
    revised = Plan(
        id="p1",
        run_id="r-a4",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.FAILED),
            Task(
                id="t1_corr",
                title="T1 corrected",
                status=TaskStatus.PENDING,
                supersedes="t1",
            ),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t1_corr"),
            TaskEdge(from_task_id="t1_corr", to_task_id="t2"),
        ],
        revision_index=initial_revision + 1,
    )

    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    drift = _drift(kind=DriftKind.TOOL_ERROR, task_id="t1")

    snapshots: list[tuple[int, str]] = []

    async def reporter() -> None:
        # Slight delay so the refine path enters the lock first.
        await asyncio.sleep(0.001)
        await steerer._wait_plan_stable(session, timeout=2.0)
        # Snapshot plan revision_index + current_task_id together. By
        # contract these must agree: pre-revision (0, "t1") or
        # post-revision (1, "t1_corr").
        assert session.plan is not None
        snapshots.append((session.plan.revision_index, session.current_task_id))

    async def refiner() -> None:
        await steerer._handle_drift(drift, session)

    await asyncio.gather(refiner(), reporter())

    assert len(snapshots) == 1
    rev, pin = snapshots[0]
    # Either both pre-revision or both post-revision — never partial.
    if rev == initial_revision:
        assert pin == "t1", "pre-revision pin must still be t1"
    else:
        assert rev == initial_revision + 1, "post-revision rev_index must equal expected"
        # supersedes integration repinned current_task_id -> t1_corr.
        assert pin == "t1_corr", (
            f"post-revision pin must be t1_corr (the replacement); got {pin!r}"
        )


async def test_wait_plan_stable_times_out_gracefully_when_lock_held() -> None:
    """When the lock is held longer than the timeout, ``_wait_plan_stable``
    proceeds with a best-effort racy read and logs (returns ``False``).
    Atomicity is a soft barrier, not a hard mutex — blocking a report
    indefinitely is worse than a stale read.
    """
    steerer = DefaultSteerer()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    lock = steerer._get_plan_lock(session)
    async with lock:
        # The waiter has a tight timeout; we never release before it.
        ok = await steerer._wait_plan_stable(session, timeout=0.05)
    assert ok is False


# ---------------------------------------------------------------------------
# Reporting integration — handlers call _await_plan_stable
# ---------------------------------------------------------------------------


async def test_report_handler_invokes_wait_plan_stable() -> None:
    """The report_task_completed handler should call the steerer's
    ``_wait_plan_stable`` before reading plan state. Duck-typed: any
    Steerer that exposes the method gets the barrier; custom Steerers
    silently fall through.
    """
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS

    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    session.plan.tasks[0].status = TaskStatus.RUNNING
    session.current_task_id = "t1"

    # Spy on _wait_plan_stable.
    call_log: list[str] = []
    original = steerer._wait_plan_stable

    async def _spy(session_arg: Any, *, timeout: Any = 1.0) -> bool:
        call_log.append("called")
        return await original(session_arg, timeout=timeout)

    steerer._wait_plan_stable = _spy  # type: ignore[method-assign]

    handler = next(
        t.handler for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_completed"
    )
    await handler({"task_id": "t1", "summary": "ok"}, session, steerer)

    assert call_log == ["called"], (
        "report_task_completed must invoke _wait_plan_stable before plan reads"
    )


async def test_report_handler_tolerates_steerer_without_wait_plan_stable() -> None:
    """A custom Steerer that doesn't expose ``_wait_plan_stable`` keeps
    working — the report_task_* handler short-circuits the barrier and
    proceeds with the pre-a4 racy read semantics. No crash.
    """
    from goldfive.reporting import BUILTIN_REPORTING_TOOLS

    class MinimalSteerer:
        """Bare-minimum Steerer stub without a4's helper."""

        async def mark_task_completed(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def mark_task_running(self, *args: Any, **kwargs: Any) -> None:
            pass

    session = _make_session()
    session.plan.tasks[0].status = TaskStatus.RUNNING
    handler = next(
        t.handler for t in BUILTIN_REPORTING_TOOLS if t.name == "report_task_completed"
    )
    out = await handler({"task_id": "t1", "summary": "ok"}, session, MinimalSteerer())
    assert out.get("acknowledged") is True


# ---------------------------------------------------------------------------
# Regression: existing PlanRevised behaviour unchanged
# ---------------------------------------------------------------------------


async def test_proto_plan_revised_event_unchanged_on_success() -> None:
    """Regression check: the proto ``PlanRevised`` carries the same
    fields it always has. The a4 attempt_id sidecar is purely additive;
    existing consumers don't observe any field change on the proto.
    """
    revised = _make_plan(("t1", "t2", "t3"))
    revised.revision_index = 1
    revised.revision_reason = ""
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    drift = _drift(kind=DriftKind.TOOL_ERROR)
    await steerer._handle_drift(drift, session)

    proto_pr = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]
    assert len(proto_pr) == 1
    pr = proto_pr[0].plan_revised
    assert pr.revision_index == revised.revision_index
    # ``trigger_event_id`` non-empty (uses drift.id when no annotation).
    assert pr.trigger_event_id
    # The proto envelope has no attempt_id field — correlation lives on
    # the dict sidecar. This assertion guards against future proto
    # changes promoting attempt_id without explicit migration intent.
    assert not hasattr(pr, "attempt_id") or pr.attempt_id == ""
