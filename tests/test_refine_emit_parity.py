"""Parity tests: refine observability events fire from every refine entry point.

Bugs A + B from goldfive G3 (2026-04-24 E2E findings):

* Bug A — :func:`goldfive.planner._check_supersedes_coverage` emits
  ``refine_orphaned_tasks`` to the wire when refine drops prior tasks
  without a supersedes link. Today's E2E found 5 dropped tasks but 0
  events on the sink because the validator's emit hook depends on the
  steerer's ``_active_session`` plumbing, and the
  :class:`~goldfive.executors.parallel.ParallelDAGExecutor` refine
  pathway bypasses the steerer entirely.

* Bug B — ``refine_attempted`` / ``refine_failed`` (PR #264) hook into
  :class:`~goldfive.steerer.DefaultSteerer._handle_drift` and
  :meth:`_promote_drift_to_steer`. The parallel executor's direct
  ``planner.refine`` call site emitted neither.

Both fixes route through a new
:meth:`~goldfive.steerer.DefaultSteerer.observe_refine` async context
manager that mints the attempt id, plumbs ``_active_session`` so the
planner-side orphan emit fires, emits ``refine_attempted`` on enter,
and ``refine_failed`` on exception. The parallel executor opts in via
its ``steerer=`` kwarg.

Tests below assert all three events land regardless of which dispatch
path triggered the refine.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.executors.parallel import ParallelDAGExecutor  # noqa: E402
from goldfive.planner import LLMPlanner  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
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
# Test doubles
# ---------------------------------------------------------------------------


class ListSink:
    """Sink that records every emitted event (proto + dict)."""

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
        return [e for e in self.dict_events if e.get("kind") == kind]


class _ScriptedLLM:
    """``call_llm`` stub: returns the next scripted response per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# observe_refine — context manager contract
# ---------------------------------------------------------------------------


async def test_observe_refine_emits_attempted_on_enter() -> None:
    """Entering observe_refine emits exactly one refine_attempted with a
    fresh attempt_id; ``_active_session`` is plumbed for the duration."""
    sink = ListSink()
    steerer = DefaultSteerer()

    class _StubPlanner:
        async def generate(self, **_: Any) -> Plan | None:
            return None

        async def refine(self, **_: Any) -> Plan | None:
            return None

        # ``set_drift_emitter`` / ``set_span_context_provider`` are
        # required by DefaultSteerer.bind() — duck-typed.
        def set_drift_emitter(self, emitter: Any) -> None:
            pass

        def set_span_context_provider(self, provider: Any) -> None:
            self.provider = provider

    planner = _StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="g")])
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="d",
        current_task_id="t1",
    )

    async with steerer.observe_refine(session, drift) as attempt_id:
        # Inside the context the planner's span-ctx provider resolves
        # to a non-None tuple, so the planner-side _emit_refine_orphaned_tasks
        # would route to the bound sinks.
        ctx = planner.provider()
        assert ctx is not None
        sinks_arg, run_id, session_id, _task_id, _seq_fn = ctx
        assert sink in sinks_arg
        assert run_id == "r1"
        assert session_id == session.id

    attempted = sink.by_kind("refine_attempted")
    assert len(attempted) == 1
    assert attempted[0]["payload"]["attempt_id"] == attempt_id
    # No failure on clean exit.
    assert sink.by_kind("refine_failed") == []
    # _active_session cleared in finally.
    assert planner.provider() is None


async def test_observe_refine_emits_failed_on_exception_and_reraises() -> None:
    """An exception inside the with-block emits one refine_failed
    stamped with the same attempt_id, then re-raises so callers' error
    paths still run."""
    sink = ListSink()
    steerer = DefaultSteerer()

    class _StubPlanner:
        async def generate(self, **_: Any) -> Plan | None:
            return None

        async def refine(self, **_: Any) -> Plan | None:
            return None

        def set_drift_emitter(self, emitter: Any) -> None:
            pass

        def set_span_context_provider(self, provider: Any) -> None:
            pass

    steerer.bind(sinks=[sink], planner=_StubPlanner())
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="g")])
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="loop",
        current_task_id="t1",
    )

    captured_attempt_id: str = ""
    with pytest.raises(RuntimeError, match="boom"):
        async with steerer.observe_refine(session, drift) as attempt_id:
            captured_attempt_id = attempt_id
            raise RuntimeError("boom")

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    assert len(attempted) == 1
    assert len(failed) == 1
    assert attempted[0]["payload"]["attempt_id"] == captured_attempt_id
    assert failed[0]["payload"]["attempt_id"] == captured_attempt_id
    assert failed[0]["payload"]["failure_kind"] == "llm_error"
    assert "boom" in failed[0]["payload"]["reason"]


# ---------------------------------------------------------------------------
# Bug A — orphan validator emits via observe_refine span ctx
# ---------------------------------------------------------------------------


async def test_planner_orphan_emit_fires_when_steerer_observes_refine() -> None:
    """Inside observe_refine, the planner-side _emit_refine_orphaned_tasks
    resolves a sink target via the steerer's bound span-context provider
    — so an orphan-producing refine produces the orphan event on the wire.

    Mirrors the live-demo scenario where a PENDING task is silently
    dropped from the revised plan (no supersedes link, not terminal),
    today's E2E found 0 events; this asserts the post-fix path emits."""
    prior = Plan(
        id="p-prior",
        run_id="r-orphan",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="research",
                title="Research",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft_intro",
                title="Draft intro",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="draft_body",
                title="Draft body",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft_intro"),
            TaskEdge(from_task_id="draft_intro", to_task_id="draft_body"),
        ],
        revision_index=0,
    )
    # LLM silently drops draft_body — no supersedes, not terminal.
    refine_response = json.dumps(
        {
            "summary": "narrowed",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_intro",
                    "title": "Draft intro",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft_intro"},
            ],
        }
    )
    scripted = _ScriptedLLM([refine_response])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(
        run_id="r-orphan", goals=[Goal(id="g1", summary="g")], plan=prior
    )
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.WARNING,
        detail="scope narrowed",
        current_task_id="draft_body",
    )

    async with steerer.observe_refine(session, drift) as _attempt_id:
        revised = await planner.refine(plan=prior, drift=drift, goals=list(session.goals))

    assert revised is not None
    orphan_events = sink.by_kind("refine_orphaned_tasks")
    assert len(orphan_events) == 1
    payload = orphan_events[0]["payload"]
    assert payload["orphan_count"] == 1
    assert payload["orphans"][0]["task_id"] == "draft_body"


async def test_planner_no_orphan_emit_when_coverage_complete() -> None:
    """Inverse: a refine with full supersedes coverage emits NO
    orphan event even though observe_refine plumbed the ctx."""
    prior = Plan(
        id="p-clean",
        run_id="r-clean",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="research",
                title="Research",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="research", to_task_id="draft")],
        revision_index=0,
    )
    refine_response = json.dumps(
        {
            "summary": "redirect",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_v2",
                    "title": "Draft v2",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                    "supersedes": "draft",
                    "supersedes_kind": "REPLACE",
                },
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "draft_v2"}],
        }
    )
    scripted = _ScriptedLLM([refine_response])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(
        run_id="r-clean", goals=[Goal(id="g1", summary="g")], plan=prior
    )
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.WARNING,
        detail="redirect",
        current_task_id="draft",
    )

    async with steerer.observe_refine(session, drift) as _attempt_id:
        revised = await planner.refine(plan=prior, drift=drift, goals=list(session.goals))

    assert revised is not None
    assert sink.by_kind("refine_orphaned_tasks") == []


# ---------------------------------------------------------------------------
# Bug B — Parallel executor refine path emits attempted/failed
# ---------------------------------------------------------------------------


def _diamond_plan() -> Plan:
    """A -> {B, C} -> D. Smaller than the canonical 5-task diamond
    because we don't need wide concurrency for these tests."""
    return Plan(
        id="plan-pe",
        run_id="run-pe",
        goal_ids=["g1"],
        tasks=[
            Task(id="A", title="A"),
            Task(id="B", title="B"),
            Task(id="C", title="C"),
            Task(id="D", title="D"),
        ],
        edges=[
            TaskEdge("A", "B"),
            TaskEdge("A", "C"),
            TaskEdge("B", "D"),
            TaskEdge("C", "D"),
        ],
    )


def _refined_diamond() -> Plan:
    """Valid refine output for the diamond. revision_index=1 so the
    structural validator (for_revision=True) accepts the swap."""
    return Plan(
        id="plan-pe",
        run_id="run-pe",
        goal_ids=["g1"],
        tasks=[
            Task(id="A", title="A", status=TaskStatus.COMPLETED),
            Task(id="B", title="B"),
            Task(id="C", title="C"),
            Task(id="D", title="D"),
        ],
        edges=[
            TaskEdge("A", "B"),
            TaskEdge("A", "C"),
            TaskEdge("B", "D"),
            TaskEdge("C", "D"),
        ],
        revision_index=1,
    )


class _DriftOnceAdapter:
    """Adapter that surfaces a drift on the first invocation of a
    designated task, then completes normally on subsequent runs.

    The drift hop is delivered via ``InvocationResult.raw['_drift_to_surface']``
    — recognised by :class:`_FixedDriftSteerer` below.
    """

    def __init__(self, *, drift_task: str, drift: DriftEvent) -> None:
        self._drift_task = drift_task
        self._drift = drift
        self._fired = False

    @property
    def available_agents(self) -> list[str]:
        return ["default"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:
        return None

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        drift: DriftEvent | None = None
        if task.id == self._drift_task and not self._fired:
            self._fired = True
            drift = self._drift
        return InvocationResult(
            task_id=task.id,
            text=f"r:{task.id}",
            raw={"_drift_to_surface": drift},
        )


class _FixedDriftSteerer(DefaultSteerer):
    """DefaultSteerer subclass that surfaces a drift attached by
    :class:`_DriftOnceAdapter` to the InvocationResult — so the parallel
    executor's drift gate fires deterministically without depending on
    the classifier heuristics. All other behaviour (the
    :meth:`observe_refine` plumbing under test) is inherited from the
    real DefaultSteerer.
    """

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:
        if isinstance(event, InvocationResult):
            raw = event.raw or {}
            if isinstance(raw, dict):
                drift = raw.get("_drift_to_surface")
                if isinstance(drift, DriftEvent):
                    return drift
        return super().detect_drift(event, session)


class _ScriptedRefinePlanner:
    """Planner that returns a scripted plan (or raises / returns None)
    when ``refine`` is called. Implements the optional contract surfaces
    (``set_drift_emitter`` / ``set_span_context_provider``) the steerer
    duck-types on."""

    def __init__(
        self,
        *,
        revised: Plan | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._revised = revised
        self._raise = raise_exc
        self.refine_calls: list[DriftEvent] = []
        self.span_ctx_provider: Any | None = None

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: Any,
        context: Any = None,
    ) -> Plan | None:
        return None

    async def refine(self, *, plan: Plan, drift: DriftEvent, goals: list[Goal]) -> Plan | None:
        self.refine_calls.append(drift)
        if self._raise is not None:
            raise self._raise
        return self._revised

    def set_drift_emitter(self, emitter: Any) -> None:
        pass

    def set_span_context_provider(self, provider: Any) -> None:
        self.span_ctx_provider = provider


def _drift(*, task_id: str = "B") -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="boom",
        current_task_id=task_id,
    )


async def test_parallel_refine_success_emits_attempted_and_correlation() -> None:
    """End-to-end through ParallelDAGExecutor: a successful refine
    produces ``refine_attempted`` + correlation ``plan_revised`` dict
    events with the same attempt_id, AND no ``refine_failed``.
    """
    drift = _drift(task_id="B")
    adapter = _DriftOnceAdapter(drift_task="B", drift=drift)
    planner = _ScriptedRefinePlanner(revised=_refined_diamond())
    sink = ListSink()
    steerer = _FixedDriftSteerer()
    executor = ParallelDAGExecutor(max_concurrency=0, drift_policy="finish_stage")
    session = Session(run_id="run-pe", goals=[Goal(id="g1", summary="g")])

    await executor.run(
        plan=_diamond_plan(),
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    revised_dicts = sink.by_kind("plan_revised")
    assert len(attempted) >= 1, (
        "expected at least one refine_attempted event from the parallel executor"
    )
    assert len(failed) == 0, "no refine_failed on success"
    assert len(revised_dicts) >= 1, "expected at least one correlation sidecar"
    # Pair the first attempted with its correlation by attempt_id.
    aid_attempted = attempted[0]["payload"]["attempt_id"]
    matched = [d for d in revised_dicts if d["payload"]["attempt_id"] == aid_attempted]
    assert matched, "expected correlation event with matching attempt_id"


async def test_parallel_refine_raise_emits_attempted_and_failed() -> None:
    """When ``planner.refine`` raises, the parallel executor emits
    ``refine_attempted`` + ``refine_failed`` (failure_kind=llm_error)
    with the same attempt_id, AND no correlation ``plan_revised``."""
    drift = _drift(task_id="B")
    adapter = _DriftOnceAdapter(drift_task="B", drift=drift)
    planner = _ScriptedRefinePlanner(raise_exc=RuntimeError("planner boom"))
    sink = ListSink()
    steerer = _FixedDriftSteerer()
    executor = ParallelDAGExecutor(max_concurrency=0, drift_policy="finish_stage")
    session = Session(run_id="run-pe", goals=[Goal(id="g1", summary="g")])

    await executor.run(
        plan=_diamond_plan(),
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    revised_dicts = sink.by_kind("plan_revised")
    assert len(attempted) >= 1
    assert len(failed) >= 1
    assert len(revised_dicts) == 0
    # Pair attempted/failed by attempt_id.
    aid_attempted = attempted[0]["payload"]["attempt_id"]
    matched_failed = [f for f in failed if f["payload"]["attempt_id"] == aid_attempted]
    assert matched_failed, "expected refine_failed paired with refine_attempted"
    assert matched_failed[0]["payload"]["failure_kind"] == "llm_error"
    assert "planner boom" in matched_failed[0]["payload"]["reason"]


async def test_parallel_refine_returns_none_emits_failed_with_parse_error() -> None:
    """``planner.refine`` returning None is treated as ``parse_error``
    failure_kind via the parallel executor's _refine path."""
    drift = _drift(task_id="B")
    adapter = _DriftOnceAdapter(drift_task="B", drift=drift)
    planner = _ScriptedRefinePlanner(revised=None)
    sink = ListSink()
    steerer = _FixedDriftSteerer()
    executor = ParallelDAGExecutor(max_concurrency=0, drift_policy="finish_stage")
    session = Session(run_id="run-pe", goals=[Goal(id="g1", summary="g")])

    await executor.run(
        plan=_diamond_plan(),
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    assert len(attempted) >= 1
    assert len(failed) >= 1
    aid_attempted = attempted[0]["payload"]["attempt_id"]
    matched = [f for f in failed if f["payload"]["attempt_id"] == aid_attempted]
    assert matched, "expected refine_failed paired with refine_attempted"
    assert matched[0]["payload"]["failure_kind"] == "parse_error"


async def test_parallel_refine_returns_invalid_plan_emits_failed_with_validator_kind() -> None:
    """Validator rejection produces ``failure_kind=validator_rejected``."""
    bad_plan = Plan(
        id="plan-pe",
        run_id="run-pe",
        goal_ids=[],
        tasks=[Task(id="A", title="A", status=TaskStatus.COMPLETED)],
        # Edge points at an id missing from tasks → validator rejects.
        edges=[TaskEdge(from_task_id="A", to_task_id="missing")],
        revision_index=1,
    )
    drift = _drift(task_id="B")
    adapter = _DriftOnceAdapter(drift_task="B", drift=drift)
    planner = _ScriptedRefinePlanner(revised=bad_plan)
    sink = ListSink()
    steerer = _FixedDriftSteerer()
    executor = ParallelDAGExecutor(max_concurrency=0, drift_policy="finish_stage")
    session = Session(run_id="run-pe", goals=[Goal(id="g1", summary="g")])

    await executor.run(
        plan=_diamond_plan(),
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    assert len(attempted) >= 1
    assert len(failed) >= 1
    aid = attempted[0]["payload"]["attempt_id"]
    matched = [f for f in failed if f["payload"]["attempt_id"] == aid]
    assert matched
    assert matched[0]["payload"]["failure_kind"] == "validator_rejected"


# ---------------------------------------------------------------------------
# Cross-path: every refine entry point emits the same event family
# ---------------------------------------------------------------------------


async def test_refine_attempted_emitted_from_both_steerer_and_executor_paths() -> None:
    """End-to-end equivalence: whether the refine fires through
    :meth:`DefaultSteerer._handle_drift` (steerer-driven) or via
    :meth:`ParallelDAGExecutor._refine` (executor-driven), a
    ``refine_attempted`` event lands on the wire.

    Today's E2E found 0 such events from the executor path; this test
    is the regression guard."""
    # --- Steerer-driven path -----------------------------------------
    revised_plan = Plan(
        id="p-st",
        run_id="r-st",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge("t1", "t2")],
        revision_index=1,
    )
    sink_st = ListSink()
    steerer_st = DefaultSteerer()
    planner_st = _ScriptedRefinePlanner(revised=revised_plan)
    steerer_st.bind(sinks=[sink_st], planner=planner_st)
    session_st = Session(
        run_id="r-st",
        goals=[Goal(id="g1", summary="g")],
        plan=Plan(
            id="p-st",
            run_id="r-st",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
            edges=[TaskEdge("t1", "t2")],
        ),
    )
    await steerer_st._handle_drift(_drift(task_id="t1"), session_st)
    assert sink_st.by_kind("refine_attempted"), (
        "steerer-driven refine emits refine_attempted"
    )

    # --- Executor-driven path -----------------------------------------
    drift_ex = _drift(task_id="B")
    adapter_ex = _DriftOnceAdapter(drift_task="B", drift=drift_ex)
    planner_ex = _ScriptedRefinePlanner(revised=_refined_diamond())
    sink_ex = ListSink()
    steerer_ex = _FixedDriftSteerer()
    executor = ParallelDAGExecutor(max_concurrency=0, drift_policy="finish_stage")
    session_ex = Session(run_id="run-pe", goals=[Goal(id="g1", summary="g")])

    await executor.run(
        plan=_diamond_plan(),
        session=session_ex,
        adapter=adapter_ex,
        steerer=steerer_ex,
        planner=planner_ex,
        sinks=[sink_ex],
    )
    assert sink_ex.by_kind("refine_attempted"), (
        "executor-driven refine emits refine_attempted"
    )
