"""Outcome-progress WIRING in ledger mode (AGENCY-PRESERVATION.md PR 11c).

Covers the observer/executor wiring of the outcome-progress judge:

* ``DriftObserver.finalize_outcomes`` (run-end): met → COMPLETED,
  CONFIDENTLY-unmet → FAILED, uncertain → stays PENDING (carry forward),
  with ``contributes_to`` stamped onto the named DISCOVERED tasks;
* forecast mode is a no-op (the judge LLM is never called);
* re-entrancy guard: the task-boundary cadence does NOT fire on OUTCOME
  transitions (the "outcome COMPLETED → boundary → no second judge"
  shape);
* single-in-flight guard on the fire-and-forget task-boundary judge;
* the executor's ``_has_live_pending_or_running`` ignores OUTCOME tasks;
* ``_executor_plan_mode`` reads the steerer config for the
  ledger+overlay_mode=False misconfiguration warning.
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

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskKind,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    async def generate(self, *, goals: Any, available_agents: Any, context: Any = None) -> Any:
        return None

    async def refine(self, *, plan: Any, drift: Any, goals: Any) -> Any:
        return None


def _make_steerer(*, plan_mode: str, judge_llm: Any, calls: list[str] | None = None):
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(observation_only=False, plan_mode=plan_mode),
        goal_drift_call_llm=judge_llm,
        goal_drift_model="m",
    )
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    return steerer


def _ledger_session() -> Session:
    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(id="o2", title="Translation delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="writer: drafted summary",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.COMPLETED,
            ),
        ),
        edges=(),
        revision_index=1,
    )
    session = Session(
        run_id="r",
        goals=[Goal(id="g", summary="summarise + translate")],
        plan=plan,
    )
    session.completed_outputs["d1"] = "Full summary of the deck: ...."
    return session


def _judge(payload: str, calls: list[str] | None = None):
    async def llm(system: str, user: str, model: str) -> str:
        if calls is not None:
            calls.append(user)
        return payload

    return llm


# ---------------------------------------------------------------------------
# finalize_outcomes — run-end transitions
# ---------------------------------------------------------------------------


def test_finalize_outcomes_completes_met_and_stamps_contributes_to() -> None:
    payload = (
        '{"outcomes": ['
        '{"task_id": "o1", "assessment": "met", "reason": "summary present", '
        '"contributing_task_ids": ["d1"]},'
        '{"task_id": "o2", "assessment": "pending", "reason": "not yet"}]}'
    )
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge(payload))
    session = _ledger_session()

    asyncio.run(steerer.finalize_outcomes(session))

    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["o1"].status is TaskStatus.COMPLETED  # met → COMPLETED
    assert by_id["o2"].status is TaskStatus.PENDING  # pending → carries forward
    # contributes_to stamped onto the DISCOVERED task that produced o1.
    assert by_id["d1"].contributes_to == "o1"


def test_finalize_outcomes_fails_confidently_unmet() -> None:
    payload = (
        '{"outcomes": [{"task_id": "o2", "assessment": "failed", '
        '"reason": "user cancelled the translation"}]}'
    )
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge(payload))
    session = _ledger_session()

    asyncio.run(steerer.finalize_outcomes(session))

    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["o2"].status is TaskStatus.FAILED  # confident-fail → FAILED at run end
    assert by_id["o1"].status is TaskStatus.PENDING  # untouched (no verdict)


def test_finalize_outcomes_forecast_mode_is_noop() -> None:
    calls: list[str] = []

    async def boom(system: str, user: str, model: str) -> str:
        calls.append(user)
        raise AssertionError("forecast mode must not invoke the outcome judge")

    steerer = _make_steerer(plan_mode="forecast", judge_llm=boom)
    session = _ledger_session()

    asyncio.run(steerer.finalize_outcomes(session))

    assert calls == []  # judge never called
    # No transitions in forecast mode.
    assert all(
        t.status in (TaskStatus.PENDING, TaskStatus.COMPLETED)
        for t in session.plan.tasks
    )
    assert {t.id: t.status for t in session.plan.tasks}["o1"] is TaskStatus.PENDING


def test_finalize_outcomes_no_outcome_tasks_skips_judge() -> None:
    calls: list[str] = []
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge("{}", calls))
    session = Session(
        run_id="r",
        goals=[Goal(id="g", summary="x")],
        plan=Plan(
            id="p",
            run_id="r",
            goal_ids=["g"],
            tasks=(Task(id="f1", title="forecast"),),
            edges=(),
        ),
    )
    asyncio.run(steerer.finalize_outcomes(session))
    assert calls == []  # no OUTCOME tasks → judge short-circuits before the LLM


# ---------------------------------------------------------------------------
# Re-entrancy + single-in-flight guards
# ---------------------------------------------------------------------------


def test_task_boundary_skips_outcome_transitions_no_judge() -> None:
    # The exact loop shape: an OUTCOME task transition must NOT fire the
    # task-boundary cadence (goal-drift OR outcome-progress) — otherwise
    # marking an OUTCOME COMPLETED would re-judge → re-mark → loop.
    calls: list[str] = []
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge("{}", calls))
    session = _ledger_session()
    outcome_task = next(t for t in session.plan.tasks if t.id == "o1")

    async def go() -> None:
        await steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=outcome_task
        )
        # Nothing was spawned: no goal-drift judge, no outcome-progress.
        assert len(steerer._background_judges) == 0
        assert getattr(session, "_outcome_progress_inflight", False) is False

    asyncio.run(go())
    assert calls == []


def test_task_boundary_on_discovered_transition_spawns() -> None:
    # A non-OUTCOME (DISCOVERED) transition DOES fire the cadence.
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge("{}"))
    session = _ledger_session()
    discovered = next(t for t in session.plan.tasks if t.id == "d1")

    async def go() -> None:
        await steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=discovered
        )
        # Background judges were spawned (goal-drift + outcome-progress).
        assert len(steerer._background_judges) >= 1
        # Drain so the test doesn't leak tasks.
        await steerer.drift.drain_session_background_tasks(session_id=session.id)

    asyncio.run(go())


def test_single_in_flight_guard_blocks_second_outcome_judge() -> None:
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge("{}"))
    session = _ledger_session()

    async def go() -> None:
        # Mark a judge already in-flight; a second spawn must no-op.
        session._outcome_progress_inflight = True
        steerer.drift._spawn_outcome_progress_background(session)
        assert len(steerer._background_judges) == 0

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Executor helpers
# ---------------------------------------------------------------------------


def test_has_live_pending_ignores_outcome_tasks() -> None:
    from goldfive.executors.sequential import _has_live_pending_or_running

    # ONLY a PENDING OUTCOME task → not "live work" (would loop the
    # nudge-replay gate forever otherwise).
    only_outcome = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(Task(id="o1", title="deliverable", kind=TaskKind.OUTCOME),),
        edges=(),
    )
    assert _has_live_pending_or_running(only_outcome) is False

    # A PENDING DISCOVERED task IS live work.
    with_discovered = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(
            Task(id="o1", title="deliverable", kind=TaskKind.OUTCOME),
            Task(id="d1", title="agent work", discovered=True, kind=TaskKind.DISCOVERED),
        ),
        edges=(),
    )
    assert _has_live_pending_or_running(with_discovered) is True

    # Forecast plan unaffected: a PENDING forecast task is live.
    forecast = Plan(
        id="p",
        run_id="r",
        goal_ids=["g"],
        tasks=(Task(id="f1", title="forecast task"),),
        edges=(),
    )
    assert _has_live_pending_or_running(forecast) is True


def test_executor_plan_mode_reads_steerer_config() -> None:
    from goldfive.executors.sequential import _executor_plan_mode

    ledger = _make_steerer(plan_mode="ledger", judge_llm=_judge("{}"))
    forecast = _make_steerer(plan_mode="forecast", judge_llm=_judge("{}"))
    assert _executor_plan_mode(ledger) == "ledger"
    assert _executor_plan_mode(forecast) == "forecast"
    # Defensive: a steerer without a typed config → forecast.
    assert _executor_plan_mode(object()) == "forecast"


# ---------------------------------------------------------------------------
# Outcome-verdict freshness gate (same-id false-complete guard).
#
# The judge computes verdicts against a plan snapshot BEFORE an LLM
# round-trip; ``session.plan`` may be swapped by a concurrent USER_STEER
# refine during that round-trip. A verdict is applied only when the live
# task of the same id still carries the snapshot's stability token.
# ---------------------------------------------------------------------------


def _swap_plan(session: Session, new_plan: Plan) -> None:
    from goldfive.types import channel_processor_active, set_session_plan

    with channel_processor_active():
        set_session_plan(session, new_plan)


def test_finalize_outcomes_skips_stale_verdict_after_reused_id_revision() -> None:
    # The judge says the OLD o1 is met, but a refine lands DURING the
    # round-trip that regenerates o1 as a DIFFERENT deliverable under the
    # same id. The stale "met" must NOT false-complete the new o1.
    session = _ledger_session()
    payload = (
        '{"outcomes": [{"task_id": "o1", "assessment": "met", '
        '"reason": "old summary present"}]}'
    )

    async def racing_llm(system: str, user: str, model: str) -> str:
        _swap_plan(
            session,
            Plan(
                id=session.plan.id,
                run_id=session.plan.run_id,
                goal_ids=list(session.plan.goal_ids),
                tasks=(
                    Task(id="o1", title="Entirely different deliverable", kind=TaskKind.OUTCOME),
                    Task(id="o2", title="Translation delivered", kind=TaskKind.OUTCOME),
                ),
                edges=(),
                revision_index=session.plan.revision_index + 1,
            ),
        )
        return payload

    steerer = _make_steerer(plan_mode="ledger", judge_llm=racing_llm)
    asyncio.run(steerer.finalize_outcomes(session))

    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["o1"].status is TaskStatus.PENDING  # stale verdict skipped


def test_finalize_outcomes_skips_verdict_when_outcome_task_removed() -> None:
    # A refine drops o1 entirely during the round-trip; the verdict for a
    # now-absent task is skipped rather than raising / mis-applying.
    session = _ledger_session()
    payload = (
        '{"outcomes": [{"task_id": "o1", "assessment": "met", "reason": "x"},'
        '{"task_id": "o2", "assessment": "met", "reason": "y"}]}'
    )

    async def racing_llm(system: str, user: str, model: str) -> str:
        _swap_plan(
            session,
            Plan(
                id=session.plan.id,
                run_id=session.plan.run_id,
                goal_ids=list(session.plan.goal_ids),
                tasks=(
                    Task(id="o2", title="Translation delivered", kind=TaskKind.OUTCOME),
                ),
                edges=(),
                revision_index=session.plan.revision_index + 1,
            ),
        )
        return payload

    steerer = _make_steerer(plan_mode="ledger", judge_llm=racing_llm)
    asyncio.run(steerer.finalize_outcomes(session))

    by_id = {t.id: t for t in session.plan.tasks}
    assert "o1" not in by_id  # dropped by the refine, not resurrected
    # o2 still matches its snapshot token → its met verdict applies.
    assert by_id["o2"].status is TaskStatus.COMPLETED


def test_finalize_outcomes_applies_verdict_when_token_stable() -> None:
    # Control: an unchanged plan (matching stability token) still applies
    # the met verdict — the freshness gate does not over-reject.
    session = _ledger_session()
    payload = (
        '{"outcomes": [{"task_id": "o1", "assessment": "met", "reason": "present"}]}'
    )
    steerer = _make_steerer(plan_mode="ledger", judge_llm=_judge(payload))
    asyncio.run(steerer.finalize_outcomes(session))

    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["o1"].status is TaskStatus.COMPLETED
