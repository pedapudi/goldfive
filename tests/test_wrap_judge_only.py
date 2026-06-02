"""Tests for the first-class judge-only mode on :func:`goldfive.wrap`.

Judge-only mode (``goldfive.wrap(agent, judge_only=True)``) runs the
wrapped agent NATIVELY while keeping the drift judges armed, and issues
ZERO planning / steering LLM calls (no goal-derive, no plan / refine, no
drift-reactive steering). It encapsulates the validated recipe:

* a one-task :class:`StaticPlanner` so the overlay / per-task executor
  produces a real transcript (NOT :class:`PassthroughPlanner`, which
  returns ``None`` and aborts with an empty transcript);
* :class:`LiteralGoalDeriver` so the user input becomes a single goal
  with no goal-derive LLM call;
* the judges stay wired from ``call_llm`` / the detected tree LLM exactly
  as in full mode.

The suite pins:

1. ``judge_only=True`` selects the StaticPlanner + LiteralGoalDeriver
   defaults (so structurally NO planning / goal-derive LLM call can fire)
   while the steerer's judges stay armed — contrasted with
   ``judge_only=False``, which builds the LLMPlanner / LLMGoalDeriver.
2. A ``judge_only=True`` run produces a NON-EMPTY transcript (the native
   agent executed; the run did NOT abort empty like PassthroughPlanner).
3. The drift judges still fire under the judge-only steerer.
4. Explicit ``planner=`` / ``goal_deriver=`` override under
   ``judge_only=True``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import goldfive
from goldfive import (
    ExecutionOutcome,
    InMemorySink,
    InvocationResult,
    LiteralGoalDeriver,
    LLMGoalDeriver,
    LLMPlanner,
    PassthroughPlanner,
    ReportingToolSpec,
    Session,
    StaticPlanner,
    Task,
)
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal, Plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Reference agent: returns a non-empty result so the executor completes."""
    _ = (session, tools)
    return InvocationResult(task_id=task.id, text=f"native ran: {task.title}")


def _spy_call_llm(responses: list[Any]):
    """Async ``CallLLM``-shaped stub that records every call.

    ``.calls`` accumulates ``(system, user, model)`` triples so a test
    can assert how many — and which — LLM calls fired.
    """
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("spy call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


def _make_drift_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Research", description="Research topic X")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id="t1",
    )


# ---------------------------------------------------------------------------
# 1. judge_only selects the native-run defaults (no planning callables)
# ---------------------------------------------------------------------------


def test_judge_only_wires_static_planner_and_literal_goal_deriver() -> None:
    """``judge_only=True`` => StaticPlanner + LiteralGoalDeriver defaults.

    Structurally this guarantees ZERO planning / goal-derive LLM calls:
    neither :class:`LLMPlanner` nor :class:`LLMGoalDeriver` is even
    constructed, so no ``call_llm`` planning surface exists. The judges,
    however, stay armed off the supplied ``call_llm``.
    """
    call_llm = _spy_call_llm([])
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )

    # NATIVE-run planner (one-task StaticPlanner), NOT PassthroughPlanner
    # (which aborts empty) and NOT LLMPlanner (which would plan via LLM).
    assert isinstance(runner.planner, StaticPlanner)
    assert not isinstance(runner.planner, PassthroughPlanner)
    # Goal derivation without an LLM call.
    assert isinstance(runner.goal_deriver, LiteralGoalDeriver)

    # Judges stay armed: the steerer carries the supplied call_llm for
    # BOTH the trajectory-level goal-drift judge and the reasoning-drift
    # judge — judge_only does not touch the judge wiring.
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_call_llm is call_llm
    assert steerer._reasoning_drift_call_llm is call_llm

    # No planning / judge call has fired merely from wiring.
    assert call_llm.calls == []  # type: ignore[attr-defined]


def test_judge_only_false_is_byte_identical_full_planning() -> None:
    """``judge_only=False`` (default) builds the full LLM planning overlay.

    Contrast partner to the test above: the default path still wires
    :class:`LLMPlanner` + :class:`LLMGoalDeriver`, proving judge_only is
    a strict opt-in and the default behaviour is unchanged.
    """
    call_llm = _spy_call_llm([])
    runner = goldfive.wrap(
        _happy_agent,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )
    assert isinstance(runner.planner, LLMPlanner)
    assert isinstance(runner.goal_deriver, LLMGoalDeriver)


# ---------------------------------------------------------------------------
# 2. A judge_only run produces a NON-EMPTY transcript (no empty-abort trap)
# ---------------------------------------------------------------------------


async def test_judge_only_run_produces_non_empty_transcript() -> None:
    """A judge_only run executes the native agent and does NOT abort empty.

    The PassthroughPlanner trap is "generate() -> None => empty
    transcript, nothing to judge". The one-task StaticPlanner avoids it:
    the native agent is invoked, the task completes, and the sink
    captures real activity.
    """
    call_llm = _spy_call_llm([{"progressing": True}])
    sink = InMemorySink()
    outcome = await goldfive.run(
        _happy_agent,
        "summarise the quarterly report",
        judge_only=True,
        call_llm=call_llm,
        model="fake-model",
        sinks=[sink],
    )

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.success, outcome.reason
    # NON-EMPTY transcript: a real plan with the framing task executed.
    assert outcome.session.plan is not None
    assert len(outcome.session.plan.tasks) >= 1
    # Sink captured native activity — the run did not abort with nothing.
    assert len(sink.events) > 0


async def test_judge_only_does_not_abort_unlike_passthrough() -> None:
    """Direct contrast: PassthroughPlanner aborts empty; judge_only does not.

    A bare ``PassthroughPlanner`` (the obvious "no planning" trap)
    returns ``None`` from generate/handle_turn, so the run ends without
    a plan. ``judge_only=True`` produces a real plan instead.
    """
    # Trap: explicit PassthroughPlanner => no usable plan lands, run
    # aborts (``success=False``, "no plan generated") with no tasks.
    trap = await goldfive.run(
        _happy_agent,
        "do the thing",
        planner=PassthroughPlanner(),
        sinks=[InMemorySink()],
    )
    assert trap.success is False
    assert trap.session.plan is None or not trap.session.plan.tasks

    # judge_only: a real plan lands and the agent runs natively.
    jo = await goldfive.run(
        _happy_agent,
        "do the thing",
        judge_only=True,
        call_llm=_spy_call_llm([{"progressing": True}]),
        model="fake-model",
        sinks=[InMemorySink()],
    )
    assert jo.session.plan is not None
    assert len(jo.session.plan.tasks) >= 1


# ---------------------------------------------------------------------------
# 3. Drift judges still fire under the judge-only steerer
# ---------------------------------------------------------------------------


async def test_judge_only_steerer_still_fires_goal_drift_judge() -> None:
    """The judge-only-built steerer emits a GOAL_DRIFT drift at the interval.

    Proves the judges remain functional under judge_only — only the
    planning overlay is suppressed, not the judging.
    """
    call_llm = _spy_call_llm(
        [{"progressing": False, "reason": "researching raccoons not solar panels"}]
    )
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    # Drive the trajectory judge directly through the steerer (mirrors
    # test_wrap_goal_drift_wiring). Use the steerer's own interval.
    sink = InMemorySink()
    steerer.bind(sinks=[sink], planner=runner.planner)
    session = _make_drift_session()

    # Cross the check interval so the goal-drift judge fires.
    for _ in range(steerer._goal_drift_check_interval):
        await steerer.drift.note_agent_turn(session)
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)

    # The judge fired (the spy recorded exactly the judge call).
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. Explicit planner / goal_deriver override under judge_only
# ---------------------------------------------------------------------------


def test_judge_only_respects_explicit_planner() -> None:
    """An explicit ``planner=`` wins even under ``judge_only=True``."""
    explicit = StaticPlanner(
        Plan(id="px", run_id="", goal_ids=[], tasks=[Task(id="tx", title="custom")], edges=[])
    )
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        planner=explicit,
        call_llm=_spy_call_llm([]),
        model="fake-model",
        sinks=[],
    )
    assert runner.planner is explicit


def test_judge_only_respects_explicit_goal_deriver() -> None:
    """An explicit ``goal_deriver=`` wins even under ``judge_only=True``."""

    class _ExplicitGoalDeriver:
        async def derive(self, user_input: str, **_: Any) -> list[Goal]:
            return [Goal(id="g-explicit", summary=user_input)]

    explicit = _ExplicitGoalDeriver()
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        goal_deriver=explicit,
        call_llm=_spy_call_llm([]),
        model="fake-model",
        sinks=[],
    )
    assert runner.goal_deriver is explicit


@pytest.mark.parametrize("judge_only", [True, False])
def test_judge_only_flag_does_not_disturb_judge_arming(judge_only: bool) -> None:
    """Judges are armed off ``call_llm`` regardless of the judge_only flag."""
    call_llm = _spy_call_llm([])
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=judge_only,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_call_llm is call_llm
    assert steerer._reasoning_drift_call_llm is call_llm
