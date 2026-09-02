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
4. Critical custom drift emits judgement and drift evidence without a
   refine call or intervention event, while full mode still refines.
5. Explicit component overrides retain their documented precedence.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

import goldfive
from goldfive import (
    CallableAdapter,
    DriftKind,
    DriftSeverity,
    ExecutionOutcome,
    InMemorySink,
    InvocationResult,
    JudgeContext,
    JudgeVerdict,
    LiteralGoalDeriver,
    LLMGoalDeriver,
    LLMPlanner,
    PassthroughPlanner,
    ReasoningDriftConfig,
    ReportingToolSpec,
    RuntimeConfig,
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


class _RecordingPlanner(StaticPlanner):
    """Planner that records drift responses and returns a valid revision."""

    def __init__(self) -> None:
        super().__init__(
            Plan(
                id="recording-plan",
                run_id="",
                goal_ids=[],
                tasks=[Task(id="t1", title="Native agent run")],
                edges=[],
            )
        )
        self.refine_calls: list[tuple[Plan, Any]] = []

    async def refine(self, *, plan: Plan, drift: Any, **_: Any) -> Plan:
        self.refine_calls.append((plan, drift))
        revised_tasks = (
            *plan.tasks,
            Task(
                id="t2",
                title="Repair output schema",
                description="Add the fields required by the output contract.",
            ),
        )
        return replace(plan, tasks=revised_tasks, summary=f"{plan.summary} (refined)")


class _CriticalSchemaJudge:
    """Custom judge whose verdict enters the full-mode refine path."""

    name = "critical_schema_judge"

    def __init__(self) -> None:
        self.calls: list[JudgeContext] = []

    async def evaluate(self, ctx: JudgeContext) -> JudgeVerdict:
        self.calls.append(ctx)
        return JudgeVerdict(
            drift_emitted=True,
            drift_kind=DriftKind.SCHEMA_VIOLATION,
            severity=DriftSeverity.CRITICAL,
            detail="required output fields are missing",
        )


class _ReasoningAdapter(CallableAdapter):
    """Adapter that emits one reasoning observation during invocation."""

    def __init__(self) -> None:
        super().__init__(self._invoke, available_agents=["schema_agent"])

    async def _invoke(
        self,
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = tools
        await self.emit_reasoning(
            "The answer must satisfy the required output schema.",
            task=task,
            session=session,
            agent_name="schema_agent",
        )
        return InvocationResult(task_id=task.id, text="native run completed")


def _proto_payload_names(sink: InMemorySink) -> list[str]:
    return [event.WhichOneof("payload") for event in sink.events if hasattr(event, "WhichOneof")]


def _dict_event_kinds(sink: InMemorySink) -> list[str]:
    return [str(event.get("kind", "")) for event in sink.events if isinstance(event, dict)]


def _assert_critical_evidence_without_response(
    sink: InMemorySink, planner: _RecordingPlanner
) -> list[str]:
    payloads = _proto_payload_names(sink)
    assert payloads.count("judgement_emitted") == 1
    assert payloads.count("drift_detected") == 1
    assert planner.refine_calls == []
    assert {
        "ladder_transition_decided",
        "policy_applied",
        "signal_delivered",
        "invocation_cancelled",
    }.isdisjoint(payloads)
    assert {"refine_attempted", "refine_failed"}.isdisjoint(_dict_event_kinds(sink))
    decisions = [
        event.steering_decision_made
        for event in sink.events
        if hasattr(event, "WhichOneof") and event.WhichOneof("payload") == "steering_decision_made"
    ]
    assert len(decisions) == 1
    assert decisions[0].outcome == "drift_observed_only"
    assert decisions[0].chosen_severity == ""
    return payloads


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


def test_judge_only_accepts_dedicated_judge_route_without_planning_calls() -> None:
    """Dedicated judge routing does not change judge-only planner defaults."""
    planner_llm = _spy_call_llm([])
    judge_llm = _spy_call_llm([])
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        call_llm=planner_llm,
        model="planner-model",
        judge_call_llm=judge_llm,
        judge_model="judge-model",
        sinks=[],
    )

    assert isinstance(runner.planner, StaticPlanner)
    assert isinstance(runner.goal_deriver, LiteralGoalDeriver)
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_call_llm is judge_llm
    assert steerer._reasoning_drift_call_llm is judge_llm
    assert steerer._goal_drift_model == "judge-model"
    assert steerer._reasoning_drift_model == "judge-model"
    assert planner_llm.calls == []  # type: ignore[attr-defined]
    assert judge_llm.calls == []  # type: ignore[attr-defined]


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


async def test_judge_only_custom_critical_drift_emits_without_response() -> None:
    """A critical custom verdict remains observable without intervention."""
    planner = _RecordingPlanner()
    sink = InMemorySink()
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        planner=planner,
        call_llm=_spy_call_llm([]),
        model="fake-model",
        judges=[_CriticalSchemaJudge()],
        sinks=[sink],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_drift_session()
    await steerer.evaluate_judges(JudgeContext(), session=session)

    _assert_critical_evidence_without_response(sink, planner)


async def test_judge_only_run_completes_after_critical_custom_verdict() -> None:
    """The public run path drains critical evidence without responding."""
    planner = _RecordingPlanner()
    judge = _CriticalSchemaJudge()
    sink = InMemorySink()

    outcome = await goldfive.run(
        _ReasoningAdapter(),
        "produce the required object",
        judge_only=True,
        planner=planner,
        judges=[judge],
        sinks=[sink],
        runtime=RuntimeConfig(reasoning_drift=ReasoningDriftConfig(mode="off")),
    )

    assert outcome.success, outcome.reason
    assert len(judge.calls) == 1
    payloads = _assert_critical_evidence_without_response(sink, planner)
    assert payloads.count("plan_revised") == 1  # initial plan installation
    judgement_index = payloads.index("judgement_emitted")
    assert "plan_revised" not in payloads[judgement_index + 1 :]


async def test_full_mode_custom_critical_drift_still_refines() -> None:
    """The same custom verdict retains the full intervention policy."""
    planner = _RecordingPlanner()
    sink = InMemorySink()
    runner = goldfive.wrap(
        _happy_agent,
        planner=planner,
        call_llm=_spy_call_llm([]),
        model="fake-model",
        judges=[_CriticalSchemaJudge()],
        sinks=[sink],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    steerer.bind(sinks=[sink], planner=planner)

    await steerer.evaluate_judges(JudgeContext(), session=_make_drift_session())

    assert len(planner.refine_calls) == 1
    assert "refine_attempted" in _dict_event_kinds(sink)
    payloads = _proto_payload_names(sink)
    assert payloads.count("judgement_emitted") == 1
    assert payloads.count("drift_detected") == 1


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


def test_judge_only_respects_explicit_steerer() -> None:
    """An explicit steerer keeps its own drift-response policy."""
    explicit = DefaultSteerer(dispatch_drift_interventions=True)
    runner = goldfive.wrap(
        _happy_agent,
        judge_only=True,
        steerer=explicit,
        call_llm=_spy_call_llm([]),
        model="fake-model",
        sinks=[],
    )
    assert runner.steerer is explicit
    assert explicit._dispatch_drift_interventions is True


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
