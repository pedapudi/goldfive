"""Runner-level multi-turn gating tests (planner-gate).

Verifies the core promise of the planning gate: a conversational
follow-up on turn 2 does not re-run goal derivation or planning and
emits no ``GoalDerived`` / ``PlanSubmitted`` events, but still
produces a terminal ``RunCompleted``. Turn 1 still runs full planning.
"""

from __future__ import annotations

from typing import Any

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_plan() -> Plan:
    return Plan(
        id="plan-fixture",
        run_id="",
        goal_ids=["g"],
        tasks=[
            Task(id="research", title="Research", assignee_agent_id="writer"),
            Task(id="draft", title="Draft", assignee_agent_id="writer"),
        ],
        edges=[TaskEdge(from_task_id="research", to_task_id="draft")],
        summary="Research then draft.",
    )


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _kinds(events: list[Any]) -> list[str]:
    out: list[str] = []
    for e in events:
        name = e.WhichOneof("payload") or ""
        out.append("".join(part.capitalize() for part in name.split("_")) if name else "")
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_conversational_turn_skips_planning_llm_gate() -> None:
    """Turn 2 classified as "conversational" must not emit PlanSubmitted."""
    # An LLM planner whose call_llm returns a full plan on every call —
    # so we can tell from event counts whether the gate actually
    # skipped ``generate`` on turn 2. The gate itself also reads the
    # planner's call_llm; we route gate calls to a separate path by
    # sniffing the system prompt.

    plan_json = (
        '{"summary":"t","tasks":[{"id":"t1","title":"T","assignee_agent_id":"writer"}]}'
    )

    calls: list[tuple[str, str]] = []

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        calls.append((system[:40], user[:40]))
        # The gate's system prompt starts with the distinctive phrase
        # "You are a turn-classifier"; route those calls to the
        # conversational verdict on turn 2 and new_work on turn 1.
        if "turn-classifier" in system:
            return '{"verdict": "conversational", "reason": "ask about location"}'
        # Otherwise it's a plan-generate call.
        return plan_json

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("make a 2-slide presentation about solar panels")
    assert out1.success
    turn1_plan_id = out1.session.plan.id
    turn1_end_index = len(sink.events)

    out2 = await runner.run("where is the presentation located?")
    await runner.close()

    turn2_kinds = _kinds(sink.events[turn1_end_index:])

    # Turn 2 emits RunStarted and RunCompleted but NEITHER
    # GoalDerived nor PlanSubmitted — the gate short-circuited the
    # planning phase.
    assert "RunStarted" in turn2_kinds
    assert "GoalDerived" not in turn2_kinds
    assert "PlanSubmitted" not in turn2_kinds
    assert "RunCompleted" in turn2_kinds or "RunAborted" in turn2_kinds

    # Session.plan on turn 2 is the SAME plan id as turn 1 — carried
    # forward verbatim, not regenerated.
    assert out2.session.plan is not None
    assert out2.session.plan.id == turn1_plan_id


async def test_first_turn_still_runs_full_planning() -> None:
    """Turn 1 with no prior plan always runs goal-derive + generate."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    await runner.run("turn one")
    await runner.close()

    kinds = _kinds(sink.events)
    assert "RunStarted" in kinds
    assert "GoalDerived" in kinds
    assert "PlanSubmitted" in kinds


async def test_planner_gate_none_disables_classifier() -> None:
    """``planner_gate=None`` restores pre-gate behaviour: every turn re-plans."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
        planner_gate=None,
    )
    await runner.run("turn one")
    turn1_end = len(sink.events)
    await runner.run("where is the artefact?")
    await runner.close()

    turn2 = _kinds(sink.events[turn1_end:])
    # With the gate disabled, every turn emits GoalDerived +
    # PlanSubmitted — the pre-gate shape.
    assert "GoalDerived" in turn2
    assert "PlanSubmitted" in turn2


async def test_caller_supplied_gate_is_invoked() -> None:
    """Callers can inject a custom classifier callable."""
    captured: dict[str, Any] = {}

    async def gate(*, prior_plan, completed_results, user_input, conversation_id):
        captured["user_input"] = user_input
        captured["conversation_id"] = conversation_id
        return "conversational"

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
        planner_gate=gate,
    )
    await runner.run("turn one")
    turn1_end = len(sink.events)
    await runner.run("just a quick question about what you did")
    await runner.close()

    assert captured["user_input"] == "just a quick question about what you did"
    assert captured["conversation_id"] == runner.conversation_id

    turn2 = _kinds(sink.events[turn1_end:])
    # Gate returned "conversational" → no PlanSubmitted on turn 2.
    assert "PlanSubmitted" not in turn2


async def test_conversational_turn_carries_prior_plan_unchanged() -> None:
    """Turn 2 session.plan identity matches the prior turn's plan id + tasks."""

    async def gate(*, prior_plan, completed_results, user_input, conversation_id):
        _ = prior_plan, completed_results, user_input, conversation_id
        return "conversational"

    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
        planner_gate=gate,
    )
    out1 = await runner.run("turn one")
    out2 = await runner.run("a follow-up question")
    await runner.close()

    assert out1.session.plan.id == out2.session.plan.id
    assert [t.id for t in out1.session.plan.tasks] == [
        t.id for t in out2.session.plan.tasks
    ]
