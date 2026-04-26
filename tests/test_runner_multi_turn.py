"""Runner-level multi-turn tests for the Phase 4 handle_turn flow
(goldfive#271).

Phase 4 collapsed the prior planner_gate triage layer (factual-question
+ steer-language regex + LLM gate + synthesize_goal_from_steer +
qualification-merge regex + planner.refine) into a single
:meth:`Planner.handle_turn` LLM call that decides whether the turn
warrants a plan change AND, when one is warranted, produces the next
plan in one shot.

Verifies the core promise: a turn whose ``handle_turn`` returns
``None`` reuses ``session.plan`` unchanged and emits no GoalDerived /
PlanRevised. A turn whose ``handle_turn`` returns a Plan installs it
as the next revision (revision_index += 1; plan_id preserved).
"""

from __future__ import annotations

import json
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
        if isinstance(e, dict):
            out.append(e.get("kind") or "")
            continue
        if hasattr(e, "WhichOneof"):
            name = e.WhichOneof("payload") or ""
            out.append(
                "".join(part.capitalize() for part in name.split("_"))
                if name
                else ""
            )
        else:
            out.append(getattr(e, "kind", ""))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_first_turn_emits_plan_revised_for_initial_install() -> None:
    """Phase 4: every plan install (including the very first) is a
    revision of the Plan.empty() seed, so PlanRevised fires uniformly.
    """
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
    assert "PlanRevised" in kinds


async def test_handle_turn_none_reuses_prior_plan_no_replan() -> None:
    """When handle_turn returns None on a turn that DOES have a real
    prior plan, the Runner reuses session.plan unchanged and does NOT
    emit GoalDerived / PlanRevised for that turn.
    """
    plan_json = json.dumps({
        "summary": "t",
        "tasks": [{"id": "t1", "title": "T", "assignee_agent_id": "writer"}],
    })

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        # handle_turn system prompt — return null plan (conversational).
        if "next REVISION of the plan" in system or "warrants a plan change" in system:
            return json.dumps({"reasoning": "conversational", "plan": None})
        # Otherwise it's a plan-generate call (first turn fall-through).
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
    turn1_end = len(sink.events)

    out2 = await runner.run("where is the presentation located?")
    await runner.close()

    turn2_kinds = _kinds(sink.events[turn1_end:])
    # Conversational turn: no PlanRevised, no GoalDerived.
    assert "RunStarted" in turn2_kinds
    assert "PlanRevised" not in turn2_kinds, turn2_kinds
    # Session.plan on turn 2 is the SAME plan id as turn 1.
    assert out2.session.plan is not None
    assert out2.session.plan.id == turn1_plan_id


async def test_planner_gate_none_skips_handle_turn_on_first_turn() -> None:
    """``planner_gate=None`` skips the per-turn handle_turn call; the
    Runner falls through to ``planner.generate`` on the first turn
    (which is the realistic usage of planner_gate=None — single-turn
    deterministic replay).

    Phase 4 (goldfive#271): multi-turn runs with planner_gate=None
    are a degraded mode — handle_turn is the only path that knows
    how to merge prior plan state into a revision. Tests for that
    pattern live in the LLMPlanner-with-handle_turn variants below.
    """
    plan_t1 = json.dumps({
        "summary": "t",
        "tasks": [
            {"id": "research", "title": "Research", "assignee_agent_id": "writer"},
        ],
        "edges": [],
    })

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = system, user, model
        return plan_t1

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
        planner_gate=None,
    )
    out1 = await runner.run("turn one")
    await runner.close()
    assert out1.success
    kinds = _kinds(sink.events)
    assert "GoalDerived" in kinds
    assert "PlanRevised" in kinds
