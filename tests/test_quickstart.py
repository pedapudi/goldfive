"""Tests for :func:`goldfive.quickstart`.

Exercises the convenience factory across its accepted ``goals`` shapes
and verifies that the returned :class:`Runner` is correctly wired with
the documented defaults (and overrides them when supplied).
"""

from __future__ import annotations

from typing import Any

from goldfive import (
    CallableAdapter,
    ExecutionOutcome,
    Goal,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    quickstart,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Reference agent: returns a non-empty result so the executor auto-completes."""
    _ = (session, tools)
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _kinds(events: list[Any]) -> list[str]:
    out: list[str] = []
    for e in events:
        if isinstance(e, dict):
            out.append(e.get("kind") or "")
        elif hasattr(e, "WhichOneof"):
            name = e.WhichOneof("payload") or ""
            out.append(
                "".join(part.capitalize() for part in name.split("_")) if name else ""
            )
        else:
            out.append(getattr(e, "kind", ""))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_quickstart_string_goal_runs_to_completion() -> None:
    """``quickstart(agent, "single goal string")`` runs end-to-end."""
    runner = quickstart(_happy_agent, "Say hello to the user")
    assert isinstance(runner, Runner)

    sink = runner.sinks[0]
    assert isinstance(sink, InMemorySink)

    outcome = await runner.run("ignored — passthrough deriver")
    await runner.close()

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.success, outcome.reason

    kinds = _kinds(sink.events)
    assert kinds[0] == "RunStarted"
    assert "GoalDerived" in kinds
    assert "PlanSubmitted" in kinds
    assert kinds[-1] == "RunCompleted"

    # Exactly one task was generated for the single goal.
    assert outcome.session.plan is not None
    assert len(outcome.session.plan.tasks) == 1
    assert len(outcome.session.goals) == 1
    assert outcome.session.goals[0].summary == "Say hello to the user"


async def test_quickstart_accepts_list_of_goals() -> None:
    """A list of ``Goal`` objects produces one task per goal."""
    goals = [
        Goal(id="g1", summary="First outcome"),
        Goal(id="g2", summary="Second outcome"),
    ]
    runner = quickstart(_happy_agent, goals)
    sink = runner.sinks[0]
    assert isinstance(sink, InMemorySink)

    outcome = await runner.run("ignored")
    await runner.close()

    assert outcome.success, outcome.reason
    assert outcome.session.plan is not None
    assert len(outcome.session.plan.tasks) == 2
    assert outcome.session.goals == goals


async def test_quickstart_accepts_list_of_strings() -> None:
    """A list of summary strings is normalised into ``Goal`` objects."""
    runner = quickstart(_happy_agent, ["alpha", "beta", "gamma"])
    outcome = await runner.run("ignored")
    await runner.close()

    assert outcome.success, outcome.reason
    assert [g.summary for g in outcome.session.goals] == ["alpha", "beta", "gamma"]
    assert outcome.session.plan is not None
    assert len(outcome.session.plan.tasks) == 3


async def test_quickstart_custom_sinks_override_default() -> None:
    """A caller-supplied ``sinks`` argument replaces the InMemorySink default."""
    custom_sink = InMemorySink()
    runner = quickstart(_happy_agent, "do the thing", sinks=[custom_sink])

    assert runner.sinks == [custom_sink]
    # No second InMemorySink is silently appended.
    assert len(runner.sinks) == 1

    outcome = await runner.run("ignored")
    await runner.close()
    assert outcome.success, outcome.reason
    # The override sink is the one that received events.
    assert any(_kinds([e])[0] == "RunCompleted" for e in custom_sink.events)


async def test_quickstart_returns_configured_runner_wiring() -> None:
    """The returned object is a Runner with the documented default wiring."""
    runner = quickstart(_happy_agent, "wiring check")

    assert isinstance(runner, Runner)
    assert isinstance(runner.executor, SequentialExecutor)
    assert isinstance(runner.goal_deriver, PassthroughGoalDeriver)
    assert isinstance(runner.planner, StaticPlanner)
    assert len(runner.sinks) == 1
    assert isinstance(runner.sinks[0], InMemorySink)


async def test_quickstart_does_not_rewrap_existing_adapter() -> None:
    """An existing ``AgentAdapter`` is used verbatim, not re-wrapped."""
    adapter = CallableAdapter(_happy_agent, available_agents=["writer"])
    runner = quickstart(adapter, "use my adapter")

    assert runner.agent is adapter

    outcome = await runner.run("ignored")
    await runner.close()
    assert outcome.success, outcome.reason


async def test_quickstart_custom_planner_override() -> None:
    """A caller-supplied ``planner`` replaces the default StaticPlanner."""
    from goldfive.types import Plan

    custom_plan = Plan(
        id="custom",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id="only", title="Only task", assignee_agent_id="default")],
        edges=[],
        summary="custom override",
    )
    custom_planner = StaticPlanner(custom_plan)
    runner = quickstart(_happy_agent, "anything", planner=custom_planner)

    assert runner.planner is custom_planner

    outcome = await runner.run("ignored")
    await runner.close()
    assert outcome.success, outcome.reason
    assert outcome.session.plan is not None
    assert [t.id for t in outcome.session.plan.tasks] == ["only"]
