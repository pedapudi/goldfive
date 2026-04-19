"""``goldfive.quickstart`` — one-call Runner factory for new users.

Wires a :class:`SequentialExecutor`, a :class:`PassthroughGoalDeriver`,
a :class:`StaticPlanner` whose plan is one task per goal, and a single
:class:`InMemorySink` so a caller can go from "I have an agent and some
goals" to a runnable :class:`Runner` in one expression. Complements —
does not replace — explicit :class:`Runner` construction.
"""

from __future__ import annotations

from typing import Any

from goldfive.adapters.callable import CallableAdapter
from goldfive.executors.sequential import SequentialExecutor
from goldfive.goal_deriver import PassthroughGoalDeriver
from goldfive.planner import StaticPlanner
from goldfive.protocols import AgentAdapter, EventSink, Planner
from goldfive.runner import Runner
from goldfive.sinks.memory import InMemorySink
from goldfive.types import Goal, Plan, Task

_DEFAULT_AGENT_ID = "default"


def _is_agent_adapter(obj: Any) -> bool:
    """True when ``obj`` already satisfies the :class:`AgentAdapter` protocol."""
    return isinstance(obj, AgentAdapter)


def _coerce_goal_list(goals: str | Goal | list[str | Goal]) -> list[Goal]:
    """Normalise ``goals`` into a non-empty ``list[Goal]``."""
    if isinstance(goals, Goal):
        return [goals]
    if isinstance(goals, str):
        if not goals.strip():
            raise ValueError("quickstart: empty goal string")
        return [Goal(id="g1", summary=goals)]
    if not isinstance(goals, list):
        raise TypeError(
            f"quickstart: goals must be str | Goal | list[str | Goal], "
            f"got {type(goals).__name__}"
        )
    if not goals:
        raise ValueError("quickstart: empty goals list")
    out: list[Goal] = []
    for i, item in enumerate(goals, start=1):
        if isinstance(item, Goal):
            out.append(item)
        elif isinstance(item, str):
            if not item.strip():
                raise ValueError(f"quickstart: empty goal string at index {i - 1}")
            out.append(Goal(id=f"g{i}", summary=item))
        else:
            raise TypeError(
                f"quickstart: goals[{i - 1}] must be str or Goal, "
                f"got {type(item).__name__}"
            )
    return out


def _default_plan(goals: list[Goal], assignee_agent_id: str) -> Plan:
    """Build a one-task-per-goal linear plan with no edges."""
    return Plan(
        id="quickstart-plan",
        run_id="",
        goal_ids=[g.id for g in goals],
        tasks=[
            Task(
                id=f"task_{g.id}",
                title=g.summary[:60] or g.id,
                description=g.summary,
                assignee_agent_id=assignee_agent_id,
            )
            for g in goals
        ],
        edges=[],
        summary="quickstart: one task per goal",
    )


def quickstart(
    agent: Any,
    goals: str | Goal | list[str | Goal],
    *,
    planner: Planner | None = None,
    sinks: list[EventSink] | None = None,
) -> Runner:
    """Build a fully-wired :class:`Runner` with sensible defaults.

    Parameters
    ----------
    agent:
        Either an existing :class:`AgentAdapter` (used verbatim) or a
        :class:`CallableAdapter`-compatible async callable (wrapped).
    goals:
        A single :class:`Goal`, a single summary string, or a list of
        either. Each entry becomes one task in the default plan.
    planner:
        Optional override. Defaults to a :class:`StaticPlanner` whose
        plan has one task per goal.
    sinks:
        Optional override. Defaults to ``[InMemorySink()]``.

    Returns
    -------
    Runner
        A :class:`Runner` ready for ``await runner.run(...)``.
    """
    goal_list = _coerce_goal_list(goals)

    if _is_agent_adapter(agent):
        adapter: AgentAdapter = agent
    else:
        adapter = CallableAdapter(agent, available_agents=[_DEFAULT_AGENT_ID])

    available = list(adapter.available_agents) or [_DEFAULT_AGENT_ID]
    assignee = available[0]

    resolved_planner: Planner = (
        planner if planner is not None else StaticPlanner(_default_plan(goal_list, assignee))
    )
    resolved_sinks: list[EventSink] = list(sinks) if sinks is not None else [InMemorySink()]

    return Runner(
        agent=adapter,
        planner=resolved_planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver(goal_list),
        sinks=resolved_sinks,
    )


__all__ = ["quickstart"]
