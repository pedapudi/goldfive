"""Drift self-reporting opt-in (goldfive#196).

The drift-related self-reporting tools (``report_plan_divergence``,
``declare_task_skipped``, ``declare_task_not_needed``) overlap with the
framework's observation-driven detectors (``classify_goal_drift``,
:class:`~goldfive.reconciler.PlanReconciler`, the steerer's refine
machinery), so registering them on every sub-agent is pure downside:
each tool's schema inflates the prompt by ~200-400 tokens AND expands
the model's hallucination surface. Goldfive#196 makes them opt-in via
``Runner(drift_self_reporting=...)``.

This module pins the contract:

* Default :class:`Runner` registers ONLY the lifecycle subset.
* ``drift_self_reporting=True`` registers the full canonical set
  (legacy / pre-#196 behaviour).
* ``drift_self_reporting=["<name>"]`` registers the lifecycle subset
  PLUS the named drift tools.
* ``report_new_work_discovered`` is intentionally NOT a drift tool —
  there is no observation analog, so it stays default-on.
* The ``select_reporting_tools`` helper that the Runner uses is
  exposed at module scope so custom adapters / Runner-likes can derive
  the same subset from the same flag.
"""

from __future__ import annotations

import json
from typing import Any

from goldfive import (
    BUILTIN_REPORTING_TOOLS,
    DRIFT_SELF_REPORTING_TOOL_NAMES,
    DRIFT_SELF_REPORTING_TOOLS,
    LIFECYCLE_REPORTING_TOOLS,
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
from goldfive.reporting import select_reporting_tools

# ---------------------------------------------------------------------------
# Constants — pin the documented split
# ---------------------------------------------------------------------------


def test_drift_tool_names_are_the_three_documented() -> None:
    """The drift-only set is exactly the three tools called out in #196."""
    assert DRIFT_SELF_REPORTING_TOOL_NAMES == frozenset(
        {
            "report_plan_divergence",
            "declare_task_skipped",
            "declare_task_not_needed",
        }
    )


def test_lifecycle_subset_excludes_every_drift_tool() -> None:
    lifecycle_names = {spec.name for spec in LIFECYCLE_REPORTING_TOOLS}
    assert lifecycle_names.isdisjoint(DRIFT_SELF_REPORTING_TOOL_NAMES)


def test_new_work_discovered_stays_in_lifecycle_subset() -> None:
    """``report_new_work_discovered`` has no observation analog and stays
    default-on (per the goldfive#196 spec)."""
    lifecycle_names = {spec.name for spec in LIFECYCLE_REPORTING_TOOLS}
    assert "report_new_work_discovered" in lifecycle_names


def test_lifecycle_subset_contains_every_status_tool() -> None:
    """Every ``report_task_*`` lifecycle tool plus ``report_awaiting_approval``
    is in the default-on subset."""
    lifecycle_names = {spec.name for spec in LIFECYCLE_REPORTING_TOOLS}
    expected = {
        "report_task_started",
        "report_task_progress",
        "report_task_completed",
        "report_task_failed",
        "report_task_blocked",
        "report_awaiting_approval",
        "report_new_work_discovered",
    }
    assert expected <= lifecycle_names


def test_drift_subset_plus_lifecycle_subset_equals_full_set() -> None:
    """Partition invariant: every BUILTIN tool is in EXACTLY ONE bucket."""
    lifecycle_names = {spec.name for spec in LIFECYCLE_REPORTING_TOOLS}
    drift_names = {spec.name for spec in DRIFT_SELF_REPORTING_TOOLS}
    builtin_names = {spec.name for spec in BUILTIN_REPORTING_TOOLS}
    assert lifecycle_names | drift_names == builtin_names
    assert lifecycle_names & drift_names == set()


# ---------------------------------------------------------------------------
# select_reporting_tools — the helper Runner.run consults
# ---------------------------------------------------------------------------


def _names(specs: list[ReportingToolSpec]) -> set[str]:
    return {spec.name for spec in specs}


def test_select_false_returns_lifecycle_only() -> None:
    selected = _names(select_reporting_tools(False))
    assert selected == _names(LIFECYCLE_REPORTING_TOOLS)
    assert selected.isdisjoint(DRIFT_SELF_REPORTING_TOOL_NAMES)


def test_select_true_returns_full_canonical_set() -> None:
    selected = _names(select_reporting_tools(True))
    assert selected == _names(BUILTIN_REPORTING_TOOLS)


def test_select_list_adds_named_drift_tools_to_lifecycle() -> None:
    selected = _names(select_reporting_tools(["report_plan_divergence"]))
    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {"report_plan_divergence"}
    assert selected == expected
    # The other drift tools stay off.
    assert "declare_task_skipped" not in selected
    assert "declare_task_not_needed" not in selected


def test_select_list_with_two_names_enables_both() -> None:
    selected = _names(
        select_reporting_tools(["declare_task_skipped", "declare_task_not_needed"])
    )
    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {
        "declare_task_skipped",
        "declare_task_not_needed",
    }
    assert selected == expected
    assert "report_plan_divergence" not in selected


def test_select_empty_list_collapses_to_lifecycle_subset() -> None:
    """An empty iterable behaves like ``False`` — no drift tools enabled."""
    selected = _names(select_reporting_tools([]))
    assert selected == _names(LIFECYCLE_REPORTING_TOOLS)


def test_select_silently_ignores_unknown_drift_tool_names() -> None:
    """Names that aren't in ``DRIFT_SELF_REPORTING_TOOL_NAMES`` are ignored;
    typos must not silently turn a non-drift tool off."""
    selected = _names(select_reporting_tools(["report_plan_divergence", "report_typo"]))
    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {"report_plan_divergence"}
    assert selected == expected


def test_select_accepts_set_and_tuple() -> None:
    set_selected = _names(select_reporting_tools({"report_plan_divergence"}))
    tup_selected = _names(select_reporting_tools(("report_plan_divergence",)))
    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {"report_plan_divergence"}
    assert set_selected == expected
    assert tup_selected == expected


# ---------------------------------------------------------------------------
# Runner integration — drift tools land on the adapter only when opted in
# ---------------------------------------------------------------------------


def _linear_plan() -> Plan:
    return Plan(
        id="plan",
        run_id="",
        goal_ids=["g"],
        tasks=[Task(id="t1", title="One", assignee_agent_id="writer")],
        edges=[],
        summary="one task",
    )


async def _no_op_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = session, tools
    return InvocationResult(task_id=task.id, text="done")


def _build_runner(drift_self_reporting: Any = False) -> tuple[Runner, CallableAdapter]:
    adapter = CallableAdapter(_no_op_agent, available_agents=["writer"])
    runner = Runner(
        agent=adapter,
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
        drift_self_reporting=drift_self_reporting,
    )
    return runner, adapter


async def test_runner_default_registers_lifecycle_subset_only() -> None:
    runner, adapter = _build_runner(drift_self_reporting=False)
    await runner.run("turn one")
    await runner.close()
    registered = {spec.name for spec in adapter._tools}
    assert registered == _names(LIFECYCLE_REPORTING_TOOLS)
    assert registered.isdisjoint(DRIFT_SELF_REPORTING_TOOL_NAMES)


async def test_runner_drift_true_registers_full_canonical_set() -> None:
    runner, adapter = _build_runner(drift_self_reporting=True)
    await runner.run("turn one")
    await runner.close()
    registered = {spec.name for spec in adapter._tools}
    assert registered == _names(BUILTIN_REPORTING_TOOLS)


async def test_runner_drift_list_registers_lifecycle_plus_named() -> None:
    runner, adapter = _build_runner(drift_self_reporting=["report_plan_divergence"])
    await runner.run("turn one")
    await runner.close()
    registered = {spec.name for spec in adapter._tools}
    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {"report_plan_divergence"}
    assert registered == expected
    assert "declare_task_skipped" not in registered
    assert "declare_task_not_needed" not in registered


async def test_runner_drift_list_persists_across_turns() -> None:
    """The flag is stored on the Runner, so every turn registers the same
    set. Materialising the list eagerly in __init__ means callers passing
    a mutable list don't see drift tools changing turn-to-turn."""
    user_list = ["report_plan_divergence"]
    runner, adapter = _build_runner(drift_self_reporting=user_list)
    # Mutate the caller-side list — must not affect Runner behaviour.
    user_list.append("declare_task_skipped")

    await runner.run("turn one")
    turn_one = {spec.name for spec in adapter._tools}

    await runner.run("turn two")
    turn_two = {spec.name for spec in adapter._tools}
    await runner.close()

    expected = _names(LIFECYCLE_REPORTING_TOOLS) | {"report_plan_divergence"}
    assert turn_one == expected
    assert turn_two == expected


def test_runner_default_attribute_is_false() -> None:
    """Constructing without the kwarg should pin the attribute to False —
    pre-existing call sites get the documented default automatically."""
    adapter = CallableAdapter(_no_op_agent, available_agents=["writer"])
    runner = Runner(
        agent=adapter,
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
    )
    assert runner.drift_self_reporting is False


# ---------------------------------------------------------------------------
# convenience.wrap forwards the kwarg
# ---------------------------------------------------------------------------


async def test_wrap_forwards_drift_self_reporting() -> None:
    """``goldfive.wrap(drift_self_reporting=True)`` propagates to the
    inner Runner so the prompt-side opt-in works through the high-level
    API too."""
    from goldfive import wrap

    plan_json = json.dumps({
        "summary": "t",
        "tasks": [{"id": "t1", "title": "T", "assignee_agent_id": "writer"}],
    })

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = system, user, model
        return plan_json

    runner = wrap(
        _no_op_agent,
        planner=LLMPlanner(call_llm=planner_llm, model="stub"),
        goal_deriver=PassthroughGoalDeriver("demo"),
        executor=SequentialExecutor(),
        sinks=[InMemorySink()],
        drift_self_reporting=True,
    )
    assert runner.drift_self_reporting is True
    await runner.close()


async def test_wrap_default_is_lifecycle_only() -> None:
    """Default ``wrap()`` has drift_self_reporting=False on the runner."""
    from goldfive import wrap

    plan_json = json.dumps({
        "summary": "t",
        "tasks": [{"id": "t1", "title": "T", "assignee_agent_id": "writer"}],
    })

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = system, user, model
        return plan_json

    runner = wrap(
        _no_op_agent,
        planner=LLMPlanner(call_llm=planner_llm, model="stub"),
        goal_deriver=PassthroughGoalDeriver("demo"),
        executor=SequentialExecutor(),
        sinks=[InMemorySink()],
    )
    assert runner.drift_self_reporting is False
    await runner.close()


# ---------------------------------------------------------------------------
# Edge: TaskEdge unused but imported keeps the test module aligned with
# the rest of the suite's import style; mark as used so linters stay
# quiet without a noqa.
# ---------------------------------------------------------------------------


_ = TaskEdge  # noqa: PIE794 — kept for cross-test consistency
