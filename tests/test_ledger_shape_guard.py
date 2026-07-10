"""Ledger-shape gating for the ledger regime (AGENCY-PRESERVATION.md PR 10).

The ledger-only pin-tier bypass keys on the LIVE PLAN's shape
(:func:`goldfive.types.plan_has_ledger_shape`), not on
``SteeringConfig.plan_mode`` alone — "StaticPlanner users keep forecast
semantics — a hand-authored plan is genuine prescriptive intent" (design doc
Stage 3). Two pieces are pinned here:

* the shape probe itself (``plan_has_ledger_shape``);
* the Runner's one-shot incoherent-combo WARNING when ``plan_mode=ledger``
  resolves but the installed plan has no ledger-shaped task (the run keeps
  forecast pin semantics — the operator should know the configured regime
  is not engaged).

The pin-path behaviour under the combo is covered (ADK-gated) in
``test_ledger_pin_bypass.py``.
"""

from __future__ import annotations

import logging

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import (  # noqa: E402
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Runner,
    SequentialExecutor,
    StaticPlanner,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskKind,
    plan_has_ledger_shape,
)

_WARNING_MARKER = "NO ledger-shaped task"


# ---------------------------------------------------------------------------
# plan_has_ledger_shape
# ---------------------------------------------------------------------------


def _plan(*tasks: Task) -> Plan:
    return Plan(id="p", run_id="r", goal_ids=["g"], tasks=list(tasks), edges=[])


def test_shape_false_for_forecast_only_plan() -> None:
    assert plan_has_ledger_shape(_plan(Task(id="t1", title="x"))) is False


def test_shape_true_for_outcome_task() -> None:
    plan = _plan(
        Task(id="t1", title="x"),
        Task(id="t2", title="y", kind=TaskKind.OUTCOME),
    )
    assert plan_has_ledger_shape(plan) is True


def test_shape_true_for_discovered_task() -> None:
    plan = _plan(Task(id="t1", title="x", kind=TaskKind.DISCOVERED))
    assert plan_has_ledger_shape(plan) is True


def test_shape_false_for_none_and_empty() -> None:
    assert plan_has_ledger_shape(None) is False
    assert plan_has_ledger_shape(_plan()) is False


def test_shape_tolerates_string_kinds() -> None:
    # Serialised / stub tasks may carry the kind as a bare string.
    class _T:
        kind = "OUTCOME"

    class _P:
        tasks = (_T(),)

    assert plan_has_ledger_shape(_P()) is True


# ---------------------------------------------------------------------------
# Runner one-shot incoherent-combo warning
# ---------------------------------------------------------------------------


def _forecast_plan() -> Plan:
    return Plan(
        id="plan-static",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id="work", title="Do the work", assignee_agent_id="writer")],
        edges=[],
        summary="One hand-authored task.",
    )


def _ledger_shaped_plan() -> Plan:
    return Plan(
        id="plan-ledger",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="deliverable",
                title="Summary delivered",
                assignee_agent_id="writer",
                kind=TaskKind.OUTCOME,
            )
        ],
        edges=[],
        summary="One OUTCOME deliverable.",
    )


async def _happy_agent(task: Task, session: Session, tools: list) -> InvocationResult:
    _ = tools
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _runner(*, plan: Plan, plan_mode: str) -> Runner:
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("do the work"),
        steerer=DefaultSteerer(steering_config=SteeringConfig(plan_mode=plan_mode)),
        sinks=[InMemorySink()],
    )


def _combo_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _WARNING_MARKER in r.message]


async def test_ledger_config_with_forecast_plan_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _runner(plan=_forecast_plan(), plan_mode="ledger")
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        await runner.run("go")
        assert len(_combo_warnings(caplog)) == 1
        # One-shot per Runner: a second turn does not re-warn.
        await runner.run("go again")
        assert len(_combo_warnings(caplog)) == 1
    await runner.close()


async def test_forecast_config_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _runner(plan=_forecast_plan(), plan_mode="forecast")
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        await runner.run("go")
    await runner.close()
    assert _combo_warnings(caplog) == []


async def test_ledger_config_with_ledger_shaped_plan_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _runner(plan=_ledger_shaped_plan(), plan_mode="ledger")
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        await runner.run("go")
    await runner.close()
    assert _combo_warnings(caplog) == []
