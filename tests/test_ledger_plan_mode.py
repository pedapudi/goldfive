"""Ledger plan mode foundations (AGENCY-PRESERVATION.md Stage 3 PR 10).

All behaviour here is NEW code behind ``SteeringConfig.plan_mode ==
"ledger"`` (env ``GOLDFIVE_PLAN_MODE``), default OFF. The forecast-mode
guarantee — every existing suite passes unmodified, forecast output is
byte-identical — is enforced by those suites; this file only exercises
the additive surface:

* ``Task.kind`` (FORECAST / OUTCOME / DISCOVERED) + ``Task.contributes_to``
  data-model slots and their proto round-trip;
* ``SteeringConfig.plan_mode`` config + env parsing;
* ``LLMPlanner.generate`` / ``handle_turn`` producing OUTCOME deliverables
  in ledger mode (and staying FORECAST otherwise);
* ``Plan.validate`` accepting edge-free OUTCOME + DISCOVERED roots;
* ``PlanReviser._build_minimal_steer_evolution`` preserving ``Task.kind``
  (the USER_STEER deterministic fallback);
* ``PlanReviser.install_descriptive_growth`` stamping
  ``kind=DISCOVERED`` only in ledger mode;
* ``StaticPlanner`` preserving a hand-authored kind (prescriptive intent).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.conv import from_pb_task, to_pb_task  # noqa: E402
from goldfive.planner import (  # noqa: E402
    LLMPlanner,
    StaticPlanner,
    _plan_mode_from_context,
    _stamp_ledger_outcome_kinds,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskKind,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Data model: Task.kind / Task.contributes_to
# ---------------------------------------------------------------------------


def test_task_kind_defaults_to_forecast() -> None:
    t = Task(id="t1", title="A")
    assert t.kind is TaskKind.FORECAST
    assert t.contributes_to == ""


@pytest.mark.parametrize(
    "kind",
    [TaskKind.FORECAST, TaskKind.OUTCOME, TaskKind.DISCOVERED],
)
def test_task_kind_proto_round_trip(kind: TaskKind) -> None:
    t = Task(id="t1", title="deliverable", kind=kind, contributes_to="g1")
    recovered = from_pb_task(to_pb_task(t))
    assert recovered == t
    assert recovered.kind is kind
    assert recovered.contributes_to == "g1"


def test_forecast_task_serialises_with_proto3_default_kind() -> None:
    # FORECAST maps to the proto3 value-0 default → not written to the
    # wire, so a forecast task's serialisation is unchanged by PR 10.
    pb = to_pb_task(Task(id="x", title="y"))
    assert pb.kind == 0
    assert pb.contributes_to == ""


def test_legacy_proto_without_kind_deserialises_forecast() -> None:
    # Simulate an old serialised Task that never set the new fields: a
    # default-constructed pb message has kind=0 / contributes_to="".
    from goldfive.conv import _pb_module

    pb = _pb_module()
    msg = pb.Task(id="t", title="legacy")
    task = from_pb_task(msg)
    assert task.kind is TaskKind.FORECAST
    assert task.contributes_to == ""


# ---------------------------------------------------------------------------
# Config: SteeringConfig.plan_mode
# ---------------------------------------------------------------------------


def test_plan_mode_default_forecast() -> None:
    assert SteeringConfig().plan_mode == "forecast"


def test_plan_mode_from_env_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDFIVE_PLAN_MODE", "ledger")
    assert SteeringConfig.from_env().plan_mode == "ledger"


def test_plan_mode_from_env_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDFIVE_PLAN_MODE", "LEDGER")
    assert SteeringConfig.from_env().plan_mode == "ledger"


def test_plan_mode_from_env_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDFIVE_PLAN_MODE", "wheelgrab")
    assert SteeringConfig.from_env().plan_mode == "forecast"


def test_plan_mode_unset_is_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOLDFIVE_PLAN_MODE", raising=False)
    assert SteeringConfig.from_env().plan_mode == "forecast"


# ---------------------------------------------------------------------------
# _plan_mode_from_context + _stamp_ledger_outcome_kinds
# ---------------------------------------------------------------------------


def test_plan_mode_from_context_resolution() -> None:
    assert _plan_mode_from_context(None) == "forecast"
    assert _plan_mode_from_context({}) == "forecast"
    assert _plan_mode_from_context({"plan_mode": "ledger"}) == "ledger"
    assert _plan_mode_from_context({"plan_mode": "Ledger"}) == "ledger"
    assert _plan_mode_from_context({"plan_mode": "forecast"}) == "forecast"
    assert _plan_mode_from_context({"plan_mode": "garbage"}) == "forecast"


def test_stamp_outcome_kinds_all_new_when_no_prior() -> None:
    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=(),
        tasks=(Task(id="o1", title="a"), Task(id="o2", title="b")),
        edges=(),
    )
    stamped = _stamp_ledger_outcome_kinds(plan, None)
    assert all(t.kind is TaskKind.OUTCOME for t in stamped.tasks)
    # Input plan is not mutated (frozen-Plan invariant).
    assert all(t.kind is TaskKind.FORECAST for t in plan.tasks)


def test_stamp_outcome_kinds_preserves_prior_ledger_kinds() -> None:
    prior = Plan(
        id="p",
        run_id="r",
        goal_ids=(),
        tasks=(
            Task(id="o1", title="outcome", kind=TaskKind.OUTCOME),
            Task(id="d1", title="discovered", discovered=True, kind=TaskKind.DISCOVERED),
        ),
        edges=(),
    )
    revised = Plan(
        id="p",
        run_id="r",
        goal_ids=(),
        tasks=(
            Task(id="o1", title="outcome"),
            Task(id="d1", title="discovered"),
            Task(id="o2", title="new outcome"),
        ),
        edges=(),
    )
    stamped = _stamp_ledger_outcome_kinds(revised, prior)
    kinds = {t.id: t.kind for t in stamped.tasks}
    assert kinds["o1"] is TaskKind.OUTCOME  # carried forward
    assert kinds["d1"] is TaskKind.DISCOVERED  # NOT relabelled to OUTCOME
    assert kinds["o2"] is TaskKind.OUTCOME  # genuinely new deliverable


# ---------------------------------------------------------------------------
# LLMPlanner.generate — ledger vs forecast
# ---------------------------------------------------------------------------


def _ledger_generate_llm(expect_ledger: bool):
    async def call_llm(system: str, user: str, model: str) -> str:
        if expect_ledger:
            assert "OUTCOME" in system and "1 and 5" in system
        else:
            assert "5 and 20" in system  # the forecast default prompt
        return json.dumps(
            {
                "summary": "Deliver the summary",
                "tasks": [
                    {
                        "id": "summary_delivered",
                        "title": "Summary delivered",
                        "description": "the user has the summary",
                    }
                ],
                "edges": [],
            }
        )

    return call_llm


def test_generate_ledger_produces_outcome_tasks() -> None:
    planner = LLMPlanner(call_llm=_ledger_generate_llm(expect_ledger=True), model="m")
    plan = asyncio.run(
        planner.generate(
            goals=[Goal(id="g1", summary="Summarise the deck")],
            available_agents=["coordinator"],
            context={"plan_mode": "ledger", "run_id": "r1"},
        )
    )
    assert plan is not None
    assert plan.tasks
    assert all(t.kind is TaskKind.OUTCOME for t in plan.tasks)
    assert plan.tasks[0].id == "summary_delivered"
    # OUTCOME tasks carry no assignee / edges.
    assert all(t.assignee_agent_id == "" for t in plan.tasks)
    assert plan.edges == ()


def test_generate_forecast_default_kind_and_prompt() -> None:
    planner = LLMPlanner(call_llm=_ledger_generate_llm(expect_ledger=False), model="m")
    plan = asyncio.run(
        planner.generate(
            goals=[Goal(id="g1", summary="Summarise the deck")],
            available_agents=["coordinator"],
            context={"run_id": "r1"},  # no plan_mode key
        )
    )
    assert plan is not None
    assert all(t.kind is TaskKind.FORECAST for t in plan.tasks)


# ---------------------------------------------------------------------------
# LLMPlanner.handle_turn — ledger revisions
# ---------------------------------------------------------------------------


def test_handle_turn_ledger_stamps_outcome() -> None:
    async def call_llm(system: str, user: str, model: str) -> str:
        assert "OUTCOME" in system and "DELIVERABLE" in system
        return json.dumps(
            {
                "reasoning": "user asked for a deliverable",
                "replaces_prior": False,
                "plan": {
                    "id": "p1",
                    "summary": "Two-slide deck delivered",
                    "tasks": [
                        {
                            "id": "deck_delivered",
                            "title": "Two-slide deck delivered",
                            "description": "deck exists with 2 slides",
                            "status": "PENDING",
                        }
                    ],
                    "edges": [],
                },
            }
        )

    planner = LLMPlanner(call_llm=call_llm, model="m")
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="make a 2-slide deck")],
        plan=Plan.empty(run_id="r1"),
    )
    plan = asyncio.run(
        planner.handle_turn(
            user_input="make me a 2-slide deck",
            session=session,
            context={"plan_mode": "ledger"},
        )
    )
    assert plan is not None
    assert all(t.kind is TaskKind.OUTCOME for t in plan.tasks)


def test_handle_turn_ledger_preserves_prior_discovered_kind() -> None:
    # Prior ledger: one terminal DISCOVERED trajectory task + one OUTCOME.
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=(
            Task(
                id="d1",
                title="debugger: located files",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.COMPLETED,
            ),
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
        ),
        edges=(),
        revision_index=2,
    )

    async def call_llm(system: str, user: str, model: str) -> str:
        # Echo the terminal DISCOVERED task verbatim + add a new OUTCOME.
        return json.dumps(
            {
                "reasoning": "add a translation deliverable",
                "replaces_prior": False,
                "plan": {
                    "id": "p1",
                    "summary": "Summary + translation delivered",
                    "tasks": [
                        {"id": "d1", "title": "debugger: located files", "status": "COMPLETED"},
                        {"id": "o1", "title": "Summary delivered", "status": "PENDING"},
                        {"id": "o2", "title": "Translation delivered", "status": "PENDING"},
                    ],
                    "edges": [],
                },
            }
        )

    planner = LLMPlanner(call_llm=call_llm, model="m")
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="summarise")], plan=prior)
    plan = asyncio.run(
        planner.handle_turn(
            user_input="also translate it",
            session=session,
            context={"plan_mode": "ledger"},
        )
    )
    assert plan is not None
    kinds = {t.id: t.kind for t in plan.tasks}
    assert kinds["d1"] is TaskKind.DISCOVERED  # preserved, not relabelled
    assert kinds["o1"] is TaskKind.OUTCOME
    assert kinds["o2"] is TaskKind.OUTCOME


# ---------------------------------------------------------------------------
# Plan.validate — edge-free OUTCOME / DISCOVERED roots
# ---------------------------------------------------------------------------


def test_validate_accepts_edge_free_outcome_plan_creation() -> None:
    plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g1", "g2"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(id="o2", title="Translation delivered", kind=TaskKind.OUTCOME),
        ),
        edges=(),
    )
    # Must not raise — every task PENDING, no edges.
    plan.validate(for_revision=False)


def test_validate_accepts_outcome_plus_discovered_revision() -> None:
    prior = Plan(
        id="p",
        run_id="r",
        goal_ids=["g1"],
        tasks=(Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),),
        edges=(),
        revision_index=1,
    )
    revised = Plan(
        id="p",
        run_id="r",
        goal_ids=["g1"],
        tasks=(
            Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(
                id="d1",
                title="agent: did work",
                discovered=True,
                kind=TaskKind.DISCOVERED,
            ),
        ),
        edges=(),
        revision_index=2,
    )
    # Edge-free DISCOVERED root alongside the OUTCOME — must validate.
    revised.validate(for_revision=True, prior=prior)


# ---------------------------------------------------------------------------
# Steerer-backed helpers (PlanReviser)
# ---------------------------------------------------------------------------


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


def _make_steerer(*, plan_mode: str = "forecast") -> DefaultSteerer:
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=True,
            plan_mode=plan_mode,
        )
    )
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    return steerer


def test_minimal_steer_evolution_preserves_task_kind() -> None:
    # A USER_STEER deterministic fallback must carry Task.kind across the
    # revision for BOTH preserved terminal tasks and cancelled pendings.
    steerer = _make_steerer(plan_mode="ledger")
    prior = Plan(
        id="p",
        run_id="r",
        goal_ids=["g1"],
        tasks=(
            Task(
                id="o_done",
                title="Summary delivered",
                kind=TaskKind.OUTCOME,
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="d_live",
                title="agent: in-flight work",
                discovered=True,
                kind=TaskKind.DISCOVERED,
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="o_pending",
                title="Translation delivered",
                kind=TaskKind.OUTCOME,
                status=TaskStatus.PENDING,
            ),
        ),
        edges=(),
        revision_index=2,
    )
    drift = DriftEvent(kind=DriftKind.USER_STEER, severity=DriftSeverity.WARNING, detail="steer")
    evolved = steerer.plans._build_minimal_steer_evolution(prior, drift)
    by_id = {t.id: t for t in evolved.tasks}
    # Terminal OUTCOME preserved verbatim (status + kind).
    assert by_id["o_done"].status is TaskStatus.COMPLETED
    assert by_id["o_done"].kind is TaskKind.OUTCOME
    # Non-terminal tasks cancelled — but kind survives.
    assert by_id["d_live"].status is TaskStatus.CANCELLED
    assert by_id["d_live"].kind is TaskKind.DISCOVERED
    assert by_id["o_pending"].status is TaskStatus.CANCELLED
    assert by_id["o_pending"].kind is TaskKind.OUTCOME
    # The fallback is always valid by construction.
    evolved.validate(for_revision=True, prior=prior)


def _growth_session() -> Session:
    return Session(
        run_id="r-grow",
        goals=[Goal(id="g", summary="exercise ledger growth")],
        plan=Plan(
            id="p-grow",
            run_id="r-grow",
            goal_ids=["g"],
            tasks=(Task(id="o1", title="Summary delivered", kind=TaskKind.OUTCOME),),
            edges=(),
            revision_index=1,
        ),
    )


def test_install_descriptive_growth_ledger_stamps_discovered_kind() -> None:
    steerer = _make_steerer(plan_mode="ledger")
    session = _growth_session()
    task = asyncio.run(
        steerer.plans.install_descriptive_growth(
            session,
            agent_name="debugger_agent",
            tool_args_json='{"request": "locate cherry tree files"}',
        )
    )
    assert task.discovered is True
    assert task.kind is TaskKind.DISCOVERED


def test_install_descriptive_growth_forecast_keeps_forecast_kind() -> None:
    steerer = _make_steerer(plan_mode="forecast")
    session = _growth_session()
    task = asyncio.run(
        steerer.plans.install_descriptive_growth(
            session,
            agent_name="debugger_agent",
            tool_args_json='{"request": "locate cherry tree files"}',
        )
    )
    # Forecast mode: discovered bool still set, but the ledger taxonomy is
    # unused → kind stays the FORECAST default (byte-identical to pre-PR-10).
    assert task.discovered is True
    assert task.kind is TaskKind.FORECAST


# ---------------------------------------------------------------------------
# StaticPlanner — prescriptive intent preserved
# ---------------------------------------------------------------------------


def test_static_planner_preserves_hand_authored_kind() -> None:
    template = Plan(
        id="static",
        run_id="",
        goal_ids=(),
        tasks=(
            Task(id="o1", title="Deliverable", kind=TaskKind.OUTCOME, contributes_to="g1"),
            Task(id="f1", title="Forecast step"),
        ),
        edges=(),
    )
    planner = StaticPlanner(template)
    plan = asyncio.run(
        planner.generate(goals=[Goal(id="g1", summary="x")], available_agents=["a"])
    )
    assert plan is not None
    by_id = {t.id: t for t in plan.tasks}
    assert by_id["o1"].kind is TaskKind.OUTCOME
    assert by_id["o1"].contributes_to == "g1"
    assert by_id["f1"].kind is TaskKind.FORECAST
