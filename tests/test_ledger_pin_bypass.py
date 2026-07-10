"""Ledger plan mode bypasses the delegation pin tiers (PR 10).

AGENCY-PRESERVATION.md Stage 3 PR 10: in ledger mode the OUTCOME tasks
are goal-anchored deliverables, never agent-behaviour forecasts, so a
delegation must NEVER be pinned to one. ``_maybe_pin_delegation_task``
therefore bypasses the tier-1/2 matching entirely and routes every
unforecast delegation through dedup-hash → descriptive growth → pin,
stamping the grown task ``kind=DISCOVERED``.

The control test proves the bypass is real: the SAME plan + delegation in
forecast mode pins to the stem-matching task via tier 2 (no growth).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskKind,
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


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeLeafTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSubAgent:
    def __init__(self, name: str, tools: list[Any] | None = None) -> None:
        self.name = name
        self.tools = list(tools or [_FakeLeafTool("do_work")])


class _FakeAgentTool:
    def __init__(self, sub_agent: _FakeSubAgent) -> None:
        self.agent = sub_agent
        self.name = sub_agent.name


class _FakeInvocationContext:
    def __init__(self, session_state: dict, agent_name: str, inv_id: str = "inv-1") -> None:
        class _ADKSession:
            def __init__(self, state: dict) -> None:
                self.state = state

        self.session = _ADKSession(session_state)
        self.invocation_id = inv_id
        self.agent = _FakeAgent(agent_name)


class _FakeToolContext:
    def __init__(self, inv_ctx: Any, function_call_id: str = "fc-1") -> None:
        self._invocation_context = inv_ctx
        self.function_call_id = function_call_id


def _make_steerer(*, plan_mode: str) -> tuple[DefaultSteerer, _ListSink]:
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=True,
            plan_mode=plan_mode,
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    return steerer, sink


def _outcome_plan() -> Plan:
    """A ledger of OUTCOME deliverables; one title carries the stem
    ``reviewer`` so a forecast-mode tier-2 match would pin to it."""
    return Plan(
        id="p-led",
        run_id="r-led",
        goal_ids=["g-led"],
        tasks=(
            Task(id="summary", title="Summary delivered", kind=TaskKind.OUTCOME),
            Task(id="review", title="reviewer sign-off delivered", kind=TaskKind.OUTCOME),
        ),
        edges=(),
        revision_index=1,
    )


def _make_session() -> Session:
    return Session(
        run_id="r-led",
        goals=[Goal(id="g-led", summary="deliver the deck")],
        plan=_outcome_plan(),
    )


def _wire_plugin(session: Session, steerer: Any):
    plugin = make_adk_plugin(host_agent_name="coordinator")
    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coordinator",
    )
    plugin.set_active_context(ctx_obj)
    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    return plugin, adk_state


async def _dispatch(plugin: Any, adk_state: dict, *, agent: _FakeSubAgent, tool_args: dict) -> Any:
    inv_ctx = _FakeInvocationContext(adk_state, "coordinator")
    tool_context = _FakeToolContext(inv_ctx)
    return await plugin.before_tool_callback(
        tool=_FakeAgentTool(agent),
        tool_args=tool_args,
        tool_context=tool_context,
    )


def _discovered(plan: Plan | None) -> list[Task]:
    return [t for t in (plan.tasks if plan else ()) if getattr(t, "discovered", False)]


async def test_ledger_bypasses_tier2_and_grows_discovered() -> None:
    """Ledger mode: a stem-matching delegation grows a DISCOVERED task
    instead of pinning to the matching OUTCOME deliverable."""
    steerer, _sink = _make_steerer(plan_mode="ledger")
    session = _make_session()
    plugin, adk_state = _wire_plugin(session, steerer)

    # ``reviewer_agent`` stem ``reviewer`` IS present in the "review"
    # OUTCOME task title — forecast tier 2 would pin to it.
    await _dispatch(
        plugin,
        adk_state,
        agent=_FakeSubAgent("reviewer_agent"),
        tool_args={"request": "review the deck"},
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, "ledger mode must grow exactly one discovered task"
    new_task = discovered[0]
    assert new_task.kind is TaskKind.DISCOVERED
    assert new_task.assignee_agent_id == "reviewer_agent"
    # The pin landed on the discovered task — NOT on the stem-matching
    # OUTCOME deliverable.
    assert session.current_task_id == new_task.id
    review = next(t for t in session.plan.tasks if t.id == "review")
    assert review.assignee_agent_id == "", "OUTCOME task must NOT be pinned in ledger mode"
    assert review.kind is TaskKind.OUTCOME


async def test_ledger_dedup_repins_without_second_growth() -> None:
    """Repeated delegations to the same (agent, args) dedup onto one
    DISCOVERED task even though the tiers are bypassed."""
    steerer, _sink = _make_steerer(plan_mode="ledger")
    session = _make_session()
    plugin, adk_state = _wire_plugin(session, steerer)

    for _ in range(3):
        await _dispatch(
            plugin,
            adk_state,
            agent=_FakeSubAgent("reviewer_agent"),
            tool_args={"request": "review the deck"},
        )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, f"dedup failed: {[t.id for t in discovered]}"
    assert discovered[0].kind is TaskKind.DISCOVERED


async def test_forecast_control_pins_via_tier2_without_growth() -> None:
    """Control: the SAME plan + delegation in forecast mode pins to the
    stem-matching task via tier 2 — proving the ledger bypass is real."""
    steerer, _sink = _make_steerer(plan_mode="forecast")
    session = _make_session()
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch(
        plugin,
        adk_state,
        agent=_FakeSubAgent("reviewer_agent"),
        tool_args={"request": "review the deck"},
    )

    # Forecast tier-2 stem match pins to the "review" task; no growth.
    assert _discovered(session.plan) == []
    assert session.current_task_id == "review"
    review = next(t for t in session.plan.tasks if t.id == "review")
    assert review.assignee_agent_id == "reviewer_agent"


def _forecast_plan() -> Plan:
    """The SAME titles as ``_outcome_plan`` but FORECAST-shaped — a
    hand-authored StaticPlanner-style plan (no ``kind`` set)."""
    return Plan(
        id="p-static",
        run_id="r-led",
        goal_ids=["g-led"],
        tasks=(
            Task(id="summary", title="Summary delivered"),
            Task(id="review", title="reviewer sign-off delivered"),
        ),
        edges=(),
        revision_index=1,
    )


async def test_ledger_config_with_forecast_shaped_plan_keeps_pin_tiers() -> None:
    """The documented contract: "StaticPlanner users keep forecast
    semantics — a hand-authored plan is genuine prescriptive intent."
    ``plan_mode="ledger"`` with a FORECAST-shaped plan (no OUTCOME /
    DISCOVERED task) must NOT bypass the pin tiers: the delegation pins to
    the stem-matching forecast task instead of silently stripping pinning
    + drift-repair from the hand-authored plan."""
    steerer, _sink = _make_steerer(plan_mode="ledger")
    session = Session(
        run_id="r-led",
        goals=[Goal(id="g-led", summary="deliver the deck")],
        plan=_forecast_plan(),
    )
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch(
        plugin,
        adk_state,
        agent=_FakeSubAgent("reviewer_agent"),
        tool_args={"request": "review the deck"},
    )

    # No ledger bypass: the tier-2 stem match pins to "review"; the
    # hand-authored tasks keep their forecast semantics.
    assert session.current_task_id == "review"
    review = next(t for t in session.plan.tasks if t.id == "review")
    assert review.assignee_agent_id == "reviewer_agent"
    assert review.kind is TaskKind.FORECAST
    # No OUTCOME task was skipped-over into growth.
    assert [t for t in session.plan.tasks if getattr(t, "discovered", False)] == []
