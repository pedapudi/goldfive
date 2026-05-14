"""Integration tests for the CAPABILITY_MISMATCH → descriptive-growth fallback.

Per goldfive#423 PR 2: when ``SteeringConfig.descriptive_growth_enabled``
is ``True`` and the structural capability detector returns a Rule C
verdict (out-of-DAG-order delegation — invoked agent's role stem
absent from the bound task but present in another PENDING task), the
ADK plugin's ``_maybe_emit_capability_mismatch`` synthesises a
``discovered=True`` task via
:meth:`~goldfive.plan_reviser.PlanReviser.install_descriptive_growth`
and re-pins the delegation to it INSTEAD of dispatching the Rule C
drift. With the flag off (the default) the legacy Rule C dispatch
fires as today.

Rule A and Rule B verdicts dispatch normally regardless of the flag —
those are skill-gap signals, not pin-mismatch signals.

Design ref: ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3, §7.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

# ADK plugin tests require the google.adk extra.
pytest.importorskip("google.adk")

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingDriftSteerer:
    """Wraps DefaultSteerer to capture drifts handed to handle_drift."""

    def __init__(self, steerer: DefaultSteerer) -> None:
        self._steerer = steerer
        self.handled_drifts: list[DriftEvent] = []
        self._real_handle = steerer.drift.handle_drift

        async def capture(drift: DriftEvent, session: Session) -> None:
            self.handled_drifts.append(drift)
            await self._real_handle(drift, session)

        steerer.drift.handle_drift = capture  # type: ignore[method-assign]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._steerer, name)


class _NullPlanner:
    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: Any,
        goals: list[Goal],
    ) -> Plan | None:
        return None


def _rule_c_plan() -> Plan:
    """Build the canonical Rule C shape from design doc §2.1.

    Coordinator delegates to ``debugger_agent`` but pin lands on
    ``find_presentation_files`` (the first eligible). The agent's role
    stem ``debugger`` is absent from the bound task and not present
    in any other PENDING task — actually that means Rule C WON'T
    fire on a generic find-files plan. We need a shape where the
    invoked agent's stem matches a different PENDING task.

    Better fixture: bind ``reviewer_agent`` to a ``draft`` task while
    a separate ``review`` task is also PENDING. The stem ``reviewer``
    → ``review`` is absent from ``draft`` and present in ``review``.
    """
    return Plan(
        id="p-rulec",
        run_id="r-rulec",
        goal_ids=["g-rulec"],
        tasks=[
            Task(
                id="draft_slides",
                title="draft slides",
                description="produce a slide draft",
                assignee_agent_id="reviewer_agent",  # mis-pinned
                status=TaskStatus.PENDING,
            ),
            Task(
                id="review_presentation",
                title="review presentation",
                description="critique the draft",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],  # both PENDING; both eligible
        revision_index=1,
    )


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r-rulec",
        goals=[Goal(id="g-rulec", summary="exercise rule c fallback")],
        plan=plan if plan is not None else _rule_c_plan(),
    )


def _build_ctx_and_plugin(session: Session, steerer: Any):
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="coordinator")
    bound_task = next(
        (
            t
            for t in session.plan.tasks
            if t.assignee_agent_id == "reviewer_agent"
        ),
        None,
    )
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=bound_task,
            tool_handlers={},
            host_agent_name="coordinator",
        )
    }
    plugin.set_active_context(state[SESSION_CONTEXT_STATE_KEY])
    return plugin, state


class _AgentToolStub:
    """Minimal AgentTool surface (a leaf-only tool list)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _InvokedAgentStub:
    """ADK agent stub with leaf tools."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Some leaf tools so Rule A doesn't fire (we want to exercise
        # specifically Rule C).
        self.tools = [_AgentToolStub("search_web"), _AgentToolStub("read_file")]


async def test_flag_off_dispatches_rule_c_drift_as_today() -> None:
    """With the feature flag OFF, the Rule C CAPABILITY_MISMATCH dispatches as today."""
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=False,  # FLAG OFF
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    recorder = _RecordingDriftSteerer(steerer)

    session = _make_session()
    # Pin the bound task as current_task_id so Strategy-2 lookup hits.
    session.current_task_id = "draft_slides"
    plugin, _state = _build_ctx_and_plugin(session, steerer)

    # Invoke the capability check directly.
    ctx = plugin._active_ctx
    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_InvokedAgentStub("reviewer_agent"),
        invoked_agent_name="reviewer_agent",
        invocation_id="inv-test-1",
        tool_args_json='{"request": "review the draft"}',
        delegation_event_id="evt-1",
    )

    # Rule C drift was dispatched.
    rule_c = [
        d
        for d in recorder.handled_drifts
        if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert (
        len(rule_c) == 1
    ), f"with flag OFF, expected 1 Rule C drift; got {len(rule_c)}"
    # No discovered task synthesised.
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert (
        discovered == []
    ), f"flag OFF must not grow discovered tasks; got {[t.id for t in discovered]}"


async def test_flag_on_grows_discovered_task_and_suppresses_rule_c() -> None:
    """With the feature flag ON, Rule C is replaced by descriptive growth."""
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=True,  # FLAG ON
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    recorder = _RecordingDriftSteerer(steerer)

    session = _make_session()
    session.current_task_id = "draft_slides"
    plugin, _state = _build_ctx_and_plugin(session, steerer)

    ctx = plugin._active_ctx
    prior_task_count = len(session.plan.tasks)

    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_InvokedAgentStub("reviewer_agent"),
        invoked_agent_name="reviewer_agent",
        invocation_id="inv-test-2",
        tool_args_json='{"request": "review the draft"}',
        delegation_event_id="evt-2",
    )

    # Rule C drift was SUPPRESSED.
    rule_c = [
        d
        for d in recorder.handled_drifts
        if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert (
        rule_c == []
    ), f"flag ON must suppress Rule C drift; got {[d.detail for d in rule_c]}"

    # Discovered task was synthesised.
    assert session.plan is not None
    assert len(session.plan.tasks) == prior_task_count + 1
    discovered = [
        t for t in session.plan.tasks if getattr(t, "discovered", False)
    ]
    assert len(discovered) == 1, (
        f"flag ON must synthesise 1 discovered task; got {len(discovered)}"
    )
    new_task = discovered[0]
    assert new_task.assignee_agent_id == "reviewer_agent"
    assert new_task.discovery_identity_hash != ""

    # Pin moved to the new discovered task.
    assert session.current_task_id == new_task.id


async def test_flag_on_rule_a_still_fires() -> None:
    """Rule A (leaf-task with AgentTool-only agent) is unaffected by the flag.

    The descriptive-growth fallback gates ONLY on Rule C detail
    substring. Rule A — coordinator-style leaf-assignment — has a
    different detail prefix and still dispatches normally.
    """

    class _AgentToolWrapper:
        """An AgentTool wrapper as detected by ``is_agent_tool``."""

        def __init__(self, name: str, agent_name: str) -> None:
            self.name = name
            self.agent = type("_A", (), {"name": agent_name})()

    class _OnlyAgentToolsAgent:
        def __init__(self) -> None:
            self.name = "coordinator_b"
            # All AgentTool wrappers — Rule A's "no leaf capability"
            # condition.
            self.tools = [
                _AgentToolWrapper("delegate_a", "agent_a"),
                _AgentToolWrapper("delegate_b", "agent_b"),
            ]

    # Plan where bound task is a clear leaf task (no delegation markers
    # in title/description).
    plan = Plan(
        id="p-rulea",
        run_id="r-rulea",
        goal_ids=["g-rulea"],
        tasks=[
            Task(
                id="write_report",
                title="write final report",
                description="produce the leaf-shaped artifact",
                assignee_agent_id="coordinator_b",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    session = Session(
        run_id="r-rulea",
        goals=[Goal(id="g-rulea", summary="rule a unchanged")],
        plan=plan,
    )
    session.current_task_id = "write_report"

    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False, descriptive_growth_enabled=True
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    recorder = _RecordingDriftSteerer(steerer)

    from goldfive.adapters._adk_plugin import (
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="coordinator")
    bound_task = plan.tasks[0]
    plugin.set_active_context(
        SessionContext(
            session=session,
            steerer=steerer,
            task=bound_task,
            tool_handlers={},
            host_agent_name="coordinator",
        )
    )

    ctx = plugin._active_ctx
    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_OnlyAgentToolsAgent(),
        invoked_agent_name="coordinator_b",
        invocation_id="inv-test-rulea",
        tool_args_json='{"request": "go write the report"}',
        delegation_event_id="evt-rulea",
    )

    # Rule A drift was dispatched normally (NOT suppressed by the
    # descriptive-growth gate).
    cap_drifts = [
        d
        for d in recorder.handled_drifts
        if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert len(cap_drifts) == 1, (
        f"flag-on should NOT suppress Rule A; got {len(cap_drifts)} drifts"
    )
    assert "has only AgentTool" in cap_drifts[0].detail, (
        f"expected Rule A detail, got: {cap_drifts[0].detail!r}"
    )
    # No discovered task synthesised.
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert discovered == [], (
        f"Rule A path must NOT synthesise discovered tasks; "
        f"got {[t.id for t in discovered]}"
    )


async def test_flag_on_dedups_repeated_unmatched_delegations() -> None:
    """Repeated same-args delegations grow the plan once, then re-pin.

    Cherry-tree run from design doc §2.1 motivating evidence: 20+
    debugger_agent delegations dedup to a single discovered task.
    """
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False, descriptive_growth_enabled=True
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    _RecordingDriftSteerer(steerer)  # silence drift dispatch

    session = _make_session()
    session.current_task_id = "draft_slides"
    plugin, _state = _build_ctx_and_plugin(session, steerer)

    ctx = plugin._active_ctx

    # Fire 10 delegations to the same (agent, tool_args).
    for i in range(10):
        await plugin._maybe_emit_capability_mismatch(
            ctx=ctx,
            invoked_agent=_InvokedAgentStub("reviewer_agent"),
            invoked_agent_name="reviewer_agent",
            invocation_id=f"inv-{i}",
            tool_args_json='{"request": "review the draft"}',
            delegation_event_id=f"evt-{i}",
        )

    # Only one discovered task.
    assert session.plan is not None
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert len(discovered) == 1, (
        f"10 same-args delegations must dedup to 1 discovered task; "
        f"got {len(discovered)}"
    )
    # current_task_id pinned to that one discovered task.
    assert session.current_task_id == discovered[0].id
