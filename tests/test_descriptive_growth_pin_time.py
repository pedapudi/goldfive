"""Descriptive growth at PIN time (goldfive#423 / AGENCY-PRESERVATION.md PR 2).

Pins the completed #423 contract per
``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3: the growth trigger
lives in the delegation pin, not the CAPABILITY_MISMATCH verdict path.
When ``SteeringConfig.descriptive_growth_enabled`` is ON (the default):

* a tier-1 (required-tools cover) or tier-2 (agent-name stem) hit pins
  the forecast task exactly as before — NO growth;
* a tier-1/2 miss grows a ``discovered=True`` task via
  :meth:`~goldfive.plan_reviser.PlanReviser.install_descriptive_growth`
  and pins it — the tier-3 topic-args scorer is NOT invoked and no
  CAPABILITY_MISMATCH rule ever sees a mispinned forecast task (the
  Rule-A-bypass gap from e2e session ``2d27ff4a``);
* repeated delegations to the same (agent, args-token-set) dedup onto
  ONE discovered task (§11.1); different args grow distinct tasks;
* the flow works identically in observation mode (the goldfive#258
  ``NEW_WORK_DISCOVERED`` carve-out) — growth is descriptive, not
  corrective;
* with the flag OFF (or a steerer that lacks the growth machinery),
  the legacy pre-#423 pin — including the tier-3 scorer — is
  preserved verbatim until its deletion (AGENCY-PRESERVATION.md PR 13).

The ``2d27ff4a`` regression test reproduces the cherry-tree failure
shape end-to-end at the plugin level: N repeated delegations to the
same unmatched agent with identical args must produce exactly one
discovered task, zero CAPABILITY_MISMATCH drifts, and zero planner
refines.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

# Plugin-level tests require the google.adk extra.
pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingPlanner:
    """Planner that records refine calls (the 2d27ff4a storm metric)."""

    def __init__(self) -> None:
        self.refine_calls: list[Any] = []

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
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append(drift)
        return None


def _capture_drifts(steerer: DefaultSteerer) -> list[DriftEvent]:
    """Wrap ``steerer.drift.handle_drift`` to record dispatched drifts."""
    handled: list[DriftEvent] = []
    real_handle = steerer.drift.handle_drift

    async def capture(drift: DriftEvent, session: Session) -> None:
        handled.append(drift)
        await real_handle(drift, session)

    steerer.drift.handle_drift = capture  # type: ignore[method-assign]
    return handled


def _make_steerer(
    *,
    descriptive_growth_enabled: bool = True,
    observation_only: bool = False,
) -> tuple[DefaultSteerer, _RecordingPlanner, _ListSink]:
    planner = _RecordingPlanner()
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=observation_only,
            descriptive_growth_enabled=descriptive_growth_enabled,
        )
    )
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, planner, sink


def _cherry_tree_plan() -> Plan:
    """The 2d27ff4a forecast shape (design doc §2.1 / §8).

    Three forecast tasks; only ``find_presentation_files`` is
    DAG-ready. None of the titles carries the stem ``debugger``, so a
    delegation to ``debugger_agent`` misses tier 1 and tier 2.
    """
    return Plan(
        id="p-cherry",
        run_id="r-cherry",
        goal_ids=["g-cherry"],
        tasks=[
            Task(id="find_presentation_files", title="Find the presentation files"),
            Task(id="read_presentation", title="Read the presentation"),
            Task(id="summarise_presentation", title="Summarise the presentation"),
        ],
        edges=[
            TaskEdge(
                from_task_id="find_presentation_files",
                to_task_id="read_presentation",
            ),
            TaskEdge(
                from_task_id="read_presentation",
                to_task_id="summarise_presentation",
            ),
        ],
        revision_index=1,
    )


def _make_session(plan: Plan) -> Session:
    return Session(
        run_id=plan.run_id,
        goals=[Goal(id=g, summary="summarise the pothos presentation") for g in plan.goal_ids],
        plan=plan,
    )


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeLeafTool:
    """FunctionTool stand-in: ``.name`` but no ``.agent``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSubAgent:
    """The nested ADK agent an AgentTool wraps (``.name`` + ``.tools``)."""

    def __init__(self, name: str, tools: list[Any] | None = None) -> None:
        self.name = name
        self.tools = list(tools or [_FakeLeafTool("find_files")])


class _FakeAgentTool:
    """ADK AgentTool stand-in: has both ``.agent`` and ``.name``."""

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


def _wire_plugin(session: Session, steerer: Any):
    """Build plugin + SessionContext + ADK-state plumbing for callbacks."""
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


async def _dispatch_delegation(
    plugin: Any,
    adk_state: dict,
    *,
    agent: _FakeSubAgent,
    tool_args: dict[str, Any],
    fc_id: str = "fc-1",
    inv_id: str = "inv-1",
) -> Any:
    """Drive ``before_tool_callback`` for one AgentTool dispatch."""
    inv_ctx = _FakeInvocationContext(adk_state, "coordinator", inv_id)
    tool_context = _FakeToolContext(inv_ctx, function_call_id=fc_id)
    return await plugin.before_tool_callback(
        tool=_FakeAgentTool(agent),
        tool_args=tool_args,
        tool_context=tool_context,
    )


def _discovered(plan: Plan | None) -> list[Task]:
    return [t for t in (plan.tasks if plan else ()) if getattr(t, "discovered", False)]


# ---------------------------------------------------------------------------
# (a) Tier-1 / tier-2 hits do NOT grow — pin as before.
# ---------------------------------------------------------------------------


async def test_tier1_required_tools_hit_pins_without_growth() -> None:
    """Required-tools cover (tier 1) still wins under growth mode."""
    steerer, planner, _sink = _make_steerer()
    plan = Plan(
        id="p-t1",
        run_id="r-t1",
        goal_ids=["g-t1"],
        tasks=[
            Task(id="B", title="general work"),
            Task(id="A", title="apply patch", required_tools=("patch_file",)),
        ],
        edges=[],
        revision_index=1,
    )
    session = _make_session(plan)
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("patcher", tools=[_FakeLeafTool("patch_file")]),
        tool_args={"x": "go"},
    )

    assert session.current_task_id == "A", "tier-1 cover must pin task A"
    assert _discovered(session.plan) == [], "a tier-1 hit must not grow"
    assert session.plan.revision_index == 1
    assert planner.refine_calls == []


async def test_tier2_stem_hit_pins_without_growth() -> None:
    """Agent-name stem match (tier 2) still wins under growth mode."""
    steerer, planner, _sink = _make_steerer()
    plan = Plan(
        id="p-t2",
        run_id="r-t2",
        goal_ids=["g-t2"],
        tasks=[
            Task(id="outline_presentation", title="Outline the presentation"),
            Task(id="review_presentation", title="Review the presentation"),
        ],
        edges=[],
        revision_index=1,
    )
    session = _make_session(plan)
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("reviewer_agent"),
        tool_args={"request": "please proceed"},
    )

    assert session.current_task_id == "review_presentation"
    assert _discovered(session.plan) == [], "a tier-2 hit must not grow"
    assert session.plan.revision_index == 1
    assert planner.refine_calls == []


# ---------------------------------------------------------------------------
# (b) Tier-1/2 miss → growth + pin; no tier-3 scorer; no CAPABILITY_MISMATCH.
# ---------------------------------------------------------------------------


async def test_tier12_miss_grows_and_pins_without_tier3_or_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The §4.3 flow: miss → grow discovered task → pin it.

    Asserts the three PR-2 binding negatives in one shot: the tier-3
    topic-args scorer is never consulted, no CAPABILITY_MISMATCH is
    dispatched, and the planner is never asked to refine.
    """
    import goldfive.adapters._adk_plugin as plugin_mod

    scorer_calls: list[Any] = []
    real_scorer = plugin_mod._score_candidates_by_args

    def recording_scorer(candidates: list[Any], tool_args: Any) -> Any:
        scorer_calls.append((candidates, tool_args))
        return real_scorer(candidates, tool_args)

    monkeypatch.setattr(
        plugin_mod, "_score_candidates_by_args", recording_scorer
    )

    steerer, planner, sink = _make_steerer()
    handled = _capture_drifts(steerer)
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, (
        f"tier-1/2 miss must grow exactly one discovered task; got "
        f"{[t.id for t in session.plan.tasks]}"
    )
    new_task = discovered[0]
    assert new_task.assignee_agent_id == "debugger_agent"
    assert new_task.discovery_identity_hash != ""
    assert new_task.title.startswith("debugger_agent:")
    # The pin landed on the discovered task.
    assert session.current_task_id == new_task.id
    # The DelegationObserved emit carries the discovered task id (it
    # runs after the growth, by construction).
    delegations = [
        e.delegation_observed
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "delegation_observed"
    ]
    assert len(delegations) == 1
    assert delegations[0].task_id == new_task.id
    # The forecast tasks were not touched.
    assert {t.id for t in session.plan.tasks if not t.discovered} == {
        "find_presentation_files",
        "read_presentation",
        "summarise_presentation",
    }
    # Binding negatives.
    assert scorer_calls == [], (
        "the tier-3 topic-args scorer must not run in the default "
        f"(growth) path; called with {scorer_calls}"
    )
    mismatches = [d for d in handled if d.kind is DriftKind.CAPABILITY_MISMATCH]
    assert mismatches == [], (
        f"growth at pin time must pre-empt CAPABILITY_MISMATCH; got "
        f"{[d.detail for d in mismatches]}"
    )
    assert planner.refine_calls == [], "no refine may fire on a tier-1/2 miss"


async def test_growth_lands_in_observation_mode_too() -> None:
    """The goldfive#258 discovery carve-out: growth is descriptive, so
    it lands a REAL plan revision under ``observation_only=True`` as
    well — the plan view must reflect what actually happened in both
    modes (design doc §6.2).
    """
    steerer, planner, _sink = _make_steerer(observation_only=True)
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, "observation mode must still grow the ledger"
    assert session.current_task_id == discovered[0].id
    assert session.plan.revision_index == 2
    assert planner.refine_calls == []


# ---------------------------------------------------------------------------
# Dedup granularity (§11.1): same args re-pin; different args grow anew.
# ---------------------------------------------------------------------------


async def test_different_args_grow_distinct_discovered_tasks() -> None:
    """Per-(agent, args-token-set) granularity: a second delegation to
    the same agent with DIFFERENT args is a new unit of work and grows
    a second discovered task (the §11.1 ``web_developer=2`` case).
    """
    steerer, _planner, _sink = _make_steerer()
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
        fc_id="fc-1",
    )
    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "inspect webserver logs"},
        fc_id="fc-2",
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 2, (
        f"different args must grow distinct tasks; got "
        f"{[t.title for t in discovered]}"
    )
    hashes = {t.discovery_identity_hash for t in discovered}
    assert len(hashes) == 2


async def test_same_args_re_pin_while_discovered_task_running() -> None:
    """The step-0 hash lookup re-pins onto a RUNNING discovered task —
    the dedup window spans the task's whole non-terminal lifetime, not
    just while it is PENDING (§11.1 TTL).
    """
    from goldfive.types import (
        channel_processor_active,
        set_session_plan,
        with_task_status,
    )

    steerer, _planner, _sink = _make_steerer()
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
        fc_id="fc-1",
    )
    first = _discovered(session.plan)[0]
    # Simulate the sub-invocation starting: discovered task → RUNNING.
    with channel_processor_active():
        set_session_plan(
            session, with_task_status(session.plan, first.id, TaskStatus.RUNNING)
        )

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
        fc_id="fc-2",
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, "same-args delegation must re-pin, not re-grow"
    assert session.current_task_id == first.id


# ---------------------------------------------------------------------------
# (e) The 2d27ff4a regression shape.
# ---------------------------------------------------------------------------


async def test_2d27ff4a_repeated_delegations_one_task_no_drift() -> None:
    """Cherry-tree regression (e2e session 2d27ff4a, 2026-05-13).

    A coordinator delegates to ``debugger_agent`` 20 times with the
    same args before touching the planned agents. Pre-#423: every
    delegation mispinned ``find_presentation_files`` and fired
    CAPABILITY_MISMATCH → refine (~20 spurious refines). Post-PR-2:

    * exactly ONE discovered task (per-(agent, args-token-set) dedup);
    * ZERO CAPABILITY_MISMATCH drifts (growth runs before the rules
      and the discovered bound task skips them — §11.4(a));
    * ZERO planner refines.
    """
    steerer, planner, sink = _make_steerer()
    handled = _capture_drifts(steerer)
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    for i in range(20):
        # One delegation per coordinator invocation, matching the real
        # session (each pre-fix refine restarted the invocation). The
        # plugin's before_run normally resets this counter per
        # invocation; reset it here so the goldfive#130
        # RUNAWAY_DELEGATION per-invocation backstop — which design
        # doc §11.5 deliberately keeps orthogonal to descriptive
        # growth — doesn't fold a second signal into this regression.
        plugin._agent_tool_spawn_count = 0
        await _dispatch_delegation(
            plugin,
            adk_state,
            agent=_FakeSubAgent("debugger_agent"),
            tool_args={"request": "locate cherry tree files"},
            fc_id=f"fc-{i}",
            inv_id=f"inv-{i}",
        )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, (
        f"20 same-args delegations must dedup to exactly 1 discovered "
        f"task; got {len(discovered)}: {[t.id for t in discovered]}"
    )
    assert session.current_task_id == discovered[0].id
    mismatches = [d for d in handled if d.kind is DriftKind.CAPABILITY_MISMATCH]
    assert mismatches == [], (
        f"2d27ff4a regression: {len(mismatches)} CAPABILITY_MISMATCH "
        f"drift(s) fired; first: {mismatches[0].detail if mismatches else ''}"
    )
    assert planner.refine_calls == [], (
        f"2d27ff4a regression: {len(planner.refine_calls)} refine(s) fired"
    )
    # Exactly one NEW_WORK_DISCOVERED revision landed (rev 1 → 2).
    assert session.plan.revision_index == 2


# ---------------------------------------------------------------------------
# Legacy escape hatches.
# ---------------------------------------------------------------------------


async def test_flag_off_keeps_legacy_tier3_pin() -> None:
    """``descriptive_growth_enabled=False`` restores the pre-#423 pin:
    the tier-3 topo-order fallback binds the first eligible forecast
    task and nothing grows. (The tier-3 scorer stays reachable ONLY on
    this path; it is deleted in AGENCY-PRESERVATION.md PR 13.)
    """
    steerer, _planner, _sink = _make_steerer(descriptive_growth_enabled=False)
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
    )

    assert _discovered(session.plan) == [], "flag off must not grow"
    # Legacy behaviour: single eligible task bound by the shortcut.
    assert session.current_task_id == "find_presentation_files"


async def test_steerer_without_growth_helper_keeps_legacy_pin() -> None:
    """A steerer whose ``plans`` lacks ``install_descriptive_growth``
    keeps the legacy pin even with the flag on — signalling growth
    with nobody to perform it would strand the delegation unpinned.
    """

    class _BarePlans:
        pass

    steerer, _planner, _sink = _make_steerer()
    steerer.plans = _BarePlans()  # type: ignore[assignment]
    session = _make_session(_cherry_tree_plan())
    plugin, adk_state = _wire_plugin(session, steerer)

    await _dispatch_delegation(
        plugin,
        adk_state,
        agent=_FakeSubAgent("debugger_agent"),
        tool_args={"request": "locate cherry tree files"},
    )

    assert _discovered(session.plan) == []
    assert session.current_task_id == "find_presentation_files"


# ---------------------------------------------------------------------------
# Reconciler growth (transfer_to_agent-style trees) — correctness req (d).
# ---------------------------------------------------------------------------


async def test_reconciler_unmatched_agent_grows_and_claims() -> None:
    """``PlanReconciler.on_before_agent`` for an off-plan agent grows a
    discovered task (degraded per-(agent, "") hash — before_agent
    observations carry no tool args; §9 forward-compat) and claims it
    RUNNING; ``on_after_agent`` then closes it COMPLETED.
    """
    from goldfive.reconciler import PlanReconciler
    from goldfive.types import discovery_identity_hash

    steerer, _planner, _sink = _make_steerer()
    session = _make_session(_cherry_tree_plan())
    rec = PlanReconciler(
        session=session, steerer=steerer, host_agent_name="coordinator"
    )

    await rec.on_before_agent(agent_name="scout_agent", invocation_id="inv-s1")

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, "unmatched agent must grow the ledger"
    grown = discovered[0]
    assert grown.assignee_agent_id == "scout_agent"
    # Degraded hash: per-(agent_name, "") — no tool args at this hook.
    assert grown.discovery_identity_hash == discovery_identity_hash(
        "scout_agent", None
    )
    # Claimed RUNNING, and no divergence was recorded for the agent.
    live = next(t for t in session.plan.tasks if t.id == grown.id)
    assert live.status is TaskStatus.RUNNING
    assert rec.divergence_events == []

    await rec.on_after_agent(agent_name="scout_agent", invocation_id="inv-s1")
    closed = next(t for t in session.plan.tasks if t.id == grown.id)
    assert closed.status is TaskStatus.COMPLETED


async def test_reconciler_degraded_hash_dedups_concurrent_observations() -> None:
    """Two reconcilers (parallel invocations) observing the same
    off-plan agent produce ONE discovered task — the degraded
    per-(agent, "") hash dedups inside the plan lock (§11.6
    linearisability applies to the reconciler trigger too).
    """
    import asyncio

    from goldfive.reconciler import PlanReconciler

    steerer, _planner, _sink = _make_steerer()
    session = _make_session(_cherry_tree_plan())
    rec_a = PlanReconciler(
        session=session, steerer=steerer, host_agent_name="coordinator"
    )
    rec_b = PlanReconciler(
        session=session, steerer=steerer, host_agent_name="coordinator"
    )

    await asyncio.gather(
        rec_a.on_before_agent(agent_name="scout_agent", invocation_id="inv-a"),
        rec_b.on_before_agent(agent_name="scout_agent", invocation_id="inv-b"),
    )

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, (
        f"concurrent unmatched observations must dedup to 1 task; got "
        f"{[t.id for t in discovered]}"
    )


async def test_reconciler_flag_off_keeps_divergence_record() -> None:
    """With growth disabled the reconciler keeps the legacy behaviour:
    a divergence record on ``divergence_events``, no plan mutation.
    """
    from goldfive.reconciler import PlanReconciler

    steerer, _planner, _sink = _make_steerer(descriptive_growth_enabled=False)
    session = _make_session(_cherry_tree_plan())
    rec = PlanReconciler(
        session=session, steerer=steerer, host_agent_name="coordinator"
    )

    await rec.on_before_agent(agent_name="scout_agent", invocation_id="inv-s1")

    assert _discovered(session.plan) == []
    assert len(rec.divergence_events) == 1


# ---------------------------------------------------------------------------
# Real dispatch path: full ADKAdapter.invoke with real ADK agents.
# ---------------------------------------------------------------------------


async def test_growth_fires_through_real_adk_adapter_invoke() -> None:
    """End-to-end through the REAL dispatch path (no direct callback
    calls): a genuine ADK coordinator + AgentTool tree driven by
    ``ADKAdapter.invoke`` with a ``DefaultSteerer``. The coordinator
    delegates once to an agent no forecast task matches — the plan
    must grow a discovered task and pin it, proving the growth call
    site is live middleware on the production callback path.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import FunctionTool
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types as genai_types

    class _OneShotDelegator(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            if self._step == 1:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id="c1",
                                    name="debugger_agent",
                                    args={"request": "locate cherry tree files"},
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="done")],
                    ),
                    turn_complete=True,
                )

    class _Quiet(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    def find_files(path: str) -> dict[str, Any]:  # noqa: ARG001
        return {"ok": True}

    debugger = Agent(
        name="debugger_agent",
        model=_Quiet(),
        instruction="",
        tools=[FunctionTool(find_files)],
    )
    coord = Agent(
        name="coord",
        model=_OneShotDelegator(),
        instruction="",
        tools=[AgentTool(debugger)],
    )

    from goldfive.adapters.adk import ADKAdapter

    adapter = ADKAdapter(coord)
    steerer, planner, _sink = _make_steerer()
    handled = _capture_drifts(steerer)
    adapter.bind_steerer(steerer)

    plan = _cherry_tree_plan()
    coord_task = Task(
        id="t-coord",
        title="Coordinate the work",
        assignee_agent_id="coord",
        status=TaskStatus.RUNNING,
    )
    plan = Plan(
        id=plan.id,
        run_id=plan.run_id,
        goal_ids=list(plan.goal_ids),
        tasks=[coord_task, *plan.tasks],
        edges=list(plan.edges),
        revision_index=plan.revision_index,
    )
    session = _make_session(plan)

    await adapter.invoke(task=coord_task, session=session)

    discovered = _discovered(session.plan)
    assert len(discovered) == 1, (
        f"real ADK dispatch must grow exactly one discovered task; "
        f"plan tasks: {[t.id for t in session.plan.tasks]}"
    )
    assert discovered[0].assignee_agent_id == "debugger_agent"
    mismatches = [d for d in handled if d.kind is DriftKind.CAPABILITY_MISMATCH]
    assert mismatches == [], (
        f"no CAPABILITY_MISMATCH may fire on the grown delegation; got "
        f"{[d.detail for d in mismatches]}"
    )
    assert planner.refine_calls == []
