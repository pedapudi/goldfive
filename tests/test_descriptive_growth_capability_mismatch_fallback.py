"""CAPABILITY_MISMATCH verdict path after the descriptive-growth move.

History: goldfive#423 Phase 1 wired a Rule C → descriptive-growth
fallback INTO ``_maybe_emit_capability_mismatch`` (this file originally
pinned that behaviour). goldfive#423 / AGENCY-PRESERVATION.md Stage 1
PR 2 moved the growth trigger to PIN time
(``_maybe_pin_delegation_task`` tier-1/2 miss →
:meth:`~goldfive.plan_reviser.PlanReviser.install_descriptive_growth`),
which runs BEFORE any capability rule, and soft-retired Rule C behind
``GOLDFIVE_CAPABILITY_RULE_C``.

This file now pins the VERDICT-path side of that contract:

* the verdict path never grows the plan (growth lives at pin time —
  see ``tests/test_descriptive_growth_pin_time.py``);
* Rule C is silent by default and dispatches as a plain drift under
  the explicit escape hatch (no growth absorption);
* Rule A still dispatches normally for non-discovered bound tasks;
* discovered bound tasks skip the capability rules entirely
  (design doc §11.4 resolution (a)).

Test migration map (every original assertion re-pointed, none deleted
silently — see the PR body for the one-line justifications):

* ``test_flag_off_dispatches_rule_c_drift_as_today`` →
  ``test_rule_c_dispatches_under_escape_hatch_without_growth``.
* ``test_flag_on_grows_discovered_task_and_suppresses_rule_c`` →
  ``test_verdict_path_no_longer_grows`` (inverted: the growth the old
  test asserted here now happens at pin time; the pin-time positive
  lives in ``test_descriptive_growth_pin_time.py``).
* ``test_flag_on_rule_a_still_fires`` → kept (signature updated: the
  growth-threading kwargs were removed from
  ``_maybe_emit_capability_mismatch``).
* ``test_flag_on_dedups_repeated_unmatched_delegations`` → migrated to
  ``test_descriptive_growth_pin_time.py::test_2d27ff4a_repeated_delegations_one_task_no_drift``
  (the dedup now exercises the pin-time path, which is where the 20×
  cherry-tree delegations actually arrive).

Design ref: ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3, §7, §11.4.
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

    Bind ``reviewer_agent`` to a ``draft`` task while a separate
    ``review`` task is also PENDING. The stem ``reviewer`` → ``review``
    is absent from ``draft`` and present in ``review`` — the Rule C
    trigger shape.
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
        goals=[Goal(id="g-rulec", summary="exercise rule c retirement")],
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
    """Minimal leaf-tool surface (just a ``.name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _InvokedAgentStub:
    """ADK agent stub with leaf tools."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Some leaf tools so Rule A doesn't fire (we want to exercise
        # specifically the Rule C shape).
        self.tools = [_AgentToolStub("search_web"), _AgentToolStub("read_file")]


async def test_rule_c_dispatches_under_escape_hatch_without_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With growth OFF and ``GOLDFIVE_CAPABILITY_RULE_C=1``, the Rule C
    CAPABILITY_MISMATCH dispatches as a plain drift — the verdict-path
    growth absorption was removed (growth lives at pin time now).

    Migrated from ``test_flag_off_dispatches_rule_c_drift_as_today``:
    Rule C is soft-retired by default, so reaching the legacy dispatch
    now additionally requires the explicit env escape hatch.
    """
    monkeypatch.setenv("GOLDFIVE_CAPABILITY_RULE_C", "1")
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=False,  # legacy escape hatch
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
    )

    # Rule C drift was dispatched.
    rule_c = [
        d
        for d in recorder.handled_drifts
        if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert (
        len(rule_c) == 1
    ), f"under the escape hatch, expected 1 Rule C drift; got {len(rule_c)}"
    # No discovered task synthesised — the verdict-path fallback is gone.
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert (
        discovered == []
    ), f"verdict path must not grow discovered tasks; got {[t.id for t in discovered]}"


async def test_rule_c_silent_by_default_on_verdict_path() -> None:
    """Same Rule C shape, no env flag: nothing fires and nothing grows.

    New default pinned by goldfive#423 / AGENCY-PRESERVATION.md PR 2
    (correctness requirement (c)): Rule C is soft-retired, and the
    verdict path no longer owns any growth, so the canonical Rule C
    fixture produces zero drift dispatches and zero plan mutations.
    """
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=False,
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    recorder = _RecordingDriftSteerer(steerer)

    session = _make_session()
    session.current_task_id = "draft_slides"
    plugin, _state = _build_ctx_and_plugin(session, steerer)

    ctx = plugin._active_ctx
    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_InvokedAgentStub("reviewer_agent"),
        invoked_agent_name="reviewer_agent",
        invocation_id="inv-test-default",
    )

    assert recorder.handled_drifts == [], (
        f"Rule C must be silent by default; got "
        f"{[d.detail for d in recorder.handled_drifts]}"
    )
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert discovered == []


async def test_verdict_path_no_longer_grows() -> None:
    """With growth ON, the verdict path neither grows nor dispatches Rule C.

    Migrated from ``test_flag_on_grows_discovered_task_and_suppresses_rule_c``
    with the polarity inverted: the growth that test asserted HERE now
    happens at pin time (positive coverage in
    ``test_descriptive_growth_pin_time.py``); the verdict path is a
    pure detector again, and Rule C's retirement means the old fixture
    produces no verdict at all.
    """
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            descriptive_growth_enabled=True,
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    recorder = _RecordingDriftSteerer(steerer)

    session = _make_session()
    session.current_task_id = "draft_slides"
    plugin, _state = _build_ctx_and_plugin(session, steerer)

    ctx = plugin._active_ctx
    prior_task_count = len(session.plan.tasks)
    prior_revision = session.plan.revision_index

    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_InvokedAgentStub("reviewer_agent"),
        invoked_agent_name="reviewer_agent",
        invocation_id="inv-test-2",
    )

    # No drift dispatched (Rule C retired; Rules A/B don't apply here).
    assert recorder.handled_drifts == []
    # No growth from the verdict path — the plan is untouched.
    assert session.plan is not None
    assert len(session.plan.tasks) == prior_task_count
    assert session.plan.revision_index == prior_revision
    discovered = [t for t in session.plan.tasks if getattr(t, "discovered", False)]
    assert discovered == []
    # The pin was not moved.
    assert session.current_task_id == "draft_slides"


async def test_flag_on_rule_a_still_fires() -> None:
    """Rule A (leaf-task with AgentTool-only agent) is unaffected by the flag.

    Rule A — coordinator-style leaf-assignment — still dispatches
    normally for NON-discovered bound tasks (Rule A is
    AGENCY-PRESERVATION.md PR 3's business, untouched by PR 2). Kept
    from the original suite; only the removed growth-threading kwargs
    changed.
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
    )

    # Rule A drift was dispatched normally.
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


async def test_discovered_bound_task_skips_capability_rules() -> None:
    """A ``discovered=True`` bound task short-circuits the detector.

    Design doc §11.4 resolution (a): the auto-derived
    ``agent_name: request`` title reads leaf-shaped, so without the
    skip an AgentTool-only agent pinned to its own discovered task
    would re-fire Rule A on every delegation — the exact 2d27ff4a
    storm descriptive growth exists to stop.
    """

    class _AgentToolWrapper:
        def __init__(self, name: str, agent_name: str) -> None:
            self.name = name
            self.agent = type("_A", (), {"name": agent_name})()

    class _OnlyAgentToolsAgent:
        def __init__(self) -> None:
            self.name = "debugger_agent"
            self.tools = [_AgentToolWrapper("delegate_x", "agent_x")]

    plan = Plan(
        id="p-disc",
        run_id="r-disc",
        goal_ids=["g-disc"],
        tasks=[
            Task(
                id="discovered-abc123",
                title="debugger_agent: locate cherry tree files",
                description='{"request": "locate cherry tree files"}',
                assignee_agent_id="debugger_agent",
                status=TaskStatus.PENDING,
                discovered=True,
                discovery_identity_hash="hash1234abcd5678",
            ),
        ],
        edges=[],
        revision_index=2,
    )
    session = Session(
        run_id="r-disc",
        goals=[Goal(id="g-disc", summary="discovered tasks skip rules")],
        plan=plan,
    )
    session.current_task_id = "discovered-abc123"

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
    plugin.set_active_context(
        SessionContext(
            session=session,
            steerer=steerer,
            task=plan.tasks[0],
            tool_handlers={},
            host_agent_name="coordinator",
        )
    )

    ctx = plugin._active_ctx
    await plugin._maybe_emit_capability_mismatch(
        ctx=ctx,
        invoked_agent=_OnlyAgentToolsAgent(),
        invoked_agent_name="debugger_agent",
        invocation_id="inv-test-disc",
    )

    assert recorder.handled_drifts == [], (
        f"capability rules must be skipped on discovered tasks "
        f"(design doc §11.4(a)); got "
        f"{[d.detail for d in recorder.handled_drifts]}"
    )
