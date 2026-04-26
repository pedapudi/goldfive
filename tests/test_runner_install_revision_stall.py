"""Regression: multi-turn install via Phase 4 :meth:`Runner._install_revision`
must not stall AND must not smear every fresh user turn into the
``goldfive.active_steer.*`` slot reserved for genuine operator
:class:`~goldfive.control.ControlMessage` STEER interventions
(goldfive#271 Phase 4 follow-up).

Phase 4's :meth:`Runner._install_revision` synthesizes a ``USER_STEER``
:class:`~goldfive.types.DriftEvent` on every plan install so the
existing :meth:`DefaultSteerer.apply_user_steer_with_plan` pipeline can
do the structural install (validate → :meth:`_apply_revision` →
``PlanRevised``). The synthesized drift carries no
:attr:`DriftEvent.raw` (no originating ControlMessage); the prior code
path still ran the USER_STEER state-write side effects on it, which
overwrote ``goldfive.active_steer.body`` with the user's raw turn input
on EVERY turn whose :meth:`Planner.handle_turn` produced a plan —
conflating "operator pushed a STEER mid-flight" with "user drove a
fresh turn". The fix gates :meth:`_apply_user_steer_state` on
``drift.raw is not None`` so only real ControlMessage-backed STEERs
write the slot.

These tests exercise:

* The fast-path: two consecutive turns through
  :meth:`Runner._install_revision` with a stable outer ``session_id``
  (the ADK-web pin path) must complete the install pipeline cleanly
  on both turns. A regression that left the per-session plan lock
  held would deadlock turn 2 here.
* The state hygiene: after Runner-driven installs,
  ``goldfive.active_steer.*`` must remain unwritten — the slot is
  reserved for genuine operator STEERs.
* The control-message path is preserved: a real STEER ControlMessage
  surfaced via :meth:`DefaultSteerer.apply_user_steer_with_plan`
  (with ``drift.raw`` populated) DOES write the active_steer slot.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    Plan,
    Runner,
    SequentialExecutor,
    Session,
    Task,
    TaskEdge,
)
from goldfive import orchestration_state as _ostate
from goldfive.control import ControlKind, ControlMessage
from goldfive.steerer import DefaultSteerer
from goldfive.types import DriftEvent, DriftKind, DriftSeverity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[Any],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _two_plans() -> tuple[str, str]:
    """Return two plan-shaped JSON blobs the LLMPlanner stub returns
    on turns 1 and 2 respectively. Turn 2 preserves turn 1's tasks as
    completed (so revision validation passes) and adds new work."""
    plan1 = json.dumps({
        "summary": "Turn 1 plan",
        "tasks": [
            {
                "id": "t1_research",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
            },
            {
                "id": "t1_draft",
                "title": "Draft solar panel slide",
                "assignee_agent_id": "writer",
            },
        ],
        "edges": [
            {"from_task_id": "t1_research", "to_task_id": "t1_draft"},
        ],
    })
    plan2 = json.dumps({
        "summary": "Turn 2 plan (topic change to solar flares)",
        "tasks": [
            # Preserve terminal tasks from turn 1.
            {
                "id": "t1_research",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
                "status": "completed",
            },
            {
                "id": "t1_draft",
                "title": "Draft solar panel slide",
                "assignee_agent_id": "writer",
                "status": "completed",
            },
            # New work for the topic change.
            {
                "id": "t2_research",
                "title": "Research solar flares",
                "assignee_agent_id": "writer",
            },
            {
                "id": "t2_draft",
                "title": "Draft solar flares slide",
                "assignee_agent_id": "writer",
            },
        ],
        "edges": [
            {"from_task_id": "t1_research", "to_task_id": "t1_draft"},
            {"from_task_id": "t2_research", "to_task_id": "t2_draft"},
        ],
    })
    return plan1, plan2


def _stub_planner(plans: list[str]) -> LLMPlanner:
    """Return an :class:`LLMPlanner` whose ``call_llm`` cycles through
    ``plans`` for ``handle_turn``. Calls counter is captured in the
    planner's closure so the test can assert how many handle_turn
    calls fired.
    """
    counter = {"i": 0}

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = user, model
        # handle_turn system prompt sentinel — return next plan in list
        # wrapped as the JSON shape the planner parser expects.
        if (
            "next REVISION of the plan" in system
            or "warrants a plan change" in system
        ):
            i = counter["i"]
            counter["i"] = i + 1
            payload = json.loads(plans[min(i, len(plans) - 1)])
            return json.dumps({"reasoning": f"turn {i}", "plan": payload})
        # planner.generate fall-through (first turn legacy path).
        return plans[0]

    return LLMPlanner(call_llm=planner_llm, model="stub")


def _runner(planner: LLMPlanner, sink: InMemorySink) -> Runner:
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )


# ---------------------------------------------------------------------------
# Stall regression
# ---------------------------------------------------------------------------


async def test_install_revision_two_turns_stable_session_id_no_stall() -> None:
    """Two consecutive Runner-driven installs on the SAME outer
    session_id must both complete inside a tight bounded timeout.

    Repro for the validation-v3 stall: turn 1 ran ``handle_turn`` →
    install → executor → stash, then turn 2's ``handle_turn`` fired but
    ``_install_revision`` never logged ``_apply_revision``. A regression
    that re-introduced a per-session leaked lock or a recursive emit
    loop in :meth:`apply_user_steer_with_plan` would manifest here as
    turn 2 timing out.
    """
    plan1, plan2 = _two_plans()
    sink = InMemorySink()
    runner = _runner(_stub_planner([plan1, plan2]), sink)
    try:
        out1 = await asyncio.wait_for(
            runner.run(
                "Create a presentation about solar panels.",
                session_id="adk-pinned-session-A",
            ),
            timeout=10.0,
        )
        assert out1.success, f"turn 1 failed: {out1.reason!r}"
        plan_id_t1 = out1.session.plan.id
        assert out1.session.plan.revision_index == 1

        out2 = await asyncio.wait_for(
            runner.run(
                "Forget solar panels, tell me about solar flares.",
                session_id="adk-pinned-session-A",
            ),
            timeout=10.0,
        )
        assert out2.success, f"turn 2 failed: {out2.reason!r}"
        # Phase 4 contract: plan_id stable across revisions.
        assert out2.session.plan.id == plan_id_t1
        # Revision index bumped 1 → 2.
        assert out2.session.plan.revision_index == 2
    finally:
        await runner.close()


async def test_install_revision_first_turn_fresh_session_no_stall() -> None:
    """The very first install on a fresh session (transitioning from
    :meth:`Plan.empty` to revision 1) must complete without stalling.

    Pairs with the multi-turn test above so a regression that breaks
    the first install in isolation (e.g. mishandling the ``Plan.empty``
    seed in the install pipeline) surfaces with a clean failure.
    """
    plan1, _ = _two_plans()
    sink = InMemorySink()
    runner = _runner(_stub_planner([plan1]), sink)
    try:
        out = await asyncio.wait_for(
            runner.run(
                "Create a presentation about solar panels.",
                session_id="adk-pinned-fresh",
            ),
            timeout=10.0,
        )
        assert out.success, f"first install failed: {out.reason!r}"
        assert out.session.plan is not None
        assert out.session.plan.revision_index == 1
        assert out.session.plan.tasks
        # PlanRevised fires for the first install (the Plan.empty seed
        # is revision 0; the first real plan is revision 1).
        which = [
            e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None
            for e in sink.events
        ]
        assert "plan_revised" in which, which
    finally:
        await runner.close()


# ---------------------------------------------------------------------------
# active_steer state hygiene
# ---------------------------------------------------------------------------


async def test_runner_synthesized_install_does_not_write_active_steer() -> None:
    """Runner-synthesized USER_STEER drifts (no ``raw``) must NOT
    smear the user's raw turn input into the
    ``goldfive.active_steer.*`` orchestration-state slot.

    That slot is reserved for the most recent operator-pushed STEER
    :class:`~goldfive.control.ControlMessage` so the planner's
    refine prompt can frame a steered turn as a directive. Writing it
    on every fresh turn driven by :meth:`Runner._install_revision`
    would conflate two distinct intervention shapes and feed stale
    "active steer" framing into every downstream prompt.

    The fix gates :meth:`_apply_user_steer_state` on
    ``drift.raw is not None``; this test confirms a Runner-driven
    install with the synthesized drift (``raw is None``) leaves the
    state slot empty.
    """
    plan1, _ = _two_plans()
    sink = InMemorySink()
    runner = _runner(_stub_planner([plan1]), sink)
    try:
        out = await asyncio.wait_for(
            runner.run(
                "Create a presentation about solar panels.",
                session_id="state-hygiene-A",
            ),
            timeout=10.0,
        )
        assert out.success
        # Phase 4 install path: USER_STEER drift was synthesized
        # by the Runner (drift.raw is None). The bookkeeping must
        # have been skipped so the state slot stays unwritten.
        body = _ostate.read(
            out.session.state, _ostate.KEY_ACTIVE_STEER_BODY, ""
        )
        author = _ostate.read(
            out.session.state, _ostate.KEY_ACTIVE_STEER_AUTHOR, ""
        )
        source = _ostate.read(
            out.session.state, _ostate.KEY_ACTIVE_STEER_SOURCE, ""
        )
        assert body == "", f"unexpected active_steer.body={body!r}"
        assert author == "", f"unexpected active_steer.author={author!r}"
        assert source == "", f"unexpected active_steer.source={source!r}"
    finally:
        await runner.close()


async def test_real_control_message_steer_writes_active_steer() -> None:
    """The genuine operator STEER path (drift carries ``raw =
    ControlMessage``) MUST still write ``goldfive.active_steer.*``.

    Pairs with the no-write test above so the gate is conditional on
    ``drift.raw``, not a blanket "skip USER_STEER bookkeeping in
    apply_user_steer_with_plan". Without this case a future tightening
    that disables USER_STEER bookkeeping entirely would silently
    regress operator-driven steers (their state slot would never be
    populated, breaking the planner's refine framing).
    """
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_stub_planner(["{}"]))
    session = Session(run_id="state-hygiene-control-B")
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="seed",
    )

    # Build the same shape :meth:`DefaultSteerer._drift_from_control`
    # produces for an operator STEER ControlMessage so the
    # ``drift.raw`` carries the genuine source.
    control = ControlMessage(
        kind=ControlKind.STEER,
        payload={
            "note": "Refocus on solar flares",
            "author": "operator-Alice",
            "annotation_id": "ann-42",
        },
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="by operator-Alice: Refocus on solar flares",
        raw=control,
        authored_by="user",
    )
    revised = Plan(
        id=session.plan.id,
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[
            Task(id="t1", title="T1", assignee_agent_id="w"),
            Task(id="t2", title="T2 (added)", assignee_agent_id="w"),
        ],
        edges=[],
        summary="post-steer",
    )
    installed = await asyncio.wait_for(
        steerer.apply_user_steer_with_plan(
            drift=drift, session=session, revised_plan=revised
        ),
        timeout=5.0,
    )
    assert installed
    # The genuine STEER path populated active_steer slot.
    body = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_BODY, "")
    author = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_AUTHOR, "")
    source = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_SOURCE, "")
    assert body == "Refocus on solar flares", body
    assert author == "operator-Alice", author
    assert source == "user", source
