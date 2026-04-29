"""Regression: multi-turn install via Phase 4 :meth:`Runner._install_revision`
must not stall AND must not smear every fresh user turn into the
``goldfive.active_steer.*`` slot reserved for genuine operator
:class:`~goldfive.control.ControlMessage` STEER interventions
(goldfive#271 Option A).

Goldfive#271 Option A: :meth:`Runner._install_revision` dispatches
across two steerer APIs based on what's actually happening:

* Turn 1 (``Plan.empty`` seed) →
  :meth:`DefaultSteerer.install_initial_plan` — emits ``PlanRevised``
  only; no ``DriftDetected``.
* Turn N+1 LLM-driven replan →
  :meth:`DefaultSteerer.install_revision_for_drift` with a real
  ``NEW_WORK_DISCOVERED`` drift.

A genuine operator-pushed STEER (``ControlMessage`` arriving via the
control channel) goes through
:meth:`DefaultSteerer.install_revision_for_user_steer`, which is the
**only** path that writes ``goldfive.active_steer.*`` — eliminating
the category error pre-Option-A worked around with a ``synthetic``
flag.

These tests exercise:

* The fast-path: two consecutive turns through
  :meth:`Runner._install_revision` with a stable outer ``session_id``
  (the ADK-web pin path) must complete the install pipeline cleanly
  on both turns. A regression that left the per-session plan lock
  held would deadlock turn 2 here.
* The state hygiene: after Runner-driven installs,
  ``goldfive.active_steer.*`` must remain unwritten — the slot is
  reserved for genuine operator STEERs.
* The wire shape: turn-1 installs emit no ``USER_STEER``
  ``DriftDetected``; turn N+1 installs emit ``NEW_WORK_DISCOVERED``;
  genuine STEER ``ControlMessage``s emit ``USER_STEER`` with the
  source ``annotation_id`` preserved.
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
)
from goldfive import orchestration_state as _ostate
from goldfive.control import ControlKind, ControlMessage
from goldfive.steerer import DefaultSteerer

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

        # Additive steer (not a pivot — keeps the same artefact). Pivot
        # phrasings ("forget X, do Y instead") now route through
        # ``install_initial_plan`` per goldfive#322 R1, which would
        # mint a fresh plan_id and bypass the revision invariant
        # this test pins.
        out2 = await asyncio.wait_for(
            runner.run(
                "Add a citations slide and make it more thorough.",
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


async def test_runner_install_does_not_write_active_steer() -> None:
    """A Runner-driven install (turn 1 :meth:`install_initial_plan`,
    turn N+1 :meth:`install_revision_for_drift`) MUST NOT smear the
    user's raw turn input into the ``goldfive.active_steer.*``
    orchestration-state slot.

    That slot is reserved for the most recent operator-pushed STEER
    :class:`~goldfive.control.ControlMessage` so the planner's
    refine prompt can frame a steered turn as a directive. Writing
    it on every fresh turn driven by :meth:`Runner._install_revision`
    would conflate two distinct intervention shapes and feed stale
    "active steer" framing into every downstream prompt.

    Goldfive#271 Option A guarantees this structurally: turn-1
    installs route to :meth:`install_initial_plan` (no USER_STEER
    bookkeeping at all) and turn N+1 LLM-driven replans route to
    :meth:`install_revision_for_drift` with a ``NEW_WORK_DISCOVERED``
    drift (also no USER_STEER bookkeeping). The active_steer slot
    only gets written by :meth:`install_revision_for_user_steer`.
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


# ---------------------------------------------------------------------------
# Option A: install paths emit accurate drift kinds (no synthetic flag)
# ---------------------------------------------------------------------------


async def test_first_turn_install_emits_no_user_steer_drift() -> None:
    """The very first install (turn 1, ``Plan.empty`` seed) MUST NOT
    emit a ``USER_STEER`` :class:`DriftDetected` on the wire.

    Goldfive#271 Option A regression. Before Option A,
    :meth:`Runner._install_revision` fabricated a ``USER_STEER``
    :class:`DriftEvent` for every plan install and routed it through
    :meth:`DefaultSteerer.apply_user_steer_with_plan` — which always
    emitted a ``DriftDetected`` of kind ``USER_STEER`` even though no
    operator had pushed a STEER. PR #302 papered over the category
    error with a ``synthetic=True`` flag and a harmonograf filter;
    Option A eliminates the fabrication entirely by routing turn-1
    installs through :meth:`DefaultSteerer.install_initial_plan`,
    which emits only ``PlanRevised`` (no ``DriftDetected``).
    """
    plan1, _ = _two_plans()
    sink = InMemorySink()
    runner = _runner(_stub_planner([plan1]), sink)
    try:
        out = await asyncio.wait_for(
            runner.run(
                "Create a presentation about solar panels.",
                session_id="option-a-first-turn",
            ),
            timeout=10.0,
        )
        assert out.success, f"install failed: {out.reason!r}"

        from goldfive.pb.goldfive.v1 import types_pb2 as _tpb

        user_steer_drifts = []
        for evt in sink.events:
            which = evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None
            if which != "drift_detected":
                continue
            dd = evt.drift_detected
            if int(dd.kind) == _tpb.DRIFT_KIND_USER_STEER:
                user_steer_drifts.append(dd)
        assert not user_steer_drifts, (
            "Option A: turn-1 install must NOT emit USER_STEER drift; "
            f"got {len(user_steer_drifts)} on the sink"
        )
        # PlanRevised still fires (revision_index 1 from Plan.empty).
        which = [
            e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None
            for e in sink.events
        ]
        assert "plan_revised" in which, which
    finally:
        await runner.close()


async def test_second_turn_install_emits_new_work_discovered_drift() -> None:
    """Turn N+1 (user message produces an LLM-driven replan) MUST emit
    a ``NEW_WORK_DISCOVERED`` :class:`DriftDetected`, not ``USER_STEER``.

    The user typed a new message — that is genuinely new work the
    planner integrated. Modelling it as ``NEW_WORK_DISCOVERED`` is
    the honest classification (Option A); modelling it as
    ``USER_STEER`` was the category error #302 worked around.
    """
    plan1, plan2 = _two_plans()
    sink = InMemorySink()
    runner = _runner(_stub_planner([plan1, plan2]), sink)
    try:
        out1 = await asyncio.wait_for(
            runner.run(
                "Create a presentation about solar panels.",
                session_id="option-a-replan",
            ),
            timeout=10.0,
        )
        assert out1.success
        # Snapshot which drift_detected events came from turn 1.
        turn1_drift_count = sum(
            1
            for evt in sink.events
            if (evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None)
            == "drift_detected"
        )
        # Additive steer — must route through the drift install path.
        # A pivot phrasing ("forget X, do Y instead") would route
        # through ``install_initial_plan`` (goldfive#322 R1) which
        # does NOT emit a NEW_WORK_DISCOVERED drift, so the assertion
        # below would fail for the wrong reason.
        out2 = await asyncio.wait_for(
            runner.run(
                "Add a citations slide and make it more thorough.",
                session_id="option-a-replan",
            ),
            timeout=10.0,
        )
        assert out2.success
        from goldfive.pb.goldfive.v1 import types_pb2 as _tpb

        new_drifts = [
            evt.drift_detected
            for evt in sink.events
            if (evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None)
            == "drift_detected"
        ][turn1_drift_count:]
        # Find the install-path drift among any other drifts that may
        # fire concurrently in a richer scenario; here only the
        # NEW_WORK_DISCOVERED install drift is expected.
        assert any(
            int(dd.kind) == _tpb.DRIFT_KIND_NEW_WORK_DISCOVERED for dd in new_drifts
        ), (
            "Option A: turn N+1 replan must emit NEW_WORK_DISCOVERED "
            f"DriftDetected; got kinds={[int(dd.kind) for dd in new_drifts]!r}"
        )
        # And NOT USER_STEER.
        assert not any(
            int(dd.kind) == _tpb.DRIFT_KIND_USER_STEER for dd in new_drifts
        ), "Option A: turn N+1 replan must NOT emit USER_STEER drift"
    finally:
        await runner.close()


async def test_install_revision_for_user_steer_emits_user_steer_with_raw() -> None:
    """A genuine operator STEER ControlMessage routed through
    :meth:`DefaultSteerer.install_revision_for_user_steer` MUST emit a
    ``USER_STEER`` :class:`DriftDetected` whose source survives:
    ``authored_by="user"`` and the bridge-supplied ``annotation_id``
    lands on the wire.

    Pairs with the first-turn-install test: USER_STEER
    ``DriftDetected`` is now reserved exclusively for genuine operator
    STEERs. Flipping this invariant would silently hide every
    operator STEER from harmonograf's interventions panel.
    """
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_stub_planner(["{}"]))
    session = Session(run_id="option-a-control-B")
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="seed",
    )

    control = ControlMessage(
        kind=ControlKind.STEER,
        payload={
            "note": "Refocus on solar flares",
            "author": "operator-Alice",
            "annotation_id": "ann-77",
        },
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
        steerer.install_revision_for_user_steer(
            session=session, raw=control, revised_plan=revised
        ),
        timeout=5.0,
    )
    assert installed
    drift_rows = [
        evt.drift_detected
        for evt in sink.events
        if (evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None)
        == "drift_detected"
    ]
    from goldfive.pb.goldfive.v1 import types_pb2 as _tpb

    user_steer_rows = [
        dd for dd in drift_rows if int(dd.kind) == _tpb.DRIFT_KIND_USER_STEER
    ]
    assert user_steer_rows, "expected a USER_STEER DriftDetected on a real STEER"
    for dd in user_steer_rows:
        assert dd.authored_by == "user", dd.authored_by
        assert dd.annotation_id == "ann-77", dd.annotation_id


async def test_install_revision_for_user_steer_writes_active_steer() -> None:
    """The genuine operator STEER path MUST write the
    ``goldfive.active_steer.*`` bookkeeping so the planner's refine
    framing can read it next turn (goldfive#152)."""
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_stub_planner(["{}"]))
    session = Session(run_id="option-a-active-steer")
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="seed",
    )
    control = ControlMessage(
        kind=ControlKind.STEER,
        payload={
            "note": "Refocus on solar flares",
            "author": "operator-Alice",
            "annotation_id": "ann-42",
        },
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
        steerer.install_revision_for_user_steer(
            session=session, raw=control, revised_plan=revised
        ),
        timeout=5.0,
    )
    assert installed
    body = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_BODY, "")
    author = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_AUTHOR, "")
    source = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_SOURCE, "")
    assert body == "Refocus on solar flares", body
    assert author == "operator-Alice", author
    assert source == "user", source


async def test_install_initial_plan_emits_only_plan_revised() -> None:
    """:meth:`DefaultSteerer.install_initial_plan` MUST emit
    :class:`PlanRevised` with ``revision_index=1`` and **no**
    :class:`DriftDetected`. This is the structural turn-1 path —
    nothing went wrong, no intervention occurred."""
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_stub_planner(["{}"]))
    session = Session(run_id="option-a-install-initial")
    session.plan = Plan.empty(run_id=session.run_id)

    plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="initial",
    )
    installed = await asyncio.wait_for(
        steerer.install_initial_plan(session=session, plan=plan),
        timeout=5.0,
    )
    assert installed
    which = [
        e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None
        for e in sink.events
    ]
    assert "plan_revised" in which, which
    # No DriftDetected envelope from a clean first install.
    assert "drift_detected" not in which, (
        f"install_initial_plan must not emit DriftDetected; got {which!r}"
    )
    assert session.plan.revision_index == 1
