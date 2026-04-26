"""Intra-session plan-id stability across two CONCURRENT turns
on the SAME pinned ``session_id`` (goldfive#271 follow-up).

The demo log v6 (2026-04-25) showed this regression on outer
session ``v7class1-1``. Forensic timeline reconstructed from
``/tmp/demo-v6.log`` + harmonograf ``goldfive_events`` table:

* 01:23:43 — turn 1 ``handle_turn`` produces plan_id ``ef85ed9f``;
  ``_apply_revision`` lands revision 1.
* 01:24:12 onwards — turn 1's executor drives sub-agents.
* 01:35:12 — turn 2's ``goal_derived`` event emitted at ``sequence=3``
  on the SAME ``session_id=v7class1-1`` (sequence reset → fresh
  ``Conversation._next_sequence`` cursor — but the convo dict
  still holds the same Conversation instance).
* 01:35:40 — adk-web closes the inner ADK Runner (turn 1's
  ``/run_sse`` stream); turn 1's ``Runner.run`` ``finally`` block
  fires ``Conversation.stash_plan(session, pinned=True)`` —
  ``stashed prior plan ... plan_id=ef85ed9f revision_index=1
  session_id=v7class1-1``.
* 01:35:45 — turn 2's ``handle_turn`` finally returns its LLM
  call's response; the log records ``prior_plan_id=6b829ad3`` —
  a fresh ``Plan.empty()`` id, NOT ``ef85ed9f``.

The smoking gun: turn 2 ENTERED ``Runner.run`` BEFORE turn 1's
``finally``-block stash fired. ``Conversation.prior_plan_for(
"v7class1-1", pinned=True)`` returned ``None`` at the point turn 2
seeded ``session.plan = Plan.empty(...)`` (runner.py line 390),
because turn 1's stash didn't land until ~28s later. Turn 2's
``handle_turn`` therefore saw the empty seed as prior, the LLM
produced a plan that inherited the empty seed's id (via
``_parse_handle_turn_response``'s ``plan_id_override`` rule), and
the conversation lost ``plan_id`` continuity that Phase 4 promised.

PR #294 fixed CROSS-session leaks (different ``session_id`` →
different Conversation). The bug here is INTRA-session: two
concurrent ``Runner.run`` calls on the SAME ``session_id`` race on
the per-Conversation ``_last_plan`` slot. ``Runner.run`` is
non-reentrant per Conversation but nothing enforces that.

The fix: serialise ``Runner.run`` per-Conversation-key with an
``asyncio.Lock``. The lock spans the entire ``run`` body — released
on both normal return AND ``BaseException`` propagation — so the
second turn sees the first turn's absorbed Conversation state
(prior plan, completed_results, sequence cursor, turns) in full.
Two concurrent runs on DIFFERENT keys still own DISTINCT locks
(``_lock_for`` is keyed on the same convo_key as
``_conversation_for``), so the fix introduces no NEW cross-session
serialisation point.

These tests pin the contract:

1. Two consecutive sequential turns on the SAME pinned session id
   must propagate the prior plan via ``handle_turn`` — same id,
   non-empty tasks. (Regression net for the existing intra-session
   continuity invariant from PR #294.)
2. Two CONCURRENT turns on the SAME pinned session id (turn 2
   entering before turn 1's finally-block stash) must serialise —
   the second turn's ``handle_turn`` sees the first's post-install
   plan as prior. (The bug from demo log v6 / v7class1-1.)
3. Two runs on DIFFERENT pinned session ids own DISTINCT
   per-key locks. (No new cross-session serialisation regression.)
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
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _scripted_planner_llm(plan_payload: dict[str, Any]) -> Any:
    """Build an async ``call_llm`` that always returns the same plan body.

    The Phase 4 ``handle_turn`` system prompt asks for a JSON object
    with optional ``plan`` key. Returning the plan unconditionally
    makes every turn produce the same revision.
    """

    payload = json.dumps({"reasoning": "ok", "plan": plan_payload})

    async def _llm(system: str, user: str, model: str) -> str:
        _ = system, user, model
        return payload

    return _llm


def _plan_payload() -> dict[str, Any]:
    return {
        "summary": "research then draft",
        "tasks": [
            {"id": "research", "title": "Research", "assignee_agent_id": "writer"},
            {"id": "draft", "title": "Draft", "assignee_agent_id": "writer"},
        ],
        "edges": [{"from_task_id": "research", "to_task_id": "draft"}],
    }


def _mk_runner() -> Runner:
    """Build a Runner with LLMPlanner + scripted call_llm + happy adapter."""
    planner = LLMPlanner(
        call_llm=_scripted_planner_llm(_plan_payload()), model="stub"
    )
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("derived"),
        sinks=[InMemorySink()],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_intra_session_handle_turn_sees_prior_plan_id() -> None:
    """Two consecutive turns on the SAME pinned ``session_id`` must
    share the prior-plan stash so the second turn's
    :meth:`Planner.handle_turn` sees the FIRST turn's post-install
    plan id, NOT a fresh ``Plan.empty()`` seed.

    The bug from demo log v6 (2026-04-25 / v7class1-1): turn 1
    landed ``plan_id=ef85ed9f`` via ``_apply_revision``; the
    Runner's ``finally`` block stashed that plan keyed by
    ``session_id=v7class1-1``; turn 2 5 seconds later (same Runner,
    same pinned id) saw ``handle_turn: prior_plan_id=6b829ad3`` — a
    brand new empty seed. The carry-forward was silently lost
    despite ``Conversation.prior_plan_for`` having all the
    bookkeeping in place.

    We assert directly on what ``handle_turn`` saw at turn 2 by
    capturing the value it logs (``LLMPlanner.handle_turn`` reads
    ``session.plan.id`` at line 3253 of planner.py and rolls it
    into ``prior_id`` for the log line). Equivalent: capture
    ``session.plan.id`` at the moment the planner is invoked.
    """
    runner = _mk_runner()

    # Hook into ``planner.handle_turn`` to record ``session.plan`` at
    # the moment the planner is invoked. This is the value
    # ``LLMPlanner.handle_turn`` would log as ``prior_plan_id``.
    seen_prior_plan_at_turn_2: list[Plan | None] = []
    real_handle_turn = runner.planner.handle_turn

    async def _capturing_handle_turn(*args: Any, **kwargs: Any) -> Any:
        sess = kwargs.get("session")
        if sess is not None:
            seen_prior_plan_at_turn_2.append(sess.plan)
        return await real_handle_turn(*args, **kwargs)

    runner.planner.handle_turn = _capturing_handle_turn  # type: ignore[method-assign]

    # Turn 1 on pinned session.
    out_1 = await runner.run("first turn", session_id="outer-pinned")
    assert out_1.success, out_1.reason
    turn_1_plan_id = out_1.session.plan.id
    assert turn_1_plan_id, "turn 1 produced no plan id"
    # Drop turn 1's observation so we only assert against turn 2.
    seen_prior_plan_at_turn_2.clear()

    # Turn 2 on the SAME pinned session id.
    await runner.run("second turn", session_id="outer-pinned")
    await runner.close()

    assert seen_prior_plan_at_turn_2, (
        "planner.handle_turn was not invoked on turn 2 — the test cannot "
        "observe the seed it would have logged as prior_plan_id"
    )
    seed_at_turn_2 = seen_prior_plan_at_turn_2[0]
    assert seed_at_turn_2 is not None
    # The strong assertion: the seed turn 2 sees IS the post-install
    # plan from turn 1, with the SAME plan id and a non-empty task
    # list. Pre-fix the seed is a fresh Plan.empty() with a brand-new
    # uuid id and zero tasks.
    assert seed_at_turn_2.id == turn_1_plan_id, (
        f"intra-session plan_id NOT stable: turn 2's handle_turn "
        f"received seed plan_id={seed_at_turn_2.id!r}, expected "
        f"{turn_1_plan_id!r} from turn 1's post-install revision. "
        f"Carry-forward via Conversation._last_plan was lost despite "
        f"both turns sharing pinned session_id='outer-pinned'."
    )
    assert seed_at_turn_2.tasks, (
        "intra-session prior-plan carry-forward regressed: turn 2's "
        "handle_turn seed has zero tasks (Plan.empty() seed) instead of "
        "the post-install revision from turn 1."
    )


async def test_intra_session_concurrent_runs_serialise_carry_forward() -> None:
    """The actual bug from demo log v6 / v7class1-1.

    Two concurrent ``Runner.run`` calls on the SAME pinned
    ``session_id`` overlap: turn 2 enters before turn 1's
    ``finally``-block stash fires. Without per-key serialisation
    turn 2's ``Conversation.prior_plan_for`` returns ``None`` and
    seeds ``session.plan = Plan.empty(...)``, then ``handle_turn``
    sees the empty seed as prior and the LLM (preserving prior id)
    produces a plan with the empty seed's id — losing the
    plan_id-stable-across-turns invariant.

    The fix serialises ``Runner.run`` per-Conversation-key so the
    second ``run`` call waits for the first's ``finally`` block
    (and ``absorb_turn``) to land before it starts its own seeding.

    Verifies: with a stalled turn 1 and a turn 2 that starts ~50ms
    later, turn 2's ``handle_turn`` is invoked AFTER turn 1's stash
    and observes turn 1's post-install plan (non-empty tasks, same
    plan id) as prior.
    """
    runner = _mk_runner()

    # Capture each handle_turn invocation's session.plan snapshot.
    handle_turn_seeds: list[Plan | None] = []
    real_handle_turn = runner.planner.handle_turn

    async def _capturing_handle_turn(*args: Any, **kwargs: Any) -> Any:
        sess = kwargs.get("session")
        handle_turn_seeds.append(sess.plan if sess is not None else None)
        return await real_handle_turn(*args, **kwargs)

    runner.planner.handle_turn = _capturing_handle_turn  # type: ignore[method-assign]

    # Turn 1 stalls inside the executor for ~250ms so turn 2 can
    # start before turn 1 finishes. Real-world this is the LLM-call
    # latency the demo's coordinator agent took mid-stream; the
    # exact duration isn't material — what matters is overlap.
    async def _slow_agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = session, tools
        await asyncio.sleep(0.25)
        return InvocationResult(task_id=task.id, text="ok")

    runner.agent = CallableAdapter(_slow_agent, available_agents=["writer"])

    # Kick off turn 1 and turn 2 concurrently on the same pin. Turn 2
    # is delayed ~50ms so it definitely enters AFTER turn 1's seed +
    # handle_turn but BEFORE turn 1's finally-block stash. That's
    # the v7class1-1 timing reproduced.
    turn_1_task = asyncio.create_task(
        runner.run("first turn", session_id="outer-pinned")
    )
    await asyncio.sleep(0.05)
    turn_2_task = asyncio.create_task(
        runner.run("second turn", session_id="outer-pinned")
    )
    out_1, out_2 = await asyncio.gather(turn_1_task, turn_2_task)
    await runner.close()

    # Turn 1 must succeed — the test's scripted LLM returns a valid
    # initial plan and the happy adapter completes both tasks. Turn
    # 2's downstream success is NOT asserted: the scripted LLM
    # returns the SAME plan body with PENDING tasks, so once turn 2
    # actually sees turn 1's COMPLETED post-install plan as prior,
    # the LLM's PENDING-status resubmission is rejected by the
    # validator's §3.1 terminal-preservation rule. This is the
    # CORRECT downstream outcome of carry-forward working — without
    # the fix, turn 2 sees an empty seed and "succeeds" only by
    # accidentally minting a fresh-empty-id plan that has no prior
    # COMPLETED tasks to violate.
    assert out_1.success, f"turn 1 failed: {out_1.reason}"
    turn_1_plan_id = out_1.session.plan.id
    assert turn_1_plan_id, "turn 1 produced no plan id"

    # Two handle_turn calls were captured — one per turn. Order is
    # the order they were INVOKED, which under per-key serialisation
    # is turn 1 then turn 2. (The race is fundamentally between turn
    # 2's seeding and turn 1's stash — the assertion is on what turn
    # 2's handle_turn saw.)
    assert len(handle_turn_seeds) == 2, (
        f"expected 2 handle_turn invocations (turn 1 + turn 2), "
        f"got {len(handle_turn_seeds)}"
    )
    seed_turn_1 = handle_turn_seeds[0]
    seed_turn_2 = handle_turn_seeds[1]
    assert seed_turn_1 is not None
    assert seed_turn_2 is not None
    # Turn 1 saw an empty seed (first turn ever on this pin).
    assert not seed_turn_1.tasks, (
        "turn 1's handle_turn should see an empty seed; got "
        f"{len(seed_turn_1.tasks)} tasks"
    )
    # Turn 2's seed must be turn 1's POST-INSTALL plan: non-empty
    # tasks AND same plan id. Pre-fix turn 2 sees an empty seed
    # (None from prior_plan_for + Plan.empty mint) — same id only
    # by coincidence (LLM reuses prior id), but tasks=0 is the
    # robust signal the carry-forward was lost.
    assert seed_turn_2.tasks, (
        f"intra-session prior-plan carry-forward regressed under "
        f"concurrent runs: turn 2's handle_turn seed has zero tasks "
        f"(Plan.empty() seed) plan_id={seed_turn_2.id!r}; turn 1's "
        f"post-install plan was {turn_1_plan_id!r} with "
        f"{len(out_1.session.plan.tasks)} task(s). The Runner did "
        f"not serialise concurrent runs on session_id='outer-pinned'."
    )
    assert seed_turn_2.id == turn_1_plan_id, (
        f"intra-session plan_id NOT stable across concurrent runs: "
        f"turn 2's handle_turn received seed plan_id={seed_turn_2.id!r}, "
        f"expected {turn_1_plan_id!r} from turn 1's post-install revision."
    )


async def test_concurrent_runs_on_distinct_session_ids_get_distinct_locks() -> None:
    """The fix's per-key serialisation must remain truly per-key:
    runs on DIFFERENT pinned ``session_id`` values must own
    DISTINCT ``asyncio.Lock`` instances so they never block on
    each other's lock.

    Goldfive's Runner doesn't otherwise guarantee parallel forward
    progress across sessions today — adapters / steerer / executor
    all hold instance state that effectively serialises end-to-end —
    so a wall-clock parallelism assertion would over-specify and
    fail for unrelated reasons. The structural assertion here is
    sufficient: two distinct keys yield two distinct Locks, so the
    per-key lock added by this fix introduces no NEW cross-session
    serialisation point.
    """
    runner = _mk_runner()

    out_a = await runner.run("a", session_id="outer-A")
    out_b = await runner.run("b", session_id="outer-B")
    await runner.close()

    assert out_a.success
    assert out_b.success
    # Distinct keys → distinct lock instances. Lazy creation in
    # :meth:`Runner._lock_for` populates the ``_convo_locks`` dict
    # on first use; both keys must own their own slot.
    lock_a = runner._convo_locks["outer-A"]
    lock_b = runner._convo_locks["outer-B"]
    assert lock_a is not lock_b, (
        "per-key lock leaked across session ids: outer-A and outer-B "
        "share one lock instance, which would serialise concurrent "
        "runs on distinct outer ADK sessions"
    )
