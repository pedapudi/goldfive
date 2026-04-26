"""Cross-session prior-plan isolation (goldfive#271 follow-up).

Validation v4 Class 1 surfaced a process-scoped leak:
:class:`Runner` stashed the prior turn's plan on
``self._last_plan``, a Runner attribute. When a single Runner served
two distinct outer ADK sessions (different ``ctx.session.id``), the
second session's first turn picked up the FIRST session's stashed
plan as its prior plan, then asked ``planner.handle_turn`` to revise
it for an unrelated user_input — every revision attempt failed
validation because the original plan's tasks made no sense for the
new request.

The fix moves the stash onto :class:`Conversation` and keys it by the
session id observed at stash time. A subsequent turn's prior-plan
seed is restored only when the new turn's session id matches; a turn
on a *different* session.id (a fresh outer ADK session sharing the
Runner) sees ``Plan.empty()`` exactly as if the Runner had just been
constructed.

These tests pin the contract:

1. Two consecutive turns on different ``session_id`` values must NOT
   share the prior-plan stash (the regression case).
2. Two consecutive turns on the SAME ``session_id`` must continue to
   propagate the prior plan (the existing intra-session continuity
   guarantee, validated so the fix doesn't regress it).
3. A turn on one ``session_id`` followed by a turn with no
   ``session_id`` pin (programmatic Runner caller) likewise must not
   leak the pinned session's plan into the unpinned turn.
"""

from __future__ import annotations

from typing import Any

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan(tag: str) -> Plan:
    """A trivial one-task plan whose id encodes ``tag`` so we can assert
    which plan a session ended up with."""
    return Plan(
        id=f"plan-{tag}",
        run_id="",
        goal_ids=[f"g-{tag}"],
        tasks=[
            Task(id=f"t-{tag}", title=f"Task {tag}", assignee_agent_id="writer"),
        ],
        edges=[],
        summary=f"Single-task plan tagged {tag}",
    )


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _mk_runner(planner_plan: Plan) -> Runner:
    """Build a Runner with a deterministic StaticPlanner and a happy
    callable adapter. ``planner_gate=None`` keeps the test isolated
    from the LLM-driven handle_turn path: the leak we're testing is
    in the Runner's seed/stash bookkeeping, not in the planner.
    """
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(planner_plan),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
        planner_gate=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_two_runs_different_session_ids_do_not_share_prior_plan() -> None:
    """The regression: turn on session A then turn on session B.

    With the leak in place, turn 2's seed restores the plan stashed
    by turn 1 (because it's a process-wide Runner attribute) — so
    turn 2's session.plan id starts as turn 1's plan id BEFORE the
    planner replaces it. With the fix, turn 2 sees ``Plan.empty()``
    seeded — the prior plan is keyed by session.id and turn 2's
    session.id doesn't match.

    We assert directly on the seed observed at the start of turn B's
    planner call — the leak's downstream symptom (validator rejecting
    every revision because the leaked tasks don't match the new user
    intent) is what produced ``outcome.success = False`` in
    validation v4 Class 1, but the root-cause assertion is on the
    seed itself.
    """
    runner = _mk_runner(_plan("static"))

    # The Runner installs the prior plan onto ``session.plan`` BEFORE
    # invoking the planner, so we can read what the planner saw by
    # inspecting the Session passed by reference. ``StaticPlanner.generate``
    # doesn't take ``session`` directly, so grab the live session off
    # the runner's last-session pointer at the moment generate runs.
    seeded_plan_at_turn_2: list[Plan | None] = []
    real_generate = runner.planner.generate

    async def _capturing_generate(*args: Any, **kwargs: Any) -> Plan:
        seeded_plan_at_turn_2.append(
            runner._last_session.plan if runner._last_session is not None else None
        )
        return await real_generate(*args, **kwargs)

    # Patch generate to record the seed. ``handle_turn`` isn't on
    # StaticPlanner so the Runner falls through to generate every
    # turn (``planner_gate=None`` reinforces this).
    runner.planner.generate = _capturing_generate  # type: ignore[method-assign]

    out_a = await runner.run("turn on session A", session_id="outer-session-A")
    assert out_a.success
    assert out_a.session.id == "outer-session-A"
    # Turn A produced a real plan with at least one task — that's the
    # precondition the stash sees as "worth carrying forward".
    assert out_a.session.plan is not None and out_a.session.plan.tasks

    # Reset the capture so we only record what turn B's planner sees.
    seeded_plan_at_turn_2.clear()

    await runner.run("turn on session B", session_id="outer-session-B")
    await runner.close()

    # The seed observed at the start of turn B's planner call must be
    # an empty seed (no tasks) — NOT the prior plan from session A.
    # Even if the run aborted downstream (it does today, on the leak,
    # because the validator rejects every revision attempt against the
    # foreign plan), the seed is the root-cause assertion.
    assert seeded_plan_at_turn_2, "planner.generate was not invoked on turn B"
    seeded = seeded_plan_at_turn_2[0]
    assert seeded is not None
    assert not seeded.tasks, (
        f"cross-session leak: turn B's seed carries {len(seeded.tasks)} "
        f"task(s) from turn A's plan id={seeded.id!r}"
    )


async def test_two_runs_same_session_id_propagate_prior_plan() -> None:
    """Within a single outer session, the prior-plan stash MUST carry
    forward — that's the intra-session continuity the original
    ``_last_plan`` field was designed to provide.

    The fix narrows the carry-forward to same-session only, so this
    behaviour must not regress.
    """
    runner = _mk_runner(_plan("static"))

    seeded_plan_at_turn_2: list[Plan | None] = []
    real_generate = runner.planner.generate

    async def _capturing_generate(*args: Any, **kwargs: Any) -> Plan:
        seeded_plan_at_turn_2.append(
            runner._last_session.plan if runner._last_session is not None else None
        )
        return await real_generate(*args, **kwargs)

    runner.planner.generate = _capturing_generate  # type: ignore[method-assign]

    out_1 = await runner.run("first turn", session_id="outer-session-X")
    assert out_1.success
    expected_plan_id = out_1.session.plan.id
    seeded_plan_at_turn_2.clear()

    # Turn 2 may abort downstream because StaticPlanner's revision-of-
    # the-same-plan attempt is rejected by the validator (no actual
    # change). That's fine — this test asserts on the SEED at the
    # planner-call boundary, which is the contract under test.
    await runner.run("second turn", session_id="outer-session-X")
    await runner.close()

    assert seeded_plan_at_turn_2, "planner.generate was not invoked on turn 2"
    seeded = seeded_plan_at_turn_2[0]
    assert seeded is not None
    # Same outer session id — turn 2's seed should carry turn 1's tasks.
    assert seeded.tasks, (
        "intra-session continuity regressed: turn 2's seed has no "
        "tasks despite sharing session.id with turn 1"
    )
    # And it should carry the SAME plan id — Phase 4 promise: plan_id
    # is the conversation's identity within a single session.
    assert seeded.id == expected_plan_id


async def test_pinned_session_does_not_leak_into_unpinned_turn() -> None:
    """A pinned-session turn followed by an unpinned (programmatic)
    turn must not leak the pinned session's plan.

    Mirrors a real pattern: an ADK-driven turn (session pinned to
    ``ctx.session.id``) followed by a bare ``runner.run("...")``
    call from test code or a different consumer.
    """
    runner = _mk_runner(_plan("static"))

    seeded_plan_observations: list[Plan | None] = []
    real_generate = runner.planner.generate

    async def _capturing_generate(*args: Any, **kwargs: Any) -> Plan:
        seeded_plan_observations.append(
            runner._last_session.plan if runner._last_session is not None else None
        )
        return await real_generate(*args, **kwargs)

    runner.planner.generate = _capturing_generate  # type: ignore[method-assign]

    # Turn 1: pinned to a specific outer session id.
    out_pinned = await runner.run("pinned turn", session_id="outer-pinned")
    assert out_pinned.success
    seeded_plan_observations.clear()

    # Turn 2: no session_id — Conversation mints a fresh uuid4 run_id.
    out_unpinned = await runner.run("unpinned turn")
    await runner.close()
    # The unpinned turn's session id must differ from the pinned one.
    assert out_unpinned.session.id != "outer-pinned"

    assert seeded_plan_observations, "planner.generate was not invoked on turn 2"
    seeded = seeded_plan_observations[0]
    assert seeded is not None
    assert not seeded.tasks, (
        "cross-session leak into unpinned turn: seed carries "
        f"{len(seeded.tasks)} task(s) from the pinned session's plan"
    )


# ---------------------------------------------------------------------------
# Unit-level coverage of :meth:`Conversation.prior_plan_for`
#
# The Runner-level tests above exercise the public-API contract; these
# pin the carry-forward matrix directly on the Conversation primitive
# so a future refactor that touches the matrix without touching the
# Runner still trips the regression net.
# ---------------------------------------------------------------------------


def test_conversation_prior_plan_unpinned_to_unpinned_carries_forward() -> None:
    from goldfive.conversation import Conversation

    convo = Conversation.new()
    sess = Session(run_id="programmatic-A", conversation_id=convo.id)
    sess.plan = _plan("first")
    convo.stash_plan(sess, pinned=False)

    # New turn, unpinned (different uuid4 run_id, as
    # next_turn_session() would mint).
    out = convo.prior_plan_for("programmatic-B", pinned=False)
    assert out is not None
    assert out.id == "plan-first"


def test_conversation_prior_plan_pinned_same_id_carries_forward() -> None:
    from goldfive.conversation import Conversation

    convo = Conversation.new()
    sess = Session(run_id="outer-X", conversation_id=convo.id)
    sess.plan = _plan("first")
    convo.stash_plan(sess, pinned=True)

    out = convo.prior_plan_for("outer-X", pinned=True)
    assert out is not None
    assert out.id == "plan-first"


def test_conversation_prior_plan_pinned_different_id_does_not_carry() -> None:
    from goldfive.conversation import Conversation

    convo = Conversation.new()
    sess = Session(run_id="outer-X", conversation_id=convo.id)
    sess.plan = _plan("first")
    convo.stash_plan(sess, pinned=True)

    assert convo.prior_plan_for("outer-Y", pinned=True) is None


def test_conversation_prior_plan_pin_state_mismatch_does_not_carry() -> None:
    from goldfive.conversation import Conversation

    convo = Conversation.new()
    sess = Session(run_id="outer-X", conversation_id=convo.id)
    sess.plan = _plan("first")
    convo.stash_plan(sess, pinned=True)

    # New turn unpinned: even though no session-id check would apply
    # for unpinned-vs-unpinned, a pinned prior crosses an externally-
    # owned boundary that must not leak.
    assert convo.prior_plan_for("any-id", pinned=False) is None

    # Symmetric: unpinned prior, then a new pinned turn establishes
    # an external owner; don't carry the bare prior into it.
    convo2 = Conversation.new()
    sess2 = Session(run_id="programmatic", conversation_id=convo2.id)
    sess2.plan = _plan("first")
    convo2.stash_plan(sess2, pinned=False)
    assert convo2.prior_plan_for("outer-Z", pinned=True) is None


def test_conversation_stash_plan_skips_empty_plan() -> None:
    from goldfive.conversation import Conversation

    convo = Conversation.new()
    sess = Session(run_id="programmatic-A", conversation_id=convo.id)
    sess.plan = Plan.empty(run_id=sess.run_id)  # no tasks
    convo.stash_plan(sess, pinned=False)

    # Empty seed never becomes a "prior" — same-shape lookup returns None.
    assert convo.prior_plan_for("programmatic-B", pinned=False) is None
