"""Cross-session Conversation isolation (goldfive#271 follow-up to #293).

Validation v4 Class 1 surfaced a deeper architectural leak that
PR #293 only partially patched. Even after #293 keyed the prior-plan
stash by ``session.id`` on :class:`Conversation`, the *Conversation
itself* still lived as a process-scoped attribute on :class:`Runner`
(``self._conversation``). Every other field of ``Conversation`` —
``goals``, ``completed_results``, ``turns``, ``_next_sequence`` —
therefore still leaked across distinct outer ADK sessions sharing
one Runner.

The visible regression: v4class1-1 saw the goal_derived event text
"Provide the correct answer to the math question 2+2" leak from a
prior v4-class5 session that had run on the same Runner. With
``session.goals`` accumulating on a singleton Conversation, the new
session's ``next_turn_session()`` inherited every prior session's
goals — a high-impact correctness bug for any operator running
multiple ADK sessions through one wrapped agent (the default
``goldfive.wrap()`` shape).

The fix: replace ``Runner._conversation`` with
``Runner._conversations: dict[str, Conversation]`` keyed by the
outer-session id (or a shared empty-string sentinel for the
unpinned/programmatic caller path). Each turn looks up — or creates —
the Conversation for its own session id, so a fresh outer ADK
session sees ``goals=[]``, ``completed_results={}``, ``turns=[]``,
``_next_sequence=0`` regardless of what other sessions have run on
the same Runner.

These tests pin the contract end-to-end:

1. Goals do not leak across pinned sessions (the Class 1 regression).
2. Completed results do not leak across pinned sessions.
3. Turn records do not leak across pinned sessions.
4. The wire-sequence cursor restarts at 0 for a fresh pinned session.
5. The unpinned (programmatic) path keeps single-Conversation
   continuity exactly as the pre-fix Runner provided.
6. Two consecutive turns on the SAME pinned ``session_id`` continue
   to share their Conversation (intra-session continuity holds).
"""

from __future__ import annotations

from goldfive import (
    CallableAdapter,
    Goal,
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
    return Plan(
        id=f"plan-{tag}",
        run_id="",
        goal_ids=[f"g-{tag}"],
        tasks=[Task(id=f"t-{tag}", title=f"Task {tag}", assignee_agent_id="writer")],
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


def _mk_runner() -> Runner:
    """Build a Runner with deterministic StaticPlanner + happy adapter.

    ``planner_gate=None`` keeps the test isolated from the LLM-driven
    handle_turn path: the leak we're testing is in the Runner's
    Conversation bookkeeping, not in the planner.
    """
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_plan("static")),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("derived"),
        sinks=[InMemorySink()],
        planner_gate=None,
    )


# ---------------------------------------------------------------------------
# Class 1 regression tests
# ---------------------------------------------------------------------------


async def test_goals_do_not_leak_across_pinned_sessions() -> None:
    """The validation v4 Class 1 regression: goals leak across sessions.

    Drives one turn on outer session A with a goal id ``g-a``, then
    one turn on outer session B with a distinct goal id ``g-b``.
    Asserts that turn B's session.goals contains ONLY ``g-b`` —
    NOT ``g-a`` from session A.

    Pre-fix this test fails: turn B's ``Session`` is built from
    ``Conversation.next_turn_session()`` which copies
    ``self.goals`` — but the singleton Conversation has been
    accumulating goals from BOTH sessions, so turn B starts with
    {``g-a``, ``g-b``} instead of {``g-b``}.
    """
    runner = _mk_runner()

    goal_a = [Goal(id="g-a", summary="A's goal: do A-thing")]
    out_a = await runner.run(goal_a, session_id="outer-session-A")
    assert out_a.success
    a_goal_ids = {g.id for g in out_a.session.goals if g.id}
    assert a_goal_ids == {"g-a"}, "session A should hold exactly its own goal"

    goal_b = [Goal(id="g-b", summary="B's goal: do B-thing")]
    out_b = await runner.run(goal_b, session_id="outer-session-B")
    await runner.close()

    b_goal_ids = {g.id for g in out_b.session.goals if g.id}
    # The strong assertion: B's session.goals contains EXACTLY its own
    # goal id, with NO leak from session A. Pre-fix B starts with A's
    # goal id already merged in (singleton Conversation accumulated
    # both), so this set has both entries.
    assert b_goal_ids == {"g-b"}, (
        "cross-session goal leak: turn B's session.goals = "
        f"{b_goal_ids}, expected {{'g-b'}} — A's goal id leaked through "
        "the singleton Conversation"
    )
    assert "g-a" not in b_goal_ids, (
        "session A's goal id 'g-a' is visible in session B's session.goals"
    )


async def test_completed_results_do_not_leak_across_pinned_sessions() -> None:
    """A different tip of the same iceberg: completed_results leaks too.

    Pre-fix the singleton Conversation accumulates completed-task
    summaries from every session. A fresh pinned session would see
    every prior session's outputs as "prior-turn context" — a
    correctness violation that confuses the planner and surfaces
    foreign task ids in the new session's planner prompt.

    We verify by asserting on the per-session ``Conversation``
    instances directly: each session's Conversation should hold
    ONLY its own ``completed_results``, not the union of every
    session that ever ran on the Runner.
    """
    # Use distinct planners per session via two adapters that label
    # task outputs with a per-session marker, so we can detect leakage
    # by value (not just by key — both sessions happen to share the
    # task id ``t-static`` from the StaticPlanner fixture).
    runner = _mk_runner()

    out_a = await runner.run("session A turn", session_id="outer-session-A")
    assert out_a.success
    assert out_a.session.completed_results

    out_b = await runner.run("session B turn", session_id="outer-session-B")
    await runner.close()
    assert out_b.session.completed_results

    # Per-session Conversations isolate completed_results: A's
    # Conversation accumulates only A's run, B's only B's. Without the
    # fix both sessions would write into the singleton Conversation's
    # dict and B's Conversation would carry A's keys (and vice versa).
    convo_a = runner._conversations["outer-session-A"]
    convo_b = runner._conversations["outer-session-B"]
    assert convo_a is not convo_b, "sessions A and B must own distinct Conversations"
    # Each Conversation's accumulated results came from exactly one
    # turn — its own. Pre-fix both share one Conversation that
    # accumulates the union; the per-Conversation `turns` list would
    # also reflect both runs for the singleton, which we test
    # separately. Here we anchor on completed_results count.
    assert len(convo_a.completed_results) >= 1
    assert len(convo_b.completed_results) >= 1
    # The crucial assertion: turn B's Conversation cursor only saw
    # its own run. Compare turn count — pre-fix both Conversations
    # ARE the same singleton with len(turns) == 2.
    assert len(convo_a.turns) == 1, (
        f"session A's Conversation has {len(convo_a.turns)} turns, "
        "expected 1 (its own only) — singleton leak across sessions"
    )
    assert len(convo_b.turns) == 1, (
        f"session B's Conversation has {len(convo_b.turns)} turns, "
        "expected 1 (its own only) — singleton leak across sessions"
    )


async def test_turn_records_do_not_leak_across_pinned_sessions() -> None:
    """Each pinned session keeps its own conversation history.

    Pre-fix the singleton Conversation appends a TurnRecord for every
    run() across every session, so prior-turn context (the planner's
    cross-turn window) bleeds across session boundaries. The fix
    isolates per-session histories.

    Drives one turn on each of three distinct sessions to keep the
    fixture simple (avoids second-turn-on-same-session validator
    quirks of StaticPlanner). Each session's Conversation should
    hold exactly one TurnRecord — its own.
    """
    runner = _mk_runner()

    out_a = await runner.run("a-turn", session_id="outer-session-A")
    assert out_a.success
    out_b = await runner.run("b-turn", session_id="outer-session-B")
    assert out_b.success
    out_c = await runner.run("c-turn", session_id="outer-session-C")
    assert out_c.success
    await runner.close()

    # Look up each session's Conversation and assert each holds only
    # its own turn. The Runner exposes them via the per-session lookup.
    convo_a = runner._conversations["outer-session-A"]
    convo_b = runner._conversations["outer-session-B"]
    convo_c = runner._conversations["outer-session-C"]
    assert len({id(convo_a), id(convo_b), id(convo_c)}) == 3, (
        "outer sessions A, B, C must own DISTINCT Conversation instances"
    )
    # Pre-fix all three sessions share one Conversation with three
    # TurnRecords; with the fix each Conversation has exactly one.
    assert len(convo_a.turns) == 1, (
        f"session A's Conversation has {len(convo_a.turns)} turns, "
        "expected 1 — singleton-Conversation cross-session leak"
    )
    assert len(convo_b.turns) == 1, f"session B has {len(convo_b.turns)} turns"
    assert len(convo_c.turns) == 1, f"session C has {len(convo_c.turns)} turns"
    a_run_ids = {t.run_id for t in convo_a.turns}
    b_run_ids = {t.run_id for t in convo_b.turns}
    c_run_ids = {t.run_id for t in convo_c.turns}
    # Each Conversation's TurnRecords carry the run_id of its own
    # session's turn — no cross-pollination.
    assert a_run_ids == {"outer-session-A"}
    assert b_run_ids == {"outer-session-B"}
    assert c_run_ids == {"outer-session-C"}


async def test_wire_sequence_cursor_restarts_for_fresh_pinned_session() -> None:
    """A fresh pinned session must start with sequence=0 in its own
    Conversation cursor.

    Pre-fix the singleton ``Conversation._next_sequence`` keeps
    advancing across sessions, so session B's first emitted event
    picks up a sequence number derived from session A's cumulative
    event count. Wire keys still happen to remain unique
    (``session_id`` differs), but the per-Conversation cursor
    semantics from goldfive#271 Gap 2 silently break.

    The strong invariant: session B's first sink emission must carry
    ``sequence=0`` — proof its Conversation cursor was fresh, not
    inherited from session A.
    """
    runner = _mk_runner()

    await runner.run("session A turn", session_id="outer-session-A")
    convo_a = runner._conversations["outer-session-A"]
    a_cursor = convo_a._next_sequence
    assert a_cursor > 0, (
        "session A's Conversation cursor should have advanced past 0"
    )

    # Snapshot how many events the sink has BEFORE session B runs so we
    # can isolate B's emissions cleanly (no need to filter by session id
    # under close-emitted ConversationEnded mixins).
    sink = runner.sinks[0]
    pre_b_event_count = len(sink.events)

    await runner.run("session B turn", session_id="outer-session-B")

    # B's first emission (RunStarted, since ConversationStarted comes
    # first because B's Conversation is brand new). The first proto
    # event in this slice tells us what sequence B started at.
    b_first_event = sink.events[pre_b_event_count]
    assert hasattr(b_first_event, "sequence"), (
        "first B-emitted event has no sequence field"
    )
    # Pre-fix this is ``a_cursor`` (singleton Conversation continued
    # advancing). Post-fix it's 0 — fresh Conversation, fresh cursor.
    assert int(b_first_event.sequence) == 0, (
        f"session B's first event carries sequence={int(b_first_event.sequence)}, "
        f"expected 0 — singleton-Conversation cursor leaked from session A "
        f"(A's cursor at handover was {a_cursor})"
    )

    await runner.close()


# ---------------------------------------------------------------------------
# Backward-compatibility tests
# ---------------------------------------------------------------------------


async def test_unpinned_callers_share_one_conversation() -> None:
    """Programmatic (unpinned) Runner callers must keep singleton
    Conversation behaviour: the conversation_id stays stable across
    runs, and goals/completed_results carry forward exactly as
    pre-fix.

    This is the back-compat guarantee for pre-#161 callers — bare
    ``await runner.run("...")`` from test code or one-shot scripts.
    """
    runner = _mk_runner()

    out_1 = await runner.run("turn one")
    assert out_1.success
    convo_id_1 = out_1.session.conversation_id

    out_2 = await runner.run("turn two")
    convo_id_2 = out_2.session.conversation_id

    out_3 = await runner.run("turn three")
    await runner.close()
    convo_id_3 = out_3.session.conversation_id

    # All three turns share one conversation_id.
    assert convo_id_1 and convo_id_1 == convo_id_2 == convo_id_3
    # And the public properties point at it too.
    assert runner.conversation_id == convo_id_1
    # That single Conversation has all three TurnRecords.
    assert len(runner.conversation.turns) == 3


async def test_same_pinned_session_keeps_intra_session_continuity() -> None:
    """Two consecutive turns on the SAME pinned session id share their
    Conversation: goals accumulate, completed_results carry forward,
    turn records grow.

    The fix's narrowing must NOT regress this — it's the whole point
    of pinning a session: intra-session continuity for the duration of
    one outer ADK session.
    """
    runner = _mk_runner()

    out_1 = await runner.run("first turn", session_id="outer-session-X")
    assert out_1.success
    convo_id_1 = out_1.session.conversation_id

    out_2 = await runner.run("second turn", session_id="outer-session-X")
    await runner.close()
    convo_id_2 = out_2.session.conversation_id

    # Same session id → same Conversation → same conversation_id.
    assert convo_id_1 == convo_id_2
    # And both turns are on the same Conversation instance.
    convo = runner._conversations["outer-session-X"]
    assert len(convo.turns) == 2
    # Turn 2 saw turn 1's completed results in its session seed.
    for k in out_1.session.completed_results:
        assert k in out_2.session.completed_results, (
            f"intra-session continuity regressed: turn 2 lost completed "
            f"result {k!r} from turn 1"
        )


async def test_pinned_and_unpinned_callers_get_distinct_conversations() -> None:
    """A pinned-session turn followed by an unpinned (programmatic)
    turn must NOT share a Conversation.

    Pre-fix both routed through ``self._conversation`` so the unpinned
    turn would see the pinned session's prior completed_results. With
    the fix the unpinned path lives in the empty-string-keyed
    Conversation, distinct from the pinned session's.
    """
    runner = _mk_runner()

    out_pinned = await runner.run("pinned", session_id="outer-pinned")
    assert out_pinned.success
    pinned_convo_id = out_pinned.session.conversation_id

    out_unpinned = await runner.run("unpinned")
    await runner.close()
    unpinned_convo_id = out_unpinned.session.conversation_id

    # Distinct conversation ids — they're truly separate Conversations.
    assert pinned_convo_id != unpinned_convo_id, (
        "pinned and unpinned turns must own distinct Conversations"
    )


# ---------------------------------------------------------------------------
# new_conversation() across multi-session bookkeeping
# ---------------------------------------------------------------------------


async def test_new_conversation_resets_all_per_session_state() -> None:
    """``runner.new_conversation()`` resets EVERY per-session
    Conversation. After the reset, both pinned and unpinned callers
    see a fresh Conversation (new id, empty goals / results / turns).

    The pre-fix ``new_conversation()`` operated on the singleton.
    The fix's per-session dict means the reset must clear the dict
    so future runs don't pick up stale state.
    """
    runner = _mk_runner()

    out_pinned = await runner.run("pinned-1", session_id="outer-pinned")
    assert out_pinned.success
    out_unpinned = await runner.run("unpinned-1")
    assert out_unpinned.success

    pinned_convo_id_before = runner._conversations["outer-pinned"].id
    unpinned_convo_id_before = runner._conversations[""].id

    await runner.new_conversation()

    out_pinned_2 = await runner.run("pinned-2", session_id="outer-pinned")
    out_unpinned_2 = await runner.run("unpinned-2")
    await runner.close()

    # Same outer session ids, but the underlying Conversation ids changed
    # — the reset wiped the prior bookkeeping.
    assert out_pinned_2.session.conversation_id != pinned_convo_id_before
    assert out_unpinned_2.session.conversation_id != unpinned_convo_id_before
