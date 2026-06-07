"""Tests for cross-turn conversation state (issue #78).

Exercises the guarantees the Phase 3 design pins:

* A second ``runner.run()`` on the same Runner sees turn 1's
  ``completed_results`` in its ``Session`` and in the planner's context.
* ``runner.new_conversation()`` resets cross-turn state and changes
  ``runner.conversation_id``.
* ``runner.conversation_id`` is stable across turns.
* The planner's prompt includes prior-turn context.
* ``ConversationStarted`` / ``ConversationEnded`` lifecycle events
  bracket the conversation.
* Session carries ``conversation_id`` equal to the Runner's.
* Single-turn callers are unaffected (backward compatibility).
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive import (
    CallableAdapter,
    Conversation,
    Goal,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
    TaskStatus,
    TurnRecord,
)
from goldfive.conversation import Conversation as _ConversationClass  # noqa: F401

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_plan(suffix: str = "") -> Plan:
    """Two-task linear plan suitable for a single turn."""
    prefix = f"{suffix}_" if suffix else ""
    return Plan(
        id=f"plan-{prefix}fixture",
        run_id="",
        goal_ids=["g"],
        tasks=[
            Task(id=f"{prefix}research", title="Research", assignee_agent_id="writer"),
            Task(id=f"{prefix}draft", title="Draft", assignee_agent_id="writer"),
        ],
        edges=[
            TaskEdge(from_task_id=f"{prefix}research", to_task_id=f"{prefix}draft"),
        ],
        summary=f"Research then draft ({suffix})" if suffix else "Research then draft",
    )


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _kinds(events: list[Any]) -> list[str]:
    out: list[str] = []
    for e in events:
        if isinstance(e, dict):
            out.append(e.get("kind") or "")
            continue
        if hasattr(e, "WhichOneof"):
            name = e.WhichOneof("payload") or ""
            out.append(
                "".join(part.capitalize() for part in name.split("_"))
                if name
                else ""
            )
        else:
            out.append(getattr(e, "kind", ""))
    return out


# ---------------------------------------------------------------------------
# Conversation dataclass — pure unit tests
# ---------------------------------------------------------------------------


def test_conversation_new_generates_unique_id() -> None:
    a = Conversation.new()
    b = Conversation.new()
    assert a.id and b.id and a.id != b.id
    assert a.turns == [] and a.goals == [] and a.completed_results == {}


def test_conversation_next_turn_session_inherits_state() -> None:
    conv = Conversation.new()
    conv.goals = [Goal(id="g1", summary="greet")]
    conv.completed_results = {"t_prior": "hello"}

    s = conv.next_turn_session()

    assert s.conversation_id == conv.id
    assert s.run_id and s.run_id != conv.id
    # Copies, not aliases — mutating the session must not rewrite history.
    assert s.goals == conv.goals
    assert s.goals is not conv.goals
    assert s.completed_results == conv.completed_results
    assert s.completed_results is not conv.completed_results
    s.goals.append(Goal(id="g2", summary="new"))
    s.completed_results["t_new"] = "x"
    assert [g.id for g in conv.goals] == ["g1"]
    assert "t_new" not in conv.completed_results


def test_conversation_absorb_turn_merges_goals_and_results() -> None:
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    conv.goals = [Goal(id="g1", summary="first")]

    session = conv.next_turn_session()
    # Turn adds a new goal and one completed result.
    session.goals.append(Goal(id="g2", summary="second"))
    session.completed_results["task1"] = "researched"
    # goldfive#247: Plan is frozen — install via helper.
    from tests._immutable_plan_helpers import force_plan, force_task_status

    force_plan(session, _linear_plan())
    # Mark one task completed so the record reflects it.
    force_task_status(session, session.plan.tasks[0].id, TaskStatus.COMPLETED)

    outcome = ExecutionOutcome(success=True, session=session)
    record = conv.absorb_turn(outcome, user_input_summary="write a haiku")

    assert [g.id for g in conv.goals] == ["g1", "g2"]
    assert conv.completed_results == {"task1": "researched"}
    assert isinstance(record, TurnRecord)
    assert record.run_id == session.run_id
    assert record.user_input_summary == "write a haiku"
    assert record.outcome_success is True
    assert "research" in record.completed_task_ids
    assert len(conv.turns) == 1


def test_conversation_carries_completed_outputs_across_turns() -> None:
    """zicato#12: the full actual output is carried across turns alongside the
    self-reported summary, with later-turns-win merge semantics."""
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    conv.completed_results = {"t_prior": "summary line"}
    conv.completed_outputs = {"t_prior": "the full prior output with ID_99"}

    # A fresh turn session inherits BOTH maps as copies.
    s = conv.next_turn_session()
    assert s.completed_results == {"t_prior": "summary line"}
    assert s.completed_outputs == {"t_prior": "the full prior output with ID_99"}
    assert s.completed_outputs is not conv.completed_outputs

    # This turn produces its own (summary, full-output) pair.
    s.completed_results["t1"] = "Found the table."
    s.completed_outputs["t1"] = "row ID_x_y_V2 = 42 full dump"

    outcome = ExecutionOutcome(success=True, session=s)
    conv.absorb_turn(outcome, user_input_summary="q")

    assert conv.completed_outputs["t_prior"] == "the full prior output with ID_99"
    assert "ID_x_y_V2" in conv.completed_outputs["t1"]
    # Summary map remains the lossy metadata, unchanged in shape.
    assert conv.completed_results["t1"] == "Found the table."


def test_conversation_absorb_turn_deduplicates_goal_ids() -> None:
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    conv.goals = [Goal(id="g1", summary="already here")]
    session = conv.next_turn_session()
    # Session sees g1 from conversation seeding; if an executor added it
    # again by mistake, absorb_turn must not duplicate.
    session.goals.append(Goal(id="g1", summary="already here"))
    outcome = ExecutionOutcome(success=True, session=session)
    conv.absorb_turn(outcome)
    assert [g.id for g in conv.goals] == ["g1"]


def test_conversation_prior_turn_context_caps_window() -> None:
    conv = Conversation.new()
    # Manually seed five turn records.
    for i in range(5):
        conv.turns.append(TurnRecord(run_id=f"r{i}", user_input_summary=f"t{i}"))
    ctx = conv.prior_turn_context(recent_turns=3)
    assert len(ctx["prior_turns"]) == 3
    assert [t["run_id"] for t in ctx["prior_turns"]] == ["r2", "r3", "r4"]
    assert ctx["turn_index"] == 5
    assert ctx["conversation_id"] == conv.id


# ---------------------------------------------------------------------------
# Wire-sequence cursor across turns (goldfive#271 Gap 2)
# ---------------------------------------------------------------------------
#
# Validation v2 of issue #271 surfaced that ``Session._next_sequence``
# resets to 0 on every ``next_turn_session()`` call, which collides
# with harmonograf's persisted-event PK ``(session_id, run_id,
# sequence)`` whenever goldfive#161's outer-session pin makes
# ``Session.run_id`` repeat across turns. The Conversation-level cursor
# guarantees that wire sequences are globally unique within the
# conversation so persisted events never silently drop on a multi-turn
# session.


def test_conversation_seeds_session_sequence_from_running_cursor() -> None:
    conv = Conversation.new()

    s1 = conv.next_turn_session()
    assert s1.next_sequence() == 0
    assert s1.next_sequence() == 1
    # Manually advance to mimic a turn that emitted three events.
    assert s1.next_sequence() == 2

    # absorb_turn lifts the high-water mark back to the Conversation.
    from goldfive.results import ExecutionOutcome

    conv.absorb_turn(ExecutionOutcome(success=True, session=s1))
    assert conv._next_sequence == 3

    # Turn 2 picks up where turn 1 left off — no collision with turn 1's
    # already-emitted (session_id, run_id, 0/1/2) wire keys.
    s2 = conv.next_turn_session()
    assert s2.next_sequence() == 3
    assert s2.next_sequence() == 4

    conv.absorb_turn(ExecutionOutcome(success=True, session=s2))
    assert conv._next_sequence == 5


def test_conversation_sequence_unique_under_outer_session_pin() -> None:
    """Reproduces goldfive#271 Gap 2: with the outer-session pin
    forcing both turns to share ``run_id`` (and therefore ``session_id``,
    since ``Session.id`` aliases ``run_id``), every emitted ``(run_id,
    sequence)`` pair must remain globally unique across the whole
    conversation. Otherwise harmonograf's INSERT OR IGNORE silently
    drops the second turn's plan_submitted, agent_invocation_started,
    and similar early-turn events.
    """
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    pinned_run_id = "outer-adk-session-abc123"

    # Turn 1 — pinned to the outer session id, emits five events.
    s1 = conv.next_turn_session()
    s1.run_id = pinned_run_id
    turn1_keys = {(s1.run_id, s1.next_sequence()) for _ in range(5)}
    conv.absorb_turn(ExecutionOutcome(success=True, session=s1))

    # Turn 2 — same pin, emits five more events.
    s2 = conv.next_turn_session()
    s2.run_id = pinned_run_id
    turn2_keys = {(s2.run_id, s2.next_sequence()) for _ in range(5)}
    conv.absorb_turn(ExecutionOutcome(success=True, session=s2))

    # The collision regression: every (run_id, sequence) pair in turn 2
    # would equal one in turn 1 prior to the fix, and harmonograf's
    # storage layer would silently drop turn 2's events.
    assert turn1_keys.isdisjoint(turn2_keys), (
        "wire-key collision across turns under outer-session pin: "
        f"turn1={sorted(turn1_keys)} turn2={sorted(turn2_keys)}"
    )
    assert len(turn1_keys) == 5 and len(turn2_keys) == 5


def test_conversation_sequence_reset_on_new_conversation() -> None:
    """new_conversation() restarts the cursor at 0 — fresh
    Conversations carry a fresh wire-sequence keyspace."""
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    s1 = conv.next_turn_session()
    for _ in range(7):
        s1.next_sequence()
    conv.absorb_turn(ExecutionOutcome(success=True, session=s1))
    assert conv._next_sequence == 7

    fresh = Conversation.new()
    assert fresh._next_sequence == 0
    s_fresh = fresh.next_turn_session()
    assert s_fresh.next_sequence() == 0


def test_conversation_absorb_turn_is_robust_to_unstamped_session() -> None:
    """absorb_turn uses ``max(...)`` so an outcome whose Session never
    emitted (e.g. an early-abort path) doesn't accidentally roll the
    cursor backwards."""
    from goldfive.results import ExecutionOutcome

    conv = Conversation.new()
    # Pretend a prior turn already consumed sequences 0-9.
    conv._next_sequence = 10

    s = conv.next_turn_session()
    assert s._next_sequence == 10
    # An aborted turn never advances its own counter.
    conv.absorb_turn(ExecutionOutcome(success=False, session=s, reason="abort"))
    assert conv._next_sequence == 10  # Did not roll backwards.


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


async def test_runner_has_stable_conversation_id_across_turns() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    conv_id = runner.conversation_id
    assert conv_id

    await runner.run("turn one")
    await runner.run("turn two")
    await runner.close()

    assert runner.conversation_id == conv_id


async def test_runner_second_turn_session_sees_prior_completed_results() -> None:
    """The central guarantee: turn 2 sees turn 1's completed_results."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("turn one")
    assert out1.success
    # The executor populated completed_results via the reporting tools;
    # if the adapter didn't auto-complete, emulate it via the
    # SequentialExecutor's text-fallback path (which writes
    # completed_results[task.id] = text).
    assert out1.session.completed_results, "turn 1 should have completed results"
    turn1_results = dict(out1.session.completed_results)

    out2 = await runner.run("turn two")
    await runner.close()

    # Session for turn 2 was seeded with turn 1's completed_results.
    for k, v in turn1_results.items():
        assert k in out2.session.completed_results
        assert out2.session.completed_results[k] == v


async def test_runner_wire_sequences_unique_under_outer_session_pin() -> None:
    """End-to-end reproduction of goldfive#271 Gap 2.

    Drives two ``runner.run(...)`` calls with the same ``session_id``
    pin (mimicking goldfive#161's outer-session pin from
    :class:`GoldfiveADKAgent`). Asserts every emitted event has a
    globally-unique ``(session_id, run_id, sequence)`` triple — the
    exact tuple harmonograf's storage layer uses as its
    ``goldfive_events`` PK. Prior to the fix turn 2's early events
    (sequence 0, 1, 2, ...) collided with turn 1 and silently dropped
    on INSERT OR IGNORE.
    """
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    pinned = "outer-adk-session-deadbeef"

    await runner.run("turn one", session_id=pinned)
    await runner.run("turn two", session_id=pinned)
    await runner.close()

    # Filter out non-proto sidecar events (e.g. plan_revised correlation
    # dicts emitted by the steerer's _emit_plan_revised_correlation).
    proto_events = [evt for evt in sink.events if hasattr(evt, "session_id")]
    keys = [
        (evt.session_id, evt.run_id, int(evt.sequence)) for evt in proto_events
    ]
    # Every event must have a unique (session_id, run_id, sequence) so
    # harmonograf's PK never collides.
    assert len(keys) == len(set(keys)), (
        f"wire-key collision in sink stream: "
        f"duplicates={[k for k in keys if keys.count(k) > 1]}"
    )
    # And the pin actually held — both turns share the same session_id /
    # run_id, which is the precondition that creates the collision risk.
    assert {evt.run_id for evt in proto_events} == {pinned}


async def test_runner_session_carries_conversation_id() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    out1 = await runner.run("a")
    out2 = await runner.run("b")
    await runner.close()

    assert out1.session.conversation_id == runner.conversation_id
    assert out2.session.conversation_id == runner.conversation_id
    assert out1.session.run_id != out2.session.run_id


async def test_runner_new_conversation_resets_state() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    first_id = runner.conversation_id

    out1 = await runner.run("first conversation turn one")
    assert out1.success
    assert out1.session.completed_results

    await runner.new_conversation()
    assert runner.conversation_id != first_id

    await runner.run("second conversation, fresh")
    await runner.close()

    # Turn after reset must not see the earlier turn's task ids unless
    # the new plan itself produced them (our fixture uses the same
    # task ids, so check against the *prior-turn context* the planner
    # saw rather than completed_results).
    assert runner.conversation.turns, "the fresh conversation should record its own turn"
    # And the fresh conversation's record count is 1, not 2.
    assert len(runner.conversation.turns) == 1


async def test_runner_emits_conversation_lifecycle_events() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    await runner.run("turn one")
    await runner.run("turn two")
    await runner.new_conversation()
    await runner.run("fresh conversation")
    await runner.close()

    kinds = _kinds(sink.events)
    # One ConversationStarted per fresh conversation.
    assert kinds.count("ConversationStarted") == 2
    # One ConversationEnded for the first conversation (on reset) plus
    # one for the second (on close).
    assert kinds.count("ConversationEnded") == 2
    # The first ConversationStarted precedes the first RunStarted.
    assert kinds.index("ConversationStarted") < kinds.index("RunStarted")


async def test_runner_planner_sees_prior_turn_context() -> None:
    """LLMPlanner.generate receives prior-turn context on turn 2."""
    captured_prompts: list[str] = []

    async def capturing_llm(system: str, user: str, model: str) -> str:
        _ = system, model
        captured_prompts.append(user)
        # Return a valid plan JSON every time.
        return (
            '{"summary":"t","tasks":[{"id":"t1","title":"T","assignee_agent_id":"writer"}]}'
        )

    planner = LLMPlanner(call_llm=capturing_llm, model="stub")
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
        # Disable the turn-aware planning gate (planner-gate): this
        # test exercises the planner-context wiring on turn 2, so we
        # need Planner.generate to actually run on turn 2 — the gate
        # would otherwise classify the short "make it funnier" input
        # as conversational and skip the planner call.
        planner_gate=None,
    )
    out1 = await runner.run("write a limerick about cats")
    assert out1.success
    await runner.run("make it funnier")
    await runner.close()

    assert len(captured_prompts) == 2
    first_prompt, second_prompt = captured_prompts
    # First turn's prompt has no prior-turn block.
    assert "Prior-turn context" not in first_prompt
    # Second turn's prompt references prior turn summaries and results.
    assert "Prior-turn context" in second_prompt
    assert "Earlier turns" in second_prompt
    # And actually references turn 1's task output.
    assert "t1" in second_prompt


async def test_conversation_kwarg_lets_caller_inject_existing_state() -> None:
    """A caller can resume a conversation by passing a pre-built Conversation."""
    conv = Conversation.new()
    conv.goals = [Goal(id="existing", summary="already accumulated")]
    conv.completed_results = {"prior_task": "prior output"}

    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
        conversation=conv,
    )
    assert runner.conversation_id == conv.id

    out = await runner.run("continue")
    await runner.close()

    # The session for this "first" run (from Runner's perspective) was
    # actually seeded with injected prior state.
    assert "prior_task" in out.session.completed_results
    assert any(g.id == "existing" for g in out.session.goals)


async def test_runner_turn_records_accumulate() -> None:
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[InMemorySink()],
    )
    await runner.run("one")
    await runner.run("two")
    await runner.run("three")
    await runner.close()

    assert len(runner.conversation.turns) == 3
    run_ids = [t.run_id for t in runner.conversation.turns]
    assert len(set(run_ids)) == 3


async def test_single_turn_backward_compatibility() -> None:
    """A single-turn caller that never references Conversation still works."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )
    out = await runner.run("one shot")
    await runner.close()

    assert out.success
    # Plain users still get a well-formed outcome + event stream.
    kinds = _kinds(sink.events)
    assert "RunStarted" in kinds
    assert "RunCompleted" in kinds


# ---------------------------------------------------------------------------
# Convenience: goldfive.wrap() builds a Runner with a Conversation
# ---------------------------------------------------------------------------


async def test_wrap_runner_has_conversation() -> None:
    """goldfive.wrap() produces a Runner that already owns a Conversation."""
    from goldfive import wrap

    async def agent_fn(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = tools, session
        return InvocationResult(task_id=task.id, text="ok")

    runner = wrap(
        CallableAdapter(agent_fn, available_agents=["writer"]),
        planner=StaticPlanner(_linear_plan()),
        sinks=[InMemorySink()],
    )
    assert runner.conversation_id
    await runner.close()


# ---------------------------------------------------------------------------
# Proto round-trip smoke test
# ---------------------------------------------------------------------------


def test_conversation_events_serialize_over_proto() -> None:
    """ConversationStarted / ConversationEnded survive proto encoding."""
    pytest.importorskip("goldfive.pb.goldfive.v1.events_pb2")
    from goldfive.events import conversation_ended_event, conversation_started_event
    from goldfive.pb.goldfive.v1 import events_pb2

    started = conversation_started_event(run_id="r1", sequence=0, conversation_id="c1")
    assert started.WhichOneof("payload") == "conversation_started"
    assert started.conversation_started.conversation_id == "c1"

    blob = started.SerializeToString()
    roundtrip = events_pb2.Event()
    roundtrip.ParseFromString(blob)
    assert roundtrip.conversation_started.conversation_id == "c1"

    ended = conversation_ended_event(
        run_id="r1", sequence=5, conversation_id="c1", turn_count=3, reason="close"
    )
    assert ended.conversation_ended.turn_count == 3
    assert ended.conversation_ended.reason == "close"
