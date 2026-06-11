"""Tests for AGENCY-PRESERVATION.md task #11 (PR 9 follow-up):

1. Agent-scoped ``peek_for_render`` — a per-(agent,task) note reaches only
   its agent; broadcast (empty agent_id) notes reach any agent;
   ``agent_id=None`` keeps the pre-task-#11 no-filter behaviour.
2. Corrections enqueue ObserverNotes at write time (agent-scoped),
   closing the "written but unread under request_context" gap.
3. Cross-surface plan-state fold — ``render_block(plan=...)`` carries the
   Status line; the boundary-replay surface renders it identically.
"""

from __future__ import annotations

from goldfive.observer_note_queue import (
    OBSERVER_NOTE_MARKER_PREFIX,
    ObserverNoteQueue,
    plan_state_line,
    render_block,
)
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session() -> Session:
    return Session(run_id="r-scope")


def _enqueue(session: Session, *, drift_id: str, agent_id: str, severity: str) -> None:
    ObserverNoteQueue.for_session(session).enqueue(
        body=f"Observation: signal {drift_id}.",
        observation=f"signal {drift_id}",
        severity=severity,
        drift_id=drift_id,
        kind="looping_tool_call",
        task_id="t1",
        agent_id=agent_id,
    )


def _plan_two_agents() -> Plan:
    return Plan(
        id="p1",
        run_id="r-scope",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="t2",
                title="Review",
                assignee_agent_id="reviewer",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
        revision_index=1,
    )


# ---------------------------------------------------------------------------
# 1. Agent-scoped peek_for_render
# ---------------------------------------------------------------------------


def test_peek_agent_scope_routes_to_right_agent() -> None:
    session = _session()
    _enqueue(session, drift_id="dW", agent_id="writer", severity="warning")
    _enqueue(session, drift_id="dB", agent_id="", severity="info")  # broadcast
    _enqueue(session, drift_id="dR", agent_id="researcher", severity="critical")
    q = ObserverNoteQueue.for_session(session)

    # writer's surface: sees writer's note + broadcast, NOT researcher's
    # (even though researcher's is more severe).
    w = q.peek_for_render(agent_id="writer")
    assert w is not None and w.drift_id == "dW"

    # researcher's surface: sees researcher's (critical) note.
    r = q.peek_for_render(agent_id="researcher")
    assert r is not None and r.drift_id == "dR"

    # An agent with no specific note sees only the broadcast.
    other = q.peek_for_render(agent_id="planner")
    assert other is not None and other.drift_id == "dB"


def test_peek_no_agent_is_unfiltered() -> None:
    session = _session()
    _enqueue(session, drift_id="dW", agent_id="writer", severity="warning")
    _enqueue(session, drift_id="dR", agent_id="researcher", severity="critical")
    q = ObserverNoteQueue.for_session(session)
    # agent_id=None → no filter → most-severe overall (researcher).
    note = q.peek_for_render(agent_id=None)
    assert note is not None and note.drift_id == "dR"


def test_peek_agent_scope_is_bare_name_matched() -> None:
    session = _session()
    _enqueue(session, drift_id="dW", agent_id="ns:writer", severity="warning")
    q = ObserverNoteQueue.for_session(session)
    # Qualified note id matches the bare surface agent name.
    assert q.peek_for_render(agent_id="writer") is not None
    # And a different agent does not pick it up.
    assert q.peek_for_render(agent_id="reviewer") is None


# ---------------------------------------------------------------------------
# 2. Cross-surface plan-state fold (render_block / plan_state_line)
# ---------------------------------------------------------------------------


def test_render_block_without_plan_has_no_plan_state() -> None:
    session = _session()
    _enqueue(session, drift_id="d1", agent_id="", severity="warning")
    note = ObserverNoteQueue.for_session(session).peek_for_render()
    block = render_block(note)
    assert "Plan state (goldfive bookkeeping)" not in block
    assert block.count(OBSERVER_NOTE_MARKER_PREFIX) == 1


def test_render_block_with_plan_folds_status() -> None:
    session = _session()
    _enqueue(session, drift_id="d1", agent_id="", severity="warning")
    note = ObserverNoteQueue.for_session(session).peek_for_render()
    block = render_block(note, plan=_plan_two_agents())
    assert "Plan state (goldfive bookkeeping)" in block
    assert "writer: 1 open" in block
    assert "reviewer: no open tasks" in block
    # Still one block (fold is INSIDE the markers).
    assert block.count(OBSERVER_NOTE_MARKER_PREFIX) == 1
    assert "Choose the agent" not in block
    assert "do NOT re-invoke" not in block


def test_plan_state_line_empty_without_plan() -> None:
    assert plan_state_line(None) == ""
    assert plan_state_line(Plan(id="p", run_id="r", goal_ids=[], tasks=[], edges=[])) == ""


# ---------------------------------------------------------------------------
# 3. Corrections enqueue ObserverNotes at write time
# ---------------------------------------------------------------------------


def _revised_with_one_correct() -> Plan:
    return Plan(
        id="p1",
        run_id="r-scope",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected scope)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="research_solar_corrected")],
        revision_index=2,
    )


def _drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="scope drifted",
        current_task_id="research_solar",
        current_agent_id="research_agent",
        authored_by="goldfive",
    )


def test_corrections_via_notes_enqueues_agent_scoped_note() -> None:
    from goldfive._correction_injection import (
        is_pending_correction_key,
        queue_corrections_for_revision,
    )

    session = _session()
    ids = queue_corrections_for_revision(
        session=session,
        revised=_revised_with_one_correct(),
        prev_plan=None,
        drift=_drift(),
        corrections_via_notes=True,
    )
    assert len(ids) == 1
    # The correction rode the note queue, NOT the pending-correction slot.
    assert not any(is_pending_correction_key(k) for k in session.state)
    pend = ObserverNoteQueue.for_session(session).pending()
    assert len(pend) == 1
    note = pend[0]
    assert note.agent_id == "research_agent"  # agent-scoped
    assert note.task_id == "research_solar_corrected"
    assert "superseded" in note.body.lower() or "revised" in note.body.lower()
    # And it only renders to research_agent's surface, not a sibling's.
    q = ObserverNoteQueue.for_session(session)
    assert q.peek_for_render(agent_id="research_agent") is not None
    assert q.peek_for_render(agent_id="writer_agent") is None


def test_corrections_legacy_path_uses_slot_not_queue() -> None:
    from goldfive._correction_injection import (
        is_pending_correction_key,
        queue_corrections_for_revision,
    )

    session = _session()
    ids = queue_corrections_for_revision(
        session=session,
        revised=_revised_with_one_correct(),
        prev_plan=None,
        drift=_drift(),
        # corrections_via_notes defaults False — legacy slot path.
    )
    assert len(ids) == 1
    assert any(is_pending_correction_key(k) for k in session.state)
    assert ObserverNoteQueue.for_session(session).pending() == []


# ---------------------------------------------------------------------------
# 4. Cross-surface: the boundary-replay surface carries the fold
# ---------------------------------------------------------------------------


async def test_boundary_replay_surface_carries_plan_state() -> None:
    from goldfive.executors.sequential import SequentialExecutor

    session = _session()
    session.plan = _plan_two_agents()
    _enqueue(session, drift_id="d1", agent_id="", severity="warning")

    block = await SequentialExecutor()._consume_observer_note_for_replay(session)
    assert block is not None
    assert "Plan state (goldfive bookkeeping)" in block
    assert "writer: 1 open" in block


# ---------------------------------------------------------------------------
# 5. Coarse-surface defense (the channel-review MEDIUM): agent-specific
#    notes render ONLY on agent-aware surfaces.
# ---------------------------------------------------------------------------


def test_peek_broadcast_only_skips_agent_specific() -> None:
    session = _session()
    _enqueue(session, drift_id="dW", agent_id="writer", severity="critical")
    _enqueue(session, drift_id="dB", agent_id="", severity="info")
    q = ObserverNoteQueue.for_session(session)
    # broadcast_only delivers the broadcast (info) note even though the
    # agent-specific (critical) one is more severe.
    note = q.peek_for_render(broadcast_only=True)
    assert note is not None and note.drift_id == "dB"
    # With only an agent-specific note pending, broadcast_only → nothing.
    session2 = _session()
    _enqueue(session2, drift_id="dW", agent_id="writer", severity="critical")
    assert ObserverNoteQueue.for_session(session2).peek_for_render(broadcast_only=True) is None


async def test_boundary_replay_skips_agent_specific_note() -> None:
    from goldfive.executors.sequential import SequentialExecutor

    session = _session()
    _enqueue(session, drift_id="dW", agent_id="writer", severity="warning")
    executor = SequentialExecutor()
    # The boundary replay is coordinator-level → it must NOT deliver the
    # agent-specific note; it stays pending for writer's own surface.
    assert await executor._consume_observer_note_for_replay(session) is None
    pend = ObserverNoteQueue.for_session(session).pending()
    assert len(pend) == 1 and pend[0].drift_id == "dW"


def test_peek_excludes_correction_notes_for_tool_annotation() -> None:
    from goldfive.observer_note_queue import CORRECTION_DRIFT_ID_PREFIX

    session = _session()
    # A correction whose triggering drift WAS a loop kind — without the
    # structural exclusion it would leak onto the loop-only annotation.
    ObserverNoteQueue.for_session(session).enqueue(
        body="correction body",
        observation="corr",
        severity="warning",
        drift_id=f"{CORRECTION_DRIFT_ID_PREFIX}writer:t1:2",
        kind="looping_tool_call",
        task_id="t1",
        agent_id="writer",
    )
    _enqueue(session, drift_id="dLoop", agent_id="", severity="warning")  # real loop note
    q = ObserverNoteQueue.for_session(session)
    loop_kinds = frozenset({"looping_tool_call", "looping_reasoning"})
    note = q.peek_for_render(kinds=loop_kinds, exclude_correction_notes=True)
    assert note is not None and note.drift_id == "dLoop"  # the correction is excluded


async def test_claude_surface_is_agent_scoped() -> None:
    """claude.invoke runs one agent → its note surface scopes to that agent."""
    from goldfive.adapters.claude import ClaudeAgentSDKAdapter
    from goldfive.config import SteeringConfig
    from goldfive.steerer import DefaultSteerer

    # Bypass __init__ (which requires the claude-agent-sdk) — we only
    # exercise the SDK-independent _consume_observer_note path.
    adapter = object.__new__(ClaudeAgentSDKAdapter)
    adapter._steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False, signal_channel="request_context"
        )
    )
    session = _session()
    _enqueue(session, drift_id="dW", agent_id="writer", severity="warning")
    _enqueue(session, drift_id="dR", agent_id="researcher", severity="critical")

    block = await adapter._consume_observer_note(
        session, surface="claude_system_prompt", current_agent_id="writer"
    )
    # writer's surface gets writer's note, NOT researcher's (more severe,
    # wrong agent). researcher's note stays pending.
    assert block is not None and "signal dW" in block
    pend = {n.drift_id for n in ObserverNoteQueue.for_session(session).pending()}
    assert pend == {"dR"}
