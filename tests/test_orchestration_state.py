"""Tests for the goldfive.* session-state namespace (goldfive#152).

Covers the six pieces of the issue:

1. ``PlanReconciler`` stamps / clears ``goldfive.current_task_*`` on
   RUNNING / terminal transitions.
2. ``DefaultSteerer`` USER_STEER handler writes ``goldfive.active_steer.*``.
3. USER_STEER synthesizes a Goal via the planner and appends it.
4. USER_STEER synthesize returning ``mode=replace`` clears goals.
5. ``_compose_steer_restart_message`` produces the override-framed body.
6. ``_heal_pending_tool_calls`` stamps
   ``goldfive.cancelled_function_call_ids`` on session.state.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import orchestration_state as _ostate  # noqa: E402
from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.reconciler import PlanReconciler  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    """Minimal planner: records refine + synthesize calls."""

    def __init__(
        self,
        *,
        synth_result: Any = None,
        revised_plan: Plan | None = None,
    ) -> None:
        self.synth_result = synth_result
        self.revised_plan = revised_plan
        self.refine_calls: list[dict[str, Any]] = []
        self.synth_calls: list[str] = []

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
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": list(goals)})
        return self.revised_plan

    async def synthesize_goal_from_steer(self, body: str) -> Any:
        self.synth_calls.append(body)
        return self.synth_result


def _plan_two_running() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="First", assignee_agent_id="agent_a"),
            Task(id="t2", title="Second", assignee_agent_id="agent_b"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_session_with_plan() -> Session:
    plan = _plan_two_running()
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="original goal")],
        plan=plan,
    )


# ---------------------------------------------------------------------------
# 1. PlanReconciler stamps and clears goldfive.current_task_*
# ---------------------------------------------------------------------------


async def test_orchestration_state_keys_lifecycle() -> None:
    """Reconciler stamps current_task_* on RUNNING, clears on COMPLETED.

    Exercises the full lifecycle: initial state empty, stamped on
    before_agent, cleared on after_agent once the task terminates.
    """
    session = _make_session_with_plan()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())

    reconciler = PlanReconciler(session=session, steerer=steerer)

    # Pre-flight: no current_task_* keys yet.
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID, "") == ""

    # Claim t1.
    await reconciler.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID) == "t1"
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_TITLE) == "First"

    # Close t1.
    await reconciler.on_after_agent(agent_name="agent_a", invocation_id="inv1")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID, "") == ""
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_TITLE, "") == ""

    # Claim t2.
    await reconciler.on_before_agent(agent_name="agent_b", invocation_id="inv2")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID) == "t2"
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_TITLE) == "Second"

    # Close t2 → cleared again.
    await reconciler.on_after_agent(agent_name="agent_b", invocation_id="inv2")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID, "") == ""


# ---------------------------------------------------------------------------
# 2. DefaultSteerer USER_STEER writes goldfive.active_steer.*
# ---------------------------------------------------------------------------


async def test_user_steer_writes_active_steer_state() -> None:
    """STEER ControlMessage → USER_STEER drift → active_steer.* stamped."""
    session = _make_session_with_plan()
    planner = _StubPlanner(revised_plan=None)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=planner)

    # Baseline: no active steer state.
    assert _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_BODY, "") == ""

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "focus on the writing instead"},
    )
    await steerer.observe(msg, session)

    body = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_BODY, "")
    assert body == "focus on the writing instead"
    # at_turn is an int and should be > 0 (emit_drift_detected bumped
    # the sequence counter).
    at_turn = _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_AT_TURN, 0)
    assert isinstance(at_turn, int)


# ---------------------------------------------------------------------------
# 3. USER_STEER synthesizes and appends a Goal (default mode).
# ---------------------------------------------------------------------------


async def test_user_steer_synthesizes_and_appends_goal() -> None:
    """Planner returns (Goal, 'append') → session.goals grows by one."""
    session = _make_session_with_plan()
    synth_goal = Goal(id="steer", summary="focus on the writing instead")
    planner = _StubPlanner(synth_result=(synth_goal, "append"))
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=planner)

    assert [g.id for g in session.goals] == ["g1"]

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "focus on the writing instead"},
    )
    await steerer.observe(msg, session)

    # Planner was asked to synthesize.
    assert planner.synth_calls == ["focus on the writing instead"]
    # Goal was appended.
    assert [g.id for g in session.goals] == ["g1", "steer"]
    assert session.goals[-1].summary == "focus on the writing instead"
    # goals_summary on the state dict refreshed.
    summary = _ostate.read(session.state, _ostate.KEY_GOALS_SUMMARY, "")
    assert "[g1]" in summary
    assert "[steer]" in summary
    assert "focus on the writing" in summary


# ---------------------------------------------------------------------------
# 4. USER_STEER synthesize decides REPLACE → goals cleared + replaced.
# ---------------------------------------------------------------------------


async def test_user_steer_synthesizes_replace_when_scrap_steer() -> None:
    """Planner returns mode='replace' → session.goals collapses to one."""
    session = _make_session_with_plan()
    # Seed two prior goals so we can prove replace clears BOTH.
    session.goals.append(Goal(id="g2", summary="secondary"))
    assert len(session.goals) == 2

    replacement = Goal(id="new", summary="actually, do this instead")
    planner = _StubPlanner(synth_result=(replacement, "replace"))
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=planner)

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "actually, do this instead"},
    )
    await steerer.observe(msg, session)

    assert [g.id for g in session.goals] == ["new"]
    assert session.goals[0].summary == "actually, do this instead"
    summary = _ostate.read(session.state, _ostate.KEY_GOALS_SUMMARY, "")
    # Prior goal ids are gone from the summary.
    assert "g1" not in summary
    assert "g2" not in summary
    assert "[new]" in summary


async def test_user_steer_falls_back_when_planner_lacks_synthesize() -> None:
    """Planner without ``synthesize_goal_from_steer`` → passthrough append."""

    class _MinimalPlanner:
        refine_calls: list[dict[str, Any]] = []  # noqa: RUF012

        async def generate(self, **kwargs: Any) -> Plan | None:
            return None

        async def refine(self, **kwargs: Any) -> Plan | None:
            self.refine_calls.append(kwargs)
            return None

    session = _make_session_with_plan()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_MinimalPlanner())

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "tangential aside"},
    )
    await steerer.observe(msg, session)

    # Passthrough goal appended with id 'steer'.
    assert len(session.goals) == 2
    assert session.goals[-1].id == "steer"
    assert session.goals[-1].summary == "tangential aside"


# ---------------------------------------------------------------------------
# 5. _compose_steer_restart_message shape
# ---------------------------------------------------------------------------


def test_compose_steer_restart_message_shape() -> None:
    """Override header + body + notes block all present in the output."""
    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "refactor the intro paragraph"},
    )
    composed = SequentialExecutor._compose_steer_restart_message(
        msg, fallback="prior input"
    )
    # Header present (leading line).
    assert composed.startswith("[USER STEERING CONTROL — supersedes prior task context]")
    # Body interpolated verbatim.
    assert "refactor the intro paragraph" in composed
    # Notes block present.
    assert "Notes:" in composed
    assert "superseded unless this message explicitly" in composed
    assert "Proceed with the new direction" in composed


def test_compose_steer_restart_message_empty_uses_fallback() -> None:
    """Empty steer body falls back to ``fallback`` (still wrapped)."""
    msg = ControlMessage(kind=ControlKind.STEER, payload={"note": ""})
    composed = SequentialExecutor._compose_steer_restart_message(
        msg, fallback="the original request"
    )
    assert composed.startswith("[USER STEERING CONTROL")
    assert "the original request" in composed


def test_compose_steer_restart_message_accepts_body_key() -> None:
    """``payload['body']`` is accepted as a courtesy key."""
    msg = ControlMessage(kind=ControlKind.STEER, payload={"body": "switch gears"})
    composed = SequentialExecutor._compose_steer_restart_message(
        msg, fallback="fallback"
    )
    assert "switch gears" in composed


# ---------------------------------------------------------------------------
# 6. _heal_pending_tool_calls stamps cancelled_function_call_ids
# ---------------------------------------------------------------------------


async def test_cancelled_function_call_ids_populated_on_heal() -> None:
    """Heal path records healed function_call ids on session.state."""
    # Avoid requiring the ADK optional dependency by unit-testing the
    # state-stamp helper the heal path uses. The adapter-integration
    # side is covered by tests/test_cancel_propagation.py + the
    # existing ADK heal tests; here we verify the orchestration-state
    # stamp semantics.
    session = _make_session_with_plan()

    # Baseline: empty list.
    assert _ostate.read_cancelled_function_call_ids(session.state) == []

    _ostate.append_cancelled_function_call_ids(session.state, ["call-1", "call-2"])
    assert _ostate.read_cancelled_function_call_ids(session.state) == [
        "call-1",
        "call-2",
    ]

    # Second heal with overlap → de-duplicated, order preserved.
    _ostate.append_cancelled_function_call_ids(session.state, ["call-2", "call-3"])
    assert _ostate.read_cancelled_function_call_ids(session.state) == [
        "call-1",
        "call-2",
        "call-3",
    ]


async def test_cancelled_function_call_ids_populated_on_adk_heal() -> None:
    """End-to-end: trigger the ADK adapter's heal path and verify the stamp.

    Skipped when the optional ADK dependency is not installed.
    """
    try:
        from google.adk.events import Event  # noqa: F401
    except ImportError:
        pytest.skip("google-adk not installed")

    from goldfive.adapters.adk import ADKAdapter

    # Build a minimal ADKAdapter and drive _heal_pending_tool_calls
    # directly with a session. No actual ADK runner; we stub the
    # pieces the heal helper touches (session_service, append_event).
    class _StubSessionService:
        def __init__(self) -> None:
            self.appended: list[Any] = []

        async def create_session(self, **kwargs: Any) -> Any:
            return object()

        async def get_session(self, **kwargs: Any) -> Any:
            return object()

        async def append_event(self, *, session: Any, event: Any) -> None:  # noqa: ARG002
            self.appended.append(event)

    class _StubRunner:
        app_name = "test"
        session_service = _StubSessionService()

    class _StubAgent:
        name = "host"

    adapter = ADKAdapter.__new__(ADKAdapter)
    adapter._runner = _StubRunner()
    adapter._agent = _StubAgent()
    adapter._user_id = "u1"
    adapter._app_name = "test"
    adapter._pending_tool_call_ids = {"call-a", "call-b"}
    adapter._pending_tool_call_names = {"call-a": "tool1", "call-b": "tool2"}

    session = _make_session_with_plan()

    await adapter._heal_pending_tool_calls(
        runner=adapter._runner,
        session_id="s1",
        invocation_id="inv-1",
        reason="cancelled_mid_invocation",
        session=session,
    )

    healed = _ostate.read_cancelled_function_call_ids(session.state)
    assert set(healed) == {"call-a", "call-b"}


# ---------------------------------------------------------------------------
# Cross-cutting: DefaultSteerer.mark_task_running also stamps state
# ---------------------------------------------------------------------------


async def test_mark_task_running_stamps_orchestration_state() -> None:
    """Legacy (non-overlay) path: transitions still stamp state keys."""
    session = _make_session_with_plan()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())

    await steerer.mark_task_running("t1", session=session, detail="starting")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID) == "t1"
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_TITLE) == "First"

    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert _ostate.read(session.state, _ostate.KEY_CURRENT_TASK_ID, "") == ""


# ---------------------------------------------------------------------------
# goldfive#171 — processed_steer_ids helpers + author on set_active_steer
# ---------------------------------------------------------------------------


def test_record_processed_steer_id_appends_and_dedupes() -> None:
    state: dict[str, Any] = {}
    _ostate.record_processed_steer_id(state, "ann_1")
    _ostate.record_processed_steer_id(state, "ann_2")
    _ostate.record_processed_steer_id(state, "ann_1")  # duplicate — dropped
    assert state[_ostate.KEY_PROCESSED_STEER_IDS] == ["ann_1", "ann_2"]


def test_record_processed_steer_id_empty_is_noop() -> None:
    state: dict[str, Any] = {}
    _ostate.record_processed_steer_id(state, "")
    assert _ostate.KEY_PROCESSED_STEER_IDS not in state


def test_has_processed_steer_id_reports_membership() -> None:
    state: dict[str, Any] = {}
    assert _ostate.has_processed_steer_id(state, "x") is False
    _ostate.record_processed_steer_id(state, "x")
    assert _ostate.has_processed_steer_id(state, "x") is True
    assert _ostate.has_processed_steer_id(state, "y") is False
    # Tolerates malformed state.
    assert (
        _ostate.has_processed_steer_id({"goldfive.processed_steer_ids": "oops"}, "x")
        is False
    )
    assert _ostate.has_processed_steer_id(state, "") is False


def test_record_processed_steer_id_evicts_fifo_at_cap() -> None:
    state: dict[str, Any] = {}
    cap = _ostate.PROCESSED_STEER_IDS_CAP
    for i in range(cap + 2):
        _ostate.record_processed_steer_id(state, f"id_{i}")
    ids = state[_ostate.KEY_PROCESSED_STEER_IDS]
    assert len(ids) == cap
    assert "id_0" not in ids
    assert "id_1" not in ids
    assert ids[0] == "id_2"
    assert ids[-1] == f"id_{cap + 1}"


def test_set_active_steer_writes_author_and_clears_it() -> None:
    state: dict[str, Any] = {}
    _ostate.set_active_steer(state, body="pivot", at_turn=7, author="alice")
    assert state[_ostate.KEY_ACTIVE_STEER_BODY] == "pivot"
    assert state[_ostate.KEY_ACTIVE_STEER_AT_TURN] == 7
    assert state[_ostate.KEY_ACTIVE_STEER_AUTHOR] == "alice"

    _ostate.clear_active_steer(state)
    assert _ostate.KEY_ACTIVE_STEER_BODY not in state
    assert _ostate.KEY_ACTIVE_STEER_AT_TURN not in state
    assert _ostate.KEY_ACTIVE_STEER_AUTHOR not in state


def test_set_active_steer_author_defaults_to_empty() -> None:
    """Back-compat: callers that omit author still write an empty key."""
    state: dict[str, Any] = {}
    _ostate.set_active_steer(state, body="pivot", at_turn=1)
    assert state[_ostate.KEY_ACTIVE_STEER_AUTHOR] == ""
