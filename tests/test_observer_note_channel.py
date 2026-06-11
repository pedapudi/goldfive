"""Observer-note channel delivery tests (AGENCY-PRESERVATION.md PR 6).

Covers the §5.2 *binding* acceptance criteria for the observer-note channel:

* **Exactly-once delivery** — a note enqueued mid-invocation (delivered via
  the ``before_model`` surface) and ALSO present at the next boundary renders
  ONCE, not twice (the classic two-mode double-delivery bug).
* **Marker strip-and-refresh idempotency** — two consecutive ``before_model``
  calls never stack observer-note blocks on the system_instruction.

Plus the channel-routing no-op-by-default guarantee (§5.1: legacy default →
queue never populated, ``pending_nudges`` used as before) and the
``SignalDelivered``-at-dispatch-point wiring (both channels emit at the
dispatch decision; the request_context surfaces render exactly-once but emit
no second event).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.events import SIGNAL_CHANNEL_REQUEST_CONTEXT  # noqa: E402
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.observer_note_queue import (  # noqa: E402
    OBSERVER_NOTE_MARKER_PREFIX,
    ObserverNoteQueue,
)
from goldfive.prompt_shaper import PromptShaper  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _Config:
    def __init__(self, system_instruction: Any = None) -> None:
        self.system_instruction = system_instruction


class _Req:
    """``llm_request`` stub whose ``append_instructions`` PERSISTS into
    ``config.system_instruction`` — mimicking the ADK case where the appended
    block carries across calls, so the strip-and-refresh idempotency is
    meaningfully exercised."""

    def __init__(self, system_instruction: Any = None) -> None:
        self.config = _Config(system_instruction)

    def append_instructions(self, instructions: list[str]) -> None:
        joined = "\n\n".join(instructions)
        existing = self.config.system_instruction
        if not existing:
            self.config.system_instruction = joined
        else:
            self.config.system_instruction = f"{existing}\n\n{joined}"


class _Ctx:
    def __init__(self, steerer: Any, session: Session) -> None:
        self.steerer = steerer
        self.session = session


def _make_steerer(
    *,
    channel: str = "request_context",
    observation_only: bool = False,
    signal_telemetry: bool = False,
    sink: _ListSink | None = None,
) -> DefaultSteerer:
    cfg = SteeringConfig(
        signal_channel=channel,
        observation_only=observation_only,
        signal_telemetry=signal_telemetry,
    )
    steerer = DefaultSteerer(steering_config=cfg)
    if sink is not None:
        steerer.bind(sinks=[sink], planner=object())
    return steerer


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="publish a memo on solar")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="Research solar", description="specs")],
            edges=[],
        ),
        current_task_id="t1",
    )


def _enqueue(
    session: Session,
    *,
    drift_id: str = "d1",
    severity: str = "warning",
    kind: str = "looping_tool_call",
    turn: int = 0,
) -> None:
    ObserverNoteQueue.for_session(session).enqueue(
        body=(
            "Observation: `search_web` was invoked 5 times\n"
            "The user's goal: publish a memo on solar\n"
            "This note is advisory."
        ),
        observation="`search_web` was invoked 5 times",
        severity=severity,
        drift_id=drift_id,
        kind=kind,
        task_id="t1",
        agent_id="agent",
        turn=turn,
    )


def _drift(
    *,
    kind: DriftKind = DriftKind.LOOPING_TOOL_CALL,
    severity: DriftSeverity = DriftSeverity.WARNING,
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail="looped",
        current_task_id="t1",
        current_agent_id="agent",
        authored_by="goldfive",
    )


def _delivered(sink: _ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "signal_delivered"
    ]


# ---------------------------------------------------------------------------
# Channel routing (no-op by default, §5.1)
# ---------------------------------------------------------------------------


async def test_legacy_channel_routes_to_pending_nudges() -> None:
    steerer = _make_steerer(channel="legacy_user_message")
    session = _make_session()
    await steerer.drift._dispatch_nudge(_drift(), session)
    # Legacy: the note queues on pending_nudges; the observer queue stays empty.
    assert len(session.pending_nudges) == 1
    assert ObserverNoteQueue.for_session(session).pending() == []


async def test_request_context_channel_routes_to_queue() -> None:
    steerer = _make_steerer(channel="request_context")
    session = _make_session()
    await steerer.drift._dispatch_nudge(_drift(), session)
    # request_context: the note goes to the observer queue, NOT pending_nudges.
    assert session.pending_nudges == []
    pend = ObserverNoteQueue.for_session(session).pending()
    assert len(pend) == 1
    assert pend[0].kind == "looping_tool_call"


def test_before_model_noop_when_queue_empty() -> None:
    """With no notes pending (the legacy default state) the shaper is a no-op."""
    session = _make_session()
    ctx = _Ctx(_make_steerer(channel="request_context"), session)
    req = _Req(system_instruction="You are a helpful agent.")
    note = PromptShaper().inject_observer_note(
        llm_request=req, session=session, session_context=ctx
    )
    assert note is None
    assert req.config.system_instruction == "You are a helpful agent."


# ---------------------------------------------------------------------------
# §5.2 — exactly-once delivery across the two-mode (before_model + boundary)
# ---------------------------------------------------------------------------


async def test_exactly_once_before_model_consumes_then_boundary_skips() -> None:
    steerer = _make_steerer(channel="request_context")
    session = _make_session()
    ctx = _Ctx(steerer, session)
    _enqueue(session, drift_id="d1")

    # Surface 1 (before_model) delivers the note exactly once.
    req = _Req(system_instruction="base")
    note = PromptShaper().inject_observer_note(
        llm_request=req, session=session, session_context=ctx
    )
    assert note is not None and note.note_id == "d1"
    assert req.config.system_instruction.count(OBSERVER_NOTE_MARKER_PREFIX) == 1

    # Surface 2 (invocation-boundary replay): the SAME note is no longer
    # pending, so the boundary renders NOTHING — never a second delivery.
    executor = SequentialExecutor()
    replay = await executor._consume_observer_note_for_replay(session)
    assert replay is None
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True


async def test_exactly_once_boundary_delivers_when_before_model_did_not() -> None:
    """The reverse path: if no model call fired, the boundary renders once."""
    session = _make_session()
    _enqueue(session, drift_id="d1")

    executor = SequentialExecutor()
    replay = await executor._consume_observer_note_for_replay(session)
    assert replay is not None
    assert replay.count(OBSERVER_NOTE_MARKER_PREFIX) == 1
    # And a second boundary pass finds nothing — exactly once.
    again = await executor._consume_observer_note_for_replay(session)
    assert again is None


# ---------------------------------------------------------------------------
# §5.2 — marker strip-and-refresh idempotency
# ---------------------------------------------------------------------------


def test_two_before_model_calls_never_stack_blocks() -> None:
    steerer = _make_steerer(channel="request_context")
    session = _make_session()
    ctx = _Ctx(steerer, session)
    _enqueue(session, drift_id="d1")

    req = _Req(system_instruction="base instruction")
    shaper = PromptShaper()

    shaper.inject_observer_note(llm_request=req, session=session, session_context=ctx)
    assert req.config.system_instruction.count(OBSERVER_NOTE_MARKER_PREFIX) == 1

    # Second consecutive call: the prior block is stripped and the (now
    # delivered) note is not re-injected — never two blocks.
    shaper.inject_observer_note(llm_request=req, session=session, session_context=ctx)
    assert req.config.system_instruction.count(OBSERVER_NOTE_MARKER_PREFIX) == 0
    assert "base instruction" in req.config.system_instruction


def test_two_pending_notes_drain_one_per_call_never_stack() -> None:
    steerer = _make_steerer(channel="request_context")
    session = _make_session()
    ctx = _Ctx(steerer, session)
    _enqueue(session, drift_id="d_warn", severity="warning")
    _enqueue(session, drift_id="d_crit", severity="critical")

    req = _Req(system_instruction="base")
    shaper = PromptShaper()

    # Call 1: most-severe (critical) delivered; exactly one block.
    n1 = shaper.inject_observer_note(
        llm_request=req, session=session, session_context=ctx
    )
    assert n1.note_id == "d_crit"
    assert req.config.system_instruction.count(OBSERVER_NOTE_MARKER_PREFIX) == 1

    # Call 2: prior block stripped, the warning note delivered — still ONE.
    n2 = shaper.inject_observer_note(
        llm_request=req, session=session, session_context=ctx
    )
    assert n2.note_id == "d_warn"
    assert req.config.system_instruction.count(OBSERVER_NOTE_MARKER_PREFIX) == 1


# ---------------------------------------------------------------------------
# observation_only — consumed as dry-run, never injected
# ---------------------------------------------------------------------------


def test_observation_only_does_not_inject_but_consumes() -> None:
    steerer = _make_steerer(channel="request_context", observation_only=True)
    session = _make_session()
    ctx = _Ctx(steerer, session)
    _enqueue(session, drift_id="d1")

    req = _Req(system_instruction="base")
    note = PromptShaper().inject_observer_note(
        llm_request=req, session=session, session_context=ctx
    )
    # The note is consumed (marked delivered) but the block is NOT injected —
    # the strict-passive operator sees the raw prompt.
    assert note is not None
    assert req.config.system_instruction == "base"
    assert ObserverNoteQueue.for_session(session).get("d1").delivered is True


# ---------------------------------------------------------------------------
# SignalDelivered at the dispatch decision point (channel=request_context)
#
# Both channels emit at the dispatch point (the PR-5 model the §5.4 shadow
# diff is built on); request_context differs only in the channel value and
# where the note is queued. The delivery surfaces render exactly-once but do
# NOT emit a second event.
# ---------------------------------------------------------------------------


async def test_signal_delivered_emitted_at_dispatch_for_request_context() -> None:
    sink = _ListSink()
    steerer = _make_steerer(
        channel="request_context", signal_telemetry=True, sink=sink
    )
    session = _make_session()

    # Dispatching the nudge enqueues the note AND emits one SignalDelivered on
    # the request_context channel — the dispatch decision point.
    await steerer.drift._dispatch_nudge(_drift(), session)

    delivered = _delivered(sink)
    assert len(delivered) == 1
    payload = delivered[0].signal_delivered
    assert payload.channel == SIGNAL_CHANNEL_REQUEST_CONTEXT
    assert payload.dry_run is False  # observation_only is False here

    # Rendering the note at a surface does NOT emit a second event — the
    # surfaces are the exactly-once rendering leg only.
    note = ObserverNoteQueue.for_session(session).peek_for_render()
    assert note is not None
    ctx = _Ctx(steerer, session)
    PromptShaper().inject_observer_note(
        llm_request=_Req(), session=session, session_context=ctx
    )
    assert len(_delivered(sink)) == 1


async def test_signal_delivered_dry_run_under_observation_only() -> None:
    sink = _ListSink()
    steerer = _make_steerer(
        channel="request_context",
        observation_only=True,
        signal_telemetry=True,
        sink=sink,
    )
    session = _make_session()
    await steerer.drift._dispatch_nudge(_drift(), session)
    delivered = _delivered(sink)
    assert len(delivered) == 1
    assert delivered[0].signal_delivered.dry_run is True
