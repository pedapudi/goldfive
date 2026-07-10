"""SignalDelivered / SignalOutcome emission wiring (AGENCY-PRESERVATION.md PR 5).

These tests assert the observe-only telemetry is *wired into the real dispatch
path* — no dead middleware (§5.6) — and behaves correctly in both
``observation_only`` modes (dry-run vs. real delivery, §5.4). They also pin the
binding requirements: the events are consumable by the JSONL sink with no sink
changes, and with the ``signal_telemetry`` flag OFF (the default) nothing is
emitted (§5.1 no-op-by-default — the zero-behavior-change guarantee the full
suite already proves by passing unmodified).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.events import (  # noqa: E402
    SIGNAL_CHANNEL_NUDGE_REPLAY,
    SIGNAL_CHANNEL_PAUSE_CONTROL,
    SIGNAL_CHANNEL_PROMOTION,
    SIGNAL_CHANNEL_STEER_CONTROL,
    SIGNAL_OUTCOME_ESCALATED,
    SIGNAL_OUTCOME_INVOCATION_ENDED,
    SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL,
    SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
    SIGNAL_OUTCOME_USER_INTERVENED,
    signal_delivered_event,
    signal_outcome_event,
)
from goldfive.pb.goldfive.v1 import events_pb2 as pb  # noqa: E402
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
# Stubs / helpers
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _RecordingPlanner:
    async def generate(self, **_: Any) -> Any:
        return None

    async def refine(self, **kwargs: Any) -> Any:
        return _revised(kwargs.get("plan"))

    async def refine_steer(self, **kwargs: Any) -> Any:
        return _revised(kwargs.get("plan"))


def _revised(prior: Plan | None) -> Plan:
    tasks = list(prior.tasks) if prior is not None else []
    tasks.append(Task(id="t2-corrective", title="Corrective follow-up"))
    return Plan(
        id="p2",
        run_id=(prior.run_id if prior is not None else "r1"),
        goal_ids=(list(prior.goal_ids) if prior is not None else ["g1"]),
        tasks=tasks,
        edges=[],
    )


def _make_session(run_id: str = "r1", task_id: str = "t1") -> Session:
    plan = Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title="Research solar", description="Find specs")],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


def _setup(
    *,
    observation_only: bool = False,
    signal_telemetry: bool = True,
    planner: Any | None = None,
) -> tuple[DefaultSteerer, Session, _ListSink]:
    cfg = SteeringConfig(observation_only=observation_only, signal_telemetry=signal_telemetry)
    steerer = DefaultSteerer(steering_config=cfg)
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=planner or _RecordingPlanner())
    return steerer, _make_session(), sink


def _drift(
    *,
    kind: DriftKind = DriftKind.LOOPING_TOOL_CALL,
    severity: DriftSeverity = DriftSeverity.WARNING,
    task_id: str = "t1",
    authored_by: str = "goldfive",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail="looped",
        current_task_id=task_id,
        current_agent_id="agent",
        authored_by=authored_by,
    )


def _delivered(sink: _ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "signal_delivered"
    ]


def _outcomes(sink: _ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "signal_outcome"
    ]


# ---------------------------------------------------------------------------
# Event factories — proto envelope shape + oneof membership
# ---------------------------------------------------------------------------


def test_signal_delivered_factory_fields() -> None:
    evt = signal_delivered_event(
        "run-1",
        7,
        drift_id="d1",
        kind="looping_tool_call",
        severity="warning",
        channel=SIGNAL_CHANNEL_NUDGE_REPLAY,
        turn=3,
        note_text="search_web called 5x",
        dry_run=True,
        task_id="t1",
        agent_id="agent",
        decision={"ladder_level": "nudge", "occurrence_count": 0},
    )
    assert evt.WhichOneof("payload") == "signal_delivered"
    p = evt.signal_delivered
    assert p.drift_id == "d1"
    assert p.kind == "looping_tool_call"
    assert p.channel == SIGNAL_CHANNEL_NUDGE_REPLAY
    assert p.turn == 3
    assert p.dry_run is True
    assert p.task_id == "t1"
    assert json.loads(p.decision_json) == {"ladder_level": "nudge", "occurrence_count": 0}


def test_signal_outcome_factory_fields() -> None:
    evt = signal_outcome_event(
        "run-1",
        8,
        drift_kind="LOOPING_TOOL_CALL",
        task_id="t1",
        outcome=SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
        turns_to_resolution=4,
        delivery_count=1,
        had_real_delivery=False,
    )
    assert evt.WhichOneof("payload") == "signal_outcome"
    p = evt.signal_outcome
    assert p.drift_kind == "LOOPING_TOOL_CALL"
    assert p.outcome == SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED
    assert p.turns_to_resolution == 4
    assert p.delivery_count == 1
    assert p.had_real_delivery is False


def test_new_payloads_in_event_oneof() -> None:
    e1 = pb.Event()
    e1.signal_delivered.channel = "nudge_replay"
    assert e1.WhichOneof("payload") == "signal_delivered"
    e2 = pb.Event()
    e2.signal_outcome.outcome = "escalated"
    assert e2.WhichOneof("payload") == "signal_outcome"


# ---------------------------------------------------------------------------
# Emission wiring — the four dispatch decision points
# ---------------------------------------------------------------------------


async def test_nudge_dispatch_emits_signal_delivered() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    rows = _delivered(sink)
    assert len(rows) == 1
    p = rows[0].signal_delivered
    assert p.channel == SIGNAL_CHANNEL_NUDGE_REPLAY
    assert p.kind == DriftKind.LOOPING_TOOL_CALL.value
    assert p.dry_run is False  # active steering
    assert p.note_text  # an observer note was composed
    assert json.loads(p.decision_json)["channel_action"] == "queued"


async def test_steer_control_dispatch_emits_signal_delivered_with_swap_targets() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_goldfive_steer_control(_drift(), session)
    rows = _delivered(sink)
    assert len(rows) == 1
    p = rows[0].signal_delivered
    assert p.channel == SIGNAL_CHANNEL_STEER_CONTROL
    decision = json.loads(p.decision_json)
    # The plan-swap targets the legacy regime would have steered toward.
    assert decision["superseded_task_ids"] == ["t1"]
    assert "replacement_task_ids" in decision


async def test_promotion_dispatch_uses_promotion_channel() -> None:
    steerer, session, sink = _setup(observation_only=False)
    # body_override marks the promote-to-steer path.
    await steerer.drift._dispatch_goldfive_steer_control(
        _drift(kind=DriftKind.OFF_TOPIC), session, body_override="promotion body"
    )
    rows = _delivered(sink)
    assert len(rows) == 1
    p = rows[0].signal_delivered
    assert p.channel == SIGNAL_CHANNEL_PROMOTION
    assert json.loads(p.decision_json)["promotion"] is True


async def test_pause_control_dispatch_emits_delivery_and_escalated_outcome() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_goldfive_pause_control(
        _drift(), session, reason="handler exhausted"
    )
    delivered = _delivered(sink)
    assert len(delivered) == 1
    assert delivered[0].signal_delivered.channel == SIGNAL_CHANNEL_PAUSE_CONTROL
    # Pause resolves the key terminally as escalated.
    outcomes = _outcomes(sink)
    assert [o.signal_outcome.outcome for o in outcomes] == [SIGNAL_OUTCOME_ESCALATED]


async def test_handle_drift_promotion_path_emits_signal_end_to_end() -> None:
    """Full chain: handle_drift -> promote_drift_to_steer -> note enqueue -> emit.

    AGENCY-PRESERVATION.md PR 7: the default-regime promotion enqueues an
    advisory note instead of dispatching GOLDFIVE_STEER, so the emitted
    ``SignalDelivered`` rides the note channel (``nudge_replay`` under the
    default ``signal_channel``) and carries ``ladder_level="promotion"`` in its
    decision payload rather than ``channel == SIGNAL_CHANNEL_PROMOTION`` (the
    legacy-only dispatch channel). The end-to-end emission is what this pins.
    """
    steerer, session, sink = _setup(observation_only=False)
    # OFF_TOPIC + WARNING is on the promote-to-steer path.
    await steerer.drift.handle_drift(_drift(kind=DriftKind.OFF_TOPIC), session)
    await steerer.drift._wait_background_drifts_idle()
    rows = _delivered(sink)
    assert any(
        json.loads(r.signal_delivered.decision_json).get("ladder_level") == "promotion"
        for r in rows
    ), "promotion path must emit a SignalDelivered with ladder_level=promotion"


# ---------------------------------------------------------------------------
# Dry-run flag — both observation_only modes (§5.4)
# ---------------------------------------------------------------------------


async def test_dry_run_true_under_observation_only() -> None:
    steerer, session, sink = _setup(observation_only=True)
    await steerer.drift._dispatch_goldfive_steer_control(_drift(), session)
    rows = _delivered(sink)
    assert len(rows) == 1
    p = rows[0].signal_delivered
    assert p.dry_run is True
    decision = json.loads(p.decision_json)
    assert decision["observation_only"] is True
    assert decision["channel_action"] == "suppressed"


async def test_dry_run_false_under_active_steering() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_goldfive_steer_control(_drift(), session)
    p = _delivered(sink)[0].signal_delivered
    assert p.dry_run is False
    assert json.loads(p.decision_json)["observation_only"] is False


async def test_flag_off_emits_nothing() -> None:
    """signal_telemetry=False (the default) → no signal events at all."""
    steerer, session, sink = _setup(observation_only=False, signal_telemetry=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    await steerer.drift._dispatch_goldfive_steer_control(_drift(), session)
    await steerer.drift._dispatch_goldfive_pause_control(_drift(), session, reason="x")
    assert _delivered(sink) == []
    assert _outcomes(sink) == []


# ---------------------------------------------------------------------------
# Outcome detection — real entry points
# ---------------------------------------------------------------------------


async def test_task_terminal_resolves_self_corrected_unaided_under_observation_only() -> None:
    steerer, session, sink = _setup(observation_only=True)
    # Deliver a (dry-run) nudge for the LOOPING_TOOL_CALL/t1 key.
    await steerer.drift._dispatch_nudge(_drift(), session)
    # Now the task completes — the real task-transition chokepoint resolves it.
    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")
    outcomes = _outcomes(sink)
    assert [o.signal_outcome.outcome for o in outcomes] == [
        SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED
    ]
    assert outcomes[0].signal_outcome.had_real_delivery is False


async def test_task_terminal_resolves_after_signal_under_active_steering() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)  # real (dry_run=False)
    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")
    outcomes = _outcomes(sink)
    assert [o.signal_outcome.outcome for o in outcomes] == [
        SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL
    ]
    assert outcomes[0].signal_outcome.had_real_delivery is True


async def test_emit_drift_detected_records_fire_on_ledger() -> None:
    """The DriftDetected chokepoint feeds the ledger (re-fire tracking)."""
    from goldfive.signal_ledger import SignalLedger

    steerer, session, sink = _setup(observation_only=False)
    d1 = _drift()
    d2 = _drift()  # distinct drift id, same (kind, task)
    await steerer.drift._emit_drift_detected(session, d1)
    await steerer.drift._emit_drift_detected(session, d2)
    entry = SignalLedger.for_session(session).entry(
        DriftKind.LOOPING_TOOL_CALL.value, "t1"
    )
    assert entry is not None
    assert entry.fire_count == 2
    assert not entry.has_delivery  # OBSERVE-level: a fire, no signal delivered


async def test_user_steer_resolves_user_intervened() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    # A USER_STEER drift flowing through the emit chokepoint resolves open
    # delivered keys as user_intervened.
    user_drift = _drift(kind=DriftKind.USER_STEER, authored_by="user")
    await steerer.drift._emit_drift_detected(session, user_drift)
    outcomes = _outcomes(sink)
    assert [o.signal_outcome.outcome for o in outcomes] == [SIGNAL_OUTCOME_USER_INTERVENED]


async def test_user_pause_does_not_black_hole_open_keys() -> None:
    # USER_PAUSE is NON-terminal (the run resumes after a later RESUME) and
    # carries authored_by="user"; it must NOT fall into the terminal
    # user-intervened branch and resolve every open key, or all post-resume
    # outcome telemetry is lost. The key stays OPEN — proven by a later
    # finalize resolving it invocation_ended (not user_intervened at pause).
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    pause_drift = _drift(kind=DriftKind.USER_PAUSE, authored_by="user")
    await steerer.drift._emit_drift_detected(session, pause_drift)
    # The pause emitted no user_intervened outcome — the key was left open.
    assert all(
        o.signal_outcome.outcome != SIGNAL_OUTCOME_USER_INTERVENED for o in _outcomes(sink)
    )
    # Still open: finalize now resolves it invocation_ended.
    await steerer.drift.finalize_signal_ledger(session)
    assert [o.signal_outcome.outcome for o in _outcomes(sink)] == [
        SIGNAL_OUTCOME_INVOCATION_ENDED
    ]


async def test_run_end_finalize_resolves_invocation_ended() -> None:
    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    await steerer.drift.finalize_signal_ledger(session)
    outcomes = _outcomes(sink)
    assert [o.signal_outcome.outcome for o in outcomes] == [SIGNAL_OUTCOME_INVOCATION_ENDED]
    # Idempotent — a second finalize finds nothing open.
    await steerer.drift.finalize_signal_ledger(session)
    assert len(_outcomes(sink)) == 1


async def test_executor_drain_helper_finalizes_ledger() -> None:
    """The run-boundary drain helper is wired to finalize the ledger."""
    from goldfive.executors.sequential import _drain_steerer_at_run_boundary

    steerer, session, sink = _setup(observation_only=False)
    await steerer.drift._dispatch_nudge(_drift(), session)
    await _drain_steerer_at_run_boundary(steerer, session)
    assert [o.signal_outcome.outcome for o in _outcomes(sink)] == [
        SIGNAL_OUTCOME_INVOCATION_ENDED
    ]


async def test_run_boundary_drains_background_before_finalizing_ledger() -> None:
    # ORDER MATTERS: a late-resolving background drift records onto the ledger
    # as it drains; if finalize ran first it would resolve every open key and
    # the drained drift would write into an already-finalized ledger (a lost or
    # double-counted outcome). The helper must drain THEN finalize.
    from goldfive.executors.sequential import _drain_steerer_at_run_boundary

    steerer, session, _sink = _setup(observation_only=False)
    order: list[str] = []
    drift_obs = steerer.drift
    real_drain = drift_obs.drain_session_background_tasks
    real_finalize = drift_obs.finalize_signal_ledger

    async def _spy_drain(*a: object, **k: object) -> object:
        order.append("drain")
        return await real_drain(*a, **k)

    async def _spy_finalize(*a: object, **k: object) -> object:
        order.append("finalize")
        return await real_finalize(*a, **k)

    drift_obs.drain_session_background_tasks = _spy_drain  # type: ignore[method-assign]
    drift_obs.finalize_signal_ledger = _spy_finalize  # type: ignore[method-assign]
    await _drain_steerer_at_run_boundary(steerer, session)
    assert order == ["drain", "finalize"]


# ---------------------------------------------------------------------------
# Sink consumability (§5.5) — JSONL round-trip with NO sink changes
# ---------------------------------------------------------------------------


async def test_signal_events_roundtrip_through_jsonl_sink(tmp_path: Any) -> None:
    from goldfive.sinks import JSONLPersistenceSink, replay_from_jsonl

    path = tmp_path / "events.jsonl"
    sink = JSONLPersistenceSink(str(path), mode="write")
    await sink.emit(
        signal_delivered_event(
            "r1",
            1,
            drift_id="d1",
            kind="LOOPING_TOOL_CALL",
            severity="WARNING",
            channel=SIGNAL_CHANNEL_NUDGE_REPLAY,
            turn=2,
            note_text="note",
            dry_run=True,
            task_id="t1",
            decision={"ladder_level": "nudge"},
        )
    )
    await sink.emit(
        signal_outcome_event(
            "r1",
            2,
            drift_kind="LOOPING_TOOL_CALL",
            task_id="t1",
            outcome=SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
            turns_to_resolution=3,
            delivery_count=1,
        )
    )
    await sink.close()

    events = replay_from_jsonl(str(path))
    kinds = [e.WhichOneof("payload") for e in events]
    assert kinds == ["signal_delivered", "signal_outcome"]
    assert events[0].signal_delivered.channel == SIGNAL_CHANNEL_NUDGE_REPLAY
    assert json.loads(events[0].signal_delivered.decision_json) == {"ladder_level": "nudge"}
    assert events[1].signal_outcome.outcome == SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED


async def test_signal_events_collectable_by_inmemory_sink() -> None:
    from goldfive.sinks.memory import InMemorySink

    sink = InMemorySink()
    await sink.emit(
        signal_delivered_event(
            "r1", 1, drift_id="d", kind="K", channel=SIGNAL_CHANNEL_PROMOTION
        )
    )
    assert sink.events[0].WhichOneof("payload") == "signal_delivered"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
