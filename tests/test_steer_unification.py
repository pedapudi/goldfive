"""Tests for the goldfive-steer-unification path (goldfive#236 follow-up).

Covers:

* Severity-aware promotion: WARNING / CRITICAL eligible goldfive drifts
  route through :meth:`DefaultSteerer._promote_drift_to_steer` instead
  of the legacy REFINE_PLAN / NUDGE / PAUSE_ESCALATE mappings.
* Source attribution: ``DriftEvent.authored_by`` + the on-the-wire
  ``DriftDetected.authored_by`` field round-trip correctly for both
  user-authored and goldfive-authored drifts.
* Active-steer state stamping: a promoted goldfive steer writes
  ``goldfive.active_steer.{body,at_turn,author,source}`` with
  source=``"goldfive"``.
* User-over-goldfive priority: a fresh user steer within the
  configured freshness window suppresses a subsequent goldfive steer.
  A stale user steer does not.
* Promotion policy knob: ``goldfive_steer_threshold="off"`` /
  ``"warning"`` / ``"critical"`` thread through correctly.
* Cancel-reason tagging: the bound adapter's ``_next_cancel_reason``
  carries a ``"goldfive_<drift_kind>"`` prefix on promotion.
* Body composition: ``drift.detail`` is used verbatim when present;
  a synthesised template is used when empty.
* Dedupe: the drift's ``id`` enters
  ``goldfive.processed_steer_ids`` so a redelivery is a no-op.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import state_store as _ostate  # noqa: E402
from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return

    @property
    def proto_events(self) -> list[Any]:
        """goldfive a4: filter dict-envelope sidecars."""
        return [e for e in self.events if hasattr(e, "WhichOneof")]


class _StubPlanner:
    """Planner stub that records refine + refine_steer calls."""

    def __init__(self, *, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:  # pragma: no cover
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised


class _FakeAdapter:
    """Adapter stub exposing only the cancel-reason slot."""

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="T2", status=TaskStatus.RUNNING),
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t2", to_task_id="t3"),
        ],
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship the thing")],
        plan=_make_plan(),
        current_task_id="t2",
    )


def _bind(
    *, threshold: str = "warning", window: int = 3
) -> tuple[DefaultSteerer, Session, ListSink, _StubPlanner, _FakeAdapter]:
    steerer = DefaultSteerer(
        goldfive_steer_threshold=threshold,
        goldfive_steer_suppression_window_turns=window,
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2b", title="Replanned T2"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2b")],
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )
    planner = _StubPlanner(revised=revised)
    session = _make_session()
    sink = ListSink()
    adapter = _FakeAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)
    return steerer, session, sink, planner, adapter


def _drift_detected_events(sink: ListSink) -> list[Any]:
    return [
        evt
        for evt in sink.proto_events
        if evt.WhichOneof("payload") == "drift_detected"
    ]


def _plan_revised_events(sink: ListSink) -> list[Any]:
    return [
        evt
        for evt in sink.proto_events
        if evt.WhichOneof("payload") == "plan_revised"
    ]


# ---------------------------------------------------------------------------
# Severity-aware promotion
# ---------------------------------------------------------------------------


async def test_goldfive_steer_promotes_at_warning_severity() -> None:
    """WARNING OFF_TOPIC drift -> cancel tag + state stamp + refine."""
    steerer, session, sink, planner, adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered into raccoons",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)

    # refine_steer was called (not the generic refine path).
    assert len(planner.refine_steer_calls) == 1
    assert planner.refine_calls == []
    # DriftDetected + PlanRevised emitted.
    assert len(_drift_detected_events(sink)) >= 1
    assert len(_plan_revised_events(sink)) == 1
    # active_steer state stamped with source="goldfive".
    assert session.state[_ostate.KEY_ACTIVE_STEER_BODY] == "agent wandered into raccoons"
    assert session.state[_ostate.KEY_ACTIVE_STEER_AUTHOR] == "goldfive"
    assert session.state[_ostate.KEY_ACTIVE_STEER_SOURCE] == "goldfive"
    # Adapter cancel reason tagged.
    assert adapter._next_cancel_reason == "goldfive_off_topic"
    # Revised plan installed. goldfive#247: identity check replaced
    # with id check (Plan is frozen).
    assert session.plan is not None and session.plan.id == planner.revised.id


async def test_goldfive_steer_does_not_promote_at_info_severity() -> None:
    """INFO-severity goldfive drift stays on the legacy OBSERVE path."""
    steerer, session, sink, planner, _adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.INFO,
        detail="minor wander",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)

    # INFO drift -> ladder OBSERVE. No refine of either flavour runs.
    assert planner.refine_steer_calls == []
    # The drift is still visible on the wire so operators can see it.
    assert _drift_detected_events(sink)
    # active_steer is NOT stamped.
    assert _ostate.KEY_ACTIVE_STEER_SOURCE not in session.state


# ---------------------------------------------------------------------------
# User > goldfive priority
# ---------------------------------------------------------------------------


async def test_goldfive_steer_suppressed_when_user_steer_active() -> None:
    """Fresh user steer within the freshness window blocks a goldfive steer.

    goldfive a4: each refine now emits two extra dict-envelope sidecars
    (``refine_attempted`` + correlation ``plan_revised``) which advance
    ``session._next_sequence`` alongside the canonical proto events.
    ``_should_promote_to_steer`` reads ``_next_sequence`` as the "current
    turn" surrogate, so the user-steer's promotion path now consumes
    more sequence positions than before. The window is widened from 3
    to 6 to keep the suppression invariant honoured under the new
    emit count. (See the PR's "ordering invariant weakened" note.)
    """
    steerer, session, sink, planner, adapter = _bind(window=6)

    # Apply a user steer first.
    user_msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-user",
        payload={"note": "focus on X", "annotation_id": "ann_user"},
    )
    await steerer.drift.observe(user_msg, session)
    user_at_turn = session.state[_ostate.KEY_ACTIVE_STEER_AT_TURN]
    user_body = session.state[_ostate.KEY_ACTIVE_STEER_BODY]
    user_refine_steer_calls = len(planner.refine_steer_calls)
    adapter._next_cancel_reason = ""  # reset

    # A subsequent goldfive drift fires "shortly after" -- within the
    # window. Must be suppressed.
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent drifting",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)

    # No new refine_steer call (the user steer's refine already ran via
    # planner.refine, but the goldfive promotion must NOT have called
    # refine_steer or mutated the active_steer state).
    assert len(planner.refine_steer_calls) == user_refine_steer_calls
    assert session.state[_ostate.KEY_ACTIVE_STEER_SOURCE] == "user"
    assert session.state[_ostate.KEY_ACTIVE_STEER_BODY] == user_body
    assert session.state[_ostate.KEY_ACTIVE_STEER_AT_TURN] == user_at_turn
    # Adapter cancel reason NOT tagged for goldfive (the user steer's
    # own tag may have happened during observe(); ensure we didn't
    # re-tag to a goldfive reason).
    assert not adapter._next_cancel_reason.startswith("goldfive_")
    # DriftDetected still emitted, with suppressed_by_user_steer=True.
    offtopic_events = [
        e
        for e in _drift_detected_events(sink)
        if e.drift_detected.detail == "agent drifting"
    ]
    assert offtopic_events
    assert offtopic_events[-1].drift_detected.suppressed_by_user_steer is True
    assert offtopic_events[-1].drift_detected.authored_by == "goldfive"


async def test_goldfive_steer_fires_when_user_steer_stale() -> None:
    """User steer outside the freshness window no longer blocks promotion."""
    steerer, session, _sink, planner, _adapter = _bind(window=1)

    # Apply a user steer then synthetically advance the session sequence
    # beyond the window.
    user_msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-stale",
        payload={"note": "initial", "annotation_id": "ann_stale"},
    )
    await steerer.drift.observe(user_msg, session)
    # Advance sequence beyond the window.
    session._next_sequence += 10

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="fresh detector fire",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)

    # refine_steer fired now (stale user steer shouldn't block).
    assert len(planner.refine_steer_calls) == 1
    # active_steer now carries the goldfive body.
    assert session.state[_ostate.KEY_ACTIVE_STEER_SOURCE] == "goldfive"
    assert session.state[_ostate.KEY_ACTIVE_STEER_BODY] == "fresh detector fire"


# ---------------------------------------------------------------------------
# Cancel reason + adapter tagging
# ---------------------------------------------------------------------------


async def test_goldfive_steer_cancel_reason_off_topic() -> None:
    steerer, session, _sink, _planner, adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="wander",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)
    assert adapter._next_cancel_reason == "goldfive_off_topic"


async def test_goldfive_steer_cancel_reason_intent_divergence_critical() -> None:
    steerer, session, _sink, _planner, adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="diverged",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)
    assert adapter._next_cancel_reason == "goldfive_intent_divergence"


# ---------------------------------------------------------------------------
# Steer body derivation
# ---------------------------------------------------------------------------


async def test_goldfive_steer_body_from_drift_detail() -> None:
    """Non-empty drift.detail is used verbatim as the steer body."""
    steerer, session, _sink, _planner, _adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent acknowledged discrepancy but chose to adopt expanded topic",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)
    assert (
        session.state[_ostate.KEY_ACTIVE_STEER_BODY]
        == "agent acknowledged discrepancy but chose to adopt expanded topic"
    )


async def test_goldfive_steer_body_fallback_when_detail_empty() -> None:
    """Empty detail falls through to the synthesised template."""
    steerer, session, _sink, _planner, _adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)
    body = session.state[_ostate.KEY_ACTIVE_STEER_BODY]
    assert "Goldfive detected" in body
    assert "OFF_TOPIC" in body
    assert "WARNING" in body
    assert "t2" in body


# ---------------------------------------------------------------------------
# Threshold knob
# ---------------------------------------------------------------------------


async def test_goldfive_steer_threshold_off_disables_promotion() -> None:
    """threshold='off' keeps every goldfive drift on the legacy ladder."""
    steerer, session, _sink, planner, adapter = _bind(threshold="off")
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        detail="even critical stays legacy",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)

    # No refine_steer call (promotion disabled).
    assert planner.refine_steer_calls == []
    # active_steer NOT stamped from the goldfive side.
    assert _ostate.read(session.state, _ostate.KEY_ACTIVE_STEER_SOURCE, "") == ""
    # Adapter cancel reason NOT tagged with the goldfive prefix.
    assert not adapter._next_cancel_reason.startswith("goldfive_")


async def test_goldfive_steer_threshold_critical_skips_warning() -> None:
    """threshold='critical' promotes only CRITICAL drifts.

    Use distinct tasks for the WARNING and CRITICAL emits so the
    goldfive#215 iter-8 P2 outcome gate (keyed on (kind, task)) does
    not fold the CRITICAL onto the WARNING's already-succeeded
    outcome — the test exercises the promotion-threshold branch,
    not the outcome-replay short-circuit.
    """
    steerer, session, _sink, planner, _adapter = _bind(threshold="critical")

    # WARNING -> legacy path
    warn = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="warning-only",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(warn, session)
    assert planner.refine_steer_calls == []

    # CRITICAL -> promoted (distinct task so the outcome gate doesn't gate it).
    crit = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.CRITICAL,
        detail="critical-promoted",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(crit, session)
    assert len(planner.refine_steer_calls) == 1


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


async def test_processed_steer_ids_records_goldfive_drift_id() -> None:
    """Promoting a goldfive drift records the drift.id for dedupe."""
    steerer, session, _sink, _planner, _adapter = _bind()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="wander",
        current_task_id="t2",
    )
    drift_id = drift.id
    await steerer.drift.handle_drift(drift, session)
    processed = session.state.get(_ostate.KEY_PROCESSED_STEER_IDS, [])
    assert drift_id in processed


# ---------------------------------------------------------------------------
# authored_by attribution
# ---------------------------------------------------------------------------


async def test_user_steer_drift_event_authored_by_user() -> None:
    """ControlMessage-sourced drifts get authored_by="user"."""
    steerer, session, sink, _planner, _adapter = _bind()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-user",
        payload={"note": "focus", "annotation_id": "ann_1"},
    )
    await steerer.drift.observe(msg, session)
    steer_drifts = [
        e
        for e in _drift_detected_events(sink)
        if e.drift_detected.kind  # any kind
    ]
    user_steer = [
        e for e in steer_drifts if e.drift_detected.authored_by == "user"
    ]
    assert user_steer


async def test_goldfive_drift_event_authored_by_goldfive_when_unset() -> None:
    """Drifts minted without an explicit authored_by get normalised to 'goldfive'."""
    steerer, session, sink, _planner, _adapter = _bind(threshold="off")
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="boom",
        current_task_id="t2",
    )
    await steerer.drift.handle_drift(drift, session)
    evts = _drift_detected_events(sink)
    assert evts
    assert all(e.drift_detected.authored_by == "goldfive" for e in evts)


# ---------------------------------------------------------------------------
# Suppression-flag round-trip
# ---------------------------------------------------------------------------


async def test_suppressed_drift_event_wire_flag() -> None:
    """DriftDetected carries suppressed_by_user_steer=True on suppression.

    ``window`` is measured against the session's monotonic event-
    sequence counter, which now includes the paired
    ``SteeringDecisionMade`` observability events introduced by
    zicato-optimization-surface (one per ``DriftDetected``). The
    suppression intent here is "fire-immediately-after-steer is
    suppressed" — a generous window keeps the test honest while the
    counter-vs-logical-turn semantic mismatch is in flight.
    """
    steerer, session, sink, planner, _adapter = _bind(window=50)
    # User steer first.
    await steerer.drift.observe(
        ControlMessage(
            kind=ControlKind.STEER,
            id="ctl-user-sup",
            payload={"note": "stay", "annotation_id": "ann_sup"},
        ),
        session,
    )
    # Goldfive drift immediately after (inside window).
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="goldfive-signal",
        current_task_id="t2",
    )
    refine_steer_before = len(planner.refine_steer_calls)
    await steerer.drift.handle_drift(drift, session)

    assert len(planner.refine_steer_calls) == refine_steer_before
    matching = [
        e
        for e in _drift_detected_events(sink)
        if e.drift_detected.detail == "goldfive-signal"
    ]
    assert matching
    assert matching[-1].drift_detected.suppressed_by_user_steer is True


# ---------------------------------------------------------------------------
# Restart-message framing
# ---------------------------------------------------------------------------


def test_compose_steer_restart_message_goldfive_header() -> None:
    from goldfive.executors.sequential import SequentialExecutor

    composed = SequentialExecutor._compose_steer_restart_message(
        None,
        fallback="drift body",
        source="goldfive",
        superseded_task_ids=["t2"],
        replacement_task_ids=["t2b"],
    )
    assert composed.startswith("[GOLDFIVE STEERING CONTROL")
    assert "drift body" in composed
    assert "t2" in composed
    assert "t2b" in composed


def test_compose_steer_restart_message_user_default() -> None:
    from goldfive.executors.sequential import SequentialExecutor

    composed = SequentialExecutor._compose_steer_restart_message(
        None,
        fallback="user body",
    )
    assert composed.startswith("[USER STEERING CONTROL")


# ---------------------------------------------------------------------------
# RuntimeConfig wiring
# ---------------------------------------------------------------------------


def test_runtime_config_threads_steering_threshold() -> None:
    from goldfive.config import RuntimeConfig, SteeringConfig

    rc = RuntimeConfig(steering=SteeringConfig(threshold="critical", suppression_window_turns=7))
    steerer = DefaultSteerer(steering_config=rc.steering)
    assert steerer._goldfive_steer_threshold == "critical"
    assert steerer._goldfive_steer_suppression_window_turns == 7


def test_steering_config_from_env_accepts_defaults(goldfive_steer_env: Any) -> None:
    from goldfive.config import SteeringConfig

    # Fixture pre-clears the steering env vars in setup.
    _ = goldfive_steer_env
    cfg = SteeringConfig.from_env()
    assert cfg.threshold == "warning"
    assert cfg.suppression_window_turns == 3


def test_steering_config_from_env_reads_threshold(
    goldfive_steer_env: Any,
) -> None:
    from goldfive.config import SteeringConfig

    goldfive_steer_env.set(threshold="critical", suppression_window_turns=9)
    cfg = SteeringConfig.from_env()
    assert cfg.threshold == "critical"
    assert cfg.suppression_window_turns == 9


def test_steering_config_from_env_rejects_unknown_threshold(
    goldfive_steer_env: Any,
) -> None:
    from goldfive.config import SteeringConfig

    goldfive_steer_env.set(threshold="nonsense")
    cfg = SteeringConfig.from_env()
    # Falls back to default
    assert cfg.threshold == "warning"


# ---------------------------------------------------------------------------
# GoldfivePlanner attribution line (when ADK is available)
# ---------------------------------------------------------------------------


def test_goldfive_planner_source_attribution_line() -> None:
    adk_available = True
    try:
        from goldfive.planners.goldfive_planner import GoldfivePlanner  # noqa: F401
    except ImportError:
        adk_available = False
    if not adk_available:
        pytest.skip("google-adk not installed")

    from goldfive.planners.goldfive_planner import GoldfivePlanner

    planner = GoldfivePlanner()

    class _FakeCtx:
        def __init__(self, state: dict[str, Any]) -> None:
            self._state = state

        @property
        def state(self) -> Any:
            return self._state

    class _FakeReq:
        pass

    # user-authored
    ctx = _FakeCtx(
        {
            _ostate.KEY_CURRENT_TASK_ID: "t2",
            _ostate.KEY_CURRENT_TASK_TITLE: "T2",
            _ostate.KEY_ACTIVE_STEER_BODY: "focus",
            _ostate.KEY_ACTIVE_STEER_SOURCE: "user",
        }
    )
    out = planner.build_planning_instruction(ctx, _FakeReq())  # type: ignore[arg-type]
    assert out is not None
    assert "Active steer (user): focus" in out

    # goldfive-authored
    ctx2 = _FakeCtx(
        {
            _ostate.KEY_CURRENT_TASK_ID: "t2",
            _ostate.KEY_CURRENT_TASK_TITLE: "T2",
            _ostate.KEY_ACTIVE_STEER_BODY: "drift corrected",
            _ostate.KEY_ACTIVE_STEER_SOURCE: "goldfive",
        }
    )
    out2 = planner.build_planning_instruction(ctx2, _FakeReq())  # type: ignore[arg-type]
    assert out2 is not None
    assert "Active steer (goldfive): drift corrected" in out2
