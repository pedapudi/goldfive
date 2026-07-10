"""Drift-condition resolution wiring — the event stream can represent recovery.

``state_store.resolve_drift`` and the ``DRIFT_LIFECYCLE_RESOLVED`` wire
enum shipped with goldfive#271 PR1 but had no production caller:
conditions only ever OPENED or ESCALATED, so ``KEY_ACTIVE_DRIFTS`` grew
monotonically per run and downstream consumers never saw an
intervention succeed. This file pins the two resolution paths:

* **Task-terminal** — every transition to a terminal status
  (COMPLETED / FAILED / CANCELLED / NOT_NEEDED) funnels through
  ``TaskStateMachine._emit_task_transitioned``, which batch-resolves
  every open condition pinned to that task and emits one
  ``DriftDetected(lifecycle=RESOLVED, severity=INFO)`` per condition.
* **On-task verdict** — a reasoning-judge ON-TASK verdict resolves the
  open conditions the reasoning pipeline itself can open (and only
  those) for the verdict's (task, agent, run), gated on the same
  late-verdict staleness check as the drift branch (goldfive#319).

Both paths are telemetry/lifecycle truth only: no intervention decision
reads the result, so behaviour is identical under
``observation_only`` True and False (asserted below).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

import json  # noqa: E402

from goldfive import state_store as _ostate  # noqa: E402
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.state_store import StateStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
)


@pytest.fixture(autouse=True)
def _reset_process_wide_reasoning_config() -> Any:
    """The judge-mode steerer installs ReasoningDriftConfig process-wide;
    clear it around each test to avoid leakage."""
    from goldfive.drift import reasoning as _reasoning

    _reasoning.configure(None)
    yield
    _reasoning.configure(None)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="research", assignee_agent_id="worker"),
            Task(id="t2", title="write", assignee_agent_id="writer"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="publish a memo")],
        plan=_plan(),
        current_task_id="t1",
    )


def _bound_steerer(sink: ListSink, *, observation_only: bool) -> DefaultSteerer:
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(observation_only=observation_only),
    )
    steerer.bind(sinks=[sink], planner=NullPlanner())
    return steerer


def _open(
    session: Session,
    *,
    kind: DriftKind,
    task_id: str,
    agent_id: str,
    severity: DriftSeverity = DriftSeverity.WARNING,
) -> str:
    """Open a condition on the session and return its condition_id."""
    drift = _ostate.open_or_escalate_drift(
        session.state,
        kind=kind,
        task_id=task_id,
        agent_id=agent_id,
        turn_id=session.run_id,
        severity=severity,
    )
    return drift.condition_id


def _resolved_events(sink: ListSink) -> list[Any]:
    """Return DriftDetected envelopes carrying DRIFT_LIFECYCLE_RESOLVED."""
    from goldfive.pb.goldfive.v1 import types_pb2

    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") != "drift_detected":
                continue
        except Exception:
            continue
        if evt.drift_detected.lifecycle == types_pb2.DRIFT_LIFECYCLE_RESOLVED:
            out.append(evt)
    return out


def _active_condition_ids(session: Session) -> set[str]:
    return {d.condition_id for d in _ostate.list_active_drifts(session.state)}


# ---------------------------------------------------------------------------
# state_store.resolve_drifts_matching — batch primitive
# ---------------------------------------------------------------------------


def test_resolve_drifts_matching_filters_and_batches() -> None:
    """One read/write pass resolves every match; non-matches survive."""
    session = _session()
    cid_t1_a = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="worker")
    cid_t1_b = _open(session, kind=DriftKind.LOOPING_TOOL_CALL, task_id="t1", agent_id="worker")
    cid_t2 = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t2", agent_id="writer")

    resolved = _ostate.resolve_drifts_matching(session.state, task_id="t1")

    assert {d.condition_id for d in resolved} == {cid_t1_a, cid_t1_b}
    assert all(d.lifecycle == _ostate.LIFECYCLE_RESOLVED for d in resolved)
    assert all(d.prev_severity is None for d in resolved)
    assert _active_condition_ids(session) == {cid_t2}
    # Idempotent: a second pass matches nothing.
    assert _ostate.resolve_drifts_matching(session.state, task_id="t1") == []


def test_resolve_drifts_matching_kind_and_agent_filters_conjoin() -> None:
    """kinds / agent_ids / turn_id filters are conjunctive membership tests."""
    session = _session()
    cid_match = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="worker")
    cid_unattributed = _open(session, kind=DriftKind.LOOPING_REASONING, task_id="t1", agent_id="")
    cid_wrong_kind = _open(
        session, kind=DriftKind.LOOPING_TOOL_CALL, task_id="t1", agent_id="worker"
    )
    cid_wrong_agent = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="other")

    resolved = _ostate.resolve_drifts_matching(
        session.state,
        task_id="t1",
        agent_ids={"worker", ""},
        turn_id="r1",
        kinds={DriftKind.OFF_TOPIC, DriftKind.LOOPING_REASONING},
    )

    assert {d.condition_id for d in resolved} == {cid_match, cid_unattributed}
    assert _active_condition_ids(session) == {cid_wrong_kind, cid_wrong_agent}


def test_state_store_veneer_resolves_matching() -> None:
    """StateStore exposes the batch helper with the same semantics."""
    session = _session()
    cid = _open(session, kind=DriftKind.GOAL_DRIFT, task_id="t1", agent_id="a")
    store = StateStore.for_session(session)
    resolved = store.resolve_drifts_matching(task_id="t1")
    assert [d.condition_id for d in resolved] == [cid]
    assert store.active_drifts() == []


# ---------------------------------------------------------------------------
# Task-terminal resolution — identical in both steering modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observation_only", [True, False])
async def test_terminal_transition_resolves_task_conditions(
    observation_only: bool,
) -> None:
    """COMPLETED resolves the task's open conditions, emits RESOLVED
    markers, and leaves other tasks' conditions open."""
    from goldfive.pb.goldfive.v1 import types_pb2

    sink = ListSink()
    steerer = _bound_steerer(sink, observation_only=observation_only)
    session = _session()

    cid_loop = _open(
        session,
        kind=DriftKind.LOOPING_TOOL_CALL,
        task_id="t1",
        agent_id="worker",
        severity=DriftSeverity.WARNING,
    )
    cid_goal = _open(
        session,
        kind=DriftKind.GOAL_DRIFT,
        task_id="t1",
        agent_id="worker",
        severity=DriftSeverity.CRITICAL,
    )
    cid_other_task = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t2", agent_id="writer")

    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")

    # KEY_ACTIVE_DRIFTS shrank to the untouched task's condition.
    assert _active_condition_ids(session) == {cid_other_task}

    resolved = _resolved_events(sink)
    assert {e.drift_detected.condition_id for e in resolved} == {cid_loop, cid_goal}
    for evt in resolved:
        payload = evt.drift_detected
        assert payload.severity == types_pb2.DRIFT_SEVERITY_INFO
        assert payload.current_task_id == "t1"
        assert payload.authored_by == "goldfive"
        assert payload.id != ""
        assert payload.id != payload.condition_id
        assert "terminal status COMPLETED" in payload.detail
    # prev_severity carries the condition's last recorded severity.
    by_cid = {e.drift_detected.condition_id: e.drift_detected for e in resolved}
    assert by_cid[cid_loop].prev_severity == types_pb2.DRIFT_SEVERITY_WARNING
    assert by_cid[cid_goal].prev_severity == types_pb2.DRIFT_SEVERITY_CRITICAL


async def test_cancel_cascade_resolves_downstream_conditions() -> None:
    """The cancellation cascade's downstream transitions also resolve."""
    sink = ListSink()
    steerer = _bound_steerer(sink, observation_only=True)
    session = _session()

    cid_t1 = _open(session, kind=DriftKind.LOOPING_TOOL_CALL, task_id="t1", agent_id="worker")
    cid_t2 = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t2", agent_id="writer")

    await steerer.tasks.mark_task_cancelled("t1", session=session, reason="stop")

    assert _active_condition_ids(session) == set()
    resolved = _resolved_events(sink)
    details = {e.drift_detected.condition_id: e.drift_detected.detail for e in resolved}
    assert set(details) == {cid_t1, cid_t2}
    assert "terminal status CANCELLED" in details[cid_t1]
    assert "terminal status CANCELLED" in details[cid_t2]


async def test_terminal_transition_without_conditions_emits_nothing() -> None:
    """No open conditions -> no RESOLVED markers on the wire."""
    sink = ListSink()
    steerer = _bound_steerer(sink, observation_only=True)
    session = _session()

    await steerer.tasks.mark_task_completed("t1", session=session)

    assert _resolved_events(sink) == []


# ---------------------------------------------------------------------------
# On-task verdict resolution — judge-authored kinds only, staleness-gated
# ---------------------------------------------------------------------------


def _on_task_steerer(sink: ListSink, *, observation_only: bool) -> DefaultSteerer:
    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"classification": "on_task", "reason": "focused"})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        steering_config=SteeringConfig(observation_only=observation_only),
    )
    steerer.bind(sinks=[sink], planner=NullPlanner())
    return steerer


async def _drain_judges(steerer: DefaultSteerer) -> None:
    pending = list(steerer._background_judges)
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"background judge raised {r!r}; expected clean completion"
        )


@pytest.mark.parametrize("observation_only", [True, False])
async def test_on_task_verdict_resolves_reasoning_pipeline_conditions(
    observation_only: bool,
) -> None:
    """A live ON-TASK verdict resolves reasoning-pipeline kinds for its
    (task, agent, run) — including unattributed embedding conditions —
    and leaves deterministic-detector and other-task conditions open."""
    sink = ListSink()
    steerer = _on_task_steerer(sink, observation_only=observation_only)
    session = _session()

    cid_judge = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="worker")
    cid_embed = _open(session, kind=DriftKind.LOOPING_REASONING, task_id="t1", agent_id="")
    cid_tool_loop = _open(
        session, kind=DriftKind.LOOPING_TOOL_CALL, task_id="t1", agent_id="worker"
    )
    cid_other_task = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t2", agent_id="worker")

    # Live invocation so the late-verdict staleness gate does not fire.
    store = StateStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        await steerer.drift.observe_reasoning(
            "compare panel efficiency specs", session=session, agent_name="worker"
        )
        await _drain_judges(steerer)
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    # Judge-authored + unattributed reasoning conditions resolved;
    # deterministic-detector and other-task conditions untouched.
    assert _active_condition_ids(session) == {cid_tool_loop, cid_other_task}
    resolved = _resolved_events(sink)
    assert {e.drift_detected.condition_id for e in resolved} == {cid_judge, cid_embed}
    for evt in resolved:
        assert "on-task verdict" in evt.drift_detected.detail


async def test_stale_on_task_verdict_does_not_resolve() -> None:
    """A verdict landing with no live invocation (goldfive#319) must not
    resolve anything — same staleness gate as the drift branch."""
    sink = ListSink()
    steerer = _on_task_steerer(sink, observation_only=True)
    session = _session()

    cid = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="worker")

    # No invocation registered — the agent has moved on.
    await steerer.drift.observe_reasoning(
        "compare panel efficiency specs", session=session, agent_name="worker"
    )
    await _drain_judges(steerer)

    assert _active_condition_ids(session) == {cid}
    assert _resolved_events(sink) == []


# ---------------------------------------------------------------------------
# Wire round-trip
# ---------------------------------------------------------------------------


async def test_resolved_event_round_trips_on_the_wire() -> None:
    """DRIFT_LIFECYCLE_RESOLVED survives serialize -> parse."""
    from goldfive.pb.goldfive.v1 import types_pb2

    sink = ListSink()
    steerer = _bound_steerer(sink, observation_only=True)
    session = _session()
    cid = _open(session, kind=DriftKind.OFF_TOPIC, task_id="t1", agent_id="worker")

    await steerer.tasks.mark_task_completed("t1", session=session)

    (evt,) = _resolved_events(sink)
    restored = type(evt).FromString(evt.SerializeToString())
    assert restored.WhichOneof("payload") == "drift_detected"
    assert restored.drift_detected.condition_id == cid
    assert restored.drift_detected.lifecycle == types_pb2.DRIFT_LIFECYCLE_RESOLVED
    assert restored.drift_detected.severity == types_pb2.DRIFT_SEVERITY_INFO
