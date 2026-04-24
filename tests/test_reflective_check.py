"""Unit tests for the opt-in reflective self-progress check.

Exercises :meth:`goldfive.steerer.DefaultSteerer.note_llm_call` and
:meth:`maybe_run_reflective_check`:

* Counter fires at the configured interval.
* A ``{"making_progress": true, "confidence": 0.9}`` response emits no
  drift.
* A ``{"making_progress": true, "confidence": 0.3}`` response emits an
  INFO ``UNCERTAIN_PROGRESS`` drift.
* A ``{"making_progress": false, ...}`` response emits a WARNING
  ``SELF_REPORTED_STUCK`` drift (and refine runs).
* Malformed / non-JSON responses do not crash; an INFO ``CUSTOM`` drift
  noting the reflective check itself failed is emitted.
* Leaving ``reflective_call_llm`` unset keeps the counter inert -- no
  check ever runs.
* Custom interval (e.g. ``5``) is respected.
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

# ---------------------------------------------------------------------------
# Stubs (mirror the shape used in test_steerer.py so failures are easy to
# cross-reference)
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    @property
    def proto_events(self) -> list[Any]:
        """goldfive a4: filter dict-envelope sidecars (refine_attempted /
        refine_failed / correlation plan_revised) so legacy proto-only
        assertions still hold."""
        return [e for e in self.events if hasattr(e, "WhichOneof")]


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, *, goals, available_agents, context=None):
        return None

    async def refine(self, *, plan, drift, goals):
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        return None


def _make_stub_call_llm(responses: list[Any]):
    """Build an async call_llm stub that pops responses in order.

    Dicts are json-encoded; strings are returned verbatim. Exceptions
    are raised by the stub -- the steerer must treat that gracefully.
    """
    queue = list(responses)

    async def _call_llm(system: str, user: str, model: str) -> str:
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)

    return _call_llm


def _make_session_with_task() -> Session:
    """Fresh session with a single RUNNING task the steerer can reflect on."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Gather facts", description="Collect 5 sources"),
            Task(id="t2", title="Write draft"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship the thing")],
        plan=plan,
    )
    session.current_task_id = "t1"
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_reflective_check_fires_at_interval() -> None:
    """15 LLM observations produce exactly one reflective check call."""
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        return json.dumps({"making_progress": True, "confidence": 0.9})

    steerer = DefaultSteerer(
        reflective_check_interval=15,
        reflective_call_llm=_call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(14):
        await steerer.note_llm_call(session)
    assert len(calls) == 0, "check should not fire before the 15th call"
    await steerer.note_llm_call(session)
    assert len(calls) == 1, "check should fire on the 15th call"
    # Counter resets after a check; another 14 should not re-fire.
    for _ in range(14):
        await steerer.note_llm_call(session)
    assert len(calls) == 1
    await steerer.note_llm_call(session)
    assert len(calls) == 2


async def test_reflective_check_making_progress_emits_no_drift() -> None:
    """{making_progress:true, confidence:0.9} → no DriftDetected event."""
    steerer = DefaultSteerer(
        reflective_check_interval=3,
        reflective_call_llm=_make_stub_call_llm(
            [{"making_progress": True, "confidence": 0.9, "reason": "tools ran"}]
        ),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(3):
        await steerer.note_llm_call(session)
    # No drift_detected events emitted (high-confidence yes).
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert "drift_detected" not in kinds
    # Planner was never asked to refine.
    assert planner.refine_calls == []


async def test_reflective_check_low_confidence_emits_uncertain() -> None:
    """{making_progress:true, confidence:0.3} → INFO UNCERTAIN_PROGRESS drift."""
    steerer = DefaultSteerer(
        reflective_check_interval=2,
        reflective_call_llm=_make_stub_call_llm(
            [{"making_progress": True, "confidence": 0.3, "reason": "not sure"}]
        ),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(2):
        await steerer.note_llm_call(session)
    drifts = [e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"]
    assert len(drifts) == 1
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drifts[0].drift_detected.kind == types_pb2.DRIFT_KIND_UNCERTAIN_PROGRESS
    assert drifts[0].drift_detected.severity == types_pb2.DRIFT_SEVERITY_INFO
    assert drifts[0].drift_detected.current_task_id == "t1"
    # INFO does not trigger refine.
    assert planner.refine_calls == []


async def test_reflective_check_stuck_emits_warning() -> None:
    """{making_progress:false} → WARNING SELF_REPORTED_STUCK drift + refine."""
    steerer = DefaultSteerer(
        reflective_check_interval=2,
        reflective_call_llm=_make_stub_call_llm(
            [
                {
                    "making_progress": False,
                    "confidence": 0.85,
                    "reason": "same tool args repeatedly",
                }
            ]
        ),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(2):
        await steerer.note_llm_call(session)
    drifts = [e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"]
    # The WARNING drift flows through _handle_drift, which also emits a
    # follow-up drift_detected when planner.refine returns None (refine
    # failure surfaces as a CRITICAL drift). Both should carry the same
    # current_task_id.
    assert len(drifts) >= 1
    from goldfive.pb.goldfive.v1 import types_pb2

    primary = drifts[0]
    assert primary.drift_detected.kind == types_pb2.DRIFT_KIND_SELF_REPORTED_STUCK
    assert primary.drift_detected.severity == types_pb2.DRIFT_SEVERITY_WARNING
    assert primary.drift_detected.current_task_id == "t1"
    # WARNING drifts trigger refine.
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.SELF_REPORTED_STUCK
    assert planner.refine_calls[0]["drift"].severity is DriftSeverity.WARNING


async def test_reflective_check_malformed_response_is_graceful() -> None:
    """Garbage / non-JSON response → INFO CUSTOM drift; no crash."""
    steerer = DefaultSteerer(
        reflective_check_interval=1,
        reflective_call_llm=_make_stub_call_llm(["not json at all, sorry"]),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    await steerer.note_llm_call(session)
    drifts = [e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"]
    assert len(drifts) == 1
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drifts[0].drift_detected.kind == types_pb2.DRIFT_KIND_CUSTOM
    assert drifts[0].drift_detected.severity == types_pb2.DRIFT_SEVERITY_INFO
    assert "reflective_check_failed" in drifts[0].drift_detected.detail


async def test_reflective_check_raised_exception_is_graceful() -> None:
    """call_llm raising → INFO CUSTOM drift; no crash."""
    steerer = DefaultSteerer(
        reflective_check_interval=1,
        reflective_call_llm=_make_stub_call_llm([RuntimeError("network down")]),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    await steerer.note_llm_call(session)  # should not raise
    drifts = [e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"]
    assert len(drifts) == 1
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drifts[0].drift_detected.kind == types_pb2.DRIFT_KIND_CUSTOM
    assert "reflective_check_failed" in drifts[0].drift_detected.detail
    assert "network down" in drifts[0].drift_detected.detail


async def test_reflective_check_disabled_by_default() -> None:
    """Without reflective_call_llm the counter is inert and no check fires."""
    steerer = DefaultSteerer()  # no reflective_call_llm
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(100):
        await steerer.note_llm_call(session)
    # Counter never incremented (feature is off), and no drift emitted.
    assert session._llm_calls_since_check == 0
    assert [e.WhichOneof("payload") for e in sink.proto_events] == []
    # Even calling maybe_run_reflective_check directly is a no-op.
    await steerer.maybe_run_reflective_check(session)
    assert [e.WhichOneof("payload") for e in sink.proto_events] == []


async def test_reflective_check_configurable_interval() -> None:
    """interval=5 → check fires every 5 LLM calls."""
    call_count = 0

    async def _call_llm(system: str, user: str, model: str) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps({"making_progress": True, "confidence": 0.9})

    steerer = DefaultSteerer(
        reflective_check_interval=5,
        reflective_call_llm=_call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    for _ in range(5):
        await steerer.note_llm_call(session)
    assert call_count == 1
    for _ in range(5):
        await steerer.note_llm_call(session)
    assert call_count == 2
    for _ in range(4):
        await steerer.note_llm_call(session)
    assert call_count == 2  # 14 total → still 2 checks
    await steerer.note_llm_call(session)
    assert call_count == 3  # 15 total → 3 checks


async def test_reflective_check_resets_counter_on_task_transition() -> None:
    """Switching current_task_id mid-window resets the counter."""

    async def _call_llm(system: str, user: str, model: str) -> str:
        return json.dumps({"making_progress": True, "confidence": 0.9})

    steerer = DefaultSteerer(
        reflective_check_interval=5,
        reflective_call_llm=_call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    # First 3 calls on t1.
    for _ in range(3):
        await steerer.note_llm_call(session)
    assert session._llm_calls_since_check == 3
    # Simulate task transition -- executor updates current_task_id.
    session.current_task_id = "t2"
    await steerer.note_llm_call(session)
    # Counter resets to 1 (one new call on t2) -- we did not carry
    # over from t1.
    assert session._llm_calls_since_check == 1
    assert session._reflective_check_task_id == "t2"


async def test_reflective_check_tolerates_fenced_markdown() -> None:
    """A response wrapped in markdown code fences is parsed successfully."""
    steerer = DefaultSteerer(
        reflective_check_interval=1,
        reflective_call_llm=_make_stub_call_llm(
            [
                '```json\n{"making_progress": false, "confidence": 0.9, '
                '"reason": "stuck on same tool call"}\n```'
            ]
        ),
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    await steerer.note_llm_call(session)
    drifts = [e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drifts[0].drift_detected.kind == types_pb2.DRIFT_KIND_SELF_REPORTED_STUCK


async def test_reflective_check_no_current_task_is_noop() -> None:
    """A session without a current task id does not fire a check."""
    calls: list[Any] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user))
        return json.dumps({"making_progress": True, "confidence": 0.9})

    steerer = DefaultSteerer(
        reflective_check_interval=1,
        reflective_call_llm=_call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)

    session = _make_session_with_task()
    session.current_task_id = ""  # no active task
    await steerer.note_llm_call(session)
    # The counter incremented (feature enabled), but the check found no
    # task and bailed cleanly without calling call_llm.
    assert calls == []
