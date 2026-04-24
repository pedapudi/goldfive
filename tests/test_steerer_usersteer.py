"""Tests for USER_STEER / USER_CANCEL / USER_PAUSE handling (part of #71).

Covers:

* :class:`DefaultSteerer.observe` translating ``ControlMessage`` values
  into the matching ``USER_*`` drift and invoking ``planner.refine``
  (or not, for ``PAUSE``).
* :class:`LLMPlanner.refine` on a ``USER_STEER`` drift taking the
  delete-and-replan path: completed tasks are preserved verbatim, the
  LLM produces fresh pending work, and revision metadata is stamped
  with ``revision_kind == "user_steer"``.
* ``CANCEL`` producing a CRITICAL-severity drift.
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

from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.planner import LLMPlanner  # noqa: E402
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
        self.closed = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True

    @property
    def proto_events(self) -> list[Any]:
        """goldfive a4: filter dict-envelope sidecars."""
        return [e for e in self.events if hasattr(e, "WhichOneof")]


class RecordingPlanner:
    """Planner stub that records ``refine`` calls and returns a fixed plan."""

    def __init__(self, *, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []

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
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        return self.revised


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


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship the thing")],
        plan=plan if plan is not None else _make_plan(),
        current_task_id="t2",
    )


def _bind_fresh() -> tuple[DefaultSteerer, Session, ListSink, RecordingPlanner]:
    steerer = DefaultSteerer()
    session = _make_session()
    sink = ListSink()
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2b", title="Replanned T2"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2b")],
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )
    planner = RecordingPlanner(revised=revised)
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink, planner


# ---------------------------------------------------------------------------
# DefaultSteerer.observe — ControlMessage handling
# ---------------------------------------------------------------------------


async def test_observe_steer_triggers_user_steer_drift_and_refine() -> None:
    steerer, session, sink, planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "focus on clarity"},
    )

    await steerer.observe(msg, session)

    # Planner.refine was called with a USER_STEER drift carrying the note.
    assert len(planner.refine_calls) == 1
    drift: DriftEvent = planner.refine_calls[0]["drift"]
    assert drift.kind is DriftKind.USER_STEER
    assert drift.severity is DriftSeverity.WARNING
    assert drift.detail == "focus on clarity"
    assert drift.current_task_id == "t2"

    # The revised plan was installed on the session.
    assert session.plan is not None
    assert [t.id for t in session.plan.tasks] == ["t1", "t2b"]

    # DriftDetected + PlanRevised events were emitted.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert "drift_detected" in kinds
    assert "plan_revised" in kinds


async def test_observe_cancel_triggers_critical_user_cancel_drift() -> None:
    steerer, session, sink, planner = _bind_fresh()
    # CANCEL also goes through the refine pipeline (CRITICAL >= WARNING).
    # A planner that returns None for USER_CANCEL is fine — the drift
    # still gets emitted.
    planner.revised = None

    msg = ControlMessage(
        kind=ControlKind.CANCEL,
        payload={"reason": "operator abort"},
    )
    await steerer.observe(msg, session)

    assert len(planner.refine_calls) == 1
    drift: DriftEvent = planner.refine_calls[0]["drift"]
    assert drift.kind is DriftKind.USER_CANCEL
    assert drift.severity is DriftSeverity.CRITICAL
    assert drift.detail == "operator abort"

    # Original DriftDetected + refine-failure follow-up DriftDetected
    # (planner returned None); no PlanRevised.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds.count("drift_detected") == 2
    assert "plan_revised" not in kinds


async def test_observe_pause_emits_drift_but_does_not_refine() -> None:
    steerer, session, sink, planner = _bind_fresh()
    msg = ControlMessage(kind=ControlKind.PAUSE, payload={"note": "coffee"})

    await steerer.observe(msg, session)

    # INFO severity → refine must NOT be called.
    assert planner.refine_calls == []

    # DriftDetected was still emitted.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected"]
    drift_pb = sink.events[0].drift_detected
    # Detail carries the note.
    assert drift_pb.detail == "coffee"


async def test_observe_non_control_event_still_uses_classifier() -> None:
    """Regression: intercepting ControlMessages must not shadow heuristics."""
    steerer, session, sink, planner = _bind_fresh()

    # A tool-error shape — detect_drift should still classify this.
    await steerer.observe({"error": "boom"}, session)

    # Should have produced a tool_error drift (classify_tool_error path).
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert "drift_detected" in kinds


# ---------------------------------------------------------------------------
# goldfive#171 — STEER idempotency, author propagation
# ---------------------------------------------------------------------------


async def test_observe_steer_dedupes_by_annotation_id() -> None:
    """Second delivery of the same STEER annotation is a no-op.

    Two ``ControlMessage`` values with different ``id`` but the same
    ``payload['annotation_id']`` must only trigger one refine. Models
    the delivery-retry scenario.
    """
    steerer, session, _sink, planner = _bind_fresh()

    first = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-retry-1",
        payload={"note": "pivot", "annotation_id": "ann_abc"},
    )
    second = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-retry-2",
        payload={"note": "pivot", "annotation_id": "ann_abc"},
    )
    await steerer.observe(first, session)
    await steerer.observe(second, session)

    assert len(planner.refine_calls) == 1
    assert session.state.get("goldfive.processed_steer_ids") == ["ann_abc"]


async def test_observe_steer_dedupes_by_control_id_when_annotation_id_absent() -> None:
    """Bridges that don't source annotations fall back to ControlMessage.id."""
    steerer, session, _sink, planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-noid-1",
        payload={"note": "pivot"},
    )

    await steerer.observe(msg, session)
    # Same id, deliberately re-delivered.
    await steerer.observe(msg, session)

    assert len(planner.refine_calls) == 1
    assert session.state.get("goldfive.processed_steer_ids") == ["ctl-noid-1"]


async def test_observe_steer_distinct_ids_each_trigger_refine() -> None:
    """Two distinct STEER annotations each fire refine exactly once."""
    steerer, session, _sink, planner = _bind_fresh()
    first = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-a",
        payload={"note": "one", "annotation_id": "ann_1"},
    )
    second = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-b",
        payload={"note": "two", "annotation_id": "ann_2"},
    )

    await steerer.observe(first, session)
    await steerer.observe(second, session)

    assert len(planner.refine_calls) == 2
    assert session.state.get("goldfive.processed_steer_ids") == ["ann_1", "ann_2"]


async def test_observe_dedupe_does_not_affect_heuristic_drifts() -> None:
    """LOOPING_REASONING-class heuristic drifts are never deduped.

    The dedupe set lives only for user-originated STEER annotations;
    content-based classifiers (tool errors, refusals, loops) must
    remain free-running so repeated heuristic signals still escalate.

    This test deliberately fires two identical back-to-back tool-error
    drifts, which would otherwise be suppressed by the plan-revision
    cooldown (goldfive feedback-loop fix). The cooldown is an
    orthogonal rate limit covered by
    ``tests/test_plan_revision_cooldown.py``; here we disable it so
    the dedup-set assertion can stand on its own.
    """
    steerer, session, _sink, planner = _bind_fresh()
    # Hot-patch the cooldown off for this test only. ``_bind_fresh``
    # builds a default-configured steerer; we need the cooldown window
    # at 0 so the two identical tool-error observations below both
    # trigger refine.
    steerer._plan_revision_cooldown_seconds = 0.0

    # A steer populates processed_steer_ids.
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-s",
        payload={"note": "go", "annotation_id": "ann_s"},
    )
    await steerer.observe(msg, session)
    # A tool-error event — classified as TOOL_ERROR, must still emit.
    await steerer.observe({"error": "boom"}, session)
    await steerer.observe({"error": "boom"}, session)

    # planner.refine: 1 steer + 2 tool_error observations.
    assert len(planner.refine_calls) == 3


async def test_observe_steer_stamps_annotation_id_on_drift_detected(
) -> None:
    """DriftDetected proto carries the source annotation_id (goldfive#176).

    Without this field, harmonograf can't dedup the drift row against the
    source annotation — a single user STEER surfaces as three cards
    (annotation row + drift row + plan_revised row) in the Intervention
    view. See harmonograf#75.
    """
    steerer, session, sink, _planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-ann",
        payload={
            "note": "refocus",
            "author": "alice",
            "annotation_id": "ann_abc123",
        },
    )

    await steerer.observe(msg, session)

    drift_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"
    ]
    assert drift_events, "USER_STEER must emit DriftDetected"
    assert drift_events[0].drift_detected.annotation_id == "ann_abc123"


async def test_observe_steer_without_annotation_id_leaves_drift_annotation_id_empty(
) -> None:
    """Back-compat: a STEER with no annotation_id in payload → empty field.

    Models the case where the bridge didn't forward one (older clients,
    programmatic STEERs without an annotation source). The field must
    serialize as the empty string so harmonograf's dedup falls through
    to the existing path (drift gets its own card).
    """
    steerer, session, sink, _planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-noann",
        payload={"note": "focus"},
    )

    await steerer.observe(msg, session)

    drift_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"
    ]
    assert drift_events
    assert drift_events[0].drift_detected.annotation_id == ""


async def test_observe_cancel_stamps_annotation_id_on_drift_detected(
) -> None:
    """USER_CANCEL also carries annotation_id when supplied by the bridge."""
    steerer, session, sink, planner = _bind_fresh()
    # A planner that returns None is fine for CANCEL — the drift still
    # gets emitted, and annotation_id must round-trip on it.
    planner.revised = None

    msg = ControlMessage(
        kind=ControlKind.CANCEL,
        id="ctl-cancel",
        payload={"reason": "operator abort", "annotation_id": "ann_cxl"},
    )

    await steerer.observe(msg, session)

    drift_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"
    ]
    # CANCEL emits the original drift + a refine-failure follow-up (None
    # planner). Only the first carries the source annotation_id; the
    # follow-up drift synthesized internally has no backing ControlMessage.
    assert len(drift_events) == 2
    assert drift_events[0].drift_detected.annotation_id == "ann_cxl"
    assert drift_events[1].drift_detected.annotation_id == ""


async def test_autonomous_drift_has_empty_annotation_id() -> None:
    """A drift goldfive minted itself (loop, tool error) has no annotation_id.

    Contract: only user-control drifts carry the annotation id; anything
    else (LOOPING_REASONING, TOOL_ERROR, GOAL_DRIFT, …) must emit with
    an empty string so harmonograf's deduper treats it as an autonomous
    intervention that owns its own card.
    """
    steerer, session, sink, _planner = _bind_fresh()
    # Use a tool-error event — classify_tool_error mints a drift directly
    # from an untyped event (no ControlMessage.raw).
    await steerer.observe({"error": "boom"}, session)

    drift_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "drift_detected"
    ]
    assert drift_events
    for evt in drift_events:
        assert evt.drift_detected.annotation_id == ""


async def test_observe_steer_propagates_author_into_state_and_detail() -> None:
    """``author`` from SteerPayload lands on state + drift detail prefix."""
    steerer, session, _sink, planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-auth",
        payload={
            "note": "focus on clarity",
            "author": "alice",
            "annotation_id": "ann_author",
        },
    )

    await steerer.observe(msg, session)

    assert session.state.get("goldfive.active_steer.author") == "alice"
    # Raw body — not the "by alice: ..." rewrite — lands on body.
    assert session.state.get("goldfive.active_steer.body") == "focus on clarity"
    # The drift.detail seen by the planner is the prefixed form.
    assert len(planner.refine_calls) == 1
    drift: DriftEvent = planner.refine_calls[0]["drift"]
    assert drift.detail == "by alice: focus on clarity"


async def test_observe_steer_without_author_leaves_author_empty() -> None:
    steerer, session, _sink, planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-noauth",
        payload={"note": "quiet nudge", "annotation_id": "ann_na"},
    )

    await steerer.observe(msg, session)

    assert session.state.get("goldfive.active_steer.author") == ""
    assert session.state.get("goldfive.active_steer.body") == "quiet nudge"
    drift: DriftEvent = planner.refine_calls[0]["drift"]
    # No "by X: " prefix when author is empty.
    assert drift.detail == "quiet nudge"


async def test_processed_steer_ids_list_evicts_oldest_when_capped() -> None:
    """The FIFO cap bounds the dedupe list so long sessions stay bounded."""
    from goldfive import orchestration_state as _ostate

    steerer, session, _sink, _planner = _bind_fresh()
    cap = _ostate.PROCESSED_STEER_IDS_CAP

    # Fire cap + 3 distinct steers.
    for i in range(cap + 3):
        msg = ControlMessage(
            kind=ControlKind.STEER,
            id=f"ctl-{i}",
            payload={"note": f"n{i}", "annotation_id": f"ann_{i}"},
        )
        await steerer.observe(msg, session)

    ids = session.state.get("goldfive.processed_steer_ids") or []
    assert len(ids) == cap
    # Oldest 3 were evicted.
    assert "ann_0" not in ids
    assert "ann_1" not in ids
    assert "ann_2" not in ids
    # Newest survives.
    assert ids[-1] == f"ann_{cap + 2}"


async def test_direct_user_steer_drift_event_without_raw_still_writes_state() -> None:
    """Back-compat: tests that build DriftEvent directly still work.

    Pre-#171 tests fabricate a USER_STEER DriftEvent without a raw
    ControlMessage behind it. The state writer must fall back to
    parsing ``detail`` so those tests still round-trip body/author
    into state.
    """
    steerer, session, _sink, _planner = _bind_fresh()
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="by bob: direct call",
        current_task_id="t2",
    )

    await steerer._apply_user_steer_state(drift, session)

    assert session.state.get("goldfive.active_steer.body") == "direct call"
    assert session.state.get("goldfive.active_steer.author") == "bob"
    # No dedupe id recoverable from a bare DriftEvent → processed list
    # is not populated.
    assert session.state.get("goldfive.processed_steer_ids") in (None, [])


# ---------------------------------------------------------------------------
# goldfive#139 — steerer tags the bound adapter's next cancel with a
# symbolic reason on USER_STEER drift, so the synthetic function_response
# the adapter appends on cancel carries LLM-actionable content.
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Minimal adapter stub that exposes ``_next_cancel_reason``.

    The real ``ADKAdapter`` carries the same attribute; this stub
    avoids pulling in the ADK optional dep just to verify the steerer
    side of the wiring.
    """

    def __init__(self) -> None:
        self._next_cancel_reason: str = ""


async def test_steerer_sets_adapter_next_cancel_reason_on_user_steer() -> None:
    """USER_STEER drift via ``observe`` → adapter gets tagged ``"user_steer"``.

    Regression for goldfive#139: the steerer must stash the symbolic
    reason on the bound adapter BEFORE the drift_detected event is
    emitted, so whatever downstream component reacts by cancelling
    the in-flight invoke sees a tagged adapter and the adapter's
    cancel handler picks up the LLM-actionable content variant.
    """
    steerer, session, _sink, _planner = _bind_fresh()
    adapter = _FakeAdapter()
    steerer.bind_adapter(adapter)

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "pivot to security posture"},
    )
    await steerer.observe(msg, session)

    assert adapter._next_cancel_reason == "user_steer"


async def test_steerer_does_not_tag_adapter_on_non_user_steer_drift() -> None:
    """Non-USER_STEER drift leaves the adapter tag untouched.

    Pause (USER_PAUSE, INFO) and other drift kinds must not set the
    tag — only USER_STEER maps to the explicit content variant. Other
    cancels fall through to the neutral content.
    """
    steerer, session, _sink, _planner = _bind_fresh()
    adapter = _FakeAdapter()
    steerer.bind_adapter(adapter)

    # PAUSE → USER_PAUSE drift (INFO severity, no refine).
    pause_msg = ControlMessage(kind=ControlKind.PAUSE, payload={"note": "wait"})
    await steerer.observe(pause_msg, session)
    assert adapter._next_cancel_reason == ""

    # A raw tool-error event also should not tag the adapter.
    await steerer.observe({"error": "boom"}, session)
    assert adapter._next_cancel_reason == ""


async def test_steerer_bind_adapter_tolerates_adapter_without_attr() -> None:
    """A third-party adapter that doesn't carry ``_next_cancel_reason``
    must not break the USER_STEER drift path.

    The steerer's tagging helper swallows the AttributeError / similar
    and logs at DEBUG. The drift still flows through ``_handle_drift``
    (refine etc.) normally.
    """
    class _FrozenAdapter:
        __slots__ = ()  # no attributes, no __dict__ — assignment raises.

    steerer, session, _sink, planner = _bind_fresh()
    steerer.bind_adapter(_FrozenAdapter())

    msg = ControlMessage(kind=ControlKind.STEER, payload={"note": "x"})
    # Must not raise.
    await steerer.observe(msg, session)
    # Refine still ran.
    assert len(planner.refine_calls) == 1


# ---------------------------------------------------------------------------
# LLMPlanner.refine — USER_STEER delete-and-replan
# ---------------------------------------------------------------------------


def _canned_user_steer_json() -> str:
    return json.dumps(
        {
            "summary": "Replanned to focus on clarity.",
            "tasks": [
                {
                    "id": "clarify",
                    "title": "Rewrite for clarity",
                    "description": "Tighten prose to match operator steer.",
                    "assignee_agent_id": "writer",
                },
                {
                    "id": "final",
                    "title": "Finalize",
                    "description": "Prep for delivery.",
                    "assignee_agent_id": "writer",
                },
            ],
            "edges": [{"from_task_id": "clarify", "to_task_id": "final"}],
        }
    )


class _StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response


async def test_llm_planner_user_steer_preserves_completed_adds_pending() -> None:
    plan = Plan(
        id="plan-abc",
        run_id="run-xyz",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Research", status=TaskStatus.COMPLETED),
            Task(id="draft", title="Draft", status=TaskStatus.FAILED),
            Task(id="review", title="Review", status=TaskStatus.PENDING),
            Task(id="publish", title="Publish", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
            TaskEdge(from_task_id="draft", to_task_id="review"),
            TaskEdge(from_task_id="review", to_task_id="publish"),
        ],
        revision_index=2,
    )
    goals = [Goal(id="g1", summary="ship the blog post")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="skip formal review, focus on clarity",
        current_task_id="review",
    )

    llm = _StubLLM(_canned_user_steer_json())
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=drift, goals=goals)
    assert revised is not None

    # Lineage preserved.
    assert revised.id == "plan-abc"
    assert revised.run_id == "run-xyz"

    # Completed/failed tasks preserved verbatim at the start.
    ids = [t.id for t in revised.tasks]
    assert ids[0] == "research"
    assert ids[1] == "draft"
    assert revised.tasks[0].status is TaskStatus.COMPLETED
    assert revised.tasks[1].status is TaskStatus.FAILED
    # Pending tasks from the old plan are gone.
    assert "review" not in ids
    assert "publish" not in ids
    # New pending tasks from the LLM are present.
    assert "clarify" in ids
    assert "final" in ids
    for t in revised.tasks[2:]:
        assert t.status is TaskStatus.PENDING

    # Revision metadata stamped.
    assert revised.revision_kind == DriftKind.USER_STEER.value
    assert revised.revision_severity == DriftSeverity.WARNING.value
    assert revised.revision_reason == "user steering: skip formal review, focus on clarity"
    assert revised.revision_index == plan.revision_index + 1

    # The LLM was called once, and the user prompt explicitly told it
    # to preserve completed tasks verbatim (delete-and-replan contract).
    assert len(llm.calls) == 1
    system_prompt, user_prompt, model = llm.calls[0]
    assert model == "test-model"
    assert "PRESERVE" in system_prompt.upper() or "preserve" in system_prompt
    assert "skip formal review" in user_prompt
    # History-context block is present; new work is not asked to repeat them.
    assert "research" in user_prompt
    assert "draft" in user_prompt

    # Old edges that referenced dropped pending tasks are gone; edges
    # that survive reference only known ids.
    known = {t.id for t in revised.tasks}
    for e in revised.edges:
        assert e.from_task_id in known
        assert e.to_task_id in known


async def test_llm_planner_user_steer_empty_response_returns_none() -> None:
    plan = _make_plan()
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
    )
    planner = LLMPlanner(call_llm=_StubLLM(""))
    assert await planner.refine(plan=plan, drift=drift, goals=goals) is None


async def test_llm_planner_non_user_steer_uses_default_refine_path() -> None:
    """Regression: non-USER_STEER drift must keep the original refine contract."""
    plan = _make_plan()
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="api 500",
    )
    llm = _StubLLM(
        json.dumps(
            {
                "summary": "retry path",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "T1",
                        "status": "COMPLETED",
                        "assignee_agent_id": "a",
                    },
                    {
                        "id": "t2",
                        "title": "T2",
                        "status": "RUNNING",
                        "assignee_agent_id": "a",
                    },
                    {
                        "id": "t3",
                        "title": "T3",
                        "status": "PENDING",
                        "assignee_agent_id": "a",
                    },
                ],
                "edges": [],
            }
        )
    )
    planner = LLMPlanner(call_llm=llm)
    revised = await planner.refine(plan=plan, drift=drift, goals=goals)
    assert revised is not None
    # Default-path stamping uses str(drift.kind), not "user_steer".
    assert revised.revision_kind == str(DriftKind.TOOL_ERROR)
    assert revised.revision_reason == "api 500"


# ---------------------------------------------------------------------------
# PlanRevised trigger_event_id stamping (goldfive#199 / harmonograf#95 rescope)
# ---------------------------------------------------------------------------


async def test_plan_revised_stamps_annotation_id_from_user_steer(
) -> None:
    """PlanRevised.trigger_event_id uses the source annotation_id for USER_STEER.

    Rescoped from goldfive#196 (harmonograf#95): the strict-id dedup key
    is now ``trigger_event_id``. For user-control refines, it resolves to
    the annotation_id carried by the originating ControlMessage.
    """
    steerer, session, sink, _planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-pr",
        payload={
            "note": "pivot",
            "author": "alice",
            "annotation_id": "ann_pr_123",
        },
    )

    await steerer.observe(msg, session)

    revised_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]
    assert revised_events, "USER_STEER with successful refine must emit PlanRevised"
    assert revised_events[0].plan_revised.trigger_event_id == "ann_pr_123"
    # The id is also persisted on the Plan proto itself so out-of-band
    # emitters (SequentialExecutor plan-swap detector) can recover it.
    assert revised_events[0].plan_revised.plan.revision_trigger_event_id == "ann_pr_123"


async def test_plan_revised_without_annotation_id_falls_back_to_drift_id(
) -> None:
    """Rescope: a STEER without annotation_id falls back to the drift.id.

    Post-goldfive#199, ``trigger_event_id`` must always be non-empty on
    refine events. When the originating ControlMessage lacks an
    annotation_id (edge case — bridge misconfigured), the steerer uses
    the ``DriftEvent.id`` so harmonograf still has a strict dedup key.
    """
    steerer, session, sink, _planner = _bind_fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-pr-noann",
        payload={"note": "pivot"},
    )

    await steerer.observe(msg, session)

    revised_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]
    assert revised_events
    # Non-empty — either annotation_id (absent here) or drift.id.
    trig = revised_events[0].plan_revised.trigger_event_id
    assert trig != ""
    assert revised_events[0].plan_revised.plan.revision_trigger_event_id == trig


async def test_apply_revision_stamps_trigger_event_id_from_annotation(
) -> None:
    """``_apply_revision`` uses the source annotation_id as the trigger_event_id
    for user-control refines (goldfive#199).

    Previously (#196) this field was only populated for user-control
    drifts via the raw ControlMessage payload. Rescope keeps that
    behaviour but the field is now the unified ``trigger_event_id``.
    """
    from goldfive.steerer import DefaultSteerer

    session = _make_session()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-ar",
        payload={"note": "refocus", "annotation_id": "ann_ar_77"},
    )
    drift = DefaultSteerer._drift_from_control(msg, session)
    assert drift is not None

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2b", title="Replan"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2b")],
    )
    DefaultSteerer._apply_revision(session, revised, drift)

    assert session.plan is revised
    assert revised.revision_trigger_event_id == "ann_ar_77"


async def test_apply_revision_stamps_drift_id_on_autonomous_drift(
) -> None:
    """``_apply_revision`` uses ``drift.id`` when no annotation_id is present.

    goldfive#199: autonomous drifts (LOOPING_REASONING, TOOL_ERROR, …)
    fall back to ``DriftEvent.id`` — the UUID4 minted at construction —
    so harmonograf can strict-id-merge the plan-revision row to the
    drift row without a time-window fallback.
    """
    from goldfive.steerer import DefaultSteerer
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    session = _make_session()
    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="loop detected",
    )
    assert drift.id  # populated by default factory

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", status=TaskStatus.COMPLETED)],
        edges=[],
    )
    DefaultSteerer._apply_revision(session, revised, drift)
    assert revised.revision_trigger_event_id == drift.id


async def test_apply_revision_preserves_prestamped_trigger_event_id() -> None:
    """A plan already carrying ``revision_trigger_event_id`` isn't overwritten.

    Protects validator-retry chaining: if the planner pre-stamps the
    retry attempt's plan with the original trigger id, re-applying the
    revision must not clobber it.
    """
    from goldfive.steerer import DefaultSteerer

    session = _make_session()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-ar2",
        payload={"note": "refocus", "annotation_id": "ann_from_drift"},
    )
    drift = DefaultSteerer._drift_from_control(msg, session)
    assert drift is not None

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", status=TaskStatus.COMPLETED)],
        edges=[],
        revision_trigger_event_id="ann_prestamped",
    )
    DefaultSteerer._apply_revision(session, revised, drift)
    assert revised.revision_trigger_event_id == "ann_prestamped"


async def test_autonomous_refine_stamps_drift_id_on_plan_revised(
) -> None:
    """Autonomous drift → PlanRevised.trigger_event_id == drift.id (goldfive#199).

    Previously (#196) autonomous refines left the field empty;
    harmonograf then relied on a time-window fallback to dedup. Rescope:
    every PlanRevised carries a strict id — drift.id for autonomous,
    annotation_id for user-control.
    """
    steerer, session, sink, _planner = _bind_fresh()
    # Feed an untyped event — classify_tool_error path, no ControlMessage.
    await steerer.observe({"error": "boom"}, session)
    revised_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "plan_revised"
    ]
    # Autonomous refines MUST carry a non-empty trigger_event_id — it
    # comes from the synthesized DriftEvent.id.
    for evt in revised_events:
        assert evt.plan_revised.trigger_event_id != ""
        # Plan mirror must match so persistence round-trips the id.
        assert (
            evt.plan_revised.plan.revision_trigger_event_id
            == evt.plan_revised.trigger_event_id
        )
