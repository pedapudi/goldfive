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

    This test deliberately fires two heuristic tool-error drifts on
    DIFFERENT current_task_ids so each opens a fresh drift condition
    (goldfive I5 — same kind+task within a turn now collapses onto one
    refine; cross-task dispatches still refine independently). The
    plan-revision cooldown is also hot-patched off so a co-located
    cluster doesn't shadow the dedupe-set assertion.
    """
    steerer, session, _sink, planner = _bind_fresh()
    # The plan-revision cooldown was deleted in goldfive#215 iter-8 P2;
    # the new outcome gate keys on ``(kind, task)`` so the two
    # tool-error observations below — on distinct pinned tasks — each
    # mint their own outcome and reach refine independently.

    # A steer populates processed_steer_ids.
    msg = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-s",
        payload={"note": "go", "annotation_id": "ann_s"},
    )
    await steerer.observe(msg, session)
    # Two tool-error events on DIFFERENT pinned tasks — each is its own
    # drift condition (per-condition gate is keyed on kind+task+agent
    # within a turn) and therefore gets its own refine.
    await steerer.observe({"error": "boom", "task_id": "t1"}, session)
    await steerer.observe({"error": "boom", "task_id": "t2"}, session)

    # planner.refine: 1 steer + 2 tool_error observations across distinct tasks.
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
    # goldfive#247: _apply_revision returns the stamped Plan; the
    # input Plan stays unchanged (frozen). goldfive#254: instance
    # method (consults ``self._should_inject()`` to gate the install
    # in observation-only mode); the test-suite fixture flips the
    # default to active-steering so the install lands.
    revised = DefaultSteerer()._apply_revision(session, revised, drift)

    assert session.plan is not None and session.plan.id == revised.id
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
    # goldfive#247: _apply_revision returns the stamped Plan; the input
    # stays unchanged (frozen). goldfive#254: instance method now.
    revised = DefaultSteerer()._apply_revision(session, revised, drift)
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
    # goldfive#254: instance method now.
    DefaultSteerer()._apply_revision(session, revised, drift)
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


# Phase 4 (goldfive#271) note: the qualification-merge regex helpers
# (_extract_qualifications, _extract_output_type, _looks_like_explicit_removal,
# _merge_prior_qualifications_into_goal) were deleted. The
# qualification-merge is now done by the planner LLM's handle_turn
# prompt directly — see test_planner_handle_turn.py for the
# replacement coverage.


# ---------------------------------------------------------------------------
# Evolution-aware refine_steer (goldfive#207 — refine cascade fix). When
# the LLM reuses a prior PENDING id, the merge preserves the id and the
# new title/description/assignee. When the LLM omits a prior PENDING id,
# it's dropped. This stabilises condition_id keying across cascading
# refines — see compute_condition_id in orchestration_state.py.
# ---------------------------------------------------------------------------


def _evolving_plan() -> Plan:
    return Plan(
        id="plan-evo",
        run_id="run-evo",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Research", status=TaskStatus.COMPLETED),
            Task(
                id="create_presentation",
                title="Create presentation",
                description="Create the presentation about solar panels",
                assignee_agent_id="web_developer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="research", to_task_id="create_presentation")],
        revision_index=1,
    )


async def test_refine_steer_reuses_prior_pending_id_when_llm_does() -> None:
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship the presentation")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="focus on efficiency facts only",
        current_task_id="create_presentation",
    )
    canned = json.dumps(
        {
            "summary": "Refined presentation focused on efficiency.",
            "tasks": [
                {
                    "id": "create_presentation",  # REUSED
                    "title": "Create efficiency-focused presentation",
                    "description": "Create the presentation, efficiency facts only",
                    "assignee_agent_id": "web_developer",
                }
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "create_presentation"}],
        }
    )
    llm = _StubLLM(canned)
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=drift, goals=goals)
    assert revised is not None

    ids = [t.id for t in revised.tasks]
    assert ids == ["research", "create_presentation"]
    evolved = next(t for t in revised.tasks if t.id == "create_presentation")
    assert evolved.title == "Create efficiency-focused presentation"
    assert "efficiency facts" in evolved.description


async def test_refine_steer_drops_prior_pending_when_llm_omits() -> None:
    plan = Plan(
        id="plan-drop",
        run_id="run-drop",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Research", status=TaskStatus.COMPLETED),
            Task(id="task_a", title="Task A", status=TaskStatus.PENDING),
            Task(id="task_b", title="Task B", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="task_a"),
            TaskEdge(from_task_id="research", to_task_id="task_b"),
        ],
        revision_index=2,
    )
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="drop task A; only do task B",
        current_task_id="task_a",
    )
    # LLM omits task_a; keeps task_b with reused id.
    canned = json.dumps(
        {
            "summary": "Drop A; keep B.",
            "tasks": [
                {
                    "id": "task_b",
                    "title": "Task B",
                    "description": "the only remaining work",
                    "assignee_agent_id": "writer",
                }
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "task_b"}],
        }
    )
    revised = await LLMPlanner(call_llm=_StubLLM(canned), model="m").refine(
        plan=plan, drift=drift, goals=goals
    )
    assert revised is not None
    ids = [t.id for t in revised.tasks]
    assert "task_a" not in ids
    assert "task_b" in ids


async def test_refine_steer_supersedes_prior_pending_with_replace_kind() -> None:
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship the presentation")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="completely different presentation now — drop the prior approach",
        current_task_id="create_presentation",
    )
    # LLM mints a fresh id with supersedes pointing at the prior PENDING id.
    canned = json.dumps(
        {
            "summary": "Brand-new presentation replacing prior.",
            "tasks": [
                {
                    "id": "new_presentation",
                    "title": "Different presentation",
                    "description": "structurally different work",
                    "assignee_agent_id": "web_developer",
                    "supersedes": "create_presentation",
                    "supersedes_kind": "REPLACE",
                }
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "new_presentation"}],
        }
    )
    revised = await LLMPlanner(call_llm=_StubLLM(canned), model="m").refine(
        plan=plan, drift=drift, goals=goals
    )
    assert revised is not None
    ids = [t.id for t in revised.tasks]
    assert "new_presentation" in ids
    assert "create_presentation" not in ids
    new = next(t for t in revised.tasks if t.id == "new_presentation")
    assert new.supersedes == "create_presentation"
    # supersedes_kind must round-trip as REPLACE (the LLM set it; the
    # _normalize_supersession_kinds pass should leave a status-correct
    # REPLACE alone).
    from goldfive.types import SupersessionKind  # noqa: PLC0415

    assert new.supersedes_kind is SupersessionKind.REPLACE


async def test_refine_steer_normalizes_unspecified_kind_against_prior_pending() -> None:
    """LLM forgets supersedes_kind on a PENDING-replacement; pass coerces to REPLACE.

    Mirrors the parity ``_refine_user`` already had (planner.py:2111). Without
    this, the executor's pin-redirect would not run on an LLM that emitted
    only ``supersedes`` without the kind.
    """
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship the presentation")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="rebuild from scratch",
        current_task_id="create_presentation",
    )
    canned = json.dumps(
        {
            "summary": "Replacement.",
            "tasks": [
                {
                    "id": "rebuilt_presentation",
                    "title": "Rebuild",
                    "description": "structurally different",
                    "assignee_agent_id": "web_developer",
                    "supersedes": "create_presentation",
                    # supersedes_kind intentionally omitted
                }
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "rebuilt_presentation"}],
        }
    )
    revised = await LLMPlanner(call_llm=_StubLLM(canned), model="m").refine(
        plan=plan, drift=drift, goals=goals
    )
    from goldfive.types import SupersessionKind  # noqa: PLC0415

    assert revised is not None
    new = next(t for t in revised.tasks if t.id == "rebuilt_presentation")
    assert new.supersedes_kind is SupersessionKind.REPLACE


def test_steer_prompt_handles_empty_prior_pending() -> None:
    """A steer against a plan with no PENDING tasks renders a valid prompt.

    Edge case: first refine after a session boot-up where every prior task
    happens to be terminal (or none exist yet). The prior_pending block
    must still serialise (empty list) without leaving stale references.
    """
    completed = [Task(id="research", title="Research", status=TaskStatus.COMPLETED)]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="next thing",
    )
    planner = LLMPlanner(call_llm=_StubLLM(""), model="m")
    prompt = planner._build_steer_prompt(
        completed,
        drift,
        [Goal(id="g1", summary="ship")],
        source="user",
        prior_pending=[],
    )
    # Block is rendered (with empty list) so the LLM sees "no prior pending".
    assert "Prior PENDING work" in prompt
    assert "[]" in prompt  # empty JSON list literal somewhere
    # Reusable-ids list is empty.
    assert "Reusable prior PENDING ids: []" in prompt


async def test_refine_steer_id_reuse_drops_llm_assignee() -> None:
    """goldfive#252: any LLM-emitted ``assignee_agent_id`` is dropped.

    Pre-#252 the LLM could redirect a task to a different agent by
    naming the new agent in ``assignee_agent_id`` on the same id-reused
    task. Post-#252 the parser unconditionally drops the value — the
    framework will populate the field observationally when a delegation
    actually happens.
    """
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship the presentation")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="reassign to research_agent for the presentation drafting",
        current_task_id="create_presentation",
    )
    canned = json.dumps(
        {
            "summary": "Reassigned.",
            "tasks": [
                {
                    "id": "create_presentation",
                    "title": "Create presentation",
                    "description": "Create the presentation about solar panels",
                    "assignee_agent_id": "research_agent",  # IGNORED post-#252
                }
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "create_presentation"}],
        }
    )
    revised = await LLMPlanner(call_llm=_StubLLM(canned), model="m").refine(
        plan=plan, drift=drift, goals=goals
    )
    assert revised is not None
    evolved = next(t for t in revised.tasks if t.id == "create_presentation")
    # LLM-emitted assignee dropped (goldfive#252).
    assert evolved.assignee_agent_id == ""


def test_steer_prompt_surfaces_prior_pending_block() -> None:
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER, severity=DriftSeverity.WARNING, detail="evolve"
    )
    planner = LLMPlanner(call_llm=_StubLLM(""), model="m")
    completed = [t for t in plan.tasks if t.status is TaskStatus.COMPLETED]
    prior_pending = [t for t in plan.tasks if t.status is TaskStatus.PENDING]
    prompt = planner._build_steer_prompt(
        completed, drift, goals, source="user", prior_pending=prior_pending
    )
    assert "Prior PENDING work" in prompt
    assert "create_presentation" in prompt
    assert "ID REUSE FOR CONTINUING WORK" in prompt
    assert "REUSE the prior pending id" in prompt


def test_steer_prompt_goldfive_source_also_carries_id_reuse_block() -> None:
    plan = _evolving_plan()
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="reasoning drift",
    )
    planner = LLMPlanner(call_llm=_StubLLM(""), model="m")
    completed = [t for t in plan.tasks if t.status is TaskStatus.COMPLETED]
    prior_pending = [t for t in plan.tasks if t.status is TaskStatus.PENDING]
    prompt = planner._build_steer_prompt(
        completed, drift, goals, source="goldfive", prior_pending=prior_pending
    )
    assert "Prior PENDING work" in prompt
    assert "ID REUSE FOR CONTINUING WORK" in prompt
    # goldfive source also gets the closing line that prefers id reuse.
    assert "REUSE the prior pending id" in prompt


async def test_refine_steer_id_reuse_preserves_inter_pending_edges() -> None:
    plan = Plan(
        id="plan-edges",
        run_id="run-edges",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Research", status=TaskStatus.COMPLETED),
            Task(id="step_a", title="Step A", status=TaskStatus.PENDING),
            Task(id="step_b", title="Step B", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="step_a"),
            TaskEdge(from_task_id="step_a", to_task_id="step_b"),
        ],
        revision_index=1,
    )
    goals = [Goal(id="g1", summary="ship")]
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="evolve both steps",
    )
    canned = json.dumps(
        {
            "summary": "Evolved.",
            "tasks": [
                {
                    "id": "step_a",
                    "title": "Step A (evolved)",
                    "description": "...",
                    "assignee_agent_id": "writer",
                },
                {
                    "id": "step_b",
                    "title": "Step B (evolved)",
                    "description": "...",
                    "assignee_agent_id": "writer",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "step_a"},
                {"from_task_id": "step_a", "to_task_id": "step_b"},
            ],
        }
    )
    revised = await LLMPlanner(call_llm=_StubLLM(canned), model="m").refine(
        plan=plan, drift=drift, goals=goals
    )
    assert revised is not None
    edges = {(e.from_task_id, e.to_task_id) for e in revised.edges}
    assert ("research", "step_a") in edges
    assert ("step_a", "step_b") in edges


# ---------------------------------------------------------------------------
# goldfive#213: structural backfill of Task.supersedes for retry-named tasks.
# ---------------------------------------------------------------------------
#
# When the LLM emits a retry-shaped task name (``retry_t0``, ``t0_v2``)
# but forgets to populate ``supersedes``, the merge-time backfill
# infers the link structurally — provided the candidate predecessor
# exists in the prior plan AND is in a retry-warranting status (FAILED
# / CANCELLED). Pure deterministic structural inference, no LLM trust.


def _retry_steer_json(
    *,
    new_task_id: str,
    supersedes: str | None = None,
    title: str = "Retry it",
) -> str:
    """Build a steer-shaped LLM response that proposes a single retry task.

    ``supersedes`` of None ⇒ key omitted from JSON (LLM didn't populate
    the link). Pass an empty string to emit the key with empty value
    (same observable shape after JSON parse).
    """
    task: dict[str, Any] = {
        "id": new_task_id,
        "title": title,
        "description": "Retry of failed predecessor.",
        "assignee_agent_id": "writer",
    }
    if supersedes is not None:
        task["supersedes"] = supersedes
    return json.dumps(
        {
            "summary": "Retry the failure.",
            "tasks": [task],
            "edges": [],
        }
    )


def _plan_with_retry_predecessor(
    *,
    predecessor_id: str = "t0",
    predecessor_status: TaskStatus = TaskStatus.FAILED,
) -> Plan:
    """Prior plan with a single predecessor in a configurable status."""
    return Plan(
        id="plan-r213",
        run_id="run-r213",
        goal_ids=["g1"],
        tasks=[
            Task(
                id=predecessor_id,
                title="Original",
                status=predecessor_status,
                assignee_agent_id="writer",
            ),
        ],
        edges=[],
        revision_index=0,
    )


def _user_steer_drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please retry",
        current_task_id="t0",
    )


def _find_task(plan: Plan, task_id: str) -> Task:
    for t in plan.tasks:
        if t.id == task_id:
            return t
    raise AssertionError(f"task {task_id!r} not in plan")


async def test_backfill_supersedes_on_retry_name_when_old_failed() -> None:
    """Prior ``t0 FAILED`` + LLM emits ``retry_t0`` with no
    supersedes ⇒ backfill sets ``retry_t0.supersedes = "t0"``.
    """
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.FAILED)
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(_retry_steer_json(new_task_id="retry_t0", supersedes=None))
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    retry = _find_task(revised, "retry_t0")
    assert retry.supersedes == "t0", (
        f"expected backfill to set supersedes='t0', got {retry.supersedes!r}"
    )
    # Kind must also have been derived from FAILED status (REPLACE).
    from goldfive.types import SupersessionKind

    assert retry.supersedes_kind is SupersessionKind.REPLACE


async def test_backfill_supersedes_on_retry_name_when_old_cancelled() -> None:
    """CANCELLED predecessor also warrants backfill — same rationale
    as FAILED (the old work is conclusively closed without delivering)."""
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.CANCELLED)
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(_retry_steer_json(new_task_id="retry_t0", supersedes=None))
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    retry = _find_task(revised, "retry_t0")
    assert retry.supersedes == "t0"


async def test_no_backfill_when_old_completed() -> None:
    """Prior ``t0 COMPLETED`` + LLM emits ``retry_t0`` ⇒ supersedes
    stays empty (no spurious link over a successful predecessor).
    """
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.COMPLETED)
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(_retry_steer_json(new_task_id="retry_t0", supersedes=None))
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    retry = _find_task(revised, "retry_t0")
    assert retry.supersedes == "", (
        f"expected supersedes empty (COMPLETED predecessor), got {retry.supersedes!r}"
    )


async def test_no_backfill_when_old_absent() -> None:
    """LLM emits ``retry_unknown`` with no ``unknown`` task in the
    prior plan ⇒ supersedes stays empty.
    """
    plan = _plan_with_retry_predecessor(predecessor_id="t0")
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(
        _retry_steer_json(new_task_id="retry_unknown", supersedes=None)
    )
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    retry = _find_task(revised, "retry_unknown")
    assert retry.supersedes == ""


async def test_backfill_skips_already_populated() -> None:
    """LLM emits explicit ``supersedes='other'`` ⇒ backfill leaves
    it alone (LLM intent wins over structural inference).
    """
    # Add a second prior task ("other") so the explicit link resolves.
    plan = Plan(
        id="plan-r213",
        run_id="run-r213",
        goal_ids=["g1"],
        tasks=[
            Task(id="t0", title="t0", status=TaskStatus.FAILED),
            Task(id="other", title="other", status=TaskStatus.FAILED),
        ],
        edges=[],
        revision_index=0,
    )
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(
        _retry_steer_json(new_task_id="retry_t0", supersedes="other")
    )
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    retry = _find_task(revised, "retry_t0")
    assert retry.supersedes == "other", (
        "explicit LLM-supplied supersedes link must not be overridden"
    )


async def test_self_supersede_stripped() -> None:
    """LLM emits ``t0_v2`` with ``supersedes='t0_v2'`` (self-reference,
    the v27 rev-2 anomaly). Normalization clears it.

    NOTE: backfill ALSO clears self-references as defence in depth, so
    this test pins the self-reference cleanup regardless of which
    layer caught it.
    """
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.FAILED)
    goals = [Goal(id="g1", summary="ship it")]
    # The LLM emits a self-referencing supersedes on a NEW id (no
    # collision with prior). Backfill / normalize must strip the
    # self-reference. Versioned suffix backfill then runs on the
    # cleared link and points to ``t0`` (the predecessor).
    llm = _StubLLM(
        _retry_steer_json(new_task_id="t0_v2", supersedes="t0_v2")
    )
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    new_task = _find_task(revised, "t0_v2")
    # Self-reference cleared. Backfill ran AFTER (sup was empty), so
    # the structural inference points to ``t0`` (the FAILED prior).
    assert new_task.supersedes == "t0", (
        "self-reference must be cleared then backfill should infer "
        f"the predecessor; got supersedes={new_task.supersedes!r}"
    )


async def test_versioned_pattern_t0_v2_backfilled() -> None:
    """Prior ``t0 FAILED`` + LLM emits ``t0_v2`` (no retry_ prefix,
    just ``_v2`` suffix) ⇒ backfill sets ``supersedes='t0'``.
    """
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.FAILED)
    goals = [Goal(id="g1", summary="ship it")]
    llm = _StubLLM(_retry_steer_json(new_task_id="t0_v2", supersedes=None))
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    versioned = _find_task(revised, "t0_v2")
    assert versioned.supersedes == "t0"


async def test_backfill_skips_unrelated_task_names() -> None:
    """Tasks with names that don't match retry/version conventions
    are left alone — the inference is conservative."""
    plan = _plan_with_retry_predecessor(predecessor_status=TaskStatus.FAILED)
    goals = [Goal(id="g1", summary="ship it")]
    # New task id ``brand_new`` doesn't strip to any prior id.
    llm = _StubLLM(_retry_steer_json(new_task_id="brand_new", supersedes=None))
    planner = LLMPlanner(call_llm=llm, model="test-model")

    revised = await planner.refine(plan=plan, drift=_user_steer_drift(), goals=goals)
    assert revised is not None

    new_task = _find_task(revised, "brand_new")
    assert new_task.supersedes == ""
