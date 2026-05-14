"""Channel-routing tests for goldfive-authored drift (Phase 2 / #246).

Phase 2 of the path-duality fix routes goldfive-authored drift through
the same ``ControlMessage`` cancel-and-restart junction as user-authored
``STEER``. These tests pin the new contract:

* Goldfive-authored CRITICAL OFF_TOPIC dispatches a ``GOLDFIVE_STEER``
  ControlMessage on the bound channel; the executor's invoke loop
  cancels in-flight work and restarts the passthrough with a
  ``[GOLDFIVE STEERING CONTROL …]`` framed corrective.
* Goldfive-authored CRITICAL GOAL_DRIFT (Level 4 PAUSE_ESCALATE)
  dispatches a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage; the
  executor's invoke loop returns a ``goldfive_pause`` outcome so the
  pre-task loop blocks for operator intervention.
* The corrective restart message carries the
  ``[GOLDFIVE STEERING CONTROL`` header so plugins / sinks can
  distinguish goldfive-authored from user-authored steers.
* Multiple goldfive-authored drifts in flight — each fans out one
  ControlMessage; the channel preserves order, the executor processes
  them one at a time (the first triggers cancel-and-restart; later
  arrivals queue on the channel and are drained on the next pre-task
  loop iteration).
* User USER_STEER still works exactly as before — the user-STEER
  branch in the executor's invoke loop is unaffected by the new
  goldfive-internal kinds.
* The deleted ``Session.pending_corrective_message`` and
  ``Session.paused_for_human_intervention`` fields are gone — a
  ``hasattr`` check pins the deletion so a future re-introduction is
  caught by the test harness.
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

from goldfive.control import (  # noqa: E402
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
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


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover
        return


class _RevisedPlanPlanner:
    """Planner that always returns the same revised plan for refine_steer / refine."""

    def __init__(self, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, **_kw: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised


def _make_plan(*, replacement_id: str = "t2b") -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="T2", status=TaskStatus.RUNNING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_revised_plan(*, replacement_id: str = "t2b") -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.COMPLETED),
            Task(id=replacement_id, title="Replacement", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id=replacement_id)],
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship it")],
        plan=_make_plan(),
        current_task_id="t2",
    )


def _bound_steerer(
    *,
    revised: Plan | None = None,
    threshold: str = "warning",
) -> tuple[DefaultSteerer, _RevisedPlanPlanner, ControlChannel, _ListSink]:
    steerer = DefaultSteerer(
        goldfive_steer_threshold=threshold,
        goldfive_steer_suppression_window_turns=3,
    )
    planner = _RevisedPlanPlanner(revised=revised or _make_revised_plan())
    sink = _ListSink()
    channel = ControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)
    return steerer, planner, channel, sink


def _drain_channel(channel: ControlChannel) -> list[ControlMessage]:
    drained: list[ControlMessage] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        drained.append(inbox.get_nowait())
    return drained


# ---------------------------------------------------------------------------
# Test 1: goldfive OFF_TOPIC → GOLDFIVE_STEER on the channel
# ---------------------------------------------------------------------------


async def test_goldfive_off_topic_drift_dispatches_goldfive_steer_control() -> None:
    """A goldfive-authored OFF_TOPIC WARNING drift clears the
    promotion threshold (threshold=warning) → ``_promote_drift_to_steer``
    runs → dispatches a ``GOLDFIVE_STEER`` ControlMessage on the bound
    channel."""
    steerer, planner, channel, _sink = _bound_steerer()
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered off into raccoons",
        current_task_id="t2",
        authored_by="goldfive",
    )
    session = _make_session()
    await steerer._handle_drift(drift, session)

    # Refine ran (the promotion path called planner.refine_steer).
    assert planner.refine_steer_calls, "refine_steer should have fired"
    # A GOLDFIVE_STEER control message landed on the channel.
    msgs = [m for m in _drain_channel(channel) if m.kind is ControlKind.GOLDFIVE_STEER]
    assert len(msgs) == 1, f"expected 1 GOLDFIVE_STEER, got {[m.kind for m in msgs]!r}"
    payload = msgs[0].payload
    assert payload["drift_kind"] == DriftKind.OFF_TOPIC.value
    # Body should be non-empty (composed from drift detail).
    assert payload["body"], "GOLDFIVE_STEER payload must carry a non-empty body"
    assert "raccoons" in payload["body"]
    # Superseded task ids carry the originating task.
    assert payload["superseded_task_ids"] == ["t2"]


# ---------------------------------------------------------------------------
# Test 2: goldfive GOAL_DRIFT CRITICAL → GOLDFIVE_PAUSE_ESCALATE
# ---------------------------------------------------------------------------


async def test_goldfive_intent_divergence_critical_dispatches_pause_escalate() -> None:
    """A goldfive-authored INTENT_DIVERGENCE CRITICAL drift routes
    through Level 4 PAUSE_ESCALATE → dispatches a
    ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage. A
    HUMAN_INTERVENTION_REQUIRED drift is emitted to the sink stream.

    Per ``DefaultSteerer._LADDER`` INTENT_DIVERGENCE CRITICAL routes
    to PAUSE_ESCALATE at the first occurrence (no need for a repeat
    threshold trip — the judge's signal is terminal). REFINE_VALIDATION_FAILED
    CRITICAL is similarly Level-4 first-occurrence; GOAL_DRIFT
    CRITICAL routes to NUDGE first (per F4) so we use INTENT_DIVERGENCE
    here for a cleaner first-occurrence assertion."""
    # Disable promotion so the legacy PAUSE_ESCALATE ladder fires on
    # the goldfive-authored drift directly.
    steerer, planner, channel, sink = _bound_steerer(threshold="off")
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="agent abandoned the bound task entirely",
        current_task_id="t2",
        authored_by="goldfive",
    )
    session = _make_session()
    await steerer._handle_drift(drift, session)

    # Level 4 does NOT call refine.
    assert planner.refine_calls == []
    assert planner.refine_steer_calls == []
    # A GOLDFIVE_PAUSE_ESCALATE control message landed on the channel.
    pause_msgs = [
        m
        for m in _drain_channel(channel)
        if m.kind is ControlKind.GOLDFIVE_PAUSE_ESCALATE
    ]
    assert len(pause_msgs) == 1
    assert (
        pause_msgs[0].payload["drift_kind"] == DriftKind.INTENT_DIVERGENCE.value
    )
    assert pause_msgs[0].payload["reason"], "reason must be non-empty"

    # HUMAN_INTERVENTION_REQUIRED drift on the sink stream.
    from goldfive.pb.goldfive.v1 import types_pb2

    assert any(
        e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_HUMAN_INTERVENTION_REQUIRED
        for e in sink.events
    )


# ---------------------------------------------------------------------------
# Test 3: corrective restart message carries the GOLDFIVE STEERING CONTROL header
# ---------------------------------------------------------------------------


async def test_goldfive_steer_restart_message_has_goldfive_header() -> None:
    """When the executor's invoke loop receives a ``GOLDFIVE_STEER``
    ControlMessage, it composes the restart user input via
    :meth:`SequentialExecutor._compose_steer_restart_message` with
    ``source="goldfive"``. The result MUST start with the
    ``[GOLDFIVE STEERING CONTROL`` header so plugins / sinks can
    distinguish goldfive-authored from user-authored steers."""
    msg = ControlMessage(
        kind=ControlKind.GOLDFIVE_STEER,
        payload={
            "drift_kind": DriftKind.OFF_TOPIC.value,
            "drift_id": "drift-abc",
            "body": "Discard prior raccoon work and proceed with the tomato fix.",
            "superseded_task_ids": ["t_research"],
            "replacement_task_ids": ["t_fix_research"],
        },
    )
    text = SequentialExecutor._compose_steer_restart_message(
        msg,
        fallback="",
        source="goldfive",
        superseded_task_ids=["t_research"],
        replacement_task_ids=["t_fix_research"],
    )
    assert text.startswith("[GOLDFIVE STEERING CONTROL")
    assert "Discard prior raccoon work" in text
    # Superseded / replacement blocks rendered.
    assert "t_research" in text
    assert "t_fix_research" in text
    # And it MUST NOT carry the user-steer header (so plugins can
    # discriminate).
    assert "[USER STEERING CONTROL" not in text


# ---------------------------------------------------------------------------
# Test 4: multiple goldfive drifts queue on the channel in order
# ---------------------------------------------------------------------------


async def test_multiple_goldfive_drifts_queue_in_order_on_channel() -> None:
    """Two back-to-back goldfive-authored drifts each dispatch their
    own ``GOLDFIVE_STEER`` ControlMessage. The channel preserves
    insertion order; the executor's invoke loop drains them one at a
    time (the first triggers cancel-and-restart; the second waits in
    the inbox for the next pre-task drain). This pins behaviour
    against the alternative "coalesce / dedupe at dispatch time" — we
    rely on the channel to serialise.

    Audit #402: dispatch fires AFTER ``_emit_plan_revised``, so the
    second drift must produce a STRUCTURALLY DIFFERENT revised plan
    or the no-op-revision short-circuit (goldfive#271) will fire
    before dispatch. We give each refine a unique replacement task
    id so the second revision is genuinely distinct from the first.
    """
    # Custom planner that returns a fresh revised plan per refine call,
    # each with a distinct replacement task id so the no-op-revision
    # short-circuit (#271) does NOT fire between drifts.
    class _DistinctRevisedPlanner:
        def __init__(self) -> None:
            self.refine_steer_calls: list[dict[str, Any]] = []

        async def generate(self, **_kw: Any) -> Plan | None:  # pragma: no cover
            return None

        async def refine(self, **kwargs: Any) -> Plan | None:
            return await self.refine_steer(**kwargs)

        async def refine_steer(self, **kwargs: Any) -> Plan | None:
            idx = len(self.refine_steer_calls) + 1
            self.refine_steer_calls.append(kwargs)
            return _make_revised_plan(replacement_id=f"t2_rev{idx}")

    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        goldfive_steer_suppression_window_turns=3,
    )
    planner = _DistinctRevisedPlanner()
    sink = _ListSink()
    channel = ControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)

    session = _make_session()
    drifts = [
        DriftEvent(
            kind=DriftKind.OFF_TOPIC,
            severity=DriftSeverity.WARNING,
            detail=f"wander #{i}",
            current_task_id=f"t{i}",
            authored_by="goldfive",
        )
        for i in (1, 2)
    ]
    for d in drifts:
        await steerer._handle_drift(d, session)

    msgs = [m for m in _drain_channel(channel) if m.kind is ControlKind.GOLDFIVE_STEER]
    assert len(msgs) == 2, f"expected 2 GOLDFIVE_STEER, got {len(msgs)}"
    # Order preserved.
    assert "wander #1" in msgs[0].payload["body"]
    assert "wander #2" in msgs[1].payload["body"]


# ---------------------------------------------------------------------------
# Test 5: USER_STEER path is unaffected (no regression)
# ---------------------------------------------------------------------------


async def test_user_steer_path_unaffected_by_goldfive_routing() -> None:
    """A user-issued ``STEER`` ControlMessage still produces a
    ``steer_message=msg`` outcome from ``dispatch_control`` — the
    Phase 2 changes added new branches but did NOT alter the existing
    USER_STEER dispatch."""
    from goldfive.executors._control import dispatch_control

    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "pivot to slide 3", "suggested_action": "narrow scope"},
    )

    class _NullSteerer:
        async def observe(self, *args: Any, **kwargs: Any) -> None:
            return

    session = _make_session()
    outcome = await dispatch_control(
        msg, session=session, steerer=_NullSteerer(), sinks=[]
    )
    assert outcome.steer_message is msg
    # Goldfive-side outcome fields are clean.
    assert outcome.goldfive_steer_message is None
    assert outcome.goldfive_pause_message is None
    # The ack reports success with the steer-queued detail.
    assert outcome.ack.detail == "steer queued"


# ---------------------------------------------------------------------------
# Test 6: deleted dead-state fields are gone
# ---------------------------------------------------------------------------


def test_deleted_session_fields_are_gone() -> None:
    """``Session.pending_corrective_message`` and
    ``Session.paused_for_human_intervention`` were removed in Phase 2
    of the path-duality fix. A future re-introduction (typo or revert)
    is caught here."""
    session = Session(run_id="r1")
    assert not hasattr(session, "pending_corrective_message"), (
        "pending_corrective_message must remain deleted (Phase 2 of #246)"
    )
    assert not hasattr(session, "paused_for_human_intervention"), (
        "paused_for_human_intervention must remain deleted (Phase 2 of #246)"
    )


def test_deleted_fields_have_no_residue_in_goldfive_package() -> None:
    """No goldfive/* source file may reference the deleted fields as
    attribute access in actual code (i.e. AST-level reads or writes).

    Walks every Python module under ``goldfive/`` and parses it with
    :mod:`ast`. Any ``ast.Attribute`` node whose ``attr`` matches one
    of the deleted field names is reported. Docstrings and comments
    are fine — those are explanatory text, not live attribute access,
    and the AST never visits them as ``Attribute`` nodes."""
    import ast
    import pathlib

    deleted_names = {
        "paused_for_human_intervention",
        "pending_corrective_message",
    }
    pkg_root = pathlib.Path(__file__).resolve().parent.parent / "goldfive"
    offenders: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self, path: pathlib.Path) -> None:
            self.path = path

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in deleted_names:
                offenders.append(
                    f"{self.path.relative_to(pkg_root)}:{node.lineno}: "
                    f".{node.attr} (live attribute access)"
                )
            self.generic_visit(node)

    for path in pkg_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:  # pragma: no cover — protects against bad files
            offenders.append(f"{path.relative_to(pkg_root)}: parse error {exc}")
            continue
        _Visitor(path).visit(tree)

    assert not offenders, (
        "Phase 2 (#246) deletion must not have live attribute access:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Test 7: executor invoke loop honours GOLDFIVE_STEER mid-invocation
# ---------------------------------------------------------------------------


async def test_executor_invoke_loop_handles_goldfive_steer() -> None:
    """A ``GOLDFIVE_STEER`` ControlMessage on the channel mid-invocation
    cancels the in-flight invoke task and returns
    ``("goldfive_steer", msg)`` so the overlay loop can compose the
    framed restart and re-invoke."""

    channel = ControlChannel()
    body = "Discard prior contaminated research and start fix_research_tomatoes."
    await channel.send(
        ControlMessage(
            kind=ControlKind.GOLDFIVE_STEER,
            payload={
                "drift_kind": DriftKind.OFF_TOPIC.value,
                "drift_id": "d-1",
                "body": body,
                "superseded_task_ids": ["t_research"],
                "replacement_task_ids": ["t_fix_research"],
            },
        )
    )

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _StubAdapter:
        async def invoke_passthrough(
            self, user_input: str, *, session: Any, reconciler: Any
        ) -> Any:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class _StubSteerer:
        pass

    class _StubReconciler:
        def reset_for_new_plan(self, plan: Any) -> None:
            pass

    session = _make_session()
    executor = SequentialExecutor()
    kind, payload = await executor._invoke_passthrough_with_control(
        adapter=_StubAdapter(),
        session=session,
        steerer=_StubSteerer(),
        sinks=[],
        control=channel,
        reconciler=_StubReconciler(),
        user_input="go",
    )
    assert kind == "goldfive_steer"
    assert isinstance(payload, ControlMessage)
    assert payload.kind is ControlKind.GOLDFIVE_STEER
    assert payload.payload["body"] == body
    # The in-flight adapter call was actually cancelled — it must have
    # observed CancelledError before we returned.
    assert cancelled.is_set()


# ---------------------------------------------------------------------------
# Test 8: best-effort dispatch when no channel is bound
# ---------------------------------------------------------------------------


async def test_no_channel_bound_dispatch_is_noop_not_raise() -> None:
    """When the steerer has no channel bound, ``GOLDFIVE_STEER`` /
    ``GOLDFIVE_PAUSE_ESCALATE`` dispatches are best-effort no-ops.
    The originating drift event on the sink stream remains the
    durable signal — the steerer must not raise from the dispatch
    helper."""
    steerer = DefaultSteerer(goldfive_steer_threshold="warning")
    sink = _ListSink()
    planner = _RevisedPlanPlanner(revised=_make_revised_plan())
    steerer.bind(sinks=[sink], planner=planner)
    # Deliberately do NOT call bind_control_channel.

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="wander",
        current_task_id="t2",
        authored_by="goldfive",
    )
    session = _make_session()
    # Must not raise.
    await steerer._handle_drift(drift, session)
    # Refine still ran — the dispatch failure does not block the
    # promotion path.
    assert planner.refine_steer_calls, (
        "refine_steer should still fire when no channel is bound"
    )
