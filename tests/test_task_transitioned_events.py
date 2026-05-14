"""Tests for goldfive#251 R4: ``TaskTransitioned`` proto event.

Every plan-state transition emits a typed ``TaskTransitioned`` sink
event with source attribution. The LLM-facing ``report_task_*``
surface is unchanged — handlers still return
``{"acknowledged": True}`` — so this is sink-only operator
observability.

Source-attribution vocabulary:

* ``llm_report``        — driven by an LLM ``report_task_*`` tool call
                           with an explicit ``task_id`` arg.
* ``handler_default``    — driven by an LLM ``report_task_*`` tool
                           call where ``task_id`` was resolved from
                           the adapter-stamped pin
                           (``goldfive.current_task_id``) rather than
                           an explicit arg.
* ``supersedes_reroute`` — driven by an LLM report on a superseded
                           task that the report-time pin classifier
                           (#266) routed to the REPLACE-kind successor.
* ``plan_revision``      — refine changed the task's status (the
                           ``_force_looper_failed`` callback in
                           ``LLMPlanner`` is the canonical example).
* ``cancellation``       — cooperative cancellation transitioned a
                           running task as a consequence (cascade-
                           cancel, control-message CANCEL routed
                           through ``executors._control``).
* ``other``              — any transition that doesn't fit a more
                           specific source label (default for un-
                           threaded callers).

Pairs with the ``TaskTransitionRefused`` proto event from #266 (pin
versioning) — refused attempts emit that envelope and DO NOT emit
``TaskTransitioned`` (no transition happened). The refused variant
was originally shipped as a dict envelope and later promoted to a
typed proto message (matches the InvocationCancelled promotion
pattern from #262).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.reporting import BUILTIN_REPORTING_TOOLS  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """Records every emitted event (proto + dict envelopes)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    """Minimal planner stub — never actually invoked in these tests."""

    async def generate(self, **kw: Any) -> Plan | None:
        return None

    async def refine(self, **kw: Any) -> Plan | None:
        return None


def _transition_events(sink: ListSink) -> list[Any]:
    """Return TaskTransitioned proto envelopes from the sink stream."""
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == "task_transitioned":
                out.append(evt)
        except Exception:
            continue
    return out


def _refused_events(sink: ListSink) -> list[Any]:
    """Return ``TaskTransitionRefused`` proto envelopes (#266 followup).

    Promoted from the dict shape #266 originally shipped to a typed
    proto message (matches the ``InvocationCancelled`` promotion from
    #262).
    """
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue
        try:
            if which("payload") == "task_transition_refused":
                out.append(evt)
        except Exception:
            continue
    return out


def _tool(name: str):
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing builtin tool {name!r}")


def _plan(*tasks: Task, edges: list[TaskEdge] | None = None, revision_index: int = 0) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=list(tasks),
        edges=list(edges or []),
        revision_index=revision_index,
    )


def _session(plan: Plan | None) -> Session:
    return Session(
        run_id="r-r4",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


def _bound_steerer(sink: ListSink) -> DefaultSteerer:
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    return steerer


# ---------------------------------------------------------------------------
# 1. report_task_started → TaskTransitioned(PENDING -> RUNNING, llm_report)
# ---------------------------------------------------------------------------


async def test_report_task_started_emits_llm_report_transition() -> None:
    plan = _plan(
        Task(id="t1", title="research", assignee_agent_id="researcher"),
        revision_index=0,
    )
    session = _session(plan)
    sink = ListSink()
    steerer = _bound_steerer(sink)

    result = await _tool("report_task_started").handler(
        {"task_id": "t1", "detail": "begin"}, session, steerer
    )

    # F1 directive ack on the real transition (acknowledged=True
    # remains the load-bearing field; the additional task + plan_state
    # keys carry the next-action anchor).
    assert result["acknowledged"] is True
    transitions = _transition_events(sink)
    assert len(transitions) == 1, "exactly one TaskTransitioned on report_task_started"
    payload = transitions[0].task_transitioned
    assert payload.task_id == "t1"
    assert payload.from_status == TaskStatus.PENDING.value
    assert payload.to_status == TaskStatus.RUNNING.value
    assert payload.source == "llm_report"
    assert payload.agent_name == "researcher"


# ---------------------------------------------------------------------------
# 2. report_task_completed → TaskTransitioned(RUNNING -> COMPLETED, llm_report)
# ---------------------------------------------------------------------------


async def test_report_task_completed_emits_llm_report_transition() -> None:
    plan = _plan(
        Task(
            id="t1",
            title="research",
            assignee_agent_id="researcher",
            status=TaskStatus.RUNNING,
        ),
        revision_index=0,
    )
    session = _session(plan)
    sink = ListSink()
    steerer = _bound_steerer(sink)

    result = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )

    # F1 directive ack on the real transition.
    assert result["acknowledged"] is True
    transitions = _transition_events(sink)
    assert len(transitions) == 1
    payload = transitions[0].task_transitioned
    assert payload.task_id == "t1"
    assert payload.from_status == TaskStatus.RUNNING.value
    assert payload.to_status == TaskStatus.COMPLETED.value
    assert payload.source == "llm_report"


# ---------------------------------------------------------------------------
# 3. REPLACE supersedes-reroute → TaskTransitioned on the new task,
#    source=supersedes_reroute.
# ---------------------------------------------------------------------------


async def test_replace_supersedes_reroute_emits_supersedes_reroute_source() -> None:
    plan = _plan(
        Task(
            id="research_solar",
            title="solar v1",
            assignee_agent_id="researcher",
            status=TaskStatus.FAILED,
        ),
        Task(
            id="research_solar_v2",
            title="solar v2",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="research_solar",
            supersedes_kind=SupersessionKind.REPLACE,
        ),
        revision_index=2,
    )
    session = _session(plan)
    # Stale pin under the old plan revision; report arrives quoting the
    # superseded task_id.
    session.state["goldfive.current_task_id"] = "research_solar"
    session.state["goldfive.current_task_revision"] = 1

    sink = ListSink()
    steerer = _bound_steerer(sink)

    result = await _tool("report_task_started").handler(
        {"task_id": "research_solar"}, session, steerer
    )

    # F1 directive ack on the rerouted transition.
    assert result["acknowledged"] is True
    transitions = _transition_events(sink)
    assert len(transitions) == 1, "exactly one TaskTransitioned for the rerouted call"
    payload = transitions[0].task_transitioned
    assert payload.task_id == "research_solar_v2", (
        "transition is recorded against the REPLACE successor, not the pin's task_id"
    )
    assert payload.from_status == TaskStatus.PENDING.value
    assert payload.to_status == TaskStatus.RUNNING.value
    assert payload.source == "supersedes_reroute"
    # No refusal — REPLACE-kind always routes.
    assert _refused_events(sink) == []


# ---------------------------------------------------------------------------
# 4. Refine that cancels an old task via REPLACE → TaskTransitioned with
#    source=plan_revision.
# ---------------------------------------------------------------------------


async def test_refine_replace_supersedes_emits_plan_revision_transition() -> None:
    """A revised plan that flips a task's status emits one
    ``TaskTransitioned(source=plan_revision)`` per status diff at the
    ``_emit_plan_revised`` site, alongside the proto ``PlanRevised``
    envelope. The transition envelopes follow the plan-flip envelope
    so a strictly-ordered consumer sees the cause-effect chain.
    """
    prev_plan = _plan(
        Task(
            id="t1",
            title="solar",
            assignee_agent_id="researcher",
            status=TaskStatus.RUNNING,
        ),
        revision_index=0,
    )
    revised = _plan(
        Task(
            id="t1",
            title="solar",
            assignee_agent_id="researcher",
            status=TaskStatus.CANCELLED,  # refine cancelled the old
        ),
        Task(
            id="t1_v2",
            title="solar v2",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="t1",
            supersedes_kind=SupersessionKind.REPLACE,
        ),
        revision_index=1,
    )
    session = _session(prev_plan)

    sink = ListSink()
    steerer = _bound_steerer(sink)

    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="solar tool failed",
        current_task_id="t1",
    )

    # Mirror the live order: _apply_revision installs ``revised`` on the
    # session, then _emit_plan_revised diffs prev_plan vs. revised and
    # emits the proto + per-transition envelopes.
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer.plans._apply_revision(session, revised, drift)
    await steerer.plans._emit_plan_revised(session, revised, drift, prev_plan=prev_plan)

    transitions = _transition_events(sink)
    # Exactly one transition: t1 RUNNING -> CANCELLED. The brand-new
    # PENDING task (t1_v2) doesn't emit a phantom "started in PENDING"
    # row.
    assert len(transitions) == 1
    payload = transitions[0].task_transitioned
    assert payload.task_id == "t1"
    assert payload.from_status == TaskStatus.RUNNING.value
    assert payload.to_status == TaskStatus.CANCELLED.value
    assert payload.source == "plan_revision"
    # Revision stamp matches the installed plan's revision_index.
    assert payload.revision_stamp == revised.revision_index


# ---------------------------------------------------------------------------
# 5. Cooperative cancellation transitioning a running task → source=cancellation.
# ---------------------------------------------------------------------------


async def test_cooperative_cancellation_emits_cancellation_source() -> None:
    """The steerer's ``mark_task_cancelled(... source="cancellation")``
    path emits a ``TaskTransitioned`` with the matching attribution.
    Stream C cancel callsites that drive a task transition pass this
    source explicitly; the cascade-cancel primitive defaults to
    ``"cancellation"`` for the same reason.
    """
    plan = _plan(
        Task(
            id="t1",
            title="research",
            assignee_agent_id="researcher",
            status=TaskStatus.RUNNING,
        ),
        Task(
            id="t2",
            title="downstream",
            assignee_agent_id="writer",
            status=TaskStatus.PENDING,
        ),
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    session = _session(plan)
    sink = ListSink()
    steerer = _bound_steerer(sink)

    await steerer.tasks.mark_task_cancelled(
        "t1",
        session=session,
        reason="adk_cancellation:invoc-1",
        source="cancellation",
    )

    transitions = _transition_events(sink)
    # One for the initiator (t1 RUNNING -> CANCELLED, source=cancellation).
    # One for the cascade-cancelled downstream (t2 PENDING -> CANCELLED,
    # source=cancellation).
    assert len(transitions) == 2
    by_task = {e.task_transitioned.task_id: e.task_transitioned for e in transitions}
    assert by_task["t1"].source == "cancellation"
    assert by_task["t1"].from_status == TaskStatus.RUNNING.value
    assert by_task["t1"].to_status == TaskStatus.CANCELLED.value
    assert by_task["t2"].source == "cancellation", (
        "cascade-cancelled downstream tasks share the cancellation source"
    )
    assert by_task["t2"].from_status == TaskStatus.PENDING.value
    assert by_task["t2"].to_status == TaskStatus.CANCELLED.value


# ---------------------------------------------------------------------------
# 6. Handler-default — pin-from-state, no explicit LLM arg → source=handler_default.
# ---------------------------------------------------------------------------


async def test_handler_default_emits_handler_default_source() -> None:
    plan = _plan(
        Task(id="t1", title="research", assignee_agent_id="researcher"),
        revision_index=0,
    )
    session = _session(plan)
    # Pin populated by the adapter at delegation time.
    session.state["goldfive.current_task_id"] = "t1"
    session.state["goldfive.current_task_revision"] = 0

    sink = ListSink()
    steerer = _bound_steerer(sink)

    # NO ``task_id`` in args — handler defaults from session.state.
    result = await _tool("report_task_started").handler(
        {"detail": "begin"}, session, steerer
    )

    # F1 directive ack on the pin-from-state transition.
    assert result["acknowledged"] is True
    transitions = _transition_events(sink)
    assert len(transitions) == 1
    payload = transitions[0].task_transitioned
    assert payload.task_id == "t1"
    assert payload.source == "handler_default", (
        "pin-from-state path attributes the transition to handler_default, "
        "not llm_report"
    )


# ---------------------------------------------------------------------------
# 7. revision_stamp matches plan.revision_index at transition time.
# ---------------------------------------------------------------------------


async def test_revision_stamp_matches_plan_revision_at_transition_time() -> None:
    plan = _plan(
        Task(id="t1", title="research", assignee_agent_id="researcher"),
        revision_index=7,  # arbitrary non-zero
    )
    session = _session(plan)
    sink = ListSink()
    steerer = _bound_steerer(sink)

    await _tool("report_task_started").handler(
        {"task_id": "t1"}, session, steerer
    )

    transitions = _transition_events(sink)
    assert len(transitions) == 1
    assert transitions[0].task_transitioned.revision_stamp == 7


# ---------------------------------------------------------------------------
# 8. invocation_id + agent_name resolve from the invocation context.
# ---------------------------------------------------------------------------


class _ReconcilerStub:
    """Minimal reconciler with the ``_invocation_agent`` map shape."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._invocation_agent = dict(mapping)


class _AdapterStub:
    """Adapter holding a plugin holding the reconciler — the chain
    :meth:`DefaultSteerer._resolve_invocation_id_for_agent` walks."""

    def __init__(self, reconciler: _ReconcilerStub) -> None:
        class _Plugin:
            pass

        plugin = _Plugin()
        plugin._reconciler = reconciler
        self._plugin = plugin


async def test_agent_name_and_invocation_id_stamped_from_context() -> None:
    plan = _plan(
        Task(id="t1", title="research", assignee_agent_id="researcher"),
        revision_index=0,
    )
    session = _session(plan)

    sink = ListSink()
    steerer = _bound_steerer(sink)
    # Wire a reconciler whose ``_invocation_agent`` maps an in-flight
    # invocation_id to the assignee. The steerer's
    # ``_resolve_invocation_id_for_agent`` walks plugin -> reconciler.
    steerer._adapter = _AdapterStub(  # type: ignore[assignment]
        _ReconcilerStub({"inv-42": "researcher"})
    )

    await _tool("report_task_started").handler(
        {"task_id": "t1"}, session, steerer
    )

    transitions = _transition_events(sink)
    assert len(transitions) == 1
    payload = transitions[0].task_transitioned
    assert payload.agent_name == "researcher", (
        "agent_name is stamped from task.assignee_agent_id"
    )
    assert payload.invocation_id == "inv-42", (
        "invocation_id resolves through adapter -> plugin -> reconciler "
        "_invocation_agent map"
    )


# ---------------------------------------------------------------------------
# 9. Refused transitions (#266) DO NOT emit TaskTransitioned.
# ---------------------------------------------------------------------------


async def test_refused_stale_correct_pin_does_not_emit_task_transitioned() -> None:
    """A refused transition (stale pin under a CORRECT-kind successor)
    emits the ``TaskTransitionRefused`` proto event from #266 and
    DOES NOT emit a typed ``TaskTransitioned``: no transition
    happened, so there is nothing to attribute.
    """
    plan = _plan(
        Task(
            id="research_solar",
            title="solar (history)",
            assignee_agent_id="researcher",
            status=TaskStatus.COMPLETED,
        ),
        Task(
            id="research_solar_corrected",
            title="solar with correction",
            assignee_agent_id="researcher",
            status=TaskStatus.PENDING,
            supersedes="research_solar",
            supersedes_kind=SupersessionKind.CORRECT,
        ),
        revision_index=2,
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "research_solar"
    session.state["goldfive.current_task_revision"] = 1  # stale

    sink = ListSink()
    steerer = _bound_steerer(sink)

    result = await _tool("report_task_completed").handler(
        {"task_id": "research_solar", "summary": "x"}, session, steerer
    )

    # LLM still sees an ack — no prompt-injection surface (#266).
    assert result == {"acknowledged": True}
    # Refused proto event from #266 (typed) is on the wire …
    refused = _refused_events(sink)
    assert len(refused) == 1
    assert refused[0].task_transition_refused.reason == "stale_pin_correct_supersedes"
    # … but NO TaskTransitioned was emitted: no transition happened.
    assert _transition_events(sink) == [], (
        "refused attempts must not emit TaskTransitioned"
    )


async def test_refused_stale_ambiguous_pin_does_not_emit_task_transitioned() -> None:
    """Same invariant for the no-supersedes / ambiguous refusal."""
    plan = _plan(
        Task(
            id="orphan",
            title="orphan",
            assignee_agent_id="researcher",
            status=TaskStatus.RUNNING,
        ),
        revision_index=3,
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "orphan"
    session.state["goldfive.current_task_revision"] = 1  # stale, no successor

    sink = ListSink()
    steerer = _bound_steerer(sink)

    await _tool("report_task_completed").handler(
        {"task_id": "orphan", "summary": "x"}, session, steerer
    )

    refused = _refused_events(sink)
    assert len(refused) == 1
    assert refused[0].task_transition_refused.reason == "stale_pin_no_supersedes"
    assert _transition_events(sink) == []


# ---------------------------------------------------------------------------
# 10. TaskTransitionRefused proto round-trip (post-promotion).
# ---------------------------------------------------------------------------


def test_task_transition_refused_proto_round_trip() -> None:
    """The factory + proto descriptors round-trip every populated field.

    Locks the wire shape after the dict→proto promotion: every field
    survives ``SerializeToString`` / ``ParseFromString`` and the
    envelope's payload oneof routes to the new variant.
    """
    from goldfive.events import task_transition_refused_event
    from goldfive.pb.goldfive.v1 import events_pb2

    evt = task_transition_refused_event(
        run_id="r-rt",
        sequence=7,
        task_id="research_solar",
        attempted_from=TaskStatus.COMPLETED.value,
        attempted_to=TaskStatus.RUNNING.value,
        reason="stale_pin_correct_supersedes",
        pin_revision=1,
        current_revision=2,
        agent_name="researcher",
        invocation_id="inv-9",
        session_id="sess-1",
    )

    raw = evt.SerializeToString()
    parsed = events_pb2.Event()
    parsed.ParseFromString(raw)

    # Envelope fields preserved.
    assert parsed.run_id == "r-rt"
    assert parsed.sequence == 7
    assert parsed.session_id == "sess-1"
    assert parsed.WhichOneof("payload") == "task_transition_refused"

    # Payload fields preserved verbatim.
    p = parsed.task_transition_refused
    assert p.task_id == "research_solar"
    assert p.attempted_from == TaskStatus.COMPLETED.value
    assert p.attempted_to == TaskStatus.RUNNING.value
    assert p.reason == "stale_pin_correct_supersedes"
    assert p.pin_revision == 1
    assert p.current_revision == 2
    assert p.agent_name == "researcher"
    assert p.invocation_id == "inv-9"


def test_task_transition_refused_factory_defaults() -> None:
    """Optional kwargs default to empty / zero without raising.

    Defends the partial-population path used by emit sites that don't
    have invocation_id / agent_name attribution available.
    """
    from goldfive.events import task_transition_refused_event

    evt = task_transition_refused_event(
        run_id="r-defaults",
        sequence=0,
        task_id="t1",
        attempted_from=TaskStatus.PENDING.value,
        attempted_to=TaskStatus.RUNNING.value,
        reason="stale_pin_no_supersedes",
    )

    assert evt.WhichOneof("payload") == "task_transition_refused"
    p = evt.task_transition_refused
    assert p.task_id == "t1"
    assert p.reason == "stale_pin_no_supersedes"
    assert p.pin_revision == 0
    assert p.current_revision == 0
    assert p.agent_name == ""
    assert p.invocation_id == ""


# ---------------------------------------------------------------------------
# 11. F10 / executor_dispatch — ParallelDAGExecutor's framework auto-start
#     emits TaskTransitioned(source="executor_dispatch") at the
#     ``task.status = TaskStatus.RUNNING`` mutation point in
#     :class:`~goldfive.executors.ParallelDAGExecutor`.
# ---------------------------------------------------------------------------


async def test_parallel_executor_dispatch_emits_executor_dispatch_transition() -> None:
    """When the parallel DAG executor flips a task from PENDING to
    RUNNING just before invoking the adapter, a TaskTransitioned must
    land with ``source="executor_dispatch"`` — distinct from
    ``handler_default`` (LLM tool call where ``task_id`` defaulted)
    and from generic ``other``.
    """
    from goldfive.executors import ParallelDAGExecutor
    from goldfive.results import InvocationResult

    plan = _plan(
        Task(id="t1", title="T1", assignee_agent_id="writer"),
        revision_index=0,
    )
    session = _session(plan)
    session.run_id = "r-pdag"
    sink = ListSink()
    steerer = _bound_steerer(sink)

    class _Adapter:
        async def register_reporting_tools(self, tools: list[Any]) -> None:
            return None

        @property
        def available_agents(self) -> list[str]:
            return ["writer"]

        async def invoke(self, task: Task, session: Session) -> InvocationResult:
            return InvocationResult(task_id=task.id, text=f"ok {task.id}")

    executor = ParallelDAGExecutor(max_concurrency=0)
    out = await executor.run(
        plan=plan,
        session=session,
        adapter=_Adapter(),
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )
    assert out.success

    transitions = _transition_events(sink)
    # The framework auto-start row is the first PENDING -> RUNNING for t1.
    matching = [
        e
        for e in transitions
        if e.task_transitioned.source == "executor_dispatch"
        and e.task_transitioned.task_id == "t1"
        and e.task_transitioned.from_status == TaskStatus.PENDING.value
        and e.task_transitioned.to_status == TaskStatus.RUNNING.value
    ]
    assert len(matching) == 1, (
        "ParallelDAGExecutor.run_one must emit exactly one "
        f"TaskTransitioned(source=executor_dispatch) for t1; "
        f"saw {len(matching)} (all transitions: "
        f"{[e.task_transitioned.source for e in transitions]!r})"
    )


# ---------------------------------------------------------------------------
# 12. F10 / control_rewind — REWIND_TO control message emits one
#     TaskTransitioned per affected task with source="control_rewind".
# ---------------------------------------------------------------------------


async def test_control_rewind_emits_control_rewind_transition() -> None:
    """A ``REWIND_TO`` control message marks the target task and every
    downstream task PENDING. F10: each affected task whose status
    actually changed emits one ``TaskTransitioned`` with
    ``source="control_rewind"`` so operator triage can distinguish a
    control-driven rewind from a cancellation cascade or a
    plan_revision flip.
    """
    from goldfive.control import ControlKind, ControlMessage
    from goldfive.executors._control import dispatch_control

    plan = _plan(
        Task(
            id="t0",
            title="T0",
            assignee_agent_id="writer",
            status=TaskStatus.COMPLETED,
        ),
        Task(
            id="t1",
            title="T1",
            assignee_agent_id="writer",
            status=TaskStatus.COMPLETED,
        ),
        Task(
            id="t2",
            title="T2",
            assignee_agent_id="writer",
            status=TaskStatus.COMPLETED,
        ),
        edges=[
            TaskEdge(from_task_id="t0", to_task_id="t1"),
            TaskEdge(from_task_id="t1", to_task_id="t2"),
        ],
        revision_index=0,
    )
    session = _session(plan)
    session.run_id = "r-rewind"
    session.completed_results = {"t0": "a", "t1": "b", "t2": "c"}

    sink = ListSink()
    steerer = _bound_steerer(sink)

    msg = ControlMessage(kind=ControlKind.REWIND_TO, payload={"task_id": "t1"})
    outcome = await dispatch_control(
        msg, session=session, steerer=steerer, sinks=[sink]
    )

    # Sanity: rewind succeeded. goldfive#247: read from session.plan
    # since the local ``plan`` reference is the pre-rewind snapshot.
    assert outcome.rewind_task_id == "t1"
    assert session.plan is not None
    assert session.plan.tasks[0].status == TaskStatus.COMPLETED  # t0 unchanged
    assert session.plan.tasks[1].status == TaskStatus.PENDING
    assert session.plan.tasks[2].status == TaskStatus.PENDING

    transitions = _transition_events(sink)
    rewind_transitions = [
        e for e in transitions if e.task_transitioned.source == "control_rewind"
    ]
    # Two affected tasks (t1 + t2 — both went COMPLETED -> PENDING).
    assert len(rewind_transitions) == 2, (
        f"expected one TaskTransitioned(source=control_rewind) per "
        f"affected task; saw {len(rewind_transitions)}"
    )
    rewind_ids = sorted(e.task_transitioned.task_id for e in rewind_transitions)
    assert rewind_ids == ["t1", "t2"]
    for evt in rewind_transitions:
        payload = evt.task_transitioned
        assert payload.from_status == TaskStatus.COMPLETED.value
        assert payload.to_status == TaskStatus.PENDING.value
