"""Observational delegation-time task pinning (goldfive#259).

#252 zeroed ``Task.assignee_agent_id`` at plan-parse time so the LLM
cannot pre-declare which sub-agent will pick up a task. The follow-up
in #259 wires the observational re-population: at ``delegation_observed``
time the plugin walks the plan, picks the eligible PENDING task this
delegation is enacting, stamps ``task.assignee_agent_id`` with the
invoked agent's name and pins ``session.current_task_id`` so the
reporting-tool pin lookup in :func:`_resolve_pinned_task_id` resolves
on the delegated sub-invocation's tool calls.

This file pins the selection algorithm and the end-to-end reporting-
tool resolution behaviour:

  * Linear plan A -> B -> C; first delegation binds A and pins it.
  * Sequential delegations: after A completes, the next delegation
    binds B.
  * Multi-eligible (two parallel PENDING tasks): topic-match in tool
    args picks the matching task.
  * Multi-eligible without topic-match in args: first by plan order
    wins (no guessing).
  * No eligible task (all PENDING tasks have non-COMPLETED predecessors)
    -> no pin, DEBUG log, no exception.
  * Integration: after the pin, ``before_tool_callback`` resolves
    ``report_task_started`` to the bound task (not the silent-ack
    no-op).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.orchestration_store import OrchestrationStore  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
    )


def _session_with(plan: Plan) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx(session: Session, host_agent_name: str = "coord") -> SessionContext:
    return SessionContext(
        session=session,
        steerer=None,
        task=None,
        tool_handlers={},
        host_agent_name=host_agent_name,
    )


def _find_task(plan: Plan, task_id: str) -> Task | None:
    for t in plan.tasks:
        if t.id == task_id:
            return t
    return None


# ---------------------------------------------------------------------------
# Selection algorithm
# ---------------------------------------------------------------------------


def test_linear_plan_first_delegation_binds_first_task() -> None:
    """3-task linear plan A->B->C, all PENDING, no pre-declared assignees.

    Delegation to agent X picks task A (only DAG-ready PENDING task),
    stamps assignee, pins current_task_id. The other two tasks remain
    untouched.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        Task(id="C", title="Review the draft"),
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="B", to_task_id="C"),
        ],
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={"request": "go"},
    )

    a = _find_task(session.plan, "A")
    b = _find_task(session.plan, "B")
    c = _find_task(session.plan, "C")
    assert a is not None and b is not None and c is not None
    assert a.assignee_agent_id == "X"
    assert b.assignee_agent_id == ""
    assert c.assignee_agent_id == ""
    assert session.current_task_id == "A"
    store = OrchestrationStore.for_session(session)
    assert store.pin_current_task() == "A"


def test_sequential_delegations_bind_next_task_after_completion() -> None:
    """After A completes, delegating to a fresh agent binds B (the new
    DAG-ready PENDING task)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic", status=TaskStatus.COMPLETED),
        Task(id="B", title="Write the draft"),
        Task(id="C", title="Review the draft"),
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="B", to_task_id="C"),
        ],
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="Y",
        tool_args={"request": "draft"},
    )

    b = _find_task(session.plan, "B")
    assert b is not None
    assert b.assignee_agent_id == "Y"
    assert session.current_task_id == "B"


def test_multi_eligible_topic_match_wins() -> None:
    """Two parallel PENDING tasks (no edge between them) and tool args
    contain a token that overlaps with one task's title — the matching
    task wins."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(
            id="t1",
            title="solar telemetry research",
            description="gather solar telemetry",
        ),
        Task(
            id="t2",
            title="quarterly invoice review",
            description="reconcile quarterly invoices",
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="researcher",
        tool_args={"topic": "please research solar telemetry"},
    )

    t1 = _find_task(session.plan, "t1")
    t2 = _find_task(session.plan, "t2")
    assert t1 is not None and t2 is not None
    assert t1.assignee_agent_id == "researcher"
    assert t2.assignee_agent_id == ""
    assert session.current_task_id == "t1"


def test_multi_eligible_no_topic_match_falls_back_to_first() -> None:
    """Two parallel PENDING tasks, tool args have no overlapping tokens
    with either title -> fall back to the first task by plan order."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="t1", title="solar telemetry research"),
        Task(id="t2", title="quarterly invoice review"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    # Tool args contain only sub-4-char tokens (the scorer filter
    # threshold) so no candidate scores > 0 and the fallback kicks in.
    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="researcher",
        tool_args={"x": "go on"},
    )

    t1 = _find_task(session.plan, "t1")
    t2 = _find_task(session.plan, "t2")
    assert t1 is not None and t2 is not None
    assert t1.assignee_agent_id == "researcher"
    assert t2.assignee_agent_id == ""
    assert session.current_task_id == "t1"


def test_no_eligible_task_leaves_session_unpinned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All PENDING tasks have non-COMPLETED predecessors -> no pin, no
    exception, DEBUG log fires."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        # A is PENDING (not COMPLETED) so B's predecessor isn't satisfied.
        Task(id="A", title="Plan the work"),
        Task(id="B", title="Execute the work"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    # Mark A non-PENDING (e.g. RUNNING) so A is not eligible either
    # (only PENDING tasks count) — that matches the brief's "no
    # eligible task" case.
    import dataclasses

    plan = dataclasses.replace(
        plan,
        tasks=(
            dataclasses.replace(plan.tasks[0], status=TaskStatus.RUNNING),
            plan.tasks[1],
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    with caplog.at_level(logging.DEBUG, logger="goldfive.adapters.adk"):
        plugin._maybe_pin_delegation_task(
            ctx=ctx,
            invoked_agent_name="Z",
            tool_args={"request": "go"},
        )

    a = _find_task(session.plan, "A")
    b = _find_task(session.plan, "B")
    assert a is not None and b is not None
    assert a.assignee_agent_id == ""
    assert b.assignee_agent_id == ""
    assert session.current_task_id == ""
    assert any(
        "no eligible PENDING task" in r.getMessage() for r in caplog.records
    ), f"expected DEBUG log; got {[r.getMessage() for r in caplog.records]}"


def test_idempotent_on_already_assigned_task() -> None:
    """Re-running the pin with the same agent on the same eligible task
    is a no-op (assignee already matches, no spurious plan swap)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={},
    )
    first_plan = session.plan
    assert first_plan is not None
    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={},
    )
    # Second call kept the same plan pointer (idempotent).
    assert session.plan is first_plan
    a = _find_task(session.plan, "A")
    assert a is not None and a.assignee_agent_id == "X"


# ---------------------------------------------------------------------------
# Integration with the reporting-tool pin lookup
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgentTool:
    """ADK AgentTool stand-in: has both ``.agent`` and ``.name``."""

    def __init__(self, agent_name: str) -> None:
        self.agent = _FakeAgent(agent_name)
        self.name = agent_name


class _FakeFunctionTool:
    """Stand-in for a reporting tool (FunctionTool with a ``func``)."""

    def __init__(self, name: str) -> None:
        self.name = name

        def _func() -> None:
            return None

        _func.__name__ = name
        self.func = _func


class _FakeInvocationContext:
    def __init__(self, session_state: dict, agent_name: str) -> None:
        class _ADKSession:
            def __init__(self, state: dict) -> None:
                self.state = state

        self.session = _ADKSession(session_state)
        self.invocation_id = "inv-1"
        self.agent = _FakeAgent(agent_name)


class _FakeToolContext:
    def __init__(self, inv_ctx: Any, function_call_id: str = "fc-1") -> None:
        self._invocation_context = inv_ctx
        self.function_call_id = function_call_id


async def test_after_pin_report_task_started_resolves_bound_task() -> None:
    """End-to-end: drive ``before_tool_callback`` through an AgentTool
    dispatch (which triggers the pin) followed by a ``report_task_started``
    call from the sub-agent. The reporting-tool lookup must find the
    bound task and short-circuit through the reporting handler — not
    the "no task pinned" silent ack.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    session = _session_with(plan)

    # Build a SessionContext that exposes the reporting-tool spec list
    # so the plugin's ``before_tool_callback`` routes
    # ``report_task_started`` through ``invoke_tool`` (not the silent
    # ack). Reuse the canonical spec list from goldfive.reporting so
    # we get the real handler.
    from goldfive.reporting import select_reporting_tools

    specs = select_reporting_tools(False)
    ctx_obj = SessionContext(
        session=session,
        steerer=None,
        task=None,
        tools=specs,
        host_agent_name="coord",
    )
    # Set the plugin's active ctx so the live-run path reaches the
    # SessionContext via ``session_context_from_invocation``.
    plugin.set_active_context(ctx_obj)

    # Stash the SessionContext on ADK state too so the legacy / unit-test
    # resolver path also finds it (defensive belt-and-braces).
    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-dispatch")

    # Phase 1: coord fires AgentTool(researcher) — triggers the pin.
    agent_tool = _FakeAgentTool("researcher")
    res = await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "please research the topic"},
        tool_context=coord_tool_context,
    )
    # AgentTool dispatch is not short-circuited (no runaway-cap).
    assert res is None or (isinstance(res, dict) and not res.get("skipped"))

    # The pin landed.
    a = _find_task(session.plan, "A")
    assert a is not None
    assert a.assignee_agent_id == "researcher"
    assert session.current_task_id == "A"

    # Phase 2: researcher (sub-agent) fires report_task_started. With
    # the pin in place, the reporting handler runs (returns a dict
    # describing the transition) rather than the silent ack.
    sub_inv = _FakeInvocationContext(adk_state, "researcher")
    sub_tool_context = _FakeToolContext(sub_inv, function_call_id="fc-report")
    report_tool = _FakeFunctionTool("report_task_started")

    res = await plugin.before_tool_callback(
        tool=report_tool,
        tool_args={},  # task_id is hidden from the LLM; the pin supplies it.
        tool_context=sub_tool_context,
    )

    # The reporting handler executed (or attempted to execute), which
    # means the pin lookup resolved — the response carries either the
    # handler's success payload or the handler's error payload (when
    # the test's stub steerer can't drive the real transition). What
    # MUST NOT happen is the no-task-pinned silent ack: the silent-ack
    # branch returns EXACTLY ``{"acknowledged": True}`` with no other
    # keys. Anything else means the pin resolved and the dispatch path
    # reached invoke_tool.
    assert isinstance(res, dict)
    assert res != {"acknowledged": True}, (
        f"reporting handler did not run; got silent ack: {res}"
    )
    # The task_id from the pin was injected into tool_args before the
    # dispatch (visible because the handler's response either succeeds
    # against task A or surfaces an error that references A). Either
    # way, the pin-resolution loop did not fall through to no-op.
    assert "missing_task_id" not in str(res), (
        f"task_id was not injected from the pin: {res}"
    )


# ---------------------------------------------------------------------------
# goldfive#262 — DelegationObserved emit happens AFTER the pin
# ---------------------------------------------------------------------------


class _SinkingSteerer:
    """Minimal steerer stub that owns a ``_sinks`` list.

    The plugin's ``_emit_observability`` reads ``steerer._sinks`` to fan
    sink events out — this stub is just enough to capture
    ``DelegationObserved`` events the plugin emits from
    ``before_tool_callback``.
    """

    def __init__(self, sink: Any) -> None:
        self._sinks = [sink]

    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _delegation_events(events: list[Any]) -> list[Any]:
    """Filter ``events`` to ``DelegationObserved`` payloads only."""
    out: list[Any] = []
    for e in events:
        if not hasattr(e, "WhichOneof"):
            continue
        if e.WhichOneof("payload") == "delegation_observed":
            out.append(e.delegation_observed)
    return out


async def test_delegation_observed_event_carries_bound_task_id() -> None:
    """The ``DelegationObserved`` event's ``task_id`` is the freshly-bound
    plan-task id (goldfive#262).

    Before #262 the emit ran BEFORE ``_maybe_pin_delegation_task``, so
    the proto field was empty on the typical orchestration-only
    coordinator turn (``ctx.task is None``). After the reorder the emit
    reads ``session.current_task_id`` which the pin just stamped — so
    the harmonograf ingest can attribute the delegation to the right
    task and stamp ``tasks.assignee_agent_id``.
    """
    from goldfive.sinks.memory import InMemorySink

    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    session = _session_with(plan)

    sink = InMemorySink()
    steerer = _SinkingSteerer(sink)

    # Orchestration-only coordinator turn: ctx.task is None — same
    # shape as the live coordinator that reproduced the bug
    # (session 4a721a07).
    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    agent_tool = _FakeAgentTool("researcher")
    await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "please research the topic"},
        tool_context=coord_tool_context,
    )

    # Confirm the pin landed.
    assert session.current_task_id == "A"
    a = _find_task(session.plan, "A")
    assert a is not None and a.assignee_agent_id == "researcher"

    # Exactly one delegation_observed event, carrying the bound task id.
    delegations = _delegation_events(sink.events)
    assert len(delegations) == 1, (
        f"expected one DelegationObserved; got {[type(e).__name__ for e in sink.events]}"
    )
    d = delegations[0]
    assert d.from_agent == "coord"
    assert d.to_agent == "researcher"
    assert d.task_id == "A", (
        f"DelegationObserved.task_id must carry the bound id; got '{d.task_id}'"
    )


async def test_delegation_observed_task_id_empty_when_no_eligible_task() -> None:
    """When the pin can't bind a task (no eligible PENDING tasks), the
    emit lands with ``task_id == ""`` (defensive — no fake binding).

    Mirror of the no-eligible-task case from the selection-algorithm
    tests, but checks the emit side rather than the pin side.
    """
    import dataclasses

    from goldfive.sinks.memory import InMemorySink

    plugin = make_adk_plugin(host_agent_name="coord")
    # Same shape as ``test_no_eligible_task_leaves_session_unpinned``:
    # A is RUNNING (not PENDING) so it's not eligible; B's predecessor A
    # is not COMPLETED so B is DAG-blocked. Zero eligible tasks.
    plan = _plan(
        Task(id="A", title="Plan the work"),
        Task(id="B", title="Execute the work"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    plan = dataclasses.replace(
        plan,
        tasks=(
            dataclasses.replace(plan.tasks[0], status=TaskStatus.RUNNING),
            plan.tasks[1],
        ),
    )
    session = _session_with(plan)

    sink = InMemorySink()
    steerer = _SinkingSteerer(sink)

    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    agent_tool = _FakeAgentTool("worker")
    await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "go"},
        tool_context=coord_tool_context,
    )

    # Pin did not bind anything.
    assert session.current_task_id == ""

    # Exactly one delegation_observed event, with empty task_id.
    delegations = _delegation_events(sink.events)
    assert len(delegations) == 1
    d = delegations[0]
    assert d.from_agent == "coord"
    assert d.to_agent == "worker"
    assert d.task_id == "", (
        f"expected empty task_id when no eligible task; got '{d.task_id}'"
    )


async def test_capability_check_still_resolves_after_reorder() -> None:
    """After the pin → emit → capability-check reorder, the capability
    detector still resolves the task via Strategy 1 (assignee, freshly
    stamped by the pin) — i.e. the reorder didn't break the goldfive#253
    detector.

    Rule A scenario: the invoked sub-agent has only AgentTool wrappers
    (no leaf tools) and the bound plan task is a leaf authoring task.
    The capability detector must fire CAPABILITY_MISMATCH.
    """
    from goldfive.types import DriftKind

    class _RecordingSteerer:
        def __init__(self) -> None:
            self._sinks: list[Any] = []
            self.drifts: list[Any] = []

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        def bind(self, **kw: Any) -> None:
            pass

        async def _handle_drift(self, drift: Any, session: Any) -> None:  # noqa: ARG002
            self.drifts.append(drift)

    plugin = make_adk_plugin(host_agent_name="coord")
    # Single leaf authoring task (PENDING, no assignee) — the pin will
    # bind it to the invoked underqualified sub-agent.
    plan = _plan(
        Task(id="t-draft", title="Draft a presentation about LLM observability"),
    )
    session = _session_with(plan)

    steerer = _RecordingSteerer()
    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    # The "underqualified" invoked agent has only an AgentTool wrapper
    # — Rule A's structural signal. Build a stand-in shape the
    # capability detector can introspect (it walks ``.tools`` on the
    # invoked agent).
    class _AgentToolWrapper:
        def __init__(self, name: str) -> None:
            self.name = name
            self.agent = _FakeAgent(name)

    class _Underqualified:
        name = "underqualified"
        tools = [_AgentToolWrapper("inner")]

    # AgentTool dispatch shape the plugin recognises.
    class _DispatchAgentTool:
        def __init__(self, sub: Any) -> None:
            self.name = sub.name
            self.agent = sub

    dispatch = _DispatchAgentTool(_Underqualified())

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    await plugin.before_tool_callback(
        tool=dispatch,
        tool_args={"request": "draft the presentation"},
        tool_context=coord_tool_context,
    )

    # Pin landed: assignee stamped, current_task_id pinned — Strategy 1
    # of the capability check resolves on this.
    t = _find_task(session.plan, "t-draft")
    assert t is not None and t.assignee_agent_id == "underqualified"
    assert session.current_task_id == "t-draft"

    # Capability detector fired and the drift reached the steerer.
    capability_drifts = [
        d for d in steerer.drifts if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert len(capability_drifts) >= 1, (
        f"expected CAPABILITY_MISMATCH; got {[d.kind for d in steerer.drifts]}"
    )
    drift = capability_drifts[0]
    assert drift.current_task_id == "t-draft"
    assert drift.current_agent_id == "underqualified"
