"""Tests for task_id wiring (goldfive#191 Layers 1 + 2).

Covers:

* Layer 1 — ``_GoldfiveADKPlugin.before_agent_callback`` pins
  ``goldfive.current_task_id`` onto BOTH the ADK session.state and the
  goldfive orchestration session.state when the starting sub-agent has
  exactly one PENDING / RUNNING task assigned to it. Ambiguous
  (multiple) and absent (zero) matches leave state unset so the
  ``missing_task_id`` error path still fires.
* Layer 2 — reporting-tool handlers fall back to
  ``session.state["goldfive.current_task_id"]`` when the model's tool
  call omits the ``task_id`` arg, yield the canonical
  ``missing_task_id`` error when neither source supplies a value, and
  honour an explicit arg over the state fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID  # noqa: E402
from goldfive.adapters._tool_invocation import invoke_tool  # noqa: E402
from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    ReportingToolSpec,
)
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


class _StateCtx:
    """Minimal ADK-callback-context stub with a ``session.state`` dict."""

    class _Session:
        def __init__(self, state: dict) -> None:
            self.state = state

    def __init__(self, state: dict) -> None:
        self._state = state

    @property
    def session(self) -> Any:
        return _StateCtx._Session(self._state)


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class _RecordingSteerer:
    """Steerer stub that records transitions + drifts for assertions."""

    def __init__(self) -> None:
        self.transitions: list[tuple[str, Any, str]] = []
        self.drifts: list[Any] = []

    async def mark_task_running(self, task_id: str, *, session: Any, detail: str = "") -> None:
        self.transitions.append((task_id, "RUNNING", detail))

    async def mark_task_progress(
        self, task_id: str, *, session: Any, fraction: float = 0.0, detail: str = ""
    ) -> None:
        self.transitions.append((task_id, "PROGRESS", detail))

    async def mark_task_completed(
        self, task_id: str, *, session: Any, summary: str = "", artifacts: Any = None
    ) -> None:
        self.transitions.append((task_id, "COMPLETED", summary))

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Any,
        reason: str = "",
        recoverable: bool = True,
    ) -> None:
        self.transitions.append((task_id, "FAILED", reason))

    async def mark_task_blocked(
        self, task_id: str, *, session: Any, blocker: str = "", needed: str = ""
    ) -> None:
        self.transitions.append((task_id, "BLOCKED", blocker))

    async def report_new_work_discovered(self, **kwargs: Any) -> None:
        pass

    async def report_plan_divergence(self, **kwargs: Any) -> None:
        pass

    async def observe(self, event: Any, session: Any) -> None:
        self.drifts.append(event)

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, event: Any, session: Any) -> Any:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _plan_with(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
    )


def _session_with(plan: Plan | None) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx_for(session: Session, agent_name: str) -> tuple[dict, Any]:
    """Build a ({adk_state}, callback_context) pair for plugin callbacks."""
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=None,
            task=None,
            tool_handlers={spec.name: spec.handler for spec in BUILTIN_REPORTING_TOOLS},
            host_agent_name=agent_name,
        ),
    }
    return state, _StateCtx(state)


# ---------------------------------------------------------------------------
# Layer 1 — before_agent_callback pins current_task_id
# ---------------------------------------------------------------------------


async def test_before_agent_callback_pins_unambiguous_task() -> None:
    """Exactly one PENDING task assigned to the starting agent → state pinned."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="research", assignee_agent_id="research_agent"),
            Task(id="t2", title="code", assignee_agent_id="web_developer"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    # ADK session.state side (sub-agent-visible).
    assert state[KEY_CURRENT_TASK_ID] == "t1"
    # Goldfive orchestration-state side (handler fallback).
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_before_agent_callback_ambiguous_match_pins_low_confidence() -> None:
    """Two PENDING tasks for the same agent → signal 8 picks deterministically.

    goldfive#264 reframed the resolver: an invoked agent must end up
    with a pin (something precipitated the call). The ambiguous-match
    case now falls through to signal 8 which picks the first
    assignee-matching candidate with low-confidence telemetry. The
    pre-#264 silent-unset behaviour is gone.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="research_agent"),
            Task(id="t2", title="second", assignee_agent_id="research_agent"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    # The ambiguous case lands on the first deterministic candidate.
    assert state[KEY_CURRENT_TASK_ID] in {"t1", "t2"}
    assert session.state["goldfive.current_task_id"] == state[KEY_CURRENT_TASK_ID]


async def test_before_agent_callback_no_match_leaves_unset() -> None:
    """Zero PENDING tasks for the agent (off-plan agent) → no stamp."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="t1", title="first", assignee_agent_id="research_agent"),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    # Agent name doesn't match any assignee in the plan.
    await plugin.before_agent_callback(
        agent=_Agent("rogue_agent"),
        callback_context=ctx,
    )

    assert KEY_CURRENT_TASK_ID not in state
    assert "goldfive.current_task_id" not in session.state


async def test_before_agent_callback_includes_running_tasks() -> None:
    """A RUNNING task for the agent also counts (re-entry / re-spawn path)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    running_task = Task(
        id="t1",
        title="research",
        assignee_agent_id="research_agent",
        status=TaskStatus.RUNNING,
    )
    session = _session_with(_plan_with(running_task))
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "t1"
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_before_agent_callback_skips_terminal_tasks() -> None:
    """A COMPLETED task does not count; if it's the only match, state stays unset."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(
                id="t1",
                title="already done",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert KEY_CURRENT_TASK_ID not in state
    assert "goldfive.current_task_id" not in session.state


# ---------------------------------------------------------------------------
# Layer 1 (goldfive#242) — DAG-aware candidate filter
# ---------------------------------------------------------------------------


async def test_pin_relaxes_dag_gate_when_upstream_incomplete() -> None:
    """A -> B, A is PENDING, agent is assigned to B -> signal 4 binds B.

    goldfive#264 reframe: the DAG-ready filter is now signal 2; if it
    rejects every candidate, signal 4 retries without the gate and
    pins the assignee match. The operator-visible safety net moves
    from "silently no-op" to "WARNING log + dag_relaxed sink event".
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(id="a", title="A", assignee_agent_id="other"),
            Task(id="b", title="B", assignee_agent_id="research_agent"),
            edges=[TaskEdge("a", "b")],
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    # Signal 4 — DAG gate relaxed — pins B.
    assert state[KEY_CURRENT_TASK_ID] == "b"
    assert session.state["goldfive.current_task_id"] == "b"


async def test_pin_selects_task_after_upstream_completes() -> None:
    """A COMPLETED, B PENDING, agent assigned to B -> B is pinned."""
    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(
        _plan_with(
            Task(
                id="a",
                title="A",
                assignee_agent_id="other",
                status=TaskStatus.COMPLETED,
            ),
            Task(id="b", title="B", assignee_agent_id="research_agent"),
            edges=[TaskEdge("a", "b")],
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "b"
    assert session.state["goldfive.current_task_id"] == "b"


async def test_pin_ambiguous_narrows_by_dag() -> None:
    """Two agent-matches, one has incomplete upstream -> singleton remains, pins."""
    plugin = make_adk_plugin(host_agent_name="coord")
    # ``t1`` is free of upstream deps; ``t2`` depends on ``gate`` which
    # is still PENDING. Without the DAG gate both would be candidates
    # and pin would bail on ambiguity; with the gate, only ``t1``
    # remains.
    session = _session_with(
        _plan_with(
            Task(id="gate", title="gate", assignee_agent_id="other"),
            Task(id="t1", title="first", assignee_agent_id="research_agent"),
            Task(id="t2", title="second", assignee_agent_id="research_agent"),
            edges=[TaskEdge("gate", "t2")],
        )
    )
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "t1"
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_pin_finalize_with_incomplete_upstream_relaxes_loudly() -> None:
    """Live-scenario regression (goldfive#242 + #264 reframe).

    5-stage plan; only the Stage-4 finalize is the coordinator's task.
    Stages 0-3 are PENDING. Pre-#242 the pin would race ahead and the
    LLM would call ``report_task_started`` on finalize while upstream
    was incomplete; #242 added a strict DAG gate that left the pin
    unset and produced silent no-ops in the reporting path. #264
    reframes again: the agent WAS invoked so something precipitated
    it. Signal 4 relaxes the DAG gate and binds finalize, but loudly —
    a ``dag_relaxed`` sink event + WARNING log gives operators the
    same anomaly visibility the silent unset used to deny them.
    """
    plugin = make_adk_plugin(host_agent_name="coordinator_agent")
    session = _session_with(
        _plan_with(
            Task(id="s0", title="Gather", assignee_agent_id="researcher"),
            Task(id="s1", title="Outline", assignee_agent_id="outliner"),
            Task(id="s2", title="Draft", assignee_agent_id="writer"),
            Task(id="s3", title="Review", assignee_agent_id="reviewer"),
            Task(
                id="finalize_and_deliver_presentation",
                title="Finalize",
                assignee_agent_id="coordinator_agent",
            ),
            edges=[
                TaskEdge("s0", "s1"),
                TaskEdge("s1", "s2"),
                TaskEdge("s2", "s3"),
                TaskEdge("s3", "finalize_and_deliver_presentation"),
            ],
        )
    )
    state, ctx = _ctx_for(session, "coordinator_agent")

    await plugin.before_agent_callback(
        agent=_Agent("coordinator_agent"),
        callback_context=ctx,
    )

    # Signal 4 binds finalize (the agent WAS invoked).
    assert state[KEY_CURRENT_TASK_ID] == "finalize_and_deliver_presentation"
    assert session.state["goldfive.current_task_id"] == "finalize_and_deliver_presentation"


async def test_pin_supersedes_redirection_tracks_replacement() -> None:
    """Edge ``A -> C`` survives supersession; readiness tracks B's status.

    goldfive#264: signal 2 (DAG-ready) consults supersession-aware
    readiness via :func:`task_upstream_ready`. With B still PENDING
    signal 2 fails, but signal 4 (DAG-relaxed) still binds C — the
    agent was invoked. Once B completes, signal 2 picks up C as the
    happy-path single-DAG-ready match.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    # A is the original, now FAILED; B replaces A (B.supersedes == "A");
    # C depends on A in the edges table. The live status driving C's
    # readiness must be B's (PENDING), not A's (FAILED).
    plan = _plan_with(
        Task(id="A", title="A", status=TaskStatus.FAILED, assignee_agent_id="other"),
        Task(
            id="B",
            title="B",
            status=TaskStatus.PENDING,
            assignee_agent_id="other",
            supersedes="A",
        ),
        Task(id="C", title="C", assignee_agent_id="research_agent"),
        edges=[TaskEdge("A", "C")],
    )
    session = _session_with(plan)
    state, ctx = _ctx_for(session, "coord")

    # B is still PENDING -> signal 2 rejects, but signal 4 binds C.
    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )
    assert state[KEY_CURRENT_TASK_ID] == "C"

    # Flip B to COMPLETED; now C is DAG-ready -> signal 2 binds.
    plan.tasks[1].status = TaskStatus.COMPLETED
    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )
    assert state[KEY_CURRENT_TASK_ID] == "C"


async def test_pin_orchestration_block_renders_pinned_task_after_relaxation() -> None:
    """When signal 4 relaxes the DAG gate, GoldfivePlanner renders the bound task.

    Pre-#264 the user-visible effect was an unset pin → ``(none)`` in
    the planner instruction block. Post-#264 the relaxed pin reaches
    the prompt — the operator-visible safety net moved to the sink
    event + log. The LLM seeing the task id is the intended behaviour:
    something precipitated the call, and the pin is goldfive's best
    structural answer to "what was I invoked for".
    """
    from goldfive.planners.goldfive_planner import GoldfivePlanner

    plugin = make_adk_plugin(host_agent_name="coordinator_agent")
    session = _session_with(
        _plan_with(
            Task(id="s0", title="Gather", assignee_agent_id="researcher"),
            Task(
                id="finalize",
                title="Finalize",
                assignee_agent_id="coordinator_agent",
            ),
            edges=[TaskEdge("s0", "finalize")],
        )
    )
    state, ctx = _ctx_for(session, "coordinator_agent")

    await plugin.before_agent_callback(
        agent=_Agent("coordinator_agent"),
        callback_context=ctx,
    )

    # Signal 4 relaxes the DAG gate and binds finalize.
    assert state[KEY_CURRENT_TASK_ID] == "finalize"

    # Build the planner instruction the LLM would see. We use a
    # tolerant readonly-context stub carrying the ADK state dict.
    planner = GoldfivePlanner(session=session)

    class _Readonly:
        def __init__(self, s: dict) -> None:
            self.state = s

    instruction = planner.build_planning_instruction(_Readonly(state), None)
    assert instruction is not None
    # The pinned task id reaches the prompt now that the resolver
    # binds aggressively. The "(none)" string was the pre-#264 marker
    # of a silent unset; the new contract surfaces the bound task and
    # records the relaxation for operators via the sink event.
    assert "finalize" in instruction
    assert "Plan task (if any): (none)" not in instruction


# ---------------------------------------------------------------------------
# Layer 2 — reporting-tool handlers default task_id from state
# ---------------------------------------------------------------------------


def _get_spec(name: str) -> ReportingToolSpec:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"builtin tool {name!r} missing")


async def test_reporting_tool_uses_state_task_id_when_arg_missing() -> None:
    """State fallback reaches the handler via ``invoke_tool``."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"detail": "starting"},  # no task_id
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    # The handler ran with the state-derived id.
    assert steerer.transitions == [("t-1", "RUNNING", "starting")]


async def test_reporting_tool_honors_explicit_arg_over_state() -> None:
    """An explicit ``task_id`` wins against the session-state fallback."""
    steerer = _RecordingSteerer()
    # Plan carries both ids so the ``unknown_task_id`` guard doesn't mask
    # the behavior we're trying to observe.
    session = _session_with(
        _plan_with(
            Task(id="t-1", title="a", assignee_agent_id="a"),
            Task(id="t-2", title="b", assignee_agent_id="b"),
        )
    )
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"task_id": "t-2", "detail": "explicit wins"},
        session,
        steerer,
    )

    assert result == {"acknowledged": True}
    # Explicit arg is what reached the handler.
    assert steerer.transitions == [("t-2", "RUNNING", "explicit wins")]


async def test_reporting_tool_errors_when_neither_arg_nor_state() -> None:
    """Neither arg nor state → canonical ``missing_task_id`` rejection."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    # session.state intentionally empty — no fallback.

    spec = _get_spec("report_task_started")
    result = await invoke_tool(
        [spec],
        "report_task_started",
        {"detail": "starting"},
        session,
        steerer,
    )

    assert result["acknowledged"] is False
    assert result["error"] == "missing_task_id"
    assert result["tool"] == "report_task_started"
    # No transition — handler must not run.
    assert steerer.transitions == []


@pytest.mark.parametrize(
    "tool_name,extra_args",
    [
        ("report_task_started", {}),
        ("report_task_progress", {"fraction": 0.5}),
        ("report_task_completed", {"summary": "done"}),
        ("report_task_failed", {"reason": "oops", "recoverable": True}),
        ("report_task_blocked", {"blocker": "missing input"}),
    ],
)
async def test_all_task_scoped_reporting_tools_default_task_id(
    tool_name: str, extra_args: dict[str, Any]
) -> None:
    """Every task-scoped reporting tool defaults ``task_id`` from state.

    Parameterized over the five task-scoped tools in
    :data:`BUILTIN_REPORTING_TOOLS`. The remaining three (the plan-level
    ``report_plan_divergence`` + ``report_new_work_discovered`` and the
    waiter-based ``report_awaiting_approval``) are not task-id-gated by
    the invoke_tool schema layer (``report_plan_divergence`` is
    plan-level; ``report_new_work_discovered`` takes ``parent_task_id``;
    ``report_awaiting_approval`` blocks on a waiter this test won't set
    up), so they follow a different shape — covered separately.
    """
    steerer = _RecordingSteerer()
    # goldfive#201: start the task RUNNING so ``report_task_progress``
    # is a legal transition under the handler's idempotency matrix
    # (progress ticks are only valid on RUNNING). Other handlers
    # (started, completed, failed, blocked) are legal from
    # RUNNING too, so this starting state works for the whole
    # parametrisation.
    session = _session_with(
        _plan_with(
            Task(
                id="t-1",
                title="do",
                assignee_agent_id="a",
                status=TaskStatus.RUNNING,
            )
        )
    )
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec(tool_name)
    args: dict[str, Any] = dict(extra_args)  # no task_id supplied
    result = await invoke_tool([spec], tool_name, args, session, steerer)

    # Each tool either ACKs or passes through to the handler with the
    # state-derived id; neither should produce a ``missing_task_id``.
    assert result.get("error") != "missing_task_id", (
        f"{tool_name} did not default task_id from state: {result!r}"
    )
    # Handler ran with the resolved id. (report_task_started is an
    # idempotent no-op on RUNNING, so it may not register a transition
    # — accept either a transition or an idempotent ACK for that one.)
    if tool_name == "report_task_started":
        assert result.get("acknowledged") is True
        assert result.get("idempotent") is True
    else:
        assert any(t[0] == "t-1" for t in steerer.transitions), (
            f"{tool_name} did not invoke the handler with t-1: transitions={steerer.transitions!r}"
        )


async def test_report_awaiting_approval_accepts_state_task_id() -> None:
    """``report_awaiting_approval`` also resolves task_id from state.

    Unlike the other tools, it fails at the ``prompt`` schema check if
    absent, but we're testing the state-fallback behavior specifically.
    We don't drive the waiter to completion — we just verify the
    handler does NOT return ``missing_task_id`` when state supplies
    the id.
    """
    import asyncio

    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))
    session.state["goldfive.current_task_id"] = "t-1"

    spec = _get_spec("report_awaiting_approval")
    # Set the waiter in advance so the handler doesn't block forever.
    waiter = asyncio.Event()
    session.pending_approvals["t-1"] = waiter
    session.pending_approvals_meta["t-1"] = {
        "kind": "task",
        "prompt": "ok?",
        "task_id": "t-1",
        "decision": "approve",
        "detail": "",
    }
    waiter.set()

    result = await invoke_tool(
        [spec],
        "report_awaiting_approval",
        {"prompt": "approve this?"},  # no task_id
        session,
        steerer,
    )

    assert result.get("error") != "missing_task_id"
    assert result.get("decision") == "approve"


async def test_missing_task_id_still_rejects_when_state_empty() -> None:
    """Regression guard: the existing ``missing_task_id`` rejection path
    still fires when neither ``args.task_id`` nor session-state supplies
    a value. The fallback must not degrade the schema guard."""
    steerer = _RecordingSteerer()
    session = _session_with(_plan_with(Task(id="t-1", title="do", assignee_agent_id="a")))

    spec = _get_spec("report_task_failed")
    for bad_args in (
        {},
        {"task_id": ""},
        {"task_id": "   "},
        {"task_id": None, "reason": "x"},
        {"reason": "x"},
    ):
        result = await invoke_tool([spec], "report_task_failed", dict(bad_args), session, steerer)
        assert result["acknowledged"] is False, bad_args
        assert result["error"] == "missing_task_id", bad_args
    # No handler runs → no transitions.
    assert steerer.transitions == []


# ---------------------------------------------------------------------------
# Compound-assignee end-to-end (goldfive#214)
# ---------------------------------------------------------------------------


async def test_before_agent_callback_pins_when_plan_came_from_compound_json() -> None:
    """Plans built from compound-form planner JSON still pin the right task.

    Regression guard for #214: the planner's JSON→TaskPlan path normalizes
    ``"<client>:<agent>"`` to the bare ADK name, so the reconciler /
    plugin match (`assignee == agent_name`) finds exactly one candidate.
    """
    from goldfive.planner import _plan_from_json

    payload = {
        "summary": "s",
        "tasks": [
            {
                "id": "t1",
                "title": "research",
                "assignee_agent_id": "presentation-orchestrated-9b2b3a9c7289:research_agent",
            },
            {
                "id": "t2",
                "title": "code",
                "assignee_agent_id": "presentation-orchestrated-9b2b3a9c7289:web_developer",
            },
        ],
    }
    plan = _plan_from_json(payload, run_id="r1", goal_ids=[])
    assert plan is not None
    # Sanity: normalization happened at parse time.
    assert [t.assignee_agent_id for t in plan.tasks] == [
        "research_agent",
        "web_developer",
    ]

    plugin = make_adk_plugin(host_agent_name="coord")
    session = _session_with(plan)
    state, ctx = _ctx_for(session, "coord")

    await plugin.before_agent_callback(
        agent=_Agent("research_agent"),
        callback_context=ctx,
    )

    assert state[KEY_CURRENT_TASK_ID] == "t1"
    assert session.state["goldfive.current_task_id"] == "t1"
