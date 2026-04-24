"""Integration tests for :class:`goldfive.Runner`.

Drives a :class:`CallableAdapter`-backed agent through goal derivation,
planning, and execution. Asserts the public event order:

    RunStarted, GoalDerived, PlanSubmitted,
    (TaskStarted, TaskCompleted) xN,
    RunCompleted

These tests are the single largest exercise of the public API surface
pinned by ``INTERFACE_SPEC.md`` — they touch every protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive import (
    BUILTIN_REPORTING_TOOLS,
    CallableAdapter,
    ExecutionOutcome,
    Goal,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hand_built_plan() -> Plan:
    """Three-task linear plan: research → draft → review."""
    return Plan(
        id="plan-fixture",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Research", assignee_agent_id="writer"),
            Task(id="draft", title="Draft", assignee_agent_id="writer"),
            Task(id="review", title="Review", assignee_agent_id="writer"),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
            TaskEdge(from_task_id="draft", to_task_id="review"),
        ],
        summary="Research, draft, review.",
    )


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Reference agent: returns a non-empty result so the executor auto-completes."""
    _ = tools
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _kinds(events: list[Any]) -> list[str]:
    def _one(e: Any) -> str:
        if isinstance(e, dict):
            return e.get("kind") or ""
        if hasattr(e, "WhichOneof"):
            name = e.WhichOneof("payload") or ""
            return "".join(part.capitalize() for part in name.split("_")) if name else ""
        return getattr(e, "kind", "")
    return [_one(e) for e in events]


def _payloads(events: list[Any]) -> list[dict[str, Any]]:
    def _one(e: Any) -> dict[str, Any]:
        if isinstance(e, dict):
            return e.get("payload") or {}
        if hasattr(e, "WhichOneof"):
            name = e.WhichOneof("payload")
            if not name:
                return {}
            msg = getattr(e, name)
            # For each field, surface its value; gracefully handle missing attrs.
            out: dict[str, Any] = {}
            for fd, v in msg.ListFields():
                out[fd.name] = v
            return out
        return getattr(e, "payload", {}) or {}
    return [_one(e) for e in events]


def _sequences(events: list[Any]) -> list[int]:
    out: list[int] = []
    for e in events:
        if isinstance(e, dict):
            out.append(int(e.get("sequence") or 0))
        else:
            out.append(int(getattr(e, "sequence", 0) or 0))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_runner_end_to_end_event_sequence() -> None:
    """Full end-to-end: derive → plan → execute → emit correct event sequence."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("Research, draft, review"),
        sinks=[sink],
    )

    outcome = await runner.run("go")
    await runner.close()

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.success, outcome.reason

    kinds = _kinds(sink.events)
    # Filter out the Conversation lifecycle events (ConversationStarted
    # prepends, ConversationEnded appends on close) — this test
    # pre-dates Phase 3 and exercises the per-run lifecycle only.
    run_kinds = [k for k in kinds if not k.startswith("Conversation")]

    # Lifecycle envelope up-front. The Runner owns the Run* lifecycle
    # events: RunStarted then GoalDerived then PlanSubmitted. The
    # executor no longer emits its own RunStarted (that would duplicate
    # the Runner's emission and produce a second event for sinks).
    assert run_kinds[0] == "RunStarted"
    assert run_kinds[1] == "GoalDerived"
    assert run_kinds[2] == "PlanSubmitted"

    # For each of the three tasks: TaskStarted → TaskTransitioned (R4)
    # → TaskCompleted → TaskTransitioned (R4). The TaskTransitioned
    # envelopes are observability-only (goldfive#251 R4); the per-status
    # proto envelopes remain the LLM-visible authoritative signal.
    task_kinds = run_kinds[3:-1]  # strip initial triple and trailing RunCompleted
    # Filter to the per-status envelopes (drop TaskTransitioned) before
    # asserting the strict TaskStarted -> TaskCompleted ordering.
    status_kinds = [
        k for k in task_kinds if k in ("TaskStarted", "TaskCompleted")
    ]
    assert len(status_kinds) == 6, run_kinds
    assert status_kinds[0::2] == ["TaskStarted"] * 3
    assert status_kinds[1::2] == ["TaskCompleted"] * 3

    # Run terminates with a RunCompleted from the executor.
    assert run_kinds[-1] == "RunCompleted"

    # Sequence numbers are strictly monotonic from zero.
    seqs = _sequences(sink.events)
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert seqs[0] == 0

    # Task order respects the DAG: research → draft → review.
    payloads = _payloads(sink.events)
    task_ids_in_order = [
        payloads[i]["task_id"] for i, k in enumerate(kinds) if k == "TaskStarted"
    ]
    assert task_ids_in_order == ["research", "draft", "review"]

    # Session is fully populated.
    assert outcome.session.plan is not None
    assert len(outcome.session.goals) == 1
    assert all(
        outcome.session.plan.tasks[i].status.value == "COMPLETED" for i in range(3)
    )


async def test_runner_accepts_prebuilt_goal_list() -> None:
    """list[Goal] input bypasses the deriver."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        sinks=[sink],
    )
    goals = [Goal(id="g1", summary="pre-baked")]
    outcome = await runner.run(goals)
    await runner.close()

    assert outcome.success
    assert outcome.session.goals == goals


async def test_runner_aborts_when_planner_returns_none() -> None:
    """No plan → ``RunAborted`` with a clear reason, no executor invocation."""
    from goldfive import PassthroughPlanner

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=[]),
        planner=PassthroughPlanner(),  # returns None from generate()
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("ignored"),
        sinks=[sink],
    )
    outcome = await runner.run("go")
    await runner.close()

    assert not outcome.success
    assert "no plan" in outcome.reason

    run_kinds = [k for k in _kinds(sink.events) if not k.startswith("Conversation")]
    assert run_kinds == ["RunStarted", "GoalDerived", "RunAborted"]


async def test_runner_registers_builtin_reporting_tools() -> None:
    """The adapter receives the canonical seven reporting tools verbatim."""
    captured: dict[str, Any] = {}

    async def agent_fn(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        captured.setdefault("tools", tools)
        return InvocationResult(task_id=task.id, text="ok")

    runner = Runner(
        agent=CallableAdapter(agent_fn, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("any"),
        sinks=[InMemorySink()],
    )
    outcome = await runner.run("go")
    await runner.close()
    assert outcome.success

    tools = captured["tools"]
    names = [t.name for t in tools]
    expected = [t.name for t in BUILTIN_REPORTING_TOOLS]
    assert names == expected


async def test_runner_close_closes_all_sinks() -> None:
    """Runner.close() invokes ``close`` on every sink."""
    closed: list[bool] = []

    class _SpySink:
        events: list[Any]

        def __init__(self) -> None:
            self.events = []

        async def emit(self, event: Any) -> None:
            self.events.append(event)

        async def close(self) -> None:
            closed.append(True)

    spy_a = _SpySink()
    spy_b = _SpySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("close test"),
        sinks=[spy_a, spy_b],
    )
    await runner.run("go")
    await runner.close()

    assert closed == [True, True]


async def test_runner_input_type_errors() -> None:
    """Non str / list[Goal] inputs raise cleanly and produce RunAborted."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        sinks=[sink],
    )

    outcome = await runner.run(42)  # type: ignore[arg-type]
    await runner.close()
    assert not outcome.success
    assert "str or list[Goal]" in outcome.reason
    run_kinds = [k for k in _kinds(sink.events) if not k.startswith("Conversation")]
    assert run_kinds == ["RunStarted", "RunAborted"]


@pytest.mark.asyncio
async def test_runner_protocol_compatibility() -> None:
    """Every default implementation must satisfy its Protocol at runtime."""
    from goldfive import (
        AgentAdapter,
        EventSink,
        Executor,
        GoalDeriver,
        Planner,
        Steerer,
    )
    from goldfive.steerer import DefaultSteerer

    adapter = CallableAdapter(_happy_agent, available_agents=["writer"])
    assert isinstance(adapter, AgentAdapter)

    planner = StaticPlanner(_hand_built_plan())
    assert isinstance(planner, Planner)

    executor = SequentialExecutor()
    assert isinstance(executor, Executor)

    steerer = DefaultSteerer()
    assert isinstance(steerer, Steerer)

    deriver = PassthroughGoalDeriver("x")
    assert isinstance(deriver, GoalDeriver)

    sink = InMemorySink()
    assert isinstance(sink, EventSink)
