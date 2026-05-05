"""Cross-cutting Runner integration test.

Drives an end-to-end run: user_input -> GoalDeriver -> Planner -> parallel
DAG executor -> JSONL persistence sink -> replay from disk. The test
skips cleanly if any of the required subsystems is not yet implemented;
once they all land, the test is the single largest exercise of the
public contract pinned by INTERFACE_SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

types = pytest.importorskip("goldfive.types")


# ---------------------------------------------------------------------------
# Local stub collaborators — these stand in for feature-PR implementations
# so the integration test can focus on *wiring*, not on individual module
# correctness (which other test files cover).
# ---------------------------------------------------------------------------


class _ScriptedGoalDeriver:
    async def derive(self, user_input, *, context=None):
        return [
            types.Goal(id="g1", summary=f"handle: {user_input}"),
        ]


class _DAGPlanner:
    """Emits a small fan-out-fan-in DAG: t1 -> {t2, t3} -> t4."""

    def __init__(self) -> None:
        self.refine_calls = 0

    async def generate(self, *, goals, available_agents, context=None):
        assert goals, "planner expects at least one goal"
        tasks = [
            types.Task(id="t1", title="prep", assignee_agent_id="default"),
            types.Task(id="t2", title="branch-a", assignee_agent_id="default"),
            types.Task(id="t3", title="branch-b", assignee_agent_id="default"),
            types.Task(id="t4", title="join", assignee_agent_id="default"),
        ]
        edges = [
            types.TaskEdge("t1", "t2"),
            types.TaskEdge("t1", "t3"),
            types.TaskEdge("t2", "t4"),
            types.TaskEdge("t3", "t4"),
        ]
        return types.Plan(
            id="plan-0",
            run_id="will-be-overwritten",
            goal_ids=[g.id for g in goals],
            tasks=tasks,
            edges=edges,
        )

    async def refine(self, *, plan, drift, goals):
        self.refine_calls += 1
        return None


async def test_dag_planner_emits_expected_topological_stages() -> None:
    """``Plan.topological_stages`` must produce the classic fan-out shape."""

    planner = _DAGPlanner()
    plan = await planner.generate(
        goals=[types.Goal(id="g1", summary="go")], available_agents=["default"]
    )
    stages = plan.topological_stages()
    flat_ids = [t.id for stage in stages for t in stage]
    assert flat_ids[0] == "t1"
    assert flat_ids[-1] == "t4"
    # t2 and t3 must both appear before t4.
    idx = {tid: i for i, tid in enumerate(flat_ids)}
    assert idx["t2"] < idx["t4"]
    assert idx["t3"] < idx["t4"]
    assert idx["t1"] < idx["t2"] and idx["t1"] < idx["t3"]


# ---------------------------------------------------------------------------
# Full Runner wiring — skipped until every required module lands.
# ---------------------------------------------------------------------------


async def test_runner_drives_goal_plan_parallel_dag_and_persists(tmp_path: Path) -> None:
    runner_mod = pytest.importorskip("goldfive.runner")
    adapters = pytest.importorskip("goldfive.adapters.callable")
    executors = pytest.importorskip("goldfive.executors")
    sinks_mod = pytest.importorskip("goldfive.sinks")
    steerers = pytest.importorskip("goldfive.steerers")

    Runner = getattr(runner_mod, "Runner", None)
    CallableAdapter = getattr(adapters, "CallableAdapter", None)
    ParallelExecutor = getattr(executors, "ParallelExecutor", None)
    JSONLPersistenceSink = getattr(sinks_mod, "JSONLPersistenceSink", None)
    InMemorySink = getattr(sinks_mod, "InMemorySink", None)
    DefaultSteerer = getattr(steerers, "DefaultSteerer", None)

    if not all(
        [
            Runner,
            CallableAdapter,
            ParallelExecutor,
            JSONLPersistenceSink,
            InMemorySink,
            DefaultSteerer,
        ]
    ):
        pytest.skip("Required goldfive modules not yet implemented")

    log_path = tmp_path / "run.jsonl"

    # The adapter routes all tasks to a single handler that reports start,
    # progress, and completion via the reporting tools injected by the
    # adapter at registration time. The CallableAdapter PR is expected to
    # accept a mapping of agent_id -> async handler.
    async def _handler(task, session):
        # Simulate a fast "do something".
        return types.InvocationResult(  # type: ignore[attr-defined]
            task_id=task.id,
            text=f"done:{task.id}",
            stop_reason="end",
        ) if hasattr(types, "InvocationResult") else None

    adapter = CallableAdapter(
        handlers={"default": _handler},
        available_agents=["default"],
    )
    planner = _DAGPlanner()
    executor = ParallelExecutor()
    steerer = DefaultSteerer()
    memory_sink = InMemorySink()
    try:
        persistence_sink = JSONLPersistenceSink(log_path)
    except TypeError:
        persistence_sink = JSONLPersistenceSink(str(log_path))

    runner = Runner(
        agent=adapter,
        planner=planner,
        executor=executor,
        goal_deriver=_ScriptedGoalDeriver(),
        steerer=steerer,
        sinks=[memory_sink, persistence_sink],
    )

    outcome = await runner.run("ship the thing")
    assert outcome.success is True
    session = outcome.session
    assert session.plan is not None
    assert session.plan.goal_ids == ("g1",)

    # All four tasks should have reached COMPLETED.
    statuses = {t.id: t.status for t in session.plan.tasks}
    assert all(s == types.TaskStatus.COMPLETED for s in statuses.values())

    # Every emitted sequence is monotonic and unique.
    mem_events = getattr(memory_sink, "events", [])
    if mem_events:
        seqs = [getattr(e, "sequence", None) for e in mem_events]
        seqs = [s for s in seqs if s is not None]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))

    # Replay the JSONL file. The count written must equal the count emitted
    # to the in-memory sink.
    assert log_path.exists()
    persisted = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if mem_events:
        assert len(persisted) == len(mem_events)


async def test_runner_accepts_prebuilt_goals_list() -> None:
    """A Runner should accept either a user_input string or a list of Goals
    directly, bypassing the GoalDeriver when goals are pre-built."""

    runner_mod = pytest.importorskip("goldfive.runner")
    adapters = pytest.importorskip("goldfive.adapters.callable")
    executors = pytest.importorskip("goldfive.executors")
    steerers = pytest.importorskip("goldfive.steerers")

    Runner = getattr(runner_mod, "Runner", None)
    CallableAdapter = getattr(adapters, "CallableAdapter", None)
    SequentialExecutor = getattr(executors, "SequentialExecutor", None)
    DefaultSteerer = getattr(steerers, "DefaultSteerer", None)
    if not all([Runner, CallableAdapter, SequentialExecutor, DefaultSteerer]):
        pytest.skip("Required goldfive modules not yet implemented")

    async def _handler(task, session):
        return None

    runner = Runner(
        agent=CallableAdapter(handlers={"default": _handler}, available_agents=["default"]),
        planner=_DAGPlanner(),
        executor=SequentialExecutor(),
        steerer=DefaultSteerer(),
        sinks=[],
    )
    outcome = await runner.run([types.Goal(id="g0", summary="prebuilt")])
    assert outcome.session.goals and outcome.session.goals[0].id == "g0"


async def test_runner_respects_max_task_invocations() -> None:
    """``max_task_invocations`` caps refine() calls; after the cap the
    run should terminate rather than loop forever."""

    runner_mod = pytest.importorskip("goldfive.runner")
    adapters = pytest.importorskip("goldfive.adapters.callable")
    executors = pytest.importorskip("goldfive.executors")
    steerers = pytest.importorskip("goldfive.steerers")

    Runner = getattr(runner_mod, "Runner", None)
    CallableAdapter = getattr(adapters, "CallableAdapter", None)
    SequentialExecutor = getattr(executors, "SequentialExecutor", None)
    DefaultSteerer = getattr(steerers, "DefaultSteerer", None)
    if not all([Runner, CallableAdapter, SequentialExecutor, DefaultSteerer]):
        pytest.skip("Required goldfive modules not yet implemented")

    # A planner that always requests a new refine; the runner must still
    # terminate because max_task_invocations caps the loop.
    class _LoopingPlanner(_DAGPlanner):
        async def refine(self, *, plan, drift, goals):
            self.refine_calls += 1
            return types.Plan(
                id=f"plan-{plan.revision_index + 1}",
                run_id=plan.run_id,
                goal_ids=plan.goal_ids,
                tasks=plan.tasks,
                edges=plan.edges,
                revision_index=plan.revision_index + 1,
            )

    async def _handler(task, session):
        return None

    planner = _LoopingPlanner()
    runner = Runner(
        agent=CallableAdapter(handlers={"default": _handler}, available_agents=["default"]),
        planner=planner,
        executor=SequentialExecutor(),
        steerer=DefaultSteerer(),
        sinks=[],
        max_task_invocations=2,
    )
    outcome = await runner.run([types.Goal(id="g1", summary="cap me")])
    # Whatever the final outcome, refine_calls must not exceed the cap.
    assert planner.refine_calls <= 2
    assert outcome is not None
