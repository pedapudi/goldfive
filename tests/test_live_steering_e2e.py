"""End-to-end live-steering integration tests (part of #71).

These exercise the full :class:`goldfive.Runner` + :class:`ControlChannel`
pipeline with a real :class:`SequentialExecutor`, real
:class:`DefaultSteerer`, and a real :class:`LLMPlanner` (using a
deterministic stub for ``call_llm``) — so we cover the actual wire-up
between runner, executor, steerer, and planner, not just one module in
isolation.

Scenarios covered:

* CANCEL arriving while a task is in flight aborts the run with
  ``RunAborted`` and cancels the adapter's in-flight task.
* STEER arriving while a task is in flight fires ``planner.refine``
  with a ``USER_STEER`` drift, swaps in the refined plan, and continues
  executing the new tasks.
* PAUSE between tasks blocks the executor until RESUME arrives.
* REWIND_TO resets a target task plus every downstream task back to
  ``PENDING`` so the executor re-walks them.
* STATUS_QUERY is a read-only probe — it does NOT emit any sink
  events; the status snapshot is returned via the ack's ``detail``
  field.

Module-level tests in ``tests/test_executor_control.py`` already cover
executor-level semantics with a stub steerer; the tests here verify the
whole stack end-to-end including the Runner's lifecycle events.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from goldfive import (
    CallableAdapter,
    DefaultSteerer,
    DriftKind,
    Goal,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
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
from goldfive.control import (
    AckResult,
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.types import DriftEvent, DriftSeverity, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_kind(evt: Any) -> str:
    """Best-effort proto/dict event-kind extraction."""
    if hasattr(evt, "WhichOneof"):
        return evt.WhichOneof("payload") or ""
    if isinstance(evt, dict):
        return str(evt.get("kind", ""))
    return ""


def _event_kinds(events: list[Any]) -> list[str]:
    return [_event_kind(e) for e in events]


def _drift_details(events: list[Any]) -> list[str]:
    out: list[str] = []
    for e in events:
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected":
            out.append(getattr(e.drift_detected, "detail", ""))
    return out


def _plan_json(plan_id: str, summary: str, task_ids: list[str]) -> str:
    """Render a minimal plan JSON the LLMPlanner can parse back."""
    return json.dumps(
        {
            "id": plan_id,
            "summary": summary,
            "tasks": [
                {
                    "id": tid,
                    "title": tid.replace("_", " ").title(),
                    "description": f"do {tid}",
                    "assignee_agent_id": "writer",
                }
                for tid in task_ids
            ],
            "edges": [
                {"from_task_id": a, "to_task_id": b}
                for a, b in zip(task_ids, task_ids[1:], strict=False)
            ],
        }
    )


def _linear_static_plan(task_ids: list[str]) -> Plan:
    return Plan(
        id="plan-static",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id=tid, title=tid.title(), assignee_agent_id="writer") for tid in task_ids],
        edges=[
            TaskEdge(from_task_id=a, to_task_id=b)
            for a, b in zip(task_ids, task_ids[1:], strict=False)
        ],
        summary="linear static plan",
    )


async def _drain_acks(channel: ControlChannel, *, count: int, timeout: float = 1.0) -> list[Any]:
    acks: list[Any] = []

    async def _consume() -> None:
        async for a in channel.acks():
            acks.append(a)
            if len(acks) >= count:
                return

    try:
        await asyncio.wait_for(_consume(), timeout=timeout)
    except TimeoutError:
        pass
    return acks


class _PlannerSpy:
    """Wraps a :class:`Planner` so tests can assert on ``refine`` calls."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.refine_calls: list[DriftEvent] = []

    async def generate(self, **kwargs: Any) -> Any:
        return await self._inner.generate(**kwargs)

    async def refine(
        self,
        *,
        plan: Any,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: Any = None,
    ) -> Any:
        self.refine_calls.append(drift)
        return await self._inner.refine(
            plan=plan, drift=drift, goals=goals, observed_actions=observed_actions
        )


# ---------------------------------------------------------------------------
# CANCEL
# ---------------------------------------------------------------------------


async def test_cancel_mid_task_aborts_run() -> None:
    """A CANCEL arriving mid-task ends the run as ``RunAborted``."""

    sink = InMemorySink()
    channel = ControlChannel()

    task_started = asyncio.Event()
    adapter_was_cancelled = asyncio.Event()

    async def _slow_agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = (session, tools)
        task_started.set()
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            adapter_was_cancelled.set()
            raise
        return InvocationResult(task_id=task.id, text="done")

    async def _call_llm(system: str, user: str, model: str) -> str:
        _ = (system, user, model)
        return _plan_json("plan-cancel", "Two tasks.", ["t0", "t1"])

    planner = LLMPlanner(call_llm=_call_llm, model="stub")
    runner = Runner(
        agent=CallableAdapter(_slow_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(max_task_invocations=5),
        goal_deriver=PassthroughGoalDeriver("cancel-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    run_task = asyncio.create_task(runner.run("do something slowly"))

    await asyncio.wait_for(task_started.wait(), timeout=2.0)
    await asyncio.sleep(0.2)
    await channel.send(ControlMessage(kind=ControlKind.CANCEL, payload={"reason": "user aborted"}))

    outcome = await asyncio.wait_for(run_task, timeout=5.0)
    await runner.close()

    assert outcome.success is False
    assert "user aborted" in (outcome.reason or "") or "cancel" in (outcome.reason or "").lower()
    assert adapter_was_cancelled.is_set(), "adapter.invoke should have been cancelled mid-task"

    kinds = _event_kinds(sink.events)
    assert "run_started" in kinds
    assert "run_aborted" in kinds
    assert "run_completed" not in kinds
    # RunAborted precedes only the terminal ConversationEnded (emitted on close).
    aborted_idx = kinds.index("run_aborted")
    later = kinds[aborted_idx + 1 :]
    assert all(k == "conversation_ended" for k in later), kinds

    acks = await _drain_acks(channel, count=1, timeout=1.0)
    assert len(acks) == 1 and acks[0].result == AckResult.SUCCESS


# ---------------------------------------------------------------------------
# STEER — refine + continue
# ---------------------------------------------------------------------------


async def test_steer_mid_task_triggers_refine_and_continues() -> None:
    """A STEER arriving mid-task fires ``planner.refine`` and keeps going."""

    sink = InMemorySink()
    channel = ControlChannel()

    first_task_started = asyncio.Event()
    observed_task_ids: list[str] = []

    async def _agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = (session, tools)
        observed_task_ids.append(task.id)
        if task.id == "slow_task":
            first_task_started.set()
            await asyncio.sleep(30.0)
        await asyncio.sleep(0.01)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    initial_plan_json = _plan_json("plan-init", "Slow initial plan.", ["slow_task"])
    refined_plan_json = _plan_json(
        "plan-steered", "Refined after user steer.", ["fast_a", "fast_b"]
    )
    responses = iter([initial_plan_json, refined_plan_json])

    async def _call_llm(system: str, user: str, model: str) -> str:
        _ = (system, user, model)
        try:
            return next(responses)
        except StopIteration:
            return "{}"

    base_planner = LLMPlanner(call_llm=_call_llm, model="stub")
    planner = _PlannerSpy(base_planner)
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(max_task_invocations=10),
        goal_deriver=PassthroughGoalDeriver("steer-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    run_task = asyncio.create_task(runner.run("start slow"))
    await asyncio.wait_for(first_task_started.wait(), timeout=2.0)
    await asyncio.sleep(0.2)
    await channel.send(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "focus on the two fast follow-ups"},
        )
    )

    outcome = await asyncio.wait_for(run_task, timeout=10.0)
    await runner.close()

    assert planner.refine_calls, (
        "DefaultSteerer should have translated STEER → USER_STEER drift "
        "and called planner.refine exactly once"
    )
    assert any(d.kind == DriftKind.USER_STEER for d in planner.refine_calls), (
        f"expected USER_STEER drift; got {[d.kind.value for d in planner.refine_calls]}"
    )
    assert all(
        d.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL) for d in planner.refine_calls
    )

    kinds = _event_kinds(sink.events)
    assert "plan_revised" in kinds, kinds

    assert "fast_a" in observed_task_ids, observed_task_ids
    assert "fast_b" in observed_task_ids, observed_task_ids

    final_plan = outcome.session.plan
    assert final_plan is not None
    final_ids = {t.id for t in final_plan.tasks}
    assert {"fast_a", "fast_b"}.issubset(final_ids)


# ---------------------------------------------------------------------------
# PAUSE + RESUME
# ---------------------------------------------------------------------------


async def test_pause_blocks_next_task_until_resume() -> None:
    """PAUSE queued before the run starts blocks task execution until RESUME.

    Tests the pre-task control drain's ``paused``-wait loop: the
    executor sees PAUSE at the top of the while loop, enters a blocking
    ``control.receive()`` waiting for a resume / cancel, and releases
    once RESUME arrives.
    """

    sink = InMemorySink()
    channel = ControlChannel()

    started: list[str] = []

    async def _agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = (session, tools)
        started.append(task.id)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    # Pre-queue PAUSE so the first pre-task drain picks it up before
    # any task runs.
    await channel.send(ControlMessage(kind=ControlKind.PAUSE))

    planner = StaticPlanner(_linear_static_plan(["t0", "t1"]))
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(max_task_invocations=8),
        goal_deriver=PassthroughGoalDeriver("pause-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    run_task = asyncio.create_task(runner.run("two tasks"))

    # Give the executor a chance to drain PAUSE and enter paused-wait.
    await asyncio.sleep(0.2)
    assert started == [], "no task should have started while paused"
    assert not run_task.done(), "runner must still be blocked under PAUSE"

    # Now RESUME — executor wakes and walks both tasks to completion.
    await channel.send(ControlMessage(kind=ControlKind.RESUME))
    outcome = await asyncio.wait_for(run_task, timeout=5.0)
    await runner.close()

    assert outcome.success is True
    assert started == ["t0", "t1"], started

    acks = await _drain_acks(channel, count=2, timeout=1.0)
    assert len(acks) == 2
    assert all(a.result == AckResult.SUCCESS for a in acks)


# ---------------------------------------------------------------------------
# REWIND_TO
# ---------------------------------------------------------------------------


async def test_rewind_resets_target_and_downstream_tasks() -> None:
    """REWIND_TO flips the target + downstream tasks back to PENDING."""

    sink = InMemorySink()
    channel = ControlChannel()

    counts: dict[str, int] = {"t0": 0, "t1": 0}

    async def _agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = (session, tools)
        counts[task.id] += 1
        # After t0's first successful run, queue PAUSE + REWIND_TO + RESUME
        # so the executor re-walks t0 and t1 on the pre-task drain.
        if task.id == "t0" and counts["t0"] == 1:
            await channel.send(ControlMessage(kind=ControlKind.PAUSE))
            await channel.send(
                ControlMessage(kind=ControlKind.REWIND_TO, payload={"task_id": "t0"})
            )
            await channel.send(ControlMessage(kind=ControlKind.RESUME))
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    planner = StaticPlanner(_linear_static_plan(["t0", "t1"]))
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(max_task_invocations=12),
        goal_deriver=PassthroughGoalDeriver("rewind-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    outcome = await asyncio.wait_for(runner.run("two tasks"), timeout=5.0)
    await runner.close()

    assert outcome.success is True
    # t0 ran twice (initial + after rewind); t1 ran once (the rewind reset
    # it back to PENDING, and it executed once on the post-resume walk).
    assert counts == {"t0": 2, "t1": 1}, counts

    final_plan = outcome.session.plan
    assert final_plan is not None
    assert all(t.status == TaskStatus.COMPLETED for t in final_plan.tasks)

    # REWIND_TO + RESUME acks are guaranteed; PAUSE may be consumed
    # mid-invoke and not always surface depending on scheduler timing.
    acks = await _drain_acks(channel, count=3, timeout=1.0)
    assert len(acks) >= 2
    assert all(a.result == AckResult.SUCCESS for a in acks)


# ---------------------------------------------------------------------------
# STATUS_QUERY
# ---------------------------------------------------------------------------


async def test_status_query_is_readonly_no_drift_snapshot_in_ack() -> None:
    """STATUS_QUERY is a read-only probe: NO sink events, snapshot in ack.

    Regression guard for the "33k bogus drift_detected events per
    5-minute e2e run" bug: every status_query poll used to produce a
    synthetic ``DriftDetected`` event with ``kind=0`` (UNSPECIFIED),
    crushing harmonograf's DB and frontend rendering. Status snapshots
    now flow back via the ack's ``detail`` field only.
    """

    sink = InMemorySink()
    channel = ControlChannel()

    status_sent = asyncio.Event()

    async def _agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        _ = (session, tools)
        if task.id == "t0":
            # Send STATUS_QUERY mid-task so _invoke_with_control picks it
            # up before the adapter returns. Using a raw string kind is
            # the forward-compat path; STATUS_QUERY isn't a ControlKind
            # member yet.
            await channel.send(
                ControlMessage(kind="STATUS_QUERY", payload={})  # type: ignore[arg-type]
            )
            status_sent.set()
            await asyncio.sleep(0.1)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    planner = StaticPlanner(_linear_static_plan(["t0", "t1"]))
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(max_task_invocations=8),
        goal_deriver=PassthroughGoalDeriver("status-demo"),
        steerer=DefaultSteerer(),
        sinks=[sink],
        control=channel,
    )

    outcome = await asyncio.wait_for(runner.run("two tasks"), timeout=5.0)
    await runner.close()

    assert outcome.success is True
    assert status_sent.is_set()

    # NO drift events should mention status_query. Any drift row tagged
    # with status_query is a regression of the 33k-drift bug.
    details = _drift_details(sink.events)
    offending = [d for d in details if "status_query" in d]
    assert offending == [], (
        f"STATUS_QUERY must not emit drift_detected events; got {offending!r}"
    )

    acks = await _drain_acks(channel, count=1, timeout=1.0)
    assert len(acks) == 1
    assert acks[0].result == AckResult.SUCCESS
    # Snapshot is returned via the ack's detail field.
    assert "status_query" in acks[0].detail
