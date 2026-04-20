"""Tests for the per-task tool-call idempotency table + loop detector.

Covers the four behaviours added in fix/reporting-tool-loop-guard:

* Duplicate calls are ACK'd as ``duplicate=True`` and skip the handler.
* A sustained burst of identical calls fires a ``LOOPING_TOOL_CALL``
  drift exactly once per task.
* Varied arg payloads on the same tool do NOT trip the loop detector
  (frontier-style progress pings stay healthy).
* Duplicate ``report_task_started`` calls do not re-enter the Steerer
  transition (idempotency works at the dispatch boundary, not just
  the handler's status guard).
"""

from __future__ import annotations

from typing import Any

from goldfive.adapters._tool_invocation import invoke_tool
from goldfive.reporting import BUILTIN_REPORTING_TOOLS, ReportingToolSpec
from goldfive.types import (
    DriftEvent,
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


def _spec(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"missing builtin tool: {name!r}")


class _RecordingSteerer:
    """Captures every ``mark_task_*`` call and drift dispatch."""

    def __init__(self) -> None:
        self.transitions: list[tuple[str, str, dict[str, Any]]] = []
        self.drifts: list[DriftEvent] = []
        self._sinks: list[Any] = []

    async def mark_task_running(
        self, task_id: str, *, session: Session, detail: str = ""
    ) -> None:
        self.transitions.append((task_id, "RUNNING", {"detail": detail}))
        task = next(
            (t for t in (session.plan.tasks if session.plan else []) if t.id == task_id),
            None,
        )
        if task is not None and task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            task.status = TaskStatus.RUNNING

    async def mark_task_progress(
        self,
        task_id: str,
        *,
        session: Session,
        fraction: float = 0.0,
        detail: str = "",
    ) -> None:
        self.transitions.append(
            (task_id, "PROGRESS", {"fraction": fraction, "detail": detail})
        )
        session.task_progress[task_id] = fraction

    async def mark_task_completed(
        self,
        task_id: str,
        *,
        session: Session,
        summary: str = "",
        artifacts: dict[str, str] | None = None,
    ) -> None:
        self.transitions.append((task_id, "COMPLETED", {"summary": summary}))
        task = next(
            (t for t in (session.plan.tasks if session.plan else []) if t.id == task_id),
            None,
        )
        if task is not None:
            task.status = TaskStatus.COMPLETED

    async def mark_task_failed(self, *args: Any, **kwargs: Any) -> None:
        self.transitions.append(("?", "FAILED", kwargs))

    async def mark_task_blocked(self, *args: Any, **kwargs: Any) -> None:
        self.transitions.append(("?", "BLOCKED", kwargs))

    async def report_new_work_discovered(self, **kwargs: Any) -> None:
        self.transitions.append(("?", "NEW_WORK", kwargs))

    async def report_plan_divergence(self, **kwargs: Any) -> None:
        self.transitions.append(("?", "DIVERGENCE", kwargs))

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:
        self.drifts.append(drift)


def _session_with_task(task_id: str = "t1") -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title="A"), Task(id="t2", title="B")],
        edges=[TaskEdge(from_task_id=task_id, to_task_id="t2")],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=plan,
    )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_duplicate_call_returns_ack_and_skips_steerer() -> None:
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_started")]

    first = await invoke_tool(
        tools, "report_task_started", {"task_id": "t1", "detail": "kick"},
        session, steerer,
    )
    assert first == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.RUNNING

    # Second identical call: should ACK as duplicate, no re-transition.
    second = await invoke_tool(
        tools, "report_task_started", {"task_id": "t1", "detail": "kick"},
        session, steerer,
    )
    assert second == {"acknowledged": True, "duplicate": True}
    # Only one transition recorded — handler was not re-entered.
    assert len(steerer.transitions) == 1


async def test_progress_with_distinct_args_is_not_duplicate() -> None:
    """Varied progress fractions are real updates, not dupes."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_progress")]

    for frac, detail in [(0.2, "early"), (0.5, "midway"), (0.9, "almost")]:
        result = await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": frac, "detail": detail},
            session,
            steerer,
        )
        assert result == {"acknowledged": True}

    # All three reached the steerer — none collapsed as duplicates.
    progresses = [t for t in steerer.transitions if t[1] == "PROGRESS"]
    assert len(progresses) == 3
    assert {p[2]["fraction"] for p in progresses} == {0.2, 0.5, 0.9}
    assert steerer.drifts == []  # Varied calls don't trip the loop guard.


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


async def test_eight_identical_calls_fires_drift_exactly_once() -> None:
    """6+/8 identical calls → one CRITICAL LOOPING_TOOL_CALL drift."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_progress")]

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    for _ in range(8):
        await invoke_tool(tools, "report_task_progress", args, session, steerer)

    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_TOOL_CALL
    assert drift.current_task_id == "t1"
    assert "report_task_progress" in drift.detail


async def test_loop_drift_does_not_re_fire_after_first_emission() -> None:
    """One-shot guard: 16 identical calls still emit only one drift."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_progress")]

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    for _ in range(16):
        await invoke_tool(tools, "report_task_progress", args, session, steerer)

    assert len(steerer.drifts) == 1


async def test_alternating_args_does_not_fire_drift() -> None:
    """Eight calls split between two distinct payloads keep us under threshold."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_progress")]

    for i in range(8):
        await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": (i % 2) * 0.5, "detail": f"step-{i}"},
            session,
            steerer,
        )

    assert steerer.drifts == []


async def test_frontier_style_progression_does_not_fire_drift() -> None:
    """Realistic 0.2/0.5/0.9 + completion sequence stays clean."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [
        _spec("report_task_started"),
        _spec("report_task_progress"),
        _spec("report_task_completed"),
    ]

    await invoke_tool(
        tools, "report_task_started", {"task_id": "t1"}, session, steerer
    )
    for frac, detail in [(0.2, "research"), (0.5, "draft"), (0.9, "review")]:
        await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": frac, "detail": detail},
            session,
            steerer,
        )
    await invoke_tool(
        tools,
        "report_task_completed",
        {"task_id": "t1", "summary": "delivered"},
        session,
        steerer,
    )

    assert steerer.drifts == []
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# LLMPlanner.refine handling for LOOPING_TOOL_CALL
# ---------------------------------------------------------------------------


async def test_llmplanner_refine_loops_drift_falls_back_when_llm_fails() -> None:
    """A LOOPING_TOOL_CALL drift forces the looping task to FAILED.

    Even when the LLM raises, the deterministic fallback produces a
    plan whose looping task is FAILED so the executor can route around
    it.
    """
    from goldfive.planner import LLMPlanner
    from goldfive.types import DriftSeverity

    async def boom(system: str, user: str, model: str) -> str:
        raise RuntimeError("LLM dead")

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="done", title="prep", status=TaskStatus.COMPLETED),
            Task(id="loop", title="stuck", status=TaskStatus.RUNNING),
            Task(id="next", title="rest", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="loop", to_task_id="next")],
    )
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.CRITICAL,
        detail="task loop kept calling 'report_task_progress'",
        current_task_id="loop",
    )
    planner = LLMPlanner(call_llm=boom)
    revised = await planner.refine(
        plan=plan, drift=drift, goals=[Goal(id="g1", summary="do it")]
    )
    assert revised is not None
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["loop"].status is TaskStatus.FAILED
    assert by_id["done"].status is TaskStatus.COMPLETED
    assert revised.revision_kind == DriftKind.LOOPING_TOOL_CALL.value
    assert revised.revision_index == plan.revision_index + 1


async def test_llmplanner_refine_loops_drift_uses_llm_response() -> None:
    """If the LLM returns a usable plan, the looper is forced FAILED."""
    import json as _json

    from goldfive.planner import LLMPlanner
    from goldfive.types import DriftSeverity

    response = {
        "summary": "route around loop",
        "tasks": [
            {"id": "done", "title": "prep", "status": "COMPLETED"},
            {"id": "loop", "title": "stuck", "status": "PENDING"},  # planner forgot
            {"id": "fresh", "title": "alt route", "status": "PENDING"},
        ],
        "edges": [{"from_task_id": "done", "to_task_id": "fresh"}],
    }

    async def llm(system: str, user: str, model: str) -> str:
        return _json.dumps(response)

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="done", title="prep", status=TaskStatus.COMPLETED),
            Task(id="loop", title="stuck", status=TaskStatus.RUNNING),
        ],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.CRITICAL,
        detail="loop drift",
        current_task_id="loop",
    )
    planner = LLMPlanner(call_llm=llm)
    revised = await planner.refine(
        plan=plan, drift=drift, goals=[Goal(id="g1", summary="do it")]
    )
    assert revised is not None
    by_id = {t.id: t for t in revised.tasks}
    # The planner forgot to mark it FAILED; the framework forces it.
    assert by_id["loop"].status is TaskStatus.FAILED
    assert by_id["fresh"].status is TaskStatus.PENDING
    assert revised.revision_kind == DriftKind.LOOPING_TOOL_CALL.value
