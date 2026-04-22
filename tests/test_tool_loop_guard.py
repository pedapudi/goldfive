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

    async def mark_task_running(self, task_id: str, *, session: Session, detail: str = "") -> None:
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
        self.transitions.append((task_id, "PROGRESS", {"fraction": fraction, "detail": detail}))
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

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        recoverable: bool = True,
    ) -> None:
        self.transitions.append((task_id, "FAILED", {"reason": reason}))
        task = next(
            (t for t in (session.plan.tasks if session.plan else []) if t.id == task_id),
            None,
        )
        if task is not None and task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            task.status = TaskStatus.FAILED

    async def mark_task_blocked(self, *args: Any, **kwargs: Any) -> None:
        self.transitions.append(("?", "BLOCKED", kwargs))

    async def report_new_work_discovered(self, **kwargs: Any) -> None:
        self.transitions.append(("?", "NEW_WORK", kwargs))

    async def report_plan_divergence(self, **kwargs: Any) -> None:
        self.transitions.append(("?", "DIVERGENCE", kwargs))

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:
        self.drifts.append(drift)


def _session_with_task(task_id: str = "t1", *, running: bool = False) -> Session:
    """Build a two-task Session.

    ``running=True`` pre-sets ``t1`` to ``RUNNING`` for tests that
    exercise handlers which are only legal on a running task (e.g.
    ``report_task_progress``). See the goldfive#201 handler matrix in
    :mod:`goldfive.reporting`.
    """
    status = TaskStatus.RUNNING if running else TaskStatus.PENDING
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id=task_id, title="A", status=status),
            Task(id="t2", title="B"),
        ],
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
        tools,
        "report_task_started",
        {"task_id": "t1", "detail": "kick"},
        session,
        steerer,
    )
    assert first == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.RUNNING

    # Second identical call: should ACK as duplicate, no re-transition.
    second = await invoke_tool(
        tools,
        "report_task_started",
        {"task_id": "t1", "detail": "kick"},
        session,
        steerer,
    )
    assert second == {"acknowledged": True, "duplicate": True}
    # Only one transition recorded — handler was not re-entered.
    assert len(steerer.transitions) == 1


async def test_progress_with_distinct_args_is_not_duplicate() -> None:
    """Varied progress fractions are real updates, not dupes."""
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
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
    """6+/8 identical calls → one CRITICAL LOOPING_TOOL_CALL drift, and
    calls 7-8 are hard-rejected with ``loop_detected`` (new behaviour).

    Pre-hardening, the sixth call fired the one-shot drift and then
    subsequent calls fell through to the handler with a plain
    duplicate ACK — giving the agent no signal to stop. Now every
    call after the flag flips gets the structured ``loop_detected``
    error so a misbehaving agent gets a clear stop signal on EVERY
    further attempt.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [_spec("report_task_progress")]

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    results: list[dict[str, Any]] = []
    for _ in range(8):
        results.append(await invoke_tool(tools, "report_task_progress", args, session, steerer))

    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_TOOL_CALL
    assert drift.current_task_id == "t1"
    assert "report_task_progress" in drift.detail

    # Call 1 = fresh ACK; calls 2..5 = duplicate ACK; call 6 triggers
    # the drift and is itself rejected; calls 7-8 are rejected without
    # re-firing drift.
    assert results[0] == {"acknowledged": True}
    for r in results[1:5]:
        assert r == {"acknowledged": True, "duplicate": True}
    for r in results[5:]:
        assert r["acknowledged"] is False
        assert r["error"] == "loop_detected"
        assert r["tool"] == "report_task_progress"
        assert r["scope"] == "per_task"


async def test_loop_drift_does_not_re_fire_after_first_emission() -> None:
    """One-shot guard: 16 identical calls still emit only one drift."""
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [_spec("report_task_progress")]

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    for _ in range(16):
        await invoke_tool(tools, "report_task_progress", args, session, steerer)

    assert len(steerer.drifts) == 1


async def test_alternating_args_does_not_fire_drift() -> None:
    """Eight calls split between two distinct payloads keep us under threshold."""
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
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


async def test_volume_cap_fires_on_args_varying_loop() -> None:
    """The volume cap catches loops that vary args on every call.

    Models in the wild often vary a free-text ``detail`` string on every
    repeated progress report, which keeps every signature unique and
    lets the exact-match window sail past without firing. The per-tool
    cumulative counter catches this regardless of args.

    Uses a non-terminal tool (``report_task_progress``) so the
    terminal-task rejection layer doesn't short-circuit the flow — this
    test is specifically for the volume-cap safety net when the task
    itself hasn't transitioned out of ``RUNNING``.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [_spec("report_task_progress")]

    # 14 calls with fresh details should stay below the volume cap.
    for i in range(14):
        await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": 0.01 * i, "detail": f"try #{i}"},
            session,
            steerer,
        )
    assert steerer.drifts == []

    # The 15th call crosses the threshold and fires exactly one drift.
    await invoke_tool(
        tools,
        "report_task_progress",
        {"task_id": "t1", "fraction": 0.14, "detail": "try #14"},
        session,
        steerer,
    )
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_TOOL_CALL
    assert drift.current_task_id == "t1"
    assert "report_task_progress" in drift.detail


async def test_volume_cap_is_per_tool_not_cross_tool() -> None:
    """Volume cap counts per tool name; mixing tools stays under the cap."""
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [
        _spec("report_task_progress"),
        _spec("report_task_blocked"),
    ]

    # 14 progress + 14 blocked = 28 total, but neither tool alone crosses 15.
    for i in range(14):
        await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": 0.1 * i, "detail": f"p{i}"},
            session,
            steerer,
        )
        await invoke_tool(
            tools,
            "report_task_blocked",
            {"task_id": "t1", "blocked_on": f"dep-{i}"},
            session,
            steerer,
        )
    assert steerer.drifts == []


async def test_volume_cap_fires_once_per_task() -> None:
    """Once the volume cap fires, further calls do not re-fire drift."""
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [_spec("report_task_progress")]

    for i in range(30):
        await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": "t1", "fraction": 0.01 * i, "detail": f"d{i}"},
            session,
            steerer,
        )
    assert len(steerer.drifts) == 1


# ---------------------------------------------------------------------------
# Terminal-task rejection (prevention layer)
# ---------------------------------------------------------------------------


async def test_reporting_on_terminal_task_returns_structured_rejection() -> None:
    """Terminal-task retry semantics (goldfive#201).

    Same-transition retries (e.g. ``report_task_failed`` on an already
    FAILED task) come back as an idempotent ACK —
    ``{"acknowledged": True, "idempotent": True}`` — so a confused
    model's innocent retry no longer masquerades as a tool-loop signal
    nor triggers a spurious plan revision.

    Cross-transitions on a terminal task (e.g. ``report_task_progress``
    on a FAILED task) come back as ``invalid_transition`` — a real
    "agent is confused about state" signal the dispatcher should
    surface rather than absorb.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_failed"), _spec("report_task_progress")]

    # First report: legitimate transition to FAILED.
    first = await invoke_tool(
        tools,
        "report_task_failed",
        {"task_id": "t1", "reason": "it broke"},
        session,
        steerer,
    )
    assert first == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.FAILED

    # Second report (same transition on the now-terminal task) is an
    # idempotent no-op ack — handler does NOT re-run the steerer.
    second = await invoke_tool(
        tools,
        "report_task_failed",
        {"task_id": "t1", "reason": "it broke again"},
        session,
        steerer,
    )
    assert second["acknowledged"] is True
    assert second["idempotent"] is True
    assert second["current_status"] == "FAILED"
    # Only the first call produced a transition; retries did not.
    assert [t[1] for t in steerer.transitions] == ["FAILED"]

    # A different reporting tool on the same terminal task is a real
    # invalid-transition signal — not absorbed as idempotent.
    third = await invoke_tool(
        tools,
        "report_task_progress",
        {"task_id": "t1", "fraction": 0.5, "detail": "still trying"},
        session,
        steerer,
    )
    assert third["acknowledged"] is False
    assert third["error"] == "invalid_transition"
    assert third["current_status"] == "FAILED"
    assert third["attempted"] == "RUNNING"


async def test_terminal_idempotent_retry_does_not_invoke_handler() -> None:
    """An idempotent retry does not re-enter the Steerer transition table."""
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_completed")]

    # Pre-mark the task as COMPLETED (e.g., via a prior legitimate call).
    session.plan.tasks[0].status = TaskStatus.COMPLETED

    result = await invoke_tool(
        tools,
        "report_task_completed",
        {"task_id": "t1", "summary": "duplicate report"},
        session,
        steerer,
    )
    assert result["acknowledged"] is True
    assert result["idempotent"] is True
    # Steerer never saw the call.
    assert steerer.transitions == []
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


async def test_terminal_idempotent_retries_do_not_drive_steerer() -> None:
    """A modest run of same-transition retries stays cheap.

    A confused model that keeps calling ``report_task_failed`` on an
    already-FAILED task must not (a) drive the steerer repeatedly
    or (b) emit drift events for the repeat. Each call returns a
    cheap idempotent ACK. (A runaway flood beyond the per-task
    volume cap will still trip the loop detector — but that is a
    separate, legitimate signal handled by the loop guard.)
    """
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_failed")]

    # Mark terminal up front.
    session.plan.tasks[0].status = TaskStatus.FAILED

    # Stay under the per-task volume cap so the loop detector doesn't
    # fire — the explicit goal here is that idempotent retries do NOT
    # themselves trigger loop drift for reasonable repeat counts.
    for i in range(10):
        result = await invoke_tool(
            tools,
            "report_task_failed",
            {"task_id": "t1", "reason": f"unique reason #{i}"},
            session,
            steerer,
        )
        assert result["acknowledged"] is True
        assert result.get("idempotent") is True

    # No handlers fired, no drift events emitted — retries are cheap.
    assert steerer.transitions == []
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

    await invoke_tool(tools, "report_task_started", {"task_id": "t1"}, session, steerer)
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
# Schema rejections (Layer 1) and hardening against adversarial agents
# ---------------------------------------------------------------------------


async def test_missing_task_id_is_rejected() -> None:
    """A task-scoped reporting call with no ``task_id`` must be rejected
    with the structured ``missing_task_id`` error, and MUST NOT reach
    the underlying handler.

    Pre-hardening, ``invoke_tool`` silently skipped the terminal-task
    layer whenever ``task_id`` was empty and then passed the call
    straight through to the handler — which would quietly do nothing
    (the handler's own ``if not task_id`` short-circuit kicks in). The
    agent received a bland ``{"acknowledged": True}`` back and
    cheerfully kept hammering the tool.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task()
    spec_template = _spec("report_task_failed")

    # Spec the handler to raise so a regression immediately fails this
    # test — if the rejection is skipped, we hit this handler.
    async def _boom(args, session, steerer):
        raise AssertionError("handler must not run when task_id is missing")

    spec = ReportingToolSpec(
        name=spec_template.name,
        description=spec_template.description,
        parameters=spec_template.parameters,
        handler=_boom,
    )

    # Try a variety of "no task_id" shapes to cover every bypass route.
    for bad_args in (
        {},
        {"task_id": ""},
        {"task_id": "   "},  # whitespace → stripped to empty
        {"task_id": None, "reason": "x"},
        {"reason": "x"},
    ):
        result = await invoke_tool([spec], "report_task_failed", bad_args, session, steerer)
        assert result["acknowledged"] is False
        assert result["error"] == "missing_task_id"
        assert result["tool"] == "report_task_failed"
        assert "task_id" in result["message"].lower()

    # No drift fires for this class of error — it's a schema problem,
    # not a loop.
    assert steerer.drifts == []
    assert steerer.transitions == []


async def test_unknown_task_id_is_rejected() -> None:
    """A reporting call that names a ``task_id`` not present in the plan
    must be rejected with ``unknown_task_id``; handler must not run.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task()

    async def _boom(args, session, steerer):
        raise AssertionError("handler must not run when task_id is unknown")

    template = _spec("report_task_failed")
    spec = ReportingToolSpec(
        name=template.name,
        description=template.description,
        parameters=template.parameters,
        handler=_boom,
    )

    result = await invoke_tool(
        [spec],
        "report_task_failed",
        {"task_id": "bogus_does_not_exist", "reason": "lost in space"},
        session,
        steerer,
    )
    assert result["acknowledged"] is False
    assert result["error"] == "unknown_task_id"
    assert result["task_id"] == "bogus_does_not_exist"
    assert result["tool"] == "report_task_failed"
    # No drift — it's a schema problem, not a loop.
    assert steerer.drifts == []
    assert steerer.transitions == []


async def test_plan_level_tools_still_work_without_task_id() -> None:
    """``report_plan_divergence`` is plan-level and MUST NOT be
    rejected for missing task_id — the Layer 1 schema check is
    scoped to task-level tools only.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_plan_divergence")]

    result = await invoke_tool(
        tools,
        "report_plan_divergence",
        {"note": "the plan drifted", "suggested_action": "replan"},
        session,
        steerer,
    )
    # The plan-level handler runs successfully → plain ACK.
    assert result == {"acknowledged": True}
    assert any(t[1] == "DIVERGENCE" for t in steerer.transitions)


async def test_loop_flagged_short_circuits_subsequent_calls() -> None:
    """After the per-task loop guard trips, every subsequent call to
    the SAME tool on the SAME task is hard-rejected with
    ``loop_detected``.

    This is the direct fix for the live-run failure: pre-hardening,
    ``detect_loop`` set ``loop_flagged=True`` on the first crossing
    and returned False forever after — so the dispatcher fell through
    to the idempotency check (which misses when args vary) and on to
    the handler. With the fix, the ``(task, tool)`` bucket stays
    hard-rejecting until refine resets it.
    """
    steerer = _RecordingSteerer()
    session = _session_with_task(running=True)
    tools = [_spec("report_task_progress")]

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    results: list[dict[str, Any]] = []
    for _ in range(20):
        results.append(await invoke_tool(tools, "report_task_progress", args, session, steerer))

    # Drift fires exactly once — the bucket is flagged on call 6 and
    # every call after that is hard-rejected without re-firing.
    assert len(steerer.drifts) == 1

    # Handler is invoked EXACTLY once (first call); the rest are a
    # mix of duplicate ACKs and hard rejections — none reach the
    # handler / steerer transition.
    progress_transitions = [t for t in steerer.transitions if t[1] == "PROGRESS"]
    assert len(progress_transitions) == 1

    # Call 1 → fresh ACK.
    assert results[0] == {"acknowledged": True}
    # Calls 2..5 → duplicate ACK (under threshold, seen in table).
    for r in results[1:5]:
        assert r == {"acknowledged": True, "duplicate": True}
    # Calls 6..20 → all hard-rejected with ``loop_detected``.
    for r in results[5:]:
        assert r["acknowledged"] is False
        assert r["error"] == "loop_detected"
        assert r["task_id"] == "t1"
        assert r["tool"] == "report_task_progress"
        assert r["scope"] == "per_task"


async def test_session_wide_volume_cap_fires_on_varying_task_ids() -> None:
    """An adversarial agent that invents a FRESH task_id on every
    call distributes 1 call per per-task bucket — the per-task volume
    cap (15) never fires. The session-wide cap (50) is the safety net
    for this pattern.

    This is the exact live-run failure scenario (237 calls to
    ``report_task_failed`` that all returned plain ACK).
    """
    steerer = _RecordingSteerer()

    # Build a plan with 60 pre-existing task ids so ``unknown_task_id``
    # doesn't short-circuit before the session counter increments.
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=f"t{i}", title=f"task-{i}") for i in range(60)],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=plan,
    )
    tools = [_spec("report_task_failed")]

    results: list[dict[str, Any]] = []
    for i in range(60):
        results.append(
            await invoke_tool(
                tools,
                "report_task_failed",
                {"task_id": f"t{i}", "reason": f"fresh reason #{i}"},
                session,
                steerer,
            )
        )

    # Exactly one drift — the session-wide cap fired on call 50 and
    # did not re-fire afterwards.
    assert len(steerer.drifts) == 1
    assert steerer.drifts[0].kind is DriftKind.LOOPING_TOOL_CALL
    assert "session-wide" in steerer.drifts[0].detail.lower()

    # Calls 1..49 → handler runs (marks each distinct task as FAILED).
    failed_transitions = [t for t in steerer.transitions if t[1] == "FAILED"]
    assert len(failed_transitions) == 49

    # Calls 50..60 → all rejected with loop_detected (scope=session).
    for r in results[:49]:
        assert r == {"acknowledged": True}
    for r in results[49:]:
        assert r["acknowledged"] is False
        assert r["error"] == "loop_detected"
        assert r["scope"] == "session"
        assert r["tool"] == "report_task_failed"


async def test_session_wide_cap_is_per_tool_not_cross_tool() -> None:
    """The session-wide cap counts per tool name. 30 progress + 30
    blocked = 60 total calls, but neither tool alone crosses the 50
    threshold, so nothing is flagged.
    """
    steerer = _RecordingSteerer()

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        # Start each task RUNNING so ``report_task_progress`` is a
        # legal transition at the handler layer (goldfive#201).
        tasks=[Task(id=f"t{i}", title=f"task-{i}", status=TaskStatus.RUNNING) for i in range(30)],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=plan,
    )
    tools = [_spec("report_task_progress"), _spec("report_task_blocked")]

    for i in range(30):
        r = await invoke_tool(
            tools,
            "report_task_progress",
            {"task_id": f"t{i}", "fraction": 0.1, "detail": f"p{i}"},
            session,
            steerer,
        )
        assert r == {"acknowledged": True}
        r = await invoke_tool(
            tools,
            "report_task_blocked",
            {"task_id": f"t{i}", "blocked_on": f"dep-{i}"},
            session,
            steerer,
        )
        assert r == {"acknowledged": True}

    # Neither tool crossed its own 50-call threshold → no drift.
    assert steerer.drifts == []


async def test_missing_task_id_does_not_count_toward_session_cap() -> None:
    """Rejections at the missing/unknown task_id layer MUST happen
    BEFORE the session counter increments — otherwise an adversarial
    spam of malformed calls would poison the session counter and
    trigger a false session-wide drift against the next legitimate
    call.
    """
    from goldfive.adapters._tool_loop_guard import guard_for

    steerer = _RecordingSteerer()
    session = _session_with_task()
    tools = [_spec("report_task_failed")]

    # 200 calls with no task_id — way above the session threshold.
    for _ in range(200):
        r = await invoke_tool(tools, "report_task_failed", {}, session, steerer)
        assert r["error"] == "missing_task_id"

    # 200 calls with unknown task_id.
    for _ in range(200):
        r = await invoke_tool(
            tools,
            "report_task_failed",
            {"task_id": "no_such_task", "reason": "spam"},
            session,
            steerer,
        )
        assert r["error"] == "unknown_task_id"

    # Neither path touched the session counter.
    guard = guard_for(session)
    assert guard.session_tool_count.get("report_task_failed", 0) == 0
    assert "report_task_failed" not in guard.session_tool_flagged
    # No drift was emitted for schema errors.
    assert steerer.drifts == []


async def test_session_wide_cap_exempts_awaiting_approval() -> None:
    """``report_awaiting_approval`` polls by design — the session-wide
    cap must not fire for it no matter how many times it's invoked.
    """
    from goldfive.adapters._tool_loop_guard import guard_for

    steerer = _RecordingSteerer()
    session = _session_with_task()

    # ``report_awaiting_approval`` is a builtin that blocks on a
    # control-channel waiter, which we don't want to drive in this
    # test — swap in a stub spec that just ACKs.
    async def _ack_handler(args, session, steerer):
        return {"acknowledged": True}

    template = _spec("report_awaiting_approval")
    spec = ReportingToolSpec(
        name=template.name,
        description=template.description,
        parameters=template.parameters,
        handler=_ack_handler,
    )

    # 200 polls with varying prompts — well above the session cap.
    for i in range(200):
        r = await invoke_tool(
            [spec],
            "report_awaiting_approval",
            {"task_id": "t1", "prompt": f"check #{i}"},
            session,
            steerer,
        )
        assert r == {"acknowledged": True}

    # The counter does tick (for observability) but the flag never trips.
    guard = guard_for(session)
    assert "report_awaiting_approval" not in guard.session_tool_flagged
    assert steerer.drifts == []


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
    revised = await planner.refine(plan=plan, drift=drift, goals=[Goal(id="g1", summary="do it")])
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
    revised = await planner.refine(plan=plan, drift=drift, goals=[Goal(id="g1", summary="do it")])
    assert revised is not None
    by_id = {t.id: t for t in revised.tasks}
    # The planner forgot to mark it FAILED; the framework forces it.
    assert by_id["loop"].status is TaskStatus.FAILED
    assert by_id["fresh"].status is TaskStatus.PENDING
    assert revised.revision_kind == DriftKind.LOOPING_TOOL_CALL.value
