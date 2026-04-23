"""Idempotency + invalid-transition tests for reporting handlers.

Covers the Bug A half of goldfive#201 — per-handler idempotency matrix.
See the module docstring in :mod:`goldfive.reporting` for the table.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.reporting import BUILTIN_REPORTING_TOOLS, ReportingToolSpec  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


def _tool(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"missing builtin tool {name!r}")


def _fresh(
    task_status: TaskStatus = TaskStatus.PENDING,
) -> tuple[DefaultSteerer, Session, _ListSink]:
    sink = _ListSink()
    planner = _StubPlanner()
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="A",
                    assignee_agent_id="worker",
                    status=task_status,
                ),
            ],
            edges=[],
        ),
    )
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink


# ---------------------------------------------------------------------------
# Happy-path regressions (retry on same terminal → idempotent)
# ---------------------------------------------------------------------------


async def test_report_task_completed_on_already_completed_returns_idempotent_ack() -> None:
    """Calling report_task_completed twice: first transitions, second is
    idempotent ACK. No second steerer call, no second task_completed
    event."""
    steerer, session, sink = _fresh(task_status=TaskStatus.RUNNING)

    first = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )
    assert first == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    completed_events_after_first = sum(
        1 for e in sink.events if e.WhichOneof("payload") == "task_completed"
    )

    second = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done (again)"}, session, steerer
    )
    assert second == {
        "acknowledged": True,
        "idempotent": True,
        "current_status": "COMPLETED",
    }
    # No additional task_completed event from the retry.
    assert (
        sum(1 for e in sink.events if e.WhichOneof("payload") == "task_completed")
        == completed_events_after_first
    )
    assert session.completed_results["t1"] == "done"  # summary preserved


async def test_report_task_completed_on_running_actually_transitions() -> None:
    """First call (on RUNNING) transitions and returns ``acknowledged=True``
    without the idempotent flag."""
    steerer, session, _ = _fresh(task_status=TaskStatus.RUNNING)

    result = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "all done"}, session, steerer
    )
    assert result == {"acknowledged": True}
    assert "idempotent" not in result
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


async def test_report_task_completed_on_pending_also_transitions() -> None:
    """PENDING → COMPLETED is a legal (if unusual) skip-ahead — real
    transition, not idempotent."""
    steerer, session, _ = _fresh(task_status=TaskStatus.PENDING)

    result = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "zero-work complete"}, session, steerer
    )
    assert result == {"acknowledged": True}
    assert "idempotent" not in result
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Invalid-transition shape
# ---------------------------------------------------------------------------


async def test_report_task_started_on_completed_returns_invalid_transition() -> None:
    steerer, session, _ = _fresh(task_status=TaskStatus.COMPLETED)

    result = await _tool("report_task_started").handler(
        {"task_id": "t1", "detail": "confused"}, session, steerer
    )

    assert result["acknowledged"] is False
    assert result["error"] == "invalid_transition"
    assert result["tool"] == "report_task_started"
    assert result["task_id"] == "t1"
    assert result["current_status"] == "COMPLETED"
    assert result["attempted"] == "RUNNING"
    assert "message" in result
    # Task state unchanged.
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


async def test_report_task_failed_on_completed_returns_invalid_transition() -> None:
    steerer, session, _ = _fresh(task_status=TaskStatus.COMPLETED)

    result = await _tool("report_task_failed").handler(
        {"task_id": "t1", "reason": "late report"}, session, steerer
    )
    assert result["acknowledged"] is False
    assert result["error"] == "invalid_transition"
    assert result["current_status"] == "COMPLETED"
    assert result["attempted"] == "FAILED"
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


async def test_report_task_progress_on_completed_returns_invalid_transition() -> None:
    steerer, session, _ = _fresh(task_status=TaskStatus.COMPLETED)

    result = await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.5}, session, steerer
    )
    assert result["acknowledged"] is False
    assert result["error"] == "invalid_transition"
    assert result["current_status"] == "COMPLETED"


async def test_report_task_progress_on_pending_returns_invalid_transition() -> None:
    """Progress ticks are only valid on RUNNING tasks."""
    steerer, session, _ = _fresh(task_status=TaskStatus.PENDING)

    result = await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.5}, session, steerer
    )
    assert result["acknowledged"] is False
    assert result["error"] == "invalid_transition"
    assert result["current_status"] == "PENDING"


# ---------------------------------------------------------------------------
# Parameterised idempotency matrix
# ---------------------------------------------------------------------------


# Each row is: (tool_name, extra_args, initial_status, expected_kind,
# expected_current_status_in_response). ``expected_kind`` is one of
# ``"idempotent"``, ``"invalid"``, ``"transition"``.
_MATRIX: list[tuple[str, dict[str, Any], TaskStatus, str, str]] = [
    # report_task_started: RUNNING → idempotent; terminal → invalid
    ("report_task_started", {}, TaskStatus.RUNNING, "idempotent", "RUNNING"),
    ("report_task_started", {}, TaskStatus.COMPLETED, "invalid", "COMPLETED"),
    ("report_task_started", {}, TaskStatus.FAILED, "invalid", "FAILED"),
    ("report_task_started", {}, TaskStatus.CANCELLED, "invalid", "CANCELLED"),
    ("report_task_started", {}, TaskStatus.NOT_NEEDED, "invalid", "NOT_NEEDED"),
    ("report_task_started", {}, TaskStatus.PENDING, "transition", "PENDING"),
    # report_task_progress: RUNNING → legal (treated as ok); terminal → invalid
    ("report_task_progress", {"fraction": 0.5}, TaskStatus.COMPLETED, "invalid", "COMPLETED"),
    ("report_task_progress", {"fraction": 0.5}, TaskStatus.PENDING, "invalid", "PENDING"),
    ("report_task_progress", {"fraction": 0.5}, TaskStatus.RUNNING, "transition", "RUNNING"),
    # report_task_completed: COMPLETED → idempotent; other terminal → invalid
    ("report_task_completed", {"summary": "x"}, TaskStatus.COMPLETED, "idempotent", "COMPLETED"),
    ("report_task_completed", {"summary": "x"}, TaskStatus.FAILED, "invalid", "FAILED"),
    ("report_task_completed", {"summary": "x"}, TaskStatus.CANCELLED, "invalid", "CANCELLED"),
    ("report_task_completed", {"summary": "x"}, TaskStatus.NOT_NEEDED, "invalid", "NOT_NEEDED"),
    ("report_task_completed", {"summary": "x"}, TaskStatus.RUNNING, "transition", "RUNNING"),
    # report_task_failed: FAILED → idempotent; COMPLETED/CANCELLED/NOT_NEEDED → invalid
    ("report_task_failed", {"reason": "x"}, TaskStatus.FAILED, "idempotent", "FAILED"),
    ("report_task_failed", {"reason": "x"}, TaskStatus.COMPLETED, "invalid", "COMPLETED"),
    ("report_task_failed", {"reason": "x"}, TaskStatus.CANCELLED, "invalid", "CANCELLED"),
    ("report_task_failed", {"reason": "x"}, TaskStatus.NOT_NEEDED, "invalid", "NOT_NEEDED"),
    ("report_task_failed", {"reason": "x"}, TaskStatus.RUNNING, "transition", "RUNNING"),
    # report_task_blocked: BLOCKED → idempotent; terminal → invalid
    ("report_task_blocked", {"blocker": "x"}, TaskStatus.BLOCKED, "idempotent", "BLOCKED"),
    ("report_task_blocked", {"blocker": "x"}, TaskStatus.COMPLETED, "invalid", "COMPLETED"),
    ("report_task_blocked", {"blocker": "x"}, TaskStatus.CANCELLED, "invalid", "CANCELLED"),
    ("report_task_blocked", {"blocker": "x"}, TaskStatus.RUNNING, "transition", "RUNNING"),
]


@pytest.mark.parametrize(
    "tool_name,extra_args,initial_status,expected_kind,expected_status",
    _MATRIX,
)
async def test_idempotency_matrix(
    tool_name: str,
    extra_args: dict[str, Any],
    initial_status: TaskStatus,
    expected_kind: str,
    expected_status: str,
) -> None:
    steerer, session, _ = _fresh(task_status=initial_status)
    args: dict[str, Any] = {"task_id": "t1", **extra_args}
    result = await _tool(tool_name).handler(args, session, steerer)

    if expected_kind == "idempotent":
        assert result["acknowledged"] is True, (tool_name, initial_status, result)
        assert result.get("idempotent") is True, (tool_name, initial_status, result)
        assert result["current_status"] == expected_status
        # Status unchanged.
        assert session.plan.tasks[0].status is initial_status
    elif expected_kind == "invalid":
        assert result["acknowledged"] is False, (tool_name, initial_status, result)
        assert result["error"] == "invalid_transition", (tool_name, initial_status, result)
        assert result["current_status"] == expected_status
        assert result["attempted"]  # non-empty
        assert session.plan.tasks[0].status is initial_status
    else:  # transition
        # Handler returns plain ACK when a real transition happens.
        assert result == {"acknowledged": True}


# ---------------------------------------------------------------------------
# Regression: missing_task_id shape preserved when no resolvable id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,extra_args",
    [
        ("report_task_started", {}),
        ("report_task_progress", {"fraction": 0.5}),
        ("report_task_completed", {"summary": "x"}),
        ("report_task_failed", {"reason": "x"}),
        ("report_task_blocked", {"blocker": "x"}),
        ("report_awaiting_approval", {"prompt": "ok?"}),
    ],
)
async def test_missing_task_id_error_shape_preserved(
    tool_name: str,
    extra_args: dict[str, Any],
) -> None:
    """Regression guard: the canonical ``missing_task_id`` shape still
    fires for the "no resolvable task_id" path. The new idempotency
    checks must not mask this error."""
    steerer, session, _ = _fresh(task_status=TaskStatus.PENDING)
    # Clear any state fallback.
    session.state.pop("goldfive.current_task_id", None)

    result = await _tool(tool_name).handler(dict(extra_args), session, steerer)

    assert result["acknowledged"] is False, (tool_name, result)
    assert result["error"] == "missing_task_id", (tool_name, result)
    assert result["tool"] == tool_name


# ---------------------------------------------------------------------------
# Unknown task_id stays an ACK (preserves existing handler tolerance)
# ---------------------------------------------------------------------------


async def test_unknown_task_id_still_acks_at_handler_level() -> None:
    """Existing invariant: when the handler is called directly (i.e., not
    through ``invoke_tool``'s schema layer) with a bogus task_id, it
    ACKs without raising. The idempotency check must short-circuit
    cleanly when ``task`` is None."""
    steerer, session, _ = _fresh(task_status=TaskStatus.PENDING)
    result = await _tool("report_task_started").handler(
        {"task_id": "does-not-exist", "detail": "x"}, session, steerer
    )
    assert result == {"acknowledged": True}


# ---------------------------------------------------------------------------
# goldfive#206: benign idempotent retries produce ZERO tool-loop drifts
# ---------------------------------------------------------------------------
#
# Regression guard for the session-``dd188a0c``-style pattern: a smaller
# model calls ``report_task_progress`` six times in a row with identical
# args after its task has already transitioned to RUNNING. Before #206
# retired the per-task ``ToolLoopGuard``, this pattern fired a CRITICAL
# ``LOOPING_TOOL_CALL`` drift at the 6th call and aborted the run; the
# newer stack (handler-owned idempotency + ToolLoopTracker's
# on_task_progress reset) absorbs it silently.


async def test_idempotent_progress_retries_produce_zero_drifts() -> None:
    """Six identical ``report_task_progress`` retries against a RUNNING
    task each return a plain ``{"acknowledged": True}`` ACK and do NOT
    emit any drift. Before goldfive#206 the per-task
    ``_tool_loop_guard`` would have fired CRITICAL ``LOOPING_TOOL_CALL``
    on the 6th call (exact=6 in window of 8) and aborted the run;
    after retirement the handler ACKs every call and no drift fires
    at this layer.
    """
    from goldfive.types import DriftEvent

    class _CapturingSteerer(DefaultSteerer):
        drifts_captured: list[DriftEvent]

        async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:  # type: ignore[override]
            self.drifts_captured.append(drift)
            await super()._handle_drift(drift, session)

    steerer = _CapturingSteerer()
    steerer.drifts_captured = []
    sink = _ListSink()
    planner = _StubPlanner()
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=Plan(
            id="p1",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="A",
                    assignee_agent_id="worker",
                    status=TaskStatus.RUNNING,
                ),
            ],
            edges=[],
        ),
    )
    steerer.bind(sinks=[sink], planner=planner)

    args = {"task_id": "t1", "fraction": 0.5, "detail": "stuck"}
    for _ in range(6):
        result = await _tool("report_task_progress").handler(args, session, steerer)
        # report_task_progress on RUNNING is a legal liveness tick —
        # plain ACK every time; the handler does not reject benign
        # retries.
        assert result == {"acknowledged": True}

    # No drifts — the retired per-task loop guard would have fired
    # CRITICAL LOOPING_TOOL_CALL at the 6th call (exact=6 in window
    # of 8) and aborted the run.
    assert steerer.drifts_captured == []
    # Task stays RUNNING; no forced transition.
    assert session.plan.tasks[0].status is TaskStatus.RUNNING


async def test_idempotent_progress_retries_do_not_fire_tool_loop_tracker() -> None:
    """The complementary regression guard for the tracker path: six
    identical progress reports where each acknowledged=True response
    resets the :class:`ToolLoopTracker` buffer via
    ``on_task_progress``. The tracker therefore emits zero drifts for
    benign progress-retry patterns, which matches the session
    ``dd188a0c`` shape that motivated goldfive#206.
    """
    from goldfive.drift.tool_loops import ToolLoopTracker

    tracker = ToolLoopTracker()
    all_drifts: list[Any] = []
    for _ in range(6):
        drifts = tracker.observe_tool_call(
            invocation_id="inv-1",
            agent_name="debugger_agent",
            tool_name="report_task_progress",
            args={"task_id": "t1", "fraction": 0.5, "detail": "stuck"},
            task_id="t1",
        )
        all_drifts.extend(drifts)
        # The ADK plugin calls on_task_progress after every
        # acknowledged=True response — simulate that here so the
        # buffer resets as it would on the real dispatch path.
        tracker.on_task_progress(invocation_id="inv-1", agent_name="debugger_agent")

    assert all_drifts == []
