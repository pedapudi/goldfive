"""Tool-loop detector exemption tightening (goldfive#192).

Defense-in-depth for the tool-loop drift detector (#181/#186). The
previous behaviour exempted ``report_task_*`` calls from loop
detection on the **call** alone -- ``on_task_progress`` cleared the
per-(invocation, agent) ring buffer unconditionally whenever one of
those tools was dispatched. In the wild this meant an agent stuck
retrying ``report_task_started`` with a bad ``task_id`` kept
receiving ``{"acknowledged": false, "error": "missing_task_id"}``
and every one of those errored calls reset the detector. The
tool-loop never fired; the reasoning-loop detector caught it
eventually but only after the agent had chewed through 8 plan
revisions.

The fix moves the reset decision into
``ADKAdapter._GoldfiveADKPlugin.after_tool_callback`` and gates it on
the response shape. Only an acknowledged-success response
(``{"acknowledged": True, ...}`` with no ``error`` key) resets the
window. Every other shape -- including errored reports, unknown
response types, and ``None`` -- falls through to the detector's
normal ``observe_tool_call`` path so the errored calls accumulate in
the ring buffer and fire at the usual thresholds.

Contracts pinned here:

1. Successful ``report_task_progress`` calls reset the window on
   each dispatch -- a long burst of legitimate progress reports
   never fires the detector.
2. Errored ``report_task_started`` calls count as ordinary tool
   calls -- three identical errored retries trip mode 1 at the
   default exact-threshold of 3.
3. Mixed traffic (success + error) resets on the successes only;
   the errored calls can still trip mode 1 when they land back-to-back.
4. Non-reporting tools are unaffected by the gate -- existing #181
   behaviour preserved.
5. Unknown response shapes (string, ``None``, bare list) are
   conservatively treated as "not a successful progress report";
   the window does not reset and the call counts toward loop
   detection.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Session, Task

# ---------------------------------------------------------------------------
# Stubs -- mirror the shapes used in ``tests/test_drift_classifiers.py`` so
# the plugin wiring looks identical to operators cross-referencing tests.
# ---------------------------------------------------------------------------


class _RecordingToolLoopSteerer:
    """Steerer stub that captures every drift routed via ``_handle_drift``."""

    def __init__(self) -> None:
        self.drifts: list[DriftEvent] = []
        self.observations: list[Any] = []
        self._sinks: list[Any] = []

    async def observe(self, event: Any, session: Any) -> None:
        self.observations.append(event)

    async def _handle_drift(self, drift: DriftEvent, session: Any) -> None:
        self.drifts.append(drift)


def _plugin_with_tool_loop_ctx(task: Task, session: Session, steerer: Any):
    """Build an ADK plugin + state-context for tool-loop wiring tests.

    Importing deferred so the module stays import-clean when
    ``google.adk`` isn't installed (the integration tests guard on
    ``importorskip``). Mirrors the fixture in
    ``test_drift_classifiers.py`` so both test suites exercise the
    exact same plugin entry point.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )

    plugin = make_adk_plugin(host_agent_name="test_agent")
    state: dict = {
        SESSION_CONTEXT_STATE_KEY: SessionContext(
            session=session,
            steerer=steerer,
            task=task,
            tool_handlers={},
            host_agent_name="test_agent",
        )
    }
    plugin.set_active_context(state[SESSION_CONTEXT_STATE_KEY])
    return plugin, state


class _ToolStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _InvCtxStub:
    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent = type("_A", (), {"name": agent_name})()


class _ToolCtxStub:
    """Minimal ADK tool_context shape exposing ``_invocation_context``."""

    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self._invocation_context = _InvCtxStub(invocation_id, agent_name)


# ---------------------------------------------------------------------------
# Helper: the ``_is_progress_report_success`` predicate -- unit-pinned so
# regressions on the success-detection heuristic fail loudly without
# requiring the full plugin harness.
# ---------------------------------------------------------------------------


def test_is_progress_report_success_true_when_acknowledged_true() -> None:
    from goldfive.adapters._adk_plugin import _is_progress_report_success

    assert _is_progress_report_success({"acknowledged": True}) is True
    assert _is_progress_report_success({"acknowledged": True, "task_id": "t1"}) is True


def test_is_progress_report_success_false_when_acknowledged_false() -> None:
    from goldfive.adapters._adk_plugin import _is_progress_report_success

    assert _is_progress_report_success({"acknowledged": False}) is False


def test_is_progress_report_success_false_when_error_present() -> None:
    """Explicit ``error`` key wins even if ``acknowledged`` is missing / True."""
    from goldfive.adapters._adk_plugin import _is_progress_report_success

    assert _is_progress_report_success({"acknowledged": False, "error": "missing_task_id"}) is False
    # Half-broken shape: ``acknowledged=True`` AND ``error`` set. The error
    # wins so we don't silently reset on contradictory payloads.
    assert _is_progress_report_success({"acknowledged": True, "error": "something"}) is False


def test_is_progress_report_success_false_on_unknown_shapes() -> None:
    """None, strings, lists, bare values -> conservative False."""
    from goldfive.adapters._adk_plugin import _is_progress_report_success

    assert _is_progress_report_success(None) is False
    assert _is_progress_report_success("ok") is False
    assert _is_progress_report_success([{"acknowledged": True}]) is False
    assert _is_progress_report_success(42) is False
    # A dict that doesn't carry ``acknowledged`` at all -> False.
    assert _is_progress_report_success({"task_id": "t1"}) is False


# ---------------------------------------------------------------------------
# Plugin-level wiring
# ---------------------------------------------------------------------------


async def test_successful_report_task_progress_resets_window() -> None:
    """Five acknowledged-true progress reports in a row -> no drift.

    Pins the legitimate path: a long-running task that regularly
    reports progress must NEVER trip the detector even though it's
    calling the same ``(tool_name, args)`` repeatedly. The gate is
    ``acknowledged=True`` on every call, so each observation is
    followed by a window reset.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-1")
    task = Task(id="t-success", title="Long task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("report_task_progress")
    tool_ctx = _ToolCtxStub("inv-success", "test_agent")
    args = {"task_id": "t-success", "fraction": 0.25}
    ack = {"acknowledged": True, "task_id": "t-success"}

    for _ in range(5):
        await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_ctx, result=ack
        )

    assert steerer.drifts == []
    # Window should be empty after the final reset.
    assert (
        plugin._tool_loop_tracker.buffer_size(invocation_id="inv-success", agent_name="test_agent")
        == 0
    )


async def test_errored_report_task_started_counts_as_loop() -> None:
    """Identical errored ``report_task_started`` retries must trip mode 1.

    This is the #192 regression shape from session
    ``e2e_final_1776866766``: 17 consecutive
    ``{"acknowledged": false, "error": "missing_task_id"}`` responses
    were previously exempt from loop detection. With the tightening,
    the third identical errored retry fires mode 1 at the default
    exact-threshold of 3.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-2")
    task = Task(id="t-err", title="Task", assignee_agent_id="debugger_agent")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("report_task_started")
    tool_ctx = _ToolCtxStub("inv-err", "test_agent")
    args = {"task_id": "bogus-task-id"}
    nack = {"acknowledged": False, "error": "missing_task_id", "task_id": "bogus-task-id"}

    # Calls 1 and 2: buffer fills but mode 1 needs count >= 3.
    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=nack)
    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=nack)
    assert steerer.drifts == []

    # Call 3: exact-threshold hit -> WARNING drift fires.
    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=nack)
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "report_task_started"
    assert drift.current_task_id == "t-err"

    # The errored call must NOT have reset the window -- subsequent
    # identical retries keep the mode-1 signal hot.
    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=nack)
    assert len(steerer.drifts) == 2


async def test_mixed_success_and_error_counts_only_errors() -> None:
    """Mixed traffic: successes reset, errors accumulate.

    With default thresholds (window=7, exact_threshold=3) the sequence

        success, error, error, success, error, error, error

    should fire mode 1 at the final error. The two successes each
    clear the buffer, so between them the detector sees
    ``[error, error]`` (count 2, no fire). After the second success
    resets, the detector sees ``[error, error, error]`` -> mode 1
    fires on the third consecutive error.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-3")
    task = Task(id="t-mixed", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    started = _ToolStub("report_task_started")
    progress = _ToolStub("report_task_progress")
    tool_ctx = _ToolCtxStub("inv-mixed", "test_agent")
    err_args = {"task_id": "bogus"}
    ok_args = {"task_id": "t-mixed", "fraction": 0.1}
    nack = {"acknowledged": False, "error": "missing_task_id"}
    ack = {"acknowledged": True}

    async def _call(tool: _ToolStub, args: dict[str, Any], result: dict[str, Any]) -> None:
        await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_ctx, result=result
        )

    # 1. success -> buffer cleared, no drift.
    await _call(progress, ok_args, ack)
    assert steerer.drifts == []
    # 2. error -> buffer = [started@err]. count=1 < 3.
    await _call(started, err_args, nack)
    # 3. error -> buffer = [started@err, started@err]. count=2 < 3.
    await _call(started, err_args, nack)
    assert steerer.drifts == []
    # 4. success -> window reset, buffer empty.
    await _call(progress, ok_args, ack)
    assert steerer.drifts == []
    assert (
        plugin._tool_loop_tracker.buffer_size(invocation_id="inv-mixed", agent_name="test_agent")
        == 0
    )
    # 5, 6. two errors after the reset -- count climbing, still < 3.
    await _call(started, err_args, nack)
    await _call(started, err_args, nack)
    assert steerer.drifts == []
    # 7. third errored-in-a-row -> mode 1 fires.
    await _call(started, err_args, nack)
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "report_task_started"


async def test_non_report_tool_unaffected() -> None:
    """A non-reporting tool called 3 times trips mode 1 exactly as before.

    #192 changes the report_* exemption semantics only; ordinary
    tool-loop detection on arbitrary tools must be unchanged. This
    pins that the new success-gate logic does not disturb the #181
    baseline for non-reporting tools.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-4")
    task = Task(id="t-wf", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("write_file")
    tool_ctx = _ToolCtxStub("inv-wf", "test_agent")
    args = {"path": "out.txt", "content": "hi"}
    # Any ``result`` shape -- the non-report path never looks at it.
    result = {"bytes_written": 2}

    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result=result
    )
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result=result
    )
    assert steerer.drifts == []

    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result=result
    )
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.kind is DriftKind.LOOPING_REASONING
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "write_file"


async def test_unknown_response_shape_does_not_reset() -> None:
    """Progress tool + weird response -> conservative no-reset, call counts.

    The exemption rationale is "agent successfully reported forward
    progress". If we can't prove the call succeeded (response is
    ``None``, a bare string, a list, anything other than an
    acknowledged-success dict), we do NOT reset -- the call counts
    toward the loop detector like any other. This closes the
    half-broken-adapter hole where a buggy reporter that returned
    ``None`` could silently mask tool-loops.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-5")
    task = Task(id="t-unknown", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("report_task_started")
    tool_ctx = _ToolCtxStub("inv-unknown", "test_agent")
    args = {"task_id": "t-unknown"}

    # Three identical calls, each returning ``None`` (or a string).
    # None must not reset; mode 1 fires at call 3.
    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=None)
    await plugin.after_tool_callback(
        tool=tool, tool_args=args, tool_context=tool_ctx, result="weird-bare-string"
    )
    assert steerer.drifts == []

    await plugin.after_tool_callback(tool=tool, tool_args=args, tool_context=tool_ctx, result=None)
    assert len(steerer.drifts) == 1
    drift = steerer.drifts[0]
    assert drift.raw is not None
    assert drift.raw.get("mode") == "exact"
    assert drift.raw.get("tool_name") == "report_task_started"


async def test_report_awaiting_approval_respects_success_gate() -> None:
    """``report_awaiting_approval`` follows the same success-conditional gate.

    The approval-gate reporter is in the progress-reporting set
    (goldfive#192). An acknowledged success resets the window; an
    errored call does not. Pins the symmetry with the
    ``report_task_*`` tools.
    """
    pytest.importorskip("google.adk")
    steerer = _RecordingToolLoopSteerer()
    session = Session(run_id="run-192-6")
    task = Task(id="t-approval", title="Task", assignee_agent_id="worker")
    plugin, _state = _plugin_with_tool_loop_ctx(task, session, steerer)

    tool = _ToolStub("report_awaiting_approval")
    tool_ctx = _ToolCtxStub("inv-approval", "test_agent")
    args = {"task_id": "t-approval", "reason": "needs human"}

    # Three errored approval reports -> mode 1 fires on call 3.
    nack = {"acknowledged": False, "error": "missing_task_id"}
    for _ in range(3):
        await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_ctx, result=nack
        )
    assert len(steerer.drifts) == 1
    assert steerer.drifts[0].raw is not None
    assert steerer.drifts[0].raw.get("tool_name") == "report_awaiting_approval"
