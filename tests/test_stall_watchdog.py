"""Wall-clock stall watchdog (flag-gated producer for ``TASK_TIMEOUT``).

Pre-fix, a run wedged in a hung async tool call or idling with no task
transitions produced ZERO signal: ``DriftKind.TASK_TIMEOUT`` had no
producer anywhere and ``GOAL_DRIFT_IDLE_SECONDS`` was exported with no
consumer. The watchdog is a per-dispatch asyncio task spawned by
``_GoldfiveADKPlugin.set_active_context`` (when
``SteeringConfig.stall_watchdog_enabled`` — default OFF) and cancelled
by ``clear_active_context``. It polls a liveness watermark —
``max(Session.task_last_progress_at.values())`` and
``Session.last_observed_event_at`` (stamped on every observation
dispatch) — and fires ``TASK_TIMEOUT`` at WARNING, escalating to
CRITICAL on continued silence, through the normal ``handle_drift``
path.

These tests drive the watchdog coroutine directly with tiny timeouts
(same style as ``tests/test_llm_call_timeout_watcher.py``) plus a few
integration cases through a real ``DefaultSteerer`` to pin the
observe-only vs active-mode split.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

adk_plugin = pytest.importorskip("goldfive.adapters._adk_plugin")
pytest.importorskip("google.adk")

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.types import DriftKind, DriftSeverity, Session  # noqa: E402


class _CapturingDrift:
    """Captures ``handle_drift`` dispatches + idle goal-judge spawns."""

    def __init__(self) -> None:
        self.handled: list[Any] = []
        self.goal_judge_spawns: list[str] = []

    async def handle_drift(self, drift: Any, session: Any) -> None:
        self.handled.append(drift)

    def _spawn_goal_drift_judge_background(self, session: Any, *, idle_note: str = "") -> None:
        self.goal_judge_spawns.append(idle_note)


class _CapturingSteerer:
    """Minimal steerer stub carrying the watchdog knobs the plugin reads."""

    def __init__(self, *, enabled: bool = True, timeout_s: float = 0.05) -> None:
        self.drift = _CapturingDrift()
        self._sinks: list[Any] = []
        self._stall_watchdog_enabled = enabled
        self._stall_timeout_s = timeout_s


def _make_ctx(steerer: Any, session: Session | None = None) -> Any:
    return SimpleNamespace(
        steerer=steerer,
        session=session if session is not None else Session(run_id="r1"),
        task=SimpleNamespace(id="t1"),
    )


async def _wait_for(cond: Any, *, timeout: float = 2.0, message: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(message or "condition not met within timeout")


# ---------------------------------------------------------------------------
# Flag gating + lifecycle
# ---------------------------------------------------------------------------


def test_steering_config_defaults_watchdog_off() -> None:
    cfg = SteeringConfig()
    assert cfg.stall_watchdog_enabled is False
    assert cfg.stall_timeout_s == 600.0


def test_steering_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED", "1")
    monkeypatch.setenv("GOLDFIVE_STEER_STALL_TIMEOUT_S", "42.5")
    cfg = SteeringConfig.from_env()
    assert cfg.stall_watchdog_enabled is True
    assert cfg.stall_timeout_s == 42.5


def test_default_steerer_stashes_watchdog_knobs() -> None:
    from goldfive.steerer import DefaultSteerer

    bare = DefaultSteerer()
    assert bare._stall_watchdog_enabled is False
    assert bare._stall_timeout_s == 600.0
    configured = DefaultSteerer(
        steering_config=SteeringConfig(stall_watchdog_enabled=True, stall_timeout_s=1.5)
    )
    assert configured._stall_watchdog_enabled is True
    assert configured._stall_timeout_s == 1.5


async def test_disabled_by_default_no_task_spawned() -> None:
    """The un-flagged path must spawn nothing (default OFF)."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    steerer = _CapturingSteerer(enabled=False)
    plugin.set_active_context(_make_ctx(steerer))
    assert plugin._stall_watchdog_task is None
    plugin.clear_active_context()


async def test_flagless_steerer_stub_spawns_nothing() -> None:
    """A steerer without the knob attributes reads as OFF (fail-safe)."""

    class _Flagless:
        drift = _CapturingDrift()

    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    plugin.set_active_context(_make_ctx(_Flagless()))
    assert plugin._stall_watchdog_task is None
    plugin.clear_active_context()


async def test_nonpositive_timeout_disables() -> None:
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    steerer = _CapturingSteerer(enabled=True, timeout_s=0.0)
    plugin.set_active_context(_make_ctx(steerer))
    assert plugin._stall_watchdog_task is None
    plugin.clear_active_context()


async def test_enabled_spawns_and_teardown_cancels_cleanly() -> None:
    """``set_active_context`` spawns; ``clear_active_context`` cancels —
    the task ends cancelled with no lingering pending task."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    steerer = _CapturingSteerer(enabled=True, timeout_s=10.0)
    plugin.set_active_context(_make_ctx(steerer))
    task = plugin._stall_watchdog_task
    assert task is not None
    assert not task.done()
    await asyncio.sleep(0)  # let the watchdog reach its poll sleep
    plugin.clear_active_context()
    assert plugin._stall_watchdog_task is None
    # The watchdog absorbs the cancel and exits; awaiting must not raise.
    await task
    assert task.done()
    assert steerer.drift.handled == []


async def test_second_dispatch_replaces_prior_watchdog() -> None:
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    steerer = _CapturingSteerer(enabled=True, timeout_s=10.0)
    plugin.set_active_context(_make_ctx(steerer))
    first = plugin._stall_watchdog_task
    plugin.set_active_context(_make_ctx(steerer))
    second = plugin._stall_watchdog_task
    assert first is not None and second is not None
    assert first is not second
    await _wait_for(first.done, message="prior watchdog not cancelled")
    plugin.clear_active_context()
    await second


# ---------------------------------------------------------------------------
# Firing behaviour — graduated severity + watermark resets
# ---------------------------------------------------------------------------


async def test_fires_warning_then_critical_on_continued_silence() -> None:
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    steerer = _CapturingSteerer()
    session = Session(run_id="r1")
    session.current_task_id = "t-live"
    ctx = _make_ctx(steerer, session)

    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=0.05))
    try:
        await _wait_for(
            lambda: len(steerer.drift.handled) >= 2,
            message="watchdog did not escalate to a second drift",
        )
    finally:
        task.cancel()
        await task

    first, second = steerer.drift.handled[0], steerer.drift.handled[1]
    assert first.kind is DriftKind.TASK_TIMEOUT
    assert first.severity is DriftSeverity.WARNING
    assert second.kind is DriftKind.TASK_TIMEOUT
    assert second.severity is DriftSeverity.CRITICAL
    assert first.current_task_id == "t-live"
    assert first.current_agent_id == "agent_x"
    assert "stall watchdog" in first.detail


async def test_observation_stamp_resets_watermark_no_fire() -> None:
    """Active tool traffic (the observation stamp) keeps the watchdog
    quiet even with zero task transitions — then a fresh episode fires
    WARNING (not CRITICAL) once the traffic stops."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    steerer = _CapturingSteerer()
    session = Session(run_id="r1")
    ctx = _make_ctx(steerer, session)

    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=0.08))
    try:
        # Simulate steady observation dispatches for ~4x the timeout.
        deadline = time.monotonic() + 0.32
        while time.monotonic() < deadline:
            session.last_observed_event_at = time.monotonic()
            await asyncio.sleep(0.01)
        assert steerer.drift.handled == []
        # Silence — a fresh episode starts at WARNING.
        await _wait_for(
            lambda: len(steerer.drift.handled) >= 1,
            message="watchdog did not fire after traffic stopped",
        )
    finally:
        task.cancel()
        await task
    assert steerer.drift.handled[0].severity is DriftSeverity.WARNING


async def test_task_progress_stamp_counts_as_liveness() -> None:
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    steerer = _CapturingSteerer()
    session = Session(run_id="r1")
    ctx = _make_ctx(steerer, session)

    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=0.08))
    try:
        deadline = time.monotonic() + 0.32
        while time.monotonic() < deadline:
            session.task_last_progress_at["t1"] = time.monotonic()
            await asyncio.sleep(0.01)
        assert steerer.drift.handled == []
    finally:
        task.cancel()
        await task


async def test_no_fire_while_llm_call_inflight_under_budget() -> None:
    """A hung LLM call under its own per-call watcher is the
    ``LLM_CALL_TIMEOUT`` watcher's case — the stall watchdog must not
    double-report. Once the per-call watcher resolves, the stall
    watchdog resumes."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    steerer = _CapturingSteerer()
    ctx = _make_ctx(steerer)

    fake_watcher = asyncio.create_task(asyncio.sleep(30.0))
    plugin._invocation_llm_pending["inv-1"] = {"watcher": fake_watcher}

    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=0.05))
    try:
        await asyncio.sleep(0.25)
        assert steerer.drift.handled == []
        # Per-call watcher resolves (call returned / watcher cancelled);
        # the stall watchdog takes over on the very next tick.
        fake_watcher.cancel()
        await _wait_for(
            lambda: len(steerer.drift.handled) >= 1,
            message="watchdog did not resume after LLM watcher resolved",
        )
    finally:
        task.cancel()
        await task
        if not fake_watcher.done():
            fake_watcher.cancel()


# ---------------------------------------------------------------------------
# Idle-based goal-drift judge trigger (the GOAL_DRIFT_IDLE_SECONDS consumer)
# ---------------------------------------------------------------------------


async def test_idle_goal_judge_fires_once_per_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from goldfive.drift import goals as goals_mod

    monkeypatch.setattr(goals_mod, "GOAL_DRIFT_IDLE_SECONDS", 0.05)
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    # Large stall timeout so only the idle trigger engages.
    steerer = _CapturingSteerer(timeout_s=30.0)
    session = Session(run_id="r1")
    ctx = _make_ctx(steerer, session)

    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=30.0))
    try:
        await _wait_for(
            lambda: len(steerer.drift.goal_judge_spawns) >= 1,
            message="idle goal-judge trigger did not fire",
        )
        # Stay idle for several more polls — still exactly one spawn.
        await asyncio.sleep(0.2)
        assert len(steerer.drift.goal_judge_spawns) == 1
        assert "since last observed activity" in steerer.drift.goal_judge_spawns[0]
        # Fresh activity ends the episode; renewed silence re-arms it.
        session.last_observed_event_at = time.monotonic()
        await _wait_for(
            lambda: len(steerer.drift.goal_judge_spawns) >= 2,
            message="idle trigger did not re-arm after a new episode",
        )
    finally:
        task.cancel()
        await task
    assert steerer.drift.handled == []


async def test_idle_note_rendered_into_goal_judge_activity_block() -> None:
    """``maybe_run_goal_drift_check(idle_note=...)`` lands the note in
    the judge's activity block (the ``_format_activity`` slot)."""
    from goldfive.steerer import DefaultSteerer

    prompts: list[str] = []

    async def judge(system: str, user: str, model: str) -> str:
        prompts.append(user)
        return '{"progressing": true}'

    steerer = DefaultSteerer(goal_drift_call_llm=judge)
    session = Session(run_id="r1")
    await steerer.drift.maybe_run_goal_drift_check(
        session, idle_note="300s since last observed activity"
    )
    assert len(prompts) == 1
    assert "idle_observed" in prompts[0]
    assert "300s since last observed activity" in prompts[0]


# ---------------------------------------------------------------------------
# Observation stamping at the drift-pipeline entry points
# ---------------------------------------------------------------------------


async def test_drift_observer_entry_points_stamp_last_observed() -> None:
    from goldfive.steerer import DefaultSteerer

    steerer = DefaultSteerer()
    session = Session(run_id="r1")
    assert session.last_observed_event_at == 0.0

    steerer.drift.note_tool_observation(
        session,
        agent_name="a",
        task_id="t1",
        tool_name="search",
        args={"q": "x"},
        result={"ok": True},
    )
    stamp1 = session.last_observed_event_at
    assert stamp1 > 0.0

    steerer.drift.note_agent_activity(session, kind="agent_invocation_started")
    assert session.last_observed_event_at >= stamp1

    await steerer.drift.observe_reasoning(
        "thinking about the task", session=session, agent_name="a"
    )
    assert session.last_observed_event_at >= stamp1


# ---------------------------------------------------------------------------
# Both modes through the real dispatch (observation_only vs active)
# ---------------------------------------------------------------------------


def _sink_has_drift_detected(sink: Any) -> bool:
    return any(
        e.WhichOneof("payload") == "drift_detected" for e in sink.events if hasattr(e, "DESCRIPTOR")
    )


async def _drive_one_warning_through(steerer: Any, *, done: Any) -> tuple[Session, Any]:
    """Run the watchdog against a real steerer until ``done(session, sink)``."""
    from goldfive import InMemorySink, Plan, StaticPlanner, Task

    plan = Plan(
        id="p",
        run_id="r1",
        goal_ids=[],
        tasks=[Task(id="t1", title="do it", assignee_agent_id="worker")],
        edges=[],
        summary="",
    )
    sink = InMemorySink()
    steerer.bind(sinks=[sink], planner=StaticPlanner(plan))
    session = Session(run_id="r1")
    session.plan = plan
    session.current_task_id = "t1"

    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x")
    ctx = _make_ctx(steerer, session)
    task = asyncio.create_task(plugin._run_stall_watchdog(ctx=ctx, timeout_s=0.05))
    try:
        await _wait_for(
            lambda: done(session, sink),
            message="watchdog dispatch did not reach the expected state",
        )
    finally:
        task.cancel()
        await task
    return session, sink


async def test_observation_only_is_telemetry_only() -> None:
    """Under the production default the drift reaches the sinks but no
    intervention lands (no nudge enqueued)."""
    from goldfive.steerer import DefaultSteerer

    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=True,
            stall_watchdog_enabled=True,
            stall_timeout_s=0.05,
        )
    )
    session, _sink = await _drive_one_warning_through(
        steerer, done=lambda session, sink: _sink_has_drift_detected(sink)
    )
    # Grace period: let any (wrongly) queued intervention land before
    # asserting its absence.
    await asyncio.sleep(0.05)
    assert session.pending_nudges == []


async def test_active_mode_warning_routes_to_nudge() -> None:
    """In active mode the ladder's TASK_TIMEOUT WARNING row queues a
    corrective nudge."""
    from goldfive.steerer import DefaultSteerer

    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False,
            stall_watchdog_enabled=True,
            stall_timeout_s=0.05,
        )
    )
    session, sink = await _drive_one_warning_through(
        steerer, done=lambda session, sink: len(session.pending_nudges) >= 1
    )
    assert len(session.pending_nudges) >= 1
    assert _sink_has_drift_detected(sink)


def test_task_timeout_has_a_ladder_row() -> None:
    """The ladder table carries the conservative TASK_TIMEOUT row:
    INFO→OBSERVE, WARNING→SIGNAL (renamed from NUDGE in
    AGENCY-PRESERVATION.md PR 7), CRITICAL→PAUSE_ESCALATE (first and
    repeat — the watchdog's CRITICAL is by construction a repeat)."""
    from goldfive.drift_observer import DriftObserver
    from goldfive.steerer import DefaultSteerer, InterventionLevel

    steerer = DefaultSteerer()
    observer = steerer.drift
    assert isinstance(observer, DriftObserver)
    assert (
        observer._ladder_level_for(DriftKind.TASK_TIMEOUT, DriftSeverity.WARNING, 0)
        is InterventionLevel.SIGNAL
    )
    assert (
        observer._ladder_level_for(DriftKind.TASK_TIMEOUT, DriftSeverity.CRITICAL, 0)
        is InterventionLevel.PAUSE_ESCALATE
    )
    assert (
        observer._ladder_level_for(DriftKind.TASK_TIMEOUT, DriftSeverity.CRITICAL, 5)
        is InterventionLevel.PAUSE_ESCALATE
    )
