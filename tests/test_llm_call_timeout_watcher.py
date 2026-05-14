"""Per-LLM-call wall-clock watcher (goldfive#271 follow-up).

Pre-fix: a single Qwen Q4 LLM dispatch could run for 9.6 minutes (9961
completion tokens, demo-v8.log) without any goldfive-level timeout. The
fix adds an asyncio watcher task spawned from
``_GoldfiveADKPlugin.before_model_callback`` that sleeps for the
configured budget; if the LLM call hasn't returned by then, the watcher
emits a CRITICAL ``LLM_CALL_TIMEOUT`` drift and flags the invocation
for cooperative cancel via :meth:`request_invocation_cancel`.

These tests pin the watcher's behaviour without exercising a real ADK
runner — we drive the watcher coroutine directly and assert on the
plugin's ``_cancel_state`` + the steerer's observation log.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# The watcher logic lives inside the closure-bound class created by
# ``make_adk_plugin``. We need google.adk to instantiate the plugin.
adk_plugin = pytest.importorskip("goldfive.adapters._adk_plugin")
pytest.importorskip("google.adk")


class _CapturingDrift:
    """Captures every observation seen + drift emitted (post #410)."""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.emitted_drifts: list[Any] = []

    async def observe(self, observation: Any, session: Any) -> None:
        self.observations.append(observation)

    async def _emit_drift_detected(self, session: Any, drift: Any) -> None:
        self.emitted_drifts.append(drift)


class _CapturingSteerer:
    """Minimal steerer stub — exposes a ``drift`` sub-component that
    captures every observation seen + every drift emitted via the
    conventional ``_emit_drift_detected`` path."""

    def __init__(self) -> None:
        self.drift = _CapturingDrift()
        self._sinks: list[Any] = []

    @property
    def observations(self) -> list[Any]:
        return self.drift.observations

    @property
    def emitted_drifts(self) -> list[Any]:
        return self.drift.emitted_drifts


def _make_session_context(steerer: Any) -> Any:
    """Build a minimal :class:`SessionContext` for the watcher.

    The watcher reads ``ctx.steerer``, ``ctx.session``, and ``ctx.task``
    via ``_safe_attr``. A simple SimpleNamespace satisfies all three.
    """
    from types import SimpleNamespace

    from goldfive.types import Session

    session = Session(run_id="r1")
    return SimpleNamespace(
        steerer=steerer,
        session=session,
        task=SimpleNamespace(id="t1"),
    )


def test_default_llm_call_timeout_is_30_minutes():
    """1800000 ms (30 min) is the pathological-hang ceiling for slow
    local models on compute-bound generation (Qwen 35B on slide
    generation, multi-step research synthesis). Tight latency SLOs are
    the operator's responsibility via ``llm_call_timeout_ms``."""
    assert adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS == 1_800_000


def test_make_adk_plugin_accepts_llm_call_timeout_ms():
    """The factory accepts a custom timeout and stashes it on the plugin."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root", llm_call_timeout_ms=5_000)
    assert plugin._llm_call_timeout_ms == 5_000


def test_make_adk_plugin_default_timeout_when_unset():
    """Omitting the kwarg uses the module default."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root")
    assert plugin._llm_call_timeout_ms == adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS


def test_make_adk_plugin_zero_disables_watcher():
    """``llm_call_timeout_ms=0`` is the operator opt-out."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="root", llm_call_timeout_ms=0)
    assert plugin._llm_call_timeout_ms == 0


@pytest.mark.asyncio
async def test_watcher_emits_drift_and_flags_cancel_on_expiry():
    """When the watcher sleeps to completion (the LLM call did NOT
    return in time), it must:

    * Emit a CRITICAL LLM_CALL_TIMEOUT drift via the steerer.
    * Flag the invocation in ``_cancel_state``.
    """
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x", llm_call_timeout_ms=10)
    steerer = _CapturingSteerer()
    ctx = _make_session_context(steerer)

    # Drive the watcher with a tiny budget. ``timeout_s=0`` would race
    # against asyncio scheduling — 0.01s is large enough to be
    # deterministic without slowing tests.
    await plugin._run_llm_call_timeout_watcher(
        invocation_id="inv-test",
        timeout_s=0.01,
        ctx=ctx,
    )

    # Watcher fired: cancel state set.
    assert "inv-test" in plugin._cancel_state
    request = plugin._cancel_state["inv-test"]
    assert request.reason == "llm_call_timeout"
    assert request.invocation_id == "inv-test"
    assert request.drift_kind == "llm_call_timeout"

    # And a drift was emitted to the steerer.
    from goldfive.types import DriftKind, DriftSeverity

    assert len(steerer.emitted_drifts) == 1
    drift = steerer.emitted_drifts[0]
    assert drift.kind == DriftKind.LLM_CALL_TIMEOUT
    assert drift.severity == DriftSeverity.CRITICAL
    assert drift.current_agent_id == "agent_x"


@pytest.mark.asyncio
async def test_watcher_returns_silently_on_cancel():
    """When the LLM call returns within budget the after_model_callback
    cancels the watcher's sleep — the watcher must absorb CancelledError
    and emit nothing."""
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x", llm_call_timeout_ms=10)
    steerer = _CapturingSteerer()
    ctx = _make_session_context(steerer)

    # Schedule the watcher with a generous timeout, then cancel it
    # before the sleep elapses.
    task = asyncio.create_task(
        plugin._run_llm_call_timeout_watcher(
            invocation_id="inv-test",
            timeout_s=10.0,  # never reached
            ctx=ctx,
        )
    )
    await asyncio.sleep(0)  # let task start
    task.cancel()
    # The watcher swallows CancelledError and returns; it must NOT
    # propagate.
    await task

    # No drift emitted, no cancel flagged.
    assert plugin._cancel_state == {}
    assert steerer.emitted_drifts == []
    assert steerer.observations == []


@pytest.mark.asyncio
async def test_watcher_uses_critical_severity():
    """The drift fired by the watcher is always CRITICAL — the steerer
    routes it to the Level-4 intervention ladder (HUMAN_INTERVENTION_REQUIRED).
    """
    plugin = adk_plugin.make_adk_plugin(host_agent_name="agent_x", llm_call_timeout_ms=10)
    steerer = _CapturingSteerer()
    ctx = _make_session_context(steerer)

    await plugin._run_llm_call_timeout_watcher(
        invocation_id="inv-test",
        timeout_s=0.01,
        ctx=ctx,
    )

    from goldfive.types import DriftSeverity

    drift = steerer.emitted_drifts[0]
    assert drift.severity == DriftSeverity.CRITICAL
