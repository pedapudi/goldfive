"""Reasoning-channel silent-disarm WARNING (goldfive#263 follow-up).

Pre-fix: on a non-thinking model (Gemma 4, Mistral, ...) the reasoning
extraction returns empty on every turn, ``observe_reasoning`` never
fires, and every LLM-judge reasoning detector silently disarms for the
whole run — the only trace was a DEBUG line behind the opt-in
``fallback_to_content_when_no_reasoning`` flag. Post-fix the plugin
tracks per-agent consecutive text responses without a reasoning stream
and, after ``_NO_REASONING_WARN_STREAK`` of them, emits a ONE-SHOT
per-agent WARNING plus a record-only sink event naming the remedy. The
fallback itself is never auto-enabled.

These tests drive ``after_model_callback`` directly with duck-typed
LLM responses — no real ADK runner, no LLM.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

adk_plugin = pytest.importorskip("goldfive.adapters._adk_plugin")
pytest.importorskip("google.adk")


class _CapturingDrift:
    """Captures observations + drifts emitted via ``_emit_drift_detected``."""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.emitted_drifts: list[Any] = []

    async def observe(self, observation: Any, session: Any) -> None:
        self.observations.append(observation)

    async def _emit_drift_detected(self, session: Any, drift: Any) -> None:
        self.emitted_drifts.append(drift)


class _CapturingSteerer:
    """Minimal steerer stub for ``after_model_callback``.

    ``_observation_only=True`` on purpose: the disarm warning is
    telemetry and must fire under the passive production default too.
    """

    def __init__(self) -> None:
        self.drift = _CapturingDrift()
        self._sinks: list[Any] = []
        self._observation_only = True


def _make_response(
    *,
    text: str = "",
    reasoning: str = "",
    function_call: bool = False,
) -> Any:
    """Duck-typed ``llm_response``: content.parts + optional reasoning."""
    parts: list[Any] = []
    if text:
        parts.append(SimpleNamespace(text=text, thought=False, function_call=None))
    if function_call:
        parts.append(
            SimpleNamespace(
                text="",
                thought=False,
                function_call=SimpleNamespace(name="some_tool", args={}),
            )
        )
    response = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=None,
    )
    if reasoning:
        # Plain string field — one of the shapes _extract_reasoning reads.
        response.reasoning_content = reasoning
    return response


def _make_plugin(host_agent_name: str = "coordinator") -> tuple[Any, _CapturingSteerer]:
    from goldfive.types import Session

    plugin = adk_plugin.make_adk_plugin(host_agent_name=host_agent_name)
    steerer = _CapturingSteerer()
    ctx = SimpleNamespace(
        steerer=steerer,
        session=Session(run_id="r1"),
        task=SimpleNamespace(id="t1"),
        host_agent_name=host_agent_name,
    )
    plugin.set_active_context(ctx)
    return plugin, steerer


def _make_callback_context(*, invocation_id: str = "inv-1", agent_name: str = "") -> Any:
    inv_ctx = SimpleNamespace(
        invocation_id=invocation_id,
        agent=SimpleNamespace(name=agent_name) if agent_name else None,
    )
    return SimpleNamespace(_invocation_context=inv_ctx, state={})


def _disarm_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "goldfive.reasoning.disarmed" in r.getMessage()
    ]


def _disarm_events(steerer: _CapturingSteerer) -> list[Any]:
    return [
        d
        for d in steerer.drift.emitted_drifts
        if "reasoning_channel_disarmed" in str(getattr(d, "detail", ""))
    ]


async def test_thinking_model_never_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Responses that carry a real reasoning stream keep the channel
    armed — no warning no matter how many turns."""
    plugin, steerer = _make_plugin()
    with caplog.at_level(logging.WARNING):
        for i in range(6):
            await plugin.after_model_callback(
                callback_context=_make_callback_context(),
                llm_response=_make_response(text=f"answer {i}", reasoning=f"thinking {i}"),
            )
    assert _disarm_warnings(caplog) == []
    assert _disarm_events(steerer) == []


async def test_non_thinking_model_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three consecutive text-only responses with no reasoning stream
    fire the WARNING + sink event exactly once, even across many more
    such turns. The warning names the remedy flag and never auto-enables
    it."""
    plugin, steerer = _make_plugin()
    with caplog.at_level(logging.WARNING):
        for i in range(7):
            await plugin.after_model_callback(
                callback_context=_make_callback_context(),
                llm_response=_make_response(text=f"answer {i}"),
            )
    warnings = _disarm_warnings(caplog)
    assert len(warnings) == 1
    assert "fallback_to_content_when_no_reasoning" in warnings[0].getMessage()

    events = _disarm_events(steerer)
    assert len(events) == 1
    from goldfive.types import DriftKind, DriftSeverity

    assert events[0].kind == DriftKind.CUSTOM
    assert events[0].severity == DriftSeverity.INFO
    assert "fallback_to_content_when_no_reasoning" in events[0].detail
    assert events[0].current_agent_id == "coordinator"


async def test_function_call_only_turns_do_not_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Function-call-only turns are weak evidence (thinking models often
    omit the stream on pure tool turns) — they neither increment nor
    reset the streak, so interleaving them among fewer than three text
    turns never warns."""
    plugin, steerer = _make_plugin()
    with caplog.at_level(logging.WARNING):
        # 2 text turns + 4 function-call-only turns: streak stays at 2.
        for response in (
            _make_response(text="a"),
            _make_response(function_call=True),
            _make_response(function_call=True),
            _make_response(text="b"),
            _make_response(function_call=True),
            _make_response(function_call=True),
        ):
            await plugin.after_model_callback(
                callback_context=_make_callback_context(),
                llm_response=response,
            )
    assert _disarm_warnings(caplog) == []
    assert _disarm_events(steerer) == []


async def test_reasoning_turn_resets_streak(caplog: pytest.LogCaptureFixture) -> None:
    """A turn that feeds the channel resets the count — 2 empty turns, a
    reasoning turn, then 2 more empty turns never reach the threshold."""
    plugin, steerer = _make_plugin()
    with caplog.at_level(logging.WARNING):
        for response in (
            _make_response(text="a"),
            _make_response(text="b"),
            _make_response(text="c", reasoning="thinking"),
            _make_response(text="d"),
            _make_response(text="e"),
        ):
            await plugin.after_model_callback(
                callback_context=_make_callback_context(),
                llm_response=response,
            )
    assert _disarm_warnings(caplog) == []
    assert _disarm_events(steerer) == []


async def test_warning_is_per_agent(caplog: pytest.LogCaptureFixture) -> None:
    """Each agent gets its own streak and its own one-shot warning."""
    plugin, steerer = _make_plugin()
    with caplog.at_level(logging.WARNING):
        for i in range(4):
            for agent in ("coordinator", "research_agent"):
                await plugin.after_model_callback(
                    callback_context=_make_callback_context(agent_name=agent),
                    llm_response=_make_response(text=f"answer {i}"),
                )
    warnings = _disarm_warnings(caplog)
    assert len(warnings) == 2
    events = _disarm_events(steerer)
    assert {e.current_agent_id for e in events} == {"coordinator", "research_agent"}
