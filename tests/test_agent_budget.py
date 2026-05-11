"""Structural per-ADK-sub-agent LLM-call budget (goldfive#256).

The framework planner has had a :attr:`LLMPlanner.MAX_OUTPUT_TOKENS`
ceiling since goldfive#271; goldfive's judges and goal_deriver have
their own caps. But ADK sub-agent calls (the user's coordinator /
research_agent / web_developer_agent / reviewer_agent / debugger_agent
/ anyone wrapped by :func:`goldfive.wrap`) flowed through
``before_model_callback`` with no cap — the model inherited its full
context window. Live e2e 2026-05-11 (ice cream session ``62dde1a6``)
captured a single research_agent dispatch that emitted 30K+ tokens at
~55 tok/s for a 5-bullet-point research task, with sustained 100%
speculative-decoding acceptance — low-entropy repetitive output that
nothing was bounding.

This suite pins the goldfive#256 fix:

1. :class:`AgentConfig` carries ``max_output_tokens`` (default 16384)
   and ``call_timeout_ms`` (default 120000ms / 2 minutes), both
   overridable via ``GOLDFIVE_AGENT_*`` env vars.
2. The ADK plugin ratchets ``llm_request.config.max_output_tokens``
   down to the configured ceiling in ``before_model_callback``.
3. Smaller-wins: a sub-agent / ADK that pinned a tighter cap KEEPS it
   — goldfive only ratchets DOWN, never up.
4. The wall-clock watcher (existing infrastructure from goldfive#271
   follow-up) fires an ``LLM_CALL_TIMEOUT`` drift on expiry, now with
   the tighter 120s default routed through :class:`AgentConfig`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)


from goldfive.config import AgentConfig, RuntimeConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Plain-config invariants
# ---------------------------------------------------------------------------


def test_agent_config_defaults() -> None:
    """Defaults must match the goldfive#256 contract documented on
    :class:`AgentConfig`: 16384 tokens (matches the planner cap) and
    120s wall-clock (Qwen 35B-class models cap at ~60-90s on long
    prompts; 120s gives headroom without inviting 30-minute hangs)."""
    cfg = AgentConfig()
    assert cfg.max_output_tokens == 16384
    assert cfg.call_timeout_ms == 120_000


def test_agent_config_from_env_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS`` overrides the field default."""
    monkeypatch.setenv("GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS", "1024")
    cfg = AgentConfig.from_env()
    assert cfg.max_output_tokens == 1024
    # Other field unaffected.
    assert cfg.call_timeout_ms == 120_000


def test_agent_config_from_env_call_timeout_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GOLDFIVE_AGENT_CALL_TIMEOUT_MS`` overrides the field default."""
    monkeypatch.setenv("GOLDFIVE_AGENT_CALL_TIMEOUT_MS", "30000")
    cfg = AgentConfig.from_env()
    assert cfg.call_timeout_ms == 30_000
    assert cfg.max_output_tokens == 16384


def test_agent_config_from_env_ignores_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-integer / non-positive env values fall back to the field default."""
    monkeypatch.setenv("GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS", "not-an-int")
    monkeypatch.setenv("GOLDFIVE_AGENT_CALL_TIMEOUT_MS", "0")
    cfg = AgentConfig.from_env()
    assert cfg.max_output_tokens == 16384
    assert cfg.call_timeout_ms == 120_000


def test_runtime_config_carries_agent_subconfig() -> None:
    """:class:`RuntimeConfig` exposes :class:`AgentConfig` so wrap() can
    thread it through to the ADK adapter."""
    runtime = RuntimeConfig()
    assert isinstance(runtime.agent, AgentConfig)
    assert runtime.agent.max_output_tokens == 16384


def test_runtime_config_from_env_threads_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RuntimeConfig.from_env()`` calls :meth:`AgentConfig.from_env`."""
    monkeypatch.setenv("GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("GOLDFIVE_AGENT_CALL_TIMEOUT_MS", "90000")
    runtime = RuntimeConfig.from_env()
    assert runtime.agent.max_output_tokens == 2048
    assert runtime.agent.call_timeout_ms == 90_000


# ---------------------------------------------------------------------------
# Plugin-side ``max_output_tokens`` ceiling
# ---------------------------------------------------------------------------


_adk_plugin = pytest.importorskip("goldfive.adapters._adk_plugin")
pytest.importorskip("google.adk")


def _make_llm_request_with_config(max_output_tokens: int | None = None) -> Any:
    """Build a minimal stand-in for ADK's ``LlmRequest`` with a mutable
    ``.config.max_output_tokens`` slot. Mirrors the duck-typed reads the
    plugin's helper performs without requiring a real ADK build."""
    config = SimpleNamespace(max_output_tokens=max_output_tokens)
    return SimpleNamespace(config=config)


def test_default_agent_max_output_tokens_module_constant() -> None:
    """The plugin's module-level default matches :class:`AgentConfig`."""
    assert _adk_plugin.DEFAULT_AGENT_MAX_OUTPUT_TOKENS == 16384


def test_make_adk_plugin_accepts_agent_max_output_tokens() -> None:
    """The factory accepts a custom ceiling and stashes it on the plugin."""
    plugin = _adk_plugin.make_adk_plugin(
        host_agent_name="root", agent_max_output_tokens=100
    )
    assert plugin._agent_max_output_tokens == 100


def test_make_adk_plugin_default_ceiling_when_unset() -> None:
    """Omitting the kwarg uses :data:`DEFAULT_AGENT_MAX_OUTPUT_TOKENS`."""
    plugin = _adk_plugin.make_adk_plugin(host_agent_name="root")
    assert plugin._agent_max_output_tokens == _adk_plugin.DEFAULT_AGENT_MAX_OUTPUT_TOKENS


def test_make_adk_plugin_zero_disables_ceiling() -> None:
    """``agent_max_output_tokens=0`` is the operator opt-out."""
    plugin = _adk_plugin.make_adk_plugin(host_agent_name="root", agent_max_output_tokens=0)
    assert plugin._agent_max_output_tokens == 0


def test_cap_applied_when_request_has_no_existing_value() -> None:
    """The helper writes the ceiling when ``llm_request.config`` carries
    no ``max_output_tokens`` (the common ADK default-config case)."""
    req = _make_llm_request_with_config(max_output_tokens=None)
    previous, applied = _adk_plugin._apply_agent_max_output_tokens_cap(req, ceiling=100)
    assert previous == 0
    assert applied == 100
    assert req.config.max_output_tokens == 100


def test_cap_ratchets_down_when_request_value_too_high() -> None:
    """When the existing value exceeds the ceiling, the plugin ratchets
    it down. This is the live e2e case: model defaults / context-window
    inheritance gives a value far above goldfive's structural cap."""
    req = _make_llm_request_with_config(max_output_tokens=999_999)
    previous, applied = _adk_plugin._apply_agent_max_output_tokens_cap(
        req, ceiling=16384
    )
    assert previous == 999_999
    assert applied == 16384
    assert req.config.max_output_tokens == 16384


def test_cap_smaller_wins_when_request_value_tighter() -> None:
    """A sub-agent that pinned a smaller cap (e.g. an LLM-as-a-judge
    sub-agent with a 512-token JSON budget) KEEPS its cap — goldfive
    only ratchets DOWN, never up. This is the structural-ceiling
    semantic."""
    req = _make_llm_request_with_config(max_output_tokens=512)
    previous, applied = _adk_plugin._apply_agent_max_output_tokens_cap(
        req, ceiling=16384
    )
    assert previous == 512
    assert applied == 512
    assert req.config.max_output_tokens == 512


def test_cap_disabled_when_ceiling_non_positive() -> None:
    """``ceiling <= 0`` leaves the request untouched (operator opt-out)."""
    req = _make_llm_request_with_config(max_output_tokens=999_999)
    previous, applied = _adk_plugin._apply_agent_max_output_tokens_cap(req, ceiling=0)
    assert previous == 0
    assert applied == 0
    # Untouched.
    assert req.config.max_output_tokens == 999_999


def test_cap_skips_when_request_has_no_config() -> None:
    """A duck-typed request without ``.config`` is tolerated (best-effort)."""
    req = SimpleNamespace()
    previous, applied = _adk_plugin._apply_agent_max_output_tokens_cap(req, ceiling=100)
    assert previous == 0
    assert applied == 0


# ---------------------------------------------------------------------------
# Wall-clock timeout integration (LLM_CALL_TIMEOUT drift)
# ---------------------------------------------------------------------------


class _CapturingSteerer:
    """Capture every observation + drift the watcher emits.

    Shape mirrors the live :class:`DefaultSteerer` interface enough for
    the watcher's emission path: ``observe(obs, session)`` and
    ``_emit_drift_detected(session, drift)``. The watcher writes to both;
    we expose both for assertions.
    """

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.emitted_drifts: list[Any] = []
        self._sinks: list[Any] = []

    async def observe(self, observation: Any, session: Any) -> None:
        self.observations.append(observation)

    async def _emit_drift_detected(self, session: Any, drift: Any) -> None:
        self.emitted_drifts.append(drift)


def _make_session_context(steerer: Any) -> Any:
    """Minimal :class:`SessionContext` for the watcher's reads."""
    from goldfive.types import Session

    session = Session(run_id="r1")
    return SimpleNamespace(
        steerer=steerer,
        session=session,
        task=SimpleNamespace(id="t1"),
    )


@pytest.mark.asyncio
async def test_timeout_fires_llm_call_timeout_drift() -> None:
    """When the wall-clock budget expires, the watcher must emit a
    CRITICAL ``LLM_CALL_TIMEOUT`` drift via the steerer AND flag the
    invocation for cooperative cancel. Together these are the building
    blocks the existing drift dispatch uses to drive the cancel +
    refine path (observation_only-dependent — see steerer tests for
    the injection-gate behaviour)."""
    plugin = _adk_plugin.make_adk_plugin(
        host_agent_name="research_agent", llm_call_timeout_ms=10
    )
    steerer = _CapturingSteerer()
    ctx = _make_session_context(steerer)

    await plugin._run_llm_call_timeout_watcher(
        invocation_id="inv-256-timeout",
        timeout_s=0.01,
        ctx=ctx,
    )

    # Cancel flag set so the next before_model / before_tool short-circuits.
    assert "inv-256-timeout" in plugin._cancel_state
    request = plugin._cancel_state["inv-256-timeout"]
    assert request.reason == "llm_call_timeout"
    assert request.drift_kind == "llm_call_timeout"

    # Drift surfaced for sinks.
    from goldfive.types import DriftKind, DriftSeverity

    assert len(steerer.emitted_drifts) == 1
    drift = steerer.emitted_drifts[0]
    assert drift.kind == DriftKind.LLM_CALL_TIMEOUT
    assert drift.severity == DriftSeverity.CRITICAL
    assert drift.current_agent_id == "research_agent"
    # ``timeout_s`` lands in the detail string so operators can correlate
    # the budget to the failure.
    assert "0.0" in drift.detail or "(0.0" in drift.detail


@pytest.mark.asyncio
async def test_timeout_watcher_cancelled_before_expiry_emits_nothing() -> None:
    """When the paired ``after_model_callback`` cancels the watcher (the
    LLM call returned within budget), the watcher absorbs CancelledError
    and emits nothing — neither a drift nor a cancel-state entry."""
    plugin = _adk_plugin.make_adk_plugin(
        host_agent_name="research_agent", llm_call_timeout_ms=10
    )
    steerer = _CapturingSteerer()
    ctx = _make_session_context(steerer)

    task = asyncio.create_task(
        plugin._run_llm_call_timeout_watcher(
            invocation_id="inv-256-noop",
            timeout_s=10.0,
            ctx=ctx,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    await task

    assert plugin._cancel_state == {}
    assert steerer.emitted_drifts == []
    assert steerer.observations == []


# ---------------------------------------------------------------------------
# End-to-end: AgentConfig threading from RuntimeConfig → ADKAdapter → plugin
# ---------------------------------------------------------------------------


def test_agent_config_threads_to_plugin_via_adkadapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The :class:`AgentConfig` fields plumb through
    :func:`goldfive.wrap` → :class:`ADKAdapter` → ``make_adk_plugin`` so
    the plugin's structural caps reflect the runtime config.

    Asserted directly on the plugin instance because constructing a
    real ADK ``InMemoryRunner`` requires a fully-built agent tree;
    instead we hand :class:`ADKAdapter` a minimal duck-typed agent.
    """
    # Avoid network/agent setup by feeding the adapter a duck-typed
    # ADK-shaped agent — _looks_like_adk_agent is satisfied by
    # ``sub_agents`` + ``name``.
    from goldfive.adapters.adk import ADKAdapter

    class _StubAgent:
        name = "research_agent"
        sub_agents: list[Any] = []
        model = None
        tools: list[Any] = []
        instruction = ""

    # Skip the heavy InMemoryRunner construction by passing a duck-typed
    # runner — _looks_like_runner needs run_async + agent + session_service.
    class _StubRunner:
        agent = _StubAgent()
        app_name = "test"

        def run_async(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise RuntimeError("stub")

        session_service = SimpleNamespace()

    adapter = ADKAdapter(
        _StubRunner(),
        agent_max_output_tokens=4096,
        llm_call_timeout_ms=60_000,
    )
    assert adapter._plugin._agent_max_output_tokens == 4096
    assert adapter._plugin._llm_call_timeout_ms == 60_000
