"""Tests for :func:`goldfive.wrap` / :func:`goldfive.run`.

Exercises the auto-adapter dispatch, LLM detection on ADK agents,
the degraded fallback when no LLM can be detected, and every
documented override knob.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

import goldfive
from goldfive import (
    CallableAdapter,
    ExecutionOutcome,
    InMemorySink,
    InvocationResult,
    LiteralGoalDeriver,
    LLMGoalDeriver,
    LLMPlanner,
    LoggingSink,
    PassthroughPlanner,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
)
from goldfive.adapters.auto import auto_adapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Reference agent: returns a non-empty result so the executor auto-completes."""
    _ = (session, tools)
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


class _MinimalAdapter:
    """Bare AgentAdapter that satisfies the protocol without extras."""

    def __init__(self) -> None:
        self._tools: list[ReportingToolSpec] = []

    @property
    def available_agents(self) -> list[str]:
        return ["minimal"]

    async def register_reporting_tools(
        self, tools: list[ReportingToolSpec]
    ) -> None:
        self._tools = list(tools)

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=f"minimal: {task.title}")

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Callable agents
# ---------------------------------------------------------------------------


async def test_wrap_callable_returns_runner() -> None:
    """``wrap(callable)`` returns a :class:`Runner` configured with defaults."""
    runner = goldfive.wrap(_happy_agent)

    assert isinstance(runner, Runner)
    assert isinstance(runner.agent, CallableAdapter)
    assert isinstance(runner.executor, SequentialExecutor)
    # No LLM detected on a bare callable → PassthroughPlanner + LiteralGoalDeriver.
    assert isinstance(runner.planner, PassthroughPlanner)
    assert isinstance(runner.goal_deriver, LiteralGoalDeriver)
    # Default sinks = [LoggingSink()].
    assert len(runner.sinks) == 1
    assert isinstance(runner.sinks[0], LoggingSink)


async def test_wrap_callable_runs_to_completion(stub_call_llm: Any) -> None:
    """``await wrap(agent).run(...)`` produces a successful outcome.

    A bare callable has no detectable LLM, so we supply ``call_llm=`` to
    wire up the default :class:`LLMPlanner` / :class:`LLMGoalDeriver`
    and exercise the full Runner loop end-to-end.
    """
    call_llm = stub_call_llm(
        [
            # LLMGoalDeriver response.
            {"goals": [{"id": "g1", "summary": "say hello"}]},
            # LLMPlanner response — one task routed to the default agent.
            {
                "summary": "one-task plan",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "say hello",
                        "description": "Say hello to the user",
                        "assignee_agent_id": "default",
                    }
                ],
                "edges": [],
            },
        ]
    )
    sink = InMemorySink()
    runner = goldfive.wrap(
        _happy_agent,
        sinks=[sink],
        call_llm=call_llm,
        model="test-model",
    )

    outcome = await runner.run("say hello")
    await runner.close()

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.success, outcome.reason
    assert outcome.session.plan is not None
    assert len(outcome.session.plan.tasks) == 1
    assert len(sink.events) > 0


async def test_run_convenience_returns_outcome() -> None:
    """``await goldfive.run(agent, input)`` returns an ExecutionOutcome."""
    outcome = await goldfive.run(
        _happy_agent,
        "do the thing",
        sinks=[InMemorySink()],
        planner=PassthroughPlanner(),  # explicit so the degraded path is OK
    )
    assert isinstance(outcome, ExecutionOutcome)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


async def test_wrap_custom_sinks_override_default(caplog: pytest.LogCaptureFixture) -> None:
    """Caller-supplied ``sinks`` replace the default ``LoggingSink``."""
    sink = InMemorySink()
    runner = goldfive.wrap(_happy_agent, sinks=[sink])

    assert runner.sinks == [sink]
    assert not any(isinstance(s, LoggingSink) for s in runner.sinks)


async def test_wrap_empty_sinks_list_is_respected() -> None:
    """An explicit empty list suppresses the default ``LoggingSink``."""
    runner = goldfive.wrap(_happy_agent, sinks=[])
    assert runner.sinks == []


async def test_wrap_custom_planner_override_wins(stub_call_llm: Any) -> None:
    """An explicit ``planner=`` argument wins over detection and defaults."""
    call_llm = stub_call_llm([])  # never called — we override the planner
    custom_planner = PassthroughPlanner()
    runner = goldfive.wrap(
        _happy_agent,
        planner=custom_planner,
        call_llm=call_llm,
        model="irrelevant",
    )
    assert runner.planner is custom_planner
    # goal_deriver should still use LLMGoalDeriver since call_llm was given.
    assert isinstance(runner.goal_deriver, LLMGoalDeriver)


async def test_wrap_call_llm_argument_drives_defaults(stub_call_llm: Any) -> None:
    """An explicit ``call_llm=`` wires both planner and goal deriver."""
    call_llm = stub_call_llm([])
    runner = goldfive.wrap(
        _happy_agent,
        call_llm=call_llm,
        model="test-model",
    )
    assert isinstance(runner.planner, LLMPlanner)
    assert isinstance(runner.goal_deriver, LLMGoalDeriver)


async def test_wrap_goal_deriver_override_wins() -> None:
    """An explicit ``goal_deriver=`` argument is used verbatim."""
    custom = LiteralGoalDeriver()
    runner = goldfive.wrap(_happy_agent, goal_deriver=custom)
    assert runner.goal_deriver is custom


async def test_wrap_forwards_kwargs_to_runner() -> None:
    """``max_plan_reinvocations`` flows into the constructed Runner."""
    runner = goldfive.wrap(_happy_agent, max_plan_reinvocations=7)
    assert runner.max_plan_reinvocations == 7
    assert isinstance(runner.executor, SequentialExecutor)
    assert runner.executor.max_plan_reinvocations == 7


# ---------------------------------------------------------------------------
# Degraded fallback
# ---------------------------------------------------------------------------


async def test_wrap_without_llm_falls_back_to_passthrough(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No LLM ⇒ PassthroughPlanner + LiteralGoalDeriver + DEBUG log line."""
    caplog.set_level(logging.DEBUG, logger="goldfive.wrap")
    runner = goldfive.wrap(_happy_agent)

    assert isinstance(runner.planner, PassthroughPlanner)
    assert isinstance(runner.goal_deriver, LiteralGoalDeriver)

    # DEBUG-level explanation of the degradation is emitted.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "PassthroughPlanner" in messages
    assert "LiteralGoalDeriver" in messages


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


async def test_wrap_passes_existing_adapter_through() -> None:
    """A pre-built :class:`AgentAdapter` is reused, not re-wrapped."""
    adapter = _MinimalAdapter()
    runner = goldfive.wrap(adapter)
    assert runner.agent is adapter


async def test_auto_adapter_rejects_unknown_shapes() -> None:
    """A non-adapter, non-callable argument raises :class:`TypeError`."""
    with pytest.raises(TypeError) as excinfo:
        auto_adapter(42)
    msg = str(excinfo.value)
    assert "goldfive.AgentAdapter" in msg
    assert "google.adk" in msg
    assert "claude_agent_sdk" in msg


async def test_auto_adapter_rejects_sync_callable() -> None:
    """A sync callable that doesn't look like a factory is rejected."""

    def sync_thing(task, session, tools):  # noqa: ANN001
        return None

    with pytest.raises(TypeError):
        auto_adapter(sync_thing)


async def test_auto_adapter_picks_callable_for_async_function() -> None:
    """Async functions with the agent signature get a :class:`CallableAdapter`."""
    adapter = auto_adapter(_happy_agent)
    assert isinstance(adapter, CallableAdapter)


# ---------------------------------------------------------------------------
# ADK-gated tests
# ---------------------------------------------------------------------------


def _has_adk() -> bool:
    try:
        import google.adk  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_adk(), reason="goldfive[adk] extra not installed")
async def test_wrap_adk_agent_chooses_adk_adapter() -> None:
    """An ADK ``BaseAgent`` dispatches to :class:`ADKAdapter`.

    Since Phase 2 (issue #77) :func:`goldfive.wrap` returns a
    polymorphic :class:`GoldfiveADKAgent` for ADK inputs. The underlying
    :class:`Runner` — reachable via ``.runner`` — still carries an
    :class:`ADKAdapter`, which is what this test verifies.
    """
    from google.adk.agents.llm_agent import LlmAgent

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent

    adk_agent = LlmAgent(
        name="test_agent",
        model="fake-model",
        description="Test agent",
        instruction="Test.",
    )
    wrapped = goldfive.wrap(adk_agent, sinks=[InMemorySink()])
    assert isinstance(wrapped, GoldfiveADKAgent)
    assert isinstance(wrapped.runner.agent, ADKAdapter)


@pytest.mark.skipif(not _has_adk(), reason="goldfive[adk] extra not installed")
async def test_adk_agent_is_not_mistaken_for_callable() -> None:
    """ADK takes precedence over callable detection when both match."""
    from google.adk.agents.llm_agent import LlmAgent

    from goldfive.adapters.adk import ADKAdapter

    adk_agent = LlmAgent(
        name="test_agent",
        model="fake-model",
        description="Test agent",
        instruction="Test.",
    )
    adapter = auto_adapter(adk_agent)
    assert isinstance(adapter, ADKAdapter)
    assert not isinstance(adapter, CallableAdapter)


# ---------------------------------------------------------------------------
# Claude-gated tests
# ---------------------------------------------------------------------------


def _has_claude_sdk() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _has_claude_sdk(), reason="claude_agent_sdk not installed"
)
async def test_wrap_claude_sdk_factory_chooses_claude_adapter() -> None:
    """A zero-arg factory returning ``ClaudeSDKClient`` dispatches to the Claude adapter."""
    from claude_agent_sdk import ClaudeSDKClient

    from goldfive.adapters.claude import ClaudeAgentSDKAdapter

    def factory() -> ClaudeSDKClient:
        return ClaudeSDKClient()

    runner = goldfive.wrap(factory, sinks=[InMemorySink()])
    assert isinstance(runner.agent, ClaudeAgentSDKAdapter)
