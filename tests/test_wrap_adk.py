"""Tests for :func:`goldfive.wrap` polymorphism on ADK ``BaseAgent``.

Gated on the ``adk`` extra — skipped entirely when ``google.adk`` is
not installed. See ``docs/guides/adk-web-integration.md`` for the
motivation: adk web loads a ``BaseAgent`` as its ``root_agent``, so the
existing ``Runner`` return value from :func:`goldfive.wrap` could not
be used there. The polymorphic :class:`GoldfiveADKAgent` unifies the
two call sites.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

import goldfive
from goldfive import InMemorySink, Runner, StaticPlanner
from goldfive.types import Plan, Task


def _one_task_planner(task_id: str = "t1", title: str = "the task") -> StaticPlanner:
    """Return a :class:`StaticPlanner` that always emits one task routed to ``inner_agent``."""
    return StaticPlanner(
        Plan(
            id="p1",
            run_id="",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id=task_id,
                    title=title,
                    description=title,
                    assignee_agent_id="inner_agent",
                )
            ],
            edges=[],
            summary="one task plan",
        )
    )


def _mk_inner(name: str = "inner_agent") -> Any:
    """Construct a bare ADK ``LlmAgent`` to wrap."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction="follow instructions",
    )


def _mk_wrapped(**kwargs: Any) -> Any:
    """Build a GoldfiveADKAgent over a fresh inner LlmAgent."""
    inner = _mk_inner()
    return goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Identity / type contract
# ---------------------------------------------------------------------------


def test_wrap_returns_goldfive_adk_agent() -> None:
    """``goldfive.wrap(adk_agent)`` returns a :class:`GoldfiveADKAgent`."""
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent

    wrapped = _mk_wrapped()
    assert isinstance(wrapped, GoldfiveADKAgent)


def test_wrapped_is_adk_base_agent() -> None:
    """The returned object satisfies ``isinstance(result, BaseAgent)``."""
    from google.adk.agents import BaseAgent  # type: ignore

    wrapped = _mk_wrapped()
    assert isinstance(wrapped, BaseAgent)


def test_wrapped_exposes_base_agent_passthroughs() -> None:
    """Name / description / sub_agents mirror the inner agent."""
    inner = _mk_inner(name="parent_agent")
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )
    assert wrapped.name == "parent_agent"
    assert wrapped.description == inner.description
    assert wrapped.sub_agents == list(inner.sub_agents)


def test_wrapped_exposes_runner_surface() -> None:
    """The wrapped object exposes ``run``, ``close``, ``sinks``, ``control``."""
    wrapped = _mk_wrapped()
    assert callable(getattr(wrapped, "run", None))
    assert callable(getattr(wrapped, "close", None))
    assert isinstance(wrapped.sinks, list)
    # control is optional and defaults to None on the inner Runner.
    assert wrapped.control is None


# ---------------------------------------------------------------------------
# Programmatic run()
# ---------------------------------------------------------------------------


async def test_run_programmatically_returns_outcome(stub_call_llm: Any) -> None:
    """``await wrapped.run(...)`` still produces a valid ExecutionOutcome."""
    from unittest.mock import AsyncMock

    from goldfive import ExecutionOutcome, InvocationResult, SequentialExecutor

    inner = _mk_inner()
    # Patch the ADK adapter invoke path: we don't want to actually drive
    # an ADK runner in this unit test. A tiny substitute adapter keeps
    # the Runner happy without spinning up a real LlmAgent turn.
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=f"ok: {task.title}")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    outcome = await wrapped.run(
        [goldfive.Goal(id="g1", summary="say hi")],
    )
    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.success, outcome.reason


# ---------------------------------------------------------------------------
# run_async yields Events
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal stub for ``InvocationContext.session`` used in tests."""

    def __init__(self) -> None:
        self.events: list[Any] = []


class _FakeCtx:
    """Stand-in for an ``InvocationContext`` carrying a user turn.

    The wrapper only reads ``user_content``, ``invocation_id`` and
    ``session`` off the context — everything else is unused by goldfive's
    code path, so we can supply these three attributes and skip the
    ``InvocationContext`` pydantic validation entirely.
    """

    def __init__(self, text: str) -> None:
        from google.genai.types import Content, Part  # type: ignore

        self.user_content = Content(role="user", parts=[Part(text=text)])
        self.session = _FakeSession()
        self.invocation_id = "invocation-test-1"
        self.end_invocation = False


async def test_run_async_yields_events_for_user_turn(stub_call_llm: Any) -> None:
    """``_run_async_impl`` yields at least a plan + terminal Event pair."""
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=f"ok: {task.title}")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    ctx = _FakeCtx("make a thing")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]

    assert len(events) >= 2
    # First event should carry the plan summary.
    first_text = _event_text(events[0])
    assert first_text, "plan summary event has no text"
    # Last event closes the turn.
    last = events[-1]
    assert last.turn_complete is True


async def test_run_async_with_empty_input_yields_a_fallback_event() -> None:
    """A context with no user turn gets a single explanatory Event."""
    from google.genai.types import Content  # type: ignore

    wrapped = _mk_wrapped()

    ctx = _FakeCtx("")
    # Replace user_content with an empty Content so both fallbacks fail.
    ctx.user_content = Content(role="user", parts=[])

    events = [evt async for evt in wrapped._run_async_impl(ctx)]
    assert len(events) == 1
    text = _event_text(events[0])
    assert "no user input" in text.lower()


# ---------------------------------------------------------------------------
# Callable agents must NOT come back as a GoldfiveADKAgent
# ---------------------------------------------------------------------------


async def test_wrap_callable_still_returns_plain_runner() -> None:
    """Non-ADK agents must keep returning a plain :class:`Runner`."""
    from goldfive.adapters.adk_wrap import GoldfiveADKAgent

    async def _bare_agent(task: Any, session: Any, tools: Any) -> Any:
        from goldfive import InvocationResult

        return InvocationResult(task_id=task.id, text="ok")

    runner = goldfive.wrap(_bare_agent, sinks=[InMemorySink()])
    assert isinstance(runner, Runner)
    assert not isinstance(runner, GoldfiveADKAgent)


# ---------------------------------------------------------------------------
# harmonograf-style sink injection
# ---------------------------------------------------------------------------


async def test_sinks_mutation_propagates_to_inner_runner() -> None:
    """``wrapped.sinks.append(sink)`` reaches the inner Runner."""
    wrapped = _mk_wrapped()
    new_sink = InMemorySink()
    wrapped.sinks.append(new_sink)
    assert new_sink in wrapped.runner.sinks


# ---------------------------------------------------------------------------
# on_run_end teardown fan-out (goldfive#196)
# ---------------------------------------------------------------------------


async def test_run_async_impl_fires_on_run_end_on_normal_exit(stub_call_llm: Any) -> None:
    """On clean ``_run_async_impl`` completion, every adapter plugin
    that defines ``on_run_end`` must have it called exactly once.

    Observability plugins (e.g. harmonograf's telemetry plugin) use
    this hook to sweep orphan INVOCATION spans left open because ADK's
    ``after_run_callback`` is placed after an ``async with
    Aclosing(...)`` block — not inside a ``finally`` — so cancelled or
    early-closed sub-Runners leak their spans. See goldfive#196.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    class _RecordingPlugin:
        def __init__(self) -> None:
            self.on_run_end_calls = 0

        def on_run_end(self) -> None:
            self.on_run_end_calls += 1

    plugin = _RecordingPlugin()
    wrapped.runner.agent._plugins.append(plugin)

    ctx = _FakeCtx("do a thing")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]
    assert len(events) >= 1

    assert plugin.on_run_end_calls == 1


async def test_run_async_impl_fires_on_run_end_on_generator_close(
    stub_call_llm: Any,
) -> None:
    """If the caller ``aclose()``s the generator early (the adk-web
    disconnect path), ``on_run_end`` still fires via the ``finally``
    block. This is the specific orphan-span scenario goldfive#196
    closes: outer adk-web cancel leaves sub-Runner spans open, so the
    teardown hook MUST run even when the generator doesn't exhaust
    naturally.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    class _RecordingPlugin:
        def __init__(self) -> None:
            self.on_run_end_calls = 0

        def on_run_end(self) -> None:
            self.on_run_end_calls += 1

    plugin = _RecordingPlugin()
    wrapped.runner.agent._plugins.append(plugin)

    ctx = _FakeCtx("do a thing")
    agen = wrapped._run_async_impl(ctx)
    # Pull one event then bail — mimics adk-web disconnecting mid-stream.
    first = await agen.__anext__()
    assert first is not None
    await agen.aclose()

    assert plugin.on_run_end_calls == 1


async def test_run_async_impl_swallows_plugin_on_run_end_exceptions(
    stub_call_llm: Any,
) -> None:
    """A faulty ``on_run_end`` hook must not break the main run path.
    Any plugin exception is swallowed with a debug log so one bad
    plugin can't crash the outer generator's finalize sequence.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    class _BrokenPlugin:
        def on_run_end(self) -> None:
            raise RuntimeError("plugin boom")

    wrapped.runner.agent._plugins.append(_BrokenPlugin())

    ctx = _FakeCtx("do a thing")
    # Must not raise even though the plugin does.
    events = [evt async for evt in wrapped._run_async_impl(ctx)]
    assert events


async def test_run_async_impl_without_on_run_end_plugins_still_works(
    stub_call_llm: Any,
) -> None:
    """Plugins that do NOT define ``on_run_end`` fall through cleanly —
    this is the back-compat path for older ADK plugins that don't
    know about the teardown hook.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    class _NoHookPlugin:
        pass

    wrapped.runner.agent._plugins.append(_NoHookPlugin())

    ctx = _FakeCtx("do a thing")
    events = [evt async for evt in wrapped._run_async_impl(ctx)]
    assert events


# ---------------------------------------------------------------------------
# Phase 2.X / goldfive#271 Gap 3: per-turn judge drain is bounded tightly
# ---------------------------------------------------------------------------


async def test_per_turn_drain_uses_short_timeout_not_runner_close_default(
    stub_call_llm: Any,
) -> None:
    """``_drain_steerer_background_judges`` passes the
    :data:`GoldfiveADKAgent._PER_TURN_DRAIN_TIMEOUT_S` to
    ``Steerer.shutdown`` so a slow judge can't stall ADK's iterator
    exit for the full :meth:`Steerer.shutdown` default (5s).

    Validation E2E (session 95ecda4c) saw the residual #275-style
    stale-session race after the per-turn drain held ADK's outer
    loop for 5s while a concurrent ``/run_sse`` invocation advanced
    the SQLite session. Tightening the per-turn timeout to 0.5s
    bounds the race window.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    # Spy the steerer's shutdown to capture the timeout argument.
    captured_timeouts: list[float | None] = []

    async def _spy_shutdown(*args: Any, **kwargs: Any) -> None:
        captured_timeouts.append(kwargs.get("timeout"))

    wrapped.runner.steerer.shutdown = _spy_shutdown  # type: ignore[method-assign]

    ctx = _FakeCtx("do a thing")
    [evt async for evt in wrapped._run_async_impl(ctx)]

    assert captured_timeouts, (
        "_drain_steerer_background_judges did not call steerer.shutdown"
    )
    timeout = captured_timeouts[0]
    assert timeout is not None, (
        "per-turn drain must pass an explicit timeout (got None)"
    )
    # Tight enough that a 5s shutdown can't blow it open. We assert
    # well under 1s as the regression bound.
    assert timeout <= 1.0, (
        f"per-turn drain timeout regressed to {timeout!r}; expected <= 1s"
    )


async def test_per_turn_drain_falls_back_when_steerer_shutdown_lacks_timeout(
    stub_call_llm: Any,
) -> None:
    """Custom Steerers without the ``timeout=`` kwarg fall through to
    the legacy default-call shape. This is the back-compat path for
    third-party Steerers that haven't been updated for the per-turn
    timeout knob.
    """
    from unittest.mock import AsyncMock

    from goldfive import InvocationResult, SequentialExecutor

    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)

    # Old-shape shutdown: no timeout kwarg, raises TypeError when one
    # is passed. The drain helper must catch the TypeError and retry
    # with no kwargs.
    call_log: list[str] = []

    async def _legacy_shutdown() -> None:
        call_log.append("no-kwargs")

    async def _entry_shutdown(*args: Any, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError("legacy shutdown takes no kwargs")
        call_log.append("entry")
        await _legacy_shutdown()

    wrapped.runner.steerer.shutdown = _entry_shutdown  # type: ignore[method-assign]

    ctx = _FakeCtx("do a thing")
    [evt async for evt in wrapped._run_async_impl(ctx)]

    # The fallthrough call shape (no kwargs) was reached after the
    # initial timeout-kwarg call raised TypeError.
    assert "entry" in call_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or ()
    return "\n".join(str(getattr(p, "text", "") or "") for p in parts).strip()
