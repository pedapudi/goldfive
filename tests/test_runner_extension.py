"""Tests for the :class:`Runner` extension API.

Covers the three post-construction extension points formalized in
issue #86:

* :meth:`Runner.add_sink` — append a sink after construction.
* :meth:`Runner.add_close_hook` — register async cleanup run by
  :meth:`Runner.close` after sinks close; close must be idempotent and
  a raising hook must not block subsequent hooks.
* :attr:`Runner.control` setter — idempotent on identity re-attach,
  raises :class:`RuntimeError` on conflicting attach.

Also covers the :class:`GoldfiveADKAgent` delegations (gated on the
``adk`` extra).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from goldfive import (
    CallableAdapter,
    ControlChannel,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _one_task_plan() -> Plan:
    return Plan(
        id="plan-ext",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="only task", assignee_agent_id="writer")],
        edges=[],
        summary="single task",
    )


async def _happy_agent(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text="ok")


def _make_runner(sinks: list[Any] | None = None) -> Runner:
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_one_task_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
        sinks=sinks if sinks is not None else [],
    )


# ---------------------------------------------------------------------------
# add_sink
# ---------------------------------------------------------------------------


async def test_add_sink_receives_events_on_subsequent_run() -> None:
    """A sink added post-construction sees the events of the next run."""
    early = InMemorySink()
    runner = _make_runner(sinks=[early])

    late = InMemorySink()
    runner.add_sink(late)

    outcome = await runner.run("go")
    await runner.close()

    assert outcome.success
    assert len(late.events) > 0
    # Both sinks got the same event count.
    assert len(late.events) == len(early.events)


# ---------------------------------------------------------------------------
# add_close_hook
# ---------------------------------------------------------------------------


async def test_close_hook_runs_exactly_once_on_close() -> None:
    """Registered hook runs on close; a second close is a no-op."""
    runner = _make_runner()
    calls: list[int] = []

    async def hook() -> None:
        calls.append(1)

    runner.add_close_hook(hook)

    await runner.close()
    await runner.close()  # idempotent — second call is a no-op.

    assert calls == [1]


async def test_close_hook_exception_is_logged_and_does_not_block_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising hook is logged; subsequent hooks still run."""
    runner = _make_runner()
    calls: list[str] = []

    async def hook_a() -> None:
        calls.append("a")

    async def hook_b() -> None:
        calls.append("b")
        raise RuntimeError("boom")

    async def hook_c() -> None:
        calls.append("c")

    runner.add_close_hook(hook_a)
    runner.add_close_hook(hook_b)
    runner.add_close_hook(hook_c)

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        await runner.close()

    assert calls == ["a", "b", "c"]
    assert any("close hook raised" in rec.message for rec in caplog.records)


async def test_close_hook_runs_after_sinks_close() -> None:
    """Sinks are closed before hooks fire (hooks depend on teardown state)."""
    order: list[str] = []

    class _OrderedSink:
        events: list[Any]

        def __init__(self, label: str) -> None:
            self.events = []
            self._label = label

        async def emit(self, event: Any) -> None:
            self.events.append(event)

        async def close(self) -> None:
            order.append(f"sink:{self._label}")

    runner = _make_runner(sinks=[_OrderedSink("x"), _OrderedSink("y")])

    async def hook() -> None:
        order.append("hook")

    runner.add_close_hook(hook)
    await runner.close()

    assert order == ["sink:x", "sink:y", "hook"]


# ---------------------------------------------------------------------------
# control setter
# ---------------------------------------------------------------------------


async def test_control_setter_attaches_when_unset() -> None:
    """Setter on a fresh runner attaches the channel."""
    runner = _make_runner()
    assert runner.control is None

    ch = ControlChannel()
    runner.control = ch
    assert runner.control is ch


async def test_control_setter_idempotent_on_same_identity() -> None:
    """Re-attaching the same channel (by ``is``) is a no-op, not an error."""
    runner = _make_runner()
    ch = ControlChannel()
    runner.control = ch
    runner.control = ch  # same identity — must not raise.
    assert runner.control is ch


async def test_control_setter_rejects_conflicting_channel() -> None:
    """Attaching a different channel when one exists raises RuntimeError."""
    runner = _make_runner()
    runner.control = ControlChannel()
    with pytest.raises(RuntimeError, match="already has a control channel"):
        runner.control = ControlChannel()


# ---------------------------------------------------------------------------
# GoldfiveADKAgent delegations — gated on the adk extra
# ---------------------------------------------------------------------------


def _adk_or_skip() -> None:
    pytest.importorskip("google.adk")


def _build_adk_wrapped() -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    import goldfive

    inner = LlmAgent(
        name="inner_agent",
        model="fake-model",
        description="wrapped",
        instruction="follow",
    )
    return goldfive.wrap(
        inner,
        planner=StaticPlanner(_one_task_plan()),
        sinks=[InMemorySink()],
    )


async def test_adk_wrapped_add_sink_delegates() -> None:
    _adk_or_skip()
    wrapped = _build_adk_wrapped()
    late = InMemorySink()
    wrapped.add_sink(late)
    assert late in wrapped.runner.sinks


async def test_adk_wrapped_close_hook_delegates() -> None:
    _adk_or_skip()
    wrapped = _build_adk_wrapped()
    calls: list[int] = []

    async def hook() -> None:
        calls.append(1)

    wrapped.add_close_hook(hook)
    await wrapped.close()
    await wrapped.close()
    assert calls == [1]


async def test_adk_wrapped_control_getter_and_setter_delegate() -> None:
    _adk_or_skip()
    wrapped = _build_adk_wrapped()
    assert wrapped.control is None

    ch = ControlChannel()
    wrapped.control = ch
    assert wrapped.control is ch
    assert wrapped.runner.control is ch

    # Idempotent on same identity.
    wrapped.control = ch

    # Conflicting channel raises.
    with pytest.raises(RuntimeError, match="already has a control channel"):
        wrapped.control = ControlChannel()
