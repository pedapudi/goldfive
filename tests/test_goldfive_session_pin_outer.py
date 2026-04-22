"""Tests for outer-session pinning on :class:`GoldfiveADKAgent` (goldfive#161).

When adk-web drives a wrapped goldfive tree, the outer
``InvocationContext.session.id`` is the session id users see in the
harmonograf UI. goldfive's ``Session.run_id`` is minted independently by
:class:`Conversation.next_turn_session` (a uuid4), and the ADKAdapter's
internal ADK session id is minted by :meth:`ADKAdapter._ensure_session`
(another uuid4). Three layers of identity — three different session ids
stamped on different pieces of the stream — produce a harmonograf UI
where the plan view is empty and the span view has no plan.

The fix: ``GoldfiveADKAgent._run_async_impl`` reads ``ctx.session.id``
and pins it onto BOTH the goldfive ``Session.run_id`` (via
``Runner.run(session_id=...)``) AND the adapter's ``_session_id`` /
``_outer_session_id``. That way every goldfive Event and every
harmonograf span stamp the same id.

Back-compat: bare ``Runner.run`` callers (no adk-web ctx) keep seeing
minted uuid4 ``run_id`` s — nothing about the programmatic entry point
changes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.adk")

import goldfive
from goldfive import (
    InMemorySink,
    InvocationResult,
    Runner,
    SequentialExecutor,
    StaticPlanner,
)
from goldfive.types import Plan, Task


# ---------------------------------------------------------------------------
# Fixtures — minimal fakes for the InvocationContext shape the wrapper reads.
# ---------------------------------------------------------------------------


class _FakeSession:
    """Mirrors ``InvocationContext.session`` — exposes ``id`` + ``events``."""

    def __init__(self, sid: str = "") -> None:
        self.id = sid
        self.events: list[Any] = []


class _FakeCtx:
    """Stand-in for an ADK ``InvocationContext`` carrying a user turn.

    The wrapper only reads ``user_content``, ``invocation_id`` and
    ``session`` off the context.
    """

    def __init__(self, text: str, session_id: str = "") -> None:
        from google.genai.types import Content, Part  # type: ignore

        self.user_content = Content(role="user", parts=[Part(text=text)])
        self.session = _FakeSession(session_id)
        self.invocation_id = "invocation-test-1"
        self.end_invocation = False


def _mk_inner(name: str = "inner_agent") -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction="follow",
    )


def _mk_wrapped() -> Any:
    """Build a wrapped ADK agent whose adapter.invoke is mocked.

    The mock short-circuits the real InMemoryRunner event loop so each
    test case can observe the final Session state without spinning up
    a live LLM.
    """
    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=StaticPlanner(
            Plan(
                id="p1",
                run_id="",
                goal_ids=["g1"],
                tasks=[
                    Task(
                        id="t1",
                        title="the task",
                        description="x",
                        assignee_agent_id="inner_agent",
                    )
                ],
                edges=[],
                summary="one-task plan",
            )
        ),
        sinks=[InMemorySink()],
    )

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    wrapped.runner.agent.invoke = AsyncMock(side_effect=_fake_invoke)
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=3)
    return wrapped


# ---------------------------------------------------------------------------
# Part A tests — Session.id picks up ctx.session.id on first run.
# ---------------------------------------------------------------------------


async def test_goldfive_session_pins_outer_ctx_session_id() -> None:
    """``Session.id`` adopts ``ctx.session.id`` on the first invocation.

    Every Event emitted through the sink carries the same session_id as
    the outer adk-web session — not the uuid4 that
    ``Conversation.next_turn_session`` would have minted. This is the
    #161 fix: goldfive events now co-locate with harmonograf spans on
    the same UI session.
    """
    sink = InMemorySink()
    wrapped = _mk_wrapped()
    wrapped.sinks.append(sink)

    ctx = _FakeCtx("go", session_id="adk-outer-X")
    # Drive the ADK hook directly so we exercise the pin path without
    # requiring a real adk-web runner.
    events = [evt async for evt in wrapped._run_async_impl(ctx)]
    assert events, "wrapper produced no events"

    proto_events = [e for e in sink.events if hasattr(e, "session_id")]
    assert proto_events, "sink saw no proto envelopes"
    mismatches = [
        (e.WhichOneof("payload"), e.session_id, e.run_id)
        for e in proto_events
        if e.session_id != "adk-outer-X"
    ]
    assert not mismatches, f"events not pinned to outer session: {mismatches!r}"
    # run_id aliases session.id — confirm the Session itself carries it.
    assert all(e.run_id == "adk-outer-X" for e in proto_events)


async def test_goldfive_adapter_session_id_pins_to_outer() -> None:
    """The ADKAdapter's internal ``_session_id`` aligns with the outer id.

    ``ADKAdapter._ensure_session`` caches ``_session_id`` and reuses it
    across invoke calls — pinning the outer session id here means the
    internal ADK runner creates its InMemorySessionService session with
    the same id as adk-web, so ``_heal_pending_tool_calls`` and
    ``_touch_session`` also operate on the matched id.
    """
    wrapped = _mk_wrapped()
    adapter = wrapped.runner.agent
    # Precondition: no constructor-pinned id.
    assert adapter._session_id is None
    assert adapter._outer_session_id is None

    ctx = _FakeCtx("go", session_id="adk-outer-Y")
    _ = [evt async for evt in wrapped._run_async_impl(ctx)]

    assert adapter._session_id == "adk-outer-Y", (
        "adapter _session_id must adopt the outer ctx session id "
        "so _ensure_session / _heal_pending_tool_calls target the "
        "same ADK session that adk-web is driving"
    )
    assert adapter._outer_session_id == "adk-outer-Y"


async def test_goldfive_session_preserved_across_steer_restart() -> None:
    """A second invocation on the same ctx keeps the pinned session id.

    The overlay loop restarts ``invoke_passthrough`` with the same
    in-memory ``Session`` object, so ``Session.id`` cannot drift
    mid-turn. The cross-turn check verifies subsequent top-level
    ``_run_async_impl`` calls — the equivalent of a follow-up turn
    from the same adk-web tab — also re-pin to the same id (adk-web
    keeps the session stable across turns).
    """
    sink = InMemorySink()
    wrapped = _mk_wrapped()
    wrapped.sinks.append(sink)

    ctx = _FakeCtx("go", session_id="adk-outer-Z")

    # First invocation.
    _ = [evt async for evt in wrapped._run_async_impl(ctx)]
    # Second invocation with the SAME session id (adk-web's stable URL session).
    _ = [evt async for evt in wrapped._run_async_impl(ctx)]

    proto_events = [e for e in sink.events if hasattr(e, "session_id")]
    assert proto_events
    ids = {e.session_id for e in proto_events}
    assert ids == {"adk-outer-Z"}, (
        f"expected every event pinned to adk-outer-Z; saw ids={ids!r}"
    )


async def test_goldfive_session_falls_back_to_uuid_when_ctx_missing() -> None:
    """Bare programmatic ``Runner.run`` callers still mint a uuid4.

    Back-compat: users who never touch the ADK wrap path keep getting
    a freshly-minted ``Session.run_id`` per turn — the legacy contract
    from :meth:`Conversation.next_turn_session`.
    """
    import re

    from goldfive import CallableAdapter, PassthroughGoalDeriver

    async def _agent(task: Task, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["w"]),
        planner=StaticPlanner(
            Plan(
                id="p1",
                run_id="",
                goal_ids=["g1"],
                tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
                edges=[],
                summary="",
            )
        ),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
        sinks=[sink],
    )

    outcome = await runner.run("go")
    await runner.close()
    assert outcome.success

    # uuid4 hex is 32 lowercase hex chars — the shape
    # Conversation.next_turn_session produces.
    sid = outcome.session.id
    assert re.fullmatch(r"[0-9a-f]{32}", sid), (
        f"expected uuid4 hex run_id when ctx is absent; got {sid!r}"
    )


async def test_goldfive_session_falls_back_to_uuid_when_ctx_session_empty() -> None:
    """Empty ``ctx.session.id`` leaves the legacy uuid4 path intact.

    Live adk-web always populates a real session id, but test harnesses
    and future ADK shapes might pass a context whose ``session.id`` is
    ``""`` or ``None``. The pin path must skip cleanly in that case.
    """
    import re

    sink = InMemorySink()
    wrapped = _mk_wrapped()
    wrapped.sinks.append(sink)

    ctx = _FakeCtx("go", session_id="")
    _ = [evt async for evt in wrapped._run_async_impl(ctx)]

    proto_events = [e for e in sink.events if hasattr(e, "session_id")]
    assert proto_events
    # Every event still carries SOME session id (the minted uuid4), and
    # the run_id should be a uuid4 hex.
    ids = {e.session_id for e in proto_events}
    assert len(ids) == 1, f"expected one minted id; saw {ids!r}"
    minted = next(iter(ids))
    assert re.fullmatch(r"[0-9a-f]{32}", minted), (
        f"expected uuid4 hex for empty-ctx path; got {minted!r}"
    )


async def test_runner_run_accepts_session_id_override() -> None:
    """Direct programmatic callers can pass ``session_id`` to pin themselves.

    Primarily a contract test so third-party embedders that don't use
    the ADK wrap path can still align their Session.id with whatever
    external id they care about (e.g. a webhook correlation id).
    """
    from goldfive import CallableAdapter, PassthroughGoalDeriver

    async def _agent(task: Task, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["w"]),
        planner=StaticPlanner(
            Plan(
                id="p1",
                run_id="",
                goal_ids=["g1"],
                tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
                edges=[],
                summary="",
            )
        ),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
        sinks=[sink],
    )

    outcome = await runner.run("go", session_id="external-correlation-42")
    await runner.close()
    assert outcome.success
    assert outcome.session.id == "external-correlation-42"

    proto_events = [e for e in sink.events if hasattr(e, "session_id")]
    assert proto_events
    assert all(e.session_id == "external-correlation-42" for e in proto_events)


async def test_runner_run_empty_session_id_is_minted_uuid() -> None:
    """``session_id=None`` / ``""`` preserves the legacy uuid4 mint."""
    import re

    from goldfive import CallableAdapter, PassthroughGoalDeriver

    async def _agent(task: Task, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text="ok")

    runner = Runner(
        agent=CallableAdapter(_agent, available_agents=["w"]),
        planner=StaticPlanner(
            Plan(
                id="p1",
                run_id="",
                goal_ids=["g1"],
                tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
                edges=[],
                summary="",
            )
        ),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
    )

    outcome = await runner.run("go", session_id="")
    await runner.close()
    assert outcome.success
    assert re.fullmatch(r"[0-9a-f]{32}", outcome.session.id)
