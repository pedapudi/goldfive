"""Tests for :mod:`goldfive.testkit.adversarial` and
:class:`goldfive.testkit.CannedCallLLM`.

Covers:

* Each adversarial primitive satisfies the AgentAdapter Protocol shape
  goldfive consumes (register_reporting_tools / invoke / emit_reasoning
  / available_agents).
* Each agent's ``expected_drift_kinds`` is consistent with its
  designed behaviour.
* Agents are deterministic when seeded.
* ``CannedCallLLM`` replays its transcript, records calls, and raises
  the documented exception on exhaustion.
"""

from __future__ import annotations

import asyncio

import pytest

from goldfive.protocols import AgentAdapter
from goldfive.reporting import ReportingToolSpec
from goldfive.runtime import clear_seed, is_seeded, seeded_random, set_seed
from goldfive.testkit import (
    CannedCallLLM,
    CannedCallLLMExhausted,
    CleanAgent,
    HallucinatingAgent,
    LoopingAgent,
    RefusingAgent,
    RunawayDelegationAgent,
    SlowAgent,
    WanderingAgent,
)
from goldfive.testkit.adversarial import as_callable
from goldfive.types import DriftKind, Plan, Session, Task, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(task_id: str = "t1", *, title: str = "do the thing") -> Task:
    return Task(id=task_id, title=title, description="", status=TaskStatus.PENDING)


def _session(run_id: str = "r1") -> Session:
    return Session(
        run_id=run_id,
        goals=[],
        plan=Plan(
            id="p1",
            run_id=run_id,
            goal_ids=[],
            tasks=[_task()],
            edges=[],
        ),
        current_task_id="t1",
    )


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CleanAgent(),
        lambda: LoopingAgent(cycle_after_turns=2),
        lambda: HallucinatingAgent(tool_name="search", fabricated_args={"q": "x"}),
        lambda: RefusingAgent(),
        lambda: WanderingAgent(off_topic_after_turns=1, off_topic_subject="penguins"),
        lambda: SlowAgent(delay_ms=1),
        lambda: RunawayDelegationAgent(target_count=3),
    ],
)
def test_adversarial_agents_satisfy_agent_adapter_protocol(factory) -> None:
    """Each adversarial agent quacks like an :class:`AgentAdapter`."""
    agent = factory()
    assert isinstance(agent, AgentAdapter)
    # Required attributes / methods.
    assert callable(agent.register_reporting_tools)
    assert callable(agent.invoke)
    assert callable(agent.emit_reasoning)
    # available_agents is a list (possibly empty).
    assert isinstance(agent.available_agents, list)


def test_adversarial_agents_declare_expected_drift_kinds() -> None:
    cases: list[tuple[type, tuple[DriftKind, ...]]] = [
        (CleanAgent, ()),
        (LoopingAgent, (DriftKind.LOOPING_TOOL_CALL, DriftKind.LOOPING_REASONING)),
        (HallucinatingAgent, (DriftKind.CONFABULATION_RISK, DriftKind.OFF_TOPIC)),
        (RefusingAgent, (DriftKind.AGENT_REFUSAL, DriftKind.MODEL_REFUSAL)),
        (WanderingAgent, (DriftKind.OFF_TOPIC, DriftKind.INTENT_DIVERGENCE)),
        (SlowAgent, (DriftKind.LLM_CALL_TIMEOUT, DriftKind.TASK_TIMEOUT)),
        (RunawayDelegationAgent, (DriftKind.RUNAWAY_DELEGATION,)),
    ]
    for cls, expected in cases:
        assert cls.expected_drift_kinds == expected, (
            f"{cls.__name__} expected_drift_kinds drifted from spec: "
            f"got {cls.expected_drift_kinds}, want {expected}"
        )


# ---------------------------------------------------------------------------
# CleanAgent — negative control
# ---------------------------------------------------------------------------


async def test_clean_agent_completes_without_emitting_reasoning() -> None:
    agent = CleanAgent(canned_response="all done")
    await agent.register_reporting_tools([])
    result = await agent.invoke(_task(), _session())
    assert result.task_id == "t1"
    assert result.text == "all done"
    assert result.stop_reason == "complete"
    assert agent.tool_calls == []


# ---------------------------------------------------------------------------
# LoopingAgent
# ---------------------------------------------------------------------------


async def test_looping_agent_cycles_after_threshold() -> None:
    agent = LoopingAgent(cycle_after_turns=2, tool_name="search")
    session = _session()
    task = _task()
    for _ in range(5):
        await agent.invoke(task, session)
    assert len(agent.tool_calls) == 5
    # Every call uses the same tool name and the same args once we cycle.
    names = [c.tool_name for c in agent.tool_calls]
    args = [c.args for c in agent.tool_calls]
    assert names == ["search"] * 5
    # Turns 3, 4, 5 (after the threshold of 2) carry identical args.
    assert args[2] == args[3] == args[4]


def test_looping_agent_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        LoopingAgent(cycle_after_turns=-1)


# ---------------------------------------------------------------------------
# HallucinatingAgent
# ---------------------------------------------------------------------------


async def test_hallucinating_agent_records_fabricated_tool_call_and_reports_completion() -> None:
    handler_calls: list[dict] = []

    async def _report_completed(**kwargs):
        handler_calls.append(kwargs)
        return {"acknowledged": True}

    spec = ReportingToolSpec(
        name="report_task_completed",
        description="",
        handler=_report_completed,
        parameters=(),
    )
    agent = HallucinatingAgent(
        tool_name="search_database",
        fabricated_args={"query": "fictional"},
        fabricated_summary="found 5 imaginary results",
    )
    await agent.register_reporting_tools([spec])
    result = await agent.invoke(_task(), _session())
    assert result.text == "found 5 imaginary results"
    assert len(agent.tool_calls) == 1
    assert agent.tool_calls[0].tool_name == "search_database"
    assert agent.tool_calls[0].args == {"query": "fictional"}
    # Verifies the agent did call the registered reporting handler.
    assert handler_calls and handler_calls[0]["task_id"] == "t1"


# ---------------------------------------------------------------------------
# RefusingAgent
# ---------------------------------------------------------------------------


async def test_refusing_agent_returns_refusal_text() -> None:
    agent = RefusingAgent(refusal_text="No.")
    result = await agent.invoke(_task(), _session())
    assert result.text == "No."
    assert result.stop_reason == "refusal"
    assert agent.tool_calls == []


# ---------------------------------------------------------------------------
# WanderingAgent
# ---------------------------------------------------------------------------


async def test_wandering_agent_stays_on_topic_then_drifts() -> None:
    agent = WanderingAgent(off_topic_after_turns=2, off_topic_subject="penguins")
    session = _session()
    task = _task(title="research solar panels")
    on_topic = await agent.invoke(task, session)
    on_topic2 = await agent.invoke(task, session)
    drifted = await agent.invoke(task, session)
    assert "penguins" not in on_topic.text
    assert "penguins" not in on_topic2.text
    assert "penguins" in drifted.text


def test_wandering_agent_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        WanderingAgent(off_topic_after_turns=-1, off_topic_subject="x")


# ---------------------------------------------------------------------------
# SlowAgent
# ---------------------------------------------------------------------------


async def test_slow_agent_sleeps_for_the_configured_delay() -> None:
    agent = SlowAgent(delay_ms=10)
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    result = await agent.invoke(_task(), _session())
    elapsed_ms = (loop.time() - t0) * 1000.0
    assert elapsed_ms >= 9.0  # generous lower bound for sleep timing
    assert result.text == "took my time."


def test_slow_agent_rejects_negative_delay() -> None:
    with pytest.raises(ValueError):
        SlowAgent(delay_ms=-5)


# ---------------------------------------------------------------------------
# RunawayDelegationAgent
# ---------------------------------------------------------------------------


async def test_runaway_delegation_agent_records_target_count_delegations() -> None:
    agent = RunawayDelegationAgent(target_count=4)
    result = await agent.invoke(_task(), _session())
    assert len(agent.tool_calls) == 4
    names = [c.args["agent_name"] for c in agent.tool_calls]
    assert names == ["sub_agent_0", "sub_agent_1", "sub_agent_2", "sub_agent_3"]
    assert "delegated to 4 sub-agents" in result.text


def test_runaway_delegation_agent_rejects_invalid_target() -> None:
    with pytest.raises(ValueError):
        RunawayDelegationAgent(target_count=0)


# ---------------------------------------------------------------------------
# Determinism under set_seed
# ---------------------------------------------------------------------------


def test_set_seed_produces_deterministic_random_stream() -> None:
    set_seed(42)
    try:
        assert is_seeded()
        r1 = seeded_random().random()
        r2 = seeded_random().random()
        set_seed(42)
        r3 = seeded_random().random()
        r4 = seeded_random().random()
        assert r1 == r3
        assert r2 == r4
    finally:
        clear_seed()
    assert not is_seeded()


# ---------------------------------------------------------------------------
# as_callable adapter helper
# ---------------------------------------------------------------------------


async def test_as_callable_routes_through_register_and_invoke() -> None:
    agent = CleanAgent(canned_response="ok")
    coro = as_callable(agent)
    spec = ReportingToolSpec(
        name="report_task_completed",
        description="",
        handler=lambda **kwargs: asyncio.sleep(0, result={"acknowledged": True}),
        parameters=(),
    )
    result = await coro(_task(), _session(), [spec])
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# CannedCallLLM
# ---------------------------------------------------------------------------


async def test_canned_call_llm_returns_transcript_in_order() -> None:
    llm = CannedCallLLM(["one", "two", "three"])
    assert llm.remaining == 3
    a = await llm("sys", "user1", "model")
    b = await llm("sys", "user2", "model")
    c = await llm("sys", "user3", "model")
    assert (a, b, c) == ("one", "two", "three")
    assert llm.call_count == 3
    assert llm.remaining == 0
    assert [call.user for call in llm.calls] == ["user1", "user2", "user3"]


async def test_canned_call_llm_raises_on_exhaustion() -> None:
    llm = CannedCallLLM(["only"])
    await llm("sys", "user", "model")
    with pytest.raises(CannedCallLLMExhausted) as exc:
        await llm("sys", "user", "model")
    assert exc.value.call_index == 1
    assert exc.value.transcript_length == 1


async def test_canned_call_llm_reset_rewinds_and_clears_calls() -> None:
    llm = CannedCallLLM(["a", "b"])
    await llm("s", "u", "m")
    llm.reset()
    assert llm.call_count == 0
    assert llm.calls == []
    again = await llm("s2", "u2", "m2")
    assert again == "a"
