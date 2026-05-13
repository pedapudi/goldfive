"""Tests for the dynamic instruction resolver (goldfive#251 Stream B).

Covers:

* Resolver semantics — no pin, pinned, correction placeholder, legacy-
  state fallback, per-turn re-evaluation.
* ``goldfive.wrap()`` integration — default-on, opt-out via
  ``dynamic_instruction=False``, LlmAgent-only tree walk, per-agent
  independent resolution.

Gated on the ``adk`` extra; ADK is where this code is load-bearing.
The non-ADK shapes (Claude SDK, CallableAdapter) don't have an
``instruction`` surface to swap, so the installer is a silent no-op
for them.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

import goldfive
from goldfive import InMemorySink, StaticPlanner
from goldfive.adapters import _adk_state_protocol as _sp
from goldfive.adapters._adk_dynainst import (
    install_dynamic_instructions,
    is_dynamic_instruction,
    pending_correction_key,
)
from goldfive.prompt_shaper import PromptShaper
from goldfive.types import Plan, Task


def make_dynamic_instruction(
    original_instruction: str,
    agent_name: str,
):
    """Local shim: Wave B1 moved this factory onto
    :class:`PromptShaper`. Kept as a free function here so the existing
    test bodies (which pre-date the move) read unchanged."""
    return PromptShaper().make_dynamic_instruction(
        original_instruction=original_instruction,
        agent_name=agent_name,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_task_planner() -> StaticPlanner:
    return StaticPlanner(
        Plan(
            id="p1",
            run_id="",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="the task",
                    description="do the thing",
                    assignee_agent_id="inner_agent",
                )
            ],
            edges=[],
            summary="one task plan",
        )
    )


def _mk_llm_agent(
    name: str = "inner_agent",
    instruction: str = "follow instructions",
) -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction=instruction,
    )


class _ReadonlyCtxStub:
    """Minimal ``ReadonlyContext`` stand-in for resolver unit tests.

    Exposes only ``state`` since that is all the resolver reads. Using a
    stub avoids constructing a full ADK ``InvocationContext`` which
    requires a session service and other machinery we don't need here.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


# ---------------------------------------------------------------------------
# Resolver semantics
# ---------------------------------------------------------------------------


def test_resolver_returns_original_when_no_pin() -> None:
    """No ``goldfive.current_task_id`` in state -> original string verbatim."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    ctx = _ReadonlyCtxStub(state={})
    assert resolver(ctx) == "you are a helper"

    # Explicit empty string also counts as "no pin" -- the plugin clears
    # the pin that way when no unambiguous match exists.
    ctx2 = _ReadonlyCtxStub(state={_sp.KEY_CURRENT_TASK_ID: ""})
    assert resolver(ctx2) == "you are a helper"


def test_resolver_composes_when_pinned() -> None:
    """Pin + title + description -> composed instruction."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    ctx = _ReadonlyCtxStub(
        state={
            _sp.KEY_CURRENT_TASK_ID: "t1",
            _sp.KEY_CURRENT_TASK_TITLE: "the task",
            _sp.KEY_CURRENT_TASK_DESCRIPTION: "do the thing",
        }
    )
    out = resolver(ctx)
    assert out.startswith("you are a helper")
    assert "Current assigned task:" in out
    assert "id: t1" in out
    assert "title: the task" in out
    assert "description: do the thing" in out


def test_resolver_uses_placeholders_for_legacy_state() -> None:
    """Task id pinned but no title/description -> placeholder strings (not crash)."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    ctx = _ReadonlyCtxStub(state={_sp.KEY_CURRENT_TASK_ID: "t1"})
    out = resolver(ctx)
    assert "id: t1" in out
    assert "(title unset)" in out
    assert "(description unset)" in out


def test_resolver_appends_correction_when_present() -> None:
    """Pending correction for (agent, task) -> correction block appended."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    key = pending_correction_key("inner_agent", "t1")
    ctx = _ReadonlyCtxStub(
        state={
            _sp.KEY_CURRENT_TASK_ID: "t1",
            _sp.KEY_CURRENT_TASK_TITLE: "the task",
            _sp.KEY_CURRENT_TASK_DESCRIPTION: "do the thing",
            key: "CORRECTION: you missed the acceptance criterion on output format",
        }
    )
    out = resolver(ctx)
    assert "CORRECTION:" in out
    assert "acceptance criterion" in out


def test_resolver_no_correction_when_keyed_to_other_agent() -> None:
    """Corrections are scoped by (agent, task); another agent's correction is ignored."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    ctx = _ReadonlyCtxStub(
        state={
            _sp.KEY_CURRENT_TASK_ID: "t1",
            _sp.KEY_CURRENT_TASK_TITLE: "the task",
            _sp.KEY_CURRENT_TASK_DESCRIPTION: "do the thing",
            pending_correction_key("other_agent", "t1"): "not for you",
            pending_correction_key("inner_agent", "t_other"): "not this task",
        }
    )
    out = resolver(ctx)
    assert "not for you" not in out
    assert "not this task" not in out


def test_resolver_re_evaluates_per_turn() -> None:
    """Changing state between calls produces a different composed string."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )
    state: dict[str, Any] = {
        _sp.KEY_CURRENT_TASK_ID: "t1",
        _sp.KEY_CURRENT_TASK_TITLE: "first",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "first description",
    }
    ctx = _ReadonlyCtxStub(state=state)
    first = resolver(ctx)
    assert "title: first" in first

    # Simulate refine landing a revised description.
    state[_sp.KEY_CURRENT_TASK_DESCRIPTION] = "refined description"
    second = resolver(ctx)
    assert "refined description" in second
    assert first != second


def test_resolver_tolerates_exotic_readonly_context() -> None:
    """Resolver degrades to the original instruction on a context without ``state``."""
    resolver = make_dynamic_instruction(
        original_instruction="you are a helper",
        agent_name="inner_agent",
    )

    class _NoState:
        pass

    assert resolver(_NoState()) == "you are a helper"


# ---------------------------------------------------------------------------
# goldfive.wrap() integration
# ---------------------------------------------------------------------------


def test_wrap_installs_resolver_by_default() -> None:
    """Default-on: wrapping an ADK agent replaces ``instruction`` with a callable."""
    inner = _mk_llm_agent()
    assert isinstance(inner.instruction, str)

    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
    )
    _ = wrapped  # keep the Runner alive; inner is mutated in-place

    assert callable(inner.instruction)
    assert is_dynamic_instruction(inner.instruction)


def test_wrap_opt_out_keeps_static_instruction() -> None:
    """``dynamic_instruction=False`` leaves the original string in place."""
    inner = _mk_llm_agent(instruction="static prompt text")

    wrapped = goldfive.wrap(
        inner,
        planner=_one_task_planner(),
        sinks=[InMemorySink()],
        dynamic_instruction=False,
    )
    _ = wrapped

    assert inner.instruction == "static prompt text"
    assert not callable(inner.instruction)


def test_install_only_touches_llm_agents_in_tree() -> None:
    """A tree mixing LlmAgent + a container agent has only the LlmAgents swapped."""
    from google.adk.agents.sequential_agent import SequentialAgent  # type: ignore

    leaf_a = _mk_llm_agent(name="leaf_a", instruction="prompt A")
    leaf_b = _mk_llm_agent(name="leaf_b", instruction="prompt B")
    container = SequentialAgent(
        name="container",
        sub_agents=[leaf_a, leaf_b],
    )

    touched = install_dynamic_instructions(container)
    assert touched == 2
    assert is_dynamic_instruction(leaf_a.instruction)
    assert is_dynamic_instruction(leaf_b.instruction)
    # SequentialAgent has no ``instruction`` field — the walk skips it.
    assert not hasattr(container, "instruction") or not is_dynamic_instruction(
        getattr(container, "instruction", None)
    )


def test_install_is_idempotent() -> None:
    """Re-running the installer on an already-wrapped tree is a no-op."""
    leaf = _mk_llm_agent()
    touched_1 = install_dynamic_instructions(leaf)
    resolver_after_1 = leaf.instruction
    touched_2 = install_dynamic_instructions(leaf)
    resolver_after_2 = leaf.instruction

    assert touched_1 == 1
    assert touched_2 == 0
    # Same resolver object — not double-wrapped.
    assert resolver_after_1 is resolver_after_2


def test_install_leaves_user_supplied_callable_alone() -> None:
    """A user already using an ``InstructionProvider`` callable is not double-wrapped."""

    def user_provider(ctx: Any) -> str:
        return "user-dynamic prompt"

    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    agent = LlmAgent(
        name="user_dyn",
        model="fake-model",
        description="user-managed dynamic",
        instruction=user_provider,  # type: ignore[arg-type]
    )

    touched = install_dynamic_instructions(agent)
    assert touched == 0
    assert agent.instruction is user_provider
    assert not is_dynamic_instruction(agent.instruction)


def test_multiple_agents_resolve_independently() -> None:
    """Two LlmAgents in a tree each get their own resolver keyed on their name."""
    from google.adk.agents.sequential_agent import SequentialAgent  # type: ignore

    leaf_a = _mk_llm_agent(name="leaf_a", instruction="prompt for A")
    leaf_b = _mk_llm_agent(name="leaf_b", instruction="prompt for B")
    container = SequentialAgent(
        name="container",
        sub_agents=[leaf_a, leaf_b],
    )

    install_dynamic_instructions(container)

    state = {
        _sp.KEY_CURRENT_TASK_ID: "tA",
        _sp.KEY_CURRENT_TASK_TITLE: "title for A",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "description for A",
        pending_correction_key("leaf_a", "tA"): "CORRECTION for A",
    }
    ctx = _ReadonlyCtxStub(state=state)

    out_a = leaf_a.instruction(ctx)  # type: ignore[operator]
    out_b = leaf_b.instruction(ctx)  # type: ignore[operator]

    assert out_a.startswith("prompt for A")
    assert out_b.startswith("prompt for B")
    assert "CORRECTION for A" in out_a
    # leaf_b sees the same task pin but no correction targeted at it.
    assert "CORRECTION for A" not in out_b


def test_resolver_survives_canonical_instruction_contract() -> None:
    """The installed callable returns a str and is picked up by ADK's
    ``LlmAgent.canonical_instruction`` (i.e. matches the ``InstructionProvider``
    type alias). Smoke test against ADK's actual consumer."""
    import asyncio

    leaf = _mk_llm_agent(instruction="static")
    install_dynamic_instructions(leaf)

    state = {
        _sp.KEY_CURRENT_TASK_ID: "t1",
        _sp.KEY_CURRENT_TASK_TITLE: "title",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "desc",
    }
    ctx = _ReadonlyCtxStub(state=state)

    # Drive LlmAgent.canonical_instruction directly: it accepts anything
    # duck-typed as a ReadonlyContext (it only passes it to the provider
    # callable; doesn't inspect its attrs itself).
    instruction, bypass_state_injection = asyncio.run(leaf.canonical_instruction(ctx))
    assert isinstance(instruction, str)
    assert "id: t1" in instruction
    # When the resolved instruction is from a provider, ADK returns True
    # so state-template ``{placeholder}`` substitution is skipped.
    assert bypass_state_injection is True
