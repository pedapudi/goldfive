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
from goldfive.adapters.adk_llm_instrumentation import (
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


# ---------------------------------------------------------------------------
# ADK session-state templating parity ({var} / {artifact.var})
# ---------------------------------------------------------------------------
#
# ADK's ``canonical_instruction`` marks callable instructions
# ``bypass_state_injection=True``, so the resolver must re-apply
# ``inject_session_state`` itself or wrapping silently disables the
# documented ``{var}`` templating. These tests drive ADK's real
# instructions request processor on wrapped vs unwrapped agents and
# require byte-identical system instructions.


async def _flow_system_instruction(
    agent: Any,
    *,
    state: dict[str, Any] | None = None,
) -> str | None:
    """Run ADK's instructions request processor and return the system instruction."""
    from google.adk.agents.invocation_context import InvocationContext  # type: ignore
    from google.adk.flows.llm_flows import instructions as adk_instructions  # type: ignore
    from google.adk.models.llm_request import LlmRequest  # type: ignore
    from google.adk.sessions.in_memory_session_service import (  # type: ignore
        InMemorySessionService,
    )

    svc = InMemorySessionService()
    session = await svc.create_session(
        app_name="app", user_id="u", state=dict(state or {})
    )
    inv_ctx = InvocationContext(
        session_service=svc,
        invocation_id="inv1",
        agent=agent,
        session=session,
    )
    llm_request = LlmRequest()
    async for _ in adk_instructions.request_processor.run_async(inv_ctx, llm_request):
        pass
    return llm_request.config.system_instruction


async def test_state_placeholder_resolves_identically_wrapped_vs_unwrapped() -> None:
    """``{var}`` substitution survives wrapping (no pin -> parity is exact)."""
    template = "Research {topic} and summarise {style?}."
    state = {"topic": "llamas", "style": "briefly"}

    unwrapped = _mk_llm_agent(name="plain", instruction=template)
    wrapped = _mk_llm_agent(name="wrapped", instruction=template)
    assert install_dynamic_instructions(wrapped) == 1

    plain_si = await _flow_system_instruction(unwrapped, state=state)
    wrapped_si = await _flow_system_instruction(wrapped, state=state)
    assert plain_si == "Research llamas and summarise briefly."
    assert wrapped_si == plain_si


async def test_placeholder_free_instruction_is_byte_identical() -> None:
    """The fast path leaves brace-less instructions untouched end-to-end."""
    template = "You are a careful research assistant."
    unwrapped = _mk_llm_agent(name="plain", instruction=template)
    wrapped = _mk_llm_agent(name="wrapped", instruction=template)
    assert install_dynamic_instructions(wrapped) == 1

    plain_si = await _flow_system_instruction(unwrapped, state={"topic": "x"})
    wrapped_si = await _flow_system_instruction(wrapped, state={"topic": "x"})
    assert wrapped_si == plain_si == template
    # The fast path also stays synchronous — no awaitable in the sync
    # resolver contract older callers rely on.
    resolver = wrapped.instruction
    assert isinstance(resolver(_ReadonlyCtxStub(state={})), str)


async def test_artifact_placeholder_resolves_identically_wrapped_vs_unwrapped() -> None:
    """``{artifact.var}`` loads through the invocation's artifact service."""
    from google.adk.artifacts.in_memory_artifact_service import (  # type: ignore
        InMemoryArtifactService,
    )
    from google.genai import types  # type: ignore

    template = "Use the notes: {artifact.notes}"

    async def _si_for(agent: Any) -> str | None:
        # The artifact must exist under the exact session id the flow
        # runs with, so this inlines the flow instead of reusing
        # _flow_system_instruction.
        svc = InMemoryArtifactService()
        from google.adk.agents.invocation_context import InvocationContext  # type: ignore
        from google.adk.flows.llm_flows import instructions as adk_instructions  # type: ignore
        from google.adk.models.llm_request import LlmRequest  # type: ignore
        from google.adk.sessions.in_memory_session_service import (  # type: ignore
            InMemorySessionService,
        )

        sess_svc = InMemorySessionService()
        session = await sess_svc.create_session(app_name="app", user_id="u")
        await svc.save_artifact(
            app_name="app",
            user_id="u",
            session_id=session.id,
            filename="notes",
            artifact=types.Part(text="llamas are camelids"),
        )
        inv_ctx = InvocationContext(
            session_service=sess_svc,
            artifact_service=svc,
            invocation_id="inv1",
            agent=agent,
            session=session,
        )
        llm_request = LlmRequest()
        async for _ in adk_instructions.request_processor.run_async(inv_ctx, llm_request):
            pass
        return llm_request.config.system_instruction

    unwrapped = _mk_llm_agent(name="plain", instruction=template)
    wrapped = _mk_llm_agent(name="wrapped", instruction=template)
    assert install_dynamic_instructions(wrapped) == 1

    plain_si = await _si_for(unwrapped)
    wrapped_si = await _si_for(wrapped)
    assert plain_si is not None and "llamas are camelids" in plain_si
    assert wrapped_si == plain_si


async def test_missing_state_var_raises_wrapped_and_unwrapped() -> None:
    """Substitution errors keep ADK's native failure mode after wrapping."""
    template = "Research {no_such_var}."
    unwrapped = _mk_llm_agent(name="plain", instruction=template)
    wrapped = _mk_llm_agent(name="wrapped", instruction=template)
    assert install_dynamic_instructions(wrapped) == 1

    with pytest.raises(KeyError):
        await _flow_system_instruction(unwrapped, state={})
    with pytest.raises(KeyError):
        await _flow_system_instruction(wrapped, state={})


async def test_templating_composes_with_task_pin_in_active_mode() -> None:
    """Active mode: {var} substitution feeds the composed instruction's base."""
    import inspect
    from types import SimpleNamespace

    resolver = make_dynamic_instruction(
        original_instruction="Research {topic}.",
        agent_name="inner_agent",
    )
    state: dict[str, Any] = {
        "topic": "llamas",
        _sp.KEY_CURRENT_TASK_ID: "t1",
        _sp.KEY_CURRENT_TASK_TITLE: "the task",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "do the thing",
    }
    ctx = _ReadonlyCtxStub(state=state)
    # inject_session_state reads state through the readonly context's
    # invocation context; give the stub the same backing dict.
    ctx._invocation_context = SimpleNamespace(
        session=SimpleNamespace(state=state), artifact_service=None
    )

    out = resolver(ctx)
    assert inspect.isawaitable(out)
    resolved = await out
    assert resolved.startswith("Research llamas.")
    assert "Current assigned task:" in resolved
    assert "{topic}" not in resolved


async def test_templating_still_runs_under_observation_only() -> None:
    """observation_only=True suppresses goldfive augmentation but NOT the
    templating ADK applies to string instructions regardless of goldfive."""
    import inspect
    from types import SimpleNamespace

    resolver = make_dynamic_instruction(
        original_instruction="Research {topic}.",
        agent_name="inner_agent",
    )
    stash = SimpleNamespace(
        steerer=SimpleNamespace(_observation_only=True), session=None
    )
    state: dict[str, Any] = {
        "topic": "llamas",
        "goldfive._session_context": stash,
        _sp.KEY_CURRENT_TASK_ID: "t1",
        _sp.KEY_CURRENT_TASK_TITLE: "the task",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "do the thing",
    }
    ctx = _ReadonlyCtxStub(state=state)
    ctx._invocation_context = SimpleNamespace(
        session=SimpleNamespace(state=state), artifact_service=None
    )

    out = resolver(ctx)
    assert inspect.isawaitable(out)
    resolved = await out
    # Templated exactly as unwrapped ADK would...
    assert resolved == "Research llamas."
    # ...with zero goldfive augmentation.
    assert "Current assigned task:" not in resolved


def test_install_skips_templated_instruction_when_inject_api_absent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No inject helper -> placeholder-bearing agents keep their string
    instruction (WARNING), placeholder-free agents still get the resolver."""
    from goldfive.adapters import adk_llm_instrumentation as mod

    monkeypatch.setattr(mod, "_adk_inject_session_state", lambda: None)

    templated = _mk_llm_agent(name="templated", instruction="Research {topic}.")
    plain = _mk_llm_agent(name="plain", instruction="Just research.")

    with caplog.at_level("WARNING", logger=mod.log.name):
        assert install_dynamic_instructions(templated) == 0
        assert install_dynamic_instructions(plain) == 1

    assert templated.instruction == "Research {topic}."
    assert is_dynamic_instruction(plain.instruction)
    assert any("inject_session_state is unavailable" in r.message for r in caplog.records)


async def test_user_callable_instruction_keeps_bypass_semantics() -> None:
    """A user InstructionProvider is never wrapped, so its output keeps
    ADK's native bypass: literal braces pass through unsubstituted."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    def user_provider(ctx: Any) -> str:
        return "user-dynamic {topic}"

    agent = LlmAgent(
        name="user_dyn",
        model="fake-model",
        description="user-managed dynamic",
        instruction=user_provider,  # type: ignore[arg-type]
    )
    assert install_dynamic_instructions(agent) == 0

    si = await _flow_system_instruction(agent, state={"topic": "llamas"})
    assert si == "user-dynamic {topic}"
