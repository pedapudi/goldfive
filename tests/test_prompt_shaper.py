"""Unit tests for :class:`goldfive.prompt_shaper.PromptShaper`.

Wave B1 of the modularisation plan extracted the four prompt-shape
injection sites + the centralised :meth:`PromptShaper.should_inject`
gate out of :mod:`goldfive.runner`, :mod:`goldfive.adapters._adk_plugin`,
and :mod:`goldfive.adapters._adk_dynainst`. These tests cover:

1. :meth:`should_inject` returns ``False`` when ``observation_only=True``
   (gate closed → suppress every site).
2. Each inject method is a no-op under
   ``observation_only=True`` — the input value is returned (or, for
   side-effecting methods, ``llm_request`` is left unchanged).
3. Each inject method matches the byte-output of the pre-refactor
   helper under ``observation_only=False``.

The behavioural strict-passive coverage lives in
``test_observation_only_strict_passive.py``; this module exercises
:class:`PromptShaper` directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from goldfive.config import SteeringConfig
from goldfive.prompt_shaper import PromptShaper
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal, Plan, Session, Task, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_steerer(*, observation_only: bool) -> DefaultSteerer:
    return DefaultSteerer(
        steering_config=SteeringConfig(observation_only=observation_only)
    )


def _make_session_with_completed_plan() -> Session:
    """A session whose plan has one COMPLETED task, used by the
    conversational-wrap site to render the "Completed tasks:" block."""
    sess = Session(run_id="r-prompt-shaper-test")
    sess.goals = [Goal(id="g1", summary="solar")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-prompt-shaper-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft slides",
                description="Write slides",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            )
        ],
        edges=[],
        summary="presentation about solar panels",
        revision_index=1,
    )
    return sess


# ---------------------------------------------------------------------------
# should_inject
# ---------------------------------------------------------------------------


def test_should_inject_true_when_steerer_active() -> None:
    """``observation_only=False`` → gate open → inject."""
    assert PromptShaper.should_inject(_make_steerer(observation_only=False)) is True


def test_should_inject_false_when_steerer_passive() -> None:
    """``observation_only=True`` → gate closed → suppress."""
    assert PromptShaper.should_inject(_make_steerer(observation_only=True)) is False


def test_should_inject_true_when_steerer_is_none() -> None:
    """Defensive: a missing steerer (unit-test stubs without a wired
    DefaultSteerer) must not flip the gate closed — pre-#271 paths
    keep working byte-identically."""
    assert PromptShaper.should_inject(None) is True


def test_should_inject_true_when_steerer_lacks_attribute() -> None:
    """A duck-typed steerer that doesn't carry ``_observation_only``
    (custom impls predating #254) returns True so pre-#254 paths keep
    working."""

    class _LegacySteerer:
        pass

    assert PromptShaper.should_inject(_LegacySteerer()) is True


# ---------------------------------------------------------------------------
# Site 1 — wrap_conversational_input
# ---------------------------------------------------------------------------


def test_wrap_conversational_input_noop_when_passive() -> None:
    """Under ``observation_only=True`` the wrapper returns ``user_input``
    verbatim — no ``[CONVERSATIONAL FOLLOW-UP]`` framing."""
    shaper = PromptShaper()
    sess = _make_session_with_completed_plan()
    steerer = _make_steerer(observation_only=True)

    out = shaper.wrap_conversational_input(
        user_input="where will the slides be saved?",
        session=sess,
        steerer=steerer,
    )
    assert out == "where will the slides be saved?"
    assert "CONVERSATIONAL FOLLOW-UP" not in out


def test_wrap_conversational_input_byte_identical_when_active() -> None:
    """Under ``observation_only=False`` the wrapper reproduces the
    pre-refactor format: a directive header, plan summary, completed
    tasks list, and the verbatim user question."""
    shaper = PromptShaper()
    sess = _make_session_with_completed_plan()
    steerer = _make_steerer(observation_only=False)

    out = shaper.wrap_conversational_input(
        user_input="where will the slides be saved?",
        session=sess,
        steerer=steerer,
    )
    # Header fingerprint — must match the canonical wording so an
    # operator grepping for the directive sees it stable across
    # releases.
    assert out.startswith(
        "[CONVERSATIONAL FOLLOW-UP — reuse prior plan, don't delegate "
        "to sub-agents]\n\n"
    )
    assert "Do NOT call any AgentTool" in out
    assert "Plan summary: presentation about solar panels" in out
    assert "Completed tasks:" in out
    assert "[t1] Draft slides (by writer)" in out
    assert out.endswith("User question: where will the slides be saved?")


def test_wrap_conversational_input_no_plan_uses_placeholder() -> None:
    """A session without a plan still renders cleanly — the wrapper
    uses ``(no summary)`` and ``(none yet)`` placeholders. This mirrors
    the pre-refactor behaviour for the (rare) follow-up turn that
    fires before any plan has been installed."""
    shaper = PromptShaper()
    sess = Session(run_id="r-no-plan")
    steerer = _make_steerer(observation_only=False)

    out = shaper.wrap_conversational_input(
        user_input="hi?",
        session=sess,
        steerer=steerer,
    )
    assert "Plan summary: (no summary)" in out
    assert "Completed tasks:\n  (none yet)" in out
    assert out.endswith("User question: hi?")


# ---------------------------------------------------------------------------
# Site 2 — inject_goldfive_planner_instruction
# ---------------------------------------------------------------------------
#
# The full inject pipeline requires google.adk + GoldfivePlanner; we
# unit-test the gate behaviour with a minimal stub here. The byte-
# identity check on the active path runs in
# ``test_goldfive_planner.py`` against the live ADK install.


class _FakeConfig:
    def __init__(self) -> None:
        self.system_instruction: Any = None


class _FakeLlmRequest:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.append_calls: list[list[str]] = []

    def append_instructions(self, instructions: list[str]) -> list:
        self.append_calls.append(list(instructions))
        return []


class _Ctx:
    """Minimal ``SessionContext``-shaped stub carrying just ``.steerer``."""

    def __init__(self, *, observation_only: bool) -> None:
        self.steerer = _make_steerer(observation_only=observation_only)
        self.session = None


def test_inject_goldfive_planner_instruction_noop_when_passive() -> None:
    """Under ``observation_only=True`` the inject is short-circuited
    BEFORE the ADK / GoldfivePlanner imports — ``llm_request`` is left
    untouched even if the rest of the pipeline would have run."""
    shaper = PromptShaper()
    llm_request = _FakeLlmRequest()

    asyncio.run(
        shaper.inject_goldfive_planner_instruction(
            callback_context=None,
            llm_request=llm_request,
            session_context=_Ctx(observation_only=True),
        )
    )

    assert llm_request.append_calls == []
    assert llm_request.config.system_instruction is None


def test_inject_goldfive_planner_instruction_no_agent_when_active() -> None:
    """Under ``observation_only=False`` with no reachable agent
    (callback_context has no ``_invocation_context.agent``) the inject
    is still a no-op — the pre-refactor silent-fall-through behaviour
    is preserved byte-identically."""
    shaper = PromptShaper()
    llm_request = _FakeLlmRequest()

    asyncio.run(
        shaper.inject_goldfive_planner_instruction(
            callback_context=None,
            llm_request=llm_request,
            session_context=_Ctx(observation_only=False),
        )
    )

    # Pipeline silently returned — no agent reachable → nothing to
    # inject. Same shape as pre-refactor.
    assert llm_request.append_calls == []
    assert llm_request.config.system_instruction is None


# ---------------------------------------------------------------------------
# Site 3 — inject_runtime_tools_hint
# ---------------------------------------------------------------------------


def test_inject_runtime_tools_hint_noop_when_passive() -> None:
    """Under ``observation_only=True`` the hint is suppressed — the
    request's ``system_instruction`` and any prior marker block are
    left untouched (no strip, no append)."""
    shaper = PromptShaper()
    sess = Session(
        run_id="r-test",
        plan=Plan(
            id="plan-1",
            run_id="r-test",
            goal_ids=[],
            tasks=[
                Task(
                    id="t1",
                    title="draft",
                    description="",
                    assignee_agent_id="writer",
                    status=TaskStatus.PENDING,
                )
            ],
            edges=[],
        ),
    )
    llm_request = _FakeLlmRequest()
    # Pre-stamp a prior hint so we can verify the strict-passive path
    # does NOT strip it (the gate is BEFORE the strip).
    from goldfive.adapters._adk_plugin import (
        _RUNTIME_TOOLS_HINT_END,
        _RUNTIME_TOOLS_HINT_PREFIX,
    )

    prior = (
        f"{_RUNTIME_TOOLS_HINT_PREFIX} stale]\n  writer: PENDING — old\n"
        f"{_RUNTIME_TOOLS_HINT_END}"
    )
    llm_request.config.system_instruction = prior

    shaper.inject_runtime_tools_hint(
        callback_context=None,
        llm_request=llm_request,
        session=sess,
        session_context=_Ctx(observation_only=True),
    )

    # Untouched: no append, no strip.
    assert llm_request.append_calls == []
    assert llm_request.config.system_instruction == prior


def test_inject_runtime_tools_hint_byte_identical_when_active() -> None:
    """Under ``observation_only=False`` the inject reproduces the
    pre-refactor write: a single ``append_instructions([hint])`` call
    whose payload is bracketed by the canonical marker tags."""
    from goldfive.adapters._adk_plugin import (
        _RUNTIME_TOOLS_HINT_END,
        _RUNTIME_TOOLS_HINT_PREFIX,
        _build_runtime_tools_hint,
    )

    shaper = PromptShaper()
    sess = Session(
        run_id="r-test",
        plan=Plan(
            id="plan-1",
            run_id="r-test",
            goal_ids=[],
            tasks=[
                Task(
                    id="t1",
                    title="draft",
                    description="",
                    assignee_agent_id="writer",
                    status=TaskStatus.PENDING,
                )
            ],
            edges=[],
        ),
    )
    llm_request = _FakeLlmRequest()

    shaper.inject_runtime_tools_hint(
        callback_context=None,
        llm_request=llm_request,
        session=sess,
        session_context=_Ctx(observation_only=False),
    )

    expected = _build_runtime_tools_hint(sess)
    assert expected is not None
    assert llm_request.append_calls == [[expected]]
    # Marker fingerprints present.
    assert _RUNTIME_TOOLS_HINT_PREFIX in expected
    assert _RUNTIME_TOOLS_HINT_END in expected


def test_inject_runtime_tools_hint_gate_open_with_no_session_context() -> None:
    """Without a ``SessionContext`` (test stubs) the gate defaults to
    open — the inject runs. This preserves the pre-#271 paths that
    drove the helper without ever wiring a steerer."""
    from goldfive.adapters._adk_plugin import _RUNTIME_TOOLS_HINT_PREFIX

    shaper = PromptShaper()
    sess = Session(
        run_id="r-test",
        plan=Plan(
            id="plan-1",
            run_id="r-test",
            goal_ids=[],
            tasks=[
                Task(
                    id="t1",
                    title="draft",
                    description="",
                    assignee_agent_id="writer",
                    status=TaskStatus.PENDING,
                )
            ],
            edges=[],
        ),
    )
    llm_request = _FakeLlmRequest()

    shaper.inject_runtime_tools_hint(
        callback_context=None,
        llm_request=llm_request,
        session=sess,
        session_context=None,
    )

    assert llm_request.append_calls, "inject must run when no SessionContext"
    assert _RUNTIME_TOOLS_HINT_PREFIX in llm_request.append_calls[0][0]


# ---------------------------------------------------------------------------
# Site 4 — make_dynamic_instruction
# ---------------------------------------------------------------------------


def _make_readonly_ctx(*, observation_only: bool, session: Session | None) -> Any:
    """Build a fake ReadonlyContext carrying the goldfive SessionContext
    stash via the legacy ``state["goldfive._session_context"]`` path."""
    from goldfive.adapters._adk_plugin import SessionContext

    steerer = _make_steerer(observation_only=observation_only)
    sess = session or Session(run_id="r-resolver-test")
    ctx_stash = SessionContext(
        session=sess,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coordinator",
    )

    class _ReadonlyCtx:
        state = {"goldfive._session_context": ctx_stash}
        _invocation_context = None

    return _ReadonlyCtx()


def test_make_dynamic_instruction_returns_original_when_passive() -> None:
    """Under ``observation_only=True`` the resolver returns
    ``original_instruction`` verbatim — no augmentation."""
    from goldfive.state_store import StateStore

    sess = Session(run_id="r-resolver-test")
    sess.goals = [Goal(id="g1", summary="x")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-resolver-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft slide",
                description="d",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            )
        ],
        edges=[],
        summary="x",
        revision_index=1,
    )
    store = StateStore.for_session(sess)
    store.set_pin_current_task("t1", title="Draft slide")

    resolver = PromptShaper().make_dynamic_instruction(
        original_instruction="you are a writer",
        agent_name="writer",
    )
    ctx = _make_readonly_ctx(observation_only=True, session=sess)

    out = resolver(ctx)
    assert out == "you are a writer"
    assert "Current assigned task" not in out


def test_make_dynamic_instruction_augments_when_active() -> None:
    """Under ``observation_only=False`` the resolver appends the
    "Current assigned task" block to the original instruction."""
    from goldfive.state_store import StateStore

    sess = Session(run_id="r-resolver-test")
    sess.goals = [Goal(id="g1", summary="x")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-resolver-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft slide",
                description="d",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            )
        ],
        edges=[],
        summary="x",
        revision_index=1,
    )
    store = StateStore.for_session(sess)
    store.set_pin_current_task("t1", title="Draft slide")

    resolver = PromptShaper().make_dynamic_instruction(
        original_instruction="you are a writer",
        agent_name="writer",
    )
    ctx = _make_readonly_ctx(observation_only=False, session=sess)

    out = resolver(ctx)
    assert out.startswith("you are a writer")
    assert "Current assigned task:" in out
    assert "id: t1" in out
    assert "Draft slide" in out


def test_make_dynamic_instruction_provenance_attrs() -> None:
    """The resolver must carry the legacy provenance attrs used by
    ``install_dynamic_instructions`` for idempotency checks."""
    resolver = PromptShaper().make_dynamic_instruction(
        original_instruction="hello",
        agent_name="writer",
    )

    assert getattr(resolver, "_goldfive_dynamic_instruction", False) is True
    assert getattr(resolver, "_goldfive_agent_name", "") == "writer"
    assert getattr(resolver, "_goldfive_original_instruction", "") == "hello"


def test_make_dynamic_instruction_no_pin_returns_original() -> None:
    """When no task is pinned the resolver returns
    ``original_instruction`` (regardless of the gate). Mirrors the
    pre-refactor "no augmentation when there's nothing to augment with"
    behaviour."""
    sess = Session(run_id="r-no-pin")
    resolver = PromptShaper().make_dynamic_instruction(
        original_instruction="raw",
        agent_name="writer",
    )
    ctx = _make_readonly_ctx(observation_only=False, session=sess)
    assert resolver(ctx) == "raw"
