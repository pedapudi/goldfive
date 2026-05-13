"""Tests for the R3 runtime tool-surface hint (Tier 1 F2 alternative).

Covers :func:`goldfive.adapters._adk_plugin._build_runtime_tools_hint`
and :meth:`goldfive.prompt_shaper.PromptShaper.inject_runtime_tools_hint`
plus the wiring through ``before_model_callback`` such that the hint
lands on ``llm_request.config.system_instruction`` without clobbering
prior instructions and without accumulating across calls.

Runs without ADK installed because the helpers are pure-Python and do
not import ADK at module load. Wiring tests skip when ADK is missing.

Wave B1 (refactor/prompt-shaper): the inject helper moved off
:mod:`goldfive.adapters._adk_plugin` and onto :class:`PromptShaper`.
A thin module-local shim preserves the legacy ``_inject_runtime_tools_hint(...)``
call shape so the test bodies below read unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.adapters._adk_plugin import (
    _RUNTIME_TOOLS_HINT_END,
    _RUNTIME_TOOLS_HINT_PREFIX,
    _build_runtime_tools_hint,
    _strip_prior_runtime_tools_hint,
)
from goldfive.prompt_shaper import PromptShaper
from goldfive.types import Plan, Session, Task, TaskStatus


def _inject_runtime_tools_hint(
    *, callback_context: Any, llm_request: Any, session: Any
) -> None:
    """Wave B1 shim: forward to :meth:`PromptShaper.inject_runtime_tools_hint`.

    These unit tests drive the helper without a ``SessionContext`` so the
    gate falls back to "inject" (steerer is None → ``should_inject`` →
    True) — exactly the pre-refactor behaviour.
    """
    PromptShaper().inject_runtime_tools_hint(
        callback_context=callback_context,
        llm_request=llm_request,
        session=session,
        session_context=None,
    )

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self) -> None:
        self.system_instruction: str | None = None


class _FakeLlmRequest:
    """LlmRequest stub mimicking ADK's ``append_instructions`` + ``config``."""

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.append_calls: list[list[str]] = []

    def append_instructions(self, instructions: list[str]) -> list:
        self.append_calls.append(list(instructions))
        new_text = "\n\n".join(instructions)
        if not self.config.system_instruction:
            self.config.system_instruction = new_text
        else:
            self.config.system_instruction += "\n\n" + new_text
        return []


def _make_session(tasks: list[Task] | None, *, run_id: str = "run-1") -> Session:
    if tasks is None:
        return Session(run_id=run_id, plan=None)
    plan = Plan(id="plan-1", run_id=run_id, goal_ids=[], tasks=tasks, edges=[])
    return Session(run_id=run_id, plan=plan)


# ---------------------------------------------------------------------------
# _build_runtime_tools_hint
# ---------------------------------------------------------------------------


def test_no_plan_returns_none() -> None:
    """A session with ``plan=None`` produces no hint."""
    session = _make_session(None)
    assert _build_runtime_tools_hint(session) is None


def test_empty_plan_returns_none() -> None:
    """A plan with zero tasks produces no hint."""
    session = _make_session([])
    assert _build_runtime_tools_hint(session) is None


def test_multi_agent_plan_lists_pending_and_done() -> None:
    """Mixed plan: one PENDING agent, one all-COMPLETED agent."""
    tasks = [
        Task(
            id="t1",
            title="Research the market",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
        Task(
            id="t2",
            title="Build the landing page",
            assignee_agent_id="web_developer_agent",
            status=TaskStatus.COMPLETED,
        ),
    ]
    session = _make_session(tasks)

    hint = _build_runtime_tools_hint(session)

    assert hint is not None
    assert "research_agent: PENDING" in hint
    assert "Research the market" in hint
    assert "web_developer_agent: all assigned tasks complete" in hint
    assert "do NOT re-invoke" in hint


def test_all_complete_emits_do_not_re_invoke() -> None:
    """A plan whose only tasks are all terminal — every line says 'do NOT re-invoke'."""
    tasks = [
        Task(
            id="t1",
            title="A",
            assignee_agent_id="research_agent",
            status=TaskStatus.COMPLETED,
        ),
        Task(
            id="t2",
            title="B",
            assignee_agent_id="research_agent",
            status=TaskStatus.NOT_NEEDED,
        ),
    ]
    session = _make_session(tasks)

    hint = _build_runtime_tools_hint(session)

    assert hint is not None
    assert "research_agent: all assigned tasks complete" in hint
    # No PENDING line for that agent
    assert "research_agent: PENDING" not in hint


def test_terminal_statuses_all_recognised() -> None:
    """COMPLETED, FAILED, CANCELLED, NOT_NEEDED all count as 'done'."""
    statuses = [
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.NOT_NEEDED,
    ]
    for st in statuses:
        tasks = [
            Task(id="t1", title="X", assignee_agent_id="alpha", status=st),
        ]
        hint = _build_runtime_tools_hint(_make_session(tasks))
        assert hint is not None, f"hint missing for status={st}"
        assert "alpha: all assigned tasks complete" in hint, f"status={st}"


def test_hint_format_marker_present() -> None:
    """The prefix marker MUST be present so dedup logic can find prior hints."""
    tasks = [
        Task(
            id="t1",
            title="Anything",
            assignee_agent_id="a",
            status=TaskStatus.PENDING,
        ),
    ]
    hint = _build_runtime_tools_hint(_make_session(tasks))
    assert hint is not None
    assert hint.startswith(_RUNTIME_TOOLS_HINT_PREFIX)
    assert _RUNTIME_TOOLS_HINT_END in hint
    assert "Choose the agent whose tasks are still PENDING" in hint


def test_namespaced_agent_id_strips_namespace() -> None:
    """``ns:agent_name`` agent ids are surfaced as the bare name."""
    tasks = [
        Task(
            id="t1",
            title="Probe",
            assignee_agent_id="my.namespace:research_agent",
            status=TaskStatus.PENDING,
        ),
    ]
    hint = _build_runtime_tools_hint(_make_session(tasks))
    assert hint is not None
    assert "research_agent: PENDING" in hint
    # The full namespaced form should not appear as the line key.
    assert "my.namespace:research_agent: PENDING" not in hint


def test_pending_summary_caps_at_three_titles() -> None:
    """Long PENDING lists are truncated to three titles for prompt brevity."""
    tasks = [
        Task(id=f"t{i}", title=f"Task {i}", assignee_agent_id="a", status=TaskStatus.PENDING)
        for i in range(5)
    ]
    hint = _build_runtime_tools_hint(_make_session(tasks))
    assert hint is not None
    # First three present
    assert "Task 0" in hint and "Task 1" in hint and "Task 2" in hint
    # 4th + 5th omitted
    assert "Task 3" not in hint
    assert "Task 4" not in hint


# ---------------------------------------------------------------------------
# _strip_prior_runtime_tools_hint
# ---------------------------------------------------------------------------


def test_strip_no_marker_is_passthrough() -> None:
    """Strings without the prefix are returned unchanged."""
    assert _strip_prior_runtime_tools_hint("hello world") == "hello world"


def test_strip_removes_full_marker_block() -> None:
    """A complete hint block bracketed by both markers is removed cleanly."""
    body = (
        "USER PROMPT TEXT\n\n"
        f"{_RUNTIME_TOOLS_HINT_PREFIX} runtime guidance, not user-authored]\n"
        "  research_agent: PENDING — Do thing\n"
        f"{_RUNTIME_TOOLS_HINT_END}\n\n"
        "TRAILING TEXT"
    )
    out = _strip_prior_runtime_tools_hint(body)
    assert _RUNTIME_TOOLS_HINT_PREFIX not in out
    assert _RUNTIME_TOOLS_HINT_END not in out
    assert "USER PROMPT TEXT" in out
    assert "TRAILING TEXT" in out
    # No tripled newlines
    assert "\n\n\n" not in out


def test_strip_handles_truncated_marker() -> None:
    """A prefix without the closing marker is dropped from prefix to end."""
    body = f"USER\n\n{_RUNTIME_TOOLS_HINT_PREFIX} ...]\n  research_agent: PENDING"
    out = _strip_prior_runtime_tools_hint(body)
    assert _RUNTIME_TOOLS_HINT_PREFIX not in out
    assert "USER" in out


# ---------------------------------------------------------------------------
# _inject_runtime_tools_hint
# ---------------------------------------------------------------------------


def test_injection_appends_when_no_prior_instruction() -> None:
    """Empty system_instruction gets the hint via append_instructions."""
    tasks = [
        Task(
            id="t1",
            title="Probe",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
    ]
    session = _make_session(tasks)
    req = _FakeLlmRequest()

    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=session)

    assert req.config.system_instruction is not None
    assert _RUNTIME_TOOLS_HINT_PREFIX in req.config.system_instruction
    assert "research_agent: PENDING" in req.config.system_instruction
    assert req.append_calls == [[req.config.system_instruction]] or len(req.append_calls) == 1


def test_injection_preserves_existing_user_system_instruction() -> None:
    """A pre-existing system_instruction (the user's coordinator prompt) survives."""
    tasks = [
        Task(
            id="t1",
            title="Probe",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
    ]
    session = _make_session(tasks)
    req = _FakeLlmRequest()
    req.config.system_instruction = "You are a coordinator. Delegate to specialists."

    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=session)

    instr = req.config.system_instruction
    assert instr is not None
    assert "You are a coordinator." in instr
    assert _RUNTIME_TOOLS_HINT_PREFIX in instr
    # Hint comes AFTER the user's prompt (append semantics).
    assert instr.index("You are a coordinator.") < instr.index(_RUNTIME_TOOLS_HINT_PREFIX)


def test_injection_replaces_prior_hint_no_accumulation() -> None:
    """A second injection swaps the stale hint for the fresh one."""
    req = _FakeLlmRequest()
    req.config.system_instruction = "USER PROMPT"

    # First call: research_agent has 1 PENDING task.
    s1 = _make_session(
        [
            Task(
                id="t1",
                title="First task",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
            )
        ]
    )
    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=s1)
    after_first = req.config.system_instruction
    assert after_first is not None
    assert "First task" in after_first
    assert after_first.count(_RUNTIME_TOOLS_HINT_PREFIX) == 1

    # Second call: same agent, task is now COMPLETED.
    s2 = _make_session(
        [
            Task(
                id="t1",
                title="First task",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            )
        ]
    )
    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=s2)
    after_second = req.config.system_instruction
    assert after_second is not None
    # User prompt is still present.
    assert "USER PROMPT" in after_second
    # Exactly ONE hint block (no accumulation).
    assert after_second.count(_RUNTIME_TOOLS_HINT_PREFIX) == 1
    assert after_second.count(_RUNTIME_TOOLS_HINT_END) == 1
    # The fresh hint reflects the new state.
    assert "all assigned tasks complete" in after_second
    # The stale "PENDING — First task" line is gone.
    assert "PENDING — First task" not in after_second


def test_injection_no_plan_strips_stale_hint() -> None:
    """When the plan disappears, an existing hint should be stripped, not left dangling."""
    req = _FakeLlmRequest()
    req.config.system_instruction = (
        "USER PROMPT\n\n"
        f"{_RUNTIME_TOOLS_HINT_PREFIX} runtime guidance, not user-authored]\n"
        "  research_agent: PENDING — Old task\n"
        f"{_RUNTIME_TOOLS_HINT_END}"
    )

    session = _make_session(None)
    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=session)

    instr = req.config.system_instruction
    assert instr is not None
    assert "USER PROMPT" in instr
    assert _RUNTIME_TOOLS_HINT_PREFIX not in instr
    assert _RUNTIME_TOOLS_HINT_END not in instr


def test_injection_no_plan_no_prior_hint_is_noop() -> None:
    """No plan AND no prior hint: nothing changes (no accidental polluting)."""
    req = _FakeLlmRequest()
    req.config.system_instruction = "USER PROMPT"

    session = _make_session(None)
    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=session)

    assert req.config.system_instruction == "USER PROMPT"
    assert req.append_calls == []


def test_injection_handles_missing_config_attr() -> None:
    """A request without ``config`` is a silent no-op (defensive path)."""

    class _Bare:
        pass

    bare = _Bare()
    session = _make_session(
        [
            Task(
                id="t1",
                title="X",
                assignee_agent_id="a",
                status=TaskStatus.PENDING,
            )
        ]
    )
    # Should not raise.
    _inject_runtime_tools_hint(callback_context=None, llm_request=bare, session=session)


def test_injection_falls_back_to_direct_write_without_append_helper() -> None:
    """When LlmRequest lacks ``append_instructions``, write directly to system_instruction."""

    class _BareReq:
        def __init__(self) -> None:
            self.config = _FakeConfig()

    req: Any = _BareReq()
    session = _make_session(
        [
            Task(
                id="t1",
                title="X",
                assignee_agent_id="a",
                status=TaskStatus.PENDING,
            )
        ]
    )
    _inject_runtime_tools_hint(callback_context=None, llm_request=req, session=session)
    assert req.config.system_instruction is not None
    assert _RUNTIME_TOOLS_HINT_PREFIX in req.config.system_instruction


# ---------------------------------------------------------------------------
# before_model_callback wiring (requires ADK)
# ---------------------------------------------------------------------------

try:  # pragma: no cover — env-dependent
    import google.adk  # noqa: F401

    _ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ADK_AVAILABLE = False

skip_without_adk = pytest.mark.skipif(
    not _ADK_AVAILABLE, reason="ADK wiring tests require google-adk"
)

if _ADK_AVAILABLE:
    from goldfive.adapters._adk_plugin import (  # noqa: E402
        SessionContext,
        make_adk_plugin,
    )


class _FakeInvocationContext:
    def __init__(self) -> None:
        self.invocation_id = "inv-r3"
        self.agent = None
        self.session = None
        self.user_content = None
        self.user_id = "u"
        self.branch = None
        self.run_config = None


class _FakeCallbackContext:
    def __init__(self, inv_ctx: _FakeInvocationContext) -> None:
        self._invocation_context = inv_ctx


@skip_without_adk
@pytest.mark.asyncio
async def test_before_model_callback_injects_runtime_tools_hint() -> None:
    """End-to-end through ``before_model_callback``: hint lands on system_instruction."""
    plugin = make_adk_plugin()

    tasks = [
        Task(
            id="t1",
            title="Survey",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
        Task(
            id="t2",
            title="Build site",
            assignee_agent_id="web_developer_agent",
            status=TaskStatus.COMPLETED,
        ),
    ]
    session = _make_session(tasks)
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=tasks[0],
        tool_handlers={},
        host_agent_name="coordinator",
    )
    plugin.set_active_context(ctx)

    inv_ctx = _FakeInvocationContext()
    cb = _FakeCallbackContext(inv_ctx)
    req = _FakeLlmRequest()
    req.config.system_instruction = "USER COORDINATOR PROMPT"

    try:
        result = await plugin.before_model_callback(callback_context=cb, llm_request=req)
        # Non-None return short-circuits ADK; the hint path must NOT do that.
        assert result is None
        instr = req.config.system_instruction
        assert instr is not None
        assert "USER COORDINATOR PROMPT" in instr
        assert _RUNTIME_TOOLS_HINT_PREFIX in instr
        assert "research_agent: PENDING" in instr
        assert "web_developer_agent: all assigned tasks complete" in instr
    finally:
        plugin.clear_active_context()


@skip_without_adk
@pytest.mark.asyncio
async def test_before_model_callback_no_session_is_noop_for_hint() -> None:
    """No active SessionContext: hint injection skips, callback still returns None."""
    plugin = make_adk_plugin()
    # Deliberately no set_active_context.

    inv_ctx = _FakeInvocationContext()
    cb = _FakeCallbackContext(inv_ctx)
    req = _FakeLlmRequest()
    req.config.system_instruction = "USER PROMPT"

    result = await plugin.before_model_callback(callback_context=cb, llm_request=req)
    assert result is None
    # No hint added.
    assert req.config.system_instruction == "USER PROMPT"
    assert _RUNTIME_TOOLS_HINT_PREFIX not in (req.config.system_instruction or "")
