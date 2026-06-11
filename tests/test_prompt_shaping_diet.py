"""Unit tests for the AGENCY-PRESERVATION.md PR 9 prompt-shaping diet.

The diet is gated on ``signal_channel == "request_context"``. Under the
default ``"legacy_user_message"`` channel every site behaves exactly as
pre-PR-9 (covered by ``test_prompt_shaper.py`` / the other suites, which
run with the legacy default and pass unmodified — §5.1). These tests
cover the NEW ``request_context`` paths:

* Site 1 — conversational wrap keeps plan context but drops the
  "Do NOT call any AgentTool" / "don't delegate" means-directive.
* Site 3 — the per-turn runtime tool-surface hint is retired (no
  injection); a stale prior hint is still stripped.
* Site 4 — the ``[CURRENT ASSIGNED TASK]`` pin is retired by default;
  ``pin_assigned_task`` is the escape hatch.
* Site 5 — the retired hint's plan-state folds into the observer-note
  block (factual, no imperative; marker count stays 1).
* config — ``pin_assigned_task`` default + env.
"""

from __future__ import annotations

import os
from typing import Any

from goldfive.config import SteeringConfig
from goldfive.prompt_shaper import PromptShaper
from goldfive.steerer import DefaultSteerer
from goldfive.types import Goal, Plan, Session, Task, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _steerer(
    *,
    channel: str = "request_context",
    observation_only: bool = False,
    pin_assigned_task: bool = False,
) -> DefaultSteerer:
    return DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=observation_only,
            signal_channel=channel,
            pin_assigned_task=pin_assigned_task,
        )
    )


class _FakeConfig:
    def __init__(self, system_instruction: Any = None) -> None:
        self.system_instruction = system_instruction


class _FakeReq:
    """LLM-request stub whose ``append_instructions`` appends to system_instruction."""

    def __init__(self, system_instruction: Any = None) -> None:
        self.config = _FakeConfig(system_instruction)

    def append_instructions(self, instructions: list[str]) -> list:
        joined = "\n\n".join(instructions)
        existing = self.config.system_instruction
        if not existing:
            self.config.system_instruction = joined
        else:
            self.config.system_instruction = f"{existing}\n\n{joined}"
        return []


class _Ctx:
    def __init__(self, steerer: DefaultSteerer, session: Session | None = None) -> None:
        self.steerer = steerer
        self.session = session


def _session_with_plan() -> Session:
    sess = Session(run_id="r-diet")
    sess.goals = [Goal(id="g1", summary="ship the report")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-diet",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Draft",
                description="d",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="t2",
                title="Review",
                description="d2",
                assignee_agent_id="reviewer",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
        summary="a report",
        revision_index=1,
    )
    return sess


def _readonly_ctx(steerer: DefaultSteerer, session: Session) -> Any:
    from goldfive.adapters._adk_plugin import SessionContext

    stash = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coordinator",
    )

    class _ReadonlyCtx:
        state = {"goldfive._session_context": stash}
        _invocation_context = None

    return _ReadonlyCtx()


# ---------------------------------------------------------------------------
# Site 1 — conversational wrap
# ---------------------------------------------------------------------------


def test_site1_request_context_drops_means_directive() -> None:
    sess = _session_with_plan()
    out = PromptShaper().wrap_conversational_input(
        user_input="what changed?",
        session=sess,
        steerer=_steerer(channel="request_context"),
    )
    # Means-directive removed.
    assert "Do NOT call any AgentTool" not in out
    assert "don't delegate" not in out
    # Context kept.
    assert "Plan summary: a report" in out
    assert "what changed?" in out


def test_site1_legacy_keeps_means_directive() -> None:
    sess = _session_with_plan()
    out = PromptShaper().wrap_conversational_input(
        user_input="what changed?",
        session=sess,
        steerer=_steerer(channel="legacy_user_message"),
    )
    assert "Do NOT call any AgentTool" in out
    assert "Plan summary: a report" in out


# ---------------------------------------------------------------------------
# Site 3 — runtime tool-surface hint
# ---------------------------------------------------------------------------


def test_site3_request_context_suppresses_hint() -> None:
    sess = _session_with_plan()
    req = _FakeReq(system_instruction="base")
    PromptShaper().inject_runtime_tools_hint(
        callback_context=None,
        llm_request=req,
        session=sess,
        session_context=_Ctx(_steerer(channel="request_context")),
    )
    # No standalone hint injected in the diet regime.
    assert "PLAN-STATE HINT" not in (req.config.system_instruction or "")
    assert req.config.system_instruction == "base"


def test_site3_legacy_injects_hint() -> None:
    sess = _session_with_plan()
    req = _FakeReq(system_instruction="base")
    PromptShaper().inject_runtime_tools_hint(
        callback_context=None,
        llm_request=req,
        session=sess,
        session_context=_Ctx(_steerer(channel="legacy_user_message")),
    )
    assert "PLAN-STATE HINT" in (req.config.system_instruction or "")


def test_site3_request_context_strips_stale_prior_hint() -> None:
    """A hint left from a legacy-channel turn is stripped under the diet."""
    from goldfive.adapters._adk_plugin import _build_runtime_tools_hint

    sess = _session_with_plan()
    prior_hint = _build_runtime_tools_hint(sess)
    assert prior_hint  # sanity
    req = _FakeReq(system_instruction=f"base\n\n{prior_hint}")
    PromptShaper().inject_runtime_tools_hint(
        callback_context=None,
        llm_request=req,
        session=sess,
        session_context=_Ctx(_steerer(channel="request_context")),
    )
    assert "PLAN-STATE HINT" not in (req.config.system_instruction or "")
    assert "base" in (req.config.system_instruction or "")


# ---------------------------------------------------------------------------
# Site 4 — [CURRENT ASSIGNED TASK] pin
# ---------------------------------------------------------------------------


def _pin_session() -> Session:
    from goldfive.state_store import StateStore

    sess = Session(run_id="r-pin")
    sess.goals = [Goal(id="g1", summary="x")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-pin",
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
    StateStore.for_session(sess).set_pin_current_task("t1", title="Draft slide")
    return sess


def test_site4_request_context_retires_pin() -> None:
    sess = _pin_session()
    resolver = PromptShaper().make_dynamic_instruction("you are a writer", "writer")
    out = resolver(_readonly_ctx(_steerer(channel="request_context"), sess))
    assert out == "you are a writer"
    assert "Current assigned task" not in out


def test_site4_pin_assigned_task_escape_hatch_re_enables() -> None:
    sess = _pin_session()
    resolver = PromptShaper().make_dynamic_instruction("you are a writer", "writer")
    out = resolver(
        _readonly_ctx(
            _steerer(channel="request_context", pin_assigned_task=True), sess
        )
    )
    assert "Current assigned task:" in out
    assert "id: t1" in out


def test_site4_legacy_keeps_pin() -> None:
    sess = _pin_session()
    resolver = PromptShaper().make_dynamic_instruction("you are a writer", "writer")
    out = resolver(_readonly_ctx(_steerer(channel="legacy_user_message"), sess))
    assert "Current assigned task:" in out


# ---------------------------------------------------------------------------
# Site 5 — plan-state fold into the observer note
# ---------------------------------------------------------------------------


def test_site5_folds_plan_state_into_note_block() -> None:
    from goldfive.observer_note_queue import (
        OBSERVER_NOTE_MARKER_PREFIX,
        ObserverNoteQueue,
    )

    sess = _session_with_plan()
    ObserverNoteQueue.for_session(sess).enqueue(
        body="Observation: a loop was observed.",
        observation="a loop was observed",
        severity="warning",
        drift_id="d1",
        kind="looping_tool_call",
        task_id="t1",
    )
    req = _FakeReq(system_instruction="base")
    note = PromptShaper().inject_observer_note(
        llm_request=req,
        session=sess,
        session_context=_Ctx(_steerer(channel="request_context"), session=sess),
    )
    assert note is not None
    si = req.config.system_instruction or ""
    # Exactly one block; plan-state folded in (factual, no imperative).
    assert si.count(OBSERVER_NOTE_MARKER_PREFIX) == 1
    assert "Plan state (goldfive bookkeeping)" in si
    assert "writer: 1 open" in si
    assert "reviewer: no open tasks" in si
    assert "Choose the agent" not in si  # the dropped imperative
    assert "do NOT re-invoke" not in si  # the dropped imperative


def test_site5_plan_state_does_not_stack_across_calls() -> None:
    from goldfive.observer_note_queue import (
        OBSERVER_NOTE_MARKER_PREFIX,
        ObserverNoteQueue,
    )

    sess = _session_with_plan()
    ObserverNoteQueue.for_session(sess).enqueue(
        body="Observation: x.",
        observation="x",
        severity="warning",
        drift_id="d1",
        kind="looping_tool_call",
        task_id="t1",
    )
    shaper = PromptShaper()
    steerer = _steerer(channel="request_context")
    req = _FakeReq(system_instruction="base")
    shaper.inject_observer_note(
        llm_request=req, session=sess, session_context=_Ctx(steerer, session=sess)
    )
    # Second call (note already delivered → nothing pending) must strip the
    # prior block, leaving zero — never stacking the plan-state line.
    shaper.inject_observer_note(
        llm_request=req, session=sess, session_context=_Ctx(steerer, session=sess)
    )
    si = req.config.system_instruction or ""
    assert si.count(OBSERVER_NOTE_MARKER_PREFIX) == 0
    assert "Plan state (goldfive bookkeeping)" not in si
    assert "base" in si


def test_plan_state_line_empty_without_plan() -> None:
    sess = Session(run_id="r-empty")
    assert PromptShaper._plan_state_line(sess) == ""


def test_site3_anti_reinvoke_fact_survives_in_note_status() -> None:
    """The retired hint's anti-re-invoke protection ("all assigned tasks
    complete") must survive via the note's folded Status line when an
    agent's tasks are ALL terminal AND a note is pending."""
    from goldfive.observer_note_queue import ObserverNoteQueue

    sess = Session(run_id="r-terminal")
    sess.goals = [Goal(id="g1", summary="x")]
    sess.plan = Plan(
        id="plan-1",
        run_id="r-terminal",
        goal_ids=["g1"],
        tasks=[
            # researcher: every task terminal (one COMPLETED, one FAILED).
            Task(
                id="t1",
                title="Gather",
                description="d",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2",
                title="Verify",
                description="d",
                assignee_agent_id="researcher",
                status=TaskStatus.FAILED,
            ),
            # writer: still has open work.
            Task(
                id="t3",
                title="Draft",
                description="d",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
        ],
        edges=[],
        summary="x",
        revision_index=1,
    )
    ObserverNoteQueue.for_session(sess).enqueue(
        body="Observation: a signal fired.",
        observation="a signal fired",
        severity="warning",
        drift_id="d1",
        kind="looping_tool_call",
        task_id="t3",
    )
    req = _FakeReq(system_instruction="base")
    note = PromptShaper().inject_observer_note(
        llm_request=req,
        session=sess,
        session_context=_Ctx(_steerer(channel="request_context"), session=sess),
    )
    assert note is not None
    si = req.config.system_instruction or ""
    # The "no remaining tracked tasks" fact for the fully-terminal agent
    # survives — factual, no imperative.
    assert "researcher: no open tasks" in si
    assert "writer: 1 open" in si
    assert "do NOT re-invoke" not in si


def test_site3_dormant_when_no_note_pending() -> None:
    """No pending note → no hint, no plan-state, nothing injected anywhere.

    With Site 3's standalone hint retired under request_context, dormancy
    wins when nothing is pending: neither inject_runtime_tools_hint nor
    inject_observer_note touches the request."""
    sess = _session_with_plan()
    steerer = _steerer(channel="request_context")

    # Site 3 standalone hint: suppressed.
    req = _FakeReq(system_instruction="base instruction")
    PromptShaper().inject_runtime_tools_hint(
        callback_context=None,
        llm_request=req,
        session=sess,
        session_context=_Ctx(steerer),
    )
    assert req.config.system_instruction == "base instruction"

    # Observer note: queue empty → nothing rendered, no plan-state line.
    note = PromptShaper().inject_observer_note(
        llm_request=req,
        session=sess,
        session_context=_Ctx(steerer, session=sess),
    )
    assert note is None
    assert req.config.system_instruction == "base instruction"
    assert "Plan state (goldfive bookkeeping)" not in req.config.system_instruction


# ---------------------------------------------------------------------------
# config — pin_assigned_task
# ---------------------------------------------------------------------------


def test_pin_assigned_task_default_false() -> None:
    assert SteeringConfig().pin_assigned_task is False


def test_pin_assigned_task_env_override() -> None:
    prev = os.environ.get("GOLDFIVE_STEER_PIN_ASSIGNED_TASK")
    os.environ["GOLDFIVE_STEER_PIN_ASSIGNED_TASK"] = "1"
    try:
        assert SteeringConfig.from_env().pin_assigned_task is True
    finally:
        if prev is None:
            os.environ.pop("GOLDFIVE_STEER_PIN_ASSIGNED_TASK", None)
        else:
            os.environ["GOLDFIVE_STEER_PIN_ASSIGNED_TASK"] = prev
