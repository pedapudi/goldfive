"""Refinement-guidance block tests (goldfive#241 Item 1).

When the planner's refine-prompt builders render their user prompt,
they must include a REFINEMENT GUIDANCE block that steers the
planner-LLM toward the conservative correction pattern (preserve the
drifted task's assignee, preserve the DAG shape, emit ``supersedes``
on replacements) rather than reshaping the plan on every small drift.

The live evidence behind this block: a research_agent drift on a
multi-stage plan caused the LLM to collapse 5 stages to 1 task and
reassign it to web_developer_agent. The correction never ran on
research_agent and the replacement landed on an agent with no
relevant context. See goldfive#241 for the full trace.

These tests are pure prompt-text assertions — the goal is to pin
that every refine path that can reshape the plan on a drift carries
the guidance block. No LLM is invoked.
"""

from __future__ import annotations

import json

from goldfive.planner import (
    _REFINEMENT_GUIDANCE_BLOCK,
    LLMPlanner,
)
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_goals() -> list[Goal]:
    return [Goal(id="g1", summary="Ship the thing.")]


def _plan_with_running_task() -> Plan:
    return Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research",
                title="Research goldfish facts",
                description="Gather facts.",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft the post",
                description="Write a 500-word draft.",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="review",
                title="Review the draft",
                description="Editorial pass.",
                assignee_agent_id="editor",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
            TaskEdge(from_task_id="draft", to_task_id="review"),
        ],
        summary="Blog post pipeline.",
        revision_index=0,
    )


# ---------------------------------------------------------------------------
# Prompt-text assertions
# ---------------------------------------------------------------------------


def test_guidance_block_constant_contains_key_instructions() -> None:
    """Sanity-pin the constant's wording so accidental edits that
    dilute the guidance fail loudly. The load-bearing phrases are the
    leave-assignee-empty rule (goldfive#252), the supersedes
    requirement, and the don't-collapse-stages clause."""
    text = _REFINEMENT_GUIDANCE_BLOCK
    assert "REFINEMENT GUIDANCE" in text
    # goldfive#252: assignee is observational; the prompt now tells the
    # LLM to leave the field empty rather than to "keep the same"
    # value the planner previously emitted.
    assert "Leave `assignee_agent_id` empty" in text
    assert "supersedes" in text
    assert "Do NOT collapse a multi-stage plan to a single task" in text


def test_guidance_appears_in_steer_prompt_user_source() -> None:
    """``_build_steer_prompt(source='user')`` is the user-steer path;
    must carry the guidance."""
    planner = LLMPlanner(call_llm=_StubLLM(response=""))
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="focus on X instead",
        current_task_id="draft",
    )
    prompt = planner._build_steer_prompt(
        completed=[_plan_with_running_task().tasks[0]],
        drift=drift,
        goals=_base_goals(),
        source="user",
    )
    assert "REFINEMENT GUIDANCE" in prompt
    # goldfive#252: prompt now tells the LLM to leave assignee empty.
    assert "Leave `assignee_agent_id` empty" in prompt


def test_guidance_appears_in_steer_prompt_goldfive_source() -> None:
    """``_build_steer_prompt(source='goldfive')`` is the promoted
    goldfive-detected-drift path (post-#240 steer unification); must
    carry the guidance."""
    planner = LLMPlanner(call_llm=_StubLLM(response=""))
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="research_agent wandered into raccoons",
        current_task_id="research",
    )
    prompt = planner._build_steer_prompt(
        completed=[],
        drift=drift,
        goals=_base_goals(),
        source="goldfive",
    )
    assert "REFINEMENT GUIDANCE" in prompt
    assert "supersedes" in prompt


def test_guidance_appears_in_generic_refine_prompt() -> None:
    """``_build_refine_prompt`` is the generic refine entry — used by
    TOOL_ERROR, AGENT_REFUSAL, NEW_WORK_DISCOVERED, etc."""
    planner = LLMPlanner(call_llm=_StubLLM(response=""))
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool foo returned 500",
        current_task_id="draft",
    )
    prompt = planner._build_refine_prompt(
        plan=_plan_with_running_task(),
        drift=drift,
        goals=_base_goals(),
    )
    assert "REFINEMENT GUIDANCE" in prompt
    assert "Do NOT collapse a multi-stage plan" in prompt


def test_guidance_appears_in_looping_tool_call_prompt() -> None:
    """``_build_looping_tool_call_prompt`` is the fail-and-regenerate
    path; must carry the guidance too — the "route around a loop"
    repair is structurally the same kind of reshape decision."""
    planner = LLMPlanner(call_llm=_StubLLM(response=""))
    plan = _plan_with_running_task()
    looping = plan.tasks[1]  # draft
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="writer stuck in a search loop",
        current_task_id="draft",
    )
    prompt = planner._build_looping_tool_call_prompt(
        plan=plan,
        drift=drift,
        goals=_base_goals(),
        looping_task=looping,
    )
    assert "REFINEMENT GUIDANCE" in prompt
    assert "assignee_agent_id" in prompt


# ---------------------------------------------------------------------------
# End-to-end: refine() threads the guidance through to the LLM call
# ---------------------------------------------------------------------------


async def test_refine_guidance_reaches_llm_via_plan_divergence() -> None:
    """Send a PLAN_DIVERGENCE drift (no observed_actions) through the
    generic refine path and assert the captured LLM user prompt
    carries the guidance block. This is the outermost assertion:
    short-circuiting the prompt builder without updating refine()
    would still pass the direct-builder tests above."""
    # Echo plan — valid revision returning the plan unchanged.
    plan = _plan_with_running_task()
    echo = json.dumps(
        {
            "summary": plan.summary,
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "assignee_agent_id": t.assignee_agent_id,
                    "status": str(t.status),
                }
                for t in plan.tasks
            ],
            "edges": [
                {"from_task_id": e.from_task_id, "to_task_id": e.to_task_id}
                for e in plan.edges
            ],
        }
    )
    stub = _StubLLM(response=echo)
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="observed agent activity diverges",
        current_task_id="draft",
    )
    await planner.refine(plan=plan, drift=drift, goals=_base_goals())
    assert stub.calls, "LLM was not invoked"
    _system, user_prompt, _model = stub.calls[0]
    assert "REFINEMENT GUIDANCE" in user_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal call_llm stub that captures prompts."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response
