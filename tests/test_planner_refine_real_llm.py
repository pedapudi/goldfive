"""Real-LLM prompt-quality tests for ``LLMPlanner.refine``.

The mock-LLM tests under ``tests/test_plan_refinement.py`` /
``tests/test_planner_refine_guidance.py`` pin the refine plumbing
(parsing, supersedes, drift dispatch) but cannot detect prompt
regressions: a strict scripted callable returns the same canned JSON
regardless of how the prompt is worded.

This file exercises the real ``_REFINE_SYSTEM_PROMPT`` against a live
OpenAI-compatible endpoint (kikuchi-hosted Qwen, vLLM, llama.cpp, LM
Studio, etc.). The scenario reproduces the v15 leakage pattern where
the refine LLM, faced with a borderline-empty drift detail, narrated
its introspection into the ``summary`` field:

    plan.summary = "Plan unchanged as drift event indicates new work
                    discovered without specific details requiring
                    task..."

The strengthened ``summary`` schema description forbids that meta-
commentary; this test is the regression pin.

Gating: skipped unless ``OPENAI_API_BASE`` is set in the environment.
The CI environment doesn't set this, so the tests are no-ops there;
run them locally against kikuchi via:

    OPENAI_API_BASE=http://kikuchi.lan:8080/v1 \\
    OPENAI_API_KEY=sk-anything \\
    USER_MODEL_NAME=Qwen3.5-35B-A3B-Q4_K_M.gguf \\
    uv run pytest tests/test_planner_refine_real_llm.py -v

One LLM round-trip; expected wall time ~10-60 s on a Qwen-class local
model.
"""

from __future__ import annotations

import os

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = [
    pytest.mark.skipif(
        not ensure_pb_available(),
        reason="goldfive protobuf stubs not available (install the `dev` extra)",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_BASE"),
        reason="OPENAI_API_BASE not set — real-LLM prompt tests skipped",
    ),
]

from goldfive import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    LLMPlanner,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Real-LLM call_llm callable (mirrors test_planner_handle_turn_real_llm.py).
# ---------------------------------------------------------------------------


def _model_name() -> str:
    """Resolve the model identifier the planner should request.

    Order of precedence: ``USER_MODEL_NAME`` → ``GOLDFIVE_TEST_MODEL`` →
    a kikuchi-friendly default.
    """
    return (
        os.environ.get("USER_MODEL_NAME")
        or os.environ.get("GOLDFIVE_TEST_MODEL")
        or "Qwen3.5-35B-A3B-Q4_K_M.gguf"
    )


def _build_call_llm():
    """Return an async ``(system, user, model) -> str`` backed by openai SDK."""
    from openai import AsyncOpenAI  # local import — only needed when the test runs

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-anything"),
        base_url=os.environ["OPENAI_API_BASE"],
    )

    async def _call(system: str, prompt: str, model: str) -> str:
        resolved = model or _model_name()
        resp = await client.chat.completions.create(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def _close() -> None:
        await client.close()

    _call.close = _close  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# Fixture: borderline-empty drift on a healthy plan.
# ---------------------------------------------------------------------------


def _agents_block() -> list[str]:
    return [
        "research_agent",
        "presentation_agent",
        "reviewer_agent",
        "coordinator_agent",
    ]


def _healthy_plan() -> Plan:
    """A small, valid plan with a clear noun-phrase summary."""
    return Plan(
        id="plan-refine-empty",
        run_id="r-test-refine-empty",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research key information about solar panels",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="create_slide1",
                title="Create slide 1 with solar panel introduction",
                assignee_agent_id="presentation_agent",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="create_slide2",
                title="Create slide 2 with benefits and applications",
                assignee_agent_id="presentation_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research_solar", to_task_id="create_slide1"),
            TaskEdge(from_task_id="create_slide1", to_task_id="create_slide2"),
        ],
        summary="Create a 2-slide presentation about solar panels.",
        revision_index=1,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


_FORBIDDEN_META_PHRASES = (
    "plan unchanged",
    "no changes",
    "no revision",
    "no specific details",
    "drift event",
    "without specific details",
    "no plan change",
    "nothing to change",
    "did not modify",
    "did not change",
)


@pytest.mark.asyncio
async def test_refine_with_borderline_empty_drift_summary_has_no_meta_commentary() -> None:
    """v15 regression: a borderline-empty drift must NOT leak meta-commentary
    into ``plan.summary``.

    Empirical failure mode (pre-fix prompt, Qwen3.5-class model):
        plan.summary = "Plan unchanged as drift event indicates new
                        work discovered without specific details
                        requiring task..."
        ↑ The LLM's introspection about WHY it produced no changes
        leaked into the summary field.

    The strengthened ``summary`` schema description in
    ``_REFINE_SYSTEM_PROMPT`` forbids that pattern; this test is the
    regression pin.
    """
    call_llm = _build_call_llm()
    try:
        planner = LLMPlanner(call_llm=call_llm, model=_model_name())
        prior = _healthy_plan()
        # Borderline-empty drift detail: technically valid input (closes
        # PR #305's empty-string trigger) but semantically vacuous, the
        # exact shape that previously coaxed small models into narrating
        # the absence of changes.
        drift = DriftEvent(
            kind=DriftKind.NEW_WORK_DISCOVERED,
            severity=DriftSeverity.INFO,
            detail=" ",  # nothing of substance to refine on
            current_task_id="create_slide1",
        )
        revised = await planner.refine(
            plan=prior,
            drift=drift,
            goals=[
                Goal(
                    id="g1",
                    summary=(
                        "Create a presentation about solar panels "
                        "containing no more than 2 slides."
                    ),
                )
            ],
            available_agents=_agents_block(),
        )
        # refine returning None is acceptable (no-op signal); the bug is
        # specifically about WHAT goes into summary when refine DOES
        # produce a plan. If None, the test is a no-op for this run.
        if revised is None:
            pytest.skip(
                "Planner returned None for borderline-empty drift "
                "(acceptable no-op); meta-commentary regression cannot "
                "occur on this path."
            )
        summary = (revised.summary or "").lower()
        offenders = [p for p in _FORBIDDEN_META_PHRASES if p in summary]
        assert not offenders, (
            f"plan.summary contains meta-commentary phrases {offenders!r}: "
            f"{revised.summary!r}. The summary must be a noun phrase "
            f"describing the GOAL, not narrate the LLM's introspection."
        )
    finally:
        await call_llm.close()
