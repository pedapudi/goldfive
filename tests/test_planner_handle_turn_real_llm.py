"""Real-LLM prompt-quality tests for ``LLMPlanner.handle_turn``.

The mock-LLM tests in ``test_planner_handle_turn.py`` pin the
``handle_turn`` plumbing (parsing, prompt assembly, plan-id reuse) but
cannot detect prompt regressions: a strict scripted callable returns
the same canned JSON regardless of how the prompt is worded.

This file exercises the real ``_HANDLE_TURN_SYSTEM_PROMPT`` against a
live OpenAI-compatible endpoint (kikuchi-hosted Qwen, vLLM, llama.cpp,
LM Studio, etc.). The two scenarios encode the failures observed in
validation v5 Class 1 against PR #291's first prompt iteration:

* **Turn-2 qualification dropping.** Prior goal carries a "no more
  than 2 slides" cap; the user pivots subject ("forget solar panels,
  tell me about solar flares instead"). The revised plan summary MUST
  preserve the "2 slides" / "2-slide" qualification — the validation
  bug was that PR #291's prompt let the LLM emit "Research solar
  flares and create presentation on solar flares" with the cap silently
  dropped.

* **Turn-4 valid-revision-shape.** A partially-executed plan (one
  COMPLETED task plus PENDING work) receives an additive constraint
  ("make sure the answer fits in just 2 slides"). The revised plan
  MUST satisfy ``Plan.validate(for_revision=True, prior=existing)`` —
  the validator complaint observed in v5 was that the LLM dropped the
  COMPLETED task from the revision, violating PLAN-LIFECYCLE.md §3.1
  terminal-task preservation.

Gating: skipped unless ``OPENAI_API_BASE`` is set in the environment.
The CI environment doesn't set this, so the tests are no-ops there;
run them locally against kikuchi via:

    OPENAI_API_BASE=http://kikuchi.lan:8080/v1 \
    OPENAI_API_KEY=sk-anything \
    USER_MODEL_NAME=Qwen3.5-35B-A3B-Q4_K_M.gguf \
    uv run pytest tests/test_planner_handle_turn_real_llm.py -v

Both tests do a single LLM round-trip; expected wall time is ~10-60 s
each on a Qwen-class local model.
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
    Goal,
    LLMPlanner,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Real-LLM call_llm callable.
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
    """Return an async ``(system, user, model) -> str`` backed by openai SDK.

    Uses the standard ``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` envvars
    so it works against kikuchi, vLLM, llama.cpp, LM Studio, and the
    public OpenAI endpoint without code changes.
    """
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
            # Local Qwen-class models are slow JSON formatters; give them
            # enough budget to render two complete plans-with-tasks plus
            # any reasoning traces they emit.
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def _close() -> None:
        await client.close()

    _call.close = _close  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# Session fixtures matching v5 Class 1 turn 2 and turn 4.
# ---------------------------------------------------------------------------


def _agents_block() -> list[str]:
    """Minimal agent registry the planner can pick from in revisions."""
    return [
        "research_agent",
        "presentation_agent",
        "reviewer_agent",
        "coordinator_agent",
    ]


def _v5_turn2_session() -> Session:
    """Reproduce the turn-2 prior state: one revision-1 plan, "2 slides" goal.

    Mirrors the empirical state observed at session ``v5class1-1``
    revision_index=2 in ``/home/sunil/git/harmonograf/data/harmonograf.db``
    just before the user pivoted from solar panels to solar flares.
    """
    s = Session(run_id="r-test-turn2")
    s.goals = [
        Goal(
            id="g1",
            summary=(
                "Create a presentation about solar panels containing no "
                "more than 2 slides."
            ),
        )
    ]
    s.plan = Plan(
        id="plan-v5",
        run_id="r-test-turn2",
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
            Task(
                id="review_presentation",
                title="Review presentation for accuracy and completeness",
                assignee_agent_id="reviewer_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research_solar", to_task_id="create_slide1"),
            TaskEdge(from_task_id="create_slide1", to_task_id="create_slide2"),
            TaskEdge(from_task_id="create_slide2", to_task_id="review_presentation"),
        ],
        summary="Create a 2-slide presentation about solar panels",
        revision_index=1,
    )
    return s


def _v5_turn4_session() -> Session:
    """Reproduce the turn-4 prior state: revision-3 plan after the topic pivot.

    Matches the post-turn-2 plan recorded at session ``v5class1-1``
    sequence=6 (revision_index=3, plan_id=f84b5f2ccd804fda...): one
    COMPLETED solar-panels research task plus the new solar-flares
    sub-DAG. The user now adds the "fits in just 2 slides" constraint.
    """
    s = Session(run_id="r-test-turn4")
    s.goals = [
        Goal(
            id="g1",
            summary=(
                "Create a presentation about solar panels containing no "
                "more than 2 slides."
            ),
        ),
        Goal(
            id="g2",
            summary="Tell me about solar flares instead of solar panels.",
        ),
    ]
    s.plan = Plan(
        id="plan-v5",
        run_id="r-test-turn4",
        goal_ids=["g1", "g2"],
        tasks=[
            Task(
                id="research_solar",
                title="Research key information about solar panels",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="research_solar_flares",
                title="Research key information about solar flares",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="create_presentation_flares",
                title="Create solar flares presentation",
                assignee_agent_id="presentation_agent",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="review_presentation",
                title="Review presentation for accuracy and completeness",
                assignee_agent_id="reviewer_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(
                from_task_id="research_solar_flares",
                to_task_id="create_presentation_flares",
            ),
            TaskEdge(
                from_task_id="create_presentation_flares",
                to_task_id="review_presentation",
            ),
        ],
        summary=(
            "Research solar flares and create presentation on solar "
            "flares instead of solar panels."
        ),
        revision_index=3,
    )
    return s


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v5class1_turn2_topic_pivot_preserves_2_slide_qualification() -> None:
    """v5 Class 1 turn 2: the LLM must carry forward the "2 slides" cap.

    Empirical failure mode (PR #291 prompt, Qwen3.5-35B):
        plan.summary = "Research solar flares and create presentation
                        on solar flares instead of solar panels"
        ↑ The "2 slides" qualifier from the prior goal is silently
        dropped during the topic pivot.

    The strengthened prompt's ``QUALIFICATION-PRESERVATION EXAMPLE``
    plus the ``MERGE persistent qualifications ... AND into the
    summary string`` directive is the fix; this test is the regression
    pin.
    """
    call_llm = _build_call_llm()
    try:
        planner = LLMPlanner(call_llm=call_llm, model=_model_name())
        plan = await planner.handle_turn(
            user_input="Forget solar panels, tell me about solar flares instead.",
            session=_v5_turn2_session(),
            available_agents=_agents_block(),
        )
        assert plan is not None, (
            "Topic-pivot steer should produce a plan revision, not None."
        )
        summary = (plan.summary or "").lower()
        # Look for either "2 slides" or "2-slide" (the prompt's example
        # uses both). Either form satisfies "merge the numeric cap".
        assert "2 slide" in summary or "2-slide" in summary, (
            f"plan.summary dropped the '2 slides' qualification: {plan.summary!r}. "
            f"All tasks: {[(t.id, t.title) for t in plan.tasks]}"
        )
    finally:
        await call_llm.close()


@pytest.mark.asyncio
async def test_v5class1_turn4_additive_constraint_produces_valid_revision() -> None:
    """v5 Class 1 turn 4: revision must pass ``Plan.validate(for_revision=True)``.

    Empirical failure mode (PR #291 prompt, Qwen3.5-35B):
        Runner.run logged "produced_plan=yes" but the steerer's
        ``apply_user_steer_with_plan`` rejected the install with
        ``plan revision rejected by validator`` — the LLM dropped the
        COMPLETED ``research_solar`` task from the revision, breaking
        PLAN-LIFECYCLE.md §3.1 (terminal-task preservation).

    The strengthened prompt's WORKED EXAMPLE A explicitly demonstrates
    echoing back COMPLETED tasks with status verbatim; this test
    enforces the contract end-to-end against the real LLM.
    """
    call_llm = _build_call_llm()
    try:
        planner = LLMPlanner(call_llm=call_llm, model=_model_name())
        prior = _v5_turn4_session()
        prior_plan = prior.plan
        plan = await planner.handle_turn(
            user_input="Make sure the answer fits in just 2 slides.",
            session=prior,
            available_agents=_agents_block(),
        )
        assert plan is not None, (
            "Additive constraint should produce a plan revision, not None."
        )
        # The whole point: the steerer will call exactly this; if it
        # raises, the install would fail with the same SCHEMA_VIOLATION
        # the v5 validation hit.
        try:
            plan.validate(for_revision=True, prior=prior_plan)
        except ValueError as exc:
            new_task_ids = sorted(t.id for t in plan.tasks)
            terminal_prior = sorted(
                t.id for t in prior_plan.tasks if t.status is TaskStatus.COMPLETED
            )
            raise AssertionError(
                f"Plan.validate(for_revision=True) rejected the LLM's "
                f"revision: {exc}. "
                f"Terminal prior task ids: {terminal_prior!r}. "
                f"Revision task ids: {new_task_ids!r}."
            ) from exc
    finally:
        await call_llm.close()
