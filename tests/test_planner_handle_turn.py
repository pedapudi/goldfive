"""Unit tests for ``LLMPlanner.handle_turn`` (goldfive#271 Phase 4).

The ``handle_turn`` method is the single per-turn decision point that
collapsed the prior triage stack:

* ``planner_gate`` regex short-circuits (factual-question +
  steer-language detection)
* ``planner_gate.classify_turn`` LLM gate
* ``synthesize_goal_from_steer`` LLM call
* regex-based qualification merge post-process
* ``planner.refine`` LLM call

— all into one LLM call that produces both the routing decision AND
the next plan in one shot. The "classification" is now emergent: the
LLM either produces a plan (warrants change) or returns null (purely
conversational).

These tests cover:

1. The basic shape: ``Plan`` returned for plan-warranted input,
   ``None`` returned for conversational input.
2. The Plan.empty() seed flow: first-turn handle_turn against an
   empty seed produces revision 1 with the user's request.
3. Multi-turn revision: a topic-shift steer produces a revision
   that preserves the prior plan_id and qualifications.
4. The four canonical scenarios from validation v3: factual
   question / topic shift / additive constraint / new-artefact
   request.
5. Failure modes: empty response, malformed JSON, LLM raise — all
   degrade to ``None`` so the Runner falls through gracefully.

The tests use a scripted ``call_llm`` so they run offline. The
prompt-engineering tests against a live LLM (e.g. kikuchi.lan) live
in the integration test suite; this file is the unit-level pin.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

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
# Test helpers
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Records calls and returns canned responses, optionally per-prompt."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self._response


def _empty_session() -> Session:
    """Build a session with an empty Plan seed (mirrors first-turn Runner)."""
    plan = Plan.empty(run_id="r-test")
    s = Session(run_id="r-test")
    s.plan = plan
    return s


def _populated_session(*, plan_id: str = "plan-prior") -> Session:
    """Build a session with a one-task prior plan (mirrors a multi-turn run).

    Includes a Goal carrying a "no more than 2 slides" qualification so
    the qualification-merge tests can verify the prompt threads it
    through.
    """
    s = Session(run_id="r-test")
    s.goals = [
        Goal(
            id="g1",
            summary=(
                "Create a presentation about solar panels with no more "
                "than 2 slides."
            ),
        )
    ]
    s.plan = Plan(
        id=plan_id,
        run_id="r-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research",
                title="Research solar panels",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft the slides",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="research", to_task_id="draft")],
        summary="Research and draft a 2-slide presentation about solar panels.",
        revision_index=1,
    )
    return s


def _plan_json(
    *,
    plan_id: str = "plan-new",
    summary: str = "x",
    tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": plan_id,
        "summary": summary,
        "tasks": tasks
        or [
            {
                "id": "t1",
                "title": "do the thing",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
        "edges": [],
    }


# ---------------------------------------------------------------------------
# 1. Basic shape: Plan returned vs None returned
# ---------------------------------------------------------------------------


async def test_handle_turn_returns_none_when_llm_emits_null_plan() -> None:
    """Conversational verdict: LLM emits ``"plan": null`` → handle_turn
    returns None. Runner reuses session.plan unchanged.
    """
    scripted = _ScriptedLLM(
        json.dumps({"reasoning": "factual question about prior", "plan": None})
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="where will the slides be saved?",
        session=session,
    )
    assert plan is None


async def test_handle_turn_returns_plan_when_llm_emits_one() -> None:
    """Plan-change verdict: LLM emits a plan → handle_turn returns it,
    with the prior plan's id preserved (so _apply_revision can bump
    revision_index cleanly).
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive constraint",
                "plan": _plan_json(plan_id="should-be-overridden"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="make sure the answer fits in 2 slides",
        session=session,
    )
    assert plan is not None
    # Plan id is forced to the prior's id so revision_index bumps.
    assert plan.id == "plan-prior-stable"
    assert plan.tasks  # non-empty


# ---------------------------------------------------------------------------
# 2. Empty-seed (first turn) flow
# ---------------------------------------------------------------------------


async def test_handle_turn_against_empty_seed_produces_first_revision() -> None:
    """First turn: session.plan = Plan.empty() (revision_index=0).
    handle_turn returns a plan; the Runner installs it as revision 1
    via _apply_revision. Here we just verify the planner produces a
    plan (the bumping is the steerer's job).
    """
    scripted = _ScriptedLLM(
        json.dumps({"reasoning": "first plan", "plan": _plan_json()})
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _empty_session()
    plan = await planner.handle_turn(
        user_input="make a presentation about solar panels",
        session=session,
    )
    assert plan is not None
    # Plan id inherits from the empty seed (so the steerer's
    # _apply_revision treats it as a revision of the seed).
    assert plan.id == session.plan.id
    # The user prompt explicitly told the LLM "this is the first turn".
    _sys, user_prompt, _model = scripted.calls[0]
    assert "first turn" in user_prompt.lower(), user_prompt


async def test_handle_turn_empty_user_input_returns_none_without_llm_call() -> None:
    """Empty user_input short-circuits before the LLM is called."""
    scripted = _ScriptedLLM("should not be called")
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(user_input="", session=session)
    assert plan is None
    assert scripted.calls == []


# ---------------------------------------------------------------------------
# 3. Multi-turn revision: prior_plan_id preserved on revisions
# ---------------------------------------------------------------------------


async def test_handle_turn_revision_preserves_prior_plan_id() -> None:
    """Phase 4 invariant (revision branch): when the user input is an
    additive / refining steer (no pivot keywords) AND the LLM does not
    set ``replaces_prior``, the parser forces the returned plan id to
    match the prior's so the steerer's ``_apply_revision`` finds it.

    Pre-R1 this test used a "forget solar panels..." pivot phrasing,
    which now correctly trips the heuristic backstop (goldfive#322
    R1 follow-up). The original revision invariant is unchanged for
    non-pivot inputs — covered here with an additive steer phrasing.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive constraint",
                "plan": _plan_json(
                    plan_id="completely-different-id",
                    summary="solar panel plan with citations",
                ),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-id")
    plan = await planner.handle_turn(
        user_input="add citations to each slide",
        session=session,
    )
    assert plan is not None
    # Even though the LLM emitted "completely-different-id", the parser
    # overrides with the prior's id — Phase 4 invariant for revisions.
    assert plan.id == "plan-prior-id"


# ---------------------------------------------------------------------------
# 4. Prior-goal qualification threading
# ---------------------------------------------------------------------------


async def test_handle_turn_prompt_includes_prior_goals_block() -> None:
    """Prior goals are surfaced in the prompt so the LLM can MERGE
    persistent qualifications (numeric caps, format hints, output
    type) into the next revision. Phase 4: the merge is now the
    LLM's job, not a regex post-process.
    """
    scripted = _ScriptedLLM(
        json.dumps({"reasoning": "x", "plan": _plan_json()})
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    await planner.handle_turn(
        user_input="forget solar — tell me about wind power instead",
        session=session,
    )
    _sys, user_prompt, _model = scripted.calls[0]
    assert "PRIOR GOALS" in user_prompt
    # The 2-slide cap from the prior goal is verbatim in the prompt
    # so the LLM can carry it forward into the revised plan.
    assert "no more than 2 slides" in user_prompt


async def test_handle_turn_prompt_includes_prior_plan_block() -> None:
    """Prior plan (summary + per-task [id / status] title) is rendered
    so the LLM can preserve terminal tasks, reuse stable ids, and add
    delta tasks on a revision.
    """
    scripted = _ScriptedLLM(
        json.dumps({"reasoning": "x", "plan": _plan_json()})
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    await planner.handle_turn(
        user_input="add a comparison slide",
        session=session,
    )
    _sys, user_prompt, _model = scripted.calls[0]
    assert "PRIOR PLAN" in user_prompt
    assert "research" in user_prompt  # task id from prior
    assert "COMPLETED" in user_prompt  # terminal status surfaced
    assert "draft" in user_prompt  # pending task id


# ---------------------------------------------------------------------------
# 5. Failure modes — must never raise; degrade to None
# ---------------------------------------------------------------------------


class _RaisingLLM:
    async def __call__(self, system: str, user: str, model: str) -> str:
        raise RuntimeError("LLM is down")


async def test_handle_turn_call_llm_raise_returns_none() -> None:
    """A misbehaving LLM must never break the run."""
    planner = LLMPlanner(call_llm=_RaisingLLM())
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is None


async def test_handle_turn_malformed_json_returns_none() -> None:
    scripted = _ScriptedLLM("this is not json at all {[")
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is None


async def test_handle_turn_response_not_object_returns_none() -> None:
    scripted = _ScriptedLLM(json.dumps([1, 2, 3]))
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is None


async def test_handle_turn_plan_present_but_not_object_returns_none() -> None:
    """If 'plan' is present but is e.g. a string instead of an object,
    treat as conversational (better than crashing on a bad shape).
    """
    scripted = _ScriptedLLM(
        json.dumps({"reasoning": "x", "plan": "not an object"})
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is None


async def test_handle_turn_empty_response_returns_none() -> None:
    scripted = _ScriptedLLM("")
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is None


# ---------------------------------------------------------------------------
# 6. Markdown fence stripping (matches the rest of the planner's parsing)
# ---------------------------------------------------------------------------


async def test_handle_turn_strips_markdown_fences() -> None:
    """LLM responses wrapped in ```json fences parse correctly."""
    scripted = _ScriptedLLM(
        "```json\n"
        + json.dumps({"reasoning": "x", "plan": _plan_json()})
        + "\n```"
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="something", session=session
    )
    assert plan is not None


# ---------------------------------------------------------------------------
# 7. Canonical scenarios from validation v3 (mocked verdicts)
#
# These pin the EXPECTED behavior shape: the LLM should return null for
# (a) and a Plan for (b)/(c)/(d). Real prompt-engineering coverage
# against a live LLM lives in the integration suite; here we just pin
# the parser/dispatch shape so a regression in the wire schema breaks
# the test.
# ---------------------------------------------------------------------------


async def test_handle_turn_factual_question_about_prior_returns_none() -> None:
    """(a) 'where will the slides be saved?' — conversational."""
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "factual question about prior plan",
                "response_hint": "the slides will be saved as slides.pptx",
                "plan": None,
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="where will the slides be saved?",
        session=session,
    )
    assert plan is None


async def test_handle_turn_topic_pivot_routes_through_pivot_branch() -> None:
    """(b) 'forget solar panels, tell me about solar flares' — explicit
    pivot phrasing. Per goldfive#322 R1 (heuristic backstop), this
    input trips :func:`detect_pivot_intent` regardless of whether the
    LLM sets ``replaces_prior``. The parser mints a fresh plan id and
    stamps the ``_goldfive_pivot`` sentinel so the runner routes
    through ``install_initial_plan``.

    Prior goals (carrying '2 slides') are still threaded into the
    prompt — the LLM is free to merge them into the new plan's summary.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "topic shift; preserve 2-slide cap",
                # Note: replaces_prior intentionally absent — the R1
                # heuristic backstop must rescue the pivot route on
                # its own, mirroring the v20 Qwen failure mode.
                "plan": _plan_json(
                    summary=(
                        "Create a 2-slide presentation about solar flares."
                    ),
                ),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-stable")
    plan = await planner.handle_turn(
        user_input="forget solar panels, tell me about solar flares",
        session=session,
    )
    assert plan is not None
    # R1: pivot route — fresh plan id, NOT inherited from the prior.
    assert plan.id != "plan-stable"
    assert plan.id, "fresh plan id must be non-empty"
    assert getattr(plan, "_goldfive_pivot", False) is True
    # The prompt surfaced the 2-slide cap; assert the canned plan kept it.
    assert "2-slide" in plan.summary or "2 slide" in plan.summary


async def test_handle_turn_additive_constraint_produces_revision() -> None:
    """(c) 'make sure the answer fits in 2 slides' on a prior with no
    such constraint — produces a revision adding the constraint.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive constraint",
                "plan": _plan_json(summary="2-slide presentation about X"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    plan = await planner.handle_turn(
        user_input="make sure the answer fits in 2 slides",
        session=session,
    )
    assert plan is not None


async def test_handle_turn_new_artefact_request_produces_revision() -> None:
    """(d) 'make a 5-page report on dark matter' on a prior solar-panel
    plan — produces a revision with the new artefact + topic. Phase 4
    collapses 'replace' into 'revision': the plan_id is preserved.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "new artefact + topic",
                "plan": _plan_json(
                    summary="5-page report on dark matter",
                    tasks=[
                        {
                            "id": "research",
                            "title": "Research dark matter",
                            "assignee_agent_id": "writer",
                            "status": "COMPLETED",
                        },
                        {
                            "id": "write",
                            "title": "Write 5-page report",
                            "assignee_agent_id": "writer",
                            "status": "PENDING",
                        },
                    ],
                ),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-stable")
    plan = await planner.handle_turn(
        user_input="make a 5-page report on dark matter",
        session=session,
    )
    assert plan is not None
    # Plan id stays the same — Phase 4: there is no fresh path; even a
    # genuine pivot is a revision of the conversation's plan.
    assert plan.id == "plan-stable"


# ---------------------------------------------------------------------------
# 8. The system prompt contains the Phase 4 design language so a
#    regression that strips it is caught here.
# ---------------------------------------------------------------------------


async def test_handle_turn_system_prompt_documents_null_for_conversational() -> None:
    """The prompt MUST tell the LLM to return null for purely
    conversational input. A regression that strips this guidance would
    cause the planner to over-eagerly produce plans for clarifying
    questions.
    """
    scripted = _ScriptedLLM(json.dumps({"reasoning": "x", "plan": None}))
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    await planner.handle_turn(user_input="x", session=session)
    sys_prompt, _user, _model = scripted.calls[0]
    assert "null" in sys_prompt.lower()
    assert "conversational" in sys_prompt.lower()


async def test_handle_turn_system_prompt_documents_qualification_merge() -> None:
    """The prompt MUST tell the LLM to MERGE persistent qualifications
    from prior_goals into a revised plan unless explicitly removed.
    This is the Phase 4 replacement for the deleted regex post-process.
    """
    scripted = _ScriptedLLM(json.dumps({"reasoning": "x", "plan": None}))
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session()
    await planner.handle_turn(user_input="x", session=session)
    sys_prompt, _user, _model = scripted.calls[0]
    assert "MERGE persistent qualifications" in sys_prompt or (
        "persistent qualifications" in sys_prompt
    )
