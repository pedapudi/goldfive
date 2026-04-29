"""Heuristic backstop for pivot detection (R1, goldfive#322 follow-up).

The Tier 2 pipeline (#323) added an LLM-side ``replaces_prior: bool``
field on ``LLMPlanner.handle_turn`` to route pivots ("forget X, do Y
instead") through :meth:`DefaultSteerer.install_initial_plan` instead
of ``install_revision_for_drift``. v20 validation found that small
thinking models (Qwen 35B Q4) do not reliably set the flag even on
textbook pivot phrasing — session
``61ddf449-8ea3-470e-b175-c38211b81220`` turn 2 was *"Forget solar
panels, tell me about solar flares instead."* and the LLM left the
flag unset. The runner re-used the prior plan id and pivot routing
was bypassed.

This module covers the deterministic backstop:

* :func:`detect_pivot_intent` — pure regex test against representative
  positive / negative phrasings (R1 unit coverage).
* :func:`_parse_handle_turn_response` — heuristic alone (LLM flag
  absent), LLM flag alone (no keywords), both signals together,
  neither signal (additive steer).
* End-to-end runner spy — heuristic-only pivot routes through
  ``install_initial_plan`` and mints a fresh plan id, mirroring the
  v20 failure mode.
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
    CallableAdapter,
    Goal,
    InMemorySink,
    InvocationResult,
    LLMPlanner,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)
from goldfive.planner import detect_pivot_intent  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_tier2_steer_pipeline.py)
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Returns canned responses, optionally per-call from a script."""

    def __init__(self, responses: str | list[str]) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        i = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[i]


def _populated_session(*, plan_id: str = "plan-prior") -> Session:
    s = Session(run_id="r-test")
    s.goals = [Goal(id="g1", summary="Make a 2-slide presentation about solar panels.")]
    s.plan = Plan(
        id=plan_id,
        run_id="r-test",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_panels",
                title="Research solar panels",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft_slides",
                title="Draft the slides",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_panels", to_task_id="draft_slides")],
        summary="Make a 2-slide presentation about solar panels.",
        revision_index=1,
    )
    return s


def _plan_dict(
    *,
    plan_id: str = "plan-new",
    summary: str = "x",
    tasks: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
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
        "edges": edges or [],
    }


async def _happy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


# ---------------------------------------------------------------------------
# Unit: regex coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        # The v20 case verbatim.
        "Forget solar panels, tell me about solar flares instead.",
        # Variants on the same opener.
        "forget the slides — write me a haiku about cats instead",
        "Forget about that, do this other thing.",
        # "instead of X" — replacement framing.
        "Instead of researching panels, draft a marketing brief.",
        # "switch to" / "switching to".
        "Switch to a different topic: dolphins.",
        "switching to a haiku format please",
        # "scratch that".
        "Scratch that — let's do something simpler.",
        # "actually let's / actually do X" — common reversal opener.
        "Actually, let's write about birds instead.",
        "actually do the marketing brief first",
        # "no, do X" / "no tell me" / "no let's".
        "No, tell me about black holes.",
        "no, do the haiku version",
        # "wait, ..." reversal.
        "Wait, do the slides about clouds instead.",
        "Wait, I want a different topic.",
        # "change topic" / "new topic".
        "Change the topic to weather.",
        "new topic: oceans",
        # "replace that".
        "Replace that with something shorter.",
        # "do X instead".
        "Do something instead — write a poem.",
    ],
)
def test_detect_pivot_intent_positive(phrase: str) -> None:
    assert detect_pivot_intent(phrase) is True, phrase


@pytest.mark.parametrize(
    "phrase",
    [
        # Additive / refining steers — must NOT trigger pivot routing.
        "Make sure the slides include a chart.",
        "Research the topic more thoroughly.",
        "Add a citation to slide 2.",
        "Please be concise.",
        "Where did the data come from?",
        "Use a different font on slide 1.",
        # Empty / whitespace / non-string.
        "",
        "   ",
    ],
)
def test_detect_pivot_intent_negative(phrase: str) -> None:
    assert detect_pivot_intent(phrase) is False, phrase


def test_detect_pivot_intent_handles_non_string() -> None:
    assert detect_pivot_intent(None) is False
    assert detect_pivot_intent(123) is False  # type: ignore[arg-type]
    assert detect_pivot_intent([]) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parser: heuristic OR LLM flag
# ---------------------------------------------------------------------------


async def test_heuristic_only_triggers_pivot_routing_in_parser() -> None:
    """Reproduces v20: LLM omits ``replaces_prior`` but the user said
    "Forget solar panels, tell me about solar flares instead." The
    keyword heuristic must trigger pivot routing — fresh plan id +
    ``_goldfive_pivot`` sentinel — without help from the LLM flag.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "the model failed to set replaces_prior",
                # NOTE: replaces_prior is intentionally absent.
                "plan": _plan_dict(plan_id="ignored-by-runner"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="Forget solar panels, tell me about solar flares instead.",
        session=session,
    )
    assert plan is not None
    assert plan.id != "plan-prior-stable", (
        "heuristic should have minted a fresh plan id"
    )
    assert plan.id, "fresh plan id must be non-empty"
    assert getattr(plan, "_goldfive_pivot", False) is True


async def test_llm_flag_only_triggers_pivot_routing() -> None:
    """LLM sets ``replaces_prior=True`` while user_input has no pivot
    keywords. Pivot routing still triggers (the LLM-only path remains
    intact — the heuristic is purely additive).
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "structured flag set",
                "replaces_prior": True,
                "plan": _plan_dict(plan_id="ignored-by-runner"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    # No pivot keywords in this input — heuristic returns False.
    user_input = "make it a marketing brief about renewables"
    assert detect_pivot_intent(user_input) is False
    plan = await planner.handle_turn(user_input=user_input, session=session)
    assert plan is not None
    assert plan.id != "plan-prior-stable"
    assert getattr(plan, "_goldfive_pivot", False) is True


async def test_both_signals_triggers_pivot_routing_no_double() -> None:
    """Both LLM flag AND keywords present. Pivot routing triggers
    cleanly — no double-routing or weird state.
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "both signals",
                "replaces_prior": True,
                "plan": _plan_dict(plan_id="ignored-by-runner"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="Forget the slides, do a haiku instead.",
        session=session,
    )
    assert plan is not None
    assert plan.id != "plan-prior-stable"
    # Idempotent: the sentinel is True (not "True True" or any other
    # weirdness), and the parser ran exactly once.
    assert getattr(plan, "_goldfive_pivot", False) is True
    assert len(scripted.calls) == 1


async def test_neither_signal_preserves_revision_route() -> None:
    """Additive steer: no LLM flag AND no pivot keywords. The prior
    plan id must be preserved (revision route — Rule 6 binding still
    applies via the runner's drift install path).
    """
    scripted = _ScriptedLLM(
        json.dumps(
            {
                "reasoning": "additive constraint, no replacement",
                "replaces_prior": False,
                "plan": _plan_dict(plan_id="ignored-by-runner"),
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    session = _populated_session(plan_id="plan-prior-stable")
    plan = await planner.handle_turn(
        user_input="make sure it includes a chart",
        session=session,
    )
    assert plan is not None
    assert plan.id == "plan-prior-stable", (
        "additive steer must preserve the prior plan id (revision route)"
    )
    assert getattr(plan, "_goldfive_pivot", False) is False


# ---------------------------------------------------------------------------
# End-to-end: heuristic-only pivot routes through install_initial_plan
# ---------------------------------------------------------------------------


async def test_heuristic_only_pivot_e2e_routes_through_install_initial_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v20 scenario, end-to-end. Drive turn 1 (initial plan) then
    turn 2 with a textbook pivot input AND an LLM stub that does NOT
    set ``replaces_prior``. Assert turn 2 still routes through
    ``install_initial_plan`` (heuristic backstop) and mints a fresh
    plan id.
    """
    plan_t1 = _plan_dict(
        plan_id="ignored-by-runner",
        summary="2-slide presentation about solar panels",
        tasks=[
            {
                "id": "research_panels",
                "title": "Research solar panels",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
            {
                "id": "draft_slides",
                "title": "Draft 2 slides about solar panels",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            },
        ],
        edges=[
            {"from_task_id": "research_panels", "to_task_id": "draft_slides"},
        ],
    )
    plan_t2_pivot = _plan_dict(
        plan_id="ignored-by-runner",
        summary="solar flares explainer",
        tasks=[
            {
                "id": "explain_flares",
                "title": "Explain solar flares",
                "assignee_agent_id": "writer",
                "status": "PENDING",
            }
        ],
        edges=[],
    )

    async def planner_llm(system: str, user: str, model: str) -> str:
        _ = model
        if "warrants a plan change" in system or "PIVOT vs REVISION" in system:
            # handle_turn call. Branch on the user message.
            #
            # The pivot turn keys off "flares" — and we INTENTIONALLY
            # omit ``replaces_prior`` so only the heuristic can rescue
            # this turn into the pivot route. This mirrors the v20
            # session.
            if "flares" in user.lower() or "forget" in user.lower():
                return json.dumps(
                    {
                        "reasoning": "topic shift — solar flares",
                        # replaces_prior intentionally absent.
                        "plan": plan_t2_pivot,
                    }
                )
            return json.dumps(
                {
                    "reasoning": "first turn",
                    "replaces_prior": False,
                    "plan": plan_t1,
                }
            )
        return json.dumps(plan_t1)

    planner = LLMPlanner(call_llm=planner_llm, model="stub")
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
    )

    initial_calls: list[tuple[str, bool]] = []
    drift_calls: list[str] = []
    real_initial = runner.steerer.install_initial_plan
    real_drift = runner.steerer.install_revision_for_drift

    async def _spy_initial(
        *, session: Session, plan: Plan, is_pivot: bool = False
    ) -> bool:
        initial_calls.append((plan.id, is_pivot))
        return await real_initial(session=session, plan=plan, is_pivot=is_pivot)

    async def _spy_drift(*, session: Session, drift: Any, revised_plan: Plan) -> bool:
        drift_calls.append(revised_plan.id)
        return await real_drift(
            session=session, drift=drift, revised_plan=revised_plan
        )

    monkeypatch.setattr(runner.steerer, "install_initial_plan", _spy_initial)
    monkeypatch.setattr(runner.steerer, "install_revision_for_drift", _spy_drift)

    out1 = await runner.run("make a 2-slide presentation about solar panels")
    assert out1.success, out1.reason
    turn1_plan_id = out1.session.plan.id

    # The v20 input verbatim.
    out2 = await runner.run(
        "Forget solar panels, tell me about solar flares instead."
    )
    await runner.close()
    assert out2.success, out2.reason
    turn2_plan_id = out2.session.plan.id

    # Heuristic-rescued pivot: 2 install_initial_plan calls (turn 1 +
    # rescued turn 2), zero drift installs.
    assert len(initial_calls) == 2, (
        f"expected 2 install_initial_plan calls (turn 1 + heuristic-rescued "
        f"turn 2); got {len(initial_calls)}"
    )
    assert initial_calls[0][1] is False, (
        f"turn 1 should be first-turn (is_pivot=False); got {initial_calls[0]}"
    )
    assert initial_calls[1][1] is True, (
        f"turn 2 should be flagged as pivot via heuristic backstop "
        f"(is_pivot=True); got {initial_calls[1]}"
    )
    assert len(drift_calls) == 0, (
        f"heuristic-rescued pivot must NOT route through "
        f"install_revision_for_drift; got {len(drift_calls)} drift install(s)"
    )
    assert turn2_plan_id != turn1_plan_id, (
        f"heuristic pivot should mint a fresh plan id; both turns got {turn1_plan_id}"
    )
