"""Unit tests for :mod:`goldfive.planner_gate`.

Covers:

* The deterministic ``heuristic_classify_turn`` fallback: the LLM-free
  path the Runner uses when ``planner.call_llm`` is missing.
* The LLM-backed ``classify_turn``: verdict parsing, JSON / bare-token
  response shapes, fallback to heuristic on parse failure, and the
  short-circuit when no prior plan is supplied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from goldfive.planner_gate import (
    classify_turn,
    heuristic_classify_turn,
)
from goldfive.types import Plan, Task, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prior_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Gather facts", status=TaskStatus.COMPLETED),
            Task(id="t2", title="Draft post", status=TaskStatus.COMPLETED),
            Task(id="t3", title="Review", status=TaskStatus.COMPLETED),
        ],
        edges=[],
        summary="Draft + review a short post.",
    )


def _empty_plan() -> Plan:
    return Plan(id="empty", run_id="", goal_ids=[], tasks=[], edges=[])


def _call_llm_returning(*chunks: str) -> Callable[..., Any]:
    """Build an async stub that yields canned strings in order."""
    queue = list(chunks)

    async def _llm(system: str, user: str, model: str) -> str:
        _ = system, user, model
        if not queue:
            raise AssertionError("stub exhausted")
        return queue.pop(0)

    return _llm


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


def test_heuristic_no_prior_plan_returns_new_work() -> None:
    verdict = heuristic_classify_turn(
        prior_plan=None,
        completed_results={},
        user_input="make a 2-slide deck about solar panels",
        conversation_id="c1",
    )
    assert verdict == "new_work"


def test_heuristic_empty_prior_plan_returns_new_work() -> None:
    verdict = heuristic_classify_turn(
        prior_plan=_empty_plan(),
        completed_results={},
        user_input="short follow-up",
        conversation_id="c1",
    )
    assert verdict == "new_work"


def test_heuristic_short_follow_up_after_prior_plan_is_conversational() -> None:
    verdict = heuristic_classify_turn(
        prior_plan=_prior_plan(),
        completed_results={"t1": "facts"},
        user_input="where is the presentation located?",
        conversation_id="c1",
    )
    assert verdict == "conversational"


def test_heuristic_long_follow_up_after_prior_plan_is_new_work() -> None:
    # > 20 tokens — the heuristic's conservative fallback.
    long_ask = " ".join(["and"] * 25) + " please make a full new workflow"
    verdict = heuristic_classify_turn(
        prior_plan=_prior_plan(),
        completed_results={"t1": "facts"},
        user_input=long_ask,
        conversation_id="c1",
    )
    assert verdict == "new_work"


def test_heuristic_non_steer_inputs_avoid_refine_existing() -> None:
    # Non-steer inputs (asks that don't open with a steer-pattern
    # directive) still resolve to conversational / new_work — the
    # heuristic only escalates to refine_existing when it sees a
    # steer-language opener (see test_heuristic_steer_language_*).
    for ui in [
        "make it funnier",
        "also add a slide about X",
        "what was the title again",
    ]:
        assert heuristic_classify_turn(
            prior_plan=_prior_plan(),
            completed_results={},
            user_input=ui,
        ) in ("conversational", "new_work")


def test_heuristic_steer_language_returns_refine_existing() -> None:
    # goldfive#270 follow-up: messages that open with steer-language
    # ("forget X", "instead", "no, don't ..., do ...", "switch to",
    # "scratch that", "actually, ...") route through refine_existing
    # so the runner emits PlanRevised + DriftDetected(USER_STEER) and
    # preserves the prior plan's structural constraints. Pre-fix the
    # heuristic returned "conversational" (short input) or "new_work"
    # (long input), both of which silently dropped sticky context.
    steer_messages = [
        "forget solar panels. tell me about solar flares instead.",
        "no, don't do solar panels — switch to solar flares.",
        "actually, change the topic to solar flares.",
        "scratch that. solar flares please.",
        "instead, do a presentation about solar flares.",
        "wait, change the plan — solar flares.",
        "stop. solar flares only.",
    ]
    for ui in steer_messages:
        verdict = heuristic_classify_turn(
            prior_plan=_prior_plan(),
            completed_results={},
            user_input=ui,
        )
        assert verdict == "refine_existing", (
            f"steer-language opener {ui!r} should route refine_existing, "
            f"got {verdict!r}"
        )


def test_heuristic_steer_language_requires_prior_plan() -> None:
    # First turn (no prior plan) still returns new_work even on a
    # steer-shaped opener — there's nothing to refine yet.
    verdict = heuristic_classify_turn(
        prior_plan=None,
        completed_results={},
        user_input="forget that. tell me about solar flares.",
    )
    assert verdict == "new_work"


def test_heuristic_inline_steer_word_does_not_match() -> None:
    # The steer regex is anchored to the start of the message OR a
    # sentence break, so "forget" appearing mid-clause as part of a
    # question doesn't mis-route. "I'll never forget the time you ..."
    # is a conversational reminisce, not a steer.
    verdict = heuristic_classify_turn(
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="I'll never forget the time you nailed it",
    )
    # Short input → conversational (NOT refine_existing).
    assert verdict == "conversational"


# ---------------------------------------------------------------------------
# LLM-backed classifier
# ---------------------------------------------------------------------------


async def test_classify_turn_without_call_llm_delegates_to_heuristic() -> None:
    verdict = await classify_turn(
        call_llm=None,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="where did you save it",
        conversation_id="c1",
    )
    assert verdict == "conversational"


async def test_classify_turn_without_prior_plan_skips_llm() -> None:
    calls: list[tuple[str, str, str]] = []

    async def _llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        return json.dumps({"verdict": "conversational", "reason": "x"})

    verdict = await classify_turn(
        call_llm=_llm,
        prior_plan=None,
        completed_results={},
        user_input="any",
        conversation_id="c1",
    )
    assert verdict == "new_work"
    # No LLM call — the gate shortcut fires when there's no prior plan.
    assert calls == []


async def test_classify_turn_parses_json_verdict_conversational() -> None:
    llm = _call_llm_returning(
        json.dumps({"verdict": "conversational", "reason": "asking about artefact location"})
    )
    verdict = await classify_turn(
        call_llm=llm,
        prior_plan=_prior_plan(),
        completed_results={"t1": "done"},
        user_input="where is the presentation located?",
        conversation_id="c1",
    )
    assert verdict == "conversational"


async def test_classify_turn_parses_json_verdict_refine_existing() -> None:
    llm = _call_llm_returning(
        json.dumps({"verdict": "refine_existing", "reason": "extends prior plan"})
    )
    verdict = await classify_turn(
        call_llm=llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="also add a slide about cost of panels",
        conversation_id="c1",
    )
    assert verdict == "refine_existing"


async def test_classify_turn_parses_json_verdict_new_work() -> None:
    llm = _call_llm_returning(
        json.dumps({"verdict": "new_work", "reason": "wholly new topic"})
    )
    verdict = await classify_turn(
        call_llm=llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="now write me a haiku about birds",
        conversation_id="c1",
    )
    assert verdict == "new_work"


async def test_classify_turn_accepts_bare_token_response() -> None:
    llm = _call_llm_returning("conversational")
    verdict = await classify_turn(
        call_llm=llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="where is it?",
        conversation_id="c1",
    )
    assert verdict == "conversational"


async def test_classify_turn_malformed_response_falls_back_to_heuristic() -> None:
    llm = _call_llm_returning("totally not valid json or verdict")
    # Short input after prior plan → heuristic gives "conversational".
    verdict = await classify_turn(
        call_llm=llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="where is it?",
        conversation_id="c1",
    )
    assert verdict == "conversational"


async def test_classify_turn_llm_raising_falls_back_to_heuristic() -> None:
    async def _boom(system: str, user: str, model: str) -> str:
        raise RuntimeError("provider down")

    # Long input without a prior-plan-aware verdict → heuristic's new_work.
    long = " ".join(["and"] * 25) + " please do something entirely new"
    verdict = await classify_turn(
        call_llm=_boom,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input=long,
        conversation_id="c1",
    )
    assert verdict == "new_work"


# ---------------------------------------------------------------------------
# Phase 2.X / goldfive#271 Gap 4: factual-question short-circuit
# ---------------------------------------------------------------------------


def test_heuristic_factual_question_routes_conversational() -> None:
    """Phase 2.X (goldfive#271 Gap 4): factual interrogatives that
    open with where/when/how/what/why/which/who + auxiliary verb route
    through ``conversational`` deterministically, not via the
    token-count fallback alone.

    The validation E2E saw "where will the slides be saved?" mis-routed
    to ``refine_existing`` by the LLM gate. The heuristic catches it
    before the LLM runs.
    """
    factual_questions = [
        "where will the slides be saved?",  # the validation regression
        "where is the file located?",
        "where are the outputs?",
        "when did you finish the deck?",
        "when will it be done?",
        "how does the slideshow work?",
        "how do I open the file?",
        "how many slides does it have?",
        "what is the title of the deck?",
        "what's the second slide about?",
        "what was the source you used?",
        "why did you pick that title?",
        "which template did you use?",
        "who wrote the second slide?",
        "did you include the cost slide?",
        "is the presentation done?",
        "are the slides ready?",
        "can you tell me where it lives?",
        "could you explain the structure?",
    ]
    for ui in factual_questions:
        verdict = heuristic_classify_turn(
            prior_plan=_prior_plan(),
            completed_results={},
            user_input=ui,
        )
        assert verdict == "conversational", (
            f"factual question {ui!r} should route conversational; "
            f"got {verdict!r}"
        )


def test_heuristic_factual_question_does_not_match_steer_phrasing() -> None:
    """Phrases with ``where``/``when``/``how`` that ARE steers must NOT
    match the factual-question heuristic — the steer regex runs first
    and wins.
    """
    # "switch to" is a steer pattern; "where would..." is conditional.
    # The first sentence is a steer; the heuristic returns refine_existing.
    verdict = heuristic_classify_turn(
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="switch to a different topic. where do we start?",
    )
    assert verdict == "refine_existing", (
        f"steer-then-question should still route refine_existing; got {verdict!r}"
    )


def test_heuristic_factual_question_requires_prior_plan() -> None:
    """First turn (no prior plan) still returns ``new_work`` even on a
    factual-question opener — there's nothing to ask about yet.
    """
    verdict = heuristic_classify_turn(
        prior_plan=None,
        completed_results={},
        user_input="where is it?",
    )
    assert verdict == "new_work"


async def test_classify_turn_factual_question_short_circuits_llm() -> None:
    """The LLM gate is bypassed when the user message matches the
    factual-question heuristic. Pre-Phase-2.X the LLM was always
    consulted; the validation E2E showed it returning ``refine_existing``
    for "where will the slides be saved?". Short-circuiting the LLM
    eliminates the regression path.
    """
    llm_calls: list[tuple[str, str, str]] = []

    async def _llm(system: str, user: str, model: str) -> str:
        llm_calls.append((system, user, model))
        # If the LLM IS called, it would return refine_existing — the
        # validation regression. The heuristic must intercept BEFORE
        # the LLM gets the chance.
        return json.dumps({"verdict": "refine_existing", "reason": "wrong"})

    verdict = await classify_turn(
        call_llm=_llm,
        prior_plan=_prior_plan(),
        completed_results={"t1": "facts"},
        user_input="where will the slides be saved?",
        conversation_id="c1",
    )
    assert verdict == "conversational", (
        f"factual question should route conversational without LLM; "
        f"got {verdict!r}"
    )
    assert llm_calls == [], (
        "LLM was called despite factual-question heuristic short-circuit; "
        "the regression path is open"
    )


async def test_classify_turn_steer_language_short_circuits_llm() -> None:
    """The LLM gate is also bypassed for steer-language openers — same
    rationale as the factual-question short-circuit. The LLM was prone
    to misclassifying topic pivots as ``new_work`` (silently dropping
    sticky constraints), so explicit steer language routes to
    ``refine_existing`` deterministically.
    """
    llm_calls: list[tuple[str, str, str]] = []

    async def _llm(system: str, user: str, model: str) -> str:
        llm_calls.append((system, user, model))
        return json.dumps({"verdict": "new_work", "reason": "wrong"})

    verdict = await classify_turn(
        call_llm=_llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input="forget solar panels. tell me about wind power instead.",
        conversation_id="c1",
    )
    assert verdict == "refine_existing"
    assert llm_calls == []


async def test_classify_turn_falls_through_to_llm_for_ambiguous_input() -> None:
    """Inputs that match neither the steer nor factual-question
    heuristic still consult the LLM. This is the "no signal" bucket
    where the LLM's nuance pays off — refining a request, asking for
    extension, or making genuinely new work.
    """
    llm_calls: list[tuple[str, str, str]] = []

    async def _llm(system: str, user: str, model: str) -> str:
        llm_calls.append((system, user, model))
        return json.dumps({"verdict": "refine_existing", "reason": "extends prior"})

    ambiguous = (
        "please make the deck more colourful and add an extra section "
        "about cost benefits over a five year horizon"
    )
    verdict = await classify_turn(
        call_llm=_llm,
        prior_plan=_prior_plan(),
        completed_results={},
        user_input=ambiguous,
        conversation_id="c1",
    )
    assert verdict == "refine_existing"
    assert len(llm_calls) == 1, (
        f"expected exactly one LLM call for ambiguous input; got {len(llm_calls)}"
    )
