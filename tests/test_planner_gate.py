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
