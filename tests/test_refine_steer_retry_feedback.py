"""Refine retry loop short-circuits on repeated same-class rejection (iter-11C).

When ``LLMPlanner._refine_steer`` runs its retry loop, the
:meth:`_build_correction_prompt` helper feeds attempt N's validator
rejection into attempt N+1's prompt — a real second attempt, not a
cold reprompt. That works for parse / goal-coverage failures (where
the LLM can read the feedback and fix the shape), but for the
mechanical structural-invariant rejections from
``Plan.validate(for_revision=True, prior=...)`` —

* ``"terminal task {id!r} missing in revision"``
* ``"terminal task {id!r} regressed to {status!r}"``
* ``"terminal->terminal edge {a!r} -> {b!r} missing in revision"``

— small models (Qwen 35B in particular) tend to re-emit the same
class of violation on the retry, even with the explicit feedback in
the prompt. A live e2e showed two refine attempts both failing with
``terminal task 'draft_slides'`` rejections, ~10s burned each before
the steerer finally fell through to supersede.

The fix: bucket the validator message into a structural-class kind,
remember the prior attempt's kind, and short-circuit to ``None`` when
two consecutive attempts hit the same kind. ``None`` signals the
caller to fall through to the supersede path. The
``REFINE_VALIDATION_FAILED`` drift is still emitted on short-circuit
so the steerer's escalation ladder gets a uniform signal regardless
of whether retries were exhausted by exhaustion or by short-circuit.

What's NOT short-circuited:

* Different rejection kinds across attempts — the LLM may yet
  converge if it's making progress.
* Non-validator errors (parse failures, goal-coverage errors,
  assignee errors) — those are correctable shapes and the existing
  feedback loop is productive there.
* The empty-response sentinel (goldfive#182) — that has its own
  no-retry path which still returns silently without emitting.

Test strategy: the loop body in ``_refine_steer`` tracks the
short-circuit kind based on the ``error`` string returned by
``_user_steer_one_attempt``. Driving the loop end-to-end through a
live LLM stub is awkward because the merge logic in
``_user_steer_one_attempt`` re-injects completed tasks before
validation, so a hand-crafted "broken" JSON usually won't actually
trigger ``terminal task ... missing/regressed`` (the merge fixes it).
We test the loop directly by patching ``_user_steer_one_attempt`` to
return scripted ``(merged_plan, error)`` tuples; this is the right
boundary because the unit under test is the retry-loop control flow,
not the merge / validator plumbing (which has its own coverage in
``test_user_steer_invariant.py`` / ``test_types.py``).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from goldfive.planner import LLMPlanner
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
# Helpers
# ---------------------------------------------------------------------------


def _goals() -> list[Goal]:
    return [
        Goal(id="g1", summary="Draft a blog post about goldfish."),
        Goal(id="g2", summary="Get one round of editorial review."),
    ]


def _running_plan() -> Plan:
    """Plan with two COMPLETED terminal tasks plus a PENDING tail."""
    return Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1", "g2"],
        tasks=[
            Task(
                id="research",
                title="Research goldfish facts",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft the post",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="review",
                title="Review the draft",
                assignee_agent_id="editor",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
            TaskEdge(from_task_id="draft", to_task_id="review"),
        ],
        summary="Draft and review a goldfish blog post.",
        revision_index=0,
    )


def _user_steer_drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please add a fact-check step before review",
    )


# Validator messages keyed on by ``_extract_rejection_kind``. These are
# the EXACT formatter outputs from ``goldfive.types.Plan.validate``
# (cross-referenced in ``tests/test_types.py``); pasted here verbatim
# so any future drift between validator and classifier surfaces here.
_TERMINAL_MISSING_ERROR: str = (
    "validator rejected revision: terminal task 'draft_slides' missing in revision"
)
_TERMINAL_REGRESSED_ERROR: str = (
    "validator rejected revision: terminal task 'draft_slides' regressed to 'PENDING'"
)
_TERMINAL_EDGE_MISSING_ERROR: str = (
    "validator rejected revision: terminal->terminal edge 'research' -> 'draft' missing in revision"
)


class _ScriptedSteerOneAttempt:
    """Replacement for ``_user_steer_one_attempt`` that scripts results.

    Each call pops the next ``(merged_plan, error)`` tuple from
    ``responses``. Used to drive ``_refine_steer``'s retry-loop
    control flow without paying the cost of a full LLM stub +
    JSON-merge round-trip (the merge re-injects completed tasks so a
    hand-crafted "broken" JSON often won't actually trigger the
    validator rejection we're trying to test).
    """

    def __init__(self, responses: list[tuple[Plan | None, str]]) -> None:
        self.responses = list(responses)
        self.user_prompts: list[str] = []

    async def __call__(self, **kwargs: Any) -> tuple[Plan | None, str]:
        self.user_prompts.append(kwargs.get("user_prompt", ""))
        if not self.responses:
            raise AssertionError("_user_steer_one_attempt called more times than expected")
        return self.responses.pop(0)


def _make_planner() -> LLMPlanner:
    """Construct an ``LLMPlanner`` with a placeholder ``call_llm`` that
    must never be reached — the tests patch ``_user_steer_one_attempt``
    so the underlying LLM dispatch is bypassed entirely."""

    async def _unreachable_call_llm(system: str, user: str, model: str) -> str:
        raise AssertionError("call_llm should not be reached in these tests")

    return LLMPlanner(call_llm=_unreachable_call_llm)


async def _drive_refine_steer(
    planner: LLMPlanner,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[Plan | None, str]],
) -> tuple[Plan | None, _ScriptedSteerOneAttempt]:
    """Patch ``_user_steer_one_attempt`` and call ``planner.refine``."""
    scripted = _ScriptedSteerOneAttempt(responses)
    monkeypatch.setattr(planner, "_user_steer_one_attempt", scripted, raising=True)
    result = await planner.refine(plan=_running_plan(), drift=_user_steer_drift(), goals=_goals())
    return result, scripted


# ---------------------------------------------------------------------------
# Unit: the classifier
# ---------------------------------------------------------------------------


def test_extract_rejection_kind_classifies_known_patterns() -> None:
    """All three documented validator shapes round-trip to a kind, plus
    the wrapper prefix from ``_user_steer_one_attempt`` is tolerated.
    Unrelated errors (and empty / None) bucket to ``None`` — the caller
    only short-circuits when both attempts produce a bucketed kind, so
    plumbing failures and goal-coverage errors are unaffected."""
    f = LLMPlanner._extract_rejection_kind

    # The three shapes from goldfive/types.py Plan.validate.
    assert f("terminal task 'draft_slides' missing in revision") == "terminal_missing"
    assert f("terminal task 't1' regressed to 'PENDING'") == "terminal_regressed"
    assert f("terminal task 't1' regressed to 'FAILED'") == "terminal_regressed"
    assert f("terminal->terminal edge 'a' -> 'b' missing in revision") == "edge_missing"

    # The validator string is wrapped by ``_user_steer_one_attempt``:
    # ``f"validator rejected revision: {exc}"``. The classifier matches
    # by substring so the prefix doesn't hide the kind.
    assert (
        f("validator rejected revision: terminal task 'x' missing in revision")
        == "terminal_missing"
    )
    assert (
        f("validator rejected revision: terminal task 'x' regressed to 'PENDING'")
        == "terminal_regressed"
    )

    # Case-insensitive: paranoia for any future caller that upper-cases.
    assert (
        f("VALIDATOR REJECTED REVISION: TERMINAL TASK 'X' MISSING IN REVISION")
        == "terminal_missing"
    )

    # Empty / None / unrelated -> None (no short-circuit).
    assert f("") is None
    assert f("JSON parse failed: Expecting value") is None
    assert f("call_llm raised: timeout") is None
    assert f("parsed JSON did not contain a usable plan") is None
    # Goal-coverage error has neither "terminal task" nor "edge" wording.
    assert (
        f(
            "revision silently drops USER_STEER goal(s) (no task references "
            "them): [g] do thing. Operator steers are sticky -- ..."
        )
        is None
    )


# ---------------------------------------------------------------------------
# Refine retry loop: short-circuits on repeated same-class rejections
# ---------------------------------------------------------------------------


async def test_refine_short_circuits_on_repeated_terminal_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both attempts return the same ``terminal_missing`` rejection →
    second attempt's failure short-circuits to ``None``.

    Default budget is 2 so this is observable as 2 attempts (same
    count as exhaustion would have produced); the load-bearing
    behaviour is the short-circuit log message + the
    ``REFINE_VALIDATION_FAILED`` emit firing exactly once. The higher-
    budget regression below confirms call-count savings when the
    budget is > 2.
    """
    caplog.set_level(logging.INFO, logger="goldfive.planner")
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    planner = _make_planner()
    planner.set_drift_emitter(emitter)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_MISSING_ERROR),
            (None, _TERMINAL_MISSING_ERROR),
        ],
    )
    assert result is None
    assert len(scripted.user_prompts) == 2
    # Attempt 2 saw the correction feedback (loop still hands feedback
    # to the LLM before the post-call short-circuit decision).
    assert "PREVIOUS ATTEMPT FAILED" in scripted.user_prompts[1]
    # Short-circuit log fired (not the regular exhaustion path).
    short_circuit_msgs = [
        r for r in caplog.records if "short-circuiting to supersede" in r.getMessage()
    ]
    assert short_circuit_msgs, "expected short-circuit INFO log"
    assert "'terminal_missing'" in short_circuit_msgs[0].getMessage()
    # REFINE_VALIDATION_FAILED still emitted (steerer ladder needs it).
    assert [e.kind for e in emitted] == [DriftKind.REFINE_VALIDATION_FAILED]


async def test_refine_short_circuits_on_repeated_terminal_regressed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same shape but with the regressed-status rejection."""
    caplog.set_level(logging.INFO, logger="goldfive.planner")
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    planner = _make_planner()
    planner.set_drift_emitter(emitter)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_REGRESSED_ERROR),
            (None, _TERMINAL_REGRESSED_ERROR),
        ],
    )
    assert result is None
    assert len(scripted.user_prompts) == 2
    short_circuit_msgs = [
        r for r in caplog.records if "short-circuiting to supersede" in r.getMessage()
    ]
    assert short_circuit_msgs, "expected short-circuit INFO log"
    assert "'terminal_regressed'" in short_circuit_msgs[0].getMessage()
    assert [e.kind for e in emitted] == [DriftKind.REFINE_VALIDATION_FAILED]


async def test_refine_does_not_short_circuit_on_different_rejection_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt 1 gets ``terminal_missing``; attempt 2 gets ``edge_missing``.
    The kinds differ → the LLM may yet converge → no short circuit;
    the loop must run the full retry budget.

    With ``max_refine_attempts=3`` we exercise the different-kind
    branch on attempt 2 (which keeps ``prior_error_kind`` updated to
    the new kind) and let attempt 3 also be a different kind so the
    full budget gets used. Confirms the kinds-differ branch does NOT
    short-circuit and the full budget is consumed.
    """
    planner = LLMPlanner(call_llm=_make_planner()._call_llm, max_refine_attempts=3)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_MISSING_ERROR),  # terminal_missing
            (None, _TERMINAL_EDGE_MISSING_ERROR),  # edge_missing — different
            (None, _TERMINAL_REGRESSED_ERROR),  # terminal_regressed — different
        ],
    )
    assert result is None
    # Full budget consumed — no short-circuit fired.
    assert len(scripted.user_prompts) == 3


async def test_refine_succeeds_on_attempt_2_after_correction_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: existing feedback-driven retry success path is
    preserved. Attempt 1 fails with a validator rejection, attempt 2
    succeeds and the loop returns the merged plan."""
    # Construct a "valid" merged plan that satisfies the loop's success
    # branch — we never reach validation because we're returning the
    # success tuple directly from the patched helper.
    success_plan = Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1", "g2"],
        tasks=[
            Task(
                id="research",
                title="Research",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="fact_check",
                title="Fact-check",
                assignee_agent_id="editor",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
        summary="ok",
        revision_index=1,
    )
    planner = _make_planner()
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_MISSING_ERROR),
            (success_plan, ""),
        ],
    )
    assert result is success_plan
    assert len(scripted.user_prompts) == 2
    # Attempt 2 saw the correction feedback.
    assert "PREVIOUS ATTEMPT FAILED" in scripted.user_prompts[1]


async def test_refine_first_attempt_success_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: first-attempt success short-circuits the retry loop
    BEFORE the rejection-kind tracking runs (the kind is only consulted
    on failure paths)."""
    success_plan = Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="ok",
        revision_index=1,
    )
    planner = LLMPlanner(call_llm=_make_planner()._call_llm, max_refine_attempts=4)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [(success_plan, "")],
    )
    assert result is success_plan
    assert len(scripted.user_prompts) == 1


async def test_refine_short_circuits_with_higher_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``max_refine_attempts`` is larger than the default 2, the
    short-circuit cuts the loop after attempt 2 instead of burning
    every attempt. Two same-kind failures → 2 attempts, not 4."""
    planner = LLMPlanner(call_llm=_make_planner()._call_llm, max_refine_attempts=4)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_MISSING_ERROR),
            (None, _TERMINAL_MISSING_ERROR),
            # The next two should never be consumed — short-circuit
            # fires at the end of attempt 2.
            (None, _TERMINAL_MISSING_ERROR),
            (None, _TERMINAL_MISSING_ERROR),
        ],
    )
    assert result is None
    assert len(scripted.user_prompts) == 2


async def test_refine_short_circuit_emits_validation_failed_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-circuit emits exactly one ``REFINE_VALIDATION_FAILED`` —
    same shape as the exhaustion path, so the steerer ladder cannot
    tell them apart and reacts identically."""
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    planner = _make_planner()
    planner.set_drift_emitter(emitter)
    await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, _TERMINAL_MISSING_ERROR),
            (None, _TERMINAL_MISSING_ERROR),
        ],
    )
    assert len(emitted) == 1
    assert emitted[0].kind is DriftKind.REFINE_VALIDATION_FAILED
    # The emit carries the last error so operator dashboards still see
    # the actual validator complaint, not a synthetic short-circuit
    # marker. (Detail format set by ``_emit_refine_validation_failed``.)
    assert "missing in revision" in emitted[0].detail


async def test_refine_does_not_short_circuit_on_unbucketed_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two attempts both fail with errors the classifier doesn't
    bucket (parse error, goal-coverage, plumbing failure). Those must
    NOT short-circuit — the LLM might still converge given another
    correction prompt, and many such errors are productive on retry.
    The loop runs to natural exhaustion."""
    planner = LLMPlanner(call_llm=_make_planner()._call_llm, max_refine_attempts=4)
    result, scripted = await _drive_refine_steer(
        planner,
        monkeypatch,
        [
            (None, "JSON parse failed: Expecting value"),
            (None, "JSON parse failed: Expecting value"),
            (None, "parsed JSON did not contain a usable plan"),
            (None, "call_llm raised: timeout"),
        ],
    )
    assert result is None
    # Full budget consumed — short-circuit only fires for bucketed kinds.
    assert len(scripted.user_prompts) == 4
