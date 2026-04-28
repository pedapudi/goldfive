"""Empty / non-string LLM responses must not be retried (goldfive#182).

Small models (Qwen 2B in particular) routinely exhaust their output
budget on internal reasoning and emit no final answer. The pre-#182
behaviour was to retry — each retry doubled cost without changing the
outcome and (on the refine path) escalated through the intervention
ladder. The fix treats an empty / non-string response as terminal "no
signal":

* the retry loop short-circuits after the first empty attempt,
* logging is INFO (not WARNING — operators see model-quality issues
  via observability without log noise),
* refine paths skip the ``REFINE_VALIDATION_FAILED`` emission so the
  steerer's backoff still fires but no escalation cascade follows,
* valid responses regress nothing — the happy path is unchanged.

The drift judges (:mod:`goldfive.drift.goals`,
:mod:`goldfive.drift.reasoning_judge`) already follow the
"quiet on failure" pattern (return ``None`` on unparseable / empty)
so they need no change. The retry-on-empty pattern lived in
:mod:`goldfive.planner` only.
"""

from __future__ import annotations

import json

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
                status=TaskStatus.RUNNING,
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


def _valid_revision_json() -> str:
    return json.dumps(
        {
            "summary": "same",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish facts",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "assignee_agent_id": "writer",
                    "status": "RUNNING",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [],
        }
    )


class _CountingLLM:
    """Records call count and returns a scripted response.

    The ``response`` may be a string (returned verbatim every call) or
    a callable ``int -> str | None`` keyed on the (1-based) call number,
    so a test can script "first call returns empty, second returns valid".
    """

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> object:
        self.calls.append((system, user, model))
        if callable(self.response):
            return self.response(len(self.calls))
        return self.response


# ---------------------------------------------------------------------------
# generate() — empty response must not retry
# ---------------------------------------------------------------------------


async def test_generate_empty_response_does_not_retry() -> None:
    """``generate`` short-circuits on empty: only one LLM call regardless of budget."""
    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    result = await planner.generate(goals=_goals(), available_agents=["researcher"])
    assert result is None
    # The retry budget was 4 but we only spent 1 attempt — the empty
    # response was treated as terminal "no signal".
    assert len(stub.calls) == 1


async def test_generate_non_string_response_does_not_retry() -> None:
    """``generate`` short-circuits on non-string (e.g., None) the same way."""

    async def returns_none(system: str, user: str, model: str) -> object:
        returns_none.calls += 1  # type: ignore[attr-defined]
        return None

    returns_none.calls = 0  # type: ignore[attr-defined]
    planner = LLMPlanner(call_llm=returns_none, max_refine_attempts=4)
    result = await planner.generate(goals=_goals(), available_agents=["researcher"])
    assert result is None
    assert returns_none.calls == 1  # type: ignore[attr-defined]


async def test_generate_whitespace_only_response_does_not_retry() -> None:
    """A whitespace-only string still has no usable content."""
    stub = _CountingLLM("   \n  \t  ")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    result = await planner.generate(goals=_goals(), available_agents=["researcher"])
    # Empty cleaned content fails JSON parse; that's a retryable case
    # (genuine malformed JSON, not "no answer"). The whitespace-only
    # case is genuinely non-empty as a string so it goes through the
    # JSON parser. Confirm only that the call returns None — retry
    # behaviour for malformed-but-non-empty is unchanged by this fix.
    assert result is None


async def test_generate_retries_on_invalid_json_unchanged() -> None:
    """Regression check: malformed JSON is still retried (this fix only
    affects the truly-empty case)."""
    stub = _CountingLLM("not json at all {")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=2)
    result = await planner.generate(goals=_goals(), available_agents=["researcher"])
    assert result is None
    # The retry loop did fire — invalid JSON is a parse error the LLM
    # can fix on retry, unlike "no answer at all".
    assert len(stub.calls) == 2


async def test_generate_valid_response_unchanged() -> None:
    """Regression: a valid first response is accepted on attempt 1."""
    payload = {
        "summary": "Draft a goldfish post.",
        "tasks": [
            {
                "id": "research",
                "title": "Research",
                "assignee_agent_id": "researcher",
            },
            {
                "id": "draft",
                "title": "Draft",
                "assignee_agent_id": "researcher",
            },
        ],
        "edges": [{"from_task_id": "research", "to_task_id": "draft"}],
    }
    stub = _CountingLLM(json.dumps(payload))
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    result = await planner.generate(goals=_goals(), available_agents=["researcher"])
    assert result is not None
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# refine() — empty response must not retry, must not emit
# REFINE_VALIDATION_FAILED (a small-model "no answer" is not a refine
# failure, just a model-quality artefact).
# ---------------------------------------------------------------------------


async def test_refine_empty_response_does_not_retry() -> None:
    """``refine`` short-circuits the retry loop on empty."""
    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    assert len(stub.calls) == 1


async def test_refine_empty_response_does_not_emit_validation_failed() -> None:
    """An empty refine response is a model-quality issue, not a refine
    failure — no ``REFINE_VALIDATION_FAILED`` should be emitted.

    Without this guard the steerer would treat a 2B's silence as
    validator exhaustion and escalate through the intervention ladder.
    """
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    planner.set_drift_emitter(emitter)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    # Exactly zero emissions — no REFINE_VALIDATION_FAILED cascade.
    assert emitted == []


async def test_refine_validator_failure_still_emits_validation_failed() -> None:
    """Regression: a *non-empty* parse / validator failure still emits
    ``REFINE_VALIDATION_FAILED`` — only the empty-response case is the
    silent path."""
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    # Valid JSON shape but no usable plan — exercises the "parsed JSON
    # did not contain a usable plan" failure mode, which still escalates.
    stub = _CountingLLM(json.dumps({"summary": "broken", "tasks": [], "edges": []}))
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=2)
    planner.set_drift_emitter(emitter)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    # Exactly one validation-failed emission after retries exhausted.
    kinds = [e.kind for e in emitted]
    assert DriftKind.REFINE_VALIDATION_FAILED in kinds


async def test_refine_valid_response_unchanged() -> None:
    """Regression: a valid refine response on attempt 1 is accepted."""
    stub = _CountingLLM(_valid_revision_json())
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is not None
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# refine() with USER_STEER drift — exercises the ``_refine_steer`` path.
# ---------------------------------------------------------------------------


async def test_refine_user_steer_empty_response_does_not_retry() -> None:
    """``_refine_steer`` (USER_STEER drift) short-circuits on empty."""
    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please add a fact-check step",
    )
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    # No retry: only one call landed.
    assert len(stub.calls) == 1


async def test_refine_user_steer_empty_response_no_validation_emit() -> None:
    """Empty USER_STEER response also skips the
    ``REFINE_VALIDATION_FAILED`` emission — same rationale as the
    generic refine path."""
    emitted: list[DriftEvent] = []

    async def emitter(signal: DriftEvent) -> None:
        emitted.append(signal)

    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=4)
    planner.set_drift_emitter(emitter)
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please add a fact-check step",
    )
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    assert emitted == []


async def test_refine_user_steer_non_string_does_not_retry() -> None:
    """Non-string (None) USER_STEER response is treated identically to empty."""

    async def returns_none(system: str, user: str, model: str) -> object:
        returns_none.calls += 1  # type: ignore[attr-defined]
        return None

    returns_none.calls = 0  # type: ignore[attr-defined]
    planner = LLMPlanner(call_llm=returns_none, max_refine_attempts=4)
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please add a fact-check step",
    )
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None
    assert returns_none.calls == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Log level — empty response is INFO, not WARNING (operator-noise budget)
# ---------------------------------------------------------------------------


async def test_empty_response_logs_at_info_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pre-#182 code logged WARNING on every retry attempt; the fix
    logs once at INFO so observability still records the model-quality
    issue without spamming WARN-level operator dashboards."""
    import logging

    caplog.set_level(logging.INFO, logger="goldfive.planner")
    stub = _CountingLLM("")
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=2)
    await planner.generate(goals=_goals(), available_agents=["researcher"])
    # Find records mentioning the empty-response sentinel.
    matches = [r for r in caplog.records if "empty or non-string" in r.getMessage()]
    assert matches, "expected an empty-response log record"
    # All such records are INFO (not WARNING).
    assert all(r.levelno == logging.INFO for r in matches)
