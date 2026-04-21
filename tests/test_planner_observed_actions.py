"""Tests for ``LLMPlanner.refine`` with ``observed_actions`` (goldfive#144).

The overlay reconciler emits ``PLAN_DIVERGENCE`` when the agent tree
performs invocations that don't match the planned dispatch. The refine
path for that drift kind takes ``observed_actions`` and asks the LLM to
either ABSORB the observed activity into a revised plan or REJECT via
the ``{"reject": true, ...}`` sentinel (returns ``None`` → caller
escalates via the intervention ladder, #142).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from goldfive.planner import LLMPlanner
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    ObservedAction,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers (local; mirror the shape of the main test_planner.py helpers)
# ---------------------------------------------------------------------------


def _goals() -> list[Goal]:
    return [
        Goal(id="g1", summary="Draft a blog post about goldfish."),
        Goal(id="g2", summary="Get one round of editorial review."),
    ]


def _running_plan() -> Plan:
    """A plan with one COMPLETED task, one RUNNING task, one PENDING."""
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


def _observed_actions() -> list[ObservedAction]:
    """Observed tree activity: research done, draft in-flight, plus a
    bonus ``citation_check`` invocation the plan didn't anticipate."""
    base = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    return [
        ObservedAction(
            agent_name="researcher",
            invocation_id="inv-1",
            parent_invocation_id="",
            started_at=base,
            completed_at=base + timedelta(seconds=30),
            status="completed",
            summary="Gathered 5 cited facts about goldfish.",
        ),
        ObservedAction(
            agent_name="writer",
            invocation_id="inv-2",
            parent_invocation_id="",
            started_at=base + timedelta(seconds=31),
            completed_at=None,
            status="running",
            summary="Drafting the post",
        ),
        ObservedAction(
            agent_name="citation_bot",
            invocation_id="inv-3",
            parent_invocation_id="inv-2",
            started_at=base + timedelta(seconds=45),
            completed_at=base + timedelta(seconds=55),
            status="completed",
            summary="Verified citations for the draft.",
        ),
    ]


class _StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response


class _ScriptedLLM:
    """Returns scripted responses in order; raises if called too often."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        if not self.responses:
            raise AssertionError("scripted LLM called more times than expected")
        return self.responses.pop(0)


def _absorbed_revision_json() -> str:
    """A legitimate ABSORB revision: observed actions reflected as
    tasks (research COMPLETED, draft COMPLETED, plus a new
    ``citation_check`` task, review PENDING)."""
    return json.dumps(
        {
            "summary": "Draft, verify citations, review the goldfish post.",
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
                    "status": "COMPLETED",
                },
                {
                    "id": "citation_check",
                    "title": "Verify citations",
                    "description": "Check all draft citations for accuracy.",
                    "assignee_agent_id": "citation_bot",
                    "status": "COMPLETED",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft"},
                {"from_task_id": "draft", "to_task_id": "citation_check"},
                {"from_task_id": "citation_check", "to_task_id": "review"},
            ],
        }
    )


def _bad_revision_dropped_terminal_task_json() -> str:
    """An invalid revision that DROPS the COMPLETED ``research`` task
    from the prior plan, violating the terminal-task preservation
    invariant (PLAN-LIFECYCLE.md §3.1)."""
    return json.dumps(
        {
            "summary": "absorb but broken",
            "tasks": [
                # ``research`` (COMPLETED in the prior plan) is missing.
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "assignee_agent_id": "writer",
                    "status": "COMPLETED",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "draft", "to_task_id": "review"},
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. ABSORB: LLM returns a revised plan reflecting observed actions
# ---------------------------------------------------------------------------


async def test_refine_plan_divergence_with_observed_actions_absorbs() -> None:
    """When the LLM absorbs the observed activity, the revised plan
    must include the observed tasks as COMPLETED (matching the summaries
    the reconciler surfaced)."""
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree called citation_bot, not in plan",
        current_task_id="draft",
    )

    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals(),
        observed_actions=_observed_actions(),
    )

    assert revised is not None
    by_id = {t.id: t for t in revised.tasks}
    # Observed-and-completed invocations flipped to COMPLETED.
    assert by_id["research"].status == TaskStatus.COMPLETED
    assert by_id["draft"].status == TaskStatus.COMPLETED
    # The invocation that was not in the plan was added.
    assert "citation_check" in by_id
    assert by_id["citation_check"].status == TaskStatus.COMPLETED
    # The still-pending task remains PENDING.
    assert by_id["review"].status == TaskStatus.PENDING
    # Revision metadata was stamped.
    assert revised.revision_kind == DriftKind.PLAN_DIVERGENCE.value
    assert revised.revision_index == 1


# ---------------------------------------------------------------------------
# 2. REJECT: LLM emits the reject sentinel, refine returns None
# ---------------------------------------------------------------------------


async def test_refine_plan_divergence_observed_actions_off_goal_returns_none() -> None:
    """When the LLM decides the divergence is off-goal, it emits
    ``{"reject": true, "reason": "..."}`` and refine returns None (→
    the caller escalates to human intervention via the ladder)."""
    stub = _StubLLM(json.dumps({"reject": True, "reason": "off-goal excursion"}))
    planner = LLMPlanner(call_llm=stub)

    emitted: list[DriftEvent] = []

    async def capture(d: DriftEvent) -> None:  # pragma: no cover - see assert below
        emitted.append(d)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree wandered",
        current_task_id="draft",
    )

    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals(),
        observed_actions=_observed_actions(),
    )

    assert revised is None
    # A reject is a successful decision, not a validation failure — no
    # REFINE_VALIDATION_FAILED drift should be emitted.
    assert emitted == []
    # The LLM was called exactly once — no retry on a valid reject.
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# 3. Back-compat: refine() without observed_actions behaves as before
# ---------------------------------------------------------------------------


async def test_refine_without_observed_actions_unchanged() -> None:
    """The legacy calling convention (no ``observed_actions``) still
    returns a refined plan using the generic refine prompt."""
    payload = {
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
        "edges": [
            {"from_task_id": "research", "to_task_id": "draft"},
            {"from_task_id": "draft", "to_task_id": "review"},
        ],
    }
    stub = _StubLLM(json.dumps(payload))
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree wandered",
        current_task_id="draft",
    )

    # No observed_actions -> generic refine path, observed block absent,
    # reject sentinel not honoured.
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert revised is not None
    assert len(revised.tasks) == 3
    _system, user_prompt, _model = stub.calls[0]
    assert "OBSERVED AGENT ACTIVITY" not in user_prompt


# ---------------------------------------------------------------------------
# 4. Prompt structure: observed actions section is in the prompt
# ---------------------------------------------------------------------------


async def test_refine_prompt_includes_observed_actions_section() -> None:
    """Structural check on the built prompt: when observed_actions is
    supplied, the prompt must include the OBSERVED AGENT ACTIVITY
    header, the summaries, and the ABSORB/REJECT decision guidance."""
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="divergence",
        current_task_id="draft",
    )

    await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals(),
        observed_actions=_observed_actions(),
    )

    system, user_prompt, _model = stub.calls[0]
    # Divergence-specific system prompt (not the generic refine prompt).
    assert "PLAN_DIVERGENCE" in system
    assert "ABSORB" in system
    assert "REJECT" in system
    # The user prompt includes the observed activity block.
    assert "OBSERVED AGENT ACTIVITY" in user_prompt
    assert "citation_bot" in user_prompt
    assert "Gathered 5 cited facts about goldfish." in user_prompt
    # Decision guidance is present.
    assert "reject" in user_prompt.lower()
    assert "revised plan" in user_prompt.lower() or "REFLECTS" in user_prompt
    # The existing invariants are preserved on the divergence path.
    assert "TERMINAL" in user_prompt.upper()


async def test_refine_without_observed_actions_uses_generic_system_prompt() -> None:
    """Even on PLAN_DIVERGENCE, if observed_actions is None the planner
    falls back to the generic refine system prompt (not the divergence
    one) — keeps legacy callers unchanged."""
    payload = {
        "summary": "same",
        "tasks": [
            {
                "id": "research",
                "title": "Research",
                "assignee_agent_id": "researcher",
                "status": "COMPLETED",
            },
            {
                "id": "draft",
                "title": "Draft",
                "assignee_agent_id": "writer",
                "status": "RUNNING",
            },
            {
                "id": "review",
                "title": "Review",
                "assignee_agent_id": "editor",
                "status": "PENDING",
            },
        ],
        "edges": [
            {"from_task_id": "research", "to_task_id": "draft"},
            {"from_task_id": "draft", "to_task_id": "review"},
        ],
    }
    stub = _StubLLM(json.dumps(payload))
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="divergence",
        current_task_id="draft",
    )
    await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    system, _user, _model = stub.calls[0]
    # The divergence system prompt has the ABSORB/REJECT sections; the
    # generic refine prompt does not.
    assert "ABSORB" not in system
    # The generic refine prompt references "maintaining an ACTIVE plan".
    assert "maintaining an ACTIVE plan" in system


# ---------------------------------------------------------------------------
# 5. Retry loop with observed actions: first attempt invalid, second ok
# ---------------------------------------------------------------------------


async def test_refine_retry_loop_with_observed_actions() -> None:
    """First attempt drops a terminal->terminal edge → validator rejects
    and the LLM is re-prompted with the error. Second attempt corrects
    and returns a valid plan. The observed_actions block must appear in
    both attempts so the second try still sees the reconciler context."""
    scripted = _ScriptedLLM(
        [
            _bad_revision_dropped_terminal_task_json(),
            _absorbed_revision_json(),
        ]
    )
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="divergence",
        current_task_id="draft",
    )

    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals(),
        observed_actions=_observed_actions(),
    )

    assert revised is not None
    assert len(scripted.calls) == 2
    _sys1, first_user, _m1 = scripted.calls[0]
    _sys2, second_user, _m2 = scripted.calls[1]
    # Observed-actions block is present on BOTH attempts.
    assert "OBSERVED AGENT ACTIVITY" in first_user
    assert "OBSERVED AGENT ACTIVITY" in second_user
    # The retry carries explicit correction feedback.
    assert "PREVIOUS ATTEMPT FAILED" in second_user
    # The final plan reflects the absorb revision.
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["citation_check"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 6. Reject sentinel is NOT honoured when observed_actions is absent
# ---------------------------------------------------------------------------


async def test_reject_sentinel_ignored_without_observed_actions() -> None:
    """The reject sentinel is only valid on the PLAN_DIVERGENCE-with-
    observed-actions path. Without observed_actions, a ``{"reject": ...}``
    response is treated as an invalid plan (no usable tasks) and refine
    returns None via the normal exhaustion path."""
    # Single attempt so we can assert the "no plan found" exhaustion
    # path collapses to None without engaging reject-sentinel logic.
    scripted = _ScriptedLLM([json.dumps({"reject": True, "reason": "x"})])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="divergence",
        current_task_id="draft",
    )
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    # Without observed_actions, the response is not a reject; it's just
    # a malformed plan, which the normal "no usable plan" path rejects.
    assert revised is None


# ---------------------------------------------------------------------------
# 7. ObservedAction dataclass smoke
# ---------------------------------------------------------------------------


def test_observed_action_dataclass_shape() -> None:
    """Sanity-check the dataclass accepts the fields documented in the
    issue and can carry an in-flight invocation (completed_at=None)."""
    now = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    a = ObservedAction(
        agent_name="writer",
        invocation_id="inv-42",
        parent_invocation_id="",
        started_at=now,
        completed_at=None,
        status="running",
        summary="drafting",
    )
    assert a.completed_at is None
    assert a.parent_invocation_id == ""
    assert a.status == "running"
