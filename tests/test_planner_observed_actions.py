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


# ---------------------------------------------------------------------------
# 8. OFF_TOPIC routes to the goal-aware refine prompt
# ---------------------------------------------------------------------------


async def test_refine_off_topic_uses_plan_divergence_system_prompt() -> None:
    """OFF_TOPIC drift is plan-context drift from the reasoning judge:
    the agent is reasoning about something that doesn't fit the bound
    task. The ABSORB path must use the goal-aware
    ``_PLAN_DIVERGENCE_SYSTEM_PROMPT`` so the LLM either revises the
    plan to absorb the new direction (when it advances a goal) or
    emits the ``{"reject": true, ...}`` sentinel (when it doesn't),
    rather than silently absorbing off-goal reasoning via the generic
    ``_REFINE_SYSTEM_PROMPT``.
    """
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="reasoning drift: agent began researching tropical fish, not goldfish",
        current_task_id="draft",
        trigger_input="Let me research tropical fish habitats and breeding patterns...",
    )

    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert revised is not None
    system, user_prompt, _model = stub.calls[0]
    # The divergence (goal-aware) system prompt was selected, NOT the
    # generic refine prompt. Two structural markers:
    #   1. Goal-aware prompt frames PLAN_DIVERGENCE / ABSORB / REJECT.
    #   2. Generic prompt opens with "maintaining an ACTIVE plan" without
    #      the ABSORB/REJECT contract.
    assert "ABSORB" in system
    assert "REJECT" in system
    # The user prompt carries the OFF_TOPIC reasoning context block —
    # the analogue of OBSERVED AGENT ACTIVITY for a reasoning-judge
    # drift — so the system prompt's referenced "what the agent did"
    # channel is honoured.
    assert "OFF-TOPIC REASONING" in user_prompt
    assert "tropical fish" in user_prompt
    assert "judge reason" in user_prompt
    # The user prompt should NOT include the OBSERVED AGENT ACTIVITY
    # header — that's the PLAN_DIVERGENCE+observed_actions channel and
    # OFF_TOPIC has no observed actions.
    assert "OBSERVED AGENT ACTIVITY" not in user_prompt


async def test_refine_off_topic_honours_reject_sentinel() -> None:
    """When the LLM judges the off-topic reasoning to be off-goal, it
    emits ``{"reject": true, "reason": "..."}`` and refine returns
    ``None`` (so the steerer can escalate via the intervention ladder).
    No REFINE_VALIDATION_FAILED is emitted because reject is a
    successful decision, not a parse failure.
    """
    stub = _StubLLM(json.dumps({"reject": True, "reason": "off-goal excursion"}))
    planner = LLMPlanner(call_llm=stub)

    emitted: list[DriftEvent] = []

    async def capture(d: DriftEvent) -> None:
        emitted.append(d)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="reasoning drifted to unrelated topic",
        current_task_id="draft",
    )

    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert revised is None
    assert emitted == []
    # Reject is final — no retry.
    assert len(stub.calls) == 1


async def test_refine_off_topic_without_trigger_input_renders_placeholder() -> None:
    """The OFF_TOPIC reasoning block has invariant shape: when the drift
    carries no ``trigger_input`` / ``raw`` / ``detail`` excerpt the
    block still renders with a placeholder so the prompt is well-formed
    (mirrors the ``(no observed activity)`` fallback on the
    PLAN_DIVERGENCE path)."""
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="",
        current_task_id="draft",
    )

    await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    _system, user_prompt, _model = stub.calls[0]
    assert "OFF-TOPIC REASONING" in user_prompt
    assert "(no reasoning excerpt available)" in user_prompt


async def test_refine_repeated_failure_keeps_generic_prompt() -> None:
    """REPEATED_FAILURE / BLOCKED are structural-recovery drifts, not
    plan-context drifts. They must continue to use the generic
    ``_REFINE_SYSTEM_PROMPT`` — accidentally migrating them to the
    goal-aware ABSORB/REJECT prompt would let the LLM emit
    ``{"reject": true}`` for what is just a structural recovery and
    that path is wired to escalate to human intervention.
    """
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
    for kind in (DriftKind.REPEATED_FAILURE, DriftKind.BLOCKED):
        stub = _StubLLM(json.dumps(payload))
        planner = LLMPlanner(call_llm=stub)
        drift = DriftEvent(
            kind=kind,
            severity=DriftSeverity.WARNING,
            detail="recovery",
            current_task_id="draft",
        )
        await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
        system, user_prompt, _model = stub.calls[0]
        # Generic refine system prompt selected, NOT the goal-aware one.
        assert "ABSORB" not in system, (
            f"{kind.value} accidentally migrated to the goal-aware prompt"
        )
        assert "maintaining an ACTIVE plan" in system
        # No OFF_TOPIC reasoning block leaks onto non-OFF_TOPIC kinds.
        assert "OFF-TOPIC REASONING" not in user_prompt
        # No OBSERVED AGENT ACTIVITY block (no observed_actions provided).
        assert "OBSERVED AGENT ACTIVITY" not in user_prompt


# ---------------------------------------------------------------------------
# 9. JUSTIFIED_DEVIATION (iter-10 PR 4) routes to the goal-aware refine prompt
# ---------------------------------------------------------------------------


async def test_refine_justified_deviation_uses_goal_aware_prompt() -> None:
    """JUSTIFIED_DEVIATION (iter-10) shares OFF_TOPIC's refine path.

    The drift's ``detail`` carries the provenance prefix from the
    parser ("justified deviation (tool_error): ...") which the
    existing ``_render_off_topic_reasoning_block`` surfaces verbatim
    so the LLM sees the provoking signal.
    """
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.JUSTIFIED_DEVIATION,
        severity=DriftSeverity.WARNING,
        detail=(
            "justified deviation (tool_error): tool 503; falling back "
            "to a different fact source"
        ),
        current_task_id="draft",
        trigger_input=(
            "The fact API returned 503; let me retry against the "
            "alternate endpoint."
        ),
    )

    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert revised is not None
    system, user_prompt, _model = stub.calls[0]
    # Goal-aware system prompt selected (ABSORB / REJECT framing).
    assert "ABSORB" in system
    assert "REJECT" in system
    # The user prompt carries the OFF-TOPIC reasoning context block —
    # _render_off_topic_reasoning_block is shared with OFF_TOPIC by
    # design (the drift detail / trigger_input shape is identical).
    assert "OFF-TOPIC REASONING" in user_prompt
    # Provenance prefix from the parser appears verbatim in the
    # rendered context, so the LLM sees the exact provoking signal
    # rather than a generic "agent drifted" framing.
    assert "justified deviation (tool_error)" in user_prompt
    assert "alternate endpoint" in user_prompt
    # Not the observed_actions channel.
    assert "OBSERVED AGENT ACTIVITY" not in user_prompt


async def test_refine_justified_deviation_honours_reject_sentinel() -> None:
    """When the LLM judges the justified deviation to actually contradict
    a sticky goal, it returns ``{"reject": true, ...}`` and refine
    returns None (caller escalates via the ladder)."""
    stub = _StubLLM(json.dumps({"reject": True, "reason": "off-goal even with provenance"}))
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.JUSTIFIED_DEVIATION,
        severity=DriftSeverity.WARNING,
        detail=(
            "justified deviation (surprising_result): the source disagrees "
            "with the sticky goal"
        ),
        current_task_id="draft",
    )

    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert revised is None
    # Reject is final — no retry on the goal-aware path.
    assert len(stub.calls) == 1


async def test_refine_repeated_failure_does_not_pull_in_justified_deviation() -> None:
    """Regression: JUSTIFIED_DEVIATION must not pull REPEATED_FAILURE /
    BLOCKED into the goal-aware path.

    The prompt-selection condition expanded in iter-10 PR 4 must NOT
    accidentally light up structural-recovery drifts — they continue
    to use the generic refine prompt.
    """
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
    for kind in (DriftKind.REPEATED_FAILURE, DriftKind.BLOCKED):
        stub = _StubLLM(json.dumps(payload))
        planner = LLMPlanner(call_llm=stub)
        drift = DriftEvent(
            kind=kind,
            severity=DriftSeverity.WARNING,
            detail="recovery",
            current_task_id="draft",
        )
        await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
        system, user_prompt, _model = stub.calls[0]
        assert "ABSORB" not in system, (
            f"{kind.value} accidentally migrated to the goal-aware prompt"
        )
        assert "maintaining an ACTIVE plan" in system
        # Neither reasoning block leaks.
        assert "OFF-TOPIC REASONING" not in user_prompt
        assert "justified deviation" not in user_prompt


async def test_refine_justified_deviation_drift_detail_carries_provenance() -> None:
    """The drift the parser builds from a justified-deviation verdict
    carries the provenance string in ``detail`` — the refine prompt
    renders it verbatim. Pinned end-to-end.
    """
    # Build a JUSTIFIED_DEVIATION drift the way the parser does.
    drift = DriftEvent(
        kind=DriftKind.JUSTIFIED_DEVIATION,
        severity=DriftSeverity.WARNING,
        detail=(
            "justified deviation (discovered_dependency): the writer needs "
            "the editor's style guide first"
        ),
        current_task_id="draft",
        trigger_input="The plan didn't list the style guide as a dep",
    )
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    _system, user_prompt, _model = stub.calls[0]
    assert "justified deviation (discovered_dependency)" in user_prompt
    assert "style guide" in user_prompt
