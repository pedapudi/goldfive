"""Tests for goal-aware ``LLMPlanner.refine`` (goldfive#154).

Scope:

* Every refine kind's prompt includes a ``CURRENT GOALS`` section that
  enumerates ``session.goals`` (dependency on #152 which populates them).
* ``PLAN_DIVERGENCE`` refine with ``observed_actions`` that contradict
  the current goals takes the reject path (``refine()`` returns ``None``
  so the steerer escalates via the intervention ladder, goldfive#142).
* ``PLAN_DIVERGENCE`` refine with ``observed_actions`` that align with
  the goals absorbs normally into a revised plan.
* Successive ``USER_STEER`` refines do NOT silently drop goals added by
  earlier USER_STEERs (sticky-goal contract; the planner validator
  catches a drop and the retry loop re-prompts).
* A missing USER_STEER goal in the refined plan triggers the retry loop
  and is corrected on the second attempt.
* NO COOLDOWN: PLAN_DIVERGENCE refine fired milliseconds after a
  USER_STEER still runs normally and absorbs a goal-aligned
  ``observed_actions`` batch. This is a regression-test-worthy
  constraint per the issue's explicit user directive — steering is
  always active, never time-suppressed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from goldfive.planner import LLMPlanner
from goldfive.types import (
    GOAL_SOURCE_USER_STEER,
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
# Helpers
# ---------------------------------------------------------------------------


def _base_goals() -> list[Goal]:
    """Original session goals (from the user's first message)."""
    return [
        Goal(id="g1", summary="Draft a blog post about goldfish."),
        Goal(id="g2", summary="Get one round of editorial review."),
    ]


def _goals_with_user_steer() -> list[Goal]:
    """Session goals after a USER_STEER ("focus on tropical habitats")."""
    return [
        Goal(id="g1", summary="Draft a blog post about goldfish."),
        Goal(id="g2", summary="Get one round of editorial review."),
        Goal(
            id="steer1",
            summary="Focus the draft on tropical goldfish habitats.",
            source=GOAL_SOURCE_USER_STEER,
        ),
    ]


def _running_plan() -> Plan:
    """Plan with one COMPLETED, one RUNNING, one PENDING task."""
    return Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1", "g2"],
        tasks=[
            Task(
                id="research",
                title="Research goldfish facts",
                description="Gather facts about goldfish.",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft the post",
                description="Write a 500-word draft.",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
            Task(
                id="review",
                title="Review the draft",
                description="Produce reviewer comments.",
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


def _aligned_observed_actions() -> list[ObservedAction]:
    """Tree activity consistent with goals and sticky USER_STEER goal."""
    base = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    return [
        ObservedAction(
            agent_name="researcher",
            invocation_id="inv-1",
            parent_invocation_id="",
            started_at=base,
            completed_at=base + timedelta(seconds=30),
            status="completed",
            summary="Gathered facts about tropical goldfish habitats.",
        ),
        ObservedAction(
            agent_name="writer",
            invocation_id="inv-2",
            parent_invocation_id="",
            started_at=base + timedelta(seconds=31),
            completed_at=None,
            status="running",
            summary="Drafting the tropical-habitat section.",
        ),
    ]


def _contradicting_observed_actions() -> list[ObservedAction]:
    """Tree activity contradicting the USER_STEER goal.

    Operator steered toward tropical habitats; tree is instead writing
    about arctic environments (observable contradiction the LLM should
    catch via the reject sentinel).
    """
    base = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    return [
        ObservedAction(
            agent_name="writer",
            invocation_id="inv-x",
            parent_invocation_id="",
            started_at=base,
            completed_at=base + timedelta(seconds=10),
            status="completed",
            summary=("Drafted section on arctic fish habitats and cold-water survival strategies."),
        ),
    ]


def _absorb_revision_preserving_sticky() -> str:
    """A revision that absorbs observed activity AND preserves the
    USER_STEER goal (tasks reference "tropical")."""
    return json.dumps(
        {
            "summary": "Draft the tropical-habitat-focused goldfish post.",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish facts",
                    "description": "Gather facts about tropical goldfish.",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "description": "Write the tropical-habitat section.",
                    "assignee_agent_id": "writer",
                    "status": "RUNNING",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "description": "Produce reviewer comments on tropical focus.",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft"},
                {"from_task_id": "draft", "to_task_id": "review"},
            ],
        }
    )


def _absorb_revision_dropping_sticky() -> str:
    """A revision that absorbs observed activity BUT silently drops the
    USER_STEER goal (no task references "tropical" / "habitats"/
    "steer1")."""
    return json.dumps(
        {
            "summary": "Draft a generic goldfish post.",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish facts",
                    "description": "Gather generic facts about goldfish.",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "description": "Write a 500-word draft.",
                    "assignee_agent_id": "writer",
                    "status": "RUNNING",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "description": "Produce reviewer comments.",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft"},
                {"from_task_id": "draft", "to_task_id": "review"},
            ],
        }
    )


class _StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        if not self.responses:
            raise AssertionError("scripted LLM called more times than expected")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# 1. Prompt structure: CURRENT GOALS section
# ---------------------------------------------------------------------------


async def test_refine_prompt_includes_goals_section() -> None:
    """Every refine kind's user prompt must include a CURRENT GOALS
    section enumerating session.goals. Without this, goal-aware refine
    decisions are ungrounded."""
    kinds = [
        DriftKind.PLAN_DIVERGENCE,
        DriftKind.TOOL_ERROR,
        DriftKind.AGENT_REFUSAL,
        DriftKind.NEW_WORK_DISCOVERED,
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.USER_STEER,
    ]
    for kind in kinds:
        stub = _StubLLM(_absorb_revision_preserving_sticky())
        planner = LLMPlanner(call_llm=stub)
        drift = DriftEvent(
            kind=kind,
            severity=DriftSeverity.WARNING,
            detail="test drift",
            current_task_id="draft",
        )
        await planner.refine(
            plan=_running_plan(),
            drift=drift,
            goals=_goals_with_user_steer(),
        )
        # At least one call happened; inspect the first user prompt.
        assert stub.calls, f"LLM not called for drift kind {kind.value}"
        _system, user_prompt, _model = stub.calls[0]
        # Every goal is enumerated by id+summary.
        assert "g1" in user_prompt, f"goal g1 missing from prompt for {kind.value}"
        assert "g2" in user_prompt, f"goal g2 missing from prompt for {kind.value}"
        assert "steer1" in user_prompt, f"steer1 goal missing for {kind.value}"
        # The dedicated CURRENT GOALS header is present (capitalised by
        # convention so it's easy to spot in logs).
        assert "CURRENT GOALS" in user_prompt, (
            f"CURRENT GOALS section missing from prompt for {kind.value}"
        )
        # Sticky goals are flagged so the LLM cannot miss them.
        assert "STICKY" in user_prompt, f"STICKY marker missing for USER_STEER goal on {kind.value}"


# ---------------------------------------------------------------------------
# 2. PLAN_DIVERGENCE reject path on goal contradiction
# ---------------------------------------------------------------------------


async def test_plan_divergence_refine_rejects_on_goal_contradiction() -> None:
    """When observed activity contradicts a sticky USER_STEER goal, the
    LLM emits the reject sentinel and refine returns None (the steerer
    escalates to Level 4 via the intervention ladder, #142)."""
    stub = _StubLLM(
        json.dumps(
            {
                "reject": True,
                "reason": (
                    "tree is writing about arctic habitats; USER_STEER "
                    "sticky goal steer1 demands tropical focus"
                ),
            }
        )
    )
    planner = LLMPlanner(call_llm=stub)

    emitted: list[DriftEvent] = []

    async def capture(d: DriftEvent) -> None:
        emitted.append(d)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree wandered off-steer",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),
        observed_actions=_contradicting_observed_actions(),
    )
    assert revised is None
    # Reject is a successful decision, not a validation failure -- no
    # REFINE_VALIDATION_FAILED drift emitted.
    assert emitted == []
    # Exactly one LLM call; no retry on a valid reject.
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# 3. PLAN_DIVERGENCE absorb path on goal alignment
# ---------------------------------------------------------------------------


async def test_plan_divergence_refine_absorbs_on_goal_alignment() -> None:
    """When observed activity aligns with goals (including sticky
    USER_STEER goals), the LLM absorbs it into a revised plan and
    refine() returns the revision with updated revision metadata."""
    stub = _StubLLM(_absorb_revision_preserving_sticky())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree executing extra invocations",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),
        observed_actions=_aligned_observed_actions(),
    )
    assert revised is not None
    assert revised.revision_index == 1
    assert revised.revision_kind == DriftKind.PLAN_DIVERGENCE.value
    # Sticky goal is addressed: at least one task description mentions
    # "tropical" (the operator's steer keyword).
    plan_text = " ".join(f"{t.title} {t.description}".lower() for t in revised.tasks)
    assert "tropical" in plan_text


# ---------------------------------------------------------------------------
# 4. USER_STEER refine preserves prior USER_STEER goals
# ---------------------------------------------------------------------------


async def test_user_steer_refine_preserves_prior_user_steer_goals() -> None:
    """A second USER_STEER ("add a code example") must not silently drop
    the first steer's sticky goal ("focus on tropical habitats"). The
    validator checks each prior USER_STEER goal is still referenced."""
    # First steer has already been applied and recorded as a sticky goal
    # on session.goals. A new steer fires now, adding new pending work.
    goals_after_first_steer = _goals_with_user_steer()

    # The LLM emits a plan that preserves "tropical" (prior steer) and
    # adds a "code_example" task (the new steer). Sticky check passes.
    good_response = json.dumps(
        {
            "summary": "Add a code example to the tropical-habitat post.",
            "tasks": [
                {
                    "id": "code_example",
                    "title": "Add a code example about tropical goldfish tank setup",
                    "description": (
                        "Include a runnable snippet showing tropical tank temperature control."
                    ),
                    "assignee_agent_id": "writer",
                },
                {
                    "id": "review_new",
                    "title": "Review the new section",
                    "description": "Verify the code example and tropical framing.",
                    "assignee_agent_id": "editor",
                },
            ],
            "edges": [
                {"from_task_id": "code_example", "to_task_id": "review_new"},
            ],
        }
    )
    stub = _StubLLM(good_response)
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="also add a code example",
        current_task_id="draft",
    )
    # Start from a plan where draft is COMPLETED so the USER_STEER
    # merge path has something to preserve.
    # goldfive#247: Plan is frozen — derive via with_task_status.
    from goldfive.types import with_task_status as _wts

    plan = _running_plan()
    plan = _wts(plan, plan.tasks[1].id, TaskStatus.COMPLETED)  # draft done
    plan = _wts(plan, plan.tasks[2].id, TaskStatus.COMPLETED)  # review done

    revised = await planner.refine(
        plan=plan,
        drift=drift,
        goals=goals_after_first_steer,
    )
    assert revised is not None
    # New PENDING tasks reference the prior steer (tropical).
    new_pending_text = " ".join(
        f"{t.title} {t.description}".lower()
        for t in revised.tasks
        if t.status is TaskStatus.PENDING
    )
    assert "tropical" in new_pending_text
    # And the new steer's work is there.
    assert "code" in new_pending_text or "example" in new_pending_text


# ---------------------------------------------------------------------------
# 5. Silently-dropped USER_STEER goal triggers retry loop
# ---------------------------------------------------------------------------


async def test_refine_output_missing_user_steer_goal_triggers_retry() -> None:
    """If the LLM's first revision silently drops the sticky USER_STEER
    goal, the validator feeds a correction message into the prompt and
    the second attempt (which honours the sticky goal) succeeds."""
    scripted = _ScriptedLLM(
        [
            _absorb_revision_dropping_sticky(),  # no reference to tropical
            _absorb_revision_preserving_sticky(),  # tasks mention tropical
        ]
    )
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree diverged",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),
        observed_actions=_aligned_observed_actions(),
    )
    assert revised is not None
    assert len(scripted.calls) == 2
    _sys1, _first_user, _m1 = scripted.calls[0]
    _sys2, second_user, _m2 = scripted.calls[1]
    # The retry prompt tells the LLM exactly what it missed.
    assert "PREVIOUS ATTEMPT FAILED" in second_user
    # And diagnoses the sticky-goal drop specifically.
    assert "USER_STEER" in second_user or "sticky" in second_user.lower()
    # Second-attempt plan references tropical.
    plan_text = " ".join(f"{t.title} {t.description}".lower() for t in revised.tasks)
    assert "tropical" in plan_text


async def test_refine_exhausted_retries_on_sticky_goal_drop_emits_validation_drift() -> None:
    """If the LLM NEVER respects the sticky goal, retries exhaust and a
    REFINE_VALIDATION_FAILED drift is emitted (so the ladder can
    escalate to human intervention). Refine returns None."""
    scripted = _ScriptedLLM(
        [
            _absorb_revision_dropping_sticky(),
            _absorb_revision_dropping_sticky(),
        ]
    )
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    emitted: list[DriftEvent] = []

    async def capture(d: DriftEvent) -> None:
        emitted.append(d)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="diverged",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),
        observed_actions=_aligned_observed_actions(),
    )
    assert revised is None
    assert len(emitted) == 1
    assert emitted[0].kind is DriftKind.REFINE_VALIDATION_FAILED


# ---------------------------------------------------------------------------
# 6. NO cooldown: refine fires immediately post-USER_STEER (regression)
# ---------------------------------------------------------------------------


async def test_no_cooldown_refine_fires_immediately_post_user_steer() -> None:
    """Explicit regression test for the "NO cooldown" user directive
    (goldfive#154). A PLAN_DIVERGENCE refine triggered one second after
    a USER_STEER must run normally — no time-windowing, no suppression.

    The durability mechanism is goal-aware refine rejecting
    contradictions, not a time-based mute. Steering should always be
    active."""
    # Scenario: USER_STEER was applied at t0. One second later the
    # reconciler flags PLAN_DIVERGENCE with aligned observed activity.
    # The refine must absorb normally -- no "too soon after a steer"
    # suppression.
    stub = _StubLLM(_absorb_revision_preserving_sticky())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="tree executing extra invocations (1s after USER_STEER)",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),  # sticky USER_STEER goal present
        observed_actions=_aligned_observed_actions(),
    )
    # Refine ran (one LLM call), returned a revised plan, not None.
    assert revised is not None
    assert revised.revision_index == 1
    assert len(stub.calls) == 1
    # Regression sentinel: no attribute on the planner that would let a
    # future maintainer reintroduce cooldown by accident.
    assert not hasattr(planner, "_steer_refine_cooldown_seconds")
    assert not hasattr(planner, "_last_user_steer_at")
    assert not hasattr(planner, "STEER_REFINE_COOLDOWN_SECONDS")


# ---------------------------------------------------------------------------
# 7. No sticky goals -> legacy behaviour unchanged
# ---------------------------------------------------------------------------


async def test_refine_without_sticky_goals_behaves_as_before() -> None:
    """When there are no USER_STEER goals, sticky-goal checks are
    inactive and the legacy PLAN_DIVERGENCE refine behaviour holds.
    Also the STICKY GOALS section must NOT appear in the prompt --
    legacy callers see the exact same prompt shape."""
    stub = _StubLLM(_absorb_revision_preserving_sticky())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="divergence",
        current_task_id="draft",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_base_goals(),
        observed_actions=_aligned_observed_actions(),
    )
    assert revised is not None
    # Legacy callers must not see the STICKY GOALS block.
    _system, user_prompt, _model = stub.calls[0]
    assert "STICKY GOALS" not in user_prompt
    # But the CURRENT GOALS header is still present on goal-aware
    # refine prompts (goldfive#154 lands that for everyone).
    assert "CURRENT GOALS" in user_prompt


# ---------------------------------------------------------------------------
# 8. Goal dataclass wiring
# ---------------------------------------------------------------------------


def test_goal_source_defaults_empty() -> None:
    """Back-compat: ``Goal`` without an explicit source defaults to ``""``."""
    g = Goal(id="g", summary="s")
    assert g.source == ""


def test_goal_source_user_steer_marker() -> None:
    """The ``GOAL_SOURCE_USER_STEER`` marker is the documented value
    for goals added by a USER_STEER directive."""
    g = Goal(id="g", summary="s", source=GOAL_SOURCE_USER_STEER)
    assert g.source == "USER_STEER"


# ---------------------------------------------------------------------------
# 9. Reject sentinel honoured for non-PLAN_DIVERGENCE refine when sticky
# ---------------------------------------------------------------------------


async def test_non_divergence_refine_honours_reject_sentinel_when_sticky() -> None:
    """When a sticky USER_STEER goal is present, even non-divergence
    refine kinds (e.g. TOOL_ERROR) can escalate via the reject sentinel
    if the LLM determines the drift irreconcilably contradicts the
    operator's steer. This is the extended reject path from #154."""
    stub = _StubLLM(
        json.dumps(
            {
                "reject": True,
                "reason": "tool error forces cold-water pivot; contradicts steer",
            }
        )
    )
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool timeout; fallback path would abandon tropical focus",
        current_task_id="research",
    )
    revised = await planner.refine(
        plan=_running_plan(),
        drift=drift,
        goals=_goals_with_user_steer(),
    )
    assert revised is None  # reject -> escalate
    # Exactly one call -- reject is terminal.
    assert len(stub.calls) == 1


async def test_non_divergence_refine_ignores_reject_sentinel_without_sticky() -> None:
    """Back-compat: without sticky goals, non-divergence refine does NOT
    honour the reject sentinel (preserves legacy behaviour — the reject
    path was originally only for PLAN_DIVERGENCE + observed_actions).
    A reject-shaped JSON is treated as a malformed plan."""
    stub = _StubLLM(json.dumps({"reject": True, "reason": "x"}))
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=1)
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool error",
        current_task_id="research",
    )
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_base_goals())
    assert revised is None  # but via "no usable plan" exhaustion, not reject
