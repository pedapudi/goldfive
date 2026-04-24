"""Unit tests for ``goldfive.planner``.

Covers both ``PassthroughPlanner`` (trivial) and ``LLMPlanner`` (the
interesting case: stubbed async ``call_llm`` returning canned JSON,
markdown-fence stripping, refinement preserving completed tasks, and
error paths that must degrade gracefully to ``None``).
"""

from __future__ import annotations

import json

import pytest

from goldfive.planner import (
    _DEFAULT_SYSTEM_PROMPT,
    LLMPlanner,
    PassthroughPlanner,
    _check_supersedes_coverage,
    _normalize_assignee,
    _plan_from_json,
    _strip_code_fences,
)
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    SupersessionKind,
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


def _canned_plan_json() -> str:
    return json.dumps(
        {
            "summary": "Draft and review a goldfish blog post.",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish facts",
                    "description": "Gather 5 cited facts.",
                    "assignee_agent_id": "researcher",
                },
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "description": "Write a 500-word first draft.",
                    "assignee_agent_id": "writer",
                },
                {
                    "id": "review",
                    "title": "Review the draft",
                    "description": "Produce reviewer comments.",
                    "assignee_agent_id": "editor",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft"},
                {"from_task_id": "draft", "to_task_id": "review"},
            ],
        }
    )


class _StubLLM:
    """Records the last call and returns a scripted response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response


# ---------------------------------------------------------------------------
# PassthroughPlanner
# ---------------------------------------------------------------------------


async def test_passthrough_generate_returns_none() -> None:
    planner = PassthroughPlanner()
    result = await planner.generate(
        goals=_goals(),
        available_agents=["a", "b"],
        context=None,
    )
    assert result is None


async def test_passthrough_refine_returns_none() -> None:
    planner = PassthroughPlanner()
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Do the thing")],
        edges=[],
    )
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=plan, drift=drift, goals=_goals())
    assert result is None


# ---------------------------------------------------------------------------
# _strip_code_fences
# ---------------------------------------------------------------------------


def test_strip_code_fences_json_tag() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(raw).strip() == '{"a": 1}'


def test_strip_code_fences_plain_fence() -> None:
    raw = '```\n{"a": 1}\n```'
    assert _strip_code_fences(raw).strip() == '{"a": 1}'


def test_strip_code_fences_no_fence() -> None:
    raw = '{"a": 1}'
    assert _strip_code_fences(raw) == raw


def test_strip_code_fences_empty() -> None:
    assert _strip_code_fences("") == ""


# ---------------------------------------------------------------------------
# _plan_from_json
# ---------------------------------------------------------------------------


def test_plan_from_json_happy_path() -> None:
    plan = _plan_from_json(
        json.loads(_canned_plan_json()),
        run_id="run-abc",
        goal_ids=["g1", "g2"],
    )
    assert plan is not None
    assert plan.run_id == "run-abc"
    assert plan.goal_ids == ["g1", "g2"]
    assert [t.id for t in plan.tasks] == ["research", "draft", "review"]
    assert all(t.status == TaskStatus.PENDING for t in plan.tasks)
    assert plan.summary == "Draft and review a goldfish blog post."
    assert len(plan.edges) == 2


def test_plan_from_json_non_mapping_returns_none() -> None:
    assert _plan_from_json("not a mapping", run_id="", goal_ids=[]) is None
    assert _plan_from_json(None, run_id="", goal_ids=[]) is None


def test_plan_from_json_empty_tasks_returns_none() -> None:
    assert _plan_from_json({"tasks": []}, run_id="", goal_ids=[]) is None


def test_plan_from_json_skips_malformed_tasks() -> None:
    payload = {
        "tasks": [
            {"id": "", "title": "no id"},  # dropped
            {"id": "t1", "title": ""},  # dropped
            "not a mapping",  # dropped
            {"id": "t2", "title": "ok"},
        ]
    }
    plan = _plan_from_json(payload, run_id="r", goal_ids=[])
    assert plan is not None
    assert [t.id for t in plan.tasks] == ["t2"]


def test_plan_from_json_invalid_status_falls_back_to_pending() -> None:
    payload = {
        "tasks": [
            {"id": "t1", "title": "ok", "status": "BOGUS"},
        ]
    }
    plan = _plan_from_json(payload, run_id="r", goal_ids=[])
    assert plan is not None
    assert plan.tasks[0].status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# LLMPlanner.generate
# ---------------------------------------------------------------------------


async def test_llm_planner_generate_parses_canned_json() -> None:
    stub = _StubLLM(_canned_plan_json())
    planner = LLMPlanner(call_llm=stub, model="test-model")
    plan = await planner.generate(
        goals=_goals(),
        available_agents=["researcher", "writer", "editor"],
        context={"run_id": "run-xyz"},
    )
    assert plan is not None
    assert plan.run_id == "run-xyz"
    assert plan.goal_ids == ["g1", "g2"]
    assert [t.id for t in plan.tasks] == ["research", "draft", "review"]

    # Sanity: the prompt the LLM saw mentioned the goals and agents.
    assert stub.calls, "stub should have been invoked"
    _system, user, model = stub.calls[0]
    assert model == "test-model"
    assert "g1" in user and "g2" in user
    assert "Draft a blog post about goldfish." in user
    assert "researcher" in user


async def test_llm_planner_generate_strips_code_fences() -> None:
    fenced = "```json\n" + _canned_plan_json() + "\n```"
    stub = _StubLLM(fenced)
    planner = LLMPlanner(call_llm=stub)
    plan = await planner.generate(
        goals=_goals(),
        available_agents=["researcher", "writer", "editor"],
    )
    assert plan is not None
    assert len(plan.tasks) == 3


async def test_llm_planner_generate_empty_goals_returns_none() -> None:
    stub = _StubLLM(_canned_plan_json())
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=[], available_agents=["a"])
    assert result is None
    # LLM must not be invoked when there are no goals.
    assert stub.calls == []


async def test_llm_planner_generate_handles_invalid_json() -> None:
    stub = _StubLLM("not json at all {")
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["a"])
    assert result is None


async def test_llm_planner_generate_handles_empty_response() -> None:
    stub = _StubLLM("")
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["a"])
    assert result is None


async def test_llm_planner_generate_handles_call_llm_exception() -> None:
    async def boom(system: str, user: str, model: str) -> str:
        raise RuntimeError("LLM unavailable")

    planner = LLMPlanner(call_llm=boom)
    result = await planner.generate(goals=_goals(), available_agents=["a"])
    assert result is None


async def test_llm_planner_generate_handles_json_without_tasks() -> None:
    stub = _StubLLM(json.dumps({"summary": "nothing to do"}))
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["a"])
    assert result is None


# ---------------------------------------------------------------------------
# LLMPlanner.generate — tree-aware registry constraints (goldfive#151)
# ---------------------------------------------------------------------------


def _tree_registry() -> list[dict[str, object]]:
    """A structured tree matching the canned plan's assignees."""
    return [
        {"name": "coordinator", "depth": 0, "parent": "", "role": "root", "kind": "LlmAgent"},
        {
            "name": "researcher",
            "depth": 1,
            "parent": "coordinator",
            "role": "leaf",
            "kind": "LlmAgent",
        },
        {
            "name": "writer",
            "depth": 1,
            "parent": "coordinator",
            "role": "leaf",
            "kind": "LlmAgent",
        },
        {
            "name": "editor",
            "depth": 1,
            "parent": "coordinator",
            "role": "leaf",
            "kind": "LlmAgent",
        },
    ]


def _off_registry_plan_json() -> str:
    """A plan whose assignees don't match the registry names."""
    return json.dumps(
        {
            "summary": "Off-registry plan.",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research",
                    "description": "Do research.",
                    "assignee_agent_id": "agent_researcher",
                },
                {
                    "id": "draft",
                    "title": "Draft",
                    "description": "Draft post.",
                    "assignee_agent_id": "agent_content",
                },
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "draft"}],
        }
    )


async def test_generate_prompt_renders_agent_tree_when_tree_given() -> None:
    """Passing the structured tree form renders an AGENT TREE section."""
    stub = _StubLLM(_canned_plan_json())
    planner = LLMPlanner(call_llm=stub)
    await planner.generate(
        goals=_goals(),
        available_agents=_tree_registry(),
    )
    assert stub.calls, "stub should have been invoked"
    _sys, user, _model = stub.calls[0]
    assert "AGENT TREE" in user
    # Tree metadata surfaces (role / kind / depth).
    assert "role=root" in user
    assert "role=leaf" in user
    assert "kind=LlmAgent" in user


async def test_generate_prompt_flat_list_renders_legacy_header() -> None:
    """Legacy ``list[str]`` callers still see the old 'Available agents:' header."""
    stub = _StubLLM(_canned_plan_json())
    planner = LLMPlanner(call_llm=stub)
    await planner.generate(
        goals=_goals(),
        available_agents=["researcher", "writer", "editor"],
    )
    _sys, user, _model = stub.calls[0]
    assert "Available agents:" in user
    assert "AGENT TREE" not in user


async def test_validator_rejects_off_registry_assignee() -> None:
    """Plans whose assignees aren't in the registry are rejected on every attempt.

    Exhausts the retry budget because the stub always returns the same
    off-registry JSON; ``generate`` eventually returns ``None``.
    """

    class _AlwaysOffRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def __call__(self, system: str, user: str, model: str) -> str:
            self.calls.append((system, user, model))
            return _off_registry_plan_json()

    stub = _AlwaysOffRegistry()
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=2)
    result = await planner.generate(
        goals=_goals(),
        available_agents=_tree_registry(),
    )
    assert result is None
    # The retry loop fired the full budget.
    assert len(stub.calls) == 2
    # The retry prompt fed the validator error back to the LLM.
    _sys, retry_user, _ = stub.calls[1]
    assert "off-registry assignee" in retry_user


async def test_retry_loop_corrects_off_registry_assignee() -> None:
    """First attempt uses bogus assignees; second attempt picks the real names.

    Demonstrates the #133/#136 retry-with-correction loop wired into
    the #151 registry check: the validator's error message feeds the
    second prompt, and the second response validates cleanly.
    """

    class _Correcting:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def __call__(self, system: str, user: str, model: str) -> str:
            self.calls.append((system, user, model))
            if len(self.calls) == 1:
                return _off_registry_plan_json()
            return _canned_plan_json()

    stub = _Correcting()
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=3)
    plan = await planner.generate(
        goals=_goals(),
        available_agents=_tree_registry(),
    )
    assert plan is not None
    # Second attempt used the real registry names.
    assignees = {t.assignee_agent_id for t in plan.tasks}
    assert assignees <= {"researcher", "writer", "editor"}
    # Correction feedback landed in the retry prompt.
    _sys, retry_user, _ = stub.calls[1]
    assert "off-registry assignee" in retry_user


async def test_generate_accepts_none_registry_backcompat() -> None:
    """``available_agents=None`` skips the registry check entirely.

    Preserves back-compat with callers that don't supply a registry —
    the plan validates structurally but no assignee-membership check
    fires.
    """
    stub = _StubLLM(_off_registry_plan_json())
    planner = LLMPlanner(call_llm=stub, max_refine_attempts=2)
    plan = await planner.generate(
        goals=_goals(),
        available_agents=None,
    )
    # With no registry, off-registry names are allowed.
    assert plan is not None
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# LLMPlanner.refine
# ---------------------------------------------------------------------------


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


async def test_llm_planner_refine_preserves_completed_tasks() -> None:
    # The LLM returns a plan that keeps the COMPLETED research task,
    # marks draft COMPLETED, leaves review PENDING, and injects a new
    # fact-check task between draft and review.
    revised_payload = {
        "summary": "Draft, fact-check, and review the goldfish post.",
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
                "id": "fact_check",
                "title": "Fact-check the draft",
                "assignee_agent_id": "researcher",
                "status": "PENDING",
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
            {"from_task_id": "draft", "to_task_id": "fact_check"},
            {"from_task_id": "fact_check", "to_task_id": "review"},
        ],
    }
    stub = _StubLLM(json.dumps(revised_payload))
    planner = LLMPlanner(call_llm=stub, model="m")
    plan = _running_plan()
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="Reviewer asked for citations before review",
        current_task_id="draft",
    )

    revised = await planner.refine(plan=plan, drift=drift, goals=_goals())

    assert revised is not None
    by_id = {t.id: t for t in revised.tasks}
    # Historical tasks preserved.
    assert by_id["research"].status == TaskStatus.COMPLETED
    assert by_id["draft"].status == TaskStatus.COMPLETED
    # New task added, pending task preserved.
    assert "fact_check" in by_id
    assert by_id["review"].status == TaskStatus.PENDING

    # Plan envelope identity is preserved across refinement.
    assert revised.id == plan.id
    assert revised.run_id == plan.run_id
    assert revised.goal_ids == ["g1", "g2"]

    # Revision metadata stamped from the drift.
    assert revised.revision_reason == "Reviewer asked for citations before review"
    assert revised.revision_kind == DriftKind.NEW_WORK_DISCOVERED.value
    assert revised.revision_severity == DriftSeverity.INFO.value
    assert revised.revision_index == 1

    # The prompt included goals, the current plan JSON, and the drift.
    # The goals section header was renamed from "Goals:" to
    # "CURRENT GOALS" by goldfive#154 so the planner-LLM treats the
    # section as operator-authored rather than informational.
    _system, user_prompt, _model = stub.calls[0]
    assert "CURRENT GOALS" in user_prompt
    assert "Current plan:" in user_prompt
    assert "Drift event:" in user_prompt
    assert "new_work_discovered" in user_prompt
    assert "research" in user_prompt and "draft" in user_prompt


async def test_llm_planner_refine_strips_fences() -> None:
    payload = {
        "summary": "same",
        "tasks": [
            {"id": "research", "title": "Research", "status": "COMPLETED"},
            {"id": "draft", "title": "Draft", "status": "RUNNING"},
            {"id": "review", "title": "Review", "status": "PENDING"},
        ],
        "edges": [],
    }
    stub = _StubLLM("```json\n" + json.dumps(payload) + "\n```")
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert revised is not None
    assert len(revised.tasks) == 3


async def test_llm_planner_refine_handles_bad_json() -> None:
    stub = _StubLLM("not json")
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert revised is None


async def test_llm_planner_refine_handles_call_llm_exception() -> None:
    async def boom(system: str, user: str, model: str) -> str:
        raise RuntimeError("boom")

    planner = LLMPlanner(call_llm=boom)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert revised is None


async def test_llm_planner_refine_increments_revision_index() -> None:
    # Must preserve the COMPLETED ``research`` task from ``_running_plan``
    # verbatim so PLAN-LIFECYCLE.md §3.1 terminal-preservation holds at
    # validation time.
    payload = {
        "summary": "same",
        "tasks": [
            {
                "id": "research",
                "title": "Research goldfish facts",
                "assignee_agent_id": "researcher",
                "status": "COMPLETED",
            },
            {"id": "draft", "title": "Draft", "status": "RUNNING"},
        ],
        "edges": [],
    }
    stub = _StubLLM(json.dumps(payload))
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)

    plan = _running_plan()
    plan.revision_index = 3
    revised = await planner.refine(plan=plan, drift=drift, goals=_goals())
    assert revised is not None
    assert revised.revision_index == 4


async def test_llm_planner_refine_custom_system_prompt_is_used() -> None:
    stub = _StubLLM(json.dumps({"tasks": [{"id": "t", "title": "t"}], "edges": []}))
    planner = LLMPlanner(
        call_llm=stub,
        system_prompt="GEN PROMPT SENTINEL",
        refine_system_prompt="REFINE PROMPT SENTINEL",
    )
    await planner.generate(goals=_goals(), available_agents=["a"])
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert stub.calls[0][0] == "GEN PROMPT SENTINEL"
    assert stub.calls[1][0] == "REFINE PROMPT SENTINEL"


# ---------------------------------------------------------------------------
# Plan validation at creation / revision (TASK-LIFECYCLE.md §7.2)
# ---------------------------------------------------------------------------


async def test_llm_planner_generate_rejects_duplicate_task_ids() -> None:
    """A plan with duplicate task ids is a soundness violation.

    ``Plan.validate()`` flags duplicate ids; the planner must return
    ``None`` so the caller treats the turn as "no plan" rather than
    silently installing a malformed DAG.
    """
    duplicate = json.dumps(
        {
            "summary": "dup",
            "tasks": [
                {"id": "same", "title": "first", "assignee_agent_id": "a"},
                {"id": "same", "title": "second", "assignee_agent_id": "a"},
            ],
            "edges": [],
        }
    )
    stub = _StubLLM(duplicate)
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["a"])
    assert result is None


async def test_llm_planner_generate_rejects_cycle() -> None:
    cyclic = json.dumps(
        {
            "summary": "cyclic",
            "tasks": [
                {"id": "a", "title": "A", "assignee_agent_id": "x"},
                {"id": "b", "title": "B", "assignee_agent_id": "x"},
            ],
            "edges": [
                {"from_task_id": "a", "to_task_id": "b"},
                {"from_task_id": "b", "to_task_id": "a"},
            ],
        }
    )
    stub = _StubLLM(cyclic)
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["x"])
    assert result is None


async def test_llm_planner_generate_rejects_unknown_edge() -> None:
    bad = json.dumps(
        {
            "summary": "bad edge",
            "tasks": [{"id": "t1", "title": "T1", "assignee_agent_id": "x"}],
            "edges": [{"from_task_id": "t1", "to_task_id": "ghost"}],
        }
    )
    stub = _StubLLM(bad)
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["x"])
    assert result is None


async def test_llm_planner_generate_rejects_non_pending_task_at_creation() -> None:
    # Newly generated plan must not carry COMPLETED/FAILED/... tasks.
    non_pending = json.dumps(
        {
            "summary": "ok",
            "tasks": [
                {
                    "id": "t1",
                    "title": "T1",
                    "assignee_agent_id": "x",
                    "status": "COMPLETED",
                }
            ],
            "edges": [],
        }
    )
    stub = _StubLLM(non_pending)
    planner = LLMPlanner(call_llm=stub)
    result = await planner.generate(goals=_goals(), available_agents=["x"])
    assert result is None


async def test_llm_planner_refine_allows_preserved_completed_tasks() -> None:
    """Refine is the revision path — COMPLETED tasks are legitimate."""
    refined = json.dumps(
        {
            "summary": "after drift",
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
                    "status": "PENDING",
                },
            ],
            "edges": [{"from_task_id": "research", "to_task_id": "draft"}],
        }
    )
    stub = _StubLLM(refined)
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is not None
    assert {t.id for t in result.tasks} == {"research", "draft"}


async def test_llm_planner_refine_rejects_duplicate_ids() -> None:
    bad = json.dumps(
        {
            "summary": "dup",
            "tasks": [
                {"id": "d", "title": "1", "assignee_agent_id": "x"},
                {"id": "d", "title": "2", "assignee_agent_id": "x"},
            ],
            "edges": [],
        }
    )
    stub = _StubLLM(bad)
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None


async def test_llm_planner_refine_rejects_cycle() -> None:
    bad = json.dumps(
        {
            "summary": "cyclic",
            "tasks": [
                {"id": "a", "title": "A", "assignee_agent_id": "x"},
                {"id": "b", "title": "B", "assignee_agent_id": "x"},
            ],
            "edges": [
                {"from_task_id": "a", "to_task_id": "b"},
                {"from_task_id": "b", "to_task_id": "a"},
            ],
        }
    )
    stub = _StubLLM(bad)
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=DriftSeverity.WARNING)
    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())
    assert result is None


# ---------------------------------------------------------------------------
# Retry-on-validation-failure + REFINE_VALIDATION_FAILED signal (issue #133)
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """LLM stub that returns scripted responses in order, one per call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        if not self.responses:
            raise AssertionError("scripted LLM called more times than expected")
        return self.responses.pop(0)


def _looping_plan() -> Plan:
    """A plan with a completed task, a completed task, and a looping task.

    The terminal-task and terminal->terminal-edge preservation invariants
    both apply here: `research` -> `structure` is a terminal->terminal
    edge that any revision must preserve verbatim.
    """
    return Plan(
        id="plan-loop",
        run_id="run-1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_installation",
                title="Research installation",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="structure_presentation",
                title="Structure presentation",
                assignee_agent_id="writer",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft_slides",
                title="Draft slides",
                assignee_agent_id="writer",
                status=TaskStatus.RUNNING,
            ),
        ],
        edges=[
            TaskEdge(
                from_task_id="research_installation",
                to_task_id="structure_presentation",
            ),
            TaskEdge(
                from_task_id="structure_presentation",
                to_task_id="draft_slides",
            ),
        ],
        summary="Build the installation presentation.",
        revision_index=0,
    )


def _bad_looping_revision_json() -> str:
    """A revision that drops the required terminal->terminal edge.

    This mirrors the exact shape that trips the validator in the e2e
    run that motivated goldfive#133 -- the LLM keeps both terminal
    tasks but forgets the edge connecting them.
    """
    return json.dumps(
        {
            "summary": "fail the looper and try again",
            "tasks": [
                {
                    "id": "research_installation",
                    "title": "Research installation",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "structure_presentation",
                    "title": "Structure presentation",
                    "assignee_agent_id": "writer",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_slides",
                    "title": "Draft slides",
                    "assignee_agent_id": "writer",
                    "status": "FAILED",
                },
                {
                    "id": "draft_slides_alt",
                    "title": "Draft slides with outline",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            # Missing research_installation -> structure_presentation edge.
            "edges": [
                {
                    "from_task_id": "structure_presentation",
                    "to_task_id": "draft_slides_alt",
                },
            ],
        }
    )


def _good_looping_revision_json() -> str:
    """A revision that preserves the required terminal->terminal edge."""
    return json.dumps(
        {
            "summary": "fail the looper and try again (fixed)",
            "tasks": [
                {
                    "id": "research_installation",
                    "title": "Research installation",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "structure_presentation",
                    "title": "Structure presentation",
                    "assignee_agent_id": "writer",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_slides",
                    "title": "Draft slides",
                    "assignee_agent_id": "writer",
                    "status": "FAILED",
                },
                {
                    "id": "draft_slides_alt",
                    "title": "Draft slides with outline",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {
                    "from_task_id": "research_installation",
                    "to_task_id": "structure_presentation",
                },
                {
                    "from_task_id": "structure_presentation",
                    "to_task_id": "draft_slides_alt",
                },
            ],
        }
    )


async def test_refine_retries_on_validation_failure_and_succeeds() -> None:
    """Attempt 1 emits a plan missing a terminal->terminal edge; attempt 2
    emits a valid plan. The planner must retry and succeed.
    """
    scripted = _ScriptedLLM([_bad_looping_revision_json(), _good_looping_revision_json()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="agent stuck calling read_file",
        current_task_id="draft_slides",
    )

    revised = await planner.refine(plan=_looping_plan(), drift=drift, goals=_goals())

    assert revised is not None
    assert len(scripted.calls) == 2
    # Second call must include the correction context so the LLM knows
    # what to fix.
    _system, second_user_prompt, _model = scripted.calls[1]
    assert "PREVIOUS ATTEMPT FAILED" in second_user_prompt
    assert "terminal->terminal edge" in second_user_prompt
    # The final plan contains the terminal->terminal edge AND the new
    # PENDING task.
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["draft_slides"].status == TaskStatus.FAILED
    assert "draft_slides_alt" in by_id
    pairs = {(e.from_task_id, e.to_task_id) for e in revised.edges}
    assert (
        "research_installation",
        "structure_presentation",
    ) in pairs


async def test_refine_exhausts_retries_and_emits_drift() -> None:
    """When every attempt fails validation, the planner must:
    1. emit a REFINE_VALIDATION_FAILED drift via the configured emitter
    2. return the deterministic fail-the-loop fallback (non-None) so
       the executor stops burning calls on the looper.
    """
    scripted = _ScriptedLLM([_bad_looping_revision_json(), _bad_looping_revision_json()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)

    emitted: list[DriftEvent] = []

    async def capture(drift: DriftEvent) -> None:
        emitted.append(drift)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="agent stuck calling read_file",
        current_task_id="draft_slides",
    )

    result = await planner.refine(plan=_looping_plan(), drift=drift, goals=_goals())

    # Exactly one REFINE_VALIDATION_FAILED drift was emitted, at CRITICAL.
    assert len(emitted) == 1
    assert emitted[0].kind is DriftKind.REFINE_VALIDATION_FAILED
    assert emitted[0].severity is DriftSeverity.CRITICAL
    assert "terminal->terminal edge" in emitted[0].detail
    # Fallback plan: looper is FAILED and the other tasks / edges remain.
    assert result is not None
    by_id = {t.id: t for t in result.tasks}
    assert by_id["draft_slides"].status == TaskStatus.FAILED
    assert by_id["research_installation"].status == TaskStatus.COMPLETED
    assert len(scripted.calls) == 2  # retry budget exhausted exactly


async def test_refine_exhausts_retries_and_returns_none_for_generic_path() -> None:
    """The generic refine path (non-LOOPING, non-USER_STEER) has no
    deterministic fallback; it returns ``None`` and emits the
    REFINE_VALIDATION_FAILED signal. The steerer then uses its backoff
    counter.
    """
    # Craft a plan with a completed task and a running one, and have
    # the LLM emit a revision that drops the completed task (violating
    # terminal-task preservation).
    bad = json.dumps(
        {
            "summary": "dropped history",
            "tasks": [
                # research (COMPLETED in prior plan) is missing -- reject.
                {
                    "id": "draft",
                    "title": "Draft the post",
                    "status": "RUNNING",
                },
            ],
            "edges": [],
        }
    )
    scripted = _ScriptedLLM([bad, bad])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)

    emitted: list[DriftEvent] = []

    async def capture(drift: DriftEvent) -> None:
        emitted.append(drift)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="tool x failed",
        current_task_id="draft",
    )

    result = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert result is None
    assert len(emitted) == 1
    assert emitted[0].kind is DriftKind.REFINE_VALIDATION_FAILED
    assert emitted[0].severity is DriftSeverity.CRITICAL


async def test_refine_prompt_enumerates_terminal_tasks_and_edges() -> None:
    """The LOOPING_TOOL_CALL refine prompt must enumerate the terminal
    tasks AND the terminal->terminal edges the revision must preserve
    verbatim -- this is the "teach the LLM about invariants" half of
    the goldfive#133 fix.
    """
    scripted = _ScriptedLLM([_good_looping_revision_json()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="stuck",
        current_task_id="draft_slides",
    )

    await planner.refine(plan=_looping_plan(), drift=drift, goals=_goals())

    _system, user_prompt, _model = scripted.calls[0]
    assert "STRUCTURAL INVARIANTS" in user_prompt
    # Both terminal task ids appear in the enumerated block.
    assert "research_installation" in user_prompt
    assert "structure_presentation" in user_prompt
    # The required terminal->terminal edge appears as a copy-paste-ready
    # JSON object; we assert on the key phrase and on the edge endpoints
    # appearing in the required-edges list.
    assert "terminal->terminal" in user_prompt.lower() or "TERMINAL->TERMINAL" in user_prompt
    assert "Required edges" in user_prompt
    # The required-edge JSON contains the endpoints in the right slot.
    # (We don't demand exact string formatting so that the edge list can
    # evolve; we just demand the key tokens appear after the "Required
    # edges" label.)
    required_idx = user_prompt.index("Required edges")
    after = user_prompt[required_idx:]
    assert "research_installation" in after
    assert "structure_presentation" in after


async def test_refine_first_try_success_unchanged() -> None:
    """When the LLM emits a valid plan on the first attempt, exactly one
    call is made -- retries and drift emission stay dormant.
    """
    scripted = _ScriptedLLM([_good_looping_revision_json()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)

    emitted: list[DriftEvent] = []

    async def capture(drift: DriftEvent) -> None:
        emitted.append(drift)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="stuck",
        current_task_id="draft_slides",
    )

    revised = await planner.refine(plan=_looping_plan(), drift=drift, goals=_goals())

    assert revised is not None
    assert len(scripted.calls) == 1
    assert emitted == []


async def test_refine_rejects_refine_validation_failed_drift() -> None:
    """The planner must refuse to refine on its own signal, defending
    against an accidental re-refine loop if the steerer ever routed
    REFINE_VALIDATION_FAILED back through ``_handle_drift``.
    """
    scripted = _ScriptedLLM([])  # Must never be called.
    planner = LLMPlanner(call_llm=scripted)
    drift = DriftEvent(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        severity=DriftSeverity.CRITICAL,
        detail="hypothetical re-entrance",
        current_task_id="draft_slides",
    )

    result = await planner.refine(plan=_looping_plan(), drift=drift, goals=_goals())

    assert result is None
    assert scripted.calls == []


# ---------------------------------------------------------------------------
# goldfive#137: CANCELLED/FAILED -> PENDING reachability invariant.
#
# The live Qwen run on #137 produced a USER_STEER refinement where the
# old ``research`` task was CANCELLED but the new plan kept
# ``research -> r1`` (r1 PENDING). The executor stalled because r1 was
# never eligible. Part 1 of the fix rejects this shape at validation
# time; Part 2 teaches the refine prompt to avoid emitting it. These
# tests cover both.
# ---------------------------------------------------------------------------


def _user_steer_plan_with_cancelled_research() -> Plan:
    """A plan where ``research`` was CANCELLED by a prior user steer.

    Mirrors the sess_2026-04-21_0021 shape on #137: an operator
    pivoted away from the original topic, so the research task is
    terminal-CANCELLED and subsequent tasks were never run.
    """
    return Plan(
        id="plan-steer",
        run_id="run-steer",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research",
                title="Research solar panels",
                assignee_agent_id="researcher",
                status=TaskStatus.CANCELLED,
            ),
            Task(
                id="old_structure",
                title="Structure panels presentation",
                assignee_agent_id="writer",
                status=TaskStatus.CANCELLED,
            ),
        ],
        edges=[TaskEdge("research", "old_structure")],
        summary="Solar panels deck (superseded)",
        revision_index=0,
    )


def _bad_qwen_shape_revision() -> str:
    """Pre-fix Qwen pathology: graft new ``r1`` onto CANCELLED ``research``.

    The exact shape observed in the live run -- the LLM "chains" the
    new PENDING root off the prior CANCELLED task, which makes r1
    unreachable. Step 7 of ``Plan.validate`` rejects this.
    """
    return json.dumps(
        {
            "summary": "Replanned for solar flares",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research solar panels",
                    "assignee_agent_id": "researcher",
                    "status": "CANCELLED",
                },
                {
                    "id": "old_structure",
                    "title": "Structure panels presentation",
                    "assignee_agent_id": "writer",
                    "status": "CANCELLED",
                },
                {
                    "id": "r1",
                    "title": "Research solar flares",
                    "assignee_agent_id": "researcher",
                    "status": "PENDING",
                },
                {
                    "id": "o1",
                    "title": "Outline solar-flare deck",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            "edges": [
                # Terminal->terminal edge preserved (required).
                {"from_task_id": "research", "to_task_id": "old_structure"},
                # BUG: CANCELLED -> PENDING graft. r1 will stall.
                {"from_task_id": "research", "to_task_id": "r1"},
                {"from_task_id": "r1", "to_task_id": "o1"},
            ],
        }
    )


def _good_post_fix_revision() -> str:
    """Post-fix shape: new sub-DAG is rooted independently.

    ``r1`` is now a root (no incoming edge from a CANCELLED task);
    the new sub-DAG is self-contained and executable from the get-go.
    """
    return json.dumps(
        {
            "summary": "Replanned for solar flares",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research solar panels",
                    "assignee_agent_id": "researcher",
                    "status": "CANCELLED",
                },
                {
                    "id": "old_structure",
                    "title": "Structure panels presentation",
                    "assignee_agent_id": "writer",
                    "status": "CANCELLED",
                },
                {
                    "id": "r1",
                    "title": "Research solar flares",
                    "assignee_agent_id": "researcher",
                    "status": "PENDING",
                },
                {
                    "id": "o1",
                    "title": "Outline solar-flare deck",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "old_structure"},
                {"from_task_id": "r1", "to_task_id": "o1"},
            ],
        }
    )


async def test_refine_retry_catches_bad_edges_and_succeeds() -> None:
    """Attempt 1 emits the live Qwen pathology (CANCELLED->PENDING edge);
    attempt 2 corrects it. The planner's retry loop must catch the bad
    edge at validation time, append the error to the prompt, and
    recover on the second try.
    """
    scripted = _ScriptedLLM([_bad_qwen_shape_revision(), _good_post_fix_revision()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    emitted: list[DriftEvent] = []

    async def capture(drift: DriftEvent) -> None:
        emitted.append(drift)

    planner.set_drift_emitter(capture)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="Replan for solar flares",
        current_task_id="research",
    )

    revised = await planner.refine(
        plan=_user_steer_plan_with_cancelled_research(),
        drift=drift,
        goals=_goals(),
    )

    # Recovery succeeds on attempt 2.
    assert revised is not None
    assert len(scripted.calls) == 2
    # No REFINE_VALIDATION_FAILED should fire -- this isn't exhaustion.
    assert emitted == []
    # The correction prompt on attempt 2 carries the validator's error
    # (mentioning the offending edge) so the LLM sees exactly what to fix.
    _system, second_user_prompt, _model = scripted.calls[1]
    assert "PREVIOUS ATTEMPT FAILED" in second_user_prompt
    assert "'research' -> 'r1'" in second_user_prompt
    assert "PENDING task unexecutable" in second_user_prompt
    # The final plan has the new PENDING root free of any terminal
    # predecessor, so the executor can pick it up immediately.
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["r1"].status == TaskStatus.PENDING
    pairs = {(e.from_task_id, e.to_task_id) for e in revised.edges}
    assert ("research", "r1") not in pairs  # offending edge is gone
    assert ("r1", "o1") in pairs


async def test_refine_prompt_includes_forbidden_edges_section() -> None:
    """The refine prompt (both the generic and LOOPING variants) must
    include the FORBIDDEN EDGES guidance so the LLM knows not to graft
    new PENDING tasks onto CANCELLED/FAILED predecessors. Structural
    test on the prompt text so the guidance can't silently regress.
    """
    # Generic refine path (non-LOOPING, non-USER_STEER).
    scripted = _ScriptedLLM([_good_post_fix_revision()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    drift = DriftEvent(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        detail="replan",
        current_task_id="research",
    )

    await planner.refine(
        plan=_user_steer_plan_with_cancelled_research(),
        drift=drift,
        goals=_goals(),
    )
    _sys, user_prompt, _model = scripted.calls[0]
    assert "FORBIDDEN EDGES" in user_prompt
    assert "CANCELLED" in user_prompt and "FAILED" in user_prompt
    assert "independent sub-DAG" in user_prompt

    # LOOPING_TOOL_CALL refine path.
    scripted = _ScriptedLLM([_good_looping_revision_json()])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=2)
    loop_drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="stuck",
        current_task_id="draft_slides",
    )
    await planner.refine(plan=_looping_plan(), drift=loop_drift, goals=_goals())
    _sys, loop_prompt, _model = scripted.calls[0]
    assert "FORBIDDEN EDGES" in loop_prompt
    assert "CANCELLED" in loop_prompt and "FAILED" in loop_prompt

    # USER_STEER refine path has its own inline invariants block.
    from goldfive.types import Task as _Task  # noqa: PLC0415 -- test-local import

    user_steer_plan = Plan(
        id="p",
        run_id="r",
        goal_ids=["g1"],
        tasks=[
            _Task(
                id="research",
                title="Research panels",
                assignee_agent_id="researcher",
                status=TaskStatus.CANCELLED,
            ),
        ],
        edges=[],
    )
    scripted = _ScriptedLLM(
        [
            json.dumps(
                {
                    "summary": "replan",
                    "tasks": [
                        {
                            "id": "r1",
                            "title": "Research flares",
                            "assignee_agent_id": "researcher",
                            "status": "PENDING",
                        }
                    ],
                    "edges": [],
                }
            )
        ]
    )
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)
    steer_drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="solar flares",
        current_task_id="research",
    )
    await planner.refine(plan=user_steer_plan, drift=steer_drift, goals=_goals())
    _sys, steer_prompt, _model = scripted.calls[0]
    assert "FORBIDDEN EDGES" in steer_prompt
    assert "CANCELLED" in steer_prompt and "FAILED" in steer_prompt


# ---------------------------------------------------------------------------
# synthesize_goal_from_steer (goldfive#152)
# ---------------------------------------------------------------------------


class _SynthLLM:
    """Minimal call_llm stub for the synthesize_goal_from_steer tests.

    Named distinctly from the file's existing ``_ScriptedLLM`` (which
    accepts a list of responses and pops per-call) because the
    synthesize API is one-shot: a single canned string response.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self._response


async def test_synthesize_goal_from_steer_returns_goal_and_append_mode() -> None:
    """Minimal well-formed response → (Goal, 'append')."""
    scripted = _SynthLLM(
        json.dumps(
            {
                "goal": {"id": "steer1", "summary": "also include a summary slide"},
                "mode": "append",
                "reason": "additive request",
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer(
        "also include a summary slide"
    )
    assert result is not None
    goal, mode = result
    assert goal.id == "steer1"
    assert goal.summary == "also include a summary slide"
    assert mode == "append"
    # The system prompt asks for JSON + mode.
    _sys, user_prompt, _model = scripted.calls[0]
    assert "STEERING DIRECTIVE" in user_prompt
    assert "also include a summary slide" in user_prompt


async def test_synthesize_goal_from_steer_returns_replace_mode() -> None:
    """``mode: replace`` propagates through to the caller."""
    scripted = _SynthLLM(
        json.dumps(
            {
                "goal": {"id": "pivot", "summary": "scrap everything and do Y"},
                "mode": "replace",
                "reason": "scrap-and-pivot",
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer(
        "forget everything — just do Y"
    )
    assert result is not None
    _goal, mode = result
    assert mode == "replace"


async def test_synthesize_goal_from_steer_handles_markdown_fences() -> None:
    """LLM response wrapped in ```json fences parses correctly."""
    scripted = _SynthLLM(
        "```json\n"
        + json.dumps(
            {
                "goal": {"id": "g", "summary": "do the thing"},
                "mode": "append",
            }
        )
        + "\n```"
    )
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer("do the thing")
    assert result is not None
    goal, mode = result
    assert goal.summary == "do the thing"
    assert mode == "append"


async def test_synthesize_goal_from_steer_returns_none_on_empty_body() -> None:
    """Empty steer body short-circuits before the LLM is called."""
    scripted = _SynthLLM("should not be called")
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer("")
    assert result is None
    assert scripted.calls == []


async def test_synthesize_goal_from_steer_returns_none_on_malformed_json() -> None:
    """Unparseable LLM response → None (caller falls back to passthrough)."""
    scripted = _SynthLLM("not json at all {[")
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer("some steer")
    assert result is None


async def test_synthesize_goal_from_steer_defaults_mode_when_invalid() -> None:
    """Invalid ``mode`` falls back to 'append'."""
    scripted = _SynthLLM(
        json.dumps(
            {
                "goal": {"id": "g", "summary": "x"},
                "mode": "nonsense",
            }
        )
    )
    planner = LLMPlanner(call_llm=scripted)
    result = await planner.synthesize_goal_from_steer("some steer")
    assert result is not None
    _goal, mode = result
    assert mode == "append"


# ---------------------------------------------------------------------------
# Compound assignee normalization (goldfive#214)
# ---------------------------------------------------------------------------


def test_normalize_assignee_strips_compound_form() -> None:
    assert (
        _normalize_assignee("presentation-orchestrated-9b2b3a9c7289:research_agent")
        == "research_agent"
    )


def test_normalize_assignee_passes_bare_form_through() -> None:
    assert _normalize_assignee("research_agent") == "research_agent"
    assert _normalize_assignee("") == ""


def test_normalize_assignee_uses_last_colon() -> None:
    # Defensive: multi-segment prefixes still yield the trailing bare name.
    assert _normalize_assignee("a:b:c:research_agent") == "research_agent"


def test_plan_from_json_normalizes_compound_assignees(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "summary": "s",
        "tasks": [
            {
                "id": "t1",
                "title": "research",
                "assignee_agent_id": "client-xyz:research_agent",
            },
            {
                "id": "t2",
                "title": "draft",
                "assignee_agent_id": "client-xyz:writer",
            },
        ],
    }
    with caplog.at_level("WARNING", logger="goldfive.planner"):
        plan = _plan_from_json(payload, run_id="r", goal_ids=[])
    assert plan is not None
    assert [t.assignee_agent_id for t in plan.tasks] == ["research_agent", "writer"]
    compound_warnings = [
        r for r in caplog.records if "compound assignee_agent_id" in r.getMessage()
    ]
    assert len(compound_warnings) == 2


def test_plan_from_json_leaves_bare_assignees_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "summary": "s",
        "tasks": [
            {"id": "t1", "title": "research", "assignee_agent_id": "research_agent"},
            {"id": "t2", "title": "draft", "assignee_agent_id": "writer"},
        ],
    }
    with caplog.at_level("WARNING", logger="goldfive.planner"):
        plan = _plan_from_json(payload, run_id="r", goal_ids=[])
    assert plan is not None
    assert [t.assignee_agent_id for t in plan.tasks] == ["research_agent", "writer"]
    compound_warnings = [
        r for r in caplog.records if "compound assignee_agent_id" in r.getMessage()
    ]
    assert compound_warnings == []


def test_plan_from_json_normalizes_mixed_assignees(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "summary": "s",
        "tasks": [
            {"id": "t1", "title": "research", "assignee_agent_id": "research_agent"},
            {
                "id": "t2",
                "title": "draft",
                "assignee_agent_id": "client-xyz:writer",
            },
            {"id": "t3", "title": "review", "assignee_agent_id": "reviewer"},
        ],
    }
    with caplog.at_level("WARNING", logger="goldfive.planner"):
        plan = _plan_from_json(payload, run_id="r", goal_ids=[])
    assert plan is not None
    assert [t.assignee_agent_id for t in plan.tasks] == [
        "research_agent",
        "writer",
        "reviewer",
    ]
    compound_warnings = [
        r for r in caplog.records if "compound assignee_agent_id" in r.getMessage()
    ]
    assert len(compound_warnings) == 1
    assert "'client-xyz:writer'" in compound_warnings[0].getMessage()
    assert "'writer'" in compound_warnings[0].getMessage()


def test_default_system_prompt_forbids_compound_assignee() -> None:
    # Regression guard against the planner prompt drifting away from the
    # explicit "bare name only" directive added alongside #214.
    assert "do NOT add a namespace" in _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Supersedes coverage validator (observability — never rejects)
# ---------------------------------------------------------------------------


def _coverage_prior_plan(*, statuses: dict[str, TaskStatus] | None = None) -> Plan:
    """A 3-task prior plan whose statuses can be tweaked per test.

    ``a`` / ``b`` / ``c`` default to PENDING / RUNNING / PENDING. Tests
    that need terminal-status priors override via ``statuses``.
    """
    statuses = statuses or {}
    return Plan(
        id="p-prior",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="a",
                title="alpha",
                assignee_agent_id="agent_x",
                status=statuses.get("a", TaskStatus.PENDING),
            ),
            Task(
                id="b",
                title="bravo",
                assignee_agent_id="agent_y",
                status=statuses.get("b", TaskStatus.RUNNING),
            ),
            Task(
                id="c",
                title="charlie",
                assignee_agent_id="agent_z",
                status=statuses.get("c", TaskStatus.PENDING),
            ),
        ],
        edges=[],
        revision_index=0,
    )


def test_supersedes_coverage_no_orphans_when_every_drop_is_superseded() -> None:
    """Test 1 — full coverage. All dropped tasks have a supersedes link
    from some new task in the revised plan. Validator returns no
    orphans."""
    prior = _coverage_prior_plan()
    revised = Plan(
        id="p-revised",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="revised",
        tasks=[
            Task(
                id="a2",
                title="alpha v2",
                assignee_agent_id="agent_x",
                status=TaskStatus.PENDING,
                supersedes="a",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(
                id="b2",
                title="bravo v2",
                assignee_agent_id="agent_y",
                status=TaskStatus.PENDING,
                supersedes="b",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(
                id="c2",
                title="charlie v2",
                assignee_agent_id="agent_z",
                status=TaskStatus.PENDING,
                supersedes="c",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    orphans = _check_supersedes_coverage(revised, prior=prior)
    assert orphans == []


def test_supersedes_coverage_terminal_drops_are_covered_by_default() -> None:
    """Test 2 — all dropped priors are FAILED / CANCELLED. Absorbing-
    terminal old tasks don't need a supersedes link."""
    prior = _coverage_prior_plan(
        statuses={
            "a": TaskStatus.FAILED,
            "b": TaskStatus.CANCELLED,
            "c": TaskStatus.FAILED,
        }
    )
    revised = Plan(
        id="p-revised",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="revised — all priors absorbed by terminal status",
        tasks=[
            Task(
                id="brand_new",
                title="fresh work, not a replacement",
                assignee_agent_id="agent_x",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    orphans = _check_supersedes_coverage(revised, prior=prior)
    assert orphans == []


def test_supersedes_coverage_pending_drop_with_no_link_is_orphan() -> None:
    """Test 3 — legitimate orphan: a PENDING prior task is dropped with
    no supersedes from any new task. Validator surfaces it."""
    prior = _coverage_prior_plan()
    # Drop ``b`` entirely (no successor, not terminal). ``a`` and ``c``
    # are kept verbatim so they do not contribute to the dropped set.
    revised = Plan(
        id="p-revised",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="revised",
        tasks=[
            prior.tasks[0],  # a — preserved
            prior.tasks[2],  # c — preserved
        ],
        edges=[],
        revision_index=1,
    )
    orphans = _check_supersedes_coverage(revised, prior=prior)
    assert [t.id for t in orphans] == ["b"]
    assert orphans[0].title == "bravo"


def test_supersedes_coverage_mixed_only_orphan_uncovered() -> None:
    """Test 4 — mixed: one drop is superseded, one is FAILED-terminal,
    one is a legitimate orphan. Only the orphan is reported."""
    prior = _coverage_prior_plan(
        statuses={
            "a": TaskStatus.PENDING,
            "b": TaskStatus.FAILED,  # terminal — absorbed
            "c": TaskStatus.PENDING,  # orphan
        }
    )
    revised = Plan(
        id="p-revised",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="revised",
        tasks=[
            Task(
                id="a2",
                title="alpha v2",
                assignee_agent_id="agent_x",
                status=TaskStatus.PENDING,
                supersedes="a",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            # ``b`` is dropped but FAILED in prior — covered by status.
            # ``c`` is dropped with no supersedes → orphan.
        ],
        edges=[],
        revision_index=1,
    )
    orphans = _check_supersedes_coverage(revised, prior=prior)
    assert [t.id for t in orphans] == ["c"]


def test_supersedes_coverage_correct_kind_chain_is_not_a_drop() -> None:
    """Test 5 — CORRECT-kind supersession: the prior COMPLETED task is
    preserved verbatim in the revision (Option B contract — terminal
    tasks are immutable across refines), and a new task supersedes it
    with kind=CORRECT. The COMPLETED task is therefore NOT in the
    dropped set, and there are no orphans."""
    prior = Plan(
        id="p-prior",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                assignee_agent_id="research_agent",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
        revision_index=0,
    )
    revised = Plan(
        id="p-revised",
        run_id="r-cov",
        goal_ids=["g1"],
        summary="revised — correction supersedes",
        tasks=[
            # Prior COMPLETED task preserved verbatim.
            prior.tasks[0],
            # New task corrects it.
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected facts)",
                assignee_agent_id="research_agent",
                status=TaskStatus.PENDING,
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    orphans = _check_supersedes_coverage(revised, prior=prior)
    # Even if the COMPLETED task were somehow dropped, CORRECT-kind
    # links count toward "covered" identically to REPLACE — the
    # validator only cares whether *some* new task names the old id.
    assert orphans == []
    # And confirm the dropped set is genuinely empty: no prior id is
    # missing from revised.
    new_ids = {t.id for t in revised.tasks}
    dropped = {t.id for t in prior.tasks} - new_ids
    assert dropped == set()


async def test_refine_emits_orphan_event_on_legitimate_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end: a refine response that drops a PENDING prior task
    without a supersedes link triggers a WARNING log + a
    ``refine_orphaned_tasks`` sink event. The refine still applies."""
    from goldfive.sinks.memory import InMemorySink

    # Prior plan: one COMPLETED task (preserved) + two PENDING tasks
    # (one of which the LLM will silently drop).
    prior = Plan(
        id="p-prior",
        run_id="r-orphan",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="research",
                title="Research goldfish",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft_intro",
                title="Draft intro",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="draft_body",
                title="Draft body",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft_intro"),
            TaskEdge(from_task_id="draft_intro", to_task_id="draft_body"),
        ],
        revision_index=0,
    )
    # LLM drops ``draft_body`` entirely — no supersedes, not terminal.
    refine_response = json.dumps(
        {
            "summary": "narrowed scope",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_intro",
                    "title": "Draft intro",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft_intro"},
            ],
        }
    )
    scripted = _ScriptedLLM([refine_response])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)

    # Wire a span context provider so the planner knows which sinks to
    # emit on. (Mirrors ``DefaultSteerer.bind`` minus the steerer.)
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> object:
        return ([sink], "r-orphan", "s-orphan", "draft_body", lambda: next(seq))

    planner.set_span_context_provider(provider)

    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.WARNING,
        detail="scope narrowed",
        current_task_id="draft_body",
    )

    with caplog.at_level("WARNING", logger="goldfive.planner"):
        revised = await planner.refine(plan=prior, drift=drift, goals=_goals())

    # Refine applied (validator does not reject — observability only).
    assert revised is not None
    assert {t.id for t in revised.tasks} == {"research", "draft_intro"}
    # WARNING log surfaced the orphan.
    orphan_warnings = [
        r
        for r in caplog.records
        if "supersedes link or terminal status" in r.getMessage()
    ]
    assert len(orphan_warnings) == 1
    assert "'draft_body'" in orphan_warnings[0].getMessage()
    # Sink event emitted with kind=refine_orphaned_tasks and orphan
    # detail in the payload.
    orphan_events = [
        e
        for e in sink.events
        if isinstance(e, dict) and e.get("kind") == "refine_orphaned_tasks"
    ]
    assert len(orphan_events) == 1
    payload = orphan_events[0]["payload"]
    assert payload["orphan_count"] == 1
    assert payload["prior_plan_id"] == "p-prior"
    assert payload["prior_revision_index"] == 0
    # ``revision_index`` on the freshly-parsed revised plan reflects the
    # LLM JSON (default 0); the caller bumps it AFTER
    # ``_call_and_validate_refine`` returns, which is downstream of this
    # validator. Asserting on the at-validation-time value documents
    # that contract.
    assert payload["revised_revision_index"] == 0
    assert len(payload["orphans"]) == 1
    orphan = payload["orphans"][0]
    assert orphan["task_id"] == "draft_body"
    assert orphan["title"] == "Draft body"
    assert orphan["status"] == TaskStatus.PENDING.value
    assert orphan["assignee_agent_id"] == "writer"


async def test_refine_emits_no_orphan_event_when_coverage_complete() -> None:
    """Inverse of the orphan integration test: a refine response with
    full supersedes coverage emits NO ``refine_orphaned_tasks`` event."""
    from goldfive.sinks.memory import InMemorySink

    prior = Plan(
        id="p-prior",
        run_id="r-clean",
        goal_ids=["g1"],
        summary="prior",
        tasks=[
            Task(
                id="research",
                title="Research goldfish",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="draft",
                title="Draft post",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
        ],
        revision_index=0,
    )
    # LLM replaces ``draft`` with ``draft_v2`` carrying a supersedes link.
    refine_response = json.dumps(
        {
            "summary": "redirected draft",
            "tasks": [
                {
                    "id": "research",
                    "title": "Research goldfish",
                    "assignee_agent_id": "researcher",
                    "status": "COMPLETED",
                },
                {
                    "id": "draft_v2",
                    "title": "Draft post (new angle)",
                    "assignee_agent_id": "writer",
                    "status": "PENDING",
                    "supersedes": "draft",
                    "supersedes_kind": "REPLACE",
                },
            ],
            "edges": [
                {"from_task_id": "research", "to_task_id": "draft_v2"},
            ],
        }
    )
    scripted = _ScriptedLLM([refine_response])
    planner = LLMPlanner(call_llm=scripted, max_refine_attempts=1)
    sink = InMemorySink()
    seq = iter(range(1000))

    def provider() -> object:
        return ([sink], "r-clean", "s-clean", "draft", lambda: next(seq))

    planner.set_span_context_provider(provider)
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.WARNING,
        detail="redirect",
        current_task_id="draft",
    )

    revised = await planner.refine(plan=prior, drift=drift, goals=_goals())

    assert revised is not None
    orphan_events = [
        e
        for e in sink.events
        if isinstance(e, dict) and e.get("kind") == "refine_orphaned_tasks"
    ]
    assert orphan_events == []
