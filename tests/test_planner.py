"""Unit tests for ``goldfive.planner``.

Covers both ``PassthroughPlanner`` (trivial) and ``LLMPlanner`` (the
interesting case: stubbed async ``call_llm`` returning canned JSON,
markdown-fence stripping, refinement preserving completed tasks, and
error paths that must degrade gracefully to ``None``).
"""

from __future__ import annotations

import json

from goldfive.planner import (
    LLMPlanner,
    PassthroughPlanner,
    _plan_from_json,
    _strip_code_fences,
)
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
    _system, user_prompt, _model = stub.calls[0]
    assert "Goals:" in user_prompt
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
    result = await planner.refine(
        plan=_running_plan(), drift=drift, goals=_goals()
    )
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
    result = await planner.refine(
        plan=_running_plan(), drift=drift, goals=_goals()
    )
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
    result = await planner.refine(
        plan=_running_plan(), drift=drift, goals=_goals()
    )
    assert result is None
