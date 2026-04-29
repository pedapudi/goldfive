"""Regression tests for R2 (#205): topic-mismatch refusal on AgentTool dispatch.

Tier 1's F3 (#324) caught the "agent's tasks are all terminal" loop by
refusing AgentTool dispatches with a redirect error. The remaining gap
this suite covers: F3 says nothing about *what* the coordinator is
asking for. A coordinator can still call ``research_agent`` with an
off-topic request while the agent has a PENDING / RUNNING task — F3
lets it through and the post-hoc PLAN_DIVERGENCE drift detector fires
only after the wasted dispatch (v20 validation: 2 sev=1 PLAN_DIVERGENCE
drifts fired on debugger_agent + web_developer_agent for exactly this
shape).

R2 closes the gap with a keyword-overlap topic check at the same
dispatch boundary. The check is optimistic (false negatives preferred
over false positives — one shared meaningful word is enough to consider
the call on-topic). These tests exercise the public helper and the
five required edge cases from the design brief.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.adapters._adk_plugin import (  # noqa: E402
    _delegation_args_off_topic,
    _extract_agent_tool_request_text,
    _maybe_refuse_topic_mismatch,
    _normalize_for_overlap,
)
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ctx_for_plan(plan: Plan) -> Any:
    """Stub the SessionContext shape ``_maybe_refuse_topic_mismatch`` reads."""

    class _Ctx:
        def __init__(self, plan: Plan) -> None:
            self.session = Session(
                run_id="r1", goals=[Goal(id="g1", summary="x")], plan=plan
            )

    return _Ctx(plan)


def _plan_with(tasks: list[Task]) -> Plan:
    return Plan(id="p1", run_id="r1", goal_ids=["g1"], tasks=tasks, edges=[])


# ---------------------------------------------------------------------------
# normalisation primitives
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_drops_punctuation() -> None:
    assert _normalize_for_overlap("Solar Panels, Roof-mounted!") == {
        "solar",
        "panels",
        "roof",
        "mounted",
    }


def test_normalize_drops_short_tokens() -> None:
    """3-character tokens are filtered (the ≥4 minimum)."""
    out = _normalize_for_overlap("the cat sat on a roof")
    # "cat", "sat", "the", "roof" — only "roof" survives the ≥4 + stop-word filter.
    assert out == {"roof"}


def test_normalize_drops_stop_words() -> None:
    """The stop-word list strips delegation filler so plans don't trivially
    match every request on shared verbs."""
    out = _normalize_for_overlap("Please research and draft the report")
    # "please" / "research" / "draft" are stop-words. "report" survives.
    assert "research" not in out
    assert "draft" not in out
    assert "please" not in out
    assert "report" in out


def test_normalize_handles_non_string_input() -> None:
    assert _normalize_for_overlap(None) == set()
    assert _normalize_for_overlap(12345) == {"12345"}


def test_extract_request_prefers_request_key() -> None:
    assert (
        _extract_agent_tool_request_text({"request": "do thing", "input": "ignore"})
        == "do thing"
    )


def test_extract_request_falls_back_to_alt_keys() -> None:
    assert _extract_agent_tool_request_text({"prompt": "go"}) == "go"
    assert _extract_agent_tool_request_text({"query": "x"}) == "x"


def test_extract_request_concatenates_strings_when_no_known_key() -> None:
    out = _extract_agent_tool_request_text({"topic": "raccoons", "depth": "deep"})
    assert "raccoons" in out and "deep" in out


def test_extract_request_handles_string_args() -> None:
    assert _extract_agent_tool_request_text("just a string") == "just a string"


def test_extract_request_empty_for_empty_inputs() -> None:
    assert _extract_agent_tool_request_text({}) == ""
    assert _extract_agent_tool_request_text(None) == ""


# ---------------------------------------------------------------------------
# _delegation_args_off_topic — direct unit tests
# ---------------------------------------------------------------------------


def test_off_topic_when_no_overlap() -> None:
    """Plan task 'Research solar panels', request 'Research raccoons' — no
    overlap on meaningful tokens, so the call is off-topic."""
    task = Task(id="t1", title="Research solar panels", assignee_agent_id="r")
    assert _delegation_args_off_topic(
        tool_request_text="research raccoons",
        plan_tasks=[task],
    )


def test_on_topic_when_single_keyword_overlaps() -> None:
    """One shared meaningful word is enough — the check is optimistic."""
    task = Task(id="t1", title="Research solar panels", assignee_agent_id="r")
    assert not _delegation_args_off_topic(
        tool_request_text="research solar generation rates",
        plan_tasks=[task],
    )


def test_on_topic_when_match_is_in_description() -> None:
    """The description text contributes alongside the title."""
    task = Task(
        id="t1",
        title="Phase 1",
        description="Investigate quarterly revenue trends",
        assignee_agent_id="r",
    )
    assert not _delegation_args_off_topic(
        tool_request_text="please look at quarterly revenue",
        plan_tasks=[task],
    )


def test_multi_task_agent_matches_one() -> None:
    """If any candidate plan task overlaps, the dispatch is on-topic."""
    tasks = [
        Task(id="t1", title="Research solar panels", assignee_agent_id="r"),
        Task(id="t2", title="Research raccoons", assignee_agent_id="r"),
    ]
    # request matches t2 — must NOT be flagged off-topic just because t1 didn't match.
    assert not _delegation_args_off_topic(
        tool_request_text="study raccoons habitat patterns",
        plan_tasks=tasks,
    )


def test_empty_request_text_is_allowed() -> None:
    """Empty request: optimistic pass — we have no signal to block on."""
    task = Task(id="t1", title="Research solar panels", assignee_agent_id="r")
    assert not _delegation_args_off_topic(
        tool_request_text="",
        plan_tasks=[task],
    )


def test_stop_word_only_request_is_allowed() -> None:
    """All-stop-word request normalises to the empty set — optimistic pass."""
    task = Task(id="t1", title="Research solar panels", assignee_agent_id="r")
    assert not _delegation_args_off_topic(
        tool_request_text="please do this for me",
        plan_tasks=[task],
    )


def test_no_plan_tasks_is_allowed() -> None:
    """Empty candidate list — caller decides; this helper says 'allow'."""
    assert not _delegation_args_off_topic(
        tool_request_text="solar panels rooftop",
        plan_tasks=[],
    )


def test_task_with_no_checkable_content_is_allowed() -> None:
    """A task whose title + description normalise to empty cannot be
    classified — optimistic pass."""
    task = Task(id="t1", title="", description="", assignee_agent_id="r")
    assert not _delegation_args_off_topic(
        tool_request_text="solar panels rooftop",
        plan_tasks=[task],
    )


# ---------------------------------------------------------------------------
# _maybe_refuse_topic_mismatch — wired-in helper tests
# ---------------------------------------------------------------------------


def test_refuses_off_topic_dispatch() -> None:
    """v20-style scenario: research_agent has 'Research solar panels' as
    its PENDING task, but the coordinator calls it with 'Research raccoons'.
    Refusal must fire with topic_mismatch=True and the expected_topic set."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": "Research raccoons"},
    )

    assert out is not None
    assert out["topic_mismatch"] is True
    assert out["expected_topic"] == "Research solar panels"
    assert "Research raccoons"[:30] in out["error"]
    assert "Research solar panels" in out["error"]


def test_allows_on_topic_dispatch() -> None:
    """Topic-overlap >0 — call goes through (returns None)."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": "Research solar panels for the presentation"},
    )

    assert out is None


def test_multi_task_agent_one_match_allowed() -> None:
    """Two PENDING tasks; request matches one — allowed."""
    tasks = [
        Task(
            id="t1",
            title="Research solar panels",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
        Task(
            id="t2",
            title="Research raccoon populations",
            assignee_agent_id="research_agent",
            status=TaskStatus.PENDING,
        ),
    ]
    ctx = _ctx_for_plan(_plan_with(tasks))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": "study raccoons populations and habitat"},
    )

    assert out is None


def test_empty_request_text_allowed_through_helper() -> None:
    """Edge case: AgentTool fired with no explicit request body — optimistic
    pass through the dispatch boundary."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": ""},
    )

    assert out is None


def test_stop_word_only_request_allowed_through_helper() -> None:
    """Request like 'please do this for me' has no checkable content — pass."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": "please do this for me"},
    )

    assert out is None


def test_off_plan_agent_not_double_flagged() -> None:
    """An agent with no assigned plan task is off-plan — that's
    PLAN_DIVERGENCE territory, not R2's problem. Must return None so the
    drift detector remains the single source of truth for off-plan flags."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="off_plan_agent",
        tool_args={"request": "totally unrelated content"},
    )

    assert out is None


def test_all_terminal_tasks_not_double_flagged() -> None:
    """When every assigned task is terminal, F3 owns the refusal.
    R2's helper must return None so the two surfaces don't double-fire."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.COMPLETED,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="research_agent",
        tool_args={"request": "research raccoons"},
    )

    # F3 territory — R2 abstains.
    assert out is None


def test_qualified_agent_name_matches_bare_assignee() -> None:
    """ADK passes fully-qualified agent paths like 'coordinator.research_agent';
    the F3-style bare-name match must round-trip here too."""
    task = Task(
        id="t1",
        title="Research solar panels",
        assignee_agent_id="research_agent",
        status=TaskStatus.PENDING,
    )
    ctx = _ctx_for_plan(_plan_with([task]))

    out = _maybe_refuse_topic_mismatch(
        ctx=ctx,
        target_agent="coordinator.research_agent",
        tool_args={"request": "Research raccoons"},
    )

    assert out is not None
    assert out["topic_mismatch"] is True


def test_no_plan_returns_none() -> None:
    """No plan installed on the session — defensive pass."""

    class _Ctx:
        def __init__(self) -> None:
            self.session = Session(
                run_id="r1", goals=[Goal(id="g1", summary="x")], plan=None
            )

    out = _maybe_refuse_topic_mismatch(
        ctx=_Ctx(),
        target_agent="research_agent",
        tool_args={"request": "anything"},
    )

    assert out is None
