"""Observational delegation-time task pinning (goldfive#259).

#252 zeroed ``Task.assignee_agent_id`` at plan-parse time so the LLM
cannot pre-declare which sub-agent will pick up a task. The follow-up
in #259 wires the observational re-population: at ``delegation_observed``
time the plugin walks the plan, picks the eligible PENDING task this
delegation is enacting, stamps ``task.assignee_agent_id`` with the
invoked agent's name and pins ``session.current_task_id`` so the
reporting-tool pin lookup in :func:`_resolve_pinned_task_id` resolves
on the delegated sub-invocation's tool calls.

This file pins the selection algorithm and the end-to-end reporting-
tool resolution behaviour:

  * Linear plan A -> B -> C; first delegation binds A and pins it.
  * Sequential delegations: after A completes, the next delegation
    binds B.
  * Multi-eligible (two parallel PENDING tasks): topic-match in tool
    args picks the matching task.
  * Multi-eligible without topic-match in args: first by plan order
    wins (no guessing).
  * No eligible task (all PENDING tasks have non-COMPLETED predecessors)
    -> no pin, DEBUG log, no exception.
  * Integration: after the pin, ``before_tool_callback`` resolves
    ``report_task_started`` to the bound task (not the silent-ack
    no-op).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.adapters._adk_plugin import (  # noqa: E402
    SESSION_CONTEXT_STATE_KEY,
    SessionContext,
    make_adk_plugin,
)
from goldfive.state_store import StateStore  # noqa: E402
from goldfive.types import (  # noqa: E402
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=list(tasks),
        edges=list(edges or []),
        summary="",
    )


def _session_with(plan: Plan) -> Session:
    return Session(run_id="r1", plan=plan)


def _ctx(session: Session, host_agent_name: str = "coord") -> SessionContext:
    return SessionContext(
        session=session,
        steerer=None,
        task=None,
        tool_handlers={},
        host_agent_name=host_agent_name,
    )


def _find_task(plan: Plan, task_id: str) -> Task | None:
    for t in plan.tasks:
        if t.id == task_id:
            return t
    return None


# ---------------------------------------------------------------------------
# Selection algorithm
# ---------------------------------------------------------------------------


def test_linear_plan_first_delegation_binds_first_task() -> None:
    """3-task linear plan A->B->C, all PENDING, no pre-declared assignees.

    Delegation to agent X picks task A (only DAG-ready PENDING task),
    stamps assignee, pins current_task_id. The other two tasks remain
    untouched.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        Task(id="C", title="Review the draft"),
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="B", to_task_id="C"),
        ],
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={"request": "go"},
    )

    a = _find_task(session.plan, "A")
    b = _find_task(session.plan, "B")
    c = _find_task(session.plan, "C")
    assert a is not None and b is not None and c is not None
    assert a.assignee_agent_id == "X"
    assert b.assignee_agent_id == ""
    assert c.assignee_agent_id == ""
    assert session.current_task_id == "A"
    store = StateStore.for_session(session)
    assert store.pin_current_task() == "A"


def test_sequential_delegations_bind_next_task_after_completion() -> None:
    """After A completes, delegating to a fresh agent binds B (the new
    DAG-ready PENDING task)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic", status=TaskStatus.COMPLETED),
        Task(id="B", title="Write the draft"),
        Task(id="C", title="Review the draft"),
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="B", to_task_id="C"),
        ],
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="Y",
        tool_args={"request": "draft"},
    )

    b = _find_task(session.plan, "B")
    assert b is not None
    assert b.assignee_agent_id == "Y"
    assert session.current_task_id == "B"


def test_multi_eligible_topic_match_wins() -> None:
    """Two parallel PENDING tasks (no edge between them) and tool args
    contain a token that overlaps with one task's title — the matching
    task wins."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(
            id="t1",
            title="solar telemetry research",
            description="gather solar telemetry",
        ),
        Task(
            id="t2",
            title="quarterly invoice review",
            description="reconcile quarterly invoices",
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="researcher",
        tool_args={"topic": "please research solar telemetry"},
    )

    t1 = _find_task(session.plan, "t1")
    t2 = _find_task(session.plan, "t2")
    assert t1 is not None and t2 is not None
    assert t1.assignee_agent_id == "researcher"
    assert t2.assignee_agent_id == ""
    assert session.current_task_id == "t1"


def test_multi_eligible_no_topic_match_falls_back_to_first() -> None:
    """Two parallel PENDING tasks, tool args have no overlapping tokens
    with either title -> fall back to the first task by plan order."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="t1", title="solar telemetry research"),
        Task(id="t2", title="quarterly invoice review"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    # Tool args contain only sub-4-char tokens (the scorer filter
    # threshold) so no candidate scores > 0 and the fallback kicks in.
    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="researcher",
        tool_args={"x": "go on"},
    )

    t1 = _find_task(session.plan, "t1")
    t2 = _find_task(session.plan, "t2")
    assert t1 is not None and t2 is not None
    assert t1.assignee_agent_id == "researcher"
    assert t2.assignee_agent_id == ""
    assert session.current_task_id == "t1"


def test_no_eligible_task_leaves_session_unpinned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All PENDING tasks have non-COMPLETED predecessors -> no pin, no
    exception, DEBUG log fires."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        # A is PENDING (not COMPLETED) so B's predecessor isn't satisfied.
        Task(id="A", title="Plan the work"),
        Task(id="B", title="Execute the work"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    # Mark A non-PENDING (e.g. RUNNING) so A is not eligible either
    # (only PENDING tasks count) — that matches the brief's "no
    # eligible task" case.
    import dataclasses

    plan = dataclasses.replace(
        plan,
        tasks=(
            dataclasses.replace(plan.tasks[0], status=TaskStatus.RUNNING),
            plan.tasks[1],
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    with caplog.at_level(logging.DEBUG, logger="goldfive"):
        plugin._maybe_pin_delegation_task(
            ctx=ctx,
            invoked_agent_name="Z",
            tool_args={"request": "go"},
        )

    a = _find_task(session.plan, "A")
    b = _find_task(session.plan, "B")
    assert a is not None and b is not None
    assert a.assignee_agent_id == ""
    assert b.assignee_agent_id == ""
    assert session.current_task_id == ""
    assert any(
        "no eligible PENDING task" in r.getMessage() for r in caplog.records
    ), f"expected DEBUG log; got {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# goldfive#265 — structural disambiguation tiers (required-tools + agent-name)
# ---------------------------------------------------------------------------


class _FakeFunctionToolForCover:
    """Stand-in for an ADK FunctionTool: carries a ``.name`` attribute.

    Used by goldfive#265 tier-1 tests where the pin introspects
    ``invoked_agent.tools[*].name`` to compute the required-tools cover.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeInvokedAgent:
    """Stand-in for an ADK ``BaseAgent``: carries ``.name`` and ``.tools``."""

    def __init__(self, name: str, tools: list[Any] | None = None) -> None:
        self.name = name
        self.tools = list(tools or [])


def test_tier1_required_tools_cover_wins_over_topo_order() -> None:
    """goldfive#265 tier 1: required-tools cover picks the matching task
    even when it is not topo-first.

    Two parallel DAG-ready PENDING tasks A and B. A.required_tools =
    ("patch_file",); B.required_tools = (). The invoked agent has only
    a ``patch_file`` tool. Tier 1 picks A; if we had fallen through to
    tier 3 we'd have picked B (first by plan order). Asserts the tier
    fired by checking A is bound.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        # B FIRST in plan order so the topo-order fallback would pick B.
        Task(id="B", title="general task"),
        Task(
            id="A",
            title="apply patch",
            required_tools=("patch_file",),
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    invoked_agent = _FakeInvokedAgent(
        name="patcher",
        tools=[_FakeFunctionToolForCover("patch_file")],
    )

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="patcher",
        tool_args={"x": "go"},
        invoked_agent=invoked_agent,
    )

    a = _find_task(session.plan, "A")
    b = _find_task(session.plan, "B")
    assert a is not None and b is not None
    assert a.assignee_agent_id == "patcher", (
        f"tier-1 should pick A (required_tools=[patch_file] covered by "
        f"agent); got A={a.assignee_agent_id!r} B={b.assignee_agent_id!r}"
    )
    assert b.assignee_agent_id == ""
    assert session.current_task_id == "A"


def test_tier2_agent_name_match_picks_reviewer_task() -> None:
    """goldfive#265 tier 2: agent name ``reviewer_agent`` picks
    ``review_presentation`` over ``outline_presentation`` and
    ``draft_presentation``.

    Reproduces the session-4538863f bug: the coordinator delegated to
    ``reviewer_agent``, but the old tier-3-only pin picked
    ``draft_presentation`` because it was DAG-next. With tier 2 the
    stem ``reviewer`` substring-matches ``review_presentation``'s
    title and the pin selects it.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    # All three DAG-ready PENDING (no edges between them).
    plan = _plan(
        Task(id="outline_presentation", title="Outline the presentation"),
        Task(id="draft_presentation", title="Draft the presentation"),
        Task(id="review_presentation", title="Review the presentation"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        # tool_args carries a generic prompt that does NOT topic-match
        # any specific candidate — proves tier 2 (not tier 3) fired.
        tool_args={"request": "please proceed"},
    )

    outline = _find_task(session.plan, "outline_presentation")
    draft = _find_task(session.plan, "draft_presentation")
    review = _find_task(session.plan, "review_presentation")
    assert outline is not None and draft is not None and review is not None
    assert review.assignee_agent_id == "reviewer_agent", (
        f"tier-2 should pick review_presentation for reviewer_agent; got "
        f"outline={outline.assignee_agent_id!r} "
        f"draft={draft.assignee_agent_id!r} "
        f"review={review.assignee_agent_id!r}"
    )
    assert outline.assignee_agent_id == ""
    assert draft.assignee_agent_id == ""
    assert session.current_task_id == "review_presentation"


def test_tier2_works_when_required_tools_empty() -> None:
    """goldfive#265 tier 2 still picks correctly when no Tier 1 match
    is possible (all candidates have empty required_tools).

    Same fixture as the reviewer_agent case but explicitly with
    ``invoked_agent`` passed and no tools — tier 1 returns ``None``
    (no required_tools on any candidate), tier 2 fires.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="outline_presentation", title="Outline the presentation"),
        Task(id="draft_presentation", title="Draft the presentation"),
        Task(id="review_presentation", title="Review the presentation"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    invoked_agent = _FakeInvokedAgent(name="reviewer_agent", tools=[])

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"request": "please proceed"},
        invoked_agent=invoked_agent,
    )

    review = _find_task(session.plan, "review_presentation")
    assert review is not None and review.assignee_agent_id == "reviewer_agent"
    assert session.current_task_id == "review_presentation"


def test_tier3_fallback_when_no_name_overlap() -> None:
    """goldfive#265 tier 3 (existing topo-order fallback) still wins
    when agent name doesn't match any task title/description and no
    required_tools are populated.

    Agent ``web_developer_agent``; tasks have titles unrelated to
    "developer" or "web". Tier 1 vacuous (no required_tools), tier 2
    vacuous (no stem match), tier 3 picks the first eligible by plan
    order.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="t1", title="Compose the executive summary"),
        Task(id="t2", title="Format the bibliography"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="web_developer_agent",
        # No useful tokens (all sub-4-char so scorer is a no-op too).
        tool_args={"x": "go"},
    )

    t1 = _find_task(session.plan, "t1")
    t2 = _find_task(session.plan, "t2")
    assert t1 is not None and t2 is not None
    # Tier 3 fallback: first eligible by plan order.
    assert t1.assignee_agent_id == "web_developer_agent"
    assert t2.assignee_agent_id == ""
    assert session.current_task_id == "t1"


def test_tier2_ambiguous_agent_name_falls_through_to_tier3() -> None:
    """goldfive#265 negative case: ambiguous agent name (e.g.
    ``helper_agent``) doesn't match any task — tier 2 returns ``None``
    and tier 3 (topo-order fallback) takes over cleanly without
    crashing.

    Also covers the "no stems after role-suffix strip" edge: the agent
    name ``agent`` alone would strip to an empty stem tuple; ensure no
    crash.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="t1", title="Compose the executive summary"),
        Task(id="t2", title="Format the bibliography"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="helper_agent",
        tool_args={"x": "go"},
    )

    # Tier 3 fallback: helper doesn't match t1 or t2 titles, so we
    # fall through to first-by-plan-order.
    t1 = _find_task(session.plan, "t1")
    assert t1 is not None
    assert t1.assignee_agent_id == "helper_agent"
    assert session.current_task_id == "t1"


def test_tier1_wins_over_tier2_on_conflict() -> None:
    """goldfive#265: when tier 1 (required_tools) and tier 2 (agent
    name) would pick different tasks, tier 1 wins.

    Two tasks:

    * ``review_data``: required_tools=("query_db",), title contains
      "review" so tier 2 would pick it.
    * ``compile_report``: required_tools=("compile_report_tool",),
      no agent-name overlap.

    The agent is ``reviewer_agent`` (tier 2 stem ``reviewer``) but
    only has the ``compile_report_tool`` tool — so tier 1 covers
    ``compile_report`` uniquely. Tier 1 should win.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(
            id="review_data",
            title="review the dataset",
            required_tools=("query_db",),
        ),
        Task(
            id="compile_report",
            title="produce the report",
            required_tools=("compile_report_tool",),
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    invoked_agent = _FakeInvokedAgent(
        name="reviewer_agent",
        tools=[_FakeFunctionToolForCover("compile_report_tool")],
    )

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"x": "go"},
        invoked_agent=invoked_agent,
    )

    review = _find_task(session.plan, "review_data")
    compile_t = _find_task(session.plan, "compile_report")
    assert review is not None and compile_t is not None
    # Tier 1 wins on conflict — picked by required_tools cover.
    assert compile_t.assignee_agent_id == "reviewer_agent"
    assert review.assignee_agent_id == ""


# ---------------------------------------------------------------------------
# Issue #405 MEDIUM #5 — Tier-2 stem-match Rule A cross-check
# ---------------------------------------------------------------------------
#
# Audit finding: ``_stem_token_match`` is bi-directionally permissive.
# ``reviewer`` substring-matches ``reviewing`` / ``previewer`` /
# ``reviews``. When capability_check Rule A is suppressed by
# ``_looks_like_delegation_task`` (delegation-shaped tasks), the
# Tier-2 stem matcher could pin a delegation-only ``reviewer_agent``
# onto a task whose title contains ``previewer`` (or another
# substring-only collision) even though Rule A would otherwise fire
# at detector time. Fix: post-Tier-2, run the Rule A predicate
# against the chosen task; fall through to Tier-3 if it would fire.


def test_405_medium5_tier2_falls_through_when_rule_a_would_fire() -> None:
    """Issue #405 MEDIUM #5 regression.

    The invoked agent is a delegation-only coordinator (all tools are
    AgentTool wrappers) named ``reviewer_agent``. Two PENDING tasks:

    * ``preview_export``: title contains ``previewer`` — stem-matches
      ``reviewer`` bi-directionally (``review`` is in ``previewer``)
      AND reads as a leaf task (no delegation verbs).
    * ``other_task``: title has no stem overlap with ``reviewer``.

    Pre-fix: Tier-2 picks ``preview_export`` because ``reviewer``
    substring-matches a ``previewer`` token; Rule A would then fire
    at delegation_observed time and force a costly cancel+refine.
    Post-fix: Tier-2's Rule A cross-check rejects the pick (delegation-
    only agent + leaf task), falls through to Tier-3, which picks the
    first eligible by plan order. The delegation-only agent never gets
    bound to a structurally-wrong leaf task.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        # ``other_task`` first so the topo-order fallback picks it.
        Task(id="other_task", title="Compose the executive summary"),
        Task(id="preview_export", title="Generate the previewer export"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    # Delegation-only agent: every tool is an AgentTool wrapper.
    invoked_agent = _FakeInvokedAgent(
        name="reviewer_agent",
        tools=[_FakeAgentTool("sub_a"), _FakeAgentTool("sub_b")],
    )

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"x": "go"},
        invoked_agent=invoked_agent,
    )

    other = _find_task(session.plan, "other_task")
    preview = _find_task(session.plan, "preview_export")
    assert other is not None and preview is not None
    # Tier-2's Rule A cross-check rejected the ``previewer`` false
    # positive; Tier-3 picked ``other_task`` (first by plan order).
    assert preview.assignee_agent_id == "", (
        f"Tier-2 must not pin a delegation-only agent onto a leaf "
        f"task that would trigger Rule A; got "
        f"preview_export.assignee={preview.assignee_agent_id!r}"
    )
    assert other.assignee_agent_id == "reviewer_agent"
    assert session.current_task_id == "other_task"


def test_405_medium5_tier2_still_picks_delegation_task() -> None:
    """Issue #405 MEDIUM #5 — Rule A is correctly suppressed by
    ``_looks_like_delegation_task`` for orchestrational task titles.

    Same fixture as the regression case but the candidate's title
    explicitly reads as delegation (``"Coordinate the reviewer
    review"``). The Tier-2 Rule A cross-check honours the delegation-
    task carve-out and lets the stem match stand. Pin lands.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="other_task", title="Compose the executive summary"),
        Task(
            id="review_round",
            title="Coordinate the reviewer review round",
        ),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    invoked_agent = _FakeInvokedAgent(
        name="reviewer_agent",
        tools=[_FakeAgentTool("sub_a")],
    )

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"x": "go"},
        invoked_agent=invoked_agent,
    )

    review = _find_task(session.plan, "review_round")
    assert review is not None
    assert review.assignee_agent_id == "reviewer_agent", (
        "delegation-shaped chosen task must NOT trigger the Rule A "
        "cross-check; Tier-2 should still pin"
    )
    assert session.current_task_id == "review_round"


def test_405_medium5_tier2_unaffected_when_agent_has_leaf_tools() -> None:
    """Issue #405 MEDIUM #5 — Rule A doesn't fire when the agent has
    any leaf tool, so Tier-2 picks normally even on a leaf task.

    Reproduces the ``review_presentation`` happy path from
    ``test_tier2_agent_name_match_picks_reviewer_task`` but with the
    agent carrying a FunctionTool. Rule A wouldn't fire at detector
    time and Tier-2's cross-check matches that — pin proceeds.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="outline_presentation", title="Outline the presentation"),
        Task(id="draft_presentation", title="Draft the presentation"),
        Task(id="review_presentation", title="Review the presentation"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    invoked_agent = _FakeInvokedAgent(
        name="reviewer_agent",
        tools=[_FakeFunctionTool("scan_pdf")],  # leaf tool
    )

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"request": "please proceed"},
        invoked_agent=invoked_agent,
    )

    review = _find_task(session.plan, "review_presentation")
    assert review is not None
    assert review.assignee_agent_id == "reviewer_agent"
    assert session.current_task_id == "review_presentation"


def test_405_medium5_legacy_callers_without_invoked_agent_unaffected() -> None:
    """Issue #405 MEDIUM #5 — callers that don't pass ``invoked_agent``
    keep the pre-#405 behaviour exactly.

    The Rule A cross-check is opt-in on the presence of
    ``invoked_agent``. Without it the Tier-2 match wins even on the
    substring-only collision (the detector still has the final say at
    delegation_observed time via the real ``detect_capability_mismatch``
    Rule A — this is a defence-in-depth, not the primary gate).
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="other_task", title="Compose the executive summary"),
        Task(id="preview_export", title="Generate the previewer export"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="reviewer_agent",
        tool_args={"x": "go"},
        # No ``invoked_agent`` — legacy code path.
    )

    preview = _find_task(session.plan, "preview_export")
    assert preview is not None
    # Substring match wins on the legacy path (Rule A cross-check
    # didn't run because ``invoked_agent`` wasn't supplied).
    assert preview.assignee_agent_id == "reviewer_agent"
    assert session.current_task_id == "preview_export"


def test_idempotent_on_already_assigned_task() -> None:
    """Re-running the pin with the same agent on the same eligible task
    is a no-op (assignee already matches, no spurious plan swap)."""
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
    )
    session = _session_with(plan)
    ctx = _ctx(session)

    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={},
    )
    first_plan = session.plan
    assert first_plan is not None
    plugin._maybe_pin_delegation_task(
        ctx=ctx,
        invoked_agent_name="X",
        tool_args={},
    )
    # Second call kept the same plan pointer (idempotent).
    assert session.plan is first_plan
    a = _find_task(session.plan, "A")
    assert a is not None and a.assignee_agent_id == "X"


# ---------------------------------------------------------------------------
# Integration with the reporting-tool pin lookup
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgentTool:
    """ADK AgentTool stand-in: has both ``.agent`` and ``.name``."""

    def __init__(self, agent_name: str) -> None:
        self.agent = _FakeAgent(agent_name)
        self.name = agent_name


class _FakeFunctionTool:
    """Stand-in for a reporting tool (FunctionTool with a ``func``)."""

    def __init__(self, name: str) -> None:
        self.name = name

        def _func() -> None:
            return None

        _func.__name__ = name
        self.func = _func


class _FakeInvocationContext:
    def __init__(self, session_state: dict, agent_name: str) -> None:
        class _ADKSession:
            def __init__(self, state: dict) -> None:
                self.state = state

        self.session = _ADKSession(session_state)
        self.invocation_id = "inv-1"
        self.agent = _FakeAgent(agent_name)


class _FakeToolContext:
    def __init__(self, inv_ctx: Any, function_call_id: str = "fc-1") -> None:
        self._invocation_context = inv_ctx
        self.function_call_id = function_call_id


async def test_after_pin_report_task_started_resolves_bound_task() -> None:
    """End-to-end: drive ``before_tool_callback`` through an AgentTool
    dispatch (which triggers the pin) followed by a ``report_task_started``
    call from the sub-agent. The reporting-tool lookup must find the
    bound task and short-circuit through the reporting handler — not
    the "no task pinned" silent ack.
    """
    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    session = _session_with(plan)

    # Build a SessionContext that exposes the reporting-tool spec list
    # so the plugin's ``before_tool_callback`` routes
    # ``report_task_started`` through ``invoke_tool`` (not the silent
    # ack). Reuse the canonical spec list from goldfive.reporting so
    # we get the real handler.
    from goldfive.reporting import select_reporting_tools

    specs = select_reporting_tools(False)
    ctx_obj = SessionContext(
        session=session,
        steerer=None,
        task=None,
        tools=specs,
        host_agent_name="coord",
    )
    # Set the plugin's active ctx so the live-run path reaches the
    # SessionContext via ``session_context_from_invocation``.
    plugin.set_active_context(ctx_obj)

    # Stash the SessionContext on ADK state too so the legacy / unit-test
    # resolver path also finds it (defensive belt-and-braces).
    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-dispatch")

    # Phase 1: coord fires AgentTool(researcher) — triggers the pin.
    agent_tool = _FakeAgentTool("researcher")
    res = await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "please research the topic"},
        tool_context=coord_tool_context,
    )
    # AgentTool dispatch is not short-circuited (no runaway-cap).
    assert res is None or (isinstance(res, dict) and not res.get("skipped"))

    # The pin landed.
    a = _find_task(session.plan, "A")
    assert a is not None
    assert a.assignee_agent_id == "researcher"
    assert session.current_task_id == "A"

    # Phase 2: researcher (sub-agent) fires report_task_started. With
    # the pin in place, the reporting handler runs (returns a dict
    # describing the transition) rather than the silent ack.
    sub_inv = _FakeInvocationContext(adk_state, "researcher")
    sub_tool_context = _FakeToolContext(sub_inv, function_call_id="fc-report")
    report_tool = _FakeFunctionTool("report_task_started")

    res = await plugin.before_tool_callback(
        tool=report_tool,
        tool_args={},  # task_id is hidden from the LLM; the pin supplies it.
        tool_context=sub_tool_context,
    )

    # The reporting handler executed (or attempted to execute), which
    # means the pin lookup resolved — the response carries either the
    # handler's success payload or the handler's error payload (when
    # the test's stub steerer can't drive the real transition). What
    # MUST NOT happen is the no-task-pinned silent ack: the silent-ack
    # branch returns EXACTLY ``{"acknowledged": True}`` with no other
    # keys. Anything else means the pin resolved and the dispatch path
    # reached invoke_tool.
    assert isinstance(res, dict)
    assert res != {"acknowledged": True}, (
        f"reporting handler did not run; got silent ack: {res}"
    )
    # The task_id from the pin was injected into tool_args before the
    # dispatch (visible because the handler's response either succeeds
    # against task A or surfaces an error that references A). Either
    # way, the pin-resolution loop did not fall through to no-op.
    assert "missing_task_id" not in str(res), (
        f"task_id was not injected from the pin: {res}"
    )


# ---------------------------------------------------------------------------
# goldfive#262 — DelegationObserved emit happens AFTER the pin
# ---------------------------------------------------------------------------


class _SinkingSteerer:
    """Minimal steerer stub that owns a ``_sinks`` list.

    The plugin's ``_emit_observability`` reads ``steerer._sinks`` to fan
    sink events out — this stub is just enough to capture
    ``DelegationObserved`` events the plugin emits from
    ``before_tool_callback``.
    """

    def __init__(self, sink: Any) -> None:
        self._sinks = [sink]

    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _delegation_events(events: list[Any]) -> list[Any]:
    """Filter ``events`` to ``DelegationObserved`` payloads only."""
    out: list[Any] = []
    for e in events:
        if not hasattr(e, "WhichOneof"):
            continue
        if e.WhichOneof("payload") == "delegation_observed":
            out.append(e.delegation_observed)
    return out


async def test_delegation_observed_event_carries_bound_task_id() -> None:
    """The ``DelegationObserved`` event's ``task_id`` is the freshly-bound
    plan-task id (goldfive#262).

    Before #262 the emit ran BEFORE ``_maybe_pin_delegation_task``, so
    the proto field was empty on the typical orchestration-only
    coordinator turn (``ctx.task is None``). After the reorder the emit
    reads ``session.current_task_id`` which the pin just stamped — so
    the harmonograf ingest can attribute the delegation to the right
    task and stamp ``tasks.assignee_agent_id``.
    """
    from goldfive.sinks.memory import InMemorySink

    plugin = make_adk_plugin(host_agent_name="coord")
    plan = _plan(
        Task(id="A", title="Research the topic"),
        Task(id="B", title="Write the draft"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    session = _session_with(plan)

    sink = InMemorySink()
    steerer = _SinkingSteerer(sink)

    # Orchestration-only coordinator turn: ctx.task is None — same
    # shape as the live coordinator that reproduced the bug
    # (session 4a721a07).
    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    agent_tool = _FakeAgentTool("researcher")
    await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "please research the topic"},
        tool_context=coord_tool_context,
    )

    # Confirm the pin landed.
    assert session.current_task_id == "A"
    a = _find_task(session.plan, "A")
    assert a is not None and a.assignee_agent_id == "researcher"

    # Exactly one delegation_observed event, carrying the bound task id.
    delegations = _delegation_events(sink.events)
    assert len(delegations) == 1, (
        f"expected one DelegationObserved; got {[type(e).__name__ for e in sink.events]}"
    )
    d = delegations[0]
    assert d.from_agent == "coord"
    assert d.to_agent == "researcher"
    assert d.task_id == "A", (
        f"DelegationObserved.task_id must carry the bound id; got '{d.task_id}'"
    )


async def test_delegation_observed_task_id_empty_when_no_eligible_task() -> None:
    """When the pin can't bind a task (no eligible PENDING tasks), the
    emit lands with ``task_id == ""`` (defensive — no fake binding).

    Mirror of the no-eligible-task case from the selection-algorithm
    tests, but checks the emit side rather than the pin side.
    """
    import dataclasses

    from goldfive.sinks.memory import InMemorySink

    plugin = make_adk_plugin(host_agent_name="coord")
    # Same shape as ``test_no_eligible_task_leaves_session_unpinned``:
    # A is RUNNING (not PENDING) so it's not eligible; B's predecessor A
    # is not COMPLETED so B is DAG-blocked. Zero eligible tasks.
    plan = _plan(
        Task(id="A", title="Plan the work"),
        Task(id="B", title="Execute the work"),
        edges=[TaskEdge(from_task_id="A", to_task_id="B")],
    )
    plan = dataclasses.replace(
        plan,
        tasks=(
            dataclasses.replace(plan.tasks[0], status=TaskStatus.RUNNING),
            plan.tasks[1],
        ),
    )
    session = _session_with(plan)

    sink = InMemorySink()
    steerer = _SinkingSteerer(sink)

    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    agent_tool = _FakeAgentTool("worker")
    await plugin.before_tool_callback(
        tool=agent_tool,
        tool_args={"request": "go"},
        tool_context=coord_tool_context,
    )

    # Pin did not bind anything.
    assert session.current_task_id == ""

    # Exactly one delegation_observed event, with empty task_id.
    delegations = _delegation_events(sink.events)
    assert len(delegations) == 1
    d = delegations[0]
    assert d.from_agent == "coord"
    assert d.to_agent == "worker"
    assert d.task_id == "", (
        f"expected empty task_id when no eligible task; got '{d.task_id}'"
    )


async def test_capability_check_still_resolves_after_reorder() -> None:
    """After the pin → emit → capability-check reorder, the capability
    detector still resolves the task via Strategy 1 (assignee, freshly
    stamped by the pin) — i.e. the reorder didn't break the goldfive#253
    detector.

    Rule A scenario: the invoked sub-agent has only AgentTool wrappers
    (no leaf tools) and the bound plan task is a leaf authoring task.
    The capability detector must fire CAPABILITY_MISMATCH.
    """
    from goldfive.types import DriftKind

    class _RecordingDrift:
        def __init__(self) -> None:
            self.drifts: list[Any] = []

        async def observe(self, *a: Any, **kw: Any) -> None:
            pass

        def detect_drift(self, *a: Any, **kw: Any) -> None:
            return None

        async def handle_drift(self, drift: Any, session: Any) -> None:  # noqa: ARG002
            self.drifts.append(drift)

    class _RecordingSteerer:
        def __init__(self) -> None:
            self._sinks: list[Any] = []
            self.drift = _RecordingDrift()

        @property
        def drifts(self) -> list[Any]:
            return self.drift.drifts

        async def transition(self, *a: Any, **kw: Any) -> None:
            pass

        def bind(self, **kw: Any) -> None:
            pass

    plugin = make_adk_plugin(host_agent_name="coord")
    # Single leaf authoring task (PENDING, no assignee) — the pin will
    # bind it to the invoked underqualified sub-agent.
    plan = _plan(
        Task(id="t-draft", title="Draft a presentation about LLM observability"),
    )
    session = _session_with(plan)

    steerer = _RecordingSteerer()
    ctx_obj = SessionContext(
        session=session,
        steerer=steerer,
        task=None,
        tool_handlers={},
        host_agent_name="coord",
    )
    plugin.set_active_context(ctx_obj)

    # The "underqualified" invoked agent has only an AgentTool wrapper
    # — Rule A's structural signal. Build a stand-in shape the
    # capability detector can introspect (it walks ``.tools`` on the
    # invoked agent).
    class _AgentToolWrapper:
        def __init__(self, name: str) -> None:
            self.name = name
            self.agent = _FakeAgent(name)

    class _Underqualified:
        name = "underqualified"
        tools = [_AgentToolWrapper("inner")]

    # AgentTool dispatch shape the plugin recognises.
    class _DispatchAgentTool:
        def __init__(self, sub: Any) -> None:
            self.name = sub.name
            self.agent = sub

    dispatch = _DispatchAgentTool(_Underqualified())

    adk_state: dict[str, Any] = {SESSION_CONTEXT_STATE_KEY: ctx_obj}
    coord_inv = _FakeInvocationContext(adk_state, "coord")
    coord_tool_context = _FakeToolContext(coord_inv, function_call_id="fc-1")

    await plugin.before_tool_callback(
        tool=dispatch,
        tool_args={"request": "draft the presentation"},
        tool_context=coord_tool_context,
    )

    # Pin landed: assignee stamped, current_task_id pinned — Strategy 1
    # of the capability check resolves on this.
    t = _find_task(session.plan, "t-draft")
    assert t is not None and t.assignee_agent_id == "underqualified"
    assert session.current_task_id == "t-draft"

    # Capability detector fired and the drift reached the steerer.
    capability_drifts = [
        d for d in steerer.drifts if d.kind is DriftKind.CAPABILITY_MISMATCH
    ]
    assert len(capability_drifts) >= 1, (
        f"expected CAPABILITY_MISMATCH; got {[d.kind for d in steerer.drifts]}"
    )
    drift = capability_drifts[0]
    assert drift.current_task_id == "t-draft"
    assert drift.current_agent_id == "underqualified"
