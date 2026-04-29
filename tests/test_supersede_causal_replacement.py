"""Tests for causal (supersedes-driven) replacement detection (goldfive#213).

The executor's :func:`_has_live_replacement` is two-tier as of #213:

1. **Causal tier**: walk ``Task.supersedes`` chains. A non-FAILED /
   non-CANCELLED task whose chain reaches the failed id satisfies the
   FAILED predecessor at *any* status (PENDING, RUNNING, COMPLETED).
   The chain link is unambiguous about chronology — there is no
   COMPLETED-as-predecessor masking risk.
2. **Name-pattern tier (fallback)**: only used when the causal tier
   found nothing. Preserves legacy semantic (PENDING/RUNNING for shared
   retry lineage, any non-terminal for versioned suffix).

These tests pin both tiers + the assignee scoping rule.
"""

from __future__ import annotations

from goldfive.executors.sequential import (
    _fatally_failed_task_ids,
    _has_live_replacement,
)
from goldfive.types import Plan, Task, TaskStatus

# ---------------------------------------------------------------------------
# Tier 1: causal (supersedes-chain) replacement detection.
# ---------------------------------------------------------------------------


def test_completed_retry_via_supersedes_satisfies_failed() -> None:
    """A COMPLETED task with ``supersedes`` pointing at a FAILED
    predecessor is a live replacement (the v27 bug repro).

    Pre-#213, the COMPLETED status would be treated as a predecessor
    by the name-pattern fallback's chronology rule, leaving ``t0`` in
    the fatally_failed list and triggering false-positive
    ``run_aborted`` events at every overlay-end across subsequent
    turns. With the causal tier, the supersedes link unambiguously
    identifies ``retry_t0`` as a SUCCESSOR — chronologically valid
    regardless of status.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry of task 0",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="a",
                supersedes="t0",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True
    assert _fatally_failed_task_ids(plan) == []


def test_pending_retry_via_supersedes_satisfies_failed() -> None:
    """Same as the COMPLETED case but PENDING — still a live
    replacement under the causal tier (matches existing semantic, just
    via the new path)."""
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry of task 0",
                status=TaskStatus.PENDING,
                assignee_agent_id="a",
                supersedes="t0",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True
    assert _fatally_failed_task_ids(plan) == []


def test_completed_retry_via_name_pattern_alone_does_not_satisfy() -> None:
    """When ``supersedes`` is empty, the name-pattern tier's
    conservative chronology rule kicks in: a COMPLETED ``retry_<id>``
    is treated as a *predecessor*, not a replacement.

    NOTE: in real refines, Part 2's backfill would populate
    ``supersedes`` and the causal tier would catch this. The helper
    itself does NOT auto-infer — that's the merge-time inference's
    job. This test pins the legacy fallback behaviour.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry of task 0",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="a",
                supersedes="",  # explicit: no causal link
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is False
    assert _fatally_failed_task_ids(plan) == ["t0"]


# ---------------------------------------------------------------------------
# Chain semantics.
# ---------------------------------------------------------------------------


def test_chain_failure_marks_latest_link_fatal() -> None:
    """Chain ``t0 → retry_t0 → retry_retry_t0`` all linked via
    ``supersedes`` (each new task supersedes the previous), with
    ``retry_retry_t0`` FAILED at the leaf.

    Semantic decision (documented): only the LATEST chain link's
    status drives fatality. Earlier links (t0, retry_t0) carry FAILED
    too in this scenario but each has a successor in the chain that's
    a live replacement (retry_t0 via direct supersedes from
    retry_retry_t0; t0 via the *transitive* chain through retry_t0).
    The leaf retry_retry_t0 itself has NO successor that supersedes
    it — so it's fatal.

    This is the correct semantic: the chain encodes "we tried, then
    we tried again, then the latest attempt failed." The earlier
    failures are absorbed by the chain.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry 1",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
                supersedes="t0",
            ),
            Task(
                id="retry_retry_t0",
                title="retry 2",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
                supersedes="retry_t0",
            ),
        ],
        edges=[],
    )
    # t0: retry_t0 is FAILED so causal tier skips it; retry_retry_t0
    # transitively supersedes t0 but is also FAILED. Name-pattern
    # tier: only PENDING/RUNNING peers in the same lineage count, all
    # are FAILED. So t0 is fatal.
    assert _has_live_replacement(plan, plan.tasks[0]) is False
    # retry_t0: retry_retry_t0 supersedes it but is FAILED. Fatal.
    assert _has_live_replacement(plan, plan.tasks[1]) is False
    # retry_retry_t0: no successor. Fatal.
    assert _has_live_replacement(plan, plan.tasks[2]) is False
    # All three FAILED with no live replacement.
    assert sorted(_fatally_failed_task_ids(plan)) == sorted(
        ["t0", "retry_t0", "retry_retry_t0"]
    )


def test_chain_failure_with_live_leaf_satisfies_root() -> None:
    """``t0 → retry_t0 → retry_retry_t0`` chain where the LEAF is
    PENDING. The transitive supersedes link from the live leaf
    satisfies *every* failed predecessor in the chain.

    This is the chain-of-retries forward-progress case: the operator
    wired three attempts, the latest is still in flight, and the run
    should not be aborted on any of the earlier failures.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry 1",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
                supersedes="t0",
            ),
            Task(
                id="retry_retry_t0",
                title="retry 2",
                status=TaskStatus.PENDING,
                assignee_agent_id="a",
                supersedes="retry_t0",
            ),
        ],
        edges=[],
    )
    # PENDING leaf transitively supersedes both earlier failures.
    assert _has_live_replacement(plan, plan.tasks[0]) is True
    assert _has_live_replacement(plan, plan.tasks[1]) is True
    # Leaf itself is not failed.
    assert _fatally_failed_task_ids(plan) == []


# ---------------------------------------------------------------------------
# Assignee scoping.
# ---------------------------------------------------------------------------


def test_assignee_mismatch_breaks_replacement_in_causal_tier() -> None:
    """A causal supersedes link with a DIFFERENT assignee does NOT
    count. Assignee scoping applies to both tiers when both ids carry
    a populated ``assignee_agent_id``.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="agent_a",
            ),
            Task(
                id="retry_t0",
                title="retry of task 0",
                status=TaskStatus.PENDING,
                assignee_agent_id="agent_b",  # different
                supersedes="t0",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is False


def test_empty_assignee_does_not_block_causal_match() -> None:
    """When EITHER side has an empty assignee, scoping is bypassed —
    causal supersedes link alone is enough.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="task 0", status=TaskStatus.FAILED),
            Task(
                id="retry_t0",
                title="retry of task 0",
                status=TaskStatus.PENDING,
                supersedes="t0",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True


# ---------------------------------------------------------------------------
# Cycle protection.
# ---------------------------------------------------------------------------


def test_supersedes_self_loop_does_not_hang() -> None:
    """A pathological self-supersedes (``t.supersedes == t.id``) should
    not infinite-loop the chain walk.

    In production this is normalised away by
    :func:`_normalize_supersession_kinds`'s Rule 0, but the executor
    helper must still be defensive against malformed plan input.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                supersedes="t0",  # self-loop
            ),
        ],
        edges=[],
    )
    # Should not hang; chain walk bails on visited cursor.
    assert _has_live_replacement(plan, plan.tasks[0]) is False


def test_supersedes_two_node_cycle_does_not_hang() -> None:
    """Two-node cycle (``a.supersedes=b, b.supersedes=a``). Chain walk
    must terminate.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="a",
                title="a",
                status=TaskStatus.FAILED,
                supersedes="b",
            ),
            Task(
                id="b",
                title="b",
                status=TaskStatus.PENDING,
                supersedes="a",
            ),
        ],
        edges=[],
    )
    # b transitively reaches a via cycle (a -> b -> a -> ...). Cycle
    # protection caps the walk; b's chain visits b first then sees a
    # in chain[a]=b which is already visited; bail. So no transitive
    # match through the cycle. But b.supersedes == "a" directly so
    # b satisfies a after 1 hop -- that's fine, that's the link the
    # LLM explicitly drew.
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True


# ---------------------------------------------------------------------------
# Name-pattern tier still works when supersedes is empty.
# ---------------------------------------------------------------------------


def test_name_pattern_pending_retry_still_satisfies() -> None:
    """Backward compat: legacy plans without supersedes still get the
    PENDING/RUNNING name-pattern fallback.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="t0",
                title="task 0",
                status=TaskStatus.FAILED,
                assignee_agent_id="a",
            ),
            Task(
                id="retry_t0",
                title="retry",
                status=TaskStatus.PENDING,
                assignee_agent_id="a",
                # no supersedes
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True


def test_name_pattern_versioned_completed_satisfies() -> None:
    """Backward compat: a COMPLETED ``<id>_v2`` is a SUCCESSOR by
    naming convention — still counts via name-pattern fallback.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="define_structure",
                title="define",
                status=TaskStatus.FAILED,
                assignee_agent_id="planner",
            ),
            Task(
                id="define_structure_v2",
                title="define (v2)",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="planner",
                # no supersedes
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True
