"""Validator tests for the corrective-predecessor `supersedes` invariant.

Covers ``Plan.validate(for_revision=True, prior=...)`` step 8
(``PLAN-LIFECYCLE.md §3.6``, goldfive#248). The motivating bug: a
corrective task ``fix_research_X`` was inserted as an independent
root while the prior PENDING ``draft_slides`` stayed reachable from
the (now-suspect) COMPLETED ``research_X``. The reconciler claimed
``draft_slides`` for in-flight drafting work while ``fix_research_X``
sat unrun — out-of-order plan execution.

These tests pin the validator semantics:

* Valid corrective predecessor shapes (Shape A: keep Y + edge X->Y;
  Shape B: re-edge every Y consumer through X) are accepted.
* Independent-root insertions (the bug shape) are rejected.
* The existing #214 REPLACE / CORRECT semantics for *terminal*
  supersedes targets remain untouched (no-op for §3.6).
* Self-supersedes and mutual-supersedes pairs are rejected.
* Empty supersedes (default) imposes no new constraint
  (back-compat for plans / tests that do not use the link at all).
"""

from __future__ import annotations

import pytest

from goldfive.planner import LLMPlanner
from goldfive.types import (
    Plan,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)


def _plan(
    tasks: list[Task],
    edges: list[TaskEdge],
    *,
    plan_id: str = "p",
    summary: str = "",
) -> Plan:
    return Plan(
        id=plan_id,
        run_id="r1",
        goal_ids=["g1"],
        tasks=tasks,
        edges=edges,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Case 1: valid supersedes of non-terminal task with full re-edge (Shape B)
# ---------------------------------------------------------------------------


def test_supersedes_nonterminal_full_reedge_accepted() -> None:
    """Shape B: prior Y -> Z, revision X -> Z (with X.supersedes = Y).

    Y stays as PENDING but is no longer the gating predecessor of Z;
    X is the new gate. This is the canonical corrective-predecessor
    shape the validator must accept.
    """
    prior = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.PENDING),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="research_X", to_task_id="draft_slides")],
    )
    revised = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.PENDING),
            Task(
                id="fix_research_X",
                title="Re-research X",
                status=TaskStatus.PENDING,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[
            # Re-edge every prior consumer of research_X through fix_research_X.
            TaskEdge(from_task_id="fix_research_X", to_task_id="draft_slides"),
        ],
    )

    revised.validate(for_revision=True, prior=prior)


# ---------------------------------------------------------------------------
# Case 2: invalid — supersedes set but downstream consumer not re-edged
# (the tomato-e2e bug shape)
# ---------------------------------------------------------------------------


def test_supersedes_nonterminal_independent_root_rejected() -> None:
    """The #248 motivating bug: fix_X as independent root, draft_slides
    still depends on the (now-suspect) X. Validator must REJECT.
    """
    prior = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.PENDING),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="research_X", to_task_id="draft_slides")],
    )
    revised = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.PENDING),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
            Task(
                id="fix_research_X",
                title="Re-research X",
                status=TaskStatus.PENDING,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        # Prior research_X -> draft_slides edge preserved; fix_research_X
        # is an independent root with no edge into the rest of the DAG.
        edges=[TaskEdge(from_task_id="research_X", to_task_id="draft_slides")],
    )

    with pytest.raises(ValueError) as exc_info:
        revised.validate(for_revision=True, prior=prior)

    msg = str(exc_info.value)
    assert "fix_research_X" in msg
    assert "research_X" in msg
    assert "not re-edged through" in msg


# ---------------------------------------------------------------------------
# Case 3: valid supersedes of TERMINAL task (#214 REPLACEMENT case)
# ---------------------------------------------------------------------------


def test_supersedes_terminal_replacement_accepted() -> None:
    """Y is terminal (FAILED) → §3.6 is a no-op; the existing #214
    REPLACEMENT topology is allowed without re-edge requirements."""
    prior = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.FAILED),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    revised = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.FAILED),
            Task(
                id="fix_research_X",
                title="Re-research X",
                status=TaskStatus.PENDING,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        # No edges from FAILED research_X (would violate §137 anyway).
        # No re-edge required by §3.6 because Y is terminal.
        edges=[
            TaskEdge(from_task_id="fix_research_X", to_task_id="draft_slides"),
        ],
    )

    revised.validate(for_revision=True, prior=prior)


def test_supersedes_terminal_correct_completed_accepted() -> None:
    """Y is terminal (COMPLETED) with the #214 CORRECT-kind shape:
    prior research_X COMPLETED -> draft_slides PENDING; revision adds
    fix_research_X with supersedes=research_X and an edge
    research_X -> fix_research_X (the historical-completed-then-correct
    DAG). §3.6 must NOT fire — Y is terminal — so the existing CORRECT
    path remains untouched.
    """
    prior = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.COMPLETED),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="research_X", to_task_id="draft_slides")],
    )
    revised = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.COMPLETED),
            Task(
                id="fix_research_X",
                title="Re-research X",
                status=TaskStatus.PENDING,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[
            # CORRECT: research_X COMPLETED stays; new edge research_X -> fix_research_X
            # injects fix_research_X as a child of the corrected work.
            TaskEdge(from_task_id="research_X", to_task_id="fix_research_X"),
            TaskEdge(from_task_id="fix_research_X", to_task_id="draft_slides"),
        ],
    )

    # §3.6 must be a no-op (research_X terminal); the existing #214
    # CORRECT topology validates cleanly.
    revised.validate(for_revision=True, prior=prior)


# ---------------------------------------------------------------------------
# Case 4: Shape A — keep Y, prepend X via X -> Y
# ---------------------------------------------------------------------------


def test_supersedes_nonterminal_shape_a_prepend_accepted() -> None:
    """Alternative valid shape: Y stays PENDING, X gates Y via X -> Y.
    Y's prior downstream edges (Y -> Z) remain unchanged."""
    prior = _plan(
        tasks=[
            Task(id="Y_task", title="Y", status=TaskStatus.PENDING),
            Task(id="Z_task", title="Z", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="Y_task", to_task_id="Z_task")],
    )
    revised = _plan(
        tasks=[
            Task(id="Y_task", title="Y", status=TaskStatus.PENDING),
            Task(id="Z_task", title="Z", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X (clarifying step)",
                status=TaskStatus.PENDING,
                supersedes="Y_task",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="X_task", to_task_id="Y_task"),
            TaskEdge(from_task_id="Y_task", to_task_id="Z_task"),
        ],
    )

    revised.validate(for_revision=True, prior=prior)


# ---------------------------------------------------------------------------
# Case 5: Multiple downstream consumers of Y — all must be re-edged
# ---------------------------------------------------------------------------


def test_supersedes_nonterminal_partial_reedge_rejected() -> None:
    """If Y has TWO downstream consumers Z1, Z2 in prior and the
    revision re-edges only Z1 through X (leaving Z2 still gated on
    the now-suspect Y), the validator REJECTS."""
    prior = _plan(
        tasks=[
            Task(id="Y_task", title="Y", status=TaskStatus.PENDING),
            Task(id="Z1", title="Z1", status=TaskStatus.PENDING),
            Task(id="Z2", title="Z2", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="Y_task", to_task_id="Z1"),
            TaskEdge(from_task_id="Y_task", to_task_id="Z2"),
        ],
    )
    revised = _plan(
        tasks=[
            Task(id="Y_task", title="Y", status=TaskStatus.PENDING),
            Task(id="Z1", title="Z1", status=TaskStatus.PENDING),
            Task(id="Z2", title="Z2", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X corrective",
                status=TaskStatus.PENDING,
                supersedes="Y_task",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        # Only Z1 re-edged through X; Z2 still hangs off Y_task only.
        edges=[
            TaskEdge(from_task_id="Y_task", to_task_id="Z1"),
            TaskEdge(from_task_id="Y_task", to_task_id="Z2"),
            TaskEdge(from_task_id="X_task", to_task_id="Z1"),
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        revised.validate(for_revision=True, prior=prior)

    msg = str(exc_info.value)
    assert "X_task" in msg
    assert "Y_task" in msg
    assert "Z2" in msg


# ---------------------------------------------------------------------------
# Case 6: Empty supersedes (default) — no constraint applied (back-compat)
# ---------------------------------------------------------------------------


def test_empty_supersedes_no_constraint() -> None:
    """A revision that adds new PENDING tasks without using `supersedes`
    at all must validate cleanly (no false positives on plans that
    pre-date the link or simply don't need it)."""
    prior = _plan(
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    revised = _plan(
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
            # Brand-new task, no supersedes, plugged in as a fresh root.
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t1", to_task_id="t3"),
        ],
    )

    revised.validate(for_revision=True, prior=prior)


# ---------------------------------------------------------------------------
# Case 7: supersedes naming a non-existent task — rejected
# ---------------------------------------------------------------------------


def test_supersedes_unknown_target_rejected() -> None:
    """X.supersedes references a task id that exists in neither
    `prior` nor the revision — dangling reference, validator rejects."""
    prior = _plan(
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
    )
    revised = _plan(
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X",
                status=TaskStatus.PENDING,
                supersedes="nonexistent_id",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
    )

    with pytest.raises(ValueError) as exc_info:
        revised.validate(for_revision=True, prior=prior)

    msg = str(exc_info.value)
    assert "X_task" in msg
    assert "nonexistent_id" in msg


# ---------------------------------------------------------------------------
# Case 8: cycle prevention — X.supersedes=Y AND Y.supersedes=X
# ---------------------------------------------------------------------------


def test_mutual_supersedes_rejected() -> None:
    """Mutual supersession (X<->Y) is structurally meaningless. Both
    directions seen in the same revision must be rejected."""
    prior = _plan(
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
    )
    revised = _plan(
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X",
                status=TaskStatus.PENDING,
                supersedes="Y_task",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(
                id="Y_task",
                title="Y",
                status=TaskStatus.PENDING,
                supersedes="X_task",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
    )

    with pytest.raises(ValueError) as exc_info:
        revised.validate(for_revision=True, prior=prior)

    msg = str(exc_info.value)
    assert "mutual supersedes" in msg
    assert "X_task" in msg
    assert "Y_task" in msg


def test_self_supersedes_rejected() -> None:
    """A task cannot supersede itself."""
    prior = _plan(
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
    )
    revised = _plan(
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X",
                status=TaskStatus.PENDING,
                supersedes="X_task",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
    )

    with pytest.raises(ValueError) as exc_info:
        revised.validate(for_revision=True, prior=prior)

    msg = str(exc_info.value)
    assert "X_task" in msg
    assert "supersedes itself" in msg


# ---------------------------------------------------------------------------
# Case 9: Refine prompt's invariants block surfaces the new contract
# ---------------------------------------------------------------------------


def test_refine_prompt_invariants_block_includes_corrective_predecessor() -> None:
    """Render the structural-invariants block on a representative plan
    and assert the §3.6 contract appears as item 6 with the key
    instructional language."""
    plan = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.COMPLETED),
            Task(id="draft_slides", title="Draft slides", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="research_X", to_task_id="draft_slides")],
    )
    block = LLMPlanner._render_structural_invariants_block(plan)

    # The new invariant is item 6 in the rendered block, after the
    # cycle-free invariant.
    assert "6. CORRECTIVE PREDECESSORS" in block
    assert "supersedes" in block
    # Both shapes are documented to the LLM.
    assert "structural predecessor" in block
    assert "every prior consumer" in block.lower() or "every prior consumer" in block
    # The reject message language matches the validator's phrasing
    # so retry prompts can correlate.
    assert "REJECTED" in block


# ---------------------------------------------------------------------------
# Case 10: bonus — Shape B with Y dropped from the revision is accepted
# ---------------------------------------------------------------------------


def test_supersedes_y_advanced_to_terminal_in_revision_accepted() -> None:
    """Chained-revision case: prior had a valid Shape A (Y stayed
    PENDING + edge X->Y). The next revision runs further and Y has
    advanced to COMPLETED (X completed Y's gating work). The
    revision should validate even though shape_a_ok now fails (Y
    is no longer PENDING) and shape_b_ok would still require
    re-edges that the Shape A topology never had — the corrective
    ordering has been resolved in time and the graph is consistent.
    """
    prior = _plan(
        tasks=[
            Task(id="research_X", title="Research X", status=TaskStatus.PENDING),
            Task(
                id="fix_research_X",
                title="Fix X",
                status=TaskStatus.PENDING,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="draft_slides", title="Draft", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="fix_research_X", to_task_id="research_X"),
            TaskEdge(from_task_id="research_X", to_task_id="draft_slides"),
        ],
    )
    revised = _plan(
        tasks=[
            # Y advanced to COMPLETED in this revision; corrective
            # X also completed.
            Task(id="research_X", title="Research X", status=TaskStatus.COMPLETED),
            Task(
                id="fix_research_X",
                title="Fix X",
                status=TaskStatus.COMPLETED,
                supersedes="research_X",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="draft_slides", title="Draft", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="fix_research_X", to_task_id="research_X"),
            TaskEdge(from_task_id="research_X", to_task_id="draft_slides"),
        ],
    )

    revised.validate(for_revision=True, prior=prior)


def test_supersedes_nonterminal_y_dropped_with_full_reedge_accepted() -> None:
    """A revision may drop Y entirely (X completely subsumes it) so
    long as every Y-consumer is re-edged through X. Y must not be
    referenced by the remaining edges."""
    prior = _plan(
        tasks=[
            Task(id="Y_task", title="Y", status=TaskStatus.PENDING),
            Task(id="Z_task", title="Z", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="Y_task", to_task_id="Z_task")],
    )
    revised = _plan(
        tasks=[
            # Y_task dropped from the revision.
            Task(id="Z_task", title="Z", status=TaskStatus.PENDING),
            Task(
                id="X_task",
                title="X corrective",
                status=TaskStatus.PENDING,
                supersedes="Y_task",  # Y resolves in prior
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[TaskEdge(from_task_id="X_task", to_task_id="Z_task")],
    )

    revised.validate(for_revision=True, prior=prior)
