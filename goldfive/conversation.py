"""Cross-turn conversation state for :class:`goldfive.Runner`.

A :class:`Conversation` persists state across successive
``runner.run()`` calls on the same Runner instance. Each ``run()`` is
still a distinct goldfive *run* with its own ``run_id`` and its own
``RunStarted``/``RunCompleted`` lifecycle — but the Runner seeds each
turn's :class:`Session` from the Conversation, then folds the turn's
outcome back in when it finishes. This gives the planner access to
prior-turn ``completed_results`` and an append-only history of goals,
which is the primary UX win: a user can say "make it funnier" on
turn 2 and the planner sees turn 1's output as context.

Phase 3 scope (see issue #78):

* ``session.goals`` accumulates across turns (deduplicated by id).
* ``session.completed_results`` carries over from prior turns.
* ``session.plan`` resets per turn. Cross-turn plan lineage is v2.
* :class:`TurnRecord` captures a one-sentence summary of each turn so
  the planner can reference them in its prompt.

A Runner always owns a Conversation — single-turn callers never notice
because the Conversation is fresh on construction and discarded with
the Runner.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import TYPE_CHECKING

from goldfive.types import Goal, Session

if TYPE_CHECKING:
    from goldfive.results import ExecutionOutcome


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclasses.dataclass
class TurnRecord:
    """Summary of one completed turn within a :class:`Conversation`.

    Stored on the Conversation and surfaced to the planner as
    prior-turn context on subsequent turns. Keep fields compact —
    these are serialised into planner prompts.
    """

    run_id: str
    user_input_summary: str = ""
    plan_summary: str = ""
    outcome_success: bool = True
    outcome_reason: str = ""
    completed_task_ids: list[str] = dataclasses.field(default_factory=list)
    started_at_ms: int = 0
    ended_at_ms: int = 0


@dataclasses.dataclass
class Conversation:
    """Cross-turn state owned by a :class:`~goldfive.Runner`.

    Attributes
    ----------
    id:
        Stable identifier for the conversation. Survives across turns;
        only changes when the caller invokes
        :meth:`Runner.new_conversation`.
    started_at_ms:
        Wall-clock ms when the Conversation was constructed.
    goals:
        Append-only list of goals seen across every turn. Deduplicated
        by ``Goal.id`` so a goal that is genuinely re-stated on a later
        turn does not appear twice.
    completed_results:
        Merged ``task_id -> summary`` map across every turn. Each new
        turn sees prior turns' completed task outputs as planner
        context.
    turns:
        Log of :class:`TurnRecord` entries, one per completed turn.
        Used by :meth:`prior_turn_context` to give the planner a
        capped recent-history window.
    """

    id: str
    started_at_ms: int
    goals: list[Goal] = dataclasses.field(default_factory=list)
    completed_results: dict[str, str] = dataclasses.field(default_factory=dict)
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)

    @classmethod
    def new(cls) -> Conversation:
        """Build a fresh, empty Conversation with a new UUID."""
        return cls(id=uuid.uuid4().hex, started_at_ms=_now_ms())

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def next_turn_session(self) -> Session:
        """Build a :class:`Session` seeded with cross-turn state.

        The returned Session has a fresh ``run_id``, inherits the
        Conversation's ``id`` as ``conversation_id``, and copies
        (not aliases) the accumulated ``goals`` and ``completed_results``
        so the executor's in-turn mutations do not retroactively
        rewrite the Conversation's record.
        """
        return Session(
            run_id=uuid.uuid4().hex,
            conversation_id=self.id,
            started_at_ms=_now_ms(),
            goals=list(self.goals),
            completed_results=dict(self.completed_results),
        )

    def absorb_turn(
        self,
        outcome: ExecutionOutcome,
        *,
        user_input_summary: str = "",
    ) -> TurnRecord:
        """Fold a turn's outcome back into the Conversation.

        Called by the Runner after ``executor.run`` returns (successful
        or not — we record the failure so the next turn's planner can
        see it). Returns the newly-appended :class:`TurnRecord`.
        """
        session = outcome.session
        # Merge goals by id so a restated goal doesn't duplicate.
        seen_ids = {g.id for g in self.goals if g.id}
        for g in session.goals:
            if g.id and g.id in seen_ids:
                continue
            self.goals.append(g)
            if g.id:
                seen_ids.add(g.id)

        # Completed results merge: later turns win on id collisions so
        # that a revised result on a follow-up turn is visible to the
        # turn after it.
        self.completed_results.update(session.completed_results)

        plan_summary = ""
        completed_ids: list[str] = []
        if session.plan is not None:
            plan_summary = session.plan.summary or ""
            completed_ids = [
                t.id
                for t in session.plan.tasks
                if t.status.value == "COMPLETED" and t.id
            ]

        record = TurnRecord(
            run_id=session.run_id,
            user_input_summary=user_input_summary,
            plan_summary=plan_summary,
            outcome_success=bool(outcome.success),
            outcome_reason=outcome.reason or "",
            completed_task_ids=completed_ids,
            started_at_ms=session.started_at_ms,
            ended_at_ms=_now_ms(),
        )
        self.turns.append(record)
        return record

    # ------------------------------------------------------------------
    # Planner context
    # ------------------------------------------------------------------

    def prior_turn_context(self, *, recent_turns: int = 3) -> dict[str, object]:
        """Return the cross-turn context dict passed to the planner.

        ``recent_turns`` caps how many ``TurnRecord`` entries are
        forwarded so prompts do not grow unbounded on very long
        conversations. The returned dict is safe to merge into the
        caller's own planner context via ``dict.update``.
        """
        if recent_turns < 0:
            recent_turns = 0
        window = self.turns[-recent_turns:] if recent_turns else []
        return {
            "conversation_id": self.id,
            "prior_completed_results": dict(self.completed_results),
            "prior_turns": [
                {
                    "run_id": t.run_id,
                    "user_input_summary": t.user_input_summary,
                    "plan_summary": t.plan_summary,
                    "outcome_success": t.outcome_success,
                    "outcome_reason": t.outcome_reason,
                    "completed_task_ids": list(t.completed_task_ids),
                }
                for t in window
            ],
            "turn_index": len(self.turns),
        }


__all__ = ["Conversation", "TurnRecord"]
