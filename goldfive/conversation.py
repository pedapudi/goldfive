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
* ``session.plan`` is seeded each turn from the prior turn's stash
  (via :meth:`Conversation.prior_plan_for`) so :meth:`Planner.handle_turn`
  always sees a non-None prior. The carry-forward is gated on
  ``session_id`` pin state so a Runner shared across multiple outer
  ADK sessions does not leak one session's plan into another's first
  turn (goldfive#271 follow-up; pre-fix this lived on
  ``Runner._last_plan`` and was process-scoped).
* :class:`TurnRecord` captures a one-sentence summary of each turn so
  the planner can reference them in its prompt.

A Runner owns a *map* of Conversations keyed by outer-session id
(``Runner._conversations``) so cross-turn state never leaks across
distinct outer ADK sessions sharing one Runner — see goldfive#271
follow-up to PR #293 / validation v4 Class 1. Programmatic
(unpinned) callers all share one Conversation under the empty-string
key, preserving pre-#161 single-Conversation continuity. Each
pinned outer-session id (e.g. ADK-web's ``ctx.session.id``) gets
its own dedicated Conversation with isolated ``goals``,
``completed_results``, ``turns``, and ``_next_sequence`` cursor.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import TYPE_CHECKING

from goldfive.types import Goal, Plan, Session

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
    # zicato#12: merged ``task_id -> full actual output`` map across turns,
    # the cross-turn mirror of :attr:`goldfive.types.Session.completed_outputs`.
    # Carried alongside ``completed_results`` (the self-reported summary) so a
    # later turn / planner / grader sees prior turns' real output, not only the
    # summary.
    completed_outputs: dict[str, str] = dataclasses.field(default_factory=dict)
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)
    # Conversation-level wire sequence cursor (goldfive#271 Gap 2).
    # Each turn's :class:`Session` seeds its private ``_next_sequence``
    # from this value, then on :meth:`absorb_turn` writes the post-turn
    # high-water mark back. Reason: when goldfive#161's outer-session
    # pin causes ``Session.run_id`` to repeat across turns,
    # harmonograf's ``goldfive_events`` PK ``(session_id, run_id,
    # sequence)`` collides on the second turn's early events (sequence
    # 0, 1, 2, ...) which match turn 1's already-persisted rows. The
    # storage layer's ``INSERT OR IGNORE`` then silently drops the
    # second turn's plan_submitted, agent_invocation_started, etc.
    # Carrying a Conversation-level cursor across turns makes
    # ``sequence`` unique per (conversation, run_id-or-session_id) so
    # the persisted-event keyspace stays collision-free even under the
    # outer-session pin. Single-turn callers see no change: the cursor
    # starts at 0 on a fresh Conversation.
    _next_sequence: int = 0
    # Most recently absorbed turn's plan. Used by the Runner to seed
    # the next turn's ``session.plan`` so :meth:`Planner.handle_turn`
    # always sees a non-None prior. Pre-fix this lived on
    # :class:`Runner` as ``_last_plan`` — a process-scoped attribute
    # that leaked across outer ADK sessions sharing one Runner
    # (validation v4 Class 1, goldfive#271 follow-up). The seed lookup
    # (:meth:`prior_plan_for`) gates carry-forward on the session-id
    # bookkeeping below so an ADK-pinned session boundary defeats the
    # carry-forward, while a programmatic Runner caller (no pin) still
    # gets Conversation-level continuity.
    _last_plan: Plan | None = None
    # Session id (== ``Session.run_id`` after any
    # :meth:`Runner.run(session_id=...)` outer-session pin) of the
    # turn whose plan is held in ``_last_plan``. Default ``""`` means
    # no plan has been stashed yet — a brand-new Conversation, the
    # state immediately after :meth:`Runner.new_conversation`.
    _last_plan_session_id: str = ""
    # Whether the turn that produced ``_last_plan`` had its session
    # id explicitly pinned via :meth:`Runner.run(session_id=...)`.
    # The seed lookup uses this together with the new turn's pin
    # state to decide whether the session-id check applies:
    # programmatic (unpinned) callers get unconditional Conversation-
    # level carry-forward; ADK (pinned) callers must match the prior
    # turn's pinned id exactly.
    _last_plan_pinned: bool = False

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

        The Session's wire-sequence counter (``Session._next_sequence``)
        is seeded from the Conversation's running cursor so per-turn
        events are globally unique within the conversation; see the
        ``_next_sequence`` docstring on :class:`Conversation`.
        """
        return Session(
            run_id=uuid.uuid4().hex,
            conversation_id=self.id,
            started_at_ms=_now_ms(),
            goals=list(self.goals),
            completed_results=dict(self.completed_results),
            completed_outputs=dict(self.completed_outputs),
            _next_sequence=self._next_sequence,
        )

    def stash_plan(self, session: Session, *, pinned: bool = False) -> None:
        """Stash ``session.plan`` keyed by ``session.id`` for the next turn.

        ``pinned`` records whether the turn that produced this plan
        had its session id explicitly pinned via
        :meth:`Runner.run(session_id=...)`. The next turn's
        :meth:`prior_plan_for` lookup combines that flag with the
        next turn's own pin state to decide whether carry-forward
        applies; see the lookup docstring for the matrix.

        The Runner calls this from its ``finally`` block so the stash
        runs even on ``BaseException`` (e.g. ``CancelledError`` from
        ADK closing the runner mid-stream) — the rationale is the
        same as goldfive#271 Gap 1's original Runner-side stash. The
        idempotency contract: a subsequent :meth:`absorb_turn` on the
        same session re-stashes the same (plan, session id, pinned)
        tuple, so callers that hit both paths produce identical state.

        Skipped (no-op) when ``session.plan`` is ``None`` or empty —
        an empty plan is not a meaningful prior to carry forward.
        """
        if session.plan is not None and session.plan.tasks:
            self._last_plan = session.plan
            self._last_plan_session_id = session.id
            self._last_plan_pinned = pinned

    def prior_plan_for(self, session_id: str, *, pinned: bool = False) -> Plan | None:
        """Return the stashed prior plan iff carry-forward applies.

        Called by :meth:`Runner.run` to seed ``session.plan`` at the
        start of a turn AFTER any
        :meth:`Runner.run(session_id=...)` outer-session pin has
        finalised the new turn's session id. ``pinned`` records
        whether the new turn's caller passed an explicit
        ``session_id`` pin.

        Carry-forward matrix:

        ===========  ===========  =============================
        prior turn   new turn     carry forward?
        ===========  ===========  =============================
        unpinned     unpinned     YES — Conversation-level
                                  continuity for programmatic
                                  Runner callers.
        unpinned     pinned       NO — the pin signals a switch
                                  to an externally-owned session
                                  identity; treat as a boundary.
        pinned       unpinned     NO — symmetric: leaving an
                                  externally-owned identity is
                                  also a boundary.
        pinned       pinned, ids  YES — same outer ADK session
                       match      across turns; intra-session
                                  continuity (the pre-fix
                                  intent of ``_last_plan``).
        pinned       pinned, ids  NO — the regression case:
                       differ     two distinct outer ADK
                                  sessions sharing one Runner,
                                  validation v4 Class 1.
        ===========  ===========  =============================

        Returning ``None`` causes the Runner to seed
        ``session.plan = Plan.empty(...)`` — the correct first-turn
        behaviour for an outer session that has never been seen
        before.

        Pre-fix the carry-forward was unconditional: a process-wide
        ``Runner._last_plan`` field. Validation v4 Class 1 (goldfive#271
        follow-up) showed that two distinct outer ADK sessions
        sharing one Runner caused the second session's first turn
        to inherit the first session's plan, then fail every
        revision attempt because the leaked plan made no sense for
        the new request.
        """
        if self._last_plan is None:
            return None
        # Either side touched the pin → require an exact session-id
        # match. Both sides unpinned → Conversation-level continuity.
        if self._last_plan_pinned or pinned:
            if self._last_plan_session_id != session_id:
                return None
        return self._last_plan

    def absorb_turn(
        self,
        outcome: ExecutionOutcome,
        *,
        user_input_summary: str = "",
        pinned: bool = False,
    ) -> TurnRecord:
        """Fold a turn's outcome back into the Conversation.

        Called by the Runner after ``executor.run`` returns (successful
        or not — we record the failure so the next turn's planner can
        see it). Returns the newly-appended :class:`TurnRecord`.

        ``pinned`` records whether this turn's caller passed an
        explicit ``Runner.run(session_id=...)`` outer-session pin so
        the next turn's :meth:`prior_plan_for` can apply the
        carry-forward matrix correctly.
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
        # zicato#12: carry the full actual output forward with the same
        # later-turns-win merge semantics as ``completed_results``.
        self.completed_outputs.update(session.completed_outputs)

        # goldfive#271 Gap 2: lift the turn's high-water sequence back to
        # the Conversation cursor so the next ``next_turn_session()``
        # picks up where this turn left off. This keeps wire sequences
        # globally unique within the Conversation, which matters when
        # goldfive#161's outer-session pin makes ``Session.run_id``
        # constant across turns. Use ``max`` to be defensive against an
        # outcome whose Session never advanced past the seed (e.g. a
        # turn that aborted before any sink emission).
        self._next_sequence = max(self._next_sequence, int(session._next_sequence))

        # Stash the turn's final plan keyed by the session id (and
        # pin state) that produced it so the next turn's
        # :meth:`prior_plan_for` lookup applies the documented
        # carry-forward matrix. Empty plans are skipped (no-op); see
        # :meth:`stash_plan` for the idempotency contract this shares
        # with the Runner's ``finally``-block invocation.
        self.stash_plan(session, pinned=pinned)

        plan_summary = ""
        completed_ids: list[str] = []
        if session.plan is not None:
            plan_summary = session.plan.summary or ""
            completed_ids = [
                t.id for t in session.plan.tasks if t.status.value == "COMPLETED" and t.id
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
