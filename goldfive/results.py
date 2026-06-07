from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldfive.types import Session

log = logging.getLogger(__name__)


#: Separator placed between successive assistant text turns when an adapter
#: joins them into :attr:`InvocationResult.full_text`. A blank line keeps the
#: turns visually distinct without inventing markup a grader might trip over.
TURN_SEPARATOR = "\n\n"


@dataclasses.dataclass
class InvocationResult:
    """Result returned by ``AgentAdapter.invoke`` for a single task.

    ``text`` is the **final** assistant text turn, ``stop_reason`` is
    adapter-specific, ``error`` is populated if the invocation raised, and
    ``raw`` carries the adapter's native result object for debugging or
    downstream inspection.

    Full-fidelity output (zicato#12)
    --------------------------------
    Agents routinely emit their substantive answer (a list, a table of ids,
    a detailed value) in one turn and a terse wrap-up ("Done — let me know if
    you need anything else") in a later turn. ``text`` keeps only the last
    non-empty turn, which silently drops the substantive turn. Graders that
    need to match against the agent's *actual* output must read:

    * ``text_turns`` — every non-empty assistant text turn, in order. The
      lossless record of what the agent produced.
    * ``full_text`` — the same turns joined by :data:`TURN_SEPARATOR`; the
      canonical, full-fidelity gradeable artifact.

    ``text`` is retained unchanged for backward compatibility (existing
    consumers that only ever wanted the final turn keep their semantics).
    Adapters that cannot distinguish turns may leave ``text_turns`` empty;
    :attr:`full_text` then falls back to ``text`` (see ``__post_init__``).
    """

    task_id: str
    text: str = ""
    stop_reason: str = ""
    error: Exception | None = None
    raw: Any = None
    #: Every non-empty assistant text turn of this invocation, in emission
    #: order. Empty for adapters that do not track per-turn text; in that
    #: case ``full_text`` falls back to ``text``.
    text_turns: list[str] = dataclasses.field(default_factory=list)
    #: All assistant text turns joined by :data:`TURN_SEPARATOR`. The
    #: full-fidelity gradeable artifact. Defaults to the joined ``text_turns``
    #: when set; otherwise falls back to ``text``.
    full_text: str = ""

    def __post_init__(self) -> None:
        # full_text is derived, not separately authored: prefer the joined
        # turns, fall back to the single ``text`` so callers that construct an
        # InvocationResult with only ``text=`` still get a sensible full_text.
        if not self.full_text:
            if self.text_turns:
                self.full_text = TURN_SEPARATOR.join(self.text_turns)
            else:
                self.full_text = self.text


@dataclasses.dataclass
class ExecutionOutcome:
    """Final outcome of an ``Executor.run`` invocation.

    ``reason`` is populated when ``success`` is False to describe why the run
    terminated (e.g., unrecoverable task failure, user cancellation).
    """

    success: bool
    session: Session
    reason: str = ""


def evaluate_goal_predicates(session: Session) -> str | None:
    """Evaluate every ``Goal.success_predicate`` on ``session``.

    Implements the third clause of :doc:`PLAN-LIFECYCLE.md §6.1
    </design/PLAN-LIFECYCLE>`: a run is successful only if every goal's
    ``success_predicate`` returns ``True`` (``None`` is treated as
    vacuously met). Executors call this after the "every task terminal +
    no orphans" gate and include the returned reason (if any) on the
    :class:`ExecutionOutcome`.

    Returns
    -------
    ``None``
        All goals are met (or ``session.goals`` is empty) — the run
        may be reported as successful.
    ``str``
        A human-readable reason describing the first unmet goal. The
        caller should set ``Outcome.success = False`` and propagate
        this string on ``Outcome.reason``.

    Behavior
    --------
    * Predicates are evaluated in ``session.goals`` order; the first
      one that is not met short-circuits.
    * A predicate that raises is logged at WARNING and treated as unmet
      (never as met): a crashing predicate is never a pass.
    """
    for goal in session.goals:
        predicate = goal.success_predicate
        if predicate is None:
            continue
        summary = goal.summary or goal.id or "<unnamed>"
        try:
            met = predicate(session)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "goal predicate raised for goal %r: %s",
                summary,
                exc,
            )
            return f"goal '{summary}' predicate raised: {exc}"
        if not met:
            return f"goal '{summary}' unmet"
    return None
