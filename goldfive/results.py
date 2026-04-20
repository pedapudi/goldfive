from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldfive.types import Session

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InvocationResult:
    """Result returned by ``AgentAdapter.invoke`` for a single task.

    ``text`` is the final assistant text, ``stop_reason`` is adapter-specific,
    ``error`` is populated if the invocation raised, and ``raw`` carries the
    adapter's native result object for debugging or downstream inspection.
    """

    task_id: str
    text: str = ""
    stop_reason: str = ""
    error: Exception | None = None
    raw: Any = None


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
