"""Per-task tool-call idempotency table + loop detector.

Smaller models occasionally fall into a "filler-call" pattern: they keep
calling a reporting tool (often ``report_task_progress``) over and over
with the same arguments instead of making forward progress. The eventual
failure mode is the host LLM's max-call budget, which surfaces as a
:class:`ResourceExhausted` outside goldfive — diagnostically useless.

This module gives the adapter dispatch layer two protections:

1. **Idempotency table** — a per-(task_id, tool_name, args_hash) memo.
   The first matching call goes through to the handler; subsequent
   matching calls within the same task return a cheap
   ``{"acknowledged": true, "duplicate": true}`` ACK and skip the
   underlying Steerer transition (which would have been a no-op anyway,
   but the dispatch is cheaper and the agent gets a clear "duplicate"
   signal in the tool result).

2. **Loop detector** — a per-task sliding window of the last
   :data:`_LOOP_WINDOW` tool-call signatures. Once
   :data:`_LOOP_THRESHOLD` of those signatures share a single
   ``(tool_name, args_hash)``, we emit a single
   :class:`DriftKind.LOOPING_TOOL_CALL` event and arm a one-shot
   suppressor so the planner sees the loop at most once per task.

The guard intentionally lives at the adapter dispatch boundary
(``invoke_tool``) rather than inside individual handlers — that way every
adapter (callable, ADK, Claude) gets the protection for free without
each handler growing its own bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only
    from goldfive.protocols import Steerer
    from goldfive.types import Session


# Loop-detector tuning. A window of 8 with a threshold of 6 means the
# guard fires after the agent has issued the same call ~6 times in the
# last 8 attempts — enough signal that the model is stuck, not just
# being slightly redundant.
_LOOP_WINDOW = 8
_LOOP_THRESHOLD = 6


@dataclass
class _TaskGuardState:
    """Per-task bookkeeping for idempotency + loop detection."""

    # (tool_name, args_hash) -> True once the first call has run.
    seen: set[tuple[str, str]] = field(default_factory=set)
    # Sliding window of the last _LOOP_WINDOW (tool_name, args_hash)
    # signatures observed for this task. Includes both first-time and
    # duplicate calls — the loop signal we care about is volume of
    # matching calls in flight, not whether the first one already ran.
    window: deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=_LOOP_WINDOW)
    )
    # Set once a LOOPING_TOOL_CALL drift has been emitted for this task,
    # so we don't pile on identical drifts every subsequent call. Cleared
    # by the planner when it refines/fails the task.
    loop_flagged: bool = False


@dataclass
class ToolLoopGuard:
    """Holds per-task guard state for the lifetime of one Session.

    A single guard instance is keyed off a Session (via
    :func:`guard_for`) and survives across all reporting-tool calls
    within that session.
    """

    _by_task: dict[str, _TaskGuardState] = field(default_factory=dict)

    def state_for(self, task_id: str) -> _TaskGuardState:
        """Return (and lazily create) the per-task state row."""
        st = self._by_task.get(task_id)
        if st is None:
            st = _TaskGuardState()
            self._by_task[task_id] = st
        return st

    def reset_task(self, task_id: str) -> None:
        """Drop guard state for ``task_id`` (e.g., after a refine).

        Lets a refined plan retry the looping tool from a clean slate
        without inheriting the prior duplicate / loop bookkeeping.
        """
        self._by_task.pop(task_id, None)


def args_signature(args: dict) -> str:
    """Stable hash of a tool-call args dict.

    JSON-encoded with ``sort_keys=True`` so dict ordering doesn't change
    the signature. Falls back to ``repr(args)`` for non-JSON-serialisable
    payloads — those are rare for reporting-tool args (which are simple
    strings + numbers + maps of strings) and the fallback at least keeps
    the hash deterministic within a single process.
    """
    try:
        encoded = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = repr(sorted(args.items())) if isinstance(args, dict) else repr(args)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def guard_for(session: Session) -> ToolLoopGuard:
    """Return the per-Session guard, creating it on first use.

    Stowed on a private attribute of the Session so the dispatch layer
    can pull the same instance on every call without threading it
    through a dozen call sites.
    """
    guard = getattr(session, "_tool_loop_guard", None)
    if isinstance(guard, ToolLoopGuard):
        return guard
    guard = ToolLoopGuard()
    # Session is a frozen-ish dataclass but Python lets us set new
    # attributes; the underscore prefix marks this as adapter-private.
    object.__setattr__(session, "_tool_loop_guard", guard)
    return guard


def detect_loop(state: _TaskGuardState, signature: tuple[str, str]) -> bool:
    """Return True iff the current window crosses the loop threshold for ``signature``.

    Counts occurrences of ``signature`` in the sliding window (which the
    caller has already updated to include the current call). Sets
    :attr:`_TaskGuardState.loop_flagged` so subsequent calls don't
    re-fire the drift.
    """
    if state.loop_flagged:
        return False
    if len(state.window) < _LOOP_THRESHOLD:
        return False
    matching = sum(1 for s in state.window if s == signature)
    if matching >= _LOOP_THRESHOLD:
        state.loop_flagged = True
        return True
    return False


async def emit_loop_drift(
    *,
    session: Session,
    steerer: Steerer,
    task_id: str,
    tool_name: str,
) -> None:
    """Push a CRITICAL ``LOOPING_TOOL_CALL`` drift through the steerer.

    Falls back silently if the steerer doesn't expose the private
    ``_handle_drift`` hook (e.g., test stubs) — losing the signal is
    less disruptive than crashing the tool dispatch.
    """
    from goldfive.types import DriftEvent, DriftKind, DriftSeverity

    drift = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.CRITICAL,
        detail=(
            f"task {task_id} kept calling {tool_name!r} with the same "
            f"arguments ({_LOOP_THRESHOLD}+ of last {_LOOP_WINDOW} calls); "
            "no forward progress detected"
        ),
        current_task_id=task_id,
    )
    handler = getattr(steerer, "_handle_drift", None)
    if handler is None:
        return
    try:
        await handler(drift, session)
    except Exception:  # noqa: BLE001 - drift dispatch must not break tools
        return


__all__ = [
    "ToolLoopGuard",
    "args_signature",
    "detect_loop",
    "emit_loop_drift",
    "guard_for",
]
