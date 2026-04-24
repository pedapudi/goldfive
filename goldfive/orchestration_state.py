"""Goldfive orchestration-level session-state namespace (goldfive#152).

This module owns the key names goldfive writes into
``goldfive.types.Session.state`` — the framework-agnostic orchestration
dict that the PlanReconciler, DefaultSteerer, executor heal paths, and
downstream planners (goldfive#153) all read and write.

**This is NOT the same surface as the ADK ``session.state``** dict the
ADK adapter writes to for agent-side reads (see
:mod:`goldfive.adapters._adk_state_protocol`). The ADK state protocol
is a per-adapter bridge between goldfive and the LLM's view; this
module is a cross-cutting orchestration convention that:

* Any goldfive component can read without pulling in ADK.
* Any planner or drift detector can consult for "what's the current
  plan, current task, recent steer, healed function_calls?" without
  having to crawl ``session.plan`` + drift stream + adapter internals.
* Downstream components (goldfive#153's GoldfivePlanner, goldfive#154's
  goal-aware refine) consume as a stable contract.

Keys live under :data:`GOLDFIVE_PREFIX`. Values are always JSON-serialisable
where practical so sinks can round-trip the dict if they want to
persist it; readers tolerate missing / malformed entries and return
typed defaults.

Namespace
---------
Plan lifecycle:

* ``goldfive.current_plan_id`` — plan id of the currently-installed
  :class:`Plan` on ``session.plan``. Stamped by plan_submitted and
  plan_revised flows.

Task lifecycle (set by PlanReconciler on RUNNING/COMPLETED):

* ``goldfive.current_task_id`` — task id most recently transitioned
  to RUNNING. Cleared when the task terminates and no new task has
  opened, and on run end.
* ``goldfive.current_task_title`` — the title of the same task.

Goals:

* ``goldfive.goals_summary`` — human-readable, formatted one-per-line
  summary of :attr:`Session.goals`. Refreshed whenever goals change
  (goldfive#152 USER_STEER path; goldfive#154 goal-aware refine).
  Prompt templates read this so they never need to walk
  ``session.goals`` themselves.

Active steer (set by DefaultSteerer on USER_STEER drift):

* ``goldfive.active_steer.body`` — the steer body as authored by the
  operator (post-``_compose_steer_restart_message`` UNWRAPPING; i.e.
  the raw steer text, not the framed restart message).
* ``goldfive.active_steer.at_turn`` — monotonic sequence value
  captured at steer-fire time. Lets downstream refines know "was this
  steer before or after the observation I'm looking at?".
* ``goldfive.active_steer.author`` — operator identity from the
  originating annotation (goldfive#171). Empty string when the bridge
  doesn't source annotations.
* ``goldfive.processed_steer_ids`` — bounded FIFO list of
  already-processed STEER ids (annotation id when available, otherwise
  the ``ControlMessage.id``). Consulted by DefaultSteerer to drop
  delivery retries and UI double-fires (goldfive#171).

Heal path (set by adapter ``_heal_pending_tool_calls``):

* ``goldfive.cancelled_function_call_ids`` — list of function_call
  ids that were healed mid-invocation. Append-only within a run; a
  later heal simply extends the list. Downstream prompt templates /
  refine paths may consult to reference "the cancelled tool call"
  without poking adapter internals.

Design notes
-------------
* **No cooldown.** Per the goldfive#152 directive, steering is always
  active — so ``goldfive.active_steer.*`` is just a durable read-back
  of the most recent USER_STEER, not a cooldown window. Consumers
  that want "has the steer expired?" compare the recorded ``at_turn``
  against the current session sequence themselves.
* **Tree-agnostic.** No presentation-specific or domain keys. The
  namespace is pure orchestration state.
* **Writers only under the prefix.** :func:`write` refuses non-goldfive
  keys so accidental clobbers are caught early.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from goldfive.types import Goal, Plan, Task, TaskStatus

GOLDFIVE_PREFIX = "goldfive."

# Plan lifecycle
KEY_CURRENT_PLAN_ID = "goldfive.current_plan_id"

# Task lifecycle (PlanReconciler)
KEY_CURRENT_TASK_ID = "goldfive.current_task_id"
KEY_CURRENT_TASK_TITLE = "goldfive.current_task_title"

# Goals
KEY_GOALS_SUMMARY = "goldfive.goals_summary"

# Active steer (DefaultSteerer)
KEY_ACTIVE_STEER_BODY = "goldfive.active_steer.body"
KEY_ACTIVE_STEER_AT_TURN = "goldfive.active_steer.at_turn"
KEY_ACTIVE_STEER_AUTHOR = "goldfive.active_steer.author"
# Source attribution for the active steer
# (goldfive-steer-unification). ``"user"`` for operator-authored steers
# (USER_STEER ControlMessage); ``"goldfive"`` for steers promoted by the
# drift-ladder (WARNING+/CRITICAL goldfive-detected drifts). Empty
# string when no steer is active or when the writer didn't know the
# source (treated as ``"user"`` for back-compat by readers that
# decide on suppression, i.e. "we have an active steer; if we can't
# prove it's goldfive-authored, treat it as user-authoritative").
KEY_ACTIVE_STEER_SOURCE = "goldfive.active_steer.source"

# USER_STEER idempotency (goldfive#171). A bounded list of already-
# processed annotation / control ids. Consulted by DefaultSteerer so a
# delivery retry or UI double-fire of the same STEER annotation doesn't
# cascade-cancel + refine twice.
KEY_PROCESSED_STEER_IDS = "goldfive.processed_steer_ids"

# Heal path (ADKAdapter._heal_pending_tool_calls)
KEY_CANCELLED_FUNCTION_CALL_IDS = "goldfive.cancelled_function_call_ids"

ALL_KEYS: tuple[str, ...] = (
    KEY_CURRENT_PLAN_ID,
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
    KEY_GOALS_SUMMARY,
    KEY_ACTIVE_STEER_BODY,
    KEY_ACTIVE_STEER_AT_TURN,
    KEY_ACTIVE_STEER_AUTHOR,
    KEY_ACTIVE_STEER_SOURCE,
    KEY_PROCESSED_STEER_IDS,
    KEY_CANCELLED_FUNCTION_CALL_IDS,
)

# Cap on how many processed steer ids we retain on ``session.state`` so
# long-lived sessions don't balloon the dedupe set. The oldest entries
# are evicted FIFO when the cap is reached.
PROCESSED_STEER_IDS_CAP = 256


# ---------------------------------------------------------------------------
# Primitive writers / readers
# ---------------------------------------------------------------------------


def _assert_goldfive_key(key: str) -> None:
    if not key.startswith(GOLDFIVE_PREFIX):
        raise ValueError(f"goldfive.orchestration_state refuses to write non-goldfive key: {key!r}")


def write(state: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Stamp ``key`` under the goldfive namespace.

    Rejects non-``goldfive.*`` keys defensively — this module is the
    single source of truth for orchestration-level state, and a
    mis-namespaced write is almost always a typo or a leaky abstraction.
    """
    _assert_goldfive_key(key)
    state[key] = value


def clear(state: MutableMapping[str, Any], key: str) -> None:
    """Remove ``key`` from ``state`` if present. Refuses non-goldfive keys."""
    _assert_goldfive_key(key)
    state.pop(key, None)


def read(state: Any, key: str, default: Any = None) -> Any:
    """Return ``state[key]`` or ``default`` when absent / malformed."""
    if not isinstance(state, Mapping):
        return default
    try:
        value = state.get(key, default)
    except Exception:
        return default
    return value if value is not None else default


# ---------------------------------------------------------------------------
# Plan / task helpers
# ---------------------------------------------------------------------------


def set_current_plan(state: MutableMapping[str, Any], plan: Plan | None) -> None:
    """Record the plan id (or clear it when ``plan`` is None)."""
    if plan is None:
        clear(state, KEY_CURRENT_PLAN_ID)
        return
    write(state, KEY_CURRENT_PLAN_ID, str(getattr(plan, "id", "") or ""))


def set_current_task(state: MutableMapping[str, Any], task: Task | None) -> None:
    """Record the active task id + title, or clear when ``task`` is None."""
    if task is None:
        clear(state, KEY_CURRENT_TASK_ID)
        clear(state, KEY_CURRENT_TASK_TITLE)
        return
    write(state, KEY_CURRENT_TASK_ID, str(getattr(task, "id", "") or ""))
    write(state, KEY_CURRENT_TASK_TITLE, str(getattr(task, "title", "") or ""))


def clear_current_task(state: MutableMapping[str, Any]) -> None:
    """Remove the current-task keys. Cheap alias used at run end."""
    clear(state, KEY_CURRENT_TASK_ID)
    clear(state, KEY_CURRENT_TASK_TITLE)


# ---------------------------------------------------------------------------
# Goals summary
# ---------------------------------------------------------------------------


def format_goals_summary(goals: Iterable[Goal] | None) -> str:
    """Return a human-readable one-per-line summary of goals.

    Shape::

        - [g1] first goal summary
        - [g2] second goal summary

    Empty / None input renders as the single line ``"(no goals)"`` so
    prompt templates can interpolate unconditionally without
    worrying about the empty case.
    """
    if not goals:
        return "(no goals)"
    lines: list[str] = []
    for goal in goals:
        gid = getattr(goal, "id", "") or "(no-id)"
        summary = getattr(goal, "summary", "") or "(no summary)"
        lines.append(f"- [{gid}] {summary}")
    if not lines:
        return "(no goals)"
    return "\n".join(lines)


def refresh_goals_summary(
    state: MutableMapping[str, Any],
    goals: Iterable[Goal] | None,
) -> None:
    """Recompute and stamp ``goldfive.goals_summary`` from ``goals``."""
    write(state, KEY_GOALS_SUMMARY, format_goals_summary(goals))


# ---------------------------------------------------------------------------
# Active steer
# ---------------------------------------------------------------------------


def set_active_steer(
    state: MutableMapping[str, Any],
    *,
    body: str,
    at_turn: int,
    author: str = "",
    source: str = "",
) -> None:
    """Record the active steer body + turn (+ optional author + source).

    See module docstring. ``author`` defaults to the empty string so
    existing callers that don't know who authored the steer keep
    working unchanged (goldfive#171).

    ``source`` (goldfive-steer-unification) indicates the origin of the
    steer: ``"user"`` for an operator-authored ControlMessage, or
    ``"goldfive"`` for a drift-ladder-promoted goldfive-internal steer.
    Empty string is tolerated for back-compat.
    """
    write(state, KEY_ACTIVE_STEER_BODY, str(body or ""))
    write(state, KEY_ACTIVE_STEER_AT_TURN, int(at_turn))
    write(state, KEY_ACTIVE_STEER_AUTHOR, str(author or ""))
    write(state, KEY_ACTIVE_STEER_SOURCE, str(source or ""))


def clear_active_steer(state: MutableMapping[str, Any]) -> None:
    """Clear the active-steer keys (e.g. at run start). Idempotent."""
    clear(state, KEY_ACTIVE_STEER_BODY)
    clear(state, KEY_ACTIVE_STEER_AT_TURN)
    clear(state, KEY_ACTIVE_STEER_AUTHOR)
    clear(state, KEY_ACTIVE_STEER_SOURCE)


# ---------------------------------------------------------------------------
# Processed steer ids (goldfive#171)
# ---------------------------------------------------------------------------


def has_processed_steer_id(state: Any, steer_id: str) -> bool:
    """True when ``steer_id`` is already in the processed set.

    Tolerant of malformed state values: a non-list / missing entry is
    treated as an empty set.
    """
    if not steer_id:
        return False
    existing = read(state, KEY_PROCESSED_STEER_IDS, [])
    if not isinstance(existing, list):
        return False
    return str(steer_id) in existing


def record_processed_steer_id(
    state: MutableMapping[str, Any],
    steer_id: str,
) -> None:
    """Append ``steer_id`` to the processed list with FIFO eviction.

    Empty ids are a no-op. Duplicates are silently dropped so the
    function is safe to call unconditionally after a ``has_`` check.
    """
    if not steer_id:
        return
    existing = read(state, KEY_PROCESSED_STEER_IDS, [])
    if not isinstance(existing, list):
        existing = []
    merged: list[str] = [str(v) for v in existing if v]
    sid = str(steer_id)
    if sid in merged:
        return
    merged.append(sid)
    overflow = len(merged) - PROCESSED_STEER_IDS_CAP
    if overflow > 0:
        merged = merged[overflow:]
    write(state, KEY_PROCESSED_STEER_IDS, merged)


# ---------------------------------------------------------------------------
# Cancelled function_call ids
# ---------------------------------------------------------------------------


def append_cancelled_function_call_ids(
    state: MutableMapping[str, Any],
    call_ids: Iterable[str],
) -> None:
    """Append ``call_ids`` to ``goldfive.cancelled_function_call_ids``.

    Creates the list on first call; extends idempotently on subsequent
    calls. De-duplicates while preserving order so a heal that fires
    twice for the same id (defensive) doesn't balloon the list.
    """
    ids = [str(c) for c in call_ids if c]
    if not ids:
        return
    existing = read(state, KEY_CANCELLED_FUNCTION_CALL_IDS, [])
    if not isinstance(existing, list):
        existing = []
    merged: list[str] = list(existing)
    seen = set(merged)
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        merged.append(cid)
    write(state, KEY_CANCELLED_FUNCTION_CALL_IDS, merged)


def read_cancelled_function_call_ids(state: Any) -> list[str]:
    value = read(state, KEY_CANCELLED_FUNCTION_CALL_IDS, [])
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]


# ---------------------------------------------------------------------------
# Convenience: task-status-aware current-task sync
# ---------------------------------------------------------------------------


def rotate_current_task_id(
    state: MutableMapping[str, Any],
    plan: Plan | None,
    agent_name: str,
) -> str | None:
    """Advance ``goldfive.current_task_id`` after a terminal transition.

    Called from :mod:`goldfive.reporting` when a task transitions to a
    terminal status (COMPLETED / FAILED / CANCELLED / NOT_NEEDED). The
    point of this helper is to keep the pin pointed at *work still to
    do* so subsequent reporting-tool calls in the same invocation
    context can continue to fall back to the pin instead of failing
    with ``missing_task_id``.

    Rules:

    * **Rotate** — when exactly one PENDING / RUNNING task remains
      assigned to ``agent_name`` in ``plan``, stamp its id and return it.
    * **Clear** — when no PENDING / RUNNING task remains assigned to
      ``agent_name`` (all done, or none ever assigned), drop the key and
      return ``None``.
    * **Clear (ambiguous)** — when multiple PENDING / RUNNING tasks are
      assigned to ``agent_name`` the helper also clears the key and
      returns ``None``, deferring to the adapter's
      ``before_agent_callback`` path to pick the next one when the agent
      runs again. Stamping an arbitrary pending task here would race
      with whatever the orchestrator ends up dispatching.

    Tolerant of degenerate input: a ``None`` plan or an empty
    ``agent_name`` clears the key and returns ``None``. Never raises.
    """
    # No plan, no rotation — clear defensively so callers don't keep
    # driving a stale pointer.
    if plan is None:
        clear(state, KEY_CURRENT_TASK_ID)
        clear(state, KEY_CURRENT_TASK_TITLE)
        return None

    candidates: list[Task] = []
    for t in getattr(plan, "tasks", ()) or ():
        assignee = str(getattr(t, "assignee_agent_id", "") or "")
        if agent_name and assignee and assignee != agent_name:
            continue
        status = getattr(t, "status", None)
        if status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            candidates.append(t)

    if len(candidates) == 1:
        next_task = candidates[0]
        set_current_task(state, next_task)
        return str(getattr(next_task, "id", "") or "") or None

    # Zero or multiple candidates — clear and let the orchestrator
    # re-pin on the next before_agent_callback.
    clear(state, KEY_CURRENT_TASK_ID)
    clear(state, KEY_CURRENT_TASK_TITLE)
    return None


def sync_current_task_from_transition(
    state: MutableMapping[str, Any],
    task: Task | None,
    to: TaskStatus,
) -> None:
    """Stamp or clear current_task_* based on a transition target.

    Called by PlanReconciler (and the DefaultSteerer's overlay path) on
    every task transition. Rules:

    * RUNNING → stamp the task's id + title.
    * COMPLETED / FAILED / CANCELLED / NOT_NEEDED → clear if the
      task is the currently-pinned one (another task may have opened
      before this one's terminal write; don't steal its stamp).
    * Any other status is a no-op.
    """
    if task is None:
        return
    if to is TaskStatus.RUNNING:
        set_current_task(state, task)
        return
    if to in (
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.NOT_NEEDED,
    ):
        current = read(state, KEY_CURRENT_TASK_ID, "")
        if current == str(getattr(task, "id", "") or ""):
            clear_current_task(state)


__all__ = [
    "ALL_KEYS",
    "GOLDFIVE_PREFIX",
    "KEY_ACTIVE_STEER_AT_TURN",
    "KEY_ACTIVE_STEER_AUTHOR",
    "KEY_ACTIVE_STEER_BODY",
    "KEY_ACTIVE_STEER_SOURCE",
    "KEY_CANCELLED_FUNCTION_CALL_IDS",
    "KEY_CURRENT_PLAN_ID",
    "KEY_CURRENT_TASK_ID",
    "KEY_CURRENT_TASK_TITLE",
    "KEY_GOALS_SUMMARY",
    "KEY_PROCESSED_STEER_IDS",
    "PROCESSED_STEER_IDS_CAP",
    "append_cancelled_function_call_ids",
    "clear",
    "clear_active_steer",
    "clear_current_task",
    "format_goals_summary",
    "has_processed_steer_id",
    "read",
    "read_cancelled_function_call_ids",
    "record_processed_steer_id",
    "refresh_goals_summary",
    "rotate_current_task_id",
    "set_active_steer",
    "set_current_plan",
    "set_current_task",
    "sync_current_task_from_transition",
    "write",
]
