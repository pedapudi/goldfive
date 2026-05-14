"""Goldfive ``Session.state`` accessors — unified read + write surface.

This module is the single source of truth for the goldfive ``goldfive.*``
key namespace on :class:`goldfive.types.Session.state`. It exposes:

* :class:`StateStore` — a typed handle (constructed via
  :meth:`StateStore.for_session` / :meth:`StateStore.for_state`) that
  groups every read + write under one object so the call site doesn't
  have to string-fish through ``state.get('goldfive.foo')`` /
  ``state[...] = v``.
* Module-level free functions — kept as the underlying primitives the
  :class:`StateStore` methods route through. Many existing call sites
  use them directly (e.g.
  ``state_store.set_current_plan(session.state, plan)``); these continue
  to work unchanged.

History
-------

This module is the result of merging two earlier siblings into one
surface (Wave A piece 1 of the goldfive refactor):

* ``goldfive.orchestration_state`` — module-level write functions and
  key constants for the ``goldfive.*`` prefix on ``Session.state``.
* ``goldfive.orchestration_store`` — typed handle (``OrchestrationStore``
  class) with read methods plus a handful of writes.

Both modules touched the same ADK session dict via different APIs; the
docstring on the old ``orchestration_store`` literally called itself
"Phase 1". They have been collapsed onto :class:`StateStore` here. The
old module names remain importable as thin deprecation shims until the
shim grace period elapses.

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

Active drift conditions (set by DefaultSteerer on every emit, goldfive#271 PR1):

* ``goldfive.active_drifts`` — dict keyed by ``condition_id`` storing
  the in-flight drift conditions for the current turn. A drift
  "condition" is a logical occurrence (kind+task+agent within a turn)
  that may emit multiple ``DriftDetected`` events (opened →
  escalating → resolved / human_intervention_required). The helpers
  :func:`open_or_escalate_drift`, :func:`resolve_drift`, and
  :func:`escalate_drift_to_human_intervention` route lifecycle
  transitions through this dict; the steerer's wire emit reads the
  resulting :class:`Drift` to stamp ``condition_id``,
  ``lifecycle``, and ``prev_severity`` on ``DriftDetected``. Same
  kind+task+agent within the same turn collapses onto one
  condition; a new turn opens a fresh condition.

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

import asyncio
import dataclasses
import hashlib
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from goldfive.types import DriftKind, DriftSeverity, Goal, Plan, Task, TaskStatus

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from goldfive.types import Session


GOLDFIVE_PREFIX = "goldfive."

# Plan lifecycle
KEY_CURRENT_PLAN_ID = "goldfive.current_plan_id"

# Task lifecycle (PlanReconciler)
KEY_CURRENT_TASK_ID = "goldfive.current_task_id"
KEY_CURRENT_TASK_TITLE = "goldfive.current_task_title"
# Revision stamp on the current_task_id pin (goldfive#266 / pin
# versioning). When the adapter's pin ladder lands a task_id, it also
# stamps the plan's ``revision_index`` at write time. The reporting
# handlers consult this stamp at report time to distinguish a "fresh"
# pin (matches current revision) from a "stale" pin (set under an
# older revision and the report arrived after a refine landed). Stale
# pins under a CORRECT-kind supersedes link refuse the transition; the
# old task's terminal state is historical fact and the correction is a
# separate work unit. Stale pins under a REPLACE-kind link route to
# the replacement (existing supersession behaviour). Missing stamp
# (legacy state, custom adapter that hasn't migrated) reads as 0 so
# pre-versioning callers preserve their current behaviour.
KEY_CURRENT_TASK_REVISION = "goldfive.current_task_revision"

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

# Active drift conditions (goldfive#271 PR1). Map condition_id ->
# serialised :class:`Drift` (a small JSON-friendly dict). A drift
# "condition" is a logical occurrence keyed by kind+task+agent within
# the current turn; multiple emits within that scope share the same
# condition_id and progress through ``opened -> escalating ->
# resolved`` (or ``human_intervention_required``). Stored on
# ``session.state`` so other goldfive components can read the active
# set without poking steerer internals; serialisable so sinks that
# want to round-trip the state dict (e.g. for replay) can do so.
KEY_ACTIVE_DRIFTS = "goldfive.active_drifts"

# State key for the new reasoning-extracted bindings slot. Lives under
# the goldfive prefix so :func:`write` accepts it. Value shape:
# ``dict[agent_name, ReasoningBinding-as-dict]`` for cheap JSON
# serialisation by sinks that round-trip the state dict.
REASONING_BINDINGS_KEY = "goldfive.reasoning_extracted_bindings"

# State key for the per-``function_call_id`` delegation-pin map
# (goldfive#241 Item 3-bis, V4 in the Phase 0 audit catalog). Phase 2.1
# of goldfive#271 — consolidated here so writers and readers agree on
# one source of truth on goldfive's own ``Session.state``.
#
# Value shape: ``dict[function_call_id, {task_id, revision, tool_args?}]``.
# Legacy entries stamped by pre-#266 adapters were bare strings; the
# :class:`DelegationPin` accessor normalises both shapes.
PENDING_DELEGATIONS_KEY = "goldfive.pending_delegations"

ALL_KEYS: tuple[str, ...] = (
    KEY_CURRENT_PLAN_ID,
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
    KEY_CURRENT_TASK_REVISION,
    KEY_GOALS_SUMMARY,
    KEY_ACTIVE_STEER_BODY,
    KEY_ACTIVE_STEER_AT_TURN,
    KEY_ACTIVE_STEER_AUTHOR,
    KEY_ACTIVE_STEER_SOURCE,
    KEY_PROCESSED_STEER_IDS,
    KEY_CANCELLED_FUNCTION_CALL_IDS,
    KEY_ACTIVE_DRIFTS,
)

# Cap on how many processed steer ids we retain on ``session.state`` so
# long-lived sessions don't balloon the dedupe set. The oldest entries
# are evicted FIFO when the cap is reached.
PROCESSED_STEER_IDS_CAP = 256


# ---------------------------------------------------------------------------
# Module-level primitives
# ---------------------------------------------------------------------------


def _assert_goldfive_key(key: str) -> None:
    if not key.startswith(GOLDFIVE_PREFIX):
        raise ValueError(f"goldfive.state_store refuses to write non-goldfive key: {key!r}")


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
    clear(state, KEY_CURRENT_TASK_REVISION)


def stamp_current_task_revision(
    state: MutableMapping[str, Any],
    revision: int,
) -> None:
    """Stamp ``goldfive.current_task_revision`` alongside the current-task pin.

    Companion writer to :func:`set_current_task` for the goldfive#266
    pin-versioning path. Callers stamp the plan ``revision_index``
    in effect at the moment they wrote the pin so report-time handlers
    can tell a fresh pin from one set under an older plan.

    Negative revisions are clamped to 0; non-int values are coerced
    via ``int()`` (defensive — readers assume an int-valued field).
    """
    try:
        rev = max(0, int(revision))
    except (TypeError, ValueError):
        rev = 0
    write(state, KEY_CURRENT_TASK_REVISION, rev)


def read_current_task_revision(state: Any) -> int:
    """Return ``goldfive.current_task_revision`` as an int, default 0.

    Tolerant of missing / malformed values (legacy state, custom
    adapters that pre-date #266) — those read as 0 and the report-time
    classifier treats them as "match" against the plan's initial
    ``revision_index=0``.
    """
    raw = read(state, KEY_CURRENT_TASK_REVISION, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


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
        clear(state, KEY_CURRENT_TASK_REVISION)
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
        # goldfive#266 — stamp the rotation moment's revision so the
        # next reporting call on this newly-pinned task reads as
        # "fresh" against the current plan. The adapter's pin ladder
        # may overwrite this on the next agent invocation; it's fine.
        rev = int(getattr(plan, "revision_index", 0) or 0)
        stamp_current_task_revision(state, rev)
        return str(getattr(next_task, "id", "") or "") or None

    # Zero or multiple candidates — clear and let the orchestrator
    # re-pin on the next before_agent_callback.
    clear(state, KEY_CURRENT_TASK_ID)
    clear(state, KEY_CURRENT_TASK_TITLE)
    clear(state, KEY_CURRENT_TASK_REVISION)
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


# ---------------------------------------------------------------------------
# Drift conditions (goldfive#271 PR1)
# ---------------------------------------------------------------------------

# Lifecycle string constants — mirror the proto3 ``DriftLifecycle`` enum
# values that the steerer stamps onto ``DriftDetected.lifecycle``. Kept
# as plain lowercase strings here so :class:`Drift` round-trips through
# ``state.dict`` without dragging the proto enum into stored state.
LIFECYCLE_OPENED: str = "opened"
LIFECYCLE_ESCALATING: str = "escalating"
LIFECYCLE_RESOLVED: str = "resolved"
LIFECYCLE_HUMAN_INTERVENTION_REQUIRED: str = "human_intervention_required"

LIFECYCLE_VALUES: tuple[str, ...] = (
    LIFECYCLE_OPENED,
    LIFECYCLE_ESCALATING,
    LIFECYCLE_RESOLVED,
    LIFECYCLE_HUMAN_INTERVENTION_REQUIRED,
)


@dataclasses.dataclass
class Drift:
    """In-flight drift condition tracked on ``session.state`` (goldfive#271 PR1).

    A *condition* is a logical occurrence of drift (kind+task+agent
    within a turn) that may emit one or more ``DriftDetected`` events
    as it evolves. Identity is the ``condition_id`` (sha1 prefix); the
    other fields are bookkeeping the lifecycle helpers maintain so the
    steerer's wire emit can stamp ``lifecycle`` + ``prev_severity``
    onto ``DriftDetected`` without re-deriving them.

    Attributes
    ----------
    condition_id:
        Stable identifier — same kind+task+agent within the same turn
        always hashes to the same value. See :func:`compute_condition_id`.
    kind:
        :class:`DriftKind` enum value — copied from the underlying
        ``DriftEvent.kind`` for read-back without re-classifying.
    task_id / agent_id / turn_id:
        Tuple that hashes to ``condition_id``. Stored for diagnostics
        / round-tripping.
    severity:
        Latest :class:`DriftSeverity` observed on the condition. Bumped
        monotonically by :func:`open_or_escalate_drift`.
    prev_severity:
        Severity before the most-recent transition. Meaningful only on
        an ``escalating`` step; ``None`` for the opening / resolving
        transitions. Wire-stamped onto ``DriftDetected.prev_severity``
        (the proto carries it as ``DRIFT_SEVERITY_UNSPECIFIED``) so
        sinks can render "INFO -> WARNING" deltas without remembering
        the previous emit.
    lifecycle:
        One of :data:`LIFECYCLE_OPENED`, :data:`LIFECYCLE_ESCALATING`,
        :data:`LIFECYCLE_RESOLVED`,
        :data:`LIFECYCLE_HUMAN_INTERVENTION_REQUIRED`. Wire-stamped onto
        ``DriftDetected.lifecycle`` after enum conversion.
    occurrences:
        Count of times the condition has emitted on this turn (1 at
        ``opened``, 2+ at successive ``escalating`` emits, frozen at
        the final emit when the condition resolves / escalates to
        human intervention).
    """

    condition_id: str
    kind: DriftKind | None
    task_id: str
    agent_id: str
    turn_id: str
    severity: DriftSeverity | None
    prev_severity: DriftSeverity | None = None
    lifecycle: str = LIFECYCLE_OPENED
    occurrences: int = 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialisation for round-tripping through state."""
        return {
            "condition_id": self.condition_id,
            "kind": _kind_to_str(self.kind),
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "turn_id": self.turn_id,
            "severity": _severity_to_str(self.severity),
            "prev_severity": _severity_to_str(self.prev_severity),
            "lifecycle": self.lifecycle,
            "occurrences": int(self.occurrences),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Drift:
        """Inverse of :meth:`to_dict`. Tolerant of unknown enum values
        (falls back to ``None``) so a sink that round-trips an old
        snapshot doesn't crash the helpers."""
        return cls(
            condition_id=str(data.get("condition_id", "") or ""),
            kind=_kind_from_str(str(data.get("kind", "") or "")),
            task_id=str(data.get("task_id", "") or ""),
            agent_id=str(data.get("agent_id", "") or ""),
            turn_id=str(data.get("turn_id", "") or ""),
            severity=_severity_from_str(str(data.get("severity", "") or "")),
            prev_severity=_severity_from_str(str(data.get("prev_severity", "") or "")),
            lifecycle=str(data.get("lifecycle", LIFECYCLE_OPENED) or LIFECYCLE_OPENED),
            occurrences=int(data.get("occurrences", 1) or 1),
        )


def _severity_to_str(severity: DriftSeverity | None) -> str:
    if severity is None:
        return ""
    try:
        return severity.value
    except AttributeError:
        return ""


def _severity_from_str(name: str) -> DriftSeverity | None:
    if not name:
        return None
    try:
        return DriftSeverity(name)
    except ValueError:
        return None


def _kind_to_str(kind: DriftKind | None) -> str:
    if kind is None:
        return ""
    try:
        return kind.value
    except AttributeError:
        return ""


def _kind_from_str(name: str) -> DriftKind | None:
    if not name:
        return None
    try:
        return DriftKind(name)
    except ValueError:
        return None


def compute_condition_id(
    *,
    kind: DriftKind,
    task_id: str,
    agent_id: str,
    turn_id: str,
) -> str:
    """Return the stable condition_id for a (kind, task, agent, turn) tuple.

    Identity rule (goldfive#271 PR1): ``sha1(f"{kind.value}|{task_id}|
    {agent_id}|{turn_id}")[:16]``. Same kind+task+agent within the
    same turn always hashes to the same 16-char prefix; a new turn
    always opens a fresh condition. The 16-char prefix is well below
    sha1's collision floor for the volumes the orchestrator emits in
    practice (a single turn rarely exceeds a few hundred drift events
    across all kinds + tasks + agents) but remains short enough to
    surface inline in logs / debug views without truncation.
    """
    payload = f"{kind.value}|{task_id}|{agent_id}|{turn_id}".encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


def _read_active_drifts(state: Any) -> dict[str, dict[str, Any]]:
    """Return the active-drifts dict, or ``{}`` when missing / malformed."""
    raw = read(state, KEY_ACTIVE_DRIFTS, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for cid, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        out[str(cid)] = dict(payload)
    return out


def _write_active_drifts(
    state: MutableMapping[str, Any],
    drifts: Mapping[str, Mapping[str, Any]],
) -> None:
    write(state, KEY_ACTIVE_DRIFTS, {str(k): dict(v) for k, v in drifts.items()})


def get_active_drift(state: Any, condition_id: str) -> Drift | None:
    """Return the in-flight :class:`Drift` for ``condition_id``, or ``None``."""
    if not condition_id:
        return None
    payload = _read_active_drifts(state).get(condition_id)
    if payload is None:
        return None
    return Drift.from_dict(payload)


def list_active_drifts(state: Any) -> list[Drift]:
    """Return all in-flight conditions on the session as :class:`Drift` objects."""
    return [Drift.from_dict(payload) for payload in _read_active_drifts(state).values()]


def open_or_escalate_drift(
    state: MutableMapping[str, Any],
    *,
    kind: DriftKind,
    task_id: str,
    agent_id: str,
    turn_id: str,
    severity: DriftSeverity,
) -> Drift:
    """Open a new condition or escalate an existing one.

    Returns the resulting :class:`Drift` with ``lifecycle`` set to
    :data:`LIFECYCLE_OPENED` for first emits or
    :data:`LIFECYCLE_ESCALATING` for repeats. The caller (typically the
    steerer) reads the returned object's ``condition_id`` /
    ``lifecycle`` / ``prev_severity`` to stamp the corresponding
    fields on the wire ``DriftDetected``.

    Severity bumping is monotonic in the ``DriftSeverity`` ordering
    (INFO < WARNING < CRITICAL); a re-emit at lower severity preserves
    the higher recorded value. ``prev_severity`` is the severity
    *before* this transition — only meaningful on the
    ``escalating`` step. ``occurrences`` increments on every escalate
    so callers can render "fired N times" without crawling the event
    stream.
    """
    cid = compute_condition_id(kind=kind, task_id=task_id, agent_id=agent_id, turn_id=turn_id)
    active = _read_active_drifts(state)
    existing = active.get(cid)
    if existing is None:
        drift = Drift(
            condition_id=cid,
            kind=kind,
            task_id=str(task_id or ""),
            agent_id=str(agent_id or ""),
            turn_id=str(turn_id or ""),
            severity=severity,
            prev_severity=None,
            lifecycle=LIFECYCLE_OPENED,
            occurrences=1,
        )
        active[cid] = drift.to_dict()
        _write_active_drifts(state, active)
        return drift

    existing_drift = Drift.from_dict(existing)
    prev_severity = existing_drift.severity
    new_severity = _max_severity(existing_drift.severity, severity)
    updated = Drift(
        condition_id=cid,
        kind=kind,
        task_id=existing_drift.task_id,
        agent_id=existing_drift.agent_id,
        turn_id=existing_drift.turn_id,
        severity=new_severity,
        prev_severity=prev_severity,
        lifecycle=LIFECYCLE_ESCALATING,
        occurrences=existing_drift.occurrences + 1,
    )
    active[cid] = updated.to_dict()
    _write_active_drifts(state, active)
    return updated


def resolve_drift(
    state: MutableMapping[str, Any],
    condition_id: str,
) -> Drift | None:
    """Mark the condition resolved and remove it from the active set.

    Returns the final :class:`Drift` (with ``lifecycle`` set to
    :data:`LIFECYCLE_RESOLVED`) so the caller can stamp the resolving
    wire emit. Returns ``None`` when the condition is unknown — the
    helper is idempotent so a duplicate resolve is a no-op.
    """
    if not condition_id:
        return None
    active = _read_active_drifts(state)
    payload = active.pop(condition_id, None)
    if payload is None:
        return None
    drift = Drift.from_dict(payload)
    drift.prev_severity = None
    drift.lifecycle = LIFECYCLE_RESOLVED
    _write_active_drifts(state, active)
    return drift


def escalate_drift_to_human_intervention(
    state: MutableMapping[str, Any],
    condition_id: str,
) -> Drift | None:
    """Mark the condition as requiring human intervention.

    Sets ``severity`` to ``CRITICAL`` (the human-intervention tier is
    always CRITICAL by contract — see ``DefaultSteerer`` Level 4) and
    removes the entry from the active set: once a condition has
    escalated to a human, further auto-escalation on the same condition
    would be misleading. Returns the final :class:`Drift` so the caller
    can stamp the ``human_intervention_required`` wire emit. Returns
    ``None`` when the condition is unknown.
    """
    if not condition_id:
        return None
    active = _read_active_drifts(state)
    payload = active.pop(condition_id, None)
    if payload is None:
        return None
    drift = Drift.from_dict(payload)
    drift.prev_severity = drift.severity
    drift.severity = DriftSeverity.CRITICAL
    drift.lifecycle = LIFECYCLE_HUMAN_INTERVENTION_REQUIRED
    _write_active_drifts(state, active)
    return drift


_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.INFO: 0,
    DriftSeverity.WARNING: 1,
    DriftSeverity.CRITICAL: 2,
}


def _severity_rank(s: DriftSeverity | None) -> int:
    if s is None:
        return -1
    return _SEVERITY_ORDER.get(s, -1)


def _max_severity(a: DriftSeverity | None, b: DriftSeverity | None) -> DriftSeverity | None:
    return a if _severity_rank(a) >= _severity_rank(b) else b


# ---------------------------------------------------------------------------
# Active-invocation registry (goldfive#271 Phase 3.5 component 1)
# ---------------------------------------------------------------------------
#
# Per-session asyncio.Task registry keyed by ``invocation_id``. Owned by
# StateStore (the per-session orchestration-state surface) rather than
# the ADK plugin instance, restoring the Phase 0 state-ownership
# contract: per-session orchestration data lives on the store, not on
# adapter plugin instances.
#
# Why a module-level dict rather than a goldfive ``Session.state`` slot:
# :class:`asyncio.Task` is intentionally non-serializable (it holds a
# reference to a live coroutine + the running loop), and :func:`write`
# enforces "values must round-trip cleanly through sinks". Storing
# tasks under the ``goldfive.*`` prefix would either break that
# invariant or force every sink to special-case the slot. Instead we
# keep tasks in this module-level registry and key the outer dict by
# ``Session.id`` so the StateStore-as-view contract still holds: each
# store instance only sees its own session's tasks.
#
# Lock: a process-wide :class:`threading.Lock` protects the outer dict's
# structural mutations (sub-dict insert / pop). Inner-dict reads /
# writes are single-threaded per session because ADK callbacks run on
# one event loop, but a concurrent ``request_invocation_cancel`` from a
# different session could otherwise race the structural setdefault.
_ACTIVE_INVOCATION_TASKS: dict[str, dict[str, asyncio.Task[Any]]] = {}
_ACTIVE_INVOCATION_LOCK = threading.Lock()

# Companion registry: per-session set of invocation ids for which a
# cooperative cancel has been requested (goldfive#242). Stamped
# synchronously at the top of
# :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`
# so the late-drift gate can short-circuit drifts that fire during the
# 4-8s window between the cancel-request landing and ADK winding down
# the cancelled invocations (during which ``active_invocation_ids()``
# still lists them).
#
# Same locking discipline as the active-task registry above. Cleared by
# :meth:`StateStore.clear_active_invocations` so a session teardown
# wipes both registries in one shot.
_CANCEL_REQUESTED_INVOCATIONS: dict[str, set[str]] = {}

# Companion registry: per-session set of invocation ids for which a
# goldfive-internal supersede-cancel is in flight (issue #405 LOW #7).
# Stamped by the steerer's
# :meth:`~goldfive.steerer.DefaultSteerer._cancel_inflight_for_revision`
# before delegating to ``request_invocation_cancel`` and consumed by
# :meth:`SequentialExecutor._run_overlay`'s cancelled branch to
# distinguish an internal supersede from an external cancel.
#
# Co-exists with the legacy ``session._supersede_pending`` bool: that
# bool is still set/cleared for back-compat with the existing 8 tests
# in ``test_executor_supersede_cancel_nonfatal.py`` plus the
# empty-resolver fallback in :meth:`DriftObserver._cancel_inflight_for_revision`
# (no invocation id to anchor a registry entry, so the bool acts as
# a session-scope sentinel). The per-invocation set provides the
# defensive isolation under concurrent overlay iterations the audit
# called for. Same locking discipline as the cancel-requested
# registry; cleared by :meth:`StateStore.clear_active_invocations`
# so a session teardown wipes all three registries in one shot.
#
# The dual-signal design is transitional — see issue #430 for the
# follow-up to retire the bool entirely in favour of a sentinel-id
# registry stamp.
_SUPERSEDE_PENDING_INVOCATIONS: dict[str, set[str]] = {}


# ---------------------------------------------------------------------------
# Typed result objects
# ---------------------------------------------------------------------------


class BindingSource(StrEnum):
    """Origin of a current-task pin write.

    Stamped onto the pin so the pin-resolution ladder's events can
    distinguish a pin set by an agent-turn callback from one set by
    a delegation site, a steerer rotation, or a reasoning-extracted
    binding.

    Phase 1 only consumes the value for observability + the new
    reasoning-extracted-binding signal in the pin ladder; Phase 2's
    full migration will use it to attribute every catalogued writer.
    """

    DELEGATION_PIN = "delegation_pin"
    AGENT_CALLBACK = "agent_callback"
    REASONING = "reasoning"
    CORRECTION_TARGET = "correction_target"
    STEERER_ROTATION = "steerer_rotation"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class ActiveSteer:
    """Typed view of the active steer slots on goldfive ``Session.state``.

    Wraps the four
    :data:`KEY_ACTIVE_STEER_BODY` / ``_AT_TURN`` / ``_AUTHOR`` /
    ``_SOURCE`` keys so callers don't string-fish through the state
    dict.
    """

    body: str
    at_turn: int
    author: str
    source: str  # "user" | "goldfive" | ""

    def is_active(self) -> bool:
        """True when a steer is currently set (non-empty body)."""
        return bool(self.body)


@dataclasses.dataclass(frozen=True)
class DelegationPin:
    """Typed view of a single ``goldfive.pending_delegations`` entry.

    The on-disk shape supports both legacy bare-string and the
    versioned ``{task_id, revision, tool_args}`` dict (goldfive#266 /
    F7). This dataclass normalises both shapes for callers.
    """

    task_id: str
    revision: int = 0
    tool_args: Mapping[str, Any] | None = None

    def is_set(self) -> bool:
        return bool(self.task_id)


@dataclasses.dataclass(frozen=True)
class ReasoningBinding:
    """Typed view of a reasoning-extracted binding.

    Persisted as a plain dict under
    :data:`REASONING_BINDINGS_KEY[agent_name]` so sinks can round-trip
    the state dict; this dataclass is the in-process view.

    ``recorded_at_turn`` is the session sequence value at the moment
    the binding was recorded; consumers can compare it against the
    current sequence to dismiss bindings older than N turns. ``0``
    means "no sequence recorded" (e.g. test fixture without a session
    sequence counter).
    """

    agent_name: str
    task_id: str
    confidence: float
    recorded_at_turn: int = 0
    run_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "confidence": float(self.confidence),
            "recorded_at_turn": int(self.recorded_at_turn),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReasoningBinding | None:
        """Construct from a state-dict value, or ``None`` on garbage.

        Tolerant of partial dicts (sink-shaped, persistence-restored)
        so a missing ``run_id`` / ``session_id`` doesn't drop a
        legitimate binding.
        """
        if not isinstance(raw, Mapping):
            return None
        try:
            task_id = str(raw.get("task_id", "") or "")
            confidence = float(raw.get("confidence", 0.0) or 0.0)
            agent_name = str(raw.get("agent_name", "") or "")
        except (TypeError, ValueError):
            return None
        if not task_id:
            return None
        try:
            recorded_at_turn = int(raw.get("recorded_at_turn", 0) or 0)
        except (TypeError, ValueError):
            recorded_at_turn = 0
        return cls(
            agent_name=agent_name,
            task_id=task_id,
            confidence=confidence,
            recorded_at_turn=recorded_at_turn,
            run_id=str(raw.get("run_id", "") or ""),
            session_id=str(raw.get("session_id", "") or ""),
        )


# ---------------------------------------------------------------------------
# StateStore — typed handle
# ---------------------------------------------------------------------------


class StateStore:
    """Typed handle over goldfive ``Session.state``.

    Construct with :meth:`for_session` (when you have a goldfive
    :class:`~goldfive.types.Session`) or :meth:`for_state` (when you
    only have the state dict, e.g. in test scaffolding or in callbacks
    that reach the goldfive Session via
    :data:`~goldfive.adapters._adk_plugin.SESSION_CONTEXT_STATE_KEY`).

    All accessors are forgiving: missing / malformed entries return
    typed defaults rather than raising. The store is a *view* — it
    does not own the underlying dict and concurrent writers can mutate
    it between calls. Callers who need a snapshot should call once and
    cache.

    History: unified surface combining the read-side
    ``OrchestrationStore`` and the write-side module-level helpers
    formerly housed in ``goldfive.orchestration_state`` /
    ``goldfive.orchestration_store``.
    """

    __slots__ = ("_state", "_session_id")

    def __init__(self, state: Any, *, session_id: str = "") -> None:
        # We accept any mapping-shaped object — goldfive's Session.state
        # is a plain dict; tests sometimes pass a ``MappingProxyType``
        # over the same dict. Read paths tolerate both; write paths
        # require :class:`MutableMapping` (raised lazily by the helper).
        self._state = state if isinstance(state, Mapping) else {}
        # ``session_id`` keys the active-invocation task registry below.
        # Empty when the store was built for an arbitrary state dict
        # (test scaffolding, callbacks reaching the store with no
        # Session in scope) — those callers cannot drive the
        # active-invocation registry, but every other read/write path
        # remains usable.
        self._session_id = str(session_id or "")

    # -- Constructors ----------------------------------------------------

    @classmethod
    def for_session(cls, session: Session | None) -> StateStore:
        """Build a store backed by ``session.state``.

        ``None`` yields an empty-state store (writes are silently
        dropped) so callers who can't guarantee a session — e.g.
        defensive paths inside ADK callbacks — never raise.
        """
        if session is None:
            return cls({})
        return cls(
            getattr(session, "state", {}),
            session_id=str(getattr(session, "id", "") or ""),
        )

    @classmethod
    def for_state(cls, state: Any, *, session_id: str = "") -> StateStore:
        """Build a store backed by an arbitrary state dict."""
        return cls(state, session_id=session_id)

    # -- Internal helpers -----------------------------------------------

    def _get(self, key: str, default: Any = None) -> Any:
        return read(self._state, key, default)

    # -- Read: pin -------------------------------------------------------

    def pin_current_task(self) -> str:
        """Return ``goldfive.current_task_id``, or ``""`` when unset.

        Replaces ``state.get('goldfive.current_task_id', '')`` with
        a typed accessor.
        """
        value = self._get(KEY_CURRENT_TASK_ID, "")
        if isinstance(value, str):
            return value
        return ""

    def pin_current_task_title(self) -> str:
        """Return ``goldfive.current_task_title``, or ``""``."""
        value = self._get(KEY_CURRENT_TASK_TITLE, "")
        if isinstance(value, str):
            return value
        return ""

    def pin_current_task_revision(self) -> int:
        """Return the revision stamp on the current pin (0 when unset)."""
        return read_current_task_revision(self._state)

    # -- Write: pin ------------------------------------------------------

    def set_pin_current_task(
        self,
        task_id: str,
        *,
        source: BindingSource = BindingSource.UNKNOWN,
        revision: int | None = None,
        title: str = "",
    ) -> None:
        """Stamp the current-task pin.

        ``source`` is the :class:`BindingSource` documenting which
        ladder-rung / callback wrote the pin. Currently used for
        observability only — pre-Wave-A Phase 2's full migration would
        have wired it through every catalogued writer for full
        attribution. Passing ``BindingSource.UNKNOWN`` is fine for
        code paths whose attribution hasn't been migrated yet.

        ``revision`` (when provided) stamps
        ``goldfive.current_task_revision`` alongside the id.
        ``None`` leaves the existing revision untouched.

        No-op when ``task_id`` is empty — callers should use
        :meth:`clear_pin_current_task` to clear.
        """
        if not task_id:
            return
        if not isinstance(self._state, dict):
            # Read-only state (e.g. a MappingProxyType view); silently
            # drop the write so the caller's defensive path remains
            # safe. Production state is always a mutable dict.
            return
        # Reuse the module-level primitives so the ``goldfive.*``-prefix
        # assertion still fires; this keeps the store's writes funneled
        # through the same place the catalog is verified against.
        write(self._state, KEY_CURRENT_TASK_ID, str(task_id))
        if title:
            write(self._state, KEY_CURRENT_TASK_TITLE, str(title))
        if revision is not None:
            stamp_current_task_revision(self._state, int(revision))
        # ``source`` is recorded as part of the binding registry below
        # rather than on the pin slot itself; the pin slot pre-dates
        # the source vocabulary.
        _ = source  # documented for migration; not stored on the pin slot

    def clear_pin_current_task(self) -> None:
        """Clear the current-task pin slots. Idempotent."""
        if not isinstance(self._state, dict):
            return
        clear_current_task(self._state)

    # -- Read: active steer ---------------------------------------------

    def get_active_steer(self) -> ActiveSteer | None:
        """Return the active steer, or ``None`` when no steer is set.

        Replaces the four-key dance
        ``state.get(KEY_ACTIVE_STEER_BODY, '')`` etc. with one typed
        read. ``None`` is returned when the body is empty (the canonical
        "no steer" signal) so callers can ``if store.get_active_steer():``
        without re-checking ``.body``.
        """
        body = self._get(KEY_ACTIVE_STEER_BODY, "")
        if not isinstance(body, str) or not body:
            return None
        try:
            at_turn = int(self._get(KEY_ACTIVE_STEER_AT_TURN, 0) or 0)
        except (TypeError, ValueError):
            at_turn = 0
        author_raw = self._get(KEY_ACTIVE_STEER_AUTHOR, "")
        source_raw = self._get(KEY_ACTIVE_STEER_SOURCE, "")
        return ActiveSteer(
            body=body,
            at_turn=at_turn,
            author=str(author_raw or ""),
            source=str(source_raw or ""),
        )

    # -- Read: goals summary --------------------------------------------

    def goals_summary(self) -> str:
        """Return ``goldfive.goals_summary``, or ``""`` when unset.

        Pre-formatted comma-joined string maintained by
        :func:`refresh_goals_summary`. The planner consumes this value
        verbatim for its per-turn instruction block.
        """
        value = self._get(KEY_GOALS_SUMMARY, "")
        if isinstance(value, str):
            return value
        return ""

    # -- Read: cancelled function-call ids -------------------------------

    def cancelled_function_call_ids(self) -> list[str]:
        """Return the list of cancelled ``function_call`` ids.

        Reuses :func:`read_cancelled_function_call_ids` so the list-shape
        guard (non-list -> ``[]``) is centralised.
        """
        return read_cancelled_function_call_ids(self._state)

    # -- Read: correction ------------------------------------------------

    def get_correction(self, agent_name: str, task_id: str) -> Any:
        """Return the pending-correction value for an ``(agent, task)``.

        The shape is whatever
        :mod:`goldfive._correction_injection` writes — typically a
        :class:`Mapping` with ``superseded_task_id`` /
        ``revision_number`` / etc. (rendered via
        :func:`goldfive.adapters.adk_llm_instrumentation.format_correction_block`)
        but tests / external callers may have written a pre-rendered
        string. ``None`` means no pending correction.
        """
        if not agent_name or not task_id:
            return None
        # Local import to avoid a hard import cycle: the adapter module
        # imports back from this module.
        from goldfive.adapters import adk_llm_instrumentation as _instr  # noqa: PLC0415

        key = _instr.pending_correction_key(agent_name, task_id)
        return self._get(key, None)

    def has_correction(self, agent_name: str, task_id: str) -> bool:
        """True when a pending correction exists for ``(agent, task)``."""
        return self.get_correction(agent_name, task_id) is not None

    def iter_corrections_for_agent(self, agent_name: str) -> list[str]:
        """Return every task_id with a pending correction for ``agent_name``.

        Used by pin-ladder signal 6 to enumerate correction targets
        without rebuilding the prefix-matching loop at every call site.
        """
        if not agent_name or not isinstance(self._state, Mapping):
            return []
        # Strip a compound prefix so callers passing the bare or the
        # compound form both find the writer's bare-form keys. Mirrors
        # the matching the existing ``_task_from_pending_correction``
        # helper does inline today.
        bare = agent_name.rsplit(":", 1)[-1]
        prefix = f"goldfive.pending_corrections.{bare}."
        out: list[str] = []
        for key in self._state:
            if not isinstance(key, str):
                continue
            if not key.startswith(prefix):
                continue
            tid = key[len(prefix) :]
            if tid:
                out.append(tid)
        return out

    # -- Read: pending delegations --------------------------------------

    def get_pending_delegation(self, fc_id: str) -> DelegationPin | None:
        """Return the per-``function_call_id`` delegation pin, or ``None``.

        Tolerant of both legacy bare-string entries and the versioned
        ``{task_id, revision, tool_args}`` dict shape (goldfive#266 +
        F7). Callers used to inline the shape-test; this normalises.
        """
        if not fc_id:
            return None
        pend = self._get(PENDING_DELEGATIONS_KEY, None)
        if not isinstance(pend, Mapping):
            return None
        raw = pend.get(fc_id)
        if raw is None:
            return None
        if isinstance(raw, str):
            tid = raw.strip()
            if not tid:
                return None
            return DelegationPin(task_id=tid)
        if isinstance(raw, Mapping):
            tid = str(raw.get("task_id", "") or "").strip()
            if not tid:
                return None
            try:
                rev = int(raw.get("revision", 0) or 0)
            except (TypeError, ValueError):
                rev = 0
            args = raw.get("tool_args")
            if not isinstance(args, Mapping):
                args = None
            return DelegationPin(task_id=tid, revision=rev, tool_args=args)
        return None

    def iter_pending_delegations(self) -> Mapping[str, Any]:
        """Return the raw ``pending_delegations`` map (or empty mapping).

        The plugin's pin-resolution ladder needs to walk every entry
        (signal 1 iterates to find a task-id match across all parallel
        dispatches; signal 3 merges every entry's ``tool_args`` into a
        single token bag for scoring). Returns the live dict so callers
        can iterate with ``.values()`` / ``.items()`` without a copy.
        """
        pend = self._get(PENDING_DELEGATIONS_KEY, None)
        if isinstance(pend, Mapping):
            return pend
        return {}

    # -- Write: pending delegations -------------------------------------

    def set_pending_delegation(
        self,
        fc_id: str,
        *,
        task_id: str,
        revision: int = 0,
        tool_args: Mapping[str, Any] | None = None,
    ) -> None:
        """Stamp a per-``function_call_id`` delegation pin.

        V4 of the Phase 0 audit (goldfive#271) — every delegation-site
        write now lands here on goldfive ``Session.state`` rather than
        on ADK ``session.state``. The pin-resolution ladder + reporting
        handlers consult this same store, so a single write is enough.

        ``tool_args`` (when provided as a non-empty Mapping) is stamped
        alongside the pin so signal 3 of the ladder can score
        candidates against the parent's literal dispatch args (F7 /
        #265 followup). Empty / non-mapping args are dropped — the
        scorer treats those as zero-token signals.

        No-op when ``fc_id`` or ``task_id`` is empty, or when the
        backing state is not a mutable dict (defensive — production
        state is always a dict; tests sometimes pass MappingProxyType
        snapshots).
        """
        if not fc_id or not task_id:
            return
        if not isinstance(self._state, dict):
            return
        existing = self._state.get(PENDING_DELEGATIONS_KEY)
        bucket: dict[str, Any]
        if isinstance(existing, dict):
            bucket = existing
        else:
            bucket = {}
        entry: dict[str, Any] = {
            "task_id": str(task_id),
            "revision": int(revision),
        }
        if isinstance(tool_args, Mapping) and tool_args:
            entry["tool_args"] = dict(tool_args)
        bucket[str(fc_id)] = entry
        write(self._state, PENDING_DELEGATIONS_KEY, bucket)

    # -- Read: reasoning-extracted bindings -----------------------------

    def get_reasoning_extracted_binding(
        self,
        agent_name: str,
    ) -> ReasoningBinding | None:
        """Return the most recent reasoning-extracted binding for ``agent_name``.

        Consumed by the pin-resolution ladder's signal 6 (and by future
        correction / drift logic that wants to consult the LLM judge's
        stated-intent attribution).

        Returns ``None`` when no binding exists for the agent (or when
        the recorded entry is malformed). Callers gate on confidence
        themselves; the store stamps whatever the writer recorded.

        The lookup strips compound-form prefixes the same way
        :meth:`iter_corrections_for_agent` does, so a compound
        ``"client42:agent_x"`` finds the bare ``agent_x`` binding the
        judge recorded.
        """
        if not agent_name:
            return None
        registry = self._get(REASONING_BINDINGS_KEY, None)
        if not isinstance(registry, Mapping):
            return None
        # Try the exact form, then the bare-form fallback so
        # compound-named callers (``"client:foo"``) still match a bare
        # binding the judge recorded for ``foo``.
        raw = registry.get(agent_name)
        if raw is None:
            bare = agent_name.rsplit(":", 1)[-1]
            if bare and bare != agent_name:
                raw = registry.get(bare)
        if raw is None:
            return None
        return ReasoningBinding.from_dict(raw)

    # -- Write: reasoning-extracted bindings ----------------------------

    def record_reasoning_extracted_binding(
        self,
        *,
        agent_name: str,
        task_id: str,
        confidence: float,
        recorded_at_turn: int = 0,
        run_id: str = "",
        session_id: str = "",
    ) -> ReasoningBinding | None:
        """Stamp a reasoning-extracted binding for ``agent_name``.

        Called by the steerer's reasoning observation path when
        :func:`~goldfive.drift.reasoning_judge.classify_reasoning_drift`
        returns a ``focused_task_id`` with confidence above the
        configured threshold.

        Returns the recorded :class:`ReasoningBinding` (for caller
        observability) or ``None`` when the inputs were rejected
        (empty ``agent_name`` / ``task_id``, or read-only state).
        Confidence is clamped to ``[0.0, 1.0]``.
        """
        if not agent_name or not task_id:
            return None
        if not isinstance(self._state, dict):
            return None
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        binding = ReasoningBinding(
            agent_name=str(agent_name),
            task_id=str(task_id),
            confidence=conf,
            recorded_at_turn=int(recorded_at_turn),
            run_id=str(run_id or ""),
            session_id=str(session_id or ""),
        )
        # Read-modify-write the registry so unrelated agents' bindings
        # are preserved. Tolerate a non-dict existing value (legacy
        # state shape) by replacing it with a fresh registry rather
        # than appending to a malformed structure.
        registry = self._state.get(REASONING_BINDINGS_KEY)
        if not isinstance(registry, dict):
            registry = {}
        # Stamp under the bare form so :meth:`get_reasoning_extracted_binding`
        # finds it for both compound and bare lookups. The judge owns
        # the agent-name normalisation policy; the store records the
        # form the caller supplied (callers normalise before calling).
        registry[binding.agent_name] = binding.to_dict()
        write(self._state, REASONING_BINDINGS_KEY, registry)
        return binding

    def clear_reasoning_extracted_binding(self, agent_name: str) -> None:
        """Drop the binding for ``agent_name`` (idempotent).

        Called on task transition for the agent so a binding that
        targeted the now-completed task doesn't leak forward and
        mis-pin the next invocation.
        """
        if not agent_name or not isinstance(self._state, dict):
            return
        registry = self._state.get(REASONING_BINDINGS_KEY)
        if not isinstance(registry, dict):
            return
        registry.pop(agent_name, None)
        # Also try the bare form in case the writer used compound but
        # the caller is clearing with the bare form (or vice-versa).
        bare = agent_name.rsplit(":", 1)[-1]
        if bare and bare != agent_name:
            registry.pop(bare, None)
        write(self._state, REASONING_BINDINGS_KEY, registry)

    # -- Active-invocation registry (goldfive#271 Phase 3.5 component 1) ---
    #
    # Per-session ``invocation_id -> asyncio.Task`` map. The goldfive task
    # boundary (the wrapper around each agent invocation in the ADK
    # plugin's ``before_agent_callback`` / ``after_agent_callback``
    # try/finally arc) registers the running task here on entry and
    # deregisters on exit. The steerer's
    # :meth:`request_invocation_cancel` looks up the task to fire
    # ``task.cancel()`` on the live invocation.
    #
    # Migrated off ``_GoldfiveADKPlugin._invocation_tasks`` (PR #303 →
    # Phase 3.5 #305): per-session orchestration state belongs on the
    # store, not on the adapter plugin instance. Restores the Phase 0
    # state-ownership contract.

    def register_invocation_task(
        self,
        invocation_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        """Register the running asyncio.Task driving ``invocation_id``.

        Called from the goldfive boundary wrapper at entry. No-op when
        ``invocation_id`` is empty or the store has no session id (e.g.
        tests that construct a bare-state store).
        """
        if not invocation_id or not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _ACTIVE_INVOCATION_TASKS.setdefault(self._session_id, {})
        bucket[str(invocation_id)] = task

    def deregister_invocation_task(self, invocation_id: str) -> None:
        """Drop ``invocation_id`` from the registry. Idempotent.

        Called from the goldfive boundary wrapper's ``finally`` clause
        at exit. The bucket itself is left in place; cleanup of the
        outer entry happens in :meth:`clear_active_invocations`.
        """
        if not invocation_id or not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _ACTIVE_INVOCATION_TASKS.get(self._session_id)
        if bucket is None:
            return
        bucket.pop(str(invocation_id), None)

    def get_invocation_task(self, invocation_id: str) -> asyncio.Task[Any] | None:
        """Return the registered task for ``invocation_id``, or ``None``."""
        if not invocation_id or not self._session_id:
            return None
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _ACTIVE_INVOCATION_TASKS.get(self._session_id)
        if bucket is None:
            return None
        return bucket.get(str(invocation_id))

    def active_invocation_ids(self) -> list[str]:
        """Return the ids of every currently-registered invocation.

        Diagnostic / test helper. Empty list when no session id or no
        registered invocations.
        """
        if not self._session_id:
            return []
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _ACTIVE_INVOCATION_TASKS.get(self._session_id)
        if bucket is None:
            return []
        return list(bucket.keys())

    def clear_active_invocations(self) -> None:
        """Drop every registered task for this session. Idempotent.

        Called from the adapter's dispatch teardown so a stale handle
        cannot target the next invocation. Also clears the companion
        cancel-requested registry (goldfive#242) so a fresh dispatch
        on the same session id starts with a clean slate.
        """
        if not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            _ACTIVE_INVOCATION_TASKS.pop(self._session_id, None)
            _CANCEL_REQUESTED_INVOCATIONS.pop(self._session_id, None)
            _SUPERSEDE_PENDING_INVOCATIONS.pop(self._session_id, None)

    # -- Cancel-requested registry (goldfive#242) -----------------------
    #
    # Closes the iter-11D race between
    # :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`
    # (synchronous flag flip) and
    # :meth:`active_invocation_ids` (transitions to empty only AFTER
    # ADK winds down each cancelled invocation, ~4-8s later). The
    # late-drift gate consults this set; a drift that fires during that
    # window sees ``cancel_requested_invocation_ids()`` non-empty and
    # is treated as late, even though the active-task registry still
    # lists the cancelled invocation.

    def mark_invocation_cancel_requested(self, invocation_id: str) -> None:
        """Stamp ``invocation_id`` as having a pending cancel request.

        Called synchronously from the top of
        :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`
        before any plugin / async work. Idempotent; multiple cancel
        requests for the same id collapse to one entry.

        No-op when ``invocation_id`` is empty or the store has no
        session id (e.g. tests that construct a bare-state store).
        """
        if not invocation_id or not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _CANCEL_REQUESTED_INVOCATIONS.setdefault(self._session_id, set())
            bucket.add(str(invocation_id))

    def is_invocation_cancel_requested(self, invocation_id: str) -> bool:
        """Return True iff a cancel was requested for ``invocation_id``."""
        if not invocation_id or not self._session_id:
            return False
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _CANCEL_REQUESTED_INVOCATIONS.get(self._session_id)
        if bucket is None:
            return False
        return str(invocation_id) in bucket

    def cancel_requested_invocation_ids(self) -> list[str]:
        """Return the ids of every invocation with a pending cancel.

        Empty list when no session id or no pending cancels.
        """
        if not self._session_id:
            return []
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _CANCEL_REQUESTED_INVOCATIONS.get(self._session_id)
        if bucket is None:
            return []
        return list(bucket)

    # -- Supersede-pending registry (issue #405 LOW #7) -----------------
    #
    # Per-invocation isolation for the goldfive-internal supersede
    # marker. The legacy ``session._supersede_pending`` bool is a
    # session-scoped flip that the executor clears at the top of every
    # overlay iteration as a defensive workaround (see the comment at
    # ``SequentialExecutor._run_overlay`` lines ~1009-1022). Under
    # concurrent overlay iterations from different invocations on the
    # same session, that global clear could mask a true supersede from
    # another invocation. The per-invocation set below mirrors the
    # ``_CANCEL_REQUESTED_INVOCATIONS`` shape so each invocation's
    # supersede state is isolated and cannot be cleared out-of-band by
    # an unrelated iteration's defensive wipe.
    #
    # Both registries are populated/consumed; the bool is kept for
    # back-compat with the existing supersede-cancel tests and the
    # empty-resolver fallback in
    # :meth:`DriftObserver._cancel_inflight_for_revision` (no
    # invocation id to anchor a registry entry, so the bool acts as
    # a session-scope sentinel). Readers should prefer the set when
    # they have an invocation_id and fall back to the bool otherwise.
    # The dual-signal design is transitional; see issue #430 for the
    # follow-up to retire the bool entirely.

    def mark_supersede_pending(self, invocation_id: str) -> None:
        """Stamp ``invocation_id`` as part of an in-flight supersede cancel.

        Called from
        :meth:`~goldfive.steerer.DefaultSteerer._cancel_inflight_for_revision`
        (via :meth:`request_invocation_cancel`) right before the cancel
        lands on the registered asyncio.Task. Idempotent; multiple
        supersede requests for the same id collapse to one entry.

        No-op when ``invocation_id`` is empty or the store has no
        session id.
        """
        if not invocation_id or not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _SUPERSEDE_PENDING_INVOCATIONS.setdefault(self._session_id, set())
            bucket.add(str(invocation_id))

    def clear_supersede_pending(self, invocation_id: str) -> None:
        """Drop ``invocation_id`` from the supersede-pending set.

        Called by the executor's cancelled branch after consuming the
        supersede signal (treating the cancel as a restart trigger
        instead of an abort). Idempotent — clearing an absent id is
        silent.
        """
        if not invocation_id or not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _SUPERSEDE_PENDING_INVOCATIONS.get(self._session_id)
            if bucket is None:
                return
            bucket.discard(str(invocation_id))
            if not bucket:
                _SUPERSEDE_PENDING_INVOCATIONS.pop(self._session_id, None)

    def is_supersede_pending(self, invocation_id: str) -> bool:
        """Return True iff a supersede cancel was stamped for ``invocation_id``.

        Per-invocation read; does not consult the legacy session-scoped
        ``_supersede_pending`` bool. Callers that need union-of-signals
        semantics should also check the bool.
        """
        if not invocation_id or not self._session_id:
            return False
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _SUPERSEDE_PENDING_INVOCATIONS.get(self._session_id)
        if bucket is None:
            return False
        return str(invocation_id) in bucket

    def supersede_pending_invocation_ids(self) -> list[str]:
        """Return the ids of every invocation with a pending supersede.

        Empty list when no session id or no pending supersedes.
        """
        if not self._session_id:
            return []
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _SUPERSEDE_PENDING_INVOCATIONS.get(self._session_id)
        if bucket is None:
            return []
        return list(bucket)

    def has_any_supersede_pending(self) -> bool:
        """Return True iff any invocation on this session is supersede-pending.

        Convenience for callers (the executor overlay loop) that don't
        track a specific invocation_id but want to know "did the steerer
        stamp a supersede for *something* on this session" — used to
        disambiguate a cancelled invocation as internal-supersede vs.
        external. See :meth:`supersede_pending_invocation_ids`.
        """
        if not self._session_id:
            return False
        with _ACTIVE_INVOCATION_LOCK:
            bucket = _SUPERSEDE_PENDING_INVOCATIONS.get(self._session_id)
        return bool(bucket)

    def clear_all_supersede_pending(self) -> None:
        """Drop every supersede-pending entry for this session. Idempotent.

        Used by the executor's cancelled branch when it has consumed
        the supersede (treating the cancel as a restart) but doesn't
        know which specific invocation_id was the supersede target.
        Mirrors the legacy ``session._supersede_pending = False`` clear
        but scoped per-session via the registry.
        """
        if not self._session_id:
            return
        with _ACTIVE_INVOCATION_LOCK:
            _SUPERSEDE_PENDING_INVOCATIONS.pop(self._session_id, None)

    # -- Active drift conditions (goldfive#271 PR1) ---------------------

    def active_drifts(self) -> list[Drift]:
        """Return all in-flight drift conditions on this session.

        Backed by the ``goldfive.active_drifts`` slot on
        ``session.state``. Each entry is a :class:`Drift` snapshot;
        the list is freshly materialised from the underlying dict so
        callers can mutate the results without affecting subsequent
        reads.
        """
        return list_active_drifts(self._state)

    def get_active_drift(self, condition_id: str) -> Drift | None:
        """Look up an in-flight condition by id, or ``None`` when absent."""
        return get_active_drift(self._state, condition_id)

    def open_or_escalate_drift(
        self,
        *,
        kind: Any,
        task_id: str,
        agent_id: str,
        turn_id: str,
        severity: Any,
    ) -> Drift:
        """Open a new condition or escalate an existing one (goldfive#271 PR1).

        Routes through the module-level :func:`open_or_escalate_drift`.
        Same-turn re-emits for the same kind+task+agent collapse onto
        the same condition_id; severity bumps are monotonic. The
        returned :class:`Drift` carries the lifecycle / prev_severity
        the caller stamps onto ``DriftDetected``.
        """
        if not isinstance(self._state, MutableMapping):
            # No mutable state — fall back to a synthetic single-shot
            # condition so the steerer's wire path still has stable
            # condition_id / lifecycle to stamp.
            return Drift(
                condition_id=compute_condition_id(
                    kind=kind,
                    task_id=task_id,
                    agent_id=agent_id,
                    turn_id=turn_id,
                ),
                kind=kind,
                task_id=str(task_id or ""),
                agent_id=str(agent_id or ""),
                turn_id=str(turn_id or ""),
                severity=severity,
                prev_severity=None,
                lifecycle=LIFECYCLE_OPENED,
                occurrences=1,
            )
        return open_or_escalate_drift(
            self._state,
            kind=kind,
            task_id=task_id,
            agent_id=agent_id,
            turn_id=turn_id,
            severity=severity,
        )

    def resolve_drift(self, condition_id: str) -> Drift | None:
        """Mark a condition resolved + remove it from the active set."""
        if not isinstance(self._state, MutableMapping):
            return None
        return resolve_drift(self._state, condition_id)

    def escalate_to_human_intervention(self, condition_id: str) -> Drift | None:
        """Mark a condition as escalated to human intervention."""
        if not isinstance(self._state, MutableMapping):
            return None
        return escalate_drift_to_human_intervention(self._state, condition_id)


__all__ = [
    "ALL_KEYS",
    "GOLDFIVE_PREFIX",
    "KEY_ACTIVE_DRIFTS",
    "KEY_ACTIVE_STEER_AT_TURN",
    "KEY_ACTIVE_STEER_AUTHOR",
    "KEY_ACTIVE_STEER_BODY",
    "KEY_ACTIVE_STEER_SOURCE",
    "KEY_CANCELLED_FUNCTION_CALL_IDS",
    "KEY_CURRENT_PLAN_ID",
    "KEY_CURRENT_TASK_ID",
    "KEY_CURRENT_TASK_REVISION",
    "KEY_CURRENT_TASK_TITLE",
    "KEY_GOALS_SUMMARY",
    "KEY_PROCESSED_STEER_IDS",
    "LIFECYCLE_ESCALATING",
    "LIFECYCLE_HUMAN_INTERVENTION_REQUIRED",
    "LIFECYCLE_OPENED",
    "LIFECYCLE_RESOLVED",
    "LIFECYCLE_VALUES",
    "PENDING_DELEGATIONS_KEY",
    "PROCESSED_STEER_IDS_CAP",
    "REASONING_BINDINGS_KEY",
    "ActiveSteer",
    "BindingSource",
    "DelegationPin",
    "Drift",
    "ReasoningBinding",
    "StateStore",
    "append_cancelled_function_call_ids",
    "clear",
    "clear_active_steer",
    "clear_current_task",
    "compute_condition_id",
    "escalate_drift_to_human_intervention",
    "format_goals_summary",
    "get_active_drift",
    "has_processed_steer_id",
    "list_active_drifts",
    "open_or_escalate_drift",
    "read",
    "read_cancelled_function_call_ids",
    "read_current_task_revision",
    "record_processed_steer_id",
    "refresh_goals_summary",
    "resolve_drift",
    "rotate_current_task_id",
    "set_active_steer",
    "set_current_plan",
    "set_current_task",
    "stamp_current_task_revision",
    "sync_current_task_from_transition",
    "write",
]
