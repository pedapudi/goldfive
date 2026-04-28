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

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from goldfive.types import DriftKind, DriftSeverity, Goal, Plan, Task, TaskStatus

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
    "PROCESSED_STEER_IDS_CAP",
    "Drift",
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
