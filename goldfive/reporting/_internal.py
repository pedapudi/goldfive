"""Shared utilities private to the :mod:`goldfive.reporting` package.

Three buckets of helpers that more than one of
:mod:`goldfive.reporting.handlers` / :mod:`goldfive.reporting.rendering`
reaches into:

* Coercion helpers (:func:`_str`, :func:`_float`, :func:`_bool`,
  :func:`_int`) — defensive parsing of model-supplied tool args.
* Plan / pin resolution (:func:`_find_task_in_session`,
  :func:`_resolve_task_id`, :func:`_resolve_effective_task_id`,
  :func:`_reroute_if_superseded`, the pin-freshness classifier) — the
  policy that decides which task a handler actually drives, including
  the goldfive#237 supersession follow and the goldfive#266 stale-pin
  refusal.
* Sink emitters (:func:`_emit_task_transition_refused`,
  :func:`_emit_task_declaration_received`,
  :func:`_emit_approval_requested`) — observability fan-out that
  handlers fire alongside the steerer drive.

The :func:`_classify_pin_freshness` outcome vocabulary
(``"match"`` / ``"stale_replace"`` / ``"stale_correct"`` /
``"stale_ambiguous"``) is documented inline; the handler-side
``_classify_and_route_pin`` (in :mod:`goldfive.reporting.handlers`)
turns those classifications into a routing/refusal decision.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from goldfive.types import SupersessionKind, TaskStatus

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.types import Session, Task

log = logging.getLogger(__name__)


# Canonical ack — handlers that don't carry the F1 directive payload
# (drift-only / declaration / refused) return a shallow copy of this
# sentinel so callers can mutate freely without aliasing.
_ACK: dict[str, Any] = {"acknowledged": True}


# ---------------------------------------------------------------------------
# Idempotency / invalid-transition machinery (goldfive#201)
# ---------------------------------------------------------------------------
#
# Each task-scoped reporting handler consults the task's current status
# before driving the steerer. Three outcomes:
#
# 1. **Real transition** — current status is a legal source for the
#    tool's target transition; the handler invokes the steerer and
#    returns ``{"acknowledged": True}``. Terminal transitions also
#    rotate ``goldfive.current_task_id`` to the next assigned
#    PENDING/RUNNING task (or clear it).
# 2. **Idempotent no-op** — current status already matches what the
#    call would move the task to (e.g. ``report_task_completed`` on a
#    COMPLETED task). Handler returns
#    ``{"acknowledged": True, "idempotent": True, "current_status": ...}``
#    without mutating state. This is the goldfive#201 fix: retries
#    from a confused model no longer masquerade as tool-loop spam.
# 3. **Invalid transition** — current status cannot legally transition
#    under this tool (e.g. ``report_task_started`` on a COMPLETED
#    task). Handler returns
#    ``{"acknowledged": False, "error": "invalid_transition",
#       "current_status": ..., "attempted": ...}``
#    as a real "agent is confused about state" signal. Loop-detector
#    owners can surface this directly; it's distinct from a benign
#    retry.
#
# See ``docs/design/TASK-LIFECYCLE.md`` for the status-machine contract.


# Tool-name → the status the call would transition the task INTO
# (i.e. "what does success look like"). Used to detect idempotent
# retries: when ``current_status`` already equals the target, the call
# is an ack-only no-op.
_TOOL_TARGET_STATUS: dict[str, TaskStatus] = {
    "report_task_started": TaskStatus.RUNNING,
    "report_task_progress": TaskStatus.RUNNING,  # progress is a liveness tick on a RUNNING task
    "report_task_completed": TaskStatus.COMPLETED,
    "report_task_failed": TaskStatus.FAILED,
    "report_task_blocked": TaskStatus.BLOCKED,
    "report_awaiting_approval": TaskStatus.BLOCKED,  # mapped onto BLOCKED by the steerer
}

# Which source statuses are *legal* starting points for each tool.
# Anything else is either an idempotent no-op (see above) or an
# ``invalid_transition``. Built from ``docs/design/TASK-LIFECYCLE.md``
# §"Status transitions". PENDING/RUNNING bookkeeping statuses are the
# canonical sources; terminal statuses are never a legal source for a
# different transition.
_TOOL_VALID_SOURCES: dict[str, frozenset[TaskStatus]] = {
    "report_task_started": frozenset({TaskStatus.PENDING, TaskStatus.BLOCKED}),
    "report_task_progress": frozenset({TaskStatus.RUNNING}),
    "report_task_completed": frozenset(
        {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.BLOCKED}
    ),
    "report_task_failed": frozenset({TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.BLOCKED}),
    "report_task_blocked": frozenset({TaskStatus.PENDING, TaskStatus.RUNNING}),
    "report_awaiting_approval": frozenset(
        {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.BLOCKED}
    ),
}

# Terminal statuses — duplicated here to avoid a runtime import of
# ``TERMINAL_TASK_STATUSES`` from ``goldfive.types``; the value set is
# pinned by ``TaskStatus`` and guarded by a test.
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.NOT_NEEDED,
    }
)


# ---------------------------------------------------------------------------
# Argument coercion helpers
# ---------------------------------------------------------------------------


def _str(args: dict[str, Any], key: str, default: str = "") -> str:
    v = args.get(key, default)
    if v is None:
        return default
    return str(v)


def _float(args: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = args.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(args: dict[str, Any], key: str, default: bool = True) -> bool:
    v = args.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes"}
    return bool(v) if v is not None else default


def _int(args: dict[str, Any], key: str, default: int = 0) -> int:
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Plan / pin resolution
# ---------------------------------------------------------------------------


def _find_task_in_session(session: Session, task_id: str) -> Task | None:
    """Return the task in ``session.plan`` with ``task_id``, or ``None``."""
    plan = getattr(session, "plan", None)
    if plan is None or not task_id:
        return None
    for t in getattr(plan, "tasks", ()) or ():
        if getattr(t, "id", "") == task_id:
            return t
    return None


def _resolve_task_id(args: dict[str, Any], session: Session) -> str:
    """Return the task_id to act on, falling back to session state.

    Order of precedence (goldfive#191):

    1. ``args["task_id"]`` — explicit model-provided id always wins.
    2. ``StateStore.pin_current_task()`` — the id pinned
       by the adapter's ``before_agent_callback`` when the current
       sub-agent has exactly one PENDING/RUNNING task assigned to
       it. Closes the loop where the LLM's tool call omits the
       arg but the orchestration layer knew the answer.

    Empty string when neither source supplies a value — caller
    should short-circuit with the canonical ``missing_task_id``
    error in that case.
    """
    return _resolve_task_id_with_source(args, session)[0]


def _resolve_task_id_with_source(args: dict[str, Any], session: Session) -> tuple[str, str]:
    """Resolve ``task_id`` and report whether it came from args or state.

    Returns ``(task_id, source)`` where ``source`` is one of:

    * ``"llm_report"`` — the LLM supplied an explicit non-empty
      ``task_id`` arg.
    * ``"handler_default"`` — the arg was absent / empty / None and the
      handler defaulted to the adapter-stamped pin
      (:meth:`StateStore.pin_current_task`). This is the
      goldfive#191 path.
    * ``""`` — neither source resolved a value (caller short-circuits
      with the canonical ``missing_task_id`` rejection).

    Used by the goldfive#251 R4 ``TaskTransitioned`` emit sites to
    distinguish a direct LLM-driven transition from one that piggy-
    backed on the adapter pin. The tuple-returning variant is
    additive; existing callers stay on :func:`_resolve_task_id`.

    Phase 2.1 of goldfive#271 — the read funnels through
    :class:`~goldfive.state_store.StateStore` so the
    handler is decoupled from goldfive ``Session.state``'s on-disk
    key strings.
    """
    raw = args.get("task_id")
    if raw is not None:
        task_id = str(raw).strip()
        if task_id:
            return task_id, "llm_report"
    from goldfive.state_store import StateStore

    store = StateStore.for_session(session)
    fallback = store.pin_current_task().strip()
    if fallback:
        log.debug(
            "reporting: defaulted task_id=%s from StateStore",
            fallback,
        )
        return fallback, "handler_default"
    return "", ""


def _resolve_effective_task_id(session: Session, task_id: str) -> str:
    """Follow the plan's ``supersedes`` chain from a terminal task to its live replacement.

    goldfive#237. The scenario: ``planner.refine`` has replaced
    ``research_solar`` (now FAILED) with ``research_solar_corrected``
    (supersedes=``research_solar``, PENDING). The agent keeps its
    previously-pinned ``current_task_id=research_solar`` and calls
    ``report_task_progress`` with that id. Without this resolver the
    handler would reject the call as an invalid transition from a
    terminal status — a direct contradiction of "agent is actively
    working, report is rejected".

    Rules:

    * If ``task_id`` refers to a task that is NOT terminal, return it
      unchanged (the pre-#237 behaviour).
    * If ``task_id`` refers to a terminal task, walk the plan looking
      for any task whose ``supersedes`` equals ``task_id``. When
      found, recurse (so A → B → C chains collapse to C), capping the
      walk at a small depth for loop safety.
    * If no replacement exists, return ``task_id`` unchanged — the
      handler's existing terminal-state rejection path takes over,
      which is still the right signal when the planner didn't produce
      a replacement.

    Empty / unknown task_id is returned unchanged.
    """
    if not task_id:
        return task_id
    plan = getattr(session, "plan", None)
    if plan is None:
        return task_id
    tasks = getattr(plan, "tasks", None) or ()
    # Index once — handlers call this up to five times per tool call.
    by_id: dict[str, Task] = {str(getattr(t, "id", "") or ""): t for t in tasks}
    # Build a reverse map supersedes -> new once. goldfive#251: a
    # ``CORRECT``-kind supersedes link DOES NOT route — the old task's
    # completion is historical fact (it is a COMPLETED node retained
    # for DAG history) and a late report on it is an idempotent no-op
    # on a completed task, not a retroactive write to the correction.
    # Only ``REPLACE`` / ``UNSPECIFIED`` (legacy) links reroute.
    replacements: dict[str, str] = {}
    for t in tasks:
        sup = str(getattr(t, "supersedes", "") or "").strip()
        tid = str(getattr(t, "id", "") or "").strip()
        if not sup or not tid:
            continue
        kind = getattr(t, "supersedes_kind", SupersessionKind.UNSPECIFIED)
        if kind is SupersessionKind.CORRECT:
            continue
        replacements[sup] = tid
    current = task_id
    visited: set[str] = {current}
    for _ in range(8):  # hard cap: plan-revision chains don't grow deep
        task = by_id.get(current)
        if task is None:
            return current
        if task.status not in _TERMINAL_STATUSES:
            return current
        nxt = replacements.get(current, "")
        if not nxt or nxt in visited:
            return current
        visited.add(nxt)
        current = nxt
    return current


def _reroute_if_superseded(session: Session, task_id: str, tool_name: str) -> str:
    """Resolve ``task_id`` through the plan's supersession chain with logging.

    Thin wrapper over :func:`_resolve_effective_task_id` that adds an
    INFO-level log record the first time a call is rerouted so
    operators can see the re-pin happening live in sessions. Idempotent
    beyond the log line: handlers can safely call it every dispatch.
    """
    resolved = _resolve_effective_task_id(session, task_id)
    if resolved != task_id:
        log.info(
            "reporting: %s called with superseded task_id=%s; routing to replacement=%s",
            tool_name,
            task_id,
            resolved,
        )
    return resolved


# ---------------------------------------------------------------------------
# Pin freshness classification (goldfive#266 / pin versioning)
# ---------------------------------------------------------------------------


def _read_pin_revision(session: Session) -> int | None:
    """Return the pin's stamped revision from session state, or ``None``.

    ``None`` indicates "no stamp present" — distinct from a stamp of 0.
    Custom adapters / legacy sessions / tests that pre-date #266 won't
    have stamped the key; those callers must keep working unchanged
    against the legacy ``_reroute_if_superseded`` semantics. Callers
    that observe ``None`` should treat the pin as fresh (the stamp
    isn't load-bearing).

    A stamp of ``0`` means "the pin was set under the initial plan
    revision" and is treated by the classifier as a stale pin against
    any plan with ``revision_index > 0`` — that's the actual
    versioning semantics, distinct from "no stamp".
    """
    from goldfive import state_store as _ostate

    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return None
    if _ostate.KEY_CURRENT_TASK_REVISION not in state:
        return None
    return _ostate.read_current_task_revision(state)


def _read_plan_revision(session: Session) -> int:
    """Return ``session.plan.revision_index`` as an int, default 0.

    Tolerant of missing / mock plans (test stubs, custom executors).
    """
    plan = getattr(session, "plan", None)
    try:
        return max(0, int(getattr(plan, "revision_index", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _supersession_successor(session: Session, task_id: str) -> tuple[str, SupersessionKind]:
    """Return ``(successor_task_id, kind)`` for ``task_id``.

    Walks the plan looking for a task whose ``supersedes`` equals
    ``task_id``. Empty successor + ``UNSPECIFIED`` when no successor
    exists. Used by the report-time pin classifier — ``REPLACE``
    successors route, ``CORRECT`` successors refuse, no successor also
    refuses (ambiguity).
    """
    plan = getattr(session, "plan", None)
    if plan is None or not task_id:
        return "", SupersessionKind.UNSPECIFIED
    for t in getattr(plan, "tasks", ()) or ():
        sup = str(getattr(t, "supersedes", "") or "").strip()
        if sup != task_id:
            continue
        successor_id = str(getattr(t, "id", "") or "").strip()
        kind = getattr(t, "supersedes_kind", SupersessionKind.UNSPECIFIED)
        return successor_id, kind
    return "", SupersessionKind.UNSPECIFIED


# Outcomes of :func:`_classify_pin_freshness`:
#
# * ``"match"`` — pin revision equals (or exceeds — defensive) current
#   plan revision. Handler proceeds on the pin's task_id.
# * ``"stale_replace"`` — pin is older than the current plan revision
#   and the pin's task has a REPLACE-kind supersedes successor. Handler
#   routes onto the successor (existing pre-#266 behaviour).
# * ``"stale_correct"`` — pin is older + the pin's task has a
#   CORRECT-kind supersedes successor. Handler REFUSES + emits a
#   ``task_transition_refused`` sink event. The old task's terminal
#   state is historical fact and the correction is a separate work
#   unit; transitioning the old task out of terminal would either
#   destroy fact or shadow the correction.
# * ``"stale_ambiguous"`` — pin is older + the pin's task has no
#   supersedes successor. Handler REFUSES; an operator must
#   disambiguate. Same shape as stale_correct.
_PinFreshness = str


def _classify_pin_freshness(
    session: Session,
    task_id: str,
    *,
    pin_revision: int,
    current_revision: int,
) -> tuple[_PinFreshness, str, SupersessionKind]:
    """Classify the relationship between a pinned task_id and the live plan.

    Returns ``(freshness, successor_task_id, supersedes_kind)``. The
    successor is empty for ``"match"`` and ``"stale_ambiguous"``.
    """
    if pin_revision >= current_revision:
        # Future revisions (pin_revision > current_revision) shouldn't
        # happen under a single executor — the writer reads
        # plan.revision_index that the report-time reader sees later,
        # and revisions only ever increment. If it does, treat as a
        # match (trust the pin) rather than refusing.
        if pin_revision > current_revision:
            log.debug(
                "reporting: pin_revision=%d > current=%d for task_id=%s (unexpected; trusting pin)",
                pin_revision,
                current_revision,
                task_id,
            )
        return "match", "", SupersessionKind.UNSPECIFIED

    # pin_revision < current_revision — the pin was set under an older
    # plan. Consult the supersedes graph.
    successor_id, kind = _supersession_successor(session, task_id)
    if successor_id and kind is SupersessionKind.REPLACE:
        return "stale_replace", successor_id, kind
    if successor_id and kind is SupersessionKind.CORRECT:
        return "stale_correct", successor_id, kind
    if successor_id and kind is SupersessionKind.UNSPECIFIED:
        # Legacy plans pre-#258 didn't carry a supersedes_kind enum.
        # Preserve the historical "treat unspecified as REPLACE" rerouting
        # path used by ``_resolve_effective_task_id`` so legacy data
        # keeps reaching the right handler.
        return "stale_replace", successor_id, kind
    return "stale_ambiguous", "", SupersessionKind.UNSPECIFIED


# ---------------------------------------------------------------------------
# Transition classifier + pin rotation
# ---------------------------------------------------------------------------


def _classify_transition(
    *,
    tool_name: str,
    current_status: TaskStatus,
) -> str:
    """Return ``"idempotent"``, ``"invalid"``, or ``"transition"``.

    * ``"idempotent"`` — the call is a no-op because the task is
      already in the status this tool would move it to (or, for
      ``report_task_progress``, the task is already RUNNING and a
      progress tick is always legal there). Handler returns an
      idempotent ACK without mutating state.
    * ``"invalid"`` — the call cannot legally transition from
      ``current_status``. Handler returns an ``invalid_transition``
      error.
    * ``"transition"`` — the call is a legitimate transition; the
      handler drives the steerer normally.
    """
    target = _TOOL_TARGET_STATUS.get(tool_name)
    if target is not None and current_status == target:
        # Special case: ``report_task_progress`` on a RUNNING task is
        # always a no-op (a progress tick has no status mutation, but we
        # treat it as idempotent so the caller sees the same shape).
        return "idempotent"
    valid = _TOOL_VALID_SOURCES.get(tool_name)
    if valid is not None and current_status in valid:
        return "transition"
    # Falls through to invalid — covers terminal statuses under every
    # tool (except when current == target above) and the degenerate
    # case of ``report_task_progress`` on PENDING etc.
    return "invalid"


def _rotate_after_terminal(
    session: Session,
    completed_task: Task,
) -> None:
    """Advance the current-task pin after a terminal transition.

    Delegates to :func:`goldfive.state_store.rotate_current_task_id`.
    Imported lazily so this module stays ADK-free and dodges the
    import-cycle risk between :mod:`goldfive.reporting` and
    :mod:`goldfive.state_store` (both are imported from
    :mod:`goldfive.steerer`).
    """
    from goldfive import state_store as _ostate
    from goldfive.state_store import StateStore

    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return
    plan = getattr(session, "plan", None)
    agent_name = str(getattr(completed_task, "assignee_agent_id", "") or "")

    # Only rotate if the terminal task was the one we'd been pointing
    # at — otherwise some other caller owns the pin and we'd clobber
    # theirs.
    pinned = StateStore.for_session(session).pin_current_task().strip()
    this_id = str(getattr(completed_task, "id", "") or "")
    if pinned and pinned != this_id:
        return

    _ostate.rotate_current_task_id(state, plan, agent_name)


# ---------------------------------------------------------------------------
# Plan-stability barrier
# ---------------------------------------------------------------------------


async def _await_plan_stable(session: Session, steerer: Steerer) -> None:
    """Block briefly until the steerer's plan mutation region is idle.

    Duck-types ``steerer.plans._wait_plan_stable``: any Steerer
    implementation that exposes a ``plans`` component with an async
    ``_wait_plan_stable(session)`` method gets a consistent plan-state
    read; implementations that don't (custom Steerers, test stubs)
    silently fall through and observe whatever plan state happens to
    be installed at call time — the pre-a4 behaviour. Atomicity is
    best-effort (the helper times out after a short interval and
    proceeds anyway), so this is a soft barrier, not a hard mutex.
    """
    plans = getattr(steerer, "plans", None)
    waiter = getattr(plans, "_wait_plan_stable", None) if plans is not None else None
    if not callable(waiter):
        return
    try:
        await waiter(session)
    except Exception as exc:  # noqa: BLE001 — barrier must never break a report
        log.debug(
            "reporting._await_plan_stable: %s (proceeding with current plan state)",
            exc,
        )


# ---------------------------------------------------------------------------
# Sink emitters
# ---------------------------------------------------------------------------


async def _emit_task_transition_refused(
    *,
    session: Session,
    steerer: Steerer,
    task_id: str,
    attempted_from: TaskStatus,
    attempted_to: TaskStatus,
    reason: str,
    pin_revision: int,
    current_revision: int,
    agent_name: str = "",
    invocation_id: str = "",
) -> None:
    """Emit a ``TaskTransitionRefused`` proto envelope onto the sink bus.

    Fired when the report-time pin classifier refuses to drive a stale
    pin's transition because the old task either has a CORRECT-kind
    supersedes successor (history vs. correction) or no successor at
    all (ambiguity — operator must decide). The LLM still sees an
    ``{"acknowledged": True}`` response so it doesn't reason against
    the refusal; operators consume this event for the audit trail.

    Typed proto envelope — promoted from the dict shape that #266
    shipped, matching the ``InvocationCancelled`` promotion pattern
    from #262. The dict-envelope path has been removed (full
    migration; harmonograf-side ingest migration lands separately).
    """
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    from goldfive.events import emit, task_transition_refused_event

    try:
        evt = task_transition_refused_event(
            session.run_id,
            session.next_sequence(),
            task_id=task_id,
            attempted_from=attempted_from.value,
            attempted_to=attempted_to.value,
            reason=reason,
            pin_revision=int(pin_revision),
            current_revision=int(current_revision),
            agent_name=agent_name,
            invocation_id=invocation_id,
            session_id=session.id,
        )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001 — observability must never break a report
        log.debug(
            "reporting._emit_task_transition_refused: failed to emit: %s",
            exc,
        )


async def _emit_task_declaration_received(
    *,
    session: Session,
    steerer: Steerer,
    kind: str,
    task_id: str,
    reason: str,
) -> None:
    """Emit a ``TaskDeclarationReceived`` dict envelope onto the sink bus.

    goldfive#271 Phase 3: structural declarations from the agent
    (``declare_task_skipped``, ``declare_task_not_needed``) are
    observability-only — they do NOT mutate plan state. Operators see
    the declaration on the sink so they can compare it against the
    imperative ``report_task_*`` surface that follows. The dict
    envelope is the same low-cost path used for ``RefineAttempted``
    / ``RefineFailed`` (PR #264) before those are promoted to typed
    proto. A future cleanup may promote ``TaskDeclarationReceived``
    to a proper proto message; until then the dict shape with a
    well-known ``kind`` field is the contract.

    ``source_signal`` is fixed to ``"DECLARATION"`` so downstream
    consumers can distinguish a self-volunteered declaration from a
    framework-driven inference (steerer rotation, reconciler
    NOT_NEEDED stamp).
    """
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    from goldfive.events import emit, make_event

    payload = {
        "kind": str(kind),
        "task_id": str(task_id),
        "reason": str(reason),
        "source_signal": "DECLARATION",
    }
    try:
        evt = make_event(
            session.run_id,
            session.next_sequence(),
            "task_declaration_received",
            payload,
            session_id=session.id,
        )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001 — observability must never break a tool call
        log.debug(
            "reporting._emit_task_declaration_received: failed to emit: %s",
            exc,
        )


async def _emit_approval_requested(
    *,
    session: Session,
    steerer: Steerer,
    target_id: str,
    kind: str,
    prompt: str,
    task_id: str,
    metadata: dict[str, str],
) -> None:
    """Emit an ``ApprovalRequested`` through the steerer's bound sinks.

    Falls back to a no-op if the steerer lacks a sinks list (test stubs
    may drop the ``bind`` attribute). Proto-build errors are logged and
    swallowed — losing the event is better than failing the tool call.
    """
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    from goldfive.events import approval_requested_event, emit

    try:
        evt = approval_requested_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            target_id=target_id,
            kind=kind,
            prompt=prompt,
            task_id=task_id,
            metadata=metadata,
            session_id=session.id,
        )
    except Exception as exc:  # noqa: BLE001 — proto stubs may be missing in unit tests
        log.debug("approval_requested: proto event build failed: %s", exc)
        return
    try:
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("approval_requested: sink emit raised: %s", exc)


__all__ = [
    "_ACK",
    "_PinFreshness",
    "_TERMINAL_STATUSES",
    "_TOOL_TARGET_STATUS",
    "_TOOL_VALID_SOURCES",
    "_await_plan_stable",
    "_bool",
    "_classify_pin_freshness",
    "_classify_transition",
    "_emit_approval_requested",
    "_emit_task_declaration_received",
    "_emit_task_transition_refused",
    "_find_task_in_session",
    "_float",
    "_int",
    "_read_pin_revision",
    "_read_plan_revision",
    "_reroute_if_superseded",
    "_resolve_effective_task_id",
    "_resolve_task_id",
    "_resolve_task_id_with_source",
    "_rotate_after_terminal",
    "_str",
    "_supersession_successor",
]
