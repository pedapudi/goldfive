"""Reporting-tool specs and handlers.

The eight canonical reporting tools — the agent-facing contract for
driving the plan's task state machine and signalling plan mutations.
Each :class:`ReportingToolSpec` pairs a stable tool name with a JSON-schema
parameters block and an async handler. Handlers receive the decoded
arguments, the live :class:`Session`, and the bound :class:`Steerer`, and
route the call into the steerer's transition / drift pipeline.

Adapters materialise these specs into whatever native tool shape their
framework wants (ADK ``FunctionTool``, Claude Agent SDK tool blocks, …).

The eighth tool, ``report_awaiting_approval``, is the task-level half of
the human-in-the-loop approval flow described in
``docs/design/APPROVAL.md``. Its handler blocks the calling tool-call
until the control dispatcher lands an ``APPROVE`` or ``REJECT``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from goldfive.types import SupersessionKind, TaskStatus

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.types import Session, Task

log = logging.getLogger(__name__)


# Framework-agnostic async handler signature.
# Handlers receive the tool-call arguments (already decoded to a dict),
# the live Session, and the Steerer, and return a JSON-serializable dict.
ReportingHandler = Callable[
    [dict[str, Any], "Session", "Steerer"],
    Awaitable[dict[str, Any]],
]


@dataclasses.dataclass
class ReportingToolSpec:
    """Framework-agnostic spec for a reporting tool.

    Adapters translate one of these into whatever tool representation their
    underlying framework expects (e.g., an ADK ``FunctionTool`` or a Claude
    Agent SDK tool definition). The canonical set of tool names is pinned in
    :data:`REPORTING_TOOL_NAMES`.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters
    handler: ReportingHandler


# The eight canonical reporting tool names. These are a stable contract: the
# adapter must surface tools with exactly these names so that the Steerer can
# interpret them uniformly across frameworks. Do not rename.
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
    "report_awaiting_approval",
)


# ---------------------------------------------------------------------------
# Handler shims
# ---------------------------------------------------------------------------


_ACK: dict[str, Any] = {"acknowledged": True}

# Orchestration-state key the adapter stamps at delegation time
# (goldfive#191). Handlers fall back to this value when the model's
# tool call omits ``task_id``. Re-declared here rather than imported
# from :mod:`goldfive.orchestration_state` to avoid a circular import
# — the string is a stable contract shared between the adapter's
# :mod:`._adk_state_protocol`, :mod:`orchestration_state`, and this
# handler module.
_STATE_KEY_CURRENT_TASK_ID = "goldfive.current_task_id"
# Companion key (goldfive#266 / pin versioning) — the plan revision
# in effect at the moment the adapter wrote the pin. Same back-compat
# stance: a missing / malformed entry reads as 0, which matches the
# default ``Plan.revision_index`` so unrevised plans remain "fresh".
_STATE_KEY_CURRENT_TASK_REVISION = "goldfive.current_task_revision"


def _resolve_task_id(args: dict[str, Any], session: Session) -> str:
    """Return the task_id to act on, falling back to session state.

    Order of precedence (goldfive#191):

    1. ``args["task_id"]`` — explicit model-provided id always wins.
    2. ``session.state["goldfive.current_task_id"]`` — the id pinned
       by the adapter's ``before_agent_callback`` when the current
       sub-agent has exactly one PENDING/RUNNING task assigned to
       it. Closes the loop where the LLM's tool call omits the
       arg but the orchestration layer knew the answer.

    Empty string when neither source supplies a value — caller
    should short-circuit with the canonical ``missing_task_id``
    error in that case.
    """
    return _resolve_task_id_with_source(args, session)[0]


def _resolve_task_id_with_source(
    args: dict[str, Any], session: Session
) -> tuple[str, str]:
    """Resolve ``task_id`` and report whether it came from args or state.

    Returns ``(task_id, source)`` where ``source`` is one of:

    * ``"llm_report"`` — the LLM supplied an explicit non-empty
      ``task_id`` arg.
    * ``"handler_default"`` — the arg was absent / empty / None and the
      handler defaulted to the adapter-stamped pin
      (``goldfive.current_task_id``). This is the goldfive#191 path.
    * ``""`` — neither source resolved a value (caller short-circuits
      with the canonical ``missing_task_id`` rejection).

    Used by the goldfive#251 R4 ``TaskTransitioned`` emit sites to
    distinguish a direct LLM-driven transition from one that piggy-
    backed on the adapter pin. The tuple-returning variant is
    additive; existing callers stay on :func:`_resolve_task_id`.
    """
    raw = args.get("task_id")
    if raw is not None:
        task_id = str(raw).strip()
        if task_id:
            return task_id, "llm_report"
    state = getattr(session, "state", None)
    if isinstance(state, dict):
        fallback = state.get(_STATE_KEY_CURRENT_TASK_ID, "")
        if isinstance(fallback, str):
            value = fallback.strip()
        elif fallback is not None:
            value = str(fallback).strip()
        else:
            value = ""
        if value:
            return value, "handler_default"
    return "", ""


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


def _find_task_in_session(session: Session, task_id: str) -> Task | None:
    """Return the task in ``session.plan`` with ``task_id``, or ``None``."""
    plan = getattr(session, "plan", None)
    if plan is None or not task_id:
        return None
    for t in getattr(plan, "tasks", ()) or ():
        if getattr(t, "id", "") == task_id:
            return t
    return None


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
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return None
    if _STATE_KEY_CURRENT_TASK_REVISION not in state:
        return None
    raw = state.get(_STATE_KEY_CURRENT_TASK_REVISION, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _read_plan_revision(session: Session) -> int:
    """Return ``session.plan.revision_index`` as an int, default 0.

    Tolerant of missing / mock plans (test stubs, custom executors).
    """
    plan = getattr(session, "plan", None)
    try:
        return max(0, int(getattr(plan, "revision_index", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _supersession_successor(
    session: Session, task_id: str
) -> tuple[str, SupersessionKind]:
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
                "reporting: pin_revision=%d > current=%d for task_id=%s "
                "(unexpected; trusting pin)",
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


def _refused_response() -> dict[str, Any]:
    """Return the ack-only response for a refused stale-pin transition.

    The LLM still sees ``{"acknowledged": True}`` rather than an error
    payload — surfacing the refusal as a structured error would create a
    prompt-injection surface (the LLM might reason against the rejection
    and bypass the contract). Operators see the refusal via the
    ``task_transition_refused`` sink event.
    """
    return dict(_ACK)


async def _classify_and_route_pin(
    *,
    session: Session,
    steerer: Steerer,
    task_id: str,
    tool_name: str,
    attempted_to: TaskStatus,
) -> tuple[str, dict[str, Any] | None, bool]:
    """Apply the goldfive#266 pin freshness classifier to ``task_id``.

    Returns ``(effective_task_id, refusal_response_or_None, rerouted)``:

    * ``("<task_id>", None, False)`` — proceed on the original task_id
      (fresh pin, no successor to follow).
    * ``("<successor_task_id>", None, True)`` — proceed; the original
      pin pointed at a superseded task and the helper routed onto its
      REPLACE-kind successor. Callers that emit ``TaskTransitioned``
      (goldfive#251 R4) should record ``source="supersedes_reroute"``
      on the resulting transition.
    * ``("", {refusal}, False)`` — refuse; the caller returns the
      refusal response (an ack-only dict) without driving the steerer.
      A ``task_transition_refused`` sink event has already been emitted.

    Uses the supersedes graph to distinguish a stale REPLACE pin
    (route) from a stale CORRECT pin (refuse: the old task's terminal
    state is historical fact, the correction is a separate work unit)
    and from a stale pin with no successor (refuse: ambiguity —
    operator must decide).

    Always falls back gracefully on the existing
    :func:`_reroute_if_superseded` helper when classification yields
    "match" — preserving the post-#258 behaviour where a fresh pin on a
    terminal task whose successor is REPLACE-kind still routes (the
    "agent was retrying its own pin id but the planner already
    replaced it" path).
    """
    pin_rev_opt = _read_pin_revision(session)
    cur_rev = _read_plan_revision(session)
    if pin_rev_opt is None:
        # No stamp present — legacy session, custom adapter, or test
        # stub that pre-dates #266. Preserve the historical
        # ``_reroute_if_superseded`` semantics: fresh pin on a
        # terminal task whose successor is REPLACE-kind routes; no
        # successor or CORRECT-kind successor falls through to the
        # handler's existing terminal-state rejection path.
        resolved = _reroute_if_superseded(session, task_id, tool_name)
        return resolved, None, resolved != task_id
    pin_rev = pin_rev_opt
    freshness, successor_id, _kind = _classify_pin_freshness(
        session,
        task_id,
        pin_revision=pin_rev,
        current_revision=cur_rev,
    )

    if freshness == "match":
        # Fresh pin — fall through to the legacy rerouting helper.
        # This still routes a fresh pin pointing at a terminal task
        # whose successor is REPLACE-kind (an LLM that retries its
        # own pin without realising the plan repointed).
        resolved = _reroute_if_superseded(session, task_id, tool_name)
        return resolved, None, resolved != task_id

    if freshness == "stale_replace":
        log.info(
            "reporting: %s called with stale pin task_id=%s "
            "(pin_rev=%d, current_rev=%d); routing to REPLACE successor=%s",
            tool_name,
            task_id,
            pin_rev,
            cur_rev,
            successor_id,
        )
        return successor_id or task_id, None, bool(successor_id and successor_id != task_id)

    # stale_correct or stale_ambiguous → refuse.
    reason = (
        "stale_pin_correct_supersedes"
        if freshness == "stale_correct"
        else "stale_pin_no_supersedes"
    )
    log.warning(
        "reporting: %s REFUSED on stale pin task_id=%s "
        "(pin_rev=%d, current_rev=%d, reason=%s)",
        tool_name,
        task_id,
        pin_rev,
        cur_rev,
        reason,
    )
    # Look up the pin's current status in the plan for a faithful
    # ``attempted_from`` value on the sink event. Falls back to PENDING
    # if the task isn't in the plan (which would be a separate bug —
    # the classifier has already consulted the supersedes graph and
    # found a successor or absence-of-one).
    pin_task = _find_task_in_session(session, task_id)
    attempted_from = (
        pin_task.status if pin_task is not None else TaskStatus.PENDING
    )
    await _emit_task_transition_refused(
        session=session,
        steerer=steerer,
        task_id=task_id,
        attempted_from=attempted_from,
        attempted_to=attempted_to,
        reason=reason,
        pin_revision=pin_rev,
        current_revision=cur_rev,
    )
    return "", _refused_response(), False


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


async def _await_plan_stable(session: Session, steerer: Steerer) -> None:
    """Block briefly until the steerer's plan mutation region is idle.

    Duck-types ``steerer._wait_plan_stable``: any Steerer implementation
    that exposes an async ``_wait_plan_stable(session)`` method gets a
    consistent plan-state read; implementations that don't (custom
    Steerers, test stubs) silently fall through and observe whatever
    plan state happens to be installed at call time — the pre-a4
    behaviour. Atomicity is best-effort (the helper times out after a
    short interval and proceeds anyway), so this is a soft barrier,
    not a hard mutex.
    """
    waiter = getattr(steerer, "_wait_plan_stable", None)
    if not callable(waiter):
        return
    try:
        await waiter(session)
    except Exception as exc:  # noqa: BLE001 — barrier must never break a report
        log.debug(
            "reporting._await_plan_stable: %s (proceeding with current plan state)",
            exc,
        )


def _idempotent_response(current_status: TaskStatus) -> dict[str, Any]:
    return {
        "acknowledged": True,
        "idempotent": True,
        "current_status": current_status.value,
    }


def _invalid_transition_response(
    *,
    tool_name: str,
    current_status: TaskStatus,
    attempted: TaskStatus,
    task_id: str,
) -> dict[str, Any]:
    return {
        "acknowledged": False,
        "error": "invalid_transition",
        "tool": tool_name,
        "task_id": task_id,
        "current_status": current_status.value,
        "attempted": attempted.value,
        "message": (
            f"Cannot {tool_name!r} task {task_id!r} from {current_status.value} "
            f"to {attempted.value}. The task is already in a terminal or "
            "otherwise-incompatible state; do not retry."
        ),
    }


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
    """Advance ``goldfive.current_task_id`` after a terminal transition.

    Delegates to :func:`goldfive.orchestration_state.rotate_current_task_id`.
    Imported lazily so this module stays ADK-free and dodges the
    import-cycle risk between :mod:`goldfive.reporting` and
    :mod:`goldfive.orchestration_state` (both are imported from
    :mod:`goldfive.steerer`).
    """
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return
    plan = getattr(session, "plan", None)
    agent_name = str(getattr(completed_task, "assignee_agent_id", "") or "")

    # Only rotate if the terminal task was the one we'd been pointing
    # at — otherwise some other caller owns the pin and we'd clobber
    # theirs.
    pinned = state.get(_STATE_KEY_CURRENT_TASK_ID, "")
    if isinstance(pinned, str):
        pinned = pinned.strip()
    else:
        pinned = str(pinned or "").strip()
    this_id = str(getattr(completed_task, "id", "") or "")
    if pinned and pinned != this_id:
        return

    from goldfive import orchestration_state as _ostate

    _ostate.rotate_current_task_id(state, plan, agent_name)


def _missing_task_id_response(tool_name: str) -> dict[str, Any]:
    """Return the canonical ``missing_task_id`` rejection shape.

    Mirrors the shape :mod:`goldfive.adapters._tool_invocation`
    returns so adapters that call the handler directly (legacy paths,
    custom adapters) surface the same structured error the
    ``invoke_tool`` dispatcher would.
    """
    return {
        "acknowledged": False,
        "error": "missing_task_id",
        "tool": tool_name,
        "message": (
            f"Tool {tool_name!r} requires a task_id; call it with the id "
            "of the task you're reporting on, or ensure the adapter "
            "has pinned goldfive.current_task_id on session state."
        ),
    }


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


async def _handle_task_started(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id, source = _resolve_task_id_with_source(args, session)
    detail = _str(args, "detail")
    if not task_id:
        return _missing_task_id_response("report_task_started")
    # goldfive a4: barrier against a concurrent fire-and-forget refine
    # mutating the plan mid-read. See ``_await_plan_stable``.
    await _await_plan_stable(session, steerer)
    # goldfive#266 — classify pin freshness; refuse stale CORRECT /
    # ambiguous, route stale REPLACE, fall through on match.
    task_id, refusal, rerouted = await _classify_and_route_pin(
        session=session,
        steerer=steerer,
        task_id=task_id,
        tool_name="report_task_started",
        attempted_to=TaskStatus.RUNNING,
    )
    if refusal is not None:
        return refusal
    if rerouted:
        source = "supersedes_reroute"
    task = _find_task_in_session(session, task_id)
    if task is not None:
        decision = _classify_transition(tool_name="report_task_started", current_status=task.status)
        if decision == "idempotent":
            return _idempotent_response(task.status)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_started",
                current_status=task.status,
                attempted=TaskStatus.RUNNING,
                task_id=task_id,
            )
    await steerer.mark_task_running(task_id, session=session, detail=detail, source=source)
    # goldfive#251 Stream D: the agent has acknowledged the (possibly
    # corrected) task; clear any queued correction scoped to this
    # ``(agent, task_id)`` pair. The correction block only needs to be
    # injected until the agent is on-task — further turns should see
    # the unadorned instruction. ``report_task_failed`` does NOT clear
    # (failure is not acknowledgment, and a re-invocation still needs
    # the correction).
    _clear_correction_on_started(session, task)
    return dict(_ACK)


def _clear_correction_on_started(session: Session, task: Task | None) -> None:
    """Best-effort GC of the pending correction for this task's assignee.

    The correction is keyed by ``(agent_name, task_id)``; we recover
    ``agent_name`` from ``task.assignee_agent_id`` on the plan. When
    the task is unknown (shouldn't happen by the time we reach this
    handler — ``_handle_task_started`` has already looked it up — but
    defensive for edge cases) the clear is silently skipped.

    Isolated so the call site in ``_handle_task_started`` stays terse
    and so the import of the correction-injection module is paid only
    in handler paths that actually need it.
    """
    if task is None:
        return
    from goldfive._correction_injection import clear_correction

    assignee = str(getattr(task, "assignee_agent_id", "") or "").strip()
    task_id = str(getattr(task, "id", "") or "").strip()
    if not assignee or not task_id:
        return
    # Normalise the same way the write-side does so keys round-trip.
    if "." in assignee:
        assignee = assignee.rsplit(".", 1)[-1]
    try:
        clear_correction(session, agent_name=assignee, task_id=task_id)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "reporting: clear_correction on report_task_started raised: %s",
            exc,
        )


async def _handle_task_progress(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id, _source = _resolve_task_id_with_source(args, session)
    fraction = _float(args, "fraction")
    detail = _str(args, "detail")
    if not task_id:
        return _missing_task_id_response("report_task_progress")
    await _await_plan_stable(session, steerer)
    task_id, refusal, _rerouted = await _classify_and_route_pin(
        session=session,
        steerer=steerer,
        task_id=task_id,
        tool_name="report_task_progress",
        attempted_to=TaskStatus.RUNNING,
    )
    if refusal is not None:
        return refusal
    task = _find_task_in_session(session, task_id)
    if task is not None:
        # report_task_progress is a liveness tick — only valid on RUNNING.
        # PENDING / BLOCKED → invalid ("hasn't started or is waiting");
        # terminal → invalid ("task is done"). RUNNING → always a no-op
        # style idempotent ACK (the steerer call itself doesn't mutate
        # status, but we treat it uniformly so the response shape matches
        # the other handlers).
        if task.status is TaskStatus.RUNNING:
            # Legal tick — proceed to steerer (records progress).
            pass
        else:
            return _invalid_transition_response(
                tool_name="report_task_progress",
                current_status=task.status,
                attempted=TaskStatus.RUNNING,
                task_id=task_id,
            )
    await steerer.mark_task_progress(task_id, session=session, fraction=fraction, detail=detail)
    return dict(_ACK)


async def _handle_task_completed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id, source = _resolve_task_id_with_source(args, session)
    summary = _str(args, "summary")
    artifacts_raw = args.get("artifacts")
    artifacts = (
        {str(k): str(v) for k, v in (artifacts_raw or {}).items()}
        if isinstance(artifacts_raw, dict)
        else {}
    )
    if not task_id:
        return _missing_task_id_response("report_task_completed")
    await _await_plan_stable(session, steerer)
    task_id, refusal, rerouted = await _classify_and_route_pin(
        session=session,
        steerer=steerer,
        task_id=task_id,
        tool_name="report_task_completed",
        attempted_to=TaskStatus.COMPLETED,
    )
    if refusal is not None:
        return refusal
    if rerouted:
        source = "supersedes_reroute"
    task = _find_task_in_session(session, task_id)
    if task is not None:
        decision = _classify_transition(
            tool_name="report_task_completed", current_status=task.status
        )
        if decision == "idempotent":
            return _idempotent_response(task.status)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_completed",
                current_status=task.status,
                attempted=TaskStatus.COMPLETED,
                task_id=task_id,
            )
    await steerer.mark_task_completed(
        task_id, session=session, summary=summary, artifacts=artifacts, source=source
    )
    # Rotate the current-task pin now that this one has landed terminal.
    if task is not None:
        _rotate_after_terminal(session, task)
    return dict(_ACK)


async def _handle_task_failed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id, source = _resolve_task_id_with_source(args, session)
    reason = _str(args, "reason")
    recoverable = _bool(args, "recoverable", default=True)
    if not task_id:
        return _missing_task_id_response("report_task_failed")
    await _await_plan_stable(session, steerer)
    task_id, refusal, rerouted = await _classify_and_route_pin(
        session=session,
        steerer=steerer,
        task_id=task_id,
        tool_name="report_task_failed",
        attempted_to=TaskStatus.FAILED,
    )
    if refusal is not None:
        return refusal
    if rerouted:
        source = "supersedes_reroute"
    task = _find_task_in_session(session, task_id)
    if task is not None:
        decision = _classify_transition(tool_name="report_task_failed", current_status=task.status)
        if decision == "idempotent":
            return _idempotent_response(task.status)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_failed",
                current_status=task.status,
                attempted=TaskStatus.FAILED,
                task_id=task_id,
            )
    await steerer.mark_task_failed(
        task_id,
        session=session,
        reason=reason,
        recoverable=recoverable,
        source=source,
    )
    if task is not None:
        _rotate_after_terminal(session, task)
    return dict(_ACK)


async def _handle_task_blocked(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    task_id, source = _resolve_task_id_with_source(args, session)
    blocker = _str(args, "blocker")
    needed = _str(args, "needed")
    if not task_id:
        return _missing_task_id_response("report_task_blocked")
    await _await_plan_stable(session, steerer)
    task_id, refusal, rerouted = await _classify_and_route_pin(
        session=session,
        steerer=steerer,
        task_id=task_id,
        tool_name="report_task_blocked",
        attempted_to=TaskStatus.BLOCKED,
    )
    if refusal is not None:
        return refusal
    if rerouted:
        source = "supersedes_reroute"
    task = _find_task_in_session(session, task_id)
    if task is not None:
        decision = _classify_transition(tool_name="report_task_blocked", current_status=task.status)
        if decision == "idempotent":
            return _idempotent_response(task.status)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_blocked",
                current_status=task.status,
                attempted=TaskStatus.BLOCKED,
                task_id=task_id,
            )
    await steerer.mark_task_blocked(
        task_id, session=session, blocker=blocker, needed=needed, source=source
    )
    return dict(_ACK)


async def _handle_new_work_discovered(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    parent_task_id = _str(args, "parent_task_id")
    title = _str(args, "title")
    description = _str(args, "description")
    assignee = _str(args, "assignee")
    await steerer.report_new_work_discovered(
        session=session,
        parent_task_id=parent_task_id,
        title=title,
        description=description,
        assignee=assignee,
    )
    return dict(_ACK)


async def _handle_plan_divergence(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    note = _str(args, "note")
    suggested_action = _str(args, "suggested_action")
    await steerer.report_plan_divergence(
        session=session,
        note=note,
        suggested_action=suggested_action,
    )
    return dict(_ACK)


async def _handle_awaiting_approval(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    """Block the current task until APPROVE / REJECT arrives on the control channel.

    Transitions the task to ``BLOCKED`` (so sinks see a concrete status
    for the "awaiting approval" card), emits ``ApprovalRequested``, and
    awaits the per-task ``asyncio.Event`` the control dispatcher sets
    when the matching ``ControlMessage(APPROVE|REJECT)`` lands.

    Returns ``{"decision": "approve" | "reject", "detail": ...}`` so the
    agent can decide whether to proceed or transition the task to
    ``FAILED`` itself. A ``timeout_ms > 0`` that elapses before a
    decision lands returns ``{"decision": "timeout", "detail": ...}``
    and leaves the task blocked (the caller may re-prompt or fail).
    """
    task_id, source = _resolve_task_id_with_source(args, session)
    prompt = _str(args, "prompt")
    timeout_ms = _int(args, "timeout_ms", 0)
    if not task_id:
        return _missing_task_id_response("report_awaiting_approval")
    # goldfive#201: reject on terminal task up front with the
    # canonical invalid_transition shape. An already-BLOCKED /
    # already-RUNNING task falls through to the waiter-reuse path
    # below (that's the semantic idempotency for approvals — they
    # block on the existing Event instead of returning a no-op ack).
    task = _find_task_in_session(session, task_id)
    if task is not None and task.status in _TERMINAL_STATUSES:
        return _invalid_transition_response(
            tool_name="report_awaiting_approval",
            current_status=task.status,
            attempted=TaskStatus.BLOCKED,
            task_id=task_id,
        )

    # Idempotency: reuse an existing waiter if one is already pending.
    waiter = session.pending_approvals.get(task_id)
    if waiter is None:
        waiter = asyncio.Event()
        session.pending_approvals[task_id] = waiter
    meta = session.pending_approvals_meta.setdefault(
        task_id,
        {"kind": "task", "prompt": prompt, "task_id": task_id},
    )
    # Refresh prompt in case it changed — the dispatcher only reads
    # ``decision`` and ``detail`` so this is safe.
    meta["prompt"] = prompt

    await steerer.mark_task_blocked(
        task_id,
        session=session,
        blocker="awaiting_approval",
        needed=prompt,
        source=source,
    )
    await _emit_approval_requested(
        session=session,
        steerer=steerer,
        target_id=task_id,
        kind="task",
        prompt=prompt,
        task_id=task_id,
        metadata={},
    )

    try:
        if timeout_ms > 0:
            await asyncio.wait_for(waiter.wait(), timeout=timeout_ms / 1000.0)
        else:
            await waiter.wait()
    except TimeoutError:
        return {
            "acknowledged": True,
            "decision": "timeout",
            "detail": f"no decision after {timeout_ms}ms",
        }

    decision = str(meta.get("decision", "")) or "approve"
    detail = str(meta.get("detail", ""))
    return {"acknowledged": True, "decision": decision, "detail": detail}


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


# ---------------------------------------------------------------------------
# Parameter schemas (JSON Schema draft-07 style)
# ---------------------------------------------------------------------------


def _object_schema(*, required: list[str], properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# NOTE: ``task_id`` is intentionally omitted from every schema's
# ``required`` list (goldfive#191). The adapter stamps
# ``goldfive.current_task_id`` onto session state at delegation time
# so the handler can default from state when the model doesn't supply
# the arg. Handlers still reject with the canonical
# ``missing_task_id`` shape when neither source resolves a value —
# so strictness is enforced at the handler layer, not the schema.

_SCHEMA_TASK_STARTED = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_PROGRESS = _object_schema(
    required=[],
    properties={
        "task_id": {"type": "string"},
        "fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "detail": {"type": "string"},
    },
)

_SCHEMA_TASK_COMPLETED = _object_schema(
    required=["summary"],
    properties={
        "task_id": {"type": "string"},
        "summary": {"type": "string"},
        "artifacts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
)

_SCHEMA_TASK_FAILED = _object_schema(
    required=["reason"],
    properties={
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
        "recoverable": {"type": "boolean"},
    },
)

_SCHEMA_TASK_BLOCKED = _object_schema(
    required=["blocker"],
    properties={
        "task_id": {"type": "string"},
        "blocker": {"type": "string"},
        "needed": {"type": "string"},
    },
)

_SCHEMA_NEW_WORK_DISCOVERED = _object_schema(
    required=["parent_task_id", "title", "description"],
    properties={
        "parent_task_id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "assignee": {"type": "string"},
    },
)

_SCHEMA_PLAN_DIVERGENCE = _object_schema(
    required=["note"],
    properties={
        "note": {"type": "string"},
        "suggested_action": {"type": "string"},
    },
)

_SCHEMA_AWAITING_APPROVAL = _object_schema(
    required=["prompt"],
    properties={
        "task_id": {"type": "string"},
        "prompt": {"type": "string"},
        "timeout_ms": {"type": "integer", "minimum": 0},
    },
)


# ---------------------------------------------------------------------------
# Built-in tool specs
# ---------------------------------------------------------------------------


BUILTIN_REPORTING_TOOLS: list[ReportingToolSpec] = [
    ReportingToolSpec(
        name="report_task_started",
        description=(
            "Report that you are beginning work on a planned task. Call this "
            "BEFORE doing the actual work so the framework knows which task "
            "is currently in progress."
        ),
        parameters=_SCHEMA_TASK_STARTED,
        handler=_handle_task_started,
    ),
    ReportingToolSpec(
        name="report_task_progress",
        description=(
            "Report mid-task progress. Optional — only call if the task has "
            "meaningful sub-steps. 'fraction' is a 0.0-1.0 hint of how far "
            "through the task you are."
        ),
        parameters=_SCHEMA_TASK_PROGRESS,
        handler=_handle_task_progress,
    ),
    ReportingToolSpec(
        name="report_task_completed",
        description=(
            "Report that you have completed a planned task successfully. "
            "Call this AFTER producing the final output. 'summary' describes "
            "the result in one or two sentences."
        ),
        parameters=_SCHEMA_TASK_COMPLETED,
        handler=_handle_task_completed,
    ),
    ReportingToolSpec(
        name="report_task_failed",
        description=(
            "Report that you were unable to complete a planned task. "
            "'recoverable=True' lets the plan route around this failure; "
            "'recoverable=False' means the whole workflow should probably stop."
        ),
        parameters=_SCHEMA_TASK_FAILED,
        handler=_handle_task_failed,
    ),
    ReportingToolSpec(
        name="report_task_blocked",
        description=(
            "Report that you cannot currently proceed with a task. Use this "
            "when an external blocker prevents progress. 'blocker' describes "
            "what is in the way; 'needed' describes what would unblock you."
        ),
        parameters=_SCHEMA_TASK_BLOCKED,
        handler=_handle_task_blocked,
    ),
    ReportingToolSpec(
        name="report_new_work_discovered",
        description=(
            "Report that you've discovered additional work the plan doesn't "
            "know about. The framework will ask the planner to add this task "
            "as a child of 'parent_task_id'."
        ),
        parameters=_SCHEMA_NEW_WORK_DISCOVERED,
        handler=_handle_new_work_discovered,
    ),
    ReportingToolSpec(
        name="report_plan_divergence",
        description=(
            "Report that the current plan no longer matches what needs to "
            "happen. The framework will trigger an explicit replan."
        ),
        parameters=_SCHEMA_PLAN_DIVERGENCE,
        handler=_handle_plan_divergence,
    ),
    ReportingToolSpec(
        name="report_awaiting_approval",
        description=(
            "Block the current task until a human approves or rejects via "
            "the control channel. Use this when the task has a side effect "
            "that needs sign-off (spending money, writing to a shared "
            "system, sending a message). The call blocks until the UI "
            "dispatches an APPROVE or REJECT and returns "
            "{'decision': 'approve' | 'reject' | 'timeout', 'detail': ...}. "
            "The agent decides what to do with the decision: on approve, "
            "proceed; on reject, typically report_task_failed with a "
            "user-rejection reason."
        ),
        parameters=_SCHEMA_AWAITING_APPROVAL,
        handler=_handle_awaiting_approval,
    ),
]


__all__ = [
    "ReportingHandler",
    "ReportingToolSpec",
    "REPORTING_TOOL_NAMES",
    "BUILTIN_REPORTING_TOOLS",
]
