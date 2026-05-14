"""Async handlers and tool-spec catalog for the canonical reporting tools.

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

from goldfive import _state_audit
from goldfive.reporting._internal import (
    _ACK,
    _TERMINAL_STATUSES,
    _await_plan_stable,
    _bool,
    _classify_pin_freshness,
    _classify_transition,
    _emit_approval_requested,
    _emit_task_declaration_received,
    _emit_task_transition_refused,
    _find_task_in_session,
    _float,
    _int,
    _read_pin_revision,
    _read_plan_revision,
    _reroute_if_superseded,
    _resolve_task_id_with_source,
    _rotate_after_terminal,
    _str,
)
from goldfive.reporting.rendering import (
    _directive_ack,
    _idempotent_response,
    _invalid_transition_response,
    _missing_required_field_response,
    _missing_task_id_response,
    _refused_response,
)
from goldfive.reporting.schemas import (
    _SCHEMA_AWAITING_APPROVAL,
    _SCHEMA_DECLARE_TASK_NOT_NEEDED,
    _SCHEMA_DECLARE_TASK_SKIPPED,
    _SCHEMA_NEW_WORK_DISCOVERED,
    _SCHEMA_PLAN_DIVERGENCE,
    _SCHEMA_TASK_BLOCKED,
    _SCHEMA_TASK_COMPLETED,
    _SCHEMA_TASK_FAILED,
    _SCHEMA_TASK_PROGRESS,
    _SCHEMA_TASK_STARTED,
)
from goldfive.types import TaskStatus

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


# The ten canonical reporting tool names. These are a stable contract: the
# adapter must surface tools with exactly these names so that the Steerer can
# interpret them uniformly across frameworks. Do not rename.
#
# goldfive#271 Phase 3: ``declare_task_skipped`` and ``declare_task_not_needed``
# are observability-only (no plan mutation, no steerer drive); they emit a
# ``TaskDeclarationReceived`` event so the operator can see "agent
# volunteered skip" before the imperative ``report_task_*`` surface
# either confirms or contradicts it.
REPORTING_TOOL_NAMES: tuple[str, ...] = (
    "report_task_started",
    "report_task_progress",
    "report_task_completed",
    "report_task_failed",
    "report_task_blocked",
    "report_new_work_discovered",
    "report_plan_divergence",
    "report_awaiting_approval",
    "declare_task_skipped",
    "declare_task_not_needed",
)


# State key for the per-Session declarations log. Lives under the
# ``goldfive.*`` namespace so :func:`goldfive.state_store.write`
# accepts it. Value shape: ``dict[(kind, task_id), {kind, task_id,
# reason, recorded_at_seq}]`` keyed by ``(declaration_kind, task_id)``
# so the second declaration of the same kind on the same task is a
# no-op.
DECLARATIONS_KEY = "goldfive.task_declarations"


# Vocabulary of declaration kinds. Mirrors the proto envelope's `kind`
# field once it gets promoted (Phase 3.5/4 cleanup); for now lives as
# a string vocabulary on the dict envelope.
DECLARATION_KIND_SKIPPED: str = "skipped"
DECLARATION_KIND_NOT_NEEDED: str = "not_needed"
DECLARATION_KINDS: tuple[str, ...] = (
    DECLARATION_KIND_SKIPPED,
    DECLARATION_KIND_NOT_NEEDED,
)


# ---------------------------------------------------------------------------
# Stale-pin classification / routing bridge
# ---------------------------------------------------------------------------


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
        "reporting: %s REFUSED on stale pin task_id=%s (pin_rev=%d, current_rev=%d, reason=%s)",
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
    attempted_from = pin_task.status if pin_task is not None else TaskStatus.PENDING
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


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


def _validate_required(
    args: dict[str, Any],
    schema: dict[str, Any],
    tool_name: str,
) -> dict[str, Any] | None:
    """Enforce the schema's ``required[]`` list at handler dispatch.

    The v15-cascade root cause: handlers ``_str``-coerced missing /
    empty fields to ``""`` and forwarded the empty payload onto the
    steerer, where it became drift detail like ``"new work under : :
    "`` — semantically empty signals that the planner correctly
    declined to act on, leaving the agent in a no-op revision tool
    loop.

    The fix: every handler validates each schema-``required`` field at
    entry. A missing field, ``None``, or whitespace-only string is a
    contract violation and the handler returns a structured error
    instead of driving the steerer. Numbers and booleans are accepted
    as-is when present (a literal ``0`` / ``False`` is a valid value;
    only the absence of the key is a violation).

    NOTE: schema-``required`` does NOT include ``task_id`` even on
    handlers that need one — see goldfive#191. ``task_id`` rejection
    flows through :func:`_missing_task_id_response` after the adapter
    pin fallback. This helper enforces the *content* fields the LLM
    must supply.

    Returns ``None`` when validation passes (caller proceeds), or an
    error response dict the handler should return verbatim.
    """
    required = schema.get("required") or []
    for field in required:
        if field not in args:
            return _missing_required_field_response(
                tool_name=tool_name,
                field=field,
                reason="missing",
                schema=schema,
            )
        val = args[field]
        if val is None:
            return _missing_required_field_response(
                tool_name=tool_name,
                field=field,
                reason="null",
                schema=schema,
            )
        if isinstance(val, str) and not val.strip():
            return _missing_required_field_response(
                tool_name=tool_name,
                field=field,
                reason="empty",
                schema=schema,
            )
    return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


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
            return _idempotent_response(task.status, session=session, task_id=task_id)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_started",
                current_status=task.status,
                attempted=TaskStatus.RUNNING,
                task_id=task_id,
            )
    # Phase 3.5 (CANCELLATION-CONTRACT.md §C5): wrap the
    # ``mark_task_running`` await in ``try/finally`` so the correction
    # GC fires even when ``CancelledError`` (a ``BaseException`` since
    # Py 3.8) propagates out mid-await. Without the ``finally`` a
    # cancelled report-task-started leaves the pending correction
    # entry on session state, and the next turn re-injects the
    # correction block on a task the agent has already acknowledged.
    #
    # goldfive#251 Stream D: the agent has acknowledged the (possibly
    # corrected) task; clear any queued correction scoped to this
    # ``(agent, task_id)`` pair. The correction block only needs to be
    # injected until the agent is on-task — further turns should see
    # the unadorned instruction. ``report_task_failed`` does NOT clear
    # (failure is not acknowledgment, and a re-invocation still needs
    # the correction).
    with _state_audit.cancellation_stash_audited(
        "reporting._handle_task_started.mark_task_running"
    ):
        try:
            await steerer.tasks.mark_task_running(
                task_id, session=session, detail=detail, source=source
            )
        finally:
            _clear_correction_on_started(session, task)
            # Phase 3.5 tripwire compliance marker (§1.1 form): the
            # GC ran inside ``finally`` regardless of how the await
            # exited. The boundary catch site can now confirm we
            # didn't bypass the stash.
            _state_audit.mark_stash_completed()
    return _directive_ack(session=session, task_id=task_id, new_status=TaskStatus.RUNNING)


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
    await steerer.tasks.mark_task_progress(
        task_id, session=session, fraction=fraction, detail=detail
    )
    return _directive_ack(session=session, task_id=task_id, new_status=TaskStatus.RUNNING)


async def _handle_task_completed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    err = _validate_required(args, _SCHEMA_TASK_COMPLETED, "report_task_completed")
    if err is not None:
        return err
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
            return _idempotent_response(task.status, session=session, task_id=task_id)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_completed",
                current_status=task.status,
                attempted=TaskStatus.COMPLETED,
                task_id=task_id,
            )
    await steerer.tasks.mark_task_completed(
        task_id, session=session, summary=summary, artifacts=artifacts, source=source
    )
    # Rotate the current-task pin now that this one has landed terminal.
    if task is not None:
        _rotate_after_terminal(session, task)
    return _directive_ack(session=session, task_id=task_id, new_status=TaskStatus.COMPLETED)


async def _handle_task_failed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    err = _validate_required(args, _SCHEMA_TASK_FAILED, "report_task_failed")
    if err is not None:
        return err
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
            return _idempotent_response(task.status, session=session, task_id=task_id)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_failed",
                current_status=task.status,
                attempted=TaskStatus.FAILED,
                task_id=task_id,
            )
    await steerer.tasks.mark_task_failed(
        task_id,
        session=session,
        reason=reason,
        recoverable=recoverable,
        source=source,
    )
    if task is not None:
        _rotate_after_terminal(session, task)
    return _directive_ack(session=session, task_id=task_id, new_status=TaskStatus.FAILED)


async def _handle_task_blocked(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    err = _validate_required(args, _SCHEMA_TASK_BLOCKED, "report_task_blocked")
    if err is not None:
        return err
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
            return _idempotent_response(task.status, session=session, task_id=task_id)
        if decision == "invalid":
            return _invalid_transition_response(
                tool_name="report_task_blocked",
                current_status=task.status,
                attempted=TaskStatus.BLOCKED,
                task_id=task_id,
            )
    await steerer.tasks.mark_task_blocked(
        task_id, session=session, blocker=blocker, needed=needed, source=source
    )
    return _directive_ack(session=session, task_id=task_id, new_status=TaskStatus.BLOCKED)


async def _handle_new_work_discovered(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    err = _validate_required(args, _SCHEMA_NEW_WORK_DISCOVERED, "report_new_work_discovered")
    if err is not None:
        return err
    parent_task_id = _str(args, "parent_task_id")
    title = _str(args, "title")
    description = _str(args, "description")
    assignee = _str(args, "assignee")
    await steerer.drift.report_new_work_discovered(
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
    err = _validate_required(args, _SCHEMA_PLAN_DIVERGENCE, "report_plan_divergence")
    if err is not None:
        return err
    note = _str(args, "note")
    suggested_action = _str(args, "suggested_action")
    await steerer.drift.report_plan_divergence(
        session=session,
        note=note,
        suggested_action=suggested_action,
    )
    return dict(_ACK)


def _record_declaration(session: Session, kind: str, task_id: str, reason: str) -> bool:
    """Record a declaration on session.state; return True if newly recorded.

    Idempotent: a second declaration of the same ``(kind, task_id)``
    pair is a no-op (returns False, no event re-emitted). The recorded
    body intentionally keeps the FIRST reason — late declarations don't
    rewrite history.
    """
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        # Defensive: callers without a real Session.state still drive
        # the emit path (no-op idempotency).
        return True
    declarations = state.get(DECLARATIONS_KEY)
    if not isinstance(declarations, dict):
        declarations = {}
        state[DECLARATIONS_KEY] = declarations
    key = f"{kind}:{task_id}"
    if key in declarations:
        return False
    declarations[key] = {
        "kind": kind,
        "task_id": task_id,
        "reason": reason,
        # Best-effort sequence stamp for ordering when read back later.
        "recorded_at_seq": int(getattr(session, "_next_sequence", 0)),
    }
    return True


async def _handle_declaration(
    args: dict[str, Any],
    session: Session,
    steerer: Steerer,
    *,
    kind: str,
    tool_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Shared body for ``declare_task_skipped`` / ``declare_task_not_needed``.

    Both handlers are observability-only:

    * Resolve ``task_id`` (explicit > pin default — same precedence as
      the ``report_task_*`` family).
    * Idempotency: skip the emit when the declaration is a duplicate
      of the same ``(kind, task_id)`` pair.
    * Emit ``TaskDeclarationReceived`` on the sink bus.
    * Return ``{"acknowledged": True}`` — never fail the agent's tool
      call on a declaration.

    DOES NOT mutate plan state. The steerer's ``_apply_revision``
    machinery remains the only path that can transition a task; this
    declaration just queues an advisory signal that the next refine
    consumes (Phase 4 work).
    """
    err = _validate_required(args, schema, tool_name)
    if err is not None:
        return err
    task_id, _source = _resolve_task_id_with_source(args, session)
    reason = _str(args, "reason")
    if not task_id:
        return _missing_task_id_response(tool_name)
    is_new = _record_declaration(session, kind, task_id, reason)
    if is_new:
        await _emit_task_declaration_received(
            session=session,
            steerer=steerer,
            kind=kind,
            task_id=task_id,
            reason=reason,
        )
    # Return shape mirrors the ``report_task_*`` family: ``acknowledged``
    # tells the LLM "we got it", and ``idempotent`` flags the duplicate
    # case so observability can distinguish first-time vs. repeat.
    if is_new:
        return dict(_ACK)
    return {"acknowledged": True, "idempotent": True}


async def _handle_declare_task_skipped(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    return await _handle_declaration(
        args,
        session,
        steerer,
        kind=DECLARATION_KIND_SKIPPED,
        tool_name="declare_task_skipped",
        schema=_SCHEMA_DECLARE_TASK_SKIPPED,
    )


async def _handle_declare_task_not_needed(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    return await _handle_declaration(
        args,
        session,
        steerer,
        kind=DECLARATION_KIND_NOT_NEEDED,
        tool_name="declare_task_not_needed",
        schema=_SCHEMA_DECLARE_TASK_NOT_NEEDED,
    )


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
    err = _validate_required(args, _SCHEMA_AWAITING_APPROVAL, "report_awaiting_approval")
    if err is not None:
        return err
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

    await steerer.tasks.mark_task_blocked(
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


# ---------------------------------------------------------------------------
# Built-in tool specs
# ---------------------------------------------------------------------------


# goldfive#196: drift-related self-reporting tools. These overlap with
# observation-driven detectors (``goldfive.reconciler.PlanReconciler``,
# the trajectory-level ``classify_goal_drift`` judge, and the steerer's
# refine machinery), so registering them on every sub-agent is pure
# downside: they inflate prompt size by ~200-400 tokens each AND expand
# the agent's hallucination surface (the model can confabulate a drift
# call when it's confused). The Runner gates them behind an opt-in flag
# (``Runner(drift_self_reporting=...)``) so the lifecycle subset stays
# default-on while the drift opinions are off by default. ``True``
# preserves the legacy behaviour of registering every canonical tool.
#
# Note: ``report_new_work_discovered`` is intentionally NOT in this set
# — there is no observation analog for an agent surfacing genuinely new
# work, so it stays default-on. See goldfive#196.
DRIFT_SELF_REPORTING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "report_plan_divergence",
        "declare_task_skipped",
        "declare_task_not_needed",
    }
)


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
    # goldfive#271 Phase 3: structural declarations. Observability-only
    # — they emit a TaskDeclarationReceived sink event so operators can
    # see the agent's stated intent without the framework auto-mutating
    # the plan in response. The imperative report_task_* surface
    # remains the only path to actually transition a task; declarations
    # are advisory signals the next refine consumes.
    ReportingToolSpec(
        name="declare_task_skipped",
        description=(
            "Declare that you are intentionally skipping a task — you read "
            "the plan, you saw the task, and you decided not to do it (e.g. "
            "duplicate work already done by another agent, or work the "
            "user clarified is no longer needed). 'reason' is a one-line "
            "explanation. This is an OBSERVABILITY signal — the framework "
            "records it but does NOT remove the task from the plan. If you "
            "want to actually fail or cancel the task, use report_task_failed. "
            "Idempotent: a second declaration of the same kind on the same "
            "task is a no-op."
        ),
        parameters=_SCHEMA_DECLARE_TASK_SKIPPED,
        handler=_handle_declare_task_skipped,
    ),
    ReportingToolSpec(
        name="declare_task_not_needed",
        description=(
            "Declare that a planned task is no longer needed — your work "
            "made the task redundant (e.g. an upstream change satisfies it, "
            "or the goal evolved past it). 'reason' is a one-line "
            "explanation. This is an OBSERVABILITY signal — the framework "
            "records it but does NOT remove the task from the plan. The "
            "next refine considers your declaration when deciding whether "
            "to mark the task NOT_NEEDED. Idempotent: a second declaration "
            "of the same kind on the same task is a no-op."
        ),
        parameters=_SCHEMA_DECLARE_TASK_NOT_NEEDED,
        handler=_handle_declare_task_not_needed,
    ),
]


# goldfive#196: lifecycle / observability subset — every BUILTIN tool
# whose name is NOT in :data:`DRIFT_SELF_REPORTING_TOOL_NAMES`. This is
# the default Runner registration set; the drift tools are gated behind
# ``Runner(drift_self_reporting=...)``. Keep this derived (rather than
# enumerated by hand) so adding a new lifecycle tool to
# :data:`BUILTIN_REPORTING_TOOLS` flows through automatically.
LIFECYCLE_REPORTING_TOOLS: list[ReportingToolSpec] = [
    spec
    for spec in BUILTIN_REPORTING_TOOLS
    if spec.name not in DRIFT_SELF_REPORTING_TOOL_NAMES
]


# goldfive#196: drift-only subset — the BUILTIN tools whose names ARE
# in :data:`DRIFT_SELF_REPORTING_TOOL_NAMES`. Convenience for callers
# that want to register the drift tools explicitly via
# ``Runner(drift_self_reporting=True)`` or
# ``Runner(drift_self_reporting=[<name>, ...])``.
DRIFT_SELF_REPORTING_TOOLS: list[ReportingToolSpec] = [
    spec
    for spec in BUILTIN_REPORTING_TOOLS
    if spec.name in DRIFT_SELF_REPORTING_TOOL_NAMES
]


def select_reporting_tools(
    drift_self_reporting: bool | list[str] | tuple[str, ...] | frozenset[str] | set[str],
) -> list[ReportingToolSpec]:
    """Return the reporting tool specs to register based on the opt-in flag.

    ``drift_self_reporting`` semantics (matches ``Runner.__init__``):

    * ``False`` (default for the Runner) — return the lifecycle subset
      only. Drift tools are NOT registered; the agent cannot
      self-report ``report_plan_divergence`` /
      ``declare_task_skipped`` / ``declare_task_not_needed``. The
      framework's observation paths (``classify_goal_drift``,
      :class:`~goldfive.reconciler.PlanReconciler`, the steerer's
      refine machinery) remain the canonical detectors.
    * ``True`` — return the full ``BUILTIN_REPORTING_TOOLS`` list
      (legacy behaviour). The agent can self-report every drift kind.
    * iterable of tool names — return the lifecycle subset PLUS the
      named drift tools. Names not in
      :data:`DRIFT_SELF_REPORTING_TOOL_NAMES` are silently ignored
      (so a typo doesn't accidentally enable a non-drift tool, and
      there is nothing meaningful to do with a name that isn't a
      drift tool — lifecycle tools are always on). An empty iterable
      collapses to the ``False`` case.

    Used by :class:`~goldfive.runner.Runner.run` step 5 to decide
    which tool specs to hand to ``agent.register_reporting_tools``.
    Exposed here (rather than as a private helper on Runner) so
    custom Runner-likes / ad-hoc adapters can derive the same subset
    from the same flag.
    """
    if drift_self_reporting is True:
        return list(BUILTIN_REPORTING_TOOLS)
    if drift_self_reporting is False:
        return list(LIFECYCLE_REPORTING_TOOLS)
    # Iterable of names — accept list / tuple / set / frozenset.
    requested = {str(n).strip() for n in drift_self_reporting if str(n).strip()}
    if not requested:
        return list(LIFECYCLE_REPORTING_TOOLS)
    selected: list[ReportingToolSpec] = list(LIFECYCLE_REPORTING_TOOLS)
    for spec in DRIFT_SELF_REPORTING_TOOLS:
        if spec.name in requested:
            selected.append(spec)
    return selected


__all__ = [
    "BUILTIN_REPORTING_TOOLS",
    "DECLARATION_KINDS",
    "DECLARATION_KIND_NOT_NEEDED",
    "DECLARATION_KIND_SKIPPED",
    "DECLARATIONS_KEY",
    "DRIFT_SELF_REPORTING_TOOLS",
    "DRIFT_SELF_REPORTING_TOOL_NAMES",
    "LIFECYCLE_REPORTING_TOOLS",
    "REPORTING_TOOL_NAMES",
    "ReportingHandler",
    "ReportingToolSpec",
    "select_reporting_tools",
]
