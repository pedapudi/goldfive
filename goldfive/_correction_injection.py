"""Correction-injection write-side for CORRECT-kind supersedes (goldfive#251 Stream D).

Goldfive#251 threads plan-causal context into wrapped-agent prompts
through three streams:

* Stream A (:mod:`goldfive.types`) — :class:`SupersessionKind` enum +
  the refiner-side topology for CORRECT-kind supersedes (old task stays
  COMPLETED; new task attached as a correction child in the DAG).
* Stream B (:mod:`goldfive.adapters.adk_llm_instrumentation`) — dynamic
  instruction resolver that appends a ``(agent_name, task_id)``-keyed
  pending correction onto the agent's system prompt every turn.
* Stream C (:mod:`goldfive.adapters._adk_plugin` +
  :mod:`goldfive.adapters._adk_state_protocol`) — cooperative cancel of
  the offending in-flight invocation when the drift is CRITICAL.

This module owns the **write-side** of Stream B's read contract. When
the refine in
:meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised` installs a
new task carrying ``supersedes_kind == SupersessionKind.CORRECT``, the
helper :func:`queue_corrections_for_revision` here composes one
structured correction dict per CORRECT-kind new task and stamps it
under :data:`goldfive.adapters._adk_state_protocol.KEY_PENDING_CORRECTIONS`
on the goldfive orchestration :class:`~goldfive.types.Session.state`.

Data flow
---------

1. Refine lands a CORRECT-kind supersedes link on a new task.
2. :func:`queue_corrections_for_revision` writes the correction dict
   into goldfive orchestration state, keyed
   ``goldfive.pending_corrections.<agent_name>.<task_id>``.
3. The dynamic instruction resolver (Stream B) reads its
   ``(agent_name, current_task_id)``-keyed entry directly off goldfive
   ``Session.state`` via the
   :class:`~goldfive.state_store.StateStore`'s
   ``get_correction`` accessor (Phase 2.0 of goldfive#271 — the bridge
   from goldfive ``Session.state`` onto ADK ``session.state`` is gone)
   and — for dict values — formats it via
   :func:`format_correction_block` before appending to the composed
   system prompt.
4. Agent sees the correction block on its next turn and proceeds on
   the corrected task. Once the agent calls ``report_task_started``
   on the correction task id, :func:`clear_correction` GC's the entry
   (agent has acknowledged the new context).

Correction-text design
----------------------

The correction body is **directive, not diagnostic**. We tell the LLM
what to do on the new task — not what went wrong with the old task —
to avoid the pattern-matching failure modes documented on goldfive#250
/ #252 / #253 / #259: when the LLM sees problem-naming language
("failed", "broken", "incorrect") it often invents workarounds (meta-
commentary about the failure, retries of the wrong thing, apologies)
instead of proceeding with the corrected work.

The diagnostic data (drift kind, drift detail, superseded task title
+ id, revision number, issued_at_ms) is still present in the
correction dict for programmatic consumers (sinks, observability) —
it's just not rendered into the prompt text.

Agent-agnostic
--------------

Every wrapped LlmAgent is a potential correction recipient. The
``agent_name`` baked into each correction key is whatever assignee
the refine named on the new task — there is no assumption that the
tree has a "coordinator" agent, explicit or implicit. Correction
routing follows the plan's own assignment, nothing more.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, MutableMapping
from typing import Any

from goldfive.adapters import _adk_state_protocol as _sp
from goldfive.types import (
    DriftEvent,
    Plan,
    SupersessionKind,
    Task,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def pending_correction_key(agent_name: str, task_id: str) -> str:
    """Return the full state key a correction is stamped under.

    Same formula as
    :func:`goldfive.adapters.adk_llm_instrumentation.pending_correction_key`
    — re-exported here so the write-side doesn't pull in the ADK adapter
    module (which has its own dependencies). The string contract is
    shared; either spelling produces the same key.
    """
    return f"{_sp.KEY_PENDING_CORRECTIONS}.{agent_name}.{task_id}"


def is_pending_correction_key(key: str) -> bool:
    """Return True when ``key`` belongs to the pending-corrections family."""
    prefix = _sp.KEY_PENDING_CORRECTIONS + "."
    return isinstance(key, str) and key.startswith(prefix)


# ---------------------------------------------------------------------------
# Write-side
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _bare_agent_name(value: Any) -> str:
    """Normalise a raw agent id to its bare form.

    Accepts strings like ``"agent.sub"`` or namespaced variants and
    returns the last dotted segment. Tolerates anything mapping-shaped
    or with a ``.name`` attribute so the caller doesn't have to
    pre-strip. Empty / malformed input yields ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
    else:
        s = str(getattr(value, "name", value) or "").strip()
    if not s:
        return ""
    # Many call sites already pass a bare name; this is defensive for
    # call sites that may pass a fully-qualified id.
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def build_correction_payload(
    *,
    new_task: Task,
    old_task: Task,
    drift: DriftEvent,
    revision_number: int,
    issued_at_ms: int | None = None,
) -> dict[str, Any]:
    """Compose one correction dict for a CORRECT-kind supersedes link.

    Does NOT write to state; pure value constructor so tests can pin
    the shape without reaching into a Session.

    Fields match the read-side contract consumed by
    :func:`format_correction_block` — changing the shape requires a
    paired change there.
    """
    drift_kind_value = ""
    drift_kind = getattr(drift, "kind", None)
    if drift_kind is not None:
        # DriftKind is a StrEnum; ``.name`` gives the python-identifier
        # spelling (e.g. ``OFF_TOPIC``) and ``.value`` the wire string
        # (``"off_topic"``). We emit the lower-case wire form so sinks
        # and prompt templates see a consistent spelling.
        drift_kind_value = str(getattr(drift_kind, "value", drift_kind) or "").lower()

    drift_reason = str(getattr(drift, "detail", "") or "")

    ts = issued_at_ms if issued_at_ms is not None else _now_ms()

    return {
        "agent_name": _bare_agent_name(getattr(new_task, "assignee_agent_id", "")),
        "task_id": str(getattr(new_task, "id", "") or ""),
        "superseded_task_id": str(getattr(new_task, "supersedes", "") or ""),
        "superseded_task_title": str(getattr(old_task, "title", "") or ""),
        "drift_kind": drift_kind_value,
        "drift_reason": drift_reason,
        "revision_number": int(revision_number),
        "issued_at_ms": int(ts),
    }


def _session_state(session: Any) -> MutableMapping[str, Any] | None:
    """Return the goldfive orchestration state dict, or None if absent."""
    state = getattr(session, "state", None)
    if isinstance(state, MutableMapping):
        return state
    return None


def write_correction(
    session: Any,
    correction: Mapping[str, Any],
) -> str | None:
    """Stamp a correction dict on the orchestration session state.

    Returns the state key used, or ``None`` when the correction was
    rejected (missing agent_name / task_id / session state dict).
    """
    state = _session_state(session)
    if state is None:
        return None
    agent_name = str(correction.get("agent_name", "") or "")
    task_id = str(correction.get("task_id", "") or "")
    if not agent_name or not task_id:
        return None
    key = pending_correction_key(agent_name, task_id)
    # Store a plain dict so downstream serialisers (sinks, persistence)
    # can round-trip the value without needing to know about this
    # module's dataclass / helper contract. ``dict(correction)`` copies
    # defensively so later mutation of the source mapping doesn't
    # silently leak into state.
    state[key] = dict(correction)
    return key


def queue_corrections_for_revision(
    *,
    session: Any,
    revised: Plan,
    prev_plan: Plan | None,
    drift: DriftEvent,
    corrections_via_notes: bool = False,
) -> list[str]:
    """Scan ``revised`` for CORRECT-kind supersedes and stamp corrections.

    Called from
    :meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised` right
    after :meth:`_integrate_correction_supersedes` has rewired the DAG.

    For every new task carrying ``supersedes_kind == CORRECT`` we:

    * Look up the superseded task on the revised plan (it stays in the
      plan as a historical COMPLETED node — the CORRECT topology is
      exactly that, old kept, new inserted as a child). Fall back to
      ``prev_plan`` when the revised plan doesn't expose it (defensive;
      in practice the CORRECT path keeps the old task).
    * Build a correction dict (title of old, drift kind + reason from
      the triggering drift, revision number).
    * Write it under ``goldfive.pending_corrections.<agent>.<task_id>``.

    Returns the list of state keys written so callers (tests, sinks)
    can assert on multi-correction revisions.

    ``corrections_via_notes`` (AGENCY-PRESERVATION.md task #11) — when
    ``True`` (the ``request_context`` regime with the Site-4 pin retired)
    each correction is enqueued onto the StateStore-backed
    :class:`~goldfive.observer_note_queue.ObserverNoteQueue` (carrying the
    assignee as ``agent_id`` so the agent-scoped delivery surfaces route
    it only to that agent) INSTEAD of the pending-correction state slot —
    closing the "written but unread under request_context" gap PR 9's
    KNOWN LIMITATION note marked. When ``False`` (the default, and the
    legacy / ``pin_assigned_task=True`` paths) the pre-task-#11
    ``write_correction`` slot is used unchanged (the dynamic-instruction
    resolver reads it), so existing suites pass unmodified (§5.1). The
    returned ids are note ids in the notes regime, state keys otherwise.

    No-op when ``revised`` has no CORRECT-kind supersedes links, when
    the session has no state dict, or when the triggering drift is
    None (defensive — the steerer always provides one in practice).
    """
    if revised is None or drift is None:
        return []
    state = _session_state(session)
    if state is None:
        return []

    revised_tasks: list[Task] = list(getattr(revised, "tasks", None) or [])
    if not revised_tasks:
        return []

    revised_by_id: dict[str, Task] = {str(t.id): t for t in revised_tasks if t.id}
    prev_by_id: dict[str, Task] = {}
    if prev_plan is not None:
        for t in getattr(prev_plan, "tasks", None) or ():
            if getattr(t, "id", ""):
                prev_by_id[str(t.id)] = t

    written_keys: list[str] = []
    revision_number = int(getattr(revised, "revision_index", 0) or 0)
    now_ms = _now_ms()

    for new_task in revised_tasks:
        if getattr(new_task, "supersedes_kind", None) is not SupersessionKind.CORRECT:
            continue
        old_id = str(getattr(new_task, "supersedes", "") or "").strip()
        new_id = str(getattr(new_task, "id", "") or "").strip()
        if not old_id or not new_id:
            continue
        assignee = _bare_agent_name(getattr(new_task, "assignee_agent_id", ""))
        if not assignee:
            # Without an assignee we have no (agent, task) key to write.
            # Log and skip; the plan is still structurally valid (the
            # correction just never materialises into a prompt).
            log.debug(
                "queue_corrections_for_revision: CORRECT-kind task %r has no assignee; "
                "skipping correction write",
                new_id,
            )
            continue
        old_task = revised_by_id.get(old_id) or prev_by_id.get(old_id)
        if old_task is None:
            # Structural validator would have rejected a CORRECT-kind
            # link pointing at an unknown id, so in practice this is
            # unreachable. Defensive skip rather than raise.
            log.debug(
                "queue_corrections_for_revision: superseded id %r not found "
                "in revised or prev plan; skipping correction for task %r",
                old_id,
                new_id,
            )
            continue
        payload = build_correction_payload(
            new_task=new_task,
            old_task=old_task,
            drift=drift,
            revision_number=revision_number,
            issued_at_ms=now_ms,
        )
        if corrections_via_notes:
            note_id = _enqueue_correction_note(
                session=session, payload=payload, drift=drift
            )
            if note_id:
                written_keys.append(note_id)
                log.info(
                    "correction routed to observer-note queue for agent=%r "
                    "task=%r (superseded=%r, drift=%s, rev=%d, note=%s)",
                    payload["agent_name"],
                    payload["task_id"],
                    payload["superseded_task_id"],
                    payload["drift_kind"] or "(none)",
                    revision_number,
                    note_id,
                )
            continue
        key = write_correction(session, payload)
        if key is not None:
            written_keys.append(key)
            log.info(
                "correction queued for agent=%r task=%r (superseded=%r, drift=%s, rev=%d)",
                payload["agent_name"],
                payload["task_id"],
                payload["superseded_task_id"],
                payload["drift_kind"] or "(none)",
                revision_number,
            )
    return written_keys


def _enqueue_correction_note(
    *,
    session: Any,
    payload: Mapping[str, Any],
    drift: DriftEvent,
) -> str | None:
    """Enqueue one CORRECT-kind correction onto the ObserverNoteQueue.

    AGENCY-PRESERVATION.md task #11. The note carries the assignee as
    ``agent_id`` (bare) so the agent-scoped delivery surfaces (notably
    the ADK ``before_model`` surface) render it only on that agent's own
    model call — never on a sibling's, never on the coordinator boundary
    replay, and never on the loop-only tool-annotation surface. The
    ``drift_id`` is a stable, goldfive-minted key
    (:data:`~goldfive.observer_note_queue.CORRECTION_DRIFT_ID_PREFIX` +
    ``<agent>:<task>:<rev>``) — never an LLM-minted id. It is unique per
    revision: a later revision (higher ``rev``) mints a NEW note;
    re-enqueuing the SAME revision (e.g. a retry of the same refine)
    coalesces onto the existing entry.
    Returns the note id, or ``None`` on any failure (best-effort — a
    correction that can't be enqueued must not break the refine path).
    """
    try:
        # Lazy imports: ``adk_llm_instrumentation`` imports this module
        # (pending_correction_key et al.), so a module-level import here
        # would be circular.
        from goldfive.adapters.adk_llm_instrumentation import format_correction_block
        from goldfive.observer_note_queue import (
            CORRECTION_DRIFT_ID_PREFIX,
            ObserverNoteQueue,
        )

        agent = str(payload.get("agent_name", "") or "")
        task_id = str(payload.get("task_id", "") or "")
        rev = int(payload.get("revision_number", 0) or 0)
        if not agent or not task_id:
            return None
        body = format_correction_block(payload)
        if not body:
            return None
        superseded = str(payload.get("superseded_task_id", "") or "")
        observation = (
            f"the plan was revised (rev {rev}); task {task_id} supersedes "
            f"{superseded or '(prior task)'}"
        )
        severity = str(
            getattr(getattr(drift, "severity", None), "value", "") or "warning"
        ).lower()
        kind = str(getattr(getattr(drift, "kind", None), "value", "") or "") or "correction"
        turn = int(getattr(session, "_reasoning_turn", 0) or 0)
        queue = ObserverNoteQueue.for_session(session)
        note = queue.enqueue(
            body=body,
            observation=observation,
            severity=severity,
            drift_id=f"{CORRECTION_DRIFT_ID_PREFIX}{agent}:{task_id}:{rev}",
            kind=kind,
            task_id=task_id,
            agent_id=agent,
            turn=turn,
            ladder_level="correction",
        )
        return note.note_id
    except Exception as exc:  # noqa: BLE001
        log.debug("queue_corrections_for_revision: note enqueue raised: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Garbage-collection (clears)
# ---------------------------------------------------------------------------


def clear_correction(
    session: Any,
    *,
    agent_name: str,
    task_id: str,
) -> bool:
    """Remove the pending correction for ``(agent_name, task_id)``.

    Returns True when a correction was cleared, False when no entry
    existed (or the session has no state dict).

    Called from :mod:`goldfive.reporting` on
    :func:`report_task_started` for the correction task — the agent
    acknowledging the new task is our cue to stop re-injecting the
    correction block on subsequent turns.
    """
    if not agent_name or not task_id:
        return False
    state = _session_state(session)
    if state is None:
        return False
    key = pending_correction_key(agent_name, task_id)
    if key in state:
        state.pop(key, None)
        log.info(
            "cleared pending correction for agent=%r task=%r",
            agent_name,
            task_id,
        )
        return True
    return False


def clear_corrections_for_task(
    session: Any,
    task_id: str,
) -> list[str]:
    """Remove every pending correction targeted at ``task_id``.

    Used by the plan-revision GC path in
    :meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised`: when a
    new revision supersedes a correction task, every correction keyed
    on that task id becomes obsolete (the agent is no longer on that
    task and the follow-up revision speaks for the current context).

    Matches across all agents — a correction is ``(agent, task)``-keyed
    but a plan-revision supersession names a task, not an agent, so
    the sweep is task-scoped. Returns the list of cleared state keys.
    """
    if not task_id:
        return []
    state = _session_state(session)
    if state is None:
        return []
    suffix = f".{task_id}"
    prefix = _sp.KEY_PENDING_CORRECTIONS + "."
    cleared: list[str] = []
    # Snapshot keys before mutating — we pop while iterating otherwise.
    for key in list(state.keys()):
        if not isinstance(key, str):
            continue
        if not key.startswith(prefix):
            continue
        if not key.endswith(suffix):
            continue
        state.pop(key, None)
        cleared.append(key)
    if cleared:
        log.info(
            "cleared %d pending correction(s) scoped to task=%r (task was superseded by revision)",
            len(cleared),
            task_id,
        )
    return cleared


def clear_obsolete_corrections_on_revision(
    session: Any,
    revised: Plan,
) -> list[str]:
    """Sweep corrections whose target task was superseded in ``revised``.

    Walks the revised plan for any task whose own ``supersedes`` points
    at an earlier task id, and drops every correction keyed on that
    earlier id. Idempotent — a second call on the same revised plan
    finds nothing to clear.

    Paired with :func:`queue_corrections_for_revision` so one refine
    can simultaneously:

    * Queue new corrections for fresh CORRECT-kind supersedes, AND
    * Evict corrections for tasks the new revision itself supersedes
      (CORRECT or REPLACE — both make the prior pending correction
      obsolete, because the agent is moving onto the replacement /
      correction-of-the-correction).

    Returns the list of cleared state keys.
    """
    if revised is None:
        return []
    superseded_ids: set[str] = set()
    for task in getattr(revised, "tasks", None) or ():
        sup = str(getattr(task, "supersedes", "") or "").strip()
        if sup:
            superseded_ids.add(sup)
    if not superseded_ids:
        return []
    cleared: list[str] = []
    for old_id in superseded_ids:
        cleared.extend(clear_corrections_for_task(session, old_id))
    return cleared


__all__ = [
    "build_correction_payload",
    "clear_correction",
    "clear_corrections_for_task",
    "clear_obsolete_corrections_on_revision",
    "is_pending_correction_key",
    "pending_correction_key",
    "queue_corrections_for_revision",
    "write_correction",
]
