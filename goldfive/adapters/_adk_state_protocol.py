"""ADK session.state protocol for the goldfive adapter.

This is a ported subset of harmonograf's ``state_protocol`` module,
re-namespaced under the ``goldfive.*`` prefix. It owns the key names
goldfive writes into an ADK ``session.state`` dict so agents can read
the active task and plan context during their turn, and reads back
agent-side writes (progress, outcome, notes, divergence flag) after
the turn.

Two directions share the state channel:

* **Goldfive -> Agents** — written by the adapter in
  ``before_model_callback`` before the LLM call. Keys under this
  direction advertise the active task, plan summary, and the set of
  reporting tools that are wired up.
* **Agents -> Goldfive** — written by the agent as ``state_delta``
  events (or by the plugin's ``before_tool_callback`` interception of
  the canonical reporting tools). Keys under this direction carry
  progress, outcome, free-form notes, and the divergence flag.

All keys live under :data:`GOLDFIVE_PREFIX`. Non-goldfive keys in the
state dict are ignored — readers never raise on missing or malformed
state, and writers refuse to stamp non-goldfive keys so this module
can't accidentally clobber application state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

GOLDFIVE_PREFIX = "goldfive."

# Goldfive -> Agents
KEY_CURRENT_TASK_ID = "goldfive.current_task_id"
KEY_CURRENT_TASK_TITLE = "goldfive.current_task_title"
KEY_CURRENT_TASK_DESCRIPTION = "goldfive.current_task_description"
KEY_CURRENT_TASK_ASSIGNEE = "goldfive.current_task_assignee"
KEY_PLAN_ID = "goldfive.plan_id"
KEY_PLAN_SUMMARY = "goldfive.plan_summary"
KEY_RUN_ID = "goldfive.run_id"
KEY_COMPLETED_TASK_RESULTS = "goldfive.completed_task_results"
KEY_AVAILABLE_TASKS = "goldfive.available_tasks"
KEY_TOOLS_AVAILABLE = "goldfive.tools_available"

# Agents -> Goldfive
KEY_TASK_PROGRESS = "goldfive.task_progress"
KEY_TASK_OUTCOME = "goldfive.task_outcome"
KEY_AGENT_NOTE = "goldfive.agent_note"
KEY_DIVERGENCE_FLAG = "goldfive.divergence_flag"

# Dynamic instruction (goldfive#251). Prefix key for correction-injection
# bodies written by Stream D's correction-injection path and read by the
# dynamic instruction resolver in :mod:`goldfive.adapters._adk_dynainst`.
# The full key is ``{prefix}.{agent_name}.{task_id}`` so a correction is
# scoped to the exact agent+task pair it was authored for and does not
# leak into sibling agents or later tasks. Reader-only in this module;
# Stream D owns the writer.
KEY_PENDING_CORRECTIONS = "goldfive.pending_corrections"

# Orchestration -> Agents (goldfive#170 — bridged from
# ``goldfive.Session.state`` into the live ADK session.state so
# :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner` sees them
# on its request-side injection path. Writers live in
# :mod:`goldfive.orchestration_state`; this module owns the ADK-side key
# names so the adapter's plugin can stamp them without importing the
# orchestration module for key constants. The string values are
# intentionally identical to the orchestration-state module's keys —
# same logical field, two readers.
KEY_ACTIVE_STEER_BODY = "goldfive.active_steer.body"
KEY_ACTIVE_STEER_AT_TURN = "goldfive.active_steer.at_turn"
KEY_GOALS_SUMMARY = "goldfive.goals_summary"
KEY_CANCELLED_FUNCTION_CALL_IDS = "goldfive.cancelled_function_call_ids"

# Cooperative cancellation (goldfive#251 Stream C / 7a). Value is a
# ``dict[str, CancellationRequest]`` keyed by ``invocation_id``. Every
# adapter callback that can short-circuit a dispatch checks for an
# entry under the current invocation_id at the top of the callback
# and, when present, consumes it (reads + clears) and short-circuits
# the dispatch. See
# :mod:`goldfive.adapters._cancel_state` for the helper API and
# :class:`goldfive.types.CancellationRequest` for the payload shape.
KEY_CANCEL_REQUESTED = "goldfive.cancel_requested"
# Parent -> invocation_id bookkeeping for cancellation propagation.
# ``dict[str, str]`` mapping ``invocation_id -> parent_invocation_id``.
# Written by the plugin's ``before_run_callback`` whenever a fresh
# invocation_id is observed; consumed by the cancel-propagation helper
# so cancelling an invocation also flags its spawned children.
# Tree-agnostic — the map is per-invocation and carries no notion of
# "coordinator" or "root"; every agent's children are handled the same
# way.
KEY_INVOCATION_PARENTS = "goldfive.invocation_parents"

_CURRENT_TASK_KEYS: tuple[str, ...] = (
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
    KEY_CURRENT_TASK_DESCRIPTION,
    KEY_CURRENT_TASK_ASSIGNEE,
)

ALL_KEYS: tuple[str, ...] = (
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
    KEY_CURRENT_TASK_DESCRIPTION,
    KEY_CURRENT_TASK_ASSIGNEE,
    KEY_PLAN_ID,
    KEY_PLAN_SUMMARY,
    KEY_RUN_ID,
    KEY_COMPLETED_TASK_RESULTS,
    KEY_AVAILABLE_TASKS,
    KEY_TOOLS_AVAILABLE,
    KEY_TASK_PROGRESS,
    KEY_TASK_OUTCOME,
    KEY_AGENT_NOTE,
    KEY_DIVERGENCE_FLAG,
    KEY_ACTIVE_STEER_BODY,
    KEY_ACTIVE_STEER_AT_TURN,
    KEY_GOALS_SUMMARY,
    KEY_CANCELLED_FUNCTION_CALL_IDS,
    KEY_CANCEL_REQUESTED,
    KEY_INVOCATION_PARENTS,
)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _safe_get(state: Any, key: str, default: Any = None) -> Any:
    if not isinstance(state, Mapping):
        return default
    try:
        value = state.get(key, default)
    except Exception:
        return default
    return value if value is not None else default


def _assert_goldfive_key(key: str) -> None:
    if not key.startswith(GOLDFIVE_PREFIX):
        raise ValueError(f"state_protocol refuses to write non-goldfive key: {key!r}")


def _set(state: MutableMapping[str, Any], key: str, value: Any) -> None:
    _assert_goldfive_key(key)
    state[key] = value


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_current_task(state: Any) -> dict:
    """Return ``{id, title, description, assignee}`` for the active task."""
    return {
        "id": _safe_str(_safe_get(state, KEY_CURRENT_TASK_ID, "")),
        "title": _safe_str(_safe_get(state, KEY_CURRENT_TASK_TITLE, "")),
        "description": _safe_str(_safe_get(state, KEY_CURRENT_TASK_DESCRIPTION, "")),
        "assignee": _safe_str(_safe_get(state, KEY_CURRENT_TASK_ASSIGNEE, "")),
    }


def read_run_id(state: Any) -> str:
    return _safe_str(_safe_get(state, KEY_RUN_ID, ""))


def read_plan_id(state: Any) -> str:
    return _safe_str(_safe_get(state, KEY_PLAN_ID, ""))


def read_plan_summary(state: Any) -> str:
    return _safe_str(_safe_get(state, KEY_PLAN_SUMMARY, ""))


def read_completed_results(state: Any) -> dict:
    value = _safe_get(state, KEY_COMPLETED_TASK_RESULTS, None)
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        if isinstance(k, str):
            out[k] = _safe_str(v)
    return out


def read_available_tasks(state: Any) -> list[dict]:
    value = _safe_get(state, KEY_AVAILABLE_TASKS, None)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def read_tools_available(state: Any) -> list[str]:
    value = _safe_get(state, KEY_TOOLS_AVAILABLE, None)
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def read_agent_outcome(state: Any, task_id: str) -> str:
    value = _safe_get(state, KEY_TASK_OUTCOME, None)
    if not isinstance(value, Mapping):
        return ""
    return _safe_str(value.get(task_id, ""))


def read_agent_progress(state: Any, task_id: str) -> float:
    value = _safe_get(state, KEY_TASK_PROGRESS, None)
    if not isinstance(value, Mapping):
        return 0.0
    raw = value.get(task_id, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def read_agent_note(state: Any) -> str:
    return _safe_str(_safe_get(state, KEY_AGENT_NOTE, ""))


def read_divergence_flag(state: Any) -> bool:
    return bool(_safe_get(state, KEY_DIVERGENCE_FLAG, False))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_current_task_id(state: MutableMapping[str, Any], task_id: str) -> None:
    """Stamp just ``goldfive.current_task_id`` onto ``state``.

    Narrower than :func:`write_current_task` which rewrites all four
    ``current_task_*`` keys from a :class:`Task`. Used by the adapter's
    ``before_agent_callback`` pin path (goldfive#191 Layer 1) where the
    plugin only knows the id at delegation time and doesn't want to
    overwrite the title/description/assignee written earlier by the
    reconciler.

    Empty / None ``task_id`` is a no-op — callers that want to clear
    the key should use :func:`clear_current_task`.
    """
    if not task_id:
        return
    _set(state, KEY_CURRENT_TASK_ID, _safe_str(task_id))


def write_current_task(state: MutableMapping[str, Any], task: Any) -> None:
    """Mutate state with current_task_* fields from a :class:`goldfive.types.Task`.

    Accepts anything that duck-types to goldfive's ``Task`` (``.id``,
    ``.title``, ``.description``, ``.assignee_agent_id``) or a mapping
    with equivalent keys. ``None`` clears the current task.
    """
    if task is None:
        clear_current_task(state)
        return

    if isinstance(task, Mapping):
        tid = task.get("id", "")
        title = task.get("title", "")
        description = task.get("description", "")
        assignee = task.get("assignee") or task.get("assignee_agent_id", "")
    else:
        tid = getattr(task, "id", "")
        title = getattr(task, "title", "")
        description = getattr(task, "description", "")
        assignee = getattr(task, "assignee_agent_id", "") or getattr(task, "assignee", "")

    _set(state, KEY_CURRENT_TASK_ID, _safe_str(tid))
    _set(state, KEY_CURRENT_TASK_TITLE, _safe_str(title))
    _set(state, KEY_CURRENT_TASK_DESCRIPTION, _safe_str(description))
    _set(state, KEY_CURRENT_TASK_ASSIGNEE, _safe_str(assignee))


def clear_current_task(state: MutableMapping[str, Any]) -> None:
    for key in _CURRENT_TASK_KEYS:
        if key in state:
            state.pop(key, None)


def write_plan_context(
    state: MutableMapping[str, Any],
    plan: Any,
    completed_results: Mapping[str, Any] | None,
    host_agent: str,
) -> None:
    """Write plan_id, summary, available_tasks, and completed_results.

    ``plan`` duck-types to :class:`goldfive.types.Plan` — anything with
    ``.id``, ``.summary``, ``.tasks``, and optional ``.edges`` works.
    ``host_agent`` is the fallback assignee rendered into
    ``available_tasks`` when a task has none.
    """
    plan_id = ""
    summary = ""
    tasks_iter: Iterable[Any] = ()
    edges_iter: Iterable[Any] = ()

    if plan is not None:
        plan_id = _safe_str(getattr(plan, "id", None) or getattr(plan, "plan_id", ""))
        summary = _safe_str(getattr(plan, "summary", ""))
        tasks_iter = getattr(plan, "tasks", ()) or ()
        edges_iter = getattr(plan, "edges", ()) or ()

    deps_by_task: dict[str, list[str]] = {}
    for edge in edges_iter:
        src = _safe_str(getattr(edge, "from_task_id", ""))
        dst = _safe_str(getattr(edge, "to_task_id", ""))
        if not src or not dst:
            continue
        deps_by_task.setdefault(dst, []).append(src)

    available: list[dict] = []
    for task in tasks_iter:
        tid = _safe_str(getattr(task, "id", ""))
        status = getattr(task, "status", "PENDING")
        # Tolerate StrEnum or plain strings.
        status_str = _safe_str(getattr(status, "value", status) or "PENDING")
        available.append(
            {
                "id": tid,
                "title": _safe_str(getattr(task, "title", "")),
                "assignee": _safe_str(getattr(task, "assignee_agent_id", "") or host_agent),
                "status": status_str,
                "deps": list(deps_by_task.get(tid, [])),
            }
        )

    _set(state, KEY_PLAN_ID, plan_id)
    _set(state, KEY_PLAN_SUMMARY, summary)
    _set(state, KEY_AVAILABLE_TASKS, available)

    results_out: dict[str, str] = {}
    if isinstance(completed_results, Mapping):
        for k, v in completed_results.items():
            if isinstance(k, str):
                results_out[k] = _safe_str(v)
    _set(state, KEY_COMPLETED_TASK_RESULTS, results_out)


def write_run_id(state: MutableMapping[str, Any], run_id: str) -> None:
    _set(state, KEY_RUN_ID, _safe_str(run_id))


def write_tools_available(state: MutableMapping[str, Any], tool_names: Iterable[str]) -> None:
    _set(
        state,
        KEY_TOOLS_AVAILABLE,
        [name for name in tool_names if isinstance(name, str)],
    )


# ---------------------------------------------------------------------------
# Bridge writers (goldfive#170)
#
# The four helpers below stamp orchestration-state values onto the ADK
# session.state dict. They exist so the adapter plugin's
# ``before_run_callback`` can copy values from ``goldfive.Session.state``
# (orchestration-level, written by DefaultSteerer / reconciler / heal
# paths) into the ADK session.state (per-agent view, read by
# :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner`).
#
# A missing / empty value is cleared rather than stamped so a prior
# bridged write from an earlier invocation doesn't linger past its
# orchestration-state clear.
# ---------------------------------------------------------------------------


def set_active_steer_on_adk_state(
    state: MutableMapping[str, Any],
    *,
    body: str | None,
    at_turn: int | None,
) -> None:
    """Bridge ``goldfive.active_steer.*`` onto ADK session.state.

    Clears both keys when ``body`` is falsy / ``at_turn`` is None so a
    later invocation after a steer clear never renders a stale body.
    """
    if not body:
        state.pop(KEY_ACTIVE_STEER_BODY, None)
        state.pop(KEY_ACTIVE_STEER_AT_TURN, None)
        return
    _set(state, KEY_ACTIVE_STEER_BODY, _safe_str(body))
    try:
        _set(state, KEY_ACTIVE_STEER_AT_TURN, int(at_turn) if at_turn is not None else 0)
    except (TypeError, ValueError):
        _set(state, KEY_ACTIVE_STEER_AT_TURN, 0)


def set_goals_summary_on_adk_state(
    state: MutableMapping[str, Any],
    summary: str | None,
) -> None:
    """Bridge ``goldfive.goals_summary`` onto ADK session.state.

    An empty / None summary clears the key so a planner that re-reads
    after ``session.goals`` was itself cleared doesn't see the stale
    rendering.
    """
    if not summary:
        state.pop(KEY_GOALS_SUMMARY, None)
        return
    _set(state, KEY_GOALS_SUMMARY, _safe_str(summary))


def set_cancelled_function_call_ids_on_adk_state(
    state: MutableMapping[str, Any],
    ids: Iterable[str] | None,
) -> None:
    """Bridge ``goldfive.cancelled_function_call_ids`` onto ADK session.state.

    Rewrites the whole list on each call (the orchestration-state owner
    is append-only within a run, but the bridge's job is just to mirror
    whatever the orchestration dict currently holds). Empty / None
    input clears the key.
    """
    if not ids:
        state.pop(KEY_CANCELLED_FUNCTION_CALL_IDS, None)
        return
    cleaned = [str(v) for v in ids if v]
    if not cleaned:
        state.pop(KEY_CANCELLED_FUNCTION_CALL_IDS, None)
        return
    _set(state, KEY_CANCELLED_FUNCTION_CALL_IDS, cleaned)


# ---------------------------------------------------------------------------
# Cooperative cancellation (goldfive#251 Stream C / 7a)
#
# The helpers below manage two state keys used by cooperative
# cancellation:
#
# * :data:`KEY_CANCEL_REQUESTED` — ``dict[str, CancellationRequest]``
#   keyed by ``invocation_id``; every adapter callback that can
#   short-circuit a dispatch reads (and consumes) the entry for its
#   own invocation_id at the top of the callback.
# * :data:`KEY_INVOCATION_PARENTS` — ``dict[str, str]`` mapping
#   ``invocation_id -> parent_invocation_id`` so cancel propagation
#   can flag children when a parent is cancelled, without requiring
#   the plugin to walk ADK's own context graph.
#
# ``CancellationRequest`` is imported lazily to avoid a module-level
# circular dependency (``goldfive.types`` imports are cheap but we
# keep this file import-lean so it can load in tests that don't have
# the ADK optional-dependency group).
# ---------------------------------------------------------------------------


def write_cancel_request(
    state: MutableMapping[str, Any],
    *,
    invocation_id: str,
    request: Any,
) -> None:
    """Stamp a :class:`~goldfive.types.CancellationRequest` on ``state``.

    No-op when ``invocation_id`` is empty. ``request`` is stored
    verbatim so consumers get the original dataclass (not a copy) —
    this is the cheapest handoff between the steerer (producer) and
    the adapter callbacks (consumers) on a per-invocation basis.

    Multiple requests for different invocation_ids coexist in the
    same dict; a second write for the same id overwrites the prior
    request (the more-recent cancel wins, same as the steerer's
    Level-3 corrective-message slot).
    """
    if not invocation_id:
        return
    bucket = state.get(KEY_CANCEL_REQUESTED)
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[str(invocation_id)] = request
    _set(state, KEY_CANCEL_REQUESTED, bucket)


def read_cancel_request(state: Any, invocation_id: str) -> Any | None:
    """Return the :class:`~goldfive.types.CancellationRequest` for
    ``invocation_id``, or ``None`` when no cancel is pending.

    Read-only — callers that want "cancel fires once" semantics must
    use :func:`consume_cancel_request` instead, which clears the
    entry after reading so re-entry into the same callback doesn't
    re-cancel.
    """
    if not invocation_id:
        return None
    bucket = _safe_get(state, KEY_CANCEL_REQUESTED, None)
    if not isinstance(bucket, Mapping):
        return None
    return bucket.get(str(invocation_id))


def consume_cancel_request(
    state: MutableMapping[str, Any],
    invocation_id: str,
) -> Any | None:
    """Read the cancel request for ``invocation_id`` and clear it.

    Returns the :class:`~goldfive.types.CancellationRequest` dataclass
    (not a copy) when one was pending, or ``None`` otherwise. The
    flag is removed from ``state`` BEFORE returning so a re-entry
    into the same callback (e.g. a retry after the LLM call was
    already cancelled and a new before_tool fires for a lingering
    tool call) does not re-emit the cancelled response.
    """
    if not invocation_id:
        return None
    bucket = state.get(KEY_CANCEL_REQUESTED) if isinstance(state, MutableMapping) else None
    if not isinstance(bucket, dict):
        return None
    request = bucket.pop(str(invocation_id), None)
    # Cleanup: if the bucket is now empty, remove the key entirely so
    # downstream readers distinguish "no cancels ever" from "a cancel
    # was processed and cleared".
    if not bucket:
        state.pop(KEY_CANCEL_REQUESTED, None)
    else:
        _set(state, KEY_CANCEL_REQUESTED, bucket)
    return request


def register_invocation_parent(
    state: MutableMapping[str, Any],
    *,
    invocation_id: str,
    parent_invocation_id: str,
) -> None:
    """Record that ``invocation_id`` was spawned by
    ``parent_invocation_id``.

    Called from the plugin's ``before_run_callback`` (or equivalent)
    once per invocation so cancel propagation can walk the chain
    without needing to know the ADK context graph. No-op when either
    id is empty.
    """
    if not invocation_id or not parent_invocation_id:
        return
    parents = state.get(KEY_INVOCATION_PARENTS)
    if not isinstance(parents, dict):
        parents = {}
    parents[str(invocation_id)] = str(parent_invocation_id)
    _set(state, KEY_INVOCATION_PARENTS, parents)


def children_of_invocation(state: Any, invocation_id: str) -> list[str]:
    """Return the immediate children of ``invocation_id``.

    Reads the ``KEY_INVOCATION_PARENTS`` map and returns the list of
    invocation_ids whose parent is ``invocation_id``. Empty list when
    the invocation has no children or the map hasn't been populated.
    """
    if not invocation_id:
        return []
    parents = _safe_get(state, KEY_INVOCATION_PARENTS, None)
    if not isinstance(parents, Mapping):
        return []
    target = str(invocation_id)
    return [str(child) for child, parent in parents.items() if parent == target]


def descendants_of_invocation(state: Any, invocation_id: str) -> list[str]:
    """Return every transitive descendant of ``invocation_id``.

    Walks the parent map breadth-first so cancel propagation can flag
    the entire sub-tree in one pass. The order is unspecified but the
    ``invocation_id`` itself is NOT included in the result — callers
    that want the full cancelled set should prepend it themselves.
    """
    if not invocation_id:
        return []
    parents = _safe_get(state, KEY_INVOCATION_PARENTS, None)
    if not isinstance(parents, Mapping):
        return []
    # Build reverse index: parent -> [children]
    reverse: dict[str, list[str]] = {}
    for child, parent in parents.items():
        reverse.setdefault(str(parent), []).append(str(child))
    out: list[str] = []
    frontier = [str(invocation_id)]
    seen: set[str] = {str(invocation_id)}
    while frontier:
        node = frontier.pop(0)
        for child in reverse.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            out.append(child)
            frontier.append(child)
    return out


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def extract_agent_writes(before: Any, after: Any) -> dict:
    """Return goldfive.* keys the agent added, changed, or removed.

    A removed key is represented as ``{key: None}`` so callers can
    distinguish it from an unset key. ``before``/``after`` may be any
    mapping; non-mapping inputs are treated as empty.
    """
    before_map: Mapping[str, Any] = before if isinstance(before, Mapping) else {}
    after_map: Mapping[str, Any] = after if isinstance(after, Mapping) else {}

    before_g = {
        k: v for k, v in before_map.items() if isinstance(k, str) and k.startswith(GOLDFIVE_PREFIX)
    }
    after_g = {
        k: v for k, v in after_map.items() if isinstance(k, str) and k.startswith(GOLDFIVE_PREFIX)
    }

    changes: dict[str, Any] = {}
    for key, value in after_g.items():
        if key not in before_g:
            changes[key] = value
        elif before_g[key] != value:
            changes[key] = value
    for key in before_g:
        if key not in after_g:
            changes[key] = None
    return changes
