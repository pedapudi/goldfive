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
