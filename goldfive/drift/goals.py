"""Periodic goal-alignment (trajectory-level) drift classifier.

Unlike every other drift classifier in :mod:`goldfive.drift`, which
decides drift from one LLM response or tool result at a time,
:func:`classify_goal_drift` takes a snapshot of the whole trajectory --
``session.goals``, ``session.plan``, and a short list of recent agent
invocations -- and asks an LLM-judge: *is the tree making progress
toward the goals?*

Design goals (see goldfive#143):

* **Cost-bounded.** At most one LLM call per invocation of this
  function. Callers (the steerer) are responsible for the scheduling
  policy (every N agent turns, after M seconds of no task transitions,
  etc).
* **No false positives on plumbing failures.** The LLM call raising,
  returning malformed JSON, or returning a truthy "progressing" answer
  all yield ``None``. Only an explicit ``{"progressing": false}`` JSON
  response produces a ``DriftEvent``.
* **Framework-neutral.** The classifier does not import from
  :mod:`goldfive.steerer` or any adapter. It takes the data it needs
  via keyword arguments and returns a :class:`DriftEvent` or ``None``,
  same shape as the rest of the classifiers in :mod:`goldfive.drift`.

The suggested prompt shape is pinned via module-level class attributes
(:data:`GOAL_DRIFT_SYSTEM_PROMPT` / :data:`GOAL_DRIFT_USER_PROMPT_TEMPLATE`)
so operators can override the wording without re-implementing the
parse logic.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "classify_goal_drift",
    "GOAL_DRIFT_SYSTEM_PROMPT",
    "GOAL_DRIFT_USER_PROMPT_TEMPLATE",
    "GOAL_DRIFT_CHECK_INTERVAL",
    "GOAL_DRIFT_IDLE_SECONDS",
]


# Default periodic-check cadence. Consumed by ``DefaultSteerer`` (see
# :class:`~goldfive.steerer.DefaultSteerer` -- ``goal_drift_check_interval``
# parameter). One LLM call per check, so defaults are deliberately
# low-frequency; operators who want tighter monitoring can shorten it.
GOAL_DRIFT_CHECK_INTERVAL: int = 5
GOAL_DRIFT_IDLE_SECONDS: int = 300


# Type alias for the async callable this classifier takes. Matches the
# ``(system, user, model) -> str`` shape used by ``LLMPlanner`` and the
# opt-in reflective check so operators can reuse the same callable.
CallLLM = Callable[[str, str, str], Awaitable[str]]


# Shared prompt templates. Module-level so tests (and subclasses of the
# steerer) can override without re-implementing the parse logic.
GOAL_DRIFT_SYSTEM_PROMPT: str = (
    "You are assessing whether an autonomous agent tree is making "
    "progress toward a stated goal. Reply with a single JSON object "
    "and nothing else."
)

GOAL_DRIFT_USER_PROMPT_TEMPLATE: str = (
    "You are assessing whether an autonomous agent tree is making "
    "progress toward a stated goal.\n\n"
    "GOALS:\n{goals_block}\n\n"
    "PLANNED TASKS:\n{tasks_block}\n\n"
    "RECENT AGENT ACTIVITY (most recent {activity_count} invocations, "
    "newest last):\n{activity_block}\n\n"
    "Decide: is the recent activity moving toward the goals? Answer "
    "STRICTLY in one of these two JSON shapes:\n"
    '{{"progressing": true}}\n'
    "OR\n"
    '{{"progressing": false, "reason": "one-sentence explanation"}}\n\n'
    "Progressing = agents are doing work that plausibly contributes to "
    "the goal.\n"
    "Not progressing = agents are looping, refusing, off-topic, or "
    "otherwise not advancing."
)


# Liberal JSON extractor: real LLMs emit markdown code fences and prose
# even with strong "reply JSON only" instructions. Mirrors the extractor
# used by the reflective-check path.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: Any) -> dict[str, Any] | None:
    """Extract the first JSON object from ``raw`` or return ``None``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    stripped = raw.strip()
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            return None
        try:
            decoded = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _format_goals(goals: Sequence[Goal] | Iterable[Any] | None) -> str:
    if not goals:
        return "(no goals recorded)"
    lines: list[str] = []
    for i, g in enumerate(goals, start=1):
        gid = str(getattr(g, "id", "") or "")
        summary = str(getattr(g, "summary", "") or "")
        if not summary and isinstance(g, str):
            summary = g
        prefix = f"{i}."
        if gid:
            lines.append(f"{prefix} [{gid}] {summary}")
        else:
            lines.append(f"{prefix} {summary}")
    return "\n".join(lines) if lines else "(no goals recorded)"


def _format_tasks(plan: Plan | Any | None) -> str:
    if plan is None:
        return "(no plan yet)"
    tasks = getattr(plan, "tasks", None) or []
    if not tasks:
        return "(plan has no tasks)"
    lines: list[str] = []
    for i, t in enumerate(tasks, start=1):
        tid = str(getattr(t, "id", "") or "")
        title = str(getattr(t, "title", "") or "")
        status = getattr(t, "status", "")
        status_str = str(getattr(status, "value", status) or "").upper() or "UNSPECIFIED"
        if tid:
            lines.append(f"{i}. [{tid}] {title} ({status_str})")
        else:
            lines.append(f"{i}. {title} ({status_str})")
    return "\n".join(lines)


def _format_activity(observed_actions: Iterable[Any] | None) -> tuple[str, int]:
    """Render recent agent activity as a short bulleted summary.

    Accepts either dict-shaped or attribute-shaped entries. Known
    fields: ``kind`` ("agent_invocation_started" / "agent_invocation_completed"
    / arbitrary), ``agent_name``, ``agent_id``, ``task_id``,
    ``invocation_id``, ``summary``, ``detail``. Unknown shapes fall back
    to ``repr(entry)`` truncated to 160 chars so we do not silently
    drop information.
    """
    items = list(observed_actions or [])
    if not items:
        return "(no recent agent activity)", 0
    lines: list[str] = []
    for entry in items:
        kind = ""
        agent = ""
        task_id = ""
        detail = ""
        if isinstance(entry, dict):
            kind = str(entry.get("kind", "") or "")
            agent = str(entry.get("agent_name", "") or entry.get("agent_id", "") or "")
            task_id = str(entry.get("task_id", "") or "")
            detail = str(entry.get("summary", "") or entry.get("detail", "") or "")
        else:
            kind = str(getattr(entry, "kind", "") or "")
            agent = str(getattr(entry, "agent_name", "") or getattr(entry, "agent_id", "") or "")
            task_id = str(getattr(entry, "task_id", "") or "")
            detail = str(getattr(entry, "summary", "") or getattr(entry, "detail", "") or "")
        if not any((kind, agent, task_id, detail)):
            lines.append(f"- {repr(entry)[:160]}")
            continue
        head = kind or "activity"
        parts: list[str] = []
        if agent:
            parts.append(f"agent={agent}")
        if task_id:
            parts.append(f"task={task_id}")
        if detail:
            parts.append(detail[:160])
        lines.append(f"- {head}" + (": " + " | ".join(parts) if parts else ""))
    return "\n".join(lines), len(items)


async def classify_goal_drift(
    *,
    goals: Sequence[Goal] | Iterable[Any] | None,
    plan: Plan | Any | None,
    observed_actions: Iterable[Any] | None,
    model: str,
    call_llm: CallLLM,
    current_task_id: str = "",
    current_agent_id: str = "",
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    sinks: list[Any] | None = None,
    run_id: str = "",
    session_id: str = "",
    sequence_fn: Callable[[], int] | None = None,
) -> DriftEvent | None:
    """Ask an LLM-judge whether recent activity is progressing the goals.

    Returns a :class:`DriftEvent` of kind
    :data:`~goldfive.types.DriftKind.GOAL_DRIFT` at
    :data:`~goldfive.types.DriftSeverity.CRITICAL` when the judge
    returns ``{"progressing": false, "reason": "..."}``. Returns
    ``None`` in every other case:

    * judge returns ``{"progressing": true}`` (the on-track signal),
    * judge returns malformed / non-JSON text,
    * judge returns a dict missing / with a non-boolean ``progressing``,
    * ``call_llm`` raises.

    The "quiet on failure" contract is deliberate: a flaky judge must
    not spam operator UIs with false-positive GOAL_DRIFT alarms. This
    is a trajectory-level check -- false positives erode trust faster
    than they help. See goldfive#143.

    Parameters
    ----------
    goals:
        The session's goals (typically ``session.goals``). Each entry
        should have ``id`` / ``summary`` attributes or be a plain str.
    plan:
        The session's current plan (typically ``session.plan``). May
        be ``None`` on early-run checks before plan submission.
    observed_actions:
        Recent agent activity -- typically a list of dicts summarising
        the last N ``AgentInvocationStarted`` / ``AgentInvocationCompleted``
        events, or any duck-typed shape with ``kind`` / ``agent_name``
        / ``task_id`` / ``detail`` attributes. Trimmed by the caller
        to keep the prompt bounded.
    model:
        Model name forwarded verbatim to ``call_llm``. Empty string is
        permitted; model-bound callables can substitute their own
        default.
    call_llm:
        Async ``(system, user, model) -> str`` callable. Never awaited
        more than once per invocation of this function.
    current_task_id / current_agent_id:
        Stamped onto the returned ``DriftEvent`` for sink correlation.
        Optional -- empty strings are fine when no task is active.
    system_prompt / user_prompt_template:
        Override the default prompts. Operators wanting a different
        judge style can pass their own; the defaults match the shape
        pinned in :data:`GOAL_DRIFT_USER_PROMPT_TEMPLATE`.
    """
    system = system_prompt or GOAL_DRIFT_SYSTEM_PROMPT
    template = user_prompt_template or GOAL_DRIFT_USER_PROMPT_TEMPLATE
    goals_block = _format_goals(goals)
    tasks_block = _format_tasks(plan)
    activity_block, activity_count = _format_activity(observed_actions)
    user = template.format(
        goals_block=goals_block,
        tasks_block=tasks_block,
        activity_block=activity_block,
        activity_count=activity_count,
    )
    try:
        from goldfive._llm_span import goldfive_llm_span

        async with goldfive_llm_span(
            sinks=list(sinks or []),
            name="judge_goal_drift",
            model=model,
            session_id=session_id,
            run_id=run_id,
            task_id=current_task_id,
            sequence_fn=sequence_fn,
        ):
            raw = await call_llm(system, user, model)
    except Exception as exc:  # noqa: BLE001 - never break the run
        log.warning("classify_goal_drift: call_llm raised %s; no drift emitted", exc)
        return None
    # Debug-log the raw judge response so operators can distinguish
    # "judge said progressing=true" from "judge returned garbage" without
    # bisecting (goldfive#219). Truncated to 500 chars to bound log size.
    raw_str = raw if isinstance(raw, str) else ""
    log.debug(
        "classify_goal_drift: raw response (%d chars): %s",
        len(raw_str),
        raw_str[:500],
    )
    parsed = _parse_response(raw)
    if parsed is None:
        log.debug(
            "classify_goal_drift: response was not JSON (raw=%r); no drift emitted",
            raw_str[:200],
        )
        return None
    progressing = parsed.get("progressing")
    if not isinstance(progressing, bool):
        log.debug(
            "classify_goal_drift: parsed=%r lacks boolean 'progressing' key; "
            "no drift emitted",
            parsed,
        )
        return None
    if progressing:
        log.debug(
            "classify_goal_drift: judge says on-track (reason=%r)",
            parsed.get("reason", ""),
        )
        return None
    reason = str(parsed.get("reason", "") or "").strip()
    log.info(
        "classify_goal_drift: drift detected (reason=%r); emitting GOAL_DRIFT event",
        reason,
    )
    detail = (
        f"goal drift detected: {reason}"
        if reason
        else "goal drift detected (judge returned no reason)"
    )
    # Stamp the activity summary the judge actually saw onto the drift
    # as ``trigger_input`` so downstream sinks can render "WHY did
    # goldfive decide this is off-goal?" on the timeline without
    # re-fetching the per-invocation activity log.
    trigger_input = _truncate_trigger_input(activity_block)
    return DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail=detail,
        current_task_id=current_task_id,
        current_agent_id=current_agent_id,
        trigger_input=trigger_input,
    )


_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS: int = 2048
_GOAL_DRIFT_TRUNCATE_SUFFIX: str = " … [truncated]"


def _truncate_trigger_input(text: str) -> str:
    """Cap the observability ``trigger_input`` at a bounded length."""
    if not isinstance(text, str):
        return ""
    if len(text) <= _GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS:
        return text
    return text[:_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS] + _GOAL_DRIFT_TRUNCATE_SUFFIX
