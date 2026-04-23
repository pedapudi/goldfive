"""Per-thinking-message LLM-as-a-judge reasoning-drift classifier.

Sibling to :mod:`goldfive.drift.goals`: the goal-drift judge asks
"is the whole trajectory progressing?" once every N agent invocations,
while :func:`classify_reasoning_drift` asks "is *this* reasoning block
on task?" at (rate-limited) per-thinking-message cadence.

Design goals (see goldfive#226):

* **Cost-bounded.** At most one LLM call per invocation of this
  function. The caller (the steerer) owns the rate-limit policy
  (see ``DefaultSteerer.reasoning_drift_rate_limit``) so flaky judges
  cannot spam the run.
* **No false positives on plumbing failures.** The LLM raising,
  returning malformed JSON, or returning a dict missing the
  ``on_task`` key all yield ``None``. Only an explicit
  ``{"on_task": false, ...}`` response produces a ``DriftEvent``.
* **Framework-neutral.** Like :func:`classify_goal_drift`, this
  classifier does not import from :mod:`goldfive.steerer` or any
  adapter -- it takes the data it needs via keyword arguments and
  returns a :class:`DriftEvent` or ``None``.

The prompt is pinned via module-level constants
(:data:`REASONING_DRIFT_SYSTEM_PROMPT` /
:data:`REASONING_DRIFT_USER_PROMPT_TEMPLATE`) so operators can override
wording without re-implementing the parse logic. Rationale for the
empirical failure of the pre-existing embedding-based pipeline lives in
goldfive#223 / #224 / #226.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Task

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "REASONING_DRIFT_MAX_REASONING_CHARS",
    "REASONING_DRIFT_SYSTEM_PROMPT",
    "REASONING_DRIFT_USER_PROMPT_TEMPLATE",
    "classify_reasoning_drift",
]


# Shape matches :mod:`goldfive.drift.goals` / ``LLMPlanner`` so operators
# can reuse the same callable across judges.
CallLLM = Callable[[str, str, str], Awaitable[str]]


# Reasoning blocks on real chain-of-thought models routinely hit several
# KB. We truncate before prompting so one long block cannot blow the
# judge's context budget. 1500 chars keeps ~300-400 tokens of reasoning
# -- enough context for an on-task/off-task call on every corpus we have
# calibrated against (goldfive#223).
REASONING_DRIFT_MAX_REASONING_CHARS: int = 1500


# Prompt templates. Module-level so tests (and subclasses of the
# steerer) can override the wording without re-implementing the parse
# logic.
REASONING_DRIFT_SYSTEM_PROMPT: str = (
    "You are assessing whether an autonomous agent's chain-of-thought "
    "is staying focused on its explicit task and goals. Reply with a "
    "single JSON object and nothing else."
)

REASONING_DRIFT_USER_PROMPT_TEMPLATE: str = (
    "You are assessing whether an autonomous agent's chain-of-thought "
    "is on task.\n\n"
    "GOALS:\n{goals_block}\n\n"
    "CURRENT TASK:\n{task_block}\n\n"
    "REASONING (the agent's most recent chain-of-thought block):\n"
    "{reasoning_block}\n\n"
    "Decide: does the reasoning stay focused on the explicit task and "
    "goals above? Answer STRICTLY in one of these two JSON shapes:\n"
    '{{"on_task": true}}\n'
    "OR\n"
    '{{"on_task": false, "severity": "info"|"warning"|"critical", '
    '"reason": "one-sentence explanation"}}\n\n'
    "on_task=true = reasoning is working toward the task / goals "
    "(clarifying sub-steps, exploring tradeoffs, working through a "
    "calculation all count as on_task).\n"
    "on_task=false = reasoning has drifted to an unrelated topic, is "
    "proposing to abandon the task, or is otherwise off-course.\n"
    "Severity guidance when on_task=false:\n"
    "- info = mild tangent that may self-correct next turn.\n"
    "- warning = clear off-topic content that deserves a nudge.\n"
    "- critical = proposing to abandon or replace the task/goal."
)


# Liberal JSON extractor. Real LLMs emit markdown code fences and prose
# even with strong "reply JSON only" instructions. Mirrors the extractor
# in :mod:`goldfive.drift.goals`.
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


def _format_task(task: Task | Any | None) -> str:
    if task is None:
        return "(no task bound)"
    tid = str(getattr(task, "id", "") or "")
    title = str(getattr(task, "title", "") or "")
    description = str(getattr(task, "description", "") or "")
    lines: list[str] = []
    if tid:
        lines.append(f"id: {tid}")
    if title:
        lines.append(f"title: {title}")
    if description:
        lines.append(f"description: {description}")
    return "\n".join(lines) if lines else "(no task bound)"


def _format_reasoning(reasoning: str) -> str:
    if not reasoning:
        return "(empty reasoning)"
    if len(reasoning) <= REASONING_DRIFT_MAX_REASONING_CHARS:
        return reasoning
    return reasoning[:REASONING_DRIFT_MAX_REASONING_CHARS] + " ... [truncated]"


# Map the judge's ``severity`` string to a :class:`DriftSeverity`. Missing
# or unknown values fall through to WARNING so a drift verdict is never
# silently swallowed by a bad severity string.
_SEVERITY_MAP: dict[str, DriftSeverity] = {
    "info": DriftSeverity.INFO,
    "warning": DriftSeverity.WARNING,
    "critical": DriftSeverity.CRITICAL,
}


def _severity_from_verdict(raw: Any) -> DriftSeverity:
    if not isinstance(raw, str):
        return DriftSeverity.WARNING
    return _SEVERITY_MAP.get(raw.strip().lower(), DriftSeverity.WARNING)


async def classify_reasoning_drift(
    *,
    reasoning: str,
    task: Task | None,
    goals: Sequence[Goal] | Iterable[Any] | None,
    model: str,
    call_llm: CallLLM,
    current_task_id: str = "",
    current_agent_id: str = "",
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> DriftEvent | None:
    """Ask an LLM-judge whether ``reasoning`` is on-task.

    Returns a :class:`DriftEvent` of kind
    :data:`~goldfive.types.DriftKind.OFF_TOPIC` when the judge returns
    ``{"on_task": false, ...}``. Severity comes from the judge's
    ``severity`` field (``info`` / ``warning`` / ``critical``);
    missing / unknown values default to
    :data:`~goldfive.types.DriftSeverity.WARNING`.

    Returns ``None`` in every other case:

    * judge returns ``{"on_task": true}`` (the on-track signal),
    * judge returns malformed / non-JSON text,
    * judge returns a dict missing / with a non-boolean ``on_task``,
    * ``call_llm`` raises.

    The "quiet on failure" contract matches :func:`classify_goal_drift`
    (goldfive#143). A flaky judge must not spam operator UIs with
    false-positive OFF_TOPIC alarms.

    Parameters
    ----------
    reasoning:
        The reasoning / chain-of-thought block to classify. Truncated
        to :data:`REASONING_DRIFT_MAX_REASONING_CHARS` before prompting
        so pathologically long blocks cannot blow the context budget.
        Empty or whitespace-only ``reasoning`` is treated as nothing to
        classify -- returns ``None`` without calling the LLM.
    task:
        The currently-bound :class:`Task` (typically from
        ``session.current_task_id``). May be ``None`` when no task is
        active; the judge then has only ``goals`` to compare against.
    goals:
        The session's goals. Each entry should have ``id`` / ``summary``
        attributes or be a plain str. May be ``None`` / empty.
    model:
        Model name forwarded verbatim to ``call_llm``. Empty string is
        permitted; model-bound callables can substitute their own
        default.
    call_llm:
        Async ``(system, user, model) -> str`` callable. Awaited at
        most once per invocation.
    current_task_id / current_agent_id:
        Stamped onto the returned ``DriftEvent`` for sink correlation.
        Optional -- empty strings are fine when no task is active.
    system_prompt / user_prompt_template:
        Override the default prompts. Operators wanting a different
        judge style can pass their own; the defaults match the shape
        pinned in :data:`REASONING_DRIFT_USER_PROMPT_TEMPLATE`.
    """
    if not reasoning or not reasoning.strip():
        return None
    system = system_prompt or REASONING_DRIFT_SYSTEM_PROMPT
    template = user_prompt_template or REASONING_DRIFT_USER_PROMPT_TEMPLATE
    user = template.format(
        goals_block=_format_goals(goals),
        task_block=_format_task(task),
        reasoning_block=_format_reasoning(reasoning),
    )
    try:
        raw = await call_llm(system, user, model)
    except Exception as exc:  # noqa: BLE001 - never break the run
        log.warning(
            "classify_reasoning_drift: call_llm raised %s; no drift emitted",
            exc,
        )
        return None
    raw_str = raw if isinstance(raw, str) else ""
    log.debug(
        "classify_reasoning_drift: raw response (%d chars): %s",
        len(raw_str),
        raw_str[:500],
    )
    parsed = _parse_response(raw)
    if parsed is None:
        log.debug(
            "classify_reasoning_drift: response was not JSON (raw=%r); "
            "no drift emitted",
            raw_str[:200],
        )
        return None
    on_task = parsed.get("on_task")
    if not isinstance(on_task, bool):
        log.debug(
            "classify_reasoning_drift: parsed=%r lacks boolean 'on_task' "
            "key; no drift emitted",
            parsed,
        )
        return None
    if on_task:
        log.debug(
            "classify_reasoning_drift: judge says on-track (reason=%r)",
            parsed.get("reason", ""),
        )
        return None
    severity = _severity_from_verdict(parsed.get("severity"))
    reason = str(parsed.get("reason", "") or "").strip()
    log.info(
        "classify_reasoning_drift: drift detected (severity=%s, reason=%r); "
        "emitting OFF_TOPIC event",
        severity.value,
        reason,
    )
    detail = (
        f"reasoning drift: {reason}"
        if reason
        else "reasoning drift detected (judge returned no reason)"
    )
    return DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=severity,
        detail=detail,
        current_task_id=current_task_id,
        current_agent_id=current_agent_id,
        raw=reasoning,
    )
