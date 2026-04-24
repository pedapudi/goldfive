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
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Task

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "REASONING_JUDGE_MAX_REASONING_INPUT_CHARS",
    "REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS",
    "REASONING_DRIFT_MAX_REASONING_CHARS",
    "REASONING_DRIFT_SYSTEM_PROMPT",
    "REASONING_DRIFT_USER_PROMPT_TEMPLATE",
    "classify_reasoning_drift",
    "truncate_for_observability",
]


# ---------------------------------------------------------------------------
# Observability truncation bounds (goldfive judge-observability event)
# ---------------------------------------------------------------------------
#
# Distinct from :data:`REASONING_DRIFT_MAX_REASONING_CHARS` (the prompt-time
# truncation that bounds what we *send* to the judge). These bounds apply to
# the ``ReasoningJudgeInvoked`` event we emit on every judge invocation so a
# very long reasoning block or a chatty judge response cannot blow up event
# sinks (in-memory lists, SQLite rows, gRPC message size caps).
REASONING_JUDGE_MAX_REASONING_INPUT_CHARS: int = 4096
REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS: int = 2048
_TRUNCATE_SUFFIX: str = " … [truncated]"


def truncate_for_observability(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars, appending ``"… [truncated]"`` when cut.

    Shared by the observability emission path on every judge call so both
    the reasoning input and the raw response use the same truncation
    convention. Callers that need a different limit pass it explicitly.
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATE_SUFFIX


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
    sink: Any = None,
    run_id: str = "",
    session_id: str = "",
    sequence_fn: Callable[[], int] | None = None,
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
    sink:
        Optional :class:`goldfive.protocols.EventSink` to notify on every
        judge invocation, regardless of verdict. When provided, emits a
        ``ReasoningJudgeInvoked`` proto event carrying the truncated
        reasoning input, truncated raw judge response, elapsed-ms
        duration, and parsed verdict. When ``None`` the judge stays
        sink-less and existing callers see no behavioural change. Sink
        emit failures are absorbed and logged so a broken observability
        sink cannot break the run. See goldfive judge-observability
        event.
    run_id / session_id / sequence_fn:
        Stamped onto the emitted ``ReasoningJudgeInvoked`` envelope when
        ``sink`` is provided. ``sequence_fn`` is called at most once per
        invocation to get the next per-run sequence number; defaults to
        ``0`` when not supplied (sinks that need gap-free sequencing
        should pass the session's ``next_sequence``).
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
    started = time.monotonic()
    call_failed = False
    # Wrap the judge call in the shared LLM span helper so harmonograf
    # renders it as a span on the goldfive lane alongside every other
    # goldfive-internal LLM call. Redundant with the
    # ``ReasoningJudgeInvoked`` event (same timing) but matches the
    # pattern every other goldfive-internal LLM call uses — frontends
    # prefer the event for the verdict detail and the span for Gantt
    # rendering. See goldfive internal-llm-spans.
    from goldfive._llm_span import goldfive_llm_span

    span_sinks = [sink] if sink is not None else []
    try:
        async with goldfive_llm_span(
            sinks=span_sinks,
            name="judge_reasoning",
            model=model,
            session_id=session_id,
            run_id=run_id,
            task_id=current_task_id,
            sequence_fn=sequence_fn,
        ):
            raw = await call_llm(system, user, model)
    except Exception as exc:  # noqa: BLE001 - never break the run
        log.warning(
            "classify_reasoning_drift: call_llm raised %s; no drift emitted",
            exc,
        )
        raw = f"<call_llm raised: {exc!r}>"
        call_failed = True
    elapsed_ms = int((time.monotonic() - started) * 1000)
    raw_str = raw if isinstance(raw, str) else ""
    if not call_failed:
        log.debug(
            "classify_reasoning_drift: raw response (%d chars): %s",
            len(raw_str),
            raw_str[:500],
        )
    parsed = None if call_failed else _parse_response(raw)
    on_task_parsed: bool | None = None
    severity_str = ""
    reason = ""
    drift: DriftEvent | None = None
    if parsed is None:
        if not call_failed:
            log.debug(
                "classify_reasoning_drift: response was not JSON (raw=%r); "
                "no drift emitted",
                raw_str[:200],
            )
    else:
        on_task_raw = parsed.get("on_task")
        if not isinstance(on_task_raw, bool):
            log.debug(
                "classify_reasoning_drift: parsed=%r lacks boolean 'on_task' "
                "key; no drift emitted",
                parsed,
            )
        else:
            on_task_parsed = on_task_raw
            reason = str(parsed.get("reason", "") or "").strip()
            if on_task_raw:
                log.debug(
                    "classify_reasoning_drift: judge says on-track (reason=%r)",
                    reason,
                )
            else:
                severity_enum = _severity_from_verdict(parsed.get("severity"))
                severity_str = severity_enum.value.lower()
                log.info(
                    "classify_reasoning_drift: drift detected (severity=%s, "
                    "reason=%r); emitting OFF_TOPIC event",
                    severity_enum.value,
                    reason,
                )
                detail = (
                    f"reasoning drift: {reason}"
                    if reason
                    else "reasoning drift detected (judge returned no reason)"
                )
                drift = DriftEvent(
                    kind=DriftKind.OFF_TOPIC,
                    severity=severity_enum,
                    detail=detail,
                    current_task_id=current_task_id,
                    current_agent_id=current_agent_id,
                    raw=reasoning,
                    trigger_input=truncate_for_observability(
                        reasoning, REASONING_JUDGE_MAX_REASONING_INPUT_CHARS
                    ),
                )
    # Emit ReasoningJudgeInvoked on every invocation, regardless of
    # verdict. Done after the drift decision so the event carries the
    # parsed outcome but independent of it — on-task, off-task, and
    # plumbing-failure paths all produce an observability event.
    if sink is not None:
        await _emit_judge_invoked(
            sink=sink,
            run_id=run_id,
            session_id=session_id,
            sequence_fn=sequence_fn,
            current_task_id=current_task_id,
            current_agent_id=current_agent_id,
            model=model,
            elapsed_ms=elapsed_ms,
            reasoning_input=reasoning,
            raw_response=raw_str,
            on_task=bool(on_task_parsed) if on_task_parsed is not None else True
            if drift is None
            else False,
            severity=severity_str,
            reason=reason,
        )
    return drift


async def _emit_judge_invoked(
    *,
    sink: Any,
    run_id: str,
    session_id: str,
    sequence_fn: Callable[[], int] | None,
    current_task_id: str,
    current_agent_id: str,
    model: str,
    elapsed_ms: int,
    reasoning_input: str,
    raw_response: str,
    on_task: bool,
    severity: str,
    reason: str,
) -> None:
    """Build and emit a ``ReasoningJudgeInvoked`` envelope onto ``sink``.

    Broken sinks must not break the run: any exception is caught and
    logged at WARNING. Proto-import failures are handled the same way
    so a partially-regenerated tree (``make proto`` not re-run) does
    not crash the judge path.
    """
    try:
        from goldfive.events import new_event
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "classify_reasoning_drift: proto import failed (%s); "
            "skipping ReasoningJudgeInvoked emission",
            exc,
        )
        return
    try:
        sequence = sequence_fn() if sequence_fn is not None else 0
        evt = new_event(run_id, sequence, session_id=session_id)
        payload = evt.reasoning_judge_invoked
        payload.run_id = run_id
        payload.task_id = current_task_id
        payload.subject_agent_id = current_agent_id
        payload.model = model
        payload.elapsed_ms = int(elapsed_ms)
        payload.reasoning_input = truncate_for_observability(
            reasoning_input, REASONING_JUDGE_MAX_REASONING_INPUT_CHARS
        )
        payload.raw_response = truncate_for_observability(
            raw_response, REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS
        )
        payload.on_task = on_task
        payload.severity = severity
        payload.reason = reason
        await sink.emit(evt)
    except Exception as exc:  # noqa: BLE001 - observability must never break
        log.warning(
            "classify_reasoning_drift: sink.emit raised %s; "
            "ReasoningJudgeInvoked dropped",
            exc,
        )
