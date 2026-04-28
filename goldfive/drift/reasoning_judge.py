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

Phase 1 of goldfive#271 adds an *attribution* signal alongside the
on-task verdict: :func:`classify_reasoning_drift_with_focus` returns
``focused_task_id`` + ``focus_confidence`` extracted from the same
LLM call. Same prompt, same cost; the prompt is extended to ask the
judge to name the plan task the reasoning is actually working on. The
caller (typically :class:`~goldfive.steerer.DefaultSteerer`) writes
the binding onto :class:`~goldfive.orchestration_store.OrchestrationStore`
when confidence is above a threshold; the pin-resolution ladder reads
it back as a real signal.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan, Task

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "PLAN_TASKS_SUMMARY_MAX_CHARS",
    "REASONING_JUDGE_MAX_OUTPUT_TOKENS",
    "REASONING_JUDGE_MAX_REASONING_INPUT_CHARS",
    "REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS",
    "REASONING_DRIFT_MAX_REASONING_CHARS",
    "REASONING_DRIFT_SYSTEM_PROMPT",
    "REASONING_DRIFT_USER_PROMPT_TEMPLATE",
    "ReasoningJudgeVerdict",
    "classify_reasoning_drift",
    "classify_reasoning_drift_with_focus",
    "format_plan_tasks_summary",
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

# Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up).
# The judge returns a small JSON verdict ({"on_task": bool, "reason":
# "...", "severity": "...", ...}); 16384 covers Qwen 3.5 thinking-model
# preludes (think + answer share the same ceiling) without permitting
# unbounded essays. Empirical: v16 on Qwen 35B exhausted a 2048 budget
# inside ``<think>`` and returned ``raw=''``, so no drift fired and the
# cascade never started — see ``call_llm_budget`` docstring for sizing
# rationale.
REASONING_JUDGE_MAX_OUTPUT_TOKENS: int = 16384


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
    "PLAN TASKS (id -> title):\n{plan_tasks_summary}\n\n"
    "CURRENTLY BOUND TASK:\n{task_block}\n\n"
    "GOALS:\n{goals_block}\n\n"
    "REASONING (the agent's most recent chain-of-thought block):\n"
    "{reasoning_block}\n\n"
    "Decide TWO things:\n"
    "1. Does the reasoning stay focused on the explicit task and "
    "goals above? (the on-task verdict)\n"
    "2. Which task in the PLAN TASKS list above is the reasoning "
    "actually working on right now? (the attribution verdict — answer "
    "with a task id from the list, or '' when the reasoning is not "
    "working on any plan task / is off-plan)\n\n"
    "Reply with a single JSON object and nothing else, in this shape:\n"
    "{{\n"
    '  "on_task": true|false,\n'
    '  "severity": "info"|"warning"|"critical",\n'
    '  "reason": "one-sentence explanation",\n'
    '  "focused_task_id": "<id from PLAN TASKS, or \'\' if off-plan>",\n'
    '  "focus_confidence": 0.0-1.0,\n'
    '  "stated_intent": "one-sentence summary of what the agent says it '
    'is doing"\n'
    "}}\n\n"
    "on_task=true = reasoning is working toward the bound task / goals "
    "(clarifying sub-steps, exploring tradeoffs, working through a "
    "calculation all count as on_task). When on_task=true, severity / "
    "reason may be omitted or empty.\n"
    "on_task=false = reasoning has drifted to an unrelated topic, is "
    "proposing to abandon the task, or is otherwise off-course.\n"
    "Severity guidance when on_task=false:\n"
    "- info = mild tangent that may self-correct next turn.\n"
    "- warning = clear off-topic content that deserves a nudge.\n"
    "- critical = proposing to abandon or replace the task/goal.\n\n"
    "focused_task_id MUST be the literal id of one of the listed plan "
    "tasks, or an empty string when the reasoning is not working on "
    "any plan task. focus_confidence is your subjective certainty in "
    "the attribution: 1.0 when the reasoning explicitly names the "
    "task, 0.0 when you are guessing."
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


# Hard cap on the plan-tasks-summary section of the prompt. Phase-1
# brief calls for truncation when the plan grows large enough that the
# rendered list would dominate the judge's context budget. 2000 chars
# is the agreed cap (~500 tokens) — comfortably below the prompt's
# overall budget while leaving room for ~50 tasks at typical title
# length.
PLAN_TASKS_SUMMARY_MAX_CHARS: int = 2000


def format_plan_tasks_summary(
    plan: Plan | None,
    *,
    max_chars: int = PLAN_TASKS_SUMMARY_MAX_CHARS,
) -> str:
    """Render ``plan.tasks`` as a one-per-line ``id -> title`` summary.

    Empty / None plan renders as ``"(no plan tasks)"`` so the prompt
    template renders cleanly when the plan hasn't been built yet.

    Truncation: when the rendered text exceeds ``max_chars`` we drop
    suffix lines and append a ``"... [N more tasks]"`` marker so the
    judge knows the list is incomplete. Truncation prefers to keep the
    head of the list (most recently planned tasks tend to be most
    relevant for a "what is the agent working on right now?" judgement
    — they are at the front of typical refines).
    """
    if plan is None or not getattr(plan, "tasks", None):
        return "(no plan tasks)"
    lines: list[str] = []
    rendered_chars = 0
    truncated = 0
    tasks = list(plan.tasks)
    for i, task in enumerate(tasks):
        tid = str(getattr(task, "id", "") or "")
        title = str(getattr(task, "title", "") or "(untitled)")
        line = f"- {tid} -> {title}" if tid else f"- (no id) -> {title}"
        if rendered_chars + len(line) + 1 > max_chars and lines:
            truncated = len(tasks) - i
            break
        lines.append(line)
        rendered_chars += len(line) + 1
    if truncated > 0:
        lines.append(f"... [{truncated} more task(s) elided]")
    if not lines:
        return "(no plan tasks)"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extended verdict (Phase 1 of goldfive#271)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReasoningJudgeVerdict:
    """Extended verdict returned by :func:`classify_reasoning_drift_with_focus`.

    Carries both the existing on-task ``DriftEvent`` (or ``None`` when
    the judge was on-task / failed quietly) and the new attribution
    fields the Phase-1 prompt extension extracts.

    ``drift`` is the same value the legacy
    :func:`classify_reasoning_drift` returns, so callers that only
    care about the drift signal can do
    ``verdict.drift if verdict else None``.

    ``focused_task_id`` is the plan-task id the judge identified as
    "what the agent is actually working on right now". Empty string
    when the judge declined to attribute (off-plan reasoning, or
    response missing the field). The pin-resolution ladder consumes
    this; the ``focus_confidence`` lets the consumer gate
    low-certainty bindings.

    ``focus_confidence`` is the judge's subjective certainty. Clamped
    to ``[0.0, 1.0]``. Defaults to ``0.0`` when the field is missing
    or malformed — the caller's threshold then naturally rejects it.

    ``stated_intent`` is the judge's one-sentence summary of what the
    agent claims to be doing. Optional — surfaced for sinks /
    observability; not consumed by any current pin-resolution logic.
    """

    drift: DriftEvent | None
    focused_task_id: str = ""
    focus_confidence: float = 0.0
    stated_intent: str = ""


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
    plan: Plan | None = None,
) -> DriftEvent | None:
    """Ask an LLM-judge whether ``reasoning`` is on-task.

    Back-compat wrapper around
    :func:`classify_reasoning_drift_with_focus`: returns just the
    drift component of the extended verdict so existing callers
    (test suite, third-party importers) keep their ``DriftEvent | None``
    return shape.

    See :func:`classify_reasoning_drift_with_focus` for the full
    parameter docs and the new attribution fields. Phase 1 of
    goldfive#271 added an optional ``plan`` keyword that the extended
    function uses to render the plan-tasks attribution prompt; legacy
    callers can omit it and the prompt renders ``"(no plan tasks)"``
    for that section.
    """
    verdict = await classify_reasoning_drift_with_focus(
        reasoning=reasoning,
        task=task,
        goals=goals,
        model=model,
        call_llm=call_llm,
        current_task_id=current_task_id,
        current_agent_id=current_agent_id,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        sink=sink,
        run_id=run_id,
        session_id=session_id,
        sequence_fn=sequence_fn,
        plan=plan,
    )
    return verdict.drift


async def classify_reasoning_drift_with_focus(
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
    plan: Plan | None = None,
) -> ReasoningJudgeVerdict:
    """Ask an LLM-judge whether ``reasoning`` is on-task AND which task it works on.

    Phase 1 of goldfive#271 — the extended judge call. Same LLM
    request, same cost as the legacy
    :func:`classify_reasoning_drift`; the prompt template is extended
    to ask the judge to attribute the reasoning to a specific plan
    task, returning ``focused_task_id`` + ``focus_confidence``
    alongside the existing ``on_task`` / ``severity`` / ``reason``
    fields.

    Returns a :class:`ReasoningJudgeVerdict`:

    * ``verdict.drift`` — same as the legacy function. A
      :class:`DriftEvent` of kind
      :data:`~goldfive.types.DriftKind.OFF_TOPIC` when the judge
      returns ``{"on_task": false, ...}``; ``None`` for on-task
      verdicts and every quiet-failure path (malformed JSON, missing
      ``on_task``, ``call_llm`` raised, empty reasoning).
    * ``verdict.focused_task_id`` — the judge's plan-task attribution.
      Empty string when the judge declined to attribute, when the
      response was malformed, or when the call failed.
    * ``verdict.focus_confidence`` — the judge's subjective certainty,
      clamped to ``[0.0, 1.0]``. ``0.0`` for every quiet-failure path.
    * ``verdict.stated_intent`` — optional one-sentence summary the
      judge produced. Empty when missing or after a failure.

    The "quiet on failure" contract matches :func:`classify_goal_drift`
    (goldfive#143). A flaky judge must not spam operator UIs with
    false-positive OFF_TOPIC alarms — the extended fields default to
    "no signal" rather than raising.

    Parameters
    ----------
    reasoning:
        The reasoning / chain-of-thought block to classify. Truncated
        to :data:`REASONING_DRIFT_MAX_REASONING_CHARS` before prompting
        so pathologically long blocks cannot blow the context budget.
        Empty or whitespace-only ``reasoning`` is treated as nothing to
        classify -- returns an empty verdict without calling the LLM.
    task:
        The currently-bound :class:`Task` (typically from
        ``session.current_task_id``). May be ``None`` when no task is
        active; the judge then has only ``goals`` and ``plan`` to
        compare against.
    goals:
        The session's goals. Each entry should have ``id`` / ``summary``
        attributes or be a plain str. May be ``None`` / empty.
    plan:
        The session's plan. The judge needs the list of plan tasks to
        attribute the reasoning ("which task is the agent actually
        working on?") — when ``None``, the prompt renders an empty
        plan-tasks block and the judge will return an empty
        ``focused_task_id``.
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
        return ReasoningJudgeVerdict(drift=None)
    system = system_prompt or REASONING_DRIFT_SYSTEM_PROMPT
    template = user_prompt_template or REASONING_DRIFT_USER_PROMPT_TEMPLATE
    user = template.format(
        plan_tasks_summary=format_plan_tasks_summary(plan),
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
    # Stamp the reasoning block goldfive is judging onto the span's
    # ``input_preview`` so harmonograf can render "what did the judge
    # see?" inline on the Gantt without re-fetching the agent transcript.
    # Truncated by the helper.
    span_input_preview = reasoning if isinstance(reasoning, str) else ""

    on_task_parsed: bool | None = None
    severity_str = ""
    reason = ""
    drift: DriftEvent | None = None
    parsed: dict[str, Any] | None = None
    # Phase 1 — extended attribution fields. Default to "no signal" so
    # every quiet-failure path (call raises, malformed JSON, missing
    # field, malformed numeric confidence) yields an empty verdict the
    # caller's threshold naturally rejects.
    focused_task_id_parsed: str = ""
    focus_confidence_parsed: float = 0.0
    stated_intent_parsed: str = ""
    try:
        async with goldfive_llm_span(
            sinks=span_sinks,
            name="judge_reasoning",
            model=model,
            session_id=session_id,
            run_id=run_id,
            task_id=current_task_id,
            sequence_fn=sequence_fn,
            input_preview=span_input_preview,
            target_agent_id=current_agent_id,
            target_task_id=current_task_id,
        ) as span:
            # Bound the dispatch — see ``REASONING_JUDGE_MAX_OUTPUT_TOKENS``.
            # Also disable thinking (goldfive#271 follow-up to #311):
            # this is meta-cognition asking a small JSON question, not
            # deep reasoning. Letting the model burn the 16k budget on
            # ``<think>`` was the v16 / Qwen 35B failure mode — the cap
            # bump was the symptom-fix, this is the cause-fix.
            from goldfive._llm import call_llm_budget, call_llm_thinking_disabled

            with call_llm_budget(REASONING_JUDGE_MAX_OUTPUT_TOKENS), call_llm_thinking_disabled():
                raw = await call_llm(system, user, model)
            # Parse inside the with-block so we can stamp
            # decision-context onto the span before the End event fires
            # on exit. The heavier handling (log.info, DriftEvent
            # construction) still runs post-with so span emission stays
            # lean.
            raw_str_inline = raw if isinstance(raw, str) else ""
            parsed = _parse_response(raw)
            if parsed is not None:
                on_task_raw = parsed.get("on_task")
                if isinstance(on_task_raw, bool):
                    on_task_parsed = on_task_raw
                    reason = str(parsed.get("reason", "") or "").strip()
                    if not on_task_raw:
                        severity_str = _severity_from_verdict(parsed.get("severity")).value.lower()
                # Extended attribution fields — extracted regardless of
                # the on_task verdict. The judge can name a focused
                # task whether or not it considers the reasoning on the
                # currently-bound one (off-task reasoning still has a
                # focus — that's how the steerer learns the agent has
                # silently switched to a different plan task).
                focused_raw = parsed.get("focused_task_id", "")
                if isinstance(focused_raw, str):
                    focused_task_id_parsed = focused_raw.strip()
                conf_raw = parsed.get("focus_confidence", 0.0)
                try:
                    focus_confidence_parsed = float(conf_raw)
                except (TypeError, ValueError):
                    focus_confidence_parsed = 0.0
                # Clamp to [0.0, 1.0]; the prompt asks for 0.0-1.0 but
                # we don't trust the LLM not to drift outside.
                focus_confidence_parsed = max(0.0, min(1.0, focus_confidence_parsed))
                intent_raw = parsed.get("stated_intent", "")
                if isinstance(intent_raw, str):
                    stated_intent_parsed = intent_raw.strip()
            # Build the span's output / decision strings from the parsed
            # verdict so harmonograf can render "judged agent/task:
            # on-task" inline.
            if on_task_parsed is None:
                # Distinguish "model returned all thinking, no answer"
                # from "model returned garbage" (goldfive#271 follow-up
                # to #311). The default ADK / OpenAI builders stash the
                # part counts on the call_llm closure; when the answer
                # is empty AND we saw ``thought=True`` parts the
                # diagnostic should say so rather than show an
                # indistinguishable ``raw=''``.
                _thought_n = int(getattr(call_llm, "last_thought_count", 0) or 0)
                if not raw_str_inline.strip() and _thought_n > 0:
                    span.output_preview = (
                        f"empty answer ({_thought_n} thought part(s); "
                        f"the model spent its budget thinking and emitted "
                        f"no JSON)"
                    )
                else:
                    span.output_preview = f"unparseable verdict; raw={raw_str_inline[:200]!r}"
                span.decision_summary = (
                    f"reasoning-judge call on "
                    f"{current_agent_id or '(no-agent)'}"
                    f"/{current_task_id or '(no-task)'}: "
                    "unparseable verdict"
                )
            else:
                span.output_preview = (
                    f"on_task={on_task_parsed}, "
                    f"severity={severity_str or '(none)'}, "
                    f"reason={reason or '(none)'}"
                )
                if on_task_parsed:
                    verdict_str = "on-task"
                else:
                    verdict_str = (
                        f"off-task ({severity_str.upper()})" if severity_str else "off-task"
                    )
                span.decision_summary = (
                    f"judged {current_agent_id or '(no-agent)'}'s reasoning "
                    f"on {current_task_id or '(no-task)'}: {verdict_str}"
                )
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
    if parsed is None and not call_failed:
        log.debug(
            "classify_reasoning_drift: response was not JSON (raw=%r); no drift emitted",
            raw_str[:200],
        )
    elif parsed is not None:
        if on_task_parsed is None:
            log.debug(
                "classify_reasoning_drift: parsed=%r lacks boolean 'on_task' key; no drift emitted",
                parsed,
            )
        elif on_task_parsed:
            log.debug(
                "classify_reasoning_drift: judge says on-track (reason=%r)",
                reason,
            )
        else:
            severity_enum = _severity_from_verdict(parsed.get("severity"))
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
            on_task=bool(on_task_parsed)
            if on_task_parsed is not None
            else True
            if drift is None
            else False,
            severity=severity_str,
            reason=reason,
        )
    return ReasoningJudgeVerdict(
        drift=drift,
        focused_task_id=focused_task_id_parsed,
        focus_confidence=focus_confidence_parsed,
        stated_intent=stated_intent_parsed,
    )


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
            "classify_reasoning_drift: sink.emit raised %s; ReasoningJudgeInvoked dropped",
            exc,
        )
