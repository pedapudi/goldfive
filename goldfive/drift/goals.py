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

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.drift.registry import (
    DetectorConfig,
    truncate_for_observability,
)
from goldfive.drift.registry import (
    format_goals_block as _format_goals,
)
from goldfive.drift.registry import (
    parse_json_response as _parse_response,
)
from goldfive.drift.registry import (
    register as _register,
)
from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "classify_goal_drift",
    "GOAL_DRIFT_SYSTEM_PROMPT",
    "GOAL_DRIFT_USER_PROMPT_TEMPLATE",
    "GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE",
    "GOAL_DRIFT_CHECK_INTERVAL",
    "GOAL_DRIFT_IDLE_SECONDS",
    "GOAL_DRIFT_MAX_OUTPUT_TOKENS",
]


# Default check cadences. One LLM call per check, so defaults are
# deliberately low-frequency; operators who want tighter monitoring
# can shorten them.
#
# ``GOAL_DRIFT_CHECK_INTERVAL`` documents the turn-based default; the
# LIVE runtime knob is :attr:`goldfive.config.GoalDriftConfig.check_interval`
# (which ``DefaultSteerer`` reads — this constant is a public re-export
# kept for back-compat and is not consulted at runtime).
#
# ``GOAL_DRIFT_IDLE_SECONDS`` is the idle-based scheduling threshold:
# when the wall-clock stall watchdog is enabled
# (``SteeringConfig.stall_watchdog_enabled``) and the session's
# liveness watermark has been silent this long, the watchdog triggers
# :meth:`~goldfive.drift_observer.DriftObserver.maybe_run_goal_drift_check`
# once per idle episode. Read live from this module attribute on every
# watchdog poll, so optimization-manifest ``setattr`` mutations take
# effect on a running watchdog.
GOAL_DRIFT_CHECK_INTERVAL: int = 5
GOAL_DRIFT_IDLE_SECONDS: int = 300

# Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up).
# The judge returns a small JSON verdict ({"progressing": bool, "reason":
# "..."}); 16384 covers Qwen 3.5 thinking-model preludes (think +
# answer share the same ceiling) without permitting unbounded essays.
# Empirical: v16 on Qwen 35B exhausted a 2048 budget inside ``<think>``
# and returned ``raw=''``, so no drift fired — see ``call_llm_budget``
# docstring for sizing rationale.
GOAL_DRIFT_MAX_OUTPUT_TOKENS: int = 16384


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
    '{{"progressing": false, "reason": "one-sentence explanation", '
    '"note_to_agent": "one or two sentences addressed to the agent '
    "itself: state only what you observed and how it relates to the "
    "goals. Neutral and factual — no commands, no instructions about "
    "which task, tool, or agent to use next, and no fault language. "
    "If your confidence is low, phrase the note as a question (e.g. "
    "'Does the current approach still serve the goal of X?')\"}}\n\n"
    "Progressing = agents are doing work that plausibly contributes to "
    "the goal.\n"
    "Not progressing = agents are looping, refusing, off-topic, or "
    "otherwise not advancing."
)


# AGENCY-PRESERVATION.md Stage 3 PR 11(a) — the GRADUATED goal-drift
# prompt. Used only in ledger plan mode (``graduated=True``). It asks for
# a three-state verdict so WARNING can fire BEFORE CRITICAL: an
# ``uncertain`` band lets goldfive surface "this may be drifting" as a
# proportional advisory note, reserving CRITICAL for clearly off-track
# trajectories. The plan rendered here is the LEDGER (goal-anchored
# OUTCOME deliverables + the descriptively-grown DISCOVERED trajectory),
# which is the agent's own observed intent — a self-consistent,
# adaptive reference rather than a forecast the agent is graded against.
GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE: str = (
    "You are assessing whether an autonomous agent tree is making "
    "progress toward a stated goal.\n\n"
    "GOALS (what the user wants — the primary reference):\n{goals_block}\n\n"
    "LEDGER (OUTCOME = deliverables that define success; DISCOVERED = "
    "what the agent has actually done so far; FORECAST = a pre-planned "
    "step, a prediction rather than an observation):\n{tasks_block}\n\n"
    "RECENT AGENT ACTIVITY (most recent {activity_count} invocations, "
    "newest last):\n{activity_block}\n\n"
    "Decide how the recent activity relates to the GOALS, given the "
    "agent's own observed trajectory in the LEDGER. Answer STRICTLY in "
    "one of these three JSON shapes:\n"
    '{{"progressing": true}}\n'
    "OR\n"
    '{{"progressing": false, "band": "uncertain", "reason": '
    '"one-sentence explanation", "note_to_agent": "..."}}\n'
    "OR\n"
    '{{"progressing": false, "band": "off_track", "reason": '
    '"one-sentence explanation", "note_to_agent": "..."}}\n\n'
    "progressing=true — activity plausibly advances the goals; no concern.\n"
    'band="uncertain" — the connection to the goals is unclear or the '
    "trajectory may be starting to drift, but it is not yet clearly "
    "wrong. This is an EARLY, proportional signal.\n"
    'band="off_track" — activity is clearly not advancing the goals '
    "(looping, refusing, pursuing a different objective).\n\n"
    "``note_to_agent`` is one or two sentences addressed to the agent "
    "itself: state only what you observed and how it relates to the "
    "goals. Neutral and factual — no commands, no instructions about "
    "which task, tool, or agent to use next, and no fault language. For "
    'the "uncertain" band especially, prefer a QUESTION (e.g. "Does the '
    'current approach still serve the goal of X?") over a statement.'
)


# Graduated-mode band → severity map. ``uncertain`` fires WARNING (the
# early, proportional signal); ``off_track`` — and any unrecognised /
# missing band on a non-progressing verdict — fires CRITICAL, matching
# the binary judge's pre-PR-11 behaviour so a graduated judge that omits
# the band degrades safely to the legacy severity.
_GRADUATED_BAND_SEVERITY: dict[str, DriftSeverity] = {
    "uncertain": DriftSeverity.WARNING,
    "off_track": DriftSeverity.CRITICAL,
}


# Liberal JSON extraction + goals rendering live in
# :mod:`goldfive.drift.registry` so the goal-drift judge and the
# per-reasoning judge share one implementation. The aliased imports at
# the top of this module preserve the historical private names (so
# external test suites mocking ``goals._parse_response`` continue to
# work) without re-declaring the function bodies here.


def _format_tasks(plan: Plan | Any | None, *, annotate_kind: bool = False) -> str:
    """Render the plan's tasks for the judge prompt.

    When ``annotate_kind`` (ledger / graduated mode), each task is
    tagged with its ``kind`` (OUTCOME / DISCOVERED / FORECAST) so the
    judge reads the plan as a LEDGER — deliverables vs the agent's
    observed trajectory — rather than a flat task list. Forecast mode
    (``annotate_kind=False``, the default) renders exactly as before.
    """
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
        kind_tag = ""
        if annotate_kind:
            kind = getattr(t, "kind", "")
            kind_str = str(getattr(kind, "value", kind) or "").upper()
            if kind_str:
                kind_tag = f"{kind_str} "
        if tid:
            lines.append(f"{i}. {kind_tag}[{tid}] {title} ({status_str})")
        else:
            lines.append(f"{i}. {kind_tag}{title} ({status_str})")
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
    graduated: bool = False,
    sinks: list[Any] | None = None,
    run_id: str = "",
    session_id: str = "",
    sequence_fn: Callable[[], int] | None = None,
    session: Any | None = None,
) -> DriftEvent | None:
    """Ask an LLM-judge whether recent activity is progressing the goals.

    Returns a :class:`DriftEvent` of kind
    :data:`~goldfive.types.DriftKind.GOAL_DRIFT` when the judge returns
    ``{"progressing": false, ...}``. The severity depends on the mode:

    * ``graduated=False`` (forecast mode, the default) — every
      non-progressing verdict is
      :data:`~goldfive.types.DriftSeverity.CRITICAL` (the pre-PR-11
      binary behaviour, byte-identical).
    * ``graduated=True`` (ledger plan mode; AGENCY-PRESERVATION.md
      PR 11(a)) — the judge is asked for a three-state verdict via
      :data:`GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE` and a
      non-progressing verdict carries a ``band``:
      ``"uncertain"`` fires :data:`~goldfive.types.DriftSeverity.WARNING`
      (the early, proportional signal) and ``"off_track"`` fires
      CRITICAL. A missing or unrecognised band degrades to CRITICAL so
      the severity never silently softens below the legacy default.

    Returns ``None`` in every other case:

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
        judge style can pass their own; an explicit
        ``user_prompt_template`` wins over the ``graduated`` template
        selection. The defaults match the shapes pinned in
        :data:`GOAL_DRIFT_USER_PROMPT_TEMPLATE` /
        :data:`GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE`.
    graduated:
        ``True`` in ledger plan mode: selects the graduated (three-band)
        prompt, renders the plan with task-kind annotations (OUTCOME /
        DISCOVERED / FORECAST), and maps the verdict ``band`` to
        severity as described above. ``False`` (the default) keeps the
        binary CRITICAL-only forecast behaviour.
    """
    system = system_prompt or GOAL_DRIFT_SYSTEM_PROMPT
    # AGENCY-PRESERVATION.md PR 11(a) — in ledger mode (``graduated``) ask
    # for the three-band verdict so WARNING can fire before CRITICAL. An
    # explicit ``user_prompt_template`` override always wins; otherwise
    # the band selection is forecast (binary) vs ledger (graduated).
    if user_prompt_template is not None:
        template = user_prompt_template
    elif graduated:
        template = GOAL_DRIFT_GRADUATED_USER_PROMPT_TEMPLATE
    else:
        template = GOAL_DRIFT_USER_PROMPT_TEMPLATE
    # goldfive#245 — capture the plan-revision the judge is observing
    # BEFORE we render the prompt or await the LLM. The post-LLM
    # re-read below uses this snapshot to detect when the plan moved
    # under the judge during the round-trip; the dispatch-time gate in
    # :meth:`DefaultSteerer._handle_drift` uses it to drop verdicts
    # against revisions the system has already moved past.
    observed_revision_index = int(getattr(plan, "revision_index", 0) or 0)
    # Snapshot the (id -> status) set of the plan as observed at this
    # call-time. Used by the post-LLM re-read fallback (no specific
    # task id in the verdict reason) to decide whether ANY task
    # transitioned during the LLM round-trip.
    pre_call_task_status: dict[str, str] = {}
    if plan is not None:
        for t in getattr(plan, "tasks", None) or []:
            tid = str(getattr(t, "id", "") or "")
            if not tid:
                continue
            status = getattr(t, "status", "")
            pre_call_task_status[tid] = str(getattr(status, "value", status) or "")
    goals_block = _format_goals(goals)
    tasks_block = _format_tasks(plan, annotate_kind=graduated)
    activity_block, activity_count = _format_activity(observed_actions)
    user = template.format(
        goals_block=goals_block,
        tasks_block=tasks_block,
        activity_block=activity_block,
        activity_count=activity_count,
    )
    # The judge is trajectory-level: ``target_agent_id`` /
    # ``target_task_id`` deliberately stay empty so harmonograf renders
    # the span on the goldfive lane without attributing to any one
    # agent. ``input_preview`` carries the activity block the judge saw
    # so operators can answer "why did goldfive think this was off-
    # track?" from the Gantt.
    parsed: dict[str, Any] | None = None
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
            input_preview=activity_block,
        ) as span:
            # Bound the dispatch — see ``GOAL_DRIFT_MAX_OUTPUT_TOKENS``.
            # Also disable thinking (goldfive#271 follow-up to #311):
            # this is meta-cognition asking a small JSON progress
            # question, not deep reasoning. See `call_llm_thinking_disabled`
            # docstring for why we don't want to share the 16k cap with
            # ``<think>`` reasoning here.
            from goldfive._llm import (
                call_llm_budget,
                call_llm_thinking_disabled,
                llm_call_diagnostics,
            )

            with (
                call_llm_budget(GOAL_DRIFT_MAX_OUTPUT_TOKENS),
                call_llm_thinking_disabled(),
                llm_call_diagnostics() as llm_diag,
            ):
                raw = await call_llm(system, user, model)
            # Parse inside the with-block so span.decision_summary /
            # output_preview see the verdict before the End emission.
            parsed = _parse_response(raw)
            if parsed is None:
                # Distinguish "model returned all thinking, no answer"
                # from "model returned garbage". The default ADK /
                # OpenAI builders record part counts into the per-call
                # diagnostics object; surface them when the raw text is
                # empty.
                _thought_n = llm_diag.thought_count
                _raw_str = raw if isinstance(raw, str) else ""
                if not _raw_str.strip() and _thought_n > 0:
                    span.output_preview = (
                        f"empty answer ({_thought_n} thought part(s); "
                        f"the model spent its budget thinking and emitted "
                        f"no JSON)"
                    )
                else:
                    span.output_preview = "(unparseable verdict)"
                span.decision_summary = "judged trajectory: unparseable verdict (no drift emitted)"
            else:
                progressing_inline = parsed.get("progressing")
                reason_inline = str(parsed.get("reason", "") or "").strip()
                if isinstance(progressing_inline, bool):
                    span.output_preview = (
                        f"progressing={progressing_inline}, reason={reason_inline or '(none)'}"
                    )
                    progressing_str = "on-track" if progressing_inline else "off-track"
                    span.decision_summary = f"judged trajectory: {progressing_str}" + (
                        f" ({reason_inline})" if reason_inline else ""
                    )
                else:
                    span.output_preview = f"missing boolean 'progressing'; raw={parsed!r}"
                    span.decision_summary = "judged trajectory: verdict missing 'progressing'"
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
    # ``parsed`` is populated inside the span block above; re-binding
    # here is intentional so the post-with path doesn't re-parse the
    # same string.
    if parsed is None:
        log.debug(
            "classify_goal_drift: response was not JSON (raw=%r); no drift emitted",
            raw_str[:200],
        )
        return None
    progressing = parsed.get("progressing")
    if not isinstance(progressing, bool):
        log.debug(
            "classify_goal_drift: parsed=%r lacks boolean 'progressing' key; no drift emitted",
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
    # goldfive#245 — post-LLM re-read. While the judge was running the
    # reconciler may have transitioned tasks (the brussels/tomato
    # false-positive class: judge complained "drafting still pending"
    # against a snapshot where draft was already DONE by the time the
    # verdict arrived). Re-read ``session.plan`` here and drop the
    # verdict if the plan-state the judge complained about no longer
    # matches reality.
    #
    # Two specificity tiers:
    #
    # 1. **Targeted task** — when the judge's reason names a specific
    #    plan task by id (e.g. mentions ``[t-draft]``), check that
    #    task's status. If it has transitioned out of PENDING the
    #    verdict is moot.
    # 2. **Generic narrative** — when the reason talks generally about
    #    "drafting" / "research" without an id, we cannot disambiguate.
    #    Fall back to comparing the plan's ``revision_index`` and the
    #    ``(task.id, task.status)`` set: if either changed materially
    #    since the snapshot, the judge's view is stale and we drop.
    #
    # ``session`` is optional so legacy / out-of-tree callers that
    # don't thread it keep their pre-#245 behaviour. The dispatch-time
    # gate in :meth:`DefaultSteerer._handle_drift` is the second line
    # of defence — it picks up stale verdicts even when the detector
    # didn't run this re-read.
    if session is not None:
        live_plan = getattr(session, "plan", None)
        if live_plan is not None:
            live_tasks_by_id: dict[str, Any] = {
                str(getattr(t, "id", "") or ""): t
                for t in (getattr(live_plan, "tasks", None) or [])
                if getattr(t, "id", None)
            }
            # Tier 1: specific task id mentioned in the reason text.
            targeted_id = ""
            for tid in live_tasks_by_id:
                if tid and tid in reason:
                    targeted_id = tid
                    break
            if targeted_id:
                live_t = live_tasks_by_id[targeted_id]
                live_status = getattr(live_t, "status", "")
                live_status_str = str(
                    getattr(live_status, "value", live_status) or ""
                )
                pre_status_str = pre_call_task_status.get(targeted_id, "")
                if live_status_str and live_status_str != pre_status_str:
                    log.info(
                        "classify_goal_drift: post-LLM re-read — task %r "
                        "transitioned %s -> %s during judge call; "
                        "verdict dropped",
                        targeted_id,
                        pre_status_str or "(unknown)",
                        live_status_str,
                    )
                    return None
            else:
                # Tier 2: generic verdict — compare revision_index +
                # (id, status) set against the pre-call snapshot.
                live_revision = int(
                    getattr(live_plan, "revision_index", 0) or 0
                )
                live_status_set: dict[str, str] = {}
                for t in getattr(live_plan, "tasks", None) or []:
                    tid = str(getattr(t, "id", "") or "")
                    if not tid:
                        continue
                    s = getattr(t, "status", "")
                    live_status_set[tid] = str(getattr(s, "value", s) or "")
                if live_revision != observed_revision_index or (
                    live_status_set != pre_call_task_status
                ):
                    log.info(
                        "classify_goal_drift: post-LLM re-read — plan "
                        "moved during judge call (revision %d -> %d, "
                        "status delta=%s); verdict dropped",
                        observed_revision_index,
                        live_revision,
                        sorted(
                            set(live_status_set.items())
                            ^ set(pre_call_task_status.items())
                        ),
                    )
                    return None
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
    # AGENCY-PRESERVATION.md PR 4 — the judge authors the agent-facing
    # observation in the same call that produced the verdict. Old-style
    # responses (key absent / non-string) degrade to "" and the
    # observer-note composer falls back to ``detail``.
    note_raw = parsed.get("note_to_agent", "")
    note_to_agent = note_raw.strip() if isinstance(note_raw, str) else ""
    # AGENCY-PRESERVATION.md PR 11(a) — graduated severity. In ledger mode
    # the judge may band a non-progressing verdict as ``uncertain`` (early,
    # proportional → WARNING) vs ``off_track`` (clear drift → CRITICAL).
    # Forecast mode (``graduated=False``) keeps the pre-PR-11 behaviour
    # exactly: every non-progressing verdict is CRITICAL. A graduated
    # verdict missing / with an unrecognised band degrades to CRITICAL so
    # the severity never silently softens below the legacy default.
    severity = DriftSeverity.CRITICAL
    if graduated:
        band = str(parsed.get("band", "") or "").strip().lower()
        severity = _GRADUATED_BAND_SEVERITY.get(band, DriftSeverity.CRITICAL)
    return DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=severity,
        detail=detail,
        current_task_id=current_task_id,
        current_agent_id=current_agent_id,
        trigger_input=trigger_input,
        observed_revision_index=observed_revision_index,
        note_to_agent=note_to_agent,
    )


# The trigger-input cap is goal-drift-specific (the activity block is
# wider than a reasoning-judge prompt), so the *limit* lives here but
# the truncation itself uses the shared registry helper so the
# " … [truncated]" suffix stays consistent.
_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS: int = 2048


def _truncate_trigger_input(text: str) -> str:
    """Cap the observability ``trigger_input`` at a bounded length."""
    return truncate_for_observability(text, _GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS)


# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------
#
# Goal-drift is one of the two LLM-as-a-judge detectors. The config
# pins the per-callsite output cap (16384, mirrors
# :data:`GOAL_DRIFT_MAX_OUTPUT_TOKENS`) and the trigger-input cap
# (2048, mirrors :data:`_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS`). The
# judge enters :func:`call_llm_thinking_disabled` for every dispatch.


_GOAL_DRIFT_CONFIG: DetectorConfig = DetectorConfig(
    uses_llm=True,
    max_input_chars=_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS,
    max_output_tokens=GOAL_DRIFT_MAX_OUTPUT_TOKENS,
    disable_thinking=True,
)


_register(
    DriftKind.GOAL_DRIFT,
    classify_goal_drift,
    _GOAL_DRIFT_CONFIG,
    is_async=True,
)
