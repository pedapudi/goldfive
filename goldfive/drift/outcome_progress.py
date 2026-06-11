"""Outcome-progress judge for ledger plan mode (AGENCY-PRESERVATION.md PR 11c).

In ledger plan mode the Plan is a ledger: goal-anchored OUTCOME tasks
(deliverables that define success) plus a descriptively-grown record of
DISCOVERED tasks (what the agent actually did). The wrapped agent never
reports on the OUTCOME tasks — it owns the means, not goldfive's
bookkeeping — so OUTCOME tasks reach a terminal status only when
goldfive's own judge decides they are met. That is what makes "run
completion = all outcomes terminal" decidable *without agent
cooperation* (design doc §2, §3 PR 11(c)).

This module is the judge. Like :mod:`goldfive.drift.goals` and
:mod:`goldfive.drift.reasoning_judge` it is framework-neutral: it takes
the data it needs by keyword and returns plain verdicts. It does NOT
mutate the plan, emit events, or import the steerer — the caller (the
drift observer's task-boundary / run-end hooks) applies the transitions.

Two pieces:

* :func:`evaluate_outcome_progress` — one cost-bounded LLM call that
  grades every non-terminal OUTCOME task against the GOALS, using the
  goldfive#447 full-output capture (``session.completed_outputs``) and
  the DISCOVERED trajectory as evidence. Quiet on every failure
  (malformed JSON, ``call_llm`` raised) — a flaky judge must never
  spuriously complete or fail a deliverable.
* :func:`plan_outcome_transitions` — a PURE function that turns verdicts
  into the set of task-status transitions and ``contributes_to`` stamps
  the caller should apply. ``met`` → COMPLETED at any cadence;
  ``unmet`` → FAILED only at run end (``run_ending=True``), because a
  deliverable that is not yet met mid-run is simply not done yet.
  ``contributes_to`` is stamped onto the DISCOVERED tasks the judge
  named as advancing each met OUTCOME, linking the trajectory lane to
  the deliverable it produced.

User-supplied goal predicates remain authoritative: the caller passes
``goal_predicates_met`` and this module refuses to COMPLETE any outcome
when a user predicate is explicitly unmet — the deterministic predicate
overrides the LLM's opinion (the existing
:func:`goldfive.results.evaluate_goal_predicates` run-success gate is
unchanged and still has the final say on ``Outcome.success``).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from goldfive.drift.registry import (
    format_goals_block as _format_goals,
)
from goldfive.drift.registry import (
    parse_json_response as _parse_response,
)
from goldfive.types import Goal, Plan, Task, TaskKind, TaskStatus

log = logging.getLogger(__name__)


__all__ = [
    "CallLLM",
    "OutcomeVerdict",
    "OutcomeTransition",
    "OUTCOME_ASSESSMENT_MET",
    "OUTCOME_ASSESSMENT_FAILED",
    "OUTCOME_ASSESSMENT_PENDING",
    "evaluate_outcome_progress",
    "plan_outcome_transitions",
    "OUTCOME_PROGRESS_SYSTEM_PROMPT",
    "OUTCOME_PROGRESS_USER_PROMPT_TEMPLATE",
    "OUTCOME_PROGRESS_MAX_OUTPUT_TOKENS",
    "OUTCOME_PROGRESS_EVIDENCE_MAX_CHARS",
]


CallLLM = Callable[[str, str, str], Awaitable[str]]

# Mirrors GOAL_DRIFT_MAX_OUTPUT_TOKENS — the verdict is a small JSON
# array, but the ceiling has to clear thinking-model preludes.
OUTCOME_PROGRESS_MAX_OUTPUT_TOKENS: int = 16384

# Per-task evidence cap so a chatty agent's full output cannot blow the
# prompt budget. The capture is the canonical gradeable artifact; the
# tail is the part the judge most needs (the final deliverable).
OUTCOME_PROGRESS_EVIDENCE_MAX_CHARS: int = 4000


OUTCOME_PROGRESS_SYSTEM_PROMPT: str = (
    "You are deciding whether each of a small set of DELIVERABLES has "
    "been produced, based on evidence of what an autonomous agent "
    "actually did. Reply with a single JSON object and nothing else."
)

OUTCOME_PROGRESS_USER_PROMPT_TEMPLATE: str = (
    "You are deciding which DELIVERABLES are DONE.\n\n"
    "GOALS (what the user wants):\n{goals_block}\n\n"
    "DELIVERABLES TO JUDGE (each is an outcome the run must produce; "
    "decide DONE or NOT DONE for each):\n{outcomes_block}\n\n"
    "WHAT THE AGENT ACTUALLY DID (its observed trajectory — use the ids "
    "to attribute which steps produced which deliverable):\n"
    "{trajectory_block}\n\n"
    "EVIDENCE (the agent's captured outputs, keyed by trajectory id):\n"
    "{evidence_block}\n\n"
    "For each deliverable in DELIVERABLES TO JUDGE, classify it into "
    "EXACTLY ONE of three states, grounded in the EVIDENCE:\n"
    '  - "met": the evidence actually contains or demonstrates the '
    "deliverable. Be strict — an attempt is not a deliverable.\n"
    '  - "failed": the goal this deliverable serves is contradicted or '
    "the deliverable provably CANNOT be produced (e.g. the user "
    "cancelled it, or a hard prerequisite is impossible). Use this ONLY "
    "when you are CONFIDENT the deliverable will never be met — not for "
    "work that simply is not done yet.\n"
    '  - "pending": not done yet, in progress, or you cannot tell from '
    "the evidence. This is the default — a run boundary is often just a "
    "pause, and pending work legitimately continues later.\n\n"
    "Reply with a single JSON object and nothing else:\n"
    "{{\n"
    '  "outcomes": [\n'
    '    {{"task_id": "<deliverable id>", '
    '"assessment": "met" | "failed" | "pending", '
    '"reason": "one-sentence justification grounded in the evidence", '
    '"contributing_task_ids": ["<trajectory id that produced it>", ...]}}\n'
    "  ]\n"
    "}}\n\n"
    "Include exactly one entry per deliverable id listed above. "
    "``contributing_task_ids`` lists the trajectory ids whose evidence "
    'supports a "met" verdict (empty list for "failed" / "pending").'
)


def _is_outcome(task: Any) -> bool:
    return getattr(task, "kind", None) is TaskKind.OUTCOME


def _is_terminal(task: Any) -> bool:
    from goldfive.types import TERMINAL_TASK_STATUSES

    return getattr(task, "status", None) in TERMINAL_TASK_STATUSES


def _format_outcomes(outcome_tasks: Sequence[Task]) -> str:
    if not outcome_tasks:
        return "(no outcome deliverables)"
    lines = []
    for t in outcome_tasks:
        desc = (getattr(t, "description", "") or "").strip()
        suffix = f" — {desc}" if desc else ""
        lines.append(f"- [{t.id}] {t.title}{suffix}")
    return "\n".join(lines)


def _format_trajectory(discovered_tasks: Sequence[Task]) -> str:
    if not discovered_tasks:
        return "(no observed trajectory yet)"
    lines = []
    for t in discovered_tasks:
        status = getattr(t, "status", "")
        status_str = str(getattr(status, "value", status) or "").upper()
        agent = (getattr(t, "assignee_agent_id", "") or "").strip()
        agent_str = f" by {agent}" if agent else ""
        lines.append(f"- [{t.id}] {t.title}{agent_str} ({status_str})")
    return "\n".join(lines)


def _format_evidence(
    completed_outputs: dict[str, str] | None,
    discovered_tasks: Sequence[Task],
    *,
    per_task_cap: int,
) -> str:
    """Render the captured outputs (#447) for the trajectory tasks.

    Keyed by task id so the judge can attribute ``contributing_task_ids``.
    Each entry is capped to ``per_task_cap`` chars (tail-truncated keeps
    the final deliverable, which is what the judge most needs).
    """
    outputs = completed_outputs or {}
    if not outputs:
        return "(no captured outputs)"
    ids = {t.id for t in discovered_tasks}
    blocks = []
    for tid, text in outputs.items():
        # Restrict to trajectory ids when we have them; otherwise show
        # all captures (defensive — a capture keyed to a non-ledger id
        # is still evidence).
        if ids and tid not in ids:
            continue
        body = (text or "").strip()
        if len(body) > per_task_cap:
            body = "…(truncated)\n" + body[-per_task_cap:]
        blocks.append(f"[{tid}]\n{body}")
    if not blocks:
        return "(no captured outputs for the observed trajectory)"
    return "\n\n".join(blocks)


#: The three outcome assessments. ``met`` → the deliverable exists;
#: ``failed`` → CONFIDENTLY unmet (goal contradicted / cannot be met);
#: ``pending`` → not done yet / uncertain (the default — carries forward
#: across a turn boundary like a goldfive#208 reachable-PENDING task). An
#: unrecognised assessment degrades to ``pending`` (never to a transition).
OUTCOME_ASSESSMENT_MET = "met"
OUTCOME_ASSESSMENT_FAILED = "failed"
OUTCOME_ASSESSMENT_PENDING = "pending"
_VALID_ASSESSMENTS = frozenset(
    {OUTCOME_ASSESSMENT_MET, OUTCOME_ASSESSMENT_FAILED, OUTCOME_ASSESSMENT_PENDING}
)


@dataclasses.dataclass(frozen=True)
class OutcomeVerdict:
    """One outcome-progress verdict (no plan mutation; advisory data).

    ``assessment`` is one of :data:`OUTCOME_ASSESSMENT_MET` /
    ``_FAILED`` / ``_PENDING``. The three-state shape (vs a ``met`` bool)
    is the goldfive#208-forced narrowing of "unmet at exit → FAILED": a
    run end is usually just a turn boundary, so only a CONFIDENTLY-unmet
    deliverable is failed; merely not-yet-met work stays PENDING and
    carries to the next turn (the dormancy-respecting choice — goldfive
    does not manufacture failure verdicts at every turn boundary).
    """

    task_id: str
    assessment: str
    reason: str = ""
    contributing_task_ids: tuple[str, ...] = ()

    @property
    def met(self) -> bool:
        return self.assessment == OUTCOME_ASSESSMENT_MET


@dataclasses.dataclass(frozen=True)
class OutcomeTransition:
    """A planned status transition for an OUTCOME task + contributes_to stamps.

    ``contributes_stamps`` maps a DISCOVERED task id → the OUTCOME id it
    advanced (the value the caller writes onto that task's
    ``contributes_to``). Only populated for ``COMPLETED`` transitions.
    """

    task_id: str
    new_status: TaskStatus
    reason: str
    contributes_stamps: tuple[tuple[str, str], ...] = ()


async def evaluate_outcome_progress(
    *,
    goals: Sequence[Goal] | Iterable[Any] | None,
    plan: Plan | Any | None,
    completed_outputs: dict[str, str] | None,
    model: str,
    call_llm: CallLLM,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    sinks: list[Any] | None = None,
    run_id: str = "",
    session_id: str = "",
    sequence_fn: Callable[[], int] | None = None,
) -> list[OutcomeVerdict]:
    """Grade every non-terminal OUTCOME task against the goals + evidence.

    One LLM call. Returns a verdict per non-terminal OUTCOME task; an
    empty list when there are no OUTCOME tasks to judge or on any
    quiet-failure path (malformed JSON, ``call_llm`` raised, no plan).
    The "quiet on failure" contract matches :func:`classify_goal_drift`:
    a flaky judge must never spuriously complete or fail a deliverable.
    """
    tasks = list(getattr(plan, "tasks", None) or [])
    outcome_tasks = [t for t in tasks if _is_outcome(t) and not _is_terminal(t)]
    if not outcome_tasks:
        return []
    discovered_tasks = [
        t
        for t in tasks
        if getattr(t, "discovered", False)
        or getattr(t, "kind", None) is TaskKind.DISCOVERED
    ]
    system = system_prompt or OUTCOME_PROGRESS_SYSTEM_PROMPT
    template = user_prompt_template or OUTCOME_PROGRESS_USER_PROMPT_TEMPLATE
    user = template.format(
        goals_block=_format_goals(goals),
        outcomes_block=_format_outcomes(outcome_tasks),
        trajectory_block=_format_trajectory(discovered_tasks),
        evidence_block=_format_evidence(
            completed_outputs, discovered_tasks, per_task_cap=OUTCOME_PROGRESS_EVIDENCE_MAX_CHARS
        ),
    )
    valid_outcome_ids = {t.id for t in outcome_tasks}
    valid_discovered_ids = {t.id for t in discovered_tasks}
    parsed: dict[str, Any] | None = None
    try:
        from goldfive._llm import call_llm_budget, call_llm_thinking_disabled
        from goldfive._llm_span import goldfive_llm_span

        async with goldfive_llm_span(
            sinks=list(sinks or []),
            name="judge_outcome_progress",
            model=model,
            session_id=session_id,
            run_id=run_id,
            task_id="",
            sequence_fn=sequence_fn,
            input_preview=_format_outcomes(outcome_tasks),
        ) as span:
            with call_llm_budget(OUTCOME_PROGRESS_MAX_OUTPUT_TOKENS), call_llm_thinking_disabled():
                raw = await call_llm(system, user, model)
            parsed = _parse_response(raw)
            if parsed is None:
                span.output_preview = "(unparseable verdict)"
                span.decision_summary = "outcome-progress: unparseable verdict (no transitions)"
            else:
                n = len(parsed.get("outcomes", []) or []) if isinstance(parsed, dict) else 0
                span.output_preview = raw[:2048] if isinstance(raw, str) else "(non-str)"
                span.decision_summary = f"outcome-progress: graded {n} deliverable(s)"
    except Exception as exc:  # noqa: BLE001 — never break the run
        log.warning("evaluate_outcome_progress: call_llm raised %s; no transitions", exc)
        return []
    if not isinstance(parsed, dict):
        return []
    raw_outcomes = parsed.get("outcomes")
    if not isinstance(raw_outcomes, list):
        log.debug("evaluate_outcome_progress: verdict missing 'outcomes' list; ignored")
        return []
    verdicts: list[OutcomeVerdict] = []
    seen: set[str] = set()
    for entry in raw_outcomes:
        if not isinstance(entry, dict):
            continue
        tid = str(entry.get("task_id", "") or "").strip()
        # Only accept verdicts for OUTCOME tasks we actually asked about.
        if tid not in valid_outcome_ids or tid in seen:
            continue
        seen.add(tid)
        assessment = str(entry.get("assessment", "") or "").strip().lower()
        # Unrecognised / missing assessment degrades to PENDING — never to
        # a transition (a flaky judge must not manufacture a terminal).
        if assessment not in _VALID_ASSESSMENTS:
            assessment = OUTCOME_ASSESSMENT_PENDING
        reason = str(entry.get("reason", "") or "").strip()
        contributing = entry.get("contributing_task_ids") or []
        contrib_ids = tuple(
            str(c).strip()
            for c in contributing
            if isinstance(c, (str, int)) and str(c).strip() in valid_discovered_ids
        )
        verdicts.append(
            OutcomeVerdict(
                task_id=tid,
                assessment=assessment,
                reason=reason,
                contributing_task_ids=contrib_ids,
            )
        )
    return verdicts


def plan_outcome_transitions(
    plan: Plan | Any | None,
    verdicts: Sequence[OutcomeVerdict],
    *,
    run_ending: bool = False,
    goal_predicates_met: bool = True,
) -> list[OutcomeTransition]:
    """Turn outcome verdicts into the transitions the caller should apply.

    PURE — no plan mutation, no I/O. Rules:

    * ``met`` OUTCOME (non-terminal) → COMPLETED at any cadence, BUT only
      when ``goal_predicates_met`` (a user-supplied predicate that is
      explicitly unmet overrides the LLM and blocks completion).
    * ``failed`` (CONFIDENTLY unmet) OUTCOME → FAILED only when
      ``run_ending``. A deliverable the judge merely cannot confirm yet is
      ``pending``, not ``failed``; a run end is usually just a turn
      boundary (goldfive#208), so uncertain work carries forward as
      reachable-PENDING and the next boundary judge re-evaluates.
    * ``pending`` OUTCOME → no transition (left PENDING to carry forward).
    * A met → COMPLETED transition carries ``contributes_stamps`` for the
      named DISCOVERED tasks (``discovered_id -> outcome_id``), but only
      for tasks that exist in the plan and are not already stamped with a
      different outcome (stable, idempotent).

    OUTCOME tasks already terminal, and verdicts whose ``task_id`` is not
    a non-terminal OUTCOME in ``plan``, are skipped.
    """
    tasks_by_id: dict[str, Task] = {
        t.id: t for t in (getattr(plan, "tasks", None) or []) if getattr(t, "id", "")
    }
    transitions: list[OutcomeTransition] = []
    for v in verdicts:
        task = tasks_by_id.get(v.task_id)
        if task is None or not _is_outcome(task) or _is_terminal(task):
            continue
        if v.assessment == OUTCOME_ASSESSMENT_MET:
            if not goal_predicates_met:
                # A user predicate is authoritative and currently unmet —
                # do not complete the deliverable on the LLM's say-so.
                continue
            stamps: list[tuple[str, str]] = []
            for cid in v.contributing_task_ids:
                ctask = tasks_by_id.get(cid)
                if ctask is None:
                    continue
                existing = (getattr(ctask, "contributes_to", "") or "").strip()
                if existing and existing != v.task_id:
                    # Already attributed to a different outcome — leave it.
                    continue
                if existing == v.task_id:
                    continue  # idempotent — no-op stamp
                stamps.append((cid, v.task_id))
            transitions.append(
                OutcomeTransition(
                    task_id=v.task_id,
                    new_status=TaskStatus.COMPLETED,
                    reason=v.reason or "outcome met",
                    contributes_stamps=tuple(stamps),
                )
            )
        elif v.assessment == OUTCOME_ASSESSMENT_FAILED and run_ending:
            transitions.append(
                OutcomeTransition(
                    task_id=v.task_id,
                    new_status=TaskStatus.FAILED,
                    reason=v.reason or "outcome cannot be met",
                )
            )
        # OUTCOME_ASSESSMENT_PENDING (and "failed" mid-run) → no transition.
    return transitions
