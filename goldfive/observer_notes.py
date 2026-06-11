"""Observation + goal note composers (AGENCY-PRESERVATION.md PR 4).

Every message goldfive composes *for the wrapped agent* renders through
this module. The content contract replaces the retired
``_CORRECTIVE_TEMPLATES`` / ``compose_corrective_user_message`` command
templates ("proceed to '{next_task_title}' via {next_task_agent}",
"do NOT retry") with the observation+goal shape from
``docs/design/AGENCY-PRESERVATION.md`` §2 (surface 3 — *speak*: honestly
attributed advisory notes) and the PR 4 entry:

* **Observation** — factual and evidence-bearing ("``search_web`` was
  called 5 times in the last 10 tool calls with identical arguments"),
  never imperative, never problem-naming ("failed" / "broken" /
  "wrong"). The neutral-framing lesson comes from
  :mod:`goldfive._correction_injection` (goldfive#250 / #252 / #253):
  problem-naming language provokes apologies / meta-commentary /
  retries-of-the-wrong-thing — but where #251 chose *directive*
  framing as the alternative, observer notes are *neither* commanding
  nor apology-bait: neutral facts plus the goal.
* **The user's goal** — verbatim from ``session.goals`` so the agent
  re-anchors on the authoritative reference, not on goldfive's opinion
  of what to do next.
* **Status** (optional) — goldfive's bookkeeping presented AS
  bookkeeping ("goldfive's tracking shows task t2 recorded as
  completed"), never as instruction. Task ids may appear here as
  ledger facts; they never appear as commands.
* **Advisory footer** — the explicit authority statement. Control
  (cancel / pause / terminate) stays on the control channel and never
  depends on the agent honouring a note (the no-prompt-contract rule).

Question form
-------------
When the triggering verdict is an LLM judge's *opinion* (rather than a
counted, observed fact) and its severity sits below CRITICAL, the note
is rendered in question form ("… does the current approach still serve
the user's goal?"). A self-generated correction integrates into a
trajectory better than an external one, and a question is the
lowest-footprint speech act available. Statement form stays for hard
observed facts (loop counters, budgets) and for CRITICAL judge
verdicts. Neither :class:`~goldfive.types.DriftEvent` nor
:class:`~goldfive.judges.JudgeVerdict` carries a per-verdict confidence
scalar today (``focus_confidence`` on ``ReasoningJudgeVerdict`` scores
*task attribution*, not the drift verdict itself), so severity is the
confidence proxy; judges asked to author ``note_to_agent`` directly are
additionally prompted to phrase the note as a question when unsure.

Rendered shape (PR 6 compatibility)
-----------------------------------
The body lines produced here match the "Rendered block shape" the PR 6
observer-note channel will wrap in ``[GOLDFIVE OBSERVER NOTE …]``
markers::

    Observation: <factual description of what was observed>
    The user's goal: <from session.goals>
    Status: <bookkeeping snapshot>            (optional)
    This note is advisory. How to proceed is your decision; the user's
    instructions remain authoritative.

Until PR 6 lands, the legacy delivery paths (nudge replay, GOLDFIVE
steer restart, active-steer body) consume this rendering as plain text;
only their ``[GOLDFIVE …]`` attribution headers are added by the
executor (delivery mechanics unchanged in this PR).

Observation sourcing — the fallback chain
-----------------------------------------
:func:`observation_for_drift` resolves the observation text in strict
preference order:

1. ``drift.note_to_agent`` — authored by the LLM judge in the same call
   that produced the verdict (PR 4 scope item 3; prompts in
   :mod:`goldfive.drift.reasoning_judge` / :mod:`goldfive.drift.goals`).
   Used verbatim; the judge already chose statement vs question form.
2. Deterministic detector facts — for tool-loop drifts the structured
   ``drift.raw`` payload (tool name, exact-match args fingerprint,
   counts, window length) renders directly into the observation line.
   No new LLM calls, no NL classification — counts and exact-match
   hashes only (the #166/#167 rule).
3. ``drift.detail`` — the human-readable reason every detector already
   emits.
4. A per-kind neutral fallback template (no detail available at all).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.types import Session

__all__ = [
    "ADVISORY_FOOTER",
    "GOAL_QUESTION",
    "OBSERVER_NOTE_BLOCK_BEGIN",
    "OBSERVER_NOTE_BLOCK_END",
    "OBSERVER_NOTE_MARKER",
    "compose_note_for_drift",
    "compose_observer_note",
    "compose_status_line",
    "observation_for_drift",
    "render_goals_text",
]


#: The advisory footer every note carries. Exact wording pinned by the
#: AGENCY-PRESERVATION.md PR 6 "Rendered block shape" — PR 6's channel
#: wraps these same lines, so the text must not fork between regimes.
ADVISORY_FOOTER: str = (
    "This note is advisory. How to proceed is your decision; the user's "
    "instructions remain authoritative."
)

#: Stable goldfive-minted marker that prefixes a rendered observer-note
#: block (AGENCY-PRESERVATION.md PR 6 "Rendered block shape"). This is the
#: SINGLE SOURCE for the marker: the PR 6 channel renders the block
#: opening with this prefix, and the PR 6b
#: :class:`~goldfive.context_editor.PruneStaleSteerRule` detects goldfive
#: notes by matching this substring. Both sides import from here so the
#: constant can never drift between the writer and the reader.
OBSERVER_NOTE_MARKER: str = "[GOLDFIVE OBSERVER NOTE"

#: The full opening / closing lines of a rendered observer-note block.
#: Derived from :data:`OBSERVER_NOTE_MARKER` so the detection prefix is
#: guaranteed to be a prefix of the rendered opening line. The PR 6
#: channel wraps each delivered note between these two lines.
OBSERVER_NOTE_BLOCK_BEGIN: str = (
    f"{OBSERVER_NOTE_MARKER} — from an external monitoring layer, "
    "not from the user]"
)
OBSERVER_NOTE_BLOCK_END: str = "[/GOLDFIVE OBSERVER NOTE]"

#: Question-form suffix appended to the observation when the verdict is
#: a low-confidence judge opinion (see module docstring). Mirrors the
#: design doc's example phrasing.
GOAL_QUESTION: str = "Does the current approach still serve the user's goal?"


# Drift kinds minted by LLM judges (subjective opinions) rather than by
# deterministic counters. Below-CRITICAL verdicts of these kinds render
# in question form when goldfive composes the observation itself (a
# judge-authored ``note_to_agent`` already chose its own form).
_JUDGE_OPINION_KINDS: frozenset[DriftKind] = frozenset(
    {
        DriftKind.OFF_TOPIC,
        DriftKind.GOAL_DRIFT,
        DriftKind.INTENT_DIVERGENCE,
        DriftKind.JUSTIFIED_DEVIATION,
        DriftKind.UNCERTAIN_PROGRESS,
    }
)


# Per-kind neutral fallback observations, used only when the drift
# carries neither a judge-authored note, structured detector facts, nor
# a ``detail`` string. ``{task}`` interpolates the current task id (or
# the readable placeholder). Wording rules: facts only; no imperative
# means-verbs ("retry", "proceed to", "call", "do not", …); no
# problem-naming ("failed" / "broken" / "wrong" — the
# ``_correction_injection`` lesson); no goldfive postmortem jargon in
# the prose ("drift", "steerer", "synthetic").
_FALLBACK_OBSERVATIONS: dict[DriftKind, str] = {
    DriftKind.LOOPING_REASONING: (
        "Repeated, near-identical activity was observed on {task} "
        "without recorded progress."
    ),
    DriftKind.LOOPING_TOOL_CALL: (
        "The same tool invocation was observed several times on {task} "
        "without recorded progress."
    ),
    DriftKind.PLAN_DIVERGENCE: (
        "goldfive's plan tracking did not match the activity observed on {task}."
    ),
    DriftKind.AGENT_REFUSAL: (
        "The most recent response on {task} declined to continue."
    ),
    DriftKind.MODEL_REFUSAL: (
        "The model declined to produce a response on {task}."
    ),
    DriftKind.INTENT_DIVERGENCE: (
        "Recent reasoning on {task} moved away from the stated goals."
    ),
    DriftKind.TOOL_ERROR: (
        "A tool invocation on {task} returned an error result."
    ),
    DriftKind.RUNAWAY_DELEGATION: (
        "A large number of delegations was observed while {task} remained open."
    ),
    DriftKind.SELF_REPORTED_STUCK: (
        "In a recent self-check, the agent's own assessment reported no "
        "progress on {task}."
    ),
    DriftKind.CONFABULATION_RISK: (
        "Output on {task} referenced external data, but no tool "
        "invocation was recorded for it."
    ),
    DriftKind.GOAL_DRIFT: (
        "Recent activity did not appear to advance the user's goal "
        "while {task} was active."
    ),
    DriftKind.OFF_TOPIC: (
        "Recent reasoning on {task} appeared unrelated to the bound "
        "task and goals."
    ),
    DriftKind.JUSTIFIED_DEVIATION: (
        "Recent reasoning on {task} departed from the recorded task in "
        "response to new information."
    ),
}

#: Generic fallback for kinds without a dedicated template. Phrased as
#: an open observation (the question-form decision still applies on
#: top of it for judge-opinion kinds).
_GENERIC_FALLBACK_OBSERVATION: str = (
    "goldfive's monitoring raised a signal while {task} was active."
)


def render_goals_text(goals: Any) -> str:
    """Project ``session.goals`` onto a one-line verbatim summary string.

    Joins each goal's ``summary`` (or the goal itself when it is a
    plain string) with ``"; "`` so a multi-goal session reads as one
    compact line. Returns ``"(no goals recorded for this run)"`` when
    no goals are available so the note shape stays invariant — the
    PR 6 block always carries the goal line.
    """
    summaries: list[str] = []
    for g in goals or ():
        summary = str(getattr(g, "summary", "") or "")
        if not summary and isinstance(g, str):
            summary = g
        summary = summary.strip()
        if summary:
            summaries.append(summary)
    if not summaries:
        return "(no goals recorded for this run)"
    return "; ".join(summaries)


def compose_observer_note(
    *,
    observation: str,
    goals_text: str = "",
    status: str = "",
    question_form: bool = False,
) -> str:
    """Render one observer note in the PR 6 block-body shape.

    Parameters
    ----------
    observation:
        The factual observation line (already neutral; see the module
        docstring for sourcing). Required — an empty observation
        renders a generic "goldfive's monitoring raised a signal."
        line rather than an empty slot.
    goals_text:
        The verbatim/derived goal summary (see
        :func:`render_goals_text`). Empty string renders the
        "(no goals recorded for this run)" placeholder so the line is
        always present.
    status:
        Optional bookkeeping snapshot (see :func:`compose_status_line`).
        Omitted from the rendering when empty.
    question_form:
        When ``True``, the observation is rendered as a question by
        appending :data:`GOAL_QUESTION` — the lowest-footprint speech
        act for a low-confidence judge opinion. Statement form is the
        default and stays for hard observed facts.
    """
    obs = (observation or "").strip() or "goldfive's monitoring raised a signal."
    if question_form and not obs.rstrip().endswith("?"):
        obs = f"{obs} {GOAL_QUESTION}"
    lines = [f"Observation: {obs}"]
    lines.append(
        f"The user's goal: {goals_text.strip() or '(no goals recorded for this run)'}"
    )
    status = (status or "").strip()
    if status:
        lines.append(f"Status: {status}")
    lines.append(ADVISORY_FOOTER)
    return "\n".join(lines)


def _readable_task(task_id: str) -> str:
    return task_id or "the current task"


def _tool_loop_observation(drift: DriftEvent) -> str:
    """Render a tool-loop drift's structured facts into one line.

    The tool-loop tracker (:mod:`goldfive.drift.tool_loops`) stamps a
    structured dict onto ``drift.raw`` carrying the mode, tool name,
    exact-match args fingerprint, repeat count, and window length — the
    facts the detector already holds. We render exactly those counts;
    no inference, no NL classification (the #166/#167 rule: exact-match
    arg hashes and counts are fine, regex over natural language is not).
    Returns ``""`` when the raw payload is not the tracker's shape so
    the caller falls through to ``drift.detail``.
    """
    raw = drift.raw
    if not isinstance(raw, dict):
        return ""
    mode = str(raw.get("mode", "") or "")
    window_len = int(raw.get("window_len", 0) or 0)
    if mode == "exact":
        tool = str(raw.get("tool_name", "") or "")
        count = int(raw.get("count", 0) or 0)
        fingerprint = str(raw.get("args_hash", "") or "")
        if not tool or count <= 0:
            return ""
        suffix = f" (args fingerprint {fingerprint})" if fingerprint else ""
        return (
            f"`{tool}` was invoked {count} times in the last {window_len} "
            f"tool invocations with identical arguments{suffix}; no task "
            "progress was recorded in that window."
        )
    if mode == "name":
        tool = str(raw.get("tool_name", "") or "")
        count = int(raw.get("count", 0) or 0)
        if not tool or count <= 0:
            return ""
        return (
            f"`{tool}` was invoked {count} times in the last {window_len} "
            "tool invocations with varying arguments; no task progress "
            "was recorded in that window."
        )
    if mode == "alternating":
        tools = [str(t) for t in (raw.get("tools") or []) if t]
        if len(tools) != 2:
            return ""
        return (
            f"`{tools[0]}` and `{tools[1]}` alternated across the last "
            f"{window_len} tool invocations without recorded task progress."
        )
    return ""


def observation_for_drift(drift: DriftEvent) -> tuple[str, bool]:
    """Resolve ``(observation_text, question_form)`` for a drift.

    Implements the fallback chain from the module docstring:
    judge-authored ``note_to_agent`` (verbatim, judge chose its own
    form) → deterministic detector facts (``drift.raw`` for tool
    loops) → ``drift.detail`` → per-kind neutral fallback.

    ``question_form`` is ``True`` only for goldfive-composed
    observations of judge-opinion kinds below CRITICAL severity
    (severity as the confidence proxy — see the module docstring).
    """
    note = str(getattr(drift, "note_to_agent", "") or "").strip()
    if note:
        # The judge authored this in the same call that produced the
        # verdict; it already chose statement vs question form.
        return note, False
    question = (
        drift.kind in _JUDGE_OPINION_KINDS
        and drift.severity is not DriftSeverity.CRITICAL
    )
    if drift.kind in (DriftKind.LOOPING_REASONING, DriftKind.LOOPING_TOOL_CALL):
        rendered = _tool_loop_observation(drift)
        if rendered:
            # Hard observed fact — statement form always.
            return rendered, False
    detail = str(getattr(drift, "detail", "") or "").strip()
    if detail:
        return detail, question
    template = _FALLBACK_OBSERVATIONS.get(drift.kind, _GENERIC_FALLBACK_OBSERVATION)
    return template.format(task=_readable_task(drift.current_task_id)), question


def compose_status_line(plan: Plan | None, current_task_id: str) -> str:
    """Render goldfive's bookkeeping snapshot for the Status line.

    Bookkeeping, not instruction: the line reports what goldfive's
    ledger currently records — the triggering task's recorded status
    and the open-work count — and deliberately does NOT name a "next"
    task or assignee (the retired templates' command surface). Task
    ids appear here as ledger facts only.

    Returns ``""`` when there is nothing to report (no plan and no
    task id), so callers can omit the Status line entirely.
    """
    parts: list[str] = []
    task_status = ""
    terminal = False
    if plan is not None and current_task_id:
        for t in plan.tasks:
            if t.id == current_task_id:
                status = getattr(t, "status", None)
                task_status = str(getattr(status, "value", status) or "")
                terminal = status in TERMINAL_TASK_STATUSES
                break
    if current_task_id:
        if task_status and terminal:
            parts.append(
                f"goldfive's tracking records task {current_task_id} as "
                f"{task_status.lower()}"
            )
        elif task_status:
            parts.append(
                f"goldfive's tracking records task {current_task_id} as "
                f"{task_status.lower()} (still open)"
            )
        else:
            parts.append(
                f"goldfive's tracking associates this note with task "
                f"{current_task_id}"
            )
    if plan is not None:
        completed = sum(
            1 for t in plan.tasks if t.status is TaskStatus.COMPLETED
        )
        open_count = sum(
            1 for t in plan.tasks if t.status not in TERMINAL_TASK_STATUSES
        )
        parts.append(
            f"its ledger shows {completed} task(s) recorded complete and "
            f"{open_count} still open"
        )
    if not parts:
        return ""
    return "; ".join(parts) + "."


def compose_note_for_drift(
    *,
    drift: DriftEvent,
    session: Session | None = None,
    plan: Plan | None = None,
    status: str | None = None,
) -> str:
    """Compose the full observer note for ``drift``.

    The one entry point every legacy delivery path renders through:
    the post-ABSORB nudge queue and Level 2 ``_dispatch_nudge``
    (:mod:`goldfive.drift_observer`), the ``GOLDFIVE_STEER`` corrective
    body (``_compose_goldfive_steer_body`` /
    ``_dispatch_goldfive_steer_control``), and the deprecated
    :func:`goldfive.steerer.compose_corrective_user_message` shim.

    ``session`` supplies ``goals`` (the goal line) and, when ``plan``
    is not given explicitly, the live plan for the Status snapshot.
    ``status=None`` derives the bookkeeping line from the plan;
    pass ``status=""`` to suppress the Status line entirely.
    """
    effective_plan = plan if plan is not None else getattr(session, "plan", None)
    goals = getattr(session, "goals", None) if session is not None else None
    observation, question_form = observation_for_drift(drift)
    if status is None:
        status = compose_status_line(effective_plan, drift.current_task_id)
    return compose_observer_note(
        observation=observation,
        goals_text=render_goals_text(goals),
        status=status,
        question_form=question_form,
    )
