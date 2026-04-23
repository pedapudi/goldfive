"""Turn-aware planning gate — decide whether to re-plan on a new turn.

Goldfive's :class:`~goldfive.runner.Runner` used to invoke
:class:`GoalDeriver.derive` + :class:`Planner.generate` on every call
to :meth:`Runner.run`. Multi-turn conversations where turn N+1 is a
conversational follow-up ("where is the presentation located?") would
therefore mint a fresh 6-task workflow plan repeating the whole prior
run, even though the coordinator LLM itself can answer in one turn
from conversation history.

This module ships a lightweight classifier that slots in *before*
``GoalDeriver.derive`` on each turn after the first. Given:

- the prior plan (if any, from ``session.plan``),
- completed_results (from ``session.completed_results``),
- the new user_input string,
- the current conversation_id,

it returns one of three verdicts:

``"new_work"``
    The user_input introduces genuinely new work that isn't covered
    by the prior plan. Runner should run full ``GoalDeriver.derive``
    + ``Planner.generate`` as before.

``"conversational"``
    The user_input is a follow-up that can be answered from existing
    context alone. Runner should skip planning entirely — no new
    GoalDerived or PlanSubmitted for this turn — and let the
    coordinator answer directly over the existing plan / completed
    results.

``"refine_existing"``
    The user_input extends or tweaks the prior plan. Runner should
    run ``Planner.refine`` against the current plan with the new
    user_input as a synthesized steer goal, adding delta tasks only
    rather than re-planning from scratch.

Two classifier modes ship:

* :func:`heuristic_classify_turn` — deterministic rule-based gate.
  Crude but safe: short follow-ups after a prior plan are assumed
  conversational; no prior plan forces ``new_work``. Used as the
  fallback when no ``call_llm`` is available and as the basis for
  the unit tests.

* :func:`classify_turn` — LLM-backed gate. Takes an async
  ``call_llm(system, user, model) -> str`` callable and prompts it
  for a one-word classification. Falls through to
  :func:`heuristic_classify_turn` on any LLM/parse error so a
  misbehaving LLM never hangs the Runner.

The Runner is responsible for opt-in wiring (see
``Runner(planner_gate=...)``): when no gate is supplied the default
is the LLM-or-heuristic hybrid, and callers can disable by passing
``planner_gate=None`` for deterministic replay / pure-unit tests.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from goldfive.types import Plan

log = logging.getLogger("goldfive.planner_gate")

#: The three classifier verdicts. The Runner's behaviour for each is
#: documented at module level.
TurnClassification = Literal["new_work", "conversational", "refine_existing"]

#: Roughly "one sentence of input". Follow-ups under this token budget
#: that arrive after a prior plan are treated as conversational by the
#: heuristic fallback. Chosen for safety, not precision — the LLM gate
#: is the primary classifier when ``call_llm`` is available.
_HEURISTIC_SHORT_TOKEN_BUDGET: int = 20

#: Canonical verdict set. Lowercased, underscore-joined.
_ALLOWED: frozenset[str] = frozenset(
    {"new_work", "conversational", "refine_existing"}
)

_SYSTEM_PROMPT = """\
You are a turn-classifier for a multi-agent orchestration system.

The system has already executed a PRIOR PLAN of tasks for the user's
earlier request(s) in this conversation. Now the user has issued a
NEW INPUT. Your job is to classify the new input into one of three
buckets so the orchestrator knows whether to re-plan.

Verdicts:

- "new_work": the new input asks for genuinely new work that is NOT
  covered by the prior plan's tasks or the completed_results. A
  wholly new topic, a wholly new artefact, a new outcome the prior
  plan did not produce.

- "conversational": the new input is a question or clarification
  about work that has ALREADY been done — e.g. "where did you save
  the output?", "what was the second slide?", "did you use source
  X?", "summarise what you did". These can be answered from the
  existing plan / completed_results / conversation history without
  running any new tasks.

- "refine_existing": the new input tweaks, extends, or revises the
  PRIOR PLAN — e.g. "make it funnier", "add a slide about Z",
  "also translate to Spanish", "change the title". The existing
  plan's completed tasks should be preserved; a small delta of new
  tasks is appended.

Guidelines:
- When in doubt between "conversational" and "refine_existing",
  prefer "conversational" — the coordinator can always answer and
  the user can restate if they actually wanted new work.
- When in doubt between "refine_existing" and "new_work", prefer
  "refine_existing" — the prior plan's completed tasks give the
  refined plan a running start.

Reply with a single JSON object and NOTHING ELSE:
{"verdict": "new_work" | "conversational" | "refine_existing",
 "reason": "<one-sentence why>"}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_count(text: str) -> int:
    """Cheap whitespace-split token count. Good enough for a budget check."""
    return len((text or "").split())


_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)?\s*\n(?P<body>.*?)\n```\s*$", re.DOTALL)


def _strip_code_fences(raw: str) -> str:
    m = _FENCE_RE.match(raw or "")
    return m.group("body") if m else (raw or "")


def _plan_brief(plan: Plan | None) -> str:
    """Render a compact human-readable summary of the prior plan.

    Used in the LLM prompt. ``None`` collapses to an empty string
    because a missing prior plan is handled by the caller before we
    get here (see :func:`classify_turn`).
    """
    if plan is None:
        return ""
    lines: list[str] = []
    if plan.summary:
        lines.append(f"Summary: {plan.summary}")
    if plan.tasks:
        lines.append("Tasks:")
        for t in plan.tasks:
            tid = t.id or "(no-id)"
            status = getattr(t.status, "value", str(t.status))
            title = t.title or t.description or ""
            lines.append(f"  - [{tid} / {status}] {title}")
    return "\n".join(lines)


def _results_brief(results: Mapping[str, str] | None, *, cap: int = 10) -> str:
    if not results:
        return ""
    entries = list(results.items())[:cap]
    lines = ["Completed results:"]
    for tid, summary in entries:
        snippet = (summary or "").strip().splitlines()
        one = snippet[0] if snippet else ""
        if len(one) > 160:
            one = one[:157] + "..."
        lines.append(f"  - {tid}: {one}")
    more = len(results) - len(entries)
    if more > 0:
        lines.append(f"  ... and {more} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


def heuristic_classify_turn(
    *,
    prior_plan: Plan | None,
    completed_results: Mapping[str, str] | None,
    user_input: str,
    conversation_id: str = "",
) -> TurnClassification:
    """Deterministic rule-based gate. LLM-free.

    Rules:

    - No prior plan → always ``"new_work"`` (first turn or fresh
      conversation).
    - Prior plan exists AND user_input is short (``< 20`` tokens) →
      ``"conversational"``. Short follow-ups are overwhelmingly
      questions about prior work, not new workflows.
    - Otherwise → ``"new_work"``. Conservative default: the heuristic
      never picks ``"refine_existing"`` on its own because getting the
      refine path wrong silently mangles the plan, whereas getting
      ``"new_work"`` wrong just means the user pays for an extra plan
      that still completes.

    The LLM-backed :func:`classify_turn` is strictly more precise and
    should be preferred when a ``call_llm`` is available.
    """
    _ = conversation_id  # reserved for future heuristics
    if prior_plan is None or not prior_plan.tasks:
        return "new_work"
    if _token_count(user_input) < _HEURISTIC_SHORT_TOKEN_BUDGET:
        return "conversational"
    return "new_work"


# ---------------------------------------------------------------------------
# LLM-backed gate
# ---------------------------------------------------------------------------


async def classify_turn(
    *,
    call_llm: Callable[[str, str, str], Awaitable[str]] | None,
    prior_plan: Plan | None,
    completed_results: Mapping[str, str] | None,
    user_input: str,
    conversation_id: str = "",
    model: str = "",
) -> TurnClassification:
    """Classify a turn as new_work / conversational / refine_existing.

    When ``call_llm`` is ``None`` OR ``prior_plan`` is falsy, skips
    the LLM entirely and delegates to :func:`heuristic_classify_turn`.
    Keeping the gate off the hot path in both cases is the point of
    the guard — short-circuiting on missing prior plan avoids an LLM
    roundtrip on turn 1 of every conversation.

    Any exception raised by ``call_llm`` or any malformed response is
    logged and collapses to :func:`heuristic_classify_turn`. A
    misbehaving gate must never hang or mis-route the Runner.
    """
    if call_llm is None or prior_plan is None or not prior_plan.tasks:
        return heuristic_classify_turn(
            prior_plan=prior_plan,
            completed_results=completed_results,
            user_input=user_input,
            conversation_id=conversation_id,
        )

    user_prompt = _build_user_prompt(
        prior_plan=prior_plan,
        completed_results=completed_results,
        user_input=user_input,
        conversation_id=conversation_id,
    )
    try:
        raw = await call_llm(_SYSTEM_PROMPT, user_prompt, model)
    except Exception as exc:  # noqa: BLE001
        log.warning("planner_gate.classify_turn: call_llm raised: %s", exc)
        return heuristic_classify_turn(
            prior_plan=prior_plan,
            completed_results=completed_results,
            user_input=user_input,
            conversation_id=conversation_id,
        )
    verdict = _parse_verdict(raw)
    if verdict is None:
        log.warning(
            "planner_gate.classify_turn: unparseable verdict from LLM; "
            "falling back to heuristic. raw=%r",
            raw,
        )
        return heuristic_classify_turn(
            prior_plan=prior_plan,
            completed_results=completed_results,
            user_input=user_input,
            conversation_id=conversation_id,
        )
    return verdict


def _build_user_prompt(
    *,
    prior_plan: Plan | None,
    completed_results: Mapping[str, str] | None,
    user_input: str,
    conversation_id: str,
) -> str:
    chunks: list[str] = []
    if conversation_id:
        chunks.append(f"Conversation id: {conversation_id}")
    prior = _plan_brief(prior_plan)
    if prior:
        chunks.append("PRIOR PLAN:\n" + prior)
    results = _results_brief(completed_results)
    if results:
        chunks.append(results)
    chunks.append("NEW USER INPUT:\n" + (user_input or ""))
    chunks.append(
        'Classify the new input. Reply JSON only: {"verdict": "...", "reason": "..."}'
    )
    return "\n\n".join(chunks)


def _parse_verdict(raw: Any) -> TurnClassification | None:
    """Parse the LLM's response into a canonical verdict, or None on failure."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _strip_code_fences(raw).strip()
    # First try JSON.
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        verdict = parsed.get("verdict")
        if isinstance(verdict, str):
            v = verdict.strip().lower().replace("-", "_").replace(" ", "_")
            if v in _ALLOWED:
                return v  # type: ignore[return-value]
    # Fallback: scan for a bare token. Some models emit just the verdict
    # string when the prompt is terse enough.
    lowered = cleaned.lower().replace("-", "_").replace(" ", "_")
    for candidate in _ALLOWED:
        # Word-boundary match to avoid "new_workflow" matching "new_work".
        if re.search(rf"\b{candidate}\b", lowered):
            return candidate  # type: ignore[return-value]
    return None


__all__ = [
    "TurnClassification",
    "classify_turn",
    "heuristic_classify_turn",
]
