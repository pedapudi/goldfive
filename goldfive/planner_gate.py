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

#: Steer-language regex. A non-empty match in the user's input forces the
#: heuristic to return ``"refine_existing"`` regardless of token count —
#: catches the "forget X, do Y instead" / "no, don't X, do Y" pivot pattern
#: that previously fell through to ``"conversational"`` (token < 20) or
#: ``"new_work"`` (token >= 20). Both fall-through verdicts dropped the
#: prior plan's constraints and bypassed the steerer's pipeline (no
#: PlanRevised, no DriftDetected(USER_STEER), no sticky-goal preservation),
#: which is what the goldfive#270 E2E hit.
#:
#: Anchored to start-of-string OR a sentence break so "I'd like to forget
#: about ..." doesn't false-positive — only leading directives match.
_STEER_PATTERN_RE = re.compile(
    r"(?:^|[.?!]\s+)(?:"
    r"forget|"
    r"never mind|"
    r"nevermind|"
    r"scratch that|"
    r"strike that|"
    r"actually,?\s+|"
    r"wait,?\s+|"
    r"no,?\s+(?:wait|don't|do not|not)|"
    r"stop\b|"
    r"instead\b|"
    r"change(?:\s+(?:that|the\s+plan|topic|to))|"
    r"switch\s+to|"
    r"do not\b|"
    r"don't\b|"
    r"no longer\b"
    r")",
    re.IGNORECASE,
)


def _looks_like_steer(text: str) -> bool:
    """Return True if ``text`` opens with a directive that revises prior work.

    See :data:`_STEER_PATTERN_RE`. Used by :func:`heuristic_classify_turn`
    to escape the token-count-only branch and route steer-shaped messages
    through ``refine_existing`` so the runner's USER_STEER pipeline fires.
    """
    if not text:
        return False
    return bool(_STEER_PATTERN_RE.search(text))


#: Factual-question regex (Phase 2.X / goldfive#271 Gap 4). Matches the
#: open-ended interrogative shape of "where/when/how/what/why/which/who"
#: + "is/will/did/does/are/was/were/have/can" when the question is about
#: the existing work. Stronger than the bare token-count heuristic — a
#: 6-token question like "where will the slides be saved?" was being
#: mis-routed to ``refine_existing`` by the LLM gate. The heuristic
#: returns ``"conversational"`` for these unconditionally.
#:
#: Anchored to start-of-string OR a sentence break so "Tell me more
#: about where the data is" doesn't false-positive.
_FACTUAL_QUESTION_RE = re.compile(
    r"(?:^|[.?!]\s+)(?:"
    r"where(?:\s+(?:is|are|was|were|will|did|does|do|can|could|should|would|the|am)\b)|"
    r"when(?:\s+(?:is|are|was|were|will|did|does|do|can|could|should|would|the|am)\b)|"
    r"how(?:\s+(?:is|are|was|were|will|did|does|do|can|could|should|would|much|many|long|the)\b)|"
    r"what(?:'s|\s+(?:is|are|was|were|will|did|does|do|can|could|should|would|happened|the))|"
    r"why(?:\s+(?:is|are|was|were|did|does|do|can|could|should|would|the))|"
    r"which(?:\s+(?:is|are|was|were|did|does|do|the|one|of))|"
    r"who(?:'s|\s+(?:is|are|was|were|did|does|do|the))|"
    r"can\s+you\s+(?:tell|show|explain|describe|list)|"
    r"could\s+you\s+(?:tell|show|explain|describe|list)|"
    r"did\s+you|"
    r"is\s+(?:the|it|that|there)|"
    r"are\s+(?:the|those|there)"
    r")",
    re.IGNORECASE,
)


def _looks_like_factual_question(text: str) -> bool:
    """Return True if ``text`` opens with a factual interrogative.

    Phase 2.X / goldfive#271 Gap 4: the LLM gate misclassified
    "where will the slides be saved?" as ``refine_existing`` despite
    the prompt's explicit example of "where did you save the output?"
    as conversational. The future-tense / "will be" framing tripped
    the LLM. This heuristic catches the canonical factual-question
    openers (where/when/how/what/why/which/who + auxiliary verb) and
    routes them through ``conversational`` deterministically before
    the LLM gate runs.

    Note: false positives are cheap (a steer phrased as a question
    just gets answered conversationally — the user can restate),
    while false negatives (mis-routing a factual question to
    refine_existing) trigger an unwanted re-plan with sticky-goal
    side effects. The asymmetry justifies a slightly aggressive
    pattern.
    """
    if not text:
        return False
    return bool(_FACTUAL_QUESTION_RE.search(text))

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
  about work that has ALREADY been done OR a question about the
  artefact's properties (where it lives, how it works, what it
  looks like, who made which part). These can be answered from the
  existing plan / completed_results / conversation history without
  running any new tasks.

  Examples (ALL conversational):
  * "where did you save the output?"
  * "where will the slides be saved?" (future-tense factual question)
  * "where is the file located?"
  * "what was the second slide?" / "what was the title?"
  * "what is this about?" / "what does it do?"
  * "did you use source X?" / "did you include Y?"
  * "is the presentation done?" / "is it ready?"
  * "how does the slideshow work?" / "how do I open it?"
  * "summarise what you did"
  * "tell me more about X" (X already covered by the plan)
  * "can you explain how Y works"

  Future tense ("will be"), present continuous, AND past tense are
  all conversational when the question targets the prior plan or
  its outputs. The grammatical tense does NOT change the bucket —
  what matters is whether the question can be ANSWERED without
  running new tasks.

- "refine_existing": the new input tweaks, extends, or revises the
  PRIOR PLAN — e.g. "make it funnier", "add a slide about Z",
  "also translate to Spanish", "change the title". The existing
  plan's completed tasks should be preserved; a small delta of new
  tasks is appended.

  This bucket also covers PIVOT directives that revise the topic
  while keeping the same artefact / output format — e.g. "forget
  X, tell me about Y instead", "no, don't do X, do Y", "switch the
  topic to Z", "instead of X let's do Y", "scratch that — Y", "I
  changed my mind — Y". These are NOT new_work: they keep the prior
  plan's structural constraints (slide count, output type, audience)
  and only swap the subject. Routing them to refine_existing
  preserves those constraints; routing them to new_work silently
  drops them.

Guidelines:
- Factual interrogatives that open with where/when/how/what/why/
  which/who + a state verb (is/are/will/did/does/was/were/can/could)
  are conversational by default — they ask about prior state, not
  request new work. Only classify them as refine_existing when the
  question contains an EXPLICIT directive ("can you ALSO add a
  slide about X?", "what if we changed the title?") — a bare
  question is just a question.
- When in doubt between "conversational" and "refine_existing",
  prefer "conversational" — the coordinator can always answer and
  the user can restate if they actually wanted new work.
- When in doubt between "refine_existing" and "new_work", prefer
  "refine_existing" — the prior plan's completed tasks give the
  refined plan a running start, and the prior plan's structural
  constraints survive the pivot.
- Steer-language openers ("forget", "instead", "no, don't ...",
  "scratch that", "actually", "wait, ...", "stop", "change the
  topic", "switch to") are strong refine_existing signals — only
  classify them as new_work when the user explicitly says they want
  to abandon the artefact entirely (e.g. "forget the slides, just
  give me the bullet points").

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

    Rules (in order):

    - No prior plan → always ``"new_work"`` (first turn or fresh
      conversation).
    - Prior plan exists AND :func:`_looks_like_steer` matches →
      ``"refine_existing"``. Steer-language openers ("forget X",
      "instead", "no, don't ..., do ...") overwhelmingly indicate the
      user is pivoting prior work, not asking a clarifying question
      and not requesting a wholly new workflow. Routing these through
      the refine pipeline preserves the prior plan's sticky context
      (slide count, output format) while letting the steerer emit
      ``DriftDetected(USER_STEER)`` and ``PlanRevised``. Without this
      branch the heuristic dropped steer messages into
      ``"conversational"`` (short input) or ``"new_work"`` (long
      input), both of which silently lost the constraints — see
      goldfive#270 E2E.
    - Prior plan exists AND :func:`_looks_like_factual_question`
      matches → ``"conversational"``. Factual interrogatives ("where
      will", "how does", "did you", "what is") are nearly always
      asking about prior work. Phase 2.X (goldfive#271 Gap 4)
      regression: "where will the slides be saved?" was mis-routed
      to ``refine_existing`` by the LLM gate despite the prompt's
      explicit example. Catching it heuristically routes around the
      LLM uncertainty.
    - Prior plan exists AND user_input is short (``< 20`` tokens) →
      ``"conversational"``. Short follow-ups that AREN'T steer-shaped
      are overwhelmingly questions about prior work.
    - Otherwise → ``"new_work"``.

    The LLM-backed :func:`classify_turn` is strictly more precise and
    should be preferred when a ``call_llm`` is available; this gate is
    a deterministic fallback for offline / mock-mode runs.
    """
    _ = conversation_id  # reserved for future heuristics
    if prior_plan is None or not prior_plan.tasks:
        return "new_work"
    if _looks_like_steer(user_input):
        return "refine_existing"
    if _looks_like_factual_question(user_input):
        return "conversational"
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
    sinks: list[Any] | None = None,
    run_id: str = "",
    session_id: str = "",
    sequence_fn: Callable[[], int] | None = None,
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

    # Phase 2.X / goldfive#271 Gap 4: heuristic short-circuit for the
    # two patterns where the LLM has historically misclassified —
    # explicit steer language and explicit factual interrogatives.
    # Both patterns are unambiguous; running the LLM for them just
    # adds latency and risks a wrong answer ("where will the slides
    # be saved?" got refine_existing in the validation E2E).
    #
    # Falls through to the LLM only when the heuristic returns
    # ``"new_work"`` — that's the broad "no signal" bucket where the
    # LLM's nuance pays off.
    if _looks_like_steer(user_input):
        log.info(
            "planner_gate.classify_turn: heuristic short-circuit "
            "(steer pattern) -> refine_existing; user_input_first=%r",
            user_input[:80],
        )
        return "refine_existing"
    if _looks_like_factual_question(user_input):
        log.info(
            "planner_gate.classify_turn: heuristic short-circuit "
            "(factual question) -> conversational; user_input_first=%r",
            user_input[:80],
        )
        return "conversational"

    user_prompt = _build_user_prompt(
        prior_plan=prior_plan,
        completed_results=completed_results,
        user_input=user_input,
        conversation_id=conversation_id,
    )
    from goldfive._llm_span import goldfive_llm_span

    # The planner gate is trajectory-level (decides whether the turn
    # goes through the planner at all), so ``target_agent_id`` /
    # ``target_task_id`` stay empty. The trigger context (user input +
    # conversation id) doubles as ``input_preview`` so harmonograf can
    # render "what did the gate see?" on the Gantt.
    gate_input_preview = (
        f"user_input: {user_input}\nconversation_id: {conversation_id}"
    )
    verdict: TurnClassification | None = None
    try:
        async with goldfive_llm_span(
            sinks=list(sinks or []),
            name="planner_gate_classify_turn",
            model=model,
            session_id=session_id,
            run_id=run_id,
            sequence_fn=sequence_fn,
            input_preview=gate_input_preview,
        ) as span:
            raw = await call_llm(_SYSTEM_PROMPT, user_prompt, model)
            verdict = _parse_verdict(raw)
            if verdict is None:
                span.output_preview = (
                    f"unparseable verdict; raw={raw!r:.200}"
                )
                span.decision_summary = (
                    "planner gate verdict: unparseable (falling back to heuristic)"
                )
            else:
                verdict_name = getattr(verdict, "name", str(verdict))
                span.output_preview = f"verdict={verdict_name}"
                span.decision_summary = f"planner gate verdict: {verdict_name}"
    except Exception as exc:  # noqa: BLE001
        log.warning("planner_gate.classify_turn: call_llm raised: %s", exc)
        return heuristic_classify_turn(
            prior_plan=prior_plan,
            completed_results=completed_results,
            user_input=user_input,
            conversation_id=conversation_id,
        )
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
