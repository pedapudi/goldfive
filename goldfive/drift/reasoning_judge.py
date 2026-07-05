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
the binding onto :class:`~goldfive.state_store.StateStore`
when confidence is above a threshold; the pin-resolution ladder reads
it back as a real signal.
"""

from __future__ import annotations

import dataclasses
import logging
import time
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
from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Goal, Plan, Task

log = logging.getLogger(__name__)


__all__ = [
    "AGENT_TREE_BLOCK_MAX_CHARS",
    "AGENT_TREE_SYSTEM_PROMPT_SUFFIX",
    "CallLLM",
    "PLAN_TASKS_SUMMARY_MAX_CHARS",
    "REASONING_JUDGE_MAX_OUTPUT_TOKENS",
    "REASONING_JUDGE_MAX_REASONING_INPUT_CHARS",
    "REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS",
    "REASONING_DRIFT_MAX_REASONING_CHARS",
    "REASONING_DRIFT_SYSTEM_PROMPT",
    "REASONING_DRIFT_TOOL_OBS_MAX_CHARS",
    "REASONING_DRIFT_TOOL_OBS_MAX_ENTRIES",
    "REASONING_DRIFT_USER_PROMPT_TEMPLATE",
    "ReasoningJudgeVerdict",
    "classify_reasoning_drift",
    "classify_reasoning_drift_with_focus",
    "format_available_agents_block",
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

# Per-callsite ``max_output_tokens`` budget (goldfive#271 follow-up).
# The judge returns a small JSON verdict ({"on_task": bool, "reason":
# "...", "severity": "...", ...}); 16384 covers Qwen 3.5 thinking-model
# preludes (think + answer share the same ceiling) without permitting
# unbounded essays. Empirical: v16 on Qwen 35B exhausted a 2048 budget
# inside ``<think>`` and returned ``raw=''``, so no drift fired and the
# cascade never started — see ``call_llm_budget`` docstring for sizing
# rationale.
REASONING_JUDGE_MAX_OUTPUT_TOKENS: int = 16384


# ``truncate_for_observability`` lives in :mod:`goldfive.drift.registry`
# (shared with the goal-drift detector via the same suffix); re-exported
# here under the historical name so the public ``__all__`` contract and
# downstream importers continue to work.


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

# iter-10 PR 3: three-state classification + lineage / tool-observation
# context. The template asks the judge for THREE decisions (classify,
# attribute, name provenance) instead of TWO. Sections in order:
# 1. PLAN TASKS               (existing)
# 2. CURRENTLY BOUND TASK     (existing)
# 3. Currently reasoning agent line  (existing — added in iter-9)
# 4. Task lineage line        (NEW — _format_task_lineage)
# 5. GOALS                    (existing)
# 6. RECENT TOOL OBSERVATIONS (NEW — _format_tool_observations)
# 7. REASONING                (existing, ≤1500 chars)
# 8. "Decide THREE things"    (NEW — classification, attribution, provenance)
# 9. JSON shape spec          (UPDATED — adds classification + provenance fields)
# 10. GUIDANCE block          (NEW — per-provenance criteria)
# 11. Severity guidance       (UPDATED for non-on_task)
REASONING_DRIFT_USER_PROMPT_TEMPLATE: str = (
    "You are assessing whether an autonomous agent's chain-of-thought "
    "is on task.\n\n"
    "PLAN TASKS (id -> title):\n{plan_tasks_summary}\n\n"
    "CURRENTLY BOUND TASK:\n{task_block}\n\n"
    "Currently reasoning agent: {current_agent_id}\n"
    "Task lineage: {task_lineage_block}\n\n"
    "GOALS:\n{goals_block}\n\n"
    "RECENT TOOL OBSERVATIONS (last {tool_obs_count}, oldest first):\n"
    "{tool_obs_block}\n\n"
    "REASONING (the agent's most recent chain-of-thought block):\n"
    "{reasoning_block}\n\n"
    "Decide THREE things:\n"
    "1. CLASSIFICATION. Which best describes the reasoning?\n"
    "   - on_task: it advances the BOUND TASK or the GOALS.\n"
    "   - justified_deviation: it departs from the BOUND TASK, but a recent\n"
    "     tool observation, surprising result, discovered dependency, or new\n"
    "     information visible above plausibly provoked the departure. The\n"
    "     provoking signal MUST be visible in the GOALS section or in the\n"
    "     RECENT TOOL OBSERVATIONS section above — the agent's CLAIM that\n"
    "     a signal exists (e.g. \"based on user instructions\", \"the user\n"
    "     asked for X\") is NOT itself evidence. Cross-check the agent's\n"
    "     claim against GOALS verbatim before accepting it.\n"
    "   - erroneous_deviation: it departs from the BOUND TASK with no such\n"
    "     provoking signal in the context above. This includes the case\n"
    "     where the agent CLAIMS user direction but the claimed topic or\n"
    "     scope does not appear in GOALS.\n"
    "2. ATTRIBUTION. Which task in the PLAN TASKS list is the reasoning\n"
    "   actually working on right now? Use the literal id, or '' when the\n"
    "   reasoning is off-plan.\n"
    "3. PROVENANCE. ONLY when classification is justified_deviation, name the\n"
    "   signal that justifies the deviation. Pick exactly one of:\n"
    "     tool_error | surprising_result | discovered_dependency | new_information\n"
    "   When classification is on_task or erroneous_deviation, set\n"
    '   provenance to "none".\n\n'
    "Reply with a single JSON object and nothing else, in this shape:\n"
    "{{\n"
    '  "classification": "on_task" | "justified_deviation" | "erroneous_deviation",\n'
    '  "severity": "info" | "warning" | "critical",\n'
    '  "reason": "one-sentence explanation",\n'
    '  "provenance": "tool_error" | "surprising_result" | '
    '"discovered_dependency" | "new_information" | "none",\n'
    '  "focused_task_id": "<id from PLAN TASKS, or \'\' if off-plan>",\n'
    '  "focus_confidence": 0.0-1.0,\n'
    '  "stated_intent": "one-sentence summary of what the agent says it '
    'is doing"\n'
    "}}\n\n"
    "GUIDANCE:\n"
    "- on_task includes clarifying sub-steps, exploring tradeoffs, and\n"
    "  working through calculations.\n"
    "- A tool_error provenance requires a recent tool observation with\n"
    "  is_error=true OR an error_message.\n"
    "- A surprising_result provenance requires a tool observation whose\n"
    "  result contradicts the reasoning's prior assumption.\n"
    "- A discovered_dependency provenance requires the reasoning to name a\n"
    "  prerequisite that was not in the plan or task description.\n"
    "- A new_information provenance requires the new information to be\n"
    "  grounded in EITHER (a) a fact surfaced by a recent tool result in\n"
    "  RECENT TOOL OBSERVATIONS, OR (b) the user's actual input as it\n"
    "  appears verbatim in GOALS. The agent's own statement that 'the\n"
    "  user asked for X' or 'based on user instructions' is NOT evidence\n"
    "  by itself — you MUST be able to find X mentioned in GOALS. If the\n"
    "  expansion topic does not appear in GOALS and is not surfaced by a\n"
    "  tool observation, classify as erroneous_deviation regardless of\n"
    "  what the agent claims the user said.\n"
    "- If you cannot point to a specific signal in the GOALS, RECENT TOOL\n"
    "  OBSERVATIONS, or REASONING (cross-checked against the above) to\n"
    "  justify a deviation, classify it as erroneous_deviation.\n\n"
    "Severity guidance when classification is non-on_task:\n"
    "  info     = mild deviation that may self-correct next turn.\n"
    "  warning  = clear deviation that deserves a refine.\n"
    "  critical = proposing to abandon the goal entirely.\n\n"
    "focused_task_id MUST be the literal id of one of the listed plan "
    "tasks, or an empty string when the reasoning is not working on "
    "any plan task. focus_confidence is your subjective certainty in "
    "the attribution: 1.0 when the reasoning explicitly names the "
    "task, 0.0 when you are guessing."
)


# Liberal JSON extraction + goals rendering live in
# :mod:`goldfive.drift.registry` so this judge and the goal-drift judge
# share one implementation. The aliased imports at the top of this
# module preserve the historical private names (so external test
# fixtures mocking ``reasoning_judge._parse_response`` continue to work)
# without re-declaring the function bodies here.


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


# ---------------------------------------------------------------------------
# iter-10 PR 3: lineage + recent-tool-observation prompt blocks
# ---------------------------------------------------------------------------
#
# Lineage and recent tool observations are surfaced to the judge as
# additional CONTEXT, never as a structural pre-gate. The LLM still
# decides; the new context just lets it distinguish a provoked
# deviation (justified_deviation) from an unprovoked one
# (erroneous_deviation). See iter10-design.md §3.3 / §5 for the
# rationale.


def _format_task_lineage(
    task_id: str,
    lineage: dict[str, set[str]] | None,
    current_agent_id: str,
) -> str:
    """Render the task-lineage block for the judge prompt.

    ``lineage`` is the per-task ``{task_id: set(agent_id)}`` map kept
    on :class:`~goldfive.types.Session` and populated by the ADK plugin
    (assignee + every observed AgentTool ``to_agent`` for the task).
    The empty / missing case renders ``"(no task lineage observed)"``
    so the prompt shape is invariant for tests; populated cases render
    a one-line set with a suffix indicating whether the currently
    reasoning agent is structurally inside the bound task's delegation
    tree.

    The judge uses this as one signal among several — never as the
    sole basis for an on-task vs deviation decision (per §5 of the
    design doc).
    """
    if not task_id or not lineage:
        return "(no task lineage observed)"
    raw = lineage.get(task_id)
    if not raw:
        return "(no task lineage observed)"
    agents = sorted(str(a) for a in raw if a)
    if not agents:
        return "(no task lineage observed)"
    line = ", ".join(agents)
    in_lineage = current_agent_id in agents
    suffix = (
        f" — {current_agent_id} IS in this lineage"
        if in_lineage
        else f" — {current_agent_id} is NOT in this lineage"
    )
    return f"observed agents for this task: {line}{suffix}"


# Cap the rendered tool-observations block at 1500 chars (matches the
# REASONING block cap). Per-entry truncation is already enforced at
# write time by ``DefaultSteerer.note_tool_observation`` (args_preview
# 240 chars / result_preview 480 chars), so this helper only bounds
# the total block size.
REASONING_DRIFT_TOOL_OBS_MAX_CHARS: int = 1500
REASONING_DRIFT_TOOL_OBS_MAX_ENTRIES: int = 8


def _format_tool_observations(
    obs: list[dict[str, Any]] | None,
    *,
    task_id: str,
    max_entries: int = REASONING_DRIFT_TOOL_OBS_MAX_ENTRIES,
    max_chars: int = REASONING_DRIFT_TOOL_OBS_MAX_CHARS,
) -> tuple[str, int]:
    """Render the recent-tool-observations prompt block.

    Returns ``(rendered_block, count_used)`` where ``count_used`` is the
    number of entries that actually made it into the block (so the
    template can stamp it onto the "(last N, oldest first)" header).

    Filters to the current task's observations first; falls back to a
    global slice when the per-task slice is empty (the §3.4 design
    decision: per-task is more relevant on average, but a deviation
    rooted in an earlier task's tool result remains useful context, so
    don't drop it entirely). Per-entry truncation is already done at
    write time; this helper enforces the total ``max_chars`` cap so
    the prompt stays bounded.
    """
    if not obs:
        return "(no recent tool observations)", 0
    scoped = [e for e in obs if isinstance(e, dict) and e.get("task_id") == task_id]
    if not scoped:
        scoped = [e for e in obs if isinstance(e, dict)]
    if not scoped:
        return "(no recent tool observations)", 0
    tail = scoped[-max_entries:]
    lines: list[str] = []
    rendered_chars = 0
    for e in tail:
        marker = "ERROR" if e.get("is_error") else "ok"
        agent = str(e.get("agent_name", "") or "?")
        tool = str(e.get("tool_name", "") or "?")
        args_preview = str(e.get("args_preview", "") or "")
        result_preview = str(e.get("result_preview", "") or "")
        line = f"- {marker} {agent} {tool}({args_preview}) -> {result_preview}"
        # +1 for the join newline between lines.
        if rendered_chars + len(line) + 1 > max_chars and lines:
            break
        lines.append(line)
        rendered_chars += len(line) + 1
    if not lines:
        return "(no recent tool observations)", 0
    return "\n".join(lines), len(lines)


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
    available_agents: Any = None,
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

    ``available_agents`` (goldfive#244) is the structured tree (same
    shape as
    :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`).
    When provided AND non-empty, lines are extended with the task's
    ``assignee_agent_id`` and a comma-separated list of that
    assignee's known sub-agents — so the judge can see at a glance
    that a coordinator-style assignee is allowed to delegate to its
    sub-agents for that task. Default ``None`` produces the byte-
    identical pre-#244 ``id -> title`` rendering required for back-
    compat with existing prompt assertions and external callers.
    """
    if plan is None or not getattr(plan, "tasks", None):
        return "(no plan tasks)"
    # Build a parent -> [child names] index from the agent tree once
    # so per-line lookup is O(1). Falsy / non-tree inputs short-circuit
    # to the legacy render (byte-identical pre-#244).
    children_by_parent: dict[str, list[str]] = {}
    has_tree = False
    if available_agents and isinstance(available_agents, (list, tuple)):
        for entry in available_agents:
            if isinstance(entry, dict) and "name" in entry:
                has_tree = True
                parent = str(entry.get("parent", "") or "")
                name = str(entry.get("name", "") or "")
                if not name or not parent:
                    continue
                children_by_parent.setdefault(parent, []).append(name)
    lines: list[str] = []
    rendered_chars = 0
    truncated = 0
    tasks = list(plan.tasks)
    for i, task in enumerate(tasks):
        tid = str(getattr(task, "id", "") or "")
        title = str(getattr(task, "title", "") or "(untitled)")
        if has_tree:
            assignee = str(getattr(task, "assignee_agent_id", "") or "").strip()
            base = f"- {tid} -> {title}" if tid else f"- (no id) -> {title}"
            if assignee:
                kids = children_by_parent.get(assignee, [])
                if kids:
                    line = (
                        f"{base} [assignee={assignee}; "
                        f"delegates to: {', '.join(kids)}]"
                    )
                else:
                    line = f"{base} [assignee={assignee}]"
            else:
                line = base
        else:
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
# goldfive#244 — agent-tree section
# ---------------------------------------------------------------------------
#
# When the steerer can resolve the wrapped agent tree (typically via
# :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`), we
# render a separate "AGENT TREE" section into the judge prompt and append
# a one-paragraph clarification to the system prompt. Without this
# context the judge cannot distinguish a coordinator-style agent
# delegating to its known sub-agent (legitimate goldfive.wrap usage —
# see ``feedback_goldfive_wrap_contract.md``) from arbitrary off-plan
# work, and it tends to fire OFF_TOPIC for the legitimate case (see
# brussels-sprouts session OFF_TOPIC at 0:26: "agent deviates from the
# plan by invoking an unlisted web_developer_agent instead of proceeding
# to the draft_slides task" — web_developer_agent was a known sub-agent
# of coordinator_agent in the wrapped tree).
#
# This is purely additional CONTEXT — the judge still decides. We do
# NOT mutate the existing user-prompt template (existing tests render
# it via ``.format(...)`` with the current key set; introducing a new
# placeholder breaks them) and we do NOT add regex / structural pre-
# gates. The default-``available_agents=None`` path produces the
# byte-identical pre-#244 prompt so existing tests, classifications,
# and external callers see no behavioural change.

#: Hard cap on the rendered AGENT TREE section. ~1200 chars (~300
#: tokens) keeps the prompt bounded for very large trees while leaving
#: enough room to enumerate ~25-40 named agents at typical name length.
AGENT_TREE_BLOCK_MAX_CHARS: int = 1200


#: System-prompt suffix appended ONLY when ``available_agents`` is
#: provided. Strengthens the on-task definition so the judge does not
#: treat sub-agent delegation as an off-plan deviation. Kept as a
#: separate constant so operators can A/B the wording without forking
#: :data:`REASONING_DRIFT_SYSTEM_PROMPT`.
AGENT_TREE_SYSTEM_PROMPT_SUFFIX: str = (
    " The user prompt may include an AGENT TREE section listing the "
    "wrapped agents and their parent/child relationships. If the agent "
    "in question invokes a known sub-agent (per that tree) to perform "
    "its assigned task, treat that as ON-TASK execution of the bound "
    "task — delegation is normal coordinator behaviour and is NOT a "
    "deviation. Mark a deviation only when the agent invokes something "
    "not in the tree, or when its reasoning wanders semantically away "
    "from the bound task and goals."
)


def format_available_agents_block(
    available_agents: Any,
    *,
    max_chars: int = AGENT_TREE_BLOCK_MAX_CHARS,
) -> str:
    """Render the AGENT TREE section listing parent/child relationships.

    ``available_agents`` mirrors the shape exposed by
    :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`: a
    list of dicts with at least a ``name`` key and optionally
    ``parent``, ``role``, ``kind``, ``depth``. The legacy ``list[str]``
    form (plain agent names without tree edges) is also accepted for
    parity with the planner's :func:`LLMPlanner._render_agents_block`;
    string entries render as bullets without a "delegates to" suffix
    because we don't have edge data.

    Empty / ``None`` inputs yield ``""`` so the caller can short-circuit
    on the empty string and skip appending the section entirely (the
    default-None code path stays byte-identical to pre-#244).

    Truncation: when the rendered text would exceed ``max_chars`` we
    drop suffix lines and append a ``"... [N more agent(s) elided]"``
    marker so the judge knows the listing is incomplete. Bounded so a
    pathologically deep tree cannot blow the prompt budget.
    """
    if not available_agents:
        return ""
    if not isinstance(available_agents, (list, tuple)):
        return ""
    structured: list[dict[str, Any]] = []
    flat_names: list[str] = []
    for entry in available_agents:
        if isinstance(entry, dict) and "name" in entry:
            structured.append(entry)
        elif isinstance(entry, str) and entry:
            flat_names.append(entry)
    if not structured and not flat_names:
        return ""
    lines: list[str] = []
    truncated = 0
    rendered_chars = 0
    if structured:
        # Index children by parent name (order-preserving).
        children_by_parent: dict[str, list[str]] = {}
        for entry in structured:
            parent = str(entry.get("parent", "") or "")
            name = str(entry.get("name", "") or "")
            if not name:
                continue
            children_by_parent.setdefault(parent, []).append(name)
        for i, entry in enumerate(structured):
            name = str(entry.get("name", "") or "")
            if not name:
                continue
            kids = children_by_parent.get(name, [])
            role = str(entry.get("role", "") or "")
            kind = str(entry.get("kind", "") or "")
            meta_bits: list[str] = []
            if role:
                meta_bits.append(f"role={role}")
            if kind:
                meta_bits.append(f"kind={kind}")
            meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
            if kids:
                line = f"- {name}{meta} delegates to: {', '.join(kids)}"
            else:
                line = f"- {name}{meta}"
            if rendered_chars + len(line) + 1 > max_chars and lines:
                truncated = len(structured) - i
                break
            lines.append(line)
            rendered_chars += len(line) + 1
    else:
        for i, name in enumerate(flat_names):
            line = f"- {name}"
            if rendered_chars + len(line) + 1 > max_chars and lines:
                truncated = len(flat_names) - i
                break
            lines.append(line)
            rendered_chars += len(line) + 1
    if truncated > 0:
        lines.append(f"... [{truncated} more agent(s) elided]")
    if not lines:
        return ""
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

    iter-10 (PR 1) adds two scaffolding fields ahead of the three-state
    classification work:

    * ``classification`` — three-state verdict, one of
      ``"on_task"``, ``"justified_deviation"``, ``"erroneous_deviation"``,
      or the empty string. The empty string is the quiet-fail sentinel:
      callers MUST treat it as identical to ``"on_task"`` for routing
      so the pre-iter-10 fail-quiet contract from #143 / #226 is
      preserved. Defaults to ``""`` so PR 1 ships pure scaffolding —
      until PR 3 lands the parser still produces an empty string here.
    * ``provenance`` — only meaningful when
      ``classification == "justified_deviation"``. One of
      ``"tool_error"``, ``"surprising_result"``,
      ``"discovered_dependency"``, ``"new_information"``, or the empty
      string. The provenance enum names the signal in the judge prompt
      that justified the deviation; an empty / unrecognised value on a
      ``"justified_deviation"`` verdict is treated as malformed by PR 3
      (demoted to ``"erroneous_deviation"``). Defaults to ``""`` for
      back-compat in PR 1.
    """

    drift: DriftEvent | None
    focused_task_id: str = ""
    focus_confidence: float = 0.0
    stated_intent: str = ""
    # iter-10 PR 1 additions — defaults preserve back-compat. Behaviour
    # change ships in PR 3 (parser) + PR 4 (routing).
    classification: str = ""
    provenance: str = ""
    # Judge-scheduling guards: measurement fields. ``judge_ran`` is
    # True iff the judge LLM was actually dispatched (False on the
    # empty-reasoning early return, the embedding-only path, and
    # ``mode="off"``), so callers can distinguish "quiet-fail sentinel"
    # (``judge_ran and not classification``) from "judge never ran".
    # ``elapsed_ms`` mirrors the value stamped on the
    # ``ReasoningJudgeInvoked`` event; 0 when ``judge_ran`` is False.
    judge_ran: bool = False
    elapsed_ms: int = 0


# Map the judge's ``severity`` string to a :class:`DriftSeverity`. Missing
# or unknown values fall through to INFO: the drift verdict is still
# emitted (never silently swallowed), but a malformed severity string
# must not be promotion-eligible —
# :meth:`DriftObserver._should_promote_to_steer` gates on WARNING-and-up.
_SEVERITY_MAP: dict[str, DriftSeverity] = {
    "info": DriftSeverity.INFO,
    "warning": DriftSeverity.WARNING,
    "critical": DriftSeverity.CRITICAL,
}


def _severity_from_verdict(raw: Any) -> DriftSeverity:
    if isinstance(raw, str):
        severity = _SEVERITY_MAP.get(raw.strip().lower())
        if severity is not None:
            return severity
    log.debug(
        "classify_reasoning_drift: severity %r missing or unrecognised; "
        "defaulting to INFO",
        raw,
    )
    return DriftSeverity.INFO


# iter-10 PR 3: three-state classification + provenance enums. Keep
# these as frozenset literals so misspellings are caught at parse
# time (the parser strips + lowercases before membership-checking).
_VALID_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"on_task", "justified_deviation", "erroneous_deviation"}
)
_VALID_PROVENANCES: frozenset[str] = frozenset(
    {"tool_error", "surprising_result", "discovered_dependency", "new_information"}
)


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
    task_lineage: dict[str, set[str]] | None = None,
    recent_tool_observations: list[dict[str, Any]] | None = None,
    available_agents: list[str] | list[dict[str, Any]] | None = None,
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
    for that section. goldfive#244 added an optional ``available_agents``
    keyword that, when provided, surfaces the wrapped agent tree
    (parent → sub-agents) so the judge does not flag legitimate
    coordinator → sub-agent delegation as off-topic.
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
        task_lineage=task_lineage,
        recent_tool_observations=recent_tool_observations,
        available_agents=available_agents,
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
    task_lineage: dict[str, set[str]] | None = None,
    recent_tool_observations: list[dict[str, Any]] | None = None,
    available_agents: list[str] | list[dict[str, Any]] | None = None,
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
    # goldfive#245 — capture the plan revision the judge is observing
    # BEFORE we render the prompt or await the LLM. Stamped onto the
    # drift below; the dispatch-time gate in
    # :meth:`DefaultSteerer._handle_drift` drops verdicts whose
    # observed revision is older than the live plan's.
    observed_revision_index = int(getattr(plan, "revision_index", 0) or 0)
    # iter-10 PR 3: lineage + recent tool observations are passed as
    # CONTEXT to the LLM (per §3.3 / §5 — never as a structural pre-gate).
    task_lineage_block = _format_task_lineage(
        current_task_id, task_lineage, current_agent_id or "(unknown)"
    )
    tool_obs_block, tool_obs_count = _format_tool_observations(
        recent_tool_observations, task_id=current_task_id
    )
    user = template.format(
        plan_tasks_summary=format_plan_tasks_summary(
            plan, available_agents=available_agents
        ),
        goals_block=_format_goals(goals),
        task_block=_format_task(task),
        reasoning_block=_format_reasoning(reasoning),
        current_agent_id=current_agent_id or "(unknown)",
        task_lineage_block=task_lineage_block,
        tool_obs_block=tool_obs_block,
        tool_obs_count=tool_obs_count,
    )
    # goldfive#244: when the steerer can resolve the wrapped agent tree,
    # append a separate AGENT TREE section to the user prompt and a
    # one-paragraph clarification to the system prompt. The default-None
    # path skips both so existing callers / tests see the byte-identical
    # pre-#244 prompt. We append (vs templating) so existing
    # ``REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(...)`` call sites
    # don't need a new key. See module-level rationale at
    # :data:`AGENT_TREE_SYSTEM_PROMPT_SUFFIX` for why this is purely
    # additional context, never a structural pre-gate.
    agent_tree_block = format_available_agents_block(available_agents)
    if agent_tree_block:
        user = (
            f"{user}\n\nAGENT TREE (parent → sub-agents; sub-agent "
            f"delegation by a listed parent is ON-TASK execution of "
            f"the bound task):\n{agent_tree_block}"
        )
        system = f"{system}{AGENT_TREE_SYSTEM_PROMPT_SUFFIX}"
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
    # iter-10 PR 3 — three-state classification + provenance. Empty
    # strings are the quiet-fail sentinel (call raised, malformed JSON,
    # neither classification nor legacy on_task readable). The parser
    # below populates these post-call.
    classification_parsed: str = ""
    provenance_parsed: str = ""
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
                # iter-10 PR 3: three-state classification (§2.4 rules
                # 1 + 2). Read ``classification`` first; fall back to
                # the legacy ``on_task`` boolean when it's missing or
                # unrecognised. Both paths converge on a populated
                # ``classification_parsed`` (or "" for quiet-fail).
                classification_raw = parsed.get("classification", "")
                if isinstance(classification_raw, str):
                    candidate = classification_raw.strip().lower()
                    if candidate in _VALID_CLASSIFICATIONS:
                        classification_parsed = candidate
                if not classification_parsed:
                    # Legacy fallback — the pre-iter-10 prompt only
                    # asked for a bool. Custom prompt-template
                    # overrides operators ship may still produce that
                    # shape, and we promised back-compat in §2.4 rule 2.
                    on_task_legacy = parsed.get("on_task")
                    if isinstance(on_task_legacy, bool):
                        classification_parsed = (
                            "on_task" if on_task_legacy else "erroneous_deviation"
                        )
                # ``on_task_parsed`` mirrors the classification for the
                # legacy proto event payload AND for the span
                # output_preview / decision_summary below — set it as
                # soon as we have a classification (success path), or
                # leave it None (quiet-fail path).
                if classification_parsed == "on_task":
                    on_task_parsed = True
                elif classification_parsed in (
                    "justified_deviation",
                    "erroneous_deviation",
                ):
                    on_task_parsed = False
                # Reason + severity are useful regardless of which
                # branch we're on (the prompt allows them empty on
                # on_task; the parser below treats empty reason
                # gracefully when constructing drift detail).
                if classification_parsed:
                    reason = str(parsed.get("reason", "") or "").strip()
                    if classification_parsed != "on_task":
                        severity_str = _severity_from_verdict(
                            parsed.get("severity")
                        ).value.lower()
                # iter-10 PR 3 §2.4 rule 3: provenance validation +
                # demotion. justified_deviation REQUIRES one of the
                # four enum values (tool_error | surprising_result |
                # discovered_dependency | new_information). Anything
                # else (missing key, "none", unknown free-text) demotes
                # the verdict to erroneous_deviation. Doing this inside
                # the span scope so ``span.output_preview`` /
                # ``decision_summary`` reflect the post-demotion shape.
                # The actual INFO-level demotion log fires below in
                # the drift-construction branch.
                if classification_parsed == "justified_deviation":
                    provenance_raw = parsed.get("provenance", "")
                    candidate = ""
                    if isinstance(provenance_raw, str):
                        candidate = provenance_raw.strip().lower()
                    if candidate in _VALID_PROVENANCES:
                        provenance_parsed = candidate
                    else:
                        # Demote — keep raw value around for the
                        # post-with-block log message.
                        classification_parsed = "erroneous_deviation"
                        provenance_parsed = ""
                        on_task_parsed = False
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
                # iter-10 PR 3: span's output_preview + decision_summary
                # surface the three-state classification (post-demotion,
                # since the §2.4 rule-3 check ran above inside this
                # span scope). For justified_deviation the verdict
                # string carries the provenance suffix so the
                # harmonograf Gantt shows the provoking signal at a
                # glance.
                span.output_preview = (
                    f"classification={classification_parsed or '(none)'}, "
                    f"provenance={provenance_parsed or '(none)'}, "
                    f"on_task={on_task_parsed}, "
                    f"severity={severity_str or '(none)'}, "
                    f"reason={reason or '(none)'}"
                )
                if classification_parsed == "on_task":
                    # Keep the legacy "on-task" wording so existing
                    # observability assertions (and harmonograf's
                    # display-string scrapers) continue to match. The
                    # output_preview above already exposes the
                    # canonical ``classification=on_task`` axis for
                    # callers that prefer the new field.
                    verdict_str = "on-task"
                elif classification_parsed == "justified_deviation":
                    verdict_str = f"justified_deviation ({provenance_parsed})"
                elif classification_parsed == "erroneous_deviation":
                    sev_suffix = (
                        f" [{severity_str.upper()}]" if severity_str else ""
                    )
                    verdict_str = f"erroneous_deviation{sev_suffix}"
                elif on_task_parsed:
                    # Defensive: only reached if classification fell
                    # through to a falsy string but on_task_parsed was
                    # set — keep legacy text for the proto event mirror.
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
        # iter-10 PR 3 §2.4: three-state classification routing. The
        # parser inside the with-block populated
        # ``classification_parsed`` (legacy fallback applied) and ran
        # the §2.4 rule-3 provenance demotion when needed. This branch
        # only logs + builds the drift event.
        if not classification_parsed:
            # Quiet-fail: neither a recognised ``classification`` nor a
            # boolean ``on_task`` legacy field. Empty verdict, no drift.
            log.debug(
                "classify_reasoning_drift: parsed=%r lacks both "
                "'classification' and boolean 'on_task' keys; no drift emitted",
                parsed,
            )
        elif classification_parsed == "on_task":
            log.debug(
                "classify_reasoning_drift: judge says on-track (reason=%r)",
                reason,
            )
        elif classification_parsed == "justified_deviation":
            severity_enum = _severity_from_verdict(parsed.get("severity"))
            log.info(
                "classify_reasoning_drift: justified deviation "
                "(provenance=%s, severity=%s, reason=%r); emitting "
                "JUSTIFIED_DEVIATION event",
                provenance_parsed,
                severity_enum.value,
                reason,
            )
            detail_body = reason or "(judge returned no reason)"
            detail = f"justified deviation ({provenance_parsed}): {detail_body}"
            drift = DriftEvent(
                kind=DriftKind.JUSTIFIED_DEVIATION,
                severity=severity_enum,
                detail=detail,
                current_task_id=current_task_id,
                current_agent_id=current_agent_id,
                raw=reasoning,
                trigger_input=truncate_for_observability(
                    reasoning, REASONING_JUDGE_MAX_REASONING_INPUT_CHARS
                ),
                observed_revision_index=observed_revision_index,
            )
        else:
            # erroneous_deviation — same wire shape as the pre-iter-10
            # OFF_TOPIC path. This branch ALSO catches a
            # justified_deviation that was demoted by the §2.4 rule-3
            # check above; in that case the original raw provenance
            # value is gone (we logged the demotion when it fired), and
            # the rendered drift looks identical to a model-emitted
            # erroneous_deviation. That's intentional — once demoted,
            # the verdict IS erroneous_deviation by every downstream
            # surface.
            assert classification_parsed == "erroneous_deviation", (
                classification_parsed
            )
            severity_enum = _severity_from_verdict(parsed.get("severity"))
            # Surface the demotion at INFO so operators can grep the
            # rate of "model claimed justified but couldn't name a
            # provenance" in production logs. The branch above
            # (provenance check inside the with-block) intentionally
            # does NOT log so the message contains the original raw
            # value and the parsed-after-demotion classification — both
            # readable here.
            raw_classification = ""
            if isinstance(parsed.get("classification"), str):
                raw_classification = parsed["classification"].strip().lower()
            if raw_classification == "justified_deviation":
                provenance_raw_log = parsed.get("provenance", "")
                log.info(
                    "classify_reasoning_drift: justified_deviation "
                    "demoted to erroneous_deviation — provenance=%r is "
                    "missing or not in the allowed enum "
                    "(tool_error|surprising_result|"
                    "discovered_dependency|new_information)",
                    provenance_raw_log,
                )
            log.info(
                "classify_reasoning_drift: drift detected "
                "(severity=%s, reason=%r); emitting OFF_TOPIC event",
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
                observed_revision_index=observed_revision_index,
            )
    # Emit ReasoningJudgeInvoked on every invocation, regardless of
    # verdict. Done after the drift decision so the event carries the
    # parsed outcome but independent of it — on-task, off-task, and
    # plumbing-failure paths all produce an observability event.
    if sink is not None:
        # iter-10 PR 3: pass the post-demotion classification onto the
        # ReasoningJudgeInvoked event payload. ``on_task`` is the
        # legacy mirror — kept for back-compat with existing harmonograf
        # columns. The mirror is computed from the final
        # ``classification_parsed`` so it stays consistent with the new
        # field even after a §2.4 rule-3 demotion.
        on_task_for_event = (
            classification_parsed == "on_task"
            if classification_parsed
            else (drift is None)
        )
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
            on_task=on_task_for_event,
            severity=severity_str,
            reason=reason,
            classification=classification_parsed,
            focused_task_id=focused_task_id_parsed,
            focus_confidence=focus_confidence_parsed,
            stated_intent=stated_intent_parsed,
            provenance=provenance_parsed,
        )
    return ReasoningJudgeVerdict(
        drift=drift,
        focused_task_id=focused_task_id_parsed,
        focus_confidence=focus_confidence_parsed,
        stated_intent=stated_intent_parsed,
        classification=classification_parsed,
        provenance=provenance_parsed,
        judge_ran=True,
        elapsed_ms=elapsed_ms,
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
    classification: str = "",
    focused_task_id: str = "",
    focus_confidence: float = 0.0,
    stated_intent: str = "",
    provenance: str = "",
) -> None:
    """Build and emit a ``ReasoningJudgeInvoked`` envelope onto ``sink``.

    Broken sinks must not break the run: any exception is caught and
    logged at WARNING. Proto-import failures are handled the same way
    so a partially-regenerated tree (``make proto`` not re-run) does
    not crash the judge path.

    ``classification`` is the iter-10 three-state verdict string; PR 1
    accepts the kwarg with a default of ``""`` so existing call sites
    don't break. PR 3 starts populating it from the parser.

    ``focused_task_id`` / ``focus_confidence`` / ``stated_intent`` /
    ``provenance`` mirror the same-named ``ReasoningJudgeVerdict``
    fields onto the wire — parsed and clamped by the caller; defaults
    keep older call sites working.
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
        payload.classification = classification
        payload.focused_task_id = focused_task_id
        payload.focus_confidence = float(focus_confidence)
        payload.stated_intent = stated_intent
        payload.provenance = provenance
        await sink.emit(evt)
    except Exception as exc:  # noqa: BLE001 - observability must never break
        log.warning(
            "classify_reasoning_drift: sink.emit raised %s; ReasoningJudgeInvoked dropped",
            exc,
        )


# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------
#
# The reasoning judge emits TWO drift kinds: ``OFF_TOPIC`` (the
# pre-iter-10 verdict, also the destination for ``erroneous_deviation``
# classifications) and ``JUSTIFIED_DEVIATION`` (iter-10 PR 3, fired when
# the judge attributes a deviation to a recent tool observation or
# user-input signal). Both share the same classifier function, the
# same config, and the same caller-facing entry point — we register
# both for completeness so dispatch lookup by kind works for either.


_REASONING_JUDGE_CONFIG: DetectorConfig = DetectorConfig(
    uses_llm=True,
    max_input_chars=REASONING_JUDGE_MAX_REASONING_INPUT_CHARS,
    max_output_tokens=REASONING_JUDGE_MAX_OUTPUT_TOKENS,
    disable_thinking=True,
)


_register(
    DriftKind.OFF_TOPIC,
    classify_reasoning_drift,
    _REASONING_JUDGE_CONFIG,
    is_async=True,
)
_register(
    DriftKind.JUSTIFIED_DEVIATION,
    classify_reasoning_drift,
    _REASONING_JUDGE_CONFIG,
    is_async=True,
)
