"""Planner implementations for goldfive.

A ``Planner`` turns a list of :class:`Goal` s into a :class:`Plan` (a DAG
of :class:`Task` s) and, after the plan starts running, produces a
revised plan in response to drift. v0.1 is strictly goal-oriented:
``generate`` takes ``list[Goal]`` rather than raw user input. Deriving
goals from free text is out-of-scope here and is handled by
``GoalDeriver`` (issue #8).

Two concrete implementations ship in this module:

* :class:`PassthroughPlanner` — ``generate`` and ``refine`` always
  return ``None``. Useful as a default when planning is opt-in.
* :class:`LLMPlanner` — delegates to a caller-supplied async
  ``call_llm`` callable, parses its JSON output, and is robust to
  markdown code fences.

Neither implementation raises into the host: any LLM or parse failure
is logged and becomes ``None``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from goldfive.types import (
    GOAL_SOURCE_USER_STEER,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    ObservedAction,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
    bump_revision,
    replace_task,
)

# Task statuses that are "done" from a steering perspective — they are
# preserved verbatim across a USER_STEER delete-and-replan.
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

log = logging.getLogger("goldfive.planner")


# Sentinel error message used by the refine / generate / steer retry loops
# to signal "the LLM returned nothing usable" (empty string or non-string).
# Detected by the loop body so we can short-circuit the retry: an empty
# response is almost always a small-model artefact (Qwen 2B exhausting its
# budget on thinking tokens with no final answer, see goldfive#182). Each
# retry doubles cost without changing the outcome, so we treat empty as
# terminal "no signal" and let the caller fall back to its no-drift /
# no-revision branch.
_EMPTY_RESPONSE_ERROR: str = "empty or non-string LLM response"


# ---------------------------------------------------------------------------
# System prompts (goal-oriented variants ported from harmonograf_client)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a task-planning assistant for a multi-agent system. Your job
is to produce the COMPLETE end-to-end execution plan that satisfies a
set of high-level GOALS before any agent begins work. Do not plan just
the first step — enumerate every task you expect the system to perform
from start to finish, so a human can see the whole shape of the
workflow upfront.

Requirements for the plan:

1. COMPREHENSIVENESS. Decompose the goals into the full DAG of tasks
   needed to satisfy every goal end-to-end. A typical plan has between
   5 and 20 tasks. Smaller is OK only for genuinely trivial goal sets;
   larger is OK for complex multi-phase work. Do not stop at "first
   research, then draft" — include verification, revision, handoffs,
   synthesis, and any follow-up the goals imply.

2. GOAL COVERAGE. Every goal in the provided list must be addressed by
   at least one task. Keep the mapping clear: prefer task descriptions
   that name the goal(s) they advance.

3. DEPENDENCIES. Use `edges` to declare ordering between tasks: an
   edge {from_task_id, to_task_id} means the "from" task must finish
   before the "to" task starts. Parallel tasks (no edge between them)
   may run concurrently. Build a real DAG, not a single linear chain,
   when independent work exists.

4. ASSIGNMENTS. Do NOT populate `assignee_agent_id`; leave it as the
   empty string. The framework will populate it observationally when
   a delegation actually happens (goldfive#252). The available_agents
   block is supplied for context only — describe each task in terms of
   the work that needs doing, not which agent will perform it.

5. STABILITY. Task ids must be short, unique, and stable strings
   (e.g. "research", "draft_intro", "review_final"). Descriptions
   should be one sentence describing what "done" looks like for the
   task.

6. SUMMARY. Provide a one-sentence `summary` describing the overall
   goal of the plan, as if writing a PR title.

Respond with a single JSON object and NOTHING ELSE — no prose, no
markdown fences. Schema:

{
  "summary": "<one-sentence description of the overall plan>",
  "tasks": [
    {
      "id": "research",
      "title": "short human-readable title",
      "description": "one sentence defining 'done' for this task"
    }
  ],
  "edges": [
    {"from_task_id": "research", "to_task_id": "draft"}
  ]
}
"""


# goldfive#214 (iter-7): the canonical SUPERSESSION INVARIANT block.
# Embedded verbatim in the user-facing refine / steer / looping system
# prompts so the planner LLM receives one consistent instruction
# regardless of drift kind. Backstops at the merge layer:
#   * _backfill_retry_supersedes covers retry_X / X_v2 patterns when
#     the LLM forgets supersedes (legacy goldfive-emitted patterns).
#   * _normalize_supersession_kinds coerces wrong supersedes_kind based
#     on old-task status.
#   * _has_live_replacement causal tier consumes the supersedes link.
# These run unchanged. Prompt strengthening reduces how often the
# safety nets need to fire — it does NOT replace them.
_SUPERSESSION_INVARIANT = (
    "SUPERSESSION INVARIANT — REUSE-OR-SUPERSEDE (mutually exclusive):\n"
    "\n"
    "Whenever a task in your response carries forward, retries, fixes, or\n"
    "re-does work from a prior task, you MUST pick exactly one of these two\n"
    "shapes — never both, never neither:\n"
    "\n"
    "  (a) REUSE THE PRIOR ID. The continuing task keeps the prior task's\n"
    "      `id`. Retitle / rewrite / reassign as needed; identity is the id.\n"
    "      Do NOT set `supersedes` (id reuse already encodes continuity).\n"
    "      A COMPLETED task cannot regress to PENDING under a reused id —\n"
    "      if you need to redo completed work, use shape (b) with\n"
    "      `supersedes_kind: CORRECT`, not id reuse.\n"
    "\n"
    "  (b) MINT A NEW ID + SUPERSEDES. The continuing task gets a fresh `id`\n"
    "      AND `\"supersedes\": \"<prior_id>\"` AND `\"supersedes_kind\"`. Use this\n"
    "      whenever the new id differs from the prior id for ANY reason —\n"
    "      terminal failure (FAILED/CANCELLED), terminal cancellation,\n"
    "      structural retry (`retry_X`, `X_v2`), corrective fix\n"
    "      (`fix_X`, `redo_X`, `revised_X`, etc.), renamed evolution, or\n"
    "      replacement under a different agent. The naming convention does\n"
    "      NOT matter; if a new task semantically replaces an older one and\n"
    "      its `id` is different, `supersedes` is REQUIRED.\n"
    "\n"
    "`supersedes_kind` rule:\n"
    "  * REPLACE — superseded task is PENDING / RUNNING / BLOCKED /\n"
    "    FAILED / CANCELLED (the new task takes its slot).\n"
    "  * CORRECT — superseded task is COMPLETED but its output is\n"
    "    drift-contaminated (the old task stays in the plan as a historical\n"
    "    COMPLETED node; an edge old -> new is added).\n"
    "\n"
    "Forgetting `supersedes` on a renamed replacement is a runtime bug: the\n"
    "executor cannot link the new task to the old one, the predecessor is\n"
    "treated as fatally-failed, and the run aborts even though the\n"
    "replacement is healthy."
)

# Few-shot examples paired (positive REPLACE + negative EVOLUTION) to
# anchor the mutual-exclusivity framing. Without the negative example,
# the LLM tends to over-apply supersedes on legitimate id-reuse-with-
# evolved-title cases — the iter-7 reviewer's risk-mitigation note.
_SUPERSESSION_EXAMPLES = (
    "EXAMPLES:\n"
    "\n"
    "  Reused id (evolution; no supersedes):\n"
    '    Prior: {"id": "research_solar", "title": "Research solar panels"}\n'
    '    New:   {"id": "research_solar", "title": "Research solar + battery\n'
    '            costs"}\n'
    "    No supersedes — id reuse already encodes that this is the same\n"
    "    logical step.\n"
    "\n"
    "  New id with supersedes (replacement; ANY name shape):\n"
    '    Prior: {"id": "review_slides", "status": "FAILED"}\n'
    '    New:   {"id": "fix_review_slides", "title": "Re-do slide review\n'
    '            with cleaner outline", "supersedes": "review_slides",\n'
    '            "supersedes_kind": "REPLACE"}\n'
    "    The id `fix_review_slides` is fresh; supersedes is REQUIRED. The\n"
    "    same applies to `redo_review_slides`, `review_slides_again`,\n"
    "    `slide_review_2`, etc."
)


_LOOPING_TOOL_CALL_SYSTEM_PROMPT = """\
You are a task-planning assistant for a multi-agent system. The
adapter's loop detector has just flagged a task whose agent is stuck
calling the same tool with identical arguments without making forward
progress. Treat that task as FAILED — its current shape is unworkable
— and regenerate the remaining plan so the goals can still be met.

You will receive:

* The set of GOALS the plan is trying to satisfy.
* A list of tasks that have already finished (COMPLETED / FAILED /
  CANCELLED) — preserve these verbatim at the start of the returned
  task list.
* The id of the LOOPING task, which you MUST include in the returned
  plan with status FAILED (use the original id, title, and assignee).
* A list of OTHER PENDING / RUNNING / BLOCKED tasks. You may keep,
  drop, or rework these; the goal is to route around the failure.

Requirements:

1. KEEP HISTORY VERBATIM. Already-finished tasks (COMPLETED / FAILED /
   CANCELLED) must appear with the same id, title, assignee, and status.
2. FAIL THE LOOPER. Emit the looping task with status=FAILED. Do NOT
   leave it PENDING/RUNNING and do NOT rename it.
3. REPLACE OR DROP the looping work as needed. If the work is still
   required, add a fresh PENDING task (with a new id) that approaches
   it differently — split it smaller, hand it to a different agent, or
   precede it with a clarifying step. When the fresh PENDING task is
   intended to carry the failed task's work forward, set
   ``"supersedes": "<failed_task_id>"`` on the new task so the
   framework can re-pin agent reporting onto the replacement. Omit
   (or empty) ``supersedes`` for tasks that are not replacements.
   Also set ``"supersedes_kind"`` on every supersedes link:
     * ``"REPLACE"`` — the superseded task was PENDING / RUNNING /
       FAILED / CANCELLED (the looper, typically).
     * ``"CORRECT"`` — the superseded task had already COMPLETED but
       its output was drift-contaminated and the new task re-does
       that work. In CORRECT mode the old task stays in the plan as
       a historical COMPLETED node; the new task is added as a
       child with an edge old -> new.
4. PRESERVE OR REWORK other non-finished tasks at your discretion.
5. GOAL COVERAGE: every unsatisfied goal must still be addressed by at
   least one task in the returned plan.

Do NOT populate `assignee_agent_id` on any new or rewritten task; leave
it as the empty string. The framework populates it observationally
(goldfive#252). Tasks you preserve verbatim from the prior plan keep
their existing assignee value.

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|BLOCKED",
      "supersedes": "<optional: id of a terminal task this one replaces>",
      "supersedes_kind": "<REPLACE|CORRECT — required when supersedes is set>"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}
""" + "\n" + _SUPERSESSION_INVARIANT + "\n"


_USER_STEER_SYSTEM_PROMPT = """\
You are a task-planning assistant for a multi-agent system. A human
operator has just issued a STEERING directive against an in-flight
plan: they want the remaining work reshaped to incorporate their note,
while the work that has already finished must be preserved verbatim
for auditability.

You will receive:

* The set of GOALS the plan is trying to satisfy.
* A list of COMPLETED / FAILED / CANCELLED tasks (status history) —
  these are shown to you purely as CONTEXT so you understand what has
  already happened. You MUST NOT remove them, renumber them, or alter
  their ids / titles / assignees / statuses. They belong verbatim at
  the start of the returned plan's task list.
* The operator's STEERING NOTE describing the change in direction.

You will also receive:

* A list of PENDING tasks ALREADY IN THE PLAN — these are the
  remaining work the prior planning round produced. Treat them as
  EVOLVABLE, not disposable: when one of your output tasks continues
  the same logical step as a prior pending task (even with a new
  title, description, or assignee), REUSE the prior task's `id`. The
  runtime tracks task identity by id; minting a fresh id for
  continuing work makes it look like brand-new work and re-runs
  the work the operator just told you to keep.

Your job is to produce the REMAINING work (PENDING tasks) that
satisfies the goals in light of the steering note. Carry forward
the prior pending tasks where the work continues, EVOLVING their
title / description as needed; mint fresh ids only for genuinely
new tasks. Omit a prior pending task entirely if the steer
makes it unnecessary. Do NOT populate `assignee_agent_id`; leave
it as the empty string. The framework populates it observationally
(goldfive#252).

Requirements:

1. DO NOT repeat the completed tasks in your response. The caller
   will prepend them to your task list. Respond only with the
   PENDING tasks (continuing or new) and their edges.
2. Any edge whose ``from_task_id`` is one of the completed tasks is
   allowed — the caller will preserve those and wire them through.
3. Stable ids: task ids must be short and unique within your
   response, and must not collide with any completed-task id. For
   prior pending ids, REUSE them when the work continues; pick a
   fresh id only when the task is fundamentally new.
4. GOAL COVERAGE: every unsatisfied goal must still be addressed.
5. Honour the steering note: if it says "skip review", don't add a
   review task; if it says "focus on X", center the remaining work
   on X.
6. ``supersedes`` / ``supersedes_kind`` are for cases where a task
   STRUCTURALLY REPLACES another (not for evolution — evolution
   just reuses the id with mutated fields). Set them when:
   - your task replaces a task that has gone FAILED / CANCELLED
     (``"supersedes_kind": "REPLACE"``), OR
   - your task replaces a prior PENDING task that you judged the
     steer wants structurally retired but with a different id
     (``"supersedes_kind": "REPLACE"``), OR
   - your task corrects work that had already COMPLETED but whose
     output the steer judges drift-contaminated
     (``"supersedes_kind": "CORRECT"`` — the correction re-does
     that work; the old task stays in the plan as a historical
     node).
   Leave both fields empty when you are simply EVOLVING a prior
   pending task (id reused, no supersession).

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "supersedes": "<optional: id of a terminal task this one replaces>",
      "supersedes_kind": "<REPLACE|CORRECT — required when supersedes is set>"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}
""" + "\n" + _SUPERSESSION_INVARIANT + "\n\n" + _SUPERSESSION_EXAMPLES + "\n"


_PLAN_DIVERGENCE_SYSTEM_PROMPT = """\
You are a task-planning assistant maintaining an ACTIVE plan for a
multi-agent system. The executor's plan reconciler has detected
PLAN_DIVERGENCE: the agent tree has executed invocations that do not
match the planned task assignments. Your job is to decide whether the
observed activity is a legitimate "found a better path" (ABSORB) or a
goal-diverting excursion (REJECT).

You will receive:

* The set of GOALS the plan is trying to satisfy.
* The current plan as JSON (each task carries its live ``status``).
* The drift event that triggered this refine.
* A list of OBSERVED AGENT ACTIVITY — real invocations the tree has
  performed (agent name, invocation id, parent invocation id, start /
  completion timestamps, status, and a short summary).

Decide between two outcomes:

A. ABSORB. If the observed activity plausibly moves the run toward
   the declared GOALS *and* preserves every STICKY goal (goals marked
   ``[STICKY — from USER_STEER]`` — the operator has already steered
   the plan toward them, so they cannot be silently dropped), emit a
   revised plan that REFLECTS the observed activity. Existing tasks
   that correspond to completed invocations should be marked
   COMPLETED; in-flight invocations may be marked RUNNING. Invocations
   that do not correspond to any existing task should be added as new
   tasks (with fresh stable ids) so the Gantt view can show them.

B. REJECT. If the observed activity CONTRADICTS the goals — the tree
   has wandered into work that doesn't advance any goal, or (most
   importantly) is actively undoing a STICKY goal the operator just
   steered toward — return a JSON object of the form
   ``{"reject": true, "reason": "..."}`` and NOTHING else. The caller
   will escalate to human intervention. Only reject when the divergence
   cannot be squared with the goals; when in doubt, absorb.

Structural invariants (apply to the ABSORB path; REJECT bypasses
validation):

1. PRESERVE HISTORY. Tasks already COMPLETED / FAILED / CANCELLED
   must appear verbatim (same id, title, assignee, terminal status).
2. TERMINAL->TERMINAL EDGES must appear verbatim.
3. FORBIDDEN EDGES: no edges from a CANCELLED or FAILED task to a
   new PENDING task (the PENDING task would be definitionally
   unexecutable; the executor only schedules a PENDING task once
   every predecessor has COMPLETED).
4. Task ids unique within ``tasks``; every edge references a known
   task id.
5. The task graph must be ACYCLIC.
6. Every unsatisfied goal must still be addressed by at least one
   task in the returned plan.

Do NOT populate `assignee_agent_id` on new or rewritten tasks; leave it
as the empty string. The framework populates it observationally
(goldfive#252). Tasks you preserve verbatim from the prior plan keep
their existing assignee value.

Respond with a single JSON object and NOTHING else. For ABSORB:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|BLOCKED"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}

For REJECT:

{"reject": true, "reason": "<why the observed activity is off-goal>"}
"""


_REFINE_SYSTEM_PROMPT = """\
You are a task-planning assistant maintaining an ACTIVE plan for a
multi-agent system. You will receive:

* The set of GOALS the plan is trying to satisfy.
* The current plan as JSON (each task carries its live `status` — one
  of PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, BLOCKED).
* A single drift event describing what just happened that warrants
  re-planning (tool error, new work discovered, agent transfer, etc.).

Your job is to return the COMPLETE updated plan that reflects both
what has happened so far and what the system still needs to do to
satisfy the goals.

You MUST:

1. PRESERVE HISTORY. Tasks that are already COMPLETED, FAILED, or
   CANCELLED must appear in the returned plan with the same id,
   title, assignee, and status. Do NOT drop or renumber them — the
   Gantt-chart view must be able to show the timeline continuously.

2. UPDATE STATUSES. If the drift event reveals that a RUNNING task
   has finished, mark it COMPLETED (or FAILED if the event is an
   error). If a PENDING task has implicitly become the current focus,
   you may leave it PENDING and let the executor mark it RUNNING when
   work actually starts.

3. ADD NEW TASKS. If the drift reveals work that the original plan
   did not anticipate (e.g., a tool result surfaces a follow-up
   question, an error requires a retry/fallback path, a transfer
   introduces a sub-workflow), ADD new PENDING tasks for that work
   with fresh stable ids and appropriate edges back into the DAG.
   If a new PENDING task is a REPLACEMENT for a task that has gone
   FAILED or CANCELLED, set ``"supersedes": "<old_task_id>"`` on the
   replacement so the framework can re-pin reporting onto it. Omit
   (or leave empty) ``supersedes`` for genuinely new work that is
   not replacing anything. Also set ``"supersedes_kind"`` on every
   supersedes link:
     * ``"REPLACE"`` — the superseded task is PENDING / RUNNING /
       FAILED / CANCELLED (the new task takes its slot in the DAG).
     * ``"CORRECT"`` — the superseded task is already COMPLETED but
       its output is drift-contaminated. Use this when the drift
       shows the agent hallucinated progress, produced off-topic
       output but still called ``report_task_completed``, or any
       similar case where the COMPLETED signal was spurious. In
       CORRECT mode the old task stays in the plan (its COMPLETED
       status is preserved — it is the historical record of the
       drift-contaminated work) and the new task is inserted as a
       child: an edge ``old -> new`` is added and any downstream
       edges are rewired so work flows through the correction.

3a. CORRECTIVE PREDECESSORS (goldfive#248). When you insert a NEW
   PENDING task X meant to RUN BEFORE an existing non-terminal task
   Y (so Y's eventual execution depends on X's output — e.g.
   ``fix_research_tomatoes`` correcting a hallucinated research
   task that was already COMPLETED, or a clarifying step that must
   precede a still-PENDING ``draft_slides``), set
   ``"supersedes": "<Y_id>"`` on X. The validator then enforces
   corrective topology: every downstream Z of Y in the prior plan
   must now depend on X (i.e. add an edge X -> Z), OR the plan
   must keep Y as PENDING and add an edge X -> Y so Y itself
   waits on X. Inserting X as an independent root while leaving
   Y still root-eligible against COMPLETED predecessors is a
   structural bug: the executor will pick whichever happens to
   match first, producing out-of-order plan execution. The
   validator REJECTS that shape.

   Example. Prior plan: research_X (COMPLETED) -> draft_slides
   (PENDING). Drift: research_X output was off-topic. Correct
   revision (Shape B — re-edge consumers through X):

     tasks:
       research_X      (COMPLETED, preserved)
       fix_research_X  (PENDING, supersedes=research_X,
                        supersedes_kind=CORRECT)
       draft_slides    (PENDING, unchanged title/assignee)
     edges:
       research_X      -> fix_research_X
       fix_research_X  -> draft_slides   <-- re-edged via X
       (drop the prior research_X -> draft_slides edge or keep it
        — both forms are accepted as long as fix_research_X ->
        draft_slides exists)

   Wrong revision (rejected by validator): adding fix_research_X
   as an independent root with no edge to draft_slides. The
   validator emits "task 'fix_research_X' supersedes
   'research_X' but downstream consumers of 'research_X' not
   re-edged through 'fix_research_X'".

4. DROP OBSOLETE PENDING TASKS. If the drift makes a PENDING task
   unnecessary (e.g., a goal has been satisfied early, a dependency
   collapsed), you may omit it from the returned plan. Never drop
   tasks that already ran.

5. ASSIGNMENTS. Do NOT populate `assignee_agent_id` on new or
   rewritten tasks; leave it as the empty string. The framework
   populates it observationally (goldfive#252). Tasks you preserve
   verbatim from the prior plan keep their existing assignee value.

6. KEEP IDS STABLE. When the underlying work is the same, keep the
   task id unchanged. Reuse ids only for the same logical task.

7. STILL COVER THE GOALS. The revised plan must still have at least
   one task addressing each goal that is not yet satisfied.

8. RETURN A COMPLETE PLAN. Always return the full plan, not a delta.

SUMMARY POLICY. The ``summary`` field MUST be a noun phrase
describing the GOAL the plan delivers. DO NOT include process
commentary, meta-reasoning, or sentences explaining why you made
(or didn't make) changes. DO NOT mention 'drift', 'revision', or
'plan unchanged'. Even when nothing changes, the ``summary`` field
MUST remain a noun phrase describing the plan's GOAL — never
narrate the absence of changes ("plan unchanged", "no revision
needed", "drift event lacked detail", etc.). Re-emit the prior
plan's summary verbatim if no goal shift warrants a new one.
  RIGHT: "Create a 2-slide presentation about solar panels."
  RIGHT: "Generate a Python script that prints fibonacci numbers up to 100."
  WRONG: "Plan unchanged because no specific details were provided."

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "<noun phrase describing the GOAL — see SUMMARY POLICY>",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|BLOCKED",
      "supersedes": "<optional: id of a terminal task this one replaces>",
      "supersedes_kind": "<REPLACE|CORRECT — required when supersedes is set>"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}

If nothing needs to change, return the current plan unchanged (but
still as a complete JSON plan, not an empty object).
""" + "\n" + _SUPERSESSION_INVARIANT + "\n\n" + _SUPERSESSION_EXAMPLES + "\n"


# ---------------------------------------------------------------------------
# Shared refinement-guidance block (issue #241)
# ---------------------------------------------------------------------------
#
# When a drift triggers refine, the LLM planner's free choice over
# assignees and DAG shape is the source of two empirically-observed
# failure modes:
#
#   1. Single-agent drift (e.g. research_agent wanders off-topic) gets
#      "corrected" by reassigning the replacement to a different agent
#      (e.g. web_developer_agent). The corrective work lands on an
#      agent that has no relevant context.
#   2. Multi-stage plans collapse to a single task on refine, because
#      the LLM treats "refine" as "propose a simpler alternative" when
#      the actual intent is a minimal correction.
#
# The guidance block below steers the planner toward the conservative
# default (preserve the assignee + DAG shape, emit ``supersedes`` on
# the replacement) and reserves restructuring for the cases where the
# drift indicates the agent itself is the problem. Shared across the
# refine prompt builders so the same contract applies to every refine
# path that can reshape the plan in response to a drift.
_REFINEMENT_GUIDANCE_BLOCK = (
    "REFINEMENT GUIDANCE:\n"
    '- The drift you\'re correcting is usually "small" — a single '
    "agent produced off-topic or flawed output on a task it's "
    "otherwise capable of. Default pattern: replace the drifted task "
    "with a corrected variant and populate `supersedes: <old_task_id>` "
    "on the replacement. Leave `assignee_agent_id` empty on the "
    "replacement — the framework populates it observationally "
    "(goldfive#252). Preserve the surrounding DAG structure (edges, "
    "sibling tasks, stage count).\n"
    "- Only reshape the plan (collapse stages, drop tasks) when the "
    "drift indicates the work itself needs restructuring — repeated "
    "failures on the same task, tool errors that can't be recovered "
    "from, or a pattern the prior shape of the work has already "
    "failed at.\n"
    "- Do NOT collapse a multi-stage plan to a single task unless "
    "the user request genuinely warrants it.\n"
    "- The `supersedes` field is required on every replacement — "
    "it's how runtime routing re-pins reports from the old task "
    "to the new one.\n"
    "- The `supersedes_kind` field MUST accompany `supersedes`:\n"
    "  * REPLACE when the superseded task was PENDING / RUNNING / "
    "FAILED / CANCELLED (the typical retry).\n"
    "  * CORRECT when the superseded task is already COMPLETED but "
    "its output is drift-contaminated (the agent wandered off-topic "
    "yet still signalled completion). CORRECT keeps the old task "
    "in the plan as a historical COMPLETED node and adds an edge "
    "old -> new so downstream work flows through the correction."
)


# ---------------------------------------------------------------------------
# JSON parsing helpers (ported from harmonograf_client.planner)
# ---------------------------------------------------------------------------

_VALID_TASK_STATUSES = frozenset(s.value for s in TaskStatus)


def _is_tree_entry_list(available_agents: Any) -> bool:
    """Return True when ``available_agents`` is a structured tree list.

    Structured entries are dicts carrying at least a ``name`` key —
    see :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`.
    Plain ``list[str]`` returns False so the legacy code path renders
    the flat bullet-list form (goldfive#151).
    """
    if not isinstance(available_agents, list) or not available_agents:
        return False
    first = available_agents[0]
    return isinstance(first, Mapping) and "name" in first


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(raw: str) -> str:
    """Remove a surrounding ```json ... ``` or ``` ... ``` fence, if any."""
    if not raw:
        return raw
    match = _FENCE_RE.match(raw)
    if match:
        return match.group(1)
    return raw


def _normalize_assignee(raw: str) -> str:
    if ":" not in raw:
        return raw
    bare = raw.rsplit(":", 1)[-1]
    log.warning(
        "planner emitted compound assignee_agent_id=%r; normalized to %r",
        raw,
        bare,
    )
    return bare


def _coerce_status(raw: Any) -> TaskStatus:
    text = str(raw or "PENDING").upper()
    if text not in _VALID_TASK_STATUSES:
        return TaskStatus.PENDING
    return TaskStatus(text)


_VALID_SUPERSESSION_KINDS = frozenset(k.value for k in SupersessionKind)


def _coerce_supersession_kind(raw: Any) -> SupersessionKind:
    """Parse an LLM-emitted ``supersedes_kind`` value.

    Accepts the dataclass-enum string (``"REPLACE"`` / ``"CORRECT"`` /
    ``"UNSPECIFIED"``) as well as the full proto name
    (``"SUPERSESSION_KIND_REPLACE"``) so prompts can quote either shape.
    Unknown / missing values fall back to ``UNSPECIFIED``; the post-
    parse validator will resolve the final kind from the old task's
    status.
    """
    if raw is None:
        return SupersessionKind.UNSPECIFIED
    text = str(raw).strip().upper()
    if not text:
        return SupersessionKind.UNSPECIFIED
    if text.startswith("SUPERSESSION_KIND_"):
        text = text[len("SUPERSESSION_KIND_") :]
    if text not in _VALID_SUPERSESSION_KINDS:
        return SupersessionKind.UNSPECIFIED
    return SupersessionKind(text)


#: Old-task statuses that anchor a REPLACE-kind supersession (the old
#: task is still mid-flight / never ran). COMPLETED triggers the
#: CORRECT path. FAILED / CANCELLED / NOT_NEEDED are modelled as
#: REPLACE (the old task never delivered; the new one takes its slot).
_REPLACE_ELIGIBLE_OLD_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.BLOCKED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.NOT_NEEDED,
    }
)


def _normalize_supersession_kinds(revised: Plan, *, prior: Plan | None) -> Plan:
    """Return a Plan with ``supersedes_kind`` normalised on every task.

    goldfive#251 Option B validator. Rules, in order:

    0. (goldfive#213) If ``task.supersedes == task.id`` (self-reference
       — the v27 rev-2 anomaly where an LLM mis-applied supersedes to
       an id-reused task), clear the link. A task cannot be its own
       predecessor; the empty-target case (Rule 1) then clears any
       lingering kind.
    1. If ``task.supersedes`` is empty, clear ``supersedes_kind`` to
       ``UNSPECIFIED`` — a kind without a target is dangling.
    2. If the supersedes target does not resolve against either
       ``revised`` (in-revision chain like C.supersedes=B with B being
       added in the same revision) or ``prior`` (referencing a task
       that was in the outgoing plan), clear the kind. The structural
       validator will reject the dangling link independently.
    3. Otherwise resolve the old-task status from ``prior`` first
       (ground truth — the new task's ``supersedes`` points backward
       in time) and fall back to ``revised`` when the old task is
       being added in the same revision (edge case; supersession
       chains C->B->A introduced at once).
       * COMPLETED old task ⇒ coerce to ``CORRECT`` (warn if the LLM
         set REPLACE).
       * PENDING / RUNNING / BLOCKED / FAILED / CANCELLED / NOT_NEEDED
         ⇒ coerce to ``REPLACE`` (warn if the LLM set CORRECT).

    The coercion honours Option B's contract: the LLM's semantic signal
    is preserved when it aligns with status; when it disagrees, old-
    task status is ground truth.

    goldfive#247: returns a NEW :class:`Plan` (Plan is frozen). When no
    coercion is needed the input is returned unchanged so callers can
    keep their reference.
    """
    prior_by_id: dict[str, Task] = (
        {t.id: t for t in getattr(prior, "tasks", ()) or () if t.id} if prior is not None else {}
    )
    revised_by_id: dict[str, Task] = {t.id: t for t in revised.tasks if t.id}
    new_tasks: list[Task] = []
    changed = False
    for task in revised.tasks:
        new_supersedes = task.supersedes
        new_supersedes_kind = task.supersedes_kind
        sup_id = (task.supersedes or "").strip()
        # Rule 0 (goldfive#213): a task can never supersede itself.
        if sup_id and task.id and sup_id == task.id:
            log.warning(
                "planner: task %r has supersedes pointing at itself; "
                "clearing self-reference (a task cannot supersede itself)",
                task.id,
            )
            new_supersedes = ""
            sup_id = ""
        if not sup_id:
            # Rule 1: dangling kind with no target.
            if new_supersedes_kind is not SupersessionKind.UNSPECIFIED:
                log.warning(
                    "planner: task %r has supersedes_kind=%s without a "
                    "supersedes target; clearing to UNSPECIFIED",
                    task.id,
                    new_supersedes_kind.value,
                )
                new_supersedes_kind = SupersessionKind.UNSPECIFIED
        else:
            old = prior_by_id.get(sup_id) or revised_by_id.get(sup_id)
            if old is None:
                # Rule 2: unresolved target — clear the kind; the plan's
                # structural validator (step 3) will reject the dangling
                # edge separately.
                if new_supersedes_kind is not SupersessionKind.UNSPECIFIED:
                    log.warning(
                        "planner: task %r supersedes %r which is not in the plan; "
                        "clearing supersedes_kind from %s to UNSPECIFIED",
                        task.id,
                        sup_id,
                        new_supersedes_kind.value,
                    )
                    new_supersedes_kind = SupersessionKind.UNSPECIFIED
            else:
                # Rule 3: resolve based on old-task status.
                if old.status is TaskStatus.COMPLETED:
                    expected = SupersessionKind.CORRECT
                elif old.status in _REPLACE_ELIGIBLE_OLD_STATUSES:
                    expected = SupersessionKind.REPLACE
                else:
                    # Defensive: any unknown / future status falls through
                    # to REPLACE (pre-#251 behaviour).
                    expected = SupersessionKind.REPLACE
                if new_supersedes_kind is SupersessionKind.UNSPECIFIED:
                    new_supersedes_kind = expected
                elif new_supersedes_kind is not expected:
                    log.warning(
                        "planner: task %r supersedes %r (status=%s) with kind=%s; "
                        "coercing to %s based on old-task status (Option B)",
                        task.id,
                        sup_id,
                        old.status.value,
                        new_supersedes_kind.value,
                        expected.value,
                    )
                    new_supersedes_kind = expected
        if new_supersedes != task.supersedes or new_supersedes_kind != task.supersedes_kind:
            changed = True
            new_tasks.append(
                dataclasses.replace(
                    task,
                    supersedes=new_supersedes,
                    supersedes_kind=new_supersedes_kind,
                )
            )
        else:
            new_tasks.append(task)
    if not changed:
        return revised
    return dataclasses.replace(revised, tasks=tuple(new_tasks))


#: Statuses that warrant a retry/replace — i.e. the old task did NOT
#: succeed. Used by :func:`_backfill_retry_supersedes` to gate the
#: structural inference: only retries that follow a non-success outcome
#: get an inferred supersedes link. COMPLETED / PENDING / RUNNING /
#: BLOCKED are intentionally excluded — those represent either durable
#: provenance (don't quietly clobber) or work in flight (a "retry"
#: spawned alongside a healthy in-flight task is a confused emit, not
#: a structural successor).
_RETRY_WARRANTING_OLD_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED}
)


# Precompiled prefix patterns matching goldfive's id conventions for
# retry / version successor tasks. Mirrors
# :data:`goldfive.executors.sequential._RETRY_PREFIX_RE` but lives here
# to avoid an executor → planner import cycle (planner is a dependency
# of executors via :class:`LLMPlanner`). These patterns are STRUCTURAL
# inference over goldfive-emitted ids, not over user-content text — so
# they don't violate the no-regex-heuristics-on-NL contract
# (``feedback_no_regex_heuristics``).
_RETRY_PREFIX_RE = re.compile(r"^retry(?:\d+)?_")
#: Match a trailing ``_v<N>`` suffix on a task id (``t0_v2`` ⇒ root ``t0``).
_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def _strip_retry_prefix_once(task_id: str) -> str:
    """Strip exactly one ``retry_`` / ``retry<N>_`` prefix if present.

    Returns the original id when no prefix is present. Differs from
    :func:`goldfive.executors.sequential._lineage_root` in that this
    is single-shot (one peel) — used by backfill to find the immediate
    predecessor candidate of a single retry name. The full lineage
    root is still useful for executor budgeting; the single-step peel
    is the right granularity for "retry of WHICH task."
    """
    if not task_id:
        return task_id
    return _RETRY_PREFIX_RE.sub("", task_id, count=1)


def _candidate_predecessor_id(task_id: str) -> str:
    """Return the most likely predecessor id for a retry/version name.

    Two patterns, tried in order:

    * ``retry_<root>`` / ``retry<N>_<root>`` ⇒ ``<root>``.
    * ``<root>_v<N>`` ⇒ ``<root>``.

    Returns ``""`` when neither pattern matches (no inference). This is
    deterministic structural inference — the candidate must still
    resolve against the prior plan AND have a retry-warranting status
    before backfill happens; see :func:`_backfill_retry_supersedes`.
    """
    if not task_id:
        return ""
    stripped = _strip_retry_prefix_once(task_id)
    if stripped != task_id:
        return stripped
    # Try _v<N> suffix.
    suffix_stripped = _VERSION_SUFFIX_RE.sub("", task_id)
    if suffix_stripped != task_id and suffix_stripped:
        return suffix_stripped
    return ""


def _backfill_retry_supersedes(revised: Plan, *, prior: Plan) -> Plan:
    """Backfill ``Task.supersedes`` for retry-named tasks.

    Goldfive#213 — when a planner LLM emits a retry-shaped task name
    (``retry_t0``, ``t0_v2``) but forgets to populate the structural
    ``supersedes`` link, replacement detection in
    :func:`goldfive.executors.sequential._has_live_replacement` falls
    through to the conservative name-pattern fallback. That makes a
    COMPLETED retry incapable of satisfying the FAILED predecessor
    (predecessors-as-replacements are chronologically ambiguous under
    name-pattern alone). The fix is to make the structural link
    deterministic at merge time so the executor's causal tier can use
    it — no LLM trust, no prompt contract.

    Backfill rule (per task in ``revised`` whose ``supersedes`` is
    empty):

    * Compute candidate predecessor id by stripping a leading
      ``retry_`` / ``retry<N>_`` prefix OR a trailing ``_v<N>``
      suffix (see :func:`_candidate_predecessor_id`).
    * If the candidate exists in ``prior.tasks`` AND its status is
      FAILED or CANCELLED (the retry-warranting set), set
      ``task.supersedes = candidate``. Don't backfill against
      COMPLETED, PENDING, RUNNING, or absent candidates — those would
      be spurious links.
    * Self-references (``task.supersedes == task.id``) are cleared
      regardless. This is the v27 rev-2 anomaly where the LLM
      mis-applied supersedes to an id-reused task. Empty-target
      cleanup runs in :func:`_normalize_supersession_kinds` as a
      defence in depth.

    The post-backfill plan is still passed through
    :func:`_normalize_supersession_kinds` to derive the right
    ``supersedes_kind`` from old-task status — backfill ONLY sets the
    target id; the kind is still status-driven.

    goldfive#247: returns a NEW :class:`Plan` (Plan is frozen). When
    no backfill is needed the input is returned unchanged.
    """
    if prior is None:
        return revised
    prior_by_id: dict[str, Task] = {t.id: t for t in prior.tasks if t.id}
    new_tasks: list[Task] = []
    changed = False
    for task in revised.tasks:
        if not task.id:
            new_tasks.append(task)
            continue
        sup = (task.supersedes or "").strip()
        # Self-reference cleanup runs unconditionally (defence in
        # depth before normalize_supersession_kinds also runs). Pre-#247
        # this was an in-place clear that fell through to the candidate
        # backfill; we mirror the same logic by clearing ``sup`` and
        # tracking that the task already changed (so the final emit
        # always uses the cleaned variant rather than the raw input).
        cleared_self_ref = bool(sup and sup == task.id)
        if cleared_self_ref:
            sup = ""
        if sup:
            # LLM intent wins: never override an explicit link.
            new_tasks.append(task)
            continue
        candidate = _candidate_predecessor_id(task.id)
        if not candidate or candidate == task.id:
            if cleared_self_ref:
                new_tasks.append(dataclasses.replace(task, supersedes=""))
                changed = True
            else:
                new_tasks.append(task)
            continue
        old = prior_by_id.get(candidate)
        if old is None or old.status not in _RETRY_WARRANTING_OLD_STATUSES:
            if cleared_self_ref:
                new_tasks.append(dataclasses.replace(task, supersedes=""))
                changed = True
            else:
                new_tasks.append(task)
            continue
        log.debug(
            "planner: backfilling supersedes %r → %r on %r (prior status %s)",
            task.id,
            candidate,
            task.id,
            old.status.value,
        )
        new_tasks.append(dataclasses.replace(task, supersedes=candidate))
        changed = True
    if not changed:
        return revised
    return dataclasses.replace(revised, tasks=tuple(new_tasks))


#: Old-task statuses that don't require a supersedes link when dropped.
#: A FAILED or CANCELLED task in the prior plan represents work that was
#: already conclusively closed — the refine output may legitimately drop
#: it without naming a successor (the run will end up with the failure /
#: cancel still on the books from the prior plan even if the rebuilt
#: revision doesn't carry the slot forward). COMPLETED is intentionally
#: excluded: a dropped COMPLETED task IS an accountability gap because
#: its result is durable provenance for the run.
_ABSORBING_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED}
)


def _check_supersedes_coverage(
    revised: Plan,
    *,
    prior: Plan | None,
) -> list[Task]:
    """Return the list of prior tasks dropped without a supersedes link.

    A "dropped" task is one whose id appears in ``prior.tasks`` but not
    in ``revised.tasks``. Coverage is satisfied when:

    * Some new task in ``revised`` has ``supersedes`` pointing at the
      dropped id (REPLACE or CORRECT — both count; the kind is the
      semantic flavour of the link, not its existence). OR
    * The dropped task's status in ``prior`` is in
      :data:`_ABSORBING_TERMINAL_STATUSES` (FAILED / CANCELLED): the
      old work is conclusively closed and doesn't need a successor.

    Tasks that fail both predicates are **orphans**: dropped from the
    plan with no provenance trail. Returns the orphan ``Task`` objects
    (from ``prior``) so callers can include id + title in operator-
    visible telemetry. Order matches ``prior.tasks`` for deterministic
    log output.

    The CORRECT-kind supersedes case (test 5) doesn't appear here at
    all: in a CORRECT chain the old task is COMPLETED and is preserved
    verbatim in ``revised`` (Option B contract), so it's never in the
    ``dropped`` set in the first place. We don't need to special-case
    CORRECT vs REPLACE — both kinds satisfy "some task supersedes me"
    identically.

    .. note::

       Future work: when an orphan has a same-assignee, semantically-
       similar-title new task in ``revised`` (e.g. via embedding-based
       title similarity), auto-assign ``supersedes`` instead of just
       reporting. Out of scope for this observability-first validator
       — rejection / auto-heal is deferred until orphans become a
       systemic problem rather than a legitimate scope-narrowing
       outcome (e.g. user steer "ignore pianos" genuinely orphaning
       the piano-presentation task).
    """
    if prior is None:
        return []
    new_ids = {t.id for t in revised.tasks if t.id}
    supersedes_targets = {
        (t.supersedes or "").strip() for t in revised.tasks if (t.supersedes or "").strip()
    }
    orphans: list[Task] = []
    for old in prior.tasks:
        if not old.id or old.id in new_ids:
            continue
        if old.id in supersedes_targets:
            continue  # covered by a new task's supersedes link
        if old.status in _ABSORBING_TERMINAL_STATUSES:
            continue  # FAILED/CANCELLED don't need a successor
        orphans.append(old)
    return orphans


def _plan_from_json(
    obj: Any,
    *,
    run_id: str,
    goal_ids: list[str],
    plan_id: str | None = None,
) -> Plan | None:
    """Build a :class:`Plan` from a dict-shaped JSON object.

    Returns ``None`` if the payload is not a mapping or contains no
    usable tasks. ``run_id`` and ``goal_ids`` come from the caller and
    are stamped onto the plan envelope; the LLM only supplies the
    ``summary``, ``tasks``, and ``edges`` fields.
    """
    if not isinstance(obj, Mapping):
        return None
    raw_tasks = obj.get("tasks") or []
    raw_edges = obj.get("edges") or []
    if not isinstance(raw_tasks, list):
        return None
    tasks: list[Task] = []
    for t in raw_tasks:
        if not isinstance(t, Mapping):
            continue
        tid = str(t.get("id") or "").strip()
        title = str(t.get("title") or "").strip()
        if not tid or not title:
            continue
        # goldfive#252: assignee is observational, not declarative. Drop
        # any LLM-supplied value.
        tasks.append(
            Task(
                id=tid,
                title=title,
                description=str(t.get("description") or ""),
                assignee_agent_id="",
                status=_coerce_status(t.get("status")),
                predicted_start_ms=int(t.get("predicted_start_ms") or 0),
                predicted_duration_ms=int(t.get("predicted_duration_ms") or 0),
                bound_span_id=str(t.get("bound_span_id") or ""),
                # goldfive#237: explicit supersession link. Populated by
                # refine paths when the LLM produces a replacement for a
                # failed/cancelled task. Empty for net-new and preserved
                # terminal tasks.
                supersedes=str(t.get("supersedes") or "").strip(),
                # goldfive#251: supersession kind. Parsed raw here
                # (``UNSPECIFIED`` when absent); the post-parse validator
                # in :func:`_normalize_supersession_kinds` coerces it to
                # REPLACE / CORRECT based on the old task's status.
                supersedes_kind=_coerce_supersession_kind(t.get("supersedes_kind")),
            )
        )
    if not tasks:
        return None
    edges: list[TaskEdge] = []
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if not isinstance(e, Mapping):
                continue
            frm = str(e.get("from_task_id") or "").strip()
            to = str(e.get("to_task_id") or "").strip()
            if frm and to:
                edges.append(TaskEdge(from_task_id=frm, to_task_id=to))
    summary = str(obj.get("summary") or "")
    return Plan(
        id=plan_id or uuid.uuid4().hex,
        run_id=run_id,
        goal_ids=list(goal_ids),
        tasks=tasks,
        edges=edges,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Planner implementations
# ---------------------------------------------------------------------------


class PassthroughPlanner:
    """No-op planner — ``generate`` / ``refine`` / ``handle_turn``
    always return ``None``.

    Makes it safe to wire a ``planner=`` kwarg everywhere without
    forcing callers to opt in to planning on day one.
    """

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        return None

    async def handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        conversation_history: list[Any] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        # Phase 4 (goldfive#271): always returns None so the Runner
        # falls through to ``generate`` (which also returns None for a
        # PassthroughPlanner — i.e., the run aborts cleanly).
        return None


class StaticPlanner:
    """Planner that returns a pre-built :class:`Plan` verbatim.

    Useful for tests, CLIs, and the ``hello_callable`` example where the
    caller already knows the DAG it wants executed. On every call to
    :meth:`generate` a fresh copy of the template is returned, with
    ``run_id`` rewritten to match the current session and ``goal_ids``
    aligned to the provided goals. :meth:`refine` always returns
    ``None`` — refinement is out of scope when the plan is hard-coded.
    """

    def __init__(self, plan: Plan) -> None:
        if plan is None:  # pragma: no cover - defensive
            raise TypeError("StaticPlanner requires a non-None Plan")
        self._template = plan

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        run_id = ""
        if context is not None:
            run_id = str(context.get("run_id") or "")
        return Plan(
            id=self._template.id or uuid.uuid4().hex,
            run_id=run_id or self._template.run_id,
            goal_ids=[g.id for g in goals if g.id] or list(self._template.goal_ids),
            tasks=[
                Task(
                    id=t.id,
                    title=t.title,
                    description=t.description,
                    assignee_agent_id=t.assignee_agent_id,
                    status=t.status,
                    predicted_start_ms=t.predicted_start_ms,
                    predicted_duration_ms=t.predicted_duration_ms,
                    bound_span_id=t.bound_span_id,
                    # goldfive#251: preserve supersession metadata when
                    # cloning the StaticPlanner template so callers that
                    # bake supersedes into a hard-coded plan (tests,
                    # CLIs) see it survive the per-call clone.
                    supersedes=t.supersedes,
                    supersedes_kind=t.supersedes_kind,
                )
                for t in self._template.tasks
            ],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in self._template.edges
            ],
            summary=self._template.summary,
        )

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        return None

    async def handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        conversation_history: list[Any] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        # Phase 4 (goldfive#271): StaticPlanner has a baked plan, so
        # there's no per-turn decision to make. Returning None lets
        # the Runner fall through to ``generate`` which produces the
        # baked plan unchanged — preserves pre-Phase-4 semantics for
        # callers using StaticPlanner with multi-turn Runners.
        return None


class LLMPlanner:
    """Delegates to a caller-supplied async LLM callable.

    Parameters
    ----------
    call_llm:
        An async callable ``(system_prompt, user_prompt, model) -> str``.
        The returned string must be JSON conforming to the plan schema
        described in the default system prompt (it may be wrapped in
        triple-backtick ``json`` fences; this class will strip them).
    model:
        Model name passed through to ``call_llm``. Empty string is
        allowed; the callable may substitute its own default.
    system_prompt:
        Optional override for the goal-oriented planning prompt.
    refine_system_prompt:
        Optional override for the refinement prompt.

    On any parse error or exception raised by ``call_llm``, both
    :meth:`generate` and :meth:`refine` log a warning and return
    ``None`` — the host continues without a plan update.
    """

    # Default number of refine attempts before falling back. Two attempts
    # covers the typical "validator error feedback fixed it" round-trip
    # without burning too many LLM calls on a structurally confused
    # planner. Override per-instance via ``max_refine_attempts=`` or
    # globally by subclassing.
    DEFAULT_MAX_REFINE_ATTEMPTS: int = 2

    # Per-callsite ``max_output_tokens`` budget for the underlying
    # ``call_llm`` (goldfive#271 follow-up). All planner LLM calls return
    # a JSON plan structure; 16384 covers typical refines and large
    # multi-task generates while leaving ample headroom for Qwen 3.5
    # thinking-model preludes (think + answer share the same ceiling).
    # Pre-fix evidence (demo-v8.log): unbounded → 9961-token / 9.6-minute
    # calls; the wall-clock backstop now lives in
    # :data:`goldfive.adapters._adk_plugin.DEFAULT_LLM_CALL_TIMEOUT_MS`.
    # Subclasses may override; the value is read once per call into
    # :func:`goldfive._llm.call_llm_budget` so per-instance overrides
    # propagate without restart.
    MAX_OUTPUT_TOKENS: int = 16384

    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        system_prompt: str | None = None,
        refine_system_prompt: str | None = None,
        user_steer_system_prompt: str | None = None,
        looping_tool_call_system_prompt: str | None = None,
        plan_divergence_system_prompt: str | None = None,
        max_refine_attempts: int | None = None,
        user_steer_one_attempt: Callable[..., Awaitable[tuple[Any, str]]] | None = None,
    ) -> None:
        self._call_llm = call_llm
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._refine_system_prompt = refine_system_prompt or _REFINE_SYSTEM_PROMPT
        self._user_steer_system_prompt = user_steer_system_prompt or _USER_STEER_SYSTEM_PROMPT
        self._looping_tool_call_system_prompt = (
            looping_tool_call_system_prompt or _LOOPING_TOOL_CALL_SYSTEM_PROMPT
        )
        self._plan_divergence_system_prompt = (
            plan_divergence_system_prompt or _PLAN_DIVERGENCE_SYSTEM_PROMPT
        )
        self._max_refine_attempts = (
            int(max_refine_attempts)
            if max_refine_attempts is not None
            else self.DEFAULT_MAX_REFINE_ATTEMPTS
        )
        # Optional injection for the single-attempt USER_STEER refine
        # helper. ``None`` (the default) routes through
        # :meth:`_user_steer_one_attempt` (the in-class implementation).
        # Tests use this seam to script ``(plan, error)`` tuples
        # without monkeypatching the private bound method.
        self._user_steer_one_attempt_override: (
            Callable[..., Awaitable[tuple[Any, str]]] | None
        ) = user_steer_one_attempt
        # Optional sink for ``REFINE_VALIDATION_FAILED`` drifts. The
        # ``DefaultSteerer`` wires this in ``bind()`` so the planner can
        # surface a structured signal when its retry budget is spent
        # without the planner needing a dependency on the sink pipeline.
        # When left as ``None`` (e.g. the planner is used standalone in
        # tests) the fallback still happens -- it is just not emitted.
        self._drift_emitter: Callable[[DriftEvent], Awaitable[None]] | None = None
        # Optional context provider for ``GoldfiveLLMCallStart/End``
        # spans (goldfive internal-llm-spans). The steerer wires this
        # up in :meth:`bind` so every planner-internal ``call_llm``
        # site shows up on harmonograf's Gantt. Returns
        # ``(sinks, run_id, session_id, task_id, sequence_fn)`` snapshotted
        # against the currently-active session, or ``None`` when the
        # planner is used standalone (tests) — spans degrade to a
        # no-op in that case.
        self._span_ctx_provider: Callable[[], Any] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_refine_attempts(self) -> int:
        return self._max_refine_attempts

    def set_user_steer_one_attempt(
        self,
        attempt: Callable[..., Awaitable[tuple[Any, str]]] | None,
    ) -> None:
        """Install (or remove) a custom one-attempt callable for refine_user_steer.

        Mirror of the ``user_steer_one_attempt`` constructor kwarg. Tests
        use this seam to script ``(merged_plan, error)`` tuples through
        the retry loop without monkeypatching the private bound method
        on the instance. Pass ``None`` to revert to the in-class
        :meth:`_user_steer_one_attempt` implementation.
        """
        self._user_steer_one_attempt_override = attempt

    def set_drift_emitter(self, emitter: Callable[[DriftEvent], Awaitable[None]] | None) -> None:
        """Install (or remove) the async callback used to signal refine
        failures.

        The steerer wires this up in ``bind()`` so that when the planner
        exhausts its refine retry budget it can emit a
        ``REFINE_VALIDATION_FAILED`` ``DriftEvent`` through the normal
        event pipeline without the planner owning a sink list itself.
        """
        self._drift_emitter = emitter

    def set_span_context_provider(self, provider: Callable[[], Any] | None) -> None:
        """Install (or remove) the callable that supplies span-emission context.

        The callable is invoked at every ``call_llm`` site to snapshot
        the currently-bound ``(sinks, run_id, session_id, task_id,
        sequence_fn)`` tuple under which the call is running. When it
        returns ``None`` (no session in scope) span emission degrades
        to a no-op — the wrapped ``await call_llm(...)`` runs exactly
        as before.

        Wired from :meth:`DefaultSteerer.bind` so goldfive-internal LLM
        calls show up as proper spans on harmonograf's Gantt without
        the planner owning a sink list itself.
        """
        self._span_ctx_provider = provider

    def _span_kwargs(self, task_id_override: str = "") -> dict[str, Any]:
        """Resolve keyword args for :func:`goldfive_llm_span` at a call site.

        Returns an empty dict when no provider is bound (span degrades
        to a no-op wrapping the call). ``task_id_override`` lets refine
        paths stamp the drift-bound task id on the span even when the
        session's ``current_task_id`` has already flipped.
        """
        if self._span_ctx_provider is None:
            return {"sinks": [], "model": self._model}
        try:
            ctx = self._span_ctx_provider()
        except Exception:  # noqa: BLE001 - observability must never break the run
            return {"sinks": [], "model": self._model}
        if ctx is None:
            return {"sinks": [], "model": self._model}
        sinks, run_id, session_id, task_id, seq_fn = ctx
        return {
            "sinks": list(sinks or []),
            "model": self._model,
            "run_id": run_id or "",
            "session_id": session_id or "",
            "task_id": task_id_override or task_id or "",
            "sequence_fn": seq_fn,
        }

    # ---- span decoration helpers ----------------------------------------
    #
    # These compose the ``input_preview`` / ``target_*`` / decision-summary
    # strings used by the ``goldfive_llm_span`` wrap sites inside the
    # planner (refine / refine_steer / refine_looping_tool_call /
    # plan_generate / planner_handle_turn). Extracted so tests can
    # assert the exact payload without re-deriving it from drift / plan
    # internals, and so the callers stay short.

    @staticmethod
    def _build_refine_span_input_preview(
        drift: DriftEvent,
        plan: Plan,
    ) -> str:
        """Render the ``input_preview`` string for a refine-type span.

        Mirrors :meth:`DefaultSteerer._build_refine_input_summary` on the
        steerer's ``PlanRevised`` event but lives here because the
        planner's span emission has to fire before the PlanRevised event
        does — duplicating the format keeps the two strings consistent
        so frontends that render both render the same summary.
        """
        parts: list[str] = []
        parts.append(f"drift: {drift.kind.value}/{drift.severity.value}")
        if drift.current_task_id:
            parts.append(f"task: {drift.current_task_id}")
        if drift.current_agent_id:
            parts.append(f"agent: {drift.current_agent_id}")
        if drift.detail:
            parts.append(f"detail: {drift.detail}")
        tasks = getattr(plan, "tasks", None) or []
        plan_line = f"current plan: rev{plan.revision_index}, {len(tasks)} task(s)"
        if tasks:
            titles = [f"{t.id} ({str(getattr(t, 'status', '') or '').lower()})" for t in tasks[:8]]
            plan_line += ": " + ", ".join(titles)
        parts.append(plan_line)
        return "\n".join(parts)

    @staticmethod
    def _build_refine_span_output_preview(revised: Plan) -> str:
        """Render the ``output_preview`` for a successful refine result."""
        tasks = getattr(revised, "tasks", None) or []
        parts: list[str] = [
            f"revision_index={revised.revision_index}",
            f"tasks={len(tasks)}",
        ]
        assignees = sorted(
            {str(t.assignee_agent_id or "") for t in tasks if getattr(t, "assignee_agent_id", "")}
        )
        if assignees:
            parts.append("assignees=[" + ", ".join(assignees) + "]")
        titles = [str(getattr(t, "title", "") or "") for t in tasks[:6]]
        titles = [t for t in titles if t]
        if titles:
            parts.append("titles=[" + ", ".join(titles) + "]")
        return " | ".join(parts)

    # ---- prompt builders -------------------------------------------------

    @staticmethod
    def _render_goals_block(goals: list[Goal]) -> str:
        """Render goals for prompt consumption.

        Goals sourced from a USER_STEER directive
        (``source == GOAL_SOURCE_USER_STEER``) are annotated ``[STICKY —
        from USER_STEER]`` so the planner-LLM sees them as operator-
        authored and knows the refine validator (goldfive#154) will
        reject revisions that silently drop them. Goals without an
        explicit source render unchanged so legacy callers see the
        exact same prompt shape.
        """
        if not goals:
            return "- (no goals provided)"
        lines: list[str] = []
        for g in goals:
            gid = g.id or "(no-id)"
            summary = g.summary or "(no summary)"
            if g.source == GOAL_SOURCE_USER_STEER:
                lines.append(f"- [{gid}] {summary}  [STICKY — from USER_STEER]")
            else:
                lines.append(f"- [{gid}] {summary}")
        return "\n".join(lines)

    @staticmethod
    def _user_steer_goals(goals: list[Goal]) -> list[Goal]:
        """Return the subset of ``goals`` added by a prior ``USER_STEER``.

        Used by the refine validator (goldfive#154) to enforce the
        "sticky goal" contract: a USER_STEER-sourced goal must remain
        addressed by the revised plan, so a later drift cannot silently
        unwind an operator steer by refining around it.
        """
        return [g for g in goals if g.source == GOAL_SOURCE_USER_STEER]

    @staticmethod
    def _render_agents_block(available_agents: list[str] | list[dict[str, Any]] | None) -> str:
        """Render the agents/AGENT TREE section for the planner prompt.

        Accepts either a plain ``list[str]`` (back-compat) or a
        structured ``list[dict]`` as produced by
        :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`
        (goldfive#151). The dict form renders as a tree-shape block
        with name, role, kind, depth, and parent so the LLM can see
        which agents it may pick and which are intermediate
        coordinators. Plain strings render as a flat bullet list
        exactly as before.

        An empty / None value renders as the literal placeholder used
        today so existing structural prompt assertions keep passing.
        """
        if not available_agents:
            return "- (none listed)"
        if _is_tree_entry_list(available_agents):
            lines: list[str] = []
            for entry in available_agents:
                name = str(entry.get("name", ""))
                if not name:
                    continue
                role = str(entry.get("role", ""))
                kind = str(entry.get("kind", ""))
                depth = int(entry.get("depth", 0) or 0)
                parent = str(entry.get("parent", ""))
                parent_frag = f" parent={parent}" if parent else ""
                lines.append(f"- {name} (role={role}, kind={kind}, depth={depth}{parent_frag})")
            return "\n".join(lines) or "- (none listed)"
        return "\n".join(f"- {a}" for a in available_agents) or "- (none listed)"

    @staticmethod
    def _flatten_agent_names(
        available_agents: list[str] | list[dict[str, Any]] | None,
    ) -> list[str]:
        """Return the list of legal ``assignee_agent_id`` values.

        Accepts either the plain-string or tree-entry shape and
        produces a de-duplicated, order-preserving list of agent
        names. ``None`` / empty yields ``[]`` — the validator treats
        an empty registry as "skip the assignee check" so existing
        callers that don't provide available_agents keep working.
        """
        if not available_agents:
            return []
        names: list[str] = []
        seen: set[str] = set()
        if _is_tree_entry_list(available_agents):
            for entry in available_agents:
                name = str(entry.get("name", ""))
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
            return names
        for a in available_agents:
            s = str(a)
            if s and s not in seen:
                seen.add(s)
                names.append(s)
        return names

    @staticmethod
    def _validate_plan_assignees(
        plan: Plan,
        available_agents: list[str] | list[dict[str, Any]] | None,
    ) -> str:
        """Return an error string if any task's assignee is off-registry.

        When ``available_agents`` is falsy the check is skipped —
        callers that did not supply a registry get the legacy
        "anything goes" behaviour. When non-empty, every task's
        ``assignee_agent_id`` (if non-empty) must appear in the
        flattened name list; offenders are enumerated in the error
        message so the retry-with-correction loop (#133/#136) has
        concrete feedback to thread back to the LLM.
        """
        names = LLMPlanner._flatten_agent_names(available_agents)
        if not names:
            return ""
        registry = set(names)
        offenders: list[tuple[str, str]] = []
        for t in plan.tasks:
            assignee = (t.assignee_agent_id or "").strip()
            if not assignee:
                continue
            if assignee not in registry:
                offenders.append((t.id, assignee))
        if not offenders:
            return ""
        listed = ", ".join(f"{tid!r}->{a!r}" for tid, a in offenders)
        return (
            f"off-registry assignee(s): {listed}. "
            f"Every task's `assignee_agent_id` must match a name from the "
            f"AGENT TREE registry: {sorted(registry)}. "
            f"Re-assign offending tasks to a registry-listed agent or, "
            f"when no specialised agent fits, to the root/coordinator."
        )

    @staticmethod
    def _render_observed_actions_block(observed_actions: list[ObservedAction]) -> str:
        """Render observed agent activity as a human-readable prompt block.

        Each entry is one line with the agent name, status, timestamps,
        invocation ids, and summary. An empty list renders as a single
        ``- (no observed activity)`` line so the prompt shape is
        invariant — the LLM always sees the header and a body, making
        structural prompt tests easier.
        """
        header = "OBSERVED AGENT ACTIVITY (what the tree has actually done):"
        if not observed_actions:
            return f"{header}\n- (no observed activity)"
        lines: list[str] = [header]
        for i, a in enumerate(observed_actions, start=1):
            started = a.started_at.isoformat() if a.started_at else ""
            completed = a.completed_at.isoformat() if a.completed_at else "(in-flight)"
            parent_frag = f" parent={a.parent_invocation_id}" if a.parent_invocation_id else ""
            summary = a.summary or "(no summary)"
            lines.append(
                f"{i}. agent={a.agent_name!r} status={a.status} "
                f"invocation_id={a.invocation_id}{parent_frag} "
                f"started_at={started} completed_at={completed}\n"
                f"   summary: {summary}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_off_topic_reasoning_block(drift: DriftEvent) -> str:
        """Render the reasoning context for an OFF_TOPIC drift refine.

        OFF_TOPIC drifts come from the reasoning judge (or sentence-level
        cosine detector); the deviation is in the agent's reasoning, not
        in the observed agent activity. To give the goal-aware refine
        prompt an analogue of ``OBSERVED AGENT ACTIVITY`` we render the
        reason and the truncated reasoning text the judge saw. Falls back
        to ``"(no reasoning excerpt available)"`` when the drift carries
        no preview, so the prompt shape is invariant for tests.
        """
        header = "OFF-TOPIC REASONING (what the agent has been reasoning about):"
        reason = drift.detail or "(judge returned no reason)"
        excerpt = drift.trigger_input or ""
        if not excerpt and isinstance(drift.raw, str):
            excerpt = drift.raw
        excerpt = excerpt.strip()
        if not excerpt:
            excerpt = "(no reasoning excerpt available)"
        return f"{header}\n- judge reason: {reason}\n- excerpt: {excerpt}"

    @staticmethod
    def _render_prior_turns_block(context: Mapping[str, Any] | None) -> str:
        """Render cross-turn context from a Conversation into a prompt block.

        Reads ``prior_completed_results`` and ``prior_turns`` off the
        planner context (see :class:`~goldfive.conversation.Conversation`)
        and produces a human-readable section for the planner to reason
        about. Returns ``""`` when the caller is not running inside a
        multi-turn Conversation (first turn, or single-turn use).
        """
        if not context:
            return ""
        turns = context.get("prior_turns") or []
        prior_results = context.get("prior_completed_results") or {}
        if not turns and not prior_results:
            return ""
        lines: list[str] = ["\nPrior-turn context (this is a multi-turn conversation):"]
        if turns:
            lines.append("\nEarlier turns (most recent last):")
            for i, t in enumerate(turns, start=1):
                if not isinstance(t, Mapping):
                    continue
                ui = str(t.get("user_input_summary") or "")
                plan_summary = str(t.get("plan_summary") or "")
                success = bool(t.get("outcome_success", True))
                status = "succeeded" if success else "failed"
                reason = str(t.get("outcome_reason") or "")
                reason_frag = f" ({reason})" if (not success and reason) else ""
                lines.append(
                    f"  {i}. user: {ui!r} -> {status}{reason_frag}"
                    f"{f'; plan: {plan_summary}' if plan_summary else ''}"
                )
        if prior_results:
            lines.append("\nResults already produced in earlier turns (task_id -> summary):")
            for task_id, summary in prior_results.items():
                lines.append(f"  - {task_id}: {summary}")
        lines.append(
            "\nWhen planning this turn, treat the user's current input as a "
            "follow-up. Reuse prior results where relevant, and only "
            "re-do work that the user's new input actually requires."
        )
        return "\n".join(lines) + "\n"

    def _build_generate_prompt(
        self,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None,
    ) -> str:
        goals_block = self._render_goals_block(goals)
        agents_block = self._render_agents_block(available_agents)
        agents_header = (
            # goldfive#252: assignee is observational, not declarative.
            # Render the tree as context only — do NOT instruct the LLM
            # to pick an assignee from it.
            "AGENT TREE (context only — do NOT populate assignee_agent_id):"
            if _is_tree_entry_list(available_agents)
            else "Available agents (context only):"
        )
        prior_block = self._render_prior_turns_block(context)
        context_block = ""
        if context:
            # Exclude the verbose cross-turn fields from the raw JSON
            # dump — they're already rendered as a human-readable
            # block above. Keep everything else.
            filtered = {
                k: v
                for k, v in dict(context).items()
                if k
                not in {
                    "prior_completed_results",
                    "prior_turns",
                    "turn_index",
                    "conversation_id",
                }
            }
            if filtered:
                try:
                    context_block = (
                        f"\nAdditional context (JSON):\n{json.dumps(filtered, default=str)}\n"
                    )
                except (TypeError, ValueError):
                    context_block = ""
        return (
            f"{agents_header}\n{agents_block}\n\n"
            f"Goals:\n{goals_block}\n"
            f"{prior_block}"
            f"{context_block}\n"
            "Respond with a single JSON object following the schema."
        )

    def _build_user_steer_prompt(
        self,
        completed: list[Task],
        drift: DriftEvent,
        goals: list[Goal],
    ) -> str:
        """Compatibility shim for :meth:`_build_steer_prompt` (user source).

        Kept for subclasses and tests that referred to the pre-
        unification name. New code should call
        :meth:`_build_steer_prompt` with an explicit ``source``.
        """
        return self._build_steer_prompt(completed, drift, goals, source="user")

    def _build_steer_prompt(
        self,
        completed: list[Task],
        drift: DriftEvent,
        goals: list[Goal],
        *,
        source: str = "user",
        prior_pending: list[Task] | None = None,
    ) -> str:
        """Build the steer-refine user prompt.

        Completed tasks are shown as read-only context; prior PENDING
        tasks are shown as EVOLVABLE work whose ids the LLM should reuse
        when continuing the same logical step. The caller prepends the
        completed tasks back onto the returned plan so lineage is
        preserved verbatim. PENDING task ids must not collide with
        completed-history ids; this remains an explicit invariant
        (goldfive#133).

        ``source`` selects the directive framing:

        * ``"user"`` — the drift carries an operator note; prompt
          labels it "Operator steering note".
        * ``"goldfive"`` — goldfive's drift ladder promoted a detector
          signal into a steer; prompt labels it "Goldfive drift
          correction (agent drift was detected in the preceding
          activity)" and instructs the LLM to discard work on the
          contaminated task.
        """
        history = [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "assignee_agent_id": t.assignee_agent_id,
                "status": str(t.status),
            }
            for t in completed
        ]
        history_json = json.dumps(history, default=str)
        history_ids = json.dumps([t.id for t in completed])
        prior_pending = list(prior_pending or [])
        prior_pending_json = json.dumps(
            [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "assignee_agent_id": t.assignee_agent_id,
                }
                for t in prior_pending
            ],
            default=str,
        )
        prior_pending_ids = json.dumps([t.id for t in prior_pending])
        goals_block = self._render_goals_block(goals)
        sticky_block = self._render_sticky_goals_block(goals)
        note = drift.detail or "(no steering note provided)"
        invariants = (
            "STRUCTURAL INVARIANTS (the validator will REJECT any response "
            "that breaks these rules):\n"
            "1. The PENDING task ids you emit MUST NOT collide with "
            f"any completed-history id. Reserved completed ids: {history_ids}\n"
            "2. Every edge's `from_task_id` and `to_task_id` must "
            "reference either a reserved history id, a prior-pending id "
            "you reused, or a new-task id from your own `tasks` array.\n"
            "3. FORBIDDEN EDGES: no edges from history task ids whose "
            "status is CANCELLED or FAILED to PENDING task "
            "ids. A PENDING task whose predecessor is CANCELLED or "
            "FAILED is unexecutable: the executor only schedules a "
            "PENDING task once every predecessor reaches COMPLETED, "
            "and CANCELLED/FAILED never fire that transition, so "
            "grafting onto them stalls the whole sub-DAG. PENDING work "
            "must start from no predecessors, from a COMPLETED history "
            "task, or from another PENDING task in your response — do "
            "NOT add edges like \"cancelled_research -> new_research\" "
            "to \"chain\" from the prior plan to your new one.\n"
            "4. Your task ids must be unique within your response.\n"
            "5. Do not introduce edges that create a cycle.\n"
            "6. ID REUSE FOR CONTINUING WORK. When a task in your "
            "response is the SAME logical step as one in 'Prior PENDING "
            "work' — even if you rename it, retitle it, reassign it, or "
            "refine its description — you MUST reuse the prior task's "
            "`id`. Mint a fresh id ONLY for genuinely new work that did "
            "not exist in the prior plan. Stable ids are how the runtime "
            "tracks task identity across revisions; minting a new id for "
            "continuing work makes the runtime treat it as brand-new and "
            "re-runs work the operator just told you to keep. Reusable "
            f"prior PENDING ids: {prior_pending_ids}\n"
            "7. " + _SUPERSESSION_INVARIANT
        )
        if source == "goldfive":
            directive_header = (
                "Goldfive drift correction (agent drift was detected in "
                "the preceding activity — discard any prior work on the "
                "contaminated task):"
            )
            closing = (
                "Generate the PENDING tasks (and their edges) that "
                "should run from here, applying the correction above. "
                "REUSE the prior pending id for any task that continues "
                "prior work; mint new ids only for truly new tasks. "
                "Omit a prior pending task entirely if the correction "
                "makes it unnecessary. Respond with JSON only."
            )
        else:
            directive_header = "Operator steering note:"
            closing = (
                "Generate the PENDING tasks (and their edges) that "
                "should run from here, taking the steering note into "
                "account. REUSE the prior pending id for any task that "
                "continues prior work; mint new ids only for truly new "
                "tasks. Omit a prior pending task entirely if the steer "
                "makes it unnecessary. Respond with JSON only."
            )
        return (
            f"CURRENT GOALS (the PENDING tasks must still advance "
            f"every goal, and MUST NOT silently drop any [STICKY] "
            f"goal carried from a prior USER_STEER):\n{goals_block}\n\n"
            f"{sticky_block}"
            f"Completed/Failed/Cancelled tasks (READ-ONLY CONTEXT — "
            "preserve these verbatim at the start of the returned plan; "
            f"do NOT repeat them in your response):\n{history_json}\n\n"
            f"Prior PENDING work (EVOLVABLE — reuse the id when your "
            "task continues this step; omit when the steer drops it; "
            "supersede with a new id ONLY when structurally replacing):"
            f"\n{prior_pending_json}\n\n"
            f"{directive_header}\n{note}\n\n"
            f"{_REFINEMENT_GUIDANCE_BLOCK}\n\n"
            f"{invariants}\n\n"
            f"{closing}"
        )

    # ---- structural-invariant helpers (issue #133) ----------------------

    @staticmethod
    def _terminal_tasks(plan: Plan) -> list[Task]:
        """Return the prior plan's tasks whose status is terminal.

        Terminal = COMPLETED / FAILED / CANCELLED. These are the tasks
        ``Plan.validate(for_revision=True, prior=...)`` requires the
        revision to preserve verbatim (PLAN-LIFECYCLE.md §3.1).
        """
        return [t for t in plan.tasks if t.status in _TERMINAL_STATUSES]

    @staticmethod
    def _terminal_to_terminal_edges(plan: Plan) -> list[TaskEdge]:
        """Return edges whose endpoints are both terminal.

        These edges must appear verbatim in any revision
        (PLAN-LIFECYCLE.md §3.2). The validator's
        ``terminal->terminal edge 'X' -> 'Y' missing in revision`` error
        is exactly the one this list guards against.
        """
        terminal_ids = {t.id for t in plan.tasks if t.status in _TERMINAL_STATUSES}
        return [
            e for e in plan.edges if e.from_task_id in terminal_ids and e.to_task_id in terminal_ids
        ]

    @classmethod
    def _render_structural_invariants_block(cls, plan: Plan) -> str:
        """Render an explicit, verbatim-mandatory invariants section.

        The planner-LLM routinely drops terminal→terminal edges when
        regenerating a plan in JSON because the "preserve history"
        instruction is implicit. Listing the tasks AND the required
        edges as copy-paste-ready JSON drops the first-attempt
        validation failure rate sharply (see goldfive#133).
        """
        terminal = cls._terminal_tasks(plan)
        tt_edges = cls._terminal_to_terminal_edges(plan)
        terminal_json = json.dumps(
            [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "assignee_agent_id": t.assignee_agent_id,
                    "status": str(t.status),
                }
                for t in terminal
            ],
            default=str,
        )
        tt_edges_json = json.dumps(
            [{"from_task_id": e.from_task_id, "to_task_id": e.to_task_id} for e in tt_edges],
            default=str,
        )
        lines: list[str] = [
            "STRUCTURAL INVARIANTS (the validator will REJECT any response "
            "that breaks these rules; reread this section before emitting "
            "JSON):",
            "",
            "1. TERMINAL TASKS (status in {COMPLETED, FAILED, CANCELLED}) "
            "MUST appear verbatim in your `tasks` array. Same `id`, "
            "`title`, `assignee_agent_id`, and same terminal `status`. "
            "You MUST NOT drop them, rename them, or regress their status "
            "back to PENDING/RUNNING/BLOCKED. NOTE: 'terminal' here means "
            "task STATUS (COMPLETED/FAILED/CANCELLED), NOT graph position "
            "— a leaf task whose status is PENDING is NOT terminal. A "
            "terminal task with no downstream successors is still terminal "
            "and still MUST be kept. The most common failure mode on "
            "weaker models is silently dropping a terminal task that "
            "has no downstream consumers because 'it looks done'; the "
            "validator rejects this with `terminal task '<id>' missing "
            "in revision` and the run escalates to human intervention. "
            "DO NOT do this.",
            f"   Terminal tasks in the current plan: {terminal_json}",
            "   VALID revision shape (terminal tasks preserved verbatim):",
            '     "tasks": [..., {"id": "<terminal_id>", "title": "...", '
            '"status": "COMPLETED", "assignee_agent_id": "..."}, ...]',
            "   INVALID revision shape (terminal task omitted — REJECTED):",
            '     "tasks": [<new PENDING tasks only; <terminal_id> missing>]',
            "",
            "2. TERMINAL->TERMINAL EDGES (edges where BOTH endpoints are "
            "terminal tasks) MUST appear verbatim in your `edges` array. "
            "These edges are frozen history; dropping even one will fail "
            "validation.",
            f"   Required edges: {tt_edges_json}",
            "",
            "3. FORBIDDEN EDGES: no edges from CANCELLED or FAILED "
            "tasks to new PENDING tasks. New work must start fresh — "
            'do NOT add edges like "old_research -> new_research" '
            "where ``old_research`` is CANCELLED or FAILED. A PENDING "
            "task whose predecessor is CANCELLED/FAILED is unexecutable "
            "because the executor only schedules a PENDING task once "
            "every predecessor reaches COMPLETED, and CANCELLED/FAILED "
            "states never fire that transition; the whole downstream "
            "sub-DAG would stall. Your new tasks must form an "
            "independent sub-DAG with their own root task (a task "
            "whose `id` appears in no edge's `to_task_id`, or whose "
            "only predecessors within the new sub-DAG are themselves "
            "new PENDING tasks). Edges from COMPLETED tasks to new "
            "PENDING tasks are allowed -- those are immediately "
            "eligible because the predecessor has already completed.",
            "",
            "4. TASK IDS must be unique within `tasks`. Every edge's "
            "`from_task_id` and `to_task_id` must reference a task id "
            "that exists in your `tasks` array.",
            "",
            "5. The task graph must be ACYCLIC. Do not introduce edges that would create a cycle.",
            "",
            "6. CORRECTIVE PREDECESSORS (goldfive#248). When you insert "
            "a new PENDING task X with `supersedes: <Y_id>` and Y is "
            "non-terminal in the current plan (PENDING / RUNNING / "
            "BLOCKED — Y's status above is one of those), X must be "
            "wired as a structural predecessor of Y's downstreams. "
            "Either: (a) add edges from X to every prior consumer of "
            "Y so the corrective work runs before any task that "
            "depended on Y, OR (b) keep Y as PENDING in your plan and "
            "add a single edge X -> Y so Y itself waits on X. "
            "Inserting X as an independent root while leaving Y's "
            "downstreams reachable without going through X first is "
            "REJECTED — the executor would race the corrective work "
            "against the work it was meant to correct.",
        ]
        return "\n".join(lines)

    _GOAL_REFERENCE_STOPWORDS: frozenset[str] = frozenset(
        {
            "the",
            "and",
            "with",
            "from",
            "that",
            "this",
            "have",
            "will",
            "into",
            "your",
            "about",
            "also",
            "ensure",
            "make",
            "keep",
            "should",
            "must",
            "while",
            "their",
            "there",
            "them",
            "then",
            "than",
            "when",
            "what",
            "which",
            "some",
            "goal",
            "goals",
            "task",
            "tasks",
            "plan",
            "plans",
            "focus",
            "draft",
            "review",
            "final",
            "work",
            "round",
        }
    )

    @classmethod
    def _goal_summary_tokens(cls, goal: Goal) -> set[str]:
        """Tokenise a goal summary into lowercase words of length >= 4
        that aren't trivial stopwords. A heuristic by design — see
        :meth:`_check_user_steer_goals_preserved` for the rationale.
        """
        tokens: set[str] = set()
        summary = goal.summary or ""
        for raw in re.split(r"[^A-Za-z0-9_]+", summary):
            w = raw.lower().strip()
            if len(w) >= 4 and w not in cls._GOAL_REFERENCE_STOPWORDS:
                tokens.add(w)
        return tokens

    @classmethod
    def _check_user_steer_goals_preserved(
        cls,
        revised: Plan,
        goals: list[Goal],
    ) -> str:
        """Ensure USER_STEER-sourced goals still appear in the revision.

        Returns ``""`` when every USER_STEER goal is referenced by the
        revised plan's task titles / descriptions (or is present in
        ``revised.goal_ids``), and a non-empty error string otherwise.
        The retry loop feeds that string back into the correction
        prompt so the LLM gets a precise diagnosis of which goal it
        silently dropped. See goldfive#154.

        The "reference" check is token-based and uses *discriminative*
        tokens only: a token from a sticky goal's summary that also
        appears in a non-sticky goal's summary is too generic to be a
        signal (e.g. "goldfish" in a post-about-goldfish session is
        present in every goal and so cannot tell you whether the
        sticky goal was addressed). Discriminative tokens are summary
        tokens that DO NOT appear in any non-sticky goal. If a sticky
        goal has no discriminative tokens (the operator steered in a
        way indistinguishable from the base goals), we fall back to a
        lenient "goal id in revised.goal_ids" check -- better to let
        the revision through than to reject forever on ambiguous text.
        """
        sticky = cls._user_steer_goals(goals)
        if not sticky:
            return ""
        non_sticky_tokens: set[str] = set()
        for g in goals:
            if g.source == GOAL_SOURCE_USER_STEER:
                continue
            non_sticky_tokens |= cls._goal_summary_tokens(g)
        plan_text = " ".join(f"{t.id} {t.title} {t.description}".lower() for t in revised.tasks)
        revised_goal_ids = {gid.lower() for gid in revised.goal_ids if gid}
        dropped: list[str] = []
        for g in sticky:
            # Goal-id match on the revised plan envelope is a strong,
            # unambiguous signal — respect it first.
            if g.id and g.id.lower() in revised_goal_ids and g.id.lower() in plan_text:
                continue
            discriminative = cls._goal_summary_tokens(g) - non_sticky_tokens
            # Fallback 1: the goal id itself is discriminative.
            if g.id and g.id.lower() not in non_sticky_tokens:
                discriminative.add(g.id.lower())
            if not discriminative:
                # No tokens unique to this sticky goal; we cannot
                # distinguish it from the base goals via text alone.
                # Be lenient and accept the revision.
                continue
            if any(tok in plan_text for tok in discriminative):
                continue
            dropped.append(f"[{g.id or '(no-id)'}] {g.summary}")
        if not dropped:
            return ""
        joined = "; ".join(dropped)
        return (
            "revision silently drops USER_STEER goal(s) (no task "
            f"references them): {joined}. Operator steers are sticky — "
            "add PENDING tasks that explicitly advance these goals, "
            "or emit the reject sentinel if they cannot be reconciled."
        )

    @staticmethod
    def _extract_rejection_kind(error: str) -> str | None:
        """Bucket validator rejection messages by structural class.

        Returns one of ``"terminal_missing"``, ``"terminal_regressed"``,
        ``"edge_missing"``, or ``None`` when the error does not match a
        known validator-rejection pattern (e.g. plumbing failure, parse
        error, goal-coverage / assignee error). Used by the refine
        retry loop to short-circuit when consecutive attempts produce
        the same structural-class violation — feeding the LLM its own
        last error a second time on the same class is empirically
        unproductive on Qwen 35B and burns ~10s/attempt.

        Matching is substring-based (case-insensitive) so the helper is
        robust to the ``_user_steer_one_attempt`` wrapper that prefixes
        the validator message with ``"validator rejected revision: "``.
        Validator strings come from ``Plan.validate`` in ``types.py``:

        * ``"terminal task {id!r} missing in revision"``
        * ``"terminal task {id!r} regressed to {status!r}"``
        * ``"terminal->terminal edge {a!r} -> {b!r} missing in revision"``
        """
        if not error:
            return None
        lowered = error.lower()
        if "terminal task" in lowered and "missing in revision" in lowered:
            return "terminal_missing"
        if "terminal task" in lowered and "regressed to" in lowered:
            return "terminal_regressed"
        if "terminal->terminal edge" in lowered and "missing in revision" in lowered:
            return "edge_missing"
        return None

    # Regexes for parsing structural-class validator rejections so the
    # retry prompt can render a copy-paste-ready snippet of the missing
    # piece. Tolerate single OR double quotes — Python's ``!r`` always
    # emits single quotes for ASCII strings, but be defensive in case a
    # future validator change switches to double quotes. The capture
    # groups intentionally use a non-greedy ``[^'"]*`` body so we never
    # span across multiple quoted segments if two errors get
    # concatenated in a single string.
    _EDGE_MISSING_RE: re.Pattern[str] = re.compile(
        r"terminal->terminal edge ['\"]([^'\"]*)['\"] -> ['\"]([^'\"]*)['\"] missing in revision",
        re.IGNORECASE,
    )
    _TERMINAL_MISSING_RE: re.Pattern[str] = re.compile(
        r"terminal task ['\"]([^'\"]*)['\"] missing in revision",
        re.IGNORECASE,
    )
    _TERMINAL_REGRESSED_RE: re.Pattern[str] = re.compile(
        r"terminal task ['\"]([^'\"]*)['\"] regressed to ['\"]([^'\"]*)['\"]",
        re.IGNORECASE,
    )

    @classmethod
    def _structural_correction_snippet(cls, error: str) -> str:
        """Render a copy-paste-ready snippet for a structural rejection.

        Returns ``""`` when the error doesn't match a structural class or
        when the matched class's regex fails to recover the expected
        capture groups (e.g. malformed validator output). The caller
        falls back silently to the unenriched correction prompt in that
        case — never raises.

        The snippets intentionally name only the missing piece (the id
        or the (from, to) pair); the full prior-plan terminal block is
        already rendered upstream by
        :meth:`_render_structural_invariants_block`, so we don't need
        to reconstruct task objects here.
        """
        if not error:
            return ""
        kind = cls._extract_rejection_kind(error)
        if kind is None:
            return ""
        if kind == "edge_missing":
            m = cls._EDGE_MISSING_RE.search(error)
            if m is None:
                return ""
            from_id, to_id = m.group(1), m.group(2)
            edge_json = json.dumps({"from_task_id": from_id, "to_task_id": to_id})
            return (
                "ADD THIS EDGE VERBATIM TO YOUR `edges` ARRAY (keep all "
                "other edges you already have; just add this one):\n"
                f"    {edge_json}"
            )
        if kind == "terminal_missing":
            m = cls._TERMINAL_MISSING_RE.search(error)
            if m is None:
                return ""
            task_id = m.group(1)
            return (
                "KEEP THIS TASK VERBATIM IN YOUR `tasks` ARRAY with its "
                "current terminal status (do not regress it to "
                "PENDING/RUNNING/BLOCKED):\n"
                f'    task id: "{task_id}"'
            )
        if kind == "terminal_regressed":
            m = cls._TERMINAL_REGRESSED_RE.search(error)
            if m is None:
                return ""
            task_id, regressed_status = m.group(1), m.group(2)
            return (
                "RESTORE THIS TASK'S TERMINAL STATUS verbatim from the "
                "prior plan (do not regress it):\n"
                f'    task id: "{task_id}" (currently regressed to '
                f'"{regressed_status}" — must remain its prior terminal '
                "status as listed in the STRUCTURAL INVARIANTS block "
                "above)"
            )
        return ""

    @classmethod
    def _build_correction_prompt(cls, base_prompt: str, error: str) -> str:
        """Append validator / parser feedback to a refine prompt for retry.

        The retry appends the specific error the previous attempt hit and
        re-emphasises the structural invariants; keeping the base prompt
        verbatim means the LLM sees the same goals, history, and drift
        context, so the retry is a real second attempt, not a cold
        re-prompt.

        For structurally-classified rejections (terminal task missing,
        terminal task regressed, terminal->terminal edge missing) the
        prompt also embeds a copy-paste-ready snippet of EXACTLY what
        the LLM must add or preserve. Empirically the LLM drops at least
        one terminal-edge on long-context refines even with the
        invariants block present further up; naming the missing piece
        right next to the rejection text fixes that on attempt 2.
        Snippet rendering falls back silently on parse failure so
        unbucketed / malformed errors retain the prior unenriched
        behaviour exactly.
        """
        snippet = cls._structural_correction_snippet(error)
        snippet_block = f"\n\n{snippet}" if snippet else ""
        return (
            f"{base_prompt}\n\n"
            "PREVIOUS ATTEMPT FAILED. The response you just emitted was "
            "rejected by the validator:\n"
            f"    {error}\n\n"
            "Re-read the STRUCTURAL INVARIANTS section above. Emit a "
            "corrected JSON plan that preserves every terminal task and "
            "every terminal->terminal edge verbatim, and does NOT add "
            "any edge from a CANCELLED or FAILED task to a new PENDING "
            "task. Respond with JSON only; no prose, no markdown "
            f"fences.{snippet_block}"
        )

    async def _emit_refine_validation_failed(
        self, plan: Plan, drift: DriftEvent, last_error: str
    ) -> None:
        """Emit a ``REFINE_VALIDATION_FAILED`` drift via the wired emitter.

        A no-op when no emitter is set (e.g. unit tests). The emitter
        failing itself is logged and swallowed -- the caller still needs
        to fall back to a deterministic plan regardless.
        """
        if self._drift_emitter is None:
            return
        signal = DriftEvent(
            kind=DriftKind.REFINE_VALIDATION_FAILED,
            severity=DriftSeverity.CRITICAL,
            detail=(
                f"refine validation failed after {self._max_refine_attempts} "
                f"attempts for drift={drift.kind.value}: {last_error}"
            ),
            current_task_id=drift.current_task_id,
            current_agent_id=drift.current_agent_id,
            # Inherit the originating drift's observed revision so the
            # dispatch-time gate (goldfive#245) treats this synthetic
            # escalation with the same freshness contract as its parent.
            observed_revision_index=drift.observed_revision_index,
        )
        try:
            await self._drift_emitter(signal)
        except Exception as exc:  # noqa: BLE001 -- emitter must never break refine
            log.warning(
                "LLMPlanner: drift emitter raised while signalling "
                "REFINE_VALIDATION_FAILED (%s); dropping signal",
                exc,
            )

    async def _emit_refine_orphaned_tasks(
        self,
        prior_plan: Plan,
        revised: Plan,
        orphans: list[Task],
    ) -> None:
        """Emit a ``refine_orphaned_tasks`` sink event for operator visibility.

        Fires when :func:`_check_supersedes_coverage` finds prior tasks
        that were dropped by the refine output without a supersedes
        link or an absorbing-terminal status. This is **telemetry only**
        — the refine still applies. Some orphans are legitimate (a
        scope-narrowing user steer can validly remove a task with no
        replacement concept), so blocking would be too strict; surfacing
        gives operators the visibility to spot careless drops.

        Routes through the planner's span-context provider (the same
        snapshot used by ``goldfive_llm_span``) so the event lands on
        every bound sink with the correct ``run_id`` / ``session_id`` /
        sequence. A no-op when the planner is used standalone (no
        provider, e.g. unit tests not exercising sinks).
        """
        if not orphans:
            return
        if self._span_ctx_provider is None:
            return
        try:
            ctx = self._span_ctx_provider()
        except Exception:  # noqa: BLE001 -- observability must never break refine
            return
        if ctx is None:
            return
        sinks, run_id, session_id, _task_id, seq_fn = ctx
        if not sinks or seq_fn is None:
            return
        try:
            from goldfive.events import emit, make_event  # noqa: PLC0415 — lazy
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "LLMPlanner: events module unavailable; dropping refine_orphaned_tasks signal (%s)",
                exc,
            )
            return
        payload: dict[str, Any] = {
            "prior_plan_id": prior_plan.id,
            "prior_revision_index": prior_plan.revision_index,
            "revised_plan_id": revised.id,
            "revised_revision_index": revised.revision_index,
            "orphan_count": len(orphans),
            "orphans": [
                {
                    "task_id": t.id,
                    "title": t.title,
                    "status": t.status.value,
                    "assignee_agent_id": t.assignee_agent_id,
                }
                for t in orphans
            ],
        }
        try:
            seq = seq_fn()
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "LLMPlanner: sequence_fn raised; dropping refine_orphaned_tasks signal (%s)",
                exc,
            )
            return
        try:
            evt = make_event(
                run_id or "",
                seq,
                "refine_orphaned_tasks",
                payload,
                session_id=session_id or "",
            )
            await emit(list(sinks), evt)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LLMPlanner: failed to emit refine_orphaned_tasks event (%s); dropping signal",
                exc,
            )

    async def _call_and_validate_refine(
        self,
        *,
        system_prompt: str,
        base_user_prompt: str,
        prior_plan: Plan,
        goals: list[Goal],
        post_parse: Callable[[Plan], Plan] | None = None,
        log_prefix: str,
        allow_reject: bool = False,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        span_name: str = "refine",
        span_input_preview: str = "",
        span_target_agent_id: str = "",
        span_target_task_id: str = "",
        span_decision_prefix: str = "refine",
    ) -> tuple[Plan | None, str, bool]:
        """Run the retry loop for a single refine call.

        Calls the LLM up to ``self._max_refine_attempts`` times, parsing
        the response into a ``Plan`` and validating against ``prior_plan``.
        On each failed attempt the user prompt is extended with the
        error message (via :meth:`_build_correction_prompt`) so the LLM
        gets explicit feedback on what to fix. Returns
        ``(plan, "", False)`` on success, ``(None, last_error, False)``
        on exhaustion, and ``(None, reason, True)`` when the LLM emitted
        a ``{"reject": true, ...}`` sentinel (only honoured when
        ``allow_reject`` is True). Exceptions from ``call_llm`` are
        treated as retryable.

        ``post_parse`` (optional) runs after ``_plan_from_json`` succeeds
        but BEFORE validation -- used by ``_refine_looping_tool_call`` to
        force the looper FAILED before the validator decides whether the
        revision honours the invariants.

        ``allow_reject`` (optional, default False): when True, a JSON
        object shaped ``{"reject": true, "reason": "..."}`` is a valid
        terminal outcome — the planner returns ``(None, reason, True)``
        and the caller is expected to treat it as "escalate to human
        intervention" rather than a validation failure. Used by the
        PLAN_DIVERGENCE path (goldfive#144).
        """
        from goldfive._llm_span import goldfive_llm_span

        user_prompt = base_user_prompt
        last_error = ""
        attempts = max(1, self._max_refine_attempts)
        for attempt in range(1, attempts + 1):
            # Per-attempt span. decision_summary / output_preview is
            # stamped inside the with-block; the outer retry loop's
            # parse/validate branches don't see the handle. The span
            # therefore carries "what the LLM returned" rather than
            # "what the validator decided" — the validator's verdict
            # becomes the aggregated refine outcome on the enclosing
            # ``PlanRevised`` event (which already carries
            # ``refine_output_summary``). This split keeps the span
            # emission lean inside the tight retry loop.
            span_kwargs = self._span_kwargs(task_id_override=span_target_task_id)
            try:
                async with goldfive_llm_span(
                    **span_kwargs,
                    name=span_name,
                    input_preview=span_input_preview,
                    target_agent_id=span_target_agent_id,
                    target_task_id=span_target_task_id,
                ) as span:
                    # Cap the underlying LLM dispatch so a runaway
                    # generation (Qwen Q4 thinking-token explosion)
                    # cannot wedge the run for minutes. See
                    # ``LLMPlanner.MAX_OUTPUT_TOKENS``. Also disable
                    # thinking (goldfive#271 follow-up to #311): refine
                    # is a structured JSON revision call, not deep
                    # reasoning — burning 16k on ``<think>`` produced
                    # empty responses in v16 evidence.
                    from goldfive._llm import (
                        call_llm_budget,
                        call_llm_thinking_disabled,
                    )

                    with (
                        call_llm_budget(self.MAX_OUTPUT_TOKENS),
                        call_llm_thinking_disabled(),
                    ):
                        raw = await self._call_llm(system_prompt, user_prompt, self._model)
                    span.output_preview = (
                        raw[:4096] if isinstance(raw, str) else "(non-str response)"
                    )
                    span.decision_summary = (
                        f"{span_decision_prefix} attempt {attempt}/{attempts}: "
                        f"LLM returned {len(raw) if isinstance(raw, str) else 0} chars"
                    )
            except Exception as exc:  # noqa: BLE001 -- retry on transient LLM errors
                last_error = f"call_llm raised: {exc}"
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            if not raw or not isinstance(raw, str):
                # Small-model artefact (goldfive#182): Qwen 2B and other
                # small models routinely exhaust their output budget on
                # internal reasoning and emit no final answer. Retrying
                # doubles cost without changing the outcome, so treat the
                # empty response as terminal "no signal" and let the
                # caller fall back to its no-revision branch. Logged at
                # INFO so operators can still see model-quality issues
                # via observability without the WARNING noise that
                # previously cascaded through retry/escalate paths.
                last_error = _EMPTY_RESPONSE_ERROR
                log.info(
                    "%s: attempt %d/%d: %s; not retrying",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                break
            cleaned = _strip_code_fences(raw).strip()
            try:
                parsed = json.loads(cleaned)
            except (ValueError, TypeError) as exc:
                last_error = f"JSON parse failed: {exc}"
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            # Honour the reject sentinel: {"reject": true, "reason": "..."}
            # is a terminal outcome on the plan-divergence path (#144).
            # The caller escalates to human intervention; we return
            # immediately without retrying.
            if allow_reject and isinstance(parsed, Mapping) and bool(parsed.get("reject")):
                reason = str(parsed.get("reason") or "").strip() or "(no reason provided)"
                log.info(
                    "%s: LLM emitted reject sentinel (reason=%s); escalating to human intervention",
                    log_prefix,
                    reason,
                )
                return None, reason, True
            revised = _plan_from_json(
                parsed,
                run_id=prior_plan.run_id,
                goal_ids=[g.id for g in goals if g.id] or list(prior_plan.goal_ids),
                plan_id=prior_plan.id,
            )
            if revised is None:
                last_error = "parsed JSON did not contain a usable plan"
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            if post_parse is not None:
                try:
                    revised = post_parse(revised)
                except Exception as exc:  # noqa: BLE001
                    last_error = f"post-parse adjustment raised: {exc}"
                    log.warning(
                        "%s: attempt %d/%d: %s",
                        log_prefix,
                        attempt,
                        attempts,
                        last_error,
                    )
                    if attempt < attempts:
                        user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                    continue
            # goldfive#213: backfill ``Task.supersedes`` for retry-named
            # tasks the LLM emitted without an explicit causal link.
            # Mirrors the call in ``_user_steer_one_attempt``'s merge
            # so that LOOPING_TOOL_CALL / LOOPING_REASONING refines (and
            # any other path through ``_call_and_validate_refine``)
            # benefit from causal replacement detection identically to
            # the user-steer path. Must run BEFORE
            # ``_normalize_supersession_kinds`` so the kind coercion
            # sees the backfilled link.
            if prior_plan is not None:
                # goldfive#247: returns a NEW Plan; rebind so subsequent
                # validator + normalize sees the backfilled tasks.
                revised = _backfill_retry_supersedes(revised, prior=prior_plan)
            # goldfive#251: Option B validator -- coerce supersedes_kind
            # on every task based on old-task status. Runs before
            # ``revised.validate`` so the structural validator sees
            # self-consistent kinds. Warnings only, never rejects
            # (Option B is about making the LLM's intent survive rather
            # than blocking a structurally-valid plan). goldfive#247:
            # returns a NEW Plan; rebind so the validator sees the
            # coerced kinds.
            revised = _normalize_supersession_kinds(revised, prior=prior_plan)
            try:
                revised.validate(for_revision=True, prior=prior_plan)
            except ValueError as exc:
                last_error = f"validator rejected revision: {exc}"
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            # Goal-awareness check (#154): any USER_STEER-sourced goals
            # that the prior plan was carrying must still be addressed
            # by the revised plan. Silently dropping them would let a
            # later drift unwind an operator steer -- the very failure
            # mode this issue was filed against. Treat a drop as a
            # validator failure so the retry loop feeds the LLM an
            # explicit correction message.
            goal_error = self._check_user_steer_goals_preserved(revised, goals)
            if goal_error:
                last_error = goal_error
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            # Registry check (goldfive#151): off-registry assignees
            # are a retryable validator error — the correction loop
            # feeds the list of legal names back on the next attempt.
            assignee_error = self._validate_plan_assignees(revised, available_agents)
            if assignee_error:
                last_error = assignee_error
                log.warning(
                    "%s: attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
                continue
            # Supersedes coverage (observability only — never rejects).
            # For every prior task absent from the revision, check that
            # some new task supersedes it OR that the prior status is
            # absorbing-terminal (FAILED/CANCELLED). Surviving orphans
            # are surfaced as a WARNING + ``refine_orphaned_tasks`` sink
            # event so operators can spot careless drops without
            # blocking legitimate scope-narrowing refines.
            orphans = _check_supersedes_coverage(revised, prior=prior_plan)
            if orphans:
                log.warning(
                    "%s: refine output dropped %d prior task(s) without a "
                    "supersedes link or terminal status: %s",
                    log_prefix,
                    len(orphans),
                    ", ".join(f"{t.id!r} ({t.title!r})" for t in orphans),
                )
                await self._emit_refine_orphaned_tasks(prior_plan, revised, orphans)
            return revised, "", False
        return None, last_error, False

    def _build_refine_prompt(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> str:
        current = {
            "id": plan.id,
            "summary": plan.summary,
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "assignee_agent_id": t.assignee_agent_id,
                    "status": str(t.status),
                    "predicted_start_ms": t.predicted_start_ms,
                    "predicted_duration_ms": t.predicted_duration_ms,
                }
                for t in plan.tasks
            ],
            "edges": [
                {"from_task_id": e.from_task_id, "to_task_id": e.to_task_id} for e in plan.edges
            ],
        }
        drift_payload = {
            "kind": str(drift.kind),
            "severity": str(drift.severity),
            "detail": drift.detail,
            "current_task_id": drift.current_task_id,
            "current_agent_id": drift.current_agent_id,
        }
        plan_json = json.dumps(current, default=str)
        drift_json = json.dumps(drift_payload, default=str)
        goals_block = self._render_goals_block(goals)
        invariants_block = self._render_structural_invariants_block(plan)
        sticky_block = self._render_sticky_goals_block(goals)
        agents_block = ""
        if available_agents:
            header = (
                "AGENT TREE (pick assignee_agent_id from this registry):"
                if _is_tree_entry_list(available_agents)
                else "Available agents:"
            )
            agents_block = f"{header}\n{self._render_agents_block(available_agents)}\n\n"
        observed_block = ""
        if observed_actions is not None:
            observed_block = (
                f"{self._render_observed_actions_block(observed_actions)}\n\n"
                "The tree may have diverged from the planned dispatch because it "
                "found a better path. Decide:\n"
                "- ABSORB: if the observed activity moves toward the GOALS "
                "(and preserves every STICKY goal), produce a revised plan "
                "that REFLECTS the observed activity (mark matching tasks "
                "COMPLETED/RUNNING, add new tasks for agent invocations not "
                "already in the plan).\n"
                "- REJECT: if the observed activity CONTRADICTS any goal -- "
                "especially any STICKY goal added by a prior USER_STEER -- "
                'return a JSON object of the form {"reject": true, "reason": '
                '"..."} (the caller will escalate to human intervention).\n\n'
            )
        elif drift.kind in (DriftKind.OFF_TOPIC, DriftKind.JUSTIFIED_DEVIATION):
            # OFF_TOPIC and JUSTIFIED_DEVIATION (iter-10 PR 4) have no
            # observed-actions channel — the deviation is in the
            # agent's reasoning, surfaced by the reasoning judge.
            # Render an analogous "what the agent did" block from the
            # drift's reasoning context (``trigger_input`` is the
            # truncated reasoning text the judge saw; ``raw`` is the
            # original, when set; ``detail`` is the judge's free-form
            # reason — for JUSTIFIED_DEVIATION the detail string
            # already carries the provenance prefix from the parser,
            # e.g. "justified deviation (tool_error): ..."). Frame it
            # with the same ABSORB/REJECT contract so the goal-aware
            # system prompt's decision shape carries through.
            observed_block = (
                f"{self._render_off_topic_reasoning_block(drift)}\n\n"
                "The agent's reasoning has drifted from the bound task. "
                "Decide:\n"
                "- ABSORB: if the new reasoning topic plausibly moves toward "
                "the GOALS (and preserves every STICKY goal), produce a "
                "revised plan that reflects the new direction (add or revise "
                "tasks so the work the agent is now reasoning about has a "
                "place in the plan).\n"
                "- REJECT: if the new reasoning topic CONTRADICTS any goal "
                "-- especially any STICKY goal added by a prior USER_STEER "
                "-- or simply does not advance any goal, return a JSON "
                'object of the form {"reject": true, "reason": "..."} '
                "(the caller will escalate to human intervention).\n\n"
            )
        return (
            f"{agents_block}"
            f"CURRENT GOALS (the revision must still advance every goal, "
            f"and MUST NOT silently drop any [STICKY] goal):\n{goals_block}\n\n"
            f"{sticky_block}"
            f"Current plan:\n{plan_json}\n\n"
            f"Drift event:\n{drift_json}\n\n"
            f"{observed_block}"
            f"{_REFINEMENT_GUIDANCE_BLOCK}\n\n"
            f"{invariants_block}\n\n"
            "If the plan should change in light of this drift event, respond "
            "with an updated JSON plan using the same schema. If no change "
            "is warranted, respond with the current plan unchanged. Respond "
            "with JSON only."
        )

    @classmethod
    def _render_sticky_goals_block(cls, goals: list[Goal]) -> str:
        """Render a dedicated STICKY GOALS block when sticky goals exist.

        Returns ``""`` when there are no USER_STEER-sourced goals, so
        legacy prompts for non-steered sessions are unchanged. When at
        least one sticky goal is present, surface an explicit section
        naming them so the LLM cannot "forget" that dropping them will
        fail validation (goldfive#154).
        """
        sticky = cls._user_steer_goals(goals)
        if not sticky:
            return ""
        lines = [
            "STICKY GOALS (added by USER_STEER — the refine validator will "
            "REJECT any revision that does not include at least one task "
            "advancing each of these; if the drift IRRECONCILABLY "
            'contradicts a sticky goal, emit {"reject": true, "reason": '
            '"..."} instead of silently dropping it):'
        ]
        for g in sticky:
            gid = g.id or "(no-id)"
            lines.append(f"- [{gid}] {g.summary}")
        return "\n".join(lines) + "\n\n"

    # ---- Planner protocol ------------------------------------------------

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        """Produce the initial plan for ``goals``.

        ``available_agents`` may be a plain ``list[str]`` (legacy
        callers) or a structured tree as produced by
        :attr:`goldfive.adapters.adk.ADKAdapter.available_agents_tree`
        (goldfive#151). When the tree form is supplied the prompt
        renders a richer "AGENT TREE" section and the validator
        rejects any task whose ``assignee_agent_id`` is not in the
        registry — the retry-with-correction loop (#133/#136) feeds
        the validator message back to the LLM on the next attempt.
        An empty / None registry skips the assignee check for
        back-compat.
        """
        if not goals:
            log.debug("LLMPlanner.generate: no goals provided; skipping plan")
            return None
        base_prompt = self._build_generate_prompt(goals, available_agents, context)
        from goldfive._llm_span import goldfive_llm_span

        run_id = ""
        if context is not None:
            run_id = str(context.get("run_id") or "")
        # ``plan_generate`` is trajectory-level: the initial plan
        # precedes any task binding so ``target_agent_id`` /
        # ``target_task_id`` stay empty. A compact user-request + goals
        # rendering doubles as ``input_preview`` so harmonograf can
        # render "what was goldfive planning for?" on the Gantt.
        user_request = ""
        if context is not None:
            user_request = str(context.get("user_request") or "")
        goal_lines = [f"- [{g.id or '(no-id)'}] {g.summary or '(no summary)'}" for g in goals]
        generate_input_preview = (
            (f"user_request: {user_request}\n\n" if user_request else "")
            + "goals:\n"
            + "\n".join(goal_lines)
        )
        user_prompt = base_prompt
        last_error = ""
        attempts = max(1, self._max_refine_attempts)
        for attempt in range(1, attempts + 1):
            try:
                async with goldfive_llm_span(
                    **self._span_kwargs(),
                    name="plan_generate",
                    input_preview=generate_input_preview,
                ) as span:
                    # Bound the dispatch — see ``LLMPlanner.MAX_OUTPUT_TOKENS``.
                    # Disable thinking — plan_generate emits structured
                    # JSON describing tasks, not deep reasoning.
                    from goldfive._llm import (
                        call_llm_budget,
                        call_llm_thinking_disabled,
                    )

                    with (
                        call_llm_budget(self.MAX_OUTPUT_TOKENS),
                        call_llm_thinking_disabled(),
                    ):
                        raw = await self._call_llm(self._system_prompt, user_prompt, self._model)
                    span.output_preview = (
                        raw[:4096] if isinstance(raw, str) else "(non-str response)"
                    )
                    span.decision_summary = (
                        f"plan_generate attempt {attempt}/{attempts}: "
                        f"LLM returned {len(raw) if isinstance(raw, str) else 0} chars"
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = f"call_llm raised: {exc}"
                log.warning(
                    "LLMPlanner.generate: attempt %d/%d: %s",
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_prompt, last_error)
                continue
            if not raw or not isinstance(raw, str):
                # See ``_call_and_validate_refine`` for the rationale —
                # an empty response is treated as terminal "no signal"
                # rather than retried (goldfive#182).
                last_error = _EMPTY_RESPONSE_ERROR
                log.info(
                    "LLMPlanner.generate: attempt %d/%d: %s; not retrying",
                    attempt,
                    attempts,
                    last_error,
                )
                break
            cleaned = _strip_code_fences(raw).strip()
            try:
                parsed = json.loads(cleaned)
            except (ValueError, TypeError) as exc:
                last_error = f"JSON parse failed: {exc}"
                log.warning(
                    "LLMPlanner.generate: attempt %d/%d: %s",
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_prompt, last_error)
                continue
            plan = _plan_from_json(
                parsed,
                run_id=run_id,
                goal_ids=[g.id for g in goals if g.id],
            )
            if plan is None:
                last_error = "parsed JSON did not contain a usable plan"
                log.warning(
                    "LLMPlanner.generate: attempt %d/%d: %s",
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_prompt, last_error)
                continue
            try:
                plan.validate(for_revision=False)
            except ValueError as exc:
                last_error = f"validator rejected plan: {exc}"
                log.warning(
                    "LLMPlanner.generate: attempt %d/%d: %s",
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_prompt, last_error)
                continue
            # Registry check (goldfive#151): reject off-registry
            # assignees so the tree-aware retry loop can ask the LLM
            # to re-pick a legal name. Skipped when no registry
            # supplied — preserves back-compat with pre-#151 callers.
            assignee_error = self._validate_plan_assignees(plan, available_agents)
            if assignee_error:
                last_error = assignee_error
                log.warning(
                    "LLMPlanner.generate: attempt %d/%d: %s",
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    user_prompt = self._build_correction_prompt(base_prompt, last_error)
                continue
            return plan
        # Empty / non-string responses already logged INFO at the
        # short-circuit site (goldfive#182); avoid a redundant WARNING
        # tail. Other exhausted-retry paths still log WARNING so a
        # genuine model-output / validator failure stays visible.
        if last_error == _EMPTY_RESPONSE_ERROR:
            return None
        log.warning(
            "LLMPlanner.generate: exhausted %d attempt(s); last_error=%s",
            attempts,
            last_error,
        )
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        """Produce a revised plan in response to a drift event.

        ``observed_actions`` (goldfive#144) is optional and only
        consulted when ``drift.kind is DriftKind.PLAN_DIVERGENCE`` —
        every other refine path ignores it so back-compat with existing
        callers that don't pass the argument is preserved. When present
        on the divergence path, the reconciler-style prompt asks the
        LLM to either ABSORB the observed activity into a revised plan
        or REJECT by emitting ``{"reject": true, "reason": "..."}``. A
        reject collapses to ``None`` — the steerer then escalates via
        the intervention ladder (goldfive#142).

        ``available_agents`` (goldfive#151) is optional. When provided
        the refine prompt renders an "AGENT TREE" section so the LLM
        picks ``assignee_agent_id`` values from the real tree, and the
        retry-with-correction loop rejects off-registry assignees with
        a concrete error message feeding the next attempt. Accepts
        either ``list[str]`` (legacy) or the structured tree form
        produced by :attr:`ADKAdapter.available_agents_tree`.
        """
        if plan is None:
            return None
        if drift.kind is DriftKind.USER_STEER:
            return await self._refine_steer(plan, drift, goals, available_agents, source="user")
        if drift.kind in (
            DriftKind.LOOPING_TOOL_CALL,
            DriftKind.LOOPING_REASONING,
        ):
            # LOOPING_REASONING shares the "fail the current task, route
            # around it" shape with LOOPING_TOOL_CALL: the symptom is a
            # stuck loop on the currently-running task, and the repair
            # is to mark it FAILED and regenerate the rest.
            return await self._refine_looping_tool_call(plan, drift, goals, available_agents)
        # REFINE_VALIDATION_FAILED is a terminal signal from ourselves --
        # refining on it would risk an infinite loop of validation
        # failures. The steerer is expected to skip refine for this
        # kind; defending here too is a belt-and-braces guard.
        if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
            return None
        # Plan-context drifts (PLAN_DIVERGENCE, OFF_TOPIC reasoning drift,
        # JUSTIFIED_DEVIATION) take the goal-aware ABSORB/REJECT prompt:
        # the LLM must either revise the plan to reflect / accept the
        # observed deviation, or emit the reject sentinel when the
        # deviation contradicts the goals.
        #
        # iter-12 (#220): prompt selection is by drift kind ALONE, not
        # gated on ``observed_actions is not None``. Production callers
        # (steerer ``_handle_drift``, the steer-promotion fallback, the
        # reconciler emission path) all invoke ``refine`` without the
        # ``observed_actions`` parameter, so the previous gate caused
        # PLAN_DIVERGENCE drifts in production to silently fall through
        # to the generic refine prompt. Selecting by kind makes drift
        # routing emission-site-independent: any PLAN_DIVERGENCE gets
        # the goal-aware framing, with the drift's ``detail`` carrying
        # the divergence context (e.g. "off-plan agent X",
        # "cross-layer delegation Y"). The OBSERVED ACTIVITY block is
        # still rendered only when ``observed_actions`` is supplied;
        # without it the goal-aware prompt still receives goals + plan
        # + drift detail, which PR #345 demonstrated is sufficient
        # context for the OFF_TOPIC path.
        #
        # OFF_TOPIC and JUSTIFIED_DEVIATION (iter-10 PR 4) have no
        # observed-actions channel — the deviation is in the agent's
        # reasoning, surfaced by the reasoning judge. The drift's
        # ``detail`` / ``trigger_input`` carries the judge's reasoning
        # excerpt, which ``_render_off_topic_reasoning_block`` surfaces
        # verbatim under the same ABSORB/REJECT framing.
        is_plan_divergence = drift.kind is DriftKind.PLAN_DIVERGENCE
        is_off_topic = drift.kind is DriftKind.OFF_TOPIC
        is_justified = drift.kind is DriftKind.JUSTIFIED_DEVIATION
        use_divergence_prompt = is_plan_divergence or is_off_topic or is_justified
        try:
            base_user_prompt = self._build_refine_prompt(
                plan,
                drift,
                goals,
                observed_actions=observed_actions,
                available_agents=available_agents,
            )
        except (TypeError, ValueError) as exc:
            log.warning("LLMPlanner.refine: failed to serialise inputs (%s)", exc)
            return None
        system_prompt = (
            self._plan_divergence_system_prompt
            if use_divergence_prompt
            else self._refine_system_prompt
        )
        # Allow the reject sentinel whenever the LLM might legitimately
        # conclude the drift cannot be reconciled with the current goals
        # (goldfive#154). True on every plan-context drift path
        # (PLAN_DIVERGENCE / OFF_TOPIC / JUSTIFIED_DEVIATION) -- those
        # are the kinds wired to the goal-aware ABSORB/REJECT contract
        # from #144 / #345. Also true whenever the session carries a
        # sticky USER_STEER goal -- an unrelated drift might surface
        # work that irreconcilably contradicts the operator's steer,
        # and escalating via reject is cleaner than exhausting retries.
        # Other callers keep the legacy "parse or bust" semantics.
        allow_reject = use_divergence_prompt or bool(self._user_steer_goals(goals))
        refine_input_preview = self._build_refine_span_input_preview(drift, plan)
        revised, last_error, rejected = await self._call_and_validate_refine(
            system_prompt=system_prompt,
            base_user_prompt=base_user_prompt,
            prior_plan=plan,
            goals=goals,
            log_prefix="LLMPlanner.refine",
            allow_reject=allow_reject,
            available_agents=available_agents,
            span_name="refine",
            span_input_preview=refine_input_preview,
            span_target_agent_id=drift.current_agent_id or "",
            span_target_task_id=drift.current_task_id or "",
            span_decision_prefix=f"refine ({drift.kind.value})",
        )
        if rejected:
            # LLM judged the divergence off-goal. Return None so the
            # steerer escalates via the intervention ladder (#142).
            # No REFINE_VALIDATION_FAILED emission here -- a reject is
            # a successful decision, not a validation failure.
            return None
        if revised is None:
            # Retries exhausted. Emit the REFINE_VALIDATION_FAILED signal
            # so the UI / operator sees that recovery failed, then
            # return ``None`` -- the steerer's backoff counter takes
            # over from here (REFINE_FAILURE_THRESHOLD = 2 consecutive
            # failures marks the task FAILED). We intentionally do NOT
            # synthesise a clone of the prior plan with a bumped
            # ``revision_index``: that would masquerade a failed refine
            # as a successful no-op revision, which is exactly the
            # silent-fallback behaviour goldfive#133 set out to
            # eliminate.
            #
            # Exception (goldfive#182): when the failure is an empty /
            # non-string LLM response, skip the validation-failed
            # emission. That signal escalates through the intervention
            # ladder, but a small-model "no answer" is not a planner
            # failure — it's a model-quality issue we already log at
            # INFO. Returning None still triggers the steerer's normal
            # backoff path; we just don't add escalation noise on top.
            if last_error != _EMPTY_RESPONSE_ERROR:
                await self._emit_refine_validation_failed(plan, drift, last_error)
            return None
        # Stamp revision metadata so downstream sinks can render it.
        # goldfive#247: Plan is frozen — derive a new instance with the
        # stamped metadata via :func:`bump_revision`.
        return bump_revision(
            revised,
            revision_index=plan.revision_index + 1,
            revision_reason=drift.detail,
            revision_kind=str(drift.kind),
            revision_severity=str(drift.severity),
        )

    def _build_looping_tool_call_prompt(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        looping_task: Task | None,
    ) -> str:
        """Build the regenerate-around-failure prompt for a LOOPING_TOOL_CALL.

        The prompt explicitly enumerates the structural invariants the
        validator will enforce (terminal-task preservation,
        terminal->terminal edge preservation, id uniqueness, acyclicity)
        so the planner-LLM has no reason to "lose" them silently. See
        goldfive#133 for the rationale.
        """
        completed = [t for t in plan.tasks if t.status in _TERMINAL_STATUSES]
        loop_id = drift.current_task_id or (looping_task.id if looping_task else "")
        history = [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "assignee_agent_id": t.assignee_agent_id,
                "status": str(t.status),
            }
            for t in completed
        ]
        others = [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "assignee_agent_id": t.assignee_agent_id,
                "status": str(t.status),
            }
            for t in plan.tasks
            if t.status not in _TERMINAL_STATUSES and t.id != loop_id
        ]
        looper_block = (
            json.dumps(
                {
                    "id": looping_task.id,
                    "title": looping_task.title,
                    "description": looping_task.description,
                    "assignee_agent_id": looping_task.assignee_agent_id,
                    "status": str(looping_task.status),
                },
                default=str,
            )
            if looping_task is not None
            else json.dumps({"id": loop_id})
        )
        goals_block = self._render_goals_block(goals)
        sticky_block = self._render_sticky_goals_block(goals)
        invariants_block = self._render_structural_invariants_block(plan)
        return (
            f"CURRENT GOALS (the revised plan must still advance every "
            f"goal, and MUST NOT silently drop any [STICKY] goal):\n"
            f"{goals_block}\n\n"
            f"{sticky_block}"
            f"Already-finished tasks (preserve verbatim):\n"
            f"{json.dumps(history, default=str)}\n\n"
            f"LOOPING task (must appear in the returned plan with "
            f"status=FAILED, same id):\n{looper_block}\n\n"
            f"Other unfinished tasks (you may keep, drop, or rework):\n"
            f"{json.dumps(others, default=str)}\n\n"
            f"Drift detail:\n{drift.detail}\n\n"
            f"{_REFINEMENT_GUIDANCE_BLOCK}\n\n"
            f"{invariants_block}\n\n"
            "Generate the updated plan. Respond with JSON only."
        )

    async def _refine_looping_tool_call(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        """Fail-and-regenerate path for ``LOOPING_TOOL_CALL`` drift.

        The looping task is forced to ``FAILED`` so the rest of the plan
        can route around it; non-looping completed tasks are preserved
        verbatim. The LLM regenerates the remaining work in light of the
        failure.

        Retry loop (goldfive#133): up to ``self._max_refine_attempts``
        attempts are made, each appending the prior failure's error
        message to the prompt so the LLM has explicit feedback on what
        to fix. On exhaustion we emit a ``REFINE_VALIDATION_FAILED``
        drift (CRITICAL) and fall back to the deterministic
        :meth:`_fallback_fail_loop_plan` -- losing the looper's slot is
        still better than leaving the run in a re-loop.
        """
        loop_id = drift.current_task_id
        looping_task = next((t for t in plan.tasks if t.id == loop_id), None)
        try:
            base_user_prompt = self._build_looping_tool_call_prompt(
                plan, drift, goals, looping_task
            )
        except (TypeError, ValueError) as exc:
            log.warning(
                "LLMPlanner._refine_looping_tool_call: serialise failed (%s)",
                exc,
            )
            return self._fallback_fail_loop_plan(plan, drift, looping_task)

        def _force_looper_failed(revised: Plan) -> Plan:
            """Stamp the looping task FAILED before validation.

            The protocol contract is that the looper cannot survive its
            own drift, so even if the LLM forgot we force the status.

            goldfive#247: Plan + Task are frozen — produce a NEW plan
            with the looper transitioned via :func:`replace_task` (or,
            when the looper was missing, prepended via dataclass
            replace + tuple concat).
            """
            if not loop_id:
                return revised
            # Step 1: flip the looper if it's still in the plan.
            existing = next((t for t in revised.tasks if t.id == loop_id), None)
            if existing is not None and existing.status not in _TERMINAL_STATUSES:
                revised = replace_task(revised, loop_id, status=TaskStatus.FAILED)
            # Step 2: prepend a synthetic FAILED looper if the LLM
            # dropped it and we have a snapshot to re-inject.
            if not any(t.id == loop_id for t in revised.tasks) and looping_task is not None:
                synthetic = Task(
                    id=looping_task.id,
                    title=looping_task.title,
                    description=looping_task.description,
                    assignee_agent_id=looping_task.assignee_agent_id,
                    status=TaskStatus.FAILED,
                    # goldfive#251: preserve the supersession provenance
                    # when re-inserting the looping task into the
                    # revised plan. Dropping these fields here turned a
                    # CORRECT-kind chain into an orphan, which the
                    # downstream supersedes-coverage validator would
                    # then flag as a regression (or, worse, the steerer
                    # would re-pin to the wrong successor). See the
                    # paired regression in test_planner.py.
                    supersedes=looping_task.supersedes,
                    supersedes_kind=looping_task.supersedes_kind,
                )
                revised = dataclasses.replace(
                    revised,
                    tasks=(synthetic,) + tuple(revised.tasks),
                )
            return revised

        refine_input_preview = self._build_refine_span_input_preview(drift, plan)
        revised, last_error, _rejected = await self._call_and_validate_refine(
            system_prompt=self._looping_tool_call_system_prompt,
            base_user_prompt=base_user_prompt,
            prior_plan=plan,
            goals=goals,
            post_parse=_force_looper_failed,
            log_prefix="LLMPlanner._refine_looping_tool_call",
            available_agents=available_agents,
            span_name="refine_looping_tool_call",
            span_input_preview=refine_input_preview,
            span_target_agent_id=drift.current_agent_id or "",
            span_target_task_id=drift.current_task_id or "",
            span_decision_prefix="refine_looping_tool_call",
        )
        if revised is None:
            # Retries exhausted. Signal the failure explicitly, then
            # return the deterministic fallback so the looping task
            # cannot re-fire on the next tick.
            #
            # Exception (goldfive#182): an empty / non-string LLM
            # response is a model-quality artefact, not a refine
            # failure — skip the escalation event. The deterministic
            # fallback still fires so the looping task can't re-arm.
            if last_error != _EMPTY_RESPONSE_ERROR:
                await self._emit_refine_validation_failed(plan, drift, last_error)
            return self._fallback_fail_loop_plan(plan, drift, looping_task)
        # goldfive#247: Plan is frozen — stamp via :func:`bump_revision`.
        return bump_revision(
            revised,
            revision_index=plan.revision_index + 1,
            revision_reason=drift.detail,
            revision_kind=drift.kind.value,
            revision_severity=str(drift.severity),
        )

    @staticmethod
    def _fallback_fail_loop_plan(
        plan: Plan,
        drift: DriftEvent,
        looping_task: Task | None,
    ) -> Plan:
        """Deterministic fallback when the LLM can't help.

        Preserves all existing tasks/edges and just stamps the looping
        task as FAILED so the executor stops retrying it. Other PENDING
        tasks proceed; the goal may be unsatisfied but the run no
        longer burns calls in the loop.
        """
        loop_id = drift.current_task_id
        new_tasks: list[Task] = []
        found = False
        for t in plan.tasks:
            if t.id == loop_id and t.status not in _TERMINAL_STATUSES:
                new_tasks.append(
                    Task(
                        id=t.id,
                        title=t.title,
                        description=t.description,
                        assignee_agent_id=t.assignee_agent_id,
                        status=TaskStatus.FAILED,
                        predicted_start_ms=t.predicted_start_ms,
                        predicted_duration_ms=t.predicted_duration_ms,
                        bound_span_id=t.bound_span_id,
                        # goldfive#251: preserve supersession provenance
                        # through the deterministic-fallback path. Without
                        # this, a CORRECT-kind chain that hits the fallback
                        # would erase the link to the prior task and the
                        # supersedes-coverage validator would orphan it.
                        supersedes=t.supersedes,
                        supersedes_kind=t.supersedes_kind,
                    )
                )
                found = True
            else:
                new_tasks.append(
                    Task(
                        id=t.id,
                        title=t.title,
                        description=t.description,
                        assignee_agent_id=t.assignee_agent_id,
                        status=t.status,
                        predicted_start_ms=t.predicted_start_ms,
                        predicted_duration_ms=t.predicted_duration_ms,
                        bound_span_id=t.bound_span_id,
                        supersedes=t.supersedes,
                        supersedes_kind=t.supersedes_kind,
                    )
                )
        if not found and looping_task is not None and loop_id:
            new_tasks.append(
                Task(
                    id=looping_task.id,
                    title=looping_task.title,
                    description=looping_task.description,
                    assignee_agent_id=looping_task.assignee_agent_id,
                    status=TaskStatus.FAILED,
                    supersedes=looping_task.supersedes,
                    supersedes_kind=looping_task.supersedes_kind,
                )
            )
        return Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=new_tasks,
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id) for e in plan.edges
            ],
            summary=plan.summary,
            revision_reason=drift.detail,
            revision_kind=drift.kind.value,
            revision_severity=str(drift.severity),
            revision_index=plan.revision_index + 1,
        )

    async def refine_steer(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        """Public entry point for a goldfive-promoted steer refine.

        Called by :meth:`DefaultSteerer._promote_drift_to_steer` when a
        goldfive-detected drift has cleared the severity threshold and
        the suppression window. Dispatches to the shared delete-and-
        replan path with ``source="goldfive"`` so the LLM prompt frames
        the refine as "goldfive detected agent drift — discard prior
        work on this task", not as "an operator typed a steer".

        The ``USER_STEER`` path continues to go through :meth:`refine`
        with ``source="user"``; the two call sites share the underlying
        implementation to guarantee identical merge / validation
        semantics.
        """
        if plan is None:
            return None
        return await self._refine_steer(plan, drift, goals, available_agents, source="goldfive")

    async def _refine_steer(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        *,
        source: str = "user",
    ) -> Plan | None:
        """Delete-and-replan path for an authoritative steer drift.

        Shared by :meth:`refine` (``USER_STEER``; ``source="user"``) and
        :meth:`refine_steer` (goldfive-promoted drift;
        ``source="goldfive"``). Completed/failed/cancelled tasks are
        preserved verbatim (same ids, titles, assignees, statuses).
        Pending/running/blocked tasks are dropped; the LLM produces a
        fresh set of PENDING tasks that honour the steer. The returned
        plan reuses ``plan.id`` and ``plan.run_id`` so lineage stays
        intact.

        The ``source`` parameter selects the prompt shape — a user
        steer reads the body as an operator directive; a goldfive
        steer reads it as a corrective drift reason ("agent drift was
        detected in the preceding activity"). Merge + validation
        behaviour is identical across both sources.
        """
        effective_source = (source or "user").strip().lower()
        if effective_source not in {"user", "goldfive"}:
            effective_source = "user"
        completed = [t for t in plan.tasks if t.status in _TERMINAL_STATUSES]
        completed_ids = {t.id for t in completed}
        prior_pending = [t for t in plan.tasks if t.status not in _TERMINAL_STATUSES]
        try:
            base_user_prompt = self._build_steer_prompt(
                completed,
                drift,
                goals,
                source=effective_source,
                prior_pending=prior_pending,
            )
        except (TypeError, ValueError) as exc:
            log.warning(
                "LLMPlanner._refine_steer: failed to serialise inputs (%s)",
                exc,
            )
            return None

        user_prompt = base_user_prompt
        attempts = max(1, self._max_refine_attempts)
        last_error = ""
        prior_error_kind: str | None = None
        for attempt in range(1, attempts + 1):
            one_attempt = (
                self._user_steer_one_attempt_override
                if self._user_steer_one_attempt_override is not None
                else self._user_steer_one_attempt
            )
            merged_plan, error = await one_attempt(
                plan=plan,
                goals=goals,
                drift=drift,
                completed=completed,
                completed_ids=completed_ids,
                user_prompt=user_prompt,
                available_agents=available_agents,
                source=effective_source,
            )
            if merged_plan is not None:
                return merged_plan
            last_error = error
            if last_error == _EMPTY_RESPONSE_ERROR:
                # Empty / non-string response is a small-model artefact
                # (goldfive#182). Retrying doubles cost without changing
                # the outcome AND emitting REFINE_VALIDATION_FAILED would
                # cascade into the intervention ladder for what is just
                # "model returned nothing useful". Log INFO and return
                # None directly — the steerer's caller treats None the
                # same way as exhausted retries (keep prior plan,
                # increment backoff) but without the escalation noise.
                log.info(
                    "LLMPlanner._refine_steer(source=%s): attempt %d/%d: %s; not retrying",
                    effective_source,
                    attempt,
                    attempts,
                    last_error,
                )
                return None
            log.warning(
                "LLMPlanner._refine_steer(source=%s): attempt %d/%d: %s",
                effective_source,
                attempt,
                attempts,
                last_error,
            )
            # iter-11C: short-circuit when consecutive attempts produce
            # the same structural-class validator rejection. Feeding the
            # LLM its own previous error a second time on the same class
            # of invariant violation is empirically unproductive on Qwen
            # 35B (live e2e: same "terminal task missing/regressed"
            # rejection on attempt 1 and 2, ~10s burned per attempt).
            # Returning None here lets the caller fall through to the
            # supersede path immediately. We still emit
            # REFINE_VALIDATION_FAILED so the steerer's escalation
            # ladder sees a uniform signal regardless of whether retries
            # were exhausted by exhaustion or by short-circuit.
            error_kind = self._extract_rejection_kind(last_error)
            if attempt > 1 and error_kind is not None and error_kind == prior_error_kind:
                log.info(
                    "LLMPlanner._refine_steer(source=%s): rejection kind "
                    "%r repeated across attempts %d-%d; short-circuiting "
                    "to supersede",
                    effective_source,
                    error_kind,
                    attempt - 1,
                    attempt,
                )
                await self._emit_refine_validation_failed(plan, drift, last_error)
                return None
            prior_error_kind = error_kind
            if attempt < attempts:
                user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
        # Retries exhausted -- signal and fall back to None (the steerer
        # keeps the prior plan and increments its backoff counter).
        await self._emit_refine_validation_failed(plan, drift, last_error)
        return None

    # Backwards-compatible alias for :meth:`_refine_steer`. Existing
    # callers / subclasses that referred to ``_refine_user_steer``
    # directly keep working; new code should use ``_refine_steer`` with
    # an explicit ``source`` or go through :meth:`refine_steer`.
    async def _refine_user_steer(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None:
        return await self._refine_steer(plan, drift, goals, available_agents, source="user")

    async def _user_steer_one_attempt(
        self,
        *,
        plan: Plan,
        goals: list[Goal],
        drift: DriftEvent,
        completed: list[Task],
        completed_ids: set[str],
        user_prompt: str,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        source: str = "user",
    ) -> tuple[Plan | None, str]:
        """Run a single LLM attempt for the USER_STEER refine path.

        Returns ``(merged_plan, "")`` on success or ``(None, error)`` on
        any parse / merge / validate failure. The caller drives the
        retry loop; this helper keeps the merge logic in one place so
        both the first attempt and every retry apply exactly the same
        plumbing.
        """
        from goldfive._llm_span import goldfive_llm_span

        # ``source`` is "user" for USER_STEER and "goldfive" for a
        # goldfive-promoted drift — surface that on the span name so
        # operators can distinguish them on the Gantt.
        span_name = "refine_user_steer" if source != "goldfive" else "refine_steer"
        refine_input_preview = self._build_refine_span_input_preview(drift, plan)
        try:
            async with goldfive_llm_span(
                **self._span_kwargs(task_id_override=drift.current_task_id),
                name=span_name,
                input_preview=refine_input_preview,
                target_agent_id=drift.current_agent_id or "",
                target_task_id=drift.current_task_id or "",
            ) as span:
                # Bound the dispatch — see ``LLMPlanner.MAX_OUTPUT_TOKENS``.
                # Disable thinking — user_steer is a structured JSON
                # revision call, not deep reasoning.
                from goldfive._llm import (
                    call_llm_budget,
                    call_llm_thinking_disabled,
                )

                with (
                    call_llm_budget(self.MAX_OUTPUT_TOKENS),
                    call_llm_thinking_disabled(),
                ):
                    raw = await self._call_llm(
                        self._user_steer_system_prompt, user_prompt, self._model
                    )
                span.output_preview = raw[:4096] if isinstance(raw, str) else "(non-str response)"
                decision_prefix = (
                    "refined plan (goldfive steer) in response to"
                    if source == "goldfive"
                    else "refined plan (user steer) in response to"
                )
                span.decision_summary = (
                    f"{decision_prefix} "
                    f"{drift.kind.value} on "
                    f"{drift.current_task_id or '(trajectory)'}"
                )
        except Exception as exc:  # noqa: BLE001
            return None, f"call_llm raised: {exc}"
        if not raw or not isinstance(raw, str):
            return None, _EMPTY_RESPONSE_ERROR
        cleaned = _strip_code_fences(raw).strip()
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as exc:
            return None, f"JSON parse failed: {exc}"
        fresh = _plan_from_json(
            parsed,
            run_id=plan.run_id,
            goal_ids=[g.id for g in goals if g.id] or list(plan.goal_ids),
            plan_id=plan.id,
        )
        if fresh is None:
            return None, "parsed JSON did not contain a usable plan"

        # goldfive#213: backfill ``Task.supersedes`` for retry-named
        # tasks (``retry_t0``, ``t0_v2``) when the LLM forgot the
        # structural link. Pure structural inference over goldfive's
        # own id conventions — no LLM-trust, no prompt contract.
        # Runs against ``fresh`` (not the post-merge plan) so the
        # predecessor candidates are resolved against the FULL prior
        # plan (including completed tasks that were dropped from
        # ``fresh``). This makes the executor's causal-tier
        # replacement detection work even for plans where the LLM
        # didn't populate ``supersedes``. goldfive#247: returns a NEW
        # Plan; rebind so downstream merge sees the backfilled tasks.
        fresh = _backfill_retry_supersedes(fresh, prior=plan)

        # Prepend completed tasks verbatim; drop any new task whose id
        # collides with a completed id so lineage ids stay stable.
        new_pending = [t for t in fresh.tasks if t.id not in completed_ids]
        merged_tasks = list(completed) + new_pending
        # Edges: keep original edges that are still fully satisfiable
        # (both endpoints either completed or new-pending), plus every
        # new edge from the LLM.
        known_ids = completed_ids | {t.id for t in new_pending}
        preserved_edges = [
            TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
            for e in plan.edges
            if e.from_task_id in known_ids and e.to_task_id in known_ids
        ]
        new_edges = [
            TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id) for e in fresh.edges
        ]
        # Deduplicate (same from/to pair) while preserving order.
        seen: set[tuple[str, str]] = set()
        merged_edges: list[TaskEdge] = []
        for e in preserved_edges + new_edges:
            key = (e.from_task_id, e.to_task_id)
            if key in seen:
                continue
            seen.add(key)
            merged_edges.append(e)

        if (source or "user").strip().lower() == "goldfive":
            revision_reason = f"goldfive steer ({drift.kind.value}): {drift.detail}"
            # Preserve the underlying drift kind on the revision so sinks
            # see the ladder-promoted source, not a synthesised USER_STEER.
            revision_kind_value = drift.kind.value
        else:
            revision_reason = f"user steering: {drift.detail}"
            revision_kind_value = DriftKind.USER_STEER.value
        merged_plan = Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(fresh.goal_ids) or list(plan.goal_ids),
            tasks=merged_tasks,
            edges=merged_edges,
            summary=fresh.summary or plan.summary,
            revision_reason=revision_reason,
            revision_kind=revision_kind_value,
            revision_severity=DriftSeverity.WARNING.value,
            revision_index=plan.revision_index + 1,
        )
        # goldfive#251 Option B: coerce supersedes_kind on every merged
        # task based on the OLD-task status in the prior plan. The
        # evolution path (id reuse without supersedes) is unaffected;
        # the supersede-with-fresh-id path gets parity with the
        # refine_user path — an LLM that sets ``supersedes_kind=UNSPECIFIED``
        # against a prior PENDING is coerced to ``REPLACE`` so the
        # executor's pin-redirect runs. Runs AFTER the goldfive#213
        # backfill (above) so the kind is derived against any
        # freshly-populated retry link. goldfive#247: returns a NEW
        # Plan; rebind so the validator below sees the coerced kinds.
        merged_plan = _normalize_supersession_kinds(merged_plan, prior=plan)
        try:
            merged_plan.validate(for_revision=True, prior=plan)
        except ValueError as exc:
            return None, f"validator rejected revision: {exc}"
        # Preserve earlier USER_STEER goals through successive steers
        # (goldfive#154). The current steer's note is handed in via
        # ``drift.detail``; any *prior* USER_STEER goals still on
        # ``session.goals`` must still be addressed by the new PENDING
        # tasks, otherwise a later steer would silently unwind an
        # earlier one.
        goal_error = self._check_user_steer_goals_preserved(merged_plan, goals)
        if goal_error:
            return None, goal_error
        assignee_error = self._validate_plan_assignees(merged_plan, available_agents)
        if assignee_error:
            return None, assignee_error
        return merged_plan, ""

    # ------------------------------------------------------------------
    # Per-turn decision (goldfive#271 Phase 4)
    # ------------------------------------------------------------------

    #: System prompt for :meth:`handle_turn`. Asks the planner LLM to
    #: produce the next plan if the user's input warrants a plan
    #: change (new task, refined constraints, topic shift, scope
    #: change), or to return null if the input is purely conversational
    #: (a question about prior work, a clarification, an
    #: acknowledgment).
    #:
    #: This single prompt replaced the prior triage stack:
    #:
    #: * the regex-based factual-question short-circuit
    #:   (``planner_gate._FACTUAL_QUESTION_RE``)
    #: * the regex-based steer-language short-circuit
    #:   (``planner_gate._STEER_PATTERN_RE``)
    #: * the gate's own LLM classifier
    #:   (``planner_gate.classify_turn``)
    #: * the qualification-merge regex post-process
    #:   (``steerer._GENERIC_VERB_PREFIX_RE`` /
    #:   ``_rewrite_output_type_prefix`` /
    #:   ``_merge_prior_qualifications_into_goal``)
    #: * the separate ``synthesize_goal_from_steer`` LLM call
    #:
    #: All of the above were collapsed into this single prompt so the
    #: LLM does the routing AND the qualification merge in one shot
    #: rather than fighting with brittle regexes for each NL shape.
    #:
    #: The "classification" is now an emergent property of "did the
    #: LLM produce a plan or not" rather than a synthetic categorical
    #: label the LLM has to be taught.
    _HANDLE_TURN_SYSTEM_PROMPT: str = """\
You are the planner for a multi-agent orchestration system. The user
sent a NEW MESSAGE on a conversation that already has a PRIOR PLAN
(possibly partially executed). Decide whether the new message warrants
a plan change. If so, produce the next plan in the same response. If
not, return null for the plan.

Reply with a single JSON object and NOTHING ELSE:
{
  "reasoning": "<one-sentence why>",
  "replaces_prior": <bool — see PIVOT vs REVISION below>,
  "plan": null OR {
    "id": "<short-id>",
    "summary": "<noun phrase describing the GOAL — see SUMMARY POLICY in PLAN SHAPE>",
    "tasks": [
      {
        "id": "<short-id>",
        "title": "<one-sentence what 'done' looks like>",
        "description": "<optional longer description>",
        "status": "PENDING"
      }
    ],
    "edges": [{"from_task_id": "...", "to_task_id": "..."}]
  }
}

PIVOT vs REVISION (the ``replaces_prior`` field):

Set ``replaces_prior: true`` when the user is REPLACING prior intent
rather than building on it — a topic pivot, an artefact abandonment,
or any "forget X, do Y instead"-shaped instruction. A pivot is a
fresh start: the runner mints a new plan id, terminal-task / edge
preservation does NOT apply, and the new plan does not need to echo
back COMPLETED tasks from the prior. Examples:

  * "forget the slides, just give me bullet points" — different artefact
  * "scratch that, do something completely different — translate this poem"
  * "no, don't do any of that. Tell me about quantum mechanics instead."
  * "stop, I want to switch topics entirely to dark matter"

Set ``replaces_prior: false`` (the default) when the user is BUILDING
on the prior plan — additive constraints, partial corrections,
qualification tweaks, even topic-shifts that keep the same artefact
shape. The runner installs the result as a revision of the prior
plan id. Terminal tasks must be preserved verbatim in the new plan.
Examples:

  * "make it 2 slides instead of 5" — additive constraint
  * "actually, make it about solar flares not solar panels" — topic
    shift on the same artefact (still a presentation)
  * "redo task 3 with the new data" — partial correction

When in doubt, prefer ``false`` — a revision is the safer default;
the structural-preservation overhead is small.

WHEN TO RETURN A PLAN (plan is non-null):

Produce the next plan if the user's input warrants a plan change —
new task, refined constraints, topic shift, scope change, additive
constraint, tone / style tweak, partial correction, or any directive
that the existing plan does not already cover. Examples:

  * "forget X, tell me about Y instead" — topic shift
  * "make a 5-page report on dark matter" (when prior plan was about
    solar panels) — scope / topic change
  * "make sure the answer fits in 2 slides" — additive constraint
  * "make it funnier" — tone tweak
  * "also translate to Spanish" — additive scope
  * "redo task 3 with the new data" — partial correction

When you produce a plan:

  * REUSE the prior plan's id (use the same id string verbatim) when
    the new plan is a revision of the prior — almost always the case
    when the user is iterating on the same artefact even with a topic
    change. Mint a fresh id ONLY when the user explicitly abandons
    the artefact entirely (e.g. "forget the slides, just give me
    bullet points" — different artefact).
  * KEEP terminal tasks (COMPLETED / FAILED / CANCELLED / NOT_NEEDED)
    verbatim — same id, same status — in the ``tasks`` array of the
    revision. The validator REJECTS any revision that drops a terminal
    task or regresses its status (e.g. COMPLETED → PENDING). Add new
    tasks for the delta work; reuse ids for tasks that represent the
    same work. New PENDING tasks may chain off a COMPLETED predecessor
    (the natural "done stage feeds into next stage" shape) but must
    NEVER chain off a CANCELLED, FAILED, or NOT_NEEDED predecessor —
    such edges are rejected because the downstream task can never
    become eligible.
  * MERGE persistent qualifications from prior_goals into the new
    plan AND into the ``summary`` string (numeric caps like "no more
    than 2 slides", format requirements like "in markdown", output
    type like "presentation" / "report", scope qualifiers like "for a
    non-technical audience"). The topic / subject changes; the
    structural frame carries forward. Drop a qualification ONLY when
    the user explicitly removes it ("forget the slide-count cap", "no
    longer needs to be a presentation").

    QUALIFICATION-PRESERVATION EXAMPLE (failing case to avoid):
      prior_goals: "Create a presentation about solar panels with no
                    more than 2 slides."
      new message: "Forget solar panels, tell me about solar flares
                    instead."
      WRONG plan.summary: "Research solar flares and create
                           presentation on solar flares."
        ↑ drops the "2 slides" qualification — REJECT this shape.
      RIGHT plan.summary: "Create a 2-slide presentation about solar
                           flares."
        ↑ subject pivots (panels → flares); the "2-slide" cap and the
        "presentation" output type carry forward verbatim.

WORKED EXAMPLES OF VALID REVISIONS:

Example A — additive constraint on a partially-executed plan.
The validator REJECTS revisions that drop terminal tasks. Echo
COMPLETED tasks back in ``tasks`` with the SAME id and SAME status:

  PRIOR PLAN (id=p1, revision_index=2):
    - [research_solar / COMPLETED]   Research solar panels
    - [draft_slides  / RUNNING]      Draft the slides
    - [review        / PENDING]      Review presentation
  NEW MESSAGE: "Make sure the answer fits in just 2 slides."
  VALID revision (same id, terminal task preserved verbatim,
  delta is the new constraint encoded into existing PENDING titles
  and a fresh enforcement task):
    {
      "id": "p1",
      "summary": "Draft and review a 2-slide presentation about solar panels.",
      "tasks": [
        {"id": "research_solar", "title": "Research solar panels",
         "status": "COMPLETED"},
        {"id": "draft_slides",   "title": "Draft EXACTLY 2 slides",
         "status": "PENDING"},
        {"id": "review",         "title": "Review the 2-slide presentation",
         "status": "PENDING"}
      ],
      "edges": [
        {"from_task_id": "research_solar", "to_task_id": "draft_slides"},
        {"from_task_id": "draft_slides",   "to_task_id": "review"}
      ]
    }

Example B — topic shift on a partially-executed plan.
The same id is reused (still iterating on the same artefact). The
COMPLETED task is preserved verbatim — it stays in ``tasks`` even
though its subject is now stale, because the validator forbids
dropping it. The new PENDING work for the new subject forms its own
sub-DAG (no edge from the now-stale COMPLETED task into the new
PENDING root, since that would imply the new work depends on stale
output):

  PRIOR PLAN (id=p1, revision_index=1):
    - [research_solar / COMPLETED]   Research solar panels
    - [draft_slides  / PENDING]      Draft 2 slides on solar panels
    - [review        / PENDING]      Review presentation
  PRIOR GOAL: "Create a 2-slide presentation about solar panels."
  NEW MESSAGE: "Forget solar panels, tell me about solar flares
                instead."
  VALID revision (qualifications "2-slide" + "presentation" carry
  forward; COMPLETED task preserved; new PENDING tasks form an
  independent sub-DAG):
    {
      "id": "p1",
      "summary": "Create a 2-slide presentation about solar flares.",
      "tasks": [
        {"id": "research_solar",   "title": "Research solar panels",
         "status": "COMPLETED"},
        {"id": "research_flares",  "title": "Research solar flares",
         "status": "PENDING"},
        {"id": "draft_flares",     "title": "Draft 2-slide presentation about solar flares",
         "status": "PENDING"},
        {"id": "review_flares",    "title": "Review the 2-slide solar flares presentation",
         "status": "PENDING"}
      ],
      "edges": [
        {"from_task_id": "research_flares", "to_task_id": "draft_flares"},
        {"from_task_id": "draft_flares",    "to_task_id": "review_flares"}
      ]
    }

WHEN TO RETURN NULL (plan is null):

Return null when the user's input is purely conversational and the
prior plan should be reused unchanged. Examples:

  * "where will the slides be saved?" — factual question about prior
  * "what was on slide 2?" — clarification
  * "did you include source X?" — verification
  * "is it ready?" / "is the presentation done?" — status question
  * "summarise what you did" — recap request
  * "thanks" / "ok" — acknowledgment

Factual interrogatives (where/when/how/what/why/which/who +
is/are/will/did/does/was/were) about prior work are conversational by
default. Tense doesn't matter — "where will the file go?" is just as
much a question as "where did the file go?"

GUIDELINES:

  * When in doubt between null and a plan, prefer null — the
    coordinator can always answer; the user can restate if they
    actually wanted new work.
  * Steer-language openers ("forget", "instead", "no, don't ...",
    "scratch that", "actually", "wait, ...", "stop", "change the
    topic", "switch to") are strong plan-change signals — return a
    plan that revises the prior.

PLAN SHAPE (when plan is non-null):

  * 5-20 tasks typically. Smaller is OK for trivial follow-ups
    (revisions often add 1-3 delta tasks).
  * Do NOT populate `assignee_agent_id`; leave it as the empty string.
    The framework populates it observationally when a delegation
    actually happens (goldfive#252). The available_agents block in
    the user prompt is supplied for context only.
  * Task ids: short, unique, stable strings ("research", "draft_intro",
    "review_final"). Reuse prior task ids when the task is the same
    work; mint new ids for delta tasks.

SUMMARY POLICY (applies to ``plan.summary``):

  * MUST be a one-sentence PR-title-shaped noun phrase describing the
    GOAL the plan delivers.
  * DO NOT include process commentary, meta-reasoning, or sentences
    explaining why you made (or didn't make) changes. DO NOT mention
    'drift', 'revision', or 'plan unchanged'.
  * If the goal hasn't shifted relative to the prior plan, reuse the
    prior plan's summary verbatim.
  * RIGHT: "Create a 2-slide presentation about solar panels."
  * RIGHT: "Generate a Python script that prints fibonacci numbers up to 100."
  * WRONG: "Plan unchanged because no specific details were provided."
"""

    async def handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        conversation_history: list[Any] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        """Single per-turn decision point (goldfive#271 Phase 4).

        Returns ``None`` when the user_input is purely conversational
        and the current revision still describes the right work.
        Returns the next :class:`Plan` revision when a plan change is
        warranted (new task, refined constraints, topic shift, scope
        change, or dropping tasks no longer relevant). The Runner
        installs the returned plan as the next revision of
        ``session.plan`` (revision_index += 1) via
        ``DefaultSteerer.install_initial_plan`` on turn 1 or
        ``DefaultSteerer.install_revision_for_drift`` on subsequent
        turns (goldfive#271 Option A).

        The prior plan + goals are read off ``session.plan`` /
        ``session.goals`` — the planner doesn't need them as separate
        kwargs. The Runner guarantees ``session.plan`` is non-None on
        every turn, including the very first (it seeds
        :meth:`Plan.empty` so the planner produces revision 1 against
        an empty prior).

        Replaces the prior multi-stage pipeline:

        1. ``planner_gate`` regex short-circuits (factual-question +
           steer-language detection)
        2. ``planner_gate.classify_turn`` LLM gate
        3. ``synthesize_goal_from_steer`` LLM call
        4. regex-based qualification merge post-process
        5. ``planner.refine`` (or ``planner.generate`` on first turn)
           LLM call

        — all collapsed into one LLM call that produces both the
        decision AND the next plan. The "classification" is now an
        emergent property of "did the LLM produce a plan or not"
        rather than a synthetic categorical label the LLM has to be
        taught.

        Any LLM / parse failure logs at WARNING and returns ``None``
        — the gate must never break the run on a misbehaving LLM.
        On a None return, the Runner reuses ``session.plan`` for this
        turn (which is the empty seed on first turn → no work done
        until a subsequent turn produces a real plan, or the
        coordinator answers from history).
        """
        text = (user_input or "").strip()
        if not text:
            return None
        prior_plan = session.plan
        prior_goals = list(session.goals)
        base_user_prompt = self._build_handle_turn_prompt(
            user_input=text,
            prior_plan=prior_plan,
            prior_goals=prior_goals,
            conversation_history=conversation_history or [],
            available_agents=available_agents,
        )
        from goldfive._llm_span import goldfive_llm_span

        # ``handle_turn`` is trajectory-level (decides whether the turn
        # goes through planning at all), so ``target_agent_id`` /
        # ``target_task_id`` stay empty. A compact rendering of the
        # user input + prior plan summary doubles as ``input_preview``.
        prior_id = ((prior_plan.id or "") if prior_plan is not None else "")[:16] or "<none>"
        gate_input_preview = f"user_input: {text}\nprior_plan_id: {prior_id}"

        # F7 (closes part of goldfive#322): validator-feedback retry.
        # Mirrors the retry pattern in ``_call_and_validate_refine``
        # (planner.py:1870+) but inlined for handle_turn so a one-off
        # validation failure on the first LLM attempt — almost always
        # a Rule 6 terminal-task / terminal->terminal-edge regression
        # — gets a second chance with an explicit error message
        # appended to the prompt. Capped at 2 attempts (one retry)
        # so a misbehaving LLM doesn't burn the per-turn budget.
        user_prompt = base_user_prompt
        plan: Plan | None = None
        last_error = ""
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                async with goldfive_llm_span(
                    **self._span_kwargs(),
                    name="planner_handle_turn",
                    input_preview=gate_input_preview,
                ) as span:
                    # Bound the dispatch — see ``LLMPlanner.MAX_OUTPUT_TOKENS``.
                    # Disable thinking — handle_turn is a small JSON gate
                    # call ("does this user input need a re-plan?"), not
                    # deep reasoning.
                    from goldfive._llm import (
                        call_llm_budget,
                        call_llm_thinking_disabled,
                    )

                    with (
                        call_llm_budget(self.MAX_OUTPUT_TOKENS),
                        call_llm_thinking_disabled(),
                    ):
                        raw = await self._call_llm(
                            self._HANDLE_TURN_SYSTEM_PROMPT,
                            user_prompt,
                            self._model,
                        )
                    span.output_preview = (
                        raw[:4096] if isinstance(raw, str) else "(non-str response)"
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "LLMPlanner.handle_turn: attempt %d/%d: call_llm raised %s; "
                    "treating as conversational (no plan change)",
                    attempt,
                    max_attempts,
                    exc,
                )
                return None
            candidate = self._parse_handle_turn_response(
                raw=raw,
                prior_plan=prior_plan,
                context=context,
            )
            if candidate is None:
                # Conversational verdict OR unparseable response. Either
                # way, no plan to validate. Return None — the runner
                # treats this as "reuse prior plan".
                log.info(
                    "LLMPlanner.handle_turn: attempt %d/%d: produced_plan=no "
                    "prior_plan_id=%s",
                    attempt,
                    max_attempts,
                    prior_id,
                )
                return None
            # Validate now (mirrors what the steerer's install path
            # would otherwise check) so a Rule-6 regression triggers
            # the retry rather than the runner aborting the turn.
            # Pivot installs go through ``install_initial_plan`` which
            # validates against the empty seed — skip the prior here.
            is_pivot = bool(getattr(candidate, "_goldfive_pivot", False))
            try:
                if is_pivot or prior_plan is None or not prior_plan.tasks:
                    candidate.validate(for_revision=True, prior=None)
                else:
                    candidate.validate(for_revision=True, prior=prior_plan)
            except ValueError as exc:
                last_error = str(exc)
                log.warning(
                    "LLMPlanner.handle_turn: attempt %d/%d: validator rejected "
                    "produced plan: %s",
                    attempt,
                    max_attempts,
                    last_error,
                )
                if attempt < max_attempts:
                    user_prompt = self._build_correction_prompt(
                        base_user_prompt, last_error
                    )
                    continue
                # Final attempt failed validation. Return the candidate
                # anyway — the runner's install path will surface the
                # validation error via SCHEMA_VIOLATION drift, matching
                # the prior behaviour for callers that disable retry.
                plan = candidate
                break
            plan = candidate
            break
        log.info(
            "LLMPlanner.handle_turn: produced_plan=%s prior_plan_id=%s%s",
            "yes" if plan is not None else "no",
            prior_id,
            f" (after {attempt} attempt(s))" if attempt > 1 else "",
        )
        return plan

    def _build_handle_turn_prompt(
        self,
        *,
        user_input: str,
        prior_plan: Plan | None,
        prior_goals: list[Goal],
        conversation_history: list[Any],
        available_agents: list[str] | list[dict[str, Any]] | None,
    ) -> str:
        """Render the user prompt for :meth:`handle_turn`.

        Includes:
        * NEW MESSAGE — the user's free-form input.
        * PRIOR PLAN — id, summary, and per-task ``[id / status] title``.
          Empty seeds (``Plan.empty()``) render as "PRIOR PLAN: empty"
          so the LLM sees the first-turn case explicitly.
        * PRIOR GOALS — verbatim summaries for the qualification-merge.
        * AVAILABLE AGENTS — the registry the LLM must pick from.
        * CONVERSATION HISTORY — capped to recent turns to bound prompt
          length. Each entry: ``[turn N] <user_input_summary>``.
        """
        chunks: list[str] = []
        chunks.append(f"NEW MESSAGE FROM USER:\n{user_input}")
        chunks.append(self._render_prior_plan_block(prior_plan))
        if prior_goals:
            goal_lines = [
                f"- [{g.id or '(no-id)'}] {g.summary or '(no summary)'}" for g in prior_goals
            ]
            chunks.append(
                "PRIOR GOALS (preserve their persistent qualifications "
                "unless the user explicitly removes them):\n" + "\n".join(goal_lines)
            )
        if conversation_history:
            # Cap to the most recent few entries — older context lives
            # on the prior plan / goals already.
            recent = conversation_history[-3:]
            hist_lines = []
            for i, t in enumerate(recent, start=max(1, len(conversation_history) - 2)):
                summary = getattr(t, "user_input_summary", "") or ""
                hist_lines.append(f"  [turn {i}] {summary}")
            if hist_lines:
                chunks.append("RECENT CONVERSATION HISTORY:\n" + "\n".join(hist_lines))
        agents_block = self._render_agents_block(available_agents)
        if agents_block:
            chunks.append(agents_block)
        chunks.append(
            'Decide and respond. Reply JSON only: {"reasoning": "...", "plan": ... | null}'
        )
        return "\n\n".join(chunks)

    @staticmethod
    def _render_prior_plan_block(plan: Plan | None) -> str:
        """One-shot rendering of the prior plan for the handle_turn prompt.

        Empty plans (``Plan.empty()`` seed used on first turn) render
        as "PRIOR PLAN: empty (this is the first turn)" so the LLM
        knows to produce the initial plan rather than treating the
        empty plan as something to revise minimally.
        """
        if plan is None or not plan.tasks:
            return (
                "PRIOR PLAN: empty (this is the first turn — produce "
                "revision 1 from the user's request)."
            )
        lines: list[str] = []
        lines.append(f"PRIOR PLAN (id={plan.id or '(no-id)'}):")
        if plan.summary:
            lines.append(f"  Summary: {plan.summary}")
        lines.append("  Tasks:")
        for t in plan.tasks:
            tid = t.id or "(no-id)"
            status = getattr(t.status, "value", str(t.status))
            title = t.title or t.description or ""
            lines.append(f"    - [{tid} / {status}] {title}")
        return "\n".join(lines)

    def _parse_handle_turn_response(
        self,
        *,
        raw: Any,
        prior_plan: Plan | None,
        context: Mapping[str, Any] | None,
    ) -> Plan | None:
        """Parse the LLM's JSON response into a :class:`Plan` or ``None``.

        On any parse failure, returns ``None`` (treated as
        conversational by the Runner). The contract: this method MUST
        NOT raise.

        When the LLM produces a plan AND ``replaces_prior`` is False
        (or absent — the safe default), install the prior plan's id
        verbatim — the steerer's ``_apply_revision`` then bumps
        ``revision_index`` cleanly. Validation v3 confirmed
        refine/replace collapse for that path.

        When ``replaces_prior`` is True (F5 — pivot routing,
        goldfive#322 Layer 2 / #204), the LLM has signalled an
        artefact-replacement intent. Mint a FRESH plan id (drop the
        prior's id) and stash the pivot flag on the Plan via the
        sentinel attribute :attr:`Plan._goldfive_pivot` so the
        Runner's ``_install_revision`` routes through
        ``install_initial_plan`` rather than
        ``install_revision_for_drift``. The fresh id +
        non-revision install path mean Rule 6
        (terminal-task / terminal->terminal-edge preservation in
        :meth:`Plan.validate`) does not gate the new plan against
        the prior — a pivot is structurally a fresh start.
        """
        if not isinstance(raw, str) or not raw.strip():
            log.warning("LLMPlanner.handle_turn: empty / non-str response")
            return None
        cleaned = _strip_code_fences(raw).strip()
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as exc:
            log.warning("LLMPlanner.handle_turn: JSON parse failed: %s", exc)
            return None
        if not isinstance(parsed, dict):
            log.warning(
                "LLMPlanner.handle_turn: response was not an object; got %r",
                type(parsed),
            )
            return None
        plan_raw = parsed.get("plan")
        if plan_raw is None:
            # Conversational verdict — the LLM emitted no plan.
            return None
        if not isinstance(plan_raw, dict):
            log.warning(
                "LLMPlanner.handle_turn: 'plan' present but not an object; "
                "got %r — treating as conversational",
                type(plan_raw),
            )
            return None
        run_id = ""
        if context is not None:
            run_id = str(context.get("run_id") or "")
        # F5: pivot detection. ``replaces_prior=true`` → mint a fresh
        # plan_id and signal the runner to route through
        # ``install_initial_plan``. False/absent (the safe default) →
        # reuse the prior id so revision_index bumps cleanly.
        replaces_prior = bool(parsed.get("replaces_prior"))
        if replaces_prior:
            plan_id_override = None  # _plan_from_json mints a fresh uuid
        else:
            plan_id_override = (
                prior_plan.id if prior_plan is not None and prior_plan.id else None
            )
        prior_goal_ids = list(prior_plan.goal_ids) if prior_plan is not None else []
        plan = _plan_from_json(
            plan_raw,
            run_id=run_id,
            goal_ids=prior_goal_ids,
            plan_id=plan_id_override,
        )
        if plan is None:
            log.warning(
                "LLMPlanner.handle_turn: 'plan' failed structural parse; treating as conversational"
            )
            return None
        # F5: stamp the pivot flag onto the Plan so the runner's
        # ``_install_revision`` can route to ``install_initial_plan``.
        # Plain attribute on the dataclass instance (Plan is mutable);
        # the runner reads via ``getattr(plan, "_goldfive_pivot", False)``.
        # goldfive#247: ``_goldfive_pivot`` is a declared field on the
        # frozen ``Plan`` dataclass; we derive a new instance with the
        # flag set rather than dynamically adding the attribute.
        # so older Plans (no flag) trivially route as revisions.
        if replaces_prior:
            plan = dataclasses.replace(plan, _goldfive_pivot=True)
        return plan


__all__ = [
    "LLMPlanner",
    "PassthroughPlanner",
    "StaticPlanner",
]
