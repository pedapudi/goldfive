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

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)

# Task statuses that are "done" from a steering perspective — they are
# preserved verbatim across a USER_STEER delete-and-replan.
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

log = logging.getLogger("goldfive.planner")


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

4. ASSIGNMENTS. Every task's `assignee_agent_id` MUST be drawn from
   the provided available_agents list. If multiple agents could do a
   task, pick the most specialised one. If no agent fits, assign it
   to the coordinator/root agent rather than inventing an id.

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
      "description": "one sentence defining 'done' for this task",
      "assignee_agent_id": "<agent id from available list>"
    }
  ],
  "edges": [
    {"from_task_id": "research", "to_task_id": "draft"}
  ]
}
"""


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
   precede it with a clarifying step.
4. PRESERVE OR REWORK other non-finished tasks at your discretion.
5. GOAL COVERAGE: every unsatisfied goal must still be addressed by at
   least one task in the returned plan.

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "assignee_agent_id": "...",
      "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|BLOCKED"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}
"""


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

Your job is to produce the REMAINING work (fresh PENDING tasks) that
satisfies the goals in light of the steering note. Existing pending
tasks should be treated as DELETED — the operator's steer overrides
whatever was pending. Design the new tasks from the ground up.

Requirements:

1. DO NOT repeat the completed tasks in your response. The caller
   will prepend them to your task list. Respond only with the NEW
   PENDING tasks and their edges.
2. Any edge whose ``from_task_id`` is one of the completed tasks is
   allowed — the caller will preserve those and wire them through.
3. Stable ids: new task ids must be short, unique, and must not
   collide with any completed-task id.
4. GOAL COVERAGE: every unsatisfied goal must still be addressed.
5. Honour the steering note: if it says "skip review", don't add a
   review task; if it says "focus on X", center the remaining work
   on X.

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "assignee_agent_id": "..."
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}
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

4. DROP OBSOLETE PENDING TASKS. If the drift makes a PENDING task
   unnecessary (e.g., a goal has been satisfied early, a dependency
   collapsed), you may omit it from the returned plan. Never drop
   tasks that already ran.

5. REASSIGN. If a task is better handled by a different available
   agent in light of the drift, update its `assignee_agent_id`.

6. KEEP IDS STABLE. When the underlying work is the same, keep the
   task id unchanged. Reuse ids only for the same logical task.

7. STILL COVER THE GOALS. The revised plan must still have at least
   one task addressing each goal that is not yet satisfied.

8. RETURN A COMPLETE PLAN. Always return the full plan, not a delta.

Respond with a single JSON object and NOTHING ELSE:

{
  "summary": "...",
  "tasks": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "assignee_agent_id": "...",
      "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|BLOCKED"
    }
  ],
  "edges": [{"from_task_id": "...", "to_task_id": "..."}]
}

If nothing needs to change, return the current plan unchanged (but
still as a complete JSON plan, not an empty object).
"""


# ---------------------------------------------------------------------------
# JSON parsing helpers (ported from harmonograf_client.planner)
# ---------------------------------------------------------------------------

_VALID_TASK_STATUSES = frozenset(s.value for s in TaskStatus)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(raw: str) -> str:
    """Remove a surrounding ```json ... ``` or ``` ... ``` fence, if any."""
    if not raw:
        return raw
    match = _FENCE_RE.match(raw)
    if match:
        return match.group(1)
    return raw


def _coerce_status(raw: Any) -> TaskStatus:
    text = str(raw or "PENDING").upper()
    if text not in _VALID_TASK_STATUSES:
        return TaskStatus.PENDING
    return TaskStatus(text)


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
        tasks.append(
            Task(
                id=tid,
                title=title,
                description=str(t.get("description") or ""),
                assignee_agent_id=str(t.get("assignee_agent_id") or ""),
                status=_coerce_status(t.get("status")),
                predicted_start_ms=int(t.get("predicted_start_ms") or 0),
                predicted_duration_ms=int(t.get("predicted_duration_ms") or 0),
                bound_span_id=str(t.get("bound_span_id") or ""),
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
    """No-op planner — ``generate`` and ``refine`` always return ``None``.

    Makes it safe to wire a ``planner=`` kwarg everywhere without
    forcing callers to opt in to planning on day one.
    """

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
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
        available_agents: list[str],
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
    ) -> Plan | None:
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

    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        system_prompt: str | None = None,
        refine_system_prompt: str | None = None,
        user_steer_system_prompt: str | None = None,
        looping_tool_call_system_prompt: str | None = None,
        max_refine_attempts: int | None = None,
    ) -> None:
        self._call_llm = call_llm
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._refine_system_prompt = refine_system_prompt or _REFINE_SYSTEM_PROMPT
        self._user_steer_system_prompt = user_steer_system_prompt or _USER_STEER_SYSTEM_PROMPT
        self._looping_tool_call_system_prompt = (
            looping_tool_call_system_prompt or _LOOPING_TOOL_CALL_SYSTEM_PROMPT
        )
        self._max_refine_attempts = (
            int(max_refine_attempts)
            if max_refine_attempts is not None
            else self.DEFAULT_MAX_REFINE_ATTEMPTS
        )
        # Optional sink for ``REFINE_VALIDATION_FAILED`` drifts. The
        # ``DefaultSteerer`` wires this in ``bind()`` so the planner can
        # surface a structured signal when its retry budget is spent
        # without the planner needing a dependency on the sink pipeline.
        # When left as ``None`` (e.g. the planner is used standalone in
        # tests) the fallback still happens -- it is just not emitted.
        self._drift_emitter: Callable[[DriftEvent], Awaitable[None]] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_refine_attempts(self) -> int:
        return self._max_refine_attempts

    def set_drift_emitter(self, emitter: Callable[[DriftEvent], Awaitable[None]] | None) -> None:
        """Install (or remove) the async callback used to signal refine
        failures.

        The steerer wires this up in ``bind()`` so that when the planner
        exhausts its refine retry budget it can emit a
        ``REFINE_VALIDATION_FAILED`` ``DriftEvent`` through the normal
        event pipeline without the planner owning a sink list itself.
        """
        self._drift_emitter = emitter

    # ---- prompt builders -------------------------------------------------

    @staticmethod
    def _render_goals_block(goals: list[Goal]) -> str:
        if not goals:
            return "- (no goals provided)"
        lines: list[str] = []
        for g in goals:
            gid = g.id or "(no-id)"
            summary = g.summary or "(no summary)"
            lines.append(f"- [{gid}] {summary}")
        return "\n".join(lines)

    @staticmethod
    def _render_agents_block(available_agents: list[str]) -> str:
        return "\n".join(f"- {a}" for a in available_agents) or "- (none listed)"

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
        available_agents: list[str],
        context: Mapping[str, Any] | None,
    ) -> str:
        goals_block = self._render_goals_block(goals)
        agents_block = self._render_agents_block(available_agents)
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
            f"Available agents:\n{agents_block}\n\n"
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
        """Build the delete-and-replan user prompt for a USER_STEER drift.

        Completed tasks are shown as read-only context; the LLM is told
        to produce only the remaining pending work in light of the
        steering note. The caller prepends the completed tasks back onto
        the returned plan so lineage is preserved verbatim. The new
        PENDING task ids must not collide with the history ids; this is
        stated as an explicit invariant (goldfive#133).
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
        goals_block = self._render_goals_block(goals)
        note = drift.detail or "(no steering note provided)"
        invariants = (
            "STRUCTURAL INVARIANTS (the validator will REJECT any response "
            "that breaks these rules):\n"
            "1. The new PENDING task ids you emit MUST NOT collide with "
            f"any history id. Reserved ids: {history_ids}\n"
            "2. Every edge's `from_task_id` and `to_task_id` must "
            "reference either a reserved history id or a new-task id "
            "from your own `tasks` array.\n"
            "3. Your task ids must be unique within your response.\n"
            "4. Do not introduce edges that create a cycle."
        )
        return (
            f"Goals:\n{goals_block}\n\n"
            f"Completed/Failed/Cancelled tasks (READ-ONLY CONTEXT — "
            "preserve these verbatim at the start of the returned plan; "
            f"do NOT repeat them in your response):\n{history_json}\n\n"
            f"Operator steering note:\n{note}\n\n"
            f"{invariants}\n\n"
            "Generate only the NEW PENDING tasks (and their edges) that "
            "should run from here, taking the steering note into account. "
            "Respond with JSON only."
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
            "back to PENDING/RUNNING/BLOCKED.",
            f"   Terminal tasks in the current plan: {terminal_json}",
            "",
            "2. TERMINAL->TERMINAL EDGES (edges where BOTH endpoints are "
            "terminal tasks) MUST appear verbatim in your `edges` array. "
            "These edges are frozen history; dropping even one will fail "
            "validation.",
            f"   Required edges: {tt_edges_json}",
            "",
            "3. TASK IDS must be unique within `tasks`. Every edge's "
            "`from_task_id` and `to_task_id` must reference a task id "
            "that exists in your `tasks` array.",
            "",
            "4. The task graph must be ACYCLIC. Do not introduce edges that would create a cycle.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_correction_prompt(base_prompt: str, error: str) -> str:
        """Append validator / parser feedback to a refine prompt for retry.

        The retry appends the specific error the previous attempt hit and
        re-emphasises the structural invariants; keeping the base prompt
        verbatim means the LLM sees the same goals, history, and drift
        context, so the retry is a real second attempt, not a cold
        re-prompt.
        """
        return (
            f"{base_prompt}\n\n"
            "PREVIOUS ATTEMPT FAILED. The response you just emitted was "
            "rejected by the validator:\n"
            f"    {error}\n\n"
            "Re-read the STRUCTURAL INVARIANTS section above. Emit a "
            "corrected JSON plan that preserves every terminal task and "
            "every terminal->terminal edge verbatim. Respond with JSON "
            "only; no prose, no markdown fences."
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
        )
        try:
            await self._drift_emitter(signal)
        except Exception as exc:  # noqa: BLE001 -- emitter must never break refine
            log.warning(
                "LLMPlanner: drift emitter raised while signalling "
                "REFINE_VALIDATION_FAILED (%s); dropping signal",
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
    ) -> tuple[Plan | None, str]:
        """Run the retry loop for a single refine call.

        Calls the LLM up to ``self._max_refine_attempts`` times, parsing
        the response into a ``Plan`` and validating against ``prior_plan``.
        On each failed attempt the user prompt is extended with the
        error message (via :meth:`_build_correction_prompt`) so the LLM
        gets explicit feedback on what to fix. Returns ``(plan, "")`` on
        success, ``(None, last_error)`` on exhaustion. Exceptions from
        ``call_llm`` are treated as retryable.

        ``post_parse`` (optional) runs after ``_plan_from_json`` succeeds
        but BEFORE validation -- used by ``_refine_looping_tool_call`` to
        force the looper FAILED before the validator decides whether the
        revision honours the invariants.
        """
        user_prompt = base_user_prompt
        last_error = ""
        attempts = max(1, self._max_refine_attempts)
        for attempt in range(1, attempts + 1):
            try:
                raw = await self._call_llm(system_prompt, user_prompt, self._model)
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
                last_error = "empty or non-string LLM response"
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
            return revised, ""
        return None, last_error

    def _build_refine_prompt(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
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
        return (
            f"Goals:\n{goals_block}\n\n"
            f"Current plan:\n{plan_json}\n\n"
            f"Drift event:\n{drift_json}\n\n"
            f"{invariants_block}\n\n"
            "If the plan should change in light of this drift event, respond "
            "with an updated JSON plan using the same schema. If no change "
            "is warranted, respond with the current plan unchanged. Respond "
            "with JSON only."
        )

    # ---- Planner protocol ------------------------------------------------

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        if not goals:
            log.debug("LLMPlanner.generate: no goals provided; skipping plan")
            return None
        prompt = self._build_generate_prompt(goals, available_agents, context)
        try:
            raw = await self._call_llm(self._system_prompt, prompt, self._model)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLMPlanner.generate: call_llm raised %s; skipping plan", exc)
            return None
        if not raw or not isinstance(raw, str):
            log.warning("LLMPlanner.generate: empty/non-string LLM response; skipping plan")
            return None
        cleaned = _strip_code_fences(raw).strip()
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as exc:
            log.warning(
                "LLMPlanner.generate: failed to parse LLM output as JSON (%s); skipping",
                exc,
            )
            return None
        run_id = ""
        if context is not None:
            run_id = str(context.get("run_id") or "")
        plan = _plan_from_json(
            parsed,
            run_id=run_id,
            goal_ids=[g.id for g in goals if g.id],
        )
        if plan is None:
            log.warning("LLMPlanner.generate: parsed JSON did not contain a usable plan; skipping")
            return None
        try:
            plan.validate(for_revision=False)
        except ValueError as exc:
            log.warning(
                "LLMPlanner.generate: plan failed validation (%s); skipping",
                exc,
            )
            return None
        return plan

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        if plan is None:
            return None
        if drift.kind is DriftKind.USER_STEER:
            return await self._refine_user_steer(plan, drift, goals)
        if drift.kind in (
            DriftKind.LOOPING_TOOL_CALL,
            DriftKind.LOOPING_REASONING,
        ):
            # LOOPING_REASONING shares the "fail the current task, route
            # around it" shape with LOOPING_TOOL_CALL: the symptom is a
            # stuck loop on the currently-running task, and the repair
            # is to mark it FAILED and regenerate the rest.
            return await self._refine_looping_tool_call(plan, drift, goals)
        # REFINE_VALIDATION_FAILED is a terminal signal from ourselves --
        # refining on it would risk an infinite loop of validation
        # failures. The steerer is expected to skip refine for this
        # kind; defending here too is a belt-and-braces guard.
        if drift.kind is DriftKind.REFINE_VALIDATION_FAILED:
            return None
        try:
            base_user_prompt = self._build_refine_prompt(plan, drift, goals)
        except (TypeError, ValueError) as exc:
            log.warning("LLMPlanner.refine: failed to serialise inputs (%s)", exc)
            return None
        revised, last_error = await self._call_and_validate_refine(
            system_prompt=self._refine_system_prompt,
            base_user_prompt=base_user_prompt,
            prior_plan=plan,
            goals=goals,
            log_prefix="LLMPlanner.refine",
        )
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
            await self._emit_refine_validation_failed(plan, drift, last_error)
            return None
        # Stamp revision metadata so downstream sinks can render it.
        revised.revision_reason = drift.detail
        revised.revision_kind = str(drift.kind)
        revised.revision_severity = str(drift.severity)
        revised.revision_index = plan.revision_index + 1
        return revised

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
        invariants_block = self._render_structural_invariants_block(plan)
        return (
            f"Goals:\n{goals_block}\n\n"
            f"Already-finished tasks (preserve verbatim):\n"
            f"{json.dumps(history, default=str)}\n\n"
            f"LOOPING task (must appear in the returned plan with "
            f"status=FAILED, same id):\n{looper_block}\n\n"
            f"Other unfinished tasks (you may keep, drop, or rework):\n"
            f"{json.dumps(others, default=str)}\n\n"
            f"Drift detail:\n{drift.detail}\n\n"
            f"{invariants_block}\n\n"
            "Generate the updated plan. Respond with JSON only."
        )

    async def _refine_looping_tool_call(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
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
            """
            if not loop_id:
                return revised
            for t in revised.tasks:
                if t.id == loop_id and t.status not in _TERMINAL_STATUSES:
                    t.status = TaskStatus.FAILED
            if not any(t.id == loop_id for t in revised.tasks) and looping_task is not None:
                revised.tasks.insert(
                    0,
                    Task(
                        id=looping_task.id,
                        title=looping_task.title,
                        description=looping_task.description,
                        assignee_agent_id=looping_task.assignee_agent_id,
                        status=TaskStatus.FAILED,
                    ),
                )
            return revised

        revised, last_error = await self._call_and_validate_refine(
            system_prompt=self._looping_tool_call_system_prompt,
            base_user_prompt=base_user_prompt,
            prior_plan=plan,
            goals=goals,
            post_parse=_force_looper_failed,
            log_prefix="LLMPlanner._refine_looping_tool_call",
        )
        if revised is None:
            # Retries exhausted. Signal the failure explicitly, then
            # return the deterministic fallback so the looping task
            # cannot re-fire on the next tick.
            await self._emit_refine_validation_failed(plan, drift, last_error)
            return self._fallback_fail_loop_plan(plan, drift, looping_task)
        revised.revision_reason = drift.detail
        revised.revision_kind = drift.kind.value
        revised.revision_severity = str(drift.severity)
        revised.revision_index = plan.revision_index + 1
        return revised

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

    async def _refine_user_steer(
        self,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        """Delete-and-replan path for ``USER_STEER`` drift.

        Completed/failed/cancelled tasks are preserved verbatim (same
        ids, titles, assignees, statuses). Pending/running/blocked
        tasks are dropped; the LLM produces a fresh set of PENDING
        tasks that honour the operator's steering note. The returned
        plan reuses ``plan.id`` and ``plan.run_id`` so lineage stays
        intact.
        """
        completed = [t for t in plan.tasks if t.status in _TERMINAL_STATUSES]
        completed_ids = {t.id for t in completed}
        try:
            base_user_prompt = self._build_user_steer_prompt(completed, drift, goals)
        except (TypeError, ValueError) as exc:
            log.warning(
                "LLMPlanner._refine_user_steer: failed to serialise inputs (%s)",
                exc,
            )
            return None

        user_prompt = base_user_prompt
        attempts = max(1, self._max_refine_attempts)
        last_error = ""
        for attempt in range(1, attempts + 1):
            merged_plan, error = await self._user_steer_one_attempt(
                plan=plan,
                goals=goals,
                drift=drift,
                completed=completed,
                completed_ids=completed_ids,
                user_prompt=user_prompt,
            )
            if merged_plan is not None:
                return merged_plan
            last_error = error
            log.warning(
                "LLMPlanner._refine_user_steer: attempt %d/%d: %s",
                attempt,
                attempts,
                last_error,
            )
            if attempt < attempts:
                user_prompt = self._build_correction_prompt(base_user_prompt, last_error)
        # Retries exhausted -- signal and fall back to None (the steerer
        # keeps the prior plan and increments its backoff counter).
        await self._emit_refine_validation_failed(plan, drift, last_error)
        return None

    async def _user_steer_one_attempt(
        self,
        *,
        plan: Plan,
        goals: list[Goal],
        drift: DriftEvent,
        completed: list[Task],
        completed_ids: set[str],
        user_prompt: str,
    ) -> tuple[Plan | None, str]:
        """Run a single LLM attempt for the USER_STEER refine path.

        Returns ``(merged_plan, "")`` on success or ``(None, error)`` on
        any parse / merge / validate failure. The caller drives the
        retry loop; this helper keeps the merge logic in one place so
        both the first attempt and every retry apply exactly the same
        plumbing.
        """
        try:
            raw = await self._call_llm(self._user_steer_system_prompt, user_prompt, self._model)
        except Exception as exc:  # noqa: BLE001
            return None, f"call_llm raised: {exc}"
        if not raw or not isinstance(raw, str):
            return None, "empty or non-string LLM response"
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

        merged_plan = Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(fresh.goal_ids) or list(plan.goal_ids),
            tasks=merged_tasks,
            edges=merged_edges,
            summary=fresh.summary or plan.summary,
            revision_reason=f"user steering: {drift.detail}",
            revision_kind=DriftKind.USER_STEER.value,
            revision_severity=DriftSeverity.WARNING.value,
            revision_index=plan.revision_index + 1,
        )
        try:
            merged_plan.validate(for_revision=True, prior=plan)
        except ValueError as exc:
            return None, f"validator rejected revision: {exc}"
        return merged_plan, ""


__all__ = [
    "LLMPlanner",
    "PassthroughPlanner",
    "StaticPlanner",
]
