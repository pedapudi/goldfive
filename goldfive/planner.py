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
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
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

    def __init__(
        self,
        *,
        call_llm: Callable[[str, str, str], Awaitable[str]],
        model: str = "",
        system_prompt: str | None = None,
        refine_system_prompt: str | None = None,
    ) -> None:
        self._call_llm = call_llm
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._refine_system_prompt = refine_system_prompt or _REFINE_SYSTEM_PROMPT

    @property
    def model(self) -> str:
        return self._model

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

    def _build_generate_prompt(
        self,
        goals: list[Goal],
        available_agents: list[str],
        context: Mapping[str, Any] | None,
    ) -> str:
        goals_block = self._render_goals_block(goals)
        agents_block = self._render_agents_block(available_agents)
        context_block = ""
        if context:
            try:
                context_block = (
                    f"\nAdditional context (JSON):\n{json.dumps(dict(context), default=str)}\n"
                )
            except (TypeError, ValueError):
                context_block = ""
        return (
            f"Available agents:\n{agents_block}\n\n"
            f"Goals:\n{goals_block}\n"
            f"{context_block}\n"
            "Respond with a single JSON object following the schema."
        )

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
        return (
            f"Goals:\n{goals_block}\n\n"
            f"Current plan:\n{plan_json}\n\n"
            f"Drift event:\n{drift_json}\n\n"
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
        try:
            user_prompt = self._build_refine_prompt(plan, drift, goals)
        except (TypeError, ValueError) as exc:
            log.warning("LLMPlanner.refine: failed to serialise inputs (%s)", exc)
            return None
        try:
            raw = await self._call_llm(self._refine_system_prompt, user_prompt, self._model)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLMPlanner.refine: call_llm raised %s", exc)
            return None
        if not raw or not isinstance(raw, str):
            log.warning("LLMPlanner.refine: empty/non-string LLM response")
            return None
        cleaned = _strip_code_fences(raw).strip()
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as exc:
            log.warning("LLMPlanner.refine: failed to parse LLM output as JSON (%s)", exc)
            return None
        revised = _plan_from_json(
            parsed,
            run_id=plan.run_id,
            goal_ids=[g.id for g in goals if g.id] or list(plan.goal_ids),
            plan_id=plan.id,
        )
        if revised is None:
            log.warning("LLMPlanner.refine: parsed JSON did not contain a usable plan")
            return None
        # Stamp revision metadata so downstream sinks can render it.
        revised.revision_reason = drift.detail
        revised.revision_kind = str(drift.kind)
        revised.revision_severity = str(drift.severity)
        revised.revision_index = plan.revision_index + 1
        return revised


__all__ = [
    "LLMPlanner",
    "PassthroughPlanner",
    "StaticPlanner",
]
