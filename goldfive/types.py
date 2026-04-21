"""Core dataclasses and enums for goldfive.

Pinned by ``docs/design/PROTOCOLS.md`` (v0.1). Types in this module are
pure data — mutation of live state happens only through a ``Steerer``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


# Terminal statuses — a task in any of these cannot transition further
# and must not be re-invoked. This set is the **single source of truth**
# used by the steerer (state-transition guards), the tool-dispatch layer
# (terminal-task rejection), and the ADK adapter (invoke-loop early
# break). Do not duplicate this set; import from here. If ``TaskStatus``
# gains a new terminal member, add it here and every consumer sees it.
# See ``docs/design/TASK-LIFECYCLE.md`` §7.1 for the rationale.
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class DriftKind(StrEnum):
    TOOL_ERROR = "tool_error"
    AGENT_REFUSAL = "agent_refusal"
    NEW_WORK_DISCOVERED = "new_work_discovered"
    PLAN_DIVERGENCE = "plan_divergence"
    USER_STEER = "user_steer"
    USER_CANCEL = "user_cancel"
    USER_PAUSE = "user_pause"
    TASK_FAILED_RECOVERABLE = "task_failed_recoverable"
    TASK_FAILED_FATAL = "task_failed_fatal"
    CONTEXT_PRESSURE = "context_pressure"
    BLOCKED = "blocked"
    WRONG_AGENT = "wrong_agent"
    AGENT_TRANSFER = "agent_transfer"
    MODEL_REFUSAL = "model_refusal"
    STOPPED_EARLY = "stopped_early"
    TOO_MANY_STEPS = "too_many_steps"
    GOAL_UNREACHABLE = "goal_unreachable"
    TASK_TIMEOUT = "task_timeout"
    REPEATED_FAILURE = "repeated_failure"
    UNEXPECTED_OUTPUT = "unexpected_output"
    SCHEMA_VIOLATION = "schema_violation"
    HALLUCINATION_SUSPECTED = "hallucination_suspected"
    SAFETY_CONCERN = "safety_concern"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    CUSTOM = "custom"
    LOOPING_TOOL_CALL = "looping_tool_call"
    LOOPING_REASONING = "looping_reasoning"
    REASONING_CLUSTER_TIGHTENING = "reasoning_cluster_tightening"
    CONFUSION = "confusion"
    OFF_TOPIC = "off_topic"
    # INTENT_DIVERGENCE fires at a *variable* severity
    # (INFO / WARNING / CRITICAL) based on how far the reasoning has
    # drifted from ``session.goals`` + the current task topic. The
    # kind is stable so callers filtering by kind see one signal;
    # severity differentiates. See
    # ``goldfive/drift/reasoning.py::detect_intent_divergence`` and
    # ``docs/design/DRIFT.md`` for the graduated similarity bands.
    INTENT_DIVERGENCE = "intent_divergence"
    # Opt-in reflective self-progress check: agent said it *is* making
    # progress but with low confidence (< 0.5). INFO severity.
    UNCERTAIN_PROGRESS = "uncertain_progress"
    # Opt-in reflective self-progress check: agent reported it is *not*
    # making progress. WARNING severity -- triggers refine.
    SELF_REPORTED_STUCK = "self_reported_stuck"
    # Cheap structural confabulation-risk signal: the current task's
    # title/description implies external data access (research, lookup,
    # verify, review, fetch, etc.) but the agent produced non-empty
    # output without calling a single tool. INFO severity -- record-only,
    # does not trigger refine. Surfaced to the user so they can decide
    # whether to cancel or let the run proceed. See
    # :func:`goldfive.drift.classify_confabulation_risk`.
    CONFABULATION_RISK = "confabulation_risk"
    # AgentTool-per-invoke cap exceeded. Fires once per invocation when
    # a coordinator's LLM delegates via ADK AgentTool more times than
    # ``ADKAdapter(agent_tool_cap=N)`` allows (default 16). The backstop
    # for user-supplied coordinator prompts that describe a pipeline
    # and keep delegating forever instead of letting goldfive drive
    # the next task round. CRITICAL severity: the current task is
    # marked failed and the Steerer is given a chance to refine /
    # retry. See goldfive#130.
    RUNAWAY_DELEGATION = "runaway_delegation"
    # The planner-LLM's refine response could not be parsed or could not
    # pass the structural validator after the configured number of retry
    # attempts. CRITICAL severity; emitted by :class:`LLMPlanner` right
    # before it falls back to the prior plan (or, for the looping-tool
    # refine path, the deterministic fail-the-looper plan). This drift
    # kind is a terminal signal -- DefaultSteerer deliberately does NOT
    # trigger another ``planner.refine`` on it (infinite-loop risk) and
    # leaves the choice (steer again, cancel, or accept the fallback) to
    # the operator. See goldfive#133.
    REFINE_VALIDATION_FAILED = "refine_validation_failed"
    # Periodic trajectory-level goal-alignment check. Unlike the other
    # event-driven drift kinds (which classify one LLM response or one
    # tool result at a time), GOAL_DRIFT fires after a configurable
    # number of agent invocations when an LLM-judge concludes the tree's
    # accumulated activity is not advancing ``session.goals``. CRITICAL
    # severity -- routed to the #142 Level 4 intervention tier (pause +
    # HUMAN_INTERVENTION_REQUIRED). Gated behind
    # ``Runner(goal_drift_enabled=...)`` and a ``goal_drift_call_llm``
    # callable on :class:`DefaultSteerer`; mock-only runs never see it.
    # See :func:`goldfive.drift.goals.classify_goal_drift` and
    # goldfive#143.
    GOAL_DRIFT = "goal_drift"


class DriftSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_RANK: dict[str, int] = {
    DriftSeverity.INFO.value: 0,
    DriftSeverity.WARNING.value: 1,
    DriftSeverity.CRITICAL.value: 2,
}


def severity_rank(sev: DriftSeverity | str) -> int:
    """Ordinal rank for comparing ``DriftSeverity`` values."""
    v = sev.value if isinstance(sev, DriftSeverity) else str(sev)
    return _SEVERITY_RANK.get(v, -1)


@dataclasses.dataclass
class Task:
    id: str
    title: str
    description: str = ""
    assignee_agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    predicted_start_ms: int = 0
    predicted_duration_ms: int = 0
    bound_span_id: str = ""


@dataclasses.dataclass
class TaskEdge:
    from_task_id: str
    to_task_id: str


@dataclasses.dataclass
class Plan:
    id: str
    run_id: str
    goal_ids: list[str]
    tasks: list[Task]
    edges: list[TaskEdge]
    summary: str = ""
    revision_reason: str = ""
    revision_kind: str = ""  # DriftKind value (str) or ""
    revision_severity: str = ""  # DriftSeverity value (str) or ""
    revision_index: int = 0

    def validate(self, for_revision: bool = False, *, prior: Plan | None = None) -> None:
        """Structurally validate this plan. Raise ``ValueError`` on failure.

        Checks, in order:

        1. Every task has a non-empty ``id``.
        2. Task ids are unique within ``tasks``.
        3. Every edge's ``from_task_id`` and ``to_task_id`` reference a
           task that exists in ``tasks``.
        4. The task graph is acyclic (every task must be placeable by
           ``topological_stages``; any leftover is a cycle member).
        5. When ``for_revision`` is ``False`` (the default — the
           creation path), every task's ``status`` must be
           ``TaskStatus.PENDING``. Revised plans legitimately carry
           COMPLETED / FAILED / CANCELLED tasks preserved from the prior
           plan, so this check is skipped when ``for_revision`` is
           ``True``.
        6. When ``for_revision`` is ``True`` and a ``prior`` plan is
           supplied, enforce the cross-revision preservation contract
           from ``docs/design/PLAN-LIFECYCLE.md`` §3.1 and §3.2:

           - **Terminal task preservation (§3.1).** Every task in
             ``prior.tasks`` whose status is terminal must appear in
             ``self.tasks`` with the same id AND the same terminal
             status — no regression from COMPLETED back to PENDING is
             allowed, and dropping a terminal task is forbidden.
           - **Terminal→terminal edge preservation (§3.2).** Every
             edge in ``prior.edges`` where both endpoints were terminal
             in ``prior`` must appear in ``self.edges``. Historical
             topology between frozen tasks is frozen.
        7. No CANCELLED/FAILED→PENDING edges (reachability invariant,
           goldfive#137). A PENDING task whose predecessor is
           ``CANCELLED`` or ``FAILED`` is definitionally unexecutable:
           the executor only schedules a PENDING task when every
           predecessor reaches ``COMPLETED``, and these absorbing
           terminal states never fire that transition. Revisions that
           graft new PENDING tasks onto the graveyard of the prior
           plan are rejected here -- new work must form an independent
           sub-DAG with its own root(s) (or chain off a COMPLETED
           predecessor, which is immediately eligible and therefore
           safe). ``COMPLETED`` predecessors are *allowed* because
           that is the natural in-flight DAG shape: a finished stage
           feeding into a still-PENDING next stage.

        This is a pure-data validator: it does not mutate the plan. It
        is intended to be called at plan creation (``LLMPlanner.generate``)
        and at plan revision (``LLMPlanner.refine`` /
        ``DefaultSteerer._apply_revision``) so malformed plans are
        rejected before they are installed on a ``Session``.
        """
        # 1. & 2. ids present and unique.
        seen: set[str] = set()
        for t in self.tasks:
            if not t.id:
                raise ValueError("plan contains a task with an empty id")
            if t.id in seen:
                raise ValueError(f"duplicate task id in plan: {t.id!r}")
            seen.add(t.id)

        # 3. edges reference known tasks.
        for e in self.edges:
            if e.from_task_id not in seen:
                raise ValueError(
                    f"edge references unknown task id (from_task_id={e.from_task_id!r})"
                )
            if e.to_task_id not in seen:
                raise ValueError(f"edge references unknown task id (to_task_id={e.to_task_id!r})")

        # 4. no cycles. topological_stages places every non-cycle task;
        # any task left over is part of a cycle. We compute placement
        # locally (rather than calling topological_stages) so we can
        # avoid its edge-tolerance behaviour — validation wants a clean
        # signal.
        indeg: dict[str, int] = {tid: 0 for tid in seen}
        children: dict[str, list[str]] = {tid: [] for tid in seen}
        for e in self.edges:
            children[e.from_task_id].append(e.to_task_id)
            indeg[e.to_task_id] += 1
        ready = [tid for tid, d in indeg.items() if d == 0]
        placed: set[str] = set()
        while ready:
            tid = ready.pop()
            if tid in placed:
                continue
            placed.add(tid)
            for child in children[tid]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
        unplaced = seen - placed
        if unplaced:
            raise ValueError(f"plan contains a cycle among tasks: {sorted(unplaced)!r}")

        # 5. creation-time: all tasks must be PENDING.
        if not for_revision:
            for t in self.tasks:
                if t.status is not TaskStatus.PENDING:
                    raise ValueError(
                        f"task {t.id!r} has non-PENDING status {t.status.value!r} at plan creation"
                    )

        # 6. revision-time with a prior plan: enforce terminal-task and
        # terminal->terminal-edge preservation (PLAN-LIFECYCLE.md §3.1,
        # §3.2). Skipped when ``prior`` is None — callers that do not
        # supply the outgoing plan get the legacy structural checks only.
        if for_revision and prior is not None:
            new_by_id: dict[str, Task] = {t.id: t for t in self.tasks}
            prior_terminal_ids: set[str] = set()
            for t in prior.tasks:
                if t.status not in TERMINAL_TASK_STATUSES:
                    continue
                prior_terminal_ids.add(t.id)
                new_t = new_by_id.get(t.id)
                if new_t is None:
                    raise ValueError(f"terminal task {t.id!r} missing in revision")
                if new_t.status is not t.status:
                    raise ValueError(f"terminal task {t.id!r} regressed to {new_t.status.value!r}")
            # Every terminal->terminal edge in the outgoing plan must
            # appear verbatim in the revision.
            new_edges: set[tuple[str, str]] = {(e.from_task_id, e.to_task_id) for e in self.edges}
            for e in prior.edges:
                if (
                    e.from_task_id in prior_terminal_ids
                    and e.to_task_id in prior_terminal_ids
                    and (e.from_task_id, e.to_task_id) not in new_edges
                ):
                    raise ValueError(
                        "terminal->terminal edge "
                        f"{e.from_task_id!r} -> {e.to_task_id!r} missing in revision"
                    )

        # 7. reachability invariant (goldfive#137): no edge from an
        # *absorbing* terminal task (CANCELLED / FAILED) to a PENDING
        # task. The executor only schedules a PENDING task once every
        # predecessor has reached COMPLETED; CANCELLED and FAILED
        # states never transition to COMPLETED, so a PENDING task
        # hanging off a CANCELLED/FAILED predecessor is definitionally
        # unexecutable -- the entire sub-DAG stalls. This catches the
        # shape LLMs emit when they "graft" new work onto the end of
        # the prior plan (e.g. ``research -> r1`` where ``research`` is
        # CANCELLED and ``r1`` is the new PENDING root). New tasks must
        # form an independent sub-DAG starting from no predecessors (or
        # from predecessors that are PENDING/RUNNING/BLOCKED and can
        # still progress to COMPLETED).
        #
        # COMPLETED predecessors are *explicitly allowed* here because
        # the executor's eligibility rule is "all predecessors must be
        # COMPLETED", so a PENDING task whose predecessor is COMPLETED
        # is immediately eligible. The natural in-flight snapshot of a
        # running plan -- a done stage feeding into a still-PENDING
        # stage -- is the archetype the validator must accept.
        #
        # This check is safe to run on every plan: the creation path
        # (``for_revision=False``) already requires all tasks to be
        # PENDING (step 5) so no CANCELLED/FAILED task exists as a
        # predecessor.
        _UNREACHABLE_PREDECESSOR_STATUSES = frozenset({TaskStatus.CANCELLED, TaskStatus.FAILED})
        tasks_by_id: dict[str, Task] = {t.id: t for t in self.tasks}
        for e in self.edges:
            from_task = tasks_by_id.get(e.from_task_id)
            to_task = tasks_by_id.get(e.to_task_id)
            if from_task is None or to_task is None:
                # step 3 guarantees both endpoints resolve; belt-and-braces.
                continue
            if (
                from_task.status in _UNREACHABLE_PREDECESSOR_STATUSES
                and to_task.status is TaskStatus.PENDING
            ):
                raise ValueError(
                    f"edge {e.from_task_id!r} -> {e.to_task_id!r} would make PENDING "
                    f"task unexecutable: from-task is {from_task.status.value}. "
                    f"New tasks must form an independent sub-DAG starting from no "
                    f"predecessors — do not graft new work onto CANCELLED or "
                    f"FAILED tasks (their status never transitions to COMPLETED, "
                    f"so downstream PENDING tasks can never become eligible)."
                )

    def topological_stages(self) -> list[list[Task]]:
        """Return tasks grouped into topological stages (Kahn's algorithm).

        Each stage contains tasks whose dependencies are all satisfied by
        tasks in earlier stages. Tasks with no deps live in stage 0.
        Cycles or edges referencing unknown task ids are tolerated — any
        task that can never be placed is appended to a final trailing
        stage so the full set is always returned.
        """
        tasks_by_id = {t.id: t for t in self.tasks if t.id}
        indeg: dict[str, int] = {tid: 0 for tid in tasks_by_id}
        children: dict[str, list[str]] = {tid: [] for tid in tasks_by_id}
        for e in self.edges:
            if e.from_task_id in tasks_by_id and e.to_task_id in tasks_by_id:
                children[e.from_task_id].append(e.to_task_id)
                indeg[e.to_task_id] += 1

        stages: list[list[Task]] = []
        ready = [tid for tid, d in indeg.items() if d == 0]
        placed: set[str] = set()
        while ready:
            stage_ids = sorted(ready)
            stages.append([tasks_by_id[tid] for tid in stage_ids])
            placed.update(stage_ids)
            next_ready: list[str] = []
            for tid in stage_ids:
                for child in children[tid]:
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        leftover = [t for tid, t in tasks_by_id.items() if tid not in placed]
        if leftover:
            stages.append(leftover)
        return stages


@dataclasses.dataclass
class Goal:
    id: str
    summary: str
    success_predicate: Callable[[Session], bool] | None = None
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None  # original event that triggered detection


@dataclasses.dataclass
class Session:
    """Live state for one Runner.run() invocation.

    ``conversation_id`` links this turn to the owning :class:`~goldfive.conversation.Conversation`
    and is stable across successive turns on the same Runner. It
    defaults to ``""`` for legacy callers that build Sessions directly
    without going through a Conversation.
    """

    run_id: str
    conversation_id: str = ""
    goals: list[Goal] = dataclasses.field(default_factory=list)
    plan: Plan | None = None
    current_task_id: str = ""
    completed_results: dict[str, str] = dataclasses.field(default_factory=dict)
    # task_id -> progress fraction 0-1
    task_progress: dict[str, float] = dataclasses.field(default_factory=dict)
    # task_id -> last agent note
    agent_notes: dict[str, str] = dataclasses.field(default_factory=dict)
    divergence_flag: bool = False
    history: list[Any] = dataclasses.field(default_factory=list)
    started_at_ms: int = 0
    # Waiters for outstanding human-in-the-loop approvals. Keyed by
    # ``target_id``: task_id for Flow A (report_awaiting_approval) and the
    # ADK function_call_id for Flow B (ADK require_confirmation). The event
    # is set by the control dispatcher when APPROVE / REJECT arrives.
    pending_approvals: dict[str, asyncio.Event] = dataclasses.field(default_factory=dict)
    # Per-approval metadata. Populated when the waiter is registered; the
    # dispatcher adds ``decision`` ("approve" | "reject") and optional
    # ``detail`` before setting the event.
    pending_approvals_meta: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # Recent reasoning-content blocks emitted by the adapter's
    # ``emit_reasoning`` hook. Bounded to the last ``reasoning_history_max``
    # entries so long runs do not accumulate chain-of-thought forever.
    # Consumed by the reasoning-drift detectors (see ``goldfive.drift_reasoning``).
    reasoning_history: list[str] = dataclasses.field(default_factory=list)
    reasoning_history_max: int = 20
    # Per-task one-shot flags for the graduated reasoning-similarity
    # ladder. ``REASONING_CLUSTER_TIGHTENING`` (INFO, 0.75 <= cosine <
    # 0.9) fires AT MOST ONCE per ``current_task_id`` to avoid flooding
    # the event stream when a run stays in a tight-cluster regime for
    # many turns. ``reasoning_loop_flagged`` is reserved for an analogous
    # one-shot dedup on the ``LOOPING_REASONING`` WARNING tier (the
    # "cliff"); it is declared here so the two flags live as a pair,
    # though the cliff detector does not consult it today. Keys are task
    # ids that have already emitted the corresponding drift on this
    # session.
    reasoning_cluster_flagged: set[str] = dataclasses.field(default_factory=set)
    reasoning_loop_flagged: set[str] = dataclasses.field(default_factory=set)
    # Per-(drift_kind_value, task_id) consecutive refine-failure counter.
    # Incremented each time ``planner.refine`` raises or returns ``None``
    # for the given (kind, task) tuple; reset on a successful refine.
    # Consumed by :class:`~goldfive.steerer.DefaultSteerer` to back off
    # and mark the task FAILED after N consecutive failures, preventing
    # the same drift from looping until ``max_task_invocations`` trips.
    refine_failure_counts: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)
    # Counter of LLM turns observed since the last reflective self-progress
    # check. Incremented by ``DefaultSteerer.note_llm_call`` (which adapters
    # call once per LLM invocation when the opt-in reflective check is
    # enabled) and reset to 0 after a check runs or on task transition. The
    # steerer fires ``maybe_run_reflective_check`` once this counter reaches
    # its configured interval. See ``docs/design/DRIFT.md`` §"Reflective
    # self-progress check" and ``docs/design/PLAN-LIFECYCLE.md`` §8.
    _llm_calls_since_check: int = 0
    # Task id for which the counter is currently tracking. Used to reset
    # the counter cleanly on task transitions without plumbing an explicit
    # reset call through every ``mark_task_*`` path.
    _reflective_check_task_id: str = ""
    # Counter of agent invocations observed since the last GOAL_DRIFT
    # trajectory-level check (goldfive#143). Incremented by
    # ``DefaultSteerer.note_agent_turn`` (which adapters call once per
    # ``after_run_callback`` / equivalent completion hook when the opt-in
    # goal-drift judge is configured) and reset to 0 after a check fires.
    # No task-id scoping: GOAL_DRIFT is a trajectory-level signal, so the
    # counter persists across task transitions.
    _agent_turns_since_goal_check: int = 0
    # Ring buffer of recent agent activity summaries fed to the
    # :func:`goldfive.drift.classify_goal_drift` judge. Adapters push
    # entries via ``DefaultSteerer.note_agent_activity``; the steerer
    # trims to ``goal_drift_activity_window`` entries to bound the
    # prompt. Each entry is a small dict (``kind``, ``agent_name``,
    # ``task_id``, ``detail``) rather than a full event proto to keep
    # this framework-neutral.
    recent_agent_activity: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # monotonic event sequence counter for sinks
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        s = self._next_sequence
        self._next_sequence = s + 1
        return s
