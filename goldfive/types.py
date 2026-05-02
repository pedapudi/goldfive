"""Core dataclasses and enums for goldfive.

Pinned by ``docs/design/PROTOCOLS.md`` (v0.1). Types in this module are
pure data — mutation of live state happens only through a ``Steerer``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    # Overlay-model (goldfive#141): a task the :class:`PlanReconciler`
    # determined was legitimately skipped by the tree. Post-invocation
    # the reconciler looks at PENDING tasks that were never exercised
    # and marks those it deems optional / superseded as NOT_NEEDED
    # rather than COMPLETED / FAILED. Terminal; distinct from CANCELLED
    # so sinks can distinguish "user/system cancelled" from "tree
    # chose not to run because the plan-to-execution mapping made it
    # redundant".
    NOT_NEEDED = "NOT_NEEDED"

    @property
    def is_terminal(self) -> bool:
        """True iff this status is terminal (no further transitions allowed).

        Mirrors :data:`TERMINAL_TASK_STATUSES` for ergonomic per-status
        checks (``task.status.is_terminal``) without forcing callers to
        import the module-level set. The two sources stay in lock-step:
        if a new terminal member is added, update both.
        """
        return self in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.NOT_NEEDED,
        )


# Terminal statuses — a task in any of these cannot transition further
# and must not be re-invoked. This set is the **single source of truth**
# used by the steerer (state-transition guards), the tool-dispatch layer
# (terminal-task rejection), and the ADK adapter (invoke-loop early
# break). Do not duplicate this set; import from here. If ``TaskStatus``
# gains a new terminal member, add it here and every consumer sees it.
# See ``docs/design/TASK-LIFECYCLE.md`` §7.1 for the rationale.
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.NOT_NEEDED}
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
    # The run is stuck in a way the planner's refine cannot fix and the
    # user's judgment is required. Emitted by :class:`DefaultSteerer` at
    # Level 4 of the intervention ladder (goldfive#142) on persistent
    # refine failures, GOAL_DRIFT (CRITICAL), REFINE_VALIDATION_FAILED,
    # and other terminal drifts that shouldn't auto-recover. CRITICAL
    # severity only. Puts the Runner into a paused state
    # (``session.paused_for_human_intervention``) until a user-initiated
    # ``CONTROL_RESUME`` or ``CONTROL_STEER`` arrives.
    HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"
    # A single ADK LLM dispatch exceeded the configured wall-clock
    # budget (default 120s; configurable via ``make_adk_plugin
    # (llm_call_timeout_ms=...)``). Emitted by the goldfive ADK plugin's
    # per-call watcher task and paired with a cooperative cancel on the
    # invocation so subsequent callbacks short-circuit. CRITICAL
    # severity. This is the safety net for runaway thinking-token
    # generations (e.g. Qwen Q4 emitting 9961 tokens in 9.6 minutes,
    # demo-v8.log) — without it, a single bad turn wedges the run for
    # minutes. See goldfive#271 follow-up.
    LLM_CALL_TIMEOUT = "llm_call_timeout"
    # Reasoning-judge produced a "justified_deviation" verdict (iter-10).
    # The bound agent's chain-of-thought departed from the bound task,
    # but the judge's prompt context (recent tool error, surprising
    # result, discovered dependency, new information) plausibly
    # justifies the departure. Routes through the same goal-aware refine
    # path as OFF_TOPIC, but the steerer ladder always ABSORBs (no
    # escalation): a reality-provoked deviation is the right input for
    # plan-extension at every severity. Distinct from OFF_TOPIC so
    # condition_id / per-(kind, task_id) cooldown lifecycle stays
    # separable from unprovoked drift. PR 1 (proto + dataclass) ships
    # the kind only; PR 4 wires the steerer ladder entry and planner
    # prompt-selection. Until then no production code path constructs
    # this kind, so the lack of a ``_LADDER`` row is intentional.
    JUSTIFIED_DEVIATION = "justified_deviation"
    # Structural artifact-verification miss (iter-11E). Emitted when
    # ``report_task_succeeded`` fires for a task that declared
    # :attr:`Task.required_tool_calls` but those tools were not observed
    # during the task's execution span (per
    # ``Session.recent_tool_observations``). Catches agents — Qwen3.6-
    # 35B-A3B-FP8 has been seen doing this — that call
    # ``report_task_succeeded`` without invoking the tools that produce
    # the actual artifact (e.g. claiming a draft is written without ever
    # calling ``write_webpage_tool``). Routes through the goal-aware
    # refine path (same as PLAN_DIVERGENCE / OFF_TOPIC) so the planner
    # can revise the plan to address the unfinished work, and escalates
    # to PAUSE_ESCALATE on repeat-CRITICAL because repeated false
    # completion is a serious agent-correctness failure. PR 1 (proto +
    # dataclass + ladder + planner prompt-selection) ships the kind;
    # PR 2 wires the verification at ``report_task_succeeded`` time.
    INCOMPLETE_TOOL_CALLS = "incomplete_tool_calls"


class DriftSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SupersessionKind(StrEnum):
    """Kind of :attr:`Task.supersedes` link (goldfive#251).

    Authoritative on the :class:`Task` dataclass (not inferred from the
    old task's status at read time) because status is mutable — a
    CORRECT-kind link must stay CORRECT even after the plan progresses
    and the replacement itself finishes. Populated by
    :mod:`goldfive.planner`'s post-parse validator from the LLM's raw
    refine output: when the LLM sets a kind that disagrees with the old
    task's actual status, the validator coerces based on status and
    logs a WARNING.

    Values:

    * ``UNSPECIFIED`` — default / no supersedes link (or legacy plan).
    * ``REPLACE`` — old task was PENDING/RUNNING; the new task takes its
      slot in the DAG. Reporting-tool calls on the old id are rerouted
      to the new one (:func:`goldfive.reporting._resolve_effective_task_id`).
    * ``CORRECT`` — old task had already COMPLETED but the refiner
      judged the output drift-contaminated. The new task is modelled
      as a correction child: the old task stays in the plan as a
      historical COMPLETED node, an edge ``old -> new`` is added, and
      reporting-tool calls on the old id are NOT rerouted (the old
      work's completion is historical fact). See
      :meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised`.
    """

    UNSPECIFIED = "UNSPECIFIED"
    REPLACE = "REPLACE"
    CORRECT = "CORRECT"


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
class CancellationRequest:
    """Cooperative-cancellation directive for one in-flight invocation (goldfive#251).

    Written by :class:`~goldfive.steerer.DefaultSteerer` into the ADK
    ``session.state`` under
    :data:`goldfive.adapters._adk_state_protocol.KEY_CANCEL_REQUESTED`
    (a ``dict[str, CancellationRequest]`` keyed by ``invocation_id``)
    when a drift at CRITICAL severity warrants aborting the in-flight
    adapter dispatch without tearing down the whole run.

    Every adapter callback that can short-circuit a dispatch
    (``before_agent_callback``, ``before_model_callback``,
    ``before_tool_callback``) checks the flag keyed by the current
    ``invocation_id`` at the top of the callback. When set, the
    callback consumes the request (read-then-clear), emits an
    ``InvocationCancelled`` sink event carrying the rich context, and
    short-circuits the dispatch:

    * ``before_agent_callback`` — returns without invoking the agent.
    * ``before_tool_callback`` — returns ``{"status": "cancelled"}`` as
      the tool response so the parent LLM sees a minimal, prompt-safe
      marker (rich context stays in the sink event, NOT in the
      LLM-visible response — see goldfive#250 / #252 / #253 lessons).
    * ``before_model_callback`` — skips the LLM call.

    The rich ``CancellationRequest`` is deliberately NOT the object
    surfaced to the LLM. It exists for PROGRAMMATIC consumers
    (harmonograf observability, future Stream D correction-injection
    logic). The LLM-visible surface is the flat ``{"status":
    "cancelled"}`` dict above.

    Fields
    ------
    invocation_id:
        ADK ``invocation_id`` of the dispatch this request targets.
        Cancellation is agent-AGNOSTIC — a parent-child chain of
        invocations is handled by propagating a separate request per
        invocation_id, not by naming an agent role.
    reason:
        Short symbolic reason (``"drift"``, ``"user_steer"``,
        ``"plan_revised"``, …). Free-form string; consumers pattern-
        match prefixes when useful, but the wire contract is just "a
        string for the sink event".
    severity:
        Severity of the triggering drift. Only CRITICAL reaches cancel
        per the severity ladder (goldfive#142); recorded here so sink
        consumers can differentiate a user-requested cancel (always
        honoured regardless of severity) from a graduated-drift cancel.
    drift_id:
        Cross-reference to the triggering ``DriftEvent.id`` when this
        request was minted from a drift. Empty string for user-control
        cancels or programmatic calls with no drift origin.
    requested_at_ms:
        Wall-clock milliseconds at request time. Defaults to 0 so
        the structure is a cheap ``dataclass()`` literal in tests; the
        steerer's cancel-request path stamps it on mint.
    """

    invocation_id: str
    reason: str = "drift"
    severity: DriftSeverity = DriftSeverity.CRITICAL
    drift_id: str = ""
    requested_at_ms: int = 0
    # Free-form human-readable detail for operator-visible sink events.
    # NOT exposed to the LLM (the LLM-visible response is
    # ``{"status": "cancelled"}`` with no reason/detail/drift_kind — see
    # lessons from goldfive#250 / #252 / #253). Populated by the steerer
    # with a short one-line description of why cancel fired so
    # harmonograf's intervention aggregator has something to render.
    detail: str = ""
    # Drift kind that triggered the cancel, if any. Sink-event only;
    # never reaches the LLM. Stored as the string value of
    # :class:`DriftKind` rather than the enum itself to keep the
    # dataclass serialisable by plain ``dataclasses.asdict`` without
    # importing the enum registry.
    drift_kind: str = ""


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
    #: goldfive#205 — structured reason for the most recent CANCELLED /
    #: FAILED transition. Colon-prefixed tag + provenance id:
    #: ``upstream_failed:<id>``, ``run_aborted:<reason>``,
    #: ``user_cancel:<annotation_id>``, ``user_steer:<annotation_id>``,
    #: ``superseded_by_revision:<replacement_id>``. Empty for PENDING /
    #: RUNNING / COMPLETED / BLOCKED tasks. Populated by downstream
    #: sinks (harmonograf's ingest pipeline, persistence sink) from the
    #: ``TaskCancelled.reason`` / ``TaskFailed.reason`` envelope fields;
    #: goldfive's own in-memory planner / steerer does not mutate this
    #: field — it exists so storage-backed consumers have a stable
    #: schema slot threaded through the Task dataclass they re-export.
    cancel_reason: str = ""
    #: Explicit supersession link (goldfive#237). When the planner
    #: produces a replacement for a task that has failed / been
    #: cancelled / needs a different shape, the new task sets
    #: ``supersedes = <old_task.id>``. Consumed by
    #: :meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised` to
    #: re-pin ``session.current_task_id`` on the replacement, and by
    #: :mod:`goldfive.reporting` to reroute reporting-tool calls from
    #: the old terminal id to the live replacement. Default empty —
    #: legacy plans and tasks that are NOT replacements leave it unset.
    supersedes: str = ""
    #: goldfive#251: kind of :attr:`supersedes` link. ``UNSPECIFIED`` on
    #: legacy plans and on tasks without a ``supersedes`` target;
    #: :attr:`SupersessionKind.REPLACE` when the old task was
    #: PENDING/RUNNING and the new task takes its slot;
    #: :attr:`SupersessionKind.CORRECT` when the old task had already
    #: COMPLETED but the refiner judged its output drift-contaminated
    #: and the new task should be modelled as a correction child. Drives
    #: :meth:`goldfive.steerer.DefaultSteerer._emit_plan_revised` (CORRECT
    #: keeps the old task in the plan and inserts the new task as a DAG
    #: child) and :func:`goldfive.reporting._resolve_effective_task_id`
    #: (CORRECT suppresses the rerouting of reports from the old id to
    #: the new id — the old work is legitimately done).
    supersedes_kind: SupersessionKind = SupersessionKind.UNSPECIFIED
    #: iter-11E: tools that must be observed during this task's
    #: execution span before ``report_task_succeeded`` is accepted.
    #: Empty list (the default) means "no requirement" — legacy /
    #: opt-in semantics, so existing plans and tasks continue to work
    #: unchanged. PR 2 wires the verification at
    #: ``report_task_succeeded`` time and emits
    #: :attr:`DriftKind.INCOMPLETE_TOOL_CALLS` when any declared tool
    #: is missing from ``Session.recent_tool_observations`` for the
    #: task's execution span.
    required_tool_calls: list[str] = dataclasses.field(default_factory=list)


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
    # Opaque identifier of the event that triggered this plan's revision.
    # Mirrors ``PlanRevised.trigger_event_id`` so out-of-band PlanRevised
    # emitters (the SequentialExecutor's post-steerer plan-swap detector)
    # can thread the id through without re-deriving it.
    #
    # Non-empty for every revision:
    #   * User-control refines: source annotation_id from the ControlMessage.
    #   * Autonomous drift refines: the ``DriftEvent.id`` of the producing drift.
    #   * Validator-retry refines: chained — same ``trigger_event_id`` as
    #     the rejected attempt.
    # Empty on the initial plan (not a revision).
    #
    # See goldfive#199 / harmonograf#95 (rescope). Replaces the narrower
    # ``revision_annotation_id`` from #196/#197 which was user-control only.
    revision_trigger_event_id: str = ""

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
        _UNREACHABLE_PREDECESSOR_STATUSES = frozenset(
            {TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.NOT_NEEDED}
        )
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

    @classmethod
    def empty(cls, *, run_id: str = "") -> Plan:
        """Construct a fresh empty plan (revision_index=0, no tasks).

        Goldfive#271 Phase 4 seed used by :meth:`Runner.run` on the
        first turn so :meth:`Planner.handle_turn` always sees a
        non-None ``session.plan``. The runner installs the plan
        returned by ``handle_turn`` as revision 1 of this seed via
        ``DefaultSteerer.install_initial_plan``, so PlanRevised
        (not PlanSubmitted) fires uniformly for every turn — there is
        only one install path post-Phase-4.
        """
        return cls(
            id=uuid.uuid4().hex,
            run_id=run_id,
            goal_ids=[],
            tasks=[],
            edges=[],
            summary="",
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


def task_upstream_ready(plan: Plan, task_id: str) -> bool:
    """Return ``True`` if every upstream task of ``task_id`` is COMPLETED.

    Walks ``plan.edges`` to find all edges with ``to_task_id == task_id``
    and checks that each ``from_task_id``'s status is
    :attr:`TaskStatus.COMPLETED`. Tasks with no upstream edges are
    trivially ready.

    Supersession-aware: when an upstream predecessor ``A`` has been
    superseded by a replacement ``B`` (``B.supersedes == A.id``) and
    the edges table still references the original ``A``, the helper
    resolves to ``B``'s status instead. This fallback is vestigial
    when the refiner copies edges to point at the replacement id
    directly -- but some refine paths leave the edges pointing at the
    pre-supersession id, so we redirect here to keep the readiness
    evaluation anchored to the live task.

    Missing-endpoint edges (unknown ``from_task_id``) are treated as
    *not ready* -- callers should validate the plan upstream, but a
    dangling edge here conservatively blocks pinning rather than
    silently allowing an underspecified downstream to run.

    This helper is the DAG-readiness primitive the ADK adapter uses
    to gate ``goldfive.current_task_id`` pinning so a downstream task
    cannot be stamped onto an agent while its predecessors are still
    in flight (goldfive#242).
    """
    if not task_id:
        return True

    tasks_by_id: dict[str, Task] = {t.id: t for t in plan.tasks if t.id}
    # Supersession map: old_id -> replacement Task. The replacement's
    # status is what we want to evaluate when the edges table still
    # references the old (likely terminal) id.
    replacement_of: dict[str, Task] = {}
    for t in plan.tasks:
        if t.supersedes and t.supersedes in tasks_by_id:
            replacement_of[t.supersedes] = t

    for e in plan.edges:
        if e.to_task_id != task_id:
            continue
        upstream = tasks_by_id.get(e.from_task_id)
        if upstream is None:
            # Dangling edge — conservatively not ready.
            return False
        # Redirect through supersession when the edge still names the
        # superseded task. The replacement's status is the live signal.
        replacement = replacement_of.get(upstream.id)
        if replacement is not None:
            upstream = replacement
        if upstream.status is not TaskStatus.COMPLETED:
            return False
    return True


#: Conventional value for :attr:`Goal.source` indicating a goal was added
#: by a ``USER_STEER`` directive (goldfive#152 / #154). Goals carrying
#: this source are treated as "sticky": ``LLMPlanner.refine`` will
#: reject any revision whose tasks silently drop the goal, so later
#: drifts cannot unwind an operator steer by merely refining around it.
GOAL_SOURCE_USER_STEER: str = "USER_STEER"


@dataclasses.dataclass
class Goal:
    id: str
    summary: str
    success_predicate: Callable[[Session], bool] | None = None
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Provenance of the goal. Empty string for goals derived from the
    #: original user input; :data:`GOAL_SOURCE_USER_STEER` for goals
    #: added by a ``USER_STEER`` (operator steer) -- these are "sticky"
    #: and the planner will reject refines that silently drop them
    #: (goldfive#154).
    source: str = ""


@dataclasses.dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None  # original event that triggered detection
    # Stable goldfive-minted UUID4 identifying this drift event. Populated
    # at construction by default so every ``DriftEvent`` — user-control or
    # autonomous — carries a strict join key. Downstream consumers
    # (harmonograf's intervention aggregator) use this as the
    # ``trigger_event_id`` for a ``PlanRevised`` that the drift produced
    # when the drift was not minted from a user annotation. See
    # goldfive#199 / harmonograf#95 (rescope).
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    # Short rendering of the data goldfive fed into the detector that
    # produced this drift — e.g. the reasoning block the LLM judge saw,
    # the activity summary shown to the goal-drift classifier, the tool
    # invocation the loop detector matched. Populated on autonomous
    # drifts; left empty on user-control drifts (their source is the
    # paired annotation/ControlMessage and not a derived prompt
    # input). Forwarded onto the wire as ``DriftDetected.trigger_input``
    # for sinks that render a Gantt / timeline and want to answer "why
    # did goldfive flag this?" without re-fetching raw agent transcripts.
    trigger_input: str = ""
    # Source attribution for this drift (goldfive-steer-unification).
    # ``"user"`` — minted from a ``ControlMessage`` (USER_STEER / USER_CANCEL
    # / USER_PAUSE). ``"goldfive"`` — minted by a goldfive-internal
    # detector (reasoning judge, embedding detectors, goal-drift, loop
    # detectors, tool-loops, confabulation). Empty string for legacy /
    # unknown origins (pre-steer-unification producers).
    #
    # The default is empty; the steerer normalises it at the entry of
    # :meth:`DefaultSteerer._handle_drift` so every on-the-wire
    # ``DriftDetected`` carries either ``"user"`` (for ``DriftKind.USER_*``
    # kinds) or ``"goldfive"`` (everything else). Explicit call sites
    # that already know the source may set this at construction.
    authored_by: str = ""
    # True when a ``_handle_drift`` invocation decided to suppress a
    # goldfive-originated steer promotion because a recent user-authored
    # steer is still active within the configured freshness window. The
    # ``DriftDetected`` event still fires so operators see the detector
    # ran; the cancel-in-flight + refine + restart-message machinery is
    # elided. Always ``False`` on user-authored drifts.
    suppressed_by_user_steer: bool = False


@dataclasses.dataclass
class ObservedAction:
    """One observed agent invocation, reconciled against the plan.

    A snapshot of what the agent tree has actually done so the planner
    can compare the planned dispatch (``Plan.tasks``) against the
    observed dispatch. Emitted by the overlay-model ``PlanReconciler``
    (goldfive#141) and consumed by :meth:`LLMPlanner.refine` when the
    drift kind is ``DriftKind.PLAN_DIVERGENCE`` (goldfive#144).

    Fields
    ------
    agent_name:
        Display name of the agent that ran (e.g. ``"researcher"``).
    invocation_id:
        Stable id of the invocation (usually the ADK ``invocation_id``).
    parent_invocation_id:
        Invocation id of the parent span; empty string for top-level
        invocations (no parent). Lets the planner reconstruct
        parent/child relationships across the observed trace.
    started_at / completed_at:
        Wall-clock timestamps bracketing the invocation.
        ``completed_at`` is ``None`` while the invocation is still
        running.
    status:
        One of ``"running"``, ``"completed"``, or ``"failed"``.
    summary:
        Human-readable summary of what the invocation did — typically
        from the ``AgentInvocationCompleted.summary`` event, synthesised
        from partial output when the invocation is still running, or a
        short failure reason when status is ``"failed"``.
    """

    agent_name: str
    invocation_id: str
    parent_invocation_id: str
    started_at: datetime
    completed_at: datetime | None
    status: str
    summary: str


@dataclasses.dataclass(slots=True)
class RefineOutcome:
    """Per-(kind, task) outcome of the last refine attempt this turn.

    goldfive#215 (iter-8) P2: replaces the split
    ``Session.refine_failure_counts`` (numerical cap) +
    ``KEY_ACTIVE_DRIFTS`` lifecycle gate with a single outcome-tracked
    state machine. Cleared on every ``run_started`` boundary by
    :meth:`DefaultSteerer.reset_for_turn`.

    ``state``:
        ``"succeeded"`` — the most recent refine for this (kind, task)
        produced a landed plan revision; subsequent same-(kind, task)
        drifts on the same turn skip refine (the prior refine already
        addressed it). ``fail_count`` is reset to ``0``.

        ``"failed"`` — the most recent refine raised, returned None,
        or produced a no-op / validator-rejected revision; same-
        (kind, task) drifts continue retrying until ``fail_count``
        reaches :attr:`DefaultSteerer.REFINE_FAILURE_THRESHOLD`, then
        escalate via the REPEATED_FAILURE drift + non-recoverable
        ``mark_task_failed``.
    """

    state: str  # "succeeded" | "failed"
    fail_count: int = 0


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
    # Per-task one-shot flag for the standalone unreferenced-keyword
    # detector (``detect_unreferenced_keyword``). Promoted from the
    # severity-bump helper because whole-block cosine empirically fails
    # to separate drifted from on-topic reasoning on real embedding
    # models (see #223); the lexical signal fires independently so we
    # gate it per-task the same way as ``reasoning_cluster_flagged`` to
    # avoid drift-spam when the same off-topic reasoning block repeats
    # across turns.
    unreferenced_keyword_flagged: set[str] = dataclasses.field(default_factory=set)
    # Per-(drift_kind_value, task_id) outcome of the last refine attempt
    # this turn. goldfive#215 (iter-8) P2 — single-source-of-truth
    # replacement for the split ``refine_failure_counts`` (numerical cap)
    # + ``KEY_ACTIVE_DRIFTS`` lifecycle gate. Each entry is a
    # :class:`RefineOutcome` carrying ``state`` (``"succeeded"`` /
    # ``"failed"``) and a ``fail_count`` consulted by the intervention
    # ladder + the REPEATED_FAILURE escalation. Cleared on every
    # ``run_started`` boundary via :meth:`DefaultSteerer.reset_for_turn`,
    # so a fresh turn always starts with empty outcomes (per-turn scope
    # is the natural boundary for retry budgets).
    #
    # Concurrency: the dict is lock-free single-writer-per-session.
    # Reads and writes happen exclusively from ``DefaultSteerer``'s
    # ``observe`` / ``_handle_drift`` / ``_promote_drift_to_steer``
    # paths; ADK's adapter callback contract serialises drift delivery
    # per session, so concurrent writes for the same Session do not
    # arise in practice. Cross-session writes target distinct dicts
    # (one per Session). The same property held for the predecessor
    # ``refine_failure_counts`` field — the lock-free pattern is the
    # established contract, not a regression.
    refine_outcomes: dict[tuple[str, str], RefineOutcome] = dataclasses.field(default_factory=dict)
    # Per-task ``time.monotonic()`` timestamp of the most recent
    # task-progress signal — set on ``mark_task_running``,
    # ``mark_task_progress``, and every ``_emit_task_transitioned`` call.
    # Consumed by :class:`~goldfive.steerer.DefaultSteerer` to gate
    # drift escalation: a drift firing on a task that has been silent
    # for longer than the configured stall threshold escalates to
    # ``HUMAN_INTERVENTION_REQUIRED`` instead of looping the planner.
    # Replaces the deleted count-based cap (goldfive#271 follow-up):
    # progress-grounded escalation is structural — a productively-
    # iterating task has continuous progress events, a stuck task
    # does not. Sentinel task_id ``""`` covers trajectory-wide signals
    # which never gate (no task to be stalled).
    task_last_progress_at: dict[str, float] = dataclasses.field(default_factory=dict)
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
    # Monotonic-ish timestamp (``time.time()``, seconds since epoch) of the
    # last GOAL_DRIFT judge call fired via the task-boundary trigger
    # (goldfive#219). Prevents two task transitions <10s apart from paying
    # for back-to-back judge calls. Distinct from
    # ``_agent_turns_since_goal_check`` because turn-based scheduling is
    # already cost-bounded by the interval; task-boundary scheduling needs
    # its own rate limit. Default 0 (never fired).
    _last_goal_drift_check_ts: float = 0.0
    # Per-(agent, task) thinking-message counter for the LLM-as-a-judge
    # reasoning drift detector (goldfive#226). Keyed by
    # ``(agent_name, task_id)`` -> count of ``observe_reasoning`` calls
    # since the last judge firing (or since task/agent transition).
    # ``DefaultSteerer`` fires the judge on the first message of every
    # (agent, task) bucket (count=0) and then every
    # ``reasoning_drift_rate_limit`` messages after that. Scoped
    # per-(agent, task) — not globally, not per-task — so unrelated
    # unpinned turns from different agents don't share the empty-task-id
    # bucket and distort judge cadence. Pre-fix the key was a single
    # string keyed on ``current_task_id or ""``; every unpinned turn
    # from every agent collapsed onto the ``""`` bucket and legitimate
    # first-block judge firings were skipped.
    _reasoning_judge_counters: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)
    # Set to ``True`` by :class:`DefaultSteerer` when it escalates a
    # drift to Level 4 of the intervention ladder (goldfive#142).
    # Executors honour this flag the same way they honour a PAUSE
    # control: the pre-task loop blocks on the control channel until the
    # user issues a CONTROL_RESUME or CONTROL_STEER, which clears the
    # flag. Independent of the control-channel's own paused state so a
    # Runner without a bound control channel can still reflect the
    # pause to its sinks.
    paused_for_human_intervention: bool = False
    # Level 2 ladder handoff (goldfive#142). Set by :class:`DefaultSteerer`
    # when a WARNING-tier drift maps to Level 2 (NUDGE). The Runner /
    # overlay loop introduced by goldfive#141 picks this up after the
    # current invocation ends and issues a soft follow-up user message.
    # Plain list so two drifts in the same turn can queue independently;
    # the consumer pops from the front. Each entry is a short human-readable
    # directive; serialization not required.
    pending_nudges: list[str] = dataclasses.field(default_factory=list)
    # Level 3 ladder handoff (goldfive#142). Set by :class:`DefaultSteerer`
    # when a Level 3 escalation wants the Runner / overlay loop (goldfive#141)
    # to cancel the in-flight invocation and re-invoke with a composed
    # corrective user message. ``None`` outside a pending dispatch. The
    # consumer reads the value and clears it. Using a single slot (rather
    # than a queue) is deliberate: a second Level 3 while one is pending
    # overwrites the first -- the more-recent directive wins.
    pending_corrective_message: str | None = None
    # Orchestration-level session state dict (goldfive#152). Goldfive
    # owns keys under the ``goldfive.*`` namespace — see
    # :mod:`goldfive.orchestration_state` for the documented key names
    # and helpers. This is NOT the same surface as the ADK
    # ``session.state`` dict the ADK adapter writes to for agent-side
    # reads (that lives on the ADK ``Session`` object; see
    # :mod:`goldfive.adapters._adk_state_protocol`). This dict is
    # goldfive-orchestration-internal: the PlanReconciler stamps the
    # current task id/title here, the steerer writes active-steer
    # bookkeeping, and the ADK heal path records cancelled function
    # call ids so prompt templates / refine paths / downstream planners
    # can read a single, framework-agnostic source of truth.
    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Runtime-reasoning agent pin. Set by the ADK plugin's
    # ``before_agent_callback`` to the agent that is about to reason
    # (``last writer wins`` — reasoning is sequential within an
    # invocation). Distinct from ``Task.assignee_agent_id`` which
    # encodes the static plan intent: when a coordinator (assignee)
    # delegates to a child via ``AgentTool``, the child reasons under
    # the parent's task pin; ``current_agent_id`` reflects the actual
    # reasoner so the reasoning judge attributes drift correctly.
    # Defaults to ``""`` for legacy callers / pre-pin races; consumers
    # should fall back to ``task.assignee_agent_id`` when empty.
    current_agent_id: str = ""
    # Observed delegation lineage per-task. Keyed by ``task.id`` →
    # set of ``agent_id`` strings observed reasoning under that task
    # (initialised to ``{task.assignee_agent_id}`` on RUNNING and
    # extended by each ``delegation_observed`` whose pinned task_id
    # matches). Cleared on task terminal transition. Consumers (e.g.
    # the reasoning judge) can use this to distinguish "child of a
    # delegation chain rooted at the assignee" from "off-plan agent".
    task_lineage: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    # Ring buffer of recent tool-call observations consumed by the
    # iter-10 three-state reasoning judge so it can distinguish a
    # provoked deviation (the agent saw a tool error / surprising
    # result and pivoted) from an unprovoked one. Adapters push
    # entries via ``DefaultSteerer.note_tool_observation`` from their
    # ``after_tool_callback`` / ``on_tool_error_callback`` hooks; the
    # steerer trims to ``recent_tool_observations_max`` so the prompt
    # stays bounded regardless of run length. Each entry is a small
    # dict (``ts_ms``, ``agent_name``, ``task_id``, ``tool_name``,
    # ``args_preview``, ``result_preview``, ``is_error``,
    # ``error_message``) — framework-neutral, no protos, so sinks /
    # tests can introspect cheaply. Per-task scoping is applied at
    # READ time by the judge's prompt renderer (PR 3): writers store
    # everything so a deviation rooted in an earlier task's artefact
    # remains visible.
    recent_tool_observations: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # Cap for ``recent_tool_observations``. Default 16 covers a couple
    # of agent invocations on typical traces while keeping the prompt
    # block under ~1.5KB even before the per-entry truncation. Tunable
    # via the ``Steerer`` config so operators on tight context budgets
    # can drop it.
    recent_tool_observations_max: int = 16
    # Monotonic event sequence counter for sinks. When this Session was
    # built by :meth:`Conversation.next_turn_session`, the seed value is
    # the Conversation's running cursor (lifted from the previous turn's
    # high-water mark on :meth:`Conversation.absorb_turn`) — see
    # goldfive#271 Gap 2. Bare ``Session(run_id=...)`` constructions
    # remain at 0 so single-Session callers (tests, programmatic use)
    # see no behaviour change.
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        s = self._next_sequence
        self._next_sequence = s + 1
        return s

    def next_sequence_and_event_id(self) -> tuple[int, str]:
        """Atomic pair: increment sequence, mint matching event_id.

        Convenience wrapper around :meth:`next_sequence` +
        :meth:`next_event_id`. Producer migration in goldfive#271 Phase 3
        Addition B threads ``(sequence, event_id)`` together at every
        emit site so the proto envelope's ``Event.sequence`` and
        ``Event.event_id`` fields agree without the caller stashing a
        local ``seq`` variable. Use this from new emit sites; older sites
        that already call ``next_sequence()`` separately may pass the
        captured int into :meth:`next_event_id` without re-incrementing.
        """
        seq = self.next_sequence()
        return seq, self.next_event_id(seq)

    def next_event_id(self, sequence: int | None = None) -> str:
        """Return a globally-unique event id for the next emitted event.

        Format: ``{run_id}:{sequence}:{uuid4_short}`` where ``uuid4_short``
        is the first 8 hex characters of a fresh UUID4. The
        ``(run_id, sequence)`` prefix preserves chronological sortability
        and per-run debuggability; the uuid4 suffix guarantees PK
        uniqueness even when an outer system collapses multiple Sessions
        onto the same outer-session id (harmonograf#61's outer-session
        pin: two turns share the outer ``session_id`` and each turn
        restarts ``_next_sequence`` at 0, so the per-turn
        ``(session_id, run_id, sequence)`` triple is no longer unique).

        ``sequence`` (optional) lets a caller mint an event id around a
        sequence value it has already pulled — the typical producer
        pattern is ``seq = session.next_sequence(); evt_id =
        session.next_event_id(seq)`` so both the ``Event.sequence`` and
        ``Event.event_id`` proto fields can be set without
        double-incrementing the counter. When ``sequence`` is ``None``
        the call advances the counter itself for the convenience of
        callers that don't otherwise need the int.

        See goldfive#271 Phase 3 Addition B for the full rationale.
        """
        if sequence is None:
            sequence = self.next_sequence()
        suffix = uuid.uuid4().hex[:8]
        return f"{self.run_id}:{sequence}:{suffix}"

    @property
    def id(self) -> str:
        """Stable identifier for this :class:`Session`.

        Aliases ``run_id`` for the goldfive#155 ``Event.session_id``
        stamping contract: a goldfive Session maps 1:1 to a run/turn, so
        ``run_id`` is the session's identity. Downstream routing
        consumers (e.g. harmonograf) use this as the multiplex key when
        a single stream carries events from multiple Sessions.
        """
        return self.run_id
