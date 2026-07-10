"""Three-arm counterfactual bench harness (AGENCY-PRESERVATION.md PR 13a).

This is the scaffold the Stage-3 measurement gate (§3 PR 13, §4
"Counterfactual gate", §5.8) is built on. It runs the **same workload**
under three steering regimes and records per-arm metrics from the
*captured artifacts* (goldfive#447 ``Session.completed_outputs`` + goal
predicates) and the *sink event stream* (``SignalDelivered`` /
``SignalOutcome`` / ``DriftDetected`` / ``RunAborted`` — the PR-5
telemetry), never from parsing agent prose:

* **arm A — ``baseline``**: ``wrap(judge_only=True)``. The judge-only
  counterfactual (goldfive#446): the wrapped agent runs natively, drift
  judges stay armed, and ZERO planning / steering authority is exercised.
  This is the bar arm B must be *non-inferior to* before any default
  flip (§3 PR 13).
* **arm B — ``signal``**: the new SIGNAL regime —
  ``observation_only=False`` plus the new-regime flags
  (``signal_channel=request_context`` once PR 6 lands, ``plan_mode=ledger``
  once PR 10 lands, …). Arms are defined as **flag dicts of environment
  variables** so the harness does NOT hard-depend on unmerged PRs: a flag
  whose env var this build's :meth:`RuntimeConfig.from_env` does not yet
  consult is applied to the environment, reported as *pending*, and simply
  no-ops until the PR that reads it lands (graceful degradation — §5.1
  "no-op by default").
* **arm C — ``legacy``**: the legacy ladder regime —
  ``GOLDFIVE_STEER_LEGACY_LADDER=1`` (PR 7's escape hatch) +
  ``observation_only=False`` + ``GOLDFIVE_CANCEL_INFLIGHT_SCOPE=all`` (the
  PR-1 kill-switch that restores cancel-on-every-install). Arm C exists so
  regressions are *measurable rather than argued about* (§5.8).

Why env-var flag dicts (not :class:`SteeringConfig` kwargs)
----------------------------------------------------------
``signal_channel`` / ``GOLDFIVE_STEER_LEGACY_LADDER`` / ``plan_mode`` do
not exist in this build (PRs 6/7/10 are unmerged on the integration
branch). Passing them as :class:`SteeringConfig` constructor kwargs would
raise ``TypeError`` the day they are defined-but-renamed; passing them as
**env vars** that :meth:`RuntimeConfig.from_env` reads means an unknown
flag is silently ignored today and *automatically* picked up the moment
the consuming PR teaches ``from_env`` about it — zero harness change. The
harness reports which arm flags this build actually consults
(:func:`arm_flag_status`) so a *pending* flag is never silently mistaken
for an *applied* one.

Determinism for the smoke test
------------------------------
``13b`` (running the real bench against a live model) is a separate,
gated task. For ``13a``'s self-test the workload is driven by a stub
``call_llm`` and an optional :func:`inject_demo_signals` deterministic
drift sequence that exercises the **real** ``DefaultSteerer`` dispatch
path (the same methods production calls) so the telemetry pipeline —
dispatch → :class:`SignalLedger` → wire event → sink → metrics / shadow
diff — is validated end-to-end. ``signal_telemetry`` ships DEFAULT OFF
(goldfive#456): every arm here enables it explicitly, and the tooling
treats *zero parsed signal events* as a loud error, never an empty report
(§5.6 integration-not-unit).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from goldfive import (
    CallableAdapter,
    Goal,
    InvocationResult,
    LiteralGoalDeriver,
    Plan,
    Session,
    StaticPlanner,
    Task,
    wrap,
)
from goldfive.config import RuntimeConfig
from goldfive.results import ExecutionOutcome, evaluate_goal_predicates
from goldfive.sinks import InMemorySink, JSONLPersistenceSink
from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    TaskKind,
    TaskStatus,
)

__all__ = [
    "Arm",
    "ArmMetrics",
    "GOAL_GRADE_ABORTED",
    "GOAL_GRADE_MET",
    "GOAL_GRADE_UNMEASURED",
    "GOAL_GRADE_UNMET",
    "MANAGED_STEER_ENV",
    "KNOWN_STEER_ENV",
    "Scenario",
    "apply_arm_env",
    "arm_flag_status",
    "default_arms",
    "inject_demo_signals",
    "make_linear_run_driver",
    "metrics_from_events",
    "outcome_terminality",
    "run_arm",
    "run_arms",
]

#: The §6.4 goal-grading vocabulary. A run is only ``MET`` / ``UNMET`` when a
#: grading SIGNAL exists — a :attr:`Goal.success_predicate` or an
#: :attr:`TaskKind.OUTCOME` deliverable task. With neither, the run is
#: ``UNMEASURED`` (NOT silently ``MET``): ``run.success`` alone is explicitly
#: not a flip signal (§6.4 rule 1, deviation 1). An aborted run is a measured
#: failure regardless of grading signals (``ABORTED``).
GOAL_GRADE_MET = "met"
GOAL_GRADE_UNMET = "unmet"
GOAL_GRADE_ABORTED = "aborted"
GOAL_GRADE_UNMEASURED = "unmeasured"


# ---------------------------------------------------------------------------
# Flag-dict configurability + graceful degradation
# ---------------------------------------------------------------------------

#: Steering env vars this build's :meth:`SteeringConfig.from_env` actually
#: consults (see ``goldfive/config.py``). An arm flag whose env var is in
#: this set is *applied* (it changes the resolved config); a flag NOT in it
#: is *pending* (applied to the environment, but inert until the PR that
#: teaches ``from_env`` about it lands). Reviewers extend this set in the
#: same PR that adds the env read — e.g. PR 6 adds
#: ``GOLDFIVE_STEER_SIGNAL_CHANNEL``, PR 7 ``GOLDFIVE_STEER_LEGACY_LADDER``,
#: PR 10 ``GOLDFIVE_PLAN_MODE``.
KNOWN_STEER_ENV: frozenset[str] = frozenset(
    {
        "GOLDFIVE_STEER_THRESHOLD",
        "GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS",
        "GOLDFIVE_STEER_OBSERVATION_ONLY",
        "GOLDFIVE_STEER_CONTEXT_EDITOR_RULES",
        "GOLDFIVE_STEER_DESCRIPTIVE_GROWTH",
        "GOLDFIVE_STEER_SIGNAL_TELEMETRY",
        "GOLDFIVE_CANCEL_INFLIGHT_SCOPE",
        "GOLDFIVE_PLAN_MODE",  # PR 10 — SteeringConfig.from_env now reads it
        "GOLDFIVE_STEER_SIGNAL_CHANNEL",  # PR 6 — SteeringConfig.from_env now reads it
        "GOLDFIVE_STEER_LEGACY_LADDER",  # PR 7 — SteeringConfig.from_env now reads it
        "GOLDFIVE_STEER_PIN_ASSIGNED_TASK",  # PR 9 — SteeringConfig.from_env now reads it
        "GOLDFIVE_STEER_GRACE_WINDOW_TURNS",  # PR 8 — SteeringConfig.from_env now reads it
    }
)

#: Forward-declared env vars the roadmap will introduce. Listed here so the
#: harness *clears* them between arms (no cross-arm leakage) even before any
#: build consults them — and so an operator reading an arm's flag dict sees
#: the intended end-state. They remain *pending* until promoted into
#: :data:`KNOWN_STEER_ENV`. (Empty now that PR 7 promoted the legacy-ladder
#: hatch; kept as the extension point for future roadmap flags.)
_PENDING_STEER_ENV: frozenset[str] = frozenset()

#: Every env var the harness owns: snapshotted-and-cleared on each arm so
#: an arm only ever sees its own flag dict plus the ambient (non-managed)
#: environment.
MANAGED_STEER_ENV: frozenset[str] = KNOWN_STEER_ENV | _PENDING_STEER_ENV


@dataclasses.dataclass(frozen=True)
class Arm:
    """One steering regime under test, defined as an env-var flag dict.

    ``env`` maps environment-variable names to string values; the harness
    applies them (clearing every other :data:`MANAGED_STEER_ENV` key first)
    around the arm's run, then builds :meth:`RuntimeConfig.from_env`. This
    is the indirection that keeps the harness independent of unmerged PRs.
    """

    name: str
    kind: str  # "baseline" | "signal" | "legacy" — the §3 PR-13 arm labels
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    judge_only: bool = False
    description: str = ""


def default_arms() -> list[Arm]:
    """The three §3 PR-13 arms as configurable flag dicts.

    Every arm enables ``signal_telemetry`` explicitly (it is DEFAULT OFF —
    goldfive#456) so the captured-event metrics and the shadow diff have
    data to read. Callers building a *shadow* campaign (§5.4) override
    ``GOLDFIVE_STEER_OBSERVATION_ONLY=1`` on arms B/C so the new decision
    logic runs dry (``SignalDelivered(dry_run=true)``); see
    :func:`shadow_arms`.
    """
    return [
        Arm(
            name="A-baseline-judge-only",
            kind="baseline",
            judge_only=True,
            env={
                "GOLDFIVE_STEER_SIGNAL_TELEMETRY": "1",
                # judge_only exercises no steering authority; observation_only
                # is moot but pinned for a clean, explicit flag dict.
                "GOLDFIVE_STEER_OBSERVATION_ONLY": "1",
            },
            description="judge-only counterfactual baseline (goldfive#446)",
        ),
        Arm(
            name="B-signal-regime",
            kind="signal",
            judge_only=False,
            env={
                "GOLDFIVE_STEER_SIGNAL_TELEMETRY": "1",
                "GOLDFIVE_STEER_OBSERVATION_ONLY": "0",
                "GOLDFIVE_CANCEL_INFLIGHT_SCOPE": "user_and_safety",
                # Pending (no-op until the consuming PR lands):
                "GOLDFIVE_STEER_SIGNAL_CHANNEL": "request_context",  # PR 6
                "GOLDFIVE_PLAN_MODE": "ledger",  # PR 10
            },
            description="new SIGNAL regime (observation_only=False + new-regime flags)",
        ),
        Arm(
            name="C-legacy-ladder",
            kind="legacy",
            judge_only=False,
            env={
                "GOLDFIVE_STEER_SIGNAL_TELEMETRY": "1",
                "GOLDFIVE_STEER_OBSERVATION_ONLY": "0",
                "GOLDFIVE_CANCEL_INFLIGHT_SCOPE": "all",  # PR-1 kill-switch
                "GOLDFIVE_STEER_LEGACY_LADDER": "1",  # PR 7 escape hatch (pending)
            },
            description="legacy ladder regime (GOLDFIVE_STEER_LEGACY_LADDER=1)",
        ),
    ]


def shadow_arms() -> list[Arm]:
    """Arms for the §5.4 shadow campaign: behavior arms forced dry-run.

    Identical to :func:`default_arms` except arms B/C run under
    ``observation_only=1`` so every dispatch records
    ``SignalDelivered(dry_run=true)`` — the new decision logic accrues
    production mileage with zero production authority, and the
    :mod:`bench.shadow_diff` tool diffs legacy-would-do vs. new-would-do on
    the resulting logs.
    """
    arms: list[Arm] = []
    for arm in default_arms():
        env = dict(arm.env)
        if arm.kind in ("signal", "legacy"):
            env["GOLDFIVE_STEER_OBSERVATION_ONLY"] = "1"
        arms.append(dataclasses.replace(arm, env=env))
    return arms


def arm_flag_status(arm: Arm) -> tuple[list[str], list[str]]:
    """Split ``arm.env`` into ``(applied, pending)`` env-var name lists.

    ``applied`` flags are consulted by this build; ``pending`` flags are
    set in the environment but inert until the PR that reads them lands.
    This is the operator-facing transparency that keeps a *pending* flag
    from being silently mistaken for an *applied* one (§5.1).
    """
    applied = sorted(k for k in arm.env if k in KNOWN_STEER_ENV)
    pending = sorted(k for k in arm.env if k not in KNOWN_STEER_ENV)
    return applied, pending


@contextlib.contextmanager
def apply_arm_env(arm: Arm) -> Iterator[None]:
    """Apply ``arm.env`` with every other managed flag cleared; restore on exit.

    The clear step is what prevents cross-arm leakage: an ambient
    ``GOLDFIVE_STEER_OBSERVATION_ONLY`` (or a flag set by a previous arm)
    cannot bleed into an arm that does not declare it. Only
    :data:`MANAGED_STEER_ENV` keys are touched; unrelated environment is
    left untouched and fully restored.
    """
    managed = MANAGED_STEER_ENV | set(arm.env)
    saved = {k: os.environ.get(k) for k in managed}
    try:
        for k in managed:
            os.environ.pop(k, None)
        for k, v in arm.env.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, prior in saved.items():
            if prior is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior


# ---------------------------------------------------------------------------
# Per-arm metrics from captured artifacts + sink events
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ArmMetrics:
    """The §3 PR-13 metric vector for one arm, from artifacts + events.

    Every field is sourced from a captured artifact (``completed_outputs``,
    goal predicates, the :class:`SignalLedger` on ``session.state``) or the
    sink event stream — never from parsing agent prose (§5.6 / project
    scar tissue).
    """

    arm_name: str
    arm_kind: str
    # --- captured-artifact outcome (goldfive#447) ---------------------
    #: ``True`` ONLY when :attr:`goal_grade` is ``MET`` (a grading signal
    #: exists and it passed). ``False`` for UNMET / ABORTED **and** for
    #: UNMEASURED — a run with no goal predicate and no OUTCOME task is NOT a
    #: silent success (§6.4 rule 1). Read :attr:`goal_grade` to tell UNMET
    #: (measured failure) from UNMEASURED (no flip signal on this workload).
    goal_success: bool
    #: One of :data:`GOAL_GRADE_MET` / ``_UNMET`` / ``_ABORTED`` / ``_UNMEASURED``.
    goal_grade: str
    goal_reason: str
    #: Number of goals carrying a :attr:`Goal.success_predicate` (the first
    #: grading signal). ``0`` on the stub workload.
    goal_predicate_count: int
    # --- OUTCOME-task terminality (§6.4 rule 1) ------------------------
    #: Count of :attr:`TaskKind.OUTCOME` tasks in the final plan (the
    #: goal-anchored deliverables; only minted in ``plan_mode=ledger``).
    outcome_tasks_total: int
    #: Terminal-disposition census over the OUTCOME tasks, deterministic key
    #: order: ``completed`` / ``failed`` / ``not_needed`` / ``cancelled`` /
    #: ``non_terminal`` (the #208 carry-forward PENDING/RUNNING/BLOCKED set).
    outcome_terminal: dict[str, int]
    #: Count of :attr:`TaskKind.DISCOVERED` tasks (descriptive-growth records).
    discovered_tasks_total: int
    completed_outputs: int
    turns: int
    tokens: int | None  # best-effort; None when the adapter reports no usage
    aborted: bool
    abort_reason: str
    # --- regime provenance: configured vs. EXERCISED ------------------
    #: The resolved :attr:`SteeringConfig.plan_mode` this build ran under
    #: (``forecast`` / ``ledger``) — what the flag *applied* to.
    plan_mode: str
    #: ``True`` iff the ledger regime was actually EXERCISED on this workload
    #: (plan_mode=ledger AND at least one OUTCOME or DISCOVERED task fired).
    #: A ``plan_mode=ledger`` arm whose workload never mints an OUTCOME /
    #: DISCOVERED task reports ``False`` — configured is not exercised, and a
    #: stub that never fires the ledger path must not read as validating it.
    ledger_exercised: bool
    # --- sink-event telemetry (PR 5) ----------------------------------
    drift_detected: int
    signals_total: int
    signals_real: int  # non-dry_run deliveries == actual interventions
    signals_dry_run: int
    intervention_count: int  # alias for signals_real, named per the brief
    by_channel: dict[str, int]
    outcomes: dict[str, int]
    self_correction_base_rate: float | None  # unaided / (unaided + after_signal)
    post_signal_refire_rate: float | None  # ledger: re-fires / delivered keys
    # --- provenance / degradation ------------------------------------
    applied_flags: list[str]
    pending_flags: list[str]
    jsonl_path: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _drift_kind_name(value: int) -> str:
    """Map a ``DriftDetected.kind`` enum int back to a ``DriftKind`` value.

    The proto stores the UPPER enum name (e.g. ``OFF_TOPIC``); the
    ``SignalDelivered.kind`` string is the ``DriftKind`` *value*
    (lowercase, e.g. ``off_topic``). Normalise to the value form so a
    re-fire correlation on the event stream lines up with deliveries.
    """
    try:
        from goldfive.pb.goldfive.v1 import events_pb2 as pb

        return str(pb.DriftKind.Name(value)).lower()
    except Exception:  # noqa: BLE001 - defensive; metrics are best-effort
        return str(value)


def _ledger_refire_rate(session: Session | None) -> float | None:
    """Post-signal drift re-fire rate from the :class:`SignalLedger`.

    The ledger (``session.state``) is the authoritative, designed home for
    re-fire bookkeeping (PR 5 docstring; PR 8 *gates* on it). Rate =
    (sum of re-fires across delivered keys) / (delivered keys). ``None``
    when no session is available (pure-JSONL callers) or no key carried a
    delivery.
    """
    if session is None:
        return None
    try:
        from goldfive.signal_ledger import SignalLedger

        entries = SignalLedger.for_session(session).entries()
    except Exception:  # noqa: BLE001 - telemetry best-effort
        return None
    delivered = [e for e in entries if e.has_delivery]
    if not delivered:
        return None
    refires = sum(int(getattr(e, "refire_count", 0) or 0) for e in delivered)
    return refires / len(delivered)


#: OUTCOME-task disposition buckets, in a fixed order so the census dict has
#: deterministic key order regardless of task iteration.
_OUTCOME_DISPOSITIONS: tuple[tuple[str, TaskStatus | None], ...] = (
    ("completed", TaskStatus.COMPLETED),
    ("failed", TaskStatus.FAILED),
    ("not_needed", TaskStatus.NOT_NEEDED),
    ("cancelled", TaskStatus.CANCELLED),
    ("non_terminal", None),  # PENDING / RUNNING / BLOCKED (#208 carry-forward)
)


def outcome_terminality(
    session: Session | None,
) -> tuple[int, dict[str, int], int]:
    """Census the OUTCOME-task terminal dispositions on a run's final plan.

    Returns ``(outcome_total, disposition_counts, discovered_total)``:

    * ``outcome_total`` — number of :attr:`TaskKind.OUTCOME` tasks (the
      goal-anchored deliverables; only minted in ``plan_mode=ledger``).
    * ``disposition_counts`` — deterministic-key census over those tasks:
      ``completed`` / ``failed`` / ``not_needed`` / ``cancelled`` /
      ``non_terminal`` (the #208 carry-forward set: uncertain OUTCOME tasks
      legitimately stay PENDING across turn boundaries — deviation 1).
    * ``discovered_total`` — number of :attr:`TaskKind.DISCOVERED` tasks
      (descriptive-growth records); used only to tell whether the ledger
      path fired at all.

    Pure over ``session.plan.tasks`` (a captured artifact); no runtime state
    is mutated. Empty/zero when no session or no plan is available.
    """
    counts: dict[str, int] = {name: 0 for name, _ in _OUTCOME_DISPOSITIONS}
    if session is None or getattr(session, "plan", None) is None:
        return 0, counts, 0
    outcome_total = 0
    discovered_total = 0
    for task in session.plan.tasks:
        if task.kind is TaskKind.DISCOVERED:
            discovered_total += 1
        if task.kind is not TaskKind.OUTCOME:
            continue
        outcome_total += 1
        status = task.status
        bucket = "non_terminal"
        for name, wanted in _OUTCOME_DISPOSITIONS:
            if wanted is not None and status == wanted:
                bucket = name
                break
        counts[bucket] += 1
    return outcome_total, counts, discovered_total


def _grade_goal(
    session: Session | None,
    *,
    aborted: bool,
    abort_reason: str,
    predicate_count: int,
    outcome_total: int,
    outcome_terminal: dict[str, int],
) -> tuple[str, str]:
    """Grade a run on goal predicates + OUTCOME-task terminality (§6.4 rule 1).

    Returns ``(grade, reason)`` where ``grade`` is one of the
    :data:`GOAL_GRADE_MET` vocabulary. The cardinal rule: a run is graded a
    success ONLY when a grading SIGNAL exists and it passes. ``run.success``
    alone is never a flip signal — with no goal ``success_predicate`` and no
    OUTCOME deliverable, the run is :data:`GOAL_GRADE_UNMEASURED`, never
    silently ``MET`` (deviation 1). An abort is a measured failure.
    """
    if session is None:
        return GOAL_GRADE_UNMEASURED, "no session captured"
    if aborted:
        return GOAL_GRADE_ABORTED, abort_reason or "run aborted"
    has_signal = predicate_count > 0 or outcome_total > 0
    if not has_signal:
        return (
            GOAL_GRADE_UNMEASURED,
            "no goal success_predicate and no OUTCOME task — run success is "
            "not gradeable (run.success alone is not a flip signal)",
        )
    if predicate_count > 0:
        reason = evaluate_goal_predicates(session)
        if reason is not None:
            return GOAL_GRADE_UNMET, reason
    if outcome_total > 0:
        failed = outcome_terminal.get("failed", 0)
        if failed:
            return GOAL_GRADE_UNMET, f"{failed} OUTCOME task(s) FAILED"
        cancelled = outcome_terminal.get("cancelled", 0)
        if cancelled:
            return GOAL_GRADE_UNMET, f"{cancelled} OUTCOME task(s) CANCELLED"
        non_terminal = outcome_terminal.get("non_terminal", 0)
        if non_terminal:
            # Conservative for the flip criterion: a deliverable that never
            # reached a successful terminal is not a demonstrated success (it
            # is legitimately uncertain/carried-forward — deviation 1 — but
            # not gradeable as MET).
            return (
                GOAL_GRADE_UNMET,
                f"{non_terminal} OUTCOME task(s) non-terminal "
                "(deliverable not demonstrably met; #208 carry-forward)",
            )
    return GOAL_GRADE_MET, ""


def metrics_from_events(
    events: Sequence[Any],
    *,
    arm: Arm,
    session: Session | None = None,
    tokens: int | None = None,
    plan_mode: str = "forecast",
    jsonl_path: str = "",
) -> ArmMetrics:
    """Reduce a captured event stream (+ optional session) to an :class:`ArmMetrics`.

    Pure over proto ``Event`` messages: it never reaches into goldfive
    internals beyond the documented ``session.completed_outputs`` /
    ``_reasoning_turn`` artifacts and the :class:`SignalLedger`. Works on
    an in-memory sink's ``.events`` or a list parsed back from JSONL.
    """
    drift_detected = 0
    signals_total = 0
    signals_real = 0
    signals_dry_run = 0
    by_channel: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    aborted = False
    abort_reason = ""
    task_terminal = 0

    for evt in events:
        which = evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None
        if which == "drift_detected":
            drift_detected += 1
        elif which == "signal_delivered":
            sd = evt.signal_delivered
            signals_total += 1
            if sd.dry_run:
                signals_dry_run += 1
            else:
                signals_real += 1
            by_channel[sd.channel] = by_channel.get(sd.channel, 0) + 1
        elif which == "signal_outcome":
            oc = evt.signal_outcome.outcome
            outcomes[oc] = outcomes.get(oc, 0) + 1
        elif which == "run_aborted":
            aborted = True
            abort_reason = evt.run_aborted.reason
        elif which in ("task_completed", "task_failed", "task_cancelled"):
            task_terminal += 1

    unaided = outcomes.get("self_corrected_unaided", 0)
    after = outcomes.get("self_corrected_after_signal", 0)
    base_rate = unaided / (unaided + after) if (unaided + after) else None

    completed_outputs = len(getattr(session, "completed_outputs", {}) or {}) if session else 0
    # Turns: prefer the goldfive#441 logical-turn clock; fall back to the
    # count of task-terminal transitions when the adapter never advances it.
    turns = 0
    if session is not None:
        turns = int(getattr(session, "_reasoning_turn", 0) or 0)
    turns = max(turns, task_terminal)

    # OUTCOME-task terminality (§6.4 rule 1) + ledger-exercised provenance.
    outcome_total, outcome_terminal, discovered_total = outcome_terminality(session)
    predicate_count = (
        sum(1 for g in session.goals if g.success_predicate is not None)
        if session is not None
        else 0
    )
    goal_grade, goal_reason = _grade_goal(
        session,
        aborted=aborted,
        abort_reason=abort_reason,
        predicate_count=predicate_count,
        outcome_total=outcome_total,
        outcome_terminal=outcome_terminal,
    )
    goal_success = goal_grade == GOAL_GRADE_MET
    ledger_exercised = plan_mode == "ledger" and (
        outcome_total > 0 or discovered_total > 0
    )

    applied, pending = arm_flag_status(arm)
    return ArmMetrics(
        arm_name=arm.name,
        arm_kind=arm.kind,
        goal_success=goal_success,
        goal_grade=goal_grade,
        goal_reason=goal_reason,
        goal_predicate_count=predicate_count,
        outcome_tasks_total=outcome_total,
        outcome_terminal=outcome_terminal,
        discovered_tasks_total=discovered_total,
        completed_outputs=completed_outputs,
        turns=turns,
        tokens=tokens,
        aborted=aborted,
        abort_reason=abort_reason,
        plan_mode=plan_mode,
        ledger_exercised=ledger_exercised,
        drift_detected=drift_detected,
        signals_total=signals_total,
        signals_real=signals_real,
        signals_dry_run=signals_dry_run,
        intervention_count=signals_real,
        by_channel=by_channel,
        outcomes=outcomes,
        self_correction_base_rate=base_rate,
        post_signal_refire_rate=_ledger_refire_rate(session),
        applied_flags=applied,
        pending_flags=pending,
        jsonl_path=str(jsonl_path),
    )


# ---------------------------------------------------------------------------
# Scenario / workload + arm runner
# ---------------------------------------------------------------------------

#: A driver runs one arm's workload against the supplied sinks under the
#: supplied resolved config and returns the run outcome. Keeping the driver
#: pluggable is what lets 13b swap in a real coordinator+AgentTool tree and
#: a live model while reusing this harness's flag-dict + metrics machinery.
Driver = Callable[..., Awaitable[ExecutionOutcome]]


@dataclasses.dataclass
class Scenario:
    """A named workload plus the driver that runs it under an arm."""

    name: str
    driver: Driver
    #: Optional best-effort token accounting; 13b wires the adapter's usage.
    extract_tokens: Callable[[Session], int | None] | None = None


async def run_arm(
    arm: Arm,
    scenario: Scenario,
    *,
    jsonl_dir: str | Path,
) -> ArmMetrics:
    """Run one arm of ``scenario`` and return its :class:`ArmMetrics`.

    Applies the arm's flag dict, builds the resolved runtime, runs the
    driver against a JSONL artifact sink (durable, the 13b/§5.4 input) and
    an in-memory sink (fast metric reduction), then reduces to metrics from
    the captured artifacts + events.
    """
    jsonl_dir = Path(jsonl_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{arm.name}.jsonl"

    with apply_arm_env(arm):
        runtime = RuntimeConfig.from_env()
        jsonl_sink = JSONLPersistenceSink(jsonl_path, mode="write")
        mem_sink = InMemorySink()
        outcome = await scenario.driver(
            sinks=[jsonl_sink, mem_sink], runtime=runtime, arm=arm
        )
        await jsonl_sink.close()

    tokens = None
    if scenario.extract_tokens is not None and outcome.session is not None:
        with contextlib.suppress(Exception):
            tokens = scenario.extract_tokens(outcome.session)

    return metrics_from_events(
        mem_sink.events,
        arm=arm,
        session=outcome.session,
        tokens=tokens,
        plan_mode=str(getattr(runtime.steering, "plan_mode", "forecast") or "forecast"),
        jsonl_path=str(jsonl_path),
    )


async def run_arms(
    arms: Sequence[Arm],
    scenario: Scenario,
    *,
    jsonl_dir: str | Path,
) -> list[ArmMetrics]:
    """Run every arm of ``scenario`` sequentially (isolated env per arm)."""
    results: list[ArmMetrics] = []
    for arm in arms:
        results.append(await run_arm(arm, scenario, jsonl_dir=jsonl_dir))
    return results


# ---------------------------------------------------------------------------
# A concrete, deterministic scenario for the smoke test (stub model)
# ---------------------------------------------------------------------------


def _linear_plan(n: int, *, goal_id: str, run_id: str = "") -> Plan:
    tasks = [
        Task(
            id=f"t{i:03d}",
            title=f"Task {i}",
            description=f"Bench task #{i}",
            assignee_agent_id="bench-agent",
        )
        for i in range(n)
    ]
    edges: list[Any] = []
    from goldfive import TaskEdge

    for i in range(n - 1):
        edges.append(TaskEdge(from_task_id=f"t{i:03d}", to_task_id=f"t{i + 1:03d}"))
    return Plan(
        id="bench-arm-plan",
        run_id=run_id,
        goal_ids=[goal_id],
        tasks=tasks,
        edges=edges,
        summary=f"Linear {n}-task bench plan",
    )


async def _noop_agent(
    task: Task, session: Session, tools: Any
) -> InvocationResult:
    _ = (session, tools)
    return InvocationResult(task_id=task.id, text=f"native ran: {task.title}")


async def _always_progressing_call_llm(system: str, user: str, model: str) -> str:
    """Stub ``call_llm`` that keeps any armed judge from drifting/exhausting.

    Returns a permissive ``progressing`` verdict for every judge call so a
    healthy run stays healthy regardless of how many times a judge fires —
    drift in the smoke test is injected deterministically by
    :func:`inject_demo_signals`, not produced by this stub.
    """
    _ = (system, user, model)
    return json.dumps({"progressing": True})


async def inject_demo_signals(
    *,
    runtime: RuntimeConfig,
    session: Session,
    sinks: Sequence[Any],
) -> None:
    """Drive a deterministic drift lifecycle through the REAL dispatch path.

    Builds a fresh :class:`DefaultSteerer` from the arm's resolved steering
    config plus a throwaway one-task *drift session* (an OPEN task, so the
    task-terminal chokepoint actually resolves the ledger keys instead of
    no-opping on the already-completed run plan), and exercises the
    production ``_dispatch_nudge`` / ``_emit_drift_detected`` / task-terminal
    / finalize chokepoints so the telemetry pipeline emits genuine
    ``SignalDelivered`` / ``SignalOutcome`` events (not hand-built
    envelopes). The resulting :class:`SignalLedger` is then merged onto the
    run's ``session.state`` so :func:`metrics_from_events` reads
    completed-output and signal metrics off one consistent session.

    The drift kinds/severities are chosen so the recorded
    ``decision.would_cancel_inflight`` *diverges by arm*: a CRITICAL
    goldfive-authored ``OFF_TOPIC`` drift cancels in-flight work under
    ``cancel_inflight_scope="all"`` (legacy arm C) but not under
    ``"user_and_safety"`` (signal arm B) — the PR-1 authority split, which
    is merged and therefore a *real* divergence the shadow diff surfaces
    today, before PR 7's ladder restructure lands. The OFF_TOPIC key also
    re-fires once after delivery so ``post_signal_refire_rate`` is exercised.

    This stands in for organically-produced drift so 13a's tooling is
    validated end-to-end; 13b drops it and lets a live model drift.
    """
    from goldfive import TaskStatus
    from goldfive.signal_ledger import KEY_SIGNAL_LEDGER

    dtask = Task(
        id="dt0",
        title="Demo drift task",
        description="Open task carrying the injected drift lifecycle",
        assignee_agent_id="bench-agent",
        status=TaskStatus.RUNNING,
    )
    drift_session = Session(
        run_id=session.run_id,
        goals=list(session.goals),
        plan=Plan(
            id="demo-drift-plan",
            run_id=session.run_id,
            goal_ids=[g.id for g in session.goals],
            tasks=[dtask],
            edges=[],
        ),
        current_task_id="dt0",
    )

    steerer = DefaultSteerer(steering_config=runtime.steering)
    steerer.bind(sinks=list(sinks), planner=StaticPlanner(drift_session.plan))

    def _off_topic() -> DriftEvent:
        return DriftEvent(
            kind=DriftKind.OFF_TOPIC,
            severity=DriftSeverity.CRITICAL,
            detail="trajectory wandered off the user goal",
            current_task_id="dt0",
            current_agent_id="bench-agent",
            authored_by="goldfive",
        )

    # First delivery on the OFF_TOPIC/dt0 key (would_cancel_inflight diverges
    # by cancel-scope regime).
    await steerer.drift._dispatch_nudge(_off_topic(), drift_session)
    # A distinct OFF_TOPIC fire AFTER the delivery → a post-signal re-fire.
    await steerer.drift._emit_drift_detected(drift_session, _off_topic())
    # A second, distinct loop-shaped key so the stream carries >1 drift.
    looping = DriftEvent(
        kind=DriftKind.LOOPING_TOOL_CALL,
        severity=DriftSeverity.WARNING,
        detail="search_web called 5x with identical args",
        current_task_id="dt0",
        current_agent_id="bench-agent",
        authored_by="goldfive",
    )
    await steerer.drift._dispatch_nudge(looping, drift_session)
    # AGENCY-PRESERVATION.md PR 8: self_corrected_after_signal attribution is
    # now keyed on VISIBILITY (the ObserverNoteQueue's rendered set), not on
    # dispatch. Under request_context with active steering the synthetic driver
    # stands in for a delivery surface (before_model / boundary) actually
    # SHOWING the queued notes — render them so the resolution attributes
    # after_signal. Under observation_only (the shadow arms) we deliberately do
    # NOT render: the note is never really shown, so the key resolves
    # self_corrected_unaided (the §2 base-rate the shadow campaign measures).
    if (
        runtime.steering.signal_channel == "request_context"
        and not runtime.steering.observation_only
    ):
        from goldfive.observer_note_queue import ObserverNoteQueue

        rq = ObserverNoteQueue.for_session(drift_session)
        render_turn = int(getattr(drift_session, "_reasoning_turn", 0) or 0)
        for note in rq.pending():
            rq.mark_delivered(
                note.note_id,
                channel="request_context",
                turn=render_turn,
                surface="bench_render",
            )
    # Resolve the bound task → SignalOutcome for both delivered keys
    # (self_corrected_unaided under observation_only / unrendered,
    # after_signal once rendered).
    await steerer.tasks.mark_task_completed("dt0", session=drift_session, summary="done")
    # Finalize any still-open delivered keys at the run boundary (idempotent).
    await steerer.drift.finalize_signal_ledger(drift_session)

    # Merge the drift session's signal ledger onto the run session so the
    # metric reducer reads one consistent session.
    from goldfive import state_store as _ostate

    ledger = _ostate.read(drift_session.state, KEY_SIGNAL_LEDGER, {})
    if ledger:
        _ostate.write(session.state, KEY_SIGNAL_LEDGER, dict(ledger))


def make_linear_run_driver(
    *,
    num_tasks: int = 3,
    inject_signals: bool = True,
    user_goal: str = "Summarise the quarterly report",
) -> Driver:
    """Build a deterministic linear-plan driver for the smoke test / demo.

    All arms share this workload so the comparison is apples-to-apples:
    a one-goal :class:`LiteralGoalDeriver` (no goal-derive LLM call), an
    explicit :class:`StaticPlanner` (so judge-only arm A runs the SAME
    tasks, not its 1-task native-run framing plan), a no-op agent, and a
    permissive stub ``call_llm`` for any armed judge. With
    ``inject_signals`` the driver appends :func:`inject_demo_signals` so the
    telemetry path is exercised.
    """
    goal = Goal(id="bench-goal", summary=user_goal)
    plan = _linear_plan(num_tasks, goal_id=goal.id)

    async def driver(*, sinks: Sequence[Any], runtime: RuntimeConfig, arm: Arm) -> ExecutionOutcome:
        runner = wrap(
            CallableAdapter(_noop_agent, available_agents=["bench-agent"]),
            judge_only=arm.judge_only,
            planner=StaticPlanner(plan),
            goal_deriver=LiteralGoalDeriver(),
            call_llm=_always_progressing_call_llm,
            model="bench-stub-model",
            runtime=runtime,
            sinks=list(sinks),
        )
        outcome = await runner.run([goal])
        if inject_signals:
            await inject_demo_signals(
                runtime=runtime, session=outcome.session, sinks=sinks
            )
        await runner.close()
        return outcome

    return driver
