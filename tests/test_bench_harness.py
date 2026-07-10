"""Smoke test for the PR 13a three-arm bench harness + shadow-diff tool.

Integration-not-unit (AGENCY-PRESERVATION.md §5.6): these assert the
telemetry actually flows *end to end* through the harness — a tool that
parses zero events because ``signal_telemetry`` was off (its default,
goldfive#456) must be a LOUD error, not an empty report — and that the
§5.4 shadow diff surfaces the *real* PR-1 cancel-authority divergence
between the legacy and new arms on the same workload.

The workload is deterministic (stub ``call_llm`` + an injected drift
lifecycle through the real ``DefaultSteerer`` dispatch path); running the
bench against a live model is task 13b, deliberately out of scope here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev`/`proto` extra)",
)

# The bench harness lives in the (non-installed) ``bench`` package at the repo
# root; make it importable regardless of pytest's import mode.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.harness import (  # noqa: E402
    KNOWN_STEER_ENV,
    Arm,
    Scenario,
    apply_arm_env,
    arm_flag_status,
    default_arms,
    make_linear_run_driver,
    metrics_from_events,
    outcome_terminality,
    run_arm,
    run_arms,
    shadow_arms,
)
from bench.shadow_diff import (  # noqa: E402
    ShadowDiffError,
    SignalRecord,
    diff_two_logs,
    load_signals,
    render_single_log_text,
    render_two_log_text,
    single_log_report,
)
from goldfive import Goal, Plan, Session, Task  # noqa: E402
from goldfive.types import TaskKind, TaskStatus  # noqa: E402


def _scenario(tasks: int = 3, inject: bool = True) -> Scenario:
    return Scenario(
        name="smoke",
        driver=make_linear_run_driver(num_tasks=tasks, inject_signals=inject),
    )


# ---------------------------------------------------------------------------
# 1. Three-arm harness runs and telemetry flows end to end
# ---------------------------------------------------------------------------


async def test_three_arm_harness_runs_and_emits_telemetry(tmp_path: Path) -> None:
    results = await run_arms(default_arms(), _scenario(), jsonl_dir=tmp_path)
    assert [m.arm_kind for m in results] == ["baseline", "signal", "legacy"]

    for m in results:
        # §6.4: the stub workload defines NO goal predicate and NO OUTCOME
        # task, so goal grading is UNMEASURED (not silently True) — the exact
        # gap that blocks a flip decision on this workload. run.success alone
        # is not a flip signal (deviation 1).
        assert m.goal_grade == "unmeasured", m.goal_reason
        assert m.goal_success is False
        assert m.goal_predicate_count == 0
        assert m.outcome_tasks_total == 0
        assert m.outcome_terminal == {
            "completed": 0,
            "failed": 0,
            "not_needed": 0,
            "cancelled": 0,
            "non_terminal": 0,
        }
        # The captured-artifact outcome is otherwise healthy.
        assert m.completed_outputs == 3
        assert m.turns >= 1
        # Telemetry flowed: the JSONL artifact exists and carries signals.
        assert Path(m.jsonl_path).exists()
        assert m.signals_total > 0, f"{m.arm_name}: 0 signals — telemetry not wired"

    by_kind = {m.arm_kind: m for m in results}
    # plan_mode=ledger is CONFIGURED on arm B (a KNOWN/applied flag), but the
    # StaticPlanner stub mints only FORECAST tasks, so the ledger regime is
    # NOT exercised. Configured-and-applied must not read as validated: a stub
    # that never fires an OUTCOME/DISCOVERED path does not validate ledger.
    assert by_kind["signal"].plan_mode == "ledger"
    assert by_kind["signal"].ledger_exercised is False
    assert by_kind["signal"].discovered_tasks_total == 0
    assert by_kind["baseline"].plan_mode == "forecast"
    assert by_kind["baseline"].ledger_exercised is False
    assert by_kind["legacy"].ledger_exercised is False
    # The baseline runs observation-only: every signal is dry-run, so the
    # self-correction *base rate* is fully unaided (the §2 counterfactual).
    base = by_kind["baseline"]
    assert base.signals_real == 0
    assert base.signals_dry_run == base.signals_total
    assert base.intervention_count == 0
    assert base.outcomes.get("self_corrected_unaided", 0) > 0
    assert base.self_correction_base_rate == 1.0

    # The behavior arms intervene for real (active steering).
    for kind in ("signal", "legacy"):
        m = by_kind[kind]
        assert m.signals_real > 0
        assert m.intervention_count == m.signals_real
        assert m.outcomes.get("self_corrected_after_signal", 0) > 0
        assert m.self_correction_base_rate == 0.0

    # post-signal re-fire rate is read from the SignalLedger (a captured
    # artifact), exercised by the injected re-fire.
    assert all(m.post_signal_refire_rate == 0.5 for m in results)


# ---------------------------------------------------------------------------
# 2. signal_telemetry OFF (the default) → a LOUD error, not an empty report
# ---------------------------------------------------------------------------


async def test_telemetry_off_is_a_loud_error(tmp_path: Path) -> None:
    off = Arm(
        name="telemetry-off",
        kind="signal",
        env={
            "GOLDFIVE_STEER_SIGNAL_TELEMETRY": "0",  # explicit OFF
            "GOLDFIVE_STEER_OBSERVATION_ONLY": "0",
        },
    )
    metrics = await run_arm(off, _scenario(), jsonl_dir=tmp_path)
    # With telemetry off, the dispatch path emits no SignalDelivered events.
    assert metrics.signals_total == 0

    # The shadow-diff loader refuses to render an empty report from a
    # zero-signal log — it raises so "the flag was off" can never masquerade
    # as "no divergence".
    with pytest.raises(ShadowDiffError) as exc:
        load_signals(metrics.jsonl_path)
    assert "signal_telemetry" in str(exc.value)

    # ...unless the caller explicitly opts into the empty case.
    assert load_signals(metrics.jsonl_path, allow_empty=True) == []


# ---------------------------------------------------------------------------
# 3. The shadow diff surfaces the real PR-1 cancel-authority divergence
# ---------------------------------------------------------------------------


async def test_shadow_diff_surfaces_cancel_authority_divergence(tmp_path: Path) -> None:
    results = await run_arms(shadow_arms(), _scenario(), jsonl_dir=tmp_path)
    by_kind = {m.arm_kind: m for m in results}

    # Shadow mode forces the behavior arms observation-only → dry-run signals.
    assert by_kind["signal"].signals_dry_run == by_kind["signal"].signals_total
    assert by_kind["legacy"].signals_dry_run == by_kind["legacy"].signals_total

    legacy = load_signals(by_kind["legacy"].jsonl_path)
    new = load_signals(by_kind["signal"].jsonl_path)
    report = diff_two_logs(legacy, new, legacy_path="legacy", new_path="new")

    # The same OFF_TOPIC/CRITICAL drift diverges on would_cancel_inflight:
    # legacy (cancel scope=all) would cancel in-flight work; new
    # (user_and_safety) would not — the merged PR-1 authority split.
    diverged = report.diverged_keys
    assert diverged, "expected the cancel-authority divergence to surface"
    off_topic = [k for k in diverged if k.kind == "off_topic"]
    assert off_topic, "off_topic key should diverge"
    kd = off_topic[0]
    assert kd.present_in == "both"
    assert "would_cancel_inflight" in kd.diverged_fields
    assert kd.legacy["would_cancel_inflight"] is True
    assert kd.new["would_cancel_inflight"] is False

    # The looping WARNING key never cancels under either regime, so it is NOT
    # a decision divergence. Since PR 6 it DOES ride a different transport (the
    # signal arm rides ``request_context``, the legacy arm ``nudge_replay``) —
    # but channel/channel_action are per-regime transport identity, excluded
    # from the divergence-driving set, so the key reports as transport-only,
    # not diverged.
    looping = [k for k in report.keys if k.kind == "looping_tool_call"]
    assert looping
    assert "would_cancel_inflight" not in looping[0].diverged_fields
    assert not looping[0].diverged
    assert looping[0].transport_only
    assert "channel" in looping[0].transport_fields

    # The rendered report names the divergence (the reviewed §5.4 artifact).
    text = render_two_log_text(report)
    assert "would_cancel_inflight" in text
    assert "VERDICT" in text


# ---------------------------------------------------------------------------
# 4. Single-log census mode
# ---------------------------------------------------------------------------


async def test_single_log_census(tmp_path: Path) -> None:
    results = await run_arms(shadow_arms(), _scenario(), jsonl_dir=tmp_path)
    legacy = {m.arm_kind: m for m in results}["legacy"]
    records = load_signals(legacy.jsonl_path)
    census = single_log_report(records)
    assert census["deliveries"] == len(records) > 0
    assert census["dry_run"] == census["deliveries"]  # shadow mode
    assert set(census["by_kind"]) == {"off_topic", "looping_tool_call"}
    # In the legacy log the OFF_TOPIC delivery would cancel in-flight work,
    # so its single-event legacy-vs-new derivation diverges.
    assert any(d["kind"] == "off_topic" for d in census["diverging_events"])


# ---------------------------------------------------------------------------
# 4b. Transport (channel/channel_action) is excluded from decision divergence
# ---------------------------------------------------------------------------


def _rec(seq: int, *, channel: str, channel_action: str, **decision: object) -> SignalRecord:
    dec = {"channel_action": channel_action, **decision}
    return SignalRecord(
        sequence=seq,
        drift_id=f"d{seq}",
        kind="off_topic",
        task_id="t0",
        channel=channel,
        severity="warning",
        turn=1,
        dry_run=True,
        ladder_level=str(decision.get("ladder_level", "nudge")),
        note_text="",
        decision=dec,
    )


def test_channel_difference_alone_is_transport_not_divergence() -> None:
    """A key differing ONLY in channel/channel_action is transport-only."""
    legacy = [
        _rec(1, channel="nudge_replay", channel_action="queued", would_cancel_inflight=False)
    ]
    new = [
        _rec(1, channel="request_context", channel_action="enqueued", would_cancel_inflight=False)
    ]
    report = diff_two_logs(legacy, new)
    kd = report.keys[0]
    assert not kd.diverged
    assert kd.transport_only
    assert set(kd.transport_fields) == {"channel", "channel_action"}
    assert report.diverged_keys == []
    assert len(report.transport_only_keys) == 1
    # The verdict reflects "no decision divergence" despite the transport swap.
    text = render_two_log_text(report)
    assert "no decision divergence" in text
    assert "transport-only (channel): 1" in text


def test_real_decision_divergence_survives_transport_difference() -> None:
    """A genuine decision diff (would_cancel_inflight) still diverges even when
    the channel also differs — transport never masks a real divergence."""
    legacy = [
        _rec(1, channel="nudge_replay", channel_action="queued", would_cancel_inflight=True)
    ]
    new = [
        _rec(1, channel="request_context", channel_action="enqueued", would_cancel_inflight=False)
    ]
    report = diff_two_logs(legacy, new)
    kd = report.keys[0]
    assert kd.diverged
    assert "would_cancel_inflight" in kd.diverged_fields
    # channel still differs, recorded as informational transport (not masking).
    assert "channel" in kd.transport_fields
    assert not kd.transport_only  # it IS a real divergence, not transport-only


# ---------------------------------------------------------------------------
# 5. Graceful degradation: pending vs. applied flags
# ---------------------------------------------------------------------------


def test_arm_flag_status_reports_promoted_flags_as_applied() -> None:
    arms = {a.kind: a for a in default_arms()}
    applied, pending = arm_flag_status(arms["signal"])
    # The promotion contract: the PR that teaches ``SteeringConfig.from_env``
    # about a flag moves it from _PENDING_STEER_ENV into KNOWN_STEER_ENV, so it
    # reports APPLIED. As of PR 7 + PR 9 every roadmap flag the bench carries is
    # promoted: GOLDFIVE_STEER_SIGNAL_CHANNEL (PR 6), GOLDFIVE_PLAN_MODE
    # (PR 10), GOLDFIVE_STEER_LEGACY_LADDER (PR 7), and
    # GOLDFIVE_STEER_PIN_ASSIGNED_TASK (PR 9).
    assert "GOLDFIVE_STEER_SIGNAL_CHANNEL" in applied
    assert "GOLDFIVE_STEER_SIGNAL_CHANNEL" not in pending
    assert "GOLDFIVE_PLAN_MODE" in applied
    assert "GOLDFIVE_PLAN_MODE" not in pending
    # PR 9 promotion: ``GOLDFIVE_STEER_PIN_ASSIGNED_TASK`` is read by
    # ``SteeringConfig.from_env``, so it is registered as APPLIED. The
    # signal arm does NOT set it — the diet keeps the pin OFF by default
    # under request_context — so its promotion is asserted via the
    # registry here + the synthetic-arm test below, not the signal arm.
    assert "GOLDFIVE_STEER_PIN_ASSIGNED_TASK" in KNOWN_STEER_ENV
    # PR 7 promotion: the legacy arm's escape hatch is now APPLIED (read into
    # from_env); with PR 9 also promoted, nothing the bench carries is pending.
    legacy_applied, legacy_pending = arm_flag_status(arms["legacy"])
    assert "GOLDFIVE_STEER_LEGACY_LADDER" in legacy_applied
    assert legacy_pending == []
    # The flags this build DOES consult are reported as applied.
    assert "GOLDFIVE_STEER_SIGNAL_TELEMETRY" in applied
    assert set(applied) <= KNOWN_STEER_ENV


def test_pin_assigned_task_env_promoted_to_applied() -> None:
    """PR 9 same-PR contract: an arm setting GOLDFIVE_STEER_PIN_ASSIGNED_TASK
    reports it APPLIED (read by from_env), never pending."""
    arm = Arm(
        name="pin-escape-hatch",
        kind="signal",
        env={"GOLDFIVE_STEER_PIN_ASSIGNED_TASK": "1"},
    )
    applied, pending = arm_flag_status(arm)
    assert "GOLDFIVE_STEER_PIN_ASSIGNED_TASK" in applied
    assert "GOLDFIVE_STEER_PIN_ASSIGNED_TASK" not in pending


async def test_unknown_flag_degrades_gracefully(tmp_path: Path) -> None:
    """An arm carrying a flag no build will ever read still runs (it no-ops)."""
    arm = Arm(
        name="unknown-flag",
        kind="signal",
        env={
            "GOLDFIVE_STEER_SIGNAL_TELEMETRY": "1",
            "GOLDFIVE_STEER_OBSERVATION_ONLY": "0",
            "GOLDFIVE_TOTALLY_MADE_UP_FLAG": "1",
        },
    )
    metrics = await run_arm(arm, _scenario(), jsonl_dir=tmp_path)
    # The arm still runs and emits telemetry (the made-up flag no-ops); goal
    # grading is UNMEASURED on the stub workload (no predicate / OUTCOME).
    assert metrics.goal_grade == "unmeasured"
    assert metrics.signals_total > 0
    _, pending = arm_flag_status(arm)
    assert "GOLDFIVE_TOTALLY_MADE_UP_FLAG" in pending


# ---------------------------------------------------------------------------
# 6. apply_arm_env isolates managed flags (no cross-arm leakage)
# ---------------------------------------------------------------------------


def test_apply_arm_env_clears_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ambient managed flag must NOT leak into an arm that does not set it.
    monkeypatch.setenv("GOLDFIVE_STEER_OBSERVATION_ONLY", "1")
    arm = Arm(name="x", kind="signal", env={"GOLDFIVE_STEER_SIGNAL_TELEMETRY": "1"})
    import os

    with apply_arm_env(arm):
        assert os.environ["GOLDFIVE_STEER_SIGNAL_TELEMETRY"] == "1"
        # cleared inside the block (declared on neither side managed-cleared):
        assert "GOLDFIVE_STEER_OBSERVATION_ONLY" not in os.environ
    # restored afterwards:
    assert os.environ["GOLDFIVE_STEER_OBSERVATION_ONLY"] == "1"
    assert "GOLDFIVE_STEER_SIGNAL_TELEMETRY" not in os.environ


# ---------------------------------------------------------------------------
# 7. Goal grading requires a signal; OUTCOME-terminality census (§6.4 rule 1)
# ---------------------------------------------------------------------------


def _outcome_session(
    statuses: list[TaskStatus],
    *,
    predicate=None,
    kind: TaskKind = TaskKind.OUTCOME,
) -> Session:
    """A synthetic run session whose plan carries tasks of the given kind."""
    tasks = [
        Task(id=f"o{i}", title=f"deliverable {i}", kind=kind, status=st)
        for i, st in enumerate(statuses)
    ]
    plan = Plan(id="p", run_id="r", goal_ids=["g"], tasks=tasks, edges=[])
    goal = Goal(id="g", summary="deliver the thing", success_predicate=predicate)
    return Session(run_id="r", goals=[goal], plan=plan)


class _FakeEvent:
    """Minimal duck-typed proto event for the metric reducer (WhichOneof)."""

    def __init__(self, which: str, payload: object) -> None:
        self._which = which
        setattr(self, which, payload)

    def WhichOneof(self, _field: str) -> str:  # noqa: N802 - proto API shape
        return self._which


class _Aborted:
    def __init__(self, reason: str) -> None:
        self.reason = reason


_ARM = Arm(name="unit", kind="signal")


def test_outcome_terminality_census_is_deterministic() -> None:
    session = _outcome_session(
        [
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.NOT_NEEDED,
            TaskStatus.PENDING,
            TaskStatus.CANCELLED,
        ]
    )
    total, census, discovered = outcome_terminality(session)
    assert total == 6
    assert discovered == 0
    # Fixed key order, exact counts.
    assert list(census.keys()) == [
        "completed",
        "failed",
        "not_needed",
        "cancelled",
        "non_terminal",
    ]
    assert census == {
        "completed": 2,
        "failed": 1,
        "not_needed": 1,
        "cancelled": 1,
        "non_terminal": 1,  # PENDING (the #208 carry-forward set)
    }
    # None session / no plan → all-zero census, not a crash.
    assert outcome_terminality(None) == (
        0,
        {k: 0 for k in census},
        0,
    )


def test_outcome_all_terminal_success_grades_met() -> None:
    session = _outcome_session([TaskStatus.COMPLETED, TaskStatus.NOT_NEEDED])
    m = metrics_from_events([], arm=_ARM, session=session, plan_mode="ledger")
    assert m.goal_grade == "met"
    assert m.goal_success is True
    assert m.outcome_tasks_total == 2
    # plan_mode=ledger AND OUTCOME tasks fired → the ledger regime was exercised.
    assert m.ledger_exercised is True


def test_outcome_failed_grades_unmet() -> None:
    session = _outcome_session([TaskStatus.COMPLETED, TaskStatus.FAILED])
    m = metrics_from_events([], arm=_ARM, session=session, plan_mode="ledger")
    assert m.goal_grade == "unmet"
    assert m.goal_success is False
    assert "FAILED" in m.goal_reason


def test_outcome_non_terminal_grades_unmet_not_silently_met() -> None:
    # A deliverable still PENDING (uncertain, #208 carry-forward) is NOT a
    # demonstrated success — it grades UNMET for the flip criterion, never MET.
    session = _outcome_session([TaskStatus.COMPLETED, TaskStatus.PENDING])
    m = metrics_from_events([], arm=_ARM, session=session, plan_mode="ledger")
    assert m.goal_grade == "unmet"
    assert "non-terminal" in m.goal_reason


def test_goal_predicate_gate_precedes_outcome() -> None:
    # A failing predicate fails the run even when the OUTCOME tasks completed.
    session = _outcome_session(
        [TaskStatus.COMPLETED], predicate=lambda _s: False
    )
    m = metrics_from_events([], arm=_ARM, session=session, plan_mode="ledger")
    assert m.goal_grade == "unmet"
    assert m.goal_predicate_count == 1
    # A passing predicate + completed OUTCOME → met.
    ok = _outcome_session([TaskStatus.COMPLETED], predicate=lambda _s: True)
    m2 = metrics_from_events([], arm=_ARM, session=ok, plan_mode="ledger")
    assert m2.goal_grade == "met"
    assert m2.goal_success is True


def test_no_signal_is_unmeasured_not_true() -> None:
    # FORECAST-only plan, no predicate → no grading signal at all. The run is
    # UNMEASURED, and goal_success is False (never a silent True).
    session = _outcome_session(
        [TaskStatus.COMPLETED, TaskStatus.COMPLETED], kind=TaskKind.FORECAST
    )
    m = metrics_from_events([], arm=_ARM, session=session, plan_mode="forecast")
    assert m.goal_grade == "unmeasured"
    assert m.goal_success is False
    assert m.outcome_tasks_total == 0
    assert m.ledger_exercised is False


def test_abort_is_a_measured_failure() -> None:
    session = _outcome_session([TaskStatus.COMPLETED])
    events = [_FakeEvent("run_aborted", _Aborted("runaway delegation"))]
    m = metrics_from_events(events, arm=_ARM, session=session, plan_mode="ledger")
    assert m.aborted is True
    assert m.goal_grade == "aborted"
    assert m.goal_success is False
    assert "runaway delegation" in m.goal_reason


def test_ledger_exercised_needs_ledger_mode_and_a_ledger_task() -> None:
    outcome_session = _outcome_session([TaskStatus.COMPLETED])
    # OUTCOME tasks present but plan_mode=forecast → not exercised.
    m_forecast = metrics_from_events(
        [], arm=_ARM, session=outcome_session, plan_mode="forecast"
    )
    assert m_forecast.ledger_exercised is False
    # DISCOVERED task in ledger mode also counts as exercised.
    disc = _outcome_session([TaskStatus.COMPLETED], kind=TaskKind.DISCOVERED)
    m_disc = metrics_from_events([], arm=_ARM, session=disc, plan_mode="ledger")
    assert m_disc.discovered_tasks_total == 1
    assert m_disc.outcome_tasks_total == 0
    assert m_disc.ledger_exercised is True


# ---------------------------------------------------------------------------
# 8. Shadow-diff un-joinable cross-regime keys (§6.4 flip-target comparison)
# ---------------------------------------------------------------------------


def _sig(seq: int, *, task_id: str, kind: str = "off_topic", channel: str = "nudge_replay",
         **decision: object) -> SignalRecord:
    return SignalRecord(
        sequence=seq,
        drift_id=f"d{seq}",
        kind=kind,
        task_id=task_id,
        channel=channel,
        severity="critical",
        turn=1,
        dry_run=True,
        ladder_level=str(decision.get("ladder_level", "nudge")),
        note_text="",
        decision=dict(decision),
    )


def test_disjoint_task_id_namespaces_are_unjoinable_not_divergence() -> None:
    # The flip-target case: the legacy log fires OFF_TOPIC on forecast id
    # "t000"; the new (ledger) log fires the same kind on OUTCOME id "oc-0".
    # The (kind, task_id, occurrence) key cannot align — this is a join
    # artifact, NOT a "regime stayed silent" divergence.
    legacy = [_sig(1, task_id="t000", would_cancel_inflight=True)]
    new = [_sig(1, task_id="oc-0", channel="request_context", would_cancel_inflight=False)]
    report = diff_two_logs(legacy, new)
    assert len(report.unjoinable_keys) == 2
    assert report.diverged_keys == []  # excluded from the verdict
    assert report.legacy_only == []  # not counted as genuine silence
    assert report.new_only == []
    text = render_two_log_text(report)
    assert "UN-JOINABLE" in text
    assert "no decision divergence" in text
    assert report.to_dict()["unjoinable_keys"] == 2


def test_genuine_one_sided_silence_is_not_flagged_unjoinable() -> None:
    # The new regime never fires LOOPING at all → a genuine new-silence, a real
    # legacy_only divergence (not a namespace artifact).
    legacy = [_sig(1, task_id="t0", kind="looping_tool_call", would_cancel_inflight=False)]
    new = [_sig(1, task_id="t0", kind="off_topic",
                channel="request_context", would_cancel_inflight=False)]
    report = diff_two_logs(legacy, new)
    looping = [k for k in report.keys if k.kind == "looping_tool_call"][0]
    assert looping.present_in == "legacy_only"
    assert looping.unjoinable is False
    assert looping.diverged is True
    assert looping in report.legacy_only


# ---------------------------------------------------------------------------
# 9. Single-log census blind spot on new-regime logs (defect 4)
# ---------------------------------------------------------------------------


def test_single_log_new_regime_is_blind_to_cancel_divergence() -> None:
    # A new-regime (request_context) log. would_cancel_inflight is the NEW
    # regime's own (narrower) verdict, so the single-log census CANNOT reveal
    # the legacy cancel divergence — it must flag the blind spot, not silently
    # report "no divergence".
    records = [
        _sig(1, task_id="t0", channel="request_context", would_cancel_inflight=False),
        _sig(2, task_id="t1", channel="request_context", would_cancel_inflight=False),
    ]
    census = single_log_report(records)
    assert census["regime"] == "new"
    assert census["blind_spot"] is True
    assert census["divergence_derivable"] is False
    assert census["undecidable_deliveries"] == 2
    assert census["diverging_events"] == []
    text = render_single_log_text(census, path="new.jsonl")
    assert "BLIND SPOT" in text
    # It must NOT claim a benign "no divergence derivable from the payloads".
    assert "no per-event legacy/new divergence derivable from the payloads" not in text


def test_single_log_legacy_regime_derivation_still_works() -> None:
    records = [
        _sig(1, task_id="t0", channel="nudge_replay", would_cancel_inflight=True),
        _sig(2, task_id="t1", channel="nudge_replay", would_cancel_inflight=False),
    ]
    census = single_log_report(records)
    assert census["regime"] == "legacy"
    assert census["blind_spot"] is False
    assert census["divergence_derivable"] is True
    # The cancelling delivery is derivably a legacy-vs-new divergence.
    assert [d["task_id"] for d in census["diverging_events"]] == ["t0"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
