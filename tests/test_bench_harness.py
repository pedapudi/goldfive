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
    run_arm,
    run_arms,
    shadow_arms,
)
from bench.shadow_diff import (  # noqa: E402
    ShadowDiffError,
    diff_two_logs,
    load_signals,
    render_two_log_text,
    single_log_report,
)


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
        # The workload completed: captured-artifact outcome is healthy.
        assert m.goal_success, m.goal_reason
        assert m.completed_outputs == 3
        assert m.turns >= 1
        # Telemetry flowed: the JSONL artifact exists and carries signals.
        assert Path(m.jsonl_path).exists()
        assert m.signals_total > 0, f"{m.arm_name}: 0 signals — telemetry not wired"

    by_kind = {m.arm_kind: m for m in results}
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

    # The looping WARNING key never cancels under either regime, so it does
    # NOT diverge on the cancel-authority dimension this test pins. (Since
    # AGENCY-PRESERVATION.md PR 6 it DOES diverge on ``channel`` — the signal
    # arm rides the new ``request_context`` observer-note channel while the
    # legacy arm rides ``nudge_replay`` — which is the expected per-regime
    # transport difference, not a cancel-authority divergence.)
    looping = [k for k in report.keys if k.kind == "looping_tool_call"]
    assert looping
    assert "would_cancel_inflight" not in looping[0].diverged_fields

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
# 5. Graceful degradation: pending vs. applied flags
# ---------------------------------------------------------------------------


def test_arm_flag_status_reports_promoted_flags_as_applied() -> None:
    arms = {a.kind: a for a in default_arms()}
    applied, pending = arm_flag_status(arms["signal"])
    # The promotion contract: the PR that teaches ``SteeringConfig.from_env``
    # about a flag moves it from _PENDING_STEER_ENV into KNOWN_STEER_ENV, so it
    # reports APPLIED. As of PR 7 every roadmap flag the bench carries is
    # promoted: GOLDFIVE_STEER_SIGNAL_CHANNEL (PR 6), GOLDFIVE_PLAN_MODE
    # (PR 10), and GOLDFIVE_STEER_LEGACY_LADDER (PR 7).
    assert "GOLDFIVE_STEER_SIGNAL_CHANNEL" in applied
    assert "GOLDFIVE_STEER_SIGNAL_CHANNEL" not in pending
    assert "GOLDFIVE_PLAN_MODE" in applied
    # The legacy arm's escape hatch is now APPLIED (PR 7 read it into from_env);
    # nothing the bench carries is pending anymore.
    legacy_applied, legacy_pending = arm_flag_status(arms["legacy"])
    assert "GOLDFIVE_STEER_LEGACY_LADDER" in legacy_applied
    assert legacy_pending == []
    # The flags this build DOES consult are reported as applied.
    assert "GOLDFIVE_STEER_SIGNAL_TELEMETRY" in applied
    assert set(applied) <= KNOWN_STEER_ENV


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
    assert metrics.goal_success
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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
