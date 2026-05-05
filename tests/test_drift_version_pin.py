"""Regression tests for goldfive#245: verdict-freshness gate.

The brussels-sprouts and tomato e2e sessions exposed a structural bug
class — drift judges fire verdicts based on a plan-state snapshot the
system has moved past by the time the verdict applies. The 1:13
GOAL_DRIFT in the tomato run said "drafting still pending" against a
state where draft was already DONE; ``classify_goal_drift`` materialised
the prompt's tasks block BEFORE the LLM await and the reconciler's
``mark_task_completed`` flipped the status during the round-trip.

The structural fix:

1. Every observation/verdict carries the plan-revision it observed
   (:attr:`DriftEvent.observed_revision_index`), stamped at the top of
   each detector BEFORE its LLM await.
2. The dispatch path
   (:meth:`~goldfive.steerer.DefaultSteerer._handle_drift`) rejects
   stale verdicts — drifts whose observed revision is strictly older
   than the live ``session.plan.revision_index``.
3. The goal-drift judge re-reads ``session.plan`` post-LLM and drops
   verdicts whose target task transitioned out of PENDING during the
   round-trip.

This module covers all three layers.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift.goals import classify_goal_drift  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


def _session(
    *,
    run_id: str = "r-245",
    revision_index: int = 1,
    statuses: list[tuple[str, TaskStatus]] | None = None,
) -> Session:
    """Build a Session whose plan has the given task ids + statuses."""
    if statuses is None:
        statuses = [("t-draft", TaskStatus.PENDING)]
    tasks = [
        Task(id=tid, title=f"task {tid}", description="", status=s)
        for (tid, s) in statuses
    ]
    plan = Plan(
        id="p-245",
        run_id=run_id,
        goal_ids=["g-245"],
        tasks=tasks,
        edges=[],
        revision_index=revision_index,
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g-245", summary="ship the doc")],
        plan=plan,
        current_task_id=tasks[0].id if tasks else "",
    )


def _drift(
    *,
    kind: DriftKind = DriftKind.OFF_TOPIC,
    severity: DriftSeverity = DriftSeverity.WARNING,
    observed_revision_index: int = 0,
    authored_by: str = "goldfive",
    task: str = "t-draft",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail="stale verdict test",
        current_task_id=task,
        current_agent_id="agent-x",
        authored_by=authored_by,
        observed_revision_index=observed_revision_index,
    )


def _drift_detected_payloads(sink: _ListSink) -> list[Any]:
    """Return the ``DriftDetected`` proto messages emitted on the sink.

    The steerer may emit a mix of proto envelopes (DriftDetected,
    PlanRevised, …) and dict envelopes (``refine_failed``, …) onto the
    sink bus; this helper filters down to the proto DriftDetected
    payloads.
    """
    out: list[Any] = []
    for evt in sink.events:
        which = getattr(evt, "WhichOneof", None)
        if which is None:
            continue  # dict envelope (refine_failed etc.) — not a DriftDetected
        if which("payload") == "drift_detected":
            out.append(evt.drift_detected)
    return out


def _emitted_drift_kinds(sink: _ListSink) -> list[int]:
    """Return the ``DriftDetected.kind`` enum-int for every emitted drift."""
    return [int(p.kind) for p in _drift_detected_payloads(sink)]


def _emitted_drift_observed_revisions(sink: _ListSink) -> list[int]:
    """Return ``DriftDetected.observed_revision_index`` for every emit."""
    return [int(p.observed_revision_index) for p in _drift_detected_payloads(sink)]


# ---------------------------------------------------------------------------
# 1. Same revision → drift dispatched (cancel + refine path runs)
# ---------------------------------------------------------------------------


def _build_steerer(
    *, sinks: list[Any] | None = None, planner: Any | None = None
) -> DefaultSteerer:
    s = DefaultSteerer()
    s.bind(sinks=sinks or [], planner=planner or _NullPlanner())
    return s


async def test_drift_stamped_with_current_revision_is_dispatched() -> None:
    """A drift stamped with revision N is dispatched when session is on N.

    The gate is keyed on ``observed_revision_index < live_revision`` —
    equality is fine: the detector observed the live plan, so the
    verdict is fresh.
    """
    sink = _ListSink()
    steerer = _build_steerer(sinks=[sink])
    session = _session(revision_index=3)
    drift = _drift(observed_revision_index=3)

    await steerer._handle_drift(drift, session)

    # DriftDetected should be on the wire (the first one is OUR drift;
    # the steerer may emit follow-up escalation drifts because the
    # _NullPlanner returns None — we only care about the first).
    payloads = _drift_detected_payloads(sink)
    assert payloads, "DriftDetected was not emitted"
    # The wire's observed_revision_index must mirror the drift on the
    # first DriftDetected emit (subsequent escalation drifts are
    # synthesized from the steerer's terminal-escalation path and
    # carry their own bookkeeping).
    assert int(payloads[0].observed_revision_index) == 3


# ---------------------------------------------------------------------------
# 2. Stale revision → drift REJECTED at dispatch (emit-only)
# ---------------------------------------------------------------------------


async def test_redundant_same_kind_same_target_drift_is_rejected() -> None:
    """Drift observed against (kind, target) already addressed at later revision is rejected.

    Per-key gating semantics: a verdict is rejected only when the SAME
    (drift kind, target task) was already addressed at a later revision.
    Set the per-key watermark to revision 4; drift observed at revision 3
    for the same (kind, target) is redundant.

    DriftDetected still emits for observability; cancel + refine skipped.
    """
    sink = _ListSink()
    planner = _NullPlanner()
    steerer = _build_steerer(sinks=[sink], planner=planner)
    session = _session(revision_index=4)
    # Stamp the per-(kind, target) watermark as if a prior refine for
    # the same (OFF_TOPIC, t-draft) addressed the concern at revision 4.
    session.last_addressed_revision_by_drift_key[
        (DriftKind.OFF_TOPIC.value, "t-draft")
    ] = 4
    drift = _drift(observed_revision_index=3, severity=DriftSeverity.WARNING)

    await steerer._handle_drift(drift, session)

    # DriftDetected fires for observability.
    assert _emitted_drift_kinds(sink), "redundant drift must still emit DriftDetected"
    # The dispatch (planner.refine) does NOT fire — the gate skipped it.
    assert planner.refine_calls == [], (
        "redundant verdict (same key, older observation) must not trigger "
        f"planner.refine; refine called {len(planner.refine_calls)} time(s)"
    )


async def test_orthogonal_kind_drift_dispatches_after_unrelated_revision() -> None:
    """Drift on a DIFFERENT (kind, target) is NOT rejected by an unrelated revision bump.

    The bug class this guards against: parallel judges fire on
    *orthogonal* concerns. Judge A observes revision N → fires OFF_TOPIC
    on task T_A → refine produces revision N+1 (addresses OFF_TOPIC/T_A).
    Judge B observes revision N → fires PLAN_DIVERGENCE on task T_B; B's
    verdict returns AFTER A's refine landed. Naive global ``observed <
    live`` gating would drop B; per-(kind, target) gating preserves it
    because (PLAN_DIVERGENCE, T_B) was never specifically addressed.

    Both PLAN_DIVERGENCE and OFF_TOPIC at WARNING severity route to the
    refine path (CANCEL_REINVOKE), so we assert ``planner.refine_calls``
    fires for the orthogonal drift.
    """
    sink = _ListSink()
    planner = _NullPlanner()
    steerer = _build_steerer(sinks=[sink], planner=planner)
    session = _session(
        revision_index=4,
        statuses=[("t-draft", TaskStatus.COMPLETED), ("t-other", TaskStatus.PENDING)],
    )
    # Watermark records an unrelated key was addressed at revision 4.
    session.last_addressed_revision_by_drift_key[
        (DriftKind.OFF_TOPIC.value, "t-draft")
    ] = 4
    # The drift is PLAN_DIVERGENCE on a different task — orthogonal
    # concern, also a refine-routing kind so we can assert the
    # dispatch path ran.
    drift = _drift(
        kind=DriftKind.PLAN_DIVERGENCE,
        severity=DriftSeverity.WARNING,
        observed_revision_index=3,
        task="t-other",
    )

    await steerer._handle_drift(drift, session)

    # DriftDetected fires AND the dispatch runs (planner.refine called).
    assert _emitted_drift_kinds(sink), "orthogonal drift must emit DriftDetected"
    assert planner.refine_calls, (
        "orthogonal-key drift must dispatch even when an unrelated key was "
        "addressed at a later revision; got 0 refine calls"
    )


async def test_same_kind_different_target_drift_dispatches() -> None:
    """Drift on same kind but DIFFERENT target task dispatches.

    Watermark on (OFF_TOPIC, t-draft) at revision 4 must NOT suppress a
    later (OFF_TOPIC, t-other) drift — they're distinct claims about
    distinct tasks.
    """
    sink = _ListSink()
    planner = _NullPlanner()
    steerer = _build_steerer(sinks=[sink], planner=planner)
    session = _session(
        revision_index=4,
        statuses=[("t-draft", TaskStatus.COMPLETED), ("t-other", TaskStatus.PENDING)],
    )
    session.last_addressed_revision_by_drift_key[
        (DriftKind.OFF_TOPIC.value, "t-draft")
    ] = 4
    drift = _drift(
        kind=DriftKind.OFF_TOPIC,
        observed_revision_index=3,
        task="t-other",
    )

    await steerer._handle_drift(drift, session)

    assert planner.refine_calls, (
        "same-kind-different-target drift must dispatch; got 0 refine calls"
    )


# ---------------------------------------------------------------------------
# 3. Unstamped drift → back-compat path, dispatched
# ---------------------------------------------------------------------------


async def test_unstamped_drift_bypasses_the_gate() -> None:
    """``observed_revision_index == 0`` means "unset / pre-#245".

    The gate is a no-op for legacy / external producers — the dispatch
    path runs unchanged. Asserting back-compat: existing producers
    (out-of-tree adapters, serialised events, older clients) keep
    working.
    """
    sink = _ListSink()
    planner = _NullPlanner()
    steerer = _build_steerer(sinks=[sink], planner=planner)
    session = _session(revision_index=10)
    # observed_revision_index defaults to 0 — the legacy/unset sentinel.
    drift = _drift(observed_revision_index=0)
    assert drift.observed_revision_index == 0

    await steerer._handle_drift(drift, session)

    # DriftDetected fires (back-compat).
    payloads = _drift_detected_payloads(sink)
    assert payloads, "unstamped drift must still emit DriftDetected"
    # The wire field for OUR drift is the unset sentinel; subsequent
    # escalation drifts (steerer follow-up) carry their own zeros, so
    # we just assert the first emit's value is what we sent.
    assert int(payloads[0].observed_revision_index) == 0


# ---------------------------------------------------------------------------
# 4. User-authored drift → bypasses the gate even when stale
# ---------------------------------------------------------------------------


async def test_user_authored_drift_bypasses_the_gate() -> None:
    """USER_STEER / USER_CANCEL / USER_PAUSE bypass even when stale.

    Operator directives must be honoured regardless of the framework's
    plan-state cursor — preserves the iter-11D / #242 contract (the
    late-drift gate also lets user-authored drifts through). A stale
    user-authored drift still makes it past the gate so the
    DriftDetected emission carries the user's intent.
    """
    sink = _ListSink()
    steerer = _build_steerer(sinks=[sink])
    session = _session(revision_index=5)

    for kind in (DriftKind.USER_STEER, DriftKind.USER_CANCEL, DriftKind.USER_PAUSE):
        sink.events.clear()
        drift = _drift(
            kind=kind,
            authored_by="user",
            observed_revision_index=2,  # very stale
        )
        await steerer._handle_drift(drift, session)
        # User drift made it past the gate — at minimum DriftDetected
        # was emitted. (The cancel + refine machinery is the steerer's
        # downstream business; we only assert the gate did not skip
        # the dispatch.)
        kinds = _emitted_drift_kinds(sink)
        assert kinds, (
            f"user-authored drift kind={kind!r} must bypass the freshness "
            "gate even when the observed revision is stale"
        )


# ---------------------------------------------------------------------------
# 5. classify_goal_drift: post-LLM re-read drops verdict if task transitioned
# ---------------------------------------------------------------------------


async def test_goal_drift_post_llm_reread_drops_when_task_transitioned() -> None:
    """The judge said "task t-draft still pending" but during the LLM
    round-trip ``mark_task_completed`` flipped t-draft to COMPLETED.
    The post-LLM re-read must drop the verdict.
    """
    session = _session(
        revision_index=2,
        statuses=[("t-draft", TaskStatus.PENDING)],
    )
    # Snapshot the plan we hand the judge: it sees t-draft PENDING.
    pre_call_plan = Plan(
        id=session.plan.id,
        run_id=session.plan.run_id,
        goal_ids=session.plan.goal_ids,
        tasks=[Task(id="t-draft", title="draft", status=TaskStatus.PENDING)],
        edges=[],
        revision_index=2,
    )

    async def _flipping_call_llm(system: str, user: str, model: str) -> str:
        # Simulate the reconciler transitioning the live plan during
        # the LLM round-trip: t-draft moves PENDING -> COMPLETED.
        session.plan.tasks[0].status = TaskStatus.COMPLETED
        # Verdict references the specific task id so the targeted
        # post-LLM re-read tier engages.
        return (
            '{"progressing": false, "reason": "task t-draft is still '
            'pending and no draft has been produced"}'
        )

    drift = await classify_goal_drift(
        goals=session.goals,
        plan=pre_call_plan,
        observed_actions=[{"kind": "agent_invocation_completed", "agent_name": "a"}],
        model="test-model",
        call_llm=_flipping_call_llm,
        current_task_id="t-draft",
        session=session,
    )
    assert drift is None, (
        "goal-drift judge must drop the verdict when the targeted task "
        "transitioned out of PENDING during the LLM round-trip"
    )


async def test_goal_drift_post_llm_reread_drops_when_revision_advanced() -> None:
    """Generic verdict (no specific task id) — the judge said "drafting
    is not progressing"; during the round-trip a refine landed and
    revision_index bumped. The fallback tier of the post-LLM re-read
    must drop the verdict.
    """
    session = _session(
        revision_index=2,
        statuses=[("t-draft", TaskStatus.PENDING)],
    )
    pre_call_plan = Plan(
        id=session.plan.id,
        run_id=session.plan.run_id,
        goal_ids=session.plan.goal_ids,
        tasks=[Task(id="t-draft", title="draft", status=TaskStatus.PENDING)],
        edges=[],
        revision_index=2,
    )

    async def _refining_call_llm(system: str, user: str, model: str) -> str:
        # Simulate a refine landing during the LLM round-trip: revision
        # bumps to 3, the original task is replaced by a new one.
        session.plan.revision_index = 3
        session.plan.tasks = [
            Task(id="t-draft-v2", title="draft v2", status=TaskStatus.PENDING),
        ]
        # Generic narrative — no task id mentioned.
        return (
            '{"progressing": false, "reason": "drafting work is not '
            'advancing toward the goal"}'
        )

    drift = await classify_goal_drift(
        goals=session.goals,
        plan=pre_call_plan,
        observed_actions=[{"kind": "agent_invocation_completed", "agent_name": "a"}],
        model="test-model",
        call_llm=_refining_call_llm,
        current_task_id="",
        session=session,
    )
    assert drift is None, (
        "generic goal-drift verdict must be dropped when the plan's "
        "revision_index or task-status set materially changed during "
        "the LLM round-trip"
    )


async def test_goal_drift_post_llm_reread_keeps_verdict_when_plan_stable() -> None:
    """Sanity: when the plan is stable across the LLM round-trip, the
    post-LLM re-read does NOT swallow a legitimate off-track verdict.
    """
    session = _session(
        revision_index=2,
        statuses=[("t-draft", TaskStatus.PENDING)],
    )
    pre_call_plan = session.plan

    async def _stable_call_llm(system: str, user: str, model: str) -> str:
        # No mutation during the call.
        return '{"progressing": false, "reason": "no draft yet"}'

    drift = await classify_goal_drift(
        goals=session.goals,
        plan=pre_call_plan,
        observed_actions=[{"kind": "agent_invocation_completed", "agent_name": "a"}],
        model="test-model",
        call_llm=_stable_call_llm,
        current_task_id="t-draft",
        session=session,
    )
    assert drift is not None
    assert drift.kind is DriftKind.GOAL_DRIFT
    # The drift carries the observation-time revision so the dispatch
    # gate has the right signal.
    assert drift.observed_revision_index == 2


# ---------------------------------------------------------------------------
# 6. Reasoning judge: drift carries the right revision_index when emitted
# ---------------------------------------------------------------------------


async def test_reasoning_judge_stamps_observed_revision_on_drift() -> None:
    """``classify_reasoning_drift`` stamps ``observed_revision_index``
    from the plan it was handed BEFORE the LLM await.
    """
    from goldfive.drift.reasoning_judge import classify_reasoning_drift

    plan = Plan(
        id="p-rj",
        run_id="r-rj",
        goal_ids=["g-rj"],
        tasks=[Task(id="t-rj", title="research", status=TaskStatus.PENDING)],
        edges=[],
        revision_index=7,
    )
    task = plan.tasks[0]
    goals = [Goal(id="g-rj", summary="answer the question")]

    async def _call_llm(system: str, user: str, model: str) -> str:
        return (
            '{"classification": "erroneous_deviation", '
            '"severity": "warning", '
            '"reason": "agent rambling"}'
        )

    drift = await classify_reasoning_drift(
        reasoning="some random reasoning that goes off-topic",
        task=task,
        goals=goals,
        model="test-model",
        call_llm=_call_llm,
        current_task_id="t-rj",
        current_agent_id="agent-rj",
        plan=plan,
    )
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    assert drift.observed_revision_index == 7, (
        "reasoning judge must stamp the observation-time plan revision "
        "onto the drift; expected 7, got "
        f"{drift.observed_revision_index}"
    )


# ---------------------------------------------------------------------------
# 7. End-to-end: stamped + advanced revision is rejected; sink emits drift
# ---------------------------------------------------------------------------


async def test_end_to_end_redundant_drift_emits_but_does_not_refine() -> None:
    """The full path: detector emits a stamped drift; before dispatch
    a refine for the SAME (kind, target) lands and stamps the watermark;
    the gate rejects the verdict as redundant but ``DriftDetected`` is
    still on the wire. Operators see the detector fired but the
    framework does not re-act on a concern already addressed.
    """
    sink = _ListSink()
    planner = _NullPlanner()
    steerer = _build_steerer(sinks=[sink], planner=planner)

    session = _session(revision_index=2)
    # Detector observed at revision 2 (the live revision at observation).
    drift = _drift(observed_revision_index=2)
    # Now a refine for the SAME (kind, target) lands externally —
    # watermark advances to revision 3 BEFORE the detector's drift
    # reaches dispatch. (In production this happens via _apply_revision
    # stamping; we simulate it here by setting the dict directly.)
    session.plan.revision_index = 3
    session.last_addressed_revision_by_drift_key[
        (DriftKind.OFF_TOPIC.value, "t-draft")
    ] = 3

    await steerer._handle_drift(drift, session)

    # Drift on the wire (observability preserved) ...
    assert _emitted_drift_kinds(sink), "DriftDetected must still be emitted"
    # ... but no refine fired.
    assert planner.refine_calls == [], (
        "redundant verdict (same key, older observation) must not drive a "
        f"refine; got {len(planner.refine_calls)} refine call(s)"
    )


async def test_apply_revision_stamps_per_key_watermark() -> None:
    """``_apply_revision`` stamps ``last_addressed_revision_by_drift_key``.

    This is the producer side of the gate: every successful goldfive-
    authored refine that lands a new plan must stamp the watermark so
    subsequent same-(kind, target) verdicts observed at older revisions
    are correctly identified as redundant.
    """
    session = _session(revision_index=2)
    drift = _drift(
        kind=DriftKind.OFF_TOPIC,
        observed_revision_index=2,
        task="t-draft",
    )
    revised = Plan(
        id="p-revised",
        run_id=session.run_id,
        goal_ids=["g-245"],
        tasks=list(session.plan.tasks),
        edges=[],
        revision_index=3,
    )

    DefaultSteerer._apply_revision(session, revised, drift)

    key = (DriftKind.OFF_TOPIC.value, "t-draft")
    assert key in session.last_addressed_revision_by_drift_key, (
        "_apply_revision must stamp the per-(kind, target) watermark"
    )
    assert session.last_addressed_revision_by_drift_key[key] == 3, (
        "watermark must equal the new revision_index"
    )


async def test_apply_revision_does_not_stamp_for_user_authored_drifts() -> None:
    """User-authored drifts bypass the gate AND don't stamp the watermark.

    Rationale: a user redirect doesn't suppress orthogonal goldfive
    concerns observed before the redirect. Goldfive verdicts whose
    (kind, target) was NEVER goldfive-addressed flow through to dispatch
    even if a USER_STEER bumped the plan in between.
    """
    session = _session(revision_index=2)
    drift = _drift(
        kind=DriftKind.USER_STEER,
        observed_revision_index=2,
        task="t-draft",
        authored_by="user",
    )
    revised = Plan(
        id="p-revised",
        run_id=session.run_id,
        goal_ids=["g-245"],
        tasks=list(session.plan.tasks),
        edges=[],
        revision_index=3,
    )

    DefaultSteerer._apply_revision(session, revised, drift)

    key = (DriftKind.USER_STEER.value, "t-draft")
    assert key not in session.last_addressed_revision_by_drift_key, (
        "user-authored drifts must NOT stamp the watermark — they're operator "
        "directives, not goldfive-issued addressings of a structural concern"
    )
