"""Regression test for audit issue #402 — dispatch-after-plan-swap.

``_promote_drift_to_steer`` pre-fix dispatched the ``GOLDFIVE_STEER``
ControlMessage BEFORE ``planner.refine_steer`` ran, so the payload's
``replacement_task_ids`` were derived from the prior plan's PENDING
tasks. The executor's overlay loop would then re-invoke against ids
the imminent revision was about to remove / cancel.

The fix (this test) requires the dispatch to land AFTER
``_emit_plan_revised`` has swapped ``session.plan`` to the revised
version. The dispatch helper re-reads ``session.plan`` to derive
``replacement_task_ids``, so dispatching post-swap surfaces the new
plan's task ids on the wire.

The test is deliberately tight on the failure surface: it builds a
prior plan whose first PENDING id is ``t_old_pending`` and a revised
plan whose first PENDING id is ``t_new_pending``. Pre-fix the
captured ControlMessage carries ``["t_old_pending"]`` (FAIL); post-fix
it carries ``["t_new_pending"]`` (PASS).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import (  # noqa: E402
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover
        return


class _RevisedPlanPlanner:
    """Planner that always returns the same revised plan for refine_steer."""

    def __init__(self, revised: Plan) -> None:
        self.revised = revised
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, **_kw: Any) -> Plan | None:  # pragma: no cover
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised


def _make_prior_plan() -> Plan:
    """Plan whose first PENDING task is ``t_old_pending``.

    ``t_running`` is the currently-running task (the drift target);
    ``t_old_pending`` is what ``_dispatch_goldfive_steer_control``
    would pick as the natural successor under the pre-fix ordering.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_done", title="Done", status=TaskStatus.COMPLETED),
            Task(id="t_running", title="Running", status=TaskStatus.RUNNING),
            Task(id="t_old_pending", title="Old pending", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t_done", to_task_id="t_running"),
            TaskEdge(from_task_id="t_running", to_task_id="t_old_pending"),
        ],
    )


def _make_revised_plan() -> Plan:
    """Revised plan whose first PENDING task is ``t_new_pending``.

    The revision drops ``t_old_pending`` entirely and replaces it
    with a corrective task ``t_new_pending``. Post-fix the dispatched
    ControlMessage must carry ``t_new_pending`` in
    ``replacement_task_ids``.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_done", title="Done", status=TaskStatus.COMPLETED),
            Task(id="t_running", title="Running", status=TaskStatus.RUNNING),
            Task(id="t_new_pending", title="New pending", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="t_done", to_task_id="t_running"),
            TaskEdge(from_task_id="t_running", to_task_id="t_new_pending"),
        ],
        revision_kind=DriftKind.OFF_TOPIC.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="ship it")],
        plan=_make_prior_plan(),
        current_task_id="t_running",
    )


def _bound_steerer(*, legacy_ladder: bool = False) -> tuple[
    DefaultSteerer, _RevisedPlanPlanner, ControlChannel, _ListSink
]:
    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        goldfive_steer_suppression_window_turns=3,
    )
    # AGENCY-PRESERVATION.md PR 7: promotion strips its GOLDFIVE_STEER dispatch
    # + active_steer stamp by default; ``legacy_ladder=True`` restores them.
    steerer._legacy_ladder = bool(legacy_ladder)
    planner = _RevisedPlanPlanner(revised=_make_revised_plan())
    sink = _ListSink()
    channel = ControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)
    return steerer, planner, channel, sink


def _drain_channel(channel: ControlChannel) -> list[ControlMessage]:
    drained: list[ControlMessage] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        drained.append(inbox.get_nowait())
    return drained


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


async def test_promote_drift_dispatches_after_plan_swap_audit_402() -> None:
    """Audit #402 regression: the ``GOLDFIVE_STEER`` ControlMessage's
    ``replacement_task_ids`` must reflect the NEW plan (post-refine),
    not the prior plan.

    Pre-fix ``_promote_drift_to_steer`` dispatched before
    ``planner.refine_steer`` ran. The payload's
    ``replacement_task_ids`` was the prior plan's first PENDING task
    (``t_old_pending``) — but the imminent revision removed that task
    and replaced it with ``t_new_pending``, leaving the executor's
    overlay loop pointed at a task that no longer existed.

    Post-fix the dispatch fires AFTER ``_emit_plan_revised`` has
    swapped ``session.plan`` to the revised version, so
    ``_dispatch_goldfive_steer_control`` reads the NEW plan's PENDING
    tasks and the wire payload carries ``t_new_pending``.

    AGENCY-PRESERVATION.md PR 7: promotion's GOLDFIVE_STEER dispatch now fires
    only under the ``legacy_ladder`` escape hatch (the default regime enqueues
    an advisory note instead — see
    ``test_promote_drift_new_regime_enqueues_note_no_steer``). This audit-#402
    ordering regression is pinned in the legacy regime where the dispatch
    survives.
    """
    steerer, planner, channel, _sink = _bound_steerer(legacy_ladder=True)
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered off into raccoons",
        current_task_id="t_running",
        authored_by="goldfive",
    )
    session = _make_session()

    # Sanity: pre-dispatch, the prior plan's first PENDING is the OLD one.
    prior_pendings = [
        t.id for t in session.plan.tasks if t.status is TaskStatus.PENDING
    ]
    assert prior_pendings[:1] == ["t_old_pending"], (
        f"fixture pre-condition: prior plan's first PENDING should be "
        f"t_old_pending, got {prior_pendings}"
    )

    await steerer.drift.handle_drift(drift, session)

    # refine_steer ran — promotion path was exercised.
    assert planner.refine_steer_calls, (
        "refine_steer should have fired (promotion path)"
    )

    # The revised plan landed on the session.
    assert session.plan is not None
    new_pendings = [
        t.id for t in session.plan.tasks if t.status is TaskStatus.PENDING
    ]
    assert new_pendings[:1] == ["t_new_pending"], (
        f"plan swap should have installed t_new_pending, "
        f"got pending={new_pendings}"
    )

    # The dispatched GOLDFIVE_STEER ControlMessage carries the NEW
    # plan's task ids (audit #402 fix).
    msgs = [
        m for m in _drain_channel(channel) if m.kind is ControlKind.GOLDFIVE_STEER
    ]
    assert len(msgs) == 1, (
        f"expected exactly 1 GOLDFIVE_STEER, got {[m.kind for m in msgs]!r}"
    )
    payload = msgs[0].payload
    assert payload["replacement_task_ids"] == ["t_new_pending"], (
        f"audit #402: replacement_task_ids must reference the NEW plan's "
        f"PENDING tasks (post-refine), not the prior plan's. "
        f"got {payload['replacement_task_ids']!r}, expected ['t_new_pending']"
    )
    # Superseded carries the originating task (unchanged by the fix).
    assert payload["superseded_task_ids"] == ["t_running"]
    # Drift kind + body still propagate.
    assert payload["drift_kind"] == DriftKind.OFF_TOPIC.value
    assert "raccoons" in payload["body"]


async def test_promote_drift_new_regime_enqueues_note_no_steer() -> None:
    """AGENCY-PRESERVATION.md PR 7: default-regime promotion strips its
    steering side-effects.

    The promotion still refines (``refine_steer``) and emits ``PlanRevised``,
    but it no longer dispatches a ``GOLDFIVE_STEER`` ControlMessage, tags the
    adapter cancel reason, or stamps ``active_steer(source="goldfive")``.
    Instead it enqueues an advisory observer note on the configured channel
    (legacy ``pending_nudges`` by default).
    """
    steerer, planner, channel, _sink = _bound_steerer()  # legacy_ladder=False
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent wandered off into raccoons",
        current_task_id="t_running",
        authored_by="goldfive",
    )
    session = _make_session()
    await steerer.drift.handle_drift(drift, session)

    # Kept: refine_steer fired + the plan swapped.
    assert planner.refine_steer_calls, "refine_steer should still fire"
    assert session.plan is not None

    # Stripped: no GOLDFIVE_STEER ControlMessage.
    steer_msgs = [
        m for m in _drain_channel(channel) if m.kind is ControlKind.GOLDFIVE_STEER
    ]
    assert steer_msgs == [], "default regime must not dispatch GOLDFIVE_STEER"

    # Stripped: active_steer not stamped with the goldfive source.
    from goldfive.state_store import StateStore

    active = StateStore.for_session(session).get_active_steer()
    assert active is None or active.source.lower() != "goldfive"

    # Added: an advisory note was enqueued (legacy_user_message channel).
    from goldfive.observer_notes import ADVISORY_FOOTER

    assert len(session.pending_nudges) == 1
    assert ADVISORY_FOOTER in session.pending_nudges[0]
