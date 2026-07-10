"""Authority gate for in-flight cancellation (AGENCY-PRESERVATION.md PR 1).

goldfive#449/#452: pre-PR-1, ``DriftObserver._cancel_inflight_for_revision``
fired on EVERY drift-driven plan install — including Level-1 ABSORB
refines and NEW_WORK_DISCOVERED installs — killing the wrapped agent's
in-flight invocation within ~one event-loop tick (design doc §1.1, "the
single largest trajectory destroyer"). PR 1 gates the cancel on drift
*authority*: only user-authored (USER_STEER / USER_CANCEL / USER_PAUSE)
or hard-safety (budget / resource protection and termination —
``DriftObserver._HARD_SAFETY_DRIFT_KINDS``) drifts may preempt in-flight
work. ``SteeringConfig.cancel_inflight_scope="all"`` (env
``GOLDFIVE_CANCEL_INFLIGHT_SCOPE=all``) is the §5.1 kill-switch that
restores the legacy behaviour exactly.

Cancel-guarantee inventory (§5.2) — what the unconditional cancel used
to provide and what proves its replacement here:

(a) **Loop-break for the v15 concurrent-invocation bug** — a slow
    refine overlapping a still-generating coordinator used to be bounded
    by killing the coordinator. Now bounded by the goldfive#405 in-flight
    refine registry (same-key concurrent verdicts short-circuit), the
    goldfive#245 verdict-freshness watermark (stale verdicts drop), and
    the goldfive#271 no-op-revision rejection.
    → ``test_v15_pin_slow_refine_with_concurrent_drift_does_not_loop``
    (THE pinned v15 regression test).

(b) **Plan coherence for reporting** — the revision still installs and
    ``PlanRevised`` still emits; only the cancel is withheld.
    → ``test_gated_install_still_installs_plan_and_emits_plan_revised``.

(c) **Restart boundary for nudge delivery** — nudges used to ride the
    cancel-forced restart. They now deliver at the *natural* invocation
    boundary via the overlay loop's scoped nudge-replay path
    (``sequential.py`` ``_run_overlay``).
    → ``test_nudge_delivers_at_natural_boundary_without_cancel``.

(d) **Supersede-flag consumption** — the executor's cancelled branch
    consumes ``session._supersede_pending``. Skipped cancels stamp
    NOTHING (flag, registry, plugin), so no stranded flag can
    misclassify a later external cancel.
    → ``test_skipped_cancel_stamps_no_supersede_flag_or_registry``;
    the stamp-then-consume contract for cancels that DO fire is pinned
    in ``tests/test_executor_supersede_cancel_nonfatal.py`` (re-pointed
    at hard-safety kinds).

Legacy-mode equivalence under ``"all"`` is parametrized at the bottom of
this file.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
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
# Stubs (no ADK / no LLM / no network).
# ---------------------------------------------------------------------------


class _CountingPlugin:
    """Records every ``request_invocation_cancel`` forwarded by the steerer."""

    def __init__(self, top_invocation_id: str = "inv-X") -> None:
        self.calls: list[dict[str, Any]] = []
        self._top_invocation_id = top_invocation_id
        self._invocation_parents: dict[str, str] = {}
        self._reconciler = None

    def request_invocation_cancel(self, **kwargs: Any) -> list[str]:
        self.calls.append(kwargs)
        return [str(kwargs.get("invocation_id", ""))]


class _CountingAdapter:
    def __init__(self, top_invocation_id: str = "inv-X") -> None:
        self._next_cancel_reason = ""
        self._plugin = _CountingPlugin(top_invocation_id)
        self.available_agents = ["coordinator"]


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None

    def payload_kinds(self) -> list[str]:
        return [e.WhichOneof("payload") for e in self.events if hasattr(e, "WhichOneof")]


def _make_plan(revision_index: int = 0) -> Plan:
    return Plan(
        id="p1",
        run_id="run-gate",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING, assignee_agent_id="coordinator"),
            Task(id="t2", title="T2", status=TaskStatus.PENDING, assignee_agent_id="worker"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        revision_index=revision_index,
    )


def _make_session(revision_index: int = 0) -> Session:
    return Session(
        run_id="run-gate",
        goals=[Goal(id="g1", summary="ship the gate")],
        plan=_make_plan(revision_index),
        current_task_id="t1",
    )


def _drift(
    kind: DriftKind,
    severity: DriftSeverity = DriftSeverity.WARNING,
    **kwargs: Any,
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=kwargs.pop("detail", "gate test"),
        current_task_id=kwargs.pop("current_task_id", "t1"),
        current_agent_id=kwargs.pop("current_agent_id", "coordinator"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. The authority predicate, in one place.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [DriftKind.USER_STEER, DriftKind.USER_CANCEL, DriftKind.USER_PAUSE],
)
@pytest.mark.parametrize(
    "severity",
    [DriftSeverity.INFO, DriftSeverity.WARNING, DriftSeverity.CRITICAL],
)
def test_predicate_user_authored_always_authorized(
    kind: DriftKind, severity: DriftSeverity
) -> None:
    """User authority is absolute (§2): any severity, always authorized."""
    steerer = DefaultSteerer()
    assert steerer.drift._drift_authorizes_inflight_cancel(_drift(kind, severity)) is True


@pytest.mark.parametrize(
    "kind",
    [
        DriftKind.RESOURCE_EXHAUSTED,
        DriftKind.RUNAWAY_DELEGATION,
        DriftKind.TOO_MANY_STEPS,
        DriftKind.TASK_TIMEOUT,
        DriftKind.LLM_CALL_TIMEOUT,
        DriftKind.HUMAN_INTERVENTION_REQUIRED,
    ],
)
def test_predicate_hard_safety_authorized(kind: DriftKind) -> None:
    """The hard-safety set keeps stop authority (budget/termination)."""
    steerer = DefaultSteerer()
    assert (
        steerer.drift._drift_authorizes_inflight_cancel(_drift(kind, DriftSeverity.CRITICAL))
        is True
    )


@pytest.mark.parametrize(
    "kind",
    [
        DriftKind.OFF_TOPIC,
        DriftKind.LOOPING_REASONING,
        DriftKind.LOOPING_TOOL_CALL,
        DriftKind.GOAL_DRIFT,
        DriftKind.NEW_WORK_DISCOVERED,
        DriftKind.CAPABILITY_MISMATCH,
        DriftKind.PLAN_DIVERGENCE,
        DriftKind.INTENT_DIVERGENCE,
        DriftKind.SELF_REPORTED_STUCK,
        DriftKind.REPEATED_FAILURE,
    ],
)
@pytest.mark.parametrize(
    "severity",
    [DriftSeverity.WARNING, DriftSeverity.CRITICAL],
)
def test_predicate_goldfive_steering_kinds_not_authorized(
    kind: DriftKind, severity: DriftSeverity
) -> None:
    """Goldfive-authored steering drift never cancels in-flight work —
    even at CRITICAL severity."""
    steerer = DefaultSteerer()
    assert steerer.drift._drift_authorizes_inflight_cancel(_drift(kind, severity)) is False


def test_predicate_scope_all_authorizes_everything() -> None:
    """The §5.1 kill-switch: ``"all"`` restores unconditional authority."""
    steerer = DefaultSteerer(cancel_inflight_scope="all")
    for kind in (DriftKind.OFF_TOPIC, DriftKind.NEW_WORK_DISCOVERED, DriftKind.USER_STEER):
        assert (
            steerer.drift._drift_authorizes_inflight_cancel(_drift(kind, DriftSeverity.WARNING))
            is True
        )


def test_predicate_defensive_default_for_ducktyped_steerer() -> None:
    """A router without the knob (duck-typed / pre-PR-1 subclass) gets
    the safe default scope, not a crash and not legacy ``"all"``."""
    steerer = DefaultSteerer()
    observer = steerer.drift
    # Simulate a router that predates the knob.
    del steerer._cancel_inflight_scope
    assert (
        observer._drift_authorizes_inflight_cancel(
            _drift(DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL)
        )
        is False
    )
    assert (
        observer._drift_authorizes_inflight_cancel(
            _drift(DriftKind.USER_STEER, DriftSeverity.WARNING)
        )
        is True
    )


# ---------------------------------------------------------------------------
# 2. The pre-refine cooperative-cancel path (_should_request_cancel_for_drift).
# ---------------------------------------------------------------------------


def test_pre_refine_cancel_user_authored_byte_identical() -> None:
    """User-authored drifts bypass the severity gate — exactly the
    pre-PR-1 contract."""
    steerer = DefaultSteerer()
    for kind in (DriftKind.USER_STEER, DriftKind.USER_CANCEL, DriftKind.USER_PAUSE):
        for severity in (DriftSeverity.INFO, DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            assert steerer.drift._should_request_cancel_for_drift(_drift(kind, severity)) is True


def test_pre_refine_cancel_hard_safety_critical_only() -> None:
    """Hard-safety kinds keep the goldfive#251 severity ladder: CRITICAL
    cancels, WARNING/INFO do not (the gate never EXPANDS cancellation)."""
    steerer = DefaultSteerer()
    assert (
        steerer.drift._should_request_cancel_for_drift(
            _drift(DriftKind.RUNAWAY_DELEGATION, DriftSeverity.CRITICAL)
        )
        is True
    )
    assert (
        steerer.drift._should_request_cancel_for_drift(
            _drift(DriftKind.RUNAWAY_DELEGATION, DriftSeverity.WARNING)
        )
        is False
    )


def test_pre_refine_cancel_goldfive_critical_no_longer_cancels() -> None:
    """The PR-1 behaviour change on this path: a CRITICAL goldfive-
    authored steering drift used to cancel; now it never does."""
    steerer = DefaultSteerer()
    assert (
        steerer.drift._should_request_cancel_for_drift(
            _drift(DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL)
        )
        is False
    )
    # Kill-switch restores it.
    legacy = DefaultSteerer(cancel_inflight_scope="all")
    assert (
        legacy.drift._should_request_cancel_for_drift(
            _drift(DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL)
        )
        is True
    )


# ---------------------------------------------------------------------------
# 3. Guarantee (d): a skipped cancel stamps NOTHING.
# ---------------------------------------------------------------------------


async def test_skipped_cancel_stamps_no_supersede_flag_or_registry() -> None:
    """A gated (goldfive-authored) drift through
    ``_cancel_inflight_for_revision`` must leave zero traces: no
    ``session._supersede_pending``, no per-invocation supersede-registry
    entry, no plugin call. A stamped-but-never-consumed flag would make
    the executor's cancelled branch misclassify a later EXTERNAL cancel
    (USER_CANCEL, upstream CancelledError) as an internal supersede —
    exactly the v22 Bug-A class.
    """
    from goldfive.state_store import StateStore

    steerer = DefaultSteerer()
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    session = _make_session()
    store = StateStore.for_session(session)

    for kind, severity in (
        (DriftKind.OFF_TOPIC, DriftSeverity.WARNING),
        (DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL),
        (DriftKind.NEW_WORK_DISCOVERED, DriftSeverity.INFO),
        (DriftKind.LOOPING_REASONING, DriftSeverity.CRITICAL),
    ):
        flagged = await steerer.drift._cancel_inflight_for_revision(_drift(kind, severity), session)
        assert flagged == []
    assert adapter._plugin.calls == [], "gated cancels must never reach the plugin"
    assert getattr(session, "_supersede_pending", False) is False, (
        "no stranded supersede flag for skipped cancels (guarantee d)"
    )
    assert store.has_any_supersede_pending() is False, (
        "no stranded per-invocation supersede-registry entries either"
    )


async def test_user_steer_cancel_still_stamps_and_fires() -> None:
    """The user-authored arm is byte-identical: supersede flag stamped,
    plugin reached with ``cancel_inflight_task=True``."""
    steerer = DefaultSteerer(
        # Explicit active mode (#488): the suite runs the shipped
        # observation-only default; the cancel machinery under test opts in.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    session = _make_session()

    flagged = await steerer.drift._cancel_inflight_for_revision(
        _drift(DriftKind.USER_STEER, DriftSeverity.WARNING, authored_by="user"),
        session,
    )
    assert flagged == ["inv-X"]
    assert len(adapter._plugin.calls) == 1
    assert adapter._plugin.calls[0]["cancel_inflight_task"] is True
    assert getattr(session, "_supersede_pending", False) is True


async def test_hard_safety_cancel_still_stamps_and_fires() -> None:
    """Hard-safety drifts keep stop authority through the install path."""
    steerer = DefaultSteerer(
        # Explicit active mode (#488): the suite runs the shipped
        # observation-only default; the cancel machinery under test opts in.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    session = _make_session()

    flagged = await steerer.drift._cancel_inflight_for_revision(
        _drift(DriftKind.RESOURCE_EXHAUSTED, DriftSeverity.CRITICAL), session
    )
    assert flagged == ["inv-X"]
    assert getattr(session, "_supersede_pending", False) is True


# ---------------------------------------------------------------------------
# 4. Guarantee (b): plan coherence — the install + PlanRevised survive
#    the withheld cancel.
# ---------------------------------------------------------------------------


async def test_gated_install_still_installs_plan_and_emits_plan_revised() -> None:
    """A goldfive-authored NEW_WORK_DISCOVERED install through
    ``install_revision_for_drift`` (the ``plan_reviser._install_with_drift``
    call site of ``_cancel_inflight_for_revision``) lands the revision
    and emits ``PlanRevised`` — reporting stays coherent — while the
    in-flight invocation is left untouched.
    """
    steerer = DefaultSteerer()
    adapter = _CountingAdapter()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=None)
    steerer.bind_adapter(adapter)
    session = _make_session()
    prior = session.plan
    assert prior is not None

    revised = dataclasses.replace(
        prior,
        tasks=(*prior.tasks, Task(id="t3", title="discovered work", assignee_agent_id="worker")),
    )
    installed = await steerer.plans.install_revision_for_drift(
        session=session,
        drift=_drift(
            DriftKind.NEW_WORK_DISCOVERED,
            DriftSeverity.INFO,
            authored_by="goldfive",
        ),
        revised_plan=revised,
    )
    assert installed is True
    assert session.plan is not None and any(t.id == "t3" for t in session.plan.tasks)
    assert "plan_revised" in sink.payload_kinds(), (
        "PlanRevised must still emit when the cancel is withheld (guarantee b)"
    )
    assert adapter._plugin.calls == [], (
        "the NEW_WORK_DISCOVERED install must not cancel in-flight work"
    )
    assert getattr(session, "_supersede_pending", False) is False


async def test_user_steer_install_path_still_cancels() -> None:
    """``install_user_steer`` (the plan_reviser user-steer install path)
    keeps its cancel: a genuine operator pivot preempts in-flight work
    exactly as before PR 1."""
    steerer = DefaultSteerer(
        # Explicit active mode (#488): the suite runs the shipped
        # observation-only default; the cancel machinery under test opts in.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=None)
    steerer.bind_adapter(adapter)
    session = _make_session()
    prior = session.plan
    assert prior is not None

    drift = _drift(
        DriftKind.USER_STEER,
        DriftSeverity.WARNING,
        detail="by operator: pivot to the new topic",
        authored_by="user",
    )
    chosen = await steerer.plans.install_user_steer(
        drift=drift, prior=prior, llm_revision=None, session=session
    )
    # The deterministic minimum cancelled the mutable tasks — a real install.
    assert chosen is not prior
    assert len(adapter._plugin.calls) == 1, (
        "USER_STEER installs must keep preempting in-flight work"
    )
    assert adapter._plugin.calls[0]["cancel_inflight_task"] is True
    assert getattr(session, "_supersede_pending", False) is True


# ---------------------------------------------------------------------------
# 5. Promotion-driven installs (goldfive-authored) no longer cancel.
# ---------------------------------------------------------------------------


async def test_promotion_install_does_not_cancel_inflight() -> None:
    """New-default counterpart of
    ``tests/test_cancel_inflight_on_refine.py::
    test_plan_divergence_refine_cancels_inflight_coordinator_task``
    (now pinned to the legacy ``"all"`` scope): an OFF_TOPIC WARNING
    drift promotes to a goldfive steer, ``refine_steer`` installs the
    revised plan — and the in-flight coordinator invocation is NOT
    cancelled. The corrective reaches the agent via the GOLDFIVE_STEER
    restart at the invocation boundary instead.
    """
    session = _make_session()
    prior = session.plan
    assert prior is not None
    revised = dataclasses.replace(
        prior,
        tasks=(*prior.tasks, Task(id="t2b", title="Replanned T2", assignee_agent_id="worker")),
    )

    class _StubPlanner:
        def __init__(self) -> None:
            self.refine_steer_calls = 0

        async def refine_steer(self, **_kwargs: Any) -> Plan:
            self.refine_steer_calls += 1
            return revised

        async def refine(self, **_kwargs: Any) -> Plan:
            return revised

    planner = _StubPlanner()
    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        # Explicit active mode (#488) — see module note.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter(top_invocation_id="inv-coord-1")
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)

    await steerer.drift.handle_drift(_drift(DriftKind.OFF_TOPIC, DriftSeverity.WARNING), session)

    assert planner.refine_steer_calls == 1
    # The revision installed (bookkeeping kept) ...
    assert session.plan is not None and any(t.id == "t2b" for t in session.plan.tasks)
    assert "plan_revised" in sink.payload_kinds()
    # ... but the steerer never asked the plugin to cancel anything.
    assert adapter._plugin.calls == [], (
        "promotion-driven (goldfive-authored) installs must not cancel "
        "in-flight work under the default cancel_inflight_scope"
    )
    assert getattr(session, "_supersede_pending", False) is False


# ---------------------------------------------------------------------------
# 6. THE PINNED v15 REGRESSION TEST (guarantee a).
# ---------------------------------------------------------------------------


async def test_v15_pin_slow_refine_with_concurrent_drift_does_not_loop() -> None:
    """**v15 pin** (goldfive#271 follow-up; AGENCY-PRESERVATION PR 1
    §5.2 correctness requirement).

    The v15 concurrent-invocation bug: a slow ``refine_steer`` (10+
    minutes on a slow planner) overlapped the coordinator's invocation,
    whose continuing output kept triggering the SAME drift — looping the
    refine. The original fix killed the coordinator
    (``_cancel_inflight_for_revision``). PR 1 withholds that cancel for
    goldfive-authored drift, so THIS test pins what bounds the loop in
    the new regime:

    * while the refine is in flight, same-``(kind, task)`` verdicts are
      short-circuited by the goldfive#405 in-flight refine registry;
    * after the refine installs, verdicts observed against the OLD
      revision are dropped by the goldfive#245 verdict-freshness
      watermark;
    * (and, not exercised here: a planner that can only return a
      structurally identical plan trips the goldfive#271
      no-op-revision rejection → HUMAN_INTERVENTION_REQUIRED, never a
      loop).

    Net: the coordinator keeps generating for the full refine duration
    and the planner refines exactly ONCE.
    """
    session = _make_session(revision_index=1)
    prior = session.plan
    assert prior is not None
    revised = dataclasses.replace(
        prior,
        tasks=(*prior.tasks, Task(id="t2b", title="Replanned T2", assignee_agent_id="worker")),
    )

    refine_started = asyncio.Event()
    release_refine = asyncio.Event()

    class _SlowPlanner:
        def __init__(self) -> None:
            self.refine_calls = 0

        async def refine_steer(self, **_kwargs: Any) -> Plan:
            self.refine_calls += 1
            refine_started.set()
            # Simulate the multi-minute LLM round-trip: block until the
            # test releases us, i.e. until the "coordinator" has had
            # time to produce more drift-triggering output.
            await release_refine.wait()
            return revised

        async def refine(self, **_kwargs: Any) -> Plan:
            return await self.refine_steer()

    planner = _SlowPlanner()
    steerer = DefaultSteerer(
        goldfive_steer_threshold="warning",
        # Explicit active mode (#488) — see module note.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter(top_invocation_id="inv-coord-1")
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)

    def _stamped_drift() -> DriftEvent:
        d = _drift(DriftKind.OFF_TOPIC, DriftSeverity.WARNING)
        # Judges stamp the observation-time revision BEFORE their LLM
        # await (goldfive#245); both gates key off this.
        d.observed_revision_index = 1
        return d

    # t0: the first verdict dispatches; its refine blocks (slow planner).
    dispatch = asyncio.create_task(steerer.drift.handle_drift(_stamped_drift(), session))
    await asyncio.wait_for(refine_started.wait(), timeout=2.0)

    # t1: the still-generating coordinator triggers the SAME (kind, task)
    # verdict twice more while the refine is in flight. The #405
    # in-flight refine registry must short-circuit both — no second
    # refine, no queueing.
    await steerer.drift.handle_drift(_stamped_drift(), session)
    await steerer.drift.handle_drift(_stamped_drift(), session)
    assert planner.refine_calls == 1, (
        "v15 REGRESSION: concurrent same-key drift during a slow refine "
        "must not dispatch a second refine"
    )

    # t2: the refine completes and installs revision 2.
    release_refine.set()
    await asyncio.wait_for(dispatch, timeout=2.0)
    assert session.plan is not None and session.plan.revision_index == 2

    # t3: a late verdict from the coordinator's PRE-refine output (still
    # observed against revision 1) arrives after the install. The #245
    # freshness watermark must drop it.
    await steerer.drift.handle_drift(_stamped_drift(), session)
    assert planner.refine_calls == 1, (
        "v15 REGRESSION: a stale verdict against the superseded revision "
        "must not re-fire the refine"
    )

    # And throughout: the coordinator was never cancelled — the steerer
    # never reached the plugin.
    assert adapter._plugin.calls == []


# ---------------------------------------------------------------------------
# 7. Guarantee (c): nudge delivery at the natural invocation boundary.
# ---------------------------------------------------------------------------


async def test_nudge_delivers_at_natural_boundary_without_cancel() -> None:
    """With cancels gated off, a Level-2 nudge queued mid-invocation by a
    real ``DefaultSteerer`` ABSORB dispatch (LOOPING_REASONING → refine →
    nudge) must still reach the coordinator: the invocation runs to its
    natural end and the overlay loop's scoped nudge-replay path re-invokes
    the passthrough with the framed nudge as the next user message.
    """
    plan = Plan(
        id="p0",
        run_id="run-nudge",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="define", assignee_agent_id="coordinator"),
            Task(id="t2", title="draft", assignee_agent_id="worker"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    session = Session(run_id="run-nudge", goals=[Goal(id="g1", summary="ship")], plan=plan)

    revised = dataclasses.replace(
        plan,
        tasks=(*plan.tasks, Task(id="t1_v2", title="define (v2)", assignee_agent_id="coordinator")),
    )

    class _StubPlanner:
        async def generate(self, **_kwargs: Any) -> Plan | None:
            return None

        async def refine(self, **_kwargs: Any) -> Plan:
            return revised

    # threshold="critical" keeps the WARNING LOOPING_REASONING on the
    # ladder's ABSORB row (the promotion path is exercised separately
    # above) so the post-ABSORB nudge queueing fires.
    steerer = DefaultSteerer(
        goldfive_steer_threshold="critical",
        # Explicit active mode (#488) — see module note.
        steering_config=SteeringConfig(observation_only=False),
    )
    sink = _ListSink()

    class _OverlayAdapter:
        """Overlay adapter whose first passthrough simulates the
        coordinator triggering a LOOPING_REASONING drift mid-invocation
        and then finishing normally."""

        def __init__(self) -> None:
            self._plugin = _CountingPlugin(top_invocation_id="inv-coord-1")
            self._next_cancel_reason = ""
            self.passthrough_calls: list[str] = []

        @property
        def available_agents(self) -> list[str]:
            return ["coordinator", "worker"]

        async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
            return None

        async def invoke(self, task: Task, session: Session) -> InvocationResult:  # noqa: ARG002
            return InvocationResult(task_id=task.id, text="")

        async def invoke_passthrough(
            self,
            user_message: str,
            *,
            session: Session,
            reconciler: Any = None,  # noqa: ARG002
            ctx: Any = None,  # noqa: ARG002
        ) -> InvocationResult:
            self.passthrough_calls.append(user_message)
            if len(self.passthrough_calls) == 1:
                # Mid-invocation: the drift pipeline fires. ABSORB →
                # refine installs t1_v2 → nudge queued. No cancel may
                # reach this (still running) invocation.
                await steerer.drift.handle_drift(
                    _drift(DriftKind.LOOPING_REASONING, DriftSeverity.WARNING),
                    session,
                )
                # The invocation then COMPLETES NATURALLY — this return
                # is the natural boundary the replay relies on.
            return InvocationResult(task_id="", text="ok")

    adapter = _OverlayAdapter()
    steerer.bind_adapter(adapter)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=_StubPlanner(),
        sinks=[sink],
        user_input="make the deck",
    )

    # The cancel was withheld for the goldfive-authored drift...
    assert adapter._plugin.calls == [], (
        "LOOPING_REASONING (goldfive-authored) must not cancel the in-flight invocation"
    )
    # ...the plan still revised...
    assert session.plan is not None and any(t.id == "t1_v2" for t in session.plan.tasks)
    # ...and the nudge delivered at the natural invocation boundary.
    assert len(adapter.passthrough_calls) == 2, (
        f"expected the overlay nudge-replay to re-invoke once; calls={adapter.passthrough_calls!r}"
    )
    assert adapter.passthrough_calls[0] == "make the deck"
    assert adapter.passthrough_calls[1].startswith("[GOLDFIVE PLAN REVISION"), (
        adapter.passthrough_calls[1]
    )
    assert session.pending_nudges == []
    assert outcome.success is True, outcome.reason


# ---------------------------------------------------------------------------
# 8. Legacy mode (`"all"`): behaviour identical to pre-PR-1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "severity"),
    [
        (DriftKind.OFF_TOPIC, DriftSeverity.WARNING),
        (DriftKind.NEW_WORK_DISCOVERED, DriftSeverity.INFO),
        (DriftKind.LOOPING_TOOL_CALL, DriftSeverity.CRITICAL),
        (DriftKind.USER_STEER, DriftSeverity.WARNING),
    ],
)
async def test_legacy_scope_all_cancel_inflight_fires_for_every_install(
    kind: DriftKind, severity: DriftSeverity
) -> None:
    """Under the ``"all"`` kill-switch, ``_cancel_inflight_for_revision``
    behaves exactly as pre-PR-1: every drift-driven install stamps the
    supersede flag and forwards ``cancel_inflight_task=True`` to the
    plugin — including the goldfive-authored kinds the default scope
    now withholds."""
    steerer = DefaultSteerer(
        cancel_inflight_scope="all",
        # Explicit active mode (#488) — see module note.
        steering_config=SteeringConfig(observation_only=False),
    )
    adapter = _CountingAdapter()
    steerer.bind_adapter(adapter)
    session = _make_session()

    flagged = await steerer.drift._cancel_inflight_for_revision(_drift(kind, severity), session)
    assert flagged == ["inv-X"]
    assert len(adapter._plugin.calls) == 1
    assert adapter._plugin.calls[0]["cancel_inflight_task"] is True
    assert getattr(session, "_supersede_pending", False) is True


@pytest.mark.parametrize(
    ("kind", "severity", "expected"),
    [
        # Pre-PR-1 contract: any CRITICAL cancels; WARNING/INFO do not;
        # user kinds bypass the severity gate.
        (DriftKind.OFF_TOPIC, DriftSeverity.CRITICAL, True),
        (DriftKind.OFF_TOPIC, DriftSeverity.WARNING, False),
        (DriftKind.GOAL_DRIFT, DriftSeverity.CRITICAL, True),
        (DriftKind.REASONING_CLUSTER_TIGHTENING, DriftSeverity.INFO, False),
        (DriftKind.USER_STEER, DriftSeverity.WARNING, True),
        (DriftKind.USER_CANCEL, DriftSeverity.INFO, True),
    ],
)
def test_legacy_scope_all_pre_refine_predicate_identical(
    kind: DriftKind, severity: DriftSeverity, expected: bool
) -> None:
    steerer = DefaultSteerer(cancel_inflight_scope="all")
    assert steerer.drift._should_request_cancel_for_drift(_drift(kind, severity)) is expected


# ---------------------------------------------------------------------------
# 9. Config plumbing.
# ---------------------------------------------------------------------------


def test_steering_config_default_scope() -> None:
    assert SteeringConfig().cancel_inflight_scope == "user_and_safety"


def test_steering_config_from_env_reads_kill_switch(monkeypatch: Any) -> None:
    monkeypatch.setenv("GOLDFIVE_CANCEL_INFLIGHT_SCOPE", "ALL")
    assert SteeringConfig.from_env().cancel_inflight_scope == "all"


def test_steering_config_from_env_rejects_typo(monkeypatch: Any) -> None:
    """A typo must never silently flip the cancel policy — fall back to
    the safe default (mirrors the threshold/bool env helpers)."""
    monkeypatch.setenv("GOLDFIVE_CANCEL_INFLIGHT_SCOPE", "everything")
    assert SteeringConfig.from_env().cancel_inflight_scope == "user_and_safety"


def test_steerer_threads_scope_from_steering_config() -> None:
    """``goldfive.wrap`` passes ``steering_config=RuntimeConfig().steering``
    to ``DefaultSteerer`` — the same threading ``observation_only`` uses —
    so the config field is what production runs read."""
    cfg = SteeringConfig(cancel_inflight_scope="all")
    assert DefaultSteerer(steering_config=cfg)._cancel_inflight_scope == "all"
    assert DefaultSteerer()._cancel_inflight_scope == "user_and_safety"


def test_steerer_explicit_kwarg_wins_over_config() -> None:
    cfg = SteeringConfig(cancel_inflight_scope="all")
    steerer = DefaultSteerer(steering_config=cfg, cancel_inflight_scope="user_and_safety")
    assert steerer._cancel_inflight_scope == "user_and_safety"


def test_steerer_rejects_unknown_scope() -> None:
    """An invalid literal falls back to the safe default with a warning
    (never crashes, never silently goes legacy)."""
    steerer = DefaultSteerer(cancel_inflight_scope="bogus")
    assert steerer._cancel_inflight_scope == "user_and_safety"
