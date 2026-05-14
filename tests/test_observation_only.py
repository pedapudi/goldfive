"""Observation-only mode for goldfive steering (goldfive#254).

``SteeringConfig.observation_only`` gates the three actual steering
injection points on :class:`~goldfive.steerer.DefaultSteerer`:

* the would-be revised plan replacing ``session.plan`` in
  :meth:`~goldfive.steerer.DefaultSteerer._apply_revision`;
* the ``GOLDFIVE_STEER`` ControlMessage enqueue in
  :meth:`~goldfive.steerer.DefaultSteerer._dispatch_goldfive_steer_control`;
* the ``request_invocation_cancel`` plugin call in
  :meth:`~goldfive.steerer.DefaultSteerer.request_invocation_cancel`.

Detection still runs in full and ``planner.refine_steer`` still runs —
operators can see what the planner WOULD have produced via the
``PlanRevised`` event with ``dry_run=True``. The in-flight invocation
is otherwise untouched.

The autouse ``_goldfive_active_steering_default`` fixture in
``tests/conftest.py`` flips the implicit default to ``False`` for the
test suite (matching pre-#254 active-steering semantics). Tests in this
file deliberately pass ``observation_only=True`` (or
``observation_only=False``) explicitly — the dataclass honours explicit
kwargs over the fixture's override, so the test's intent wins.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

import goldfive  # noqa: E402
from goldfive.config import (  # noqa: E402
    RuntimeConfig,
    SteeringConfig,
    _resolve_observation_only_default,
)
from goldfive.control import ControlKind  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
from goldfive.state_store import StateStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Lightweight stubs (mirrors of test_judge_task_lifetime.py — kept local
# so this regression bundle is self-contained).
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class RecordingPlanner:
    """Planner stub whose ``refine`` / ``refine_steer`` record every call
    and return a structurally-distinct revised plan so the steerer's
    no-op-revision rejection does not short-circuit before the gate.
    """

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return _revised_plan(kwargs.get("plan"))

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return _revised_plan(kwargs.get("plan"))


class RecordingControlChannel:
    """Minimal control-channel stub: records every ``send`` for assertions."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, msg: Any) -> None:
        self.sent.append(msg)


class RecordingAdapter:
    """Adapter stub whose ``_plugin.request_invocation_cancel`` records
    every call. Used to assert the gated cancel never reaches the plugin.
    """

    class _Plugin:
        def __init__(self, top_invocation_id: str = "inv-live") -> None:
            self.calls: list[dict[str, Any]] = []
            # Read by ``DefaultSteerer._resolve_active_invocation_ids``
            # when the reconciler lookup yields no match — without
            # this the cancel path can't resolve a target id and the
            # "no cancel under observation_only" assertion in the
            # tests below would pass for the wrong reason.
            self._top_invocation_id = top_invocation_id

        def request_invocation_cancel(
            self,
            *,
            invocation_id: str,
            request: Any,
            cancel_inflight_task: bool = False,
        ) -> list[str]:
            self.calls.append(
                {
                    "invocation_id": invocation_id,
                    "request": request,
                    "cancel_inflight_task": cancel_inflight_task,
                }
            )
            return [invocation_id]

    def __init__(self) -> None:
        self._plugin = self._Plugin()

    @property
    def available_agents(self) -> list[str]:  # noqa: D401 — duck-type
        return ["agent"]


def _make_session(run_id: str = "r1", task_id: str = "t1") -> Session:
    task = Task(
        id=task_id, title="Research solar panels", description="Find specs"
    )
    plan = Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


def _revised_plan(prior: Plan | None) -> Plan:
    """Return a structurally-distinct plan derived from ``prior``.

    Adds a new task so the no-op-revision rejection in
    :meth:`_handle_drift` doesn't short-circuit and bypass the gate.
    """
    tasks: list[Task] = list(prior.tasks) if prior is not None else []
    tasks.append(Task(id="t2-corrective", title="Corrective follow-up"))
    return Plan(
        id="p2",
        run_id=(prior.run_id if prior is not None else "r1"),
        goal_ids=(list(prior.goal_ids) if prior is not None else ["g1"]),
        tasks=tasks,
        edges=[],
    )


def _drift_warning(
    *,
    kind: DriftKind = DriftKind.OFF_TOPIC,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    authored_by: str = "goldfive",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail="reasoning judge marked this off-task",
        current_task_id=task_id,
        current_agent_id="agent",
        authored_by=authored_by,
    )


def _plan_revised_events(sink: ListSink) -> list[Any]:
    return [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "plan_revised"
    ]


# ---------------------------------------------------------------------------
# Conftest fixture interaction
# ---------------------------------------------------------------------------


def test_conftest_fixture_flips_implicit_default_to_active() -> None:
    """Inside the test suite, the implicit default is active-steering.

    Sanity check that the autouse
    ``_goldfive_active_steering_default`` fixture in
    ``tests/conftest.py`` is in effect — without it the bulk of the
    existing test corpus would silently skip the injection paths and
    pass for the wrong reason.
    """
    assert _resolve_observation_only_default() is False
    # SteeringConfig() (no explicit kwarg) honours the override.
    assert SteeringConfig().observation_only is False
    # An explicit kwarg still wins over the fixture (test intent
    # beats the fixture's override).
    assert SteeringConfig(observation_only=True).observation_only is True


# ---------------------------------------------------------------------------
# WARNING reasoning-judge drift through ``_handle_drift``
# ---------------------------------------------------------------------------


async def test_warning_drift_observation_only_suppresses_all_three_injections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The full suppression block under ``observation_only=True``.

    Mirror of the shape from ``test_judge_task_lifetime.py``'s
    WARNING-drift coverage. Under observation-only:

    1. ``planner.refine`` is called exactly once (detection + planner
       LLM still run in full).
    2. ``PlanRevised`` is emitted with ``dry_run=True``.
    3. ``session.plan`` is NOT mutated.
    4. No ``GOLDFIVE_STEER`` ControlMessage is enqueued on the bound
       control channel.
    5. The plugin's ``request_invocation_cancel`` is NOT invoked, and
       ``StateStore.cancel_requested_invocation_ids()`` stays
       empty.
    """
    cfg = SteeringConfig(observation_only=True)
    assert cfg.observation_only is True
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    # Register a fake live invocation so the cancel path would have a
    # target to flag IF the gate were open — without this, the "no
    # cancel" half of the assertion below would pass trivially (no live
    # invocation → nothing to cancel regardless of the gate).
    store = StateStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        prior_plan = session.plan

        drift = _drift_warning(kind=DriftKind.OFF_TOPIC)
        with caplog.at_level(logging.INFO, logger="goldfive"):
            await steerer._handle_drift(drift, session)

        # 1. The planner produced a revision exactly once (detection +
        # planner LLM unaffected by the gate). OFF_TOPIC + WARNING is on
        # the goldfive-steer promotion path, so the call lands on
        # ``refine_steer`` rather than ``refine``; either count
        # contributes to the invariant — what matters is "the planner ran
        # once and saw the drift".
        refine_total = len(planner.refine_calls) + len(
            planner.refine_steer_calls
        )
        assert refine_total == 1, (
            "observation_only must NOT skip planner refine/refine_steer; "
            "the operator's whole reason for using it is to see what the "
            f"planner WOULD have produced. Got refine="
            f"{len(planner.refine_calls)} "
            f"refine_steer={len(planner.refine_steer_calls)}"
        )

        # 2. PlanRevised emitted with dry_run=True.
        revised_events = _plan_revised_events(sink)
        assert len(revised_events) == 1, (
            f"PlanRevised must still fire so operators can preview the "
            f"would-have-applied plan; got {len(revised_events)} event(s)"
        )
        assert revised_events[0].plan_revised.dry_run is True, (
            "dry_run must be True so consumers can distinguish a preview "
            "from a real revision"
        )

        # 3. session.plan unchanged.
        assert session.plan is prior_plan, (
            "observation_only must NOT install the revised plan onto "
            "session.plan; the live agent must keep reasoning against "
            "the prior plan"
        )

        # 4. No GOLDFIVE_STEER ControlMessage on the bound channel.
        goldfive_steer_msgs = [
            m
            for m in control.sent
            if getattr(m, "kind", None) is ControlKind.GOLDFIVE_STEER
        ]
        assert goldfive_steer_msgs == [], (
            "observation_only must NOT enqueue GOLDFIVE_STEER; the "
            "executor must keep running the in-flight invocation "
            "without cancel-and-restart"
        )

        # 5. request_invocation_cancel never reached the plugin AND no
        # cancel is registered on StateStore.
        assert adapter._plugin.calls == [], (
            "observation_only must NOT propagate the cancel to the plugin"
        )
        assert store.cancel_requested_invocation_ids() == [], (
            "StateStore must NOT see any cancel-requested "
            "invocations under observation_only"
        )
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)


async def test_warning_drift_active_steering_drives_all_three_injections() -> None:
    """Positive control: ``observation_only=False`` produces the historical behaviour.

    Same scenario as the suppression test but with the gate open: all
    four side effects fire and ``dry_run`` is ``False`` on the emitted
    ``PlanRevised``. Confirms the gate is the only difference — no
    second drift mechanism was disturbed by the refactor.
    """
    cfg = SteeringConfig(observation_only=False)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    # Register a fake live invocation so the cancel resolves a target;
    # without an active invocation ``_resolve_active_invocation_ids``
    # returns empty and the plugin is never called — orthogonal to the
    # observation-only gate and would falsely satisfy the "no cancel"
    # half of the test below.
    store = StateStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        prior_plan = session.plan
        drift = _drift_warning(kind=DriftKind.OFF_TOPIC)
        await steerer._handle_drift(drift, session)

        refine_total = len(planner.refine_calls) + len(
            planner.refine_steer_calls
        )
        assert refine_total == 1
        revised_events = _plan_revised_events(sink)
        assert len(revised_events) == 1
        assert revised_events[0].plan_revised.dry_run is False, (
            "dry_run must be False on a real revision so consumers don't "
            "treat it as a preview"
        )
        # session.plan WAS mutated.
        assert session.plan is not prior_plan
        assert session.plan is not None and session.plan.id == "p2"
        # GOLDFIVE_STEER ControlMessage WAS enqueued.
        goldfive_steer_msgs = [
            m
            for m in control.sent
            if getattr(m, "kind", None) is ControlKind.GOLDFIVE_STEER
        ]
        assert len(goldfive_steer_msgs) == 1
        # request_invocation_cancel DID reach the plugin.
        assert len(adapter._plugin.calls) >= 1
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# ``goldfive.wrap(...)`` threads the flag onto the resulting steerer
# ---------------------------------------------------------------------------


async def _noop_agent(task: Any, session: Any, tools: Any) -> InvocationResult:
    return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


def test_wrap_threads_observation_only_into_steerer() -> None:
    """``RuntimeConfig(steering=SteeringConfig(observation_only=True))`` propagates.

    Confirms the flag lives where the brief promised: on
    ``SteeringConfig`` and reached via ``RuntimeConfig.steering`` —
    no new constructor parameter on ``DefaultSteerer.__init__``,
    ``Runner.__init__``, or ``goldfive.wrap()``.
    """
    cfg = RuntimeConfig(steering=SteeringConfig(observation_only=True))
    runner = goldfive.wrap(_noop_agent, runtime=cfg, sinks=[])
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._observation_only is True
    assert steerer._should_inject() is False

    cfg2 = RuntimeConfig(steering=SteeringConfig(observation_only=False))
    runner2 = goldfive.wrap(_noop_agent, runtime=cfg2, sinks=[])
    steerer2 = runner2.steerer
    assert isinstance(steerer2, DefaultSteerer)
    assert steerer2._observation_only is False
    assert steerer2._should_inject() is True


# ---------------------------------------------------------------------------
# Env-var surface for ``SteeringConfig.observation_only``
# ---------------------------------------------------------------------------


def test_steering_config_from_env_observation_only(
    goldfive_steer_env: Any,
) -> None:
    """``GOLDFIVE_STEER_OBSERVATION_ONLY`` flips the field via ``from_env``.

    Coverage matrix:

    * ``"0"`` / ``"false"`` / ``"no"`` / ``"off"`` -> False.
    * ``"1"`` / ``"true"`` / ``"yes"`` / ``"on"`` -> True.
    * Unset -> the effective default. In the test-suite fixture path
      that's ``False`` (the autouse fixture overrides the implicit
      default); a test that suppresses the fixture would see the
      production ``True`` default.
    """
    for falsey in ("0", "false", "no", "off"):
        goldfive_steer_env.set(observation_only=falsey)
        assert SteeringConfig.from_env().observation_only is False

    for truthy in ("1", "true", "yes", "on"):
        goldfive_steer_env.set(observation_only=truthy)
        assert SteeringConfig.from_env().observation_only is True

    # Unset: in the suite, the autouse fixture overrides to False.
    goldfive_steer_env.unset("observation_only")
    assert SteeringConfig.from_env().observation_only is False


def test_steering_config_from_env_production_default(
    monkeypatch: pytest.MonkeyPatch,
    goldfive_steer_env: Any,
) -> None:
    """Outside the suite override, the production default is ``True``.

    Suppresses the autouse fixture's override for this test (by
    flipping the module-level hook back to ``None``) so the unset env
    path returns the production default. Restored on exit.
    """
    from goldfive import config as _gf_config

    goldfive_steer_env.unset("observation_only")
    prior = _gf_config._OBSERVATION_ONLY_DEFAULT
    _gf_config._OBSERVATION_ONLY_DEFAULT = None
    try:
        assert SteeringConfig().observation_only is True
        assert SteeringConfig.from_env().observation_only is True
    finally:
        _gf_config._OBSERVATION_ONLY_DEFAULT = prior


# ---------------------------------------------------------------------------
# CAPABILITY_MISMATCH drift routes through the gated ``_handle_drift``
# ---------------------------------------------------------------------------


async def test_capability_mismatch_under_observation_only_suppresses_injections() -> (
    None
):
    """A CRITICAL ``CAPABILITY_MISMATCH`` (the kind ``_adk_plugin.py``
    fires at delegation-observed time, goldfive#253) flows through the
    same gated ``_handle_drift`` path: no cancel, no plan mutation.

    The ADK adapter's ``_maybe_emit_capability_mismatch`` builds a
    DriftEvent and calls ``steerer._handle_drift(drift, session)``
    directly (see ``goldfive/adapters/_adk_plugin.py``). We exercise the
    steerer side of that contract with a hand-built drift so the test
    is independent of the ADK adapter and runs without the ``google.adk``
    extra.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    prior_plan = session.plan
    drift = _drift_warning(
        kind=DriftKind.CAPABILITY_MISMATCH,
        severity=DriftSeverity.CRITICAL,
    )
    await steerer._handle_drift(drift, session)

    # Plan was NOT mutated.
    assert session.plan is prior_plan
    # No cancel propagated to the plugin or the StateStore.
    assert adapter._plugin.calls == []
    store = StateStore.for_session(session)
    assert store.cancel_requested_invocation_ids() == []
    # No GOLDFIVE_STEER ControlMessage on the bound channel.
    assert [
        m
        for m in control.sent
        if getattr(m, "kind", None) is ControlKind.GOLDFIVE_STEER
    ] == []


# ---------------------------------------------------------------------------
# Negative regression: INFO drift that wouldn't promote anyway
# ---------------------------------------------------------------------------


async def test_info_drift_below_promotion_threshold_unchanged_by_flag() -> None:
    """An INFO drift that wouldn't promote anyway sees no injections,
    regardless of the flag's value.

    The intervention ladder maps most INFO drifts to OBSERVE (no
    refine, no cancel, no plan mutation). This invariant is unrelated
    to observation_only; the test guards against accidentally
    promoting INFO drifts when reorganising the gate.
    """
    for observation_only in (True, False):
        cfg = SteeringConfig(observation_only=observation_only)
        steerer = DefaultSteerer(steering_config=cfg)
        session = _make_session()
        sink = ListSink()
        planner = RecordingPlanner()
        control = RecordingControlChannel()
        adapter = RecordingAdapter()
        steerer.bind(sinks=[sink], planner=planner)
        steerer.bind_control_channel(control)
        steerer.bind_adapter(adapter)

        prior_plan = session.plan
        drift = _drift_warning(
            kind=DriftKind.CONFABULATION_RISK,
            severity=DriftSeverity.INFO,
        )
        await steerer._handle_drift(drift, session)

        # OBSERVE: no refine, no plan revision, no cancel, no steer.
        assert planner.refine_calls == [], (
            f"INFO drift must not refine (observation_only={observation_only})"
        )
        assert _plan_revised_events(sink) == [], (
            f"INFO drift must not emit PlanRevised "
            f"(observation_only={observation_only})"
        )
        assert session.plan is prior_plan
        assert adapter._plugin.calls == []
        store = StateStore.for_session(session)
        assert store.cancel_requested_invocation_ids() == []


# ---------------------------------------------------------------------------
# Integration-shape: drive observe_reasoning end-to-end with a WARNING
# on_task=False verdict.
# ---------------------------------------------------------------------------


async def test_observe_reasoning_warning_verdict_under_observation_only() -> None:
    """End-to-end through the background reasoning judge.

    ``observe_reasoning`` spawns the fire-and-forget judge; the judge
    calls back into ``_handle_drift``. Under observation_only the same
    suppression block applies: refine runs, PlanRevised emits with
    dry_run=True, session.plan is unchanged, no cancel propagates.
    """

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps(
            {
                "on_task": False,
                "severity": "warning",
                "reason": "drifted to raccoons",
            }
        )

    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        steering_config=cfg,
    )
    session = _make_session()
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    # Register a fake live invocation so the late-drift guard doesn't
    # short-circuit (the guard would otherwise drop the verdict before
    # it reaches the observation-only gate).
    store = StateStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        prior_plan = session.plan
        await steerer.observe_reasoning("raccoons are nocturnal", session=session)
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)

        # Detection ran -> planner ran exactly once (route depends on
        # drift kind / severity — refine vs refine_steer; either way
        # the invariant is "the planner saw the drift").
        refine_total = len(planner.refine_calls) + len(planner.refine_steer_calls)
        assert refine_total == 1, (
            f"got refine={len(planner.refine_calls)} "
            f"refine_steer={len(planner.refine_steer_calls)}"
        )

        # PlanRevised emitted with dry_run=True.
        revised_events = _plan_revised_events(sink)
        assert len(revised_events) == 1
        assert revised_events[0].plan_revised.dry_run is True

        # session.plan unchanged.
        assert session.plan is prior_plan

        # Adapter plugin saw no cancel; StateStore has no
        # cancel-pending entry.
        assert adapter._plugin.calls == []
        assert store.cancel_requested_invocation_ids() == []

        # No GOLDFIVE_STEER ControlMessage on the bound channel.
        assert [
            m
            for m in control.sent
            if getattr(m, "kind", None) is ControlKind.GOLDFIVE_STEER
        ] == []
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# goldfive#255 carve-outs: bootstrap install + user-authored revision must
# always land, even under observation_only=True. Only goldfive-authored
# corrective drifts on an already-bootstrapped session are gated.
# ---------------------------------------------------------------------------


def _bootstrap_session(run_id: str = "r-bootstrap") -> Session:
    """Build a fresh session with ``session.plan = None``.

    Mirrors the live reproduction in goldfive#255: bootstrap install
    enters ``_apply_revision`` with ``prev = session.plan = None`` and
    the pre-fix gate left ``session.plan`` permanently ``None`` under
    observation_only=True — detectors collapsed, harmonograf rendered
    an empty DAG.
    """
    return Session(
        run_id=run_id,
        goals=[Goal(id="g-bootstrap", summary="Publish a memo on solar panels")],
        plan=None,
        current_task_id="",
    )


def _initial_plan(run_id: str = "r-bootstrap") -> Plan:
    """Build a structurally-valid initial plan for the bootstrap test."""
    return Plan(
        id="p-bootstrap",
        run_id=run_id,
        goal_ids=["g-bootstrap"],
        tasks=[Task(id="t-bootstrap", title="Research solar panels")],
        edges=[],
    )


async def test_bootstrap_install_lands_under_observation_only() -> None:
    """goldfive#255: ``session.plan = None`` install must land even with
    ``observation_only=True``.

    Bootstrap is structural, not corrective — the gate carve-out for
    ``prev is None`` keeps the very first install of a session's plan
    real. Without this, ``session.plan`` would stay ``None`` forever,
    detectors would collapse, and harmonograf's task_plans dispatch
    would never see authoritative content (the live reproduction in
    session ``62dde1a6``).
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _bootstrap_session()
    assert session.plan is None
    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    initial = _initial_plan()
    installed = await steerer.install_initial_plan(session=session, plan=initial)
    assert installed is True, "bootstrap must succeed"

    # session.plan was set (non-None) post-install.
    assert session.plan is not None, (
        "bootstrap install must land session.plan even under observation_only; "
        "the gate carve-out is the goldfive#255 fix"
    )
    assert session.plan.id == initial.id

    # PlanRevised emitted with dry_run=False (real revision, not preview).
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False on a bootstrap install — operators should "
        "see the install as authoritative, not as a would-have-applied "
        "preview"
    )

    # Watermark behaviour: the bootstrap placeholder is goldfive-authored
    # (kind=NEW_WORK_DISCOVERED, authored_by="goldfive") so the existing
    # stamping site at the end of ``_apply_revision`` will write the
    # ("new_work_discovered", "") key — the placeholder has empty
    # current_task_id. This is innocuous (the key collides with no real
    # drift verdict on the freshness gate), but we assert the install
    # itself worked rather than relying on watermark absence.
    # The brief asked to "verify this still behaves correctly" — the
    # behaviour we care about is the install landing. Watermark stamping
    # is a secondary effect we leave under the existing
    # authored_by != "user" rule.


async def test_user_authored_revision_lands_under_observation_only() -> None:
    """goldfive#255: a USER_STEER revision must land even with
    ``observation_only=True``.

    The carve-out for ``drift.authored_by == "user"`` keeps operator
    STEER ControlMessage deliveries authoritative even when goldfive
    is configured to passively observe its own drifts.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    prior_plan = session.plan
    assert prior_plan is not None
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    # Build a structurally-distinct revised plan (avoids the no-op
    # revision short-circuit in ``_install_with_drift``).
    revised = _revised_plan(prior_plan)

    # Drive the canonical user-authored install path. ``raw=None`` is
    # accepted: ``_unpack_steer_context`` falls back to drift.detail
    # when raw is absent.
    installed = await steerer.install_revision_for_user_steer(
        session=session,
        raw=None,
        revised_plan=revised,
    )
    assert installed is True, "user-authored revision must install"

    # session.plan mutated to the revised plan.
    assert session.plan is not prior_plan, (
        "user-authored revision must land session.plan even under "
        "observation_only — operator directives override the gate"
    )
    assert session.plan is not None and session.plan.id == revised.id

    # PlanRevised emitted with dry_run=False — operator action is
    # authoritative, not a preview.
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False for a user-authored revision; an operator "
        "STEER is never a preview"
    )


async def test_goldfive_corrective_drift_is_gated_under_observation_only() -> None:
    """goldfive#255 regression: OFF_TOPIC (goldfive-authored, non-bootstrap)
    MUST stay gated under ``observation_only=True``.

    This is the path the refactor must NOT accidentally open. The
    carve-outs in ``_apply_revision`` are narrow — only bootstrap and
    user-authored drifts bypass the gate. A goldfive-authored
    corrective drift on a session with a non-None plan stays gated:
    no install, dry_run=True on the emitted PlanRevised.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    prior_plan = session.plan
    assert prior_plan is not None
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    drift = _drift_warning(
        kind=DriftKind.OFF_TOPIC,
        authored_by="goldfive",
    )
    await steerer._handle_drift(drift, session)

    # session.plan unchanged — gate held.
    assert session.plan is prior_plan, (
        "goldfive-authored corrective drift must NOT land session.plan "
        "under observation_only; the carve-outs apply only to bootstrap "
        "and user-authored revisions"
    )

    # PlanRevised emitted with dry_run=True (preview only).
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1, (
        "PlanRevised must still emit so operators see the would-have-"
        f"applied plan; got {len(revised_events)} event(s)"
    )
    assert revised_events[0].plan_revised.dry_run is True, (
        "dry_run must be True for a gated corrective drift — it's a "
        "preview, not a real revision"
    )


async def test_goldfive_corrective_drift_lands_with_observation_only_false() -> None:
    """goldfive#255 positive control: with ``observation_only=False`` the
    same corrective drift lands.

    Toggling the flag off removes ALL three sources of gating
    (``gate_active`` becomes False because ``not self._should_inject()``
    is False). The refactor must not introduce a hidden gate that
    survives the flag flip.
    """
    cfg = SteeringConfig(observation_only=False)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    prior_plan = session.plan
    assert prior_plan is not None
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    drift = _drift_warning(
        kind=DriftKind.OFF_TOPIC,
        authored_by="goldfive",
    )
    await steerer._handle_drift(drift, session)

    # session.plan mutated to the revised plan.
    assert session.plan is not prior_plan, (
        "with observation_only=False the corrective drift must land "
        "session.plan"
    )
    assert session.plan is not None and session.plan.id == "p2"

    # PlanRevised emitted with dry_run=False (real revision).
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False on a real revision so consumers don't "
        "treat it as a preview"
    )


# ---------------------------------------------------------------------------
# goldfive#258 carve-out: NEW_WORK_DISCOVERED revisions always land
# ---------------------------------------------------------------------------


async def test_discovery_revision_lands_under_observation_only() -> None:
    """goldfive#258: a goldfive-authored ``NEW_WORK_DISCOVERED`` drift on
    an already-bootstrapped session must land as a real revision under
    ``observation_only=True``.

    Discovery represents observed/described work (the planner integrated
    new work from the user's next-turn message; a sub-agent reported
    new work via ``report_new_work_discovered``) — NOT a framework-driven
    correction. The pre-#258 gate fired here because ``prev`` is non-None
    (an existing plan is on the session) and ``authored_by`` is
    ``"goldfive"``, leaving the discovery as a dry-run preview.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    prior_plan = session.plan
    assert prior_plan is not None
    sink = ListSink()
    planner = RecordingPlanner()
    control = RecordingControlChannel()
    adapter = RecordingAdapter()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)
    steerer.bind_adapter(adapter)

    # Build a structurally-distinct revised plan so the no-op revision
    # rejection in ``_install_with_drift`` doesn't short-circuit before
    # the gate decision.
    revised = _revised_plan(prior_plan)
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="planner integrated new work from the user's next turn",
        authored_by="goldfive",
    )

    installed = await steerer.install_revision_for_drift(
        session=session,
        drift=drift,
        revised_plan=revised,
    )
    assert installed is True, "discovery revision must install"

    # session.plan WAS mutated to the revised plan.
    assert session.plan is not prior_plan, (
        "NEW_WORK_DISCOVERED revision must land session.plan even under "
        "observation_only=True — discovery is a description of observed "
        "work, not a corrective intervention. goldfive#258 carve-out."
    )
    assert session.plan is not None and session.plan.id == revised.id

    # PlanRevised emitted with dry_run=False (real revision, not preview).
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False for a NEW_WORK_DISCOVERED revision under "
        "observation_only — discovery is authoritative, not a preview"
    )


async def test_discovery_revision_lands_under_observation_only_initial_plan_seed() -> (
    None
):
    """goldfive#258 live-bug path: turn-1 ``install_initial_plan`` with a
    pre-seeded ``Plan.empty()`` must still land under observation_only.

    The runner seeds ``session.plan = Plan.empty(run_id=...)`` before
    :meth:`Planner.handle_turn` runs (so the planner always sees a
    non-None ``session.plan``). By the time
    :meth:`install_initial_plan`'s placeholder
    ``NEW_WORK_DISCOVERED`` drift reaches ``_apply_revision``, ``prev``
    is that empty seed — **not** ``None`` — so the original #255
    ``prev is None`` carve-out did NOT match.

    Pre-#258 reproduction (live session ``793ef00b``, gemma-4-26B):

        DefaultSteerer._apply_revision: observation_only=True —
        SKIPPING plan install (gate_active; bootstrap=False user=False).
        ... revision_index=1 drift_kind=new_work_discovered

    The fresh discovery drift on revision 1 was rendered as a dry-run,
    leaving ``session.plan`` permanently the empty seed. This test is
    the regression guard.
    """
    from goldfive.types import channel_processor_active, set_session_plan

    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)

    # Build the live runner's turn-1 state: session.plan is the
    # ``Plan.empty()`` seed (non-None, but tasks empty + revision_index 0).
    session = Session(
        run_id="r-seed",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=None,
        current_task_id="",
    )
    with channel_processor_active():
        set_session_plan(session, Plan.empty(run_id="r-seed"))
    seed = session.plan
    assert seed is not None, "Plan.empty() seed should populate session.plan"
    assert seed.tasks == (), "empty seed must have no tasks"

    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # The runner builds a real plan via Planner.handle_turn then routes
    # through install_initial_plan with first_turn=True (because
    # session.plan.tasks is empty). install_initial_plan synthesises a
    # NEW_WORK_DISCOVERED placeholder for _apply_revision metadata
    # stamping — the same placeholder that fires in the live bug.
    initial = Plan(
        id="p-initial",
        run_id="r-seed",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Research solar panels")],
        edges=[],
    )
    installed = await steerer.install_initial_plan(session=session, plan=initial)
    assert installed is True, "initial plan install must succeed"

    # session.plan was swapped from the empty seed onto the real plan.
    assert session.plan is not seed, (
        "install_initial_plan must swap the empty seed for the real plan "
        "even under observation_only — goldfive#258 regression guard"
    )
    assert session.plan is not None and session.plan.id == initial.id
    assert session.plan.tasks, (
        "post-install session.plan must carry the planner-authored tasks; "
        "the empty seed leaking past the install is the goldfive#258 bug"
    )

    # PlanRevised emitted with dry_run=False — the install is
    # authoritative, not a preview.
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "dry_run must be False on the turn-1 install under observation_only "
        "— the empty-seed -> real-plan transition is a discovery, not a "
        "corrective intervention. goldfive#258."
    )


async def test_discovery_via_install_revision_for_drift_under_observation_only() -> None:
    """goldfive#258: turn N+1 replan path with ``NEW_WORK_DISCOVERED``.

    Mirrors the runner's :meth:`Runner._install_revision` non-first-turn
    branch (``runner.py`` ≈ line 1681): the user's fresh message caused
    ``Planner.handle_turn`` to surface new work, and the runner installs
    via ``install_revision_for_drift`` with
    ``DriftKind.NEW_WORK_DISCOVERED, authored_by="goldfive"``. Under
    observation_only this must still produce a real revision — the
    discovery is a description of observed work, not a correction.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _make_session()
    prior_plan = session.plan
    assert prior_plan is not None
    sink = ListSink()
    planner = RecordingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    revised = _revised_plan(prior_plan)
    drift = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.INFO,
        detail="the user asked a follow-up that surfaced new work",
        authored_by="goldfive",
    )
    installed = await steerer.install_revision_for_drift(
        session=session,
        drift=drift,
        revised_plan=revised,
    )
    assert installed is True

    # session.plan was mutated.
    assert session.plan is not prior_plan
    assert session.plan is not None and session.plan.id == revised.id

    # dry_run False on the emitted PlanRevised — the install is real.
    revised_events = _plan_revised_events(sink)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False
