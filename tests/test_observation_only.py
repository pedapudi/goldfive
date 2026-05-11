"""Coverage for the observation-only mode (goldfive#254).

When ``observation_only=True`` is wired onto ``DefaultSteerer`` /
:class:`Runner` / :func:`goldfive.wrap`, the steerer:

* still runs every detector (judges, drift classifiers);
* still calls ``planner.refine`` / ``refine_steer`` (operators see what
  would have been steered);
* still emits ``DriftDetected`` AND ``PlanRevised`` events on the sink
  bus (the latter with ``dry_run=True`` so harmonograf renders the
  would-have-applied preview);

but the three injection sites are suppressed:

1. ``session.plan`` is NOT mutated by :meth:`DefaultSteerer._apply_revision`.
2. No ``GOLDFIVE_STEER`` ControlMessage is enqueued onto the bound
   control channel.
3. :meth:`DefaultSteerer.request_invocation_cancel` returns ``[]``
   without writing to the cancel-pending registry.

Note: every test here that exercises the observation_only behaviour
**explicitly** passes ``observation_only=True`` to its DefaultSteerer /
Runner / wrap call — the autouse fixture in :mod:`tests.conftest`
only governs the unspecified default, so explicit-True wins.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import ControlChannel  # noqa: E402
from goldfive.orchestration_store import OrchestrationStore  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Stubs (small, local, no dependency on production stubs)
# ---------------------------------------------------------------------------


class _ListSink:
    """Sink that records every emitted event in-order."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class _RecordingPlanner:
    """Records every ``refine`` / ``refine_steer`` call and returns a fixed plan."""

    def __init__(self, revised: Plan | None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_steer_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised

    async def refine_steer(self, **kwargs: Any) -> Plan | None:
        self.refine_steer_calls.append(kwargs)
        return self.revised

    @property
    def total_refine_calls(self) -> int:
        """Combined count of ``refine`` + ``refine_steer`` calls.

        Goldfive routes some drift kinds through ``refine_steer`` (the
        promotion path) and others through ``refine`` (the regular
        ladder). For the observation-only tests we care about "did
        the planner round-trip happen at all", not which entry point.
        """
        return len(self.refine_calls) + len(self.refine_steer_calls)


class _RecordingCancelPlugin:
    """ADK-like plugin stub that records cancel-flag writes.

    Exposes ``_top_invocation_id`` so
    :meth:`DefaultSteerer._resolve_active_invocation_ids` finds a target
    id; without it, the steerer's empty-invocation guard fires and the
    test can't distinguish "cancel was gated" from "cancel had nothing
    to target".
    """

    def __init__(self, top_invocation_id: str = "inv-live") -> None:
        self._top_invocation_id = top_invocation_id
        self.cancel_calls: list[dict[str, Any]] = []

    def request_invocation_cancel(
        self,
        *,
        invocation_id: str,
        request: Any,
        cancel_inflight_task: bool = False,
    ) -> list[str]:
        self.cancel_calls.append(
            {
                "invocation_id": invocation_id,
                "request": request,
                "cancel_inflight_task": cancel_inflight_task,
            }
        )
        return [invocation_id]


class _RecordingAdapter:
    """Adapter stub exposing a plugin and a next-cancel-reason slot."""

    def __init__(self) -> None:
        self._plugin = _RecordingCancelPlugin()
        self._next_cancel_reason: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_initial_plan(run_id: str = "r1", task_id: str = "t1") -> Plan:
    return Plan(
        id="p-initial",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[
            Task(id=task_id, title="Initial task", description="Research solar panels"),
        ],
        edges=[],
    )


def _make_revised_plan(run_id: str = "r1") -> Plan:
    """Build a revised plan whose shape is valid against the initial.

    The initial plan has t1 PENDING; we keep t1 PENDING and add a
    sibling t2 PENDING (no edges between them) so the validator
    accepts the revision both as ``for_revision=True`` (the new sub-
    DAG forms from no predecessor of a cancelled task) and as
    structurally distinct (different task count → not a no-op).
    """
    return Plan(
        id="p-initial",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Initial task", description="Research solar panels"),
            Task(id="t2", title="Replanned task", description="Pivot focus"),
        ],
        edges=[],
        revision_index=1,
    )


def _make_session(run_id: str = "r1", task_id: str = "t1") -> Session:
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=_make_initial_plan(run_id, task_id),
        current_task_id=task_id,
    )


def _drift_events(events: list[Any]) -> list[Any]:
    return [
        e for e in events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]


def _plan_revised_events(events: list[Any]) -> list[Any]:
    return [
        e for e in events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "plan_revised"
    ]


async def _judge_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
    return json.dumps(
        {
            "on_task": False,
            "severity": "warning",
            "reason": "drifted to raccoons",
        }
    )


async def _register_live_invocation(
    session: Session, store: OrchestrationStore
) -> asyncio.Task[None]:
    """Register a long-running placeholder task so the late-drift gate
    sees a live invocation."""

    async def _body() -> None:
        await asyncio.sleep(5.0)

    fake = asyncio.create_task(_body())
    store.register_invocation_task("inv-live", fake)
    return fake


# ---------------------------------------------------------------------------
# observation_only=True end-to-end (reasoning-judge drift)
# ---------------------------------------------------------------------------


async def test_observation_only_judge_drift_suppresses_three_injection_points() -> None:
    """A WARNING reasoning-judge drift in observation-only mode:

    * still emits ``DriftDetected`` on the sink (detector ran);
    * still calls ``planner.refine`` (LLM round-trip happened);
    * emits ``PlanRevised`` with ``dry_run=True`` (sink preview);
    * leaves ``session.plan`` unchanged;
    * does NOT enqueue a GOLDFIVE_STEER ControlMessage;
    * does NOT write to the cancel-pending registry.
    """
    session = _make_session()
    sink = _ListSink()
    planner = _RecordingPlanner(revised=_make_revised_plan())
    control = ControlChannel()
    adapter = _RecordingAdapter()

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=_judge_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        observation_only=True,
    )
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)
    steerer.bind_control_channel(control)
    assert steerer.observation_only is True

    # Register a live invocation so the late-drift gate does NOT fire —
    # we want the judge verdict to reach _handle_drift and exercise all
    # three injection sites.
    store = OrchestrationStore.for_session(session)
    fake_task = await _register_live_invocation(session, store)

    pre_plan = session.plan
    pre_plan_id = id(pre_plan)
    pre_revision = pre_plan.revision_index

    # Snapshot the channel's inbox depth before the judge runs.
    pre_inbox_depth = control._inbox.qsize()

    try:
        await steerer.observe_reasoning("raccoons are nocturnal", session=session)
        # Drain background judge tasks.
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)
        await steerer._wait_background_drifts_idle()
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    # 1. DriftDetected emitted (detector ran).
    drift_events = _drift_events(sink.events)
    assert len(drift_events) >= 1, (
        f"observation-only must NOT suppress DriftDetected; got "
        f"{len(drift_events)} drift event(s)"
    )

    # 2. planner.refine / refine_steer was called (LLM round-trip happened).
    # OFF_TOPIC WARNING routes through ``_promote_drift_to_steer`` →
    # ``refine_steer``; the assertion is on the combined count so the
    # contract is "did the planner round-trip at all".
    assert planner.total_refine_calls == 1, (
        f"observation-only must NOT suppress planner.refine / refine_steer; "
        f"got refine={len(planner.refine_calls)} "
        f"refine_steer={len(planner.refine_steer_calls)}"
    )

    # 3. PlanRevised emitted with dry_run=True.
    revised_events = _plan_revised_events(sink.events)
    assert len(revised_events) == 1, (
        f"observation-only must emit PlanRevised so sinks render a "
        f"preview; got {len(revised_events)} plan_revised event(s)"
    )
    assert revised_events[0].plan_revised.dry_run is True, (
        "PlanRevised.dry_run must be True in observation-only mode"
    )

    # 4. session.plan unchanged.
    assert session.plan is pre_plan, (
        "observation-only must NOT replace session.plan in-place"
    )
    assert id(session.plan) == pre_plan_id
    assert session.plan.revision_index == pre_revision

    # 5. No GOLDFIVE_STEER ControlMessage on the channel.
    assert control._inbox.qsize() == pre_inbox_depth, (
        "observation-only must NOT enqueue a GOLDFIVE_STEER ControlMessage"
    )

    # 6. No cancel-pending entry in the OrchestrationStore.
    assert store.cancel_requested_invocation_ids() == [], (
        "observation-only must NOT write to the cancel-pending registry"
    )
    # 6b. Belt+braces: the plugin's cancel hook was never called.
    assert adapter._plugin.cancel_calls == [], (
        "observation-only must not invoke plugin.request_invocation_cancel"
    )


async def test_active_steering_judge_drift_fires_all_three_injection_points() -> None:
    """Positive-control twin of the test above: ``observation_only=False``
    means every side effect happens.

    Acts as a regression guard so a future refactor that breaks the gate
    can't quietly pass by also breaking the positive path.
    """
    session = _make_session()
    sink = _ListSink()
    planner = _RecordingPlanner(revised=_make_revised_plan())
    control = ControlChannel()
    adapter = _RecordingAdapter()

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=_judge_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        observation_only=False,
    )
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)
    steerer.bind_control_channel(control)
    assert steerer.observation_only is False

    store = OrchestrationStore.for_session(session)
    fake_task = await _register_live_invocation(session, store)
    try:
        await steerer.observe_reasoning("raccoons are nocturnal", session=session)
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)
        await steerer._wait_background_drifts_idle()
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    # PlanRevised emitted with dry_run=False (default).
    revised_events = _plan_revised_events(sink.events)
    assert len(revised_events) == 1
    assert revised_events[0].plan_revised.dry_run is False, (
        "active steering must emit PlanRevised with dry_run=False"
    )

    # session.plan WAS replaced — revision_index bumped.
    assert session.plan is not None
    assert session.plan.revision_index >= 1

    # planner.refine / refine_steer was still called.
    assert planner.total_refine_calls == 1

    # cancel-pending entry written (active cancel fired).
    assert "inv-live" in store.cancel_requested_invocation_ids()


# ---------------------------------------------------------------------------
# "Flag wins ties" — Runner observation_only ignored when caller pre-builds
# a Steerer
# ---------------------------------------------------------------------------


async def test_runner_observation_only_ignored_when_explicit_steerer_passed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the caller passes a pre-built Steerer, the Runner's
    ``observation_only`` parameter is ignored (the steerer's own flag
    wins) and an INFO log line records the mismatch.
    """
    import logging

    from goldfive.adapters.callable import CallableAdapter
    from goldfive.executors.sequential import SequentialExecutor
    from goldfive.planner import PassthroughPlanner
    from goldfive.runner import Runner

    pre_built = DefaultSteerer(observation_only=True)
    adapter = CallableAdapter(
        _trivial_agent_callable, available_agents=["default"]
    )

    with caplog.at_level(logging.INFO, logger="goldfive.runner"):
        runner = Runner(
            agent=adapter,
            planner=PassthroughPlanner(),
            executor=SequentialExecutor(),
            steerer=pre_built,
            observation_only=False,
        )

    assert runner.steerer is pre_built
    assert pre_built.observation_only is True, (
        "the pre-built steerer's flag must NOT be silently overwritten "
        "by Runner.__init__"
    )
    # The Runner stamps the parameter it was given, not the steerer's
    # resolved flag. Keep this distinction so debugging is precise.
    assert runner.observation_only is False
    # INFO log records the mismatch.
    info_records = [
        r
        for r in caplog.records
        if r.name == "goldfive.runner"
        and r.levelno == logging.INFO
        and "observation_only" in r.getMessage()
        and "steerer's flag wins" in r.getMessage()
    ]
    assert len(info_records) == 1, (
        f"Runner must log the observation_only mismatch at INFO; got "
        f"{len(info_records)} matching record(s)"
    )


# ---------------------------------------------------------------------------
# goldfive.wrap forwards observation_only into the default steerer
# ---------------------------------------------------------------------------


async def _trivial_agent_callable(task: Task, session: Session, tools: list[Any]) -> Any:  # noqa: ARG001
    from goldfive.results import InvocationResult

    return InvocationResult(summary="ok", artifacts={})


async def test_wrap_forwards_observation_only_to_default_steerer() -> None:
    """``goldfive.wrap(tree, observation_only=True)`` results in a Runner
    whose default steerer is in observation-only mode.

    Uses a CallableAdapter (no ADK dependency) so the test is fast.
    """
    from goldfive.adapters.callable import CallableAdapter
    from goldfive.convenience import wrap

    handler_adapter = CallableAdapter(
        _trivial_agent_callable, available_agents=["default"]
    )
    runner = wrap(handler_adapter, observation_only=True)

    assert getattr(runner.steerer, "observation_only", None) is True, (
        "wrap() must forward observation_only into the default steerer"
    )
    assert runner.observation_only is True


async def test_wrap_default_is_observation_only_in_production_resolution() -> None:
    """In production resolution, the wrap default is True.

    The test suite's autouse fixture flips
    :data:`goldfive.steerer._test_default_observation_only` to ``False``
    so the existing pre-#254 active-steering tests stay green. To
    exercise the production default we temporarily restore it here.
    """
    from goldfive import steerer as steerer_mod
    from goldfive.adapters.callable import CallableAdapter
    from goldfive.convenience import wrap

    handler_adapter = CallableAdapter(
        _trivial_agent_callable, available_agents=["default"]
    )

    prior = steerer_mod._test_default_observation_only
    steerer_mod._test_default_observation_only = None
    try:
        runner = wrap(handler_adapter)  # no explicit observation_only
    finally:
        steerer_mod._test_default_observation_only = prior

    assert getattr(runner.steerer, "observation_only", None) is True, (
        "the production wrap default must be observation_only=True"
    )


# ---------------------------------------------------------------------------
# Negative regression: a drift kind that wouldn't have steered anyway
# ---------------------------------------------------------------------------


async def test_observation_only_does_not_change_passive_drift_kinds() -> None:
    """A filtered-out reasoning verdict (``on_task=True``) emits no
    drift, no refine, no PlanRevised — in BOTH modes.

    Confirms the gate is targeted: it only suppresses INJECTION, never
    promotes a no-op into a side effect.
    """

    async def on_task_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"on_task": True})

    for obs in (True, False):
        session = _make_session()
        sink = _ListSink()
        planner = _RecordingPlanner(revised=_make_revised_plan())
        steerer = DefaultSteerer(
            reasoning_drift_call_llm=on_task_call_llm,
            reasoning_drift_model="fake",
            reasoning_drift_mode="judge",
            observation_only=obs,
        )
        steerer.bind(sinks=[sink], planner=planner)
        await steerer.observe_reasoning("totally on topic", session=session)
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)
        await steerer._wait_background_drifts_idle()

        assert _drift_events(sink.events) == [], (
            f"on-task verdict must not emit DriftDetected (obs={obs})"
        )
        assert planner.total_refine_calls == 0, (
            f"on-task verdict must not call planner.refine (obs={obs})"
        )
        assert _plan_revised_events(sink.events) == [], (
            f"on-task verdict must not emit PlanRevised (obs={obs})"
        )


# ---------------------------------------------------------------------------
# Proto round-trip: dry_run serialises through bytes correctly
# ---------------------------------------------------------------------------


def test_plan_revised_dry_run_proto_round_trip() -> None:
    """The new ``dry_run`` field round-trips through proto serialisation."""
    from goldfive.pb.goldfive.v1 import events_pb2

    evt = events_pb2.Event()
    evt.plan_revised.dry_run = True
    evt.plan_revised.reason = "would-have-applied"
    data = evt.SerializeToString()

    parsed = events_pb2.Event()
    parsed.ParseFromString(data)
    assert parsed.plan_revised.dry_run is True
    assert parsed.plan_revised.reason == "would-have-applied"

    # Default is False — confirm a freshly-built envelope keeps the
    # pre-#254 wire shape.
    evt2 = events_pb2.Event()
    evt2.plan_revised.reason = "live revision"
    data2 = evt2.SerializeToString()
    parsed2 = events_pb2.Event()
    parsed2.ParseFromString(data2)
    assert parsed2.plan_revised.dry_run is False


# ---------------------------------------------------------------------------
# Integration shape: observe_reasoning end-to-end with all suppressions
# verified in a single block.
# ---------------------------------------------------------------------------


async def test_observation_only_integration_full_suppression_block() -> None:
    """End-to-end: a WARNING reasoning-judge ``on_task=False`` verdict
    drives the full pipeline; every observation-only suppression is
    asserted in one block so a regression failing the contract trips
    here even if the unit-level tests above drift.
    """
    session = _make_session()
    sink = _ListSink()
    planner = _RecordingPlanner(revised=_make_revised_plan())
    control = ControlChannel()
    adapter = _RecordingAdapter()

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=_judge_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        observation_only=True,
    )
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_adapter(adapter)
    steerer.bind_control_channel(control)

    pre_plan = session.plan

    store = OrchestrationStore.for_session(session)
    fake_task = await _register_live_invocation(session, store)
    try:
        await steerer.observe_reasoning(
            "raccoons are nocturnal", session=session
        )
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)
        await steerer._wait_background_drifts_idle()
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    drift_events = _drift_events(sink.events)
    revised_events = _plan_revised_events(sink.events)

    # One single assertion block summarising the entire contract.
    assert all(
        [
            len(drift_events) >= 1,
            planner.total_refine_calls == 1,
            len(revised_events) == 1,
            revised_events[0].plan_revised.dry_run is True,
            session.plan is pre_plan,
            control._inbox.qsize() == 0,
            store.cancel_requested_invocation_ids() == [],
            adapter._plugin.cancel_calls == [],
        ]
    ), (
        "observation_only contract violated; "
        f"drift_events={len(drift_events)} "
        f"refine_total={planner.total_refine_calls} "
        f"revised_events={len(revised_events)} "
        f"dry_run={revised_events[0].plan_revised.dry_run if revised_events else 'N/A'} "
        f"plan_changed={session.plan is not pre_plan} "
        f"channel_depth={control._inbox.qsize()} "
        f"cancel_ids={store.cancel_requested_invocation_ids()} "
        f"plugin_calls={len(adapter._plugin.cancel_calls)}"
    )


# ---------------------------------------------------------------------------
# Steerer ``last_addressed_revision_by_drift_key`` not stamped on dry-run
# ---------------------------------------------------------------------------


async def test_observation_only_does_not_stamp_addressed_watermark() -> None:
    """Dry-run revisions must NOT update
    ``last_addressed_revision_by_drift_key`` — that would dampen
    subsequent real detection on the same (kind, target).
    """
    session = _make_session()
    sink = _ListSink()
    planner = _RecordingPlanner(revised=_make_revised_plan())

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=_judge_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        observation_only=True,
    )
    steerer.bind(sinks=[sink], planner=planner)

    store = OrchestrationStore.for_session(session)
    fake_task = await _register_live_invocation(session, store)
    try:
        await steerer.observe_reasoning("raccoons are nocturnal", session=session)
        pending = list(steerer._background_judges)
        await asyncio.gather(*pending, return_exceptions=True)
        await steerer._wait_background_drifts_idle()
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    assert session.last_addressed_revision_by_drift_key == {}, (
        "observation-only must NOT stamp the addressed watermark; got "
        f"{dict(session.last_addressed_revision_by_drift_key)}"
    )


# ---------------------------------------------------------------------------
# Mark a smaller-surface variant via direct _handle_drift call — covers the
# tool-flow path that doesn't depend on the reasoning judge.
# ---------------------------------------------------------------------------


async def test_observation_only_tool_flow_refine_skips_revision_injection() -> None:
    """A synchronous tool-flow drift (mark_task_failed →
    TASK_FAILED_RECOVERABLE) in observation-only mode emits
    DriftDetected, calls planner.refine, and emits a dry-run
    ``PlanRevised`` — but the REVISED plan (the new sub-DAG the
    planner returned) is NOT installed onto ``session.plan``. The
    only mutation that lands is the imperative ``Task.status =
    FAILED`` from the reporting-tool handler, which is a separate
    code path (it's the agent's report being honoured, not an
    injection).
    """
    session = _make_session()
    sink = _ListSink()
    revised_plan = _make_revised_plan()
    planner = _RecordingPlanner(revised=revised_plan)
    control = ControlChannel()

    steerer = DefaultSteerer(observation_only=True)
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(control)

    initial_plan_id = session.plan.id

    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )
    await steerer._wait_background_drifts_idle()

    # The plan id is unchanged — the revised plan from refine was NOT
    # installed. The plan's *contents* changed only because the
    # mark_task_failed reporting-tool handler set t1.status=FAILED
    # in-place (separate from the refine injection point we gate).
    assert session.plan is not None
    assert session.plan.id == initial_plan_id
    # Critically: t2 (the new task the revised plan would have
    # introduced) is NOT on the session plan. If the revision had been
    # injected, t2 would be present.
    task_ids = {t.id for t in session.plan.tasks}
    assert "t2" not in task_ids, (
        "observation-only must NOT install the refined plan's new "
        "task ids onto session.plan; saw t2"
    )
    # No control message on the channel.
    assert control._inbox.qsize() == 0
    # PlanRevised was still emitted (dry_run=True) for the would-have-
    # applied revision.
    revised_events = _plan_revised_events(sink.events)
    assert revised_events, "observation-only must emit PlanRevised preview"
    assert all(e.plan_revised.dry_run is True for e in revised_events)
    # planner.refine was called.
    assert planner.total_refine_calls >= 1, (
        "observation-only must still drive planner.refine"
    )


# ---------------------------------------------------------------------------
# Defaults match documentation
# ---------------------------------------------------------------------------


def test_default_steerer_production_default_is_observation_only() -> None:
    """In production resolution, ``DefaultSteerer()`` is observation-only.

    The autouse fixture flips the default to False during tests, so we
    temporarily restore it here to exercise the production behaviour
    documented in the constructor docstring.
    """
    from goldfive import steerer as steerer_mod

    prior = steerer_mod._test_default_observation_only
    steerer_mod._test_default_observation_only = None
    try:
        steerer = DefaultSteerer()
    finally:
        steerer_mod._test_default_observation_only = prior

    assert steerer.observation_only is True, (
        "production DefaultSteerer() default must be observation_only=True"
    )
