"""Tests for the adapter-level planner-gate (goldfive#270 follow-up).

The runner-side gate (:meth:`goldfive.runner.Runner._classify_turn`)
ships the same three-verdict classifier on every call to
:meth:`Runner.run`. For ADK-web turns the classifier ALSO runs at the
adapter boundary — :meth:`GoldfiveADKAgent._run_async_impl` calls
:meth:`GoldfiveADKAgent._classify_user_message_for_adk` BEFORE handing
control to :meth:`Runner.run_streamed`, threads the verdict through
``context["_adk_pre_classified_verdict"]``, and Runner.run honours it
instead of running the gate twice.

Coverage:

1. ``refine_existing`` verdict at the adapter boundary makes Runner.run
   route through the steerer's USER_STEER pipeline (PlanRevised emitted,
   revision_index bumped, no fresh PlanSubmitted).
2. ``conversational`` verdict skips planning AND skips the steerer
   entirely.
3. ``new_work`` verdict triggers the normal generate / dispatch path
   exactly as it would have without the adapter pre-pass.
4. First turn (no prior plan) — adapter classifier returns ``None``,
   Runner.run runs its own gate (which itself returns ``new_work``).
5. Replay dedup — re-driving the same (invocation_id, user_input)
   does NOT classify a second time.
6. Suppression interaction — once a USER_STEER fired this turn, a
   later goldfive-detected drift gets ``suppressed_by_user_steer=True``
   on the wire (the runner's existing _should_promote_to_steer
   machinery already handles this; the test verifies it still works
   when the steer was routed via the adapter pre-classification).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.adk")

import goldfive
from goldfive import InMemorySink, InvocationResult, SequentialExecutor, StaticPlanner
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Event-shape helpers
#
# Goldfive sinks observe a mix of proto-shaped events (set via the typed
# `make_event_*` factories in goldfive.events) and dict-shaped events
# (set via `make_event` for callers that don't need protos). The helpers
# below normalise both shapes so a single test asserts against either —
# the test layer doesn't care which envelope a given emission picked.
# ---------------------------------------------------------------------------


def _is_proto(evt: Any) -> bool:
    return hasattr(evt, "HasField")


def _proto_drift_detected(evt: Any) -> Any | None:
    """Return the drift_detected sub-message if this is a proto event with that field set."""
    if not _is_proto(evt):
        return None
    try:
        if evt.HasField("drift_detected"):
            return evt.drift_detected
    except Exception:  # noqa: BLE001
        return None
    return None


def _drift_user_steer_events(events: list[Any]) -> list[Any]:
    """Filter for DriftDetected(USER_STEER) on either envelope shape.

    Proto kind is an enum int — resolve via the proto pb2 enum module
    so we don't hard-code the underlying integer value (it's stable but
    enum-managed elsewhere).
    """
    # Lazy import to keep this module light when proto isn't installed.
    from goldfive.pb.goldfive.v1 import types_pb2  # type: ignore

    user_steer_value = types_pb2.DriftKind.Value("DRIFT_KIND_USER_STEER")
    out: list[Any] = []
    for e in events:
        dd = _proto_drift_detected(e)
        if dd is not None and int(dd.kind) == user_steer_value:
            out.append(e)
            continue
        # Dict shape (not currently used by USER_STEER emit, but
        # defensive): kind == "drift_detected", payload includes
        # drift_kind == "user_steer".
        if isinstance(e, dict) and e.get("kind") == "drift_detected":
            payload = e.get("payload") or {}
            if str(payload.get("drift_kind", "")).lower().endswith("user_steer"):
                out.append(e)
    return out


def _all_drift_events(events: list[Any]) -> list[Any]:
    """All DriftDetected events on either envelope shape (any kind)."""
    out: list[Any] = []
    for e in events:
        if _proto_drift_detected(e) is not None:
            out.append(e)
            continue
        if isinstance(e, dict) and e.get("kind") == "drift_detected":
            out.append(e)
    return out


def _drift_suppressed_by_user_steer(evt: Any) -> bool:
    dd = _proto_drift_detected(evt)
    if dd is not None:
        return bool(getattr(dd, "suppressed_by_user_steer", False))
    if isinstance(evt, dict):
        payload = evt.get("payload") or {}
        return bool(payload.get("suppressed_by_user_steer", False))
    return False


def _plan_submitted_events(events: list[Any]) -> list[Any]:
    out: list[Any] = []
    for e in events:
        if _is_proto(e):
            try:
                if e.HasField("plan_submitted"):
                    out.append(e)
            except Exception:  # noqa: BLE001
                pass
            continue
        if isinstance(e, dict) and e.get("kind") == "plan_submitted":
            out.append(e)
    return out


def _plan_revised_events(events: list[Any]) -> list[Any]:
    out: list[Any] = []
    for e in events:
        if _is_proto(e):
            try:
                if e.HasField("plan_revised"):
                    out.append(e)
            except Exception:  # noqa: BLE001
                pass
            continue
        if isinstance(e, dict) and e.get("kind") == "plan_revised":
            out.append(e)
    return out


def _plan_revised_revision_index(evt: Any) -> int | None:
    if _is_proto(evt):
        try:
            if evt.HasField("plan_revised"):
                return int(evt.plan_revised.revision_index)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(evt, dict) and evt.get("kind") == "plan_revised":
        payload = evt.get("payload") or {}
        ri = payload.get("revision_index")
        if isinstance(ri, int):
            return ri
        try:
            return int(ri) if ri is not None else None
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _completed_plan() -> Plan:
    return Plan(
        id="prior-1",
        run_id="r-prior",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Gather facts",
                description="x",
                assignee_agent_id="inner_agent",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
        summary="Prior plan completed",
    )


def _two_task_planner() -> StaticPlanner:
    return StaticPlanner(
        Plan(
            id="p1",
            run_id="",
            goal_ids=["g1"],
            tasks=[
                Task(
                    id="t1",
                    title="Task one",
                    description="x",
                    assignee_agent_id="inner_agent",
                ),
                Task(
                    id="t2",
                    title="Task two",
                    description="y",
                    assignee_agent_id="inner_agent",
                ),
            ],
            edges=[],
            summary="two task plan",
        )
    )


def _mk_inner(name: str = "inner_agent") -> Any:
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    return LlmAgent(
        name=name,
        model="fake-model",
        description="a wrapped agent",
        instruction="follow instructions",
    )


class _FakeSession:
    def __init__(self, sid: str = "outer-sess") -> None:
        self.id = sid
        self.events: list[Any] = []


class _FakeCtx:
    def __init__(self, text: str, *, invocation_id: str = "inv-1") -> None:
        from google.genai.types import Content, Part  # type: ignore

        self.user_content = Content(role="user", parts=[Part(text=text)])
        self.session = _FakeSession()
        self.invocation_id = invocation_id
        self.end_invocation = False


def _build_wrapped(
    *,
    planner: Any | None = None,
    planner_gate: Any = "auto",
    sinks: list[Any] | None = None,
) -> Any:
    inner = _mk_inner()
    wrapped = goldfive.wrap(
        inner,
        planner=planner or _two_task_planner(),
        sinks=sinks if sinks is not None else [InMemorySink()],
    )
    wrapped.runner._planner_gate = planner_gate
    wrapped.runner.executor = SequentialExecutor(max_task_invocations=4)
    adapter = wrapped.runner.agent

    async def _fake_invoke(task: Task, session: Any) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=f"ok: {task.id}")

    adapter.invoke = AsyncMock(side_effect=_fake_invoke)
    return wrapped


def _seed_prior_plan(wrapped: Any) -> None:
    """Stamp a completed plan onto the runner so turn-2 has prior context."""
    wrapped.runner._last_plan = _completed_plan()


# ---------------------------------------------------------------------------
# 1. refine_existing → USER_STEER pipeline fires from the adapter pre-pass
# ---------------------------------------------------------------------------


async def test_adapter_gate_refine_existing_routes_through_steerer() -> None:
    """``refine_existing`` from the adapter pre-pass drives the steerer's
    USER_STEER pipeline: PlanRevised emits with bumped revision_index,
    no fresh PlanSubmitted is observed.
    """
    sink = InMemorySink()
    wrapped = _build_wrapped(planner_gate="auto", sinks=[sink])

    # Seed a prior plan so the adapter-level gate runs (no-prior-plan
    # short-circuits to None and lets Runner.run handle).
    _seed_prior_plan(wrapped)

    # Force the gate to refine_existing by replacing _classify_turn on
    # the runner. The adapter helper delegates to this method.
    classify_calls: list[Any] = []

    async def _force_refine(*, prior_plan, completed_results, user_input, session):
        classify_calls.append(user_input)
        return "refine_existing"

    wrapped.runner._classify_turn = _force_refine

    # Stub the planner.refine to return a clean revised plan so the
    # steerer's _apply_revision installs it cleanly.
    revised = Plan(
        id="prior-1",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Gather facts",
                description="x",
                assignee_agent_id="inner_agent",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2-new",
                title="Pivot to flares",
                description="z",
                assignee_agent_id="inner_agent",
            ),
        ],
        edges=[],
        summary="Pivoted plan",
        revision_index=0,  # _apply_revision bumps from prior+1
    )

    async def _fake_refine(*, plan, drift, goals, **kwargs):
        return revised

    wrapped.runner.planner.refine = _fake_refine  # type: ignore[method-assign]

    ctx = _FakeCtx("forget solar panels. tell me about solar flares instead.")
    events_yielded = [e async for e in wrapped._run_async_impl(ctx)]
    assert events_yielded  # smoke

    # The classifier was consulted exactly once via the adapter pre-pass.
    # Runner.run's own _classify_turn would also call it, but the pre-
    # classified verdict short-circuits it — so we expect exactly one.
    assert len(classify_calls) == 1, (
        f"expected one classifier call from adapter pre-pass; got "
        f"{len(classify_calls)}: {classify_calls}"
    )

    # Wire-level checks: a DriftDetected(USER_STEER) and a PlanRevised
    # event landed on the sink.
    sink_events = list(sink.events)
    drift_events = _drift_user_steer_events(sink_events)
    plan_revised = _plan_revised_events(sink_events)
    assert drift_events, (
        "no DriftDetected(USER_STEER) on the wire; the adapter-level "
        "refine_existing path didn't reach steerer._handle_drift"
    )
    assert plan_revised, (
        "no PlanRevised on the wire; the steerer's _apply_revision did "
        "not install the revised plan"
    )
    # revision_index must increment from the prior's 0 → 1.
    ri = _plan_revised_revision_index(plan_revised[-1])
    assert ri == 1, f"expected revision_index=1; got {ri!r}"


# ---------------------------------------------------------------------------
# 2. conversational → no steerer, no planner.refine
# ---------------------------------------------------------------------------


async def test_adapter_gate_conversational_skips_steerer() -> None:
    """``conversational`` verdict from the adapter pre-pass leaves the
    steerer's pipeline entirely alone — no DriftDetected, no PlanRevised.
    """
    sink = InMemorySink()
    wrapped = _build_wrapped(planner_gate="auto", sinks=[sink])
    _seed_prior_plan(wrapped)

    async def _force_conv(*, prior_plan, completed_results, user_input, session):
        return "conversational"

    wrapped.runner._classify_turn = _force_conv

    refine_calls: list[Any] = []

    async def _trip_refine(**kwargs):
        refine_calls.append(kwargs)
        return None

    wrapped.runner.planner.refine = _trip_refine  # type: ignore[method-assign]

    ctx = _FakeCtx("where are the slides saved?")
    [e async for e in wrapped._run_async_impl(ctx)]

    sink_events = list(sink.events)
    assert not _drift_user_steer_events(sink_events), (
        "USER_STEER fired on a conversational turn"
    )
    assert not _plan_revised_events(sink_events), (
        "PlanRevised fired on a conversational turn"
    )
    assert not refine_calls, "planner.refine called on a conversational turn"


# ---------------------------------------------------------------------------
# 3. new_work → normal generate + plan submitted
# ---------------------------------------------------------------------------


async def test_adapter_gate_new_work_runs_normal_planner_path() -> None:
    """``new_work`` verdict produces a fresh PlanSubmitted (full
    re-planning). The adapter pre-pass should not interfere.
    """
    sink = InMemorySink()
    wrapped = _build_wrapped(planner_gate="auto", sinks=[sink])
    _seed_prior_plan(wrapped)

    async def _force_new(*, prior_plan, completed_results, user_input, session):
        return "new_work"

    wrapped.runner._classify_turn = _force_new

    ctx = _FakeCtx("a wholly new and unrelated workflow about pasta")
    [e async for e in wrapped._run_async_impl(ctx)]

    sink_events = list(sink.events)
    plan_submitted = _plan_submitted_events(sink_events)
    plan_revised = _plan_revised_events(sink_events)
    assert plan_submitted, "expected PlanSubmitted on a new_work turn"
    assert not plan_revised, "PlanRevised should not fire on a new_work turn"


# ---------------------------------------------------------------------------
# 4. First turn — gate returns None, Runner.run handles
# ---------------------------------------------------------------------------


async def test_adapter_gate_first_turn_short_circuits_to_none() -> None:
    """No prior plan ⇒ adapter classifier returns None ⇒ Runner.run runs
    its own gate (which also returns new_work for first-turn callers)
    and the run proceeds normally.
    """
    wrapped = _build_wrapped(planner_gate="auto")
    # No prior plan stamped — first turn.
    classify_calls: list[Any] = []

    async def _spy(*, prior_plan, completed_results, user_input, session):
        classify_calls.append(user_input)
        return "new_work"

    wrapped.runner._classify_turn = _spy

    ctx = _FakeCtx("make a thing")
    events = [e async for e in wrapped._run_async_impl(ctx)]
    assert events  # smoke — the run executed

    # The adapter pre-pass short-circuits on no-prior-plan and never
    # calls _classify_turn. Runner.run also short-circuits because
    # _last_plan is None on the first turn (per its own gate guard).
    # Net: zero classifier calls.
    assert classify_calls == [], (
        f"expected no classifier calls on first turn (no prior plan); "
        f"got {classify_calls}"
    )


# ---------------------------------------------------------------------------
# 5. Replay dedup — same invocation_id + user_input is not classified twice
# ---------------------------------------------------------------------------


async def test_adapter_gate_replay_does_not_double_classify() -> None:
    """Re-driving the same invocation_id with identical user_input must
    only classify once: the adapter caches (invocation_id, user_input)
    on the wrapper.
    """
    wrapped = _build_wrapped(planner_gate="auto")
    _seed_prior_plan(wrapped)

    classify_calls: list[Any] = []

    async def _spy(*, prior_plan, completed_results, user_input, session):
        classify_calls.append(user_input)
        return "new_work"

    wrapped.runner._classify_turn = _spy

    ctx_a = _FakeCtx("update the slides please", invocation_id="inv-replay")
    [e async for e in wrapped._run_async_impl(ctx_a)]
    first_calls = list(classify_calls)
    assert len(first_calls) == 1, "expected 1 call on the first drive"

    # Replay the same invocation_id + user_input.
    ctx_b = _FakeCtx("update the slides please", invocation_id="inv-replay")
    [e async for e in wrapped._run_async_impl(ctx_b)]

    assert classify_calls == first_calls, (
        f"replay double-classified: {classify_calls!r}"
    )


async def test_adapter_gate_distinct_invocation_id_classifies_again() -> None:
    """The dedup key includes invocation_id, so a fresh invocation with
    the same text DOES re-classify (a real turn, not a replay).
    """
    wrapped = _build_wrapped(planner_gate="auto")
    _seed_prior_plan(wrapped)

    classify_calls: list[Any] = []

    async def _spy(*, prior_plan, completed_results, user_input, session):
        classify_calls.append(user_input)
        return "new_work"

    wrapped.runner._classify_turn = _spy

    ctx_a = _FakeCtx("ok then continue", invocation_id="inv-A")
    [e async for e in wrapped._run_async_impl(ctx_a)]
    ctx_b = _FakeCtx("ok then continue", invocation_id="inv-B")
    [e async for e in wrapped._run_async_impl(ctx_b)]

    assert len(classify_calls) == 2, (
        f"distinct invocation_ids should classify twice; got "
        f"{classify_calls!r}"
    )


# ---------------------------------------------------------------------------
# 6. Suppression — goldfive drift after a fresh USER_STEER carries the flag
# ---------------------------------------------------------------------------


async def test_user_steer_suppresses_subsequent_goldfive_drift() -> None:
    """Within a USER_STEER's freshness window the steerer's
    :meth:`_should_promote_to_steer` stamps ``suppressed_by_user_steer``
    on a subsequent goldfive-authored drift.

    This test pins the contract that the adapter-routed USER_STEER path
    leaves the same orchestration state behind that the historical
    direct-steerer path did, so the existing suppression machinery
    keeps working without modification.

    The run-end ``clear_active_steer`` call in Runner.run wipes the
    state when the run completes, so this test re-enacts the
    in-window state explicitly before firing the goldfive drift —
    that's what the steerer would see during a still-running turn
    in production (drift detected mid-run, after the USER_STEER lands
    but before the run completes).
    """
    sink = InMemorySink()
    wrapped = _build_wrapped(planner_gate="auto", sinks=[sink])
    _seed_prior_plan(wrapped)

    async def _force_refine(*, prior_plan, completed_results, user_input, session):
        return "refine_existing"

    wrapped.runner._classify_turn = _force_refine

    revised = Plan(
        id="prior-1",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Gather facts",
                description="x",
                assignee_agent_id="inner_agent",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
        summary="No-op revision",
    )

    async def _fake_refine(*, plan, drift, goals, **kwargs):
        return revised

    wrapped.runner.planner.refine = _fake_refine  # type: ignore[method-assign]

    # Drive the run end-to-end so the USER_STEER lands.
    ctx = _FakeCtx("forget panels. flares instead.")
    events_yielded = [e async for e in wrapped._run_async_impl(ctx)]
    assert events_yielded  # smoke

    # Re-stamp active_steer state to simulate an in-window mid-run
    # observation (the run-end clear_active_steer would otherwise have
    # wiped it). This is the state the suppression check actually sees
    # in production when a goldfive drift fires mid-turn.
    steerer = wrapped.runner.steerer
    session = wrapped.runner._last_session
    assert session is not None
    from goldfive.orchestration_state import set_active_steer

    set_active_steer(
        session.state,
        body="forget panels. flares instead.",
        at_turn=int(getattr(session, "_next_sequence", 0) or 0),
        author="user",
        source="user",
    )

    later_drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="goldfive observation",
        authored_by="goldfive",
    )
    await steerer._handle_drift(later_drift, session)

    # Find the LAST DriftDetected on the wire — the goldfive one we
    # just fired — and assert the suppression flag is set.
    sink_events = list(sink.events)
    drift_events = _all_drift_events(sink_events)
    assert len(drift_events) >= 2, (
        "expected USER_STEER + goldfive drift on the wire; got "
        f"{len(drift_events)} drift event(s)"
    )
    last_suppressed = _drift_suppressed_by_user_steer(drift_events[-1])
    assert last_suppressed is True, (
        "goldfive drift after a fresh USER_STEER did not carry "
        "suppressed_by_user_steer=True; suppression machinery missed "
        "the adapter-routed steer"
    )


# ---------------------------------------------------------------------------
# 7. Heuristic-mode integration — the heuristic now catches steer language
# ---------------------------------------------------------------------------


async def test_adapter_gate_heuristic_catches_steer_language() -> None:
    """End-to-end with the heuristic gate (no LLM): a steer-shaped
    user message produces refine_existing → USER_STEER → PlanRevised.

    Previously the heuristic returned ``conversational`` (short input)
    or ``new_work`` (long input), both of which silently dropped the
    prior plan's structural constraints. This test pins the fixed
    behaviour.
    """
    sink = InMemorySink()
    wrapped = _build_wrapped(planner_gate="auto", sinks=[sink])
    # No call_llm → "auto" falls through to heuristic_classify_turn.
    wrapped.runner.planner._call_llm = None  # type: ignore[attr-defined]
    _seed_prior_plan(wrapped)

    revised = Plan(
        id="prior-1",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Gather facts",
                description="x",
                assignee_agent_id="inner_agent",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2-new",
                title="Pivot",
                description="z",
                assignee_agent_id="inner_agent",
            ),
        ],
        edges=[],
        summary="Heuristic-routed pivot",
    )

    async def _fake_refine(*, plan, drift, goals, **kwargs):
        return revised

    wrapped.runner.planner.refine = _fake_refine  # type: ignore[method-assign]

    # Steer-shaped opener — the heuristic now matches and routes to
    # refine_existing.
    ctx = _FakeCtx("forget solar panels. tell me about solar flares instead.")
    [e async for e in wrapped._run_async_impl(ctx)]

    sink_events = list(sink.events)
    user_steers = _drift_user_steer_events(sink_events)
    plan_revised = _plan_revised_events(sink_events)
    assert user_steers, (
        "heuristic gate did not route a steer-shaped message through "
        "USER_STEER; the regression goldfive#270 documented is back"
    )
    assert plan_revised, "no PlanRevised on the wire after USER_STEER"
