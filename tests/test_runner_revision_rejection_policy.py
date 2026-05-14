"""Tests for the configurable abort policy on revision rejection.

Pin: ``Runner(fail_fast_on_revision_rejection=...)`` (and the env-var
fallback ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1``) controls whether
a goldfive-authored revision rejected by the validator aborts the
turn. Default is non-fatal — see ``docs/design/PLAN-LIFECYCLE.md``
§4.5.1.

User-authored drifts (USER_STEER) are NEVER affected by this flag —
that contract is enforced structurally by
:meth:`DefaultSteerer.install_user_steer` (see §4.2.1) and tested in
``test_user_steer_invariant.py``. The matching assertion here is a
belt-and-braces: regardless of the flag, a user-steer install does
not produce a ``run_aborted`` event.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import (  # noqa: E402, I001
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    Runner,
    SequentialExecutor,
    Session,
    Task,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    ObservedAction,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Adapter / planner stubs
# ---------------------------------------------------------------------------


async def _happy_agent(
    task: Task, session: Session, tools: list[Any]
) -> InvocationResult:
    _ = tools, session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


def _good_first_plan(*, run_id: str) -> Plan:
    return Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g"],
        tasks=[Task(id="t1", title="T1", assignee_agent_id="w")],
        edges=[],
        summary="turn 1",
    )


def _invalid_revision(*, run_id: str, prior_id: str) -> Plan:
    """Revision that drops a prior-COMPLETED task — fails §3.1
    terminal-task preservation, so :meth:`Plan.validate` raises.
    """
    return Plan(
        id=prior_id,
        run_id=run_id,
        goal_ids=["g"],
        # Note: drops "t1" (COMPLETED in the prior plan after turn 1).
        tasks=[Task(id="t2", title="T2", assignee_agent_id="w")],
        edges=[],
        summary="turn 2 invalid",
    )


class _ProgrammablePlanner:
    """Planner stub whose ``handle_turn`` returns ``plans[turn_index]``.

    On turn N (1-indexed), returns ``plans[N-1]`` if available, else
    ``None``. ``generate`` is wired so the first turn falls through
    cleanly when ``handle_turn`` returns ``None``.
    """

    def __init__(self, plans: list[Plan | None]) -> None:
        self._plans = plans
        self._turn_idx = 0

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: Any,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        # Only used as fallback when handle_turn returns None on turn 1.
        run_id = ""
        if context is not None:
            run_id = str(context.get("run_id") or "")
        return _good_first_plan(run_id=run_id or "fallback")

    async def handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        conversation_history: list[Any] | None = None,
        available_agents: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        idx = self._turn_idx
        self._turn_idx += 1
        if idx >= len(self._plans):
            return None
        return self._plans[idx]

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: Any = None,
    ) -> Plan | None:
        return None


def _runner(
    planner: _ProgrammablePlanner,
    sink: InMemorySink,
    *,
    fail_fast: bool | None = None,
) -> Runner:
    kwargs: dict[str, Any] = {}
    if fail_fast is not None:
        kwargs["fail_fast_on_revision_rejection"] = fail_fast
    return Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["w"]),
        planner=planner,
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("demo"),
        sinks=[sink],
        **kwargs,
    )


def _payload_kinds(sink: InMemorySink) -> list[str]:
    out = []
    for evt in sink.events:
        which = evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None
        if which:
            out.append(which)
    return out


def _drift_kinds(sink: InMemorySink) -> list[int]:
    out = []
    for evt in sink.events:
        which = evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None
        if which != "drift_detected":
            continue
        out.append(int(evt.drift_detected.kind))
    return out


# ---------------------------------------------------------------------------
# Default behavior: non-fatal on goldfive-authored revision rejection
# ---------------------------------------------------------------------------


async def test_default_revision_rejection_is_non_fatal() -> None:
    """Default behaviour (no kwarg, no env var):
    a goldfive-authored revision rejected by the validator does NOT
    abort the run. The existing plan is retained, a
    HUMAN_INTERVENTION_REQUIRED INFO drift is emitted for
    observability, and the executor continues."""
    sink = InMemorySink()

    # Turn 1: good plan; turn 2: invalid revision (drops the now-COMPLETED t1).
    plans: list[Plan | None] = []
    planner = _ProgrammablePlanner(plans)
    runner = _runner(planner, sink)

    try:
        # First, run with a good first plan to get t1 COMPLETED.
        plans.append(_good_first_plan(run_id="r"))
        out1 = await runner.run("turn 1", session_id="default-policy")
        assert out1.success, f"turn 1 failed: {out1.reason!r}"
        plan_id_after_t1 = out1.session.plan.id
        index_after_t1 = out1.session.plan.revision_index

        # Now turn 2: invalid revision. The runner's handle_turn-driven
        # install path is goldfive-authored.
        sink.events.clear()
        plans.append(_invalid_revision(run_id="r", prior_id=plan_id_after_t1))
        out2 = await runner.run("turn 2", session_id="default-policy")
    finally:
        await runner.close()

    # No run_aborted on the wire from the install rejection.
    aborted_with_validator_reason = [
        e
        for e in sink.events
        if (e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None)
        == "run_aborted"
        and "plan revision rejected" in e.run_aborted.reason
    ]
    assert not aborted_with_validator_reason, (
        "default policy: validator rejection MUST NOT abort the run; "
        f"got {[e.run_aborted.reason for e in aborted_with_validator_reason]!r}"
    )

    # HUMAN_INTERVENTION_REQUIRED INFO drift was emitted on the sink.
    from goldfive.pb.goldfive.v1 import types_pb2 as _tpb

    hir_drifts = [
        evt.drift_detected
        for evt in sink.events
        if (evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else None)
        == "drift_detected"
        and int(evt.drift_detected.kind)
        == _tpb.DRIFT_KIND_HUMAN_INTERVENTION_REQUIRED
    ]
    assert hir_drifts, (
        "default policy: expected a HUMAN_INTERVENTION_REQUIRED "
        "observability drift after revision rejection; "
        f"saw drift_kinds={_drift_kinds(sink)!r}"
    )
    # INFO severity, authored_by goldfive.
    assert int(hir_drifts[0].severity) == _tpb.DRIFT_SEVERITY_INFO
    assert hir_drifts[0].authored_by == "goldfive"

    # session.plan unchanged from turn 1's revision.
    assert out2.session.plan.id == plan_id_after_t1
    assert out2.session.plan.revision_index == index_after_t1


# ---------------------------------------------------------------------------
# Strict opt-in: kwarg
# ---------------------------------------------------------------------------


async def test_strict_kwarg_aborts_on_revision_rejection() -> None:
    """``Runner(fail_fast_on_revision_rejection=True)`` restores the
    pre-PR1 abort behaviour: validator rejection emits ``run_aborted``
    and the turn ends with ``success=False``."""
    sink = InMemorySink()
    plans: list[Plan | None] = []
    planner = _ProgrammablePlanner(plans)
    runner = _runner(planner, sink, fail_fast=True)

    try:
        plans.append(_good_first_plan(run_id="r"))
        out1 = await runner.run("turn 1", session_id="strict-kwarg")
        assert out1.success
        plan_id_after_t1 = out1.session.plan.id
        sink.events.clear()
        plans.append(_invalid_revision(run_id="r", prior_id=plan_id_after_t1))
        out2 = await runner.run("turn 2", session_id="strict-kwarg")
    finally:
        await runner.close()

    assert not out2.success, "strict mode: invalid revision must fail the turn"
    assert "plan revision rejected" in (out2.reason or ""), out2.reason
    payload_kinds = _payload_kinds(sink)
    assert "run_aborted" in payload_kinds, (
        f"strict mode: expected run_aborted on the sink; got {payload_kinds!r}"
    )


# ---------------------------------------------------------------------------
# Env-var fallback: GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1
# ---------------------------------------------------------------------------


async def test_strict_env_var_fallback(
    goldfive_fail_fast_env: Any,
) -> None:
    """When the constructor kwarg is omitted (``None``), the env var
    ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1`` opts in to strict mode.
    """
    goldfive_fail_fast_env.set(revision_rejection="1")
    sink = InMemorySink()
    plans: list[Plan | None] = []
    planner = _ProgrammablePlanner(plans)
    # No fail_fast kwarg — env var should govern.
    runner = _runner(planner, sink, fail_fast=None)

    try:
        plans.append(_good_first_plan(run_id="r"))
        out1 = await runner.run("turn 1", session_id="env-var")
        assert out1.success
        plan_id_after_t1 = out1.session.plan.id
        sink.events.clear()
        plans.append(_invalid_revision(run_id="r", prior_id=plan_id_after_t1))
        out2 = await runner.run("turn 2", session_id="env-var")
    finally:
        await runner.close()

    assert not out2.success, "env var=1 should opt into strict mode"
    payload_kinds = _payload_kinds(sink)
    assert "run_aborted" in payload_kinds


async def test_explicit_kwarg_overrides_env_var(
    goldfive_fail_fast_env: Any,
) -> None:
    """Explicit ``fail_fast_on_revision_rejection=False`` wins over
    ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1`` so tests can pin
    behaviour without unsetting the env first.
    """
    goldfive_fail_fast_env.set(revision_rejection="1")
    sink = InMemorySink()
    plans: list[Plan | None] = []
    planner = _ProgrammablePlanner(plans)
    runner = _runner(planner, sink, fail_fast=False)

    try:
        plans.append(_good_first_plan(run_id="r"))
        out1 = await runner.run("turn 1", session_id="kwarg-override")
        assert out1.success
        plan_id_after_t1 = out1.session.plan.id
        sink.events.clear()
        plans.append(_invalid_revision(run_id="r", prior_id=plan_id_after_t1))
        await runner.run("turn 2", session_id="kwarg-override")
    finally:
        await runner.close()

    payload_kinds = _payload_kinds(sink)
    aborted_with_validator_reason = [
        e
        for e in sink.events
        if (e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None)
        == "run_aborted"
        and "plan revision rejected" in e.run_aborted.reason
    ]
    assert not aborted_with_validator_reason, (
        "explicit fail_fast=False MUST override env var=1; "
        f"saw run_aborted with validator reason in {payload_kinds!r}"
    )


# ---------------------------------------------------------------------------
# L2 contract: user-steer install never aborts, regardless of the flag
# ---------------------------------------------------------------------------


async def test_user_steer_install_never_aborts_default_flag() -> None:
    """A user-authored steer routed through
    :meth:`DefaultSteerer.install_user_steer` MUST NOT emit
    ``run_aborted`` even when the LLM revision is invalid. The
    ``fail_fast_on_revision_rejection`` flag does not gate this — the
    L2 contract (PLAN-LIFECYCLE.md §4.2.1) is structural.
    """
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)
    session = Session(run_id="l2-default")
    prior = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[
            Task(id="done", title="done", assignee_agent_id="w", status=TaskStatus.COMPLETED),
            Task(id="pending", title="pending", assignee_agent_id="w"),
        ],
        edges=[],
        summary="prior",
    )
    session.plan = prior

    # An invalid LLM revision (drops the COMPLETED task).
    invalid = Plan(
        id=prior.id,
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="new", title="new", assignee_agent_id="w")],
        edges=[],
        summary="invalid",
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        authored_by="user",
    )
    returned = await steerer.plans.install_user_steer(
        drift=drift, prior=prior, llm_revision=invalid, session=session
    )
    # The fallback fired — pending was cancelled, done preserved.
    assert isinstance(returned, Plan)
    statuses = {t.id: t.status for t in returned.tasks}
    assert statuses["done"] is TaskStatus.COMPLETED
    assert statuses["pending"] is TaskStatus.CANCELLED
    # No run_aborted on the wire.
    payload_kinds = _payload_kinds(sink)
    assert "run_aborted" not in payload_kinds, (
        f"L2 contract: user-steer install must not emit run_aborted; "
        f"got {payload_kinds!r}"
    )


async def test_user_steer_install_never_aborts_strict_flag(
    goldfive_fail_fast_env: Any,
) -> None:
    """Same contract as the default-flag test, but with the strict env
    var on. The flag governs goldfive-authored installs only; user-
    authored installs go through the L2 type-safe path that has no
    failure mode."""
    goldfive_fail_fast_env.set(revision_rejection="1")
    # Same setup as the default test — the steerer call doesn't
    # consult the env var.
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=None)
    session = Session(run_id="l2-strict")
    prior = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[
            Task(id="done", title="done", assignee_agent_id="w", status=TaskStatus.COMPLETED),
            Task(id="pending", title="pending", assignee_agent_id="w"),
        ],
        edges=[],
        summary="prior",
    )
    session.plan = prior
    invalid = Plan(
        id=prior.id,
        run_id=session.run_id,
        goal_ids=["g"],
        tasks=[Task(id="new", title="new", assignee_agent_id="w")],
        edges=[],
        summary="invalid",
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        authored_by="user",
    )
    await steerer.plans.install_user_steer(
        drift=drift, prior=prior, llm_revision=invalid, session=session
    )
    payload_kinds = _payload_kinds(sink)
    assert "run_aborted" not in payload_kinds


# ---------------------------------------------------------------------------
# After a non-fatal rejection, the run continues — executor.run was
# invoked, the registered ``_happy_agent`` adapter ran, and turn-2 lands
# with success=True (no work to do because the prior plan's only task
# completed in turn 1, but the run completes cleanly rather than
# aborting).
# ---------------------------------------------------------------------------


async def test_default_run_continues_after_rejection() -> None:
    """After a non-fatal validator rejection on turn N, the executor
    is still invoked and the turn produces a normal outcome (not the
    pre-PR1 ``success=False`` from the abort path).
    """
    sink = InMemorySink()
    plans: list[Plan | None] = []
    planner = _ProgrammablePlanner(plans)
    runner = _runner(planner, sink)

    try:
        plans.append(_good_first_plan(run_id="r"))
        out1 = await runner.run("turn 1", session_id="continues")
        assert out1.success
        plan_id_after_t1 = out1.session.plan.id
        sink.events.clear()
        plans.append(_invalid_revision(run_id="r", prior_id=plan_id_after_t1))
        out2 = await runner.run("turn 2", session_id="continues")
    finally:
        await runner.close()

    # Outcome is not the pre-PR1 abort one.
    assert (out2.reason or "") != "plan revision rejected by validator"
    # session.plan still reflects turn 1's revision (unchanged).
    assert out2.session.plan.id == plan_id_after_t1
