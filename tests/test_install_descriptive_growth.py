"""Unit tests for :meth:`PlanReviser.install_descriptive_growth` (goldfive#423 PR 2).

Covers the §4.3 + §5 contract of the helper:

* **Idempotence by identity_hash** — repeated calls with the same
  ``(agent_name, tool_args_json)`` return the SAME discovered task; the
  plan grows once.
* **Forward-compat with empty tool_args_json** — empty / missing args
  fall back to a coarser per-``(agent_name, "")`` hash; the helper
  still produces a valid discovered task.
* **Plan revision shape** — new task lands with ``discovered=True``,
  no predecessor edges, ``status=PENDING``, ``assignee_agent_id``
  populated from ``agent_name``.
* **NEW_WORK_DISCOVERED severity = INFO** — design doc §6 +
  ``_apply_revision`` discovery carve-out semantics: framework-
  synthesised discoveries are observational, not corrective, so they
  emit at INFO severity.
* **Observation-mode parity** — discovery lands identically under
  ``observation_only=True`` (the goldfive#258 carve-out covers it).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
    discovery_identity_hash,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    """No-op planner — the install_descriptive_growth helper does NOT
    call planner.refine, so this stub never needs to do anything."""

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: Any,
        goals: list[Goal],
    ) -> Plan | None:
        return None


def _make_steerer(*, observation_only: bool = False) -> DefaultSteerer:
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=observation_only,
            descriptive_growth_enabled=True,
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())
    return steerer


def _initial_plan() -> Plan:
    return Plan(
        id="p-pr2",
        run_id="r-pr2",
        goal_ids=["g-pr2"],
        tasks=[
            Task(id="t1", title="planned-task-1", status=TaskStatus.PENDING),
            Task(id="t2", title="planned-task-2", status=TaskStatus.PENDING),
        ],
        edges=[],
        revision_index=1,
    )


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r-pr2",
        goals=[Goal(id="g-pr2", summary="exercise descriptive growth")],
        plan=plan if plan is not None else _initial_plan(),
    )


async def test_grows_plan_with_discovered_task_shape() -> None:
    """Growth lands one new task with discovered=True + correct shape."""
    session = _make_session()
    steerer = _make_steerer()
    prior_count = len(session.plan.tasks)
    prior_rev = session.plan.revision_index

    task = await steerer.plans.install_descriptive_growth(
        session,
        agent_name="debugger_agent",
        tool_args_json='{"request": "locate cherry tree files"}',
        delegation_event_id="evt-1",
    )

    assert task is not None
    # New task has the expected shape.
    assert task.discovered is True
    assert task.assignee_agent_id == "debugger_agent"
    assert task.status is TaskStatus.PENDING
    assert task.discovery_identity_hash != ""
    # The hash is deterministic.
    expected_hash = discovery_identity_hash(
        "debugger_agent", '{"request": "locate cherry tree files"}'
    )
    assert task.discovery_identity_hash == expected_hash

    # Plan has grown by exactly one task; revision_index advanced.
    assert session.plan is not None
    assert len(session.plan.tasks) == prior_count + 1
    assert session.plan.revision_index == prior_rev + 1
    # The new task appears in the plan with the same id.
    plan_ids = {t.id for t in session.plan.tasks}
    assert task.id in plan_ids
    # The discovered task is a sub-DAG root — no predecessor edges
    # (the helper doesn't add any).
    incoming = [e for e in session.plan.edges if e.to_task_id == task.id]
    assert (
        incoming == []
    ), f"discovered task must be a sub-DAG root; got incoming edges: {incoming}"


async def test_idempotent_under_same_args() -> None:
    """Repeated calls with same (agent, tool_args_json) dedup to one task."""
    session = _make_session()
    steerer = _make_steerer()

    args = '{"request": "find log files in /var/log"}'

    t1 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json=args
    )
    t2 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json=args
    )
    t3 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json=args
    )

    # All three calls return the SAME task.
    assert t1.id == t2.id == t3.id
    # The plan has only ONE discovered task.
    assert session.plan is not None
    discovered = [t for t in session.plan.tasks if t.discovered]
    assert len(discovered) == 1, (
        f"dedup failed: 3 same-args calls produced {len(discovered)} "
        f"discovered tasks: {[t.id for t in discovered]}"
    )


async def test_idempotent_across_args_whitespace_and_case() -> None:
    """Trivial whitespace/case variants in tool_args_json dedup to one task.

    The §4.3.0 normalisation lowercases + tokenises on word characters
    so "Cherry Trees" and "cherry trees" hash identically.
    """
    session = _make_session()
    steerer = _make_steerer()

    t1 = await steerer.plans.install_descriptive_growth(
        session,
        agent_name="debugger_agent",
        tool_args_json='{"request": "Cherry Trees"}',
    )
    t2 = await steerer.plans.install_descriptive_growth(
        session,
        agent_name="debugger_agent",
        tool_args_json='{"request": "cherry  trees"}',
    )
    # Both calls dedup to the same task.
    assert t1.id == t2.id
    assert session.plan is not None
    assert len([t for t in session.plan.tasks if t.discovered]) == 1


async def test_distinct_agents_produce_distinct_discovered_tasks() -> None:
    """Different agents with the same args → distinct discovered tasks."""
    session = _make_session()
    steerer = _make_steerer()

    args = '{"request": "same args"}'

    t1 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json=args
    )
    t2 = await steerer.plans.install_descriptive_growth(
        session, agent_name="reviewer_agent", tool_args_json=args
    )

    assert t1.id != t2.id
    assert t1.discovery_identity_hash != t2.discovery_identity_hash
    assert session.plan is not None
    discovered_ids = {t.id for t in session.plan.tasks if t.discovered}
    assert {t1.id, t2.id} == discovered_ids


async def test_empty_tool_args_json_forward_compat() -> None:
    """Empty tool_args_json (legacy events) degrades to coarser dedup.

    Design doc §9 forward-compat: old events without ``tool_args_json``
    default-empty; PR 2's dedup falls back to per-``(agent, "")`` so
    every legacy delegation for an unmatched agent still dedups to a
    single discovered task per agent. The helper must accept ``""``
    and produce a valid (non-empty) hash.
    """
    session = _make_session()
    steerer = _make_steerer()

    t1 = await steerer.plans.install_descriptive_growth(
        session, agent_name="legacy_agent", tool_args_json=""
    )
    t2 = await steerer.plans.install_descriptive_growth(
        session, agent_name="legacy_agent", tool_args_json=""
    )

    # Dedup to one task per (agent, "") tuple.
    assert t1.id == t2.id
    assert t1.discovery_identity_hash != ""  # hash is still well-defined
    assert session.plan is not None
    assert len([t for t in session.plan.tasks if t.discovered]) == 1


async def test_grows_seed_plan_when_session_plan_is_none() -> None:
    """When session.plan is None, helper seeds a fresh single-task plan."""
    session = _make_session(plan=None)
    steerer = _make_steerer()

    task = await steerer.plans.install_descriptive_growth(
        session, agent_name="seed_agent", tool_args_json='{"foo": "bar"}'
    )

    assert task is not None
    assert task.discovered is True
    assert session.plan is not None
    assert task.id in {t.id for t in session.plan.tasks}


async def test_emits_plan_revised_with_new_work_discovered_kind() -> None:
    """The off-lock emit produces a PlanRevised carrying NEW_WORK_DISCOVERED."""
    session = _make_session()
    sink = _ListSink()
    steerer = DefaultSteerer(
        steering_config=SteeringConfig(
            observation_only=False, descriptive_growth_enabled=True
        )
    )
    steerer.bind(sinks=[sink], planner=_NullPlanner())

    await steerer.plans.install_descriptive_growth(
        session,
        agent_name="debugger_agent",
        tool_args_json='{"request": "find files"}',
    )

    # Extract PlanRevised envelopes from the sink.
    plan_revised_events = [
        e for e in sink.events if hasattr(e, "plan_revised") and e.HasField("plan_revised")
    ]
    assert plan_revised_events, (
        "growth did not emit a PlanRevised envelope; sink events: "
        f"{[type(e).__name__ for e in sink.events]}"
    )
    pr = plan_revised_events[-1]
    # drift_kind on the wire is the proto enum integer for
    # NEW_WORK_DISCOVERED. We compare via the steerer's helper to
    # avoid hard-coding the enum int.
    expected_kind = steerer._drift_kind_pb_value(DriftKind.NEW_WORK_DISCOVERED)
    assert pr.plan_revised.drift_kind == expected_kind, (
        "growth must emit PlanRevised.drift_kind=NEW_WORK_DISCOVERED; "
        f"got {pr.plan_revised.drift_kind} vs expected {expected_kind}"
    )
    expected_sev = steerer._drift_severity_pb_value(DriftSeverity.INFO)
    assert pr.plan_revised.severity == expected_sev, (
        "framework-synthesised discovery must be INFO severity per "
        f"design doc §4.6 / §6; got {pr.plan_revised.severity}"
    )
    # The growth is observational; dry_run is False (real revision,
    # not a would-have-been preview).
    assert pr.plan_revised.dry_run is False


async def test_growth_lands_under_observation_only_mode() -> None:
    """Discovery installs identically under observation_only=True.

    Per design doc §6.2 + the goldfive#258 ``_apply_revision`` carve-out
    for ``NEW_WORK_DISCOVERED``: observation mode suppresses corrective
    revisions but allows discovery. The descriptive-growth helper
    inlines the install slice that bypasses the observation_only gate.
    """
    session = _make_session()
    steerer = _make_steerer(observation_only=True)
    prior_count = len(session.plan.tasks)

    task = await steerer.plans.install_descriptive_growth(
        session,
        agent_name="debugger_agent",
        tool_args_json='{"request": "find files"}',
    )

    # Plan grew even though observation_only=True.
    assert session.plan is not None
    assert len(session.plan.tasks) == prior_count + 1
    assert task.id in {t.id for t in session.plan.tasks}


async def test_title_derived_from_request_arg() -> None:
    """Title preference: tool_args.request > tool_args.task > tool_args.goal > fallback."""
    session = _make_session()
    steerer = _make_steerer()

    # request key takes precedence.
    t = await steerer.plans.install_descriptive_growth(
        session,
        agent_name="agent_a",
        tool_args_json='{"request": "the request text", "task": "ignored"}',
    )
    assert "the request text" in t.title

    # task key when request absent.
    session2 = _make_session()
    t2 = await steerer.plans.install_descriptive_growth(
        session2,
        agent_name="agent_b",
        tool_args_json='{"task": "fallback to task"}',
    )
    assert "fallback to task" in t2.title

    # Fallback when no recognised key.
    session3 = _make_session()
    t3 = await steerer.plans.install_descriptive_growth(
        session3, agent_name="agent_c", tool_args_json='{"unrelated": "key"}'
    )
    assert "agent_c" in t3.title
    assert "discovered work" in t3.title


async def test_dedup_preserves_first_task_id_across_concurrent_calls() -> None:
    """Sequential dedup: second call returns the FIRST task's id."""
    import asyncio

    session = _make_session()
    steerer = _make_steerer()

    args = '{"request": "stable args"}'

    # Sequential: second call deduplicates against first.
    first = await steerer.plans.install_descriptive_growth(
        session, agent_name="agent_x", tool_args_json=args
    )
    second = await steerer.plans.install_descriptive_growth(
        session, agent_name="agent_x", tool_args_json=args
    )
    assert first.id == second.id

    # Concurrent: 10 simultaneous → all return first.id.
    fresh_session = _make_session()
    fresh_steerer = _make_steerer()

    async def grow() -> Task:
        return await fresh_steerer.plans.install_descriptive_growth(
            fresh_session, agent_name="agent_x", tool_args_json=args
        )

    results = await asyncio.gather(*(grow() for _ in range(10)))
    unique_ids = {t.id for t in results}
    assert len(unique_ids) == 1, (
        "concurrent same-args calls must dedup to a single discovered "
        f"task; got ids: {unique_ids}"
    )


async def test_dedup_ttl_terminal_discovered_task_regrows() -> None:
    """§11.1 TTL: dedup window ends when the discovered task goes terminal.

    A fresh delegation whose identity hash matches a COMPLETED (or
    otherwise terminal) discovered task is a genuinely new unit of
    work — the helper must grow a NEW task instead of re-pinning the
    finished one. (goldfive#423 / AGENCY-PRESERVATION.md PR 2.)
    """
    from goldfive.types import (
        channel_processor_active,
        set_session_plan,
        with_task_status,
    )

    session = _make_session()
    steerer = _make_steerer()

    t1 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json='{"request": "locate"}'
    )
    # The discovered work finishes.
    with channel_processor_active():
        set_session_plan(
            session, with_task_status(session.plan, t1.id, TaskStatus.COMPLETED)
        )

    t2 = await steerer.plans.install_descriptive_growth(
        session, agent_name="debugger_agent", tool_args_json='{"request": "locate"}'
    )

    assert t2.id != t1.id, (
        "a terminal discovered task must NOT absorb a fresh same-hash "
        "delegation (§11.1 TTL)"
    )
    assert t2.discovery_identity_hash == t1.discovery_identity_hash
    live = [t for t in session.plan.tasks if t.discovered]
    assert {t.id for t in live} == {t1.id, t2.id}
