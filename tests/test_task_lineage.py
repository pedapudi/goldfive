"""Tests for the observation-tracked agent lineage on ``Session.task_lineage``.

The reasoning judge identifies the reasoning agent via ``current_agent_id``
which used to be derived from ``task.assignee_agent_id`` — the static plan
intent. When a coordinator delegates to a child via AgentTool the child
reasons under the parent's task pin and the judge mis-attributed every
drift to the coordinator. The fix tracks the *observed* agent lineage so
consumers can distinguish "delegated child of the assignee" from
"off-plan agent".

Lifecycle contract exercised here:

* ``mark_task_running`` initialises ``session.task_lineage[task.id]``
  with ``{task.assignee_agent_id}``.
* ``before_tool_callback`` AgentTool path adds ``to_agent`` to that set
  whenever ``session.current_task_id`` matches the in-progress task.
* Multiple delegations under the same task accumulate (idempotent
  ``set.add``).
* Each terminal transition (``mark_task_completed`` /
  ``mark_task_failed`` / ``mark_task_cancelled`` /
  ``mark_task_not_needed``) drops the lineage entry.
* The cascade path also clears entries for downstream cancelled tasks
  (it bypasses ``mark_task_cancelled`` and does the cleanup inline).
* When ``session.current_task_id`` is empty at delegation time the
  lineage update is a no-op (race with task in-progress is fine).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)


class _NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover - convenience
        return None


def _build_session_with_running_task(
    *, task_id: str = "t1", assignee: str = "coordinator"
) -> tuple[DefaultSteerer, Session, Task]:
    task = Task(id=task_id, title="Demo", assignee_agent_id=assignee)
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="demo goal")],
        plan=plan,
    )
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    return steerer, session, task


async def test_lineage_initialises_on_task_running() -> None:
    """``mark_task_running`` seeds ``task_lineage`` with the assignee."""
    steerer, session, task = _build_session_with_running_task(assignee="coordinator")
    assert task.id not in session.task_lineage  # pre-condition

    await steerer.mark_task_running(task.id, session=session)

    assert session.task_lineage[task.id] == {"coordinator"}


async def test_lineage_initialises_empty_set_when_no_assignee() -> None:
    """Plans built without an assignee still get a lineage entry (empty set).

    A blank assignee should not leave ``task_lineage`` keyless on the
    in-progress task — downstream consumers expect ``task_id in lineage``
    to be the "is this task running" signal, distinct from "what agents
    have we observed".
    """
    steerer, session, task = _build_session_with_running_task(assignee="")
    await steerer.mark_task_running(task.id, session=session)
    assert session.task_lineage[task.id] == set()


async def test_delegation_observed_extends_lineage_on_pinned_task() -> None:
    """Simulating the ``before_tool_callback`` lineage write.

    The plugin calls ``session.task_lineage[current_task_id].add(to_agent)``
    when an AgentTool dispatch fires. We exercise the contract directly
    rather than going through the full ADK plugin so the test stays
    framework-neutral.
    """
    steerer, session, task = _build_session_with_running_task(assignee="coordinator")
    await steerer.mark_task_running(task.id, session=session)
    assert session.current_task_id == task.id

    # Simulate the plugin's idempotent set.add on a delegated child.
    session.task_lineage[session.current_task_id].add("research_agent")
    assert session.task_lineage[task.id] == {"coordinator", "research_agent"}


async def test_lineage_accumulates_multiple_delegations() -> None:
    """Two delegations under the same task end up in one set."""
    steerer, session, task = _build_session_with_running_task(assignee="coordinator")
    await steerer.mark_task_running(task.id, session=session)

    session.task_lineage[task.id].add("research_agent")
    session.task_lineage[task.id].add("writer_agent")
    # Idempotent re-add is a no-op.
    session.task_lineage[task.id].add("research_agent")

    assert session.task_lineage[task.id] == {
        "coordinator",
        "research_agent",
        "writer_agent",
    }


async def test_lineage_clears_on_completed() -> None:
    steerer, session, task = _build_session_with_running_task()
    await steerer.mark_task_running(task.id, session=session)
    assert task.id in session.task_lineage

    await steerer.mark_task_completed(task.id, session=session)
    assert task.id not in session.task_lineage


async def test_lineage_clears_on_failed_recoverable() -> None:
    steerer, session, task = _build_session_with_running_task()
    await steerer.mark_task_running(task.id, session=session)

    await steerer.mark_task_failed(task.id, session=session, recoverable=True)
    assert task.id not in session.task_lineage


async def test_lineage_clears_on_failed_fatal() -> None:
    steerer, session, task = _build_session_with_running_task()
    await steerer.mark_task_running(task.id, session=session)

    await steerer.mark_task_failed(task.id, session=session, recoverable=False)
    assert task.id not in session.task_lineage


async def test_lineage_clears_on_cancelled() -> None:
    steerer, session, task = _build_session_with_running_task()
    await steerer.mark_task_running(task.id, session=session)

    await steerer.mark_task_cancelled(task.id, session=session)
    assert task.id not in session.task_lineage


async def test_lineage_clears_on_not_needed() -> None:
    """``NOT_NEEDED`` is the closest TaskStatus analog to "superseded".

    The reconciler marks redundant pending tasks NOT_NEEDED post-
    invocation. The lineage entry should be dropped just like any
    other terminal transition.
    """
    steerer, session, task = _build_session_with_running_task()
    await steerer.mark_task_running(task.id, session=session)

    await steerer.mark_task_not_needed(task.id, session=session)
    assert task.id not in session.task_lineage


async def test_lineage_clears_on_cascade_cancel_downstream() -> None:
    """The cascade BFS path also clears lineage for downstream tasks.

    ``cascade_cancel_downstream`` deliberately does not recurse through
    ``mark_task_cancelled``; it transitions tasks in place. We mirror
    the cleanup so a fatal upstream failure that cancels three
    downstream tasks leaves no orphaned lineage entries.
    """
    upstream = Task(id="u", title="Upstream", assignee_agent_id="a1")
    downstream = Task(id="d", title="Downstream", assignee_agent_id="a2")
    # goldfive#247: build the plan with edges at construction (Plan
    # is frozen — no in-place mutation).
    from goldfive.types import TaskEdge

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[upstream, downstream],
        edges=[TaskEdge(from_task_id="u", to_task_id="d")],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="demo")],
        plan=plan,
    )
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())

    await steerer.mark_task_running("u", session=session)
    await steerer.mark_task_running("d", session=session)
    assert "u" in session.task_lineage
    assert "d" in session.task_lineage

    # Fatal failure on upstream cascades cancellation to downstream;
    # both lineage entries should be cleared.
    await steerer.mark_task_failed("u", session=session, recoverable=False)
    assert "u" not in session.task_lineage
    assert "d" not in session.task_lineage
    # goldfive#247: read live status from session.plan; the local
    # ``downstream`` reference is the pre-mutation snapshot.
    assert session.plan is not None
    live_d = next(t for t in session.plan.tasks if t.id == "d")
    assert live_d.status is TaskStatus.CANCELLED


async def test_delegation_observed_with_no_pinned_task_is_noop() -> None:
    """If ``current_task_id`` is empty at delegation time, no lineage write.

    The plugin guards on ``current_task_id`` and ``pinned_task_id in
    session.task_lineage`` before writing, so a delegation observed
    before any task transitioned to RUNNING (race with task in_progress)
    is silently dropped.
    """
    session = Session(run_id="r1")
    assert session.current_task_id == ""
    assert session.task_lineage == {}

    # Mimic the plugin's guarded write — same condition as in
    # ``before_tool_callback``: only write when both the pin AND the
    # lineage key are present. With both empty, nothing changes.
    pinned = session.current_task_id
    if pinned and pinned in session.task_lineage:  # pragma: no cover - intentional
        session.task_lineage[pinned].add("research_agent")
    assert session.task_lineage == {}
