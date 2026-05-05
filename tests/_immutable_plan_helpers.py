"""Shared test helpers for migrating to the goldfive#247 frozen-Plan world.

Tests built before the freeze used the in-place mutation idioms:

* ``session.plan.tasks[i].status = TaskStatus.X``
* ``session.plan.tasks.append(Task(...))``
* ``session.plan.edges.append(TaskEdge(...))``

Under :class:`goldfive.types.Plan` ``frozen=True`` those raise
:class:`dataclasses.FrozenInstanceError`. The helpers below give tests
a one-call equivalent that derives a new :class:`Plan` and swaps it
onto the session inside :func:`channel_processor_active`, satisfying
the runtime single-writer check.

Use these helpers rather than re-importing the underlying primitives
in every test — keeps the migration churn small and consistent.
"""

from __future__ import annotations

from goldfive.types import (
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
    add_tasks,
    channel_processor_active,
    replace_edges,
    replace_task,
    set_session_plan,
    with_task_status,
)


def force_task_status(session: Session, task_id: str, status: TaskStatus) -> None:
    """Synchronously force ``task_id``'s status on ``session.plan``.

    Test-only equivalent of the pre-#247 ``task.status = X`` setup
    pattern. Wraps the swap in :func:`channel_processor_active` so
    the runtime single-writer check (which fires on every
    :func:`set_session_plan` call) treats this as a legitimate
    channel-processor write — tests are the sole writer here.
    """
    assert session.plan is not None
    with channel_processor_active():
        set_session_plan(session, with_task_status(session.plan, task_id, status))


def force_task_replace(session: Session, task_id: str, **changes: object) -> None:
    """Replace ``task_id``'s fields with ``changes`` (e.g. ``status=...``,
    ``cancel_reason=...``). See :func:`force_task_status` for rationale.
    """
    assert session.plan is not None
    with channel_processor_active():
        set_session_plan(session, replace_task(session.plan, task_id, **changes))


def append_tasks(session: Session, tasks: list[Task]) -> None:
    """Append ``tasks`` onto ``session.plan.tasks``."""
    assert session.plan is not None
    with channel_processor_active():
        set_session_plan(session, add_tasks(session.plan, tasks))


def append_edge(session: Session, frm: str, to: str) -> None:
    """Append a single edge onto ``session.plan.edges``."""
    assert session.plan is not None
    new_edges = list(session.plan.edges) + [TaskEdge(from_task_id=frm, to_task_id=to)]
    with channel_processor_active():
        set_session_plan(session, replace_edges(session.plan, new_edges))


def force_plan(session: Session, plan: Plan | None) -> None:
    """Swap ``session.plan`` to ``plan`` under the channel processor.

    Test-only — production code paths route through the steerer's
    :meth:`_apply_revision`. Tests sometimes need to install a
    pre-built plan directly (no steerer hooked up); this is the
    one-call form.
    """
    with channel_processor_active():
        set_session_plan(session, plan)
