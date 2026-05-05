"""Conversation-end orphan PENDING audit at Runner.close (goldfive#212).

The seam is :meth:`Runner.close`: it already iterates per-key
``Conversation`` entries and emits a ``ConversationEnded`` marker for
each. PR #339 added a per-turn reachability audit inside the executor;
this test suite exercises the conversation-scope counterpart that runs
inside ``close()``. At conversation end there is no next turn, so
every still-PENDING task is by definition orphaned (no engaging
turn will ever pick it up). The audit cancels them in place via
``steerer.transition(..., TaskStatus.CANCELLED, cancel_reason=
'conversation_ended:no_engaging_turn', ...)`` BEFORE the
``ConversationEnded`` marker fires, so persisted logs see a coherent
plan-end state.

These tests construct a Runner with stub components — no ADK, no LLM,
no executor.run() — and drive ``close()`` directly after manually
seeding the per-key conversation bookkeeping that a real turn would
have populated. The seam under test is the close-time audit, not the
turn lifecycle that produced the PENDING tasks.
"""

from __future__ import annotations

from typing import Any

from goldfive.conversation import Conversation
from goldfive.protocols import EventSink
from goldfive.runner import Runner
from goldfive.types import (
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs.
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True

    def payload_kinds(self) -> list[str]:
        kinds: list[str] = []
        for e in self.events:
            if hasattr(e, "WhichOneof"):
                kinds.append(e.WhichOneof("payload") or "")
            elif isinstance(e, dict):
                kinds.append(e.get("kind", ""))
            else:
                kinds.append(getattr(e, "payload_kind", ""))
        return kinds


class StubSteerer:
    """Records every transition the audit triggers and emits a synthetic
    ``TaskCancelled`` envelope onto every bound sink.

    Mirrors the real :class:`DefaultSteerer.mark_task_cancelled` shape
    only as far as the tests need: it (1) no-ops on already-terminal
    tasks (so idempotency is observable), (2) flips ``task.status`` so a
    second pass sees no PENDING, and (3) emits a TaskCancelled envelope
    on each bound sink so we can verify event order vs.
    ``ConversationEnded``.
    """

    _TERMINAL = frozenset(
        {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.NOT_NEEDED,
        }
    )

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.transitions: list[tuple[str, TaskStatus, str]] = []

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:  # noqa: ARG002
        return None

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",  # noqa: ARG002
        session: Session,
        cancel_reason: str = "",
        source: str = "other",  # noqa: ARG002
    ) -> None:
        # Locate the task; bail if missing or already terminal so the
        # idempotency invariant ("second close() is a no-op") matches
        # the real steerer.
        task = None
        if session.plan is not None:
            for t in session.plan.tasks:
                if t.id == task_id:
                    task = t
                    break
        if task is None or task.status in self._TERMINAL:
            return
        self.transitions.append((task_id, to, cancel_reason))
        # goldfive#247: Plan + Task are frozen — derive a new plan via
        # with_task_status and swap. Mirrors what DefaultSteerer does.
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )
        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))
        # Emit a synthetic envelope on every bound sink so tests can
        # verify event order. We don't depend on the goldfive.events
        # factory here — a dict is enough for ordering / payload-kind
        # assertions and avoids tight coupling to proto regen.
        if to is TaskStatus.CANCELLED:
            for sink in self._sinks:
                await sink.emit(
                    {
                        "kind": "task_cancelled",
                        "task_id": task_id,
                        "reason": cancel_reason,
                    }
                )

    async def shutdown(self) -> None:
        return None


class StubPlanner:
    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class StubAdapter:
    """Minimal adapter — Runner.close never calls into it."""

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    async def invoke(self, task: Any, session: Any) -> Any:  # noqa: ARG002
        return None


class StubExecutor:
    """Minimal executor — Runner.close never calls into it."""

    async def run(self, **kwargs: Any) -> Any:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Plan helpers.
# ---------------------------------------------------------------------------


def _three_task_plan_one_completed() -> Plan:
    """a COMPLETED -> b PENDING -> c PENDING."""
    return Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="a", title="task a", status=TaskStatus.COMPLETED),
            Task(id="b", title="task b", status=TaskStatus.PENDING),
            Task(id="c", title="task c", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="b", to_task_id="c"),
        ],
    )


def _all_terminal_plan() -> Plan:
    """Every task already terminal — nothing for the audit to do."""
    return Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="a", title="a", status=TaskStatus.COMPLETED),
            Task(id="b", title="b", status=TaskStatus.CANCELLED),
            Task(id="c", title="c", status=TaskStatus.FAILED),
        ],
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="b", to_task_id="c"),
        ],
    )


def _failed_with_replacement_plan() -> Plan:
    """a COMPLETED, b FAILED, retry_b PENDING (supersedes=b), c PENDING (depends on retry_b)."""
    return Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="a", title="a", status=TaskStatus.COMPLETED),
            Task(id="b", title="b", status=TaskStatus.FAILED),
            Task(
                id="retry_b",
                title="retry b",
                status=TaskStatus.PENDING,
                supersedes="b",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="c", title="c", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="retry_b", to_task_id="c"),
        ],
    )


def _make_runner(sinks: list[EventSink]) -> tuple[Runner, StubSteerer]:
    steerer = StubSteerer()
    runner = Runner(
        agent=StubAdapter(),
        planner=StubPlanner(),
        executor=StubExecutor(),
        steerer=steerer,
        sinks=sinks,
        # Off so the steerer-attribute fixup in __init__ never tries
        # to detach a goal-drift judge that doesn't exist.
        goal_drift_enabled=False,
    )
    # The Runner only binds the steerer to sinks inside ``run()``.
    # Since we drive ``close()`` directly, do that wiring by hand so
    # the audit's TaskCancelled envelopes flow through every sink.
    steerer.bind(sinks=runner.sinks, planner=runner.planner)
    return runner, steerer


def _seed_announced_conversation(
    runner: Runner,
    *,
    plan: Plan | None,
    key: str = "",
) -> tuple[Conversation, Session | None]:
    """Mimic the bookkeeping a real turn does so close() observes a
    conversation that actually announced and an anchor session with
    the given plan attached.
    """
    conv = Conversation.new()
    runner._conversations[key] = conv
    runner._conversation_announced[key] = True
    if plan is None:
        # The "anchor=None" no-op variant.
        runner._last_session_by_key[key] = None
        return conv, None
    session = Session(run_id="r1", conversation_id=conv.id)
    session.plan = plan
    runner._last_session_by_key[key] = session
    return conv, session


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


async def test_close_cancels_orphan_pending_at_session_end() -> None:
    """Plan a COMPLETED -> b PENDING -> c PENDING. close() cancels b and c."""
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    plan = _three_task_plan_one_completed()
    _seed_announced_conversation(runner, plan=plan)

    await runner.close()

    cancelled_ids = {tid for (tid, to, _r) in steerer.transitions if to is TaskStatus.CANCELLED}
    assert cancelled_ids == {"b", "c"}, (
        f"expected b and c cancelled, got {cancelled_ids!r}"
    )
    # All transitions carried the structured cancel reason.
    for tid, to, reason in steerer.transitions:
        if to is TaskStatus.CANCELLED:
            assert reason == "conversation_ended:no_engaging_turn", (
                f"task {tid!r} cancelled with unexpected reason {reason!r}"
            )

    # Sink event order: TaskCancelled events come BEFORE ConversationEnded.
    kinds = sink.payload_kinds()
    cancelled_idx = [i for i, k in enumerate(kinds) if k == "task_cancelled"]
    ended_idx = [i for i, k in enumerate(kinds) if k == "conversation_ended"]
    assert cancelled_idx, f"no task_cancelled events emitted, got kinds {kinds!r}"
    assert ended_idx, f"no conversation_ended event emitted, got kinds {kinds!r}"
    assert max(cancelled_idx) < min(ended_idx), (
        f"TaskCancelled must precede ConversationEnded; kinds={kinds!r}"
    )

    # Plan now has both PENDING tasks flipped to CANCELLED.
    # goldfive#247: read from the live session.plan; the local
    # ``plan`` reference is the pre-mutation snapshot.
    live_session = runner._last_session_by_key.get("")
    assert live_session is not None and live_session.plan is not None
    by_id = {t.id: t.status for t in live_session.plan.tasks}
    assert by_id["a"] is TaskStatus.COMPLETED
    assert by_id["b"] is TaskStatus.CANCELLED
    assert by_id["c"] is TaskStatus.CANCELLED


async def test_close_idempotent_audit() -> None:
    """Calling close() twice cancels each orphan exactly once."""
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    plan = _three_task_plan_one_completed()
    _seed_announced_conversation(runner, plan=plan)

    await runner.close()
    transitions_after_first = list(steerer.transitions)
    events_after_first = list(sink.events)

    await runner.close()  # second call is a no-op (Runner._closed gate)

    assert steerer.transitions == transitions_after_first, (
        f"second close() emitted extra transitions: "
        f"{steerer.transitions[len(transitions_after_first):]!r}"
    )
    assert sink.events == events_after_first, (
        "second close() emitted extra sink events"
    )
    # And exactly one cancellation per orphan task — never two.
    cancel_count: dict[str, int] = {}
    for tid, to, _r in steerer.transitions:
        if to is TaskStatus.CANCELLED:
            cancel_count[tid] = cancel_count.get(tid, 0) + 1
    assert cancel_count == {"b": 1, "c": 1}, (
        f"expected one cancel per orphan, got {cancel_count!r}"
    )


async def test_close_with_no_announced_conversation_no_op() -> None:
    """No announced conversation -> close() emits zero TaskCancelled events."""
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    # No call to _seed_announced_conversation: every key has
    # announced=False (the Runner __init__ default), so the audit
    # branch never runs.

    await runner.close()

    cancelled_transitions = [
        (tid, to) for (tid, to, _r) in steerer.transitions if to is TaskStatus.CANCELLED
    ]
    assert cancelled_transitions == [], (
        f"expected no cancellations on un-announced runner, got {cancelled_transitions!r}"
    )
    assert "task_cancelled" not in sink.payload_kinds()
    assert "conversation_ended" not in sink.payload_kinds()


async def test_close_with_terminal_only_plan_no_op() -> None:
    """Plan with all tasks terminal -> close() emits zero TaskCancelled events."""
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    plan = _all_terminal_plan()
    _seed_announced_conversation(runner, plan=plan)

    await runner.close()

    cancelled_transitions = [
        (tid, to) for (tid, to, _r) in steerer.transitions if to is TaskStatus.CANCELLED
    ]
    assert cancelled_transitions == [], (
        f"expected no cancellations on all-terminal plan, got {cancelled_transitions!r}"
    )
    assert "task_cancelled" not in sink.payload_kinds()
    # ConversationEnded still fires — that's the existing contract,
    # unchanged by the audit.
    assert "conversation_ended" in sink.payload_kinds()


async def test_close_with_failed_task_having_live_replacement_only_cancels_pending() -> None:
    """The audit cancels every PENDING — including a live replacement
    spawned by a refine. ``b`` is already FAILED (terminal) and the
    steerer's no-op-on-terminal guard means the audit doesn't try to
    re-cancel it. ``retry_b`` and ``c`` are PENDING and get cancelled.

    This is the deliberately-broad behaviour: at conversation end any
    PENDING is orphaned, including a refine's replacement task that
    never got to run a turn.
    """
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    plan = _failed_with_replacement_plan()
    _seed_announced_conversation(runner, plan=plan)

    await runner.close()

    cancelled_ids = {tid for (tid, to, _r) in steerer.transitions if to is TaskStatus.CANCELLED}
    assert cancelled_ids == {"retry_b", "c"}, (
        f"expected retry_b and c cancelled, got {cancelled_ids!r}"
    )
    # b stays FAILED — the steerer's terminal guard prevented a
    # double-transition even if the audit had asked.
    # goldfive#247: read from the live session.plan after the swap.
    live_session = runner._last_session_by_key.get("")
    assert live_session is not None and live_session.plan is not None
    by_id = {t.id: t.status for t in live_session.plan.tasks}
    assert by_id["a"] is TaskStatus.COMPLETED
    assert by_id["b"] is TaskStatus.FAILED, (
        f"b must remain FAILED (terminal); got {by_id['b']}"
    )
    assert by_id["retry_b"] is TaskStatus.CANCELLED
    assert by_id["c"] is TaskStatus.CANCELLED


async def test_close_with_anchor_none_no_op() -> None:
    """Conversation announced but the per-key anchor is None — the
    Runner.close loop skips the whole entry (its existing
    ``announced and anchor is not None`` gate). Nothing crashes; no
    audit fires.
    """
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    _seed_announced_conversation(runner, plan=None)  # anchor stays None

    await runner.close()  # must not raise

    cancelled_transitions = [
        (tid, to) for (tid, to, _r) in steerer.transitions if to is TaskStatus.CANCELLED
    ]
    assert cancelled_transitions == []
    assert "task_cancelled" not in sink.payload_kinds()
    # No ConversationEnded either — the same gate that guarded the
    # audit also guarded the marker emission.
    assert "conversation_ended" not in sink.payload_kinds()


async def test_close_event_kind_is_task_cancelled_not_new_event() -> None:
    """The audit reuses ``TaskCancelled`` with the structured cancel
    reason. No new ``conversation_ended_audit`` (or similar) event
    kind is minted — sinks that don't know goldfive#212 can still
    surface the orphans as cancellations.
    """
    sink = RecordingSink()
    runner, steerer = _make_runner([sink])
    plan = _three_task_plan_one_completed()
    _seed_announced_conversation(runner, plan=plan)

    await runner.close()

    kinds = sink.payload_kinds()
    # Exactly the expected kinds: task_cancelled (x2) then conversation_ended.
    assert kinds.count("task_cancelled") == 2, (
        f"expected exactly 2 task_cancelled events, got kinds {kinds!r}"
    )
    assert kinds.count("conversation_ended") == 1, (
        f"expected exactly 1 conversation_ended event, got kinds {kinds!r}"
    )
    # No fabricated event kinds.
    allowed = {"task_cancelled", "conversation_ended"}
    leaked = [k for k in kinds if k not in allowed and k]
    assert not leaked, f"unexpected event kinds emitted by audit: {leaked!r}"
    # And the cancel_reason carries the structured tag.
    cancel_reasons = {
        reason for (_tid, to, reason) in steerer.transitions if to is TaskStatus.CANCELLED
    }
    assert cancel_reasons == {"conversation_ended:no_engaging_turn"}, (
        f"unexpected cancel reasons: {cancel_reasons!r}"
    )
