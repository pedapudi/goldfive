from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from goldfive.conversation import TurnRecord
    from goldfive.reporting import ReportingToolSpec
    from goldfive.results import ExecutionOutcome, InvocationResult
    from goldfive.types import (
        DriftEvent,
        Goal,
        ObservedAction,
        Plan,
        Session,
        Task,
        TaskStatus,
    )


@runtime_checkable
class GoalDeriver(Protocol):
    """Derives a list of ``Goal`` objects from free-form user input."""

    async def derive(
        self,
        user_input: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[Goal]: ...


@runtime_checkable
class Planner(Protocol):
    """Generates and refines ``Plan`` instances from goals and drift events.

    The optional :meth:`handle_turn` is the goldfive#271 Phase 4
    consolidated entrypoint: a single planner LLM call decides whether
    the new user_input requires a plan change at all, and — when one
    is warranted — produces the next plan in the same response. The
    Runner prefers it over the legacy ``generate`` / ``refine`` pair on
    every turn after the first; legacy planners that don't implement
    it fall back to the unconditional ``generate`` path.
    """

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str] | list[dict[str, Any]] | None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None: ...

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
        observed_actions: list[ObservedAction] | None = None,
        available_agents: list[str] | list[dict[str, Any]] | None = None,
    ) -> Plan | None: ...

    # Optional — checked via ``hasattr(planner, "handle_turn")``. Legacy
    # planners (PassthroughPlanner, third-party stubs that predate #271)
    # may omit it; the Runner falls through to ``generate`` for them.
    #
    # Returns ``None`` when the user_input is purely conversational and
    # the current revision still describes the right work. Returns the
    # next :class:`Plan` revision when a plan change is warranted; the
    # Runner installs it as a revision of ``session.plan``
    # (revision_index += 1) via the unified install path. The Runner
    # guarantees ``session.plan`` is non-None on every turn (it seeds
    # :meth:`Plan.empty` on the first turn so the planner produces
    # revision 1 against an empty prior).
    async def handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        conversation_history: list[TurnRecord],
        available_agents: list[str] | list[dict[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None: ...


@runtime_checkable
class Steerer(Protocol):
    """Observes execution events, drives task transitions, and detects drift."""

    async def observe(self, event: Any, session: Session) -> None: ...

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        agent_name: str = "",
    ) -> None: ...

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
        cancel_reason: str = "",
    ) -> None: ...

    async def cascade_cancel_downstream(
        self,
        session: Session,
        cancelled_id: str,
    ) -> None:
        """BFS-cancel every downstream non-terminal task of ``cancelled_id``.

        Shared cancellation-fanout primitive for both PLAN-LIFECYCLE.md
        §6.2 (unrecoverable cascade) and §6.3 (cancel cascade). A
        conforming Steerer MUST:

        - Walk the current plan's forward edges from ``cancelled_id``.
        - Transition every reachable non-terminal task to CANCELLED.
        - Emit exactly one ``TaskCancelled`` event per transitioned
          task, with a reason that identifies ``cancelled_id`` as the
          cascade source.
        - Skip tasks already in a terminal status (no re-cancellation,
          no event re-emission).
        - De-duplicate diamond-DAG reachability (emit at most one event
          per downstream task per call).

        The initiator (``cancelled_id``) itself is *not* transitioned
        or emitted here — callers own that transition before invoking
        this primitive.
        """

    def detect_drift(
        self,
        event: Any,
        session: Session,
    ) -> DriftEvent | None: ...

    # Called by executors to wire sinks/planner into the steerer.
    def bind(
        self,
        *,
        sinks: list[EventSink],
        planner: Planner,
    ) -> None: ...


@runtime_checkable
class AgentAdapter(Protocol):
    """Wraps an underlying agent framework (ADK, Claude Agent SDK, etc.)."""

    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None: ...

    async def invoke(
        self,
        task: Task,
        session: Session,
    ) -> InvocationResult: ...

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",
        agent_name: str = "",
    ) -> None: ...

    @property
    def available_agents(self) -> list[str]: ...

    # NOTE: goldfive#151 added a structured ``available_agents_tree``
    # property on the shipped adapters (ADK, Claude, Callable) for the
    # tree-aware planner, but it is intentionally *not* part of the
    # Protocol so custom / legacy adapters that only expose
    # ``available_agents`` still pass ``isinstance(x, AgentAdapter)``
    # checks. Call sites look the attribute up via ``getattr`` and
    # fall back to the flat list when absent.


@runtime_checkable
class Executor(Protocol):
    """Executes a ``Plan`` by dispatching tasks to the adapter and steerer.

    Overlay-model executors (goldfive#141) additionally accept a
    ``user_input`` kwarg carrying the caller's original string input
    so they can forward it verbatim to ``adapter.invoke_passthrough``.
    Legacy per-task executors simply ignore the kwarg.
    """

    async def run(
        self,
        *,
        plan: Plan,
        session: Session,
        adapter: AgentAdapter,
        steerer: Steerer,
        planner: Planner,
        sinks: list[EventSink],
    ) -> ExecutionOutcome: ...


@runtime_checkable
class EventSink(Protocol):
    """Receives proto ``Event`` messages emitted by executors and steerers."""

    async def emit(self, event_pb: Any) -> None: ...  # pb Event message

    async def close(self) -> None: ...
