from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from goldfive.reporting import ReportingToolSpec
    from goldfive.results import ExecutionOutcome, InvocationResult
    from goldfive.types import (
        DriftEvent,
        Goal,
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
    """Generates and refines ``Plan`` instances from goals and drift events."""

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None: ...

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
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
    ) -> None: ...

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None: ...

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
    ) -> None: ...

    @property
    def available_agents(self) -> list[str]: ...


@runtime_checkable
class Executor(Protocol):
    """Executes a ``Plan`` by dispatching tasks to the adapter and steerer."""

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
