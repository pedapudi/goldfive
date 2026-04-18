"""CallableAdapter — the reference :class:`AgentAdapter` implementation.

Wraps an async callable of shape::

    async def agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult

The callable is free to invoke any of the registered reporting tool
handlers directly (they are plain awaitables) to drive state transitions.
This adapter is the canonical example other adapters (ADK, Claude) follow,
and is the preferred vehicle for deterministic tests of the orchestration
layer because it removes LLM non-determinism from the loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from goldfive.reporting import ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task

AgentCallable = Callable[
    [Task, Session, list[ReportingToolSpec]],
    Awaitable[InvocationResult],
]


class CallableAdapter:
    """Adapter that delegates :meth:`invoke` to a user-supplied async callable.

    The adapter holds the reporting-tool specs registered by the executor
    and forwards them to the callable on every invocation, so the callable
    can invoke any reporting tool handler it needs (e.g. ``report_task_started``,
    ``report_task_completed``) to drive the session state machine.

    Parameters
    ----------
    agent:
        Async callable invoked for each task.
    available_agents:
        Optional list of agent identifiers this adapter can dispatch to.
        Exposed via the :attr:`available_agents` property so planners can
        enumerate routable agents. Defaults to an empty list.
    """

    def __init__(
        self,
        agent: AgentCallable,
        *,
        available_agents: list[str] | None = None,
    ) -> None:
        self._agent = agent
        self._available_agents: list[str] = list(available_agents) if available_agents else []
        self._tools: list[ReportingToolSpec] = []

    async def register_reporting_tools(self, tools: list[ReportingToolSpec]) -> None:
        """Store the reporting tool specs for later forwarding to the callable."""
        self._tools = list(tools)

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        """Invoke the wrapped callable and return its :class:`InvocationResult`."""
        return await self._agent(task, session, self._tools)

    @property
    def available_agents(self) -> list[str]:
        """Return the configured list of available agent identifiers."""
        return list(self._available_agents)
