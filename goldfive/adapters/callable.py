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
from typing import Any

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

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",  # noqa: ARG002 -- part of the protocol
    ) -> None:
        """Route a reasoning-content block to the bound steerer (if any).

        :class:`CallableAdapter` has no intrinsic way to capture reasoning
        (the wrapped callable is opaque), so this entry point is exposed
        for tests and for callables that choose to forward reasoning
        themselves.
        """
        steerer = getattr(self, "_steerer", None)
        if steerer is None:
            return
        observe = getattr(steerer, "observe_reasoning", None)
        if observe is None:
            return
        await observe(text, task=task, session=session, provider=provider)

    def bind_steerer(self, steerer: object | None) -> None:
        """Attach the active :class:`~goldfive.protocols.Steerer`.

        Enables :meth:`emit_reasoning` to route into the steerer. Safe
        to call with ``None`` to unbind.
        """
        self._steerer = steerer

    @property
    def available_agents(self) -> list[str]:
        """Return the configured list of available agent identifiers."""
        return list(self._available_agents)

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        """Return a flat single-level tree describing the configured agents.

        CallableAdapter has no real tree — every configured name is
        rendered as a depth-0 root leaf so planners that consume
        :attr:`available_agents_tree` (goldfive#151) see a consistent
        shape. Adapters that model a real tree (ADK) override with a
        richer walker.
        """
        return [
            {
                "name": name,
                "depth": 0,
                "parent": "",
                "role": "root",
                "kind": "Callable",
            }
            for name in self._available_agents
        ]
