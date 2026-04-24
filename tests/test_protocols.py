from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from goldfive.protocols import (
    AgentAdapter,
    EventSink,
    Executor,
    GoalDeriver,
    Planner,
    Steerer,
)

ALL_PROTOCOLS = (GoalDeriver, Planner, Steerer, AgentAdapter, Executor, EventSink)


@pytest.mark.parametrize("proto", ALL_PROTOCOLS)
def test_protocol_is_runtime_checkable(proto: type) -> None:
    # ``@runtime_checkable`` sets this dunder attribute on the Protocol class.
    assert getattr(proto, "_is_runtime_protocol", False) is True


# ---------------------------------------------------------------------------
# Trivial stubs implementing each Protocol's surface area.
# ---------------------------------------------------------------------------


class _GoalDeriverStub:
    async def derive(
        self,
        user_input: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        return []


class _PlannerStub:
    async def generate(
        self,
        *,
        goals: list[Any],
        available_agents: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Any | None:
        return None

    async def refine(
        self,
        *,
        plan: Any,
        drift: Any,
        goals: list[Any],
    ) -> Any | None:
        return None


class _SteererStub:
    async def observe(self, event: Any, session: Any) -> None:
        return None

    async def observe_reasoning(
        self,
        text: str,
        *,
        task: Any = None,
        session: Any,
        provider: str = "",
        agent_name: str = "",
    ) -> None:
        return None

    async def transition(
        self,
        task_id: str,
        to: Any,
        *,
        detail: str = "",
        session: Any,
    ) -> None:
        return None

    async def cascade_cancel_downstream(self, session: Any, cancelled_id: str) -> None:
        return None

    def detect_drift(self, event: Any, session: Any) -> Any | None:
        return None

    def bind(self, *, sinks: list[Any], planner: Any) -> None:
        return None


class _AgentAdapterStub:
    async def register_reporting_tools(self, tools: list[Any]) -> None:
        return None

    async def invoke(self, task: Any, session: Any) -> Any:
        return None

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Any = None,
        session: Any,
        provider: str = "",
        call_id: str = "",
    ) -> None:
        return None

    @property
    def available_agents(self) -> list[str]:
        return []


class _ExecutorStub:
    async def run(
        self,
        *,
        plan: Any,
        session: Any,
        adapter: Any,
        steerer: Any,
        planner: Any,
        sinks: list[Any],
    ) -> Any:
        return None


class _EventSinkStub:
    async def emit(self, event_pb: Any) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "stub, proto",
    [
        (_GoalDeriverStub(), GoalDeriver),
        (_PlannerStub(), Planner),
        (_SteererStub(), Steerer),
        (_AgentAdapterStub(), AgentAdapter),
        (_ExecutorStub(), Executor),
        (_EventSinkStub(), EventSink),
    ],
)
def test_stub_satisfies_protocol(stub: object, proto: type) -> None:
    assert isinstance(stub, proto)
