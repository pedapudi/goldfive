"""AgentTool-per-invoke cap tests (goldfive#130).

The plugin counts AgentTool spawns scoped to a single top-level
invocation. When the count exceeds ``ADKAdapter(agent_tool_cap=N)``,
the plugin:

* emits a ``RUNAWAY_DELEGATION`` drift at CRITICAL severity,
* sets ``plugin.runaway_delegation_tripped`` so the adapter's invoke
  loop breaks,
* short-circuits subsequent AgentTool calls in the same invocation
  with a "skipped" dict so the runner wraps up quickly.

Under the single-Runner model this is the backstop for a user-supplied
coordinator whose prompt describes a pipeline and keeps delegating via
AgentTool. Goldfive cannot require prompt cooperation (users bring
their own trees), so a hard per-invocation cap is the only
architecture-independent guard.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")


def _make_tool_looper_llm(tool_name: str, calls: int) -> Any:
    """LLM that calls an AgentTool ``calls`` times then yields a final text."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _Looper(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            if self._step <= calls:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id=f"c{self._step}",
                                    name=tool_name,
                                    args={"request": "go"},
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="done")],
                    ),
                    turn_complete=True,
                )

    return _Looper


def _make_quiet_llm() -> Any:
    """Sub-agent LLM that just says "ok" and terminates."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _Quiet(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    return _Quiet


class _RecordingSteerer:
    """Steerer stub that records drift events routed through
    ``_handle_drift`` so tests can assert on the RUNAWAY_DELEGATION
    emission without spinning up a full ``DefaultSteerer``.
    """

    def __init__(self) -> None:
        self._sinks: list[Any] = []
        self.drifts: list[Any] = []

    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass

    async def _handle_drift(self, drift: Any, session: Any) -> None:  # noqa: ARG002
        self.drifts.append(drift)


async def test_below_cap_runs_cleanly_with_no_drift() -> None:
    """A coordinator that delegates ``cap`` times (at the boundary) must
    NOT trip the cap — the strict-greater-than comparison means the
    Nth call is the last allowed one.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    cap = 4
    sub = Agent(name="sub", model=_make_quiet_llm()(), instruction="")
    coord = Agent(
        name="coord",
        model=_make_tool_looper_llm("sub", calls=cap)(),
        instruction="",
        tools=[AgentTool(sub)],
    )
    adapter = ADKAdapter(coord, agent_tool_cap=cap)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="go")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    result = await adapter.invoke(task=task, session=session)

    # No drift — we hit the boundary but did not exceed it.
    runaway = [d for d in steerer.drifts if d.kind.value == "runaway_delegation"]
    assert runaway == [], f"cap {cap} with exactly {cap} calls must not trip; got {runaway}"
    # Run completed normally.
    assert result.stop_reason != "runaway_delegation"


async def test_above_cap_trips_drift_and_short_circuits() -> None:
    """When the coordinator delegates ``cap + 1`` times the plugin emits
    RUNAWAY_DELEGATION and the adapter's invoke breaks out of the loop.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import DriftKind, DriftSeverity, Plan, Session, Task

    cap = 3
    sub = Agent(name="sub", model=_make_quiet_llm()(), instruction="")
    # Attempt to call the AgentTool 10 times — well above the cap.
    coord = Agent(
        name="coord",
        model=_make_tool_looper_llm("sub", calls=10)(),
        instruction="",
        tools=[AgentTool(sub)],
    )
    adapter = ADKAdapter(coord, agent_tool_cap=cap)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="go")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    result = await adapter.invoke(task=task, session=session)

    runaway = [d for d in steerer.drifts if d.kind is DriftKind.RUNAWAY_DELEGATION]
    assert len(runaway) == 1, (
        f"expected exactly one RUNAWAY_DELEGATION drift (fires once at "
        f"threshold crossing); got {len(runaway)}"
    )
    assert runaway[0].severity is DriftSeverity.CRITICAL
    assert runaway[0].current_task_id == "t1"
    # Adapter's stop_reason reflects the trip.
    assert result.stop_reason == "runaway_delegation"


async def test_cap_counter_resets_between_invocations() -> None:
    """Two back-to-back invocations each see their own count — a
    trip on invoke #1 does not poison invoke #2.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    cap = 2
    sub = Agent(name="sub", model=_make_quiet_llm()(), instruction="")
    coord = Agent(
        name="coord",
        model=_make_tool_looper_llm("sub", calls=10)(),
        instruction="",
        tools=[AgentTool(sub)],
    )
    adapter = ADKAdapter(coord, agent_tool_cap=cap)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="go")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)
    # Trip count after invoke #1.
    after_first = len([d for d in steerer.drifts if d.kind.value == "runaway_delegation"])
    assert after_first == 1

    # The plugin's counter resets on clear_active_context (called from
    # invoke's finally block). Invoke #2 should trip again — NOT skip
    # because it inherited a stale counter.
    await adapter.invoke(task=Task(id="t2", title="go"), session=session)
    after_second = len([d for d in steerer.drifts if d.kind.value == "runaway_delegation"])
    assert after_second == 2, (
        "second invocation did not get a fresh counter — the "
        "runaway trip from invoke #1 leaked into invoke #2. "
        "clear_active_context must reset the counter."
    )


async def test_cap_disabled_when_zero() -> None:
    """``agent_tool_cap=0`` disables the guard — coordinator can delegate
    arbitrarily without tripping.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.types import Plan, Session, Task

    sub = Agent(name="sub", model=_make_quiet_llm()(), instruction="")
    # 20 spawns — would easily trip the default 16 cap.
    coord = Agent(
        name="coord",
        model=_make_tool_looper_llm("sub", calls=20)(),
        instruction="",
        tools=[AgentTool(sub)],
    )
    adapter = ADKAdapter(coord, agent_tool_cap=0)
    steerer = _RecordingSteerer()
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="go")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    result = await adapter.invoke(task=task, session=session)

    runaway = [d for d in steerer.drifts if d.kind.value == "runaway_delegation"]
    assert runaway == [], f"cap=0 must disable the guard; got {len(runaway)} drifts"
    assert result.stop_reason != "runaway_delegation"


def test_default_cap_exposed_as_module_constant() -> None:
    """``goldfive.adapters.adk.DEFAULT_AGENT_TOOL_CAP`` is the documented
    default (16) so tests and docs can reference it by name.
    """
    from goldfive.adapters.adk import DEFAULT_AGENT_TOOL_CAP

    assert DEFAULT_AGENT_TOOL_CAP == 16


def test_adapter_records_configured_cap() -> None:
    """``ADKAdapter(..., agent_tool_cap=N)`` is stored on the adapter
    so callers can introspect / tests can assert without monkey-
    patching the plugin's private attr.
    """
    from google.adk.agents import Agent

    from goldfive.adapters.adk import ADKAdapter

    agent = Agent(name="solo", model=_make_quiet_llm()(), instruction="")
    adapter = ADKAdapter(agent, agent_tool_cap=7)
    assert adapter._agent_tool_cap == 7
    # The plugin's internal cap tracks the adapter's.
    assert adapter._plugin._agent_tool_cap == 7
