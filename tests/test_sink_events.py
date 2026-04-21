"""Sink-event emission tests for the registry-dispatch model.

Phase 1 added three sink events to observe the dispatch tree:

* ``AgentInvocationStarted`` on every runner entry (top-level + sub-Runner)
* ``AgentInvocationCompleted`` on every runner exit
* ``DelegationObserved`` when the plugin sees an AgentTool call

These tests pin the nested shape
``started(A) → delegation(A→B) → started(B) → completed(B) → completed(A)``
and assert key correlation fields (task_id inherited through sub-Runners,
parent_invocation_id populated on nested started events).

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.types import Plan, Session, Task


class _SinkingSteerer:
    def __init__(self, sink: Any) -> None:
        self._sinks = [sink]

    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _make_b_llm() -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _B(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="b done")],
                ),
                turn_complete=True,
            )

    return _B


def _make_a_llm(tool_name: str) -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _A(BaseLlm):
        model: str = "fake-model"
        _step: int = 0

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            self._step += 1
            if self._step == 1:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    id="c1", name=tool_name, args={"request": "go"}
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="a done")],
                    ),
                    turn_complete=True,
                )

    return _A


def _make_terminal_llm() -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _T(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="hi")],
                ),
                turn_complete=True,
            )

    return _T


def _kinds(events: list[Any]) -> list[str]:
    return [e.WhichOneof("payload") for e in events]


# ---------------------------------------------------------------------------
# Top-level dispatch: started + completed for the assignee, nothing else.
# ---------------------------------------------------------------------------


async def test_single_agent_dispatch_emits_started_and_completed_pair() -> None:
    """Leaf-agent invoke → exactly one started + one completed for that agent.

    A 3-agent tree with only the assignee dispatched: NO other agent's
    started/completed shows up because goldfive does not speculatively
    drive the others' runners.
    """
    from google.adk.agents import Agent

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    # Build a tree of three agents: a (assignee), b, c.
    a = Agent(name="a", model=_make_terminal_llm()(), instruction="")
    b = Agent(name="b", model=_make_terminal_llm()(), instruction="")
    c = Agent(name="c", model=_make_terminal_llm()(), instruction="")
    root = Agent(
        name="root",
        model=_make_terminal_llm()(),
        instruction="",
        sub_agents=[a, b, c],
    )

    adapter = ADKAdapter(root)
    sink = InMemorySink()
    adapter.bind_steerer(_SinkingSteerer(sink))

    task = Task(id="T", title="go", assignee_agent_id="a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    starts = [e for e in sink.events if e.WhichOneof("payload") == "agent_invocation_started"]
    completes = [
        e for e in sink.events if e.WhichOneof("payload") == "agent_invocation_completed"
    ]

    assert len(starts) == 1, f"expected 1 start; got {len(starts)}: {_kinds(sink.events)}"
    assert len(completes) == 1
    assert starts[0].agent_invocation_started.agent_name == "a"
    assert starts[0].agent_invocation_started.task_id == "T"
    # parent_invocation_id is empty on top-level dispatches.
    assert starts[0].agent_invocation_started.parent_invocation_id == ""
    assert completes[0].agent_invocation_completed.agent_name == "a"
    assert completes[0].agent_invocation_completed.task_id == "T"


# ---------------------------------------------------------------------------
# AgentTool delegation: nested shape + correlation fields
# ---------------------------------------------------------------------------


async def test_agent_tool_delegation_produces_nested_started_completed_shape() -> None:
    """When A calls ``AgentTool(B)``, the sink stream must include
    A.started → delegation(A→B) → B.started → B.completed → A.completed.

    The shape is observable: sinks can reconstruct the delegation tree
    from these five events alone. A regression that swallowed any one
    would break downstream timeline UIs.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    agent_b = Agent(name="agent_b", model=_make_b_llm()(), instruction="")
    agent_a = Agent(
        name="agent_a",
        model=_make_a_llm("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    sink = InMemorySink()
    adapter.bind_steerer(_SinkingSteerer(sink))

    task = Task(id="T", title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    # Filter the event stream down to the three kinds we care about and
    # check the ordering reads "A.started → delegation(A→B) → B.started
    # → B.completed → A.completed".
    ordered: list[tuple[str, str]] = []
    for e in sink.events:
        kind = e.WhichOneof("payload")
        if kind == "agent_invocation_started":
            ordered.append((kind, e.agent_invocation_started.agent_name))
        elif kind == "agent_invocation_completed":
            ordered.append((kind, e.agent_invocation_completed.agent_name))
        elif kind == "delegation_observed":
            ordered.append(
                (
                    kind,
                    f"{e.delegation_observed.from_agent}->{e.delegation_observed.to_agent}",
                )
            )

    # Find the five anchor events for agent_a / agent_b / delegation.
    # Wrap the StopIteration as a clean AssertionError so the async
    # harness doesn't surface it as "coroutine raised StopIteration".
    def _find(predicate) -> int:
        for i, entry in enumerate(ordered):
            if predicate(entry):
                return i
        raise AssertionError(f"event not found in stream; ordering={ordered}")

    a_start_idx = _find(lambda e: e == ("agent_invocation_started", "agent_a"))
    deleg_idx = _find(lambda e: e == ("delegation_observed", "agent_a->agent_b"))
    b_start_idx = _find(lambda e: e == ("agent_invocation_started", "agent_b"))
    b_end_idx = _find(lambda e: e == ("agent_invocation_completed", "agent_b"))
    a_end_idx = _find(lambda e: e == ("agent_invocation_completed", "agent_a"))

    # Nested ordering: A started BEFORE B started; B completed BEFORE A completed.
    assert a_start_idx < deleg_idx < b_start_idx < b_end_idx < a_end_idx, (
        f"expected nested event shape but got ordering={ordered}"
    )


async def test_sub_runner_events_inherit_outer_task_id() -> None:
    """B's started/completed events carry A's task_id — not an empty id.

    The plugin's ``_active_ctx`` is a shared instance across all runners,
    so the sub-Runner reads the outer task's id off the same ctx.
    Regression would drop the task_id on B's events and break
    per-task timeline rendering.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    agent_b = Agent(name="agent_b", model=_make_b_llm()(), instruction="")
    agent_a = Agent(
        name="agent_a",
        model=_make_a_llm("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    sink = InMemorySink()
    adapter.bind_steerer(_SinkingSteerer(sink))

    outer_task_id = "outer-task-xyz"
    task = Task(id=outer_task_id, title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    b_events = [
        e
        for e in sink.events
        if (
            e.WhichOneof("payload") == "agent_invocation_started"
            and e.agent_invocation_started.agent_name == "agent_b"
        )
        or (
            e.WhichOneof("payload") == "agent_invocation_completed"
            and e.agent_invocation_completed.agent_name == "agent_b"
        )
    ]
    assert b_events, "no events for agent_b on the sub-Runner"
    for e in b_events:
        k = e.WhichOneof("payload")
        if k == "agent_invocation_started":
            assert e.agent_invocation_started.task_id == outer_task_id
        else:
            assert e.agent_invocation_completed.task_id == outer_task_id


async def test_sub_runner_started_event_parent_invocation_id_equals_outer() -> None:
    """B's ``started.parent_invocation_id`` equals A's ``started.invocation_id``.

    This is the wire that sinks use to reconstruct the delegation tree
    into a single root span. A regression that failed to pin the
    ``_top_invocation_id`` would leave every nested started event with
    empty ``parent_invocation_id``.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    agent_b = Agent(name="agent_b", model=_make_b_llm()(), instruction="")
    agent_a = Agent(
        name="agent_a",
        model=_make_a_llm("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    sink = InMemorySink()
    adapter.bind_steerer(_SinkingSteerer(sink))

    task = Task(id="T", title="x", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    a_starts = [
        e
        for e in sink.events
        if e.WhichOneof("payload") == "agent_invocation_started"
        and e.agent_invocation_started.agent_name == "agent_a"
    ]
    b_starts = [
        e
        for e in sink.events
        if e.WhichOneof("payload") == "agent_invocation_started"
        and e.agent_invocation_started.agent_name == "agent_b"
    ]
    assert a_starts, "no agent_a started event"
    assert b_starts, "no agent_b started event"
    a_start = a_starts[0]
    b_start = b_starts[0]

    assert a_start.agent_invocation_started.invocation_id, (
        "agent_a's top-level started event must carry an invocation_id"
    )
    # Sub-Runner sees its own distinct invocation id, but its
    # parent_invocation_id field points back at agent_a's top-level one.
    assert (
        b_start.agent_invocation_started.parent_invocation_id
        == a_start.agent_invocation_started.invocation_id
    ), (
        "B's parent_invocation_id must equal A's invocation_id so sinks "
        "can reconstruct the delegation span tree; regressing this would "
        "leave B as an orphan span."
    )
    # B's invocation_id is distinct from A's (it's a separate Runner call).
    assert (
        b_start.agent_invocation_started.invocation_id
        != a_start.agent_invocation_started.invocation_id
    )


async def test_delegation_observed_carries_outer_task_id() -> None:
    """The ``DelegationObserved`` event's task_id is the OUTER task's id.

    Sinks use ``task_id`` to attribute a delegation to the right task;
    without it, delegations show up as un-attributed spans in the UI.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.sinks.memory import InMemorySink

    agent_b = Agent(name="agent_b", model=_make_b_llm()(), instruction="")
    agent_a = Agent(
        name="agent_a",
        model=_make_a_llm("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    sink = InMemorySink()
    adapter.bind_steerer(_SinkingSteerer(sink))

    task = Task(id="outer-42", title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    delegations = [e for e in sink.events if e.WhichOneof("payload") == "delegation_observed"]
    assert delegations, "expected at least one delegation_observed event"
    d = delegations[0].delegation_observed
    assert d.from_agent == "agent_a"
    assert d.to_agent == "agent_b"
    assert d.task_id == "outer-42"


async def test_no_sink_events_when_steerer_is_unbound() -> None:
    """Without a bound steerer (or one without ``_sinks``), emission is a no-op.

    Guards against a regression where the plugin emits to a ``None``
    sink list and raises, killing a live run.
    """
    from google.adk.agents import Agent

    from goldfive.adapters.adk import ADKAdapter

    agent = Agent(name="a", model=_make_terminal_llm()(), instruction="")
    adapter = ADKAdapter(agent)
    # No bind_steerer — the adapter's _steerer is None.

    task = Task(id="t1", title="x", assignee_agent_id="a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    # Must not raise.
    result = await adapter.invoke(task=task, session=session)
    assert result.task_id == "t1"
