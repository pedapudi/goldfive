"""State-protocol reliability tests for the registry-dispatch model.

CRITICAL: goldfive's state-protocol writes cannot be best-effort. They
MUST propagate into AgentTool-spawned sub-Runners so the sub-agent can
read the active task, plan context, run id, and tools_available off its
own live session state.

Phase 1 moved the authoritative write into the plugin's
``before_run_callback`` (against the LIVE invocation session) precisely
so the write lands on the session the sub-Runner actually runs against.
These tests pin that contract.

Skipped entirely when ``google.adk`` is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("google.adk")

from goldfive.types import Plan, Session, Task

# ---------------------------------------------------------------------------
# Scripted LLMs used to trigger AgentTool dispatch
# ---------------------------------------------------------------------------


def _make_agent_a(tool_name: str) -> Any:
    """Build agent A whose first LLM turn calls ``tool_name`` (an AgentTool).

    Turn 1: emit a ``function_call`` for ``tool_name``.
    Turn 2: return ``turn_complete=True`` after the sub-agent responds.
    """
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _ScriptedA(BaseLlm):
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
                                    id="call_b",
                                    name=tool_name,
                                    args={"request": "do it"},
                                )
                            ),
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="ok")],
                    ),
                    turn_complete=True,
                )

    return _ScriptedA


def _make_agent_b_llm() -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _ScriptedB(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="done")],
                ),
                turn_complete=True,
            )

    return _ScriptedB


class _StubSteerer:
    async def observe(self, *a: Any, **kw: Any) -> None:
        pass

    async def transition(self, *a: Any, **kw: Any) -> None:
        pass

    def detect_drift(self, *a: Any, **kw: Any) -> None:
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _read_pin_via_before_model(observed: dict[str, Any]):
    """Return a ``before_model_callback`` that snapshots the pin into ``observed``.

    Phase 2.1 of goldfive#271 — readers consult goldfive
    ``Session.state`` via the plugin reference, not ADK
    ``session.state``. Walks ``invocation_context.plugin_manager`` for
    the goldfive plugin and reads its ``_active_ctx.session`` — same
    shape as :func:`session_context_from_invocation`.
    """
    from goldfive.adapters._adk_plugin import session_context_from_invocation
    from goldfive.state_store import StateStore

    def _before_model(callback_context: Any, llm_request: Any) -> None:  # noqa: ARG001
        inv_ctx = getattr(callback_context, "_invocation_context", None) or getattr(
            callback_context, "invocation_context", None
        )
        ctx = session_context_from_invocation(inv_ctx)
        session = getattr(ctx, "session", None) if ctx is not None else None
        if observed:
            return None
        store = StateStore.for_session(session)
        observed["pin_current_task"] = store.pin_current_task()
        return None

    return _before_model


# ---------------------------------------------------------------------------
# State propagation through an AgentTool boundary
# ---------------------------------------------------------------------------


async def test_state_protocol_current_task_id_visible_in_sub_runner() -> None:
    """B's ``before_model_callback`` resolves the pin via the plugin
    reference and sees ``my-task-id`` on the goldfive Session — the
    pin must be reachable across an AgentTool boundary.

    Phase 2.1 of goldfive#271 — the pin no longer lives on ADK
    ``session.state``. The plugin instance is shared with the
    sub-Runner so :func:`session_context_from_invocation` resolves
    the same goldfive Session no matter which invocation fires the
    callback.
    """
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive.adapters.adk import ADKAdapter

    observed: dict[str, Any] = {}
    agent_b = Agent(
        name="agent_b",
        model=_make_agent_b_llm()(),
        instruction="",
        before_model_callback=_read_pin_via_before_model(observed),
    )
    agent_a = Agent(
        name="agent_a",
        model=_make_agent_a("agent_b")(),
        instruction="",
        tools=[AgentTool(agent_b)],
    )

    adapter = ADKAdapter(agent_a)
    adapter.bind_steerer(_StubSteerer())

    task = Task(id="my-task-id", title="compound", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)
    await adapter.invoke(task=task, session=session)

    assert observed, "B's before_model_callback never ran — AgentTool dispatch broke"
    assert observed.get("pin_current_task") == "my-task-id", (
        f"B's sub-Runner saw pin={observed.get('pin_current_task')!r}; "
        "expected 'my-task-id'. The pin write in _stamp_current_task_id "
        "did not propagate through AgentTool."
    )


# ---------------------------------------------------------------------------
# Phase 2.0 of goldfive#271 — V1 / V2 / V5 migrated. Tests that verified
# those bridge writes (run_id / plan_id / plan_summary / tools_available)
# crossing onto ADK ``session.state`` are removed: those writes are gone.
# Production reads of those values now go through the goldfive Session
# directly. The dynamic-instruction-resolver / planner integration tests
# in :mod:`tests.test_dynamic_instruction` and the e2e steer test below
# cover the new read path.
# ---------------------------------------------------------------------------


async def test_before_run_callback_no_op_when_no_active_ctx() -> None:
    """Without an active ``SessionContext``, ``before_run_callback`` is a no-op.

    Regression: eager writes when ``_active_ctx is None`` would crash
    tests that instantiate the plugin outside the adapter. The callback
    returns silently and writes nothing.
    """
    from goldfive.adapters._adk_plugin import make_adk_plugin

    plugin = make_adk_plugin(host_agent_name="agent_a")

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self) -> None:
            self.session = _Session()
            self.invocation_id = "inv-1"
            self.agent = None

    inv_ctx = _InvCtx()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    # No keys written.
    assert inv_ctx.session.state == {}


async def test_top_level_invocation_id_pinned_then_released() -> None:
    """Top-level ``before_run`` pins ``_top_invocation_id``; matching
    ``after_run`` releases it. Ensures nested sub-Runners can correctly
    attribute themselves via ``parent_invocation_id``.
    """
    from goldfive.adapters._adk_plugin import SessionContext, make_adk_plugin

    plugin = make_adk_plugin(host_agent_name="agent_a")
    session = Session(run_id="run-1")
    task = Task(id="t1", title="x")
    ctx = SessionContext(
        session=session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="agent_a",
    )
    plugin.set_active_context(ctx)

    class _Session:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    class _InvCtx:
        def __init__(self, inv_id: str) -> None:
            self.session = _Session()
            self.invocation_id = inv_id
            self.agent = type("A", (), {"name": "agent_a"})()

    top = _InvCtx("inv-top")
    await plugin.before_run_callback(invocation_context=top)
    assert plugin._top_invocation_id == "inv-top"

    # Nested sub-Runner fires before_run too — must NOT overwrite the pin.
    nested = _InvCtx("inv-nested")
    await plugin.before_run_callback(invocation_context=nested)
    assert plugin._top_invocation_id == "inv-top", (
        "nested AgentTool sub-Runner overwrote the top-level invocation pin"
    )

    # Nested after_run does NOT release (its id != top).
    await plugin.after_run_callback(invocation_context=nested)
    assert plugin._top_invocation_id == "inv-top"

    # Top-level after_run RELEASES.
    await plugin.after_run_callback(invocation_context=top)
    assert plugin._top_invocation_id == ""




# ---------------------------------------------------------------------------
# Phase 2.0 of goldfive#271 — orchestration-state bridge eliminated.
#
# DefaultSteerer (#152) writes active_steer / goals_summary /
# cancelled_function_call_ids onto goldfive ``Session.state`` —
# framework-agnostic orchestration dict.
# :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner` and
# :mod:`~goldfive.adapters.adk_llm_instrumentation` now read those values from
# goldfive ``Session.state`` directly via the
# :class:`~goldfive.state_store.StateStore` typed
# accessor — there is no copy onto ADK ``session.state``. This is the
# fix for goldfive#275 (the stale-session race that broke ADK-web).
# ---------------------------------------------------------------------------


async def test_user_steer_to_planner_instruction_no_bridge() -> None:
    """USER_STEER → DefaultSteerer → goldfive.Session.state →
    GoldfivePlanner.build_planning_instruction (via SessionContext +
    StateStore — no ADK-state copy).

    This is the high-level e2e replacement for the pre-Phase-2.0
    bridge test. The data path used to be:

        steerer → goldfive.Session.state
                → before_run_callback bridge
                → ADK session.state
                → planner reads ADK state

    After Phase 2.0 the data path is:

        steerer → goldfive.Session.state
                → planner reads goldfive Session via SessionContext
                  + StateStore

    Asserts: no ValueError, no race, the steer body still lands in
    the planner's injected instruction.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.control import ControlKind, ControlMessage
    from goldfive.steerer import DefaultSteerer

    captured_instructions: list[str] = []

    class _ScriptedLLM(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            config = getattr(llm_request, "config", None)
            si = getattr(config, "system_instruction", "") if config is not None else ""
            captured_instructions.append(si if isinstance(si, str) else str(si))
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    agent = Agent(
        name="agent_a",
        model=_ScriptedLLM(),
        instruction="",
    )
    adapter = ADKAdapter(agent)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)  # type: ignore[arg-type]
    steerer.bind_adapter(adapter)
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="ship it", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)

    # USER_STEER via the steerer — same path a control-channel STEER
    # takes in production. Writes active_steer onto goldfive.Session.state.
    await steerer.observe(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "pivot toward reliability"},
        ),
        session,
    )

    await adapter.invoke(task=task, session=session)

    assert captured_instructions, (
        "LLM generate_content_async never ran — invocation did not dispatch"
    )
    first = captured_instructions[0]
    assert "[GOLDFIVE ORCHESTRATION CONTEXT]" in first, (
        "GoldfivePlanner's orchestration block missing from the system instruction"
    )
    assert "pivot toward reliability" in first, (
        f"Steer body missing from injected instruction; saw:\n{first}"
    )


# ---------------------------------------------------------------------------
# Regression test for goldfive#275 — stale-session ValueError from the
# V2 bridge writing ADK ``session.state`` from inside a callback frame.
# Phase 2.0's resolver/planner reads from goldfive ``Session.state``
# directly, so the race no longer exists.
# ---------------------------------------------------------------------------


async def test_user_steer_between_invocations_no_stale_session_valueerror() -> None:
    """Reproduce the goldfive#275 race scenario.

    Before Phase 2.0: a USER_STEER arriving between invocations
    triggered an in-callback write to ADK ``session.state`` (V2's
    bridge in ``before_run_callback``) which raced ADK's optimistic-
    concurrency check, surfacing as ``ValueError: stale session`` and
    tearing down the steerer (zero observability events emitted on
    the failed turn).

    After Phase 2.0: the bridge is gone. The steerer writes goldfive
    ``Session.state``; the planner / resolver read it directly via
    the SessionContext stash exposed through the plugin manager.
    Nothing inside a callback writes to ADK ``session.state``, so
    the race cannot fire.

    Asserts:

    * Two consecutive invokes against the same session run without
      raising :class:`ValueError`.
    * The DriftDetected(USER_STEER) sink event fires when the steer
      arrives.
    * The PlanRevised sink event fires when the steerer's refine
      lands.
    * The planner's per-turn injection on the SECOND invoke contains
      the steer body — the data path goldfive#275 was breaking.
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive import InMemorySink
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.control import ControlKind, ControlMessage
    from goldfive.steerer import DefaultSteerer
    from goldfive.types import DriftEvent, Goal, ObservedAction

    captured_instructions: list[str] = []

    class _ScriptedLLM(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            config = getattr(llm_request, "config", None)
            si = getattr(config, "system_instruction", "") if config is not None else ""
            captured_instructions.append(si if isinstance(si, str) else str(si))
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    # Refining stub planner so the steerer's USER_STEER promotion path
    # produces a revised plan and emits PlanRevised. Keeping it simple
    # — return a one-task plan whose title reflects the steer body
    # so an assertion can confirm the refine landed.
    class _RefiningPlanner:
        async def generate(
            self,
            *,
            goals: list[Goal],
            available_agents: list[str] | list[dict] | None = None,
            context: Mapping[str, Any] | None = None,
        ) -> Plan | None:
            return Plan(
                id="p1",
                run_id="r1",
                goal_ids=[g.id for g in goals if g.id],
                tasks=[Task(id="t1", title="ship it", assignee_agent_id="agent_a")],
                edges=[],
            )

        async def refine(
            self,
            *,
            plan: Plan,
            drift: DriftEvent,
            goals: list[Goal],
            observed_actions: list[ObservedAction] | None = None,
            available_agents: list[str] | list[dict] | None = None,
        ) -> Plan | None:
            return Plan(
                id="p2",
                run_id="r1",
                goal_ids=plan.goal_ids,
                tasks=[
                    Task(
                        id="t1_steered",
                        title=f"steered: {drift.detail}",
                        assignee_agent_id="agent_a",
                    )
                ],
                edges=[],
                summary="post-steer plan",
                revision_index=plan.revision_index + 1,
                revision_kind=str(getattr(drift, "kind", "")),
                revision_severity=str(getattr(drift, "severity", "")),
            )

    agent = Agent(name="agent_a", model=_ScriptedLLM(), instruction="")
    adapter = ADKAdapter(agent)
    sink = InMemorySink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_RefiningPlanner())  # type: ignore[arg-type]
    steerer.bind_adapter(adapter)
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="ship it", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)

    # First turn: no steer, baseline.
    await adapter.invoke(task=task, session=session)

    # User-steer arrives between turns — same path the ADK-web demo
    # exercises. The control-channel STEER drives DefaultSteerer's
    # _apply_user_steer_state which writes the active_steer keys onto
    # goldfive Session.state.
    await steerer.observe(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "pivot to reliability"},
        ),
        session,
    )

    # Second turn: with the active steer in place, the planner's
    # injected instruction should contain the steer body. Before
    # Phase 2.0 this was the path that raised the stale-session
    # ValueError because before_run_callback wrote the active_steer
    # keys onto ADK session.state from inside the callback frame.
    await adapter.invoke(task=task, session=session)

    # No ValueError raised — the race that #275 reproduced is gone.
    # Both invokes returned cleanly.
    assert len(captured_instructions) >= 2, (
        f"expected >=2 LLM calls (one per invoke); got {len(captured_instructions)}"
    )

    # Steer body landed in the SECOND invocation's injected
    # instruction — the data path that #275 broke.
    second = captured_instructions[1]
    assert "pivot to reliability" in second, (
        f"Steer body missing from second invocation's instruction; saw:\n{second}"
    )

    # Sink received the DriftDetected(USER_STEER) and PlanRevised
    # events that #275 was suppressing when the race tore down the
    # steerer. Proto encodes ``DriftKind.USER_STEER`` as int 5
    # (see ``proto/goldfive/v1/types.proto``). Sink events are a
    # mix of proto Events (with ``payload`` oneof) and plain dicts
    # for legacy ControlMessage / drift records — both shapes are
    # accepted.
    user_steer_seen = False
    plan_revised_count = 0
    for evt in sink.events:
        if hasattr(evt, "WhichOneof"):
            payload_name = evt.WhichOneof("payload")
            if payload_name == "drift_detected" and evt.drift_detected.kind == 5:
                user_steer_seen = True
            if payload_name == "plan_revised":
                plan_revised_count += 1
    assert user_steer_seen, (
        "DriftDetected(USER_STEER, kind=5) missing from sink events — "
        "the steerer was torn down before the event could fire."
    )
    payload_kinds = [
        evt.WhichOneof("payload") if hasattr(evt, "WhichOneof") else type(evt).__name__
        for evt in sink.events
    ]
    assert plan_revised_count >= 1, (
        f"PlanRevised missing from sink events; payloads={payload_kinds!r}"
    )


async def test_phase_0_tripwire_stays_green_through_user_steer_flow() -> None:
    """Phase 0's runtime tripwire must NOT raise during the
    user-steer flow that previously violated the contract.

    With the bridge gone, the only ADK-state writes from inside a
    callback frame come from the still-catalogued V3 (per-agent pin)
    and V4 (delegation pin) sites. A USER_STEER → invoke flow
    exercises both. The tripwire is on (autouse fixture in
    :mod:`tests.conftest`); a regression that re-introduces a write
    from before_run_callback / before_model_callback would surface
    here as ``StateOwnershipViolation`` (BaseException — not caught
    by the defensive try/except blocks).
    """
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    from goldfive.adapters.adk import ADKAdapter
    from goldfive.control import ControlKind, ControlMessage
    from goldfive.steerer import DefaultSteerer

    class _ScriptedLLM(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="ok")],
                ),
                turn_complete=True,
            )

    agent = Agent(name="agent_a", model=_ScriptedLLM(), instruction="")
    adapter = ADKAdapter(agent)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[], planner=None)  # type: ignore[arg-type]
    steerer.bind_adapter(adapter)
    adapter.bind_steerer(steerer)

    task = Task(id="t1", title="ship it", assignee_agent_id="agent_a")
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[task], edges=[])
    session = Session(run_id="r1", plan=plan)

    # No ValueError, no StateOwnershipViolation:
    await adapter.invoke(task=task, session=session)
    await steerer.observe(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "stay reliable"},
        ),
        session,
    )
    await adapter.invoke(task=task, session=session)
