"""§5.7 layered e2e: ledger-mode runaway → clean PAUSE (task #12).

AGENCY-PRESERVATION.md §5.7 (layered e2e, functional pass) + the
"narrow regression criteria pass on broken runs" scar tissue. The PR-12
Q2 routing (RUNAWAY_DELEGATION → PAUSE_ESCALATE in ledger mode) is
unit-tested at the control-message disposition level
(``tests/test_ledger_refine_retirement.py::test_ledger_runaway_delegation_pauses_not_note``),
but "clean stop" is an INTEGRATION property: it depends on the executor's
overlay pause-block consuming the GOLDFIVE_PAUSE_ESCALATE control AND the
ledger plan structure (OUTCOME root stays PENDING, the nudge-replay
liveness gate ignores it).

This drives a REAL coordinator+AgentTool tree through the full
``wrap()`` → ``Runner.run()`` path in ledger mode
(``plan_mode=ledger``, ``observation_only=False``), trips a runaway
delegation (the coordinator delegates past ``agent_tool_cap``), and
asserts the run reaches a CLEAN PAUSE — terminates within a wall-clock
bound (never a hang), surfaces the durable HUMAN_INTERVENTION_REQUIRED
drift, and the outcome carries the pause reason (never a SILENT success
that masks a broken stop).

Belongs on the pre-PR-13b §5.7 checklist; not a PR-12 merge blocker.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

pytest.importorskip("google.adk")


def _make_tool_looper_llm(tool_name: str, calls: int) -> Any:
    """Coordinator LLM that calls an AgentTool ``calls`` times then finishes."""
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
                        role="model", parts=[genai_types.Part(text="done")]
                    ),
                    turn_complete=True,
                )

    return _Looper


def _make_quiet_llm() -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class _Quiet(BaseLlm):
        model: str = "fake-model"

        async def generate_content_async(self, llm_request: Any, stream: bool = False):  # noqa: ARG002
            yield LlmResponse(
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text="ok")]
                ),
                turn_complete=True,
            )

    return _Quiet


async def test_ledger_runaway_reaches_clean_pause() -> None:
    from google.adk.agents import Agent
    from google.adk.tools.agent_tool import AgentTool

    from goldfive import wrap
    from goldfive.adapters.adk import ADKAdapter
    from goldfive.config import RuntimeConfig, SteeringConfig
    from goldfive.control import ControlChannel
    from goldfive.conv import _drift_kind_to_pb
    from goldfive.planner import StaticPlanner
    from goldfive.sinks import InMemorySink
    from goldfive.types import DriftKind, Plan, Task, TaskKind, TaskStatus

    cap = 3
    sub = Agent(name="sub", model=_make_quiet_llm()(), instruction="")
    # The coordinator tries to delegate 10× — well past the cap — the
    # user-supplied-coordinator runaway shape (goldfive#130).
    coord = Agent(
        name="coord",
        model=_make_tool_looper_llm("sub", calls=10)(),
        instruction="",
        tools=[AgentTool(sub)],
    )
    adapter = ADKAdapter(coord, agent_tool_cap=cap)

    # A ledger plan: one goal-anchored OUTCOME deliverable. It stays
    # PENDING for the whole run (the agent owns the means), so the run can
    # only terminate via goldfive's own disposition — here, the pause.
    ledger_plan = Plan(
        id="p-runaway",
        run_id="",
        goal_ids=[],
        tasks=[Task(id="o1", title="Deliver the thing", kind=TaskKind.OUTCOME)],
        edges=[],
    )

    sink = InMemorySink()
    channel = ControlChannel()
    runner = wrap(
        adapter,
        planner=StaticPlanner(ledger_plan),
        control=channel,
        sinks=[sink],
        runtime=RuntimeConfig(
            steering=SteeringConfig(plan_mode="ledger", observation_only=False)
        ),
    )

    # (1) NEVER A HANG: the run must terminate within a generous wall-clock
    #     bound. A wedged pause/cancel interaction would blow this.
    outcome = await asyncio.wait_for(runner.run(user_input="go"), timeout=30.0)
    assert outcome is not None

    drift_kinds = [
        e.drift_detected.kind
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    # The runaway actually tripped (precondition: the e2e exercised the path).
    assert _drift_kind_to_pb(DriftKind.RUNAWAY_DELEGATION) in drift_kinds, (
        "runaway delegation did not trip — the e2e did not exercise the path"
    )
    # (2) CLEAN PAUSE, not a manufactured failure or a silent success: the
    #     durable HUMAN_INTERVENTION_REQUIRED drift is on the sink stream...
    assert _drift_kind_to_pb(DriftKind.HUMAN_INTERVENTION_REQUIRED) in drift_kinds, (
        "ledger runaway must escalate to a human-intervention pause"
    )
    # ...and the run outcome carries the pause reason (so a plain
    # completion can never masquerade as the stop) — AND that reason names
    # the RUNAWAY as the cause, binding the PAUSE to this drift specifically
    # (a forecast-mode run pauses too, but via a different path, so the
    # cause string is what pins the PR-12 ledger runaway rung).
    assert "goldfive_pause_escalate" in (outcome.reason or ""), (
        f"expected a pause-reason outcome; got reason={outcome.reason!r}"
    )
    assert "runaway_delegation" in (outcome.reason or ""), (
        f"the pause must be caused by the runaway; got reason={outcome.reason!r}"
    )

    # (3) The run paused WITHOUT delivering the outcome: the OUTCOME
    #     deliverable o1 stays non-terminal (PENDING). This discriminates a
    #     "silent success that completed the deliverable" — a broken stop
    #     that produced output anyway. (We deliberately do NOT assert
    #     outcome.success: a clean ledger pause is success=True with the
    #     deliverable still PENDING — the "don't grade ledger runs on
    #     run.success alone" caution. The control channel is consumed by
    #     the executor's pause-block to end the turn, so it is not a
    #     post-run observable — outcome.reason + the HIR drift + the
    #     undelivered OUTCOME are the canonical pause observables.)
    o1 = next(t for t in outcome.session.plan.tasks if t.id == "o1")
    assert o1.status is TaskStatus.PENDING, (
        f"the OUTCOME deliverable should be undelivered (PENDING) after the "
        f"pause; got {o1.status}"
    )
