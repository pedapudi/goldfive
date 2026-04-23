"""Tests for :func:`goldfive.wrap` wiring ``call_llm`` into the default steerer.

Regression suite for goldfive#217: ``goldfive.wrap`` was constructing a
bare ``DefaultSteerer()`` whose ``goal_drift_call_llm`` was ``None``,
which silently disarmed the trajectory-level GOAL_DRIFT judge
(goldfive#143). The fix:

* ``wrap()`` forwards the resolved ``call_llm`` / ``model`` into
  ``DefaultSteerer(goal_drift_call_llm=..., goal_drift_model=...)``
  when the caller did not supply an explicit ``steerer=``.
* ``Runner`` emits a one-shot WARNING when ``goal_drift_enabled=True``
  but the steerer has no callable wired — the docstring at
  ``DefaultSteerer.__init__`` had promised this wiring; we now honour
  that promise and surface the gap when it is absent.
* The end-to-end judge-fires path is exercised by routing enough
  ``note_agent_turn`` calls through the steerer to cross the interval
  with a stub ``call_llm`` that returns ``{"progressing": false}``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

import goldfive  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
from goldfive.runner import Runner  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    async def generate(self, *, goals, available_agents, context=None):
        return None

    async def refine(self, *, plan, drift, goals):
        return None


def _stub_call_llm(responses: list[Any]):
    """Async ``CallLLM``-shaped stub that pops responses in order."""
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


async def _noop_agent(task: Any, session: Any, tools: Any) -> InvocationResult:
    return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


def _make_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Research", description="Research topic X")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id="t1",
    )


# ---------------------------------------------------------------------------
# wrap() wiring
# ---------------------------------------------------------------------------


def test_wrap_wires_call_llm_into_default_steerer() -> None:
    """goldfive.wrap(agent, call_llm=...) → DefaultSteerer carries the callable."""
    call_llm = _stub_call_llm([{"progressing": True}])
    runner = goldfive.wrap(
        _noop_agent,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._goal_drift_call_llm is call_llm
    assert steerer._goal_drift_model == "fake-model"


def test_wrap_preserves_explicit_steerer() -> None:
    """wrap(steerer=...) never overrides the caller's steerer."""
    call_llm = _stub_call_llm([{"progressing": True}])
    # Explicit steerer with no judge callable -- the caller is opting out.
    explicit = DefaultSteerer(goal_drift_call_llm=None)
    runner = goldfive.wrap(
        _noop_agent,
        call_llm=call_llm,
        steerer=explicit,
        sinks=[],
    )
    assert runner.steerer is explicit
    # Because goal_drift_enabled defaults to True and the explicit
    # steerer has no callable, the Runner logs a warning but does NOT
    # retroactively attach one (opt-out stays opt-out).
    assert runner.steerer._goal_drift_call_llm is None


def test_wrap_without_call_llm_leaves_steerer_unarmed() -> None:
    """Degraded path: no call_llm, no LLM detectable -> DefaultSteerer() stays bare."""
    runner = goldfive.wrap(_noop_agent, sinks=[])
    assert isinstance(runner.steerer, DefaultSteerer)
    assert runner.steerer._goal_drift_call_llm is None


# ---------------------------------------------------------------------------
# Runner warning on mis-configuration
# ---------------------------------------------------------------------------


def test_runner_warns_when_goal_drift_enabled_without_call_llm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runner(goal_drift_enabled=True, steerer=DefaultSteerer()) warns once."""

    class _NoopAdapter:
        available_agents: list[str] = []

        async def register_reporting_tools(self, tools):
            return None

    class _NoopExecutor:
        async def run(self, **kwargs):
            raise AssertionError("not invoked")

    steerer = DefaultSteerer()  # no judge callable
    with caplog.at_level(logging.WARNING, logger="goldfive.runner"):
        Runner(
            agent=_NoopAdapter(),
            planner=StubPlanner(),
            executor=_NoopExecutor(),
            steerer=steerer,
            goal_drift_enabled=True,
        )
    matching = [
        r for r in caplog.records
        if r.name == "goldfive.runner"
        and "goal-drift judge disabled" in r.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected exactly one warning, got {len(matching)}: "
        f"{[r.getMessage() for r in matching]}"
    )


def test_runner_does_not_warn_when_call_llm_is_wired(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning is strictly conditional on missing wiring."""

    class _NoopAdapter:
        available_agents: list[str] = []

        async def register_reporting_tools(self, tools):
            return None

    class _NoopExecutor:
        async def run(self, **kwargs):
            raise AssertionError("not invoked")

    call_llm = _stub_call_llm([{"progressing": True}])
    steerer = DefaultSteerer(goal_drift_call_llm=call_llm)
    with caplog.at_level(logging.WARNING, logger="goldfive.runner"):
        Runner(
            agent=_NoopAdapter(),
            planner=StubPlanner(),
            executor=_NoopExecutor(),
            steerer=steerer,
            goal_drift_enabled=True,
        )
    assert not any(
        "goal-drift judge disabled" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Judge fires end-to-end when wired
# ---------------------------------------------------------------------------


async def test_goal_drift_judge_fires_when_wired() -> None:
    """Steerer with a wired judge emits a GOAL_DRIFT drift at the interval.

    This is the regression guard on the top-of-stack symptom in
    goldfive#217: the pipeline must transform a ``progressing=false``
    verdict into a ``DriftKind.GOAL_DRIFT`` event on the sink rather
    than silently skip the check.
    """
    call_llm = _stub_call_llm(
        [{"progressing": False, "reason": "researching raccoons instead of solar panels"}]
    )
    steerer = DefaultSteerer(
        goal_drift_check_interval=3,
        goal_drift_call_llm=call_llm,
        goal_drift_model="fake-model",
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    # Below interval -- no judge call yet.
    for _ in range(2):
        await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 0  # type: ignore[attr-defined]

    # Third turn crosses the interval -> judge fires, drift is emitted.
    await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]

    from goldfive.pb.goldfive.v1 import types_pb2

    drifts = [
        e for e in sink.events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_GOAL_DRIFT
    ]
    assert len(drifts) == 1, f"expected one GOAL_DRIFT event, got {len(drifts)}"
    assert "raccoons" in drifts[0].drift_detected.detail
