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

import asyncio
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


def test_wrap_wires_call_llm_into_reasoning_drift_judge() -> None:
    """goldfive.wrap threads ``call_llm`` into the reasoning-drift judge too.

    Regression guard on goldfive#226: ``wrap()`` must wire
    ``reasoning_drift_call_llm`` / ``reasoning_drift_model`` on the
    default steerer, otherwise the default mode (``"judge"``) would
    silently no-op despite the user passing a callable.
    """
    call_llm = _stub_call_llm([])
    runner = goldfive.wrap(
        _noop_agent,
        call_llm=call_llm,
        model="fake-model",
        sinks=[],
    )
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    assert steerer._reasoning_drift_call_llm is call_llm
    assert steerer._reasoning_drift_model == "fake-model"
    # Default mode stays ``"judge"`` -- same callable wires both judges.
    assert steerer._reasoning_drift_mode == "judge"


def test_wrap_does_not_arm_reasoning_judge_when_steerer_explicit() -> None:
    """An explicit ``steerer=`` wins: wrap() does not patch kwargs in."""
    call_llm = _stub_call_llm([])
    explicit = DefaultSteerer(reasoning_drift_call_llm=None)
    runner = goldfive.wrap(
        _noop_agent,
        call_llm=call_llm,
        steerer=explicit,
        sinks=[],
    )
    assert runner.steerer is explicit
    assert runner.steerer._reasoning_drift_call_llm is None


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


def test_wrap_still_auto_detects_when_planner_and_goal_deriver_explicit() -> None:
    """Regression guard: `detect_llm` runs even when both planner and
    goal_deriver are explicit, so the judges still get armed.

    Prior to this fix, `goldfive.wrap(tree, planner=P, goal_deriver=G)`
    skipped `detect_llm` on the assumption that `call_llm` existed only
    to feed those two callables. PR #218 / #226 then wired the judges
    through the same callable, but the auto-detect guard was never
    updated — so the common "bring-your-own-planner" path silently
    disarmed both judges. This was the root cause of the harmonograf
    demo session where drift in the researcher's raccoon-injected
    reasoning went undetected (session
    ``1aa68419-00f3-41eb-bf6e-22d0bdff21ed``, zero drift_detected
    events).
    """
    from goldfive.results import InvocationResult as _IR  # noqa: F401

    class _AgentWithDetectableLLM:
        """Agent whose shape `detect_llm` recognises."""

        def __init__(self) -> None:
            self.model = "fake-detected-model"

        async def __call__(self, task, session, tools):  # pragma: no cover - unused
            return InvocationResult(task_id=getattr(task, "id", ""), text="ok")

    explicit_planner = StubPlanner()

    class _ExplicitGoalDeriver:
        async def derive(self, user_input: str, **_: Any):
            return []

    runner = goldfive.wrap(
        _AgentWithDetectableLLM(),
        planner=explicit_planner,
        goal_deriver=_ExplicitGoalDeriver(),
        sinks=[],
    )
    # Either detect_llm found a callable (steerer is armed) or it
    # returned None (agent shape not recognised). Either way the
    # auto-detect MUST have been attempted — the guard no longer
    # short-circuits on planner+goal_deriver being explicit.
    steerer = runner.steerer
    assert isinstance(steerer, DefaultSteerer)
    # Our stub agent isn't a real ADK agent, so detect_llm returns
    # None. The meaningful assertion is that with an explicit
    # planner+goal_deriver AND no call_llm, the steerer is
    # unarmed — but that's the correct graceful-degradation state.
    assert steerer._goal_drift_call_llm is None
    assert steerer._reasoning_drift_call_llm is None


def test_wrap_warns_when_judge_mode_without_call_llm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operators should see a WARNING at wrap-time when the judge is disarmed.

    The default `reasoning_drift_mode` is "judge" — silently running
    with the judge mode selected but no `call_llm` wired is the most
    common mis-configuration. Emit a single WARNING so the gap is
    diagnosable from logs.
    """
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(_noop_agent, sinks=[])
    matching = [
        r
        for r in caplog.records
        if "LLM-as-a-judge drift detection is disabled" in r.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected exactly one WARNING, got {len(matching)}: "
        f"{[r.getMessage() for r in matching]}"
    )


def test_wrap_does_not_warn_when_mode_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning when the operator has explicitly disabled judge-mode."""
    from goldfive.config import ReasoningDriftConfig, RuntimeConfig

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(
            _noop_agent,
            runtime=RuntimeConfig(
                reasoning_drift=ReasoningDriftConfig(mode="off"),
            ),
            sinks=[],
        )
    matching = [
        r
        for r in caplog.records
        if "LLM-as-a-judge drift detection is disabled" in r.getMessage()
    ]
    assert matching == []


# ---------------------------------------------------------------------------
# Named-model WARNING on auto-detect inheritance (goldfive silent-disarm
# follow-up). When the judges' LLM was inherited from ``detect_llm`` we
# surface the model name so cloud-billed endpoints are visible from logs.
# ---------------------------------------------------------------------------


def _make_detect_llm(model_name: str) -> tuple[Any, Any]:
    """Build a stubbed LLM detector returning ``(stub_call_llm, model_name)``.

    The real ``detect_llm`` requires ADK-shaped input; for the
    named-model WARNING tests we just need it to report "I found a
    callable on model X". Returning a simple tuple keeps the test
    independent of the ADK optional dep. Callers thread the detector
    through ``goldfive.wrap(llm_detector=...)``.
    """
    stub_call_llm = _stub_call_llm([])

    def _fake_detect(_agent: Any) -> tuple[Any, str]:
        return stub_call_llm, model_name

    return stub_call_llm, _fake_detect


class _MyAgent:
    """Async-callable agent shape accepted by ``auto_adapter``.

    No ``.name`` attribute — the named-model WARNING falls through to
    ``type(agent).__name__`` (``_MyAgent``) and the assertion below
    exercises that fallback.
    """

    async def __call__(self, task: Any, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


class _NamedAgent:
    """Async-callable agent shape with an ADK-style ``.name`` attribute.

    Exercises the preferred branch of the named-model WARNING: prefer
    the agent's own ``.name`` (``coordinator_agent`` in the real ADK
    demo) over the Python class name (``LlmAgent`` / ``_NamedAgent``).
    """

    name = "coordinator_agent"

    async def __call__(self, task: Any, session: Any, tools: Any) -> InvocationResult:
        return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


def test_wrap_warns_named_model_on_detection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Auto-detected judge LLM fires the named-model WARNING with model + agent type."""
    _, detector = _make_detect_llm("gpt-4o-mini")

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(_MyAgent(), sinks=[], llm_detector=detector)

    matching = [
        r
        for r in caplog.records
        if "judge LLM not explicitly configured" in r.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected exactly one named-model WARNING, got {len(matching)}: "
        f"{[r.getMessage() for r in matching]}"
    )
    msg = matching[0].getMessage()
    assert "gpt-4o-mini" in msg
    assert "_MyAgent" in msg
    assert "GOLDFIVE_JUDGE_BASE_URL" in msg


def test_wrap_warns_named_model_prefers_agent_name_over_class_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the agent has a ``.name`` attribute the WARNING uses that.

    Regression guard: the first version of this WARNING printed
    ``type(agent).__name__`` unconditionally, so live demo logs
    showed "agent 'LlmAgent'" — the ADK base class name — instead
    of "agent 'coordinator_agent'", which is what operators actually
    care about.
    """
    _, detector = _make_detect_llm("gpt-4o-mini")

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(_NamedAgent(), sinks=[], llm_detector=detector)

    matching = [
        r
        for r in caplog.records
        if "judge LLM not explicitly configured" in r.getMessage()
    ]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "coordinator_agent" in msg
    assert "_NamedAgent" not in msg  # class name should NOT appear


def test_wrap_suppresses_named_model_warning_when_call_llm_explicit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit ``call_llm=`` means the operator already owns the decision."""
    _, detector = _make_detect_llm("should-not-appear")
    call_llm = _stub_call_llm([])

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(
            _noop_agent, call_llm=call_llm, sinks=[], llm_detector=detector
        )

    matching = [
        r
        for r in caplog.records
        if "judge LLM not explicitly configured" in r.getMessage()
    ]
    assert matching == []


def test_wrap_suppresses_named_model_warning_when_judge_config_explicit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An explicit ``JudgeConfig.base_url`` also suppresses the warning."""
    from goldfive.config import JudgeConfig, RuntimeConfig

    _, detector = _make_detect_llm("should-not-appear")

    # Force the judge-llm build to succeed (we don't actually care about
    # the returned callable here, just that JudgeConfig is honoured).
    def _fake_build(_config: Any) -> tuple[Any, str]:
        return _stub_call_llm([]), "judge-model"

    runtime = RuntimeConfig(
        judge=JudgeConfig(base_url="http://judge:9000", model="judge-model"),
    )

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(
            _noop_agent,
            runtime=runtime,
            sinks=[],
            llm_detector=detector,
            judge_call_llm_builder=_fake_build,
        )

    matching = [
        r
        for r in caplog.records
        if "judge LLM not explicitly configured" in r.getMessage()
    ]
    assert matching == []


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
    with caplog.at_level(logging.WARNING, logger="goldfive"):
        Runner(
            agent=_NoopAdapter(),
            planner=StubPlanner(),
            executor=_NoopExecutor(),
            steerer=steerer,
            goal_drift_enabled=True,
        )
    matching = [
        r for r in caplog.records
        if "goal-drift judge disabled" in r.getMessage()
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
    with caplog.at_level(logging.WARNING, logger="goldfive"):
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
    # Drain any spawned judges (none expected below interval) so the
    # call-llm count assertion is post-judge.
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    assert len(call_llm.calls) == 0  # type: ignore[attr-defined]

    # Third turn crosses the interval -> judge fires, drift is emitted.
    await steerer.note_agent_turn(session)
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]

    from goldfive.pb.goldfive.v1 import types_pb2

    drifts = [
        e for e in sink.events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_GOAL_DRIFT
    ]
    assert len(drifts) == 1, f"expected one GOAL_DRIFT event, got {len(drifts)}"
    assert "raccoons" in drifts[0].drift_detected.detail
