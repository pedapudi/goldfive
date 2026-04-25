"""Unit tests for :mod:`goldfive.drift.reasoning_judge` and the
per-task rate limit on :meth:`DefaultSteerer.observe_reasoning`.

Covers (see goldfive#226):

* Happy-path JSON responses (on_task=true / on_task=false with each
  severity).
* Malformed JSON / missing keys / LLM raises -> quiet (``None``).
* Log lines: DEBUG for parse paths, INFO on drift emission, WARNING
  when ``call_llm`` raises. Mirrors the logging pattern from
  :mod:`goldfive.drift.goals`.
* Rate limit: first thinking message on every task always fires; then
  every ``reasoning_drift_rate_limit`` messages after. Counters reset
  on task transition.
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

from goldfive.drift import reasoning_judge as rjudge  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


def _stub_call_llm(responses: list[Any]):
    """Async ``CallLLM``-shaped stub popping responses in order."""
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        if isinstance(resp, Exception):
            raise resp
        return str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


def _raising_call_llm(exc: Exception):
    async def _call_llm(system: str, user: str, model: str) -> str:
        raise exc

    return _call_llm


def _task() -> Task:
    return Task(id="t1", title="Research solar panels", description="Find specs")


def _goals() -> list[Goal]:
    return [Goal(id="g1", summary="Publish a memo on solar panels")]


async def _wait_for_judges(steerer: DefaultSteerer) -> None:
    """Drain any background reasoning-judge tasks the steerer scheduled.

    Tests that assert on ``call_llm.calls`` / sink events after
    :meth:`DefaultSteerer.observe_reasoning` need to wait for the
    fire-and-forget judge task to finish — the method itself returns
    before the judge runs (goldfive#251).
    """
    pending = list(steerer._background_judges)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _session_with_task(task_id: str = "t1") -> Session:
    task = Task(id=task_id, title="Research solar panels", description="Find specs")
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=_goals(),
        plan=plan,
        current_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# classify_reasoning_drift: happy paths
# ---------------------------------------------------------------------------


async def test_on_task_true_returns_none() -> None:
    call_llm = _stub_call_llm([{"on_task": True}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="I should look up the solar panel efficiency ratings.",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is None
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "severity_str, expected_severity",
    [
        ("info", DriftSeverity.INFO),
        ("warning", DriftSeverity.WARNING),
        ("critical", DriftSeverity.CRITICAL),
    ],
)
async def test_on_task_false_maps_severity(
    severity_str: str, expected_severity: DriftSeverity
) -> None:
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": severity_str, "reason": "drifted to raccoons"}]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="Raccoons have masks but this is about solar panels.",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="a1",
    )
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    assert drift.severity is expected_severity
    assert "raccoons" in drift.detail
    assert drift.current_task_id == "t1"
    assert drift.current_agent_id == "a1"


async def test_missing_severity_defaults_to_warning() -> None:
    call_llm = _stub_call_llm([{"on_task": False, "reason": "drifted"}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="off-topic thought",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


async def test_unknown_severity_defaults_to_warning() -> None:
    call_llm = _stub_call_llm([{"on_task": False, "severity": "CATASTROPHIC"}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="off-topic thought",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


async def test_tolerates_markdown_fenced_json() -> None:
    raw = '```json\n{"on_task": false, "severity": "warning", "reason": "off"}\n```'
    call_llm = _stub_call_llm([raw])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


# ---------------------------------------------------------------------------
# classify_reasoning_drift: quiet-on-failure
# ---------------------------------------------------------------------------


async def test_malformed_json_returns_none_with_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_llm = _stub_call_llm(["not json at all"])
    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning_judge"):
        drift = await rjudge.classify_reasoning_drift(
            reasoning="thought",
            task=_task(),
            goals=_goals(),
            model="fake",
            call_llm=call_llm,
        )
    assert drift is None
    assert any(
        "response was not JSON" in r.getMessage()
        and r.name == "goldfive.drift.reasoning_judge"
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


async def test_missing_on_task_key_returns_none_with_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_llm = _stub_call_llm([{"verdict": "drift"}])
    with caplog.at_level(logging.DEBUG, logger="goldfive.drift.reasoning_judge"):
        drift = await rjudge.classify_reasoning_drift(
            reasoning="thought",
            task=_task(),
            goals=_goals(),
            model="fake",
            call_llm=call_llm,
        )
    assert drift is None
    assert any(
        "lacks boolean 'on_task'" in r.getMessage()
        and r.name == "goldfive.drift.reasoning_judge"
        for r in caplog.records
    )


async def test_non_boolean_on_task_returns_none() -> None:
    call_llm = _stub_call_llm([{"on_task": "yes"}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is None


async def test_call_llm_raises_returns_none_with_warning_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_llm = _raising_call_llm(RuntimeError("boom"))
    with caplog.at_level(logging.WARNING, logger="goldfive.drift.reasoning_judge"):
        drift = await rjudge.classify_reasoning_drift(
            reasoning="thought",
            task=_task(),
            goals=_goals(),
            model="fake",
            call_llm=call_llm,
        )
    assert drift is None
    matching = [
        r for r in caplog.records
        if r.name == "goldfive.drift.reasoning_judge"
        and "call_llm raised" in r.getMessage()
    ]
    assert len(matching) == 1
    assert matching[0].levelno == logging.WARNING


async def test_drift_emission_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "critical", "reason": "off"}]
    )
    with caplog.at_level(logging.INFO, logger="goldfive.drift.reasoning_judge"):
        drift = await rjudge.classify_reasoning_drift(
            reasoning="thought",
            task=_task(),
            goals=_goals(),
            model="fake",
            call_llm=call_llm,
        )
    assert drift is not None
    assert any(
        "drift detected" in r.getMessage() and r.levelno == logging.INFO
        for r in caplog.records
    )


async def test_empty_reasoning_does_not_call_llm() -> None:
    call_llm = _stub_call_llm([{"on_task": True}])
    drift = await rjudge.classify_reasoning_drift(
        reasoning="   ",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert drift is None
    # LLM untouched — rate-limiting / short-circuit semantics.
    assert call_llm.calls == []  # type: ignore[attr-defined]


async def test_truncates_long_reasoning_in_prompt() -> None:
    call_llm = _stub_call_llm([{"on_task": True}])
    big = "x " * (rjudge.REASONING_DRIFT_MAX_REASONING_CHARS + 500)
    await rjudge.classify_reasoning_drift(
        reasoning=big,
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    # The user prompt carries the truncation marker, not the full text.
    _system, user, _model = call_llm.calls[0]  # type: ignore[attr-defined]
    assert "[truncated]" in user
    assert len(user) < len(big) + 2000  # prompt framing but not full 3500 chars


# ---------------------------------------------------------------------------
# DefaultSteerer: per-task rate limit
# ---------------------------------------------------------------------------


async def test_rate_limit_fires_first_call_then_every_N(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rate limit 3: judge fires on messages 1, 4, 7, ... per task."""
    call_llm = _stub_call_llm([{"on_task": True}] * 10)
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_rate_limit=3,
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # 1st turn -> judge fires (count=0 starts the task).
    await steerer.observe_reasoning("turn 1", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # 2nd / 3rd turn -> skip (count=1,2).
    await steerer.observe_reasoning("turn 2", session=session)
    await steerer.observe_reasoning("turn 3", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # 4th turn -> fire again (count=3 = 3 % 3 == 0).
    await steerer.observe_reasoning("turn 4", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]
    # 5th-6th -> skip.
    await steerer.observe_reasoning("turn 5", session=session)
    await steerer.observe_reasoning("turn 6", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]
    # 7th -> fire.
    await steerer.observe_reasoning("turn 7", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 3  # type: ignore[attr-defined]


async def test_rate_limit_resets_on_task_transition() -> None:
    """The first reasoning message on a fresh task always fires a judge call."""
    call_llm = _stub_call_llm([{"on_task": True}] * 10)
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_rate_limit=3,
        reasoning_drift_mode="judge",
    )
    session = _session_with_task("t1")
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # Two thinking messages on t1 -> one judge call (1st fires, 2nd skips).
    await steerer.observe_reasoning("t1 turn 1", session=session)
    await steerer.observe_reasoning("t1 turn 2", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]

    # Task transition: add a fresh task and bind it as current.
    new_task = Task(id="t2", title="Different task", description="...")
    assert session.plan is not None
    session.plan.tasks.append(new_task)
    session.current_task_id = "t2"

    # First reasoning on t2 -> fresh judge call (the counter for t2 is 0).
    await steerer.observe_reasoning("t2 turn 1", session=session)
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]


async def test_judge_disabled_when_call_llm_is_none() -> None:
    """Default-mode judge with no call_llm silently no-ops."""
    steerer = DefaultSteerer(
        reasoning_drift_mode="judge",
        reasoning_drift_call_llm=None,
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.observe_reasoning("any thought", session=session)
    await _wait_for_judges(steerer)
    # Only confusion/looping always-on detectors ran, and neither fires
    # on a single clean-text thinking message with no history.
    assert sink.events == []


async def test_judge_rate_limit_buckets_per_agent_not_globally() -> None:
    """Two agents firing unpinned thinking blocks don't share a counter bucket.

    Pre-fix the rate-limit counter keyed on ``current_task_id or ""``,
    so every unpinned turn from every agent collapsed onto the ``""``
    bucket. Agent B's legitimate first thinking block on an unpinned
    turn could legitimately fail to fire the judge because unrelated
    agent A's unpinned turn had already incremented the ``""`` counter.

    Post-fix the key is ``(agent_name, task_id)``. Each agent gets
    its own counter, so agent A's first unpinned block fires, and
    agent B's first unpinned block ALSO fires independently.

    Test matrix (rate_limit=3, task_id=""):

        agent A turn 1 -> FIRE (count=0)    expected calls=1
        agent B turn 1 -> FIRE (count=0)    expected calls=2
        agent A turn 2 -> skip (count=1)    expected calls=2
        agent B turn 2 -> skip (count=1)    expected calls=2
        agent A turn 3 -> skip (count=2)    expected calls=2
        agent B turn 3 -> skip (count=2)    expected calls=2

    With the old global-""-bucket code calls would have been 1 after
    agent B's first turn (skipped because agent A already consumed
    the ``count=0`` firing slot).
    """
    call_llm = _stub_call_llm([{"on_task": True}] * 10)
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_rate_limit=3,
        reasoning_drift_mode="judge",
    )
    # Session with no current_task_id — both agents' turns are unpinned.
    session = Session(
        run_id="r1",
        goals=_goals(),
        plan=Plan(id="p1", run_id="r1", goal_ids=["g1"], tasks=[], edges=[]),
        current_task_id="",
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # Round 1: both agents fire on their first unpinned block.
    await steerer.observe_reasoning("A turn 1", session=session, agent_name="agent_a")
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    await steerer.observe_reasoning("B turn 1", session=session, agent_name="agent_b")
    await _wait_for_judges(steerer)
    # The bug reproduces here: pre-fix this would stay at 1 because
    # agent A's turn had already incremented the shared ``""`` bucket.
    assert len(call_llm.calls) == 2, (  # type: ignore[attr-defined]
        f"agent_b's first unpinned block must fire the judge independently "
        f"of agent_a's. Got {len(call_llm.calls)} calls; expected 2. "  # type: ignore[attr-defined]
        f"Pre-fix both agents shared the ``(\"\", \"\")`` bucket."
    )

    # Round 2: both skip (count=1 for each agent's bucket).
    await steerer.observe_reasoning("A turn 2", session=session, agent_name="agent_a")
    await steerer.observe_reasoning("B turn 2", session=session, agent_name="agent_b")
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]

    # Round 3: both skip again (count=2).
    await steerer.observe_reasoning("A turn 3", session=session, agent_name="agent_a")
    await steerer.observe_reasoning("B turn 3", session=session, agent_name="agent_b")
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]

    # Round 4: both fire (count=3 % 3 == 0). Confirms the per-agent
    # counters advance independently and are NOT a shared global bucket.
    await steerer.observe_reasoning("A turn 4", session=session, agent_name="agent_a")
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 3  # type: ignore[attr-defined]
    await steerer.observe_reasoning("B turn 4", session=session, agent_name="agent_b")
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 4  # type: ignore[attr-defined]

    # Sanity: the counters dict is keyed by (agent_name, task_id) tuples.
    counters = session._reasoning_judge_counters
    assert ("agent_a", "") in counters
    assert ("agent_b", "") in counters
    assert counters[("agent_a", "")] == 4
    assert counters[("agent_b", "")] == 4


# ---------------------------------------------------------------------------
# DefaultSteerer: fire-and-forget judge path (goldfive#251)
# ---------------------------------------------------------------------------


async def test_observe_reasoning_returns_fast_when_judge_is_slow() -> None:
    """observe_reasoning must not block on the judge LLM.

    The judge's ``call_llm`` sleeps for 60s. ``observe_reasoning``
    must return within 100 ms because the judge is scheduled as a
    background task, not awaited inline. This is the correctness
    target of goldfive#251: the adapter's model-response callback is
    on the critical path for ADK tool dispatch.
    """

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        await asyncio.sleep(60)
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=slow_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await steerer.observe_reasoning("clean on-task reasoning", session=session)
    elapsed = loop.time() - t0
    try:
        assert elapsed < 0.1, (
            f"observe_reasoning blocked for {elapsed:.3f}s; expected <0.1s "
            "(fire-and-forget regression)"
        )
        # The background task is live and still sleeping.
        assert len(steerer._background_judges) == 1
    finally:
        # Cancel the slow judge so the test doesn't linger.
        for task in list(steerer._background_judges):
            task.cancel()
        await asyncio.gather(
            *steerer._background_judges, return_exceptions=True
        )


async def test_observe_reasoning_judge_still_fires_in_background() -> None:
    """The backgrounded judge runs after observe_reasoning returns.

    Schedules a judge that emits an OFF_TOPIC WARNING drift (well
    below the USER_STEER promotion threshold so the test doesn't
    tangle with refine-fallback follow-up drifts).
    :meth:`observe_reasoning` returns immediately; awaiting
    ``_background_judges`` then completes the judge path and the
    ``DriftDetected`` sink event materializes on the sink.
    """
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "info", "reason": "slightly off"}]
    )
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.observe_reasoning("raccoons are nocturnal", session=session)
    # Judge hasn't run yet -> no drift emitted.
    drift_events_pre = [
        e for e in sink.events if e.WhichOneof("payload") == "drift_detected"
    ]
    assert drift_events_pre == []
    # Drain the background task.
    await _wait_for_judges(steerer)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    drift_events_post = [
        e for e in sink.events if e.WhichOneof("payload") == "drift_detected"
    ]
    # Exactly one DriftDetected: the INFO OFF_TOPIC from the judge.
    # (INFO severity stays under the intervention ladder's refine
    # threshold so no downstream refine-failure drift cascades.)
    assert len(drift_events_post) == 1


async def test_observe_reasoning_judge_exception_does_not_crash(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the background judge coroutine is swallowed.

    We simulate a failure below the level
    :func:`classify_reasoning_drift` normally catches by patching
    :func:`~goldfive.drift.reasoning.analyze_reasoning` to raise
    directly. The background task's outer try/except must log at
    WARNING on the steerer logger and keep the set drainable.
    ``observe_reasoning`` itself must not propagate the failure to
    the adapter callback that scheduled it.
    """
    from goldfive.drift import reasoning as reasoning_mod

    async def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("judge exploded")

    # The steerer's background judge uses ``analyze_reasoning_with_focus``
    # since Phase 1 of goldfive#271 (extended verdict path). Patch both
    # entry points so the test stays robust whichever one the steerer
    # ends up calling.
    monkeypatch.setattr(reasoning_mod, "analyze_reasoning", _boom)
    monkeypatch.setattr(reasoning_mod, "analyze_reasoning_with_focus", _boom)

    # Any valid judge wiring -- analyze_reasoning is monkeypatched
    # before the background task dispatches.
    call_llm = _stub_call_llm([{"on_task": True}])
    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    with caplog.at_level(logging.WARNING, logger="goldfive.steerer"):
        await steerer.observe_reasoning("some thought", session=session)
        await _wait_for_judges(steerer)

    # No drift was emitted (analyze_reasoning never returned a verdict).
    assert [
        e for e in sink.events if e.WhichOneof("payload") == "drift_detected"
    ] == []
    # The set drains cleanly (the raising task resolved).
    assert steerer._background_judges == set()
    # The raise was swallowed and surfaced as a WARNING on the steerer
    # logger.
    steerer_warnings = [
        r for r in caplog.records
        if r.name == "goldfive.steerer"
        and r.levelno == logging.WARNING
        and "background reasoning-judge raised" in r.getMessage()
    ]
    assert len(steerer_warnings) == 1


async def test_shutdown_bounded_timeout_does_not_wait_for_slow_judge() -> None:
    """``shutdown(timeout=...)`` returns within the bound even if a judge hangs.

    A hung judge (10s sleep) must not stall runner teardown. The
    shutdown cancels the stragglers and returns.
    """

    async def hung_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        await asyncio.sleep(10)
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=hung_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # Schedule a judge that will be running when shutdown fires.
    await steerer.observe_reasoning("slow-path thought", session=session)
    assert len(steerer._background_judges) == 1

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await steerer.shutdown(timeout=0.5)
    elapsed = loop.time() - t0
    # 0.5 timeout + 0.5 cancel-drain budget = ~1.0s ceiling; we want
    # well under the judge's 10s. Allow a little jitter for slow CI.
    assert elapsed < 2.0, (
        f"shutdown took {elapsed:.3f}s; expected <2.0s (bounded-timeout regression)"
    )


async def test_shutdown_is_noop_when_no_background_judges() -> None:
    """Calling ``shutdown`` with nothing in flight is instant and safe."""
    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=NullPlanner())
    # No observe_reasoning call -> empty set.
    assert steerer._background_judges == set()
    await steerer.shutdown(timeout=5.0)


async def test_shutdown_quiet_when_zero_tasks_actually_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase 2.X (goldfive#271 Gap 3): the ``shutdown`` WARNING fires
    only when there were actually stragglers to cancel.

    A judge that completes in the same instant the timeout fires can
    leave ``still_pending`` empty even though ``wait_for`` raised
    ``TimeoutError``. The previous unconditional WARNING logged
    ``cancelled 0 tasks`` which was both confusing and noisy in the
    demo log. The DEBUG line preserves diagnostic visibility.
    """

    async def fast_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        # Returns immediately so the judge completes before / during
        # the shutdown's wait_for. With timeout=0.0 we deterministically
        # hit the "TimeoutError but 0 still pending" branch.
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=fast_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session_with_task()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.observe_reasoning("a thought", session=session)

    with caplog.at_level(logging.WARNING, logger="goldfive.steerer"):
        # timeout=0.0 → wait_for raises TimeoutError immediately even
        # if the gather wins the race in the same instant.
        await steerer.shutdown(timeout=0.0)

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "exceeded" in r.getMessage()
    ]
    if warnings:
        # If a WARNING fired, it must NOT be the misleading "0 tasks"
        # variant — that's the regression we're guarding against.
        for rec in warnings:
            assert "0 background judge task(s)" not in rec.getMessage(), (
                "shutdown WARNING regressed to 'cancelled 0 tasks' shape; "
                f"got {rec.getMessage()!r}"
            )


# ---------------------------------------------------------------------------
# Phase 1 of goldfive#271 — extended verdict (focused_task_id)
# ---------------------------------------------------------------------------


async def test_classify_with_focus_returns_verdict_with_attribution() -> None:
    """The judge's focused_task_id + confidence surface on the verdict."""
    call_llm = _stub_call_llm(
        [
            {
                "on_task": True,
                "focused_task_id": "t1",
                "focus_confidence": 0.9,
                "stated_intent": "researching solar panels",
            }
        ]
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="Research solar panels")],
        edges=[],
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="I'll start by reviewing the solar-panel datasheets.",
        task=_task(),
        goals=_goals(),
        plan=plan,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None  # on-task
    assert verdict.focused_task_id == "t1"
    assert verdict.focus_confidence == 0.9
    assert verdict.stated_intent == "researching solar panels"


async def test_classify_with_focus_off_task_carries_drift_and_attribution() -> None:
    """An off-task verdict still records the focus the agent has switched to."""
    call_llm = _stub_call_llm(
        [
            {
                "on_task": False,
                "severity": "warning",
                "reason": "switched to write_report",
                "focused_task_id": "t2",
                "focus_confidence": 0.95,
                "stated_intent": "writing the final report",
            }
        ]
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Research"),
            Task(id="t2", title="Write report"),
        ],
        edges=[],
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="Let me just start drafting the report instead.",
        task=Task(id="t1", title="Research"),
        goals=[],
        plan=plan,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind == DriftKind.OFF_TOPIC
    assert verdict.drift.severity == DriftSeverity.WARNING
    # Off-task reasoning still reveals attribution: t2.
    assert verdict.focused_task_id == "t2"
    assert verdict.focus_confidence == 0.95


async def test_classify_with_focus_clamps_confidence_to_unit_interval() -> None:
    """Confidence outside [0, 1] is clamped at parse time."""
    call_llm = _stub_call_llm(
        [{"on_task": True, "focused_task_id": "t1", "focus_confidence": 5.0}]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.focus_confidence == 1.0


async def test_classify_with_focus_off_plan_yields_empty_focus() -> None:
    """A judge that returns no attribution gives focused_task_id=''."""
    call_llm = _stub_call_llm(
        [
            {
                "on_task": False,
                "severity": "info",
                "focused_task_id": "",
                "focus_confidence": 0.0,
            }
        ]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="random tangent",
        task=_task(),
        goals=[],
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.focused_task_id == ""
    assert verdict.focus_confidence == 0.0


async def test_classify_with_focus_malformed_response_returns_empty_verdict() -> None:
    """A non-JSON response yields a verdict with everything empty."""
    call_llm = _stub_call_llm(["this is not JSON at all"])
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.focused_task_id == ""
    assert verdict.focus_confidence == 0.0


async def test_classify_with_focus_call_llm_raises_returns_empty_verdict() -> None:
    """An LLM exception leaves attribution fields at default ('no signal')."""
    call_llm = _raising_call_llm(RuntimeError("503"))
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thinking",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.focused_task_id == ""
    assert verdict.focus_confidence == 0.0


async def test_classify_with_focus_empty_reasoning_skips_llm_call() -> None:
    """An empty reasoning block returns an empty verdict without calling the LLM."""
    call_llm = _stub_call_llm([])  # would fail if called
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.focused_task_id == ""


async def test_legacy_classify_reasoning_drift_returns_drift_only() -> None:
    """The back-compat wrapper returns just the drift component.

    Existing call sites depend on ``DriftEvent | None`` — the
    extended fields live on
    :func:`classify_reasoning_drift_with_focus`. This test pins the
    delegating wrapper.
    """
    call_llm = _stub_call_llm(
        [
            {
                "on_task": False,
                "severity": "critical",
                "focused_task_id": "t1",
                "focus_confidence": 0.9,
            }
        ]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="abandon ship",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
    )
    assert isinstance(drift, DriftEvent)
    assert drift.severity == DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Plan-tasks summary truncation
# ---------------------------------------------------------------------------


def test_format_plan_tasks_summary_renders_id_arrow_title() -> None:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t1", title="Alpha"),
            Task(id="t2", title="Beta"),
        ],
        edges=[],
    )
    rendered = rjudge.format_plan_tasks_summary(plan)
    assert "- t1 -> Alpha" in rendered
    assert "- t2 -> Beta" in rendered


def test_format_plan_tasks_summary_truncates_when_over_budget() -> None:
    """A pathologically long task list is truncated with a marker."""
    tasks = [
        Task(id=f"t{i}", title="x" * 80)
        for i in range(100)
    ]
    plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=tasks, edges=[])
    rendered = rjudge.format_plan_tasks_summary(plan, max_chars=200)
    # Truncation marker is present.
    assert "more task" in rendered
    # The rendered text obeys the cap (modulo the truncation line).
    body_lines = [
        line for line in rendered.splitlines()
        if not line.startswith("...")
    ]
    assert sum(len(line) for line in body_lines) <= 220  # cap + small slack


def test_format_plan_tasks_summary_empty_plan_renders_placeholder() -> None:
    assert rjudge.format_plan_tasks_summary(None) == "(no plan tasks)"
    empty_plan = Plan(id="p1", run_id="r1", goal_ids=[], tasks=[], edges=[])
    assert rjudge.format_plan_tasks_summary(empty_plan) == "(no plan tasks)"


def test_classify_with_focus_renders_plan_tasks_into_prompt() -> None:
    """The judge's user prompt includes the plan-tasks summary section."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[Task(id="t1", title="Alpha"), Task(id="t2", title="Beta")],
        edges=[],
    )
    rendered = rjudge.REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(
        plan_tasks_summary=rjudge.format_plan_tasks_summary(plan),
        goals_block="(no goals)",
        task_block="(no task bound)",
        reasoning_block="thinking",
    )
    assert "PLAN TASKS" in rendered
    assert "t1 -> Alpha" in rendered
    assert "focused_task_id" in rendered
    assert "focus_confidence" in rendered
