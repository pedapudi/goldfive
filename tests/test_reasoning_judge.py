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
    # iter-10 PR 3 retitled the log line: now mentions BOTH the
    # three-state ``classification`` key and the legacy ``on_task``
    # bool, since the parser tries both before quiet-failing.
    assert any(
        "lacks both 'classification' and boolean 'on_task'" in r.getMessage()
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
    # goldfive#247: Plan is frozen — extend via add_tasks.
    from goldfive.types import (
        add_tasks,
        channel_processor_active,
        set_session_plan,
    )
    with channel_processor_active():
        set_session_plan(session, add_tasks(session.plan, [new_task]))
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
    # Only the looping always-on detector ran, and it does not fire
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
        current_agent_id="(unknown)",
        task_lineage_block="(no task lineage observed)",
        tool_obs_block="(no recent tool observations)",
        tool_obs_count=0,
    )
    assert "PLAN TASKS" in rendered
    assert "t1 -> Alpha" in rendered
    assert "focused_task_id" in rendered
    assert "focus_confidence" in rendered


# ---------------------------------------------------------------------------
# iter-10 PR 3 — three-state classification parser
# ---------------------------------------------------------------------------


async def test_parser_accepts_on_task_classification() -> None:
    """``classification: "on_task"`` (no legacy on_task field) → no drift."""
    call_llm = _stub_call_llm([{"classification": "on_task", "reason": "ok"}])
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.classification == "on_task"
    assert verdict.provenance == ""


@pytest.mark.parametrize(
    "provenance",
    ["tool_error", "surprising_result", "discovered_dependency", "new_information"],
)
async def test_parser_accepts_justified_deviation_with_each_provenance(
    provenance: str,
) -> None:
    """All four provenance enum values produce a JUSTIFIED_DEVIATION drift."""
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "Got a surprise",
                "provenance": provenance,
            }
        ]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="reasoning that pivots after a tool result",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="a1",
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.JUSTIFIED_DEVIATION
    assert verdict.drift.severity is DriftSeverity.WARNING
    assert verdict.classification == "justified_deviation"
    assert verdict.provenance == provenance
    # Detail prefix carries the provenance string so the refine prompt
    # surfaces it via _render_off_topic_reasoning_block.
    assert f"justified deviation ({provenance})" in verdict.drift.detail


async def test_parser_accepts_erroneous_deviation_classification() -> None:
    """``classification: "erroneous_deviation"`` produces an OFF_TOPIC drift."""
    call_llm = _stub_call_llm(
        [
            {
                "classification": "erroneous_deviation",
                "severity": "warning",
                "reason": "drifted to raccoons",
            }
        ]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="raccoons have stripes; let me look those up",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="a1",
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"
    assert verdict.provenance == ""
    assert "raccoons" in verdict.drift.detail


async def test_parser_demotes_justified_deviation_with_none_provenance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``provenance: "none"`` on a justified verdict → demote to OFF_TOPIC.

    Per §2.4 rule 3: ``"none"`` is a legal value only on the on_task
    or erroneous branches. On a justified_deviation it is treated as
    "model couldn't name a signal" → demote.
    """
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "claimed-justified-but-unsourced",
                "provenance": "none",
            }
        ]
    )
    with caplog.at_level(logging.INFO, logger="goldfive.drift.reasoning_judge"):
        verdict = await rjudge.classify_reasoning_drift_with_focus(
            reasoning="thought",
            task=_task(),
            goals=_goals(),
            plan=None,
            model="fake",
            call_llm=call_llm,
        )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"
    assert verdict.provenance == ""
    # Demotion is audited at INFO so operators can grep the rate.
    assert any(
        "demoted to erroneous_deviation" in r.getMessage()
        and r.name == "goldfive.drift.reasoning_judge"
        for r in caplog.records
    )


async def test_parser_demotes_justified_deviation_with_unknown_provenance() -> None:
    """An unrecognised provenance string → demote to erroneous_deviation."""
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "claimed-justified-with-bogus-provenance",
                "provenance": "guesswork",
            }
        ]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"
    assert verdict.provenance == ""


async def test_parser_demotes_justified_deviation_with_missing_provenance() -> None:
    """Missing ``provenance`` key on a justified verdict → demote."""
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "no-provenance-key",
            }
        ]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"
    assert verdict.provenance == ""


async def test_parser_legacy_on_task_true_yields_on_task() -> None:
    """Legacy ``{"on_task": true}`` (no classification) → on_task / no drift.

    §2.4 rule 2: back-compat for custom prompt-template overrides.
    """
    call_llm = _stub_call_llm([{"on_task": True}])
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.classification == "on_task"


async def test_parser_legacy_on_task_false_yields_erroneous() -> None:
    """Legacy ``{"on_task": false, ...}`` → erroneous_deviation / OFF_TOPIC."""
    call_llm = _stub_call_llm(
        [{"on_task": False, "severity": "warning", "reason": "drifted"}]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"


async def test_parser_quiet_fail_on_missing_classification_and_on_task() -> None:
    """Neither ``classification`` nor a boolean ``on_task`` → quiet-fail."""
    call_llm = _stub_call_llm([{"severity": "warning"}])
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is None
    assert verdict.classification == ""
    assert verdict.provenance == ""


async def test_parser_tolerates_markdown_fences_three_state() -> None:
    """Three-state JSON wrapped in ```json fences``` parses cleanly."""
    raw = (
        "```json\n"
        '{"classification": "justified_deviation", '
        '"severity": "warning", '
        '"reason": "tool 503", '
        '"provenance": "tool_error"}\n'
        "```"
    )
    call_llm = _stub_call_llm([raw])
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="retrying after 503",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.JUSTIFIED_DEVIATION
    assert verdict.classification == "justified_deviation"
    assert verdict.provenance == "tool_error"


async def test_parser_unknown_classification_falls_back_to_legacy_on_task() -> None:
    """Half-correct ``classification`` (not in the enum) → legacy fallback.

    Defensive: if the model returns ``classification: "drift"`` or
    similar, the parser must still fall through to the legacy
    ``on_task`` field. This protects custom-prompt operators who may
    inadvertently trigger novel verdict shapes.
    """
    call_llm = _stub_call_llm(
        [{"classification": "drift", "on_task": False, "severity": "warning", "reason": "x"}]
    )
    verdict = await rjudge.classify_reasoning_drift_with_focus(
        reasoning="thought",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
    )
    assert verdict.drift is not None
    assert verdict.drift.kind is DriftKind.OFF_TOPIC
    assert verdict.classification == "erroneous_deviation"


# ---------------------------------------------------------------------------
# iter-10 PR 3 — prompt rendering helpers
# ---------------------------------------------------------------------------


def test_format_task_lineage_empty() -> None:
    """No lineage → ``(no task lineage observed)``."""
    assert (
        rjudge._format_task_lineage("t1", None, "a1") == "(no task lineage observed)"
    )
    assert (
        rjudge._format_task_lineage("t1", {}, "a1") == "(no task lineage observed)"
    )
    # Task not in lineage map → also empty.
    assert (
        rjudge._format_task_lineage("t1", {"t2": {"a1"}}, "a1")
        == "(no task lineage observed)"
    )
    # Empty set for task → also empty.
    assert (
        rjudge._format_task_lineage("t1", {"t1": set()}, "a1")
        == "(no task lineage observed)"
    )


def test_format_task_lineage_includes_agent_when_in_lineage() -> None:
    rendered = rjudge._format_task_lineage("t1", {"t1": {"a1", "b1"}}, "a1")
    # Sorted membership-list, with "IS in this lineage" suffix.
    assert "observed agents for this task: a1, b1" in rendered
    assert "a1 IS in this lineage" in rendered


def test_format_task_lineage_when_agent_not_in_lineage() -> None:
    rendered = rjudge._format_task_lineage("t1", {"t1": {"a1", "b1"}}, "c1")
    assert "observed agents for this task: a1, b1" in rendered
    assert "c1 is NOT in this lineage" in rendered


def test_format_tool_observations_empty() -> None:
    block, count = rjudge._format_tool_observations(None, task_id="t1")
    assert block == "(no recent tool observations)"
    assert count == 0
    block, count = rjudge._format_tool_observations([], task_id="t1")
    assert block == "(no recent tool observations)"
    assert count == 0


def test_format_tool_observations_filters_by_task() -> None:
    """Per-task filtering: when current task has obs, render ONLY those."""
    obs = [
        {
            "task_id": "t_other",
            "agent_name": "a1",
            "tool_name": "fetch",
            "args_preview": "{}",
            "result_preview": "ok",
            "is_error": False,
        },
        {
            "task_id": "t1",
            "agent_name": "a2",
            "tool_name": "search",
            "args_preview": '{"q": "x"}',
            "result_preview": "[]",
            "is_error": False,
        },
    ]
    block, count = rjudge._format_tool_observations(obs, task_id="t1")
    assert count == 1
    assert "search" in block
    assert "fetch" not in block


def test_format_tool_observations_falls_back_to_global() -> None:
    """When current task has no obs, fall back to the global slice."""
    obs = [
        {
            "task_id": "t_other",
            "agent_name": "a1",
            "tool_name": "fetch",
            "args_preview": "{}",
            "result_preview": "ok",
            "is_error": False,
        },
        {
            "task_id": "t_other2",
            "agent_name": "a2",
            "tool_name": "search",
            "args_preview": "{}",
            "result_preview": "[]",
            "is_error": True,
        },
    ]
    block, count = rjudge._format_tool_observations(obs, task_id="t_missing")
    assert count == 2
    assert "fetch" in block
    assert "search" in block
    # Error marker rendered for the failing entry.
    assert "ERROR" in block
    assert "ok" in block  # the success marker prefix


def test_format_tool_observations_caps_at_max_chars() -> None:
    """Block stays ≤ 1500 chars even when fed many observations.

    Each entry's args_preview / result_preview are already bounded at
    write time (240 / 480 chars). The helper enforces total block
    size as the second-line defence.
    """
    big_arg = "x" * 240
    big_result = "y" * 480
    obs = [
        {
            "task_id": "t1",
            "agent_name": f"a{i}",
            "tool_name": "tool",
            "args_preview": big_arg,
            "result_preview": big_result,
            "is_error": False,
        }
        for i in range(50)
    ]
    block, count = rjudge._format_tool_observations(obs, task_id="t1")
    assert len(block) <= rjudge.REASONING_DRIFT_TOOL_OBS_MAX_CHARS
    # Count must be > 0 (we got at least one entry rendered) and < 50
    # (the cap kicked in before we ran out of entries).
    assert 0 < count < 50


def test_prompt_includes_lineage_block_when_set() -> None:
    """Render lineage with two agents into the user prompt."""
    rendered = rjudge.REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(
        plan_tasks_summary="(no plan tasks)",
        goals_block="(no goals)",
        task_block="(no task bound)",
        reasoning_block="thinking",
        current_agent_id="a1",
        task_lineage_block=rjudge._format_task_lineage(
            "t1", {"t1": {"a1", "b1"}}, "a1"
        ),
        tool_obs_block="(no recent tool observations)",
        tool_obs_count=0,
    )
    assert "Task lineage:" in rendered
    assert "observed agents for this task: a1, b1" in rendered
    assert "a1 IS in this lineage" in rendered


def test_prompt_lineage_block_when_absent() -> None:
    rendered = rjudge.REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(
        plan_tasks_summary="(no plan tasks)",
        goals_block="(no goals)",
        task_block="(no task bound)",
        reasoning_block="thinking",
        current_agent_id="a1",
        task_lineage_block=rjudge._format_task_lineage("t1", None, "a1"),
        tool_obs_block="(no recent tool observations)",
        tool_obs_count=0,
    )
    assert "Task lineage: (no task lineage observed)" in rendered


def test_prompt_lineage_block_when_agent_not_in_set() -> None:
    rendered = rjudge.REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(
        plan_tasks_summary="(no plan tasks)",
        goals_block="(no goals)",
        task_block="(no task bound)",
        reasoning_block="thinking",
        current_agent_id="c1",
        task_lineage_block=rjudge._format_task_lineage(
            "t1", {"t1": {"a1", "b1"}}, "c1"
        ),
        tool_obs_block="(no recent tool observations)",
        tool_obs_count=0,
    )
    assert "c1 is NOT in this lineage" in rendered


def test_prompt_renders_three_state_decision_section() -> None:
    """The user prompt template carries the iter-10 'Decide THREE things' block.

    Snapshot-style contains-check: pins the literal markers so a
    well-meaning prompt edit that drops one of the three decisions
    fails this test.
    """
    rendered = rjudge.REASONING_DRIFT_USER_PROMPT_TEMPLATE.format(
        plan_tasks_summary="(no plan tasks)",
        goals_block="(no goals)",
        task_block="(no task bound)",
        reasoning_block="thinking",
        current_agent_id="a1",
        task_lineage_block="(no task lineage observed)",
        tool_obs_block="(no recent tool observations)",
        tool_obs_count=0,
    )
    assert "Decide THREE things:" in rendered
    # The three decisions are listed.
    assert "1. CLASSIFICATION." in rendered
    assert "2. ATTRIBUTION." in rendered
    assert "3. PROVENANCE." in rendered
    # Provenance enum literals appear verbatim (the LLM's output must
    # match these strings; the parser strip+lowercases before
    # comparison but the prompt asks for the canonical names).
    for token in (
        "tool_error",
        "surprising_result",
        "discovered_dependency",
        "new_information",
    ):
        assert token in rendered, token
    # Three-state classification literals also appear verbatim.
    for token in ("on_task", "justified_deviation", "erroneous_deviation"):
        assert token in rendered, token
    # GUIDANCE block is present (LLM behaviour depends on its specifics).
    assert "GUIDANCE:" in rendered


# ---------------------------------------------------------------------------
# iter-10 PR 3 — verdict propagation through the legacy wrapper
# ---------------------------------------------------------------------------


async def test_classify_reasoning_drift_legacy_wrapper_threads_lineage() -> None:
    """The back-compat wrapper accepts and threads the new kwargs.

    Pins the API surface — external callers who upgrade to iter-10
    can pass lineage / tool observations through the legacy wrapper
    too without switching to ``classify_reasoning_drift_with_focus``.
    """
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "tool 503",
                "provenance": "tool_error",
            }
        ]
    )
    drift = await rjudge.classify_reasoning_drift(
        reasoning="retrying after 503",
        task=_task(),
        goals=_goals(),
        model="fake",
        call_llm=call_llm,
        task_lineage={"t1": {"a1"}},
        recent_tool_observations=[
            {
                "task_id": "t1",
                "agent_name": "a1",
                "tool_name": "fetch",
                "args_preview": "{}",
                "result_preview": "503",
                "is_error": True,
            }
        ],
    )
    assert drift is not None
    assert drift.kind is DriftKind.JUSTIFIED_DEVIATION


# ---------------------------------------------------------------------------
# iter-10 PR 3 — span / observability event
# ---------------------------------------------------------------------------


async def test_reasoning_judge_invoked_carries_classification_three_state() -> None:
    """ReasoningJudgeInvoked event payload populates ``classification`` for each
    of the three states.

    Pins the proto-event wiring: PR 1 added the field; PR 3 starts
    populating it from the parser. on_task / justified_deviation /
    erroneous_deviation must all surface their canonical
    classification name on the wire.
    """
    cases: list[tuple[dict[str, Any], str]] = [
        ({"classification": "on_task", "reason": "ok"}, "on_task"),
        (
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "503",
                "provenance": "tool_error",
            },
            "justified_deviation",
        ),
        (
            {
                "classification": "erroneous_deviation",
                "severity": "warning",
                "reason": "drifted",
            },
            "erroneous_deviation",
        ),
    ]
    for response, expected in cases:
        sink = ListSink()
        call_llm = _stub_call_llm([response])
        await rjudge.classify_reasoning_drift_with_focus(
            reasoning="thinking",
            task=_task(),
            goals=_goals(),
            plan=None,
            model="fake",
            call_llm=call_llm,
            current_task_id="t1",
            current_agent_id="a1",
            sink=sink,
            run_id="r1",
            session_id="s1",
        )
        # Pull the ReasoningJudgeInvoked event out of the sink stream.
        invoked: list[Any] = []
        for evt in sink.events:
            payload = getattr(evt, "reasoning_judge_invoked", None)
            if payload is None:
                continue
            # Distinguish between an unset oneof (returns the default
            # message but the envelope's WhichOneof is something else)
            # and a real RJI envelope by checking model is set.
            if getattr(payload, "model", "") == "fake":
                invoked.append(payload)
        assert len(invoked) == 1, (expected, invoked)
        assert invoked[0].classification == expected, (expected, invoked[0])


async def test_judge_span_decision_summary_three_state() -> None:
    """Span ``decision_summary`` carries the three-state classification.

    For justified_deviation, the provenance is rendered alongside
    (e.g. ``"justified_deviation (tool_error)"``).
    """
    sink = ListSink()
    call_llm = _stub_call_llm(
        [
            {
                "classification": "justified_deviation",
                "severity": "warning",
                "reason": "503",
                "provenance": "tool_error",
            }
        ]
    )
    await rjudge.classify_reasoning_drift_with_focus(
        reasoning="retrying after 503",
        task=_task(),
        goals=_goals(),
        plan=None,
        model="fake",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="a1",
        sink=sink,
        run_id="r1",
        session_id="s1",
    )
    # Span end events are emitted onto the sink; find the
    # judge_reasoning span end and check its decision_summary.
    decision_summaries: list[str] = []
    for evt in sink.events:
        end = getattr(evt, "goldfive_llm_call_end", None)
        if end is None:
            continue
        # Distinguish unset oneof from a real End — only consider
        # envelopes whose ``span_id`` is populated (the End helper
        # always sets it).
        if not getattr(end, "span_id", ""):
            continue
        if getattr(end, "decision_summary", ""):
            decision_summaries.append(end.decision_summary)
    assert any(
        "justified_deviation (tool_error)" in s for s in decision_summaries
    ), decision_summaries
