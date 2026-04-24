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
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # 2nd / 3rd turn -> skip (count=1,2).
    await steerer.observe_reasoning("turn 2", session=session)
    await steerer.observe_reasoning("turn 3", session=session)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # 4th turn -> fire again (count=3 = 3 % 3 == 0).
    await steerer.observe_reasoning("turn 4", session=session)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]
    # 5th-6th -> skip.
    await steerer.observe_reasoning("turn 5", session=session)
    await steerer.observe_reasoning("turn 6", session=session)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]
    # 7th -> fire.
    await steerer.observe_reasoning("turn 7", session=session)
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
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]

    # Task transition: add a fresh task and bind it as current.
    new_task = Task(id="t2", title="Different task", description="...")
    assert session.plan is not None
    session.plan.tasks.append(new_task)
    session.current_task_id = "t2"

    # First reasoning on t2 -> fresh judge call (the counter for t2 is 0).
    await steerer.observe_reasoning("t2 turn 1", session=session)
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
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    await steerer.observe_reasoning("B turn 1", session=session, agent_name="agent_b")
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
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]

    # Round 3: both skip again (count=2).
    await steerer.observe_reasoning("A turn 3", session=session, agent_name="agent_a")
    await steerer.observe_reasoning("B turn 3", session=session, agent_name="agent_b")
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]

    # Round 4: both fire (count=3 % 3 == 0). Confirms the per-agent
    # counters advance independently and are NOT a shared global bucket.
    await steerer.observe_reasoning("A turn 4", session=session, agent_name="agent_a")
    assert len(call_llm.calls) == 3  # type: ignore[attr-defined]
    await steerer.observe_reasoning("B turn 4", session=session, agent_name="agent_b")
    assert len(call_llm.calls) == 4  # type: ignore[attr-defined]

    # Sanity: the counters dict is keyed by (agent_name, task_id) tuples.
    counters = session._reasoning_judge_counters
    assert ("agent_a", "") in counters
    assert ("agent_b", "") in counters
    assert counters[("agent_a", "")] == 4
    assert counters[("agent_b", "")] == 4
