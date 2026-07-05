"""Judge-scheduling guards: concurrency cap, queued-window coalescing,
verdict-utility ledger, and the wrap()-time shared-endpoint cost warning.

Deterministic guards + measurement for the reasoning-judge pipeline.
No cadence / coverage / threshold change is asserted here on purpose —
that work is deferred pending the regression harness. What IS pinned:

* at most ``ReasoningDriftConfig.max_concurrent_judges`` background
  judge LLM calls run concurrently (per-steerer semaphore);
* a judge request that is still QUEUED behind the semaphore coalesces
  with a newer observation for the same (agent, task) key — the newest
  window always wins and a RUNNING call is never coalesced;
* the per-session verdict-utility ledger counts
  {acted_on, emitted_late, emitted_redundant, parse_fail} at the
  existing verdict code points and surfaces a
  ``reasoning_judge_utility_summary`` event at the run-boundary drain
  (with a shutdown flush fallback);
* :func:`goldfive.wrap`'s named-model WARNING names the concurrency /
  token cost when the judges inherit the agent tree's own model.
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
from goldfive.config import ReasoningDriftConfig, SteeringConfig  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
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


@pytest.fixture(autouse=True)
def _reset_process_wide_reasoning_config() -> Any:
    """Tests below install ReasoningDriftConfig process-wide via the
    steerer / wrap(); clear it around each test to avoid leakage."""
    from goldfive.drift import reasoning as _reasoning

    _reasoning.configure(None)
    yield
    _reasoning.configure(None)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class NullPlanner:
    """Non-None planner recording refine calls (handle_drift requires one)."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


def _session(run_id: str = "r1", task_id: str = "t1") -> Session:
    task = Task(id=task_id, title="Research solar panels", description="Find specs")
    plan = Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


async def _drain_judges(steerer: DefaultSteerer) -> None:
    pending = list(steerer._background_judges)
    results = await asyncio.gather(*pending, return_exceptions=True)
    for r in results:
        assert not isinstance(r, BaseException), (
            f"background judge raised {r!r}; expected clean completion"
        )


def _summaries(sink: ListSink) -> list[dict[str, Any]]:
    return [
        e
        for e in sink.events
        if isinstance(e, dict) and e.get("kind") == "reasoning_judge_utility_summary"
    ]


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


async def test_semaphore_caps_concurrent_judge_calls() -> None:
    """Five queued judges never exceed ``max_concurrent_judges=2`` in flight."""
    inflight = 0
    max_seen = 0
    calls = 0

    async def slow_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        nonlocal inflight, max_seen, calls
        calls += 1
        inflight += 1
        max_seen = max(max_seen, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=slow_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        reasoning_drift_config=ReasoningDriftConfig(max_concurrent_judges=2),
    )
    session = _session()
    steerer.bind(sinks=[ListSink()], planner=NullPlanner())

    # Distinct agent names -> distinct coalescing keys -> five separate
    # judge windows all contending on the semaphore (mirrors N agents
    # thinking concurrently).
    texts = [
        "study raccoon habitats in urban parks",
        "compare monocrystalline and polycrystalline panels",
        "draft the memo introduction",
        "collect inverter efficiency benchmarks",
        "summarise net-metering policy changes",
    ]
    for i, text in enumerate(texts):
        await steerer.drift.observe_reasoning(text, session=session, agent_name=f"agent-{i}")
    assert len(steerer._background_judges) == 5
    await _drain_judges(steerer)

    assert calls == 5, "distinct (agent, task) keys must never coalesce"
    assert max_seen == 2, (
        f"semaphore must cap concurrency at 2 (and allow reaching 2); observed max {max_seen}"
    )


async def test_semaphore_size_defaults_and_reads_config() -> None:
    """Cap comes from ``ReasoningDriftConfig.max_concurrent_judges``; bare
    steerers use the dataclass default; the gate is per-instance."""
    bare = DefaultSteerer()
    assert bare.drift._judge_semaphore._value == 3

    configured = DefaultSteerer(
        reasoning_drift_config=ReasoningDriftConfig(max_concurrent_judges=7),
    )
    assert configured.drift._judge_semaphore._value == 7
    assert configured.drift._judge_semaphore is not bare.drift._judge_semaphore

    clamped = DefaultSteerer(
        reasoning_drift_config=ReasoningDriftConfig(max_concurrent_judges=0),
    )
    assert clamped.drift._judge_semaphore._value == 1


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


async def test_coalescing_replaces_queued_window_and_never_running() -> None:
    """A QUEUED window coalesces onto the newest observation; the RUNNING
    call keeps the window it started with; the newest text is never lost."""
    gate = asyncio.Event()
    prompts: list[str] = []

    async def gated_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        prompts.append(user)
        await gate.wait()
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=gated_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
        reasoning_drift_config=ReasoningDriftConfig(max_concurrent_judges=1),
    )
    session = _session()
    steerer.bind(sinks=[ListSink()], planner=NullPlanner())

    await steerer.drift.observe_reasoning("study raccoon habitats", session=session, agent_name="a")
    # Let the first task acquire the semaphore: QUEUED -> RUNNING, so
    # its coalescing slot must be gone and its window frozen.
    for _ in range(20):
        if prompts:
            break
        await asyncio.sleep(0)
    assert prompts, "first judge call must have started"
    assert steerer.drift._queued_judge_windows == {}

    await steerer.drift.observe_reasoning(
        "compare solar panel specs", session=session, agent_name="a"
    )
    assert len(steerer.drift._queued_judge_windows) == 1
    tasks_after_second = len(steerer._background_judges)

    await steerer.drift.observe_reasoning("draft the final memo", session=session, agent_name="a")
    # Coalesced: no third task; the queued window carries the newest text.
    assert len(steerer._background_judges) == tasks_after_second
    (queued,) = steerer.drift._queued_judge_windows.values()
    assert queued.text == "draft the final memo"
    assert queued.coalesced == 1

    gate.set()
    await _drain_judges(steerer)
    assert steerer.drift._queued_judge_windows == {}

    # Two judge calls total: the RUNNING one saw the first window
    # (never mutated); the queued one ran with the newest. The
    # superseded middle window was never dispatched.
    assert len(prompts) == 2
    assert "study raccoon habitats" in prompts[0]
    assert "draft the final memo" in prompts[1]
    assert not any("compare solar panel specs" in p for p in prompts)


async def test_cancelled_queued_window_is_removed_from_registry() -> None:
    """A task cancelled while still waiting on the semaphore must drop its
    registry entry so later observations don't coalesce into a dead window."""
    gate = asyncio.Event()

    async def gated_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        await gate.wait()
        return json.dumps({"on_task": True})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=gated_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        reasoning_drift_rate_limit=1,
        reasoning_drift_config=ReasoningDriftConfig(max_concurrent_judges=1),
    )
    session = _session()
    steerer.bind(sinks=[ListSink()], planner=NullPlanner())

    await steerer.drift.observe_reasoning("study raccoon habitats", session=session, agent_name="a")
    await asyncio.sleep(0)  # first task acquires the semaphore
    await steerer.drift.observe_reasoning(
        "compare solar panel specs", session=session, agent_name="a"
    )
    await asyncio.sleep(0)  # second task starts waiting on the semaphore
    assert len(steerer.drift._queued_judge_windows) == 1

    for task in list(steerer._background_judges):
        task.cancel()
    await asyncio.gather(*list(steerer._background_judges), return_exceptions=True)

    assert steerer.drift._queued_judge_windows == {}, (
        "cancelled queued window must not linger in the coalescing registry"
    )
    gate.set()


# ---------------------------------------------------------------------------
# Verdict-utility ledger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observation_only", [True, False])
async def test_ledger_counts_acted_on_and_drain_emits_summary(
    observation_only: bool,
) -> None:
    """An off-task verdict on a live invocation counts ``acted_on`` (in both
    observation-only and active modes) and the run-boundary drain emits one
    summary event; a second drain emits nothing."""
    from goldfive.state_store import StateStore

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"on_task": False, "severity": "warning", "reason": "drifted"})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
        steering_config=SteeringConfig(observation_only=observation_only),
    )
    session = _session()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # Live invocation so the late gate does not fire.
    store = StateStore.for_session(session)

    async def _placeholder() -> None:
        await asyncio.sleep(0.5)

    fake_task = asyncio.create_task(_placeholder())
    store.register_invocation_task("inv-live", fake_task)
    try:
        await steerer.drift.observe_reasoning("raccoons are nocturnal", session=session)
        await _drain_judges(steerer)
    finally:
        store.deregister_invocation_task("inv-live")
        fake_task.cancel()
        await asyncio.gather(fake_task, return_exceptions=True)

    ledger = steerer.drift._verdict_ledgers[session.id]
    assert ledger["acted_on"] == 1
    assert ledger["emitted_late"] == 0
    assert ledger["parse_fail"] == 0
    assert len(ledger["elapsed_ms"]) == 1

    await steerer.drift.drain_session_background_tasks(session_id=session.id)
    summaries = _summaries(sink)
    assert len(summaries) == 1
    payload = summaries[0]["payload"]
    assert payload["acted_on"] == 1
    assert payload["emitted_late"] == 0
    assert payload["emitted_redundant"] == 0
    assert payload["parse_fail"] == 0
    assert payload["judge_calls"] == 1
    assert payload["elapsed_ms_p50"] >= 0
    assert payload["elapsed_ms_p95"] >= payload["elapsed_ms_p50"]
    assert summaries[0]["session_id"] == session.id

    # Idempotent: the pop makes a second drain a no-op.
    await steerer.drift.drain_session_background_tasks(session_id=session.id)
    assert len(_summaries(sink)) == 1


async def test_ledger_counts_emitted_late() -> None:
    """A verdict landing after its invocation terminated counts
    ``emitted_late`` and not ``acted_on``."""

    async def call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return json.dumps({"on_task": False, "severity": "warning", "reason": "drifted"})

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    # No invocation registered — the agent has moved on (goldfive#319).
    await steerer.drift.observe_reasoning("raccoons are nocturnal", session=session)
    await _drain_judges(steerer)

    ledger = steerer.drift._verdict_ledgers[session.id]
    assert ledger["emitted_late"] == 1
    assert ledger["acted_on"] == 0
    assert ledger["parse_fail"] == 0


async def test_ledger_counts_parse_fail_and_shutdown_flushes_summary() -> None:
    """A quiet-failed judge response (empty classification sentinel) counts
    ``parse_fail``; :meth:`shutdown` flushes the summary for sessions that
    never hit a run-boundary drain."""

    async def garbage_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return "definitely not json"

    steerer = DefaultSteerer(
        reasoning_drift_call_llm=garbage_call_llm,
        reasoning_drift_model="fake",
        reasoning_drift_mode="judge",
    )
    session = _session()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=NullPlanner())

    await steerer.drift.observe_reasoning("raccoons are nocturnal", session=session)
    await _drain_judges(steerer)

    ledger = steerer.drift._verdict_ledgers[session.id]
    assert ledger["parse_fail"] == 1
    assert ledger["acted_on"] == 0
    assert len(ledger["elapsed_ms"]) == 1

    await steerer.drift.shutdown(timeout=1.0)
    summaries = _summaries(sink)
    assert len(summaries) == 1
    assert summaries[0]["payload"]["parse_fail"] == 1
    assert steerer.drift._verdict_ledgers == {}


async def test_ledger_counts_redundant_verdicts_at_handle_drift_gates() -> None:
    """Both handle_drift entry gates (addressed-watermark and in-flight
    refine) count ``emitted_redundant`` and skip refine."""
    steerer = DefaultSteerer()
    session = _session()
    sink = ListSink()
    planner = NullPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Gate 1: same (kind, target) already addressed at a later revision.
    session.last_addressed_revision_by_drift_key[(DriftKind.OFF_TOPIC.value, "t1")] = 5
    stale = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="stale view",
        current_task_id="t1",
        observed_revision_index=3,
    )
    await steerer.drift.handle_drift(stale, session)
    assert steerer.drift._verdict_ledgers[session.id]["emitted_redundant"] == 1

    # Gate 2: same key already in-flight.
    session.last_addressed_revision_by_drift_key.clear()
    steerer.drift._inflight_refine_keys.add((session.id, DriftKind.OFF_TOPIC.value, "t1"))
    concurrent = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="concurrent view",
        current_task_id="t1",
        observed_revision_index=4,
    )
    await steerer.drift.handle_drift(concurrent, session)
    assert steerer.drift._verdict_ledgers[session.id]["emitted_redundant"] == 2

    assert planner.refine_calls == [], "gated verdicts must never reach refine"
    drift_events = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    assert len(drift_events) == 2, "gated verdicts stay on the wire (emit-only)"


# ---------------------------------------------------------------------------
# wrap()-time shared-endpoint cost warning
# ---------------------------------------------------------------------------


async def _noop_agent(task: Any, session: Any, tools: Any) -> InvocationResult:  # noqa: ARG001
    return InvocationResult(task_id=getattr(task, "id", ""), text="ok")


def _scripted_detector(model: str = "gpt-4o-mini") -> Any:
    async def call_llm(system: str, user: str, model_str: str) -> str:  # noqa: ARG001
        return ""

    def detect(agent: Any) -> tuple[Any, str]:  # noqa: ARG001
        return (call_llm, model)

    return detect


def test_wrap_warning_names_shared_endpoint_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The auto-detect WARNING fires exactly once and names the concurrency
    cap + per-call output-token ceiling of the inherited judge traffic."""
    from goldfive.drift.reasoning_judge import REASONING_JUDGE_MAX_OUTPUT_TOKENS

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(_noop_agent, sinks=[], llm_detector=_scripted_detector())

    matching = [
        r for r in caplog.records if "judge LLM not explicitly configured" in r.getMessage()
    ]
    assert len(matching) == 1, f"expected exactly one shared-endpoint WARNING, got {len(matching)}"
    msg = matching[0].getMessage()
    assert "gpt-4o-mini" in msg
    assert "up to 3 concurrent background calls" in msg
    assert f"{REASONING_JUDGE_MAX_OUTPUT_TOKENS} output" in msg
    assert "GOLDFIVE_JUDGE_BASE_URL" in msg


def test_wrap_warning_suppressed_when_call_llm_explicit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No shared-endpoint WARNING when the operator supplied ``call_llm=``."""

    async def explicit_call_llm(system: str, user: str, model: str) -> str:  # noqa: ARG001
        return ""

    with caplog.at_level(logging.WARNING, logger="goldfive"):
        goldfive.wrap(
            _noop_agent,
            call_llm=explicit_call_llm,
            sinks=[],
            llm_detector=_scripted_detector("should-not-appear"),
        )

    assert not [
        r for r in caplog.records if "judge LLM not explicitly configured" in r.getMessage()
    ]
