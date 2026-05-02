"""Fixture-driven classification-table tests for the iter-10 reasoning judge.

Per the iter-10 design doc §8: each trace exercises one of the three
states (on_task / justified_deviation / erroneous_deviation) with a
deterministic mock ``call_llm`` keyed by trace id. The point is NOT to
test the LLM — it's to lock in the prompt-shape expectation and the
parser's reading of representative responses.

Three traces:

* Trace A — on-task reasoning. Mock returns
  ``{"classification": "on_task", ...}``. Asserts no drift.
* Trace B — provoked deviation with a matching tool observation in
  ``recent_tool_observations``. Mock returns
  ``{"classification": "justified_deviation", "provenance": "tool_error", ...}``.
  Asserts ``drift.kind == DriftKind.JUSTIFIED_DEVIATION`` and the
  provenance is preserved on the verdict.
* Trace C — unprovoked deviation with no relevant tool observations.
  Mock returns
  ``{"classification": "erroneous_deviation", "severity": "warning", ...}``.
  Asserts ``drift.kind == DriftKind.OFF_TOPIC``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift.reasoning_judge import (  # noqa: E402
    classify_reasoning_drift_with_focus,
)
from goldfive.types import (  # noqa: E402
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
)


@dataclasses.dataclass(frozen=True)
class _Trace:
    """One trace + the canned mock-LLM response that should classify it."""

    trace_id: str
    reasoning: str
    recent_tool_observations: list[dict[str, Any]]
    canned_response: dict[str, Any]
    expected_drift_kind: DriftKind | None
    expected_classification: str
    expected_provenance: str


def _task() -> Task:
    return Task(
        id="t_post",
        title="Draft a memo on solar panel ROI",
        description="One-page memo with citations",
    )


def _goals() -> list[Goal]:
    return [Goal(id="g1", summary="Publish a one-page memo on solar panel ROI")]


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[_task()],
        edges=[],
    )


_TRACES: list[_Trace] = [
    _Trace(
        trace_id="A_on_task",
        reasoning=(
            "I should look up commercial-rooftop solar panel efficiency "
            "ratings and pull together a quick comparison for the memo."
        ),
        recent_tool_observations=[],
        canned_response={
            "classification": "on_task",
            "reason": "researching the bound task",
            "focused_task_id": "t_post",
            "focus_confidence": 0.9,
            "stated_intent": "researching solar efficiency for the memo",
        },
        expected_drift_kind=None,
        expected_classification="on_task",
        expected_provenance="",
    ),
    _Trace(
        trace_id="B_justified_deviation_tool_error",
        reasoning=(
            "The pricing API returned 503 again; I'll fall back to "
            "scraping the public datasheet PDF instead so the memo "
            "still has a citation."
        ),
        recent_tool_observations=[
            {
                "ts_ms": 1000,
                "agent_name": "researcher",
                "task_id": "t_post",
                "tool_name": "fetch_pricing_api",
                "args_preview": '{"sku":"sp-100"}',
                "result_preview": '{"error":"503 Service Unavailable"}',
                "is_error": True,
                "error_message": "503 Service Unavailable",
            }
        ],
        canned_response={
            "classification": "justified_deviation",
            "severity": "warning",
            "reason": (
                "agent pivoted to a fallback source after a real tool "
                "failure recorded in the recent tool observations"
            ),
            "provenance": "tool_error",
            "focused_task_id": "t_post",
            "focus_confidence": 0.85,
            "stated_intent": "scraping the datasheet as a fallback citation",
        },
        expected_drift_kind=DriftKind.JUSTIFIED_DEVIATION,
        expected_classification="justified_deviation",
        expected_provenance="tool_error",
    ),
    _Trace(
        trace_id="C_erroneous_deviation",
        reasoning=(
            "Actually, raccoons are surprisingly clever animals — I "
            "should look up some raccoon facts before continuing."
        ),
        recent_tool_observations=[],  # No provoking tool signal.
        canned_response={
            "classification": "erroneous_deviation",
            "severity": "warning",
            "reason": (
                "the reasoning pivoted to raccoons with no provoking "
                "signal in the prompt context"
            ),
            "focused_task_id": "",
            "focus_confidence": 0.0,
            "stated_intent": "looking up raccoon facts",
        },
        expected_drift_kind=DriftKind.OFF_TOPIC,
        expected_classification="erroneous_deviation",
        expected_provenance="",
    ),
]


def _make_call_llm(canned: dict[str, Any]):
    """Async ``CallLLM``-shaped stub that always returns the same JSON.

    The stub also records what it saw in ``calls`` so the test can
    assert the prompt actually contained the provoking tool
    observation (for trace B).
    """

    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        return json.dumps(canned)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


@pytest.mark.parametrize(
    "trace",
    _TRACES,
    ids=[t.trace_id for t in _TRACES],
)
async def test_classification_table_round_trip(trace: _Trace) -> None:
    """Each trace round-trips through the parser to its expected drift shape."""
    call_llm = _make_call_llm(trace.canned_response)
    verdict = await classify_reasoning_drift_with_focus(
        reasoning=trace.reasoning,
        task=_task(),
        goals=_goals(),
        plan=_plan(),
        model="fake",
        call_llm=call_llm,
        current_task_id="t_post",
        current_agent_id="researcher",
        recent_tool_observations=trace.recent_tool_observations,
        task_lineage={"t_post": {"researcher"}},
    )
    if trace.expected_drift_kind is None:
        assert verdict.drift is None, trace.trace_id
    else:
        assert verdict.drift is not None, trace.trace_id
        assert verdict.drift.kind is trace.expected_drift_kind, (
            trace.trace_id,
            verdict.drift.kind,
        )
        # Severity is WARNING in every non-on_task trace.
        assert verdict.drift.severity is DriftSeverity.WARNING, trace.trace_id
    assert verdict.classification == trace.expected_classification, trace.trace_id
    assert verdict.provenance == trace.expected_provenance, trace.trace_id


async def test_trace_b_prompt_includes_tool_observation() -> None:
    """Trace B's user prompt carries the provoking tool error.

    Pins the prompt-rendering wiring: the recent_tool_observations
    actually make it into the LLM's context window for the judge to
    use as evidence.
    """
    trace = _TRACES[1]  # B
    call_llm = _make_call_llm(trace.canned_response)
    await classify_reasoning_drift_with_focus(
        reasoning=trace.reasoning,
        task=_task(),
        goals=_goals(),
        plan=_plan(),
        model="fake",
        call_llm=call_llm,
        current_task_id="t_post",
        current_agent_id="researcher",
        recent_tool_observations=trace.recent_tool_observations,
        task_lineage={"t_post": {"researcher"}},
    )
    assert len(call_llm.calls) == 1
    _system, user, _model = call_llm.calls[0]
    assert "RECENT TOOL OBSERVATIONS" in user
    assert "fetch_pricing_api" in user
    assert "ERROR" in user
    assert "503" in user


async def test_trace_a_prompt_includes_empty_tool_obs_marker() -> None:
    """Trace A has no tool observations → empty-marker placeholder.

    The prompt shape is invariant — the section header is always
    present and the body is the canonical "(no recent tool
    observations)" placeholder when nothing has been observed.
    """
    trace = _TRACES[0]  # A
    call_llm = _make_call_llm(trace.canned_response)
    await classify_reasoning_drift_with_focus(
        reasoning=trace.reasoning,
        task=_task(),
        goals=_goals(),
        plan=_plan(),
        model="fake",
        call_llm=call_llm,
        current_task_id="t_post",
        current_agent_id="researcher",
        recent_tool_observations=trace.recent_tool_observations,
    )
    _system, user, _model = call_llm.calls[0]
    assert "RECENT TOOL OBSERVATIONS" in user
    assert "(no recent tool observations)" in user
