"""Scaffolding tests for the iter-11E structural artifact-verification
mechanism (see the iter-11E PR-1 brief).

PR 1 ships *pure scaffolding* ahead of the verification logic in PR 2:

* ``DriftKind.INCOMPLETE_TOOL_CALLS`` is added (StrEnum + proto enum).
* ``Task.required_tool_calls`` is added (Python dataclass + proto
  field). Default empty list — back-compat with every existing plan.
* ``DefaultSteerer._LADDER`` gets an entry mirroring PLAN_DIVERGENCE /
  OFF_TOPIC routing semantics, with PAUSE_ESCALATE on repeat-CRITICAL
  because repeated false completion is a serious agent-correctness
  issue.
* ``LLMPlanner.refine`` extends its prompt-selection branch so the new
  drift kind takes the goal-aware ABSORB/REJECT path
  (``_PLAN_DIVERGENCE_SYSTEM_PROMPT``).

The tests pin those scaffolding properties only — the behavioural
contract (verification at ``report_task_succeeded`` time) is exercised
by PR 2 tests.
"""

from __future__ import annotations

import json

import pytest

from goldfive.planner import LLMPlanner
from goldfive.steerer import DefaultSteerer, InterventionLevel
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)
from tests._pbsetup import ensure_pb_available

# ---------------------------------------------------------------------------
# Proto round-trip: DriftKind.INCOMPLETE_TOOL_CALLS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)
def test_drift_kind_incomplete_tool_calls_round_trip() -> None:
    """The new DriftKind enum value round-trips through proto.

    Pins the wire contract: a stable integer wire value (41 by design,
    next free slot after JUSTIFIED_DEVIATION=40) plus name resolution
    via the descriptor. If a future ``make proto`` reshuffles enum
    numbers this test catches it before harmonograf consumers do.
    Also exercises the ``DriftDetected`` envelope to confirm the new
    kind serialises losslessly on the wire path PR 2 will use.
    """
    from goldfive.pb.goldfive.v1 import events_pb2, types_pb2  # type: ignore[import]

    # Symbol is exported and lives at the expected enum slot.
    assert hasattr(types_pb2, "DRIFT_KIND_INCOMPLETE_TOOL_CALLS")
    proto_value = types_pb2.DRIFT_KIND_INCOMPLETE_TOOL_CALLS
    assert isinstance(proto_value, int)
    assert proto_value == 41

    # Name <-> value resolution via the descriptor matches.
    enum_descriptor = types_pb2.DriftKind.DESCRIPTOR
    by_name = enum_descriptor.values_by_name["DRIFT_KIND_INCOMPLETE_TOOL_CALLS"]
    by_number = enum_descriptor.values_by_number[proto_value]
    assert by_name.number == 41
    assert by_number.name == "DRIFT_KIND_INCOMPLETE_TOOL_CALLS"

    # Python ``DriftKind`` StrEnum exposes the new member with the
    # expected string value, and is distinct from the kinds whose
    # ladder/refine semantics it shares.
    assert hasattr(DriftKind, "INCOMPLETE_TOOL_CALLS")
    assert DriftKind.INCOMPLETE_TOOL_CALLS.value == "incomplete_tool_calls"
    assert DriftKind.INCOMPLETE_TOOL_CALLS is not DriftKind.OFF_TOPIC
    assert DriftKind.INCOMPLETE_TOOL_CALLS is not DriftKind.PLAN_DIVERGENCE

    # Wire round-trip via DriftDetected — the path PR 2 will emit on.
    src = events_pb2.DriftDetected(
        kind=types_pb2.DRIFT_KIND_INCOMPLETE_TOOL_CALLS,
        severity=types_pb2.DRIFT_SEVERITY_WARNING,
        detail="missing required tool calls: ['write_webpage_tool']",
        current_task_id="draft_slides",
        current_agent_id="coordinator",
        id="drift-1",
    )
    dst = events_pb2.DriftDetected()
    dst.ParseFromString(src.SerializeToString())
    assert dst.kind == types_pb2.DRIFT_KIND_INCOMPLETE_TOOL_CALLS
    assert dst.detail == "missing required tool calls: ['write_webpage_tool']"


# ---------------------------------------------------------------------------
# Proto round-trip: Task.required_tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)
def test_task_required_tool_calls_round_trip() -> None:
    """A populated ``required_tool_calls`` round-trips through the proto.

    Exercises both the dataclass converter (``to_pb_task`` /
    ``from_pb_task`` in :mod:`goldfive.conv`) and the wire path: the
    field must arrive on the other side as a Python list of strings,
    not as a char-iterated string or anything else.
    """
    from goldfive.conv import from_pb_task, to_pb_task

    t = Task(
        id="draft_slides",
        title="Draft the slide deck",
        description="Write slide content via write_webpage_tool",
        assignee_agent_id="presenter",
        status=TaskStatus.PENDING,
        required_tool_calls=["write_webpage_tool", "save_artifact"],
    )
    pb = to_pb_task(t)

    # Pb side surfaces the field as a repeated string container; the
    # values are str instances, not bytes, not char-split.
    assert list(pb.required_tool_calls) == ["write_webpage_tool", "save_artifact"]
    for v in pb.required_tool_calls:
        assert isinstance(v, str)

    recovered = from_pb_task(pb)
    assert recovered == t
    assert isinstance(recovered.required_tool_calls, list)
    assert recovered.required_tool_calls == ["write_webpage_tool", "save_artifact"]


@pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)
def test_task_required_tool_calls_default_empty() -> None:
    """A default-constructed Task has empty ``required_tool_calls``.

    Back-compat invariant: every legacy Task / plan must continue to
    construct without specifying the new field, and the PR 2
    verification logic must read empty-list as "no requirement".
    Round-tripping the empty case must also preserve the empty-list
    semantics (not surface as ``None`` or any other sentinel).
    """
    from goldfive.conv import from_pb_task, to_pb_task

    t = Task(id="t1", title="A")
    # Field exists, defaults to a fresh empty list (default_factory,
    # not a shared mutable default).
    assert t.required_tool_calls == []
    assert isinstance(t.required_tool_calls, list)

    # Two default-constructed Tasks must have *independent* empty lists
    # — appending to one must not mutate the other (pins the
    # ``default_factory`` shape).
    t2 = Task(id="t2", title="B")
    t.required_tool_calls.append("write_webpage_tool")
    assert t2.required_tool_calls == []

    # Round-trip preserves the empty default.
    fresh = Task(id="t3", title="C")
    recovered = from_pb_task(to_pb_task(fresh))
    assert recovered == fresh
    assert recovered.required_tool_calls == []


# ---------------------------------------------------------------------------
# Steerer ladder entry for INCOMPLETE_TOOL_CALLS
# ---------------------------------------------------------------------------


_LADDER_CASES = [
    # INFO -> OBSERVE (record-only; no plan mutation).
    (
        DriftKind.INCOMPLETE_TOOL_CALLS,
        DriftSeverity.INFO,
        0,
        InterventionLevel.OBSERVE,
    ),
    # WARNING -> ABSORB (refine via the goal-aware prompt).
    (
        DriftKind.INCOMPLETE_TOOL_CALLS,
        DriftSeverity.WARNING,
        0,
        InterventionLevel.ABSORB,
    ),
    # CRITICAL first -> CANCEL_REINVOKE (cancel the in-flight invocation
    # and re-invoke after refine; mirrors PLAN_DIVERGENCE / OFF_TOPIC).
    (
        DriftKind.INCOMPLETE_TOOL_CALLS,
        DriftSeverity.CRITICAL,
        0,
        InterventionLevel.CANCEL_REINVOKE,
    ),
    # CRITICAL repeat -> PAUSE_ESCALATE (repeated false completion is a
    # serious agent-correctness issue the planner alone can't fix).
    (
        DriftKind.INCOMPLETE_TOOL_CALLS,
        DriftSeverity.CRITICAL,
        2,
        InterventionLevel.PAUSE_ESCALATE,
    ),
]


@pytest.mark.parametrize(
    "kind,severity,occurrence,expected",
    _LADDER_CASES,
    ids=[f"{k.value}-{s.value}-occ{o}-{exp.name}" for k, s, o, exp in _LADDER_CASES],
)
def test_ladder_entry_for_incomplete_tool_calls(
    kind: DriftKind,
    severity: DriftSeverity,
    occurrence: int,
    expected: InterventionLevel,
) -> None:
    """``DefaultSteerer._ladder_level_for`` honours the new entry.

    Mirrors PLAN_DIVERGENCE / OFF_TOPIC routing semantics: ABSORB at
    WARNING (so the goal-aware refine engages), CANCEL_REINVOKE at
    CRITICAL-first, PAUSE_ESCALATE on repeat. Pins the shape so a
    future ladder reshuffle can't silently downgrade the response.
    """
    steerer = DefaultSteerer()
    level = steerer._ladder_level_for(kind, severity, occurrence)
    assert level is expected, (
        f"ladder({kind.value}, {severity.value}, occ={occurrence}) "
        f"= {level.name}, expected {expected.name}"
    )


# ---------------------------------------------------------------------------
# Planner.refine prompt-selection for INCOMPLETE_TOOL_CALLS
# ---------------------------------------------------------------------------


def _goals() -> list[Goal]:
    return [
        Goal(id="g1", summary="Produce a slide deck about goldfish."),
    ]


def _running_plan() -> Plan:
    return Plan(
        id="plan-1",
        run_id="run-1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="draft_slides",
                title="Draft the slide deck",
                assignee_agent_id="presenter",
                status=TaskStatus.RUNNING,
                required_tool_calls=["write_webpage_tool"],
            ),
            Task(
                id="review",
                title="Review the slide deck",
                assignee_agent_id="editor",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="draft_slides", to_task_id="review")],
        summary="Draft and review the slide deck.",
        revision_index=0,
    )


def _absorbed_revision_json() -> str:
    """A legitimate ABSORB revision: redo the drafting step."""
    return json.dumps(
        {
            "summary": "Redo the draft via write_webpage_tool, then review.",
            "tasks": [
                {
                    "id": "draft_slides",
                    "title": "Draft the slide deck",
                    "assignee_agent_id": "presenter",
                    "status": "RUNNING",
                },
                {
                    "id": "review",
                    "title": "Review the slide deck",
                    "assignee_agent_id": "editor",
                    "status": "PENDING",
                },
            ],
            "edges": [{"from_task_id": "draft_slides", "to_task_id": "review"}],
        }
    )


class _StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, system: str, user: str, model: str) -> str:
        self.calls.append((system, user, model))
        return self.response


async def test_planner_refine_uses_divergence_prompt_for_incomplete_tool_calls() -> None:
    """INCOMPLETE_TOOL_CALLS routes to the goal-aware refine prompt.

    The verification logic in PR 2 will emit this kind with a
    ``detail`` carrying the list of missing tools. The planner's
    ABSORB path must use the goal-aware
    ``_PLAN_DIVERGENCE_SYSTEM_PROMPT`` so the LLM either revises the
    plan to address the unfinished work or emits the
    ``{"reject": true, ...}`` sentinel — NOT the generic
    ``_REFINE_SYSTEM_PROMPT`` which has no goal-alignment guidance and
    could absorb a vacuous "claim done" into the plan.
    """
    stub = _StubLLM(_absorbed_revision_json())
    planner = LLMPlanner(call_llm=stub)
    drift = DriftEvent(
        kind=DriftKind.INCOMPLETE_TOOL_CALLS,
        severity=DriftSeverity.WARNING,
        detail="missing required tool calls: ['write_webpage_tool']",
        current_task_id="draft_slides",
        current_agent_id="presenter",
    )

    revised = await planner.refine(plan=_running_plan(), drift=drift, goals=_goals())

    assert revised is not None
    assert len(stub.calls) == 1
    system, user_prompt, _model = stub.calls[0]
    # The divergence (goal-aware) system prompt was selected, NOT the
    # generic refine prompt — same structural markers as the OFF_TOPIC
    # / JUSTIFIED_DEVIATION sibling tests:
    assert "ABSORB" in system
    assert "REJECT" in system
    # The user prompt carries the OFF-TOPIC reasoning context block —
    # PR 1 reuses ``_render_off_topic_reasoning_block`` for the new
    # kind, with the missing-tools detail surfacing as the judge
    # reason. PR 2 may extend this with a dedicated rendering, but
    # the goal-aware decision shape is in place today.
    assert "OFF-TOPIC REASONING" in user_prompt
    assert "missing required tool calls" in user_prompt
    # No OBSERVED AGENT ACTIVITY header — that's the
    # PLAN_DIVERGENCE+observed_actions channel; INCOMPLETE_TOOL_CALLS
    # has no observed-actions input.
    assert "OBSERVED AGENT ACTIVITY" not in user_prompt
