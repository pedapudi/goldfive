"""Tests for correction injection (goldfive#251 Stream D).

Covers the write-side machinery in :mod:`goldfive._correction_injection`
and its integration with:

* :mod:`goldfive.steerer` — ``_emit_plan_revised`` queues corrections
  for every CORRECT-kind supersedes and GC's corrections for tasks
  superseded by the revision.
* :mod:`goldfive.reporting` — ``_handle_task_started`` clears the
  correction once the agent acknowledges the new task.
* :mod:`goldfive.adapters.adk_llm_instrumentation` —
  :func:`format_correction_block` renders the dict into directive,
  NOT diagnostic, prompt text; the dynamic resolver picks the rendered
  block up via the ``(agent_name, task_id)``-keyed state entry.
* :mod:`goldfive.adapters._adk_plugin` — the bridge propagates every
  ``goldfive.pending_corrections.*`` key from orchestration state onto
  ADK session.state every invocation.
* :mod:`goldfive.adapters._adk_state_protocol` — cancellation does
  NOT perturb queued corrections (Streams C + D are orthogonal by
  design).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive._correction_injection import (  # noqa: E402
    build_correction_payload,
    clear_correction,
    clear_corrections_for_task,
    clear_obsolete_corrections_on_revision,
    is_pending_correction_key,
    pending_correction_key,
    queue_corrections_for_revision,
    write_correction,
)
from goldfive.adapters import _adk_state_protocol as _sp  # noqa: E402
from goldfive.reporting import BUILTIN_REPORTING_TOOLS  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    CancellationRequest,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    SupersessionKind,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class _StubPlanner:
    def __init__(self, revised: Plan | None = None) -> None:
        self.revised = revised

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return self.revised


def _tool(name: str):
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"builtin tool {name!r} missing")


def _session(plan: Plan) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="drive solar research to a report")],
        plan=plan,
    )


def _drift(kind: DriftKind = DriftKind.OFF_TOPIC, detail: str = "drifted off topic") -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.WARNING,
        detail=detail,
        current_task_id="research_solar",
    )


def _base_plan() -> Plan:
    """Seed plan with one COMPLETED research task that will be CORRECTED."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="write_report")],
        revision_index=0,
    )


def _revised_with_one_correct() -> Plan:
    """Refine output: research_solar COMPLETED; corrected task is a CORRECT child."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected scope)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="write_report")],
        revision_index=1,
    )


async def _emit(
    steerer: DefaultSteerer, session: Session, revised: Plan, drift: DriftEvent
) -> None:
    prev = session.plan
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer._apply_revision(session, revised, drift)
    await steerer._emit_plan_revised(session, revised, drift, prev_plan=prev)


# ---------------------------------------------------------------------------
# 1. CORRECT-kind refine writes a correction to state.
# ---------------------------------------------------------------------------


async def test_correct_kind_supersedes_writes_correction_to_state() -> None:
    session = _session(_base_plan())
    revised = _revised_with_one_correct()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())

    drift = _drift(kind=DriftKind.OFF_TOPIC, detail="veered to batteries")
    await _emit(steerer, session, revised, drift)

    key = pending_correction_key("research_agent", "research_solar_corrected")
    assert key in session.state
    payload = session.state[key]
    assert isinstance(payload, dict)
    assert payload["agent_name"] == "research_agent"
    assert payload["task_id"] == "research_solar_corrected"
    assert payload["superseded_task_id"] == "research_solar"
    assert payload["superseded_task_title"] == "Research solar options"
    assert payload["drift_kind"] == "off_topic"
    assert payload["drift_reason"] == "veered to batteries"
    assert payload["revision_number"] >= 1
    assert isinstance(payload["issued_at_ms"], int)
    assert payload["issued_at_ms"] > 0


# ---------------------------------------------------------------------------
# 2. REPLACE-kind refine does NOT write a correction.
# ---------------------------------------------------------------------------


async def test_replace_kind_supersedes_does_not_write_correction() -> None:
    # Seed plan: research_solar is PENDING (not COMPLETED) — the usual
    # REPLACE scenario.
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
            ),
        ],
        edges=[],
        revision_index=0,
    )
    session = _session(plan)

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar_v2",
                title="Research solar options (rescoped)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
        revision_index=1,
    )

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())
    await _emit(steerer, session, revised, _drift())

    # No pending_corrections.* keys should exist.
    correction_keys = [k for k in session.state if is_pending_correction_key(k)]
    assert correction_keys == []


# ---------------------------------------------------------------------------
# 3. Multiple CORRECT supersedes in one refine -> multiple corrections.
# ---------------------------------------------------------------------------


async def test_multiple_correct_supersedes_write_multiple_corrections() -> None:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="analyze_solar",
                title="Analyze solar findings",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="analysis_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="analyze_solar")],
        revision_index=0,
    )
    session = _session(plan)

    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="analyze_solar",
                title="Analyze solar findings",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="analysis_agent",
            ),
            Task(
                id="analyze_solar_corrected",
                title="Analyze solar findings (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="analysis_agent",
                supersedes="analyze_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="analyze_solar")],
        revision_index=1,
    )

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())
    await _emit(steerer, session, revised, _drift())

    k1 = pending_correction_key("research_agent", "research_solar_corrected")
    k2 = pending_correction_key("analysis_agent", "analyze_solar_corrected")
    assert k1 in session.state
    assert k2 in session.state
    assert session.state[k1]["superseded_task_id"] == "research_solar"
    assert session.state[k2]["superseded_task_id"] == "analyze_solar"


# ---------------------------------------------------------------------------
# 4. Correction cleared on report_task_started.
# ---------------------------------------------------------------------------


async def test_correction_cleared_on_report_task_started() -> None:
    plan = _revised_with_one_correct()
    session = _session(plan)
    # Pre-seed a correction as Stream D would have written it.
    key = pending_correction_key("research_agent", "research_solar_corrected")
    session.state[key] = {
        "agent_name": "research_agent",
        "task_id": "research_solar_corrected",
        "superseded_task_id": "research_solar",
        "superseded_task_title": "Research solar options",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 1,
        "issued_at_ms": 123456789,
    }
    assert key in session.state

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())

    await _tool("report_task_started").handler(
        {"task_id": "research_solar_corrected", "detail": "starting"},
        session,
        steerer,
    )
    assert key not in session.state


# ---------------------------------------------------------------------------
# 5. Correction NOT cleared on report_task_failed.
# ---------------------------------------------------------------------------


async def test_correction_not_cleared_on_report_task_failed() -> None:
    plan = _revised_with_one_correct()
    # Mark the correction task RUNNING so failed is a legal transition.
    # goldfive#247: Plan + Task are frozen — derive a new plan via
    # with_task_status.
    from goldfive.types import with_task_status as _wts

    plan = _wts(plan, "research_solar_corrected", TaskStatus.RUNNING)
    session = _session(plan)
    key = pending_correction_key("research_agent", "research_solar_corrected")
    session.state[key] = {
        "agent_name": "research_agent",
        "task_id": "research_solar_corrected",
        "superseded_task_id": "research_solar",
        "superseded_task_title": "Research solar options",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 1,
        "issued_at_ms": 123456789,
    }

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())

    await _tool("report_task_failed").handler(
        {"task_id": "research_solar_corrected", "reason": "dead end"},
        session,
        steerer,
    )
    assert key in session.state, "failure is not acknowledgment — correction must persist"


# ---------------------------------------------------------------------------
# 6. Correction cleared when the correction task itself is superseded.
# ---------------------------------------------------------------------------


async def test_correction_cleared_when_correction_task_itself_superseded() -> None:
    # Seed a plan where research_solar_corrected already exists with a
    # queued correction; then a second revision supersedes _corrected
    # with _corrected_v2.
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="research_solar_corrected")],
        revision_index=1,
    )
    session = _session(plan)
    stale_key = pending_correction_key("research_agent", "research_solar_corrected")
    session.state[stale_key] = {
        "agent_name": "research_agent",
        "task_id": "research_solar_corrected",
        "superseded_task_id": "research_solar",
        "superseded_task_title": "Research solar",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 1,
        "issued_at_ms": 100,
    }

    # Revision 2: correction task itself is REPLACEd.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected_v2",
                title="Research solar (corrected v2)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar_corrected",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research_solar", to_task_id="research_solar_corrected_v2")
        ],
        revision_index=2,
    )

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())
    await _emit(steerer, session, revised, _drift())

    assert stale_key not in session.state, (
        "correction for the task that was just superseded should be GC'd"
    )


# ---------------------------------------------------------------------------
# 7. Correction survives a Stream C cancellation.
# ---------------------------------------------------------------------------


def test_correction_survives_stream_c_cancellation() -> None:
    """Cancellation of the in-flight invocation does not evict a queued correction."""
    plan = _revised_with_one_correct()
    session = _session(plan)

    corr_key = pending_correction_key("research_agent", "research_solar_corrected")
    session.state[corr_key] = {
        "agent_name": "research_agent",
        "task_id": "research_solar_corrected",
        "superseded_task_id": "research_solar",
        "superseded_task_title": "Research solar options",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 1,
        "issued_at_ms": 100,
    }

    # Simulate Stream C cancel writing + consuming a request.
    _sp.write_cancel_request(
        session.state,
        invocation_id="inv-1",
        request=CancellationRequest(
            invocation_id="inv-1",
            reason="drift",
            severity=DriftSeverity.CRITICAL,
            drift_id="did-1",
            drift_kind=DriftKind.OFF_TOPIC.value,
            detail="CRITICAL off-topic",
        ),
    )
    consumed = _sp.consume_cancel_request(session.state, "inv-1")
    assert consumed is not None

    # Correction must still be present — cancellation and correction
    # are orthogonal per-axis concerns.
    assert corr_key in session.state


# ---------------------------------------------------------------------------
# 8. Format helper language is directive, not diagnostic.
# ---------------------------------------------------------------------------


def test_format_correction_block_is_directive_not_diagnostic() -> None:
    from goldfive.adapters.adk_llm_instrumentation import format_correction_block

    block = format_correction_block(
        {
            "agent_name": "research_agent",
            "task_id": "T_corrected",
            "superseded_task_id": "T_old",
            "superseded_task_title": "Research solar options",
            "drift_kind": "off_topic",
            "drift_reason": "the agent veered into batteries",
            "revision_number": 3,
            "issued_at_ms": 123,
        }
    )
    # Directive language: present.
    assert "Focus only on" in block
    assert "Do not propagate" in block

    # Diagnostic / problem-naming language: MUST be absent so the LLM
    # doesn't pattern-match on failure-shape prompts (#250/#252/#253/#259).
    forbidden = [
        "was broken",
        "broke",
        "failed",
        "error",
        "mistake",
        "incorrect",
        "wrong",
    ]
    low = block.lower()
    for word in forbidden:
        assert word not in low, f"format_correction_block leaked diagnostic word {word!r}"

    # Structural expectations on the rendered block.
    assert "REV 3" in block
    assert "Research solar options" in block  # superseded task title is referenced
    # The triggering drift detail is NOT interpolated into the LLM-facing block
    # (it is retained in the dict for sinks / observability).
    assert "veered into batteries" not in block
    assert "off_topic" not in block


# ---------------------------------------------------------------------------
# 9. Integration with dynamic resolver.
# ---------------------------------------------------------------------------


def test_dynamic_resolver_picks_up_correction_dict() -> None:
    pytest.importorskip("google.adk")
    # Wave B1 (refactor/prompt-shaper): the factory moved to
    # :class:`~goldfive.prompt_shaper.PromptShaper`.
    from goldfive.prompt_shaper import PromptShaper

    resolver = PromptShaper().make_dynamic_instruction(
        original_instruction="you are the researcher",
        agent_name="research_agent",
    )

    class _Ctx:
        def __init__(self, state: dict[str, Any]) -> None:
            self.state = state

    state: dict[str, Any] = {
        _sp.KEY_CURRENT_TASK_ID: "research_solar_corrected",
        _sp.KEY_CURRENT_TASK_TITLE: "Research solar options (corrected)",
        _sp.KEY_CURRENT_TASK_DESCRIPTION: "narrowed scope",
        pending_correction_key("research_agent", "research_solar_corrected"): {
            "agent_name": "research_agent",
            "task_id": "research_solar_corrected",
            "superseded_task_id": "research_solar",
            "superseded_task_title": "Research solar options",
            "drift_kind": "off_topic",
            "drift_reason": "veered",
            "revision_number": 2,
            "issued_at_ms": 123,
        },
    }
    out = resolver(_Ctx(state))
    assert "you are the researcher" in out
    assert "Current assigned task:" in out
    # Directive correction block:
    assert "Focus only on" in out
    assert "REV 2" in out


# ---------------------------------------------------------------------------
# 10. StateStore reads pending corrections directly.
# ---------------------------------------------------------------------------


def test_orchestration_store_reads_pending_correction_off_goldfive_session() -> None:
    """Phase 2.0 of goldfive#271 — bridge eliminated.

    Pending corrections live on goldfive ``Session.state`` (written by
    :func:`write_correction`). The dynamic-instruction resolver reads
    them directly via
    :meth:`goldfive.state_store.StateStore.get_correction`;
    no copy onto ADK ``session.state`` is needed.
    """
    from goldfive.state_store import StateStore
    from goldfive.types import Session

    session = Session(run_id="r1")
    corr_a = pending_correction_key("agent_a", "task_1")
    corr_b = pending_correction_key("agent_b", "task_2")
    session.state[corr_a] = {
        "agent_name": "agent_a",
        "task_id": "task_1",
        "revision_number": 1,
    }
    session.state[corr_b] = {
        "agent_name": "agent_b",
        "task_id": "task_2",
        "revision_number": 1,
    }

    store = StateStore.for_session(session)

    # Each (agent, task) returns its own correction; no cross-leakage.
    assert store.get_correction("agent_a", "task_1") == session.state[corr_a]
    assert store.get_correction("agent_b", "task_2") == session.state[corr_b]
    assert store.get_correction("agent_a", "task_2") is None
    assert store.get_correction("agent_b", "task_1") is None

    # Clearing the gf-side entry makes the read return None — the
    # mechanism by which a correction-clear reaches the resolver.
    del session.state[corr_a]
    assert store.get_correction("agent_a", "task_1") is None
    assert store.get_correction("agent_b", "task_2") is not None


# ---------------------------------------------------------------------------
# Unit tests for the pure helpers.
# ---------------------------------------------------------------------------


def test_build_correction_payload_shapes_fields() -> None:
    old = Task(id="T_old", title="Old work")
    new = Task(
        id="T_new",
        title="New work (corrected)",
        assignee_agent_id="agent_x",
        supersedes="T_old",
        supersedes_kind=SupersessionKind.CORRECT,
    )
    d = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="veered",
    )
    payload = build_correction_payload(
        new_task=new, old_task=old, drift=d, revision_number=5, issued_at_ms=42
    )
    assert payload == {
        "agent_name": "agent_x",
        "task_id": "T_new",
        "superseded_task_id": "T_old",
        "superseded_task_title": "Old work",
        "drift_kind": "off_topic",
        "drift_reason": "veered",
        "revision_number": 5,
        "issued_at_ms": 42,
    }


def test_write_correction_round_trips() -> None:
    session = _session(_base_plan())
    key = write_correction(
        session,
        {
            "agent_name": "a1",
            "task_id": "t1",
            "superseded_task_id": "t0",
            "superseded_task_title": "old",
            "drift_kind": "off_topic",
            "drift_reason": "veered",
            "revision_number": 1,
            "issued_at_ms": 7,
        },
    )
    assert key == pending_correction_key("a1", "t1")
    assert session.state[key]["task_id"] == "t1"


def test_write_correction_rejects_when_agent_or_task_missing() -> None:
    session = _session(_base_plan())
    assert write_correction(session, {"agent_name": "a1", "task_id": ""}) is None
    assert write_correction(session, {"agent_name": "", "task_id": "t1"}) is None
    assert not any(is_pending_correction_key(k) for k in session.state)


def test_clear_correction_removes_entry() -> None:
    session = _session(_base_plan())
    key = pending_correction_key("a1", "t1")
    session.state[key] = {"agent_name": "a1", "task_id": "t1"}
    assert clear_correction(session, agent_name="a1", task_id="t1") is True
    assert key not in session.state
    # Second call — idempotent no-op.
    assert clear_correction(session, agent_name="a1", task_id="t1") is False


def test_clear_corrections_for_task_sweeps_all_agents() -> None:
    session = _session(_base_plan())
    session.state[pending_correction_key("a1", "t1")] = {"agent_name": "a1", "task_id": "t1"}
    session.state[pending_correction_key("a2", "t1")] = {"agent_name": "a2", "task_id": "t1"}
    session.state[pending_correction_key("a1", "t2")] = {"agent_name": "a1", "task_id": "t2"}
    cleared = clear_corrections_for_task(session, "t1")
    assert sorted(cleared) == sorted(
        [
            pending_correction_key("a1", "t1"),
            pending_correction_key("a2", "t1"),
        ]
    )
    assert pending_correction_key("a1", "t2") in session.state


def test_clear_obsolete_on_revision_drops_superseded_task_corrections() -> None:
    session = _session(_base_plan())
    session.state[pending_correction_key("agent_a", "t_old")] = {
        "agent_name": "agent_a",
        "task_id": "t_old",
    }
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_old", title="old", assignee_agent_id="agent_a"),
            Task(
                id="t_new",
                title="new",
                assignee_agent_id="agent_a",
                supersedes="t_old",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
        revision_index=2,
    )
    cleared = clear_obsolete_corrections_on_revision(session, revised)
    assert cleared == [pending_correction_key("agent_a", "t_old")]


def test_queue_corrections_skips_task_without_assignee() -> None:
    """No assignee -> correction is silently skipped (no key can be formed)."""
    session = _session(_base_plan())
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t_old", title="old", status=TaskStatus.COMPLETED),
            Task(
                id="t_new",
                title="new",
                # No assignee_agent_id
                supersedes="t_old",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    keys = queue_corrections_for_revision(
        session=session,
        revised=revised,
        prev_plan=None,
        drift=_drift(),
    )
    assert keys == []
    assert not any(is_pending_correction_key(k) for k in session.state)


def test_pending_correction_key_prefix_is_stable() -> None:
    """The key prefix is part of the state-bridge contract."""
    assert pending_correction_key("a", "t") == "goldfive.pending_corrections.a.t"
    assert is_pending_correction_key("goldfive.pending_corrections.a.t")
    assert not is_pending_correction_key("goldfive.current_task_id")
