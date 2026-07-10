"""Unit tests for goldfive.events helpers.

Focused on :func:`goldfive.events.build_plan_revision_diff`, the helper
that powers the ``PlanRevisionDiff`` sidecar attached to every
``PlanRevised`` event (closes PLAN-LIFECYCLE.md §8 gap #4). The helper
is deliberately small and pure so these tests pin its identity rules
(PLAN-LIFECYCLE.md §3.0) in isolation — full end-to-end wiring is
exercised in ``tests/test_steerer.py``.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.events import (  # noqa: E402
    build_plan_revision_diff,
    control_received_event,
    delegation_observed_event,
    drift_detected_event,
    invocation_cancelled_event,
    plan_revised_event,
)
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)


def _plan(
    tasks: list[Task] | None = None,
    edges: list[TaskEdge] | None = None,
    *,
    plan_id: str = "p1",
    run_id: str = "r1",
) -> Plan:
    return Plan(
        id=plan_id,
        run_id=run_id,
        goal_ids=["g1"],
        tasks=list(tasks or []),
        edges=list(edges or []),
    )


def test_plan_revision_diff_detects_added_task() -> None:
    """Tasks present in new but not old land in ``added_task_ids``."""
    old = _plan(tasks=[Task(id="t1", title="T1")])
    new = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_task_ids) == ["t2"]
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_edges) == []
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_detects_removed_task() -> None:
    """Tasks present in old but not new land in ``removed_task_ids``."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
        ]
    )
    new = _plan(tasks=[Task(id="t1", title="T1")])
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == ["t2"]
    assert list(diff.modified_task_ids) == []


def test_plan_revision_diff_detects_modified_task() -> None:
    """Same id, differing tracked metadata → ``modified_task_ids``.

    Tracked fields are ``title``, ``description``, ``assignee_agent_id``,
    ``status``. A change to any one of them is enough to surface the
    task as modified.
    """
    old = _plan(
        tasks=[
            Task(id="t1", title="old title", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3", status=TaskStatus.PENDING),
        ]
    )
    new = _plan(
        tasks=[
            # title changed
            Task(id="t1", title="new title", description="d", assignee_agent_id="a1"),
            # assignee changed
            Task(id="t2", title="T2", assignee_agent_id="a2"),
            # status changed
            Task(id="t3", title="T3", status=TaskStatus.COMPLETED),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    # All three modified; preserve new-plan order for deterministic UI.
    assert list(diff.modified_task_ids) == ["t1", "t2", "t3"]
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []


def test_plan_revision_diff_unchanged_task_not_modified() -> None:
    """Identical metadata → task must NOT appear in ``modified_task_ids``."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
        ]
    )
    new = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a1"),
            Task(id="t2", title="T2"),
        ]
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []


def test_plan_revision_diff_detects_added_edge() -> None:
    """An edge present in new but not old lands in ``added_edges``."""
    old = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[],
    )
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(old, new)
    assert len(diff.added_edges) == 1
    assert diff.added_edges[0].from_task_id == "t1"
    assert diff.added_edges[0].to_task_id == "t2"
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_detects_removed_edge() -> None:
    """An edge present in old but not new lands in ``removed_edges``."""
    old = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[],
    )
    diff = build_plan_revision_diff(old, new)
    assert list(diff.added_edges) == []
    assert len(diff.removed_edges) == 1
    assert diff.removed_edges[0].from_task_id == "t1"
    assert diff.removed_edges[0].to_task_id == "t2"


def test_plan_revision_diff_edge_re_target_surfaces_both_added_and_removed() -> None:
    """Re-targeting an edge is modelled as (removed old, added new)."""
    old = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    new = _plan(
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
            Task(id="t3", title="T3"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t3")],
    )
    diff = build_plan_revision_diff(old, new)
    added = [(e.from_task_id, e.to_task_id) for e in diff.added_edges]
    removed = [(e.from_task_id, e.to_task_id) for e in diff.removed_edges]
    assert added == [("t1", "t3")]
    assert removed == [("t1", "t2")]


def test_plan_revision_diff_old_plan_none_treats_all_as_added() -> None:
    """An absent old plan means every task/edge in new is a fresh add."""
    new = _plan(
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(None, new)
    assert list(diff.added_task_ids) == ["t1", "t2"]
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert [(e.from_task_id, e.to_task_id) for e in diff.added_edges] == [("t1", "t2")]
    assert list(diff.removed_edges) == []


def test_plan_revision_diff_identity_revision_is_empty() -> None:
    """Pathological case: new plan deep-equal to old plan → empty diff."""
    plan = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a"),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    # Build a second plan with the same contents (distinct objects; the
    # helper must not short-circuit on identity).
    other = _plan(
        tasks=[
            Task(id="t1", title="T1", description="d", assignee_agent_id="a"),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    diff = build_plan_revision_diff(plan, other)
    assert list(diff.added_task_ids) == []
    assert list(diff.removed_task_ids) == []
    assert list(diff.modified_task_ids) == []
    assert list(diff.added_edges) == []
    assert list(diff.removed_edges) == []


# ---------------------------------------------------------------------------
# plan_revised_event — trigger_event_id propagation (goldfive#199)
# ---------------------------------------------------------------------------


def _revised_plan(*, trigger_event_id: str = "") -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1", status=TaskStatus.PENDING)],
        edges=[],
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_reason="pivot",
        revision_index=1,
        revision_trigger_event_id=trigger_event_id,
    )


def test_plan_revised_event_reads_trigger_event_id_from_plan() -> None:
    """When ``plan.revision_trigger_event_id`` is set, the event copies it."""
    plan = _revised_plan(trigger_event_id="ann_plan_src")
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift_kind=plan.revision_kind,
        severity=plan.revision_severity,
        reason=plan.revision_reason,
        revision_index=plan.revision_index,
    )
    assert evt.plan_revised.trigger_event_id == "ann_plan_src"
    # Plan sub-message also carries it so persisted plans round-trip.
    assert evt.plan_revised.plan.revision_trigger_event_id == "ann_plan_src"


def test_plan_revised_event_user_steer_uses_annotation_id_from_drift() -> None:
    """User-control drift: trigger_event_id resolves to ControlMessage.annotation_id."""
    plan = _revised_plan(trigger_event_id="")  # not stamped on plan
    raw = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-x",
        payload={"note": "pivot", "annotation_id": "ann_from_drift"},
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=raw,
    )
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
    )
    assert evt.plan_revised.trigger_event_id == "ann_from_drift"


def test_plan_revised_event_explicit_kwarg_wins() -> None:
    """Explicit ``trigger_event_id`` kwarg overrides both plan and drift sources."""
    plan = _revised_plan(trigger_event_id="ann_plan_src")
    raw = ControlMessage(
        kind=ControlKind.STEER,
        id="ctl-y",
        payload={"note": "pivot", "annotation_id": "ann_from_drift"},
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="pivot",
        raw=raw,
    )
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
        trigger_event_id="ann_override",
    )
    assert evt.plan_revised.trigger_event_id == "ann_override"


def test_plan_revised_event_autonomous_refine_stamps_drift_id() -> None:
    """Rescope: autonomous drift → trigger_event_id == drift.id (goldfive#199).

    Previously the field was left empty; harmonograf then relied on a
    time-window fallback. The rescope removes that guesswork — every
    refine carries a strict id.
    """
    plan = _revised_plan(trigger_event_id="")
    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="loop detected",
        raw={"event": "tool_error"},  # dict, not ControlMessage
    )
    assert drift.id  # DriftEvent defaults id to a UUID4 at construction
    evt = plan_revised_event(
        run_id="r1",
        sequence=5,
        plan=plan,
        drift=drift,
    )
    assert evt.plan_revised.trigger_event_id == drift.id


def test_plan_revised_event_proto_round_trips_trigger_event_id() -> None:
    """Serialize + deserialize the envelope — the field survives the wire."""
    from goldfive.pb.goldfive.v1 import events_pb2

    plan = _revised_plan(trigger_event_id="ann_roundtrip")
    evt = plan_revised_event(
        run_id="r1",
        sequence=9,
        plan=plan,
        drift_kind=plan.revision_kind,
        revision_index=plan.revision_index,
    )
    encoded = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(encoded)
    assert decoded.plan_revised.trigger_event_id == "ann_roundtrip"
    assert decoded.plan_revised.plan.revision_trigger_event_id == "ann_roundtrip"


# ---------------------------------------------------------------------------
# drift_detected_event / plan_revised_event — kind & severity stamping.
#
# Regression for the telemetry bug where both factories resolved DriftKind /
# DriftSeverity through ``_events_pb_module().DriftKind`` (absent — the enums
# live in types_pb2), which AttributeError'd and, swallowed by a broad
# ``except``, left both enums UNSPECIFIED (0) on every event they produced.
# ---------------------------------------------------------------------------


def _types_pb() -> object:
    from goldfive.pb.goldfive.v1 import types_pb2

    return types_pb2


def test_drift_detected_event_stamps_kind_and_severity() -> None:
    types_pb2 = _types_pb()
    drift = DriftEvent(kind=DriftKind.OFF_TOPIC, severity=DriftSeverity.WARNING, detail="x")
    evt = drift_detected_event("r", 1, drift)
    # The proto values are prefixed + nonzero; the StrEnum value ("off_topic")
    # differs in case from the .name ("OFF_TOPIC") the bridge resolves by.
    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_OFF_TOPIC
    assert evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_WARNING
    assert evt.drift_detected.kind != 0
    assert evt.drift_detected.severity != 0
    # Sibling string fields still stamped correctly (never regressed).
    assert evt.drift_detected.detail == "x"


@pytest.mark.parametrize(
    "kind",
    [
        DriftKind.OFF_TOPIC,
        DriftKind.TOOL_ERROR,
        DriftKind.USER_STEER,
        DriftKind.RUNAWAY_DELEGATION,
        DriftKind.CAPABILITY_MISMATCH,
    ],
)
@pytest.mark.parametrize(
    "severity",
    [DriftSeverity.INFO, DriftSeverity.WARNING, DriftSeverity.CRITICAL],
)
def test_drift_detected_event_kind_severity_table(kind: DriftKind, severity: DriftSeverity) -> None:
    types_pb2 = _types_pb()
    # Every DriftKind's StrEnum .value is the lowercase of its .name, so
    # str(kind) != kind.name in case; the bridge resolves by .name.
    assert str(kind) != kind.name
    expected_kind = getattr(types_pb2, f"DRIFT_KIND_{kind.name}")
    expected_sev = getattr(types_pb2, f"DRIFT_SEVERITY_{severity.name}")
    assert expected_kind != 0
    assert expected_sev != 0
    drift = DriftEvent(kind=kind, severity=severity, detail="d")
    evt = drift_detected_event("r", 2, drift)
    assert evt.drift_detected.kind == expected_kind
    assert evt.drift_detected.severity == expected_sev


def test_drift_detected_event_accepts_bare_string_kind() -> None:
    """The bridge also resolves a lowercase value string (plan_revised path)."""
    types_pb2 = _types_pb()

    class _DuckDrift:
        kind = "off_topic"  # StrEnum .value form, not the enum
        severity = "warning"
        detail = "d"

    evt = drift_detected_event("r", 3, _DuckDrift())
    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_OFF_TOPIC
    assert evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_WARNING


def test_control_received_event_stamps_kind_and_severity() -> None:
    """The control-drift wrapper (fires on every operator steer/cancel) stamps kind."""
    types_pb2 = _types_pb()
    evt = control_received_event("r", 4, ControlKind.STEER, "ctl-1")
    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_USER_STEER
    assert evt.drift_detected.kind != 0
    assert evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_WARNING
    # CANCEL maps to USER_CANCEL — also nonzero.
    cancel = control_received_event("r", 5, ControlKind.CANCEL, "ctl-2")
    assert cancel.drift_detected.kind == types_pb2.DRIFT_KIND_USER_CANCEL
    assert cancel.drift_detected.kind != 0


def test_drift_detected_event_unknown_kind_degrades_to_unspecified() -> None:
    """A synthetic/unknown kind name degrades to UNSPECIFIED without raising."""
    types_pb2 = _types_pb()

    class _DuckDrift:
        kind = "TOTALLY_BOGUS_KIND"
        severity = "ALSO_BOGUS"
        detail = "d"

    # Must not raise.
    evt = drift_detected_event("r", 6, _DuckDrift())
    assert evt.drift_detected.kind == types_pb2.DRIFT_KIND_UNSPECIFIED == 0
    assert evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_UNSPECIFIED == 0
    # detail is unaffected by the enum miss.
    assert evt.drift_detected.detail == "d"


def test_drift_detected_event_round_trips_kind_severity() -> None:
    """kind/severity survive SerializeToString + FromString."""
    from goldfive.pb.goldfive.v1 import events_pb2

    types_pb2 = _types_pb()
    drift = DriftEvent(
        kind=DriftKind.CAPABILITY_MISMATCH, severity=DriftSeverity.CRITICAL, detail="d"
    )
    evt = drift_detected_event("r", 7, drift)
    decoded = events_pb2.Event()
    decoded.ParseFromString(evt.SerializeToString())
    assert decoded.drift_detected.kind == types_pb2.DRIFT_KIND_CAPABILITY_MISMATCH != 0
    assert decoded.drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL != 0


def test_plan_revised_event_stamps_drift_kind_and_severity() -> None:
    """Sibling regression: plan_revised_event also stamps the two enums."""
    types_pb2 = _types_pb()
    plan = _revised_plan(trigger_event_id="ann_x")
    evt = plan_revised_event(
        run_id="r1",
        sequence=8,
        plan=plan,
        drift_kind=plan.revision_kind,  # "user_steer" (StrEnum value)
        severity=plan.revision_severity,  # "warning"
        reason=plan.revision_reason,
    )
    assert evt.plan_revised.drift_kind == types_pb2.DRIFT_KIND_USER_STEER != 0
    assert evt.plan_revised.severity == types_pb2.DRIFT_SEVERITY_WARNING != 0


def test_plan_revised_event_round_trips_drift_kind_and_severity() -> None:
    from goldfive.pb.goldfive.v1 import events_pb2

    types_pb2 = _types_pb()
    plan = _revised_plan(trigger_event_id="ann_y")
    evt = plan_revised_event(
        run_id="r1",
        sequence=9,
        plan=plan,
        drift_kind=DriftKind.RUNAWAY_DELEGATION.value,
        severity=DriftSeverity.CRITICAL.value,
    )
    decoded = events_pb2.Event()
    decoded.ParseFromString(evt.SerializeToString())
    assert decoded.plan_revised.drift_kind == types_pb2.DRIFT_KIND_RUNAWAY_DELEGATION != 0
    assert decoded.plan_revised.severity == types_pb2.DRIFT_SEVERITY_CRITICAL != 0


def test_factory_and_steerer_resolve_enums_identically() -> None:
    """Lockstep guard: the shared bridge means the primary steerer emit path
    (DefaultSteerer._drift_*_pb_value) and the factories never re-diverge."""
    from goldfive.steerer import DefaultSteerer

    for kind in [DriftKind.OFF_TOPIC, DriftKind.RUNAWAY_DELEGATION, DriftKind.USER_STEER]:
        evt = drift_detected_event("r", 1, DriftEvent(kind=kind, severity=DriftSeverity.WARNING))
        assert evt.drift_detected.kind == DefaultSteerer._drift_kind_pb_value(kind) != 0
    for sev in [DriftSeverity.INFO, DriftSeverity.WARNING, DriftSeverity.CRITICAL]:
        evt = drift_detected_event("r", 1, DriftEvent(kind=DriftKind.OFF_TOPIC, severity=sev))
        assert evt.drift_detected.severity == DefaultSteerer._drift_severity_pb_value(sev) != 0


# ---------------------------------------------------------------------------
# invocation_cancelled_event factory (goldfive#251 Stream C / A5 promotion)
# ---------------------------------------------------------------------------


def test_invocation_cancelled_event_builds_proto_with_all_fields() -> None:
    """Factory populates the typed payload with every field provided."""
    evt = invocation_cancelled_event(
        run_id="run-1",
        sequence=3,
        invocation_id="inv-abc",
        agent_name="planner_agent",
        reason="user_steer",
        severity="critical",
        drift_id="drift-xyz",
        drift_kind="off_topic",
        detail="agent wandered into raccoons",
        tool_name="search",
        session_id="sess-1",
    )
    assert evt.WhichOneof("payload") == "invocation_cancelled"
    assert evt.run_id == "run-1"
    assert evt.sequence == 3
    assert evt.session_id == "sess-1"
    payload = evt.invocation_cancelled
    assert payload.invocation_id == "inv-abc"
    assert payload.agent_name == "planner_agent"
    assert payload.reason == "user_steer"
    assert payload.severity == "critical"
    assert payload.drift_id == "drift-xyz"
    assert payload.drift_kind == "off_topic"
    assert payload.detail == "agent wandered into raccoons"
    assert payload.tool_name == "search"


def test_invocation_cancelled_event_defaults_optional_fields_to_empty() -> None:
    """Only ``invocation_id`` is required; other fields default to ``""``."""
    evt = invocation_cancelled_event(
        run_id="run-2",
        sequence=0,
        invocation_id="inv-only",
    )
    payload = evt.invocation_cancelled
    assert payload.invocation_id == "inv-only"
    assert payload.agent_name == ""
    assert payload.reason == ""
    assert payload.severity == ""
    assert payload.drift_id == ""
    assert payload.drift_kind == ""
    assert payload.detail == ""
    assert payload.tool_name == ""


def test_invocation_cancelled_event_round_trips_through_wire() -> None:
    """Serialize + deserialize the envelope — fields survive the wire."""
    from goldfive.pb.goldfive.v1 import events_pb2

    evt = invocation_cancelled_event(
        run_id="run-rt",
        sequence=7,
        invocation_id="inv-rt",
        agent_name="sub_agent",
        reason="drift",
        severity="warning",
        drift_id="d-1",
        drift_kind="agent_refusal",
        detail="loopy",
        tool_name="lookup",
        session_id="sess-rt",
    )
    encoded = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(encoded)
    assert decoded.WhichOneof("payload") == "invocation_cancelled"
    assert decoded.run_id == "run-rt"
    assert decoded.sequence == 7
    assert decoded.session_id == "sess-rt"
    payload = decoded.invocation_cancelled
    assert payload.invocation_id == "inv-rt"
    assert payload.agent_name == "sub_agent"
    assert payload.reason == "drift"
    assert payload.severity == "warning"
    assert payload.drift_id == "d-1"
    assert payload.drift_kind == "agent_refusal"
    assert payload.detail == "loopy"
    assert payload.tool_name == "lookup"


# ---------------------------------------------------------------------------
# Plan-descriptive-growth proto extension (goldfive#423 PR 1; design doc
# ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.3.0, §9, §13).
#
# The ``DelegationObserved.tool_args_json`` field is the canonical
# observed-fact carrier PR 2's dedup hash reads from. PR 1 just verifies
# the field flows through the event builder + survives the wire +
# defaults to empty when the producer does not pass it.
# ---------------------------------------------------------------------------


def test_delegation_observed_event_default_tool_args_json_is_empty() -> None:
    """Producers that do not pass ``tool_args_json`` produce the same
    wire shape as pre-PR-1 — empty string default, no field on the wire.
    """
    evt = delegation_observed_event(
        run_id="r",
        sequence=1,
        from_agent="coordinator",
        to_agent="researcher",
    )
    assert evt.delegation_observed.tool_args_json == ""


def test_delegation_observed_event_carries_tool_args_json() -> None:
    """The new kwarg is stamped onto the proto payload when provided."""
    payload = '{"topic": "cherry trees"}'
    evt = delegation_observed_event(
        run_id="r",
        sequence=1,
        from_agent="coordinator",
        to_agent="researcher",
        tool_args_json=payload,
    )
    assert evt.delegation_observed.tool_args_json == payload


def test_delegation_observed_event_round_trips_tool_args_json_through_wire() -> None:
    """Serialise + parse — the JSON payload survives the wire format."""
    from goldfive.pb.goldfive.v1 import events_pb2

    payload = '{"request": "locate cherry tree files", "format": "json"}'
    evt = delegation_observed_event(
        run_id="r-rt",
        sequence=5,
        from_agent="coordinator",
        to_agent="debugger_agent",
        task_id="t1",
        invocation_id="inv-1",
        tool_args_json=payload,
    )
    encoded = evt.SerializeToString()
    decoded = events_pb2.Event()
    decoded.ParseFromString(encoded)
    assert decoded.WhichOneof("payload") == "delegation_observed"
    assert decoded.delegation_observed.tool_args_json == payload


def test_delegation_observed_old_wire_format_parses_with_default_tool_args() -> None:
    """Old serialised events (no ``tool_args_json`` field) deserialize.

    Critical UI-safety guarantee: harmonograf builds that ingested old
    events MUST still be loadable post-PR-1. We simulate the pre-PR-1
    wire shape by serialising a fresh DelegationObserved without ever
    setting the new field, then parsing through the current schema.
    The default empty string lights up — never a missing-field error.
    """
    from goldfive.pb.goldfive.v1 import events_pb2

    # Construct without touching tool_args_json (matches the pre-PR-1
    # producer that did not know about the field).
    msg = events_pb2.DelegationObserved(
        from_agent="coordinator",
        to_agent="debugger_agent",
        task_id="t1",
        invocation_id="inv-1",
    )
    wire = msg.SerializeToString()
    parsed = events_pb2.DelegationObserved()
    parsed.ParseFromString(wire)
    # The default-empty back-compat default lights up.
    assert parsed.tool_args_json == ""
    # And the other fields survive — back-compat is additive only.
    assert parsed.from_agent == "coordinator"
    assert parsed.to_agent == "debugger_agent"
    assert parsed.task_id == "t1"
    assert parsed.invocation_id == "inv-1"


def test_delegation_observed_event_wire_bytes_identical_to_pre_pr1_when_no_tool_args() -> None:
    """Canonical UI-safety regression guard: a caller that does NOT pass
    ``tool_args_json`` MUST produce byte-identical wire to the pre-PR-1
    producer that did not know about the field.

    This is the load-bearing claim from PR 1's brief:
    *conditional emission means byte-identical old wire*. Proto3 omits
    default-valued scalars from the wire, but the helper applies
    ``if tool_args_json:`` defensively even for the empty-string default
    — this test pins that contract so any future regression (e.g. a
    refactor that drops the conditional and assigns unconditionally)
    fails loudly.

    We compare the current helper's output against a "pre-PR-1 shape"
    proto built without ever touching ``tool_args_json``.
    """
    from goldfive.pb.goldfive.v1 import events_pb2

    # New-code path: helper called without tool_args_json kwarg.
    evt_new = delegation_observed_event(
        run_id="r-bc",
        sequence=7,
        from_agent="coordinator",
        to_agent="researcher",
        task_id="t1",
        invocation_id="inv-1",
        event_id="e1",
        observed_at=events_pb2.DelegationObserved().observed_at.__class__(seconds=42),
    )
    new_wire = evt_new.SerializeToString()

    # Pre-PR-1 shape: build the same DelegationObserved without ever
    # touching tool_args_json. (Schema knows about the field, but the
    # producer doesn't set it — same wire-output condition as a build
    # that pre-dates the field.)
    pre_inner = events_pb2.DelegationObserved(
        from_agent="coordinator",
        to_agent="researcher",
        task_id="t1",
        invocation_id="inv-1",
    )
    pre_inner.observed_at.CopyFrom(evt_new.delegation_observed.observed_at)
    pre_evt = events_pb2.Event(
        event_id="e1",
        run_id="r-bc",
        sequence=7,
        delegation_observed=pre_inner,
    )
    pre_evt.emitted_at.CopyFrom(evt_new.emitted_at)
    pre_wire = pre_evt.SerializeToString()

    assert new_wire == pre_wire, (
        "Byte-identity contract broken: new event helper without "
        "tool_args_json must serialise to the same bytes as a pre-PR-1 "
        "producer that did not know about the field."
    )


def test_delegation_observed_event_wire_bytes_change_only_when_tool_args_passed() -> None:
    """Companion to the byte-identity guard: when ``tool_args_json`` IS
    provided, the wire MUST differ (otherwise the field would be a
    no-op and PR 2's dedup would silently fail).
    """
    common = dict(
        run_id="r-bc2",
        sequence=8,
        from_agent="coordinator",
        to_agent="researcher",
        task_id="t1",
        invocation_id="inv-1",
        event_id="e2",
    )
    evt_empty = delegation_observed_event(**common)
    evt_with_args = delegation_observed_event(**common, tool_args_json='{"x":"y"}')
    # Align observed_at + emitted_at so only tool_args_json differs.
    evt_with_args.delegation_observed.observed_at.CopyFrom(
        evt_empty.delegation_observed.observed_at
    )
    evt_with_args.emitted_at.CopyFrom(evt_empty.emitted_at)
    assert evt_empty.SerializeToString() != evt_with_args.SerializeToString()


# ---------------------------------------------------------------------------
# emit: fan-out isolation
# ---------------------------------------------------------------------------


class _RaisingSink:
    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, event_pb: object) -> None:
        self.calls += 1
        raise RuntimeError("sink is broken")

    async def close(self) -> None:
        return None


async def test_emit_logs_and_swallows_sink_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising sink is logged with its class name; siblings still see
    the event and the caller never observes the exception."""
    import logging

    from goldfive.events import emit, new_event
    from goldfive.sinks import InMemorySink

    bad = _RaisingSink()
    good = InMemorySink()
    evt = new_event(run_id="r1", sequence=0)

    with caplog.at_level(logging.ERROR, logger="goldfive.events"):
        await emit([bad, good], evt)

    assert bad.calls == 1
    assert [e.sequence for e in good.events] == [0]
    messages = [r.getMessage() for r in caplog.records]
    assert any("_RaisingSink" in m for m in messages)


async def test_runner_survives_raising_sink() -> None:
    """Lifecycle emits in Runner.run must not abort the turn when a
    sink raises (documented fan-out isolation)."""
    from goldfive import (
        CallableAdapter,
        InMemorySink,
        InvocationResult,
        Plan,
        ReportingToolSpec,
        Runner,
        SequentialExecutor,
        Session,
        StaticPlanner,
    )

    async def agent(
        task: Task, session: Session, tools: list[ReportingToolSpec]
    ) -> InvocationResult:
        return InvocationResult(task_id=task.id, text=task.title)

    plan = Plan(
        id="p",
        run_id="",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="a", assignee_agent_id="worker")],
        edges=[],
        summary="",
    )
    bad = _RaisingSink()
    good = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(agent, available_agents=["worker"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(),
        sinks=[bad, good],
    )
    outcome = await runner.run("go")
    await runner.close()

    assert outcome.success
    assert bad.calls > 0
    kinds = [e.WhichOneof("payload") for e in good.events if hasattr(e, "DESCRIPTOR")]
    assert "run_started" in kinds
    assert "task_completed" in kinds
