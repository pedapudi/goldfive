"""Tests for the Phase-1 :class:`StateStore` typed handle.

goldfive#271 Phase 1. Pins:

* Each accessor's read/write contract — pin / active steer / correction
  / pending-delegation / reasoning-extracted binding.
* Reasoning extraction binding workflow: judge returns
  ``focused_task_id``; store records; pin resolution sees the new
  binding (covered in :mod:`tests.test_pin_resolution_ladder`).
* The store doesn't leak to ADK ``session.state`` — Phase 0's
  tripwire stays green throughout (autouse fixture in conftest.py
  already sets ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1``; any unintended
  ADK-state mutation would raise here).
* Read fallback: store-first; legacy state slot as backup for compat.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import state_store as _ostate  # noqa: E402
from goldfive.state_store import (  # noqa: E402
    REASONING_BINDINGS_KEY,
    ActiveSteer,
    BindingSource,
    DelegationPin,
    ReasoningBinding,
    StateStore,
)
from goldfive.types import Plan, Session, Task  # noqa: E402

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_for_session_uses_session_state_dict() -> None:
    """``for_session`` returns a store backed by ``session.state``."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    # Empty initially.
    assert store.pin_current_task() == ""
    # Mutating through the store reflects on ``session.state``.
    store.set_pin_current_task("t1", source=BindingSource.AGENT_CALLBACK)
    assert session.state[_ostate.KEY_CURRENT_TASK_ID] == "t1"


def test_for_session_none_yields_empty_view() -> None:
    """``for_session(None)`` is safe — reads default; writes drop silently."""
    store = StateStore.for_session(None)
    assert store.pin_current_task() == ""
    # Write should silently drop — no exception, no side effect.
    store.set_pin_current_task("t1")


def test_for_state_uses_arbitrary_dict() -> None:
    """``for_state`` accepts a plain dict (test scaffolding)."""
    state: dict = {}
    store = StateStore.for_state(state)
    store.set_pin_current_task("tA", source=BindingSource.AGENT_CALLBACK)
    assert state[_ostate.KEY_CURRENT_TASK_ID] == "tA"


def test_construction_tolerates_non_mapping() -> None:
    """A non-mapping ``state`` argument degrades to an empty store."""
    store = StateStore(state=object())
    assert store.pin_current_task() == ""
    # And writes are no-ops, since the underlying dict isn't present.
    store.set_pin_current_task("xx")


# ---------------------------------------------------------------------------
# Pin
# ---------------------------------------------------------------------------


def test_pin_round_trip() -> None:
    """``set_pin_current_task`` writes; ``pin_current_task`` reads back."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.set_pin_current_task(
        "task-42",
        source=BindingSource.AGENT_CALLBACK,
        title="Drive forward",
        revision=3,
    )
    assert store.pin_current_task() == "task-42"
    assert store.pin_current_task_title() == "Drive forward"
    assert store.pin_current_task_revision() == 3


def test_pin_empty_id_is_noop() -> None:
    """An empty task_id does not clobber the existing pin."""
    session = Session(run_id="r1")
    session.state[_ostate.KEY_CURRENT_TASK_ID] = "existing"
    StateStore.for_session(session).set_pin_current_task("")
    assert session.state[_ostate.KEY_CURRENT_TASK_ID] == "existing"


def test_pin_clear() -> None:
    """``clear_pin_current_task`` removes id, title, revision."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.set_pin_current_task("t", title="x", revision=1)
    store.clear_pin_current_task()
    assert _ostate.KEY_CURRENT_TASK_ID not in session.state
    assert _ostate.KEY_CURRENT_TASK_TITLE not in session.state


def test_pin_revision_default_zero_when_unset() -> None:
    """Reading the revision before any write returns ``0``."""
    store = StateStore.for_session(Session(run_id="r1"))
    assert store.pin_current_task_revision() == 0


# ---------------------------------------------------------------------------
# Active steer
# ---------------------------------------------------------------------------


def test_get_active_steer_returns_none_when_unset() -> None:
    """No body recorded → ``get_active_steer()`` returns ``None``."""
    store = StateStore.for_session(Session(run_id="r1"))
    assert store.get_active_steer() is None


def test_get_active_steer_round_trip() -> None:
    """A stamped active steer reads back as a typed view."""
    session = Session(run_id="r1")
    _ostate.set_active_steer(
        session.state,
        body="focus on X",
        at_turn=12,
        author="op@example",
        source="user",
    )
    active = StateStore.for_session(session).get_active_steer()
    assert isinstance(active, ActiveSteer)
    assert active.body == "focus on X"
    assert active.at_turn == 12
    assert active.author == "op@example"
    assert active.source == "user"
    assert active.is_active()


def test_active_steer_empty_body_returns_none() -> None:
    """Explicitly-empty body still reads as 'no steer' (None)."""
    session = Session(run_id="r1")
    session.state[_ostate.KEY_ACTIVE_STEER_BODY] = ""
    session.state[_ostate.KEY_ACTIVE_STEER_AT_TURN] = 5
    assert StateStore.for_session(session).get_active_steer() is None


# ---------------------------------------------------------------------------
# Pending corrections
# ---------------------------------------------------------------------------


def test_get_correction_returns_none_when_unset() -> None:
    store = StateStore.for_session(Session(run_id="r1"))
    assert store.get_correction("agent_x", "task_y") is None
    assert not store.has_correction("agent_x", "task_y")


def test_get_correction_round_trip_dict_payload() -> None:
    """A dict correction payload reads back unchanged."""
    session = Session(run_id="r1")
    payload = {
        "agent_name": "agent_x",
        "task_id": "task_y",
        "superseded_task_id": "task_old",
        "revision_number": 2,
    }
    session.state["goldfive.pending_corrections.agent_x.task_y"] = payload
    store = StateStore.for_session(session)
    assert store.has_correction("agent_x", "task_y")
    assert store.get_correction("agent_x", "task_y") == payload


def test_iter_corrections_for_agent() -> None:
    session = Session(run_id="r1")
    session.state["goldfive.pending_corrections.agent_x.task_a"] = {"a": 1}
    session.state["goldfive.pending_corrections.agent_x.task_b"] = {"b": 1}
    session.state["goldfive.pending_corrections.other.task_c"] = {"c": 1}
    found = sorted(
        StateStore.for_session(session).iter_corrections_for_agent(
            "agent_x"
        )
    )
    assert found == ["task_a", "task_b"]


def test_iter_corrections_strips_compound_agent_form() -> None:
    """Compound ``client42:agent_x`` finds bare ``agent_x`` writes."""
    session = Session(run_id="r1")
    session.state["goldfive.pending_corrections.agent_x.task_a"] = {"a": 1}
    found = StateStore.for_session(session).iter_corrections_for_agent(
        "client42:agent_x"
    )
    assert found == ["task_a"]


# ---------------------------------------------------------------------------
# Pending delegations
# ---------------------------------------------------------------------------


def test_get_pending_delegation_versioned_dict() -> None:
    """The new ``{task_id, revision, tool_args}`` shape resolves cleanly."""
    session = Session(run_id="r1")
    session.state["goldfive.pending_delegations"] = {
        "fc-1": {
            "task_id": "tA",
            "revision": 7,
            "tool_args": {"q": "solar"},
        },
    }
    pin = StateStore.for_session(session).get_pending_delegation("fc-1")
    assert isinstance(pin, DelegationPin)
    assert pin.task_id == "tA"
    assert pin.revision == 7
    assert pin.tool_args == {"q": "solar"}
    assert pin.is_set()


def test_get_pending_delegation_legacy_string_shape() -> None:
    """The pre-#266 bare-string entry shape still resolves."""
    session = Session(run_id="r1")
    session.state["goldfive.pending_delegations"] = {"fc-1": "tLegacy"}
    pin = StateStore.for_session(session).get_pending_delegation("fc-1")
    assert pin is not None
    assert pin.task_id == "tLegacy"
    assert pin.revision == 0


def test_get_pending_delegation_missing_returns_none() -> None:
    store = StateStore.for_session(Session(run_id="r1"))
    assert store.get_pending_delegation("fc-zzz") is None
    assert store.get_pending_delegation("") is None


# ---------------------------------------------------------------------------
# Reasoning-extracted bindings (NEW Phase 1)
# ---------------------------------------------------------------------------


def test_record_reasoning_binding_round_trip() -> None:
    """A recorded binding reads back via ``get_reasoning_extracted_binding``."""
    session = Session(run_id="run-1")
    store = StateStore.for_session(session)
    rec = store.record_reasoning_extracted_binding(
        agent_name="agent_x",
        task_id="t1",
        confidence=0.85,
        recorded_at_turn=4,
        run_id="run-1",
        session_id="run-1",
    )
    assert rec is not None
    assert rec.confidence == pytest.approx(0.85)
    fetched = store.get_reasoning_extracted_binding("agent_x")
    assert fetched is not None
    assert fetched.task_id == "t1"
    assert fetched.confidence == pytest.approx(0.85)
    assert fetched.recorded_at_turn == 4


def test_record_reasoning_binding_clamps_confidence() -> None:
    """Confidence outside ``[0, 1]`` is clamped at write time."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    rec = store.record_reasoning_extracted_binding(
        agent_name="a", task_id="t", confidence=1.5
    )
    assert rec is not None
    assert rec.confidence == 1.0
    rec2 = store.record_reasoning_extracted_binding(
        agent_name="b", task_id="t", confidence=-0.4
    )
    assert rec2 is not None
    assert rec2.confidence == 0.0


def test_record_reasoning_binding_rejects_empty_inputs() -> None:
    """Empty agent / task name skips the write."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    assert store.record_reasoning_extracted_binding(
        agent_name="", task_id="t", confidence=0.9
    ) is None
    assert store.record_reasoning_extracted_binding(
        agent_name="a", task_id="", confidence=0.9
    ) is None
    assert REASONING_BINDINGS_KEY not in session.state


def test_get_reasoning_binding_compound_falls_back_to_bare() -> None:
    """A compound ``client42:agent_x`` lookup finds the bare-form binding."""
    session = Session(run_id="r1")
    StateStore.for_session(session).record_reasoning_extracted_binding(
        agent_name="agent_x", task_id="t1", confidence=0.9
    )
    fetched = StateStore.for_session(session).get_reasoning_extracted_binding(
        "client42:agent_x"
    )
    assert fetched is not None
    assert fetched.task_id == "t1"


def test_clear_reasoning_binding_drops_entry() -> None:
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.record_reasoning_extracted_binding(
        agent_name="agent_x", task_id="t1", confidence=0.9
    )
    store.clear_reasoning_extracted_binding("agent_x")
    assert store.get_reasoning_extracted_binding("agent_x") is None


def test_record_preserves_unrelated_bindings() -> None:
    """Recording for one agent doesn't clobber another agent's binding."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.record_reasoning_extracted_binding(
        agent_name="alpha", task_id="t_alpha", confidence=0.8
    )
    store.record_reasoning_extracted_binding(
        agent_name="beta", task_id="t_beta", confidence=0.9
    )
    assert store.get_reasoning_extracted_binding("alpha").task_id == "t_alpha"
    assert store.get_reasoning_extracted_binding("beta").task_id == "t_beta"


def test_record_overwrites_same_agent_binding() -> None:
    """A second record for the same agent supersedes the first."""
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.record_reasoning_extracted_binding(
        agent_name="a", task_id="t1", confidence=0.5
    )
    store.record_reasoning_extracted_binding(
        agent_name="a", task_id="t2", confidence=0.95
    )
    fetched = store.get_reasoning_extracted_binding("a")
    assert fetched.task_id == "t2"
    assert fetched.confidence == pytest.approx(0.95)


def test_reasoning_binding_dict_round_trip() -> None:
    """``ReasoningBinding.from_dict`` reverses ``to_dict`` faithfully."""
    b = ReasoningBinding(
        agent_name="a",
        task_id="t",
        confidence=0.7,
        recorded_at_turn=4,
        run_id="r",
        session_id="s",
    )
    rebuilt = ReasoningBinding.from_dict(b.to_dict())
    assert rebuilt == b


def test_reasoning_binding_from_dict_rejects_garbage() -> None:
    assert ReasoningBinding.from_dict(None) is None
    assert ReasoningBinding.from_dict("not a dict") is None
    assert ReasoningBinding.from_dict({}) is None  # missing task_id
    assert ReasoningBinding.from_dict({"task_id": ""}) is None


# ---------------------------------------------------------------------------
# Read fallback / no-leak invariants
# ---------------------------------------------------------------------------


def test_store_does_not_write_to_adk_state() -> None:
    """The store wraps goldfive ``Session.state``, never ADK's session.state.

    The autouse ``_state_audit_enabled`` fixture in conftest.py sets
    ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1`` for every test. If the store
    leaked a write to ADK's session.state from a callback frame, the
    tripwire would raise. Exercising every write path here provides
    structural coverage that the store doesn't introduce a new
    Phase-0 violation.
    """
    session = Session(run_id="r1")
    store = StateStore.for_session(session)
    store.set_pin_current_task("t1", source=BindingSource.AGENT_CALLBACK, revision=2)
    store.record_reasoning_extracted_binding(
        agent_name="a", task_id="t1", confidence=0.9
    )
    store.clear_reasoning_extracted_binding("a")
    store.clear_pin_current_task()
    # If the audit had fired, we'd never have reached the assertions.
    assert _ostate.KEY_CURRENT_TASK_ID not in session.state


def test_store_reads_match_legacy_state_layout() -> None:
    """A pre-existing legacy state dict is readable through the store.

    Drives the back-compat path: code that wrote keys directly under
    the goldfive prefix (orchestration_state primitives) is read by
    the new typed accessors without re-stamping. Phase 1's read
    migration relies on this — existing writers stay where they are
    (Phase 2's job to migrate them) while readers move to the typed
    accessors.
    """
    session = Session(run_id="r1")
    # Legacy direct-state writes (the writers Phase 2 will migrate).
    session.state[_ostate.KEY_CURRENT_TASK_ID] = "legacy-task"
    session.state[_ostate.KEY_CURRENT_TASK_TITLE] = "Legacy Title"
    session.state[_ostate.KEY_CURRENT_TASK_REVISION] = 5
    _ostate.set_active_steer(
        session.state,
        body="legacy",
        at_turn=2,
        author="op",
        source="user",
    )
    session.state["goldfive.pending_corrections.agent_x.legacy-task"] = {
        "agent_name": "agent_x",
        "task_id": "legacy-task",
    }
    session.state["goldfive.pending_delegations"] = {"fc-1": "legacy-task"}

    store = StateStore.for_session(session)
    assert store.pin_current_task() == "legacy-task"
    assert store.pin_current_task_title() == "Legacy Title"
    assert store.pin_current_task_revision() == 5
    active = store.get_active_steer()
    assert active is not None and active.body == "legacy"
    assert store.has_correction("agent_x", "legacy-task")
    pin = store.get_pending_delegation("fc-1")
    assert pin is not None and pin.task_id == "legacy-task"


def test_binding_workflow_with_plan_pin_resolution_signal() -> None:
    """End-to-end: store records → pin ladder helper finds matching task.

    This is the integration shape the steerer's reasoning-judge
    background path depends on. Validated end-to-end in
    ``test_pin_resolution_ladder`` against the live plugin
    callback; this test pins the orchestration_store contract that
    the binding's ``task_id`` matches a plan task by id.
    """
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t_alpha", title="Alpha", assignee_agent_id="agent_x"),
            Task(id="t_beta", title="Beta", assignee_agent_id="agent_x"),
        ],
        edges=[],
        summary="",
    )
    session = Session(run_id="r1", plan=plan)
    store = StateStore.for_session(session)
    store.record_reasoning_extracted_binding(
        agent_name="agent_x",
        task_id="t_beta",
        confidence=0.92,
    )
    binding = store.get_reasoning_extracted_binding("agent_x")
    assert binding is not None and binding.task_id == "t_beta"
    # The matching plan task is locatable by id.
    matching = next((t for t in plan.tasks if t.id == binding.task_id), None)
    assert matching is not None
    assert matching.title == "Beta"
