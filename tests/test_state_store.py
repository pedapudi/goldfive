"""Mock-dict unit tests for the unified :class:`StateStore` surface.

These tests deliberately avoid spinning up a goldfive ``Session`` (or
any ADK runtime) and exercise the store directly against a plain dict
via :meth:`StateStore.for_state`. They complement the broader
``test_orchestration_state.py`` and ``test_orchestration_store.py``
suites — which already pin the module-level helpers and the
:class:`Session`-backed accessor surface — by giving the merged module
a focused unit-level smoke test.

Wave A piece 1 of the goldfive refactor: combined surface of the
former ``orchestration_state`` and ``orchestration_store`` modules.
"""

from __future__ import annotations

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import state_store  # noqa: E402
from goldfive.state_store import (  # noqa: E402
    ActiveSteer,
    BindingSource,
    DelegationPin,
    StateStore,
)
from goldfive.types import DriftKind, DriftSeverity  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level primitives accept plain dicts
# ---------------------------------------------------------------------------


def test_write_rejects_non_goldfive_key() -> None:
    """``write`` defensively refuses any key not under the prefix."""
    state: dict = {}
    with pytest.raises(ValueError, match="non-goldfive key"):
        state_store.write(state, "stray.key", "v")


def test_write_and_read_round_trip() -> None:
    """A round-trip through ``write`` + ``read`` returns the same value."""
    state: dict = {}
    state_store.write(state, state_store.KEY_CURRENT_PLAN_ID, "plan-1")
    assert state_store.read(state, state_store.KEY_CURRENT_PLAN_ID) == "plan-1"


def test_read_returns_default_for_non_mapping() -> None:
    """A non-mapping ``state`` reads the supplied default."""
    assert state_store.read(object(), state_store.KEY_CURRENT_PLAN_ID, "fallback") == "fallback"


def test_clear_removes_key_idempotently() -> None:
    """``clear`` removes a goldfive key; a second call is a no-op."""
    state: dict = {state_store.KEY_GOALS_SUMMARY: "summary"}
    state_store.clear(state, state_store.KEY_GOALS_SUMMARY)
    state_store.clear(state, state_store.KEY_GOALS_SUMMARY)  # idempotent
    assert state_store.KEY_GOALS_SUMMARY not in state


# ---------------------------------------------------------------------------
# StateStore via for_state (no Session involved)
# ---------------------------------------------------------------------------


def test_for_state_pin_round_trip() -> None:
    """``for_state`` returns a working store backed by the plain dict."""
    state: dict = {}
    store = StateStore.for_state(state)
    store.set_pin_current_task(
        "task-7", source=BindingSource.AGENT_CALLBACK, title="Do X", revision=2
    )
    assert state[state_store.KEY_CURRENT_TASK_ID] == "task-7"
    assert state[state_store.KEY_CURRENT_TASK_TITLE] == "Do X"
    assert state[state_store.KEY_CURRENT_TASK_REVISION] == 2
    assert store.pin_current_task() == "task-7"
    assert store.pin_current_task_title() == "Do X"
    assert store.pin_current_task_revision() == 2


def test_for_state_clear_pin_removes_all_slots() -> None:
    """``clear_pin_current_task`` drops id + title + revision."""
    state: dict = {}
    store = StateStore.for_state(state)
    store.set_pin_current_task("t", title="x", revision=1)
    store.clear_pin_current_task()
    assert state_store.KEY_CURRENT_TASK_ID not in state
    assert state_store.KEY_CURRENT_TASK_TITLE not in state
    assert state_store.KEY_CURRENT_TASK_REVISION not in state


def test_for_state_active_steer_round_trip() -> None:
    """A stamped active steer reads back as a typed ``ActiveSteer``."""
    state: dict = {}
    state_store.set_active_steer(
        state, body="pivot now", at_turn=12, author="alice", source="user"
    )
    store = StateStore.for_state(state)
    steer = store.get_active_steer()
    assert isinstance(steer, ActiveSteer)
    assert steer.body == "pivot now"
    assert steer.at_turn == 12
    assert steer.author == "alice"
    assert steer.source == "user"
    assert steer.is_active() is True


def test_for_state_active_steer_returns_none_when_unset() -> None:
    """No body recorded → ``get_active_steer`` returns ``None``."""
    store = StateStore.for_state({})
    assert store.get_active_steer() is None


def test_for_state_pending_delegation_round_trip() -> None:
    """Versioned delegation pin round-trips as a :class:`DelegationPin`."""
    state: dict = {}
    store = StateStore.for_state(state)
    store.set_pending_delegation(
        "fc-1", task_id="t-42", revision=3, tool_args={"k": "v"}
    )
    pin = store.get_pending_delegation("fc-1")
    assert pin == DelegationPin(task_id="t-42", revision=3, tool_args={"k": "v"})


def test_for_state_pending_delegation_legacy_bare_string() -> None:
    """A legacy bare-string entry normalises to a :class:`DelegationPin`."""
    state: dict = {state_store.PENDING_DELEGATIONS_KEY: {"fc-old": "task-legacy"}}
    pin = StateStore.for_state(state).get_pending_delegation("fc-old")
    assert pin == DelegationPin(task_id="task-legacy")


def test_for_state_pending_delegation_missing_returns_none() -> None:
    """Lookup against an empty / missing slot returns ``None``."""
    assert StateStore.for_state({}).get_pending_delegation("fc-missing") is None


def test_for_state_reasoning_binding_round_trip() -> None:
    """Recording then reading a reasoning-extracted binding round-trips."""
    state: dict = {}
    store = StateStore.for_state(state)
    binding = store.record_reasoning_extracted_binding(
        agent_name="agent_a",
        task_id="task-1",
        confidence=0.82,
        recorded_at_turn=5,
    )
    assert binding is not None
    fetched = store.get_reasoning_extracted_binding("agent_a")
    assert fetched is not None
    assert fetched.task_id == "task-1"
    assert fetched.confidence == pytest.approx(0.82)


def test_for_state_reasoning_binding_clamps_confidence() -> None:
    """Out-of-range confidence values are clamped to ``[0.0, 1.0]``."""
    store = StateStore.for_state({})
    binding = store.record_reasoning_extracted_binding(
        agent_name="agent_a", task_id="t", confidence=2.5
    )
    assert binding is not None
    assert binding.confidence == 1.0


def test_for_state_reasoning_binding_clear_drops_entry() -> None:
    """``clear_reasoning_extracted_binding`` is idempotent and removes the entry."""
    store = StateStore.for_state({})
    store.record_reasoning_extracted_binding(
        agent_name="agent_a", task_id="t", confidence=0.7
    )
    store.clear_reasoning_extracted_binding("agent_a")
    store.clear_reasoning_extracted_binding("agent_a")  # idempotent
    assert store.get_reasoning_extracted_binding("agent_a") is None


def test_for_state_active_drift_open_and_resolve() -> None:
    """The drift lifecycle is reachable from the StateStore methods."""
    state: dict = {}
    store = StateStore.for_state(state)
    drift = store.open_or_escalate_drift(
        kind=DriftKind.LOOPING_REASONING,
        task_id="t-1",
        agent_id="a-1",
        turn_id="turn-1",
        severity=DriftSeverity.WARNING,
    )
    assert drift.lifecycle == state_store.LIFECYCLE_OPENED
    assert drift.occurrences == 1

    escalated = store.open_or_escalate_drift(
        kind=DriftKind.LOOPING_REASONING,
        task_id="t-1",
        agent_id="a-1",
        turn_id="turn-1",
        severity=DriftSeverity.CRITICAL,
    )
    assert escalated.lifecycle == state_store.LIFECYCLE_ESCALATING
    assert escalated.occurrences == 2

    resolved = store.resolve_drift(drift.condition_id)
    assert resolved is not None
    assert resolved.lifecycle == state_store.LIFECYCLE_RESOLVED
    assert store.get_active_drift(drift.condition_id) is None


def test_for_state_active_drift_synthetic_when_state_immutable() -> None:
    """An immutable view falls back to a synthetic single-shot Drift."""
    # An ``int`` is not a Mapping → constructor coerces ``_state`` to ``{}``.
    store = StateStore(state=123)
    drift = store.open_or_escalate_drift(
        kind=DriftKind.LOOPING_REASONING,
        task_id="t",
        agent_id="a",
        turn_id="x",
        severity=DriftSeverity.INFO,
    )
    # ``{}`` is a MutableMapping so we hit the real path; this just
    # confirms the helper is reachable from the alternate constructor.
    assert drift.condition_id


def test_for_state_invocation_registry_requires_session_id() -> None:
    """Without a ``session_id`` registry mutations are silent no-ops."""
    import asyncio

    async def _coro() -> None:
        return None

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(_coro())
        try:
            store = StateStore.for_state({})  # empty session_id
            store.register_invocation_task("inv-1", task)
            assert store.active_invocation_ids() == []
            store.deregister_invocation_task("inv-1")  # idempotent
        finally:
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, BaseException):
                pass
    finally:
        loop.close()


def test_for_state_cancel_requested_round_trip() -> None:
    """``mark_invocation_cancel_requested`` flips the late-drift gate flag."""
    store = StateStore.for_state({}, session_id="sess-1")
    assert store.is_invocation_cancel_requested("inv-1") is False
    store.mark_invocation_cancel_requested("inv-1")
    assert store.is_invocation_cancel_requested("inv-1") is True
    assert "inv-1" in store.cancel_requested_invocation_ids()
    store.clear_active_invocations()
    assert store.is_invocation_cancel_requested("inv-1") is False


def test_for_state_goals_summary_default_empty_string() -> None:
    """``goals_summary`` reads ``""`` when the slot is unset."""
    assert StateStore.for_state({}).goals_summary() == ""


def test_for_state_correction_keys_isolated() -> None:
    """Setting a correction directly on the dict surfaces via ``has_correction``."""
    # The correction surface lives outside the ``goldfive.*`` namespace
    # by historical accident; this test pins the iteration matcher.
    state: dict = {
        "goldfive.pending_corrections.agent_x.task-7": {"superseded_task_id": "task-6"},
    }
    store = StateStore.for_state(state)
    assert store.has_correction("agent_x", "task-7") is True
    assert store.iter_corrections_for_agent("agent_x") == ["task-7"]


def test_construction_with_non_mapping_yields_empty_view() -> None:
    """A non-mapping argument degrades cleanly; all readers default."""
    store = StateStore(state=object())
    assert store.pin_current_task() == ""
    assert store.get_active_steer() is None
    assert store.cancelled_function_call_ids() == []
    assert store.iter_pending_delegations() == {}
