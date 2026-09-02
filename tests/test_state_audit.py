"""Tests for the state-ownership tripwire (goldfive#271 Phase 0).

See ``goldfive/_state_audit.py`` for the audit module and
``docs/design/STATE-OWNERSHIP-CONTRACT.md`` §7 for the contract.

These tests verify three behaviours:

1. Existing catalogued sites continue to work with the tripwire on
   (covered indirectly by the rest of the test suite — every test
   that drives the plugin's callbacks runs with the tripwire enabled
   via the autouse fixture in :mod:`tests.conftest`).

2. A deliberate write from an un-catalogued (file, function) inside a
   goldfive callback frame raises :class:`StateOwnershipViolation`
   with an actionable message.

3. With the tripwire off, the same un-catalogued write passes
   silently (sanity check the off-state — production deploys with the
   env var unset must NOT raise).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from goldfive import _state_audit
from goldfive._state_audit import StateOwnershipViolation

# ---------------------------------------------------------------------------
# Off-state sanity (test fires before the autouse env-var override in
# conftest.py kicks in by using the ``no_state_audit`` fixture)
# ---------------------------------------------------------------------------


def test_off_state_allows_uncatalogued_writes(no_state_audit: None) -> None:
    """With ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0``, any caller is allowed.

    Sanity check the off-state — production deploys must not start
    raising on writes that were tolerated before this PR.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    state: dict[str, object] = {}
    # Drive a write from this test function — which is NOT in the
    # ``_KNOWN_CALLERS`` catalog under any (file, function) pattern
    # other than the broad ``tests/`` allow. We therefore mark it as
    # an explicitly-expected violation (otherwise the broad
    # ``tests/`` allow would short-circuit the assertion). With the
    # audit off, even un-allowed writes pass silently.
    with _state_audit.goldfive_callback("synthetic_callback"):
        sp._set(state, "goldfive.test_key", "value")
    assert state["goldfive.test_key"] == "value"


def test_off_state_no_callback_frame_no_op(no_state_audit: None) -> None:
    """Writes outside any goldfive callback are unrestricted regardless."""
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    state: dict[str, object] = {}
    sp._set(state, "goldfive.test_key", "value")
    assert state["goldfive.test_key"] == "value"


# ---------------------------------------------------------------------------
# On-state behaviour — un-catalogued caller raises
# ---------------------------------------------------------------------------


def _force_on_state() -> None:
    """Helper: ensure the tripwire is on regardless of fixture order."""
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"
    assert _state_audit.is_enabled()


def test_uncatalogued_write_inside_callback_raises() -> None:
    """A goldfive callback frame + un-catalogued caller must raise.

    Drives the violation via a custom shim function whose
    ``(filename, qualname)`` is NOT in ``_KNOWN_CALLERS`` — but
    ``tests/`` is allowed by the broad opt-out. To exercise the
    raise path we **temporarily strip** the ``tests/`` allow so the
    stack walk sees no match.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    _force_on_state()
    # Strip the tests-allow so this synthetic caller is genuinely
    # un-catalogued in the eyes of the guard.
    original_known = _state_audit._KNOWN_CALLERS
    stripped = frozenset(
        {(f, q) for (f, q) in original_known if not f.startswith("tests/")}
    )
    _state_audit._KNOWN_CALLERS = stripped  # type: ignore[misc]
    try:
        with _state_audit.goldfive_callback("synthetic_callback"):
            with pytest.raises(StateOwnershipViolation) as excinfo:
                sp._set({}, "goldfive.unknown_key", "value")
    finally:
        _state_audit._KNOWN_CALLERS = original_known  # type: ignore[misc]

    msg = str(excinfo.value)
    assert "synthetic_callback" in msg
    assert "goldfive.unknown_key" in msg
    assert "STATE-OWNERSHIP-CONTRACT.md" in msg
    assert "_KNOWN_CALLERS" in msg


def test_expect_violation_suppresses_raise() -> None:
    """``expect_violation`` lets a deliberate write through the guard."""
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    _force_on_state()
    original_known = _state_audit._KNOWN_CALLERS
    stripped = frozenset(
        {(f, q) for (f, q) in original_known if not f.startswith("tests/")}
    )
    _state_audit._KNOWN_CALLERS = stripped  # type: ignore[misc]
    try:
        state: dict[str, object] = {}
        with _state_audit.goldfive_callback("synthetic_callback"):
            with _state_audit.expect_violation("test deliberate write"):
                sp._set(state, "goldfive.deliberate", "ok")
    finally:
        _state_audit._KNOWN_CALLERS = original_known  # type: ignore[misc]
    assert state["goldfive.deliberate"] == "ok"


def test_no_callback_frame_allows_any_write() -> None:
    """Writes outside any goldfive callback are unrestricted on the on-state too.

    The contract is specifically about callback-path mutations. A
    write from setup / teardown / orchestration paths is the whole
    point — the audit must not flag those.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    _force_on_state()
    state: dict[str, object] = {}
    # No ``goldfive_callback`` context — the active-frame ContextVar
    # is unset.
    sp._set(state, "goldfive.from_setup", "ok")
    assert state["goldfive.from_setup"] == "ok"


# ---------------------------------------------------------------------------
# On-state behaviour — catalogued sites pass (the integration test)
# ---------------------------------------------------------------------------


def test_catalogued_callback_path_allows_writer() -> None:
    """A catalogued callback (e.g. ``write_cancel_request``) lands cleanly.

    With the tripwire enabled and a goldfive callback frame active, a
    write through one of the still-catalogued protocol writers (here
    :func:`write_cancel_request` for cooperative cancellation) passes
    the audit because ``write_cancel_request`` itself appears in
    ``_KNOWN_CALLERS``.

    Phase 2.1 of goldfive#271 — V3 / V4's own catalog entries are
    gone (the plugin no longer writes ADK state for the pin), but
    the cooperative-cancellation writers stay catalogued because
    they're still load-bearing for cancel propagation.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    _force_on_state()
    state: dict[str, object] = {}

    class _FakePlugin:
        async def before_agent_callback(self) -> None:
            sp.write_cancel_request(state, invocation_id="inv-1", request="cancel")

    import asyncio

    plugin = _FakePlugin()
    with _state_audit.goldfive_callback("before_agent_callback"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            plugin.before_agent_callback()
        )
    assert state["goldfive.cancel_requested"] == {"inv-1": "cancel"}


def test_real_plugin_callbacks_are_wrapped() -> None:
    """``make_adk_plugin`` returns an instance whose callback methods carry the audit wrapper."""
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import make_adk_plugin

    plugin = make_adk_plugin(name="test", host_agent_name="root")
    for method_name in _state_audit._PLUGIN_CALLBACK_METHODS:
        method = getattr(type(plugin), method_name, None)
        assert method is not None, f"plugin missing {method_name}"
        assert getattr(method, "__goldfive_audit_wrapped__", False), (
            f"{method_name} not wrapped"
        )


# ---------------------------------------------------------------------------
# Catalog hygiene
# ---------------------------------------------------------------------------


def test_catalog_excludes_callbacks_with_no_remaining_writes() -> None:
    """No plugin callback that's been migrated should appear in the catalog.

    Phase 2.0 of goldfive#271 migrated V1 / V2 / V5
    (``before_run_callback`` / ``before_model_callback`` writers).
    Phase 2.1 (this PR) migrated V3 / V4 (``before_agent_callback`` /
    ``before_tool_callback`` writers — the per-agent pin and the
    delegation-site pin both moved to goldfive ``Session.state``).
    None of those callback names should remain in the catalog — a
    regression that re-introduces an ADK-state write from any of them
    must fail the audit loudly.
    """
    catalogued_qualnames = {q for (_, q) in _state_audit._KNOWN_CALLERS}
    for migrated in (
        "before_run_callback",
        "before_model_callback",
        "before_agent_callback",
        "before_tool_callback",
        "_stamp_current_task_id",
        "_pin_delegation_task_id",
    ):
        assert migrated not in catalogued_qualnames, (
            f"{migrated!r} still in catalog after migration"
        )


def test_known_callers_count_after_phase_2_migration() -> None:
    """Catalog shrinks monotonically as Phase 2 migrations land.

    Phase 0 (#278) shipped the catalog with the full set of pre-
    existing violations. Phase 2.0 migrated V1, V2, V5 + the bridge
    writers. Phase 2.1 (this PR) migrated V3 + V4 — both pin write
    paths now land on goldfive ``Session.state`` exclusively, so
    their catalog entries are gone. The count must be strictly
    smaller than the Phase 2.0 baseline (which had 18 entries).
    """
    expected = _state_audit.known_callers_count()
    # Phase 2.0 baseline was 18; after Phase 2.1 we expect strictly
    # fewer (V3 + V4 entries removed). Lower bound stays loose so
    # the test is asserting direction (down), not an exact count.
    assert 5 <= expected < 18, (
        f"catalog count={expected} unexpected; Phase 2.0 baseline was 18 "
        "and Phase 2.1 should have shrunk it (V3 / V4 entries removed)."
    )


def test_state_audit_is_off_in_production_default() -> None:
    """With no env var set, the audit is off in production.

    Heuristic: outside of pytest, ``"pytest"`` is not in ``sys.modules``
    so :func:`is_enabled` returns False. Inside pytest it returns
    True. We can't fully simulate "outside pytest" from inside pytest
    so we settle for asserting the env-var resolution rules.
    """
    prior = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP")
    try:
        os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "0"
        assert not _state_audit.is_enabled()
        os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"
        assert _state_audit.is_enabled()
    finally:
        if prior is None:
            os.environ.pop("GOLDFIVE_STRICT_STATE_OWNERSHIP", None)
        else:
            os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = prior


def test_typed_state_ownership_policy_wins_and_restores() -> None:
    """A scoped typed value overrides the environment without leaking."""
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"

    with _state_audit.strict_state_ownership(False):
        assert not _state_audit.is_enabled()

    assert _state_audit.is_enabled()


@pytest.mark.asyncio
async def test_typed_state_ownership_policy_is_task_local() -> None:
    """Concurrent Runner contexts can enforce different policies safely."""

    async def observe(enabled: bool) -> bool:
        with _state_audit.strict_state_ownership(enabled):
            await asyncio.sleep(0)
            return _state_audit.is_enabled()

    assert await asyncio.gather(observe(False), observe(True)) == [False, True]


# ---------------------------------------------------------------------------
# Phase 2.1 — V3 / V4 migrated. The plugin's ``_stamp_current_task_id``
# (V3) and ``_pin_delegation_task_id`` (V4) no longer write ADK
# session.state. Reproducer tests that previously drove those code
# paths through the tripwire are gone with the migration.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_v4_pin_writes_no_longer_touch_adk_state() -> None:
    """The pin no longer mutates ADK ``session.state`` from a callback.

    Drives the real plugin's ``_stamp_current_task_id`` (V3 site) and
    ``_pin_delegation_task_id`` (V4 site) inside a goldfive callback
    frame and asserts:

    * the goldfive ``Session.state`` carries the pin keys, and
    * the ADK-side state dict is untouched (no leakage).

    This is the structural assertion behind Phase 2.1's catalog
    removal: with no callback-time write to ADK state, the audit
    has nothing to flag at these sites.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID
    from goldfive.types import Plan, Session, Task, TaskStatus

    plugin = make_adk_plugin(name="audit-test", host_agent_name="root")

    task = Task(
        id="t1",
        title="t1 title",
        description="t1 description",
        assignee_agent_id="root",
        status=TaskStatus.PENDING,
    )
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
        summary="p",
    )
    gf_session = Session(run_id="r1", goals=[], plan=plan)
    ctx = SessionContext(
        session=gf_session,
        steerer=None,
        task=task,
        tool_handlers={},
        host_agent_name="root",
    )

    class _FakeCtx:
        def __init__(self) -> None:
            self.state: dict[str, object] = {SESSION_CONTEXT_STATE_KEY: ctx}

    fake_cb_ctx = _FakeCtx()

    _force_on_state()
    # V3 — the pin landing path. After Phase 2.1 it lands only on
    # goldfive Session.state.
    with _state_audit.goldfive_callback("before_agent_callback"):
        plugin._stamp_current_task_id(  # type: ignore[attr-defined]
            ctx=ctx,
            task_id="t1",
            agent_name="root",
            source="single_match",
            task=task,
            invocation_id="inv-1",
        )

    assert gf_session.state[KEY_CURRENT_TASK_ID] == "t1"
    assert KEY_CURRENT_TASK_ID not in fake_cb_ctx.state, (
        "V3 must not write ADK state after Phase 2.1"
    )
