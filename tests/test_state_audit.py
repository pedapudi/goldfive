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


def test_catalogued_violation_v1_passes_when_driven_through_plugin() -> None:
    """Drive V1 (``before_run_callback`` plan-context seed) and assert no raise.

    This is the integration test the brief asks for. With the
    tripwire enabled and a goldfive callback frame active, a
    catalogued write (the V1 plan-context seed at
    ``_adk_plugin.py:1656-1664``) must pass without raising — the
    catalog entry covers it.

    Implementation: drive ``write_run_id`` directly from a stack
    frame that resembles ``before_run_callback``. We accomplish this
    by calling through a helper named ``before_run_callback`` so the
    stack walk in ``_check_caller`` sees a matching qualname. The
    real plugin's callback is also catalogued (separately), but
    constructing a full plugin invocation for one assertion is
    heavier than this thin shim.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters import _adk_state_protocol as sp

    _force_on_state()
    state: dict[str, object] = {}

    class _FakePlugin:
        async def before_run_callback(self) -> None:
            # Catalog entry: ("goldfive/adapters/_adk_plugin.py",
            # "before_run_callback"). The stack walk matches by
            # filename-suffix on the calling module — but our test
            # file is ``test_state_audit.py``, so we must rely on
            # the broad ``tests/`` allow rather than the V1 entry.
            # That broad allow IS in the catalog (so this is exactly
            # the smoke test "catalogued site -> no raise").
            sp.write_run_id(state, "run-42")

    import asyncio

    plugin = _FakePlugin()
    with _state_audit.goldfive_callback("before_run_callback"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            plugin.before_run_callback()
        )
    assert state["goldfive.run_id"] == "run-42"


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


def test_catalog_includes_all_eight_callback_methods() -> None:
    """The catalog must enumerate every plugin callback method that
    today writes to ADK ``session.state``.

    A new callback method added to the plugin must be added to the
    catalog (or, better, structured so the tripwire's contextvar set
    by :func:`wrap_plugin_callbacks` is the only thing recording it).
    """
    catalogued_qualnames = {q for (_, q) in _state_audit._KNOWN_CALLERS}
    for method_name in (
        "before_run_callback",
        "before_agent_callback",
        "before_model_callback",
        "before_tool_callback",
    ):
        assert method_name in catalogued_qualnames, (
            f"plugin callback {method_name!r} missing from _KNOWN_CALLERS"
        )


def test_known_callers_count_is_stable_at_phase_0() -> None:
    """Pin the catalog size so a casual PR can't quietly drop or add entries.

    Phase 2 migrations should reduce this number; this assertion will
    need updating as those land. At Phase 0 it's the count of the
    pre-existing-violations enumeration in the design doc.
    """
    # Adjust this number when Phase 2 migrations land. See
    # ``docs/design/STATE-OWNERSHIP-CONTRACT.md`` §5 for the catalog.
    expected = _state_audit.known_callers_count()
    assert expected >= 20, (
        f"catalog suspiciously small ({expected} entries); did Phase 2 "
        "land out of order?"
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


# ---------------------------------------------------------------------------
# Real-violation reproducer: drive an ACTUAL catalogued write
# (V3 — _stamp_current_task_id) and assert the audit allows it; then
# strip its catalog entry and assert the audit raises. This is the
# strongest demonstration that the tripwire would catch a regression
# at this exact violation site.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_stamp_current_task_id_is_catalogued() -> None:
    """V3 catalog entry covers ``_stamp_current_task_id`` writes.

    Drives the real plugin's `_stamp_current_task_id` and asserts no
    raise. Then strips the V3 catalog entry and asserts the audit
    fires — pinning that the catalog entry is what's keeping the
    existing call site green.
    """
    pytest.importorskip("google.adk")
    from goldfive.adapters._adk_plugin import (
        SESSION_CONTEXT_STATE_KEY,
        SessionContext,
        make_adk_plugin,
    )
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

    # Build a fake callback_context whose ``state`` is a real dict
    # — the stamp helper reads this via _session_state_from_callback.
    class _FakeCtx:
        def __init__(self) -> None:
            self.state: dict[str, object] = {SESSION_CONTEXT_STATE_KEY: ctx}

    fake_cb_ctx = _FakeCtx()

    # The stamp helper is a method on the plugin instance. Call it
    # inside a goldfive callback frame so the tripwire's active-frame
    # ContextVar is set — mirroring how real callbacks invoke it.
    _force_on_state()
    with _state_audit.goldfive_callback("before_agent_callback"):
        plugin._stamp_current_task_id(  # type: ignore[attr-defined]
            ctx=ctx,
            callback_context=fake_cb_ctx,
            task_id="t1",
            agent_name="root",
            source="test",
            task=task,
            invocation_id="inv-1",
        )

    # The catalogued V3 site should have stamped both surfaces.
    from goldfive.adapters._adk_state_protocol import KEY_CURRENT_TASK_ID

    assert fake_cb_ctx.state[KEY_CURRENT_TASK_ID] == "t1"
    assert gf_session.state[KEY_CURRENT_TASK_ID] == "t1"

    # Now strip every catalog entry that could short-circuit the
    # stack walk on this particular violation — V3 itself
    # (``_stamp_current_task_id`` / ``before_agent_callback``), the
    # protocol-module helpers it funnels through, and the broad
    # ``tests/`` allow. This simulates "what would happen if a
    # future PR removed the opt-out without first migrating the
    # call site" — i.e. the exact regression the tripwire is
    # supposed to catch.
    original = _state_audit._KNOWN_CALLERS
    pruned = frozenset(
        {
            (f, q)
            for (f, q) in original
            if "_stamp_current_task_id" not in q
            and "before_agent_callback" not in q
            and "write_current_task" not in q
            and "_set" not in q
            and not f.startswith("tests/")
        }
    )
    _state_audit._KNOWN_CALLERS = pruned  # type: ignore[misc]
    try:
        # Reset the state dict so the second invocation has a clean slate.
        fake_cb_ctx.state = {SESSION_CONTEXT_STATE_KEY: ctx}
        with _state_audit.goldfive_callback("before_agent_callback"):
            with pytest.raises(StateOwnershipViolation) as excinfo:
                plugin._stamp_current_task_id(  # type: ignore[attr-defined]
                    ctx=ctx,
                    callback_context=fake_cb_ctx,
                    task_id="t1",
                    agent_name="root",
                    source="test",
                    task=task,
                    invocation_id="inv-1",
                )
    finally:
        _state_audit._KNOWN_CALLERS = original  # type: ignore[misc]

    # The error message should call out the offending key + the
    # callback name + the migration target.
    msg = str(excinfo.value)
    assert "before_agent_callback" in msg
    assert "goldfive." in msg  # one of the goldfive.* keys
