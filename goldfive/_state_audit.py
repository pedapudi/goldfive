"""Debug-mode runtime assertion for the state-ownership contract.

Phase 0 of goldfive#271. See ``docs/design/STATE-OWNERSHIP-CONTRACT.md``.

Background
----------

ADK considers writes to ``Session.state`` exclusive to its own
``session_service.append_event`` machinery. Every direct ``state[key] = v``
that goldfive performs from inside an ADK callback path races with
ADK's optimistic-concurrency contract — the symptom in production is
the stale-session ``ValueError`` documented in goldfive#275.

This module installs an opt-in guard that detects new violations of
the rule "goldfive callbacks do not mutate ADK ``session.state``". The
existing violations are enumerated in the audit catalog of the design
doc; each catalogued site is pre-listed in :data:`_KNOWN_CALLERS` so
the existing test suite stays green. A NEW write at an un-listed
``(file, function)`` location raises :class:`StateOwnershipViolation`.

When does it fire?
------------------

By default:

* **Production:** off. ``GOLDFIVE_STRICT_STATE_OWNERSHIP`` is unset on
  fresh deploys; nothing happens.
* **Tests:** on. ``tests/conftest.py`` enables it via the auto-applied
  ``_state_audit_enabled`` fixture, so every pytest run with the
  fixture loaded enforces the rule.

Override via env var:

* ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1`` — force on everywhere.
* ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0`` — force off everywhere.
* unset — caller controls (test fixture vs production default).

Mechanism
---------

Phase 0 leaves the existing call sites intact (no behaviour change).
The guard runs by:

1. Patching :func:`goldfive.adapters._adk_state_protocol._set` (the
   single funnel for protocol-module writes) to call
   :func:`_check_caller` before performing the write.
2. Exposing :func:`assert_can_write` for any future site that wants
   to check directly without going through the protocol module.

When :func:`_check_caller` fires it walks the call stack looking for
a frame whose ``(filename, function)`` matches an entry in
:data:`_KNOWN_CALLERS`. If one is found, the write is allowed. If
none match — the write is being attempted from an un-catalogued site
— the guard raises :class:`StateOwnershipViolation`.

Each catalog entry in :data:`_KNOWN_CALLERS` corresponds to one entry
in the §5 audit catalog of the design doc. As Phase 2 migrates a
violation, its entry is removed from this list AND from the catalog
in lockstep. When the list is empty + the bridge is gone, the
migration is complete.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("goldfive._state_audit")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StateOwnershipViolation(BaseException):
    """Raised when a goldfive callback writes to ADK ``session.state`` from
    an un-catalogued (file, function) location.

    The message records the offending key + the goldfive callback the
    write fired inside + a suggested migration target. See
    ``docs/design/STATE-OWNERSHIP-CONTRACT.md`` §3.3.

    **Inherits from :class:`BaseException`, NOT :class:`Exception`.**
    Many of the catalogued sites are inside broad ``try / except
    Exception`` blocks (the state writes are best-effort by design, so
    a missing key shouldn't crash the run). If the violation inherited
    from ``Exception`` those defensive blocks would swallow it
    silently — which is exactly the failure mode in production today
    (the #275 stale-session ValueError IS getting raised by ADK's
    optimistic-concurrency lock and IS being caught by the same
    defensive blocks). The audit's value is in surfacing the
    violation loudly in tests; making it a ``BaseException`` ensures
    only ``except BaseException`` (or unwrapped propagation) catches
    it.
    """


class CancellationStashViolation(BaseException):
    """Raised when ``CancelledError`` propagates through a stash-owning await
    without entering a ``finally`` block.

    goldfive#271 Phase 3 Addition A. Validation v2 found that
    ``runner.py:411``'s ``try / except Exception`` block silently
    bypassed all post-execution housekeeping when ADK closed the
    runner mid-stream — the executor's broad ``except Exception``
    let ``CancelledError`` skate past every state-stash, every event
    flush, every drift-detector teardown, because ``CancelledError``
    has been a ``BaseException`` subclass since Python 3.8 and is NOT
    caught by ``except Exception``.

    **Inherits from :class:`BaseException`** for the same reason as
    :class:`StateOwnershipViolation`: the defensive ``try / except
    Exception`` blocks the audit catalogues would otherwise swallow
    the violation marker.

    Phase 3 shipped only the EXCEPTION CLASS + the audit document under
    ``docs/design/CANCELLATION-CONTRACT.md`` (sibling to
    ``STATE-OWNERSHIP-CONTRACT.md``). Phase 3.5 wires the runtime
    tripwire: every audited stash-owning site (§C1-C6) is wrapped in
    :func:`cancellation_stash_audited`. Each site's own
    :class:`_AuditedSite.__exit__` checks its marker on exit and
    raises this class — chained onto the original ``CancelledError``
    as ``__cause__`` — when the block exited via a non-``Exception``
    ``BaseException`` without the compliance branch having called
    :func:`mark_stash_completed`. (Earlier shapes did the assertion
    centrally from :meth:`ADKAdapter._invoke_internal`'s catch arm;
    goldfive#326 moved the assertion to per-site exit because the
    central walk fired before any outer ``finally`` block had a
    chance to run.)

    **Default-off contract.** The tripwire is opt-in via
    ``GOLDFIVE_STRICT_STATE_OWNERSHIP`` (the same env gate as
    :class:`StateOwnershipViolation`). Production deploys with the
    variable unset never pay more than a per-site contextvar
    push/pop; the boundary check itself short-circuits on
    :func:`is_enabled` returning False before walking any state.
    """


@dataclass(frozen=True)
class _CallbackFrame:
    """Bookkeeping for an active goldfive callback.

    Set by :func:`goldfive_callback` at the entry of each plugin
    callback method; cleared at exit. The audit guard consults this to
    distinguish "write from inside a goldfive callback" (governed by
    the contract) from "write outside any callback" (always allowed —
    e.g. orchestration writes from ``Runner.run``'s setup phase).
    """

    name: str  # e.g. "before_run_callback"


# ContextVar so concurrent invocations (parallel sub-Runners) each have
# their own active-frame slot. ``None`` means "no goldfive callback is
# currently active" — writes are unrestricted.
_active_callback: contextvars.ContextVar[_CallbackFrame | None] = contextvars.ContextVar(
    "goldfive_active_callback",
    default=None,
)

# ContextVar carrying an ad-hoc "expected violation" tag set by callers
# that want to suppress the guard for a known-safe block (e.g. tests
# that drive a violation deliberately to assert a downstream effect).
_expected_violation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "goldfive_expected_violation",
    default=None,
)


def is_enabled() -> bool:
    """Return whether the tripwire is currently enabled.

    Resolution order:

    1. ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1`` -> True.
    2. ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0`` -> False.
    3. Anything else (unset, ``auto``) -> True iff a pytest run is
       loaded (heuristic: ``pytest`` in ``sys.modules``). Production
       deploys never import pytest, so this defaults to off there.
    """
    raw = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Auto: default-on under pytest, default-off otherwise.
    return "pytest" in sys.modules


def enable() -> None:
    """Force-enable the tripwire for the current process.

    Equivalent to ``GOLDFIVE_STRICT_STATE_OWNERSHIP=1``. The patch is
    idempotent — calling :func:`enable` twice is a no-op the second
    time.
    """
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"
    _install_protocol_patch()


def disable() -> None:
    """Force-disable the tripwire for the current process.

    Equivalent to ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0``. Existing
    patches stay installed (uninstalling cleanly is harder than
    leaving the patched ``_set`` to consult :func:`is_enabled` on
    every call); the patch becomes a pass-through.
    """
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "0"


@contextmanager
def goldfive_callback(name: str) -> Iterator[None]:
    """Context manager — mark the wrapped block as a goldfive callback.

    Set at the top of every plugin callback method. Writes to ADK
    state inside this block are governed by the contract; writes
    outside it are unrestricted (they are presumably happening from
    setup paths that ADK itself isn't observing yet).

    Composes with :func:`expect_violation` for tests that drive a
    catalogued violation deliberately.
    """
    token = _active_callback.set(_CallbackFrame(name=name))
    try:
        yield
    finally:
        _active_callback.reset(token)


@contextmanager
def expect_violation(reason: str) -> Iterator[None]:
    """Context manager — declare a planned violation in the wrapped block.

    Tests use this to drive a deliberate write through a catalogued
    site (e.g. asserting that disabling the tripwire allows a write
    that would otherwise raise). Production code MUST NOT use this —
    the catalog is the only legitimate suppression mechanism, and
    Phase 2 migrations remove catalog entries rather than wrap call
    sites in :func:`expect_violation`.
    """
    token = _expected_violation.set(reason)
    try:
        yield
    finally:
        _expected_violation.reset(token)


def assert_can_write(
    state: Any,
    key: str,
    *,
    surface: str = "adk",
) -> None:
    """Raise :class:`StateOwnershipViolation` for un-catalogued writes.

    Public entry point for code paths that perform an ADK
    ``session.state`` mutation outside the ``_adk_state_protocol``
    funnel (e.g. inline ``state[k] = v`` at a call site). The current
    Phase-0 catalog has a few such sites (V2's ``adk_state[k] = v``
    inline, V3-V5's inline subscript writes, V7-V8's
    ``SESSION_CONTEXT_STATE_KEY`` stamping); in Phase 0 they remain
    untouched and the guard relies on :func:`_check_caller` to
    recognise their (file, function) signature.

    This helper exists so Phase 2 migrations have a single place to
    add ``state_audit.assert_can_write(...)`` calls when refactoring
    a site that doesn't go through the protocol funnel.

    Surface ``adk`` is checked against the ADK contract. Other
    surfaces (e.g. ``goldfive``) are no-ops for now — placeholder for
    future contracts that govern goldfive-side writes too.
    """
    if surface != "adk":
        return
    _check_caller(key=key, surface=surface, state_repr=_state_repr(state))


# ---------------------------------------------------------------------------
# Request-side mutation catalog (goldfive#397)
# ---------------------------------------------------------------------------
#
# The ``llm_request.contents`` list reaches the model via ADK's flow
# AFTER every ``before_model_callback`` has returned. Phase 0/2 of the
# state-ownership audit covers ADK ``session.state`` writes only —
# request-side mutation is NOT currently guarded by a runtime tripwire
# because there is no single funnel (each ADK plugin can mutate
# ``llm_request.contents`` independently).
#
# The :class:`~goldfive.context_editor.ContextEditor` is the ONE goldfive
# site authorised to mutate ``llm_request.contents``. This holds even
# for the PR-6b byte-monotonic-replace rules
# (``PruneTransientErrorRule`` redacts a transient-error payload;
# ``CompactPriorReasoningRule`` summarizes a collapsed run): each rule
# returns a NEW list built from ``copy.deepcopy``'d ``Content`` objects
# and NEVER mutates the live ``contents`` list or its objects in place —
# only ``ContextEditor.apply`` swaps the ``llm_request.contents``
# reference, and only after the invariant chain passes. Every other
# goldfive code path that touches the field is strictly read-only:
#
# * ``goldfive/adapters/adk_llm_instrumentation.py::_measure_request_chars``
#   — reads ``contents`` to instrument ``llm.request.chars`` and
#   ``llm.request.messages_count``. Never writes. Re-exported from
#   :mod:`goldfive.adapters._adk_plugin` for backwards compatibility.
# * ``goldfive/context_editor.py::_content_bytes`` — same
#   read-only character measurement, shared so the editor's emitted
#   ``ContextEdited.bytes_before`` / ``.bytes_after`` are byte-aligned
#   with the instrumentation line.
#
# The catalog entry below is documentation; no runtime check fires.
# Extending the audit to a runtime list-mutation tripwire is a
# follow-up (would need to wrap the ``contents`` list in a tracked
# proxy, which has measurable overhead in the hot path — left to a
# focused PR if a regression surfaces).
_REQUEST_CONTENTS_AUTHORISED_SITES: tuple[tuple[str, str], ...] = (
    ("goldfive/context_editor.py", "ContextEditor.apply"),
)


# ---------------------------------------------------------------------------
# Catalog of known callers (Phase 0 — pre-existing violations)
# ---------------------------------------------------------------------------


# Each entry corresponds to a row in the §5 audit catalog of
# ``docs/design/STATE-OWNERSHIP-CONTRACT.md``. As Phase 2 migrates a
# site, the entry is removed from this set AND from the catalog. When
# the set is empty (and the bridge code itself is gone), the
# migration is complete.
#
# Format: ``(callable_filename_suffix, callable_qualname_suffix)``.
# The suffix-match keeps the catalog robust against absolute-path
# differences across worktrees / CI environments. ``filename`` is
# matched by ``str.endswith`` against the frame's filename; qualname
# is matched as a substring against ``frame.f_code.co_qualname``
# (Python 3.11+) or the function name on older runtimes.
_KNOWN_CALLERS: frozenset[tuple[str, str]] = frozenset(
    {
        # V1 / V2 / V5 — MIGRATED in Phase 2.0 (goldfive#271). The
        # before_run_callback initial seed, the orchestration-state
        # bridge, and the before_model_callback duplicate seed all
        # deleted. The dynamic-instruction resolver and GoldfivePlanner
        # read goldfive Session directly via the SessionContext stash.
        # V3 / V4 — MIGRATED in Phase 2.1 (goldfive#271). The per-agent
        # pin (``_stamp_current_task_id``) and the delegation-site pin
        # (``_pin_delegation_task_id``) now write only to goldfive
        # ``Session.state`` via :class:`StateStore`. The
        # readers (the dynamic-instruction resolver, reporting handlers,
        # ``_resolve_pinned_task_id``) consult goldfive Session via the
        # plugin reference (``session_context_from_invocation``) — no
        # callback-time write to ADK ``session.state`` from inside the
        # wrap remains.
        # V7 / V8 — ADKAdapter.invoke stashes / clears
        # SESSION_CONTEXT_STATE_KEY before / after run_async. Outside
        # any callback frame — listed here for completeness; the
        # callback-frame check normally short-circuits writes from
        # outside callbacks regardless.
        ("goldfive/adapters/adk.py", "invoke"),
        ("goldfive/adapters/adk.py", "_invoke_internal"),
        # The protocol module's writers themselves: every helper
        # funnels through ``_set``, which the patch wraps. The
        # _set frame appears between the call site and the dict
        # mutation, so we catalog it too — otherwise the stack walk
        # might stop at _set instead of reaching the real caller.
        ("goldfive/adapters/_adk_state_protocol.py", "_set"),
        ("goldfive/adapters/_adk_state_protocol.py", "write_cancel_request"),
        ("goldfive/adapters/_adk_state_protocol.py", "register_invocation_parent"),
        ("goldfive/adapters/_adk_state_protocol.py", "consume_cancel_request"),
        # Tests legitimately drive the protocol module against fake
        # state dicts. Every test file under tests/ is allowed to
        # invoke writers — the contract is about goldfive callback
        # paths, not about test scaffolding.
        ("tests/", ""),
    }
)


def known_callers_count() -> int:
    """Return the count of catalogued opt-out entries.

    Phase 2 migrations should monotonically decrease this number. A
    Phase-2 PR that doesn't change this count is suspect — either
    it didn't migrate anything or it added a new violation.
    """
    return len(_KNOWN_CALLERS)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _state_repr(state: Any) -> str:
    """Return a short repr for a state dict suitable for error messages."""
    try:
        return f"{type(state).__name__}@{hex(id(state))}"
    except Exception:  # noqa: BLE001 — defensive
        return "<state>"


def _frame_matches(frame: Any, filename_suffix: str, qualname_suffix: str) -> bool:
    """Return True if ``frame`` matches a (filename, qualname) pair."""
    try:
        fname = str(getattr(frame, "f_code", None).co_filename or "")
    except Exception:  # noqa: BLE001
        return False
    if not fname.endswith(filename_suffix):
        # Cross-platform path separators: also try forward-slash form.
        if not fname.replace("\\", "/").endswith(filename_suffix):
            return False
    if not qualname_suffix:
        return True
    code = getattr(frame, "f_code", None)
    qualname = getattr(code, "co_qualname", None) or getattr(code, "co_name", "") or ""
    return qualname_suffix in str(qualname)


def _stack_has_known_caller() -> bool:
    """Walk the live call stack looking for a catalogued caller frame."""
    frame = inspect.currentframe()
    if frame is None:
        # No introspection available; conservative default = allow so
        # we never crash production unexpectedly.
        return True
    # Skip the immediate two frames (this function + _check_caller).
    walker = frame.f_back
    if walker is not None:
        walker = walker.f_back
    seen = 0
    # Cap the walk depth — pathological deep stacks would otherwise
    # be expensive on the hot path. 64 frames is generous; the
    # deepest catalogued caller is ~5 frames from the leaf write.
    while walker is not None and seen < 64:
        for fname_sfx, qname_sfx in _KNOWN_CALLERS:
            if _frame_matches(walker, fname_sfx, qname_sfx):
                return True
        walker = walker.f_back
        seen += 1
    return False


def _check_caller(*, key: str, surface: str, state_repr: str) -> None:
    """Raise if the live call stack has no catalogued caller frame.

    Called from the patched protocol ``_set`` and from the public
    :func:`assert_can_write` entry point. Conservative: short-circuits
    cheaply when the tripwire is disabled or no goldfive callback is
    currently active, so the steady-state cost is one ContextVar read
    + one env-var lookup.
    """
    if not is_enabled():
        return
    if _expected_violation.get() is not None:
        # A test (or other caller) has explicitly declared this
        # violation as expected. Allow.
        return
    active = _active_callback.get()
    if active is None:
        # Write happening outside a goldfive callback frame. The
        # contract is about callback-path mutations specifically;
        # writes from setup / teardown / orchestration paths are the
        # whole point and are unrestricted.
        return
    if _stack_has_known_caller():
        return
    # No catalogued match — the write is from a new (file, function)
    # site. Raise with an actionable message.
    raise StateOwnershipViolation(
        f"goldfive callback {active.name!r} attempted to mutate ADK "
        f"session.state[{key!r}] (state={state_repr}) from a site "
        f"not listed in goldfive._state_audit._KNOWN_CALLERS.\n"
        f"\n"
        f"Phase 0 of goldfive#271: writes to ADK session.state from "
        f"inside a goldfive callback violate the state-ownership "
        f"contract (see docs/design/STATE-OWNERSHIP-CONTRACT.md).\n"
        f"\n"
        f"If this is a NEW write: don't add it. Use "
        f"goldfive.state_store.write(session.state, ...) "
        f"to update the goldfive-owned dict instead, and let the "
        f"bridge propagate it to ADK.\n"
        f"\n"
        f"If this IS a catalogued site that was missed by the audit: "
        f"add the (file, function) pair to _KNOWN_CALLERS and update "
        f"the catalog in the design doc to match.\n"
        f"\n"
        f"To suppress in tests, wrap the call in "
        f"`with goldfive._state_audit.expect_violation('reason'):`."
    )


_INSTALLED = False


def _install_protocol_patch() -> None:
    """Patch ``_adk_state_protocol._set`` to call :func:`_check_caller`.

    Idempotent. The patch wraps the original ``_set`` rather than
    replacing it so the goldfive-prefix assertion keeps firing on
    typo'd keys.
    """
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return
    try:
        from goldfive.adapters import _adk_state_protocol as sp  # noqa: PLC0415
    except ImportError:
        # ADK adapter optional-dep group not installed; nothing to patch.
        return
    original_set = sp._set

    # Look up _check_caller through the module globals so that test
    # code patching the symbol on the module is honoured (the
    # closure-bound reference would otherwise be frozen at install
    # time).
    this_module = sys.modules[__name__]

    def guarded_set(state: Any, key: str, value: Any) -> None:  # type: ignore[no-untyped-def]
        this_module._check_caller(  # type: ignore[attr-defined]
            key=key, surface="adk", state_repr=_state_repr(state)
        )
        original_set(state, key, value)

    guarded_set.__wrapped__ = original_set  # type: ignore[attr-defined]
    sp._set = guarded_set  # type: ignore[attr-defined]
    _INSTALLED = True


# The 8 callback methods on the goldfive ADK plugin. Wrapped at plugin-
# construction time by :func:`wrap_plugin_callbacks` so a write inside
# any of them sets the active-callback ContextVar, which the guard
# consults to decide whether the contract applies.
_PLUGIN_CALLBACK_METHODS: tuple[str, ...] = (
    "before_run_callback",
    "before_agent_callback",
    "before_model_callback",
    "before_tool_callback",
    "after_tool_callback",
    "after_model_callback",
    "after_agent_callback",
    "after_run_callback",
)


def wrap_plugin_callbacks(plugin: Any) -> Any:
    """Wrap each goldfive plugin callback in a :func:`goldfive_callback` block.

    Called once from :func:`goldfive.adapters._adk_plugin.make_adk_plugin`
    before the plugin instance is returned. The wrapping is structural
    — entry sets ``_active_callback``, exit clears it — and is unguarded
    by ``is_enabled()`` because the active-callback ContextVar is also
    consulted by the protocol-module patch and any future direct
    :func:`assert_can_write` callers; keeping it set unconditionally
    (cost = one ContextVar set per callback) is cheaper than gating
    every site on a separate env-var read.

    Idempotent: a method already wrapped (carries a ``__goldfive_audit_wrapped__``
    attribute) is left alone.

    Returns the plugin so the call can be inlined with the construction
    path: ``return wrap_plugin_callbacks(_GoldfiveADKPlugin())``.
    """
    cls = type(plugin)
    for method_name in _PLUGIN_CALLBACK_METHODS:
        original = getattr(cls, method_name, None)
        if original is None or getattr(original, "__goldfive_audit_wrapped__", False):
            continue

        def _wrap(method: Any, name: str) -> Any:
            async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                with goldfive_callback(name):
                    return await method(self, *args, **kwargs)

            wrapper.__goldfive_audit_wrapped__ = True  # type: ignore[attr-defined]
            wrapper.__name__ = method.__name__
            wrapper.__qualname__ = method.__qualname__
            wrapper.__wrapped__ = method  # type: ignore[attr-defined]
            return wrapper

        setattr(cls, method_name, _wrap(original, method_name))
    return plugin


# Install the patch at import time. The guard inside ``_check_caller``
# short-circuits when the tripwire is disabled, so this is cheap and
# always safe — production deploys with the env var unset never pay
# more than one ContextVar read per protocol write.
_install_protocol_patch()


# ---------------------------------------------------------------------------
# Cancellation-stash tripwire (Phase 3.5)
# ---------------------------------------------------------------------------
#
# The mechanism: every audited site (C1-C6 in
# ``docs/design/CANCELLATION-CONTRACT.md``) wraps its stash-owning
# block in :func:`cancellation_stash_audited`. The context manager
# pushes a marker on entry. Compliant sites (``try / finally`` per
# §1.1, or ``except BaseException: stash; raise`` per §1.2) call
# :func:`mark_stash_completed` from inside their stash branch BEFORE
# the block exits — the marker records that the compliance branch
# fired.
#
# Per-site assertion (goldfive#326). Each audited site checks its OWN
# marker on ``__exit__``: if the block exits via a ``BaseException``
# that isn't an :class:`Exception` (i.e. ``CancelledError`` or kin) AND
# the marker's ``completed`` flag is still False, ``__exit__`` raises
# :class:`CancellationStashViolation` chained onto the original
# ``BaseException``. The assertion fires at the right frame: the
# outer site's ``__exit__`` runs AFTER its ``finally`` block, so a
# correctly-instrumented outer site (whose ``finally`` calls
# :func:`mark_stash_completed`) sees ``completed == True`` and does
# not raise — even when an inner ``except CancelledError`` arm in a
# deeper frame would have made a centralised stack walk see the outer
# marker as still-uncompleted. (See goldfive#326 for the regression
# this fix addresses.)
#
# Default-off: every entry point short-circuits on
# :func:`is_enabled` so production deploys (env var unset, no pytest
# loaded) pay one ContextVar read per site enter/exit and nothing
# more.


@dataclass
class _StashMarker:
    """Per-audited-site bookkeeping for the cancellation tripwire.

    The ``completed`` flag is set by :func:`mark_stash_completed`
    when the audited site's compliance branch fires (its ``finally``
    block per §1.1, or its ``except BaseException`` arm per §1.2).
    The site's own ``__exit__`` reads the flag and raises
    :class:`CancellationStashViolation` if the block exits via a
    non-``Exception`` ``BaseException`` (i.e. ``CancelledError``)
    without the flag set.
    """

    name: str
    completed: bool = False


# Stack of currently-open audited sites. A list-valued ContextVar so
# nested audited sites layer correctly across asyncio task boundaries
# (a sub-task inherits the parent's stack snapshot at spawn time, so
# its own ``cancellation_stash_audited`` enters push onto its own
# private list).
_open_stash_markers: contextvars.ContextVar[tuple[_StashMarker, ...]] = (
    contextvars.ContextVar("goldfive_open_stash_markers", default=())
)


class _AuditedSite:
    """Context manager for a Phase-3.5 audited stash-owning site.

    On entry: pushes a fresh :class:`_StashMarker` onto the
    open-markers stack.

    On exit (goldfive#326): the marker is always popped. If the
    block exits via a :class:`BaseException` that isn't an
    :class:`Exception` (i.e. :class:`asyncio.CancelledError` and
    other non-``Exception`` ``BaseException`` subclasses) AND the
    marker's ``completed`` flag is still False — the site never
    called :func:`mark_stash_completed` from a compliance branch —
    ``__exit__`` raises :class:`CancellationStashViolation` chained
    onto the original cancel as ``__cause__``. Normal returns and
    plain :class:`Exception` propagation pop the marker without
    raising.

    Why per-site? The earlier design did a single boundary-side
    walk in :func:`assert_stash_invariant`, called from the
    ``except CancelledError`` arm at
    :meth:`ADKAdapter._invoke_internal`. That walk fired before any
    outer ``finally`` block had a chance to run — control was still
    inside the deeper catch frame — so an outer audited site whose
    ``finally`` correctly stashes was nonetheless flagged as
    "started but never compliance-marked." goldfive#326 documents
    the v19 regression. Per-site assertion fixes the frame
    ordering: the outer ``__exit__`` runs AFTER its ``finally``,
    sees ``completed == True``, and does not raise.

    Implementation note. We rebuild the marker-stack tuple by
    identity (``self._marker`` is unique per ``__enter__``) rather
    than using ``ContextVar.reset(token)`` so nested audited sites
    layer correctly even if a sibling-site context shifted the
    stack between enter and exit.
    """

    __slots__ = ("_marker", "_active")

    def __init__(self, name: str) -> None:
        self._marker = _StashMarker(name=name, completed=False)
        self._active = False

    def __enter__(self) -> None:
        if not is_enabled():
            return
        self._active = True
        _open_stash_markers.set((*_open_stash_markers.get(), self._marker))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        if not self._active:
            return False
        # Pop ourselves off the marker stack regardless of exit
        # shape. The per-site assertion below decides whether to
        # raise; the stack mutation is unconditional so a
        # subsequent unrelated audited site doesn't see a stale
        # marker.
        current = _open_stash_markers.get()
        rebuilt = tuple(m for m in current if m is not self._marker)
        if rebuilt != current:
            _open_stash_markers.set(rebuilt)
        # Per-site assertion (goldfive#326). The contract surfaces
        # the ``except Exception`` blind spot — a stash-owning
        # ``await`` whose surrounding broad catch lets
        # ``CancelledError`` skate past without running the
        # compliance branch. The non-``Exception`` ``BaseException``
        # filter matches that exact bug shape; plain ``Exception``
        # propagation does not need the assertion (it would have
        # been caught by the broad catch).
        if exc is None or isinstance(exc, Exception):
            return False
        if not isinstance(exc, BaseException):
            return False
        if self._marker.completed:
            return False
        # Bypass detected. Build the diagnostic frame chain so the
        # operator can see WHICH audited site observed the
        # violation — cheap, only walks when we're already raising.
        raise _build_violation(self._marker.name, cause=exc)


def cancellation_stash_audited(name: str) -> _AuditedSite:
    """Mark the wrapped block as a Phase-3.5 audited stash-owning site.

    Returns a context manager (:class:`_AuditedSite`) wrapping the
    ``try / except / finally`` (or
    ``try / except Exception ... except BaseException: stash; raise``)
    that owns a state-stash duty across an ``await``. The wrapped
    block MUST call :func:`mark_stash_completed` from inside its
    stash branch (the ``finally`` block, or the
    ``except BaseException`` arm before ``raise``) so the site's
    own ``__exit__`` does not raise
    :class:`CancellationStashViolation` (goldfive#326).

    Default-off: when :func:`is_enabled` returns False the
    context manager is a pass-through (one ContextVar read on
    ``__enter__`` and nothing else).

    Example::

        async def _refine(...):
            with cancellation_stash_audited("ParallelDAGExecutor._refine"):
                try:
                    refined = await planner.refine(...)
                except BaseException:
                    await self._escalate_refine_failure_as_critical_drift(...)
                    mark_stash_completed()
                    raise
    """
    return _AuditedSite(name)


def mark_stash_completed() -> None:
    """Mark the innermost open audited site as having run its stash branch.

    Called from inside an audited site's compliance branch
    (the ``finally`` block per §1.1, or the
    ``except BaseException: stash; raise`` arm per §1.2) BEFORE the
    block exits / re-raises. The owning site's ``__exit__`` reads
    the ``completed`` flag to decide whether to raise
    :class:`CancellationStashViolation`.

    No-op when :func:`is_enabled` returns False or no audited site is
    currently open. Safe to call from non-cancellation paths (e.g.
    from a normal ``finally`` block — the marker just records that
    the stash fired regardless of exit shape).
    """
    if not is_enabled():
        return
    stack = _open_stash_markers.get()
    if not stack:
        return
    # Flag the innermost (top-of-stack) marker. Per-site assertion
    # (goldfive#326) means markers are popped on ``__exit__``, so
    # the top of the stack is always the innermost still-active
    # site — no ``exited`` filter required.
    stack[-1].completed = True


def _build_violation(
    site_name: str, *, cause: BaseException | None = None
) -> CancellationStashViolation:
    """Build a :class:`CancellationStashViolation` with diagnostic frame chain.

    Used by :class:`_AuditedSite.__exit__` when its own marker
    indicates a bypass. Walks up to 16 caller frames so the
    operator can see WHICH boundary observed the violation; the
    walk is cheap because it only fires on the raise path.
    """
    frames: list[str] = []
    walker = inspect.currentframe()
    if walker is not None:
        walker = walker.f_back  # skip ourselves
    seen = 0
    while walker is not None and seen < 16:
        code = getattr(walker, "f_code", None)
        if code is not None:
            qual = getattr(code, "co_qualname", None) or getattr(code, "co_name", "")
            fname = str(getattr(code, "co_filename", ""))
            frames.append(f"  {fname}:{walker.f_lineno} {qual}")
        walker = walker.f_back
        seen += 1
    msg = (
        "CancelledError propagated past Phase-3.5 audited stash "
        "site(s) without entering the compliance branch:\n"
        f"  - {site_name}"
        + "\n\nExit-site frame chain:\n"
        + "\n".join(frames)
        + "\n\nEach audited site MUST call "
        "goldfive._state_audit.mark_stash_completed() from its "
        "``finally`` block (CANCELLATION-CONTRACT.md §1.1) or its "
        "``except BaseException: stash; raise`` arm "
        "(CANCELLATION-CONTRACT.md §1.2) BEFORE the cancel "
        "propagates. See docs/design/CANCELLATION-CONTRACT.md."
    )
    violation = CancellationStashViolation(msg)
    if cause is not None:
        violation.__cause__ = cause
    return violation


def assert_stash_invariant(*, cause: BaseException | None = None) -> None:  # noqa: ARG001
    """Deprecated no-op retained for backward compatibility.

    Earlier versions of Phase 3.5 walked the open-marker stack from
    a central catch site (``ADKAdapter._invoke_internal``'s
    ``except CancelledError`` arm) to assert each audited site ran
    its compliance branch before the cancel propagated past it.
    goldfive#326 documents why that shape was wrong: the central
    walk fires while control is still inside the deeper catch
    frame, BEFORE any outer audited site's ``finally`` block has
    had a chance to call :func:`mark_stash_completed`. The walk
    therefore mis-flagged correctly-instrumented outer sites as
    bypassing.

    The fix moves the assertion to :class:`_AuditedSite.__exit__`,
    where it fires at the right frame: after the site's own
    ``finally`` block. This function remains as a callable stub so
    older external call sites (and any third-party code that
    imported the symbol) keep working without churn; new code MUST
    NOT rely on this function for correctness — the per-site
    ``__exit__`` assertion is the canonical mechanism.

    The ``cause`` parameter is preserved for signature compatibility
    only.
    """
    return None
