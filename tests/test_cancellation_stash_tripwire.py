"""Phase 3.5 component 2 (goldfive#271): the cancellation-stash tripwire.

PR #290 added the :class:`goldfive._state_audit.CancellationStashViolation`
exception class but deferred the runtime check. PR #307 landed the
canonical ``CancelledError`` catch site at
:meth:`ADKAdapter._invoke_internal`. Phase 3.5 component 2 (this PR)
ties the two together: every audited stash-owning site listed in
:doc:`docs/design/CANCELLATION-CONTRACT.md` §C1-C6 is wrapped in
:func:`goldfive._state_audit.cancellation_stash_audited`, and the
catch site invokes :func:`assert_stash_invariant` so a site that
bypasses its compliance branch raises
:class:`CancellationStashViolation` (a ``BaseException`` so
``except Exception`` cannot swallow it).

These tests exercise the tripwire itself with synthetic audited
sites — the production C1-C5 instrumentation is exercised
end-to-end by :file:`tests/test_cancellation_stash_audit.py`. Here
we drive ``CancelledError`` at a synthetic ``await`` to verify:

* a synthetic site that DOES call :func:`mark_stash_completed`
  (compliant) lets the cancel propagate without raising the violation;
* a synthetic site that does NOT call :func:`mark_stash_completed`
  (bypass) raises :class:`CancellationStashViolation` chained onto
  the original ``CancelledError`` as ``__cause__``;
* the strict-mode default-off contract is preserved: with
  ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0`` the tripwire never fires
  regardless of whether the audited site is compliant.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import pytest

from goldfive import _state_audit
from goldfive._state_audit import (
    CancellationStashViolation,
    assert_stash_invariant,
    cancellation_stash_audited,
    mark_stash_completed,
)

# ---------------------------------------------------------------------------
# Fixture: ensure strict mode is the default for these tests, but let
# individual tests opt out.
# ---------------------------------------------------------------------------


@pytest.fixture
def strict_on() -> Iterator[None]:
    """Force the tripwire ON for the duration of one test."""
    prior = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP")
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("GOLDFIVE_STRICT_STATE_OWNERSHIP", None)
        else:
            os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = prior


@pytest.fixture
def strict_off() -> Iterator[None]:
    """Force the tripwire OFF for the duration of one test."""
    prior = os.environ.get("GOLDFIVE_STRICT_STATE_OWNERSHIP")
    os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = "0"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("GOLDFIVE_STRICT_STATE_OWNERSHIP", None)
        else:
            os.environ["GOLDFIVE_STRICT_STATE_OWNERSHIP"] = prior


# ---------------------------------------------------------------------------
# Synthetic boundary helper — mimics _invoke_internal's CancelledError
# catch arm without dragging in the full ADK adapter.
# ---------------------------------------------------------------------------


async def _run_with_boundary(coro_factory):
    """Run ``coro_factory()`` inside a synthetic boundary.

    Mirrors the catch site at
    :meth:`ADKAdapter._invoke_internal`: catches ``CancelledError``,
    invokes :func:`assert_stash_invariant` (chaining the cancel as
    ``__cause__``), then re-raises whichever exception came out.
    """
    try:
        return await coro_factory()
    except asyncio.CancelledError as exc:
        assert_stash_invariant(cause=exc)
        raise


# ---------------------------------------------------------------------------
# Test 1 — strict mode + bypass site → CancellationStashViolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bypass_site_under_strict_raises_violation(strict_on: None) -> None:
    """Synthetic site that does NOT mark the stash-completion flag.

    Drives ``CancelledError`` at the inner await; the audit context
    manager's ``__exit__`` runs and pops the marker, but
    :func:`mark_stash_completed` is never called. The boundary's
    :func:`assert_stash_invariant` should observe the bypass and
    raise :class:`CancellationStashViolation`.

    Note the violation MUST be a :class:`BaseException` (not an
    ``Exception``) so the existing ``try / except Exception``
    blocks the audit was created to surface cannot swallow it. We
    verify the ``isinstance`` shape explicitly.
    """

    async def bypass_call() -> None:
        with cancellation_stash_audited("synthetic.bypass"):
            try:
                # Simulate a CancelledError raised by the inner
                # ``await`` — the bug shape that motivated the
                # contract: a broad ``except Exception`` lets
                # ``CancelledError`` skate past, but no compliance
                # branch (no ``finally`` calling
                # ``mark_stash_completed``, no
                # ``except BaseException: stash; raise``) ran.
                raise asyncio.CancelledError("inner-await-cancelled")
            except Exception:  # noqa: BLE001
                # Broad catch that does NOT fire on CancelledError —
                # this body is only here to mirror the bug shape.
                pytest.fail(  # pragma: no cover — defensive
                    "except Exception should not catch CancelledError"
                )

    with pytest.raises(CancellationStashViolation) as exc_info:
        await _run_with_boundary(bypass_call)
    # The violation MUST inherit from BaseException, not Exception —
    # otherwise the defensive blocks the contract surfaces would
    # swallow it silently.
    assert isinstance(exc_info.value, BaseException)
    assert not isinstance(exc_info.value, Exception)
    # The original CancelledError is chained as __cause__ so the
    # diagnostic preserves the trigger.
    assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)
    # The site's name appears in the violation message.
    assert "synthetic.bypass" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2 — strict mode + compliant site → no violation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compliant_site_under_strict_does_not_raise(strict_on: None) -> None:
    """Synthetic site that DOES call :func:`mark_stash_completed`.

    Mirrors the §1.2 ``except BaseException: stash; raise`` form. The
    cancel propagates normally; no violation is raised.
    """

    async def compliant_call() -> None:
        with cancellation_stash_audited("synthetic.compliant"):
            try:
                raise asyncio.CancelledError("inner-await-cancelled")
            except BaseException:  # noqa: BLE001
                # Compliance branch fires the marker BEFORE re-raising.
                mark_stash_completed()
                raise

    with pytest.raises(asyncio.CancelledError):
        await _run_with_boundary(compliant_call)
    # No violation raised; the original CancelledError surfaced
    # cleanly (pytest.raises above caught CancelledError, not the
    # tripwire).


# ---------------------------------------------------------------------------
# Test 3 — strict mode + try/finally form → no violation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compliant_finally_form_under_strict_does_not_raise(
    strict_on: None,
) -> None:
    """The §1.1 ``try / finally`` shape also satisfies the tripwire.

    The marker fires from inside the ``finally`` block, regardless of
    whether the await returned normally or raised ``CancelledError``.
    """

    async def compliant_call() -> None:
        with cancellation_stash_audited("synthetic.try_finally"):
            try:
                raise asyncio.CancelledError("inner-await-cancelled")
            finally:
                # Compliance branch: stash + mark.
                mark_stash_completed()

    with pytest.raises(asyncio.CancelledError):
        await _run_with_boundary(compliant_call)


# ---------------------------------------------------------------------------
# Test 4 — strict mode OFF: bypass site is silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bypass_site_silent_under_strict_off(strict_off: None) -> None:
    """With ``GOLDFIVE_STRICT_STATE_OWNERSHIP=0`` the tripwire never fires.

    Mirrors the production default contract: deploys with the env var
    unset (or set to 0/false/off) pay only one ContextVar read per
    audited site enter/exit and never raise from
    :func:`assert_stash_invariant`. Even an obviously bypassing site
    (no compliance branch at all) must surface only the original
    ``CancelledError`` — never the violation.
    """
    assert not _state_audit.is_enabled()

    async def bypass_call() -> None:
        with cancellation_stash_audited("synthetic.bypass.strict_off"):
            raise asyncio.CancelledError("inner-await-cancelled")

    with pytest.raises(asyncio.CancelledError):
        await _run_with_boundary(bypass_call)


# ---------------------------------------------------------------------------
# Test 5 — strict mode OFF: compliant site is also silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compliant_site_silent_under_strict_off(strict_off: None) -> None:
    """Strict-off + compliant site → original CancelledError only.

    Defensive symmetry test: the strict-off contract holds whether
    the site is compliant or not.
    """
    assert not _state_audit.is_enabled()

    async def compliant_call() -> None:
        with cancellation_stash_audited("synthetic.compliant.strict_off"):
            try:
                raise asyncio.CancelledError("inner-await-cancelled")
            except BaseException:
                mark_stash_completed()
                raise

    with pytest.raises(asyncio.CancelledError):
        await _run_with_boundary(compliant_call)


# ---------------------------------------------------------------------------
# Test 6 — nested audited sites: only the bypassing one is reported
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_sites_report_only_bypass(strict_on: None) -> None:
    """Two audited sites stacked; outer is compliant, inner bypasses.

    The violation message names the bypassing site only — the
    compliant site already cleared its flag before the cancel
    reached it. Verifies the marker stack tracks per-site flags
    independently.
    """

    async def nested_call() -> None:
        with cancellation_stash_audited("outer.compliant"):
            try:
                with cancellation_stash_audited("inner.bypass"):
                    # Inner CancelledError — neither branch
                    # marks the inner stash, so it bypasses.
                    raise asyncio.CancelledError("inner")
            except BaseException:
                # Outer compliance fires before re-raise.
                mark_stash_completed()
                raise

    with pytest.raises(CancellationStashViolation) as exc_info:
        await _run_with_boundary(nested_call)
    msg = str(exc_info.value)
    assert "inner.bypass" in msg
    # Outer was compliant — must NOT appear in the bypass list.
    # (It can appear elsewhere in the message — the catch-site
    # frame chain — but not in the "- name" bullet list.)
    assert "- outer.compliant" not in msg


# ---------------------------------------------------------------------------
# Test 7 — non-cancel exceptions still propagate normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_exception_does_not_trigger_tripwire(strict_on: None) -> None:
    """Non-``CancelledError`` exceptions never reach the boundary's check.

    The boundary catches only ``asyncio.CancelledError``; a plain
    ``RuntimeError`` raised from inside an audited site flows through
    its caller's normal exception handling without any tripwire
    machinery firing.
    """

    async def raising_call() -> None:
        with cancellation_stash_audited("synthetic.raises_runtime_error"):
            raise RuntimeError("boom")

    # The boundary helper only catches CancelledError, so the
    # RuntimeError propagates directly — no tripwire involvement.
    with pytest.raises(RuntimeError, match="boom"):
        await _run_with_boundary(raising_call)
