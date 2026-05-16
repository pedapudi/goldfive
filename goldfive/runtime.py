"""Process-wide determinism handles for goldfive.

Optimization / evaluation harnesses driving goldfive need to compare
runs byte-for-byte across iterations. Production goldfive uses
``uuid.uuid4()`` for event ids, drift ids, plan ids, and several
internal correlation tokens — those are random by design. This module
exposes a single :func:`set_seed` hook that installs a seeded
:class:`random.Random` instance as the canonical source of UUIDs and
random ints; goldfive's own code consults it via :func:`seeded_uuid4`
and :func:`seeded_random`.

Default behaviour is unchanged: when :func:`set_seed` has not been
called, :func:`seeded_uuid4` falls through to ``uuid.uuid4()`` (the
production path) and :func:`seeded_random` returns the module-level
:mod:`random` singleton. Tests / harnesses install a seed at the top of
the run; goldfive's emit paths consult the helpers to produce
deterministic output without code changes elsewhere.

Concurrency: the seeded state is process-global. Two harnesses sharing
one process cannot install different seeds without stepping on each
other; the contract is "one run per process" for byte-identical
reproducibility. The module is goroutine-safe to read; concurrent
:func:`set_seed` calls produce a last-writer-wins state.
"""

from __future__ import annotations

import random
import threading
import uuid
from typing import Final

__all__ = [
    "clear_seed",
    "is_seeded",
    "seeded_random",
    "seeded_uuid4",
    "set_seed",
]


_LOCK: Final[threading.RLock] = threading.RLock()
_SEEDED: random.Random | None = None


def set_seed(seed: int) -> None:
    """Install a process-global seed for UUID generation + internal randomness.

    Subsequent :func:`seeded_uuid4` calls return deterministic UUID4s
    drawn from the seeded :class:`random.Random` instance; subsequent
    :func:`seeded_random` calls return the same instance.

    Two runs of the same goldfive program with the same input + the
    same ``seed`` produce identical event id streams, identical drift
    ids, and identical plan ids. The contract is best-effort
    end-to-end: code paths that bypass these helpers (third-party
    libraries, sinks, downstream consumers) remain non-deterministic.
    """
    global _SEEDED
    with _LOCK:
        _SEEDED = random.Random(seed)


def clear_seed() -> None:
    """Drop the installed seed; subsequent randoms revert to system entropy.

    Inverse of :func:`set_seed`. Test fixtures that scope determinism
    to a single test call this in their teardown so a later test that
    does not call :func:`set_seed` sees the unmodified production
    behaviour.
    """
    global _SEEDED
    with _LOCK:
        _SEEDED = None


def is_seeded() -> bool:
    """Return ``True`` iff a seed is currently installed."""
    with _LOCK:
        return _SEEDED is not None


def seeded_random() -> random.Random:
    """Return the seeded :class:`random.Random`, or the module singleton.

    When :func:`set_seed` has not been called, falls back to
    :mod:`random`'s default instance so callers see standard
    non-deterministic behaviour. When a seed is installed, returns the
    seeded instance — callers should use this rather than the bare
    :mod:`random` module to inherit determinism.
    """
    with _LOCK:
        return _SEEDED if _SEEDED is not None else random._inst  # type: ignore[attr-defined]


def seeded_uuid4() -> uuid.UUID:
    """Return a UUID4 — deterministic when seeded, ``uuid.uuid4()`` otherwise.

    Replacement for ``uuid.uuid4()`` at every goldfive call site that
    feeds determinism-sensitive output (event ids, drift ids, plan
    ids, attempt ids). The fallback path matches ``uuid.uuid4()`` byte
    for byte so unseeded callers see no behaviour change.

    The seeded path produces a v4-shaped UUID drawn from the seeded
    :class:`random.Random` (16 random bytes with the v4 / RFC4122
    variant bits set), matching the construction in Python's standard
    library ``uuid.uuid4``.
    """
    with _LOCK:
        seeded = _SEEDED
    if seeded is None:
        return uuid.uuid4()
    raw = seeded.getrandbits(128).to_bytes(16, "big")
    # Apply the version-4 + RFC-4122 variant bits, matching
    # ``uuid.uuid4`` byte for byte.
    raw_bytes = bytearray(raw)
    raw_bytes[6] = (raw_bytes[6] & 0x0F) | 0x40  # version 4
    raw_bytes[8] = (raw_bytes[8] & 0x3F) | 0x80  # variant RFC 4122
    return uuid.UUID(bytes=bytes(raw_bytes))
