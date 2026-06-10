"""Observe-only signal ledger (AGENCY-PRESERVATION.md PR 5 — "Telemetry first").

The :class:`SignalLedger` is goldfive's bookkeeping for *corrective signals*:
which ``(drift_kind, task_id)`` keys had a steering signal delivered (or, under
``observation_only``, would-have-been delivered), how often each key re-fired
after a signal, and what eventually became of the key. It is the durable
substrate the §5.4 shadow/differential validation diffs over real traffic and
the grace-window bookkeeping PR 8 will *gate* on.

**This module gates NOTHING.** It records deliveries, drift re-fires, and
resolution observations; it never suppresses a dispatch, never delays an
escalation, never changes a single control-flow decision. PR 5 is observe-only
by construction (AGENCY-PRESERVATION.md §5.1): every behavior change lands in a
later PR behind a flag. The grace-window fields recorded here (turn-stamped
deliveries on the goldfive#441 ``Session._reasoning_turn`` clock, re-fire
counts) are written so PR 8 can read them — PR 8 owns the pacing logic.

Key discipline (binding — AGENCY-PRESERVATION.md §5.6, the project's
stable-keys scar tissue)
------------------------------------------------------------------------
Entries are keyed ``(drift_kind, task_id)`` — exactly the tuple
``DefaultSteerer._record_refine_outcome`` already keys ``refine_outcomes`` on.
``task_id`` is a goldfive-minted stable task id (``Task.id``), NEVER an
LLM-minted identifier. A churning key (e.g. an id the model re-mints every
turn) would open a fresh ledger entry per observation and the lifecycle gates
PR 8 builds on top would never engage. The drift kind is the ``DriftKind``
*value* string (e.g. ``"looping_tool_call"``), so a ledger round-tripped to
JSON by a sink stays human-readable.

Storage shape
-------------
State lives on ``Session.state`` under :data:`KEY_SIGNAL_LEDGER` (a
``goldfive.``-prefixed key so :func:`goldfive.state_store.write` accepts it),
following the StateStore accessor convention (cf. ``active_steer`` /
``processed_steer_ids`` / ``active_drifts``). The value is a JSON-serialisable
``dict[composed_key, entry_dict]``; every entry carries its own
``drift_kind`` / ``task_id`` so the map is reconstructible without parsing the
composed key (the "ledger state always parseable" interleaving invariant).

Concurrency
-----------
goldfive's race history (#405 dedup registry, the per-session plan lock, growth
dedup linearisability — §5.5) says concurrency is where its bugs live. The
ledger is therefore written to be idempotent and order-tolerant:

* a ``drift_id`` is counted as a fire at most once and produces at most one
  delivery per channel (no double-count per ``drift_id``);
* turn stamps are folded with min/max so out-of-order concurrent records keep
  ``first_fire_turn <= last_fire_turn`` and a non-negative
  ``turns_to_resolution``;
* a key reaches **exactly one** terminal outcome — the first resolution wins,
  every later one is a no-op.

The ledger itself holds no lock; callers run under goldfive's existing
single-session serialisation (the drift dispatch path and the task-transition
emit path are already serialised per session). The idempotent folds mean even
a missed-serialisation interleaving cannot corrupt the recorded counts.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from goldfive import state_store as _ostate
from goldfive.events import (
    SIGNAL_OUTCOME_ESCALATED,
    SIGNAL_OUTCOME_INVOCATION_ENDED,
    SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL,
    SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED,
    SIGNAL_OUTCOME_USER_INTERVENED,
)

if TYPE_CHECKING:  # pragma: no cover - type-check only
    from goldfive.types import Session

#: ``Session.state`` key for the signal ledger. ``goldfive.``-prefixed so the
#: state_store write-guard accepts it; sibling of ``goldfive.active_drifts``.
KEY_SIGNAL_LEDGER = "goldfive.signal_ledger"

#: Separator for the composed ``(drift_kind, task_id)`` storage key. ASCII unit
#: separator — never appears in a ``DriftKind`` value or a goldfive task id, so
#: the composed key is unambiguous. The entry also stores both components
#: explicitly, so nothing downstream depends on splitting this back apart.
_KEY_SEP = "\x1f"

#: Cap on the per-entry retained fire-id list (the dedup set + audit trail).
#: ``fire_count`` is the authoritative monotone counter; the id list is bounded
#: so a pathologically long-lived key cannot balloon ``Session.state``. A key
#: that fires more than this many distinct times in one run is already a hard
#: guardrail case (the loop detectors would have tripped long before).
_FIRE_IDS_CAP = 512


def compose_key(drift_kind: str, task_id: str) -> str:
    """Return the storage key for a ``(drift_kind, task_id)`` pair."""
    return f"{str(drift_kind or '')}{_KEY_SEP}{str(task_id or '')}"


@dataclasses.dataclass
class DeliveryRecord:
    """One signal delivery (or dry-run would-be delivery) on a ledger key.

    ``dry_run`` mirrors the ``SignalDelivered`` event's flag: ``True`` when the
    steering injection gate was shut (``observation_only``), i.e. the signal
    carried no production authority. ``ladder_level`` / ``severity`` /
    ``note_text`` snapshot what the dispatch path computed so the record is
    self-describing for the §5.4 divergence report.
    """

    drift_id: str
    channel: str
    turn: int
    dry_run: bool
    severity: str = ""
    ladder_level: str = ""
    note_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_id": str(self.drift_id or ""),
            "channel": str(self.channel or ""),
            "turn": int(self.turn),
            "dry_run": bool(self.dry_run),
            "severity": str(self.severity or ""),
            "ladder_level": str(self.ladder_level or ""),
            "note_text": str(self.note_text or ""),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeliveryRecord:
        return cls(
            drift_id=str(data.get("drift_id", "") or ""),
            channel=str(data.get("channel", "") or ""),
            turn=_coerce_int(data.get("turn", 0)),
            dry_run=bool(data.get("dry_run", False)),
            severity=str(data.get("severity", "") or ""),
            ladder_level=str(data.get("ladder_level", "") or ""),
            note_text=str(data.get("note_text", "") or ""),
        )


@dataclasses.dataclass
class LedgerEntry:
    """The recorded history of one ``(drift_kind, task_id)`` key.

    ``fire_count`` is the number of *distinct* drift ids observed for the key
    (``len(fired_drift_ids)`` modulo the cap). ``first_delivery_turn`` is
    ``-1`` until a signal is recorded; ``outcome`` is ``""`` until the key
    resolves and ``outcome_turn`` is ``-1`` until then.
    """

    drift_kind: str
    task_id: str
    fire_count: int = 0
    #: Distinct fires observed strictly after the first delivery was recorded.
    #: The grace-window quantity PR 8 reads: how often a key kept drifting once
    #: a signal had been delivered. Persisted (not derived) so it survives the
    #: read-modify-write round-trip every mutator performs.
    refire_count: int = 0
    fired_drift_ids: list[str] = dataclasses.field(default_factory=list)
    first_fire_turn: int = -1
    last_fire_turn: int = -1
    deliveries: list[DeliveryRecord] = dataclasses.field(default_factory=list)
    first_delivery_turn: int = -1
    outcome: str = ""
    outcome_turn: int = -1

    # -- derived views ---------------------------------------------------

    @property
    def has_delivery(self) -> bool:
        return bool(self.deliveries)

    @property
    def has_real_delivery(self) -> bool:
        """True when at least one non-dry-run signal was delivered."""
        return any(not d.dry_run for d in self.deliveries)

    @property
    def is_open(self) -> bool:
        """A key is open until it reaches a terminal outcome."""
        return not self.outcome

    def turns_to_resolution(self) -> int:
        """Logical turns from first observation to resolution (clamped >= 0)."""
        if self.outcome_turn < 0:
            return 0
        anchor = self.first_fire_turn if self.first_fire_turn >= 0 else self.first_delivery_turn
        if anchor < 0:
            return 0
        return max(0, self.outcome_turn - anchor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_kind": str(self.drift_kind or ""),
            "task_id": str(self.task_id or ""),
            "fire_count": int(self.fire_count),
            "refire_count": int(self.refire_count),
            "fired_drift_ids": [str(x) for x in self.fired_drift_ids],
            "first_fire_turn": int(self.first_fire_turn),
            "last_fire_turn": int(self.last_fire_turn),
            "deliveries": [d.to_dict() for d in self.deliveries],
            "first_delivery_turn": int(self.first_delivery_turn),
            "outcome": str(self.outcome or ""),
            "outcome_turn": int(self.outcome_turn),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerEntry:
        raw_deliveries = data.get("deliveries", [])
        deliveries = [
            DeliveryRecord.from_dict(d)
            for d in (raw_deliveries if isinstance(raw_deliveries, list) else [])
            if isinstance(d, Mapping)
        ]
        raw_fire_ids = data.get("fired_drift_ids", [])
        fired_drift_ids = [
            str(x) for x in (raw_fire_ids if isinstance(raw_fire_ids, list) else []) if x
        ]
        return cls(
            drift_kind=str(data.get("drift_kind", "") or ""),
            task_id=str(data.get("task_id", "") or ""),
            fire_count=_coerce_int(data.get("fire_count", 0)),
            refire_count=_coerce_int(data.get("refire_count", 0)),
            fired_drift_ids=fired_drift_ids,
            first_fire_turn=_coerce_int(data.get("first_fire_turn", -1), default=-1),
            last_fire_turn=_coerce_int(data.get("last_fire_turn", -1), default=-1),
            deliveries=deliveries,
            first_delivery_turn=_coerce_int(data.get("first_delivery_turn", -1), default=-1),
            outcome=str(data.get("outcome", "") or ""),
            outcome_turn=_coerce_int(data.get("outcome_turn", -1), default=-1),
        )


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SignalLedger:
    """StateStore-backed, observe-only ledger of corrective signals.

    Construct via :meth:`for_session` (or directly on a ``state`` mapping for
    tests). Every mutator is a pure, synchronous, idempotent fold over the
    serialised dict on ``Session.state`` — no sinks, no event loop. Callers
    that want the matching ``SignalDelivered`` / ``SignalOutcome`` wire events
    pair a ledger call with an emit (see ``DefaultSteerer._emit_signal_*`` /
    ``TaskStateMachine``); keeping emission out of the ledger lets the
    hypothesis interleaving tests hammer the bookkeeping directly.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state

    @classmethod
    def for_session(cls, session: Session | None) -> SignalLedger:
        """Build a ledger over ``session.state`` (or a throwaway dict)."""
        state = getattr(session, "state", None)
        if not isinstance(state, MutableMapping):
            state = {}
        return cls(state)

    # -- raw map access --------------------------------------------------

    def _read_all(self) -> dict[str, dict[str, Any]]:
        raw = _ostate.read(self._state, KEY_SIGNAL_LEDGER, {})
        if not isinstance(raw, dict):
            return {}
        # Shallow copy so a caller mutating the returned dict cannot corrupt
        # the live state out from under a concurrent reader before write-back.
        return {str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)}

    def _write_all(self, data: Mapping[str, dict[str, Any]]) -> None:
        _ostate.write(self._state, KEY_SIGNAL_LEDGER, dict(data))

    def _get_entry(self, store: dict[str, dict[str, Any]], key: str) -> LedgerEntry | None:
        raw = store.get(key)
        if not isinstance(raw, Mapping):
            return None
        return LedgerEntry.from_dict(raw)

    def _put_entry(self, store: dict[str, dict[str, Any]], entry: LedgerEntry) -> None:
        store[compose_key(entry.drift_kind, entry.task_id)] = entry.to_dict()

    # -- reads -----------------------------------------------------------

    def entry(self, drift_kind: str, task_id: str) -> LedgerEntry | None:
        return self._get_entry(self._read_all(), compose_key(drift_kind, task_id))

    def entries(self) -> list[LedgerEntry]:
        return [LedgerEntry.from_dict(v) for v in self._read_all().values()]

    def open_entries(self) -> list[LedgerEntry]:
        return [e for e in self.entries() if e.is_open]

    # -- mutators --------------------------------------------------------

    def record_fire(
        self,
        *,
        drift_kind: str,
        task_id: str,
        turn: int,
        drift_id: str,
    ) -> LedgerEntry:
        """Record that a drift fired for ``(drift_kind, task_id)``.

        Idempotent per ``drift_id``: a given drift increments ``fire_count``
        at most once (re-emits of the same drift condition fold away). Creates
        the entry on first sight. Turn bounds are folded with min/max so
        out-of-order concurrent records stay sane. Never resolves anything.
        """
        store = self._read_all()
        key = compose_key(drift_kind, task_id)
        entry = self._get_entry(store, key) or LedgerEntry(
            drift_kind=str(drift_kind or ""), task_id=str(task_id or "")
        )
        self._fold_fire(entry, drift_id=drift_id, turn=turn)
        self._put_entry(store, entry)
        self._write_all(store)
        return entry

    @staticmethod
    def _fold_fire(entry: LedgerEntry, *, drift_id: str, turn: int) -> None:
        did = str(drift_id or "")
        t = _coerce_int(turn)
        # Count + append only for a genuinely-new, non-empty drift id (dedup —
        # no double-count per drift_id). A distinct fire recorded once a signal
        # has already been delivered is a re-fire (the key kept drifting after
        # goldfive spoke). Within one ``handle_drift`` the fire is folded
        # before the delivery, so the first drift never counts as a re-fire of
        # its own delivery.
        if did and did not in entry.fired_drift_ids:
            entry.fired_drift_ids.append(did)
            if len(entry.fired_drift_ids) > _FIRE_IDS_CAP:
                entry.fired_drift_ids = entry.fired_drift_ids[-_FIRE_IDS_CAP:]
            entry.fire_count += 1
            if entry.first_delivery_turn >= 0:
                entry.refire_count += 1
        # ALWAYS fold the turn bounds — even a duplicate / out-of-order re-emit
        # legitimately widens [first_fire_turn, last_fire_turn] so the bounds
        # stay monotone under any interleaving.
        entry.first_fire_turn = t if entry.first_fire_turn < 0 else min(entry.first_fire_turn, t)
        entry.last_fire_turn = max(entry.last_fire_turn, t)

    def record_delivery(
        self,
        *,
        drift_kind: str,
        task_id: str,
        drift_id: str,
        channel: str,
        turn: int,
        dry_run: bool,
        severity: str = "",
        ladder_level: str = "",
        note_text: str = "",
    ) -> tuple[LedgerEntry, bool]:
        """Record a delivered (or would-be, ``dry_run``) signal.

        Dedups by ``(drift_id, channel)`` — a redelivery of the same drift on
        the same channel folds away (no double-count per ``drift_id``). Also
        folds a fire for ``drift_id`` so a delivery without a prior
        :meth:`record_fire` still anchors the key's turn bounds. Returns
        ``(entry, recorded)`` where ``recorded`` is ``False`` for a deduped
        redelivery.
        """
        store = self._read_all()
        key = compose_key(drift_kind, task_id)
        entry = self._get_entry(store, key) or LedgerEntry(
            drift_kind=str(drift_kind or ""), task_id=str(task_id or "")
        )
        # Anchor turn bounds via the fire fold (idempotent on drift_id).
        self._fold_fire(entry, drift_id=drift_id, turn=turn)
        did = str(drift_id or "")
        chan = str(channel or "")
        already = any(d.drift_id == did and d.channel == chan for d in entry.deliveries)
        recorded = False
        if not already:
            t = _coerce_int(turn)
            entry.deliveries.append(
                DeliveryRecord(
                    drift_id=did,
                    channel=chan,
                    turn=t,
                    dry_run=bool(dry_run),
                    severity=str(severity or ""),
                    ladder_level=str(ladder_level or ""),
                    note_text=str(note_text or ""),
                )
            )
            entry.first_delivery_turn = (
                t if entry.first_delivery_turn < 0 else min(entry.first_delivery_turn, t)
            )
            recorded = True
        self._put_entry(store, entry)
        self._write_all(store)
        return entry, recorded

    def _resolve_in_store(
        self,
        store: dict[str, dict[str, Any]],
        entry: LedgerEntry,
        *,
        outcome: str,
        turn: int,
    ) -> LedgerEntry | None:
        """Set the terminal outcome on ``entry`` iff it is open + delivered.

        Returns the resolved entry on a NEW resolution, ``None`` otherwise
        (already resolved, or never carried a delivery — outcomes pair 1:1
        with ``SignalDelivered`` so an undelivered key emits no outcome).
        """
        if not entry.is_open or not entry.has_delivery:
            return None
        entry.outcome = str(outcome or "")
        entry.outcome_turn = _coerce_int(turn)
        self._put_entry(store, entry)
        return entry

    def resolve_task(
        self,
        *,
        task_id: str,
        turn: int,
    ) -> list[LedgerEntry]:
        """Resolve every open, delivered key bound to ``task_id``.

        Called when the task reaches a terminal state (the conservative
        "resolved" signal — we do not attempt to detect mid-task "progressing"
        here; that never over-claims self-correction). A key with at least one
        real (non-dry-run) delivery resolves ``self_corrected_after_signal``;
        a key with only dry-run deliveries resolves ``self_corrected_unaided``
        (the ``observation_only`` base-rate case).
        """
        store = self._read_all()
        resolved: list[LedgerEntry] = []
        for raw in list(store.values()):
            entry = LedgerEntry.from_dict(raw)
            if entry.task_id != str(task_id or ""):
                continue
            if not entry.is_open or not entry.has_delivery:
                continue
            outcome = (
                SIGNAL_OUTCOME_SELF_CORRECTED_AFTER_SIGNAL
                if entry.has_real_delivery
                else SIGNAL_OUTCOME_SELF_CORRECTED_UNAIDED
            )
            done = self._resolve_in_store(store, entry, outcome=outcome, turn=turn)
            if done is not None:
                resolved.append(done)
        if resolved:
            self._write_all(store)
        return resolved

    def resolve_escalated(
        self,
        *,
        drift_kind: str,
        task_id: str,
        turn: int,
    ) -> LedgerEntry | None:
        """Resolve the ``(drift_kind, task_id)`` key as ``escalated``.

        Called from the pause-control dispatch — a key that escalated to a
        pause is terminal regardless of ``observation_only`` (the escalation
        *decision* happened even when the pause itself was suppressed).
        """
        store = self._read_all()
        entry = self._get_entry(store, compose_key(drift_kind, task_id))
        if entry is None:
            return None
        done = self._resolve_in_store(
            store, entry, outcome=SIGNAL_OUTCOME_ESCALATED, turn=turn
        )
        if done is not None:
            self._write_all(store)
        return done

    def resolve_user_intervened(self, *, turn: int) -> list[LedgerEntry]:
        """Resolve every open, delivered key as ``user_intervened``.

        Called when a USER_STEER / USER_CANCEL arrives. Conservative scope: we
        resolve only keys that actually carried a goldfive signal (so we can
        honestly say "goldfive signaled, then the user took over"); keys that
        never received a signal are left for the task-terminal or run-end
        sweep rather than over-attributed to the user.
        """
        store = self._read_all()
        resolved: list[LedgerEntry] = []
        for raw in list(store.values()):
            entry = LedgerEntry.from_dict(raw)
            done = self._resolve_in_store(
                store, entry, outcome=SIGNAL_OUTCOME_USER_INTERVENED, turn=turn
            )
            if done is not None:
                resolved.append(done)
        if resolved:
            self._write_all(store)
        return resolved

    def finalize_open(self, *, turn: int) -> list[LedgerEntry]:
        """Resolve every still-open, delivered key as ``invocation_ended``.

        Called once at run end (the conservative catch-all: the invocation
        ended before the key resolved, and we will not guess whether the agent
        self-corrected). Idempotent — a second call finds nothing open.
        """
        store = self._read_all()
        resolved: list[LedgerEntry] = []
        for raw in list(store.values()):
            entry = LedgerEntry.from_dict(raw)
            done = self._resolve_in_store(
                store, entry, outcome=SIGNAL_OUTCOME_INVOCATION_ENDED, turn=turn
            )
            if done is not None:
                resolved.append(done)
        if resolved:
            self._write_all(store)
        return resolved


__all__ = [
    "KEY_SIGNAL_LEDGER",
    "DeliveryRecord",
    "LedgerEntry",
    "SignalLedger",
    "compose_key",
]
