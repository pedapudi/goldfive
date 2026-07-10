"""§5.4 shadow / differential-validation tool (AGENCY-PRESERVATION.md PR 13a).

The roadmap makes a *reviewed divergence report over real traffic* the
**exit criterion** for enabling any behavior PR (§5.4, §5.8): before a
default flip, the new steering decision logic must run dry
(``observation_only`` → ``SignalDelivered(dry_run=true)``) and its
would-be decisions must be diffed against the legacy regime's would-be
decisions *on the same runs*, with the divergences reviewed.

This tool consumes the ``SignalDelivered`` telemetry (PR 5) out of one or
two JSONL sink logs and renders that divergence report. It reads events
through goldfive's own :func:`replay_from_jsonl` (proto-canonical, so it
is immune to the camelCase JSON field naming), never by string-matching
the log.

Two modes
---------
* **two-log (primary)** — ``--legacy LEG.jsonl --new NEW.jsonl``: the same
  workload run once under the legacy regime and once under the new regime
  (both ``observation_only`` so signals are dry-run). The tool aligns
  deliveries across the logs by a stable cross-run key
  ``(kind, task_id, occurrence#)`` — drift ids are per-run minted, so they
  are NOT used as the join key (the project's stable-keys discipline) —
  and reports, per drift, exactly how the two regimes' *decisions* differ,
  plus drifts that fired in only one regime. The delivery ``channel`` /
  ``channel_action`` are per-regime *transport identity* (since PR 6 the new
  regime always rides ``request_context`` and the legacy regime
  ``nudge_replay``), so they are reported informationally but excluded from
  the divergence-driving comparison — otherwise every aligned key would
  trivially "diverge" on transport and drown the real decision divergences.
* **single-log (census)** — one positional ``LOG.jsonl``: a per-event
  legacy-would-do vs. new-would-do derivation from each delivery's own
  ``decision`` payload, plus a census of channels / ladder levels /
  dry-run. Useful for a quick read of one shadow run.

Loud on zero (§5.6 integration-not-unit)
----------------------------------------
``signal_telemetry`` is DEFAULT OFF (goldfive#456). A log produced with it
off contains **no** ``SignalDelivered`` events — and a tool that silently
renders an empty report from such a log is the exact "dead middleware /
flag was off" trap the roadmap calls out. So :func:`load_signals` raises
:class:`ShadowDiffError` when a log carries zero deliveries, unless
``allow_empty=True`` (``--allow-empty``) is passed for a run known to be
genuinely drift-free.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = [
    "ShadowDiffError",
    "SignalRecord",
    "KeyDivergence",
    "DivergenceReport",
    "load_signals",
    "diff_two_logs",
    "single_log_report",
    "render_two_log_text",
    "render_single_log_text",
    "main",
]


class ShadowDiffError(RuntimeError):
    """Raised on an unusable input — most importantly, zero deliveries.

    A zero-delivery log almost always means ``signal_telemetry`` was off
    (its default). Surfacing that as a hard error rather than an empty
    report is the point (§5.6).
    """


# --- comparable decision fields -------------------------------------------
# The subset of the ``SignalDelivered.decision`` payload (+ envelope) that
# defines *what a regime would do* to the agent. Prose (``note_text``) is
# deliberately excluded from the divergence set — §5.4 diffs decisions, not
# wording — but is retained on the record for the human-readable detail.
#
# ``channel`` / ``channel_action`` are deliberately NOT here: they are
# per-regime *transport identity*, not a steering decision. Since PR 6 the new
# regime always rides ``request_context`` and the legacy regime always rides
# ``nudge_replay`` (``channel_action`` enqueued vs. queued), so EVERY aligned
# key would trivially differ on them — that noise would swamp the real decision
# divergences (``would_cancel_inflight``, ``ladder_level``, plan-swap targets)
# the §5.4 review exists to surface. They live in :data:`_TRANSPORT_FIELDS`
# instead, shown informationally per key but never counted as a divergence.
_DECISION_FIELDS: tuple[str, ...] = (
    "ladder_level",
    "would_cancel_inflight",
    "promotion",
    "superseded_task_ids",
    "replacement_task_ids",
    "dry_run",
)

#: Per-regime transport identity — which delivery channel carried the signal.
#: Reported informationally (a transport line on each key) but excluded from
#: the divergence-driving comparison; a key that differs ONLY here is NOT a
#: decision divergence (AGENCY-PRESERVATION.md §5.4 diffs decisions).
_TRANSPORT_FIELDS: tuple[str, ...] = ("channel", "channel_action")

#: All fields surfaced in a record's comparable projection (decision +
#: transport), so the rendered report can still show channel/channel_action.
_VIEW_FIELDS: tuple[str, ...] = _DECISION_FIELDS + _TRANSPORT_FIELDS


@dataclasses.dataclass
class SignalRecord:
    """One parsed ``SignalDelivered`` event."""

    sequence: int
    drift_id: str
    kind: str
    task_id: str
    channel: str
    severity: str
    turn: int
    dry_run: bool
    ladder_level: str
    note_text: str
    decision: dict[str, Any]

    def decision_view(self) -> dict[str, Any]:
        """The comparable projection (decision + transport fields) for diffing."""
        view: dict[str, Any] = {}
        for field in _VIEW_FIELDS:
            if field == "channel":
                view[field] = self.channel
            elif field == "dry_run":
                view[field] = self.dry_run
            elif field == "ladder_level":
                view[field] = self.ladder_level or self.decision.get("ladder_level", "")
            else:
                view[field] = self.decision.get(field)
        return view


@dataclasses.dataclass
class KeyDivergence:
    """The legacy-vs-new comparison for one aligned drift key."""

    kind: str
    task_id: str
    occurrence: int
    present_in: str  # "both" | "legacy_only" | "new_only"
    legacy: dict[str, Any] | None
    new: dict[str, Any] | None
    diverged_fields: list[str]
    #: Per-regime transport fields (channel / channel_action) that differ.
    #: Informational only — a key that differs ONLY here is NOT ``diverged``.
    transport_fields: list[str] = dataclasses.field(default_factory=list)
    #: ``True`` when a one-sided key is a **join artifact**, not a real
    #: silence: the other regime DID fire this drift kind, but on a disjoint
    #: ``task_id`` namespace, so ``(kind, task_id, occurrence)`` structurally
    #: cannot match. The flip-target comparison (ledger OUTCOME task ids vs.
    #: forecast task ids) is exactly this case — the ids differ by
    #: construction. Un-joinable keys are EXCLUDED from the divergence verdict
    #: and logged separately, so a 13b operator is not misled by a silently
    #: partial diff (a run-vs-run join that never aligned).
    unjoinable: bool = False

    @property
    def diverged(self) -> bool:
        if self.unjoinable:
            # A join artifact, not a decision divergence: the other regime
            # fired this kind on a different task-id namespace.
            return False
        return self.present_in != "both" or bool(self.diverged_fields)

    @property
    def transport_only(self) -> bool:
        """True iff the key differs ONLY in transport (not a decision divergence)."""
        return self.present_in == "both" and not self.diverged_fields and bool(
            self.transport_fields
        )


@dataclasses.dataclass
class DivergenceReport:
    """Aligned two-log divergence report (the §5.4 exit-criterion artifact)."""

    legacy_path: str
    new_path: str
    legacy_count: int
    new_count: int
    keys: list[KeyDivergence]

    @property
    def diverged_keys(self) -> list[KeyDivergence]:
        return [k for k in self.keys if k.diverged]

    @property
    def legacy_only(self) -> list[KeyDivergence]:
        """Drifts the LEGACY regime fired and the new regime genuinely did not.

        Excludes un-joinable keys (a join artifact — the new regime fired the
        same kind on a different task-id namespace, so it is not a silence).
        """
        return [
            k for k in self.keys if k.present_in == "legacy_only" and not k.unjoinable
        ]

    @property
    def new_only(self) -> list[KeyDivergence]:
        """Drifts the NEW regime fired and the legacy regime genuinely did not."""
        return [
            k for k in self.keys if k.present_in == "new_only" and not k.unjoinable
        ]

    @property
    def field_divergences(self) -> list[KeyDivergence]:
        return [k for k in self.keys if k.present_in == "both" and k.diverged_fields]

    @property
    def transport_only_keys(self) -> list[KeyDivergence]:
        """Keys that differ ONLY in transport (channel) — informational, not divergences."""
        return [k for k in self.keys if k.transport_only]

    @property
    def unjoinable_keys(self) -> list[KeyDivergence]:
        """One-sided keys that are join artifacts, not decision divergences.

        The other regime fired the same drift kind on a disjoint ``task_id``
        namespace (e.g. ledger OUTCOME ids vs. forecast ids), so the
        ``(kind, task_id, occurrence)`` key structurally cannot align. Logged
        separately so a partial join is never silently read as divergence.
        """
        return [k for k in self.keys if k.unjoinable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_path": self.legacy_path,
            "new_path": self.new_path,
            "legacy_delivery_count": self.legacy_count,
            "new_delivery_count": self.new_count,
            "aligned_keys": len(self.keys),
            "diverged_keys": len(self.diverged_keys),
            "legacy_only": len(self.legacy_only),
            "new_only": len(self.new_only),
            "field_divergences": len(self.field_divergences),
            "transport_only_keys": len(self.transport_only_keys),
            "unjoinable_keys": len(self.unjoinable_keys),
            "keys": [dataclasses.asdict(k) for k in self.keys],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_signals(path: str | Path, *, allow_empty: bool = False) -> list[SignalRecord]:
    """Parse the ``SignalDelivered`` events out of one JSONL sink log.

    Reads via goldfive's proto-canonical :func:`replay_from_jsonl`. Raises
    :class:`ShadowDiffError` if the log has events but **no** deliveries
    (the ``signal_telemetry``-was-off trap) unless ``allow_empty``.
    """
    try:
        from goldfive.sinks import replay_from_jsonl
    except ImportError as exc:  # pragma: no cover - proto extra missing
        raise ShadowDiffError(
            "goldfive proto stubs unavailable; install the `proto`/`dev` extra "
            "to parse JSONL sink logs"
        ) from exc
    if replay_from_jsonl is None:  # pragma: no cover - proto extra missing
        raise ShadowDiffError(
            "goldfive proto stubs unavailable; install the `proto`/`dev` extra"
        )

    p = Path(path)
    if not p.exists():
        raise ShadowDiffError(f"log not found: {p}")

    events = replay_from_jsonl(p)
    records: list[SignalRecord] = []
    for evt in events:
        if not hasattr(evt, "WhichOneof") or evt.WhichOneof("payload") != "signal_delivered":
            continue
        sd = evt.signal_delivered
        try:
            decision = json.loads(sd.decision_json) if sd.decision_json else {}
        except (TypeError, ValueError):
            decision = {}
        if not isinstance(decision, dict):
            decision = {}
        records.append(
            SignalRecord(
                sequence=int(getattr(evt, "sequence", 0) or 0),
                drift_id=sd.drift_id,
                kind=sd.kind,
                task_id=sd.task_id,
                channel=sd.channel,
                severity=sd.severity,
                turn=int(sd.turn),
                dry_run=bool(sd.dry_run),
                ladder_level=str(decision.get("ladder_level", "") or ""),
                note_text=sd.note_text,
                decision=decision,
            )
        )

    if not records and not allow_empty:
        raise ShadowDiffError(
            f"{p}: parsed {len(events)} event(s) but 0 SignalDelivered — is "
            "signal_telemetry enabled? It is DEFAULT OFF (goldfive#456); a "
            "shadow run must set GOLDFIVE_STEER_SIGNAL_TELEMETRY=1. Pass "
            "allow_empty/--allow-empty only if this run was genuinely "
            "drift-free."
        )
    return records


def _assign_occurrences(
    records: list[SignalRecord],
) -> list[tuple[tuple[str, str, int], SignalRecord]]:
    """Key each record by ``(kind, task_id, occurrence#)`` in sequence order.

    drift ids are per-run minted and so cannot join across two logs; the
    stable cross-run key is the running occurrence index of the
    ``(kind, task_id)`` pair (deterministic workloads line up exactly;
    real-traffic mismatches surface as legacy_only / new_only).
    """
    counters: dict[tuple[str, str], int] = defaultdict(int)
    keyed: list[tuple[tuple[str, str, int], SignalRecord]] = []
    for rec in sorted(records, key=lambda r: r.sequence):
        base = (rec.kind, rec.task_id)
        occ = counters[base]
        counters[base] += 1
        keyed.append(((rec.kind, rec.task_id, occ), rec))
    return keyed


# ---------------------------------------------------------------------------
# Two-log divergence (primary)
# ---------------------------------------------------------------------------


def diff_two_logs(
    legacy: list[SignalRecord],
    new: list[SignalRecord],
    *,
    legacy_path: str = "legacy",
    new_path: str = "new",
) -> DivergenceReport:
    """Align two logs per drift key and report decision divergences.

    The cross-run join key is ``(kind, task_id, occurrence#)`` — drift ids are
    per-run minted so cannot join across logs. ``task_id`` is stable within a
    regime but NOT across regimes when the plan taxonomy differs (the
    flip-target case: ledger OUTCOME task ids vs. forecast task ids). A
    one-sided key whose kind DID fire in the other regime but on a disjoint
    ``task_id`` namespace is therefore a **join artifact**, not a real
    silence; it is flagged ``unjoinable`` and excluded from the divergence
    verdict (see :attr:`KeyDivergence.unjoinable`) so a partial join is never
    misread as divergence. A true cross-regime deliverable-identity join would
    need a goal-anchored id on the wire (``SignalDelivered`` has none today —
    a dependency for a sibling runtime PR); until then the un-joinable set is
    surfaced explicitly rather than silently mis-bucketed.
    """
    legacy_keyed = dict(_assign_occurrences(legacy))
    new_keyed = dict(_assign_occurrences(new))
    all_keys = sorted(set(legacy_keyed) | set(new_keyed))

    # Per-kind task-id namespaces, to tell a genuine one-sided silence (the
    # other regime never fired this kind at all) from a join artifact (it
    # fired the kind, but on task ids that cannot align across regimes).
    legacy_tids: dict[str, set[str]] = defaultdict(set)
    new_tids: dict[str, set[str]] = defaultdict(set)
    for rec in legacy:
        legacy_tids[rec.kind].add(rec.task_id)
    for rec in new:
        new_tids[rec.kind].add(rec.task_id)

    keys: list[KeyDivergence] = []
    for key in all_keys:
        kind, task_id, occ = key
        lrec = legacy_keyed.get(key)
        nrec = new_keyed.get(key)
        if lrec is not None and nrec is not None:
            lview = lrec.decision_view()
            nview = nrec.decision_view()
            diverged = [f for f in _DECISION_FIELDS if lview.get(f) != nview.get(f)]
            transport = [f for f in _TRANSPORT_FIELDS if lview.get(f) != nview.get(f)]
            keys.append(
                KeyDivergence(
                    kind=kind,
                    task_id=task_id,
                    occurrence=occ,
                    present_in="both",
                    legacy=lview,
                    new=nview,
                    diverged_fields=diverged,
                    transport_fields=transport,
                )
            )
        elif lrec is not None:
            # Un-joinable iff the new regime fired this kind but never on this
            # task_id — same drift, disjoint task-id namespace (a join
            # artifact), not a new-regime silence.
            unjoinable = bool(new_tids[kind]) and task_id not in new_tids[kind]
            keys.append(
                KeyDivergence(
                    kind=kind,
                    task_id=task_id,
                    occurrence=occ,
                    present_in="legacy_only",
                    legacy=lrec.decision_view(),
                    new=None,
                    diverged_fields=[],
                    unjoinable=unjoinable,
                )
            )
        else:
            assert nrec is not None
            unjoinable = bool(legacy_tids[kind]) and task_id not in legacy_tids[kind]
            keys.append(
                KeyDivergence(
                    kind=kind,
                    task_id=task_id,
                    occurrence=occ,
                    present_in="new_only",
                    legacy=None,
                    new=nrec.decision_view(),
                    diverged_fields=[],
                    unjoinable=unjoinable,
                )
            )

    return DivergenceReport(
        legacy_path=str(legacy_path),
        new_path=str(new_path),
        legacy_count=len(legacy),
        new_count=len(new),
        keys=keys,
    )


def render_two_log_text(report: DivergenceReport) -> str:
    """Human-readable two-log divergence report."""
    lines: list[str] = []
    lines.append("goldfive shadow-diff — legacy vs. new regime (§5.4)")
    lines.append("=" * 64)
    lines.append(f"legacy log:  {report.legacy_path}  ({report.legacy_count} deliveries)")
    lines.append(f"new log:     {report.new_path}  ({report.new_count} deliveries)")
    lines.append("-" * 64)
    lines.append(f"aligned drift keys:        {len(report.keys)}")
    lines.append(f"  diverged:                {len(report.diverged_keys)}")
    lines.append(f"  decision-field diffs:    {len(report.field_divergences)}")
    lines.append(f"  legacy-only (new silent):{len(report.legacy_only)}")
    lines.append(f"  new-only (legacy silent):{len(report.new_only)}")
    lines.append(
        f"  transport-only (channel): {len(report.transport_only_keys)}  "
        "(per-regime transport, not a divergence)"
    )
    unjoinable = report.unjoinable_keys
    lines.append(
        f"  UN-JOINABLE (task-id ns):{len(unjoinable)}  "
        "(disjoint task-id namespaces — NOT a divergence)"
    )
    lines.append("-" * 64)

    if unjoinable:
        lines.append(
            f"WARNING: {len(unjoinable)} drift key(s) could not be joined across "
            "regimes:"
        )
        lines.append(
            "  the two logs fire the same drift kind on DISJOINT task-id "
            "namespaces (e.g. ledger OUTCOME ids vs. forecast ids), so the"
        )
        lines.append(
            "  (kind, task_id, occurrence) key cannot align. These are join "
            "artifacts, NOT decision divergences — do not read them as one"
        )
        lines.append(
            "  regime staying silent. A goal-anchored deliverable id on the "
            "wire would join them (a runtime dependency, not this tool's)."
        )
        for kd in unjoinable:
            lines.append(
                f"  [{kd.kind} / {kd.task_id} #{kd.occurrence}] {kd.present_in} "
                "(un-joinable)"
            )
        lines.append("-" * 64)

    if not report.diverged_keys:
        lines.append("VERDICT: no decision divergence — the two regimes' steering")
        lines.append("decisions are identical on this traffic (transport channel aside).")
        lines.append("(Expected before the behavior PRs land; once PR 7's ladder")
        lines.append("restructure merges, ladder_level diffs appear here.)")
        return "\n".join(lines)

    lines.append(f"VERDICT: {len(report.diverged_keys)} drift key(s) diverge — review below.")
    lines.append("")
    for kd in report.diverged_keys:
        head = f"[{kd.kind} / {kd.task_id} #{kd.occurrence}] {kd.present_in}"
        lines.append(head)
        if kd.present_in == "both":
            for field in kd.diverged_fields:
                lines.append(
                    f"    {field}: legacy={kd.legacy.get(field)!r}  ->  new={kd.new.get(field)!r}"
                )
            for field in kd.transport_fields:
                lines.append(
                    f"    [transport] {field}: legacy={kd.legacy.get(field)!r}  ->  "
                    f"new={kd.new.get(field)!r}  (informational)"
                )
        elif kd.present_in == "legacy_only":
            lines.append(
                f"    legacy signalled ({kd.legacy.get('ladder_level')!r} on "
                f"{kd.legacy.get('channel')!r}); new regime stayed silent."
            )
        else:
            lines.append(
                f"    new regime signalled ({kd.new.get('ladder_level')!r} on "
                f"{kd.new.get('channel')!r}); legacy stayed silent."
            )
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Single-log census + per-event legacy-vs-new derivation
# ---------------------------------------------------------------------------


#: Delivery channels the LEGACY regime rides (the four goldfive-authored
#: dispatch decision points). On a log made of these, ``would_cancel_inflight``
#: IS the legacy regime's own cancel verdict, so the single-event
#: legacy-vs-new derivation below is valid.
_LEGACY_TRANSPORT: frozenset[str] = frozenset(
    {"nudge_replay", "steer_control", "pause_control", "promotion"}
)
#: The channel the NEW (signal) regime rides (PR 6). On a new-regime log,
#: ``would_cancel_inflight`` is the NEW regime's (narrower) cancel verdict —
#: it CANNOT reveal what legacy would have done, so a single-log census of a
#: new-regime log is blind to cancel divergences (see :func:`single_log_report`).
_NEW_TRANSPORT: frozenset[str] = frozenset({"request_context"})


def _record_regime(rec: SignalRecord) -> str:
    """Infer which regime a delivery belongs to from its transport channel.

    Returns ``"new"`` for the request_context channel, ``"legacy"`` otherwise
    (the four legacy dispatch channels; unknown channels default to legacy so
    the derivation stays applicable rather than silently dropping the record).
    """
    return "new" if rec.channel in _NEW_TRANSPORT else "legacy"


def _legacy_vs_new_within_event(rec: SignalRecord) -> dict[str, Any]:
    """Derive legacy-would-do vs. new-would-do from one delivery's payload.

    VALID ONLY on a LEGACY-regime delivery. The ``decision`` payload carries
    the running regime's cancel intent (``would_cancel_inflight``); on a
    legacy log that IS the legacy verdict, so the divergence within a single
    event is: does the legacy regime cancel / swap the plan where the new
    regime would only signal? On a NEW-regime log the same field is the new
    regime's (narrower) verdict and reveals nothing about legacy — the caller
    (:func:`single_log_report`) must gate this to legacy-transport records.
    """
    cancels = bool(rec.decision.get("would_cancel_inflight")) or rec.channel in (
        "steer_control",
        "promotion",
    )
    ladder = rec.ladder_level or rec.decision.get("ladder_level", "") or "signal"
    if cancels:
        # The legacy regime preempts in-flight work / swaps the plan here; the
        # new regime only signals. A genuine decision divergence.
        return {
            "legacy_action": "cancel_reinvoke+plan_swap",
            "new_action": f"signal({ladder}) [no in-flight cancel]",
            "diverges": True,
        }
    # Both regimes only signal — same action, no divergence (the trailing
    # "[no in-flight cancel]" qualifier is cosmetic, not a decision change).
    return {
        "legacy_action": f"signal({ladder})",
        "new_action": f"signal({ladder})",
        "diverges": False,
    }


def single_log_report(records: list[SignalRecord]) -> dict[str, Any]:
    """Census + per-event legacy-vs-new derivation for one shadow log.

    The legacy-vs-new derivation is derived ONLY from LEGACY-transport
    deliveries: ``would_cancel_inflight`` is the *running* regime's cancel
    verdict, so on a NEW-regime (request_context) log it is the new regime's
    narrower verdict and reveals nothing about what legacy would have done.
    Those deliveries are counted as ``undecidable`` and the report flags a
    ``blind_spot`` — a single new-regime log CANNOT surface a cancel
    divergence; use two-log mode (``--legacy``/``--new``) for that.
    """
    by_channel: dict[str, int] = defaultdict(int)
    by_ladder: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    dry = 0
    legacy_transport = 0
    new_transport = 0
    undecidable = 0
    diverging: list[dict[str, Any]] = []
    for rec in records:
        by_channel[rec.channel] += 1
        by_ladder[rec.ladder_level or rec.decision.get("ladder_level", "") or "?"] += 1
        by_kind[rec.kind] += 1
        if rec.dry_run:
            dry += 1
        if _record_regime(rec) == "new":
            # would_cancel_inflight here is the NEW regime's (narrower) cancel
            # verdict; it cannot reveal a legacy cancel divergence.
            new_transport += 1
            undecidable += 1
            continue
        legacy_transport += 1
        derived = _legacy_vs_new_within_event(rec)
        if derived["diverges"]:
            diverging.append(
                {
                    "kind": rec.kind,
                    "task_id": rec.task_id,
                    "drift_id": rec.drift_id,
                    "legacy_action": derived["legacy_action"],
                    "new_action": derived["new_action"],
                }
            )
    if legacy_transport and new_transport:
        regime = "mixed"
    elif new_transport:
        regime = "new"
    else:
        regime = "legacy"
    return {
        "deliveries": len(records),
        "dry_run": dry,
        "real": len(records) - dry,
        "by_channel": dict(by_channel),
        "by_ladder_level": dict(by_ladder),
        "by_kind": dict(by_kind),
        "regime": regime,
        "legacy_transport": legacy_transport,
        "new_transport": new_transport,
        # A single log can only derive legacy-vs-new from legacy-transport
        # deliveries; new-transport deliveries are blind to cancel divergence.
        "divergence_derivable": legacy_transport > 0,
        "blind_spot": new_transport > 0,
        "undecidable_deliveries": undecidable,
        "diverging_events": diverging,
    }


def render_single_log_text(report: dict[str, Any], *, path: str) -> str:
    lines: list[str] = []
    lines.append("goldfive shadow-diff — single-log census (§5.4)")
    lines.append("=" * 64)
    lines.append(f"log:           {path}")
    lines.append(
        f"deliveries:    {report['deliveries']}  "
        f"(dry_run={report['dry_run']}, real={report['real']})"
    )
    lines.append(f"by channel:    {report['by_channel']}")
    lines.append(f"by ladder:     {report['by_ladder_level']}")
    lines.append(f"by kind:       {report['by_kind']}")
    lines.append(
        f"regime:        {report['regime']}  "
        f"(legacy-transport={report['legacy_transport']}, "
        f"new-transport={report['new_transport']})"
    )
    lines.append("-" * 64)
    if report["blind_spot"]:
        lines.append(
            f"BLIND SPOT: {report['undecidable_deliveries']} delivery(ies) ride "
            "the NEW regime's transport (request_context)."
        )
        lines.append(
            "  would_cancel_inflight on those is the NEW regime's own "
            "(narrower) cancel verdict — it CANNOT reveal what legacy would"
        )
        lines.append(
            "  have done, so a single-log census is blind to cancel "
            "divergences here. Use two-log mode (--legacy/--new) to diff them."
        )
        lines.append("-" * 64)
    diverging = report["diverging_events"]
    if not diverging:
        if report["divergence_derivable"]:
            lines.append(
                "no per-event legacy/new divergence derivable from the "
                "legacy-transport payloads."
            )
        else:
            lines.append(
                "no legacy-transport deliveries — no legacy/new divergence is "
                "derivable from this log alone (see BLIND SPOT above)."
            )
        return "\n".join(lines)
    lines.append(f"{len(diverging)} delivery(ies) where legacy != new:")
    for d in diverging:
        lines.append(
            f"  [{d['kind']} / {d['task_id']}] "
            f"legacy={d['legacy_action']}  ->  new={d['new_action']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shadow_diff",
        description=(
            "§5.4 shadow / differential-validation: diff legacy-would-do vs. "
            "new-would-do from goldfive SignalDelivered JSONL logs."
        ),
    )
    p.add_argument(
        "log",
        nargs="?",
        help="single JSONL sink log (single-log census mode)",
    )
    p.add_argument("--legacy", help="legacy-regime JSONL log (two-log mode)")
    p.add_argument("--new", help="new-regime JSONL log (two-log mode)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p.add_argument("--out", help="write the report to this path instead of stdout")
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="do not error on a log with zero SignalDelivered events",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    two_log = bool(args.legacy or args.new)

    try:
        if two_log:
            if not (args.legacy and args.new):
                raise ShadowDiffError("two-log mode needs both --legacy and --new")
            legacy = load_signals(args.legacy, allow_empty=args.allow_empty)
            new = load_signals(args.new, allow_empty=args.allow_empty)
            report = diff_two_logs(
                legacy, new, legacy_path=args.legacy, new_path=args.new
            )
            text = render_two_log_text(report)
            payload = report.to_dict()
        else:
            if not args.log:
                raise ShadowDiffError("provide a LOG (single-log) or --legacy/--new (two-log)")
            records = load_signals(args.log, allow_empty=args.allow_empty)
            payload = single_log_report(records)
            text = render_single_log_text(payload, path=args.log)
    except ShadowDiffError as exc:
        print(f"shadow_diff: error: {exc}", file=sys.stderr)
        return 2

    out = json.dumps(payload, indent=2, sort_keys=True, default=str) if args.json else text
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
