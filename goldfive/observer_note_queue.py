"""The observer-note delivery queue (AGENCY-PRESERVATION.md PR 6).

PR 4 (``goldfive/observer_notes.py``) composes *what* goldfive says to the
wrapped agent — an observation + the user's goal + an advisory footer. PR 6
is the *delivery* layer: where the composed note actually reaches the agent,
and the bookkeeping that makes delivery happen **exactly once** across the
several surfaces a supervisor at the callback layer can reach.

The queue is the shared substrate for four delivery points (the doc's
"Delivery points, in preference order"):

1. ADK ``before_model_callback`` — :meth:`PromptShaper.inject_observer_note`
   renders the most-severe pending note as a marker-bracketed block on the
   request's ``system_instruction`` (reaches a *mid-invocation* agent on its
   next model call — the surface that removes the only remaining
   justification for cancel-as-information-delivery).
2. Invocation-boundary replay — the scoped overlay loop in
   :mod:`goldfive.executors.sequential` consumes this queue instead of
   ``session.pending_nudges`` and re-invokes with the note as the next turn.
3. claude-agent-sdk adapter — a system-prompt section + a ``PostToolUse``
   hook's ``additionalContext``.
4. Tool-result annotation — an append-only, attributed one-liner on the
   function_response of a loop-shaped drift's repeated tool
   (:func:`render_tool_annotation`).

Exactly-once contract (binding — AGENCY-PRESERVATION.md §5.2)
------------------------------------------------------------
A note enqueued while an invocation is mid-flight (delivered via surface 1)
and ALSO still present at the next boundary (surface 2) must render **once,
not twice** — the classic two-mode double-delivery bug. The queue enforces
this with a per-note ``delivered`` flag: the first surface to render a note
marks it delivered; every other surface only ever looks at *pending*
(undelivered) notes (:meth:`peek_for_render`), so a delivered note is never
re-selected. The flag is the single source of truth; the SignalLedger's
``(drift_id, channel)`` dedup (PR 5) is a second, independent layer.

Per-request coalescing
-----------------------
At most ONE rendered block leaves the queue per LLM request, and the
*most-severe* pending note wins (:meth:`peek_for_render`). Lower-severity
notes stay pending and surface on subsequent requests (one per request); no
note is silently dropped. Ties break toward the newest note (higher turn,
later enqueue) so the freshest signal is shown first.

Key discipline (binding — §5.6)
-------------------------------
A note's identity (:attr:`ObserverNote.note_id`) is goldfive-minted and
stable: the originating ``DriftEvent.id`` when present (itself
goldfive-minted), else a content hash. NEVER an LLM-minted identifier — a
churning key would let the same drift re-deliver every turn and defeat the
exactly-once flag. Re-fires of a ``(kind, task)`` carry distinct drift ids
(PR 5's fire model), so each is legitimately a new note; only a true
duplicate emit of the *same* drift id coalesces.

Storage
-------
State lives on ``Session.state`` under :data:`KEY_OBSERVER_NOTE_QUEUE` (a
``goldfive.``-prefixed key so :func:`goldfive.state_store.write` accepts it),
as a JSON-serialisable ``{"notes": [entry_dict, ...]}`` map — the same
StateStore convention the SignalLedger / active-drifts slots follow, so a
sink that round-trips ``Session.state`` keeps working.

Concurrency
-----------
Like the SignalLedger, every mutator is a pure, synchronous read-modify-write
fold over the serialised dict; callers run under goldfive's existing
per-session serialisation. ``enqueue`` is idempotent per ``note_id`` and
``mark_delivered`` is idempotent per note, so even a missed-serialisation
interleaving cannot double-deliver or corrupt the list.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from goldfive import state_store as _ostate

# Single-source observer-note marker constants live in ``observer_notes`` (the
# PR 4 composer module) so the PR 6 channel (writer) and the PR 6b
# ``PruneStaleSteerRule`` (reader) can never drift — both import from there.
# ``OBSERVER_NOTE_BLOCK_BEGIN`` / ``_END`` are the rendered opening / closing
# lines; ``OBSERVER_NOTE_MARKER_PREFIX`` is the detection substring (a prefix
# of the opening line by construction) used as the strip-and-refresh anchor.
from goldfive.observer_notes import (
    OBSERVER_NOTE_BLOCK_BEGIN,
    OBSERVER_NOTE_BLOCK_END,
    OBSERVER_NOTE_MARKER_PREFIX,
)

if TYPE_CHECKING:  # pragma: no cover - type-check only
    from goldfive.types import Session

#: ``Session.state`` key for the observer-note queue. ``goldfive.``-prefixed
#: so the state_store write-guard accepts it; sibling of
#: ``goldfive.signal_ledger`` / ``goldfive.active_drifts``.
KEY_OBSERVER_NOTE_QUEUE = "goldfive.observer_note_queue"

#: ``drift_id`` prefix minted for correction-origin notes (task #11). The
#: SINGLE SOURCE for the prefix: ``_correction_injection`` mints
#: ``correction:<agent>:<task>:<rev>`` and the queue's structural filters
#: (``peek_for_render(exclude_correction_notes=True)``) recognise it — so a
#: correction (agent-targeted) is never surfaced where only loop
#: observations belong (the tool-result annotation).
CORRECTION_DRIFT_ID_PREFIX = "correction:"

#: Cap on the retained note list. Delivered notes are kept as tombstones so
#: ``enqueue`` dedup and the exactly-once flag survive, but a pathologically
#: long-lived session cannot balloon ``Session.state``: on overflow the
#: oldest *delivered* notes are evicted first (they have served their
#: purpose), pending notes are always retained.
_NOTES_CAP = 256

# Severity ordering (mirrors goldfive.state_store._SEVERITY_ORDER) — kept as
# a string map so the queue never has to import the proto enum to rank a
# serialised note.
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}


def _severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(str(severity or "").lower(), -1)


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mint_note_id(*, drift_id: str, kind: str, task_id: str, turn: int, body: str) -> str:
    """Return a stable, goldfive-minted note id.

    Prefers the originating ``DriftEvent.id`` (goldfive-minted upstream); when
    that is empty (deterministic-detector drifts in some test stubs) falls
    back to a content hash of ``(kind, task, turn, body)`` so the id is still
    stable and reproducible — never an LLM-minted identifier (§5.6).
    """
    did = str(drift_id or "").strip()
    if did:
        return did
    payload = f"{kind}|{task_id}|{int(turn)}|{body}".encode()
    return "n_" + hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


@dataclasses.dataclass
class ObserverNote:
    """One advisory note queued for delivery to the wrapped agent.

    ``body`` is the full composed note (observation + goal + status + advisory
    footer) from :func:`goldfive.observer_notes.compose_note_for_drift` — the
    text the block surfaces (1, 2) wrap in markers. ``observation`` is the
    factual observation line alone (no goal/footer), used by the compact
    tool-result annotation (surface 4). ``delivered`` is the exactly-once
    flag; the surface that first renders the note sets it.
    """

    note_id: str
    body: str
    observation: str
    severity: str
    drift_id: str = ""
    kind: str = ""
    task_id: str = ""
    agent_id: str = ""
    turn: int = 0
    ladder_level: str = ""
    enqueued_seq: int = 0
    delivered: bool = False
    delivered_channel: str = ""
    delivered_surface: str = ""
    delivered_turn: int = -1

    @property
    def is_pending(self) -> bool:
        return not self.delivered

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": str(self.note_id or ""),
            "body": str(self.body or ""),
            "observation": str(self.observation or ""),
            "severity": str(self.severity or ""),
            "drift_id": str(self.drift_id or ""),
            "kind": str(self.kind or ""),
            "task_id": str(self.task_id or ""),
            "agent_id": str(self.agent_id or ""),
            "turn": int(self.turn),
            "ladder_level": str(self.ladder_level or ""),
            "enqueued_seq": int(self.enqueued_seq),
            "delivered": bool(self.delivered),
            "delivered_channel": str(self.delivered_channel or ""),
            "delivered_surface": str(self.delivered_surface or ""),
            "delivered_turn": int(self.delivered_turn),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObserverNote:
        return cls(
            note_id=str(data.get("note_id", "") or ""),
            body=str(data.get("body", "") or ""),
            observation=str(data.get("observation", "") or ""),
            severity=str(data.get("severity", "") or ""),
            drift_id=str(data.get("drift_id", "") or ""),
            kind=str(data.get("kind", "") or ""),
            task_id=str(data.get("task_id", "") or ""),
            agent_id=str(data.get("agent_id", "") or ""),
            turn=_coerce_int(data.get("turn", 0)),
            ladder_level=str(data.get("ladder_level", "") or ""),
            enqueued_seq=_coerce_int(data.get("enqueued_seq", 0)),
            delivered=bool(data.get("delivered", False)),
            delivered_channel=str(data.get("delivered_channel", "") or ""),
            delivered_surface=str(data.get("delivered_surface", "") or ""),
            delivered_turn=_coerce_int(data.get("delivered_turn", -1), default=-1),
        )


def _bare_agent(name: Any) -> str:
    """Return the bare (namespace-stripped, lowercased) agent name.

    Agent ids may be fully qualified (``ns:writer``); the bare segment is
    what surfaces resolve and what notes are scoped on. Tolerant of
    ``None`` / non-str.
    """
    try:
        s = str(name or "").strip()
    except Exception:  # noqa: BLE001
        return ""
    if ":" in s:
        s = s.split(":")[-1]
    return s.lower()


def plan_state_line(plan: Any) -> str:
    """Return a factual per-agent open-work line for the note Status fold.

    AGENCY-PRESERVATION.md task #11 (cross-surface plan-state fold) — the
    content the retired Site-3 hint used to inject per turn now rides the
    observer note on EVERY block surface (before_model, boundary-replay,
    claude), composed here so the three surfaces stay identical. Groups
    ``plan.tasks`` by assignee and reports each agent's open
    (non-terminal) vs. no-open-tasks state. Bookkeeping only — no "choose
    the agent" / "do NOT re-invoke" imperative (dropped in PR 9). Returns
    ``""`` when there is no plan / no tasks. Never raises.
    """
    try:
        tasks = getattr(plan, "tasks", None) if plan is not None else None
        if not tasks:
            return ""
        from goldfive.types import TERMINAL_TASK_STATUSES

        by_agent: dict[str, list[int]] = {}
        for t in tasks:
            agent = getattr(t, "assignee_agent_id", "") or "<unassigned>"
            bucket = by_agent.setdefault(agent, [0, 0])  # [open, done]
            status = getattr(t, "status", None)
            if status in TERMINAL_TASK_STATUSES:
                bucket[1] += 1
            else:
                bucket[0] += 1
        frags: list[str] = []
        for agent in sorted(by_agent):
            bare = agent.split(":")[-1] if ":" in agent else agent
            open_n, _done_n = by_agent[agent]
            frags.append(f"{bare}: {open_n} open" if open_n else f"{bare}: no open tasks")
        if not frags:
            return ""
        return "Plan state (goldfive bookkeeping): " + "; ".join(frags) + "."
    except Exception:  # noqa: BLE001
        return ""


def render_block(note: ObserverNote, *, plan: Any = None) -> str:
    """Render ``note`` as the marker-bracketed system-prompt / replay block.

    Shape (AGENCY-PRESERVATION.md PR 6 "Rendered block shape")::

        [GOLDFIVE OBSERVER NOTE — from an external monitoring layer, not from the user]
        Observation: <...>
        The user's goal: <...>
        Status: <...>
        This note is advisory. How to proceed is your decision; ...
        Plan state (goldfive bookkeeping): <...>      (task #11 fold, when ``plan`` given)
        [/GOLDFIVE OBSERVER NOTE]

    The body already carries the advisory footer (composed by PR 4); this
    function adds the attribution header + closing marker, and — when
    ``plan`` is supplied (task #11 cross-surface fold) — a factual
    plan-state line INSIDE the block (so strip-and-refresh removes it as
    one unit). ``plan=None`` keeps the pre-task-#11 shape byte-identical.
    """
    body = (note.body or "").strip()
    extra = plan_state_line(plan) if plan is not None else ""
    inner = f"{body}\n{extra}" if extra else body
    return f"{OBSERVER_NOTE_BLOCK_BEGIN}\n{inner}\n{OBSERVER_NOTE_BLOCK_END}"


def render_tool_annotation(note: ObserverNote) -> str:
    """Render ``note`` as the compact, attributed tool-result annotation.

    Surface 4 (the system-reminder pattern): a single attributed line landing
    adjacent to the evidence at the moment of maximal relevance. Append-only;
    the caller never modifies the real tool result. Uses the factual
    observation line (falling back to the body) so the annotation carries the
    same neutral fact the block would — never an imperative.
    """
    text = (note.observation or note.body or "").strip()
    return f"[goldfive observer: {text}]"


def strip_prior_block(existing: str) -> str:
    """Remove a previously-injected observer-note block from ``existing``.

    Mirrors ``adk_llm_instrumentation._strip_prior_runtime_tools_hint``: the
    block is bracketed by :data:`OBSERVER_NOTE_MARKER_PREFIX` (the detection
    anchor, a prefix of the opening line) / :data:`OBSERVER_NOTE_BLOCK_END`;
    when found both markers and the text between them are removed and orphan
    blank lines collapsed. Returns ``existing`` unchanged when no prior block
    is present. This is the strip half of the strip-and-refresh idempotency
    contract (two consecutive ``before_model`` calls never stack blocks).
    """
    if OBSERVER_NOTE_MARKER_PREFIX not in existing:
        return existing
    start = existing.find(OBSERVER_NOTE_MARKER_PREFIX)
    end = existing.find(OBSERVER_NOTE_BLOCK_END, start)
    if end == -1:
        cleaned = existing[:start]
    else:
        cleaned = existing[:start] + existing[end + len(OBSERVER_NOTE_BLOCK_END) :]
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip("\n")


class ObserverNoteQueue:
    """StateStore-backed queue of advisory observer notes (PR 6).

    Construct via :meth:`for_session` (or directly on a ``state`` mapping for
    tests). Every mutator is a pure, synchronous fold over the serialised dict
    on ``Session.state``; emission of the matching ``SignalDelivered`` event
    stays on the drift observer (callers pair a delivery with an emit) so the
    interleaving tests can hammer the bookkeeping in isolation.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state

    @classmethod
    def for_session(cls, session: Session | None) -> ObserverNoteQueue:
        """Build a queue over ``session.state`` (or a throwaway dict)."""
        state = getattr(session, "state", None)
        if not isinstance(state, MutableMapping):
            state = {}
        return cls(state)

    # -- raw access ------------------------------------------------------

    def _read_notes(self) -> list[dict[str, Any]]:
        raw = _ostate.read(self._state, KEY_OBSERVER_NOTE_QUEUE, {})
        if not isinstance(raw, Mapping):
            return []
        notes = raw.get("notes")
        if not isinstance(notes, list):
            return []
        return [dict(n) for n in notes if isinstance(n, Mapping)]

    def _write_notes(self, notes: list[dict[str, Any]]) -> None:
        _ostate.write(self._state, KEY_OBSERVER_NOTE_QUEUE, {"notes": list(notes)})

    def _next_seq(self, notes: list[dict[str, Any]]) -> int:
        return 1 + max((_coerce_int(n.get("enqueued_seq", 0)) for n in notes), default=0)

    # -- reads -----------------------------------------------------------

    def notes(self) -> list[ObserverNote]:
        return [ObserverNote.from_dict(n) for n in self._read_notes()]

    def pending(self) -> list[ObserverNote]:
        return [n for n in self.notes() if n.is_pending]

    def get(self, note_id: str) -> ObserverNote | None:
        for n in self._read_notes():
            if str(n.get("note_id", "")) == str(note_id):
                return ObserverNote.from_dict(n)
        return None

    def peek_for_render(
        self,
        *,
        kinds: frozenset[str] | None = None,
        agent_id: str | None = None,
        broadcast_only: bool = False,
        exclude_correction_notes: bool = False,
    ) -> ObserverNote | None:
        """Return the most-severe pending note (per-request coalescing).

        The single note that wins the ≤1-block-per-request slot. Ties break
        toward the newest signal (higher turn, then later enqueue order).
        Returns ``None`` when nothing is pending. Pure read — does NOT mark
        anything delivered; the caller marks via :meth:`mark_delivered` once
        it has rendered the note onto its surface.

        ``kinds`` (when provided) restricts selection to notes whose
        ``kind`` is in the set — used by the tool-result annotation surface
        (4) to pick only loop-shaped notes, leaving other notes for the
        block surfaces.

        ``agent_id`` (AGENCY-PRESERVATION.md task #11 — agent-scoped
        delivery) restricts selection so a per-(agent, task) note reaches
        only the right agent's surfaces:

        * ``None`` (the default) — NO agent filter. Every caller that does
          not / cannot resolve the current agent keeps the pre-task-#11
          broadcast behaviour (and every existing test, which passes
          agentless stubs, is unaffected — §5.1).
        * ``"<name>"`` — select only notes whose ``agent_id`` is empty
          (broadcast / coordinator-level) OR matches ``<name>`` by bare
          name (namespace-stripped). An agent-specific note for a
          DIFFERENT agent is skipped, so e.g. a correction enqueued for
          ``writer`` is never rendered onto ``researcher``'s model call.
          The surface that knows its agent (notably the ADK
          ``before_model`` surface) passes it; coarse surfaces leave it
          ``None``.

        ``broadcast_only`` (task #11 — coarse-surface defense) selects ONLY
        broadcast (empty ``agent_id``) notes, skipping every agent-specific
        one. A surface that cannot resolve its agent (the boundary replay,
        which re-invokes at the coordinator level) sets this so an
        agent-specific note is never misdelivered there — it stays pending
        for its own agent's agent-aware surface (better undelivered than
        misdelivered; §0 dormancy bias). Takes precedence over ``agent_id``.

        ``exclude_correction_notes`` (task #11) skips notes whose
        ``drift_id`` carries :data:`CORRECTION_DRIFT_ID_PREFIX` — used by the
        tool-result annotation surface so agent-targeted corrections never
        ride the loop-observation channel, regardless of their drift kind.
        """
        target = _bare_agent(agent_id) if agent_id else None
        best: ObserverNote | None = None
        best_key: tuple[int, int, int] = (-2, -1, -1)
        for n in self.notes():
            if not n.is_pending:
                continue
            if kinds is not None and n.kind not in kinds:
                continue
            if exclude_correction_notes and str(n.drift_id or "").startswith(
                CORRECTION_DRIFT_ID_PREFIX
            ):
                continue
            note_agent = _bare_agent(n.agent_id)
            if broadcast_only:
                # Only broadcast notes here; agent-specific notes wait for
                # their own agent-aware surface.
                if note_agent:
                    continue
            elif target is not None:
                # Empty note_agent = broadcast (reaches any agent); a
                # non-empty agent-specific note must match the target.
                if note_agent and note_agent != target:
                    continue
            key = (_severity_rank(n.severity), int(n.turn), int(n.enqueued_seq))
            if key > best_key:
                best_key = key
                best = n
        return best

    # -- pacing reads (AGENCY-PRESERVATION.md PR 8) -----------------------
    #
    # The grace window keys on VISIBILITY — when a note for a ``(kind, task)``
    # key was actually RENDERED onto a surface (``delivered_turn``, stamped by
    # :meth:`mark_delivered`) — NOT on when it was dispatched/enqueued. Under
    # ``request_context`` a dispatch and its render can be turns apart (the
    # note waits in the queue until a surface peeks it), so the SignalLedger's
    # dispatch-turn record is the wrong clock for "has the agent had time to
    # self-correct since it SAW the signal?" (binding requirement, #462
    # review). PR 8 reads these; PR 6's ``delivered`` flag supplies them.

    @staticmethod
    def _is_signal_note(n: ObserverNote) -> bool:
        """True for a drift-signal note, False for a task-#11 correction note.

        The PR-8 pacing reads (grace window / escalation / attribution) gate
        and attribute goldfive's drift SIGNALS only. A *correction* note
        (task #11 — drift_id carries :data:`CORRECTION_DRIFT_ID_PREFIX`) is a
        distinct mechanism (plan-revision notice on the agent-scoped channel),
        not a drift advisory: it must NOT start a signal grace window, count
        toward signal escalation, or stand in for "the agent saw the SIGNAL"
        in ``after_signal`` attribution. Corrections also carry no SignalLedger
        entry, so excluding them keeps the queue pacing reads aligned with the
        ledger's signal keys.
        """
        return not str(n.drift_id or "").startswith(CORRECTION_DRIFT_ID_PREFIX)

    def last_rendered_turn(self, kind: str, task_id: str) -> int:
        """Return the most-recent SIGNAL render turn for a ``(kind, task)`` key.

        The max ``delivered_turn`` over delivered (rendered) *signal* notes
        matching ``(kind, task_id)`` (correction notes excluded — see
        :meth:`_is_signal_note`); ``-1`` when no signal for that key has been
        rendered yet. This is the grace-window anchor: an enqueued-but-never-
        rendered signal returns ``-1`` (it never started a grace window — the
        agent has not seen it).
        """
        k = str(kind or "")
        t = str(task_id or "")
        best = -1
        for n in self.notes():
            if n.kind == k and n.task_id == t and n.delivered and self._is_signal_note(n):
                best = max(best, int(n.delivered_turn))
        return best

    def signal_count(self, kind: str, task_id: str) -> int:
        """Return the number of SIGNAL notes ENQUEUED for a ``(kind, task)`` key.

        Each enqueue is one signal that passed the upstream gates (re-fires
        suppressed inside a grace window are never enqueued — see the dispatch
        path), so this is the escalation counter: count ``0`` is the first
        signal, ``1`` the second (re-authored quoting the first), and
        ``>= REFINE_FAILURE_THRESHOLD`` escalates to a pause. Correction notes
        (task #11) are excluded — they are not drift signals and must not
        trip signal escalation.
        """
        k = str(kind or "")
        t = str(task_id or "")
        return sum(
            1
            for n in self.notes()
            if n.kind == k and n.task_id == t and self._is_signal_note(n)
        )

    def rendered_keys(self) -> set[tuple[str, str]]:
        """Return every ``(kind, task)`` with at least one RENDERED SIGNAL note.

        The visibility source of truth for ``self_corrected_after_signal``
        attribution (binding requirement): a key resolves ``after_signal`` only
        if the agent actually SAW a drift SIGNAL for it; an
        enqueued-but-never-rendered key resolves ``self_corrected_unaided``.
        Correction notes (task #11) are excluded — a rendered correction is
        not "the agent saw the SIGNAL" (and corrections carry no ledger entry
        to attribute anyway).
        """
        out: set[tuple[str, str]] = set()
        for n in self.notes():
            if n.delivered and self._is_signal_note(n):
                out.add((n.kind, n.task_id))
        return out

    # -- mutators --------------------------------------------------------

    def enqueue(
        self,
        *,
        body: str,
        observation: str,
        severity: str,
        drift_id: str = "",
        kind: str = "",
        task_id: str = "",
        agent_id: str = "",
        turn: int = 0,
        ladder_level: str = "",
    ) -> ObserverNote:
        """Enqueue a composed note for delivery; idempotent per ``note_id``.

        The ``note_id`` is minted from ``drift_id`` (stable, goldfive-minted)
        — re-enqueuing the same drift coalesces onto the existing entry rather
        than duplicating. A pending existing entry refreshes its mutable
        fields (body/observation/severity/turn — latest wins); a *delivered*
        entry is never resurrected (exactly-once). Returns the resulting note.
        """
        notes = self._read_notes()
        note_id = _mint_note_id(
            drift_id=drift_id, kind=kind, task_id=task_id, turn=turn, body=body
        )
        for raw in notes:
            if str(raw.get("note_id", "")) == note_id:
                existing = ObserverNote.from_dict(raw)
                if existing.delivered:
                    # Already delivered — do not resurrect (exactly-once).
                    return existing
                # Refresh mutable fields on the still-pending duplicate.
                existing.body = str(body or "")
                existing.observation = str(observation or "")
                existing.severity = str(severity or "")
                existing.turn = int(turn)
                existing.ladder_level = str(ladder_level or "")
                raw.update(existing.to_dict())
                self._write_notes(notes)
                return existing

        note = ObserverNote(
            note_id=note_id,
            body=str(body or ""),
            observation=str(observation or ""),
            severity=str(severity or ""),
            drift_id=str(drift_id or ""),
            kind=str(kind or ""),
            task_id=str(task_id or ""),
            agent_id=str(agent_id or ""),
            turn=int(turn),
            ladder_level=str(ladder_level or ""),
            enqueued_seq=self._next_seq(notes),
        )
        notes.append(note.to_dict())
        self._prune(notes)
        self._write_notes(notes)
        return note

    def mark_delivered(
        self,
        note_id: str,
        *,
        channel: str,
        turn: int,
        surface: str = "",
    ) -> bool:
        """Mark ``note_id`` delivered; return ``True`` iff newly delivered.

        Idempotent: a second mark (or a mark of an unknown id) returns
        ``False`` and leaves the record untouched. This is the exactly-once
        chokepoint — the first surface to render a note flips the flag, and
        :meth:`peek_for_render` skips it forever after, so no other surface
        re-delivers it. The ``True`` return is the caller's signal to emit
        exactly one ``SignalDelivered``.
        """
        notes = self._read_notes()
        for raw in notes:
            if str(raw.get("note_id", "")) != str(note_id):
                continue
            if bool(raw.get("delivered", False)):
                return False
            raw["delivered"] = True
            raw["delivered_channel"] = str(channel or "")
            raw["delivered_surface"] = str(surface or "")
            raw["delivered_turn"] = _coerce_int(turn)
            self._write_notes(notes)
            return True
        return False

    def _prune(self, notes: list[dict[str, Any]]) -> None:
        """Evict oldest *delivered* notes when over the cap (in place)."""
        if len(notes) <= _NOTES_CAP:
            return
        delivered_idx = [
            i for i, n in enumerate(notes) if bool(n.get("delivered", False))
        ]
        # Oldest delivered first (list order is enqueue order).
        to_drop = len(notes) - _NOTES_CAP
        for i in delivered_idx[:to_drop]:
            notes[i] = None  # type: ignore[call-overload]
        notes[:] = [n for n in notes if n is not None]


__all__ = [
    "CORRECTION_DRIFT_ID_PREFIX",
    "KEY_OBSERVER_NOTE_QUEUE",
    # Re-exported from goldfive.observer_notes (single source) for callers that
    # import the marker constants alongside the queue API.
    "OBSERVER_NOTE_BLOCK_BEGIN",
    "OBSERVER_NOTE_BLOCK_END",
    "OBSERVER_NOTE_MARKER_PREFIX",
    "ObserverNote",
    "ObserverNoteQueue",
    "plan_state_line",
    "render_block",
    "render_tool_annotation",
    "strip_prior_block",
]
