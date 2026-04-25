"""Typed accessor over goldfive ``Session.state`` for orchestration data.

Phase 1 of goldfive#271 — see ``docs/design/STATE-OWNERSHIP-CONTRACT.md``
for the contract this layer enforces, and Phase 0's audit catalog
(``§5`` of that doc) for the full set of writers/readers Phase 2 will
collapse onto this surface.

What this module is
-------------------

A thin, typed handle wrapping goldfive's own
:class:`~goldfive.types.Session.state` dict. Replaces ad-hoc
``session.state.get('goldfive.foo')`` reads and ``session.state[...] = v``
writes scattered across the codebase with named methods + small typed
result objects (``ActiveSteer``, ``DelegationPin``, ``ReasoningBinding``).

What this module is NOT
-----------------------

* **NOT a wrapper over ADK's ``session.state``.** That dict is owned by
  ADK; goldfive callbacks must not mutate it (Phase 0 contract §3.3).
  The ADK-side reads in
  :mod:`~goldfive.adapters._adk_dynainst` and
  :mod:`~goldfive.planners.goldfive_planner` consult ADK's state
  copy that the bridge (Phase 2 migration target V2) maintains; this
  store is the goldfive-internal source of truth that bridge reads
  *from*.
* **NOT a replacement for** :mod:`goldfive.orchestration_state`'s
  primitive read/write helpers. Those keep their existing call sites
  untouched in Phase 1 — this module is a read-side veneer that
  supersedes the *ad-hoc* ``state.get(...)`` callers cataloged in the
  Phase 0 audit. Phase 2 collapses the writers; Phase 3 collapses the
  bridge.
* **NOT exhaustive.** Phase 1's surface covers only the recurring
  operations Phase 1's brief calls out plus the new reasoning-extracted
  binding. Subsequent phases extend the surface as they migrate
  additional sites.

Initial surface (Phase 1)
-------------------------

Reads (replacing ad-hoc ``state.get('goldfive.xxx')`` callers):

* :meth:`OrchestrationStore.pin_current_task` — current task pin id
* :meth:`OrchestrationStore.pin_current_task_revision` — pin revision
* :meth:`OrchestrationStore.get_active_steer` — active steer, typed
* :meth:`OrchestrationStore.get_correction` — pending-correction body
* :meth:`OrchestrationStore.get_pending_delegation` — per-fc_id pin

Writes:

* :meth:`OrchestrationStore.set_pin_current_task` — write the pin
  (with a documented :class:`BindingSource`)
* :meth:`OrchestrationStore.record_reasoning_extracted_binding` —
  the **only NEW write site** introduced by Phase 1. Records the LLM
  judge's stated-intent binding so the pin-resolution ladder's signal
  6 can consume it as a real signal.
* :meth:`OrchestrationStore.clear_reasoning_extracted_binding` —
  companion clear (called on task transition).

Reasoning-extracted bindings
----------------------------

A new orchestration-state slot,
``goldfive.reasoning_extracted_bindings`` (a dict keyed by agent name),
records the LLM judge's stated-intent attribution from
:func:`~goldfive.drift.reasoning_judge.classify_reasoning_drift`. When
the judge returns ``focused_task_id`` + ``focus_confidence`` and the
confidence is above a configured threshold, the steerer records a
binding here; the pin-resolution ladder's signal 6 consults it.

Threshold semantics:

* The threshold is a steerer-level config knob (default ``0.7``); the
  store stamps whatever confidence it's given and lets the consumer
  decide.
* The recorded binding includes the originating ``run_id`` /
  ``session_id`` / ``agent_name`` / ``task_id`` / ``confidence``
  plus a ``recorded_at_turn`` so consumers can dismiss stale bindings.

Surface evolution
-----------------

Each Phase-2 migration of a writer adds a corresponding ``set_*`` /
``clear_*`` method here. As the bridge collapses (Phase 3) the read
side migrates to take a ``ReadonlyContext`` adapter; until then the
reads return cached views of goldfive ``Session.state`` and the bridge
copies them onto ADK's state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from goldfive import orchestration_state as _ostate
from goldfive.adapters import _adk_dynainst as _dynainst

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from goldfive.types import Session


__all__ = [
    "ActiveSteer",
    "BindingSource",
    "DelegationPin",
    "OrchestrationStore",
    "REASONING_BINDINGS_KEY",
    "ReasoningBinding",
]


# State key for the new reasoning-extracted bindings slot. Lives under
# the goldfive prefix so :func:`goldfive.orchestration_state.write`
# accepts it. Value shape: ``dict[agent_name, ReasoningBinding-as-dict]``
# for cheap JSON serialisation by sinks that round-trip the state dict.
REASONING_BINDINGS_KEY = "goldfive.reasoning_extracted_bindings"


# ---------------------------------------------------------------------------
# Typed result objects
# ---------------------------------------------------------------------------


class BindingSource(StrEnum):
    """Origin of a current-task pin write.

    Stamped onto the pin so the pin-resolution ladder's events can
    distinguish a pin set by an agent-turn callback from one set by
    a delegation site, a steerer rotation, or a reasoning-extracted
    binding.

    Phase 1 only consumes the value for observability + the new
    reasoning-extracted-binding signal in the pin ladder; Phase 2's
    full migration will use it to attribute every catalogued writer.
    """

    DELEGATION_PIN = "delegation_pin"
    AGENT_CALLBACK = "agent_callback"
    REASONING = "reasoning"
    CORRECTION_TARGET = "correction_target"
    STEERER_ROTATION = "steerer_rotation"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class ActiveSteer:
    """Typed view of the active steer slots on goldfive ``Session.state``.

    Wraps the four
    :data:`~goldfive.orchestration_state.KEY_ACTIVE_STEER_BODY` /
    ``_AT_TURN`` / ``_AUTHOR`` / ``_SOURCE`` keys so callers don't
    string-fish through the state dict.
    """

    body: str
    at_turn: int
    author: str
    source: str  # "user" | "goldfive" | ""

    def is_active(self) -> bool:
        """True when a steer is currently set (non-empty body)."""
        return bool(self.body)


@dataclasses.dataclass(frozen=True)
class DelegationPin:
    """Typed view of a single ``goldfive.pending_delegations`` entry.

    The on-disk shape supports both legacy bare-string and the
    versioned ``{task_id, revision, tool_args}`` dict (goldfive#266 /
    F7). This dataclass normalises both shapes for callers.
    """

    task_id: str
    revision: int = 0
    tool_args: Mapping[str, Any] | None = None

    def is_set(self) -> bool:
        return bool(self.task_id)


@dataclasses.dataclass(frozen=True)
class ReasoningBinding:
    """Typed view of a reasoning-extracted binding.

    Persisted as a plain dict under
    :data:`REASONING_BINDINGS_KEY[agent_name]` so sinks can round-trip
    the state dict; this dataclass is the in-process view.

    ``recorded_at_turn`` is the session sequence value at the moment
    the binding was recorded; consumers can compare it against the
    current sequence to dismiss bindings older than N turns. ``0``
    means "no sequence recorded" (e.g. test fixture without a session
    sequence counter).
    """

    agent_name: str
    task_id: str
    confidence: float
    recorded_at_turn: int = 0
    run_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "confidence": float(self.confidence),
            "recorded_at_turn": int(self.recorded_at_turn),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReasoningBinding | None:
        """Construct from a state-dict value, or ``None`` on garbage.

        Tolerant of partial dicts (sink-shaped, persistence-restored)
        so a missing ``run_id`` / ``session_id`` doesn't drop a
        legitimate binding.
        """
        if not isinstance(raw, Mapping):
            return None
        try:
            task_id = str(raw.get("task_id", "") or "")
            confidence = float(raw.get("confidence", 0.0) or 0.0)
            agent_name = str(raw.get("agent_name", "") or "")
        except (TypeError, ValueError):
            return None
        if not task_id:
            return None
        try:
            recorded_at_turn = int(raw.get("recorded_at_turn", 0) or 0)
        except (TypeError, ValueError):
            recorded_at_turn = 0
        return cls(
            agent_name=agent_name,
            task_id=task_id,
            confidence=confidence,
            recorded_at_turn=recorded_at_turn,
            run_id=str(raw.get("run_id", "") or ""),
            session_id=str(raw.get("session_id", "") or ""),
        )


# ---------------------------------------------------------------------------
# OrchestrationStore — typed handle
# ---------------------------------------------------------------------------


class OrchestrationStore:
    """Typed handle over goldfive ``Session.state``.

    Construct with :meth:`for_session` (when you have a goldfive
    :class:`~goldfive.types.Session`) or :meth:`for_state` (when you
    only have the state dict, e.g. in test scaffolding or in callbacks
    that reach the goldfive Session via
    :data:`~goldfive.adapters._adk_plugin.SESSION_CONTEXT_STATE_KEY`).

    All accessors are forgiving: missing / malformed entries return
    typed defaults rather than raising. The store is a *view* — it
    does not own the underlying dict and concurrent writers can mutate
    it between calls. Callers who need a snapshot should call once and
    cache.
    """

    __slots__ = ("_state",)

    def __init__(self, state: Any) -> None:
        # We accept any mapping-shaped object — goldfive's Session.state
        # is a plain dict; tests sometimes pass a ``MappingProxyType``
        # over the same dict. Read paths tolerate both; write paths
        # require :class:`MutableMapping` (raised lazily by the helper).
        self._state = state if isinstance(state, Mapping) else {}

    # -- Constructors ----------------------------------------------------

    @classmethod
    def for_session(cls, session: Session | None) -> OrchestrationStore:
        """Build a store backed by ``session.state``.

        ``None`` yields an empty-state store (writes are silently
        dropped) so callers who can't guarantee a session — e.g.
        defensive paths inside ADK callbacks — never raise.
        """
        if session is None:
            return cls({})
        return cls(getattr(session, "state", {}))

    @classmethod
    def for_state(cls, state: Any) -> OrchestrationStore:
        """Build a store backed by an arbitrary state dict."""
        return cls(state)

    # -- Internal helpers -----------------------------------------------

    def _get(self, key: str, default: Any = None) -> Any:
        return _ostate.read(self._state, key, default)

    # -- Read: pin -------------------------------------------------------

    def pin_current_task(self) -> str:
        """Return ``goldfive.current_task_id``, or ``""`` when unset.

        Replaces ``state.get('goldfive.current_task_id', '')`` with
        a typed accessor. Phase 1's primary read-migration target on
        the goldfive side.
        """
        value = self._get(_ostate.KEY_CURRENT_TASK_ID, "")
        if isinstance(value, str):
            return value
        return ""

    def pin_current_task_title(self) -> str:
        """Return ``goldfive.current_task_title``, or ``""``."""
        value = self._get(_ostate.KEY_CURRENT_TASK_TITLE, "")
        if isinstance(value, str):
            return value
        return ""

    def pin_current_task_revision(self) -> int:
        """Return the revision stamp on the current pin (0 when unset)."""
        return _ostate.read_current_task_revision(self._state)

    # -- Write: pin ------------------------------------------------------

    def set_pin_current_task(
        self,
        task_id: str,
        *,
        source: BindingSource = BindingSource.UNKNOWN,
        revision: int | None = None,
        title: str = "",
    ) -> None:
        """Stamp the current-task pin.

        ``source`` is the :class:`BindingSource` documenting which
        ladder-rung / callback wrote the pin. Phase 1 uses it for
        observability only — Phase 2 wires it through every catalogued
        writer for full attribution. Passing ``BindingSource.UNKNOWN``
        is fine for code paths whose attribution hasn't been migrated
        yet.

        ``revision`` (when provided) stamps
        ``goldfive.current_task_revision`` alongside the id.
        ``None`` leaves the existing revision untouched.

        No-op when ``task_id`` is empty — callers should use
        :meth:`clear_pin_current_task` to clear.
        """
        if not task_id:
            return
        if not isinstance(self._state, dict):
            # Read-only state (e.g. a MappingProxyType view); silently
            # drop the write so the caller's defensive path remains
            # safe. Production state is always a mutable dict.
            return
        # Reuse the orchestration_state primitives so the
        # ``goldfive.*``-prefix assertion still fires; this keeps the
        # store's writes funneled through the same place the catalog
        # is verified against.
        _ostate.write(self._state, _ostate.KEY_CURRENT_TASK_ID, str(task_id))
        if title:
            _ostate.write(self._state, _ostate.KEY_CURRENT_TASK_TITLE, str(title))
        if revision is not None:
            _ostate.stamp_current_task_revision(self._state, int(revision))
        # ``source`` is recorded as part of the binding registry below
        # rather than on the pin slot itself; the pin slot pre-dates
        # the source vocabulary and Phase 2 will collapse the two.
        _ = source  # documented for migration; not stored on the pin slot

    def clear_pin_current_task(self) -> None:
        """Clear the current-task pin slots. Idempotent."""
        if not isinstance(self._state, dict):
            return
        _ostate.clear_current_task(self._state)

    # -- Read: active steer ---------------------------------------------

    def get_active_steer(self) -> ActiveSteer | None:
        """Return the active steer, or ``None`` when no steer is set.

        Replaces the four-key dance
        ``state.get(KEY_ACTIVE_STEER_BODY, '')`` etc. with one typed
        read. ``None`` is returned when the body is empty (the canonical
        "no steer" signal) so callers can ``if store.get_active_steer():``
        without re-checking ``.body``.
        """
        body = self._get(_ostate.KEY_ACTIVE_STEER_BODY, "")
        if not isinstance(body, str) or not body:
            return None
        try:
            at_turn = int(self._get(_ostate.KEY_ACTIVE_STEER_AT_TURN, 0) or 0)
        except (TypeError, ValueError):
            at_turn = 0
        author_raw = self._get(_ostate.KEY_ACTIVE_STEER_AUTHOR, "")
        source_raw = self._get(_ostate.KEY_ACTIVE_STEER_SOURCE, "")
        return ActiveSteer(
            body=body,
            at_turn=at_turn,
            author=str(author_raw or ""),
            source=str(source_raw or ""),
        )

    # -- Read: goals summary --------------------------------------------

    def goals_summary(self) -> str:
        """Return ``goldfive.goals_summary``, or ``""`` when unset.

        Pre-formatted comma-joined string maintained by
        :func:`goldfive.orchestration_state.refresh_goals_summary`. The
        planner consumes this value verbatim for its per-turn
        instruction block.
        """
        value = self._get(_ostate.KEY_GOALS_SUMMARY, "")
        if isinstance(value, str):
            return value
        return ""

    # -- Read: cancelled function-call ids -------------------------------

    def cancelled_function_call_ids(self) -> list[str]:
        """Return the list of cancelled ``function_call`` ids.

        Reuses :func:`goldfive.orchestration_state.read_cancelled_function_call_ids`
        so the list-shape guard (non-list -> ``[]``) is centralised.
        """
        return _ostate.read_cancelled_function_call_ids(self._state)

    # -- Read: correction ------------------------------------------------

    def get_correction(self, agent_name: str, task_id: str) -> Any:
        """Return the pending-correction value for an ``(agent, task)``.

        The shape is whatever
        :mod:`goldfive._correction_injection` writes — typically a
        :class:`Mapping` with ``superseded_task_id`` /
        ``revision_number`` / etc. (rendered via
        :func:`goldfive.adapters._adk_dynainst.format_correction_block`)
        but tests / external callers may have written a pre-rendered
        string. ``None`` means no pending correction.
        """
        if not agent_name or not task_id:
            return None
        key = _dynainst.pending_correction_key(agent_name, task_id)
        return self._get(key, None)

    def has_correction(self, agent_name: str, task_id: str) -> bool:
        """True when a pending correction exists for ``(agent, task)``."""
        return self.get_correction(agent_name, task_id) is not None

    def iter_corrections_for_agent(self, agent_name: str) -> list[str]:
        """Return every task_id with a pending correction for ``agent_name``.

        Used by pin-ladder signal 6 to enumerate correction targets
        without rebuilding the prefix-matching loop at every call site.
        """
        if not agent_name or not isinstance(self._state, Mapping):
            return []
        # Strip a compound prefix so callers passing the bare or the
        # compound form both find the writer's bare-form keys. Mirrors
        # the matching the existing ``_task_from_pending_correction``
        # helper does inline today.
        bare = agent_name.rsplit(":", 1)[-1]
        prefix = f"goldfive.pending_corrections.{bare}."
        out: list[str] = []
        for key in self._state:
            if not isinstance(key, str):
                continue
            if not key.startswith(prefix):
                continue
            tid = key[len(prefix):]
            if tid:
                out.append(tid)
        return out

    # -- Read: pending delegations --------------------------------------

    def get_pending_delegation(self, fc_id: str) -> DelegationPin | None:
        """Return the per-``function_call_id`` delegation pin, or ``None``.

        Tolerant of both legacy bare-string entries and the versioned
        ``{task_id, revision, tool_args}`` dict shape (goldfive#266 +
        F7). Callers used to inline the shape-test; this normalises.
        """
        if not fc_id:
            return None
        pend = self._get("goldfive.pending_delegations", None)
        if not isinstance(pend, Mapping):
            return None
        raw = pend.get(fc_id)
        if raw is None:
            return None
        if isinstance(raw, str):
            tid = raw.strip()
            if not tid:
                return None
            return DelegationPin(task_id=tid)
        if isinstance(raw, Mapping):
            tid = str(raw.get("task_id", "") or "").strip()
            if not tid:
                return None
            try:
                rev = int(raw.get("revision", 0) or 0)
            except (TypeError, ValueError):
                rev = 0
            args = raw.get("tool_args")
            if not isinstance(args, Mapping):
                args = None
            return DelegationPin(task_id=tid, revision=rev, tool_args=args)
        return None

    # -- Read: reasoning-extracted bindings (NEW Phase 1) ---------------

    def get_reasoning_extracted_binding(
        self,
        agent_name: str,
    ) -> ReasoningBinding | None:
        """Return the most recent reasoning-extracted binding for ``agent_name``.

        Phase 1's NEW read path — consumed by the pin-resolution
        ladder's signal 6 (and by future correction / drift logic that
        wants to consult the LLM judge's stated-intent attribution).

        Returns ``None`` when no binding exists for the agent (or when
        the recorded entry is malformed). Callers gate on confidence
        themselves; the store stamps whatever the writer recorded.

        The lookup strips compound-form prefixes the same way
        :meth:`iter_corrections_for_agent` does, so a compound
        ``"client42:agent_x"`` finds the bare ``agent_x`` binding the
        judge recorded.
        """
        if not agent_name:
            return None
        registry = self._get(REASONING_BINDINGS_KEY, None)
        if not isinstance(registry, Mapping):
            return None
        # Try the exact form, then the bare-form fallback so
        # compound-named callers (``"client:foo"``) still match a bare
        # binding the judge recorded for ``foo``.
        raw = registry.get(agent_name)
        if raw is None:
            bare = agent_name.rsplit(":", 1)[-1]
            if bare and bare != agent_name:
                raw = registry.get(bare)
        if raw is None:
            return None
        return ReasoningBinding.from_dict(raw)

    # -- Write: reasoning-extracted bindings (NEW Phase 1) --------------

    def record_reasoning_extracted_binding(
        self,
        *,
        agent_name: str,
        task_id: str,
        confidence: float,
        recorded_at_turn: int = 0,
        run_id: str = "",
        session_id: str = "",
    ) -> ReasoningBinding | None:
        """Stamp a reasoning-extracted binding for ``agent_name``.

        Phase 1's only NEW write site. Called by the steerer's reasoning
        observation path when
        :func:`~goldfive.drift.reasoning_judge.classify_reasoning_drift`
        returns a ``focused_task_id`` with confidence above the
        configured threshold.

        Returns the recorded :class:`ReasoningBinding` (for caller
        observability) or ``None`` when the inputs were rejected
        (empty ``agent_name`` / ``task_id``, or read-only state).
        Confidence is clamped to ``[0.0, 1.0]``.
        """
        if not agent_name or not task_id:
            return None
        if not isinstance(self._state, dict):
            return None
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        binding = ReasoningBinding(
            agent_name=str(agent_name),
            task_id=str(task_id),
            confidence=conf,
            recorded_at_turn=int(recorded_at_turn),
            run_id=str(run_id or ""),
            session_id=str(session_id or ""),
        )
        # Read-modify-write the registry so unrelated agents' bindings
        # are preserved. Tolerate a non-dict existing value (legacy
        # state shape) by replacing it with a fresh registry rather
        # than appending to a malformed structure.
        registry = self._state.get(REASONING_BINDINGS_KEY)
        if not isinstance(registry, dict):
            registry = {}
        # Stamp under the bare form so :meth:`get_reasoning_extracted_binding`
        # finds it for both compound and bare lookups. The judge owns
        # the agent-name normalisation policy; the store records the
        # form the caller supplied (callers normalise before calling).
        registry[binding.agent_name] = binding.to_dict()
        _ostate.write(self._state, REASONING_BINDINGS_KEY, registry)
        return binding

    def clear_reasoning_extracted_binding(self, agent_name: str) -> None:
        """Drop the binding for ``agent_name`` (idempotent).

        Called on task transition for the agent so a binding that
        targeted the now-completed task doesn't leak forward and
        mis-pin the next invocation.
        """
        if not agent_name or not isinstance(self._state, dict):
            return
        registry = self._state.get(REASONING_BINDINGS_KEY)
        if not isinstance(registry, dict):
            return
        registry.pop(agent_name, None)
        # Also try the bare form in case the writer used compound but
        # the caller is clearing with the bare form (or vice-versa).
        bare = agent_name.rsplit(":", 1)[-1]
        if bare and bare != agent_name:
            registry.pop(bare, None)
        _ostate.write(self._state, REASONING_BINDINGS_KEY, registry)
