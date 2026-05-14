"""Tool-call loop drift detector (goldfive#181, #204).

Post-steer replays on weaker models occasionally degenerate into a
pattern where the coordinator / sub-agent LLM emits the same
``function_call`` (or a near-same one) over and over without advancing
any task. Observed empirically on Qwen runs where a single stuck
invocation consumed ~30 minutes of wall-clock time before a max-call
budget tripped the ADK runner. The existing
:data:`~goldfive.types.DriftKind.LOOPING_REASONING` detector in
:mod:`goldfive.drift.reasoning` watches LLM *text* output for
hash / cosine loops -- it does NOT look at the function_call stream,
so tight tool-loops that never emit the same reasoning text slip
through.

Before goldfive#206 an args-aware :class:`ToolLoopGuard` covered the
reporting-tool slice (calls dispatched via
:func:`~goldfive.adapters._tool_invocation.invoke_tool`). That guard
has been retired; this tracker is now the sole tool-loop detector
goldfive ships and sees **every** tool call the ADK plugin observes
(reporting tools plus AgentTool delegations, MCP tools, custom
adapter-native tools).

This module is the complementary detector: it observes **every** tool
call the ADK plugin sees (reporting or otherwise), keyed per
``(invocation_id, agent_name)``, and fires a
``LOOPING_REASONING``-kind :class:`~goldfive.types.DriftEvent` when
one of several patterns is detected.

Graduated severity (goldfive#204)
---------------------------------

Not every tool loop is the same. ``report_task_completed`` retrying 3
times is probably the agent *reporting* state it's already reported
(cheap, often idempotent on a healthy handler). ``web_developer_agent``
retrying 3 times is a *work* loop burning LLM tokens.

The tracker now classifies each tool call into one of two categories
via :func:`_classify_tool_category`:

* **meta** -- progress-reporting / metadata tools (``report_task_*``,
  ``report_awaiting_approval``). Retries here are usually cheap and
  benign.
* **work** -- every other tool. Retries here are expensive.

Each category has its own graduated severity ladder with three tiers.
At classify time the tracker picks the **highest** tier matched by the
current window and emits one drift at that severity -- it does NOT
cascade INFO + WARNING + CRITICAL on the same window:

+----------+-------+---------+---------+----------+
| Category | Tier  | INFO    | WARNING | CRITICAL |
+==========+=======+=========+=========+==========+
| meta     | exact | 3       | 6       | 10       |
+----------+-------+---------+---------+----------+
| meta     | name  | (none)  | (none)  | (none)   |
+----------+-------+---------+---------+----------+
| work     | exact | 3       | 3       | 6        |
+----------+-------+---------+---------+----------+
| work     | name  | (none)  | 5       | 7        |
+----------+-------+---------+---------+----------+

(The ``exact`` column counts identical ``(name, args_hash)`` signatures
in the window; the ``name`` column counts same-``name``-any-args
signatures. Category is determined by the tool being matched, not by
any other call in the window.)

Work-tier thresholds are chosen to be backwards-compatible with the
pre-#204 detector: three identical work calls still fire WARNING (what
the single-threshold detector did), and same-name 5-in-window still
fires WARNING. CRITICAL is new -- escalates plan revision into cancel-
reinvoke at higher counts. Meta-tier thresholds push the first WARNING
out from 3 to 6 so benign ``report_task_*`` retries trigger only an
INFO drift (OBSERVE -- no plan mutation) until the loop genuinely
persists.

The alternating-cycle mode (A,B,A,B,A in the tail) remains INFO-only
and is independent of category -- it's a textural signal that can
involve either tool kind and the response should be to observe, not
intervene.

Mode 2's "no task progress" gate (same-name-any-args) is satisfied by
calling :meth:`ToolLoopTracker.on_task_progress` whenever a task
transitions RUNNING -> COMPLETED (or any other meaningful progress
signal). That clears the per-agent buffer so a legitimate burst of the
same tool (e.g. a scripted lint + build + test pipeline) is not
flagged when each call completes its step.

.. note::

   The exemption for ``report_task_*`` / ``report_awaiting_approval``
   calls is **success-conditional**. The ADK plugin's
   ``after_tool_callback`` only invokes :meth:`on_task_progress` when
   the tool response indicates an acknowledged success
   (``{"acknowledged": True, ...}`` with no ``error`` key).
   Errored progress reports (missing ``task_id``, malformed payload,
   etc.) are treated as ordinary tool calls and DO count toward
   loop detection. This closes goldfive#192 where an agent stuck
   retrying ``report_task_started`` with a bad ``task_id`` received
   16 consecutive ``missing_task_id`` errors and the detector never
   fired because the old unconditional-reset behaviour exempted them
   on the call alone. The classifier here is unchanged; the gate
   lives in the plugin so ``on_task_progress`` still resets the
   window unconditionally when called directly (tests and any
   future direct callers).

Isolation: trackers are keyed on ``(scope_key, agent_name)`` where
``scope_key`` is the ``session_run_id`` when the caller supplies one
(the ADK plugin does, threading ``session.run_id`` through), and the
ADK ``invocation_id`` otherwise (legacy callers / tests). Scoping on
``session.run_id`` is the goldfive#420 fix: when a coordinator
re-invokes the same sub-agent multiple times within a session (the
debugger_agent / re-delegation pattern empirically observed on the
cherry-tree e2e), each re-invocation got a fresh ``invocation_id`` —
so 20 tool calls spread across 11 invocations only saw ~2 calls per
bucket, never tripping the configured CRITICAL threshold (7 same-name
calls). Re-keying on ``(session.run_id, agent_name)`` lets the
detector see the cumulative window across re-invocations of the same
agent within the same run. Parallel sub-agents within the same run
remain isolated via ``agent_name``. State is intentionally held in a
single :class:`ToolLoopTracker` instance on the plugin -- we don't
need per-session persistence, the whole window is ephemeral to one
run.

Configuration
-------------

The window size and the alternating-pattern length remain env-
overridable (``GOLDFIVE_TOOL_LOOP_WINDOW`` and
``GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD``). The graduated category
thresholds are module-level constants (:data:`_META_THRESHOLDS`,
:data:`_WORK_THRESHOLDS`) grouped so a future PR can surface them via
``ServerConfig`` / env vars if ops needs to tune. The legacy
``GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD`` / ``GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD``
vars are still read by :func:`load_thresholds_from_env` for backwards
compatibility and override the **work** category's WARNING tier
(preserving the pre-#204 single-threshold semantics).

The detector is intentionally deterministic (no embeddings, no LLM
calls). O(1) per tool call modulo the `window` length.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity

if TYPE_CHECKING:
    from goldfive.config import ToolLoopConfig

__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_EXACT_THRESHOLD",
    "DEFAULT_NAME_THRESHOLD",
    "DEFAULT_ALTERNATING_THRESHOLD",
    "ToolLoopTracker",
    "args_hash",
    "load_thresholds_from_env",
    "thresholds_from_config",
]


log = logging.getLogger("goldfive.drift.tool_loops")


# Installed per-process tool-loop config (goldfive#225). When non-None,
# :func:`resolve_thresholds` returns the config's fields; otherwise it
# falls back to :func:`load_thresholds_from_env`. Process-wide for the
# same reason :mod:`goldfive.drift.reasoning` is: the ADK plugin builds
# its tracker at plugin-init time before any Runner context is bound,
# and we want ``goldfive.wrap(runtime=...)`` to influence the tracker
# the plugin builds for that Runner. Operators who need per-Runner
# isolation in a multi-Runner process can call :func:`configure` just
# before ``wrap()`` for each Runner.
_CONFIG: ToolLoopConfig | None = None


def configure(config: ToolLoopConfig | None) -> None:
    """Install a :class:`~goldfive.config.ToolLoopConfig` for this process.

    Called by :func:`goldfive.wrap` with ``runtime.tool_loops``.
    Passing ``None`` reverts to env-driven / default behaviour.
    """
    global _CONFIG
    _CONFIG = config


def resolve_thresholds() -> dict[str, int]:
    """Return tracker-constructor kwargs sourced from the active config.

    Precedence: the installed :class:`~goldfive.config.ToolLoopConfig`
    (via :func:`configure`) wins over :func:`load_thresholds_from_env`
    which in turn wins over the module-level defaults. This is the
    helper plugins should splat into :class:`ToolLoopTracker` under
    goldfive#225.
    """
    if _CONFIG is not None:
        return thresholds_from_config(_CONFIG)
    return load_thresholds_from_env()


# ---------------------------------------------------------------------------
# Tool-category classifier (goldfive#204)
# ---------------------------------------------------------------------------

#: Tool-name prefixes that identify "meta" (progress-reporting) tools.
#: Kept as a tuple so ``str.startswith`` can match any element in one
#: call. Covers ``report_task_started``, ``_progress``, ``_completed``,
#: ``_failed``, ``_blocked``.
_META_TOOL_PREFIXES: tuple[str, ...] = ("report_task_",)

#: Tool-name literals that identify meta tools without a ``report_task_``
#: prefix. ``report_awaiting_approval`` is metadata for an approval
#: state transition -- same "not real work" character.
_META_TOOL_NAMES: frozenset[str] = frozenset({"report_awaiting_approval"})


def _classify_tool_category(tool_name: str) -> str:
    """Return ``"meta"`` for progress-reporting / metadata tools, ``"work"`` otherwise.

    Meta tools don't advance task state; their retries are cheap and
    often benign (the healthy-path handlers are idempotent -- a repeat
    ``report_task_completed`` on an already-completed task ACKs without
    mutating anything). Work tools burn real LLM time / tokens / have
    side effects, so loops there are expensive and the detector should
    escalate sooner.

    The classifier is intentionally name-only so it's cheap and
    deterministic -- no args inspection, no runtime registry lookup.
    Future meta tools added by adapters should either follow the
    ``report_task_`` prefix convention or be added to
    :data:`_META_TOOL_NAMES` explicitly.
    """
    if not tool_name:
        return "work"
    if tool_name.startswith(_META_TOOL_PREFIXES) or tool_name in _META_TOOL_NAMES:
        return "meta"
    return "work"


# ---------------------------------------------------------------------------
# Graduated thresholds per category (goldfive#204)
# ---------------------------------------------------------------------------
#
# Each category exposes three tiers. Each tier has an ``"exact"`` count
# (minimum identical ``(name, args_hash)`` repeats in the window) and a
# ``"name"`` count (minimum same-``name``-any-args repeats in the
# window). ``None`` means that tier/axis does not fire.
#
# Reading the table:
#
# * At each observation, find the highest tier (CRITICAL > WARNING >
#   INFO) whose threshold is matched by the window, emit one drift at
#   that severity, and stop. Do NOT fire all three when thresholds
#   stack.
# * "exact" match is checked first; "name" match is checked only if
#   "exact" did not already classify the tool at the same-or-higher
#   severity (this preserves the pre-#204 "exact preempts name on the
#   same tool" suppression rule).
# * Thresholds chosen so the **work** category's WARNING tier is
#   backwards-compatible with the pre-#204 single-threshold defaults
#   (exact=3, name=5). CRITICAL is new at 6/7 respectively. Meta
#   thresholds push WARNING from 3 to 6 so benign reporting retries
#   produce only an INFO drift (OBSERVE) until the loop persists.

_META_THRESHOLDS: dict[str, dict[str, int | None]] = {
    "info": {"exact": 3, "name": None},
    "warning": {"exact": 6, "name": None},
    "critical": {"exact": 10, "name": None},
}

_WORK_THRESHOLDS: dict[str, dict[str, int | None]] = {
    "info": {"exact": 3, "name": None},
    "warning": {"exact": 3, "name": 5},
    "critical": {"exact": 6, "name": 7},
}

#: Severity tiers in ascending order. Used by :meth:`ToolLoopTracker._classify`
#: to walk from highest to lowest when picking the single tier to emit.
_SEVERITY_TIERS: tuple[tuple[str, DriftSeverity], ...] = (
    ("critical", DriftSeverity.CRITICAL),
    ("warning", DriftSeverity.WARNING),
    ("info", DriftSeverity.INFO),
)


# ---------------------------------------------------------------------------
# Legacy tunables (retained for env-override compatibility + docs)
# ---------------------------------------------------------------------------

#: Ring-buffer size: how many recent tool calls to retain per
#: ``(invocation_id, agent_name)`` key. ``10`` is the default so the
#: meta-CRITICAL 10-in-window check has room to fire; name-repeat and
#: alternating checks are unaffected since they only inspect the last
#: ``name_threshold`` / ``alternating_threshold`` slots.
DEFAULT_WINDOW: int = 10

#: Legacy alias for the work-category WARNING exact threshold. Retained
#: so callers (and tests) that read the old single-threshold constant
#: keep working, and so
#: :envvar:`GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` still has somewhere to
#: land.
DEFAULT_EXACT_THRESHOLD: int = 3

#: Legacy alias for the work-category WARNING name threshold. Same
#: rationale as above.
DEFAULT_NAME_THRESHOLD: int = 5

#: Mode 3 threshold: alternating pattern length (A,B,A,B,A == 5).
DEFAULT_ALTERNATING_THRESHOLD: int = 5


def _read_int_env(name: str, default: int) -> int:
    """Best-effort integer env override; returns default on parse failure."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        log.debug(
            "tool-loop detector: ignoring non-integer %s=%r (using default %d)",
            name,
            raw,
            default,
        )
        return default
    if val <= 0:
        log.debug(
            "tool-loop detector: ignoring non-positive %s=%d (using default %d)",
            name,
            val,
            default,
        )
        return default
    return val


def load_thresholds_from_env() -> dict[str, int]:
    """Read ``GOLDFIVE_TOOL_LOOP_*`` env vars; return overrides dict.

    Suitable for ``**kwargs``-splatting into :class:`ToolLoopTracker`.
    Missing / malformed vars fall back to the module defaults. Returned
    keys match the tracker's constructor kwargs.

    The ``exact_threshold`` / ``name_threshold`` keys remain in the
    returned dict for backwards compatibility; under #204 they override
    the **work** category's WARNING tier only (preserving the pre-#204
    single-threshold semantics). The graduated CRITICAL tiers and the
    meta-category thresholds are not env-tunable in this PR -- they
    live as module constants grouped so a follow-up can make them
    configurable.
    """
    return {
        "window": _read_int_env("GOLDFIVE_TOOL_LOOP_WINDOW", DEFAULT_WINDOW),
        "exact_threshold": _read_int_env(
            "GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD", DEFAULT_EXACT_THRESHOLD
        ),
        "name_threshold": _read_int_env(
            "GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD", DEFAULT_NAME_THRESHOLD
        ),
        "alternating_threshold": _read_int_env(
            "GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD", DEFAULT_ALTERNATING_THRESHOLD
        ),
    }


def thresholds_from_config(config: ToolLoopConfig) -> dict[str, int]:
    """Adapt a :class:`~goldfive.config.ToolLoopConfig` to tracker kwargs.

    Mirrors :func:`load_thresholds_from_env` so callers can splat the
    result into :class:`ToolLoopTracker` without worrying about which
    source they read from. Added under goldfive#225 alongside the
    typed-config refactor.
    """
    return {
        "window": config.window,
        "exact_threshold": config.exact_threshold,
        "name_threshold": config.name_threshold,
        "alternating_threshold": config.alternating_threshold,
    }


def args_hash(args: Any) -> str:
    """Stable 8-char hex hash of a tool-call args payload.

    ``sort_keys=True`` so dict ordering doesn't perturb the hash.
    ``default=str`` falls back to ``str()`` for non-JSON values (dates,
    dataclasses, etc.) so we never raise from the hot path. An empty
    / non-mapping payload hashes deterministically so the mode 1 check
    still lights up on argument-free tool loops.
    """
    try:
        encoded = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = repr(args)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ToolLoopTracker:
    """Per-(invocation, agent) tool-call ring buffer + loop classifier.

    Instantiate one tracker per plugin instance (session-scoped). Call
    :meth:`observe_tool_call` on every tool dispatch; the returned list
    of :class:`~goldfive.types.DriftEvent` is what the plugin should
    route through the steerer. Call :meth:`on_task_progress` whenever a
    task on the same invocation transitions to a progress state so mode
    2's no-progress gate clears.

    Under goldfive#204 the classifier emits **graduated severity**: for
    a tool matching one of the threshold tiers, the tracker picks the
    highest tier (CRITICAL > WARNING > INFO) and emits one drift. The
    alternating-cycle mode (independent, A,B,A,B,A-shaped) still fires
    INFO only.

    The classifier never mutates its buffers after firing -- we do not
    dedupe drifts here. Upstream (the steerer's intervention ladder)
    already dedupes by ``(kind, task_id)`` occurrence count, and a
    repeating loop SHOULD keep emitting so the ladder escalates if the
    agent doesn't recover.

    ``exact_threshold`` / ``name_threshold`` kwargs are retained for
    backwards compatibility with callers that constructed the tracker
    with a single-threshold config (tests, env overrides). When
    provided, they **override the work category's WARNING tier** only;
    the graduated CRITICAL tiers and the meta category still use the
    module constants. This preserves the pre-#204 single-threshold
    semantics for work tools. Tests that exercise CRITICAL should
    leave these kwargs at their defaults.
    """

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
        name_threshold: int = DEFAULT_NAME_THRESHOLD,
        alternating_threshold: int = DEFAULT_ALTERNATING_THRESHOLD,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if exact_threshold <= 0:
            raise ValueError(f"exact_threshold must be positive, got {exact_threshold}")
        if name_threshold <= 0:
            raise ValueError(f"name_threshold must be positive, got {name_threshold}")
        if alternating_threshold <= 0:
            raise ValueError(f"alternating_threshold must be positive, got {alternating_threshold}")

        self.window = window
        self.exact_threshold = exact_threshold
        self.name_threshold = name_threshold
        self.alternating_threshold = alternating_threshold

        # Build the effective threshold tables. The work-WARNING tier
        # is overridden by the legacy kwargs so callers supplying the
        # old single-threshold config get pre-#204 behaviour there.
        # Meta thresholds and work-CRITICAL/INFO come from the module
        # constants unchanged.
        self._meta_thresholds = _META_THRESHOLDS
        work: dict[str, dict[str, int | None]] = {
            "info": dict(_WORK_THRESHOLDS["info"]),
            "warning": {"exact": exact_threshold, "name": name_threshold},
            "critical": dict(_WORK_THRESHOLDS["critical"]),
        }
        # Ensure INFO's exact threshold is never HIGHER than WARNING's
        # -- otherwise a caller passing exact_threshold=2 would make
        # INFO unreachable. Clamp defensively.
        info_exact = work["info"]["exact"]
        war_exact = work["warning"]["exact"]
        if info_exact is not None and war_exact is not None and info_exact > war_exact:
            work["info"]["exact"] = war_exact
        self._work_thresholds = work

        # Sanity: if the window is too small to hold the thresholds,
        # the detector can never fire for that mode. We log a debug
        # warning but still accept the config -- tests pin specific
        # (window, threshold) tuples and we don't want to reject them.
        if window < exact_threshold:
            log.debug(
                "tool-loop: window=%d < exact_threshold=%d -- work-WARNING exact cannot fire",
                window,
                exact_threshold,
            )
        if window < name_threshold:
            log.debug(
                "tool-loop: window=%d < name_threshold=%d -- work-WARNING name cannot fire",
                window,
                name_threshold,
            )
        if window < alternating_threshold:
            log.debug(
                "tool-loop: window=%d < alternating_threshold=%d -- alternating-cycle cannot fire",
                window,
                alternating_threshold,
            )

        # Keyed by (invocation_id, agent_name). Value: deque of
        # (tool_name, args_hash). Uses ``defaultdict`` with a bound
        # ``maxlen`` so slots auto-size.
        self._buffers: dict[tuple[str, str], deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    # -- Public API ---------------------------------------------------------

    def observe_tool_call(
        self,
        *,
        invocation_id: str,
        agent_name: str,
        tool_name: str,
        args: Any,
        task_id: str = "",
        observed_revision_index: int = 0,
        session_run_id: str = "",
    ) -> list[DriftEvent]:
        """Record one tool call; return any drift observations it triggers.

        ``task_id`` is stamped onto ``DriftEvent.current_task_id`` so
        the intervention ladder can scope occurrence counts per task
        (matches how other drift classifiers populate the field).
        Passing ``""`` is safe -- sinks just won't associate the drift
        with a specific task.

        ``observed_revision_index`` (goldfive#245) is stamped onto every
        produced :class:`DriftEvent` so the dispatch-time gate in
        :meth:`DefaultSteerer._handle_drift` can drop verdicts whose
        observed revision is older than the live plan's. Defaults to
        ``0`` (the "unset / pre-#245" sentinel) for legacy callers and
        unit tests that don't thread the session's plan revision.

        ``session_run_id`` (goldfive#420) is the scope key used to
        accumulate tool calls across re-invocations of the same agent
        within one run. When supplied (the ADK plugin threads
        ``session.run_id`` here), the bucket key is
        ``(session_run_id, agent_name)`` — so 11 re-invocations of
        ``debugger_agent`` calling ``find_presentation_files`` 2x each
        accumulate into one 22-entry window instead of 11 isolated
        2-entry windows. When empty (legacy callers, unit tests that
        haven't been updated, third-party adapters), the bucket key
        falls back to ``(invocation_id, agent_name)`` — preserving the
        pre-#420 isolation semantics. ``invocation_id`` is still
        recorded on the produced :class:`DriftEvent` so the dispatch-
        time helpers (active-invocation lookup, cancel) get the right
        target.
        """
        # goldfive#420: prefer the run-scoped key when the caller
        # supplied one. Falls back to invocation-scoped for legacy
        # callers (tests, third-party adapters) so the pre-#420
        # contract is preserved when no session id is plumbed.
        scope_key = session_run_id or invocation_id or ""
        key = (scope_key, agent_name or "")
        signature = (tool_name or "", args_hash(args))
        self._buffers[key].append(signature)
        return self._classify(
            key,
            invocation_id=invocation_id or "",
            current_task_id=task_id,
            observed_revision_index=observed_revision_index,
        )

    def on_task_progress(
        self,
        *,
        invocation_id: str,
        agent_name: str,
        session_run_id: str = "",
    ) -> None:
        """Reset the per-(scope, agent) buffer after task progress.

        Called whenever a task transitions to a progress state so
        mode 2's "no task progress in window" gate starts from a clean
        slate. A legitimate sequence like
        ``read_file read_file read_file`` that COMPLETES the task is
        not flagged because the next observation starts from an empty
        window.

        ``session_run_id`` (goldfive#420) must match the value the
        caller threads through :meth:`observe_tool_call` so the reset
        targets the same bucket. Empty / unset falls back to
        ``invocation_id`` for pre-#420 callers.

        .. note::

           This method unconditionally clears the per-key buffer.
           The **policy** of when to call it lives outside the
           tracker. In the ADK plugin it is gated on an acknowledged
           success response from the progress-reporting tool
           (goldfive#192) so that errored ``report_task_*`` retries
           accumulate in the ring buffer and trigger loop detection
           at the configured thresholds. Direct callers (unit tests,
           alternate adapters) may still invoke this method
           whenever they have out-of-band knowledge that genuine
           task progress occurred.
        """
        scope_key = session_run_id or invocation_id or ""
        key = (scope_key, agent_name or "")
        buf = self._buffers.get(key)
        if buf is not None:
            buf.clear()

    def clear(self) -> None:
        """Drop every per-(invocation, agent) buffer.

        Called from the plugin's ``clear_active_context`` so state
        doesn't leak across sessions when the plugin instance is
        reused.
        """
        self._buffers.clear()

    def buffer_size(
        self,
        *,
        invocation_id: str,
        agent_name: str,
        session_run_id: str = "",
    ) -> int:
        """Current entry count for the given key (test introspection helper).

        ``session_run_id`` (goldfive#420) selects the run-scoped key
        when present; falls back to ``invocation_id`` otherwise.
        """
        scope_key = session_run_id or invocation_id or ""
        key = (scope_key, agent_name or "")
        buf = self._buffers.get(key)
        return 0 if buf is None else len(buf)

    # -- Internal -----------------------------------------------------------

    def _thresholds_for_tool(self, tool_name: str) -> dict[str, dict[str, int | None]]:
        """Return the per-tier threshold table for ``tool_name``'s category."""
        if _classify_tool_category(tool_name) == "meta":
            return self._meta_thresholds
        return self._work_thresholds

    def _classify(
        self,
        key: tuple[str, str],
        *,
        current_task_id: str,
        observed_revision_index: int = 0,
        invocation_id: str = "",
    ) -> list[DriftEvent]:
        """Return zero or more drift observations for the current window.

        Emits **at most one** exact/name-based drift (the highest
        severity tier matched for the tool that hit threshold) plus,
        independently, **at most one** alternating-cycle INFO drift.
        Alternating is suppressed when an exact/name drift already
        fired -- same rationale as pre-#204 (the weaker signal would
        be noise on top of the stronger).

        ``invocation_id`` (goldfive#420) is stamped onto the emitted
        :class:`DriftEvent`'s ``raw.invocation_id`` so the dispatch-
        time cancel helper can target the actual in-flight invocation
        even when the bucket key scoped on ``session_run_id``. Falls
        back to the bucket scope when not supplied (legacy callers).
        """
        buf = list(self._buffers[key])
        observations: list[DriftEvent] = []
        scope_key, agent_name = key
        # Prefer the explicitly-provided invocation id (goldfive#420);
        # fall back to the bucket scope when callers didn't thread one
        # through (legacy in-tree callers, third-party adapters).
        emit_invocation_id = invocation_id or scope_key

        # Pre-compute per-signature and per-name counts in the window.
        exact_counts: dict[tuple[str, str], int] = {}
        name_counts: dict[str, int] = {}
        for sig in buf:
            exact_counts[sig] = exact_counts.get(sig, 0) + 1
            name_counts[sig[0]] = name_counts.get(sig[0], 0) + 1

        # --- Graduated exact/name classification ---------------------------
        #
        # For each distinct tool name in the window, find the highest
        # tier matched (CRITICAL > WARNING > INFO) and emit one drift
        # at that severity. "exact" axis is checked first; "name" axis
        # is checked only if "exact" didn't already match the same
        # tool at the same-or-higher tier (preserves the pre-#204
        # "exact preempts name on same tool" suppression).
        #
        # We iterate over unique tool names (not signatures) so a tool
        # with varied args contributes to "name" counts even when no
        # single signature hits the "exact" threshold. We pick the
        # name with the highest matched tier; if several names match,
        # we pick the highest severity first (ties broken by
        # dictionary iteration order, which is stable on CPython).
        best_drift: DriftEvent | None = None
        best_tier_index: int | None = None  # 0=critical, 1=warning, 2=info (lower = better)

        for name in name_counts:
            thresholds = self._thresholds_for_tool(name)
            name_total = name_counts[name]
            # Max exact-signature count for this tool in the window.
            exact_for_name = max(
                (c for sig, c in exact_counts.items() if sig[0] == name),
                default=0,
            )
            # Walk tiers from highest severity to lowest.
            for tier_index, (tier_key, severity) in enumerate(_SEVERITY_TIERS):
                tier = thresholds.get(tier_key) or {}
                exact_thr = tier.get("exact")
                name_thr = tier.get("name")
                hit_exact = exact_thr is not None and exact_for_name >= exact_thr
                hit_name = name_thr is not None and name_total >= name_thr
                if not (hit_exact or hit_name):
                    continue

                # Found the highest tier for this tool. Build a drift.
                # Prefer the "exact" detail when both axes are
                # satisfied -- the more specific signal.
                mode = "exact" if hit_exact else "name"
                if mode == "exact":
                    sig = next(
                        sig
                        for sig, c in exact_counts.items()
                        if sig[0] == name and c == exact_for_name
                    )
                    detail = f"tool_loop_exact: {name} x {exact_for_name} in last {len(buf)} calls"
                    raw: dict[str, Any] = {
                        "mode": "exact",
                        "tool_name": name,
                        "args_hash": sig[1],
                        "count": exact_for_name,
                        "window_len": len(buf),
                        "invocation_id": emit_invocation_id,
                        "category": _classify_tool_category(name),
                        "tier": tier_key,
                    }
                else:
                    detail = (
                        f"tool_loop_name: {name} x {name_total} in last "
                        f"{len(buf)} calls (no task progress)"
                    )
                    raw = {
                        "mode": "name",
                        "tool_name": name,
                        "count": name_total,
                        "window_len": len(buf),
                        "invocation_id": emit_invocation_id,
                        "category": _classify_tool_category(name),
                        "tier": tier_key,
                    }

                # trigger_input: summarise the window the detector
                # matched so sinks can render "what were the tool calls
                # goldfive saw?" without re-fetching the session buffer.
                tool_names_seen = [sig[0] for sig in buf]
                trigger_input = (
                    f"tool_loop window ({len(buf)} calls, tool={name!r}): "
                    + " -> ".join(tool_names_seen[-16:])
                )
                candidate = DriftEvent(
                    kind=DriftKind.LOOPING_REASONING,
                    severity=severity,
                    detail=detail,
                    current_task_id=current_task_id,
                    current_agent_id=agent_name,
                    raw=raw,
                    trigger_input=trigger_input,
                    observed_revision_index=observed_revision_index,
                )
                # Keep the highest-severity candidate seen across all
                # tools. tier_index is 0 for CRITICAL, so "lower is
                # better" for our "best" accumulator.
                if best_tier_index is None or tier_index < best_tier_index:
                    best_drift = candidate
                    best_tier_index = tier_index
                break  # stop at the highest tier for this tool

        if best_drift is not None:
            observations.append(best_drift)

        # --- Mode 3: alternating A,B,A,B,A pattern -------------------------
        if len(buf) >= self.alternating_threshold:
            tail = buf[-self.alternating_threshold :]
            names = [sig[0] for sig in tail]
            if (
                len(set(names)) == 2
                and names[0] != names[1]
                and all(names[i] == names[i % 2] for i in range(len(names)))
            ):
                # Don't double-fire if an exact/name drift already
                # flagged this window -- the alternating signal is
                # weaker (INFO) and would be noise on top.
                if best_drift is None:
                    a_name, b_name = names[0], names[1]
                    observations.append(
                        DriftEvent(
                            kind=DriftKind.LOOPING_REASONING,
                            severity=DriftSeverity.INFO,
                            detail=(
                                f"tool_loop_alternating: "
                                f"{a_name} <-> {b_name} "
                                f"in last {len(tail)} calls"
                            ),
                            current_task_id=current_task_id,
                            current_agent_id=agent_name,
                            raw={
                                "mode": "alternating",
                                "tools": [a_name, b_name],
                                "window_len": len(tail),
                                "invocation_id": emit_invocation_id,
                            },
                            trigger_input=(
                                f"tool_loop alternating ({len(tail)} calls): "
                                + " -> ".join(names)
                            ),
                            observed_revision_index=observed_revision_index,
                        )
                    )
        return observations
