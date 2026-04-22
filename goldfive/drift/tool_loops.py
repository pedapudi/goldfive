"""Tool-call loop drift detector (goldfive#181).

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

The existing :class:`goldfive.adapters._tool_loop_guard.ToolLoopGuard`
IS args-aware but is scoped to **reporting tools** (tools dispatched
via :func:`~goldfive.adapters._tool_invocation.invoke_tool`). Tool-loops
on arbitrary agent-visible tools (AgentTool delegations, MCP tools,
custom adapter-native tools) are not covered there.

This module is the complementary detector: it observes **every** tool
call the ADK plugin sees (reporting or otherwise), keyed per
``(invocation_id, agent_name)``, and fires a
``LOOPING_REASONING``-kind :class:`~goldfive.types.DriftEvent` when
one of three patterns is detected:

1. **Exact loop** -- same ``(tool_name, args_hash)`` repeats >= the
   configured ``exact_threshold`` in the last ``window`` calls.
   WARNING severity.
2. **Name loop** -- same ``tool_name`` (any args) repeats >= the
   configured ``name_threshold`` in the last ``window`` calls AND no
   task-state transition has been recorded in the window. WARNING
   severity.
3. **Alternating cycle** -- A,B,A,B,A pattern in the last
   ``alternating_threshold`` calls. INFO severity.

Mode 2's "no task progress" gate is satisfied by calling
:meth:`ToolLoopTracker.on_task_progress` whenever a task transitions
RUNNING -> COMPLETED (or any other meaningful progress signal). That
clears the per-agent buffer so a legitimate burst of the same tool
(e.g. a scripted lint + build + test pipeline) is not flagged when
each call completes its step.

Isolation: trackers are keyed on ``(invocation_id, agent_name)`` so
each ADK invocation gets its own window, and parallel sub-agents
within the same invocation (via AgentTool) are also isolated. State
is intentionally held in a single :class:`ToolLoopTracker` instance
on the plugin -- we don't need per-session persistence, the whole
window is ephemeral to one run.

Configuration overrides via environment variables. Defaults chosen
to balance signal vs noise (see below).

* ``GOLDFIVE_TOOL_LOOP_WINDOW`` -- ring-buffer size (default ``7``).
  Picked so mode 2's 5-in-7 name-repeat check has two "any-other-tool"
  slots for variance before firing; a stricter ``5`` window made the
  detector fire on legitimate pipelines in early manual testing.
* ``GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD`` -- minimum identical
  signatures to fire mode 1 (default ``3``). Three identical calls in
  a row is the smallest number that cannot plausibly be a benign
  retry.
* ``GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD`` -- minimum same-name calls to
  fire mode 2 (default ``5``). Five same-name calls in a 7-call window
  means at most two distinct tools were invoked -- strong signal the
  agent is stuck grinding on one capability.
* ``GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD`` -- pattern length for
  mode 3 (default ``5``). A,B,A,B,A is the smallest alternating pattern
  that isn't just "ping once, then ping again" noise.

The detector is intentionally deterministic (no embeddings, no LLM
calls). O(1) per tool call modulo the `window` length.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict, deque
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity

__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_EXACT_THRESHOLD",
    "DEFAULT_NAME_THRESHOLD",
    "DEFAULT_ALTERNATING_THRESHOLD",
    "ToolLoopTracker",
    "args_hash",
    "load_thresholds_from_env",
]


log = logging.getLogger("goldfive.drift.tool_loops")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Ring-buffer size: how many recent tool calls to retain per
#: ``(invocation_id, agent_name)`` key. ``7`` is the default so the
#: 5-in-7 name-repeat check has room for two "other-tool" calls without
#: firing.
DEFAULT_WINDOW: int = 7

#: Mode 1 threshold: exact ``(name, args_hash)`` repeats in the window.
DEFAULT_EXACT_THRESHOLD: int = 3

#: Mode 2 threshold: same ``tool_name`` repeats in the window.
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

    The classifier never mutates its buffers after firing -- we do not
    dedupe drifts here. Upstream (the steerer's intervention ladder)
    already dedupes by ``(kind, task_id)`` occurrence count, and a
    repeating loop SHOULD keep emitting so the ladder escalates if the
    agent doesn't recover.
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
        # Sanity: if the window is too small to hold the thresholds,
        # the detector can never fire for that mode. We log a debug
        # warning but still accept the config -- tests pin specific
        # (window, threshold) tuples and we don't want to reject them.
        if window < exact_threshold:
            log.debug(
                "tool-loop: window=%d < exact_threshold=%d -- mode 1 cannot fire",
                window,
                exact_threshold,
            )
        if window < name_threshold:
            log.debug(
                "tool-loop: window=%d < name_threshold=%d -- mode 2 cannot fire",
                window,
                name_threshold,
            )
        if window < alternating_threshold:
            log.debug(
                "tool-loop: window=%d < alternating_threshold=%d -- mode 3 cannot fire",
                window,
                alternating_threshold,
            )

        self.window = window
        self.exact_threshold = exact_threshold
        self.name_threshold = name_threshold
        self.alternating_threshold = alternating_threshold
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
    ) -> list[DriftEvent]:
        """Record one tool call; return any drift observations it triggers.

        ``task_id`` is stamped onto ``DriftEvent.current_task_id`` so
        the intervention ladder can scope occurrence counts per task
        (matches how other drift classifiers populate the field).
        Passing ``""`` is safe -- sinks just won't associate the drift
        with a specific task.
        """
        key = (invocation_id or "", agent_name or "")
        signature = (tool_name or "", args_hash(args))
        self._buffers[key].append(signature)
        return self._classify(key, current_task_id=task_id)

    def on_task_progress(
        self,
        *,
        invocation_id: str,
        agent_name: str,
    ) -> None:
        """Reset the per-(invocation, agent) buffer after task progress.

        Called whenever a task transitions to a progress state so
        mode 2's "no task progress in window" gate starts from a clean
        slate. A legitimate sequence like
        ``read_file read_file read_file`` that COMPLETES the task is
        not flagged because the next observation starts from an empty
        window.
        """
        key = (invocation_id or "", agent_name or "")
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

    def buffer_size(self, *, invocation_id: str, agent_name: str) -> int:
        """Current entry count for the given key (test introspection helper)."""
        key = (invocation_id or "", agent_name or "")
        buf = self._buffers.get(key)
        return 0 if buf is None else len(buf)

    # -- Internal -----------------------------------------------------------

    def _classify(
        self,
        key: tuple[str, str],
        *,
        current_task_id: str,
    ) -> list[DriftEvent]:
        """Return zero or more drift observations for the current window."""
        buf = list(self._buffers[key])
        observations: list[DriftEvent] = []
        invocation_id, agent_name = key

        # --- Mode 1: exact-signature repeat --------------------------------
        exact_counts: dict[tuple[str, str], int] = {}
        for sig in buf:
            exact_counts[sig] = exact_counts.get(sig, 0) + 1
        fired_exact = False
        for sig, count in exact_counts.items():
            if count >= self.exact_threshold:
                tool_name, sig_hash = sig
                observations.append(
                    DriftEvent(
                        kind=DriftKind.LOOPING_REASONING,
                        severity=DriftSeverity.WARNING,
                        detail=(f"tool_loop_exact: {tool_name} x {count} in last {len(buf)} calls"),
                        current_task_id=current_task_id,
                        current_agent_id=agent_name,
                        raw={
                            "mode": "exact",
                            "tool_name": tool_name,
                            "args_hash": sig_hash,
                            "count": count,
                            "window_len": len(buf),
                            "invocation_id": invocation_id,
                        },
                    )
                )
                fired_exact = True
                break  # at most one exact drift per classify call

        # --- Mode 2: same-name repeat (only if no task progress) -----------
        # ``on_task_progress`` clears the window, so if we're here with
        # a full-enough buffer, no progress has been recorded since the
        # window started filling. No separate token needed.
        name_counts: dict[str, int] = {}
        for sig in buf:
            name_counts[sig[0]] = name_counts.get(sig[0], 0) + 1
        fired_name = False
        for name, count in name_counts.items():
            if count < self.name_threshold:
                continue
            # Suppress mode 2 when mode 1 already fired on the same
            # tool -- it's the same signal, more specific. Mode 2
            # still fires independently when mode 1 didn't (args
            # varied across the burst).
            if fired_exact and any(
                sig[0] == name and c >= self.exact_threshold for sig, c in exact_counts.items()
            ):
                continue
            observations.append(
                DriftEvent(
                    kind=DriftKind.LOOPING_REASONING,
                    severity=DriftSeverity.WARNING,
                    detail=(
                        f"tool_loop_name: {name} x {count} in last "
                        f"{len(buf)} calls (no task progress)"
                    ),
                    current_task_id=current_task_id,
                    current_agent_id=agent_name,
                    raw={
                        "mode": "name",
                        "tool_name": name,
                        "count": count,
                        "window_len": len(buf),
                        "invocation_id": invocation_id,
                    },
                )
            )
            fired_name = True
            break

        # --- Mode 3: alternating A,B,A,B,A pattern -------------------------
        if len(buf) >= self.alternating_threshold:
            tail = buf[-self.alternating_threshold :]
            names = [sig[0] for sig in tail]
            if (
                len(set(names)) == 2
                and names[0] != names[1]
                and all(names[i] == names[i % 2] for i in range(len(names)))
            ):
                # Don't double-fire if mode 1 or 2 already flagged
                # this same pair -- the alternating signal is weaker
                # (INFO) and would be noise on top of a WARNING.
                if not (fired_exact or fired_name):
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
                                "invocation_id": invocation_id,
                            },
                        )
                    )
        return observations
