"""Drift-detector registry + shared LLM-judge boilerplate.

Wave A piece 2 of the goldfive modularisation plan. Centralises the
*genuinely-shared* boilerplate across the five drift detectors so each
detector contributes only its classifier-specific logic.

What lives here (truly shared across detectors)
-----------------------------------------------

* :class:`DetectorConfig` — per-detector knobs (output-token cap,
  input/output truncation limits, whether the detector dispatches an LLM
  call). Pinned per-detector via :func:`register`.
* :func:`parse_json_response` — liberal JSON extractor that tolerates
  markdown fences / prose preludes. Used by every LLM-as-a-judge
  detector (currently :mod:`reasoning_judge` and :mod:`goals`).
* :func:`truncate_for_observability` — bounded-length text truncation
  with a uniform ``" … [truncated]"`` suffix, shared by every detector
  that stamps a long string onto an event payload.
* :func:`format_goals_block` — render
  :class:`~goldfive.types.Goal` sequences as a human-readable block.
  Identical between the goal-drift and reasoning-judge prompts.
* :func:`register` / :func:`classify` — the registry API. Each detector
  registers a ``(DriftKind, classifier_fn, DetectorConfig)`` triple;
  callers can dispatch by kind via :func:`classify` without importing
  the detector module directly.

What stays in the detector modules (deliberately not centralised)
-----------------------------------------------------------------

* **Prompt construction** — every detector has its own template + its
  own helper functions (``_format_task``, ``_format_activity``,
  ``_format_task_lineage``, ``_format_tool_observations``,
  ``format_plan_tasks_summary``, ``format_available_agents_block``,
  …). Moving these here would be churn for no real reuse.
* **Span ``decision_summary`` / ``output_preview`` strings** — each
  detector produces a verdict-shaped span tail (``"judged trajectory:
  on-track"`` vs ``"judged X's reasoning on Y: justified_deviation
  (tool_error)"``). Centralising those would obscure the very tail
  strings operators grep for in production logs.
* **Sink emission** — only :func:`classify_reasoning_drift` emits a
  ``ReasoningJudgeInvoked`` event; centralising the emitter would force
  the abstraction onto detectors that don't have an observability
  channel.
* **Post-LLM re-read / staleness gates** — only the goal-drift judge
  re-reads ``session.plan`` after the LLM round-trip. Detector-specific.
* **DriftEvent construction** — each kind has its own ``detail`` shape
  and severity logic.

Honest scope note (PR delta)
----------------------------

The Wave-A brief anticipated ~500 LOC reduction across all five
detectors. In practice only two of the five (``reasoning_judge`` and
``goals``) actually wrap an LLM call + parse JSON; the other three
(``capability_check``, ``tool_loops``, ``reasoning``) are structural
or embedding-based and have no LLM/JSON boilerplate to share. The
real LOC reduction is modest — the value of this module is the single
source of truth for the JSON extractor, the goals renderer, and the
truncation helper, plus a uniform registry-dispatch API for callers
that want to look up detectors by ``DriftKind``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from goldfive.types import DriftKind, Goal

log = logging.getLogger(__name__)


__all__ = [
    "DetectorConfig",
    "TRUNCATE_SUFFIX",
    "JSON_OBJECT_RE",
    "classify",
    "format_goals_block",
    "get_config",
    "list_registered",
    "parse_json_response",
    "register",
    "truncate_for_observability",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------


#: Uniform truncation marker used by every detector's observability-bound
#: string. Lifted here from the per-detector copies so a future change to
#: the suffix only happens in one place.
TRUNCATE_SUFFIX: str = " … [truncated]"


#: Liberal JSON-object extractor. Real LLMs emit markdown code fences
#: and prose even with strong "reply JSON only" instructions; this
#: matches the *first* ``{...}`` span in the response, leaving the
#: per-detector parser to validate the dict shape. Used by every
#: LLM-as-a-judge detector.
JSON_OBJECT_RE: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# DetectorConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    """Per-detector boilerplate knobs.

    Each detector registers a config via :func:`register`. Callers can
    look up the config via :func:`get_config` to read the per-detector
    caps without having to know which detector module owns the kind.

    Attributes
    ----------
    uses_llm:
        ``True`` when the detector dispatches an LLM call. The two
        currently-registered LLM detectors are ``OFF_TOPIC`` (the
        reasoning judge) and ``GOAL_DRIFT``. Purely structural detectors
        (``CAPABILITY_MISMATCH``, ``TOOL_ERROR``, ``CONFABULATION_RISK``,
        the loop trackers, the embedding-based reasoning detectors) set
        this to ``False``.
    max_input_chars:
        Hard cap on the *prompt* / *trigger_input* observability payload
        the detector emits. Reasoning judge: 4096. Goal drift: 2048.
        Pure-structural detectors with no observability payload set
        this to ``0`` and the helper is a no-op.
    max_output_tokens:
        Per-callsite ``max_output_tokens`` budget bound around
        ``await call_llm(...)`` via
        :func:`goldfive._llm.call_llm_budget`. Both LLM detectors
        currently use 16384 (Qwen 3.5 thinking-mode headroom — see
        :data:`goldfive._llm.DEFAULT_MAX_OUTPUT_TOKENS` for the
        rationale).
    disable_thinking:
        When ``True``, the LLM dispatch enters
        :func:`goldfive._llm.call_llm_thinking_disabled`. Both LLM
        detectors set this — they ask small JSON-shaped meta-cognition
        questions and don't need ``<think>`` reasoning preludes.
    timeout_seconds:
        Soft wall-clock budget the detector advertises. Goldfive's
        ``call_llm`` callables already have a wall-clock backstop
        (``DEFAULT_LLM_CALL_TIMEOUT_MS`` in the ADK plugin); this is a
        per-detector advisory limit that downstream observability can
        surface but is not enforced here.
    """

    uses_llm: bool = False
    max_input_chars: int = 0
    max_output_tokens: int = 0
    disable_thinking: bool = False
    timeout_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def truncate_for_observability(text: Any, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars, appending :data:`TRUNCATE_SUFFIX` when cut.

    Returns the empty string when ``text`` is not a ``str`` (consistent
    with the historical per-detector helpers in
    :mod:`reasoning_judge` and the goal-drift trigger-input helper).
    A ``limit`` of ``0`` or below is treated as "no truncation"
    (returns the input unchanged when it's a string).
    """
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + TRUNCATE_SUFFIX


def parse_json_response(raw: Any) -> dict[str, Any] | None:
    """Extract the first JSON object from ``raw`` or return ``None``.

    The shared LLM-judge response parser. Matches the byte-for-byte
    behaviour of the previous per-detector copies in
    :mod:`goldfive.drift.goals` and
    :mod:`goldfive.drift.reasoning_judge`:

    1. If ``raw`` is not a non-empty string, return ``None``.
    2. Try :func:`json.loads` on the stripped text.
    3. On failure, search for the first ``{...}`` span via
       :data:`JSON_OBJECT_RE` and try :func:`json.loads` on that.
    4. Return the decoded dict, or ``None`` if either the response
       wasn't valid JSON or the top-level value isn't a dict.

    The "return ``None`` on any failure" contract is deliberate — both
    LLM judges quiet-fail on malformed responses to avoid false-
    positive drifts (goldfive#143 / #226 / #244).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    stripped = raw.strip()
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        match = JSON_OBJECT_RE.search(stripped)
        if match is None:
            return None
        try:
            decoded = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def format_goals_block(goals: Sequence[Goal] | Iterable[Any] | None) -> str:
    """Render ``goals`` as a numbered ``[id] summary`` block.

    Shared by every LLM-judge prompt that needs to surface
    ``session.goals``. Byte-for-byte identical to the per-detector
    helpers it replaces in :mod:`goldfive.drift.goals` and
    :mod:`goldfive.drift.reasoning_judge` (the two previous copies
    were literally identical).

    Empty / ``None`` goals render as ``"(no goals recorded)"`` so the
    prompt template can interpolate the block without conditional
    formatting.
    """
    if not goals:
        return "(no goals recorded)"
    lines: list[str] = []
    for i, g in enumerate(goals, start=1):
        gid = str(getattr(g, "id", "") or "")
        summary = str(getattr(g, "summary", "") or "")
        if not summary and isinstance(g, str):
            summary = g
        prefix = f"{i}."
        if gid:
            lines.append(f"{prefix} [{gid}] {summary}")
        else:
            lines.append(f"{prefix} {summary}")
    return "\n".join(lines) if lines else "(no goals recorded)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Type alias for a registered classifier. Each detector exposes its
#: own positional/keyword shape; the registry stores them as ``**kwargs``
#: callables so :func:`classify` can dispatch without knowing the per-
#: detector signature.
ClassifierFn = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class _Registration:
    """Internal record stored by :func:`register`."""

    classifier: ClassifierFn
    config: DetectorConfig
    is_async: bool


_REGISTRY: dict[DriftKind, _Registration] = {}


def register(
    kind: DriftKind,
    classifier_fn: ClassifierFn,
    config: DetectorConfig,
    *,
    is_async: bool = False,
) -> None:
    """Register a classifier for ``kind`` with the given ``config``.

    Re-registration overwrites a prior entry (and logs a debug line)
    so test fixtures that swap classifiers in and out don't accumulate
    stale registrations.

    Parameters
    ----------
    kind:
        The :class:`~goldfive.types.DriftKind` the classifier emits.
        Note that one detector module may register multiple kinds
        (e.g. the reasoning judge emits both ``OFF_TOPIC`` and
        ``JUSTIFIED_DEVIATION``); the per-kind classifier in that case
        points to the same underlying function and the dispatcher just
        forwards by kind.
    classifier_fn:
        The detector entry point. May be sync or async — set
        ``is_async`` accordingly so :func:`classify` can ``await`` it
        when appropriate. The classifier owns its own kwargs shape; the
        registry forwards ``**observation`` verbatim.
    config:
        The :class:`DetectorConfig` pinning per-detector knobs.
    is_async:
        ``True`` when ``classifier_fn`` is a coroutine function. The
        registry does not introspect; the caller passes the flag
        explicitly so wrappers around an async function (e.g. a
        ``functools.partial``) register correctly.
    """
    prior = _REGISTRY.get(kind)
    _REGISTRY[kind] = _Registration(
        classifier=classifier_fn, config=config, is_async=is_async
    )
    if prior is not None:
        log.debug(
            "drift.registry.register: overwrote prior registration for %r", kind
        )


def get_config(kind: DriftKind) -> DetectorConfig | None:
    """Return the :class:`DetectorConfig` registered for ``kind``, or ``None``.

    Useful for callers that want to read the per-detector caps (e.g.
    ``max_input_chars`` for an observability payload) without invoking
    the classifier.
    """
    reg = _REGISTRY.get(kind)
    return reg.config if reg is not None else None


def list_registered() -> tuple[DriftKind, ...]:
    """Return the currently-registered :class:`DriftKind` set.

    Order is insertion-order (Python ``dict`` semantics). Primarily for
    tests and diagnostics.
    """
    return tuple(_REGISTRY.keys())


def classify(*, kind: DriftKind, **observation: Any) -> Any:
    """Dispatch to the classifier registered for ``kind``.

    Forwards ``**observation`` verbatim to the registered classifier.
    Returns the classifier's return value as-is — typically a
    :class:`~goldfive.types.DriftEvent` or ``None``, but the reasoning
    judge returns a :class:`~goldfive.drift.reasoning_judge.ReasoningJudgeVerdict`,
    which is wider than ``DriftEvent | None``. Callers that prefer the
    narrow shape can ignore the extra fields or call the detector
    entry point directly.

    Raises :class:`KeyError` when no detector is registered for ``kind``
    so a typo at the call site is loud rather than silent.

    When the registered classifier is async (``is_async=True`` at
    registration time), the function returns the awaitable from the
    classifier — callers must ``await`` the result. Sync classifiers
    return their value directly. Mixing in one call site is rare
    enough that auto-detection isn't worth the runtime cost; pass
    ``is_async`` explicitly at registration.
    """
    reg = _REGISTRY.get(kind)
    if reg is None:
        raise KeyError(
            f"no drift detector registered for kind={kind!r}; "
            f"registered kinds: {list_registered()!r}"
        )
    return reg.classifier(**observation)


# ---------------------------------------------------------------------------
# Auto-registration entry point
# ---------------------------------------------------------------------------
#
# Each detector module is responsible for calling :func:`register` at
# import time. We expose a thin :func:`_ensure_registered` entry point
# that imports every detector module once so the registry is fully
# populated when callers introspect via :func:`list_registered`. Lazy
# by design: the import is cheap (the detector modules are already
# imported by the steerer in any realistic runtime), and tests that
# only exercise a subset of detectors don't need to pull in
# everything.


_AUTO_REGISTERED: bool = False


def _ensure_registered() -> None:
    """Import every detector module so its registration runs.

    Idempotent — safe to call multiple times. The drift module's
    public ``__init__.py`` calls this on first non-attribute access
    via the lazy ``__getattr__`` hook.
    """
    global _AUTO_REGISTERED
    if _AUTO_REGISTERED:
        return
    # Local imports (avoid top-of-module import cycles via drift/__init__).
    from goldfive.drift import (  # noqa: F401 — side-effect import for registration
        capability_check,
        goals,
        reasoning_judge,
        tool_loops,
    )

    _AUTO_REGISTERED = True
