"""Structural capability-mismatch detector for ``delegation_observed`` (goldfive#253).

Fires :class:`~goldfive.types.DriftKind.CAPABILITY_MISMATCH` when the
agent the coordinator delegated to *structurally* cannot perform the
bound task. Three narrow rules; false positives are worse than false
negatives at this layer because every fire cancels the in-flight
invocation and triggers a planner refine.

Replaces the planner-LLM ``PLAN_DIVERGENCE`` comparison for the
"wrong-assignee" case: instead of comparing the planner's *predicted*
assignee against the runtime delegation (and getting it wrong when the
planner LLM hallucinated the assignee), we ground the comparison in
*actual* tool capability surfaced by the ADK agent object.

Detection rules (intentionally surgical):

* **Rule A — coordinator-style leaf-assignment. SOFT-RETIRED
  (AGENCY-PRESERVATION.md PR 3): OFF by default.** If every tool on the
  invoked agent is an ``AgentTool`` instance (i.e. its only capability
  is to delegate further) AND the bound task reads as a leaf authoring
  task ("draft", "write", "review", "research", "patch", "locate" …
  rather than "coordinate", "delegate", "orchestrate"), the agent
  cannot actually do the work. The leaf-task read is stem/keyword NL
  classification — the #166/#167 anti-pattern — and was the source of
  the e2e 2d27ff4a refine storm, so Rule A no longer runs unless the
  operator sets ``GOLDFIVE_CAPABILITY_RULE_A=1``. Even when re-enabled
  it only OBSERVEs: the CAPABILITY_MISMATCH ladder demotes the CRITICAL
  cells (PR 3). Fires CRITICAL (for the observability signal). Hard
  deletion follows in PR 13.

* **Rule B — required-tools advisory.** If
  :attr:`~goldfive.types.Task.required_tools` is non-empty and the
  invoked agent's tool names do not cover every required name, fire
  WARNING. Skipped entirely when the advisory is empty (legacy plans
  and planners that don't populate it are a no-op). Rule B is the one
  capability rule kept as a steering trigger (user-declared
  ``required_tools`` is genuine intent, not a forecast); it fires
  WARNING rather than CRITICAL ("WARNING-max", PR 3) so the ladder
  refines but never escalates to cancel/pause.

* **Rule C — out-of-DAG-order delegation (goldfive#268). SOFT-RETIRED
  (goldfive#423 / AGENCY-PRESERVATION.md PR 2): OFF by default.** When
  the invoked agent's *role stem* (the head of its name, with
  role-suffix tokens like ``agent``/``worker`` trimmed) is absent from
  the bound task's title+description AND present in some OTHER pending
  task, the pin has bound the delegation to a structurally-wrong task —
  typically because the coordinator dispatched a downstream agent
  before its DAG predecessors completed and the pin had only one
  eligible candidate to choose from. Fires CRITICAL. Requires the
  caller to pass ``all_pending_tasks`` so the cross-task lookup can
  see non-DAG-ready tasks too; otherwise the rule is silent.

  With descriptive growth at pin time enabled
  (``SteeringConfig.descriptive_growth_enabled`` — default OFF pending
  the AGENCY-PRESERVATION.md §6.4 13b bench gate; #497 reverted the
  interim default-ON) the "no right task existed" cause Rule C papered
  over is fixed at the source — the plan grows a ``discovered=True``
  task carrying the agent's role stem, so Rule C's trigger shape no
  longer occurs (design doc §7). The rule is disabled unless the
  operator explicitly re-enables it via ``GOLDFIVE_CAPABILITY_RULE_C=1``
  (per design doc §7.1 soft retirement; hard deletion follows in
  AGENCY-PRESERVATION.md PR 13). Note the resulting pure-defaults
  posture (growth OFF *and* Rule C OFF: neither the rescue nor the
  detection runs) is a recorded 13b decision — see
  AGENCY-PRESERVATION.md §6.6.

All three rules return :class:`~goldfive.types.DriftEvent` carrying the
agent name + bound task id + the structural gap, suitable for the
goldfive intervention ladder. The detector is framework-neutral: it
takes ADK ``Tool`` objects but only reads their attributes (``.agent``
for AgentTool detection, ``.name`` for the required-tools cover).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Task

log = logging.getLogger(__name__)


#: Env flag that re-enables the soft-retired Rule C (goldfive#423 /
#: AGENCY-PRESERVATION.md PR 2; retirement plan in design doc §7.1).
#: Named so re-enabling is an explicit operator act:
#: ``GOLDFIVE_CAPABILITY_RULE_C=1``. Read per call (not cached at
#: import) so tests and operators can flip it without re-importing.
_RULE_C_ENV_VAR = "GOLDFIVE_CAPABILITY_RULE_C"

#: Env flag that re-enables the soft-retired Rule A (goldfive#253 /
#: AGENCY-PRESERVATION.md PR 3). Rule A is default OFF for the same
#: reason Rule C is: it is stem/keyword NL classification of whether a
#: task "reads as a leaf authoring task" (the ``_looks_like_delegation_
#: task`` marker scan + the leaf-title heuristic) — the exact
#: #166/#167 anti-pattern the project retired twice. Its real-world
#: failure mode is the cherry-tree refine storm (a coordinator
#: delegating to a worker 20+ times, each delegation firing
#: CAPABILITY_MISMATCH → refine; e2e session 2d27ff4a). Read per call
#: so tests/operators can flip it without re-importing. When re-enabled
#: it still only OBSERVEs (the CAPABILITY_MISMATCH ladder CRITICAL cells
#: are OBSERVE per PR 3) — the flag restores the observability *signal*
#: for debugging, not steering. Hard deletion follows in PR 13.
_RULE_A_ENV_VAR = "GOLDFIVE_CAPABILITY_RULE_A"

#: Truthy spellings accepted for :data:`_RULE_C_ENV_VAR` /
#: :data:`_RULE_A_ENV_VAR` — same vocabulary as
#: :func:`goldfive.config._read_bool_env` so the env surface stays
#: consistent across the project. (No falsy/typo handling needed:
#: anything that is not an explicit truthy spelling leaves the rule
#: retired, which is the safe default.)
_RULE_C_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})


def _rule_c_enabled() -> bool:
    """Return True iff the operator explicitly re-enabled Rule C.

    Rule C is soft-retired (default OFF) because descriptive growth at
    pin time — when enabled; it is itself default-OFF pending 13b
    (#497) — removes the trigger shape Rule C existed to catch. See the
    module docstring, ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §7, and
    AGENCY-PRESERVATION.md §6.6 for the pure-defaults posture.
    """
    raw = os.environ.get(_RULE_C_ENV_VAR, "").strip().lower()
    return raw in _RULE_C_TRUTHY


def _rule_a_enabled() -> bool:
    """Return True iff the operator explicitly re-enabled Rule A.

    Rule A is soft-retired (default OFF) per AGENCY-PRESERVATION.md
    PR 3: its leaf-task heuristic is NL classification (#166/#167), and
    even when re-enabled it routes to OBSERVE (the CAPABILITY_MISMATCH
    ladder demotes the CRITICAL cells). Same env vocabulary and
    read-per-call discipline as :func:`_rule_c_enabled`.
    """
    raw = os.environ.get(_RULE_A_ENV_VAR, "").strip().lower()
    return raw in _RULE_C_TRUTHY


__all__ = [
    "AGENT_NAME_ROLE_SUFFIXES",
    "DELEGATION_VERB_MARKERS",
    "agent_name_stems",
    "detect_capability_mismatch",
    "is_agent_tool",
    "stem_token_match",
    "tokenize_for_matching",
]


#: Trailing role-suffix tokens stripped from an invoked-agent name during
#: stem extraction. Lifted out of :mod:`goldfive.adapters._adk_plugin`
#: (the goldfive#265 tier-2 disambiguator) so both call sites — the
#: delegation-pin selector AND the goldfive#268 Rule C detector — share
#: one definition.
#:
#: Conservative by design: only role-marker tokens that almost never
#: carry topic meaning. Keep small — adding common verbs/nouns here
#: would suppress legitimate matches (e.g. dropping ``researcher`` from
#: ``researcher_agent`` leaves the empty stem and nothing matches).
AGENT_NAME_ROLE_SUFFIXES: frozenset[str] = frozenset(
    {"agent", "worker", "assistant", "bot", "tool"}
)


def tokenize_for_matching(text: Any) -> set[str]:
    """Return the set of lowercase alphanumeric tokens of length ≥4.

    Shared with :mod:`goldfive.adapters._adk_plugin` (Tier-3 args
    scorer + Tier-2 stem match). The ≥4 threshold filters out noisy
    short-word matches ("in", "of", "the") that would otherwise
    saturate every comparison. No regex — goldfive#166 / #167.
    """
    if not isinstance(text, str):
        text = str(text or "")
    tokens: set[str] = set()
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tok = "".join(buf)
                if len(tok) >= 4:
                    tokens.add(tok)
                buf.clear()
    if buf:
        tok = "".join(buf)
        if len(tok) >= 4:
            tokens.add(tok)
    return tokens


def agent_name_stems(agent_name: str) -> tuple[str, ...]:
    """Return ordered lowercase stem tokens derived from an ADK agent name.

    Splits on underscore/hyphen/space, lowercases, trims trailing
    role-suffix tokens (``agent``, ``worker``, …), and keeps tokens of
    length ≥4. Same surface :mod:`goldfive.adapters._adk_plugin` uses
    for its goldfive#265 tier-2 disambiguator; lifted here so the #268
    Rule C detector can share one source of truth.

    Examples
    --------
    >>> agent_name_stems("reviewer_agent")
    ('reviewer',)
    >>> agent_name_stems("web_developer_agent")
    ('developer',)
    >>> agent_name_stems("helper_agent")
    ('helper',)
    >>> agent_name_stems("agent")
    ()

    No regex (goldfive#166 / #167). Pure str ops.
    """
    if not isinstance(agent_name, str) or not agent_name:
        return ()
    norm = agent_name.replace("_", " ").replace("-", " ").lower()
    raw_tokens = [tok for tok in norm.split() if tok]
    while raw_tokens and raw_tokens[-1] in AGENT_NAME_ROLE_SUFFIXES:
        raw_tokens.pop()
    return tuple(tok for tok in raw_tokens if len(tok) >= 4)


def stem_token_match(stem: str, token: str) -> bool:
    """Return True when ``stem`` and ``token`` share a meaningful root.

    Bi-directional substring match: ``review`` → ``reviewer`` AND
    ``reviewer`` → ``review``. Catches the common role-noun/verb pair
    where the agent name carries the agent-noun form and the task
    carries the verb form (or vice versa). Pure str ops — no regex
    (goldfive#166 / #167).
    """
    if not stem or not token:
        return False
    if stem == token:
        return True
    if stem in token or token in stem:
        return True
    return False


def _task_text_contains_stem(task: Task, stem: str) -> bool:
    """Return True when ``stem`` appears in task title+description tokens.

    Bi-directional containment via :func:`stem_token_match` against the
    ≥4-char tokens extracted from the task's title+description, mirror-
    ing the goldfive#265 Tier-2 selector exactly.
    """
    title = str(getattr(task, "title", "") or "")
    desc = str(getattr(task, "description", "") or "")
    tokens = tokenize_for_matching(f"{title} {desc}")
    if not tokens:
        return False
    for tok in tokens:
        if stem_token_match(stem, tok):
            return True
    return False


#: Phrases that mark a task as *delegation/coordination* shaped rather
#: than a leaf authoring task. When a task description matches any of
#: these (case-insensitive substring), Rule A is suppressed even if the
#: invoked agent has only ``AgentTool`` wrappers — coordinating IS what
#: a coordinator does, and that is the agent's actual capability.
#:
#: Conservative by design: false positives here would suppress real
#: capability mismatches. Only include phrases that strongly imply the
#: task itself is orchestrational. Verbs like "review", "research",
#: "patch", "locate" are deliberately NOT here — they are leaf-task
#: verbs that a coordinator structurally cannot perform.
DELEGATION_VERB_MARKERS: tuple[str, ...] = (
    "coordinate",
    "delegate",
    "orchestrate",
    "dispatch",
    "route to",
    "hand off",
    "handoff",
)


def is_agent_tool(tool: Any) -> bool:
    """Return True if ``tool`` is an ADK ``AgentTool`` (sub-agent wrapper).

    Mirrors the discriminator in :mod:`goldfive.adapters._adk_plugin`:
    prefers ``isinstance(tool, AgentTool)`` when the optional ``adk``
    extra is importable, with a duck-typed fallback (``.agent``
    attribute) for test stubs and forward-compatibility. A plain
    ``FunctionTool`` carries ``.func`` instead, so the absence of
    ``.agent`` is a robust no-AgentTool signal.
    """
    if tool is None:
        return False
    try:
        from google.adk.tools import AgentTool  # type: ignore  # noqa: PLC0415

        if isinstance(tool, AgentTool):
            return True
    except Exception:  # noqa: BLE001 — adk extra not installed / import edge
        pass
    return getattr(tool, "agent", None) is not None


def _tool_name(tool: Any) -> str:
    """Best-effort extraction of an ADK ``Tool``'s public name."""
    if tool is None:
        return ""
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    func = getattr(tool, "func", None)
    if func is not None:
        func_name = getattr(func, "__name__", None)
        if isinstance(func_name, str) and func_name:
            return func_name
    return ""


def _looks_like_delegation_task(task: Task) -> bool:
    """Return True when the task title/description reads as orchestrational.

    Substring scan against :data:`DELEGATION_VERB_MARKERS`,
    case-insensitive. Empty title + description is treated as
    *non-delegation* (the conservative default for Rule A: if we can't
    tell, prefer to fire).
    """
    title = str(getattr(task, "title", "") or "")
    description = str(getattr(task, "description", "") or "")
    haystack = f"{title}\n{description}".lower()
    if not haystack.strip():
        return False
    return any(marker in haystack for marker in DELEGATION_VERB_MARKERS)


def detect_capability_mismatch(
    *,
    invoked_agent_name: str,
    invoked_agent_tools: list[Any],
    task: Task,
    all_pending_tasks: Sequence[Task] | None = None,
) -> DriftEvent | None:
    """Return a CAPABILITY_MISMATCH drift if ``invoked_agent_name`` cannot perform ``task``.

    Parameters
    ----------
    invoked_agent_name:
        Display name of the agent the coordinator delegated to. Used
        only for the human-readable detail string; an empty value still
        produces a usable event.
    invoked_agent_tools:
        Live ADK ``Tool`` objects from the invoked agent. The detector
        introspects each via :func:`is_agent_tool` and ``_tool_name``;
        no calls are made.
    task:
        The :class:`~goldfive.types.Task` the coordinator bound to the
        delegation. ``required_tools`` powers Rule B; ``title`` /
        ``description`` power Rule A's leaf-task heuristic and Rule C's
        cross-task stem check.
    all_pending_tasks:
        Every ``PENDING`` task in the plan (DAG-ready and not). Powers
        the goldfive#268 Rule C cross-task lookup. ``None`` /empty
        disables Rule C — legacy callers and tests that don't pass it
        keep their pre-#268 behaviour exactly. Note Rule C is
        soft-retired and additionally requires
        ``GOLDFIVE_CAPABILITY_RULE_C=1`` to run at all (goldfive#423 /
        AGENCY-PRESERVATION.md PR 2).

    Returns
    -------
    DriftEvent | None
        ``None`` when no rule trips OR when ``invoked_agent_tools`` is
        empty AND ``required_tools`` is empty AND Rules A/C are disabled
        or have no signal. Otherwise a ``DriftEvent`` whose severity is
        ``WARNING`` for Rule B (the "WARNING-max" survivor) and
        ``CRITICAL`` for Rule A / Rule C (both soft-retired, default
        OFF; CRITICAL is the observability signal — the ladder demotes
        their cells to OBSERVE). Rules evaluate in order B → A → C; the
        first to fire wins so the higher-confidence signal (B, then A,
        then C) takes precedence and the steerer sees one verdict per
        delegation. By default only Rule B can fire (A/C gated off).
    """
    if task is None:
        return None

    tools = list(invoked_agent_tools or [])
    task_id = str(getattr(task, "id", "") or "")
    required_tools = tuple(getattr(task, "required_tools", ()) or ())

    # Rule B first — it consults explicit planner output, so it is the
    # higher-confidence signal. Fires only when populated; an empty
    # advisory is not a no-op miss, it's "no opinion".
    #
    # AGENCY-PRESERVATION.md PR 3 ("WARNING-max"): Rule B is the ONE
    # capability rule that survives as a steering trigger, because a
    # user-declared ``required_tools`` advisory is genuine prescriptive
    # intent (not goldfive's forecast). It now fires WARNING, not
    # CRITICAL: the CAPABILITY_MISMATCH ladder maps WARNING→ABSORB
    # (refine + continue) but CRITICAL→OBSERVE, so emitting WARNING
    # keeps Rule B's refine while capping it below the cancel/pause
    # escalation tier. (CAPABILITY_MISMATCH is not in
    # ``_GOLDFIVE_STEER_ELIGIBLE_KINDS``, so WARNING cannot auto-promote
    # to a steer either.)
    if required_tools:
        agent_tool_names = {n for n in (_tool_name(t) for t in tools) if n}
        missing = tuple(name for name in required_tools if name not in agent_tool_names)
        if missing:
            detail = (
                f"agent {invoked_agent_name!r} delegated for task "
                f"{task_id!r} is missing required tool(s) "
                f"{list(missing)!r}; available tools: "
                f"{sorted(agent_tool_names)!r}"
            )
            return DriftEvent(
                kind=DriftKind.CAPABILITY_MISMATCH,
                severity=DriftSeverity.WARNING,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=invoked_agent_name,
            )

    # Rule A — coordinator-style leaf-assignment. SOFT-RETIRED
    # (AGENCY-PRESERVATION.md PR 3): default OFF, re-enabled only via
    # ``GOLDFIVE_CAPABILITY_RULE_A=1``. Its leaf-task heuristic
    # (``_looks_like_delegation_task`` keyword scan + the AgentTool-only
    # leaf-title read) is stem/keyword NL classification — the
    # #166/#167 anti-pattern — and was the source of the 2d27ff4a
    # refine storm. Even when re-enabled it OBSERVEs (the ladder demotes
    # the CRITICAL cells); the flag restores the signal for debugging,
    # not steering. Hard deletion follows in PR 13. Empty tool list does
    # not trip Rule A regardless: we cannot distinguish "agent has no
    # tools" from "test stub / introspection failure".
    if _rule_a_enabled() and tools and all(is_agent_tool(t) for t in tools):
        if not _looks_like_delegation_task(task):
            detail = (
                f"agent {invoked_agent_name!r} has only AgentTool "
                f"wrappers ({len(tools)} delegation tools, no leaf "
                f"capability) but was delegated leaf task "
                f"{task_id!r} ({(task.title or '')[:80]!r}); "
                f"the agent structurally cannot perform this task"
            )
            return DriftEvent(
                kind=DriftKind.CAPABILITY_MISMATCH,
                severity=DriftSeverity.CRITICAL,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=invoked_agent_name,
            )

    # Rule C — out-of-DAG-order delegation (goldfive#268). When the
    # invoked agent's role stem doesn't appear in the bound task but
    # DOES appear in another PENDING task, the pin has bound the
    # delegation to a structurally-wrong task. Mirror live evidence:
    # ``reviewer_agent`` pinned to ``draft_slides`` while
    # ``review_presentation`` sits PENDING and not-yet-DAG-ready, the
    # pin's tier-2 stem disambiguator couldn't help because only one
    # task was eligible. Rule C fires on that exact shape.
    #
    # SOFT-RETIRED (goldfive#423 / AGENCY-PRESERVATION.md PR 2): the
    # rule no longer runs unless the operator explicitly re-enables it
    # via ``GOLDFIVE_CAPABILITY_RULE_C=1``. Descriptive growth at pin
    # time grows the plan when no structurally-right task exists, so
    # the mispin shape Rule C detected no longer occurs (design doc
    # §7). Hard deletion of ``_rule_c_dag_order`` + the
    # ``all_pending_tasks`` parameter follows in AGENCY-PRESERVATION.md
    # PR 13 (§7.1 step 2).
    #
    # Cross-task signal — needs the full PENDING set (DAG-ready and
    # not). Silent when the caller didn't pass it, when the agent
    # name produces no usable stem (generic ``coordinator_agent`` style
    # names degrade gracefully), or when no other PENDING task
    # mentions the stem (in which case Rule C has nothing to say — the
    # pin is just doing its best with a generic agent).
    if _rule_c_enabled():
        drift_c = _rule_c_dag_order(
            invoked_agent_name=invoked_agent_name,
            bound_task=task,
            all_pending_tasks=all_pending_tasks,
        )
        if drift_c is not None:
            return drift_c

    return None


def _rule_c_dag_order(
    *,
    invoked_agent_name: str,
    bound_task: Task,
    all_pending_tasks: Sequence[Task] | None,
) -> DriftEvent | None:
    """Rule C: detect out-of-DAG-order delegation (goldfive#268).

    Returns a CAPABILITY_MISMATCH ``DriftEvent`` iff the agent's role
    stem is *absent* from the bound task's title+description AND
    *present* in some other PENDING task. Otherwise ``None``.

    Conservative bail-outs (all return ``None``):

    * ``all_pending_tasks`` is ``None`` or empty.
    * The agent name produces no stem of length ≥4
      (``coordinator_agent`` → ``coordinator`` is a stem, but a generic
      ``agent`` / ``worker`` collapses to the empty tuple and we skip).
    * The stem is already present in the bound task — no conflict, the
      pin is on the right task.
    * No other PENDING task mentions the stem either — Rule C has no
      strong signal of cross-task confusion.
    """
    if not all_pending_tasks:
        return None
    stems = agent_name_stems(invoked_agent_name)
    if not stems:
        return None

    bound_id = str(getattr(bound_task, "id", "") or "")

    # For each stem, check: absent-from-bound AND present-in-other.
    # Iterate stems so multi-stem names ("draft_writer_agent" -> draft +
    # writer) get a chance to match. Fire on the first stem that has
    # the (absent here, present there) shape.
    for stem in stems:
        if _task_text_contains_stem(bound_task, stem):
            # Stem present in bound — no conflict for this stem, try
            # the next stem before giving up.
            continue
        other_match: Task | None = None
        for other in all_pending_tasks:
            other_id = str(getattr(other, "id", "") or "")
            if not other_id or other_id == bound_id:
                continue
            if _task_text_contains_stem(other, stem):
                other_match = other
                break
        if other_match is None:
            continue
        other_id = str(getattr(other_match, "id", "") or "")
        other_title = str(getattr(other_match, "title", "") or "")
        bound_title = str(getattr(bound_task, "title", "") or "")
        detail = (
            f"agent {invoked_agent_name!r} bound to task "
            f"{bound_id!r} ({bound_title[:80]!r}) but agent's role-stem "
            f"{stem!r} matches PENDING task {other_id!r} "
            f"({other_title[:80]!r}); the coordinator likely "
            f"delegated out of DAG order"
        )
        return DriftEvent(
            kind=DriftKind.CAPABILITY_MISMATCH,
            severity=DriftSeverity.CRITICAL,
            detail=detail,
            current_task_id=bound_id,
            current_agent_id=invoked_agent_name,
        )
    return None


# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------
#
# CAPABILITY_MISMATCH is a purely structural detector — no LLM call, no
# JSON parsing, no observability truncation. The config is therefore
# all-default (``uses_llm=False``); we register so callers that
# discover detectors via the registry can dispatch by kind uniformly.


from goldfive.drift.registry import DetectorConfig as _DetectorConfig  # noqa: E402
from goldfive.drift.registry import register as _register  # noqa: E402

_register(
    DriftKind.CAPABILITY_MISMATCH,
    detect_capability_mismatch,
    _DetectorConfig(uses_llm=False),
    is_async=False,
)
