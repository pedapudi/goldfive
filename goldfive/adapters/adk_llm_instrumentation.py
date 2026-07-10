"""Request-side ADK LLM instrumentation for the goldfive adapter.

Wave B2 of the modularization plan consolidates the request-side
``LlmRequest`` mutation + measurement surface into one audit-friendly
module. Before this file existed the surface was spread across three
files (``_adk_dynainst.py``, ``_adk_state_protocol.py``,
``_adk_plugin.py``); now everything that touches an ``LlmRequest`` on
the way IN to the model lives here.

The module owns four concerns:

1. **Per-call instrumentation** (:func:`_measure_request_chars`) —
   read-only character / message counters consumed by the plugin's
   ``before_model_callback`` for the ``goldfive.llm.request`` log line
   and event payloads (goldfive#172).
2. **Runtime tool-surface hint** (:data:`_RUNTIME_TOOLS_HINT_PREFIX`,
   :data:`_RUNTIME_TOOLS_HINT_END`, :func:`_build_runtime_tools_hint`,
   :func:`_strip_prior_runtime_tools_hint`) — composes the marker-
   bracketed "currently-relevant tools" block that's appended to the
   system instruction every turn; strip-then-re-inject keeps the
   marker count at exactly one (R3 dedup contract). The actual
   injection site lives in
   :class:`~goldfive.prompt_shaper.PromptShaper` per Wave B1; this
   module owns the composition primitives.
3. **Max-output-tokens ratchet** (:func:`_apply_agent_max_output_tokens_cap`,
   :data:`DEFAULT_AGENT_MAX_OUTPUT_TOKENS`) — structural ceiling
   applied to ``llm_request.config.max_output_tokens`` for sub-agent
   LLM calls (goldfive#256). Smaller-wins; non-positive ceiling is
   the operator opt-out.
4. **Dynamic instruction resolver plumbing**
   (:func:`install_dynamic_instructions`, :func:`format_correction_block`,
   :func:`pending_correction_key`, :func:`is_dynamic_instruction`, etc.)
   — replaces each wrapped ``LlmAgent``'s static ``instruction``
   string with a per-turn callable that re-resolves the current
   task from goldfive Session.state (goldfive#251 plan-causal
   prompting). The closure shape itself is produced by
   :meth:`~goldfive.prompt_shaper.PromptShaper.make_dynamic_instruction`
   (the ``observation_only`` gate lives there per Wave B1); this
   module installs the resolver on the agent tree and supplies the
   reading / rendering primitives the closure delegates to.

Boundary with :mod:`goldfive.adapters._adk_state_protocol` (intentionally
left as a sibling):

  The state-protocol module owns the audit-guarded ``_set`` shim and
  the key-name constants for ``goldfive.*`` ADK ``session.state``
  writes. :mod:`goldfive._state_audit` patches its ``_set`` at import
  time and catalogues its file path in ``_KNOWN_CALLERS``. Merging it
  here would force a catalog-and-patch surface migration that is
  high-risk and bloats the new module unhelpfully — the state-protocol
  layer is delicate (V2/V3/V4 migration history; ``KEY_CURRENT_TASK_ID``
  etc. are contract) and the audit's stack-walk caller-allowlist
  encodes the file path explicitly.

Boundary with :mod:`goldfive.adapters._adk_plugin` (kept):

  The response-side callbacks (``before_tool_callback``,
  ``after_model_callback``, ``on_event_callback``, ``on_tool_error_callback``)
  stay in the plugin module — they are not request-side mutation. The
  plugin re-exports the request-side helpers below at module load so
  in-module references (and the historical
  ``from goldfive.adapters._adk_plugin import _measure_request_chars``
  callsites in tests) keep resolving without change.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from goldfive.adapters import _adk_state_protocol as _sp

log = logging.getLogger("goldfive.adapters.adk_llm_instrumentation")


# ---------------------------------------------------------------------------
# Local defensive attribute reader
# ---------------------------------------------------------------------------
#
# Mirrors the helper inside :mod:`._adk_plugin` so this module stays
# independent of an optional google-adk install. Same semantics: a
# stub object that doesn't carry the attribute degrades to ``default``
# instead of raising.
def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Per-LLM-call instrumentation (goldfive#172)
# ---------------------------------------------------------------------------


def _measure_request_chars(llm_request: Any) -> tuple[int, int]:
    """Return ``(total_chars, messages_count)`` for an ADK ``LlmRequest``.

    Used by the goldfive#172 per-LLM-call instrumentation in
    :meth:`_GoldfiveADKPlugin.before_model_callback`. We walk
    ``llm_request.contents`` (a list of ``Content`` whose ``parts`` hold
    ``text`` / ``function_call`` / ``function_response`` leaves) and
    sum the character count of each serialised part. The three leaf
    shapes we care about:

    * ``part.text`` -- plain assistant / user / system text.
    * ``part.function_call`` -- a model-emitted tool call. We serialise
      as ``name + json(args)`` because both contribute to the prompt
      tokens the model pays for.
    * ``part.function_response`` -- a tool's return payload. Serialised
      as ``name + json(response)`` for the same reason.

    Unknown part shapes fall through silently (count zero) so a novel
    ADK part type doesn't break instrumentation. The system instruction
    on ``llm_request.config.system_instruction`` is counted separately
    when present, since it's a prompt-prefix the model must process on
    every call — often the dominant contributor right after a
    GoldfivePlanner injection.

    Returns ``(0, 0)`` on any failure — instrumentation must never
    raise into the caller path.
    """
    try:
        contents = _safe_attr(llm_request, "contents", None) or []
        total_chars = 0
        messages_count = 0
        for content in contents:
            messages_count += 1
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                text = _safe_attr(part, "text", "") or ""
                if text:
                    total_chars += len(str(text))
                    continue
                fc = _safe_attr(part, "function_call", None)
                if fc is not None:
                    name = str(_safe_attr(fc, "name", "") or "")
                    args = _safe_attr(fc, "args", None)
                    total_chars += len(name)
                    if args is not None:
                        try:
                            total_chars += len(json.dumps(args, default=repr))
                        except Exception:  # noqa: BLE001
                            total_chars += len(repr(args))
                    continue
                fr = _safe_attr(part, "function_response", None)
                if fr is not None:
                    name = str(_safe_attr(fr, "name", "") or "")
                    resp = _safe_attr(fr, "response", None)
                    total_chars += len(name)
                    if resp is not None:
                        try:
                            total_chars += len(json.dumps(resp, default=repr))
                        except Exception:  # noqa: BLE001
                            total_chars += len(repr(resp))
                    continue
        # Include the system instruction — a GoldfivePlanner injection
        # lands here and typically dominates the prompt prefix.
        config = _safe_attr(llm_request, "config", None)
        sys_inst = _safe_attr(config, "system_instruction", "") or ""
        if isinstance(sys_inst, str):
            total_chars += len(sys_inst)
        elif sys_inst is not None:
            try:
                total_chars += len(str(sys_inst))
            except Exception:  # noqa: BLE001
                pass
        return total_chars, messages_count
    except Exception:  # noqa: BLE001 — instrumentation must never raise
        return 0, 0


# ---------------------------------------------------------------------------
# Runtime tool-surface hint (goldfive#168 R3)
# ---------------------------------------------------------------------------
#
# Marker tags bracketing the runtime tool-surface hint that the
# request-side prompt shaper appends to ``system_instruction``. The
# pair lets a follow-up call locate and strip the previously injected
# block before appending the freshly-computed one, so the marker count
# is exactly one per request (R3 dedup contract). The actual injection
# logic lives on :class:`~goldfive.prompt_shaper.PromptShaper`; the
# composition / strip primitives live here.

_RUNTIME_TOOLS_HINT_PREFIX: str = "[GOLDFIVE PLAN-STATE HINT —"
_RUNTIME_TOOLS_HINT_END: str = "[/GOLDFIVE PLAN-STATE HINT]"


def _build_runtime_tools_hint(session: Any) -> str | None:
    """Compose a 'currently-relevant tools' hint for injection into the LLM context.

    Walks ``session.plan.tasks`` and groups them by ``assignee_agent_id``.
    For each agent, summarise:

    * tasks already DONE (terminal — COMPLETED / FAILED / CANCELLED / NOT_NEEDED)
    * tasks PENDING (with their titles, capped at three for brevity)
    * whether the agent has any remaining work

    Returns a multi-line string suitable for prepending to the LLM
    request as a system-level guidance message. Returns ``None`` when
    there's no plan to summarise (turn 1, or pre-plan-install
    windows), or when the plan groups produce no useful signal.

    The output is bracketed by :data:`_RUNTIME_TOOLS_HINT_PREFIX` and
    :data:`_RUNTIME_TOOLS_HINT_END` so a follow-up call can detect and
    strip the previous hint from ``system_instruction`` before
    appending the fresh one (R3 dedup contract).
    """
    plan = _safe_attr(session, "plan", None)
    if plan is None:
        return None
    tasks = _safe_attr(plan, "tasks", None)
    if not tasks:
        return None

    try:
        from goldfive.types import TERMINAL_TASK_STATUSES
    except ImportError:  # pragma: no cover — should never happen
        return None

    by_agent: dict[str, dict[str, list[str]]] = {}
    for t in tasks:
        agent = _safe_attr(t, "assignee_agent_id", "") or "<unassigned>"
        bucket = by_agent.setdefault(agent, {"done": [], "remaining": []})
        status = _safe_attr(t, "status", None)
        title = _safe_attr(t, "title", "") or _safe_attr(t, "id", "") or "?"
        if status in TERMINAL_TASK_STATUSES:
            bucket["done"].append(str(title))
        else:
            bucket["remaining"].append(str(title))

    body_lines: list[str] = []
    for agent in sorted(by_agent):
        info = by_agent[agent]
        # Strip any namespace separator so the hint matches what the
        # LLM sees as the bare tool / sub-agent name.
        bare = agent.split(":")[-1] if ":" in agent else agent
        if info["remaining"]:
            tasks_summary = "; ".join(info["remaining"][:3])
            body_lines.append(f"  {bare}: PENDING — {tasks_summary}")
        elif info["done"]:
            body_lines.append(f"  {bare}: all assigned tasks complete; do NOT re-invoke this agent")

    if not body_lines:
        return None

    lines: list[str] = [f"{_RUNTIME_TOOLS_HINT_PREFIX} runtime guidance, not user-authored]"]
    lines.extend(body_lines)
    lines.append(
        "Choose the agent whose tasks are still PENDING. Do not re-invoke "
        "agents whose tasks are already complete."
    )
    lines.append(_RUNTIME_TOOLS_HINT_END)
    return "\n".join(lines)


def _strip_prior_runtime_tools_hint(existing: str) -> str:
    """Remove a previously-injected runtime-tools hint from ``existing``.

    The hint is bracketed by :data:`_RUNTIME_TOOLS_HINT_PREFIX` and
    :data:`_RUNTIME_TOOLS_HINT_END`. When found, both markers and the
    text between them are removed; surrounding ``\\n\\n`` separators
    are normalised so the result has no orphan blank lines.

    Returns the input unchanged when no prior hint marker is present.
    """
    if _RUNTIME_TOOLS_HINT_PREFIX not in existing:
        return existing
    start = existing.find(_RUNTIME_TOOLS_HINT_PREFIX)
    end = existing.find(_RUNTIME_TOOLS_HINT_END, start)
    if end == -1:
        # Truncated / malformed — drop from prefix to end of string.
        cleaned = existing[:start]
    else:
        cleaned = existing[:start] + existing[end + len(_RUNTIME_TOOLS_HINT_END) :]
    # Collapse any 3+ consecutive newlines created by the removal.
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip("\n")


# ---------------------------------------------------------------------------
# Max-output-tokens ratchet (goldfive#256)
# ---------------------------------------------------------------------------

#: Default per-ADK-sub-agent ``max_output_tokens`` ceiling enforced by
#: :class:`_GoldfiveADKPlugin` (goldfive#256). When a sub-agent's
#: ``llm_request.config.max_output_tokens`` is unset, OR is greater
#: than this value, the plugin RATCHETS IT DOWN to this ceiling. When
#: the sub-agent (or ADK's defaults) already supplied a smaller cap,
#: the smaller value wins — this is a structural CEILING, not an
#: override. Set to ``0`` (or any negative int) to disable the
#: ratcheting entirely (the plugin then leaves ``max_output_tokens``
#: untouched, which is the pre-#256 behaviour). Default 16384 matches
#: :attr:`goldfive.planner.LLMPlanner.MAX_OUTPUT_TOKENS` — the same
#: budget the planner uses for refine-shaped completions. Operators
#: tune via the typed :class:`~goldfive.config.AgentConfig`
#: (``RuntimeConfig(agent=AgentConfig(max_output_tokens=...))``) or
#: the env var ``GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS``.
DEFAULT_AGENT_MAX_OUTPUT_TOKENS: int = 16384


def _apply_agent_max_output_tokens_cap(llm_request: Any, ceiling: int) -> tuple[int, int]:
    """Ratchet ``llm_request.config.max_output_tokens`` down to ``ceiling`` (goldfive#256).

    Smaller-wins semantics: when ``config.max_output_tokens`` is already
    set to a value smaller than ``ceiling`` we leave it alone — the
    sub-agent / ADK chose a tighter cap and goldfive only ratchets DOWN.
    When the existing value is missing, zero, negative, or larger than
    ``ceiling``, we write ``ceiling``.

    ``ceiling <= 0`` is the operator opt-out: the function returns
    ``(0, 0)`` and leaves ``llm_request`` untouched (the same shape as
    setting ``GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS`` to a non-positive int).

    Returns ``(previous_value, applied_value)`` where ``previous_value``
    is what the request carried on entry (``0`` for "missing / unset")
    and ``applied_value`` is what it carries on exit. Useful for tests
    and the diagnostic INFO log the caller emits.

    Best-effort: any failure reading or writing ``llm_request.config``
    is swallowed at DEBUG so a future ADK schema change can't crash the
    callback. The cap is a structural safety net, not a hard invariant
    — the watcher and the planner cap still bound runaway calls when
    this helper short-circuits.
    """
    if ceiling <= 0:
        return (0, 0)
    config = getattr(llm_request, "config", None)
    if config is None:
        return (0, 0)
    try:
        existing = getattr(config, "max_output_tokens", None)
    except Exception:  # noqa: BLE001
        existing = None
    previous = int(existing) if isinstance(existing, int) and existing > 0 else 0
    # Smaller-wins: a sub-agent / ADK that pinned a tighter cap keeps it.
    if previous > 0 and previous <= ceiling:
        return (previous, previous)
    try:
        config.max_output_tokens = int(ceiling)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "_apply_agent_max_output_tokens_cap: could not set max_output_tokens: %s",
            exc,
        )
        return (previous, previous)
    return (previous, int(ceiling))


# ---------------------------------------------------------------------------
# Dynamic instruction resolver (goldfive#251 — plan-causal prompting)
# ---------------------------------------------------------------------------
#
# Each wrapped ``LlmAgent``'s ``instruction`` field is bound at
# ``LlmAgent(...)`` construction time. When goldfive's plan changes
# mid-run (refine landing, task supersedes, correction injection), the
# plan updates in goldfive ``Session.state`` but the agent's baked-in
# prompt does not — so the LLM keeps executing its original instruction
# and only "observes" the plan shift through unrelated channels.
#
# The resolver fixes that by replacing each wrapped ``LlmAgent``'s
# static ``instruction`` string with a callable
# ``(ReadonlyContext) -> str`` that re-resolves the agent's current
# task from goldfive ``Session.state`` at every turn. ADK's
# ``canonical_instruction`` invokes the callable per turn and returns
# ``bypass_state_injection=True`` for the result, so refine landing in
# state is picked up on the NEXT turn with no transcript rewrite.
# Because of that bypass, the resolver itself re-applies ADK's
# ``inject_session_state`` to the original template when it carries
# ``{var}`` / ``{artifact.var}`` placeholders — otherwise wrapping
# would silently disable the documented session-state templating.
#
# The resolver is **agent-agnostic**: works for any wrapped ``LlmAgent``
# (coordinator, sub-agent, root-with-tools, leaf). Current-task
# resolution is driven entirely by whatever is pinned in state for the
# specific agent invocation via
# :data:`goldfive.adapters._adk_state_protocol.KEY_CURRENT_TASK_ID`.

_MISSING_TITLE_PLACEHOLDER = "(title unset)"
_MISSING_DESCRIPTION_PLACEHOLDER = "(description unset)"


def pending_correction_key(agent_name: str, task_id: str) -> str:
    """Return the full state key for a pending correction.

    Keyed by ``(agent_name, task_id)`` so a correction targeted at one
    agent/task pair does not leak into another agent's prompt. Stream D
    (correction injection) owns the writer; this module owns the reader.
    """
    return f"{_sp.KEY_PENDING_CORRECTIONS}.{agent_name}.{task_id}"


def format_correction_block(correction: Mapping[str, Any]) -> str:
    """Render a pending-correction dict into the prompt block a resolver appends.

    Stream D (goldfive#251 :mod:`goldfive._correction_injection`) writes a
    structured dict describing the correction; this helper is the Stream
    B (resolver) side of that contract and owns the exact language the
    LLM sees.

    Design principle: **directive, not diagnostic.** We tell the LLM
    what to do on the corrected task ("focus only on", "do not
    propagate"), NOT what went wrong with the prior task ("was broken",
    "failed"). Problem-naming language is an attractor for LLM pattern-
    matching failure modes — meta-commentary, apologies, retries of the
    wrong thing (see goldfive#250 / #252 / #253 / #259's lessons on
    response-shape minimality). The diagnostic data (drift kind, drift
    reason, revision number) is still present in the dict for sinks /
    observability but is deliberately NOT interpolated into the LLM-
    visible block.

    The rendered block is designed to slot after the ``Current
    assigned task`` section via :func:`_compose_instruction`:

        ---
        Plan was revised (REV {n}). Your prior output for
        task "{superseded_task_title}" (id {superseded_task_id}) was
        superseded.

        Focus only on the revised scope as described above in
        "Current assigned task." Do not propagate the superseded
        content into downstream dispatches.
        ---

    Missing fields degrade to sensible placeholders so a partial dict
    (sink-shaped, persistence-restored) still renders cleanly.
    """
    if not isinstance(correction, Mapping):
        return ""
    rev = correction.get("revision_number", 0)
    try:
        rev_num = int(rev) if rev is not None else 0
    except (TypeError, ValueError):
        rev_num = 0
    superseded_title = str(correction.get("superseded_task_title", "") or "(prior task)")
    superseded_id = str(correction.get("superseded_task_id", "") or "")
    id_fragment = f" (id {superseded_id})" if superseded_id else ""

    return (
        "---\n"
        f"Plan was revised (REV {rev_num}). Your prior output for "
        f"task \"{superseded_title}\"{id_fragment} was superseded.\n"
        "\n"
        "Focus only on the revised scope as described above in "
        "\"Current assigned task.\" Do not propagate the superseded "
        "content into downstream dispatches.\n"
        "---"
    )


def _read_pending_correction(
    *,
    session: Any,
    state: Mapping[str, Any],
    agent_name: str,
    current_task_id: str,
) -> str:
    """Resolve the pending-correction block for ``(agent, task)``.

    Phase 2.0 of goldfive#271 — bridge eliminated. The goldfive
    :class:`~goldfive.state_store.StateStore` is now
    the read of record. Falls back to reading ADK ``state`` directly
    only when the SessionContext stash is unreachable (legacy unit
    tests / custom adapters that drive the resolver against a plain
    state dict without the stash).

    Always returns a string (empty when no correction exists / the
    payload is malformed). Caller appends the result to the
    instruction block when non-empty.
    """
    if session is not None:
        from goldfive.state_store import StateStore

        store = StateStore.for_session(session)
        value = store.get_correction(agent_name, current_task_id)
        return _resolve_pending_correction(value)
    # Legacy fallback: no SessionContext reachable. Read directly off
    # ADK state. Used by unit tests that drive the resolver with a
    # plain state dict; production paths always carry the stash.
    return _resolve_pending_correction(
        state.get(pending_correction_key(agent_name, current_task_id))
    )


def _resolve_pending_correction(raw: Any) -> str:
    """Normalise a pending-correction state value to the final prompt string.

    Accepts three shapes so the resolver stays forgiving as Stream D's
    write contract evolves:

    * Mapping — the structured dict Stream D writes; rendered via
      :func:`format_correction_block`.
    * Non-empty string — treated as a pre-rendered block (back-compat
      path for tests and operators who stamped a literal string into
      state before Stream D existed, or who want to override the
      template for a one-off).
    * Anything else (None, empty string, bool, int) — degrades to an
      empty string, which :func:`_compose_instruction` then skips.
    """
    if raw is None:
        return ""
    if isinstance(raw, Mapping):
        return format_correction_block(raw)
    if isinstance(raw, str):
        return raw
    return ""


def _state_from_readonly_context(readonly_ctx: Any) -> Mapping[str, Any]:
    """Pull ``session.state`` off a ``ReadonlyContext`` defensively.

    ADK exposes ``readonly_context.state`` as a ``MappingProxyType`` over
    the live session.state dict. We accept anything mapping-shaped and
    degrade to an empty dict on exotic / stub contexts so the resolver
    never raises from inside ADK's instruction pipeline.
    """
    state = getattr(readonly_ctx, "state", None)
    if isinstance(state, Mapping):
        return state
    return {}


def _compose_instruction(
    *,
    original: str,
    task_id: str,
    task_title: str,
    task_description: str,
    pending_correction: str,
    task_kind: str = "",
    goals_block: str = "",
) -> str:
    """Assemble the final prompt string the LLM sees this turn.

    Shape (FORECAST / OUTCOME pin — the legacy task block):

        {original}

        ---

        Current assigned task:
          id: {task_id}
          title: {task_title}
          description: {task_description}

        {pending_correction}  # appended only when non-empty

    AGENCY-PRESERVATION.md Stage 3 PR 12 — when the pinned task is
    DISCOVERED-kind (``task_kind == TaskKind.DISCOVERED.value`` and a
    non-empty ``goals_block`` is supplied), render a ``[GOALS]`` block
    INSTEAD of the ``Current assigned task:`` block:

        {original}

        ---

        [GOALS]
        {goals_block}

        {pending_correction}

    Rationale: a DISCOVERED task is the agent's OWN observed means-work
    (the trajectory lane of the ledger), not a forecast the agent is
    graded against. Pinning it back as a prescription re-imposes exactly
    the forecast framing PR 11/12 retire — so for a discovered pin we
    ground the agent on the user's GOALS and let it own the means.
    ``task_kind``/``goals_block`` default to ``""`` so every legacy /
    forecast / OUTCOME caller renders the unchanged task block (the
    DISCOVERED kind only exists in ledger plan mode — forecast-mode pins
    are byte-identical, §5.1).

    Interaction with the PR 9 prompt-shaping diet: under
    ``signal_channel == "request_context"`` with ``pin_assigned_task``
    off, the resolver returns the original instruction BEFORE reaching
    this function (the pin is retired entirely), so the ``[GOALS]`` block
    applies only where a pin WOULD render — legacy ``signal_channel`` or
    ``pin_assigned_task=True``.
    """
    from goldfive.types import TaskKind

    if task_kind == TaskKind.DISCOVERED.value and goals_block:
        block = (
            f"{original}\n"
            "\n"
            "---\n"
            "\n"
            "[GOALS]\n"
            f"{goals_block}\n"
        )
        if pending_correction:
            block = f"{block}\n{pending_correction}\n"
        return block

    title = task_title or _MISSING_TITLE_PLACEHOLDER
    description = task_description or _MISSING_DESCRIPTION_PLACEHOLDER

    block = (
        f"{original}\n"
        "\n"
        "---\n"
        "\n"
        "Current assigned task:\n"
        f"  id: {task_id}\n"
        f"  title: {title}\n"
        f"  description: {description}\n"
    )
    if pending_correction:
        block = f"{block}\n{pending_correction}\n"
    return block


def _goldfive_session_from_readonly_context(readonly_ctx: Any) -> Any:
    """Return the goldfive ``Session`` reachable from a ReadonlyContext.

    Phase 2.0 of goldfive#271. Resolution order:

    1. Walk ``readonly_ctx._invocation_context.plugin_manager.plugins``
       for the goldfive plugin and read its ``_active_ctx.session``.
       This is the live-run path — set by the plugin's
       :meth:`set_active_context` (called from
       :meth:`ADKAdapter._invoke_internal` before
       ``runner.run_async``).
    2. The legacy ``"goldfive._session_context"`` stash on the
       readonly context's ``state`` (V7 in the Phase 0 audit). Used
       only by unit tests that drive the resolver against a
       hand-built state dict without going through the plugin's
       lifecycle.

    Returns ``None`` when neither resolves so the resolver can fall
    back to reading from ADK state directly.
    """
    ctx = _goldfive_session_context_from_readonly_context(readonly_ctx)
    if ctx is not None:
        session = getattr(ctx, "session", None)
        if session is not None:
            return session
    # Legacy fallback — the V7 stash on the state dict (already
    # handled by :func:`_goldfive_session_context_from_readonly_context`
    # via the ``"goldfive._session_context"`` lookup, but kept here
    # explicitly so a caller-supplied legacy ctx without a usable
    # ``session`` attribute still returns ``None`` cleanly).
    return None


def _goldfive_session_context_from_readonly_context(readonly_ctx: Any) -> Any:
    """Return the goldfive ``SessionContext`` reachable from ``readonly_ctx``.

    Same resolution order as :func:`_goldfive_session_from_readonly_context`
    but yields the ``SessionContext`` itself so callers can read other
    fields (notably ``.steerer`` for the goldfive#271 strict-passive
    observation_only gate). Returns ``None`` when no context is
    reachable.
    """
    from goldfive.adapters._adk_plugin import session_context_from_invocation

    inv_ctx = getattr(readonly_ctx, "_invocation_context", None) or getattr(
        readonly_ctx, "invocation_context", None
    )
    if inv_ctx is not None:
        ctx = session_context_from_invocation(inv_ctx)
        if ctx is not None:
            return ctx
    # Legacy fallback — the V7 stash on the state dict.
    state = getattr(readonly_ctx, "state", None)
    if not isinstance(state, Mapping):
        return None
    legacy_ctx = state.get("goldfive._session_context")
    if legacy_ctx is None:
        return None
    return legacy_ctx


def _task_title_description_from_session(session: Any, task_id: str) -> tuple[str, str]:
    """Look up ``(title, description)`` for ``task_id`` in ``session.plan``.

    The typed :class:`~goldfive.types.Task` on ``Session.plan.tasks`` is
    the source of truth — the resolver reaches into it for the prompt
    block instead of reading the de-normalised
    ``goldfive.current_task_title`` /
    ``goldfive.current_task_description`` keys that V1 / V3 / V5 used to
    stamp onto ADK state.

    Returns ``("", "")`` when the plan / task is missing or doesn't
    expose ``.title`` / ``.description`` so the caller's placeholder
    rendering still kicks in cleanly.
    """
    if session is None or not task_id:
        return "", ""
    plan = getattr(session, "plan", None)
    if plan is None:
        return "", ""
    tasks = getattr(plan, "tasks", None) or ()
    for task in tasks:
        if str(getattr(task, "id", "") or "") == task_id:
            title = str(getattr(task, "title", "") or "")
            description = str(getattr(task, "description", "") or "")
            return title, description
    return "", ""


def _task_kind_from_session(session: Any, task_id: str) -> str:
    """Return the :attr:`Task.kind` VALUE for ``task_id`` in ``session.plan``.

    AGENCY-PRESERVATION.md Stage 3 PR 12. Returns the kind's string value
    (e.g. ``"DISCOVERED"`` / ``"OUTCOME"`` / ``"FORECAST"``) so the
    resolver can choose the ``[GOALS]`` block for a discovered pin. Returns
    ``""`` when the plan / task is missing or carries no kind — the caller
    then renders the unchanged task block. Forecast-mode tasks are always
    FORECAST-kind, so this never selects the discovered branch outside
    ledger plan mode (§5.1 forecast bit-identity).
    """
    if session is None or not task_id:
        return ""
    plan = getattr(session, "plan", None)
    if plan is None:
        return ""
    for task in getattr(plan, "tasks", None) or ():
        if str(getattr(task, "id", "") or "") == task_id:
            kind = getattr(task, "kind", None)
            return str(getattr(kind, "value", kind) or "")
    return ""


def _goals_block_from_session(session: Any) -> str:
    """Render ``session.goals`` as the body of the PR-12 ``[GOALS]`` block.

    One bullet per goal summary; goals without a summary are skipped.
    Returns ``""`` when there are no goals so the caller falls back to the
    task block (a discovered pin with no goals has nothing to ground on).
    Never raises.
    """
    try:
        goals = getattr(session, "goals", None) or ()
        lines: list[str] = []
        for g in goals:
            summary = str(getattr(g, "summary", "") or "").strip()
            if summary:
                lines.append(f"  - {summary}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def is_dynamic_instruction(value: Any) -> bool:
    """Return True when ``value`` is a resolver produced by this module."""
    return bool(getattr(value, "_goldfive_dynamic_instruction", False))


def _adk_inject_session_state() -> Any:
    """Return ADK's async ``inject_session_state`` helper, or ``None``.

    ADK treats a callable ``instruction`` as ``bypass_state_injection=
    True`` (``LlmAgent.canonical_instruction``), so swapping a string
    instruction for a resolver silently disables the documented
    ``{var}`` / ``{artifact.var}`` templating unless the resolver
    re-applies it. The resolver calls this helper's return value on the
    original template to restore wrapped == unwrapped semantics.

    Probed lazily (never at import) so the module stays importable
    without google-adk, and defensively so an ADK release that moves
    the helper degrades to ``None`` instead of raising — the installer
    then skips placeholder-bearing instructions with a WARNING rather
    than break their templating.
    """
    try:
        from google.adk.utils import instructions_utils
    except Exception:  # noqa: BLE001 — ADK absent or restructured
        return None
    fn = getattr(instructions_utils, "inject_session_state", None)
    return fn if callable(fn) else None


def _looks_like_llm_agent(node: Any) -> bool:
    """Duck-type check for an ADK ``LlmAgent`` without importing ADK.

    ``LlmAgent`` carries an ``instruction`` pydantic field (``Union[str,
    InstructionProvider]``). ``SequentialAgent`` / ``ParallelAgent`` /
    custom ``BaseAgent`` subclasses do not. Presence of ``instruction``
    is a stable discriminator and avoids forcing an ADK import at
    wrap-time.
    """
    return node is not None and hasattr(node, "instruction")


def install_dynamic_instructions(root_agent: Any) -> int:
    """Replace every reachable ``LlmAgent``'s ``instruction`` with a resolver.

    Traverses the same three edges the rest of the adapter uses
    (``sub_agents`` / ``inner_agent`` / ``AgentTool.agent``) so any
    wrapped-tree shape is covered: coordinator+AgentTool, native
    ``sub_agents``, nested tool trees, etc.

    For every ``LlmAgent`` encountered:

    * If the existing ``instruction`` is already a dynamic resolver (re-
      wrap on the same tree): skip. Idempotent.
    * If it is a callable (user-supplied ``InstructionProvider``): leave
      it alone — the caller is already managing their own dynamic
      resolution, and we must not double-wrap them.
    * If it is a string (the common case): capture the string as the
      closure's ``original``, capture ``agent.name`` for correction
      lookup, install the resolver.
    * If the string carries ``{var}`` placeholders but ADK's
      ``inject_session_state`` cannot be resolved: skip with a WARNING
      so the agent keeps ADK's native string-instruction templating.

    Non-``LlmAgent`` nodes are walked through but not modified.

    Returns the number of agents that had their instruction replaced.
    Silent on assignment failures (frozen pydantic models): one bad
    agent must not block the rest of the tree.
    """
    if root_agent is None:
        return 0

    touched = 0
    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        # Walk edges BEFORE the LlmAgent check so non-LlmAgent containers
        # (SequentialAgent, ParallelAgent) still propagate to their
        # children. Same pattern as _attach_goldfive_planner_to_tree.
        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for tool in getattr(cur, "tools", None) or ():
            nested = getattr(tool, "agent", None)
            if nested is not None:
                stack.append(nested)

        if not _looks_like_llm_agent(cur):
            continue

        existing = getattr(cur, "instruction", "")
        if is_dynamic_instruction(existing):
            log.debug(
                "dynamic instruction already installed for %r — skipping",
                getattr(cur, "name", "<unnamed>"),
            )
            continue
        if callable(existing):
            log.debug(
                "agent %r has a user-supplied InstructionProvider "
                "(callable) — leaving it alone",
                getattr(cur, "name", "<unnamed>"),
            )
            continue
        original = existing if isinstance(existing, str) else ""
        if "{" in original and _adk_inject_session_state() is None:
            # ADK substitutes {var}/{artifact.var} for string
            # instructions only; a resolver bypasses that. Without the
            # inject helper the resolver cannot re-apply it, so leave
            # the string in place — native templating beats goldfive's
            # per-turn augmentation for this agent.
            log.warning(
                "goldfive.wrap: agent %r has a templated instruction "
                "but ADK's inject_session_state is unavailable — "
                "leaving the static instruction in place "
                "(dynamic instruction disabled for this agent)",
                getattr(cur, "name", "<unnamed>"),
            )
            continue

        agent_name = str(getattr(cur, "name", "") or "")
        # Wave B1 (refactor/prompt-shaper): the resolver factory lives
        # on :class:`~goldfive.prompt_shaper.PromptShaper` so the four
        # prompt-shape injection sites share one ``observation_only``
        # gate. The closure shape (provenance attrs, ADK
        # ``InstructionProvider`` signature) is preserved
        # byte-identically.
        from goldfive.prompt_shaper import PromptShaper

        resolver = PromptShaper().make_dynamic_instruction(
            original_instruction=original,
            agent_name=agent_name,
        )
        try:
            cur.instruction = resolver
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "could not install dynamic instruction for agent %r: %s",
                agent_name,
                exc,
            )
            continue
        log.debug("dynamic instruction installed for %r", agent_name)
        touched += 1

    return touched


def log_dynamic_instruction_opt_out(root_agent: Any) -> None:
    """Emit one INFO log per reachable ``LlmAgent`` when the caller opted out.

    Walks the same edges as :func:`install_dynamic_instructions` but
    does not mutate anything. Exists so the operator log shows which
    agents in the tree are running with static instructions — useful
    for debugging "why isn't refine landing in this agent's prompt?"
    after the fact.
    """
    if root_agent is None:
        return
    seen: set[int] = set()
    stack: list[Any] = [root_agent]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        for sub in getattr(cur, "sub_agents", None) or ():
            stack.append(sub)
        inner = getattr(cur, "inner_agent", None)
        if inner is not None:
            stack.append(inner)
        for tool in getattr(cur, "tools", None) or ():
            nested = getattr(tool, "agent", None)
            if nested is not None:
                stack.append(nested)

        if _looks_like_llm_agent(cur):
            log.info(
                "goldfive.wrap: dynamic_instruction OPT-OUT for %r",
                getattr(cur, "name", "<unnamed>"),
            )


__all__ = [
    "DEFAULT_AGENT_MAX_OUTPUT_TOKENS",
    "format_correction_block",
    "install_dynamic_instructions",
    "is_dynamic_instruction",
    "log_dynamic_instruction_opt_out",
    "pending_correction_key",
]
