"""Dynamic instruction resolver for goldfive-wrapped ADK ``LlmAgent``\\s.

Goldfive#251 — plan-causal prompting.

Each wrapped ``LlmAgent``'s ``instruction`` field is bound at
``LlmAgent(...)`` construction time. When goldfive's plan changes
mid-run (refine landing, task supersedes, correction injection), the
plan updates in ``session.state`` but the agent's baked-in prompt does
not — so the LLM keeps executing its original instruction and only
"observes" the plan shift through unrelated channels.

This module fixes that by replacing each wrapped ``LlmAgent``'s static
``instruction`` string with a callable ``(ReadonlyContext) -> str`` that
resolves the agent's current task from ``session.state`` at every turn.
ADK's ``canonical_instruction`` invokes the callable per turn and
returns ``bypass_state_injection=True`` for the result, so refine
landing in state is picked up on the NEXT turn with no transcript
rewrite.

The resolver is **agent-agnostic**: works for any wrapped ``LlmAgent``
(coordinator, sub-agent, root-with-tools, leaf). The current-task
resolution is driven entirely by whatever is pinned in state for this
specific agent invocation via
:data:`goldfive.adapters._adk_state_protocol.KEY_CURRENT_TASK_ID`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from goldfive.adapters import _adk_state_protocol as _sp

log = logging.getLogger("goldfive.adapters._adk_dynainst")


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
) -> str:
    """Assemble the final prompt string the LLM sees this turn.

    Shape:

        {original}

        ---

        Current assigned task:
          id: {task_id}
          title: {task_title}
          description: {task_description}

        {pending_correction}  # appended only when non-empty
    """
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


def make_dynamic_instruction(
    original_instruction: str,
    agent_name: str,
) -> Callable[[Any], str]:
    """Return a resolver matching ADK's ``InstructionProvider`` signature.

    The returned callable:

    * Reads ``session.state`` off the ``ReadonlyContext``.
    * If no current-task id is pinned for this agent, returns the
      ``original_instruction`` verbatim (pre-plan turns stay unchanged).
    * Otherwise composes the original + a current-task block.
    * If a pending correction exists for ``(agent_name, current_task_id)``
      the block is appended (Stream D will write; we just read).

    The resolver is pure: given the same state it returns the same
    string. No side effects, no persistence.
    """

    def resolver(readonly_ctx: Any) -> str:
        try:
            state = _state_from_readonly_context(readonly_ctx)

            current_task_id = str(state.get(_sp.KEY_CURRENT_TASK_ID, "") or "")
            if not current_task_id:
                # No pin — pre-plan turn, or an agent that doesn't need
                # plan-causal augmentation this turn. Return the caller's
                # instruction verbatim.
                return original_instruction

            current_task_title = str(
                state.get(_sp.KEY_CURRENT_TASK_TITLE, "") or ""
            )
            current_task_description = str(
                state.get(_sp.KEY_CURRENT_TASK_DESCRIPTION, "") or ""
            )

            pending_correction = _resolve_pending_correction(
                state.get(
                    pending_correction_key(agent_name, current_task_id),
                )
            )

            return _compose_instruction(
                original=original_instruction,
                task_id=current_task_id,
                task_title=current_task_title,
                task_description=current_task_description,
                pending_correction=pending_correction,
            )
        except Exception as exc:  # noqa: BLE001
            # Instrumentation path — any failure here degrades to the
            # original instruction so the agent still runs. ADK's own
            # pipeline would otherwise surface this as an InternalError
            # mid-turn, which is the worst possible failure mode.
            log.debug(
                "dynamic_instruction resolver raised for agent=%r: %s "
                "(falling back to original instruction)",
                agent_name,
                exc,
            )
            return original_instruction

    # Stamp provenance on the closure so test code and tree-walk
    # idempotency checks can recognise it without relying on repr.
    resolver._goldfive_dynamic_instruction = True  # type: ignore[attr-defined]
    resolver._goldfive_agent_name = agent_name  # type: ignore[attr-defined]
    resolver._goldfive_original_instruction = original_instruction  # type: ignore[attr-defined]
    return resolver


def is_dynamic_instruction(value: Any) -> bool:
    """Return True when ``value`` is a resolver produced by this module."""
    return bool(getattr(value, "_goldfive_dynamic_instruction", False))


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

        agent_name = str(getattr(cur, "name", "") or "")
        resolver = make_dynamic_instruction(
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
    "format_correction_block",
    "install_dynamic_instructions",
    "is_dynamic_instruction",
    "log_dynamic_instruction_opt_out",
    "make_dynamic_instruction",
    "pending_correction_key",
]
