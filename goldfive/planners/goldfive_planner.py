"""Goldfive's structural-steering ``BasePlanner`` (goldfive#153).

The :class:`GoldfivePlanner` is an ADK ``BasePlanner`` subclass that
attaches to every ``LlmAgent`` in a tree wrapped by :func:`goldfive.wrap`
and performs two structural jobs on every LLM call that agent makes:

1. **Request side.** Builds a short orchestration context block
   assembled from ``session.state['goldfive.*']`` keys (see
   :mod:`goldfive.adapters._adk_state_protocol`) and asks the agent
   to read the current plan task / goals / active user steer from it.
   The block is tree-agnostic — no presentation-layer or
   domain-specific vocabulary — so the same planner works for single
   LlmAgent, flat specialists, or deep hierarchies alike.

2. **Response side.** Filters the LLM's response parts for two
   structural signals:

   * ``function_call`` parts whose ``id`` appears in
     ``session.state['goldfive.cancelled_function_call_ids']`` are
     stripped (the LLM tried to retry a call goldfive already
     cancelled — e.g. on USER_STEER).
   * ``function_call`` parts to agents outside the adapter's
     registry emit a ``PLAN_DIVERGENCE`` drift via the steerer.
     **The call is not blocked** — this is a signal, not a gate.

Request-side injection requires a plugin workaround
-----------------------------------------------------

ADK's ``flows/llm_flows/_nl_planning.py`` gates request-side
instruction injection on ``isinstance(planner, PlanReActPlanner)``
(see ``_NlPlanningRequestProcessor.run_async``). Subclassing
``PlanReActPlanner`` would inherit its ReAct response filtering —
which constrains the agent's output to a tag-based shape we don't
want to impose. So :class:`GoldfivePlanner` subclasses ``BasePlanner``
directly, and :class:`~goldfive.adapters._adk_plugin._GoldfiveADKPlugin`'s
``before_model_callback`` takes over the injection: it detects when
the running agent carries a :class:`GoldfivePlanner`, calls
:meth:`build_planning_instruction`, and appends the result to
``llm_request.config.system_instruction`` via ``append_instructions``.

The response-side gate in ``_nl_planning.py`` is permissive — it
fires for any ``BasePlanner`` subclass other than ``BuiltInPlanner``
— so :meth:`process_planning_response` runs natively without a
workaround.

Compose-with-user-planner
-------------------------

If the user attaches their own ``BasePlanner`` to an agent before
``goldfive.wrap`` runs, the auto-attachment path preserves it by
passing it in as :attr:`user_planner`. On the request side goldfive
calls the user planner's ``build_planning_instruction`` first and
**prepends** that result ahead of goldfive's orchestration context
block (the user planner's framing lands above goldfive's, since the
user planner's context is typically more general — meta-instructions
— and goldfive's is per-turn ambient state). On the response side
goldfive's structural filters run first (strip cancelled calls;
signal divergence); then whatever parts remain are passed to the
user planner's ``process_planning_response`` so it can apply its
own transformations on top.

Tree-agnostic, no domain content
--------------------------------

Every placeholder in the orchestration block reads from session
state. The planner refuses to hard-code any presentation / research
/ coordinator / specialist vocabulary. Tree shape (flat, nested,
deep) is irrelevant — every agent in the tree receives the same
block format, with the state values differing only when the
goldfive driver has populated them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters._adk_state_protocol import (
    KEY_CURRENT_TASK_ID,
    KEY_CURRENT_TASK_TITLE,
)

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from google.adk.agents.callback_context import CallbackContext  # type: ignore
    from google.adk.agents.readonly_context import ReadonlyContext  # type: ignore
    from google.adk.models.llm_request import LlmRequest  # type: ignore
    from google.genai import types  # type: ignore


log = logging.getLogger("goldfive.planners.goldfive_planner")


# --- Additional state keys owned by this module --------------------------
#
# These keys are READ by GoldfivePlanner. They are populated by other
# layers (#152 state namespace expansion, the adapter/cancel path, the
# user-steer path) and default-safe — missing keys just render as
# ``(none)`` in the instruction block or are treated as empty sets in
# the response filters. Writing is scoped to the owning layer, not to
# this module.

KEY_GOALS_SUMMARY = "goldfive.goals_summary"
"""Comma-joined summary of ``session.goals`` maintained by the runner."""

KEY_ACTIVE_STEER_BODY = "goldfive.active_steer.body"
"""Text of the most recent user steering command, if any."""

KEY_CANCELLED_FUNCTION_CALL_IDS = "goldfive.cancelled_function_call_ids"
"""List of ADK ``function_call`` ids the adapter cancelled mid-invoke.

Written by the USER_STEER / REPLAN cancel paths and consumed by
:meth:`GoldfivePlanner.process_planning_response` to strip any
retry attempts the LLM emits for the same ids on the next turn.
"""


# Sentinel rendered into the orchestration block when a state key is
# missing or empty. Kept intentionally short so the block stays compact
# when the driver has not yet populated state.
_NONE_MARKER = "(none)"


def _state_get(state: Any, key: str, default: str = "") -> str:
    """Tolerant mapping read for ``state`` — returns ``default`` on any failure."""
    if not isinstance(state, Mapping):
        return default
    try:
        value = state.get(key, default)
    except Exception:  # noqa: BLE001 -- best-effort state access
        return default
    if value is None:
        return default
    return str(value)


def _state_list(state: Any, key: str) -> list[str]:
    """Tolerant list-of-strings read for ``state``."""
    if not isinstance(state, Mapping):
        return []
    try:
        value = state.get(key, None)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str) and v:
            out.append(v)
    return out


# Lazily import ``BasePlanner`` so importing this module without ADK
# installed fails with the same "requires 'pip install goldfive[adk]'"
# shape the rest of the adapters use.
try:
    from google.adk.planners.base_planner import BasePlanner  # type: ignore
except ImportError as exc:  # pragma: no cover — covered via importorskip in tests
    raise ImportError(
        "goldfive.planners.goldfive_planner requires 'pip install goldfive[adk]'"
    ) from exc


class GoldfivePlanner(BasePlanner):
    """Structural ``BasePlanner`` auto-attached by :func:`goldfive.wrap`.

    Parameters
    ----------
    user_planner:
        Optional pre-existing ``BasePlanner`` the user already attached
        to an agent. When provided :class:`GoldfivePlanner` composes
        with it:

        * **Request side** — calls
          ``user_planner.build_planning_instruction(...)`` first; the
          result is **prepended** ahead of goldfive's orchestration
          context block in the returned string.
        * **Response side** — goldfive's structural filters (strip
          cancelled ``function_call`` ids, signal PLAN_DIVERGENCE on
          off-registry agent calls) run first; the remaining parts
          are passed to ``user_planner.process_planning_response(...)``
          so the user planner can apply its own transformations on
          top.
    agent_registry:
        Optional iterable of agent names that are considered "on the
        registry" — function_calls to agents whose name is not in
        this set emit a ``PLAN_DIVERGENCE`` drift when
        :meth:`process_planning_response` runs. When ``None`` the
        divergence check is disabled (no registry → nothing to
        diverge from). The adapter's auto-attachment pass fills this
        in with :attr:`~goldfive.adapters.adk.ADKAdapter.available_agents`.
    steerer:
        Optional :class:`~goldfive.protocols.Steerer` used to emit the
        PLAN_DIVERGENCE drift. When ``None`` the divergence-signal
        path is silent (but the off-registry count is still tracked
        via log.debug).
    session:
        Optional :class:`~goldfive.types.Session` passed to
        ``steerer._handle_drift``. Same binding contract as
        ``steerer`` — set by the adapter when it auto-attaches.
    """

    def __init__(
        self,
        *,
        user_planner: BasePlanner | None = None,
        agent_registry: Iterable[str] | None = None,
        steerer: Any = None,
        session: Any = None,
    ) -> None:
        # ``BasePlanner`` is an ABC with no ``__init__`` state of its own
        # beyond the abstract method contract; invoking super() is still
        # correct for the MRO but we have no base fields to initialise.
        super().__init__()
        self._user_planner = user_planner
        self._agent_registry: set[str] | None = (
            {str(n) for n in agent_registry if isinstance(n, str) and n}
            if agent_registry is not None
            else None
        )
        self._steerer = steerer
        self._session = session

    # ------------------------------------------------------------------
    # Public rebinding — called by the adapter once it knows the
    # registry / steerer / session for the current invocation.
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        agent_registry: Iterable[str] | None = None,
        steerer: Any = None,
        session: Any = None,
    ) -> None:
        """Rebind the per-invocation collaborators.

        Called by :class:`~goldfive.adapters.adk.ADKAdapter` after
        :func:`goldfive.wrap` attaches the planner to each agent but
        before ``runner.run_async`` fires. Any argument set to a
        non-``None`` value replaces the prior binding; ``None`` leaves
        the existing value intact so partial rebinding works.
        """
        if agent_registry is not None:
            self._agent_registry = {str(n) for n in agent_registry if isinstance(n, str) and n}
        if steerer is not None:
            self._steerer = steerer
        if session is not None:
            self._session = session

    @property
    def user_planner(self) -> BasePlanner | None:
        """The composed user-supplied planner, if any (see class docstring)."""
        return self._user_planner

    # ------------------------------------------------------------------
    # BasePlanner: build_planning_instruction
    # ------------------------------------------------------------------

    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> str | None:
        """Return the orchestration context block for this LLM turn.

        Reads ``session.state['goldfive.*']`` off ``readonly_context``
        and emits a short, tree-agnostic block. When a
        :attr:`user_planner` is set its own ``build_planning_instruction``
        is called first and **prepended** so the user planner's framing
        lands above goldfive's per-turn state block.

        Returns ``None`` only when an internal error prevents building
        any instruction — never returns an empty string (empty would
        cause ADK to skip the append, hiding the bug).
        """
        state = _extract_state(readonly_context)

        # Collect the placeholders off state. All keys default to the
        # ``(none)`` marker so the block still renders when state is
        # sparse — e.g. before #152 populates the new keys — making
        # this module a soft dependency.
        task_id = _state_get(state, KEY_CURRENT_TASK_ID) or _NONE_MARKER
        task_title = _state_get(state, KEY_CURRENT_TASK_TITLE) or _NONE_MARKER
        goals_summary = _state_get(state, KEY_GOALS_SUMMARY) or _NONE_MARKER
        steer_body = _state_get(state, KEY_ACTIVE_STEER_BODY) or _NONE_MARKER

        goldfive_block = (
            "[GOLDFIVE ORCHESTRATION CONTEXT]\n"
            "\n"
            f"Plan task (if any): {task_id}\n"
            f"  title: {task_title}\n"
            f"Current goals: {goals_summary}\n"
            f"Active user steer (if any): {steer_body}\n"
            "\n"
            "Notes:\n"
            "- The steering direction above supersedes prior context "
            "unless your response explicitly references prior work.\n"
            "- Do not retry cancelled tool calls (their ids are in "
            "state['goldfive.cancelled_function_call_ids'])."
        )

        # Compose with any user-supplied planner. Prepend the user
        # planner's result: user-authored meta-instructions typically
        # describe "how to think", goldfive's block is per-turn ambient
        # state, and LLMs parse a two-section instruction more
        # reliably when the more-general section comes first.
        if self._user_planner is not None:
            try:
                user_out = self._user_planner.build_planning_instruction(
                    readonly_context, llm_request
                )
            except Exception as exc:  # noqa: BLE001 — user code; don't propagate
                log.debug(
                    "GoldfivePlanner: user_planner.build_planning_instruction raised: %s",
                    exc,
                )
                user_out = None
            if user_out:
                return f"{user_out}\n\n{goldfive_block}"

        return goldfive_block

    # ------------------------------------------------------------------
    # BasePlanner: process_planning_response
    # ------------------------------------------------------------------

    def process_planning_response(
        self,
        callback_context: CallbackContext,
        response_parts: list[types.Part],
    ) -> list[types.Part] | None:
        """Apply structural filters to the LLM's response parts.

        Two independent filters run in order:

        1. Strip ``function_call`` parts whose ``function_call.id`` is
           in ``session.state['goldfive.cancelled_function_call_ids']``.
           Keeps the rest of the parts intact.
        2. For every retained ``function_call`` whose ``name`` is not
           in :attr:`_agent_registry` (when the registry is set),
           emit a ``PLAN_DIVERGENCE`` drift at INFO severity via
           ``steerer._handle_drift``. The call is NOT blocked; this
           is signal-only so the steerer can decide policy.

        After goldfive's filters run, if a :attr:`user_planner` is set
        its own ``process_planning_response`` is called on the
        remaining parts so it can layer its own transformations.
        """
        state = _extract_state(callback_context)
        cancelled_ids = {s for s in _state_list(state, KEY_CANCELLED_FUNCTION_CALL_IDS)}

        # Filter 1: strip cancelled function_call ids.
        kept: list[Any] = []
        stripped_count = 0
        for part in response_parts or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and cancelled_ids:
                fc_id = getattr(fc, "id", None)
                if fc_id and str(fc_id) in cancelled_ids:
                    stripped_count += 1
                    continue
            kept.append(part)

        if stripped_count:
            log.debug(
                "GoldfivePlanner: stripped %d cancelled function_call part(s)",
                stripped_count,
            )

        # Filter 2: signal PLAN_DIVERGENCE on function_calls to agents
        # outside the registry. We emit best-effort and never block;
        # the steerer decides whether to escalate.
        if self._agent_registry is not None and self._steerer is not None:
            off_registry: list[str] = []
            for part in kept:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                name = getattr(fc, "name", "") or ""
                if not name:
                    continue
                # Only call names that LOOK like agent delegation
                # targets are checked — goldfive cannot distinguish
                # an AgentTool name from any other tool name purely
                # from the Part, so we rely on the registry being
                # authoritative: if the tool name is in
                # _agent_registry it's on-plan; if not, it MAY be a
                # regular tool (fine) OR an off-registry agent call
                # (divergence). We emit on not-in-registry only when
                # the name matches the delegation shape we care about
                # — today that's "not a known agent and not prefixed
                # with the reporting-tool 'report_' namespace".
                if name in self._agent_registry:
                    continue
                if name.startswith("report_"):
                    # Reporting tools (report_task_started, etc.) are
                    # protocol calls, not agent delegation. Skip.
                    continue
                # Heuristic: genuine ADK tool names usually include
                # verbs (web_search, read_file) and rarely coincide
                # with agent names. We keep the signal permissive —
                # over-reporting once is fine; the steerer's INFO
                # severity is absorbable.
                off_registry.append(name)

            for name in off_registry:
                self._emit_divergence_signal(name, callback_context)

        # Compose with user planner: their filter runs AFTER goldfive's
        # so they see the structurally-clean parts.
        if self._user_planner is not None:
            try:
                user_out = self._user_planner.process_planning_response(callback_context, kept)
            except Exception as exc:  # noqa: BLE001 — user code; don't propagate
                log.debug(
                    "GoldfivePlanner: user_planner.process_planning_response raised: %s",
                    exc,
                )
                user_out = None
            if user_out is not None:
                return list(user_out)

        # Return ``kept`` when we modified the list; return ``None`` —
        # ADK's "leave response untouched" signal — when we kept
        # everything AND did not emit any divergence signals. The
        # latter gate keeps this a no-op in the common case where no
        # cancelled-ids / off-registry calls exist.
        if stripped_count or (
            self._agent_registry is not None
            and self._steerer is not None
            and any(getattr(p, "function_call", None) is not None for p in kept)
        ):
            return kept
        return None

    # ------------------------------------------------------------------
    # Drift emission — plumbed through steerer._handle_drift to hit
    # the same pipeline PlanReconciler / RUNAWAY_DELEGATION use.
    # ------------------------------------------------------------------

    def _emit_divergence_signal(self, off_registry_name: str, callback_context: Any) -> None:
        """Emit a ``PLAN_DIVERGENCE`` drift for an off-registry function_call.

        Non-blocking: the call still reaches ADK's dispatcher and is
        executed (or fails naturally if the tool name is unknown). We
        only SIGNAL the divergence so the steerer's policy can decide
        whether to escalate via the intervention ladder (#142) or let
        it ride.

        Routed through ``steerer._handle_drift`` (pre-built
        :class:`DriftEvent` path) matching the reconciler's
        convention — :meth:`observe` would round-trip the event back
        through classification and drop it.
        """
        steerer = self._steerer
        if steerer is None:
            return
        try:
            from goldfive.types import (  # noqa: PLC0415 — lazy
                DriftEvent,
                DriftKind,
                DriftSeverity,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "GoldfivePlanner._emit_divergence_signal: cannot import DriftEvent: %s",
                exc,
            )
            return

        # Prefer state-derived current_task_id; fall back to empty.
        state = _extract_state(callback_context)
        current_task_id = _state_get(state, KEY_CURRENT_TASK_ID, "")

        drift = DriftEvent(
            kind=DriftKind.PLAN_DIVERGENCE,
            severity=DriftSeverity.INFO,
            detail=(
                f"LLM emitted function_call to off-registry agent "
                f"{off_registry_name!r} — call not blocked, signal only"
            ),
            current_task_id=current_task_id,
            current_agent_id=off_registry_name,
        )
        handle = getattr(steerer, "_handle_drift", None)
        if not callable(handle):
            log.debug("GoldfivePlanner: steerer has no _handle_drift; divergence signal dropped")
            return

        session = self._session
        # The steerer handler is async; since
        # ``process_planning_response`` is SYNC by ADK's contract, we
        # schedule on the running event loop. When no loop is running
        # (hypothetical tests calling the method synchronously) fall
        # through silently — the divergence signal is best-effort.
        try:
            import asyncio  # noqa: PLC0415 — lazy

            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug(
                "GoldfivePlanner: no running loop; divergence signal for %s dropped",
                off_registry_name,
            )
            return

        coro = handle(drift, session)
        # Create a task; do not await (we are sync). The loop will
        # drive the coroutine to completion; we attach a done-callback
        # that swallows exceptions to keep the signal fire-and-forget.
        task = loop.create_task(coro)

        def _swallow(t: Any) -> None:
            try:
                t.result()
            except Exception as exc:  # noqa: BLE001
                log.debug("GoldfivePlanner: _handle_drift task raised: %s", exc)

        task.add_done_callback(_swallow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_state(ctx: Any) -> Any:
    """Return the state mapping from a readonly or callback context.

    ADK's :class:`ReadonlyContext` exposes ``.state`` as a
    MappingProxyType; :class:`CallbackContext` exposes a ``State``
    object (which also supports ``.get`` / ``__getitem__``). In both
    cases treating the object as a Mapping works for our read-only
    needs. For robustness against test stubs we also try
    ``.session.state`` and ``._invocation_context.session.state``.
    """
    for attr_chain in (
        ("state",),
        ("session", "state"),
        ("_invocation_context", "session", "state"),
        ("invocation_context", "session", "state"),
    ):
        cur: Any = ctx
        ok = True
        for part in attr_chain:
            try:
                cur = getattr(cur, part, None)
            except Exception:  # noqa: BLE001
                cur = None
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return {}


__all__ = [
    "KEY_ACTIVE_STEER_BODY",
    "KEY_CANCELLED_FUNCTION_CALL_IDS",
    "KEY_GOALS_SUMMARY",
    "GoldfivePlanner",
]
