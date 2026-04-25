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
   * ``function_call`` parts are classified via a three-stage gate
     (see :meth:`GoldfivePlanner.process_planning_response`):

     1. **Own tool** — name is in the currently-running agent's
        ``tools`` list. Legitimate; no drift.
     2. **Cross-layer agent** — name is in the tree's agent registry
        but wasn't exposed to this agent. LLM attempted delegation
        past its layer → ``PLAN_DIVERGENCE`` (WARNING).
     3. **Nowhere** — name is neither a tool nor a known agent. Pure
        hallucination → ``CONFABULATION_RISK`` (WARNING).

     **The call is not blocked** in any case — this is a signal, not
     a gate.

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

KEY_ACTIVE_STEER_SOURCE = "goldfive.active_steer.source"
"""Source attribution for the active steer (``"user"`` / ``"goldfive"`` / empty).

Written by :class:`~goldfive.steerer.DefaultSteerer`. Consulted by
:meth:`GoldfivePlanner.build_planning_instruction` to render the
active-steer line with explicit attribution — operators can tell a
user-authored steer from a goldfive drift-promoted steer at a glance.
"""

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


def _goldfive_session_from_context(ctx: Any) -> Any:
    """Return the goldfive ``Session`` reachable from a callback / readonly context.

    Phase 2.0 of goldfive#271. Resolution order:

    1. Walk ``ctx._invocation_context.plugin_manager.plugins`` for the
       goldfive plugin and read its ``_active_ctx.session``. This is
       the live-run path — set by
       :meth:`~goldfive.adapters._adk_plugin._GoldfiveADKPlugin.set_active_context`
       (called from :meth:`ADKAdapter._invoke_internal` before
       ``runner.run_async``).
    2. The legacy ``"goldfive._session_context"`` stash on the
       context's ``state`` — used only by unit tests that drive the
       planner against a hand-built state dict.

    Returns ``None`` when neither resolves.
    """
    from goldfive.adapters._adk_plugin import session_context_from_invocation

    inv_ctx = getattr(ctx, "_invocation_context", None) or getattr(
        ctx, "invocation_context", None
    )
    if inv_ctx is not None:
        sc = session_context_from_invocation(inv_ctx)
        if sc is not None:
            session = getattr(sc, "session", None)
            if session is not None:
                return session
    state = getattr(ctx, "state", None)
    if not isinstance(state, Mapping):
        return None
    legacy_ctx = state.get("goldfive._session_context")
    if legacy_ctx is None:
        return None
    return getattr(legacy_ctx, "session", None)


def _task_title_from_plan(session: Any, task_id: str) -> str:
    """Look up ``task_id``'s title in ``session.plan.tasks``.

    The typed :class:`~goldfive.types.Task` on ``Session.plan`` is the
    source of truth — the planner reaches into it for the prompt block
    instead of reading the de-normalised
    ``goldfive.current_task_title`` key. Returns ``""`` when the plan
    or the task is missing.
    """
    if session is None or not task_id:
        return ""
    plan = getattr(session, "plan", None)
    if plan is None:
        return ""
    tasks = getattr(plan, "tasks", None) or ()
    for task in tasks:
        if str(getattr(task, "id", "") or "") == task_id:
            return str(getattr(task, "title", "") or "")
    return ""


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

        Reads from goldfive's
        :class:`~goldfive.orchestration_store.OrchestrationStore` when
        the planner can reach the goldfive
        :class:`~goldfive.types.Session` via the
        ``goldfive._session_context`` stash on ADK state. Falls back
        to reading ADK state directly only for custom adapters that
        don't stash a SessionContext / legacy unit tests.

        When a :attr:`user_planner` is set its own
        ``build_planning_instruction`` is called first and **prepended**
        so the user planner's framing lands above goldfive's per-turn
        state block.

        Phase 2.0 of goldfive#271 — the bridge from goldfive
        ``Session.state`` onto ADK ``session.state`` is gone. The
        planner reads goldfive Session directly, eliminating the
        callback-time write to ADK state that raced with ADK's
        optimistic-concurrency contract (see goldfive#275).

        Returns ``None`` only when an internal error prevents building
        any instruction — never returns an empty string (empty would
        cause ADK to skip the append, hiding the bug).
        """
        state = _extract_state(readonly_context)
        gf_session = _goldfive_session_from_context(readonly_context)

        if gf_session is not None:
            from goldfive.orchestration_store import OrchestrationStore

            store = OrchestrationStore.for_session(gf_session)
            pin_id = store.pin_current_task()
            task_id = pin_id or _NONE_MARKER
            # Title comes from the typed ``Session.plan.tasks`` lookup
            # — V3's pin write only stamps id+revision on goldfive
            # ``Session.state``, and the typed Task is the authoritative
            # source. Falls back to the de-normalised
            # ``goldfive.current_task_title`` key if a custom path
            # populated it.
            task_title = (
                _task_title_from_plan(gf_session, pin_id)
                or store.pin_current_task_title()
                or _NONE_MARKER
            )
            active = store.get_active_steer()
            if active is not None:
                steer_body_raw = active.body
                steer_source_raw = active.source.strip().lower()
            else:
                steer_body_raw = ""
                steer_source_raw = ""
            goals_summary = store.goals_summary() or _NONE_MARKER
        else:
            # Legacy ADK-state fallback: tests that drive the planner
            # against a plain state dict without the SessionContext
            # stash. Production paths always carry the stash.
            task_id = _state_get(state, KEY_CURRENT_TASK_ID) or _NONE_MARKER
            task_title = _state_get(state, KEY_CURRENT_TASK_TITLE) or _NONE_MARKER
            steer_body_raw = _state_get(state, KEY_ACTIVE_STEER_BODY)
            steer_source_raw = _state_get(state, KEY_ACTIVE_STEER_SOURCE).strip().lower()
            goals_summary = _state_get(state, KEY_GOALS_SUMMARY) or _NONE_MARKER

        # goldfive-steer-unification: source-aware attribution line so
        # the LLM sees whether the active steer is an operator
        # directive or a goldfive-detected drift correction. The label
        # matches the framing used by
        # :meth:`SequentialExecutor._compose_steer_restart_message`.
        if not steer_body_raw:
            steer_line = f"Active steer (if any): {_NONE_MARKER}"
        elif steer_source_raw == "goldfive":
            steer_line = f"Active steer (goldfive): {steer_body_raw}"
        elif steer_source_raw == "user":
            steer_line = f"Active steer (user): {steer_body_raw}"
        else:
            # Empty / unknown source — treat as user-authored for
            # back-compat (pre-unification stamps had no source key).
            steer_line = f"Active steer (user): {steer_body_raw}"

        goldfive_block = (
            "[GOLDFIVE ORCHESTRATION CONTEXT]\n"
            "\n"
            f"Plan task (if any): {task_id}\n"
            f"  title: {task_title}\n"
            f"Current goals: {goals_summary}\n"
            f"{steer_line}\n"
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

        Two independent concerns run in order:

        1. **Cancelled-id filter.** Strip ``function_call`` parts
           whose ``function_call.id`` is in
           ``session.state['goldfive.cancelled_function_call_ids']``.
           Keeps the rest of the parts intact. This runs on EVERY part
           regardless of which drift-stage its name falls in.
        2. **Three-stage drift classification** (for each retained
           ``function_call`` part):

           - *Stage 1 — own tool.* Name appears in the currently-
             running agent's ``tools`` list (read from
             ``callback_context._invocation_context.agent.tools``).
             Legitimate; no drift.
           - *Stage 2 — cross-layer agent.* Name matches an agent in
             :attr:`_agent_registry` but is not in the running agent's
             tools. The LLM attempted to delegate past its exposed
             layer. Emit :data:`DriftKind.PLAN_DIVERGENCE` at WARNING.
           - *Stage 3 — nowhere.* Name is neither a tool nor a known
             agent. Emit :data:`DriftKind.CONFABULATION_RISK` at
             WARNING.

           Calls are NEVER blocked — this is signal-only; the steerer
           decides whether to escalate.

        After goldfive's filters run, if a :attr:`user_planner` is set
        its own ``process_planning_response`` is called on the
        remaining parts so it can layer its own transformations.
        """
        state = _extract_state(callback_context)
        gf_session = _goldfive_session_from_context(callback_context)
        if gf_session is not None:
            from goldfive.orchestration_store import OrchestrationStore

            cancelled_ids = set(
                OrchestrationStore.for_session(gf_session).cancelled_function_call_ids()
            )
        else:
            # Legacy ADK-state fallback: tests that drive the planner
            # against a plain state dict without the SessionContext
            # stash.
            cancelled_ids = {s for s in _state_list(state, KEY_CANCELLED_FUNCTION_CALL_IDS)}

        # Filter 1: strip cancelled function_call ids. Runs on every
        # part regardless of which drift-stage its name would fall in.
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

        # Filter 2: three-stage tool-call drift classification. We
        # need the currently-running agent's own tool set to
        # distinguish legitimate tool calls (stage 1) from cross-layer
        # delegation (stage 2) and hallucination (stage 3).
        own_tool_names = _extract_own_tool_names(callback_context)
        divergence_fired = False
        if self._steerer is not None:
            for part in kept:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                name = getattr(fc, "name", "") or ""
                if not name:
                    continue

                # Stage 1 — agent's own tool. Always legitimate.
                if name in own_tool_names:
                    continue
                # Reporting tools (report_task_started, etc.) are
                # protocol calls. They may be injected onto the agent
                # tree post-construction; treat the ``report_`` prefix
                # as an always-legitimate protocol namespace even if
                # the tool list didn't reflect it.
                if name.startswith("report_"):
                    continue

                # Stage 2 — name matches an agent in the registry but
                # wasn't exposed to this agent. Cross-layer delegation.
                if self._agent_registry is not None and name in self._agent_registry:
                    self._emit_tool_call_drift(
                        name,
                        callback_context,
                        kind_name="PLAN_DIVERGENCE",
                        detail=(
                            f"LLM emitted function_call to agent "
                            f"{name!r} which is in the tree registry "
                            f"but was not exposed as a tool to the "
                            f"currently-running agent — cross-layer "
                            f"delegation attempt, call not blocked"
                        ),
                    )
                    divergence_fired = True
                    continue

                # Stage 3 — name is neither a tool nor a known agent.
                # Pure hallucination.
                self._emit_tool_call_drift(
                    name,
                    callback_context,
                    kind_name="CONFABULATION_RISK",
                    detail=(
                        f"LLM emitted function_call to {name!r} which "
                        f"is neither one of the current agent's tools "
                        f"nor a known agent in the tree registry — "
                        f"hallucinated tool, call not blocked"
                    ),
                )
                divergence_fired = True

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
        # cancelled-ids / divergence classifications fired.
        if stripped_count or divergence_fired:
            return kept
        return None

    # ------------------------------------------------------------------
    # Drift emission — plumbed through steerer._handle_drift to hit
    # the same pipeline PlanReconciler / RUNAWAY_DELEGATION use.
    # ------------------------------------------------------------------

    def _emit_tool_call_drift(
        self,
        function_call_name: str,
        callback_context: Any,
        *,
        kind_name: str,
        detail: str,
    ) -> None:
        """Emit a structural tool-call drift via ``steerer._handle_drift``.

        ``kind_name`` selects which :class:`DriftKind` member to fire
        — one of ``"PLAN_DIVERGENCE"`` (cross-layer delegation) or
        ``"CONFABULATION_RISK"`` (hallucinated tool) per the three-
        stage classification in :meth:`process_planning_response`.

        Non-blocking: the call still reaches ADK's dispatcher and is
        executed (or fails naturally if the tool name is unknown). We
        only SIGNAL the drift so the steerer's policy can decide
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
                "GoldfivePlanner._emit_tool_call_drift: cannot import DriftEvent: %s",
                exc,
            )
            return

        kind = getattr(DriftKind, kind_name, None)
        if kind is None:
            log.debug(
                "GoldfivePlanner._emit_tool_call_drift: unknown DriftKind %r; signal dropped",
                kind_name,
            )
            return

        # Prefer state-derived current_task_id; fall back to empty.
        state = _extract_state(callback_context)
        current_task_id = _state_get(state, KEY_CURRENT_TASK_ID, "")

        drift = DriftEvent(
            kind=kind,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=current_task_id,
            current_agent_id=function_call_name,
        )
        handle = getattr(steerer, "_handle_drift", None)
        if not callable(handle):
            log.debug(
                "GoldfivePlanner: steerer has no _handle_drift; %s signal dropped",
                kind_name,
            )
            return

        session = self._session
        # The steerer handler is async; since
        # ``process_planning_response`` is SYNC by ADK's contract, we
        # schedule on the running event loop. When no loop is running
        # (hypothetical tests calling the method synchronously) fall
        # through silently — the drift signal is best-effort.
        try:
            import asyncio  # noqa: PLC0415 — lazy

            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug(
                "GoldfivePlanner: no running loop; %s signal for %s dropped",
                kind_name,
                function_call_name,
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


def _extract_own_tool_names(callback_context: Any) -> set[str]:
    """Return the set of tool names exposed to the currently-running agent.

    Reads ``callback_context._invocation_context.agent.tools`` and
    collects each tool's ``.name`` attribute (falling back to the
    underlying function's ``__name__`` for ``FunctionTool`` wrappers
    that don't carry an explicit ``name``). Tools that expose neither
    are skipped silently.

    Returns an empty set when the attribute chain is missing (e.g.
    test stubs that don't plumb an invocation_context) — the three-
    stage classifier treats every function_call as either stage 2 or
    stage 3 in that case, which is the safe fallback (we err toward
    reporting rather than suppressing).
    """
    # Prefer the standard chain CallbackContext exposes in ADK:
    # ``_invocation_context.agent`` → the agent whose LLM turn we're
    # processing.
    agent = None
    for attr_chain in (
        ("_invocation_context", "agent"),
        ("invocation_context", "agent"),
        ("agent",),
    ):
        cur: Any = callback_context
        ok = True
        for part in attr_chain:
            try:
                cur = getattr(cur, part, None)
            except Exception:  # noqa: BLE001 — best-effort
                cur = None
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            agent = cur
            break
    if agent is None:
        return set()

    tools = getattr(agent, "tools", None)
    if not tools:
        return set()

    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name:
            # FunctionTool sometimes carries its name on the
            # wrapped function rather than on the tool object.
            func = getattr(tool, "func", None)
            name = getattr(func, "__name__", None) if func is not None else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


__all__ = [
    "KEY_ACTIVE_STEER_BODY",
    "KEY_ACTIVE_STEER_SOURCE",
    "KEY_CANCELLED_FUNCTION_CALL_IDS",
    "KEY_GOALS_SUMMARY",
    "GoldfivePlanner",
]
