"""Centralised gate + injection sites for goldfive prompt-shaping.

Goldfive's runtime augments the prompt the underlying LLM sees in four
distinct places. Each site is a different mechanism with a different
caller — but all four share the same enforcement contract:
:class:`~goldfive.config.SteeringConfig.observation_only` must suppress
*every* prompt-shape injection so the operator observes the RAW
caller-supplied prompt the coordinator was constructed with.

Before this module the gate logic was hand-rolled at each of the four
sites: a ``_observation_only_active`` helper in
:mod:`goldfive.adapters._adk_plugin`, a near-identical one in
:mod:`goldfive.adapters.adk_llm_instrumentation` (then
``_adk_dynainst``), and an inline check in
:meth:`goldfive.runner.Runner.run`. The duplication invited drift — a
future site added without remembering the gate would silently leak the
shaping under ``observation_only=True``. :class:`PromptShaper`
collapses the four gates into a single :meth:`should_inject` predicate
and owns the four injection bodies.

Sites
-----

1. **Conversational follow-up wrap** —
   :meth:`PromptShaper.wrap_conversational_input`. Frames the user's
   raw input as a "[CONVERSATIONAL FOLLOW-UP — reuse prior plan]"
   directive (closes goldfive#277). Used by
   :meth:`goldfive.runner.Runner.run` on a conversational follow-up
   turn (``handle_turn`` returned ``None`` on a turn with a real prior
   plan).

2. **GoldfivePlanner request-side instruction** —
   :meth:`PromptShaper.inject_goldfive_planner_instruction`. Appends
   the planner's :meth:`build_planning_instruction` output to
   ``llm_request.config.system_instruction`` (workaround for ADK's
   ``_nl_planning`` request-side gating on ``PlanReActPlanner``;
   goldfive#153).

3. **Runtime tool-surface hint** —
   :meth:`PromptShaper.inject_runtime_tools_hint`. Appends a
   ``[GOLDFIVE PLAN-STATE HINT — …]`` block listing each agent's
   PENDING / DONE task summary so the coordinator has structural
   guidance about which sub-agent to call next (R3 / F2-alternative).
   AGENCY-PRESERVATION.md PR 9 prompt-shaping diet: RETIRED under the
   ``signal_channel == "request_context"`` regime (its per-turn
   footprint violates dormancy §1.1); the plan-state content folds into
   the observer note's Status surface (Site 5). Legacy regime unchanged.

4. **Dynamic instruction resolver** —
   :meth:`PromptShaper.make_dynamic_instruction`. Returns an ADK
   ``InstructionProvider`` callable that resolves the current pinned
   task + pending-correction block on every turn and appends them to
   the agent's ``original_instruction`` (goldfive#251).
   AGENCY-PRESERVATION.md PR 9: the ``[CURRENT ASSIGNED TASK]`` pin is
   RETIRED by default under ``request_context`` (the wrapped agent owns
   its decomposition / ordering); ``SteeringConfig.pin_assigned_task``
   re-enables it for trees built around the pinned block. Legacy regime
   unchanged.

5. **Observer-note channel** —
   :meth:`PromptShaper.inject_observer_note` (AGENCY-PRESERVATION.md
   PR 6). The one remaining goldfive injection surface in the
   ``request_context`` regime (alongside Site 2's planner contract):
   renders the most-severe pending observer note onto the request,
   carrying the folded plan-state line (Site 3).

Each method first asks :meth:`should_inject` whether the gate is open
for its context. When :meth:`should_inject` returns ``False`` the
method is a no-op (no-op = "return the raw / pre-existing value" for
each site's contract).

Note on shape: the four sites have intentionally different signatures
because the underlying injection mechanisms are different (message-
body rewrite, ``system_instruction`` append, callable closure). They
are NOT collapsed into one ``inject(...)`` method.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from goldfive.steerer import signal_channel as _signal_channel_of
from goldfive.steerer import steering_is_active

if TYPE_CHECKING:
    from goldfive.types import Session


log = logging.getLogger("goldfive.prompt_shaper")


class PromptShaper:
    """Stateless namespace for goldfive's prompt-shaping injection sites.

    The class holds no instance state — methods are organised on a class
    so call sites read naturally (``PromptShaper().wrap_conversational_input(...)``
    or ``shaper.inject_runtime_tools_hint(...)``) and so a future
    extension can carry per-runner configuration without a signature
    change. The four methods correspond to the four injection sites in
    goldfive's runtime; :meth:`should_inject` is the single
    ``observation_only`` gate they all consult.
    """

    # ----------------------------------------------------------------
    # Gate
    # ----------------------------------------------------------------

    @staticmethod
    def should_inject(steerer: Any) -> bool:
        """Return True when prompt-shaping injections are permitted.

        Delegates to :func:`goldfive.steerer.steering_is_active` — the
        one kill-switch predicate for
        :class:`~goldfive.config.SteeringConfig.observation_only`.
        ``steerer is None`` and steerers that don't expose
        :meth:`~goldfive.steerer.DefaultSteerer.is_active_steering`
        resolve to ``False`` (injections suppressed): passive is the
        fail-safe direction. Pre-refactor this site defaulted ACTIVE on
        a missing attribute; the flip to the passive fallback is
        deliberate.
        """
        return steering_is_active(steerer)

    @staticmethod
    def _signal_channel(steerer: Any) -> str:
        """Return the steerer's resolved ``signal_channel`` (default legacy).

        AGENCY-PRESERVATION.md PR 9 gates the prompt-shaping diet on the
        ``request_context`` regime: under ``"legacy_user_message"`` (the
        production default and what every existing suite runs with) the
        four sites behave byte-identically to pre-PR-9, so existing tests
        pass unmodified (§5.1). Tolerant of ``None`` / minimal stubs —
        both resolve to ``"legacy_user_message"`` so the legacy path is
        the safe fallback.
        """
        channel = _signal_channel_of(steerer)
        return str(channel or "legacy_user_message")

    @classmethod
    def _request_context(cls, steerer: Any) -> bool:
        """True when the steerer is in the PR-9 ``request_context`` diet regime."""
        return cls._signal_channel(steerer) == "request_context"

    @staticmethod
    def _pin_assigned_task(steerer: Any) -> bool:
        """Return ``SteeringConfig.pin_assigned_task`` off the steerer.

        The Site-4 escape hatch (AGENCY-PRESERVATION.md PR 9): when
        ``True`` the ``[CURRENT ASSIGNED TASK]`` instruction pin injects
        even under ``request_context``. Defaults to ``False`` and is
        tolerant of steerers without a typed config (returns ``False``).
        """
        cfg = getattr(steerer, "_steering_config", None)
        return bool(getattr(cfg, "pin_assigned_task", False))

    # ----------------------------------------------------------------
    # Site 1 — conversational follow-up wrap
    # ----------------------------------------------------------------

    def wrap_conversational_input(
        self,
        *,
        user_input: str,
        session: Session,
        steerer: Any,
    ) -> str:
        """Return either the wrapped follow-up directive or the raw input.

        F6 (closes goldfive#277). When :meth:`Planner.handle_turn`
        returns ``None`` on a turn with a real prior plan, the runner
        reuses ``session.plan`` but still drives the executor over the
        input. The wrapper supplies the coordinator with the prior-plan
        context (summary + completed tasks) so it can answer the
        follow-up from history. Lives in the message body (no
        system-prompt contract — users bring their own coordinator
        prompts; goldfive must not require a specific contract).

        Two regimes (AGENCY-PRESERVATION.md PR 9 prompt-shaping diet):

        * ``signal_channel == "legacy_user_message"`` (the default) —
          the legacy wrapper: the prior-plan context PLUS an explicit
          "do not delegate / Do NOT call any AgentTool" directive. Kept
          byte-identical so existing suites pass unmodified (§5.1).
        * ``signal_channel == "request_context"`` — the diet wrapper:
          the same prior-plan CONTEXT, but WITHOUT the means-directive.
          goldfive owns goals; the wrapped agent owns MEANS (whether to
          delegate is its call). The agent is informed, not commanded.

        Note: there is no ADK-plugin "tool-surface-tightening
        interceptor" — it was described in a stale docstring but never
        shipped (verified PR 9). ``session._conversational_turn`` is
        consumed only by the runner itself to decide whether to call
        this wrapper.

        Under ``observation_only=True`` (:meth:`should_inject` →
        ``False``) the wrapper is skipped and ``user_input`` is
        returned verbatim. The strict-passive operator sees the RAW
        coordinator behaviour on a follow-up turn — which may include
        re-delegation; that's the diagnostic value of strict-passive,
        not a regression.
        """
        if not self.should_inject(steerer):
            log.info(
                "PromptShaper.wrap_conversational_input: observation_only=True — "
                "SKIPPING conversational-follow-up wrap "
                "(user_input passes through raw); session_id=%s",
                (session.id or "")[:16] or "<none>",
            )
            return user_input

        from goldfive.types import TaskStatus

        plan = session.plan
        plan_summary = (plan.summary if plan is not None else "") or "(no summary)"
        completed_lines: list[str] = []
        if plan is not None:
            for t in plan.tasks:
                if t.status is TaskStatus.COMPLETED:
                    title = t.title or t.description or t.id
                    assignee = t.assignee_agent_id or "(no-assignee)"
                    completed_lines.append(f"  - [{t.id}] {title} (by {assignee})")
        completed_block = (
            "\n".join(completed_lines) if completed_lines else "  (none yet)"
        )

        if self._request_context(steerer):
            # Diet: prior-plan CONTEXT only — no means-directive. The
            # agent decides whether to delegate.
            return (
                "[CONVERSATIONAL FOLLOW-UP — prior plan context for "
                "reference]\n\n"
                "The user is asking a follow-up question about prior work. "
                "The prior plan and completed tasks are below for context.\n\n"
                f"Plan summary: {plan_summary}\n"
                f"Completed tasks:\n{completed_block}\n\n"
                f"User question: {user_input}"
            )

        return (
            "[CONVERSATIONAL FOLLOW-UP — reuse prior plan, don't delegate "
            "to sub-agents]\n\n"
            "The user is asking a follow-up question about prior work. "
            "Answer briefly using the conversation history and existing "
            "artifacts. Do NOT call any AgentTool — answer directly.\n\n"
            f"Plan summary: {plan_summary}\n"
            f"Completed tasks:\n{completed_block}\n\n"
            f"User question: {user_input}"
        )

    # ----------------------------------------------------------------
    # Site 2 — GoldfivePlanner request-side system_instruction
    # ----------------------------------------------------------------

    async def inject_goldfive_planner_instruction(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
        session_context: Any = None,
    ) -> None:
        """Append :class:`GoldfivePlanner` output to ``llm_request.config.system_instruction``.

        ADK's ``flows/llm_flows/_nl_planning.py`` request-side processor
        gates instruction injection on ``isinstance(planner,
        PlanReActPlanner)`` — so a ``BasePlanner`` subclass that is NOT
        a PlanReAct subclass gets its ``build_planning_instruction``
        called on the RESPONSE side (via
        ``process_planning_response``) but never on the REQUEST side.
        That's fine for response filtering but it starves the model of
        goldfive's orchestration context block on the turn that
        matters.

        This helper is the workaround: detect when the running agent's
        ``.planner`` is a :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner`
        (NOT a ``PlanReActPlanner`` or ``BuiltInPlanner`` — ADK handles
        those on its own via ``_nl_planning``) and append the planner's
        :meth:`build_planning_instruction` output to the request's
        system instruction using ADK's own ``append_instructions``
        method.

        Silent fall-throughs in priority order:

        * ADK not installed or ``BasePlanner`` import fails → skip.
        * Running agent has no ``planner`` attribute or planner is
          ``None`` → skip (plain ADK LlmAgent with no goldfive
          attachment).
        * Planner is a ``PlanReActPlanner`` / ``BuiltInPlanner``
          subclass → skip (ADK will handle it natively).
        * ``build_planning_instruction`` returns ``None`` / empty →
          skip (planner opted out for this turn).
        * ``llm_request`` lacks ``append_instructions`` → fall back to
          writing directly into ``config.system_instruction``.

        Under ``observation_only=True`` (:meth:`should_inject` →
        ``False``) the injection is suppressed and
        ``llm_request.config.system_instruction`` is left untouched.
        The strict-passive operator sees whatever ``system_instruction``
        ADK / the caller stamped, with no goldfive augmentation.

        ``session_context`` carries the live
        :class:`~goldfive.adapters._adk_plugin.SessionContext` from the
        plugin so the gate can reach the steerer. ``None`` (or a
        context without a steerer) resolves passive — injections
        suppressed, the fail-safe direction.
        """
        steerer = getattr(session_context, "steerer", None) if session_context is not None else None
        if not self.should_inject(steerer):
            log.info(
                "PromptShaper.inject_goldfive_planner_instruction: "
                "observation_only=True — SKIPPING GoldfivePlanner "
                "request-side instruction injection "
                "(system_instruction unchanged)"
            )
            return

        try:
            from goldfive.planners.goldfive_planner import GoldfivePlanner
        except ImportError:  # pragma: no cover — ADK not installed
            return
        try:
            from google.adk.planners.base_planner import BasePlanner  # type: ignore
            from google.adk.planners.built_in_planner import BuiltInPlanner  # type: ignore
            from google.adk.planners.plan_re_act_planner import (  # type: ignore
                PlanReActPlanner,
            )
        except ImportError:  # pragma: no cover — ADK not installed
            return

        # Find the running agent on the callback_context. ADK exposes
        # it through the invocation context; tests may supply a context
        # that carries ``.agent`` directly.
        inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
            callback_context, "invocation_context", None
        )
        agent = _safe_attr(inv_ctx, "agent", None)
        if agent is None:
            agent = _safe_attr(callback_context, "agent", None)
        if agent is None:
            return

        planner = _safe_attr(agent, "planner", None)
        if planner is None:
            return
        # If ADK itself will inject for this planner type, skip.
        # ``BuiltInPlanner`` never emits a text instruction (it
        # configures thinking on the request instead),
        # ``PlanReActPlanner`` is the one ADK gates on.
        if isinstance(planner, (PlanReActPlanner, BuiltInPlanner)):
            return
        if not isinstance(planner, BasePlanner):
            return
        # Narrow further: the adapter attaches GoldfivePlanner
        # specifically. A custom BasePlanner subclass that is not
        # GoldfivePlanner should be respected by ADK's own (response-
        # side) dispatch only, not re-injected here.
        if not isinstance(planner, GoldfivePlanner):
            return

        # Build the ReadonlyContext ADK expects. When invocation_context
        # is available we use ADK's real class; otherwise we fall back
        # to ``callback_context`` itself (test stubs carrying ``.state``
        # work through GoldfivePlanner's tolerant _extract_state).
        readonly = callback_context
        try:
            from google.adk.agents.readonly_context import ReadonlyContext  # type: ignore

            if inv_ctx is not None:
                readonly = ReadonlyContext(inv_ctx)
        except Exception as exc:  # noqa: BLE001 — use fallback
            log.debug(
                "PromptShaper.inject_goldfive_planner_instruction: "
                "ReadonlyContext unavailable: %s",
                exc,
            )

        instruction = planner.build_planning_instruction(readonly, llm_request)
        if not instruction:
            return

        append = getattr(llm_request, "append_instructions", None)
        if not callable(append):
            # Best-effort write directly into
            # ``config.system_instruction`` when the test stub doesn't
            # carry the helper. Preserves the existing value if it's a
            # string.
            config = getattr(llm_request, "config", None)
            if config is None:
                return
            existing = getattr(config, "system_instruction", None)
            if not existing:
                try:
                    config.system_instruction = instruction
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PromptShaper.inject_goldfive_planner_instruction: "
                        "could not set system_instruction: %s",
                        exc,
                    )
            elif isinstance(existing, str):
                try:
                    config.system_instruction = existing + "\n\n" + instruction
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PromptShaper.inject_goldfive_planner_instruction: "
                        "could not append system_instruction: %s",
                        exc,
                    )
            return

        try:
            append([instruction])
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "PromptShaper.inject_goldfive_planner_instruction: "
                "append_instructions raised: %s",
                exc,
            )

    # ----------------------------------------------------------------
    # Site 3 — runtime tool-surface hint
    # ----------------------------------------------------------------

    def inject_runtime_tools_hint(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
        session: Any,
        session_context: Any = None,
    ) -> None:
        """Inject (or refresh) the runtime tool-surface hint on ``llm_request``.

        R3 (F2 alternative). Builds the current plan-state hint via
        :func:`goldfive.adapters._adk_plugin._build_runtime_tools_hint`
        and writes it to ``llm_request.config.system_instruction``. If
        a prior goldfive hint is present (detected by
        :data:`goldfive.adapters._adk_plugin._RUNTIME_TOOLS_HINT_PREFIX`),
        it is stripped first so we don't accumulate snapshots across
        calls.

        Best-effort: never raises. A ``None`` hint (no plan, or plan
        with no informative groups) is a no-op so we don't pollute the
        prompt with empty markers.

        Under ``observation_only=True`` (:meth:`should_inject` →
        ``False``) the injection (and stale-hint strip) is suppressed
        — the strict-passive operator sees the unaltered
        ``system_instruction`` ADK / the caller set.

        ``session_context`` carries the live
        :class:`~goldfive.adapters._adk_plugin.SessionContext` from the
        plugin so the gate can reach the steerer. May be ``None`` on
        unit-test stubs that never wire one up.
        """
        steerer = getattr(session_context, "steerer", None) if session_context is not None else None
        if not self.should_inject(steerer):
            log.info(
                "PromptShaper.inject_runtime_tools_hint: "
                "observation_only=True — SKIPPING runtime tool-surface "
                "hint injection (system_instruction unchanged)"
            )
            return

        # Defer to the helpers that still live in _adk_plugin so the
        # marker constants + hint composition stay in one place (they
        # are also publicly imported by the runtime-tool-surface-hint
        # unit tests).
        from goldfive.adapters._adk_plugin import (
            _RUNTIME_TOOLS_HINT_PREFIX,
            _build_runtime_tools_hint,
            _strip_prior_runtime_tools_hint,
        )

        config = _safe_attr(llm_request, "config", None)
        if config is None:
            return
        existing = getattr(config, "system_instruction", None)

        # Strip any prior hint regardless of regime. A None ``hint``
        # (plan disappeared or all groups empty) — and the diet regime
        # below — should still remove a stale marker block left over from
        # a legacy-channel turn.
        if isinstance(existing, str) and _RUNTIME_TOOLS_HINT_PREFIX in existing:
            try:
                config.system_instruction = _strip_prior_runtime_tools_hint(existing) or None
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PromptShaper.inject_runtime_tools_hint: "
                    "could not strip prior hint: %s",
                    exc,
                )
                return
            existing = getattr(config, "system_instruction", None)

        # AGENCY-PRESERVATION.md PR 9 — Site 3 diet. Under the
        # ``request_context`` regime the per-turn standalone plan-state
        # hint is RETIRED (it was a per-turn footprint — the dormancy
        # violation §1.1 calls out). Its plan-state content is folded
        # into the observer note's Status surface (see
        # :meth:`inject_observer_note`). Legacy regime is unchanged.
        if self._request_context(steerer):
            return

        hint = _build_runtime_tools_hint(session)
        if not hint:
            return

        append = getattr(llm_request, "append_instructions", None)
        if callable(append):
            try:
                append([hint])
                return
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PromptShaper.inject_runtime_tools_hint: "
                    "append_instructions raised: %s",
                    exc,
                )
                # fall through to direct write

        # Fallback for stubs that lack ``append_instructions`` (unit
        # tests).
        if not existing:
            try:
                config.system_instruction = hint
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PromptShaper.inject_runtime_tools_hint: "
                    "could not set system_instruction: %s",
                    exc,
                )
        elif isinstance(existing, str):
            try:
                config.system_instruction = existing + "\n\n" + hint
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PromptShaper.inject_runtime_tools_hint: "
                    "could not append system_instruction: %s",
                    exc,
                )

    # ----------------------------------------------------------------
    # Site 4 — dynamic instruction resolver
    # ----------------------------------------------------------------

    def make_dynamic_instruction(
        self,
        original_instruction: str,
        agent_name: str,
    ) -> Callable[[Any], str | Awaitable[str]]:
        """Return a resolver matching ADK's ``InstructionProvider`` signature.

        The returned callable:

        * Reads ``session.state`` off the ``ReadonlyContext`` to reach
          the :class:`~goldfive.adapters._adk_plugin.SessionContext`
          stash and from there the goldfive
          :class:`~goldfive.types.Session`.
        * Reads the current-task pin via
          :class:`~goldfive.state_store.StateStore`. If
          no pin is set, returns the ``original_instruction`` verbatim
          (pre-plan turns stay unchanged).
        * Looks up the task in ``Session.plan.tasks`` for ``title`` /
          ``description`` (the typed :class:`~goldfive.types.Task` is
          the source of truth — the resolver does not consult
          de-normalised ADK-state keys).
        * If a pending correction exists for
          ``(agent_name, current_task_id)`` the block is appended
          (Stream D writes; the store reads).

        The resolver is pure: given the same Session.state it returns
        the same string. No side effects, no persistence.

        ADK session-state templating: ``LlmAgent.canonical_instruction``
        marks callable instructions ``bypass_state_injection=True``, so
        installing the resolver over a string instruction would silently
        disable the documented ``{var}`` / ``{artifact.var}``
        substitution. When ``original_instruction`` carries a ``{`` the
        resolver therefore returns an awaitable that first runs ADK's
        own ``inject_session_state`` over the template (substitution
        errors propagate exactly as they would unwrapped) and then
        applies the goldfive augmentation to the templated result.
        Placeholder-free instructions keep the synchronous, byte-
        identical fast path. Both shapes satisfy ADK's
        ``InstructionProvider`` alias (``str | Awaitable[str]``).

        Phase 2.0 of goldfive#271 — the bridge from goldfive
        Session.state onto ADK session.state is gone. The resolver
        reads goldfive Session directly via the SessionContext stash,
        eliminating the callback-time write to ADK state that raced
        with ADK's optimistic-concurrency contract (see
        goldfive#275).

        Under ``observation_only=True`` (:meth:`should_inject` →
        ``False``) the resolver returns the templated
        ``original_instruction`` and nothing else — no "Current
        assigned task" block, no pending-correction block, no goldfive
        augmentation of any kind. State templating still runs because
        ADK applies it to string instructions regardless of goldfive;
        suppressing it would itself be a behaviour change.

        Legacy fallback: when the SessionContext stash is unreachable
        (a unit test drives the resolver against a plain state dict
        without the stash), the resolver reads the pin / title /
        description / correction directly off ADK state. Production
        paths always carry the stash.
        """
        # Capture ``self`` so the closure can consult the gate via the
        # shaper's :meth:`should_inject` predicate.
        shaper = self

        def _resolve(readonly_ctx: Any, base_instruction: str) -> str:
            # ``base_instruction`` is ``original_instruction`` with ADK
            # session-state templating already applied (or verbatim on
            # the placeholder-free fast path). Every degradation path
            # below returns it so a goldfive failure never costs the
            # agent its templated instruction.
            try:
                # Reach the goldfive SessionContext to read the
                # steerer + session. The same walk supplies both the
                # gate's steerer and the resolver's session.
                from goldfive.adapters import _adk_state_protocol as _sp
                from goldfive.adapters.adk_llm_instrumentation import (
                    _compose_instruction,
                    _goals_block_from_session,
                    _goldfive_session_context_from_readonly_context,
                    _read_pending_correction,
                    _state_from_readonly_context,
                    _task_kind_from_session,
                    _task_title_description_from_session,
                )

                ctx = _goldfive_session_context_from_readonly_context(readonly_ctx)
                steerer = getattr(ctx, "steerer", None) if ctx is not None else None

                if not shaper.should_inject(steerer):
                    log.info(
                        "PromptShaper.make_dynamic_instruction resolver: "
                        "observation_only=True — SKIPPING goldfive "
                        "prompt augmentation for agent=%r "
                        "(returning the caller's instruction un-augmented)",
                        agent_name,
                    )
                    return base_instruction

                # AGENCY-PRESERVATION.md PR 9 — Site 4 diet. Under the
                # ``request_context`` regime the per-turn
                # ``[CURRENT ASSIGNED TASK]`` pin is RETIRED by default:
                # the wrapped agent owns its own decomposition / ordering
                # and does not need goldfive restating the bound task into
                # every model call. ``pin_assigned_task`` is the escape
                # hatch for trees built around the pinned-task block.
                # Legacy ``signal_channel`` keeps the pin (byte-identical;
                # §5.1).
                #
                # Pending corrections (which legacy rides on this pin) are
                # NOT lost in this regime: task #11 routes them to the
                # agent-scoped ObserverNoteQueue at write time
                # (``queue_corrections_for_revision(corrections_via_notes=
                # True)``), delivered via the observer-note surfaces — the
                # ``peek_for_render`` agent filter ensures a per-(agent,task)
                # correction reaches only its agent. (The PR-9 "written but
                # unread" gap this note once flagged is now closed.) The
                # ``pin_assigned_task=True`` escape hatch instead keeps the
                # pin AND the legacy correction-slot read.
                if shaper._request_context(steerer) and not shaper._pin_assigned_task(
                    steerer
                ):
                    log.info(
                        "PromptShaper.make_dynamic_instruction resolver: "
                        "request_context regime + pin_assigned_task=False — "
                        "retiring the [CURRENT ASSIGNED TASK] pin for "
                        "agent=%r (returning the templated instruction "
                        "un-augmented)",
                        agent_name,
                    )
                    # goldfive#477: return the TEMPLATED base, not the raw
                    # ``original_instruction`` — the resolver bypasses ADK's
                    # own state injection, so returning the raw template
                    # would leak literal ``{var}`` placeholders to the
                    # agent on the diet path.
                    return base_instruction

                state = _state_from_readonly_context(readonly_ctx)
                session = getattr(ctx, "session", None) if ctx is not None else None

                if session is not None:
                    from goldfive.state_store import StateStore

                    store = StateStore.for_session(session)
                    current_task_id = store.pin_current_task()
                else:
                    current_task_id = str(state.get(_sp.KEY_CURRENT_TASK_ID, "") or "")

                if not current_task_id:
                    # No pin — pre-plan turn, or an agent that doesn't
                    # need plan-causal augmentation this turn. Return
                    # the caller's instruction un-augmented.
                    return base_instruction

                if session is not None:
                    current_task_title, current_task_description = (
                        _task_title_description_from_session(session, current_task_id)
                    )
                else:
                    # Legacy ADK-state fallback path.
                    current_task_title = str(
                        state.get(_sp.KEY_CURRENT_TASK_TITLE, "") or ""
                    )
                    current_task_description = str(
                        state.get(_sp.KEY_CURRENT_TASK_DESCRIPTION, "") or ""
                    )

                pending_correction = _read_pending_correction(
                    session=session,
                    state=state,
                    agent_name=agent_name,
                    current_task_id=current_task_id,
                )

                # AGENCY-PRESERVATION.md Stage 3 PR 12 — for a DISCOVERED
                # pin (ledger plan mode), render a [GOALS] block instead of
                # the task block (the agent owns its own means-work; ground
                # it on goals, don't re-pin the discovered task as a
                # prescription). ``_compose_instruction`` falls back to the
                # task block when the kind is not DISCOVERED or there are no
                # goals, so forecast / OUTCOME pins are byte-identical. This
                # branch is reached only where the pin WOULD render — under
                # the PR 9 request_context+pin-off diet the resolver already
                # returned above.
                task_kind = (
                    _task_kind_from_session(session, current_task_id)
                    if session is not None
                    else ""
                )
                goals_block = (
                    _goals_block_from_session(session) if session is not None else ""
                )

                return _compose_instruction(
                    original=base_instruction,
                    task_id=current_task_id,
                    task_title=current_task_title,
                    task_description=current_task_description,
                    pending_correction=pending_correction,
                    task_kind=task_kind,
                    goals_block=goals_block,
                )
            except Exception as exc:  # noqa: BLE001
                # Instrumentation path — any failure here degrades to
                # the base instruction so the agent still runs.
                # ADK's own pipeline would otherwise surface this as an
                # InternalError mid-turn, which is the worst possible
                # failure mode.
                log.debug(
                    "PromptShaper.make_dynamic_instruction resolver "
                    "raised for agent=%r: %s "
                    "(falling back to the caller's instruction)",
                    agent_name,
                    exc,
                )
                return base_instruction

        def resolver(readonly_ctx: Any) -> str | Awaitable[str]:
            # ADK marks callable instructions ``bypass_state_injection=
            # True`` (``LlmAgent.canonical_instruction``), so the
            # ``{var}`` / ``{artifact.var}`` templating a string
            # instruction receives from ADK's flow must be re-applied
            # here. A placeholder-free template cannot substitute, so
            # it keeps the synchronous byte-identical path.
            if "{" not in original_instruction:
                return _resolve(readonly_ctx, original_instruction)

            from goldfive.adapters.adk_llm_instrumentation import (
                _adk_inject_session_state,
            )

            inject = _adk_inject_session_state()
            if inject is None:
                # install_dynamic_instructions refuses to wrap templated
                # instructions when the helper is absent; a resolver
                # built directly degrades to the literal template.
                return _resolve(readonly_ctx, original_instruction)

            async def _inject_then_resolve() -> str:
                # Substitution errors (missing state var / artifact)
                # propagate — a string instruction fails the same way
                # unwrapped.
                base = await inject(original_instruction, readonly_ctx)
                return _resolve(readonly_ctx, base)

            return _inject_then_resolve()

        # Stamp provenance on the closure so test code and tree-walk
        # idempotency checks can recognise it without relying on repr.
        resolver._goldfive_dynamic_instruction = True  # type: ignore[attr-defined]
        resolver._goldfive_agent_name = agent_name  # type: ignore[attr-defined]
        resolver._goldfive_original_instruction = original_instruction  # type: ignore[attr-defined]
        return resolver

    # ----------------------------------------------------------------
    # Site 5 — observer-note channel (AGENCY-PRESERVATION.md PR 6)
    # ----------------------------------------------------------------

    @staticmethod
    def _plan_state_line(session: Any) -> str:
        """Return the factual per-agent open-work line for the note Status fold.

        AGENCY-PRESERVATION.md PR 9 introduced this fold on the
        before_model surface; task #11 centralised the composition in
        :func:`goldfive.observer_note_queue.plan_state_line` so the
        boundary-replay + claude surfaces render an identical line. This
        thin wrapper preserves the ``(session)`` call shape; the shared
        helper takes the plan. Returns ``""`` when there is no plan.
        """
        try:
            from goldfive.observer_note_queue import plan_state_line

            return plan_state_line(_safe_attr(session, "plan", None))
        except Exception:  # noqa: BLE001
            return ""

    def inject_observer_note(
        self,
        *,
        llm_request: Any,
        session: Any,
        session_context: Any = None,
        current_agent_name: str = "",
    ) -> Any:
        """Render the most-severe pending observer note onto the request.

        Surface 1 of the PR 6 observer-note channel — the preferred surface,
        reaching a *mid-invocation* agent on its next model call (which removes
        the only remaining justification for cancel-as-information-delivery).
        Uses the marker strip-and-refresh pattern (mirrors
        :meth:`inject_runtime_tools_hint`): any prior observer-note block,
        bracketed by
        :data:`~goldfive.observer_note_queue.OBSERVER_NOTE_MARKER_PREFIX` /
        ``OBSERVER_NOTE_BLOCK_END``, is stripped from ``system_instruction``
        before the current one is appended — so two consecutive
        ``before_model`` calls never stack blocks (the idempotency half of §5.2).

        Per-request coalescing: at most ONE block is rendered, the most-severe
        pending note wins
        (:meth:`~goldfive.observer_note_queue.ObserverNoteQueue.peek_for_render`),
        and that note is marked delivered — the exactly-once *rendering*
        chokepoint, so the invocation-boundary replay never re-renders a note
        this surface showed.

        Under ``observation_only=True`` (:meth:`should_inject` → ``False``) the
        block is NOT appended (the strict-passive operator sees the raw prompt)
        but the note is still consumed so the queue does not re-evaluate it —
        the dispatch-point ``SignalDelivered(dry_run=True)`` already recorded
        what *would* have been delivered (§5.4).

        ``SignalDelivered`` is emitted once at the dispatch decision point
        (``DriftObserver._route_corrective_note``), NOT here — this surface is
        purely the *rendering* leg. Returns the
        :class:`~goldfive.observer_note_queue.ObserverNote` it rendered (for
        observability / tests), or ``None`` when nothing was pending (or on a
        defensive failure). Best-effort: never raises into
        ``before_model_callback``.

        ``session`` MUST be the goldfive :class:`~goldfive.types.Session` (the
        queue lives on goldfive ``Session.state``, the same dict the drift
        observer enqueues onto — NOT ADK ``session.state``, which is
        shallow-copied across the callback boundary).
        """
        try:
            from goldfive.observer_note_queue import (
                OBSERVER_NOTE_MARKER_PREFIX,
                ObserverNoteQueue,
                render_block,
                strip_prior_block,
            )
        except Exception:  # pragma: no cover - defensive import
            return None

        steerer = (
            getattr(session_context, "steerer", None)
            if session_context is not None
            else None
        )

        config = _safe_attr(llm_request, "config", None)
        if config is None:
            return None
        existing = getattr(config, "system_instruction", None)
        # Strip any prior observer-note block regardless of whether we
        # re-inject this call (strip-and-refresh: never stack blocks).
        if isinstance(existing, str) and OBSERVER_NOTE_MARKER_PREFIX in existing:
            try:
                config.system_instruction = strip_prior_block(existing) or None
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PromptShaper.inject_observer_note: could not strip prior block: %s",
                    exc,
                )
                return None
            existing = getattr(config, "system_instruction", None)

        try:
            queue = ObserverNoteQueue.for_session(session)
            # Agent-scoped (task #11): the before_model surface KNOWS the
            # agent whose model call this is, so an agent-specific note
            # (e.g. a per-(agent,task) correction) is rendered only on
            # its own agent's call — never on a sibling's. ``""`` (no
            # agent resolved, e.g. a unit-test stub) → no filter,
            # preserving the pre-task-#11 broadcast behaviour (§5.1).
            note = queue.peek_for_render(agent_id=current_agent_name or None)
        except Exception as exc:  # noqa: BLE001
            log.debug("PromptShaper.inject_observer_note: queue read raised: %s", exc)
            return None
        if note is None:
            return None

        rendered = self.should_inject(steerer)
        if rendered:
            # task #11 cross-surface fold: render_block composes the
            # plan-state Status line from the plan INSIDE the marker block
            # (strip-and-refresh removes it as one unit; marker count stays
            # 1). Centralised so this surface, the boundary replay, and the
            # claude surface render an identical line.
            block = render_block(note, plan=_safe_attr(session, "plan", None))
            append = getattr(llm_request, "append_instructions", None)
            wrote = False
            if callable(append):
                try:
                    append([block])
                    wrote = True
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PromptShaper.inject_observer_note: "
                        "append_instructions raised: %s",
                        exc,
                    )
            if not wrote:
                # Fallback for stubs / requests without ``append_instructions``.
                try:
                    if not existing:
                        config.system_instruction = block
                    elif isinstance(existing, str):
                        config.system_instruction = existing + "\n\n" + block
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PromptShaper.inject_observer_note: "
                        "could not write system_instruction: %s",
                        exc,
                    )
        else:
            log.info(
                "PromptShaper.inject_observer_note: observation_only=True — "
                "consuming note %s as a dry-run delivery "
                "(system_instruction unchanged)",
                getattr(note, "note_id", "?"),
            )

        turn = int(_safe_attr(session, "_reasoning_turn", 0) or 0)
        try:
            newly = queue.mark_delivered(
                note.note_id,
                channel="request_context",
                turn=turn,
                surface="before_model",
                dry_run=not rendered,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("PromptShaper.inject_observer_note: mark_delivered raised: %s", exc)
            return None
        return note if newly else None


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Local copy of the defensive attribute reader used by the ADK plugin.

    Keeps this module independent of an optional google-adk install —
    callers that pass ADK objects get attribute access; stubs that
    don't carry the attribute degrade to ``default``.
    """
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


__all__ = ["PromptShaper"]
