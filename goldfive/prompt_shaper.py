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

4. **Dynamic instruction resolver** —
   :meth:`PromptShaper.make_dynamic_instruction`. Returns an ADK
   ``InstructionProvider`` callable that resolves the current pinned
   task + pending-correction block on every turn and appends them to
   the agent's ``original_instruction`` (goldfive#251).

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

        Reads ``steerer._observation_only`` and returns its logical
        complement: under strict-passive (``observation_only=True``)
        injections are suppressed; otherwise the active-steering path
        runs.

        Tolerant of ``steerer is None`` and steerers that don't carry
        a ``_observation_only`` attribute — both cases return ``True``
        so pre-#271 paths and minimal test stubs (which never set up a
        steerer) keep working byte-identically.

        This is the single source of truth the four inject methods
        consult. Mirrors :meth:`DefaultSteerer._should_inject` in
        intent — that predicate gates enforcement-side dispatch (steer,
        pause-escalate, cancel); this one gates prompt-shape
        injections.
        """
        if steerer is None:
            return True
        return not bool(getattr(steerer, "_observation_only", False))

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
        input — without this wrapper the coordinator typically treats
        the question as a fresh task and re-delegates to sub-agents,
        wasting an invocation. The wrapper:

        * tags the message as a conversational follow-up,
        * gives the coordinator the plan summary + completed task
          context so it can answer from history,
        * asks it explicitly NOT to delegate.

        Lives in the message body (no system-prompt contract). A
        parallel layer (the ADK plugin's pre-dispatch interceptor,
        keyed off ``session._conversational_turn``) tightens the
        tool surface so the coordinator literally cannot delegate
        even if it tried; this wrapper is the cooperative half.

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
        plugin so the gate can reach the steerer. May be ``None`` on
        unit-test stubs that never wire one up — those paths default
        to the active-steering branch (pre-#271 behaviour).
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

        hint = _build_runtime_tools_hint(session)

        config = _safe_attr(llm_request, "config", None)
        if config is None:
            return
        existing = getattr(config, "system_instruction", None)

        # Strip any prior hint regardless of whether we'll re-inject. A
        # None ``hint`` (plan disappeared or all groups empty) should
        # still remove the stale marker block from the request.
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
    ) -> Callable[[Any], str]:
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

        Phase 2.0 of goldfive#271 — the bridge from goldfive
        Session.state onto ADK session.state is gone. The resolver
        reads goldfive Session directly via the SessionContext stash,
        eliminating the callback-time write to ADK state that raced
        with ADK's optimistic-concurrency contract (see
        goldfive#275).

        Under ``observation_only=True`` (:meth:`should_inject` →
        ``False``) the resolver returns ``original_instruction``
        verbatim — no "Current assigned task" block, no pending-
        correction block, no goldfive augmentation of any kind.

        Legacy fallback: when the SessionContext stash is unreachable
        (a unit test drives the resolver against a plain state dict
        without the stash), the resolver reads the pin / title /
        description / correction directly off ADK state. Production
        paths always carry the stash.
        """
        # Capture ``self`` so the closure can consult the gate via the
        # shaper's :meth:`should_inject` predicate.
        shaper = self

        def resolver(readonly_ctx: Any) -> str:
            try:
                # Reach the goldfive SessionContext to read the
                # steerer + session. The same walk supplies both the
                # gate's steerer and the resolver's session.
                from goldfive.adapters import _adk_state_protocol as _sp
                from goldfive.adapters.adk_llm_instrumentation import (
                    _compose_instruction,
                    _goldfive_session_context_from_readonly_context,
                    _read_pending_correction,
                    _state_from_readonly_context,
                    _task_title_description_from_session,
                )

                ctx = _goldfive_session_context_from_readonly_context(readonly_ctx)
                steerer = getattr(ctx, "steerer", None) if ctx is not None else None

                if not shaper.should_inject(steerer):
                    log.info(
                        "PromptShaper.make_dynamic_instruction resolver: "
                        "observation_only=True — SKIPPING goldfive "
                        "prompt augmentation for agent=%r "
                        "(returning original instruction verbatim)",
                        agent_name,
                    )
                    return original_instruction

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
                    # the caller's instruction verbatim.
                    return original_instruction

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

                return _compose_instruction(
                    original=original_instruction,
                    task_id=current_task_id,
                    task_title=current_task_title,
                    task_description=current_task_description,
                    pending_correction=pending_correction,
                )
            except Exception as exc:  # noqa: BLE001
                # Instrumentation path — any failure here degrades to
                # the original instruction so the agent still runs.
                # ADK's own pipeline would otherwise surface this as an
                # InternalError mid-turn, which is the worst possible
                # failure mode.
                log.debug(
                    "PromptShaper.make_dynamic_instruction resolver "
                    "raised for agent=%r: %s "
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

    # ----------------------------------------------------------------
    # Site 5 — observer-note channel (AGENCY-PRESERVATION.md PR 6)
    # ----------------------------------------------------------------

    def inject_observer_note(
        self,
        *,
        llm_request: Any,
        session: Any,
        session_context: Any = None,
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
            note = queue.peek_for_render()
        except Exception as exc:  # noqa: BLE001
            log.debug("PromptShaper.inject_observer_note: queue read raised: %s", exc)
            return None
        if note is None:
            return None

        if self.should_inject(steerer):
            block = render_block(note)
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
