"""The :class:`Runner` — goldfive's single public entrypoint.

A Runner composes the six pluggable components of a goldfive run:

* :class:`GoalDeriver` — turns ``user_input`` into ``list[Goal]``.
* :class:`Planner` — turns goals into a :class:`Plan`.
* :class:`Executor` — walks the plan, dispatches to the adapter.
* :class:`AgentAdapter` — talks to the underlying agent framework.
* :class:`Steerer` — runs the state machine, detects drift.
* :class:`EventSink` — persists / observes the event stream.

``Runner.run`` emits a ``RunStarted`` event, derives goals, generates a
plan, registers the canonical reporting tools on the adapter (the
lifecycle subset by default, the drift tools opted in via
``drift_self_reporting=`` — see :class:`Runner`), binds the steerer to
the sinks+planner, and hands everything to the executor. The returned
:class:`ExecutionOutcome` carries the final live :class:`Session` so
callers can inspect completed tasks / artifacts.

Event lifecycle ownership
-------------------------
* The Runner owns ``Run*`` lifecycle events (``RunStarted``,
  ``GoalDerived``, ``PlanSubmitted``, and pre-executor ``RunAborted``).
* Executors own ``Task*`` events, ``PlanRevised``, and the terminal
  ``RunCompleted`` / ``RunAborted`` they emit when their own state
  machine reaches the end of the run.
* The Steerer owns ``DriftDetected`` and the per-task ``mark_task_*``
  emissions.

All sink emissions are proto :class:`Event` envelopes — built via the
typed factories in :mod:`goldfive.events`.

No ADK or Claude Agent SDK imports live in this module. Optional
adapter implementations live under ``goldfive.adapters.<framework>``
and are loaded lazily by callers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import os
import subprocess
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from importlib import metadata as _importlib_metadata
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any

from goldfive import _state_audit
from goldfive import state_store as _ostate
from goldfive._llm import maybe_close_call_llm
from goldfive.conversation import Conversation
from goldfive.events import (
    conversation_ended_event,
    conversation_started_event,
    emit,
    goal_derived_event,
    run_aborted_event,
    run_started_event,
)
from goldfive.executors.sequential import _pending_task_ids
from goldfive.goal_deriver import PassthroughGoalDeriver
from goldfive.reporting import select_reporting_tools
from goldfive.results import ExecutionOutcome
from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    TaskStatus,
    channel_processor_active,
    set_session_plan,
)

if TYPE_CHECKING:
    from goldfive.control import ControlChannel
    from goldfive.protocols import (
        AgentAdapter,
        EventSink,
        Executor,
        GoalDeriver,
        Planner,
        Steerer,
    )

log = logging.getLogger("goldfive.runner")


def _detect_build_identity() -> tuple[str, str]:
    """Return ``(version, sha)`` for the running goldfive build.

    Both fields default to ``"unknown"`` and the function never raises —
    callers log the result on Runner construction so users can answer
    "is the change actually deployed?" from the logs (the diagnostic
    trap captured by ``feedback_verify_running_build.md``).

    Detection order:

    * ``importlib.metadata.version("goldfive")`` — set when goldfive is
      installed as a wheel / editable install.
    * ``goldfive.__version__`` — fallback for source checkouts that
      bypass the metadata path.
    * ``git rev-parse --short HEAD`` run from the package's install
      directory — only attempted when a ``.git`` directory exists at
      the repo root, and all exceptions are swallowed so missing-git
      environments still construct cleanly.
    """
    version = "unknown"
    try:
        version = _importlib_metadata.version("goldfive")
    except Exception:
        try:
            from goldfive import __version__ as _pkg_version

            version = _pkg_version or "unknown"
        except Exception:
            log.debug("goldfive version detection failed", exc_info=True)

    sha = "unknown"
    try:
        pkg_root = _Path(__file__).resolve().parent
        repo_root = pkg_root.parent
        if (repo_root / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if result.returncode == 0:
                sha = result.stdout.strip() or "unknown"
    except Exception:
        log.debug("goldfive git sha detection failed", exc_info=True)

    return version, sha


class Runner:
    """The public entrypoint for a goldfive run.

    Parameters
    ----------
    agent:
        An :class:`AgentAdapter` wrapping the underlying agent framework.
    planner:
        A :class:`Planner` instance. Pass a planner configured with a
        pre-baked plan when you already know the tasks (see
        :class:`PassthroughGoalDeriver` for an analogous convenience on
        the goals side).
    executor:
        An :class:`Executor` (e.g. :class:`SequentialExecutor` or
        :class:`ParallelDAGExecutor`).
    goal_deriver:
        Optional — defaults to ``PassthroughGoalDeriver("run")``. When
        the caller passes ``user_input`` as ``list[Goal]`` the deriver
        is bypassed entirely.
    steerer:
        Optional — defaults to :class:`DefaultSteerer`.
    sinks:
        Optional list of :class:`EventSink` instances. Defaults to ``[]``.
    control:
        Optional :class:`~goldfive.control.ControlChannel` for live
        pause / resume / cancel / steer / rewind from an external
        controller (harmonograf UI, CLI, tests). When provided, the
        Runner forwards it into the executor, which polls the channel
        between tasks and races against adapter invocations mid-task.
    max_task_invocations:
        Optional safety cap on adapter invocations per run. Stamped onto
        the planner context so executors that honour it can enforce the
        cap. Defaults to ``None`` (unbounded); per-task / per-tool caps
        are the primary guards against runaway loops.
    goal_drift_enabled:
        Opt-in switch for the trajectory-level GOAL_DRIFT periodic
        check (goldfive#143). ``True`` (default) leaves the steerer's
        own ``goal_drift_call_llm`` wiring intact -- operators who
        pass a steerer configured with a judge callable get the
        check. ``False`` forcibly disables it by detaching
        ``_goal_drift_call_llm`` on the steerer, which is the shape
        unit tests driving mock runners want so they never see
        spurious GOAL_DRIFT firings from the bookkeeping path.
        Has no effect when the steerer was never configured with a
        ``goal_drift_call_llm`` (the feature is already inert).
    planner_gate:
        Per-turn planning behaviour. Goldfive#271 Phase 4 collapsed
        the prior gate-then-refine pipeline into the planner's own
        :meth:`Planner.handle_turn` method, so the gate is no longer
        a separate layer. This kwarg is retained as a feature switch:

        * ``"auto"`` (default) — call ``planner.handle_turn`` on
          every turn. The planner LLM either produces the next plan
          (warrants change) or returns ``None`` (purely conversational
          — the Runner reuses ``session.plan`` unchanged). The
          "classification" is emergent: did the LLM produce a plan
          or not. Recommended production setting.
        * ``None`` — disable handle_turn entirely; every turn falls
          through to ``Planner.generate`` (pre-#271 behaviour, useful
          for deterministic replay).

        Skipped when ``user_input`` is already a ``list[Goal]`` (the
        caller has opted out of natural-language derivation).
    drift_self_reporting:
        Opt-in switch for the drift-related self-reporting tools
        (goldfive#196). These are tools that ask the agent to volunteer
        a drift opinion the framework can already observe by other
        means:

        * ``report_plan_divergence`` —
          :class:`~goldfive.reconciler.PlanReconciler` covers this
          observationally.
        * ``declare_task_skipped`` / ``declare_task_not_needed`` —
          observability-only declarations whose downstream consumer
          (the next refine) is also fed by the imperative
          ``report_task_*`` family.

        Each registered tool inflates the sub-agent's prompt by
        ~200-400 tokens AND expands the model's hallucination surface
        (a confused agent can confabulate a drift call). Default
        ``False`` registers ONLY the lifecycle subset
        (``report_task_started`` / ``_progress`` / ``_completed`` /
        ``_failed`` / ``_blocked`` / ``_awaiting_approval`` /
        ``report_new_work_discovered``). The framework's observation
        paths — ``classify_goal_drift``,
        :class:`~goldfive.reconciler.PlanReconciler`, the steerer's
        refine machinery — remain the canonical drift detectors.

        Accepted shapes:

        * ``False`` (default) — lifecycle subset only.
        * ``True`` — full canonical set (pre-#196 behaviour).
        * ``list[str]`` — lifecycle subset PLUS the named drift tools
          (e.g. ``["report_plan_divergence"]`` re-enables that one
          tool while leaving the declarations off). Names not in
          :data:`~goldfive.reporting.DRIFT_SELF_REPORTING_TOOL_NAMES`
          are silently ignored.

        ``report_new_work_discovered`` is intentionally NOT a drift
        tool — there is no observation analog for an agent surfacing
        genuinely new work, so it stays default-on.
    fail_fast_on_revision_rejection:
        Strict-abort opt-in for goldfive-authored revisions that fail
        ``Plan.validate(for_revision=True, prior=...)``. Default
        (``False``) is non-fatal: when a goldfive-authored autonomous
        refine produces an invalid revision, the runner keeps the
        existing ``session.plan``, emits a
        ``HUMAN_INTERVENTION_REQUIRED`` INFO ``DriftDetected`` for
        observability, and continues the turn. The next refine cycle
        gets another attempt; the existing
        ``REFINE_FAILURE_THRESHOLD=2`` escalation still fires after two
        consecutive failures of the same ``(kind, task_id)``.

        Operators wanting strict abort-on-rejection — useful for CI,
        regression testing, or debugging refine logic — opt in by
        passing ``True`` (or setting the env var
        ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1``). When ``None``
        (the default), the env var is consulted; explicit ``False`` /
        ``True`` always wins over the env.

        **User-authored** drifts (``USER_STEER`` from a
        ``ControlMessage``) are NEVER affected by this flag — the
        :meth:`DefaultSteerer.install_user_steer` API guarantees a
        valid ``Plan`` returns even when the LLM revision fails
        validation (PLAN-LIFECYCLE.md §4.2.1). User-steer rejection is
        structurally impossible.

        See PLAN-LIFECYCLE.md §4.5.1 for the full rationale.
    """

    def __init__(
        self,
        *,
        agent: AgentAdapter,
        planner: Planner,
        executor: Executor,
        goal_deriver: GoalDeriver | None = None,
        steerer: Steerer | None = None,
        sinks: list[EventSink] | None = None,
        control: ControlChannel | None = None,
        max_task_invocations: int | None = None,
        conversation: Conversation | None = None,
        goal_drift_enabled: bool = True,
        planner_gate: Any = "auto",
        drift_self_reporting: bool | list[str] = False,
        fail_fast_on_revision_rejection: bool | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        # Stamp the running build's identity at construction so logs
        # answer "did the change actually deploy?" without spelunking.
        # See ``feedback_verify_running_build.md`` for the motivating
        # 30-minute diagnosis trap. INFO once per Runner; never raises.
        _version, _sha = _detect_build_identity()
        log.info("goldfive runner starting: version=%s sha=%s", _version, _sha)
        if "max_plan_reinvocations" in legacy_kwargs:
            legacy_value = legacy_kwargs.pop("max_plan_reinvocations")
            warnings.warn(
                "Runner(max_plan_reinvocations=...) is deprecated; use "
                "max_task_invocations=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_task_invocations is None:
                max_task_invocations = legacy_value
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Runner got unexpected keyword argument(s): {unexpected}")
        self.agent = agent
        self.planner = planner
        self.executor = executor
        self.goal_deriver: GoalDeriver = goal_deriver or PassthroughGoalDeriver("run")
        self.steerer: Steerer = steerer or DefaultSteerer()
        # Wave B1 (refactor/prompt-shaper): centralised prompt-shaping
        # injection sites + ``observation_only`` gate. The Runner only
        # uses :meth:`PromptShaper.wrap_conversational_input` directly;
        # the ADK plugin holds its own instance for the three
        # request-side sites. Stateless — instantiation here keeps the
        # gate predicate co-located with the runner's other dispatch
        # helpers.
        from goldfive.prompt_shaper import PromptShaper

        self._prompt_shaper = PromptShaper()
        # goldfive#143: opt-in gate for the trajectory-level GOAL_DRIFT
        # periodic check. ``True`` (default) is a no-op -- the steerer's
        # own ``goal_drift_call_llm`` wiring governs whether the check
        # fires. ``False`` forcibly detaches the callable so mock-only
        # runs never see GOAL_DRIFT firings, even if a test accidentally
        # wires a callable through. Guarded on ``hasattr`` so custom
        # ``Steerer`` implementations that predate this attribute still
        # construct cleanly.
        self.goal_drift_enabled: bool = goal_drift_enabled
        if not goal_drift_enabled and hasattr(self.steerer, "_goal_drift_call_llm"):
            self.steerer._goal_drift_call_llm = None
        elif (
            goal_drift_enabled
            and hasattr(self.steerer, "_goal_drift_call_llm")
            and self.steerer._goal_drift_call_llm is None
        ):
            # Soft-fail: the docstring at ``DefaultSteerer.__init__`` says
            # the Runner wires its planner LLM here when the feature is
            # on. If no callable is present the judge can never fire;
            # surface that once rather than failing silently. Don't raise
            # -- existing Runner(...) callers that build a steerer
            # without a judge intentionally (mock tests, degraded LLMs)
            # must still construct cleanly. See goldfive#217.
            log.warning(
                "goal_drift_enabled=True but no call_llm wired on steerer; "
                "goal-drift judge disabled"
            )
        self.sinks: list[EventSink] = list(sinks) if sinks else []
        self._control: ControlChannel | None = control
        self._close_hooks: list[Callable[[], Awaitable[None]]] = []
        self._closed: bool = False
        self.max_task_invocations: int | None = max_task_invocations
        # Per-outer-session :class:`Conversation` map (goldfive#271
        # follow-up to PR #293 / validation v4 Class 1).
        #
        # Pre-fix the Runner held one ``self._conversation``: a
        # singleton on the Runner instance. PR #293 keyed the
        # *prior-plan stash* on session id, but every other
        # Conversation field (``goals``, ``completed_results``,
        # ``turns``, ``_next_sequence``) still leaked across distinct
        # outer ADK sessions sharing one Runner. The visible regression:
        # v4class1-1 saw "Provide the correct answer to 2+2" leaked
        # from a prior v4-class5 session that had run on the same
        # Runner — every subsequent session inherited every prior
        # session's accumulated goals.
        #
        # Keying the entire :class:`Conversation` by the outer-session
        # id used at :meth:`run`'s entry isolates per-session state in
        # full. The empty-string key (``""``) holds the conversation
        # for unpinned (programmatic) callers — preserving the legacy
        # single-Conversation continuity for pre-#161 callers and
        # one-shot scripts. Pinned callers (typically ADK-web via
        # :class:`GoldfiveADKAgent`) get their own Conversation per
        # ``ctx.session.id``.
        #
        # Lifetime: Conversations live forever for the Runner's
        # lifetime. The dict is small (one entry per distinct outer
        # session ever observed) and Conversations are bounded
        # (``recent_turns`` cap on prior-turn context). If a future
        # use case introduces churn (many short-lived outer sessions),
        # add a cleanup hook on session end.
        seed_conv = conversation or Conversation.new()
        self._conversations: dict[str, Conversation] = {"": seed_conv}
        # Per-key bookkeeping. ``_conversation_announced`` flips to
        # True after the per-key Conversation's ConversationStarted
        # event fires; ``_last_session_by_key`` holds the most recent
        # turn's Session so ConversationEnded can piggy-back on its
        # ``next_sequence()`` cursor (the terminal marker must share
        # its run_id's sequence keyspace).
        self._conversation_announced: dict[str, bool] = {"": False}
        self._last_session_by_key: dict[str, Session] = {}
        # Per-key serialisation lock for :meth:`run` so two concurrent
        # ``runner.run(session_id=X)`` calls on the SAME outer session
        # id do not race on the per-Conversation prior-plan stash. The
        # bug this guards against (goldfive#271 follow-up to PR #294 /
        # demo log v6 / v7class1-1): adk-web fires a second ``/run_sse``
        # while the first is still in-flight; turn 2 enters
        # :meth:`run` BEFORE turn 1's ``finally``-block
        # ``Conversation.stash_plan`` lands, so turn 2's
        # ``Conversation.prior_plan_for`` returns ``None`` and seeds
        # ``session.plan = Plan.empty(...)``. Turn 2's
        # ``Planner.handle_turn`` then sees an empty seed and the
        # produced plan inherits the empty seed's id — the
        # ``plan_id`` stable across turns invariant breaks. With this
        # lock, turn 2 waits for turn 1's full lifecycle (including
        # ``finally`` stash and ``absorb_turn``) before its own
        # bookkeeping runs. Concurrent runs on DIFFERENT keys still
        # proceed in parallel — distinct outer ADK sessions are
        # independent and have always been intended to run in
        # parallel on a shared Runner.
        self._convo_locks: dict[str, asyncio.Lock] = {}
        # Last turn's Session (any key) — kept for back-compat with
        # tests / inspectors that read ``runner._last_session``
        # directly. Updated on every :meth:`run` regardless of pin.
        self._last_session: Session | None = None
        # Turn-aware planning gate. Goldfive#271 Phase 4: the gate is
        # now ``planner.handle_turn`` (a single LLM call that classifies
        # AND produces the merged plan). ``"auto"`` (default) calls it
        # on every turn after the first when the planner exposes the
        # method; ``None`` disables it entirely so every turn re-plans
        # via ``Planner.generate`` (pre-#271 behaviour, useful for
        # deterministic replay).
        self._planner_gate: Any = planner_gate
        # goldfive#196: opt-in switch for the drift-related self-
        # reporting tools (``report_plan_divergence``,
        # ``declare_task_skipped``, ``declare_task_not_needed``).
        # Stored as-is so :meth:`run` step 5 can pass it through to
        # :func:`goldfive.reporting.select_reporting_tools` on every
        # turn — supports the ``True`` / ``False`` / ``list[str]``
        # shapes documented in :class:`Runner`. Materialise lists
        # eagerly so callers passing a generator / mutable list see
        # stable behaviour across turns.
        if isinstance(drift_self_reporting, bool):
            self.drift_self_reporting: bool | list[str] = drift_self_reporting
        else:
            self.drift_self_reporting = [str(n) for n in drift_self_reporting]
        # Configurable abort policy on goldfive-authored revision
        # rejection. See PLAN-LIFECYCLE.md §4.5.1. Default (kwarg
        # ``None``): consult env var; falsy unless the env explicitly
        # opts in. Explicit ``True`` / ``False`` from the kwarg wins
        # over the env so tests can pin behaviour without unsetting the
        # env first. User-authored drifts (USER_STEER) are never gated
        # by this flag — see :meth:`DefaultSteerer.install_user_steer`.
        if fail_fast_on_revision_rejection is None:
            fail_fast_on_revision_rejection = (
                os.environ.get(
                    "GOLDFIVE_FAIL_FAST_REVISION_REJECTION", "0"
                )
                == "1"
            )
        self._fail_fast_on_revision_rejection: bool = bool(
            fail_fast_on_revision_rejection
        )
        # One-shot latch for the ledger-mode / forecast-shaped-plan
        # incoherent-combo warning (see
        # :meth:`_warn_if_ledger_mode_without_ledger_plan`).
        self._ledger_shape_warned: bool = False
        # Cross-turn prior-plan stash lives on :class:`Conversation`
        # (keyed by session id) so a Runner shared across multiple
        # outer ADK sessions does not leak one session's plan into
        # another's first turn. See :meth:`Conversation.prior_plan_for`
        # and the goldfive#271 follow-up validation v4 Class 1
        # post-mortem.

    # ------------------------------------------------------------------
    # per-session Conversation lookup
    # ------------------------------------------------------------------

    def _resolve_plan_mode(self) -> str:
        """Return the steerer's plan mode ("forecast" / "ledger").

        AGENCY-PRESERVATION.md Stage 3 PR 10. The Runner threads
        ``SteeringConfig.plan_mode`` into the per-turn planner ``context``
        so :meth:`LLMPlanner.generate` / :meth:`LLMPlanner.handle_turn`
        can switch to the OUTCOME-deliverable prompts in ledger mode —
        the planner-side analogue of the pin path reading
        ``descriptive_growth_enabled`` off the steerer. Defensive: any
        read failure (custom steerer without a typed config) resolves to
        ``"forecast"``, the bit-identical default.
        """
        try:
            cfg = getattr(self.steerer, "_steering_config", None)
            if cfg is None:
                return "forecast"
            mode = str(getattr(cfg, "plan_mode", "forecast")).strip().lower()
            return "ledger" if mode == "ledger" else "forecast"
        except Exception:  # noqa: BLE001
            return "forecast"

    def _warn_if_ledger_mode_without_ledger_plan(self, session: Any) -> None:
        """One-shot WARNING for the ledger-config / forecast-plan combo.

        AGENCY-PRESERVATION.md Stage 3 PR 10 incoherent-combo guard:
        ``SteeringConfig.plan_mode == "ledger"`` resolved but the installed
        plan carries no ledger-shaped task (no ``TaskKind.OUTCOME`` /
        ``DISCOVERED``) — typically a hand-authored ``StaticPlanner``
        template under a ledger config. Per the documented contract
        ("StaticPlanner users keep forecast semantics — a hand-authored
        plan is genuine prescriptive intent") the ledger-only pin-tier
        bypass keys on the live plan's shape, so such a run keeps forecast
        pin semantics; this warning is the operator's signal that the
        configured ledger regime is not actually engaged. Fires at most
        once per Runner; never raises.
        """
        if self._ledger_shape_warned:
            return
        try:
            from goldfive.steerer import plan_mode_is_ledger
            from goldfive.types import plan_has_ledger_shape

            plan = getattr(session, "plan", None)
            tasks = getattr(plan, "tasks", None) if plan is not None else None
            if (
                plan_mode_is_ledger(self.steerer)
                and tasks
                and not plan_has_ledger_shape(plan)
            ):
                self._ledger_shape_warned = True
                log.warning(
                    "Runner.run: plan_mode=ledger is configured but the "
                    "installed plan has NO ledger-shaped task (no OUTCOME/"
                    "DISCOVERED kind) — likely a hand-authored StaticPlanner "
                    "plan. This run keeps forecast pin semantics (the "
                    "ledger pin-tier bypass keys on plan shape, not config); "
                    "use an LLMPlanner or opt tasks into the ledger taxonomy "
                    "via Task.kind to engage ledger mode. plan_id=%s tasks=%d",
                    (getattr(plan, "id", "") or "")[:16] or "<none>",
                    len(tasks),
                )
        except Exception as exc:  # noqa: BLE001 — advisory only
            log.debug(
                "Runner._warn_if_ledger_mode_without_ledger_plan raised: %s", exc
            )

    def _conversation_key(self, session_id: str | None) -> str:
        """Map an optional outer-session-id pin to the lookup key.

        ``None`` / ``""`` (the unpinned / programmatic caller) use the
        empty-string key — a single shared Conversation for that path,
        preserving pre-#161 single-Conversation continuity. Any
        non-empty pin gets its own keyed Conversation (typically the
        ``ctx.session.id`` from :class:`GoldfiveADKAgent`).
        """
        return session_id or ""

    def _conversation_for(self, key: str) -> Conversation:
        """Return the :class:`Conversation` for ``key``, creating on miss.

        ``key=""`` is the unpinned-caller slot; non-empty keys belong
        to outer ADK sessions. Created Conversations get fresh ids,
        empty goals / completed_results / turns, and a zero
        ``_next_sequence`` cursor — exactly the state a brand-new
        outer session deserves to see.
        """
        convo = self._conversations.get(key)
        if convo is None:
            convo = Conversation.new()
            self._conversations[key] = convo
            self._conversation_announced[key] = False
        return convo

    def _lock_for(self, key: str) -> asyncio.Lock:
        """Return the per-key :class:`asyncio.Lock` for serialising
        :meth:`run`, creating on miss. See ``self._convo_locks``
        commentary in :meth:`__init__` for the rationale.

        Lazy lookup pattern matches :meth:`_conversation_for`. The
        ``setdefault`` is atomic under asyncio's single-thread model
        so two concurrent first-time lookups land on the same Lock
        instance (the second discards its just-built Lock as the
        ``setdefault`` returns the existing one).
        """
        lock = self._convo_locks.get(key)
        if lock is None:
            lock = self._convo_locks.setdefault(key, asyncio.Lock())
        return lock

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ExecutionOutcome:
        """Execute one end-to-end goldfive run and return the outcome.

        ``session_id`` optionally overrides the ``Session.run_id`` /
        ``Session.id`` that :class:`Conversation.next_turn_session`
        would otherwise mint. Used by :class:`GoldfiveADKAgent` to
        adopt the outer adk-web ``InvocationContext.session.id`` so
        every goldfive Event emitted through sinks stamps the same
        session id that harmonograf spans carry (goldfive#161). Empty
        / ``None`` preserves the legacy uuid4 mint so bare programmatic
        Runner callers see no behaviour change.

        Two concurrent calls with the SAME ``session_id`` (or both
        unpinned, sharing the ``""`` key) serialise on a per-key
        :class:`asyncio.Lock` so the second turn's prior-plan
        carry-forward (Phase 4 ``handle_turn`` seeding) sees the
        first turn's post-install plan rather than racing the first's
        ``finally``-block ``Conversation.stash_plan``. See
        ``self._convo_locks`` commentary in :meth:`__init__` and the
        v7class1-1 forensic timeline in
        ``tests/test_intra_session_plan_carry_forward.py``. Concurrent
        calls on DIFFERENT keys still proceed in parallel.
        """

        # 1. Resolve the per-outer-session :class:`Conversation`
        # (goldfive#271 follow-up to PR #293). Every outer ADK session
        # gets its own Conversation so cross-turn state (goals,
        # completed_results, turns, wire-sequence cursor, prior-plan
        # stash) is isolated. The unpinned (programmatic) path uses
        # the shared ``""`` key for back-compat with pre-#161 callers.
        # See :meth:`_conversation_for` and validation v4 Class 1.
        convo_key = self._conversation_key(session_id)
        # Per-key serialisation: the lock is acquired BEFORE
        # :meth:`_conversation_for` so two concurrent first-time
        # lookups on the same key cannot both Conversation.new() and
        # write into the dict slot. The ``async with`` releases on
        # both normal return AND ``BaseException`` propagation so a
        # cancelled turn's stash always lands before the next turn's
        # seeding runs.
        async with self._lock_for(convo_key):
            return await self._run_locked(
                user_input,
                context=context,
                session_id=session_id,
                convo_key=convo_key,
            )

    async def _run_locked(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None,
        session_id: str | None,
        convo_key: str,
    ) -> ExecutionOutcome:
        """Body of :meth:`run`, executed under the per-key lock.

        Split out so :meth:`run` can wrap the entire lifecycle
        (:meth:`Conversation.next_turn_session` →
        :meth:`Planner.handle_turn` → :meth:`Executor.run` →
        ``finally``-block stash → :meth:`Conversation.absorb_turn`)
        in a single ``async with self._lock_for(convo_key):``. The
        lock prevents two concurrent ``runner.run(session_id=X)``
        calls from racing on the per-Conversation prior-plan stash;
        see the v7class1-1 demo log timeline in
        ``tests/test_intra_session_plan_carry_forward.py``.
        """
        convo = self._conversation_for(convo_key)

        # 2. Build Session seeded by the Conversation. The Session's
        #    run_id is fresh for this turn; conversation_id is stable
        #    across turns; goals / completed_results are pre-populated
        #    with prior-turn state — all scoped to the convo above.
        session = convo.next_turn_session()
        # Outer-session pin (goldfive#161): when the caller supplies a
        # non-empty ``session_id`` (typically ``ctx.session.id`` from
        # adk-web), override the freshly-minted ``run_id`` so every
        # Event emitted this turn carries that id. Sinks stamp
        # ``Event.session_id`` from ``Session.id`` (= ``run_id``), so
        # this aligns goldfive events with the ADK session that
        # harmonograf spans already target — resolving the
        # "plan view has empty Gantt" regression from the overlay
        # architecture.
        # Track whether the caller explicitly pinned the session id so
        # the Conversation's prior-plan carry-forward (and the
        # symmetric stash on absorb_turn) can apply the documented
        # carry-forward matrix — see :meth:`Conversation.prior_plan_for`
        # and goldfive#271 follow-up validation v4 Class 1.
        pinned = bool(session_id)
        if pinned:
            session.run_id = session_id  # type: ignore[assignment]
        self._last_session = session
        self._last_session_by_key[convo_key] = session

        # 2b. Announce the Conversation on its first turn (per key).
        if not self._conversation_announced.get(convo_key, False):
            await self._emit_conversation_started(session, conversation=convo)
            self._conversation_announced[convo_key] = True

        # 3. Emit RunStarted before anything else for this turn.
        await self._emit_run_started(session, user_input)

        # goldfive#215 (iter-8) P2: reset per-turn refine-outcome
        # bookkeeping immediately after the run-started boundary so
        # each turn starts with an empty outcome table. ``getattr``
        # so custom steerers without the hook degrade gracefully.
        _reset_for_turn = getattr(self.steerer, "reset_for_turn", None)
        if callable(_reset_for_turn):
            _reset_for_turn(session)

        # 3a. Seed session.plan with the prior plan (or Plan.empty()
        # on the very first turn) so :meth:`Planner.handle_turn` always
        # sees a non-None ``session.plan``. The Runner has a single
        # install path post-Phase-4: every plan landed by the planner
        # becomes the next revision of this seed (revision_index += 1).
        #
        # The prior-plan lookup combines ``session.id`` (== ``run_id``
        # after any outer-session pin above) with the caller's pin
        # state. A turn on a different outer ADK session sharing this
        # Runner (pinned-vs-pinned with mismatched ids, or a
        # pinned-prior followed by an unpinned new turn) sees
        # ``Plan.empty()`` rather than the stash from another session
        # (validation v4 Class 1 / goldfive#271 follow-up). Programmatic
        # callers (both prior and new turn unpinned) keep the
        # Conversation-level continuity the original ``_last_plan``
        # field provided.
        prior_plan = convo.prior_plan_for(session.id, pinned=pinned)
        # goldfive#247: Plan is frozen — derive a stamped variant via
        # ``dataclasses.replace`` rather than mutating in place. The
        # initial pin onto ``session.plan`` is the run-setup phase of
        # the channel-processor's mutation lifecycle, so we wrap in
        # :func:`channel_processor_active` to satisfy the runtime
        # single-writer check.
        with channel_processor_active():
            if prior_plan is not None:
                set_session_plan(
                    session,
                    dataclasses.replace(prior_plan, run_id=session.run_id),
                )
            else:
                set_session_plan(session, Plan.empty(run_id=session.run_id))
        _ostate.set_current_plan(session.state, session.plan)

        # 4. Derive (or accept) goals for this turn. Cross-turn state
        #    lives on ``session.goals`` already (seeded by the
        #    Conversation); we append newly-derived goals that weren't
        #    already present by id so the planner sees the full history.
        try:
            new_goals = await self._resolve_goals(user_input, context, session=session)
        except Exception as exc:  # noqa: BLE001
            log.exception("goal derivation failed")
            return await self._abort_turn(
                session=session,
                convo=convo,
                user_input=user_input,
                pinned=pinned,
                reason=f"goal derivation failed: {exc}",
            )

        # F9 (closes goldfive#322 Layer 4): mint fresh goal ids when
        # the deriver's LLM-supplied id collides with an existing
        # session goal. The deriver's prompt prompts ``g1`` every
        # turn; the prior dedup-on-collision path silently dropped
        # legitimate new goals, so multi-turn sessions accumulated
        # only the first turn's goals. Renumber the new goal so it
        # lands in ``session.goals`` instead of being discarded.
        #
        # Renumbering is done by replacing the Goal with a fresh
        # ``dataclasses.replace`` copy so we never mutate the
        # deriver's stored list in place — some derivers (e.g.
        # :class:`PassthroughGoalDeriver`) return shallow copies
        # that share Goal objects with their internal cache; an
        # in-place mutation would silently rewrite the deriver's
        # state for subsequent turns.
        existing_ids = {g.id for g in session.goals if g.id}
        next_seq = len(session.goals) + 1
        for g in new_goals:
            if g.id and g.id in existing_ids:
                while True:
                    candidate = f"g{next_seq}"
                    next_seq += 1
                    if candidate not in existing_ids:
                        break
                log.info(
                    "Runner.run: goal id collision (%r) — renumbering to %r",
                    g.id,
                    candidate,
                )
                g = dataclasses.replace(g, id=candidate)
            session.goals.append(g)
            if g.id:
                existing_ids.add(g.id)
                next_seq = max(next_seq, len(session.goals) + 1)

        # goldfive#152: refresh the orchestration-state goals summary
        # so prompt templates / handle_turn / downstream planners see
        # an up-to-date ``goldfive.goals_summary``.
        _ostate.refresh_goals_summary(session.state, session.goals)

        await self._emit_goal_derived(session)

        # 4a. Per-turn planner decision (goldfive#271 Phase 4).
        # ``handle_turn`` is a single LLM call that decides whether
        # the new user_input warrants a plan change and, when it
        # does, produces the next revision of session.plan in one
        # shot. Replaces the prior multi-stage pipeline (regex
        # short-circuits + LLM gate + synthesize_goal_from_steer +
        # qualification-merge regex + planner.refine). All plan
        # changes are revisions: the conversation's plan_id is
        # stable, revision_index increments monotonically.
        #
        # Returns ``None`` when the user_input is purely
        # conversational and the current revision still describes
        # the right work — the Runner reuses ``session.plan`` for
        # this turn (driving the executor over the existing plan).
        #
        # Returns the next ``Plan`` revision when a change is
        # warranted — the Runner installs it via the unified
        # ``_install_revision`` path so PlanRevised fires uniformly.
        #
        # Skipped when ``user_input`` is already a ``list[Goal]``
        # (caller has opted out of NL derivation), when
        # ``planner_gate=None`` (deterministic replay mode), and
        # when the planner doesn't implement ``handle_turn`` (legacy
        # PassthroughPlanner / third-party stubs — Runner falls
        # through to ``planner.generate`` once for back-compat).
        next_plan: Plan | None = None
        decided = False
        if (
            self._planner_gate is not None
            and isinstance(user_input, str)
            and hasattr(self.planner, "handle_turn")
        ):
            try:
                next_plan = await self._invoke_handle_turn(
                    user_input=user_input,
                    session=session,
                    context=context,
                    conversation=convo,
                )
                decided = True
                log.info(
                    "Runner.run: handle_turn produced_plan=%s "
                    "(source=runner-inline) prior_plan_id=%s "
                    "user_input_first=%r",
                    "yes" if next_plan is not None else "no",
                    (session.plan.id or "")[:16] or "<none>",
                    user_input[:80],
                )
            except Exception as exc:  # noqa: BLE001
                # A misbehaving handle_turn must never break the run;
                # fall through to generate (legacy first-turn path).
                log.warning(
                    "planner.handle_turn raised; falling through to generate: %s",
                    exc,
                )
                decided = False

        # If handle_turn was skipped, raised, OR the planner doesn't
        # implement handle_turn meaningfully (returns None on the
        # very first turn against an empty seed — true for
        # PassthroughPlanner / StaticPlanner / non-LLM planners),
        # fall through to ``planner.generate`` so a brand-new plan
        # still lands. ``planner.generate`` is the legacy path the
        # Runner used pre-Phase-4; preserved for back-compat with
        # planners that don't implement Phase 4's per-turn LLM call.
        first_turn_seed = not session.plan.tasks
        needs_generate_fallback = (not decided) or (
            decided and next_plan is None and first_turn_seed
        )
        if needs_generate_fallback:
            available_agents: Any
            tree = getattr(self.agent, "available_agents_tree", None)
            if isinstance(tree, list) and tree:
                available_agents = list(tree)
            else:
                available_agents = list(self.agent.available_agents)
            planner_context: dict[str, Any] = {
                "run_id": session.run_id,
                "max_task_invocations": self.max_task_invocations,
            }
            planner_context.update(convo.prior_turn_context())
            if context:
                planner_context.update(context)
            planner_context["run_id"] = session.run_id
            # AGENCY-PRESERVATION.md Stage 3 PR 10 — surface the plan mode
            # to the planner so ledger mode produces OUTCOME deliverables.
            # Set last so a caller-supplied context cannot silently
            # override the steerer-configured mode.
            planner_context["plan_mode"] = self._resolve_plan_mode()
            try:
                next_plan = await self.planner.generate(
                    goals=session.goals,
                    available_agents=available_agents,
                    context=planner_context,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("planner.generate raised")
                return await self._abort_turn(
                    session=session,
                    convo=convo,
                    user_input=user_input,
                    pinned=pinned,
                    reason=f"planner.generate raised: {exc}",
                )

        # Install the produced plan as the next revision of
        # session.plan, OR (when next_plan is None and a real prior
        # exists) reuse session.plan unchanged so the executor drives
        # the coordinator over existing context.
        if next_plan is not None:
            installed = await self._install_revision(
                session=session,
                user_input=user_input,
                revised_plan=next_plan,
            )
            if not installed:
                # Configurable abort policy on goldfive-authored
                # revision rejection (PLAN-LIFECYCLE.md §4.5.1). The
                # install path here is always goldfive-authored — it
                # handles handle_turn-driven NEW_WORK_DISCOVERED
                # revisions; genuine operator USER_STEER takes the
                # executor's STEER control loop and routes through the
                # steerer's user-steer pipeline (which is structurally
                # incapable of aborting — see PLAN-LIFECYCLE.md
                # §4.2.1 and ``DefaultSteerer.install_user_steer``).
                #
                # Default (``fail_fast_on_revision_rejection=False``):
                # keep ``session.plan`` unchanged, emit a
                # HUMAN_INTERVENTION_REQUIRED INFO drift for
                # observability, and continue the turn. The plan that
                # was previously valid is still valid; agents can
                # still make progress; the next refine cycle (if the
                # underlying drift persists) gets another attempt.
                # The existing ``REFINE_FAILURE_THRESHOLD=2``
                # escalation in :meth:`DefaultSteerer._install_with_drift`
                # still fires after two consecutive failures of the
                # same ``(kind, task_id)``.
                #
                # Opt-in strict mode (``fail_fast_on_revision_rejection
                # =True`` or ``GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1``):
                # emit ``run_aborted`` as the pre-PR1 behaviour. Useful
                # for CI / regression / debugging refine logic.
                log.warning(
                    "Runner.run: goldfive-authored revision rejected "
                    "by validator; keeping existing plan, continuing "
                    "run (set fail_fast_on_revision_rejection=True or "
                    "GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1 for "
                    "strict mode). prior_plan_id=%s revision_index=%d",
                    (session.plan.id or "")[:16] or "<none>",
                    int(getattr(session.plan, "revision_index", 0)),
                )
                # Observability drift on the same sinks the steerer
                # uses, so harmonograf / sinks see the rejection
                # alongside the surrounding refine activity. Routed
                # through the steerer's ``_emit_drift_detected`` so
                # the lifecycle / condition_id stamping fires
                # consistently.
                obs_drift = DriftEvent(
                    kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
                    severity=DriftSeverity.INFO,
                    detail=(
                        "autonomous refine produced an invalid plan "
                        "revision; existing plan retained; next "
                        "refine cycle may try again"
                    ),
                    authored_by="goldfive",
                )
                emit_helper = getattr(
                    getattr(self.steerer, "drift", None),
                    "_emit_drift_detected",
                    None,
                )
                if callable(emit_helper):
                    try:
                        await emit_helper(session, obs_drift)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "Runner.run: emitting "
                            "HUMAN_INTERVENTION_REQUIRED observability "
                            "drift raised: %s",
                            exc,
                        )

                if self._fail_fast_on_revision_rejection:
                    return await self._abort_turn(
                        session=session,
                        convo=convo,
                        user_input=user_input,
                        pinned=pinned,
                        reason="plan revision rejected by validator",
                    )
                # Default: keep existing plan, continue. ``session.plan``
                # is unchanged because ``_install_revision``'s
                # rejection path returns False BEFORE applying the
                # revision (see :meth:`_install_with_drift` —
                # ``revised_plan.validate`` raises before
                # ``_apply_revision`` runs).
        elif not session.plan.tasks:
            # First turn AND handle_turn returned None (purely
            # conversational on an empty seed). No plan to drive the
            # executor over — abort cleanly.
            return await self._abort_turn(
                session=session,
                convo=convo,
                user_input=user_input,
                pinned=pinned,
                reason="no plan generated",
            )
        else:
            # Conversational follow-up on a real prior plan. Reuse
            # session.plan unchanged. No PlanRevised — the prior
            # revision is still the right one for this turn.
            #
            # F6 (closes goldfive#277): mark this turn as a
            # conversational follow-up so the executor handoff below
            # wraps ``user_input`` with the prior-plan context via
            # :meth:`PromptShaper.wrap_conversational_input`. The wrap
            # lives in the message body — NOT the system prompt — to
            # preserve the no-prompt-contract principle: users bring
            # their own coordinator prompts; goldfive must not require a
            # specific system-prompt contract. (Historical note: a
            # docstring once described a parallel ADK-plugin "pre-dispatch
            # interceptor" keyed off this flag that tightened the tool
            # surface; that interceptor was never built — the flag's only
            # consumer is the runner's own wrap gating below. Verified
            # AGENCY-PRESERVATION.md PR 9.)
            log.info(
                "Runner.run: conversational turn — reusing prior plan_id=%s revision_index=%d",
                (session.plan.id or "")[:16] or "<none>",
                int(session.plan.revision_index),
            )
            session._conversational_turn = True  # type: ignore[attr-defined]

        # Incoherent-combo guard (AGENCY-PRESERVATION.md PR 10): warn once
        # when plan_mode=ledger resolved but the plan that just landed has
        # no ledger-shaped task — that run keeps forecast pin semantics.
        self._warn_if_ledger_mode_without_ledger_plan(session)

        # 5. Register the canonical reporting tools on the adapter.
        # goldfive#196: ``drift_self_reporting`` decides whether the
        # drift opinions (``report_plan_divergence``,
        # ``declare_task_skipped``, ``declare_task_not_needed``) ride
        # along with the lifecycle subset. Default is OFF — the
        # framework's observation paths (``classify_goal_drift``,
        # :class:`~goldfive.reconciler.PlanReconciler`, the steerer's
        # refine machinery) are the canonical detectors.
        try:
            await self.agent.register_reporting_tools(
                select_reporting_tools(self.drift_self_reporting)
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("register_reporting_tools raised")
            return await self._abort_turn(
                session=session,
                convo=convo,
                user_input=user_input,
                pinned=pinned,
                reason=f"register_reporting_tools raised: {exc}",
            )

        # 6. Bind the steerer. (The executor may re-bind — that's fine.)
        try:
            self.steerer.bind(sinks=list(self.sinks), planner=self.planner)
        except Exception as exc:  # noqa: BLE001
            log.exception("steerer.bind raised")
            return await self._abort_turn(
                session=session,
                convo=convo,
                user_input=user_input,
                pinned=pinned,
                reason=f"steerer.bind raised: {exc}",
            )

        # 6b. Wire the steerer into the adapter. Adapter plugin callbacks
        # (e.g. ADKAdapter's ``_emit_observability``) short-circuit when
        # ``SessionContext.steerer`` is ``None`` — without this call the
        # new sink events (``AgentInvocationStarted`` /
        # ``AgentInvocationCompleted`` / ``DelegationObserved``) never
        # fire. Every built-in adapter (ADK, Claude, Callable) exposes
        # ``bind_steerer``; probe with getattr so third-party adapters
        # that predate the protocol addition still work.
        bind_adapter_steerer = getattr(self.agent, "bind_steerer", None)
        if bind_adapter_steerer is not None:
            try:
                bind_adapter_steerer(self.steerer)
            except Exception as exc:  # noqa: BLE001
                log.exception("adapter.bind_steerer raised")
                return await self._abort_turn(
                    session=session,
                    convo=convo,
                    user_input=user_input,
                    pinned=pinned,
                    reason=f"adapter.bind_steerer raised: {exc}",
                )

        # 6c. Wire the adapter back into the steerer. Optional hook
        # (goldfive#139) the steerer uses to tag the adapter's next
        # mid-invocation cancel with a symbolic reason on USER_STEER
        # drift, so the synthetic function_response the adapter appends
        # on cancel carries LLM-actionable content. Duck-typed on
        # purpose — custom Steerers that don't implement
        # ``bind_adapter`` skip this silently.
        bind_steerer_adapter = getattr(self.steerer, "bind_adapter", None)
        if callable(bind_steerer_adapter):
            try:
                bind_steerer_adapter(self.agent)
            except Exception as exc:  # noqa: BLE001
                log.debug("steerer.bind_adapter raised: %s", exc)

        # 6d. Wire the control channel into the steerer (Phase 2 of
        # the path-duality fix). The steerer mints
        # ``GOLDFIVE_STEER`` and ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessages
        # onto this channel so goldfive-authored drift rides the same
        # cancel-and-restart junction as user-authored ``STEER`` /
        # ``PAUSE``. Duck-typed on purpose — custom Steerers that
        # don't implement ``bind_control_channel`` skip this silently;
        # their CANCEL_REINVOKE / PAUSE_ESCALATE paths fall back to
        # the originating drift event on the sink stream.
        bind_steerer_channel = getattr(self.steerer, "bind_control_channel", None)
        if callable(bind_steerer_channel):
            try:
                bind_steerer_channel(self._control)
            except Exception as exc:  # noqa: BLE001
                log.debug("steerer.bind_control_channel raised: %s", exc)

        # 7. Hand off to the executor.
        # Phase 3.5 (goldfive#271): wrap the executor.run call site
        # in the cancellation-stash audit context so the boundary's
        # tripwire can verify the prior-plan stash fired before the
        # cancel propagated past us. The compliance branch lives in
        # the ``finally`` block below — it calls
        # ``mark_stash_completed`` after performing the stash.
        with _state_audit.cancellation_stash_audited("Runner.run.executor_drive"):
            try:
                executor_kwargs: dict[str, Any] = dict(
                    plan=session.plan,
                    session=session,
                    adapter=self.agent,
                    steerer=self.steerer,
                    planner=self.planner,
                    sinks=list(self.sinks),
                )
                if self.control is not None:
                    executor_kwargs["control"] = self.control
                # Overlay model (goldfive#141): pass the original user
                # request through to the executor so an overlay-capable
                # :class:`SequentialExecutor` can hand it verbatim to
                # ``adapter.invoke_passthrough``. Best-effort via
                # inspection — executors that don't accept
                # ``user_input=`` keep working with the legacy kwargs.
                #
                # F6 (goldfive#277): on a conversational turn, hand the
                # executor the wrapped directive (frames the input as a
                # follow-up question, asks the coordinator not to
                # delegate). The wrapping lives at the executor handoff
                # so absorb_turn / event summaries above still see the
                # user's actual question.
                #
                # goldfive#271 strict-passive carve-out + Wave B1: the
                # wrap + ``observation_only`` gate live in
                # :class:`~goldfive.prompt_shaper.PromptShaper` so the
                # four prompt-shaping sites share one gate. Under
                # ``observation_only=True`` ``wrap_conversational_input``
                # returns the raw input unchanged; otherwise it returns
                # the composed F6 directive.
                if isinstance(user_input, str):
                    run_sig = inspect.signature(self.executor.run)
                    if "user_input" in run_sig.parameters:
                        executor_user_input = user_input
                        if (
                            getattr(session, "_conversational_turn", False)
                            and user_input.strip()
                        ):
                            executor_user_input = self._prompt_shaper.wrap_conversational_input(
                                user_input=user_input,
                                session=session,
                                steerer=self.steerer,
                            )
                        executor_kwargs["user_input"] = executor_user_input
                outcome = await self.executor.run(**executor_kwargs)
            except Exception as exc:  # noqa: BLE001
                log.exception("executor.run raised")
                return await self._abort_turn(
                    session=session,
                    convo=convo,
                    user_input=user_input,
                    pinned=pinned,
                    reason=f"executor.run raised: {exc}",
                )
            finally:
                # planner-gate: snapshot the turn's final plan so the next
                # turn's planner_gate can classify against it.
                #
                # Rationale (Phase 2.X v2 / goldfive#271 Gap 1): the prior
                # post-success-path stash (PR #282) was bypassed when ADK
                # closed the runner mid-flight — the executor coroutine was
                # cancelled, ``CancelledError`` propagated out of
                # ``await self.executor.run(...)``, and since Py 3.8
                # ``CancelledError`` is a ``BaseException`` (not an
                # ``Exception``) the ``except Exception`` handler above did
                # NOT catch it. Control flowed out of ``run`` entirely and
                # the stash was skipped, leaving the prior plan empty for
                # the next turn even though the turn produced a real plan.
                # The ADK-web user-steer flow hit this on validation v2:
                # zero stash log lines across 4 turns.
                #
                # Putting the stash in ``finally`` runs it regardless of how
                # the executor exited — normal success, ``Exception`` (e.g.
                # planner bind error), or ``BaseException`` (e.g.
                # ``CancelledError`` from ADK closing the runner mid-stream).
                # The exception still propagates after the stash; this block
                # does not swallow it.
                #
                # The stash itself lives on :class:`Conversation`
                # (validation v4 Class 1 / goldfive#271 follow-up): scoping
                # by session id means a turn on a fresh outer ADK session
                # sharing this Runner does not inherit a prior plan from
                # another session. :meth:`Conversation.absorb_turn` (called
                # on every normal-completion / handled-exception return
                # path below) folds the stash in alongside the goals /
                # completed_results merge. The explicit ``stash_plan`` call
                # here covers the ``BaseException`` (e.g. ``CancelledError``
                # from ADK closing the runner mid-stream) path that bypasses
                # ``absorb_turn`` entirely — the same rationale as the
                # original Gap 1 fix that put the stash in ``finally`` to
                # begin with. ``stash_plan`` is idempotent: a subsequent
                # ``absorb_turn`` re-stashes the same plan + session id.
                if session.plan is not None and session.plan.tasks:
                    convo.stash_plan(session, pinned=pinned)
                    log.info(
                        "Runner.run: stashed prior plan for next turn's "
                        "handle_turn (plan_id=%s revision_index=%d "
                        "session_id=%s)",
                        (session.plan.id or "")[:16] or "<none>",
                        int(session.plan.revision_index),
                        (session.id or "")[:16] or "<none>",
                    )
                # Phase 3.5 (goldfive#271) tripwire compliance marker:
                # the stash above ran (idempotent and unconditional
                # within this block). The boundary catch site at
                # ``ADKAdapter._invoke_internal`` will assert this
                # marker fired before ``CancelledError`` propagated
                # past us.
                _state_audit.mark_stash_completed()

        # goldfive#152: clear the current_task_* stamp at run end.
        # The plan id + goals summary stay (they remain meaningful
        # cross-turn on the owning Conversation).
        _ostate.clear_current_task(session.state)
        _ostate.clear_active_steer(session.state)

        convo.absorb_turn(
            outcome,
            user_input_summary=_initial_goal_summary(user_input),
            pinned=pinned,
        )
        return outcome

    # ------------------------------------------------------------------
    # run_streamed — yield inner-adapter framework events in real time
    # ------------------------------------------------------------------

    async def run_streamed(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Any]:
        """Execute a run and stream inner-adapter framework events out as they arrive.

        Async generator that yields — in order — every framework-native
        event the adapter observes mid-invocation (e.g. ADK ``Event``
        objects: ``transfer_to_agent``, model text parts, function
        calls, function responses) followed by exactly one trailing
        :class:`~goldfive.results.ExecutionOutcome` as the final
        yielded element when the run finishes.

        The trailing ``ExecutionOutcome`` is how callers recover the
        completed run's success flag, reason, and live
        :class:`~goldfive.types.Session`. Consumers distinguish the
        two yielded shapes via ``isinstance(item, ExecutionOutcome)``.

        The equivalent of :meth:`run` — same lifecycle, same sinks, same
        plugin callbacks, same conversation bookkeeping — is driven in
        the background. :meth:`run_streamed` does NOT call :meth:`run`
        recursively; it subscribes to the adapter's event fan-out
        (:meth:`ADKAdapter.subscribe_adk_events`, when available) and
        forwards those events through an :class:`asyncio.Queue` so
        backpressure from the consumer cannot stall the inner Runner.

        Non-ADK adapters (callable, Claude SDK) have no streamable
        framework events; :meth:`run_streamed` still works for them —
        it simply yields no mid-run events and produces the outcome at
        the end, exactly as :meth:`run` would. Callers do not need to
        switch on adapter type.

        This is the primary path used by
        :class:`~goldfive.adapters.adk_wrap.GoldfiveADKAgent` so
        ``adk web`` sees per-agent activity (LLM responses, tool calls,
        agent transitions) in its UI while the goldfive pipeline runs.

        Parameters mirror :meth:`run` — see that docstring for
        ``session_id`` semantics.
        """
        # Import here so non-ADK consumers don't pay the optional-ADK
        # import cost when they never call run_streamed.
        queue: asyncio.Queue[Any] = asyncio.Queue()

        # Subscribe a sync listener to the adapter's raw-event fan-out
        # when the adapter supports it. The listener enqueues every
        # event into ``queue`` for us to re-yield. Non-ADK adapters
        # simply don't expose ``subscribe_adk_events`` — the run still
        # completes and we yield only the final outcome.
        subscribe = getattr(self.agent, "subscribe_adk_events", None)
        unsubscribe = getattr(self.agent, "unsubscribe_adk_events", None)

        def _listener(event: Any) -> None:
            # ``put_nowait`` is correct here: the queue is unbounded
            # so it never raises, and we must NOT block the adapter's
            # event loop on a consumer that's slow to pull.
            try:
                queue.put_nowait(event)
            except Exception:  # noqa: BLE001 — defensive; unbounded queue shouldn't raise
                log.debug("run_streamed: queue.put_nowait unexpectedly raised")

        if callable(subscribe):
            subscribe(_listener)

        # Sentinel that tells the consumer loop the run is done and
        # any remaining events have already been enqueued.
        _DONE = object()

        async def _drive() -> ExecutionOutcome:
            try:
                return await self.run(
                    user_input,
                    context=context,
                    session_id=session_id,
                )
            finally:
                # Signal end-of-stream regardless of success / failure.
                # The consumer drains any remaining buffered events,
                # then stops when it sees the sentinel.
                queue.put_nowait(_DONE)

        run_task: asyncio.Task[ExecutionOutcome] = asyncio.create_task(_drive())

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                yield item
            outcome = await run_task
            yield outcome
        except (asyncio.CancelledError, GeneratorExit):
            # Upstream cancelled us (adk-web disconnect) OR the caller
            # aclose()'d the generator early. Propagate the cancel into
            # the driver so its ``try/finally`` teardown runs, then
            # await it to collect the final state — suppressing the
            # CancelledError so the generator exits cleanly.
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            raise
        finally:
            if callable(unsubscribe):
                try:
                    unsubscribe(_listener)
                except Exception as exc:  # noqa: BLE001
                    log.debug("run_streamed: unsubscribe raised: %s", exc)

    # ------------------------------------------------------------------
    # resume — best-effort replay
    # ------------------------------------------------------------------

    async def resume(self, persistence_path: str) -> ExecutionOutcome:
        """Replay a JSONL persistence log and return the recovered outcome.

        Best-effort for v0.1: uses ``goldfive.sinks.reconstruct_session``
        when the proto stubs are available (JSONL events are proto-
        encoded). Returns the reconstructed session as an
        :class:`ExecutionOutcome` reflecting the latest terminal marker
        (``RunCompleted`` / ``RunAborted``) seen in the log.

        We do **not** continue execution from the latest cursor — full
        resume semantics require planner/executor co-operation that is
        out-of-scope for this PR. Callers who need a live continuation
        should construct a new :class:`Runner` with the goals recovered
        from the log.

        TODO(#15): once executors grow a ``resume_from`` hook, continue
        execution from the latest un-finished task rather than returning
        the recovered session as-is.
        """
        try:
            from goldfive.sinks import reconstruct_session, replay_from_jsonl
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "goldfive.sinks.reconstruct_session is not available; "
                "install the `proto` extra to enable JSONL replay."
            ) from exc

        events = replay_from_jsonl(persistence_path)
        session = reconstruct_session(events)

        success = False
        reason = "run did not complete before persistence ended"
        for evt in events:
            payload = getattr(evt, "WhichOneof", lambda _: None)("payload")
            if payload == "run_completed":
                success = True
                reason = ""
            elif payload == "run_aborted":
                success = False
                reason = getattr(evt.run_aborted, "reason", "")

        return ExecutionOutcome(success=success, session=session, reason=reason)

    # ------------------------------------------------------------------
    # cross-turn conversation
    # ------------------------------------------------------------------

    @property
    def conversation_id(self) -> str:
        """The default (unpinned-key) Conversation's stable id.

        Returns the id of the empty-string-keyed :class:`Conversation`
        — the slot used by programmatic / unpinned :meth:`run` callers.
        Each pinned outer ADK session owns its own Conversation
        (look it up via ``runner._conversations[session_id]``); the
        public property keeps single-Conversation semantics for
        backward-compatibility with pre-#161 callers and inspection
        tools.
        """
        return self._conversations[""].id

    @property
    def conversation(self) -> Conversation:
        """The default (unpinned-key) :class:`Conversation`.

        Read-only handle for inspection. Per-pinned-session
        Conversations live under ``runner._conversations``; this
        property returns the empty-string-keyed slot used by
        programmatic callers, preserving the pre-#293 single-
        Conversation public surface.
        """
        return self._conversations[""]

    async def new_conversation(self, *, reason: str = "") -> None:
        """Reset cross-turn state across every per-session Conversation.

        Emits a ``ConversationEnded`` event for every outgoing
        Conversation that had announced its start (one per outer
        session id observed so far), then installs fresh
        Conversations. The next :meth:`run` for any session id —
        pinned or unpinned — starts a brand-new Conversation; its
        ``ConversationStarted`` is emitted lazily on that call.
        """
        for key, outgoing in list(self._conversations.items()):
            announced = self._conversation_announced.get(key, False)
            anchor = self._last_session_by_key.get(key)
            if announced and anchor is not None:
                await self._emit_conversation_ended(
                    conversation=outgoing,
                    session_anchor=anchor,
                    reason=reason or "new_conversation",
                )
        # Reset the per-session bookkeeping. The default ("") slot is
        # restored eagerly so the public ``conversation`` /
        # ``conversation_id`` properties keep returning a real handle
        # right after the reset (callers may inspect them before the
        # next :meth:`run`). Pinned slots are recreated lazily by
        # :meth:`_conversation_for` on the next run for that session.
        self._conversations = {"": Conversation.new()}
        self._conversation_announced = {"": False}
        self._last_session_by_key = {}
        self._last_session = None
        # planner-gate: reset turn-aware gate state so the first turn
        # of the new conversation runs full planning. Cross-turn
        # plan stash now lives on :class:`Conversation` (keyed by
        # session id; see goldfive#271 follow-up); the fresh
        # ``Conversation.new()`` above already starts with
        # ``_last_plan = None``, so no Runner-side reset is needed.

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close every sink, then invoke registered close hooks. Idempotent.

        Emits ``ConversationEnded`` for the active Conversation (if any
        turns ran) before closing sinks, so persisted logs have a clean
        terminal marker. Close hooks registered via
        :meth:`add_close_hook` run in registration order AFTER sinks are
        closed; a raising hook is logged and does not prevent subsequent
        hooks from running. A second call to :meth:`close` is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        # Emit ConversationEnded for every per-session Conversation
        # that announced its start. Pinned (ADK-session) and unpinned
        # slots both flow through here so persisted logs always carry
        # a clean terminal marker for each conversation_id observed.
        for key, conv in list(self._conversations.items()):
            announced = self._conversation_announced.get(key, False)
            anchor = self._last_session_by_key.get(key)
            if not (announced and anchor is not None):
                continue
            # goldfive#212: audit and cancel any orphan PENDING tasks
            # BEFORE the ConversationEnded marker. At conversation end
            # there is no next turn that could engage them, so every
            # PENDING task is by definition orphaned (no reachability
            # split applies — that's a per-turn concern). The audit is
            # idempotent: ``mark_task_cancelled`` no-ops on terminal
            # tasks, and the outer ``self._closed`` gate prevents
            # double-firing across repeated ``close()`` calls.
            try:
                await self._audit_conversation_pending_at_close(
                    conversation=conv,
                    anchor=anchor,
                    key=key,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "conversation-end orphan-PENDING audit raised: %s", exc
                )
            try:
                await self._emit_conversation_ended(
                    conversation=conv,
                    session_anchor=anchor,
                    reason="runner_close",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("conversation_ended emission raised: %s", exc)
            self._conversation_announced[key] = False
        # Drain background reasoning-judge tasks the steerer scheduled
        # via its fire-and-forget judge path (goldfive#251). Bounded
        # shutdown so a hung LLM judge doesn't stall close. Duck-typed
        # — custom steerers without ``shutdown`` fall through cleanly.
        steerer_shutdown = getattr(self.steerer, "shutdown", None)
        if callable(steerer_shutdown):
            try:
                await steerer_shutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning("steerer.drift.shutdown() raised: %s", exc)
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("sink.close() raised: %s", exc)
        # Auto-close LLM callables on the planner and goal-deriver if they
        # implement the optional ``close`` shape (see goldfive._llm).
        # Standard SDK clients (OpenAI AsyncOpenAI, ADK LiteLlm, …) own
        # an aiohttp session that leaks unless explicitly closed.
        await maybe_close_call_llm(
            getattr(self.planner, "_call_llm", None), label="planner.call_llm"
        )
        await maybe_close_call_llm(
            getattr(self.goal_deriver, "_call_llm", None),
            label="goal_deriver.call_llm",
        )
        for hook in self._close_hooks:
            try:
                await hook()
            except Exception as exc:  # noqa: BLE001
                log.warning("close hook raised: %s", exc)

    # ------------------------------------------------------------------
    # Extension API — post-construction wiring for sinks, hooks, control
    # ------------------------------------------------------------------

    def add_sink(self, sink: EventSink) -> None:
        """Register an additional :class:`EventSink`.

        Takes effect for events emitted by subsequent calls to
        :meth:`run`. In-flight runs continue with whatever sink list
        they were handed to the executor at kickoff.
        """
        self.sinks.append(sink)

    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """Register an async callable invoked by :meth:`close` after sinks.

        Hooks fire in registration order, AFTER the Runner's internal
        teardown (sinks closed). An exception in one hook is logged
        via :mod:`logging` and does not prevent subsequent hooks from
        running — failing cleanup must not hang a process.
        """
        self._close_hooks.append(hook)

    @property
    def control(self) -> ControlChannel | None:
        """The attached :class:`~goldfive.control.ControlChannel`, if any."""
        return self._control

    @control.setter
    def control(self, value: ControlChannel) -> None:
        """Attach a :class:`ControlChannel` post-construction.

        Idempotent when the same channel (by identity, ``is``) is
        re-attached. Raises :class:`RuntimeError` if a different
        channel is already attached — callers must construct a fresh
        Runner rather than swap channels mid-lifetime.
        """
        if self._control is value:
            return
        if self._control is not None:
            raise RuntimeError(
                "Runner already has a control channel attached; "
                "detach it first or construct the runner with a specific one."
            )
        self._control = value

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _invoke_handle_turn(
        self,
        *,
        user_input: str,
        session: Session,
        context: Mapping[str, Any] | None,
        conversation: Conversation,
    ) -> Plan | None:
        """Invoke ``planner.handle_turn`` with the runner's per-turn context.

        Goldfive#271 Phase 4 entry point. The planner reads the prior
        plan + goals off ``session.plan`` / ``session.goals``; the
        Runner threads the available agents and the per-turn context
        (run_id, max_task_invocations, prior_turns) so the planner has
        everything it needs in one call.

        ``conversation`` is the per-outer-session :class:`Conversation`
        the caller resolved at :meth:`run` entry — passed in rather
        than read off ``self`` because the Runner now holds a dict of
        per-session Conversations (see :meth:`_conversation_for`).
        """
        # Prefer the richer tree shape (goldfive#151) when the adapter
        # exposes it. Adapters that don't implement the property fall
        # through to the legacy flat list — keeps back-compat.
        available_agents: Any
        tree = getattr(self.agent, "available_agents_tree", None)
        if isinstance(tree, list) and tree:
            available_agents = list(tree)
        else:
            available_agents = list(self.agent.available_agents)
        planner_context: dict[str, Any] = {
            "run_id": session.run_id,
            "max_task_invocations": self.max_task_invocations,
        }
        planner_context.update(conversation.prior_turn_context())
        if context:
            planner_context.update(context)
        planner_context["run_id"] = session.run_id
        # AGENCY-PRESERVATION.md Stage 3 PR 10 — surface the plan mode to
        # the planner so ledger mode produces OUTCOME-deliverable
        # revisions. Set last so caller context cannot override it.
        planner_context["plan_mode"] = self._resolve_plan_mode()
        return await self.planner.handle_turn(
            user_input=user_input,
            session=session,
            conversation_history=list(conversation.turns),
            available_agents=available_agents,
            context=planner_context,
        )

    async def _install_revision(
        self,
        *,
        session: Session,
        user_input: str | list[Goal],
        revised_plan: Plan,
    ) -> bool:
        """Install ``revised_plan`` as the next revision of ``session.plan``.

        Goldfive#271 Option A: dispatches across two steerer APIs
        based on what's actually happening:

        * Turn 1 install (``session.plan`` is the :meth:`Plan.empty`
          seed) → :meth:`DefaultSteerer.install_initial_plan`. No
          ``DriftDetected`` is emitted — installing the first plan
          is structural, not an intervention. (Eliminates the
          synthetic ``USER_STEER`` drift fabricated by the pre-Option-A
          path.)
        * Turn N+1 LLM-driven replan (``planner.handle_turn``
          returned a new revision in response to a fresh user
          message) → :meth:`DefaultSteerer.install_revision_for_drift`
          with a ``NEW_WORK_DISCOVERED`` drift. The user genuinely
          surfaced new work that prompted the replan; the drift is
          real, not synthetic.

        Genuine operator STEER ``ControlMessage`` deliveries do **not**
        flow through this method — they take the executor's steer
        loop and go straight to
        :meth:`DefaultSteerer.install_revision_for_user_steer`.

        Returns ``True`` on success, ``False`` on validation failure
        (the caller surfaces RunAborted).
        """
        # Bind the steerer + adapter so the install pipeline has
        # sinks + planner + adapter wiring. bind() is idempotent —
        # the executor handoff below re-binds with the same args.
        try:
            self.steerer.bind(sinks=list(self.sinks), planner=self.planner)
            bind_adapter_steerer = getattr(self.agent, "bind_steerer", None)
            if bind_adapter_steerer is not None:
                bind_adapter_steerer(self.steerer)
            bind_steerer_adapter = getattr(self.steerer, "bind_adapter", None)
            if callable(bind_steerer_adapter):
                bind_steerer_adapter(self.agent)
        except Exception as exc:  # noqa: BLE001
            log.warning("Runner._install_revision: steerer/adapter bind raised: %s", exc)
            return False
        # Stamp run_id on the revised plan so sink emissions correlate
        # with this turn. goldfive#247: Plan is frozen — derive a new
        # instance with the run_id stamped rather than mutating in
        # place.
        if not revised_plan.run_id:
            revised_plan = dataclasses.replace(revised_plan, run_id=session.run_id)
        # Branch on first-turn vs pivot vs replan.
        #
        # * First turn — ``session.plan`` is the empty seed (no tasks);
        #   route through ``install_initial_plan`` (no Rule 6 binding).
        # * Pivot turn (F5, goldfive#322 Layer 2 / #204) — the planner's
        #   ``handle_turn`` set ``_goldfive_pivot`` on the revised plan
        #   to signal that the user is replacing prior intent rather
        #   than building on it. The plan already carries a fresh
        #   plan_id (minted in ``_parse_handle_turn_response``); route
        #   through ``install_initial_plan`` so Rule 6 doesn't reject
        #   the legitimate pivot for "dropping" the prior's terminal
        #   tasks.
        # * Otherwise — replan via ``install_revision_for_drift`` with
        #   a NEW_WORK_DISCOVERED drift (the existing path).
        #
        # ``session.plan`` is non-None on every turn (the Runner seeds
        # Plan.empty on turn 1); the absence of tasks is the
        # unambiguous "first install" signal.
        first_turn = session.plan is None or not session.plan.tasks
        is_pivot = bool(getattr(revised_plan, "_goldfive_pivot", False))
        try:
            if first_turn or is_pivot:
                if is_pivot:
                    log.info(
                        "Runner._install_revision: pivot detected — routing "
                        "through install_initial_plan (fresh plan_id=%s, "
                        "prior_plan_id=%s)",
                        (revised_plan.id or "")[:16] or "<none>",
                        (session.plan.id or "")[:16]
                        if session.plan is not None
                        else "<none>",
                    )
                installed = await self.steerer.plans.install_initial_plan(
                    session=session, plan=revised_plan, is_pivot=is_pivot
                )
            else:
                # Turn N+1 replan: the user's fresh message caused
                # handle_turn to surface new/revised work.
                # NEW_WORK_DISCOVERED at INFO severity is the
                # honest classification — not an intervention,
                # not a USER_STEER (no operator ControlMessage
                # exists), just additional work the planner
                # integrated.
                user_text = (
                    user_input.strip()
                    if isinstance(user_input, str)
                    else _initial_goal_summary(user_input)
                )
                drift = DriftEvent(
                    kind=DriftKind.NEW_WORK_DISCOVERED,
                    severity=DriftSeverity.INFO,
                    detail=user_text,
                    authored_by="goldfive",
                )
                installed = await self.steerer.plans.install_revision_for_drift(
                    session=session,
                    drift=drift,
                    revised_plan=revised_plan,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Runner._install_revision: steerer install raised: %s",
                exc,
            )
            return False
        return bool(installed)

    async def _resolve_goals(
        self,
        user_input: str | list[Goal],
        context: Mapping[str, Any] | None,
        session: Session | None = None,
    ) -> list[Goal]:
        if isinstance(user_input, list):
            if not user_input:
                raise ValueError("Runner.run: empty goal list")
            if not all(isinstance(g, Goal) for g in user_input):
                raise TypeError("Runner.run: list input must be list[Goal]")
            return list(user_input)
        if not isinstance(user_input, str):
            raise TypeError(
                f"Runner.run: user_input must be str or list[Goal], got {type(user_input).__name__}"
            )
        # Merge span-emission context (sinks + session correlation) into
        # the context dict the deriver sees so an ``LLMGoalDeriver`` can
        # emit ``GoldfiveLLMCallStart/End`` spans around its internal
        # call. Overrides caller-supplied values deliberately — the
        # Runner owns the sink list and session id.
        span_ctx: dict[str, Any] = dict(context or {})
        if session is not None:
            span_ctx.setdefault("run_id", session.run_id)
            span_ctx.setdefault("session_id", session.id)
            span_ctx.setdefault("next_sequence", session.next_sequence)
        if self.sinks:
            span_ctx.setdefault("sinks", list(self.sinks))
        goals = await self.goal_deriver.derive(user_input, context=span_ctx)
        if not goals:
            raise ValueError("GoalDeriver returned an empty goals list")
        return list(goals)

    async def _emit_run_started(self, session: Session, user_input: str | list[Goal]) -> None:
        evt = run_started_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goal_summary=_initial_goal_summary(user_input),
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_goal_derived(self, session: Session) -> None:
        evt = goal_derived_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            goals=list(session.goals),
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _emit_run_aborted(self, session: Session, reason: str) -> None:
        evt = run_aborted_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            reason=reason,
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _abort_turn(
        self,
        *,
        session: Session,
        convo: Conversation,
        user_input: str | list[Goal],
        pinned: bool,
        reason: str,
    ) -> ExecutionOutcome:
        """Shared abort tail for every pre-/mid-turn failure in :meth:`_run_locked`.

        Emits ``RunAborted``, builds the failed outcome, and absorbs
        the turn into the Conversation so the next turn's carry-forward
        sees a consistent stash. The only per-site deltas — the reason
        string and whether ``log.exception`` fires — stay at the call
        sites.
        """
        await self._emit_run_aborted(session, reason)
        outcome = ExecutionOutcome(success=False, session=session, reason=reason)
        convo.absorb_turn(
            outcome,
            user_input_summary=_initial_goal_summary(user_input),
            pinned=pinned,
        )
        return outcome

    async def _emit_conversation_started(
        self, session: Session, *, conversation: Conversation
    ) -> None:
        evt = conversation_started_event(
            run_id=session.run_id,
            sequence=session.next_sequence(),
            conversation_id=conversation.id,
            session_id=session.id,
        )
        await emit(self.sinks, evt)

    async def _audit_conversation_pending_at_close(
        self,
        *,
        conversation: Conversation,  # noqa: ARG002
        anchor: Session | None,
        key: str,
    ) -> None:
        """Cancel any orphan PENDING tasks at conversation end (goldfive#212).

        Counterpart to the per-turn reachability audit in
        :class:`SequentialExecutor` (PR #339): there, a PENDING task is
        orphaned only when every path to it crosses a CANCELLED / FAILED
        predecessor. At conversation end there is no next turn — so any
        task still PENDING is by definition orphaned (no engaging turn
        will ever pick it up). The audit reuses the existing
        ``TaskCancelled`` envelope with the structured cancel reason
        ``conversation_ended:no_engaging_turn`` so harmonograf's
        Trajectory view can render it without a proto change.

        Idempotent: ``mark_task_cancelled`` no-ops on already-terminal
        tasks (steerer.py:1074), and the outer ``Runner.close()`` is
        gated on ``self._closed`` so a second ``close()`` walk simply
        finds every previously-PENDING task already CANCELLED and emits
        nothing.
        """
        if anchor is None or anchor.plan is None:
            return
        pending = _pending_task_ids(anchor.plan)
        if not pending:
            return
        for tid in pending:
            await self.steerer.transition(
                tid,
                TaskStatus.CANCELLED,
                detail="conversation ended; no further turns to engage this work",
                cancel_reason="conversation_ended:no_engaging_turn",
                session=anchor,
            )
        log.info(
            "Runner._audit_conversation_pending_at_close: cancelled %d "
            "orphan PENDING task(s) for key=%r: %s",
            len(pending),
            key,
            ", ".join(pending),
        )

    async def _emit_conversation_ended(
        self,
        *,
        conversation: Conversation,
        session_anchor: Session,
        reason: str,
    ) -> None:
        # Piggy-back on the last turn's sequence counter so the
        # envelope's sequence field stays monotonic within its run_id.
        evt = conversation_ended_event(
            run_id=session_anchor.run_id,
            sequence=session_anchor.next_sequence(),
            conversation_id=conversation.id,
            turn_count=len(conversation.turns),
            reason=reason,
            session_id=session_anchor.id,
        )
        await emit(self.sinks, evt)


def _initial_goal_summary(user_input: str | list[Goal]) -> str:
    """Best-effort one-liner for the RunStarted event before goals derive."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list) and user_input:
        first = user_input[0]
        return getattr(first, "summary", "") or ""
    return ""


__all__ = ["Runner"]
