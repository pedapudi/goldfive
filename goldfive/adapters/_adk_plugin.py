"""Internal ADK ``BasePlugin`` used by :class:`goldfive.adapters.adk.ADKAdapter`.

The plugin is the routing layer between ADK's callback lifecycle and
goldfive's :class:`~goldfive.protocols.Steerer`. It does four jobs:

1. **State protocol** — ``before_model_callback`` writes the current
   task and plan context into the ADK session state under the
   ``goldfive.*`` keys (see :mod:`._adk_state_protocol`) so agents can
   read them during their turn.
2. **Reporting-tool interception** — ``before_tool_callback`` watches
   for the eight canonical reporting tools. When one fires the plugin
   routes the call's arguments to the corresponding
   :class:`~goldfive.reporting.ReportingToolSpec` handler and returns
   a short-circuit acknowledgment so ADK doesn't execute the stub
   shim the :class:`FunctionTool` actually wraps.
3. **Tool confirmation bridge** — the same ``before_tool_callback``
   intercepts any ADK tool flagged ``require_confirmation=True``
   (Flow B in ``docs/design/APPROVAL.md``), registers a waiter on
   ``session.pending_approvals`` keyed by the ADK ``function_call_id``,
   emits ``ApprovalRequested``, and suspends the tool call until the
   goldfive control dispatcher lands ``APPROVE`` or ``REJECT``. On
   reject, returns a "skipped" dict so ADK does not run the tool body.
4. **Drift observation** — ``after_model_callback``,
   ``on_event_callback`` (transfer/escalation), and
   ``on_tool_error_callback`` feed raw signals into
   ``steerer.observe(...)`` so the steerer can classify drift.

The plugin never imports ``google.adk`` at module load. It imports the
ADK ``BasePlugin`` base class lazily inside :func:`make_adk_plugin`,
which is only called from the adapter's ``__init__``. That keeps this
module importable from non-ADK code for type-checking and allows unit
tests to patch the base class with a stub when ADK is not installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters._tool_invocation import invoke_tool
from goldfive.orchestration_store import (
    PENDING_DELEGATIONS_KEY,
    BindingSource,
    OrchestrationStore,
)

if TYPE_CHECKING:
    from goldfive.protocols import Steerer
    from goldfive.reporting import ReportingToolSpec
    from goldfive.types import Session


log = logging.getLogger("goldfive.adapters.adk")


# ``SessionContext`` is stashed on ADK ``session.state`` under this key
# so test scaffolding that drives the plugin with a hand-built state
# dict can reach back to the goldfive session. NOT used on the live-
# run path: ADK's ``InMemorySessionService.get_session`` returns a
# (deep-)copy of the session, so a write to ``session.state`` from
# :class:`ADKAdapter._invoke_internal` does not propagate to the
# session the runner actually streams against.
#
# The live-run path uses :func:`session_context_from_invocation` —
# a tree-walk that finds the goldfive plugin on the
# :class:`InvocationContext.plugin_manager` and reads its
# ``_active_ctx`` (set by :meth:`_GoldfiveADKPlugin.set_active_context`).
SESSION_CONTEXT_STATE_KEY = "goldfive._session_context"


def session_context_from_invocation(invocation_context: Any) -> SessionContext | None:
    """Return the live :class:`SessionContext` reachable from an ADK invocation.

    Walks ``invocation_context.plugin_manager.plugins`` looking for the
    goldfive plugin (identified by the ``__goldfive_adk_plugin__``
    marker attribute set in :func:`make_adk_plugin`) and returns its
    ``_active_ctx``. The plugin's ``_active_ctx`` is set by
    :meth:`set_active_context` (called from
    :meth:`ADKAdapter._invoke_internal` before ``runner.run_async``)
    so by the time any callback fires the plugin's local field is
    populated.

    Returns ``None`` when:

    * ``invocation_context`` is ``None``.
    * No goldfive plugin is registered on the invocation.
    * The plugin's ``_active_ctx`` hasn't been set yet (out-of-band
      invocations / unit tests that drive callbacks directly).

    Phase 2.0 of goldfive#271 — replaces the V7 state-stash that did
    not propagate through ADK's session_service deep-copy. Closes
    goldfive#275 by giving the resolver / planner a reliable read
    path that doesn't depend on writing ADK ``session.state`` from
    inside a callback frame.
    """
    if invocation_context is None:
        return None
    plugin_manager = getattr(invocation_context, "plugin_manager", None)
    if plugin_manager is None:
        return None
    plugins = getattr(plugin_manager, "plugins", None) or ()
    for plugin in plugins:
        if not getattr(plugin, "__goldfive_adk_plugin__", False):
            continue
        ctx = getattr(plugin, "_active_ctx", None)
        if ctx is not None:
            return ctx
    return None


class SessionContext:
    """Per-invocation context the adapter stashes on ADK state.

    Carries the goldfive :class:`~goldfive.types.Session`, the active
    :class:`~goldfive.protocols.Steerer`, the task the adapter is about
    to invoke, the registered reporting-tool handler map, and the full
    :class:`~goldfive.reporting.ReportingToolSpec` list. The plugin
    picks it up from the ADK callback's ``callback_context`` via
    :func:`_session_context_from_callback`.

    The ``tools`` field is the authoritative source the plugin's
    ``before_tool_callback`` uses to route calls through
    :func:`~goldfive.adapters._tool_invocation.invoke_tool` — that
    routing is what picks up the terminal-task rejection, idempotency,
    and loop-guard layers. ``tool_handlers`` is kept for backward
    compatibility with callers that construct a ``SessionContext``
    directly (examples + test stubs) without a full spec list; when
    only ``tool_handlers`` is supplied the plugin synthesizes minimal
    ``ReportingToolSpec`` shims so the dispatch path still flows
    through ``invoke_tool``.
    """

    __slots__ = (
        "session",
        "steerer",
        "task",
        "tool_handlers",
        "tools",
        "host_agent_name",
    )

    def __init__(
        self,
        *,
        session: Session,
        steerer: Steerer | None,
        task: Any,
        tool_handlers: Mapping[str, Any] | None = None,
        tools: list[ReportingToolSpec] | None = None,
        host_agent_name: str,
    ) -> None:
        self.session = session
        self.steerer = steerer
        self.task = task
        self.host_agent_name = host_agent_name

        # Prefer an explicit ``tools`` list (the adapter constructs
        # ``SessionContext`` that way). Fall back to materialising
        # lightweight specs from ``tool_handlers`` so existing external
        # callers (approval_gated_agent example, legacy tests) still
        # route through ``invoke_tool`` and get the protection layers.
        if tools is not None:
            self.tools = list(tools)
            # Keep ``tool_handlers`` populated as a legacy read-only
            # view for any code that introspected it (e.g.
            # ``before_model_callback`` used to surface the list of
            # reporting-tool names to the state protocol).
            self.tool_handlers = {spec.name: spec.handler for spec in tools}
        else:
            self.tool_handlers = dict(tool_handlers or {})
            self.tools = _tools_from_handlers(self.tool_handlers)


def _tools_from_handlers(
    tool_handlers: Mapping[str, Any],
) -> list[ReportingToolSpec]:
    """Materialise a list of ``ReportingToolSpec`` from a name→handler map.

    Used when a caller builds :class:`SessionContext` with just a
    handler map (legacy path — examples / test stubs). We fabricate
    minimal specs so the dispatch path still routes through
    :func:`invoke_tool` and picks up the terminal-task rejection,
    idempotency, and loop-guard layers.

    The synthesized specs carry empty ``description`` / ``parameters``
    — the plugin only needs ``name`` + ``handler`` at dispatch time;
    the rich schema is surfaced through the native SDK tool wrapping
    done earlier in the adapter's ``register_reporting_tools``.
    """
    from goldfive.reporting import ReportingToolSpec

    specs: list[ReportingToolSpec] = []
    for name, handler in tool_handlers.items():
        specs.append(
            ReportingToolSpec(
                name=str(name),
                description="",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
    return specs


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return value if value is not None else default


@dataclasses.dataclass(frozen=True)
class _InvocationCancelled:
    """Structured marker produced by the goldfive boundary's canonical
    ``except CancelledError`` catch site (goldfive#271 Phase 3.5
    component 1).

    The boundary catches ``CancelledError`` exactly once — inside
    :meth:`ADKAdapter._invoke_internal` — converts it to this marker so
    the rest of the runtime sees a normal completion shape, runs the
    ``finally`` cleanup (heal pending tool calls, emit
    ``InvocationBoundaryExited``, drop registered tasks), then
    re-raises ``CancelledError`` per asyncio's contract.

    The marker is operator-visible only — it never leaks to the LLM
    (the parent agent sees ``{"status": "cancelled"}`` from the bare
    sub-invocation response, not this marker). It exists so that any
    follow-up Phase 3.5 work (the cancellation-stash tripwire) can
    inspect WHY ``CancelledError`` reached the boundary.
    """

    invocation_id: str
    reason: str = "cancelled"
    detail: str = ""


class _InvocationTaskRegistryView:
    """Backwards-compat view over the OrchestrationStore-owned
    invocation-task registry (goldfive#271 Phase 3.5 component 1).

    PR #303 placed the per-invocation asyncio.Task registry on the
    plugin instance as ``_invocation_tasks: dict[str, asyncio.Task]``.
    Phase 3.5 relocates the storage to
    :class:`~goldfive.orchestration_store.OrchestrationStore` per the
    Phase 0 state-ownership contract, but keeps the legacy attribute
    accessible: the steerer's cancel path and tests still index into
    ``plugin._invocation_tasks[inv_id]``.

    This view forwards the ``dict``-shaped operations the existing
    callers use (``[]``, ``get``, ``setdefault`` via assignment,
    ``pop``, ``clear``, ``in``) onto the store. The store is resolved
    from the plugin's currently-active ``SessionContext`` so writes
    reach the correct per-session bucket; reads / writes against an
    unbound plugin (no session context) silently no-op (returning
    ``None`` for reads, dropping writes), matching the pre-migration
    behaviour of an empty dict.
    """

    __slots__ = ("_plugin",)

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    def _store(self) -> Any:
        # Lazy resolution: the active context is set by
        # ``set_active_context`` before any callback fires, so a read
        # path that runs outside an active dispatch (e.g. a unit test
        # constructing the plugin and probing ``_invocation_tasks``
        # without driving a callback) sees an empty registry.
        ctx = getattr(self._plugin, "_active_ctx", None)
        if ctx is None:
            return None
        from goldfive.orchestration_store import OrchestrationStore

        return OrchestrationStore.for_session(ctx.session)

    def __setitem__(self, invocation_id: str, task: asyncio.Task[Any]) -> None:
        store = self._store()
        if store is None:
            return
        store.register_invocation_task(invocation_id, task)

    def __getitem__(self, invocation_id: str) -> asyncio.Task[Any]:
        store = self._store()
        if store is None:
            raise KeyError(invocation_id)
        task = store.get_invocation_task(invocation_id)
        if task is None:
            raise KeyError(invocation_id)
        return task

    def get(
        self,
        invocation_id: str,
        default: Any = None,
    ) -> asyncio.Task[Any] | None:
        store = self._store()
        if store is None:
            return default
        task = store.get_invocation_task(invocation_id)
        return task if task is not None else default

    def pop(self, invocation_id: str, *args: Any) -> Any:
        store = self._store()
        if store is None:
            if args:
                return args[0]
            raise KeyError(invocation_id)
        task = store.get_invocation_task(invocation_id)
        store.deregister_invocation_task(invocation_id)
        if task is None:
            if args:
                return args[0]
            raise KeyError(invocation_id)
        return task

    def __contains__(self, invocation_id: str) -> bool:
        store = self._store()
        if store is None:
            return False
        return store.get_invocation_task(invocation_id) is not None

    def clear(self) -> None:
        store = self._store()
        if store is None:
            return
        store.clear_active_invocations()

    def keys(self) -> list[str]:
        store = self._store()
        if store is None:
            return []
        return store.active_invocation_ids()

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


def _safe_task_cancel(task: asyncio.Task[Any], invocation_id: str) -> None:
    """Cancel ``task`` and log the outcome — used as a deferred
    ``loop.call_soon`` callback (goldfive#271 follow-up).

    Wrapping the cancel in a free function keeps the
    ``loop.call_soon(_safe_task_cancel, t, inv_id)`` site reference-
    free of mutable plugin state — the loop holds only the task and
    id strings until the cancel fires. Swallows any failure so an
    already-finished task doesn't surface a spurious traceback to
    the loop's exception handler.
    """
    if task.done():
        return
    try:
        cancelled_now = task.cancel()
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "_safe_task_cancel: task.cancel() for invocation_id=%s raised: %s",
            invocation_id,
            exc,
        )
        return
    if cancelled_now:
        log.info(
            "goldfive.cancel.task: cancelled in-flight task for invocation_id=%s",
            invocation_id,
        )


def _session_state_from_callback(ctx: Any) -> Any:
    """Return the ADK session.state mutable mapping for a callback ctx.

    Tolerates the several shapes ADK has used across versions:
    ``ctx.session.state``, ``ctx._invocation_context.session.state``,
    ``ctx.state``. Returns an empty dict if none match.
    """
    for attr_chain in (
        ("session", "state"),
        ("_invocation_context", "session", "state"),
        ("invocation_context", "session", "state"),
    ):
        cur: Any = ctx
        ok = True
        for part in attr_chain:
            cur = _safe_attr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            return cur
    direct = _safe_attr(ctx, "state", None)
    if direct is not None:
        return direct
    return {}


def _session_context_from_callback(ctx: Any) -> SessionContext | None:
    state = _session_state_from_callback(ctx)
    if not isinstance(state, Mapping):
        return None
    value = state.get(SESSION_CONTEXT_STATE_KEY)
    if isinstance(value, SessionContext):
        return value
    return None


# --------------------------------------------------------------------------
# goldfive#191 — Layer 3: task_id arg injection for reporting tools
# --------------------------------------------------------------------------
#
# When an LLM emits a report_task_* / report_awaiting_approval call with a
# missing or obviously-placeholder ``task_id``, we fall back to the pinned
# ``goldfive.current_task_id`` that ``before_agent_callback`` (Layer 1)
# stamps into session.state at the start of every agent turn. The reporting-
# tool handler itself (Layer 2) also defaults from state, so this layer is
# the outermost safety net: the goal is to have the handler see a valid
# task_id no matter which dispatch path runs it.
#
# We ONLY rewrite when the arg is missing or is a well-known placeholder
# string. A real-looking task_id (even if it's for the wrong task) is left
# alone so the handler surfaces the mismatch as a proper terminal-task /
# not-found failure rather than silently re-targeting the call. Wrong
# task_ids are better surfaced as failures than masked.

# Reporting-tool names that must always target a specific task. The match
# is an exact-set test for ``report_awaiting_approval`` plus a prefix test
# for ``report_task_*`` (see goldfive.reporting for the canonical list).
_REPORT_AWAITING_APPROVAL = "report_awaiting_approval"

# Case-insensitive set of strings we treat as "the LLM didn't supply a real
# task_id". Whitespace is stripped before comparison. Keep this list
# conservative — any real-looking slug should NOT be on it.
_PLACEHOLDER_TASK_IDS: frozenset[str] = frozenset(
    {"", "placeholder", "unknown", "todo", "none", "null", "n/a"}
)


def _is_placeholder_task_id(value: Any) -> bool:
    """Return True if ``value`` is missing or an obvious placeholder.

    Used by :meth:`_GoldfiveADKPlugin.before_tool_callback` to decide
    whether to overwrite ``tool_args["task_id"]`` with the pinned
    ``goldfive.current_task_id`` from session.state. Case-insensitive and
    whitespace-tolerant. Non-string values are treated as placeholders
    (an int task_id isn't real either).
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return True
    return value.strip().lower() in _PLACEHOLDER_TASK_IDS


def _is_reporting_tool_name(tool_name: str) -> bool:
    """Return True if ``tool_name`` is a reporting-tool that needs a task_id."""
    if not tool_name:
        return False
    return tool_name.startswith("report_task_") or tool_name == _REPORT_AWAITING_APPROVAL


def _is_agent_tool_dispatch(tool: Any) -> bool:
    """Return True if ``tool`` is an ADK ``AgentTool`` (sub-agent delegation).

    The cooperative-cancel short-circuit in :meth:`before_tool_callback`
    discriminates on this: AgentTool dispatches spawn a fresh sub-Runner
    and stream its full transcript back, so short-circuiting them is
    correct (we don't want to spawn long sub-agent work on a cancelled
    parent). Plain ``FunctionTool`` dispatches (write_webpage, patch_file,
    user-provided side-effect helpers) finish in milliseconds and have
    already had their args chosen by the LLM — short-circuiting them
    discards committed work for no benefit, and the next
    ``before_model_callback`` short-circuit ends the dispatch cleanly
    anyway. See goldfive#211610 / Bug C from v23 validation.

    Detection prefers an ``isinstance(tool, AgentTool)`` check when the
    optional ``adk`` extra is importable, with a duck-typed fallback for
    test stubs and forward-compatibility: AgentTool exposes ``.agent``
    pointing at the wrapped ``BaseAgent``. A non-callable ``.agent``
    attribute on a ``FunctionTool`` is unheard of upstream, but the
    duck-typed branch is conservative anyway — when unsure, treat the
    tool as a plain function and let it run.
    """
    if tool is None:
        return False
    try:
        from google.adk.tools import AgentTool  # type: ignore

        if isinstance(tool, AgentTool):
            return True
    except Exception:  # noqa: BLE001 — adk extra not installed / import edge
        pass
    # Duck-typed fallback: AgentTool carries a ``.agent`` BaseAgent
    # pointer (the sub-agent it delegates to). FunctionTool carries a
    # ``.func`` instead. Treat the presence of ``.agent`` as the AgentTool
    # discriminator.
    return _safe_attr(tool, "agent", None) is not None


def _agent_has_pending_candidates(ctx: Any, agent_name: str) -> bool:
    """Return True if the plan has any PENDING/RUNNING task for ``agent_name``.

    Used to distinguish the two failure modes when the reporting-tool
    ``task_id`` pin cannot be resolved (goldfive#250 follow-up):

    * **No candidates** — the agent's work was moved to other agents by
      a plan refine; a silent-ack no-op is correct.
    * **Has candidates** — the pin SHOULD have worked (single match, or
      delegation-site pin, or fallback). The tool response is still
      ``{"acknowledged": True}`` so the LLM can't pattern-match on an
      error shape (observed live: models read ``"error": "pin_unresolved"``
      as a reasoning cue and bypass the reporting contract). Operator
      visibility is preserved via a WARNING log + a ``DriftDetected``
      sink event; see :meth:`_emit_pin_unresolved_drift`.

    NOTE: the DAG-readiness gate from
    :meth:`_pin_current_task_id_for_agent` is intentionally NOT applied
    here. Even a DAG-gated candidate means the agent's turn shouldn't
    be happening yet, which is also a stall worth surfacing rather than
    silencing.

    Silent on every failure (missing ctx / plan / non-iterable tasks);
    the caller falls back to the conservative silent-ack path when this
    returns ``False``.
    """
    if not agent_name:
        return False
    try:
        plan = _safe_attr(ctx, "session", None)
        plan = _safe_attr(plan, "plan", None) if plan is not None else None
        if plan is None:
            return False
        tasks = _safe_attr(plan, "tasks", None) or ()
        from goldfive.types import TaskStatus

        for task in tasks:
            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
            if assignee != agent_name:
                continue
            status = _safe_attr(task, "status", None)
            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                return True
    except Exception:  # noqa: BLE001 — diagnostic-only
        return False
    return False


def _maybe_redirect_completed_agent(
    *,
    ctx: Any,
    target_agent: str,
) -> dict[str, Any] | None:
    """Tier 1 / F3: pre-dispatch redirect for AgentTool calls on completed work.

    Returns a redirect-error response dict when the coordinator is
    invoking an AgentTool whose plan tasks are ALL terminal AND a
    non-terminal next_pending task exists assigned to a different agent.
    Returns ``None`` to allow the dispatch in every other case:

    * the target agent has at least one PENDING / RUNNING / BLOCKED task
      assigned (legitimate dispatch),
    * the target agent has no plan match at all (off-plan agent — the
      existing PLAN_DIVERGENCE drift detector handles that path; we do
      NOT double-handle here),
    * no plan is installed on the session (defensive),
    * the next_pending task is also assigned to ``target_agent``
      (re-dispatch onto the same agent for follow-up work — allow).

    The "all terminal AND next_pending elsewhere" guard is the precise
    shape of the loop the cap ``_MAX_NUDGE_REPLAYS`` was historically
    catching post-hoc: the LLM has just received an "ack" for the
    completed task and re-invokes the same agent because nothing told
    it to stop. F1's directive payload is the proactive anchor; this
    is the structural fence.
    """
    if not target_agent:
        return None
    try:
        plan = _safe_attr(ctx, "session", None)
        plan = _safe_attr(plan, "plan", None) if plan is not None else None
        if plan is None:
            return None
        tasks = list(_safe_attr(plan, "tasks", None) or ())
        if not tasks:
            return None
        from goldfive.types import TERMINAL_TASK_STATUSES, TaskStatus

        # Collect tasks assigned to the target agent. Match on bare
        # agent name (last dot-separated segment) so fully-qualified
        # ADK paths like ``coordinator.research_agent`` round-trip.
        def _bare(name: str) -> str:
            n = (name or "").strip()
            return n.rsplit(".", 1)[-1] if "." in n else n

        target_bare = _bare(target_agent)
        assigned: list[Any] = []
        for task in tasks:
            assignee = _bare(str(_safe_attr(task, "assignee_agent_id", "") or ""))
            if assignee and assignee == target_bare:
                assigned.append(task)
        if not assigned:
            # Off-plan agent — let PLAN_DIVERGENCE handle it.
            return None

        # If any assigned task is non-terminal, the dispatch is legitimate.
        if any(_safe_attr(t, "status", None) not in TERMINAL_TASK_STATUSES for t in assigned):
            return None

        # All assigned tasks are terminal. Find the next PENDING task
        # whose every predecessor is terminal — same definition the F1
        # plan_state helper uses, kept local so this module stays
        # decoupled from goldfive.reporting.
        edges = list(_safe_attr(plan, "edges", None) or ())
        by_id = {str(_safe_attr(t, "id", "") or ""): t for t in tasks}
        incoming: dict[str, list[str]] = {tid: [] for tid in by_id}
        for e in edges:
            to_id = str(_safe_attr(e, "to_task_id", "") or "")
            from_id = str(_safe_attr(e, "from_task_id", "") or "")
            if to_id in incoming and from_id:
                incoming[to_id].append(from_id)

        next_pending: Any = None
        for task in tasks:
            if _safe_attr(task, "status", None) is not TaskStatus.PENDING:
                continue
            tid = str(_safe_attr(task, "id", "") or "")
            if not tid:
                continue
            preds = incoming.get(tid, [])
            if all(
                by_id.get(p) is not None
                and _safe_attr(by_id[p], "status", None) in TERMINAL_TASK_STATUSES
                for p in preds
            ):
                next_pending = task
                break

        if next_pending is None:
            # Plan is effectively done; nothing useful to redirect to.
            # Let the dispatch fall through — the runner's own end-of-
            # plan handling will close out the run.
            return None

        next_assignee = _bare(str(_safe_attr(next_pending, "assignee_agent_id", "") or ""))
        if next_assignee == target_bare:
            # Next pending is on the same agent — this dispatch is
            # legitimate follow-up work, not a loop.
            return None

        next_title = str(_safe_attr(next_pending, "title", "") or "") or str(
            _safe_attr(next_pending, "id", "") or ""
        )
        return {
            "error": (
                f"All plan tasks for {target_bare} are complete. Next "
                f"pending task is '{next_title}' assigned to "
                f"{next_assignee or 'the next planned agent'}. "
                "Please invoke that agent."
            ),
            "redirect_to": next_assignee,
        }
    except Exception as exc:  # noqa: BLE001 — defensive; never break dispatch
        log.debug("_maybe_redirect_completed_agent: classification raised: %s", exc)
        return None


def _inject_task_id_from_state(
    *,
    tool_name: str,
    tool_args: Any,
    tool_context: Any,
) -> bool:
    """Populate ``tool_args['task_id']`` from state for reporting tools.

    goldfive#241 — ``task_id`` is hidden from the LLM-facing reporting-
    tool schema (see :func:`goldfive.adapters.adk._apply_llm_signature`),
    so the model cannot supply it. Every reporting-tool call lands here
    with no ``task_id`` arg; this function is the authoritative
    resolution layer.

    Resolution order, keyed by the invocation's ``function_call_id``:

    1. ``goldfive.pending_delegations[<function_call_id>]`` — the
       delegation-site pin stamped by :meth:`before_tool_callback` for
       the AgentTool dispatch that spawned the current sub-invocation.
       This path handles coordinators that fire multiple parallel
       AgentTool calls to the same sub-agent on the same turn — each
       parallel dispatch gets its own pin rather than racing on the
       single ``goldfive.current_task_id`` slot.
    2. ``session.state[goldfive.current_task_id]`` — the agent-turn
       pin written by ``before_agent_callback`` at the start of every
       agent invocation (goldfive#191/#195).

    Returns ``True`` when ``tool_args`` now carries a usable ``task_id``
    (either pre-existing non-placeholder, or freshly populated from
    state) and ``False`` when no pin is available — in that case the
    caller short-circuits with a bare ``{"acknowledged": True}`` and
    emits operator observability (WARNING log + ``DriftDetected`` sink
    event) rather than letting the handler run on an unpinned call.
    Pre-#241 the call would have fallen through to the handler's
    ``missing_task_id`` error, but that path goes back to the LLM via a
    successful tool response shape and confused the model into retry
    loops. Post-#252-followup, even the structured-error shape is gone
    from the LLM-visible response — see :meth:`_emit_pin_unresolved_drift`.

    NEVER raises — any internal failure degrades to "no pin".
    """
    try:
        if not _is_reporting_tool_name(tool_name):
            return True
        if not isinstance(tool_args, MutableMapping):
            return True
        existing = tool_args.get("task_id", "")
        if not _is_placeholder_task_id(existing):
            # A real-looking id was supplied (e.g. legacy caller,
            # custom tool that didn't opt into the hidden schema).
            # Leave it alone so the handler surfaces mismatches as
            # terminal-task / not-found failures rather than silently
            # re-targeting the call.
            return True
        resolved = _resolve_pinned_task_id(
            tool_context=tool_context,
        )
        if not resolved:
            return False
        tool_args["task_id"] = resolved
        return True
    except Exception:  # noqa: BLE001
        log.debug(
            "before_tool_callback: task_id injection failed for tool=%s",
            tool_name,
            exc_info=True,
        )
        return False


# Re-export under the legacy module-private name so downstream tests
# and custom adapters that import :data:`_PENDING_DELEGATIONS_KEY` keep
# working unchanged. Phase 2.1 (goldfive#271) moved the canonical
# definition into :mod:`goldfive.orchestration_store`; the value is
# unchanged.
_PENDING_DELEGATIONS_KEY = PENDING_DELEGATIONS_KEY

# Map the pin-ladder ``source`` strings used in :meth:`_stamp_current_task_id`'s
# log line onto :class:`BindingSource` enum values so the orchestration
# store records typed attribution alongside the pin write. Phase 2.1 of
# goldfive#271 — the source label is now a structural part of the pin,
# not just a free-form log token.
_BINDING_SOURCE_BY_LADDER: dict[str, BindingSource] = {
    "delegation_pin": BindingSource.DELEGATION_PIN,
    "single_match": BindingSource.AGENT_CALLBACK,
    "arg_scored": BindingSource.AGENT_CALLBACK,
    "dag_relaxed": BindingSource.AGENT_CALLBACK,
    "parent_pin_downstream": BindingSource.AGENT_CALLBACK,
    "reasoning_binding": BindingSource.REASONING,
    "correction_target": BindingSource.CORRECTION_TARGET,
    "assignee_normalised": BindingSource.AGENT_CALLBACK,
    "low_confidence": BindingSource.LOW_CONFIDENCE,
}


def _resolve_pinned_task_id(*, tool_context: Any) -> str:
    """Return the task_id pinned for this tool invocation, or ``""``.

    Consults the delegation-site map first
    (:meth:`OrchestrationStore.get_pending_delegation` keyed by the
    current invocation's ``function_call_id``), then falls back to
    the agent-turn pin (:meth:`OrchestrationStore.pin_current_task`).
    Returns ``""`` when neither path yields a value — the caller
    should then short-circuit with a bare ``{"acknowledged": True}``
    (plus a ``DriftDetected`` sink event for operator visibility on
    the has-candidates branch) rather than invoking the handler on an
    unpinned arg. Pre-#252-followup the has-candidates branch emitted
    an ``error: pin_unresolved`` payload, which the LLM read as a
    reasoning cue and used to bypass the reporting contract — see
    the pin-leak fix in that PR.

    Phase 2.1 of goldfive#271 — V4 reader. The store walks goldfive
    ``Session.state`` reached either via the live plugin reference
    (production path) or the legacy V7 SessionContext stash on ADK
    state (unit-test path). No callback-time read of ADK state for
    pin keys remains.
    """
    session = _goldfive_session_from_tool_context(tool_context)
    store = OrchestrationStore.for_session(session) if session is not None else None
    fc_id = _function_call_id_from_tool_context(tool_context)
    via = ""
    pinned = ""
    if store is not None and fc_id:
        delegation = store.get_pending_delegation(fc_id)
        if delegation is not None and delegation.is_set():
            pinned = delegation.task_id
            via = "delegation_pin"
    if not pinned and store is not None:
        agent_pin = store.pin_current_task()
        if agent_pin:
            pinned = agent_pin
            via = "agent_pin"
    if pinned:
        log.debug(
            "goldfive.pin.read: task_id=%s via=%s fc_id=%s",
            pinned,
            via,
            fc_id or "-",
        )
        return pinned
    return ""


def _goldfive_session_from_tool_context(tool_context: Any) -> Any:
    """Return the goldfive ``Session`` reachable from a ``tool_context``.

    Phase 2.1 of goldfive#271. Resolution order matches the
    dynamic-instruction resolver's :func:`_goldfive_session_from_readonly_context`:

    1. Walk the invocation's ``plugin_manager.plugins`` for the
       goldfive plugin and read its ``_active_ctx.session``. Live-run
       path — set by :meth:`set_active_context`.
    2. Legacy V7 ``SESSION_CONTEXT_STATE_KEY`` stash on the callback's
       ``session.state``. Used by unit tests that drive the plugin
       with a hand-built state dict without going through the
       plugin's lifecycle.

    Returns ``None`` when neither resolves so the caller can degrade
    cleanly.
    """
    inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
        tool_context, "invocation_context", None
    )
    if inv_ctx is not None:
        ctx = session_context_from_invocation(inv_ctx)
        if ctx is not None:
            session = getattr(ctx, "session", None)
            if session is not None:
                return session
    legacy_ctx = _session_context_from_callback(tool_context)
    if legacy_ctx is not None:
        return getattr(legacy_ctx, "session", None)
    return None


def _delegation_pin_task_id(raw: Any) -> str:
    """Return the task_id from a pending-delegations entry.

    Tolerates both pre-#266 shapes:

    * ``str`` — legacy direct task_id.
    * ``Mapping`` with a ``"task_id"`` key — versioned shape that also
      carries ``"revision"`` for the report-time classifier.

    Returns ``""`` for any malformed / empty input. Strips whitespace
    so consumers can treat the result as a clean key.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, Mapping):
        tid = raw.get("task_id", "")
        if isinstance(tid, str):
            return tid.strip()
        if tid is not None:
            return str(tid).strip()
    return ""


def _delegation_pin_revision(raw: Any) -> int:
    """Return the pin's revision stamp from a pending-delegations entry.

    Pre-#266 (string-shaped) entries return 0 — they predate the
    revision stamp and the report-time classifier treats 0 as the
    initial revision (matches the default ``Plan.revision_index=0``,
    so an unrevised plan looks fresh).
    """
    if isinstance(raw, Mapping):
        rev = raw.get("revision", 0)
        try:
            return max(0, int(rev))
        except (TypeError, ValueError):
            return 0
    return 0


def _delegation_pin_tool_args(raw: Any) -> Mapping[str, Any] | None:
    """Return the parent AgentTool's tool-call args from an entry, or ``None``.

    Pre-F7 (string-shaped or ``{task_id, revision}``) entries return
    ``None`` — they predate the tool-args stamp added in #265-followup.
    Signal 3 of :meth:`_pin_current_task_id_for_agent` consults this
    to score DAG-ready candidates against the dispatch args (the
    strongest available signal: the parent literally said "go do
    *this*"). Falls back to steer body / goals when ``None`` or empty.
    """
    if not isinstance(raw, Mapping):
        return None
    args = raw.get("tool_args", None)
    if not isinstance(args, Mapping):
        return None
    if not args:
        return None
    return args


def _function_call_id_from_tool_context(tool_context: Any) -> str:
    """Best-effort extraction of the current ``function_call_id``.

    ADK's ToolContext carries the active ``function_call_id`` directly;
    legacy / test stubs may not have the attribute, in which case we
    degrade to ``""`` (falls back to the agent-turn pin).
    """
    fc_id = _safe_attr(tool_context, "function_call_id", "")
    if isinstance(fc_id, str) and fc_id.strip():
        return fc_id.strip()
    return ""


def _tokenize_for_matching(text: Any) -> set[str]:
    """Return the set of lowercase alphanumeric tokens of length ≥4.

    Used by :func:`_score_candidates_by_args` to score candidate tasks
    against AgentTool args. The ≥4 threshold filters out noisy
    short-word matches ("in", "of", "the") that would otherwise
    saturate every candidate's score.
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


def _score_candidates_by_args(candidates: list[Any], tool_args: Any) -> Any:
    """Return the best-scoring candidate, or ``None`` on tie / zero match.

    Scoring: tokenise ``tool_args`` and each candidate's
    ``title + description``. Candidate with the highest overlap wins.
    Ties (two or more candidates with the same top non-zero score)
    return ``None`` so the caller falls through to the no-pin path;
    guessing would be worse than letting the sub-agent path handle
    the ambiguity.
    """
    if not candidates:
        return None
    # Serialise args into a single token bag. Keys contribute too
    # (``topic=solar`` contributes "topic" and "solar") because the
    # key names often hint at which task the call is about.
    args_text = ""
    if isinstance(tool_args, Mapping):
        parts: list[str] = []
        for k, v in tool_args.items():
            parts.append(str(k))
            parts.append(str(v))
        args_text = " ".join(parts)
    elif isinstance(tool_args, str):
        args_text = tool_args
    arg_tokens = _tokenize_for_matching(args_text)
    if not arg_tokens:
        return None
    best_score = 0
    best: Any = None
    tied = False
    for cand in candidates:
        title = str(_safe_attr(cand, "title", "") or "")
        desc = str(_safe_attr(cand, "description", "") or "")
        cand_tokens = _tokenize_for_matching(f"{title} {desc}")
        score = len(arg_tokens & cand_tokens)
        if score > best_score:
            best_score = score
            best = cand
            tied = False
        elif score == best_score and score > 0:
            tied = True
    if best_score == 0 or tied:
        return None
    return best


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


def _extract_usage_metadata(llm_response: Any) -> dict[str, int]:
    """Pull prompt / completion / total token counts off an ADK ``LlmResponse``.

    ADK normalises per-provider usage onto
    ``llm_response.usage_metadata`` (a
    ``google.genai.types.GenerateContentResponseUsageMetadata`` with
    ``prompt_token_count`` / ``candidates_token_count`` /
    ``total_token_count``). Returns an empty dict when the backend
    didn't report usage (some LiteLLM-fronted providers skip it, some
    streaming responses defer it to a trailing chunk).

    Returns only fields that are present and non-zero — callers treat
    missing keys as "not reported".
    """
    out: dict[str, int] = {}
    usage = _safe_attr(llm_response, "usage_metadata", None)
    if usage is None:
        return out
    for src_attr, dst_key in (
        ("prompt_token_count", "prompt_tokens"),
        ("candidates_token_count", "completion_tokens"),
        ("total_token_count", "total_tokens"),
    ):
        value = _safe_attr(usage, src_attr, None)
        if isinstance(value, int) and value > 0:
            out[dst_key] = value
    return out


def _extract_text_parts(llm_response: Any) -> list[str]:
    content = _safe_attr(llm_response, "content", None)
    if content is None:
        return []
    parts = _safe_attr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        if _safe_attr(part, "thought", False):
            continue
        text = _safe_attr(part, "text", "") or ""
        if text:
            texts.append(str(text))
    return texts


def _infer_provider(llm_response: Any) -> str:
    """Guess which backend produced ``llm_response`` for observability.

    Returns one of ``"openai"`` / ``"anthropic"`` / ``"google"`` /
    ``""``. Used only to tag reasoning-drift events, so an approximate
    answer is fine -- misclassification does not change detector
    behavior.
    """
    module = type(llm_response).__module__ if llm_response is not None else ""
    module_lower = module.lower()
    if "anthropic" in module_lower:
        return "anthropic"
    if "openai" in module_lower or "litellm" in module_lower:
        return "openai"
    if "google" in module_lower or "genai" in module_lower:
        return "google"
    # ADK's LlmResponse carries a Content with parts that include
    # thought-flagged entries: treat as google when no better signal.
    content = _safe_attr(llm_response, "content", None)
    if content is not None and _safe_attr(content, "parts", None) is not None:
        return "google"
    if _safe_attr(llm_response, "choices", None) is not None:
        return "openai"
    return ""


def _extract_reasoning(llm_response: Any) -> str:
    """Best-effort per-provider reasoning extraction.

    Reasoning-content / thinking blocks live in different places on
    different providers. This helper walks the known shapes in
    priority order and returns the first non-empty reasoning text it
    finds, or ``""`` when the response carries none.

    Shapes handled:

    * ADK ``content.parts[i]`` with ``thought=True`` -- Google's
      Gemini surface (thought blocks are standard parts flagged as
      such).
    * OpenAI-compat ``response.choices[0].message.reasoning_content``
      -- Qwen3.5 via LiteLLM, some o1-series models, Deepseek.
    * Anthropic ``response.content[i].type == "thinking"`` blocks
      -- Claude extended thinking.
    * Plain string fields ``reasoning`` / ``reasoning_content`` on
      the response itself (tolerant fallback).

    Returns the concatenated reasoning text. Callers downstream treat
    empty strings as "no reasoning available".
    """
    # ADK thought parts: parts with .thought=True carry the
    # chain-of-thought when Google's Gemini returns one.
    content = _safe_attr(llm_response, "content", None)
    if content is not None:
        parts = _safe_attr(content, "parts", None) or []
        thoughts: list[str] = []
        for part in parts:
            if not _safe_attr(part, "thought", False):
                continue
            text = _safe_attr(part, "text", "") or ""
            if text:
                thoughts.append(str(text))
        if thoughts:
            return "\n".join(thoughts)

    # OpenAI-compat: response.choices[0].message.reasoning_content.
    # Used by LiteLLM-fronted Qwen3.5 and some o1 / Deepseek models.
    try:
        choices = _safe_attr(llm_response, "choices", None) or []
        if choices:
            msg = _safe_attr(choices[0], "message", None)
            if msg is not None:
                rc = _safe_attr(msg, "reasoning_content", None) or _safe_attr(
                    msg, "reasoning", None
                )
                if rc:
                    return str(rc)
    except Exception:  # noqa: BLE001 -- best-effort extraction
        pass

    # Anthropic: content blocks with type="thinking".
    try:
        blocks = _safe_attr(llm_response, "content", None)
        if isinstance(blocks, list):
            for block in blocks:
                if _safe_attr(block, "type", "") == "thinking":
                    t = _safe_attr(block, "thinking", "") or ""
                    if t:
                        return str(t)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: a flat attribute on the response itself.
    for attr in ("reasoning_content", "reasoning", "thinking"):
        v = _safe_attr(llm_response, attr, None)
        if isinstance(v, str) and v:
            return v

    return ""


def _extract_function_calls(llm_response: Any) -> list[dict]:
    content = _safe_attr(llm_response, "content", None)
    if content is None:
        return []
    parts = _safe_attr(content, "parts", None) or []
    calls: list[dict] = []
    for part in parts:
        fc = _safe_attr(part, "function_call", None)
        if fc is None:
            continue
        calls.append(
            {
                "name": str(_safe_attr(fc, "name", "") or ""),
                "args": _safe_attr(fc, "args", None),
            }
        )
    return calls


def _as_observation(
    *,
    kind: str,
    detail: str = "",
    raw: Any = None,
    task: Any = None,
    agent_id: str = "",
) -> dict[str, Any]:
    """Build the lightweight observation dict handed to ``steerer.observe``.

    The steerer is responsible for classifying this into a
    :class:`~goldfive.types.DriftEvent` via ``detect_drift``. This
    adapter just translates ADK-native events into a stable shape.
    """
    return {
        "source": "adk",
        "kind": kind,
        "detail": detail,
        "task_id": _safe_attr(task, "id", "") or "",
        "agent_id": agent_id or "",
        "raw": raw,
    }


def _is_progress_report_success(response: Any) -> bool:
    """Return ``True`` iff ``response`` indicates a successful progress report.

    Used by the ADK plugin's ``after_tool_callback`` to decide whether a
    ``report_task_*`` / ``report_awaiting_approval`` call should reset
    the tool-loop tracker's per-(invocation, agent) window (goldfive#192).

    Previously the exemption triggered on the **call** regardless of
    outcome: an agent stuck retrying ``report_task_started`` with a
    bad ``task_id`` would keep getting ``{"acknowledged": false,
    "error": "missing_task_id"}`` and every one of those errored calls
    reset the detector window -- masking an obvious tool-loop. This
    helper tightens the gate so only acknowledged-success responses
    reset. Everything else (errored, missing field, unknown shape) is
    conservatively treated as NOT a legitimate progress report so the
    loop detector gets to count the call.

    Shape contract: the reporting tools return a dict with
    ``acknowledged`` set to ``True`` on success and ``False`` on
    failure (with an ``error`` key describing what went wrong); see
    :mod:`goldfive.adapters._tool_invocation`. Any other shape
    (``None``, a string, a bare list, etc.) is treated as "unknown"
    and does NOT reset the window.
    """
    if response is None:
        return False
    if isinstance(response, Mapping):
        # Explicit failure wins: even if ``acknowledged`` is somehow
        # True alongside an ``error`` key, treat it as errored so we
        # don't silently reset on half-broken shapes.
        if "error" in response:
            return False
        if response.get("acknowledged") is True:
            return True
        return False
    return False  # unknown shape (str, list, bare value) -- conservative no-reset


def _tool_requires_confirmation(tool: Any, tool_args: Any) -> bool:
    """Return True if ``tool`` opts into ADK's require_confirmation flag.

    ``FunctionTool`` stores the flag on ``_require_confirmation``; we also
    accept a public ``require_confirmation`` attribute so tests can
    supply minimal stub tools without subclassing. The value may be a
    bool or a callable that receives the tool args; per ADK semantics
    the callable resolves the decision per-call.
    """
    flag = getattr(tool, "_require_confirmation", None)
    if flag is None:
        flag = getattr(tool, "require_confirmation", None)
    if flag is None:
        return False
    if callable(flag):
        try:
            args = dict(tool_args) if isinstance(tool_args, Mapping) else {}
            return bool(flag(**args))
        except TypeError:
            try:
                return bool(flag(tool_args))
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001
            return False
    return bool(flag)


def _function_call_id(tool_context: Any) -> str:
    """Best-effort pull of the ADK ``function_call_id`` for a tool invocation.

    ADK sets this on ``ToolContext._function_call_id``; exposes it via
    a public property in recent versions. Falls back to generating a
    fresh ``adk-<uuid>`` so correlation still works when tests pass a
    minimal stub context.
    """
    for attr in ("function_call_id", "_function_call_id"):
        value = _safe_attr(tool_context, attr, None)
        if value:
            return str(value)
    import uuid as _uuid

    return f"adk-{_uuid.uuid4().hex}"


async def _emit_approval_requested_from_plugin(
    *,
    session: Any,
    steerer: Any,
    target_id: str,
    prompt: str,
    tool_name: str,
    tool_args: Mapping[str, Any] | dict[str, Any],
    task_id: str,
) -> None:
    sinks = getattr(steerer, "_sinks", None) or []
    if not sinks:
        return
    try:
        args_json = json.dumps(
            {k: _jsonable(v) for k, v in dict(tool_args).items()},
            sort_keys=True,
        )
    except Exception:  # noqa: BLE001
        args_json = "{}"
    try:
        from goldfive.events import approval_requested_event, emit

        evt = approval_requested_event(
            run_id=getattr(session, "run_id", ""),
            sequence=session.next_sequence(),
            target_id=target_id,
            kind="tool",
            prompt=prompt,
            task_id=task_id,
            metadata={"tool_name": tool_name, "args_json": args_json},
            session_id=getattr(session, "id", ""),
        )
        await emit(sinks, evt)
    except Exception as exc:  # noqa: BLE001
        log.debug("_emit_approval_requested_from_plugin: sink emit failed: %s", exc)


def _jsonable(v: Any) -> Any:
    """Best-effort coerce ``v`` to a JSON-serializable shape for metadata."""
    if isinstance(v, str | int | float | bool) or v is None:
        return v
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return repr(v)


async def _await_tool_approval(
    *,
    tool: Any,
    tool_name: str,
    tool_args: Any,
    tool_context: Any,
    session_ctx: Any,
) -> dict[str, Any] | None:
    """Gate ``tool`` on a goldfive control-channel APPROVE / REJECT.

    Registers an ``asyncio.Event`` on
    ``session.pending_approvals[function_call_id]``, emits
    ``ApprovalRequested``, and awaits the waiter. On REJECT returns a
    skipped dict (which ADK treats as the tool's response to the model,
    bypassing ``tool.run_async``). On APPROVE returns ``None`` so ADK
    proceeds with the original args.

    No timeout on the wait by design: the control channel is the
    authoritative signal. Callers wanting a timeout should layer it via
    a CANCEL control.
    """
    session = session_ctx.session
    steerer = session_ctx.steerer
    target_id = _function_call_id(tool_context)

    prompt = _tool_approval_prompt(tool, tool_name, tool_args)
    waiter = session.pending_approvals.get(target_id)
    if waiter is None:
        waiter = asyncio.Event()
        session.pending_approvals[target_id] = waiter
    session.pending_approvals_meta.setdefault(
        target_id,
        {
            "kind": "tool",
            "tool_name": tool_name,
            "args": dict(tool_args) if isinstance(tool_args, Mapping) else {},
            "task_id": session_ctx.task.id if session_ctx.task is not None else "",
            "prompt": prompt,
        },
    )

    await _emit_approval_requested_from_plugin(
        session=session,
        steerer=steerer,
        target_id=target_id,
        prompt=prompt,
        tool_name=tool_name,
        tool_args=tool_args if isinstance(tool_args, Mapping) else {},
        task_id=(session_ctx.task.id if session_ctx.task is not None else ""),
    )

    await waiter.wait()
    meta = session.pending_approvals_meta.get(target_id, {})
    decision = str(meta.get("decision", "")) or "approve"
    detail = str(meta.get("detail", ""))

    if decision == "reject":
        return {
            "skipped": True,
            "reason": "user_rejected",
            "tool_name": tool_name,
            "detail": detail,
        }
    # APPROVE: fall through so ADK runs the tool normally.
    return None


def _tool_approval_prompt(tool: Any, tool_name: str, tool_args: Any) -> str:
    """Human-readable prompt the UI shows to the human.

    Prefers an explicit ``approval_prompt`` attribute on the tool so
    tool authors can own the copy; otherwise synthesises a short form
    of ``tool_name(arg=value, ...)``.
    """
    explicit = _safe_attr(tool, "approval_prompt", "")
    if explicit:
        return str(explicit)
    if isinstance(tool_args, Mapping) and tool_args:
        parts = [f"{k}={v!r}" for k, v in tool_args.items()]
        return f"Run {tool_name}({', '.join(parts)})?"
    return f"Run {tool_name}()?"


async def _inject_goldfive_planner_instruction(
    *,
    callback_context: Any,
    llm_request: Any,
) -> None:
    """Append :class:`GoldfivePlanner` output to ``llm_request.config.system_instruction``.

    ADK's ``flows/llm_flows/_nl_planning.py`` request-side processor
    gates instruction injection on ``isinstance(planner,
    PlanReActPlanner)`` — so a ``BasePlanner`` subclass that is NOT a
    PlanReAct subclass gets its ``build_planning_instruction`` called
    on the RESPONSE side (via ``process_planning_response``) but never
    on the REQUEST side. That's fine for response filtering but it
    starves the model of goldfive's orchestration context block on
    the turn that matters.

    This helper is the workaround: it detects when the running
    agent's ``.planner`` is a :class:`~goldfive.planners.goldfive_planner.GoldfivePlanner`
    (NOT a ``PlanReActPlanner`` or ``BuiltInPlanner`` — ADK handles
    those on its own via ``_nl_planning``) and appends the planner's
    :meth:`build_planning_instruction` output to the request's system
    instruction using ADK's own ``append_instructions`` method.

    Silent fall-throughs in priority order:

    * ADK not installed or ``BasePlanner`` import fails → skip.
    * Running agent has no ``planner`` attribute or planner is None
      → skip (plain ADK LlmAgent with no goldfive attachment).
    * Planner is a ``PlanReActPlanner`` / ``BuiltInPlanner`` subclass
      → skip (ADK will handle it natively).
    * ``build_planning_instruction`` returns ``None`` / empty →
      skip (planner opted out for this turn).
    * ``llm_request`` lacks ``append_instructions`` → skip (unit-test
      request stubs).
    """
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

    # Find the running agent on the callback_context. ADK exposes it
    # through the invocation context; tests may supply a context that
    # carries ``.agent`` directly.
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
    # If ADK itself will inject for this planner type, skip. ``BuiltInPlanner``
    # never emits a text instruction (it configures thinking on the
    # request instead), ``PlanReActPlanner`` is the one ADK gates on.
    if isinstance(planner, (PlanReActPlanner, BuiltInPlanner)):
        return
    if not isinstance(planner, BasePlanner):
        return
    # Narrow further: the adapter attaches GoldfivePlanner specifically.
    # A custom BasePlanner subclass that is not GoldfivePlanner should
    # be respected by ADK's own (response-side) dispatch only, not
    # re-injected here. This keeps the hook behaviour predictable for
    # users who subclass BasePlanner themselves.
    if not isinstance(planner, GoldfivePlanner):
        return

    # Build the ReadonlyContext ADK expects. When invocation_context
    # is available we use ADK's real class; otherwise we fall back to
    # ``callback_context`` itself (test stubs carrying ``.state`` work
    # through GoldfivePlanner's tolerant _extract_state).
    readonly = callback_context
    try:
        from google.adk.agents.readonly_context import ReadonlyContext  # type: ignore

        if inv_ctx is not None:
            readonly = ReadonlyContext(inv_ctx)
    except Exception as exc:  # noqa: BLE001 — use fallback
        log.debug(
            "_inject_goldfive_planner_instruction: ReadonlyContext unavailable: %s",
            exc,
        )

    instruction = planner.build_planning_instruction(readonly, llm_request)
    if not instruction:
        return

    append = getattr(llm_request, "append_instructions", None)
    if not callable(append):
        # Best-effort write directly into ``config.system_instruction``
        # when the test stub doesn't carry the helper. Preserves the
        # existing value if it's a string.
        config = getattr(llm_request, "config", None)
        if config is None:
            return
        existing = getattr(config, "system_instruction", None)
        if not existing:
            try:
                config.system_instruction = instruction
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_inject_goldfive_planner_instruction: could not set system_instruction: %s",
                    exc,
                )
        elif isinstance(existing, str):
            try:
                config.system_instruction = existing + "\n\n" + instruction
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_inject_goldfive_planner_instruction: could not append system_instruction: %s",
                    exc,
                )
        return

    try:
        append([instruction])
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "_inject_goldfive_planner_instruction: append_instructions raised: %s",
            exc,
        )


# R3 (F2 alternative) — runtime tool-surface hint.
#
# Tier 1's F2 wanted to mutate ``llm_request.config.tools`` mid-callback
# to remove agents whose tasks have all completed, but mutating the
# tool list mid-flight is fragile (ADK caches declarations on
# ``tools_dict`` and the model API rejects requests where the
# function declarations don't match the names referenced in
# ``contents``). R3 takes the non-invasive route: keep all tools
# available, but pre-emptively tell the LLM — at every model call —
# which agents still have PENDING work and which agents' work is
# already DONE. The LLM then has structural guidance to choose the
# right next action without us having to alter the tool surface.
#
# Why a prefix marker: the hint is per-call. Each
# ``before_model_callback`` invocation must inject the CURRENT plan
# state, not accumulate prior snapshots. ``system_instruction`` in
# ADK is a single string, so we detect-and-strip any previous
# goldfive hint by its ``[GOLDFIVE PLAN-STATE HINT —`` opener before
# appending the fresh one.
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


def _inject_runtime_tools_hint(
    *,
    callback_context: Any,
    llm_request: Any,
    session: Any,
) -> None:
    """Inject (or refresh) the runtime tool-surface hint on ``llm_request``.

    R3 (F2 alternative). Builds the current plan-state hint via
    :func:`_build_runtime_tools_hint` and writes it to
    ``llm_request.config.system_instruction``. If a prior goldfive
    hint is present (detected by :data:`_RUNTIME_TOOLS_HINT_PREFIX`),
    it is stripped first so we don't accumulate snapshots across calls.

    Best-effort: never raises. A None hint (no plan, or plan with no
    informative groups) is a no-op so we don't pollute the prompt with
    empty markers.
    """
    hint = _build_runtime_tools_hint(session)

    config = _safe_attr(llm_request, "config", None)
    if config is None:
        return
    existing = getattr(config, "system_instruction", None)

    # Strip any prior hint regardless of whether we'll re-inject. A
    # None ``hint`` (plan disappeared or all groups empty) should still
    # remove the stale marker block from the request.
    if isinstance(existing, str) and _RUNTIME_TOOLS_HINT_PREFIX in existing:
        try:
            config.system_instruction = _strip_prior_runtime_tools_hint(existing) or None
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "_inject_runtime_tools_hint: could not strip prior hint: %s",
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
                "_inject_runtime_tools_hint: append_instructions raised: %s",
                exc,
            )
            # fall through to direct write

    # Fallback for stubs that lack ``append_instructions`` (unit tests).
    if not existing:
        try:
            config.system_instruction = hint
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "_inject_runtime_tools_hint: could not set system_instruction: %s",
                exc,
            )
    elif isinstance(existing, str):
        try:
            config.system_instruction = existing + "\n\n" + hint
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "_inject_runtime_tools_hint: could not append system_instruction: %s",
                exc,
            )


#: Default per-LLM-call wall-clock budget (milliseconds) enforced by
#: :class:`_GoldfiveADKPlugin` (goldfive#271 follow-up). When a single
#: ADK LLM dispatch exceeds this budget, the plugin emits a CRITICAL
#: ``LLM_CALL_TIMEOUT`` drift and flags the invocation for cancel so
#: subsequent callbacks short-circuit. Set to ``0`` (or any negative
#: int) to disable the watcher entirely. Default 1800000ms (30 minutes)
#: is the pathological-hang ceiling for slow local models on
#: compute-bound generation (e.g. Qwen 35B on slide generation or
#: multi-step research synthesis). The watcher's job is catching wedged
#: invocations, not enforcing latency SLOs — operators who want a
#: tighter SLO pass an explicit ``llm_call_timeout_ms`` to
#: :func:`make_adk_plugin`.
DEFAULT_LLM_CALL_TIMEOUT_MS: int = 1_800_000


def _make_cancelled_llm_response() -> Any:
    """Return a synthetic ``LlmResponse`` representing a cancelled call.

    Returned from :meth:`_GoldfiveADKPlugin.before_model_callback` when
    the active invocation has been flagged for cooperative cancel.
    Per ADK's ``BasePlugin.before_model_callback`` contract a non-``None``
    return value short-circuits the LLM dispatch and is propagated as
    the response — returning ``None`` lets the request proceed normally,
    which is the source of the demo-v12.log regression where a single
    ``LLM_CALL_TIMEOUT`` drift fired multiple times on the same
    invocation. Lazy import keeps this module loadable without
    ``google.adk`` installed; on import error we fall back to a plain
    sentinel object — ADK still treats any non-``None`` return as a
    short-circuit, so the LLM call is skipped either way.
    """
    try:
        from google.adk.models.llm_response import LlmResponse  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception:  # noqa: BLE001
        return {"goldfive_cancelled": True}
    try:
        return LlmResponse(
            content=genai_types.Content(
                parts=[genai_types.Part(text="[goldfive: cancelled]")],
                role="model",
            ),
            turn_complete=True,
        )
    except Exception:  # noqa: BLE001
        # Fallback to a bare LlmResponse if Content construction
        # fails on a future ADK schema change. Any non-None still
        # short-circuits per ADK's contract.
        try:
            return LlmResponse()
        except Exception:  # noqa: BLE001
            return {"goldfive_cancelled": True}


def make_adk_plugin(
    *,
    name: str = "goldfive_adk_plugin",
    host_agent_name: str = "",
    agent_tool_cap: int = 16,
    llm_call_timeout_ms: int = DEFAULT_LLM_CALL_TIMEOUT_MS,
) -> Any:
    """Build the ADK plugin class bound to goldfive's protocol.

    The class is built lazily so this module can be imported without
    ``google.adk`` installed. The plugin routes the callbacks we care
    about (``before_run``, ``after_run``, ``before_model``,
    ``before_tool``, ``after_model``, ``on_event``, ``on_tool_error``)
    through the :class:`SessionContext` stashed on ADK state.

    ``host_agent_name`` is the fallback name rendered into
    ``goldfive.available_tasks`` entries whose task has no explicit
    assignee — typically the wrapped root agent's name.

    ``agent_tool_cap`` is the maximum number of ``AgentTool`` spawns
    the plugin will tolerate in a single top-level invocation before
    emitting a ``RUNAWAY_DELEGATION`` drift and signalling cancel.
    Set to ``0`` or a negative value to disable. See goldfive#130 —
    the cap is the backstop against a coordinator whose prompt
    describes a pipeline and keeps delegating forever.

    ``llm_call_timeout_ms`` (goldfive#271 follow-up) is the per-LLM-call
    wall-clock budget. When a single ADK LLM dispatch exceeds this
    duration, the plugin emits a CRITICAL ``LLM_CALL_TIMEOUT`` drift
    and flags the invocation for cooperative cancel. The current
    in-flight LLM call is NOT terminated mid-stream (ADK doesn't
    expose a hook for that, and forcing task cancellation across the
    HTTP transport is fragile); instead, the next callback the
    invocation reaches short-circuits via the existing cancel-state
    plumbing. This is a safety net against runaway thinking-token
    explosions (Qwen Q4 emitting 9961 completion tokens in 9.6 minutes,
    demo-v8.log) — without it a single bad turn can wedge the run for
    minutes. Set to ``0`` or any negative int to disable the watcher.
    Default ``DEFAULT_LLM_CALL_TIMEOUT_MS`` (1800000 / 30 minutes) —
    sized as the pathological-hang ceiling for slow local models on
    compute-bound generation, not an SLO.
    """
    try:
        from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
    except ImportError as exc:  # pragma: no cover — tested via importorskip
        raise ImportError("goldfive.adapters.adk requires 'pip install goldfive[adk]'") from exc

    class _GoldfiveADKPlugin(BasePlugin):  # type: ignore[misc, valid-type]
        """Routes ADK callbacks into the goldfive steerer + state protocol."""

        # Class-level discriminator: lets
        # :func:`session_context_from_invocation` (and any future
        # walkers of ``InvocationContext.plugin_manager.plugins``)
        # identify a goldfive plugin instance without importing the
        # closure-local class.
        __goldfive_adk_plugin__: bool = True

        def __init__(self) -> None:
            super().__init__(name=name)
            self._host_agent_name = host_agent_name
            self._agent_tool_cap = agent_tool_cap
            # Per-LLM-call wall-clock budget (goldfive#271 follow-up).
            # ``0`` or negative disables the watcher entirely. See
            # ``make_adk_plugin`` docstring.
            self._llm_call_timeout_ms = int(llm_call_timeout_ms)
            # Active :class:`SessionContext` for the invocation that is
            # currently driving this plugin's runner. Set by
            # :meth:`ADKAdapter.invoke` before ``run_async`` and cleared
            # in its ``finally`` block. Callbacks prefer this field over
            # the ADK-state lookup because ADK's
            # :class:`~google.adk.sessions.in_memory_session_service.InMemorySessionService`
            # returns a **shallow copy** of the stored session on every
            # ``get_session`` call (see ``_light_copy`` /
            # ``copy.copy(session.state)``) — so a SessionContext written
            # into the adapter's own ``get_session`` copy never reaches
            # the fresh copy that ``runner.run_async`` materialises for
            # the invocation, and the callbacks would see an empty state
            # and silently fall through to the ACK shim.
            #
            # A module-level fallback path (``_session_context_from_callback``)
            # is kept so unit tests that stash a ``SessionContext`` in a
            # plain dict they control still work — the state-based lookup
            # there is authoritative for those synthetic harnesses.
            self._active_ctx: SessionContext | None = None
            # Overlay-model reconciler (goldfive#141). Attached by
            # :meth:`ADKAdapter.invoke_passthrough` before ``run_async``;
            # cleared in its ``finally``. When present, the plugin
            # forwards ``before_agent_callback`` / ``after_agent_callback``
            # and delegation observations to the reconciler so it can
            # transition plan tasks based on observed agent activity.
            # None outside the overlay path — ``invoke(task)`` and
            # ``invoke_follow_up`` keep the per-task model and do NOT
            # attach a reconciler.
            self._reconciler: Any = None
            # Track the top-level invocation_id on the current dispatch so
            # AgentTool-spawned sub-Runners' before_run_callbacks can
            # attribute themselves with a ``parent_invocation_id``.
            self._top_invocation_id: str = ""
            # Per-invocation tool-call counters and last-text buffers
            # keyed by ADK ``invocation_id``. Feeds the
            # CONFABULATION_RISK classifier in ``after_run_callback``:
            # a research-shaped task that completes with zero tool calls
            # and non-empty text is the suspicious pattern worth
            # surfacing. Reset per invocation so nested AgentTool
            # sub-Runners get their own counters.
            self._invocation_tool_calls: dict[str, int] = {}
            self._invocation_last_text: dict[str, str] = {}
            # AgentTool-per-invoke counter (goldfive#130). Scoped to
            # the current top-level invocation; reset in
            # :meth:`clear_active_context`. When the counter exceeds
            # :attr:`_agent_tool_cap` the plugin sets
            # :attr:`runaway_delegation_tripped`, emits a
            # ``RUNAWAY_DELEGATION`` drift, and short-circuits
            # subsequent AgentTool calls in the same invocation.
            self._agent_tool_spawn_count: int = 0
            # One-shot flag: True once the cap has been exceeded in the
            # current invocation. The adapter's ``invoke`` loop reads
            # this to break out of ``run_async`` cleanly — the drift
            # event has already been emitted.
            self.runaway_delegation_tripped: bool = False
            # Per-LLM-call instrumentation (goldfive#172). Keyed by
            # ADK ``invocation_id`` — ``before_model_callback`` stashes
            # the start time + request chars + message count here and
            # ``after_model_callback`` pops it to compute
            # ``llm.call.duration_ms``. Since ADK fires before/after
            # pairs synchronously for a single invocation, a single
            # slot per invocation_id is sufficient (nested AgentTool
            # sub-Runners get their own invocation_id). Each entry is
            # a small dict to keep the payload auditable in tests.
            self._invocation_llm_pending: dict[str, dict[str, Any]] = {}
            # Tool-loop drift detector (goldfive#181). Observes every
            # tool call the plugin sees in ``after_tool_callback`` and
            # fires a ``LOOPING_REASONING`` drift when any of the
            # three configured patterns (exact / name / alternating)
            # trips. Thresholds are sourced from
            # :func:`~goldfive.drift.tool_loops.resolve_thresholds`,
            # which prefers an installed
            # :class:`~goldfive.config.ToolLoopConfig` (goldfive#225,
            # wired by :func:`goldfive.wrap`) and falls back to
            # ``GOLDFIVE_TOOL_LOOP_*`` env vars and then the module
            # defaults. Lazy import so the plugin module stays
            # importable without the drift helpers materialised —
            # matches the pattern used for the confabulation classifier.
            from goldfive.drift import tool_loops as _tool_loops

            self._tool_loop_tracker = _tool_loops.ToolLoopTracker(
                **_tool_loops.resolve_thresholds()
            )
            # Reporting-tool names that indicate forward task progress
            # and therefore can clear the tool-loop tracker's window
            # for the current (invocation, agent) key — SUBJECT to the
            # acknowledged-success gate in ``after_tool_callback``
            # (goldfive#192). Matches the set exposed by the adapter's
            # state-transition protocol, plus the approval-gate
            # reporter which is the other call an agent uses to signal
            # forward progress on a running task.
            self._progress_reporting_tools: frozenset[str] = frozenset(
                {
                    "report_task_started",
                    "report_task_progress",
                    "report_task_completed",
                    "report_task_failed",
                    "report_task_blocked",
                    "report_awaiting_approval",
                }
            )
            # Cooperative cancellation state (goldfive#251 Stream C / 7a).
            # Authoritative source for the cancel-requested flag:
            # ``dict[str, CancellationRequest]`` keyed by ``invocation_id``.
            # The steerer writes entries here via
            # :meth:`request_invocation_cancel` on the adapter; every
            # adapter callback checks the dict at the top of its body
            # and, when an entry matches the current invocation_id,
            # consumes it (read + clear — same-invocation re-entry won't
            # re-cancel) and short-circuits the dispatch.
            #
            # Stored on the plugin instance rather than ADK session.state
            # because ``InMemorySessionService`` shallow-copies state on
            # every ``get_session`` (see the same rationale that drove
            # goldfive#170 for the _active_ctx field), which would make
            # cross-callback reads unreliable. The state-protocol module
            # (:data:`_adk_state_protocol.KEY_CANCEL_REQUESTED`) documents
            # the key semantics so external consumers see a stable
            # contract; the plugin's dict is the live source of truth.
            self._cancel_state: dict[str, Any] = {}
            # Sticky-cancelled set (goldfive#271 follow-up). Once a
            # callback has consumed a ``_cancel_state`` entry for an
            # invocation, the id is added here and EVERY subsequent
            # callback for the same invocation short-circuits — even
            # though ``_cancel_state`` itself was popped (consume-once
            # semantic) so that the InvocationCancelled sink event
            # only fires once. Without this, after the first
            # cancellation, follow-up ``before_model_callback`` /
            # ``before_tool_callback`` invocations on the SAME
            # invocation_id would see an empty ``_cancel_state`` and
            # let the LLM call / tool dispatch proceed — exactly the
            # bug reproduced in /tmp/demo-v12.log where a single
            # ``LLM_CALL_TIMEOUT`` drift on ``e-1e9e1f05`` was
            # followed by 3 more watcher firings on the SAME
            # invocation. Cleared in :meth:`clear_active_context` and
            # in :meth:`after_run_callback` for the top-level id.
            self._cancelled_invocations: set[str] = set()
            # Parent/child invocation map for cancel propagation.
            # ``dict[str, str]`` mapping ``invocation_id ->
            # parent_invocation_id``. Populated on every
            # ``before_run_callback`` that observes a fresh invocation_id
            # with a known parent (the top-level invocation_id pinned on
            # the first ``before_run``). Consumed by
            # :meth:`request_invocation_cancel` so that cancelling a
            # parent also flags its spawned sub-invocations — this is
            # how a cancelled coordinator's mid-flight AgentTool child
            # short-circuits cleanly instead of emitting its turn and
            # poisoning the parent's history.
            self._invocation_parents: dict[str, str] = {}
            # Per-invocation pinned task_id (goldfive#264 — aggressive
            # pin resolution). ``dict[str, str]`` mapping
            # ``invocation_id -> task_id`` populated by
            # :meth:`_stamp_current_task_id` whenever a pin lands.
            # Consumed by signal 5 of :meth:`_pin_current_task_id_for_agent`
            # so a child invocation can read its parent's pinned task
            # without racing on the single ``goldfive.current_task_id``
            # slot. Cleared on ``clear_active_context``.
            self._invocation_pinned_task_id: dict[str, str] = {}
            # Per-invocation asyncio.Task registry — MIGRATED in Phase 3.5
            # (goldfive#271 component 1) off this plugin instance and onto
            # :class:`~goldfive.orchestration_store.OrchestrationStore`.
            # The plugin no longer owns the registry; the store owns it
            # keyed by ``Session.id``.
            #
            # Backwards-compat shim: tests and the steerer's cancel path
            # still touch ``plugin._invocation_tasks``. The
            # :class:`_InvocationTaskRegistryView` exposed below presents
            # the OrchestrationStore-backed registry through the legacy
            # ``dict``-shaped attribute so existing call sites continue
            # to work unchanged. Look up via OrchestrationStore directly
            # for new code (see
            # :meth:`~goldfive.orchestration_store.OrchestrationStore.get_invocation_task`).
            #
            # Why migrated: per-session orchestration state belongs on
            # OrchestrationStore — Phase 0 state-ownership contract.
            # PR #303 placed the registry on the plugin as a bridge; this
            # phase relocates the storage while preserving the API.
            self._invocation_tasks: _InvocationTaskRegistryView = _InvocationTaskRegistryView(self)
            # Goldfive boundary wrapper bookkeeping (goldfive#271 Phase 3.5
            # component 1). The set tracks every invocation_id whose
            # ``before_agent_callback`` emitted ``InvocationBoundaryEntered``
            # and is therefore expected to emit a paired ``Exited`` —
            # whether that comes from ``after_agent_callback`` (normal
            # completion) or from the canonical ``except CancelledError``
            # in :meth:`ADKAdapter._invoke_internal` (cancel/error path).
            # Per-invocation pin ensures the pair is exactly-once even
            # when ADK skips ``after_agent_callback`` on cancel.
            self._boundary_entered_invocations: set[str] = set()

        def set_active_context(self, ctx: SessionContext) -> None:
            """Attach the ``SessionContext`` for the running invocation.

            Called once per :meth:`ADKAdapter.invoke` before
            ``runner.run_async``. The plugin's callback methods prefer
            this context over any value stashed in ADK session state
            (which is an unreliable channel because InMemorySessionService
            copies state on every get). Overwriting a non-``None`` value
            is accepted — sequential invocations reuse the adapter.

            The dynamic-instruction resolver and planner's per-turn
            injection reach this same field via
            :func:`session_context_from_invocation` so the goldfive
            :class:`~goldfive.types.Session` is reachable from inside
            an ADK callback frame without depending on a write to ADK
            ``session.state``. Phase 2.0 of goldfive#271 — closes
            goldfive#275.
            """
            self._active_ctx = ctx
            # Reset the runaway-delegation bookkeeping for the new
            # invocation so a prior trip doesn't leak into this one.
            self._agent_tool_spawn_count = 0
            self.runaway_delegation_tripped = False

        def clear_active_context(self) -> None:
            """Release the active ``SessionContext`` reference.

            Called from ``ADKAdapter.invoke``'s ``finally`` block. Safe
            to call when no context is active.
            """
            # Phase 3.5: clear OrchestrationStore-backed registry BEFORE
            # dropping ``_active_ctx`` so the registry view can still
            # resolve the session to clear. Once ``_active_ctx = None``
            # the view no-ops and the OrchestrationStore-side bucket
            # would leak across dispatches on the same plugin instance.
            try:
                self._invocation_tasks.clear()
            except Exception as exc:  # noqa: BLE001
                log.debug("clear_active_context: registry clear raised: %s", exc)
            self._active_ctx = None
            self._top_invocation_id = ""
            self._agent_tool_spawn_count = 0
            self.runaway_delegation_tripped = False
            self._reconciler = None
            # Drop any straggling per-LLM-call metrics entries
            # (goldfive#172). Normal operation pops each entry in
            # ``after_model_callback``; this catches the case where a
            # model turn errored between before/after and never paired.
            # Also cancel any pending wall-clock watcher so it doesn't
            # leak into the next dispatch (goldfive#271 follow-up).
            for pending in self._invocation_llm_pending.values():
                watcher = pending.get("watcher") if isinstance(pending, dict) else None
                if watcher is not None and not watcher.done():
                    watcher.cancel()
            self._invocation_llm_pending.clear()
            # Drop per-(invocation, agent) tool-loop ring buffers so
            # state from the just-finished dispatch doesn't leak into
            # the next one on the same plugin instance (goldfive#181).
            self._tool_loop_tracker.clear()
            # Drop any lingering cancellation state / parent map so the
            # next dispatch starts clean (goldfive#251). A request that
            # was never consumed means the callback path never ran —
            # still safe to drop because the invocation it targeted is
            # gone, and keeping it would misfire on an unrelated future
            # invocation_id collision.
            self._cancel_state.clear()
            self._cancelled_invocations.clear()
            self._invocation_parents.clear()
            # goldfive#264 — drop per-invocation pin map.
            self._invocation_pinned_task_id.clear()
            # goldfive#271 follow-up — invocation-task handle clearing
            # already happened at the top of this method (Phase 3.5
            # restructuring) so the registry view could still resolve
            # the OrchestrationStore. Nothing to do here.
            # goldfive#271 Phase 3.5 — drop boundary-pair bookkeeping.
            # Anything still in this set means the boundary's exit emit
            # never fired, which only happens when the dispatch was
            # torn down outside the canonical try/finally arc (test
            # scaffolding, bug). We do NOT emit a synthetic Exited here
            # because the active context is already being cleared and
            # the sinks may be unreachable; the operator-visible signal
            # is the missing pair on the wire, surfaced by the audit.
            self._boundary_entered_invocations.clear()

        # --- Cooperative cancellation (goldfive#251 Stream C / 7a) -----

        def request_invocation_cancel(
            self,
            *,
            invocation_id: str,
            request: Any,
            propagate_to_children: bool = True,
            cancel_inflight_task: bool = False,
        ) -> list[str]:
            """Flag ``invocation_id`` (and optionally its descendants)
            for cooperative cancellation.

            Called by :class:`~goldfive.steerer.DefaultSteerer` when a
            drift at CRITICAL severity (or a user-initiated cancel)
            warrants aborting an in-flight adapter dispatch. Writes an
            entry to the plugin's ``_cancel_state`` dict keyed by the
            invocation id; every adapter callback consults the dict at
            the top of its body and short-circuits when its own id
            matches.

            When ``propagate_to_children`` is True (the default), the
            recorded parent/child map is walked breadth-first and an
            entry is added for every transitive descendant of
            ``invocation_id`` so an in-flight AgentTool sub-invocation
            is also flagged. The returned list contains every id that
            was flagged — the target itself plus any descendants — so
            callers can sink-report the full set if they want.

            Tree-agnostic: the parent/child map is per-invocation, the
            plugin has no notion of "coordinator" vs "sub-agent", and
            every level in the tree is flagged the same way.

            When ``cancel_inflight_task`` is True (goldfive#271 follow-
            up — v15 concurrent-invocation bug), ALSO fire
            ``task.cancel()`` on the registered asyncio.Task for each
            flagged invocation (deferred via
            :meth:`asyncio.AbstractEventLoop.call_soon` so an inline
            same-task caller still completes its current emission
            work before the cancel lands). Default False so the
            existing pre-refine cancel paths in
            :meth:`DefaultSteerer._handle_drift` keep their flag-only
            semantics — the post-refine path
            :meth:`DefaultSteerer._cancel_inflight_for_revision`
            opts in explicitly so the cancel only fires AFTER a
            superseding plan has been installed.
            """
            if not invocation_id:
                return []
            flagged: list[str] = [str(invocation_id)]
            # Walk the parent/child map when propagation is enabled.
            # Order is unspecified; deduplication happens inside
            # ``descendants_of_invocation`` via a seen-set.
            if propagate_to_children:
                try:
                    from goldfive.adapters import _adk_state_protocol as _sp_local

                    descendants = _sp_local.descendants_of_invocation(
                        {_sp_local.KEY_INVOCATION_PARENTS: self._invocation_parents},
                        invocation_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_GoldfiveADKPlugin.request_invocation_cancel: descendant walk raised: %s",
                        exc,
                    )
                    descendants = []
                flagged.extend(descendants)
            for flagged_id in flagged:
                # Preserve the first-writer's request for each id so a
                # parent cancel with reason="user_steer" doesn't get
                # silently overwritten by a descendant-propagation pass
                # reusing the parent's request object. When a descendant
                # already has a distinct request pending (uncommon but
                # possible), keep the earlier one — the more-recent
                # overwrite semantics are only for same-id re-writes
                # from the steerer itself.
                self._cancel_state.setdefault(flagged_id, request)
            # Fire ``task.cancel()`` on the registered asyncio.Task for
            # each flagged invocation when the caller opted in via
            # ``cancel_inflight_task=True`` (goldfive#271 follow-up —
            # v15 concurrent-invocation bug). Without this step the
            # cancel-state flag short-circuits only the NEXT
            # ``before_model_callback`` / ``before_tool_callback``; the
            # already-running LLM streaming call (potentially 10+
            # minutes on a slow model) keeps generating output that
            # triggers more drift, and the steerer's `refine_steer`
            # span ends up overlapping the coordinator span by the
            # full duration of the in-flight LLM call. Cancelling the
            # task raises ``CancelledError`` inside
            # ``ADKAdapter._invoke_internal``'s ``async for`` over
            # ``runner.run_async``; the existing heal path emits the
            # synthetic ``function_response`` so ADK's
            # function_call/_response invariant holds.
            #
            # IMPORTANT — defer via :meth:`asyncio.AbstractEventLoop.call_soon`:
            # Drift-detection paths can reach this method synchronously
            # from inside the same asyncio task that drives the
            # dispatch (e.g. PlanReconciler emits PLAN_DIVERGENCE from
            # ``before_agent_callback`` and awaits the steerer's
            # ``_handle_drift`` inline; the steerer's own
            # post-refine helper calls us right before
            # ``_emit_plan_revised``). Calling ``task.cancel()``
            # directly there would schedule a ``CancelledError`` to
            # land on the very next ``await`` in the caller's call
            # chain — including the paired ``_emit_plan_revised``
            # emit — losing the on-the-wire PlanRevised event.
            # ``loop.call_soon`` queues the cancel for the NEXT
            # event-loop turn so the calling
            # ``_handle_drift`` / ``_promote_drift_to_steer`` /
            # ``install_revision_for_drift`` /
            # ``install_revision_for_user_steer`` finishes its
            # emission work before the dispatch task observes the
            # cancellation at its next yield (typically the next
            # iteration of the ``async for`` in the adapter).
            #
            # Best-effort: a missing entry (e.g. the cancel fired
            # between ``before_run`` registering and the LLM stream
            # starting) is OK — the cancel-state flag still
            # short-circuits subsequent callbacks. A done task is also
            # OK — ``cancel()`` no-ops and we move on. A loop that's
            # not running falls back to a direct ``cancel()`` so
            # tests driving the cancel under ``asyncio.run`` still see
            # it applied.
            if cancel_inflight_task:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                for flagged_id in flagged:
                    t = self._invocation_tasks.get(flagged_id)
                    if t is None or t.done():
                        continue
                    if loop is not None:
                        loop.call_soon(_safe_task_cancel, t, flagged_id)
                    else:
                        try:
                            if t.cancel():
                                log.info(
                                    "goldfive.cancel.task: cancelled "
                                    "in-flight task for invocation_id=%s "
                                    "(no running loop; direct cancel)",
                                    flagged_id,
                                )
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "_GoldfiveADKPlugin.request_invocation_cancel: "
                                "task.cancel() for %s raised: %s",
                                flagged_id,
                                exc,
                            )
            return flagged

        def consume_cancel_for_invocation(self, invocation_id: str) -> Any | None:
            """Read the pending cancel for ``invocation_id`` and clear it.

            Callback-facing helper. Returns the
            :class:`~goldfive.types.CancellationRequest` when one was
            pending, or ``None`` otherwise. Clearing before returning
            gives the "cancel fires once" semantic: a re-entry into the
            same callback (e.g. after the LLM call was already skipped
            and a follow-up tool call fires) doesn't re-emit the
            cancelled marker.
            """
            if not invocation_id:
                return None
            return self._cancel_state.pop(str(invocation_id), None)

        def peek_cancel_for_invocation(self, invocation_id: str) -> Any | None:
            """Return the pending cancel for ``invocation_id`` without
            clearing it.

            Diagnostic / test helper. Production callback paths use
            :meth:`consume_cancel_for_invocation`; this method exists so
            the adapter's ``invoke`` loop can check whether a cancel was
            flagged on the current dispatch without side-effecting the
            consume-once semantic.
            """
            if not invocation_id:
                return None
            return self._cancel_state.get(str(invocation_id))

        def is_invocation_cancelled(self, invocation_id: str) -> bool:
            """Return True if ``invocation_id`` is flagged for cancel.

            Sticky: returns True both when an unconsumed
            :class:`~goldfive.types.CancellationRequest` is pending in
            ``_cancel_state`` AND when an earlier callback already
            consumed it (recorded in ``_cancelled_invocations``). The
            sticky bit is what gives us "every subsequent callback for
            the same invocation short-circuits", fixing the demo-v12.log
            regression where the watcher fired multiple times on a
            single invocation. See class-init comment on
            ``_cancelled_invocations`` for the full rationale.
            """
            if not invocation_id:
                return False
            inv_id = str(invocation_id)
            if inv_id in self._cancelled_invocations:
                return True
            return self._cancel_state.get(inv_id) is not None

        def set_reconciler(self, reconciler: Any) -> None:
            """Attach a :class:`~goldfive.reconciler.PlanReconciler`.

            Set by :meth:`ADKAdapter.invoke_passthrough` for the
            duration of a single overlay-model invocation. The plugin
            will forward before/after-agent + delegation observations
            to the reconciler's hooks. Pass ``None`` to detach.
            """
            self._reconciler = reconciler

        def _resolve_ctx(self, adk_ctx: Any) -> SessionContext | None:
            """Return the live ``SessionContext`` or ``None`` if unbound.

            Prefers the plugin-local ``_active_ctx`` (authoritative for
            real ADK runs) but falls through to the state-dict lookup
            for unit tests that drive the plugin with a hand-built
            ``tool_context`` holding a populated state mapping. The two
            paths are never inconsistent in production — only one is
            populated at a time.
            """
            if self._active_ctx is not None:
                return self._active_ctx
            return _session_context_from_callback(adk_ctx)

        # --- Invocation lifecycle --------------------------------------

        async def before_run_callback(self, *, invocation_context: Any) -> None:
            """Seed ``goldfive.*`` state on the LIVE invocation session and
            emit :class:`AgentInvocationStarted`.

            This is the RELIABILITY-CRITICAL state-protocol write path.
            ``invocation_context.session`` is the session ADK is actually
            running the invocation against — writes here are visible to
            every subsequent callback and tool on the same session,
            including AgentTool-spawned sub-Runners (whose own
            ``before_run_callback`` fires with their own session and
            therefore gets its own authoritative seed).

            Previously the adapter wrote these keys against a session
            fetched via ``session_service.get_session`` — which
            ``InMemorySessionService`` returns as a shallow copy, so the
            writes landed on a stranded dict the runner never saw. That
            was flagged "best-effort" and left the state-protocol keys
            unreliable, which is unacceptable for correctness — see
            docs/design/TASK-LIFECYCLE.md §5.
            """
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None:
                return None

            # Determine parent_invocation_id: if a top-level one is
            # already pinned on this plugin, we're in a nested AgentTool
            # sub-Runner. The top-level adapter invocation is the first
            # one to fire before_run_callback.
            inv_id = str(_safe_attr(invocation_context, "invocation_id", "") or "")
            parent_inv_id = ""
            if self._top_invocation_id:
                parent_inv_id = self._top_invocation_id
            else:
                # First before_run for this dispatch — pin it so nested
                # AgentTool sub-Runners can attribute themselves below.
                self._top_invocation_id = inv_id

            # Register the parent/child relationship for cooperative
            # cancellation propagation (goldfive#251). A cancel targeting
            # the parent id can then flag this child id without the
            # steerer having to know the tree shape.
            if inv_id and parent_inv_id:
                self._invocation_parents[inv_id] = parent_inv_id

            # Register the asyncio.Task currently driving this invocation
            # (goldfive#271 follow-up — v15 concurrent-invocation bug).
            # ADK's ``Runner._exec_with_plugin`` runs ``before_run_callback``
            # in the same task as the dispatch, so
            # ``asyncio.current_task()`` here IS the task whose
            # cancellation will raise ``CancelledError`` inside the
            # adapter's ``async for event in runner.run_async(...)``
            # loop. :meth:`request_invocation_cancel` consults this map
            # to fire ``task.cancel()`` when a refine supersedes the
            # in-flight plan.
            if inv_id:
                current = asyncio.current_task()
                if current is not None:
                    self._invocation_tasks[inv_id] = current

            # Cooperative-cancellation check (goldfive#251 Stream C / 7a).
            # If the adapter / steerer flagged this invocation before
            # ``run_async`` actually yielded to ADK, short-circuit the
            # whole dispatch: skip the state-protocol write, skip the
            # AgentInvocationStarted emit, and emit an
            # InvocationCancelled sink event so operators see the
            # cancel in harmonograf. The outer ``ADKAdapter.invoke``
            # loop honours the short-circuit via
            # :meth:`peek_cancel_for_invocation` (below).
            if inv_id and self.is_invocation_cancelled(inv_id):
                pending = self._cancel_state.get(inv_id)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id)
                    self._cancelled_invocations.add(inv_id)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id,
                        agent_name="",
                        request=request,
                    )
                return None

            # Reset the per-invocation counters so the CONFABULATION_RISK
            # check in ``after_run_callback`` sees only the tool calls
            # and final text produced by THIS invocation. Nested
            # AgentTool sub-Runners fire their own before_run_callback
            # with a fresh ``invocation_id`` so they get their own slot.
            if inv_id:
                self._invocation_tool_calls[inv_id] = 0
                self._invocation_last_text[inv_id] = ""

            # Phase 2.0 of goldfive#271 — V1 (initial seed) and V2
            # (orchestration-state bridge) both deleted. The dynamic
            # instruction resolver and GoldfivePlanner now read
            # goldfive Session directly via the SessionContext stash;
            # no callback-time write to ADK state is required.

            # Emit AgentInvocationStarted. Best-effort: observability
            # only, so a sink / proto issue must not block the run.
            agent_name = str(_safe_attr(ctx, "host_agent_name", "") or "") or self._host_agent_name
            running_agent = _safe_attr(invocation_context, "agent", None)
            running_agent_name = str(_safe_attr(running_agent, "name", "") or "")
            if running_agent_name:
                agent_name = running_agent_name
            await self._emit_observability(
                "agent_invocation_started",
                agent_name=agent_name,
                task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                invocation_id=inv_id,
                parent_invocation_id=parent_inv_id,
            )
            # Feed the trajectory-level activity buffer used by the
            # GOAL_DRIFT judge (goldfive#143). Duck-typed: custom
            # steerers without ``note_agent_activity`` fall through to
            # a no-op. Always safe — the recorder does not itself
            # trigger an LLM call.
            note_activity = getattr(ctx.steerer, "note_agent_activity", None)
            if note_activity is not None:
                try:
                    note_activity(
                        ctx.session,
                        kind="agent_invocation_started",
                        agent_name=agent_name,
                        task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("before_run_callback: note_agent_activity raised: %s", exc)
            return None

        async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            """Pin ``goldfive.current_task_id`` for the starting sub-agent and
            forward an agent-turn start to the overlay reconciler.

            Fires once per agent invocation (including sub-agents
            inside AgentTool sub-Runners). Two jobs:

            1. **Task-id pinning (goldfive#191 Layer 1).** At delegation
               time, find the unique plan task whose
               ``assignee_agent_id == agent.name`` and whose status is
               PENDING or RUNNING, and stamp its id onto both the live
               ADK ``session.state`` and the goldfive orchestration
               ``session.state`` under the ``goldfive.current_task_id``
               key. Sub-agents' reporting-tool handlers read this key
               as a fallback when the model's tool call omits
               ``task_id``, so delegated work doesn't retry-loop on
               the structured ``missing_task_id`` error.

               Zero matches (off-plan agent) and multiple matches
               (ambiguous — a coordinator with two pending siblings
               for the same assignee) intentionally leave the state
               unset. The ``missing_task_id`` error path still fires
               in those cases — better an explicit rejection than a
               mis-attributed report.

            2. **Overlay reconciler forward.** When a
               :class:`~goldfive.reconciler.PlanReconciler` is
               attached, we forward ``agent.name``, the invocation
               id, and the parent_invocation_id (the outer runner's
               id, when the current invocation is nested inside an
               ``AgentTool`` sub-Runner) so the reconciler can
               resolve parent chains for contextual matching
               (goldfive#151).
            """
            agent_name = str(_safe_attr(agent, "name", "") or "")
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            parent_inv_id = ""
            if inv_id and self._top_invocation_id and inv_id != self._top_invocation_id:
                parent_inv_id = self._top_invocation_id

            # Goldfive boundary entry (goldfive#271 Phase 3.5 component 1).
            # Mark this invocation_id as having entered the boundary and
            # emit the paired ``InvocationBoundaryEntered`` sink event.
            # The boundary is the single canonical try/finally arc the
            # rest of Phase 3.5 hangs off of: the steerer's
            # cancel-inflight-task path can now target the registered
            # task knowing the canonical exit emit will land regardless
            # of which path observes the CancelledError.
            #
            # Done BEFORE the cooperative-cancel short-circuit so a
            # cancel flagged before the boundary fires still produces
            # the entry/exit pair (the cancel checkpoint below emits
            # InvocationCancelled then short-circuits, and the boundary
            # exit emit fires from ``after_agent_callback`` or from
            # the canonical CancelledError catch in _invoke_internal).
            ctx_task = self._active_ctx.task if self._active_ctx else None
            ctx_task_id = str(_safe_attr(ctx_task, "id", "") or "") if ctx_task else ""
            if inv_id:
                await self._emit_boundary_entered(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    task_id=ctx_task_id,
                )

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a).
            # When a cancel was flagged for this invocation_id (by the
            # steerer's CRITICAL-severity ladder path, or by a programmatic
            # caller), consume the request, emit an InvocationCancelled
            # sink event, and short-circuit the callback — the agent's
            # turn is skipped entirely. Done BEFORE the pinning /
            # reconciler forward work so a cancelled turn leaves no
            # side-effects on orchestration state.
            if inv_id and self.is_invocation_cancelled(inv_id):
                pending = self._cancel_state.get(inv_id)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id)
                    self._cancelled_invocations.add(inv_id)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id,
                        agent_name=agent_name,
                        request=request,
                    )
                # Boundary exit emit fires here too — the agent's turn
                # is being skipped, so the boundary's ``finally`` is
                # logically about to run. Reason="cancelled" so the
                # paired InvocationCancelled event explains why.
                if inv_id:
                    await self._emit_boundary_exited(
                        invocation_id=inv_id,
                        agent_name=agent_name,
                        task_id=ctx_task_id,
                        reason="cancelled",
                    )
                return None

            # Pin the runtime-reasoning agent on the goldfive Session
            # so the reasoning-drift judge can attribute reasoning to
            # the agent that actually produced it (rather than the
            # static plan assignee). When a coordinator delegates to a
            # child via AgentTool the child reasons under the parent's
            # task pin; reading ``task.assignee_agent_id`` on the
            # judge dispatch path mis-attributed every drift to the
            # coordinator. Last writer wins — reasoning is sequential
            # within an invocation, so by the time the model-response
            # callback fires the most recent ``before_agent_callback``
            # is the agent producing the reasoning. Pre-pin races
            # (the LLM judge calling before this fires for the first
            # time) read empty and the consumer falls back to
            # ``task.assignee_agent_id`` for back-compat.
            try:
                gf_session = self._active_ctx.session if self._active_ctx else None
                if gf_session is not None and agent_name:
                    gf_session.current_agent_id = agent_name
            except Exception as exc:  # noqa: BLE001 — pinning must never raise
                log.debug(
                    "before_agent_callback: current_agent_id pin raised: %s",
                    exc,
                )

            # Layer 1: pin the starting sub-agent's task_id so its
            # reporting-tool calls can default the arg from state
            # (goldfive#191). Best-effort: a raise here must never
            # break the invocation. goldfive#264 — multi-signal
            # resolution, async to allow the PinResolved sink emit.
            try:
                await self._pin_current_task_id_for_agent(
                    agent_name=agent_name,
                    callback_context=callback_context,
                    invocation_id=inv_id,
                    parent_invocation_id=parent_inv_id,
                )
            except Exception as exc:  # noqa: BLE001 — pinning must never raise
                log.debug(
                    "before_agent_callback: current_task_id pin raised: %s",
                    exc,
                )

            reconciler = self._reconciler
            if reconciler is None:
                return None
            try:
                await reconciler.on_before_agent(
                    agent_name=agent_name,
                    invocation_id=inv_id,
                    parent_invocation_id=parent_inv_id,
                )
            except TypeError:
                # Custom reconciler without the #151 kwarg — fall back
                # to the pre-#151 signature. Keeps back-compat.
                try:
                    await reconciler.on_before_agent(
                        agent_name=agent_name,
                        invocation_id=inv_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_agent_callback: reconciler.on_before_agent raised: %s",
                        exc,
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_agent_callback: reconciler.on_before_agent raised: %s",
                    exc,
                )
            return None

        async def _pin_current_task_id_for_agent(
            self,
            *,
            agent_name: str,
            callback_context: Any,
            invocation_id: str = "",
            parent_invocation_id: str = "",
        ) -> None:
            """Aggressive multi-signal pin resolution (goldfive#264).

            Reframed from the original "exactly-1 DAG-ready single
            match" gate after live operator feedback: *if an agent was
            invoked, something precipitated the call*. The previous
            implementation gave up silently on zero/multiple matches,
            so the agent ran without a pin, every reporting-tool call
            short-circuited as a no-op, and the orchestration loop
            stagnated.

            Replaced with an 8-signal resolution ladder, picking the
            first signal that yields a single best candidate:

            1. **Delegation-site pin** — the parent's ``before_tool_callback``
               already stamped a per-function_call_id pin on
               ``pending_delegations``. Authoritative.
            2. **DAG-ready exactly-1** — assignee match, status PENDING /
               RUNNING, all upstream predecessors COMPLETED. The pre-
               existing happy path; preserved as the fast short-circuit.
            3. **Tool-arg scoring over DAG-ready candidates** — when (2)
               returns 2+ matches, score each against the parent
               AgentTool's args via :func:`_score_candidates_by_args`
               and pick the winner. Falls through on tie.
            4. **DAG gate relaxed** — drop the upstream-completion check
               and retry the assignee+status filter. The agent was
               invoked so something precipitated it; surface the pin
               and emit a WARNING + low-confidence sink event so
               operators see the anomaly. Tool-arg scoring breaks ties.
            5. **Parent-pin downstream** — if a parent invocation has a
               pinned task on this plugin, prefer candidates whose id
               is a downstream of the parent's pinned task in
               ``plan.edges``.
            6. **Recent drift / correction targeting** — if a
               ``goldfive.pending_corrections.<agent>.<task_id>`` entry
               exists in session state, pin the named task. The plan-
               revision pipeline writes these for CORRECT-kind
               supersedes; pinning to one is a strong signal that the
               agent was invoked specifically to act on the correction.
            7. **Assignee normalization fallback** — re-run signals
               2-4 with bare/compound forms of ``agent_name`` swapped
               in. PR #215 fixed planner-side normalisation; this is
               defence-in-depth for transcripts that retain a compound
               assignee.
            8. **Low-confidence best-guess** — if every prior signal
               failed, pick the highest-scoring candidate from the
               full assignee+status set (or, lacking any, the highest-
               scoring PENDING/RUNNING task in the plan) and emit a
               ``pin_resolved_low_confidence`` sink event so the LLM
               gets to continue and the operator sees the weakening.

            Every successful pin emits a single ``pin_resolved`` (dict-
            envelope) sink event labelled with ``via_signal`` so
            harmonograf and operators can chart how often the happy
            path is short-circuiting vs. how often the relaxed signals
            are firing — a leading indicator that pin invariants are
            weakening.

            Silent on every failure mode (no agent name, no ctx, no
            plan, state not a mapping). Never raises.
            """
            if not agent_name:
                return
            ctx = self._resolve_ctx(callback_context)
            if ctx is None:
                return
            plan = _safe_attr(ctx.session, "plan", None)
            if plan is None:
                return
            tasks = _safe_attr(plan, "tasks", None) or ()
            # Import here so the type is available without forcing a
            # top-level import for a rarely-hot-path enum compare.
            from goldfive.types import TaskStatus, task_upstream_ready

            tasks_list = list(tasks)

            # ---- Signal 1: delegation-site pin -------------------------
            # goldfive#241 Item 3-bis — delegation-site pin takes
            # precedence. If the parent coordinator's
            # ``before_tool_callback`` stamped a task_id for THIS
            # AgentTool dispatch (keyed by function_call_id), trust it.
            gf_state_early = _safe_attr(ctx.session, "state", None)
            if isinstance(gf_state_early, Mapping):
                pend = gf_state_early.get(_PENDING_DELEGATIONS_KEY)
                if isinstance(pend, Mapping) and pend:
                    for raw_entry in pend.values():
                        # goldfive#266 — entries may be either the
                        # legacy bare-string task_id or the new
                        # ``{task_id, revision}`` dict. Use the
                        # back-compat extractor.
                        tid = _delegation_pin_task_id(raw_entry)
                        if not tid:
                            continue
                        for task in tasks_list:
                            if str(_safe_attr(task, "id", "") or "") != tid:
                                continue
                            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                            if assignee != agent_name:
                                continue
                            status = _safe_attr(task, "status", None)
                            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                                self._stamp_current_task_id(
                                    ctx=ctx,
                                    task_id=tid,
                                    agent_name=agent_name,
                                    source="delegation_pin",
                                    task=task,
                                    invocation_id=invocation_id,
                                )
                                await self._emit_pin_resolved(
                                    ctx=ctx,
                                    agent_name=agent_name,
                                    task_id=tid,
                                    via_signal="delegation_pin",
                                    score=1.0,
                                    invocation_id=invocation_id,
                                    candidate_count=1,
                                )
                                return

            # Build the assignee+status candidate set once; signals 2-4
            # all reuse it. We also keep the parent's tool args (best-
            # effort) so signals 3 / 4 / 8 can score candidates.
            assignee_candidates = self._candidates_for_agent(tasks_list, agent_name)
            scoring_args = self._scoring_args_for(
                ctx=ctx,
                parent_invocation_id=parent_invocation_id,
            )

            # ---- Signal 2: DAG-ready exactly-1 -------------------------
            dag_ready = self._filter_dag_ready(plan, assignee_candidates, task_upstream_ready)
            if len(dag_ready) == 1:
                task = dag_ready[0]
                task_id = str(_safe_attr(task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="single_match",
                        task=task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="dag_ready_single",
                        score=1.0,
                        invocation_id=invocation_id,
                        candidate_count=1,
                    )
                    return

            # ---- Signal 3: tool-arg scoring over DAG-ready -------------
            if len(dag_ready) > 1 and scoring_args is not None:
                chosen = _score_candidates_by_args(dag_ready, scoring_args)
                if chosen is not None:
                    task_id = str(_safe_attr(chosen, "id", "") or "")
                    if task_id:
                        self._stamp_current_task_id(
                            ctx=ctx,
                            task_id=task_id,
                            agent_name=agent_name,
                            source="arg_scored",
                            task=chosen,
                            invocation_id=invocation_id,
                        )
                        await self._emit_pin_resolved(
                            ctx=ctx,
                            agent_name=agent_name,
                            task_id=task_id,
                            via_signal="arg_scored",
                            score=1.0,
                            invocation_id=invocation_id,
                            candidate_count=len(dag_ready),
                        )
                        return

            # ---- Signal 4: DAG gate relaxed ----------------------------
            # The user's reframe: "if an agent was invoked, something
            # precipitated it." We've lost ground truth already; bind
            # to the most-plausible task and surface the anomaly to
            # operators rather than silent-no-op.
            if assignee_candidates:
                relaxed = assignee_candidates
                if len(relaxed) > 1 and scoring_args is not None:
                    chosen = _score_candidates_by_args(relaxed, scoring_args)
                    if chosen is None:
                        # Tie / no overlap — fall through.
                        chosen = None
                else:
                    chosen = relaxed[0] if len(relaxed) == 1 else None
                if chosen is not None:
                    task_id = str(_safe_attr(chosen, "id", "") or "")
                    if task_id:
                        log.warning(
                            "pin: DAG-gate relaxed, bound %s for agent %s "
                            "(upstreams not yet complete; %d assignee candidates)",
                            task_id,
                            agent_name,
                            len(relaxed),
                        )
                        self._stamp_current_task_id(
                            ctx=ctx,
                            task_id=task_id,
                            agent_name=agent_name,
                            source="dag_relaxed",
                            task=chosen,
                            invocation_id=invocation_id,
                        )
                        await self._emit_pin_resolved(
                            ctx=ctx,
                            agent_name=agent_name,
                            task_id=task_id,
                            via_signal="dag_relaxed",
                            score=0.7,
                            invocation_id=invocation_id,
                            candidate_count=len(relaxed),
                        )
                        return

            # ---- Signal 5: parent-pin downstream -----------------------
            parent_pin_task = self._task_from_parent_pin_downstream(
                plan=plan,
                tasks=tasks_list,
                parent_invocation_id=parent_invocation_id,
                agent_name=agent_name,
                scoring_args=scoring_args,
            )
            if parent_pin_task is not None:
                task_id = str(_safe_attr(parent_pin_task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="parent_pin_downstream",
                        task=parent_pin_task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="parent_pin_downstream",
                        score=0.6,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # ---- Signal 6: recent drift / correction / reasoning ------
            #
            # Phase 1 of goldfive#271 — Signal 6 has two sub-signals,
            # consulted in confidence order:
            #
            #   6a. Reasoning-extracted binding. The LLM judge in
            #       :mod:`goldfive.drift.reasoning_judge` returns a
            #       ``focused_task_id`` + ``focus_confidence`` per
            #       chain-of-thought block. When the steerer recorded
            #       a binding for this agent and confidence is at the
            #       configured threshold, the ladder consumes it as
            #       a real signal — the agent's *stated intent* names
            #       which plan task it's working on, which is a
            #       stronger signal than guessing from token overlap.
            #
            #   6b. Pending-correction targeting (the pre-existing
            #       Signal 6). When a CORRECT-kind supersedes wrote a
            #       ``goldfive.pending_corrections.<agent>.<task>``
            #       key, the ladder pins to the correction target.
            #
            # When 6a doesn't fire (no binding, low-confidence binding,
            # binding's task is no longer PENDING/RUNNING) the ladder
            # falls through to 6b unchanged.
            reasoning_task = self._task_from_reasoning_binding(
                ctx=ctx,
                tasks=tasks_list,
                agent_name=agent_name,
            )
            if reasoning_task is not None:
                task_id = str(_safe_attr(reasoning_task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="reasoning_binding",
                        task=reasoning_task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="reasoning_binding",
                        score=0.85,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            correction_task = self._task_from_pending_correction(
                ctx=ctx,
                tasks=tasks_list,
                agent_name=agent_name,
            )
            if correction_task is not None:
                task_id = str(_safe_attr(correction_task, "id", "") or "")
                if task_id:
                    self._stamp_current_task_id(
                        ctx=ctx,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="correction_target",
                        task=correction_task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="correction_target",
                        score=0.9,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # ---- Signal 7: assignee bare/compound normalisation -------
            normalised_alt = self._alternate_agent_name_form(agent_name)
            if normalised_alt and normalised_alt != agent_name:
                alt_assignee = self._candidates_for_agent(tasks_list, normalised_alt)
                if alt_assignee:
                    alt_dag_ready = self._filter_dag_ready(plan, alt_assignee, task_upstream_ready)
                    chosen = None
                    if len(alt_dag_ready) == 1:
                        chosen = alt_dag_ready[0]
                    elif len(alt_dag_ready) > 1 and scoring_args is not None:
                        chosen = _score_candidates_by_args(alt_dag_ready, scoring_args)
                    elif len(alt_dag_ready) == 0 and len(alt_assignee) == 1:
                        chosen = alt_assignee[0]
                    elif len(alt_assignee) > 1 and scoring_args is not None:
                        chosen = _score_candidates_by_args(alt_assignee, scoring_args)
                    if chosen is not None:
                        task_id = str(_safe_attr(chosen, "id", "") or "")
                        if task_id:
                            log.warning(
                                "pin: assignee normalisation %r->%r found candidate %s",
                                agent_name,
                                normalised_alt,
                                task_id,
                            )
                            self._stamp_current_task_id(
                                ctx=ctx,
                                task_id=task_id,
                                agent_name=agent_name,
                                source="assignee_normalised",
                                task=chosen,
                                invocation_id=invocation_id,
                            )
                            await self._emit_pin_resolved(
                                ctx=ctx,
                                agent_name=agent_name,
                                task_id=task_id,
                                via_signal="assignee_normalised",
                                score=0.5,
                                invocation_id=invocation_id,
                                candidate_count=len(alt_assignee),
                            )
                            return

            # ---- Signal 8: low-confidence best-guess -------------------
            best_guess = self._low_confidence_best_guess(
                tasks=tasks_list,
                agent_name=agent_name,
                scoring_args=scoring_args,
            )
            if best_guess is not None:
                task, score = best_guess
                task_id = str(_safe_attr(task, "id", "") or "")
                if task_id:
                    log.warning(
                        "pin: low-confidence best-guess %s for agent %s "
                        "(score=%.2f); every prior signal failed",
                        task_id,
                        agent_name,
                        score,
                    )
                    self._stamp_current_task_id(
                        ctx=ctx,
                        task_id=task_id,
                        agent_name=agent_name,
                        source="low_confidence",
                        task=task,
                        invocation_id=invocation_id,
                    )
                    await self._emit_pin_resolved(
                        ctx=ctx,
                        agent_name=agent_name,
                        task_id=task_id,
                        via_signal="low_confidence",
                        score=score,
                        invocation_id=invocation_id,
                        candidate_count=0,
                    )
                    return

            # All signals failed AND no best-guess candidate at all
            # (empty plan / agent has nothing remotely matching). Leave
            # state unset — there's nothing better than the existing
            # ``missing_task_id`` error path here.
            log.debug(
                "before_agent_callback: pin resolution exhausted all "
                "signals for agent %s; leaving state unset",
                agent_name,
            )

        # ---- Pin resolution helpers (goldfive#264) --------------------

        @staticmethod
        def _candidates_for_agent(tasks: list[Any], agent_name: str) -> list[Any]:
            """Return PENDING/RUNNING tasks whose assignee matches ``agent_name``.

            Pre-DAG candidate set used by signals 2/3/4/8. Pure helper,
            no side effects.
            """
            from goldfive.types import TaskStatus

            out: list[Any] = []
            for task in tasks:
                assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                if assignee != agent_name:
                    continue
                status = _safe_attr(task, "status", None)
                if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                    out.append(task)
            return out

        @staticmethod
        def _filter_dag_ready(
            plan: Any,
            candidates: list[Any],
            task_upstream_ready: Any,
        ) -> list[Any]:
            """Filter ``candidates`` to those whose upstream is COMPLETED."""
            ready: list[Any] = []
            for task in candidates:
                task_id = str(_safe_attr(task, "id", "") or "")
                if not task_id:
                    continue
                try:
                    if task_upstream_ready(plan, task_id):
                        ready.append(task)
                except Exception as exc:  # noqa: BLE001 — never raise from pin
                    log.debug(
                        "_filter_dag_ready: task_upstream_ready raised "
                        "for %s: %s — treating as not-ready",
                        task_id,
                        exc,
                    )
            return ready

        def _scoring_args_for(
            self,
            *,
            ctx: SessionContext,
            parent_invocation_id: str,
        ) -> Any:
            """Return a token-bag string-or-mapping for tool-arg scoring, or ``None``.

            Signal 3/4/8 score candidates by token overlap with whatever
            we have for "what was this agent invoked for". In priority
            order:

            1. Parent invocation's AgentTool tool-call args, recorded
               on ``pending_delegations[<fc_id>].tool_args`` by
               :meth:`_pin_delegation_task_id` (F7 — #265 followup).
               This is the strongest signal: the parent literally
               said "go do *this*". Skipped silently when the entry
               is pre-F7-shaped (string-only or
               ``{task_id, revision}`` without ``tool_args``) or when
               the recorded args produce zero meaningful tokens (e.g.
               an opaque blob / empty dispatch).
            2. The active steer body — operators usually phrase the
               steer with task-named tokens.
            3. The session's goal summary — broad fallback that at
               least disambiguates by domain vocabulary.
            4. Goals on the session itself (some test harnesses don't
               populate the orchestration-state mirror).

            Returns whatever non-empty string-or-mapping we found, or
            ``None`` when there's nothing to score against (in which
            case the score-based signals fall through silently).
            """
            session_state = _safe_attr(ctx.session, "state", None)
            # 1) Parent's AgentTool tool-call args. Iterate
            # pending_delegations and merge any tool_args payloads we
            # find — pending entries are usually short-lived (stamped
            # right before the sub-agent fires, cleared after report)
            # so the union is a reasonable heuristic when multiple
            # parallel dispatches landed. Skip when the merged dict
            # tokenises to nothing.
            if isinstance(session_state, Mapping):
                pend = session_state.get(_PENDING_DELEGATIONS_KEY)
                if isinstance(pend, Mapping) and pend:
                    merged: dict[str, Any] = {}
                    for raw_entry in pend.values():
                        ta = _delegation_pin_tool_args(raw_entry)
                        if ta is None:
                            continue
                        merged.update(ta)
                    if merged and _tokenize_for_matching(
                        " ".join(f"{k} {v}" for k, v in merged.items())
                    ):
                        return merged
            # 2) Active steer body / goals summary mirrored on session.state.
            if isinstance(session_state, Mapping):
                steer_body = session_state.get("goldfive.active_steer.body", "")
                if isinstance(steer_body, str) and steer_body.strip():
                    return steer_body
                goals_summary = session_state.get("goldfive.goals_summary", "")
                if isinstance(goals_summary, str) and goals_summary.strip():
                    return goals_summary
            # 3) Goals on the session itself.
            goals = _safe_attr(ctx.session, "goals", None) or []
            if goals:
                summaries = [str(_safe_attr(g, "summary", "") or "") for g in goals]
                joined = " ".join(s for s in summaries if s).strip()
                if joined:
                    return joined
            # 4) Nothing to score against.
            _ = parent_invocation_id  # intentionally unused; kept for future plumbing
            return None

        def _task_from_parent_pin_downstream(
            self,
            *,
            plan: Any,
            tasks: list[Any],
            parent_invocation_id: str,
            agent_name: str,
            scoring_args: Any,
        ) -> Any:
            """Signal 5 — pick a candidate downstream of the parent's pin.

            Reads ``self._invocation_pinned_task_id[parent_invocation_id]``
            to find the parent's pin, then scans ``plan.edges`` for
            tasks whose id is a downstream of the parent pin. Among
            those, restrict to assignee-matching PENDING/RUNNING tasks
            (re-using :meth:`_candidates_for_agent`). If multiple,
            fall back to tool-arg scoring; if zero, return ``None``.
            """
            if not parent_invocation_id:
                return None
            parent_pin = self._invocation_pinned_task_id.get(parent_invocation_id, "")
            if not parent_pin:
                return None
            edges = _safe_attr(plan, "edges", None) or ()
            downstream_ids: set[str] = set()
            for e in edges:
                from_id = str(_safe_attr(e, "from_task_id", "") or "")
                if from_id != parent_pin:
                    continue
                to_id = str(_safe_attr(e, "to_task_id", "") or "")
                if to_id:
                    downstream_ids.add(to_id)
            if not downstream_ids:
                return None
            assignee_candidates = self._candidates_for_agent(tasks, agent_name)
            preferred: list[Any] = [
                t
                for t in assignee_candidates
                if str(_safe_attr(t, "id", "") or "") in downstream_ids
            ]
            if not preferred:
                return None
            if len(preferred) == 1:
                return preferred[0]
            if scoring_args is not None:
                return _score_candidates_by_args(preferred, scoring_args)
            return None

        @staticmethod
        def _task_from_reasoning_binding(
            *,
            ctx: SessionContext,
            tasks: list[Any],
            agent_name: str,
        ) -> Any:
            """Signal 6a — pin to the task named by a reasoning-extracted binding.

            Phase 1 of goldfive#271. Consults
            :class:`~goldfive.orchestration_store.OrchestrationStore`
            for a binding stamped by the steerer's reasoning-judge
            background path; the binding's
            ``focused_task_id`` is matched against the plan and the
            matching PENDING/RUNNING task is returned (or ``None`` if
            the binding doesn't exist, was below the steerer's
            configured confidence threshold and was thus never recorded,
            or names a task that's no longer in the active set).

            The store handles agent-name normalisation (compound /
            bare-form fallback) so a coordinator that fires the judge
            against ``"agent_x"`` and a sub-runner pinning under
            ``"client42:agent_x"`` both find the same binding.
            """
            from goldfive.types import TaskStatus

            store = OrchestrationStore.for_session(_safe_attr(ctx, "session", None))
            binding = store.get_reasoning_extracted_binding(agent_name)
            if binding is None or not binding.task_id:
                return None
            tasks_by_id = {str(_safe_attr(t, "id", "") or ""): t for t in tasks}
            task = tasks_by_id.get(binding.task_id)
            if task is None:
                return None
            status = _safe_attr(task, "status", None)
            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                return task
            return None

        @staticmethod
        def _task_from_pending_correction(
            *,
            ctx: SessionContext,
            tasks: list[Any],
            agent_name: str,
        ) -> Any:
            """Signal 6 — pin to a task targeted by a pending correction.

            Reads ``goldfive.pending_corrections.<agent>.<task_id>``
            keys off the orchestration session state (written by
            :mod:`goldfive._correction_injection` for CORRECT-kind
            supersedes). When at least one entry exists for the bare
            form of ``agent_name``, returns the first matching plan
            task that is PENDING or RUNNING.
            """
            from goldfive.types import TaskStatus

            state = _safe_attr(ctx.session, "state", None)
            if not isinstance(state, Mapping):
                return None
            # Strip a compound prefix on agent_name to match the
            # bare-form keys the writer uses.
            bare_agent = agent_name.rsplit(":", 1)[-1]
            prefix = f"goldfive.pending_corrections.{bare_agent}."
            target_task_ids: list[str] = []
            for key in state:
                if not isinstance(key, str):
                    continue
                if not key.startswith(prefix):
                    continue
                tid = key[len(prefix) :]
                if tid:
                    target_task_ids.append(tid)
            if not target_task_ids:
                return None
            # Resolve to the first PENDING/RUNNING plan task with a
            # matching id. We do not require assignee-equality here —
            # the writer keyed on the agent already, and we trust that
            # the correction is for this agent's turn.
            tasks_by_id = {str(_safe_attr(t, "id", "") or ""): t for t in tasks}
            for tid in target_task_ids:
                task = tasks_by_id.get(tid)
                if task is None:
                    continue
                status = _safe_attr(task, "status", None)
                if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                    return task
            return None

        @staticmethod
        def _alternate_agent_name_form(agent_name: str) -> str:
            """Return the bare/compound alternate of ``agent_name``.

            ``"compound:foo"`` -> ``"foo"``; ``"foo"`` -> ``""`` (no
            compound prefix to add — there's no convention for which
            prefix to try without context). Signal 7 only exercises
            the strip direction, since the planner-side normalisation
            (PR #215) already strips on the way in.
            """
            if not agent_name:
                return ""
            if ":" in agent_name:
                return agent_name.rsplit(":", 1)[-1]
            return ""

        def _low_confidence_best_guess(
            self,
            *,
            tasks: list[Any],
            agent_name: str,
            scoring_args: Any,
        ) -> tuple[Any, float] | None:
            """Signal 8 — return a best-guess (task, confidence) pair.

            Last-resort: if every prior signal failed but there's
            something resembling work for this agent in the plan, pick
            the most-plausible task and tag the resolution as
            low-confidence so the operator-visible event makes the
            uncertainty explicit.

            Strategy: assignee-match candidates (any status that's
            PENDING/RUNNING) scored against tool args. If empty, no
            pin — there's nothing better than ``missing_task_id``.
            """
            assignee_candidates = self._candidates_for_agent(tasks, agent_name)
            if not assignee_candidates:
                return None
            if len(assignee_candidates) == 1:
                # Single assignee match but DAG-relaxed already would
                # have caught this — getting here means the relaxed
                # path didn't run (e.g. signals 5/6 fell through with
                # parent or correction context but neither matched).
                # Pin with low confidence.
                return assignee_candidates[0], 0.4
            if scoring_args is not None:
                chosen = _score_candidates_by_args(assignee_candidates, scoring_args)
                if chosen is not None:
                    return chosen, 0.4
            # Tie / no scoring available — pick the first deterministically
            # so behaviour is reproducible across runs. The low-
            # confidence event makes the uncertainty visible.
            return assignee_candidates[0], 0.2

        async def _emit_pin_resolved(
            self,
            *,
            ctx: SessionContext,
            agent_name: str,
            task_id: str,
            via_signal: str,
            score: float,
            invocation_id: str,
            candidate_count: int,
        ) -> None:
            """Emit a ``pin_resolved`` (or ``pin_resolved_low_confidence``)
            sink event so operators see which signal landed the pin.

            Uses :func:`goldfive.events.make_event` (dict envelope)
            because the proto schema doesn't yet carry a PinResolved
            slot — adding one would expand scope. Best-effort: every
            failure is logged and swallowed.
            """
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            kind = (
                "pin_resolved_low_confidence" if via_signal == "low_confidence" else "pin_resolved"
            )
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            payload: dict[str, Any] = {
                "agent_name": str(agent_name or ""),
                "task_id": str(task_id or ""),
                "via_signal": str(via_signal or ""),
                "score": float(score),
                "invocation_id": str(invocation_id or ""),
                "candidate_count": int(candidate_count),
            }
            try:
                from goldfive.events import emit, make_event  # noqa: PLC0415

                evt = make_event(run_id, seq, kind, payload, session_id=session_id)
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug("_emit_pin_resolved: failed to emit %s: %s", kind, exc)

        def _stamp_current_task_id(
            self,
            *,
            ctx: SessionContext,
            task_id: str,
            agent_name: str,
            source: str,
            task: Any = None,
            invocation_id: str = "",
        ) -> None:
            """Write ``task_id`` onto goldfive ``Session.state`` for the sub-agent.

            Shared by every signal in :meth:`_pin_current_task_id_for_agent`
            (goldfive#264). The ``source`` label threads into the log
            line so operators see which signal landed the pin
            (delegation_pin / single_match / arg_scored / dag_relaxed /
            parent_pin_downstream / correction_target /
            assignee_normalised / low_confidence).

            ``invocation_id`` (when non-empty) is also recorded onto
            ``self._invocation_pinned_task_id`` so signal 5 of a
            child invocation's pin can read this invocation's pin
            without racing on the single ``goldfive.current_task_id``
            slot.

            Phase 2.1 of goldfive#271 — V3 of the audit catalog. The
            pin lands on goldfive's own ``Session.state`` exclusively,
            via :class:`~goldfive.orchestration_store.OrchestrationStore`.
            The dynamic-instruction resolver and reporting handlers
            both read goldfive Session via the plugin reference
            (:func:`session_context_from_invocation`) — no callback-time
            mutation of ADK ``session.state`` happens here anymore.
            """
            # goldfive#266 — resolve the plan revision in effect at this
            # write so the report-time classifier in
            # :mod:`goldfive.reporting` can distinguish a fresh pin from
            # one set under an older revision. Defensive: missing /
            # malformed plan reads as 0 so legacy paths keep behaving
            # like the initial revision.
            plan_for_rev = _safe_attr(ctx.session, "plan", None)
            try:
                pin_revision = int(_safe_attr(plan_for_rev, "revision_index", 0) or 0)
            except (TypeError, ValueError):
                pin_revision = 0
            store = OrchestrationStore.for_session(ctx.session)
            title = ""
            if task is not None:
                title = str(_safe_attr(task, "title", "") or "")
            store.set_pin_current_task(
                task_id,
                source=_BINDING_SOURCE_BY_LADDER.get(source, BindingSource.UNKNOWN),
                revision=pin_revision,
                title=title,
            )
            log.info(
                "goldfive.pin.set: task_id=%s agent=%s source=%s revision=%d invocation_id=%s",
                task_id,
                agent_name,
                source,
                pin_revision,
                invocation_id or "-",
            )
            # goldfive#264 — record per-invocation pin so child
            # invocations can resolve their parent's pin (signal 5).
            if invocation_id and task_id:
                self._invocation_pinned_task_id[invocation_id] = task_id

        def _pin_delegation_task_id(
            self,
            *,
            ctx: SessionContext,
            tool_context: Any,
            to_agent: str,
            tool_args: Any,
        ) -> None:
            """Stamp a per-``function_call_id`` task pin for an AgentTool
            dispatch so parallel same-agent invocations don't race on the
            single ``goldfive.current_task_id`` slot.

            goldfive#241 Item 3-bis. Resolution algorithm:

            1. Collect PENDING/RUNNING tasks whose ``assignee_agent_id``
               matches ``to_agent``.
            2. Keep only the tasks whose upstream edges all point at
               COMPLETED predecessors (DAG-aware; a task whose
               dependency isn't done yet cannot be the target of THIS
               dispatch).
            3. If exactly one candidate, that's the pin.
            4. If multiple candidates, score each against ``tool_args``
               by keyword overlap with its ``title + description`` and
               pick the top. Ties or zero-overlap fall through to "no
               pin" — the sub-agent's ``before_agent_callback`` takes
               over via the legacy single-match path.
            5. If zero candidates, no pin.

            Phase 2.1 of goldfive#271 — V4 of the audit catalog. The
            pin lands on goldfive's ``Session.state`` exclusively,
            via :meth:`OrchestrationStore.set_pending_delegation`. The
            sub-invocation's ``before_tool_callback`` reads it back via
            :func:`_resolve_pinned_task_id` (which now consults goldfive
            Session via the plugin reference). Silent on every failure
            mode — the worst case is we fall through to the legacy
            behaviour.
            """
            if not to_agent:
                return
            fc_id = _function_call_id_from_tool_context(tool_context)
            if not fc_id:
                return
            plan = _safe_attr(ctx.session, "plan", None)
            if plan is None:
                return
            tasks = _safe_attr(plan, "tasks", None) or ()
            edges = _safe_attr(plan, "edges", None) or ()
            from goldfive.types import TaskStatus

            completed_ids: set[str] = set()
            for t in tasks:
                if _safe_attr(t, "status", None) is TaskStatus.COMPLETED:
                    tid = str(_safe_attr(t, "id", "") or "")
                    if tid:
                        completed_ids.add(tid)

            def _upstream_ok(task_id: str) -> bool:
                for e in edges:
                    to_id = str(_safe_attr(e, "to_task_id", "") or "")
                    if to_id != task_id:
                        continue
                    from_id = str(_safe_attr(e, "from_task_id", "") or "")
                    if from_id and from_id not in completed_ids:
                        return False
                return True

            candidates: list[Any] = []
            for task in tasks:
                assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                if assignee != to_agent:
                    continue
                status = _safe_attr(task, "status", None)
                if status is not TaskStatus.PENDING and status is not TaskStatus.RUNNING:
                    continue
                tid = str(_safe_attr(task, "id", "") or "")
                if not tid or not _upstream_ok(tid):
                    continue
                candidates.append(task)

            if not candidates:
                return
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = _score_candidates_by_args(candidates, tool_args)
                if chosen is None:
                    log.debug(
                        "before_tool_callback: %d candidates for %s; "
                        "args did not disambiguate — no pin",
                        len(candidates),
                        to_agent,
                    )
                    return
            task_id = str(_safe_attr(chosen, "id", "") or "")
            if not task_id:
                return
            try:
                pin_revision = int(_safe_attr(plan, "revision_index", 0) or 0)
            except (TypeError, ValueError):
                pin_revision = 0
            store = OrchestrationStore.for_session(ctx.session)
            store.set_pending_delegation(
                fc_id,
                task_id=task_id,
                revision=pin_revision,
                tool_args=tool_args if isinstance(tool_args, Mapping) else None,
            )
            log.info(
                "goldfive.delegation_pin.set: task_id=%s fc_id=%s "
                "sub_agent=%s candidates=%d revision=%d",
                task_id,
                fc_id,
                to_agent,
                len(candidates),
                pin_revision,
            )

        async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            """Forward an agent-turn end to the overlay reconciler and
            emit ``InvocationBoundaryExited`` to close the goldfive
            boundary wrapper (goldfive#271 Phase 3.5 component 1)."""
            agent_name = str(_safe_attr(agent, "name", "") or "")
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            parent_inv_id = ""
            if inv_id and self._top_invocation_id and inv_id != self._top_invocation_id:
                parent_inv_id = self._top_invocation_id
            summary = self._invocation_last_text.get(inv_id, "") if inv_id else ""

            # Boundary exit emit (Phase 3.5). Done in a finally-style
            # block so a reconciler raise below cannot prevent the
            # boundary from closing — the boundary is the canonical
            # exit-point contract; observability must not depend on
            # third-party reconciler hooks.
            ctx_task = self._active_ctx.task if self._active_ctx else None
            ctx_task_id = str(_safe_attr(ctx_task, "id", "") or "") if ctx_task else ""

            reconciler = self._reconciler
            try:
                if reconciler is not None:
                    try:
                        await reconciler.on_after_agent(
                            agent_name=agent_name,
                            invocation_id=inv_id,
                            error=None,
                            summary=summary,
                            parent_invocation_id=parent_inv_id,
                        )
                    except TypeError:
                        try:
                            await reconciler.on_after_agent(
                                agent_name=agent_name,
                                invocation_id=inv_id,
                                error=None,
                                summary=summary,
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "after_agent_callback: reconciler.on_after_agent raised: %s",
                                exc,
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_agent_callback: reconciler.on_after_agent raised: %s",
                            exc,
                        )
            finally:
                if inv_id:
                    await self._emit_boundary_exited(
                        invocation_id=inv_id,
                        agent_name=agent_name,
                        task_id=ctx_task_id,
                        reason="completed",
                    )
            return None

        async def after_run_callback(self, *, invocation_context: Any) -> None:
            """Emit :class:`AgentInvocationCompleted` when an invocation ends.

            Fires once per runner invocation: top-level (goldfive
            dispatch) and per-AgentTool sub-Runner.

            Also runs the cheap structural CONFABULATION_RISK classifier
            (issue #128) before cleanup: if the current task's
            description reads like external-data work but the
            invocation produced non-empty text with zero tool calls, a
            goldfive.drift.classify_confabulation_risk call surfaces the
            suspicious pattern as an INFO drift through the same path
            AGENT_REFUSAL uses.
            """
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None:
                return None
            inv_id = str(_safe_attr(invocation_context, "invocation_id", "") or "")
            agent_name = str(_safe_attr(ctx, "host_agent_name", "") or "") or self._host_agent_name
            running_agent = _safe_attr(invocation_context, "agent", None)
            running_agent_name = str(_safe_attr(running_agent, "name", "") or "")
            if running_agent_name:
                agent_name = running_agent_name

            # Confabulation-risk check. We run this BEFORE emitting
            # "agent_invocation_completed" so the drift lands in the
            # event stream adjacent to the invocation it describes,
            # matching the AGENT_REFUSAL ordering. Gated on:
            #   * a live steerer + task,
            #   * assignee on the task matching the agent that just
            #     finished (so nested AgentTool sub-Runners do not
            #     misattribute their inner text to the outer task),
            #   * the counters we tracked per invocation_id.
            await self._maybe_emit_confabulation_risk(
                ctx=ctx,
                inv_id=inv_id,
                finishing_agent_name=agent_name,
            )

            # If the finishing invocation is the top-level one, release
            # the pin so a subsequent invoke() on the same plugin gets a
            # fresh dispatch.
            if self._top_invocation_id and self._top_invocation_id == inv_id:
                self._top_invocation_id = ""
            # Drop the per-invocation counters now that the check has
            # run — keeps the dict bounded across long-lived plugins.
            if inv_id:
                self._invocation_tool_calls.pop(inv_id, None)
                self._invocation_last_text.pop(inv_id, None)
                # Drop the sticky-cancelled marker for this invocation
                # so a future invocation_id collision (e.g. test
                # harness reuse) doesn't inherit the cancel bit.
                self._cancelled_invocations.discard(inv_id)
                # Drop the registered task handle (goldfive#271 follow-
                # up). The invocation has finished — successfully,
                # cancelled, or errored — and the task will not be
                # cancellable beyond this point. Leaving the entry in
                # the map would be harmless but would let an unrelated
                # late-firing ``request_invocation_cancel`` no-op (the
                # task is done) AT BEST or, worse, target a future
                # invocation that happens to reuse the id.
                self._invocation_tasks.pop(inv_id, None)
            await self._emit_observability(
                "agent_invocation_completed",
                agent_name=agent_name,
                task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                invocation_id=inv_id,
                summary="",
            )
            # Feed the GOAL_DRIFT activity buffer + counter
            # (goldfive#143). Duck-typed: custom steerers without
            # these hooks fall through cleanly. The counter is
            # trajectory-level and persists across task transitions.
            if ctx.steerer is not None:
                note_activity = getattr(ctx.steerer, "note_agent_activity", None)
                if note_activity is not None:
                    try:
                        note_activity(
                            ctx.session,
                            kind="agent_invocation_completed",
                            agent_name=agent_name,
                            task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_run_callback: note_agent_activity raised: %s",
                            exc,
                        )
                note_agent_turn = getattr(ctx.steerer, "note_agent_turn", None)
                if note_agent_turn is not None:
                    try:
                        await note_agent_turn(ctx.session)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("after_run_callback: note_agent_turn raised: %s", exc)
            return None

        async def _maybe_emit_confabulation_risk(
            self,
            *,
            ctx: SessionContext,
            inv_id: str,
            finishing_agent_name: str,
        ) -> None:
            """Fire ``CONFABULATION_RISK`` if the invocation shape is suspicious.

            See :func:`goldfive.drift.classify_confabulation_risk` for
            the exact trigger conditions. Silent when any precondition
            fails — tasks without a clear assignee, sub-agent
            invocations whose assignee does not match, or invocations
            with no tracked state all fall through to no-op so we never
            over-report.
            """
            if ctx.steerer is None or ctx.task is None:
                return
            task = ctx.task
            task_id = str(_safe_attr(task, "id", "") or "")
            if not task_id:
                # Out-of-scope per issue #128: tasks without a clear id
                # / assignee fall through to no-op.
                return
            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
            if assignee and finishing_agent_name and assignee != finishing_agent_name:
                # Nested AgentTool sub-Runner whose agent is not the
                # task's owner — let the outer runner's after_run fire
                # the check against the outer text.
                return
            tool_calls = self._invocation_tool_calls.get(inv_id, 0)
            final_text = self._invocation_last_text.get(inv_id, "")
            from goldfive.drift import classify_confabulation_risk

            drift = classify_confabulation_risk(
                task=task,
                tool_call_count=tool_calls,
                output_text=final_text,
            )
            if drift is None:
                return
            observation = _as_observation(
                kind="confabulation_risk",
                detail=drift.detail,
                raw={
                    "tool_call_count": tool_calls,
                    "output_text": final_text[:500],
                },
                task=task,
                agent_id=finishing_agent_name or self._host_agent_name,
            )
            # Route through steerer.observe so the drift hits the same
            # pipeline AGENT_REFUSAL uses (DriftDetected sink event,
            # severity-based refine decision). We pre-classified above
            # so the steerer's classify_* cascade will no-op on the
            # observation dict — we still fire it explicitly by calling
            # _handle_drift directly when the steerer exposes it.
            handle = getattr(ctx.steerer, "_handle_drift", None)
            if handle is not None:
                try:
                    await handle(drift, ctx.session)
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_maybe_emit_confabulation_risk: _handle_drift raised: %s",
                        exc,
                    )
            # Fallback for steerer stubs without _handle_drift: feed the
            # observation through observe() so custom steerers still
            # see the signal.
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_maybe_emit_confabulation_risk: steerer.observe raised: %s",
                    exc,
                )

        async def _emit_runaway_delegation_drift(
            self,
            *,
            ctx: SessionContext,
            from_agent: str,
            to_agent: str,
            task_id: str,
            invocation_id: str,
            spawn_count: int,
        ) -> None:
            """Emit a ``RUNAWAY_DELEGATION`` drift at CRITICAL severity.

            Built and dispatched directly (not through
            ``steerer.observe`` → ``detect_drift``) because the cap is
            an observed invariant violation, not a heuristic. Routes
            through ``steerer._handle_drift`` when available so the
            planner gets a refine hook; falls back to a direct
            ``_emit_drift_detected`` if the steerer doesn't expose
            ``_handle_drift``. Failures swallowed — observability
            cannot block the invocation, and the adapter's invoke loop
            will break out on ``runaway_delegation_tripped`` regardless.
            """
            steerer = ctx.steerer
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
                    "_emit_runaway_delegation_drift: cannot import types: %s",
                    exc,
                )
                return

            detail = (
                f"AgentTool-per-invoke cap of {self._agent_tool_cap} "
                f"exceeded (spawn #{spawn_count}); last delegation "
                f"{from_agent or '?'} -> {to_agent or '?'} at invocation "
                f"{invocation_id or '?'}"
            )
            drift = DriftEvent(
                kind=DriftKind.RUNAWAY_DELEGATION,
                severity=DriftSeverity.CRITICAL,
                detail=detail,
                current_task_id=task_id,
                current_agent_id=from_agent or self._host_agent_name,
            )
            # Prefer _handle_drift so the full refine/emit path fires.
            handle = getattr(steerer, "_handle_drift", None)
            if callable(handle):
                try:
                    await handle(drift, ctx.session)
                    return
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_emit_runaway_delegation_drift: _handle_drift raised: %s",
                        exc,
                    )
            # Fallback: direct sink emission.
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    drift_detected_event,
                    emit,
                )

                run_id = str(_safe_attr(ctx.session, "run_id", "") or "")
                session_id = str(_safe_attr(ctx.session, "id", "") or "") or run_id
                try:
                    seq = ctx.session.next_sequence()
                except Exception:  # noqa: BLE001
                    seq = 0
                evt = drift_detected_event(run_id, seq, drift, session_id=session_id)
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_runaway_delegation_drift: direct sink emit failed: %s",
                    exc,
                )

        async def _emit_pin_unresolved_drift(
            self,
            *,
            ctx: SessionContext,
            agent_name: str,
            tool_name: str,
            candidate_ids: list[str],
        ) -> None:
            """Emit a ``DriftDetected`` for an unresolvable reporting-tool pin.

            Used when ``before_tool_callback`` can't resolve a pin on a
            reporting tool and the current agent has PENDING / RUNNING
            candidates in the plan (so the pin SHOULD have worked — not
            an orchestration-only turn). The tool response to the LLM is
            a bare ``{"acknowledged": True}``; this drift event is the
            operator-visible signal that a stall occurred.

            Uses ``DriftKind.OFF_TOPIC`` with a ``reason=pin_unresolved: …``
            prefix (not a new ``PIN_UNRESOLVED`` proto kind) because the
            invariant is observer visibility, not wire-level classification.
            Sink dispatch fails-safe — observability must not block an
            invocation.
            """
            steerer = ctx.steerer
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
                    "_emit_pin_unresolved_drift: cannot import types: %s",
                    exc,
                )
                return

            detail = (
                f"pin_unresolved: {tool_name} for agent={agent_name or '?'}; "
                f"candidates=[{', '.join(candidate_ids)}]"
            )
            drift = DriftEvent(
                kind=DriftKind.OFF_TOPIC,
                severity=DriftSeverity.WARNING,
                detail=detail,
                current_task_id=str(_safe_attr(ctx.task, "id", "") or ""),
                current_agent_id=agent_name or self._host_agent_name,
            )
            # Direct sink emission (bypass _handle_drift): this signal is
            # purely observability — no refine needed. A pin_unresolved
            # stall gets resolved by the LLM retrying or the orchestrator
            # intervening, not by a plan revision.
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    drift_detected_event,
                    emit,
                )

                run_id = str(_safe_attr(ctx.session, "run_id", "") or "")
                session_id = str(_safe_attr(ctx.session, "id", "") or "") or run_id
                try:
                    seq = ctx.session.next_sequence()
                except Exception:  # noqa: BLE001
                    seq = 0
                evt = drift_detected_event(run_id, seq, drift, session_id=session_id)
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_pin_unresolved_drift: direct sink emit failed: %s",
                    exc,
                )

        async def _emit_observability(self, kind: str, **fields: Any) -> None:
            """Fan out an observability event to the session's sinks.

            Sinks live on the steerer (``steerer._sinks``) — same
            channel the approval-requested path uses. Failures are
            swallowed: observability cannot block an invocation.
            """
            ctx = self._active_ctx
            if ctx is None:
                return
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    agent_invocation_completed_event,
                    agent_invocation_started_event,
                    delegation_observed_event,
                    emit,
                )

                # Only thread session_id when caller hasn't supplied it
                # explicitly, so the plugin's stamping stays back-compat
                # with callers that set the field themselves.
                fields.setdefault("session_id", session_id)
                if kind == "agent_invocation_started":
                    evt = agent_invocation_started_event(run_id, seq, **fields)
                elif kind == "agent_invocation_completed":
                    evt = agent_invocation_completed_event(run_id, seq, **fields)
                elif kind == "delegation_observed":
                    evt = delegation_observed_event(run_id, seq, **fields)
                else:
                    return
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug("_emit_observability: failed to emit %s: %s", kind, exc)

        async def _emit_invocation_cancelled(
            self,
            *,
            invocation_id: str,
            agent_name: str = "",
            request: Any = None,
            tool_name: str = "",
        ) -> None:
            """Emit an ``InvocationCancelled`` sink event (goldfive#251).

            Operator-visible only — does NOT propagate to the LLM
            (that's what the minimal ``{"status": "cancelled"}`` tool
            response is for). Rich context: the invocation id, agent
            name, triggering reason / drift kind / severity / drift
            id, and an optional tool name when the cancel fired at a
            tool-dispatch checkpoint.

            Uses the typed
            :func:`goldfive.events.invocation_cancelled_event` factory
            (proto path). The dict-envelope path that Stream C (PR
            #259) shipped to dodge a proto regen has been removed —
            consumers that haven't picked up the typed message must
            update. The harmonograf-side switch off its placeholder
            ``harmonograf.v1.InvocationCancelled`` lands as a separate
            submodule-bump PR (Wave 2 / A8).

            Best-effort: every failure is logged and swallowed —
            observability must never block a callback.
            """
            ctx = self._active_ctx
            if ctx is None:
                return
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            # Extract fields from the CancellationRequest dataclass if
            # provided. Duck-typed so a plain dict or an unfamiliar
            # shape still rounds-trips without raising.
            reason = ""
            severity = ""
            drift_id = ""
            drift_kind = ""
            detail = ""
            if request is not None:
                reason = str(_safe_attr(request, "reason", "") or "")
                sev_val = _safe_attr(request, "severity", None)
                severity = str(getattr(sev_val, "value", sev_val) or "")
                drift_id = str(_safe_attr(request, "drift_id", "") or "")
                drift_kind = str(_safe_attr(request, "drift_kind", "") or "")
                detail = str(_safe_attr(request, "detail", "") or "")
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    emit,
                    invocation_cancelled_event,
                )

                evt = invocation_cancelled_event(
                    run_id,
                    seq,
                    invocation_id=str(invocation_id or ""),
                    agent_name=str(agent_name or ""),
                    reason=reason,
                    severity=severity,
                    drift_id=drift_id,
                    drift_kind=drift_kind,
                    detail=detail,
                    tool_name=str(tool_name or ""),
                    session_id=session_id,
                )
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_invocation_cancelled: failed to emit: %s",
                    exc,
                )

        # --- Goldfive boundary wrapper emits (goldfive#271 Phase 3.5) ---

        async def _emit_boundary_entered(
            self,
            *,
            invocation_id: str,
            agent_name: str = "",
            task_id: str = "",
        ) -> None:
            """Emit ``InvocationBoundaryEntered`` and pin the entry.

            The pin (``_boundary_entered_invocations``) is what guarantees
            the paired ``Exited`` fires exactly once even when ADK skips
            ``after_agent_callback`` on cancel — the canonical
            ``except CancelledError`` in
            :meth:`ADKAdapter._invoke_internal` consults the set to
            decide which boundaries still need an exit emit.
            """
            if not invocation_id:
                return
            if invocation_id in self._boundary_entered_invocations:
                # Idempotent: a second ``before_agent_callback`` for the
                # same invocation_id (transfer-to-agent inside one
                # invocation) does not re-emit the boundary.
                return
            self._boundary_entered_invocations.add(invocation_id)
            ctx = self._active_ctx
            if ctx is None:
                return
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    emit,
                    invocation_boundary_entered_event,
                )

                evt = invocation_boundary_entered_event(
                    run_id,
                    seq,
                    invocation_id=str(invocation_id or ""),
                    agent_name=str(agent_name or ""),
                    task_id=str(task_id or ""),
                    session_id=session_id,
                )
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_boundary_entered: failed to emit: %s",
                    exc,
                )

        async def _emit_boundary_exited(
            self,
            *,
            invocation_id: str,
            agent_name: str = "",
            task_id: str = "",
            reason: str = "completed",
        ) -> None:
            """Emit ``InvocationBoundaryExited`` paired with a prior Entered.

            No-op when the invocation_id was never marked entered (or was
            already exited). Exit-once semantic mirrors entry-once: ADK
            can fire ``after_agent_callback`` on the same invocation
            multiple times in degenerate flows, but the boundary pair is
            recorded once.

            Best-effort observability — every failure is logged and
            swallowed.
            """
            if not invocation_id:
                return
            if invocation_id not in self._boundary_entered_invocations:
                return
            self._boundary_entered_invocations.discard(invocation_id)
            ctx = self._active_ctx
            if ctx is None:
                return
            steerer = ctx.steerer
            if steerer is None:
                return
            sinks = getattr(steerer, "_sinks", None) or []
            if not sinks:
                return
            session = ctx.session
            run_id = str(_safe_attr(session, "run_id", "") or "")
            session_id = str(_safe_attr(session, "id", "") or "") or run_id
            try:
                seq = session.next_sequence()
            except Exception:  # noqa: BLE001
                seq = 0
            try:
                from goldfive.events import (  # noqa: PLC0415 — lazy
                    emit,
                    invocation_boundary_exited_event,
                )

                evt = invocation_boundary_exited_event(
                    run_id,
                    seq,
                    invocation_id=str(invocation_id or ""),
                    agent_name=str(agent_name or ""),
                    task_id=str(task_id or ""),
                    reason=str(reason or "completed"),
                    session_id=session_id,
                )
                await emit(sinks, evt)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_emit_boundary_exited: failed to emit: %s",
                    exc,
                )

        async def close_open_boundaries(self, *, reason: str) -> None:
            """Emit ``InvocationBoundaryExited`` for every still-open boundary.

            Called from the canonical ``except CancelledError`` /
            ``except Exception`` site in
            :meth:`ADKAdapter._invoke_internal` so a CancelledError
            tearing through ADK's machinery (which skips
            ``after_agent_callback``) still produces the paired exit
            emit. Iterates a snapshot of the entered set so the
            individual ``_emit_boundary_exited`` calls (which discard
            from the set) don't mutate during iteration.
            """
            if not self._boundary_entered_invocations:
                return
            ctx = self._active_ctx
            agent_name = ""
            task_id = ""
            if ctx is not None:
                agent_name = str(getattr(ctx, "host_agent_name", "") or "")
                task = getattr(ctx, "task", None)
                if task is not None:
                    task_id = str(_safe_attr(task, "id", "") or "")
            for inv_id in list(self._boundary_entered_invocations):
                await self._emit_boundary_exited(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    task_id=task_id,
                    reason=reason,
                )

        # --- Per-LLM-call wall-clock watcher (goldfive#271 follow-up) ---

        async def _run_llm_call_timeout_watcher(
            self,
            *,
            invocation_id: str,
            timeout_s: float,
            ctx: SessionContext,
        ) -> None:
            """Sleep for ``timeout_s`` then emit a ``LLM_CALL_TIMEOUT`` drift.

            Spawned by :meth:`before_model_callback` and cancelled from
            the paired :meth:`after_model_callback` when the LLM call
            completes within budget. If the watcher wakes up first the
            in-flight LLM call has exceeded the wall-clock budget — we
            do NOT terminate the call mid-stream (ADK doesn't expose a
            hook for that), but we:

            * Emit a CRITICAL ``LLM_CALL_TIMEOUT`` drift to the steerer
              so the configured policy / sinks see the event.
            * Flag the invocation for cooperative cancel via
              :meth:`request_invocation_cancel` so subsequent callbacks
              short-circuit. The current LLM call still completes, but
              the invocation as a whole stops at the next checkpoint.

            CancelledError propagates back to the caller (the
            after_model_callback path that cancels us) so the watcher
            exits cleanly when the LLM call ends in time.
            """
            try:
                await asyncio.sleep(timeout_s)
            except asyncio.CancelledError:
                return
            # Wall-clock budget exceeded.
            steerer = ctx.steerer if ctx is not None else None
            session = ctx.session if ctx is not None else None
            log.warning(
                "goldfive.llm.timeout invocation_id=%s timeout_s=%.1f agent=%s task_id=%s",
                invocation_id,
                timeout_s,
                self._host_agent_name or "?",
                str(_safe_attr(getattr(ctx, "task", None), "id", "") or "") or "?",
            )
            try:
                from goldfive.types import (  # noqa: PLC0415 — lazy
                    CancellationRequest,
                    DriftEvent,
                    DriftKind,
                    DriftSeverity,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_run_llm_call_timeout_watcher: cannot import types: %s",
                    exc,
                )
                return
            # Emit a CRITICAL drift so the steerer's policy fires. Best-
            # effort: any failure must not block the cancel-flag write.
            if steerer is not None and session is not None:
                try:
                    drift = DriftEvent(
                        kind=DriftKind.LLM_CALL_TIMEOUT,
                        severity=DriftSeverity.CRITICAL,
                        detail=(
                            f"LLM call exceeded wall-clock budget "
                            f"({timeout_s:.1f}s) on invocation "
                            f"{invocation_id}"
                        ),
                        current_task_id=str(_safe_attr(getattr(ctx, "task", None), "id", "") or ""),
                        current_agent_id=self._host_agent_name or "",
                    )
                    observation = _as_observation(
                        kind="llm_call_timeout",
                        detail=drift.detail,
                        raw={"invocation_id": invocation_id, "timeout_s": timeout_s},
                        task=getattr(ctx, "task", None),
                        agent_id=self._host_agent_name,
                    )
                    await steerer.observe(observation, session)
                    # Also surface as a structured DriftDetected so
                    # sinks see the drift even if the steerer's
                    # observe() routes elsewhere.
                    emit_drift = getattr(steerer, "_emit_drift_detected", None)
                    if emit_drift is not None:
                        try:
                            await emit_drift(session, drift)
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "_run_llm_call_timeout_watcher: _emit_drift_detected raised: %s",
                                exc,
                            )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "_run_llm_call_timeout_watcher: drift emission raised: %s",
                        exc,
                    )
            # Flag the invocation for cooperative cancel so the next
            # callback (whether after_model fires first, or the next
            # before_tool / before_model on a follow-up) short-circuits.
            try:
                request = CancellationRequest(
                    invocation_id=invocation_id,
                    reason="llm_call_timeout",
                    severity=DriftSeverity.CRITICAL,
                    drift_kind=DriftKind.LLM_CALL_TIMEOUT.value,
                    detail=(f"LLM call exceeded wall-clock budget ({timeout_s:.1f}s)"),
                    requested_at_ms=int(time.time() * 1000),
                )
                self.request_invocation_cancel(
                    invocation_id=invocation_id,
                    request=request,
                    propagate_to_children=True,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_run_llm_call_timeout_watcher: cancel-flag write raised: %s",
                    exc,
                )

        # --- Plan + current-task context -------------------------------

        async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> Any:
            """GoldfivePlanner request-side instruction injection
            + per-LLM-call instrumentation (goldfive#172).

            Performs GoldfivePlanner request-side instruction injection
            (goldfive#153): ADK's ``_nl_planning.py`` request-side gate
            fires only for ``PlanReActPlanner`` subclasses, not every
            ``BasePlanner``. Since goldfive deliberately subclasses
            ``BasePlanner`` directly (we don't want PlanReAct's response
            filtering), we invoke
            :meth:`GoldfivePlanner.build_planning_instruction` ourselves
            here and append the returned string to
            ``llm_request.config.system_instruction`` via the same
            ``append_instructions`` helper ADK uses internally.

            Phase 2.0 of goldfive#271 — V5 (the defensive duplicate
            seed of ``goldfive.*`` keys onto ADK ``session.state``)
            deleted. The planner now reads goldfive Session directly
            via the SessionContext stash; nothing on the ADK side
            needs the seed.

            Finally, it stamps per-LLM-call metrics (goldfive#172):
            counts ``llm_request.contents`` chars and message count,
            stashes a start-time on the plugin keyed by invocation_id
            so the paired ``after_model_callback`` can compute call
            duration, and logs the measurements at INFO with
            structured fields so operators running the live kikuchi
            endpoint can correlate post-steer slowdowns to context
            growth (see issue #172, hypothesis 1).
            """
            ctx = self._resolve_ctx(callback_context)
            if ctx is None:
                return None

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a;
            # sticky-flag fix from goldfive#271 follow-up). Skip the
            # LLM call when this invocation is flagged for cancel. This
            # is the checkpoint that matters most in practice: a
            # mid-flight LLM call is the expensive work whose output
            # would contaminate the parent transcript.
            #
            # Returns a synthetic ``LlmResponse`` (NOT ``None``) so ADK
            # actually short-circuits the dispatch — per ADK's
            # ``BasePlugin.before_model_callback`` contract, returning
            # ``None`` lets the LLM request proceed normally.
            #
            # Uses the sticky :meth:`is_invocation_cancelled` so every
            # subsequent callback on the same invocation also
            # short-circuits, even though the cancel-state entry is
            # popped on first consume (consume-once semantic for the
            # InvocationCancelled sink event).
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id_check = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if inv_id_check and self.is_invocation_cancelled(inv_id_check):
                pending = self._cancel_state.get(inv_id_check)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id_check)
                    self._cancelled_invocations.add(inv_id_check)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id_check,
                        agent_name="",
                        request=request,
                    )
                else:
                    log.info(
                        "goldfive.llm.skip invocation_id=%s reason=cancel-flag-set",
                        inv_id_check,
                    )
                return _make_cancelled_llm_response()

            # GoldfivePlanner request-side injection (goldfive#153).
            # Best-effort: never raise from this path; injection failure
            # degrades to "LLM runs without goldfive's orchestration
            # context block" which is safe — the planner reads off
            # goldfive Session directly so no callback-time write
            # is required.
            try:
                await _inject_goldfive_planner_instruction(
                    callback_context=callback_context,
                    llm_request=llm_request,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_model_callback: goldfive planner injection raised: %s",
                    exc,
                )

            # R3 (F2 alternative) — runtime tool-surface hint.
            #
            # Tier 1's F1/F3/F4 fire AFTER the LLM has already emitted /
            # received a tool call. They can't prevent the model's NEXT
            # inference loop from picking the same already-completed
            # agent. The hint here is the PRE-EMPTIVE signal: at every
            # model call we tell the LLM which agents still have
            # PENDING work and which are DONE, so the model has
            # structural guidance about what to call next without
            # needing user-prompt cooperation.
            #
            # Best-effort, like the planner injection above. The hint
            # is bracketed with marker tags so subsequent calls can
            # strip the stale block before appending the fresh one
            # (per-call, not accumulated).
            try:
                if ctx is not None and ctx.session is not None:
                    _inject_runtime_tools_hint(
                        callback_context=callback_context,
                        llm_request=llm_request,
                        session=ctx.session,
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_model_callback: runtime tools hint injection raised: %s",
                    exc,
                )

            # Per-LLM-call instrumentation (goldfive#172). Measure the
            # request AFTER GoldfivePlanner has appended its
            # instruction so the reported chars reflect what the
            # model actually sees. Stash a start-time so the paired
            # after_model_callback can compute duration.
            try:
                chars, messages_count = _measure_request_chars(llm_request)
                inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                    callback_context, "invocation_context", None
                )
                inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
                start_mono = time.monotonic()
                if inv_id:
                    pending: dict[str, Any] = {
                        "start_mono": start_mono,
                        "chars": chars,
                        "messages_count": messages_count,
                    }
                    # Per-LLM-call wall-clock watcher (goldfive#271
                    # follow-up). Spawned as an asyncio task so it
                    # runs concurrently with the in-flight
                    # ``generate_content_async`` stream. The paired
                    # ``after_model_callback`` cancels the watcher
                    # when the LLM call returns within budget. Skip
                    # entirely when the budget is non-positive
                    # (operator opted out) or when no SessionContext
                    # is available (the watcher can't emit drifts
                    # without one).
                    if (
                        self._llm_call_timeout_ms > 0
                        and ctx is not None
                        and not self.is_invocation_cancelled(inv_id)
                    ):
                        timeout_s = self._llm_call_timeout_ms / 1000.0
                        try:
                            watcher = asyncio.create_task(
                                self._run_llm_call_timeout_watcher(
                                    invocation_id=inv_id,
                                    timeout_s=timeout_s,
                                    ctx=ctx,
                                ),
                                name=f"goldfive_llm_watcher_{inv_id}",
                            )
                            pending["watcher"] = watcher
                        except RuntimeError as exc:
                            # No running loop (extremely unusual in an
                            # async callback, but the harness kicks
                            # off with no loop in some unit tests).
                            log.debug(
                                "before_model_callback: cannot schedule LLM-timeout watcher: %s",
                                exc,
                            )
                    self._invocation_llm_pending[inv_id] = pending
                # INFO log so an operator running a live e2e (kikuchi)
                # can tail stderr and correlate context growth against
                # the subsequent duration line.
                log.info(
                    "goldfive.llm.request invocation_id=%s "
                    "llm.request.chars=%d llm.request.messages_count=%d "
                    "task_id=%s agent=%s",
                    inv_id or "?",
                    chars,
                    messages_count,
                    str(_safe_attr(ctx.task, "id", "") or "") or "?",
                    self._host_agent_name or "?",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "before_model_callback: LLM-call instrumentation raised: %s",
                    exc,
                )
            return None

        # --- Reporting-tool interception + tool-confirmation bridge ---

        async def before_tool_callback(
            self, *, tool: Any, tool_args: Any, tool_context: Any
        ) -> dict[str, Any] | None:
            ctx = self._resolve_ctx(tool_context)
            if ctx is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            if not tool_name:
                func = _safe_attr(tool, "func", None)
                tool_name = str(_safe_attr(func, "__name__", "") or "")

            # Cooperative-cancellation checkpoint (goldfive#251 Stream C / 7a).
            # When this invocation was flagged for cancel (either the
            # steerer at CRITICAL severity or a user-initiated cancel),
            # skip *AgentTool* dispatch and return a MINIMAL LLM-visible
            # tool response: ``{"status": "cancelled"}``. The minimal
            # shape is deliberate — richer shapes (``reason``, ``detail``,
            # ``drift_kind``) become prompt-injection vectors (see
            # lessons from goldfive#250 / #252 / #253 where LLMs
            # pattern-matched on error strings and invented workarounds).
            # Rich context for operators lives on the
            # InvocationCancelled sink event emitted by
            # :meth:`_emit_invocation_cancelled`.
            #
            # FunctionTool dispatches (write_webpage, patch_file, and any
            # user-provided side-effect helpers) are NOT short-circuited
            # here — see :func:`_is_agent_tool_dispatch` and Bug C from
            # v23 validation. Their work has already been committed by
            # the model (args chosen, function_call event emitted) and
            # discarding the actual call loses the side-effect (no file
            # written, no row patched) for no semantic benefit: the very
            # next ``before_model_callback`` on this same invocation
            # short-circuits the LLM call regardless, ending the
            # dispatch cleanly. Letting the FunctionTool run preserves
            # committed work; short-circuiting it would silently strand
            # it on every supersede-cancel.
            inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
                tool_context, "invocation_context", None
            )
            inv_id_check = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if (
                inv_id_check
                and self.is_invocation_cancelled(inv_id_check)
                and _is_agent_tool_dispatch(tool)
            ):
                pending = self._cancel_state.get(inv_id_check)
                if pending is not None:
                    request = self.consume_cancel_for_invocation(inv_id_check)
                    self._cancelled_invocations.add(inv_id_check)
                    await self._emit_invocation_cancelled(
                        invocation_id=inv_id_check,
                        agent_name="",
                        request=request,
                        tool_name=tool_name,
                    )
                # MINIMAL LLM-visible response — single-key dict, no
                # ``reason`` / ``detail`` / ``drift_kind``. The parent
                # LLM that receives this as an AgentTool response can
                # pattern-match only on the word "cancelled" and
                # should defer to the plan-revised context it sees on
                # its next turn to decide whether to re-dispatch.
                return {"status": "cancelled"}

            # Reporting-tool short-circuit takes precedence: a tool named
            # e.g. report_task_started should never also be gated by
            # confirmation — the protocol handlers are control-plane
            # calls, not side-effects.
            #
            # Route through ``invoke_tool`` (NOT a direct handler call)
            # so every reporting-tool dispatch picks up schema validation
            # (missing / unknown ``task_id`` → structured error) and
            # then reaches the reporting handlers, which own the rest of
            # the protection stack:
            #
            #   * idempotency — same-transition retries return
            #     ``{"acknowledged": True, "idempotent": True, ...}``
            #     (goldfive#201, #203),
            #   * invalid-transition — cross-transitions on terminal
            #     tasks return ``{"acknowledged": False,
            #     "error": "invalid_transition", ...}``,
            #   * tool-loop detection — covered independently by
            #     :class:`goldfive.drift.tool_loops.ToolLoopTracker`
            #     at ``after_tool_callback`` (goldfive#181, #204).
            #
            # See ``docs/design/TASK-LIFECYCLE.md`` §5 for the contract.
            #
            # goldfive#241 — task_id is hidden from the LLM-facing
            # reporting-tool schema so the model never supplies it.
            # Resolve from state (delegation-site pin first, then the
            # agent-turn pin). If neither resolves, the response branch
            # depends on whether the current agent actually has work in
            # the plan (goldfive#250 follow-up):
            #
            # * **No PENDING/RUNNING candidates for this agent** — a
            #   legit orchestration-only turn (e.g. coordinator whose
            #   tasks were superseded by a plan refine into tasks
            #   assigned to other agents). Return a bare silent
            #   acknowledgment so the agent's reporting protocol does
            #   NOT crash on plan-revision boundaries. A loud error
            #   here makes the LLM bypass the reporting protocol —
            #   observed live.
            # * **Has candidates** — the pin SHOULD have worked; a
            #   silent ack could mask a real stall. To keep the stall
            #   visible WITHOUT leaking a prompt-injection surface to
            #   the LLM (observed live: research_agent read an
            #   ``error: pin_unresolved`` payload and reasoned "This
            #   might be related to the plan/task system. Let me try
            #   a different approach — I'll just compile the research
            #   and create the presentation content directly", bypassing
            #   the reporting contract), the tool response is the same
            #   bare ``{"acknowledged": True}``. Operator visibility is
            #   preserved via a WARNING log AND a ``DriftDetected`` sink
            #   event (``DriftKind.OFF_TOPIC`` with a ``pin_unresolved:``
            #   reason prefix). See goldfive#252 follow-up + PR notes.
            #
            # The silent-ack response carries NO ``detail`` / ``error`` /
            # ``no_task_pinned`` / ``pin_unresolved`` keys — tool responses
            # go back to the LLM verbatim and any editorialising string
            # (or error-shaped payload) is treated as actionable context
            # (observed live: research_agent paraphrased a detail string
            # into its reasoning and proceeded with stale pre-refine
            # instructions, ignoring the refined scope).
            pinned = _inject_task_id_from_state(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_context=tool_context,
            )
            if _is_reporting_tool_name(tool_name) and not pinned:
                # Resolve the current agent name — prefer the live
                # invocation's agent (tool_context._invocation_context.
                # agent.name), fall back to the host agent from
                # SessionContext. Any resolution failure degrades to
                # the silent-ack path (conservative — avoid breaking
                # runs on edge-cases).
                agent_name = ""
                try:
                    inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
                        tool_context, "invocation_context", None
                    )
                    running_agent = _safe_attr(inv_ctx, "agent", None)
                    agent_name = str(_safe_attr(running_agent, "name", "") or "")
                    if not agent_name:
                        agent_name = str(_safe_attr(ctx, "host_agent_name", "") or "")
                except Exception:  # noqa: BLE001
                    agent_name = ""

                has_candidates = False
                try:
                    has_candidates = _agent_has_pending_candidates(ctx, agent_name)
                except Exception:  # noqa: BLE001 — conservative fall-through
                    has_candidates = False

                if has_candidates:
                    # Gather candidate ids for the diagnostic WARNING +
                    # drift event (nice-to-have; swallow any resolution
                    # errors).
                    candidate_ids: list[str] = []
                    try:
                        from goldfive.types import TaskStatus

                        plan = _safe_attr(ctx.session, "plan", None)
                        tasks = _safe_attr(plan, "tasks", None) or ()
                        for task in tasks:
                            assignee = str(_safe_attr(task, "assignee_agent_id", "") or "")
                            if assignee != agent_name:
                                continue
                            status = _safe_attr(task, "status", None)
                            if status is TaskStatus.PENDING or status is TaskStatus.RUNNING:
                                candidate_ids.append(str(_safe_attr(task, "id", "") or ""))
                    except Exception:  # noqa: BLE001
                        pass
                    log.warning(
                        "before_tool_callback: pin_unresolved for %s "
                        "(agent=%s, candidates=[%s]); emitting "
                        "DriftDetected(pin_unresolved) and returning silent "
                        "ack so the LLM cannot pattern-match on the error",
                        tool_name,
                        agent_name or "?",
                        ", ".join(candidate_ids),
                    )
                    # Surface the stall to operators via a sink event so
                    # it's visible in harmonograf without being visible
                    # to the LLM. Reuse OFF_TOPIC with a reason prefix
                    # rather than adding a new proto DriftKind (heavier
                    # change; the invariant here is operator observability,
                    # not a new wire-level classification).
                    try:
                        await self._emit_pin_unresolved_drift(
                            ctx=ctx,
                            agent_name=agent_name,
                            tool_name=tool_name,
                            candidate_ids=candidate_ids,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "before_tool_callback: pin_unresolved drift emit raised: %s",
                            exc,
                        )
                    return {"acknowledged": True}

                log.info(
                    "before_tool_callback: no task pinned for %s; "
                    "returning no-op acknowledgment (orchestration-only turn)",
                    tool_name,
                )
                return {"acknowledged": True}

            tool_names_registered = {spec.name for spec in ctx.tools}
            if tool_name in tool_names_registered:
                args_map: dict[str, Any]
                if isinstance(tool_args, Mapping):
                    args_map = dict(tool_args)
                else:
                    args_map = {}
                try:
                    result = await invoke_tool(
                        ctx.tools,
                        tool_name,
                        args_map,
                        ctx.session,
                        ctx.steerer,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "before_tool_callback: handler for %s raised: %s",
                        tool_name,
                        exc,
                    )
                    # Fall back to the canonical acknowledgment so the
                    # agent doesn't see a tool error for a protocol call.
                    return {"acknowledged": True, "error": str(exc)}
                # Return a non-None result to short-circuit ADK tool dispatch.
                # ``invoke_tool`` always returns a dict; preserve it verbatim
                # so the terminal-rejection / duplicate-ACK payloads reach
                # the agent unchanged.
                if isinstance(result, dict):
                    return result
                return {"acknowledged": True}

            # AgentTool detection. We look both at the tool's class
            # name (avoids an unconditional ADK import on module load)
            # and its ``agent`` attribute so we catch wrapper / subclass
            # shapes. Two jobs here:
            #   1. Emit DelegationObserved (observability).
            #   2. Count toward the per-invocation AgentTool cap
            #      (goldfive#130 runaway-delegation backstop).
            nested_agent = _safe_attr(tool, "agent", None)
            tool_cls_name = type(tool).__name__
            if nested_agent is not None or tool_cls_name == "AgentTool":
                to_agent = str(_safe_attr(nested_agent, "name", "") or "") or tool_name
                from_agent = str(_safe_attr(ctx, "host_agent_name", "") or "")
                task_id = str(_safe_attr(ctx.task, "id", "") or "")
                inv_id = ""
                inv_ctx = _safe_attr(tool_context, "_invocation_context", None)
                if inv_ctx is not None:
                    inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
                await self._emit_observability(
                    "delegation_observed",
                    from_agent=from_agent,
                    to_agent=to_agent,
                    task_id=task_id,
                    invocation_id=inv_id,
                )
                # Extend the per-task observed-agent lineage with the
                # delegated child. Idempotent ``set.add`` so repeat
                # delegations to the same child do not balloon the set.
                # No-op when there is no current task pin (race with
                # task in_progress) or no delegated agent name — those
                # are best-effort observability signals, not invariants.
                try:
                    gf_session = ctx.session if ctx is not None else None
                    pinned_task_id = (
                        gf_session.current_task_id if gf_session is not None else ""
                    )
                    if (
                        gf_session is not None
                        and pinned_task_id
                        and to_agent
                        and pinned_task_id in gf_session.task_lineage
                    ):
                        gf_session.task_lineage[pinned_task_id].add(to_agent)
                except Exception as exc:  # noqa: BLE001 — observability is best-effort
                    log.debug(
                        "before_tool_callback: task_lineage update raised: %s",
                        exc,
                    )
                if self._reconciler is not None:
                    try:
                        await self._reconciler.on_delegation_observed(
                            from_agent=from_agent,
                            to_agent=to_agent,
                            invocation_id=inv_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "before_tool_callback: reconciler.on_delegation_observed raised: %s",
                            exc,
                        )

                # goldfive#241 Item 3-bis — delegation-site task_id
                # pinning. When the coordinator fires multiple parallel
                # AgentTool calls to the same sub-agent in one turn,
                # each dispatch spawns its own sub-invocation; the
                # sub-agent's ``before_agent_callback`` cannot
                # disambiguate because all N parallel calls share the
                # same (agent_name, session.state) pair. We resolve
                # the candidate task for THIS AgentTool dispatch and
                # stash it on ``pending_delegations[<function_call_id>]``
                # so the reporting-tool callback (which DOES see the
                # function_call_id) can read the correct pin back.
                try:
                    self._pin_delegation_task_id(
                        ctx=ctx,
                        tool_context=tool_context,
                        to_agent=to_agent,
                        tool_args=tool_args,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    log.debug(
                        "before_tool_callback: delegation pin raised: %s",
                        exc,
                    )

                # Runaway-delegation cap. Count BEFORE short-circuiting
                # so the drift fires exactly once at the threshold
                # crossing; subsequent AgentTool calls in the same
                # invocation return a short-circuit skipped dict so
                # the runner wraps up quickly.
                if self._agent_tool_cap > 0:
                    self._agent_tool_spawn_count += 1
                    if (
                        self._agent_tool_spawn_count > self._agent_tool_cap
                        and not self.runaway_delegation_tripped
                    ):
                        self.runaway_delegation_tripped = True
                        await self._emit_runaway_delegation_drift(
                            ctx=ctx,
                            from_agent=from_agent,
                            to_agent=to_agent,
                            task_id=task_id,
                            invocation_id=inv_id,
                            spawn_count=self._agent_tool_spawn_count,
                        )
                    if self.runaway_delegation_tripped:
                        # Short-circuit the spawn: return a skipped dict
                        # so ADK does not drive the sub-agent. The
                        # adapter's invoke loop notices the tripped
                        # flag between events and breaks out.
                        return {
                            "skipped": True,
                            "reason": "goldfive_runaway_delegation_cap",
                            "tool_name": tool_name,
                            "detail": (
                                f"AgentTool-per-invoke cap of "
                                f"{self._agent_tool_cap} exceeded "
                                f"(spawn #{self._agent_tool_spawn_count})"
                            ),
                        }

                # Tier 1 / F3 (loop prevention): pre-dispatch interception.
                # When every plan task assigned to this AgentTool's
                # target agent is terminal AND there is a non-terminal
                # next_pending task assigned to a *different* agent,
                # refuse the dispatch with a redirect error so the
                # coordinator stops re-invoking a completed-work agent.
                # The post-hoc PLAN_DIVERGENCE detector still exists as
                # a safety net, but closing the loop at the dispatch
                # point eliminates the round-trip-and-detect cost.
                #
                # We deliberately use a tool-error response shape rather
                # than a richer structured payload — the LLM only needs
                # to read "go to other agent" and the adapter's drift
                # surface owns the operator-side observability.
                redirect = _maybe_redirect_completed_agent(
                    ctx=ctx,
                    target_agent=to_agent,
                )
                if redirect is not None:
                    log.info(
                        "before_tool_callback: F3 redirect — all plan tasks "
                        "for %s are terminal; redirecting coordinator to %s",
                        to_agent,
                        redirect.get("redirect_to") or "?",
                    )
                    return redirect
                # Fall through: AgentTool still runs, we're just observing.

            # Tool-level approval (Flow B). If the tool opts into
            # confirmation via ADK's native `require_confirmation` flag,
            # bridge the gate onto goldfive's control channel.
            if _tool_requires_confirmation(tool, tool_args):
                return await _await_tool_approval(
                    tool=tool,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_context=tool_context,
                    session_ctx=ctx,
                )
            return None

        # --- Drift observation -----------------------------------------

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
            ctx = self._resolve_ctx(callback_context)
            if ctx is None or ctx.steerer is None:
                return None
            texts = _extract_text_parts(llm_response)
            calls = _extract_function_calls(llm_response)
            reasoning = _extract_reasoning(llm_response)
            finish = _safe_attr(llm_response, "finish_reason", None)
            # Feed the per-invocation counters used by the
            # CONFABULATION_RISK check in after_run_callback. We track:
            #   * tool-call count: cumulative across the invocation's
            #     LLM turns so we only fire if NO tool was ever used,
            #   * last non-empty text: used as the output signal — an
            #     invocation that ended with empty text is not
            #     suspicious regardless of tool calls.
            inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
                callback_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            if inv_id:
                if calls:
                    self._invocation_tool_calls[inv_id] = self._invocation_tool_calls.get(
                        inv_id, 0
                    ) + len(calls)
                if texts:
                    joined = " ".join(texts).strip()
                    if joined:
                        self._invocation_last_text[inv_id] = joined
            # Per-LLM-call instrumentation (goldfive#172). Pair with the
            # before_model_callback stash to compute duration, extract
            # token usage, log the result, and enrich the observation
            # raw dict so custom steerer sinks can surface the metrics
            # alongside each LLM turn. Any failure in this block is
            # swallowed — instrumentation must not shadow a real LLM
            # response from the steerer.
            metrics: dict[str, Any] = {}
            try:
                pending = self._invocation_llm_pending.pop(inv_id, None) if inv_id else None
                if pending is not None:
                    duration_ms = int((time.monotonic() - pending["start_mono"]) * 1000)
                    metrics["llm.call.duration_ms"] = duration_ms
                    metrics["llm.request.chars"] = int(pending.get("chars", 0))
                    metrics["llm.request.messages_count"] = int(pending.get("messages_count", 0))
                    # Cancel the per-LLM-call wall-clock watcher
                    # (goldfive#271 follow-up). The LLM call returned
                    # within budget, so the watcher's pending sleep is
                    # no longer needed. Tolerate the watcher having
                    # already fired (race: it can complete just before
                    # we reach this line if the LLM call landed
                    # exactly at the budget boundary).
                    watcher = pending.get("watcher")
                    if watcher is not None and not watcher.done():
                        watcher.cancel()
                usage = _extract_usage_metadata(llm_response)
                for key, value in usage.items():
                    metrics[f"llm.usage.{key}"] = value
                if metrics:
                    log.info(
                        "goldfive.llm.response invocation_id=%s "
                        "llm.call.duration_ms=%s llm.request.chars=%s "
                        "llm.request.messages_count=%s "
                        "llm.usage.prompt_tokens=%s "
                        "llm.usage.completion_tokens=%s "
                        "llm.usage.total_tokens=%s "
                        "task_id=%s agent=%s",
                        inv_id or "?",
                        metrics.get("llm.call.duration_ms", "?"),
                        metrics.get("llm.request.chars", "?"),
                        metrics.get("llm.request.messages_count", "?"),
                        metrics.get("llm.usage.prompt_tokens", "?"),
                        metrics.get("llm.usage.completion_tokens", "?"),
                        metrics.get("llm.usage.total_tokens", "?"),
                        str(_safe_attr(ctx.task, "id", "") or "") or "?",
                        self._host_agent_name or "?",
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "after_model_callback: LLM-call instrumentation raised: %s",
                    exc,
                )
            raw: dict[str, Any] = {
                "texts": texts,
                "function_calls": calls,
                "reasoning": reasoning,
                "finish_reason": str(finish) if finish is not None else "",
            }
            if metrics:
                raw["metrics"] = metrics
            observation = _as_observation(
                kind="llm_response",
                detail=" ".join(texts)[:500],
                raw=raw,
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("after_model_callback: steerer.observe raised: %s", exc)
            if reasoning:
                # Cooperative-cancel gate (iter-11D, goldfive#251 follow-up).
                # Skip reasoning observation for cancelled invocations:
                # the agent has been told to stop, so judging its
                # still-streaming or already-buffered reasoning
                # produces noise (CONFUSION drifts on zombie reasoning,
                # wasted LLM judge calls). The plugin's sticky cancel
                # flag is already battle-tested by 6+ other callbacks
                # (before_model, before_tool, before/after_agent,
                # before_run, on_event) so reusing it here keeps the
                # behaviour aligned with the rest of the cancel-aware
                # surface.
                if inv_id and self.is_invocation_cancelled(inv_id):
                    log.debug(
                        "after_model_callback: skipping observe_reasoning for "
                        "cancelled invocation inv_id=%s",
                        inv_id,
                    )
                else:
                    observe_reasoning = getattr(ctx.steerer, "observe_reasoning", None)
                    if observe_reasoning is not None:
                        # Resolve the live agent name so the steerer's
                        # per-(agent, task) reasoning-judge rate-limit
                        # bucket isolates agents (goldfive#252 follow-up).
                        # Prefer the invocation's running agent, fall back
                        # to the host agent so single-agent runs keep their
                        # historical bucketing.
                        reasoning_agent_name = ""
                        try:
                            running_agent = _safe_attr(inv_ctx, "agent", None)
                            reasoning_agent_name = str(_safe_attr(running_agent, "name", "") or "")
                        except Exception:  # noqa: BLE001
                            reasoning_agent_name = ""
                        if not reasoning_agent_name:
                            reasoning_agent_name = self._host_agent_name or ""
                        try:
                            await observe_reasoning(
                                reasoning,
                                task=ctx.task,
                                session=ctx.session,
                                provider=_infer_provider(llm_response),
                                agent_name=reasoning_agent_name,
                            )
                        except TypeError:
                            # Back-compat: custom steerer without the
                            # ``agent_name`` kwarg. Fall back silently.
                            try:
                                await observe_reasoning(
                                    reasoning,
                                    task=ctx.task,
                                    session=ctx.session,
                                    provider=_infer_provider(llm_response),
                                )
                            except Exception as exc:  # noqa: BLE001
                                log.debug(
                                    "after_model_callback: observe_reasoning (fallback) raised: %s",
                                    exc,
                                )
                        except Exception as exc:  # noqa: BLE001
                            log.debug(
                                "after_model_callback: observe_reasoning raised: %s",
                                exc,
                            )
            # Note this turn for the opt-in reflective self-progress check.
            # ``note_llm_call`` is a no-op unless the steerer was
            # constructed with ``reflective_call_llm``; adapters that
            # don't ship this hook simply skip the counter.
            #
            # Also gate on cancellation: ``note_llm_call`` is not pure
            # book-keeping — when its counter hits
            # ``reflective_check_interval`` it fires
            # ``maybe_run_reflective_check`` which makes a fresh LLM
            # call against the cancelled invocation's tool-call /
            # reasoning summary. That's the same "judging zombie work"
            # failure mode the reasoning-observation gate above
            # protects against, so it gets the same treatment. Skipping
            # the counter increment on cancelled turns is also correct:
            # those turns aren't meaningful work, so they shouldn't
            # consume the next reflective check's window.
            if inv_id and self.is_invocation_cancelled(inv_id):
                log.debug(
                    "after_model_callback: skipping note_llm_call for "
                    "cancelled invocation inv_id=%s",
                    inv_id,
                )
            else:
                note_llm_call = getattr(ctx.steerer, "note_llm_call", None)
                if note_llm_call is not None:
                    try:
                        await note_llm_call(ctx.session)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("after_model_callback: note_llm_call raised: %s", exc)
            return None

        async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
            ctx = self._resolve_ctx(invocation_context)
            if ctx is None or ctx.steerer is None:
                return None
            # Detect transfer / escalation actions on the event payload.
            actions = _safe_attr(event, "actions", None)
            transfer_to = _safe_attr(actions, "transfer_to_agent", "") or ""
            escalate = bool(_safe_attr(actions, "escalate", False))
            if not transfer_to and not escalate:
                return None
            kind = "agent_transfer" if transfer_to else "agent_escalation"
            detail = f"transfer -> {transfer_to}" if transfer_to else "escalate"
            observation = _as_observation(
                kind=kind,
                detail=detail,
                raw=event,
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("on_event_callback: steerer.observe raised: %s", exc)
            return None

        async def on_tool_error_callback(
            self,
            *,
            tool: Any,
            tool_args: Any,
            tool_context: Any,
            error: Any,
        ) -> None:
            ctx = self._resolve_ctx(tool_context)
            if ctx is None or ctx.steerer is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            observation = _as_observation(
                kind="tool_error",
                detail=f"{tool_name}: {error}",
                raw={"tool": tool_name, "error": repr(error)},
                task=ctx.task,
                agent_id=self._host_agent_name,
            )
            try:
                await ctx.steerer.observe(observation, ctx.session)
            except Exception as exc:  # noqa: BLE001
                log.debug("on_tool_error_callback: steerer.observe raised: %s", exc)
            # Iter-10 PR 2: also record the raised-error path on
            # ``session.recent_tool_observations`` so the three-state
            # reasoning judge (PR 3) can recognise a provoked
            # deviation rooted in a hard tool exception. The one-shot
            # ``Observation`` above is consumed for drift dispatch
            # (e.g. ``classify_tool_error``) but never persisted on
            # the session.
            try:
                note_obs = getattr(ctx.steerer, "note_tool_observation", None)
                if note_obs is not None:
                    gf_session = ctx.session
                    pinned_agent = (
                        str(_safe_attr(gf_session, "current_agent_id", "") or "")
                        or self._host_agent_name
                    )
                    pinned_task = (
                        str(_safe_attr(gf_session, "current_task_id", "") or "")
                        or str(_safe_attr(ctx.task, "id", "") or "")
                    )
                    note_obs(
                        gf_session,
                        agent_name=pinned_agent,
                        task_id=pinned_task,
                        tool_name=tool_name,
                        args=tool_args,
                        result=None,
                        error=error,
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "on_tool_error_callback: note_tool_observation raised: %s",
                    exc,
                )
            return None

        # --- Tool-loop drift detection (goldfive#181) ------------------

        async def after_tool_callback(
            self,
            *,
            tool: Any,
            tool_args: Any,
            tool_context: Any,
            result: Any,
        ) -> None:
            """Feed the tool-loop tracker and emit any drifts it raises.

            Runs after every tool ADK dispatched (reporting tools,
            AgentTool delegations, MCP/custom tools) so the detector
            sees the real function_call stream the agent is emitting.
            A reporting-tool progress call (``report_task_started`` /
            ``_progress`` / ``_completed`` / ``_failed`` / ``_blocked`` /
            ``report_awaiting_approval``) additionally clears the
            per-(invocation, agent) window — but ONLY when the call
            was acknowledged (``result == {"acknowledged": True, ...}``)
            so mode 2's "no task progress" gate is correct.

            The acknowledged-success gate (goldfive#192) is the
            tightening over goldfive#181's original behaviour: errored
            progress reports (``acknowledged=False`` or responses with
            an ``error`` key) do NOT reset the window, so an agent
            stuck retrying a failing ``report_task_*`` with a bad
            ``task_id`` gets caught as a tool-loop at the normal
            thresholds.

            The detector is deterministic and O(1) per call modulo the
            tracker's ``window`` length; any failure is swallowed so a
            buggy classifier never breaks tool dispatch. Drifts are
            routed through ``steerer._handle_drift`` when available so
            the intervention ladder sees them; falls back to
            ``steerer.observe`` for stubs that don't expose
            ``_handle_drift``.
            """
            ctx = self._resolve_ctx(tool_context)
            if ctx is None or ctx.steerer is None:
                return None
            tool_name = str(_safe_attr(tool, "name", "") or "")
            if not tool_name:
                func = _safe_attr(tool, "func", None)
                tool_name = str(_safe_attr(func, "__name__", "") or "")
            # Resolve invocation_id + agent_name so the tracker's
            # per-(invocation, agent) buckets match the reconciler's
            # isolation model. Missing fields fall back to "" so the
            # tracker still keys consistently on an ephemeral "unknown"
            # bucket (tests exercise this path).
            inv_ctx = _safe_attr(tool_context, "_invocation_context", None) or _safe_attr(
                tool_context, "invocation_context", None
            )
            inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
            running_agent = _safe_attr(inv_ctx, "agent", None)
            agent_name = str(_safe_attr(running_agent, "name", "") or "") or self._host_agent_name
            task_id = str(_safe_attr(ctx.task, "id", "") or "")

            # Every tool call is observed — regardless of kind. For a
            # progress-reporting tool (``report_task_*`` /
            # ``report_awaiting_approval``) we THEN look at the
            # ``result`` payload and reset the per-(invocation, agent)
            # window only when the call was *acknowledged* successfully
            # (goldfive#192). Previously the exemption triggered on the
            # call alone, so an agent stuck retrying a failing
            # ``report_task_started`` kept resetting the window and the
            # loop detector never fired. By observing first and resetting
            # only on acknowledged success, errored report_* calls
            # accumulate in the ring buffer and trigger loop detection
            # at the normal thresholds.
            try:
                # tool_args may be None / missing on adapter edge cases;
                # the tracker's hash helper copes with both.
                args_payload = tool_args if isinstance(tool_args, Mapping) else {}
                drifts = self._tool_loop_tracker.observe_tool_call(
                    invocation_id=inv_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    args=dict(args_payload),
                    task_id=task_id,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "after_tool_callback: tool-loop tracker raised: %s",
                    exc,
                )
                return None

            # Iter-10 PR 2: record the call in the session-scoped
            # ``recent_tool_observations`` ring buffer so the
            # three-state reasoning judge (PR 3) can distinguish a
            # provoked deviation (the agent saw a tool error or
            # surprising result and pivoted) from an unprovoked one.
            # We capture both successful calls and acknowledged
            # failures (``result`` is a dict with shape
            # ``{"error": ...}``); raised exceptions are captured by
            # ``on_tool_error_callback`` instead. The live agent /
            # task pin set by iter-9's ``before_agent_callback`` is
            # the authoritative source — fall back to the ADK-resolved
            # agent_name and the ctx-task id when those are empty.
            try:
                note_obs = getattr(ctx.steerer, "note_tool_observation", None)
                if note_obs is not None:
                    gf_session = ctx.session
                    pinned_agent = (
                        str(_safe_attr(gf_session, "current_agent_id", "") or "")
                        or agent_name
                        or self._host_agent_name
                    )
                    pinned_task = (
                        str(_safe_attr(gf_session, "current_task_id", "") or "") or task_id
                    )
                    note_obs(
                        gf_session,
                        agent_name=pinned_agent,
                        task_id=pinned_task,
                        tool_name=tool_name,
                        args=tool_args,
                        result=result,
                    )
            except Exception as exc:  # noqa: BLE001
                # Defensive — observability must never break tool
                # dispatch. The steerer method already swallows its
                # own internals; this catches the lookup path.
                log.debug(
                    "after_tool_callback: note_tool_observation raised: %s",
                    exc,
                )

            # Post-observation: reset the window only on acknowledged
            # success for progress-reporting tools. An errored report
            # (``acknowledged=False`` or a response containing an
            # ``error`` key) falls through to the regular drift
            # dispatch path so a stuck retry-loop still lights up.
            if tool_name in self._progress_reporting_tools:
                if _is_progress_report_success(result):
                    try:
                        self._tool_loop_tracker.on_task_progress(
                            invocation_id=inv_id,
                            agent_name=agent_name,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_tool_callback: on_task_progress raised: %s",
                            exc,
                        )

            if not drifts:
                return None

            # Prefer ``_handle_drift`` so the intervention ladder
            # sees the signal. Fall back to ``observe`` for stubs.
            handle = getattr(ctx.steerer, "_handle_drift", None)
            for drift in drifts:
                if handle is not None:
                    try:
                        await handle(drift, ctx.session)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "after_tool_callback: _handle_drift raised: %s",
                            exc,
                        )
                # Fallback path for steerer stubs that don't expose
                # _handle_drift.
                observation = _as_observation(
                    kind="tool_loop_detected",
                    detail=drift.detail,
                    raw=drift.raw if isinstance(drift.raw, dict) else {"drift": repr(drift.raw)},
                    task=ctx.task,
                    agent_id=agent_name or self._host_agent_name,
                )
                try:
                    await ctx.steerer.observe(observation, ctx.session)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "after_tool_callback: steerer.observe raised: %s",
                        exc,
                    )
            return None

    # goldfive#271 Phase 0 — wrap each callback in a state-audit
    # bookkeeping context so the runtime tripwire can recognise
    # "writes from inside a goldfive callback" and gate them against
    # the catalog. The wrapping is structural (one ContextVar set per
    # callback entry / exit) and runs unconditionally; the actual
    # check fires only when ``GOLDFIVE_STRICT_STATE_OWNERSHIP`` /
    # the test fixture flips it on. See
    # ``docs/design/STATE-OWNERSHIP-CONTRACT.md`` §7.
    from goldfive import _state_audit  # noqa: PLC0415

    return _state_audit.wrap_plugin_callbacks(_GoldfiveADKPlugin())
