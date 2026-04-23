"""Polymorphic wrapper that makes a goldfive :class:`Runner` look like an ADK agent.

``goldfive.wrap(adk_agent)`` returns a :class:`GoldfiveADKAgent` — a
``google.adk.agents.BaseAgent`` subclass that also exposes the
programmatic :meth:`Runner.run` / :meth:`Runner.close` surface. The
single return value works in two different contexts without the caller
having to choose:

* ``adk web`` loads the object as a ``BaseAgent`` and drives it via
  ``run_async(ctx)``. Each invocation flows through goldfive's
  goal-derive → plan → execute pipeline and is reported back to the ADK
  UI as a short stream of ``Event`` objects.
* Programmatic callers can still ``await wrapped.run("do the thing")``
  and receive a regular :class:`~goldfive.results.ExecutionOutcome`.

The ADK SDK is an optional install — this module lazy-imports ``google.adk``
inside the class body so importing :mod:`goldfive.adapters.adk_wrap`
without the ``adk`` extra raises a clear :class:`ImportError` with an
install hint.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive.results import ExecutionOutcome

if TYPE_CHECKING:
    from goldfive.control import ControlChannel
    from goldfive.protocols import EventSink
    from goldfive.runner import Runner
    from goldfive.types import Goal

log = logging.getLogger("goldfive.adapters.adk_wrap")


try:  # noqa: SIM105 — explicit import-time guard with install hint
    from google.adk.agents import BaseAgent  # type: ignore
    from google.adk.agents.invocation_context import InvocationContext  # type: ignore
    from google.adk.events import Event  # type: ignore
    from google.adk.events.event_actions import EventActions  # type: ignore
    from google.genai.types import Content, Part  # type: ignore
    from pydantic import PrivateAttr
except ImportError:  # pragma: no cover — covered via importorskip in tests
    raise ImportError(
        "goldfive.adapters.adk_wrap requires 'pip install goldfive[adk]'"
    ) from None


def _extract_user_input(ctx: InvocationContext) -> str:
    """Return the latest user-turn text from an ADK :class:`InvocationContext`.

    ADK's context shape varies across versions — the most stable signal
    is ``ctx.user_content``, a ``google.genai.types.Content`` with one
    or more ``Part`` children. If that is missing we fall back to
    ``ctx.session.events`` and pick the last user author. As a last
    resort we return an empty string so the goldfive pipeline can still
    degrade gracefully rather than raising mid-run.
    """
    # Preferred: the user_content on the invocation context.
    content = getattr(ctx, "user_content", None)
    text = _text_from_content(content)
    if text:
        return text

    # Fallback: walk the session's event history for the last user turn.
    session = getattr(ctx, "session", None)
    events = getattr(session, "events", None) or ()
    for event in reversed(list(events)):
        if getattr(event, "author", "") != "user":
            continue
        text = _text_from_content(getattr(event, "content", None))
        if text:
            return text

    return ""


def _text_from_content(content: Any) -> str:
    """Best-effort join of the non-empty ``text`` parts of an ADK Content."""
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or ()
    out: list[str] = []
    for part in parts:
        text = getattr(part, "text", "") or ""
        if text:
            out.append(str(text))
    return "\n".join(out).strip()


def _plan_summary_event(outcome: ExecutionOutcome, author: str, invocation_id: str) -> Event:
    """Build the opening Event that summarises the plan we're about to run."""
    session = outcome.session
    plan = getattr(session, "plan", None)
    return _plan_summary_event_from_plan(plan, author=author, invocation_id=invocation_id)


def _plan_summary_event_from_plan(plan: Any, author: str, invocation_id: str) -> Event:
    """Build a plan-summary Event from a bare :class:`Plan` (may be ``None``)."""
    summary = getattr(plan, "summary", "") or "goldfive plan"
    tasks = list(getattr(plan, "tasks", None) or ())
    lines = [f"**{summary}**"]
    for idx, task in enumerate(tasks, 1):
        title = getattr(task, "title", "") or task.id
        lines.append(f"{idx}. {title}")
    return _text_event("\n".join(lines), author=author, invocation_id=invocation_id)


def _task_result_event(
    task_id: str, title: str, text: str, author: str, invocation_id: str
) -> Event:
    """Build an Event reporting one completed task's result."""
    headline = f"**{title or task_id}**"
    body = text.strip() if text else "(no output)"
    return _text_event(f"{headline}\n{body}", author=author, invocation_id=invocation_id)


def _drift_event(detail: str, author: str, invocation_id: str) -> Event:
    """Build an Event surfacing a drift observation as a warning line."""
    return _text_event(f"drift: {detail}", author=author, invocation_id=invocation_id)


def _terminal_event(
    outcome: ExecutionOutcome, author: str, invocation_id: str
) -> Event:
    """Build the final Event that closes out a run for the ADK UI."""
    if outcome.success:
        text = "goldfive run complete."
    else:
        reason = outcome.reason or "run aborted"
        text = f"goldfive run aborted: {reason}"
    evt = _text_event(text, author=author, invocation_id=invocation_id)
    evt.turn_complete = True
    return evt


def _text_event(text: str, *, author: str, invocation_id: str) -> Event:
    """Construct a minimal assistant-authored Event carrying a text payload."""
    return Event(
        invocation_id=invocation_id or "",
        author=author or "goldfive",
        content=Content(role="model", parts=[Part(text=text)]),
        actions=EventActions(),
    )


async def _outcome_to_adk_events(
    outcome: ExecutionOutcome, ctx: InvocationContext, author: str
) -> AsyncIterator[Event]:
    """Yield one Event per interesting step of ``outcome`` for the ADK UI.

    The stream is intentionally minimal: a plan summary up top, one
    Event per completed task, a best-effort line per recorded drift,
    and a terminal turn-complete Event at the end.

    Retained for back-compat (callers that want to render a full
    outcome synchronously after the fact). The streaming path
    (:meth:`GoldfiveADKAgent._run_async_impl`) now uses the finer-
    grained :func:`_post_run_framing_events` to interleave real
    inner-Runner events with the goldfive-owned framing.
    """
    invocation_id = str(getattr(ctx, "invocation_id", "") or "")
    yield _plan_summary_event(outcome, author=author, invocation_id=invocation_id)
    async for event in _post_run_framing_events(outcome, ctx, author=author):
        yield event


async def _post_run_framing_events(
    outcome: ExecutionOutcome, ctx: InvocationContext, author: str
) -> AsyncIterator[Event]:
    """Yield goldfive's own framing events for the end of a streamed run.

    Emits one Event per completed task, one per recorded drift, and the
    terminal turn-complete Event. These wrap / follow the real inner-
    Runner events streamed through :meth:`Runner.run_streamed` so
    adk-web sees goldfive-owned structure around the agent tree's
    native activity.
    """
    invocation_id = str(getattr(ctx, "invocation_id", "") or "")
    session = outcome.session
    completed = getattr(session, "completed_results", None) or {}
    plan = getattr(session, "plan", None)
    tasks_by_id = {t.id: t for t in (getattr(plan, "tasks", None) or ())}

    for task_id, result_text in completed.items():
        task = tasks_by_id.get(task_id)
        title = getattr(task, "title", "") if task is not None else ""
        yield _task_result_event(
            task_id=task_id,
            title=title,
            text=result_text,
            author=author,
            invocation_id=invocation_id,
        )

    history = getattr(session, "history", None) or ()
    for entry in history:
        detail = _drift_detail(entry)
        if detail:
            yield _drift_event(detail, author=author, invocation_id=invocation_id)

    yield _terminal_event(outcome, author=author, invocation_id=invocation_id)


def _drift_detail(entry: Any) -> str:
    """Return a short one-line drift description from a session history entry."""
    if entry is None:
        return ""
    kind = getattr(entry, "kind", None)
    if kind is None:
        return ""
    detail = getattr(entry, "detail", "") or ""
    kind_str = getattr(kind, "value", None) or str(kind)
    if detail:
        return f"{kind_str}: {detail}"
    return kind_str


class GoldfiveADKAgent(BaseAgent):
    """BaseAgent that runs goldfive's pipeline for each invocation.

    Appears to ADK as a normal agent — it carries the inner agent's
    ``name``, ``description``, and ``sub_agents`` so adk web can render
    it. When the ADK runtime invokes :meth:`_run_async_impl` (via the
    final :meth:`BaseAgent.run_async`), we extract the latest user turn,
    drive the inner goldfive :class:`Runner` for that input, and yield
    synthesized ADK ``Event`` objects summarising the outcome.

    The same instance also exposes the programmatic :meth:`run` /
    :meth:`close` surface of :class:`Runner` so existing goldfive
    callers keep working unchanged::

        root_agent = goldfive.wrap(my_adk_agent)
        app = App(name="demo", root_agent=root_agent)  # adk web path
        outcome = await root_agent.run("do the thing")  # programmatic path
    """

    _inner: Any = PrivateAttr(default=None)
    _runner: Runner = PrivateAttr(default=None)  # type: ignore[assignment]

    def __init__(self, *, inner: Any, runner: Runner) -> None:
        super().__init__(
            name=getattr(inner, "name", "goldfive_agent"),
            description=getattr(inner, "description", "") or "",
            sub_agents=list(getattr(inner, "sub_agents", None) or ()),
        )
        self._inner = inner
        self._runner = runner

    # ------------------------------------------------------------------
    # BaseAgent extension point
    # ------------------------------------------------------------------

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncIterator[Event]:
        """Drive one goldfive run for the latest user turn in ``ctx``.

        :meth:`BaseAgent.run_async` is ``@final`` — overriding this
        protected hook is ADK's documented extension seam.
        """
        # Pin the outer adk-web session id onto the goldfive Session
        # AND the ADKAdapter BEFORE any sub-agent dispatch kicks in
        # (goldfive#161). Three layers of session identity exist under
        # overlay + AgentTool:
        #
        #   1. adk-web's outer ``ctx.session.id`` (the URL session id
        #      users see in the harmonograf UI)
        #   2. goldfive ``Session.id`` (== ``run_id``, minted uuid4
        #      from :class:`Conversation.next_turn_session`)
        #   3. the ADKAdapter's internal ``_session_id`` (minted uuid4
        #      in :meth:`ADKAdapter._ensure_session`)
        #
        # If we let each layer pick its own id, goldfive Event
        # envelopes stamp ``Session.id`` (layer 2) via the #155
        # per-event session routing field, but harmonograf spans carry
        # ``ctx.session.id`` (layer 1). The plan view then shows an
        # empty Gantt: the spans are on the adk-web session, but the
        # plan / task rows are on the goldfive session. See goldfive#161.
        #
        # Pinning layer 1's id onto layer 2 (via the ``session_id=``
        # override on :meth:`Runner.run`) and layer 3 (via the
        # adapter's ``_session_id`` / ``_outer_session_id``) makes all
        # three equal. Harmonograf's plugin (per goldfive#162) caches
        # the same id for every span it stamps. Result: plan +
        # execution co-locate on one harmonograf session.
        #
        # Idempotent / safe fallback:
        # * Empty ``ctx.session.id`` (bare contexts, test harnesses
        #   that skip ``ctx.session``) falls through to the legacy
        #   uuid4 mint from :meth:`Conversation.next_turn_session` +
        #   :meth:`ADKAdapter._ensure_session`, preserving back-compat.
        # * Subsequent invocations (overlay STEER restart, follow-up
        #   turns) receive the SAME ``ctx.session.id`` from adk-web
        #   because adk-web keeps the session stable across turns on
        #   the same URL, so the pin stays consistent.
        outer_sid = self._outer_session_id_from_ctx(ctx)
        self._pin_outer_session_on_adapter(outer_sid)

        user_input = _extract_user_input(ctx)
        if not user_input:
            yield _text_event(
                "goldfive: no user input found in invocation context.",
                author=self.name,
                invocation_id=str(getattr(ctx, "invocation_id", "") or ""),
            )
            return

        # try/finally wraps the entire run so ``_notify_plugins_on_run_end``
        # fires on ALL exit paths — normal completion, adapter raise,
        # upstream ``CancelledError`` (adk-web disconnect mid-stream), or
        # ``GeneratorExit`` when the caller ``aclose()``s this generator
        # early. Observability plugins rely on this hook to close any
        # INVOCATION spans their own ``before_run_callback`` opened but
        # whose matching ``after_run_callback`` never fired — see
        # goldfive#196. ADK's plugin-manager ``after_run_callback`` is
        # placed AFTER an ``async with Aclosing(execute_fn(...))`` block
        # in :meth:`Runner._exec_with_plugin`, NOT inside a ``finally``,
        # so a cancelled or early-closed generator leaks open spans. The
        # teardown hook here is the goldfive-side guarantee that orphan
        # INVOCATION spans get flushed regardless of ADK's gap.
        #
        # Streaming model (goldfive: stream-inner-adk-events):
        # :meth:`Runner.run_streamed` yields every raw ADK Event the
        # inner ``InMemoryRunner`` emits (``transfer_to_agent``,
        # ``function_call``, ``function_response``, model text parts,
        # etc.) as they arrive, followed by exactly one trailing
        # :class:`ExecutionOutcome`. We forward the ADK events through
        # verbatim so adk-web sees the real agent tree's activity, then
        # emit goldfive-owned framing (plan summary up front, per-task
        # result blocks at the end, drift lines, terminal
        # turn-complete) around them.
        invocation_id = str(getattr(ctx, "invocation_id", "") or "")
        plan_summary_yielded = False
        try:
            outcome: ExecutionOutcome | None = None
            async for item in self._runner.run_streamed(
                user_input,
                context={"adk_ctx": ctx},
                session_id=outer_sid or None,
            ):
                if isinstance(item, ExecutionOutcome):
                    outcome = item
                    continue
                # Lazy plan-summary emission: the first real inner-Runner
                # event reaches us AFTER the Runner has installed a plan
                # on the session, so we can synthesize the plan summary
                # with real task titles and yield it BEFORE the first
                # tree event. On runs that produce zero inner events
                # (degenerate no-op trees, immediate abort) we still
                # get to emit it from the outcome branch below.
                if not plan_summary_yielded and outcome is None:
                    session = getattr(self._runner, "_last_session", None)
                    plan = getattr(session, "plan", None) if session is not None else None
                    if plan is not None:
                        yield _plan_summary_event_from_plan(
                            plan,
                            author=self.name,
                            invocation_id=invocation_id,
                        )
                        plan_summary_yielded = True
                yield item
            # Final framing. If we never yielded a plan summary (no
            # inner events landed before the outcome) fall back to
            # synthesizing it from the outcome's session so adk-web
            # still gets a plan header.
            if outcome is not None:
                if not plan_summary_yielded:
                    yield _plan_summary_event(
                        outcome, author=self.name, invocation_id=invocation_id
                    )
                    plan_summary_yielded = True
                async for adk_event in _post_run_framing_events(
                    outcome, ctx, author=self.name
                ):
                    yield adk_event
        finally:
            self._notify_plugins_on_run_end()

    def _notify_plugins_on_run_end(self) -> None:
        """Fire ``on_run_end()`` on every adapter plugin that defines it.

        Fire-and-forget: any plugin exception is swallowed so a faulty
        observability hook cannot mask the real outcome of the run.
        Duck-typed — plugins without the hook (older builds, plugins
        that don't track per-invocation spans) fall through cleanly.

        Why this exists (goldfive#196):
        The harmonograf telemetry plugin opens an INVOCATION span on
        every ``before_run_callback``, keyed by the sub-Runner's ADK
        ``invocation_id``. On normal completion the matching
        ``after_run_callback`` closes it. But when the outer
        :class:`GoldfiveADKAgent` run is cancelled mid-flight (adk-web
        client disconnect, STEER during an AgentTool sub-Runner, crash
        in a sibling sub-Runner), ADK's plugin-manager does NOT fire
        ``after_run_callback`` — it's placed after
        ``async with Aclosing(execute_fn(...))`` in
        :meth:`Runner._exec_with_plugin`, outside any ``finally``. The
        sub-Runner's span then leaks ``status=RUNNING`` in the
        harmonograf DB forever, and the frontend's Live Activity panel
        shows a stuck "N RUNNING" header.

        ADKAdapter.invoke already calls ``on_cancellation`` on the
        OUTER invocation on ``CancelledError``. This hook is the
        broader sweep that fires on EVERY exit path and lets plugins
        close every span they opened during the run — including
        orphaned sub-Runner spans the outer cancel path cannot reach.
        """
        adapter = getattr(self._runner, "agent", None)
        plugins = getattr(adapter, "_plugins", None)
        if not plugins:
            return
        for plugin in plugins:
            hook = getattr(plugin, "on_run_end", None)
            if hook is None:
                continue
            try:
                hook()
            except Exception as exc:  # noqa: BLE001 — defensive
                log.debug(
                    "GoldfiveADKAgent: plugin %r on_run_end raised "
                    "(swallowed): %s",
                    plugin,
                    exc,
                )

    @staticmethod
    def _outer_session_id_from_ctx(ctx: InvocationContext) -> str:
        """Return ``ctx.session.id`` when usable, ``""`` otherwise.

        Defensive: ADK contexts in tests may omit ``session`` or
        substitute it with a value whose ``id`` is ``None`` / empty.
        Centralises the guard so the pin path and the Runner handoff
        stay in sync.
        """
        session = getattr(ctx, "session", None)
        if session is None:
            return ""
        sid = getattr(session, "id", None)
        if not isinstance(sid, str):
            return ""
        return sid

    def _pin_outer_session_on_adapter(self, outer_sid: str) -> None:
        """Adopt ``outer_sid`` on the ADKAdapter's session bookkeeping.

        Pins both ``_outer_session_id`` (for structural consumers that
        want the raw outer id for logging) and ``_session_id`` (used
        by :meth:`ADKAdapter._ensure_session` and
        :meth:`ADKAdapter._heal_pending_tool_calls`). The goal is for
        the adapter's internal ADK session creation to use the SAME id
        as the outer adk-web session so the harmonograf plugin sees
        one session id across both layers.

        No-op when ``outer_sid`` is empty — the adapter's lazy-uuid
        mint still runs and the legacy behaviour is preserved.

        Never overwrites an adapter that already carries a pinned
        ``_session_id`` (constructor ``session_id=`` is authoritative).
        """
        if not outer_sid:
            return
        adapter = getattr(self._runner, "agent", None)
        if adapter is None:
            return
        # Pin the forensic-only outer id when the attribute exists.
        if hasattr(adapter, "_outer_session_id") and not adapter._outer_session_id:
            adapter._outer_session_id = outer_sid
        # Pin the adapter's effective ADK session id. Constructor
        # ``session_id=`` wins; we only set when it was empty.
        if hasattr(adapter, "_session_id") and not adapter._session_id:
            adapter._session_id = outer_sid

    # ------------------------------------------------------------------
    # Runner passthrough — keeps the programmatic entry point working.
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str | list[Goal],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ExecutionOutcome:
        """Run the wrapped goldfive pipeline programmatically."""
        return await self._runner.run(user_input, context=context)

    async def close(self) -> None:
        """Close the wrapped :class:`Runner` (flushes every sink)."""
        await self._runner.close()

    # ------------------------------------------------------------------
    # Runner attribute access — enables harmonograf_client.observe() and
    # anything else that expects to see the Runner's sinks / control.
    # ------------------------------------------------------------------

    @property
    def sinks(self) -> list[Any]:
        """Expose the inner :class:`Runner`'s sink list (mutable)."""
        return self._runner.sinks

    def add_sink(self, sink: EventSink) -> None:
        """Register an additional sink on the inner :class:`Runner`."""
        self._runner.add_sink(sink)

    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """Register an async close hook on the inner :class:`Runner`."""
        self._runner.add_close_hook(hook)

    def add_plugin(self, plugin: Any) -> None:
        """Install an ADK ``BasePlugin`` on the underlying ADK Runner.

        Delegates to :meth:`ADKAdapter.add_plugin` on the wrapped
        adapter. Used by observability integrations (e.g.
        ``harmonograf_client.observe()``) to attach telemetry plugins
        after ``goldfive.wrap(...)`` has built the adapter.
        """
        self._runner.agent.add_plugin(plugin)

    @property
    def control(self) -> ControlChannel | None:
        """Expose the inner :class:`Runner`'s optional ControlChannel."""
        return self._runner.control

    @control.setter
    def control(self, value: ControlChannel) -> None:
        """Attach a :class:`ControlChannel` on the inner :class:`Runner`."""
        self._runner.control = value

    @property
    def inner_agent(self) -> Any:
        """Return the wrapped ADK ``BaseAgent`` — used by subtree walks."""
        return self._inner

    @property
    def runner(self) -> Runner:
        """Return the wrapped goldfive :class:`Runner`."""
        return self._runner


__all__ = ["GoldfiveADKAgent"]
