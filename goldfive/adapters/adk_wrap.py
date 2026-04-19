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

if TYPE_CHECKING:
    from goldfive.control import ControlChannel
    from goldfive.protocols import EventSink
    from goldfive.results import ExecutionOutcome
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
    and a terminal turn-complete Event at the end. This is enough for
    adk web to render a coherent turn; richer views can subscribe to
    goldfive's own Event stream via a sink.
    """
    invocation_id = str(getattr(ctx, "invocation_id", "") or "")
    yield _plan_summary_event(outcome, author=author, invocation_id=invocation_id)

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
        user_input = _extract_user_input(ctx)
        if not user_input:
            yield _text_event(
                "goldfive: no user input found in invocation context.",
                author=self.name,
                invocation_id=str(getattr(ctx, "invocation_id", "") or ""),
            )
            return

        outcome = await self._runner.run(
            user_input, context={"adk_ctx": ctx}
        )
        async for adk_event in _outcome_to_adk_events(
            outcome, ctx, author=self.name
        ):
            yield adk_event

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
