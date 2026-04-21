"""goldfive adapter for the Claude Agent SDK.

The Claude Agent SDK (``claude_agent_sdk`` on PyPI) exposes its agent loop
through :class:`claude_agent_sdk.ClaudeSDKClient`, a streaming client with
no shared session-state dict. Unlike Google ADK — where per-step callbacks
and a mutable ``session.state`` carry context across turns — goldfive has
to re-render the agent's system prompt on every ``client.query(...)`` call
and consume the returned message stream to drive the
:class:`goldfive.protocols.Steerer`.

Key design points:

* The ``claude_agent_sdk`` import is **optional**; importing this module
  without the SDK installed raises :class:`ImportError` with an install
  hint (``pip install goldfive[claude]``). Every SDK type referenced in
  type hints is guarded behind ``TYPE_CHECKING``.
* Reporting tools are exposed to the agent as an **inline MCP server**
  built from :class:`claude_agent_sdk.SdkMcpTool` — this is the SDK's
  one and only path for bespoke tools. A matching ``PreToolUse`` hook
  intercepts each reporting-tool invocation, routes it through the
  goldfive :class:`~goldfive.reporting.ReportingToolSpec.handler`, and
  blocks the SDK from running the handler a second time (by returning
  ``permissionDecision="deny"`` with the handler's result payload as
  the reason — the cleanest way to short-circuit a tool call from a
  ``PreToolUse`` hook in the current SDK).
* The system prompt template is public and overrideable — see
  :mod:`goldfive.adapters._claude_prompt`.

This module is pinned to the shapes in ``docs/design/PROTOCOLS.md``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from goldfive.adapters._claude_prompt import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from goldfive.adapters._tool_invocation import invoke_tool
from goldfive.drift import classify_stop_reason
from goldfive.reporting import REPORTING_TOOL_NAMES, ReportingToolSpec
from goldfive.results import InvocationResult
from goldfive.types import Session, Task

# --------------------------------------------------------------------------- #
# Optional-dependency guard
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - exercised by optional-install CI job
    import claude_agent_sdk as _sdk  # noqa: F401
except ImportError as _sdk_import_err:  # pragma: no cover
    _SDK_IMPORT_ERROR: ImportError | None = _sdk_import_err
    _sdk = None  # type: ignore[assignment]
else:
    _SDK_IMPORT_ERROR = None


if TYPE_CHECKING:
    # Imported only for type checkers; runtime code uses the module above.
    from claude_agent_sdk import ClaudeSDKClient  # noqa: F401


def _require_sdk() -> Any:
    """Return the imported SDK module, or raise a clear error."""

    if _SDK_IMPORT_ERROR is not None:
        raise ImportError(
            "claude_agent_sdk is not installed. Install the optional extra "
            "with `pip install goldfive[claude]` to use "
            "ClaudeAgentSDKAdapter."
        ) from _SDK_IMPORT_ERROR
    return _sdk


# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

ClientFactory = Callable[[], "ClaudeSDKClient"]
SteererLike = Any  # goldfive.protocols.Steerer (avoid circular import)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class ClaudeAgentSDKAdapter:
    """``AgentAdapter`` implementation backed by the Claude Agent SDK.

    Parameters
    ----------
    client_factory:
        Zero-arg callable returning a fresh ``ClaudeSDKClient``. Called
        once per :meth:`invoke` so each task gets its own connection and
        its own freshly-rendered system prompt — required because the
        SDK treats ``system_prompt`` as immutable after ``connect()``.
    steerer:
        The :class:`goldfive.protocols.Steerer` to route observations
        and reporting-tool calls through. Optional at construction time
        so the adapter can be handed to a Runner that wires it later
        via :meth:`bind_steerer`.
    system_prompt_template:
        Optional ``str.format``-style template overriding
        :data:`goldfive.adapters._claude_prompt.DEFAULT_SYSTEM_PROMPT_TEMPLATE`.
        See that module for the placeholder contract.
    model:
        Optional model alias (``"sonnet"``, ``"opus"``, ``"haiku"``) or
        full model ID passed through to ``ClaudeAgentOptions.model``.
    available_agents:
        Identifiers surfaced by :attr:`available_agents` for planners
        that need to know what agents the adapter can route to. Claude
        Agent SDK treats subagents (``Task`` tool) opaquely in v0.1,
        so this is a simple passthrough list.
    """

    # Name used for the inline MCP server that publishes the reporting
    # tools. The SDK namespaces MCP tools as ``mcp__<server>__<tool>``,
    # which we resolve transparently in the PreToolUse hook.
    _MCP_SERVER_NAME: str = "goldfive_reporting"

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        steerer: SteererLike | None = None,
        system_prompt_template: str | None = None,
        model: str | None = None,
        available_agents: list[str] | None = None,
    ) -> None:
        _require_sdk()
        self._client_factory: ClientFactory = client_factory
        self._steerer: SteererLike | None = steerer
        self._template: str | None = system_prompt_template
        self._model: str | None = model
        self._available_agents: list[str] = list(available_agents or [])

        self._reporting_specs: dict[str, ReportingToolSpec] = {}
        # Populated by register_reporting_tools. Kept separate so an
        # adapter can be re-registered without losing the inline-tool
        # configuration across Runner.run() invocations.
        self._mcp_server_config: Any = None
        self._mcp_tool_names: list[str] = []

    # --------------------------------------------------------------- #
    # Public AgentAdapter surface
    # --------------------------------------------------------------- #

    @property
    def available_agents(self) -> list[str]:
        return list(self._available_agents)

    @property
    def available_agents_tree(self) -> list[dict[str, Any]]:
        """Return a flat single-level tree describing the configured agents.

        ClaudeAdapter is a single-model adapter — every configured name
        is rendered as a depth-0 root leaf so planners that consume
        :attr:`available_agents_tree` (goldfive#151) see the same shape
        across adapters.
        """
        return [
            {
                "name": name,
                "depth": 0,
                "parent": "",
                "role": "root",
                "kind": "Claude",
            }
            for name in self._available_agents
        ]

    def bind_steerer(self, steerer: SteererLike) -> None:
        """Wire a :class:`Steerer` in after construction.

        The Runner/Executor typically calls this once per run so the
        adapter and steerer share state. It is safe to overwrite.
        """

        self._steerer = steerer

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "anthropic",
        call_id: str = "",  # noqa: ARG002 -- part of the protocol
    ) -> None:
        """Route an Anthropic ``thinking`` block to the bound steerer.

        Called opportunistically; the Claude SDK currently surfaces
        thinking blocks via the same assistant-message channel the
        observation hook already listens to. Dedicated extraction
        lives in the hook; this method keeps the adapter protocol
        uniform across backends.
        """
        steerer = getattr(self, "_steerer", None)
        if steerer is None or not text:
            return
        observe = getattr(steerer, "observe_reasoning", None)
        if observe is None:
            return
        await observe(text, task=task, session=session, provider=provider)

    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None:
        """Translate goldfive specs into an inline SDK MCP server.

        Each ``ReportingToolSpec`` becomes an
        :class:`claude_agent_sdk.SdkMcpTool` with a no-op default handler
        — the real routing happens in the PreToolUse hook, which blocks
        the no-op before it runs.
        """

        sdk = _require_sdk()

        self._reporting_specs = {spec.name: spec for spec in tools}

        sdk_tools: list[Any] = []
        qualified_names: list[str] = []

        for spec in tools:
            # The SDK's in-CLI tool name is ``mcp__<server>__<tool>``;
            # goldfive registers tools under bare canonical names, so
            # we track both.
            qualified = f"mcp__{self._MCP_SERVER_NAME}__{spec.name}"
            qualified_names.append(qualified)

            # SdkMcpTool needs an input_schema (JSON Schema dict) plus an
            # async handler. The handler here is the "no-op default" —
            # the PreToolUse hook intercepts before this runs. We still
            # return an SDK-shaped ``content`` payload so that if the
            # hook is ever bypassed the agent gets a valid response.
            sdk_tools.append(
                sdk.SdkMcpTool(
                    name=spec.name,
                    description=spec.description,
                    input_schema=spec.parameters,
                    handler=_make_fallback_handler(spec.name),
                )
            )

        self._mcp_server_config = sdk.create_sdk_mcp_server(
            name=self._MCP_SERVER_NAME,
            tools=sdk_tools,
        )
        self._mcp_tool_names = qualified_names

    async def invoke(
        self,
        task: Task,
        session: Session,
    ) -> InvocationResult:
        """Run one agent turn for ``task`` and return an :class:`InvocationResult`."""

        sdk = _require_sdk()

        prompt_text = _build_user_prompt(task)
        system_prompt = render_system_prompt(
            self._template,
            task=task,
            goals=session.goals,
            plan_summary=_plan_summary(session),
            completed=session.completed_results,
        )

        options = self._build_options(sdk=sdk, system_prompt=system_prompt, session=session)

        client = self._client_factory()
        # ``ClaudeSDKClient`` options are set on the instance; some
        # factories already attach them. We only overwrite when the
        # caller left ``options=None``.
        if getattr(client, "options", None) is None:
            try:
                client.options = options  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - exotic client subclasses
                pass

        final_text_parts: list[str] = []
        stop_reason: str = ""
        result_raw: Any = None
        error: Exception | None = None

        try:
            # ``connect(None)`` leaves the input stream open so we can
            # send one prompt and still iterate over responses.
            await _maybe_await(client.connect(None))
            await client.query(prompt_text)
            async for message in client.receive_response():
                result_raw = message
                # Every message is observed by the steerer so it can do
                # drift detection and emit events. Errors in observe
                # must never kill the invocation loop.
                await _safe_observe(self._steerer, message, session)

                final_text_parts.extend(_collect_text_from(message))
                if _is_result_message(message):
                    stop_reason = getattr(message, "stop_reason", "") or ""
                    break
        except Exception as exc:  # noqa: BLE001 - surface to caller
            error = exc
        finally:
            try:
                await _maybe_await(client.disconnect())
            except Exception:  # pragma: no cover - cleanup best-effort
                pass

        # Classify benign vs drift-worthy stop reasons. Only drift feeds
        # the steerer — benign stops are reported via the return value.
        drift = classify_stop_reason(
            stop_reason,
            current_task_id=task.id,
            current_agent_id=task.assignee_agent_id,
        )
        if drift is not None and self._steerer is not None:
            await _safe_observe(self._steerer, drift, session)

        return InvocationResult(
            task_id=task.id,
            text="\n".join(p for p in final_text_parts if p),
            stop_reason=stop_reason,
            error=error,
            raw=result_raw,
        )

    # --------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------- #

    def _build_options(
        self,
        *,
        sdk: Any,
        system_prompt: str,
        session: Session,
    ) -> Any:
        """Assemble a fresh ``ClaudeAgentOptions`` for this invocation."""

        hook_matcher = sdk.HookMatcher(
            matcher="|".join(REPORTING_TOOL_NAMES),
            hooks=[self._make_pretooluse_hook(session)],
        )

        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "hooks": {"PreToolUse": [hook_matcher]},
        }
        if self._model is not None:
            kwargs["model"] = self._model
        if self._mcp_server_config is not None:
            kwargs["mcp_servers"] = {
                self._MCP_SERVER_NAME: self._mcp_server_config,
            }
            # Make sure the reporting tools are actually callable.
            kwargs["allowed_tools"] = list(self._mcp_tool_names)
        return sdk.ClaudeAgentOptions(**kwargs)

    def _make_pretooluse_hook(
        self,
        session: Session,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Build the async PreToolUse hook closed over this session.

        The hook matches on reporting tool names (both the qualified
        ``mcp__goldfive_reporting__*`` form the SDK emits and the bare
        canonical name, to be robust to SDK naming changes), routes to
        the goldfive handler, and returns a ``deny`` permission decision
        so the SDK never executes the no-op stub inside the inline MCP
        server. Returning the handler's ACK text in
        ``permissionDecisionReason`` is the cleanest way to surface the
        result to the model while short-circuiting execution.
        """

        specs = self._reporting_specs
        steerer = self._steerer

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data.get("tool_name", "") or ""
            bare = _strip_mcp_prefix(tool_name)
            if bare not in specs:
                # Not a reporting tool — let the SDK run it normally.
                return {}

            tool_input = input_data.get("tool_input", {}) or {}
            # Observe as well so the steerer sees tool_use alongside
            # text blocks — useful for drift detectors that watch for
            # report_plan_divergence / report_new_work_discovered.
            await _safe_observe(
                steerer,
                _ToolCallObservation(
                    name=bare,
                    arguments=tool_input,
                    tool_use_id=tool_use_id or "",
                ),
                session,
            )

            # Route through ``invoke_tool`` (NOT ``spec.handler`` direct)
            # so every reporting-tool dispatch picks up the three
            # protection layers: terminal-task rejection, idempotency,
            # and loop-guard. See
            # ``docs/design/TASK-LIFECYCLE.md`` §5 for the contract.
            try:
                ack = await invoke_tool(
                    list(specs.values()),
                    bare,
                    dict(tool_input) if isinstance(tool_input, Mapping) else {},
                    session,
                    steerer,
                )
            except Exception as exc:  # noqa: BLE001 - surface to agent
                ack = {"ok": False, "error": str(exc)}

            reason = json.dumps(ack) if not isinstance(ack, str) else ack
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

        return _hook


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _ToolCallObservation:
    """Lightweight record passed to ``steerer.observe`` for tool_use.

    Using a plain object (instead of an SDK type) keeps the steerer
    framework-agnostic — it only needs ``name`` and ``arguments``.
    """

    __slots__ = ("name", "arguments", "tool_use_id")

    def __init__(self, *, name: str, arguments: dict[str, Any], tool_use_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.tool_use_id = tool_use_id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_ToolCallObservation(name={self.name!r}, args={self.arguments!r})"


def _make_fallback_handler(tool_name: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return an async handler that the SDK calls only if the hook is bypassed."""

    async def _fallback(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"goldfive: tool {tool_name!r} acknowledged via fallback "
                        "handler (PreToolUse hook did not short-circuit)."
                    ),
                }
            ]
        }

    return _fallback


def _build_user_prompt(task: Task) -> str:
    """Simple user-turn prompt — the heavy lifting lives in system_prompt."""

    return (
        f"Begin work on task {task.id!r}: {task.title}.\n"
        "Use the goldfive reporting tools to report progress and completion."
    )


def _plan_summary(session: Session) -> str:
    plan = session.plan
    if plan is None:
        return ""
    if plan.summary:
        return plan.summary
    # Fall back to a bullet list of task titles.
    return "; ".join(f"{t.id}: {t.title}" for t in plan.tasks)


def _collect_text_from(message: Any) -> Iterable[str]:
    """Yield text chunks from any SDK message type we care about."""

    # AssistantMessage.content is a list of ContentBlock — we only pull
    # TextBlock.text. ThinkingBlock stays private; ToolUseBlock text is
    # exposed via the PreToolUse hook.
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                yield text
    # ResultMessage.result holds the final assistant text for non-stream
    # runs; include it so callers always see something in .text.
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        yield result_text


def _is_result_message(message: Any) -> bool:
    """Best-effort check that works whether the SDK is installed or not."""

    if _sdk is not None and isinstance(message, getattr(_sdk, "ResultMessage", ())):
        return True
    return type(message).__name__ == "ResultMessage"


def _strip_mcp_prefix(name: str) -> str:
    """``mcp__goldfive_reporting__report_task_started`` → ``report_task_started``."""

    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


async def _safe_observe(steerer: SteererLike | None, event: Any, session: Session) -> None:
    """Call ``steerer.observe`` without letting exceptions escape."""

    if steerer is None:
        return
    observe = getattr(steerer, "observe", None)
    if observe is None:
        return
    try:
        await observe(event, session)
    except Exception:  # noqa: BLE001 - steerer errors must not kill the stream
        pass


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is a coroutine; otherwise return it verbatim.

    Some SDK subclasses/mocks make ``connect`` / ``disconnect`` sync; this
    shim lets tests pass plain functions.
    """

    if hasattr(value, "__await__"):
        return await value
    return value


__all__ = [
    "ClaudeAgentSDKAdapter",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
]
