# 06. Adapters and Instrumentation

## Read this chapter when...

- You are adding a new adapter for some agent framework (LangGraph, MCP,
  a bespoke SDK) and need the exact `AgentAdapter` contract — which
  members are load-bearing, which degrade gracefully, and what breaks if
  you skip one.
- You are touching `goldfive/adapters/adk.py` — tree augmentation,
  `GoldfivePlanner` attach, `available_agents` / `available_agents_tree`,
  the `invoke` / `invoke_passthrough` / `invoke_follow_up` overlay
  surface, or the one-runner construction path.
- You are touching request-side LLM instrumentation:
  `goldfive/adapters/adk_llm_instrumentation.py` (dynamic-instruction
  resolver plumbing, the `max_output_tokens` ratchet, per-call char
  measurement, the runtime tool-surface hint) or the resolver closure in
  `goldfive/prompt_shaper.py`.
- You are debugging "why didn't my `{var}` template substitute after
  goldfive wrapped my agent?" (that is the #477 `inject_session_state`
  re-application path — read it before editing).
- You are debugging the Claude adapter (`goldfive/adapters/claude.py`) or
  the reference `CallableAdapter` (`goldfive/adapters/callable.py`).
- You need the honest parity table: which goldfive features are
  ADK-only (overlay, plugin callbacks, streaming, `ContextEditor`,
  `AgentConfig` structural ceilings) and which every adapter gets.

## Files covered

| File | What lives here |
|---|---|
| `goldfive/protocols.py` | The `AgentAdapter` `Protocol` (`register_reporting_tools` / `invoke` / `emit_reasoning` / `available_agents`) and its optional extensions. |
| `goldfive/adapters/auto.py` | `auto_adapter()` — shape detection that turns any supported agent object into a concrete adapter. Powers `goldfive.wrap`. |
| `goldfive/adapters/callable.py` | `CallableAdapter` — the reference implementation; the preferred vehicle for deterministic tests. |
| `goldfive/adapters/claude.py` | `ClaudeAgentSDKAdapter` — Claude Agent SDK backend (inline MCP server + `PreToolUse` hook). |
| `goldfive/adapters/_claude_prompt.py` | `render_system_prompt` + `DEFAULT_SYSTEM_PROMPT_TEMPLATE` for the Claude adapter. |
| `goldfive/adapters/adk.py` | `ADKAdapter` + tree-walk helpers (`_augment_subtree_with_reporting`, `_attach_goldfive_planner_to_tree`, `_collect_reachable_agent_tree`). |
| `goldfive/adapters/adk_llm_instrumentation.py` | Request-side `LlmRequest` surface: dynamic-instruction resolver plumbing, `max_output_tokens` ratchet, `_measure_request_chars`, runtime tool-surface hint. |
| `goldfive/prompt_shaper.py` | `PromptShaper.make_dynamic_instruction` — the resolver closure and the #477 `inject_session_state` re-application. |
| `goldfive/adapters/adk_reentry.py` | `ReentryKind` + `reentry()` context manager (duplicate-suppression contract). |
| `goldfive/adapters/_tool_invocation.py` | `invoke_tool` — the shared reporting-tool dispatch used by every adapter. |
| `goldfive/convenience.py` | `wrap()` — where `install_dynamic_instructions` and the `AgentConfig` / `ContextEditor` threading actually happens (NOT in the adapter constructor). |
| `.agents/adapters.md`, `.agents/how-to-add-a-new-adapter.md` | Skill docs. Useful but the code on main wins where they disagree — see "Doc drift" callouts below. |

## Invariants that bind you here

1. **No prompt-cooperation contracts** (CANON hard invariant 1). Every
   adapter surface — reporting tools, observation, cancel, terminal
   status flow — must work even if the wrapped agent never calls a
   goldfive reporting tool and never follows an injected instruction. The
   overlay `invoke_passthrough` path exists precisely so the agent runs
   its native prompt untouched while goldfive observes from the outside.
2. **No regex / keyword heuristics for natural-language classification**
   (CANON hard invariant 2). The adapters classify *structured* data
   (ADK class names, tool names, stop reasons, plugin names) by
   exact-equality / duck-typing — that is allowed. Do not add a regex
   that reads an agent's *prose* to decide anything.
3. **Any ADK tree shape must work** (CANON hard invariant 3). Every
   tree-walk in `adk.py` and `adk_llm_instrumentation.py` follows the
   same three edges — `sub_agents` / `inner_agent` / `AgentTool.agent` —
   and is idempotent by `id(agent)`. Coordinator+AgentTool is a
   first-class shape, not an edge case.
4. **Adaptive over predictive** (CANON hard invariant 4). Adapters
   *observe* what the tree did (via plugin callbacks, the reconciler,
   the event stream) and capture it. They do not predict which agent
   will run next or pre-decide a task's fate.
5. **`observation_only=True` is strictly passive** (CANON hard invariant
   5). The dynamic-instruction resolver's *only* sanctioned read of the
   kill-switch is `shaper.should_inject(steerer)`, which resolves through
   `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)`.
   Under `observation_only=True` the resolver returns the caller's
   (templated) instruction **un-augmented** — no "Current assigned task"
   block, no correction block.
6. **Lifecycle gates need stable identity keys** (CANON hard invariant
   6). Correction lookups key on `(agent_name, task_id)` via
   `pending_correction_key`; the current-task pin keys on
   `KEY_CURRENT_TASK_ID`. Never key adapter state on an LLM-minted /
   churning id.

Cross-references: the plugin callback surface (`before_model_callback`,
`after_model_callback`, `before_tool_callback`, delegation observation)
lives in **05-adk-plugin.md**. The overlay executor that calls
`invoke_passthrough` lives in **04-executors-and-control.md**. The
reporting tools and `invoke_tool` protection layers are in
**13-reporting-tools-and-approval.md**. The steerer's `is_active_steering`
kill-switch is in **09-steering-ladder-and-gates.md**. Config surfaces
(`AgentConfig`, `SteeringConfig`) are catalogued in **14-config-reference.md**.

---

## 1. The `AgentAdapter` protocol — required vs optional

The adapter is the one seam between goldfive's orchestration layer and a
concrete agent runtime. It is how goldfive stays framework-agnostic. The
contract is a `@runtime_checkable` `Protocol` in `goldfive/protocols.py`:

```python
# goldfive/protocols.py
@runtime_checkable
class AgentAdapter(Protocol):
    """Wraps an underlying agent framework (ADK, Claude Agent SDK, etc.)."""

    async def register_reporting_tools(
        self,
        tools: list[ReportingToolSpec],
    ) -> None: ...

    async def invoke(
        self,
        task: Task,
        session: Session,
    ) -> InvocationResult: ...

    async def emit_reasoning(
        self,
        text: str,
        *,
        task: Task | None = None,
        session: Session,
        provider: str = "",
        call_id: str = "",
        agent_name: str = "",
    ) -> None: ...

    @property
    def available_agents(self) -> list[str]: ...
```

Because the `Protocol` is `@runtime_checkable`, any duck-typed object
carrying these four members passes `isinstance(x, AgentAdapter)`.
`auto_adapter` (`goldfive/adapters/auto.py`) and the `_is_agent_adapter`
helper in `goldfive/quickstart.py` both rely on that.

### 1.1 The four Protocol members and their degradation profile

| Member | Kind | Called by | If missing / broken |
|---|---|---|---|
| `register_reporting_tools(tools)` | required | executor, once per run between planning and execution | Agent never sees `report_task_*` tools → no reporting-driven transitions. For ADK, `ADKAdapter.register_reporting_tools` also *raises* if augmentation misses a reachable agent (see §6.1). |
| `invoke(task, session)` | required | legacy per-task executors; **deprecated** for ADK (see §7) | No dispatch happens at all. |
| `emit_reasoning(text, ...)` | required-in-Protocol, **optional-in-practice** | test adapters and adapters that self-extract reasoning; **not called by the ADK plugin** (see §8) | Reasoning-channel drift judging degrades to whatever the runtime surfaces through other channels. Every shipped adapter implements it as a safe no-op-if-unbound. |
| `available_agents` (property) | required | planners, to constrain `assignee_agent_id` | Planner assumes no routable agents → emits default-assignee tasks or refuses to plan. |

> **Doc drift note.** `.agents/adapters.md` says "three required members
> plus two overlay extensions" and omits `emit_reasoning` from the
> required set. The **code** (`protocols.py`) lists `emit_reasoning` as a
> Protocol method. The code wins: implement `emit_reasoning`, but make it
> a graceful no-op when no steerer is bound (that is exactly what
> `CallableAdapter.emit_reasoning` and `ClaudeAgentSDKAdapter.emit_reasoning`
> do). See §8 for why the ADK path never calls it.

### 1.2 The optional extensions (NOT part of the Protocol)

These are **not** in the `Protocol` body. Call sites look them up with
`getattr(...)` and fall back when absent, so a legacy or custom adapter
that only implements the four core members still works everywhere.

| Extension | Introduced | Consumer | Fallback when absent |
|---|---|---|---|
| `invoke_passthrough(user_message, *, session, reconciler, ctx)` | goldfive#141 (overlay) | `SequentialExecutor(overlay_mode=True)` | Executor falls back to per-task `invoke`. Only `ADKAdapter` implements it. |
| `available_agents_tree` (property) | goldfive#151 | `GoldfivePlanner` orchestration context, `LLMPlanner` assignee-hint selection | Callers fall back to the flat `available_agents` list. |
| `invoke_follow_up(task, session)` | goldfive#141 | external callers / interactive tooling only (the overlay stopped calling it at #163) | No follow-up nudge; overlay marks missed tasks `NOT_NEEDED` instead. |
| `subscribe_adk_events` / `unsubscribe_adk_events` | streaming | `Runner.run_streamed` | Streaming yields goldfive events only, not raw ADK events (see §9). |
| `bind_steerer(steerer)` | — | executor / Runner wiring | `emit_reasoning` and reporting-tool routing have no steerer to reach. |
| `add_plugin(plugin)` | goldfive#166 | `harmonograf_client.observe()` post-`wrap` | No post-construction ADK plugin install. |

The `Protocol` docstring in `protocols.py` is explicit about why
`available_agents_tree` is *deliberately* excluded:

```python
# goldfive/protocols.py (AgentAdapter)
# NOTE: goldfive#151 added a structured ``available_agents_tree``
# property on the shipped adapters (ADK, Claude, Callable) for the
# tree-aware planner, but it is intentionally *not* part of the
# Protocol so custom / legacy adapters that only expose
# ``available_agents`` still pass ``isinstance(x, AgentAdapter)``
# checks. Call sites look the attribute up via ``getattr`` and
# fall back to the flat list when absent.
```

**Rule for weak models:** if you want to add a capability that only some
adapters can implement, do NOT add it to the `AgentAdapter` `Protocol`.
Add it to the concrete adapters and look it up with `getattr` at the call
site, following the `available_agents_tree` precedent. Adding a member to
the `Protocol` breaks `isinstance(x, AgentAdapter)` for every third-party
adapter that predates your change.

---

## 2. `auto_adapter` and `goldfive.wrap` dispatch

`goldfive.wrap(agent, ...)` (`goldfive/convenience.py`, `wrap`) is the
front door. It calls `auto_adapter(agent, ...)` in
`goldfive/adapters/auto.py`, which turns any supported "agent" shape into
a concrete `AgentAdapter`. The detector deliberately avoids importing the
ADK and Claude SDKs at module load so callers who only install the base
extras never pay the import cost.

### 2.1 Dispatch order (from `auto_adapter`)

1. `isinstance(agent, AgentAdapter)` → return it verbatim (already an
   adapter).
2. `_looks_like_adk_agent(agent)` **or** `_looks_like_adk_runner(agent)`
   → build `ADKAdapter` (lazy import; raises `ImportError` if the `adk`
   extra is missing). ADK wins ambiguous cases.
3. `_looks_like_claude_client_factory(agent)` → build
   `ClaudeAgentSDKAdapter(client_factory=agent)`.
4. `_looks_like_async_agent_callable(agent)` → wrap in
   `CallableAdapter(agent, available_agents=["default"])`.
5. Otherwise `raise TypeError` with the list of recognised shapes.

The ADK-over-callable precedence is intentional — an ADK `BaseAgent`
quacks callable (its `run_async` is a coroutine method), so an object
that matches both is almost always ADK. `_looks_like_adk_agent` checks the
MRO for a `google.adk.*` base, then falls back to a `sub_agents` + `name`
duck-type; `_looks_like_claude_client_factory` inspects the signature for
a zero-required-arg callable returning a `ClaudeSDKClient`.

### 2.2 What `wrap()` threads into the adapter

`wrap()` (`goldfive/convenience.py`) resolves a `RuntimeConfig` (from the
`runtime=` kwarg or `RuntimeConfig.from_env()`) and forwards three typed
config surfaces into `auto_adapter`, which forwards them **only** to
`ADKAdapter` (non-ADK shapes ignore them — they have no analogous
surface):

```python
# goldfive/convenience.py (wrap)
adapter = auto_adapter(
    agent,
    plugins=plugins,
    llm_call_timeout_ms=resolved_runtime.agent.call_timeout_ms,
    agent_max_output_tokens=resolved_runtime.agent.max_output_tokens,
    context_editor=context_editor,
)
```

**Critical ordering fact for weak models:** `install_dynamic_instructions`
is called from **`wrap()`**, *before* `auto_adapter` builds the adapter —
NOT from the `ADKAdapter` constructor:

```python
# goldfive/convenience.py (wrap)
if is_adk_agent(agent):
    from goldfive.adapters.adk_llm_instrumentation import (
        install_dynamic_instructions,
        log_dynamic_instruction_opt_out,
    )
    if dynamic_instruction:
        touched = install_dynamic_instructions(agent)
        ...
    else:
        log_dynamic_instruction_opt_out(agent)
```

So if you construct `ADKAdapter(agent)` directly (bypassing `wrap`), you
get reporting-tool augmentation and `GoldfivePlanner` attach (those *are*
in the constructor) but **not** the dynamic-instruction resolver. Tests
that need plan-causal prompting must go through `wrap` or call
`install_dynamic_instructions` themselves. This split is a common source
of "why isn't refine landing in the prompt?" confusion.

---

## 3. `CallableAdapter` — the reference implementation

`goldfive/adapters/callable.py`. This is the canonical example every
other adapter follows, and the preferred vehicle for deterministic tests
of the orchestration layer because it removes LLM non-determinism from the
loop. It wraps an async callable of shape:

```python
async def agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult: ...
```

### 3.1 What it does

- `register_reporting_tools(tools)` — stores `list(tools)` on
  `self._tools`. Forwarded to the callable verbatim on every `invoke`.
- `invoke(task, session)` — `return await self._agent(task, session, self._tools)`.
  The callable drives reporting-tool handlers directly (they are plain
  awaitables).
- `emit_reasoning(text, ...)` — routes to
  `steerer.drift.observe_reasoning` **if a steerer was bound via
  `bind_steerer`**, otherwise returns silently. `CallableAdapter` has no
  intrinsic reasoning to capture (the callable is opaque), so this exists
  for tests and for callables that choose to forward reasoning themselves.
- `available_agents` / `available_agents_tree` — the tree is a flat
  single-level shape (every configured name is a `depth=0` `role="root"`
  `kind="Callable"` leaf), so planners that consume
  `available_agents_tree` see a consistent shape across adapters.

### 3.2 The `emit_reasoning` degradation pattern (copy this)

Every shipped adapter's `emit_reasoning` follows the same defensive
shape. Note the `TypeError` fallback — it exists so a steerer built
before the `agent_name` kwarg landed still works:

```python
# goldfive/adapters/callable.py (CallableAdapter.emit_reasoning)
steerer = getattr(self, "_steerer", None)
if steerer is None:
    return
observe = getattr(getattr(steerer, "drift", None), "observe_reasoning", None)
if observe is None:
    return
try:
    await observe(text, task=task, session=session,
                  provider=provider, agent_name=agent_name)
except TypeError:
    await observe(text, task=task, session=session, provider=provider)
```

When you write a new adapter, copy this shape verbatim. It never raises,
it degrades to a no-op when unbound, and it survives a steerer signature
change. `bind_steerer(steerer)` (which may be called with `None` to
unbind) sets `self._steerer`.

---

## 4. `ClaudeAgentSDKAdapter`

`goldfive/adapters/claude.py`. Backs the Claude Agent SDK
(`claude_agent_sdk` on PyPI, optional extra `goldfive[claude]`). This is
the most important adapter to understand for the parity discussion (§12)
because it shows what a **non-ADK** adapter can and cannot do.

### 4.1 Optional-dependency guard

The SDK import is optional. The module imports without the SDK installed;
`_require_sdk()` raises a clear `ImportError` with an install hint
(`pip install goldfive[claude]`) only when the adapter is actually
constructed or used. Every SDK type in a type hint is behind
`TYPE_CHECKING`. When you touch this file, keep that guard — importing
`goldfive.adapters.claude` must never hard-fail on a base install.

### 4.2 How reporting tools reach the agent

The Claude SDK's only path for bespoke tools is an **inline MCP server**
built from `claude_agent_sdk.SdkMcpTool`. `register_reporting_tools`:

1. Stores the specs in `self._reporting_specs` (keyed by name).
2. For each spec, builds an `SdkMcpTool` with a *no-op fallback handler*
   (`_make_fallback_handler`) and records the qualified name
   `mcp__goldfive_reporting__<tool>`.
3. Calls `sdk.create_sdk_mcp_server(...)` and stashes the config +
   qualified names (`_mcp_server_config`, `_mcp_tool_names`).

The real routing happens in a **`PreToolUse` hook**
(`_make_pretooluse_hook`, closed over the session). When the agent calls a
reporting tool, the hook:

1. Strips the `mcp__goldfive_reporting__` prefix (`_strip_mcp_prefix`).
2. `_safe_observe`s a `_ToolCallObservation` so the steerer sees the
   `tool_use` alongside text blocks (useful for detectors watching
   `report_plan_divergence` / `report_new_work_discovered`).
3. Routes through `invoke_tool(...)` — **not** `spec.handler` directly —
   so the three protection layers fire (see §13-reporting-tools). This is
   the exact same dispatch `ADKAdapter` uses.
4. Returns `permissionDecision="deny"` with the handler's ACK JSON as
   `permissionDecisionReason`, which short-circuits the SDK's no-op stub
   and surfaces the result to the model. This is the cleanest way to
   short-circuit a tool call from a `PreToolUse` hook in the current SDK.

### 4.3 Why a fresh client per `invoke`

`invoke` calls `self._client_factory()` once per task. The SDK treats
`system_prompt` as immutable after `connect()`, so each task needs its own
connection and its own freshly-rendered system prompt. Unlike ADK — which
has per-step callbacks and a mutable `session.state` carrying context
across turns — the Claude adapter re-renders the whole system prompt every
turn via `render_system_prompt` (`goldfive/adapters/_claude_prompt.py`).

`invoke` observes every SDK message through `_safe_observe` (errors never
kill the loop), collects text via `_collect_text_from` (`TextBlock.text`
and `ResultMessage.result`; `ThinkingBlock` stays private), and on the
terminal `ResultMessage` classifies the `stop_reason` via
`classify_stop_reason(...)` — only drift-worthy stops feed the steerer;
benign stops are returned in the `InvocationResult`.

### 4.4 The system-prompt template

`render_system_prompt` uses a plain `str.format` template
(`DEFAULT_SYSTEM_PROMPT_TEMPLATE`) with four required placeholders:
`{goal_block}`, `{task_block}`, `{plan_summary}`, `{completed}`. A
`KeyError` from `str.format` surfaces unchanged so template bugs are loud,
not silent. Callers override with `system_prompt_template=`.

---

## 5. `ADKAdapter` — construction and the one-runner model

`goldfive/adapters/adk.py`, `class ADKAdapter`. This is the largest and
most feature-complete adapter. It implements the single-Runner model
(goldfive#130): one ADK `InMemoryRunner` is built around the caller's root
agent, and every task drives that one runner. Delegation *inside* the tree
happens via ADK's native `AgentTool` / `transfer_to_agent` / `sub_agents`
— goldfive does not build a dispatch registry.

### 5.1 Two construction modes

`ADKAdapter.__init__(agent_or_runner, ...)`:

- **Agent mode** (the common case): `agent_or_runner` is a `BaseAgent`.
  The adapter builds an `InMemoryRunner` via `_build_runner(agent, plugins=...)`,
  installs the goldfive plugin, augments the tree, and attaches
  `GoldfivePlanner`.
- **Degraded pre-built-runner mode**: `_looks_like_runner(agent_or_runner)`
  is true. The adapter uses the caller's runner verbatim, sets
  `self._degraded_prebuilt_runner = True`, and **skips** the tree-walk
  augmentation integrity check, the `GoldfivePlanner` attach, and the
  reachable-agent enumeration (it only knows about the root agent). Use
  this only when a caller has taken over runner construction.

### 5.2 Constructor knobs (all optional)

| Kwarg | Default | Effect |
|---|---|---|
| `user_id` | `"goldfive_user"` | ADK session lookup id. |
| `session_id` | `None` | Stable ADK session id; else minted lazily. |
| `app_name` | runner's / agent's name | ADK `app_name`. |
| `plugins` | `[]` | Extra ADK `BasePlugin`s installed on the one runner (ADK propagates them into `AgentTool` sub-Runners). |
| `agent_tool_cap` | `DEFAULT_AGENT_TOOL_CAP` (16) | Max `AgentTool` spawns per top-level invocation; on exceed the plugin emits `RUNAWAY_DELEGATION` and cancels. `0`/negative disables. |
| `llm_call_timeout_ms` | plugin default | Per-LLM-call wall-clock budget. `0`/negative disables the watcher. |
| `agent_max_output_tokens` | plugin default (`DEFAULT_AGENT_MAX_OUTPUT_TOKENS` = 16384) | Structural `max_output_tokens` ceiling for sub-agent LLM calls (goldfive#256, §10.3). `0`/negative disables ratcheting. |
| `context_editor` | `None` | Request-side `ContextEditor` (goldfive#397); `None` short-circuits the plugin's `before_model_callback` with one `is None` check. |

### 5.3 Wrap-time integrity check

At the end of `__init__` (non-degraded mode) the adapter asserts the
goldfive plugin actually landed on the runner, and raises `RuntimeError`
if not — because without it, reporting callbacks, state-protocol writes,
and drift observation would all be silently broken:

```python
# goldfive/adapters/adk.py (ADKAdapter.__init__)
if not self._degraded_prebuilt_runner:
    plugin_name = getattr(self._plugin, "name", "")
    installed = list(getattr(getattr(self._runner, "plugin_manager", None), "plugins", []))
    if not any(getattr(p, "name", "") == plugin_name for p in installed):
        raise RuntimeError(
            f"ADKAdapter: goldfive plugin {plugin_name!r} failed to "
            f"install on the runner — reporting callbacks, "
            f"state-protocol writes, and drift observation would all "
            f"be broken"
        )
```

Do not "helpfully" downgrade this to a warning. A broken plugin install
means silent orchestration failure, which is far worse than a loud crash.

### 5.4 Per-session state isolation (do not collapse the dicts)

The `ADKAdapter` is shared across every goldfive `Session` driven by one
`Runner`. Four pieces of per-invocation state are keyed by session so
concurrent sessions on one adapter do not leak into each other (PR #294
audit / goldfive#271 follow-up, building on PR #301):

- `_adk_session_ids_by_conv` — keyed by `Session.conversation_id` (stable
  across turns of one Conversation, unique across concurrent ones).
- `_next_cancel_reasons` — keyed by `Session.id`; set via
  `set_next_cancel_reason`, consumed by `_consume_next_cancel_reason`.
- `_pending_tool_call_ids_by_session` / `_pending_tool_call_names_by_session`
  — keyed by `Session.id`.
- `_inflight_invoke_tasks` — keyed by `Session.id`; captured via
  `asyncio.current_task()` at `_invoke_internal` entry so
  `request_cancel` can fire `task.cancel()` on the right session's
  in-flight invocation.

Each has a legacy bare-attribute `@property` shim (e.g.
`_next_cancel_reason`, `_inflight_invoke_task`) over the empty-key (`""`)
bucket, so single-session tests that read/write the bare attribute keep
working. **Do not** revert these to bare instance attributes — two
concurrent sessions would then share one `set`/`dict` and session A's
cancel could heal session B's pending tool calls, corrupting both ADK
sessions' function-call/response pairing. This is a stable-identity-key
concern (invariant 6): the key here is the goldfive `Session.id`, not an
LLM-minted id.

---

## 6. Tree augmentation: reporting tools, GoldfivePlanner, agent-tree walk

Four tree-walks in `adk.py` share one traversal contract: follow
`sub_agents` / `inner_agent` / `AgentTool.agent` edges, dedupe by
`id(agent)`, and push children **before** any per-node skip check so
non-matching container nodes (`SequentialAgent`, `ParallelAgent`) still
propagate the walk to their children. Memorise this shape — every walk in
the adapter (and in `adk_llm_instrumentation.install_dynamic_instructions`)
uses it.

### 6.1 `_augment_subtree_with_reporting`

Appends the reporting `FunctionTool`s to every reachable agent that
carries a `tools` list. Idempotent: agents already carrying any canonical
reporting tool name are skipped. Called from `register_reporting_tools`
(which also augments the root agent separately, then calls the subtree
walk).

Coverage across the *whole* tree matters because an `AgentTool`
sub-invocation can itself report terminal status for the outer task — so
every reachable agent needs the reporting tools regardless of which one
drives each turn. After augmentation, `register_reporting_tools` runs a
**second** full walk and `raise RuntimeError` if any reachable named agent
is missing a reporting tool (non-degraded mode). That is deliberate: a
partial augmentation leaves a sub-agent unable to report terminal status,
which silently breaks the early-exit-on-terminal optimization inside an
`AgentTool` sub-invocation. The error message names the missing agents and
tools.

`_build_function_tool(spec)` wraps a `ReportingToolSpec` as a
`google.adk.tools.FunctionTool` around an ACK shim
(`_build_ack_shim` + `_apply_llm_signature`) — the shim carries the tool's
LLM-visible signature; the real routing runs through the plugin's
`before_tool_callback` → `invoke_tool`.

### 6.2 `_attach_goldfive_planner_to_tree` (goldfive#153)

Attaches a `GoldfivePlanner` to every reachable `LlmAgent` (duck-typed by
presence of a settable `planner` attribute — `SequentialAgent` /
`ParallelAgent` / custom `BaseAgent` subclasses have no `planner` field
and are skipped silently). Per node:

- Opt-out: if `getattr(agent, GOLDFIVE_PLANNER_OPT_OUT_ATTR, False)` is
  truthy (`_goldfive_planner_opt_out = True`), skip.
- `agent.planner is None` → attach a fresh `GoldfivePlanner()`.
- Already a `GoldfivePlanner` → skip (idempotent; re-wrap must not stack).
- Otherwise **compose**: `GoldfivePlanner(user_planner=existing)` — the
  user's `BasePlanner` is wrapped, not replaced.

Individual assignment failures (frozen pydantic models) are logged at
DEBUG and skipped so one bad agent doesn't block the rest of the tree.
`_rebind_goldfive_planners(root, agent_registry=, steerer=, session=)`
runs once per `_invoke_internal` right before `runner.run_async` to bind
each `GoldfivePlanner` to the live steerer, session, and agent registry.

> This is the ADK-side auto-attach. The `GoldfivePlanner` /
> `BasePlanner` design itself is in **10-planning-and-revision.md**.

### 6.3 `available_agents` and `available_agents_tree`

- `_collect_reachable_agent_names(root)` → sorted unique names →
  `available_agents`. A shallow observation for the planner's benefit,
  NOT a dispatch registry (single-Runner model, goldfive#130).
- `_collect_reachable_agent_tree(root)` → list of dicts with
  `name` / `depth` / `parent` / `role` (`"root"` | `"intermediate"` |
  `"leaf"`) / `kind` (the raw ADK class name — tree-agnostic, no semantic
  interpretation). BFS so depth/parent are minimal; first visit wins;
  duplicate reachable edges collapse by `id(agent)`. This is the reference
  implementation for `available_agents_tree`. `LLMPlanner` (goldfive#151)
  renders it as an "AGENT TREE" section and validates that every task's
  `assignee_agent_id` is reachable.

The property returns a fresh shallow copy per access
(`[dict(entry) for entry in self._available_agents_tree]`) so callers
cannot mutate the cache in place. In degraded (pre-built-runner) mode the
adapter only knows the root, so both surfaces contain a single root entry.

---

## 7. The overlay path: `invoke` / `invoke_passthrough` / `invoke_follow_up`

### 7.1 The three entry points and the message shapes

All three delegate to `_invoke_internal(task, session, new_message, reconciler)`.
The difference is the message content and whether a `task` / `reconciler`
is threaded:

| Method | Status | `task` | Message builder | Message shape |
|---|---|---|---|---|
| `invoke_passthrough(user_message, *, session, reconciler, ctx)` | **primary overlay path** (goldfive#141) | `None` | `_passthrough_message_parts` | The user's original request **verbatim** — no task framing, no goldfive jargon. |
| `invoke(task, session)` | **DEPRECATED** | the task | `_follow_up_message_parts` | `"Also, please: {title}. {description}"` |
| `invoke_follow_up(task, session)` | retained for external callers only | the task | `_follow_up_message_parts` | same gentle phrasing |

The overlay model exists because of goldfive#141: the old
`_new_message_parts` shape — `"Task: X. Use the goldfive.* session-state
keys ..."` — caused coordinator agents with flow-oriented prompts to treat
*every* plan task as a brand-new user request and re-run their full
pipeline for each one. `_new_message_parts` is now a DEPRECATED alias for
`_follow_up_message_parts`, kept only for back-compat.

`_passthrough_message_parts` sends the operator's input plain so the tree
runs its native pipeline exactly as it would under bare ADK, and goldfive
observes via plugin callbacks + the `PlanReconciler`. This is invariant 1
(no prompt-cooperation) made concrete: the agent never has to know
goldfive exists. On the passthrough path the `SessionContext.task` field
is `None` (there is no single "current task" during a passthrough); the
plugin's state-protocol writes still run but stamp no current-task
metadata, and reporting-tool calls route through the reconciler's
observation pipeline instead of per-task attribution. The returned
`InvocationResult` has an empty `task_id`.

> **Doc drift note.** `.agents/adapters.md` documents `invoke_follow_up`
> as an active overlay method. The code says it is **no longer called by
> the overlay** as of goldfive#163 — the overlay now marks PENDING tasks
> `NOT_NEEDED` at the end of the passthrough invocation instead of
> dispatching follow-ups (flow-prompted coordinators re-ran their whole
> pipeline on each follow-up, turning a ~10-min run into 40+ min). STEER
> is the supported user-driven path for uncovered work. Code wins.

### 7.2 The re-entry contract (`adk_reentry.py`)

`invoke_passthrough` wraps its inner `runner.run_async` call in
`with reentry(ReentryKind.OVERLAY_REPLAY):`. This pins the
`current_reentry_kind` contextvar so plugins observing the inner runner's
`on_user_message_callback` can tell "this user-message is goldfive
re-feeding the operator's input" from a genuine fresh user turn — the
outer (adk-web) runner already observed and emitted the operator's input
once (harmonograf#234 root cause: 6 `UserMessageReceived` envelopes for a
4-turn session).

`ReentryKind` values (`goldfive/adapters/adk_reentry.py`):

| Value | Meaning |
|---|---|
| `USER_TURN` (default) | Plain ADK use; no behaviour change for non-goldfive callers. |
| `OVERLAY_REPLAY` | goldfive re-feeding the operator's verbatim input. |
| `NUDGE_REPLAY` | goldfive replaying a soft nudge. |
| `STEER_REPLAY` | goldfive replaying a steer. |
| `GOLDFIVE_STEER_REPLAY` | goldfive-internal steer replay (distinct from a user-driven STEER). |

The `reentry()` context manager keeps the **most-specific** label visible
under nesting: a `STEER_REPLAY` / `NUDGE_REPLAY` layered inside an
`OVERLAY_REPLAY` (the natural shape when the executor pins the steer label
before `invoke_passthrough` enters `OVERLAY_REPLAY`) keeps the steer/nudge
label visible. Entering `OVERLAY_REPLAY` while already inside a
`STEER_REPLAY`/`NUDGE_REPLAY` does NOT downgrade the visible label. The var
is always reset on exit, including on exception.

When you touch the overlay path, do not remove the `reentry(...)` wrapper
— you will silently reintroduce duplicate user-message emission in any
downstream telemetry plugin.

---

## 8. `emit_reasoning` and where reasoning actually gets extracted

This is the single most-misunderstood part of the adapter surface. There
are **two** reasoning routes, and the ADK production path does NOT use
`adapter.emit_reasoning`:

1. **ADK path (production):** the goldfive plugin's `after_model_callback`
   (`goldfive/adapters/_adk_plugin.py`) extracts the reasoning text
   (`_extract_reasoning` / the reasoning-content fallback, goldfive#263)
   and routes it **directly** to `steerer.drift.observe_reasoning`. It
   never calls `adapter.emit_reasoning`. Grepping confirms zero calls to
   `emit_reasoning` inside `_adk_plugin.py`.

2. **`adapter.emit_reasoning` (protocol-uniform surface):** used by
   adapters that can self-extract reasoning but have no plugin doing it
   for them — the Claude adapter's uniform surface, `CallableAdapter` for
   callables that forward their own reasoning, and the testkit adversarial
   adapters (`goldfive/testkit/adversarial.py`) which call
   `self.emit_reasoning(...)` to inject synthetic reasoning for drift
   tests.

**Consequence for weak models:** if you are wiring reasoning observation
for the ADK backend, the code you want is in the plugin's
`after_model_callback` (05-adk-plugin.md), NOT in `ADKAdapter.emit_reasoning`.
`ADKAdapter.emit_reasoning` exists for protocol uniformity and manual/
external callers; changing it will not affect the ADK reasoning-drift
pipeline. Conversely, when you write a **non-ADK** adapter that can see
reasoning tokens, `emit_reasoning` is your route — implement it with the
§3.2 defensive shape.

---

## 9. `subscribe_adk_events` — inner ADK event fan-out

Streaming (`Runner.run_streamed`, `goldfive/runner.py`) needs the raw
inner-Runner ADK `Event` stream, not just goldfive's own events. The
adapter exposes an optional fan-out:

- `subscribe_adk_events(listener)` — registers a **sync** callable
  invoked once per event the adapter consumes from `runner.run_async`,
  **in order**, **before** the adapter's own bookkeeping (pending-tool
  tracking, final-event detection, runaway-delegation cap). This
  guarantees the outer consumer sees the exact stream ADK delivers,
  unfiltered. Registration is deduped — the same listener is added once.
- Listeners **MUST be sync** and **MUST NOT block on I/O** — a
  `queue.put_nowait` or `list.append` is the expected shape. Any exception
  is swallowed with a DEBUG log (`_dispatch_adk_event`) so a faulty
  subscriber cannot break the real run. The adapter fans out to listeners
  from inside `_invoke_internal`'s event loop via `_dispatch_adk_event`.
- `unsubscribe_adk_events(listener)` — idempotent removal.

`Runner.run_streamed` looks the method up with `getattr(self.agent,
"subscribe_adk_events", None)`; adapters that don't expose it still run —
streaming just yields goldfive events without the raw ADK layer. This is
the optional-extension pattern again (§1.2). Streaming is ADK-only (see
the parity table §12).

---

## 10. `adk_llm_instrumentation.py` — the request-side surface

`goldfive/adapters/adk_llm_instrumentation.py` (Wave B2 of the
modularization plan) consolidates *everything that touches an `LlmRequest`
on the way IN to the model* into one audit-friendly module. It owns four
concerns. Three are covered here; the resolver (concern 4) gets its own
section (§11) because it is the trickiest and carries the #477 fix.

### 10.1 Per-call char measurement — `_measure_request_chars` (goldfive#172)

Read-only. Returns `(total_chars, messages_count)` for an `LlmRequest` by
walking `llm_request.contents` and summing the char count of each leaf
part (`part.text`, `part.function_call` as `name + json(args)`,
`part.function_response` as `name + json(response)`) plus the
`config.system_instruction`. Unknown part shapes fall through silently
(count zero). **Returns `(0, 0)` on any failure** — instrumentation must
never raise into the caller path. Consumed by the plugin's
`before_model_callback` for the `goldfive.llm.request` log line and event
payloads.

### 10.2 Runtime tool-surface hint (goldfive#168 R3)

`_build_runtime_tools_hint(session)` composes a "currently-relevant tools"
block from `session.plan.tasks` grouped by `assignee_agent_id`: for each
agent it summarises PENDING titles (capped at three) or "all assigned
tasks complete; do NOT re-invoke this agent" (using
`TERMINAL_TASK_STATUSES` to decide done vs remaining). The block is
bracketed by `_RUNTIME_TOOLS_HINT_PREFIX` (`[GOLDFIVE PLAN-STATE HINT —`)
and `_RUNTIME_TOOLS_HINT_END` (`[/GOLDFIVE PLAN-STATE HINT]`).
`_strip_prior_runtime_tools_hint(existing)` removes a previously-injected
block before appending a fresh one so the marker count stays exactly one
per request (the R3 dedup contract).

The composition/strip primitives live here; the **injection site** lives
on `PromptShaper` (Wave B1), which owns the `observation_only` gate. This
uses exact-string marker matching (structured, allowed) — not NL
classification (invariant 2).

### 10.3 `max_output_tokens` ratchet (goldfive#256)

`_apply_agent_max_output_tokens_cap(llm_request, ceiling)` ratchets
`llm_request.config.max_output_tokens` **down** to `ceiling` with
**smaller-wins** semantics: an already-tighter cap is left alone; a
missing/zero/negative/larger value is set to `ceiling`. `ceiling <= 0` is
the operator opt-out (returns `(0, 0)`, touches nothing). Returns
`(previous_value, applied_value)`. `DEFAULT_AGENT_MAX_OUTPUT_TOKENS` is
16384 (matching `LLMPlanner.MAX_OUTPUT_TOKENS`). Best-effort: any
read/write failure on `llm_request.config` is swallowed at DEBUG so a
future ADK schema change can't crash the callback — the watcher and
planner cap still bound runaway calls.

Operators tune it via `AgentConfig(max_output_tokens=...)` (threaded
through `wrap` → `auto_adapter` → `ADKAdapter` → the plugin) or the
`GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS` env var.

### 10.4 Module boundaries (do not merge these files)

The module docstring records two deliberate boundaries. Respect them:

- **`_adk_state_protocol.py` stays separate.** It owns the audit-guarded
  `_set` shim and the `goldfive.*` state-key constants
  (`KEY_CURRENT_TASK_ID`, etc.). `goldfive._state_audit` patches its
  `_set` at import time and catalogues its file path in `_KNOWN_CALLERS`.
  Merging it here would force a catalog-and-patch surface migration —
  high-risk (V2/V3/V4 migration history) for no benefit.
- **`_adk_plugin.py` stays separate.** The *response-side* callbacks
  (`before_tool_callback`, `after_model_callback`, `on_event_callback`,
  `on_tool_error_callback`) are not request-side mutation. The plugin
  re-exports the request-side helpers at load so legacy
  `from goldfive.adapters._adk_plugin import _measure_request_chars`
  callsites (in tests) keep resolving.

---

## 11. The dynamic-instruction resolver and the #477 `inject_session_state` fix

This is the load-bearing correctness surface of the chapter. Read it fully
before editing either `install_dynamic_instructions`
(`adk_llm_instrumentation.py`) or `PromptShaper.make_dynamic_instruction`
(`prompt_shaper.py`).

### 11.1 The problem it solves (goldfive#251, plan-causal prompting)

An `LlmAgent`'s `instruction` field is bound at `LlmAgent(...)`
construction. When goldfive's plan changes mid-run (refine lands, a task
is superseded, a correction is injected), the plan updates in goldfive's
`Session` but the agent's baked-in prompt does not — so the LLM keeps
executing its *original* instruction. The resolver fixes that by replacing
each wrapped `LlmAgent`'s static `instruction` string with a callable
`(ReadonlyContext) -> str | Awaitable[str]` that re-resolves the agent's
current task from goldfive `Session` state **every turn**. ADK's
`canonical_instruction` invokes the callable per turn and returns
`bypass_state_injection=True` for the result, so refine landing in state is
picked up on the NEXT turn with no transcript rewrite.

### 11.2 Two files, one closure

The work is split:

- `install_dynamic_instructions(root_agent)` in
  `adk_llm_instrumentation.py` — the **installer**: walks the tree and
  decides per node whether to install a resolver.
- `PromptShaper().make_dynamic_instruction(original_instruction, agent_name)`
  in `prompt_shaper.py` — the **factory**: produces the resolver closure
  (Wave B1 moved the closure here so the four prompt-shape injection sites
  share one `observation_only` gate). The installer calls the factory:

```python
# goldfive/adapters/adk_llm_instrumentation.py (install_dynamic_instructions)
from goldfive.prompt_shaper import PromptShaper
resolver = PromptShaper().make_dynamic_instruction(
    original_instruction=original,
    agent_name=agent_name,
)
cur.instruction = resolver
```

### 11.3 The installer's per-node decision table

`install_dynamic_instructions` walks `sub_agents` / `inner_agent` /
`AgentTool.agent`, pushing children before the `LlmAgent` check so
containers still propagate. It duck-types an `LlmAgent` via
`_looks_like_llm_agent(node)` (`hasattr(node, "instruction")` —
`SequentialAgent`/`ParallelAgent` lack the field). For each `LlmAgent`:

| Existing `instruction` | Action |
|---|---|
| already a dynamic resolver (`is_dynamic_instruction` true) | **skip** — idempotent re-wrap. |
| any callable (user-supplied `InstructionProvider`) | **leave alone** — do not double-wrap the caller's own resolver. |
| a string with **no** `{` | install resolver (common case). |
| a string with `{` **and** `_adk_inject_session_state()` is `None` | **skip with WARNING** — keep the static string so ADK's native templating still works. |
| a string with `{` and inject helper present | install resolver (the resolver re-applies templating — see §11.5). |

Assignment failures (frozen pydantic) are logged at DEBUG and skipped so
one bad agent doesn't block the tree. Returns the count of agents touched.
`log_dynamic_instruction_opt_out(root_agent)` walks the same edges without
mutating, emitting one INFO per reachable `LlmAgent`, so the operator log
shows which agents run static when the caller passed
`dynamic_instruction=False`.

The `callable` branch is critical: a user who supplied their own
`InstructionProvider` is already managing dynamic resolution; wrapping it
would double-resolve and corrupt their prompt.

### 11.4 The `observation_only` gate (invariant 5)

Inside the resolver closure (`_resolve` in `prompt_shaper.py`), the FIRST
thing after reaching the `SessionContext` is the kill-switch check:

```python
# goldfive/prompt_shaper.py (make_dynamic_instruction._resolve)
ctx = _goldfive_session_context_from_readonly_context(readonly_ctx)
steerer = getattr(ctx, "steerer", None) if ctx is not None else None
if not shaper.should_inject(steerer):
    log.info(... "observation_only=True — SKIPPING goldfive prompt augmentation" ...)
    return base_instruction
```

`shaper.should_inject(steerer)` is the resolver's only sanctioned read of
the kill-switch — it resolves through `is_active_steering()` /
`steering_is_active(steerer)` (missing/None/raising → passive). Under
`observation_only=True` the resolver returns the **templated but
un-augmented** `base_instruction` — no "Current assigned task" block, no
correction block. State templating still runs (ADK applies it to string
instructions regardless of goldfive; suppressing *that* would itself be a
behaviour change). Do not add any other kill-switch read here; do not gate
on a raw `observation_only` config field — go through the steerer
predicate. `test_observation_only_strict_passive.py` guards this.

### 11.5 The #477 fix — preserving ADK `{var}` templating

**This is the fix you must not break.** ADK's
`LlmAgent.canonical_instruction` marks a **callable** instruction
`bypass_state_injection=True`. So the moment goldfive swaps a *string*
instruction for a *callable* resolver, ADK stops applying its documented
`{var}` / `{artifact.var}` session-state substitution — wrapping would
silently disable the caller's templating. #477 (`77afb44`, "preserve ADK
`{var}` session-state templating under dynamic_instruction") fixes this in
the outer `resolver` closure with a **three-way branch**:

```python
# goldfive/prompt_shaper.py (make_dynamic_instruction.resolver)
def resolver(readonly_ctx: Any) -> str | Awaitable[str]:
    # 1. Placeholder-free fast path: byte-identical, synchronous.
    if "{" not in original_instruction:
        return _resolve(readonly_ctx, original_instruction)

    from goldfive.adapters.adk_llm_instrumentation import _adk_inject_session_state
    inject = _adk_inject_session_state()

    # 2. Templated but inject helper absent: degrade to the literal template.
    if inject is None:
        return _resolve(readonly_ctx, original_instruction)

    # 3. Templated + helper present: re-apply ADK templating, THEN augment.
    async def _inject_then_resolve() -> str:
        base = await inject(original_instruction, readonly_ctx)
        return _resolve(readonly_ctx, base)
    return _inject_then_resolve()
```

The three paths, spelled out:

1. **Placeholder-free fast path** (`"{" not in original_instruction`):
   returns `_resolve(...)` **synchronously**. Byte-identical to the
   pre-#477 behaviour. This is a hot path — most instructions have no
   `{`, and keeping it synchronous avoids an awaitable allocation and an
   ADK inject round-trip every turn.
2. **Awaitable path** (`{` present, `inject` available): returns the
   `_inject_then_resolve()` coroutine, which first runs ADK's own
   `inject_session_state` over the *original* template (so `{var}`
   substitution and its error behaviour match unwrapped ADK exactly —
   substitution errors for missing state vars/artifacts propagate the same
   way), then feeds the substituted result into `_resolve` as the base
   instruction for goldfive augmentation.
3. **Skip-with-WARNING fallback** (`{` present, `inject` is `None`): the
   installer already refuses to wrap this case (§11.3), but a resolver
   built directly degrades to `_resolve(readonly_ctx, original_instruction)`
   — the literal template, un-substituted.

Both branch 1 and branch 3 return a `str`; branch 2 returns an
`Awaitable[str]`. Both satisfy ADK's `InstructionProvider` alias
(`str | Awaitable[str]`).

`_adk_inject_session_state()` probes `google.adk.utils.instructions_utils.
inject_session_state` **lazily** (never at import, so the module stays
importable without google-adk) and **defensively** (an ADK release that
moves the helper degrades to `None` rather than raising).

### 11.6 The resolver's read path (Phase 2.0, goldfive#271)

Inside `_resolve` (when active-steering), the resolver:

1. Reaches the goldfive `SessionContext` via
   `_goldfive_session_context_from_readonly_context(readonly_ctx)` —
   which walks
   `readonly_ctx._invocation_context.plugin_manager.plugins` for the
   goldfive plugin and reads its `_active_ctx` (live-run path), falling
   back to the legacy `"goldfive._session_context"` state stash (unit
   tests only).
2. Reads the current-task pin via `StateStore.for_session(session).pin_current_task()`.
   No pin → returns `base_instruction` un-augmented (pre-plan turn).
3. Looks up `(title, description)` in `Session.plan.tasks` (the typed
   `Task` is the source of truth — the resolver does NOT read
   de-normalised ADK-state keys in the production path;
   `_task_title_description_from_session` returns `("", "")` on a miss so
   placeholders still render).
4. Reads any pending correction for `(agent_name, current_task_id)` via
   `StateStore.get_correction` (`_read_pending_correction`).
5. Composes via `_compose_instruction` (`{original}` + "Current assigned
   task" block + correction block when non-empty).

**Any exception in `_resolve` degrades to `base_instruction`** — a
goldfive failure must never cost the agent its (templated) instruction,
because ADK would otherwise surface it as a mid-turn `InternalError`, the
worst possible failure mode.

Phase 2.0 eliminated the old bridge that wrote goldfive plan state onto
ADK `session.state` at callback time — that write raced with ADK's
optimistic-concurrency contract (goldfive#275). The resolver now reads the
goldfive `Session` directly.

### 11.7 The correction block — directive, not diagnostic

`format_correction_block(correction)` (in `adk_llm_instrumentation.py`)
renders a pending-correction dict into the LLM-visible block. It is keyed
by `pending_correction_key(agent_name, task_id)` (invariant 6 — a stable
`(agent, task)` identity, not an LLM-minted id, so one agent/task's
correction never leaks into another's prompt).

Design principle: **directive, not diagnostic**. It tells the LLM what to
do on the corrected task ("Focus only on the revised scope", "Do not
propagate the superseded content") — NOT what went wrong ("was broken",
"failed"). Problem-naming language is an attractor for LLM pattern-matching
failure modes (meta-commentary, apologies, retrying the wrong thing —
goldfive#250 / #252 / #253 / #259). The diagnostic data (drift kind,
reason, revision) stays in the dict for sinks/observability but is
deliberately NOT interpolated into the LLM-visible text. If you "improve"
this block by adding the failure reason, you are reintroducing a known
regression.

`_resolve_pending_correction` accepts three shapes forgivingly: a
`Mapping` (rendered via `format_correction_block`), a non-empty string
(treated as pre-rendered, back-compat), or anything else (→ empty string,
skipped by `_compose_instruction`).

### 11.8 Provenance stamps and idempotency

The factory stamps three attributes on the returned closure so tree-walk
idempotency checks and tests can recognise it without relying on `repr`:

```python
# goldfive/prompt_shaper.py (make_dynamic_instruction)
resolver._goldfive_dynamic_instruction = True
resolver._goldfive_agent_name = agent_name
resolver._goldfive_original_instruction = original_instruction
```

`is_dynamic_instruction(value)` (in `adk_llm_instrumentation.py`) checks
`getattr(value, "_goldfive_dynamic_instruction", False)`. Keep these
stamps — the installer's idempotency skip and several tests read them.

---

## 12. The honest parity table

Not every adapter can do everything. Several of goldfive's most powerful
features are **ADK-only** because they depend on ADK's plugin callback
surface and mutable `session.state`. State this honestly — do not promise
a Claude or Callable user a feature that only ADK has.

| Capability | ADK | Claude | Callable | Why |
|---|---|---|---|---|
| Reporting tools (`report_task_*`) | yes (FunctionTool) | yes (inline MCP + `PreToolUse` hook) | yes (forwarded specs) | All three route through `invoke_tool`. |
| `available_agents` | yes (tree walk) | yes (passthrough list) | yes (configured list) | Core protocol. |
| `available_agents_tree` | yes (real depth/parent) | yes (flat depth-0) | yes (flat depth-0) | Only ADK has a real tree. |
| `emit_reasoning` (protocol) | yes (uniform; unused by plugin) | yes | yes | ADK extracts reasoning in the plugin instead (§8). |
| **Overlay** (`invoke_passthrough`) | yes | no (per-task `invoke`) | no (per-task `invoke`) | Overlay needs the reconciler + plugin observation. |
| **Plugin callbacks** (before/after model, delegation observation, tool-loop, runaway-delegation cap) | yes | no | no | Depends on ADK's `BasePlugin` / `PluginManager`. |
| **Streaming** (`subscribe_adk_events` / `run_streamed` raw events) | yes | no | no | Only ADK exposes a raw inner-Runner event stream. |
| **`ContextEditor`** (request-side, goldfive#397) | yes (plugin `before_model_callback`) | no | no | Needs the request-side callback. |
| **`AgentConfig` structural ceilings** (`max_output_tokens` ratchet #256, `llm_call_timeout_ms`) | yes | no | no | Enforced in the plugin's `before_model_callback` / call watcher. |
| **Dynamic-instruction resolver** (#251, #477) | yes | no | no | Mutates `LlmAgent.instruction`; only ADK has that field. |
| **`GoldfivePlanner` auto-attach** | yes | no | no | Attaches to `LlmAgent.planner`. |
| Re-entry dedup (`ReentryKind`) | yes | no | no | Overlay-only concern. |

`auto_adapter` and `wrap` reflect this honestly: `llm_call_timeout_ms`,
`agent_max_output_tokens`, `context_editor`, and `plugins` are forwarded
only to `ADKAdapter` and **ignored** for the other shapes (there is no
analogous surface). `dynamic_instruction` is a no-op for non-ADK agents.

**When a user reports a missing feature on Claude/Callable:** the honest
answer is usually "that is ADK-only; here is why". Do not fake it by, e.g.,
adding a regex to a Claude message stream to simulate delegation
observation — that violates invariant 2 and does not actually work.

---

## 13. How to add an adapter — the contract checklist

The full worked example is in **16-recipes.md** and
`docs/guides/writing-an-agent-adapter.md`. Here is the load-bearing
checklist. Follow it in order.

1. **Implement the four Protocol members** (§1.1). `register_reporting_tools`,
   `invoke`, `emit_reasoning`, `available_agents`. The `@runtime_checkable`
   `Protocol` means a duck-typed class passes `isinstance` — you do NOT
   need to subclass anything.
2. **Store the full `ReportingToolSpec` list**, not a name→handler map.
   You need `spec.description`, `spec.parameters` (JSON Schema), and
   `spec.handler` to surface each tool to your framework.
3. **Route tool calls through `invoke_tool`** (`goldfive/adapters/_tool_invocation.py`),
   NOT `spec.handler` directly. `invoke_tool(specs, name, args, session,
   steerer)` picks up schema validation and the orchestration-state
   task-id fallback (goldfive#191); the handlers themselves own
   terminal-state idempotency and invalid-transition rejection
   (goldfive#201). It raises `KeyError` on an unknown tool name (mirroring
   a real SDK when the agent hallucinates a tool). This is what both the
   ADK `before_tool_callback` and the Claude `PreToolUse` hook do.
   Skipping it means terminal tasks can be re-transitioned and benign
   retries look like loops.
4. **Return cleanly OR call a terminal tool — not both, not neither.** The
   executor auto-completes a still-`PENDING`/`RUNNING` task when `invoke`
   returns. If the agent already called `report_task_completed`/`_failed`/
   `_blocked`/`_cancelled`, the auto-complete is a no-op. If it moved the
   task to a non-terminal state without a terminal call, the executor
   re-invokes until a cap trips.
5. **`invoke` must not crash on well-formed input.** Catch framework
   exceptions and return `InvocationResult(..., error=exc)` so the
   executor surfaces a `TaskFailed`. (ADK: `_invoke_internal` does this;
   Claude: `invoke` wraps its receive loop in `try/except` and returns
   `error=`.)
6. **`emit_reasoning` must be a safe no-op when unbound** — copy the §3.2
   defensive shape verbatim (`getattr` chain + `TypeError` fallback).
7. **`available_agents` must match what planners emit.** Enumerate every
   routable leaf. Empty → planners assume no routable agents.
8. **Add `bind_steerer(steerer)`** so the Runner can wire the live steerer
   after construction. Support `bind_steerer(None)` to unbind.
9. **Optional extensions** — add `available_agents_tree` (even a flat
   depth-0 shape helps tree-aware planners), and `invoke_passthrough` only
   if your framework can genuinely run the tree naturally while you
   observe from the outside.
10. **Register it in `auto_adapter`** if you want `goldfive.wrap` to
    auto-detect it — add a `_looks_like_*` duck-type check and a lazy-
    import branch. Otherwise document that users pass the adapter directly
    to `goldfive.wrap(adapter, ...)`.
11. **Keep optional-SDK imports lazy** (see `claude.py`'s `_require_sdk`)
    so `import goldfive.adapters.<yours>` never hard-fails on a base
    install.

---

## 14. Common mistakes

Concrete wrong edits a weaker model would plausibly make here, each with
the correct alternative.

### 14.1 Assuming the Claude adapter has plugin callbacks

**Wrong:** wiring a `before_model_callback`-style hook, `ContextEditor`,
or `max_output_tokens` ratchet onto `ClaudeAgentSDKAdapter`, or telling a
user that `goldfive.wrap(claude_factory, runtime=RuntimeConfig(agent=AgentConfig(max_output_tokens=8000)))`
will cap Claude's output.

**Right:** those are **ADK-only** (§12). `auto_adapter` ignores
`agent_max_output_tokens` / `context_editor` for the Claude shape. The
Claude adapter's only interception seam is the `PreToolUse` hook for
reporting tools. If you need request-side mutation on Claude, you would
have to add it through the SDK's own options (`ClaudeAgentOptions`), not
goldfive's plugin surface.

### 14.2 Wrapping an already-callable instruction

**Wrong:** in `install_dynamic_instructions`, "simplifying" the per-node
branch to install a resolver on any non-resolver instruction, dropping the
`if callable(existing): leave it alone` branch.

**Right:** a callable `instruction` is a user-supplied
`InstructionProvider` that is already doing its own dynamic resolution.
Double-wrapping it corrupts their prompt. Keep the branch order exactly:
already-resolver → skip; callable → leave alone; string → install (or
skip-with-WARNING when templated + inject helper absent). `test_dynamic_instruction.py`
covers this.

### 14.3 Breaking the byte-identical placeholder-free fast path

**Wrong:** making the resolver always return the awaitable
`_inject_then_resolve()` "for consistency", or moving the
`_adk_inject_session_state()` probe above the `"{" not in original_instruction`
check.

**Right:** the placeholder-free branch MUST stay synchronous and MUST NOT
touch ADK's inject helper (§11.5). Making it async allocates a coroutine
and does an ADK round-trip on every turn for the common case, and changes
observable timing/behaviour. The `"{" not in original_instruction` check
is the fast-path guard — keep it first. `test_prompt_shaper.py` and
`test_determinism.py` assert the byte-identical shape.

### 14.4 Reverting the per-session state dicts to bare attributes

**Wrong:** "cleaning up" `_next_cancel_reasons` /
`_pending_tool_call_ids_by_session` / `_inflight_invoke_tasks` back to
single instance attributes because "the legacy property already exists".

**Right:** the per-session dicts prevent cross-session leakage when one
adapter drives concurrent goldfive sessions (§5.4). The legacy properties
are shims over the `""` bucket for single-session tests; they are NOT the
storage. `test_adk_adapter_concurrent_sessions.py` and
`test_adk_adapter_pending_tool_isolation.py` cover this.

### 14.5 Calling `spec.handler` directly instead of `invoke_tool`

**Wrong:** in a new adapter's tool dispatch, `await spec.handler(args,
session, steerer)`.

**Right:** `await invoke_tool(specs, name, args, session, steerer)`. Only
`invoke_tool` applies schema validation and the orchestration-state
task-id fallback. Bypassing it means a tool call with a missing/unknown
`task_id` hits the handler unguarded. Both the ADK plugin and the Claude
hook route through `invoke_tool`.

### 14.6 Adding a member to the `AgentAdapter` Protocol

**Wrong:** putting `available_agents_tree` (or your new capability) into
the `Protocol` body "so it's part of the contract".

**Right:** optional capabilities go on concrete adapters and are looked up
with `getattr` at the call site (§1.2). Adding to the `Protocol` breaks
`isinstance(x, AgentAdapter)` for every third-party adapter — the exact
mistake the `available_agents_tree` NOTE in `protocols.py` warns against.

### 14.7 Expecting `ADKAdapter.emit_reasoning` to drive ADK reasoning-drift

**Wrong:** editing `ADKAdapter.emit_reasoning` to fix a reasoning-drift
issue on the ADK backend.

**Right:** the ADK path extracts reasoning in the plugin's
`after_model_callback` and calls `steerer.drift.observe_reasoning`
directly — it never calls `adapter.emit_reasoning` (§8). Fix it in
`_adk_plugin.py` (see 05-adk-plugin.md). `ADKAdapter.emit_reasoning` is
protocol-uniform surface for manual/external callers.

### 14.8 Reintroducing the jargon-heavy per-task message

**Wrong:** "improving" `invoke_passthrough` to send a `"Task: X. Use the
goldfive.* session-state keys..."` framing so the agent "knows what to
do".

**Right:** that shape (the retired `_new_message_parts`) makes
flow-prompted coordinators re-run their full pipeline per task
(goldfive#141). The overlay sends the operator's input verbatim
(`_passthrough_message_parts`) and observes from the outside — this is
invariant 1 (no prompt-cooperation). Follow-up nudges, when used at all,
use the gentle `"Also, please: ..."` phrasing.

### 14.9 Removing the `reentry(...)` wrapper from `invoke_passthrough`

**Wrong:** dropping `with reentry(ReentryKind.OVERLAY_REPLAY):` because it
"looks like a no-op".

**Right:** it pins the contextvar that lets telemetry plugins suppress
duplicate user-message envelopes (harmonograf#234, §7.2). Removing it
silently doubles `UserMessageReceived` emission in downstream sinks.
`test_adk_reentry.py` covers the nesting semantics.

### 14.10 Downgrading the wrap-time integrity `RuntimeError`s

**Wrong:** turning the "plugin failed to install" or "reporting-tool set
did not land on N reachable agents" `RuntimeError`s into warnings so a run
can proceed.

**Right:** both mean silent orchestration failure (no observation, or a
sub-agent that cannot report terminal status through an `AgentTool`). A
loud crash at wrap time is correct — that is where the problem is
diagnosable. Keep them as raises.

### 14.11 Constructing `ADKAdapter` directly and expecting refine to land

**Wrong:** in a test, `ADKAdapter(agent)` then expecting the dynamic-
instruction resolver to be active.

**Right:** `install_dynamic_instructions` runs in `wrap()`, not the
constructor (§2.2). Go through `goldfive.wrap(agent, ...)` or call
`install_dynamic_instructions(agent)` yourself before constructing the
adapter.

---

## 15. Verification checklist

Run these after touching this subsystem. Commands assume repo root
`/home/sunil/git/goldfive` and the dev+adk extras installed
(`uv sync --extra dev --extra adk`).

### 15.1 Targeted test files

```bash
# Core adapter surface + protocol conformance
uv run pytest -q tests/test_callable_adapter.py \
                 tests/test_claude_adapter.py \
                 tests/test_adk_adapter.py \
                 tests/test_wrap_adk.py

# Overlay path (invoke_passthrough / reconciler / NOT_NEEDED marking)
uv run pytest -q tests/test_adk_adapter_overlay.py \
                 tests/test_adk_wrap_passthrough.py

# Re-entry / duplicate-suppression contract
uv run pytest -q tests/test_adk_reentry.py

# Concurrent-session isolation (the per-session dicts)
uv run pytest -q tests/test_adk_adapter_concurrent_sessions.py \
                 tests/test_adk_adapter_pending_tool_isolation.py

# Dynamic-instruction resolver + #477 templating + observation_only gate
uv run pytest -q tests/test_dynamic_instruction.py \
                 tests/test_prompt_shaper.py \
                 tests/test_observation_only_strict_passive.py \
                 tests/test_reasoning_content_fallback.py

# Runtime-config threading (AgentConfig ceilings reach the adapter)
uv run pytest -q tests/test_wrap_runtime_config.py

# Determinism (asserts the byte-identical placeholder-free fast path)
uv run pytest -q tests/test_determinism.py
```

### 15.2 Grep audits

```bash
# The four tree-walks must follow the SAME three edges. If you added a
# walk, confirm it includes inner_agent + AgentTool.agent, not just sub_agents.
grep -n "sub_agents\|inner_agent\|getattr(t, \"agent\"\|getattr(tool, \"agent\"" goldfive/adapters/adk.py

# No adapter may call spec.handler directly — all tool dispatch goes
# through invoke_tool. This grep should return ZERO hits in adapters/
# outside _tool_invocation.py itself.
grep -rn "\.handler(" goldfive/adapters/ | grep -v "invoke_tool\|_tool_invocation"

# The resolver's only kill-switch read is should_inject / is_active_steering.
# A raw observation_only field read in the resolver is a bug.
grep -n "observation_only\|should_inject\|is_active_steering" goldfive/prompt_shaper.py

# emit_reasoning must NOT be called from the ADK plugin (ADK routes
# reasoning via observe_reasoning directly). Expect zero hits.
grep -n "emit_reasoning" goldfive/adapters/_adk_plugin.py

# The #477 fast-path guard must be the first branch in resolver().
grep -n '"{" not in original_instruction\|_inject_then_resolve\|_adk_inject_session_state' goldfive/prompt_shaper.py
```

### 15.3 Lint and full suite

```bash
# Must stay clean. Do NOT ruff-format the repo (it is not format-clean).
ruff check .

# Full suite: ~30s, expect ~2912 passed / ~61 skipped on main.
uv run pytest -q
```

### 15.4 Manual sanity for a new adapter

If you added or changed an adapter, confirm it round-trips through the
runtime `isinstance` check and exposes a consistent tree shape:

```bash
uv run python -c "
import asyncio
from goldfive.protocols import AgentAdapter
from goldfive import CallableAdapter, InvocationResult
async def agent(task, session, tools):
    return InvocationResult(task_id=task.id, text='ok')
a = CallableAdapter(agent, available_agents=['worker'])
assert isinstance(a, AgentAdapter), 'adapter fails runtime_checkable Protocol'
assert a.available_agents == ['worker']
assert a.available_agents_tree[0]['kind'] == 'Callable'
print('OK: adapter conforms')
"
```

---

## See also

- **05-adk-plugin.md** — the response-side plugin callbacks
  (`before_model_callback`, `after_model_callback`, `before_tool_callback`,
  delegation observation, tool-loop, runaway-delegation cap) and where ADK
  reasoning extraction actually happens.
- **04-executors-and-control.md** — the overlay `SequentialExecutor` that
  calls `invoke_passthrough`, and the `NOT_NEEDED` marking that replaced
  follow-up nudges.
- **10-planning-and-revision.md** — `GoldfivePlanner` / `BasePlanner` and
  how `available_agents_tree` feeds the tree-aware planner.
- **13-reporting-tools-and-approval.md** — `invoke_tool` and the three
  reporting-tool protection layers every adapter shares.
- **09-steering-ladder-and-gates.md** — `is_active_steering()` /
  `steering_is_active()`, the kill-switch the resolver's `should_inject`
  resolves through.
- **14-config-reference.md** — `AgentConfig`, `SteeringConfig`,
  `RuntimeConfig`, and the env vars (`GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS`)
  that tune the request-side ceilings.
- **16-recipes.md** — the full worked "write a new adapter" recipe.
