# 15. Testing Guide

## Read this chapter when...

- You are about to write or change a test in `tests/` and want the harness idioms that already exist (stubs, fixtures, canned LLMs) instead of reinventing them.
- You changed production code and need to prove it works without fooling yourself — especially when the change touches a *gate*, a *detector*, a *judge*, or the `observation_only` kill-switch.
- Your test passes but you suspect it is a no-op (a stub silently fell back to PASSIVE, or you asserted on a log line that will never fire).
- You need the exact pytest paths / grep commands to run after touching a subsystem (the per-subsystem chapters cross-reference this one for the commands).
- You are wiring an *intervention* feature and need to know why one test is never enough (you owe BOTH a passive-suppressed test AND an active-behavior test).

This chapter reconciles and extends `.agents/testing.md`. Where `.agents/testing.md` and the live suite disagree, **the live suite on `main` wins** — and one place they disagree is called out explicitly in [Mode discipline](#mode-discipline-post-488-the-single-most-important-rule) below.

## Files covered

| Path | What it is |
| --- | --- |
| `pyproject.toml` `[tool.pytest.ini_options]` | `asyncio_mode = "auto"`, `testpaths = ["tests"]` |
| `tests/conftest.py` | shared fixtures: state-audit autouse, per-domain env controllers, `stub_call_llm`, `session_factory`, `in_memory_runner`, `active_steering_config`, `make_active_steerer` |
| `tests/_pbsetup.py` | `ensure_pb_available()` — gate for tests that need the committed proto stubs |
| `goldfive/testkit/__init__.py` | public testkit exports (adversarial agents + `CannedCallLLM`) |
| `goldfive/testkit/adversarial.py` | `AdversarialAgentBase` + 7 concrete drift-provoking agents |
| `goldfive/testkit/canned_call_llm.py` | `CannedCallLLM` deterministic transcript replay |
| `goldfive/sinks/memory.py` | `InMemorySink` — the canonical event ground-truth fixture |
| `tests/test_goal_drift_classifier.py` | exemplar `_stub_call_llm`, `ListSink`, `StubPlanner`, background-judge draining |
| `tests/test_overlay_forward_progress.py` | exemplar `RecordingSink`, `StubSteerer`, `StubAdapter`, `OverlayStubAdapter` |
| `tests/test_adk_adapter.py` | exemplar scripted `BaseLlm` (`_ScriptedLLM`) real-ADK-runner tests |
| `tests/test_embedding_backend.py` | exemplar fake-clock via module-attribute shim (`_install_fake_clock`) |
| `tests/test_pause_deadline.py` | exemplar `asyncio.wait_for` around anything that could hang |
| `tests/test_stall_watchdog.py` | exemplar `_wait_for` poll loop + module-constant monkeypatch |
| `tests/test_optimization_manifest.py` | exemplar AST-based manifest-liveness test |
| `tests/test_observation_only_*.py` | the BOTH-modes intervention-gate test family |

## Invariants that bind you here

These are the CANON invariants that a *test* is most likely to violate. Memorize them before writing an assertion.

1. **`observation_only=True` is the shipped production default and the suite runs it (post-#488).** The autouse fixture that used to flip the whole corpus to active mode is **deleted**. A test that needs an intervention to fire must request active mode EXPLICITLY (fixture `active_steering_config` / `make_active_steerer`, or `SteeringConfig(observation_only=False)` inline). See [Mode discipline](#mode-discipline-post-488-the-single-most-important-rule).
2. **The only sanctioned read of the kill-switch is `DefaultSteerer.is_active_steering()` / the module helper `steering_is_active(steerer)`** (`goldfive/steerer.py`). Missing / `None` / raising ⇒ **PASSIVE**. A stub steerer that forgets `is_active_steering` silently disables every intervention it is wired into — your "active" test becomes a no-op. See [The `is_active_steering` trap](#the-is_active_steering-trap-your-most-likely-silent-no-op).
3. **No prompt-cooperation contracts.** A test must not assert that termination/observability only works *because* the agent called a goldfive tool. Drive an agent that never cooperates (`RefusingAgent`, a bare `CallableAdapter` that ignores the reporting tools) and assert the control path still fires.
4. **No regex/keyword NL classification.** Do not add a test that pins a regex heuristic for natural-language classification — those were retired (#166/#167). Exact-equality / hash matching of *structured* data (e.g. `(name, args_hash)` tool-loop keys) IS allowed and IS tested that way.
5. **Adaptive over predictive.** Test observed facts (events emitted, state transitions taken), not predicted agent behavior.
6. **Stable identity keys.** When you test a lifecycle gate, assert it keys on a stable identity (full agent path, task id) — never on an LLM-minted / churning id.

---

## The 30-second loop

```bash
uv sync --extra dev --extra adk      # one-time; installs proto stubs + ADK
uv run pytest -q                     # full suite, ~30s, ~2912 passed / 61 skipped
uv run ruff check .                  # MUST stay clean
```

DO NOT run `ruff format` or `ruff format --check` and "fix" what it reports. **The repo is not ruff-format-clean and must not be mass-reformatted** — a format sweep will blow up your diff and get the PR rejected. `.agents/testing.md` still tells you to run `ruff format --check`; that instruction is stale — ignore it. Lint (`ruff check .`) is the gate that matters.

Running subsets:

```bash
uv run pytest tests/test_runner.py                      # one file
uv run pytest tests/test_runner.py::test_x              # one test
uv run pytest -k drift                                  # by keyword
uv run pytest -q --lf                                   # last-failed only
uv run pytest tests/test_pause_deadline.py -x -q        # stop at first failure
```

### The suite at a glance

- ~2912 passed / 61 skipped, ~30s wall on a dev laptop with `--extra dev --extra adk`.
- One test file per component under `tests/`; ~127 `importorskip` guard points for optional extras.
- Proto stubs are committed (`goldfive/pb/goldfive/v1/*_pb2.py`), so tests needing them normally run; `tests/_pbsetup.ensure_pb_available()` is the guard for a stubs-missing worktree.
- The 61 skips are almost entirely optional-extra gates on a minimal install; **CI installs all extras, so nothing is skipped in CI.** A green local run with skips is NOT the same coverage CI runs — see [CI parity](#ci-parity).

### pytest-asyncio auto mode

`pyproject.toml` sets:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

Consequences you must follow:

- **`async def test_*` runs with no decorator.** DO NOT add `@pytest.mark.asyncio` — it is redundant in auto mode and inconsistent with the whole existing suite. A reviewer will flag it.
- Every `await` in a test runs on the event loop pytest-asyncio spins up per test. There is no shared loop across tests.
- Fixtures may be `async def` too; auto mode awaits them.

---

## Conventions: naming, placement, structure

- **One file per component.** `tests/test_<component>.py` (`test_planner.py`, `test_sequential_executor.py`), adapters as `tests/test_<adapter>_adapter.py`, sinks as `tests/test_<sink>.py`. A regression for a specific issue often gets its own file named for the behavior (`test_pause_deadline.py`, `test_iter11d_cancel_race.py`) — that is fine and preferred over burying a subtle regression in a 2000-line file.
- **Test function names are sentences.** `test_goldfive_steer_dispatch_suppressed_under_observation_only` — a reader knows the mode and the expected outcome without opening the body. Copy that density; avoid `test_1` / `test_it_works`.
- **Keep fixtures local unless shared.** The suite deliberately avoids shared mutable state across files. Put a one-off stub in the test module (as `ListSink`/`StubPlanner` do); only promote to `tests/conftest.py` when two+ files need it.
- **Docstring the WHY, especially for regressions.** The strong tests in this suite open with a docstring naming the issue number, the live-run reproduction, and the exact symptom a regression would reproduce (see `tests/test_observation_only_strict_passive.py`). That docstring is what saves the next debugger — but do not trust docstrings as ground truth (some are stale post-#488; see the mode-discipline warning).
- **Imports after a module-level `pytestmark`/`importorskip` carry `# noqa: E402`.** That is expected and lint-clean.

## ADK-specific test hazards

The ADK adapter/plugin tests are the highest-friction area. Rules that are not obvious:

1. **`pytest.importorskip("google.adk")` at module top** — these tests do not run on a `--extra dev`-only install, but DO run in CI (which installs `adk`). A silent local skip is not a pass.
2. **The state-ownership tripwire is autouse-on and it targets ADK-state writes.** A plugin callback that stashes goldfive bookkeeping in ADK `session.state` raises `StateOwnershipViolation` during the test. If your callback needs to persist something, it belongs in goldfive-owned state (`Session`), not ADK state. To drive a deliberate violation, wrap with `goldfive._state_audit.expect_violation(reason)`.
3. **`session.state` shallow-copy handoff is a known trap** (MEMORY: callback-context handoff). ADK/ADK-SDKs may hand a callback a shallow copy of `session.state`; a write on one side can be invisible on the callback side. Test the real handoff with a scripted `BaseLlm` run (not a simulated dict) — `test_reporting_tool_guards_fire_under_real_adk_runner` exists precisely because a dict simulation hid this bug.
4. **`goldfive.wrap` must accept any ADK tree shape**, including coordinator + `AgentTool` (MEMORY: wrap contract). When you test wrap, include the coordinator+AgentTool shape; do not assume a flat single agent.
5. **Concurrent sessions** must not cross-contaminate — `tests/test_adk_adapter_concurrent_sessions.py` drives two sessions and asserts isolation. If you touch per-session plugin state, add/extend that test.

## Mode discipline (post-#488): the single most important rule

Before #488, an autouse conftest fixture flipped the *implicit* `observation_only` default to `False` for the whole corpus, plus there was a module-global test hook (`_OBSERVATION_ONLY_DEFAULT` / `_resolve_observation_only_default`). PR #488 (`41094d1`, "one public predicate for the observation_only kill-switch; suite runs the shipped default") **DELETED both**. The suite now runs against the shipped strict-passive default. `git show 41094d1` touched ~90 test files to add explicit active-mode opt-ins.

> **Stale-docstring warning.** Some test docstrings still say "tests/conftest.py flips the *implicit* default to False for the legacy corpus" (e.g. the header of `tests/test_observation_only_nudge_gate.py`). That sentence is **wrong on current main** — the flip fixture is gone. The code (no autouse flip in `tests/conftest.py`) wins. Do not trust that docstring; verify against `tests/conftest.py`.

### What this means for you

1. **A bare `DefaultSteerer()` is PASSIVE.** `DefaultSteerer(steering_config=SteeringConfig())` has `observation_only=True`. Its `is_active_steering()` returns `False`. Nudges are not enqueued, `GOLDFIVE_STEER` control messages are not dispatched, plan mutations are not applied, the plugin cancel-flag is not written.

2. **To make an intervention fire, opt into active mode explicitly.** Three sanctioned ways:

   ```python
   # (a) the config directly, inline:
   steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))

   # (b) the conftest fixture that hands you the config:
   async def test_x(active_steering_config):
       steerer = DefaultSteerer(steering_config=active_steering_config)

   # (c) the factory fixture that builds an active DefaultSteerer:
   async def test_y(make_active_steerer):
       steerer = make_active_steerer(goldfive_steer_threshold="warning")
   ```

   `active_steering_config` and `make_active_steerer` are defined in `tests/conftest.py`:

   ```python
   # tests/conftest.py
   @pytest.fixture
   def active_steering_config() -> Any:
       config_mod = pytest.importorskip("goldfive.config")
       return config_mod.SteeringConfig(observation_only=False)

   @pytest.fixture
   def make_active_steerer() -> Callable[..., Any]:
       config_mod = pytest.importorskip("goldfive.config")
       steerer_mod = pytest.importorskip("goldfive.steerer")
       def _factory(**kwargs: Any) -> Any:
           kwargs.setdefault(
               "steering_config", config_mod.SteeringConfig(observation_only=False)
           )
           return steerer_mod.DefaultSteerer(**kwargs)
       return _factory
   ```

3. **Every intervention feature owes TWO tests.** This is not optional and is the pattern the whole `tests/test_observation_only_*.py` family enforces:

   | Test | Mode | Asserts |
   | --- | --- | --- |
   | passive-suppressed | `observation_only=True` | detection ran (drift emitted, refine still called), telemetry stamped (`PolicyApplied` outcome `"suppressed"`), **but the injection/mutation did NOT happen** (no control message enqueued, no plan swap, no nudge on `session.pending_nudges`, no cancel flag) |
   | active-behavior | `observation_only=False` | the injection/mutation DID happen |

   The canonical pair, verbatim from `tests/test_goldfive_drift_routing.py`:

   ```python
   # active: OFF_TOPIC WARNING dispatches a GOLDFIVE_STEER control message
   async def test_goldfive_off_topic_drift_dispatches_goldfive_steer_control() -> None:
       steerer = DefaultSteerer(
           goldfive_steer_threshold="warning",
           goldfive_steer_suppression_window_turns=3,
           steering_config=SteeringConfig(observation_only=False),
       )
       ...
       assert _drain_channel(channel)  # message reached the channel

   # passive: same drift, detection + refine still run, channel stays empty
   async def test_goldfive_steer_dispatch_suppressed_under_observation_only() -> None:
       steerer = DefaultSteerer(
           goldfive_steer_threshold="warning",
           goldfive_steer_suppression_window_turns=3,
           steering_config=SteeringConfig(observation_only=True),
       )
       ...
       await steerer.drift.handle_drift(drift, session)
       assert planner.refine_steer_calls, "refine_steer must still run passively"
       assert _drain_channel(channel) == []   # NOTHING enqueued
   ```

   The passive test is the load-bearing one. It proves the invariant "strict-passive means observe, never inject." If you only ship the active test, a future edit that leaks an injection into passive mode will not be caught.

### The strict-passive carve-outs you must respect

`observation_only=True` gates far more than control-message dispatch. `DefaultSteerer.is_active_steering()` (`goldfive/steerer.py:1339`) is the named gate for ALL of these injection points — when you test any of them, add the passive-suppressed case:

- plan mutation in `PlanReviser._apply_revision` (`set_session_plan`);
- `GOLDFIVE_STEER` enqueue in `DriftObserver._dispatch_goldfive_steer_control`;
- the plugin cancel flag in `DriftObserver.request_invocation_cancel` (this is what #476's LLM_CALL_TIMEOUT watcher writes);
- `session.pending_nudges` enqueues in `DriftObserver._dispatch_nudge` and the post-ABSORB handoff (#202);
- prompt-shape injections behind `PromptShaper.should_inject` (#271): the runner conversational-follow-up wrap, the ADK `before_model_callback` `system_instruction` injections, and the dynamic-instruction resolver. See `tests/test_observation_only_strict_passive.py` for all three.

The one carve-out that is NOT gated by `observation_only`: **`pause_escalate` / real `TERMINATE`** has its own carve-out semantics (#482); see `tests/test_observation_only_pause_escalate_carveout.py` and `tests/test_pause_deadline.py::test_terminate_row_stays_gated_under_observation_only`. Do not assume "passive means nothing ever aborts" — read those tests before touching the terminate path.

---

## The `is_active_steering` trap: your most likely silent no-op

The module helper is the fail-safe. From `goldfive/steerer.py`:

```python
def steering_is_active(steerer: Any) -> bool:
    predicate = getattr(steerer, "is_active_steering", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001
        return False
```

Executors, the ADK plugin, prompt shaping, and reporting acks all resolve the kill-switch through `steering_is_active(steerer)`. **They never read `_observation_only` directly.** The failure mode this creates in tests:

> You write a "this intervention fires" test. You hand the executor a hand-rolled `StubSteerer` that has `observe` / `transition` / `_handle_drift` but **no `is_active_steering`**. `steering_is_active` returns `False`. The executor treats the run as passive and skips the very injection you were trying to assert. Your test then asserts the injection *did not* happen — and passes — but for the wrong reason. Or worse, you assert it *did* happen, the test fails, and you "fix" it by loosening the assertion.

**Rule: any stub steerer used in an active-mode test MUST implement `is_active_steering` returning `True`.** The overlay-progress stub does exactly this — copy it:

```python
# tests/test_overlay_forward_progress.py
class StubSteerer:
    def is_active_steering(self) -> bool:
        # Explicit active mode: the executor's nudge replay and
        # abort-on-fatal-failure enforcement under test are suppressed
        # under the shipped observation-only default.
        return True
```

For a stub that must model BOTH modes (e.g. the LLM-call-timeout watcher's `_CapturingSteerer` in `tests/test_llm_call_timeout_watcher.py`), gate the predicate on a constructor flag:

```python
class _CapturingSteerer:
    def __init__(self, *, observation_only: bool = False) -> None:
        self._observation_only = observation_only
        ...
    def is_active_steering(self) -> bool:
        return not self._observation_only
```

Then drive the SAME stub in both modes and assert the cancel-flag write happens only in active mode — that is the #476 test pair in one file.

---

## The harness toolbox

Everything here removes non-determinism. **No test hits a real network or a real LLM. Ever.** If you find yourself wanting to, you are missing one of these tools.

### 1. `CallableAdapter` — the LLM-free agent

The canonical way to remove agent non-determinism. Your "agent" is a pure coroutine. From `.agents/testing.md`, still current:

```python
from goldfive import (
    CallableAdapter, InMemorySink, InvocationResult, Plan, ReportingToolSpec,
    Runner, SequentialExecutor, Session, StaticPlanner, Task, TaskEdge,
)

async def agent(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    return InvocationResult(task_id=task.id, text=task.title)

runner = Runner(
    agent=CallableAdapter(agent, available_agents=["worker"]),
    planner=StaticPlanner(plan),
    executor=SequentialExecutor(),
    sinks=[InMemorySink()],
)
outcome = await runner.run("go")
await runner.close()
```

Two constructor shapes exist — verify which you need against `goldfive/adapters/callable.py`:
- `CallableAdapter(agent_coro, available_agents=[...])` — a single coroutine handling every task (the `.agents/testing.md` shape above).
- `CallableAdapter(handlers={...}, available_agents=[...])` — a dict of per-agent handlers (the shape `in_memory_runner` uses in `tests/conftest.py`).

**Always `await runner.close()`** in a runner test. Buffered sinks (none in the default set, but `GRPCSink`) drop events on drop-floor without it and the test flakes.

### 2. The sink zoo — pick the smallest that answers your question

| Sink | Where | Use it when |
| --- | --- | --- |
| `InMemorySink` | `goldfive/sinks/memory.py` (production module, re-exported from `goldfive`) | you want the real ground-truth sink; `.events` is a live list in emit order |
| `ListSink` | inline in many tests (e.g. `tests/test_goal_drift_classifier.py`) | you want a 4-line local stub with `emit`/`close` and nothing else |
| `RecordingSink` | inline in `tests/test_overlay_forward_progress.py` | you want convenience accessors like `last_run_aborted_reason()` |

`InMemorySink` verbatim contract (`goldfive/sinks/memory.py`): `.events` is a property returning the live list; `emit` appends and **never raises**; `close` is a no-op and the list stays populated after close so you can inspect post-run.

The inline `ListSink` pattern — copy it when you don't want to import the production sink:

```python
class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []
    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)
    async def close(self) -> None:
        pass
```

**Assert on events, not on logs.** See [Common mistakes](#common-mistakes). The canonical event filter used across the suite:

```python
def _events_of_kind(sink: Any, kind: str) -> list[Any]:
    return [e for e in sink.events if e.WhichOneof("payload") == kind]

kinds = [e.WhichOneof("payload") for e in sink.events if hasattr(e, "DESCRIPTOR")]
assert "task_completed" in kinds
```

The `hasattr(e, "DESCRIPTOR")` guard duck-types across proto events and legacy dict envelopes from `goldfive.events.make_event`. Current `main` emits proto (PR #55); keep the guard if you assert across both.

### 3. `stub_call_llm` — the deterministic `call_llm` (conftest fixture)

Any component that takes a `call_llm` (planners, judges, goal-drift classifier) gets a pure-Python stub. There is a **conftest fixture** and an **inline pattern**; know both.

The fixture (`tests/conftest.py`), returns a factory:

```python
async def test_x(stub_call_llm):
    call_llm = stub_call_llm([
        {"tasks": [{"id": "t1", "title": "do it"}], "summary": ""},  # dict -> json.dumps
        "on_task",                                                   # str -> verbatim
    ])
    planner = LLMPlanner(call_llm=call_llm, model="stub")
    ...
```

Semantics baked into the fixture: dicts/lists are `json.dumps`-encoded (so string-expecting callers get a string, structured-expecting callers `json.loads` it); exhaustion raises `AssertionError("stub_call_llm exhausted; no more canned responses")` — **it does NOT silently return `""`**, because silent exhaustion masks "why is the judge always returning on_task?" bugs.

The inline pattern (`tests/test_goal_drift_classifier.py::_stub_call_llm`) adds two things the fixture lacks — it **records call args** and it **raises `Exception` entries** so you can inject synthetic plumbing failures the classifier must absorb:

```python
def _stub_call_llm(responses: list[Any]):
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []
    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, Exception):
            raise resp
        ...
    _call_llm.calls = calls  # type: ignore[attr-defined]  # recorded calls attached as an attribute
    return _call_llm
```

Use the inline recording form when you must assert on the exact `system`/`user` prompt the component sent to the judge (prompt-shape regressions). Use `CannedCallLLM` (below) when you want the same recording + reset/inspection API as a reusable object.

### 4. `CannedCallLLM` — recorded-transcript replay (testkit)

`goldfive/testkit/canned_call_llm.py`. Same `async (system, user, model) -> str` shape, but a real class with introspection: `.calls` (list of `CannedCallLLMCall(system, user, model)`), `.remaining`, `.call_count`, `.reset()`. Exhaustion raises `CannedCallLLMExhausted(call_index, transcript_length)` — again, never silent. Thread-safe (an in-process tuning loop drives judges from background tasks). Prefer this over an ad-hoc stub when a downstream (zicato) harness needs to replay a trace offline or when you want to assert `call_llm.calls[0].system == SOME_PROMPT`.

### 5. The adversarial agent catalog (testkit)

`goldfive/testkit/adversarial.py` synthesizes negative-class test data: agents that each trip one detector class deterministically. Each exposes `expected_drift_kinds` (a `ClassVar[tuple[DriftKind, ...]]`) so a harness can sanity-check without coupling to the concrete class.

| Agent | Provokes (`expected_drift_kinds`) | Knob |
| --- | --- | --- |
| `CleanAgent` | `()` — negative control, any drift = false positive | `canned_response` |
| `LoopingAgent` | `LOOPING_TOOL_CALL`, `LOOPING_REASONING` | `cycle_after_turns` |
| `HallucinatingAgent` | `CONFABULATION_RISK`, `OFF_TOPIC` | `fabricated_args` |
| `RefusingAgent` | `AGENT_REFUSAL`, `MODEL_REFUSAL` | `refusal_text` |
| `WanderingAgent` | `OFF_TOPIC`, `INTENT_DIVERGENCE` | `off_topic_after_turns`, `off_topic_subject` |
| `SlowAgent` | `LLM_CALL_TIMEOUT`, `TASK_TIMEOUT` | `delay_ms` (routes through `asyncio.sleep`) |
| `RunawayDelegationAgent` | `RUNAWAY_DELEGATION` | `target_count` |

Determinism is load-bearing: all randomness routes through `goldfive.runtime.seeded_random`; pin `goldfive.runtime.set_seed(...)` for byte-identical output. `expected_drift_kinds` is "kinds we were designed to provoke" — a real run may add incidental drifts (a slow agent also crossing a loop threshold) and those are NOT regressions.

The agents implement the bare `AgentAdapter` Protocol (no `google.adk` needed). To drive one through the production executor, wrap it with `as_callable`:

```python
from goldfive.testkit.adversarial import LoopingAgent, as_callable
from goldfive.adapters.callable import CallableAdapter

agent = LoopingAgent(cycle_after_turns=2)
adapter = CallableAdapter(as_callable(agent))
# then: await Runner(adapter, ...).run(...)
```

`CleanAgent` is your false-positive canary: a run against it that produces ANY drift means a detector threshold is too tight — assert `sink` has zero `drift_detected` events.

Full-run adversarial harness pattern (drive an adversarial agent through a real `Runner` and assert against `expected_drift_kinds` without coupling to the concrete class):

```python
from goldfive.runtime import set_seed
from goldfive.testkit.adversarial import LoopingAgent, as_callable
from goldfive.adapters.callable import CallableAdapter

async def test_looping_agent_trips_tool_loop_detector(make_active_steerer) -> None:
    set_seed(7)                                    # byte-identical run
    agent = LoopingAgent(cycle_after_turns=2)
    runner = Runner(
        agent=CallableAdapter(as_callable(agent)),
        planner=StaticPlanner(_single_task_plan()),
        executor=SequentialExecutor(),
        steerer=make_active_steerer(),
        sinks=[sink := InMemorySink()],
    )
    await runner.run("loop please"); await runner.close()

    fired = {e.drift_detected.kind for e in sink.events
             if e.WhichOneof("payload") == "drift_detected"}
    # assert a DESIGNED kind fired; do NOT assert set-equality (incidental
    # drifts like a crossed timeout are not regressions).
    assert set(agent.expected_drift_kinds) & fired
```

The `set` intersection (not equality) is the contract: `expected_drift_kinds` is "kinds we were designed to provoke," and an incidental extra drift is explicitly allowed. Asserting `== set(agent.expected_drift_kinds)` is over-asserting and will flake as detectors evolve. `agent.tool_calls` (on the base class) records the exact `(tool_name, args)` sequence for finer assertions when you need them.

### 6. The executor/steerer stub set (`test_overlay_forward_progress.py`)

When you test executor stages you usually don't want a real `DefaultSteerer`. The overlay-progress file is the reference set of minimal stubs — `StubSteerer`, `StubPlanner`, `StubAdapter`, `OverlayStubAdapter`, `RecordingSink`. Key points to copy exactly:

- `StubSteerer.is_active_steering()` returns `True` (active-mode stub — see the trap above).
- `StubSteerer.transition` must respect the **frozen Plan/Task** contract (#247): derive a new plan via `with_task_status` and swap it with `set_session_plan` inside `channel_processor_active()`. Do NOT mutate `task.status` in place on a plan you got from a real session — Plan/Task are frozen dataclasses.

  ```python
  from goldfive.types import channel_processor_active, set_session_plan, with_task_status
  with channel_processor_active():
      set_session_plan(session, with_task_status(session.plan, task_id, to))
  ```

- `OverlayStubAdapter.invoke_passthrough` records `passthrough_calls` and runs an optional `passthrough_effect` callback so you can simulate the real steerer's ABSORB-nudge mutation without a real steerer.

### 7. Scripted `BaseLlm` — real ADK runner, deterministic model

When the thing under test is the **ADK plugin lifecycle** (callbacks firing in the real ADK run loop), stubbing `call_llm` is not enough — you need a real `google.adk` agent driven by a deterministic model. Subclass `BaseLlm` and script `generate_content_async` turn-by-turn. From `tests/test_adk_adapter.py`:

```python
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

class _ScriptedLLM(BaseLlm):
    model: str = "fake-model"
    _step: int = 0
    async def generate_content_async(self, llm_request: Any, stream: bool = False):
        self._step += 1
        if self._step == 1:
            yield LlmResponse(content=genai_types.Content(role="model", parts=[
                genai_types.Part(function_call=genai_types.FunctionCall(
                    id="call_1", name="report_task_completed",
                    args={"task_id": "raccoon_research", "summary": "done"}))]))
        else:
            yield LlmResponse(
                content=genai_types.Content(role="model",
                    parts=[genai_types.Part(text="ok")]),
                turn_complete=True)
```

This is how you prove the real reporting-tool dispatch reaches your handler (not the ACK shim). It is a `google.adk` test, so it must `pytest.importorskip("google.adk")` (see below).

### 8. conftest fixtures you should reuse, not re-derive

| Fixture | Gives you |
| --- | --- |
| `session_factory` | fresh `Session(run_id=..., goals=..., plan=...)`; skips if `goldfive.types` not importable |
| `in_memory_runner` | a `Runner` wired with in-memory/no-op collaborators; accepts keyword overrides (`adapter=`, `planner=`, `steerer=`, `sinks=`) |
| `tmp_jsonl_path` | a `pathlib.Path` to `events.jsonl` inside pytest `tmp_path` (for `JSONLPersistenceSink` tests) |
| `active_steering_config` / `make_active_steerer` | active-mode opt-in (see mode discipline) |
| `no_state_audit` | disables the state-ownership tripwire for one test |
| the `goldfive_*_env` family | per-domain env controllers (see config-knob recipe) |

**Autouse fixtures you get for free (do not fight them):**

- `_state_audit_enabled` — sets `GOLDFIVE_STRICT_STATE_OWNERSHIP=1` so any ADK-state mutation from inside a goldfive callback raises `StateOwnershipViolation` (STATE-OWNERSHIP-CONTRACT.md §7). If your test *deliberately* drives a violation, wrap it with `goldfive._state_audit.expect_violation(reason)`; to turn the audit off entirely for a test, request the `no_state_audit` fixture.
- `_isolate_orchestration_store_registries` — clears the module-level `_ACTIVE_INVOCATION_TASKS` and `_CANCEL_REQUESTED_INVOCATIONS` dicts in `goldfive.state_store` before AND after every test. These are keyed by `session.id` (aliased to `run_id`) and many tests share `run_id="r1"`; without this a leaked cancel-pending entry flips the late-drift gate and produces a flaky CI-only failure in a *different* test. **Do not add a test that leaves those dicts dirty and relies on this fixture to hide it — but also do not assume they persist across tests.**

---

## importorskip conventions for optional extras

The suite must stay green on a minimal install. Anything needing an optional extra guards its import.

**ADK / anthropic / other importable module — top-of-file skip:**

```python
import pytest
pytest.importorskip("google.adk")
from goldfive.adapters.adk import ADKAdapter  # noqa: E402
```

**Proto stubs — use `_pbsetup`, not a bare importorskip.** Proto stubs are committed, so normally present, but a dirty worktree after `make clean` can drop them. Gate the whole module:

```python
from tests._pbsetup import ensure_pb_available
pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)
from goldfive.config import SteeringConfig  # noqa: E402
```

**Optional sink that is `None` when its extra is missing:**

```python
from goldfive.sinks import JSONLPersistenceSink
pytestmark = pytest.mark.skipif(
    JSONLPersistenceSink is None, reason="goldfive[proto] not installed"
)
```

The optional extras and what each gates (`pyproject.toml` `[project.optional-dependencies]`):

| Extra | Guards | importorskip target |
| --- | --- | --- |
| `adk` | `google-adk` — the ADK adapter/plugin tests | `pytest.importorskip("google.adk")` |
| `claude` | `anthropic` — the Claude adapter path | verify per test: `tests/test_claude_adapter.py` actually skips on `pytest.importorskip("claude_agent_sdk")` (a different package from the `anthropic` the extra pins — check the import the test needs, not the extra name) |
| `embedding` | `sentence-transformers` — embedding-backed drift tests | `pytest.importorskip("sentence_transformers")` |
| `proto` | `grpcio` + `mypy-protobuf` — gRPC sink + proto codegen tests | `ensure_pb_available()` / `GRPCSink is None` |
| `dev` | pytest, ruff, mypy, protobuf, grpcio-tools — always required | n/a (base) |

CI installs `dev + adk + proto`. It does NOT install `claude` or `embedding` by default — verify against the workflow before assuming a `claude`/`embedding` test runs in CI; if it must, it needs to be added to the CI extras or it silently skips there too.

Rules:
- Put `importorskip` / `pytestmark = skipif` at module top **before** the guarded imports; add `# noqa: E402` to the post-skip imports (imports-not-at-top is expected and lint-clean with the noqa).
- Never `try/except ImportError: pass` around a test import — a silent skip is invisible; `importorskip` prints the skip reason.
- CI installs `dev + adk + proto` extras, so these skips do NOT fire in CI. They exist for contributors on a minimal `--extra dev` install. Do not rely on a skip to hide a real failure.

---

## How to test each artifact type

These are recipes. Each says: what to stub, what to drive, what to assert, and the invariant to also cover.

### Copy-from index — the exemplar file per artifact

Before writing a new test, open the closest existing one and copy its stub shapes. Do not re-derive stubs from scratch — the suite's stub idioms carry hard-won invariants (frozen-plan transitions, `is_active_steering`, background-judge draining) that a fresh stub will silently omit.

| You are testing... | Copy from | Gives you |
| --- | --- | --- |
| a deterministic detector | `tests/test_tool_loops.py` | `args_hash` helpers, severity-cap assertions |
| an LLM judge | `tests/test_goal_drift_classifier.py` | `_stub_call_llm`, `_drain_background_judges`, `ListSink`, `StubPlanner` |
| judge scheduling | `tests/test_judge_scheduling_guards.py` | semaphore / coalescing / ledger drivers |
| a gate / routing | `tests/test_goldfive_drift_routing.py` | BOTH-modes pair, `_drain_channel` |
| observation-only carve-out | `tests/test_observation_only_strict_passive.py` | passive-suppression assertions |
| an executor stage | `tests/test_overlay_forward_progress.py` | `StubSteerer`(+`is_active_steering`), `OverlayStubAdapter`, `RecordingSink` |
| control / pause / deadline | `tests/test_pause_deadline.py` | `asyncio.wait_for` bounds, `_ControlCancelled` |
| an ADK callback | `tests/test_adk_adapter.py` | `_ScriptedLLM`, real-runner invoke |
| a proto event | `tests/test_control_proto.py`, `tests/test_events.py` | round-trip + enum-alignment |
| decision telemetry | `tests/test_decision_telemetry.py` | factory + emit-site pattern |
| a config knob | `tests/test_stall_watchdog.py`, `tests/test_runtime_config.py` | default / env / stash / liveness |
| a sink | `tests/test_sink_events.py` | dispatch-shape assertions |
| approval / human-in-loop | `tests/test_approval_flow.py` | `wait_for` + `unavailable`/`timeout` decisions |
| cancel / race | `tests/test_iter11d_cancel_race.py` | state-store gate branches |
| determinism | `tests/test_determinism.py` | `set_seed` byte-identical |

### A deterministic detector (e.g. a tool-loop / drift classifier)

- **Stub:** nothing LLM — detectors are pure over structured observations. Build the input state directly.
- **Drive:** feed the detector the sequence it keys on. For `LOOPING_TOOL_CALL` that is N `(name, args_hash)` records; `args_hash` is order-insensitive over kwargs (see `tests/test_tool_loops.py::test_args_hash_is_order_insensitive`).
- **Assert:** on the returned `DriftEvent` — `drift.kind`, `drift.severity`, `drift.detail`, and `drift.raw` structured fields.
- **Invariants to also cover:**
  - The precision cap (#484): a name-axis-only loop (same tool name, distinct args) is capped at **INFO** without exact-repeat corroboration (`>=2` identical `(name, args_hash)`). Assert `drift.severity == INFO` and `drift.raw.get("severity_capped_from")` is set. Knob: `name_axis_max_severity`. Exact-repeat (`test_exact_repeat_fires_warning`) asserts `drift.raw.get("mode") == "exact"` and `"tool_loop_exact" in drift.detail`.
  - **No regex NL heuristic** — if your "detector" classifies natural-language content, it must be an LLM judge (next recipe) or a structural/hash check. Do not add a keyword-list detector; #166/#167 retired those.
- **File to model:** `tests/test_tool_loops.py`, `tests/test_drift_classifiers.py`, `tests/test_drift_taxonomy.py`.

### An LLM judge (reasoning / goal-drift classifier)

- **Stub:** `call_llm` via the inline recording stub (`_stub_call_llm`) or `CannedCallLLM`. Feed the judge's verdict as a canned response.
- **Drive:** call the classifier (`classify_goal_drift`, the reasoning judge) or drive `steerer.drift.note_agent_turn` / `steerer.tasks.mark_task_*`.
- **CRITICAL — drain background judges before asserting.** Goal-drift and reasoning judges are fire-and-forget tasks on `DefaultSteerer._background_judges` (#251/#319/v22). After `await note_agent_turn(...)` the judge has NOT run yet. Drain first:

  ```python
  async def _drain_background_judges(steerer: DefaultSteerer) -> None:
      pending = list(steerer._background_judges)
      if not pending:
          return
      await asyncio.gather(*pending, return_exceptions=True)
      await asyncio.sleep(0)   # let add_done_callback discard from the set
  ```

  This helper is verbatim from `tests/test_goal_drift_classifier.py`. If you skip the drain, `sink.events` and `call_llm` call-count assertions see the pre-judge state and flake or falsely pass.
- **Assert:** the emitted `DriftEvent` (kind, CRITICAL severity for goal drift), OR that no drift fired when the judge says on-task, AND that the judge was called at most once per check.
- **Invariants to also cover:**
  - **Robustness to LLM failure** — inject an `Exception` in the transcript; the classifier must absorb it and return `None`, not crash the run. (`sink` exceptions never abort runs, #479.)
  - **Malformed severity → INFO** (#479): feed a judge response with an out-of-range/garbage severity and assert the drift lands at INFO, not a crash.
  - **Judge scheduling guards** (#483): per-steerer semaphore default 3 (`ReasoningDriftConfig.max_concurrent_judges`, `DefaultSteerer.drift._judge_semaphore._value`), queued-window coalescing (a QUEUED window coalesces onto the newest observation; a RUNNING call is never coalesced), and the verdict-utility ledger counting `{acted_on, emitted_late, emitted_redundant, parse_fail}`. Distinct `(agent, task)` keys must NEVER coalesce. See `tests/test_judge_scheduling_guards.py`.
- **File to model:** `tests/test_goal_drift_classifier.py`, `tests/test_drift_reasoning.py`, `tests/test_judge_scheduling_guards.py`, `tests/test_judge_empty_response_no_retry.py`.

### A gate (a suppression / routing / carve-out predicate)

- **Stub:** a `DefaultSteerer` (real one — gates live on it) in the mode under test, `ListSink`, and a stub planner recording `refine`/`refine_steer` calls.
- **Drive:** `await steerer.drift.handle_drift(drift, session)` with a hand-built `DriftEvent`.
- **Assert BOTH modes** (see mode discipline): passive ⇒ detection ran + `PolicyApplied` outcome `"suppressed"` + no injection; active ⇒ injection happened.
- **Invariants to also cover:**
  - Stable identity keys — assert the gate keys on a stable key (full agent path per #479 correction keys, task id), never on an LLM-minted id. To test: drive the same drift with a churning id field and assert the gate still engages (does not open a fresh entry per observation).
  - Staleness / lifecycle: `DRIFT_LIFECYCLE_RESOLVED` emits at task-terminal transitions (#486); `GOAL_DRIFT` resolves ONLY at task-terminal. `TERMINAL_TASK_STATUSES` is the canonical set (#485) incl. `NOT_NEEDED` — use it, do not hand-list statuses.
- **File to model:** `tests/test_goldfive_drift_routing.py`, `tests/test_drift_lifecycle.py`, `tests/test_drift_resolution_wiring.py`, `tests/test_observation_only_*.py`.

### An executor stage

- **Stub:** the overlay-progress stub set (`StubSteerer` with `is_active_steering() -> True`, `StubPlanner`, `StubAdapter`/`OverlayStubAdapter`, `RecordingSink`).
- **Drive:** construct the executor (`SequentialExecutor` / `ParallelExecutor`), a `Session` with a `Plan`, and `await executor.run(...)` or the specific stage method.
- **Assert:** on transitions recorded by the stub and on sink events (`RecordingSink.last_run_aborted_reason()` for abort tests).
- **Invariants to also cover:**
  - Frozen Plan/Task (#247) — never mutate `task.status` in place; use `with_task_status` + `set_session_plan` under `channel_processor_active()`.
  - Terminal-task skipping (#485) — the parallel scheduler skips terminal tasks incl. `NOT_NEEDED`; assert a `NOT_NEEDED` task is never invoked.
  - `fail_fast` respects live replacements (#202) — a FAILED task with a live `retry_<id>`/`<id>_v2` replacement does NOT abort; assert via `_has_live_replacement(plan, failed) is True`.
  - Abort path uses `Runner._abort_turn` (#489, 8 former copy-paste sites) — if you touch abort, assert the `run_aborted` event carries the escalation lineage (#482).
- **File to model:** `tests/test_overlay_forward_progress.py`, `tests/test_sequential_executor_overlay.py`, `tests/test_executor_control.py`, `tests/test_pause_deadline.py`.

### A plugin callback (ADK)

- **Stub:** a real `google.adk` `Agent` with a `_ScriptedLLM` (scripted `BaseLlm`), plus a minimal `_StubSteerer` (implement `observe`/`transition`/`detect_drift`/`bind` — and `is_active_steering` if the callback path you exercise reads the kill-switch).
- **Drive:** `ADKAdapter(agent)`, `register_reporting_tools([...])`, `bind_steerer(...)`, then `await adapter.invoke(task=..., session=...)` — this runs the *actual* ADK runner so callbacks fire for real.
- **Assert:** on the side effects the callback produces — your handler was called with the exact args, task status flipped, the drift was observed. `tests/test_adk_adapter.py::test_reporting_tool_guards_fire_under_real_adk_runner` asserts `handler_calls == [{...}]` with a long failure message explaining that empty `handler_calls` means dispatch fell through to the ACK shim (bypassing every guard); `test_reporting_tool_guards_fire_across_back_to_back_invocations` proves the plugin-instance handoff resets between invocations (the filler-loop outage).
- **Invariants to also cover:**
  - **State-ownership tripwire is autouse-on** — a callback that writes ADK `session.state` for goldfive bookkeeping raises `StateOwnershipViolation`. If that's intentional bookkeeping, it belongs in goldfive-owned state, not ADK state.
  - `dynamic_instruction` must preserve ADK `{var}` templating via `inject_session_state` (#477) — assert the resolved instruction still contains the template resolution, not a mangled literal.
  - Under `observation_only=True` the plugin's `before_model_callback` `system_instruction` injections and the dynamic-instruction resolver are SKIPPED (#271) — the passive test asserts the coordinator's `system_instruction` is verbatim what the caller set.
- **File to model:** `tests/test_adk_adapter.py`, `tests/test_adk_plugin_tool_observations.py`, `tests/test_before_tool_task_id_injection.py`, `tests/test_dynamic_instruction.py`.

### A proto event round-trip

- **Stub:** none. Build the event via its `goldfive.events` factory.
- **Drive/Assert:** serialize + parse, then assert the oneof and fields survive:

  ```python
  pb_msg = some_event(...)
  assert pb_msg.WhichOneof("payload") == "policy_applied"
  # round-trip through the wire:
  raw = pb_msg.SerializeToString()
  clone = type(pb_msg)()
  clone.ParseFromString(raw)
  assert clone.WhichOneof("payload") == pb_msg.WhichOneof("payload")
  ```

- **Invariants to also cover:**
  - Enum alignment — a proto enum and its Python `DriftKind`/`ControlKind` twin must stay aligned; `tests/test_control_proto.py::test_control_kind_enum_alignment` pins this. When you add an enum value, add the alignment assertion.
  - New payloads must be in the `Event.payload` oneof — `tests/test_decision_telemetry.py::test_new_events_are_in_event_payload_oneof` asserts each new factory's `WhichOneof("payload")`.
  - Decision-telemetry field names (#480): `DriftEvent.detector_name`, `PolicyApplied.outcome` (`drift_dropped_stale` / `drift_dropped_inflight` / `suppressed`), `ReasoningJudgeInvoked` proto fields 12–15 (`focused_task_id`, `focus_confidence`, `stated_intent`, `provenance`). Assert on these exact names — a rename is a wire break.
  - Event id format + global uniqueness — `tests/test_event_id_format.py`, `tests/test_event_id_globally_unique.py`. Never key a lifecycle gate on these ids (they churn).
- **File to model:** `tests/test_control_proto.py`, `tests/test_events.py`, `tests/test_decision_telemetry.py`, `tests/test_event_sequence.py`.

### A config knob (+ env override + manifest liveness)

A knob has FOUR things to test. Do all four.

1. **Default:** `assert SteeringConfig().stall_watchdog_enabled is False` (`tests/test_stall_watchdog.py::test_steering_config_defaults_watchdog_off`).
2. **Env override:** use the per-domain env-controller fixture — never poke `os.environ` directly. The controller maps a short key to the full env-var name and refuses unknown keys:

   ```python
   def test_env_override(goldfive_steer_env):
       goldfive_steer_env.set(observation_only="1")   # -> GOLDFIVE_STEER_OBSERVATION_ONLY=1
       cfg = SteeringConfig.from_env()
       assert cfg.observation_only is True
   ```

   The controllers and the short-key → env-var mapping they own (from the `_*_ENV` dicts in `tests/conftest.py`) — the fixture owns this mapping; callers pass the short key, never the raw string:

   | Fixture | Short keys → env vars |
   | --- | --- |
   | `goldfive_agent_env` | `max_output_tokens`→`GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS`, `call_timeout_ms`→`GOLDFIVE_AGENT_CALL_TIMEOUT_MS` |
   | `goldfive_embedding_env` | `base_url`, `model`, `api_key`, `timeout_ms`, `breaker_cooldown_s` → `GOLDFIVE_EMBEDDING_*` |
   | `goldfive_steer_env` | `observation_only`→`GOLDFIVE_STEER_OBSERVATION_ONLY`, `threshold`→`GOLDFIVE_STEER_THRESHOLD`, `suppression_window_turns`→`GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS` |
   | `goldfive_tool_loop_env` | `window`, `exact_threshold`, `name_threshold`, `alternating_threshold`, `name_axis_max_severity` → `GOLDFIVE_TOOL_LOOP_*` |
   | `goldfive_reasoning_drift_env` | `mode`, `off_topic_distance`, `intent_*_similarity`, `looping_similarity`, `cluster_similarity`, `looping_hash_window`, `max_concurrent_judges`, `fallback_to_content` → `GOLDFIVE_DRIFT_*` |
   | `goldfive_goal_drift_env` | `check_interval`→`GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL`, `activity_window`→`GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW` |
   | `goldfive_judge_env` | `base_url`, `model`, `api_key`, `timeout_ms` → `GOLDFIVE_JUDGE_*` |
   | `goldfive_fail_fast_env` | `revision_rejection`→`GOLDFIVE_FAIL_FAST_REVISION_REJECTION`, `invoke_cancel`→`GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` |
   | `goldfive_examples_env` | `topic`→`GOLDFIVE_EXAMPLE_TOPIC`, `openai_api_key`→`OPENAI_API_KEY`, `harmonograf_server`→`HARMONOGRAF_SERVER` |

   The aggregate `goldfive_runtime_env` bundles embedding/tool_loop/reasoning_drift/goal_drift/judge/steer/agent controllers into one dict for `RuntimeConfig.from_env()` tests. Each controller pre-clears its vars before yielding (no ambient leakage) and unwinds via monkeypatch. The controller REFUSES unknown short keys (raises `KeyError`) so a `goldfive_steer_env.set(threshld=...)` typo fails loudly instead of poking nothing. To test whitespace/case variants use `raw_setenv(FULL_NAME, value)` (only accepts names the controller owns). Never touch `os.environ` directly and never `monkeypatch.setenv("GOLDFIVE_...")` inline — route through the controller so the env surface a test consumes is visible from the fixture name.
3. **Steerer stashes it:** `assert DefaultSteerer(steering_config=SteeringConfig(stall_watchdog_enabled=True))._stall_watchdog_enabled is True` — verify the config value actually reaches the consumer, not just the config object.
4. **Manifest liveness** (if the knob is a numeric knob zicato can mutate): the AST test in `tests/test_optimization_manifest.py::test_numeric_mutations_have_live_runtime_consumers` statically counts *read* sites for the knob's leaf name across the `goldfive` package (excluding generated `pb/` stubs). **Zero reads = dead knob = test failure.** A constant that is merely defined + re-exported but read by no runtime code (the historical `GOAL_DRIFT_CHECK_INTERVAL`) fails this. If you add a manifest entry, its `python_attr` must point at the attribute a runtime consumer actually reads (e.g. `GoalDriftConfig.check_interval`, not the shadow module constant). The self-check `test_liveness_counter_flags_the_known_dead_constant` proves the heuristic still detects the known-dead constant — don't delete it.

- **File to model:** `tests/test_stall_watchdog.py`, `tests/test_runtime_config.py`, `tests/test_wrap_runtime_config.py`, `tests/test_optimization_manifest.py`.

### The intervention ladder — a parametrized truth table (protected keep)

The `(drift_kind, severity, occurrence_count) -> InterventionLevel` mapping is pinned as a parametrized table in `tests/test_intervention_ladder.py`. This is the canonical pattern for testing a pure routing function — a `list[tuple[...]]` of cases fed to `@pytest.mark.parametrize`:

```python
_CASES: list[tuple[DriftKind, DriftSeverity, int, InterventionLevel]] = [
    (DriftKind.CONFABULATION_RISK, DriftSeverity.INFO, 0, InterventionLevel.OBSERVE),
    (DriftKind.AGENT_REFUSAL, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (DriftKind.AGENT_REFUSAL, DriftSeverity.CRITICAL, 0, InterventionLevel.CANCEL_REINVOKE),
    # LOOPING_REASONING graduated severity (goldfive#204):
    (DriftKind.LOOPING_REASONING, DriftSeverity.WARNING, 0, InterventionLevel.ABSORB),
    (DriftKind.LOOPING_REASONING, DriftSeverity.CRITICAL, 0, InterventionLevel.NUDGE),  # NUDGE-first
    # ... CRITICAL repeat -> PAUSE_ESCALATE
]
```

`InterventionLevel` imports from `goldfive.steerer`. Two things a weak model must not get wrong here:

1. **`LOOPING_REASONING` CRITICAL routes to NUDGE-first, then PAUSE_ESCALATE on repeat** (#204/#206). This is a PROTECTED KEEP — the "obvious simplification" of routing loops straight to CANCEL is wrong and reverting it needs human sign-off. If you edit the ladder and this row flips, you broke it.
2. The `occurrence_count` axis matters — first CRITICAL and repeat CRITICAL route differently. Add cases for both counts, not just the first.

When you add a `DriftKind` or change a severity mapping, add its rows to `_CASES`. The table IS the spec; do not test the ladder with a single ad-hoc call.

### `wrap()` threads `RuntimeConfig` into every subsystem

`tests/test_wrap_runtime_config.py` proves a non-default `RuntimeConfig` passed to `goldfive.wrap(agent, runtime=cfg, sinks=[])` propagates into every subsystem (detectors, judges, steerer). It has both a from-config path (`test_wrap_threads_runtime_config`) and a from-env path (`test_wrap_threads_reasoning_drift_mode_from_env`, using the env-controller fixtures). Because `wrap()` can install process-level config, this file's tests **must leave the process env clean** — they use the `goldfive_runtime_env` bundle to pre-clear every sub-domain in setup. If you add a wrap-level config knob, add BOTH a from-config and a from-env test here, and make sure your test does not leak env state (route through the controller, never `os.environ`).

### A reporting tool + human approval (#478)

`report_awaiting_approval` is the 8th reporting tool. It must NEVER hang. The three states to test (`tests/test_approval_flow.py`):

1. **No control channel bound ⇒ immediate `"unavailable"` ack** — the handler cannot register a waiter, so it degrades instead of blocking:
   ```python
   result = await asyncio.wait_for(
       handler({"task_id": "t1", "prompt": "?"}, session, steerer), timeout=1.0)
   assert result["decision"] == "unavailable"
   ```
   The `asyncio.wait_for(..., timeout=1.0)` is the whole point: if the handler ever blocks, the test fails fast instead of hanging CI.
2. **Approve / reject resumes the waiter** — bind a control channel, spawn the handler as a task, push an `APPROVE`/`REJECT` `ControlMessage`, and assert the resumed `result["decision"]`.
3. **Timeout emits `HUMAN_INTERVENTION_REQUIRED`** — pass `timeout_ms=50`, assert `result["decision"] == "timeout"` AND exactly one `DriftKind.HUMAN_INTERVENTION_REQUIRED` drift at `WARNING` with `current_task_id == "t1"`. The default timeout is finite (600s) so a wedged approval self-heals.

Relatedly (#478), the F1 `plan_state` directive is stripped from reporting-tool acks under `observation_only=True` — a passive observer must not receive goldfive-shaped plan bookkeeping in the tool result. That is a separate BOTH-modes pair in `tests/test_observation_only_acks.py`: `test_directive_ack_omits_plan_state_under_observation_only` (`"plan_state" not in out`) vs `test_directive_ack_includes_plan_state_in_active_mode` (`out["plan_state"]["completed_task_ids"] == ["t1"]`).

### The verdict-utility ledger and its teardown flush (#483)

The per-session ledger lives at `DefaultSteerer.drift._verdict_ledgers[session.id]` and counts `{acted_on, emitted_late, emitted_redundant, parse_fail}` plus `elapsed_ms`. Assert on the dict directly:

```python
# tests/test_judge_scheduling_guards.py
ledger = steerer.drift._verdict_ledgers[session.id]
assert ledger["emitted_late"] == 1
assert ledger["acted_on"] == 0
```

The teardown summary event is emitted by `shutdown()`. After
`await steerer.drift.shutdown()` or the runner close path, assert
`steerer.drift._verdict_ledgers == {}`. A redundant verdict at the
`handle_drift` gates increments `emitted_redundant`; a parse failure
increments `parse_fail`.

The `wrap()`-time endpoint-contention warning has separate assertions.
`test_wrap_warning_names_shared_endpoint_cost` asserts that one WARNING
fires when the judge and agent share an endpoint. The warning-suppression
tests cover explicit `judge_call_llm=`, `call_llm=`, and `JudgeConfig`
routes. These sanctioned `caplog` assertions protect the warning
contract.

---

## More integration patterns

### Multi-turn runner / conversation

A `Runner` is reusable across turns (conversation carry-forward, #271 / intra-session plan carry). To test a follow-up turn, call `run` twice on the same runner without `close` between:

```python
runner = in_memory_runner(...)          # or build directly
first = await runner.run("do X")
second = await runner.run("now also Y") # follow-up: prior plan is in session
await runner.close()
```

Assert the second turn REUSED the prior plan (carry-forward) or refined it — inspect `sink.events` for `plan_revised` vs a fresh `plan_generated`. Under `observation_only=True` the conversational-follow-up wrap is skipped (#271), so a passive test must assert the executor received the raw `"now also Y"`, not a `[CONVERSATIONAL FOLLOW-UP ...]` directive — `tests/test_observation_only_strict_passive.py` shows the shape.

### Cancel / race window (state-store gate)

The late-drift gate (`_is_late_drift_for_terminated_invocation`) keys on the state-store registries `_ACTIVE_INVOCATION_TASKS` / `_CANCEL_REQUESTED_INVOCATIONS`. `tests/test_iter11d_cancel_race.py` (goldfive#242) tests the race window directly:

- `test_gate_skips_drift_when_cancel_pending_with_active_invocations` — cancel requested + active invocation present ⇒ the drift is NOT treated as late (the invocation is still live).
- `test_gate_still_classifies_empty_active_list_as_late` — cancel requested + empty active list ⇒ late (the invocation already tore down).

When you touch cancel plumbing, drive both branches. The autouse `_isolate_orchestration_store_registries` keeps these registries clean between tests — but if your test seeds them, seed them explicitly inside the test, do not rely on residue from a prior test. Wrap the cancelled run in `asyncio.wait_for` so a stuck cancel path fails fast.

### Determinism (`set_seed`) — the byte-identical contract

`tests/test_determinism.py` pins the contract `goldfive.runtime.set_seed(N)` advertises: two runs with the same seed and same input produce byte-identical `events.jsonl`. When you add anything that mints ids or randomizes ordering, route it through `goldfive.runtime.seeded_uuid4` / `seeded_random` and add a determinism assertion:

```python
from goldfive.runtime import set_seed, seeded_uuid4
set_seed(123); a = seeded_uuid4()
set_seed(123); b = seeded_uuid4()
assert a == b
```

Never call `uuid.uuid4()` or `random.random()` directly in production paths that feed the event stream — it breaks the determinism test and the testkit's byte-identical guarantee.

### A sink round-trip (JSONL persistence / delegation shape)

`tests/test_sink_events.py` drives a real run through a sink and asserts the *shape* of the emitted stream — e.g. a single-agent dispatch emits a started/completed pair, an AgentTool delegation produces a nested started/completed shape, and sub-runner events inherit the outer task id. For the JSONL persistence sink use the `tmp_jsonl_path` fixture, run, close, then read the file back and `json.loads` each line. goldfive's JSONL sink emits **camelCase** keys (the zicato/harmonograf wire contract) — assert on `camelCase` field names, not `snake_case`. `test_no_sink_events_when_steerer_is_unbound` pins that an unbound steerer emits nothing — a useful negative to copy.

### Control plane: pause / escalate / terminate (#482)

The escalation ladder terminates in a real pause-with-deadline. `tests/test_pause_deadline.py` is the reference and its assertions are worth copying exactly because the semantics are subtle:

- **Deadline expiry aborts the run.** Sequential and parallel executors have the SAME deadline semantics on their pre-stage wait — test both:
  ```python
  outcome = await asyncio.wait_for(run_task, timeout=5.0)
  assert "pause escalation deadline expired" in (outcome.reason or "")
  assert outcome.reason.startswith("run_aborted:pause_escalate_deadline:")
  ```
- **No deadline ⇒ the pause blocks indefinitely** (default, behavior-preserving): `assert not run_task.done()` after letting the loop spin, then release it with a test-side control message.
- **A deadline arriving mid-pause adopts and bounds the already-blocked wait** — `pytest.raises(_ControlCancelled, match="deadline expired")`. Import `_ControlCancelled` and `pause_deadline_s` from `goldfive.executors._control`.
- **TERMINATE always carries a built-in deadline** (`DEFAULT_TERMINATE_PAUSE_DEADLINE_S`, 600s): `assert pause_deadline_s(msgs[0]) == DEFAULT_TERMINATE_PAUSE_DEADLINE_S`. A configured `pause_escalate_deadline_s` overrides it.
- **Level-4 pause_escalate attaches the config deadline only when set** — default has no `deadline_s` in the payload (`"deadline_s" not in msgs[0].payload`, `pause_deadline_s(msgs[0]) is None`).
- **TERMINATE stays gated under `observation_only`** — `test_terminate_row_stays_gated_under_observation_only`. This is why "passive means nothing ever aborts" is a trap: the terminate ROW is gated, but the deadline machinery is real. Read the carve-out tests (`tests/test_observation_only_pause_escalate_carveout.py`, `tests/test_observation_only_abort_carveout.py`) before touching this path.

Every one of these wraps the run in `asyncio.wait_for(timeout=5.0)` so a genuine hang fails in 5s instead of stalling CI. The deadlines under test are tiny (1.5s) — never use a production-scale deadline in a test.

### Decision telemetry — factory vs emit-site tests

`tests/test_decision_telemetry.py` shows the two-layer pattern every telemetry event needs. The four events (events.proto tags 40–43) are `LadderTransitionDecided`, `DetectorDispatchOrdered`, `PolicyApplied`, `RetryBudgetSpent`, built by `ladder_transition_decided_event` / `detector_dispatch_ordered_event` / `policy_applied_event` / `retry_budget_spent_event` in `goldfive.events`.

1. **Factory test** — call the factory, assert basic fields + `WhichOneof("payload")`. Cheap, pins the wire shape. `test_new_events_are_in_event_payload_oneof` asserts each lands in the `Event.payload` oneof.
2. **Emit-site test** — drive the *production* path that emits it and assert it reaches the sink with the right `outcome`. The reference build helper:
   ```python
   def _build_steerer() -> tuple[DefaultSteerer, _ListSink, Session]:
       sink = _ListSink(); steerer = DefaultSteerer()
       steerer.bind(sinks=[sink], planner=_NullPlanner())
       return steerer, sink, Session(run_id="run-test")
   def _payloads(sink, kind): return [e for e in sink.events if e.WhichOneof("payload") == kind]
   ```
   `PolicyApplied` fires at three sites: refine-failure-threshold, refine-outcome-succeeded, and the observation-only gate — with `outcome` one of `"suppressed"`, `"drift_dropped_stale"`, `"drift_dropped_inflight"`, `"no_drift"`. Pre-seed a `RefineOutcome` in `session.refine_outcomes[(kind.value, task_id)]` to reach the threshold site. `RetryBudgetSpent` coerces a non-numeric remaining-budget to zero — assert that (`test_retry_budget_spent_factory_coerces_non_numeric_to_zero`) so a garbage value can never crash the emitter.

When you add a telemetry event you owe BOTH a factory test and an emit-site test. A factory-only test proves nothing about whether the production code ever calls it.

### Drift lifecycle resolution and terminal statuses (#485/#486)

- `DRIFT_LIFECYCLE_RESOLVED` emits on task-terminal transitions and on staleness-guarded on-task verdicts (#486). `GOAL_DRIFT` resolves ONLY at a task-terminal transition — a test that expects it to resolve mid-task is wrong. See `tests/test_drift_lifecycle.py`, `tests/test_drift_resolution_wiring.py`.
- Use the canonical `TERMINAL_TASK_STATUSES` set (#485) — it includes `NOT_NEEDED`. `tests/test_terminal_statuses_single_source.py` is an AST test that pins this as the single source; a hand-rolled `{COMPLETED, FAILED}` set will fail it because the parallel scheduler must skip `NOT_NEEDED` tasks too.

### Protected keeps — tests you must NOT delete

Some machinery is disabled or dormant but deliberately kept. If you are doing a dead-code sweep (#490-style) and a test covers one of these, **it is not dead — leave it**:

- `LOOPING_TOOL_CALL` enum / ladder / promotion / planner surfaces (#204/#206): tool loops deliberately emit `LOOPING_REASONING` with NUDGE-first CRITICAL routing. Tests: `tests/test_tool_loops.py`, `tests/test_intervention_ladder.py`, `tests/test_structural_loop_prevention.py`.
- `PLAN_DIVERGENCE` machinery (#252-disabled but branch-KEEP).
- `reconciler.get_missed_tasks` (#163).

Deleting one of these — or its test — requires explicit human sign-off. The manifest-liveness AST test will NOT catch this for you (these are not numeric knobs), so the guard is human review, not a test.

---

## Flaky-avoidance rules

A flaky test is worse than no test — it trains you to ignore red. Hard rules:

### 1. No real sleeps to advance state. Use a fake clock via module-attribute shim.

Monkeypatch the *module's* time reference, not `time.sleep`. The breaker tests (#479) do this: they replace `_embed.time` with a `SimpleNamespace` whose `monotonic` reads a mutable dict, then advance the dict:

```python
# tests/test_embedding_backend.py
def _install_fake_clock(monkeypatch, start: float = 1000.0) -> dict[str, float]:
    import types
    clock = {"t": start}
    monkeypatch.setattr(
        _embed, "time", types.SimpleNamespace(monotonic=lambda: clock["t"])
    )
    return clock

# in the test — advance instantly, no wall-clock wait:
clock = _install_fake_clock(monkeypatch)
clock["t"] += 59.0
assert _embed._get_model() is None          # cooldown not elapsed
clock["t"] += 2.0
assert _embed._get_model() is failing       # half-open probe admitted
```

For a knob-driven timeout, monkeypatch the module constant directly: `monkeypatch.setattr(goals_mod, "GOAL_DRIFT_IDLE_SECONDS", 0.05)` (`tests/test_stall_watchdog.py::test_idle_goal_judge_fires_once_per_episode`).

### 2. Wrap anything that could hang in `asyncio.wait_for`.

The pause-deadline tests (#482) bound every run task so a bug that hangs surfaces as a fast timeout failure, not a stuck CI job:

```python
# tests/test_pause_deadline.py
outcome = await asyncio.wait_for(run_task, timeout=5.0)
assert "pause escalation deadline expired" in (outcome.reason or "")
```

The deadlines under test are tiny (`pause_escalate_deadline_s=1.5`); the `wait_for` timeout is a generous outer bound (5s) that only trips on a genuine hang.

### 3. For real background tasks, poll a condition — never `sleep(fixed)`.

The stall-watchdog tests spawn real `asyncio` tasks and poll:

```python
# tests/test_stall_watchdog.py
async def _wait_for(cond, *, timeout: float = 2.0, message: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0)   # yield, don't burn a fixed delay
    raise AssertionError(message or "condition not met before timeout")
```

`await asyncio.sleep(0)` yields control without consuming wall time. A fixed `await asyncio.sleep(0.5)` is a flake generator — under CI load the task may not have run yet.

### 4. Drain fire-and-forget work before asserting.

See the judge recipe: `_drain_background_judges`. Any `_background_*` task set on the steerer must be gathered before you read the sink or a call count.

### 5. No real network, no real LLM, no real embedding server.

- LLM: `stub_call_llm` / `CannedCallLLM`.
- Embedding: `_embed.set_backend_loader(lambda _url: fake_backend)` (with `request.addfinalizer` to reset it) — `tests/test_embedding_backend.py`. The breaker's `_MODEL_UNAVAILABLE` state is module-global; reset it or use the env fixture.
- ADK: scripted `BaseLlm`, never a real model endpoint.

### 6. Isolate module-global state.

`_isolate_orchestration_store_registries` (autouse) handles the state-store dicts. The embedding breaker's module globals (`_MODEL_UNAVAILABLE`, `_RUNTIME_FAILURE_TRIPPED`, `_RUNTIME_FAILURE_COUNT`) need explicit reset in tests that trip them — use `_embed.reset_circuit_breaker()` or the fixture teardown. A leaked breaker-tripped flag makes the *next* test's embedding call short-circuit and return `[]`.

---

## Two fully worked examples

### Worked example 1 — a goal-drift judge (active mode, with drain)

The mistakes a weak model makes here: forgetting the drain, forgetting active mode, and asserting before the background task ran. This template avoids all three.

```python
async def test_goal_drift_critical_routes_to_refine(active_steering_config) -> None:
    call_llm = _stub_call_llm([
        {"off_goal": True, "reason": "coordinator abandoned the goal"},  # judge verdict
    ])
    sink = ListSink()
    planner = StubPlanner()
    steerer = DefaultSteerer(
        steering_config=active_steering_config,       # ACTIVE — or the refine is suppressed
        goal_drift_call_llm=call_llm,                 # verify the real kwarg name in goldfive/steerer.py
    )
    steerer.bind(sinks=[sink], planner=planner)
    session = _session_with_plan()

    await steerer.drift.note_agent_turn(session)      # spawns a background judge (fire-and-forget)
    await _drain_background_judges(steerer)           # <-- MUST drain before asserting

    drifts = [e for e in sink.events
              if e.WhichOneof("payload") == "drift_detected"
              and e.drift_detected.kind == DriftKind.GOAL_DRIFT]
    assert drifts, "goal-drift judge verdict never produced a drift"
    assert planner.refine_calls, "CRITICAL goal drift must route to refine in active mode"
    assert len(call_llm.calls) == 1, "judge must be called at most once per check"
```

Then write the passive twin: same setup with `SteeringConfig(observation_only=True)`, assert the drift still emits but `planner.refine_calls` reflects only the passive `refine`/`refine_steer` behavior the gate permits (check `tests/test_goldfive_drift_routing.py` for exactly which planner method still runs passively — `refine_steer` does, the control-message enqueue does not).

> Verify the exact `call_llm` kwarg name (`goal_drift_call_llm` vs `reasoning_drift_call_llm`) against `DefaultSteerer.__init__` in `goldfive/steerer.py` before you write it — the two judge families take different kwargs and a wrong name silently disables the judge (the steerer builds a no-op judge when its `call_llm` is `None`).

### Worked example 2 — a tool-loop detector severity cap (#484)

```python
def test_name_only_loop_caps_at_info() -> None:
    tracker = ToolLoopTracker()                       # defaults; verify ctor in goldfive/drift/...
    # five same-NAME calls, DISTINCT args -> no exact-repeat corroboration
    drifts: list[DriftEvent] = []
    for i in range(5):
        drifts = tracker.observe_tool_call(           # keyword-only; returns list[DriftEvent]
            invocation_id="inv", agent_name="a", tool_name="search",
            args={"q": f"query-{i}"},
        )
    assert drifts, "final call should surface a name-axis loop drift"
    drift = drifts[-1]                                            # observe_tool_call returns a list
    assert drift.severity is DriftSeverity.INFO                  # capped
    assert drift.raw.get("severity_capped_from") is not None     # audit trail
    assert drift.raw.get("mode") == "name"                       # name-axis mode value is "name"

def test_exact_repeat_promotes_to_warning() -> None:
    tracker = ToolLoopTracker()
    drifts: list[DriftEvent] = []
    for _ in range(3):                                 # >=2 identical (name, args_hash)
        drifts = tracker.observe_tool_call(
            invocation_id="inv", agent_name="a", tool_name="search",
            args={"q": "same"},
        )
    drift = drifts[-1]
    assert drift.severity is DriftSeverity.WARNING
    assert drift.raw.get("mode") == "exact"
    assert "tool_loop_exact" in drift.detail
```

Verify `ToolLoopTracker`'s constructor signature, method name (`observe` vs `record`), and the exact `raw` keys against `goldfive/drift/` before writing — the example above uses the shapes asserted in `tests/test_tool_loops.py` but the method name is the kind of thing that rots.

---

## Assertion granularity — assert the contract, not the incidental

Weak models fail in two opposite directions: over-asserting (pinning incidental output so any refactor breaks the test) and under-asserting (assert `outcome.success` and nothing about *what* happened). Aim for the contract.

DO assert:
- the **presence and payload of the load-bearing event** (`WhichOneof("payload") == "drift_detected"`, `payload.outcome == "suppressed"`);
- the **state transition taken** (task status, `planner.refine_calls`, `session.pending_nudges`);
- the **negative** where the whole point is suppression (channel empty, no re-invoke, handler not called);
- **counts** where "at most once" is the contract (`len(call_llm.calls) == 1`).

DON'T assert:
- exact free-text of a `detail` string (assert a substring that IS the contract, e.g. `"tool_loop_exact" in drift.detail`, not the whole sentence);
- event ordering beyond what the contract guarantees — use `WhichOneof` membership, not positional indexing, unless order IS the contract (`test_event_sequence.py`);
- LLM-minted ids for equality (they churn — assert shape/format via `test_event_id_format.py` helpers, not a literal);
- private counters that a refactor may rename, when a public event carries the same fact.

### Negative-space tests are first-class

Half the invariants in this codebase are "X must NOT happen." Those tests are as important as the positive ones and are the ones a weak model forgets:

- passive mode: the injection did NOT happen (empty channel, no plan swap, no nudge);
- `CleanAgent`: NO drift fired (false-positive canary);
- unbound steerer: NO sink events (`test_no_sink_events_when_steerer_is_unbound`);
- terminal task: NOT invoked by the scheduler (#485);
- approval with no channel: does NOT block (returns `"unavailable"`).

If your feature has a "must not" clause, it owes a negative-space test. A positive-only test suite passes on a build that leaks the exact behavior the invariant forbids.

## Debugging a failing or flaky test

| Symptom | Command / move |
| --- | --- |
| want the failure fast | `uv run pytest tests/test_x.py -x -q` (stop at first) |
| want print / log output | `uv run pytest tests/test_x.py -s` (disable capture) |
| want a debugger at the failure | `uv run pytest tests/test_x.py --pdb` |
| re-run only what failed | `uv run pytest -q --lf` (last-failed), `--sw` (stepwise) |
| see which log records exist | add `caplog` param, `print([r.getMessage() for r in caplog.records])` under `-s` |
| suspect it passes for the wrong reason | stash the production fix and confirm the test FAILS (coverage discipline below) |
| passes solo, fails in suite | module-global leak — run `uv run pytest tests/test_x.py tests/test_neighbor.py` to reproduce ordering; check breaker / state-store globals |
| passes locally, fails in CI | extras gap (`--extra dev` vs `dev+adk+proto`) or a wall-clock assumption on a slow runner |
| hangs forever | a missing `asyncio.wait_for` bound — the run is genuinely stuck; add the bound and it becomes a fast, debuggable failure |

When a stub-driven "active" test asserts the intervention did NOT fire and you expected it to: first check the stub has `is_active_steering() -> True` and the steerer is `observation_only=False`. That single omission accounts for most "why is my intervention test a no-op" confusion (see [the trap](#the-is_active_steering-trap-your-most-likely-silent-no-op)).

---

## Coverage discipline: a fix PR MUST include a test that failed pre-fix

This is the anti-self-fooling rule. A weak model's most common failure is to write a test that passes against the *fixed* code and never verified it would have caught the bug. Enforce it mechanically:

1. Write the fix and the test together.
2. **Stash the fix, keep the test:**
   ```bash
   git stash push -- goldfive/          # stash production changes only
   uv run pytest tests/test_your_new.py -q   # MUST FAIL now
   git stash pop
   uv run pytest tests/test_your_new.py -q   # MUST PASS now
   ```
   If step 2's first run passes, your test does not exercise the bug — rewrite it. (If your fix and test are in the same file, stash by path or temporarily revert the production hunk with `git checkout -p`.)
3. The failure message must be specific. The reporting-tool regression test's assert message names the exact production symptom ("500+ plain ACKs on a single task") so a future regressor sees *what broke*, not just `assert x == y`.

For a refactor PR (extraction, dead-code delete — #489/#490/#491) the discipline inverts: the test suite must be **unchanged in intent**. If an extraction "changes behavior" such that a test now fails, **the extraction is wrong — fix the code, do not edit the test to match.** See [Common mistakes](#common-mistakes). #490 deleted dead code "with archaeology" — the proof it was dead is a passing suite plus the liveness test; if deleting a constant makes the manifest-liveness self-check fail, you deleted a *live* constant.

---

## CI parity

CI runs `lint-and-test` on Python **3.11 and 3.12** with the `dev + adk + proto` extras. It runs:

```bash
uv run ruff check .        # lint — must be clean
uv run pytest -q           # full suite — ~2912 passed / 61 skipped
```

CI does NOT run `ruff format`. Match CI locally before every push:

```bash
uv sync --extra dev --extra adk        # (+ proto if you touched proto)
uv run ruff check . && uv run pytest -q
```

Why the full local suite and not just your file: the `importorskip` skips do not fire in CI (all extras installed there), so a test you think is skipped locally on a minimal install WILL run in CI. And the module-global isolation fixtures mean a leak in your test can only fail *another* test — which you won't see running a single file. Run the whole suite once before pushing.

If a test passes for you but only fails in CI, suspect (in order): test ordering / module-global leakage (the `_isolate_*` and breaker-reset story above), a `dev`-only vs `dev+adk+proto` extras gap, and wall-clock assumptions on a slower runner (rule #3 above). See MEMORY note "Lifecycle gates need stable identity keys" and the "flaky CI-only failure" comment in `tests/conftest.py` (`_isolate_orchestration_store_registries`).

The matrix runs the SAME two commands on 3.11 and 3.12. A version-specific failure is almost always a typing / stdlib-behavior difference (e.g. `dict` ordering guarantees, `asyncio` timing) — reproduce locally with the failing interpreter (`uv run --python 3.11 pytest ...`) rather than guessing. The `proto` extra regenerates/installs the committed stubs; if a proto test skips locally it is because you did not install it — `uv sync --extra dev --extra adk --extra proto` mirrors CI exactly. Never merge on a green *local* run that skipped proto or adk tests; those are the exact tests CI will run and you will not have.

---

## Common mistakes

| DON'T | DO |
| --- | --- |
| Test only active mode ("I flipped `observation_only=False` and it works"). | Ship BOTH: a passive-suppressed test (detection + telemetry, no injection) AND an active-behavior test. The passive test is the one that catches leaks. |
| Give an active-mode stub steerer no `is_active_steering`. | Add `def is_active_steering(self) -> bool: return True` — otherwise `steering_is_active` returns PASSIVE and your intervention silently never fires. |
| Assert on log text (`caplog`, "flow logged 'nudging coordinator'"). | Assert on emitted events via `sink.events` + `WhichOneof("payload")`. Logs are not the contract; events are. (`caplog` is fine only when the *log record's structured fields* ARE the thing under test, e.g. `goldfive.llm.request` fields.) |
| `await note_agent_turn(...)` then immediately assert on `sink.events`. | Drain background judges first (`_drain_background_judges`). Judges are fire-and-forget on `_background_judges`. |
| Use `await asyncio.sleep(0.5)` to "let the task run." | Poll with `_wait_for(cond)` using `await asyncio.sleep(0)`, or advance a fake clock. |
| Mutate `task.status = TaskStatus.X` on a session plan. | Plan/Task are frozen (#247): `set_session_plan(session, with_task_status(plan, id, X))` inside `channel_processor_active()`. |
| Edit a test to make it pass after a refactor "changed behavior." | The refactor is the bug. #489/#490/#491 are behavior-preserving by contract — fix the extraction, keep the test. |
| Add a keyword/regex to classify NL content in a detector test. | Use an LLM judge stub (`stub_call_llm`) or a structural/hash check. #166/#167 retired regex NL heuristics. |
| Hand-list terminal statuses (`{COMPLETED, FAILED}`). | Import `TERMINAL_TASK_STATUSES` (#485) — it includes `NOT_NEEDED`. |
| Key a test assertion on an event id / LLM-minted id being stable. | Assert on the stable identity (full agent path, task id). Ids churn. |
| Let `stub_call_llm` / `CannedCallLLM` fall through silently. | They raise on exhaustion by design — if you hit that, your transcript is short, not the stub broken. Add the missing response. |
| `try/except ImportError: pass` around an optional-extra import. | `pytest.importorskip("...")` or `pytestmark = skipif(...)` — visible skip reason. |
| Run `ruff format --check` and reformat the repo. | Only `ruff check .`. The repo is intentionally not format-clean. |
| Forget `await runner.close()`. | Always close the runner; buffered sinks drop events otherwise. |
| Add `@pytest.mark.asyncio`. | Auto mode — `async def test_*` just works. |
| Assert `GOAL_DRIFT` resolves mid-task. | It resolves ONLY at a task-terminal transition (#486). |
| Run `ruff format` and commit the reformat. | The repo is intentionally not format-clean; only `ruff check .` gates. |
| Use a production-scale timeout (600s) as a test deadline. | Use a tiny deadline (1.5s) bounded by an outer `wait_for(5.0)`. |
| Trust a test docstring's claim about conftest behavior. | Verify against `tests/conftest.py` on main — several docstrings are stale post-#488. |
| Poke `os.environ["GOLDFIVE_..."]` inline. | Use the `goldfive_*_env` controller fixture; it owns the name mapping and cleans up. |
| Simulate ADK `session.state` with a plain dict in a plugin test. | Drive a real scripted-`BaseLlm` ADK run — the shallow-copy handoff bug hides behind a dict simulation. |
| Delete a "disabled" test (`PLAN_DIVERGENCE`, `LOOPING_*`, `get_missed_tasks`) during a cleanup. | Protected keep — needs explicit human sign-off. |
| Depend on module-global breaker/state-store residue from a prior test. | Reset in-test (`_embed.reset_circuit_breaker()`); the autouse fixture clears state-store dicts but do not lean on ordering. |

### A worked "silent no-op" post-mortem

You want to prove a NUDGE fires. You write:

```python
steerer = DefaultSteerer()                      # BUG 1: observation_only=True default
executor = SequentialExecutor()
executor.bind_steerer(MyStub())                 # BUG 2: MyStub has no is_active_steering
...
assert session.pending_nudges                   # fails — you "fix" it by asserting == []
```

Both bugs push the run PASSIVE. Your final "passing" test asserts the nudge did NOT fire and calls it done — you have tested nothing. The fix: `DefaultSteerer(steering_config=SteeringConfig(observation_only=False))` (or `make_active_steerer`), and `MyStub.is_active_steering() -> True`. Then the correct assertion is `assert session.pending_nudges`. Then run the stash-the-fix check to prove it would have failed before the production change.

---

## Self-review before you push

Read your own diff once more against this list before committing. This is the difference between a test that protects the codebase and one that decorates it.

1. **Mode**: does every intervention I touched have BOTH a passive-suppressed and an active-behavior test? Did I request active mode explicitly (`SteeringConfig(observation_only=False)` / `make_active_steerer`)?
2. **`is_active_steering`**: does every active-mode stub steerer implement it returning `True`? (Grep your new test for `class .*Steerer` and check.)
3. **Drain**: if I called `note_agent_turn` / `mark_task_*` / anything that spawns a background judge, did I `_drain_background_judges` before asserting?
4. **Frozen plan**: did I mutate `task.status` in place anywhere? Replace with `with_task_status` + `set_session_plan` under `channel_processor_active()`.
5. **Clock**: any real `asyncio.sleep(>0)` / `time.sleep`? Replace with a fake clock, `_wait_for` poll, or `asyncio.sleep(0)`.
6. **Hang guard**: is every run/await that could block wrapped in `asyncio.wait_for`?
7. **Events not logs**: am I asserting on `sink.events`, not `caplog` text (unless the log record's fields ARE the contract)?
8. **Coverage proof**: for a fix, did I stash the production change and confirm the test FAILS without it?
9. **Refactor honesty**: for a refactor, did I change any test's *intent*? If a test failed after my extraction, did I fix the code (right) or the test (wrong)?
10. **Env hygiene**: did I poke `os.environ` directly, or route through the env-controller fixture? Does the test leave the process env clean?
11. **Protected keeps**: did I delete or weaken a test covering `LOOPING_TOOL_CALL`, `PLAN_DIVERGENCE`, or `reconciler.get_missed_tasks`? Stop — that needs human sign-off.
12. **Skips**: any `try/except ImportError`? Replace with `importorskip` so the skip is visible.
13. **Lint + full suite**: `uv run ruff check . && uv run pytest -q` green locally.

## Verification checklist

Run these after touching the corresponding subsystem. All paths are runnable as-is.

**Always, before push (CI parity):**
```bash
uv sync --extra dev --extra adk
uv run ruff check .
uv run pytest -q                       # expect ~2912 passed / 61 skipped, ~30s
```

**Touched the kill-switch / any gate / any intervention:**
```bash
uv run pytest tests/test_observation_only_strict_passive.py \
              tests/test_observation_only_nudge_gate.py \
              tests/test_observation_only_pause_escalate_carveout.py \
              tests/test_observation_only_abort_carveout.py \
              tests/test_goldfive_drift_routing.py -q
grep -rn "_observation_only" goldfive/ | grep -v "steerer.py"   # should find NO consumer reading it directly outside steerer.py
grep -rn "is_active_steering\|steering_is_active" goldfive/     # the sanctioned reads
```

**Touched a detector / tool-loop / drift classifier:**
```bash
uv run pytest tests/test_tool_loops.py tests/test_drift_classifiers.py \
              tests/test_drift_taxonomy.py tests/test_drift_lifecycle.py -q
```

**Touched a judge / judge scheduling:**
```bash
uv run pytest tests/test_goal_drift_classifier.py tests/test_drift_reasoning.py \
              tests/test_judge_scheduling_guards.py \
              tests/test_judge_empty_response_no_retry.py \
              tests/test_judge_task_lifetime.py -q
grep -rn "_background_judges" tests/ goldfive/    # confirm any new judge path is drained in its test
```

**Touched an executor stage:**
```bash
uv run pytest tests/test_overlay_forward_progress.py \
              tests/test_sequential_executor_overlay.py \
              tests/test_executor_control.py tests/test_pause_deadline.py -q
```

**Touched the ADK plugin / a callback:**
```bash
uv run pytest tests/test_adk_adapter.py tests/test_adk_plugin_tool_observations.py \
              tests/test_dynamic_instruction.py \
              tests/test_before_tool_task_id_injection.py -q
```

**Touched a proto / event factory:**
```bash
uv run pytest tests/test_control_proto.py tests/test_events.py \
              tests/test_decision_telemetry.py tests/test_event_sequence.py \
              tests/test_event_id_format.py tests/test_event_id_globally_unique.py -q
```

**Touched a config knob:**
```bash
uv run pytest tests/test_runtime_config.py tests/test_wrap_runtime_config.py \
              tests/test_stall_watchdog.py tests/test_optimization_manifest.py -q
# and if it's a numeric zicato knob:
uv run pytest tests/test_optimization_manifest.py::test_numeric_mutations_have_live_runtime_consumers -q
```

**Touched the embedding breaker / anything time-driven:**
```bash
uv run pytest tests/test_embedding_backend.py -q
grep -rn "asyncio.sleep([1-9]\|time.sleep\|await asyncio.sleep(0\.[1-9]" tests/your_new_test.py   # find real-sleep flakes
```

**Fix PR — prove the test catches the bug:**
```bash
git stash push -- goldfive/
uv run pytest tests/test_your_new.py -q        # MUST fail
git stash pop
uv run pytest tests/test_your_new.py -q        # MUST pass
```

Cross-references: mode discipline and the ladder gates in `09-steering-ladder-and-gates.md`; detector internals in `07-deterministic-drift-detection.md`; judge internals in `08-llm-judges.md`; executor stages in `04-executors-and-control.md`; the ADK plugin in `05-adk-plugin.md`; event/sink shapes in `12-events-sinks-telemetry.md`; every config knob in `14-config-reference.md`; the invariants themselves in `17-invariants-hazards-history.md`.
