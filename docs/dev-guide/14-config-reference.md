# 14. Config Reference

> **⚠ Predates the agency-preservation merge.** This chapter describes the
> pre-merge mechanics; the merge (PRs #453–#504) renamed `NUDGE`→`SIGNAL`,
> replaced corrective templates with advisory observer notes, and added the
> default-OFF ledger/signal regimes. Default-flag behavior described here is
> still accurate; for the merged as-built state read
> `docs/design/AGENCY-PRESERVATION.md` §6 first — it wins on any conflict.

## Read this chapter when...

- You are adding, removing, renaming, or changing the default of ANY tunable knob.
- You need to know which environment variable feeds a given config field, or what a field does at runtime.
- You are debugging "why did my env var have no effect" — the precedence rules and the process-wide `configure()` last-Runner-wins caveat live here.
- You are touching `optimization/manifest.toml` (the zicato-facing knob inventory) and need the AST liveness-test contract that gates it.
- You want to know which knobs are FROZEN — cannot have their default changed without explicit human sign-off (`observation_only` and everything the agency-preservation branch owns).

This chapter is the exhaustive enumeration. For the *behaviour* behind a knob, follow the cross-references: steering promotion policy is in `09-steering-ladder-and-gates.md`, the drift detectors that read the thresholds are in `07-deterministic-drift-detection.md` and `08-llm-judges.md`, planning knobs in `10-planning-and-revision.md`, the manifest/optimizer surface in `12-events-sinks-telemetry.md`, and the invariants that constrain default changes in `17-invariants-hazards-history.md`.

## Files covered

| File / symbol | What lives there |
|---|---|
| `goldfive/config.py` | The typed config: `RuntimeConfig` + seven sub-configs (`EmbeddingConfig`, `JudgeConfig`, `ToolLoopConfig`, `ReasoningDriftConfig`, `GoalDriftConfig`, `SteeringConfig`, `AgentConfig`), all `from_env()` classmethods, and the env-parse helpers (`_read_bool_env`, `_read_int_env`, `_read_float_env`, `_read_optional_float_env`, `_read_str_env`, `_read_optional_str_env`, `_read_steer_threshold_env`, `_read_reasoning_drift_mode_env`). |
| `goldfive/convenience.py` | `wrap()` and `run()` — every user-facing kwarg. |
| `goldfive/runner.py` | `Runner.__init__` kwargs; `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` reader. |
| `goldfive/executors/sequential.py` | `SequentialExecutor.__init__` kwargs; `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL`; `_DEFAULT_MAX_NUDGE_REPLAYS`. |
| `goldfive/executors/parallel.py` | `ParallelDAGExecutor.__init__` kwargs; `ParallelDAGExecutor.REFINE_FAILURE_THRESHOLD`. |
| `goldfive/steerer.py` | `DefaultSteerer.__init__` kwargs; `REFINE_FAILURE_THRESHOLD`, `PROGRESS_STALL_THRESHOLD_SECONDS`. |
| `goldfive/drift/tool_loops.py` | `load_thresholds_from_env`, `thresholds_from_config`, `configure`, `_read_int_env`, `_read_severity_env`; module `DEFAULT_*` constants. |
| `goldfive/drift/reasoning.py` | `configure` + the `_CONFIG`-vs-constant lookup helpers; module threshold constants. |
| `goldfive/drift/_embed.py` | `configure`; `GOLDFIVE_EMBEDDING_*` reads; `_RUNTIME_FAILURE_THRESHOLD`, `_RUNTIME_RECOVERY_COOLDOWN_S`, `_CACHE_MAX`; `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S`. |
| `goldfive/_llm.py` | `DEFAULT_MAX_OUTPUT_TOKENS`, `MAX_OUTPUT_TOKENS_VAR`, `THINKING_DISABLED_VAR`, `THINKING_DISABLE_CAPABILITIES`, `LLM_CALL_DIAGNOSTICS_VAR`. |
| `goldfive/types.py`, `goldfive/_state_audit.py` | `GOLDFIVE_STRICT_STATE_OWNERSHIP`. |
| `goldfive/optimization/manifest.toml`, `goldfive/optimization/manifest.py` | The zicato-facing knob inventory + loader/validator. |
| `tests/test_optimization_manifest.py` | The AST liveness test that pins every numeric manifest knob to a live runtime read. |

## Invariants that bind you here

1. **`observation_only=True` is the production default and it is FROZEN.** `SteeringConfig.observation_only` defaults to `True` (`goldfive/config.py`, `SteeringConfig.observation_only`). Do not flip that default. It is bench-gated behind step 13b of the agency-preservation roadmap and changing it requires explicit human sign-off. See "Frozen / sign-off-gated defaults" below.
2. **Every knob needs three things or it is broken:** (a) a field on the config object OR a module constant; (b) an env reader that uses one of the existing helpers in `config.py`/`tool_loops.py`/`_embed.py` — never a bespoke `os.environ` parse with a local default; (c) if it is a numeric knob you want zicato to tune, a `manifest.toml` entry whose `python_attr` points at a symbol a runtime consumer actually READS (enforced by the AST liveness test). Skipping any one produces a silently-dead knob.
3. **Read flags through the config object, not `getattr(obj, "flag", default)`.** A local default in a `getattr` fork silently diverges from `config.py`'s default. Resolve through the dataclass field.
4. **`configure()` is process-wide and last-Runner-wins.** `goldfive.drift.reasoning.configure`, `.tool_loops.configure`, and `._embed.configure` install module-global state. In a process hosting multiple Runners with different thresholds, the last `wrap()` call wins. This is a documented tradeoff, not a bug to "fix" by adding per-call plumbing without the design discussion in `config.py` `ReasoningDriftConfig` docstring.
5. **`observation_only` is read via ONE predicate only** — `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)`. The config *field* is the source of truth for construction; the runtime *read* goes through the predicate. Never re-read the boolean field at an injection site. (This is the Wave 1-4 passivity invariant; see `09-steering-ladder-and-gates.md`.)
6. **No new bespoke env parsers.** `config.py` already has typed readers for bool / int / float / optional-float / str / optional-str / steer-threshold / reasoning-mode, and `tool_loops.py` has int + severity readers. Reuse them.

---

## 1. The config model at a glance

goldfive's runtime configuration is a single typed aggregate, `RuntimeConfig` (`goldfive/config.py`), holding seven sub-config dataclasses:

```python
# goldfive/config.py
@dataclasses.dataclass
class RuntimeConfig:
    embedding: EmbeddingConfig
    tool_loops: ToolLoopConfig
    reasoning_drift: ReasoningDriftConfig
    goal_drift: GoalDriftConfig
    judge: JudgeConfig
    steering: SteeringConfig
    agent: AgentConfig
    fail_fast_on_revision_rejection: bool | None
    fail_fast_on_invoke_cancel: bool | None
    strict_state_ownership: bool | None
```

Every sub-config is a **mutable** (`frozen=False`) dataclass with:

- field defaults that match the built-in behaviour,
- a `from_env()` classmethod that reads the `GOLDFIVE_*` env surface into an instance,
- and (for the three that install module-global state) a matching `configure()` in the relevant `drift/` module.

`RuntimeConfig.from_env()` aggregates all seven `from_env()` calls. Each sub-config is independent — a missing env var in one subsystem never affects another.

### How a `RuntimeConfig` gets installed

`goldfive.wrap()` (`goldfive/convenience.py`) is the single install path:

```python
# goldfive/convenience.py — wrap()
resolved_runtime = runtime if runtime is not None else RuntimeConfig.from_env()
_embed_module.configure(resolved_runtime.embedding)
_reasoning_module.configure(resolved_runtime.reasoning_drift)
_tool_loops_module.configure(resolved_runtime.tool_loops)
```

- If the caller passes `wrap(..., runtime=RuntimeConfig(...))`, that object is used verbatim.
- If `runtime=` is omitted (the common case), `wrap()` calls `RuntimeConfig.from_env()`, so a bare `wrap(tree)` is byte-identical to pre-#225 env-driven behaviour.
- The `embedding`, `reasoning_drift`, and `tool_loops` sub-configs are installed **process-wide** via their module `configure()`.
- `agent`, `steering`, `goal_drift`, `judge` are threaded into the ADK adapter and the `DefaultSteerer` per-Runner (see the wrap kwargs section).
- The two fail-fast fields are passed to the `Runner` and the wrap-owned
  `SequentialExecutor`. `strict_state_ownership` is applied by the `Runner`
  for the complete async run lifecycle.

**Dataclasses are deliberately mutable.** The `config.py` module docstring blesses the pattern of "load defaults from env, then bump one field for a debugging run". If you want a snapshot, `dataclasses.replace(...)` a variant rather than mutating the source.

### Where each sub-config lands (data flow through `wrap()`)

Tracing `resolved_runtime` through `goldfive/convenience.py` `wrap()`, each of the seven sub-configs has a distinct destination. Know this before you "add a field to `RuntimeConfig`" — the field is inert until it is threaded to a consumer.

| Sub-config | Threaded to | How | Scope |
|---|---|---|---|
| `embedding` | `goldfive.drift._embed` | `_embed_module.configure(resolved_runtime.embedding)` | process-wide (`configure`) |
| `reasoning_drift` | `goldfive.drift.reasoning` + `DefaultSteerer` | `_reasoning_module.configure(...)` AND steerer kwargs `reasoning_drift_config=` / `reasoning_drift_mode=` | process-wide detectors + per-Runner steerer |
| `tool_loops` | `goldfive.drift.tool_loops` + `DefaultSteerer` | `_tool_loops_module.configure(...)` AND steerer kwarg `tool_loop_config=` | process-wide + per-Runner |
| `goal_drift` | `DefaultSteerer` | steerer kwarg `goal_drift_config=` | per-Runner |
| `judge` | judge routing in `wrap()` | `resolved_runtime.judge.base_url` selects the judge endpoint when neither explicit callable route is present | per-Runner |
| `steering` | `ContextEditor` build + `DefaultSteerer` | `build_editor_from_config(...context_editor_rules...)` AND steerer kwarg `steering_config=` | per-Runner |
| `agent` | ADK adapter | `auto_adapter(..., llm_call_timeout_ms=resolved_runtime.agent.call_timeout_ms, agent_max_output_tokens=resolved_runtime.agent.max_output_tokens)` | per-Runner (ADK only) |
| `fail_fast_on_revision_rejection` | `Runner` | constructor kwarg | per-Runner |
| `fail_fast_on_invoke_cancel` | wrap-owned `SequentialExecutor` | constructor kwarg; caller-supplied executors retain their own policy | per-executor |
| `strict_state_ownership` | `Runner.run` | context-local policy around the complete run | per-Runner and safe across concurrent async tasks |

Note `reasoning_drift` and `tool_loops` are threaded TWICE — once process-wide into their detector modules (thresholds the deterministic detectors read) and once into the steerer (which owns the reasoning-judge scheduling and tool-loop tracker construction). This is why an explicit `steerer=` still gets the process-wide detector thresholds but NOT the steerer-side scheduling config. `agent` is ADK-only — the Claude / callable adapters ignore `llm_call_timeout_ms` / `agent_max_output_tokens` (no analogous surface today).

### Four ways to construct a `RuntimeConfig`

| Pattern | Code | When |
|---|---|---|
| All defaults | `RuntimeConfig()` | Use shipped defaults. The six legacy behavior fields remain `None`, so their existing low-level env fallbacks still apply. |
| From env | `RuntimeConfig.from_env()` | Reproduce the `wrap(tree)` (no `runtime=`) behaviour explicitly, e.g. to inspect what env resolved to. |
| From env, then tweak | `cfg = RuntimeConfig.from_env(); cfg.steering.threshold = "critical"` | The blessed debugging pattern — dataclasses are mutable by design. |
| Snapshot variant | `dataclasses.replace(cfg, steering=SteeringConfig(observation_only=False))` | You want a variant WITHOUT mutating the source (e.g. two Runners from one base). |

A subtlety: `wrap(tree)` with no `runtime=` uses `RuntimeConfig.from_env()`,
which resolves every supported environment setting. Passing
`wrap(tree, runtime=RuntimeConfig())` bypasses `from_env()` and ignores the
environment for most fields. Six legacy behavior fields are deliberate
exceptions because their `None` defaults preserve the fallback that existed
before the typed field: the embedding-breaker cooldown, capability Rules A and
C, revision-rejection fail-fast, invoke-cancel fail-fast, and strict state
ownership. Set any of those fields to a concrete value to override its
environment fallback. With no relevant environment setting, their effective
production behavior matches the built-in defaults even though the raw default
objects contain `None` (see §22 debugging step 1).

### Precedence, in one sentence

Most knobs use **explicit kwarg > explicit `RuntimeConfig` or config-object
field > `GOLDFIVE_*` env var > built-in default**. For the six legacy fields
listed above, a concrete typed value is explicit and wins; `None` means
"consult the legacy fallback." Judge-callable routing has its own precedence
order.

### Master environment-variable index

Every environment variable goldfive reads, in one place. The "Owner" column tells you which section documents it in full. Anything NOT in this table that matches `GOLDFIVE_*` is a constant / enum / proto symbol, NOT an env var (see §15).

| Env var | Owner (§) | Type | Default | Reader |
|---|---|---|---|---|
| `GOLDFIVE_STEER_THRESHOLD` | §3 `SteeringConfig` | `off`/`warning`/`critical` | `warning` | `_read_steer_threshold_env` |
| `GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS` | §3 | int | `3` | `_read_int_env` |
| `GOLDFIVE_STEER_OBSERVATION_ONLY` | §3 | bool | `True` (FROZEN) | `_read_bool_env` |
| `GOLDFIVE_CAPABILITY_RULE_A` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_CAPABILITY_RULE_C` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_STEER_CONTEXT_EDITOR_RULES` | §3 | csv | `None` | inline comma-split |
| `GOLDFIVE_STEER_DESCRIPTIVE_GROWTH` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_STEER_SIGNAL_TELEMETRY` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_CANCEL_INFLIGHT_SCOPE` | §3 | `user_and_safety`/`all` | `user_and_safety` | `_read_cancel_inflight_scope_env` |
| `GOLDFIVE_STEER_SIGNAL_CHANNEL` | §3 | `legacy_user_message`/`request_context` | `legacy_user_message` | `_read_signal_channel_env` |
| `GOLDFIVE_PLAN_MODE` | §3 | `forecast`/`ledger` | `forecast` | `_read_plan_mode_env` |
| `GOLDFIVE_STEER_LEGACY_LADDER` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_STEER_PIN_ASSIGNED_TASK` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_STEER_GRACE_WINDOW_TURNS` | §3 | int | `3` | `_read_int_env` |
| `GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS` | §3 | int | `600000` | `_read_int_env` |
| `GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S` | §3 | float\|None | `None` | `_read_optional_float_env` |
| `GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED` | §3 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_STEER_STALL_TIMEOUT_S` | §3 | float | `600.0` | `_read_float_env` |
| `GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS` | §4 `AgentConfig` | int | `16384` | `_read_int_env` |
| `GOLDFIVE_AGENT_CALL_TIMEOUT_MS` | §4 | int | `120000` | `_read_int_env` |
| `GOLDFIVE_DRIFT_REASONING_MODE` | §5 `ReasoningDriftConfig` | mode | `judge` | `_read_reasoning_drift_mode_env` |
| `GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE` | §5 | float | `0.7` | `_read_float_env` |
| `GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY` | §5 | float | `0.6` | `_read_float_env` |
| `GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY` | §5 | float | `0.4` | `_read_float_env` |
| `GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY` | §5 | float | `0.2` | `_read_float_env` |
| `GOLDFIVE_DRIFT_LOOPING_SIMILARITY` | §5 | float | `0.9` | `_read_float_env` |
| `GOLDFIVE_DRIFT_CLUSTER_SIMILARITY` | §5 | float | `0.75` | `_read_float_env` |
| `GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW` | §5 | int | `5` | `_read_int_env` |
| `GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES` | §5 | int | `3` | `_read_int_env` |
| `GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT` | §5 | bool | `False` | `_read_bool_env` |
| `GOLDFIVE_TOOL_LOOP_WINDOW` | §6 `ToolLoopConfig` | int | `10` | `_read_int_env` |
| `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` | §6 | int | `3` | `_read_int_env` |
| `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` | §6 | int | `5` | `_read_int_env` |
| `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` | §6 | int | `5` | `_read_int_env` |
| `GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY` | §6 | severity | `info` | `_read_severity_env` |
| `GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL` | §7 `GoalDriftConfig` | int | `5` | `_read_int_env` |
| `GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW` | §7 | int | `10` | `_read_int_env` |
| `GOLDFIVE_EMBEDDING_BASE_URL` | §8 `EmbeddingConfig` | str\|None | `None` | `_read_optional_str_env` |
| `GOLDFIVE_EMBEDDING_MODEL` | §8 | str | `""` | `_read_str_env` |
| `GOLDFIVE_EMBEDDING_API_KEY` | §8 | str\|None | `None` | `_read_optional_str_env` |
| `GOLDFIVE_EMBEDDING_TIMEOUT_MS` | §8 | int | `10000` | `_read_int_env` |
| `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S` | §8 | float | `60.0` | `_read_float_env` |
| `GOLDFIVE_JUDGE_BASE_URL` | §9 `JudgeConfig` | str\|None | `None` | `_read_optional_str_env` |
| `GOLDFIVE_JUDGE_MODEL` | §9 | str | `""` | `_read_str_env` |
| `GOLDFIVE_JUDGE_API_KEY` | §9 | str\|None | `None` | `_read_optional_str_env` |
| `GOLDFIVE_JUDGE_TIMEOUT_MS` | §9 | int | `10000` | `_read_int_env` |
| `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` | §14 | `"1"`/else | off | inline `== "1"` |
| `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` | §14 | `"1"`/else | off | inline `== "1"` |
| `GOLDFIVE_STRICT_STATE_OWNERSHIP` | §14 | tri-state | auto (on under pytest) | inline |

That is the complete list — 49 environment variables. The `_read_*` readers all live in `goldfive/config.py`; `_read_severity_env` lives in `goldfive/drift/tool_loops.py`. The fail-fast variables retain exact-`"1"` compatibility parsing, and strict state ownership retains its tri-state resolver.

---

## 2. Env-parse helpers (reuse these — do NOT hand-roll)

All of these live in `goldfive/config.py` unless noted. When you add an env-backed field, call the matching helper. They share one design rule: **a malformed value never crashes and never silently flips a policy knob** — it logs and falls back to the supplied default.

| Helper (`goldfive/config.py`) | Return | Accepts | On bad value |
|---|---|---|---|
| `_read_bool_env(name, default)` | `bool` | truthy `1/true/yes/on/y/t`; falsy `0/false/no/off/n/f/""` (case-insensitive) | logs WARNING, returns `default` |
| `_read_int_env(name, default)` | `int` | positive integer literal | non-int OR `<= 0` → debug log, `default` |
| `_read_float_env(name, default)` | `float` | any float (incl. `0.0`, negatives) | parse failure → debug log, `default` |
| `_read_optional_float_env(name, default)` | `float \| None` | positive float → value; `<= 0` → `None` (explicit disable) | parse failure → `default` |
| `_read_str_env(name, default)` | `str` | anything, incl. `""` (empty is a legit model name) | missing → `default` |
| `_read_optional_str_env(name, default)` | `str \| None` | non-empty stripped string; `""` → `None` | missing → `default` |
| `_read_steer_threshold_env(name, default)` | `str` | `off` / `warning` / `critical` (case-insensitive) | logs WARNING, `default` |
| `_read_reasoning_drift_mode_env(name, default)` | `ReasoningDriftMode` | `judge` / `embedding` / `both` / `off` | logs WARNING, `default` |

`goldfive/drift/tool_loops.py` carries its own parallel pair (kept because the tool-loop module predates the typed config and `ToolLoopConfig.from_env` imports one of them):

| Helper (`goldfive/drift/tool_loops.py`) | Return | Notes |
|---|---|---|
| `_read_int_env(name, default)` | `int` | Same lenient positive-int contract as `config.py`'s. |
| `_read_severity_env(name, default)` | `str` | Validates against `_VALID_SEVERITY_NAMES`; used for `name_axis_max_severity`. `ToolLoopConfig.from_env` imports this one. |

> **Key distinction — `_read_int_env` rejects non-positive; `_read_float_env` does not.** A reasoning-drift threshold of `0.0` is a valid (if degenerate) config, so floats allow it. A window size or timeout of `0` is nonsense, so ints clamp it out. `_read_optional_float_env` treats `<= 0` as the explicit *disable* sentinel (`None`), which is exactly what `pause_escalate_deadline_s` wants.

### The two most-error-prone helpers, verbatim

`_read_bool_env` — the one that logs a WARNING (not a debug) on a typo, because a silently-flipped policy knob is the worst outcome:

```python
# goldfive/config.py
def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if value in {"0", "false", "no", "off", "n", "f", ""}:
        return False
    log.warning(
        "ignoring unknown %s=%r (expected a boolean literal); using default %r",
        name, raw, default,
    )
    return default
```

`_read_optional_float_env` — note the `<= 0` → `None` "disable" mapping that distinguishes it from `_read_float_env`:

```python
# goldfive/config.py
def _read_optional_float_env(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.debug(...)
        return default
    return value if value > 0 else None
```

**Which helper for which type — the decision table:**

| Field Python type | Semantic "unset" | Use |
|---|---|---|
| `bool` | the default | `_read_bool_env` |
| `int` (must be positive: window, timeout, count) | the default | `_read_int_env` |
| `float` (`0.0` / negative are valid) | the default | `_read_float_env` |
| `float \| None` (`None` = feature off) | `None` | `_read_optional_float_env` |
| `str` (`""` is a real value, e.g. model name) | the default | `_read_str_env` |
| `str \| None` (`None` = unset, e.g. api_key, base_url) | `None` | `_read_optional_str_env` |
| steer-threshold literal | the default | `_read_steer_threshold_env` |
| reasoning-drift mode literal | the default | `_read_reasoning_drift_mode_env` |
| tool-loop severity literal | the default | `tool_loops._read_severity_env` |

---

## 3. `SteeringConfig` — the steering / drift-promotion policy

`goldfive/config.py`, class `SteeringConfig`. This is the most load-bearing sub-config and the one with the most Wave-era additions. Threaded into the `DefaultSteerer` as `steering_config=` and consulted at the gated injection sites (see `09-steering-ladder-and-gates.md`). Env surface is the `GOLDFIVE_STEER_*` family, read by `SteeringConfig.from_env`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `threshold` | `str` | `"warning"` | `GOLDFIVE_STEER_THRESHOLD` | `_read_steer_threshold_env` |
| `suppression_window_turns` | `int` | `3` | `GOLDFIVE_STEER_SUPPRESSION_WINDOW_TURNS` | `_read_int_env` |
| `observation_only` | `bool` | `True` | `GOLDFIVE_STEER_OBSERVATION_ONLY` | `_read_bool_env` |
| `capability_rule_a_enabled` | `bool \| None` | `None` (effective `False`) | `GOLDFIVE_CAPABILITY_RULE_A` | `_read_bool_env` |
| `capability_rule_c_enabled` | `bool \| None` | `None` (effective `False`) | `GOLDFIVE_CAPABILITY_RULE_C` | `_read_bool_env` |
| `context_editor_rules` | `list[str] \| None` | `None` | `GOLDFIVE_STEER_CONTEXT_EDITOR_RULES` | comma-split (bespoke, see below) |
| `descriptive_growth_enabled` | `bool` | `False` | `GOLDFIVE_STEER_DESCRIPTIVE_GROWTH` | `_read_bool_env` |
| `signal_telemetry` | `bool` | `False` | `GOLDFIVE_STEER_SIGNAL_TELEMETRY` | `_read_bool_env` |
| `cancel_inflight_scope` | `str` | `"user_and_safety"` | `GOLDFIVE_CANCEL_INFLIGHT_SCOPE` | `_read_cancel_inflight_scope_env` |
| `signal_channel` | `str` | `"legacy_user_message"` | `GOLDFIVE_STEER_SIGNAL_CHANNEL` | `_read_signal_channel_env` |
| `plan_mode` | `str` | `"forecast"` | `GOLDFIVE_PLAN_MODE` | `_read_plan_mode_env` |
| `legacy_ladder` | `bool` | `False` | `GOLDFIVE_STEER_LEGACY_LADDER` | `_read_bool_env` |
| `pin_assigned_task` | `bool` | `False` | `GOLDFIVE_STEER_PIN_ASSIGNED_TASK` | `_read_bool_env` |
| `grace_window_turns` | `int` | `3` | `GOLDFIVE_STEER_GRACE_WINDOW_TURNS` | `_read_int_env` |
| `approval_default_timeout_ms` | `int` | `600_000` | `GOLDFIVE_STEER_APPROVAL_DEFAULT_TIMEOUT_MS` | `_read_int_env` |
| `pause_escalate_deadline_s` | `float \| None` | `None` | `GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S` | `_read_optional_float_env` |
| `stall_watchdog_enabled` | `bool` | `False` | `GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED` | `_read_bool_env` |
| `stall_timeout_s` | `float` | `600.0` | `GOLDFIVE_STEER_STALL_TIMEOUT_S` | `_read_float_env` |

### Field-by-field

**`threshold`** — drift → steer promotion gate. `"off"`: every goldfive-detected drift stays on the legacy passive `REFINE_PLAN` path (no cancel-in-flight, no restart message). `"warning"` (default): `WARNING`+ drifts get promoted to a full steer. `"critical"`: only `CRITICAL` drifts promote (for high-noise trees where the WARNING judge over-fires). Eligible drift kinds are gated separately by `DriftObserver._GOLDFIVE_STEER_ELIGIBLE_KINDS` (a frozenset, NOT an env var — see the disambiguation table).

**`suppression_window_turns`** — number of *logical turns* (one reasoning observation = `Session._reasoning_turn`, NOT a raw event-sequence increment; keying on `_next_sequence` let observability-event volume shrink the window — goldfive#441) within which a fresh user-authored steer suppresses a goldfive-authored promotion. The goldfive drift still surfaces as `DriftDetected` with `suppressed_by_user_steer=true`; no cancel or refine fires. Default `3`.

**`observation_only`** — the master kill-switch. **Default `True`. FROZEN — do not change this default.** When `True`, detection runs in full and `planner.refine_steer` still runs (operators see the would-be plan via `PlanRevised` `dry_run=True`), but the write-paths are skipped:

- plan install in `DefaultSteerer._apply_revision`,
- the `GOLDFIVE_STEER` `ControlMessage` enqueue in `DefaultSteerer._dispatch_goldfive_steer_control`,
- the `request_invocation_cancel` plugin call in `DefaultSteerer.request_invocation_cancel`,
- the Level-2 nudge enqueues onto `session.pending_nudges` (`_dispatch_nudge` + the post-ABSORB handoff, goldfive#202); the executor drain carries a matching defense-in-depth gate (goldfive#264 pattern).

Three revision categories always land even under `observation_only=True`: **bootstrap** (first install on a cold session, `prev is None`), **user-authored** (operator `ControlMessage` STEER, `drift.authored_by == "user"`), and **discovery** (`DriftKind.NEW_WORK_DISCOVERED`, goldfive#258). At runtime the flag is read ONLY through `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)` (missing / `None` / raising → PASSIVE). See invariant 5 and `09-steering-ladder-and-gates.md`.

**`capability_rule_a_enabled` and `capability_rule_c_enabled`** — opt in to
two soft-retired, text-heuristic `CAPABILITY_MISMATCH` signals. Rule A infers
whether an AgentTool-only agent received a leaf task. Rule C compares the
invoked agent's name with other pending task text. `None` preserves the legacy
environment fallback for direct steerer and detector callers; without an
environment value, the effective default is `False`. Explicit `True` or `False`
is authoritative. `RuntimeConfig.from_env()` resolves both fields to concrete
booleans before the default steerer carries them to the ADK plugin.

**`context_editor_rules`** (goldfive#397) — names of `ContextEditRule`s to register on the ADK plugin's `ContextEditor`. `None` (default) AND `[]` both leave the editor unwired (zero overhead — `wrap()` never even builds the editor). Set to e.g. `["prune_cancelled_reasoning"]` to opt in. Unknown names are logged and dropped at registration. Per-rule (not one master switch) so a single rule's e2e regression can be bisected. The env reader is a **deliberate exception** to "reuse a helper": comma-separated splitting with empty→`None` collapse, inline in `from_env`:

```python
# goldfive/config.py — SteeringConfig.from_env
raw_rules = os.environ.get("GOLDFIVE_STEER_CONTEXT_EDITOR_RULES", "").strip()
if not raw_rules:
    rules = defaults.context_editor_rules
else:
    rules = [r.strip() for r in raw_rules.split(",") if r.strip()] or None
```

**`descriptive_growth_enabled`** (goldfive#423 PR 2) — plan-descriptive growth fallback for unmatched delegations. When `True` and the structural capability detector returns a Rule C (out-of-DAG-order) verdict, the steerer synthesises a `discovered=True` task via `PlanReviser.install_descriptive_growth` and re-pins the delegation instead of firing `CAPABILITY_MISMATCH`. Rule A/B unaffected. Default `False` (behind the flag; a later PR was slated to flip it after validation — verify current state before assuming). The PR-1 data-model fields (`Task.discovered`, `Task.discovery_identity_hash`, `DelegationObserved.tool_args_json`) are always present regardless.

**`approval_default_timeout_ms`** (#478) — wait budget substituted when a `report_awaiting_approval` tool call omits `timeout_ms` (or passes `<= 0`) while a control channel is attached. Default `600_000` (= `DEFAULT_APPROVAL_TIMEOUT_MS`, a module constant in `config.py`). On expiry the handler returns `decision="timeout"` and emits a `HUMAN_INTERVENTION_REQUIRED` WARNING drift. An explicit positive `timeout_ms` from the agent still wins; a non-positive config value falls back to `DEFAULT_APPROVAL_TIMEOUT_MS`. Prevents the historical forever-hang (no invocation wall clock covers tool waits). See `13-reporting-tools-and-approval.md`.

**`pause_escalate_deadline_s`** (#482) — deadline (seconds) on the executor's pause wait after a Level-4 `GOLDFIVE_PAUSE_ESCALATE` dispatch. `None` (default) preserves block-forever for Level 4. On expiry the executor drains background steerer/judge tasks, CANCELs every non-terminal task, and emits `RunAborted` carrying the escalation lineage (originating drift kind + ladder level). **Exception:** Level 5 (TERMINATE) must terminate — with no configured deadline it falls back to `goldfive.drift_observer.DEFAULT_TERMINATE_PAUSE_DEADLINE_S` (`600.0`). Operator-issued `PAUSE` controls are NEVER bounded by this knob. `_read_optional_float_env` maps `<= 0` → `None` (disable).

**`stall_watchdog_enabled`** (#487) — wall-clock stall watchdog, default OFF. When enabled, the ADK plugin spawns one background task per dispatch watching a liveness watermark: `max(Session.task_last_progress_at stamps, Session.last_observed_event_at)`. When the watermark goes silent for `stall_timeout_s`, a `TASK_TIMEOUT` drift fires at WARNING, escalating to CRITICAL on continued silence. Routed through normal drift dispatch, so under `observation_only` it is telemetry-only. This is the `TASK_TIMEOUT` producer for runs wedged in a hung async tool call or idling with no task transitions (previously ZERO signal).

**`stall_timeout_s`** (#487) — idle threshold (seconds) before the first `TASK_TIMEOUT` WARNING; each further multiple with no fresh activity fires CRITICAL. Default `600.0`. Non-positive values disable the watchdog even when `stall_watchdog_enabled=True`. Uses `_read_float_env` (allows any float; the "non-positive disables" logic is in the watchdog, not the reader).

> **Interaction warning — stall watchdog vs. idle goal-judge.** `stall_watchdog_enabled` also enables the idle goal-judge trigger, which consumes `GOAL_DRIFT_IDLE_SECONDS` (`goldfive/drift/goals.py`, default `300`) — a *separate* threshold from `stall_timeout_s`. The watchdog fires `TASK_TIMEOUT` at `stall_timeout_s`; the idle goal-judge fires a `GOAL_DRIFT` check at `GOAL_DRIFT_IDLE_SECONDS`. Both are inert unless `stall_watchdog_enabled` is set. Do not conflate them.

### `threshold` × `observation_only` — the interaction matrix

These two knobs are orthogonal but interact. `threshold` decides WHICH drifts are eligible for promotion to a full steer; `observation_only` decides whether an eligible promotion actually WRITES anything. The matrix:

| `threshold` | `observation_only` | Behaviour |
|---|---|---|
| `off` | `True` | Detection runs; no promotion at any severity; nothing injected. Pure telemetry. (This is the most passive config.) |
| `off` | `False` | Every drift stays on the legacy passive `REFINE_PLAN` path — no cancel-in-flight, no restart message — even though active mode is on. `off` overrides. |
| `warning` (default) | `True` (default) | WARNING+ drifts are computed as promotions and `refine_steer` runs (dry-run `PlanRevised` visible), but the three write-paths are skipped. **The shipped production default.** |
| `warning` | `False` | WARNING+ drifts promote AND inject: cancel in-flight + stamp `goldfive.active_steer.*` + refine + restart message. |
| `critical` | `True` | Only CRITICAL drifts computed as promotions; still no injection. |
| `critical` | `False` | Only CRITICAL drifts promote and inject. |

Takeaway: to go from "watching" to "acting" you flip `observation_only` to `False` (per-Runner, §18). To tune HOW aggressively an active Runner acts, move `threshold` between `warning` and `critical`. Setting `threshold=off` neuters promotion regardless of `observation_only` — useful for a "detectors only, never even compute a steer" deployment.

### Which drift kinds are promotion-eligible

`threshold` gates by SEVERITY; a separate frozenset `DriftObserver._GOLDFIVE_STEER_ELIGIBLE_KINDS` gates by KIND. A drift must clear BOTH to promote: its severity ≥ `threshold`'s floor AND its `DriftKind` ∈ the eligible set. Kinds outside the set stay on the passive refine path no matter the severity. This frozenset is NOT env-configurable and is a PROTECTED surface (edits to what tool-loops / plan-divergence route to need the history in `17-invariants-hazards-history.md`, e.g. `LOOPING_TOOL_CALL` deliberately routes NUDGE-first at CRITICAL, #204/#206). Do not "simplify" the eligible-kinds set.

---

## 4. `AgentConfig` — per-agent LLM-call budget

`goldfive/config.py`, class `AgentConfig` (goldfive#256). Covers the **user-tree sub-agent** LLM calls flowing through ADK / litellm (coordinator, research_agent, etc.). Distinct from `JudgeConfig` (goldfive's judge calls) and `LLMPlanner.MAX_OUTPUT_TOKENS` (the planner's own calls). Threaded into the ADK adapter by `wrap()` as `llm_call_timeout_ms=` and `agent_max_output_tokens=`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `max_output_tokens` | `int` | `16384` | `GOLDFIVE_AGENT_MAX_OUTPUT_TOKENS` | `_read_int_env` |
| `call_timeout_ms` | `int` | `120_000` | `GOLDFIVE_AGENT_CALL_TIMEOUT_MS` | `_read_int_env` |

**`max_output_tokens`** — caps generation per ADK sub-agent LLM call. Applied as a **structural ceiling** in `_GoldfiveADKPlugin.before_model_callback` (`goldfive/adapters/_adk_plugin.py`): if the sub-agent or ADK already supplied a *smaller* `max_output_tokens`, that smaller value wins — goldfive only ratchets DOWN, never up. Without it, a looping generation can burn tens of thousands of tokens uninterrupted (live e2e 2026-05-11: 30K+ tokens for a 5-bullet research task).

**`call_timeout_ms`** — wall-clock budget per call. On expiry an `LLM_CALL_TIMEOUT` drift fires (CRITICAL, capacity-shaped). **Under `observation_only` (the production default) the drift is telemetry-only — the in-flight call is NEVER cancelled** (#476: healthy models can genuinely need longer). In active mode the invocation is also flagged for cooperative cancel. Default `120_000` (120s). The legacy 30-min backstop is `DEFAULT_LLM_CALL_TIMEOUT_MS` (`1_800_000`) in `_adk_plugin.py`; operators wanting it pass `call_timeout_ms=1_800_000` or `GOLDFIVE_AGENT_CALL_TIMEOUT_MS=1800000`.

> **Two different timeout defaults, don't confuse them.** `AgentConfig.call_timeout_ms = 120_000` is the operator-facing latency SLO that `wrap()` threads into the adapter as `llm_call_timeout_ms=`. `DEFAULT_LLM_CALL_TIMEOUT_MS = 1_800_000` in `_adk_plugin.py` is the module fallback used ONLY when the adapter is built WITHOUT a threaded value (e.g. a raw adapter construction bypassing `wrap()`). In the `wrap()` path the `AgentConfig` value always wins. The manifest exposes BOTH (`agent_call_timeout_ms_default` at 120000, `default_llm_call_timeout_ms` at 1800000).

> **`max_output_tokens` interaction with the reasoning judge.** The default `16384` matches `REASONING_JUDGE_MAX_OUTPUT_TOKENS` — deliberately, because on a thinking model the judge's `<think>` prelude plus the JSON verdict must fit under one ceiling (the v16 empty-response failure). Do not lower `AgentConfig.max_output_tokens` below what a thinking judge needs if the judges share the tree endpoint (the default auto-detect). Route judges to a separate endpoint (`JudgeConfig`) if you want a tight agent ceiling.

---

## 5. `ReasoningDriftConfig` — reasoning-drift thresholds + judge scheduling

`goldfive/config.py`, class `ReasoningDriftConfig`. Field defaults match the module constants in `goldfive/drift/reasoning.py` one-for-one. Installed **process-wide** via `goldfive.drift.reasoning.configure`. Env surface: `GOLDFIVE_DRIFT_*`, read by `ReasoningDriftConfig.from_env`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `mode` | `ReasoningDriftMode` | `"judge"` (`DEFAULT_REASONING_DRIFT_MODE`) | `GOLDFIVE_DRIFT_REASONING_MODE` | `_read_reasoning_drift_mode_env` |
| `off_topic_distance_threshold` | `float` | `0.7` | `GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE` | `_read_float_env` |
| `intent_divergence_healthy_similarity` | `float` | `0.6` | `GOLDFIVE_DRIFT_INTENT_HEALTHY_SIMILARITY` | `_read_float_env` |
| `intent_divergence_minor_similarity` | `float` | `0.4` | `GOLDFIVE_DRIFT_INTENT_MINOR_SIMILARITY` | `_read_float_env` |
| `intent_divergence_warning_similarity` | `float` | `0.2` | `GOLDFIVE_DRIFT_INTENT_WARNING_SIMILARITY` | `_read_float_env` |
| `looping_reasoning_similarity_threshold` | `float` | `0.9` | `GOLDFIVE_DRIFT_LOOPING_SIMILARITY` | `_read_float_env` |
| `reasoning_cluster_similarity_threshold` | `float` | `0.75` | `GOLDFIVE_DRIFT_CLUSTER_SIMILARITY` | `_read_float_env` |
| `looping_reasoning_hash_window` | `int` | `5` | `GOLDFIVE_DRIFT_LOOPING_HASH_WINDOW` | `_read_int_env` |
| `max_concurrent_judges` | `int` | `3` | `GOLDFIVE_DRIFT_MAX_CONCURRENT_JUDGES` | `_read_int_env` |
| `fallback_to_content_when_no_reasoning` | `bool` | `False` | `GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT` | `_read_bool_env` |

**`mode`** — `"judge"` (LLM-as-a-judge only, the default), `"embedding"`
(deterministic embedding detectors only), `"both"`, or `"off"`.
`DEFAULT_REASONING_DRIFT_MODE = "judge"` in `reasoning.py`. When
`mode in ("judge", "both")` but no judge callable is wired, `wrap()`
logs a WARNING that LLM-as-a-judge detection is disabled for the Runner.

**`off_topic_distance_threshold`** — cosine-distance above which a reasoning block is flagged `OFF_TOPIC`. Consumed via `reasoning._off_topic_distance_threshold()`.

**`intent_divergence_*`** — the three-tier `INTENT_DIVERGENCE` band: `>= healthy` (0.6) is healthy (no drift); between `minor` (0.4) and `healthy` → INFO; between `warning` (0.2) and `minor` → WARNING; below `warning` → CRITICAL. Consumed via `_intent_healthy_similarity()` / `_intent_minor_similarity()` / `_intent_warning_similarity()`.

**`looping_reasoning_similarity_threshold`** (0.9) / **`reasoning_cluster_similarity_threshold`** (0.75) / **`looping_reasoning_hash_window`** (5) — the LOOPING_REASONING + cluster-tightening early-warning knobs. Consumed via `_looping_similarity_threshold()` / `_cluster_similarity_threshold()` / `_looping_hash_window()`.

**`max_concurrent_judges`** (#483) — per-steerer cap on concurrently RUNNING background reasoning-judge LLM calls. N agents thinking in the same event-loop tick used to fire N parallel judge calls (each with a 16384-token ceiling) at the judge endpoint — which under the default auto-detect is the agent tree's OWN model. The cap bounds the burst via a per-steerer `asyncio.Semaphore`; requests beyond it queue and per-`(agent, task)` queued windows coalesce onto the newest observation. Values below 1 are clamped to 1. See `08-llm-judges.md`.

> **`max_concurrent_judges` (config) vs `reasoning_drift_rate_limit` (steerer kwarg) — two different throttles.** `max_concurrent_judges` is a CONCURRENCY cap (how many judge calls run at once, semaphore-bounded). `reasoning_drift_rate_limit` (default `3`, a `DefaultSteerer` kwarg NOT on `RuntimeConfig`) throttles issue FREQUENCY. They compose: the rate limit decides whether to issue at all, the concurrency cap decides how many issued calls run simultaneously. `max_concurrent_judges` lives on `ReasoningDriftConfig` (env-tunable); `reasoning_drift_rate_limit` does not (no env, no `RuntimeConfig` field).

**`fallback_to_content_when_no_reasoning`** (goldfive#263) — synthesize a reasoning signal from the response body on non-thinking models (Gemma-4, Mistral, base-model deployments) where `_extract_reasoning` returns `""` and the reasoning judges silently disarm. When `True`, the ADK plugin's `after_model_callback` feeds the response `content` into `observe_reasoning` iff real reasoning extraction was empty. Deliberately lossy ("what the agent decided" mixes with "what it reasoned about"). Default `False` (opt-in). Real reasoning always wins; the fallback only kicks in on a genuine empty. Consumed via `reasoning._fallback_to_content_when_no_reasoning()`, read by `_choose_reasoning_text` in the ADK plugin.

> **Process-wide / last-Runner-wins.** `reasoning.configure(config)` sets a module-global `_CONFIG`; the detector call sites read through helpers that prefer `_CONFIG` and fall back to the module constant when `_CONFIG is None`. Two Runners in one process with different reasoning thresholds → the last `wrap()` wins. `configure(None)` clears the override (test teardown). If per-Runner reasoning thresholds ever become real, #225's follow-up is to move the config onto `Session` and read per-session.

---

## 6. `ToolLoopConfig` — tool-loop detector thresholds

`goldfive/config.py`, class `ToolLoopConfig`. Installed process-wide via `goldfive.drift.tool_loops.configure`. Env surface: `GOLDFIVE_TOOL_LOOP_*`, read by `ToolLoopConfig.from_env` (which imports `_read_severity_env` from `tool_loops.py`).

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `window` | `int` | `10` | `GOLDFIVE_TOOL_LOOP_WINDOW` | `_read_int_env` |
| `exact_threshold` | `int` | `3` | `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` | `_read_int_env` |
| `name_threshold` | `int` | `5` | `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` | `_read_int_env` |
| `alternating_threshold` | `int` | `5` | `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` | `_read_int_env` |
| `name_axis_max_severity` | `str` | `"info"` | `GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY` | `_read_severity_env` |

**`window`** — ring-buffer size for recent tool calls per `(invocation_id, agent_name)`.

**`exact_threshold`** / **`name_threshold`** — these override the **work** category's WARNING tier ONLY (preserving pre-#204 single-threshold semantics). The graduated CRITICAL tiers and meta-category thresholds remain module constants (not env-tunable). `exact` = same `(name, args_hash)`; `name` = same name, any args.

**`alternating_threshold`** — alternating-pattern length (A,B,A,B,A counts as a loop at 5).

**`name_axis_max_severity`** (#484) — caps the severity of same-name-varied-args (name-axis) hits that LACK exact-repeat corroboration (`>= 2` identical `(name, args_hash)`) in the window. `"info"` (default) keeps the uncorroborated name axis signal-only; `"critical"` restores legacy uncapped behaviour. When capped, the raw payload carries `severity_capped_from`. The corroborated name-axis and the exact axis are unaffected. See `07-deterministic-drift-detection.md`.

> **Two ways to build the tracker kwargs.** `tool_loops.load_thresholds_from_env()` reads the env into a kwargs dict; `tool_loops.thresholds_from_config(config)` adapts a `ToolLoopConfig` into the same dict shape. Both splat into `ToolLoopTracker`. The `wrap()` path uses `configure()` + `thresholds_from_config`; do not add a third source.

> **The env overrides touch the WORK category's WARNING tier only.** The tool-loop detector has graduated per-category tiers (work / read-only / meta) with separate CRITICAL thresholds that are NOT env-tunable — they remain module constants grouped in `tool_loops.py` so a follow-up can surface them. `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD=2` lowers the work-category WARNING floor; it does NOT move the CRITICAL promotion or the read-only/meta thresholds. If you need those tunable, that is a new config surface, not a reinterpretation of the existing env vars. `DEFAULT_NAME_AXIS_MAX_SEVERITY` is a string enum and is env-tunable (`name_axis_max_severity`) but is deliberately NOT in the manifest (the manifest only carries numeric + prompt knobs).

---

## 7. `GoalDriftConfig` — trajectory-level GOAL_DRIFT judge scheduling

`goldfive/config.py`, class `GoalDriftConfig` (#143). Threaded into `DefaultSteerer` as `goal_drift_config=`. Env surface: `GOLDFIVE_GOAL_DRIFT_*`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `check_interval` | `int` | `5` | `GOLDFIVE_GOAL_DRIFT_CHECK_INTERVAL` | `_read_int_env` |
| `activity_window` | `int` | `10` | `GOLDFIVE_GOAL_DRIFT_ACTIVITY_WINDOW` | `_read_int_env` |

**`check_interval`** — number of agent-invocation turns between GOAL_DRIFT judge calls. Read live by `DefaultSteerer` via the `GoalDriftConfig`. **Important for the manifest:** the manifest's `goal_drift_check_interval` entry points at `goldfive.config:GoalDriftConfig.check_interval`, NOT the legacy `goldfive.drift.goals:GOAL_DRIFT_CHECK_INTERVAL` constant — that constant is a back-compat re-export with ZERO runtime consumers (it fails the AST liveness test; see §16). Do not repoint the manifest back at the dead constant.

**`activity_window`** — bounds the agent-activity subset of `session.recent_events` (goldfive#239 unified buffer) fed into the judge prompt, hence prompt size.

> **`check_interval` only advances when `goal_drift_enabled=True` on the Runner AND a `goal_drift_call_llm` is wired.** `DefaultSteerer.note_agent_turn` short-circuits before advancing the counter if no GOAL_DRIFT callable exists. `wrap()` wires the callable (from the resolved judge LLM) precisely so the counter advances — this is the goldfive#217 fix. If you build a steerer directly with no `goal_drift_call_llm`, the GOAL_DRIFT judge never fires regardless of `check_interval`.

> **The idle path is separate.** Beyond the turn-count cadence (`check_interval`), the GOAL_DRIFT judge ALSO fires on an idle trigger when the stall watchdog is on (consuming `GOAL_DRIFT_IDLE_SECONDS`, §3) and on task-boundary transitions (spaced by `DriftObserver._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S = 10.0`s to amortise cost). `check_interval` is the turn-count path only.

---

## 8. `EmbeddingConfig` — embedding backend for reasoning-drift detectors

`goldfive/config.py`, class `EmbeddingConfig` (#221). Installed process-wide via `goldfive.drift._embed.configure`. Env surface: `GOLDFIVE_EMBEDDING_*`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `base_url` | `str \| None` | `None` | `GOLDFIVE_EMBEDDING_BASE_URL` | `_read_optional_str_env` |
| `model` | `str` | `""` | `GOLDFIVE_EMBEDDING_MODEL` | `_read_str_env` |
| `api_key` | `str \| None` | `None` | `GOLDFIVE_EMBEDDING_API_KEY` | `_read_optional_str_env` |
| `timeout_ms` | `int` | `10_000` | `GOLDFIVE_EMBEDDING_TIMEOUT_MS` | `_read_int_env` |
| `breaker_cooldown_s` | `float \| None` | `None` (effective `60.0`) | `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S` | `_read_float_env` |

- `base_url = None` → the OpenAI-compatible HTTP backend is skipped and the lazy-load path falls through to sentence-transformers (if the `goldfive[embedding]` extra is installed).
- `base_url` set → HTTP backend used exclusively, **no silent fall-through** to sentence-transformers on HTTP failure (matches the pre-#225 env-driven contract).
- `model = ""` is a legitimate value (llama.cpp tolerates `model=""`), which is why the reader is `_read_str_env` (empty is NOT "missing") rather than `_read_optional_str_env`.

> **Note:** `_embed.py` STILL reads `GOLDFIVE_EMBEDDING_BASE_URL` / `_MODEL` / `_API_KEY` / `_TIMEOUT_MS` directly in `_try_load_openai_backend` as a lazy-init fallback for the no-`configure()` path. When `wrap()` runs, `configure(config)` supersedes those reads. Both point at the same env names, so the observable surface is identical.

### Embedding backend selection and the circuit breaker

The embedding backend is only exercised when a reasoning-drift detector needs a vector — i.e. `reasoning_drift.mode` is `"embedding"` or `"both"`. In pure `"judge"` mode (the default) the embedding path is never hit, so `EmbeddingConfig` is inert.

Backend resolution order at first encode:
1. `base_url` set → the OpenAI-compatible HTTP backend, used exclusively (no fallback on failure).
2. `base_url` unset + `goldfive[embedding]` extra installed → sentence-transformers (local).
3. Neither → the detectors degrade to no-signal (they cannot embed).

The circuit breaker (`_embed.py`) protects against a flapping HTTP backend: after `_RUNTIME_FAILURE_THRESHOLD` (3) consecutive failures the breaker TRIPS and the detectors degrade to no-signal rather than blocking the run on a dead endpoint. After the effective `EmbeddingConfig.breaker_cooldown_s` it half-opens and retries one call; success closes it, failure re-trips. A concrete typed value wins over the legacy direct-module environment fallback. `None` retains that fallback, whose built-in default is `60.0` seconds. `_RUNTIME_FAILURE_THRESHOLD` and `_CACHE_MAX` remain manifest-tunable module constants.

---

## 9. `JudgeConfig` — dedicated LLM endpoint for goldfive's judges

`goldfive/config.py`, class `JudgeConfig`. When `base_url` is set, `wrap()` routes the two LLM-as-a-judge drift detectors (trajectory-level GOAL_DRIFT judge #218 + per-thinking-message reasoning-drift judge #226) through this dedicated endpoint instead of inheriting the tree's LLM. **Planner and goal_deriver keep the tree's LLM — only the judges split off.** Env surface: `GOLDFIVE_JUDGE_*`.

| Field | Type | Default | Env var | Reader |
|---|---|---|---|---|
| `base_url` | `str \| None` | `None` | `GOLDFIVE_JUDGE_BASE_URL` | `_read_optional_str_env` |
| `model` | `str` | `""` | `GOLDFIVE_JUDGE_MODEL` | `_read_str_env` |
| `api_key` | `str \| None` | `None` | `GOLDFIVE_JUDGE_API_KEY` | `_read_optional_str_env` |
| `timeout_ms` | `int` | `10_000` | `GOLDFIVE_JUDGE_TIMEOUT_MS` | `_read_int_env` |

**Judge-routing precedence** (from `wrap()`, for the two drift judges' `call_llm` / `model`):

1. Explicit `goldfive.wrap(judge_call_llm=...)` — dedicated caller-owned route.
2. Explicit `goldfive.wrap(call_llm=...)` — shared route.
3. `JudgeConfig.base_url` (i.e. `GOLDFIVE_JUDGE_BASE_URL`) — dedicated judge endpoint.
4. Auto-detected tree LLM (`detect_llm`).

An explicit `judge_model=` overrides the model name associated with any
callable route. If `JudgeConfig.base_url` is set but a `CallLLM` cannot be
built, `wrap()` logs a WARNING and the judges fall back to the tree LLM.
When the judges inherit the tree LLM via auto-detect, `wrap()` emits the
**named-model WARNING** naming the agent and model that handle judge
traffic. The warning also reports the concurrent-call count and
`REASONING_JUDGE_MAX_OUTPUT_TOKENS`. A `JudgeConfig`-created endpoint
registers a `close` hook on the Runner. An explicit `judge_call_llm`
remains caller-owned and receives no judge close hook.

> **Cost model — why a separate judge endpoint matters.** Under the default auto-detect, up to `max_concurrent_judges` (3) background reasoning-judge calls run concurrently, EACH budgeting `REASONING_JUDGE_MAX_OUTPUT_TOKENS` (16384) output tokens, ON THE SAME endpoint the agent tree bills against. On a rate-limited cloud model that competes with the agent's own turns for capacity. `JudgeConfig` moves that traffic to a cheap local model. The two judges affected are the trajectory-level GOAL_DRIFT judge (#218) and the per-thinking-message reasoning judge (#226); the planner and goal_deriver stay on the tree LLM regardless.

> **`JudgeConfig` only reroutes the two DRIFT judges.** It does NOT touch the planner, goal_deriver, reflective check, or the pluggable `judges=` list's own LLM wiring. If you pass custom `Judge` instances via `judges=`, wire their LLM yourself — `JudgeConfig` does not reach them.

> **`timeout_ms` is the HTTP client timeout, not the LLM wall clock.** `JudgeConfig.timeout_ms` (10000) bounds the HTTP request to the judge endpoint. It is unrelated to `AgentConfig.call_timeout_ms` (the agent-call wall clock that fires `LLM_CALL_TIMEOUT`). A judge call that exceeds `timeout_ms` fails the HTTP request (handled by the judge's error path — malformed/failed judge → INFO, #479), not a drift.

---

## 10. `wrap()` kwargs (`goldfive/convenience.py`)

`wrap(agent, *, ...)` is the primary user entrypoint. Every component is
overridable. `run(agent, user_input, *, context=None,
judge_call_llm=None, judge_model=None, **wrap_kwargs)` forwards its judge
arguments and all remaining keywords to `wrap()`.

| Kwarg | Type | Default | Effect |
|---|---|---|---|
| `agent` | any | — (positional) | The tree: `AgentAdapter`, ADK `BaseAgent`/`Runner`, Claude SDK factory, or async `(task, session, tools) -> InvocationResult` callable. |
| `planner` | `Planner \| None` | `None` | Wins over the default `LLMPlanner` / `PassthroughPlanner`. |
| `goal_deriver` | `GoalDeriver \| None` | `None` | Wins over `LLMGoalDeriver` / `LiteralGoalDeriver`. |
| `executor` | `Executor \| None` | `None` | Wins over `SequentialExecutor(max_task_invocations=..., overlay_mode=True)`. |
| `steerer` | `Steerer \| None` | `None` | Wins over `DefaultSteerer`. **An explicit steerer means `wrap()` does NOT thread the runtime sub-configs into it — you own its construction.** |
| `sinks` | `list[EventSink] \| None` | `None` → `[LoggingSink()]` | Explicit `[]` suppresses all sinks. |
| `control` | `ControlChannel \| None` | `None` | Live pause/resume/cancel/steer/rewind channel forwarded to the Runner. |
| `call_llm` | `CallLLM \| None` | `None` | Async `(system, user, model) -> str`. Used for the default planner and goal deriver. It is the shared route for built-in drift judges when `judge_call_llm` is absent. Overrides any detected tree LLM. |
| `model` | `str \| None` | `None` | Model name for `LLMPlanner` / `LLMGoalDeriver`. Ignored when no LLM. |
| `judge_call_llm` | `CallLLM \| None` | `None` | Dedicated caller-owned callable for the default goal-drift and reasoning-drift judges. It outranks `call_llm`, `RuntimeConfig.judge`, and the detected tree LLM. Goldfive does not register a judge close hook for it. |
| `judge_model` | `str \| None` | `None` | Model name passed only to the default goal-drift and reasoning-drift judges. It outranks the model associated with every fallback judge route. |
| `max_task_invocations` | `int \| None` | `None` (unbounded) | Cap on total adapter invocations per run. Flowed into both the `SequentialExecutor` default AND the `Runner`. Accepts deprecated `max_plan_reinvocations` for one release (DeprecationWarning). |
| `plugins` | `list[Any] \| None` | `None` | ADK `BasePlugin`s installed on every per-agent runner (so `AgentTool` / `sub_agents` dispatches see the same plugins as the coordinator). Ignored for non-ADK. goldfive#121. |
| `runtime` | `RuntimeConfig \| None` | `None` → `RuntimeConfig.from_env()` | The typed-config aggregate. See §1. |
| `dynamic_instruction` | `bool` | `True` | goldfive#251. Replaces each reachable `LlmAgent`'s static `instruction` with a callable resolver re-reading current-task context from `session.state` every turn (plan-causal prompting; #477 preserves ADK `{var}` templating via `inject_session_state`). `False` keeps static strings. Ignored for non-ADK. |
| `drift_self_reporting` | `bool \| list[str]` | `False` | goldfive#196. `False`: lifecycle reporting tools only. `True`: full pre-#196 set incl. drift opinions. `list[str]`: lifecycle subset + named drift tools. Forwarded to `Runner`. |
| `judge_only` | `bool` | `False` | First-class JUDGE-ONLY mode (#446). `True` → native un-steered run with judges armed and ZERO planning/steering LLM calls. Sets defaults for `planner` (one-task `StaticPlanner`) and `goal_deriver` (`LiteralGoalDeriver`); does NOT touch judges. Explicit `planner=`/`goal_deriver=`/`steerer=` still win. |
| `llm_detector` | `Any` | `None` | Test seam: replaces `detect_llm` for this call. Leave `None` in production. |
| `judge_call_llm_builder` | `Any` | `None` | Test seam: replaces `_build_judge_call_llm`. Leave `None` in production. |
| `judges` | `list[Any] \| None` | `None` → `default_judges()` | goldfive#437. Custom `Judge` list. `[]` opts out of the `JudgementEmitted` envelope surface (legacy hardcoded detector path still runs). Mutually exclusive with `disable_judges`. |
| `disable_judges` | `Iterable[BuiltinJudge \| str] \| None` | `None` | Drops named built-ins from the DEFAULT set. `TypeError` if combined with `judges=`. Unrecognised entries ignored (forward-compatible). |
| `**legacy_kwargs` | — | — | Only `max_plan_reinvocations` is accepted (deprecated); anything else → `TypeError`. |

### `judge_only` vs `observation_only` — do not confuse them

`SteeringConfig.observation_only` gates only the three drift-reactive INJECTION points — the planner's goal-derivation, per-turn planning, and refine STILL run and burn LLM calls. `judge_only=True` additionally swaps in a `StaticPlanner` + `LiteralGoalDeriver` so ZERO planning/steering LLM calls fire. If your goal is "judge the agent's native behaviour with no goldfive LLM spend on planning", use `judge_only`, not `observation_only`. (Why a one-task `StaticPlanner` and not `PassthroughPlanner`: `PassthroughPlanner.generate` returns `None`, so the Runner has no plan and the run aborts with an EMPTY transcript — nothing for the judges to score. `StaticPlanner` returns a baked single-task plan that drives ONE `invoke_passthrough` and whose `refine` returns `None`, so a real transcript is produced with no refine/steer call.)

### Why the default executor is overlay mode

`wrap()` builds `SequentialExecutor(max_task_invocations=..., overlay_mode=True)` because `ADKAdapter` exposes `invoke_passthrough`, the safe path for coordinator trees (goldfive#141). Callers supplying their own `executor=` keep full control.

### ADK return type

When `agent` is an ADK `BaseAgent`, `wrap()` returns a `GoldfiveADKAgent` (a `BaseAgent` subclass that also exposes the Runner surface), so the same call site works both programmatically and as `root_agent` of an `adk web` app. The declared return type stays `Runner`.

### Kwarg interaction notes (the combinations that surprise people)

| Combination | What actually happens |
|---|---|
| `steerer=` + `runtime=` | The `runtime` sub-configs (`steering`, `goal_drift`, `tool_loops`, `reasoning_drift`) are NOT threaded into your steerer — `wrap()` only configures the DEFAULT steerer it builds. But `runtime.embedding` / `.reasoning_drift` / `.tool_loops` ARE still `configure()`d process-wide (that happens before the steerer branch). So detector thresholds apply; steerer policy does not. Construct your steerer with the configs you want. |
| `judge_call_llm=` + `call_llm=` | The default planner and goal deriver use `call_llm`. The built-in goal-drift and reasoning-drift judges use `judge_call_llm`. |
| `judge_call_llm=` + `runtime=RuntimeConfig(judge=JudgeConfig(base_url=...))` | The explicit judge callable wins, so Goldfive does not construct a client from `JudgeConfig` and does not register a close hook for the caller-owned callable. |
| `call_llm=` + `runtime=RuntimeConfig(judge=JudgeConfig(base_url=...))` | `call_llm=` wins for the judges when `judge_call_llm` is absent. `JudgeConfig.base_url` is ignored. |
| `judges=[...]` + `disable_judges=[...]` | `TypeError`. They are mutually exclusive — an explicit `judges=` already spells out the exact set. |
| `judges=[]` (empty) | Opts out of the `JudgementEmitted` envelope surface. The legacy hardcoded detector path STILL runs and still fires `DriftDetected`; you just lose the new judge-envelope events. |
| `planner=` + `judge_only=True` | Your explicit `planner=` wins; `judge_only` only supplies DEFAULTS for `planner`/`goal_deriver`. Same for explicit `goal_deriver=` / `steerer=`. |
| `max_task_invocations=N` with an explicit `executor=` | The `N` is still passed to the `Runner`, but your executor was built by YOU without it — pass `max_task_invocations=N` to your executor's constructor too, or the executor won't enforce it. |
| `sinks=[]` (empty) vs `sinks=None` | `[]` suppresses ALL sinks (including the default `LoggingSink`). `None` gets the `[LoggingSink()]` default. A bare `Runner` (not via `wrap()`) defaults to `[]`. |
| `dynamic_instruction=True` on a non-ADK agent | No-op. The resolver installer only touches `LlmAgent.instruction`; other tree shapes are silently skipped. |
| `plugins=[...]` on a non-ADK agent | Ignored. ADK-only (goldfive#121). |

### The named-model WARNING you will see (and how to silence it)

When `wrap(tree)` receives no `judge_call_llm=`, no `call_llm=`, and no
`JudgeConfig`, the judges inherit the auto-detected tree LLM. `wrap()`
logs a WARNING that names the agent, model, concurrent-call count, and
`REASONING_JUDGE_MAX_OUTPUT_TOKENS`. The warning makes use of a billed or
rate-limited cloud endpoint visible. Select a judge route by passing
`judge_call_llm=`, passing `call_llm=`, or setting
`GOLDFIVE_JUDGE_BASE_URL` / `GOLDFIVE_JUDGE_MODEL`.

### The silent-disarm WARNING (judges wired but no callable)

If `reasoning_drift_mode in ("judge", "both")` but no judge callable
resolved, `wrap()` logs a WARNING that LLM-as-a-judge detection is
disabled for the Runner. A judge callable is absent when the caller
provides no dedicated or shared callable, `JudgeConfig` is not set, and
`detect_llm` finds nothing on the tree. The usual cause is a non-ADK
agent without `judge_call_llm=` or `call_llm=`, or an ADK agent that
`detect_llm` cannot introspect. See goldfive#218/#226 history.

---

## 11. `Runner.__init__` kwargs (`goldfive/runner.py`)

Constructed by `wrap()`, but usable directly. Keyword-only.

| Kwarg | Type | Default | Effect |
|---|---|---|---|
| `agent` | `AgentAdapter` | — (required) | The wrapped adapter. |
| `planner` | `Planner` | — (required) | Plan generator / refiner. |
| `executor` | `Executor` | — (required) | Sequential or parallel executor. |
| `goal_deriver` | `GoalDeriver \| None` | `None` → `PassthroughGoalDeriver("run")` | Bypassed entirely when `user_input` is `list[Goal]`. |
| `steerer` | `Steerer \| None` | `None` → `DefaultSteerer` | Drift observer / steerer. |
| `sinks` | `list[EventSink] \| None` | `None` → `[]` | Event fan-out. (Note: `Runner`'s bare default is `[]`; `wrap()` supplies `[LoggingSink()]`.) |
| `control` | `ControlChannel \| None` | `None` | Live-control channel; forwarded to the executor. |
| `max_task_invocations` | `int \| None` | `None` (unbounded) | Cap on total adapter invocations per run. |
| `conversation` | `Conversation \| None` | `None` | Cross-turn state carrier (prior-plan stash keyed by session id). See `03-runner-and-conversation.md`. |
| `goal_drift_enabled` | `bool` | `True` | Master enable for the trajectory-level GOAL_DRIFT judge counter. When on, `wrap()` wires the planner LLM into the steerer's `goal_drift_call_llm`. |
| `planner_gate` | `Any` | `"auto"` | Gate controlling when the planner is consulted. See `10-planning-and-revision.md`. |
| `drift_self_reporting` | `bool \| list[str]` | `False` | See the `wrap()` table + `13-reporting-tools-and-approval.md`. |
| `fail_fast_on_revision_rejection` | `bool \| None` | `None` | Strict-abort opt-in for goldfive-authored revisions failing `Plan.validate(for_revision=True, prior=...)`. See below. |
| `strict_state_ownership` | `bool \| None` | `None` | Context-local state and plan ownership tripwire. `None` retains the direct-constructor env/pytest fallback. |
| `**legacy_kwargs` | — | — | Reserved; unexpected kwargs raise. |

**`fail_fast_on_revision_rejection`** — the one `Runner` env-with-kwarg-override knob. `None` (default) → consult `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` (`"1"` → `True`, else `False`). Explicit `True`/`False` from the kwarg ALWAYS wins over the env (so tests pin behaviour without unsetting the env). Default behaviour (`False`): a goldfive-authored autonomous refine producing an invalid revision keeps the existing `session.plan`, emits a `HUMAN_INTERVENTION_REQUIRED` INFO `DriftDetected`, and continues; the `REFINE_FAILURE_THRESHOLD=2` escalation still fires after two consecutive `(kind, task_id)` failures. `True` → strict abort (CI / regression / debug). **User-authored `USER_STEER` drifts are NEVER gated by this flag** — `DefaultSteerer.install_user_steer` guarantees a valid `Plan` even when the LLM revision fails validation (PLAN-LIFECYCLE.md §4.2.1/§4.5.1).

```python
# goldfive/runner.py — resolution
if fail_fast_on_revision_rejection is None:
    fail_fast_on_revision_rejection = (
        os.environ.get("GOLDFIVE_FAIL_FAST_REVISION_REJECTION", "0") == "1"
    )
```

### Per-kwarg semantics for the non-obvious ones

- **`goal_drift_enabled`** (`True`) — master enable for the trajectory-level GOAL_DRIFT judge turn counter. When `True`, `DefaultSteerer.note_agent_turn` advances the counter and fires the judge every `goal_drift_config.check_interval` turns. When `False`, the counter short-circuits and the GOAL_DRIFT judge never runs (the reasoning-drift judge is unaffected — it is a separate detector). `wrap()` relies on this being `True` to make the GOAL_DRIFT judge fire; it wires the planner LLM into `goal_drift_call_llm` for exactly this reason (goldfive#217).
- **`planner_gate`** (`"auto"`) — controls WHEN the Runner consults the planner between turns. `"auto"` is the adaptive default. See `10-planning-and-revision.md` for the gate states; it is not an env-configurable knob.
- **`conversation`** (`None`) — a `Conversation` carries cross-turn state (notably the prior-plan stash keyed by session id) when one Runner serves multiple outer ADK sessions. Passing `None` gets a fresh per-run conversation. Sharing a Runner across sessions WITHOUT a `Conversation` is safe because the stash is session-keyed (goldfive#271 v4 Class 1 fix); see `03-runner-and-conversation.md`.
- **`drift_self_reporting`** — the three accepted shapes: `False` (default) registers only the lifecycle reporting tools (`report_task_started` / `_progress` / `_completed` / `_failed` / `_blocked` / `_awaiting_approval` / `report_new_work_discovered`); `True` restores the full pre-#196 set including the drift-opinion tools (`report_plan_divergence`, `declare_task_skipped`, `declare_task_not_needed`); a `list[str]` is the lifecycle subset PLUS named drift tools (names not in `DRIFT_SELF_REPORTING_TOOL_NAMES` are silently ignored). `report_new_work_discovered` is NOT a drift tool and stays default-on (no observation analog for genuinely-new work). See `13-reporting-tools-and-approval.md`.

### The `configure()` process-wide side effect and the last-Runner-wins caveat

Constructing a `Runner` does NOT itself call `configure()` — that happens in `wrap()`. If you build a `Runner` directly (bypassing `wrap()`) and rely on `RuntimeConfig` reasoning/tool-loop/embedding thresholds, you must call the module `configure()` functions yourself (or the detectors read the module constants). This is the seam most likely to bite a direct-`Runner` test.

---

## 12. Executor kwargs

### `SequentialExecutor.__init__` (`goldfive/executors/sequential.py`)

Keyword-only.

| Kwarg | Type | Default | Effect |
|---|---|---|---|
| `max_task_invocations` | `int \| None` | `None` (unbounded) | Cap on total adapter invocations. |
| `max_retries_per_task_lineage` | `int` | `3` | Per-lineage retry budget. |
| `fail_fast` | `bool` | `True` | Abort the run on first unrecovered task failure. |
| `overlay_mode` | `bool` | `False` (but `wrap()` passes `True`) | Overlay driving via `invoke_passthrough` (goldfive#141). |
| `fail_fast_on_invoke_cancel` | `bool \| None` | `None` | Abort policy on the overlay's goldfive-internal supersede-cancel branch (goldfive#332-followup). See below. |
| `**legacy_kwargs` | — | — | Reserved. |

**`fail_fast_on_invoke_cancel`** — `None` → consult `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` (`"1"` → `True`, else `False`); explicit `True`/`False` wins. **Only affects goldfive-internal supersede-cancels** — external cancels (`USER_CANCEL`, `asyncio.CancelledError` from the caller) ALWAYS abort regardless.

```python
# goldfive/executors/sequential.py
if fail_fast_on_invoke_cancel is None:
    fail_fast_on_invoke_cancel = (
        os.environ.get("GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL", "0") == "1"
    )
```

**`_MAX_NUDGE_REPLAYS`** — a **class attribute** (`= _DEFAULT_MAX_NUDGE_REPLAYS = 3`), NOT a constructor kwarg. Caps the overlay's Level-2 nudge-driven re-invoke loop per task (goldfive#202). Subclasses tune it by overriding the class attribute (keeps it out of the public constructor + `legacy_kwargs`). It IS a manifest knob (`executor_max_nudge_replays` → `goldfive.executors.sequential:_DEFAULT_MAX_NUDGE_REPLAYS`).

### `ParallelDAGExecutor.__init__` (`goldfive/executors/parallel.py`)

Positional-or-keyword.

| Kwarg | Type | Default | Effect |
|---|---|---|---|
| `max_concurrency` | `int` | `0` (unbounded) | Max concurrent in-flight tasks per stage. |
| `drift_policy` | `Literal["cancel_stage", "finish_stage"]` | `"finish_stage"` | On drift mid-stage: cancel the running stage or let it finish. |
| `max_task_invocations` | `int \| None` | `None` | Cap on total adapter invocations. |
| `**legacy_kwargs` | — | — | Reserved. |

**`ParallelDAGExecutor.REFINE_FAILURE_THRESHOLD`** — class attribute (`= 2`), the parallel-branch mirror of the steerer-side knob, scoped to stage-level refine failures. Manifest knob `parallel_executor_refine_failure_threshold`. The parallel scheduler also skips terminal tasks (incl. `NOT_NEEDED`) via the canonical `TERMINAL_TASK_STATUSES` set (#485).

**Per-kwarg semantics:**

- `max_retries_per_task_lineage` (Sequential, `3`) — a lineage is a task and its refine-descendants; this bounds how many times the executor re-attempts one lineage before giving up. Distinct from `max_task_invocations` (a global adapter-call ceiling across the whole run).
- `fail_fast` (Sequential, `True`) — abort the whole run on the first unrecovered task failure. `False` lets independent tasks continue.
- `overlay_mode` (Sequential, `False`; `wrap()` passes `True`) — when `True`, task dispatch runs the NATIVE tree via `invoke_passthrough` (goldfive#141 coordinator-safe path) and the executor drains `session.pending_nudges` between passes (gated by `steering_is_active`). When `False`, the executor dispatches per-task assignees directly.
- `max_concurrency` (Parallel, `0`) — `0` means unbounded; a positive value caps concurrent in-flight tasks per stage.
- `drift_policy` (Parallel, `"finish_stage"`) — on a mid-stage drift, `"finish_stage"` lets the running stage complete before reacting; `"cancel_stage"` cancels in-flight tasks immediately. Only these two literals are valid.

### Shared executor helpers (#485)

The canonical `TERMINAL_TASK_STATUSES` frozenset is defined in `goldfive/types.py` (`TERMINAL_TASK_STATUSES`, #485) and imported by both executors (`executors/_shared.py`, `executors/parallel.py`) plus `steerer.py` and `task_state_machine.py` — so "is this task done?" is decided one way. `NOT_NEEDED` is a terminal status: a task the reconciler marked unnecessary is skipped by the parallel scheduler exactly like a `COMPLETED` one. `executors/_shared.py` provides the shared scheduler helper functions that consult it. Do not re-derive a local "terminal" set in a new executor path; import `TERMINAL_TASK_STATUSES` from `types` (drift-condition resolution in #486 depends on the single definition).

### Class-attribute knobs (no constructor kwarg, no env — subclass or manifest)

Several knobs are intentionally class attributes, NOT constructor kwargs and NOT env vars. The rationale (stated at each site) is to keep the public constructor surface small and out of `legacy_kwargs` handling; subclasses override the attribute, and zicato tunes them via the manifest. If you are looking for a kwarg to set one of these and can't find it, this is why.

| Class attribute | Value | Where | Manifest id | How to change |
|---|---|---|---|---|
| `SequentialExecutor._MAX_NUDGE_REPLAYS` | `3` | `executors/sequential.py` | `executor_max_nudge_replays` | subclass override or manifest |
| `DefaultSteerer.REFINE_FAILURE_THRESHOLD` | `2` | `steerer.py` | `refine_failure_threshold` | subclass or manifest |
| `DefaultSteerer.PROGRESS_STALL_THRESHOLD_SECONDS` | `600.0` | `steerer.py` | `progress_stall_threshold_seconds` | subclass or manifest |
| `ParallelDAGExecutor.REFINE_FAILURE_THRESHOLD` | `2` | `executors/parallel.py` | `parallel_executor_refine_failure_threshold` | subclass or manifest |
| `LLMPlanner.DEFAULT_MAX_REFINE_ATTEMPTS` | `2` | `planner.py` | `planner_default_max_refine_attempts` | subclass, per-instance override, or manifest |
| `LLMPlanner.MAX_OUTPUT_TOKENS` | `16384` | `planner.py` | `planner_max_output_tokens` | subclass or manifest |
| `LLMGoalDeriver.MAX_OUTPUT_TOKENS` | `8192` | `goal_deriver.py` | `goal_deriver_max_output_tokens` | subclass or manifest |
| `DriftObserver.REFLECTIVE_MAX_OUTPUT_TOKENS` | `16384` | `drift_observer.py` | `reflective_max_output_tokens` | subclass or manifest |
| `DriftObserver._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` | `10.0` | `drift_observer.py` | `goal_drift_task_boundary_min_interval_seconds` | subclass or manifest |

> **`DEFAULT_MAX_REFINE_ATTEMPTS` also has a per-instance override.** `LLMPlanner(..., max_refine_attempts=N)` overrides the class default per Runner. The class attribute is the fallback when no per-instance value is supplied and it is the manifest's tuning target. The others in the table have no per-instance kwarg — subclass or manifest only.

---

## 13. `DefaultSteerer.__init__` kwargs (`goldfive/steerer.py`)

Mostly constructed by `wrap()` from the runtime sub-configs, but the constructor also exposes standalone kwargs (some legacy, some for callers building their own steerer). Keyword-only.

| Kwarg | Type | Default | Effect / source |
|---|---|---|---|
| `reflective_check_interval` | `int` | `15` | Turns between opt-in reflective self-progress checks. |
| `reflective_call_llm` | `ReflectiveCallLLM \| None` | `None` | LLM for the reflective check. |
| `reflective_model` | `str` | `""` | Model for the reflective check. |
| `goal_drift_check_interval` | `int \| None` | `None` | Legacy override; superseded by `goal_drift_config.check_interval` when a config is passed. |
| `goal_drift_call_llm` | `ReflectiveCallLLM \| None` | `None` | The GOAL_DRIFT judge callable. `wrap()` supplies the callable from its resolved judge route. |
| `goal_drift_model` | `str` | `""` | GOAL_DRIFT judge model. |
| `goal_drift_activity_window` | `int \| None` | `None` | Legacy override; superseded by `goal_drift_config.activity_window`. |
| `goal_drift_config` | `GoalDriftConfig \| None` | `None` | Typed config (preferred). |
| `tool_loop_config` | `ToolLoopConfig \| None` | `None` | Typed config. |
| `reasoning_drift_config` | `ReasoningDriftConfig \| None` | `None` | Typed config. |
| `reasoning_drift_mode` | `ReasoningDriftMode` | `"judge"` (`DEFAULT_REASONING_DRIFT_MODE`) | Reasoning-drift mode. |
| `reasoning_drift_call_llm` | `ReflectiveCallLLM \| None` | `None` | Reasoning-judge callable. |
| `reasoning_drift_model` | `str` | `""` | Reasoning-judge model. |
| `reasoning_drift_rate_limit` | `int` | `3` | Reasoning-judge rate limit. |
| `reasoning_binding_confidence_threshold` | `float` | `0.7` | Confidence floor for binding a reasoning observation to a task. |
| `steering_config` | `SteeringConfig \| None` | `None` | The promotion-policy config (`observation_only`, `threshold`, watchdog, etc.). |
| `goldfive_steer_threshold` | `str \| None` | `None` | Legacy override for `steering_config.threshold`. |
| `goldfive_steer_suppression_window_turns` | `int \| None` | `None` | Legacy override for `steering_config.suppression_window_turns`. |
| `judges` | `list[Any] \| None` | `None` | Pluggable judge list (also settable via `set_judges`, which `wrap()` calls). |

Steerer class-attribute knobs surfaced in the manifest: `DefaultSteerer.REFINE_FAILURE_THRESHOLD` (`= 2`) and `DefaultSteerer.PROGRESS_STALL_THRESHOLD_SECONDS` (`= 600.0`).

> **When `wrap()` builds the steerer** it passes `goal_drift_config`, `tool_loop_config`, `reasoning_drift_config`, `reasoning_drift_mode`, `steering_config`, and (when a judge callable resolved) `goal_drift_call_llm` / `goal_drift_model` / `reasoning_drift_call_llm` / `reasoning_drift_model` — all from `resolved_runtime`. Passing your own `steerer=` to `wrap()` means NONE of this threading happens; you must construct the steerer with the configs you want.

### Legacy scalar kwargs vs. typed configs (which wins)

The steerer constructor carries BOTH the typed config objects and the older per-scalar overrides that predate them. The typed config is preferred; the scalars are back-compat. When both are supplied:

| Typed config field | Legacy scalar kwarg | Resolution |
|---|---|---|
| `goal_drift_config.check_interval` | `goal_drift_check_interval` | Config wins when a `goal_drift_config=` is passed; the scalar is the fallback when no config object is supplied. |
| `goal_drift_config.activity_window` | `goal_drift_activity_window` | Same. |
| `steering_config.threshold` | `goldfive_steer_threshold` | Config wins; scalar is legacy. |
| `steering_config.suppression_window_turns` | `goldfive_steer_suppression_window_turns` | Config wins; scalar is legacy. |

Do NOT introduce new scalar kwargs — extend the typed config. The scalars exist only so pre-#225 callers don't break; new code passes `goal_drift_config=` / `steering_config=`.

### Non-config-threaded steerer kwargs

Several steerer kwargs are NOT sourced from `RuntimeConfig` — they have no `RuntimeConfig` field and `wrap()` leaves them at their constructor defaults:

- `reflective_check_interval` (`15`), `reflective_call_llm`, `reflective_model` — the opt-in reflective self-progress check. `wrap()` does not wire a reflective LLM, so the reflective check is inert by default; a caller building their own steerer opts in.
- `reasoning_drift_rate_limit` (`3`) — reasoning-judge rate limit (distinct from `max_concurrent_judges`, which is the concurrency cap; the rate limit throttles issue frequency).
- `reasoning_binding_confidence_threshold` (`0.7`) — confidence floor for binding a reasoning observation to a task before a task-scoped drift can fire.
- `judges` — set by `wrap()` via `set_judges(...)`, NOT the constructor kwarg, when it resolves the default/custom judge list.

If you need one of these operator-configurable, that is a `RuntimeConfig` extension (add the field, thread it in `wrap()`), not a new bare kwarg.

---

## 14. Low-level compatibility fallbacks

All behavior-bearing settings in this section are represented in typed runtime
configuration. The low-level consumers still read the original environment
variables when constructed or called directly without an explicit typed value.

| Env var | Read at | Default / semantics |
|---|---|---|
| `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` | `goldfive/runner.py` (`Runner.__init__`) | `"1"` → strict abort on invalid goldfive-authored revision; else non-fatal. Only consulted when the `fail_fast_on_revision_rejection` kwarg is `None`. |
| `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` | `goldfive/executors/sequential.py` | `"1"` → abort on goldfive-internal supersede-cancel; else continue. Only when the kwarg is `None`. |
| `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S` | `goldfive/drift/_embed.py` (`_recovery_cooldown_s`) | Used only when no `EmbeddingConfig` is installed. |
| `GOLDFIVE_STRICT_STATE_OWNERSHIP` | `goldfive/_state_audit.py` (`is_enabled`) | Used when no per-Runner override is active. Tri-state: `1/true/yes/on` → force-on; `0/false/no/off` → force-off; unset/`auto`/other → auto (on under pytest, off otherwise). |
| `GOLDFIVE_CAPABILITY_RULE_A`, `GOLDFIVE_CAPABILITY_RULE_C` | `goldfive/drift/capability_check.py` | Used only when a direct detector caller omits the corresponding explicit policy. |

**Circuit-breaker constants** in `_embed.py` (not env, but tunable via the manifest): `_RUNTIME_FAILURE_THRESHOLD = 3` (consecutive backend failures before the breaker trips → detectors degrade to no-signal), `_RUNTIME_RECOVERY_COOLDOWN_S = 60.0` (half-open cooldown), `_CACHE_MAX = 512` (per-text encode LRU size). See `07-deterministic-drift-detection.md` for the half-open recovery behaviour (#479).

---

## 14b. The `_llm.py` LLM-call knob surface (#491)

`goldfive/_llm.py` is the ONE module that owns goldfive's internal LLM dispatch (#491 folded every call path into it). It is not env-configurable, but it holds three per-call "knobs" carried via `contextvars.ContextVar` (which replaced the old closure-attribute smuggling that lost data under concurrent judges) plus the model-capability table. When you touch a call site that dispatches through `call_llm`, these are the surfaces you configure per-dispatch — do NOT re-add closure attributes.

### Module constants

| Symbol | Type | Value | Meaning |
|---|---|---|---|
| `DEFAULT_MAX_OUTPUT_TOKENS` | `int` | `4096` | Process-default output cap when no per-callsite budget is in scope. Manifest knob `default_max_output_tokens`. |
| `THINKING_DISABLE_CAPABILITIES` | tuple | `(("qwen", ThinkingDisableCaps(openai_enable_thinking_field=True, no_think_prompt_prefix=True)),)` | Vendor thinking-disable conventions, matched by lowercase substring on the model name. |

### The three ContextVars (per-callsite, not env)

| ContextVar | Setter (context manager) | Getter | Effect |
|---|---|---|---|
| `MAX_OUTPUT_TOKENS_VAR` (`int \| None`, default `None`) | `call_llm_budget(n)` | `get_max_output_tokens()` (→ `n` or `DEFAULT_MAX_OUTPUT_TOKENS`) | Binds a per-callsite output-token cap around one `await call_llm(...)`. Judges/planner/reflective bind 16k, goal_deriver 8k. |
| `THINKING_DISABLED_VAR` (`bool \| None`, default `None`) | `call_llm_thinking_disabled()` | `get_thinking_disabled()` | Suppresses model "thinking" for goldfive's own JSON-shaped meta-cognition calls (judges / goal_deriver / planner refine / reflective). Agent-side calls keep their natural thinking. |
| `LLM_CALL_DIAGNOSTICS_VAR` (`LlmCallDiagnostics \| None`, default `None`) | `llm_call_diagnostics()` (yields the object) | `record_llm_call_diagnostics(thought_count=, answer_count=)` | Per-dispatch thought/answer part counts so a caller can distinguish "spent budget thinking, emitted no answer" from "returned garbage". No-op when no consumer installed one. |

### Why ContextVars, not attributes on the callable

The counts and flags used to be mutated onto the shared `call_llm` callable (`call_llm.last_thought_count`). That is **last-writer-wins once concurrent background judges dispatch through the same closure** (the exact bug `max_concurrent_judges` bounds but does not eliminate). ContextVars scope each value to the dispatching task's context, so concurrent judges cannot observe each other's counts. **Do not "simplify" this back to closure attributes** — it re-introduces the concurrency bug.

### The thinking-disable capability table (config, NOT NL classification)

How "disable thinking" is expressed on the wire is a **vendor convention**, keyed off the model name — this is a lookup table (allowed: exact/substring match of structured model-name data), NOT a natural-language classifier (banned; see invariant/#166/#167). `thinking_disable_caps(model_name)` matches `THINKING_DISABLE_CAPABILITIES` by lowercase substring so litellm-prefixed names (`"openai/Qwen3-32B"`, `"hosted_vllm/Qwen/Qwen3-32B"`) all route to the Qwen family. Two Qwen-specific hacks ride the OpenAI-compatible wire format and are Qwen/litellm-family ONLY (#491 narrowed them from "everyone"):

- `openai_enable_thinking_field` → `extra_body={"enable_thinking": False}` on `chat.completions.create`.
- `no_think_prompt_prefix` → `/no_think` prepended to the system prompt (fallback for endpoints that drop unknown request fields).

The genai `ThinkingConfig(include_thoughts=False, thinking_budget=0)` opt-out on the ADK/Gemini path is applied for EVERY model regardless of this table — it is first-class SDK surface, not a vendor hack. **To add a new vendor's thinking-disable convention:** append a `(marker, ThinkingDisableCaps(...))` pair to `THINKING_DISABLE_CAPABILITIES` — do not scatter model-name checks across call sites.

### Using the ContextVars correctly (and testing them)

The ContextVars are always entered via their context managers — `call_llm_budget(n)`, `call_llm_thinking_disabled()`, `llm_call_diagnostics()` — which set-and-reset around a single `await call_llm(...)`. Never `.set()` a ContextVar directly at a call site without the manager: the manager's `finally: reset(token)` is what prevents a value leaking into a sibling call in the same async context. In tests, wrap the dispatch under test in the same manager and assert on the yielded diagnostics object AFTER the `await` returns (the object outlives the `with` block). The counts are per-call and concurrency-safe precisely because each `llm_call_diagnostics()` installs a fresh object — you can run concurrent judge dispatches in a test and each sees only its own counts.

The `DEFAULT_MAX_OUTPUT_TOKENS` fallback (`4096`) applies ONLY when no `call_llm_budget(...)` is in scope. Every goldfive-internal consumer (planner / goal_deriver / judges / reflective) enters a `call_llm_budget` with its own cap (16k or 8k), so the `4096` default is what a USER-supplied `call_llm` sees if it reads `get_max_output_tokens()` outside any goldfive budget scope. A user callable is not required to read it — the cost of ignoring it is an uncapped dispatch.

---

## 15. `GOLDFIVE_*` names that are NOT environment variables (disambiguation)

A grep for `GOLDFIVE_` turns up several symbols that look like env vars but are **not** — do not add env readers for them, and do not document them as config knobs.

| Symbol | What it actually is | Where |
|---|---|---|
| `GOLDFIVE_STEER` | A `ControlMessage` kind (string constant `"GOLDFIVE_STEER"`) — the internal steer control enqueued onto the executor channel. | `goldfive/control.py` |
| `GOLDFIVE_PAUSE_ESCALATE` | A `ControlMessage` kind — the Level-4 pause-escalate control. | `goldfive/control.py` |
| `GOLDFIVE_STEER_REPLAY` | A `ReentryKind` enum value (`"goldfive_steer_replay"`). | `goldfive/adapters/adk_reentry.py` |
| `GOLDFIVE_PREFIX` | The `"goldfive."` state-key namespace prefix. | `goldfive/state_store.py`, `goldfive/adapters/_adk_state_protocol.py` |
| `GOLDFIVE_PLANNER_OPT_OUT_ATTR` | The attribute name (`"_goldfive_planner_opt_out"`) checked on ADK agents. | `goldfive/adapters/adk.py` |
| `_GOLDFIVE_STEER_ELIGIBLE_KINDS` | A `frozenset[DriftKind]` gating which kinds are promotion-eligible. | `goldfive/drift_observer.py` |
| `GOLDFIVE_LLM_CALL_START_FIELD_NUMBER` / `GOLDFIVE_LLM_CALL_END_FIELD_NUMBER` | Generated protobuf field-number constants. | `goldfive/pb/goldfive/v1/events_pb2.pyi` |

If you are adding a genuinely new env var, it must (a) start with `GOLDFIVE_`, (b) be read through a `config.py`/`tool_loops.py` helper, and (c) appear in the env-var table of the owning sub-config's `from_env` docstring.

### Why the distinction matters

A weak model that greps `GOLDFIVE_` and treats every hit as an env var will "add an env reader" for `GOLDFIVE_STEER` (a `ControlMessage` kind) or `GOLDFIVE_PREFIX` (a state-key namespace) and produce nonsense — an env var that shadows a control-message constant, or a runtime that reads `os.environ["goldfive."]`. The reliable filter: **an env var is read via `os.environ.get(...)` / `os.getenv(...)`; a constant is assigned with `=` or is an enum member / proto field-number.** Confirm with `grep -n "os.environ.*<NAME>\|getenv.*<NAME>" goldfive/` — if that returns nothing, it is NOT an env var. The master index in §1 is the authoritative env-var list; anything not in it is a constant.

---

## 16. `optimization/manifest.toml` — the zicato-facing knob inventory

`goldfive/optimization/manifest.toml` is the source-of-truth inventory of every prompt + numeric knob the offline meta-loop optimizer (zicato) is allowed to mutate on goldfive's steering path. `goldfive/optimization/manifest.py` loads it into typed `Mutation` records (`Manifest.load()`), validates proposed updates (`Manifest.validate`), and resolves live values. Read with `tomllib` (stdlib, 3.11+).

### Entry shape

Each `[[mutation]]` entry has:

- `id` — stable optimizer-facing identifier.
- `kind` — `"prompt"` or `"numeric"`.
- `source` — either `goldfive/optimization/prompts/<name>.md` (a prompt) or `goldfive/<module>.py:<ATTR>` (a Python attribute).
- `python_attr` — `module.path:Attr` (one level of dotted nesting for classes), e.g. `goldfive.drift.goals:GOAL_DRIFT_SYSTEM_PROMPT` or `goldfive.drift_observer:DriftObserver.REFLECTIVE_SYSTEM_PROMPT`. This is what the optimizer `setattr`s / reads.
- Prompt entries: `required_placeholders` (a list of `{name}` tokens the body MUST contain — validated by `Manifest.validate`).
- Numeric entries: `type` (`"int"`/`"float"`), `range` `[low, high]` inclusive, `default` (tracks the repo's current value at authorship, kept in sync by the manifest tests).
- `tags` — grouping for sweeps.

### The knob families in the manifest

The manifest currently covers (grouped by comment section):

- **Prompt knobs** — reasoning-judge system/user/agent-tree-suffix; goal-drift system/user; reflective-check system/user; planner refine/user_steer/looping_tool_call/plan_divergence/generate; goal-derive system. Plus the three plan-template supersession fragments (`_SUPERSESSION_INVARIANT`, `_SUPERSESSION_EXAMPLES`, `_REFINEMENT_GUIDANCE_BLOCK`).
- **Reasoning-drift numeric thresholds** — `OFF_TOPIC_DISTANCE_THRESHOLD`, `LOOPING_REASONING_SIMILARITY_THRESHOLD`, `REASONING_CLUSTER_SIMILARITY_THRESHOLD`, `LOOPING_REASONING_HASH_WINDOW`, the three `INTENT_DIVERGENCE_*`, `SENTENCE_LEVEL_MIN_BLOCK_LENGTH`, `SENTENCE_LEVEL_MAX_SENTENCES`, `_SENTENCE_MIN_LENGTH`.
- **Tool-loop numeric thresholds** — `DEFAULT_WINDOW`, `DEFAULT_EXACT_THRESHOLD`, `DEFAULT_NAME_THRESHOLD`, `DEFAULT_ALTERNATING_THRESHOLD`. (Note: `DEFAULT_NAME_AXIS_MAX_SEVERITY` is a string enum, not a numeric — it is NOT in the manifest.)
- **Goal-drift scheduling** — `goal_drift_check_interval` → `goldfive.config:GoalDriftConfig.check_interval`; `goal_drift_idle_seconds` → `goldfive.drift.goals:GOAL_DRIFT_IDLE_SECONDS`; `stall_watchdog_timeout_seconds` → `goldfive.config:SteeringConfig.stall_timeout_s`.
- **Reasoning-judge / goal-drift prompt-size budgets + LLM caps** — `REASONING_DRIFT_MAX_REASONING_CHARS`, `..._TOOL_OBS_MAX_CHARS`/`_ENTRIES`, `REASONING_JUDGE_MAX_*`, `PLAN_TASKS_SUMMARY_MAX_CHARS`, `AGENT_TREE_BLOCK_MAX_CHARS`, `GOAL_DRIFT_MAX_OUTPUT_TOKENS`, `_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS`.
- **Planner retry budgets + LLM caps** — `LLMPlanner.DEFAULT_MAX_REFINE_ATTEMPTS` (2), `LLMPlanner.MAX_OUTPUT_TOKENS` (16384), `LLMGoalDeriver.MAX_OUTPUT_TOKENS` (8192), `goldfive._llm:DEFAULT_MAX_OUTPUT_TOKENS` (4096).
- **Steerer ladder policy** — `DefaultSteerer.REFINE_FAILURE_THRESHOLD` (2), `DefaultSteerer.PROGRESS_STALL_THRESHOLD_SECONDS` (600), `_DEFAULT_MAX_NUDGE_REPLAYS` (3), `ParallelDAGExecutor.REFINE_FAILURE_THRESHOLD` (2), `DriftObserver.REFLECTIVE_MAX_OUTPUT_TOKENS`, `DriftObserver._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` (10.0).
- **Detector internals / dedup / breaker** — `_embed:_RUNTIME_FAILURE_THRESHOLD` (3), `_embed:_CACHE_MAX` (512), `state_store:PROCESSED_STEER_IDS_CAP` (256).
- **Adapter LLM-call watcher** — `_adk_plugin:DEFAULT_LLM_CALL_TIMEOUT_MS` (1800000), `adk_llm_instrumentation:DEFAULT_AGENT_MAX_OUTPUT_TOKENS` (16384).
- **Config dataclass defaults** — `EmbeddingConfig.timeout_ms` (10000), `JudgeConfig.timeout_ms` (10000), `AgentConfig.call_timeout_ms` (120000).

### Full prompt-knob table (id → python_attr, required placeholders)

Every `prompt` `[[mutation]]`. Each has a canonical `.md` under `goldfive/optimization/prompts/<name>.md` (the optimizer-facing text) and a live Python attribute (`python_attr`). Prompts with `required_placeholders` are rendered with `.format(...)` and MUST contain every listed token or `Manifest.validate` rejects the proposed update. See `08-llm-judges.md` and `10-planning-and-revision.md` for what each prompt drives.

| id | `python_attr` | required placeholders |
|---|---|---|
| `reasoning_judge_system_prompt` | `goldfive.drift.reasoning_judge:REASONING_DRIFT_SYSTEM_PROMPT` | (none) |
| `reasoning_judge_user_prompt` | `goldfive.drift.reasoning_judge:REASONING_DRIFT_USER_PROMPT_TEMPLATE` | `{plan_tasks_summary}`, `{task_block}`, `{current_agent_id}`, `{task_lineage_block}`, `{goals_block}`, `{tool_obs_count}`, `{tool_obs_block}`, `{reasoning_block}` |
| `reasoning_judge_agent_tree_suffix` | `goldfive.drift.reasoning_judge:AGENT_TREE_SYSTEM_PROMPT_SUFFIX` | (none) |
| `goal_drift_system_prompt` | `goldfive.drift.goals:GOAL_DRIFT_SYSTEM_PROMPT` | (none) |
| `goal_drift_user_prompt` | `goldfive.drift.goals:GOAL_DRIFT_USER_PROMPT_TEMPLATE` | `{goals_block}`, `{tasks_block}`, `{activity_count}`, `{activity_block}` |
| `reflective_check_system_prompt` | `goldfive.drift_observer:DriftObserver.REFLECTIVE_SYSTEM_PROMPT` | (none) |
| `reflective_check_user_prompt` | `goldfive.drift_observer:DriftObserver.REFLECTIVE_USER_PROMPT_TEMPLATE` | `{task_id}`, `{task_title}`, `{task_description}`, `{window}`, `{tool_call_summary}`, `{reasoning_summary}` |
| `refine_system_prompt` | `goldfive.planner:_REFINE_SYSTEM_PROMPT` | (none) |
| `user_steer_system_prompt` | `goldfive.planner:_USER_STEER_SYSTEM_PROMPT` | (none) |
| `looping_tool_call_system_prompt` | `goldfive.planner:_LOOPING_TOOL_CALL_SYSTEM_PROMPT` | (none) |
| `plan_divergence_system_prompt` | `goldfive.planner:_PLAN_DIVERGENCE_SYSTEM_PROMPT` | (none) |
| `plan_generate_system_prompt` | `goldfive.planner:_DEFAULT_SYSTEM_PROMPT` | (none) |
| `goal_derive_system_prompt` | `goldfive.goal_deriver:DEFAULT_SYSTEM_PROMPT` | (none) |
| `plan_template_supersession_invariant` | `goldfive.planner:_SUPERSESSION_INVARIANT` | (none) |
| `plan_template_supersession_examples` | `goldfive.planner:_SUPERSESSION_EXAMPLES` | (none) |
| `plan_template_refinement_guidance` | `goldfive.planner:_REFINEMENT_GUIDANCE_BLOCK` | (none) |

> The three `plan_template_*` fragments are embedded verbatim into EVERY refine system prompt (refine / user_steer / looping_tool_call / plan_divergence) via string concatenation — tuning them moves supersession precision/recall without touching a drift-specific prompt body. The `test_manifest_prompt_entries_and_shipped_prompts_are_a_bijection` test enforces one `.md` per entry and one entry per `.md`; adding a prompt file without a manifest entry (or vice-versa) fails the suite.

### Full numeric-knob table (id → python_attr, type, range, default)

Every `numeric` `[[mutation]]` in `manifest.toml`, verbatim from the current file. Use this as the reference for "what can zicato tune and within what bounds". The `range` is `[low, high]` inclusive; zicato's `Manifest.validate` rejects out-of-range or wrong-type proposals.

| id | `python_attr` | type | range | default |
|---|---|---|---|---|
| `off_topic_distance_threshold` | `goldfive.drift.reasoning:OFF_TOPIC_DISTANCE_THRESHOLD` | float | `[0.0, 1.0]` | `0.7` |
| `looping_reasoning_similarity_threshold` | `goldfive.drift.reasoning:LOOPING_REASONING_SIMILARITY_THRESHOLD` | float | `[0.5, 0.999]` | `0.9` |
| `reasoning_cluster_similarity_threshold` | `goldfive.drift.reasoning:REASONING_CLUSTER_SIMILARITY_THRESHOLD` | float | `[0.4, 0.95]` | `0.75` |
| `looping_reasoning_hash_window` | `goldfive.drift.reasoning:LOOPING_REASONING_HASH_WINDOW` | int | `[2, 50]` | `5` |
| `intent_divergence_healthy_similarity` | `goldfive.drift.reasoning:INTENT_DIVERGENCE_HEALTHY_SIMILARITY` | float | `[0.0, 1.0]` | `0.6` |
| `intent_divergence_minor_similarity` | `goldfive.drift.reasoning:INTENT_DIVERGENCE_MINOR_SIMILARITY` | float | `[0.0, 1.0]` | `0.4` |
| `intent_divergence_warning_similarity` | `goldfive.drift.reasoning:INTENT_DIVERGENCE_WARNING_SIMILARITY` | float | `[0.0, 1.0]` | `0.2` |
| `sentence_level_min_block_length` | `goldfive.drift.reasoning:SENTENCE_LEVEL_MIN_BLOCK_LENGTH` | int | `[50, 4000]` | `200` |
| `sentence_level_max_sentences` | `goldfive.drift.reasoning:SENTENCE_LEVEL_MAX_SENTENCES` | int | `[1, 100]` | `10` |
| `reasoning_sentence_min_length` | `goldfive.drift.reasoning:_SENTENCE_MIN_LENGTH` | int | `[1, 200]` | `10` |
| `tool_loop_window` | `goldfive.drift.tool_loops:DEFAULT_WINDOW` | int | `[3, 100]` | `10` |
| `tool_loop_exact_threshold` | `goldfive.drift.tool_loops:DEFAULT_EXACT_THRESHOLD` | int | `[2, 20]` | `3` |
| `tool_loop_name_threshold` | `goldfive.drift.tool_loops:DEFAULT_NAME_THRESHOLD` | int | `[2, 20]` | `5` |
| `tool_loop_alternating_threshold` | `goldfive.drift.tool_loops:DEFAULT_ALTERNATING_THRESHOLD` | int | `[3, 21]` | `5` |
| `goal_drift_check_interval` | `goldfive.config:GoalDriftConfig.check_interval` | int | `[1, 100]` | `5` |
| `goal_drift_idle_seconds` | `goldfive.drift.goals:GOAL_DRIFT_IDLE_SECONDS` | int | `[10, 3600]` | `300` |
| `stall_watchdog_timeout_seconds` | `goldfive.config:SteeringConfig.stall_timeout_s` | float | `[10.0, 7200.0]` | `600.0` |
| `reasoning_drift_max_reasoning_chars` | `goldfive.drift.reasoning_judge:REASONING_DRIFT_MAX_REASONING_CHARS` | int | `[200, 32768]` | `1500` |
| `reasoning_drift_tool_obs_max_chars` | `goldfive.drift.reasoning_judge:REASONING_DRIFT_TOOL_OBS_MAX_CHARS` | int | `[200, 32768]` | `1500` |
| `reasoning_drift_tool_obs_max_entries` | `goldfive.drift.reasoning_judge:REASONING_DRIFT_TOOL_OBS_MAX_ENTRIES` | int | `[1, 64]` | `8` |
| `planner_default_max_refine_attempts` | `goldfive.planner:LLMPlanner.DEFAULT_MAX_REFINE_ATTEMPTS` | int | `[1, 10]` | `2` |
| `planner_max_output_tokens` | `goldfive.planner:LLMPlanner.MAX_OUTPUT_TOKENS` | int | `[512, 65536]` | `16384` |
| `goal_deriver_max_output_tokens` | `goldfive.goal_deriver:LLMGoalDeriver.MAX_OUTPUT_TOKENS` | int | `[512, 32768]` | `8192` |
| `default_max_output_tokens` | `goldfive._llm:DEFAULT_MAX_OUTPUT_TOKENS` | int | `[256, 32768]` | `4096` |
| `refine_failure_threshold` | `goldfive.steerer:DefaultSteerer.REFINE_FAILURE_THRESHOLD` | int | `[1, 10]` | `2` |
| `progress_stall_threshold_seconds` | `goldfive.steerer:DefaultSteerer.PROGRESS_STALL_THRESHOLD_SECONDS` | float | `[10.0, 7200.0]` | `600.0` |
| `executor_max_nudge_replays` | `goldfive.executors.sequential:_DEFAULT_MAX_NUDGE_REPLAYS` | int | `[1, 20]` | `3` |
| `reasoning_judge_max_reasoning_input_chars` | `goldfive.drift.reasoning_judge:REASONING_JUDGE_MAX_REASONING_INPUT_CHARS` | int | `[256, 16384]` | `4096` |
| `reasoning_judge_max_raw_response_chars` | `goldfive.drift.reasoning_judge:REASONING_JUDGE_MAX_RAW_RESPONSE_CHARS` | int | `[256, 16384]` | `2048` |
| `reasoning_judge_max_output_tokens` | `goldfive.drift.reasoning_judge:REASONING_JUDGE_MAX_OUTPUT_TOKENS` | int | `[512, 65536]` | `16384` |
| `reasoning_judge_plan_tasks_summary_max_chars` | `goldfive.drift.reasoning_judge:PLAN_TASKS_SUMMARY_MAX_CHARS` | int | `[200, 16384]` | `2000` |
| `reasoning_judge_agent_tree_block_max_chars` | `goldfive.drift.reasoning_judge:AGENT_TREE_BLOCK_MAX_CHARS` | int | `[200, 8192]` | `1200` |
| `goal_drift_max_output_tokens` | `goldfive.drift.goals:GOAL_DRIFT_MAX_OUTPUT_TOKENS` | int | `[512, 65536]` | `16384` |
| `goal_drift_trigger_input_max_chars` | `goldfive.drift.goals:_GOAL_DRIFT_TRIGGER_INPUT_MAX_CHARS` | int | `[256, 16384]` | `2048` |
| `embed_runtime_failure_threshold` | `goldfive.drift._embed:_RUNTIME_FAILURE_THRESHOLD` | int | `[1, 50]` | `3` |
| `embed_cache_max` | `goldfive.drift._embed:_CACHE_MAX` | int | `[16, 16384]` | `512` |
| `processed_steer_ids_cap` | `goldfive.state_store:PROCESSED_STEER_IDS_CAP` | int | `[16, 16384]` | `256` |
| `default_llm_call_timeout_ms` | `goldfive.adapters._adk_plugin:DEFAULT_LLM_CALL_TIMEOUT_MS` | int | `[0, 7200000]` | `1800000` |
| `default_agent_max_output_tokens` | `goldfive.adapters.adk_llm_instrumentation:DEFAULT_AGENT_MAX_OUTPUT_TOKENS` | int | `[256, 131072]` | `16384` |
| `reflective_max_output_tokens` | `goldfive.drift_observer:DriftObserver.REFLECTIVE_MAX_OUTPUT_TOKENS` | int | `[512, 65536]` | `16384` |
| `goal_drift_task_boundary_min_interval_seconds` | `goldfive.drift_observer:DriftObserver._GOAL_DRIFT_TASK_BOUNDARY_MIN_INTERVAL_S` | float | `[0.0, 600.0]` | `10.0` |
| `parallel_executor_refine_failure_threshold` | `goldfive.executors.parallel:ParallelDAGExecutor.REFINE_FAILURE_THRESHOLD` | int | `[1, 10]` | `2` |
| `embedding_default_timeout_ms` | `goldfive.config:EmbeddingConfig.timeout_ms` | int | `[500, 600000]` | `10000` |
| `judge_default_timeout_ms` | `goldfive.config:JudgeConfig.timeout_ms` | int | `[500, 600000]` | `10000` |
| `agent_call_timeout_ms_default` | `goldfive.config:AgentConfig.call_timeout_ms` | int | `[1000, 7200000]` | `120000` |

> **`default_llm_call_timeout_ms` range starts at `0` on purpose** — `0` disables the ADK plugin's LLM-call watcher entirely. It is the only knob whose lower bound is a "disable" sentinel rather than a floor. Note this manifest entry's default (`1800000` = 30 min) is the plugin's built-in backstop, which is DIFFERENT from `AgentConfig.call_timeout_ms`'s `120000` (2 min) operator default — the plugin ceiling is a pathological-hang backstop, the `AgentConfig` value is the latency SLO threaded in by `wrap()`.

### The AST liveness contract (#487) — READ THIS BEFORE ADDING A NUMERIC KNOB

`tests/test_optimization_manifest.py::test_numeric_mutations_have_live_runtime_consumers` statically walks every `.py` under `goldfive/` (skipping `pb/`) and counts **read** sites per attribute leaf-name via `ast`. A "read" is:

- a bare `Name` in `Load` context,
- an `Attribute` in `Load` context (`config.check_interval`, `self.MAX_OUTPUT_TOKENS`),
- a `getattr(obj, "leaf", ...)` call with a string-literal name.

Definitions (`Store`), import aliases, `__all__` strings, comments, and docstrings do NOT count. For every `numeric` manifest entry, it extracts the leaf name (`python_attr.partition(":")[2].split(".")[-1]`) and asserts `counts[leaf] > 0`. **A numeric knob whose `python_attr` points at a defined-but-never-read symbol FAILS the test.**

```python
# tests/test_optimization_manifest.py — the assertion core
for mut in manifest:
    if mut.kind != "numeric":
        continue
    leaf = mut.python_attr.partition(":")[2].split(".")[-1]
    if counts[leaf] == 0:
        dead.append(f"{mut.id} ({mut.python_attr})")
assert not dead, ...
```

This is exactly why `goal_drift_check_interval` was repointed from the dead `goldfive.drift.goals:GOAL_DRIFT_CHECK_INTERVAL` re-export to the live `goldfive.config:GoalDriftConfig.check_interval`. The self-check test `test_liveness_counter_flags_the_known_dead_constant` pins `counts["GOAL_DRIFT_CHECK_INTERVAL"] == 0` and `counts["GOAL_DRIFT_IDLE_SECONDS"] >= 1`.

> **The check is name-based, not consumer-exact.** A generic leaf like `timeout_ms` aliases across classes, so a shared leaf name can pass the liveness test even if the specific attribute is dead. It is a smoke test, not a proof. When you add a knob with a generic leaf, verify the actual consumer by grepping, not by trusting the green test.

### Other manifest self-tests you must keep green

- `test_prompt_mutations_match_live_python_attrs` / `test_numeric_mutations_match_live_python_attrs` — the manifest's `default` (numeric) or the prompt `.md` body must match the live Python value.
- `test_manifest_prompt_entries_and_shipped_prompts_are_a_bijection` — every prompt entry has a shipped `.md` and vice-versa.
- `test_manifest_covers_expansion_entries` / `test_manifest_size_target` — coverage-count guards.
- `test_manifest_covers_stall_watchdog_knob` — pins `stall_watchdog_timeout_seconds` → `goldfive.config:SteeringConfig.stall_timeout_s`, default `600.0`.
- `test_from_text_rejects_duplicate_ids` / `test_from_text_rejects_invalid_source_path` / `test_from_text_rejects_default_outside_range` — manifest self-validation.

### Adding a manifest knob — the checklist

1. Append the `[[mutation]]` entry to `manifest.toml` with `id`, `kind`, `source`, `python_attr`, and (numeric) `type`/`range`/`default` or (prompt) `required_placeholders`.
2. Point `python_attr` at a symbol a runtime consumer READS (grep to confirm; the liveness test enforces it for numerics).
3. For a prompt, ship the `.md` under `goldfive/optimization/prompts/` (bijection test).
4. Add the coverage row in `tests/test_optimization_manifest.py`.
5. Keep the `default` / prompt body in sync with the live value (the sync tests enforce it).

### The loader/validator API (`goldfive/optimization/manifest.py`)

zicato (and tests) drive the manifest through these entry points. You will touch them only if you change the manifest schema; know they exist so you keep them consistent with a schema change.

| Symbol | Purpose |
|---|---|
| `Manifest.load()` (classmethod, `manifest.py`) | Load the bundled `manifest.toml` via `importlib.resources` + `tomllib` into a `Manifest` of typed `Mutation` records. |
| `Manifest.from_text(text)` (classmethod) | Parse an arbitrary TOML string (used by the self-validation tests). Raises `ManifestLoadError` on malformed / duplicate-id / bad-source / default-out-of-range. |
| `Manifest.validate(updates)` | Check a batch of proposed `{id: value}` updates against per-entry constraints (numeric `range`/`type`; prompt `required_placeholders`; non-empty string body). Batches all errors into one `ValidationError`. |
| `Mutation` (`manifest.py`) | Frozen dataclass: `id`, `kind`, `source`, `python_attr`, `required_placeholders`, `type`, `range`, `default`, `tags`. |
| `ManifestLoadError` / `ValidationError` | The two error types — load-time (bad manifest) vs. proposal-time (bad update). Re-exported from `goldfive.optimization`. |

The prompt-override runtime is a SEPARATE module, `goldfive/optimization/prompts.py` (module-level functions, not `Manifest` methods):

| Function (`optimization/prompts.py`) | Purpose |
|---|---|
| `load(name)` | Return the canonical prompt text for a prompt name (from the `.md` on disk, or a bound override). Caches. |
| `bind(name, value)` | Install a prompt override (how zicato applies a swept/validated prompt at runtime). |
| `reset(name=None)` | Clear one override; `reset(None)` clears ALL overrides and the cache. |
| `available_prompts()` | The sorted tuple of every shipped prompt name. |

The schema regexes are strict: `_PYTHON_ATTR_RE` requires `module.path:Attr` (one dotted nesting level for classes); `_SOURCE_PROMPT_RE` requires `goldfive/optimization/prompts/<name>.md`; `_SOURCE_PYTHON_RE` requires `goldfive/<module>.py:<ATTR>`. A new entry that doesn't match these fails `from_text` at load.

### Numeric-knob validation rules (what `validate` rejects)

- value outside `[low, high]` inclusive → rejected.
- wrong type (a float where `type = "int"`) → rejected — EXCEPT a float with an integer value (`5.0` for an int knob) is accepted (`test_validate_accepts_float_with_integer_value_for_int_knob`).
- a boolean for a numeric knob → rejected (bool is not a valid numeric here).
- unknown `id` → rejected.

### Prompt-knob validation rules

- body missing any `required_placeholder` token → rejected.
- empty / whitespace-only body → rejected.
- non-string body → rejected.

A proposed prompt update that passes `validate` is then installable via `bind` — the round-trip is covered by `test_validated_prompt_update_is_installable_via_bind`.

---

## 17. Precedence rules, per family

The general rule is **explicit kwarg > config-object field > env var > built-in default**, but each family has a concrete resolution point:

| Family | Resolution point | Precedence detail |
|---|---|---|
| All seven sub-configs | `wrap(runtime=...)` vs `RuntimeConfig.from_env()` | If `runtime=` is passed, env is NOT read for those sub-configs. If omitted, `from_env()` reads env; env-missing → dataclass default. |
| `embedding` / `reasoning_drift` / `tool_loops` | `configure()` (process-global) | Installed by `wrap()`. Last `wrap()` in the process wins. `configure(None)` clears → detectors fall back to module constants. |
| `_embed.py` direct env reads | lazy `_try_load_openai_backend` | Only used when no `configure()` ran; same env names, so identical surface. |
| `agent` / `steering` / `goal_drift` / `judge` | Threaded per-Runner into adapter / steerer | Per-Runner; no process-global state. |
| LLM and judge routing | `wrap()` | Built-in judges use `judge_call_llm=` > `call_llm=` > `JudgeConfig.base_url` > detected tree LLM. `judge_model=` overrides the judge model name. The planner and goal deriver stay on `call_llm` or the detected tree LLM. |
| `fail_fast_on_revision_rejection` | `Runner.__init__` | kwarg `True`/`False` > env `GOLDFIVE_FAIL_FAST_REVISION_REJECTION` (only when kwarg is `None`) > default `False`. |
| `fail_fast_on_invoke_cancel` | `SequentialExecutor.__init__` | kwarg `True`/`False` > env `GOLDFIVE_FAIL_FAST_ON_INVOKE_CANCEL` (only when kwarg is `None`) > default `False`. |
| `strict_state_ownership` | `Runner.run` | Runner kwarg `True`/`False` > tri-state env/pytest fallback when the kwarg is `None`. `wrap()` passes the resolved `RuntimeConfig` field. |
| Capability Rules A and C | ADK plugin → detector | `SteeringConfig` field > direct-detector env fallback > default `False`. |
| `report_awaiting_approval` timeout | handler | explicit positive `timeout_ms` arg > `SteeringConfig.approval_default_timeout_ms` > `DEFAULT_APPROVAL_TIMEOUT_MS` (when config value non-positive). |
| `pause_escalate_deadline_s` | executor | config value (`None` = block forever for L4); Level 5 falls back to `DEFAULT_TERMINATE_PAUSE_DEADLINE_S` when `None`. |
| `AgentConfig.max_output_tokens` | ADK plugin `before_model_callback` | ratchet-DOWN only: a smaller sub-agent / ADK value wins over the config ceiling. |
| `GOLDFIVE_STRICT_STATE_OWNERSHIP` | `_state_audit.py` | per-Runner typed value > explicit env value > auto (`pytest in sys.modules`). |

### Worked precedence resolutions

**`off_topic_distance_threshold`** (a reasoning threshold, process-wide via `configure`):
1. `wrap(runtime=RuntimeConfig(reasoning_drift=ReasoningDriftConfig(off_topic_distance_threshold=0.6)))` → `0.6` installed via `reasoning.configure`.
2. Else `wrap()` with no `runtime=`, `GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE=0.55` set → `RuntimeConfig.from_env()` reads `0.55`, installed via `configure`.
3. Else nothing set → `configure` installs the dataclass default `0.7`; detector helper `_off_topic_distance_threshold()` returns it.
4. If `configure(None)` was called (test teardown) → helper falls back to the module constant `OFF_TOPIC_DISTANCE_THRESHOLD = 0.7`.

**`observation_only`** (per-Runner, on the steerer):
1. Custom `steerer=DefaultSteerer(steering_config=SteeringConfig(observation_only=False))` → active.
2. Else `wrap(runtime=RuntimeConfig(steering=SteeringConfig(observation_only=False)))` → threaded into the default steerer → active.
3. Else `GOLDFIVE_STEER_OBSERVATION_ONLY=0` with no `runtime=` → `from_env` reads `False` → active.
4. Else default → `True` (passive). At runtime always read via `is_active_steering()`.

**`fail_fast_on_revision_rejection`** (kwarg-with-env-fallback):
1. `Runner(..., fail_fast_on_revision_rejection=True)` → `True` (kwarg wins even if env says `0`).
2. `Runner(..., fail_fast_on_revision_rejection=False)` → `False` (kwarg wins even if env says `1`).
3. `Runner(...)` with the kwarg omitted (`None`) and `GOLDFIVE_FAIL_FAST_REVISION_REJECTION=1` → `True`.
4. Omitted + env unset → `False`.

**`report_awaiting_approval` timeout** (per-call):
1. Agent calls `report_awaiting_approval(timeout_ms=30000)` → `30000` (explicit positive wins).
2. Agent omits `timeout_ms` (or passes `<= 0`), steerer carries `SteeringConfig(approval_default_timeout_ms=120000)` → `120000`.
3. Config value non-positive / steerer carries no `SteeringConfig` → `DEFAULT_APPROVAL_TIMEOUT_MS = 600000`.

### The `configure()` last-Runner-wins caveat, spelled out

`goldfive.drift.reasoning`, `.tool_loops`, and `._embed` each hold a module-level `_CONFIG` (or equivalent). `configure(config)` overwrites it globally. In a process running two Runners with **different** reasoning / tool-loop / embedding thresholds, the second `wrap()` clobbers the first's config for BOTH Runners — because detector call sites read the shared module global, not a per-Runner object. This is a documented tradeoff (see `ReasoningDriftConfig` docstring in `config.py`). Do NOT "fix" it by threading per-call config through the detectors without the design conversation — the follow-up plan is to move the config onto `Session` and read per-session, which is a larger change than it looks.

### The three process-wide `configure()` functions

| Function | Installs | Reset (test teardown) |
|---|---|---|
| `goldfive.drift.reasoning.configure(config)` | `ReasoningDriftConfig` → module `_CONFIG` | `configure(None)` → helpers fall back to module constants |
| `goldfive.drift.tool_loops.configure(config)` | `ToolLoopConfig` | `configure(None)` |
| `goldfive.drift._embed.configure(config)` | `EmbeddingConfig` (base_url/model/api_key/timeout/breaker cooldown) | `configure(None)` |

Each is idempotent (last call wins) and takes `None` to clear the override — the documented test-teardown path so one test's thresholds don't leak into the next. The `configure` calls happen at the TOP of `wrap()` (before the steerer branch), so they run even when you pass your own `steerer=`. If you construct detectors directly in a test (bypassing `wrap()`), call `configure()` yourself or the detector reads the module constants. The reasoning detectors read through helper functions (`_off_topic_distance_threshold()`, `_looping_hash_window()`, etc.) that consult `_CONFIG` first — never read the module constant directly at a detector call site or you defeat the override.

---

## 18. Frozen / sign-off-gated defaults ("safe defaults")

Some defaults are **not free to change**. Changing them requires explicit human sign-off because they are bench-gated or owned by the unmerged agency-preservation branch.

### FROZEN — do not change the default without human sign-off

| Knob | Current default | Why frozen |
|---|---|---|
| `SteeringConfig.observation_only` | `True` | The production passivity default. Flipping it to active steering globally is step 13b of the agency-preservation roadmap — LOCKED on explicit user sign-off pending a three-arm bench run + measurement-gated flip. Changing the DEFAULT (not an operator setting it per-Runner) is a sign-off decision. |
| `SteeringConfig.stall_watchdog_enabled` | `False` | Flag-gated producer (#487). Default-off until validated; enabling by default changes the `TASK_TIMEOUT` signal volume for everyone. |
| `SteeringConfig.descriptive_growth_enabled` | `False` | Behind the flag (goldfive#423 PR 2). A default flip (the planned PR 4) is a deliberate validated decision, not a casual edit. |
| `ReasoningDriftConfig.fallback_to_content_when_no_reasoning` | `False` | Opt-in lossy signal (goldfive#263); default-off so the behaviour change stays opt-in. |

### Bench-frozen / roadmap-owned (context, not a knob you flip here)

The **agency-preservation branch** (#453-#474, UNMERGED) owns Stages 1-3 behind its own default-OFF flags (`plan_mode=forecast`, `signal_channel=legacy`, `observation_only=True`, `signal_telemetry=False`). Those flags do NOT exist on `main` — do not document them as `main` config, and do not copy their defaults into `config.py`. Step 13b (bench + measurement-gated default flips + hard deletions) is locked on explicit user sign-off. See `17-invariants-hazards-history.md` and MEMORY `project_agency_preservation_roadmap.md`.

### Rule of thumb

Before changing ANY default in `config.py`: is the field a behaviour gate (`observation_only`, a `*_enabled` flag, a threshold that changes signal volume for all users)? If yes, it needs sign-off. Tuning a per-Runner value in your own code / env is always fine; changing the shipped DEFAULT is the gated action.

### Known future / deferred config work (do NOT present as current)

These are on the roadmap but NOT on `main`. Do not add config fields for them speculatively, and do not document them as existing knobs.

- **Judge windowing / cadence expansion** — a richer judge-scheduling surface (beyond `max_concurrent_judges` + `check_interval`) is blocked on a judge regression harness. Until that lands, `max_concurrent_judges` (concurrency cap) and `goal_drift_config.check_interval` (turn cadence) are the only judge-scheduling knobs.
- **Evidence-ledger replacement of the stacked `handle_drift` suppression gates** — the ~7 stacked suppression gates would become a single evidence ledger, but this is blocked on the agency-preservation branch-merge decision. No config surface for it exists on `main`.
- **Twin-refine-pipeline extraction** — also blocked on the agency-preservation merge.
- **Session-scoped detector config** — the fix for the `configure()` last-Runner-wins limitation (moving reasoning/tool-loop config onto `Session`) is #225's documented follow-up; not started. Until then, accept process-wide `configure()`.
- **Agency-preservation branch flags** (`plan_mode`, `signal_channel`, `signal_telemetry`, and that branch's own `observation_only` semantics) live on the UNMERGED `agency-preservation` branch behind default-OFF flags. They do NOT exist in `main`'s `config.py`. Main-side code must not copy them; doc text must not claim they exist here.

---

## 19. Worked example: add a new knob end-to-end

This is the exact, ordered procedure for adding one new tunable — a hypothetical `SteeringConfig.max_pending_nudges: int = 5` that caps how many nudges may queue on `session.pending_nudges`. Follow every step; skipping one produces a silently-dead knob (invariant 2).

**Step 1 — add the dataclass field.** In `goldfive/config.py`, `SteeringConfig`:

```python
#: Cap on the number of Level-2 nudges that may queue on
#: ``session.pending_nudges`` before the oldest is evicted. Env:
#: ``GOLDFIVE_STEER_MAX_PENDING_NUDGES``.
max_pending_nudges: int = 5
```
Place the `#:` docstring ABOVE the field so it renders as the attribute doc and names the env var.

**Step 2 — add the env reader in `from_env`.** Same class, inside `SteeringConfig.from_env`, add to the `return cls(...)`:

```python
max_pending_nudges=_read_int_env(
    "GOLDFIVE_STEER_MAX_PENDING_NUDGES",
    defaults.max_pending_nudges,
),
```
Use `_read_int_env` because it is a positive-int count. Do NOT write `int(os.environ.get(...))`.

**Step 3 — document the env var in the `from_env` docstring.** Add a bullet to the "Env surface:" list so operators can discover it without reading code.

**Step 4 — wire the runtime consumer.** The field is inert until something READS it. Find the nudge-enqueue site (`DefaultSteerer._dispatch_nudge` in `goldfive/steerer.py`) and read it through the config object:

```python
cap = self._steering_config.max_pending_nudges   # NOT getattr(..., 5)
if len(session.pending_nudges) >= cap:
    session.pending_nudges.pop(0)
```
Read through the real `SteeringConfig` instance — never `getattr(self._steering_config, "max_pending_nudges", 5)` with a local default (it forks `config.py`'s default; invariant 3).

**Step 5 — decide: should zicato tune it?** If yes, add a `[[mutation]]` to `manifest.toml`:

```toml
[[mutation]]
id = "steer_max_pending_nudges"
kind = "numeric"
source = "goldfive/config.py:SteeringConfig.max_pending_nudges"
python_attr = "goldfive.config:SteeringConfig.max_pending_nudges"
description = "Cap on queued Level-2 nudges before oldest eviction."
type = "int"
range = [1, 50]
default = 5
tags = ["config", "steerer", "nudge"]
```
Point `python_attr` at the dataclass field — but only because Step 4 makes a runtime consumer READ the leaf `max_pending_nudges`. If nothing reads it, the AST liveness test fails.

**Step 6 — add the manifest coverage row + a liveness self-check** in `tests/test_optimization_manifest.py` (an entry in the expansion coverage list, and if the leaf name is generic, an explicit assertion that `counts["max_pending_nudges"] >= 1`).

**Step 7 — decide: is the default frozen?** Is `max_pending_nudges` a behaviour gate that changes signal volume for everyone? A queue cap that only bounds memory is fine to pick a sensible default for. If it were a `*_enabled` flag or `observation_only`-class gate, it would need sign-off (§18).

**Step 8 — run the guards:**
```bash
uv run pytest -q tests/test_optimization_manifest.py tests/test_wrap_runtime_config.py tests/test_steerer.py
ruff check .
```

**Step 9 — sanity-check byte-identical default.** With no env and no `runtime=`, `wrap(tree)` must behave as before your change EXCEPT for the new cap taking effect at its default. If your default changes existing behaviour, confirm that is intended.

### The same procedure for a module-constant knob (not on a config object)

If the knob is a plain module constant (like a reasoning threshold), the steps differ:

1. Add the constant in the owning `drift/` module.
2. Add a field to the matching sub-config (`ReasoningDriftConfig`) whose default equals the constant.
3. Add a lookup helper in the module (`_my_threshold()` that returns `_CONFIG.my_field` when `_CONFIG` is set, else the constant) — mirror the existing `_off_topic_distance_threshold()` pattern.
4. Read through the helper at the detector call site (never the bare constant, or the config override won't apply).
5. Env reader in the sub-config's `from_env`.
6. Manifest entry pointing at the CONSTANT (that is the optimizer's `setattr` target) — and confirm a runtime consumer reads its leaf name.

The split (constant + config field + helper) exists because the detectors run in hot loops and the helper keeps the config/no-config branch obvious. Do not collapse it by reading `_CONFIG` inline everywhere.

---

## 20. Config recipes (copy-paste, both env and typed)

Each recipe shows the env-var form (for zero-code deploys) AND the typed-config form (for programmatic control). All are per-Runner unless noted process-wide.

### Route goldfive's judges to a separate cheap endpoint

Keep the agent tree billing against a cloud model, run the two drift judges on a local llama.cpp / Ollama box.

```bash
export GOLDFIVE_JUDGE_BASE_URL=http://localhost:8080/v1
export GOLDFIVE_JUDGE_MODEL=qwen2.5-7b
```
```python
from goldfive.config import RuntimeConfig, JudgeConfig
runner = goldfive.wrap(tree, runtime=RuntimeConfig(
    judge=JudgeConfig(base_url="http://localhost:8080/v1", model="qwen2.5-7b"),
))
```
This suppresses the named-model WARNING (you made a deliberate choice) and registers a Runner close-hook so the judge HTTP client tears down on `runner.close()`.

Callers that already manage a judge client can inject its callable
directly. The planner and goal deriver continue to use `agent_call_llm`:

```python
runner = goldfive.wrap(
    tree,
    call_llm=agent_call_llm,
    model="agent-model",
    judge_call_llm=judge_call_llm,
    judge_model="judge-model",
)
```

Goldfive does not register a judge close hook for `judge_call_llm` in
this form. The code that created the callable must close its client.

### Turn OFF the reasoning judge, keep deterministic detectors only

```bash
export GOLDFIVE_DRIFT_REASONING_MODE=embedding
```
```python
RuntimeConfig(reasoning_drift=ReasoningDriftConfig(mode="embedding"))
```
`mode="off"` disables reasoning-drift entirely; `mode="both"` runs judge AND embedding.

### Non-thinking model (Gemma / base model): recover a reasoning signal

```bash
export GOLDFIVE_DRIFT_FALLBACK_TO_CONTENT=1
```
```python
RuntimeConfig(reasoning_drift=ReasoningDriftConfig(fallback_to_content_when_no_reasoning=True))
```
Feeds the response body into `observe_reasoning` when the model emits no `<think>` stream. Lossy but strictly better than silent disarm.

### Tighten tool-loop detection for a chatty tree

```bash
export GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD=2
export GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY=critical
```
`name_axis_max_severity=critical` restores the legacy uncapped behaviour (fires CRITICAL on same-name-varied-args without exact-repeat corroboration). Leave it `info` unless you have a specific need.

### Enable the stall watchdog (catch wedged / idle runs)

```bash
export GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED=1
export GOLDFIVE_STEER_STALL_TIMEOUT_S=300
```
```python
RuntimeConfig(steering=SteeringConfig(stall_watchdog_enabled=True, stall_timeout_s=300.0))
```
Under `observation_only` (the default) the resulting `TASK_TIMEOUT` drift is telemetry-only. Remember this also arms the idle goal-judge trigger at `GOAL_DRIFT_IDLE_SECONDS` (300, separate knob).

### Bound an unattended pause-escalation so it can't hang forever

```bash
export GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S=1800
```
```python
RuntimeConfig(steering=SteeringConfig(pause_escalate_deadline_s=1800.0))
```
On expiry the executor aborts the run with `RunAborted` carrying the escalation lineage. `None` (default) blocks forever at Level 4; Level 5 TERMINATE always has a deadline (`DEFAULT_TERMINATE_PAUSE_DEADLINE_S=600` fallback).

### Graduate a Runner to ACTIVE steering (per-Runner, NOT a default flip)

```bash
export GOLDFIVE_STEER_OBSERVATION_ONLY=0
```
```python
RuntimeConfig(steering=SteeringConfig(observation_only=False))
```
This is how the ~90 active-mode tests opt in (#488). It is fine per-Runner; changing the shipped DEFAULT is the frozen action (§18).

### Judge-only benchmark run (native agent, judges armed, zero planning spend)

```python
runner = goldfive.wrap(tree, judge_only=True, call_llm=my_judge_llm)
```
Do NOT use `observation_only` for this — that still burns planning/goal-derive/refine LLM calls. See §10.

### Load from env, then tweak one field (the blessed mutation pattern)

```python
cfg = RuntimeConfig.from_env()
cfg.goal_drift.check_interval = 2      # judge more often for a debug run
cfg.reasoning_drift.off_topic_distance_threshold = 0.6
runner = goldfive.wrap(tree, runtime=cfg)
```
Dataclasses are mutable by design (`config.py` module docstring). For a snapshot, `dataclasses.replace(cfg, ...)`.

### Restore the legacy 30-minute LLM-call backstop (slow hardware / long synthesis)

```bash
export GOLDFIVE_AGENT_CALL_TIMEOUT_MS=1800000
```
The default `120000` (2 min) is a latency SLO, not a hang ceiling. On very slow local models it can fire `LLM_CALL_TIMEOUT` on healthy long calls (telemetry-only under `observation_only`, but noisy).

### Drop one built-in judge (agent legitimately makes no tool calls)

```python
from goldfive.builtin_judges import BuiltinJudge
runner = goldfive.wrap(tree, disable_judges=[BuiltinJudge.TOOL_ERROR])
```
Keeps the full default judge set minus the named one. Do NOT combine with `judges=` (that raises `TypeError`). Unrecognised names are silently ignored (forward-compatible).

### Register a custom judge alongside the built-ins

```python
runner = goldfive.wrap(tree, judges=[
    goldfive.builtin_judges.reasoning_drift(),
    MyRubricJudge(rubric="..."),
])
```
An explicit `judges=` list REPLACES the default set — re-list the built-ins you want to keep. Wire your custom judge's LLM yourself; `JudgeConfig` does not reach `judges=` instances.

### Opt into a context-edit rule

```bash
export GOLDFIVE_STEER_CONTEXT_EDITOR_RULES=prune_cancelled_reasoning
```
```python
RuntimeConfig(steering=SteeringConfig(context_editor_rules=["prune_cancelled_reasoning"]))
```
`None` / `[]` leaves the `ContextEditor` unwired (zero overhead). Unknown rule names are logged and dropped at registration. Per-rule so a single rule's regression can be bisected.

### Whole-deployment env baseline (no code, all Runners in the process)

```bash
export GOLDFIVE_JUDGE_BASE_URL=http://localhost:8080/v1
export GOLDFIVE_JUDGE_MODEL=qwen2.5-7b
export GOLDFIVE_DRIFT_REASONING_MODE=judge
export GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED=1
# observation_only stays True (frozen default) — do NOT set it to 0 process-wide
# unless every Runner in the process is meant to actively steer.
```
Every `wrap(tree)` with no `runtime=` reads these via `RuntimeConfig.from_env()`. Remember the `configure()`-installed families (embedding / reasoning / tool_loops) are last-Runner-wins process-wide.

---

## 21. Common mistakes (concrete wrong edits)

**Mistake: adding a config field but no env reader.**
You add `SteeringConfig.my_new_knob: bool = False` but forget to add the read in `SteeringConfig.from_env`. Now `GOLDFIVE_STEER_MY_NEW_KNOB` silently does nothing and `wrap()`-with-env callers can't set it.
→ **Correct:** add `my_new_knob=_read_bool_env("GOLDFIVE_STEER_MY_NEW_KNOB", defaults.my_new_knob)` to `from_env`, AND document the env var in the `from_env` docstring's "Env surface" list.

**Mistake: hand-rolling an env parse.**
```python
# WRONG
self.my_flag = os.environ.get("GOLDFIVE_STEER_MY_FLAG", "0") == "1"
```
This ignores the `true/yes/on/y/t` literals, doesn't log on typos, and diverges from every other bool knob.
→ **Correct:** `_read_bool_env("GOLDFIVE_STEER_MY_FLAG", default)`. (The two intentional exceptions — the `Runner`/`SequentialExecutor` `== "1"` fail-fast reads and the `context_editor_rules` comma-split — are documented above; do not add a third without a reason.)

**Mistake: reading a flag via `getattr` with a local default.**
```python
# WRONG — the local default silently diverges from config.py's default
if getattr(self._config, "observation_only", True):
    ...
```
If `config.py`'s default ever changes, this fork keeps the stale one. Worse for `observation_only`: reading the field at all bypasses the sanctioned predicate.
→ **Correct:** resolve through the dataclass field on a real `SteeringConfig`, and for `observation_only` specifically read via `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)` — never the raw field at an injection site.

**Mistake: adding a numeric manifest knob pointing at a dead constant.**
You add a `[[mutation]]` with `python_attr = "goldfive.drift.goals:GOAL_DRIFT_CHECK_INTERVAL"`. The default-sync test passes (the constant exists with the right value), but the knob is DEAD — nothing reads it, so zicato mutating it changes nothing.
→ **Correct:** point `python_attr` at the symbol a runtime consumer reads (here `goldfive.config:GoalDriftConfig.check_interval`). `test_numeric_mutations_have_live_runtime_consumers` will fail your PR otherwise.

**Mistake: adding a `stall_*`-style knob without a liveness consumer.**
You add `SteeringConfig.my_timeout_s` and a manifest entry but never wire it into the watchdog / a detector. The liveness test flags a generic leaf as "read" (aliasing) and you ship a no-op knob.
→ **Correct:** grep for the actual read site of your attribute leaf; confirm a consumer reads it before trusting the green liveness test. Wire the consumer FIRST.

**Mistake: assuming an env var override applies when you passed `runtime=`.**
You pass `wrap(tree, runtime=RuntimeConfig())` (all defaults) and also set `GOLDFIVE_DRIFT_OFF_TOPIC_DISTANCE=0.5` in the env, expecting `0.5`. It stays `0.7` — an explicit `runtime=` skips `from_env()` entirely.
→ **Correct:** either omit `runtime=` (env is read) or set the field on the `RuntimeConfig` you pass (`RuntimeConfig(reasoning_drift=ReasoningDriftConfig(off_topic_distance_threshold=0.5))`), or build from env then tweak: `cfg = RuntimeConfig.from_env(); cfg.reasoning_drift.off_topic_distance_threshold = 0.5`.

The six legacy fields listed in §1 are the exception: their `None` defaults
still delegate to the environment at the low-level consumer. Give those fields
a concrete value when an explicit runtime must be independent of ambient state.

**Mistake: expecting two Runners in one process to have independent reasoning/tool-loop/embedding thresholds.**
`configure()` is process-global; the second `wrap()` clobbers the first.
→ **Correct:** run them in separate processes, or accept last-Runner-wins, or (if this becomes a real need) do the `Session`-scoped-config redesign — not an ad-hoc per-call thread.

**Mistake: passing your own `steerer=` and expecting `runtime=`'s steering config to apply.**
`wrap()` only threads the sub-configs into the DEFAULT steerer it builds. An explicit `steerer=` is used verbatim.
→ **Correct:** construct your steerer with `steering_config=`, `goal_drift_config=`, `tool_loop_config=`, `reasoning_drift_config=` yourself.

**Mistake: flipping `observation_only=False` as the shipped default to "make steering work in tests".**
`observation_only=True` is FROZEN and ~90 tests explicitly opt into active mode by passing `observation_only=False` per-Runner (#488).
→ **Correct:** in a test, pass `SteeringConfig(observation_only=False)` to that Runner. Never change the module default.

**Mistake: adding a knob to `config.py` but not to the manifest, then wondering why zicato can't tune it.**
The typed config and the manifest are separate surfaces. A field on `SteeringConfig` is operator-tunable; it is NOT optimizer-tunable until it has a `manifest.toml` entry.
→ **Correct:** if zicato should sweep it, add the manifest entry per §16's checklist. If it's operator-only, that's fine — but say so.

**Mistake: using `int(os.environ.get("GOLDFIVE_TOOL_LOOP_WINDOW", "10"))` and crashing on a typo'd env.**
A bad value (`"ten"`) raises `ValueError` at import/init time — the worst kind of failure, because it takes down every Runner in the process.
→ **Correct:** `_read_int_env("GOLDFIVE_TOOL_LOOP_WINDOW", 10)` degrades to the default with a debug log; the process stays up.

**Mistake: pointing a manifest `python_attr` at a re-export alias.**
`goldfive/drift/goals.py` re-exports `GOAL_DRIFT_CHECK_INTERVAL` for back-compat. It exists and has the right value, so the default-sync test passes — but no runtime code reads it (the live path is `GoalDriftConfig.check_interval`). The AST liveness test fails.
→ **Correct:** grep the leaf name across `goldfive/` and point at the symbol with a nonzero read count.

**Mistake: treating `stall_timeout_s` and `GOAL_DRIFT_IDLE_SECONDS` as the same knob.**
They are two different thresholds both gated by `stall_watchdog_enabled`. `stall_timeout_s` (600) drives `TASK_TIMEOUT`; `GOAL_DRIFT_IDLE_SECONDS` (300) drives the idle goal-judge trigger.
→ **Correct:** set the one you mean. Enabling the watchdog arms both.

**Mistake: assuming `_read_optional_float_env` and `_read_float_env` behave the same on `0`.**
`_read_float_env("X", 5.0)` on `X=0` returns `0.0`. `_read_optional_float_env("X", 5.0)` on `X=0` returns `None` (disable). Using the wrong one for `pause_escalate_deadline_s` would make `0` mean "deadline of zero seconds" instead of "disabled".
→ **Correct:** `float | None` "disable-on-nonpositive" fields use `_read_optional_float_env`; plain floats use `_read_float_env`.

**Mistake: adding a fourth inline `os.environ` read because "the fail-fast knobs do it".**
The `== "1"` reads in `Runner` and `SequentialExecutor` are the two documented exceptions (they are `bool | None` kwargs that only consult env when `None`). They are not a precedent for new bool env vars.
→ **Correct:** new bool env vars use `_read_bool_env`, which accepts the full literal set and logs on typos.

**Mistake: reading `observation_only` off the config to gate an injection.**
```python
# WRONG — bypasses the sanctioned predicate
if not self._steering_config.observation_only:
    self._cancel_invocation(...)
```
→ **Correct:** `if self.is_active_steering(): ...` (or `steering_is_active(steerer)` when you hold a maybe-steerer). Missing/None/raising must read as PASSIVE. This is the Wave 1-4 invariant enforced by #488.

**Mistake: mass-reformatting `config.py` with `ruff format` while editing.**
The repo is NOT ruff-format-clean; a format pass produces a huge unrelated diff and breaks review.
→ **Correct:** edit only the lines you need. `ruff check .` must pass, but do not run `ruff format`.

**Mistake: adding a scalar override kwarg to `DefaultSteerer` instead of extending the typed config.**
You want `my_new_scheduling_knob` configurable, so you add it as a bare `DefaultSteerer.__init__` kwarg like the legacy `goldfive_steer_threshold`. Now it has no env surface, no `RuntimeConfig` field, and `wrap()` can't thread it.
→ **Correct:** add the field to the relevant sub-config (`GoalDriftConfig` / `SteeringConfig` / `ReasoningDriftConfig`), thread it in `wrap()`, and env-read it. The legacy scalars are back-compat only; do not grow that surface.

**Mistake: assuming `judges=[]` disables detection.**
`judges=[]` only opts out of the `JudgementEmitted` envelope stream. The legacy hardcoded detector path still runs and still fires `DriftDetected`.
→ **Correct:** to actually turn OFF a detector, use its own config (`reasoning_drift.mode="off"`, `threshold="off"`, or `disable_judges=` for a named built-in), not an empty `judges=`.

**Mistake: changing a manifest `default` to match a new code value without checking the `range`.**
You bump `LLMPlanner.MAX_OUTPUT_TOKENS` to `70000` and update the manifest `default`, but the manifest `range` is `[512, 65536]`. `from_text` rejects the manifest at load (`test_from_text_rejects_default_outside_range`), breaking every test that calls `Manifest.load()`.
→ **Correct:** widen the `range` in the same edit if the new default falls outside it, and confirm the range still makes sense as an optimizer sweep bound.

---

## 22. Verification checklist

Run these after touching anything in this chapter's surface.

**1. Config + manifest unit tests:**
```bash
uv run pytest -q tests/test_optimization_manifest.py tests/test_wrap_runtime_config.py
```

**2. The AST liveness + default-sync tests specifically (fastest guard against a dead manifest knob):**
```bash
uv run pytest -q tests/test_optimization_manifest.py \
  -k "live_runtime_consumers or match_live_python_attrs or liveness_counter or points_at_live_config or stall_watchdog"
```

**3. Confirm every new field has an env reader (grep the field name appears in both the dataclass and `from_env`):**
```bash
grep -n "my_new_field" goldfive/config.py   # expect: the field def AND a line inside from_env()
```

**4. Enumerate the live env surface to sanity-check you didn't miss one:**
```bash
grep -rhoE "GOLDFIVE_[A-Z_]+" goldfive/ | sort -u
```

**5. Confirm no bespoke `os.environ` parse crept in (the only sanctioned raw reads are the fail-fast exact-`"1"` pair, the `context_editor_rules` split, strict-state tri-state resolution, capability-detector compatibility fallbacks, and `_embed`'s lazy backend/cooldown fallbacks):**
```bash
grep -rn "os.environ\|getenv" goldfive/ | grep -v "config.py\|_read_"
```

**6. If you touched steering/observation gating, run the passivity + ladder suites:**
```bash
uv run pytest -q tests/test_steerer.py tests/test_intervention_ladder.py tests/test_executor_control.py
```

**7. If you touched the manifest prompt bijection or a prompt `.md`:**
```bash
uv run pytest -q tests/test_optimization_manifest.py -k "bijection or prompt"
```

**8. Full suite + lint (must both be clean; the repo is NOT ruff-format-clean, so do not mass-reformat):**
```bash
uv run pytest -q          # ~30s, ~2912 passed / 61 skipped
ruff check .
```

**9. Byte-identical-default check:** a bare `wrap(tree)` with no `runtime=` and no `GOLDFIVE_*` env set must behave as if you constructed `RuntimeConfig()` — if you changed a dataclass default, that IS a behaviour change; confirm it is intended and not a frozen/sign-off-gated default (§18).

### Debugging: "I set the knob and nothing changed"

Work down this list in order — each is a real, previously-hit cause:

1. **Did you pass `runtime=` AND expect the env to apply?** An explicit `runtime=` skips `from_env()`. Either omit `runtime=` or set the field on the object you pass. (§17 precedence.)
2. **Is the field actually READ at runtime?** Grep the field/leaf name across `goldfive/`. A field with zero read sites is dead. For numeric manifest knobs, `test_numeric_mutations_have_live_runtime_consumers` catches this — run it.
3. **Did you pass your own `steerer=`?** Then `steering` / `goal_drift` / `tool_loops` / `reasoning_drift` steerer-side config was NOT threaded. Construct the steerer with the configs.
4. **Is it a `configure()` family and did another `wrap()` in the process clobber it?** Last-Runner-wins. Check for a second `wrap()`.
5. **Is the detector even running?** A reasoning threshold does nothing if `reasoning_drift.mode="judge"` (no embedding path) or `"off"`. An embedding config does nothing in `"judge"` mode.
6. **Did the env value parse?** Check logs for the "ignoring unknown …" WARNING (bool / threshold / mode readers) or a debug "ignoring non-integer …" (int/float readers). A malformed value silently used the default.
7. **Is it `observation_only` you're fighting?** If steering "doesn't act", the default `observation_only=True` is why — flip it per-Runner (§18), don't change the default.
8. **Is the knob gated behind another flag?** `stall_timeout_s` does nothing unless `stall_watchdog_enabled=True`; `descriptive_growth` Rule-C fallback only fires when `descriptive_growth_enabled=True`; the `ContextEditor` is unwired unless `context_editor_rules` is non-empty.
9. **Did you read it via `getattr(obj, "name", default)`?** The local default masks the config value if the attribute name is slightly off. Read the dataclass field directly.

---

## Appendix A. All config-object field defaults at a glance

Every field of every sub-config, in one table, for quick lookup. `F` in the "Frozen" column marks a default that requires human sign-off to change (§18).

| Sub-config | Field | Default | Frozen |
|---|---|---|---|
| `EmbeddingConfig` | `base_url` | `None` | |
| `EmbeddingConfig` | `model` | `""` | |
| `EmbeddingConfig` | `api_key` | `None` | |
| `EmbeddingConfig` | `timeout_ms` | `10_000` | |
| `EmbeddingConfig` | `breaker_cooldown_s` | `None` (effective `60.0`) | |
| `JudgeConfig` | `base_url` | `None` | |
| `JudgeConfig` | `model` | `""` | |
| `JudgeConfig` | `api_key` | `None` | |
| `JudgeConfig` | `timeout_ms` | `10_000` | |
| `ToolLoopConfig` | `window` | `10` | |
| `ToolLoopConfig` | `exact_threshold` | `3` | |
| `ToolLoopConfig` | `name_threshold` | `5` | |
| `ToolLoopConfig` | `alternating_threshold` | `5` | |
| `ToolLoopConfig` | `name_axis_max_severity` | `"info"` | |
| `ReasoningDriftConfig` | `mode` | `"judge"` | |
| `ReasoningDriftConfig` | `off_topic_distance_threshold` | `0.7` | |
| `ReasoningDriftConfig` | `intent_divergence_healthy_similarity` | `0.6` | |
| `ReasoningDriftConfig` | `intent_divergence_minor_similarity` | `0.4` | |
| `ReasoningDriftConfig` | `intent_divergence_warning_similarity` | `0.2` | |
| `ReasoningDriftConfig` | `looping_reasoning_similarity_threshold` | `0.9` | |
| `ReasoningDriftConfig` | `reasoning_cluster_similarity_threshold` | `0.75` | |
| `ReasoningDriftConfig` | `looping_reasoning_hash_window` | `5` | |
| `ReasoningDriftConfig` | `max_concurrent_judges` | `3` | |
| `ReasoningDriftConfig` | `fallback_to_content_when_no_reasoning` | `False` | F |
| `GoalDriftConfig` | `check_interval` | `5` | |
| `GoalDriftConfig` | `activity_window` | `10` | |
| `SteeringConfig` | `threshold` | `"warning"` | |
| `SteeringConfig` | `suppression_window_turns` | `3` | |
| `SteeringConfig` | `observation_only` | `True` | F |
| `SteeringConfig` | `capability_rule_a_enabled` | `None` (effective `False`) | |
| `SteeringConfig` | `capability_rule_c_enabled` | `None` (effective `False`) | |
| `SteeringConfig` | `context_editor_rules` | `None` | |
| `SteeringConfig` | `descriptive_growth_enabled` | `False` | F |
| `SteeringConfig` | `signal_telemetry` | `False` | |
| `SteeringConfig` | `cancel_inflight_scope` | `"user_and_safety"` | |
| `SteeringConfig` | `signal_channel` | `"legacy_user_message"` | |
| `SteeringConfig` | `plan_mode` | `"forecast"` | |
| `SteeringConfig` | `legacy_ladder` | `False` | |
| `SteeringConfig` | `pin_assigned_task` | `False` | |
| `SteeringConfig` | `grace_window_turns` | `3` | |
| `SteeringConfig` | `approval_default_timeout_ms` | `600_000` | |
| `SteeringConfig` | `pause_escalate_deadline_s` | `None` | |
| `SteeringConfig` | `stall_watchdog_enabled` | `False` | F |
| `SteeringConfig` | `stall_timeout_s` | `600.0` | |
| `AgentConfig` | `max_output_tokens` | `16384` | |
| `AgentConfig` | `call_timeout_ms` | `120_000` | |
| `RuntimeConfig` | `fail_fast_on_revision_rejection` | `None` (effective `False`) | |
| `RuntimeConfig` | `fail_fast_on_invoke_cancel` | `None` (effective `False`) | |
| `RuntimeConfig` | `strict_state_ownership` | `None` (env/pytest fallback) | |

Module-level constants that back these (the "no config installed" fallbacks and the `configure`-overridable detector thresholds) live in `goldfive/drift/reasoning.py`, `goldfive/drift/tool_loops.py`, and `goldfive/drift/_embed.py`; the manifest §16 numeric table is the authoritative list of the tunable ones with their ranges.

## Appendix B. One-line summary of every section

- §1 — `RuntimeConfig` aggregate, install path, sub-config data flow, precedence in one sentence, master env index.
- §2 — the env-parse helpers; reuse them, never hand-roll.
- §3-§9 — the seven sub-configs, field by field, with env vars and readers.
- §10 — `wrap()` kwargs + interaction notes + the two WARNINGs you'll see.
- §11 — `Runner` kwargs; the `fail_fast_on_revision_rejection` env fallback.
- §12 — executor kwargs + the class-attribute knob table.
- §13 — `DefaultSteerer` kwargs; typed configs vs legacy scalars.
- §14 / §14b — low-level environment compatibility fallbacks + the `_llm.py` per-call knob surface.
- §15 — `GOLDFIVE_*` symbols that are NOT env vars.
- §16 — the manifest: full numeric + prompt tables, AST liveness contract, loader/validator API.
- §17 — precedence resolutions + the process-wide `configure()` caveat.
- §18 — frozen / sign-off-gated defaults; deferred future work.
- §19 — add-a-knob-end-to-end walkthrough.
- §20 — copy-paste config recipes.
- §21 — common mistakes.
- §22 — verification commands + the "my knob had no effect" debugging list.
- Appendix A — all sub-config field defaults in one table (with frozen markers).

When in doubt, the CODE ON MAIN wins over this chapter: `goldfive/config.py` is the authoritative default surface, `manifest.toml` is the authoritative optimizer-knob surface, and `tests/test_optimization_manifest.py` is the authoritative liveness/coverage guard. If this chapter and the code disagree, trust the code and fix the chapter.
