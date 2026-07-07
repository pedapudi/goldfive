# 07. Deterministic Drift Detection

## Read this chapter when...

- You are touching any detector under `goldfive/drift/` that is **not** an LLM judge — the tool-loop tracker, the reasoning hash/embedding detectors, the structural capability-mismatch detector, the stall watchdog, or the detector registry.
- You are tuning drift thresholds (`GOLDFIVE_TOOL_LOOP_*`, reasoning similarity bands, off-topic distance) and need to know which knob feeds which tier and what the corroboration cap does to your change.
- You are adding a **new** deterministic detector and need to know the `DriftEvent` contract, where to register it, and the negative-class / condition-key rules it must satisfy.
- You are debugging why a tool loop fired `LOOPING_REASONING` and not `LOOPING_TOOL_CALL`, or why the name axis "should have" escalated but stayed at INFO.
- You are chasing an embedding-related bug (circuit breaker, half-open recovery, "why is off-topic silent on my box").

For the **LLM judges** (reasoning-judge three-state classifier, goal-drift judge) see `08-llm-judges.md`. For where these detectors are *invoked from* (the ADK plugin callbacks, `DefaultSteerer.observe_reasoning`) see `05-adk-plugin.md` and `09-steering-ladder-and-gates.md`. For how the emitted `DriftEvent` is routed / gated / dispatched see `09-steering-ladder-and-gates.md`. For the `DriftEvent` wire mapping and telemetry labels see `12-events-sinks-telemetry.md`. For the config dataclasses that override these thresholds see `14-config-reference.md`.

## Files covered

| File | What it holds |
| --- | --- |
| `goldfive/drift/tool_loops.py` | `ToolLoopTracker` — exact/name/alternating axes, graduated meta/work tier tables, #484 corroboration cap, `(run_id, agent)` window keying. |
| `goldfive/drift/reasoning.py` | `analyze_reasoning*` pipeline entry + the embedding/hash detectors (`detect_looping_reasoning`, `detect_intent_divergence`, `detect_off_topic`, `detect_reasoning_cluster_tightening`) and `reasoning_hash`. |
| `goldfive/drift/_embed.py` | Optional embedding backends (HTTP + sentence-transformers), the runtime circuit breaker + #479 half-open recovery, `max_similarity` / `distance_to_topic`. |
| `goldfive/drift/capability_check.py` | Structural `detect_capability_mismatch` (Rules A/B/C) over `delegation_observed`. |
| `goldfive/drift/registry.py` | `DetectorConfig`, `register` / `get_config` / `list_registered`, `parse_json_response`, `truncate_for_observability`, `format_goals_block`. Post-#490 surface — `classify()` is deleted. |
| `goldfive/drift/__init__.py` | Structural classifiers `classify_tool_error`, `classify_refusal`, `classify_confabulation_risk`, `classify_stop_reason` + marker tables; lazy re-exports. |
| `goldfive/adapters/_adk_plugin.py` (`_run_stall_watchdog`) | The wall-clock stall watchdog (#487) — the `TASK_TIMEOUT` producer. Full detail in `05-adk-plugin.md`; treated here as a detector. |
| `goldfive/types.py` (`DriftEvent`, `DriftKind`, `DriftSeverity`) | The event dataclass every detector emits + the taxonomy. |

## Invariants that bind you here

These are the CANON hard invariants, specialised to this subsystem. Violating any of them is a defect regardless of whether tests pass.

1. **No regex/keyword heuristics for natural-language classification.** goldfive retired `_GENERIC_VERB_PREFIX_RE` (#166), `_FACTUAL_QUESTION_RE` (#167), the `CONFUSION` uncertainty-marker detector, and `_has_unreferenced_keyword` (#226/#230). Do NOT reintroduce anything shaped like "count English tokens / scan for phrases and emit a drift". Exact-equality and hash matching of *structured* data (tool `(name, args_hash)`, reasoning SHA-256) is allowed and is what most of this subsystem is built on. See §"Common mistakes" and `17-invariants-hazards-history.md`.
2. **Adaptive over predictive.** Every detector classifies **observed** facts (a tool call that already happened, a reasoning block already emitted, a delegation already bound). None of them predict what the agent will do. Do not add a detector that fires on a *predicted* future action.
3. **No prompt-cooperation contract.** These detectors observe the ADK stream; they never require the agent to call a goldfive tool or follow an instruction. `on_task_progress` resetting the tool-loop window is driven by the plugin observing a task transition, not by the agent "telling" goldfive it made progress.
4. **`observation_only=True` is strictly passive (Waves 1-4).** These detectors *emit* `DriftEvent`s unconditionally — passivity is enforced downstream in dispatch (see `09-steering-ladder-and-gates.md`). A detector must NEVER read the kill-switch itself, must NEVER cancel/mutate state, and must NEVER gate its own emission on `observation_only`. Emit the observation; let the steerer decide. The only sanctioned read of the kill-switch is `DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)`, and it lives in the dispatch path, not in a detector.
5. **Lifecycle gates need stable identity keys.** Condition ids are `sha1(f"{kind.value}|{task_id}|{agent_id}|{turn_id}")[:16]` (`goldfive/state_store.py::compute_condition_id`). Never build a detector whose `DriftEvent.current_task_id` / `current_agent_id` churns per observation (e.g. an LLM-minted id embedded in the task id) — it opens a fresh condition every fire and no gate ever engages.
6. **Protected KEEP surfaces.** `DriftKind.LOOPING_TOOL_CALL` and its ladder/promotion/planner rows are protected (#204/#206 history — see §"Why tool loops emit LOOPING_REASONING"). `PLAN_DIVERGENCE` machinery is #252-disabled but KEEP. Do not delete or "clean up" these without explicit human sign-off.

---

## The shape of a deterministic detector

Every detector in this subsystem is a pure(ish), side-effect-light function or method that:

1. Takes a best-effort view of some observed fact (a tool call, a reasoning string, a delegation, a tool response).
2. Returns `DriftEvent | None` (or `list[DriftEvent]` for the tool-loop tracker, which can co-emit an alternating INFO with a graduated hit).
3. Never dispatches control, never reads `observation_only`, never mutates the session except for narrowly-scoped one-shot bookkeeping (`session.reasoning_cluster_flagged`, the tool-loop ring buffer).

One-line facts to keep front of mind (each expanded below):

- Detectors **emit**; they never dispatch, cancel, or read `observation_only`.
- The tool-loop tracker emits `LOOPING_REASONING` (not `LOOPING_TOOL_CALL`) and stamps `detector_name="tool_loops"`.
- The name axis is INFO-capped without ≥2 exact-repeat corroboration; the exact axis is never capped.
- The tool-loop window keys on `(session.run_id, agent)`, accumulating across re-invocations.
- Embedding detectors are opt-in — unreachable in a default install, and `mode="judge"` doesn't select them anyway.
- Stamp `observed_revision_index` before any `await`; use a **stable** `current_task_id`.
- `GOAL_DRIFT` and deterministic-detector conditions resolve only at task-terminal; reasoning-pipeline kinds also resolve on an on-task verdict.

The two categories in this chapter:

- **Structural detectors** — deterministic, no LLM, no embeddings. `tool_loops`, `capability_check`, the `classify_*` functions in `drift/__init__.py`, and the stall watchdog. `DetectorConfig(uses_llm=False)`.
- **Embedding detectors** — deterministic given the encoder, but the encoder is an *optional* dependency. `detect_off_topic`, `detect_reasoning_cluster_tightening`, the cosine tier of `detect_looping_reasoning` and `detect_intent_divergence`. They degrade to "no signal" when the encoder is unreachable.

The **LLM judges** (`reasoning_judge`, `goals`) live in this directory too but are covered in `08-llm-judges.md`; they are the only two detectors with `DetectorConfig(uses_llm=True)`.

### Detector summary matrix

One row per deterministic detector in this chapter. `Embed?` = requires the embedding encoder; `Reg?` = self-registers via `registry.register`; `State?` = holds mutable state.

| Detector | Entry function | Kind(s) | Sev range | Embed? | Reg? | State? |
| --- | --- | --- | --- | --- | --- | --- |
| Tool loops | `ToolLoopTracker.observe_tool_call` | `LOOPING_REASONING` | INFO–CRITICAL | no | no¹ | yes (ring buffers) |
| Reasoning loop (cliff) | `detect_looping_reasoning` | `LOOPING_REASONING` | WARNING | hash no / cosine yes | no | reads history |
| Cluster tightening | `detect_reasoning_cluster_tightening` | `REASONING_CLUSTER_TIGHTENING` | INFO | yes | no | one-shot flag |
| Intent divergence | `detect_intent_divergence` | `INTENT_DIVERGENCE` | INFO–CRITICAL² | cosine yes / pattern no | no | no |
| Off-topic | `detect_off_topic` | `OFF_TOPIC` | WARNING | yes | no | no |
| Capability mismatch | `detect_capability_mismatch` | `CAPABILITY_MISMATCH` | CRITICAL | no | **yes** | no |
| Tool error | `classify_tool_error` | `TOOL_ERROR` | WARNING | no | no | no |
| Refusal | `classify_refusal` | `AGENT_REFUSAL` | INFO–CRITICAL | no | no | no |
| Confabulation | `classify_confabulation_risk` | `CONFABULATION_RISK` | INFO | no | no | no |
| Stop reason | `classify_stop_reason` | `CONTEXT_PRESSURE` | WARNING | no | no | no |
| Stall watchdog | `_run_stall_watchdog` | `TASK_TIMEOUT` | WARNING–CRITICAL | no | no | task-local |

¹ `tool_loops` is imported by `_ensure_registered` for parity but the tracker is instantiated directly by the plugin, not dispatched via the registry. ² Pattern-fallback path is flat WARNING.

### `DriftEvent` anatomy

`goldfive/types.py::DriftEvent` is the single event every detector emits. The load-bearing fields:

```python
# goldfive/types.py
@dataclasses.dataclass
class DriftEvent:
    kind: DriftKind
    severity: DriftSeverity
    detail: str = ""
    current_task_id: str = ""
    current_agent_id: str = ""
    raw: Any = None                       # the source object/text that triggered detection
    id: str = field(default_factory=_uuid_hex)
    trigger_input: str = ""               # short render of what the detector saw
    authored_by: str = ""                 # normalised to "goldfive" downstream
    suppressed_by_user_steer: bool = False
    observed_revision_index: int = 0      # #245 — plan revision at observation time
    detector_name: str = ""               # #480 — symbolic source, e.g. "tool_loops"
```

Field-by-field, what a detector author must get right:

| Field | Rule |
| --- | --- |
| `kind` | A `DriftKind` enum value. Determines the intervention-ladder row (see `09-steering-ladder-and-gates.md`). |
| `severity` | `INFO` / `WARNING` / `CRITICAL`. INFO routes to OBSERVE (telemetry-only). Do NOT emit CRITICAL casually — CRITICAL is the only tier that can reach cancel/pause in active mode. |
| `current_task_id` | Stamp it. Used to scope the intervention ladder's occurrence counter per task and to key condition-lifecycle resolution. Empty string is safe (unscoped) but weaker. |
| `current_agent_id` | The live agent whose activity produced the observation. For delegated sub-agents this is the sub-agent, not the coordinator. |
| `raw` | The original object/text. For the reasoning detectors this is the reasoning `text`; for tool-loops it is a structured `dict`; for `classify_tool_error` it is the tool-response event. |
| `trigger_input` | A short human-readable render of what the detector matched, for sinks that answer "why did goldfive flag this?" without re-fetching transcripts. Populated on autonomous drifts; empty on user-control drifts. |
| `observed_revision_index` | **Must** be stamped from `session.plan.revision_index` at the TOP of the detector, BEFORE any `await`. The dispatch-time freshness gate drops verdicts observed against a plan the reconciler has moved past (`09-steering-ladder-and-gates.md`). `0` means "unset / pre-#245" and the gate treats it as legacy (no-op). |
| `detector_name` | **Stamp it whenever your `kind` is not unique to your detector.** The canonical example: the tool-loop tracker emits `LOOPING_REASONING` — the *same kind* as the embedding reasoning-loop detector — so a kind-keyed telemetry lookup would misattribute every tool-loop fire. `tool_loops` stamps `detector_name="tool_loops"`. If your kind is unique, you may leave it empty (`_detector_name_for_drift` falls back to a kind-keyed table). Added #480. |

`DriftKind` (`goldfive/types.py`, near line 111) is a `StrEnum`. `DriftSeverity` has exactly three values: `INFO`, `WARNING`, `CRITICAL`. Use `goldfive.types.severity_rank()` when you need to compare severities ordinally.

### `DriftKind` values this subsystem produces

`DriftKind` is a large taxonomy (many values are minted by the steerer, the executor, or user control — not by detectors). The subset the deterministic detectors in this chapter emit:

| `DriftKind` | Producer (this chapter) | Typical severity | Notes |
| --- | --- | --- | --- |
| `LOOPING_REASONING` | `ToolLoopTracker` (tool loops) **and** `detect_looping_reasoning` (reasoning loops) | INFO / WARNING / CRITICAL | Shared kind — disambiguate by `detector_name`. |
| `REASONING_CLUSTER_TIGHTENING` | `detect_reasoning_cluster_tightening` | INFO | Embedding-only, one-shot per task. |
| `INTENT_DIVERGENCE` | `detect_intent_divergence` | INFO / WARNING / CRITICAL | Graduated by cosine; pattern fallback flat WARNING. |
| `OFF_TOPIC` | `detect_off_topic` | WARNING | Embedding-only. |
| `CAPABILITY_MISMATCH` | `detect_capability_mismatch` | CRITICAL | Structural; every fire cancels + refines. |
| `TOOL_ERROR` | `classify_tool_error` | WARNING | Structural, dict/attr shape. |
| `AGENT_REFUSAL` | `classify_refusal` | INFO / WARNING / CRITICAL | Tiered marker tables. |
| `CONFABULATION_RISK` | `classify_confabulation_risk` | INFO | Record-only; human decides. |
| `CONTEXT_PRESSURE` | `classify_stop_reason` | WARNING | Truncation stop reasons. |
| `TASK_TIMEOUT` | stall watchdog (`_run_stall_watchdog`) | WARNING / CRITICAL | Flag-gated, default OFF. |

Kinds you will see referenced but that are **not** produced here (covered in sibling chapters): `GOAL_DRIFT`, `OFF_TOPIC` / `JUSTIFIED_DEVIATION` from the LLM judge (`08-llm-judges.md`); `RUNAWAY_DELEGATION`, `LLM_CALL_TIMEOUT`, `HUMAN_INTERVENTION_REQUIRED`, `REFINE_VALIDATION_FAILED` from the plugin/steerer (`05-adk-plugin.md`, `09-steering-ladder-and-gates.md`); `PLAN_DIVERGENCE` (#252-disabled, KEEP) and `LOOPING_TOOL_CALL` (protected KEEP, no current producer). `USER_STEER` / `USER_CANCEL` / `USER_PAUSE` are user-control kinds.

### Data flow: which observation reaches which detector

Every detector in this chapter is reached from an ADK plugin callback (`05-adk-plugin.md`) or from `DefaultSteerer` (`09-steering-ladder-and-gates.md`). Knowing the entry point tells you where to add a call site and where to look when a detector "isn't firing".

| Detector | Reached from | Trigger |
| --- | --- | --- |
| `ToolLoopTracker.observe_tool_call` | ADK plugin `after_tool_callback` (and `before_tool_callback` for the observation) | Every tool dispatch the plugin sees. |
| `ToolLoopTracker.on_task_progress` | ADK plugin `after_tool_callback` | Acknowledged-success `report_task_*` response. |
| `analyze_reasoning*` (reasoning pipeline) | `DefaultSteerer.observe_reasoning` | A captured `reasoning_content` / `thinking` block from `after_model_callback`. |
| `detect_looping_reasoning` (always-on cheap loop) | `DefaultSteerer.observe_reasoning` upstream of the mode-selected pipeline | Every reasoning block, in every mode incl. `"off"`. |
| `detect_capability_mismatch` | `DefaultSteerer` on the `delegation_observed` signal | AgentTool delegation bound to a task (#253). |
| `classify_tool_error` | ADK plugin `after_tool_callback` | Tool response shape. |
| `classify_refusal` / `classify_stop_reason` | ADK plugin `after_model_callback` | LLM response text / finish reason. |
| `classify_confabulation_risk` | `DefaultSteerer` at invocation end | Task completed with zero tool calls + non-empty output. |
| stall watchdog (`TASK_TIMEOUT`) | ADK plugin `_run_stall_watchdog` background task | Liveness watermark silent past `stall_timeout_s`. |

The common exit for all of them is the same: the produced `DriftEvent`(s) are routed through `DriftObserver.handle_drift` (`drift_observer.py`), which emits `DriftDetected`, applies the freshness/PLAN_DIVERGENCE/authored-by entry guards, and runs the intervention ladder. **No detector dispatches control itself** — this is the passive-emit invariant made concrete.

### `raw` conventions across the subsystem

`DriftEvent.raw` holds the source that triggered detection. Conventions differ by detector; know them so you populate/read the right shape:

| Detector | `raw` holds |
| --- | --- |
| `ToolLoopTracker` | A structured `dict` (`mode`, `tool_name`, `count`, `tier`, `invocation_id`, …). |
| reasoning detectors (`detect_*`) | The reasoning `text` string. |
| `classify_tool_error` | The original tool-response event (dict or object). |
| `classify_refusal` | The original text/object passed in. |
| `classify_stop_reason` | The original stop-reason value/enum. |
| `capability_check` | Not set (`None`) — the detail string carries everything; no source object worth pinning. |
| `classify_confabulation_risk` | Not set — structural, detail carries the matched keyword. |
| stall watchdog | Not set — detail carries idle time + threshold. |

Rule of thumb: pin `raw` when a downstream consumer might want to re-inspect the exact source (tool-loop windows, reasoning text). Leave it `None` when the `detail` string is a complete account.

### DO / DON'T for `DriftEvent` construction

| DO | DON'T |
| --- | --- |
| Stamp `observed_revision_index` at the top of the detector before any `await`. | Read `session.plan.revision_index` after an LLM round-trip. |
| Stamp `detector_name` when your `kind` is shared with another detector. | Rely on the kind alone to attribute telemetry for a shared kind. |
| Use a stable `current_task_id` (plan task id). | Put an LLM-minted or per-call id in `current_task_id`. |
| Emit INFO for record-only signals (OBSERVE routing). | Emit CRITICAL unless the signal genuinely warrants cancel/pause. |
| Return `None` when there is no signal. | Return a placeholder / zero-severity `DriftEvent` to "mark that you ran". |
| Put rich detail in `detail` / `raw` / `trigger_input`. | Put rich context in a field the LLM will see (that surface is `{"status": "cancelled"}` only). |

### The freshness gate (#245), worked

`observed_revision_index` exists because a detector observes a plan state, then the verdict may arrive after the reconciler has moved the plan on. Concrete sequence:

1. Reasoning-judge detector observes a block against `session.plan.revision_index == 4`; it stamps `observed_revision_index=4` at the top, before the LLM `await`.
2. During the multi-second LLM round-trip, the reconciler installs a revised plan → `revision_index` becomes 5.
3. The judge's verdict (a `DriftEvent` stamped `4`) arrives at `handle_drift`.
4. The dispatch-time gate sees `4 < 5` (live) → emits `DriftDetected` for observability but **skips** the cancel + refine machinery: the verdict is stale against a plan-state the system already moved past.

`observed_revision_index=0` (unset / pre-#245 / legacy producers) is treated as "not stamped" — the gate is a no-op for it (keyed on a truthy stamp). User-authored drifts (`USER_STEER` / `USER_CANCEL` / `USER_PAUSE`) bypass the gate unconditionally. **The detector's only job here is to capture the number before any await** — the gate logic lives in the steerer (`09-steering-ladder-and-gates.md`). Synchronous detectors (tool loops, capability check) can't race an await, but they still stamp it so the gate has a consistent field to read.

### Deterministic detectors and decision telemetry (#480)

Every routed `DriftEvent` produces a `SteeringDecisionMade` telemetry record whose `detector_name` label attributes the decision. `DriftObserver._detector_name_for_drift(drift)` resolves it: it uses `drift.detector_name` when stamped, else a kind-keyed fallback table. This is exactly why the tool-loop tracker MUST stamp `detector_name="tool_loops"` — its kind (`LOOPING_REASONING`) is shared with the embedding reasoning-loop detector, so the kind-keyed fallback would misattribute every tool-loop fire to the reasoning path. #480 also fixed four label corruptions (`DriftEvent.detector_name`, `drift_dropped_stale`/`inflight` outcomes, the `capability_check` negative class, and `ReasoningJudgeInvoked` proto fields). If you add a detector whose kind is unique, you may skip the stamp; if it shares a kind, stamp it and add a `test_goldfive_drift_routing.py` assertion.

---

## Tool-loop detection (`goldfive/drift/tool_loops.py`)

This is the deepest deterministic detector and the one most likely to be mis-tuned. Read this section fully before touching any `GOLDFIVE_TOOL_LOOP_*` knob.

### What it observes and why

Post-steer replays on weaker models sometimes degenerate: the LLM emits the same `function_call` over and over without advancing any task. Empirically observed on Qwen runs where a single stuck invocation burned ~30 minutes before ADK's max-call budget tripped. The embedding reasoning-loop detector (`reasoning.py`) watches LLM *text* — it does NOT see the `function_call` stream, so tight tool loops that never repeat reasoning text slip through. `ToolLoopTracker` is the complementary detector: it sees **every** tool call the ADK plugin observes (reporting tools, AgentTool delegations, MCP tools, custom adapter-native tools).

Before #206 an args-aware `ToolLoopGuard` covered only the reporting-tool slice; it was retired. `ToolLoopTracker` is now the sole tool-loop detector.

### Public API

The tracker's surface you will call or test against:

```python
# goldfive/drift/tool_loops.py
class ToolLoopTracker:
    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,                       # 10
        exact_threshold: int = DEFAULT_EXACT_THRESHOLD,     # 3  (work-WARNING exact override)
        name_threshold: int = DEFAULT_NAME_THRESHOLD,       # 5  (work-WARNING name override)
        alternating_threshold: int = DEFAULT_ALTERNATING_THRESHOLD,  # 5
        name_axis_max_severity: str = DEFAULT_NAME_AXIS_MAX_SEVERITY, # "info"
    ) -> None: ...

    def observe_tool_call(
        self, *, invocation_id: str, agent_name: str, tool_name: str, args: Any,
        task_id: str = "", observed_revision_index: int = 0, session_run_id: str = "",
    ) -> list[DriftEvent]: ...

    def on_task_progress(
        self, *, invocation_id: str, agent_name: str, session_run_id: str = "",
    ) -> None: ...

    def clear(self) -> None: ...
    def buffer_size(self, *, invocation_id: str, agent_name: str, session_run_id: str = "") -> int: ...
```

The constructor validates every numeric kwarg (`> 0` or `ValueError`) and `name_axis_max_severity` (must be one of `{"info","warning","critical"}`). `observe_tool_call` is the hot path — call it on every tool dispatch; it returns the list of `DriftEvent`s to route. `buffer_size` is a test-introspection helper.

Module-level helpers you will also touch: `args_hash(args) -> str`, `load_thresholds_from_env() -> dict`, `thresholds_from_config(config) -> dict`, `resolve_thresholds() -> dict` (the precedence resolver), and `configure(config)` (installs a process-wide `ToolLoopConfig`).

### The window: keying and accumulation

Each tool call is recorded as a signature `(tool_name, args_hash(args))` in a bounded `deque` (`maxlen=window`, default 10) keyed on:

```
scope_key = session_run_id or invocation_id or ""
key       = (scope_key, agent_name or "")
```

- **`session.run_id` is the preferred scope key (#420).** When a coordinator re-invokes the same sub-agent multiple times within a session, each re-invocation gets a fresh `invocation_id`. Under the old invocation-keyed scheme, 20 tool calls spread across 11 invocations only saw ~2 per bucket and never tripped the CRITICAL threshold. Re-keying on `(session.run_id, agent_name)` accumulates the cumulative window across re-invocations of the same agent within one run.
- **Parallel sub-agents stay isolated** via `agent_name`.
- **Legacy callers** (tests, third-party adapters) that don't thread `session_run_id` fall back to `(invocation_id, agent_name)` — the pre-#420 contract.
- The ADK plugin threads `session.run_id` through `observe_tool_call(session_run_id=...)`. When you write a test, thread it too or you get the isolated-bucket behaviour.

`args_hash(args)` (`tool_loops.py`) is a stable 8-char md5 hex prefix over `json.dumps(args, sort_keys=True, default=str)`, falling back to `repr(args)` on serialisation failure. Its properties, which matter for exact-axis correctness:

- `sort_keys=True` — dict key ordering does not perturb the hash. `{"a":1,"b":2}` and `{"b":2,"a":1}` hash identically (same call, same signature).
- `default=str` — non-JSON values (dates, dataclasses) stringify rather than raise, so the hot path never throws.
- Empty / non-mapping payloads hash deterministically, so an argument-free tool called repeatedly still lights up the exact axis (identical empty args → identical hash → exact repeat).
- It is **not** cryptographic — md5 truncated to 8 hex chars. Collisions are astronomically unlikely at per-run tool-call volumes and a collision would only merge two distinct signatures into one exact bucket (a benign over-count, never a missed real loop).

This is exact-equality matching of **structured** data — the sanctioned deterministic primitive, not NL classification.

**Window reset on progress.** `on_task_progress(invocation_id=, agent_name=, session_run_id=)` unconditionally `.clear()`s the per-key buffer. The *policy* of when to call it lives OUTSIDE the tracker: the ADK plugin's `after_tool_callback` calls it only on an **acknowledged success** response from a `report_task_*` tool (`{"acknowledged": True, ...}` with no `error` key). This is the #192 fix — errored `report_task_started` retries with a bad `task_id` used to be exempted on the call alone, so an agent stuck getting 16 `missing_task_id` errors never tripped the detector. Now errored progress reports count as ordinary tool calls; only genuine success clears the window. (Direct callers — unit tests, alternate adapters — may still call `on_task_progress` when they have out-of-band knowledge of real progress; the tracker's reset is unconditional, only the plugin gates it.)

### The two axes

Two loop-shape signals are computed over the window on every call:

- **exact axis** — max count of any identical `(name, args_hash)` signature. Identical-args repeats are *definitionally* redundant work.
- **name axis** — max count of any same-`name`-any-args signature. Ambiguous: a stuck loop *or* healthy iteration (reading six different files with the same tool is doing the job).

Plus an independent **alternating** mode: an `A,B,A,B,A` cycle in the tail of length `alternating_threshold` (default 5). INFO-only, always. Suppressed when a graduated exact/name drift already fired on the same window (the weaker signal would be noise on top of the stronger).

### The graduated tier tables (meta vs work)

Not every tool loop is equal. `report_task_completed` retrying 3× is probably benign re-reporting (cheap, often idempotent). `web_developer_agent` retrying 3× is a work loop burning tokens. `_classify_tool_category(tool_name)` splits tools:

- **meta** — `tool_name.startswith(("report_task_",))` OR `tool_name in {"report_awaiting_approval"}`. Name-only, no args inspection, no registry lookup.
- **work** — everything else (including empty tool name).

Each category has its own three-tier table. Reproduced verbatim from `tool_loops.py`:

```python
# goldfive/drift/tool_loops.py
_META_THRESHOLDS = {
    "info":     {"exact": 3,  "name": None},
    "warning":  {"exact": 6,  "name": None},
    "critical": {"exact": 10, "name": None},
}
_WORK_THRESHOLDS = {
    "info":     {"exact": 3, "name": None},
    "warning":  {"exact": 3, "name": 5},
    "critical": {"exact": 6, "name": 7},
}
```

As a table (the "—" cells mean that tier/axis never fires):

| Category | Axis | INFO | WARNING | CRITICAL |
| --- | --- | --- | --- | --- |
| meta | exact | 3 | 6 | 10 |
| meta | name | — | — | — |
| work | exact | 3 | 3 | 6 |
| work | name | — | 5 | 7 |

Reading rules (implemented in `ToolLoopTracker._classify`):

1. For each **distinct tool name** in the window (not each signature), find the **highest** tier matched (CRITICAL > WARNING > INFO) and emit **one** drift at that severity. Do NOT cascade INFO + WARNING + CRITICAL on the same window.
2. The exact axis is checked first at each tier; the name axis only if exact didn't already classify at the same-or-higher tier (preserves the pre-#204 "exact preempts name on the same tool" suppression).
3. Category is determined by the tool being matched, not by other calls in the window.
4. Across tools, keep the candidate with the highest **emitted** severity (post-cap — see below), so a capped name-axis INFO on tool X never shadows tool Y's genuine exact-axis WARNING.

**Why these numbers.** Work-tier WARNING is backwards-compatible with the pre-#204 single-threshold detector: 3 identical work calls still fires WARNING; same-name 5-in-window still matches the WARNING tier (subject to the cap below). Work CRITICAL (6/7) is new — it escalates plan revision into cancel-reinvoke at higher counts. Meta WARNING is pushed from 3 out to 6 so benign `report_task_*` retries produce only an INFO drift (OBSERVE, no plan mutation) until the loop genuinely persists.

### The #484 corroboration cap (name-axis precision)

This is the most important recent change and the one you are most likely to break when tuning. The name axis counts same-`name`-any-args calls — ambiguous between a stuck loop and definitionally-healthy varied-args iteration. Because the window accumulates across re-invocations (#420), the old WARNING-at-5 / CRITICAL-at-7 name tiers false-positived on healthy varied-args bursts.

**The rule (#484):** a name-axis hit is emitted at its tier's severity **only if the window corroborates the loop hypothesis with exact-repeat evidence** — at least `_NAME_AXIS_CORROBORATION_MIN_EXACT = 2` identical `(name, args_hash)` calls for the same tool. Without that corroboration the emitted severity is **capped at `name_axis_max_severity`** (default `"info"`).

Critically, the **tier keeps its threshold** — the matched tier is still recorded on `raw["tier"]`, and `raw["severity_capped_from"]` names the uncapped severity that *would* have fired. So telemetry consumers see the full signal; only the routed severity is softened. The exact axis is **never** capped — identical-args repeats are definitionally redundant and keep full graduated severities.

The relevant excerpt from `_classify`:

```python
# goldfive/drift/tool_loops.py — inside the name-axis branch
corroborated = exact_for_name >= _NAME_AXIS_CORROBORATION_MIN_EXACT
cap = self._name_axis_severity_cap
if not corroborated and _SEVERITY_RANK[severity] < _SEVERITY_RANK[cap]:
    emit_severity = cap
    capped_from = severity.value
```

Note the tier walk does NOT `break` on a capped hit — it keeps walking lower tiers so a lower tier's *uncapped exact* hit is not shadowed by the capped name hit (`if not capped_from: break`).

### The `_classify` algorithm, step by step

`_classify(key, *, current_task_id, observed_revision_index, invocation_id)` is the whole brain. The algorithm, matching the code:

1. Snapshot the buffer to a list; compute `exact_counts: dict[sig, int]` and `name_counts: dict[name, int]` over the window in one pass.
2. Initialise `best_drift = None`, `best_rank = None` (rank = `_SEVERITY_RANK` of emitted severity; lower is more severe).
3. **For each distinct tool `name` in `name_counts`:**
   a. Look up its threshold table (`_thresholds_for_tool` → meta or work).
   b. Compute `exact_for_name` = the max exact-signature count for that name; `name_total` = `name_counts[name]`.
   c. Walk `_SEVERITY_TIERS` high→low (`critical`, `warning`, `info`). At each tier, `hit_exact = exact_thr is not None and exact_for_name >= exact_thr`; `hit_name = name_thr is not None and name_total >= name_thr`. Skip if neither.
   d. On the first hit: `mode = "exact" if hit_exact else "name"` (exact preferred — more specific). If `mode == "name"`, apply the corroboration cap. Build the `DriftEvent` candidate with the full `raw` dict.
   e. Track the tool's best candidate by emitted rank. `if not capped_from: break` — an uncapped hit ends the walk (lower tiers can't emit higher). A capped hit keeps walking so a lower uncapped exact tier isn't shadowed.
   f. After the walk, promote the tool's best into `best_drift` if it out-ranks the current best (highest *emitted* severity wins across tools).
4. Append `best_drift` if any.
5. **Independently, mode 3 (alternating):** if `len(buf) >= alternating_threshold`, inspect the last `alternating_threshold` names; if exactly two distinct names in strict `names[i] == names[i%2]` order AND `best_drift is None`, append an INFO alternating drift.
6. Return the (0, 1, or 2)-element list.

Two invariants fall out of this: (a) at most one graduated drift per call regardless of how many tools hit tiers; (b) the alternating INFO co-emits only when nothing stronger fired.

**Routing consequence:** a capped name-axis hit emits at INFO, and INFO routes to OBSERVE in the intervention ladder — telemetry only, no plan mutation. That is the intended "signal-only" behaviour. If you raise a name threshold expecting it to *escalate*, remember the cap: escalation on the name axis requires exact-repeat corroboration regardless of the threshold. Operators who want legacy uncapped behaviour set `name_axis_max_severity="critical"`.

Edge cases the classifier handles that a naive rewrite would break:

- **Two tools both hit tiers** — the highest *emitted* severity wins; a capped INFO never shadows another tool's genuine WARNING/CRITICAL (Trace 5).
- **Exact and name both satisfied on one tool** — `mode="exact"` (the more specific signal); the name axis is not separately emitted for that tool.
- **`window < threshold`** — the tier is unreachable; the constructor logs a DEBUG line but accepts the config (tests pin small windows).
- **`exact_threshold=2`** — INFO's exact is clamped to WARNING's so INFO stays reachable; WARNING still wins the highest-tier selection at count 2.
- **Alternating + graduated on the same window** — the alternating INFO is suppressed (`if best_drift is None`).

### The `raw` dict conventions

Each emitted tool-loop `DriftEvent` carries a structured `raw` dict. Modes and their keys:

| `mode` | Keys |
| --- | --- |
| `"exact"` | `mode`, `tool_name`, `args_hash`, `count`, `window_len`, `invocation_id`, `category`, `tier` |
| `"name"` | `mode`, `tool_name`, `count`, `window_len`, `invocation_id`, `category`, `tier`, and `severity_capped_from` (only when capped) |
| `"alternating"` | `mode`, `tools` (`[a, b]`), `window_len`, `invocation_id` |

`raw["invocation_id"]` is the **actual in-flight invocation** (`emit_invocation_id = invocation_id or scope_key`), stamped so the dispatch-time cancel helper can target the real invocation even though the *bucket* is keyed on `session_run_id`. This is the #420 subtlety: bucket-key ≠ cancel-target. Every tool-loop `DriftEvent` also carries `kind=LOOPING_REASONING`, `detector_name="tool_loops"`, `current_task_id` (from the `task_id` kwarg), `current_agent_id` (the bucket's agent), `observed_revision_index`, and a `trigger_input` summarising the last ≤16 tool names in the window.

### Worked classification traces

These are the exact windows-to-verdict traces a weak model should internalise before touching the classifier. Assume default thresholds (`window=10`, `name_axis_max_severity="info"`) and read each row as "the window contents (oldest→newest) produce this verdict". `sig = (name, args_hash)`.

**Trace 1 — work tool, identical args, exact axis escalates freely.**

| Window (work tool `build`, same args `h1`) | exact count | Highest tier | Emitted |
| --- | --- | --- | --- |
| `build/h1` ×3 | 3 | work-INFO (exact 3) | INFO |
| `build/h1` ×3 (again, exact still 3 but ≥ WARNING exact 3) | 3 | work-WARNING (exact 3) | **WARNING** |
| `build/h1` ×6 | 6 | work-CRITICAL (exact 6) | **CRITICAL** |

Note the INFO and WARNING work-exact thresholds are both 3 — so the *first* time exact hits 3 it matches the WARNING tier (WARNING > INFO, highest-tier-wins), and the emitted severity is WARNING, not INFO. The work-INFO exact=3 row is only reachable when a caller lowers `exact_threshold` (the constructor clamps INFO's exact to WARNING's so INFO stays reachable at e.g. `exact_threshold=2`: INFO fires at 2, WARNING at 2 as well — WARNING wins).

**Trace 2 — work tool, varied args, name axis is capped (the #484 heart).**

| Window (`read_file` with 5 distinct arg-hashes) | name count | exact (max) | Matched tier | `severity_capped_from` | Emitted |
| --- | --- | --- | --- | --- | --- |
| `read/ha read/hb read/hc read/hd read/he` | 5 | 1 | work-WARNING (name 5) | `"warning"` | **INFO** (capped) |
| add `read/hf read/hg` (name 7, still all distinct) | 7 | 1 | work-CRITICAL (name 7) | `"critical"` | **INFO** (capped) |
| replace two entries so `read/ha` appears twice (name 7, exact 2) | 7 | 2 | work-CRITICAL (name 7), **corroborated** | (none) | **CRITICAL** |

The first two rows are the healthy "reading six different files" pattern — name axis matches the tier but is capped to INFO (→ OBSERVE) because `exact < 2`. `raw["tier"]` records `"warning"` / `"critical"` and `raw["severity_capped_from"]` records the uncapped severity, so telemetry sees the full picture. The third row adds one exact repeat (`exact_for_name = 2 >= _NAME_AXIS_CORROBORATION_MIN_EXACT`), corroborating the loop hypothesis — now the full CRITICAL fires.

**Trace 3 — meta tool, benign reporting retries stay quiet longer.**

| Window (`report_task_completed`, same args) | exact | Matched tier | Emitted |
| --- | --- | --- | --- |
| ×3 | 3 | meta-INFO (exact 3) | INFO |
| ×6 | 6 | meta-WARNING (exact 6) | WARNING |
| ×10 | 10 | meta-CRITICAL (exact 10) | CRITICAL |

Meta thresholds push the first non-INFO fire out to 6 (vs work's 3). The meta name axis never fires (`"name": None` at every tier) — same-name-varied-args on a reporting tool is meaningless. Remember `on_task_progress` clears the window on an acknowledged-success report, so in a healthy run these never accumulate to 3 in the first place; the meta ladder only matters for *errored* report retries.

**Trace 4 — alternating cycle, INFO only, suppressed by a stronger hit.**

| Window tail (length 5) | Alternating shape? | Other drift? | Emitted |
| --- | --- | --- | --- |
| `A B A B A` (2 distinct names, strict A/B alternation) | yes | none | INFO (alternating) |
| `A B A B A` but `A` also hit exact-3 | yes | work-WARNING on `A` | WARNING only (alternating suppressed) |
| `A B A C A` (3 distinct names) | no (`len(set) != 2`) | — | nothing from alternating |

The alternating check inspects only the last `alternating_threshold` (5) slots and requires exactly two distinct names in strict `names[i] == names[i%2]` order. It co-emits with a graduated hit only when no graduated hit fired (`if best_drift is None`).

**Trace 5 — cross-tool selection (highest EMITTED severity wins).**

Window contains `build/h1 ×6` (work exact 6 → CRITICAL) interleaved with `read` ×5 varied-args (name 5 → capped INFO). Two tools hit tiers; the classifier keeps the candidate with the highest *emitted* severity → the `build` CRITICAL is returned, the capped `read` INFO is dropped. This is why the cap comparison uses **emitted** severity, not tier: a capped INFO must never shadow a genuine CRITICAL on another tool.

### Config, env knobs, and legacy overrides

Threshold resolution precedence (`resolve_thresholds()`):

1. An installed `ToolLoopConfig` (via `configure(config)`, called by `goldfive.wrap(runtime.tool_loops)`) — `thresholds_from_config`.
2. Else `load_thresholds_from_env()` — the `GOLDFIVE_TOOL_LOOP_*` vars.
3. Else module defaults.

The env vars (all read leniently — malformed / non-positive → default, logged at DEBUG):

| Env var | Constructor kwarg | Default | Effect |
| --- | --- | --- | --- |
| `GOLDFIVE_TOOL_LOOP_WINDOW` | `window` | 10 | Ring-buffer size per key. Must be ≥ the largest threshold you want reachable (meta CRITICAL needs 10). |
| `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` | `exact_threshold` | 3 | **Legacy override of the WORK category's WARNING-exact tier only.** |
| `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` | `name_threshold` | 5 | **Legacy override of the WORK category's WARNING-name tier only.** |
| `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` | `alternating_threshold` | 5 | Alternating-cycle tail length. |
| `GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY` | `name_axis_max_severity` | `"info"` | The #484 cap. `"info"`\|`"warning"`\|`"critical"`. |

**The legacy `exact_threshold` / `name_threshold` kwargs override ONLY the work-WARNING tier.** They do NOT touch meta thresholds, work-CRITICAL, or work-INFO. This preserves pre-#204 single-threshold semantics. If a test passes `exact_threshold=2`, the constructor defensively clamps work-INFO's exact down so INFO stays reachable (`if info_exact > war_exact: work["info"]["exact"] = war_exact`). Tests that exercise CRITICAL must leave these kwargs at defaults — the graduated CRITICAL tiers come from the module constants, not the legacy kwargs.

The graduated CRITICAL tiers and the meta thresholds are **not** env-tunable in the current build — they are module constants (`_META_THRESHOLDS`, `_WORK_THRESHOLDS`) grouped so a future PR can surface them via `ServerConfig`. The `ToolLoopConfig` dataclass (`goldfive/config.py`, near line 383) exposes `window`, `exact_threshold`, `name_threshold`, `alternating_threshold`, `name_axis_max_severity` — the same five as the env path.

### The tracker is O(window) and deterministic

No embeddings, no LLM. O(window) per tool call. State is a single `ToolLoopTracker` instance on the plugin — ephemeral to one run, no per-session persistence. `clear()` (called from the plugin's `clear_active_context`) drops every buffer so state doesn't leak across sessions when the plugin instance is reused. The classifier never mutates its buffers after firing — it does not dedupe drifts; the steerer's intervention ladder already dedupes by `(kind, task_id)` occurrence count, and a persistent loop SHOULD keep emitting so the ladder escalates.

### Why tool loops emit `LOOPING_REASONING` (and why `LOOPING_TOOL_CALL` is protected KEEP)

**This trips up every reader.** The tool-loop tracker emits `DriftKind.LOOPING_REASONING`, NOT `DriftKind.LOOPING_TOOL_CALL`. That is deliberate (#204):

- `LOOPING_REASONING` has a ladder row that routes tool loops through NUDGE-first CRITICAL routing — the desired response to a tight tool loop.
- `LOOPING_TOOL_CALL` remains a **defined enum value with a ladder/promotion/planner surface** from the #204/#206 history. It is a **PROTECTED KEEP** decision. Do NOT delete it, do NOT "consolidate" it into `LOOPING_REASONING`, do NOT repoint the tracker at it — without explicit human sign-off.
- Because the tracker's kind (`LOOPING_REASONING`) is *shared* with the embedding reasoning-loop detector, the tracker stamps `detector_name="tool_loops"` (#480) so telemetry attributes the fire to the tool-loop tracker and not the embedding path.

If you are tempted to "fix" the apparent mismatch (a tool loop labelled as a *reasoning* loop), stop. The kind choice is the routing choice; the source is disambiguated by `detector_name`. See `17-invariants-hazards-history.md`.

---

## Reasoning detectors (`goldfive/drift/reasoning.py`)

Thinking-capable models (Qwen3.5, Claude extended-thinking, o1-style) expose chain-of-thought via `reasoning_content` / `thinking` blocks. The adapter captures those and hands them to `DefaultSteerer.observe_reasoning`, which runs this pipeline. Four drift kinds live here; two of the four require the embedding extra.

### Tunables reference

Every threshold has a module-level constant (the default) and, when a `ReasoningDriftConfig` is installed via `configure`, a config field that overrides it. The constants:

| Constant | Default | Config field | Governs |
| --- | --- | --- | --- |
| `LOOPING_REASONING_HASH_WINDOW` | 5 | `looping_reasoning_hash_window` | How many prior blocks the loop/cluster detectors compare against. |
| `LOOPING_REASONING_SIMILARITY_THRESHOLD` | 0.9 | `looping_reasoning_similarity_threshold` | Cosine cliff for `LOOPING_REASONING`. |
| `REASONING_CLUSTER_SIMILARITY_THRESHOLD` | 0.75 | `reasoning_cluster_similarity_threshold` | Lower bound of the cluster-tightening band `[0.75, 0.9)`. |
| `OFF_TOPIC_DISTANCE_THRESHOLD` | 0.7 | `off_topic_distance_threshold` | `1 - cosine` distance for `OFF_TOPIC`. |
| `INTENT_DIVERGENCE_HEALTHY_SIMILARITY` | 0.6 | `intent_divergence_healthy_similarity` | ≥ this ⇒ no intent drift. |
| `INTENT_DIVERGENCE_MINOR_SIMILARITY` | 0.4 | `intent_divergence_minor_similarity` | Boundary between INFO and WARNING. |
| `INTENT_DIVERGENCE_WARNING_SIMILARITY` | 0.2 | `intent_divergence_warning_similarity` | Boundary between WARNING and CRITICAL. |
| `SENTENCE_LEVEL_MIN_BLOCK_LENGTH` | 200 | — (not config-exposed) | Char length above which off-topic runs the per-sentence path. |
| `SENTENCE_LEVEL_MAX_SENTENCES` | 10 | — | Cap on sentences embedded per off-topic call. |

Each has a private accessor (`_looping_hash_window`, `_looping_similarity_threshold`, `_cluster_similarity_threshold`, `_off_topic_distance_threshold`, `_intent_healthy_similarity`, `_intent_minor_similarity`, `_intent_warning_similarity`) that returns the config field when `_CONFIG is not None` else the constant. Read thresholds through these accessors, never the bare constant, so a per-Runner override is honoured.

### Pipeline entry-point signatures

```python
# goldfive/drift/reasoning.py
async def analyze_reasoning(
    text: str, session: Session, *,
    mode: ReasoningDriftMode = "embedding",   # note: the STEERER passes "judge"
    call_llm: JudgeCallLLM | None = None, model: str = "",
    sink: Any = None, agent_name: str = "",
    available_agents: list[str] | list[dict[str, Any]] | None = None,
    embedding_pipeline: Any = None, judge_classifier: Any = None,  # test seams
) -> DriftEvent | None: ...

async def analyze_reasoning_with_focus(...) -> ReasoningJudgeVerdict: ...   # #271 focused verdict
```

`DEFAULT_REASONING_DRIFT_MODE = "judge"` is the production default; the `mode="embedding"` default on the free function is a legacy artefact — the steerer always passes an explicit mode. `embedding_pipeline` (a `(text, session) -> DriftEvent | None` callable) and `judge_classifier` / `judge_runner` (async) are **test seams** — they replace `_embedding_pipeline` / `_run_judge` so tests script each side without monkeypatching detector symbols. Both default to `None` (use the in-module implementations) so production sees no behaviour change.

### The reasoning-history ring and `reasoning_hash`

`session.reasoning_history` is a list of prior reasoning blocks; the steerer appends the new block **before** running the pipeline. Detectors read but do not mutate it (except the one-shot cluster flag). Background judge runs pass a `history` / `reasoning_history` *snapshot* pinned at schedule time so entries appended by later turns can't produce a false self-match.

`reasoning_hash(text)` is the byte-identical loop primitive:

```python
# goldfive/drift/reasoning.py
def reasoning_hash(text: str) -> str:
    if not text:
        return ""
    normalised = " ".join(text.split()).strip().lower().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()[:16]
```

Whitespace-collapsed, lowercased, SHA-256, first 16 hex chars. This is exact-equality matching of structured (normalised) text — **allowed** under the no-NL-heuristics invariant. It is not classifying meaning; it is detecting byte-identical repetition.

### Detector signatures

```python
# goldfive/drift/reasoning.py
def detect_looping_reasoning(text: str, session: Session, history: Sequence[str] | None = None) -> DriftEvent | None
def detect_reasoning_cluster_tightening(text: str, session: Session, history: Sequence[str] | None = None) -> DriftEvent | None
def detect_intent_divergence(text: str, session: Session) -> DriftEvent | None
def detect_off_topic(text: str, session: Session) -> DriftEvent | None
def reasoning_hash(text: str) -> str
```

The `history` parameter on the two history-window detectors overrides `session.reasoning_history` — the background judge passes the snapshot pinned at schedule time so entries appended by later turns cannot produce a false self-match. When `history is None` they read the live session list. All four capture `observed_revision_index` from the session's plan at the top of the function.

### `detect_looping_reasoning` — the cliff detector

Returns `LOOPING_REASONING` at WARNING when `text` matches a recent history entry. Two checks:

1. **Hash-exact — always on.** Compares `reasoning_hash(text)` against the hashes of the last `hash_window` (`LOOPING_REASONING_HASH_WINDOW`, default 5) prior blocks (`history[-hash_window-1:-1]`). No embeddings needed.
2. **Semantic — embedding-only.** `_embed.max_similarity(text, history) >= LOOPING_REASONING_SIMILARITY_THRESHOLD` (0.9). Fires only when the encoder is loadable; degrades to no-op otherwise (`max_similarity` returns 0.0).

**The history slice, worked.** The steerer has already appended the current block, so `session.reasoning_history[-1]` IS `text`. The detectors compare against `history[-hash_window-1:-1]` — the `-1` end deliberately excludes the just-appended current block so it can't self-match. Example with `hash_window=5` and a 8-entry history `[b0, b1, b2, b3, b4, b5, b6, b7=text]`: the slice `[-6:-1]` is `[b2, b3, b4, b5, b6]` — the five blocks *before* the current one. If `text` byte-normalises to any of those, hash-exact fires. The empty-string filter (`if h`) drops blank entries so a run of empty reasoning blocks doesn't spuriously match.

### `detect_reasoning_cluster_tightening` — the early-warning tier

Returns `REASONING_CLUSTER_TIGHTENING` at **INFO** when the max cosine similarity against the last `hash_window` prior blocks falls in the half-open band `[0.75, 0.9)` (`REASONING_CLUSTER_SIMILARITY_THRESHOLD` up to but not including the loop threshold). The agent's CoT is repeating concepts but hasn't collapsed into a loop yet.

- **Embedding-only** — silent when the encoder is unavailable (semantic tightening, not byte-identical repetition, so there is no hash fallback).
- **One-shot per task** — fires at most once per `session.current_task_id`, tracked via `session.reasoning_cluster_flagged` (a set). This is the anti-spam guard; it mutates the session set. This is the sanctioned narrow bookkeeping mutation — keyed on the stable `current_task_id`.
- The cliff detector (0.9+) owns the high-similarity regime; cluster-tightening explicitly returns `None` when `sim >= loop_threshold` so the two never double-fire.

### `detect_intent_divergence` — graduated by cosine

Returns `INTENT_DIVERGENCE` at a **variable** severity based on how far the reasoning is from `session.goals` + the current task topic. The kind is stable; severity differentiates. Bands (embedding path):

| Cosine similarity | Severity |
| --- | --- |
| `>= 0.6` (`INTENT_DIVERGENCE_HEALTHY_SIMILARITY`) | no drift (None) |
| `0.4 <= sim < 0.6` (`MINOR`) | INFO |
| `0.2 <= sim < 0.4` (`WARNING`) | WARNING |
| `< 0.2` | CRITICAL |

`_severity_from_similarity(sim)` maps the score. When the encoder is unavailable, it falls back to `_pattern_intent_divergence` — a WARNING-only path.

Worked severity mapping (reference = goals + current task topic):

| Observed cosine | Band | Emitted |
| --- | --- | --- |
| 0.72 | ≥ healthy (0.6) | None (no drift) |
| 0.55 | minor ≤ x < healthy | INFO |
| 0.31 | warning ≤ x < minor | WARNING |
| 0.08 | < warning (0.2) | CRITICAL |

The kind is `INTENT_DIVERGENCE` in all firing rows — only severity differentiates. A caller filtering by kind sees one signal; a caller that cares about urgency reads `severity`. There is a `_bump_severity` helper (saturating INFO→WARNING→CRITICAL) retained in the module, but the historical severity bump it served (`_has_unreferenced_keyword`) was removed — the cosine band alone determines severity now.

**`_pattern_intent_divergence` is a known banned-class survivor — handle with care.** It uses `_INTENT_DIVERGENCE_MARKERS`, a regex over "my goal is / let's focus on / pivot to" phrasing, plus a token-overlap check against goal summaries. Mechanically: it finds a marker, takes the 120 chars after it as a cheap proxy for the proposed new goal, extracts ≥4-char tokens (minus `_STOPWORDS`), and fires flat WARNING only if none of those tokens appear in the goal summary (if any overlaps, it assumes the model is *restating* not diverging). This is exactly the NL-heuristic shape the invariant forbids — it survives only as the *no-embeddings fallback* and was deliberately **demoted, not deleted**: the historical `_has_unreferenced_keyword` CRITICAL-bump was removed (#226/#230; the helper deleted) because it fired on generic English vocabulary ("wants", "asking", "interactive", "slideshow") and contaminated the embedding signal. Pattern-path severity is now flat WARNING. **Do not extend the marker regex, do not add tiers to the pattern path, do not port this shape to a new detector.** If you want richer intent-divergence signal, add embeddings or teach the LLM judge (`08-llm-judges.md`).

### `detect_off_topic` — whole-text then per-sentence

Returns `OFF_TOPIC` at WARNING when reasoning is far from the current task topic. Requires the embedding extra; returns `None` when the encoder is unavailable or there is no bound current task. Two checks in order:

1. **Whole-text distance** — `_embed.distance_to_topic(text, topic) >= OFF_TOPIC_DISTANCE_THRESHOLD` (0.7). `distance_to_topic` returns `1 - cosine`, or `-1.0` when unavailable (the `-1.0` floor lets callers distinguish "can't compute" from "genuine zero distance"; the `dist >= 0` guard skips the unavailable case).
2. **Per-sentence min-distance (#224)** — only when `_looks_multi_sentence(text)` (len > 200 chars OR ≥2 terminators). Splits on `[.!?]\s+`, embeds up to `SENTENCE_LEVEL_MAX_SENTENCES` (10) candidate sentences, and fires if ANY non-trivial sentence has distance `>= 0.7`. Rationale (#223 raccoon-stimulus calibration): on real embedding models the shared vocabulary of a long reasoning block swamps a brief drift tangent, so whole-text cosine stays high even when the block clearly drifted. A single sentence is short enough that drift tokens are a large fraction of the embedding, so the same 0.7 threshold applies roughly unchanged — deliberately no new knob.

The sentence splitter (`_split_sentences`) and candidate filter (`_is_sentence_candidate`, requires a 5+ char alpha token) are pragmatic string ops, not NL classification — they select *which text to embed*, they do not themselves emit drift from lexical patterns. That distinction is what keeps them on the right side of the invariant.

The off-topic sentence-level helpers, precisely:

| Helper | Behaviour |
| --- | --- |
| `_looks_multi_sentence(text)` | `True` if `len(text) > 200` OR `text` has ≥2 `.!?` terminators. Gates the per-sentence path. |
| `_split_sentences(text)` | Split on `[.!?]\s+`, trim trailing terminators, drop empties. No abbreviation/ellipsis handling (the #223 corpus has none). |
| `_is_sentence_candidate(s)` | `True` if `len(s) >= 10` AND `s` has a `[a-zA-Z]{5,}` token. Filters `"OK"`, `"Step 1"` fragments that burn HTTP budget for no signal. |

The per-sentence scan embeds at most `SENTENCE_LEVEL_MAX_SENTENCES` (10) candidates, tracks the worst distance, and fires WARNING if `worst >= 0.7`. If any sentence's `distance_to_topic` returns `-1.0` mid-scan (encoder went unavailable), the whole scan bails to `None` — matching the whole-text graceful-degrade contract.

### Reasoning pipeline ordering — why worst-signal-wins

`_embedding_pipeline` returns the first non-None from this fixed order; the order is not arbitrary:

| Order | Detector | Rationale for position |
| --- | --- | --- |
| 1 | `detect_intent_divergence` | Runs first even at INFO so the kind is stable; callers filter by severity. Can reach CRITICAL (the worst signal). |
| 2 | `detect_looping_reasoning` (0.9+) | Before cluster-tightening so a tight loop emits the cliff, never the INFO tier. |
| 3 | `detect_off_topic` | After loops (a loop is a stronger/cheaper signal than topic distance). |
| 4 | `detect_reasoning_cluster_tightening` (0.75–0.9) | Last; the weakest INFO early-warning, only reached when nothing above fired. |

At most one drift per call keeps embedding cost bounded (each detector may pay several HTTP encodes).

### Embedding availability — decision table

When does an embedding detector actually run? `_get_model()` resolves in this order; the outcome column is what the detectors see:

| `set_model` installed? | `GOLDFIVE_EMBEDDING_BASE_URL` set? | `goldfive[embedding]` installed? | Breaker tripped? | Outcome |
| --- | --- | --- | --- | --- |
| yes | — | — | — | Use the installed encoder. |
| no | yes, reachable | — | no | Use the HTTP backend. |
| no | yes, unreachable | — | (trips after 3) | No signal; probe once per cooldown. |
| no | yes, build-failed | — | — | `_MODEL_UNAVAILABLE` (does NOT fall back to sentence-transformers). |
| no | no | yes | — | Use sentence-transformers `all-MiniLM-L6-v2`. |
| no | no | no | — | **No signal — the default install.** Embedding detectors no-op silently. |

The bottom row is the production default. Combined with `mode="judge"` (which doesn't select the embedding pipeline at all), the embedding detectors are dead weight unless an operator opts in (Recipe E). Don't write code that *assumes* they ran.

### The pipeline entry points

`analyze_reasoning(text, session, *, mode=...)` and its focus-returning sibling `analyze_reasoning_with_focus(...)` select the pipeline by `mode: ReasoningDriftMode`:

| Mode | Behaviour |
| --- | --- |
| `"judge"` (**default**, `DEFAULT_REASONING_DRIFT_MODE`) | LLM-as-a-judge only (`08-llm-judges.md`). Silently no-ops if `call_llm is None`. |
| `"embedding"` | The embedding pipeline `_embedding_pipeline`. |
| `"both"` | Run both; higher-severity drift wins; ties broken by the embedding path (synchronous, runs first). Degrades to `"embedding"` when `call_llm is None`. |
| `"off"` | Skip the mode-selected pipeline entirely. |

**The always-on cheap loop detector runs upstream in `DefaultSteerer.observe_reasoning` in every mode**, including `"off"` — the mode only selects the off-topic / intent / judge pipeline, not the loop detector.

`_embedding_pipeline` runs the four embedding detectors in **worst-signal-wins order** and returns the first hit: `INTENT_DIVERGENCE` → `LOOPING_REASONING` → `OFF_TOPIC` → `REASONING_CLUSTER_TIGHTENING`. INTENT_DIVERGENCE runs first even at INFO so its kind is stable; LOOPING (0.9+) runs before CLUSTER_TIGHTENING (0.75–0.9) so a tight loop emits the cliff, never the INFO tier. At most one drift per call — cost bounded.

In `"both"` mode the worst-severity drift wins and the judge's attribution fields (`focused_task_id`, `focus_confidence`, `stated_intent`, provenance) are preserved onto the returned verdict via `dataclasses.replace` even when the embedding drift wins — see `08-llm-judges.md` for the focused-verdict shape.

### `"both"` mode and the focused verdict — the merge rules

`analyze_reasoning_with_focus` returns a `ReasoningJudgeVerdict` (drift + attribution). In `"both"` mode the merge is:

1. Run the embedding pipeline (synchronous) → `embedding_drift`.
2. Run the judge (if `call_llm` set) → `judge_verdict`.
3. If the judge didn't run → return `ReasoningJudgeVerdict(drift=embedding_drift)` (empty attribution — the embedding pipeline can't attribute against the plan-tasks list).
4. If embedding found nothing → return the `judge_verdict` as-is.
5. Both fired → worst-severity wins (`_SEVERITY_ORDER` compare), **embedding wins ties** (deterministic, ran first). Either way the attribution fields come from the judge, preserved via `dataclasses.replace(judge_verdict, drift=embedding_drift)` when the embedding drift wins — so the judge's `focused_task_id` / `focus_confidence` / `stated_intent` / provenance / measurement fields (`judge_ran`, `elapsed_ms`) survive onto the returned verdict.

The legacy `analyze_reasoning` delegates to `analyze_reasoning_with_focus` and returns `verdict.drift` for back-compat with callers that only want the drift. Don't duplicate the merge logic — call the focus variant and drop the attribution if you don't need it.

### How the intervention ladder consumes these drifts

You don't dispatch from a detector, but you should understand the consumer so your severity/kind/task-id choices route correctly (full detail in `09-steering-ladder-and-gates.md`):

- `handle_drift` emits `DriftDetected`, then computes an `InterventionLevel` from `(kind, severity, occurrence_count)` via the `_LADDER` table.
- **`occurrence_count`** is per-`(kind, task_id)` — this is why a **stable** `current_task_id` matters: a churning id resets the counter, so a repeating loop never escalates from ABSORB → CANCEL_REINVOKE → PAUSE_ESCALATE.
- INFO → OBSERVE (telemetry-only, no plan mutation). WARNING → typically ABSORB (plan-extension refine) or NUDGE. CRITICAL → the escalation pair (first occurrence vs repeat).
- The tracker deliberately does **not** dedupe its own drifts — a persistent loop SHOULD keep emitting so the ladder's counter advances. Don't add dedup in the detector.
- Under `observation_only` (production default) every level above OBSERVE is neutralised at dispatch — the drift is still emitted for telemetry, but no cancel/refine/nudge fires. That is the passive-mode contract, enforced in exactly one place (`is_active_steering`), not in the detector.

### Tool-loop tuning DO / DON'T

| DO | DON'T |
| --- | --- |
| Tune the **exact** axis for redundant-work loops (never capped). | Expect the **name** axis to escalate by lowering its threshold alone. |
| Keep `window` ≥ the largest threshold you want reachable (meta CRITICAL needs 10). | Shrink `window` below a threshold and wonder why that tier never fires (it logs a DEBUG warning). |
| Leave meta name thresholds `None`. | Add a name threshold to the meta category. |
| Set `name_axis_max_severity="critical"` for legacy uncapped behaviour, knowingly. | Assume the name axis escalates by default — it is INFO-capped. |
| Leave `exact_threshold`/`name_threshold` at defaults when testing CRITICAL. | Pass legacy single-threshold kwargs and expect the graduated CRITICAL tiers to move. |

**Deleted, do not revive:** the historical lexical keyword detector `detect_unreferenced_keyword` was unwired from every mode (#226) and has since been deleted. The `CONFUSION` uncertainty-marker detector was retired (see the `DriftKind.CONFUSION` comment in `types.py` — proto value 28, gone). Both are the NL-heuristic anti-pattern. The reasoning judge covers the same ground semantically.

### Per-Runner threshold overrides

`reasoning.py` holds a module-global `_CONFIG: ReasoningDriftConfig | None` installed by `configure(config)` (called from `goldfive.wrap(runtime.reasoning_drift)`). When `None` (default) the detectors read the module-level constants (byte-identical to pre-#225). When installed, the `_looping_*` / `_intent_*` / `_off_topic_*` / `_cluster_*` helper functions read the config's fields. Installation is **process-wide** — two Runners in one process share the last-installed config (documented as an acceptable minor race for heuristic thresholds). `ReasoningDriftConfig` lives at `goldfive/config.py` near line 436. One additional flag lives here: `fallback_to_content_when_no_reasoning` (#263), read by the ADK plugin's `_choose_reasoning_text` to decide whether to synthesise a reasoning signal from the response body on non-thinking models.

---

## The embedding backends (`goldfive/drift/_embed.py`)

This module is why the embedding detectors are "deterministic given the encoder, else no-signal". Understand three things: backend selection, the circuit breaker, and why it is unreachable under the default install.

### Backend selection (`_get_model`, first match wins)

1. A caller-installed model via `set_model(...)` (tests / custom runtimes).
2. An OpenAI-compatible HTTP backend when `GOLDFIVE_EMBEDDING_BASE_URL` (or `EmbeddingConfig.base_url`) is set — POSTs to `{BASE_URL}/v1/embeddings` against any llama.cpp / Ollama / OpenAI-compatible endpoint. Zero extra local install; the user's LLM server is the embedding server. Env: `GOLDFIVE_EMBEDDING_MODEL`, `_API_KEY`, `_TIMEOUT_MS` (default 10000).
3. A `sentence-transformers` model (`all-MiniLM-L6-v2`) from the `goldfive[embedding]` extra.

Import and network I/O are deferred to first call — importing `_embed` is always safe from minimal installs. The model is cached in `_MODEL`; the first failed path flips `_MODEL_UNAVAILABLE` so subsequent calls skip the cost. Every public function returns the "no-signal" value silently when nothing is reachable: `max_similarity` → `0.0`, `distance_to_topic` → `-1.0`. Note the "configured-but-unreachable" rule: if `GOLDFIVE_EMBEDDING_BASE_URL` is set but the backend fails to build, `_get_model` flips to unavailable rather than silently falling back to sentence-transformers — the operator asked for an HTTP endpoint, so they get "no signal", not surprise-local-encoding.

**Why the embedding detectors are unreachable under a default install:** with no `GOLDFIVE_EMBEDDING_BASE_URL` set and the `embedding` extra not installed, `_get_model()` returns `None`, `available()` is `False`, and every embedding-only detector (`detect_off_topic`, `detect_reasoning_cluster_tightening`, the cosine tiers of looping/intent) silently no-ops. The default production path is `mode="judge"` (LLM judge), so the embedding pipeline is not even selected by default. Treat the embedding detectors as an **opt-in** surface: they only do anything when an operator has explicitly configured an embedding endpoint or installed the extra.

### The runtime circuit breaker + #479 half-open recovery

An HTTP backend whose endpoint is unreachable at *runtime* (not import time) would otherwise pay the timeout on every `max_similarity` / `distance_to_topic` call. The breaker:

- `_note_backend_failure` increments `_RUNTIME_FAILURE_COUNT` on each empty encode. After `_RUNTIME_FAILURE_THRESHOLD = 3` **consecutive** failures it trips: sets `_RUNTIME_FAILURE_TRIPPED`, `_MODEL_UNAVAILABLE = True`, drops the cached `_MODEL` (so `_get_model` actually short-circuits — it only short-circuits on `_MODEL_UNAVAILABLE` when `_MODEL is None`), stamps `_RUNTIME_TRIPPED_AT`, and logs one WARNING naming the endpoint.
- `_note_backend_success` resets the counter on **any** non-empty encode, so a transient outage (fail, fail, success, fail, fail) never trips — only *consecutive* failures count.
- **#479 half-open recovery:** a tripped breaker does not disable embeddings for the process lifetime. After a cooldown (`_RUNTIME_RECOVERY_COOLDOWN_S = 60.0`, overridable at read time via `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S`), `_get_model` admits **one** probe encode (`_breaker_cooldown_elapsed()` → sets `_MODEL_UNAVAILABLE=False`, restarts the cooldown clock). A success closes the breaker (`_note_backend_success`); a failure re-opens it and restarts the cooldown (the `if _RUNTIME_FAILURE_TRIPPED:` branch in `_note_backend_failure`).

An import-failure `_MODEL_UNAVAILABLE` (sentence-transformers missing) or a `force_unavailable()` has **no** recovery path by design — `_breaker_cooldown_elapsed()` returns False unless the breaker was *tripped* (as opposed to import-disabled).

Test escape hatches (never call from prod), summarised:

| Hatch | Effect |
| --- | --- |
| `set_model(model)` | Install a fake encoder (`encode(list[str]) -> list[vec]`); `None` clears + re-enters lazy-load. Resets breaker + cache. |
| `set_model(None)` | Clear the cached model; next call re-enters lazy-load. |
| `force_unavailable()` | Mark unavailable **without** an HTTP probe — `_get_model()` returns `None` regardless of env/config. No recovery path (not a breaker trip). |
| `reset_circuit_breaker()` | Clear the failure counter + un-trip; clears `_MODEL_UNAVAILABLE` only when it was set by the trip. |
| `set_backend_loader(fn)` | Override the OpenAI-backend constructor (assert which `base_url` is selected). |
| `set_backend_class(cls)` | Override the backend class the default loader builds (assert ctor kwargs). |
| `set_cache_max(n)` | Shrink/restore the LRU cap for eviction tests. |

`configure()` (the prod path) also resets the breaker and cache. `configure()` has a subtlety: it flushes the cached backend only when the current `_MODEL` is an `_OpenAIEmbeddingBackend` — a test-installed fake encoder (`set_model`) is kept alive so a test that sets a fake and also installs a config still sees the fake win.

### The per-text LRU cache

`_cached_encode` memoises per-text vectors in a 512-entry (`_CACHE_MAX`) `OrderedDict` keyed on `(backend_name, text)` where `backend_name = f"{type(model).__name__}:{id(model)}"`. The HTTP path pays a round-trip per call and `max_similarity` re-encodes each history entry on every observation — the cache makes repeated history comparisons cheap. Swapping the model via `set_model` produces a different `id(model)` → a different cache bucket → no stale vectors.

`_parse_openai_response` is defensive against three real footguns: missing `data` (error responses), `embedding` nested one level deep (`[[...]]` from some llama.cpp builds — unwrapped), and non-numeric embedding items. `_cosine` uses NumPy when importable and falls back to a pure-Python dot product otherwise.

### The `_OpenAIEmbeddingBackend.encode` flow

`_OpenAIEmbeddingBackend` exposes the same `encode(list[str]) -> list[list[float]]` surface as the sentence-transformers adapter. Per call:

1. If `texts` is empty → `[]` (no breaker touch).
2. If the OpenAI SDK client built at construction (`_prefer_sdk`), try `_encode_via_sdk`. On success → `_note_backend_success()` + return.
3. Otherwise (SDK absent or SDK call failed) fall through to `_encode_via_httpx` (raw `httpx.Client`). On non-empty vectors → `_note_backend_success()` + return.
4. On empty result → `_note_backend_failure(base_url)` + return `[]`.

The SDK client is preferred for auth/retry consistency with the harmonograf client stack; the httpx path is the dependency-light fallback. Both send `model` explicitly (empty string when unconfigured — llama.cpp/Ollama tolerate `model=""`, strict OpenAI servers then complain loudly rather than embedding silence). The success/failure hooks are the **only** places the circuit breaker state moves — keep it that way; do not scatter breaker mutations into callers.

### `max_similarity` / `distance_to_topic` — the two public scores

```python
# goldfive/drift/_embed.py
def max_similarity(current: str, history: list[str]) -> float:
    # max cosine(current, h) over history; 0.0 if unavailable / any encode error
def distance_to_topic(text: str, topic: str) -> float:
    # 1 - cosine(text, topic) in [0, 2]; -1.0 if unavailable
```

`max_similarity`'s `0.0` floor is deliberately loop-safe: a zero can never *exceed* a positive threshold, so an unavailable encoder never manufactures a false loop. `distance_to_topic`'s `-1.0` sentinel is distinguishable from a genuine zero distance (`0.0` = identical), which is why off-topic guards with `if dist >= 0`. Both route through the LRU-cached `_cached_encode`, so a `max_similarity(current, [h1..h5])` call encodes `current` once and reuses cached `h1..h5` vectors from prior observations.

---

## Capability-mismatch detector (`goldfive/drift/capability_check.py`)

Fires `CAPABILITY_MISMATCH` at CRITICAL when the agent a coordinator delegated to *structurally* cannot perform the bound task. Runs at `delegation_observed` time (#253). Replaces the planner-LLM `PLAN_DIVERGENCE` "wrong-assignee" comparison — instead of comparing the planner's *predicted* assignee (wrong when the planner LLM hallucinated), it grounds the comparison in *actual* tool capability surfaced by the live ADK agent object.

```python
# goldfive/drift/capability_check.py
def detect_capability_mismatch(
    *,
    invoked_agent_name: str,
    invoked_agent_tools: list[Any],       # live ADK Tool objects; only attrs read (.agent, .name, .func)
    task: Task,
    all_pending_tasks: Sequence[Task] | None = None,  # None disables Rule C
) -> DriftEvent | None: ...
```

**Three narrow rules, evaluated B → A → C; first to fire wins.** The ordering encodes confidence: B consults explicit planner output (highest), A grounds in tool-shape, C is the cross-task lexical inference (lowest). Returning the first hit means the steerer's refine sees exactly one verdict per delegation. Returns `None` when no rule trips, or when `invoked_agent_tools` is empty AND `required_tools` is empty AND Rule C has no signal. False positives are worse than false negatives here — every fire cancels the in-flight invocation and triggers a refine.

- **Rule B — required-tools advisory (highest confidence, consults explicit planner output).** If `task.required_tools` is non-empty and the invoked agent's tool names don't cover every required name, fire. Skipped entirely when the advisory is empty (an empty advisory is "no opinion", not a miss).
- **Rule A — coordinator-style leaf-assignment.** If the agent has tools AND **every** tool is an `AgentTool` (its only capability is to delegate further) AND the bound task does NOT read as orchestrational (`_looks_like_delegation_task` — substring scan against `DELEGATION_VERB_MARKERS` = `coordinate`, `delegate`, `orchestrate`, `dispatch`, `route to`, `hand off`, `handoff`), the agent structurally cannot do the leaf work. Fires. An **empty** tool list does NOT trip Rule A — can't distinguish "no tools" from "test stub / introspection failure", and the false-positive cost is high.
- **Rule C — out-of-DAG-order delegation (#268).** When the invoked agent's *role stem* is absent from the bound task's title+description but present in some OTHER pending task, the pin bound the delegation to a structurally-wrong task. Requires `all_pending_tasks` (DAG-ready and not); silent without it. Concrete shape: `reviewer_agent` pinned to `draft_slides` while `review_presentation` sits PENDING and not-yet-DAG-ready.

`is_agent_tool(tool)` prefers `isinstance(tool, AgentTool)` when the `adk` extra is importable, with a duck-typed `.agent`-attribute fallback for stubs (a `FunctionTool` carries `.func` instead, so absence of `.agent` is a robust no-AgentTool signal).

**Stem extraction, by example.** `agent_name_stems(agent_name)` normalises on `_`/`-`/space, lowercases, trims trailing role-suffix tokens, and keeps tokens ≥4 chars:

```python
# goldfive/drift/capability_check.py — doctest-verified examples
agent_name_stems("reviewer_agent")       # ('reviewer',)
agent_name_stems("web_developer_agent")  # ('developer',)   -- "web" is <4 chars, dropped
agent_name_stems("helper_agent")         # ('helper',)
agent_name_stems("agent")                # ()               -- collapses to empty, Rule C stays silent
```

`stem_token_match(stem, token)` is bidirectional substring (`review` ↔ `reviewer`), catching the role-noun/verb pair. `tokenize_for_matching(text)` returns lowercase alphanumeric tokens ≥4 chars — no regex, pure char-buffer accumulation (the module comments cite #166/#167 four times).

**Rule C worked example.** Coordinator delegates `reviewer_agent` and the pin binds it to task `draft_slides` ("Draft the slide deck"). `all_pending_tasks` also contains `review_presentation` ("Review the presentation for errors"), PENDING but not yet DAG-ready.

1. `agent_name_stems("reviewer_agent")` → `('reviewer',)`.
2. `_task_text_contains_stem(draft_slides, "reviewer")` → tokens `{draft, slide, deck}`; no bidirectional match → **absent from bound**.
3. Scan other pending: `_task_text_contains_stem(review_presentation, "reviewer")` → tokens `{review, presentation, errors}`; `stem_token_match("reviewer", "review")` → `"review" in "reviewer"` → True → **present in other**.
4. Absent-here + present-there ⇒ fire CRITICAL: "delegated out of DAG order".

Bail-outs (all return `None`): `all_pending_tasks` empty, stem collapses to `()`, stem already in the bound task, or no other pending task mentions the stem.

**Rule precedence, worked (B beats A).** Suppose a coordinator-style agent (only `AgentTool` wrappers) is delegated a task with `required_tools=["run_tests"]` that reads as a leaf task ("Run the test suite"). Both Rule B (agent's tool names don't cover `run_tests`) and Rule A (all-AgentTool + leaf task) would fire. Because evaluation is B → A → C first-wins, **Rule B's detail is returned** — the higher-confidence explanation ("missing required tool run_tests; available tools: [...]") rather than Rule A's generic "only AgentTool wrappers". The refine sees exactly one verdict. If you reorder these rules, you change which explanation operators see and can surface a lower-confidence signal over a higher one — don't.

**Rule C's lexical caveat — a demoted-not-fixed banned-class survivor.** Rule C's stem matching (`agent_name_stems`, `stem_token_match`, `tokenize_for_matching`, `_task_text_contains_stem`) is *lexical*: it splits the agent name into ≥4-char tokens, trims role-suffixes (`agent`, `worker`, `assistant`, `bot`, `tool` — `AGENT_NAME_ROLE_SUFFIXES`), and does bidirectional substring matching against task-text tokens. This is a lexical heuristic over natural-language task titles — the same *class* the invariant is wary of. It is flagged as a **known banned-class survivor**, kept because it operates on the structural agent-name↔task-name correspondence (not on classifying the *meaning* of reasoning text) and because its conservative bail-outs make false fires rare: silent when no stem ≥4 chars survives, when the stem is already in the bound task, or when no other pending task mentions it. The functions are pure str ops with **no regex** (the module comments cite #166/#167 repeatedly). **Do not extend this into meaning-based matching, and do not treat it as license to add lexical NL classifiers elsewhere.** If Rule C proves too blunt, the correct fix is to strengthen the structural signal (required_tools advisories, DAG-readiness gating on the pin), not to add more lexical rules.

Registration lives at the bottom of the module:

```python
# goldfive/drift/capability_check.py
_register(
    DriftKind.CAPABILITY_MISMATCH,
    detect_capability_mismatch,
    _DetectorConfig(uses_llm=False),
    is_async=False,
)
```

---

## Structural classifiers in `goldfive/drift/__init__.py`

Four small event/text classifiers, all deterministic, all side-effect-free:

| Function | Kind | Severity | Signal |
| --- | --- | --- | --- |
| `classify_tool_error(event)` | `TOOL_ERROR` | WARNING | dict with truthy `error`, or `status` in `{FAILED, ERROR}`, or `ok is False`, or an object with a truthy `.error`. |
| `classify_refusal(text)` | `AGENT_REFUSAL` | graduated | Tiered substring markers scanned CRITICAL → WARNING → INFO, first-match-wins (a safety refusal is never downgraded to hedging). |
| `classify_confabulation_risk(*, task, tool_call_count, output_text)` | `CONFABULATION_RISK` | INFO | Task title/desc matches `CONFABULATION_TRIGGER_KEYWORDS` AND `tool_call_count == 0` AND non-empty output. |
| `classify_stop_reason(reason)` | `CONTEXT_PRESSURE` | WARNING | Normalised stop reason in `CONTEXT_PRESSURE_STOP_REASONS` (`MAX_TOKENS`, `LENGTH`, `TRUNCATED`, …). |

**`classify_tool_error` recognised shapes.** It accepts both harmonograf-ish and OpenAI-ish tool responses:

- `dict` with a truthy `error` key → `err_detail = str(error)`.
- `dict` with `status` (upper-cased) in `{"FAILED", "ERROR"}` → `err_detail` from `message` or the status.
- `dict` with `ok is False` → `err_detail` from `message` or `"tool returned ok=false"`.
- an object exposing a truthy `.error` attribute.

`tool_name` is pulled from `tool` / `name` (dict or attr); `task_id` from `task_id`. Returns `None` when no error detail is found — a successful tool response is not a drift. The detail string is `f"tool {tool_name!r} errored: {err_detail}"` (or the tool-less variant). These are structural shape checks over a tool-result object — no NL classification.

**On the marker tables and keyword sets:** `LLM_REFUSAL_MARKERS_*` and `CONFABULATION_TRIGGER_KEYWORDS` are substring tables, not regexes, and they are **conservative by design** — they classify *structural signals* (a refusal phrase, an external-data-access-shaped task), and every set carries an explicit comment warning against adding generic verbs. `classify_confabulation_risk` is INFO-only (record-and-surface, human decides). These predate and are narrower than the retired NL detectors; they are not a template for new lexical classifiers. When you extend `CONFABULATION_TRIGGER_KEYWORDS`, add only phrases that *strongly* imply external fetch/consult, never synthesis verbs like "write" / "summarize" / "draft" — a false positive surfaces on every clean research run.

`classify_stop_reason` normalises via `raw_name.upper().rsplit(".", 1)[-1]` so an enum-style `FinishReason.MAX_TOKENS` and a bare `"MAX_TOKENS"` both match. `classify_refusal`'s tiered scan (CRITICAL → WARNING → INFO, first-match-wins) guarantees a policy/safety refusal is never downgraded just because the text also contains a hedging phrase.

The module's `__getattr__` lazily re-exports the reasoning/goals/tool-loop helpers so `from goldfive.drift import classify_tool_error` stays cheap (defers the regex / optional-embedding imports until first access of a reasoning symbol).

### The refusal marker tiers

`classify_refusal` scans three tiers, first-match-wins, CRITICAL → WARNING → INFO:

- **`LLM_REFUSAL_MARKERS_CRITICAL`** — policy/safety refusals: `"i must decline"`, `"cannot assist with"`, `"against my guidelines"`, `"for safety reasons"`, `"i will not proceed"`. Surfaced at the highest severity so operators see the refusal clearly; refine usually can't fix these.
- **`LLM_REFUSAL_MARKERS_WARNING`** — capability refusals without a policy invocation: `"i cannot"`, `"i can't"`, `"i'm unable"`, `"beyond my capabilities"`, `"outside my scope"`, `"unable to locate"`, `"no viable approach"`, etc. The most common tier; triggers refine.
- **`LLM_REFUSAL_MARKERS_INFO`** — hedging/deferral: `"i'm not confident"`, `"i may not be the best fit"`, `"i think this might"`, `"not particularly well suited"`. Observational only; INFO does not trigger refine.

`LLM_REFUSAL_MARKERS` (flat concatenation, CRITICAL+WARNING+INFO) is **deprecated** — kept for external callers that imported the old flat tuple. Prefer the tiered tables + `classify_refusal`. These are substring markers (not regex); the first-match-wins scan is why the order CRITICAL→WARNING→INFO matters.

### The stop-reason set

`CONTEXT_PRESSURE_STOP_REASONS = frozenset({"MAX_TOKENS", "LENGTH", "MAX_OUTPUT_TOKENS", "TRUNCATED", "CONTENT_FILTER"})` — normalised, upper-cased, last `.`-segment. A match emits `CONTEXT_PRESSURE` at WARNING. This is exact-set membership of a *structured* enum value (not NL classification) — fully allowed.

### The confabulation gate

`CONFABULATION_TRIGGER_KEYWORDS` verbatim (the pinned contract — `test_drift_classifiers.py` asserts it; extend only with strong external-fetch phrases):

```python
# goldfive/drift/__init__.py
CONFABULATION_TRIGGER_KEYWORDS = (
    "research", "gather", "look up", "lookup", "verify", "review", "fetch",
    "search", "analyze the file", "analyze the document", "check the",
    "find information about", "find information on", "read the file",
    "read the document", "investigate", "consult", "cross-reference",
    "cross reference",
)
```

`classify_confabulation_risk(*, task, tool_call_count, output_text)` fires `CONFABULATION_RISK` at INFO only when **all three** hold: (a) `task.title`/`description` contains a `CONFABULATION_TRIGGER_KEYWORDS` phrase (substring, case-insensitive), (b) `tool_call_count == 0`, (c) `output_text.strip()` is non-empty. The keyword set (`research`, `gather`, `look up`, `verify`, `review`, `fetch`, `search`, `read the file`, `investigate`, `consult`, `cross-reference`, …) is conservative — it deliberately omits synthesis verbs (`write`, `summarize`, `format`, `draft`) because those describe pure-synthesis work where zero tool calls is the *expected* shape. INFO severity means record-and-surface; the human decides whether to cancel. Extending the set with a synthesis verb would surface a false drift on every clean synthesis run.

---

## The stall watchdog as a detector (#487)

Full mechanics are in `05-adk-plugin.md`; here is its identity as a **detector**. It is the sole producer of `DriftKind.TASK_TIMEOUT`.

- **Flag-gated, default OFF.** `SteeringConfig.stall_watchdog_enabled` (default `False`), `stall_timeout_s` (default 600.0). The steerer stashes these as `_stall_watchdog_enabled` / `_stall_timeout_s`; the ADK plugin reads them in `_maybe_start_stall_watchdog`. With the flag off there is no watchdog task and `TASK_TIMEOUT` is never produced.
- **One asyncio task per dispatch** (`_run_stall_watchdog`), spawned by `set_active_context`, cancelled by `clear_active_context`. Polls the session's liveness watermark.
- **Liveness stamp.** The `DriftObserver` liveness helper refreshes `session.last_observed_event_at = time.monotonic()` on every observed event (drift, tool observation). `_stall_liveness_watermark` reads it, floored at the watchdog's start time.
- **Graduated severity — same shape as tool_loops.** Idle beyond `timeout_s` → `TASK_TIMEOUT` at WARNING; each further multiple of `timeout_s` with no fresh activity → CRITICAL (`severity = WARNING if episode_fires == 0 else CRITICAL`). Fresh activity resets `episode_fires`.
- **Idle goal-judge trigger.** Idle beyond `GOAL_DRIFT_IDLE_SECONDS` (default 300, read **live** from `goldfive.drift.goals` per poll so an optimization-manifest `setattr` takes effect on a running watchdog) fires the trajectory-level goal-drift judge **once per idle episode** — the long-promised #143 idle scheduling; the watchdog is its producer. There is an AST-based manifest-liveness test guarding this live-read contract.
- **Does not double-report with the LLM watcher.** Skipped while an LLM call is in flight under its own per-call budget (`_llm_watcher_inflight()`) — `_run_llm_call_timeout_watcher` owns that hang and emits `LLM_CALL_TIMEOUT`. Worked: a single Qwen turn generates thinking tokens for 3 minutes. The idle watermark is silent (no observed events during generation), so the stall watchdog would otherwise count it as idle — but `_llm_watcher_inflight()` is `True`, so the watchdog `continue`s past its fire condition and the LLM-call watcher emits `LLM_CALL_TIMEOUT` instead. Exactly one drift, correctly attributed to the LLM hang, not a spurious `TASK_TIMEOUT`.
- **Passive-by-construction.** The `TASK_TIMEOUT` drift is routed through `steerer.drift.handle_drift`, so under `observation_only` (production default) it is telemetry-only; the ladder handles it only in active mode. The watchdog stamps `observed_revision_index` from `session.plan.revision_index` like every other detector.
- **Poll cadence.** The loop sleeps `max(0.005, min(timeout_s, idle_goal_s) / 8.0)` per tick — tracking the tighter of the two thresholds so neither fires grossly late, floored so a tiny test timeout can't busy-spin. Each tick reads the watermark, resets episode bookkeeping if it advanced, then checks the idle-goal trigger and the timeout trigger in that order.
- **Honest limitation:** it is an asyncio task on the wrapped tree's own event loop. A synchronously-blocking tool starves the loop and the watchdog cannot fire until the block ends — sync-blocked stalls are out of scope. Covered: hung *async* tool calls and idle-with-no-transitions runs.

Ladder row (`drift_observer.py`): `TASK_TIMEOUT: (OBSERVE, NUDGE, (PAUSE_ESCALATE, PAUSE_ESCALATE))` — a stall is a liveness signal, not a plan defect, so WARNING nudges rather than refining (ABSORB would loop the planner against a plan that isn't wrong).

### The two idle constants (`goldfive/drift/goals.py`)

The watchdog consumes two constants that live in the goal-drift module (`goldfive/drift/goals.py`), not in the config dataclass:

- `GOAL_DRIFT_IDLE_SECONDS = 300` — wall-clock idle above which the watchdog triggers the trajectory-level goal-drift judge once per idle episode. Read **live** per poll (module-attribute lookup) so an optimization-manifest `setattr` mutates a running watchdog. `optimization/manifest.toml` registers it (`source = "goldfive/drift/goals.py:GOAL_DRIFT_IDLE_SECONDS"`).
- `GOAL_DRIFT_CHECK_INTERVAL = 5` — the turn-based judge cadence default. **Not consulted at runtime** — it is a back-compat re-export; the live knob is `GoalDriftConfig.check_interval` (read by `DefaultSteerer`). The manifest note calls this out explicitly. Do not "wire it up" thinking it is dead — it is a documented back-compat alias.

---

## The detector registry (`goldfive/drift/registry.py`) — post-#490 surface

The registry centralises the *genuinely-shared* boilerplate across detectors so each detector contributes only its classifier-specific logic. **`classify()` was deleted in #490** — it was verified-dead code (a registry-dispatch entry point nobody called). Do not reintroduce a `classify()` dispatcher.

The real production surface:

| Symbol | What it does |
| --- | --- |
| `DetectorConfig` (frozen dataclass) | Per-detector knobs: `uses_llm`, `max_input_chars`, `max_output_tokens`, `disable_thinking`, `timeout_seconds`. Pure-structural detectors register `DetectorConfig(uses_llm=False)` (all defaults). Only the two LLM judges set `uses_llm=True`. |
| `register(kind, classifier_fn, config, *, is_async=False)` | Register a `(classifier, config, is_async)` triple keyed by `DriftKind`. Re-registration overwrites (logs DEBUG) so test fixtures don't accumulate stale entries. |
| `get_config(kind)` | Look up a detector's `DetectorConfig` by kind without importing the module. |
| `list_registered()` | The currently-registered `DriftKind` tuple (insertion order). Tests / diagnostics. |
| `truncate_for_observability(text, limit)` | Cap text at `limit` chars with a uniform `" … [truncated]"` suffix (`TRUNCATE_SUFFIX`). Non-str → `""`; `limit <= 0` → no truncation. |
| `format_goals_block(goals)` | Render `Goal` sequences as a numbered `[id] summary` block. Shared by the two LLM-judge prompts. Empty → `"(no goals recorded)"`. |
| `parse_json_response(raw)` | Liberal JSON extractor tolerating markdown fences / prose — used by the LLM judges only (`08-llm-judges.md`), not by the deterministic detectors. Quiet-fails to `None`. |

`_ensure_registered()` imports `capability_check`, `goals`, `reasoning_judge`, `tool_loops` once so the registry is populated for introspection — lazy by design (the steerer already imports these detectors in any realistic runtime). Note: only `capability_check` (structural) and the two LLM judges self-register via `register`. The deterministic reasoning detectors and the embedding backend register nothing. `tool_loops` is imported for parity but `ToolLoopTracker` is instantiated directly by the plugin, not dispatched via the registry.

### `DetectorConfig` fields and what each detector sets

```python
# goldfive/drift/registry.py
@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    uses_llm: bool = False
    max_input_chars: int = 0
    max_output_tokens: int = 0
    disable_thinking: bool = False
    timeout_seconds: float = 0.0
```

| Field | Meaning | Structural detectors | Reasoning judge (`OFF_TOPIC`) | Goal-drift judge |
| --- | --- | --- | --- | --- |
| `uses_llm` | Detector dispatches an LLM call. | `False` | `True` | `True` |
| `max_input_chars` | Cap on the prompt / `trigger_input` observability payload. | `0` (no-op) | 4096 | 2048 |
| `max_output_tokens` | Per-callsite budget around `call_llm`. | `0` | 16384 | 16384 |
| `disable_thinking` | Enter the thinking-disabled LLM path. | `False` | `True` | `True` |
| `timeout_seconds` | Advisory soft budget (not enforced here). | `0.0` | per-detector | per-detector |

The three purely-structural detectors registered (only `CAPABILITY_MISMATCH` actually self-registers; `tool_loops` / `reasoning` embedding detectors are structural but not registered) use all-default `DetectorConfig(uses_llm=False)`. Callers that want a detector's caps without invoking it read them via `get_config(kind)`.

**Honest scope note (from the module docstring):** the Wave-A brief anticipated ~500 LOC reduction across five detectors; in practice only the two LLM-wrapping detectors shared real boilerplate. The value of `registry.py` is the single source of truth for the JSON extractor, the goals renderer, and the truncation helper — not a heavyweight dispatch layer.

---

## Boundary with the LLM judges (`08-llm-judges.md`)

Two detectors in `goldfive/drift/` are LLM judges and are covered in `08-llm-judges.md`, not here: `reasoning_judge` (`OFF_TOPIC` / `JUSTIFIED_DEVIATION`, the three-state classifier over thinking tokens) and `goals` (`GOAL_DRIFT`, the trajectory-level judge). What belongs in *this* chapter is the **deterministic surface around them**:

- The reasoning pipeline's mode selection and the always-on cheap loop detector (this chapter); the judge dispatch itself (chapter 08).
- The goal-drift judge's **scheduling** is partly deterministic and touches this subsystem: turn-based cadence via `note_agent_turn` + `GoalDriftConfig.check_interval`, and **idle-based scheduling via the stall watchdog** consuming `GOAL_DRIFT_IDLE_SECONDS` (this chapter, #487) — the watchdog is the deterministic *producer* that triggers the LLM judge. The judge's prompt/verdict parsing is chapter 08.
- `parse_json_response`, `format_goals_block`, `truncate_for_observability`, and the `DetectorConfig(uses_llm=True)` registrations live in `registry.py` (this chapter) but are consumed by the judges (chapter 08).
- `GOAL_DRIFT` conditions resolve **only at task-terminal** (this chapter's condition-lifecycle section), never on a reasoning-judge on-task verdict.

When code and this chapter disagree with chapter 08 on where a boundary sits, the rule is the same as everywhere: read the code on main.

---

## Condition-lifecycle resolution (#486) — where `current_task_id` / `current_agent_id` become identity

Deterministic detectors don't just fire; their `DriftEvent`s open **conditions** in the state store that must later *resolve* or the active-drift set grows monotonically per run and downstream consumers never see an intervention succeed. This is why the stable-identity-key invariant is load-bearing for detector authors.

- `compute_condition_id(*, kind, task_id, agent_id, turn_id)` = `sha1(f"{kind.value}|{task_id}|{agent_id}|{turn_id}")[:16]` (`goldfive/state_store.py`). Same kind+task+agent within the same turn always hashes identically; a new turn opens a fresh condition.
- **`DRIFT_LIFECYCLE_RESOLVED`** is emitted by `DriftObserver` in two ways (#486): (a) `resolve_conditions_for_terminal_task` — when a task goes terminal (COMPLETED / FAILED / CANCELLED / **NOT_NEEDED**), every condition pinned to it is mooted (no further observation on that task can escalate/recover them); (b) `_resolve_conditions_on_on_task_verdict` — a reasoning-judge ON-TASK verdict resolves only the kinds that pipeline can open (`_REASONING_PIPELINE_DRIFT_KINDS`), staleness-guarded, leaving deterministic-detector conditions (tool loops, task failures) their own lifecycle. `GOAL_DRIFT` resolves **only** at task-terminal.
- Resolution is pure lifecycle telemetry — no intervention decision reads it, so behaviour is identical under `observation_only` True and False. The resolving emit is INFO with `prev_severity` carrying the last recorded severity (so sinks render "recovered from WARNING"), and it deliberately does NOT route through `_emit_drift_detected` (resolution is not a detector decision — no paired `SteeringDecisionMade`, no `session.drift_events` append).

**The detector-author consequence:** if your `current_task_id` or `current_agent_id` churns per observation, `compute_condition_id` mints a fresh condition every fire, the ladder's occurrence counter never advances, and no condition ever resolves. Always stamp a **stable** task id (the plan's task id, not an LLM-minted or per-call value) and the real agent name.

### The two resolution sets (exact contents)

Which conditions resolve is governed by two frozensets — know them exactly:

`TERMINAL_TASK_STATUSES` (`goldfive/types.py`, the **single source of truth** — import it, do not duplicate):

```python
TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.NOT_NEEDED}
)
```

`resolve_conditions_for_terminal_task` runs when a task enters any of these. `NOT_NEEDED` is included (#485) — the planner marks superseded/optional tasks `NOT_NEEDED`, and their open conditions must moot too, or the active-drift set grows on every plan revision.

`_REASONING_PIPELINE_DRIFT_KINDS` (`drift_observer.py`) — the only kinds an ON-TASK reasoning-judge verdict resolves:

```python
_REASONING_PIPELINE_DRIFT_KINDS = frozenset({
    DriftKind.LOOPING_REASONING,
    DriftKind.REASONING_CLUSTER_TIGHTENING,
    DriftKind.OFF_TOPIC,
    DriftKind.JUSTIFIED_DEVIATION,
    DriftKind.INTENT_DIVERGENCE,
})
```

An on-task verdict is the reasoning pipeline's own clean bill for the current `(task, agent, run)`, so only conditions that pipeline can open resolve on it. **Deterministic-detector conditions (tool loops, tool errors, capability mismatch) keep their own lifecycle** — they resolve only at task-terminal, never on a reasoning-judge verdict. `GOAL_DRIFT` also resolves only at task-terminal. If you add a reasoning-pipeline kind, add it to this set or its conditions will never resolve on a clean verdict.

---

## Recipes

Concrete, numbered procedures for the changes you are most likely to be asked to make. Follow them step by step; do not skip a step because it "looks like a one-liner".

### Recipe A — Add a new deterministic (structural) detector

Goal: emit a new `DriftKind` from an observed structural fact, no LLM, no embeddings.

1. **Add the enum value.** In `goldfive/types.py::DriftKind`, add `MY_KIND = "my_kind"` with a docstring comment describing the signal, severity, producer, and issue number — match the existing verbose comment style. If it maps to a proto value, keep the numbering consistent with the `.proto`.
2. **Decide the negative class and lifecycle up front** (see Common Mistake #2). What does "no signal" emit? What resolves the condition your fire opens? Write those down before coding.
3. **Write the detector function.** Put it in the most specific existing module (`drift/__init__.py` for event/text classifiers, a new module for a large detector). Signature returns `DriftEvent | None`. Stamp `kind`, `severity`, `current_task_id` (stable), `current_agent_id`, `detail`, `raw`, and `observed_revision_index` (from `session.plan.revision_index`, captured before any `await`). Stamp `detector_name` if `MY_KIND` is not unique to you.
4. **Never read `observation_only`; never dispatch control; never mutate the session** except a stable-key one-shot flag if strictly needed.
5. **Register it.** At module bottom: `_register(DriftKind.MY_KIND, my_detector, DetectorConfig(uses_llm=False), is_async=False)`. Add the module to `registry._ensure_registered()`'s import list if it is a new module.
6. **Add a ladder row** in `drift_observer.py`'s `_load_ladder_tables` (`09-steering-ladder-and-gates.md`) — an INFO-severity signal usually wants `(OBSERVE, ...)`. Without a row it falls to the default `(INFO→OBSERVE, WARNING→ABSORB, CRITICAL→PAUSE_ESCALATE/ABSORB)`.
7. **Wire the call site.** Find the ADK plugin callback that observes your fact (`05-adk-plugin.md`) and call your detector there, routing the result through `steerer.drift.handle_drift`. Grep for the call site to confirm the detector is not dead code (per the "integration not unit" lesson).
8. **Tests + docs.** Positive case, negative case (returns `None`), and a `list_registered()` membership assert. Run the full suite + `ruff check .`.

Skeleton for step 3+5 (a structural detector):

```python
# goldfive/drift/my_detector.py
from goldfive.types import DriftEvent, DriftKind, DriftSeverity, Session

def detect_my_signal(*, observed_fact, session: Session) -> DriftEvent | None:
    observed_rev = int(getattr(getattr(session, "plan", None), "revision_index", 0) or 0)
    if not _is_bad(observed_fact):        # structural predicate, NOT an NL scan
        return None
    return DriftEvent(
        kind=DriftKind.MY_KIND,
        severity=DriftSeverity.INFO,       # default low; escalate only with tested cancel cost
        detail=f"my signal on {observed_fact!r}",
        current_task_id=str(getattr(session, "current_task_id", "") or ""),  # STABLE key
        current_agent_id="",               # the real agent name if known
        observed_revision_index=observed_rev,
        # detector_name="my_detector",     # stamp only if MY_KIND is shared with another detector
    )

from goldfive.drift.registry import DetectorConfig, register  # noqa: E402
register(DriftKind.MY_KIND, detect_my_signal, DetectorConfig(uses_llm=False), is_async=False)
```

### Recipe B — Add a new "meta" (progress-reporting) tool the tool-loop tracker should treat as cheap

1. If the tool name starts with `report_task_`, do nothing — `_classify_tool_category` already treats it as meta.
2. Otherwise add the literal to `_META_TOOL_NAMES` in `tool_loops.py` (a `frozenset`). Example: `report_awaiting_approval` is there for exactly this reason.
3. If the tool signals genuine task progress on success, make the plugin's `after_tool_callback` call `on_task_progress` on an acknowledged-success response (gate on `{"acknowledged": True}` with no `error`), mirroring the `report_task_*` handling (#192).
4. Add a `test_tool_loops.py` case asserting the tool classifies as meta (INFO until 6 exact) and that errored retries still accumulate.

### Recipe C — Tune the tool-loop name axis to actually escalate

The name axis is signal-only by default (capped at INFO). To make same-name-varied-args escalate:

1. **Preferred:** rely on the exact axis instead — tune `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` (or `ToolLoopConfig.exact_threshold`). Identical-args repeats are never capped and are the higher-confidence loop signal.
2. **If you truly need the name axis to escalate:** set `GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY="warning"` (or `"critical"`). Accept the healthy-varied-args false-positive risk the cap exists to prevent (the #420 debugger-agent regression).
3. Do **not** just lower `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` and expect escalation — that changes which *tier matches* but the emitted severity is still capped without exact corroboration.
4. Verify with a test asserting both `raw["tier"]` and the emitted `severity`.

### Recipe D — Enable the stall watchdog

1. Set `SteeringConfig.stall_watchdog_enabled=True` (or `GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED=1`) and pick `stall_timeout_s` (env `GOLDFIVE_STEER_STALL_TIMEOUT_S`, default 600.0). The flag is default-OFF for a reason — it spawns one asyncio task per dispatch.
2. Confirm the wrapped tree does async I/O in its tools, not sync-blocking work — a sync-blocking tool starves the loop and the watchdog cannot fire (documented limitation).
3. The `TASK_TIMEOUT` drift routes through `handle_drift`, so under `observation_only` (default) it is telemetry-only. To have it *act*, run in active mode (`09-steering-ladder-and-gates.md`).
4. Test with `test_stall_watchdog.py`; the AST-based manifest-liveness test guards the `GOAL_DRIFT_IDLE_SECONDS` live-read.

### Recipe E — Point the embedding detectors at a remote endpoint

1. `export GOLDFIVE_EMBEDDING_BASE_URL=http://host:port` (no trailing slash; `/v1/embeddings` is appended). Optionally `GOLDFIVE_EMBEDDING_MODEL`, `_API_KEY`, `_TIMEOUT_MS`.
2. Or install `goldfive[embedding]` for the local sentence-transformers backend (used only when `BASE_URL` is unset).
3. Switch the reasoning mode to `"embedding"` or `"both"` (via `ReasoningDriftConfig`) — the default `"judge"` mode does not run the embedding pipeline at all.
4. If the endpoint flaps, tune `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S` (default 60) — the breaker trips after 3 consecutive failures and probes once per cooldown.

## History and rationale (the load-bearing PR/issue trail)

Where code and design docs disagree, **the code on main wins** — the design docs under `docs/design/` are a source but the #492 accuracy sweep is the most recent reconciliation, not a guarantee. This trail explains *why* the current shapes exist so you don't "simplify" a scar back into the wound it healed.

| Issue/PR | What it did | Why it matters to you |
| --- | --- | --- |
| #166 | Retired `_GENERIC_VERB_PREFIX_RE` (NL verb-prefix regex). | First of the two "no regex for NL classification" precedents. |
| #167 | Retired `_FACTUAL_QUESTION_RE`. | Second precedent. Both are cited across the drift modules as the reason not to reintroduce lexical NL heuristics. |
| #181 | Original tool-loop detector. | Establishes the tracker's existence. |
| #192 | Made the `on_task_progress` exemption **success-conditional** (in the plugin). | Errored `report_task_*` retries now count toward loop detection; the tracker's reset stays unconditional, the *policy* lives in the plugin. |
| #204 | Graduated meta/work tier tables; tool loops emit `LOOPING_REASONING`. | The whole graduated-severity design; the shared-kind decision. |
| #206 | Retired the args-aware `ToolLoopGuard`; `LOOPING_TOOL_CALL` surfaces kept. | `ToolLoopTracker` is now the sole detector; `LOOPING_TOOL_CALL` is PROTECTED KEEP. |
| #223 / #224 | Off-topic per-sentence min-distance path (raccoon-stimulus calibration). | Long reasoning blocks dilute a brief tangent's whole-text cosine; the sentence path catches it without a new threshold. |
| #225 | Typed per-Runner config (`ToolLoopConfig`, `ReasoningDriftConfig`, `EmbeddingConfig`) + `configure()`. | The process-wide `_CONFIG` override channel in every detector module. |
| #226 / #230 | Unwired + deleted `detect_unreferenced_keyword` and the `_has_unreferenced_keyword` severity bump. | Removed a lexical NL heuristic that fired on generic English; intent-divergence severity is now cosine-only (embedding) / flat-WARNING (pattern). |
| #245 | `observed_revision_index` stamped by every detector before any `await`. | The freshness gate; capture-before-await discipline. |
| #253 | `capability_check` replaces the planner-LLM `PLAN_DIVERGENCE` wrong-assignee check. | Structural capability grounding vs hallucinated planner assignees. |
| #263 | `fallback_to_content_when_no_reasoning` on the reasoning config. | Non-thinking models can synthesise a reasoning signal from the response body. |
| #268 | Rule C (out-of-DAG-order delegation) in `capability_check`. | The lexical-stem banned-class survivor; conservative bail-outs. |
| #271 | Reasoning judge focus verdict + `compute_condition_id` (stable condition identity). | The stable-key rule; the focused-verdict return type in `analyze_reasoning_with_focus`. |
| #420 | Re-key the tool-loop window on `(session.run_id, agent_name)`. | Accumulate across re-invocations; the bucket-key ≠ cancel-target subtlety. |
| #479 | Embedding breaker half-open recovery; malformed judge severity→INFO; sink exceptions never abort; correction keys use full agent path; judge history snapshot-pinned. | The half-open probe; hardening that keeps detectors from aborting runs. |
| #480 | `detector_name` on `DriftEvent` + decision-telemetry label fixes. | Disambiguates the shared `LOOPING_REASONING` kind; stamp it for shared kinds. |
| #484 | Tool-loop name-axis capped at INFO without exact-repeat corroboration. | The corroboration cap; `name_axis_max_severity`, `severity_capped_from`. |
| #485 | Canonical `TERMINAL_TASK_STATUSES` (incl. `NOT_NEEDED`). | The terminal-status set condition-resolution consults. |
| #486 | Drift-condition resolution wired (`DRIFT_LIFECYCLE_RESOLVED`). | Terminal tasks + on-task verdicts moot open conditions; `GOAL_DRIFT` resolves only at task-terminal. |
| #487 | Flag-gated wall-clock stall watchdog = the `TASK_TIMEOUT` producer. | The watchdog; `last_observed_event_at`; idle goal-judge trigger. |
| #490 | Deleted verified-dead code incl. `registry.classify`, the keyword detector, `_LADDER_BY_VALUE`, deprecated shims. | `registry.classify()` stays deleted; don't revive removed detectors. |
| #491 | One internal LLM-call module `goldfive/_llm.py`; per-call diagnostics via `ContextVar`. | The judges (not the deterministic detectors) route through this; capability table `THINKING_DISABLE_CAPABILITIES`. |
| #492 | Design-doc accuracy sweep. | The most recent doc/code reconciliation — but code still wins on any residual disagreement. |

### On the retired NL detectors (the anti-pattern you will be tempted to recreate)

Three detectors were removed because they classified natural language with lexical rules and fired on generic English:

- **`CONFUSION`** (proto value 28, enum comment retained in `types.py`) — counted uncertainty markers in reasoning text.
- **`detect_unreferenced_keyword`** — flagged any 5+ char token absent from goals+task; fired on `wants`, `asking`, `interactive`, `slideshow`.
- **`_has_unreferenced_keyword`** — a severity *bump* on intent-divergence built on the same signal; it pushed real embedding triggers to spurious CRITICAL.

The lesson (recorded in `feedback_no_regex_heuristics` and `17-invariants-hazards-history.md`): the fix for a wanted-but-noisy NL signal is to **teach the LLM judge**, add embeddings, or design the classification away — never a new regex/keyword table. The two survivors (`_pattern_intent_divergence`'s marker regex, `capability_check` Rule C's lexical stems) are demoted, narrowly-scoped fallbacks, explicitly flagged, and are not licence to add more.

## Common mistakes

Concrete wrong edits a weaker model would plausibly make in this subsystem, each with the correct alternative.

### 1. Reintroducing an NL regex/keyword heuristic

**Wrong:** "Add a `CONFUSION` detector that counts uncertainty markers (`maybe`, `not sure`, `I think`) in the reasoning text and emits a drift at WARNING." Or: "Extend `_INTENT_DIVERGENCE_MARKERS` with more phrases." Or: "Add a keyword scan of reasoning text for off-topic words."

**Why it's wrong:** This is the exact anti-pattern goldfive retired repeatedly (`_GENERIC_VERB_PREFIX_RE` #166, `_FACTUAL_QUESTION_RE` #167, `CONFUSION`, `_has_unreferenced_keyword` #226/#230). It fires on generic English vocabulary and produces noisy false CRITICALs.

**Correct:** Teach the LLM reasoning judge (`08-llm-judges.md`) — it covers the same semantic ground robustly. Or, if you have an embedding endpoint, add a cosine-based detector. Exact-hash matching of normalised text (`reasoning_hash`) and exact `(name, args_hash)` matching are the *allowed* deterministic primitives.

### 2. Adding a new detector without a negative-class decision

**Wrong:** Add `detect_foo` that returns a `DriftEvent` when the bad pattern is present, and returns `None` otherwise, with no thought to what "no drift" telemetry looks like or when the condition *resolves*.

**Why it's wrong:** A detector that only ever *opens* conditions (never resolves them, never emits a no-drift/aggregated decision) grows the active-drift set monotonically. The tool-loop tracker emits **aggregated no-drift decisions** (#484) and the stall watchdog / reasoning judge resolve conditions on the positive lifecycle. Silent-on-clean detectors leave operators unable to tell "ran and found nothing" from "never ran".

**Correct:** Decide up front: (a) what the negative class emits (an aggregated no-drift decision, or nothing but a resolvable condition), and (b) which lifecycle event resolves your condition (task-terminal via `resolve_conditions_for_terminal_task`, an on-task verdict, or your own explicit resolution). Register the detector's `DetectorConfig`. See how `tool_loops` emits aggregated tool-loop no-drift decisions and how #486 wires resolution.

### 3. Unstable condition keys

**Wrong:** Stamp `current_task_id` with an LLM-minted id, a per-invocation UUID, or a value that changes every observation ("to make each fire unique").

**Why it's wrong:** `compute_condition_id` hashes `(kind, task_id, agent_id, turn_id)`. A churning task id opens a fresh condition per fire; the intervention ladder's occurrence counter (which drives escalation from ABSORB → CANCEL_REINVOKE → PAUSE_ESCALATE) never advances, and the condition never resolves. This is the stable-keys invariant.

**Correct:** Stamp `current_task_id` from `session.current_task_id` / the plan's stable task id, and `current_agent_id` from the real agent name. If a component upstream is churning the id, fix the churn upstream — don't coarsen the key.

### 4. Forgetting the corroboration cap when tuning tool-loop thresholds

**Wrong:** "The name axis should escalate sooner, so lower `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` from 5 to 3." Then observe that it still only produces INFO drifts and conclude the change "didn't work" or the code is broken.

**Why it's wrong:** The #484 corroboration cap (`name_axis_max_severity`, default `"info"`) caps the *emitted* severity of any name-axis hit **without** ≥2 identical-args repeats — independent of the threshold. Lowering the threshold makes the tier *match* sooner, but the emitted severity is still capped at INFO (→ OBSERVE, no plan mutation). `raw["tier"]` shows WARNING, `raw["severity_capped_from"]` shows what would have fired.

**Correct:** Understand that the name axis is **signal-only by default**. To make it escalate you must either (a) set `name_axis_max_severity="warning"` / `"critical"` (accepting the healthy-varied-args false-positive risk the cap exists to prevent), or (b) rely on exact-repeat corroboration (the exact axis is never capped — tune `exact_threshold` instead). For genuine redundant-work loops, the exact axis is the right knob.

### 5. "Fixing" the `LOOPING_REASONING` label on tool loops

**Wrong:** "The tool-loop tracker emits `LOOPING_REASONING` but it's a *tool* loop — repoint it at `LOOPING_TOOL_CALL`" (or delete `LOOPING_TOOL_CALL` as unused).

**Why it's wrong:** The kind choice is the routing choice (#204 — `LOOPING_REASONING` has the NUDGE-first CRITICAL ladder row wanted for tool loops); the source is disambiguated by `detector_name="tool_loops"` (#480). `LOOPING_TOOL_CALL` and its ladder/promotion/planner surfaces are a **PROTECTED KEEP** decision (#204/#206). Both edits break routing or delete protected surface.

**Correct:** Leave the kind as `LOOPING_REASONING` and the `detector_name` stamp in place. If telemetry misattributes a tool loop, check that `detector_name` is being stamped and read by `DriftObserver._detector_name_for_drift`, not that the kind is "wrong".

### 6. Making the embedding detectors mandatory or crashing when the encoder is absent

**Wrong:** Assume `_embed.max_similarity` / `distance_to_topic` always return a real score, or raise when the encoder is unavailable.

**Why it's wrong:** The default install has no embedding backend. `max_similarity` returns `0.0` and `distance_to_topic` returns `-1.0` as "no signal" sentinels; the detectors are written to no-op silently on those. Adding a `raise`, or treating `0.0` as a genuine loop, would break every default-install run.

**Correct:** Gate embedding use on `_embed.available()` (as `detect_intent_divergence` does) or handle the sentinel (`if dist >= 0 and dist >= threshold`). Preserve the graceful-degrade contract.

### 7. Reading `observation_only` inside a detector

**Wrong:** Add `if steerer.is_active_steering(): emit_drift()` inside a detector, or gate emission on the kill-switch.

**Why it's wrong:** Detectors are strictly emit-only. Passivity is enforced downstream at dispatch. A detector that gates its own emission on `observation_only` breaks telemetry (operators lose the signal that the detector ran) and duplicates a concern that lives in exactly one sanctioned place (`DefaultSteerer.is_active_steering()` / `steering_is_active(steerer)`, #488). See `09-steering-ladder-and-gates.md`.

**Correct:** Emit the `DriftEvent` unconditionally. Let `handle_drift` and the ladder decide whether to act.

### 8. Stamping `observed_revision_index` after an `await`

**Wrong:** In an async detector (a judge, or a detector that does I/O), read `session.plan.revision_index` *after* the `await`.

**Why it's wrong:** The freshness gate relies on the observed revision being captured at the moment of observation, *before* the round-trip. Reading it after the await lets the reconciler advance the plan between observation and emit without the gate noticing — the whole point of #245.

**Correct:** Capture `observed_revision_index` at the TOP of the detector, before any `await`, exactly as `_observed_revision_index(session)` is called in the reasoning detectors and as the stall watchdog does inline.

### 9. Extending Rule C's lexical stem matching

**Wrong:** "Rule C misses a case; add a synonym table / fuzzy-match / a regex over the task description to catch it."

**Why it's wrong:** Rule C is already a flagged banned-class survivor. Every lexical addition drifts it further toward meaning-based NL classification and raises false-fire risk on a detector whose every fire cancels an invocation.

**Correct:** Strengthen the *structural* signal instead — populate `required_tools` advisories (Rule B is higher-confidence), or gate the delegation pin on DAG-readiness so the out-of-order delegation never binds in the first place. Keep the lexical surface exactly as narrow as it is.

### 10. Forgetting to thread `session_run_id` in a tool-loop test

**Wrong:** Write a test that calls `observe_tool_call(invocation_id=..., agent_name=..., ...)` a fresh `invocation_id` each iteration (simulating re-invocation) without `session_run_id`, then assert CRITICAL fires.

**Why it's wrong:** Without `session_run_id`, the bucket key falls back to `(invocation_id, agent_name)` — each re-invocation is an isolated 1-2 entry window, exactly the #420 bug the run-scope key fixed. Your loop never accumulates and the assertion fails (or worse, passes for the wrong reason).

**Correct:** Thread a constant `session_run_id="run-1"` through every `observe_tool_call` in the test to accumulate across re-invocations, matching what the ADK plugin does. Use `buffer_size(session_run_id="run-1", agent_name=...)` to introspect.

### 11. Adding a `name` threshold to the meta category

**Wrong:** "Meta tools should also catch same-name loops — add `"name": 6` to `_META_THRESHOLDS["warning"]`."

**Why it's wrong:** Same-name-varied-args on a reporting tool is meaningless (a coordinator reporting on six different tasks with `report_task_completed` is healthy). The meta name axis is deliberately all-`None`. Adding a threshold reintroduces false positives on healthy reporting.

**Correct:** Leave meta name thresholds `None`. If a specific meta tool genuinely loops with identical args, the meta exact axis (3/6/10) already catches it.

### 12. Assuming a reasoning detector mutates `reasoning_history`

**Wrong:** In a new reasoning detector, append `text` to `session.reasoning_history` "so the next turn sees it".

**Why it's wrong:** The steerer appends the new block to `reasoning_history` **before** running the pipeline. A detector that appends again double-counts and can create a self-match (the block matching itself). The detectors read `history[-hash_window-1:-1]` — a slice that deliberately excludes the last entry (the current block) precisely because the steerer already added it.

**Correct:** Read `session.reasoning_history` (or the passed `history` snapshot); never mutate it. The only sanctioned mutation is `session.reasoning_cluster_flagged.add(task_id)` for the one-shot cluster guard.

### 13. Clearing the tool-loop buffer from the wrong place

**Wrong:** Call `tracker.clear()` or `on_task_progress` inside a detector or on every task transition unconditionally in the plugin.

**Why it's wrong:** `clear()` drops **every** buffer (cross-agent) and is only for session teardown (`clear_active_context`). `on_task_progress` must be gated on acknowledged-success (#192) — calling it on every transition (including errored reports) re-opens the exact hole #192 closed.

**Correct:** `clear()` only from `clear_active_context`. `on_task_progress` only from the plugin's `after_tool_callback` on a `{"acknowledged": True}` response with no `error`.

### 14. Emitting CRITICAL from a structural detector without accepting the cancel cost

**Wrong:** A new structural detector emits CRITICAL "to be safe" / "so it's visible".

**Why it's wrong:** In active mode CRITICAL is the only severity that reaches cancel-reinvoke / pause-escalate. A CRITICAL that fires on an ambiguous signal cancels in-flight invocations and triggers refines on false positives — expensive and disruptive. `capability_check` fires CRITICAL only because its three rules are surgically narrow and false-positives there are explicitly worse than false-negatives.

**Correct:** Default to INFO (OBSERVE, telemetry-only) or WARNING (ABSORB, plan-extension). Reserve CRITICAL for signals where you have accepted — and tested — the cancel/refine cost, with conservative bail-outs like `capability_check`'s.

### 15. Treating the `mode="embedding"` default as production behaviour

**Wrong:** Read `analyze_reasoning`'s `mode="embedding"` default and conclude the embedding pipeline runs in production.

**Why it's wrong:** `DEFAULT_REASONING_DRIFT_MODE = "judge"` and the steerer always passes an explicit mode. The free-function default is a legacy artefact. In a default install with no embedding endpoint, the embedding detectors are unreachable regardless.

**Correct:** Assume `mode="judge"` for production reasoning drift. The embedding detectors are an opt-in surface (Recipe E). Test embedding detectors by calling them directly or via `_embedding_pipeline`, not by relying on the free-function default.

### 16. Duplicating `TERMINAL_TASK_STATUSES` or dropping `NOT_NEEDED`

**Wrong:** Inline a `{COMPLETED, FAILED, CANCELLED}` set in a new terminal check, or "clean up" the `NOT_NEEDED` member as an edge case.

**Why it's wrong:** `TERMINAL_TASK_STATUSES` is the single source of truth (`goldfive/types.py`) consumed by the steerer, tool-dispatch, the ADK adapter, and condition resolution. Dropping `NOT_NEEDED` (#485) means conditions on planner-superseded tasks never moot and the active-drift set grows on every revision.

**Correct:** `from goldfive.types import TERMINAL_TASK_STATUSES` and use it. If `TaskStatus` gains a terminal member, add it there once and every consumer sees it.

### 17. Reviving `registry.classify()` or a deleted detector

**Wrong:** "There's no dispatch entry point on the registry — add a `classify(kind, ...)` that looks up and calls the classifier." Or re-add `detect_unreferenced_keyword` / the keyword `CONFUSION` detector because "the signal would be useful".

**Why it's wrong:** `registry.classify` was verified-dead and deleted in #490; nobody dispatched through it. The keyword detectors were deleted as NL anti-patterns. Re-adding either resurrects removed complexity or a banned class.

**Correct:** Call detectors at their real call sites (the plugin callbacks / steerer), which is how they are actually reached. For a wanted NL signal, teach the LLM judge.

---

## Where each detector's state lives

Most detectors are stateless functions. The ones that hold state, and where:

| Detector | State | Lives on | Lifetime |
| --- | --- | --- | --- |
| `ToolLoopTracker` | Per-`(scope, agent)` ring buffers | A single tracker instance on the ADK plugin | One run; `clear()` on `clear_active_context`. |
| `detect_reasoning_cluster_tightening` | One-shot `reasoning_cluster_flagged` set | `session.reasoning_cluster_flagged` | Session; keyed by `current_task_id`. |
| reasoning history-window detectors | `reasoning_history` list (read-only) | `session.reasoning_history` | Session; the steerer appends, detectors read. |
| `_embed` encoder + breaker + cache | `_MODEL`, breaker counters, LRU | Module globals in `_embed.py` | Process; reset via `configure`/test hatches. |
| stall watchdog | `episode_fires`, `last_watermark`, `goal_judge_fired` | Locals in the `_run_stall_watchdog` task | One dispatch; task cancelled on teardown. |
| `_CONFIG` overrides | Installed config | Module globals in each detector module | Process; `configure(None)` clears. |

The tracker state is intentionally on the plugin, not per-session-persisted — the whole window is ephemeral to one run, and holding it in one place lets `goldfive.wrap(runtime=...)` influence the tracker the plugin builds. Detectors' module-global `_CONFIG` and `_embed`'s process globals are the two places cross-test leakage happens; reset them in teardown.

## Constants and env-var quick reference

Consolidated defaults across the subsystem (all overridable per the precedence rules in each section):

| Constant / env var | Default | Module |
| --- | --- | --- |
| `DEFAULT_WINDOW` / `GOLDFIVE_TOOL_LOOP_WINDOW` | 10 | `tool_loops` |
| `DEFAULT_EXACT_THRESHOLD` / `GOLDFIVE_TOOL_LOOP_EXACT_THRESHOLD` | 3 | `tool_loops` (work-WARNING exact) |
| `DEFAULT_NAME_THRESHOLD` / `GOLDFIVE_TOOL_LOOP_NAME_THRESHOLD` | 5 | `tool_loops` (work-WARNING name) |
| `DEFAULT_ALTERNATING_THRESHOLD` / `GOLDFIVE_TOOL_LOOP_ALTERNATING_THRESHOLD` | 5 | `tool_loops` |
| `DEFAULT_NAME_AXIS_MAX_SEVERITY` / `GOLDFIVE_TOOL_LOOP_NAME_AXIS_MAX_SEVERITY` | `"info"` | `tool_loops` |
| `_NAME_AXIS_CORROBORATION_MIN_EXACT` | 2 | `tool_loops` (not env-tunable) |
| `LOOPING_REASONING_HASH_WINDOW` | 5 | `reasoning` |
| `LOOPING_REASONING_SIMILARITY_THRESHOLD` | 0.9 | `reasoning` |
| `REASONING_CLUSTER_SIMILARITY_THRESHOLD` | 0.75 | `reasoning` |
| `OFF_TOPIC_DISTANCE_THRESHOLD` | 0.7 | `reasoning` |
| `INTENT_DIVERGENCE_HEALTHY / MINOR / WARNING_SIMILARITY` | 0.6 / 0.4 / 0.2 | `reasoning` |
| `SENTENCE_LEVEL_MIN_BLOCK_LENGTH` / `_MAX_SENTENCES` | 200 / 10 | `reasoning` |
| `DEFAULT_REASONING_DRIFT_MODE` | `"judge"` | `reasoning` |
| `GOLDFIVE_EMBEDDING_BASE_URL` / `_MODEL` / `_API_KEY` / `_TIMEOUT_MS` | unset / "" / none / 10000 | `_embed` |
| `_RUNTIME_FAILURE_THRESHOLD` | 3 | `_embed` (breaker trip) |
| `_RUNTIME_RECOVERY_COOLDOWN_S` / `GOLDFIVE_EMBEDDING_BREAKER_COOLDOWN_S` | 60.0 | `_embed` (half-open) |
| `_CACHE_MAX` | 512 | `_embed` (LRU) |
| `_DEFAULT_MODEL_NAME` | `all-MiniLM-L6-v2` | `_embed` (sentence-transformers) |
| `GOAL_DRIFT_IDLE_SECONDS` | 300 | `goals` (live-read by watchdog) |
| `GOAL_DRIFT_CHECK_INTERVAL` | 5 | `goals` (back-compat re-export, not runtime) |
| `stall_watchdog_enabled` / `GOLDFIVE_STEER_STALL_WATCHDOG_ENABLED` | `False` | `config.SteeringConfig` |
| `stall_timeout_s` | 600.0 | `config.SteeringConfig` |

## Verification checklist

Run these after touching anything in this chapter's surface. Commands assume repo root `/home/sunil/git/goldfive` and the dev+adk extras installed (`uv sync --extra dev --extra adk`).

### Targeted test runs by file touched

| If you touched... | Run |
| --- | --- |
| `tool_loops.py` | `uv run pytest -q tests/test_tool_loops.py tests/test_tool_loop_name_axis_precision.py tests/test_tool_loop_exemption_tightening.py` |
| `reasoning.py` | `uv run pytest -q tests/test_drift_reasoning.py tests/test_reasoning_sentence_level.py tests/test_reasoning_mode_dispatch.py tests/test_reasoning_content_fallback.py tests/test_reasoning_logging.py` |
| `_embed.py` | `uv run pytest -q tests/test_embedding_backend.py` |
| `capability_check.py` | `uv run pytest -q tests/test_capability_mismatch.py tests/test_descriptive_growth_capability_mismatch_fallback.py` |
| `registry.py` | `uv run pytest -q tests/test_drift_registry.py` |
| `drift/__init__.py` classifiers | `uv run pytest -q tests/test_drift_classifiers.py tests/test_drift_taxonomy.py` |
| stall watchdog | `uv run pytest -q tests/test_stall_watchdog.py` |
| condition lifecycle / resolution | `uv run pytest -q tests/test_drift_lifecycle.py tests/test_drift_resolution_wiring.py tests/test_drift_outcomes.py tests/test_terminal_drift_closes_spans.py` |
| `DriftEvent` / detector_name / routing | `uv run pytest -q tests/test_drift_version_pin.py tests/test_goldfive_drift_routing.py` |

### Test-file reference (what each suite actually asserts)

| Test file | Asserts |
| --- | --- |
| `tests/test_tool_loops.py` | Core tier/axis behaviour, alternating mode, window keying, `on_task_progress` reset. |
| `tests/test_tool_loop_name_axis_precision.py` | The #484 cap: capped INFO, `severity_capped_from`, corroborated escalation. |
| `tests/test_tool_loop_exemption_tightening.py` | #192 success-conditional `on_task_progress`; errored reports accumulate. |
| `tests/test_drift_reasoning.py` | Reasoning pipeline ordering, intent-divergence bands, loop detection. |
| `tests/test_reasoning_sentence_level.py` | The #224 per-sentence off-topic path. |
| `tests/test_reasoning_mode_dispatch.py` | `judge` / `embedding` / `both` / `off` mode selection + tie-breaks. |
| `tests/test_reasoning_content_fallback.py` | `fallback_to_content_when_no_reasoning` (#263). |
| `tests/test_embedding_backend.py` | Backend selection, breaker trip/half-open, LRU, `_parse_openai_response`. |
| `tests/test_capability_mismatch.py` | Rules A/B/C, ordering, bail-outs. |
| `tests/test_drift_registry.py` | `register` / `get_config` / `list_registered`; `DetectorConfig`. |
| `tests/test_drift_classifiers.py` | `classify_tool_error` / `_refusal` / `_confabulation_risk` / `_stop_reason`. |
| `tests/test_drift_taxonomy.py` | `DriftKind` / `DriftSeverity` enum pins (protected KEEP values). |
| `tests/test_stall_watchdog.py` | Watchdog graduated severity, idle judge trigger, LLM-watcher skip, manifest-liveness. |
| `tests/test_drift_lifecycle.py`, `test_drift_resolution_wiring.py` | Condition open/resolve, `DRIFT_LIFECYCLE_RESOLVED`, terminal + on-task resolution. |
| `tests/test_drift_outcomes.py` | Decision outcomes / aggregated no-drift decisions. |
| `tests/test_goldfive_drift_routing.py` | `detector_name` attribution through routing. |
| `tests/test_drift_version_pin.py` | `observed_revision_index` freshness gate. |

### Full-suite gates (always, before you call it done)

```bash
uv run pytest -q                 # ~30s, expect ~2912 passed / ~61 skipped
ruff check .                     # must stay clean
```

Do **not** run `ruff format` — the repo is intentionally not format-clean; a mass reformat pollutes the diff.

### Grep audits (invariant guards)

Run these to confirm you did not reintroduce a banned pattern or break a stamp:

```bash
# 1. No new NL regexes in the reasoning/capability detectors. The only
#    tolerated regexes are the demoted _INTENT_DIVERGENCE_MARKERS fallback
#    and the sentence-splitter. A NEW re.compile over reasoning/task text
#    is a red flag — justify it against #166/#167 before committing.
grep -n "re\.compile\|re\.search\|re\.findall" goldfive/drift/reasoning.py goldfive/drift/capability_check.py

# 2. detector_name is stamped wherever the kind is shared. Every DriftEvent
#    with kind=LOOPING_REASONING minted by tool_loops MUST carry
#    detector_name="tool_loops".
grep -n "detector_name" goldfive/drift/tool_loops.py

# 3. No detector reads the kill-switch. Should return ZERO hits in drift/.
grep -rn "is_active_steering\|steering_is_active\|observation_only" goldfive/drift/

# 4. observed_revision_index is stamped before any await in async detectors.
grep -n "observed_revision_index" goldfive/drift/reasoning.py

# 5. LOOPING_TOOL_CALL protected KEEP surface is intact (enum + ladder).
grep -rn "LOOPING_TOOL_CALL" goldfive/types.py goldfive/drift_observer.py goldfive/steerer.py

# 6. registry.classify() stays deleted (#490).
grep -rn "def classify\b" goldfive/drift/registry.py     # expect NO match

# 7. The tool-loop tracker still keys on session_run_id first (the #420 fix).
grep -n "scope_key = session_run_id or invocation_id" goldfive/drift/tool_loops.py

# 8. The name-axis corroboration cap is still 2 and the exact axis is uncapped.
grep -n "_NAME_AXIS_CORROBORATION_MIN_EXACT" goldfive/drift/tool_loops.py

# 9. TERMINAL_TASK_STATUSES still includes NOT_NEEDED (#485).
grep -n "NOT_NEEDED" goldfive/types.py

# 10. The stall watchdog reads GOAL_DRIFT_IDLE_SECONDS live (module attr, not captured).
grep -n "GOAL_DRIFT_IDLE_SECONDS\|_goal_drift_idle_seconds" goldfive/adapters/_adk_plugin.py
```

### Symbol → chapter cross-reference

When a symbol you meet here is really owned by another subsystem, follow it there:

| Symbol / concept | Chapter |
| --- | --- |
| `DefaultSteerer.observe_reasoning`, `handle_drift`, the intervention ladder, `is_active_steering` | `09-steering-ladder-and-gates.md` |
| ADK callbacks (`after_tool_callback`, `after_model_callback`, `set_active_context`), the stall-watchdog task lifecycle | `05-adk-plugin.md` |
| `reasoning_judge`, `goals` (LLM judges), `parse_json_response` consumers, `_llm.py` | `08-llm-judges.md` |
| `DriftDetected` / `SteeringDecisionMade` / `ReasoningJudgeInvoked` wire mapping, sinks | `12-events-sinks-telemetry.md` |
| `ToolLoopConfig`, `ReasoningDriftConfig`, `EmbeddingConfig`, `SteeringConfig` field docs | `14-config-reference.md` |
| `compute_condition_id`, `resolve_drifts_matching`, `KEY_ACTIVE_DRIFTS` | `11-state-ownership.md` |
| `TERMINAL_TASK_STATUSES`, `Task`, `TaskStatus`, plan/revision | `10-planning-and-revision.md`, `11-state-ownership.md` |
| The invariants, retired-detector history, protected KEEP surfaces | `17-invariants-hazards-history.md` |

### Invariant → guard mapping

Each subsystem invariant has a concrete guard. After a change, confirm the relevant guard still holds:

| Invariant | Guard |
| --- | --- |
| No NL regex/keyword heuristics | Grep #1 above; `test_drift_reasoning.py` asserts intent-divergence severity comes from cosine bands, not lexical bumps. |
| Adaptive over predictive | Review: your detector reads an observed fact, not a predicted action. No test enforces this — it is a design check. |
| No prompt-cooperation contract | `on_task_progress` is driven by the plugin observing a transition; grep the plugin call site, not the agent's tool calls. |
| `observation_only` strictly passive | Grep #3 (zero hits in `drift/`); `test_stall_watchdog.py` / routing tests confirm `handle_drift` gates action, not the detector. |
| Stable identity keys | `test_drift_lifecycle.py` + `compute_condition_id` unit coverage; a churn regression shows up as conditions never resolving. |
| Protected KEEP surfaces | Grep #5 (`LOOPING_TOOL_CALL` present); `test_drift_taxonomy.py` pins the enum. |
| `detector_name` on shared kinds | Grep #2; `test_goldfive_drift_routing.py` asserts tool-loop fires attribute to `tool_loops`. |

### Testing patterns (how to write a good test here)

- **Tool-loop tests** thread a constant `session_run_id` and build the window with repeated `observe_tool_call`; assert on the returned `list[DriftEvent]` — its length, `[0].severity`, `[0].raw["mode"]`, `[0].raw["tier"]`, and (for cap cases) `[0].raw["severity_capped_from"]`. Use `buffer_size(...)` to sanity-check accumulation. Leave `exact_threshold`/`name_threshold` at defaults when exercising CRITICAL.
- **Reasoning-detector tests** install a fake encoder via `_embed.set_model(FakeEncoder(...))` whose `encode` returns scripted vectors, or pass an `embedding_pipeline` / `judge_runner` test seam to `analyze_reasoning*`. Always `_embed.set_model(None)` (or use the autouse teardown) so vectors don't leak across cases. For the no-embeddings path, `_embed.force_unavailable()`.
- **Embedding-backend tests** use `set_backend_loader` / `set_backend_class` to assert which `base_url`/kwargs the lazy-load path selects, and drive `_note_backend_failure` × 3 to assert the trip, then `_recovery_cooldown_s` + a probe to assert half-open recovery. Reset with `reset_circuit_breaker()` between cases.
- **Capability-check tests** pass duck-typed tool stubs (objects with `.agent` for AgentTool, `.name`/`.func` for FunctionTool) — no live ADK needed. Assert rule ordering (B → A → C) by constructing inputs that would trip more than one rule and checking which detail string comes back.
- **Config-override tests** call the detector module's `configure(SomeConfig(...))` then assert the threshold moved, and `configure(None)` in teardown to avoid cross-test leakage (the `_CONFIG` module globals are process-wide).

### Behavioural spot-checks (weak-model traps)

- After changing a tool-loop threshold, add/adjust a test asserting BOTH the matched `raw["tier"]` AND the emitted `severity` — the cap can make those disagree, and a test that only asserts the tier passes on a mis-routed severity.
- After changing an embedding threshold, confirm the no-embeddings path still no-ops: `test_embedding_backend.py` exercises `force_unavailable()` / breaker-trip; add a case if your change adds a new sentinel branch.
- After adding a detector, assert it appears in `list_registered()` (if you registered it) and that a clean input returns `None` / an aggregated no-drift decision — never a silent nothing that leaves an unresolvable condition open.
- After touching condition resolution, run `test_drift_resolution_wiring.py` and confirm `NOT_NEEDED` transitions still resolve open conditions (the #485 inclusion is easy to drop when refactoring the terminal-status check).

### Definition of done for a drift-detection change

Before you call a change complete, confirm every line:

- [ ] The detector emits only; it does not dispatch, cancel, or read `observation_only`.
- [ ] No new NL regex/keyword table over reasoning or task text (grep #1 clean).
- [ ] `observed_revision_index` stamped before any `await`; `current_task_id` is a stable plan id.
- [ ] `detector_name` stamped if the kind is shared with another detector.
- [ ] Negative class defined (returns `None` or an aggregated no-drift decision) and the condition has a resolution path.
- [ ] Protected KEEP surfaces untouched (`LOOPING_TOOL_CALL`, `PLAN_DIVERGENCE`, `reconciler.get_missed_tasks`).
- [ ] Registered with a `DetectorConfig` if it self-registers; added to `_ensure_registered` if a new module.
- [ ] Targeted tests + a `list_registered()`/routing assertion where applicable.
- [ ] `uv run pytest -q` green (~2912 passed / ~61 skipped); `ruff check .` clean; no `ruff format`.
