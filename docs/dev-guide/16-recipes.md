# 16. Recipes

## Read this chapter when...

You are about to make a **specific kind of change** to goldfive and want a
numbered, copy-adaptable procedure instead of re-deriving the pattern from
the code. Each recipe is a self-contained checklist: preconditions, the
exact files and symbols to edit, a verbatim skeleton to adapt, the tests to
write (by name), the verification commands to run, and an "If any step
fails" triage block. The recipes are ordered from most-common (add a drift
kind) to most-delicate (delete dead code, edit a design doc).

This chapter supersedes `.agents/how-to-add-a-drift-kind.md` and
`.agents/how-to-add-a-new-adapter.md` where they disagree with it — those
skills predate the retirement of the regex classifiers (#166/#167), the
registry (`goldfive/drift/registry.py`), the intervention ladder living on
`DriftObserver` (not `DefaultSteerer`), and the strict-passive
`observation_only` regime (Waves 1-4). **Where a skill and this chapter
disagree, this chapter wins; where this chapter and the code on `main`
disagree, the code wins** — verify every citation before you edit.

## Files covered

Recipes touch nearly every subsystem, so this list is per-recipe:

| # | Recipe | Primary files |
|---|---|---|
| 1 | Add a `DriftKind` | `goldfive/types.py`, `proto/goldfive/v1/types.proto`, `goldfive/drift/`, `goldfive/drift_observer.py`, `docs/design/DRIFT.md` |
| 2 | Add a deterministic detector | `goldfive/drift/*.py`, `goldfive/drift/registry.py`, `goldfive/config.py` |
| 3 | Add / extend an LLM judge | `goldfive/drift/reasoning_judge.py`, `goldfive/drift/goals.py`, `goldfive/judges/` |
| 4 | Add an intervention surface | `goldfive/drift_observer.py`, `goldfive/steerer.py` |
| 5 | Add a config knob | `goldfive/config.py`, `goldfive/steerer.py` |
| 6 | Add a proto field | `proto/goldfive/v1/*.proto`, `goldfive/events.py` |
| 7 | Add an event sink | `goldfive/sinks/`, `goldfive/protocols.py` |
| 8 | Add an ADK observation point | `goldfive/adapters/_adk_plugin.py` |
| 9 | Add a reporting tool | `goldfive/reporting/` |
| 10 | Add an adapter | `goldfive/adapters/`, `goldfive/protocols.py` |
| 11 | Safe dead-code deletion | anywhere + `tests/` |
| 12 | Update a design doc | `docs/design/*.md` |

## Invariants that bind you here

Every recipe below is written to keep these true. Re-read them before you
adapt any skeleton:

1. **No prompt-cooperation contracts.** Detection, termination, and
   observability must work even if the agent never calls a goldfive tool or
   follows an instruction. Recipes never add "the agent must call X".
2. **No regex/keyword heuristics for NL classification.** #166 retired
   `_GENERIC_VERB_PREFIX_RE`, #167 retired `_FACTUAL_QUESTION_RE`, and #490
   deleted the last keyword detector (`detect_unreferenced_keyword`). New
   NL classification goes through an LLM judge or is designed away.
   Exact-equality / hash matching of **structured** data (tool name + args
   hash, task-status enums) is allowed and encouraged.
3. **Any ADK tree shape works,** including coordinator + `AgentTool`. Recipe
   8 and 10 never assume a flat single-agent tree.
4. **Adaptive over predictive.** Capture observed facts via extended protos
   / events; do not intercept at pin/dispatch time to predict what the agent
   will do next.
5. **`observation_only=True` is the production default and is strictly
   passive.** The ONLY sanctioned read of the kill-switch is
   `DefaultSteerer.is_active_steering()` (in `goldfive/steerer.py`) or the
   module helper `steering_is_active(steerer)` (also `goldfive/steerer.py`),
   which returns `False` for a missing/None/raising steerer. Never read
   `SteeringConfig.observation_only` or `DefaultSteerer._observation_only`
   directly from a new call site.
6. **Lifecycle gates need stable identity keys.** Never key a gate, cooldown,
   or dedupe set on an LLM-minted or churning id. Use the emit-time
   `state_store.compute_condition_id(...)` value, the full agent path,
   `(kind, task_id)`, or a content hash. (`condition_id` is computed at emit
   and stamped on the proto `DriftDetected` — it is not a `DriftEvent`
   attribute.)

Cross-references: the ladder itself is `09-steering-ladder-and-gates.md`;
detectors are `07-deterministic-drift-detection.md`; judges are
`08-llm-judges.md`; config is `14-config-reference.md`; events/sinks are
`12-events-sinks-telemetry.md`; reporting tools are
`13-reporting-tools-and-approval.md`; the plugin is `05-adk-plugin.md`;
adapters are `06-adapters-and-instrumentation.md`; testing is
`15-testing-guide.md`; the invariants in full are
`17-invariants-hazards-history.md`.

---

## Recipe 1 — Add a `DriftKind` end-to-end

**What this gives you.** A new drift signal that flows: Python enum → proto
enum → generated stubs → detector/classifier → intervention ladder row →
`DriftDetected` on the wire → taxonomy doc. This is the most common
extension and the one with the most moving parts.

**Preconditions.**

- You have `uv sync --extra dev --extra adk --extra proto` in a clean tree.
- You know the drift's **default severity** (`INFO` / `WARNING` / `CRITICAL`,
  or "graduated" if the same kind fires at different severities based on the
  observation).
- You know whether it is **detected by** a structural detector (Recipe 2), an
  LLM judge (Recipe 3), or emitted inline from a reporting-tool handler
  (Recipe 9).
- You know whether it is **recoverable** (refine can route around it) or
  **terminal** (must escalate to a human).

### Step 1.1 — Add the Python enum value

File: `goldfive/types.py`, class `DriftKind` (a `StrEnum`, at the top of the
enum block). Append your member with a docstring-style comment. Place it near
kins of the same category so readers find it where they expect.

```python
# goldfive/types.py — inside class DriftKind(StrEnum)
    # One-line trigger. Default severity: WARNING. Recoverable: refine
    # routes around it. Detector: goldfive.drift.<module>.<fn>. See
    # goldfive#NNN. If severity is *graduated*, say so explicitly here —
    # callers filtering by kind must not assume a fixed severity.
    MY_NEW_DRIFT = "my_new_drift"
```

Rules:

- The **string value** is `snake_case` of the member name. `StrEnum` means
  `DriftKind.MY_NEW_DRIFT == "my_new_drift"` and it serialises as that
  string. Keep the value equal to `name.lower()` — several call sites do
  `kind.value` ↔ `kind.name` round-trips.
- **Never renumber or reuse a retired value.** `CONFUSION` (proto 28) is
  retired and reserved; do not add a lexical/keyword detector back under any
  name (invariant 2).
- If the kind fires at **graduated severity** (like `INTENT_DIVERGENCE` at
  line ~157, or `AGENT_REFUSAL`), the classifier picks the severity per
  observation; the kind stays stable. Say so in the comment.

### Step 1.2 — Mirror the value in the proto enum

File: `proto/goldfive/v1/types.proto`, enum `DriftKind`. **Append** a
`DRIFT_KIND_<NAME>` member at the next free number. As of PR #492 the highest
assigned slot is `DRIFT_KIND_CAPABILITY_MISMATCH = 41`, so the next new kind
is `42`. Verify the current maximum before you pick — never fill a gap, never
collide.

```proto
// proto/goldfive/v1/types.proto — inside enum DriftKind
  // One-line trigger. See goldfive#NNN.
  DRIFT_KIND_MY_NEW_DRIFT = 42;
```

**The Python↔proto bridge is by name.** The emit path resolves the proto
value by name via `getattr(types_pb2, f"DRIFT_KIND_{kind.name}")` — see
`DefaultSteerer._drift_kind_pb_value` in `goldfive/steerer.py`, used on the
real `DriftDetected` emit at `DriftObserver._emit_drift_detected`
(`goldfive/drift_observer.py:484`). Deserialisation is the reverse via
`pb.DriftKind.Value("DRIFT_KIND_" + name)`. The proto suffix **must** equal the
Python enum member name exactly: `MY_NEW_DRIFT` ↔ `DRIFT_KIND_MY_NEW_DRIFT`. A
mismatch silently drops the kind to `UNSPECIFIED` on the wire. (Do **not** copy
the mechanism from the `drift_detected_event()` factory in
`goldfive/events.py` — it calls `pb.DriftKind.Value(str(drift.kind).upper())`
without the `DRIFT_KIND_` prefix, so it drops the kind to `UNSPECIFIED`.)

If you are retiring a value instead of adding one, follow the `CONFUSION`
pattern (line ~120 of `types.proto`): a comment naming the retired value, a
`reserved <number>;` line, and a `reserved "DRIFT_KIND_<NAME>";` line.

### Step 1.3 — Regenerate the proto stubs

```bash
uv sync --extra proto
make proto
```

`make proto` runs `grpc_tools.protoc` (see the `proto:` target in the
`Makefile`) and regenerates `goldfive/pb/goldfive/v1/types_pb2.py`,
`types_pb2.pyi`, and the gRPC stubs. Verify:

```bash
uv run python -c "from goldfive.pb.goldfive.v1 import types_pb2 as t; print(t.DriftKind.Value('DRIFT_KIND_MY_NEW_DRIFT'))"
```

It must print your number (e.g. `42`), not raise `ValueError`.

**Commit the regenerated stubs.** `goldfive/pb/**` is checked in — CI does
not run `make proto`. A stub that drifts from the `.proto` is a silent bug.

### Step 1.4 — Wire the detector (three homes, pick one)

Where the classifier lives depends on how the drift is detected:

**(a) Structural / tool-facing drift** — add a pure function to
`goldfive/drift/__init__.py` alongside `classify_tool_error`,
`classify_refusal`, `classify_confabulation_risk`, `classify_stop_reason`.
Signature: `def classify_my_new_drift(event: Any) -> DriftEvent | None`. Full
procedure in **Recipe 2**.

**(b) Reasoning-channel / LLM-judge drift** — extend
`goldfive/drift/reasoning_judge.py` or `goldfive/drift/goals.py`. Full
procedure in **Recipe 3**.

**(c) Reporting-tool-originated drift** — the handler emits the `DriftEvent`
inline (no separate classifier). e.g. `report_plan_divergence` →
`steerer.drift.report_plan_divergence(...)`. Full procedure in **Recipe 9**.

In every case the detector constructs a `DriftEvent` (dataclass in
`goldfive/types.py`, line ~1434):

```python
from goldfive.types import DriftEvent, DriftKind, DriftSeverity

DriftEvent(
    kind=DriftKind.MY_NEW_DRIFT,
    severity=DriftSeverity.WARNING,
    detail="human-readable one-liner",
    current_task_id=task_id,          # "" if none
    current_agent_id=agent_id,        # "" if none
    raw=triggering_event,             # for debugging; not serialised
    trigger_input=short_render,       # what the detector saw (<=2048 chars)
    observed_revision_index=rev,      # session.plan.revision_index BEFORE any await
    detector_name="my_module",        # ONLY if the kind is shared across detectors
)
```

- Only `id` (a UUID4) is stamped automatically on the `DriftEvent` dataclass —
  do **not** set it by hand. `condition_id` is **not** a `DriftEvent`
  attribute; it is computed at emit time by
  `state_store.compute_condition_id(...)`
  (= `sha1(f"{kind.value}|{task_id}|{agent_id}|{turn_id}")[:16]`) and stamped
  on the proto `DriftDetected` (field 12) — a **stable lifecycle key**
  (invariant 6). If you need a gate keyed on "this logical drift", compute it
  via `compute_condition_id(...)`, never off `drift.id`.
- `detector_name` is only needed when the same `DriftKind` is minted by more
  than one detector. The tool-loop tracker deliberately emits
  `LOOPING_REASONING` (same kind as the embedding reasoning-loop detector),
  so it stamps `detector_name="tool_loops"` — see the field comment at
  `goldfive/types.py:1503`. A kind with a unique detector leaves it `""`.
- `observed_revision_index` **must** be captured at the top of the detector,
  **before** any `await call_llm(...)`, so the stale-verdict gate in
  `handle_drift` can notice the plan moved on during the LLM round-trip
  (goldfive#245).

### Step 1.5 — Add an intervention-ladder row

File: `goldfive/drift_observer.py`, method `DriftObserver._load_ladder_tables`
(around line 3543), which populates the class attribute `_LADDER`.

> Note: the ladder was moved OFF `DefaultSteerer` and onto `DriftObserver` in
> the bucket-3c refactor. The stale skill still says "wire the steerer
> dispatch" — the actual home is `DriftObserver._LADDER`.

The `_LADDER` value is a 3-tuple: `(info_level, warning_level,
(critical_first, critical_repeat))`. Each level is an `InterventionLevel`
(defined in `goldfive/steerer.py`: `OBSERVE=0`, `ABSORB=1`, `NUDGE=2`,
`CANCEL_REINVOKE=3`, `PAUSE_ESCALATE=4`, `TERMINATE=5`). A `None` in the INFO
or WARNING slot means "fall back to `OBSERVE`".

```python
# goldfive/drift_observer.py — inside DriftObserver._load_ladder_tables, cls._LADDER = { ...
            DriftKind.MY_NEW_DRIFT: (
                _IL.OBSERVE,                          # INFO -> observe only
                _IL.ABSORB,                           # WARNING -> refine plan
                (_IL.CANCEL_REINVOKE, _IL.PAUSE_ESCALATE),  # CRITICAL first / repeat
            ),
```

How the tuple is read (`_ladder_level_for`, line ~3656):

- `INFO` severity → `info_level or OBSERVE`.
- `WARNING` severity → `warning_level or OBSERVE`.
- `CRITICAL` severity → `critical_pair[1] if is_repeat else critical_pair[0]`,
  where `is_repeat = occurrence_count >= steerer.REFINE_FAILURE_THRESHOLD`.

If you **omit** the row entirely, the fallback at the bottom of
`_ladder_level_for` applies: INFO→OBSERVE, WARNING→ABSORB, CRITICAL→
(ABSORB, PAUSE_ESCALATE). That is a fine default for a recoverable
WARNING-severity drift; add an explicit row only when you need a different
shape (e.g. a NUDGE-first loop-prevention kind, or a terminal kind that must
PAUSE_ESCALATE immediately).

**Ladder-design cheatsheet** (mirror existing rows):

| Intent | Row shape | Example kind |
|---|---|---|
| Record-only, never intervene | `(OBSERVE, OBSERVE, (OBSERVE, OBSERVE))` | `REASONING_CLUSTER_TIGHTENING` |
| Recoverable, refine then escalate | `(OBSERVE, ABSORB, (CANCEL_REINVOKE, PAUSE_ESCALATE))` | `TOOL_ERROR`, `OFF_TOPIC` |
| Loop — nudge first, never plan-swap | `(None, ABSORB, (NUDGE, PAUSE_ESCALATE))` | `LOOPING_REASONING` |
| Reality-provoked, always absorb | `(OBSERVE, ABSORB, (ABSORB, ABSORB))` | `JUSTIFIED_DEVIATION` |
| Terminal, human required | `(None, None, (PAUSE_ESCALATE, TERMINATE))` | `HUMAN_INTERVENTION_REQUIRED` |

If your CRITICAL routing needs a follow-up nudge on a successful ABSORB, add
the kind to `_ABSORB_NUDGE_KINDS` (frozenset in `goldfive/steerer.py`, line
~174) — but only for "coordinator-stuck" shapes; read the comment there
first.

### Step 1.6 — Add a corrective template (only if it can NUDGE/CANCEL_REINVOKE)

If the ladder can route your kind to a corrective user message (NUDGE,
CANCEL_REINVOKE, or the ABSORB→nudge handoff), add a `_CORRECTIVE_TEMPLATES`
entry in `goldfive/steerer.py` (dict at line ~196). Keep it **short,
action-focused, jargon-free** — no "drift", "synthetic", "orphan":

```python
# goldfive/steerer.py — inside _CORRECTIVE_TEMPLATES
    DriftKind.MY_NEW_DRIFT: (
        "The prior attempt stalled on {current_task_id}. "
        "Refined plan: proceed with {next_task_title}."
    ),
```

`compose_corrective_user_message` fills `{current_task_id}` and
`{next_task_title}` from the revised plan. A kind with no template falls back
to a generic message; that is fine for OBSERVE/ABSORB-only kinds.

### Step 1.7 — Tests

Add to `tests/test_drift_taxonomy.py` (the canonical home):

- `test_my_new_drift_enum_exists` — `DriftKind.MY_NEW_DRIFT` resolves, value
  is `"my_new_drift"`.
- `test_my_new_drift_proto_mirror` — `types_pb2.DriftKind.Value("DRIFT_KIND_MY_NEW_DRIFT")`
  is non-zero and equals the number you assigned.
- `test_my_new_drift_severity_roundtrips` — assert the by-name bridge
  directly (the path used for real emission):
  `assert types_pb2.DriftKind.Value("DRIFT_KIND_" + DriftKind.MY_NEW_DRIFT.name) != 0`
  and the `DriftSeverity` twin. Do **not** route this through
  `drift_detected_event()`: that factory does **not** round-trip kind/severity
  (it swallows the `ValueError` and leaves them `UNSPECIFIED`). The real
  stamping is `DefaultSteerer._drift_kind_pb_value` /
  `_drift_severity_pb_value`, invoked from `DriftObserver` at
  `drift_observer.py:484`.

Add to `tests/test_intervention_ladder.py`:

- `test_my_new_drift_ladder_levels` — for each severity band, assert
  `observer._ladder_level_for(DriftKind.MY_NEW_DRIFT, sev, count)` returns
  the level you intended (both `count=0` and `count>=REFINE_FAILURE_THRESHOLD`
  for CRITICAL).

Add the detector's own test to the module suite (Recipe 2 → `test_drift_classifiers.py`,
Recipe 3 → `test_reasoning_judge.py` / `test_goal_drift_classifier.py`).

If severity is graduated, add one test per band (INFO/WARNING/CRITICAL) with
representative inputs, mirroring `INTENT_DIVERGENCE` in
`tests/test_drift_reasoning.py`.

### Step 1.8 — Update `docs/design/DRIFT.md`

Add a row under the correct category table (Error / Divergence / Structural /
Discovery / User / Goal / Reasoning / Reflective). The columns are
`| Kind | Trigger | Default severity | Recoverable |`:

```markdown
| `MY_NEW_DRIFT` | One-line trigger naming the event/tool/threshold. | `warning` | yes |
```

If graduated, write "tiered (`info`/`warning`/`critical`)" in the severity
column and add a sub-section below the table with the band boundaries,
mirroring the `INTENT_DIVERGENCE` treatment. See **Recipe 12** for the
design-doc discipline.

### Verification checklist (Recipe 1)

```bash
# proto round-trips
uv run python -c "from goldfive.pb.goldfive.v1 import types_pb2 as t; print(t.DriftKind.Value('DRIFT_KIND_MY_NEW_DRIFT'))"
# targeted tests
uv run pytest -q tests/test_drift_taxonomy.py tests/test_intervention_ladder.py tests/test_drift_classifiers.py
# name-parity grep: every Python member must have a proto twin
uv run python -c "from goldfive.types import DriftKind; from goldfive.pb.goldfive.v1 import types_pb2 as t; [t.DriftKind.Value('DRIFT_KIND_'+k.name) for k in DriftKind]; print('parity OK')"
# full suite + lint
uv run pytest -q
uv run ruff check .
```

### If any step fails

- `ValueError: Enum DriftKind has no value...` → proto suffix ≠ Python name
  (Step 1.2), or you forgot `make proto` (Step 1.3), or you did not commit
  the regenerated `_pb2.py`.
- Kind serialises as `DRIFT_KIND_UNSPECIFIED` on the wire → same name-parity
  bug; run the parity grep above.
- Ladder test returns the wrong level → check the tuple order (INFO, WARNING,
  (critical_first, critical_repeat)) and that `is_repeat` uses
  `REFINE_FAILURE_THRESHOLD`, not a literal.
- `ruff` complains about an unused import in `types.py` → you added a comment
  referencing a symbol you did not import; comments do not need imports.

---

## Recipe 2 — Add a deterministic (structural) detector

**What this gives you.** A pure, LLM-free classifier that turns an observed
structural fact (a tool error shape, a delegation to a mis-capable agent, a
loop of identical tool calls) into a `DriftEvent`. Deterministic detectors
are cheap, run on every observation, and are the backbone of goldfive's
mock-testable pipeline.

**Preconditions.**

- The signal is **structural**, not natural-language classification. If you
  find yourself matching English phrasing, STOP — that is invariant 2
  territory; use Recipe 3 (an LLM judge) instead. Exact-equality / hash
  matching of structured data (tool name, args hash, task status enum,
  tool-surface introspection) is allowed.
- You have completed Recipe 1 Steps 1.1–1.3 (the kind exists in Python +
  proto + stubs).

### Step 2.1 — Decide the detector's home and shape

| Trigger source | Home | Signature |
|---|---|---|
| Adapter event (tool error, stop reason, transfer) | `goldfive/drift/__init__.py` | `def classify_x(event) -> DriftEvent \| None` |
| Delegation-time tool-surface check | new module `goldfive/drift/<name>.py` | see `capability_check.py` |
| Tool-call sequence / loop | `goldfive/drift/tool_loops.py` | stateful tracker |

Reference the smallest analogous detector. `capability_check.py`
(`detect_capability_mismatch`, line ~255) is the cleanest full example of a
standalone structural detector: it introspects the invoked agent's tools, has
an explicit **negative class**, and self-registers.

### Step 2.2 — Write the classifier with an explicit negative class

The single most important design decision is **when to return `None`**. A
detector that fires too eagerly is worse than none — every fire may cancel an
in-flight invocation in active mode. Copy the `detect_capability_mismatch`
discipline: name the negative class in the docstring and return `None` for it
first.

```python
# goldfive/drift/my_detector.py
from __future__ import annotations

from typing import Any

from goldfive.types import DriftEvent, DriftKind, DriftSeverity


def detect_my_signal(*, observed: Any, task: Any) -> DriftEvent | None:
    """Return MY_NEW_DRIFT iff <structural condition>, else None.

    Negative class (return None): <enumerate exactly what does NOT fire>.
    No LLM call, no regex over NL, no keyword matching.
    """
    if task is None:
        return None
    # 1. Capture the plan revision BEFORE any work, for the stale-verdict gate.
    observed_rev = int(getattr(getattr(task, "_plan", None), "revision_index", 0) or 0)
    # 2. Structural test on hashable/enum data only.
    if not _structural_condition(observed, task):
        return None                      # <- the negative class
    # 3. Construct the event.
    return DriftEvent(
        kind=DriftKind.MY_NEW_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail=f"agent {getattr(observed, 'name', '')} cannot do {getattr(task, 'id', '')}",
        current_task_id=str(getattr(task, "id", "") or ""),
        trigger_input="",                # structural detectors often have none
        observed_revision_index=observed_rev,
        detector_name="my_detector",     # if the kind is shared; else omit
    )
```

Rules:

- **Pure function.** No sink emits, no session mutation, no `await`. The
  steerer owns emission and dispatch.
- **Stable keys.** If the detector maintains state across calls (loop
  tracker), key its buckets on structured identity — `(name, args_hash)` for
  tool loops, `(task_id, agent_path)` — never on an LLM-minted id (invariant
  6). The tool-loop tracker's name-axis cap (#484) is the canonical example:
  a name-only match caps at INFO unless corroborated by
  `>=_NAME_AXIS_CORROBORATION_MIN_EXACT` (2) **identical** `(name, args_hash)`
  repeats.
- **Observation-only neutrality.** A detector must behave **identically**
  whether `observation_only` is True or False — it only *observes*. It must
  NOT read `is_active_steering()` or the kill-switch. Gating happens later, in
  the steerer's dispatch (Recipe 4). This keeps detection independent of
  injection (invariant 5).

### Step 2.3 — Register the detector

Every detector self-registers at import time via
`goldfive/drift/registry.py::register`. Add this at the bottom of your module:

```python
# goldfive/drift/my_detector.py — module bottom
from goldfive.drift.registry import DetectorConfig as _DetectorConfig  # noqa: E402
from goldfive.drift.registry import register as _register  # noqa: E402

_register(
    DriftKind.MY_NEW_DRIFT,
    detect_my_signal,
    _DetectorConfig(uses_llm=False),   # structural => all-default config
    is_async=False,
)
```

`DetectorConfig` (frozen dataclass in `registry.py`, line ~114) fields:
`uses_llm`, `max_input_chars`, `max_output_tokens`, `disable_thinking`,
`timeout_seconds`. A pure-structural detector sets `uses_llm=False` and
leaves the rest default (`0`/`False`) — there is no observability payload to
truncate. `register` is idempotent-overwriting: re-registration replaces and
logs a debug line.

### Step 2.4 — Make the module auto-load

The registry's `_ensure_registered()` (line ~354) imports the detector
modules so their `register(...)` side-effects run. If your detector is a
**new module**, add it to that import list:

```python
# goldfive/drift/registry.py — inside _ensure_registered()
    from goldfive.drift import (  # noqa: F401 — side-effect import for registration
        capability_check,
        goals,
        my_detector,          # <- add here
        reasoning_judge,
        tool_loops,
    )
```

If you added the classifier to an already-imported module
(`drift/__init__.py`, `tool_loops.py`), no change is needed.

### Step 2.5 — Call the detector from the observation path

A registered detector is discoverable but not yet **called**. Wire the call
site where the triggering fact is observed:

- **Adapter-event classifiers** (`classify_tool_error` etc.) are called from
  `DriftObserver.detect_drift` in `goldfive/drift_observer.py` (line ~1243).
  Add your `classify_x(event)` to the chain there.
- **Delegation-time detectors** like `detect_capability_mismatch` are called
  from the ADK plugin's `delegation_observed` path in
  `goldfive/adapters/_adk_plugin.py` (grep for `detect_capability_mismatch`).
- **Loop trackers** are fed from the tool-invocation hook.

The call site does `drift = detect_my_signal(...)` then, if non-`None`,
`await self.handle_drift(drift, session)`. **Verify the wiring** — a
registered-but-uncalled detector is dead code that passes its unit test
(project scar tissue: `feedback_integration_not_unit`). Grep for the call
site after adding:

```bash
grep -rn "detect_my_signal\|classify_my_signal" goldfive/ | grep -v "def detect_my_signal\|_register\|test"
```

### Step 2.6 — Add manifest / config knobs (optional)

If the detector has tunable thresholds, they belong in a `SteeringConfig`- or
`ToolLoopConfig`-style dataclass with an env reader — **Recipe 5**. Do NOT
read `os.environ` directly inside the detector. Module-level constants
prefixed with the kind name (e.g. `_NAME_AXIS_CORROBORATION_MIN_EXACT`) are
fine for fixed structural thresholds that are not operator-tunable.

### Step 2.7 — Tests

Add to `tests/test_drift_classifiers.py` (or a dedicated file like
`tests/test_capability_mismatch.py` if the detector is large):

- `test_my_signal_fires_on_positive` — feed a positive input, assert the
  returned `DriftEvent.kind` and `.severity`.
- `test_my_signal_negative_class_returns_none` — feed each member of the
  negative class, assert `None`. **This is the load-bearing test.**
- `test_my_signal_observed_revision_captured` — assert
  `event.observed_revision_index` equals the plan revision at call time.
- Add to `tests/test_drift_registry.py`:
  `DriftKind.MY_NEW_DRIFT in registry.list_registered()` after
  `registry._ensure_registered()`.

### Verification checklist (Recipe 2)

```bash
uv run pytest -q tests/test_drift_classifiers.py tests/test_drift_registry.py
# registration actually ran:
uv run python -c "from goldfive.drift import registry; registry._ensure_registered(); from goldfive.types import DriftKind; assert DriftKind.MY_NEW_DRIFT in registry.list_registered(); print('registered')"
# call site exists (integration, not just unit):
grep -rn "detect_my_signal" goldfive/ | grep -v test
uv run pytest -q
uv run ruff check .
```

### If any step fails

- Detector never fires in an e2e run but unit test passes → Step 2.5, no call
  site. Grep confirms.
- `KeyError` / kind not in `list_registered()` → Step 2.4, module not in
  `_ensure_registered`, or the module was never imported.
- Detector fires in observation-only runs and cancels things → the detector
  is reading the kill-switch (it must not) OR the ladder row is wrong; the
  detector itself is always neutral, the *steerer* gates (Recipe 4).
- Firing on generic inputs → your "structural" test is actually matching NL;
  reread invariant 2, move to Recipe 3.

---

## Recipe 3 — Add or extend an LLM judge

**What this gives you.** A natural-language classification (is this reasoning
on-task? is the trajectory advancing the goal?) done by an LLM, not a regex.
This is the sanctioned path for any NL signal (invariant 2).

**Preconditions.**

- The signal genuinely requires reading free-form agent text. Structural
  facts go to Recipe 2.
- You accept the **no-harness caveat**: there is currently **no automated
  judge regression harness** on `main` (it is DEFERRED, blocked on building
  that harness). You cannot prove a prompt change is a net improvement with a
  test. You must capture before/after `ReasoningJudgeInvoked` events and
  eyeball them. Judge windowing/cadence expansion and judge-facade dispatch
  authority are likewise deferred — do not add them.

### Step 3.1 — Choose: new judge vs. extend the reasoning judge

- **Extend the existing reasoning judge** (most common) — change the prompt
  or parsing in `goldfive/drift/reasoning_judge.py`. This is the
  three-state classifier (`on_task` / `justified_deviation` /
  `erroneous_deviation`) that already emits `OFF_TOPIC` and
  `JUSTIFIED_DEVIATION`.
- **New periodic/trajectory judge** — follow `goldfive/drift/goals.py`
  (`classify_goal_drift`), which runs every N invocations against
  `session.goals`.
- **New pluggable judge** — implement the `Judge` base in
  `goldfive/judges/base.py` and register it (see `goldfive/judges/builtins.py`
  and `tests/test_pluggable_judges.py`).

### Step 3.2 — Extending the reasoning-judge prompt

The prompt is two module constants in `reasoning_judge.py`, both overridable
by operators (so **never inline prompt text at a call site** — thread through
these):

- `REASONING_DRIFT_SYSTEM_PROMPT` (line ~134)
- `REASONING_DRIFT_USER_PROMPT_TEMPLATE` (line ~154)

If you add a judged field, you must change **four things in lockstep**:

1. The prompt template — add the field to the "Decide THREE things" block and
   the JSON shape spec.
2. The parser — populate the new field on `ReasoningJudgeVerdict` (dataclass,
   line ~603). Quiet-fail to the empty/zero default on malformed responses
   (`parse_json_response` in `registry.py` returns `None` on any failure; the
   judge treats that as "no drift" — goldfive#143/#226/#244).
3. The proto — add a field to `ReasoningJudgeInvoked` in
   `proto/goldfive/v1/events.proto` (Recipe 6). The last four were
   `focused_task_id=12`, `focus_confidence=13`, `stated_intent=14`,
   `provenance=15` (#480); your new field is `16`.
4. The emission — pass it through `_emit_judge_invoked` (line ~1314) and set
   it on the `ReasoningJudgeInvoked` payload.

**Clamp and default at parse time.** `focus_confidence` is clamped to
`[0.0, 1.0]`; a missing key becomes `""`/`0.0`. Never let an unparsed judge
field reach a gate as a truthy garbage value.

### Step 3.3 — A new detector-style judge (goals.py pattern)

```python
# goldfive/drift/my_judge.py
async def classify_my_trajectory(
    *,
    activity_summary: str,
    goals,
    call_llm,                    # injected; None => judge disabled
    session_id: str = "",
    run_id: str = "",
    sequence_fn=None,
    sink=None,
) -> DriftEvent | None:
    if call_llm is None:
        return None              # mock-only runs never see this drift
    observed_rev = ...           # capture BEFORE the await
    from goldfive.drift.registry import format_goals_block, parse_json_response
    prompt = _USER_TEMPLATE.format(goals=format_goals_block(goals), activity=activity_summary)
    raw = await call_llm(prompt, system=_SYSTEM_PROMPT, max_output_tokens=..., disable_thinking=True)
    parsed = parse_json_response(raw)      # None on any malformed response
    # ... emit the observability event on EVERY call, drift or not ...
    if parsed is None or not parsed.get("off_goal"):
        return None
    return DriftEvent(kind=DriftKind.MY_NEW_DRIFT, severity=..., observed_revision_index=observed_rev, ...)
```

Then register with `DetectorConfig(uses_llm=True, max_input_chars=2048,
max_output_tokens=16384, disable_thinking=True)` (Recipe 2 Step 2.3–2.4).
Reuse the shared helpers in `registry.py`: `parse_json_response`,
`format_goals_block`, `truncate_for_observability` — do not re-roll them
(#490 deleted the duplicate copies).

### Step 3.4 — Emit the observability event on EVERY invocation

The judge must emit `ReasoningJudgeInvoked` (or your judge's equivalent) on
**every** call — on-task, off-task, and plumbing-failure — so an operator can
diagnose from the wire alone (see `reasoning_judge.py` line ~1266). The
`raw_response` and `reasoning_input` are truncated (2048 / 4096 chars). This
event is the ONLY way to evaluate a prompt change without a harness (the
caveat in the preconditions).

### Step 3.5 — Manual evaluation protocol (the no-harness workaround)

1. Capture a baseline: run the e2e (or a fixture) with the current prompt and
   dump `ReasoningJudgeInvoked` events (an `InMemorySink` or a JSONL sink).
2. Apply your prompt/parser change.
3. Re-run the identical input, dump again.
4. Diff `classification` / `severity` / `reason` per invocation. Confirm the
   change moves the verdicts you intended and does not flip unrelated ones.
5. Record the eyeball result in the PR description — there is no green test to
   point at.

### Step 3.6 — Tests (what you CAN test)

You cannot test judge *quality*, but you must test *plumbing*:

- `tests/test_reasoning_judge.py` — feed a **mock `call_llm`** returning a
  canned JSON string; assert the parsed `ReasoningJudgeVerdict` fields and the
  returned `DriftEvent`. Add cases for: valid JSON, JSON in a code fence,
  malformed JSON (→ quiet-fail, `drift=None`), missing key (→ default).
- `tests/test_judge_thinking_disabled.py` — assert `disable_thinking=True`
  reached `call_llm`.
- `tests/test_judge_token_caps.py` — assert the `max_output_tokens` budget.
- `tests/test_judge_scheduling_guards.py` — if your judge runs under the
  per-steerer semaphore (default 3, #483), assert it respects the guard.
- Proto round-trip for the new `ReasoningJudgeInvoked` field
  (`tests/test_events.py`).

### Verification checklist (Recipe 3)

```bash
uv run pytest -q tests/test_reasoning_judge.py tests/test_goal_drift_classifier.py \
  tests/test_judge_thinking_disabled.py tests/test_judge_token_caps.py \
  tests/test_judge_scheduling_guards.py tests/test_events.py
# a malformed judge response must NOT raise and must NOT manufacture a drift:
uv run pytest -q tests/test_judge_empty_response_no_retry.py
uv run pytest -q
uv run ruff check .
```

### If any step fails

- Judge test hangs → your mock `call_llm` is not async or never returns; judge
  calls are always `await`ed.
- A malformed response crashes the run → you bypassed `parse_json_response`;
  route through it, it returns `None` on any failure.
- New proto field always empty on the wire → Step 3.2 item 4, you added the
  field but never set it in `_emit_judge_invoked`.
- You "prove" the prompt is better with a new assertion → there is no harness;
  delete the brittle assertion, do the manual eval, and record it in the PR.

---

## Recipe 4 — Add an intervention surface (THE critical one)

**What this gives you.** A new way for the steerer to *act* on a drift (a new
ladder dispatch, a new injection point, a new control message). This is the
most invariant-dense recipe: it touches the strict-passive kill-switch
(invariant 5), decision telemetry, and the ladder.

**Preconditions.**

- You have a drift that reaches the ladder (Recipe 1) and a level that should
  dispatch to your new surface (`OBSERVE`/`ABSORB`/`NUDGE`/`CANCEL_REINVOKE`/
  `PAUSE_ESCALATE`/`TERMINATE`, or a new dispatch inside one of those).
- You understand that **anything that mutates plan state, injects a message,
  or cancels an invocation is a write-path and must be gated**.

### Step 4.1 — Locate the dispatch fan-out

File: `goldfive/drift_observer.py`, method `handle_drift` →
`_handle_drift_dispatch` (line ~3855). The level fan-out is around line 3997:

```python
        if level is InterventionLevel.OBSERVE:
            return
        if level is InterventionLevel.NUDGE:
            await self._dispatch_nudge(drift, session)
            return
        if level is InterventionLevel.PAUSE_ESCALATE:
            await self._dispatch_pause_escalate(drift, session)
            return
        if level is InterventionLevel.TERMINATE:
            await self._dispatch_pause_escalate(drift, session, terminate=True)
            return
        # ABSORB / CANCEL_REINVOKE fall through to the shared refine body.
```

Add your dispatch as a new `_dispatch_*` method and route to it here (or
inside an existing dispatch).

### Step 4.2 — Gate EVERY write-path on `is_active_steering()`

This is the load-bearing rule. Copy `_dispatch_nudge` (line ~4416)
**verbatim** as your template — it is the canonical gated write-path:

```python
    async def _dispatch_my_surface(self, drift: DriftEvent, session: Session) -> None:
        # 1. Compute WHAT WOULD happen unconditionally (so telemetry is truthful).
        payload = self._compose_my_payload(drift, session)
        # 2. The ONLY sanctioned kill-switch read.
        if not self._steerer.is_active_steering():
            # 3a. Log the would-be action at INFO (operators see the counterfactual).
            log.info(
                "DefaultSteerer._dispatch_my_surface: observation_only=True — "
                "SKIPPING. would_have_done kind=%s task=%s payload=%r",
                drift.kind.value,
                drift.current_task_id or "-",
                payload[:200],
            )
            # 3b. Stamp decision telemetry so zicato / harmonograf can count it.
            await self._emit_policy_applied(
                session=session,
                policy_name="observation_only_gate",
                outcome="suppressed",
                reason="observation_only=True",
                detail=(
                    f"intervention=my_surface "
                    f"kind={drift.kind.value} "
                    f"task_id={drift.current_task_id or ''}"
                ),
            )
            return
        # 4. Active mode ONLY: the actual write.
        session.pending_nudges.append(payload)  # or dispatch a ControlMessage, etc.
```

**Non-negotiables:**

- The gate is `self._steerer.is_active_steering()` (this method lives on
  `DefaultSteerer` in `goldfive/steerer.py`, line ~1339). If your surface is
  in a consumer holding a *maybe-steerer* (an executor, the plugin, a
  reporting ack), use the module helper `steering_is_active(steerer)` instead
  — it returns `False` for None/missing/raising. **Never** read
  `SteeringConfig.observation_only` or `_observation_only`.
- The three carve-outs that land **even under `observation_only`** are
  **bootstrap** (first plan install), **user-authored** (`authored_by ==
  "user"`), and **discovery** (`NEW_WORK_DISCOVERED`). If your surface is one
  of those, it is not a corrective intervention and the gate does not apply —
  but that is a rare, deliberate choice; document it and mirror the existing
  carve-out tests (`tests/test_observation_only_*_carveout.py`).
- **Compute the payload before the gate** so the suppressed-path telemetry
  logs the *real* counterfactual, not a stub. This is what makes the
  optimizer (zicato) able to measure "what active mode would have done".

### Step 4.3 — Emit `PolicyApplied` on every suppression AND every non-trivial gate drop

`_emit_policy_applied` (line ~830) is the generic decision-telemetry
envelope. Its signature:

```python
    async def _emit_policy_applied(
        self, *, session, policy_name, outcome, reason="", detail=""
    ) -> None:
```

Conventions (match existing call sites so downstream label parsers stay
stable — #480 was a label-fix sweep):

| `policy_name` | `outcome` | When |
|---|---|---|
| `observation_only_gate` | `suppressed` | strict-passive skip of a write-path |
| `refine_outcome_succeeded_skip` | `skipped` | same-turn replay of an addressed condition |
| `refine_failure_threshold` | `suppressed` | `fail_count >= REFINE_FAILURE_THRESHOLD` |

`policy_name` must be a **stable symbolic** string (invariant 6 for
telemetry): downstream consumers group on it. Do not interpolate a task id or
count into `policy_name`; put those in `detail`.

### Step 4.4 — Add a deadline if the surface can wait for a human

If your surface *blocks* (a pause waiting for an operator control message), it
**must** be bounded or it can hang an unattended deployment forever
(goldfive#478/#482). The pattern:

- Add a config knob `*_deadline_s: float | None` (Recipe 5). `None` = block
  forever (only acceptable for operator-issued PAUSE, never for a
  goldfive-minted escalation).
- Level 5 `TERMINATE` **always** terminates: with no configured deadline it
  falls back to `DEFAULT_TERMINATE_PAUSE_DEADLINE_S` (in
  `goldfive/drift_observer.py`). See `_dispatch_pause_escalate(..., terminate=True)`.
- On expiry: drain background steerer/judge tasks, CANCEL every non-terminal
  task, emit `RunAborted` carrying the escalation lineage (originating drift
  kind + ladder level).

### Step 4.5 — Emit the ladder transition

Every dispatch is preceded by an `_emit_ladder_transition` (line ~3990) so
the `LadderTransitionDecided` event records `from_level` → `to_level` and a
reason ("first occurrence" / "repeat (count=N)"). If you add a dispatch
*inside* an existing level, you inherit its transition; if you add a new
level path, emit the transition before the dispatch.

### Step 4.6 — Tests (BOTH modes, always)

An intervention surface has two behaviours and you must test both:

- `tests/test_observation_only_my_surface.py` (new) — with
  `SteeringConfig(observation_only=True)` (the shipped default), assert the
  surface **does not** write (no `pending_nudges` append, no ControlMessage
  dispatched) and **does** emit a `PolicyApplied` with
  `policy_name="observation_only_gate"`, `outcome="suppressed"`.
- Active-mode test — with `SteeringConfig(observation_only=False)`, assert the
  surface performs the write. Note that since #488 the suite runs the shipped
  `observation_only=True` default and ~90 tests **explicitly opt into** active
  mode; your active-mode test must set `observation_only=False` explicitly.
- `tests/test_intervention_ladder.py` — assert the ladder routes your drift to
  the level that dispatches your surface.
- If it has a deadline, add an expiry test asserting `RunAborted` with the
  lineage (mirror `tests/test_observation_only_pause_escalate_carveout.py`).

### Verification checklist (Recipe 4)

```bash
uv run pytest -q tests/test_observation_only_strict_passive.py \
  tests/test_observation_only_nudge_gate.py tests/test_intervention_ladder.py \
  tests/test_promote_drift_to_steer.py
# grep: no NEW direct kill-switch reads snuck in
grep -rn "_observation_only\|observation_only" goldfive/ | grep -v "config.py\|steerer.py\|test\|#\|\"\"\"\|is_active_steering\|steering_is_active"
uv run pytest -q
uv run ruff check .
```

The grep should return **nothing** outside `config.py` (definition) and
`steerer.py` (the two sanctioned predicates). Any other hit is an invariant-5
violation.

### If any step fails

- Test shows a write under `observation_only=True` → your gate is missing,
  inverted, or reading the wrong thing. It must be
  `if not self._steerer.is_active_steering(): ...; return`.
- `PolicyApplied` missing in the suppressed path → Step 4.3; the optimizer
  can't see your counterfactual.
- The grep in the checklist returns a hit in your new file → you read the
  flag directly. Replace with `is_active_steering()` /
  `steering_is_active(steerer)`.
- A pause hangs a test forever → Step 4.4, you added a blocking wait with no
  deadline.

---

## Recipe 5 — Add a config knob

**What this gives you.** A new operator-tunable setting with a dataclass
field, an env override, correct precedence, a docstring, and a runtime
consumer.

**Preconditions.** You know the knob's type, default, and which config
dataclass it belongs to (`SteeringConfig`, `ToolLoopConfig`,
`ReasoningDriftConfig`, `GoalDriftConfig`, `EmbeddingConfig`, `JudgeConfig` —
all in `goldfive/config.py`).

### Step 5.1 — Add the dataclass field with a docstring

File: `goldfive/config.py`. Add the field to the right dataclass with a `#:`
Sphinx comment covering: what it does, the default's rationale, the env var
name, and any interaction with other knobs. Mirror `stall_timeout_s`
(line ~776):

```python
# goldfive/config.py — inside the relevant @dataclass
    #: One-line purpose. Default X because <rationale>. When <edge case>,
    #: <behaviour>. Non-positive disables the feature. Routed through the
    #: normal <path>, so under ``observation_only`` (the production
    #: default) the effect is telemetry-only.
    #: Env: ``GOLDFIVE_<GROUP>_MY_KNOB``.
    my_knob: float = 600.0
```

Choose the type carefully:

- `bool` for a feature flag (default it **OFF** if it can change behaviour —
  every agency-preservation flag defaults OFF).
- `int` for a positive count/window (the int reader rejects non-positive).
- `float` for a threshold (the float reader allows `0.0`).
- `float | None` for "feature disabled when None" (a deadline).

### Step 5.2 — Read it in `from_env` using the shared helpers

Each config dataclass has a `from_env` classmethod. Add your read there using
the existing helpers (do **not** call `os.environ` directly — the helpers log
a warning on a typo instead of silently flipping a policy):

| Helper | For | Bad-value behaviour |
|---|---|---|
| `_read_bool_env(name, default)` | flags | unknown literal → WARNING + default |
| `_read_int_env(name, default)` | positive ints | non-int / `<=0` → debug + default |
| `_read_float_env(name, default)` | thresholds (allows 0) | non-float → debug + default |
| `_read_optional_float_env(name, default)` | `float\|None` deadlines | `<=0` → `None` |
| `_read_str_env(name, default)` | strings | empty passes through |
| `_read_steer_threshold_env(name, default)` | `off/warning/critical` | unknown → WARNING + default |

```python
# goldfive/config.py — inside SteeringConfig.from_env (or the right group)
        return cls(
            ...
            my_knob=_read_float_env("GOLDFIVE_STEER_MY_KNOB", defaults.my_knob),
        )
```

And document the env var in the `from_env` docstring's "Env surface" list.

### Step 5.3 — Understand and preserve precedence

The precedence is **explicit constructor arg > `SteeringConfig` field >
built-in default**, and **env vars are only read in `from_env`** — never at a
consumer. See `DefaultSteerer.__init__` (`goldfive/steerer.py` line ~443): a
direct ctor kwarg (e.g. `goldfive_steer_suppression_window_turns`) wins over
`steering_config.suppression_window_turns`, which wins over the built-in.
Follow that ladder for any knob that also has a ctor kwarg; most knobs do NOT
need a ctor kwarg and are read straight off the `SteeringConfig`.

### Step 5.4 — Thread it to the runtime consumer

A knob nobody reads is dead config. Stash it on the steerer in `__init__` and
read it where it applies. Mirror `stall_watchdog_enabled` /`stall_timeout_s`
(`goldfive/steerer.py` line ~710):

```python
# goldfive/steerer.py — DefaultSteerer.__init__
        if steering_config is not None:
            self._my_knob: float = float(steering_config.my_knob)
        else:
            self._my_knob = SteeringConfig().my_knob   # bare-ctor default
```

The consumer (the plugin, the executor) reads it via `getattr(steerer,
"_my_knob", <default>)` so a stub steerer without the attribute degrades
safely — see how the ADK plugin reads `_stall_timeout_s`
(`_adk_plugin.py` line ~5335): `float(getattr(steerer, "_stall_timeout_s",
0.0) or 0.0)`.

### Step 5.5 — Liveness / manifest (only if it feeds zicato)

If the knob is part of the optimization surface, add it to
`optimization/manifest.toml` so the offline optimizer (zicato) can sweep it,
and add an AST-based manifest-liveness test that asserts the manifest key maps
to a real dataclass field (mirror the `#487` AST manifest-liveness test). Most
knobs do not need this.

### Step 5.6 — Tests

Add to `tests/test_config.py` (or the group's test file):

- `test_my_knob_default` — `SteeringConfig().my_knob == 600.0`.
- `test_my_knob_from_env` — set `GOLDFIVE_STEER_MY_KNOB=42`, assert
  `SteeringConfig.from_env().my_knob == 42.0`.
- `test_my_knob_bad_env_falls_back` — set it to `"garbage"`, assert the
  default (and, for the WARNING helpers, that a warning was logged).
- `test_my_knob_consumer` — construct a steerer with the knob set, assert the
  consumer honours it (behavioural).

### Verification checklist (Recipe 5)

```bash
uv run pytest -q tests/test_config.py
# no direct os.environ reads outside config.py's helpers:
grep -rn "os.environ" goldfive/ | grep -v "config.py\|test"
uv run pytest -q
uv run ruff check .
```

### If any step fails

- Env override ignored → `from_env` doesn't read your var, or the consumer
  reads the dataclass default instead of the constructed instance.
- Typo in env silently uses default with no warning → you used
  `_read_float_env` where the operator expected a warning; the float/int
  readers only debug-log. Use a validating reader (bool / steer-threshold)
  for policy-critical knobs.
- Direct `os.environ` grep hits your file → move the read into `from_env`.

---

## Recipe 6 — Add a proto field or payload

**What this gives you.** A new field on an existing event message (or a whole
new message), with a fresh field number, regenerated stubs, an emission site,
and a round-trip test.

**Preconditions.** `uv sync --extra proto`; you know which `.proto` file and
message (`events.proto` for events, `types.proto` for enums/dataclassed
types, `control.proto` for control messages, `service.proto` for RPCs).

### Step 6.1 — Add the field with a FRESH number

File: `proto/goldfive/v1/events.proto` (or the relevant one). **Append** the
field at the next free tag inside the message. Never reuse a `reserved` tag.
Example — the last four `ReasoningJudgeInvoked` fields are 12–15, so the next
is 16:

```proto
// proto/goldfive/v1/events.proto — inside message ReasoningJudgeInvoked
  // What it carries; when it is populated; the zero-value meaning for
  // old readers. See goldfive#NNN.
  string my_field = 16;
```

Rules:

- **Additive only.** Old wire bytes must deserialise: a new field defaults to
  `""`/`0`/`false` for readers that predate it, and old events deserialise to
  the default for new readers. This is proto3's guarantee — do not break it by
  renumbering.
- Comment the **zero-value semantics** explicitly ("empty on legacy / not-yet-
  routed emit paths") — every existing field does.
- To retire a field, `reserved <number>;` + `reserved "<name>";` and a comment
  (see `DriftDetected` field 11 `synthetic`, retired in #271).

### Step 6.2 — Regenerate stubs

```bash
uv sync --extra proto
make proto
```

Verify the field exists:

```bash
uv run python -c "from goldfive.pb.goldfive.v1 import events_pb2 as e; m=e.Event(); print('my_field' in [f.name for f in m.reasoning_judge_invoked.DESCRIPTOR.fields])"
```

Commit the regenerated `goldfive/pb/**`.

### Step 6.3 — Set the field at the emission site

Events are built by factory functions in `goldfive/events.py`. Find the
factory (e.g. `policy_applied_event` line ~1135, `drift_detected_event` line
~1197) and set the new field. If the factory does not exist yet (new message),
write one mirroring `policy_applied_event`:

```python
# goldfive/events.py
def my_event(
    run_id: str,
    sequence: int,
    *,
    my_field: str = "",
    session_id: str = "",
    event_id: str = "",
) -> Any:
    evt = new_event(run_id, sequence, session_id=session_id, event_id=event_id)
    payload = evt.my_message
    payload.my_field = str(my_field or "")
    return payload_or_evt   # follow the surrounding factories' return convention
```

- Always coerce to the proto type (`str(x or "")`, `max(0, int(x))`) — a
  `None` or wrong-typed value raises at `payload.my_field = ...`.
- Route emission through `session.next_sequence_and_event_id()` for the seq +
  event id (see `_emit_policy_applied`), and through `steerer._emit(evt)` /
  the sink bus. `session_id` at tag 5 must be set (sinks route on it —
  goldfive#155).

### Step 6.4 — Round-trip test + camelCase JSONL check

The JSONL persistence sink serialises with
`MessageToJson(event, sort_keys=True)` (see `goldfive/sinks/persistence.py`
line ~110), which emits **camelCase** field names by default
(`my_field` → `myField`). Assert both the proto round-trip and the JSONL key:

- `tests/test_events.py` — build the event, `SerializeToString()` →
  `ParseFromString()`, assert `my_field` survives.
- JSONL check:

```python
from google.protobuf.json_format import MessageToJson
import json
line = MessageToJson(evt, sort_keys=True, indent=None)
assert "myField" in json.loads(line)["myMessage"]   # camelCase on the wire
```

- `tests/test_event_session_id.py` / `test_event_sequence.py` if the field
  interacts with routing/ordering.

### Verification checklist (Recipe 6)

```bash
make proto && git status --porcelain goldfive/pb   # stubs regenerated + staged
uv run pytest -q tests/test_events.py tests/test_control_proto.py
uv run python -c "from goldfive.pb.goldfive.v1 import events_pb2  # import must not raise"
uv run pytest -q
uv run ruff check .
```

### If any step fails

- `ValueError: Field number N is already used` → you reused a tag; pick the
  next free one.
- Old JSONL files fail to replay → you changed a field type or number
  (non-additive). Revert to additive.
- Field is `snake_case` in the JSONL test → you asserted the proto name;
  `MessageToJson` emits camelCase. Assert `myField`.
- `AttributeError` on `payload.my_field` → stubs not regenerated (`make
  proto`) or not committed.

---

## Recipe 7 — Add an event sink

**What this gives you.** A new destination for the event stream (a new
database, a metrics push, a dashboard feed) conforming to the `EventSink`
protocol.

**Preconditions.** You know your backend's client and whether it needs the
`proto` extra.

### Step 7.1 — Implement the two-method protocol

Contract: `goldfive/protocols.py::EventSink` (`@runtime_checkable`):

```python
@runtime_checkable
class EventSink(Protocol):
    async def emit(self, event_pb: Any) -> None: ...
    async def close(self) -> None: ...
```

Skeleton (new file `goldfive/sinks/my_sink.py`):

```python
from __future__ import annotations
from typing import Any


class MySink:
    def __init__(self, client) -> None:
        self._client = client
        self._buffer: list[Any] = []

    async def emit(self, event: Any) -> None:
        # DESCRIPTOR duck-typing: emit may receive a proto Event OR a dict
        # envelope from the Runner. Skip dicts cleanly (or translate).
        if not hasattr(event, "DESCRIPTOR"):
            return
        self._buffer.append(event)
        if len(self._buffer) >= 64:
            await self._flush()

    async def close(self) -> None:
        if self._buffer:
            await self._flush()
        await self._client.shutdown()

    async def _flush(self) -> None:
        batch, self._buffer = self._buffer, []
        await self._client.send_many(batch)
```

### Step 7.2 — Honour the sink design constraints

- **Duck-type on `hasattr(event, "DESCRIPTOR")`.** Current `main` emits proto
  `Event` messages, but the Runner also emits a few **dict** lifecycle
  envelopes. A proto-only sink skips dicts; do not `AttributeError` on them.
- **Error isolation.** `emit` runs under
  `asyncio.gather(..., return_exceptions=True)` in `goldfive.events.emit`; a
  raise is captured and re-raised after all sinks saw the event, so one faulty
  sink does not starve the others — but it still propagates. Catch-and-log if
  you want best-effort. Since #479, sink exceptions never abort the *run*.
- **Flush in `close`.** `runner.close()` calls every `sink.close()`; buffered
  data is lost if you don't flush.
- **Route on `event.session_id`** (tag 5) when multiplexing per-session
  buffers (goldfive#155) — never on client-global state. Under the adk-web
  pin it equals the outer session id, stable across overlay restarts and
  nested `AgentTool` sub-Runners.
- **Never block the loop.** No synchronous I/O in `emit`; use an async client
  or `asyncio.to_thread`.

### Step 7.3 — Wire the import guard (if it needs an extra)

If your sink needs the `proto` extra (or any optional dep), add a guarded
import to `goldfive/sinks/__init__.py` so the top-level package stays
importable without the extra:

```python
# goldfive/sinks/__init__.py
try:
    from goldfive.sinks.my_sink import MySink
except ImportError:  # pragma: no cover — extra not installed
    MySink = None  # type: ignore[assignment]
```

Callers then assert `MySink is not None` before use (the None-guard is a
documented gotcha — `Runner(sinks=[None])` is silently accepted).

### Step 7.4 — Update the choosing-a-sink doc

Add a row to `docs/guides/choosing-a-sink.md` and the parity table in the
`sinks` skill (`.agents/sinks.md`) and the shipped-implementations table:
`| MySink | goldfive.sinks.my_sink | <extra> | <use> |`.

### Step 7.5 — Tests

- `tests/test_sinks_*` (or a new `tests/test_my_sink.py`) —
  `isinstance(MySink(fake_client), EventSink)`; `emit` buffers a proto event;
  `emit` skips a dict without raising; `close` flushes.
- If it needs an extra, a test asserting the None-guard when the extra is
  absent (or `pytest.importorskip`).

### Verification checklist (Recipe 7)

```bash
uv run pytest -q tests/test_grpc_sink.py   # nearest shipped-sink reference
uv run python -c "from goldfive.protocols import EventSink; from goldfive.sinks.my_sink import MySink; assert isinstance(MySink(object()), EventSink)"
uv run pytest -q
uv run ruff check .
```

### If any step fails

- `AttributeError` on a Runner lifecycle envelope → you did not duck-type on
  `DESCRIPTOR`.
- Data missing after a run → you did not flush in `close`, or the caller
  forgot `await runner.close()`.
- `isinstance(sink, EventSink)` is False → you are missing `emit` or `close`,
  or one is not `async`.
- Import of `goldfive` fails when the extra is absent → you added an unguarded
  top-level import; wrap it in the `try/except ImportError` guard.

---

## Recipe 8 — Add an ADK-tree observation point

**What this gives you.** A new fact captured from the running ADK tree (a new
callback hook, or new work inside an existing callback) fed into the drift
pipeline. This is the "adaptive over predictive" surface (invariant 4).

**Preconditions.** You know the ADK callback that sees your fact
(`before_agent_callback`, `before_model_callback`, `before_tool_callback`,
`after_tool_callback`, `after_model_callback`, `after_agent_callback`,
`on_event_callback`). All live on `_GoldfiveADKPlugin(BasePlugin)` inside
`make_adk_plugin` in `goldfive/adapters/_adk_plugin.py`.

### Step 8.1 — Find the callback and read the tree defensively

Every attribute read off an ADK object goes through `_safe_attr` (line ~258):

```python
def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return value if value is not None else default
```

**Use `_safe_attr` for every ADK-object attribute access.** ADK internals
change shape across versions and across tree positions (a coordinator vs a
leaf vs an `AgentTool` sub-Runner). A bare `getattr` or attribute access will
`AttributeError` on some tree shape and violate invariant 3 (any tree must
work). Chain it for nested reads:

```python
        inv_ctx = _safe_attr(callback_context, "_invocation_context", None) or _safe_attr(
            callback_context, "invocation_context", None
        )
        inv_id = str(_safe_attr(inv_ctx, "invocation_id", "") or "")
        agent_name = str(_safe_attr(agent, "name", "") or "")
```

### Step 8.2 — Resolve the current task from the plan, not from a guess

To find "which plan task is this agent working on", match the plan by
**assignee + status**, exactly as `before_agent_callback` does (line ~2777):
find the unique task whose `assignee_agent_id == agent.name` and whose status
is PENDING or RUNNING. **Zero matches and multiple matches both leave the
state unset** — an explicit `missing_task_id` rejection is better than a
mis-attributed report. Do not predict; capture the unambiguous fact or
nothing (invariant 4).

### Step 8.3 — Respect per-invocation state discipline

Any state you stash per-invocation must be **keyed on `invocation_id`** and
**reset/consumed per invocation** — never carried across invocations on a bare
instance attribute (concurrent sessions and nested sub-Runners share the
plugin instance). The cancel state (`_cancel_state`, keyed by `invocation_id`,
read-then-clear via `consume_cancel_for_invocation`) is the reference. The
per-invocation asyncio-task registry lives on the **state store** (PR #303),
not on the plugin, precisely so concurrent sessions don't collide.

- Cross-invocation state that must survive belongs on a Python reference the
  plugin instance closes over (the adapter/plugin instance), **not** on
  `session.state` — ADK deep-copies `session.state` between turns and drops
  your mutation (the callback-context-handoff scar; see
  `feedback_callback_context_handoff`).
- `before_agent_callback` marks the goldfive boundary (`InvocationBoundaryEntered`)
  and the paired exit fires in `after_agent_callback` or the canonical
  `except CancelledError` catch in `_invoke_internal`. If you short-circuit a
  callback, do it **after** the boundary-entered emit so the entry/exit pair
  is balanced.

### Step 8.4 — Feed the observed fact into the pipeline, don't act in the callback

The callback **observes**; the steerer **classifies and dispatches**. Build a
`DriftEvent` (or call a registered detector) and hand it to the steerer:

```python
        drift = detect_my_signal(observed=..., task=...)
        if drift is not None and self._steerer is not None:
            await self._steerer.observe(...)   # or handle_drift for a ready DriftEvent
```

Do **not** gate on `is_active_steering()` in the callback — detection is
mode-independent (invariant 5); the steerer's dispatch does the gating
(Recipe 4). If your observation stamps a liveness watermark, update
`session.last_observed_event_at` (the #487 stall-watchdog liveness stamp).

### Step 8.5 — Tests

- `tests/test_adk_plugin_tool_observations.py` — feed a fake ADK
  callback_context/agent/tool through your callback, assert the observed
  `DriftEvent` (or the pinned `goldfive.current_task_id`).
- `tests/test_adk_adapter_concurrent_sessions.py` — assert two concurrent
  invocations don't cross-contaminate your per-invocation state.
- `tests/test_adk_adapter_pending_tool_isolation.py` if you touch tool
  lifecycle.
- Use the `testkit` fakes for ADK objects; never require a live model.

### Verification checklist (Recipe 8)

```bash
uv run pytest -q tests/test_adk_plugin_tool_observations.py \
  tests/test_adk_adapter.py tests/test_adk_adapter_concurrent_sessions.py \
  tests/test_adk_adapter_overlay.py
# every new ADK attr read goes through _safe_attr:
grep -n "getattr(.*callback_context\|getattr(.*agent\b\|\.invocation_context" goldfive/adapters/_adk_plugin.py | grep -v _safe_attr
uv run pytest -q
uv run ruff check .
```

The second grep should not surface your new reads (they should all be
`_safe_attr`).

### If any step fails

- `AttributeError` in a coordinator+AgentTool run but not a flat run →
  invariant 3; you used a bare attribute access, switch to `_safe_attr`.
- State leaks between concurrent sessions → you stashed on a bare instance
  attr instead of keying on `invocation_id`.
- Mutation invisible on the callback side → you wrote to `session.state`; ADK
  copied it. Bind to the plugin instance instead.
- Drift fires in observation-only and cancels → you acted in the callback
  instead of handing to the steerer; the callback must not gate or act.

---

## Recipe 9 — Add a reporting tool

**What this gives you.** A new agent-facing tool in the canonical reporting
set, with a schema, a handler, response-rendering shapes (including the
observation-only variant), registration, and updated docs counts.

**Preconditions.** The tool is genuinely part of the plan/task contract. Note
that reporting tools are **advisory** — an agent may never call them
(invariant 1); the tool must add signal, not become a required step.

### Step 9.1 — Define the JSON schema (task_id is OPTIONAL)

File: `goldfive/reporting/schemas.py`. Use the `_object_schema` helper. **A
task-scoped tool must NOT require `task_id`** — it is resolved from the pin
default (`goldfive.current_task_id`) when omitted, so a delegated sub-agent
that doesn't know its task id still works:

```python
# goldfive/reporting/schemas.py
_SCHEMA_MY_TOOL = _object_schema(
    required=["reason"],                 # NOT task_id
    properties={
        "task_id": {"type": "string"},   # optional; pin default fills it
        "reason": {"type": "string"},
    },
)
```

Add the constant name to the module `__all__` at the bottom.

### Step 9.2 — Write the handler

File: `goldfive/reporting/handlers.py`. Handler signature is
`async def _handle_x(args, session, steerer) -> dict[str, Any]`. Validate
required fields, resolve the task id (explicit > pin default), drive the
steerer or emit an observability event, return an ack dict:

```python
# goldfive/reporting/handlers.py
async def _handle_my_tool(
    args: dict[str, Any], session: Session, steerer: Steerer
) -> dict[str, Any]:
    err = _validate_required(args, _SCHEMA_MY_TOOL, "report_my_tool")
    if err is not None:
        return err
    task_id, _source = _resolve_task_id_with_source(args, session)
    if not task_id:
        return _missing_task_id_response("report_my_tool")
    reason = _str(args, "reason")
    # Either drive a transition via the steerer, OR (observability-only)
    # emit an event without mutating the plan — mirror _handle_declaration.
    await steerer.drift.report_my_thing(session=session, task_id=task_id, reason=reason)
    return dict(_ACK)
```

- **Never mutate plan state directly.** The steerer's `_apply_revision` is the
  only path that transitions a task. Observability-only tools (like
  `declare_task_skipped`) just emit a `TaskDeclarationReceived` and record an
  idempotent declaration (`_record_declaration`, keyed on `(kind, task_id)` —
  a stable key, invariant 6).
- Idempotency: a duplicate call should be a no-op that returns
  `{"acknowledged": True, "idempotent": True}`.

### Step 9.3 — Response shapes, including the observation-only variant

File: `goldfive/reporting/rendering.py`. Directive acks carry a `plan_state`
block (completed ids + next-pending hand-off, the F1 loop-prevention pattern)
— but **`plan_state` is goldfive-authored guidance and rides the
`observation_only` gate**. Use the shared builders, which already gate:

```python
# _directive_ack (rendering.py ~157) — the gate is already there:
    if steering_is_active(steerer):
        response["plan_state"] = _build_plan_state(getattr(session, "plan", None))
    return response
```

Under strict-passive, the ack keeps only the **factual echo** of the
transition the agent itself reported (`{"acknowledged": True, "task": {...}}`)
— no goldfive-authored `plan_state`. If your tool returns a directive ack, use
`_directive_ack(..., steerer=steerer)` and pass the steerer so the gate
applies. Do **not** hand-roll `plan_state` — it would bypass the gate
(invariant 5).

Rejection shapes are also in `rendering.py`: `_missing_task_id_response`,
`_missing_required_field_response`, `_invalid_transition_response`,
`_idempotent_response`, `_refused_response`. Reuse them.

### Step 9.4 — Register and update the canonical name list

File: `goldfive/reporting/handlers.py`. Add a `ReportingToolSpec` to
`BUILTIN_REPORTING_TOOLS` (list at line ~979) and add the name to
`REPORTING_TOOL_NAMES` (tuple at line ~114):

```python
# BUILTIN_REPORTING_TOOLS
    ReportingToolSpec(
        name="report_my_tool",
        description=(
            "Clear, imperative instruction for the LLM. Say WHEN to call it "
            "and what each arg means. Note it is advisory."
        ),
        parameters=_SCHEMA_MY_TOOL,
        handler=_handle_my_tool,
    ),
```

Adapters materialise `BUILTIN_REPORTING_TOOLS` automatically — no per-adapter
change needed (that is the point of the spec indirection). The `invoke_tool`
guard layers (`goldfive/adapters/_tool_invocation.py`) apply to your tool for
free.

### Step 9.5 — Update the docs counts

The docstrings and docs say "the eight/ten canonical reporting tools" in
several places. Grep and update **every** count:

```bash
grep -rn "canonical reporting tool\|eight\|ten canonical\|REPORTING_TOOL_NAMES" goldfive/ docs/ .agents/ | grep -i "tool"
```

Update `docs/reference/tool-protocol.md`, `goldfive/reporting/__init__.py`
module docstring, `.agents/how-to-add-a-new-adapter.md` "the seven reporting
tools" line, and the `REPORTING_TOOL_NAMES` comment ("The ten canonical
reporting tool names").

### Step 9.6 — Tests

- `tests/test_reporting_*` (or a new file) — handler validates required
  fields; resolves task_id from the pin when omitted; returns `_ACK` on
  success; returns the rejection shape on a bad call; is idempotent.
- `tests/test_observation_only_acks.py` — under `observation_only=True`, the
  ack does **not** include `plan_state`; under active mode it does.
- `tests/test_drift_reporting_optin.py` if the tool participates in the
  drift-opt-in selection.
- A test asserting `len(REPORTING_TOOL_NAMES) == len(BUILTIN_REPORTING_TOOLS)`
  and that every name has a spec (parity).

### Verification checklist (Recipe 9)

```bash
uv run pytest -q tests/test_observation_only_acks.py tests/test_drift_reporting_optin.py \
  tests/test_callable_adapter.py tests/test_adk_adapter.py
# name<->spec parity and that your tool is present:
uv run python -c "from goldfive.reporting import REPORTING_TOOL_NAMES, BUILTIN_REPORTING_TOOLS as B; names={s.name for s in B}; assert set(REPORTING_TOOL_NAMES)==names; assert 'report_my_tool' in names; print(len(names))"
grep -rn "canonical reporting tool" goldfive/ docs/   # counts updated?
uv run pytest -q
uv run ruff check .
```

### If any step fails

- A delegated sub-agent retry-loops on `missing_task_id` → you put `task_id`
  in `required`; remove it, resolve from the pin default.
- `plan_state` shows up under `observation_only=True` → you hand-rolled the
  ack instead of using `_directive_ack(..., steerer=steerer)`.
- Count-mismatch test fails → Step 9.5; a docstring still says "eight".
- Tool not visible to the agent → you added the spec but an adapter caches a
  name-map; adapters must materialise the full `BUILTIN_REPORTING_TOOLS`
  (Recipe 10 Step 10.1).

---

## Recipe 10 — Add an adapter

**What this gives you.** goldfive wrapping a new agent framework (LangGraph,
MCP, a bare callable, another SDK) as an `AgentAdapter`.

**Preconditions.** You can hook the framework's tool-dispatch and (ideally)
its chain-of-thought surface.

### Step 10.1 — Conform to the protocol

Live contract: `goldfive/protocols.py::AgentAdapter` (`@runtime_checkable`):

```python
@runtime_checkable
class AgentAdapter(Protocol):
    async def register_reporting_tools(self, tools: list[ReportingToolSpec]) -> None: ...
    async def invoke(self, task: Task, session: Session) -> InvocationResult: ...
    async def emit_reasoning(self, text: str, *, task=None, session, provider="", call_id="") -> None: ...
    @property
    def available_agents(self) -> list[str]: ...
```

Store the **full** `ReportingToolSpec` list (not a name→handler map) — you need
the specs to feed `invoke_tool`, which runs the guard layers:

```python
    async def register_reporting_tools(self, tools):
        self._tools: list[ReportingToolSpec] = list(tools)
        # translate each spec into the framework's native tool shape
```

### Step 10.2 — Route EVERY tool call through `invoke_tool`

Non-negotiable. The framework's tool hook MUST call
`goldfive.adapters._tool_invocation.invoke_tool` (line ~84), never
`spec.handler` directly:

```python
from goldfive.adapters._tool_invocation import invoke_tool
ack = await invoke_tool(self._tools, name, args, session, self._steerer)
```

`invoke_tool` runs four guards before `spec.handler`: schema rejection
(missing/unknown `task_id` on a task-scoped tool), terminal-task rejection
(`task_already_terminal` on COMPLETED/FAILED/CANCELLED), per-task loop guard
(duplicate-args → `duplicate`; bursts → hard-reject), and session-wide volume
cap. Calling `spec.handler` directly bypasses all four — that was the root of
the pre-#108 filler-loop class of bugs.

### Step 10.3 — Hook observation + reasoning

Forward every non-reporting event to the steerer as a raw observation:

```python
        if self._steerer is not None:
            await self._steerer.observe(event, session)
```

If the framework surfaces chain-of-thought (OpenAI `reasoning_content`,
Anthropic `thinking`, Google thought parts), implement `emit_reasoning` →
`steerer.observe_reasoning(...)`. An adapter that can't surface reasoning
simply never calls it; the reasoning-drift pipeline degrades to the structural
detectors. **Never require the agent to cooperate** (invariant 1) — you
observe what the framework exposes.

### Step 10.4 — Return a well-formed `InvocationResult`

```python
        return InvocationResult(
            task_id=task.id,
            text=final_assistant_text,
            stop_reason=framework_stop_reason,   # "end_turn" / "max_tokens" / ...
            error=exc_if_the_framework_raised_else_None,
            raw=raw_framework_response,
        )
```

The executor reads `stop_reason` for `CONTEXT_PRESSURE` / `TOO_MANY_STEPS` and
`error` to route through `steerer.mark_task_failed`. **Exceptions from the
wrapped agent must surface as `InvocationResult.error`, not as a raise out of
`invoke`.**

### Step 10.5 — Multi-agent trees + `available_agents`

If the framework nests agents, **walk the tree yourself** and wire reporting
tools into every descendant — do not require the caller to attach tools to
each child (invariant 3). `ADKAdapter._augment_subtree_with_reporting` walks
`sub_agents`, `inner_agent`, and every `tool.agent`; it is idempotent (agents
already carrying the canonical names are skipped). `available_agents` returns
every dispatchable name in the tree.

### Step 10.6 — Register auto-detection

File: `goldfive/adapters/auto.py`. Add a `_looks_like_*` sniffer and a dispatch
branch in `auto_adapter` (line ~117), mirroring the ADK / Claude / callable
branches. Order matters — put the most specific sniffer first. An object that
already `isinstance(x, AgentAdapter)` is passed through unchanged (first check).

### Step 10.7 — Update the parity table

Add a row to the adapter parity table in `.agents/adapters.md` (table at line
~93) and `docs/design/PROTOCOLS.md`:
`| MyAdapter | goldfive.adapters.my | goldfive[my] |`, plus a feature-parity
note (does it surface reasoning? nested agents? cancellation?).

### Step 10.8 — Tests

Reference: `tests/test_callable_adapter.py`, `tests/test_claude_adapter.py`,
`tests/test_adk_adapter.py`. At minimum:

- `isinstance(adapter, AgentAdapter)` — protocol-compliant.
- `register_reporting_tools([])` — empty list doesn't crash.
- `invoke` routes tool calls through `invoke_tool` — a terminal-status task
  receives `task_already_terminal` (proves the guard ran).
- A wrapped-agent exception surfaces as `InvocationResult.error`, not a raise.
- `auto_adapter(my_agent)` returns your adapter.

### Verification checklist (Recipe 10)

```bash
uv run pytest -q tests/test_callable_adapter.py tests/test_adk_adapter.py tests/test_protocols.py
uv run python -c "from goldfive.protocols import AgentAdapter; from goldfive.adapters.my import MyAdapter; assert isinstance(MyAdapter(...), AgentAdapter)"
# no direct spec.handler calls (must go through invoke_tool):
grep -n "spec.handler\|\.handler(" goldfive/adapters/my.py
uv run pytest -q
uv run ruff check .
```

### If any step fails

- Filler-loop / agent re-invoked after COMPLETED → Step 10.2, you called
  `spec.handler` directly and bypassed the terminal-task guard.
- `isinstance` False → missing a protocol method or a non-async method.
- Nested agents don't report → Step 10.5, you didn't walk the tree.
- `invoke` raises on a framework error instead of returning → Step 10.4, catch
  it into `InvocationResult.error`.

---

## Recipe 11 — Safe dead-code deletion (the archaeology protocol)

**What this gives you.** A defensible deletion that does not remove
load-bearing or protected code. goldfive has repeatedly resurfaced code that
*looked* dead but was reachable through a non-obvious path, and has a formal
KEEP list of code that looks deletable but must stay.

**Preconditions.** You believe a symbol is dead. You are about to prove it —
or discover it isn't. #490 (commit `f9d7e91`) is the reference for the full
protocol.

### Step 11.1 — Grep the whole tree for consumers

```bash
grep -rn "my_symbol" goldfive/ tests/ examples/ bench/ docs/ .agents/
```

Include tests, examples, docs, and the `.agents` skills. A symbol referenced
only by its own definition + its own test is a candidate; a symbol referenced
by production code is **not dead**.

### Step 11.2 — `git log -S` archaeology

```bash
git log -S "my_symbol" --oneline -- goldfive/       # every commit that added/removed the string
git log -p -S "my_symbol" -- goldfive/ | head -100  # WHY it was added
```

You are looking for: was this ever wired into a real path? Was it wired and
then **unwired** by a specific PR (e.g. `detect_unreferenced_keyword` was
unwired in #226/#230)? If it was deliberately unwired and never rewired, it is
a safe delete. If it was added "for a future PR" that hasn't landed, check the
next steps before deleting.

### Step 11.3 — Check the PROTECTED KEEP list

**Do NOT delete or "fix" these without explicit human sign-off**, no matter
how dead they look:

| Protected | Why | History |
|---|---|---|
| `LOOPING_TOOL_CALL` enum / ladder / promotion / planner surfaces | tool loops deliberately emit `LOOPING_REASONING` with NUDGE-first CRITICAL routing | #204 / #206 |
| `PLAN_DIVERGENCE` machinery | disabled in #252 but the branch is a deliberate KEEP | #252 |
| `reconciler.get_missed_tasks` | KEEP | #163 |

If your symbol is on this list (or is a `JUSTIFIED_DEVIATION`-style
"shipped-but-not-yet-wired" surface — see the `types.py` comment at line ~236
saying "the lack of a `_LADDER` row is intentional"), STOP. Its absence of
callers is by design, not death.

### Step 11.4 — Check the agency-preservation roadmap

The **agency-preservation branch** (`#453-#474`, unmerged) holds Stages 1–3
behind default-OFF flags. **Main-side code must not copy from it, and you must
not delete a main-side symbol that the branch's as-built record depends on.**
The roadmap doc is `docs/design/AGENCY-PRESERVATION.md` (on `main` it goes
through §5 Correctness strategy; the §6 as-built record + KEEP annotations
live on the branch). If your symbol appears in that doc's plan, confirm with a
human before deleting — the branch may re-wire it.

Also confirm the symbol is not on the **DEFERRED work** list (twin-refine-
pipeline extraction, evidence-ledger replacement of the ~7 stacked
`handle_drift` suppression gates, judge windowing, judge-facade dispatch
authority, checkpoint-rollback / tool-gating / fork-and-judge) — deferred
scaffolding sometimes lands ahead of the feature.

### Step 11.5 — Delete the code AND its tests in the same commit

Once proven dead and unprotected:

- Delete the symbol.
- Delete its now-orphaned tests **in the same commit** (a test for deleted
  code is itself dead).
- Fix any docstrings/comments that referenced it (the #490 commit "fixed the
  docstrings that claimed [`_LADDER_BY_VALUE`] was populated").
- Write a commit body that records the archaeology per symbol: the grep
  result, the `git log -S` finding, and the KEEP-list check. Mirror the #490
  commit message format.

### Verification checklist (Recipe 11)

```bash
# consumers really gone (should return only the deletion diff context):
grep -rn "my_symbol" goldfive/ tests/ examples/ docs/
# full suite must pass unchanged (no test relied on it):
uv run pytest -q
uv run ruff check .
# nothing imports the deleted name:
uv run python -c "import goldfive; print('import OK')"
```

### If any step fails

- `pytest` collection error after deletion → a test still imports the symbol;
  delete that test (it was testing dead code) or you deleted something live.
- `grep` still finds a production consumer → **not dead**; abort the deletion.
- The symbol is on the KEEP list or in AGENCY-PRESERVATION → do not delete
  without human sign-off.
- Unsure whether "added for a future PR" scaffolding is safe → leave it; the
  cost of a wrong delete (re-doing the work, re-litigating the design) exceeds
  the cost of a few dead lines.

---

## Recipe 12 — Update a design doc

**What this gives you.** A `docs/design/*.md` change that stays consistent
with the code and with sibling docs. Design docs are **sources, not ground
truth** — the code on `main` wins. #492 was a whole-repo accuracy sweep for
exactly this reason.

**Preconditions.** You changed behaviour (or found a doc that lies about
current behaviour) and need the doc to match.

### Step 12.1 — Apply the code-wins rule

Before writing a sentence, verify the claim against `main`:

- Cite **file paths + symbol names**, not bare line numbers (they rot). When a
  line number genuinely helps, pair it with the symbol.
- If the doc and the code disagree, **change the doc to match the code** (or,
  if the code is wrong, fix the code in a separate PR — do not paper over a
  bug in prose).
- Say so when you correct a doc: "Pre-#NNN the doc claimed X; the code does
  Y."

### Step 12.2 — Verify each claim you touch

For every factual claim in the paragraph you edit, run the grep/command that
proves it:

```bash
# claim: "the default is observation_only=True"
grep -n "observation_only: bool" goldfive/config.py
# claim: "there are ten reporting tools"
uv run python -c "from goldfive.reporting import REPORTING_TOOL_NAMES; print(len(REPORTING_TOOL_NAMES))"
# claim: "GOAL_DRIFT routes to NUDGE at WARNING"
grep -n "GOAL_DRIFT" goldfive/drift_observer.py
```

Do not restate a number, default, or routing from memory — pull it from the
code every time.

### Step 12.3 — Cross-doc consistency greps

A fact often appears in several docs. When you change one, grep the others and
update them in lockstep:

```bash
grep -rn "observation_only\|default is .*True" docs/design/
grep -rn "REFINE_FAILURE_THRESHOLD\|reporting tool" docs/design/ .agents/
grep -rn "DRIFT_KIND_\|DriftKind\." docs/design/DRIFT.md docs/design/VOCABULARY.md
```

The taxonomy in `DRIFT.md` and the enum-by-enum reference in `VOCABULARY.md
§5` must agree with `goldfive/types.py::DriftKind` and
`proto/goldfive/v1/types.proto`. The intervention-ladder descriptions in
`CONTROL.md` / `DRIFT.md` must agree with `DriftObserver._LADDER`.

### Step 12.4 — Respect the doc's role in the guide

These design docs are cross-referenced by the dev guide chapters. When you
change `DRIFT.md`, the drift chapters (`07`, `08`) and this recipe's Recipe 1
Step 1.8 point at it — keep the taxonomy table shape stable (columns
`| Kind | Trigger | Default severity | Recoverable |`). Do not restructure a
table other chapters link into without updating them.

### Step 12.5 — No behaviour change, no test — but run the doc-adjacent checks

A doc-only change has no runtime surface, so there is nothing to unit-test.
But run the consistency greps above as your "test", and if the doc embeds a
code snippet, verify the snippet still imports/runs:

```bash
# if the doc shows a code example, sanity-check it:
uv run python -c "<paste the doc's example>"
```

### Verification checklist (Recipe 12)

```bash
# every DriftKind in the doc exists in code:
grep -oE "DRIFT_KIND_[A-Z_]+" docs/design/DRIFT.md | sort -u | while read k; do
  uv run python -c "from goldfive.pb.goldfive.v1 import types_pb2 as t; t.DriftKind.Value('$k')" || echo "STALE: $k"
done
# cross-doc counts agree:
grep -rn "canonical reporting tool" docs/ goldfive/
# lint is unaffected but run it anyway (catches broken code fences via examples):
uv run ruff check .
```

### If any step fails

- A `DRIFT_KIND_*` in the doc doesn't resolve → the doc is stale (kind
  retired/renamed) or you typo'd; fix the doc.
- Two docs now disagree → Step 12.3; you updated one and not the other.
- A doc code example no longer runs → the API changed under it; update the
  example to match `main` (code wins).
- Tempted to document a feature from the agency-preservation branch as if it's
  on `main` → don't. Those features are behind default-OFF flags on an
  unmerged branch; doc text on `main` must not claim they exist here.

---

## Appendix — the universal post-change gate

Every recipe ends by running these. Memorise them:

```bash
uv sync --extra dev --extra adk           # (+ --extra proto if you touched .proto)
uv run pytest -q                          # ~30s, expect ~2912 passed / 61 skipped
uv run ruff check .                       # must stay clean
```

Notes that apply to all recipes:

- The suite runs the shipped `observation_only=True` default since #488; only
  ~90 tests opt into active mode explicitly. If your change assumes active
  steering, set `SteeringConfig(observation_only=False)` in the test, don't
  flip a global.
- The repo is **not** `ruff-format`-clean — run `ruff check`, never a
  mass-reformat (it would explode the diff).
- CI is `lint-and-test` on Python 3.11 and 3.12 with the `dev`, `adk`, and
  `proto` extras. `make proto` is NOT run in CI — regenerate and commit stubs
  yourself.
- **No Claude co-author trailer** in goldfive commits.
- Ship tests + docs **with** the code, in the same PR — never "after".
