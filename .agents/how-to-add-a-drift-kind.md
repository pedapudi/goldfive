---
name: how-to-add-a-drift-kind
description: Checklist for adding a new DriftKind to goldfive. Covers the Python enum, the proto enum, codegen, classifier wiring, the steerer dispatch, tests, and the docs/design/DRIFT.md taxonomy row.
applies-when: ["add drift kind", "new drift signal", "DriftKind enum", "drift taxonomy"]
---

# Add a new `DriftKind`

> **Canonical procedure: [docs/dev-guide/16-recipes.md](../docs/dev-guide/16-recipes.md)
> Recipe 1 (Add a DriftKind end-to-end).** This skill is the terse
> checklist; the recipe is the full, code-verified walk (enum → proto →
> stubs → detector → ladder row → wire → taxonomy doc). Where they
> differ, the recipe wins. Read
> [docs/dev-guide/17-invariants-hazards-history.md](../docs/dev-guide/17-invariants-hazards-history.md)
> first — invariant 2 (no regex/keyword NL classifiers) binds this task.

Every drift signal goldfive understands is a `DriftKind` enum
value. Adding one is a walk across Python, proto, a detector, the
intervention ladder, tests, and the taxonomy doc.

## 1. Add the Python enum value

`goldfive/types.py::DriftKind` — add a new `StrEnum` member. Place
it with other members of the same category (error, divergence,
structural, discovery, user, goal, reasoning, reflective, escape)
so code-readers navigating the enum find it where they expect.

```python
class DriftKind(StrEnum):
    ...
    # Opt-in reflective self-progress check: agent said it *is*
    # making progress but with low confidence (< 0.5). INFO severity.
    UNCERTAIN_PROGRESS = "uncertain_progress"
```

Include a short docstring-style comment explaining default
severity, trigger, and any feature gate. If the kind fires at a
**variable** severity (like `INTENT_DIVERGENCE`), say so
explicitly — callers filtering by kind shouldn't assume severity
is fixed.

## 2. Mirror the value in the proto enum

`proto/goldfive/v1/types.proto` — add `DRIFT_KIND_<NAME>` to the
`DriftKind` enum. Preserve existing numbering; append. Proto3
requires stable numbering for forward compatibility.

```proto
enum DriftKind {
  ...
  DRIFT_KIND_UNCERTAIN_PROGRESS = 30;
  DRIFT_KIND_SELF_REPORTED_STUCK = 31;
}
```

The Python-to-proto bridge is by name: the steerer looks up
`getattr(types_pb2, f"DRIFT_KIND_{kind.name}")` at emission time
(see `DefaultSteerer._emit_drift_detected` and
`SequentialExecutor._plan_divergence_drift_event`). The Python
enum name and the proto enum suffix must match exactly.

**Next free slot.** As of #492 the highest assigned slot is
`DRIFT_KIND_CAPABILITY_MISMATCH = 41`, so the next new kind is `42`.
Always re-check the current maximum in `types.proto` before picking —
never fill a gap, never reuse a retired number (e.g. `CONFUSION = 28`
is reserved), and append at the end so two concurrent PRs don't
collide.

## 3. Regenerate the proto stubs

```bash
uv sync --extra proto
make proto
```

This regenerates `goldfive/pb/goldfive/v1/types_pb2.py` and
`types_pb2.pyi`. Verify the new value is listed:

```bash
uv run python -c "from goldfive.pb.goldfive.v1 import types_pb2; print([n for n in dir(types_pb2) if 'DRIFT_KIND' in n])"
```

## 4. Wire a detector (if the detection is non-trivial)

For drifts whose detection is non-trivial add a detector helper.
Allowed detection methods: exact-equality / hash matching of
structured data, embedding cosine, threshold on session state, or an
LLM classifier. **NOT allowed** (invariant 2, retired #166/#167):
regex or keyword heuristics over natural language. If detection needs
to read intent from prose, use an LLM judge (Recipe 3), not a pattern.
Two homes:

- **Upstream / tool-facing drifts** — `goldfive/drift/__init__.py`
  alongside `classify_tool_error`, `classify_refusal`,
  `classify_stop_reason`. Signature:
  ```python
  def classify_my_new_thing(event: Any) -> DriftEvent | None: ...
  ```
  Pure function. Returns `DriftEvent` or `None`.

- **Reasoning-channel drifts** —
  `goldfive/drift/reasoning.py`. Follow the pattern of
  `detect_intent_divergence`: pattern-first fallback, optional
  embedding path when `goldfive[embedding]` is installed, any
  tunable thresholds as module-level constants prefixed with the
  drift kind name
  (e.g. `INTENT_DIVERGENCE_WARNING_SIMILARITY`,
  `LOOPING_REASONING_SIMILARITY_THRESHOLD`).

For drifts that fire directly from a reporting-tool handler
(`report_plan_divergence`, `report_new_work_discovered`,
`report_task_blocked`), there's no separate classifier — the
handler emits the `DriftEvent` inline.

## 5. Add the intervention-ladder row (NOT the steerer)

The ladder moved OFF `DefaultSteerer` and onto `DriftObserver` in the
bucket-3c refactor. Its home is
`goldfive/drift_observer.py::DriftObserver._load_ladder_tables`, which
populates the class attribute `_LADDER` (an
`{InterventionLevel-tuple}` per kind). Omit the row to inherit the
default shape; add a row only for a non-default shape. See Recipe 1
Step 1.5 for how the 3-tuple `(info, warning, (crit_first, crit_repeat))`
is read.

**Stamp `detector_name` when the kind is shared (#480).** When you
construct the `DriftEvent`, set `detector_name` ONLY if more than one
detector can mint this `DriftKind` (e.g. the tool-loop tracker stamps
`detector_name="tool_loops"` because it also emits `LOOPING_REASONING`).
A kind with a single detector leaves it `""`. It feeds
`SteeringDecisionMade.detector_name` in decision telemetry.

**Tool-loop name-axis cap (#484).** If your detector is a tool-loop
variant, a repeat on the *name* axis alone is capped at INFO
(`name_axis_max_severity`, `raw["severity_capped_from"]`) unless there
are `>=2` identical `(name, args_hash)` exact repeats. Don't promote on
name-only repetition.

**Capability-check negative class (#480).** If you extend
`capability_check`, remember it emits a negative (no-mismatch)
decision-telemetry class too — don't drop the negative path.

`goldfive/steerer.py::DefaultSteerer.detect_drift` (for passive
classification on adapter events) or a direct call site in the
reporting-tool handler / post-invoke hook (for signals that
originate from a specific entry point).

For reflective / periodic checks — the #112 pattern — add a new
public method to `DefaultSteerer` that the adapter invokes on
its own schedule (e.g. once per LLM turn). Gate behind a ctor
flag so operators who don't enable it pay zero cost:

```python
def __init__(self, *, my_check_interval: int = 0, my_call_llm=None):
    self._my_check_interval = my_check_interval
    self._my_call_llm = my_call_llm

async def note_llm_call(self, session):
    if self._my_check_interval <= 0 or self._my_call_llm is None:
        return
    session._llm_calls_since_check += 1
    if session._llm_calls_since_check < self._my_check_interval:
        return
    session._llm_calls_since_check = 0
    drift = await self._run_my_check(session)
    if drift is not None:
        await self._handle_drift(drift, session)
```

If the drift fires at graduated severity (like `INTENT_DIVERGENCE`
does after #114), the classifier decides severity based on the
observation; the kind stays stable and only the
`DriftEvent.severity` varies. Callers filtering by kind then see
one signal; severity differentiates urgency.

## 6. Add taxonomy tests

`tests/test_drift_taxonomy.py` is the canonical home. At minimum:

- **The enum value exists** — no ImportError, `DriftKind.MY_NEW` resolves.
- **Proto mirror** — `types_pb2.DRIFT_KIND_MY_NEW` exists with a non-zero value.
- **Severity round-trips** — `DriftEvent(kind=MY_NEW, severity=WARNING)`
  serialises through `drift_detected_event` and deserialises without
  collapsing to `UNSPECIFIED`.
- **Classifier returns the right shape** — feed an in-class event,
  assert the returned `DriftEvent.kind` / `severity` match.

If the drift fires with graduated severity, add a severity-band
test per band (INFO / WARNING / CRITICAL) with representative
inputs.

For reasoning-channel drifts, the reasoning-specific test suite is
`tests/test_drift_reasoning.py`. Add band / threshold tests there.

## 7. Update `docs/design/DRIFT.md`

Add a row to the taxonomy table under the appropriate category
(Error / Divergence / Structural / Discovery / User / Goal /
Reasoning / Reflective). Include:

- Trigger — one line naming the event / tool / threshold that fires it.
- Default severity — `info`, `warning`, `critical`, or "graduated".
- Recoverable — yes / no / sometimes.

If the drift is graduated-severity, add a sub-section below the
table with the band boundaries and rationale, mirroring the
`INTENT_DIVERGENCE` treatment in DRIFT.md.

## 8. (Optional) UI metadata in harmonograf

Harmonograf's front-end renders per-kind icons and colors. If the
new kind should not fall through to the generic drift badge, file
an issue / PR on harmonograf to add the icon/color mapping. Not
required for the kind to function — the UI degrades gracefully.

## Checklist (copy-paste)

```
[ ] goldfive/types.py::DriftKind — enum value + comment
[ ] proto/goldfive/v1/types.proto — DRIFT_KIND_<NAME> (next free slot; 42+ as of #492)
[ ] make proto — regenerate types_pb2 (commit the stubs)
[ ] goldfive/drift/__init__.py or drift/reasoning.py — detector (no regex/keyword NL)
[ ] DriftEvent construction — detector_name only if kind is shared (#480)
[ ] goldfive/drift_observer.py::_load_ladder_tables — _LADDER row (only if non-default)
[ ] tests/test_drift_taxonomy.py — enum + proto + severity; BOTH modes for any wire action
[ ] tests/test_drift_reasoning.py — if reasoning-channel
[ ] docs/design/DRIFT.md — taxonomy row + optional sub-section
[ ] CHANGELOG.md — Added / Changed entry
[ ] (harmonograf) UI metadata
```

## Related

- [docs/dev-guide/16-recipes.md](../docs/dev-guide/16-recipes.md) — **Recipe 1 (canonical, code-verified end-to-end procedure).**
- [docs/dev-guide/07-deterministic-drift-detection.md](../docs/dev-guide/07-deterministic-drift-detection.md) / [08-llm-judges.md](../docs/dev-guide/08-llm-judges.md) — where detectors and judges live.
- [docs/design/DRIFT.md](../docs/design/DRIFT.md) — full taxonomy + refine policy.
- [docs/design/VOCABULARY.md §5](../docs/design/VOCABULARY.md#5-driftkind-taxonomy) — enum-value-by-enum-value reference.
- [events.md](events.md) — proto event factories.
- `goldfive/drift/reasoning.py` — reference implementation for a reasoning-channel drift (with both pattern and embedding paths).
