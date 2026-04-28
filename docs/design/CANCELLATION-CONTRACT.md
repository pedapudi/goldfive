# Cancellation propagation contract

**Status.** Phase 3 Addition A of goldfive#271. Sibling to
[`STATE-OWNERSHIP-CONTRACT.md`](STATE-OWNERSHIP-CONTRACT.md). The
exception class :class:`goldfive._state_audit.CancellationStashViolation`
ships in Phase 3; the C2-C6 audit conversions ship in Phase 3.5
(this PR). The runtime tripwire mechanism still lands with the
hard-cancel boundary work, when the goldfive task boundary becomes
the legitimate catch site.

## tl;dr

`asyncio.CancelledError` has been a `BaseException` subclass since
Python 3.8. `except Exception:` does NOT catch it. Code that wraps
an `await` in `try / except Exception:` and owns state-stash duties
(plan snapshot, event flush, drift-detector teardown) silently
**bypasses** the housekeeping when `CancelledError` propagates —
because the broad catch fires only on `Exception` subclasses, and
`CancelledError` skates past straight through whatever cleanup the
broader `try` block would have performed before the catch.

The validation-v2 audit found this happening at
`goldfive/runner.py:411` (the executor's `try / except Exception`
that wraps the per-task drive). When ADK closed the runner
mid-stream, the executor's `except Exception` block skipped every
post-execution housekeeping step. The fix shipped as the
**`runner.py:411` `try / finally`** in PR #287 (goldfive#271 Gap 1
v2): the prior plan snapshot now lives in `finally`, not in the
`except Exception` body, so it runs regardless of whether the await
returned normally, raised an `Exception`, or raised
`CancelledError`.

## §1 Compliant patterns

### §1.1 Preferred — `try / finally`

```python
try:
    result = await some_call()
    self._handle_result(result)
except Exception as exc:
    self._handle_failure(exc)
finally:
    # ALWAYS runs, including when CancelledError propagates.
    self._stash_prior_state()
    self._flush_events()
```

This is the form Gap 1 v2 (PR #287) used to fix the runner.py bug.

### §1.2 Acceptable — `except Exception ... except BaseException: stash; raise`

When the `Exception` branch needs specific recovery behaviour that
doesn't fit the bare-`finally` shape:

```python
try:
    result = await some_call()
except Exception as exc:
    self._record_recoverable_failure(exc)
    return None
except BaseException:
    # CancelledError (or KeyboardInterrupt) — preserve the stash AND re-raise.
    self._stash_prior_state()
    raise
```

The `except BaseException: stash; raise` MUST `raise` — swallowing
`CancelledError` would defeat asyncio's cancellation contract. This
form is reserved for the cases where the `Exception` block does
something the `finally` shape can't (e.g. computing a recoverable
default before swallowing the exception entirely).

## §2 Catalog of audited sites

This section mirrors the §5 audit catalog of
[`STATE-OWNERSHIP-CONTRACT.md`](STATE-OWNERSHIP-CONTRACT.md).
Phase 3 ships an audit pass; Phase 3.5 wires the runtime tripwire.

| ID | File | Function | Owns | Status | Notes |
|---|---|---|---|---|---|
| C1 | `goldfive/runner.py:411` | `_drive_internal` | prior-plan stash | **FIXED** in #287 (`try / finally`) | Validation v2 root-cause |
| C2 | `goldfive/executors/parallel.py` (refine) | `_refine` (steerer-bound + legacy) | CRITICAL drift mirror (`_emit_refine_failure`) | **FIXED** in Phase 3.5 (`except BaseException: emit; raise`) | Both `await planner.refine` call sites; CancelledError now lands the operator-visible "refine cancelled" mirror before propagating |
| C3 | `goldfive/executors/sequential.py` | refine paths | event sequence + sink emit | **FIXED** transitively via C4 | Sequential refines route through `DefaultSteerer._handle_drift` / `_promote_drift_to_steer`; no direct `await planner.refine` call site here. The mid-stage `await steerer.observe` blocks own no post-await stash (the sequence cursor self-heals on the next emit), so the strict criterion (await + state-stash bypass) does not match any sequential.py site. |
| C4 | `goldfive/steerer.py` | `_handle_drift`, `_promote_drift_to_steer`, `observe_refine` | paired `refine_failed` (attempt-id correlation) | **FIXED** in Phase 3.5 (`except BaseException: emit; raise`) | All three refine entry points now emit the paired `refine_failed` envelope on `CancelledError` before re-raising, so sinks never see an unmatched `refine_attempted`. The fire-and-forget `_run_judge_background` was already correct (CancelledError handled before `except Exception`; the `_background_judges` set is drained via `shutdown` + `add_done_callback`) |
| C5 | `goldfive/reporting.py` (`_handle_task_started`) | post-`mark_task_running` correction GC | per-task pending-correction entry | **FIXED** in Phase 3.5 (`try / finally`) | The post-await `_clear_correction_on_started` is now in `finally`, so a `CancelledError` raised inside `await steerer.mark_task_running` no longer leaves a pending correction wedged on session state for an already-acknowledged task. The standalone `clear_correction` helper is sync and was never the bypass site — the bypass was the call-site sequencing in `_handle_task_started` |
| C6 | `goldfive/reporting.py` (sink emits) | various | event id stamping | **FIXED** by construction | Every reporting sink-emit helper calls `session.next_sequence()` BEFORE the `await emit(...)`, so the cursor is already advanced by the time `CancelledError` lands. The `try / except Exception` around the emit is best-effort observability, not a state-stash. The pinning regression test in `tests/test_cancellation_stash_audit.py::test_c6_sequence_cursor_advances_before_emit_await` makes a future refactor that moves `next_sequence()` after the await regress loudly. |

### §2.1 Severity classification

* **HIGH** — stash carries cross-await state that downstream state
  machines depend on. C1 (the runner.py:411 bypass) was HIGH; the
  prior plan snapshot is consumed by user-steer routing, so
  losing it means the next user steer is mis-routed. **Status:
  fixed**.
* **MEDIUM** — stash owns observability state (event sequence
  cursors, sink dispatch). Loss is observable in the timeline gap
  but doesn't break state machines downstream. C2 / C3 are
  MEDIUM; the sequence cursor self-heals on the next emit, but
  the gap is operator-visible.
* **LOW** — stash is best-effort GC of transient state. C5 / C6
  are LOW; the GC will run on the next applicable event.

### §2.2 Phase 3.5 plan

When the goldfive task boundary lands (Phase 3.5):

1. **Done.** Survey C2-C6 with the full audit pattern; convert
   HIGH/MEDIUM sites to `try / finally` or
   `except BaseException: stash; raise`. Shipped in the
   Phase 3.5 cancellation-stash PR — see §2 catalog status column.
2. Wire the runtime tripwire — install
   :class:`CancellationStashViolation` raise machinery (paired with
   the boundary's stack-walk; the boundary becomes the one
   legitimate catch site, and any other site that absorbs
   `CancelledError` without entering its `finally` raises the
   tripwire).
3. **Done.** Test surface:
   `tests/test_cancellation_stash_audit.py` — fires
   `CancelledError` at each audited `await`, asserts the stash
   invariant. Each test fails on `origin/main` (without the
   conversion) and passes with the fix. C6 is pinned by a
   complementary regression test that asserts
   `session.next_sequence()` runs strictly before the sink
   `await`.

## §3 Why the boundary is prerequisite

The goldfive task boundary (Phase 3.5) is the ONE place we expect
`CancelledError` to be caught — the boundary converts it into a
structured marker (`InvocationBoundaryExited(reason="cancelled")`)
and emits an `InvocationCancelled` sink event before letting ADK
see a normal return. Without that boundary, every `try / finally`
conversion in the audit catalog would bubble `CancelledError` all
the way to ADK's runner, which has its own ideas about what
cancellation means and may surface it as a tool failure or a model
exception.

The audit-pass + class-only ship in Phase 3 lets us land the
boundary in 3.5 without simultaneously fighting a regression
cascade across every catalogued site. The ordering is deliberate:

1. **Phase 3 (this PR).** Audit catalog. Class definition. Doc.
2. **Phase 3 follow-up.** Convert HIGH-severity sites to
   `try / finally`. Most are already done (C1 in #287; the
   sequential executor's cooperative-cancel paths are
   self-cleaning by design).
3. **Phase 3.5.** Goldfive task boundary becomes the catch site.
   Runtime tripwire installs. Remaining audit sites convert in
   the same PR.
4. **Phase 4+.** Hard cancel wires `task.cancel()` from the steerer.
