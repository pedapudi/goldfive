# Cancellation propagation contract

**Status.** Phase 3 Addition A of goldfive#271. Sibling to
[`STATE-OWNERSHIP-CONTRACT.md`](STATE-OWNERSHIP-CONTRACT.md). The
exception class :class:`goldfive._state_audit.CancellationStashViolation`
ships in Phase 3; the runtime tripwire mechanism lands with Phase 3.5
(hard-cancel) when the goldfive task boundary becomes the legitimate
catch site.

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
| C2 | `goldfive/executors/parallel.py` (multiple) | various | event sequence + sink emit | survey ongoing | Several `try / except Exception` blocks wrap awaits but most stash via `_lock` re-entry — re-audit when boundary lands |
| C3 | `goldfive/executors/sequential.py` (multiple) | various | event sequence + sink emit | survey ongoing | Same shape as C2 |
| C4 | `goldfive/steerer.py` (multiple) | refine paths | `_background_judges` set | survey ongoing | Background tasks deliberately swallow on cancel via shutdown drain |
| C5 | `goldfive/_correction_injection.py` | `clear_correction` | per-task correction body | LOW priority | Best-effort GC; `try / except Exception` swallow is acceptable |
| C6 | `goldfive/reporting.py` (sink emits) | various | event id stamping | LOW priority | Sink emits are best-effort; broken sink shouldn't break tool calls |

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

1. Survey C2-C6 with the full audit pattern; convert HIGH/MEDIUM
   sites to `try / finally` or `except BaseException: stash; raise`.
2. Wire the runtime tripwire — install
   :class:`CancellationStashViolation` raise machinery (paired with
   the boundary's stack-walk; the boundary becomes the one
   legitimate catch site, and any other site that absorbs
   `CancelledError` without entering its `finally` raises the
   tripwire).
3. Test surface: `tests/test_cancellation_stash_audit.py` —
   parametrized over the catalog, fires `CancelledError` at the
   await, asserts the stash invariant.

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
