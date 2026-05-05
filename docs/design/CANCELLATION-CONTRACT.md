# Cancellation propagation contract

See [CONTROL-CHANNEL.md](CONTROL-CHANNEL.md) for the actor-model
context: cancellations propagate as part of channel-mediated plan
revisions, and the audit catalog below covers the channel-processor
sites where post-cancel state stash must survive `CancelledError`.

**Status.** Phase 3 Addition A of goldfive#271. Sibling to
[`STATE-OWNERSHIP-CONTRACT.md`](STATE-OWNERSHIP-CONTRACT.md).

* :class:`goldfive._state_audit.CancellationStashViolation` shipped
  in Phase 3 (PR #290).
* The C2-C6 audit conversions shipped in Phase 3.5 component 1
  (PRs #306, #307 + the boundary catch site at
  :meth:`ADKAdapter._invoke_internal`).
* The runtime tripwire — every audited site wrapped in
  :func:`goldfive._state_audit.cancellation_stash_audited` and the
  boundary catch site invoking
  :func:`goldfive._state_audit.assert_stash_invariant` — shipped
  in Phase 3.5 component 2 (this PR). **The tripwire is now active.**

**Default-off.** The tripwire is gated on
``GOLDFIVE_STRICT_STATE_OWNERSHIP`` (the same env switch as
:class:`StateOwnershipViolation`). Production deploys with the
variable unset never raise from the boundary check; the audited
sites pay one ContextVar push/pop per enter/exit and nothing more.
Tests run with strict mode on by default via the auto-applied
``_state_audit_enabled`` fixture in ``tests/conftest.py``.

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
| C2 | `goldfive/executors/parallel.py` (refine) | `_refine` (steerer-bound + legacy) | CRITICAL drift mirror (`_escalate_refine_failure_as_critical_drift`) | **FIXED** in Phase 3.5 (`except BaseException: emit; raise`) | Both `await planner.refine` call sites; CancelledError now lands the operator-visible "refine cancelled" mirror before propagating |
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

### §2.2 Phase 3.5 plan — completed

1. **Done.** Survey C2-C6 with the full audit pattern; convert
   HIGH/MEDIUM sites to `try / finally` or
   `except BaseException: stash; raise`. Shipped in the
   Phase 3.5 cancellation-stash PR — see §2 catalog status column.
2. **Done.** Wire the runtime tripwire. Every audited site
   (C1-C5) is wrapped in
   `goldfive._state_audit.cancellation_stash_audited(name)`; the
   compliance branch (`finally:` or `except BaseException:`) calls
   `goldfive._state_audit.mark_stash_completed()` before re-raising;
   the boundary at `ADKAdapter._invoke_internal`'s
   `except asyncio.CancelledError:` arm calls
   `goldfive._state_audit.assert_stash_invariant(cause=...)` which
   walks the open-marker stack and raises
   `CancellationStashViolation` (a `BaseException` so
   `except Exception` cannot swallow it) for any retained marker
   whose compliance flag is False. C6 needs no instrumentation
   because the sequence cursor advances before the `await` by
   construction.
3. **Done.** Test surface:
   * `tests/test_cancellation_stash_audit.py` — fires
     `CancelledError` at each audited `await`, asserts the stash
     invariant. Each test fails on `origin/main` (without the
     conversion) and passes with the fix.
   * `tests/test_cancellation_stash_tripwire.py` — synthetic
     audited sites that verify the tripwire raises
     `CancellationStashViolation` for a bypass site under strict
     mode, never raises for a compliant site, and never raises
     under strict-mode-off regardless of compliance.

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
cascade across every catalogued site. The ordering was:

1. **Phase 3.** Audit catalog. Class definition. Doc.
2. **Phase 3 follow-up.** Convert HIGH-severity sites to
   `try / finally`. Most were already done (C1 in #287; the
   sequential executor's cooperative-cancel paths are
   self-cleaning by design).
3. **Phase 3.5 component 1.** Goldfive task boundary becomes the
   catch site. C2-C5 audit conversions to
   `except BaseException: stash; raise` or `try / finally` shipped.
4. **Phase 3.5 component 2 (this PR).** Runtime tripwire wired:
   audited sites instrumented via
   `cancellation_stash_audited` + `mark_stash_completed`; boundary
   catch site invokes `assert_stash_invariant`.
5. **Phase 4+.** Hard cancel wires `task.cancel()` from the steerer.

## §4 Channel-routed atomic cancel-and-restart (Phase 2 of #246)

The audit catalog above covers `CancelledError` propagation through
`await` boundaries. Phase 2 of [#246](https://github.com/pedapudi/goldfive/issues/246)
adds a complementary contract for **goldfive-authored cancel-and-
restart**: when the steerer detects drift mid-invocation and decides
to redirect the LLM, the cancel + restart must be atomic from the
operator's point of view. Pre-Phase-2 the steerer wrote
`session.pending_corrective_message` and let the overlay loop pick
it up at the next invocation boundary — but the in-flight LLM kept
running for the duration, generating contaminated output that
triggered more drift.

### §4.1 The two new ControlMessage kinds

* **`ControlKind.GOLDFIVE_STEER`** — minted by
  `DefaultSteerer._dispatch_goldfive_steer_control` from the
  CANCEL_REINVOKE branch of `_handle_drift` and from
  `_promote_drift_to_steer`. Payload:
  ```python
  {
      "drift_kind": str,            # the originating drift's kind value
      "drift_id": str,              # for dedupe / tracing
      "body": str,                  # corrective text to wrap in the
                                    # [GOLDFIVE STEERING CONTROL …] header
      "superseded_task_ids": [str], # tasks the LLM should NOT resume
      "replacement_task_ids": [str],# tasks that supersede the above
  }
  ```
  Atomic semantics: by the time this message lands on the channel,
  the steerer has ALREADY swapped `session.plan` (via
  `_apply_revision`) and run `_cancel_inflight_for_revision`. The
  channel message is the executor-side signal that the swap happened
  and a goldfive-authored corrective should now be injected as the
  new user message on the next passthrough invocation.

* **`ControlKind.GOLDFIVE_PAUSE_ESCALATE`** — minted by
  `DefaultSteerer._dispatch_goldfive_pause_control` from
  `_dispatch_pause_escalate`,
  `_emit_progress_stalled_escalation`, and
  `_emit_handler_exhausted_escalation`. Payload:
  ```python
  {
      "drift_kind": str,
      "drift_id": str,
      "reason": str,                # human-readable explanation
  }
  ```
  Atomic semantics: cancel any in-flight invoke task and park the
  run in the same blocking pre-task wait that a user-issued PAUSE
  uses. The originating drift event on the sink stream
  (`HUMAN_INTERVENTION_REQUIRED`) is the durable signal sinks /
  observers see; the control message is the channel-side signal
  that drives the executor.

### §4.2 Executor handling

The executor's `_invoke_passthrough_with_control` polls the channel
concurrently with `adapter.invoke_passthrough` (`asyncio.wait` on
both the invoke task and `channel.receive()`). When a
`GOLDFIVE_STEER` arrives:

1. The dispatch helper returns `goldfive_steer_message=msg` on the
   `ControlOutcome` (see `goldfive/executors/_control.py`).
2. The invoke loop calls `_cancel_invoke_task(invoke_task)` — same
   helper as the user-`STEER` branch.
3. It returns `("goldfive_steer", msg)` to the overlay loop.
4. The overlay loop's `goldfive_steer` branch composes the framed
   restart via
   `SequentialExecutor._compose_steer_restart_message(msg, source="goldfive", ...)`,
   resets the reconciler against the freshly-installed plan, pins
   `ReentryKind.GOLDFIVE_STEER_REPLAY`, and re-invokes the
   passthrough with the framed body as the new user input.

When a `GOLDFIVE_PAUSE_ESCALATE` arrives:

1. The dispatch helper returns
   `goldfive_pause_message=msg, request_pause=True`.
2. The invoke loop cancels the in-flight invoke and returns
   `("goldfive_pause", msg)`.
3. The overlay loop drains background steerer tasks and returns an
   `ExecutionOutcome(success=True, ...)` with a
   `goldfive_pause_escalate:<reason>` reason string. The pre-task
   loop on the next run cycle sees `request_pause` from the channel
   drain and blocks waiting for an operator `RESUME` / `STEER` /
   `CANCEL`.

### §4.3 Why this is part of the cancellation contract

The atomic cancel-and-restart these new kinds drive *uses* the
`CancelledError` propagation paths the rest of this document
catalogs. The `_cancel_invoke_task(invoke_task)` call is the
boundary where `CancelledError` lands inside the adapter's
streaming loop; the audit invariants in §1-§3 must hold for both
user-`STEER` cancels and the new goldfive-internal cancels — and
they do, because the boundary catch site at
`ADKAdapter._invoke_internal` is unchanged.

The deleted `Session.pending_corrective_message` and
`Session.paused_for_human_intervention` fields are gone (see
`goldfive/types.py`). A future re-introduction is caught by
`tests/test_goldfive_drift_routing.py::test_deleted_fields_have_no_residue_in_goldfive_package`.
