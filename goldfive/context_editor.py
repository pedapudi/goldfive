"""Request-side context editing as a steering capability (goldfive#397).

This module is the single, centralised gate goldfive uses to edit the
``contents`` list of an ADK :class:`LlmRequest` before it reaches the
model. Today goldfive's steering ladder treats the LLM transcript as
immutable — Plan revisions, synthetic ``USER_STEER`` injection, cancel
+ reinvoke, ``system_instruction`` prefixes, and ``max_output_tokens``
ratcheting are the only levers. Nothing prunes, redacts, or rewrites
``contents``. See ``docs/design/CONTEXT-EDITING.md`` for the motivation
and the catalogue of rules planned across follow-up PRs.

Architecture
------------

The :class:`ContextEditor` is held by the goldfive ADK plugin and
invoked from ``before_model_callback`` AFTER ``PromptShaper`` (today:
after :func:`_inject_goldfive_planner_instruction` and
:func:`_inject_runtime_tools_hint` in
``goldfive/adapters/_adk_plugin.py``) and BEFORE the model dispatch.
It walks a list of registered :class:`ContextEditRule` instances in
registration order, each receiving the output of the prior, applies
five mandatory invariants on the result, and either keeps the edit or
reverts to the original ``contents``.

Mandatory invariants (Phase 1, all enforced)
--------------------------------------------

1. **`observation_only` gate.** When the steerer's ``observation_only``
   flag is set (the default in production — goldfive#254 / goldfive#271
   strict-passive pattern), the entire pipeline is bypassed. No edit
   fires; ``contents`` is untouched. This mirrors the strict-passive
   discipline established by goldfive#271: any editorial surface that
   could mutate runtime state MUST be a complete no-op in passive
   mode.

2. **`tool_call_id` pairing.** After all rules have run, every
   ``function_call`` part still in ``contents`` must have its matching
   ``function_response`` (and vice versa). ADK errors hard on orphans
   (see :func:`goldfive.adapters.adk._heal_pending_tool_calls` for the
   primary-path equivalent on cancel). On violation, the editor
   reverts to the original ``contents`` and emits a
   ``ContextEditRejected`` event.

3. **Drop-only / no injection.** Rules can prune or rewrite-to-shorter
   existing parts; they cannot add material that wasn't in the prior
   ``contents``. Additive shaping stays in ``PromptShaper``'s lane
   (the ``system_instruction`` injections), where it is auditable.
   Enforced via the byte / count monotonicity gate: the post-edit
   byte total and content-count MUST be ≤ the pre-edit totals. A
   rule that violates this triggers a revert.

4. **Idempotence per revision.** A deterministic rule applied twice
   to the same ``(observed_revision_index, contents)`` MUST produce
   the same output. The editor stamps ``observed_revision_index`` on
   the emitted ``ContextEdited`` event so a downstream divergence
   between two emits at the same revision surfaces in the harmonograf
   timeline. Not assertable from inside one ``apply`` call — this
   invariant is contractual on rule authors and is validated by the
   unit tests.

5. **Log-out-of-loop.** Edits are append-only against the *persisted*
   transcript — goldfive's sinks see the original event stream via
   the plugin's other callbacks; only the model's request is edited.
   The editor emits a ``ContextEdited`` event so harmonograf can show
   what the model didn't see (rule name, byte delta, content-count
   delta, ``observed_revision_index``).

Failure modes
-------------

* **Rule strips half a `function_call` / `function_response` pair** —
  pairing-invariant check reverts the edit; ``ContextEditRejected``
  emitted with ``reason='tool_call_id_pair_violation'``.
* **Rule produces empty `contents`** — ADK requires at least one
  user turn. Revert; ``ContextEditRejected`` with
  ``reason='empty_contents'``.
* **Rule grows ``contents`` (count or bytes)** — violates drop-only
  invariant. Revert; ``ContextEditRejected`` with
  ``reason='not_drop_only'``.
* **Rule raises** — best-effort: log + skip the rule, keep the
  pre-rule ``contents``. Subsequent rules run normally. Editor never
  raises into the callback path.

State-ownership audit
---------------------

The state-audit module (:mod:`goldfive._state_audit`) currently patches
:func:`goldfive.adapters._adk_state_protocol._set` for ADK
``session.state`` writes. The request-side ``llm_request.contents``
list is NOT under that audit's remit today; this Phase-1 PR catalogs
:meth:`ContextEditor.apply` as the **sole** authorised mutation site
for ``contents`` by convention (no runtime tripwire — extension of the
audit to list-level mutation is left for a follow-up). The convention
holds because ``contents`` is read by ADK's flow only after every
``before_model_callback`` returns, and every other goldfive code path
that touches it (``_measure_request_chars``) is strictly read-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("goldfive.context_editor")


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ContextEditRule(Protocol):
    """A single rule that can drop / rewrite-to-shorter parts of ``contents``.

    Rules are registered with :meth:`ContextEditor.register` and walked
    in registration order. Each rule's :meth:`edit` receives the
    ``contents`` list output by the prior rule (or the original
    ``contents`` for the first rule) and either returns a new list (the
    edit) or ``None`` (no change at this revision — skip).

    Naming
    ------
    ``name`` is the stable string identifier used in
    :class:`~goldfive.config.SteeringConfig.context_editor_rules` to
    opt in. Use snake_case; should match the class's identifier in the
    rule catalog (e.g. ``"prune_cancelled_reasoning"``).

    Determinism contract
    --------------------
    Rules MUST be deterministic for a given ``(observed_revision_index,
    contents)`` input pair. The editor relies on idempotence to
    guarantee that two callbacks at the same revision produce the same
    edit. Rules that need stateful information (e.g. cancelled
    function_call ids) read it off ``ctx.session.state`` via the
    relevant ``state_store`` helper — that state is
    monotonically growing and revision-stamped, so deterministic per
    call.

    Drop-only contract
    ------------------
    A rule's output ``contents`` MUST have ≤ the input's content count
    AND ≤ the input's byte total. Rules may shorten ``part.text`` /
    ``part.function_response.response`` payloads but MUST NOT introduce
    new ``Content`` entries or grow any existing one. Adding belongs in
    PromptShaper.
    """

    name: str

    def edit(
        self,
        contents: list[Any],
        ctx: ContextEditContext,
    ) -> list[Any] | None:
        """Return a new ``contents`` list, or ``None`` for no change.

        ``contents`` is the live list off ``llm_request.contents``;
        rules MUST NOT mutate it in place — return a NEW list when the
        rule applies, or ``None`` when it doesn't. The editor swaps
        the request's ``contents`` reference only after invariants pass.
        """
        ...


# ---------------------------------------------------------------------------
# ContextEditContext — read-only per-call context handed to rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextEditContext:
    """Read-only context handed to every rule's :meth:`edit` call.

    Carries the goldfive :class:`~goldfive.types.Session`, the host
    agent name (the agent owning the wrapped ADK runner), and the
    snapshot ``observed_revision_index`` captured at the top of
    :meth:`ContextEditor.apply`.

    Rules use ``session`` to read orchestration state
    (e.g. ``state_store.read_cancelled_function_call_ids``) and
    ``observed_revision_index`` to stamp idempotence keys. The session
    handle is the goldfive Session (NOT the ADK session) — same handle
    every other steering surface reads from.
    """

    session: Any
    host_agent_name: str
    observed_revision_index: int


# ---------------------------------------------------------------------------
# Sentinel for "rule made no change" return
# ---------------------------------------------------------------------------


_REJECTED_REASONS = (
    "tool_call_id_pair_violation",
    "empty_contents",
    "not_drop_only",
    "rule_raised",
    "unknown_rule",
)


# ---------------------------------------------------------------------------
# Helpers — contents introspection
# ---------------------------------------------------------------------------


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that never raises (mirrors the plugin module helper).

    Local copy so this module has no import-time dependency on the ADK
    plugin module (which is large and itself imports ``google.adk``).
    """
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


def _content_bytes(contents: list[Any]) -> int:
    """Sum the serialised byte cost of every part in ``contents``.

    Mirrors :func:`goldfive.adapters._adk_plugin._measure_request_chars`
    semantics (text + name + json(args) for function_call; name +
    json(response) for function_response) so the byte delta on the
    ``ContextEdited`` event is directly comparable with the
    ``goldfive.llm.request`` instrumentation line. Returns 0 on any
    failure — measurement must never raise.
    """
    try:
        total = 0
        for content in contents:
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                text = _safe_attr(part, "text", "") or ""
                if text:
                    total += len(str(text))
                    continue
                fc = _safe_attr(part, "function_call", None)
                if fc is not None:
                    name = str(_safe_attr(fc, "name", "") or "")
                    args = _safe_attr(fc, "args", None)
                    total += len(name)
                    if args is not None:
                        try:
                            total += len(json.dumps(args, default=repr))
                        except Exception:  # noqa: BLE001
                            total += len(repr(args))
                    continue
                fr = _safe_attr(part, "function_response", None)
                if fr is not None:
                    name = str(_safe_attr(fr, "name", "") or "")
                    resp = _safe_attr(fr, "response", None)
                    total += len(name)
                    if resp is not None:
                        try:
                            total += len(json.dumps(resp, default=repr))
                        except Exception:  # noqa: BLE001
                            total += len(repr(resp))
                    continue
        return total
    except Exception:  # noqa: BLE001
        return 0


def _function_call_ids(contents: list[Any]) -> set[str]:
    """Return the set of ``function_call.id`` values present in ``contents``."""
    ids: set[str] = set()
    try:
        for content in contents:
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                fc = _safe_attr(part, "function_call", None)
                if fc is None:
                    continue
                fc_id = _safe_attr(fc, "id", "")
                if fc_id:
                    ids.add(str(fc_id))
    except Exception:  # noqa: BLE001
        pass
    return ids


def _function_response_ids(contents: list[Any]) -> set[str]:
    """Return the set of ``function_response.id`` values present in ``contents``."""
    ids: set[str] = set()
    try:
        for content in contents:
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                fr = _safe_attr(part, "function_response", None)
                if fr is None:
                    continue
                fr_id = _safe_attr(fr, "id", "")
                if fr_id:
                    ids.add(str(fr_id))
    except Exception:  # noqa: BLE001
        pass
    return ids


def _is_tool_call_id_paired(contents: list[Any]) -> bool:
    """Return True iff every ``function_call`` has a matching ``function_response``.

    The invariant ADK relies on: a turn whose ``function_call`` lacks a
    matching ``function_response`` (or vice versa) hits ADK's "Missing
    tool results" error during request assembly. The pairing is
    symmetric — both directions must hold.

    The empty-set case (no function_call / function_response parts at
    all — a fully-text turn) is paired by definition.
    """
    fc_ids = _function_call_ids(contents)
    fr_ids = _function_response_ids(contents)
    return fc_ids == fr_ids


def _contents_hash(contents: list[Any]) -> str:
    """Stable short hash of ``contents`` for idempotence-key stamping.

    Hashes the same JSON-serialised view that ``_content_bytes`` uses
    so two ``contents`` with identical observable shape hash to the
    same value. Best-effort — returns ``""`` on any failure.
    """
    try:
        records: list[Any] = []
        for content in contents:
            role = _safe_attr(content, "role", "") or ""
            parts_view: list[Any] = []
            parts = _safe_attr(content, "parts", None) or []
            for part in parts:
                fc = _safe_attr(part, "function_call", None)
                fr = _safe_attr(part, "function_response", None)
                if fc is not None:
                    parts_view.append(
                        {
                            "kind": "fc",
                            "id": str(_safe_attr(fc, "id", "") or ""),
                            "name": str(_safe_attr(fc, "name", "") or ""),
                        }
                    )
                elif fr is not None:
                    parts_view.append(
                        {
                            "kind": "fr",
                            "id": str(_safe_attr(fr, "id", "") or ""),
                            "name": str(_safe_attr(fr, "name", "") or ""),
                        }
                    )
                else:
                    text = str(_safe_attr(part, "text", "") or "")
                    parts_view.append({"kind": "text", "len": len(text)})
            records.append({"role": role, "parts": parts_view})
        payload = json.dumps(records, sort_keys=True, default=repr)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# ContextEditor
# ---------------------------------------------------------------------------


@dataclass
class _EditResult:
    """Bookkeeping for one ``apply()`` invocation.

    Exposed so the unit tests can assert against the per-rule decision
    chain without grepping log lines.
    """

    applied_rules: list[str] = field(default_factory=list)
    rejected_rules: list[tuple[str, str]] = field(
        default_factory=list
    )  # (rule_name, reason)
    skipped_rules: list[str] = field(default_factory=list)  # rule returned None
    bytes_before: int = 0
    bytes_after: int = 0
    contents_count_before: int = 0
    contents_count_after: int = 0


class ContextEditor:
    """Walks registered rules over ``llm_request.contents`` under invariants.

    Constructed lazily by the ADK plugin only when
    :class:`~goldfive.config.SteeringConfig.context_editor_rules`
    resolves to a non-empty list of recognised rule names. The plugin
    does NOT instantiate the editor when the rule list is empty — the
    feature-flag-off path has zero overhead (one ``is None`` check in
    the callback).

    Rules
    -----
    Register with :meth:`register` at construction time
    (typically inside :func:`make_adk_plugin`'s factory). Rules walk
    in registration order; the editor passes the prior rule's output
    forward so two rules can compose (e.g. prune cancelled pairs THEN
    prune transient errors over the survivors).

    Sinks
    -----
    ``sinks`` is the same fan-out list goldfive's other ``make_event``
    emit sites use (steerer, planner, pin_resolved). The editor emits
    ``ContextEdited`` on success and ``ContextEditRejected`` on
    invariant violation. When ``sinks`` is empty the editor still
    applies edits (the work is real) but skips emission silently —
    same shape as the steerer's emit paths.

    Reverts
    -------
    Invariant violations revert to the pre-rule ``contents`` and
    continue with the next rule. The chain is NOT aborted on a single
    rejection — a subsequent rule can still apply cleanly to the
    pre-rule state.
    """

    def __init__(
        self,
        *,
        rules: list[ContextEditRule] | None = None,
        sinks: list[Any] | None = None,
    ) -> None:
        self._rules: list[ContextEditRule] = list(rules or [])
        self._sinks: list[Any] = list(sinks or [])

    def register(self, rule: ContextEditRule) -> None:
        """Append ``rule`` to the rule chain.

        Idempotent on rule identity (the same instance registered
        twice is held once) but NOT on rule name — two distinct
        instances with the same ``name`` are both kept, which is
        intentional: a future rule taxonomy may want parameterised
        instances of the same rule type registered with different
        configs.
        """
        if rule in self._rules:
            return
        self._rules.append(rule)

    @property
    def rules(self) -> tuple[ContextEditRule, ...]:
        """Immutable snapshot of the registered rule chain (for tests / debug)."""
        return tuple(self._rules)

    async def apply(
        self,
        llm_request: Any,
        *,
        session: Any,
        host_agent_name: str,
        observation_only: bool,
    ) -> _EditResult:
        """Apply the rule chain to ``llm_request.contents`` under invariants.

        Returns an :class:`_EditResult` recording which rules applied,
        which were rejected (and why), and the pre/post byte +
        content-count totals. The caller (the ADK plugin's
        ``before_model_callback``) discards the return value in
        production; it exists for tests and operator introspection.

        ``observation_only`` is a hard gate (Invariant 1). When True
        the entire pipeline is bypassed and the result reports the
        pre-edit totals as both pre and post.

        Never raises into the caller — every rule and every emit is
        wrapped in a defensive ``try/except``. Callback paths in the
        ADK plugin are best-effort by contract; an editor crash MUST
        NOT take down the LLM dispatch.
        """
        result = _EditResult()

        # Invariant 1 — observation_only gate. Complete no-op.
        if observation_only:
            log.debug(
                "ContextEditor.apply: observation_only=True — pipeline skipped"
            )
            return result

        if not self._rules:
            return result

        contents = _safe_attr(llm_request, "contents", None) or []
        # The list is the live mutable reference ADK's flow will read;
        # we never mutate it in place — only swap the attribute when
        # an edit passes invariants.
        original_contents = list(contents)
        bytes_before = _content_bytes(original_contents)
        result.bytes_before = bytes_before
        result.contents_count_before = len(original_contents)

        # Capture the observed_revision_index ONCE at the top of the
        # call so every emitted event for this call shares the same
        # revision stamp — even if a concurrent revision bump lands
        # mid-chain. Idempotence-per-revision (Invariant 4) depends on
        # this stamping site being upstream of any rule that might
        # await.
        observed_rev = _resolve_observed_revision_index(session)
        edit_ctx = ContextEditContext(
            session=session,
            host_agent_name=host_agent_name,
            observed_revision_index=observed_rev,
        )

        current = original_contents
        for rule in self._rules:
            rule_name = str(_safe_attr(rule, "name", "") or rule.__class__.__name__)

            # Defensive: any rule that raises is logged + skipped.
            # We do NOT abort the chain — a subsequent rule can still
            # apply cleanly.
            try:
                candidate = rule.edit(list(current), edit_ctx)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "ContextEditor.apply: rule %r raised — skipping: %s",
                    rule_name,
                    exc,
                )
                result.rejected_rules.append((rule_name, "rule_raised"))
                await self._emit_rejected(
                    session=session,
                    rule_name=rule_name,
                    reason="rule_raised",
                    observed_rev=observed_rev,
                )
                continue

            if candidate is None or candidate == current:
                # Rule chose not to edit this revision. Common path —
                # most rules don't apply on most turns.
                result.skipped_rules.append(rule_name)
                continue

            # Invariant 3 — drop-only / no-injection. Byte total and
            # content count must be ≤ pre-rule.
            cand_bytes = _content_bytes(candidate)
            pre_rule_bytes = _content_bytes(current)
            if cand_bytes > pre_rule_bytes or len(candidate) > len(current):
                log.warning(
                    "ContextEditor.apply: rule %r violated drop-only "
                    "(bytes %d->%d, count %d->%d) — reverting",
                    rule_name,
                    pre_rule_bytes,
                    cand_bytes,
                    len(current),
                    len(candidate),
                )
                result.rejected_rules.append((rule_name, "not_drop_only"))
                await self._emit_rejected(
                    session=session,
                    rule_name=rule_name,
                    reason="not_drop_only",
                    observed_rev=observed_rev,
                )
                continue

            # Invariant 2 — tool_call_id pairing.
            if not _is_tool_call_id_paired(candidate):
                log.warning(
                    "ContextEditor.apply: rule %r broke tool_call_id "
                    "pairing — reverting",
                    rule_name,
                )
                result.rejected_rules.append((rule_name, "tool_call_id_pair_violation"))
                await self._emit_rejected(
                    session=session,
                    rule_name=rule_name,
                    reason="tool_call_id_pair_violation",
                    observed_rev=observed_rev,
                )
                continue

            # Pairing-invariant addendum: empty contents is also a
            # revert — ADK requires at least one user turn.
            if not candidate:
                log.warning(
                    "ContextEditor.apply: rule %r produced empty contents "
                    "— reverting",
                    rule_name,
                )
                result.rejected_rules.append((rule_name, "empty_contents"))
                await self._emit_rejected(
                    session=session,
                    rule_name=rule_name,
                    reason="empty_contents",
                    observed_rev=observed_rev,
                )
                continue

            # Edit passed all invariants. Accept; subsequent rules
            # see the edited contents.
            current = candidate
            result.applied_rules.append(rule_name)
            try:
                await self._emit_applied(
                    session=session,
                    rule_name=rule_name,
                    bytes_before=pre_rule_bytes,
                    bytes_after=cand_bytes,
                    contents_count_before=len(original_contents),
                    contents_count_after=len(candidate),
                    observed_rev=observed_rev,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "ContextEditor.apply: emit ContextEdited raised: %s",
                    exc,
                )

        # Commit the final ``contents`` onto the request only if a
        # rule actually applied. The defensive ``current is
        # original_contents`` check avoids touching the attribute when
        # nothing changed.
        if current is not original_contents and result.applied_rules:
            try:
                llm_request.contents = current
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "ContextEditor.apply: could not swap llm_request.contents "
                    "— reverting: %s",
                    exc,
                )
                result.applied_rules.clear()
                current = original_contents

        result.bytes_after = _content_bytes(current)
        result.contents_count_after = len(current)
        return result

    # ---- Internal: event emission --------------------------------------

    async def _emit_applied(
        self,
        *,
        session: Any,
        rule_name: str,
        bytes_before: int,
        bytes_after: int,
        contents_count_before: int,
        contents_count_after: int,
        observed_rev: int,
    ) -> None:
        """Emit a ``ContextEdited`` event onto every registered sink.

        Uses :func:`goldfive.events.make_event` (dict envelope) because
        the proto schema doesn't yet carry a ``ContextEdited`` slot —
        same pattern as ``pin_resolved`` in the ADK plugin (see
        ``_emit_pin_resolved``). Best-effort: every failure is logged
        and swallowed.
        """
        if not self._sinks:
            return
        run_id = str(_safe_attr(session, "run_id", "") or "")
        session_id = str(_safe_attr(session, "id", "") or "") or run_id
        try:
            seq = session.next_sequence()
        except Exception:  # noqa: BLE001
            seq = 0
        payload: dict[str, Any] = {
            "rule_name": str(rule_name),
            "bytes_before": int(bytes_before),
            "bytes_after": int(bytes_after),
            "contents_count_before": int(contents_count_before),
            "contents_count_after": int(contents_count_after),
            "observed_revision_index": int(observed_rev),
        }
        try:
            from goldfive.events import emit, make_event  # noqa: PLC0415

            evt = make_event(run_id, seq, "context_edited", payload, session_id=session_id)
            await emit(self._sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug("ContextEditor._emit_applied: failed: %s", exc)

    async def _emit_rejected(
        self,
        *,
        session: Any,
        rule_name: str,
        reason: str,
        observed_rev: int,
    ) -> None:
        """Emit a ``ContextEditRejected`` event when an invariant tripped.

        Carries the rule name + reason (one of :data:`_REJECTED_REASONS`)
        + the ``observed_revision_index``. Operators reading the
        harmonograf timeline see exactly which rule was reverted and
        why.
        """
        if not self._sinks:
            return
        run_id = str(_safe_attr(session, "run_id", "") or "")
        session_id = str(_safe_attr(session, "id", "") or "") or run_id
        try:
            seq = session.next_sequence()
        except Exception:  # noqa: BLE001
            seq = 0
        payload: dict[str, Any] = {
            "rule_name": str(rule_name),
            "reason": str(reason),
            "observed_revision_index": int(observed_rev),
        }
        try:
            from goldfive.events import emit, make_event  # noqa: PLC0415

            evt = make_event(
                run_id, seq, "context_edit_rejected", payload, session_id=session_id
            )
            await emit(self._sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug("ContextEditor._emit_rejected: failed: %s", exc)


def _resolve_observed_revision_index(session: Any) -> int:
    """Return the goldfive session's current ``plan.revision_index``.

    Stamped onto every emitted event so harmonograf can correlate a
    ``ContextEdited`` against the surrounding plan revision. Returns
    ``0`` when the session has no plan yet (cold-session edge case).
    Never raises.
    """
    try:
        plan = _safe_attr(session, "plan", None)
        if plan is None:
            return 0
        return int(_safe_attr(plan, "revision_index", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Initial rule — PruneCancelledReasoningRule
# ---------------------------------------------------------------------------


class PruneCancelledReasoningRule:
    """Strip ``function_call`` / ``function_response`` pairs from cancelled invocations.

    Companion to goldfive#230 — that issue silenced *judges* on cancelled
    reasoning; this rule silences the *model itself* by removing the
    cancelled pairs from the LLM's view on the NEXT turn after a cancel.

    Identification
    --------------
    The rule reads the goldfive-owned list at
    ``goldfive.cancelled_function_call_ids`` on ``session.state``
    (written by :meth:`ADKAdapter._heal_pending_tool_calls` after every
    cancel — see
    :func:`goldfive.state_store.append_cancelled_function_call_ids`).
    Every ``function_call`` part whose ``id`` appears in that list is
    eligible for pruning, alongside its paired ``function_response``.
    The list is append-only and de-duplicated, so the rule's decision
    is deterministic for a given ``session.state`` snapshot.

    Pairing safety
    --------------
    The rule prunes every Content whose parts touch a cancelled
    ``function_call_id`` — whether the cancelled half is a
    ``function_call`` or its paired ``function_response``. Since calls
    and responses share the same id and are usually packaged together,
    a healthy pair drops as a unit and the editor's pairing invariant
    (Invariant 2) holds trivially. If pre-rule ``contents`` already
    contained an orphan half (e.g. cancel-mid-tool already dropped one
    side), this rule drops the surviving orphan too — which fixes
    rather than introduces a pairing violation. The editor's pairing
    invariant catches any residual asymmetry as a safety net.

    Empty-contents safety
    ---------------------
    If pruning would leave ``contents`` empty (every entry was a
    cancelled pair) the rule returns the original list unchanged. The
    editor's empty-contents invariant would catch this anyway, but the
    rule prefers to short-circuit rather than rely on a downstream
    revert.

    Whole-content-vs-part scoping
    -----------------------------
    A ``Content`` whose ``parts`` mix a cancelled ``function_call``
    with non-cancelled text is rare but possible. The rule drops the
    *whole Content* when ANY of its parts is a cancelled function_call
    or function_response — the text alongside such a call is the
    model's reasoning leading up to the cancelled action, which is
    exactly the material we're trying to strip. Conservative: this
    behaviour is what motivates the rule's existence.
    """

    name = "prune_cancelled_reasoning"

    def edit(
        self,
        contents: list[Any],
        ctx: ContextEditContext,
    ) -> list[Any] | None:
        # Read cancelled ids off goldfive Session state. Best-effort:
        # any failure (missing key, malformed state) returns the
        # rule-as-skipped signal.
        cancelled_ids = _read_cancelled_ids(ctx.session)
        if not cancelled_ids:
            return None

        # Walk contents, dropping every Content that touches a
        # cancelled function_call id.
        survivors: list[Any] = []
        any_dropped = False
        for content in contents:
            if _content_touches_cancelled_id(content, cancelled_ids):
                any_dropped = True
                continue
            survivors.append(content)

        if not any_dropped:
            return None

        # Empty-contents safety — don't ship an empty list; the editor
        # would revert anyway, but skipping is cheaper.
        if not survivors:
            return None

        return survivors


def _read_cancelled_ids(session: Any) -> set[str]:
    """Read ``goldfive.cancelled_function_call_ids`` off the session as a set.

    Goes through :func:`goldfive.state_store.read_cancelled_function_call_ids`
    (the canonical reader) so the rule stays in sync with the writer's
    schema. Returns an empty set on any failure.
    """
    try:
        from goldfive import state_store as _ostate  # noqa: PLC0415

        state = _safe_attr(session, "state", None)
        if state is None:
            return set()
        return set(_ostate.read_cancelled_function_call_ids(state))
    except Exception:  # noqa: BLE001
        return set()


def _content_touches_cancelled_id(content: Any, cancelled_ids: set[str]) -> bool:
    """Return True iff ``content`` has any part with a cancelled function_call_id."""
    parts = _safe_attr(content, "parts", None) or []
    for part in parts:
        fc = _safe_attr(part, "function_call", None)
        if fc is not None:
            fc_id = str(_safe_attr(fc, "id", "") or "")
            if fc_id and fc_id in cancelled_ids:
                return True
        fr = _safe_attr(part, "function_response", None)
        if fr is not None:
            fr_id = str(_safe_attr(fr, "id", "") or "")
            if fr_id and fr_id in cancelled_ids:
                return True
    return False


# ---------------------------------------------------------------------------
# Rule registry — name -> factory
# ---------------------------------------------------------------------------


#: Map of opt-in rule name (the string used in
#: :class:`~goldfive.config.SteeringConfig.context_editor_rules`) to a
#: zero-arg factory that builds the rule instance. New rules add a row
#: here AND under the catalog in ``docs/design/CONTEXT-EDITING.md``.
_RULE_REGISTRY: dict[str, Any] = {
    "prune_cancelled_reasoning": PruneCancelledReasoningRule,
}


def build_editor_from_config(
    rule_names: list[str] | None,
    sinks: list[Any] | None = None,
) -> ContextEditor | None:
    """Construct a :class:`ContextEditor` from a list of rule names.

    Returns ``None`` when ``rule_names`` is ``None``, empty, or
    contains only unknown names — the caller (the ADK plugin) treats
    ``None`` as "feature disabled, codepath bypassed". Unknown rule
    names are logged at WARNING and dropped; known names are
    instantiated in the order given so registration order honours the
    operator's config.

    ``sinks`` is forwarded onto the editor for telemetry emission.
    """
    if not rule_names:
        return None
    rules: list[ContextEditRule] = []
    seen_names: set[str] = set()
    for raw_name in rule_names:
        name = str(raw_name).strip().lower()
        if not name or name in seen_names:
            continue
        factory = _RULE_REGISTRY.get(name)
        if factory is None:
            log.warning(
                "build_editor_from_config: unknown context_editor_rule %r "
                "— ignoring. Known: %s",
                name,
                sorted(_RULE_REGISTRY.keys()),
            )
            continue
        try:
            rule = factory()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "build_editor_from_config: rule %r factory raised: %s",
                name,
                exc,
            )
            continue
        rules.append(rule)
        seen_names.add(name)
    if not rules:
        return None
    return ContextEditor(rules=rules, sinks=list(sinks or []))


__all__ = [
    "ContextEditContext",
    "ContextEditRule",
    "ContextEditor",
    "PruneCancelledReasoningRule",
    "build_editor_from_config",
]
