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

3. **Byte/count monotonicity + rule-class scoping.** The post-edit
   byte total AND content-count MUST be ≤ the pre-rule totals — the
   structural gate that keeps every rule subtractive in aggregate. A
   rule that grows either triggers a revert
   (``reason='not_drop_only'``). On top of that gate, each rule
   declares a ``rule_class`` (PR 6b):

   * ``"drop_only"`` (the Phase-1 default for rules that don't declare
     one) — a rule may ONLY remove whole ``Content`` entries. Every
     entry in its output must be identity-present in the input;
     synthesizing or modifying an entry is reverted
     (``reason='injected_content'``). ``PruneCancelledReasoningRule``
     and ``PruneStaleSteerRule`` are drop-only.
   * ``"byte_monotonic_replace"`` (the PR 6b relaxation) — a rule may
     REWRITE an entry in place (redact a transient-error
     ``function_response`` payload) or REPLACE a run of entries with a
     single shorter summarized entry. Synthesized text is permitted
     *only* for this class and *only* under the byte/count gate above,
     so the edit stays subtractive in aggregate.
     ``PruneTransientErrorRule`` and ``CompactPriorReasoningRule`` are
     replace-class.

   Free-form additive shaping (injecting brand-new guidance) stays in
   ``PromptShaper``'s lane (the ``system_instruction`` injections),
   where it is auditable. The editor never grows the transcript.

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

import copy
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

    Rule-class contract
    -------------------
    A rule's output ``contents`` MUST have ≤ the input's content count
    AND ≤ the input's byte total (the structural monotonicity gate).
    Beyond that, the rule's optional ``rule_class`` attribute scopes
    what shapes of edit are permitted:

    * ``"drop_only"`` (the default when the attribute is absent) — the
      rule may only DROP whole ``Content`` entries; every surviving
      entry must be one of the input objects. Synthesizing or modifying
      an entry is rejected (``reason='injected_content'``).
    * ``"byte_monotonic_replace"`` — the rule may additionally redact a
      payload in place or replace a run of entries with one shorter
      summary. Synthesized text is permitted *only* under the
      monotonicity gate above. Such a rule MUST build NEW ``Content`` /
      part objects (e.g. via :func:`copy.deepcopy`) rather than mutate
      the input objects, so a downstream invariant revert restores the
      original transcript byte-for-byte.

    Free-form additive shaping belongs in PromptShaper, never here.
    """

    name: str
    #: One of :data:`_RULE_CLASSES`. Optional — a rule without it is
    #: treated as ``"drop_only"`` (the Phase-1 contract).
    rule_class: str

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
    agent name (the agent owning the wrapped ADK runner), the snapshot
    ``observed_revision_index`` captured at the top of
    :meth:`ContextEditor.apply`, and the set of currently-tripped drift
    kinds.

    Rules use ``session`` to read orchestration state
    (e.g. ``state_store.read_cancelled_function_call_ids``) and
    ``observed_revision_index`` to stamp idempotence keys. The session
    handle is the goldfive Session (NOT the ADK session) — same handle
    every other steering surface reads from.

    Dormancy gate (AGENCY-PRESERVATION.md §0)
    ----------------------------------------
    ``active_drift_kinds`` is the set of :class:`~goldfive.types.DriftKind`
    *string values* for the conditions currently OPEN or ESCALATING on
    the session (``state_store.list_active_drifts`` — resolved /
    human-escalated conditions are popped, so this is the live "tripped"
    set). It is the editor's contribution to the dormancy discipline:
    context-editing rules are a steering surface, so they must stay
    dormant on healthy turns and fire ONLY on a tripped guardrail
    counter or a drift verdict. Every production rule self-gates on a
    non-healthy trigger it can observe directly (cancelled ids, a
    transient-error response in ``contents``, a stale goldfive note,
    ≥N identical failed tool calls); rules that want to additionally
    arm on a recorded drift *verdict* (e.g.
    :class:`CompactPriorReasoningRule` lowering its repeat threshold
    when a ``LOOPING_TOOL_CALL`` condition is open) read this set.
    Empty by default — a rule MUST behave as a no-op when the set is
    empty AND its own structural trigger is absent.
    """

    session: Any
    host_agent_name: str
    observed_revision_index: int
    active_drift_kinds: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Sentinel for "rule made no change" return
# ---------------------------------------------------------------------------


_REJECTED_REASONS = (
    "tool_call_id_pair_violation",
    "empty_contents",
    "not_drop_only",
    "injected_content",
    "rule_raised",
    "unknown_rule",
)

# Recognised rule classes (the ``rule_class`` attribute on a rule).
#
# * ``"drop_only"`` (the default for any rule that doesn't declare one)
#   — the Phase-1 contract: a rule may only REMOVE whole ``Content``
#   entries. Every entry in its output MUST be identity-present in the
#   input. Structurally enforced by :meth:`ContextEditor.apply` — a
#   drop-only rule that returns a synthesized / modified ``Content`` is
#   reverted with ``reason='injected_content'``.
# * ``"byte_monotonic_replace"`` (PR 6b relaxation, AGENCY-PRESERVATION.md
#   §"PR 6b") — a rule may REWRITE existing entries in place (redact a
#   ``function_response`` payload) or REPLACE a run of entries with one
#   shorter summarized entry. Synthesized text is permitted, but the
#   structural byte/count monotonicity gate (Invariant 3) still binds:
#   the post-edit byte total AND content count MUST be ≤ the pre-rule
#   totals. The identity check is skipped for this class.
_RULE_CLASS_DROP_ONLY = "drop_only"
_RULE_CLASS_BYTE_MONOTONIC_REPLACE = "byte_monotonic_replace"
_RULE_CLASSES = (_RULE_CLASS_DROP_ONLY, _RULE_CLASS_BYTE_MONOTONIC_REPLACE)


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

    Mirrors :func:`goldfive.adapters.adk_llm_instrumentation._measure_request_chars`
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
        # Dormancy gate (AGENCY-PRESERVATION.md §0): snapshot the set of
        # currently-tripped drift kinds so rules can arm on a recorded
        # verdict in addition to their own structural trigger. Captured
        # once, alongside the revision stamp, so the whole chain sees a
        # consistent view.
        active_drift_kinds = _resolve_active_drift_kinds(session)
        edit_ctx = ContextEditContext(
            session=session,
            host_agent_name=host_agent_name,
            observed_revision_index=observed_rev,
            active_drift_kinds=active_drift_kinds,
        )

        current = original_contents
        for rule in self._rules:
            rule_name = str(_safe_attr(rule, "name", "") or rule.__class__.__name__)
            rule_class = _resolve_rule_class(rule)

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

            # Invariant 3 (cont.) — rule-class scoping. A ``drop_only``
            # rule may only REMOVE whole entries: every surviving
            # ``Content`` must be one of the input objects (identity
            # match). A rule that synthesizes or modifies an entry while
            # declaring (or defaulting to) ``drop_only`` is reverted.
            # ``byte_monotonic_replace`` rules skip this check — they are
            # explicitly permitted to redact / summarize in place, bounded
            # only by the byte/count gate above.
            if rule_class == _RULE_CLASS_DROP_ONLY and not _is_identity_subset(
                candidate, current
            ):
                log.warning(
                    "ContextEditor.apply: drop-only rule %r returned a "
                    "synthesized/modified Content — reverting",
                    rule_name,
                )
                result.rejected_rules.append((rule_name, "injected_content"))
                await self._emit_rejected(
                    session=session,
                    rule_name=rule_name,
                    reason="injected_content",
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
                    rule_class=rule_class,
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
        rule_class: str,
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
            "rule_class": str(rule_class),
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


def _resolve_active_drift_kinds(session: Any) -> frozenset[str]:
    """Return the set of currently-tripped drift-kind string values.

    Reads ``state_store.list_active_drifts`` off the goldfive session —
    the conditions OPEN or ESCALATING right now (resolved /
    human-escalated entries are popped by the store, so the set is the
    live "tripped" view). Used by the dormancy gate so rules can arm on
    a recorded drift verdict in addition to their own structural
    trigger. Returns an empty frozenset on any failure or when no plan/
    state is present — never raises into the callback path.
    """
    try:
        from goldfive import state_store as _ostate  # noqa: PLC0415

        state = _safe_attr(session, "state", None)
        if state is None:
            return frozenset()
        kinds: set[str] = set()
        for drift in _ostate.list_active_drifts(state):
            kind = _safe_attr(drift, "kind", None)
            value = _safe_attr(kind, "value", None)
            kinds.add(str(value if value is not None else kind))
        kinds.discard("None")
        kinds.discard("")
        return frozenset(kinds)
    except Exception:  # noqa: BLE001
        return frozenset()


def _resolve_rule_class(rule: Any) -> str:
    """Return the rule's declared ``rule_class``, defaulting to drop-only.

    An unrecognised value is coerced to ``"drop_only"`` (the strictest
    contract) and logged — a typo in a rule's class MUST NOT silently
    grant it replace privileges.
    """
    raw = _safe_attr(rule, "rule_class", None)
    if raw is None:
        return _RULE_CLASS_DROP_ONLY
    value = str(raw).strip().lower()
    if value not in _RULE_CLASSES:
        log.warning(
            "ContextEditor: rule %r declares unknown rule_class %r — "
            "treating as %r",
            _safe_attr(rule, "name", rule.__class__.__name__),
            raw,
            _RULE_CLASS_DROP_ONLY,
        )
        return _RULE_CLASS_DROP_ONLY
    return value


def _is_identity_subset(candidate: list[Any], current: list[Any]) -> bool:
    """Return True iff every entry in ``candidate`` is identity-present in ``current``.

    The structural enforcement of the ``drop_only`` rule class: a
    drop-only rule may reorder / remove entries but every surviving
    entry must be one of the input objects (``is`` identity, not
    equality). A rule that builds a fresh / modified ``Content`` fails
    this check — that is the ``byte_monotonic_replace`` privilege it did
    not claim.
    """
    current_ids = {id(c) for c in current}
    return all(id(c) in current_ids for c in candidate)


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

    Dormancy
    --------
    The trigger IS a non-healthy condition: ``contents`` is only edited
    when ``goldfive.cancelled_function_call_ids`` is non-empty, i.e. a
    cancel + heal actually happened. On a healthy turn the list is empty
    and the rule returns ``None`` (skip). Drop-only: it removes whole
    ``Content`` entries only.
    """

    name = "prune_cancelled_reasoning"
    rule_class = _RULE_CLASS_DROP_ONLY

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
# Shared part / payload introspection helpers (rules 2-4)
# ---------------------------------------------------------------------------


def _iter_parts(content: Any) -> list[Any]:
    """Return ``content.parts`` or ``[]`` — never raises."""
    return _safe_attr(content, "parts", None) or []


def _content_text(content: Any) -> str:
    """Concatenate every ``part.text`` on ``content`` into one string."""
    chunks: list[str] = []
    for part in _iter_parts(content):
        text = _safe_attr(part, "text", "") or ""
        if text:
            chunks.append(str(text))
    return "\n".join(chunks)


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion for status-code fields. ``None`` on failure."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _lower_keyed(resp: dict[Any, Any]) -> dict[str, Any]:
    """Return ``resp`` with top-level keys lowercased + stripped."""
    out: dict[str, Any] = {}
    for k, v in resp.items():
        try:
            out[str(k).strip().lower()] = v
        except Exception:  # noqa: BLE001
            continue
    return out


#: Recognised status / error-code field names (lowercased) inspected on a
#: ``function_response.response`` dict. Structural — NOT a free-text scan.
_CODE_FIELDS: tuple[str, ...] = (
    "status_code",
    "statuscode",
    "http_status",
    "httpstatus",
    "error_code",
    "errorcode",
    "code",
    "status",
)

#: ``status`` string values that denote a failed call.
_ERROR_STATUS_WORDS: frozenset[str] = frozenset(
    {"error", "failed", "failure", "timeout", "unavailable", "exception"}
)

#: Error-indicator + error-text field names (lowercased) inspected on a
#: ``function_response.response`` dict.
_ERROR_TEXT_FIELDS: tuple[str, ...] = (
    "error",
    "exception",
    "message",
    "detail",
    "reason",
    "type",
    "status",
    "title",
)


def _response_has_error_indicator(resp: Any) -> bool:
    """Return True iff a ``function_response.response`` dict has an error SHAPE.

    Conservative + structural: an HTTP-ish code ≥ 400 in a recognised
    code field, a truthy ``error`` / ``exception`` field, an explicit
    ``success``/``ok`` of ``False``, or a ``status`` string in
    :data:`_ERROR_STATUS_WORDS`. Plain free text is NOT inspected — the
    rule only flags structured statuses, never "anything that looks like
    an error" (CONTEXT-EDITING.md rule-2/4 contract).
    """
    if not isinstance(resp, dict):
        return False
    lowered = _lower_keyed(resp)
    for code_field in _CODE_FIELDS:
        if code_field in lowered:
            code = _coerce_int(lowered[code_field])
            if code is not None and code >= 400:
                return True
    if lowered.get("error"):
        return True
    if lowered.get("exception"):
        return True
    if lowered.get("success") is False or lowered.get("ok") is False:
        return True
    status = lowered.get("status")
    if isinstance(status, str) and status.strip().lower() in _ERROR_STATUS_WORDS:
        return True
    return False


def _response_error_text(resp: Any) -> str:
    """Gather lowercased text from a response dict's error-shaped fields only.

    Used by :class:`PruneTransientErrorRule` to test transient markers
    against the error message — NOT against arbitrary data fields, so a
    benign payload that merely contains the word "timeout" never trips.
    """
    if not isinstance(resp, dict):
        return ""
    lowered = _lower_keyed(resp)
    chunks: list[str] = []
    for text_field in _ERROR_TEXT_FIELDS:
        v = lowered.get(text_field)
        if isinstance(v, str):
            chunks.append(v)
        elif isinstance(v, dict):
            for kk in ("message", "reason", "detail", "type", "status", "code", "description"):
                vv = v.get(kk)
                if vv is not None:
                    chunks.append(str(vv))
        elif v is not None and not isinstance(v, (list, tuple)):
            chunks.append(str(v))
    return " ".join(chunks).lower()


def _function_call_signature(fc: Any) -> tuple[str, str]:
    """Return ``(name, canonical-args-json)`` identifying a function call.

    Two calls with the same name and the same (order-insensitive) args
    share a signature — the "identical tool call" key
    :class:`CompactPriorReasoningRule` groups on. Best-effort: args that
    don't serialise fall back to ``repr``.
    """
    name = str(_safe_attr(fc, "name", "") or "")
    args = _safe_attr(fc, "args", None)
    if args is None:
        return (name, "")
    try:
        return (name, json.dumps(args, sort_keys=True, default=repr))
    except Exception:  # noqa: BLE001
        return (name, repr(args))


def _content_owned_by_id(content: Any, fc_id: str) -> bool:
    """Return True iff every fc/fr part on ``content`` belongs to ``fc_id``.

    The ownership guard for :class:`CompactPriorReasoningRule`: it only
    drops / rewrites a ``Content`` when that content's tool parts ALL
    belong to the call id being collapsed. A content batching parts for
    several ids is left untouched (conservative — dropping it would
    affect unrelated calls). Text parts are allowed (the reasoning text
    alongside the collapsed call is part of what we're compacting).
    """
    for part in _iter_parts(content):
        fc = _safe_attr(part, "function_call", None)
        if fc is not None:
            pid = str(_safe_attr(fc, "id", "") or "")
            if pid and pid != fc_id:
                return False
        fr = _safe_attr(part, "function_response", None)
        if fr is not None:
            pid = str(_safe_attr(fr, "id", "") or "")
            if pid and pid != fc_id:
                return False
    return True


# ---------------------------------------------------------------------------
# Rule — PruneTransientErrorRule (byte-monotonic replace)
# ---------------------------------------------------------------------------


class PruneTransientErrorRule:
    """Redact transient-error ``function_response`` payloads in place.

    A 429 / 5xx / timeout / network blip / parse failure that propagated
    through a ``function_response`` becomes a permanent fixture in the
    transcript: subsequent turns re-read it, waste tokens, and may bias
    their reasoning toward the failure (CONTEXT-EDITING.md "Motivation").
    This rule replaces the offending response payload with a tiny marker
    while leaving the ``function_call`` / ``function_response`` pair
    structurally intact.

    Why redact, not drop
    --------------------
    Dropping the ``function_response`` part alone would orphan its
    ``function_call`` and trip the editor's pairing invariant (a revert);
    dropping the whole pair would erase the fact that the tool was tried.
    Redaction keeps the pair (same ``id`` + ``name``) so pairing holds
    and the model still sees that it called the tool — only the noisy
    error body is elided. This is why the rule is
    ``byte_monotonic_replace``, not ``drop_only``.

    Detection (conservative, structural)
    ------------------------------------
    A response is transient when it is a ``dict`` AND either (a) a
    recognised code field (:data:`_CODE_FIELDS`) holds a status in the
    configured transient set, or (b) it has an explicit error shape
    (:func:`_response_has_error_indicator`) AND its error text matches a
    configured transient marker. Free-form data fields are never scanned
    — only flagged statuses, never "anything that looks like an error".

    NL-heuristics boundary (binding)
    --------------------------------
    The marker allowlist matches ONLY **machine-generated error
    signatures**: HTTP status reason phrases (RFC-standardised:
    "Too Many Requests", "Service Unavailable", …), SDK / runtime
    exception class names ("RateLimitError", "APITimeoutError",
    "JSONDecodeError", …), and structured error codes / types
    ("rate_limit_exceeded", "ECONNRESET", "overloaded_error", …). These
    are emitted by infrastructure, not authored by the agent. This rule
    MUST NOT be extended into semantic matching of natural-language prose
    — neither agent reasoning nor free-text tool output — which is the
    #166/#167 anti-pattern (retired ``_GENERIC_VERB_PREFIX_RE`` /
    ``_FACTUAL_QUESTION_RE``). The matched fields are restricted to the
    error-shaped keys of a response that already carries a structural
    error indicator, precisely to keep the boundary at "machine error
    payload", never "looks like an error in English".

    Dormancy
    --------
    The trigger is the transient-error response itself — a guardrail-
    observed fact (a failed tool result). A healthy turn carries no such
    response, so the rule returns ``None``.
    """

    name = "prune_transient_error"
    rule_class = _RULE_CLASS_BYTE_MONOTONIC_REPLACE

    #: Default transient HTTP-ish status codes. Configurable per instance.
    _DEFAULT_STATUS_CODES: frozenset[int] = frozenset(
        {408, 425, 429, 500, 502, 503, 504, 509, 529}
    )
    #: Default transient error-text markers (lowercased substrings). STRICTLY
    #: machine-generated error signatures — see the rule's "NL-heuristics
    #: boundary" docstring. Do NOT add natural-language prose here.
    _DEFAULT_MARKERS: tuple[str, ...] = (
        # SDK / runtime exception class names
        "ratelimiterror",
        "apitimeouterror",
        "apiconnectionerror",
        "serviceunavailableerror",
        "internalservererror",
        "overloadederror",
        "timeouterror",
        "readtimeout",
        "connecttimeout",
        "connectionreseterror",
        "connectionerror",
        "jsondecodeerror",
        # structured error codes / types
        "rate_limit_exceeded",
        "rate_limit",
        "too_many_requests",
        "service_unavailable",
        "gateway_timeout",
        "deadline_exceeded",
        "overloaded_error",
        "econnreset",
        "etimedout",
        "econnaborted",
        # RFC-standardised HTTP status reason phrases
        "too many requests",
        "service unavailable",
        "gateway timeout",
        "bad gateway",
        "request timeout",
        "rate limit exceeded",
    )

    def __init__(
        self,
        *,
        status_codes: frozenset[int] | set[int] | None = None,
        markers: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self._status_codes: frozenset[int] = (
            frozenset(int(c) for c in status_codes)
            if status_codes is not None
            else self._DEFAULT_STATUS_CODES
        )
        self._markers: tuple[str, ...] = (
            tuple(str(m).strip().lower() for m in markers)
            if markers is not None
            else self._DEFAULT_MARKERS
        )

    def edit(
        self,
        contents: list[Any],
        ctx: ContextEditContext,
    ) -> list[Any] | None:
        out: list[Any] | None = None
        for idx, content in enumerate(contents):
            redacted = self._redact_content(content)
            if redacted is None:
                continue
            if out is None:
                out = list(contents)
            out[idx] = redacted
        return out

    # ---- internals ----------------------------------------------------

    def _redaction_value(self) -> dict[str, str]:
        """A fresh, minimal redaction payload (new object each call)."""
        return {"goldfive_redacted": "transient_error_elided"}

    def _is_transient(self, resp: Any) -> bool:
        if not isinstance(resp, dict):
            return False
        lowered = _lower_keyed(resp)
        for code_field in _CODE_FIELDS:
            if code_field in lowered:
                code = _coerce_int(lowered[code_field])
                if code is not None and code in self._status_codes:
                    return True
        if not _response_has_error_indicator(resp):
            return False
        text = _response_error_text(resp)
        return any(m in text for m in self._markers)

    def _redact_content(self, content: Any) -> Any | None:
        """Return a redacted deepcopy of ``content``, or ``None`` for no-op.

        Only redacts a ``function_response`` part when the response is
        transient AND the redaction strictly shrinks that part's
        serialised bytes — guaranteeing the editor's byte-monotonicity
        gate holds without a revert.
        """
        redaction = self._redaction_value()
        redaction_len = len(json.dumps(redaction, default=repr))
        target_part_indices: list[int] = []
        parts = _iter_parts(content)
        for i, part in enumerate(parts):
            fr = _safe_attr(part, "function_response", None)
            if fr is None:
                continue
            resp = _safe_attr(fr, "response", None)
            if resp is None or not self._is_transient(resp):
                continue
            try:
                resp_len = len(json.dumps(resp, default=repr))
            except Exception:  # noqa: BLE001
                continue
            if resp_len <= redaction_len:
                # Redaction wouldn't shrink this part — leave it.
                continue
            target_part_indices.append(i)
        if not target_part_indices:
            return None
        try:
            clone = copy.deepcopy(content)
        except Exception:  # noqa: BLE001
            return None
        clone_parts = _iter_parts(clone)
        if len(clone_parts) != len(parts):
            return None
        for i in target_part_indices:
            fr = _safe_attr(clone_parts[i], "function_response", None)
            if fr is None:
                continue
            try:
                fr.response = self._redaction_value()
            except Exception:  # noqa: BLE001
                return None
        return clone


# ---------------------------------------------------------------------------
# Rule — PruneStaleSteerRule (drop-only)
# ---------------------------------------------------------------------------


class PruneStaleSteerRule:
    """Drop goldfive's own synthetic steer / observer-note user-messages once stale.

    Today a synthetic steer message goldfive injected (the advisory note
    delivered as a user turn) sticks in the transcript forever — even
    after the steering took effect — wasting tokens and biasing the model
    toward a correction that is no longer relevant (CONTEXT-EDITING.md
    rule 3).

    Identification (stable-keyed, single-source)
    --------------------------------------------
    A candidate is a ``Content`` whose text carries one of goldfive's
    OWN constants: the observer-note marker
    (``goldfive.observer_notes.OBSERVER_NOTE_MARKER_PREFIX``) or the
    pinned advisory footer (``goldfive.observer_notes.ADVISORY_FOOTER``).
    Both constants live ONLY in :mod:`goldfive.observer_notes` (the #455
    module) — the SAME place PR 6's channel renders them from — so the
    writer and this reader can never drift. Both are goldfive-minted
    strings, so matching is a stable-identity check, NOT an NL heuristic
    over agent output (the #166/#167 anti-pattern). Other contents are
    never touched.

    Staleness (the trigger)
    -----------------------
    A candidate is stale when it is no longer the CURRENTLY-ACTIVE steer.
    "Active" is read from ``goldfive.active_steer.body`` on session state
    (:func:`goldfive.state_store.set_active_steer` /
    :func:`~goldfive.state_store.clear_active_steer`): the note whose
    text still contains the active-steer body is kept; every other
    goldfive note is stale. When no steer is active (the body was
    cleared once the steered work resolved — runner clears it on
    completion) ALL goldfive notes are stale.

    This is the stable-keyed proxy for "the steered plan revision is
    COMPLETED": goldfive clears / supersedes the active steer when the
    correction has taken, so a goldfive note that no longer matches the
    active steer is a residue of a resolved correction.

    Dormancy
    --------
    A healthy run never produced a steer, so there are no candidate
    notes and the rule returns ``None``. Drop-only: it removes whole
    note ``Content`` entries (plain user-text turns — no tool parts, so
    pairing is unaffected).

    PR-6 dependency
    ---------------
    Full coverage of legacy plain-text notes arrives when AGENCY-
    PRESERVATION.md PR 6's channel wraps every delivered note in
    ``goldfive.observer_notes.OBSERVER_NOTE_MARKER_PREFIX``; until then
    the advisory-footer match already catches notes rendered through
    :mod:`goldfive.observer_notes`.
    """

    name = "prune_stale_steer"
    rule_class = _RULE_CLASS_DROP_ONLY

    def edit(
        self,
        contents: list[Any],
        ctx: ContextEditContext,
    ) -> list[Any] | None:
        footer, marker = self._note_markers()
        if not footer and not marker:
            return None
        note_idx: set[int] = set()
        for i, content in enumerate(contents):
            if self._is_goldfive_note(content, footer, marker):
                note_idx.add(i)
        if not note_idx:
            return None

        active_body = self._read_active_steer_body(ctx.session)
        survivors: list[Any] = []
        dropped = False
        for i, content in enumerate(contents):
            if i in note_idx and self._is_stale(content, active_body):
                dropped = True
                continue
            survivors.append(content)

        if not dropped or not survivors:
            return None
        return survivors

    # ---- internals ----------------------------------------------------

    #: Literal fail-safe used ONLY if :mod:`goldfive.observer_notes` can't
    #: be imported. The canonical source is
    #: ``observer_notes.OBSERVER_NOTE_MARKER_PREFIX``; this mirror exists so
    #: a (near-impossible) import failure of a core module degrades to
    #: still-functioning detection rather than silently matching nothing.
    #: observer_notes is the single source in every normal path.
    _MARKER_FALLBACK = "[GOLDFIVE OBSERVER NOTE"

    @classmethod
    def _note_markers(cls) -> tuple[str, str]:
        """Return ``(advisory_footer, observer_note_marker_prefix)`` — both
        goldfive constants, single-sourced from
        :mod:`goldfive.observer_notes`.

        On the (near-impossible) import failure of that core module, the
        marker degrades to the literal fallback so detection keeps
        functioning (fail-safe); the footer has no safe literal mirror
        (it is long pinned prose) and degrades to empty — the marker
        alone still catches PR-6-wrapped notes.
        """
        try:
            from goldfive.observer_notes import (  # noqa: PLC0415
                ADVISORY_FOOTER,
                OBSERVER_NOTE_MARKER_PREFIX,
            )

            return (
                str(ADVISORY_FOOTER or ""),
                str(OBSERVER_NOTE_MARKER_PREFIX or "") or cls._MARKER_FALLBACK,
            )
        except Exception:  # noqa: BLE001
            return ("", cls._MARKER_FALLBACK)

    def _is_goldfive_note(self, content: Any, footer: str, marker: str) -> bool:
        text = _content_text(content)
        if not text:
            return False
        if marker and marker in text:
            return True
        return bool(footer) and footer in text

    def _is_stale(self, content: Any, active_body: str) -> bool:
        if not active_body:
            # No active steer — every goldfive note is residue.
            return True
        return active_body not in _content_text(content)

    @staticmethod
    def _read_active_steer_body(session: Any) -> str:
        try:
            from goldfive import state_store as _ostate  # noqa: PLC0415

            state = _safe_attr(session, "state", None)
            if state is None:
                return ""
            return str(_ostate.read(state, _ostate.KEY_ACTIVE_STEER_BODY, "") or "")
        except Exception:  # noqa: BLE001
            return ""


# ---------------------------------------------------------------------------
# Rule — CompactPriorReasoningRule (byte-monotonic replace)
# ---------------------------------------------------------------------------


class CompactPriorReasoningRule:
    """Collapse N identical FAILED tool-call pairs into one summarized survivor.

    When the wrapped agent loops — calling the same tool with the same
    arguments and getting the same error every time — the failed trail
    accumulates in context and the model re-anchors on it (largely what
    looping *is*; CONTEXT-EDITING.md rule 4). This rule keeps the FIRST
    call/response pair of each identical-failed run, replaces its
    response with a one-line summary noting how many duplicates were
    collapsed, and drops the rest.

    Why byte-monotonic-replace
    --------------------------
    The summary is goldfive-synthesized text (it did not exist in the
    prior ``contents``), so the rule is ``byte_monotonic_replace``, not
    ``drop_only``. The structural gate still binds: dropping N-1 full
    call/response pairs vastly outweighs the one short summary, so the
    edit stays subtractive in both byte total and content count.

    Pairing safety
    --------------
    Both halves of each dropped repeat (call content + response content)
    are removed together, and the kept pair keeps its ``id`` + ``name``
    (only the response payload is summarized), so the editor's pairing
    invariant holds. An ownership guard (:func:`_content_owned_by_id`)
    skips any content that batches parts for multiple call ids, so
    unrelated calls are never disturbed.

    Dormancy (the trigger)
    ----------------------
    Compaction fires only when a group of ≥ ``min_repeats`` identical
    FAILED calls exists — itself the structural ``LOOPING_TOOL_CALL``
    guardrail condition. When a ``looping_tool_call`` /
    ``looping_reasoning`` drift verdict is ALSO recorded on the session
    (``ctx.active_drift_kinds``) the threshold drops to 2 — the
    guardrail counter has already tripped, so a single confirmed
    duplicate is enough to act. A healthy run has no repeated failed
    calls and the rule returns ``None``.
    """

    name = "compact_prior_reasoning"
    rule_class = _RULE_CLASS_BYTE_MONOTONIC_REPLACE

    _DEFAULT_MIN_REPEATS = 3
    _LOOPING_DRIFT_KINDS: frozenset[str] = frozenset(
        {"looping_tool_call", "looping_reasoning"}
    )

    def __init__(self, *, min_repeats: int | None = None) -> None:
        raw = self._DEFAULT_MIN_REPEATS if min_repeats is None else int(min_repeats)
        # Never below 2 — collapsing a single call is meaningless.
        self._min_repeats = max(2, raw)

    def edit(
        self,
        contents: list[Any],
        ctx: ContextEditContext,
    ) -> list[Any] | None:
        threshold = self._min_repeats
        if ctx.active_drift_kinds & self._LOOPING_DRIFT_KINDS:
            threshold = 2

        # Locate every call id's call content + signature.
        call_idx_by_id: dict[str, int] = {}
        sig_by_id: dict[str, tuple[str, str]] = {}
        for i, content in enumerate(contents):
            for part in _iter_parts(content):
                fc = _safe_attr(part, "function_call", None)
                if fc is None:
                    continue
                fc_id = str(_safe_attr(fc, "id", "") or "")
                if not fc_id or fc_id in call_idx_by_id:
                    continue
                call_idx_by_id[fc_id] = i
                sig_by_id[fc_id] = _function_call_signature(fc)

        # Locate every call id's response content + whether it failed.
        resp_idx_by_id: dict[str, int] = {}
        failed_ids: set[str] = set()
        for i, content in enumerate(contents):
            for part in _iter_parts(content):
                fr = _safe_attr(part, "function_response", None)
                if fr is None:
                    continue
                fr_id = str(_safe_attr(fr, "id", "") or "")
                if not fr_id or fr_id in resp_idx_by_id:
                    continue
                resp_idx_by_id[fr_id] = i
                resp = _safe_attr(fr, "response", None)
                if _response_has_error_indicator(resp):
                    failed_ids.add(fr_id)

        # Group failed, fully-paired ids by call signature.
        groups: dict[tuple[str, str], list[str]] = {}
        for fc_id, sig in sig_by_id.items():
            if fc_id in resp_idx_by_id and fc_id in failed_ids:
                groups.setdefault(sig, []).append(fc_id)

        drop_indices: set[int] = set()
        summarize_clone: dict[int, Any] = {}  # kept resp content idx -> clone
        for sig, ids in groups.items():
            if len(ids) < threshold:
                continue
            ids_sorted = sorted(ids, key=lambda c: call_idx_by_id[c])
            owned: dict[str, tuple[int, int]] = {}
            ownership_ok = True
            for fc_id in ids_sorted:
                ci = call_idx_by_id[fc_id]
                ri = resp_idx_by_id[fc_id]
                if not _content_owned_by_id(contents[ci], fc_id) or not _content_owned_by_id(
                    contents[ri], fc_id
                ):
                    ownership_ok = False
                    break
                owned[fc_id] = (ci, ri)
            if not ownership_ok:
                continue
            keep = ids_sorted[0]
            _keep_ci, keep_ri = owned[keep]

            # Build the summarized survivor and only collapse this group
            # when doing so STRICTLY reduces bytes. The summary is
            # synthesized text, so a run of tiny failed responses could
            # otherwise grow the transcript — in which case there is no
            # benefit and the editor's byte gate would revert the whole
            # edit. Self-checking here keeps the rule a clean no-op for
            # the not-beneficial case instead of emitting a spurious
            # ContextEditRejected.
            clone = self._summarize_response(
                contents[keep_ri], name=sig[0], count=len(ids_sorted)
            )
            if clone is None:
                continue
            saved = 0
            for fc_id in ids_sorted[1:]:
                ci, ri = owned[fc_id]
                saved += _content_bytes([contents[ci]]) + _content_bytes([contents[ri]])
            saved += _content_bytes([contents[keep_ri]])  # kept response is replaced
            added = _content_bytes([clone])
            if saved <= added:
                continue

            for fc_id in ids_sorted[1:]:
                ci, ri = owned[fc_id]
                drop_indices.add(ci)
                drop_indices.add(ri)
            summarize_clone[keep_ri] = clone

        if not drop_indices and not summarize_clone:
            return None

        out: list[Any] = []
        for i, content in enumerate(contents):
            if i in drop_indices:
                continue
            if i in summarize_clone:
                out.append(summarize_clone[i])
            else:
                out.append(content)
        return out

    # ---- internals ----------------------------------------------------

    @staticmethod
    def _summary_value(*, name: str, count: int) -> dict[str, str]:
        return {
            "goldfive_compacted": (
                f"The tool '{name}' was invoked {count} times with identical "
                f"arguments and identical results; {count - 1} duplicate "
                f"invocations were collapsed to keep the context focused."
            )
        }

    def _summarize_response(self, content: Any, *, name: str, count: int) -> Any | None:
        """Return a deepcopy of ``content`` with its response part(s) summarized."""
        summary = self._summary_value(name=name, count=count)
        try:
            clone = copy.deepcopy(content)
        except Exception:  # noqa: BLE001
            return None
        touched = False
        for part in _iter_parts(clone):
            fr = _safe_attr(part, "function_response", None)
            if fr is None:
                continue
            try:
                fr.response = dict(summary)
                touched = True
            except Exception:  # noqa: BLE001
                return None
        return clone if touched else None


# ---------------------------------------------------------------------------
# Rule registry — name -> factory
# ---------------------------------------------------------------------------


#: Map of opt-in rule name (the string used in
#: :class:`~goldfive.config.SteeringConfig.context_editor_rules`) to a
#: zero-arg factory that builds the rule instance. New rules add a row
#: here AND under the catalog in ``docs/design/CONTEXT-EDITING.md``.
_RULE_REGISTRY: dict[str, Any] = {
    "prune_cancelled_reasoning": PruneCancelledReasoningRule,
    "prune_transient_error": PruneTransientErrorRule,
    "prune_stale_steer": PruneStaleSteerRule,
    "compact_prior_reasoning": CompactPriorReasoningRule,
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
    "CompactPriorReasoningRule",
    "ContextEditContext",
    "ContextEditRule",
    "ContextEditor",
    "PruneCancelledReasoningRule",
    "PruneStaleSteerRule",
    "PruneTransientErrorRule",
    "build_editor_from_config",
]
