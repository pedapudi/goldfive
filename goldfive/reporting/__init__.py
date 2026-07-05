"""Reporting-tool specs and handlers.

The eight canonical reporting tools — the agent-facing contract for
driving the plan's task state machine and signalling plan mutations.
Each :class:`ReportingToolSpec` pairs a stable tool name with a JSON-schema
parameters block and an async handler. Handlers receive the decoded
arguments, the live :class:`Session`, and the bound :class:`Steerer`, and
route the call into the steerer's transition / drift pipeline.

Adapters materialise these specs into whatever native tool shape their
framework wants (ADK ``FunctionTool``, Claude Agent SDK tool blocks, …).

The eighth tool, ``report_awaiting_approval``, is the task-level half of
the human-in-the-loop approval flow described in
``docs/design/APPROVAL.md``. Its handler blocks the calling tool-call
until the control dispatcher lands an ``APPROVE`` or ``REJECT`` (with a
finite timeout, and an immediate degraded ack when no control channel
is bound).

Package structure (post-split, Wave A piece 3):

* :mod:`goldfive.reporting.handlers` — the contract: what reporting
  tools DO when invoked. Holds ``ReportingToolSpec``,
  ``BUILTIN_REPORTING_TOOLS``, the ``_handle_*`` async functions, the
  drift-opt-in selection helper, and the declaration-key constants.
* :mod:`goldfive.reporting.schemas` — JSON-schema parameter blocks
  for the canonical tools. These go over the wire to LLMs as tool
  schemas; the marshaling layer embeds them verbatim.
* :mod:`goldfive.reporting.rendering` — payload construction for what
  the LLM sees on the response side: directive ack with ``plan_state``,
  idempotent / invalid-transition / missing-arg rejection shapes.
* :mod:`goldfive.reporting._internal` — shared private utilities
  (argument coercion, pin freshness classification, sink emit
  helpers). Not part of the public API.

The public API is unchanged: every name previously importable from
``goldfive.reporting`` is re-exported here.
"""

from __future__ import annotations

# Public surface — the legacy ``from goldfive.reporting import X`` paths.
# Private symbols (underscore-prefixed) are re-exported here for
# back-compat with tests / docs that historically reached into the
# flat ``goldfive.reporting`` module; they appear in ``__all__`` so
# tools (ruff F401) treat them as intentional re-exports. New code
# should NOT import these — they are package-private and may move.
from goldfive.reporting._internal import (
    _ACK,
    _TERMINAL_STATUSES,
    _TOOL_TARGET_STATUS,
    _TOOL_VALID_SOURCES,
    _await_plan_stable,
    _bool,
    _classify_pin_freshness,
    _classify_transition,
    _emit_approval_requested,
    _emit_task_declaration_received,
    _emit_task_transition_refused,
    _find_task_in_session,
    _float,
    _int,
    _PinFreshness,
    _read_pin_revision,
    _read_plan_revision,
    _reroute_if_superseded,
    _resolve_effective_task_id,
    _resolve_task_id,
    _resolve_task_id_with_source,
    _rotate_after_terminal,
    _str,
    _supersession_successor,
)
from goldfive.reporting.handlers import (
    BUILTIN_REPORTING_TOOLS,
    DECLARATION_KIND_NOT_NEEDED,
    DECLARATION_KIND_SKIPPED,
    DECLARATION_KINDS,
    DECLARATIONS_KEY,
    DRIFT_SELF_REPORTING_TOOL_NAMES,
    DRIFT_SELF_REPORTING_TOOLS,
    LIFECYCLE_REPORTING_TOOLS,
    REPORTING_TOOL_NAMES,
    ReportingHandler,
    ReportingToolSpec,
    _classify_and_route_pin,
    _clear_correction_on_started,
    _handle_awaiting_approval,
    _handle_declaration,
    _handle_declare_task_not_needed,
    _handle_declare_task_skipped,
    _handle_new_work_discovered,
    _handle_plan_divergence,
    _handle_task_blocked,
    _handle_task_completed,
    _handle_task_failed,
    _handle_task_progress,
    _handle_task_started,
    _record_declaration,
    _validate_required,
    select_reporting_tools,
)
from goldfive.reporting.rendering import (
    _bare_agent_name,
    _build_plan_state,
    _directive_ack,
    _idempotent_response,
    _invalid_transition_response,
    _missing_required_field_response,
    _missing_task_id_response,
    _next_pending_with_completed_predecessors,
    _refused_response,
)
from goldfive.reporting.schemas import (
    _SCHEMA_AWAITING_APPROVAL,
    _SCHEMA_DECLARE_TASK_NOT_NEEDED,
    _SCHEMA_DECLARE_TASK_SKIPPED,
    _SCHEMA_NEW_WORK_DISCOVERED,
    _SCHEMA_PLAN_DIVERGENCE,
    _SCHEMA_TASK_BLOCKED,
    _SCHEMA_TASK_COMPLETED,
    _SCHEMA_TASK_FAILED,
    _SCHEMA_TASK_PROGRESS,
    _SCHEMA_TASK_STARTED,
    _object_schema,
)

__all__ = [
    # Public API (legacy ``from goldfive.reporting import X``).
    "BUILTIN_REPORTING_TOOLS",
    "DECLARATIONS_KEY",
    "DECLARATION_KINDS",
    "DECLARATION_KIND_NOT_NEEDED",
    "DECLARATION_KIND_SKIPPED",
    "DRIFT_SELF_REPORTING_TOOLS",
    "DRIFT_SELF_REPORTING_TOOL_NAMES",
    "LIFECYCLE_REPORTING_TOOLS",
    "REPORTING_TOOL_NAMES",
    "ReportingHandler",
    "ReportingToolSpec",
    "select_reporting_tools",
    # Private re-exports — tests and adapters historically reach into
    # these via ``goldfive.reporting._foo``. Listed here so ruff F401
    # treats them as intentional, and so ``from goldfive.reporting
    # import _foo`` keeps working post-split.
    "_ACK",
    "_PinFreshness",
    "_SCHEMA_AWAITING_APPROVAL",
    "_SCHEMA_DECLARE_TASK_NOT_NEEDED",
    "_SCHEMA_DECLARE_TASK_SKIPPED",
    "_SCHEMA_NEW_WORK_DISCOVERED",
    "_SCHEMA_PLAN_DIVERGENCE",
    "_SCHEMA_TASK_BLOCKED",
    "_SCHEMA_TASK_COMPLETED",
    "_SCHEMA_TASK_FAILED",
    "_SCHEMA_TASK_PROGRESS",
    "_SCHEMA_TASK_STARTED",
    "_TERMINAL_STATUSES",
    "_TOOL_TARGET_STATUS",
    "_TOOL_VALID_SOURCES",
    "_await_plan_stable",
    "_bare_agent_name",
    "_bool",
    "_build_plan_state",
    "_classify_and_route_pin",
    "_classify_pin_freshness",
    "_classify_transition",
    "_clear_correction_on_started",
    "_directive_ack",
    "_emit_approval_requested",
    "_emit_task_declaration_received",
    "_emit_task_transition_refused",
    "_find_task_in_session",
    "_float",
    "_handle_awaiting_approval",
    "_handle_declaration",
    "_handle_declare_task_not_needed",
    "_handle_declare_task_skipped",
    "_handle_new_work_discovered",
    "_handle_plan_divergence",
    "_handle_task_blocked",
    "_handle_task_completed",
    "_handle_task_failed",
    "_handle_task_progress",
    "_handle_task_started",
    "_idempotent_response",
    "_int",
    "_invalid_transition_response",
    "_missing_required_field_response",
    "_missing_task_id_response",
    "_next_pending_with_completed_predecessors",
    "_object_schema",
    "_read_pin_revision",
    "_read_plan_revision",
    "_record_declaration",
    "_refused_response",
    "_reroute_if_superseded",
    "_resolve_effective_task_id",
    "_resolve_task_id",
    "_resolve_task_id_with_source",
    "_rotate_after_terminal",
    "_str",
    "_supersession_successor",
    "_validate_required",
]
