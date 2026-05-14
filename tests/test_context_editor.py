"""Unit tests for ``goldfive.context_editor`` (goldfive#397).

Coverage matrix
---------------

* Empty rule set is a no-op.
* ``observation_only=True`` skips the entire pipeline (strict-passive
  pattern from goldfive#271).
* ``tool_call_id`` pairing invariant — a rule that strips half a pair
  is rejected, ``ContextEditRejected`` is emitted, and the
  ``llm_request.contents`` reference is unchanged.
* Idempotence — the same rule applied twice with the same input
  produces the same output.
* ``PruneCancelledReasoningRule`` — synthetic ``contents`` with one
  cancelled pair and one live pair: editor strips the cancelled pair
  and leaves the live one.
* ``ContextEdited`` event emission on successful edit.
* ``build_editor_from_config`` returns ``None`` for empty / unknown
  rule names (gating the zero-overhead path).
* Drop-only / no-injection invariant — a rule that grows ``contents``
  is rejected.

The tests use plain Python objects with the same duck-typed surface
goldfive's ``_content_bytes`` and ``_function_call_ids`` helpers read
(``role``, ``parts``, ``part.text``, ``part.function_call.{id,name,args}``,
``part.function_response.{id,name,response}``). They do NOT require
google-genai / ADK to be installed — context_editor was designed against
the duck-typed surface for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from goldfive.context_editor import (
    ContextEditor,
    PruneCancelledReasoningRule,
    build_editor_from_config,
)
from goldfive.sinks.memory import InMemorySink

# ---------------------------------------------------------------------------
# Fake content / part / request objects mirroring the ADK duck-type
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    id: str = ""
    name: str = ""
    args: dict[str, Any] | None = None


@dataclass
class FakeFunctionResponse:
    id: str = ""
    name: str = ""
    response: dict[str, Any] | None = None


@dataclass
class FakePart:
    text: str = ""
    function_call: FakeFunctionCall | None = None
    function_response: FakeFunctionResponse | None = None


@dataclass
class FakeContent:
    role: str = "user"
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeRequest:
    contents: list[FakeContent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fake Session shim — only the surface ContextEditor + rule read
# ---------------------------------------------------------------------------


@dataclass
class FakePlan:
    revision_index: int = 0


class FakeSession:
    """Minimal Session shim — exposes ``state``, ``plan``, ``run_id``, ``id``,
    and ``next_sequence`` so the editor + emit paths exercise without
    pulling in the full ``goldfive.types.Session``.

    ``state`` is a plain dict; the
    :class:`~goldfive.context_editor.PruneCancelledReasoningRule` reads
    ``goldfive.cancelled_function_call_ids`` off it via
    :func:`goldfive.state_store.read_cancelled_function_call_ids`.
    """

    def __init__(
        self,
        *,
        run_id: str = "r-test",
        session_id: str = "s-test",
        plan: FakePlan | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.id = session_id
        self.plan = plan or FakePlan()
        self.state = dict(state or {})
        self._seq = 0

    def next_sequence(self) -> int:
        self._seq += 1
        return self._seq


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def _text_content(text: str, role: str = "user") -> FakeContent:
    return FakeContent(role=role, parts=[FakePart(text=text)])


def _call_content(fc_id: str, name: str = "do_work") -> FakeContent:
    return FakeContent(
        role="model",
        parts=[
            FakePart(
                function_call=FakeFunctionCall(id=fc_id, name=name, args={"k": "v"})
            )
        ],
    )


def _response_content(
    fc_id: str, name: str = "do_work", response: dict[str, Any] | None = None
) -> FakeContent:
    return FakeContent(
        role="user",
        parts=[
            FakePart(
                function_response=FakeFunctionResponse(
                    id=fc_id, name=name, response=response or {"ok": True}
                )
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_rule_set_is_noop() -> None:
    """An editor with zero rules MUST leave the request untouched."""
    editor = ContextEditor(rules=[], sinks=[InMemorySink()])
    request = FakeRequest(
        contents=[_text_content("hello"), _call_content("fc-1"), _response_content("fc-1")]
    )
    original_contents = list(request.contents)
    session = FakeSession()

    result = await editor.apply(
        request,
        session=session,
        host_agent_name="agent_a",
        observation_only=False,
    )

    assert request.contents == original_contents
    assert result.applied_rules == []
    assert result.rejected_rules == []


@pytest.mark.asyncio
async def test_observation_only_skips_pipeline() -> None:
    """``observation_only=True`` MUST short-circuit the entire chain.

    Even when rules are registered AND would apply, no edit fires.
    Mirrors the strict-passive contract from goldfive#271.
    """

    class _AlwaysFires:
        name = "always_fires"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            # Would strip everything if allowed — observation_only must
            # gate that.
            return []

    sink = InMemorySink()
    editor = ContextEditor(rules=[_AlwaysFires()], sinks=[sink])
    request = FakeRequest(contents=[_text_content("hello")])
    original_contents = list(request.contents)

    result = await editor.apply(
        request,
        session=FakeSession(),
        host_agent_name="agent_a",
        observation_only=True,
    )

    assert request.contents == original_contents
    assert result.applied_rules == []
    assert result.rejected_rules == []
    # No event fires under observation_only — the pipeline is
    # completely bypassed.
    assert sink.events == []


@pytest.mark.asyncio
async def test_tool_call_id_pair_violation_reverts() -> None:
    """A rule stripping half a pair MUST be reverted + ContextEditRejected emitted."""

    class _HalfPairStripper:
        """Drops the function_response but keeps the function_call —
        breaks the pairing invariant."""

        name = "half_pair_stripper"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            # Keep everything EXCEPT the function_response — that
            # leaves the function_call orphaned.
            return [
                c
                for c in contents
                if not any(p.function_response is not None for p in c.parts)
            ]

    sink = InMemorySink()
    editor = ContextEditor(rules=[_HalfPairStripper()], sinks=[sink])
    request = FakeRequest(
        contents=[_text_content("hi"), _call_content("fc-X"), _response_content("fc-X")]
    )
    original_contents = list(request.contents)

    result = await editor.apply(
        request,
        session=FakeSession(),
        host_agent_name="agent_a",
        observation_only=False,
    )

    # Edit reverted — contents unchanged.
    assert request.contents == original_contents
    assert result.applied_rules == []
    assert ("half_pair_stripper", "tool_call_id_pair_violation") in result.rejected_rules

    # A ContextEditRejected event fired with the right reason.
    rejected_events = [
        e for e in sink.events if isinstance(e, dict) and e.get("kind") == "context_edit_rejected"
    ]
    assert len(rejected_events) == 1
    assert rejected_events[0]["payload"]["rule_name"] == "half_pair_stripper"
    assert rejected_events[0]["payload"]["reason"] == "tool_call_id_pair_violation"


@pytest.mark.asyncio
async def test_idempotence_same_input_same_output() -> None:
    """Applying the same rule twice with the same input MUST produce the same result.

    Idempotence-per-revision (Invariant 4): a deterministic rule
    evaluated twice on identical ``contents`` AND identical
    ``observed_revision_index`` MUST return the same output. Catches
    accidental nondeterminism in rule authors.
    """

    class _DeterministicDropper:
        name = "deterministic_dropper"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            # Drop any Content whose first part text starts with
            # "drop:".
            survivors = []
            any_dropped = False
            for c in contents:
                first_text = c.parts[0].text if c.parts else ""
                if first_text.startswith("drop:"):
                    any_dropped = True
                    continue
                survivors.append(c)
            return survivors if any_dropped else None

    sinks = [InMemorySink(), InMemorySink()]
    base_contents = [
        _text_content("keep me"),
        _text_content("drop: this"),
        _text_content("keep too"),
    ]

    # Two independent editors so they don't share state.
    editor_a = ContextEditor(rules=[_DeterministicDropper()], sinks=[sinks[0]])
    editor_b = ContextEditor(rules=[_DeterministicDropper()], sinks=[sinks[1]])

    request_a = FakeRequest(contents=list(base_contents))
    request_b = FakeRequest(contents=list(base_contents))
    session = FakeSession(plan=FakePlan(revision_index=7))

    result_a = await editor_a.apply(
        request_a, session=session, host_agent_name="a", observation_only=False
    )
    result_b = await editor_b.apply(
        request_b, session=session, host_agent_name="a", observation_only=False
    )

    assert result_a.applied_rules == result_b.applied_rules == ["deterministic_dropper"]
    # Same content count and the survivors are the same shape.
    assert [c.parts[0].text for c in request_a.contents] == [
        c.parts[0].text for c in request_b.contents
    ]
    assert request_a.contents != base_contents  # The edit DID apply.


@pytest.mark.asyncio
async def test_prune_cancelled_reasoning_smoke() -> None:
    """Synthetic contents with one cancelled pair + one live pair.

    The rule MUST strip the cancelled pair (both the function_call and
    its paired function_response) and leave the live pair + any text
    intact.
    """
    from goldfive import state_store as _ostate

    session = FakeSession()
    # Stamp one cancelled function_call_id on goldfive Session state
    # — same write path :meth:`ADKAdapter._heal_pending_tool_calls`
    # uses in production.
    _ostate.append_cancelled_function_call_ids(session.state, ["fc-CANCELLED"])

    sink = InMemorySink()
    editor = ContextEditor(rules=[PruneCancelledReasoningRule()], sinks=[sink])
    request = FakeRequest(
        contents=[
            _text_content("user input"),
            _call_content("fc-LIVE", name="research"),
            _response_content("fc-LIVE", name="research"),
            _call_content("fc-CANCELLED", name="aborted_work"),
            _response_content("fc-CANCELLED", name="aborted_work"),
            _text_content("model continues", role="model"),
        ]
    )

    result = await editor.apply(
        request, session=session, host_agent_name="a", observation_only=False
    )

    assert result.applied_rules == ["prune_cancelled_reasoning"]
    # The cancelled pair is gone; the live pair survives.
    fc_ids_surviving = [
        p.function_call.id
        for c in request.contents
        for p in c.parts
        if p.function_call is not None
    ]
    fr_ids_surviving = [
        p.function_response.id
        for c in request.contents
        for p in c.parts
        if p.function_response is not None
    ]
    assert fc_ids_surviving == ["fc-LIVE"]
    assert fr_ids_surviving == ["fc-LIVE"]
    # Both text contents survive.
    text_parts = [
        p.text for c in request.contents for p in c.parts if p.text
    ]
    assert text_parts == ["user input", "model continues"]


@pytest.mark.asyncio
async def test_context_edited_event_emitted_on_apply() -> None:
    """A successful edit MUST emit a ``ContextEdited`` event with byte deltas."""

    class _DropFirstContent:
        name = "drop_first"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            if not contents:
                return None
            return contents[1:]

    sink = InMemorySink()
    editor = ContextEditor(rules=[_DropFirstContent()], sinks=[sink])
    # Two text contents so the pairing invariant is satisfied (no
    # function_call/response pairs at all).
    request = FakeRequest(
        contents=[
            _text_content("preamble to drop"),
            _text_content("keeper one"),
            _text_content("keeper two"),
        ]
    )
    session = FakeSession(plan=FakePlan(revision_index=3))

    result = await editor.apply(
        request, session=session, host_agent_name="a", observation_only=False
    )

    assert result.applied_rules == ["drop_first"]
    edited_events = [
        e for e in sink.events if isinstance(e, dict) and e.get("kind") == "context_edited"
    ]
    assert len(edited_events) == 1
    payload = edited_events[0]["payload"]
    assert payload["rule_name"] == "drop_first"
    assert payload["bytes_after"] < payload["bytes_before"]
    assert payload["contents_count_after"] == 2
    assert payload["contents_count_before"] == 3
    assert payload["observed_revision_index"] == 3


@pytest.mark.asyncio
async def test_build_editor_from_config_gating() -> None:
    """``build_editor_from_config`` returns ``None`` for empty / unknown rules.

    The plugin's ``before_model_callback`` short-circuits on the
    editor being ``None`` — this gates the zero-overhead path.
    """
    # None / empty → None
    assert build_editor_from_config(None) is None
    assert build_editor_from_config([]) is None

    # Unknown rule name → None (filtered + logged at WARNING)
    assert build_editor_from_config(["nope"]) is None
    assert build_editor_from_config(["nope", "also_nope"]) is None

    # Known rule → editor with that rule registered
    editor = build_editor_from_config(["prune_cancelled_reasoning"])
    assert editor is not None
    rules = editor.rules
    assert len(rules) == 1
    assert rules[0].name == "prune_cancelled_reasoning"

    # Mixed known + unknown — unknowns dropped silently, knowns honoured.
    editor = build_editor_from_config(["prune_cancelled_reasoning", "nope"])
    assert editor is not None
    assert [r.name for r in editor.rules] == ["prune_cancelled_reasoning"]


@pytest.mark.asyncio
async def test_drop_only_invariant_rejects_growth() -> None:
    """A rule that grows ``contents`` MUST be reverted with reason ``not_drop_only``."""

    class _Grower:
        name = "grower"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            # Doubles the contents — clear violation of drop-only.
            return list(contents) + list(contents)

    sink = InMemorySink()
    editor = ContextEditor(rules=[_Grower()], sinks=[sink])
    request = FakeRequest(contents=[_text_content("a"), _text_content("b")])
    original_contents = list(request.contents)

    result = await editor.apply(
        request, session=FakeSession(), host_agent_name="a", observation_only=False
    )

    assert request.contents == original_contents
    assert result.applied_rules == []
    assert ("grower", "not_drop_only") in result.rejected_rules
    rejected_events = [
        e for e in sink.events if isinstance(e, dict) and e.get("kind") == "context_edit_rejected"
    ]
    assert len(rejected_events) == 1
    assert rejected_events[0]["payload"]["reason"] == "not_drop_only"


@pytest.mark.asyncio
async def test_prune_cancelled_reasoning_no_cancelled_ids_is_noop() -> None:
    """When no ids are stamped on session.state, the rule returns None (skip)."""
    sink = InMemorySink()
    editor = ContextEditor(rules=[PruneCancelledReasoningRule()], sinks=[sink])
    request = FakeRequest(
        contents=[
            _text_content("user input"),
            _call_content("fc-LIVE"),
            _response_content("fc-LIVE"),
        ]
    )
    original_contents = list(request.contents)

    result = await editor.apply(
        request,
        session=FakeSession(),  # no cancelled ids stamped
        host_agent_name="a",
        observation_only=False,
    )

    assert request.contents == original_contents
    assert result.applied_rules == []
    # Skipped (not rejected) — rule returned None.
    assert "prune_cancelled_reasoning" in result.skipped_rules


@pytest.mark.asyncio
async def test_rule_that_raises_is_skipped_chain_continues() -> None:
    """A raising rule is logged + skipped; subsequent rules still run."""

    class _Raiser:
        name = "raiser"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    class _DropFirst:
        name = "drop_first"

        def edit(self, contents, ctx):  # type: ignore[no-untyped-def]
            return contents[1:] if len(contents) > 1 else None

    sink = InMemorySink()
    editor = ContextEditor(rules=[_Raiser(), _DropFirst()], sinks=[sink])
    request = FakeRequest(
        contents=[_text_content("drop me"), _text_content("survive")]
    )

    result = await editor.apply(
        request, session=FakeSession(), host_agent_name="a", observation_only=False
    )

    assert ("raiser", "rule_raised") in result.rejected_rules
    assert "drop_first" in result.applied_rules
    assert [c.parts[0].text for c in request.contents] == ["survive"]
