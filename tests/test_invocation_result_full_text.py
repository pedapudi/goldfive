"""Tests for full-fidelity output capture on :class:`InvocationResult`.

Background (zicato#12): evaluators that grade an agent's output must see the
agent's *actual* produced text, not a lossy single-turn slice. ``text`` keeps
its legacy meaning (the final assistant turn) for backward compatibility;
``text_turns`` / ``full_text`` are the new additive, full-fidelity channel.
"""

from __future__ import annotations

from goldfive.results import TURN_SEPARATOR, InvocationResult


def test_text_only_construction_back_compat() -> None:
    """A legacy caller passing only ``text=`` is unchanged, and ``full_text``
    falls back to ``text`` so a grader reading the new field still works."""
    r = InvocationResult(task_id="t1", text="the answer is 42")
    assert r.text == "the answer is 42"
    # No per-turn record supplied — full_text mirrors text.
    assert r.text_turns == []
    assert r.full_text == "the answer is 42"


def test_text_turns_populate_full_text() -> None:
    """When the adapter records every turn, ``full_text`` is the joined turns
    and ``text`` stays the LAST turn (legacy semantics)."""
    turns = [
        "Found the KEYWORD_____ID_x_y_V2 table with 3 matching rows.",
        "Done — let me know if you need anything else.",
    ]
    r = InvocationResult(task_id="t1", text=turns[-1], text_turns=turns)
    # Legacy field: last turn only.
    assert r.text == "Done — let me know if you need anything else."
    # Full-fidelity field: every turn, in order — the substantive answer is
    # NOT dropped.
    assert r.full_text == TURN_SEPARATOR.join(turns)
    assert "KEYWORD_____ID_x_y_V2" in r.full_text
    # And the substantive id is absent from the lossy last-turn field, which is
    # precisely the precision bug graders hit when keying off ``text``.
    assert "KEYWORD_____ID_x_y_V2" not in r.text


def test_explicit_full_text_is_respected() -> None:
    """An adapter may author ``full_text`` directly; it is not overwritten."""
    r = InvocationResult(
        task_id="t1",
        text="last",
        text_turns=["a", "b"],
        full_text="explicit override",
    )
    assert r.full_text == "explicit override"


def test_empty_invocation() -> None:
    r = InvocationResult(task_id="t1")
    assert r.text == ""
    assert r.text_turns == []
    assert r.full_text == ""
