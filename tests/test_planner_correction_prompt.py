"""Structural-class enrichment for ``LLMPlanner._build_correction_prompt``.

When the validator rejects a refine attempt with a structurally-
classified error (``edge_missing``, ``terminal_missing``,
``terminal_regressed``), the retry prompt must embed a copy-paste-ready
JSON snippet of EXACTLY what the LLM must add or preserve, in addition
to the existing "Re-read the STRUCTURAL INVARIANTS section above"
pointer. Empirically the LLM drops at least one terminal-edge on long-
context refines even with the invariants block present further up;
naming the missing piece right next to the rejection text fixes that
on attempt 2.

Unknown / malformed rejections must fall back silently to the prior
unenriched behaviour — never raise — so plumbing failures and goal-
coverage errors are unaffected.
"""

from __future__ import annotations

from goldfive.planner import LLMPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A representative ``base_prompt`` — the helper appends to whatever it
# receives. The tests don't care about the prompt body; they only care
# about what's appended after.
_BASE_PROMPT = "BASE PROMPT BODY (goals, history, invariants, etc.)"

# The pointer is invariant across all enriched/non-enriched prompts; we
# verify it remains present so the existing guidance is additive, not
# replaced.
_INVARIANTS_POINTER = "Re-read the STRUCTURAL INVARIANTS section above"

# Validator strings come straight from ``Plan.validate`` in
# ``goldfive/types.py`` (and get wrapped by ``_user_steer_one_attempt``
# with the ``"validator rejected revision: "`` prefix). Mirroring both
# wrapped and unwrapped forms here so any future drift surfaces.
_PREFIX = "validator rejected revision: "


# ---------------------------------------------------------------------------
# edge_missing
# ---------------------------------------------------------------------------


def test_edge_missing_appends_copy_pasteable_edge_json() -> None:
    """``terminal->terminal edge 'a' -> 'b' missing in revision`` →
    snippet contains a verbatim ``{"from_task_id": "a", "to_task_id":
    "b"}`` JSON object the LLM can paste into its ``edges`` array."""
    error = (
        _PREFIX
        + "terminal->terminal edge 'draft_slides' -> 'review_slides' missing in revision"
    )
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    # Raw error preserved.
    assert error in out
    # Existing pointer preserved (additive, not replacement).
    assert _INVARIANTS_POINTER in out
    # Copy-paste-ready edge JSON.
    assert '{"from_task_id": "draft_slides", "to_task_id": "review_slides"}' in out
    # Direct ADD instruction.
    assert "ADD THIS EDGE VERBATIM" in out
    # And of course the base prompt is still at the head.
    assert out.startswith(_BASE_PROMPT)


def test_edge_missing_handles_double_quoted_endpoints() -> None:
    """Defensive: tolerate double-quoted endpoints even though Python's
    ``!r`` always emits single quotes for ASCII. A future validator
    change must not silently disable enrichment."""
    error = 'terminal->terminal edge "a" -> "b" missing in revision'
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert '{"from_task_id": "a", "to_task_id": "b"}' in out


def test_edge_missing_unwrapped_form_still_enriches() -> None:
    """The classifier and parser both work without the
    ``_user_steer_one_attempt`` wrapper prefix — ``_build_correction_prompt``
    is occasionally fed raw errors by other call sites in planner.py."""
    error = "terminal->terminal edge 'x' -> 'y' missing in revision"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert '{"from_task_id": "x", "to_task_id": "y"}' in out


def test_malformed_edge_missing_falls_back_silently() -> None:
    """Error is bucketed as ``edge_missing`` (substring match) but is
    missing the second quoted endpoint, so the regex can't recover the
    capture group. The helper must NOT raise — it must fall back to
    the unenriched form (raw error + invariants pointer)."""
    # Substring "terminal->terminal edge" + "missing in revision"
    # buckets to edge_missing, but the regex fails to capture both ids.
    error = "terminal->terminal edge 'draft_slides' -> missing in revision"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    # No JSON snippet leaked.
    assert "from_task_id" not in out
    assert "ADD THIS EDGE VERBATIM" not in out
    # Raw error and pointer still present.
    assert error in out
    assert _INVARIANTS_POINTER in out


# ---------------------------------------------------------------------------
# terminal_missing
# ---------------------------------------------------------------------------


def test_terminal_missing_names_the_missing_task_id() -> None:
    """``terminal task 'task_x' missing in revision`` → snippet names
    ``task_x`` verbatim and tells the LLM to keep it with its current
    terminal status."""
    error = _PREFIX + "terminal task 'task_x' missing in revision"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert error in out
    assert _INVARIANTS_POINTER in out
    assert "KEEP THIS TASK VERBATIM" in out
    assert 'task id: "task_x"' in out
    # Don't try to reconstruct the full task object — only the id.
    assert '"title"' not in out
    assert '"assignee_agent_id"' not in out


def test_terminal_missing_handles_double_quoted_id() -> None:
    error = 'terminal task "task_x" missing in revision'
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert 'task id: "task_x"' in out


def test_malformed_terminal_missing_falls_back_silently() -> None:
    """Bucketed as ``terminal_missing`` but the id quoting is broken so
    the regex returns no match → fall back to unenriched form."""
    # The substrings "terminal task" + "missing in revision" both appear
    # so the classifier returns ``terminal_missing``, but no quoted id
    # is captured.
    error = "terminal task missing in revision"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert "KEEP THIS TASK VERBATIM" not in out
    assert "task id:" not in out
    assert error in out
    assert _INVARIANTS_POINTER in out


# ---------------------------------------------------------------------------
# terminal_regressed
# ---------------------------------------------------------------------------


def test_terminal_regressed_names_task_and_regressed_status() -> None:
    """``terminal task 'draft_slides' regressed to 'PENDING'`` →
    snippet names BOTH the id and the regressed status it must NOT
    keep, and points back to the invariants block for the canonical
    prior status."""
    error = _PREFIX + "terminal task 'draft_slides' regressed to 'PENDING'"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert error in out
    assert _INVARIANTS_POINTER in out
    assert "RESTORE THIS TASK'S TERMINAL STATUS" in out
    assert 'task id: "draft_slides"' in out
    assert '"PENDING"' in out
    # The snippet itself points back to the invariants block for the
    # actual prior status (we don't reconstruct it from the error).
    assert "STRUCTURAL INVARIANTS block above" in out


def test_terminal_regressed_handles_double_quoted_args() -> None:
    error = 'terminal task "t1" regressed to "RUNNING"'
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert 'task id: "t1"' in out
    assert '"RUNNING"' in out


def test_malformed_terminal_regressed_falls_back_silently() -> None:
    """Bucketed as ``terminal_regressed`` but only the id is quoted, so
    the regex (which requires BOTH id and status quoted) misses → fall
    back to unenriched form."""
    error = "terminal task 't1' regressed to PENDING"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert "RESTORE THIS TASK'S TERMINAL STATUS" not in out
    assert error in out
    assert _INVARIANTS_POINTER in out


# ---------------------------------------------------------------------------
# Unknown / unbucketed errors
# ---------------------------------------------------------------------------


def test_unknown_error_preserves_existing_unenriched_behaviour() -> None:
    """The unknown-rejection-class branch must be byte-identical to
    the current behaviour — existing callers see no change. We pin the
    full output so any accidental regression in the prior structure
    (raw error + invariants pointer + JSON-only directive) surfaces.
    """
    error = "JSON parse failed: Expecting value"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)

    expected = (
        f"{_BASE_PROMPT}\n\n"
        "PREVIOUS ATTEMPT FAILED. The response you just emitted was "
        "rejected by the validator:\n"
        f"    {error}\n\n"
        "Re-read the STRUCTURAL INVARIANTS section above. Emit a "
        "corrected JSON plan that preserves every terminal task and "
        "every terminal->terminal edge verbatim, and does NOT add "
        "any edge from a CANCELLED or FAILED task to a new PENDING "
        "task. Respond with JSON only; no prose, no markdown fences."
    )
    assert out == expected


def test_unknown_error_no_snippet_keys_leak() -> None:
    """Defensive: none of the structural-snippet keywords appear when
    the error is unbucketed."""
    error = "call_llm raised: timeout"
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    assert "ADD THIS EDGE VERBATIM" not in out
    assert "KEEP THIS TASK VERBATIM" not in out
    assert "RESTORE THIS TASK'S TERMINAL STATUS" not in out
    assert "from_task_id" not in out


def test_empty_error_preserves_unenriched_behaviour() -> None:
    """Empty error string buckets to ``None`` (per the classifier's
    contract) → fall back. Doesn't raise."""
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, "")
    assert "ADD THIS EDGE VERBATIM" not in out
    assert "KEEP THIS TASK VERBATIM" not in out
    assert _INVARIANTS_POINTER in out


# ---------------------------------------------------------------------------
# Cross-cutting: pointer remains in every enriched prompt
# ---------------------------------------------------------------------------


def test_invariants_pointer_present_in_all_enriched_prompts() -> None:
    """The structural-invariants pointer is additive — every enriched
    prompt must still contain the existing pointer so the LLM still
    knows to consult the invariants block above."""
    for error in (
        "terminal->terminal edge 'a' -> 'b' missing in revision",
        "terminal task 'x' missing in revision",
        "terminal task 'x' regressed to 'PENDING'",
    ):
        out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
        assert _INVARIANTS_POINTER in out, (
            f"pointer missing for error={error!r}; output was:\n{out}"
        )


def test_snippet_does_not_overmatch_concatenated_errors() -> None:
    """Two edge_missing errors joined into a single string must not
    cause the regex to span across both segments. The non-greedy body
    pattern on the capture group ensures we capture the ids of the
    FIRST quoted pair, not a span that swallows everything between
    error 1's first quote and error 2's last quote."""
    error = (
        "terminal->terminal edge 'a' -> 'b' missing in revision; "
        "terminal->terminal edge 'c' -> 'd' missing in revision"
    )
    out = LLMPlanner._build_correction_prompt(_BASE_PROMPT, error)
    # We capture the FIRST (a, b) pair, not "a' -> 'b' missing ... -> 'c"
    # or similar over-broad match.
    assert '{"from_task_id": "a", "to_task_id": "b"}' in out
    # Sanity: no overshoot to 'd'.
    assert '"to_task_id": "d"' not in out
