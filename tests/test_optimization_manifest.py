"""Tests for the optimization manifest + prompt loader.

Covers:

* The bundled ``manifest.toml`` parses and self-validates.
* Every prompt mutation's source path resolves to a shipped markdown
  file whose loaded body matches the live Python attribute (drift /
  planner / goal-deriver source-of-truth — catches the manifest going
  stale w.r.t. the code).
* Every numeric mutation's ``default`` matches the live Python
  attribute at the named path.
* :meth:`Manifest.validate` accepts well-formed updates and rejects
  the documented failure modes.
* The prompt loader caches, honours :func:`bind` overrides, and
  resets cleanly.
"""

from __future__ import annotations

import importlib
import textwrap

import pytest

from goldfive.optimization import (
    Manifest,
    bind_prompt,
    load_prompt,
    reset_prompts,
)
from goldfive.optimization.manifest import ManifestLoadError
from goldfive.optimization.prompts import (
    PromptNotFound,
    available_prompts,
)

# ---------------------------------------------------------------------------
# Manifest structural shape
# ---------------------------------------------------------------------------


def test_load_parses_bundled_manifest() -> None:
    manifest = Manifest.load()
    assert len(manifest) >= 10
    assert len(set(manifest.ids())) == len(manifest)


def test_package_reexports_public_exception_types() -> None:
    """The exceptions the public API raises are importable from the package.

    ``Manifest.load`` raises :class:`ManifestLoadError` and
    ``load_prompt`` raises :class:`PromptNotFound`; a caller that imports
    ``Manifest`` / ``load_prompt`` from the package must be able to catch
    those without reaching into a submodule.
    """
    import goldfive.optimization as opt

    assert opt.ManifestLoadError is ManifestLoadError
    assert opt.PromptNotFound is PromptNotFound
    assert "ManifestLoadError" in opt.__all__
    assert "PromptNotFound" in opt.__all__


def test_manifest_covers_required_mutation_kinds() -> None:
    """Manifest must include the prompts + thresholds the brief calls out."""
    manifest = Manifest.load()
    required_ids = {
        # Prompts the brief names explicitly.
        "reasoning_judge_system_prompt",
        "reasoning_judge_user_prompt",
        "goal_drift_system_prompt",
        "goal_drift_user_prompt",
        "reflective_check_system_prompt",
        "reflective_check_user_prompt",
        "refine_system_prompt",
        # Tier 1A example threshold.
        "looping_reasoning_similarity_threshold",
    }
    missing = required_ids - set(manifest.ids())
    assert not missing, f"manifest missing required entries: {sorted(missing)}"


def test_manifest_covers_expansion_entries() -> None:
    """manifest-and-decision-telemetry expansion targets must be present."""
    manifest = Manifest.load()
    required_ids = {
        # Plan-template fragments extracted in the expansion.
        "plan_template_supersession_invariant",
        "plan_template_supersession_examples",
        "plan_template_refinement_guidance",
        # Planner retry budgets / caps.
        "planner_default_max_refine_attempts",
        "planner_max_output_tokens",
        "goal_deriver_max_output_tokens",
        # Steerer ladder policy knobs.
        "refine_failure_threshold",
        "progress_stall_threshold_seconds",
        "executor_max_nudge_replays",
        # Reasoning-judge LLM dispatch caps.
        "reasoning_judge_max_reasoning_input_chars",
        "reasoning_judge_max_raw_response_chars",
        "reasoning_judge_max_output_tokens",
        # Trajectory-level goal-drift caps.
        "goal_drift_max_output_tokens",
        "goal_drift_trigger_input_max_chars",
        # Adapter watcher knobs.
        "default_llm_call_timeout_ms",
    }
    missing = required_ids - set(manifest.ids())
    assert not missing, f"expansion entries missing: {sorted(missing)}"


def test_manifest_size_target() -> None:
    """manifest-and-decision-telemetry expansion brings the total to 60+."""
    manifest = Manifest.load()
    assert len(manifest) >= 60, (
        f"manifest has {len(manifest)} entries; expected >= 60 per the "
        "expansion target (was ~31 before the manifest-and-decision-"
        "telemetry expansion)."
    )


def test_manifest_filter_by_tag() -> None:
    manifest = Manifest.load()
    judge = manifest.filter_by_tag("judge")
    assert len(judge) >= 3
    for mut in judge:
        assert "judge" in mut.tags


def test_manifest_prompt_entries_and_shipped_prompts_are_a_bijection() -> None:
    """Every shipped prompt has exactly one manifest entry and vice versa.

    ``test_prompt_mutations_match_live_python_attrs`` already catches a
    manifest entry pointing at a missing prompt (``load_prompt`` raises).
    The reverse direction is the gap this test closes: a markdown file
    added to :func:`goldfive.optimization.prompts.available_prompts`
    without a paired manifest entry would otherwise go unnoticed — the
    optimizer would see a prompt it cannot resolve a ``python_attr`` for.
    """
    manifest = Manifest.load()
    manifest_prompt_names = {
        mut.source.rsplit("/", 1)[1].removesuffix(".md")
        for mut in manifest
        if mut.kind == "prompt"
    }
    shipped = set(available_prompts())
    assert manifest_prompt_names == shipped, (
        "manifest prompt entries and shipped prompts diverged — "
        f"shipped-only: {sorted(shipped - manifest_prompt_names)}, "
        f"manifest-only: {sorted(manifest_prompt_names - shipped)}"
    )


# ---------------------------------------------------------------------------
# Manifest reflects live code values
# ---------------------------------------------------------------------------


def _resolve_python_attr(python_attr: str) -> object:
    module_path, _, attr_path = python_attr.partition(":")
    module = importlib.import_module(module_path)
    obj: object = module
    for piece in attr_path.split("."):
        obj = getattr(obj, piece)
    return obj


def test_prompt_mutations_match_live_python_attrs() -> None:
    """Every prompt mutation's markdown body must equal the live Python attr.

    Catches the manifest going stale w.r.t. the drift / planner /
    goal-deriver source-of-truth: a developer who edits a prompt
    constant in Python without re-syncing the markdown copy fails this
    test.
    """
    manifest = Manifest.load()
    drifts: list[tuple[str, object, str]] = []
    for mut in manifest:
        if mut.kind != "prompt":
            continue
        # source is "goldfive/optimization/prompts/<name>.md".
        name = mut.source.rsplit("/", 1)[1].removesuffix(".md")
        body = load_prompt(name)
        live = _resolve_python_attr(mut.python_attr)
        if body != live:
            drifts.append((mut.id, live, body))
    if drifts:
        ids = [d[0] for d in drifts]
        pytest.fail(
            "prompt body diverged from live Python attribute for: "
            + ", ".join(ids)
            + " (regenerate the markdown copy or update the manifest)"
        )


def test_numeric_mutations_match_live_python_attrs() -> None:
    """Every numeric mutation's ``default`` must equal the live attr value."""
    manifest = Manifest.load()
    drifts: list[tuple[str, object, object]] = []
    for mut in manifest:
        if mut.kind != "numeric":
            continue
        live = _resolve_python_attr(mut.python_attr)
        if mut.type == "int":
            if int(live) != int(mut.default):  # type: ignore[arg-type]
                drifts.append((mut.id, live, mut.default))
        else:
            if float(live) != float(mut.default):  # type: ignore[arg-type]
                drifts.append((mut.id, live, mut.default))
    if drifts:
        pytest.fail(
            "numeric default diverged from live Python attribute for: "
            + ", ".join(f"{i} (code={c!r}, manifest={m!r})" for i, c, m in drifts)
        )


# ---------------------------------------------------------------------------
# Manifest.validate
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed_updates() -> None:
    manifest = Manifest.load()
    # A numeric within range + a prompt with every placeholder present.
    valid_prompt = load_prompt("reasoning_judge_user")
    errors = manifest.validate(
        {
            "looping_reasoning_similarity_threshold": 0.85,
            "reasoning_judge_user_prompt": valid_prompt,
        }
    )
    assert errors == []


def test_validate_rejects_unknown_id() -> None:
    manifest = Manifest.load()
    errors = manifest.validate({"this_does_not_exist": 0.5})
    assert len(errors) == 1
    assert errors[0].code == "unknown_id"
    assert errors[0].mutation_id == "this_does_not_exist"


def test_validate_rejects_numeric_out_of_range() -> None:
    manifest = Manifest.load()
    errors = manifest.validate(
        {"looping_reasoning_similarity_threshold": 2.0}
    )
    assert len(errors) == 1
    assert errors[0].code == "out_of_range"


def test_validate_rejects_numeric_type_mismatch() -> None:
    manifest = Manifest.load()
    errors = manifest.validate(
        {"tool_loop_exact_threshold": 3.7}
    )
    assert len(errors) == 1
    assert errors[0].code == "type"


def test_validate_rejects_boolean_for_numeric() -> None:
    """``bool`` subclasses ``int``; manifest must refuse it explicitly."""
    manifest = Manifest.load()
    errors = manifest.validate(
        {"tool_loop_exact_threshold": True}  # type: ignore[dict-item]
    )
    assert len(errors) == 1
    assert errors[0].code == "type"


def test_validate_accepts_float_with_integer_value_for_int_knob() -> None:
    """``3.0`` is a valid value for an int knob (round-trips through JSON)."""
    manifest = Manifest.load()
    errors = manifest.validate(
        {"tool_loop_exact_threshold": 3.0}
    )
    assert errors == []


def test_validate_rejects_prompt_missing_placeholder() -> None:
    manifest = Manifest.load()
    # The reasoning-judge user prompt requires {plan_tasks_summary}, etc.
    # Drop them all.
    errors = manifest.validate(
        {"reasoning_judge_user_prompt": "Just decide on_task or off_task."}
    )
    # One error per missing placeholder.
    placeholder_errors = [
        e for e in errors if e.code == "missing_placeholder"
    ]
    assert len(placeholder_errors) > 0
    for err in placeholder_errors:
        assert err.mutation_id == "reasoning_judge_user_prompt"


def test_validate_rejects_empty_prompt_body() -> None:
    manifest = Manifest.load()
    errors = manifest.validate({"reasoning_judge_system_prompt": "   \n  "})
    assert any(e.code == "empty_body" for e in errors)


def test_validate_rejects_non_string_prompt_body() -> None:
    manifest = Manifest.load()
    errors = manifest.validate(
        {"reasoning_judge_system_prompt": 42}  # type: ignore[dict-item]
    )
    assert any(e.code == "type" for e in errors)


def test_validate_batches_multiple_errors() -> None:
    """One call returns errors for every failing entry — not just the first."""
    manifest = Manifest.load()
    errors = manifest.validate(
        {
            "totally_unknown_id": 0.0,
            "looping_reasoning_similarity_threshold": 99.0,
        }
    )
    codes = sorted(e.code for e in errors)
    assert codes == ["out_of_range", "unknown_id"]


# ---------------------------------------------------------------------------
# Loader hand-rolled error paths
# ---------------------------------------------------------------------------


def test_from_text_rejects_malformed_toml() -> None:
    with pytest.raises(ManifestLoadError):
        Manifest.from_text("this is not = toml [")


def test_from_text_rejects_duplicate_ids() -> None:
    with pytest.raises(ManifestLoadError, match="duplicate"):
        Manifest.from_text(
            textwrap.dedent(
                """
                [[mutation]]
                id = "dup"
                kind = "numeric"
                source = "goldfive/drift/reasoning.py:OFF_TOPIC_DISTANCE_THRESHOLD"
                python_attr = "goldfive.drift.reasoning:OFF_TOPIC_DISTANCE_THRESHOLD"
                description = ""
                type = "float"
                range = [0.0, 1.0]
                default = 0.5

                [[mutation]]
                id = "dup"
                kind = "numeric"
                source = "goldfive/drift/reasoning.py:OFF_TOPIC_DISTANCE_THRESHOLD"
                python_attr = "goldfive.drift.reasoning:OFF_TOPIC_DISTANCE_THRESHOLD"
                description = ""
                type = "float"
                range = [0.0, 1.0]
                default = 0.5
                """
            )
        )


def test_from_text_rejects_default_outside_range() -> None:
    with pytest.raises(ManifestLoadError, match="outside declared range"):
        Manifest.from_text(
            textwrap.dedent(
                """
                [[mutation]]
                id = "bad"
                kind = "numeric"
                source = "goldfive/drift/reasoning.py:OFF_TOPIC_DISTANCE_THRESHOLD"
                python_attr = "goldfive.drift.reasoning:OFF_TOPIC_DISTANCE_THRESHOLD"
                description = ""
                type = "float"
                range = [0.0, 1.0]
                default = 2.5
                """
            )
        )


def test_from_text_rejects_invalid_source_path() -> None:
    with pytest.raises(ManifestLoadError, match="prompt source path"):
        Manifest.from_text(
            textwrap.dedent(
                """
                [[mutation]]
                id = "bad_path"
                kind = "prompt"
                source = "this is not a path"
                python_attr = "goldfive.drift.goals:GOAL_DRIFT_SYSTEM_PROMPT"
                description = ""
                required_placeholders = []
                """
            )
        )


# ---------------------------------------------------------------------------
# Mutation dataclass
# ---------------------------------------------------------------------------


def test_mutation_dataclass_is_frozen() -> None:
    manifest = Manifest.load()
    mut = next(iter(manifest))
    with pytest.raises(dataclasses_attribute_error_types()):
        mut.id = "tampered"  # type: ignore[misc]


def dataclasses_attribute_error_types() -> tuple[type[BaseException], ...]:
    """Both 3.11+ and older raise different exceptions for frozen mutation."""
    from dataclasses import FrozenInstanceError

    return (FrozenInstanceError, AttributeError)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


def test_load_prompt_returns_canonical_text() -> None:
    reset_prompts()
    body = load_prompt("reasoning_judge_system")
    assert "single JSON object" in body
    assert not body.endswith("\n")


def test_load_prompt_caches() -> None:
    """Two loads of the same name return the SAME object identity (cached)."""
    reset_prompts()
    first = load_prompt("goal_drift_user")
    second = load_prompt("goal_drift_user")
    # str is interned across calls only when cached deliberately; check
    # via ``is`` for true cache identity.
    assert first is second


def test_bind_overrides_load() -> None:
    reset_prompts()
    custom = "OVERRIDDEN BODY"
    bind_prompt("reasoning_judge_system", custom)
    assert load_prompt("reasoning_judge_system") == custom


def test_reset_clears_overrides_and_cache() -> None:
    bind_prompt("goal_drift_system", "OVERRIDE")
    assert load_prompt("goal_drift_system") == "OVERRIDE"
    reset_prompts("goal_drift_system")
    body = load_prompt("goal_drift_system")
    assert body != "OVERRIDE"
    assert "single JSON object" in body


def test_reset_all_clears_every_override() -> None:
    bind_prompt("goal_drift_system", "A")
    bind_prompt("goal_drift_user", "B")
    reset_prompts()
    assert load_prompt("goal_drift_system") != "A"
    assert load_prompt("goal_drift_user") != "B"


def test_load_prompt_rejects_unknown_name() -> None:
    with pytest.raises(PromptNotFound):
        load_prompt("this_prompt_does_not_exist")


def test_available_prompts_is_sorted_and_complete() -> None:
    names = available_prompts()
    assert names == tuple(sorted(names))
    assert "reasoning_judge_system" in names
    assert "goal_drift_user" in names


# ---------------------------------------------------------------------------
# Manifest mutation flow (the integration the brief asks for: a
# proposed update that passes :meth:`validate` is then installable via
# :func:`bind_prompt` and observable through :func:`load_prompt`).
# ---------------------------------------------------------------------------


def test_validated_prompt_update_is_installable_via_bind() -> None:
    manifest = Manifest.load()
    reset_prompts()
    # Build a valid replacement: keep every required placeholder.
    template = load_prompt("goal_drift_user")
    new_body = "PREFIX::\n" + template
    errors = manifest.validate({"goal_drift_user_prompt": new_body})
    assert errors == []
    bind_prompt("goal_drift_user", new_body)
    assert load_prompt("goal_drift_user") == new_body
    reset_prompts()
    # Now the loader returns the unmodified canonical body.
    assert load_prompt("goal_drift_user") == template
