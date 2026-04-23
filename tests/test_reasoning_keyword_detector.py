"""Unit tests for :func:`goldfive.drift.reasoning.detect_unreferenced_keyword`.

The standalone detector promotes the ``_has_unreferenced_keyword``
helper to a first-class signal because whole-block cosine empirically
fails to separate drifted from on-topic reasoning on real embedding
models (see #223).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import reasoning as dreason  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _session_with_task(
    task_id: str = "t1",
    *,
    title: str = "Research solar panels for a presentation",
    description: str = "Slideshow on solar panels; cover efficiency types market",
    goals: list[Goal] | None = None,
) -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=task_id, title=title, description=description)],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=goals or [Goal(id="g1", summary="solar panels presentation")],
        plan=plan,
        current_task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Core behaviour — severity laddered by surplus keyword count.
# ---------------------------------------------------------------------------


def test_fires_on_single_surplus_keyword() -> None:
    session = _session_with_task()
    # Every 5+ char non-stopword token except "raccoons" is a substring
    # of the goals+task reference ("solar panels presentation" +
    # "Research solar panels for a presentation" + "Slideshow on solar
    # panels; cover efficiency types market"). "raccoons" alone is
    # surplus -> INFO severity.
    text = "research solar panels slideshow presentation raccoons slide"
    drift = dreason.detect_unreferenced_keyword(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    assert drift.severity is DriftSeverity.INFO
    assert "raccoons" in drift.detail
    assert drift.raw == text
    assert drift.current_task_id == "t1"


def test_escalates_severity_with_keyword_count() -> None:
    session = _session_with_task()
    # Four surplus 5+ char non-stopword tokens absent from goals+task:
    # raccoons, habitat, species, behaviour. Threshold for CRITICAL is 4.
    # The other 5+ char tokens are all substrings of the reference.
    text = (
        "slide research solar panels presentation raccoons habitat "
        "species behaviour"
    )
    drift = dreason.detect_unreferenced_keyword(text, session)
    assert drift is not None
    assert drift.severity is DriftSeverity.CRITICAL
    assert "raccoons" in drift.detail
    # First three surplus keywords are quoted verbatim, remainder are
    # reported as "+N more" so operators see why severity escalated.
    assert "(+1 more)" in drift.detail


def test_escalates_to_warning_with_two_surplus_keywords() -> None:
    session = _session_with_task()
    # Two surplus tokens ("raccoons", "habitat"); all others sit inside
    # the reference as substrings.
    text = "research solar panels slideshow raccoons habitat presentation"
    drift = dreason.detect_unreferenced_keyword(text, session)
    assert drift is not None
    assert drift.severity is DriftSeverity.WARNING


def test_no_fire_when_keywords_match_goals() -> None:
    session = _session_with_task()
    # Every 5+ char non-stopword token is in goals+task reference.
    text = "solar panels presentation efficiency market types slideshow"
    assert dreason.detect_unreferenced_keyword(text, session) is None


def test_one_shot_per_task() -> None:
    session = _session_with_task()
    text = "talk about raccoons and their habitat on slide two"
    first = dreason.detect_unreferenced_keyword(text, session)
    assert first is not None
    # Second call for the SAME task is a no-op — avoids drift-spam when
    # the same off-topic reasoning block repeats across turns.
    second = dreason.detect_unreferenced_keyword(text, session)
    assert second is None
    # Switching to a new task re-enables firing.
    session.plan.tasks.append(
        Task(id="t2", title="Research solar panels", description="Slideshow")
    )
    session.current_task_id = "t2"
    third = dreason.detect_unreferenced_keyword(text, session)
    assert third is not None


def test_stopword_filtering() -> None:
    # Text contains only stopwords + task-matching words. No 5+ char
    # non-stopword token is surplus, so the detector stays silent.
    session = _session_with_task()
    text = "these could show those from solar panels and presentation"
    assert dreason.detect_unreferenced_keyword(text, session) is None


def test_empty_reasoning_no_fire() -> None:
    session = _session_with_task()
    assert dreason.detect_unreferenced_keyword("", session) is None


def test_no_fire_when_no_reference() -> None:
    # No goals and no bound current task -> nothing to compare against.
    session = Session(run_id="r1", goals=[], plan=None, current_task_id="")
    text = "arbitrary reasoning with surplus keywords raccoons habitat"
    assert dreason.detect_unreferenced_keyword(text, session) is None


def test_detail_caps_preview_at_three_keywords() -> None:
    session = _session_with_task()
    text = (
        "raccoons habitat species behaviour climate weather zoology "
        "ecology anatomy"
    )
    drift = dreason.detect_unreferenced_keyword(text, session)
    assert drift is not None
    # First three keywords are named, the rest summarised as +N more.
    assert "raccoons, habitat, species" in drift.detail
    assert "+" in drift.detail and "more" in drift.detail


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_fires_unreferenced_keyword_when_off_topic_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the embedding model is unavailable (so ``detect_off_topic``
    and ``detect_intent_divergence``'s embedding path stay quiet), the
    lexical keyword detector still fires via ``analyze_reasoning``.

    This is the #223 raccoon regression: whole-block cosine missed the
    drift on real models; the keyword path owns the signal.
    """
    from goldfive.drift import _embed as _embed_mod
    from goldfive.drift._embed import set_model

    set_model(None)
    monkeypatch.setattr(_embed_mod, "_MODEL_UNAVAILABLE", True, raising=False)
    session = _session_with_task()
    text = (
        "The user wants research on solar panels for a presentation. "
        "Slide 1: Solar Panels. Slide 2: Raccoons (habitat, diet, "
        "behaviour). Let me compile comprehensive info."
    )
    drift = dreason.analyze_reasoning(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    # At least raccoons, habitat, behaviour are surplus -> WARNING or
    # CRITICAL depending on whether "diet" (4 chars, skipped) and
    # "compile" / "comprehensive" / "presentation" land in the
    # reference. We only assert severity is at least WARNING.
    assert drift.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)


def test_pipeline_runs_unreferenced_keyword_after_off_topic() -> None:
    """``detect_unreferenced_keyword`` runs AFTER ``detect_off_topic``
    and BEFORE ``detect_reasoning_cluster_tightening``. Verify by
    isolating the detector call against a session where whole-text
    embedding is unavailable — keyword-only signal must still fire.
    """
    from goldfive.drift import _embed as _embed_mod
    from goldfive.drift._embed import set_model

    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True
    session = _session_with_task()
    text = (
        "Slide 1: solar panels. Slide 2: raccoons habitat species "
        "behaviour zoology climate."
    )
    drift = dreason.analyze_reasoning(text, session)
    assert drift is not None
    assert drift.kind is DriftKind.OFF_TOPIC
    # Keyword detector's diagnostic contains "off-task keywords:".
    assert "off-task keywords" in drift.detail


# ---------------------------------------------------------------------------
# Session attribute plumbing
# ---------------------------------------------------------------------------


def test_session_attribute_default_is_empty_set() -> None:
    session = _session_with_task()
    assert session.unreferenced_keyword_flagged == set()


def test_detect_populates_session_flag_set() -> None:
    session = _session_with_task()
    text = "discuss raccoons in the deck"
    dreason.detect_unreferenced_keyword(text, session)
    assert "t1" in session.unreferenced_keyword_flagged


# ---------------------------------------------------------------------------
# Consistency with ``_has_unreferenced_keyword``
# ---------------------------------------------------------------------------


def test_detector_matches_helper_token_rule() -> None:
    """The standalone detector and the existing ``_has_unreferenced_keyword``
    helper must share the same 5+ char / stopword rule so the two paths
    stay behaviourally consistent (severity-bump vs standalone).
    """
    session = _session_with_task()
    # "raccoons" is 8 chars, clearly not a stopword, absent from goals +
    # task -> both the helper and the detector should treat it as a
    # surplus keyword.
    goals = "solar panels presentation"
    task = "solar panels slideshow efficiency"
    text = "mention raccoons please"
    assert dreason._has_unreferenced_keyword(text, goals, task) is True
    drift = dreason.detect_unreferenced_keyword(text, session)
    assert drift is not None
    assert "raccoons" in drift.detail


def _clear_embedding_model() -> Any:
    """Ensure no custom encoder leaks between tests in this module."""
    from goldfive.drift import _embed as _embed_mod
    from goldfive.drift._embed import set_model

    set_model(None)
    _embed_mod._MODEL_UNAVAILABLE = True


@pytest.fixture(autouse=True)
def _reset_embed_model_between_tests() -> Any:
    _clear_embedding_model()
    yield
    _clear_embedding_model()
