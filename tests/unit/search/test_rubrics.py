"""Unit tests for search-slice rubric instances."""

from __future__ import annotations

from research_agent.search.rubrics import SUGGESTION_QUALITY_RUBRIC


def test_suggestion_quality_rubric_has_concrete_checklist() -> None:
    rubric = SUGGESTION_QUALITY_RUBRIC
    assert rubric.name == "suggestion-quality"
    assert len(rubric.criteria) == 4
    assert rubric.examples == ()
    assert rubric.scoring.strip()
    assert len(rubric.fail_conditions) >= 1
    joined = " ".join(rubric.criteria).lower()
    assert "paper" in joined or "abstract" in joined
    assert "actionable" in joined or "next step" in joined
    assert "query" in joined
    assert "read" in joined
    fails = " ".join(rubric.fail_conditions).lower()
    assert "actionable" in fails or "direction" in fails
    assert "paper" in fails or "abstract" in fails
    text = rubric.checklist()
    assert text.startswith("1. ")
    assert "\n4. " in text
