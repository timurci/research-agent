"""Unit tests for the shared ``Rubric`` value object."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.shared.rubric import Rubric, RubricExample

_SCORING = (
    "Assign 0.0, 0.5, or 1.0: 1.0 all criteria met; "
    "0.5 mixed; 0.0 largely fails. Prefer the lower band when unsure."
)
_FAIL = ("Invents facts not in context.",)


def _rubric(
    *,
    name: str = "demo",
    criteria: tuple[str, ...] = ("A concrete check",),
    scoring: str = _SCORING,
    fail_conditions: tuple[str, ...] = _FAIL,
    examples: tuple[RubricExample, ...] = (),
) -> Rubric:
    return Rubric(
        name=name,
        criteria=criteria,
        scoring=scoring,
        fail_conditions=fail_conditions,
        examples=examples,
    )


def test_rubric_checklist_is_numbered() -> None:
    rubric = _rubric(criteria=("First check", "Second check", "Third check"))
    assert rubric.checklist() == "1. First check\n2. Second check\n3. Third check"


def test_rubric_fail_checklist_is_numbered() -> None:
    rubric = _rubric(fail_conditions=("Mode A", "Mode B"))
    assert rubric.fail_checklist() == "1. Mode A\n2. Mode B"


def test_rubric_defaults_to_empty_examples() -> None:
    assert _rubric().examples == ()


def test_rubric_accepts_few_shot_examples() -> None:
    example = RubricExample(
        input="How do I reset my password?",
        output="Click Settings, then Reset Password. Allow 5 minutes.",
        context="Support policy: always give a timeline.",
        score=1.0,
        passing=True,
        reason="Identifies request, gives a next step, includes a timeline.",
    )
    rubric = _rubric(
        name="support",
        criteria=("Identifies request", "Gives a next step"),
        examples=(example,),
    )
    assert len(rubric.examples) == 1
    assert rubric.examples[0].passing is True
    assert rubric.examples[0].score == 1.0
    assert rubric.examples[0].context.startswith("Support policy")


def test_rubric_example_allows_empty_context() -> None:
    example = RubricExample(
        input="query",
        output="answer",
        score=0.0,
        passing=False,
        reason="Vague deflection.",
    )
    assert example.context == ""


def test_rubric_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        _rubric(name="")


def test_rubric_rejects_empty_criteria_tuple() -> None:
    with pytest.raises(ValidationError):
        _rubric(criteria=())


def test_rubric_rejects_blank_criterion_item() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        _rubric(criteria=("ok", "   "))


def test_rubric_rejects_empty_fail_conditions() -> None:
    with pytest.raises(ValidationError):
        _rubric(fail_conditions=())


def test_rubric_rejects_blank_scoring() -> None:
    with pytest.raises(ValidationError):
        _rubric(scoring="   ")


def test_rubric_example_rejects_blank_reason() -> None:
    with pytest.raises(ValidationError):
        RubricExample(
            input="q",
            output="a",
            score=0.5,
            passing=True,
            reason="",
        )


def test_rubric_example_rejects_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        RubricExample(
            input="q",
            output="a",
            score=1.5,
            passing=True,
            reason="ok",
        )
