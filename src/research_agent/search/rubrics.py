"""Search-slice rubrics for subjective quality.

Layer: Domain.

Slice-specific ``Rubric`` instances. The ``Rubric`` type lives in
``research_agent.shared.rubric``. Length is scored by code metrics, not
by these rubrics.
"""

from typing import Final

from research_agent.shared.rubric import Rubric

SUGGESTION_QUALITY_RUBRIC: Final[Rubric] = Rubric(
    name="suggestion-quality",
    criteria=(
        (
            "Paper-grounded: every substantive claim and recommendation is "
            "supported by the provided paper titles or abstracts "
            "(no fabricated findings or methods)."
        ),
        (
            "Actionable direction: gives concrete next steps a human "
            "researcher can take (what to read, try, or prioritize), not a "
            "generic call for further study or a full literature review."
        ),
        (
            "Query-relevant: organises the direction around the research "
            "query and uses the papers as evidence for that question."
        ),
        (
            "Readable: clear, scannable structure so the reader can grasp "
            "the direction with low cognitive effort."
        ),
    ),
    scoring=(
        "Assign exactly one of 0.0, 0.5, or 1.0 from the quality criteria "
        "(not from fail conditions):\n"
        "- 1.0: substantially meets all four criteria.\n"
        "- 0.5: mixed — some criteria clearly met, others weak or missing.\n"
        "- 0.0: largely fails the criteria or is unusable as a research "
        "direction.\n"
        "When between two bands, choose the lower score. Do not invent "
        "other numeric values or average partial points."
    ),
    fail_conditions=(
        (
            "Refers to a paper, finding, method, or result that is not "
            "present in the provided titles or abstracts."
        ),
        (
            "Provides no actionable reading or research direction "
            "(only vague encouragement, a bare paper list, or a literature "
            "review without next steps)."
        ),
    ),
)
