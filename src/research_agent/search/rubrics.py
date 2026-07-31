"""Search-slice rubrics for subjective quality.

Layer: Domain.

Slice-specific ``Rubric`` instances and the text formatters that render
domain objects as judge task fields. The ``Rubric`` type lives in
``research_agent.shared.rubric``. Length is scored by code metrics, not
by these rubrics.
"""

from typing import TYPE_CHECKING, Final

from research_agent.shared.rubric import Rubric

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery

JUDGE_ABSTRACT_CHARS: Final[int] = 500

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


def format_suggestion_judge_input(query: ResearchQuery) -> str:
    """Format a research query as judge task input text."""
    if query.domains:
        domains = ", ".join(query.domains)
        return f"Query: {query.text}\nDomains: {domains}"
    return f"Query: {query.text}"


def format_suggestion_judge_context(papers: list[PaperInfo]) -> str:
    """Format paper titles and abstract snippets as judge context.

    Abstracts are truncated to ``JUDGE_ABSTRACT_CHARS``.
    """
    if not papers:
        return "(no papers provided)"
    blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        abstract = paper.abstract[:JUDGE_ABSTRACT_CHARS]
        blocks.append(
            f"[{index}] Title: {paper.title}\nAbstract: {abstract}",
        )
    return "\n\n".join(blocks)
