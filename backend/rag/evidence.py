"""A transparent evidence gate, not a probability or a factual-support verifier."""

from pydantic import BaseModel, Field

from backend.rag.lexical import contains_literal, query_tokens, required_literals


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str
    keyword_coverage: float = Field(ge=0, le=1)
    required_coverage: float = Field(ge=0, le=1)
    supporting_chunk_ids: list[str] = Field(default_factory=list)


def assess_evidence(
    question: str, chunks: list[dict], minimum_coverage: float
) -> EvidenceAssessment:
    """Require literal constraints and question-keyword coverage within a passage.

    A conservative local rule prevents unrelated nearest neighbors from triggering
    generation. Paraphrases with no shared vocabulary can be rejected; evaluate and
    tune this policy on labeled queries instead of treating distances as confidence.
    """
    terms = query_tokens(question)
    literals = required_literals(question)
    best = 0.0
    supported = []
    for chunk in chunks:
        text = chunk["text"]
        coverage = sum(contains_literal(text, term) for term in terms) / max(1, len(terms))
        best = max(best, coverage)
        if (
            terms
            and coverage >= minimum_coverage
            and all(contains_literal(text, literal) for literal in literals)
        ):
            supported.append(chunk["chunk_id"])
    reason = (
        "keyword_and_literal_support"
        if supported
        else ("no_retrieved_passages" if not chunks else "insufficient_keyword_or_literal_support")
    )
    return EvidenceAssessment(
        sufficient=bool(supported),
        reason=reason,
        keyword_coverage=best,
        required_coverage=minimum_coverage,
        supporting_chunk_ids=supported,
    )
