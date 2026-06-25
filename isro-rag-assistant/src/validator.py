"""
validator.py — Citation verification and hallucination detection.

For every citation in a RAGResponse:
  1. Check that the cited document exists in the retrieved chunks.
  2. Check that the cited page number matches.
  3. Check that the chunk_text is a plausible substring / near-match of the
     actual chunk text (fuzzy match with configurable threshold).

Returns a ValidationReport with per-citation verdicts and an overall
hallucination flag.

Usage:
    from validator import validate_response
    report = validate_response(rag_response, retrieved_chunks)
    print(report.hallucination_detected)
    print(report.verdicts)
"""

from __future__ import annotations

import difflib
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from generator import Citation, RAGResponse
from utils import configure_logging

configure_logging()

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum similarity ratio (0–1) for chunk_text to be considered a valid match
# Gemini tends to paraphrase rather than quote verbatim, so we use a lower threshold
SIMILARITY_THRESHOLD = 0.15


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class CitationVerdict(BaseModel):
    """Verdict for a single citation."""

    citation_index: int = Field(..., description="0-based index in citations list")
    doc: str
    page: int
    is_doc_found: bool = Field(..., description="Cited doc exists in retrieved chunks")
    is_page_found: bool = Field(..., description="Cited page found in matching doc chunks")
    text_similarity: float = Field(
        ..., description="Fuzzy similarity of chunk_text vs best matching chunk"
    )
    is_text_grounded: bool = Field(
        ..., description="chunk_text similarity ≥ threshold"
    )
    is_valid: bool = Field(..., description="All three checks passed")
    reason: str = Field(default="", description="Human-readable explanation if invalid")


class ValidationReport(BaseModel):
    """Full validation report for a RAGResponse."""

    question: str
    answer_snippet: str = Field(..., description="First 200 chars of the answer")
    verdicts: list[CitationVerdict]
    total_citations: int
    valid_citations: int
    hallucination_detected: bool = Field(
        ...,
        description="True if any citation fails validation or answer has no citations",
    )
    hallucination_rate: float = Field(
        ..., description="Fraction of invalid citations (0–1)"
    )
    summary: str


# ── Similarity helper ─────────────────────────────────────────────────────────


def _text_similarity(a: str, b: str) -> float:
    """
    Compute SequenceMatcher similarity between *a* and *b*.
    Both strings are lowercased before comparison.
    """
    return difflib.SequenceMatcher(
        None, a.lower().strip(), b.lower().strip()
    ).ratio()


def _best_similarity(needle: str, haystack_chunks: list[dict[str, Any]]) -> float:
    """Return the highest similarity score between *needle* and any chunk text."""
    if not haystack_chunks:
        return 0.0
    return max(_text_similarity(needle, c["text"]) for c in haystack_chunks)


# ── Core validation ───────────────────────────────────────────────────────────


def _verify_citation(
    idx: int,
    citation: Citation,
    retrieved_chunks: list[dict[str, Any]],
) -> CitationVerdict:
    """Verify a single *citation* against *retrieved_chunks*."""

    # 1. Check document exists
    doc_chunks = [
        c for c in retrieved_chunks
        if c.get("source_file", "").lower() == citation.doc.lower()
    ]
    is_doc_found = len(doc_chunks) > 0

    # 2. Check page number
    page_chunks = [
        c for c in doc_chunks
        if c.get("page_number") == citation.page
    ]
    is_page_found = len(page_chunks) > 0

    # 3. Fuzzy text match — search across all chunks from that doc
    # (page might be off by one due to extraction artefacts, so we search all)
    search_pool = doc_chunks if doc_chunks else retrieved_chunks
    similarity = _best_similarity(citation.chunk_text, search_pool)
    is_text_grounded = similarity >= SIMILARITY_THRESHOLD

    is_valid = is_doc_found and is_page_found and is_text_grounded

    reasons: list[str] = []
    if not is_doc_found:
        reasons.append(f"Document '{citation.doc}' not in retrieved chunks")
    if not is_page_found:
        reasons.append(f"Page {citation.page} not found in '{citation.doc}' chunks")
    if not is_text_grounded:
        reasons.append(
            f"chunk_text similarity {similarity:.2f} < threshold {SIMILARITY_THRESHOLD}"
        )

    return CitationVerdict(
        citation_index=idx,
        doc=citation.doc,
        page=citation.page,
        is_doc_found=is_doc_found,
        is_page_found=is_page_found,
        text_similarity=round(similarity, 4),
        is_text_grounded=is_text_grounded,
        is_valid=is_valid,
        reason="; ".join(reasons),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def validate_response(
    response: RAGResponse,
    retrieved_chunks: list[dict[str, Any]],
    question: str = "",
) -> ValidationReport:
    """
    Validate all citations in *response* against *retrieved_chunks*.

    Args:
        response:         RAGResponse from generator.py.
        retrieved_chunks: Chunks returned by the retriever for the same query.
        question:         Original user question (for the report).

    Returns:
        ValidationReport with per-citation verdicts and hallucination flag.
    """
    verdicts: list[CitationVerdict] = []

    for idx, citation in enumerate(response.citations):
        verdict = _verify_citation(idx, citation, retrieved_chunks)
        verdicts.append(verdict)
        if not verdict.is_valid:
            logger.warning(
                f"Citation {idx} INVALID: {verdict.reason}"
            )

    total = len(verdicts)
    valid = sum(1 for v in verdicts if v.is_valid)
    invalid = total - valid

    # Hallucination if any citation is invalid, OR if answer is non-trivial
    # but has zero citations
    answer_is_substantive = len(response.answer.strip()) > 50
    no_citations = total == 0
    hallucination_detected = (invalid > 0) or (answer_is_substantive and no_citations)

    hallucination_rate = (invalid / total) if total > 0 else (1.0 if answer_is_substantive else 0.0)

    if hallucination_detected:
        summary = (
            f"⚠️  Hallucination detected: {invalid}/{total} citations failed validation."
            if total > 0
            else "⚠️  Answer provided with no citations."
        )
    else:
        summary = f"✅ All {total} citations verified successfully."

    logger.info(summary)

    return ValidationReport(
        question=question,
        answer_snippet=response.answer[:200],
        verdicts=verdicts,
        total_citations=total,
        valid_citations=valid,
        hallucination_detected=hallucination_detected,
        hallucination_rate=round(hallucination_rate, 4),
        summary=summary,
    )


def validate_and_log(
    response: RAGResponse,
    retrieved_chunks: list[dict[str, Any]],
    question: str = "",
) -> ValidationReport:
    """
    Convenience wrapper: validate and print a human-readable report.
    """
    report = validate_response(response, retrieved_chunks, question)

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Question : {question[:80]}")
    print(f"Answer   : {report.answer_snippet[:80]} …")
    print(f"Summary  : {report.summary}")
    print(f"Hallucination rate: {report.hallucination_rate:.1%}")
    print()
    for v in report.verdicts:
        status = "✅" if v.is_valid else "❌"
        print(
            f"  {status} Citation {v.citation_index}: {v.doc} p.{v.page} "
            f"(sim={v.text_similarity:.2f})"
        )
        if v.reason:
            print(f"     Reason: {v.reason}")
    print("=" * 60 + "\n")

    return report
