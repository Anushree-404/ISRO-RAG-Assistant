"""
tests/test_validator.py — Unit tests for citation validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

try:
    from generator import Citation, RAGResponse
    from validator import (
        SIMILARITY_THRESHOLD,
        ValidationReport,
        _best_similarity,
        _text_similarity,
        validate_response,
    )
except ImportError as _e:
    pytest.skip(f"Skipping test_validator: {_e}", allow_module_level=True)


# ── Text similarity ───────────────────────────────────────────────────────────

class TestTextSimilarity:
    def test_identical_strings(self) -> None:
        assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_strings(self) -> None:
        assert _text_similarity("", "") == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        score = _text_similarity("abc", "xyz")
        assert score < 0.5

    def test_case_insensitive(self) -> None:
        assert _text_similarity("Hello", "hello") == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        score = _text_similarity("Chandrayaan-3 mission", "Chandrayaan-3 landing mission")
        assert score > 0.5


class TestBestSimilarity:
    def test_finds_best_match(self) -> None:
        chunks = [
            {"text": "completely unrelated content"},
            {"text": "Chandrayaan-3 propulsion system uses liquid fuel"},
        ]
        score = _best_similarity("Chandrayaan-3 propulsion", chunks)
        assert score > 0.3

    def test_empty_chunks(self) -> None:
        assert _best_similarity("query", []) == 0.0


# ── validate_response ─────────────────────────────────────────────────────────

def _make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc_p0001_c000",
            "source_file": "chandrayaan3.pdf",
            "page_number": 1,
            "text": "Chandrayaan-3 uses a 440 N liquid apogee motor for propulsion.",
            "mission_name": "Chandrayaan-3",
        },
        {
            "chunk_id": "doc_p0002_c000",
            "source_file": "chandrayaan3.pdf",
            "page_number": 2,
            "text": "The lander module has a mass of approximately 1752 kg.",
            "mission_name": "Chandrayaan-3",
        },
    ]


class TestValidateResponse:
    def test_valid_citation(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="Chandrayaan-3 uses a liquid apogee motor.",
            citations=[
                Citation(
                    doc="chandrayaan3.pdf",
                    page=1,
                    chunk_text="Chandrayaan-3 uses a 440 N liquid apogee motor for propulsion.",
                )
            ],
            confidence_score=0.9,
        )
        report = validate_response(response, chunks, "What propulsion does Chandrayaan-3 use?")
        assert report.total_citations == 1
        assert report.valid_citations == 1
        assert not report.hallucination_detected

    def test_invalid_doc_citation(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="Some answer.",
            citations=[
                Citation(
                    doc="nonexistent.pdf",
                    page=1,
                    chunk_text="Some text",
                )
            ],
            confidence_score=0.5,
        )
        report = validate_response(response, chunks)
        assert report.hallucination_detected
        assert report.valid_citations == 0

    def test_wrong_page_citation(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="The lander has a mass of 1752 kg.",
            citations=[
                Citation(
                    doc="chandrayaan3.pdf",
                    page=99,  # wrong page
                    chunk_text="The lander module has a mass of approximately 1752 kg.",
                )
            ],
            confidence_score=0.8,
        )
        report = validate_response(response, chunks)
        # Page not found → invalid
        assert report.verdicts[0].is_page_found is False

    def test_no_citations_with_substantive_answer(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="Chandrayaan-3 is a very important mission with many scientific objectives.",
            citations=[],
            confidence_score=0.7,
        )
        report = validate_response(response, chunks)
        assert report.hallucination_detected
        assert report.hallucination_rate == 1.0

    def test_empty_answer_no_citations(self) -> None:
        response = RAGResponse(
            answer="No.",
            citations=[],
            confidence_score=0.1,
        )
        report = validate_response(response, [])
        # Short answer with no citations — not flagged as hallucination
        assert report.hallucination_detected is False

    def test_multiple_citations_mixed(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="Chandrayaan-3 uses liquid propulsion and has a 1752 kg lander.",
            citations=[
                Citation(
                    doc="chandrayaan3.pdf",
                    page=1,
                    chunk_text="Chandrayaan-3 uses a 440 N liquid apogee motor for propulsion.",
                ),
                Citation(
                    doc="fake.pdf",  # invalid
                    page=5,
                    chunk_text="Some hallucinated text",
                ),
            ],
            confidence_score=0.8,
        )
        report = validate_response(response, chunks)
        assert report.total_citations == 2
        assert report.valid_citations == 1
        assert report.hallucination_detected
        assert report.hallucination_rate == pytest.approx(0.5)

    def test_report_fields(self) -> None:
        chunks = _make_chunks()
        response = RAGResponse(
            answer="Test answer.",
            citations=[],
            confidence_score=0.5,
        )
        report = validate_response(response, chunks, "Test question?")
        assert isinstance(report, ValidationReport)
        assert report.question == "Test question?"
        assert isinstance(report.summary, str)
        assert 0.0 <= report.hallucination_rate <= 1.0
