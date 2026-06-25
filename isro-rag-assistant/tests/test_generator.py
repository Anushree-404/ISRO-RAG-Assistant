"""
tests/test_generator.py — Unit tests for generator utilities (no API calls).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Lazy import — skip entire module if heavy deps not installed
try:
    from generator import (
        Citation,
        RAGResponse,
        _extract_json,
        _format_context,
    )
except ImportError as _e:
    pytest.skip(f"Skipping test_generator: {_e}", allow_module_level=True)


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_clean_json(self) -> None:
        raw = '{"answer": "test", "citations": [], "confidence_score": 0.8}'
        result = _extract_json(raw)
        assert result["answer"] == "test"
        assert result["confidence_score"] == 0.8

    def test_json_with_markdown_fence(self) -> None:
        raw = '```json\n{"answer": "hello", "citations": [], "confidence_score": 0.5}\n```'
        result = _extract_json(raw)
        assert result["answer"] == "hello"

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here is the response: {"answer": "embedded", "citations": [], "confidence_score": 0.6}'
        result = _extract_json(raw)
        assert result["answer"] == "embedded"

    def test_invalid_json_returns_fallback(self) -> None:
        raw = "This is not JSON at all."
        result = _extract_json(raw)
        assert "answer" in result
        assert result["citations"] == []

    def test_nested_citations(self) -> None:
        raw = json_str = (
            '{"answer": "Chandrayaan-3 uses liquid fuel.", '
            '"citations": [{"doc": "c3.pdf", "page": 1, "chunk_text": "liquid fuel"}], '
            '"confidence_score": 0.9}'
        )
        result = _extract_json(raw)
        assert len(result["citations"]) == 1
        assert result["citations"][0]["doc"] == "c3.pdf"


# ── _format_context ───────────────────────────────────────────────────────────

class TestFormatContext:
    def _make_chunk(self, i: int) -> dict:
        return {
            "source_file": f"doc{i}.pdf",
            "page_number": i,
            "mission_name": "Chandrayaan-3",
            "text": f"This is chunk number {i} with some content.",
        }

    def test_single_chunk(self) -> None:
        chunks = [self._make_chunk(1)]
        result = _format_context(chunks)
        assert "CHUNK 1" in result
        assert "doc1.pdf" in result
        assert "Page: 1" in result

    def test_multiple_chunks_separated(self) -> None:
        chunks = [self._make_chunk(i) for i in range(1, 4)]
        result = _format_context(chunks)
        assert "CHUNK 1" in result
        assert "CHUNK 2" in result
        assert "CHUNK 3" in result
        assert "---" in result

    def test_empty_chunks(self) -> None:
        result = _format_context([])
        assert result == ""


# ── RAGResponse schema ────────────────────────────────────────────────────────

class TestRAGResponse:
    def test_valid_response(self) -> None:
        r = RAGResponse(
            answer="Test answer",
            citations=[Citation(doc="a.pdf", page=1, chunk_text="excerpt")],
            confidence_score=0.85,
        )
        assert r.answer == "Test answer"
        assert len(r.citations) == 1
        assert r.confidence_score == pytest.approx(0.85)

    def test_confidence_clamped_above_1(self) -> None:
        r = RAGResponse(answer="x", citations=[], confidence_score=1.5)
        assert r.confidence_score == pytest.approx(1.0)

    def test_confidence_clamped_below_0(self) -> None:
        r = RAGResponse(answer="x", citations=[], confidence_score=-0.5)
        assert r.confidence_score == pytest.approx(0.0)

    def test_confidence_invalid_type(self) -> None:
        r = RAGResponse(answer="x", citations=[], confidence_score="high")  # type: ignore
        assert r.confidence_score == pytest.approx(0.5)

    def test_default_timing_fields(self) -> None:
        r = RAGResponse(answer="x", citations=[], confidence_score=0.5)
        assert r.retrieval_time_ms == 0.0
        assert r.generation_time_ms == 0.0
