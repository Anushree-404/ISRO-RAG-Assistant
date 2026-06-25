"""
tests/test_retriever.py — Unit tests for the retrieval pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

try:
    from retriever import _reciprocal_rank_fusion, _tokenise
except ImportError as _e:
    pytest.skip(f"Skipping test_retriever: {_e}", allow_module_level=True)


# ── Tokeniser ─────────────────────────────────────────────────────────────────

class TestTokenise:
    def test_basic(self) -> None:
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_empty(self) -> None:
        assert _tokenise("") == []

    def test_multiple_spaces(self) -> None:
        tokens = _tokenise("a  b   c")
        assert "a" in tokens
        assert "b" in tokens


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

class TestRRF:
    def test_single_list(self) -> None:
        ranked = [[0, 1, 2]]
        result = _reciprocal_rank_fusion(ranked)
        indices = [r[0] for r in result]
        # First element should have highest score
        assert indices[0] == 0

    def test_two_lists_agreement(self) -> None:
        # Both lists agree on top item
        ranked = [[0, 1, 2], [0, 2, 1]]
        result = _reciprocal_rank_fusion(ranked)
        assert result[0][0] == 0  # index 0 should win

    def test_two_lists_disagreement(self) -> None:
        # List 1 ranks 0 first, list 2 ranks 1 first
        ranked = [[0, 1, 2], [1, 0, 2]]
        result = _reciprocal_rank_fusion(ranked)
        # Both 0 and 1 should be in top 2
        top_indices = {r[0] for r in result[:2]}
        assert 0 in top_indices
        assert 1 in top_indices

    def test_scores_are_positive(self) -> None:
        ranked = [[0, 1, 2], [2, 1, 0]]
        result = _reciprocal_rank_fusion(ranked)
        for _, score in result:
            assert score > 0

    def test_empty_lists(self) -> None:
        result = _reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_k_parameter(self) -> None:
        ranked = [[0, 1]]
        result_default = _reciprocal_rank_fusion(ranked, k=60)
        result_small_k = _reciprocal_rank_fusion(ranked, k=1)
        # Smaller k → higher scores
        assert result_small_k[0][1] > result_default[0][1]


# ── HybridRetriever (mocked) ──────────────────────────────────────────────────

class TestHybridRetrieverMocked:
    """Test HybridRetriever logic with mocked FAISS and metadata."""

    def _make_retriever(self):  # type: ignore[return]
        from retriever import HybridRetriever

        # Build minimal fake metadata
        metadata = [
            {
                "chunk_id": f"doc_p000{i}_c000",
                "source_file": "doc.pdf",
                "page_number": i + 1,
                "section_title": f"Section {i}",
                "mission_name": "Chandrayaan-3",
                "text": f"Chandrayaan-3 mission detail number {i} about propulsion and landing.",
                "char_count": 60,
            }
            for i in range(20)
        ]

        # Build a real FAISS index with random vectors
        import faiss

        dim = 384
        index = faiss.IndexFlatL2(dim)
        rng = np.random.default_rng(42)
        vecs = rng.random((len(metadata), dim)).astype(np.float32)
        # Normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        index.add(vecs)

        return HybridRetriever(top_k=6, index=index, metadata=metadata)

    def test_retrieve_returns_list(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("What is the propulsion system?")
        assert isinstance(results, list)

    def test_retrieve_top_k(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("Chandrayaan-3 landing", top_k=3)
        assert len(results) <= 3

    def test_retrieve_has_metadata(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("mission detail")
        for r in results:
            assert "source_file" in r
            assert "page_number" in r
            assert "text" in r
            assert "retrieval_score" in r

    def test_retrieve_scores_positive(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("propulsion")
        for r in results:
            assert r["retrieval_score"] > 0
