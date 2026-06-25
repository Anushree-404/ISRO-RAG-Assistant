"""
retriever.py — Hybrid retrieval: FAISS vector search + BM25 keyword search,
merged with Reciprocal Rank Fusion (RRF).

Returns top-6 chunks with full metadata.

Usage (as a module):
    from retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("What is the propulsion system of Chandrayaan-3?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from embed import get_model, load_index, load_metadata
from utils import configure_logging

configure_logging()

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TOP_K = 6
RRF_K = 60  # RRF constant — controls rank smoothing


# ── Tokeniser for BM25 ────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists of document indices using RRF.

    Args:
        ranked_lists: Each inner list is a ranking of chunk indices
                      (best first).
        k:            RRF smoothing constant.

    Returns:
        List of (chunk_index, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS dense retrieval with BM25 sparse retrieval via RRF.

    Attributes:
        top_k:    Number of final results to return.
        index:    Loaded FAISS index.
        metadata: Parallel list of chunk metadata dicts.
        bm25:     BM25Okapi instance built from chunk texts.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        index: faiss.IndexFlatL2 | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.top_k = top_k
        self.index: faiss.IndexFlatL2 = index or load_index()
        self.metadata: list[dict[str, Any]] = metadata or load_metadata()

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) does not match "
                f"FAISS index size ({self.index.ntotal}). "
                "Re-run `python src/embed.py --rebuild`."
            )

        logger.info(f"Building BM25 index over {len(self.metadata)} chunks …")
        tokenised_corpus = [_tokenise(c["text"]) for c in self.metadata]
        self.bm25 = BM25Okapi(tokenised_corpus)
        logger.success("HybridRetriever ready.")

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_retrieve(self, query: str, n: int) -> list[int]:
        """Return top-*n* chunk indices via FAISS L2 search."""
        model = get_model()
        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        distances, indices = self.index.search(query_emb, n)
        # Filter out -1 (FAISS returns -1 for empty slots)
        return [int(i) for i in indices[0] if i >= 0]

    # ── Sparse retrieval ──────────────────────────────────────────────────────

    def _sparse_retrieve(self, query: str, n: int) -> list[int]:
        """Return top-*n* chunk indices via BM25 scoring."""
        tokens = _tokenise(query)
        scores = self.bm25.get_scores(tokens)
        # argsort descending
        ranked = np.argsort(scores)[::-1][:n]
        return [int(i) for i in ranked]

    # ── Public retrieve ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: Natural-language question.
            top_k: Override default top_k.

        Returns:
            List of chunk dicts (metadata + text + retrieval_score).
        """
        k = top_k or self.top_k
        # Retrieve more candidates before fusion
        candidate_n = min(k * 4, len(self.metadata))

        dense_ranked = self._dense_retrieve(query, candidate_n)
        sparse_ranked = self._sparse_retrieve(query, candidate_n)

        fused = _reciprocal_rank_fusion([dense_ranked, sparse_ranked])

        results: list[dict[str, Any]] = []
        for idx, score in fused[:k]:
            chunk = dict(self.metadata[idx])
            chunk["retrieval_score"] = round(score, 6)
            results.append(chunk)

        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:60]!r}")
        return results

    # ── Reload after incremental indexing ────────────────────────────────────

    def reload(self) -> None:
        """Reload index and metadata from disk (after add_document())."""
        self.index = load_index()
        self.metadata = load_metadata()
        tokenised_corpus = [_tokenise(c["text"]) for c in self.metadata]
        self.bm25 = BM25Okapi(tokenised_corpus)
        logger.info(f"HybridRetriever reloaded: {len(self.metadata)} chunks")


# ── Convenience function ──────────────────────────────────────────────────────

_retriever_singleton: HybridRetriever | None = None


def get_retriever(top_k: int = DEFAULT_TOP_K) -> HybridRetriever:
    """Return a cached HybridRetriever instance."""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever(top_k=top_k)
    return _retriever_singleton


def reset_retriever() -> None:
    """Force re-initialisation of the singleton (e.g. after new docs added)."""
    global _retriever_singleton
    _retriever_singleton = None
