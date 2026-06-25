"""
reranker.py — Cross-encoder re-ranking stage.

After hybrid retrieval returns top-K candidates, the cross-encoder
scores each (query, chunk) pair jointly and re-orders them.
This typically improves precision significantly over bi-encoder
retrieval alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (6-layer MiniLM)
  - Trained on MS MARCO passage ranking
  - Returns a relevance logit; higher = more relevant

Usage:
    from reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, chunks, top_n=4)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from utils import configure_logging

configure_logging()

# Force offline mode so the cross-encoder uses the local cache
import os as _os
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N   = 4


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for passage re-ranking.

    The model is loaded lazily on first use and cached as a class attribute
    so multiple instances share the same weights.
    """

    _model: Any = None  # class-level cache

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = DEFAULT_TOP_N) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self._ensure_model()

    @classmethod
    def _ensure_model(cls) -> None:
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
                cls._model = CrossEncoder(RERANKER_MODEL, max_length=512)
                logger.success("Cross-encoder loaded.")
            except Exception as exc:
                logger.warning(f"Cross-encoder unavailable ({exc}). Re-ranking disabled.")
                cls._model = None

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Re-rank *chunks* by relevance to *query*.

        Args:
            query:  User question.
            chunks: Candidate chunks from hybrid retrieval (already have
                    retrieval_score).
            top_n:  How many to return after re-ranking (default: self.top_n).

        Returns:
            Re-ranked list of chunk dicts, each with an added
            ``rerank_score`` field.  Falls back to original order if the
            cross-encoder is unavailable.
        """
        n = top_n or self.top_n

        if self._model is None or not chunks:
            # Graceful fallback — return original order
            for c in chunks:
                c.setdefault("rerank_score", c.get("retrieval_score", 0.0))
            return chunks[:n]

        pairs = [(query, c["text"]) for c in chunks]
        try:
            scores: np.ndarray = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning(f"Re-ranking failed: {exc}. Using original order.")
            for c in chunks:
                c.setdefault("rerank_score", c.get("retrieval_score", 0.0))
            return chunks[:n]

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        logger.debug(
            f"Re-ranked {len(chunks)} → top {n} | "
            f"best={reranked[0]['rerank_score']:.3f} "
            f"worst={reranked[min(n-1, len(reranked)-1)]['rerank_score']:.3f}"
        )
        return reranked[:n]

    def is_available(self) -> bool:
        """Return True if the cross-encoder model loaded successfully."""
        return self._model is not None


# ── Singleton ─────────────────────────────────────────────────────────────────

_reranker_singleton: CrossEncoderReranker | None = None


def get_reranker(top_n: int = DEFAULT_TOP_N) -> CrossEncoderReranker:
    """Return a cached CrossEncoderReranker instance."""
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = CrossEncoderReranker(top_n=top_n)
    return _reranker_singleton
