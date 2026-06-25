"""
cache.py — Semantic query cache.

Avoids redundant LLM calls by caching (query → RAGResponse) pairs.
Uses two layers:
  1. Exact match  — hash of normalised query string.
  2. Semantic match — cosine similarity of query embeddings (threshold 0.92).

Cache is persisted to data/cache/query_cache.json and survives restarts.

Usage:
    from cache import SemanticCache

    cache = SemanticCache()
    hit = cache.get("What fuel does Chandrayaan-3 use?")
    if hit:
        return hit   # RAGResponse dict

    # ... run pipeline ...
    cache.put(query, response_dict)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from utils import PROJECT_ROOT, configure_logging, load_json, save_json

configure_logging()

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH: Path = CACHE_DIR / "query_cache.json"

SEMANTIC_THRESHOLD = 0.92   # cosine similarity above this → cache hit
MAX_CACHE_SIZE     = 500    # evict oldest entries beyond this
TTL_SECONDS        = 86400 * 7  # 7-day TTL


def _normalise(query: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", query.lower().strip())


def _hash(query: str) -> str:
    return hashlib.sha256(_normalise(query).encode()).hexdigest()[:16]


class SemanticCache:
    """
    Two-layer query cache: exact hash + semantic similarity.

    Attributes:
        threshold: Cosine similarity threshold for semantic hits.
        _store:    Dict mapping hash → {query, response, embedding, ts}.
    """

    def __init__(self, threshold: float = SEMANTIC_THRESHOLD) -> None:
        self.threshold = threshold
        self._store: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, np.ndarray] = {}  # hash → embedding vector
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, query: str) -> dict[str, Any] | None:
        """
        Look up *query* in the cache.

        Returns the cached response dict, or None on miss.
        """
        norm = _normalise(query)
        key  = _hash(norm)

        # 1. Exact match
        if key in self._store:
            entry = self._store[key]
            if not self._is_expired(entry):
                logger.debug(f"Cache HIT (exact): {query[:60]!r}")
                entry["hits"] = entry.get("hits", 0) + 1
                return entry["response"]
            else:
                del self._store[key]

        # 2. Semantic match
        query_emb = self._embed(norm)
        if query_emb is not None:
            best_key, best_sim = self._find_nearest(query_emb)
            if best_key and best_sim >= self.threshold:
                entry = self._store[best_key]
                if not self._is_expired(entry):
                    logger.debug(
                        f"Cache HIT (semantic, sim={best_sim:.3f}): {query[:60]!r}"
                    )
                    entry["hits"] = entry.get("hits", 0) + 1
                    return entry["response"]

        logger.debug(f"Cache MISS: {query[:60]!r}")
        return None

    def put(self, query: str, response: dict[str, Any]) -> None:
        """Store *response* for *query*."""
        norm = _normalise(query)
        key  = _hash(norm)
        emb  = self._embed(norm)

        self._store[key] = {
            "query":    norm,
            "response": response,
            "ts":       time.time(),
            "hits":     0,
        }
        if emb is not None:
            self._embeddings[key] = emb

        self._evict_if_needed()
        self._save()
        logger.debug(f"Cache PUT: {query[:60]!r}")

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache."""
        key = _hash(_normalise(query))
        if key in self._store:
            del self._store[key]
            self._embeddings.pop(key, None)
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Wipe the entire cache."""
        self._store.clear()
        self._embeddings.clear()
        self._save()
        logger.info("Cache cleared.")

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        total_hits = sum(e.get("hits", 0) for e in self._store.values())
        return {
            "entries": self.size,
            "total_hits": total_hits,
            "threshold": self.threshold,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_expired(entry: dict[str, Any]) -> bool:
        return (time.time() - entry.get("ts", 0)) > TTL_SECONDS

    def _embed(self, text: str) -> np.ndarray | None:
        """Embed *text* using the shared sentence-transformer model."""
        try:
            from embed import get_model
            model = get_model()
            vec = model.encode([text], convert_to_numpy=True,
                               normalize_embeddings=True)[0]
            return vec.astype(np.float32)
        except Exception as exc:
            logger.warning(f"Cache embedding failed: {exc}")
            return None

    def _find_nearest(
        self, query_emb: np.ndarray
    ) -> tuple[str | None, float]:
        """Return (key, cosine_similarity) of the nearest cached embedding."""
        if not self._embeddings:
            return None, 0.0

        keys = list(self._embeddings.keys())
        matrix = np.stack([self._embeddings[k] for k in keys])  # (N, D)
        # Both query_emb and stored embeddings are L2-normalised → dot = cosine
        sims = matrix @ query_emb
        best_idx = int(np.argmax(sims))
        return keys[best_idx], float(sims[best_idx])

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if cache exceeds MAX_CACHE_SIZE."""
        if len(self._store) <= MAX_CACHE_SIZE:
            return
        sorted_keys = sorted(self._store, key=lambda k: self._store[k]["ts"])
        to_remove = sorted_keys[: len(self._store) - MAX_CACHE_SIZE]
        for k in to_remove:
            del self._store[k]
            self._embeddings.pop(k, None)
        logger.debug(f"Cache evicted {len(to_remove)} old entries.")

    def _save(self) -> None:
        """Persist cache to disk (without numpy arrays)."""
        try:
            serialisable = {
                k: {kk: vv for kk, vv in v.items() if kk != "embedding"}
                for k, v in self._store.items()
            }
            save_json(serialisable, CACHE_PATH)
        except Exception as exc:
            logger.warning(f"Cache save failed: {exc}")

    def _load(self) -> None:
        """Load cache from disk."""
        data = load_json(CACHE_PATH)
        if isinstance(data, dict):
            self._store = data
            logger.debug(f"Cache loaded: {len(self._store)} entries")


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache_singleton: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = SemanticCache()
    return _cache_singleton
