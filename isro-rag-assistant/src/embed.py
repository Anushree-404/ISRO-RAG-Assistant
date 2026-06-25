"""
embed.py — Embedding pipeline.

Steps:
  1. Load chunks from data/processed/chunks.json.
  2. Embed all chunk texts using sentence-transformers (all-MiniLM-L6-v2).
  3. Build a FAISS IndexFlatL2 index.
  4. Save index → embeddings/index.faiss
     Save parallel metadata → embeddings/chunks_metadata.json
  5. Expose add_document() for incremental indexing.

Usage:
    python src/embed.py                    # embed all chunks
    python src/embed.py --rebuild          # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import (
    configure_logging,
    get_chunks_path,
    get_faiss_index_path,
    get_chunks_metadata_path,
    load_json,
    save_json,
)

configure_logging()

# Force HuggingFace to use local cache only (no network calls)
import os as _os
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2
BATCH_SIZE = 64


# ── Model singleton ───────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns:
        Float32 numpy array of shape (N, EMBEDDING_DIM).
    """
    model = get_model()
    embeddings: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + BATCH_SIZE]
        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via dot product
            show_progress_bar=False,
        )
        embeddings.append(batch_emb.astype(np.float32))

    return np.vstack(embeddings)


# ── FAISS index helpers ───────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a new FAISS IndexFlatL2 from *embeddings*."""
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatL2, path: Path | None = None) -> None:
    """Persist FAISS index to disk."""
    path = path or get_faiss_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info(f"FAISS index saved → {path}")


def load_index(path: Path | None = None) -> faiss.IndexFlatL2:
    """Load FAISS index from disk. Raises FileNotFoundError if missing."""
    path = path or get_faiss_index_path()
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Run `python src/embed.py` first."
        )
    index = faiss.read_index(str(path))
    logger.info(f"FAISS index loaded: {index.ntotal} vectors from {path}")
    return index  # type: ignore[return-value]


# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_metadata(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the embeddings-aligned metadata list."""
    path = path or get_chunks_metadata_path()
    return load_json(path)  # type: ignore[return-value]


def save_metadata(metadata: list[dict[str, Any]], path: Path | None = None) -> None:
    """Save the embeddings-aligned metadata list."""
    path = path or get_chunks_metadata_path()
    save_json(metadata, path)


# ── Full build ────────────────────────────────────────────────────────────────

def build_embeddings(rebuild: bool = False) -> tuple[faiss.IndexFlatL2, list[dict[str, Any]]]:
    """
    Load all chunks, embed them, build FAISS index, and persist everything.

    Args:
        rebuild: If True, ignore existing index and rebuild from scratch.

    Returns:
        (index, metadata_list)
    """
    chunks_path = get_chunks_path()
    chunks: list[dict[str, Any]] = load_json(chunks_path)  # type: ignore[assignment]

    if not chunks:
        raise ValueError(
            f"No chunks found at {chunks_path}. "
            "Run `python src/ingest.py --dir data/raw` first."
        )

    index_path = get_faiss_index_path()
    meta_path = get_chunks_metadata_path()

    if not rebuild and index_path.exists() and meta_path.exists():
        existing_meta = load_metadata(meta_path)
        if len(existing_meta) == len(chunks):
            logger.info("Index is up-to-date. Use --rebuild to force rebuild.")
            return load_index(index_path), existing_meta

    logger.info(f"Embedding {len(chunks)} chunks …")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    index = build_index(embeddings)
    save_index(index, index_path)
    save_metadata(chunks, meta_path)

    logger.success(f"Embedding complete: {index.ntotal} vectors, dim={EMBEDDING_DIM}")
    return index, chunks


# ── Incremental add_document() ────────────────────────────────────────────────

def add_document(
    new_chunks: list[dict[str, Any]],
    index: faiss.IndexFlatL2 | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> tuple[faiss.IndexFlatL2, list[dict[str, Any]]]:
    """
    Incrementally add *new_chunks* to an existing FAISS index.

    If *index* / *metadata* are None, they are loaded from disk.
    The updated index and metadata are persisted automatically.

    Args:
        new_chunks: List of chunk dicts (same schema as ingest.py output).
        index:      Existing FAISS index (loaded from disk if None).
        metadata:   Existing metadata list (loaded from disk if None).

    Returns:
        (updated_index, updated_metadata)
    """
    if index is None:
        index = load_index()
    if metadata is None:
        metadata = load_metadata()

    # Deduplicate by chunk_id
    existing_ids = {m["chunk_id"] for m in metadata}
    unique_new = [c for c in new_chunks if c["chunk_id"] not in existing_ids]

    if not unique_new:
        logger.info("No new chunks to add (all already indexed).")
        return index, metadata

    logger.info(f"Adding {len(unique_new)} new chunks to index …")
    texts = [c["text"] for c in unique_new]
    new_embeddings = embed_texts(texts)

    index.add(new_embeddings.astype(np.float32))
    metadata.extend(unique_new)

    save_index(index)
    save_metadata(metadata)

    logger.success(f"Index updated: {index.ntotal} total vectors")
    return index, metadata


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS embedding index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe existing index and rebuild from scratch",
    )
    args = parser.parse_args()
    build_embeddings(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
