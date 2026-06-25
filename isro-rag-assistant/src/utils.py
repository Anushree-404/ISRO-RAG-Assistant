"""
utils.py — Shared utilities: logging, config loading, path helpers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")


# ── Directory helpers ─────────────────────────────────────────────────────────

def get_data_raw_dir() -> Path:
    """Return the raw data directory, creating it if needed."""
    path = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_processed_dir() -> Path:
    """Return the processed data directory, creating it if needed."""
    path = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_embeddings_dir() -> Path:
    """Return the embeddings directory, creating it if needed."""
    path = PROJECT_ROOT / os.getenv("EMBEDDINGS_DIR", "embeddings")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chunks_path() -> Path:
    """Return path to the chunks metadata JSON file."""
    return get_data_processed_dir() / "chunks.json"


def get_faiss_index_path() -> Path:
    """Return path to the FAISS index file."""
    return get_embeddings_dir() / "index.faiss"


def get_chunks_metadata_path() -> Path:
    """Return path to the embeddings-aligned metadata JSON."""
    return get_embeddings_dir() / "chunks_metadata.json"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    """Load JSON from *path*. Returns empty list if file does not exist."""
    if not path.exists():
        logger.warning(f"JSON file not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Path) -> None:
    """Serialise *data* to *path* with UTF-8 encoding and indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON → {path}")


# ── Text cleaning helpers ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove common PDF artefacts:
    - Excessive whitespace / newlines
    - Page numbers standing alone on a line
    - Repeated dashes / underscores used as dividers
    """
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone page-number lines like "  3  " or "Page 3 of 10"
    text = re.sub(r"(?m)^\s*(Page\s+\d+\s+of\s+\d+|\d+)\s*$", "", text)
    # Remove long dash/underscore dividers
    text = re.sub(r"[-_]{4,}", "", text)
    # Normalise whitespace within lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_section_title(text: str) -> str:
    """
    Heuristically extract a section title from the first non-empty line
    of a chunk.  Returns empty string if nothing useful is found.
    """
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 120:
            return line
    return ""


# ── Env helpers ───────────────────────────────────────────────────────────────

def get_env(key: str, default: str = "") -> str:
    """Return environment variable *key*, falling back to *default*."""
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """Return environment variable *key* or raise if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example → .env and fill in your values."
        )
    return value


# ── Logging setup ─────────────────────────────────────────────────────────────

def configure_logging(level: str | None = None) -> None:
    """Configure loguru with a sensible format."""
    import sys

    level = level or get_env("LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )
