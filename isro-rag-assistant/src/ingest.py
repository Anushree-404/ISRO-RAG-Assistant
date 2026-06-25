"""
ingest.py — PDF ingestion pipeline.

Steps:
  1. Extract text from PDFs using PyMuPDF.
  2. Fall back to Tesseract OCR for scanned / image-only pages.
  3. Clean headers, footers, and artefacts.
  4. Chunk with RecursiveCharacterTextSplitter (chunk_size=800, overlap=100).
  5. Attach metadata: source_file, page_number, section_title, mission_name.
  6. Save chunks as JSON to data/processed/chunks.json.

Usage:
    python src/ingest.py --pdf data/raw/chandrayaan3.pdf
    python src/ingest.py --dir data/raw          # batch-ingest all PDFs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, Field

from utils import (
    clean_text,
    configure_logging,
    extract_section_title,
    get_chunks_path,
    get_data_raw_dir,
    load_json,
    save_json,
)

configure_logging()

# ── Pydantic schema for a single chunk ───────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Metadata attached to every text chunk."""

    chunk_id: str = Field(..., description="Unique identifier: <stem>_p<page>_c<idx>")
    source_file: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="1-based page number")
    section_title: str = Field(default="", description="Heuristic section heading")
    mission_name: str = Field(default="", description="Inferred ISRO mission name")
    text: str = Field(..., description="Chunk text content")
    char_count: int = Field(..., description="Character length of chunk text")


# ── OCR fallback ──────────────────────────────────────────────────────────────


def _ocr_page(page: fitz.Page) -> str:
    """
    Render *page* to an image and run Tesseract OCR.
    Returns extracted text or empty string on failure.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        mat = fitz.Matrix(2.0, 2.0)  # 2× zoom for better OCR accuracy
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text: str = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception as exc:
        logger.warning(f"OCR failed on page {page.number + 1}: {exc}")
        return ""


# ── Mission name inference ────────────────────────────────────────────────────

_MISSION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"chandrayaan[-\s]?3", re.I), "Chandrayaan-3"),
    (re.compile(r"chandrayaan[-\s]?2", re.I), "Chandrayaan-2"),
    (re.compile(r"chandrayaan[-\s]?1", re.I), "Chandrayaan-1"),
    (re.compile(r"mangalyaan|mars\s+orbiter", re.I), "Mangalyaan"),
    (re.compile(r"aditya[-_\s]?l1", re.I), "Aditya-L1"),
    (re.compile(r"gaganyaan", re.I), "Gaganyaan"),
    (re.compile(r"pslv", re.I), "PSLV"),
    (re.compile(r"gslv", re.I), "GSLV"),
    (re.compile(r"cartosat", re.I), "Cartosat"),
    (re.compile(r"risat", re.I), "RISAT"),
]


def _infer_mission(filename: str, text_sample: str) -> str:
    """Infer ISRO mission name from filename and first-page text."""
    combined = filename + " " + text_sample[:500]
    for pattern, name in _MISSION_PATTERNS:
        if pattern.search(combined):
            return name
    return "Unknown"


# ── Per-page text extraction ──────────────────────────────────────────────────

_MIN_TEXT_CHARS = 50  # below this threshold → try OCR


def _extract_page_text(page: fitz.Page) -> str:
    """Extract text from *page*, falling back to OCR if needed."""
    text = page.get_text("text")  # type: ignore[arg-type]
    if len(text.strip()) < _MIN_TEXT_CHARS:
        logger.debug(f"Page {page.number + 1}: sparse text ({len(text.strip())} chars), trying OCR")
        ocr_text = _ocr_page(page)
        if len(ocr_text.strip()) > len(text.strip()):
            return ocr_text
    return text


# ── Header / footer removal ───────────────────────────────────────────────────

def _remove_headers_footers(pages_text: list[str]) -> list[str]:
    """
    Remove repeated lines that appear on ≥60 % of pages (likely headers/footers).
    """
    if len(pages_text) < 3:
        return pages_text

    # Collect first and last non-empty lines per page
    candidates: dict[str, int] = {}
    for text in pages_text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in (lines[:2] + lines[-2:]):
            candidates[line] = candidates.get(line, 0) + 1

    threshold = max(2, int(len(pages_text) * 0.6))
    repeated = {line for line, count in candidates.items() if count >= threshold}

    cleaned: list[str] = []
    for text in pages_text:
        lines = text.splitlines()
        filtered = [l for l in lines if l.strip() not in repeated]
        cleaned.append("\n".join(filtered))
    return cleaned


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Ingest a single PDF and return a list of chunk dicts.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of serialised ChunkMetadata dicts.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Ingesting: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    for page in doc:
        raw = _extract_page_text(page)
        pages_text.append(raw)

    doc.close()

    # Remove repeated headers/footers
    pages_text = _remove_headers_footers(pages_text)

    # Infer mission name from filename + first page
    mission_name = _infer_mission(pdf_path.name, pages_text[0] if pages_text else "")

    # Build splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict[str, Any]] = []

    for page_idx, raw_text in enumerate(pages_text):
        page_number = page_idx + 1
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        page_chunks = splitter.split_text(cleaned)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{pdf_path.stem}_p{page_number:04d}_c{chunk_idx:03d}"
            section_title = extract_section_title(chunk_text)

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=pdf_path.name,
                page_number=page_number,
                section_title=section_title,
                mission_name=mission_name,
                text=chunk_text,
                char_count=len(chunk_text),
            )
            chunks.append(meta.model_dump())

    logger.success(f"  → {len(chunks)} chunks from {len(pages_text)} pages")
    return chunks


def ingest_directory(directory: Path) -> list[dict[str, Any]]:
    """
    Ingest all PDFs in *directory* and return combined chunk list.
    """
    directory = Path(directory).resolve()
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return []

    all_chunks: list[dict[str, Any]] = []
    for pdf_file in pdf_files:
        try:
            chunks = ingest_pdf(pdf_file)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error(f"Failed to ingest {pdf_file.name}: {exc}")

    logger.info(f"Total chunks from directory: {len(all_chunks)}")
    return all_chunks


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_chunks(chunks: list[dict[str, Any]], output_path: Path | None = None) -> Path:
    """Save *chunks* to JSON. Merges with existing chunks (deduplicates by chunk_id)."""
    output_path = output_path or get_chunks_path()

    existing: list[dict[str, Any]] = load_json(output_path)
    existing_ids = {c["chunk_id"] for c in existing}

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    merged = existing + new_chunks

    save_json(merged, output_path)
    logger.info(f"Saved {len(new_chunks)} new chunks ({len(merged)} total) → {output_path}")
    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into chunk JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=Path, help="Path to a single PDF file")
    group.add_argument("--dir", type=Path, help="Directory containing PDF files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/processed/chunks.json)",
    )
    args = parser.parse_args()

    if args.pdf:
        chunks = ingest_pdf(args.pdf)
    else:
        chunks = ingest_directory(args.dir or get_data_raw_dir())

    if chunks:
        save_chunks(chunks, args.output)
    else:
        logger.warning("No chunks produced — check your PDF files.")


if __name__ == "__main__":
    main()
