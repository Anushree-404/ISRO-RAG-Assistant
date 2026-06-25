"""
tests/test_ingest.py — Unit tests for the ingestion pipeline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from utils import clean_text, extract_section_title


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_excessive_newlines(self) -> None:
        raw = "Line one\n\n\n\n\nLine two"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_removes_lone_page_numbers(self) -> None:
        raw = "Some content\n   3   \nMore content"
        result = clean_text(raw)
        assert "   3   " not in result

    def test_removes_dash_dividers(self) -> None:
        raw = "Section A\n----------\nSection B"
        result = clean_text(raw)
        assert "----------" not in result

    def test_normalises_whitespace(self) -> None:
        raw = "Word1    Word2     Word3"
        result = clean_text(raw)
        assert "    " not in result

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_preserves_content(self) -> None:
        raw = "Chandrayaan-3 landed on the Moon."
        result = clean_text(raw)
        assert "Chandrayaan-3" in result
        assert "Moon" in result


# ── extract_section_title ─────────────────────────────────────────────────────

class TestExtractSectionTitle:
    def test_returns_first_short_line(self) -> None:
        text = "Introduction\nThis section covers the mission overview."
        title = extract_section_title(text)
        assert title == "Introduction"

    def test_skips_empty_lines(self) -> None:
        text = "\n\nMission Objectives\nThe primary objective is …"
        title = extract_section_title(text)
        assert title == "Mission Objectives"

    def test_empty_text(self) -> None:
        assert extract_section_title("") == ""

    def test_very_long_first_line_skipped(self) -> None:
        long_line = "A" * 200
        text = f"{long_line}\nShort Title\nBody text."
        title = extract_section_title(text)
        assert title == "Short Title"


# ── ChunkMetadata schema ──────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_valid_chunk(self) -> None:
        from ingest import ChunkMetadata

        chunk = ChunkMetadata(
            chunk_id="test_p0001_c000",
            source_file="test.pdf",
            page_number=1,
            section_title="Introduction",
            mission_name="Chandrayaan-3",
            text="Sample text content for testing.",
            char_count=32,
        )
        assert chunk.chunk_id == "test_p0001_c000"
        assert chunk.page_number == 1
        assert chunk.char_count == 32

    def test_serialisation(self) -> None:
        from ingest import ChunkMetadata

        chunk = ChunkMetadata(
            chunk_id="test_p0001_c000",
            source_file="test.pdf",
            page_number=1,
            text="Hello world",
            char_count=11,
        )
        d = chunk.model_dump()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "test_p0001_c000"
        assert d["mission_name"] == ""  # default


# ── Mission inference ─────────────────────────────────────────────────────────

class TestMissionInference:
    def test_chandrayaan3(self) -> None:
        from ingest import _infer_mission
        assert _infer_mission("chandrayaan3_report.pdf", "") == "Chandrayaan-3"

    def test_mangalyaan(self) -> None:
        from ingest import _infer_mission
        assert _infer_mission("mangalyaan_overview.pdf", "") == "Mangalyaan"

    def test_aditya(self) -> None:
        from ingest import _infer_mission
        assert _infer_mission("aditya_l1_doc.pdf", "") == "Aditya-L1"

    def test_unknown(self) -> None:
        from ingest import _infer_mission
        assert _infer_mission("random_document.pdf", "") == "Unknown"

    def test_inferred_from_text(self) -> None:
        from ingest import _infer_mission
        assert _infer_mission("doc.pdf", "This document covers Gaganyaan mission") == "Gaganyaan"


# ── save_chunks / load_json ───────────────────────────────────────────────────

class TestSaveChunks:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        from ingest import save_chunks
        from utils import load_json

        chunks = [
            {
                "chunk_id": "a_p0001_c000",
                "source_file": "a.pdf",
                "page_number": 1,
                "section_title": "",
                "mission_name": "Test",
                "text": "Hello",
                "char_count": 5,
            }
        ]
        out = tmp_path / "chunks.json"
        save_chunks(chunks, out)

        loaded = load_json(out)
        assert len(loaded) == 1
        assert loaded[0]["chunk_id"] == "a_p0001_c000"

    def test_deduplication(self, tmp_path: Path) -> None:
        from ingest import save_chunks
        from utils import load_json

        chunk = {
            "chunk_id": "dup_p0001_c000",
            "source_file": "dup.pdf",
            "page_number": 1,
            "section_title": "",
            "mission_name": "Test",
            "text": "Duplicate",
            "char_count": 9,
        }
        out = tmp_path / "chunks.json"
        save_chunks([chunk], out)
        save_chunks([chunk], out)  # second call should not duplicate

        loaded = load_json(out)
        assert len(loaded) == 1
