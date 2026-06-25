"""
memory.py — Conversation memory + query rewriting.

Features:
  1. ConversationMemory — stores (question, answer) turns with a sliding
     window so the LLM always sees the last N exchanges.
  2. QueryRewriter — rewrites a follow-up question into a self-contained
     query using the conversation history, so retrieval works correctly
     even for pronouns / references like "What about its fuel?" after
     asking about Chandrayaan-3.
  3. SessionStore — persists sessions to JSON so they survive restarts.

Usage:
    from memory import ConversationMemory, QueryRewriter

    mem = ConversationMemory(session_id="demo", window=4)
    mem.add_turn("What is Chandrayaan-3?", "Chandrayaan-3 is …")

    rewriter = QueryRewriter()
    standalone = rewriter.rewrite("What about its propulsion?", mem)
    # → "What is the propulsion system of Chandrayaan-3?"
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from utils import PROJECT_ROOT, configure_logging, save_json, load_json

configure_logging()

# ── Paths ─────────────────────────────────────────────────────────────────────
SESSIONS_DIR: Path = PROJECT_ROOT / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class Turn(BaseModel):
    """A single conversation turn."""
    turn_id: int
    question: str
    answer: str
    timestamp: float = Field(default_factory=time.time)
    confidence: float = 0.0
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ConversationMemory:
    """
    Sliding-window conversation memory.

    Args:
        session_id: Unique session identifier (auto-generated if None).
        window:     Number of recent turns to keep in the active context.
        persist:    If True, save/load turns from disk.
    """

    def __init__(
        self,
        session_id: str | None = None,
        window: int = 4,
        persist: bool = True,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.window = window
        self.persist = persist
        self._turns: list[Turn] = []

        if persist:
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(
        self,
        question: str,
        answer: str,
        confidence: float = 0.0,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a new turn to memory."""
        turn = Turn(
            turn_id=len(self._turns),
            question=question,
            answer=answer,
            confidence=confidence,
            citations=citations or [],
        )
        self._turns.append(turn)
        if self.persist:
            self._save()

    def get_window(self) -> list[Turn]:
        """Return the last *window* turns."""
        return self._turns[-self.window :]

    def format_history(self, max_chars: int = 2000) -> str:
        """
        Format recent turns as a compact dialogue string for prompt injection.
        Truncates oldest turns first if total length exceeds *max_chars*.
        """
        turns = self.get_window()
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.question}")
            # Truncate long answers
            ans = t.answer[:300] + " …" if len(t.answer) > 300 else t.answer
            lines.append(f"Assistant: {ans}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def clear(self) -> None:
        """Clear all turns from memory."""
        self._turns = []
        if self.persist:
            self._save()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "window": self.window,
            "turns": [t.model_dump() for t in self._turns],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _session_path(self) -> Path:
        return SESSIONS_DIR / f"session_{self.session_id}.json"

    def _save(self) -> None:
        try:
            save_json(self.to_dict(), self._session_path())
        except Exception as exc:
            logger.warning(f"Could not save session {self.session_id}: {exc}")

    def _load(self) -> None:
        path = self._session_path()
        if not path.exists():
            return
        try:
            data = load_json(path)
            if isinstance(data, dict):
                self._turns = [Turn(**t) for t in data.get("turns", [])]
                logger.debug(
                    f"Loaded session {self.session_id}: {len(self._turns)} turns"
                )
        except Exception as exc:
            logger.warning(f"Could not load session {self.session_id}: {exc}")


# ── Query Rewriter ────────────────────────────────────────────────────────────

# Pronouns and references that signal a follow-up question
_FOLLOWUP_SIGNALS = re.compile(
    r"\b(it|its|they|their|this|that|these|those|the mission|the spacecraft"
    r"|the rover|the lander|the satellite|the rocket|same|above|previous)\b",
    flags=re.I,
)

import re  # noqa: F811 — already imported at top


class QueryRewriter:
    """
    Rewrites follow-up questions into self-contained queries.

    Strategy:
      1. If the question contains no follow-up signals → return as-is.
      2. Otherwise, prepend the last assistant answer's key noun phrases
         as context, or use a simple heuristic substitution.
      3. If an LLM is available, use it for high-quality rewriting.
    """

    def __init__(self, use_llm: bool = False) -> None:
        """
        Args:
            use_llm: If True, use Claude for rewriting (requires API key).
                     If False, use fast heuristic rewriting.
        """
        self.use_llm = use_llm

    def needs_rewrite(self, question: str) -> bool:
        """Return True if the question likely refers to prior context."""
        return bool(_FOLLOWUP_SIGNALS.search(question))

    def rewrite(self, question: str, memory: ConversationMemory) -> str:
        """
        Rewrite *question* into a standalone query using *memory*.

        Returns the original question if no rewrite is needed or memory
        is empty.
        """
        if memory.is_empty or not self.needs_rewrite(question):
            return question

        if self.use_llm:
            return self._llm_rewrite(question, memory)
        return self._heuristic_rewrite(question, memory)

    def _heuristic_rewrite(self, question: str, memory: ConversationMemory) -> str:
        """
        Fast heuristic: prepend the last question's subject as context.
        Scans only the questions (not answers) to avoid picking up
        mission names mentioned in answers.
        """
        # Search only the user questions in recent history for mission names
        mission_pattern = re.compile(
            r"(Chandrayaan-3|Chandrayaan-2|Chandrayaan-1|Mangalyaan|"
            r"Aditya-L1|Gaganyaan|PSLV|GSLV|Mars Orbiter|Vikram|Pragyan)",
            re.I,
        )
        recent_turns = memory.get_window()
        subject = ""
        # Walk turns newest-first, look only at questions
        for turn in reversed(recent_turns):
            found = mission_pattern.findall(turn.question)
            if found:
                subject = found[-1]
                break

        if subject:
            rewritten = f"Regarding {subject}: {question}"
        else:
            last_q = recent_turns[-1].question if recent_turns else ""
            rewritten = f"Following up on '{last_q[:60]}': {question}"

        logger.debug(f"Query rewrite: {question!r} → {rewritten!r}")
        return rewritten

    def _llm_rewrite(self, question: str, memory: ConversationMemory) -> str:
        """Use Claude to rewrite the question (requires ANTHROPIC_API_KEY)."""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage
            from utils import require_env

            llm = ChatAnthropic(
                model="claude-haiku-4-20250514",
                anthropic_api_key=require_env("ANTHROPIC_API_KEY"),
                max_tokens=128,
                temperature=0.0,
            )
            history = memory.format_history(max_chars=800)
            prompt = (
                f"Given this conversation history:\n{history}\n\n"
                f"Rewrite this follow-up question as a fully self-contained "
                f"question that can be understood without the history. "
                f"Return ONLY the rewritten question, nothing else.\n\n"
                f"Follow-up: {question}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
            logger.debug(f"LLM query rewrite: {question!r} → {rewritten!r}")
            return rewritten
        except Exception as exc:
            logger.warning(f"LLM rewrite failed ({exc}), using heuristic.")
            return self._heuristic_rewrite(question, memory)


# ── Session store ─────────────────────────────────────────────────────────────

class SessionStore:
    """List and manage persisted sessions."""

    @staticmethod
    def list_sessions() -> list[dict[str, Any]]:
        """Return summary of all saved sessions."""
        sessions = []
        for path in sorted(SESSIONS_DIR.glob("session_*.json")):
            try:
                data = load_json(path)
                if isinstance(data, dict):
                    turns = data.get("turns", [])
                    sessions.append({
                        "session_id": data.get("session_id", path.stem),
                        "turn_count": len(turns),
                        "last_active": turns[-1]["timestamp"] if turns else 0,
                        "path": str(path),
                    })
            except Exception:
                pass
        return sorted(sessions, key=lambda s: s["last_active"], reverse=True)

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session file. Returns True if deleted."""
        path = SESSIONS_DIR / f"session_{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def export_session(session_id: str) -> str:
        """Export a session as a formatted markdown string."""
        path = SESSIONS_DIR / f"session_{session_id}.json"
        if not path.exists():
            return ""
        data = load_json(path)
        if not isinstance(data, dict):
            return ""
        lines = [f"# ISRO RAG Session — {session_id}\n"]
        for t in data.get("turns", []):
            lines.append(f"**Q{t['turn_id']+1}:** {t['question']}\n")
            lines.append(f"**A:** {t['answer']}\n")
            if t.get("citations"):
                for c in t["citations"]:
                    lines.append(
                        f"  - 📎 {c.get('doc','?')} p.{c.get('page','?')}\n"
                    )
            lines.append("---\n")
        return "\n".join(lines)
