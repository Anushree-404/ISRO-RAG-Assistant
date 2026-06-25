"""
generator.py — LangChain RAG chain backed by Claude.

Advanced features added:
  - Reads CLAUDE_MODEL from .env
  - Optional cross-encoder re-ranking (reranker.py)
  - Conversation memory injection (memory.py)
  - Semantic query cache (cache.py)
  - Streaming generation via stream_run()
  - Query rewriting for follow-up questions

The chain:
  1. Rewrite query if follow-up (memory-aware)
  2. Check semantic cache
  3. Retrieve top-K chunks via HybridRetriever
  4. Re-rank with CrossEncoder (optional)
  5. Send structured prompt to Claude
  6. Parse → RAGResponse Pydantic model
  7. Store in cache + memory

Usage:
    from generator import RAGChain
    chain = RAGChain()
    response = chain.run("What fuel does Chandrayaan-3 use?")

    # With memory
    from memory import ConversationMemory
    mem = ConversationMemory(session_id="user1")
    response = chain.run("What about its rover?", memory=mem)

    # Streaming
    for token in chain.stream_run("Explain Aditya-L1"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import re
import time
import random
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from retriever import HybridRetriever, get_retriever
from utils import configure_logging, get_env, require_env


def _build_llm(provider: str, model_name: str, max_tokens: int, temperature: float):
    """
    Build and return a LangChain-compatible LLM for the given provider.
    Supports 'gemini' (Google new SDK) and 'claude' (Anthropic).
    """
    if provider == "gemini":
        # Use the new google-genai SDK wrapped in a thin LangChain-compatible class
        api_key = require_env("GOOGLE_API_KEY")
        llm = _GeminiDirectLLM(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info(f"LLM provider: Google Gemini (google-genai SDK) | model: {model_name}")
        return llm
    else:
        from langchain_anthropic import ChatAnthropic
        api_key = require_env("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info(f"LLM provider: Anthropic Claude | model: {model_name}")
        return llm


class _GeminiDirectLLM:
    """
    Thin wrapper around google-genai SDK.
    Uses Gemini's native system_instruction so the model receives
    a clean system prompt separate from the user message.
    """

    def __init__(self, model_name: str, api_key: str,
                 max_tokens: int = 2048, temperature: float = 0.1) -> None:
        from google import genai as google_genai
        from google.genai import types as genai_types
        self._client      = google_genai.Client(api_key=api_key)
        self._model       = model_name
        self._max_tokens  = max_tokens
        self._temperature = temperature
        self._types       = genai_types

    def _make_config(self, system_instruction: str):
        return self._types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
            response_mime_type="application/json",   # ← forces JSON output
        )

    def _split_messages(self, messages: list) -> tuple[str, str]:
        """Split LangChain messages into (system_text, user_text)."""
        system_parts: list[str] = []
        user_parts:   list[str] = []
        for msg in messages:
            role    = type(msg).__name__  # SystemMessage / HumanMessage
            content = msg.content if hasattr(msg, "content") else str(msg)
            if "System" in role:
                system_parts.append(content)
            else:
                user_parts.append(content)
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    def invoke(self, messages: list) -> Any:
        system_text, user_text = self._split_messages(messages)
        config = self._make_config(system_text)
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_text,
            config=config,
        )
        raw = response.text or ""
        class _Resp:
            def __init__(self, text: str):
                self.content = text
        return _Resp(raw)

    def stream(self, messages: list):
        """Yield string tokens one by one (streaming mode — plain text)."""
        system_text, user_text = self._split_messages(messages)
        # Streaming uses plain text, not JSON
        config = self._types.GenerateContentConfig(
            system_instruction=system_text,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=user_text,
            config=config,
        ):
            if chunk.text:
                yield type("_Tok", (), {"content": chunk.text})()

configure_logging()

# ── Pydantic response schema ──────────────────────────────────────────────────


class Citation(BaseModel):
    """A single citation linking a claim to a source chunk."""
    doc: str        = Field(..., description="Source document filename")
    page: int       = Field(..., description="1-based page number")
    chunk_text: str = Field(..., description="Verbatim excerpt from the source chunk")


class RAGResponse(BaseModel):
    """Structured output from the RAG chain."""
    answer: str = Field(..., description="Full answer to the user question")
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_time_ms: float  = Field(default=0.0)
    generation_time_ms: float = Field(default=0.0)
    reranked: bool            = Field(default=False, description="Whether re-ranking was applied")
    cache_hit: bool           = Field(default=False, description="Whether answer came from cache")
    rewritten_query: str      = Field(default="", description="Rewritten query (if different)")

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for ISRO (Indian Space Research Organisation) mission documents.

RULES — follow them strictly:
1. Answer ONLY from the provided context chunks. Do NOT use outside knowledge.
2. For EVERY factual claim in your answer, cite the exact source document and page number.
3. If the context does not contain enough information to answer, say so clearly — do NOT hallucinate.
4. Be precise, technical, and concise.

OUTPUT FORMAT — respond with valid JSON only, no markdown fences, no extra text:
{
  "answer": "<your detailed answer here>",
  "citations": [
    {
      "doc": "<source_file>",
      "page": <page_number>,
      "chunk_text": "<verbatim excerpt from the chunk that supports this claim>"
    }
  ],
  "confidence_score": <float between 0.0 and 1.0>
}

If you cannot answer from the context:
{
  "answer": "The provided documents do not contain sufficient information to answer this question.",
  "citations": [],
  "confidence_score": 0.0
}"""

SYSTEM_PROMPT_WITH_HISTORY = SYSTEM_PROMPT + """

CONVERSATION HISTORY (for context only — do NOT cite from it):
{history}
"""

STREAMING_SYSTEM_PROMPT = """You are an expert ISRO mission document assistant.
Answer the question using ONLY the provided context. Be precise and cite sources.
Format: plain prose (no JSON needed for streaming). Mention document name and page for each fact."""


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        score_info = ""
        if "rerank_score" in chunk:
            score_info = f" | Rerank: {chunk['rerank_score']:.3f}"
        parts.append(
            f"[CHUNK {i}]\n"
            f"Source: {chunk.get('source_file', 'unknown')} | "
            f"Page: {chunk.get('page_number', '?')} | "
            f"Mission: {chunk.get('mission_name', 'unknown')}{score_info}\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(parts)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from the model response.
    Handles: clean JSON, markdown fences, JSON embedded in text,
    and Gemini's occasional double-encoding.
    """
    if not raw or not raw.strip():
        return {"answer": "No response received.", "citations": [], "confidence_score": 0.0}

    text = raw.strip()

    # 1. Strip markdown code fences  ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 2. Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return data
    except json.JSONDecodeError:
        pass

    # 3. Find the outermost { ... } block
    start = text.find("{")
    if start != -1:
        # Walk forward counting braces to find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "answer" in data:
                            return data
                    except json.JSONDecodeError:
                        break

    # 4. Gemini sometimes returns the JSON as a Python-repr string — try ast.literal_eval
    try:
        import ast
        data = ast.literal_eval(text)
        if isinstance(data, dict) and "answer" in data:
            return data
    except Exception:
        pass

    # 5. Last resort — treat the whole response as the answer
    logger.warning(f"Could not parse JSON, using raw text as answer. First 200 chars: {text[:200]}")
    return {
        "answer": text,
        "citations": [],
        "confidence_score": 0.3,
    }


# ── RAGChain ──────────────────────────────────────────────────────────────────

class RAGChain:
    """
    End-to-end RAG chain with reranking, memory, caching, and streaming.

    Args:
        retriever:    HybridRetriever (loaded from disk if None).
        model_name:   Claude model (reads CLAUDE_MODEL env var).
        max_tokens:   Max generation tokens.
        temperature:  Sampling temperature.
        use_reranker: Enable cross-encoder re-ranking.
        use_cache:    Enable semantic query cache.
        rerank_top_n: Number of chunks to keep after re-ranking.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        model_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        use_reranker: bool = True,
        use_cache: bool = True,
        rerank_top_n: int = 4,
    ) -> None:
        self.retriever    = retriever or get_retriever()

        # Determine provider and model from env
        self.provider   = get_env("LLM_PROVIDER", "gemini").lower().strip()
        if self.provider == "gemini":
            default_model = get_env("GEMINI_MODEL", "gemini-2.0-flash")
        else:
            default_model = get_env("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.model_name   = model_name or default_model

        self.max_tokens   = max_tokens
        self.temperature  = temperature
        self.use_reranker = use_reranker
        self.use_cache    = use_cache
        self.rerank_top_n = rerank_top_n

        self.llm = _build_llm(self.provider, self.model_name, max_tokens, temperature)

        # Lazy-load optional components
        self._reranker: Any = None
        self._cache: Any    = None
        self._rewriter: Any = None

        if use_reranker:
            try:
                from reranker import get_reranker
                self._reranker = get_reranker(top_n=rerank_top_n)
            except Exception as exc:
                logger.warning(f"Reranker unavailable: {exc}")

        if use_cache:
            try:
                from cache import get_cache
                self._cache = get_cache()
            except Exception as exc:
                logger.warning(f"Cache unavailable: {exc}")

        try:
            from memory import QueryRewriter
            self._rewriter = QueryRewriter(use_llm=False)
        except Exception as exc:
            logger.warning(f"QueryRewriter unavailable: {exc}")

        logger.info(
            f"RAGChain ready | provider={self.provider} model={self.model_name} "
            f"reranker={'✓' if self._reranker else '✗'} "
            f"cache={'✓' if self._cache else '✗'}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _retrieve_and_rerank(
        self, query: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Retrieve chunks and optionally re-rank. Returns (chunks, reranked)."""
        chunks = self.retriever.retrieve(query)
        if self._reranker and self._reranker.is_available() and chunks:
            chunks = self._reranker.rerank(query, chunks, top_n=self.rerank_top_n)
            return chunks, True
        return chunks, False

    def _build_messages(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        memory: Any | None = None,
    ) -> list:
        context_block = _format_context(chunks)
        user_content  = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "Respond with JSON only."
        )
        if memory and not memory.is_empty:
            history = memory.format_history()
            # Use % substitution to avoid KeyError on JSON braces in the prompt
            system = SYSTEM_PROMPT_WITH_HISTORY.replace("{history}", history)
        else:
            system = SYSTEM_PROMPT

        return [SystemMessage(content=system), HumanMessage(content=user_content)]

    def _parse_response(
        self,
        raw: str,
        retrieval_ms: float,
        generation_ms: float,
        reranked: bool,
        cache_hit: bool = False,
        rewritten_query: str = "",
    ) -> RAGResponse:
        parsed = _extract_json(raw)
        try:
            return RAGResponse(
                **parsed,
                retrieval_time_ms=round(retrieval_ms, 2),
                generation_time_ms=round(generation_ms, 2),
                reranked=reranked,
                cache_hit=cache_hit,
                rewritten_query=rewritten_query,
            )
        except Exception as exc:
            logger.error(f"Response parsing failed: {exc}")
            return RAGResponse(
                answer=parsed.get("answer", raw),
                citations=[],
                confidence_score=0.3,
                retrieval_time_ms=round(retrieval_ms, 2),
                generation_time_ms=round(generation_ms, 2),
                reranked=reranked,
                cache_hit=cache_hit,
                rewritten_query=rewritten_query,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        memory: Any | None = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline.

        Args:
            question: User question.
            memory:   ConversationMemory instance for multi-turn context.

        Returns:
            RAGResponse with answer, citations, confidence, timing, flags.
        """
        # 1. Query rewriting
        effective_query = question
        rewritten = ""
        if self._rewriter and memory:
            effective_query = self._rewriter.rewrite(question, memory)
            if effective_query != question:
                rewritten = effective_query
                logger.info(f"Query rewritten: {question!r} → {effective_query!r}")

        # 2. Cache lookup
        if self._cache:
            cached = self._cache.get(effective_query)
            if cached:
                try:
                    resp = RAGResponse(**cached)
                    resp.cache_hit = True
                    resp.rewritten_query = rewritten
                    if memory:
                        memory.add_turn(
                            question, resp.answer, resp.confidence_score,
                            [c.model_dump() for c in resp.citations],
                        )
                    return resp
                except Exception:
                    pass

        # 3. Retrieve + rerank
        t0 = time.perf_counter()
        chunks, reranked = self._retrieve_and_rerank(effective_query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                citations=[], confidence_score=0.0,
                retrieval_time_ms=retrieval_ms,
            )

        # 4. Build messages and call LLM (with retry on rate limit)
        messages = self._build_messages(effective_query, chunks, memory)
        t1 = time.perf_counter()
        raw_content = ""
        last_exc = None
        for attempt in range(4):
            try:
                llm_response = self.llm.invoke(messages)
                raw_content = llm_response.content  # type: ignore[assignment]
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                # Extract retry delay from error message if present
                delay_match = re.search(r"retry in (\d+)", err_str, re.I)
                wait = int(delay_match.group(1)) if delay_match else (2 ** attempt + random.uniform(0, 1))
                wait = min(wait, 65)  # cap at 65s
                logger.warning(f"LLM call attempt {attempt+1} failed (rate limit). Retrying in {wait:.0f}s …")
                time.sleep(wait)

        if last_exc is not None:
            logger.error(f"LLM call failed after retries: {last_exc}")
            return RAGResponse(
                answer=f"Generation error: {last_exc}",
                citations=[], confidence_score=0.0,
                retrieval_time_ms=retrieval_ms,
            )
        generation_ms = (time.perf_counter() - t1) * 1000

        # 5. Parse
        resp = self._parse_response(
            raw_content, retrieval_ms, generation_ms,
            reranked, cache_hit=False, rewritten_query=rewritten,
        )

        # 6. Cache + memory
        if self._cache:
            self._cache.put(effective_query, resp.model_dump())
        if memory:
            memory.add_turn(
                question, resp.answer, resp.confidence_score,
                [c.model_dump() for c in resp.citations],
            )

        logger.info(
            f"RAG | retrieval={retrieval_ms:.0f}ms gen={generation_ms:.0f}ms "
            f"conf={resp.confidence_score:.2f} reranked={reranked} "
            f"cache_hit=False"
        )
        return resp

    def run_with_chunks(
        self,
        question: str,
        memory: Any | None = None,
    ) -> tuple[RAGResponse, list[dict[str, Any]]]:
        """Same as run() but also returns the retrieved (and re-ranked) chunks."""
        effective_query = question
        rewritten = ""
        if self._rewriter and memory:
            effective_query = self._rewriter.rewrite(question, memory)
            if effective_query != question:
                rewritten = effective_query

        if self._cache:
            cached = self._cache.get(effective_query)
            if cached:
                try:
                    resp = RAGResponse(**cached)
                    resp.cache_hit = True
                    resp.rewritten_query = rewritten
                    if memory:
                        memory.add_turn(
                            question, resp.answer, resp.confidence_score,
                            [c.model_dump() for c in resp.citations],
                        )
                    return resp, []
                except Exception:
                    pass

        t0 = time.perf_counter()
        chunks, reranked = self._retrieve_and_rerank(effective_query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not chunks:
            return (
                RAGResponse(
                    answer="No relevant documents found.",
                    citations=[], confidence_score=0.0,
                    retrieval_time_ms=retrieval_ms,
                ),
                [],
            )

        messages = self._build_messages(effective_query, chunks, memory)
        t1 = time.perf_counter()
        raw_content = ""
        last_exc = None
        for attempt in range(4):
            try:
                llm_response = self.llm.invoke(messages)
                raw_content = llm_response.content  # type: ignore[assignment]
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                delay_match = re.search(r"retry in (\d+)", err_str, re.I)
                wait = int(delay_match.group(1)) if delay_match else (2 ** attempt + random.uniform(0, 1))
                wait = min(wait, 65)
                logger.warning(f"LLM call attempt {attempt+1} failed. Retrying in {wait:.0f}s …")
                time.sleep(wait)

        if last_exc is not None:
            logger.error(f"LLM call failed after retries: {last_exc}")
            return (
                RAGResponse(
                    answer=f"Generation error: {last_exc}",
                    citations=[], confidence_score=0.0,
                    retrieval_time_ms=retrieval_ms,
                ),
                chunks,
            )
        generation_ms = (time.perf_counter() - t1) * 1000

        resp = self._parse_response(
            raw_content, retrieval_ms, generation_ms,
            reranked, cache_hit=False, rewritten_query=rewritten,
        )

        if self._cache:
            self._cache.put(effective_query, resp.model_dump())
        if memory:
            memory.add_turn(
                question, resp.answer, resp.confidence_score,
                [c.model_dump() for c in resp.citations],
            )

        return resp, chunks

    def stream_run(
        self,
        question: str,
        memory: Any | None = None,
    ) -> Iterator[str]:
        """
        Stream the answer token-by-token.

        Yields string tokens as they arrive from Claude.
        Does NOT return a RAGResponse — use run() for structured output.
        """
        effective_query = question
        if self._rewriter and memory:
            effective_query = self._rewriter.rewrite(question, memory)

        chunks, _ = self._retrieve_and_rerank(effective_query)
        if not chunks:
            yield "No relevant documents found in the knowledge base."
            return

        context_block = _format_context(chunks)
        user_content = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer in clear prose, citing document name and page for each fact."
        )

        history_block = ""
        if memory and not memory.is_empty:
            history_block = f"\nCONVERSATION HISTORY:\n{memory.format_history()}\n"

        messages = [
            SystemMessage(content=STREAMING_SYSTEM_PROMPT + history_block),
            HumanMessage(content=user_content),
        ]

        try:
            full_answer = ""
            for chunk in self.llm.stream(messages):
                token: str = chunk.content  # type: ignore[assignment]
                full_answer += token
                yield token
            if memory:
                memory.add_turn(question, full_answer)

        except Exception as exc:
            logger.error(f"Streaming failed: {exc}")
            yield f"\n[Streaming error: {exc}]"


# ── Singleton ─────────────────────────────────────────────────────────────────

_chain_singleton: RAGChain | None = None


def get_chain(
    use_reranker: bool = True,
    use_cache: bool = True,
) -> RAGChain:
    global _chain_singleton
    if _chain_singleton is None:
        _chain_singleton = RAGChain(
            use_reranker=use_reranker,
            use_cache=use_cache,
        )
    return _chain_singleton
