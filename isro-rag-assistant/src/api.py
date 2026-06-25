"""
api.py — FastAPI REST layer for the ISRO RAG Assistant.

Endpoints:
  POST /query          — standard RAG query, returns RAGResponse JSON
  POST /query/stream   — streaming SSE endpoint, tokens arrive in real time
  GET  /health         — liveness check
  GET  /docs-list      — list indexed documents
  GET  /cache/stats    — cache statistics
  DELETE /cache        — clear the cache
  GET  /sessions       — list saved sessions
  GET  /sessions/{id}  — export a session as markdown
  DELETE /sessions/{id}— delete a session

Run:
    uvicorn src.api:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "What is Chandrayaan-3?"}'
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from utils import configure_logging, get_chunks_metadata_path, load_json

configure_logging()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ISRO RAG Assistant API",
    description=(
        "Retrieval-Augmented Generation over ISRO mission documents. "
        "Hybrid FAISS+BM25 retrieval, cross-encoder re-ranking, "
        "conversation memory, semantic caching."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str       = Field(..., min_length=3, description="User question")
    session_id: str     = Field(default="", description="Session ID for memory (optional)")
    use_reranker: bool  = Field(default=True,  description="Enable cross-encoder re-ranking")
    use_cache: bool     = Field(default=True,  description="Enable semantic cache")
    top_k: int          = Field(default=6,     ge=1, le=20, description="Chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]]
    confidence_score: float
    retrieval_time_ms: float
    generation_time_ms: float
    reranked: bool
    cache_hit: bool
    rewritten_query: str
    session_id: str
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str
    timestamp: float


# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_chain: Any = None
_retriever: Any = None


def _get_chain(use_reranker: bool = True, use_cache: bool = True) -> Any:
    global _chain
    if _chain is None:
        from generator import RAGChain
        _chain = RAGChain(use_reranker=use_reranker, use_cache=use_cache)
    return _chain


def _get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness check — also reports index size."""
    try:
        r = _get_retriever()
        index_size = r.index.ntotal
    except Exception:
        index_size = -1

    return HealthResponse(
        status="ok",
        index_size=index_size,
        model="claude-sonnet-4-20250514",
        timestamp=time.time(),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest) -> QueryResponse:
    """
    Run a RAG query and return a structured response with citations.
    """
    t_start = time.perf_counter()

    try:
        chain = _get_chain(
            use_reranker=req.use_reranker,
            use_cache=req.use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG chain unavailable: {exc}")

    # Load memory if session_id provided
    memory = None
    if req.session_id:
        try:
            from memory import ConversationMemory
            memory = ConversationMemory(session_id=req.session_id)
        except Exception as exc:
            logger.warning(f"Memory unavailable: {exc}")

    try:
        response, _ = chain.run_with_chunks(req.question, memory=memory)
    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    total_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        question=req.question,
        answer=response.answer,
        citations=[c.model_dump() for c in response.citations],
        confidence_score=response.confidence_score,
        retrieval_time_ms=response.retrieval_time_ms,
        generation_time_ms=response.generation_time_ms,
        reranked=response.reranked,
        cache_hit=response.cache_hit,
        rewritten_query=response.rewritten_query,
        session_id=req.session_id,
        total_time_ms=round(total_ms, 2),
    )


@app.post("/query/stream", tags=["RAG"])
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Stream the answer token-by-token using Server-Sent Events (SSE).

    Connect with:
        curl -N -X POST http://localhost:8000/query/stream \\
             -H "Content-Type: application/json" \\
             -d '{"question": "Explain Chandrayaan-3"}'
    """
    try:
        chain = _get_chain(
            use_reranker=req.use_reranker,
            use_cache=req.use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG chain unavailable: {exc}")

    memory = None
    if req.session_id:
        try:
            from memory import ConversationMemory
            memory = ConversationMemory(session_id=req.session_id)
        except Exception:
            pass

    def event_generator():
        try:
            for token in chain.stream_run(req.question, memory=memory):
                # SSE format: "data: <token>\n\n"
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/docs-list", tags=["Documents"])
async def list_documents() -> dict[str, Any]:
    """List all indexed documents with chunk counts and page ranges."""
    meta_path = get_chunks_metadata_path()
    if not meta_path.exists():
        return {"documents": [], "total_chunks": 0}

    chunks = load_json(meta_path)
    if not isinstance(chunks, list):
        return {"documents": [], "total_chunks": 0}

    doc_map: dict[str, dict[str, Any]] = {}
    for c in chunks:
        src = c.get("source_file", "unknown")
        if src not in doc_map:
            doc_map[src] = {
                "source_file": src,
                "mission_name": c.get("mission_name", "Unknown"),
                "chunk_count": 0,
                "max_page": 0,
            }
        doc_map[src]["chunk_count"] += 1
        doc_map[src]["max_page"] = max(
            doc_map[src]["max_page"], c.get("page_number", 0)
        )

    return {
        "documents": list(doc_map.values()),
        "total_chunks": len(chunks),
        "total_documents": len(doc_map),
    }


@app.get("/cache/stats", tags=["Cache"])
async def cache_stats() -> dict[str, Any]:
    """Return semantic cache statistics."""
    try:
        from cache import get_cache
        return get_cache().stats()
    except Exception as exc:
        return {"error": str(exc)}


@app.delete("/cache", tags=["Cache"])
async def clear_cache() -> dict[str, str]:
    """Clear the semantic query cache."""
    try:
        from cache import get_cache
        get_cache().clear()
        return {"status": "Cache cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions", tags=["Sessions"])
async def list_sessions() -> dict[str, Any]:
    """List all saved conversation sessions."""
    try:
        from memory import SessionStore
        return {"sessions": SessionStore.list_sessions()}
    except Exception as exc:
        return {"sessions": [], "error": str(exc)}


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def export_session(session_id: str) -> dict[str, str]:
    """Export a session as markdown."""
    try:
        from memory import SessionStore
        md = SessionStore.export_session(session_id)
        if not md:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"session_id": session_id, "markdown": md}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a conversation session."""
    try:
        from memory import SessionStore
        deleted = SessionStore.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"status": f"Session {session_id} deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
