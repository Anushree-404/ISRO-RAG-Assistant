"""
dashboard/app.py — Advanced Streamlit UI for ISRO RAG Assistant v2.

New features vs v1:
  ✦ Multi-turn conversation memory (per-session, persisted)
  ✦ Streaming answer display (token-by-token)
  ✦ Cross-encoder re-ranking toggle
  ✦ Semantic cache indicator (⚡ Cache Hit badge)
  ✦ Query rewrite indicator
  ✦ Session management (new / switch / export / delete)
  ✦ Advanced analytics: confidence histogram, retrieval heatmap,
    cache hit rate, rerank delta chart
  ✦ Dark-mode analytics panel with Plotly
  ✦ Export chat as Markdown

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ── Force offline mode immediately — prevents slow HuggingFace network checks ─
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import configure_logging, get_chunks_metadata_path, get_data_raw_dir, load_json

configure_logging()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRO RAG Assistant v2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chat bubbles */
.chat-user { background:#1e3a5f; border-radius:12px; padding:12px 16px; margin:6px 0; }
.chat-bot  { background:#0d2137; border-radius:12px; padding:12px 16px; margin:6px 0; }

/* Citation card */
.citation-card {
    background:#112240; border-left:3px solid #4fc3f7;
    border-radius:6px; padding:10px 14px; margin:4px 0; font-size:0.85rem;
}

/* Badges */
.badge-cache    { background:#1b5e20; color:#a5d6a7; border-radius:4px;
                  padding:2px 8px; font-size:0.75rem; font-weight:bold; }
.badge-reranked { background:#1a237e; color:#90caf9; border-radius:4px;
                  padding:2px 8px; font-size:0.75rem; font-weight:bold; }
.badge-rewrite  { background:#4a148c; color:#ce93d8; border-radius:4px;
                  padding:2px 8px; font-size:0.75rem; font-weight:bold; }

/* Confidence colours */
.conf-high { color:#66bb6a; font-weight:bold; }
.conf-mid  { color:#ffa726; font-weight:bold; }
.conf-low  { color:#ef5350; font-weight:bold; }

/* Metric cards */
.metric-row { display:flex; gap:12px; margin:8px 0; }
.metric-card {
    background:#0a1929; border-radius:8px; padding:10px 14px;
    flex:1; text-align:center; border:1px solid #1e3a5f;
}
.metric-val { font-size:1.4rem; font-weight:bold; color:#4fc3f7; }
.metric-lbl { font-size:0.75rem; color:#90a4ae; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "messages":         [],
        "session_id":       str(uuid.uuid4())[:8],
        "query_count":      0,
        "retrieval_times":  [],
        "generation_times": [],
        "confidence_scores":[],
        "cache_hits":       0,
        "reranked_count":   0,
        "use_reranker":     False,
        "use_cache":        True,
        "use_streaming":    False,
        "memory":           None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Resource loaders ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG pipeline … (first load takes ~30 sec)")
def _load_chain(use_reranker: bool, use_cache: bool):
    try:
        from generator import RAGChain
        return RAGChain(use_reranker=use_reranker, use_cache=use_cache)
    except Exception as exc:
        st.error(f"Failed to load RAG chain: {exc}")
        return None

@st.cache_resource(show_spinner="Loading retriever …")
def _load_retriever():
    try:
        from retriever import HybridRetriever
        return HybridRetriever()
    except Exception as exc:
        st.warning(f"Retriever unavailable: {exc}")
        return None

def _get_memory():
    """Return or create a ConversationMemory for the current session."""
    if st.session_state.memory is None:
        try:
            from memory import ConversationMemory
            st.session_state.memory = ConversationMemory(
                session_id=st.session_state.session_id,
                window=6,
            )
        except Exception:
            pass
    return st.session_state.memory

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_indexed_docs() -> list[dict[str, Any]]:
    meta_path = get_chunks_metadata_path()
    if not meta_path.exists():
        return []
    chunks = load_json(meta_path)
    if not isinstance(chunks, list):
        return []
    doc_map: dict[str, dict[str, Any]] = {}
    for c in chunks:
        src = c.get("source_file", "unknown")
        if src not in doc_map:
            doc_map[src] = {
                "source_file": src,
                "mission_name": c.get("mission_name", "Unknown"),
                "chunk_count": 0, "max_page": 0,
            }
        doc_map[src]["chunk_count"] += 1
        doc_map[src]["max_page"] = max(doc_map[src]["max_page"], c.get("page_number", 0))
    return list(doc_map.values())

def _conf_class(score: float) -> str:
    return "conf-high" if score >= 0.7 else ("conf-mid" if score >= 0.4 else "conf-low")

def _handle_pdf_upload(uploaded_file: Any) -> bool:
    raw_dir = get_data_raw_dir()
    dest = raw_dir / uploaded_file.name
    with dest.open("wb") as fh:
        fh.write(uploaded_file.getbuffer())
    prog = st.sidebar.progress(0, text="Ingesting …")
    try:
        from ingest import ingest_pdf, save_chunks
        chunks = ingest_pdf(dest)
        save_chunks(chunks)
        prog.progress(50, text="Embedding …")
        from embed import add_document, load_index, load_metadata
        try:
            add_document(chunks, load_index(), load_metadata())
        except FileNotFoundError:
            from embed import build_embeddings
            build_embeddings(rebuild=True)
        prog.progress(90, text="Reloading …")
        st.cache_resource.clear()
        prog.progress(100, text="Done!")
        time.sleep(0.4)
        prog.empty()
        return True
    except Exception as exc:
        prog.empty()
        st.sidebar.error(f"Upload failed: {exc}")
        return False

def _export_chat_markdown() -> str:
    lines = [f"# ISRO RAG Chat — Session {st.session_state.session_id}\n"]
    for msg in st.session_state.messages:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {msg['content']}\n")
        resp = msg.get("response")
        if resp and resp.citations:
            for c in resp.citations:
                lines.append(f"  - 📎 {c.doc} p.{c.page}\n")
        lines.append("---\n")
    return "\n".join(lines)

# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/"
        "Indian_Space_Research_Organisation_Logo.svg/200px-Indian_Space_Research_Organisation_Logo.svg.png",
        width=72,
    )
    st.sidebar.title("🚀 ISRO RAG v2")
    st.sidebar.caption(f"Session: `{st.session_state.session_id}`")
    st.sidebar.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Settings")
    st.session_state.use_reranker = st.sidebar.toggle(
        "Cross-encoder re-ranking", value=st.session_state.use_reranker,
        help="Re-rank retrieved chunks with ms-marco-MiniLM cross-encoder"
    )
    st.session_state.use_cache = st.sidebar.toggle(
        "Semantic cache", value=st.session_state.use_cache,
        help="Cache answers for semantically similar queries"
    )
    st.session_state.use_streaming = st.sidebar.toggle(
        "Streaming mode", value=st.session_state.use_streaming,
        help="Stream answer tokens in real time (no structured JSON)"
    )
    st.sidebar.markdown("---")

    # ── Session management ────────────────────────────────────────────────────
    st.sidebar.subheader("💬 Session")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("🆕 New", use_container_width=True):
        st.session_state.messages         = []
        st.session_state.session_id       = str(uuid.uuid4())[:8]
        st.session_state.memory           = None
        st.session_state.query_count      = 0
        st.session_state.retrieval_times  = []
        st.session_state.generation_times = []
        st.session_state.confidence_scores= []
        st.session_state.cache_hits       = 0
        st.session_state.reranked_count   = 0
        st.rerun()

    if col2.button("📥 Export", use_container_width=True):
        md = _export_chat_markdown()
        st.sidebar.download_button(
            "⬇️ Download .md", md,
            file_name=f"isro_chat_{st.session_state.session_id}.md",
            mime="text/markdown",
        )

    # List saved sessions
    try:
        from memory import SessionStore
        saved = SessionStore.list_sessions()
        if saved:
            with st.sidebar.expander(f"📂 Saved sessions ({len(saved)})", expanded=False):
                for s in saved[:5]:
                    sid = s["session_id"]
                    sc1, sc2 = st.columns([3, 1])
                    sc1.caption(f"`{sid}` — {s['turn_count']} turns")
                    if sc2.button("🗑", key=f"del_{sid}"):
                        SessionStore.delete_session(sid)
                        st.rerun()
    except Exception:
        pass

    st.sidebar.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.sidebar.subheader("📄 Upload PDF")
    uploaded = st.sidebar.file_uploader("Drop PDF", type=["pdf"],
                                         label_visibility="collapsed")
    if uploaded and st.sidebar.button("Index Document", use_container_width=True):
        with st.spinner(f"Indexing {uploaded.name} …"):
            if _handle_pdf_upload(uploaded):
                st.sidebar.success(f"✅ {uploaded.name} indexed!")
                st.rerun()

    st.sidebar.markdown("---")

    # ── Indexed docs ──────────────────────────────────────────────────────────
    st.sidebar.subheader("📚 Indexed Documents")
    docs = _get_indexed_docs()
    if docs:
        for doc in docs:
            with st.sidebar.expander(f"📄 {doc['source_file']}", expanded=False):
                st.write(f"**Mission:** {doc['mission_name']}")
                st.write(f"**Chunks:** {doc['chunk_count']}  |  **Pages:** {doc['max_page']}")
    else:
        st.sidebar.info("No documents indexed yet.")

    st.sidebar.markdown("---")

    # ── Analytics ─────────────────────────────────────────────────────────────
    st.sidebar.subheader("📊 Analytics")
    qc = st.session_state.query_count
    rt = st.session_state.retrieval_times
    gt = st.session_state.generation_times
    cs = st.session_state.confidence_scores

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Queries", qc)
    c2.metric("Cache Hits", st.session_state.cache_hits)

    c3, c4 = st.sidebar.columns(2)
    c3.metric("Avg Retrieval", f"{sum(rt)/len(rt):.0f}ms" if rt else "—")
    c4.metric("Re-ranked", st.session_state.reranked_count)

    # Mission pie
    if docs:
        mc: dict[str, int] = {}
        for d in docs:
            mc[d["mission_name"]] = mc.get(d["mission_name"], 0) + d["chunk_count"]
        fig = px.pie(names=list(mc.keys()), values=list(mc.values()),
                     title="Chunks by Mission", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=200, margin=dict(t=28,b=0,l=0,r=0),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          showlegend=True, legend=dict(font=dict(size=9)))
        st.sidebar.plotly_chart(fig, use_container_width=True)

    # Confidence histogram
    if len(cs) >= 2:
        fig2 = go.Figure(go.Histogram(
            x=cs, nbinsx=10, marker_color="#4fc3f7",
            xbins=dict(start=0, end=1, size=0.1),
        ))
        fig2.update_layout(
            title="Confidence Distribution", height=150,
            margin=dict(t=28,b=20,l=30,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis=dict(range=[0,1]),
        )
        st.sidebar.plotly_chart(fig2, use_container_width=True)

    # Retrieval time line
    if len(rt) > 1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=rt, mode="lines+markers",
                                   name="Retrieval ms", line=dict(color="#4fc3f7")))
        if gt:
            fig3.add_trace(go.Scatter(y=gt, mode="lines+markers",
                                       name="Generation ms", line=dict(color="#ff8a65")))
        fig3.update_layout(
            title="Latency (ms)", height=160,
            margin=dict(t=28,b=20,l=30,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", legend=dict(font=dict(size=9)),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.sidebar.plotly_chart(fig3, use_container_width=True)

# ── Citation card ─────────────────────────────────────────────────────────────

def _render_citation(cit: Any, idx: int) -> None:
    label = f"📎 [{idx+1}] {cit.doc} — page {cit.page}"
    with st.expander(label, expanded=False):
        st.markdown(
            f'<div class="citation-card">{cit.chunk_text}</div>',
            unsafe_allow_html=True,
        )

# ── Message renderer ──────────────────────────────────────────────────────────

def _render_message(msg: dict[str, Any]) -> None:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🛰️"):
            st.markdown(msg["content"])
            resp = msg.get("response")
            if resp:
                # Badges row
                badges = []
                if resp.cache_hit:
                    badges.append('<span class="badge-cache">⚡ Cache Hit</span>')
                if resp.reranked:
                    badges.append('<span class="badge-reranked">🔀 Re-ranked</span>')
                if resp.rewritten_query:
                    badges.append('<span class="badge-rewrite">✏️ Query Rewritten</span>')
                if badges:
                    st.markdown(" ".join(badges), unsafe_allow_html=True)

                # Confidence + timing
                cc = _conf_class(resp.confidence_score)
                st.markdown(
                    f'<span class="{cc}">Confidence: {resp.confidence_score:.0%}</span>'
                    f' &nbsp;<small>⏱ {resp.retrieval_time_ms:.0f}ms retrieval'
                    f' | {resp.generation_time_ms:.0f}ms generation</small>',
                    unsafe_allow_html=True,
                )

                # Rewritten query
                if resp.rewritten_query:
                    st.caption(f"✏️ Searched as: *{resp.rewritten_query}*")

                # Citations
                if resp.citations:
                    st.markdown("**Sources:**")
                    for i, cit in enumerate(resp.citations):
                        _render_citation(cit, i)

            report = msg.get("validation")
            if report:
                if report.hallucination_detected:
                    st.warning(f"⚠️ {report.summary}")
                else:
                    st.success(report.summary)

# ── Main panel ────────────────────────────────────────────────────────────────

def _render_main() -> None:
    st.title("🛰️ ISRO Mission Document Assistant")
    st.caption(
        "Multi-turn RAG with cross-encoder re-ranking, semantic caching, "
        "and conversation memory. Every answer is grounded with page-level citations."
    )

    # Render history
    for msg in st.session_state.messages:
        _render_message(msg)

    # Chat input
    if prompt := st.chat_input("Ask about Chandrayaan, Mangalyaan, Aditya-L1, Gaganyaan …"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🛰️"):
            chain = _load_chain(
                st.session_state.use_reranker,
                st.session_state.use_cache,
            )
            if chain is None:
                st.error("RAG chain unavailable. Set ANTHROPIC_API_KEY in .env")
                return

            memory = _get_memory()

            # ── Streaming mode ────────────────────────────────────────────────
            if st.session_state.use_streaming:
                placeholder = st.empty()
                full_text = ""
                try:
                    for token in chain.stream_run(prompt, memory=memory):
                        full_text += token
                        placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)
                    st.session_state.query_count += 1
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                    })
                except Exception as exc:
                    st.error(f"Streaming error: {exc}")
                return

            # ── Standard mode ─────────────────────────────────────────────────
            with st.spinner("Retrieving, re-ranking, generating …"):
                try:
                    from validator import validate_response
                    response, chunks = chain.run_with_chunks(prompt, memory=memory)

                    st.session_state.query_count += 1
                    st.session_state.retrieval_times.append(response.retrieval_time_ms)
                    st.session_state.generation_times.append(response.generation_time_ms)
                    st.session_state.confidence_scores.append(response.confidence_score)
                    if response.cache_hit:
                        st.session_state.cache_hits += 1
                    if response.reranked:
                        st.session_state.reranked_count += 1

                    report = validate_response(response, chunks, question=prompt)

                    # Display
                    st.markdown(response.answer)

                    # Badges
                    badges = []
                    if response.cache_hit:
                        badges.append('<span class="badge-cache">⚡ Cache Hit</span>')
                    if response.reranked:
                        badges.append('<span class="badge-reranked">🔀 Re-ranked</span>')
                    if response.rewritten_query:
                        badges.append('<span class="badge-rewrite">✏️ Query Rewritten</span>')
                    if badges:
                        st.markdown(" ".join(badges), unsafe_allow_html=True)

                    cc = _conf_class(response.confidence_score)
                    st.markdown(
                        f'<span class="{cc}">Confidence: {response.confidence_score:.0%}</span>'
                        f' &nbsp;<small>⏱ {response.retrieval_time_ms:.0f}ms retrieval'
                        f' | {response.generation_time_ms:.0f}ms generation</small>',
                        unsafe_allow_html=True,
                    )

                    if response.rewritten_query:
                        st.caption(f"✏️ Searched as: *{response.rewritten_query}*")

                    if response.citations:
                        st.markdown("**Sources:**")
                        for i, cit in enumerate(response.citations):
                            _render_citation(cit, i)

                    if report.hallucination_detected:
                        st.warning(f"⚠️ {report.summary}")
                    else:
                        st.success(report.summary)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "response": response,
                        "validation": report,
                        "chunks": chunks,
                    })

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {exc}",
                    })

    # Bottom toolbar
    if st.session_state.messages:
        c1, c2, c3 = st.columns([1, 1, 4])
        if c1.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()
        md = _export_chat_markdown()
        c2.download_button(
            "📥 Export", md,
            file_name=f"isro_chat_{st.session_state.session_id}.md",
            mime="text/markdown",
        )

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_main()

main()
