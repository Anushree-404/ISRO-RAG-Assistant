"""
run_project.py  —  Complete end-to-end project run with Gemini 2.5 Flash.
Shows every stage: index stats, retrieval, generation, validation, multi-turn.
"""

import os, sys, time, textwrap
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

W   = 70
SEP = "=" * W

def banner(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

def section(t):
    print(f"\n{'─'*W}\n  {t}\n{'─'*W}")

def wrap(text, indent=4):
    return textwrap.fill(text, width=W,
                         initial_indent=" "*indent,
                         subsequent_indent=" "*indent)

# ─────────────────────────────────────────────────────────────────────────────
banner("ISRO RAG ASSISTANT  —  FULL PROJECT RUN")
print(f"""
  Provider  : {os.getenv('LLM_PROVIDER','gemini').upper()}
  Model     : {os.getenv('GEMINI_MODEL','gemini-2.5-flash')}
  API Key   : {os.getenv('GOOGLE_API_KEY','')[:12]}...{os.getenv('GOOGLE_API_KEY','')[-4:]}
""")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 1 — INDEX STATUS")

from utils import get_chunks_metadata_path, get_faiss_index_path
from embed import load_index, load_metadata

index    = load_index()
metadata = load_metadata()

doc_map: dict = {}
for c in metadata:
    s = c["source_file"]
    if s not in doc_map:
        doc_map[s] = {"chunks":0,"pages":set(),"mission":c["mission_name"]}
    doc_map[s]["chunks"] += 1
    doc_map[s]["pages"].add(c["page_number"])

print(f"\n  {'Document':<42} {'Mission':<16} {'Pages':>5} {'Chunks':>6}")
print(f"  {'─'*42} {'─'*16} {'─'*5} {'─'*6}")
for doc, info in doc_map.items():
    print(f"  {doc:<42} {info['mission']:<16} "
          f"{max(info['pages']):>5} {info['chunks']:>6}")
print(f"\n  Total vectors in FAISS : {index.ntotal}")
print(f"  Embedding model        : all-MiniLM-L6-v2  (dim=384)")
print(f"  Index type             : IndexFlatL2")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 2 — HYBRID RETRIEVAL  (FAISS + BM25 + RRF)")

from retriever import HybridRetriever
retriever = HybridRetriever(top_k=6, index=index, metadata=metadata)

test_queries = [
    "propulsion system Chandrayaan-3 lander engines",
    "Mangalyaan scientific instruments payload Mars",
    "Aditya L1 Lagrange point solar observation",
    "Gaganyaan astronauts crew mission duration",
]

print(f"\n  {'Query':<45} {'Top-1 Document':<35} {'Score':>7}")
print(f"  {'─'*45} {'─'*35} {'─'*7}")
for q in test_queries:
    t0 = time.perf_counter()
    results = retriever.retrieve(q)
    ms = (time.perf_counter()-t0)*1000
    top = results[0]
    print(f"  {q[:44]:<45} {top['source_file'][:34]:<35} {top['retrieval_score']:>7.5f}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 3 — CROSS-ENCODER RE-RANKING")

from reranker import CrossEncoderReranker
reranker = CrossEncoderReranker(top_n=4)

q = "What propulsion system does Chandrayaan-3 use?"
chunks_before = retriever.retrieve(q)
chunks_after  = reranker.rerank(q, list(chunks_before), top_n=4)

print(f"\n  Query: \"{q}\"\n")
print(f"  {'Rank':<5} {'Document':<38} {'Page':>4}  {'RRF Score':>10}  {'Rerank Score':>12}")
print(f"  {'─'*5} {'─'*38} {'─'*4}  {'─'*10}  {'─'*12}")
for i, c in enumerate(chunks_after, 1):
    print(f"  {i:<5} {c['source_file']:<38} {c['page_number']:>4}  "
          f"{c['retrieval_score']:>10.5f}  {c.get('rerank_score',0):>12.4f}")

print(f"\n  ✓ Chunks retrieved: {len(chunks_before)}  →  re-ranked top: {len(chunks_after)}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 4 — LIVE RAG GENERATION  (Gemini 2.5 Flash)")

from generator import RAGChain
from validator import validate_response
from memory   import ConversationMemory
from cache    import get_cache

get_cache().clear()
chain  = RAGChain(use_reranker=True, use_cache=True)
memory = ConversationMemory(session_id="run_project", window=6)
memory.clear()

QUESTIONS = [
    "What is the propulsion system of Chandrayaan-3?",
    "What scientific instruments does Mangalyaan carry?",
    "What is the significance of the L1 Lagrange point for Aditya-L1?",
    "How many astronauts will Gaganyaan carry and what is the mission duration?",
    "What did Chandrayaan-3 discover at the lunar south pole?",
]

results_summary = []

for i, question in enumerate(QUESTIONS, 1):
    section(f"Q{i}: {question}")

    t0 = time.perf_counter()
    response, chunks = chain.run_with_chunks(question, memory=memory)
    total_ms = (time.perf_counter()-t0)*1000

    report = validate_response(response, chunks, question)

    # Print answer
    print(f"\n  Answer:")
    for line in textwrap.wrap(response.answer, width=64):
        print(f"    {line}")

    # Print metadata
    print(f"\n  ┌─ Metadata ──────────────────────────────────────────┐")
    print(f"  │  Confidence     : {response.confidence_score:.0%}")
    print(f"  │  Retrieval time : {response.retrieval_time_ms:.0f} ms")
    print(f"  │  Generation time: {response.generation_time_ms:.0f} ms")
    print(f"  │  Total time     : {total_ms:.0f} ms")
    print(f"  │  Re-ranked      : {'✓' if response.reranked else '✗'}")
    print(f"  │  Cache hit      : {'✓ (instant)' if response.cache_hit else '✗'}")
    if response.rewritten_query:
        print(f"  │  Query rewrite  : {response.rewritten_query[:55]}")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Print citations
    print(f"\n  Citations ({len(response.citations)}):")
    for j, cit in enumerate(response.citations, 1):
        print(f"    [{j}] {cit.doc}  —  page {cit.page}")
        print(f"        \"{cit.chunk_text[:90].replace(chr(10),' ')} …\"")

    # Validation
    v_icon = "✅ GROUNDED" if not report.hallucination_detected else "⚠️  NEEDS REVIEW"
    print(f"\n  Validation : {v_icon}  "
          f"({report.valid_citations}/{report.total_citations} citations verified)")

    results_summary.append({
        "q": question[:50],
        "conf": response.confidence_score,
        "rt": response.retrieval_time_ms,
        "gt": response.generation_time_ms,
        "cit": len(response.citations),
        "valid": report.valid_citations,
        "total": report.total_citations,
        "grounded": not report.hallucination_detected,
        "cache": response.cache_hit,
        "reranked": response.reranked,
    })

    if i < len(QUESTIONS):
        print(f"\n  ⏳ Waiting 5s (rate limit) …")
        time.sleep(5)

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 5 — MULTI-TURN CONVERSATION MEMORY")

mem2 = ConversationMemory(session_id="run_project_mt", window=4)
mem2.clear()

followups = [
    ("What is Chandrayaan-3?",          False),
    ("What instruments does it carry?", True),   # follow-up
    ("What did it discover?",           True),   # follow-up
]

print()
for q, is_followup in followups:
    resp, chunks = chain.run_with_chunks(q, memory=mem2)
    rw = f"\n    ✏️  Rewritten → \"{resp.rewritten_query}\"" if resp.rewritten_query else ""
    print(f"  👤 {q}{rw}")
    print(f"  🛰️  {resp.answer[:180].replace(chr(10),' ')} …")
    print(f"     conf={resp.confidence_score:.0%}  citations={len(resp.citations)}  "
          f"cache={'HIT' if resp.cache_hit else 'miss'}")
    print()
    time.sleep(3)

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 6 — SEMANTIC CACHE DEMO")

cache = get_cache()
print(f"\n  Cache populated from Stage 4: {cache.size} entries\n")

cache_tests = [
    ("What is the propulsion system of Chandrayaan-3?",   "EXACT MATCH"),
    ("Chandrayaan-3 propulsion and engines",               "SEMANTIC MATCH"),
    ("What is the weather on Mars?",                       "MISS"),
]

print(f"  {'Test':<20} {'Query':<48} {'Result':>8}  {'Time':>7}")
print(f"  {'─'*20} {'─'*48} {'─'*8}  {'─'*7}")
for label, q, _ in [(a,b,c) for a,b,c in [(x[1],x[0],None) for x in cache_tests]]:
    t0 = time.perf_counter()
    hit = cache.get(q)
    ms  = (time.perf_counter()-t0)*1000
    res = "HIT ✅" if hit else "MISS ❌"
    print(f"  {label:<20} {q[:47]:<48} {res:>8}  {ms:>5.1f}ms")

# ─────────────────────────────────────────────────────────────────────────────
banner("FINAL SUMMARY")

grounded_count = sum(1 for r in results_summary if r["grounded"])
avg_conf  = sum(r["conf"] for r in results_summary) / len(results_summary)
avg_rt    = sum(r["rt"]   for r in results_summary) / len(results_summary)
avg_gt    = sum(r["gt"]   for r in results_summary) / len(results_summary)
reranked  = sum(1 for r in results_summary if r["reranked"])

def bar(v, w=28):
    return "█"*int(v*w) + "░"*(w-int(v*w))

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  ISRO RAG ASSISTANT  —  RUN COMPLETE                             │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  LLM Provider      : Gemini 2.5 Flash (google-genai SDK)         │
  │  Documents indexed : {len(doc_map):<44}│
  │  Total chunks      : {len(metadata):<44}│
  │  FAISS vectors     : {index.ntotal:<44}│
  │                                                                  │
  │  Questions asked   : {len(results_summary):<44}│
  │  Fully grounded    : {grounded_count}/{len(results_summary)}  ({grounded_count/len(results_summary):.0%}){' '*38}│
  │  Re-ranked         : {reranked}/{len(results_summary)} queries used cross-encoder{' '*22}│
  │  Cache entries     : {cache.size:<44}│
  │                                                                  │
  │  Avg confidence    : {avg_conf:.0%}  {bar(avg_conf):<30}│
  │  Avg retrieval     : {avg_rt:.0f} ms{' '*(44-len(f'{avg_rt:.0f} ms'))}│
  │  Avg generation    : {avg_gt:.0f} ms{' '*(44-len(f'{avg_gt:.0f} ms'))}│
  │                                                                  │
  ├──────────────────────────────────────────────────────────────────┤
  │  STATUS : ✅  PROJECT RUNNING SUCCESSFULLY                        │
  └──────────────────────────────────────────────────────────────────┘

  Dashboard : streamlit run dashboard/app.py  →  http://localhost:8501
  REST API  : uvicorn src.api:app --port 8000  →  http://localhost:8000/docs
""")
