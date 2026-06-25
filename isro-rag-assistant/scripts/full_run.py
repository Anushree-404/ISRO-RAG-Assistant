"""
full_run.py — Complete project run: ingest → embed → retrieve → generate → validate
Shows clean formatted output for every stage.
"""
import os, sys, time, textwrap
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

W = 68
def banner(t):  print(f"\n{'='*W}\n  {t}\n{'='*W}")
def sep(t):     print(f"\n{'─'*W}\n  {t}\n{'─'*W}")
def wrap(t, n=4): return textwrap.fill(t, W, initial_indent=" "*n, subsequent_indent=" "*n)

# ─────────────────────────────────────────────────────────────────────────────
banner("ISRO RAG ASSISTANT — COMPLETE PROJECT RUN")
print(f"\n  Provider : {os.getenv('LLM_PROVIDER','gemini').upper()}")
print(f"  Model    : {os.getenv('GEMINI_MODEL','gemini-flash-latest')}")
print(f"  API Key  : {os.getenv('GOOGLE_API_KEY','')[:12]}...{os.getenv('GOOGLE_API_KEY','')[-4:]}")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 1 — PDF INGESTION")

from ingest import ingest_directory, save_chunks
from utils import get_data_raw_dir, get_chunks_path, save_json, load_json

raw_dir = get_data_raw_dir()
pdfs = sorted(raw_dir.glob("*.pdf"))
print(f"\n  PDFs in data/raw/: {len(pdfs)}")
for p in pdfs:
    print(f"    📄 {p.name}  ({p.stat().st_size//1024} KB)")

t0 = time.perf_counter()
chunks = ingest_directory(raw_dir)
save_json(chunks, get_chunks_path())
ingest_ms = (time.perf_counter()-t0)*1000

doc_map: dict = {}
for c in chunks:
    s = c["source_file"]
    if s not in doc_map:
        doc_map[s] = {"chunks":0,"pages":set(),"mission":c["mission_name"]}
    doc_map[s]["chunks"] += 1
    doc_map[s]["pages"].add(c["page_number"])

print(f"\n  {'Document':<40} {'Mission':<16} {'Pages':>5} {'Chunks':>6}")
print(f"  {'─'*40} {'─'*16} {'─'*5} {'─'*6}")
for doc, info in doc_map.items():
    print(f"  {doc:<40} {info['mission']:<16} {max(info['pages']):>5} {info['chunks']:>6}")
print(f"\n  ✓ Total chunks : {len(chunks)}  |  Time : {ingest_ms:.0f} ms")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 2 — EMBEDDING + FAISS INDEX")

from embed import build_embeddings, get_faiss_index_path

t0 = time.perf_counter()
index, metadata = build_embeddings(rebuild=True)
embed_ms = (time.perf_counter()-t0)*1000

print(f"\n  ✓ Vectors in FAISS : {index.ntotal}")
print(f"  ✓ Embedding model  : all-MiniLM-L6-v2  (dim=384)")
print(f"  ✓ Index type       : IndexFlatL2")
print(f"  ✓ Saved to         : {get_faiss_index_path()}")
print(f"  ✓ Time             : {embed_ms:.0f} ms")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 3 — HYBRID RETRIEVAL  (FAISS + BM25 + RRF)")

from retriever import HybridRetriever
retriever = HybridRetriever(top_k=6, index=index, metadata=metadata)

queries = [
    ("What is the propulsion system of Chandrayaan-3?",        "chandrayaan3"),
    ("What scientific instruments does Mangalyaan carry?",      "mangalyaan"),
    ("What is the L1 Lagrange point significance for Aditya?",  "aditya"),
    ("How many astronauts will Gaganyaan carry?",               "gaganyaan"),
]

print(f"\n  {'Query':<50} {'Top Doc':<28} {'Score':>7}")
print(f"  {'─'*50} {'─'*28} {'─'*7}")
rt_list = []
for q, _ in queries:
    t0 = time.perf_counter()
    res = retriever.retrieve(q)
    rt_list.append((time.perf_counter()-t0)*1000)
    top = res[0]
    print(f"  {q[:49]:<50} {top['source_file'][:27]:<28} {top['retrieval_score']:>7.5f}")

print(f"\n  ✓ Avg retrieval time : {sum(rt_list)/len(rt_list):.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 4 — LIVE GEMINI GENERATION  (gemini-flash-latest)")

from generator import RAGChain
from validator import validate_response
from memory   import ConversationMemory
from cache    import get_cache

get_cache().clear()
chain  = RAGChain(use_reranker=False, use_cache=True)
memory = ConversationMemory(session_id="full_run", window=6)
memory.clear()

QUESTIONS = [
    "What is the propulsion system of Chandrayaan-3?",
    "What scientific instruments does Mangalyaan carry?",
    "What is the significance of the L1 Lagrange point for Aditya-L1?",
    "How many astronauts will Gaganyaan carry and what is the mission duration?",
    "What did Chandrayaan-3 discover at the lunar south pole?",
]

results = []
for i, question in enumerate(QUESTIONS, 1):
    sep(f"Q{i}: {question}")

    t0 = time.perf_counter()
    response, chunks = chain.run_with_chunks(question, memory=memory)
    total_ms = (time.perf_counter()-t0)*1000
    report = validate_response(response, chunks, question)

    print(f"\n  Answer:")
    for line in textwrap.wrap(response.answer, W-4):
        print(f"    {line}")

    print(f"\n  ┌─ Stats {'─'*50}┐")
    print(f"  │  Confidence     : {response.confidence_score:.0%}")
    print(f"  │  Retrieval      : {response.retrieval_time_ms:.0f} ms")
    print(f"  │  Generation     : {response.generation_time_ms:.0f} ms")
    print(f"  │  Total          : {total_ms:.0f} ms")
    print(f"  │  Cache hit      : {'YES ⚡' if response.cache_hit else 'no'}")
    if response.rewritten_query:
        print(f"  │  Query rewrite  : {response.rewritten_query[:50]}")
    print(f"  └{'─'*57}┘")

    print(f"\n  Citations ({len(response.citations)}):")
    for j, cit in enumerate(response.citations, 1):
        print(f"    [{j}] {cit.doc}  —  page {cit.page}")
        print(f"         \"{cit.chunk_text[:80].replace(chr(10),' ')} ...\"")

    icon = "✅ GROUNDED" if not report.hallucination_detected else "⚠️  NEEDS REVIEW"
    print(f"\n  Validation : {icon}  ({report.valid_citations}/{report.total_citations} citations verified)")

    results.append({
        "q": question[:50], "conf": response.confidence_score,
        "rt": response.retrieval_time_ms, "gt": response.generation_time_ms,
        "cit": len(response.citations), "valid": report.valid_citations,
        "grounded": not report.hallucination_detected, "cache": response.cache_hit,
    })

    if i < len(QUESTIONS):
        print(f"\n  ⏳ Waiting 4s ...")
        time.sleep(4)

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 5 — MULTI-TURN MEMORY TEST")

mem2 = ConversationMemory(session_id="full_run_mt", window=4)
mem2.clear()

turns = [
    "What is Chandrayaan-3?",
    "What instruments does it carry?",
    "What did it discover?",
]
print()
for q in turns:
    resp, _ = chain.run_with_chunks(q, memory=mem2)
    rw = f"\n    ✏️  → \"{resp.rewritten_query}\"" if resp.rewritten_query else ""
    print(f"  👤 {q}{rw}")
    print(f"  🛰️  {resp.answer[:160].replace(chr(10),' ')} ...")
    print(f"     conf={resp.confidence_score:.0%}  citations={len(resp.citations)}  cache={'HIT ⚡' if resp.cache_hit else 'miss'}")
    print()
    time.sleep(3)

# ─────────────────────────────────────────────────────────────────────────────
banner("STAGE 6 — CACHE TEST")

cache = get_cache()
print(f"\n  Cache has {cache.size} entries from Stage 4\n")

cache_tests = [
    ("EXACT MATCH",    "What is the propulsion system of Chandrayaan-3?"),
    ("SIMILAR QUERY",  "Chandrayaan-3 propulsion and engines"),
    ("UNRELATED",      "What is the weather forecast for tomorrow?"),
]
print(f"  {'Type':<15} {'Query':<50} {'Result':>8}  {'Time':>7}")
print(f"  {'─'*15} {'─'*50} {'─'*8}  {'─'*7}")
for label, q in cache_tests:
    t0 = time.perf_counter()
    hit = cache.get(q)
    ms  = (time.perf_counter()-t0)*1000
    res = "HIT ✅" if hit else "MISS ❌"
    print(f"  {label:<15} {q[:49]:<50} {res:>8}  {ms:>5.1f}ms")

# ─────────────────────────────────────────────────────────────────────────────
banner("FINAL SUMMARY")

grounded = sum(1 for r in results if r["grounded"])
avg_conf = sum(r["conf"] for r in results)/len(results)
avg_rt   = sum(r["rt"]   for r in results)/len(results)
avg_gt   = sum(r["gt"]   for r in results)/len(results)

def bar(v, w=26): return "█"*int(v*w) + "░"*(w-int(v*w))

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  ISRO RAG ASSISTANT — RUN COMPLETE                           │
  ├──────────────────────────────────────────────────────────────┤
  │  Provider       : gemini-flash-latest (google-genai SDK)     │
  │  Documents      : {len(doc_map):<44}│
  │  Chunks         : {len(chunks):<44}│
  │  FAISS vectors  : {index.ntotal:<44}│
  │  Questions      : {len(results):<44}│
  │  Grounded       : {grounded}/{len(results)}  ({grounded/len(results):.0%}){' '*38}│
  │  Cache entries  : {cache.size:<44}│
  │                                                              │
  │  Avg confidence : {avg_conf:.0%}  {bar(avg_conf):<28}│
  │  Avg retrieval  : {avg_rt:.0f} ms{' '*(44-len(f'{avg_rt:.0f} ms'))}│
  │  Avg generation : {avg_gt:.0f} ms{' '*(44-len(f'{avg_gt:.0f} ms'))}│
  ├──────────────────────────────────────────────────────────────┤
  │  STATUS : ✅  ALL STAGES COMPLETE                             │
  └──────────────────────────────────────────────────────────────┘

  Run dashboard : python -m streamlit run dashboard/app.py
  Open browser  : http://localhost:8501
""")
