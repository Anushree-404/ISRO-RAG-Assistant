"""
scripts/test_gemini.py
======================
Tests the full RAG pipeline end-to-end using Google Gemini.
Runs 5 real questions and prints structured answers with citations.
"""

import os, sys, time
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEP  = "=" * 70
SEP2 = "-" * 70

print(SEP)
print("  ISRO RAG ASSISTANT — GEMINI LIVE TEST")
print(SEP)

# ── Step 1: verify env ────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os
key = os.getenv("GOOGLE_API_KEY", "")
provider = os.getenv("LLM_PROVIDER", "gemini")
model    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

print(f"\n  Provider : {provider}")
print(f"  Model    : {model}")
print(f"  API Key  : {key[:12]}...{key[-4:]}  (length={len(key)})")

if not key or key == "your_gemini_api_key_here":
    print("\n  ✗ GOOGLE_API_KEY not set in .env")
    sys.exit(1)

# ── Step 2: load chain ────────────────────────────────────────────────────────
print(f"\n  Loading RAGChain with Gemini …")
from generator import RAGChain
from validator import validate_response

chain = RAGChain(use_reranker=True, use_cache=True)
print(f"  ✓ Chain ready\n")

# ── Step 3: run 5 questions ───────────────────────────────────────────────────
QUESTIONS = [
    "What is the propulsion system of Chandrayaan-3?",
    "What scientific instruments does Mangalyaan carry?",
    "What is the significance of the L1 Lagrange point for Aditya-L1?",
    "How many astronauts will Gaganyaan carry and what is the mission duration?",
    "What did Chandrayaan-3 discover at the lunar south pole?",
]

from memory import ConversationMemory
memory = ConversationMemory(session_id="gemini_test", window=4)
memory.clear()

for i, question in enumerate(QUESTIONS, 1):
    print(f"{SEP2}")
    print(f"  Q{i}: {question}")
    print(SEP2)

    response, chunks = chain.run_with_chunks(question, memory=memory)
    report = validate_response(response, chunks, question)
    # Answer
    print(f"\n  Answer:")
    for line in response.answer.split(". "):
        line = line.strip()
        if line:
            print(f"    {line}.")

    # Metadata
    print(f"\n  Confidence     : {response.confidence_score:.0%}")
    print(f"  Retrieval time : {response.retrieval_time_ms:.0f} ms")
    print(f"  Generation time: {response.generation_time_ms:.0f} ms")
    print(f"  Re-ranked      : {response.reranked}")
    print(f"  Cache hit      : {response.cache_hit}")
    if response.rewritten_query:
        print(f"  Rewritten query: {response.rewritten_query}")

    # Citations
    print(f"\n  Citations ({len(response.citations)}):")
    for j, cit in enumerate(response.citations, 1):
        print(f"    [{j}] {cit.doc}  page {cit.page}")
        print(f"        \"{cit.chunk_text[:100].replace(chr(10),' ')} …\"")

    # Validation
    status = "✅ GROUNDED" if not report.hallucination_detected else "⚠️  HALLUCINATION"
    print(f"\n  Validation: {status}  |  "
          f"{report.valid_citations}/{report.total_citations} citations valid")
    if i < len(QUESTIONS):
        print(f"  (waiting 5s before next question …)\n")
        time.sleep(5)

# ── Step 4: multi-turn follow-up ──────────────────────────────────────────────
print(SEP)
print("  MULTI-TURN FOLLOW-UP TEST")
print(SEP)

followups = [
    "What is Chandrayaan-3?",
    "What about its rover?",
    "What did it discover?",
]

mem2 = ConversationMemory(session_id="gemini_followup", window=4)
mem2.clear()

for q in followups:
    resp, chunks = chain.run_with_chunks(q, memory=mem2)
    rw = f"  → rewritten: '{resp.rewritten_query}'" if resp.rewritten_query else ""
    print(f"\n  Q: {q}{rw}")
    print(f"  A: {resp.answer[:200].replace(chr(10),' ')} …")
    print(f"     conf={resp.confidence_score:.0%}  citations={len(resp.citations)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ✅ ALL TESTS COMPLETE — Gemini is working with ISRO RAG Assistant")
print(f"  Provider : {provider}  |  Model : {model}")
print(SEP)
