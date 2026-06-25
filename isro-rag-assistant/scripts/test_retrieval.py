"""Quick retrieval smoke test."""
import sys
sys.path.insert(0, "src")

from retriever import HybridRetriever

r = HybridRetriever(top_k=6)
queries = [
    "What is the propulsion system of Chandrayaan-3?",
    "What instruments does Mangalyaan carry?",
    "What is the L1 Lagrange point for Aditya-L1?",
    "How many astronauts will Gaganyaan carry?",
]

for q in queries:
    print(f"\nQ: {q}")
    results = r.retrieve(q)
    for i, c in enumerate(results, 1):
        src = c["source_file"]
        pg  = c["page_number"]
        sc  = c["retrieval_score"]
        txt = c["text"][:110].replace("\n", " ").strip()
        print(f"  [{i}] {src} p.{pg}  score={sc:.5f}  |  {txt}...")
