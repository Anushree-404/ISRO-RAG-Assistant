"""
scripts/demo_pipeline.py
Full offline demo: shows retrieval + mock generation + validation output.
Run this to see the complete pipeline working without an API key.
For real LLM answers, set ANTHROPIC_API_KEY in .env and use generator.py directly.
"""

import sys, json, time
sys.path.insert(0, "src")

from retriever import HybridRetriever
from generator import Citation, RAGResponse, _format_context
from validator import validate_response, validate_and_log

SEPARATOR = "=" * 70

def mock_generate(question: str, chunks: list) -> RAGResponse:
    """
    Simulate what Claude would return — builds a grounded answer
    directly from the top retrieved chunk so citations are always valid.
    """
    top = chunks[0]
    second = chunks[1] if len(chunks) > 1 else chunks[0]

    answer = (
        f"Based on the indexed ISRO mission documents: {top['text'][:300].strip()} "
        f"Additionally, {second['text'][:200].strip()}"
    )

    citations = [
        Citation(
            doc=top["source_file"],
            page=top["page_number"],
            chunk_text=top["text"][:200].strip(),
        ),
        Citation(
            doc=second["source_file"],
            page=second["page_number"],
            chunk_text=second["text"][:200].strip(),
        ),
    ]

    return RAGResponse(
        answer=answer,
        citations=citations,
        confidence_score=0.87,
        retrieval_time_ms=12.4,
        generation_time_ms=0.0,
    )


def run_demo():
    print(SEPARATOR)
    print("  ISRO RAG ASSISTANT — FULL PIPELINE DEMO")
    print("  (Mock generation — set ANTHROPIC_API_KEY for real Claude answers)")
    print(SEPARATOR)

    # ── Load retriever ────────────────────────────────────────────────────────
    print("\n[1/4] Loading HybridRetriever (FAISS + BM25) …")
    retriever = HybridRetriever(top_k=6)
    print(f"      ✓ Index loaded: {retriever.index.ntotal} vectors, "
          f"{len(retriever.metadata)} chunks across "
          f"{len({m['source_file'] for m in retriever.metadata})} documents\n")

    # ── Show indexed documents ────────────────────────────────────────────────
    print("[2/4] Indexed documents:")
    doc_stats: dict = {}
    for c in retriever.metadata:
        src = c["source_file"]
        if src not in doc_stats:
            doc_stats[src] = {"chunks": 0, "mission": c["mission_name"], "pages": set()}
        doc_stats[src]["chunks"] += 1
        doc_stats[src]["pages"].add(c["page_number"])

    for doc, info in doc_stats.items():
        print(f"      📄 {doc}")
        print(f"         Mission: {info['mission']}  |  "
              f"Chunks: {info['chunks']}  |  Pages: {max(info['pages'])}")
    print()

    # ── Run queries ───────────────────────────────────────────────────────────
    queries = [
        "What is the propulsion system of Chandrayaan-3?",
        "What scientific instruments does Mangalyaan carry?",
        "What is the significance of the L1 point for Aditya-L1?",
        "How many astronauts will Gaganyaan carry and what is the mission duration?",
        "What did Chandrayaan-3 discover at the lunar south pole?",
    ]

    print("[3/4] Running RAG pipeline on 5 queries …")
    print(SEPARATOR)

    all_reports = []
    for qi, question in enumerate(queries, 1):
        print(f"\nQuery {qi}: {question}")
        print("-" * 60)

        # Retrieve
        t0 = time.perf_counter()
        chunks = retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        print(f"  Retrieved {len(chunks)} chunks in {retrieval_ms:.1f} ms:")
        for i, c in enumerate(chunks[:3], 1):
            print(f"    [{i}] {c['source_file']} p.{c['page_number']}  "
                  f"score={c['retrieval_score']:.5f}")
            print(f"        {c['text'][:90].replace(chr(10),' ').strip()} …")

        # Generate (mock)
        response = mock_generate(question, chunks)
        response.retrieval_time_ms = round(retrieval_ms, 1)

        print(f"\n  Answer (first 300 chars):")
        print(f"    {response.answer[:300].replace(chr(10),' ').strip()} …")
        print(f"\n  Confidence: {response.confidence_score:.0%}  |  "
              f"Citations: {len(response.citations)}")

        # Validate
        report = validate_response(response, chunks, question)
        all_reports.append(report)

        status = "✅ GROUNDED" if not report.hallucination_detected else "⚠️  HALLUCINATION"
        print(f"  Validation: {status}  |  "
              f"Valid citations: {report.valid_citations}/{report.total_citations}  |  "
              f"Hallucination rate: {report.hallucination_rate:.0%}")

        for v in report.verdicts:
            icon = "✅" if v.is_valid else "❌"
            print(f"    {icon} Citation {v.citation_index+1}: "
                  f"{v.doc} p.{v.page}  sim={v.text_similarity:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + SEPARATOR)
    print("[4/4] PIPELINE SUMMARY")
    print(SEPARATOR)
    total_q = len(all_reports)
    grounded = sum(1 for r in all_reports if not r.hallucination_detected)
    avg_valid = sum(r.valid_citations for r in all_reports) / total_q
    avg_total = sum(r.total_citations for r in all_reports) / total_q

    print(f"  Queries run          : {total_q}")
    print(f"  Fully grounded       : {grounded}/{total_q}  ({grounded/total_q:.0%})")
    print(f"  Avg valid citations  : {avg_valid:.1f} / {avg_total:.1f}")
    print(f"  Index size           : {retriever.index.ntotal} vectors")
    print(f"  Documents indexed    : {len(doc_stats)}")
    print()
    print("  To use real Claude answers:")
    print("    1. Add ANTHROPIC_API_KEY to .env")
    print("    2. Run: streamlit run dashboard/app.py")
    print("    3. Or:  python evaluation/evaluate.py --limit 5")
    print(SEPARATOR)


if __name__ == "__main__":
    run_demo()
