import os, sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv(".env")

# Clear cache
from cache import get_cache
get_cache().clear()
print("Cache cleared\n")

# Test _extract_json
from generator import _extract_json

tests = [
    # (label, input, expected_answer_prefix)
    (
        "Clean JSON",
        '{"answer": "Chandrayaan-3 uses a 440N LAM.", "citations": [{"doc": "c3.pdf", "page": 1, "chunk_text": "440N motor"}], "confidence_score": 0.95}',
        "Chandrayaan-3"
    ),
    (
        "Markdown fenced JSON",
        '```json\n{"answer": "Fenced answer", "citations": [], "confidence_score": 0.8}\n```',
        "Fenced"
    ),
    (
        "JSON embedded in text",
        'Here is my response: {"answer": "Embedded answer", "citations": [], "confidence_score": 0.7} end',
        "Embedded"
    ),
    (
        "No JSON fallback",
        "This is just plain text with no JSON",
        "This is just"
    ),
]

all_passed = True
for label, inp, expected in tests:
    result = _extract_json(inp)
    ok = result["answer"].startswith(expected)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_passed = False
    print(f"  [{status}] {label}")
    print(f"         answer   : {result['answer'][:60]}")
    print(f"         citations: {len(result['citations'])}")
    print(f"         confidence: {result['confidence_score']}")
    print()

print("=" * 50)
print("All tests PASSED" if all_passed else "SOME TESTS FAILED")

# Now do a real live Gemini call to verify end-to-end
print("\n" + "=" * 50)
print("LIVE GEMINI TEST")
print("=" * 50)

from generator import RAGChain
from validator import validate_response

chain = RAGChain(use_reranker=False, use_cache=False)
response, chunks = chain.run_with_chunks("What is Chandrayaan-3?")

print(f"\nAnswer:\n{response.answer}\n")
print(f"Confidence    : {response.confidence_score:.0%}")
print(f"Citations     : {len(response.citations)}")
for i, c in enumerate(response.citations, 1):
    print(f"  [{i}] {c.doc}  page {c.page}")
    print(f"       {c.chunk_text[:80]} ...")

report = validate_response(response, chunks, "What is Chandrayaan-3?")
print(f"\nValidation    : {'GROUNDED' if not report.hallucination_detected else 'NEEDS REVIEW'}")
print(f"Valid citations: {report.valid_citations}/{report.total_citations}")
print(f"\nSummary: {report.summary}")
