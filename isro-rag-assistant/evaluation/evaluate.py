"""
evaluate.py — RAGAS evaluation pipeline.

Runs all 30 QA pairs through the RAG pipeline and measures:
  - Faithfulness       : Are claims in the answer supported by the context?
  - Answer Relevancy   : Is the answer relevant to the question?
  - Context Recall     : Does the retrieved context cover the ground truth?

Outputs a formatted metrics table and saves results to
evaluation/results/ragas_results.json.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --limit 5   # quick smoke test
    python evaluation/evaluate.py --output evaluation/results/my_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)
from tqdm import tqdm

from utils import configure_logging, load_json, save_json

configure_logging()

# ── Paths ─────────────────────────────────────────────────────────────────────
QA_PAIRS_PATH = Path(__file__).parent / "qa_pairs.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_on_question(
    question: str,
    chain: Any,
) -> dict[str, Any]:
    """
    Run the RAG chain on *question* and return a dict with:
      question, answer, contexts (list of chunk texts), ground_truth (empty here).
    """
    try:
        response, chunks = chain.run_with_chunks(question)
        contexts = [c["text"] for c in chunks]
        return {
            "question": question,
            "answer": response.answer,
            "contexts": contexts,
            "confidence_score": response.confidence_score,
            "retrieval_time_ms": response.retrieval_time_ms,
            "generation_time_ms": response.generation_time_ms,
            "citations": [c.model_dump() for c in response.citations],
        }
    except Exception as exc:
        logger.error(f"Pipeline failed for question '{question[:60]}': {exc}")
        return {
            "question": question,
            "answer": f"ERROR: {exc}",
            "contexts": [],
            "confidence_score": 0.0,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "citations": [],
        }


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def build_ragas_dataset(
    pipeline_results: list[dict[str, Any]],
    qa_pairs: list[dict[str, Any]],
) -> Dataset:
    """
    Build a HuggingFace Dataset in the format RAGAS expects.

    RAGAS requires columns: question, answer, contexts, ground_truth.
    """
    rows: list[dict[str, Any]] = []
    qa_map = {qa["question"]: qa.get("ground_truth", "") for qa in qa_pairs}

    for result in pipeline_results:
        rows.append({
            "question": result["question"],
            "answer": result["answer"],
            "contexts": result["contexts"] if result["contexts"] else ["No context retrieved."],
            "ground_truth": qa_map.get(result["question"], ""),
        })

    return Dataset.from_list(rows)


def run_ragas_evaluation(dataset: Dataset) -> dict[str, float]:
    """
    Run RAGAS metrics on *dataset*.

    Returns dict of metric_name → score.
    """
    logger.info("Running RAGAS evaluation …")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    return dict(result)


# ── Metrics table printer ─────────────────────────────────────────────────────

def print_metrics_table(
    metrics: dict[str, float],
    pipeline_results: list[dict[str, Any]],
) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 65)
    print("  RAGAS EVALUATION RESULTS — ISRO RAG ASSISTANT")
    print("=" * 65)

    metric_display = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
    }

    for key, label in metric_display.items():
        score = metrics.get(key, float("nan"))
        bar_len = int(score * 30) if not (score != score) else 0  # nan check
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<22} {bar}  {score:.4f}")

    print("-" * 65)

    # Pipeline stats
    times_r = [r["retrieval_time_ms"] for r in pipeline_results if r["retrieval_time_ms"] > 0]
    times_g = [r["generation_time_ms"] for r in pipeline_results if r["generation_time_ms"] > 0]
    confs = [r["confidence_score"] for r in pipeline_results]
    errors = sum(1 for r in pipeline_results if r["answer"].startswith("ERROR"))

    print(f"  Questions evaluated : {len(pipeline_results)}")
    print(f"  Errors              : {errors}")
    if times_r:
        print(f"  Avg retrieval time  : {sum(times_r)/len(times_r):.0f} ms")
    if times_g:
        print(f"  Avg generation time : {sum(times_g)/len(times_g):.0f} ms")
    if confs:
        print(f"  Avg confidence      : {sum(confs)/len(confs):.3f}")

    print("=" * 65 + "\n")


# ── Per-question detail table ─────────────────────────────────────────────────

def print_detail_table(pipeline_results: list[dict[str, Any]]) -> None:
    """Print per-question answer preview."""
    print("\n" + "-" * 65)
    print("  PER-QUESTION SUMMARY")
    print("-" * 65)
    for i, r in enumerate(pipeline_results, 1):
        q = r["question"][:55]
        a = r["answer"][:60].replace("\n", " ")
        conf = r["confidence_score"]
        n_ctx = len(r["contexts"])
        n_cit = len(r["citations"])
        print(f"  [{i:02d}] Q: {q}")
        print(f"       A: {a} …")
        print(f"       conf={conf:.2f}  ctx={n_ctx}  citations={n_cit}")
    print("-" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on QA pairs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs to evaluate (default: all 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "ragas_results.json",
        help="Output JSON path for detailed results",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS scoring (just run pipeline and print answers)",
    )
    args = parser.parse_args()

    # Load QA pairs
    qa_pairs: list[dict[str, Any]] = load_json(QA_PAIRS_PATH)  # type: ignore[assignment]
    if not qa_pairs:
        logger.error(f"No QA pairs found at {QA_PAIRS_PATH}")
        sys.exit(1)

    if args.limit:
        qa_pairs = qa_pairs[: args.limit]
        logger.info(f"Limiting evaluation to {len(qa_pairs)} QA pairs")

    # Load chain
    logger.info("Loading RAG chain …")
    try:
        from generator import RAGChain
        chain = RAGChain()
    except Exception as exc:
        logger.error(f"Failed to load RAG chain: {exc}")
        sys.exit(1)

    # Run pipeline
    logger.info(f"Running pipeline on {len(qa_pairs)} questions …")
    pipeline_results: list[dict[str, Any]] = []

    for qa in tqdm(qa_pairs, desc="Evaluating"):
        result = run_pipeline_on_question(qa["question"], chain)
        result["ground_truth"] = qa.get("ground_truth", "")
        result["mission"] = qa.get("mission", "")
        result["id"] = qa.get("id", "")
        pipeline_results.append(result)
        time.sleep(0.5)  # Rate limiting

    print_detail_table(pipeline_results)

    # RAGAS evaluation
    ragas_metrics: dict[str, float] = {}
    if not args.skip_ragas:
        try:
            dataset = build_ragas_dataset(pipeline_results, qa_pairs)
            ragas_metrics = run_ragas_evaluation(dataset)
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            logger.info("Falling back to basic metrics …")
            ragas_metrics = _compute_basic_metrics(pipeline_results)
    else:
        ragas_metrics = _compute_basic_metrics(pipeline_results)

    print_metrics_table(ragas_metrics, pipeline_results)

    # Save results
    output_data = {
        "metrics": ragas_metrics,
        "pipeline_results": pipeline_results,
        "num_questions": len(pipeline_results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(output_data, args.output)
    logger.success(f"Results saved → {args.output}")


def _compute_basic_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute simple proxy metrics when RAGAS is unavailable.
    These are rough approximations, not the real RAGAS scores.
    """
    total = len(results)
    if total == 0:
        return {}

    # Faithfulness proxy: fraction of answers with ≥1 citation
    with_citations = sum(1 for r in results if r["citations"])
    faithfulness_proxy = with_citations / total

    # Answer relevancy proxy: fraction with confidence > 0.3
    relevant = sum(1 for r in results if r["confidence_score"] > 0.3)
    relevancy_proxy = relevant / total

    # Context recall proxy: fraction with ≥3 context chunks
    good_context = sum(1 for r in results if len(r["contexts"]) >= 3)
    recall_proxy = good_context / total

    return {
        "faithfulness": round(faithfulness_proxy, 4),
        "answer_relevancy": round(relevancy_proxy, 4),
        "context_recall": round(recall_proxy, 4),
    }


if __name__ == "__main__":
    main()
