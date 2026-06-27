# 🛸 ISRO RAG Assistant

**A production-quality Retrieval-Augmented Generation system for querying ISRO mission documents**

---

## 📌 Overview

ISRO RAG Assistant lets you ask natural-language questions against ISRO mission documents
(Chandrayaan-3, Mangalyaan, Aditya-L1, Gaganyaan, and more).
Every answer comes with **grounded citations**, a **confidence score**, and an automatic
**hallucination check**.

**Example:**

```json
{
  "answer": "The Chandrayaan-3 propulsion module uses a 440 N Liquid Apogee Motor...",
  "citations": [
    {
      "doc": "chandrayaan3_mission_report.pdf",
      "page": 1,
      "chunk_text": "It uses a 440 N liquid apogee motor (LAM)..."
    }
  ],
  "confidence_score": 1.0
}
```

---

## ✨ Features

| Feature | Details |
| --- | --- |
| 🔍 Hybrid Retrieval | FAISS dense + BM25 keyword search merged with Reciprocal Rank Fusion |
| 🔀 Cross-Encoder Re-ranking | ms-marco-MiniLM re-scores candidates for higher precision |
| 💬 Conversation Memory | Sliding-window multi-turn memory with automatic query rewriting |
| ⚡ Semantic Cache | Exact + cosine-similarity cache avoids redundant API calls |
| 📋 Structured Output | Every answer returns `{answer, citations, confidence_score}` as JSON |
| ✅ Hallucination Detection | Fuzzy text matching verifies every citation against source chunks |
| 📤 Live Upload | Drop a PDF in the dashboard — indexed and queryable in real time |
| 🌐 REST API | FastAPI layer with `/query`, `/query/stream` (SSE), `/health` endpoints |
| 📊 RAGAS Evaluation | Faithfulness, Answer Relevancy, Context Recall on 30 QA pairs |

---

## 🏗 Architecture

```
PDF Documents
      │
      ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐
│  ingest.py  │──▶│   embed.py   │──▶│    retriever.py      │
│  PyMuPDF    │   │ MiniLM-L6-v2 │   │  FAISS + BM25 + RRF  │
│  Tesseract  │   │  FAISS Index │   └──────────┬───────────┘
└─────────────┘   └──────────────┘              │
                                                 ▼
                                      ┌──────────────────────┐
                                      │    reranker.py       │
                                      │  CrossEncoder MiniLM │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                                      ┌──────────────────────┐
                                      │    generator.py      │
                                      │  Gemini Flash + JSON │
                                      │  memory + cache      │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                                      ┌──────────────────────┐
                                      │    validator.py      │
                                      │  Citation Grounding  │
                                      └──────────┬───────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                           ┌────────▼────────┐   ┌───────────▼──────┐
                           │  dashboard/     │   │  evaluation/     │
                           │  Streamlit UI   │   │  RAGAS Metrics   │
                           └─────────────────┘   └──────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
| --- | --- |
| PDF Extraction | PyMuPDF + Tesseract OCR |
| Text Chunking | LangChain RecursiveCharacterTextSplitter (800 chars, 100 overlap) |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 (384-dim) |
| Vector Store | FAISS IndexFlatL2 |
| Keyword Search | rank-bm25 (BM25 Okapi) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Google Gemini (gemini-flash-latest) via google-genai SDK |
| Memory | Sliding-window conversation memory with query rewriting |
| Cache | Semantic cache (exact + cosine similarity, 0.92 threshold) |
| REST API | FastAPI + uvicorn with SSE streaming |
| Dashboard | Streamlit + Plotly |
| Evaluation | RAGAS |

---

## 📁 Project Structure

```
isro-rag-assistant/
├── src/
│   ├── ingest.py          # PDF extraction + chunking
│   ├── embed.py           # Embeddings + FAISS index
│   ├── retriever.py       # Hybrid FAISS+BM25+RRF retrieval
│   ├── reranker.py        # Cross-encoder re-ranking
│   ├── generator.py       # Gemini RAG chain + memory + cache
│   ├── validator.py       # Citation verification + hallucination detection
│   ├── memory.py          # Conversation memory + query rewriting
│   ├── cache.py           # Semantic query cache
│   ├── api.py             # FastAPI REST layer
│   └── utils.py           # Shared utilities
├── dashboard/
│   └── app.py             # Streamlit chat UI
├── evaluation/
│   ├── qa_pairs.json      # 30 ISRO QA pairs with ground truth
│   └── evaluate.py        # RAGAS evaluation pipeline
├── embeddings/
│   ├── index.faiss        # (generated)
│   └── chunks_metadata.json  # (generated)
├── data/
│   ├── raw/               # Place your PDFs here
│   ├── processed/
│   │   └── chunks.json    # (generated)
│   ├── sessions/          # (generated) conversation sessions
│   └── cache/             # (generated) semantic cache
├── tests/
│   ├── test_ingest.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_validator.py
├── scripts/
│   ├── create_sample_pdfs.py
│   └── full_run.py
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-flash-latest
```

Get a free key at: <https://aistudio.google.com>

### Step 3 — Generate sample PDFs (or add your own)

```bash
python scripts/create_sample_pdfs.py
```

Or place your own PDFs inside `data/raw/`.

### Step 4 — Ingest documents

```bash
python src/ingest.py --dir data/raw
```

### Step 5 — Build embeddings

```bash
python src/embed.py
```

### Step 6 — Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open <http://localhost:8501> in your browser.

---

## 💬 Sample Questions

**Chandrayaan-3**

```
What is the propulsion system of Chandrayaan-3?
What did Chandrayaan-3 discover at the lunar south pole?
What instruments does the Pragyan rover carry?
What is the landing site of Chandrayaan-3?
```

**Mangalyaan**

```
What scientific instruments does Mangalyaan carry?
How long did Mangalyaan take to reach Mars?
What is the orbit of Mangalyaan around Mars?
```

**Aditya-L1**

```
What is the significance of the L1 Lagrange point for Aditya-L1?
What payloads does Aditya-L1 carry?
What does the VELC instrument do?
```

**Gaganyaan**

```
How many astronauts will Gaganyaan carry?
What launch vehicle does Gaganyaan use?
What is the mission duration of Gaganyaan?
```

---

## 🌐 REST API

Start the API server:

```bash
uvicorn src.api:app --port 8000
```

Example requests:

```bash
# Ask a question
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"What is Chandrayaan-3?\", \"session_id\": \"s1\"}"

# Health check
curl http://localhost:8000/health

# List indexed documents
curl http://localhost:8000/docs-list
```

API docs: <http://localhost:8000/docs>

---

## 📊 Evaluation

```bash
# Full RAGAS evaluation (30 questions)
python evaluation/evaluate.py

# Quick smoke test (5 questions)
python evaluation/evaluate.py --limit 5

# Skip RAGAS — just run pipeline and print answers
python evaluation/evaluate.py --skip-ragas
```

| Metric | Measures |
| --- | --- |
| Faithfulness | Are claims supported by retrieved context? |
| Answer Relevancy | Is the answer on-topic? |
| Context Recall | Does retrieved context cover the ground truth? |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

| Test File | Covers |
| --- | --- |
| test_ingest.py | PDF parsing, chunking, mission inference |
| test_retriever.py | FAISS + BM25, RRF score ordering |
| test_generator.py | JSON extraction, citation schema |
| test_validator.py | Hallucination detection accuracy |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Required. Gemini API key from aistudio.google.com |
| `LLM_PROVIDER` | `gemini` | `gemini` or `claude` |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model name |
| `ANTHROPIC_API_KEY` | — | Required only if LLM_PROVIDER=claude |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model name |
| `MAX_TOKENS` | `2048` | Max generation tokens |
| `TEMPERATURE` | `0.1` | Sampling temperature |
| `TOP_K_RETRIEVAL` | `6` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 💡 Design Decisions

**Why Reciprocal Rank Fusion?**
RRF is parameter-free and robust to scale differences between FAISS L2 distances and BM25 scores — no manual weight tuning required.

**Why all-MiniLM-L6-v2?**
Fast (384-dim), strong semantic similarity on technical text, runs on CPU without GPU.

**Why cross-encoder re-ranking?**
Bi-encoder retrieval (FAISS) is fast but imprecise. The cross-encoder jointly scores (query, chunk) pairs for much higher precision at the cost of ~50 ms extra latency.

**Why structured JSON output?**
Forces the LLM to be explicit about citations, making hallucination detection tractable via fuzzy matching.

**Why chunk_size=800 with overlap=100?**
800 chars (~150 words) fits within the model attention window while preserving enough context. 100-char overlap prevents splitting mid-sentence.

**Why semantic cache?**
Gemini free tier has rate limits. Caching identical/near-identical queries avoids redundant API calls and makes repeated questions instant.
