# 🚀 Production-Grade Advanced RAG Pipeline

> *"If you can't change your chunking strategy without touching your LLM call, your architecture has failed."*

A **production-grade, enterprise-ready Retrieval-Augmented Generation (RAG) system** built with LangChain, ChromaDB, OpenAI, FlashRank, RAGAS, Langfuse, and Streamlit. This project is engineered following the same three-axis design philosophy used by senior ML engineers at scale: **data flow**, **control flow**, and **feedback flow**.

---

## 🧠 Problem Statement

Organizations possess vast amounts of unstructured data (research papers, PDF reports, contracts) but lack an efficient way to accurately query and verify that information at scale.

Naive RAG prototypes suffer from three critical failures:
1. **Hallucinations** — LLMs generate confident but fabricated answers when context is weak.
2. **Poor Retrieval** — Vector-only search misses exact keywords; BM25-only search misses semantic meaning. Neither alone is production-ready.
3. **Monolithic Architecture & Silent Failures** — Single-script prototypes are impossible to debug, scale, evaluate, trace, or hand off to a team.

**This project solves all three.**

---

## 🏗️ Architectural Reality of a Production RAG System

The pipeline is organized across three orthogonal axes:

| Axis | Responsibility |
|---|---|
| **Ingestion Pipeline** | PDF parsing → Chunking strategy (Mathematically Optimized to 2000 Chunks) → Dual-index storage (Vector DB + BM25) |
| **Retrieval & Ranking** | Query processing → Hybrid Retrieval (RRF) → Score Fusion → Cross-encoder Reranking (FlashRank) |
| **Generation + Eval** | Prompt assembly → LLM generation (GPT-4o-mini) → Source citation → Automated RAGAS evaluation |

Plus two horizontal layers that run across everything:
- **Observability Layer (Langfuse)** — End-to-end telemetry tracing every chunk retrieval time, reranker score, and LLM prompt. 
- **Config & Infrastructure** — strict YAML hyperparameters and decoupled modules.

---

## ✅ Features Implemented

### Phase 1: Modular Ingestion & Engine Baseline
- **PDF Ingestion Engine**: Automatically parses and recursively chunks PDFs from the `data/` folder.
- **Persistent Vector Store**: ChromaDB for local, persistent storage of document vectors (`text-embedding-3-small`).
- **Conversational Memory**: Full chat history leveraging LangChain's `RunnableWithMessageHistory`.
- **Decoupled Architecture**: Strictly organized OOP separating state, configuration, and components.

### Phase 2: Advanced Hybrid Retrieval & Reranking
- **Sparse Retrieval (BM25)**: `rank_bm25` for exact-match keyword search — critical for serial numbers, acronyms, and technical jargon.
- **Dense Retrieval (Vector Search)**: ChromaDB for deep semantic search.
- **Hybrid Fusion**: Merges both result sets using **Reciprocal Rank Fusion (RRF)** via LangChain's `EnsembleRetriever`.
- **Cross-Encoder Reranking**: `FlashrankRerank` scores the top fused candidates against the actual user query and prunes to the absolute best chunks. *Retrieval is cheap and broad; reranking is expensive and precise.*

### Phase 3: Mathematical Evaluation (RAGAS)
- Benchmarked the pipeline using synthetic Golden QA generative datasets.
- Tested `512`, `1000`, and `2000` chunk sizes against **Faithfulness**, **Answer Relevancy**, and **Context Precision**.
- **Result:** We discovered a performance drop in the "uncanny valley" of 1000 chunk sizes. The architecture config is permanently locked to a `2000` chunk-size architecture, proven to optimize deep conceptual contextualization without compromising exact match parameters.

### Phase 4: Langfuse Observability & Premium UI
- **Glassmorphic Streamlit App**: A stunning, production-ready interface that supports real-time dual-index data ingestion triggering and dynamic path cleaning for source references.
- **Langfuse Telemetry**: `@observe` decorators and callbacks are deeply integrated, offering granular millisecond profiling on retrieval time vs generation time, preventing silent regressions.

---

## 💻 Getting Started

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key
- Langfuse API Keys (Public, Secret, Host)

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY="sk-proj-..."

# Observability
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com" 
```

### 3. Installation
```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

### 4. Usage

**Option A: Launch the Premium Web UI**
```bash
streamlit run app.py
```
*Note: You can run the entire pipeline, ingest knowledge, and track Langfuse interactions directly from the Streamlit UI.*

**Option B: Execute Evaluation Scripts / Benchmarks**
```bash
python run_benchmark.py
python run_ragas.py
```

---

## 🔬 Production Design Decisions

Based on industry patterns from senior ML engineers, the following decisions guide this architecture:

- **Chunk size/overlap are hyperparameters**, not constants. They live in `params.yaml` and vary by document type.
- **Retrieval is broad, reranking is precise.** We fetch top-10 candidates and cut to the best top-3 before sending to the LLM. Sending all results is expensive and harmful.
- **Every module fails loudly.** Custom exceptions capture the exact file and line number so debugging is never guesswork.
- **Configuration is decoupled from code.** Swapping ChromaDB for Pinecone or Groq for OpenAI requires editing a YAML — not the source code.
- **Observability is not optional.** When retrieval fails, we look at the Langfuse waterfall graph to diagnose exactly which retrieval chain layer broke.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| LLM / Embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Vector Store | ChromaDB |
| Sparse Retrieval | rank-bm25 |
| Reranker | FlashRank |
| Orchestration | LangChain LCEL |
| Evaluation Framework | Python Synthetic Sets + RAGAS |
| Observability | Langfuse |
| Web UI | Streamlit |
| Config Management | PyYAML + python-box |


## adding this new comment for testing evaluaion gate 
