# Chunking Strategy Comparison (Multi-PDF / Large Evaluation)

Now that we have tested the pipeline on 1 PDF and 5 questions, this file will serve as the benchmark for a dense corpus (4+ PDFs) and a large evaluation dataset (20-30 questions).

## Step 1: Ingest Multiple PDFs
Add 3 or more new PDF files to the `data/raw` folder along with the existing one. This tests how the architectures handle conflicting and dense information across multiple topics.

## Step 2: Set Testset Scope
Update `SAMPLE_CHUNK_COUNT` in `src/components/model_evaluation.py` to 25 or 30 before running the evaluation. Also, remember to delete or rename your old `artifacts/evaluation/testset*.csv` files so the script generates a fresh batch of 30 questions based on the new PDFs!

## Score Tracking

### 1. Recursive Character Chunking (Baseline)
- **Exact Match:** Pending
- **Context Relevance:** Pending
- **Answer Similarity:** Pending

### 2. Semantic Chunking
- **Exact Match:** Pending
- **Context Relevance:** Pending
- **Answer Similarity:** Pending

### 3. Parent-Document Retrieval
- **Exact Match:** Pending
- **Context Relevance:** Pending
- **Answer Similarity:** Pending

**Final Large-Scale Analysis:**
(We will populate the analysis here once all metrics have been generated over the larger dataset).
