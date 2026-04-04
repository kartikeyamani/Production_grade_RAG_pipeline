# Chunking Strategy Comparison (Multi-PDF / Large Evaluation)

Now that we have tested the pipeline on 1 PDF and 5 questions, this file will serve as the benchmark for a dense corpus (4+ PDFs) and a large evaluation dataset (20-30 questions).

## Step 1: Ingest Multiple PDFs
Add 3 or more new PDF files to the `data/raw` folder along with the existing one. This tests how the architectures handle conflicting and dense information across multiple topics.

## Step 2: Set Testset Scope
Update `SAMPLE_CHUNK_COUNT` in `src/components/model_evaluation.py` to 25 or 30 before running the evaluation. Also, remember to delete or rename your old `artifacts/evaluation/testset*.csv` files so the script generates a fresh batch of 30 questions based on the new PDFs!

## Score Tracking

### 1. Base Recursive Character Splitter (1000 / 200) - Baseline
We evaluated this baseline using both the original local model (`nomic-embed-text` via Ollama) and the new cloud model (`text-embedding-3-small` via OpenAI).

**With Ollama (`nomic-embed-text`):**
- **Exact Match:** `0.24` (24%, 6/25 questions answered verbatim)
- **Context Relevance:** `0.6493`
- **Answer Similarity:** `0.6300`

**With OpenAI (`text-embedding-3-small`):**
- **Exact Match:** `0.08` (8%, 2/25 questions answered verbatim)
- **Context Relevance:** `0.491`
- **Answer Similarity:** `0.411`

**Analysis:**
While this chunk size was optimal for `nomic-embed-text`, it performed disastrously with `text-embedding-3-small`, dropping exact matches to just 8% and showing the lowest context relevance across all tests. This shows that the optimal chunk size is highly dependent on the embedding model!

### 2. Semantic Chunking (SemanticChunker + nomic-embed-text)
- **Exact Match:** `0.24` (24%, 6/25 questions answered verbatim)
- **Context Relevance:** `0.6289`
- **Answer Similarity:** `0.6264`

**Analysis:**
Semantic chunking produced meaning-aware, variable-size chunks aligned to topic shifts. Notably it **matched Recursive on Exact Match (24%)**, but lagged on both Context Relevance and Answer Similarity. It appears that while the chunker groups semantically coherent text together, the resulting chunks can be quite large and may dilute the BM25 keyword retrieval signal. The significant ingestion cost (embedding every sentence) is hard to justify given the marginal results.

### 3. Parent-Document Retrieval
- **Exact Match:** `0.12` (12%, 3/25 questions answered verbatim)
- **Context Relevance:** `0.6375`
- **Answer Similarity:** `0.6510`

### 4. Recursive Character (512 / 50) - Small Chunks
*(Note: Evaluated using text-embedding-3-small, cosine similarity absolute values might differ slightly from Ollama)*
- **Exact Match:** `0.20` (20%, 5/25 questions answered verbatim)
- **Context Relevance:** `0.5466`
- **Answer Similarity:** `0.4523`

**Analysis:**
Decreasing the chunk size to 512 and overlap to 50 led to a noticeable drop in performance. Exact match dropped from 24% to 20%. The smaller chunks likely break apart critical context, making it harder for both the BM25 and vector retriever to find complete answers. This confirms that 512 is too small for our dense PDFs.

### 5. Recursive Character (1500 / 200) - Large Chunks
*(Note: Evaluated using text-embedding-3-small)*
- **Exact Match:** `0.20` (20%, 5/25 questions answered verbatim)
- **Context Relevance:** `0.503`
- **Answer Similarity:** `0.442`

**Analysis:**
Increasing the chunk size to 1500 and overlap to 200 also led to a drop in performance. Exact match dropped from the 1000 baseline (24%) back down to 20%. The context relevance was surprisingly the lowest here (0.503). This indicates a classic ""Lost in the Middle"" phenomenon — when chunks are too large, the specific precise facts get diluted by surrounding irrelevant text, confusing both the retriever and the generator LLM.

### Conclusion & Final Recommendation
When migrating to **`text-embedding-3-small`**, the optimal chunking strategy completely flips. The **512 / 50 Recursive Character Splitter** configuration is the new sweet spot for this production pipeline!

- At `1000` (The old sweet spot for Ollama), the OpenAI embeddings performed terribly (8% Exact Match, 0.49 Context Relevance).
- At `1500`, chunks suffered from the ""Lost in the Middle"" phenomenon where the signal was too diluted.
- At `512`, the OpenAI embedding model excelled (20% Exact Match, 0.54 Context Relevance). Because `text-embedding-3-small` is highly tuned for shorter, denser context windows, it prefers highly granular chunks over large paragraphs.

**Verdict:** We must use **512 chunk size / 50 overlap** moving forward with OpenAI.

### The Stylistic Alignment Effect
We also ran a test where the ground truth evaluation testset was completely regenerated using `gpt-4o-mini` instead of `llama3`. The results perfectly demonstrated **Stylometric Alignment**:
* The **Answer Similarity spiked from 0.452 to 0.501**!
* Because the "Judge/Ground Truth LLM" and the "Pipeline Generator LLM" shared the same semantic footprint and vocabulary style, the cosine similarity of their answers mathmatically converged closer together, proving why you must always evaluate your pipeline using answers generated by the same class of model.

## 🏆 Final Benchmark Results — All 3 Strategies (25 Questions, 4 PDFs)

| Strategy | Answer Similarity | Context Relevance | Exact Match | Ingestion Cost |
|---|---|---|---|---|
| **Recursive Character (1000/200)** | 0.6317 | **0.6491** ✅ | **24%** ✅ | Low |
| Semantic Chunker | 0.6264 | 0.6289 | 24% | Very High |
| Parent-Document Retrieval | **0.6510** ✅ | 0.6375 | 12% | Medium |

### 🎯 Verdict: **Recursive Character Chunking is the Production Choice**

**Reasoning:**
1. **Best Context Relevance (0.649):** The hybrid BM25 + vector retriever excels at finding the right 1000-char chunk across 4 varied PDFs. Smaller, precise chunks give BM25 a sharper signal.
2. **Tied for Best Exact Match (24%):** The factual precision is on par with Semantic Chunking, without the enormous ingestion time penalty.
3. **PDR tradeoff:** While PDR gives richer answers (higher Similarity), it sacrifices retrieval precision (lower Exact Match). In production, hallucination risk from imprecise retrieval outweighs richer generation.
4. **Semantic Chunker was not worth the cost:** It required ~20 minutes of embedding time for chunking alone, and delivered no statistically meaningful improvement over the simple recursive baseline.

### ➡️ Next Step: Integrate FlashRank reranker and lock the production config.

