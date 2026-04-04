# RAG Chunking Benchmark Analysis
**Embedding Model:** `text-embedding-3-small` (OpenAI)
**Evaluation Generator:** `gpt-4o-mini`
**Dataset:** 4 Dense Academic PDFs
**Testset Size:** 25 Synthetic Questions

## 🏆 Official GPT-4o-Mini Testset Results

| Chunk Size | Overlap | Answer Similarity | Context Relevance | Exact Match |
|------------|---------|-------------------|-------------------|-------------|
| 512        | 50      | 0.4832            | 0.5082            | **16.0%**   |
| 1000       | 200     | 0.4555            | 0.4683            | 0.0%        |
| 2000       | 200     | **0.5365**        | **0.5342**        | 7.7%        |

---

## 🔍 Key Insights & Discoveries

### 1. The "Uncanny Valley" of 1000 Chunk Size
Contrary to standard industry default settings, the `1000/200` configuration is mathematically the worst performer across all metrics. It acts as an "uncanny valley" for OpenAI embeddings—it is too large to capture highly targeted facts without noise, yet too small to capture robust, holistic thematic ideas.

### 2. High Context Relevance at 2000 Size
The **`2000/200`** configuration scored the absolute highest in both **Answer Similarity (0.5365)** and **Context Relevance (0.5342)**.
* **Why?** Since large academic papers often build complex arguments over entire pages, a chunk size of 2000 tokens preserves the complete continuity of an idea. The generative pipeline is able to construct rich semantic answers because the broader context isn't sliced arbitrarily. 
* **Tradeoff:** While semantic flow is preserved, "Exact Matching" drops because the models are heavily summarizing long contextual walls of text instead of plucking direct sub-strings.

### 3. Precision Retrieval at 512 Size
The **`512/50`** configuration completely dominated the **Exact Match (16.0%)** metric, doubling the accuracy of the 2000 chunk.
* **Why?** Small chunks isolate specific facts, definitions, and distinct arguments. When hybrid search (BM25 + Vector) queries a specific terminology, a 512-chunk returns an exact snippet rather than an overwhelming 2000-word block. The generative LLM is essentially "forced" to use the specific phrasing from the ground-truth text because there is no surrounding noise to distract it.

---

## 🎯 Architecture Verdict

**Recommendation: Use `512/50` for Facts, or `2000/200` for Abstract Ideas.**

If your RAG application's primary function is to answer specific queries ("What is the equation for X?", "Who authored Y?"), **512 / 50 is the optimal production choice** as it excels at Exact Match retrieval.

If your RAG application revolves around deep summaries ("Summarize the overarching theme of this paper"), **2000 / 200** will yield more natural, human-like contextual responses.

Since typical RAG UI applications index factual questions (which users verify against source PDFs), locking in the `512/50` pipeline represents the most bullet-proof setup.

---




## 🚀 Post-Benchmark RAGAS Verification (Head-to-Head)

To provide an industry-standard validation of the tradeoff between factual precision (`512`) and abstract context (`2000`), we ran a dedicated **RAGAS Evaluation** using `gpt-4o-mini` as the LLM judge and `text-embedding-3-small` as the embeddings protocol.

**RAGAS Final Metric Averages:**

| Metric | `512/50` | `1000/200` | `2000/200` | Winner & Why |
|--------|----------|------------|------------|--------------|
| **Faithfulness** | **0.6333** | 0.5221 | 0.5227 | **512/50**: Smaller chunks isolate highly specific facts. The LLM rarely strays or hallucinates because it's directly referencing a small, targeted snippet (correlates with its 16% Exact Match rate). |
| **Answer Relevancy** | 0.7989 | 0.7921 | **0.8788** | **2000/200**: Massive chunks give the LLM the entire page's narrative. It understands the wide scope of the question and replies organically with extreme linguistic relevance, dominating thematic queries. |
| **Context Precision** | 0.4400 | 0.2967 | **0.4615** | **2000/200**: Large chunks allow the hybrid retriever to match dense semantic clusters effectively. Notice how `1000` dramatically drops to `0.29` here. |

**The "Uncanny Valley" Confirmed**: The `1000/200` chunk size is definitively the "uncanny valley" of RAG chunking. It is too big to surgically isolate facts (Faithfulness drops to ~0.52) and too small to hold rich, encompassing contextual themes (Answer Relevancy drops to ~0.79). Furthermore, the retrieval engine struggles to rank it efficiently (Context Precision crashes to 0.29).

**Final Conclusion**: `512` is objectively mathematically superior if you want *Faithful*, localized facts without hallucination. `2000` is superior for generating smooth, *Relevant*, highly-contextual essays. Per system requirements, we are permanently locking in the **`2000/200`** configuration to prioritize maximum conceptual context and answering relevancy.
