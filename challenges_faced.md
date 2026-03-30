# Ragas Evaluation & Pipeline Challenges

During the implementation of Phase 3 (Automated Evaluation with Ragas), we encountered several significant blockers typical of moving a RAG system from prototype to production. Here is a log of the challenges faced and the architectural shifts we made to resolve them.

---

### Challenge 1: API Rate Limits and Cost Exhaustion
**The Problem:**
Initially, the `TestsetGenerator` was configured to use the Groq API (`llama-3.3-70b-versatile`). Because Ragas makes intense, repeated LLM calls to process document chunks, we immediately hit the Groq free tier limit (`Error 429: Rate limit reached... Limit 100000 tokens per day`). 

**The Resolution:**
We shifted the entire evaluation heavy-lifting to **Local Open-Source Models**.
- Modified `params.yaml` and `config_entity.py` to support `ollama_eval_model`.
- Switched the evaluation component to use local `llama3` (8B) running via Ollama. 
- *Result:* Zero API costs, no network rate limits, and full data privacy.

---

### Challenge 2: Local Processing Bottleneck (Hours of compute)
**The Problem:**
Even after moving to local models, the built-in Ragas `TestsetGenerator` proved unviable for local hardware. To generate the testset, Ragas runs three heavy extraction passes (`SummaryExtractor`, `CustomNodeFilter`, `ThemesExtractor`) linearly across *every single chunk* in the database. 
For just 52 document chunks, doing 156 local LLM inferences took upwards of 2.5 hours on CPU. This feedback loop is far too slow for active development.

**The Resolution:**
We abandoned the bulky built-in generator and wrote a **Custom Sampling Generator**.
- Instead of processing all chunks, the pipeline dynamically samples a small subset (e.g., `SAMPLE_CHUNK_COUNT = 5`).
- It passes each sampled chunk through a single, highly-optimized prompt, instructing `llama3` to extract precisely one grounded factual question and answer.
- *Result:* Testset generation time collapsed from 3+ hours to ~5 minutes.

---

### Challenge 3: Silent Metric Failures (NaN Scores)
**The Problem:**
Once the testset was generated, we ran the evaluation stage. The pipeline executed successfully, retrieved contexts, and generated answers, but the output `results.csv` had `NaN` (blank) values for every single Ragas metric (`faithfulness`, `answer_relevancy`, etc.).
This occurred because Ragas' built-in LLM metrics require the judging LLM to output highly strict JSON schemas internally, which small 8B/4B local models frequently fail to conform to, causing silent failures inside the Ragas parsing logic.

**The Resolution:**
We replaced the brittle "LLM-as-a-judge" metrics with **Embedding-Based Mathematical Metrics**.
We leveraged our local `nomic-embed-text` to compute dense vector embeddings, allowing us to build three robust, transparent metrics:
1. **Answer Similarity**: Cosine similarity between the pipeline's answer embedding and the ground-truth answer embedding.
2. **Context Relevance**: Cosine similarity between the user's question embedding and the top retrieved chunk.
3. **Exact Match**: Binary check for hard factual extraction.
- *Result:* The evaluation pipeline now runs reliably on every execution. It is deterministic, requires zero structured JSON from the LLM, and successfully scores the system locally.
