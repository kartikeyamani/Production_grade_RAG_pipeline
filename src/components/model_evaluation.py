import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.entity.config_entity import EvaluationConfig, RAGEngineConfig
from src.components.rag_engine import RAGEngine
from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

# Number of chunks to sample for testset generation (keeps it fast on local models)
SAMPLE_CHUNK_COUNT = 5


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ModelEvaluation:
    def __init__(self, eval_config: EvaluationConfig, rag_config: RAGEngineConfig):
        self.config = eval_config
        self.rag_config = rag_config

    def _get_base_url(self):
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def _get_local_llm(self):
        return ChatOllama(
            model=self.config.ollama_eval_model,
            base_url=self._get_base_url(),
            temperature=self.config.llm_temperature
        )

    def _get_local_embeddings(self):
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self._get_base_url()
        )

    def _load_raw_chunks(self):
        """Loads raw document chunks from the pickle file."""
        try:
            chunks_path = self.config.raw_chunks_path
            if not os.path.exists(chunks_path):
                raise FileNotFoundError(
                    f"Raw chunks not found at {chunks_path}. Run --ingest first."
                )
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded {len(chunks)} raw chunks.")
            return chunks
        except Exception as e:
            raise CustomException(e, sys)

    def generate_testset(self):
        """
        Generates a synthetic Q&A testset by randomly sampling a small number
        of chunks and asking the local LLM to produce one question+answer per chunk.
        Saves result as a CSV to the configured testset path.
        """
        try:
            # Skip generation if testset already exists
            if os.path.exists(self.config.testset_path):
                logger.info(f"Testset already exists at {self.config.testset_path}. Skipping generation.")
                return pd.read_csv(self.config.testset_path)

            raw_data = self._load_raw_chunks()
            # Handle InMemoryStore dict maps or standard lists
            chunks = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
            
            sample_size = min(SAMPLE_CHUNK_COUNT, len(chunks))
            sampled_chunks = random.sample(chunks, sample_size)
            logger.info(f"Sampled {sample_size}/{len(chunks)} chunks for fast testset generation.")

            llm = self._get_local_llm()
            rows = []

            for i, chunk in enumerate(sampled_chunks):
                context = chunk.page_content.strip()
                prompt = (
                    f"Based on the following passage, write one clear factual question "
                    f"that can be answered directly from it, and provide the exact answer.\n\n"
                    f"Passage:\n{context}\n\n"
                    f"Respond in this exact format:\n"
                    f"Question: <your question here>\n"
                    f"Answer: <your answer here>"
                )
                logger.info(f"Generating Q&A pair {i+1}/{sample_size}...")
                response = llm.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)

                question, answer = "", ""
                for line in text.strip().splitlines():
                    if line.lower().startswith("question:"):
                        question = line.split(":", 1)[-1].strip()
                    elif line.lower().startswith("answer:"):
                        answer = line.split(":", 1)[-1].strip()

                if question and answer:
                    rows.append({
                        "user_input": question,
                        "reference": answer,
                        "reference_context": context
                    })
                else:
                    logger.warning(f"Could not parse Q&A from chunk {i+1}. Skipping.")

            if not rows:
                raise ValueError("No valid Q&A pairs generated. Check your Ollama model is running.")

            testset_df = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(self.config.testset_path), exist_ok=True)
            testset_df.to_csv(self.config.testset_path, index=False)
            logger.info(f"Testset with {len(testset_df)} samples saved to {self.config.testset_path}")
            return testset_df

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self):
        """
        Loads the saved testset, runs each question through the live RAGEngine,
        and scores results using embedding-based metrics — fully local, no structured
        JSON output required from the LLM, no rate limits.

        Metrics:
          - answer_similarity   : cosine similarity between pipeline answer and ground truth
          - context_relevance   : cosine similarity between question and best retrieved chunk
          - exact_match         : whether the ground truth string appears in the answer (0 or 1)
        """
        try:
            if not os.path.exists(self.config.testset_path):
                raise FileNotFoundError(
                    f"Testset not found at {self.config.testset_path}. Run generate_testset() first."
                )

            testset_df = pd.read_csv(self.config.testset_path)
            logger.info(f"Loaded testset with {len(testset_df)} samples. Scoring pipeline...")

            embeddings_model = self._get_local_embeddings()

            # Setup the live RAG pipeline
            rag_engine = RAGEngine(config=self.rag_config)
            rag_chain = rag_engine.setup_rag_pipeline()
            if not rag_chain:
                raise RuntimeError("Failed to initialize RAG pipeline for evaluation.")

            results = []

            for _, row in testset_df.iterrows():
                question = str(row["user_input"])
                ground_truth = str(row.get("reference", ""))

                logger.info(f"Evaluating: {question[:70]}...")

                # Run through live RAG pipeline
                response = rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "eval_session"}}
                )
                pipeline_answer = response.get("answer", "")
                retrieved_contexts = [doc.page_content for doc in response.get("context", [])]

                # --- Metric 1: Answer Semantic Similarity ---
                # How close is the pipeline's answer to the ground truth answer?
                answer_emb = embeddings_model.embed_query(pipeline_answer)
                truth_emb = embeddings_model.embed_query(ground_truth)
                answer_similarity = cosine_similarity(answer_emb, truth_emb)

                # --- Metric 2: Context Relevance ---
                # Is the best retrieved chunk relevant to the question?
                question_emb = embeddings_model.embed_query(question)
                context_scores = [
                    cosine_similarity(question_emb, embeddings_model.embed_query(ctx))
                    for ctx in retrieved_contexts
                ]
                context_relevance = max(context_scores) if context_scores else 0.0

                # --- Metric 3: Exact Match ---
                # Does the ground truth appear literally in the pipeline's answer?
                exact_match = 1.0 if ground_truth.lower() in pipeline_answer.lower() else 0.0

                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "pipeline_answer": pipeline_answer,
                    "retrieved_contexts": " | ".join(retrieved_contexts),
                    "answer_similarity": round(answer_similarity, 4),
                    "context_relevance": round(context_relevance, 4),
                    "exact_match": exact_match
                })

            results_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(self.config.results_path), exist_ok=True)
            results_df.to_csv(self.config.results_path, index=False)
            logger.info(f"Results saved to {self.config.results_path}")

            # Print clean summary table
            print("\n" + "="*60)
            print("Evaluation Results (Embedding-Based, Fully Local)")
            print("="*60)
            print(f"  {'answer_similarity':<25}: {results_df['answer_similarity'].mean():.4f}")
            print(f"  {'context_relevance':<25}: {results_df['context_relevance'].mean():.4f}")
            print(f"  {'exact_match':<25}: {results_df['exact_match'].mean():.4f}  "
                  f"({int(results_df['exact_match'].sum())}/{len(results_df)} questions)")
            print("="*60)
            print("\nPer-question breakdown:")
            for _, r in results_df.iterrows():
                print(f"\n  Q: {r['question'][:70]}")
                print(f"  Ground Truth : {r['ground_truth']}")
                print(f"  Pipeline Ans : {r['pipeline_answer'][:100]}")
                print(f"  Similarity={r['answer_similarity']} | Relevance={r['context_relevance']} | ExactMatch={int(r['exact_match'])}")
            print("="*60 + "\n")

            return results_df

        except Exception as e:
            raise CustomException(e, sys)
