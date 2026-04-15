"""
CI Eval Gate Script
===================
Runs RAGAS evaluation on the frozen 25-question testset and compares results
against the committed baseline in tests/baseline_metrics.json.

Exit codes:
  0 → All metrics pass (>= baseline). PR can merge.
  1 → One or more metrics failed. PR is blocked.
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FROZEN_TESTSET = ROOT / "tests" / "frozen_testset.csv"
BASELINE_FILE  = ROOT / "tests" / "baseline_metrics.json"
CI_RESULTS_OUT = ROOT / "tests" / "ci_eval_results.csv"

# ── Bootstrap env ──────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=ROOT / ".env")


def load_baseline() -> dict:
    with open(BASELINE_FILE) as f:
        return json.load(f)


def build_rag_chain():
    """Instantiate the full production RAG pipeline. """
    # Add project root so src.* imports resolve
    sys.path.insert(0, str(ROOT))
    from src.config.configuration import ConfigurationManager
    from src.components.rag_engine import RAGEngine

    config_manager = ConfigurationManager()
    rag_config = config_manager.get_rag_engine_config()
    engine = RAGEngine(config=rag_config)
    chain = engine.setup_rag_pipeline()
    if chain is None:
        print("❌  RAG pipeline could not be initialised. Was ingestion run?")
        sys.exit(1)
    return chain


def run_pipeline_on_testset(chain, testset_df: pd.DataFrame) -> dict:
    """
    Run every question through the live RAG chain and collect:
      question, answer, contexts (list), ground_truth
    Returns a dict suitable for datasets.Dataset.from_dict()
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for _, row in testset_df.iterrows():
        question    = str(row["user_input"])
        ground_truth = str(row["reference"])

        print(f"  → Querying: {question[:80]}…")
        response = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "ci_eval_session"}}
        )
        pipeline_answer = response.get("answer", "")
        retrieved_docs  = response.get("context", [])
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        questions.append(question)
        answers.append(pipeline_answer)
        contexts_list.append(retrieved_texts if retrieved_texts else [""])
        ground_truths.append(ground_truth)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    }


def run_ragas(data: dict) -> pd.DataFrame:
    """Run RAGAS faithfulness, answer_relevancy, context_precision."""
    dataset = Dataset.from_dict(data)

    llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        ragas_llm = llm
        ragas_emb = embeddings

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    return result.to_pandas()


def compare_and_gate(new_metrics: dict, baseline: dict) -> bool:
    """
    Returns True if ALL new metrics are >= baseline.
    Prints a rich comparison table.
    """
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision"]
    passed_all  = True

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'Baseline':>10} {'New':>10}  {'Status':>8}")
    print("=" * 65)

    for key in metric_keys:
        base_val = baseline.get(key, 0.0)
        new_val  = new_metrics.get(key, 0.0)
        delta    = new_val - base_val
        status   = "✅ PASS" if new_val >= base_val else "❌ FAIL"
        if new_val < base_val:
            passed_all = False
        print(f"  {key:<25} {base_val:>10.4f} {new_val:>10.4f}  {status}  (Δ {delta:+.4f})")

    print("=" * 65)
    return passed_all


def main():
    parser = argparse.ArgumentParser(description="CI RAGAS Eval Gate")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="After a successful run, update baseline_metrics.json with new values."
    )
    args = parser.parse_args()

    # ── 1. Load frozen testset ─────────────────────────────────────────────────
    print(f"\n📋  Loading frozen testset from {FROZEN_TESTSET} …")
    if not FROZEN_TESTSET.exists():
        print(f"❌  Frozen testset not found at {FROZEN_TESTSET}")
        sys.exit(1)
    testset_df = pd.read_csv(FROZEN_TESTSET)
    print(f"    Loaded {len(testset_df)} questions.")

    # ── 2. Load baseline ───────────────────────────────────────────────────────
    print(f"📊  Loading baseline from {BASELINE_FILE} …")
    baseline = load_baseline()
    print(f"    Baseline → faithfulness={baseline['faithfulness']}, "
          f"answer_relevancy={baseline['answer_relevancy']}, "
          f"context_precision={baseline['context_precision']}")

    # ── 3. Boot RAG pipeline & run queries ────────────────────────────────────
    print("\n🚀  Initialising RAG pipeline …")
    chain = build_rag_chain()

    print(f"\n🔍  Running {len(testset_df)} questions through RAG pipeline …")
    ragas_data = run_pipeline_on_testset(chain, testset_df)

    # ── 4. Run RAGAS evaluation ────────────────────────────────────────────────
    print("\n🧪  Running RAGAS evaluation (faithfulness / answer_relevancy / context_precision) …")
    results_df = run_ragas(ragas_data)
    results_df.to_csv(CI_RESULTS_OUT, index=False)
    print(f"    Results saved → {CI_RESULTS_OUT}")

    # ── 5. Gate decision ───────────────────────────────────────────────────────
    new_metrics = {
        "faithfulness":      results_df["faithfulness"].mean(),
        "answer_relevancy":  results_df["answer_relevancy"].mean(),
        "context_precision": results_df["context_precision"].mean(),
    }

    passed = compare_and_gate(new_metrics, baseline)

    # ── 6. Optionally update baseline ─────────────────────────────────────────
    if passed and args.update_baseline:
        from datetime import date
        updated = {
            **baseline,
            "faithfulness":      round(new_metrics["faithfulness"], 4),
            "answer_relevancy":  round(new_metrics["answer_relevancy"], 4),
            "context_precision": round(new_metrics["context_precision"], 4),
            "recorded_at":       str(date.today()),
        }
        with open(BASELINE_FILE, "w") as f:
            json.dump(updated, f, indent=2)
        print(f"\n✅  Baseline updated in {BASELINE_FILE}")

    if passed:
        print("\n✅  CI EVAL GATE: ALL METRICS PASSED — PR is safe to merge.\n")
        sys.exit(0)
    else:
        print("\n❌  CI EVAL GATE: METRICS BELOW BASELINE — PR is blocked.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
