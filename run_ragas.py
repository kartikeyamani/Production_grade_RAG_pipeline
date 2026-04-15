import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langfuse.langchain import CallbackHandler

# Load environment variables (mostly OPENAI_API_KEY)
load_dotenv()

results_path = "artifacts/evaluation/multipdfresults/results_1000.csv"

# Check if the generated pipeline results exist
if not os.path.exists(results_path):
    print(f"Error: Could not find {results_path}. Make sure benchmark ran completely.")
    exit(1)

print(f"Loading existing pipeline outputs from {results_path}...")
df = pd.read_csv(results_path)

# RAGAS requires exactly these column names
data_samples = {
    "question": df["question"].tolist(),
    "answer": df["pipeline_answer"].tolist(),
    "contexts": [[c.strip() for c in str(str(ctx)).split(" | ")] for ctx in df["retrieved_contexts"]],
    "ground_truth": [str(gt) for gt in df["ground_truth"].tolist()]
}

dataset = Dataset.from_dict(data_samples)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

print("Starting RAGAS Evaluation...")
print("Metrics: faithfulness, answer_relevancy, context_precision")

# Explicitly instantiate LangChain models so Ragas doesn't crash on broken defaults
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    # Attempt to wrap for newer RAGAS versions if needed, or pass directly
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings_model)
except ImportError:
    # Older RAGAS versions accept LangChain objects directly
    ragas_llm = llm
    ragas_emb = embeddings_model

print("Initializing Langfuse Telemetry...")
langfuse_handler = CallbackHandler()
langfuse_handler.auth_check()

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ],
    llm=ragas_llm,
    embeddings=ragas_emb,
    callbacks=[langfuse_handler]
)

ragas_df = result.to_pandas()
out_file = "artifacts/evaluation/multipdfresults/ragas_results_1000.csv"
ragas_df.to_csv(out_file, index=False)

print(f"\n✅ RAGAS evaluation complete! Saved to {out_file}")
print("\n--- RAGAS Final Metric Averages (1000/200) ---")
print(f"Faithfulness       : {ragas_df['faithfulness'].mean():.4f}")
print(f"Answer Relevancy   : {ragas_df['answer_relevancy'].mean():.4f}")
print(f"Context Precision  : {ragas_df['context_precision'].mean():.4f}")
