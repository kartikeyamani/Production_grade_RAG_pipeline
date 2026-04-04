import pandas as pd
import glob

files = [
    'artifacts/evaluation/multipdfresults/results_512.csv',
    'artifacts/evaluation/multipdfresults/results_1000.csv',
    'artifacts/evaluation/multipdfresults/results_2000.csv'
]

print("\n--- NEW GPT-4O-MINI TESTSET RESULTS ---\n")
print("| Chunk Size / Overlap | Answer Similarity | Context Relevance | Exact Match |")
print("|----------------------|-------------------|-------------------|-------------|")

for f in files:
    try:
        df = pd.read_csv(f)
        size = f.split('_')[-1].split('.')[0]
        overlap = "50" if size == "512" else "200"
        
        sim = df['answer_similarity'].mean()
        rel = df['context_relevance'].mean()
        em = df['exact_match'].mean()
        
        print(f"| {size} / {overlap} | {sim:.4f} | {rel:.4f} | {em*100:.1f}% |")
    except Exception as e:
        print(f"Error reading {f}: {e}")
print("\n")
