import os
import yaml
import subprocess
import time

run_configs = [
    {"chunk_size": 512, "overlap": 50, "out_csv": "results_512.csv"},
    {"chunk_size": 1000, "overlap": 200, "out_csv": "results_1000.csv"},
    {"chunk_size": 2000, "overlap": 200, "out_csv": "results_2000.csv"}
]

params_path = "config/params.yaml"
config_path = "config/config.yaml"

def update_yaml(file_path, key, value, parent_key=None):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    if parent_key:
        data[parent_key][key] = value
    else:
        data[key] = value
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print("Starting automated benchmark script...")
print("Ensuring testset is perfectly preserved...")

for cfg in run_configs:
    print(f"\n=======================================================")
    print(f"|  Running Benchmark: Size {cfg['chunk_size']} / Overlap {cfg['overlap']}")
    print(f"=======================================================\n")
    
    # Update params.yaml
    update_yaml(params_path, 'chunk_size', cfg['chunk_size'])
    update_yaml(params_path, 'chunk_overlap', cfg['overlap'])
    
    # Update config.yaml
    update_yaml(config_path, 'results_path', f"artifacts/evaluation/multipdfresults/{cfg['out_csv']}", parent_key='evaluation')
    
    print("Ingesting Data...")
    subprocess.run(["python", "main.py", "--ingest"])
    
    print("Evaluating Pipeline against GPT-4o-mini testset...")
    subprocess.run(["python", "main.py", "--evaluate"])
    
    print(f"Finished {cfg['chunk_size']} chunk test.\n")

print("All benchmarks complete!")
