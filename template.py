import os
from pathlib import Path

project_name = "src"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/custom_logger.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/custom_exception.py",
    "config/config.yaml",
    "config/params.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/01_data_ingestion.ipynb",
    "artifacts/.gitkeep",
    "scripts/run.sh",
    "tests/__init__.py",
    "docs/architecture.md",
]

for filepath in list_of_files:
    # Use Path from pathlib to handle OS specific paths correctly
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir} for the file {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            if str(filepath) == "artifacts\\.gitkeep" or str(filepath) == "artifacts/.gitkeep":
                pass # Just keep it empty
            pass
            print(f"Creating empty file: {filepath}")
    else:
        print(f"{filename} already exists!")

print("\nProject Template Scaffolded Successfully!")
