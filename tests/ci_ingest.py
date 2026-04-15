"""
CI Data Ingestion Script
========================
Ingests PDF files from data/raw/ into ChromaDB.
Used exclusively by the GitHub Actions CI pipeline.

Usage:
    python tests/ci_ingest.py
"""

import sys
from pathlib import Path

# Make src.* importable from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

import os
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.vector_store import VectorStore
from src.logger.custom_logger import logger


def main():
    data_dir = ROOT / "data" / "raw"
    if not data_dir.exists() or not any(data_dir.glob("*.pdf")):
        print(f"❌  No PDFs found in {data_dir}")
        print("    Commit your source PDFs to data/raw/ for CI to work.")
        sys.exit(1)

    pdf_count = len(list(data_dir.glob("*.pdf")))
    print(f"📄  Found {pdf_count} PDF(s) in {data_dir}. Starting ingestion …")

    config_manager = ConfigurationManager()

    # 1. Parse + chunk PDFs
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    chunks = data_ingestion.initiate_data_ingestion()

    if not chunks:
        print("❌  No chunks produced. Aborting.")
        sys.exit(1)

    print(f"✅  Generated {len(chunks)} chunks.")

    # 2. Embed + store in ChromaDB
    vector_store_config = config_manager.get_vector_store_config()
    vector_store = VectorStore(config=vector_store_config)
    vector_store.initiate_vector_store(chunks)

    print("✅  Vector store built successfully.")


if __name__ == "__main__":
    main()
