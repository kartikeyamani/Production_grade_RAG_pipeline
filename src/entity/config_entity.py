from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_dir: Path
    chunk_size: int
    chunk_overlap: int

@dataclass(frozen=True)
class VectorStoreConfig:
    root_dir: Path
    db_dir: Path
    raw_chunks_path: Path
    ollama_embedding_model: str

@dataclass(frozen=True)
class RAGEngineConfig:
    root_dir: Path
    db_dir: Path
    raw_chunks_path: Path
    top_k_vector: int
    top_k_bm25: int
    ensemble_weights: list
    flashrank_top_n: int
    groq_model: str
    llm_temperature: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    testset_path: Path
    results_path: Path
    testset_size: int
    groq_model: str
    llm_temperature: float
    raw_chunks_path: Path
    ollama_eval_model: str
