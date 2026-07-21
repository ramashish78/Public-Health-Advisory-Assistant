from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
VECTOR_STORE_DIR = ARTIFACTS_DIR / "vector_store"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


@dataclass(slots=True)
class Settings:
    project_name: str = "Public Health Advisory Assistant"
    host: str = os.getenv("PHAA_HOST", "0.0.0.0")
    port: int = int(os.getenv("PHAA_PORT", "8000"))
    embedding_model: str = os.getenv("PHAA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    allow_model_download: bool = os.getenv("PHAA_ALLOW_MODEL_DOWNLOAD", "false").lower() == "true"
    retrieval_top_k: int = int(os.getenv("PHAA_TOP_K", "3"))
    report_retention_days: int = int(os.getenv("PHAA_REPORT_RETENTION_DAYS", "30"))
    model_artifact_path: Path = MODELS_DIR / "disease_classifier.joblib"
    model_metrics_path: Path = MODELS_DIR / "training_metrics.json"
    training_dataset_path: Path = PROCESSED_DATA_DIR / "symptom_disease_dataset.csv"
    seed_dataset_path: Path = RAW_DATA_DIR / "symptom_disease_seed.csv"
    medical_corpus_path: Path = RAW_DATA_DIR / "medical_corpus.jsonl"
    faiss_index_path: Path = VECTOR_STORE_DIR / "medical.index"
    retrieval_metadata_path: Path = VECTOR_STORE_DIR / "metadata.json"
    retrieval_vectorizer_path: Path = VECTOR_STORE_DIR / "vectorizer.joblib"
    retrieval_documents_path: Path = VECTOR_STORE_DIR / "documents.json"

    def ensure_directories(self) -> None:
        for path in (
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            MODELS_DIR,
            VECTOR_STORE_DIR,
            REPORTS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
