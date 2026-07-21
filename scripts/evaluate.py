from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.ml.train import train_classifier
from app.rag.indexer import build_vector_store
from app.services.model_service import ModelService
from app.services.retrieval_service import RetrievalService


def evaluate_retrieval() -> dict:
    service = RetrievalService()
    seed_df = pd.read_csv(settings.seed_dataset_path)
    hit_count = 0
    total = 0
    for _, row in seed_df.iterrows():
        query = f"{row['disease']} symptoms {row['symptoms'].replace('|', ', ')}"
        results = service.retrieve(query, top_k=3)
        total += 1
        if any(result.get("disease") == row["disease"] for result in results):
            hit_count += 1
    return {"retrieval_hit_rate_at_3": round(hit_count / total, 4)}


if __name__ == "__main__":
    build_vector_store()
    train_classifier()
    model_service = ModelService()
    training_metrics = json.loads(settings.model_metrics_path.read_text(encoding="utf-8"))

    sample = model_service.predict_from_text(
        "I have fever, cough, fatigue and chills for 3 days.",
        raw_text="i have fever, cough, fatigue and chills for 3 days.",
        symptoms=["fever", "cough", "fatigue", "chills"],
    )
    retrieval_metrics = evaluate_retrieval()
    print(
        {
            "training_metrics": training_metrics,
            "retrieval_metrics": retrieval_metrics,
            "sample_prediction": sample,
        }
    )
