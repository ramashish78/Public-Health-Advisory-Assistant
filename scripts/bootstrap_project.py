from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ml.train import train_classifier
from app.rag.indexer import build_vector_store
from scripts.generate_dataset import generate_dataset


if __name__ == "__main__":
    dataset = generate_dataset()
    metrics = train_classifier()
    retrieval = build_vector_store()
    print(
        {
            "generated_rows": len(dataset),
            "training_metrics": metrics,
            "retrieval": retrieval,
        }
    )
