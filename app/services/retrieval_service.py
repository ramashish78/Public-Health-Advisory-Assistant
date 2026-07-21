from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.rag.embeddings import EmbeddingEngine

try:  # pragma: no cover - depends on environment
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class RetrievalService:
    def __init__(
        self,
        metadata_path: Path | None = None,
        documents_path: Path | None = None,
    ) -> None:
        metadata_path = metadata_path or settings.retrieval_metadata_path
        documents_path = documents_path or settings.retrieval_documents_path
        if not metadata_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                "Retrieval artifacts are missing. Run scripts/bootstrap_project.py first."
            )

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.documents = json.loads(documents_path.read_text(encoding="utf-8"))
        self.engine = EmbeddingEngine(self.metadata["embedding_model"])
        self.engine.load_from_metadata(self.metadata, settings.retrieval_vectorizer_path)
        self.backend = self.metadata["index_backend"]

        if self.backend == "faiss":
            self.index = faiss.read_index(str(settings.faiss_index_path))
            self.embeddings = None
        else:  # pragma: no cover - depends on faiss availability
            self.index = None
            self.embeddings = np.load(settings.faiss_index_path.with_suffix(".npy"))

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        vector = self.engine.transform([query]).astype("float32")
        if self.backend == "faiss":
            scores, indices = self.index.search(vector, top_k)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:  # pragma: no cover
            similarity = self.embeddings @ vector[0]
            top_indices = np.argsort(similarity)[::-1][:top_k]
            pairs = [(int(index), float(similarity[index])) for index in top_indices]

        results = []
        for index, score in pairs:
            if index < 0:
                continue
            document = self.documents[index]
            results.append(
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "disease": document["disease"],
                    "tags": document.get("tags", []),
                    "content": document["content"],
                    "precautions": document.get("precautions", []),
                    "escalation": document.get("escalation", ""),
                    "score": round(float(score), 4),
                }
            )
        return results
