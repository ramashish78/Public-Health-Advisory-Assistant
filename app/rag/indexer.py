from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.rag.embeddings import EmbeddingEngine

try:  # pragma: no cover - import availability depends on environment
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _chunk_document(document: dict) -> dict:
    combined = " ".join(
        [
            document["title"],
            document["content"],
            "Precautions: " + ", ".join(document.get("precautions", [])),
            "Escalation: " + document.get("escalation", ""),
        ]
    )
    return {
        **document,
        "embedding_text": combined.strip(),
    }


def build_vector_store(
    corpus_path: Path | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict:
    corpus_path = corpus_path or settings.medical_corpus_path
    index_path = index_path or settings.faiss_index_path
    metadata_path = metadata_path or settings.retrieval_metadata_path
    settings.ensure_directories()

    documents = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(_chunk_document(json.loads(line)))

    engine = EmbeddingEngine(settings.embedding_model)
    embeddings = engine.fit_transform([doc["embedding_text"] for doc in documents])

    backend = "faiss" if faiss is not None else "numpy"
    if backend == "faiss":
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, str(index_path))
    else:  # pragma: no cover - depends on faiss availability
        np.save(index_path.with_suffix(".npy"), embeddings.astype("float32"))

    settings.retrieval_documents_path.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    if engine.mode == "tfidf":
        engine.save_vectorizer(settings.retrieval_vectorizer_path)

    metadata = {
        "embedding_mode": engine.mode,
        "embedding_model": settings.embedding_model,
        "index_backend": backend,
        "document_count": len(documents),
        "dimension": int(embeddings.shape[1]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
