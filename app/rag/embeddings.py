from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingArtifact:
    mode: str
    model_name: str
    vectorizer: TfidfVectorizer | None = None


class EmbeddingEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.mode: str | None = None
        self.model = None
        self.vectorizer: TfidfVectorizer | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer

            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            if settings.allow_model_download:
                self.model = SentenceTransformer(self.model_name)
            else:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            self.mode = "sentence-transformer"
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.astype("float32")
        except Exception as exc:  # pragma: no cover - fallback path depends on environment
            logger.warning("Falling back to TF-IDF embeddings because SentenceTransformer failed: %s", exc)
            self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self.mode = "tfidf"
            matrix = self.vectorizer.fit_transform(texts)
            return normalize(matrix).astype("float32").toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.mode == "sentence-transformer" and self.model is not None:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        if self.mode == "tfidf" and self.vectorizer is not None:
            matrix = self.vectorizer.transform(texts)
            return normalize(matrix).astype("float32").toarray()
        raise RuntimeError("Embedding engine is not initialized.")

    def save_vectorizer(self, path) -> None:
        if self.vectorizer is not None:
            dump(self.vectorizer, path)

    def load_from_metadata(self, metadata: dict, vectorizer_path) -> None:
        self.mode = metadata["embedding_mode"]
        if self.mode == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            if settings.allow_model_download:
                self.model = SentenceTransformer(metadata["embedding_model"])
            else:
                self.model = SentenceTransformer(metadata["embedding_model"], local_files_only=True)
            self.model_name = metadata["embedding_model"]
        elif self.mode == "tfidf":
            self.vectorizer = load(vectorizer_path)
            self.model_name = metadata["embedding_model"]
