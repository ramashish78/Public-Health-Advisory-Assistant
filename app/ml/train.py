from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings


def top_k_accuracy(probabilities, labels, true_labels, k: int = 3) -> float:
    matches = 0
    for row, truth in zip(probabilities, true_labels):
        top_indices = row.argsort()[::-1][:k]
        top_labels = {labels[index] for index in top_indices}
        if truth in top_labels:
            matches += 1
    return matches / len(true_labels)


def train_classifier(
    dataset_path: Path | None = None,
    model_output_path: Path | None = None,
    metrics_output_path: Path | None = None,
) -> dict:
    dataset_path = dataset_path or settings.training_dataset_path
    model_output_path = model_output_path or settings.model_artifact_path
    metrics_output_path = metrics_output_path or settings.model_metrics_path
    settings.ensure_directories()

    dataframe = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["training_text"],
        dataframe["disease"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["disease"],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    labels = list(pipeline.named_steps["classifier"].classes_)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "macro_f1": round(f1_score(y_test, predictions, average="macro"), 4),
        "top_3_accuracy": round(top_k_accuracy(probabilities, labels, y_test.tolist(), k=3), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "training_rows": int(len(dataframe)),
    }

    artifact = {
        "pipeline": pipeline,
        "labels": labels,
        "feature_columns": ["training_text"],
    }
    joblib.dump(artifact, model_output_path)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
