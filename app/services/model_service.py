from __future__ import annotations

from pathlib import Path

import joblib

from app.core.config import settings
from app.core.constants import DISEASE_PROFILES
from app.services.clinical_rules import ClinicalRuleEngine


class ModelService:
    def __init__(self, artifact_path: Path | None = None) -> None:
        artifact_path = artifact_path or settings.model_artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {artifact_path}. Run scripts/bootstrap_project.py first."
            )
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.labels = artifact["labels"]
        self.rule_engine = ClinicalRuleEngine()

    def predict_from_text(
        self,
        text: str,
        raw_text: str | None = None,
        symptoms: list[str] | None = None,
        top_k: int = 3,
    ) -> dict:
        probabilities = self.pipeline.predict_proba([text])[0]
        ranked = sorted(
            [
                {
                    "disease": label,
                    "probability": round(float(probability), 4),
                    "description": DISEASE_PROFILES.get(label, {}).get("description", ""),
                }
                for label, probability in zip(self.labels, probabilities)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        if raw_text is not None and symptoms is not None:
            rule_result = self.rule_engine.adjust_predictions(
                ranked,
                raw_text=raw_text,
                symptoms=symptoms,
                top_k=top_k,
            )
            return {
                "predictions": rule_result.predictions,
                "rule_notes": rule_result.notes,
            }

        return {
            "predictions": ranked[:top_k],
            "rule_notes": [],
        }
