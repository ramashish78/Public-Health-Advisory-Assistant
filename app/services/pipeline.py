from __future__ import annotations

from app.core.constants import DISEASE_PROFILES, UNCERTAIN_CONDITION
from app.nlp.symptom_extractor import SymptomExtractor
from app.nlp.preprocess import sentence_split
from app.services.advisory_generator import AdvisoryGenerator
from app.services.clarification_engine import ClarificationEngine
from app.services.model_service import ModelService
from app.services.reporting import ReportingService
from app.services.retrieval_service import RetrievalService
from app.services.risk_engine import RiskEngine


class AdvisoryPipeline:
    def __init__(self) -> None:
        self.extractor = SymptomExtractor()
        self.model_service = ModelService()
        self.retrieval_service = RetrievalService()
        self.risk_engine = RiskEngine()
        self.clarification_engine = ClarificationEngine()
        self.advisory_generator = AdvisoryGenerator()
        self.reporting_service = ReportingService()

    def predict(self, text: str, top_k: int = 3) -> dict:
        extraction = self.extractor.extract(text)
        model_input = self._build_model_input(extraction.normalized_text, extraction.symptoms)
        model_output = self.model_service.predict_from_text(
            model_input,
            raw_text=extraction.normalized_text,
            symptoms=extraction.symptoms,
            top_k=top_k,
        )
        return {
            "normalized_text": extraction.normalized_text,
            "symptoms": extraction.symptoms,
            "matched_phrases": extraction.matched_phrases,
            "predictions": model_output["predictions"],
            "rule_notes": model_output["rule_notes"],
        }

    def retrieve(self, text: str, top_k: int = 3) -> dict:
        results = self.retrieval_service.retrieve(text, top_k=top_k)
        return {"query": text, "results": results}

    def analyze(self, text: str, top_k: int = 3, answered_question_ids: list[str] | None = None) -> dict:
        prediction_payload = self.predict(text, top_k=top_k)
        top_condition = prediction_payload["predictions"][0]["disease"] if prediction_payload["predictions"] else None
        retrieval_query = prediction_payload["normalized_text"]
        if top_condition and top_condition != UNCERTAIN_CONDITION:
            retrieval_query = f"{prediction_payload['normalized_text']} {top_condition}"
        retrieved = self.retrieval_service.retrieve(retrieval_query, top_k=top_k)
        risk = self.risk_engine.assess(
            prediction_payload["symptoms"],
            prediction_payload["normalized_text"],
            top_condition=top_condition,
        )
        advice = self.advisory_generator.generate(
            user_text=text,
            symptoms=prediction_payload["symptoms"],
            predictions=prediction_payload["predictions"],
            retrieval_results=retrieved,
            risk_assessment=risk,
            rule_notes=prediction_payload["rule_notes"],
        )
        primary_condition = self._build_primary_condition(
            predictions=prediction_payload["predictions"],
            symptoms=prediction_payload["symptoms"],
            retrieval_results=retrieved,
        )
        clarification = self.clarification_engine.build(
            user_text=text,
            symptoms=prediction_payload["symptoms"],
            predictions=prediction_payload["predictions"],
            risk_level=risk.level,
            answered_question_ids=answered_question_ids or [],
        )

        return {
            "input_text": text,
            "normalized_text": prediction_payload["normalized_text"],
            "symptoms": prediction_payload["symptoms"],
            "matched_phrases": prediction_payload["matched_phrases"],
            "predictions": prediction_payload["predictions"],
            "primary_condition": primary_condition,
            "clarification": clarification,
            "decision_support": {
                "rule_notes": prediction_payload["rule_notes"],
            },
            "retrieval_results": retrieved,
            "risk": {
                "level": risk.level,
                "reasons": risk.reasons,
                "recommended_action": risk.recommended_action,
            },
            "advice": advice,
        }

    def report(self, text: str, top_k: int = 3, answered_question_ids: list[str] | None = None) -> dict:
        analysis = self.analyze(text, top_k=top_k, answered_question_ids=answered_question_ids)
        return self.reporting_service.create_report(analysis)

    @staticmethod
    def _build_model_input(normalized_text: str, symptoms: list[str]) -> str:
        symptom_string = ", ".join(symptoms) if symptoms else "none extracted"
        return f"{normalized_text} Symptoms: {symptom_string}"

    @staticmethod
    def _build_primary_condition(
        predictions: list[dict],
        symptoms: list[str],
        retrieval_results: list[dict],
    ) -> dict | None:
        if not predictions:
            return None

        top_prediction = predictions[0]
        disease = top_prediction["disease"]
        profile = DISEASE_PROFILES.get(disease, {})
        hallmark_symptoms = profile.get("hallmark_symptoms", [])
        if disease == UNCERTAIN_CONDITION:
            matched_symptoms = symptoms
            unmatched_symptoms = []
        else:
            matched_symptoms = [symptom for symptom in hallmark_symptoms if symptom in symptoms]
            unmatched_symptoms = [symptom for symptom in hallmark_symptoms if symptom not in symptoms]
        related_documents = [
            {
                "title": item["title"],
                "source": item["source"],
                "score": item["score"],
                "snippet": sentence_split(item["content"])[0] if item.get("content") else "",
            }
            for item in retrieval_results
            if item.get("disease") in {disease, "General"}
        ][:2]

        return {
            "disease": disease,
            "confidence": top_prediction["probability"],
            "is_uncertain": disease == UNCERTAIN_CONDITION,
            "description": profile.get("description", top_prediction.get("description", "")),
            "matched_symptoms": matched_symptoms,
            "hallmark_symptoms": hallmark_symptoms,
            "related_symptoms_to_watch": unmatched_symptoms[:4],
            "care_priorities": profile.get("self_care", [])[:3],
            "escalation": profile.get("escalation", ""),
            "focused_evidence": related_documents,
        }
