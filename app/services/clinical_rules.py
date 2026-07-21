from __future__ import annotations

from dataclasses import dataclass

from app.core.constants import DISEASE_PROFILES, UNCERTAIN_CONDITION

GLYCEMIC_CONDITION = "Hyperglycemia / Diabetes Concern"
UTI_CONDITION = "Urinary Tract Infection"
KIDNEY_STONE_CONDITION = "Kidney Stone / Renal Colic"
BACK_STRAIN_CONDITION = "Musculoskeletal Back Strain"
METABOLIC_TERMS = {
    "blood sugar",
    "glucose",
    "sugar level",
    "hyperglycemia",
}
UTI_SPECIFIC_SYMPTOMS = {"burning urination", "fever", "back pain"}
GLYCEMIC_SUPPORT_SYMPTOMS = {
    "high blood sugar",
    "frequent urination",
    "excessive thirst",
    "fatigue",
    "blurred vision",
    "weight loss",
    "increased hunger",
}
KIDNEY_STONE_SUPPORT_SYMPTOMS = {"back pain", "blood in urine", "nausea", "vomiting", "burning urination"}
BACK_STRAIN_SUPPORT_SYMPTOMS = {"back pain", "stiffness", "body ache", "fatigue"}
ANATOMICAL_PAIN_SYMPTOMS = {"left upper abdominal pain"}
LOW_CONFIDENCE_THRESHOLD = 0.45
LOW_MARGIN_THRESHOLD = 0.12


@dataclass(slots=True)
class RuleResult:
    predictions: list[dict]
    notes: list[str]


class ClinicalRuleEngine:
    def adjust_predictions(
        self,
        predictions: list[dict],
        raw_text: str,
        symptoms: list[str],
        top_k: int = 3,
    ) -> RuleResult:
        if not predictions:
            return RuleResult(predictions=[], notes=[])

        lowered = raw_text.lower()
        symptom_set = set(symptoms)
        adjusted = {item["disease"]: float(item["probability"]) for item in predictions}
        descriptions = {item["disease"]: item.get("description", "") for item in predictions}
        notes: list[str] = []

        has_metabolic_language = (
            "high blood sugar" in symptom_set
            or any(term in lowered for term in METABOLIC_TERMS)
        )
        has_glycemic_cluster = len(GLYCEMIC_SUPPORT_SYMPTOMS.intersection(symptom_set)) >= 2
        has_uti_specific_features = bool(UTI_SPECIFIC_SYMPTOMS.intersection(symptom_set))

        if has_metabolic_language and GLYCEMIC_CONDITION in adjusted:
            boost = 0.42
            if "frequent urination" in symptom_set:
                boost += 0.16
            if has_glycemic_cluster:
                boost += 0.12
            adjusted[GLYCEMIC_CONDITION] += boost
            notes.append(
                "Glucose-related wording and metabolic symptoms increased the weight of a hyperglycemia pattern."
            )

            if UTI_CONDITION in adjusted and not has_uti_specific_features:
                adjusted[UTI_CONDITION] *= 0.18
                notes.append(
                    "UTI confidence was reduced because classic urinary-infection features such as burning urination, fever, or back pain were not present."
                )

        if {"burning urination", "frequent urination"}.issubset(symptom_set) and UTI_CONDITION in adjusted:
            adjusted[UTI_CONDITION] += 0.28
            notes.append(
                "Burning plus frequent urination strengthened the urinary-infection pattern."
            )

        has_kidney_stone_cluster = len(KIDNEY_STONE_SUPPORT_SYMPTOMS.intersection(symptom_set)) >= 2
        has_renal_language = any(term in lowered for term in {"flank", "side pain", "sharp side pain"})
        if KIDNEY_STONE_CONDITION in adjusted and (has_kidney_stone_cluster or has_renal_language):
            boost = 0.34
            if {"back pain", "blood in urine"}.issubset(symptom_set):
                boost += 0.18
            if has_renal_language:
                boost += 0.08
            adjusted[KIDNEY_STONE_CONDITION] += boost
            notes.append(
                "Blood-in-urine and flank or back-pain language increased the weight of a kidney-stone pattern."
            )

            if UTI_CONDITION in adjusted and "burning urination" not in symptom_set:
                adjusted[UTI_CONDITION] *= 0.4
                notes.append(
                    "UTI confidence was reduced because urinary burning was not present while flank or blood-in-urine features favored a kidney-stone pattern."
                )

        has_back_strain_cluster = len(BACK_STRAIN_SUPPORT_SYMPTOMS.intersection(symptom_set)) >= 2
        has_strain_language = any(term in lowered for term in {"lift", "lifting", "lifted", "strain", "strained", "pulled"})
        has_urinary_features = bool({"blood in urine", "burning urination", "frequent urination"}.intersection(symptom_set))
        if BACK_STRAIN_CONDITION in adjusted and (has_back_strain_cluster or has_strain_language) and not has_urinary_features:
            boost = 0.3
            if "back pain" in symptom_set:
                boost += 0.12
            if has_strain_language:
                boost += 0.12
            adjusted[BACK_STRAIN_CONDITION] += boost
            notes.append(
                "Back-pain wording with exertion or strain language increased the weight of a musculoskeletal back-strain pattern."
            )

        total = sum(max(score, 0.0) for score in adjusted.values())
        if total <= 0:
            return RuleResult(predictions=predictions[:top_k], notes=notes)

        reranked = sorted(
            [
                {
                    "disease": disease,
                    "probability": round(max(score, 0.0) / total, 4),
                    "description": descriptions.get(disease, ""),
                }
                for disease, score in adjusted.items()
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        return self._apply_uncertainty_guardrails(
            reranked=reranked,
            raw_text=raw_text,
            symptoms=symptoms,
            notes=notes,
            top_k=top_k,
        )

    def _apply_uncertainty_guardrails(
        self,
        reranked: list[dict],
        raw_text: str,
        symptoms: list[str],
        notes: list[str],
        top_k: int,
    ) -> RuleResult:
        if not reranked:
            return RuleResult(predictions=[], notes=notes)

        if not symptoms:
            notes.append(
                "No structured symptom entities were recognized, so the assistant is falling back to an uncertainty response instead of forcing a disease label."
            )
            return RuleResult(
                predictions=self._build_uncertain_predictions(reranked, reason="no_symptoms", top_k=top_k),
                notes=notes,
            )

        top_prediction = reranked[0]
        second_probability = reranked[1]["probability"] if len(reranked) > 1 else 0.0
        top_probability = top_prediction["probability"]
        probability_margin = top_probability - second_probability
        top_profile = DISEASE_PROFILES.get(top_prediction["disease"], {})
        top_hallmarks = set(top_profile.get("hallmark_symptoms", []))
        symptom_overlap = len(top_hallmarks.intersection(symptoms))

        if ANATOMICAL_PAIN_SYMPTOMS.intersection(symptoms):
            notes.append(
                "Regional pain around the spleen or left upper abdomen is outside the current disease-class coverage, so the assistant is using a cautious uncertainty fallback."
            )
            return RuleResult(
                predictions=self._build_uncertain_predictions(reranked, reason="anatomical_pain", top_k=top_k),
                notes=notes,
            )

        if top_prediction["disease"] == UNCERTAIN_CONDITION:
            return RuleResult(predictions=reranked[:top_k], notes=notes)

        if symptom_overlap >= 3 and top_probability >= 0.15:
            return RuleResult(predictions=reranked[:top_k], notes=notes)

        if symptom_overlap >= 2 and top_probability >= 0.2:
            return RuleResult(predictions=reranked[:top_k], notes=notes)

        if top_probability < LOW_CONFIDENCE_THRESHOLD or probability_margin < LOW_MARGIN_THRESHOLD:
            notes.append(
                "The top model score was too weak or too close to alternatives, so the assistant is avoiding an exact disease highlight."
            )
            return RuleResult(
                predictions=self._build_uncertain_predictions(reranked, reason="low_confidence", top_k=top_k),
                notes=notes,
            )

        if symptom_overlap == 0 and top_probability < 0.6:
            notes.append(
                "The leading disease did not overlap with the extracted symptoms strongly enough to justify a confident disease-specific response."
            )
            return RuleResult(
                predictions=self._build_uncertain_predictions(reranked, reason="low_overlap", top_k=top_k),
                notes=notes,
            )

        return RuleResult(predictions=reranked[:top_k], notes=notes)

    @staticmethod
    def _build_uncertain_predictions(reranked: list[dict], reason: str, top_k: int) -> list[dict]:
        fallback_probability = reranked[0]["probability"] if reranked else 0.0
        uncertain_prediction = {
            "disease": UNCERTAIN_CONDITION,
            "probability": round(float(fallback_probability), 4),
            "description": DISEASE_PROFILES[UNCERTAIN_CONDITION]["description"],
            "reason": reason,
        }
        alternatives = [item for item in reranked if item["disease"] != UNCERTAIN_CONDITION]
        return [uncertain_prediction, *alternatives[: max(0, top_k - 1)]]
