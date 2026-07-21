from __future__ import annotations

from app.core.constants import (
    DISEASE_PROFILES,
    GENERIC_PREVENTION_GUIDANCE,
    MEDICAL_DISCLAIMER,
    UNCERTAIN_CONDITION,
)
from app.nlp.preprocess import sentence_split


class AdvisoryGenerator:
    def generate(
        self,
        user_text: str,
        symptoms: list[str],
        predictions: list[dict],
        retrieval_results: list[dict],
        risk_assessment,
        rule_notes: list[str] | None = None,
    ) -> dict:
        top_prediction = predictions[0] if predictions else None
        top_condition = top_prediction["disease"] if top_prediction else "Unclear pattern"
        profile = DISEASE_PROFILES.get(top_condition, {})
        hallmark_symptoms = set(profile.get("hallmark_symptoms", []))
        symptom_overlap = sorted(hallmark_symptoms.intersection(symptoms))

        evidence_snippets = []
        for result in retrieval_results[:2]:
            sentence = sentence_split(result["content"])[0]
            evidence_snippets.append(
                {
                    "title": result["title"],
                    "source": result["source"],
                    "snippet": sentence,
                }
            )

        summary = (
            f"The symptom pattern is most consistent with {top_condition} "
            f"({round(top_prediction['probability'] * 100, 1)}% model confidence)."
            if top_prediction
            else "The model could not identify a clear condition pattern from the current text."
        )

        if symptom_overlap:
            explanation = (
                f"The leading pattern aligns with the extracted symptoms {', '.join(symptom_overlap)}."
            )
        elif symptoms:
            explanation = (
                f"The model used the extracted symptom set {', '.join(symptoms)} together with the free-text description."
            )
        else:
            explanation = "No structured symptom entities were extracted, so the model relied mainly on free-text wording."

        if "high blood sugar" in symptoms and "frequent urination" in symptoms:
            explanation = (
                "The combination of high blood sugar wording and frequent urination is more aligned with a glucose-regulation concern than a typical urinary infection pattern."
            )

        if top_condition == UNCERTAIN_CONDITION:
            summary = (
                "The current symptom description does not map confidently to a single supported disease pattern in this assistant."
            )
            if "left upper abdominal pain" in symptoms:
                explanation = (
                    "The wording suggests pain around the left upper abdomen or spleen region, which needs cautious clinical evaluation rather than a forced disease label."
                )
            elif symptoms:
                explanation = (
                    f"The assistant extracted {', '.join(symptoms)}, but the current classifier coverage is not strong enough to safely highlight one disease."
                )
            else:
                explanation = (
                    "The assistant could not extract a reliable structured symptom from the wording, so it is avoiding a disease-specific answer."
                )

        care_steps = profile.get("self_care", []) + GENERIC_PREVENTION_GUIDANCE
        return {
            "summary": summary,
            "explanation": explanation,
            "care_steps": list(dict.fromkeys(care_steps))[:6],
            "when_to_escalate": profile.get("escalation", risk_assessment.recommended_action),
            "risk_level": risk_assessment.level,
            "risk_reasons": risk_assessment.reasons,
            "rule_notes": rule_notes or [],
            "evidence_snippets": evidence_snippets,
            "disclaimer": MEDICAL_DISCLAIMER,
            "input_text": user_text,
        }
