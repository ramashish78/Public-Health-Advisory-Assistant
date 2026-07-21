from __future__ import annotations

from dataclasses import dataclass

from app.core.constants import EMERGENCY_SYMPTOMS, UNCERTAIN_CONDITION, URGENT_PHRASES
from app.nlp.preprocess import parse_duration_days


@dataclass(slots=True)
class RiskAssessment:
    level: str
    reasons: list[str]
    recommended_action: str


class RiskEngine:
    def assess(self, symptoms: list[str], text: str, top_condition: str | None = None) -> RiskAssessment:
        lowered = text.lower()
        reasons: list[str] = []
        duration_days = parse_duration_days(lowered)

        if any(symptom in symptoms for symptom in EMERGENCY_SYMPTOMS):
            reasons.append("Emergency-pattern symptom detected, such as chest pain or breathing difficulty.")
        if any(phrase in lowered for phrase in URGENT_PHRASES):
            reasons.append("The message contains urgent red-flag language.")
        if top_condition in {"Pneumonia", "COVID-19"} and "shortness of breath" in symptoms:
            reasons.append(f"{top_condition} with breathing symptoms increases risk.")
        if top_condition == "Hyperglycemia / Diabetes Concern" and any(
            phrase in lowered for phrase in ("fruity breath", "confused", "vomiting", "very drowsy")
        ):
            reasons.append("High blood sugar with metabolic red flags may indicate an urgent complication.")
        if reasons:
            return RiskAssessment(
                level="HIGH",
                reasons=reasons,
                recommended_action="Seek urgent in-person medical evaluation or emergency care now.",
            )

        if "fever" in symptoms and duration_days is not None and duration_days >= 3:
            reasons.append("Fever has persisted for several days.")
        if "dehydration" in symptoms or ("vomiting" in symptoms and "diarrhea" in symptoms):
            reasons.append("Fluid loss raises dehydration risk.")
        if "left upper abdominal pain" in symptoms:
            reasons.append("Persistent localized pain around the left upper abdomen or spleen region needs clinician evaluation.")
        if "right lower abdominal pain" in symptoms:
            reasons.append("Right-lower-abdominal pain can need urgent abdominal evaluation, especially if it is worsening.")
        if "blood in urine" in symptoms:
            reasons.append("Blood in urine should be clinically assessed, especially with pain or urinary symptoms.")
        if top_condition in {"Influenza", "Bronchitis", "Urinary Tract Infection", "Dengue-like Viral Fever"}:
            reasons.append(f"{top_condition} often needs timely clinician follow-up when symptoms persist.")
        if top_condition in {"Kidney Stone / Renal Colic", "Appendicitis Concern", "Gastritis / Acid Peptic Disorder", "Ear Infection", "Mononucleosis-like Illness"}:
            reasons.append(f"{top_condition} is a pattern that benefits from direct clinician assessment rather than watchful waiting alone.")
        if top_condition == "Hyperglycemia / Diabetes Concern" or "high blood sugar" in symptoms:
            reasons.append("High blood sugar symptoms should be reviewed clinically, especially if they persist for days.")
        if top_condition == UNCERTAIN_CONDITION and (symptoms or duration_days):
            reasons.append("The symptom pattern is outside the assistant's confident disease coverage, so a clinician should assess it directly.")
        if reasons:
            return RiskAssessment(
                level="MEDIUM",
                reasons=reasons,
                recommended_action="Arrange a clinician review within 24 hours and monitor closely.",
            )

        return RiskAssessment(
            level="LOW",
            reasons=["No immediate red-flag pattern was detected from the current symptom description."],
            recommended_action="Monitor symptoms, use conservative self-care, and escalate if symptoms worsen.",
        )
