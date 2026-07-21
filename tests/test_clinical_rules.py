from app.services.clinical_rules import ClinicalRuleEngine


def test_clinical_rules_prefer_hyperglycemia_over_uti_without_uti_signs() -> None:
    engine = ClinicalRuleEngine()
    base_predictions = [
        {
            "disease": "Urinary Tract Infection",
            "probability": 0.86,
            "description": "UTI pattern",
        },
        {
            "disease": "Hyperglycemia / Diabetes Concern",
            "probability": 0.08,
            "description": "Glucose pattern",
        },
        {
            "disease": "Influenza",
            "probability": 0.06,
            "description": "Flu pattern",
        },
    ]

    result = engine.adjust_predictions(
        base_predictions,
        raw_text="i am getting frequent urination and the blood sugar level is high from last 4 days",
        symptoms=["frequent urination", "high blood sugar"],
        top_k=3,
    )

    assert result.predictions[0]["disease"] == "Hyperglycemia / Diabetes Concern"
    assert any("UTI confidence was reduced" in note for note in result.notes)


def test_clinical_rules_fall_back_to_uncertain_for_anatomical_pain() -> None:
    engine = ClinicalRuleEngine()
    base_predictions = [
        {
            "disease": "Urinary Tract Infection",
            "probability": 0.26,
            "description": "UTI pattern",
        },
        {
            "disease": "COVID-19",
            "probability": 0.13,
            "description": "Respiratory pattern",
        },
        {
            "disease": "Pneumonia",
            "probability": 0.12,
            "description": "Lung pattern",
        },
    ]

    result = engine.adjust_predictions(
        base_predictions,
        raw_text="i am feeling pain in back of the spleen from past 3 days",
        symptoms=["left upper abdominal pain"],
        top_k=3,
    )

    assert result.predictions[0]["disease"] == "Unclear Pattern / Needs Clinical Evaluation"
    assert any("outside the current disease-class coverage" in note for note in result.notes)


def test_clinical_rules_prefer_kidney_stone_for_blood_in_urine_with_side_pain() -> None:
    engine = ClinicalRuleEngine()
    base_predictions = [
        {
            "disease": "Urinary Tract Infection",
            "probability": 0.38,
            "description": "UTI pattern",
        },
        {
            "disease": "Kidney Stone / Renal Colic",
            "probability": 0.32,
            "description": "Renal colic pattern",
        },
        {
            "disease": "Hyperglycemia / Diabetes Concern",
            "probability": 0.11,
            "description": "Glucose pattern",
        },
    ]

    result = engine.adjust_predictions(
        base_predictions,
        raw_text="there is blood in my urine and sharp side pain",
        symptoms=["blood in urine", "back pain"],
        top_k=3,
    )

    assert result.predictions[0]["disease"] == "Kidney Stone / Renal Colic"
    assert any("kidney-stone pattern" in note for note in result.notes)


def test_clinical_rules_prefer_back_strain_for_lifting_related_back_pain() -> None:
    engine = ClinicalRuleEngine()
    base_predictions = [
        {
            "disease": "Kidney Stone / Renal Colic",
            "probability": 0.22,
            "description": "Renal colic pattern",
        },
        {
            "disease": "Musculoskeletal Back Strain",
            "probability": 0.2,
            "description": "Back strain pattern",
        },
        {
            "disease": "Urinary Tract Infection",
            "probability": 0.15,
            "description": "UTI pattern",
        },
    ]

    result = engine.adjust_predictions(
        base_predictions,
        raw_text="my lower back hurts after lifting something heavy",
        symptoms=["back pain", "stiffness"],
        top_k=3,
    )

    assert result.predictions[0]["disease"] == "Musculoskeletal Back Strain"
    assert any("back-strain pattern" in note for note in result.notes)
