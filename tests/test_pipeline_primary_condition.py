from app.services.pipeline import AdvisoryPipeline


def test_build_primary_condition_focuses_on_top_disease() -> None:
    primary = AdvisoryPipeline._build_primary_condition(
        predictions=[
            {
                "disease": "Hyperglycemia / Diabetes Concern",
                "probability": 0.94,
                "description": "Glucose concern",
            }
        ],
        symptoms=["high blood sugar", "frequent urination"],
        retrieval_results=[
            {
                "title": "High Blood Sugar Symptom Pattern",
                "source": "Simulated endocrine clinic guidance",
                "score": 0.31,
                "content": "High blood sugar can cause frequent urination and excessive thirst.",
                "disease": "Hyperglycemia / Diabetes Concern",
            },
            {
                "title": "General Safety",
                "source": "Simulated notes",
                "score": 0.11,
                "content": "Seek care if symptoms worsen.",
                "disease": "General",
            },
        ],
    )

    assert primary is not None
    assert primary["disease"] == "Hyperglycemia / Diabetes Concern"
    assert "high blood sugar" in primary["matched_symptoms"]
    assert primary["focused_evidence"][0]["title"] == "High Blood Sugar Symptom Pattern"
