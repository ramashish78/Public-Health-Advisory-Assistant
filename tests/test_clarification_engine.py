from app.services.clarification_engine import ClarificationEngine


def test_clarification_engine_requests_follow_up_for_uncertain_spleen_area_pain() -> None:
    engine = ClarificationEngine()
    result = engine.build(
        user_text="feeling pain in back of spleen",
        symptoms=["left upper abdominal pain"],
        predictions=[
            {
                "disease": "Unclear Pattern / Needs Clinical Evaluation",
                "probability": 0.68,
                "description": "Uncertain pattern",
            },
            {
                "disease": "Musculoskeletal Back Strain",
                "probability": 0.14,
                "description": "Back strain",
            },
        ],
        risk_level="MEDIUM",
    )

    assert result["should_clarify"] is True
    assert result["priority"] == "HIGH"
    assert result["questions"]
    assert any(question["id"] == "trauma_trigger" for question in result["questions"])


def test_clarification_engine_requests_urinary_questions_for_ambiguous_uti_vs_stone() -> None:
    engine = ClarificationEngine()
    result = engine.build(
        user_text="I have pain while peeing and side discomfort",
        symptoms=["burning urination", "back pain"],
        predictions=[
            {
                "disease": "Urinary Tract Infection",
                "probability": 0.44,
                "description": "UTI pattern",
            },
            {
                "disease": "Kidney Stone / Renal Colic",
                "probability": 0.33,
                "description": "Stone pattern",
            },
        ],
        risk_level="MEDIUM",
    )

    question_ids = {question["id"] for question in result["questions"]}
    assert result["should_clarify"] is True
    assert result["reason"] == "ambiguous"
    assert {"uti_vs_stone_burning", "uti_vs_stone_blood"}.intersection(question_ids)


def test_clarification_engine_stays_quiet_for_confident_supported_pattern() -> None:
    engine = ClarificationEngine()
    result = engine.build(
        user_text="I am getting frequent urination and the blood sugar level is high from last 4 days.",
        symptoms=["frequent urination", "high blood sugar"],
        predictions=[
            {
                "disease": "Hyperglycemia / Diabetes Concern",
                "probability": 0.91,
                "description": "Glucose concern",
            },
            {
                "disease": "Kidney Stone / Renal Colic",
                "probability": 0.04,
                "description": "Stone concern",
            },
        ],
        risk_level="MEDIUM",
    )

    assert result["should_clarify"] is False
    assert result["questions"] == []


def test_clarification_engine_does_not_repeat_questions_after_follow_up_round() -> None:
    engine = ClarificationEngine()
    result = engine.build(
        user_text="I have headache. Additional detail: I have had fever or a measured high temperature.",
        symptoms=["fever", "headache"],
        predictions=[
            {
                "disease": "Dengue-like Viral Fever",
                "probability": 0.52,
                "description": "Viral fever pattern",
            },
            {
                "disease": "Migraine",
                "probability": 0.23,
                "description": "Headache pattern",
            },
        ],
        risk_level="MEDIUM",
        answered_question_ids=["confirm_fever"],
    )

    assert result["should_clarify"] is False
    assert result["questions"] == []


def test_clarification_engine_keeps_headache_questions_in_headache_domain() -> None:
    engine = ClarificationEngine()
    result = engine.build(
        user_text="I have headache",
        symptoms=["headache"],
        predictions=[
            {
                "disease": "Unclear Pattern / Needs Clinical Evaluation",
                "probability": 0.26,
                "description": "Uncertain pattern",
            },
            {
                "disease": "COVID-19",
                "probability": 0.19,
                "description": "Respiratory pattern",
            },
            {
                "disease": "Migraine",
                "probability": 0.16,
                "description": "Headache pattern",
            },
        ],
        risk_level="MEDIUM",
    )

    questions = " ".join(question["question"].lower() for question in result["questions"])
    assert "chest tightness" not in questions
    assert "shortness of breath" not in questions
    assert "headache-related" in questions or "infection-like symptoms" in questions
