from app.services.risk_engine import RiskEngine


def test_high_risk_for_breathing_issue() -> None:
    engine = RiskEngine()
    result = engine.assess(
        symptoms=["shortness of breath", "cough"],
        text="I have shortness of breath and cough for 1 day",
        top_condition="Pneumonia",
    )
    assert result.level == "HIGH"


def test_medium_risk_for_persistent_fever() -> None:
    engine = RiskEngine()
    result = engine.assess(
        symptoms=["fever", "cough"],
        text="I have fever and cough for 4 days",
        top_condition="Influenza",
    )
    assert result.level == "MEDIUM"


def test_low_risk_for_mild_allergy_pattern() -> None:
    engine = RiskEngine()
    result = engine.assess(
        symptoms=["sneezing", "itchy eyes", "runny nose"],
        text="I have sneezing and itchy eyes for 1 day",
        top_condition="Allergic Rhinitis",
    )
    assert result.level == "LOW"
