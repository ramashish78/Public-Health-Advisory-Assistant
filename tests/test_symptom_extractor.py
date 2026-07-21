from app.nlp.symptom_extractor import SymptomExtractor


def test_symptom_extractor_finds_canonical_terms() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("I have a high temperature, dry cough and body pain.")
    assert "fever" in result.symptoms
    assert "cough" in result.symptoms
    assert "body ache" in result.symptoms


def test_symptom_extractor_handles_negation() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("I have a cough but no fever today.")
    assert "cough" in result.symptoms
    assert "fever" not in result.symptoms


def test_symptom_extractor_detects_high_blood_sugar_language() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("I am having frequent urination and my blood sugar level is high.")
    assert "frequent urination" in result.symptoms
    assert "high blood sugar" in result.symptoms


def test_symptom_extractor_detects_spleen_region_pain() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("I am feeling pain in back of the spleen from past 3 days.")
    assert "left upper abdominal pain" in result.symptoms


def test_symptom_extractor_detects_back_strain_language() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("My lower back hurts after lifting something heavy.")
    assert "back pain" in result.symptoms
    assert "stiffness" in result.symptoms


def test_symptom_extractor_detects_blood_in_urine_language() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("There is blood in my urine and sharp side pain.")
    assert "blood in urine" in result.symptoms
    assert "back pain" in result.symptoms


def test_symptom_extractor_detects_colloquial_urinary_pain_language() -> None:
    extractor = SymptomExtractor()
    result = extractor.extract("I have pain while peeing and side discomfort.")
    assert "burning urination" in result.symptoms
    assert "back pain" in result.symptoms
