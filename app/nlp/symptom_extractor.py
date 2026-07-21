from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.matcher import PhraseMatcher

from app.core.constants import SYMPTOM_SYNONYMS
from app.nlp.preprocess import normalize_text


NEGATION_TERMS = {"no", "not", "without", "never", "denies"}


@dataclass(slots=True)
class SymptomExtractionResult:
    normalized_text: str
    symptoms: list[str]
    matched_phrases: list[str]


class SymptomExtractor:
    def __init__(self) -> None:
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.alias_to_canonical: dict[str, str] = {}
        patterns = []
        for canonical, variants in SYMPTOM_SYNONYMS.items():
            for phrase in {canonical, *variants}:
                self.alias_to_canonical[phrase.lower()] = canonical
                patterns.append(self.nlp.make_doc(phrase))
        self.matcher.add("SYMPTOM", patterns)

    def extract(self, text: str) -> SymptomExtractionResult:
        normalized = normalize_text(text)
        doc = self.nlp(normalized)
        symptoms: list[str] = []
        matched_phrases: list[str] = []
        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            if self._is_negated(doc, start):
                continue
            phrase = span.text.lower()
            canonical = self.alias_to_canonical.get(phrase)
            if canonical and canonical not in symptoms:
                symptoms.append(canonical)
                matched_phrases.append(phrase)

        self._apply_anatomical_heuristics(normalized, symptoms, matched_phrases)

        return SymptomExtractionResult(
            normalized_text=normalized,
            symptoms=sorted(symptoms),
            matched_phrases=matched_phrases,
        )

    @staticmethod
    def _append_symptom(
        symptoms: list[str],
        matched_phrases: list[str],
        canonical: str,
        phrase: str,
    ) -> None:
        if canonical not in symptoms:
            symptoms.append(canonical)
            matched_phrases.append(phrase)

    @staticmethod
    def _is_negated(doc, start: int) -> bool:
        window_start = max(0, start - 3)
        prior_tokens = {token.text.lower() for token in doc[window_start:start]}
        return any(token in NEGATION_TERMS for token in prior_tokens)

    @classmethod
    def _apply_anatomical_heuristics(
        cls,
        normalized_text: str,
        symptoms: list[str],
        matched_phrases: list[str],
    ) -> None:
        pain_terms = ("pain", "hurt", "hurts", "hurting", "ache", "aches", "aching", "sore")
        spleen_markers = (
            "spleen",
            "left upper abdomen",
            "left upper belly",
            "left upper quadrant",
            "under left ribs",
        )
        if any(term in normalized_text for term in pain_terms) and any(
            marker in normalized_text for marker in spleen_markers
        ):
            cls._append_symptom(
                symptoms,
                matched_phrases,
                "left upper abdominal pain",
                "spleen-region pain heuristic",
            )

        back_markers = (
            "back pain",
            "backache",
            "back hurts",
            "lower back",
            "upper back",
            "mid back",
            "flank",
            "side pain",
            "sharp side pain",
        )
        exertion_markers = ("lift", "lifting", "lifted", "heavy", "strain", "strained", "pulled")
        has_specific_back_language = any(marker in normalized_text for marker in back_markers)
        has_exertional_back_language = "back" in normalized_text and any(
            marker in normalized_text for marker in exertion_markers
        )
        if (has_specific_back_language or has_exertional_back_language) and any(
            term in normalized_text for term in (*pain_terms, *exertion_markers)
        ):
            cls._append_symptom(
                symptoms,
                matched_phrases,
                "back pain",
                "back-pain heuristic",
            )
            if any(marker in normalized_text for marker in exertion_markers):
                cls._append_symptom(
                    symptoms,
                    matched_phrases,
                    "stiffness",
                    "back-strain heuristic",
                )

        urine_markers = ("urine", "pee", "urinating")
        blood_markers = ("blood", "bloody", "red urine", "pink urine")
        if any(marker in normalized_text for marker in urine_markers) and any(
            marker in normalized_text for marker in blood_markers
        ):
            cls._append_symptom(
                symptoms,
                matched_phrases,
                "blood in urine",
                "blood-in-urine heuristic",
            )
