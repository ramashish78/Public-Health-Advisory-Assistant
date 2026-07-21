from __future__ import annotations

from app.core.constants import DISEASE_PROFILES, UNCERTAIN_CONDITION


CONFIDENCE_THRESHOLD = 0.72
MARGIN_THRESHOLD = 0.18
RESPIRATORY_CONTEXT_SYMPTOMS = {
    "cough",
    "shortness of breath",
    "wheezing",
    "chest pain",
    "fever",
    "congestion",
    "runny nose",
    "sore throat",
    "loss of taste",
    "loss of smell",
}
UPPER_AIRWAY_CONTEXT_SYMPTOMS = {
    "congestion",
    "runny nose",
    "sneezing",
    "sore throat",
    "facial pain",
    "itchy eyes",
}
ABDOMINAL_CONTEXT_SYMPTOMS = {
    "abdominal pain",
    "right lower abdominal pain",
    "left upper abdominal pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "heartburn",
    "bloating",
    "appetite loss",
}
URINARY_CONTEXT_SYMPTOMS = {
    "burning urination",
    "frequent urination",
    "blood in urine",
    "back pain",
}
HEADACHE_CONTEXT_SYMPTOMS = {
    "headache",
    "sensitivity to light",
    "dizziness",
    "facial pain",
    "ear pain",
}
RESPIRATORY_TERMS = ("cough", "breath", "wheez", "chest", "congestion", "nose", "throat")
UPPER_AIRWAY_TERMS = ("sinus", "congestion", "runny", "stuffy", "sneez", "itchy eyes", "cheeks", "forehead")
ABDOMINAL_TERMS = ("abdomen", "abdominal", "belly", "stomach", "flank", "ribs", "nausea", "vomit", "diarrhea")
URINARY_TERMS = ("urinat", "pee", "urine", "bathroom", "burn", "stinging")
HEADACHE_TERMS = ("headache", "head pain", "migraine", "light hurts", "photophobia")


class ClarificationEngine:
    def build(
        self,
        user_text: str,
        symptoms: list[str],
        predictions: list[dict],
        risk_level: str = "LOW",
        answered_question_ids: list[str] | None = None,
    ) -> dict:
        if not predictions:
            return self._empty_payload()

        answered_set = set(answered_question_ids or [])
        if answered_set:
            return self._empty_payload()

        top_prediction = predictions[0]
        second_prediction = predictions[1] if len(predictions) > 1 else None
        top_condition = top_prediction["disease"]
        top_probability = float(top_prediction.get("probability", 0.0))
        second_probability = float(second_prediction.get("probability", 0.0)) if second_prediction else 0.0
        probability_margin = top_probability - second_probability
        matched_hallmarks = set(DISEASE_PROFILES.get(top_condition, {}).get("hallmark_symptoms", [])).intersection(
            symptoms
        )

        reason: str | None = None
        priority = "LOW"
        if not symptoms:
            reason = "no_symptoms"
            priority = "HIGH"
        elif top_condition == UNCERTAIN_CONDITION:
            reason = "uncertain"
            priority = "HIGH" if risk_level in {"MEDIUM", "HIGH"} else "MEDIUM"
        elif top_probability < CONFIDENCE_THRESHOLD or probability_margin < MARGIN_THRESHOLD:
            reason = "ambiguous"
            priority = "MEDIUM"
        elif len(matched_hallmarks) <= 1 and top_probability < 0.88:
            reason = "missing_hallmarks"
            priority = "LOW" if risk_level == "LOW" else "MEDIUM"

        if reason is None:
            return self._empty_payload()

        questions = self._collect_questions(
            reason=reason,
            user_text=user_text,
            symptoms=symptoms,
            predictions=predictions,
            answered_question_ids=answered_set,
        )
        if not questions:
            return self._empty_payload()

        summary, rationale = self._build_summary(
            reason=reason,
            top_prediction=top_prediction,
            second_prediction=second_prediction,
            symptoms=symptoms,
        )

        return {
            "should_clarify": True,
            "priority": priority,
            "reason": reason,
            "summary": summary,
            "rationale": rationale,
            "question_count": len(questions),
            "questions": questions,
            "suggested_reply": self._build_suggested_reply(questions),
        }

    @staticmethod
    def _empty_payload() -> dict:
        return {
            "should_clarify": False,
            "priority": "LOW",
            "reason": "",
            "summary": "",
            "rationale": "",
            "question_count": 0,
            "questions": [],
            "suggested_reply": "",
        }

    def _collect_questions(
        self,
        reason: str,
        user_text: str,
        symptoms: list[str],
        predictions: list[dict],
        answered_question_ids: set[str],
    ) -> list[dict]:
        top_condition = predictions[0]["disease"] if predictions else ""
        second_condition = predictions[1]["disease"] if len(predictions) > 1 else ""
        symptom_set = set(symptoms)

        questions: list[dict] = []
        if not symptoms:
            questions.extend(self._general_questions())
        else:
            if reason == "ambiguous":
                questions.extend(
                    self._questions_for_condition_pair(
                        user_text=user_text,
                        symptoms=symptoms,
                        top=top_condition,
                        second=second_condition,
                    )
                )
                questions.extend(self._questions_for_symptoms(user_text=user_text, symptoms=symptoms))
            else:
                questions.extend(self._questions_for_symptoms(user_text=user_text, symptoms=symptoms))
                questions.extend(
                    self._questions_for_condition_pair(
                        user_text=user_text,
                        symptoms=symptoms,
                        top=top_condition,
                        second=second_condition,
                    )
                )

            if reason in {"ambiguous", "missing_hallmarks"}:
                questions.extend(
                    self._questions_for_missing_hallmarks(
                        condition=top_condition,
                        symptom_set=symptom_set,
                        user_text=user_text,
                    )
                )
            elif reason == "uncertain":
                next_best = next(
                    (
                        item["disease"]
                        for item in predictions
                        if item["disease"] != UNCERTAIN_CONDITION
                    ),
                    "",
                )
                questions.extend(
                    self._questions_for_missing_hallmarks(
                        condition=next_best,
                        symptom_set=symptom_set,
                        user_text=user_text,
                    )
                )

        unique_questions: list[dict] = []
        seen_ids: set[str] = set()
        for question in questions:
            question_id = question["id"]
            if question_id in seen_ids or question_id in answered_question_ids:
                continue
            seen_ids.add(question_id)
            unique_questions.append(question)
            if len(unique_questions) == 3:
                break
        return unique_questions

    def _build_summary(
        self,
        reason: str,
        top_prediction: dict,
        second_prediction: dict | None,
        symptoms: list[str],
    ) -> tuple[str, str]:
        top_condition = top_prediction["disease"]
        top_probability = top_prediction["probability"] * 100
        second_condition = second_prediction["disease"] if second_prediction else ""
        second_probability = second_prediction["probability"] * 100 if second_prediction else 0.0

        if reason == "no_symptoms":
            return (
                "A few follow-up questions would help the assistant understand the symptom location and body system before ranking conditions.",
                "No structured symptom entities were extracted from the wording, so targeted follow-up details are needed for a safer match.",
            )
        if reason == "uncertain":
            return (
                "The current wording is too open-ended for a disease-specific highlight, so the assistant is asking for a few details first.",
                f"Current fallback: {top_condition} ({top_probability:.1f}%), which means the assistant is deliberately abstaining until the symptom picture is clearer.",
            )
        if reason == "ambiguous":
            return (
                "A small set of differentiating questions could separate the top condition from nearby alternatives more safely.",
                f"Current split: {top_condition} ({top_probability:.1f}%) versus {second_condition or 'other nearby patterns'} ({second_probability:.1f}%).",
            )

        symptom_context = ", ".join(symptoms[:3]) if symptoms else "the current wording"
        return (
            "The leading condition is plausible, but a few hallmark details are still missing.",
            f"The assistant already recognized {symptom_context}, yet it would benefit from one or two confirmatory symptoms before treating {top_condition} as the clearest match.",
        )

    def _questions_for_symptoms(self, user_text: str, symptoms: list[str]) -> list[dict]:
        lowered = user_text.lower()
        symptom_set = set(symptoms)
        questions: list[dict] = []

        if "headache" in symptom_set:
            questions.extend(
                [
                    self._headache_associated_features_question(),
                    self._headache_infection_question(),
                    self._headache_sinus_question(),
                ]
            )

        if "left upper abdominal pain" in symptom_set:
            questions.extend(
                [
                    self._trauma_question(),
                    self._yes_no_question(
                        question_id="luq_fever_vomiting",
                        question="Do you also have fever, vomiting, or worsening tenderness in that area?",
                        yes_text="I also have fever, vomiting, or worsening tenderness in that area.",
                        no_text="I do not have fever, vomiting, or worsening tenderness in that area.",
                        why="This helps separate a vague abdominal or spleen-area concern from a more urgent abdominal pattern.",
                    ),
                    self._yes_no_question(
                        question_id="luq_shoulder_breathing",
                        question="Is the pain worse with deep breathing or spreading toward the shoulder?",
                        yes_text="The pain is worse with deep breathing or it spreads toward the shoulder.",
                        no_text="The pain is not worse with deep breathing and it does not spread toward the shoulder.",
                        why="That pattern can change how urgently localized upper-abdominal pain should be assessed.",
                    ),
                ]
            )

        if "abdominal pain" in symptom_set and "right lower abdominal pain" not in symptom_set:
            questions.append(self._abdominal_location_question())

        if "back pain" in symptom_set and "blood in urine" not in symptom_set:
            questions.append(
                self._yes_no_question(
                    question_id="urine_blood_check",
                    question="Have you noticed blood, pink, or red urine?",
                    yes_text="I have noticed blood or pink or red urine.",
                    no_text="I have not noticed blood or pink or red urine.",
                    why="This helps separate musculoskeletal back pain from kidney-stone or urinary patterns.",
                )
            )

        if "back pain" in symptom_set and "stiffness" not in symptom_set:
            questions.append(self._strain_question())

        if "burning urination" in symptom_set or "frequent urination" in symptom_set:
            questions.append(
                self._yes_no_question(
                    question_id="urinary_fever_back",
                    question="Do you also have fever or flank or lower-back pain with the urinary symptoms?",
                    yes_text="I also have fever or flank or lower-back pain with the urinary symptoms.",
                    no_text="I do not have fever or flank or lower-back pain with the urinary symptoms.",
                    why="Those details help distinguish a simple urinary pattern from a more complicated urinary or kidney issue.",
                )
            )

        if self._has_respiratory_context(symptom_set, lowered):
            questions.append(
                self._yes_no_question(
                    question_id="resp_breathing",
                    question="Is there shortness of breath, chest tightness, or wheezing right now?",
                    yes_text="I have shortness of breath, chest tightness, or wheezing right now.",
                    no_text="I do not have shortness of breath, chest tightness, or wheezing right now.",
                    why="These symptoms sharply change risk and help separate allergy or viral illness from airway problems.",
                )
            )

        if "high blood sugar" in symptom_set or "frequent urination" in symptom_set:
            questions.append(
                self._yes_no_question(
                    question_id="glycemic_support",
                    question="Are you also unusually thirsty, fatigued, or noticing blurred vision?",
                    yes_text="I am also unusually thirsty, fatigued, or noticing blurred vision.",
                    no_text="I am not unusually thirsty, fatigued, or noticing blurred vision.",
                    why="That cluster makes a glucose-regulation concern more likely than a urinary-only issue.",
                )
            )

        return questions

    def _questions_for_condition_pair(
        self,
        user_text: str,
        symptoms: list[str],
        top: str,
        second: str,
    ) -> list[dict]:
        pair = {top, second}
        symptom_set = set(symptoms)
        lowered = user_text.lower()
        if {"Urinary Tract Infection", "Kidney Stone / Renal Colic"}.issubset(pair) and (
            self._has_urinary_context(symptom_set, lowered) or "back pain" in symptom_set
        ):
            return [
                self._yes_no_question(
                    question_id="uti_vs_stone_burning",
                    question="Does urination burn or sting?",
                    yes_text="Urination burns or stings.",
                    no_text="Urination does not burn or sting.",
                    why="Burning urination supports a urinary infection more than a kidney-stone pattern.",
                ),
                self._yes_no_question(
                    question_id="uti_vs_stone_blood",
                    question="Have you noticed blood, pink, or red urine?",
                    yes_text="I have noticed blood or pink or red urine.",
                    no_text="I have not noticed blood or pink or red urine.",
                    why="Blood in urine makes a kidney-stone pattern more likely.",
                ),
                self._yes_no_question(
                    question_id="uti_vs_stone_fever",
                    question="Do you have fever with the urinary symptoms?",
                    yes_text="I have fever with the urinary symptoms.",
                    no_text="I do not have fever with the urinary symptoms.",
                    why="Fever raises concern for a more significant urinary infection or complication.",
                ),
            ]

        respiratory_group = {
            "Influenza",
            "Bronchitis",
            "Pneumonia",
            "COVID-19",
            "Asthma Exacerbation",
            "Common Cold",
            "Allergic Rhinitis",
            "Sinusitis",
        }
        if len(pair.intersection(respiratory_group)) >= 2 and self._has_respiratory_context(symptom_set, lowered):
            return [
                self._yes_no_question(
                    question_id="resp_fever_chills",
                    question="Do you have fever or chills with the breathing or upper-respiratory symptoms?",
                    yes_text="I do have fever or chills with the breathing or upper-respiratory symptoms.",
                    no_text="I do not have fever or chills with the breathing or upper-respiratory symptoms.",
                    why="Fever pushes the assistant away from allergy-only explanations and toward infectious ones.",
                ),
                self._yes_no_question(
                    question_id="resp_wheeze",
                    question="Are you wheezing or feeling chest tightness?",
                    yes_text="I am wheezing or feeling chest tightness.",
                    no_text="I am not wheezing and I do not feel chest tightness.",
                    why="Wheezing and tightness increase concern for airway inflammation or asthma-like patterns.",
                ),
                self._yes_no_question(
                    question_id="resp_taste_smell",
                    question="Any loss of taste or smell?",
                    yes_text="I also have loss of taste or smell.",
                    no_text="I do not have loss of taste or smell.",
                    why="Loss of taste or smell helps separate COVID-like patterns from other respiratory illnesses.",
                ),
            ]

        abdominal_group = {
            "Gastroenteritis",
            "Gastritis / Acid Peptic Disorder",
            "Appendicitis Concern",
        }
        if len(pair.intersection(abdominal_group)) >= 2 and self._has_abdominal_context(symptom_set, lowered):
            return [
                self._abdominal_location_question(),
                self._yes_no_question(
                    question_id="abd_vomiting_diarrhea",
                    question="Is there vomiting or diarrhea with the abdominal symptoms?",
                    yes_text="I have vomiting or diarrhea with the abdominal symptoms.",
                    no_text="I do not have vomiting or diarrhea with the abdominal symptoms.",
                    why="Vomiting and diarrhea support a stomach or intestinal pattern more than a localized abdominal emergency.",
                ),
                self._yes_no_question(
                    question_id="abd_heartburn",
                    question="Do you notice heartburn, bloating, or pain related to eating?",
                    yes_text="I notice heartburn, bloating, or pain related to eating.",
                    no_text="I do not notice heartburn, bloating, or pain related to eating.",
                    why="Meal-related discomfort points more toward gastritis than appendicitis or gastroenteritis.",
                ),
            ]

        cold_allergy_group = {"Common Cold", "Allergic Rhinitis", "Sinusitis"}
        if len(pair.intersection(cold_allergy_group)) >= 2 and self._has_upper_airway_context(symptom_set, lowered):
            return [
                self._yes_no_question(
                    question_id="allergy_itchy_eyes",
                    question="Are itchy or watery eyes part of this pattern?",
                    yes_text="I have itchy or watery eyes.",
                    no_text="I do not have itchy or watery eyes.",
                    why="That detail is common in allergy patterns and less common in infection-driven sinus pressure.",
                ),
                self._yes_no_question(
                    question_id="sinus_facial_pressure",
                    question="Do you have facial pressure or pain around the cheeks or forehead?",
                    yes_text="I have facial pressure or pain around the cheeks or forehead.",
                    no_text="I do not have facial pressure or pain around the cheeks or forehead.",
                    why="Facial pressure makes sinusitis more likely than a simple cold or allergy flare.",
                ),
                self._yes_no_question(
                    question_id="cold_fever",
                    question="Have you had fever with these symptoms?",
                    yes_text="I have had fever with these symptoms.",
                    no_text="I have not had fever with these symptoms.",
                    why="Fever can shift the pattern toward infection rather than allergy-only symptoms.",
                ),
            ]

        return []

    def _questions_for_missing_hallmarks(
        self,
        condition: str,
        symptom_set: set[str],
        user_text: str,
    ) -> list[dict]:
        if not condition:
            return []
        hallmark_symptoms = DISEASE_PROFILES.get(condition, {}).get("hallmark_symptoms", [])
        questions: list[dict] = []
        lowered = user_text.lower()
        for symptom in hallmark_symptoms:
            if symptom in symptom_set:
                continue
            question = self._question_for_symptom(
                symptom=symptom,
                symptom_set=symptom_set,
                lowered=lowered,
            )
            if question:
                questions.append(question)
            if len(questions) == 2:
                break
        return questions

    def _question_for_symptom(
        self,
        symptom: str,
        symptom_set: set[str] | None = None,
        lowered: str = "",
    ) -> dict | None:
        symptom_set = symptom_set or set()
        if symptom in {"shortness of breath", "wheezing", "loss of smell", "loss of taste", "cough"} and not self._has_respiratory_context(symptom_set, lowered):
            return None
        if symptom in {"congestion", "runny nose", "sore throat"} and not self._has_upper_airway_context(symptom_set, lowered):
            return None
        if symptom in {"sensitivity to light", "dizziness"} and not self._has_headache_context(symptom_set, lowered):
            return None

        follow_up_map = {
            "fever": self._yes_no_question(
                question_id="confirm_fever",
                question="Have you had fever or a measured high temperature?",
                yes_text="I have had fever or a measured high temperature.",
                no_text="I have not had fever or a measured high temperature.",
                why="Fever often helps separate infection-related patterns from noninfectious ones.",
            ),
            "shortness of breath": self._yes_no_question(
                question_id="confirm_breathing",
                question="Are you short of breath or having trouble breathing?",
                yes_text="I am short of breath or having trouble breathing.",
                no_text="I am not short of breath and I am not having trouble breathing.",
                why="Breathing difficulty significantly changes both condition ranking and urgency.",
            ),
            "wheezing": self._yes_no_question(
                question_id="confirm_wheezing",
                question="Do you notice wheezing or a whistling sound while breathing?",
                yes_text="I notice wheezing or a whistling sound while breathing.",
                no_text="I do not notice wheezing or a whistling sound while breathing.",
                why="Wheezing can help separate airway-tightening patterns from other respiratory illnesses.",
            ),
            "burning urination": self._yes_no_question(
                question_id="confirm_burning_urination",
                question="Does urination burn or sting?",
                yes_text="Urination burns or stings.",
                no_text="Urination does not burn or sting.",
                why="That detail is important for separating a urinary infection from metabolic or kidney-stone patterns.",
            ),
            "blood in urine": self._yes_no_question(
                question_id="confirm_blood_urine",
                question="Have you noticed blood, pink, or red urine?",
                yes_text="I have noticed blood or pink or red urine.",
                no_text="I have not noticed blood or pink or red urine.",
                why="Blood in urine shifts the ranking toward stone or complicated urinary patterns.",
            ),
            "back pain": self._yes_no_question(
                question_id="confirm_back_pain",
                question="Is there lower-back, side, or flank pain with this issue?",
                yes_text="I have lower-back, side, or flank pain with this issue.",
                no_text="I do not have lower-back, side, or flank pain with this issue.",
                why="Back or flank pain helps separate urinary, kidney, and musculoskeletal patterns.",
            ),
            "excessive thirst": self._yes_no_question(
                question_id="confirm_thirst",
                question="Are you unusually thirsty or drinking much more than normal?",
                yes_text="I am unusually thirsty or drinking much more than normal.",
                no_text="I am not unusually thirsty or drinking much more than normal.",
                why="This strongly supports a glucose-regulation concern when paired with urinary symptoms.",
            ),
            "blurred vision": self._yes_no_question(
                question_id="confirm_blurred_vision",
                question="Have you noticed blurred or unusually cloudy vision?",
                yes_text="I have noticed blurred or unusually cloudy vision.",
                no_text="I have not noticed blurred or unusually cloudy vision.",
                why="Blurred vision can help confirm a metabolic pattern like high blood sugar.",
            ),
            "right lower abdominal pain": self._abdominal_location_question(),
            "vomiting": self._yes_no_question(
                question_id="confirm_vomiting",
                question="Are you vomiting or struggling to keep fluids down?",
                yes_text="I am vomiting or struggling to keep fluids down.",
                no_text="I am not vomiting and I can keep fluids down.",
                why="Vomiting changes the urgency and helps separate digestive patterns.",
            ),
            "diarrhea": self._yes_no_question(
                question_id="confirm_diarrhea",
                question="Do you also have diarrhea or loose stools?",
                yes_text="I also have diarrhea or loose stools.",
                no_text="I do not have diarrhea or loose stools.",
                why="This helps distinguish intestinal illness from other causes of nausea or abdominal pain.",
            ),
            "heartburn": self._yes_no_question(
                question_id="confirm_heartburn",
                question="Do you notice heartburn or burning discomfort after eating?",
                yes_text="I notice heartburn or burning discomfort after eating.",
                no_text="I do not notice heartburn or burning discomfort after eating.",
                why="Meal-related burning points more toward acid irritation patterns.",
            ),
            "facial pain": self._yes_no_question(
                question_id="confirm_facial_pain",
                question="Is there facial pressure or pain around the cheeks, eyes, or forehead?",
                yes_text="There is facial pressure or pain around the cheeks, eyes, or forehead.",
                no_text="There is no facial pressure or pain around the cheeks, eyes, or forehead.",
                why="Facial pressure helps separate sinusitis from a simple cold or allergy flare.",
            ),
            "loss of smell": self._yes_no_question(
                question_id="confirm_loss_smell",
                question="Have you lost smell or taste?",
                yes_text="I have lost smell or taste.",
                no_text="I have not lost smell or taste.",
                why="That symptom can separate COVID-like patterns from many other respiratory conditions.",
            ),
            "swollen glands": self._yes_no_question(
                question_id="confirm_swollen_glands",
                question="Do you have swollen glands or tender lumps in the neck?",
                yes_text="I have swollen glands or tender lumps in the neck.",
                no_text="I do not have swollen glands or tender lumps in the neck.",
                why="Swollen glands support mono-like throat and fatigue patterns.",
            ),
            "stiffness": self._strain_question(),
        }
        return follow_up_map.get(symptom)

    @staticmethod
    def _headache_associated_features_question() -> dict:
        return {
            "id": "headache_features",
            "question": "Which of these headache-related features is closest to what you are feeling?",
            "why": "These details help separate migraine-like headaches from infection-related or sinus-related patterns.",
            "options": [
                {
                    "label": "Nausea or vomiting",
                    "append_text": "I also have nausea or vomiting with the headache.",
                },
                {
                    "label": "Light sensitivity or dizziness",
                    "append_text": "I also have sensitivity to light or dizziness with the headache.",
                },
                {
                    "label": "Both of those",
                    "append_text": "I also have nausea, sensitivity to light, or dizziness with the headache.",
                },
                {
                    "label": "None of these",
                    "append_text": "I do not have nausea, vomiting, sensitivity to light, or dizziness with the headache.",
                },
            ],
        }

    @staticmethod
    def _headache_infection_question() -> dict:
        return {
            "id": "headache_infection_pattern",
            "question": "Do you also have infection-like symptoms with the headache?",
            "why": "Fever, body aches, or rash can shift the pattern away from a migraine-only explanation.",
            "options": [
                {
                    "label": "Fever or body aches",
                    "append_text": "I also have fever or body ache with the headache.",
                },
                {
                    "label": "Rash",
                    "append_text": "I also have a rash with the headache.",
                },
                {
                    "label": "Both",
                    "append_text": "I also have fever, body ache, or rash with the headache.",
                },
                {
                    "label": "None of these",
                    "append_text": "I do not have fever, body ache, or rash with the headache.",
                },
            ],
        }

    @staticmethod
    def _headache_sinus_question() -> dict:
        return {
            "id": "headache_sinus_pattern",
            "question": "Is the headache linked to sinus or ear symptoms?",
            "why": "Facial pressure, congestion, ear pain, or sore throat can point toward sinus or ear-related causes.",
            "options": [
                {
                    "label": "Congestion or facial pressure",
                    "append_text": "I also have congestion or facial pain with the headache.",
                },
                {
                    "label": "Ear pain or sore throat",
                    "append_text": "I also have ear pain or sore throat with the headache.",
                },
                {
                    "label": "None of these",
                    "append_text": "I do not have congestion, facial pain, ear pain, or sore throat with the headache.",
                },
            ],
        }

    @staticmethod
    def _has_respiratory_context(symptom_set: set[str], lowered: str) -> bool:
        return bool(RESPIRATORY_CONTEXT_SYMPTOMS.intersection(symptom_set)) or any(
            term in lowered for term in RESPIRATORY_TERMS
        )

    @staticmethod
    def _has_upper_airway_context(symptom_set: set[str], lowered: str) -> bool:
        return bool(UPPER_AIRWAY_CONTEXT_SYMPTOMS.intersection(symptom_set)) or any(
            term in lowered for term in UPPER_AIRWAY_TERMS
        )

    @staticmethod
    def _has_abdominal_context(symptom_set: set[str], lowered: str) -> bool:
        return bool(ABDOMINAL_CONTEXT_SYMPTOMS.intersection(symptom_set)) or any(
            term in lowered for term in ABDOMINAL_TERMS
        )

    @staticmethod
    def _has_urinary_context(symptom_set: set[str], lowered: str) -> bool:
        return bool(URINARY_CONTEXT_SYMPTOMS.intersection(symptom_set)) or any(
            term in lowered for term in URINARY_TERMS
        )

    @staticmethod
    def _has_headache_context(symptom_set: set[str], lowered: str) -> bool:
        return bool(HEADACHE_CONTEXT_SYMPTOMS.intersection(symptom_set)) or any(
            term in lowered for term in HEADACHE_TERMS
        )

    @staticmethod
    def _general_questions() -> list[dict]:
        return [
            {
                "id": "general_location",
                "question": "Where is the symptom most clearly located?",
                "why": "Location is the fastest way to narrow the body system before ranking conditions.",
                "options": [
                    {
                        "label": "Chest or breathing",
                        "append_text": "The symptom is mainly in my chest or breathing.",
                    },
                    {
                        "label": "Abdomen or side",
                        "append_text": "The symptom is mainly in my abdomen or side.",
                    },
                    {
                        "label": "Back or muscles",
                        "append_text": "The symptom is mainly in my back or muscles.",
                    },
                    {
                        "label": "Urinary or bathroom related",
                        "append_text": "The symptom is mainly urinary or bathroom related.",
                    },
                ],
            },
            ClarificationEngine._yes_no_question(
                question_id="general_fever",
                question="Have you had fever or chills with this problem?",
                yes_text="I have had fever or chills with this problem.",
                no_text="I have not had fever or chills with this problem.",
                why="Fever changes both urgency and the likely type of illness.",
            ),
            ClarificationEngine._yes_no_question(
                question_id="general_redflags",
                question="Is there any trouble breathing, repeated vomiting, or blood in urine or stool?",
                yes_text="There is trouble breathing, repeated vomiting, or blood in urine or stool.",
                no_text="There is no trouble breathing, repeated vomiting, or blood in urine or stool.",
                why="These red-flag details affect safety guidance immediately.",
            ),
        ]

    @staticmethod
    def _abdominal_location_question() -> dict:
        return {
            "id": "abd_location",
            "question": "Where is the abdominal pain strongest?",
            "why": "Pain location helps separate stomach upset from appendix, kidney, or upper-abdominal patterns.",
            "options": [
                {
                    "label": "Lower right abdomen",
                    "append_text": "The abdominal pain is strongest in the lower right abdomen.",
                },
                {
                    "label": "Upper abdomen or under the ribs",
                    "append_text": "The abdominal pain is strongest in the upper abdomen or under the ribs.",
                },
                {
                    "label": "Side or flank",
                    "append_text": "The pain is strongest on the side or flank.",
                },
                {
                    "label": "Diffuse or hard to localize",
                    "append_text": "The abdominal pain is diffuse or hard to localize.",
                },
            ],
        }

    @staticmethod
    def _strain_question() -> dict:
        return {
            "id": "strain_trigger",
            "question": "Did the pain start after lifting, twisting, exercise, or physical strain?",
            "why": "A clear strain trigger supports musculoskeletal pain more than urinary or abdominal causes.",
            "options": [
                {
                    "label": "Yes, after lifting or strain",
                    "append_text": "The pain started after lifting, twisting, exercise, or physical strain.",
                },
                {
                    "label": "No clear strain trigger",
                    "append_text": "The pain did not start after lifting, twisting, exercise, or physical strain.",
                },
                {
                    "label": "Not sure",
                    "append_text": "I am unsure whether there was a strain trigger.",
                },
            ],
        }

    @staticmethod
    def _trauma_question() -> dict:
        return {
            "id": "trauma_trigger",
            "question": "Did this start after a fall, sports hit, or other injury to the area?",
            "why": "Recent trauma changes how localized spleen-area or upper-abdominal pain should be interpreted.",
            "options": [
                {
                    "label": "Yes, after an injury",
                    "append_text": "This started after a fall, sports hit, or other injury to the area.",
                },
                {
                    "label": "No injury",
                    "append_text": "This did not start after a fall, sports hit, or other injury to the area.",
                },
                {
                    "label": "Not sure",
                    "append_text": "I am unsure whether there was an injury trigger.",
                },
            ],
        }

    @staticmethod
    def _yes_no_question(
        question_id: str,
        question: str,
        yes_text: str,
        no_text: str,
        why: str,
    ) -> dict:
        return {
            "id": question_id,
            "question": question,
            "why": why,
            "options": [
                {
                    "label": "Yes",
                    "append_text": yes_text,
                },
                {
                    "label": "No",
                    "append_text": no_text,
                },
                {
                    "label": "Not sure",
                    "append_text": "I am unsure about that follow-up detail.",
                },
            ],
        }

    @staticmethod
    def _build_suggested_reply(questions: list[dict]) -> str:
        prompts = []
        for question in questions:
            first_option = question.get("options", [{}])[0]
            label = first_option.get("append_text")
            if label:
                prompts.append(label)
        return "Example follow-up reply: " + " ".join(prompts[:2])
