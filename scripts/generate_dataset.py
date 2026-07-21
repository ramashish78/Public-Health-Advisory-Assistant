from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.core.constants import UNCERTAIN_CONDITION


TEMPLATES = [
    "I have {symptoms} for {duration}.",
    "For the last {duration}, I have been dealing with {symptoms}.",
    "My main problems are {symptoms} and it started {duration} ago.",
    "Patient reports {symptoms} since {duration}.",
    "I have been noticing {symptoms} over the past {duration}.",
    "These symptoms started {duration} ago: {symptoms}.",
    "Since {duration}, I have had {symptoms}.",
    "Over the last {duration}, I am experiencing {symptoms}.",
]
METABOLIC_TEMPLATES = [
    "My blood sugar level has been high with {symptoms} for {duration}.",
    "For the last {duration}, my glucose has been high and I also have {symptoms}.",
    "I have high blood sugar along with {symptoms} for {duration}.",
]
ABDOMINAL_TEMPLATES = [
    "I have {symptoms} and the discomfort has been there for {duration}.",
    "The pain has been {symptoms} for {duration}.",
]
RENAL_TEMPLATES = [
    "There is blood in my urine with sharp side pain for {duration}.",
    "I have flank pain and blood in my urine for {duration}.",
    "For {duration}, I have had pain on my side with blood in the urine and nausea.",
]
BACK_STRAIN_TEMPLATES = [
    "My lower back hurts after lifting something heavy for {duration}.",
    "I strained my back and now have lower back pain with stiffness for {duration}.",
    "For {duration}, my back has been aching after physical strain.",
]
UNCERTAIN_TEMPLATES = [
    "I am feeling pain in back of the spleen from past {duration}.",
    "There is pain near my spleen area for {duration}.",
    "I have left upper abdominal discomfort under my ribs for {duration}.",
    "I feel unusual pain around the left upper side of my abdomen for {duration}.",
    "My symptoms are hard to describe, mostly pain around my spleen region for {duration}.",
    "I have localized pain under the left ribs for {duration}.",
]

DURATIONS = ["1 day", "2 days", "3 days", "4 days", "5 days", "1 week"]
NOISE_SYMPTOMS = [
    "fatigue",
    "headache",
    "dizziness",
    "chills",
    "congestion",
    "blurred vision",
    "excessive thirst",
    "bloating",
    "heartburn",
    "stiffness",
]


def build_patient_text(chosen_symptoms: list[str], duration: str, randomizer: random.Random) -> str:
    symptom_text = (
        ", ".join(chosen_symptoms[:-1]) + f" and {chosen_symptoms[-1]}"
        if len(chosen_symptoms) > 1
        else chosen_symptoms[0]
    )

    if "high blood sugar" in chosen_symptoms:
        secondary = [item for item in chosen_symptoms if item != "high blood sugar"]
        if secondary:
            secondary_text = (
                ", ".join(secondary[:-1]) + f" and {secondary[-1]}"
                if len(secondary) > 1
                else secondary[0]
            )
            template = randomizer.choice(METABOLIC_TEMPLATES)
            return template.format(symptoms=secondary_text, duration=duration)

    if "right lower abdominal pain" in chosen_symptoms or "left upper abdominal pain" in chosen_symptoms:
        return randomizer.choice(ABDOMINAL_TEMPLATES).format(symptoms=symptom_text, duration=duration)

    if {"back pain", "blood in urine"}.issubset(chosen_symptoms):
        return randomizer.choice(RENAL_TEMPLATES).format(duration=duration)

    if {"back pain", "stiffness"}.issubset(chosen_symptoms):
        return randomizer.choice(BACK_STRAIN_TEMPLATES).format(duration=duration)

    return randomizer.choice(TEMPLATES).format(symptoms=symptom_text, duration=duration)


def build_uncertain_records(randomizer: random.Random, rows: int) -> list[dict]:
    records: list[dict] = []
    for _ in range(rows):
        duration = randomizer.choice(DURATIONS)
        patient_text = randomizer.choice(UNCERTAIN_TEMPLATES).format(duration=duration)
        records.append(
            {
                "patient_text": patient_text,
                "symptoms": "left upper abdominal pain",
                "training_text": f"{patient_text} Symptoms: left upper abdominal pain",
                "disease": UNCERTAIN_CONDITION,
                "risk_hint": "MEDIUM",
                "category": "Uncertain",
            }
        )
    return records


def generate_dataset(rows_per_disease: int = 180, seed: int = 42) -> pd.DataFrame:
    settings.ensure_directories()
    randomizer = random.Random(seed)
    seed_df = pd.read_csv(settings.seed_dataset_path)
    records: list[dict] = []

    for _, row in seed_df.iterrows():
        symptoms = row["symptoms"].split("|")
        for _ in range(rows_per_disease):
            chosen_count = randomizer.randint(max(2, len(symptoms) // 2), len(symptoms))
            chosen_symptoms = randomizer.sample(symptoms, chosen_count)
            if randomizer.random() < 0.35:
                extra = randomizer.choice([item for item in NOISE_SYMPTOMS if item not in chosen_symptoms])
                chosen_symptoms.append(extra)

            duration = randomizer.choice(DURATIONS)
            patient_text = build_patient_text(chosen_symptoms, duration, randomizer)
            training_text = f"{patient_text} Symptoms: {', '.join(chosen_symptoms)}"

            records.append(
                {
                    "patient_text": patient_text,
                    "symptoms": "|".join(chosen_symptoms),
                    "training_text": training_text,
                    "disease": row["disease"],
                    "risk_hint": row["risk_hint"],
                    "category": row["category"],
                }
            )

    records.extend(build_uncertain_records(randomizer, rows=max(240, rows_per_disease * 2)))

    dataframe = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dataframe.to_csv(settings.training_dataset_path, index=False)
    return dataframe


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated dataset with {len(df)} rows at {settings.training_dataset_path}")
