# Public Health Advisory Assistant

Public Health Advisory Assistant is a fully local healthcare advisory system built without any external LLM APIs. It combines symptom extraction, a supervised disease classifier, FAISS-backed retrieval, and rule-based safety logic to produce explainable health guidance that is suitable for an AI/ML internship demo or final-year capstone.

## 1. Project Folder Structure

```text
Public Health Advisory Assistant/
|-- app/
|   |-- api/
|   |-- core/
|   |-- ml/
|   |-- nlp/
|   |-- rag/
|   `-- services/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- artifacts/
|-- frontend/
|   `-- streamlit_app.py
|-- scripts/
|   |-- bootstrap_project.py
|   |-- build_vector_store.py
|   |-- evaluate.py
|   |-- generate_dataset.py
|   `-- train_model.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## 2. Dataset Explanation

### Symptom-disease dataset

- `data/raw/symptom_disease_seed.csv` contains curated seed mappings for 19 supported health-condition patterns plus an explicit uncertainty class.
- Covered patterns now include respiratory, gastrointestinal, urinary, metabolic, sinus, asthma, abdominal-pain, ear-infection, mono-like, kidney-stone, and musculoskeletal back-strain scenarios.
- `scripts/generate_dataset.py` expands the seed set into a larger training dataset using symptom combinations, duration phrases, and natural-language templates, including more colloquial phrasing such as blood in urine, side pain, lifting-related back pain, and organ-location uncertainty.
- The generated file is saved to `data/processed/symptom_disease_dataset.csv`.

### Medical knowledge corpus

- `data/raw/medical_corpus.jsonl` stores a simulated WHO/CDC-style document collection.
- Each document includes:
  - `title`
  - `source`
  - `disease`
  - `tags`
  - `content`
  - `precautions`
  - `escalation`
- This corpus is indexed for retrieval and used as evidence during response generation.

## 3. ML Model Code

The classifier lives in `app/ml/train.py`.

### Approach

- Text input: patient symptom description plus extracted symptom list.
- Feature engineering: `TfidfVectorizer` with uni-grams and bi-grams.
- Model: `LogisticRegression` multi-class classifier.
- Outputs:
  - disease probabilities
  - top-k ranked conditions
  - training metrics

### Metrics captured

- Accuracy
- Macro F1
- Top-3 accuracy
- Detailed per-class report

Artifacts are saved to:

- `data/artifacts/models/disease_classifier.joblib`
- `data/artifacts/models/training_metrics.json`

## 4. RAG Pipeline Code

The RAG flow is split across:

- `app/rag/embeddings.py`
- `app/rag/indexer.py`
- `app/services/retrieval_service.py`

### Pipeline

1. Read the corpus from `medical_corpus.jsonl`
2. Build document embeddings using `SentenceTransformer`
3. Store vectors in FAISS
4. Retrieve top-k documents for a symptom query
5. Feed retrieved evidence into the template-based response generator

### Local-only generation

No external LLM API is used. The final response is built from:

- extracted symptoms
- top disease predictions
- retrieved medical evidence
- rule-based risk logic
- hand-crafted response templates

The retrieval corpus also includes disease-specific explainers for newer patterns such as hyperglycemia concern, kidney-stone style symptoms, appendicitis concern, musculoskeletal back strain, and uncertainty-first clinical evaluation.

If `SentenceTransformer` is unavailable in the environment, the code falls back to TF-IDF embeddings so the project remains runnable offline. Set `PHAA_ALLOW_MODEL_DOWNLOAD=true` only if you want the app to attempt downloading the embedding model automatically.

## 5. Backend APIs

The FastAPI app lives in `app/api/main.py`.

### Endpoints

- `GET /health`
  - Service readiness check
- `POST /predict`
  - Extract symptoms and return top disease probabilities
- `POST /retrieve`
  - Run semantic retrieval against the medical corpus
- `POST /analyze`
  - Full pipeline: NLP + model + retrieval + risk + advisory
- `POST /report`
  - Full analysis plus persistent JSON report generation

### Example request

```json
{
  "text": "I have fever, cough and fatigue for 3 days",
  "top_k": 3
}
```

## 6. Frontend UI

The frontend is implemented in `frontend/streamlit_app.py`.

### Features

- chat-style symptom input
- cinematic glassmorphism dashboard with animated 3D hero visuals
- color-coded risk level badge
- quick-launch scenario cards for demo flows
- clarification-question flow that asks 1-3 targeted follow-up questions when the symptom pattern is vague, uncertain, or split across similar conditions
- probability chart for predicted conditions
- retrieved evidence display
- downloadable JSON health report
- backend-first execution with local pipeline fallback
- uncertainty fallback for unsupported or weakly matched symptom patterns

## 7. Setup and Run Instructions

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Build the dataset, train the model, and create the vector store

```powershell
python scripts/bootstrap_project.py
```

### Step 4: Run the backend

```powershell
python -m uvicorn app.api.main:app --reload
```

### Step 5: Run the frontend

```powershell
python -m streamlit run frontend/streamlit_app.py
```

## Evaluation

Run the evaluation script:

```powershell
python scripts/evaluate.py
```

This script reports:

- classifier training metrics
- retrieval hit rate at top-3
- one sample prediction output

## Safety, Ethics, and Production Notes

- The assistant never claims to provide an exact diagnosis.
- It includes a medical disclaimer in every advisory response.
- Risk stratification is conservative for chest pain, breathing issues, persistent fever, dehydration, and back pain with urinary symptoms.
- Symptom extraction now includes alias normalization plus lightweight heuristics for phrasing such as `blood in my urine`, `sharp side pain`, `my lower back hurts after lifting`, and spleen-region descriptions.
- When confidence is weak or the wording is incomplete, the assistant now returns targeted clarification questions with ready-to-use answer options instead of forcing a premature condition match.
- Unsupported symptom phrases, unusual anatomy-related pain, or low-confidence model outputs fall back to a cautious clinical-evaluation response instead of forcing a disease label.
- All logic is local and explainable, making the system suitable for audit and extension.

## Suggested Demo Flow

1. Launch the FastAPI backend.
2. Launch the Streamlit dashboard.
3. Enter a symptom query such as chest pain plus shortness of breath.
4. Show the high-risk output, retrieved evidence, and JSON report export.
5. Compare it with a mild allergy or common cold query to highlight risk differentiation.
