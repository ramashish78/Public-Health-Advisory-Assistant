# Public Health Advisory Assistant

Local AI/ML healthcare advisory assistant using NLP, classification, FAISS retrieval, and rule-based risk assessment[cite: 2].

## 🛠 Tech Stack

*   **API Framework:** FastAPI and Uvicorn[cite: 1, 2].
*   **Frontend User Interface:** Streamlit[cite: 1].
*   **Machine Learning & Data Processing:** Scikit-learn, Pandas, Numpy, and Joblib[cite: 1].
*   **NLP & Vector Search (RAG):** SpaCy, Sentence-Transformers, and FAISS (CPU)[cite: 1].

## 🏗 Architecture & Core Pipeline

The core logic is orchestrated by the `AdvisoryPipeline`, which coordinates multiple micro-services to process text and generate medical insights[cite: 3]:

*   **SymptomExtractor:** Extracts and normalizes symptoms from raw user text[cite: 3].
*   **ModelService:** Predicts potential conditions using extracted symptoms and normalized text[cite: 3].
*   **RetrievalService:** Queries the FAISS vector store to retrieve relevant medical documentation[cite: 2, 3].
*   **RiskEngine:** Assesses risk levels and determines recommended actions or escalations[cite: 3].
*   **ClarificationEngine & AdvisoryGenerator:** Builds tailored questions for the user and formulates the final advice based on predictions and risk[cite: 3].

## 📡 API Endpoints

The FastAPI backend exposes the following REST endpoints[cite: 2]:

*   `GET /health`: Returns system status and verifies if the model artifact and vector store are loaded and ready[cite: 2].
*   `POST /predict`: Accepts a text payload and returns disease predictions and extracted symptoms[cite: 2, 3].
*   `POST /retrieve`: Accepts a text payload and retrieves the most relevant context documents[cite: 2, 3].
*   `POST /analyze`: Runs the end-to-end pipeline, returning predictions, risk assessment, primary conditions, and clarification questions[cite: 2, 3].
*   `POST /report`: Generates a structured summary report of the analysis[cite: 2, 3].

## ⚙️ Local Setup & Environment Variables

To run this project locally, create a `.env` file in the root directory. Use the following configuration based on the `.env.example` file[cite: 4]:

```env
PHAA_HOST=0.0.0.0
PHAA_PORT=8000
PHAA_EMBEDDING_MODEL=all-MiniLM-L6-v2
PHAA_ALLOW_MODEL_DOWNLOAD=false
PHAA_TOP_K=3
PHAA_REPORT_RETENTION_DAYS=30
