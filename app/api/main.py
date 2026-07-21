from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import AnalyzeResponse, PredictResponse, ReportResponse, RetrieveResponse, TextRequest
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.pipeline import AdvisoryPipeline

configure_logging()
settings.ensure_directories()

app = FastAPI(
    title="Public Health Advisory Assistant",
    version="1.0.0",
    description="Local AI/ML healthcare advisory assistant using NLP, classification, FAISS retrieval, and rule-based risk assessment.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_pipeline() -> AdvisoryPipeline:
    return AdvisoryPipeline()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": settings.model_artifact_path.exists(),
        "vector_store_ready": settings.retrieval_metadata_path.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: TextRequest) -> dict:
    try:
        return get_pipeline().predict(request.text, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: TextRequest) -> dict:
    try:
        return get_pipeline().retrieve(request.text, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: TextRequest) -> dict:
    try:
        return get_pipeline().analyze(
            request.text,
            top_k=request.top_k,
            answered_question_ids=request.answered_question_ids,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/report", response_model=ReportResponse)
def report(request: TextRequest) -> dict:
    try:
        return get_pipeline().report(
            request.text,
            top_k=request.top_k,
            answered_question_ids=request.answered_question_ids,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
