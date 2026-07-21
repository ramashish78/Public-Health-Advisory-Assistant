from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="User symptom description in natural language.")
    top_k: int = Field(3, ge=1, le=5)
    answered_question_ids: list[str] = Field(default_factory=list, max_length=10)


class PredictResponse(BaseModel):
    normalized_text: str
    symptoms: list[str]
    matched_phrases: list[str]
    predictions: list[dict[str, Any]]
    rule_notes: list[str]


class RetrieveResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


class ClarificationOption(BaseModel):
    label: str
    append_text: str


class ClarificationQuestion(BaseModel):
    id: str
    question: str
    why: str
    options: list[ClarificationOption]


class ClarificationPayload(BaseModel):
    should_clarify: bool
    priority: str
    reason: str
    summary: str
    rationale: str
    question_count: int
    questions: list[ClarificationQuestion]
    suggested_reply: str


class AnalyzeResponse(BaseModel):
    input_text: str
    normalized_text: str
    symptoms: list[str]
    matched_phrases: list[str]
    predictions: list[dict[str, Any]]
    primary_condition: dict[str, Any] | None
    clarification: ClarificationPayload
    decision_support: dict[str, Any]
    retrieval_results: list[dict[str, Any]]
    risk: dict[str, Any]
    advice: dict[str, Any]


class ReportResponse(AnalyzeResponse):
    report_id: str
    created_at: str
