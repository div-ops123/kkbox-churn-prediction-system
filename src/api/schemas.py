"""
Pydantic request/response schemas for the Score API.

All responses use these models — no raw dicts are returned by routers.
ErrorResponse is the uniform envelope for every 4xx/5xx response.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRecord(BaseModel):
    """Response body for GET /score/{user_id}."""

    user_id: str = Field(..., description="MSNO user identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Churn probability [0, 1]")
    risk_tier: Literal["HIGH", "MED", "LOW"]
    scoring_date: date
    expiry_date: date
    model_version: str = Field(..., description="MLflow run_id of the scoring model")
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "Qzj4T3SAdgb7ow0JGJohhA==",
                "score": 0.742,
                "risk_tier": "HIGH",
                "scoring_date": "2026-05-19",
                "expiry_date": "2026-06-01",
                "model_version": "abc123run_id",
                "created_at": "2026-05-19T02:14:33Z",
            }
        }
    }


class CohortItem(BaseModel):
    """One prediction row inside CohortResponse. Omits created_at for payload efficiency."""

    user_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal["HIGH", "MED", "LOW"]
    expiry_date: date
    model_version: str


class CohortResponse(BaseModel):
    """Response body for GET /cohort."""

    scoring_date: date
    count: int = Field(..., ge=0)
    predictions: list[CohortItem]


class ErrorResponse(BaseModel):
    """Uniform error envelope for all 4xx and 5xx responses."""

    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable description")
    field: str | None = Field(None, description="Which field caused the error, if applicable")
    retry_after_seconds: int | None = Field(None, description="Set for 503 responses only")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok", "degraded"]
    version: str
    database: Literal["connected", "unreachable"]
    timestamp: datetime
