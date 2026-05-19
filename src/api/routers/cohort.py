"""
GET /cohort?date=YYYY-MM-DD — all predictions for a scoring date.

A date with no predictions returns 200 with an empty list (not 404).
A missing or malformed date returns 422 (FastAPI regex validation).
A logically invalid date (e.g. 2026-13-99) returns 400 via the ValueError handler.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_prediction_repository
from ..exceptions import ErrorResponse
from ..repositories.base import AbstractPredictionRepository
from ..schemas import CohortResponse

router = APIRouter()


@router.get(
    "/cohort",
    response_model=CohortResponse,
    summary="Get all predictions for a scoring date",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid date value (e.g. 2026-13-99)"},
        422: {"model": ErrorResponse, "description": "Missing or malformed date parameter"},
        503: {"model": ErrorResponse, "description": "Prediction store unavailable"},
    },
    tags=["predictions"],
)
async def get_cohort(
    date_str: str = Query(
        ...,
        alias="date",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Scoring date in YYYY-MM-DD format",
        openapi_examples={
            "march_2017": {
                "summary": "March 2017 cohort",
                "value": "2017-03-01",
            }
        },
    ),
    repo: AbstractPredictionRepository = Depends(get_prediction_repository),
) -> CohortResponse:
    # Regex above ensures the format is correct; fromisoformat validates semantics.
    try:
        scoring_date = date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(
            f"Query parameter 'date' must be a valid calendar date in YYYY-MM-DD format. "
            f"Got: '{date_str}'"
        )
    return repo.get_cohort_for_date(scoring_date)
