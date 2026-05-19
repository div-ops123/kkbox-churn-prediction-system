"""
GET /score/{user_id} — latest churn prediction for a single user.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path

from ..dependencies import get_prediction_repository
from ..exceptions import ErrorResponse, UserNotFoundError
from ..repositories.base import AbstractPredictionRepository
from ..schemas import PredictionRecord

router = APIRouter()


@router.get(
    "/score/{user_id}",
    response_model=PredictionRecord,
    summary="Get latest churn prediction for a user",
    responses={
        404: {"model": ErrorResponse, "description": "No prediction found for this user"},
        422: {"model": ErrorResponse, "description": "Invalid user_id format"},
        503: {"model": ErrorResponse, "description": "Prediction store unavailable"},
    },
    tags=["predictions"],
)
async def get_user_score(
    user_id: str = Path(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-+/=]{1,64}$",
        description="User MSNO identifier",
        openapi_examples={
            "typical": {
                "summary": "Base64-encoded MSNO",
                "value": "Qzj4T3SAdgb7ow0JGJohhA==",
            }
        },
    ),
    repo: AbstractPredictionRepository = Depends(get_prediction_repository),
) -> PredictionRecord:
    result = repo.get_latest_for_user(user_id)
    if result is None:
        raise UserNotFoundError(user_id)
    return result
