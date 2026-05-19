"""
FastAPI dependency providers — Dependency Injection layer.

get_prediction_repository is overridden in tests with an InMemoryRepository
via app.dependency_overrides, enabling full router testing without a real DB.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import psycopg2.pool
from fastapi import Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ApiConfig, load_api_config

from .repositories.base import AbstractPredictionRepository
from .repositories.postgres import PostgresPredictionRepository


@lru_cache(maxsize=1)
def get_api_config() -> ApiConfig:
    """Read env vars once at startup; cached for the process lifetime."""
    return load_api_config()


def get_connection_pool(request: Request) -> psycopg2.pool.ThreadedConnectionPool:
    """
    Retrieve the application-level connection pool from app.state.
    The pool is created in the FastAPI lifespan (main.py) — this dependency
    only fetches it; it never creates a new pool.
    """
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise RuntimeError(
            "Connection pool is not initialised. "
            "Did the FastAPI lifespan context manager run?"
        )
    return pool


def get_prediction_repository(request: Request) -> AbstractPredictionRepository:
    """
    Create a PostgresPredictionRepository per request using the shared pool.
    Override this in tests:
        app.dependency_overrides[get_prediction_repository] = lambda: InMemoryRepo()
    """
    pool = get_connection_pool(request)
    return PostgresPredictionRepository(pool)
