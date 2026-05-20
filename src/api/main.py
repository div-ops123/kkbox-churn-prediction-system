"""
FastAPI application factory — Score API entry point.

Start the server:
    uvicorn src.api.main:app --reload --port 8000

OpenAPI docs (auto-generated):
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

# Must be before relative imports: they chain into dependencies.py which also
# needs config.  parent.parent resolves to src/ where config.py lives.
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import load_api_config  # noqa: E402

import psycopg2.pool
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .exceptions import register_exception_handlers
from .routers import cohort, score
from .schemas import HealthResponse

logger = logging.getLogger(__name__)

_API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Create the psycopg2 connection pool on startup; close it on shutdown.
    The pool lives on app.state.pool for the process lifetime.
    """
    config = load_api_config()
    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))

    app.state.pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=config.pool_min_conn,
        maxconn=config.pool_max_conn,
        dsn=config.postgres.dsn,
    )
    logger.info(
        "API startup complete — pool initialised (min=%d max=%d)",
        config.pool_min_conn,
        config.pool_max_conn,
    )

    yield

    app.state.pool.closeall()
    logger.info("API shutdown complete — pool closed")


def create_app() -> FastAPI:
    """
    Application factory. Called by Uvicorn and by test fixtures.
    Tests can call create_app() and override dependencies independently.
    """
    app = FastAPI(
        title="KKBox Churn Score API",
        version=_API_VERSION,
        description=(
            "Read-only API for querying pre-computed churn predictions. "
            "Does **not** run the model — reads from the prediction store only."
        ),
        contact={"name": "ML Engineering"},
        openapi_tags=[
            {"name": "predictions", "description": "Query scored users"},
            {"name": "ops", "description": "Health and readiness"},
        ],
        lifespan=lifespan,
    )

    app.include_router(score.router)
    app.include_router(cohort.router)
    register_exception_handlers(app)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="API liveness and database readiness probe",
        tags=["ops"],
    )
    async def health(request: Request) -> JSONResponse:
        pool = getattr(request.app.state, "pool", None)
        db_status = "unreachable"
        http_status = 503

        if pool is not None:
            conn = None
            try:
                conn = pool.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                db_status = "connected"
                http_status = 200
            except Exception:
                db_status = "unreachable"
            finally:
                if conn is not None:
                    try:
                        pool.putconn(conn)
                    except Exception:
                        pass

        body = HealthResponse(
            status="ok" if db_status == "connected" else "degraded",
            version=_API_VERSION,
            database=db_status,
            timestamp=datetime.now(timezone.utc),
        )
        return JSONResponse(status_code=http_status, content=body.model_dump(mode="json"))

    return app


app = create_app()
