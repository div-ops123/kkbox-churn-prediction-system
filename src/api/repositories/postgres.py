"""
PostgreSQL-backed prediction repository.

Owns all SQL for the Score API. Returns typed Pydantic objects — never
raw dicts or psycopg2 cursors to callers.

For large cohort queries (> 50,000 rows) a named server-side cursor is used
so that results are streamed in batches rather than loaded fully into memory.
"""

from __future__ import annotations

from datetime import date

import psycopg2
import psycopg2.extras
import psycopg2.pool

from ..exceptions import RepositoryConnectionError
from ..schemas import CohortItem, CohortResponse, PredictionRecord
from .base import AbstractPredictionRepository

_LARGE_RESULT_THRESHOLD = 50_000
_STREAM_BATCH_SIZE = 5_000


class PostgresPredictionRepository(AbstractPredictionRepository):
    """
    Concrete repository backed by a psycopg2 ThreadedConnectionPool.
    The pool is created once in the FastAPI lifespan and injected here.
    """

    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool) -> None:
        self._pool = pool

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _borrow(self) -> psycopg2.extensions.connection:
        try:
            return self._pool.getconn()
        except Exception as exc:
            raise RepositoryConnectionError(
                f"Failed to borrow connection from pool: {exc}"
            ) from exc

    def _return(self, conn: psycopg2.extensions.connection) -> None:
        try:
            self._pool.putconn(conn)
        except Exception:
            pass  # best-effort return

    # ── Repository methods ────────────────────────────────────────────────────

    def get_latest_for_user(self, user_id: str) -> PredictionRecord | None:
        sql = """
            SELECT msno, score, risk_tier, scoring_date, expiry_date,
                   model_version, created_at
            FROM predictions
            WHERE msno = %s
            ORDER BY scoring_date DESC
            LIMIT 1
        """
        conn = self._borrow()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
        except psycopg2.OperationalError as exc:
            raise RepositoryConnectionError(f"DB error on get_latest_for_user: {exc}") from exc
        finally:
            self._return(conn)

        if row is None:
            return None
        return PredictionRecord(
            user_id=row["msno"],
            score=float(row["score"]),
            risk_tier=row["risk_tier"],
            scoring_date=row["scoring_date"],
            expiry_date=row["expiry_date"],
            model_version=row["model_version"],
            created_at=row["created_at"],
        )

    def get_cohort_for_date(self, scoring_date: date) -> CohortResponse:
        count_sql = "SELECT COUNT(*) FROM predictions WHERE scoring_date = %s"
        data_sql = """
            SELECT msno, score, risk_tier, expiry_date, model_version
            FROM predictions
            WHERE scoring_date = %s
            ORDER BY score DESC
        """
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(count_sql, (scoring_date,))
                total: int = cur.fetchone()[0]

            items: list[CohortItem] = []
            if total > 0:
                if total > _LARGE_RESULT_THRESHOLD:
                    # Stream via named server-side cursor to avoid loading all rows into RAM.
                    with conn.cursor(
                        name="cohort_cursor",
                        cursor_factory=psycopg2.extras.RealDictCursor,
                    ) as cur:
                        cur.execute(data_sql, (scoring_date,))
                        while True:
                            batch = cur.fetchmany(_STREAM_BATCH_SIZE)
                            if not batch:
                                break
                            for row in batch:
                                items.append(
                                    CohortItem(
                                        user_id=row["msno"],
                                        score=float(row["score"]),
                                        risk_tier=row["risk_tier"],
                                        expiry_date=row["expiry_date"],
                                        model_version=row["model_version"],
                                    )
                                )
                else:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(data_sql, (scoring_date,))
                        for row in cur.fetchall():
                            items.append(
                                CohortItem(
                                    user_id=row["msno"],
                                    score=float(row["score"]),
                                    risk_tier=row["risk_tier"],
                                    expiry_date=row["expiry_date"],
                                    model_version=row["model_version"],
                                )
                            )
        except psycopg2.OperationalError as exc:
            raise RepositoryConnectionError(f"DB error on get_cohort_for_date: {exc}") from exc
        finally:
            self._return(conn)

        return CohortResponse(
            scoring_date=scoring_date,
            count=len(items),
            predictions=items,
        )

    def health_check(self) -> bool:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except psycopg2.OperationalError as exc:
            raise RepositoryConnectionError(f"Health check failed: {exc}") from exc
        finally:
            self._return(conn)
        return True
