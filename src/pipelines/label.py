"""
Monthly labeling pipeline — Step 5.

Identifies users whose subscription expired in a target cohort month, derives
is_churn labels from transaction renewal patterns, and writes results to the
PostgreSQL labels table.

Labels are always derived from transactions: is_churn = 0 if the user has a
valid (is_cancel=0) transaction within 30 days of their anchor_expiry_date.

Run locally (single execution):
    python src/label.py --cohort-month 2017-03

Run via Prefect:
    prefect deployment run kkbox-labeling-pipeline/local --param cohort_month=2017-03

Parameters
----------
cohort_month : str  — "YYYY-MM" target cohort (e.g. "2017-03")
data_source  : str  — "postgres" (default) | "csv" (dev/testing only)
data_dir     : str  — relative path to CSV directory (default "data/")
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd
import psycopg2
import psycopg2.extras
from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LabelingConfig, load_labeling_config
from core.label_module import LABEL_COLS, build_cohort, compute_labels


# ── Custom exceptions ─────────────────────────────────────────────────────────

class LabelingConfigError(Exception):
    """PostgreSQL connection failed or config is invalid."""

class CohortBuildError(Exception):
    """build_cohort() returned an unexpected error."""

class LabelComputationError(Exception):
    """compute_labels() returned an unexpected error."""

class LabelWriteError(Exception):
    """psycopg2 upsert to the labels table failed."""

class LabelingMetricsWriteError(Exception):
    """Write to monitoring_metrics failed — non-fatal."""


# ── DB context manager ────────────────────────────────────────────────────────

@contextmanager
def _pg_conn(config: LabelingConfig) -> Generator[psycopg2.extensions.connection, None, None]:
    """Open a psycopg2 connection, yield it, close in finally."""
    conn = None
    try:
        conn = psycopg2.connect(config.postgres.dsn)
        yield conn
    except psycopg2.OperationalError as exc:
        raise LabelingConfigError(f"Cannot connect to PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="build-label-cohort", retries=3, retry_delay_seconds=60, tags=["db", "cohort"])
def build_label_cohort(config: LabelingConfig) -> pd.DataFrame:
    """
    Find users whose latest valid subscription expiry falls in config.cohort_month.

    Returns a DataFrame with columns [msno, anchor_expiry_date, feature_cutoff_date].
    Returns an empty DataFrame when no users expire in the target month.
    """
    logger = get_run_logger()
    try:
        cohort_df = build_cohort(config)
    except Exception as exc:
        raise CohortBuildError(f"build_cohort() failed for {config.cohort_month}: {exc}") from exc

    logger.info(
        f"build_label_cohort complete | cohort_month={config.cohort_month}"
        f" | cohort_size={len(cohort_df)} | data_source={config.data_source}"
    )
    return cohort_df


@task(name="compute-churn-labels", retries=0, tags=["labeling"])
def compute_churn_labels(cohort_df: pd.DataFrame, config: LabelingConfig) -> pd.DataFrame:
    """
    Assign is_churn to each user in the cohort.

    retries=0 — labeling is deterministic; a failure is a code bug, not transience.

    Returns a DataFrame with columns LABEL_COLS.
    """
    logger = get_run_logger()
    try:
        labels_df = compute_labels(cohort_df, config)
    except Exception as exc:
        raise LabelComputationError(f"compute_labels() failed: {exc}") from exc

    nan_count = int(labels_df["is_churn"].isna().sum())
    churn_rate = float(labels_df["is_churn"].dropna().mean()) if len(labels_df) > 0 else 0.0

    logger.info(
        f"compute_churn_labels complete | rows_labeled={len(labels_df)}"
        f" | churn_rate={round(churn_rate, 4)}"
    )
    return labels_df


@task(name="write-labels-postgres", retries=3, retry_delay_seconds=60, tags=["db", "write"])
def write_labels_postgres(labels_df: pd.DataFrame, config: LabelingConfig) -> int:
    """
    Upsert labels into the PostgreSQL labels table.

    Uses ON CONFLICT (msno, anchor_expiry_date) DO UPDATE — safe to re-run.
    Batches of 5,000 rows. Returns total rows upserted.
    """
    logger = get_run_logger()

    valid_df = labels_df
    if valid_df.empty:
        logger.warning("No rows to write — labels_df is empty")
        return 0

    sql = """
        INSERT INTO labels
            (msno, anchor_expiry_date, feature_cutoff_date, is_churn, labeled_date)
        VALUES %s
        ON CONFLICT (msno, anchor_expiry_date) DO UPDATE SET
            feature_cutoff_date = EXCLUDED.feature_cutoff_date,
            is_churn            = EXCLUDED.is_churn,
            labeled_date        = EXCLUDED.labeled_date
    """
    rows = list(
        zip(
            valid_df["msno"].tolist(),
            valid_df["anchor_expiry_date"].tolist(),
            valid_df["feature_cutoff_date"].tolist(),
            valid_df["is_churn"].astype(int).tolist(),
            valid_df["labeled_date"].tolist(),
        )
    )

    batch_size = 5_000
    total_written = 0
    try:
        with _pg_conn(config) as conn:
            with conn.cursor() as cur:
                for i in range(0, len(rows), batch_size):
                    batch = rows[i : i + batch_size]
                    psycopg2.extras.execute_values(cur, sql, batch)
                    conn.commit()
                    total_written += len(batch)
    except LabelingConfigError:
        raise
    except Exception as exc:
        raise LabelWriteError(f"Upsert to labels table failed: {exc}") from exc

    logger.info(f"write_labels_postgres complete | rows_upserted={total_written} | cohort_month={config.cohort_month}")
    return total_written


@task(name="log-labeling-metrics", retries=1, retry_delay_seconds=30, tags=["db", "monitoring"])
def log_labeling_metrics(
    labels_df: pd.DataFrame,
    rows_written: int,
    config: LabelingConfig,
) -> None:
    """
    Write pipeline health metrics to the monitoring_metrics table.

    Alerts when: cohort_size < 100, churn_rate > 0.5.
    """
    logger = get_run_logger()

    churn_rate = float(labels_df["is_churn"].mean()) if len(labels_df) > 0 else 0.0

    metrics = [
        ("cohort_size",     float(len(labels_df)),  len(labels_df) < 100),
        ("labels_written",  float(rows_written),    False),
        ("churn_rate",      churn_rate,             churn_rate > 0.5),
    ]

    rows = [
        (config.labeled_date, "labeling", name, value, alert)
        for name, value, alert in metrics
    ]

    sql = """
        INSERT INTO monitoring_metrics
            (run_date, pipeline_type, metric_name, metric_value, alert_triggered)
        VALUES %s
    """
    try:
        with _pg_conn(config) as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, sql, rows)
            conn.commit()
    except Exception as exc:
        raise LabelingMetricsWriteError(f"Failed to write labeling metrics: {exc}") from exc

    alerted = [name for name, _, alert in metrics if alert]
    if alerted:
        logger.warning(f"Labeling alerts triggered: {alerted}")
    logger.info(f"log_labeling_metrics complete | metrics_written={len(metrics)} | alerts={alerted}")


def _write_empty_cohort_metric(config: LabelingConfig) -> None:
    """Write a single monitoring row when the cohort is empty. Non-fatal."""
    sql = """
        INSERT INTO monitoring_metrics
            (run_date, pipeline_type, metric_name, metric_value, alert_triggered)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    try:
        with _pg_conn(config) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (config.labeled_date, "labeling", "cohort_size", 0.0, True))
            conn.commit()
    except Exception:
        pass  # non-fatal; don't mask the empty-cohort early-exit


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-labeling-pipeline",
    description=(
        "Monthly labeling pipeline: identify expiring cohort, assign churn labels, "
        "write to the labels table."
    ),
    version="1.0.0",
)
def run_labeling_pipeline(
    cohort_month: str,
    data_source: str = "postgres",
    data_dir: str = "data/",
) -> dict:
    """
    Orchestrate the monthly labeling pipeline.

    Parameters
    ----------
    cohort_month : str  — "YYYY-MM" target cohort month.
    data_source  : str  — "postgres" (default) or "csv" (dev only).
    data_dir     : str  — relative path to CSV directory.

    Returns
    -------
    dict with keys: cohort_month, cohort_size, labels_written, churn_rate, status.
    status is "success" when all tasks completed, "partial" when metrics write failed.
    """
    logger = get_run_logger()
    config = load_labeling_config(
        cohort_month=cohort_month,
        data_source=data_source,
        data_dir=data_dir,
    )

    logger.info(
        "run_labeling_pipeline started | cohort_month=%s | data_source=%s",
        cohort_month,
        data_source,
    )

    # ── Step 1: Cohort ────────────────────────────────────────────────────────
    cohort_df = build_label_cohort(config=config)

    if len(cohort_df) == 0:
        logger.warning(
            f"Empty cohort for cohort_month={cohort_month}. No labels written."
        )
        _write_empty_cohort_metric(config)
        return {
            "cohort_month": cohort_month,
            "cohort_size": 0,
            "labels_written": 0,
            "churn_rate": None,
            "status": "success",
        }

    # ── Step 2: Labels ────────────────────────────────────────────────────────
    labels_df = compute_churn_labels(cohort_df=cohort_df, config=config)

    # ── Step 3: Write ─────────────────────────────────────────────────────────
    rows_written = write_labels_postgres(labels_df=labels_df, config=config)

    # ── Step 4: Health metrics (non-fatal) ────────────────────────────────────
    status = "success"
    try:
        log_labeling_metrics(labels_df=labels_df, rows_written=rows_written, config=config)
    except LabelingMetricsWriteError as exc:
        logger.error(f"Labeling metrics write failed: {exc}")
        status = "partial"

    churn_rate = float(labels_df["is_churn"].dropna().mean()) if len(labels_df) > 0 else 0.0

    result = {
        "cohort_month": cohort_month,
        "cohort_size": len(cohort_df),
        "labels_written": rows_written,
        "churn_rate": round(churn_rate, 4),
        "status": status,
    }
    logger.info(
        f"run_labeling_pipeline complete | cohort_month={result['cohort_month']}"
        f" | cohort_size={result['cohort_size']} | labels_written={result['labels_written']}"
        f" | churn_rate={result['churn_rate']} | status={result['status']}"
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the KKBox churn labeling pipeline.")
    parser.add_argument(
        "--cohort-month",
        required=True,
        help="Target cohort month in YYYY-MM format (e.g. 2017-03).",
    )
    parser.add_argument(
        "--data-source",
        choices=["postgres", "csv"],
        default="postgres",
        help="DuckDB read source: 'postgres' (default) or 'csv' (dev only).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Relative path to the CSV directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_labeling_pipeline(
        cohort_month=args.cohort_month,
        data_source=args.data_source,
        data_dir=args.data_dir,
    )
    print(result)
