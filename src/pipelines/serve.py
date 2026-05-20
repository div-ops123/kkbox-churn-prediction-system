"""
Daily batch scoring pipeline — Step 3.

Fetches users whose subscription expires in 13-15 days, builds features
via the shared feature module, scores with the production LightGBM model,
and writes results to the PostgreSQL predictions table and a Parquet file.

Run locally (single execution):
    python src/serve.py

Run via Prefect:
    prefect deployment run kkbox-churn-serving/local --param scoring_date=2017-03-01

Parameters
----------
scoring_date : date, optional
    Date to score for. Defaults to today. Pass a historical date to backfill.
use_mlflow : bool
    True (default) — load the "production"-aliased model from MLflow registry.
    False          — load from models/lgbm_baseline.pkl (local dev / testing).
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from prefect import flow, task, get_run_logger

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ServingConfig,
    load_feature_config,
    load_risk_tiers_config,
    load_serving_config,
)
from feature_module import FEATURE_COLS, build_features
from model_loader import (
    ModelLoadError,
    ModelNotFoundError,
    ScoringModel,
    make_model_loader,
)
from risk_tier import TierAssignmentError, make_tier_strategy


# ── Custom exceptions ─────────────────────────────────────────────────────────

class DatabaseConnectionError(Exception):
    """psycopg2 connection failed."""

class CohortQueryError(Exception):
    """SQL query for the expiring cohort returned an unexpected error."""

class EmptyFeatureMatrixError(Exception):
    """build_features() returned 0 rows for a non-empty cohort."""

class FeatureBuildError(Exception):
    """Wraps any exception raised by build_features()."""

class ScoringError(Exception):
    """Wraps any exception raised by model.predict_proba()."""

class DatabaseWriteError(Exception):
    """psycopg2 failure during predictions upsert."""

class ParquetWriteError(Exception):
    """OSError or pyarrow error during Parquet write."""

class MetricsWriteError(Exception):
    """Failure writing to monitoring_metrics — non-fatal."""


# ── DB context manager ────────────────────────────────────────────────────────

@contextmanager
def _pg_conn(config: ServingConfig) -> Generator[psycopg2.extensions.connection, None, None]:
    """Open a psycopg2 connection, yield it, close in finally."""
    conn = None
    try:
        conn = psycopg2.connect(config.postgres.dsn)
        yield conn
    except psycopg2.OperationalError as exc:
        raise DatabaseConnectionError(f"Cannot connect to PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="fetch-expiring-cohort", retries=3, retry_delay_seconds=60, tags=["db", "cohort"])
def fetch_expiring_cohort(
    config: ServingConfig,
    scoring_date: date,
    window_days_min: int = 13,
    window_days_max: int = 15,
) -> pd.DataFrame:
    """
    Query PostgreSQL for users whose membership_expire_date falls in
    [scoring_date + window_days_min, scoring_date + window_days_max].

    Returns a DataFrame with columns ["msno", "expiry_date"].
    Returns an empty DataFrame if no users are in the window.
    """
    logger = get_run_logger()
    lo = scoring_date + timedelta(days=window_days_min)
    hi = scoring_date + timedelta(days=window_days_max)
    lo_int = int(lo.strftime("%Y%m%d"))
    hi_int = int(hi.strftime("%Y%m%d"))

    sql = """
        SELECT
            msno,
            TO_DATE(CAST(MAX(membership_expire_date) AS TEXT), 'YYYYMMDD') AS expiry_date
        FROM transactions
        WHERE membership_expire_date BETWEEN %(lo)s AND %(hi)s
          AND membership_expire_date >= transaction_date
        GROUP BY msno
    """
    try:
        with _pg_conn(config) as conn:
            df = pd.read_sql(sql, conn, params={"lo": lo_int, "hi": hi_int})
    except DatabaseConnectionError:
        raise
    except Exception as exc:
        raise CohortQueryError(f"Cohort query failed: {exc}") from exc

    logger.info(
        "fetch_expiring_cohort complete",
        extra={
            "cohort_size": len(df),
            "window_start": str(lo),
            "window_end": str(hi),
            "scoring_date": str(scoring_date),
        },
    )
    return df


@task(name="build-feature-matrix", retries=1, retry_delay_seconds=30, tags=["features"])
def build_feature_matrix(
    cohort_df: pd.DataFrame,
    config: ServingConfig,
    p99_secs: float,
) -> pd.DataFrame:
    """
    Call build_features() with data_source="postgres" for the cohort.
    Validates output shape and column names before returning.
    """
    logger = get_run_logger()
    try:
        feature_df = build_features(
            msno_list=cohort_df["msno"].tolist(),
            expire_dates=cohort_df["expiry_date"].tolist(),
            p99_secs=p99_secs,
            data_source="postgres",
            pg_conn_str=config.postgres.conn_str,
        )
    except Exception as exc:
        raise FeatureBuildError(f"build_features() failed: {exc}") from exc

    if len(feature_df) == 0:
        raise EmptyFeatureMatrixError(
            f"build_features() returned 0 rows for a cohort of {len(cohort_df)}"
        )
    assert list(feature_df.columns) == FEATURE_COLS, (
        f"Feature column mismatch. Expected {FEATURE_COLS}, got {list(feature_df.columns)}"
    )

    null_rates = feature_df.isna().mean()
    overall_null = float(null_rates.mean())
    logger.info(
        "build_feature_matrix complete",
        extra={
            "rows": len(feature_df),
            "overall_null_rate": round(overall_null, 4),
            "scoring_date": str(feature_df.index.name),
        },
    )
    for col, rate in null_rates.items():
        logger.debug(f"null_rate {col}={rate:.4f}")

    return feature_df


@task(name="load-production-model", retries=2, retry_delay_seconds=120, tags=["mlflow", "model"])
def load_production_model(
    config: ServingConfig,
    use_mlflow: bool = True,
) -> tuple[ScoringModel, str]:
    """
    Load the production model. Returns (model, model_version_str).
    model_version_str is the MLflow run_id (or a label for local pickle).
    """
    logger = get_run_logger()
    loader = make_model_loader(
        use_mlflow=use_mlflow,
        mlflow_config=config.mlflow if use_mlflow else None,
        pkl_path=BASE_DIR / "models" / "lgbm_baseline.pkl" if not use_mlflow else None,
        version_label="lgbm_baseline_local",
    )
    model = loader.load()
    version = loader.get_model_version()
    logger.info(
        "load_production_model complete",
        extra={
            "model_version": version,
            "use_mlflow": use_mlflow,
            "model_name": config.mlflow.model_name,
            "alias": config.mlflow.model_alias,
        },
    )
    return model, version


@task(name="score-cohort", retries=0, tags=["inference"])
def score_cohort(
    feature_df: pd.DataFrame,
    model: ScoringModel,
) -> np.ndarray:
    """
    Run predict_proba on the feature matrix.
    Returns a 1-D array of churn probabilities (positive class).
    """
    logger = get_run_logger()
    try:
        # LightGBM Booster (from lgb.train) uses .predict() — returns 1-D probabilities.
        # Sklearn-style wrappers use .predict_proba() — returns 2-D (n, 2) array.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_df[FEATURE_COLS])
            scores = proba[:, 1] if proba.ndim == 2 else proba
        else:
            scores = model.predict(feature_df[FEATURE_COLS])
    except Exception as exc:
        raise ScoringError(f"model scoring failed: {exc}") from exc

    assert scores.shape == (len(feature_df),), (
        f"Unexpected score shape {scores.shape}, expected ({len(feature_df)},)"
    )
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ScoringError("predict_proba returned values outside [0, 1]")

    logger.info(
        "score_cohort complete",
        extra={
            "n_users": len(scores),
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "score_mean": round(float(scores.mean()), 4),
            "score_p50": round(float(np.percentile(scores, 50)), 4),
            "score_p90": round(float(np.percentile(scores, 90)), 4),
        },
    )
    return scores


@task(name="assign-risk-tiers", retries=0, tags=["classification"])
def assign_risk_tiers(scores: np.ndarray, tier_config: dict) -> list[str]:
    """Assign HIGH/MED/LOW to each score using the loaded tier strategy."""
    logger = get_run_logger()
    strategy = make_tier_strategy(tier_config)
    tiers = strategy.assign(scores)

    for name in ("HIGH", "MED", "LOW"):
        count = tiers.count(name)
        pct = count / len(tiers) if tiers else 0.0
        logger.info(f"tier_distribution {name}: count={count} pct={pct:.3f}")

    logger.info("assign_risk_tiers complete", extra=strategy.describe())
    return tiers


@task(name="write-predictions-postgres", retries=3, retry_delay_seconds=60, tags=["db", "write"])
def write_predictions_postgres(
    cohort_df: pd.DataFrame,
    scores: np.ndarray,
    tiers: list[str],
    scoring_date: date,
    model_version: str,
    config: ServingConfig,
) -> int:
    """
    Upsert predictions into the PostgreSQL predictions table.
    Uses ON CONFLICT (msno, scoring_date) DO UPDATE — safe to re-run.
    Batches of 5,000 rows. Returns total rows upserted.
    """
    logger = get_run_logger()
    sql = """
        INSERT INTO predictions
            (msno, score, risk_tier, scoring_date, expiry_date, model_version)
        VALUES %s
        ON CONFLICT (msno, scoring_date) DO UPDATE SET
            score         = EXCLUDED.score,
            risk_tier     = EXCLUDED.risk_tier,
            expiry_date   = EXCLUDED.expiry_date,
            model_version = EXCLUDED.model_version,
            created_at    = NOW()
    """
    rows = [
        (
            str(msno),
            float(score),
            str(tier),
            scoring_date,
            exp_date,
            model_version,
        )
        for msno, score, tier, exp_date in zip(
            cohort_df["msno"], scores, tiers, cohort_df["expiry_date"]
        )
    ]

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
    except DatabaseConnectionError:
        raise
    except Exception as exc:
        raise DatabaseWriteError(f"Upsert to predictions table failed: {exc}") from exc

    logger.info(
        "write_predictions_postgres complete",
        extra={"rows_upserted": total_written, "scoring_date": str(scoring_date)},
    )
    return total_written


@task(name="write-predictions-parquet", retries=2, retry_delay_seconds=30, tags=["file", "write"])
def write_predictions_parquet(
    cohort_df: pd.DataFrame,
    scores: np.ndarray,
    tiers: list[str],
    scoring_date: date,
    output_dir: Path,
) -> Path:
    """
    Write predictions to data/predictions_<scoring_date>.parquet.
    Overwrites if a file already exists (idempotent re-run).
    """
    logger = get_run_logger()
    out_path = output_dir / f"predictions_{scoring_date}.parquet"

    if out_path.exists():
        logger.warning(f"Overwriting existing Parquet for scoring_date={scoring_date}: {out_path}")

    out_df = pd.DataFrame({
        "msno":         cohort_df["msno"].values,
        "score":        scores.astype("float32"),
        "risk_tier":    tiers,
        "scoring_date": scoring_date,
        "expiry_date":  cohort_df["expiry_date"].values,
    })
    try:
        out_df.to_parquet(out_path, index=False)
    except Exception as exc:
        raise ParquetWriteError(f"Failed to write Parquet to {out_path}: {exc}") from exc

    logger.info(
        "write_predictions_parquet complete",
        extra={"path": str(out_path), "rows": len(out_df)},
    )
    return out_path


@task(name="log-pipeline-health-metrics", retries=2, retry_delay_seconds=30, tags=["db", "monitoring"])
def log_pipeline_health_metrics(
    scoring_date: date,
    cohort_size: int,
    feature_df: pd.DataFrame,
    scores: np.ndarray,
    tiers: list[str],
    rows_written: int,
    config: ServingConfig,
) -> None:
    """
    Write pipeline health metrics to the monitoring_metrics table.
    Alerts when cohort_size < 100 or overall null rate > 0.99.
    """
    logger = get_run_logger()

    txn_cols = FEATURE_COLS[:12]
    log_cols = FEATURE_COLS[12:18]
    mem_cols = FEATURE_COLS[18:]

    null_overall = float(feature_df.isna().mean().mean())
    null_txn = float(feature_df[txn_cols].isna().mean().mean())
    null_log = float(feature_df[log_cols].isna().mean().mean())
    null_mem = float(feature_df[mem_cols].isna().mean().mean())

    n = len(tiers)
    high_pct = tiers.count("HIGH") / n if n else 0.0
    med_pct = tiers.count("MED") / n if n else 0.0
    low_pct = tiers.count("LOW") / n if n else 0.0

    metrics = [
        ("cohort_size",         float(cohort_size),             cohort_size < 100),
        ("null_rate_overall",   null_overall,                   null_overall > 0.99),
        ("null_rate_txn",       null_txn,                       False),
        ("null_rate_log",       null_log,                       False),
        ("null_rate_member",    null_mem,                       False),
        ("score_mean",          float(np.mean(scores)),         False),
        ("score_p50",           float(np.percentile(scores, 50)), False),
        ("score_p90",           float(np.percentile(scores, 90)), False),
        ("high_tier_pct",       high_pct,                       False),
        ("med_tier_pct",        med_pct,                        False),
        ("low_tier_pct",        low_pct,                        False),
        ("rows_written",        float(rows_written),            False),
    ]

    rows = [
        (scoring_date, "pipeline_health", name, value, alert)
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
        raise MetricsWriteError(f"Failed to write pipeline health metrics: {exc}") from exc

    alerted = [name for name, _, alert in metrics if alert]
    if alerted:
        logger.warning(f"Pipeline health alerts triggered: {alerted}")
    logger.info(
        "log_pipeline_health_metrics complete",
        extra={"metrics_written": len(metrics), "alerts": alerted},
    )


def _write_zero_cohort_metric(scoring_date: date, config: ServingConfig) -> None:
    """Write a single monitoring metric row when the cohort is empty."""
    sql = """
        INSERT INTO monitoring_metrics
            (run_date, pipeline_type, metric_name, metric_value, alert_triggered)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    try:
        with _pg_conn(config) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (scoring_date, "pipeline_health", "cohort_size", 0.0, True))
            conn.commit()
    except Exception:
        pass  # non-fatal; don't mask the empty-cohort early-exit


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-churn-serving",
    description="Daily batch scoring pipeline: fetch cohort, compute features, score, write predictions.",
    version="1.0.0",
)
def run_serving_pipeline(
    scoring_date: date | None = None,
    use_mlflow: bool = True,
) -> dict:
    """
    Orchestrate the daily scoring pipeline.

    Parameters
    ----------
    scoring_date : date, optional — defaults to today.
    use_mlflow   : bool — False uses local pickle (dev/test only).

    Returns
    -------
    dict with keys: scoring_date, cohort_size, rows_written, model_version,
                    parquet_path, status ("success" | "partial" | "failed")
    """
    logger = get_run_logger()
    if scoring_date is None:
        scoring_date = date.today()

    config = load_serving_config()
    p99_secs = load_feature_config(config.feature_config_path)["p99_secs"]
    tier_config = load_risk_tiers_config(config.risk_tiers_config_path)

    logger.info(
        "run_serving_pipeline started | scoring_date=%s | use_mlflow=%s",
        scoring_date,
        use_mlflow,
    )

    # ── Step 1: Cohort ────────────────────────────────────────────────────────
    cohort_df = fetch_expiring_cohort(config=config, scoring_date=scoring_date)

    if len(cohort_df) == 0:
        logger.warning(
            f"Empty cohort for scoring_date={scoring_date}. No predictions written."
        )
        _write_zero_cohort_metric(scoring_date, config)
        return {
            "scoring_date": str(scoring_date),
            "cohort_size": 0,
            "rows_written": 0,
            "model_version": None,
            "parquet_path": None,
            "status": "success",
        }

    # ── Steps 2-6: Model, features, scores, tiers, writes ────────────────────
    model, model_version = load_production_model(config=config, use_mlflow=use_mlflow)
    feature_df = build_feature_matrix(cohort_df=cohort_df, config=config, p99_secs=p99_secs)
    scores = score_cohort(feature_df=feature_df, model=model)
    tiers = assign_risk_tiers(scores=scores, tier_config=tier_config)
    rows_written = write_predictions_postgres(
        cohort_df=cohort_df,
        scores=scores,
        tiers=tiers,
        scoring_date=scoring_date,
        model_version=model_version,
        config=config,
    )

    # ── Step 7: Parquet (non-fatal) ───────────────────────────────────────────
    parquet_path: str | None = None
    try:
        parquet_path = str(
            write_predictions_parquet(
                cohort_df=cohort_df,
                scores=scores,
                tiers=tiers,
                scoring_date=scoring_date,
                output_dir=config.predictions_parquet_dir,
            )
        )
    except ParquetWriteError as exc:
        logger.warning(f"Parquet write failed (non-fatal): {exc}")

    # ── Step 8: Health metrics (non-fatal) ────────────────────────────────────
    status = "success"
    try:
        log_pipeline_health_metrics(
            scoring_date=scoring_date,
            cohort_size=len(cohort_df),
            feature_df=feature_df,
            scores=scores,
            tiers=tiers,
            rows_written=rows_written,
            config=config,
        )
    except MetricsWriteError as exc:
        logger.error(f"Health metrics write failed: {exc}")
        status = "partial"

    result = {
        "scoring_date": str(scoring_date),
        "cohort_size": len(cohort_df),
        "rows_written": rows_written,
        "model_version": model_version,
        "parquet_path": parquet_path,
        "status": status,
    }
    logger.info("run_serving_pipeline complete", extra=result)
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the KKBox churn serving pipeline.")
    parser.add_argument("--date", type=lambda s: date.fromisoformat(s), default=None,
                        help="Scoring date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Use local pickle model instead of MLflow registry.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_serving_pipeline(scoring_date=args.date, use_mlflow=not args.no_mlflow)
    print(result)
