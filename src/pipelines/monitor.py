"""
Production monitoring pipeline — Track 1 (drift) and Track 2 (performance).

Track 1 (run_drift_monitoring): Compares the day's serving cohort feature
    distribution against baseline training features using PSI. Computes and
    logs 21 metrics (20 per-feature PSI values + 1 summary) — console output
    only, no persistence. In the common case it runs chained onto
    pipelines/serve.py's flow, reusing the feature matrix serve.py already
    built (via the current_df parameter) instead of rebuilding it. The
    standalone CLI invocation below still rebuilds via build_features() —
    use it to backfill a historical date or to re-run drift monitoring
    after a failed serving run.

Track 2 (run_performance_monitoring): Monthly. Joins month M predictions with
    month M labels to compute AUC-PR, AUC-ROC, Precision@K, Recall@K.
    Computes and logs 6 metrics — console output only. Optionally triggers
    retraining when AUC-PR falls below threshold.

Track 3 (pipeline health) is handled inline by serve.py, label.py,
    build_dataset.py, train.py, and validate.py.

Run locally:
    python src/pipelines/monitor.py drift --date 2017-03-15 --data-source csv
    python src/pipelines/monitor.py performance --cohort-month 2017-03
    python src/pipelines/monitor.py performance --cohort-month 2017-03 --no-retrain
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator

import mlflow
import pandas as pd
import psycopg2
from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MonitoringConfig, load_monitoring_config
from core.drift_module import compute_feature_drift, evaluate_cohort_performance
from core.feature_module import FEATURE_COLS, build_features
from core.model_loader import ModelNotFoundError, get_model_version_dataset_id


# ── Custom exceptions ──────────────────────────────────────────────────────────

class MonitoringConfigError(Exception):
    """psycopg2 connection failed or config is invalid."""

class ServingCohortQueryError(Exception):
    """SQL on predictions table failed."""

class EmptyServingCohortError(Exception):
    """No rows for scoring_date — early exit, not a pipeline failure."""

class DriftFeatureBuildError(Exception):
    """build_features() failed during drift monitoring."""

class BaselineNotFoundError(Exception):
    """baseline_features.parquet is missing — run training pipeline first."""

class DriftComputationError(Exception):
    """compute_feature_drift() raised an unexpected error."""

class PredictionsQueryError(Exception):
    """SQL on predictions table failed during performance monitoring."""

class LabelsQueryError(Exception):
    """SQL on labels table failed."""

class PerformanceComputationError(Exception):
    """evaluate_cohort_performance() failed."""

class RetrainingTriggerError(Exception):
    """run_training_pipeline() raised — non-fatal."""


# ── DB context manager ─────────────────────────────────────────────────────────

@contextmanager
def _pg_conn(config: MonitoringConfig) -> Generator[psycopg2.extensions.connection, None, None]:
    conn = None
    try:
        conn = psycopg2.connect(config.postgres.dsn)
        yield conn
    except psycopg2.OperationalError as exc:
        raise MonitoringConfigError(f"Cannot connect to PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


# ── Flow 1 tasks: Drift monitoring ────────────────────────────────────────────

@task(name="load-serving-cohort", retries=3, retry_delay_seconds=60, tags=["db", "drift"])
def load_serving_cohort(config: MonitoringConfig, scoring_date: date) -> pd.DataFrame:
    """
    Load the cohort scored on scoring_date from the predictions table.

    Returns DataFrame[msno, expiry_date].

    Raises:
        EmptyServingCohortError: If no predictions exist for scoring_date.
        ServingCohortQueryError: If the SQL fails.
    """
    logger = get_run_logger()
    sql = "SELECT msno, expiry_date FROM predictions WHERE scoring_date = %(d)s"
    try:
        with _pg_conn(config) as conn:
            df = pd.read_sql(sql, conn, params={"d": scoring_date})
    except MonitoringConfigError:
        raise
    except Exception as exc:
        raise ServingCohortQueryError(
            f"Failed to query predictions for scoring_date={scoring_date}: {exc}"
        ) from exc

    if len(df) == 0:
        raise EmptyServingCohortError(
            f"No predictions found for scoring_date={scoring_date}"
        )

    logger.info(f"load_serving_cohort | scoring_date={scoring_date} | rows={len(df)}")
    return df


@task(name="resolve-production-dataset", retries=2, retry_delay_seconds=30, tags=["mlflow", "drift"])
def resolve_production_dataset_version_id(config: MonitoringConfig) -> str:
    """
    Look up the dataset_version_id tag on whichever model version currently
    holds the production alias, without loading the model itself.

    Raises BaselineNotFoundError if no model currently holds the production
    alias (first run, before any training has happened), or if it has no
    dataset_version_id tag (registered before this pipeline existed).
    """
    logger = get_run_logger()
    try:
        dataset_version_id = get_model_version_dataset_id(config.mlflow, config.mlflow.model_alias)
    except ModelNotFoundError as exc:
        raise BaselineNotFoundError(
            f"No production model found: {exc}. Run the training + validation "
            "pipelines to establish one."
        ) from exc

    if dataset_version_id is None:
        raise BaselineNotFoundError(
            "Production model has no dataset_version_id tag — it was likely "
            "registered before this pipeline existed. Retrain to backfill it."
        )
    logger.info(f"resolve_production_dataset_version_id | dataset_version_id={dataset_version_id}")
    return dataset_version_id


@task(name="load-baseline-and-p99", retries=2, retry_delay_seconds=30, tags=["mlflow", "drift"])
def load_baseline_and_p99(dataset_version_id: str, config: MonitoringConfig) -> tuple[pd.DataFrame, float]:
    """
    Fetch baseline_features.parquet and p99_secs from the dataset-build
    MLflow run that trained the current production model — never a local
    file, so drift is always compared against what production actually
    trained on.
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=dataset_version_id, artifact_path="dataset", dst_path=tmpdir,
        )
        baseline_df = pd.read_parquet(Path(local_dir) / "baseline_features.parquet")
        with open(Path(local_dir) / "feature_config.json") as f:
            p99_secs = json.load(f)["p99_secs"]

    logger.info(
        f"load_baseline_and_p99 complete | dataset_version_id={dataset_version_id}"
        f" | baseline_rows={len(baseline_df)} | p99_secs={p99_secs}"
    )
    return baseline_df, p99_secs


@task(name="build-drift-features", retries=0, tags=["features", "drift"])
def build_drift_features(
    serving_cohort_df: pd.DataFrame,
    config: MonitoringConfig,
    p99_secs: float,
) -> pd.DataFrame:
    """
    Build features for the serving cohort using the shared feature_module.

    Args:
        serving_cohort_df: DataFrame[msno, expiry_date] from load_serving_cohort.
        config:            MonitoringConfig.
        p99_secs:          Winsorization threshold from the frozen feature_config.json.

    Returns DataFrame indexed by msno with FEATURE_COLS.

    Raises:
        DriftFeatureBuildError: If build_features() fails.
    """
    logger = get_run_logger()
    msno_list = serving_cohort_df["msno"].tolist()
    expire_dates = serving_cohort_df["expiry_date"].tolist()
    pg_conn_str = config.postgres.conn_str if config.data_source == "postgres" else None

    try:
        current_df = build_features(
            msno_list, expire_dates,
            p99_secs=p99_secs,
            data_source=config.data_source,
            data_dir=config.data_dir,
            pg_conn_str=pg_conn_str,
        )
    except Exception as exc:
        raise DriftFeatureBuildError(f"build_features() failed: {exc}") from exc

    logger.info(f"build_drift_features | rows={len(current_df)}")
    return current_df


@task(name="run-drift-check", retries=0, tags=["drift"])
def run_drift_check(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    config: MonitoringConfig,
    scoring_date: date,
) -> dict:
    """
    Compute PSI per feature against baseline_df, log 21 metrics (console only).

    Logs:
      - 20 values: psi_{col} per feature in FEATURE_COLS.
      - 1 summary value: n_drifted_features.

    Returns:
        Dict with n_drifted_features (int) and drifted_features (list of str).
    """
    logger = get_run_logger()

    try:
        psi_scores = compute_feature_drift(baseline_df, current_df, list(FEATURE_COLS))
    except Exception as exc:
        raise DriftComputationError(f"compute_feature_drift() failed: {exc}") from exc

    drifted = [col for col, psi in psi_scores.items() if psi > config.psi_alert_threshold]
    n_drifted = len(drifted)

    if drifted:
        logger.warning(f"run_drift_check | drifted features (PSI > {config.psi_alert_threshold}): {drifted}")
    logger.info(
        f"run_drift_check | psi_scores={psi_scores}"
        f" | n_drifted={n_drifted} | scoring_date={scoring_date}"
    )
    return {"n_drifted_features": n_drifted, "drifted_features": drifted}


# ── Flow 2 tasks: Performance monitoring ─────────────────────────────────────

@task(name="load-predictions-for-month", retries=3, retry_delay_seconds=60, tags=["db", "perf"])
def load_predictions_for_month(config: MonitoringConfig, cohort_month: str) -> pd.DataFrame:
    """
    Load predictions for the given cohort month from the predictions table.

    Filters by TO_CHAR(expiry_date, 'YYYY-MM') = cohort_month.

    Returns DataFrame[msno, score, expiry_date].

    Raises:
        PredictionsQueryError: If the SQL fails.
    """
    logger = get_run_logger()
    sql = """
        SELECT msno, score, expiry_date
        FROM predictions
        WHERE TO_CHAR(expiry_date, 'YYYY-MM') = %(m)s
    """
    try:
        with _pg_conn(config) as conn:
            df = pd.read_sql(sql, conn, params={"m": cohort_month})
    except MonitoringConfigError:
        raise
    except Exception as exc:
        raise PredictionsQueryError(
            f"Failed to query predictions for cohort_month={cohort_month}: {exc}"
        ) from exc

    logger.info(f"load_predictions_for_month | cohort_month={cohort_month} | rows={len(df)}")
    return df


@task(name="load-labels-for-month", retries=3, retry_delay_seconds=60, tags=["db", "perf"])
def load_labels_for_month(config: MonitoringConfig, cohort_month: str) -> pd.DataFrame:
    """
    Load labels for the given cohort month from the labels table.

    Filters by TO_CHAR(anchor_expiry_date, 'YYYY-MM') = cohort_month.

    Returns DataFrame[msno, anchor_expiry_date, is_churn].

    Raises:
        LabelsQueryError: If the SQL fails.
    """
    logger = get_run_logger()
    sql = """
        SELECT msno, anchor_expiry_date, is_churn
        FROM labels
        WHERE TO_CHAR(anchor_expiry_date, 'YYYY-MM') = %(m)s
    """
    try:
        with _pg_conn(config) as conn:
            df = pd.read_sql(sql, conn, params={"m": cohort_month})
    except MonitoringConfigError:
        raise
    except Exception as exc:
        raise LabelsQueryError(
            f"Failed to query labels for cohort_month={cohort_month}: {exc}"
        ) from exc

    logger.info(f"load_labels_for_month | cohort_month={cohort_month} | rows={len(df)}")
    return df


@task(name="compute-and-log-performance", retries=0, tags=["perf"])
def compute_and_log_performance(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: MonitoringConfig,
    cohort_month: str,
) -> dict:
    """
    Join predictions with labels, compute performance metrics, log 6 metrics
    (console only): auc_pr, auc_roc, precision_at_k, recall_at_k, n_total, n_churned.

    Args:
        predictions_df: DataFrame[msno, score, expiry_date] filtered to cohort_month.
        labels_df:      DataFrame[msno, anchor_expiry_date, is_churn] filtered to cohort_month.
        config:         MonitoringConfig.
        cohort_month:   "YYYY-MM" — used only for logging.

    Returns:
        Perf dict with auc_pr, auc_roc, precision_at_k, recall_at_k, n_total,
        n_churned, top_k, and auc_pr_alert (bool).

    Raises:
        PerformanceComputationError: If evaluate_cohort_performance fails.
    """
    logger = get_run_logger()

    try:
        perf = evaluate_cohort_performance(
            predictions_df[["msno", "score"]],
            labels_df[["msno", "is_churn"]],
            top_k=config.top_k,
        )
    except ValueError as exc:
        logger.error(
            f"compute_and_log_performance | empty join | cohort_month={cohort_month} | {exc}"
        )
        raise PerformanceComputationError(str(exc)) from exc
    except Exception as exc:
        raise PerformanceComputationError(str(exc)) from exc

    auc_pr = perf["auc_pr"]
    auc_pr_alert = auc_pr is not None and auc_pr < config.auc_pr_alert_threshold
    auc_pr_str = f"{auc_pr:.4f}" if auc_pr is not None else "N/A"

    logger.info(
        f"compute_and_log_performance | cohort_month={cohort_month}"
        f" | auc_pr={auc_pr_str} | auc_roc={perf['auc_roc']}"
        f" | precision_at_k={perf['precision_at_k']} | recall_at_k={perf['recall_at_k']}"
        f" | n_total={perf['n_total']} | n_churned={perf['n_churned']}"
        f" | auc_pr_alert={auc_pr_alert}"
    )
    return {**perf, "auc_pr_alert": auc_pr_alert}


@task(name="maybe-trigger-retraining", retries=0, tags=["retraining"])
def maybe_trigger_retraining(perf_result: dict, trigger_retraining: bool) -> dict:
    """
    Trigger dataset-build -> training -> validation if AUC-PR is below
    threshold and retraining is enabled.

    Uses Option A (direct sub-flow call): Prefect 3.x supports calling @flow from
    inside @task, which creates a child flow run traceable in the Prefect UI.
    run_retrain_and_validate_pipeline is a thin orchestrator over three
    independent pipelines (build_dataset, train, validate) — each stays
    separately invocable; this is just the common "auto-retrain" wiring.

    Args:
        perf_result:        Dict returned by compute_and_log_performance.
        trigger_retraining: False skips retraining regardless of alert state.

    Returns:
        Dict with "triggered" bool and optional "retrain_result".

    Raises:
        RetrainingTriggerError: If run_retrain_and_validate_pipeline() raises (non-fatal at flow level).
    """
    # Import here to avoid circular import at module load time.
    from pipelines.retrain import run_retrain_and_validate_pipeline

    logger = get_run_logger()

    if not trigger_retraining or not perf_result.get("auc_pr_alert", False):
        logger.info(
            f"maybe_trigger_retraining | skipped"
            f" | trigger_retraining={trigger_retraining}"
            f" | auc_pr_alert={perf_result.get('auc_pr_alert')}"
        )
        return {"triggered": False}

    logger.warning(
        f"maybe_trigger_retraining | AUC-PR alert active — triggering retrain-and-validate"
        f" | auc_pr={perf_result.get('auc_pr')}"
    )
    try:
        retrain_result = run_retrain_and_validate_pipeline(cohort_months=None)
        logger.info(f"maybe_trigger_retraining | retrain-and-validate complete | result={retrain_result}")
        return {"triggered": True, "retrain_result": retrain_result}
    except Exception as exc:
        raise RetrainingTriggerError(f"run_retrain_and_validate_pipeline() failed: {exc}") from exc


# ── Flow 1: Drift monitoring ──────────────────────────────────────────────────

@flow(name="kkbox-drift-monitoring", version="1.0.0")
def run_drift_monitoring(
    scoring_date: date | None = None,
    data_source: str = "postgres",
    data_dir: str = "data/",
    current_df: pd.DataFrame | None = None,
) -> dict:
    """
    Track 1 monitoring: daily PSI-based feature drift detection.

    Compares the serving cohort's feature distribution for scoring_date against
    the baseline features from the dataset-build run that trained whichever
    model currently holds the production alias (fetched from MLflow, never a
    local file — see resolve_production_dataset_version_id / load_baseline_and_p99).

    Args:
        scoring_date: Date to check predictions for (defaults to today).
        data_source:  "postgres" (default) or "csv" (dev only). Ignored when
                      current_df is provided.
        data_dir:     Relative path to CSV directory. Ignored when current_df
                      is provided.
        current_df:   Pre-built feature matrix (indexed by msno, FEATURE_COLS
                      columns), typically handed in by serve.py right after it
                      scores the same cohort — skips load_serving_cohort and
                      build_drift_features entirely, avoiding a redundant
                      rebuild of features the caller already has in memory.
                      Leave None for the standalone CLI path (backfilling a
                      historical date, or re-running after serve.py failed
                      that day), which rebuilds via build_features() as before.

    Returns:
        Dict with keys: scoring_date, cohort_size, n_drifted_features,
        drifted_features, status.
        status: "success" | "empty_cohort" | "baseline_missing"
    """
    logger = get_run_logger()
    config = load_monitoring_config(data_source=data_source, data_dir=data_dir)
    target_date = scoring_date or date.today()

    logger.info(
        f"run_drift_monitoring started"
        f" | scoring_date={target_date} | data_source={data_source}"
        f" | current_df_provided={current_df is not None}"
    )

    if current_df is not None:
        # ── Chained path: reuse the feature matrix serve.py already built ────
        if len(current_df) == 0:
            logger.warning(f"Empty current_df for scoring_date={target_date} — exiting early.")
            return {
                "scoring_date":       str(target_date),
                "cohort_size":        0,
                "n_drifted_features": 0,
                "drifted_features":   [],
                "status":             "empty_cohort",
            }
        cohort_size = len(current_df)
    else:
        # ── Standalone path: load cohort, rebuild features from scratch ──────
        try:
            cohort_df = load_serving_cohort(config=config, scoring_date=target_date)
        except EmptyServingCohortError:
            logger.warning(f"Empty serving cohort for scoring_date={target_date} — exiting early.")
            return {
                "scoring_date":       str(target_date),
                "cohort_size":        0,
                "n_drifted_features": 0,
                "drifted_features":   [],
                "status":             "empty_cohort",
            }
        cohort_size = len(cohort_df)

    # ── Step 2: Resolve the dataset that trained the production model, ────────
    #            then fetch baseline features + p99_secs from it (never local).
    try:
        dataset_version_id = resolve_production_dataset_version_id(config=config)
    except BaselineNotFoundError as exc:
        logger.error(f"run_drift_monitoring | baseline missing: {exc}")
        return {
            "scoring_date":       str(target_date),
            "cohort_size":        cohort_size,
            "n_drifted_features": 0,
            "drifted_features":   [],
            "status":             "baseline_missing",
        }

    baseline_df, p99_secs = load_baseline_and_p99(dataset_version_id=dataset_version_id, config=config)

    # ── Step 3: Current features — reuse if provided, else build ─────────────
    if current_df is None:
        current_df = build_drift_features(
            serving_cohort_df=cohort_df, config=config, p99_secs=p99_secs
        )

    # ── Step 4: Compute PSI and write metrics ──────────────────────────────────
    status = "success"
    drift_result: dict = {"n_drifted_features": 0, "drifted_features": []}
    try:
        drift_result = run_drift_check(
            current_df=current_df, baseline_df=baseline_df, config=config, scoring_date=target_date
        )
    except BaselineNotFoundError as exc:
        logger.error(f"run_drift_monitoring | baseline missing: {exc}")
        return {
            "scoring_date":       str(target_date),
            "cohort_size":        cohort_size,
            "n_drifted_features": 0,
            "drifted_features":   [],
            "status":             "baseline_missing",
        }

    result = {
        "scoring_date":       str(target_date),
        "cohort_size":        cohort_size,
        "n_drifted_features": drift_result["n_drifted_features"],
        "drifted_features":   drift_result["drifted_features"],
        "status":             status,
    }
    logger.info(
        f"run_drift_monitoring complete | status={status}"
        f" | n_drifted={drift_result['n_drifted_features']}"
    )
    return result


# ── Flow 2: Performance monitoring ───────────────────────────────────────────

@flow(name="kkbox-performance-monitoring", version="1.0.0")
def run_performance_monitoring(
    cohort_month: str,
    trigger_retraining: bool = True,
    data_source: str = "postgres",
    data_dir: str = "data/",
) -> dict:
    """
    Track 2 monitoring: monthly model performance evaluation.

    Joins month M predictions (filtered by expiry_date) with month M labels
    (filtered by anchor_expiry_date) on msno to compute performance metrics.
    Triggers retraining when AUC-PR falls below the configured threshold.

    Args:
        cohort_month:       "YYYY-MM" target cohort (e.g. "2017-03").
        trigger_retraining: Set False to evaluate without triggering retraining.
        data_source:        Passed through to load_monitoring_config.
        data_dir:           Passed through to load_monitoring_config.

    Returns:
        Dict with keys: cohort_month, n_predictions, n_labels, n_joined,
        auc_pr, auc_roc, auc_pr_alert, retraining_triggered, status.
        status: "success" | "partial" | "empty_predictions" | "empty_labels"
    """
    logger = get_run_logger()
    config = load_monitoring_config(data_source=data_source, data_dir=data_dir)

    logger.info(
        f"run_performance_monitoring started"
        f" | cohort_month={cohort_month} | trigger_retraining={trigger_retraining}"
    )

    # ── Step 1: Load predictions ──────────────────────────────────────────────
    predictions_df = load_predictions_for_month(config=config, cohort_month=cohort_month)
    if len(predictions_df) == 0:
        logger.warning(f"No predictions for cohort_month={cohort_month} — exiting early.")
        return {
            "cohort_month":         cohort_month,
            "n_predictions":        0,
            "n_labels":             0,
            "n_joined":             0,
            "auc_pr":               None,
            "auc_roc":              None,
            "auc_pr_alert":         True,
            "retraining_triggered": False,
            "status":               "empty_predictions",
        }

    # ── Step 2: Load labels ───────────────────────────────────────────────────
    labels_df = load_labels_for_month(config=config, cohort_month=cohort_month)
    if len(labels_df) == 0:
        logger.warning(f"No labels for cohort_month={cohort_month} — exiting early.")
        return {
            "cohort_month":         cohort_month,
            "n_predictions":        len(predictions_df),
            "n_labels":             0,
            "n_joined":             0,
            "auc_pr":               None,
            "auc_roc":              None,
            "auc_pr_alert":         True,
            "retraining_triggered": False,
            "status":               "empty_labels",
        }

    # ── Step 3: Compute and log performance ────────────────────────────────────
    status = "success"
    perf_result: dict = {}
    try:
        perf_result = compute_and_log_performance(
            predictions_df=predictions_df,
            labels_df=labels_df,
            config=config,
            cohort_month=cohort_month,
        )
    except PerformanceComputationError as exc:
        logger.error(f"run_performance_monitoring | compute failed (non-fatal): {exc}")
        status = "partial"

    # ── Step 4: Maybe trigger retraining ──────────────────────────────────────
    retrain_result: dict = {"triggered": False}
    try:
        retrain_result = maybe_trigger_retraining(
            perf_result=perf_result,
            trigger_retraining=trigger_retraining,
        )
    except RetrainingTriggerError as exc:
        logger.error(f"run_performance_monitoring | retraining trigger failed (non-fatal): {exc}")
        status = "partial"

    result = {
        "cohort_month":         cohort_month,
        "n_predictions":        len(predictions_df),
        "n_labels":             len(labels_df),
        "n_joined":             perf_result.get("n_total", 0),
        "auc_pr":               perf_result.get("auc_pr"),
        "auc_roc":              perf_result.get("auc_roc"),
        "auc_pr_alert":         perf_result.get("auc_pr_alert", False),
        "retraining_triggered": retrain_result.get("triggered", False),
        "status":               status,
    }
    logger.info(
        f"run_performance_monitoring complete | status={status}"
        f" | auc_pr={perf_result.get('auc_pr')} | n_joined={result['n_joined']}"
        f" | retraining_triggered={retrain_result.get('triggered')}"
    )
    return result


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import re as _re

    parser = argparse.ArgumentParser(
        description="KKBox monitoring pipeline — drift and performance monitoring."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    drift_p = sub.add_parser("drift", help="Run daily feature drift monitoring (Track 1).")
    drift_p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Scoring date to evaluate (default: today).",
    )
    drift_p.add_argument(
        "--data-source",
        choices=["postgres", "csv"],
        default="postgres",
        help="Feature data source: 'postgres' (default) or 'csv' (dev only).",
    )
    drift_p.add_argument("--data-dir", default="data/", help="Relative path to CSV directory.")

    perf_p = sub.add_parser("performance", help="Run monthly performance monitoring (Track 2).")
    perf_p.add_argument(
        "--cohort-month",
        required=True,
        metavar="YYYY-MM",
        help="Cohort month to evaluate (e.g. 2017-03).",
    )
    perf_p.add_argument(
        "--no-retrain",
        action="store_true",
        help="Evaluate performance without triggering retraining on alert.",
    )
    perf_p.add_argument("--data-source", choices=["postgres", "csv"], default="postgres")
    perf_p.add_argument("--data-dir", default="data/")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    if args.command == "drift":
        scoring_date = None
        if args.date:
            if not _re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
                parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
            scoring_date = date.fromisoformat(args.date)
        result = run_drift_monitoring(
            scoring_date=scoring_date,
            data_source=args.data_source,
            data_dir=args.data_dir,
        )
    else:
        if not _re.match(r"^\d{4}-\d{2}$", args.cohort_month):
            parser.error(f"--cohort-month must be YYYY-MM, got {args.cohort_month!r}")
        result = run_performance_monitoring(
            cohort_month=args.cohort_month,
            trigger_retraining=not args.no_retrain,
            data_source=args.data_source,
            data_dir=args.data_dir,
        )

    print(result)
