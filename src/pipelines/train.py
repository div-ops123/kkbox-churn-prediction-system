"""
Production training pipeline — retrains from the derived labels table.

Reads churn labels from PostgreSQL (written by pipelines/label.py), builds
features via the shared feature_module, trains LightGBM, compares against the
current production model, and promotes if the candidate is better.

Triggered two ways:
  1. Monthly cron — deploy with CronSchedule("0 9 2 * *") so it runs on the
     2nd of each month, after the labeling pipeline runs on the 1st.
  2. Monitoring event — the monitoring pipeline calls run_training_pipeline()
     directly when AUC-PR falls below threshold (Option A: direct call).

Run locally:
    python src/pipelines/train.py --cohort-months 2017-03
    python src/pipelines/train.py --data-source csv --no-promote

Parameters
----------
cohort_months    : one or more "YYYY-MM" strings; omit to use all labeled months.
data_source      : "postgres" (default) | "csv" (dev/testing only).
data_dir         : relative path to the CSV directory (default "data/").
promote_if_better: promote the candidate to "production" alias in MLflow if it
                   beats the current production model on the validation set.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator

import mlflow
import mlflow.lightgbm
import pandas as pd
import psycopg2
import psycopg2.extras
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TrainingConfig, load_training_config
from core.feature_module import FEATURE_COLS, build_features
from core.model_loader import ModelNotFoundError, make_model_loader
from core.model_trainer import (
    TrainArtifact,
    _auto_scale_pos_weight,
    compute_p99_secs,
    evaluate_on_holdout,
    split_cohort,
    train_lgbm,
)


# ── Custom exceptions ──────────────────────────────────────────────────────────

class TrainingConfigError(Exception):
    """PostgreSQL connection failed or config is invalid."""

class LabelLoadError(Exception):
    """Failed to load labels from the PostgreSQL labels table."""

class FeatureBuildError(Exception):
    """Feature computation via feature_module failed."""

class ModelTrainingError(Exception):
    """LightGBM training or evaluation failed."""

class MLflowRegistrationError(Exception):
    """MLflow logging or model registration failed."""

class TrainingMetricsWriteError(Exception):
    """Write to monitoring_metrics failed — non-fatal."""


# ── DB context manager ────────────────────────────────────────────────────────

@contextmanager
def _pg_conn(config: TrainingConfig) -> Generator[psycopg2.extensions.connection, None, None]:
    conn = None
    try:
        conn = psycopg2.connect(config.postgres.dsn)
        yield conn
    except psycopg2.OperationalError as exc:
        raise TrainingConfigError(f"Cannot connect to PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="load-label-cohort", retries=3, retry_delay_seconds=30, tags=["db", "labels"])
def load_label_cohort(config: TrainingConfig) -> pd.DataFrame:
    """
    Query the labels table for training rows.

    If config.cohort_months is non-empty, only those months are loaded;
    otherwise all available labeled rows are returned.

    Returns DataFrame[msno, anchor_expiry_date, is_churn].
    """
    logger = get_run_logger()

    if config.cohort_months:
        placeholders = ", ".join(["%s"] * len(config.cohort_months))
        sql = f"""
            SELECT msno, anchor_expiry_date, is_churn
            FROM labels
            WHERE TO_CHAR(anchor_expiry_date, 'YYYY-MM') IN ({placeholders})
            ORDER BY anchor_expiry_date
        """
        params = list(config.cohort_months)
    else:
        sql = "SELECT msno, anchor_expiry_date, is_churn FROM labels ORDER BY anchor_expiry_date"
        params = []

    try:
        with _pg_conn(config) as conn:
            df = pd.read_sql(sql, conn, params=params)
    except TrainingConfigError:
        raise
    except Exception as exc:
        raise LabelLoadError(f"Failed to load labels: {exc}") from exc

    logger.info(
        f"load_label_cohort complete | rows={len(df)}"
        f" | cohort_months={list(config.cohort_months) or 'all'}"
    )
    return df


@task(name="compute-p99-and-build-features", retries=0, tags=["features"])
def compute_p99_and_build_features(
    cohort_df: pd.DataFrame, config: TrainingConfig
) -> pd.DataFrame:
    """
    Compute p99_secs from this cohort, persist to feature_config.json, then
    call build_features() from the shared feature_module.

    Returns DataFrame indexed by msno with FEATURE_COLS + [is_churn, cohort_month].
    cohort_month is used by split_cohort() for time-based train/val splitting.
    """
    logger = get_run_logger()

    msno_list = cohort_df["msno"].tolist()
    expire_dates = cohort_df["anchor_expiry_date"].tolist()
    pg_conn_str = config.postgres.conn_str if config.data_source == "postgres" else None

    try:
        p99 = compute_p99_secs(
            msno_list, expire_dates,
            data_source=config.data_source,
            data_dir=config.data_dir,
            pg_conn_str=pg_conn_str,
        )
    except Exception as exc:
        raise FeatureBuildError(f"compute_p99_secs failed: {exc}") from exc

    config.feature_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.feature_config_path, "w") as f:
        json.dump({"p99_secs": round(p99, 6)}, f, indent=2)
    logger.info(f"feature_config.json updated | p99_secs={p99:,.1f}")

    try:
        features_df = build_features(
            msno_list, expire_dates,
            p99_secs=p99,
            data_source=config.data_source,
            data_dir=config.data_dir,
            pg_conn_str=pg_conn_str,
        )
    except Exception as exc:
        raise FeatureBuildError(f"build_features() failed: {exc}") from exc

    config.baseline_features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df[FEATURE_COLS].to_parquet(config.baseline_features_path, index=True)
    logger.info(f"baseline_features.parquet saved | rows={len(features_df)} | path={config.baseline_features_path}")

    cohort_meta = cohort_df.set_index("msno")[["is_churn"]].copy()
    cohort_meta["cohort_month"] = (
        pd.to_datetime(cohort_df.set_index("msno")["anchor_expiry_date"])
        .dt.strftime("%Y-%m")
    )
    result = features_df.join(cohort_meta, how="left")

    null_feature_pct = result[FEATURE_COLS].isna().mean().mean() * 100
    logger.info(
        f"compute_p99_and_build_features complete"
        f" | rows={len(result)} | avg_null_pct={null_feature_pct:.1f}%"
    )
    return result


@task(name="train-candidate-model", retries=0, tags=["training"])
def train_candidate_model(
    features_df: pd.DataFrame, config: TrainingConfig
) -> tuple[TrainArtifact, pd.DataFrame]:
    """
    Split features_df into train/val, then train LightGBM.

    Returns (TrainArtifact, val_df) — val_df is forwarded to compare_and_decide
    so the production model is evaluated on the same holdout.

    retries=0 — training is deterministic; a crash is a code bug.
    """
    logger = get_run_logger()

    try:
        train_df, val_df = split_cohort(features_df, val_fraction=config.val_fraction)

        scale_pos = _auto_scale_pos_weight(train_df["is_churn"])
        logger.info(
            f"train_candidate_model | n_train={len(train_df)} | n_val={len(val_df)}"
            f" | scale_pos_weight={scale_pos:.2f}"
        )

        with open(config.feature_config_path) as f:
            p99_secs = json.load(f)["p99_secs"]

        artifact = train_lgbm(
            train_df, val_df,
            scale_pos_weight=scale_pos,
            n_estimators=config.n_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
            p99_secs=p99_secs,
        )
    except Exception as exc:
        raise ModelTrainingError(f"LightGBM training failed: {exc}") from exc

    logger.info(
        f"train_candidate_model complete"
        f" | val_auc_pr={artifact.val_auc_pr:.4f}"
        f" | val_auc_roc={artifact.val_auc_roc:.4f}"
        f" | precision@100={artifact.precision_at_k:.4f}"
    )
    return artifact, val_df


@task(name="load-production-model", retries=2, retry_delay_seconds=30, tags=["mlflow"])
def load_production_model_for_comparison(config: TrainingConfig):
    """
    Load the current production model from MLflow for head-to-head comparison.

    Returns the model object on success, or None if no production model exists
    yet (first training run always promotes).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    try:
        loader = make_model_loader(
            use_mlflow=True,
            mlflow_config=config.mlflow,
        )
        model = loader.load()
        logger.info(
            f"load_production_model_for_comparison | version={loader.get_model_version()}"
        )
        return model
    except ModelNotFoundError:
        logger.warning(
            "No production model found in MLflow registry — first run, will promote."
        )
        return None
    except Exception as exc:
        logger.warning(f"Could not load production model: {exc} — treating as first run.")
        return None


@task(name="compare-and-decide", retries=0, tags=["evaluation"])
def compare_and_decide(
    candidate: TrainArtifact,
    production,
    val_df: pd.DataFrame,
    config: TrainingConfig,
) -> bool:
    """
    Returns True if the candidate should be promoted.

    Compares candidate.val_auc_pr against the production model's AUC-PR on
    the same val_df. If production is None (first run) → always True.
    """
    logger = get_run_logger()

    if production is None:
        logger.info("compare_and_decide | no production model → promoting candidate")
        return True

    X_val = val_df[FEATURE_COLS]
    y_val = val_df["is_churn"]

    try:
        prod_metrics = evaluate_on_holdout(production, X_val, y_val, top_k=100)
    except Exception as exc:
        logger.warning(
            f"compare_and_decide | production model evaluation failed: {exc}"
            " — defaulting to promote"
        )
        return True

    prod_auc_pr = prod_metrics["val_auc_pr"]
    should_promote = candidate.val_auc_pr > prod_auc_pr

    logger.info(
        f"compare_and_decide"
        f" | candidate_auc_pr={candidate.val_auc_pr:.4f}"
        f" | production_auc_pr={prod_auc_pr:.4f}"
        f" | promote={should_promote}"
    )
    return should_promote


@task(name="register-to-mlflow", retries=2, retry_delay_seconds=30, tags=["mlflow"])
def register_to_mlflow(
    candidate: TrainArtifact,
    promote: bool,
    config: TrainingConfig,
) -> str:
    """
    Log params, metrics, and the model artifact to MLflow. Optionally promote.

    Logs feature_config.json as an artifact so p99_secs is versioned alongside
    the model — serving loads whichever p99_secs matches its production model.

    Returns the MLflow run_id.
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment("kkbox-churn-retraining")

    try:
        with mlflow.start_run(run_name="training_pipeline") as run:
            mlflow.log_params({
                "p99_secs":               candidate.p99_secs,
                "scale_pos_weight":       round(candidate.scale_pos_weight, 4),
                "n_train":                candidate.n_train,
                "n_val":                  candidate.n_val,
                "val_fraction":           config.val_fraction,
                "n_estimators":           config.n_estimators,
                "early_stopping_rounds":  config.early_stopping_rounds,
                "data_source":            config.data_source,
                "cohort_months":          ",".join(config.cohort_months) or "all",
            })
            mlflow.log_metrics({
                "val_auc_pr":         candidate.val_auc_pr,
                "val_auc_roc":        candidate.val_auc_roc,
                "precision_at_100":   candidate.precision_at_k,
                "recall_at_100":      candidate.recall_at_k,
                "churn_rate_train":   candidate.churn_rate_train,
            })
            mlflow.lightgbm.log_model(candidate.booster, artifact_path="model")
            if config.feature_config_path.exists():
                mlflow.log_artifact(
                    str(config.feature_config_path),
                    artifact_path="feature_config",
                )
            run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=config.mlflow.model_name)

        if promote:
            client = MlflowClient()
            # Preserve the current production model as "stable" before promoting
            # the new candidate — gives a named rollback target without deleting.
            try:
                old_prod = client.get_model_version_by_alias(
                    name=mv.name, alias=config.mlflow.model_alias
                )
                client.set_registered_model_alias(
                    name=mv.name,
                    alias="stable",
                    version=old_prod.version,
                )
                logger.info(
                    f"register_to_mlflow | archived old production as 'stable'"
                    f" | version={old_prod.version}"
                )
            except Exception:
                pass  # No previous production model to archive — first run
            client.set_registered_model_alias(
                name=mv.name,
                alias=config.mlflow.model_alias,
                version=mv.version,
            )
            logger.info(
                f"register_to_mlflow | promoted version={mv.version}"
                f" | alias='{config.mlflow.model_alias}' | run_id={run_id}"
            )
        else:
            logger.info(
                f"register_to_mlflow | logged version={mv.version}"
                f" (not promoted) | run_id={run_id}"
            )
    except MLflowRegistrationError:
        raise
    except Exception as exc:
        raise MLflowRegistrationError(f"MLflow registration failed: {exc}") from exc

    return run_id


@task(name="write-training-metrics", retries=1, retry_delay_seconds=30, tags=["db", "monitoring"])
def write_training_metrics(
    candidate: TrainArtifact,
    run_id: str,
    promoted: bool,
    config: TrainingConfig,
) -> None:
    """
    Write training pipeline health metrics to the monitoring_metrics table.

    Non-fatal: the task raises TrainingMetricsWriteError on failure; the flow
    catches it, logs WARNING, and continues with status="partial".

    Alerts when val_auc_pr < 0.45 (model did not improve meaningfully).
    """
    logger = get_run_logger()
    run_date = date.today()
    auc_alert = candidate.val_auc_pr < 0.45

    metrics = [
        ("val_auc_pr",       candidate.val_auc_pr,        auc_alert),
        ("val_auc_roc",      candidate.val_auc_roc,        False),
        ("precision_at_100", candidate.precision_at_k,     False),
        ("recall_at_100",    candidate.recall_at_k,        False),
        ("n_train",          float(candidate.n_train),     False),
        ("n_val",            float(candidate.n_val),       False),
        ("promoted",         float(int(promoted)),         False),
    ]

    rows = [
        (run_date, "training", name, value, alert)
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
    except TrainingConfigError:
        raise
    except Exception as exc:
        raise TrainingMetricsWriteError(
            f"Failed to write training metrics: {exc}"
        ) from exc

    alerted = [name for name, _, alert in metrics if alert]
    if alerted:
        logger.warning(f"Training alerts triggered: {alerted}")
    logger.info(
        f"write_training_metrics complete | metrics_written={len(metrics)}"
        f" | alerts={alerted} | run_id={run_id}"
    )


def _write_empty_cohort_metric(config: TrainingConfig) -> None:
    """Write a single monitoring row when the labels table returns no rows. Non-fatal."""
    sql = """
        INSERT INTO monitoring_metrics
            (run_date, pipeline_type, metric_name, metric_value, alert_triggered)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    try:
        with _pg_conn(config) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (date.today(), "training", "cohort_size", 0.0, True))
            conn.commit()
    except Exception:
        pass  # non-fatal; don't mask the empty-cohort early-exit path


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-training-pipeline",
    description=(
        "Production retraining pipeline: load derived labels, build features via "
        "the shared feature_module, train LightGBM, compare against production, "
        "and promote if the candidate is better."
    ),
    version="1.0.0",
)
def run_training_pipeline(
    cohort_months: list[str] | None = None,
    data_source: str = "postgres",
    data_dir: str = "data/",
    promote_if_better: bool = True,
) -> dict:
    """
    Orchestrate the production retraining pipeline.

    Parameters
    ----------
    cohort_months    : "YYYY-MM" strings to train on; None / [] uses all labels.
    data_source      : "postgres" (default) or "csv" (dev only).
    data_dir         : relative path to CSV directory.
    promote_if_better: set False to train + log without touching the production alias.

    Returns
    -------
    dict with keys: status, run_id, val_auc_pr, n_train, n_val.
    status is "promoted", "not_promoted", or "partial" (metrics write failed).
    """
    logger = get_run_logger()
    config = load_training_config(
        cohort_months=cohort_months,
        data_source=data_source,
        data_dir=data_dir,
    )

    logger.info(
        "run_training_pipeline started"
        f" | cohort_months={list(config.cohort_months) or 'all'}"
        f" | data_source={data_source}"
        f" | promote_if_better={promote_if_better}"
    )

    # ── Step 1: Labels ────────────────────────────────────────────────────────
    cohort_df = load_label_cohort(config=config)

    if len(cohort_df) == 0:
        logger.warning("Empty label cohort — no training rows found. Exiting early.")
        _write_empty_cohort_metric(config)
        return {
            "status": "empty_cohort",
            "run_id": None,
            "val_auc_pr": None,
            "n_train": 0,
            "n_val": 0,
        }

    # ── Step 2: Features ──────────────────────────────────────────────────────
    features_df = compute_p99_and_build_features(cohort_df=cohort_df, config=config)

    # ── Step 3: Train candidate ───────────────────────────────────────────────
    candidate, val_df = train_candidate_model(features_df=features_df, config=config)

    # ── Step 4: Compare against production ───────────────────────────────────
    production = load_production_model_for_comparison(config=config)
    should_promote = compare_and_decide(
        candidate=candidate, production=production, val_df=val_df, config=config
    )

    # ── Step 5: Register to MLflow ────────────────────────────────────────────
    effective_promote = should_promote and promote_if_better
    run_id = register_to_mlflow(
        candidate=candidate, promote=effective_promote, config=config
    )

    # ── Step 6: Health metrics (non-fatal) ────────────────────────────────────
    status = "promoted" if effective_promote else "not_promoted"
    try:
        write_training_metrics(
            candidate=candidate,
            run_id=run_id,
            promoted=effective_promote,
            config=config,
        )
    except TrainingMetricsWriteError as exc:
        logger.error(f"Training metrics write failed (non-fatal): {exc}")
        status = "partial"

    result = {
        "status": status,
        "run_id": run_id,
        "val_auc_pr": round(candidate.val_auc_pr, 4),
        "n_train": candidate.n_train,
        "n_val": candidate.n_val,
    }
    logger.info(
        f"run_training_pipeline complete"
        f" | status={result['status']}"
        f" | val_auc_pr={result['val_auc_pr']}"
        f" | n_train={result['n_train']}"
        f" | run_id={result['run_id']}"
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the KKBox production retraining pipeline."
    )
    parser.add_argument(
        "--cohort-months",
        nargs="+",
        metavar="YYYY-MM",
        default=None,
        help="One or more cohort months (e.g. 2017-03 2017-04). Omit for all available.",
    )
    parser.add_argument(
        "--data-source",
        choices=["postgres", "csv"],
        default="postgres",
        help="DuckDB read source for features: 'postgres' (default) or 'csv' (dev only).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Relative path to the CSV directory (default: data/).",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Train and log to MLflow but do not update the 'production' alias.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_training_pipeline(
        cohort_months=args.cohort_months,
        data_source=args.data_source,
        data_dir=args.data_dir,
        promote_if_better=not args.no_promote,
    )
    print(result)
