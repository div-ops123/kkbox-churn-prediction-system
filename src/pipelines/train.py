"""
Training pipeline — fits a candidate model on an already-built dataset.

This pipeline does not touch raw data, the labels table, or feature
computation — all of that is the dataset-build pipeline's job
(pipelines/build_dataset.py). This pipeline fetches train.parquet and
val.parquet for a given dataset_version_id (the dataset-build MLflow run_id),
fits LightGBM, and registers the result to MLflow.

It never promotes anything: it always registers a new model version (so it
exists with a version number and can be inspected/loaded), tags it with the
dataset_version_id it was trained on, and stops there. Deciding whether this
candidate beats production and should become the "challenger" is the
validation pipeline's job (pipelines/validate.py) — not this one.

Run locally:
    python src/pipelines/build_dataset.py --cohort-months 2017-03   # first, to get a dataset_version_id
    python src/pipelines/train.py --dataset-version-id <run_id>

Parameters
----------
dataset_version_id    : MLflow run_id of the dataset-build run to train on.
n_estimators          : LightGBM boosting rounds ceiling (early stopping applies).
early_stopping_rounds : LightGBM early stopping patience.
"""

from __future__ import annotations

import logging
import sys
import tempfile
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
from core.model_trainer import TrainArtifact, _auto_scale_pos_weight, train_lgbm


# ── Custom exceptions ──────────────────────────────────────────────────────────

class TrainingConfigError(Exception):
    """PostgreSQL connection failed or config is invalid."""

class DatasetLoadError(Exception):
    """Failed to fetch the dataset-build artifacts for dataset_version_id."""

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

@task(name="load-dataset-artifacts", retries=2, retry_delay_seconds=30, tags=["mlflow", "dataset"])
def load_dataset_artifacts(
    dataset_version_id: str, config: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fetch train.parquet, val.parquet, and feature_config.json (for p99_secs)
    from the dataset-build MLflow run identified by dataset_version_id.

    Returns (train_df, val_df, p99_secs).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=dataset_version_id, artifact_path="dataset", dst_path=tmpdir,
            )
            train_df = pd.read_parquet(Path(local_dir) / "train.parquet")
            val_df = pd.read_parquet(Path(local_dir) / "val.parquet")
            import json
            with open(Path(local_dir) / "feature_config.json") as f:
                p99_secs = json.load(f)["p99_secs"]
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load dataset artifacts for dataset_version_id={dataset_version_id}: {exc}"
        ) from exc

    if len(train_df) == 0:
        raise DatasetLoadError(
            f"train.parquet for dataset_version_id={dataset_version_id} is empty"
        )

    logger.info(
        f"load_dataset_artifacts complete | dataset_version_id={dataset_version_id}"
        f" | n_train={len(train_df)} | n_val={len(val_df)} | p99_secs={p99_secs:,.1f}"
    )
    return train_df, val_df, p99_secs


@task(name="train-candidate-model", retries=0, tags=["training"])
def train_candidate_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    p99_secs: float,
    config: TrainingConfig,
) -> TrainArtifact:
    """
    Fit LightGBM on train_df, early-stopping against val_df.

    retries=0 — training is deterministic; a crash is a code bug.
    """
    logger = get_run_logger()

    try:
        scale_pos = _auto_scale_pos_weight(train_df["is_churn"])
        logger.info(
            f"train_candidate_model | n_train={len(train_df)} | n_val={len(val_df)}"
            f" | scale_pos_weight={scale_pos:.2f}"
        )
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
    return artifact


@task(name="register-to-mlflow", retries=2, retry_delay_seconds=30, tags=["mlflow"])
def register_to_mlflow(
    candidate: TrainArtifact,
    dataset_version_id: str,
    config: TrainingConfig,
) -> tuple[str, str]:
    """
    Log params, metrics, and the model artifact to MLflow, then register a
    new model version tagged with dataset_version_id.

    Never sets an alias — this pipeline only produces an unpromoted,
    inspectable model version. Whether it becomes the "challenger" is decided
    by the validation pipeline (pipelines/validate.py).

    Returns (run_id, model_version).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment("kkbox-churn-training")

    try:
        with mlflow.start_run(run_name="training_pipeline") as run:
            mlflow.log_params({
                "dataset_version_id":     dataset_version_id,
                "p99_secs":               candidate.p99_secs,
                "scale_pos_weight":       round(candidate.scale_pos_weight, 4),
                "n_train":                candidate.n_train,
                "n_val":                  candidate.n_val,
                "n_estimators":           config.n_estimators,
                "early_stopping_rounds":  config.early_stopping_rounds,
            })
            mlflow.log_metrics({
                "val_auc_pr":         candidate.val_auc_pr,
                "val_auc_roc":        candidate.val_auc_roc,
                "precision_at_100":   candidate.precision_at_k,
                "recall_at_100":      candidate.recall_at_k,
                "churn_rate_train":   candidate.churn_rate_train,
            })
            mlflow.lightgbm.log_model(candidate.booster, artifact_path="model")
            run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=config.mlflow.model_name)

        client = MlflowClient()
        client.set_model_version_tag(
            name=mv.name, version=mv.version,
            key="dataset_version_id", value=dataset_version_id,
        )
        logger.info(
            f"register_to_mlflow complete | version={mv.version}"
            f" | dataset_version_id={dataset_version_id} | run_id={run_id}"
        )
    except MLflowRegistrationError:
        raise
    except Exception as exc:
        raise MLflowRegistrationError(f"MLflow registration failed: {exc}") from exc

    return run_id, mv.version


@task(name="write-training-metrics", retries=1, retry_delay_seconds=30, tags=["db", "monitoring"])
def write_training_metrics(
    candidate: TrainArtifact,
    run_id: str,
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


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-training-pipeline",
    description=(
        "Fit a candidate LightGBM model on an already-built dataset "
        "(see pipelines/build_dataset.py) and register it to MLflow, "
        "unpromoted."
    ),
    version="2.0.0",
)
def run_training_pipeline(
    dataset_version_id: str,
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = None,
) -> dict:
    """
    Orchestrate the training pipeline.

    Parameters
    ----------
    dataset_version_id    : MLflow run_id of the dataset-build run to train on.
    n_estimators          : override for TRAINING_N_ESTIMATORS env var.
    early_stopping_rounds : override for TRAINING_EARLY_STOPPING_ROUNDS env var.

    Returns
    -------
    dict with keys: status, run_id, model_version, dataset_version_id,
    val_auc_pr, n_train, n_val.
    status is "registered" or "partial" (metrics write failed).
    """
    logger = get_run_logger()
    config = load_training_config(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )

    logger.info(
        f"run_training_pipeline started | dataset_version_id={dataset_version_id}"
    )

    # ── Step 1: Fetch dataset ──────────────────────────────────────────────────
    train_df, val_df, p99_secs = load_dataset_artifacts(
        dataset_version_id=dataset_version_id, config=config
    )

    # ── Step 2: Train candidate ───────────────────────────────────────────────
    candidate = train_candidate_model(
        train_df=train_df, val_df=val_df, p99_secs=p99_secs, config=config
    )

    # ── Step 3: Register to MLflow (never promoted) ──────────────────────────
    run_id, model_version = register_to_mlflow(
        candidate=candidate, dataset_version_id=dataset_version_id, config=config
    )

    # ── Step 4: Health metrics (non-fatal) ────────────────────────────────────
    status = "registered"
    try:
        write_training_metrics(candidate=candidate, run_id=run_id, config=config)
    except TrainingMetricsWriteError as exc:
        logger.error(f"Training metrics write failed (non-fatal): {exc}")
        status = "partial"

    result = {
        "status": status,
        "run_id": run_id,
        "model_version": model_version,
        "dataset_version_id": dataset_version_id,
        "val_auc_pr": round(candidate.val_auc_pr, 4),
        "n_train": candidate.n_train,
        "n_val": candidate.n_val,
    }
    logger.info(
        f"run_training_pipeline complete"
        f" | status={result['status']}"
        f" | model_version={result['model_version']}"
        f" | val_auc_pr={result['val_auc_pr']}"
        f" | run_id={result['run_id']}"
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the KKBox training pipeline against an already-built dataset."
    )
    parser.add_argument(
        "--dataset-version-id",
        required=True,
        help="MLflow run_id of the dataset-build run to train on.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override TRAINING_N_ESTIMATORS env var.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="Override TRAINING_EARLY_STOPPING_ROUNDS env var.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_training_pipeline(
        dataset_version_id=args.dataset_version_id,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(result)
