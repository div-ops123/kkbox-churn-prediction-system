"""
Dataset-build pipeline — Extract the labeled cohort, Transform it via the
shared feature_module, Load train/val/test/baseline datasets to MLflow.

This is the only place a training dataset gets built. The training pipeline
and the validation pipeline both consume its output by dataset_version_id
(the MLflow run_id of the "dataset_build" run) instead of building features
themselves — this pipeline never trains or evaluates a model.

Rolling 3-way split: the most recent labeled month becomes the test set
(held out entirely — never used for training or early stopping, consumed
only by the validation pipeline's promotion decision), the second-most-recent
becomes val (used by the training pipeline's early stopping), everything
older becomes train.

Run locally:
    python src/pipelines/build_dataset.py --cohort-months 2017-01 2017-02 2017-03
    python src/pipelines/build_dataset.py --data-source csv

Parameters
----------
cohort_months : one or more "YYYY-MM" strings; omit to use all labeled months.
data_source   : "postgres" (default) | "csv" (dev/testing only).
data_dir      : relative path to the CSV directory (default "data/").
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import mlflow
import pandas as pd
import psycopg2
from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DatasetBuildConfig, load_dataset_build_config
from core.feature_module import FEATURE_COLS, build_features
from core.model_trainer import compute_p99_secs, split_cohort_3way


# ── Custom exceptions ──────────────────────────────────────────────────────────

class DatasetConfigError(Exception):
    """PostgreSQL connection failed or config is invalid."""

class LabelLoadError(Exception):
    """Failed to load labels from the PostgreSQL labels table."""

class FeatureBuildError(Exception):
    """Feature computation via feature_module failed."""

class DatasetRegistrationError(Exception):
    """MLflow logging of the dataset artifacts failed."""


# ── DB context manager ────────────────────────────────────────────────────────

@contextmanager
def _pg_conn(config: DatasetBuildConfig) -> Generator[psycopg2.extensions.connection, None, None]:
    conn = None
    try:
        conn = psycopg2.connect(config.postgres.dsn)
        yield conn
    except psycopg2.OperationalError as exc:
        raise DatasetConfigError(f"Cannot connect to PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="load-label-cohort", retries=3, retry_delay_seconds=30, tags=["db", "labels"])
def load_label_cohort(config: DatasetBuildConfig) -> pd.DataFrame:
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
    except DatasetConfigError:
        raise
    except Exception as exc:
        raise LabelLoadError(f"Failed to load labels: {exc}") from exc

    logger.info(
        f"load_label_cohort complete | rows={len(df)}"
        f" | cohort_months={list(config.cohort_months) or 'all'}"
    )
    return df


@task(name="build-cohort-features", retries=0, tags=["features"])
def build_cohort_features(
    cohort_df: pd.DataFrame, config: DatasetBuildConfig
) -> tuple[pd.DataFrame, float]:
    """
    Compute p99_secs from this cohort, then call build_features() from the
    shared feature_module.

    Returns (result, p99_secs) where result is indexed by msno with
    FEATURE_COLS + [is_churn, cohort_month]. cohort_month is used by
    split_cohort_3way() for the rolling time-based split.
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
    logger.info(f"build_cohort_features | p99_secs={p99:,.1f}")

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

    cohort_meta = cohort_df.set_index("msno")[["is_churn"]].copy()
    cohort_meta["cohort_month"] = (
        pd.to_datetime(cohort_df.set_index("msno")["anchor_expiry_date"])
        .dt.strftime("%Y-%m")
    )
    result = features_df.join(cohort_meta, how="left")

    null_feature_pct = result[FEATURE_COLS].isna().mean().mean() * 100
    logger.info(
        f"build_cohort_features complete"
        f" | rows={len(result)} | avg_null_pct={null_feature_pct:.1f}%"
    )
    return result, p99


@task(name="split-and-persist-dataset", retries=0, tags=["mlflow", "dataset"])
def split_and_persist_dataset(
    features_df: pd.DataFrame,
    p99_secs: float,
    config: DatasetBuildConfig,
) -> dict:
    """
    Split features_df into (train, val, test) via the rolling time-based
    split, then log train.parquet / val.parquet / test.parquet /
    baseline_features.parquet / feature_config.json as artifacts of a
    dedicated "dataset_build" MLflow run.

    baseline_features.parquet = train + val only (never test) — it must
    represent what a model trained on this dataset actually saw, for
    monitor.py's drift check to compare serving distributions against.

    Returns dict with dataset_version_id (the run_id) and split sizes.
    """
    logger = get_run_logger()

    train_df, val_df, test_df = split_cohort_3way(
        features_df,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    baseline_df = pd.concat([train_df, val_df])[FEATURE_COLS]

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment("kkbox-churn-dataset-build")

    try:
        with mlflow.start_run(run_name="dataset_build") as run:
            mlflow.log_params({
                "cohort_months": ",".join(config.cohort_months) or "all",
                "data_source":   config.data_source,
                "n_train":       len(train_df),
                "n_val":         len(val_df),
                "n_test":        len(test_df),
                "p99_secs":      round(p99_secs, 6),
            })
            mlflow.log_metrics({
                "churn_rate_train": float(train_df["is_churn"].mean()),
                "churn_rate_val":   float(val_df["is_churn"].mean()),
                "churn_rate_test":  float(test_df["is_churn"].mean()),
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                train_df.to_parquet(tmp / "train.parquet", index=True)
                val_df.to_parquet(tmp / "val.parquet", index=True)
                test_df.to_parquet(tmp / "test.parquet", index=True)
                baseline_df.to_parquet(tmp / "baseline_features.parquet", index=True)
                with open(tmp / "feature_config.json", "w") as f:
                    json.dump({"p99_secs": round(p99_secs, 6)}, f, indent=2)

                mlflow.log_artifacts(str(tmp), artifact_path="dataset")

            dataset_version_id = run.info.run_id
    except Exception as exc:
        raise DatasetRegistrationError(f"Failed to log dataset artifacts: {exc}") from exc

    logger.info(
        f"split_and_persist_dataset complete | dataset_version_id={dataset_version_id}"
        f" | n_train={len(train_df)} | n_val={len(val_df)} | n_test={len(test_df)}"
    )
    return {
        "dataset_version_id": dataset_version_id,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "p99_secs": p99_secs,
    }


@task(name="log-dataset-metrics", retries=0, tags=["monitoring"])
def log_dataset_metrics(dataset_result: dict) -> None:
    """Log dataset-build health metrics (n_train/n_val/n_test) — console output only."""
    logger = get_run_logger()
    logger.info(
        f"log_dataset_metrics | n_train={dataset_result['n_train']}"
        f" | n_val={dataset_result['n_val']} | n_test={dataset_result['n_test']}"
    )


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-dataset-build-pipeline",
    description=(
        "Extract the labeled cohort, transform it via the shared feature_module, "
        "and load a rolling train/val/test split plus a drift baseline to MLflow."
    ),
    version="1.0.0",
)
def run_dataset_build_pipeline(
    cohort_months: list[str] | None = None,
    data_source: str = "postgres",
    data_dir: str = "data/",
) -> dict:
    """
    Orchestrate the dataset-build pipeline.

    Parameters
    ----------
    cohort_months : "YYYY-MM" strings to build from; None / [] uses all labels.
    data_source   : "postgres" (default) or "csv" (dev only).
    data_dir      : relative path to the CSV directory.

    Returns
    -------
    dict with keys: status, dataset_version_id, n_train, n_val, n_test.
    status is "built" or "empty_cohort".
    """
    logger = get_run_logger()
    config = load_dataset_build_config(
        cohort_months=cohort_months,
        data_source=data_source,
        data_dir=data_dir,
    )

    logger.info(
        "run_dataset_build_pipeline started"
        f" | cohort_months={list(config.cohort_months) or 'all'}"
        f" | data_source={data_source}"
    )

    # ── Step 1: Labels ────────────────────────────────────────────────────────
    cohort_df = load_label_cohort(config=config)

    if len(cohort_df) == 0:
        logger.warning("Empty label cohort — no rows found. Exiting early.")
        return {
            "status": "empty_cohort",
            "dataset_version_id": None,
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
        }

    # ── Step 2: Features ──────────────────────────────────────────────────────
    features_df, p99_secs = build_cohort_features(cohort_df=cohort_df, config=config)

    # ── Step 3: Split + persist to MLflow ─────────────────────────────────────
    dataset_result = split_and_persist_dataset(
        features_df=features_df, p99_secs=p99_secs, config=config
    )

    # ── Step 4: Health metrics (logged only) ──────────────────────────────────
    log_dataset_metrics(dataset_result=dataset_result)

    result = {"status": "built", **dataset_result}
    logger.info(
        f"run_dataset_build_pipeline complete"
        f" | status={result['status']}"
        f" | dataset_version_id={result['dataset_version_id']}"
        f" | n_train={result['n_train']} | n_val={result['n_val']} | n_test={result['n_test']}"
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the KKBox dataset-build pipeline."
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_dataset_build_pipeline(
        cohort_months=args.cohort_months,
        data_source=args.data_source,
        data_dir=args.data_dir,
    )
    print(result)
