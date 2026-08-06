"""
Validation pipeline — decides whether a trained candidate beats production.

Consumes the held-out test set from a dataset-build run (the fold the
training pipeline never sees — not used for fitting, not used for early
stopping) plus a specific candidate model version. Scores both the candidate
and the current production model on that same test set and compares.

If the candidate wins, this pipeline sets the "challenger" alias on it. It
never touches the "production" alias — promoting challenger -> production
is a deployment-strategy decision (canary, shadow mode, etc.) that is
explicitly out of scope for this project.

Run locally:
    python src/pipelines/validate.py --dataset-version-id <run_id> --candidate-version <v>

Parameters
----------
dataset_version_id : MLflow run_id of the dataset-build run whose test.parquet
                      to evaluate on.
candidate_version   : MLflow registered model version to evaluate (the one
                      the training pipeline just produced).
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ValidationConfig, load_validation_config
from core.feature_module import FEATURE_COLS
from core.model_loader import ModelNotFoundError, make_model_loader
from core.model_trainer import evaluate_on_holdout


# ── Custom exceptions ──────────────────────────────────────────────────────────

class TestSetLoadError(Exception):
    """Failed to fetch test.parquet for dataset_version_id."""

class CandidateLoadError(Exception):
    """Failed to load the candidate model version from MLflow."""

class EvaluationError(Exception):
    """Scoring the candidate on the test set failed."""

class ChallengerAliasError(Exception):
    """Failed to set the 'challenger' alias in MLflow."""


# ── Prefect tasks ─────────────────────────────────────────────────────────────

@task(name="load-test-set", retries=2, retry_delay_seconds=30, tags=["mlflow", "dataset"])
def load_test_set(dataset_version_id: str, config: ValidationConfig) -> pd.DataFrame:
    """Fetch test.parquet from the dataset-build MLflow run — never seen in training."""
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=dataset_version_id, artifact_path="dataset", dst_path=tmpdir,
            )
            test_df = pd.read_parquet(Path(local_dir) / "test.parquet")
    except Exception as exc:
        raise TestSetLoadError(
            f"Failed to load test.parquet for dataset_version_id={dataset_version_id}: {exc}"
        ) from exc

    if len(test_df) == 0:
        raise TestSetLoadError(
            f"test.parquet for dataset_version_id={dataset_version_id} is empty"
        )

    logger.info(f"load_test_set complete | dataset_version_id={dataset_version_id} | n_test={len(test_df)}")
    return test_df


@task(name="load-candidate-model", retries=1, retry_delay_seconds=30, tags=["mlflow"])
def load_candidate_model(candidate_version: str, config: ValidationConfig):
    """Load the specific candidate model version under evaluation."""
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    model_uri = f"models:/{config.mlflow.model_name}/{candidate_version}"
    try:
        model = mlflow.lightgbm.load_model(model_uri)
    except Exception as exc:
        raise CandidateLoadError(f"Failed to load candidate from {model_uri}: {exc}") from exc

    logger.info(f"load_candidate_model complete | version={candidate_version}")
    return model


@task(name="load-production-model", retries=2, retry_delay_seconds=30, tags=["mlflow"])
def load_production_model_for_comparison(config: ValidationConfig):
    """
    Load the current production model for head-to-head comparison.

    Returns the model object on success, or None if no production model
    exists yet (first validation run always promotes to challenger).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    try:
        loader = make_model_loader(use_mlflow=True, mlflow_config=config.mlflow)
        model = loader.load()
        logger.info(f"load_production_model_for_comparison | version={loader.get_model_version()}")
        return model
    except ModelNotFoundError:
        logger.warning("No production model found in MLflow registry — first run, will promote.")
        return None
    except Exception as exc:
        logger.warning(f"Could not load production model: {exc} — treating as first run.")
        return None


@task(name="evaluate-candidate", retries=0, tags=["evaluation"])
def evaluate_candidate(candidate_model, test_df: pd.DataFrame, config: ValidationConfig) -> dict:
    """Score the candidate on the held-out test set."""
    logger = get_run_logger()
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["is_churn"]
    metrics = evaluate_on_holdout(candidate_model, X_test, y_test, top_k=config.top_k)
    logger.info(f"evaluate_candidate complete | test_auc_pr={metrics['val_auc_pr']:.4f}")
    return metrics


@task(name="compare-and-decide", retries=0, tags=["evaluation"])
def compare_and_decide(
    candidate_metrics: dict,
    production,
    test_df: pd.DataFrame,
    config: ValidationConfig,
) -> tuple[bool, dict | None]:
    """
    Returns (should_promote, production_metrics).

    Compares candidate_metrics against the production model's performance on
    the same test_df the candidate never trained or early-stopped on. If
    production is None (first run) -> always promote.
    """
    logger = get_run_logger()

    if production is None:
        logger.info("compare_and_decide | no production model -> promoting candidate to challenger")
        return True, None

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["is_churn"]

    try:
        production_metrics = evaluate_on_holdout(production, X_test, y_test, top_k=config.top_k)
    except Exception as exc:
        logger.warning(
            f"compare_and_decide | production model evaluation failed: {exc}"
            " — defaulting to promote"
        )
        return True, None

    should_promote = candidate_metrics["val_auc_pr"] > production_metrics["val_auc_pr"]
    logger.info(
        f"compare_and_decide"
        f" | candidate_test_auc_pr={candidate_metrics['val_auc_pr']:.4f}"
        f" | production_test_auc_pr={production_metrics['val_auc_pr']:.4f}"
        f" | promote={should_promote}"
    )
    return should_promote, production_metrics


@task(name="promote-to-challenger", retries=2, retry_delay_seconds=30, tags=["mlflow"])
def promote_to_challenger(candidate_version: str, should_promote: bool, config: ValidationConfig) -> None:
    """
    Set the 'challenger' alias on candidate_version if it won. Never touches
    'production'. If the candidate lost, any existing 'challenger' alias is
    left untouched.
    """
    logger = get_run_logger()
    if not should_promote:
        logger.info("promote_to_challenger | candidate did not beat production — skipped")
        return

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    try:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=config.mlflow.model_name, alias="challenger", version=candidate_version,
        )
    except Exception as exc:
        raise ChallengerAliasError(f"Failed to set 'challenger' alias: {exc}") from exc

    logger.info(f"promote_to_challenger complete | version={candidate_version} | alias=challenger")


@task(name="log-validation-metrics", retries=0, tags=["monitoring"])
def log_validation_metrics(
    candidate_version: str,
    candidate_metrics: dict,
    production_metrics: dict | None,
    promoted: bool,
) -> None:
    """Log validation results — console output only."""
    logger = get_run_logger()
    prod_auc_pr = production_metrics["val_auc_pr"] if production_metrics else 0.0

    metrics = [
        ("test_auc_pr_candidate",    candidate_metrics["val_auc_pr"],     False),
        ("test_auc_roc_candidate",   candidate_metrics["val_auc_roc"],    False),
        ("precision_at_k_candidate", candidate_metrics["precision_at_k"], False),
        ("recall_at_k_candidate",    candidate_metrics["recall_at_k"],    False),
        ("test_auc_pr_production",   prod_auc_pr,                         False),
        ("promoted_to_challenger",   float(int(promoted)),                False),
    ]

    logger.info(
        f"log_validation_metrics complete | metrics={metrics}"
        f" | candidate_version={candidate_version} | promoted={promoted}"
    )


# ── Prefect flow ──────────────────────────────────────────────────────────────

@flow(
    name="kkbox-validation-pipeline",
    description=(
        "Score a candidate model version and the current production model on "
        "the held-out test set from a dataset-build run, and promote the "
        "candidate to the 'challenger' alias if it wins."
    ),
    version="1.0.0",
)
def run_validation_pipeline(dataset_version_id: str, candidate_version: str) -> dict:
    """
    Orchestrate the validation pipeline.

    Parameters
    ----------
    dataset_version_id : MLflow run_id of the dataset-build run to evaluate on.
    candidate_version   : MLflow registered model version to evaluate.

    Returns
    -------
    dict with keys: status, candidate_version, test_auc_pr_candidate,
    test_auc_pr_production, promoted.
    status is "challenger" or "not_promoted".
    """
    logger = get_run_logger()
    config = load_validation_config()

    logger.info(
        f"run_validation_pipeline started | dataset_version_id={dataset_version_id}"
        f" | candidate_version={candidate_version}"
    )

    # ── Step 1: Fetch test set + candidate model ──────────────────────────────
    test_df = load_test_set(dataset_version_id=dataset_version_id, config=config)
    candidate_model = load_candidate_model(candidate_version=candidate_version, config=config)
    candidate_metrics = evaluate_candidate(candidate_model=candidate_model, test_df=test_df, config=config)

    # ── Step 2: Compare against production ────────────────────────────────────
    production = load_production_model_for_comparison(config=config)
    should_promote, production_metrics = compare_and_decide(
        candidate_metrics=candidate_metrics, production=production, test_df=test_df, config=config
    )

    # ── Step 3: Promote to challenger (never touches production) ─────────────
    promote_to_challenger(candidate_version=candidate_version, should_promote=should_promote, config=config)

    # ── Step 4: Health metrics (logged only) ──────────────────────────────────
    log_validation_metrics(
        candidate_version=candidate_version,
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics,
        promoted=should_promote,
    )

    result = {
        "status": "challenger" if should_promote else "not_promoted",
        "candidate_version": candidate_version,
        "test_auc_pr_candidate": round(candidate_metrics["val_auc_pr"], 4),
        "test_auc_pr_production": round(production_metrics["val_auc_pr"], 4) if production_metrics else None,
        "promoted": should_promote,
    }
    logger.info(
        f"run_validation_pipeline complete"
        f" | status={result['status']}"
        f" | candidate_version={result['candidate_version']}"
        f" | test_auc_pr_candidate={result['test_auc_pr_candidate']}"
        f" | promoted={result['promoted']}"
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the KKBox validation pipeline against an already-built dataset."
    )
    parser.add_argument(
        "--dataset-version-id",
        required=True,
        help="MLflow run_id of the dataset-build run whose test set to evaluate on.",
    )
    parser.add_argument(
        "--candidate-version",
        required=True,
        help="MLflow registered model version to evaluate.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_validation_pipeline(
        dataset_version_id=args.dataset_version_id,
        candidate_version=args.candidate_version,
    )
    print(result)
