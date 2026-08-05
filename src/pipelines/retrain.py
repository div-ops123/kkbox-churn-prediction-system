"""
Retrain-and-validate orchestrator — thin caller, not a pipeline of its own.

Wires the three independent pipelines together for the common "retrain, then
maybe promote to challenger" path: monthly cron, or a monitoring-triggered
retrain (see pipelines/monitor.py's maybe_trigger_retraining).

Each sub-pipeline stays fully independent and separately invocable — this
file exists only so the common case is a single call instead of three manual
ones. Re-running validation alone against an older candidate, or re-running
training against an existing dataset_version_id, still works without
touching this file.

Run locally:
    python src/pipelines/retrain.py --cohort-months 2017-03
    python src/pipelines/retrain.py --data-source csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from prefect import flow, get_run_logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.build_dataset import run_dataset_build_pipeline
from pipelines.train import run_training_pipeline
from pipelines.validate import run_validation_pipeline


@flow(
    name="kkbox-retrain-and-validate",
    description=(
        "Thin orchestrator: build a fresh dataset, train a candidate on it, "
        "then validate the candidate against production and promote to "
        "'challenger' if it wins. Each step is a fully independent pipeline."
    ),
    version="1.0.0",
)
def run_retrain_and_validate_pipeline(
    cohort_months: list[str] | None = None,
    data_source: str = "postgres",
    data_dir: str = "data/",
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = None,
) -> dict:
    """
    Orchestrate dataset-build -> training -> validation in sequence.

    Parameters
    ----------
    cohort_months          : "YYYY-MM" strings to build the dataset from;
                              None / [] uses all labeled months.
    data_source             : "postgres" (default) or "csv" (dev only).
    data_dir                : relative path to the CSV directory.
    n_estimators            : override for TRAINING_N_ESTIMATORS env var.
    early_stopping_rounds   : override for TRAINING_EARLY_STOPPING_ROUNDS env var.

    Returns
    -------
    dict with keys: status, dataset, training, validation.
    status is "empty_cohort" (no labeled rows to build from) or "completed"
    (dataset build + training + validation all ran, regardless of whether
    the candidate was promoted to challenger — check validation["promoted"]
    for that).
    """
    logger = get_run_logger()

    dataset = run_dataset_build_pipeline(
        cohort_months=cohort_months, data_source=data_source, data_dir=data_dir,
    )
    if dataset["dataset_version_id"] is None:
        logger.warning("run_retrain_and_validate_pipeline | empty cohort — skipping training/validation")
        return {"status": "empty_cohort", "dataset": dataset, "training": None, "validation": None}

    training = run_training_pipeline(
        dataset_version_id=dataset["dataset_version_id"],
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    validation = run_validation_pipeline(
        dataset_version_id=dataset["dataset_version_id"],
        candidate_version=training["model_version"],
    )

    logger.info(
        f"run_retrain_and_validate_pipeline complete"
        f" | dataset_version_id={dataset['dataset_version_id']}"
        f" | model_version={training['model_version']}"
        f" | promoted={validation['promoted']}"
    )
    return {
        "status": "completed",
        "dataset": dataset,
        "training": training,
        "validation": validation,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the KKBox retrain-and-validate orchestrator."
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
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    result = run_retrain_and_validate_pipeline(
        cohort_months=args.cohort_months,
        data_source=args.data_source,
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(result)
