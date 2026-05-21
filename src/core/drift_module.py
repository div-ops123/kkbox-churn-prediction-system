"""
Pure drift and performance monitoring domain logic for the KKBox monitoring pipeline.

Called by pipelines/monitor.py (Prefect orchestration). No Prefect, no psycopg2,
no file I/O.

Public API
----------
    compute_psi(...)                 → float
    compute_feature_drift(...)       → dict[str, float]
    evaluate_cohort_performance(...) → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_psi(baseline: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI) between baseline and current distributions.

    PSI = Σ (current% − baseline%) × ln(current% / baseline%) over n_bins bins.

    Bin edges are derived from baseline only (anchored to training distribution).
    The rightmost edge is extended to np.inf so current values beyond the training
    range fall into the last bin rather than being silently dropped.

    Zero-proportion bins are replaced with 1e-4 to avoid log(0).

    Args:
        baseline: Series of baseline (training-time) values.
        current:  Series of current (serving-time) values.
        n_bins:   Number of histogram bins (default 10).

    Returns:
        PSI as a float. < 0.1 = stable; 0.1–0.2 = slight shift; > 0.2 = significant drift.
        Returns 0.0 if either series is empty after NaN drop.
    """
    baseline = baseline.dropna()
    current = current.dropna()
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    _, bin_edges = np.histogram(baseline, bins=n_bins)
    bin_edges[-1] = np.inf  # capture current values beyond training-time range

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    baseline_pct = np.where(baseline_pct == 0, 1e-4, baseline_pct)
    current_pct = np.where(current_pct == 0, 1e-4, current_pct)

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def compute_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Compute PSI for each feature column between baseline and current DataFrames.

    Args:
        baseline_df:  DataFrame with training-time feature values.
        current_df:   DataFrame with current serving-time feature values.
        feature_cols: List of column names to compute PSI for.

    Returns:
        Dict mapping feature name → PSI value.

    Raises:
        ValueError: If any requested column is missing from either DataFrame.
    """
    missing_baseline = [c for c in feature_cols if c not in baseline_df.columns]
    missing_current = [c for c in feature_cols if c not in current_df.columns]
    if missing_baseline:
        raise ValueError(f"Columns missing from baseline_df: {missing_baseline}")
    if missing_current:
        raise ValueError(f"Columns missing from current_df: {missing_current}")

    return {col: compute_psi(baseline_df[col], current_df[col]) for col in feature_cols}


def evaluate_cohort_performance(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    top_k: int = 100,
) -> dict:
    """
    Evaluate model performance using pre-computed scores from the predictions table.

    Inner-joins predictions_df and labels_df on msno. Cohort filtering (by
    expiry_date / anchor_expiry_date month) must be done upstream before calling
    this function.

    Args:
        predictions_df: DataFrame with columns (msno, score).
        labels_df:      DataFrame with columns (msno, is_churn).
        top_k:          Number of highest-scored users for Precision/Recall@K.

    Returns:
        Dict with keys: auc_pr, auc_roc, precision_at_k, recall_at_k,
        n_total, n_churned, top_k.
        auc_pr and auc_roc are None if no churners appear in the joined cohort
        (metrics are undefined with a single class).

    Raises:
        ValueError: If the inner join on msno produces an empty DataFrame.
            The error message includes sample expiry/anchor_expiry values to
            help diagnose month-alignment mismatches.
    """
    joined = predictions_df[["msno", "score"]].merge(
        labels_df[["msno", "is_churn"]], on="msno", how="inner"
    )
    if len(joined) == 0:
        pred_expiry_sample = predictions_df.get("expiry_date", pd.Series()).head(3).tolist()
        label_anchor_sample = labels_df.get("anchor_expiry_date", pd.Series()).head(3).tolist()
        raise ValueError(
            f"Empty join between predictions and labels on msno. "
            f"predictions expiry sample: {pred_expiry_sample}; "
            f"labels anchor_expiry sample: {label_anchor_sample}. "
            "Check that both DataFrames are filtered to the same cohort month."
        )

    y = joined["is_churn"]
    scores = joined["score"]
    n_churned = int(y.sum())

    auc_pr: float | None = None
    auc_roc: float | None = None
    if n_churned > 0 and n_churned < len(joined):
        auc_pr = float(average_precision_score(y, scores))
        auc_roc = float(roc_auc_score(y, scores))

    top_k_idx = scores.nlargest(top_k).index
    precision_at_k = float(y.loc[top_k_idx].mean())
    recall_at_k = float(y.loc[top_k_idx].sum() / y.sum()) if y.sum() > 0 else 0.0

    return {
        "auc_pr":         auc_pr,
        "auc_roc":        auc_roc,
        "precision_at_k": precision_at_k,
        "recall_at_k":    recall_at_k,
        "n_total":        len(joined),
        "n_churned":      n_churned,
        "top_k":          top_k,
    }
