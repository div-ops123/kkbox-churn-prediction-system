"""
Pure training domain logic for the KKBox churn retraining pipeline.

Called by pipelines/train.py (Prefect orchestration). No Prefect, no psycopg2,
no file I/O. Each public function opens and closes its own DuckDB connection.

Public API
----------
    TrainArtifact            — dataclass: trained Booster + all metrics frozen at training time
    compute_p99_secs(...)    → float
    split_cohort(...)        → (train_df, val_df)
    train_lgbm(...)          → TrainArtifact
    evaluate_on_holdout(...) → dict
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import duckdb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from core.feature_module import CAT_COLS, FEATURE_COLS


BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainArtifact:
    """All outputs of a single training run, bundled for hand-off to the Prefect task."""

    booster: lgb.Booster
    feature_cols: list[str]    # snapshot of FEATURE_COLS at training time
    cat_cols: list[str]        # snapshot of CAT_COLS at training time
    p99_secs: float            # frozen winsorization threshold — must match serving
    scale_pos_weight: float
    n_train: int
    n_val: int
    churn_rate_train: float
    val_auc_pr: float          # primary metric; used for promotion comparison
    val_auc_roc: float
    precision_at_k: float      # k=100, business-facing metric
    recall_at_k: float


# ── Public functions ──────────────────────────────────────────────────────────

def compute_p99_secs(
    msno_list: Sequence[str],
    expire_dates: Sequence[date | str],
    data_source: str = "csv",
    data_dir: Path | None = None,
    pg_conn_str: str | None = None,
) -> float:
    """
    P99 of total_secs (capped at 86400) for this cohort's user_logs, filtered
    at each user's feature_cutoff_dt = expire_date - 14 days.

    Returns a scalar float. If no rows match (e.g. log-coverage gap), returns
    86400.0 as a safe fallback (the hard cap ceiling).

    Parameters
    ----------
    msno_list    : user IDs in the training cohort
    expire_dates : subscription expiry date per user
    data_source  : "csv" or "postgres"
    data_dir     : CSV directory; defaults to <project_root>/data
    pg_conn_str  : PostgreSQL URI; required when data_source="postgres"
    """
    _dir = Path(data_dir) if data_dir else BASE_DIR / "data"
    _expire = [_as_date(d) for d in expire_dates]

    cohort_df = pd.DataFrame({
        "msno": list(msno_list),
        "feature_cutoff_dt": [(d - timedelta(days=14)).isoformat() for d in _expire],
    })

    con = duckdb.connect()
    try:
        _register_user_logs_source(con, data_source, _dir, pg_conn_str)
        con.register("_p99_cohort", cohort_df)
        con.execute("""
            CREATE TEMP TABLE _p99_cohort_t AS
            SELECT msno, CAST(feature_cutoff_dt AS DATE) AS feature_cutoff_dt
            FROM _p99_cohort
        """)
        row = con.execute("""
            SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY capped_secs)
            FROM (
                SELECT LEAST(l.total_secs, 86400) AS capped_secs
                FROM v_user_logs l
                INNER JOIN _p99_cohort_t c ON l.msno = c.msno
                WHERE STRPTIME(CAST(l.date AS VARCHAR), '%Y%m%d') <= c.feature_cutoff_dt
            )
        """).fetchone()
    finally:
        con.close()

    if row is None or row[0] is None:
        return 86400.0
    return float(row[0])


def split_cohort(
    features_df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features_df into (train_df, val_df).

    Strategy
    --------
    - If features_df has a 'cohort_month' column with >= 2 distinct values:
        time-based split — the most recent cohort_month becomes val.
    - Otherwise:
        stratified random split on is_churn.

    features_df must have an 'is_churn' column. Index is preserved.
    """
    if "cohort_month" in features_df.columns:
        months = sorted(features_df["cohort_month"].dropna().unique())
        if len(months) >= 2:
            val_month = months[-1]
            train_df = features_df[features_df["cohort_month"] != val_month].copy()
            val_df = features_df[features_df["cohort_month"] == val_month].copy()
            return train_df, val_df

    train_df, val_df = train_test_split(
        features_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=features_df["is_churn"],
    )
    return train_df.copy(), val_df.copy()


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scale_pos_weight: float,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    p99_secs: float = 0.0,
    seed: int = 42,
) -> TrainArtifact:
    """
    Train a LightGBM binary classifier with AUC-PR as the eval metric.

    Does NOT write files or call MLflow — all I/O is the caller's responsibility.

    Parameters
    ----------
    train_df / val_df : DataFrames with FEATURE_COLS + 'is_churn' column.
    scale_pos_weight  : imbalance ratio (retained / churned).
    p99_secs          : winsorization threshold, stored in the returned artifact.
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["is_churn"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["is_churn"]

    params = _lgbm_params(scale_pos_weight, seed)

    ds_train = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CAT_COLS, free_raw_data=False
    )
    ds_val = lgb.Dataset(
        X_val, label=y_val, categorical_feature=CAT_COLS,
        reference=ds_train, free_raw_data=False,
    )

    booster = lgb.train(
        params,
        ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    metrics = evaluate_on_holdout(booster, X_val, y_val, top_k=100)

    return TrainArtifact(
        booster=booster,
        feature_cols=list(FEATURE_COLS),
        cat_cols=list(CAT_COLS),
        p99_secs=p99_secs,
        scale_pos_weight=scale_pos_weight,
        n_train=len(train_df),
        n_val=len(val_df),
        churn_rate_train=float(y_train.mean()),
        val_auc_pr=metrics["val_auc_pr"],
        val_auc_roc=metrics["val_auc_roc"],
        precision_at_k=metrics["precision_at_k"],
        recall_at_k=metrics["recall_at_k"],
    )


def evaluate_on_holdout(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 100,
) -> dict:
    """
    Score model on X, return AUC-PR, AUC-ROC, Precision@K, Recall@K.

    Handles both lgb.Booster (.predict) and sklearn wrapper (.predict_proba)
    — same dispatch logic as the serving pipeline.

    Parameters
    ----------
    model  : lgb.Booster or sklearn-compatible model
    X      : feature DataFrame (must contain FEATURE_COLS)
    y      : true binary labels
    top_k  : number of highest-scored users to evaluate for Precision/Recall@K
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    auc_pr = float(average_precision_score(y, scores))
    auc_roc = float(roc_auc_score(y, scores))

    scores_s = pd.Series(scores, index=y.index)
    top_k_idx = scores_s.nlargest(top_k).index
    precision_at_k = float(y.loc[top_k_idx].mean())
    recall_at_k = float(y.loc[top_k_idx].sum() / y.sum()) if y.sum() > 0 else 0.0

    return {
        "val_auc_pr": auc_pr,
        "val_auc_roc": auc_roc,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _lgbm_params(scale_pos_weight: float, seed: int) -> dict:
    """Fixed LightGBM training params — mirrors the baseline in experiments/train_baseline.py."""
    return {
        "objective":         "binary",
        "metric":            "average_precision",
        "scale_pos_weight":  scale_pos_weight,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbosity":         -1,
        "random_state":      seed,
    }


def _auto_scale_pos_weight(y: pd.Series) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def _register_user_logs_source(
    con: duckdb.DuckDBPyConnection,
    data_source: str,
    data_dir: Path,
    pg_conn_str: str | None,
) -> None:
    """Create a v_user_logs view pointing at either CSV or PostgreSQL."""
    if data_source == "csv":
        d = data_dir.as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW v_user_logs AS SELECT * FROM read_csv_auto('{d}/user_logs.csv')"
        )
    else:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{pg_conn_str}' AS pg (TYPE POSTGRES, READ_ONLY)")
        con.execute("CREATE OR REPLACE VIEW v_user_logs AS SELECT * FROM pg.user_logs")


def _as_date(d: date | str | object) -> date:
    if isinstance(d, date):
        return d if type(d) is date else d.date() if hasattr(d, "date") else d
    return date.fromisoformat(str(d)[:10])
