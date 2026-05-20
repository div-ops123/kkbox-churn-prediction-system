"""
Core labeling logic for the KKBox churn labeling pipeline.

Called by label.py (Prefect orchestration). Pure — no Prefect, no psycopg2,
no side effects. Each public function opens and closes its own DuckDB connection.

Public API
----------
    build_cohort(config)               → pd.DataFrame [msno, anchor_expiry_date, feature_cutoff_date]
    compute_labels(cohort_df, config)  → pd.DataFrame [LABEL_COLS]
    LABEL_COLS                         — column order matching the labels table schema
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from config import LabelingConfig


BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_COLS: list[str] = [
    "msno",
    "anchor_expiry_date",
    "feature_cutoff_date",
    "is_churn",
    "labeled_date",
]


# ── Public functions ───────────────────────────────────────────────────────────

def build_cohort(config: LabelingConfig) -> pd.DataFrame:
    """
    Identify users whose latest valid subscription expiry falls in config.cohort_month.

    Excludes records where membership_expire_date < transaction_date (Error 2 from EDA).
    Computes feature_cutoff_date = anchor_expiry_date - config.feature_cutoff_lag_days.

    Returns
    -------
    pd.DataFrame with columns [msno, anchor_expiry_date, feature_cutoff_date].
    Returns an empty DataFrame (0 rows) when no users expire in the target month.
    """
    month_start_int, month_end_int = _month_bounds_int(config.cohort_month)

    con = duckdb.connect()
    try:
        _register_txn_source(con, config)
        cohort_df = _fetch_anchor_expiries(con, month_start_int, month_end_int)
    finally:
        con.close()

    if cohort_df.empty:
        return pd.DataFrame(columns=["msno", "anchor_expiry_date", "feature_cutoff_date"])

    cohort_df["feature_cutoff_date"] = cohort_df["anchor_expiry_date"].apply(
        lambda d: (d if isinstance(d, date) else d.date()) - timedelta(days=config.feature_cutoff_lag_days)
    )
    return cohort_df[["msno", "anchor_expiry_date", "feature_cutoff_date"]]


def compute_labels(cohort_df: pd.DataFrame, config: LabelingConfig) -> pd.DataFrame:
    """
    Assign is_churn to each user in the cohort.

    label_source="train_csv"    — LEFT JOIN against the official train.csv / train_labels
                                   Postgres table. Users absent from the source get
                                   is_churn = NaN (dropped in write_labels_postgres).
    label_source="transactions" — Derive from renewal patterns: is_churn = 1 if no
                                   valid (is_cancel=0) transaction appears within
                                   config.renewal_window_days after anchor_expiry_date.

    Returns
    -------
    pd.DataFrame with columns LABEL_COLS.
    """
    if cohort_df.empty:
        return pd.DataFrame(columns=LABEL_COLS)

    con = duckdb.connect()
    try:
        _register_txn_source(con, config)

        con.register("_label_cohort_input", cohort_df)
        con.execute("""
            CREATE TEMP TABLE label_cohort AS
            SELECT
                msno,
                CAST(anchor_expiry_date  AS DATE) AS anchor_expiry_date,
                CAST(feature_cutoff_date AS DATE) AS feature_cutoff_date
            FROM _label_cohort_input
        """)

        if config.label_source == "train_csv":
            result_df = _label_from_train_csv(con, config)
        else:
            result_df = _label_from_transactions(con, config)
    finally:
        con.close()

    result_df["labeled_date"] = config.labeled_date
    return result_df[LABEL_COLS]


# ── Private helpers ────────────────────────────────────────────────────────────

def _month_bounds_int(cohort_month: str) -> tuple[int, int]:
    """Convert "YYYY-MM" to (YYYYMM01, YYYYMMdd) as YYYYMMDD integers."""
    year = int(cohort_month[:4])
    month = int(cohort_month[5:7])
    last_day = calendar.monthrange(year, month)[1]
    start_int = year * 10_000 + month * 100 + 1
    end_int = year * 10_000 + month * 100 + last_day
    return start_int, end_int


def _register_txn_source(con: duckdb.DuckDBPyConnection, config: LabelingConfig) -> None:
    """
    Create the v_transactions view pointing at either CSV files or PostgreSQL.
    When data_source="postgres", also attaches the DB as 'pg' so that
    pg.train_labels is accessible in subsequent queries.
    """
    if config.data_source == "csv":
        d = config.data_dir.as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW v_transactions AS SELECT * FROM read_csv_auto('{d}/transactions.csv')"
        )
    else:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{config.postgres.conn_str}' AS pg (TYPE POSTGRES, READ_ONLY)")
        con.execute("CREATE OR REPLACE VIEW v_transactions AS SELECT * FROM pg.transactions")


def _fetch_anchor_expiries(
    con: duckdb.DuckDBPyConnection,
    month_start_int: int,
    month_end_int: int,
) -> pd.DataFrame:
    """
    Query v_transactions for users whose latest valid expiry falls in the target month.

    Excludes records where membership_expire_date < transaction_date (EDA Error 2 —
    backdated cancellation records that would corrupt the anchor date).

    Returns [msno, anchor_expiry_date] as a DataFrame.
    """
    return con.execute(
        """
        SELECT
            msno,
            MAX(
                STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')
            )::DATE AS anchor_expiry_date
        FROM v_transactions
        WHERE membership_expire_date BETWEEN ? AND ?
          AND membership_expire_date >= transaction_date
        GROUP BY msno
        """,
        [month_start_int, month_end_int],
    ).df()


def _label_from_train_csv(
    con: duckdb.DuckDBPyConnection,
    config: LabelingConfig,
) -> pd.DataFrame:
    """
    Assign is_churn by joining the cohort against the official train labels.

    When data_source="postgres": reads from pg.train_labels (already attached).
    When data_source="csv":      reads from the train.csv file on disk.

    Users in the cohort but absent from the label source get is_churn = NULL (NaN).
    """
    if config.data_source == "postgres":
        label_source_sql = "SELECT msno, is_churn FROM pg.train_labels"
    else:
        train_path = config.train_csv_path.as_posix()
        label_source_sql = f"SELECT msno, is_churn FROM read_csv_auto('{train_path}')"

    return con.execute(
        f"""
        SELECT
            c.msno,
            c.anchor_expiry_date,
            c.feature_cutoff_date,
            t.is_churn
        FROM label_cohort c
        LEFT JOIN ({label_source_sql}) t ON c.msno = t.msno
        """
    ).df()


def _label_from_transactions(
    con: duckdb.DuckDBPyConnection,
    config: LabelingConfig,
) -> pd.DataFrame:
    """
    Derive is_churn from transaction renewal patterns.

    A user is labeled is_churn = 0 if at least one valid (is_cancel=0) transaction
    exists within renewal_window_days after their anchor_expiry_date.
    All others are labeled is_churn = 1.

    The renewal window integer is injected via f-string — safe because it is
    validated as a positive int in load_labeling_config().
    """
    window = config.renewal_window_days
    return con.execute(
        f"""
        WITH renewals AS (
            SELECT DISTINCT t.msno
            FROM v_transactions t
            INNER JOIN label_cohort c ON t.msno = c.msno
            WHERE STRPTIME(CAST(t.transaction_date AS VARCHAR), '%Y%m%d') > c.anchor_expiry_date
              AND STRPTIME(CAST(t.transaction_date AS VARCHAR), '%Y%m%d')
                  <= c.anchor_expiry_date + INTERVAL '{window} days'
              AND t.is_cancel = 0
              AND t.membership_expire_date >= t.transaction_date
        )
        SELECT
            c.msno,
            c.anchor_expiry_date,
            c.feature_cutoff_date,
            CASE WHEN r.msno IS NOT NULL THEN 0 ELSE 1 END AS is_churn
        FROM label_cohort c
        LEFT JOIN renewals r ON c.msno = r.msno
        """
    ).df()
