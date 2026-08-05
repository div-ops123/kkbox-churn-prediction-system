"""
Shared feature computation module.

Called by both experiments/build_training_set.py and serving (serve.py).

Accepts a list of users and their subscription expiry dates; returns a feature
DataFrame. The 14-day pre-expiry cutoff is enforced INTERNALLY — callers must
pass the subscription expiry date, never the cutoff date itself.

Public API
----------
    build_features(msno_list, expire_dates, p99_secs, data_source, ...)
    FEATURE_COLS   — ordered list of feature column names
    CAT_COLS       — categorical subset of FEATURE_COLS
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

FEATURE_COLS: list[str] = [
    # transaction features
    "n_txn", "n_cancel", "ever_cancelled", "ever_promo",
    "last_is_auto_renew", "last_is_cancel", "last_actual_amount_paid",
    "last_payment_plan_days", "last_plan_list_price", "last_discount",
    "cancel_rate", "days_since_last_txn",
    # engagement-log features
    "log_days", "avg_daily_secs", "total_secs_sum",
    "avg_completion_ratio", "avg_daily_unq_songs", "days_since_last_log",
    # member features
    "registered_via", "tenure_days",
]

CAT_COLS: list[str] = [
    "last_is_auto_renew", "last_is_cancel", "ever_cancelled",
    "ever_promo", "registered_via",
]


def build_features(
    msno_list: Sequence[str],
    expire_dates: Sequence[date | str],
    p99_secs: float,
    data_source: str = "csv",
    data_dir: Path | None = None,
    pg_conn_str: str | None = None,
) -> pd.DataFrame:
    """
    Build feature vectors for a cohort of users.

    Parameters
    ----------
    msno_list    : user IDs
    expire_dates : subscription expiry date per user (same order as msno_list);
                   accepts datetime.date, pd.Timestamp, or ISO string
    p99_secs     : total_secs winsorization threshold — load from
                   models/feature_config.json, never recompute (training-time
                   value must match at serving time)
    data_source  : "csv"      — read from CSV files via DuckDB (training / EDA)
                   "postgres" — read from PostgreSQL via DuckDB postgres extension
    data_dir     : CSV directory; defaults to <project_root>/data
    pg_conn_str  : PostgreSQL URI, required when data_source="postgres"
                   e.g. "postgresql://kkbox:kkbox@localhost:5433/kkbox"

    Returns
    -------
    pd.DataFrame indexed by msno, columns = FEATURE_COLS, one row per input user.
    Users with no data in a source table get NaN for those columns — do NOT impute
    (LightGBM handles NaN natively; absence != zero for this dataset).
    """
    if len(msno_list) != len(expire_dates):
        raise ValueError("msno_list and expire_dates must have the same length")
    if not msno_list:
        return pd.DataFrame(columns=FEATURE_COLS)
    if data_source == "postgres" and not pg_conn_str:
        raise ValueError("pg_conn_str required when data_source='postgres'")

    _dir = Path(data_dir) if data_dir else DATA_DIR
    _expire = [_as_date(d) for d in expire_dates]

    cohort_df = pd.DataFrame({
        "msno":             list(msno_list),
        "feature_cutoff_dt": [(d - timedelta(days=14)).isoformat() for d in _expire],
    })

    con = duckdb.connect()
    try:
        _register_sources(con, data_source, _dir, pg_conn_str)

        con.register("_cohort_input", cohort_df)
        con.execute("""
            CREATE TEMP TABLE cohort AS
            SELECT msno, CAST(feature_cutoff_dt AS DATE) AS feature_cutoff_dt
            FROM _cohort_input
        """)

        _build_txn(con)
        _build_log(con, p99_secs)
        _build_member(con)

        result = con.execute("""
            SELECT
                c.msno,
                t.n_txn, t.n_cancel, t.ever_cancelled, t.ever_promo,
                t.last_is_auto_renew, t.last_is_cancel,
                t.last_actual_amount_paid, t.last_payment_plan_days,
                t.last_plan_list_price, t.last_discount,
                t.cancel_rate, t.days_since_last_txn,
                l.log_days, l.avg_daily_secs, l.total_secs_sum,
                l.avg_completion_ratio, l.avg_daily_unq_songs, l.days_since_last_log,
                m.registered_via, m.tenure_days
            FROM cohort c
            LEFT JOIN _txn_features    t ON c.msno = t.msno
            LEFT JOIN _log_features    l ON c.msno = l.msno
            LEFT JOIN _member_features m ON c.msno = m.msno
        """).df()
    finally:
        con.close()

    return result.set_index("msno")[FEATURE_COLS]


# ── private helpers ─────────────────────────────────────────────────────────

def _as_date(d: date | str | object) -> date:
    if isinstance(d, date):
        return d if type(d) is date else d.date() if hasattr(d, "date") else d
    return date.fromisoformat(str(d)[:10])


def _register_sources(
    con: duckdb.DuckDBPyConnection,
    data_source: str,
    data_dir: Path,
    pg_conn_str: str | None,
) -> None:
    """Create v_transactions, v_user_logs, v_members views pointing at the source."""
    if data_source == "csv":
        # Use posix paths — DuckDB handles them on all platforms.
        d = data_dir.as_posix()
        con.execute(f"CREATE OR REPLACE VIEW v_transactions AS SELECT * FROM read_csv_auto('{d}/transactions.csv')")
        con.execute(f"CREATE OR REPLACE VIEW v_user_logs    AS SELECT * FROM read_csv_auto('{d}/user_logs.csv')")
        con.execute(f"CREATE OR REPLACE VIEW v_members      AS SELECT * FROM read_csv_auto('{d}/members.csv')")
    else:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{pg_conn_str}' AS pg (TYPE POSTGRES, READ_ONLY)")
        con.execute("CREATE OR REPLACE VIEW v_transactions AS SELECT * FROM pg.transactions")
        con.execute("CREATE OR REPLACE VIEW v_user_logs    AS SELECT * FROM pg.user_logs")
        con.execute("CREATE OR REPLACE VIEW v_members      AS SELECT * FROM pg.members")


def _build_txn(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TEMP TABLE _txn_features AS
        WITH txns AS (
            SELECT
                t.msno,
                STRPTIME(CAST(t.transaction_date AS VARCHAR), '%Y%m%d')     AS txn_dt,
                t.is_auto_renew,
                t.is_cancel,
                t.actual_amount_paid,
                t.payment_plan_days,
                t.plan_list_price,
                t.plan_list_price - t.actual_amount_paid                     AS discount,
                CASE WHEN t.actual_amount_paid = 0 THEN 1 ELSE 0 END         AS is_promo,
                CASE
                    WHEN t.payment_plan_days  = 0
                     AND t.plan_list_price    = 0
                     AND t.actual_amount_paid > 0 THEN 1 ELSE 0
                END                                                           AS is_legacy
            FROM v_transactions t
            WHERE t.membership_expire_date >= t.transaction_date
        ),
        filtered AS (
            SELECT tx.*, c.feature_cutoff_dt
            FROM txns tx
            INNER JOIN cohort c ON tx.msno = c.msno
            WHERE tx.txn_dt <= c.feature_cutoff_dt
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY msno ORDER BY txn_dt DESC) AS rn
            FROM filtered
        )
        SELECT
            msno,
            COUNT(*)                                                                      AS n_txn,
            SUM(is_cancel)                                                                AS n_cancel,
            MAX(is_cancel)                                                                AS ever_cancelled,
            MAX(is_promo)                                                                 AS ever_promo,
            MAX(CASE WHEN rn = 1 THEN is_auto_renew              END)                     AS last_is_auto_renew,
            MAX(CASE WHEN rn = 1 THEN is_cancel                  END)                     AS last_is_cancel,
            MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN actual_amount_paid   END)         AS last_actual_amount_paid,
            MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN payment_plan_days    END)         AS last_payment_plan_days,
            MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN plan_list_price      END)         AS last_plan_list_price,
            MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN discount             END)         AS last_discount,
            ROUND(SUM(is_cancel)::DOUBLE / NULLIF(COUNT(*), 0), 4)                        AS cancel_rate,
            DATE_DIFF('day', MAX(txn_dt), MAX(feature_cutoff_dt))                         AS days_since_last_txn
        FROM ranked
        GROUP BY msno
    """)


def _build_log(con: duckdb.DuckDBPyConnection, p99_secs: float) -> None:
    con.execute(f"""
        CREATE TEMP TABLE _log_features AS
        WITH logs AS (
            SELECT
                l.msno,
                STRPTIME(CAST(l.date AS VARCHAR), '%Y%m%d')                    AS log_dt,
                LEAST(LEAST(l.total_secs, 86400), {p99_secs})                  AS total_secs_w,
                l.num_100,
                l.num_unq,
                l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100        AS total_songs
            FROM v_user_logs l
        ),
        filtered AS (
            SELECT lg.*, c.feature_cutoff_dt
            FROM logs lg
            INNER JOIN cohort c ON lg.msno = c.msno
            WHERE lg.log_dt <= c.feature_cutoff_dt
        )
        SELECT
            msno,
            COUNT(*)                                                             AS log_days,
            ROUND(AVG(total_secs_w), 1)                                         AS avg_daily_secs,
            ROUND(SUM(total_secs_w), 0)                                         AS total_secs_sum,
            ROUND(AVG(num_100::DOUBLE / NULLIF(total_songs, 0)), 4)             AS avg_completion_ratio,
            ROUND(AVG(num_unq::DOUBLE), 2)                                      AS avg_daily_unq_songs,
            DATE_DIFF('day', MAX(log_dt), MAX(feature_cutoff_dt))               AS days_since_last_log
        FROM filtered
        GROUP BY msno
    """)


def _build_member(con: duckdb.DuckDBPyConnection) -> None:
    # Exclude registration dates in the future relative to each user's feature cutoff,
    # and dates before 2000-01-01 (data quality artifact from original members.csv).
    con.execute("""
        CREATE TEMP TABLE _member_features AS
        SELECT
            m.msno,
            m.registered_via,
            DATE_DIFF(
                'day',
                STRPTIME(CAST(m.registration_init_time AS VARCHAR), '%Y%m%d'),
                c.feature_cutoff_dt
            ) AS tenure_days
        FROM v_members m
        INNER JOIN cohort c ON m.msno = c.msno
        WHERE STRPTIME(CAST(m.registration_init_time AS VARCHAR), '%Y%m%d')
              BETWEEN DATE '2000-01-01' AND c.feature_cutoff_dt
    """)
