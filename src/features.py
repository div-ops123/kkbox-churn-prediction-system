"""
Build train_features.parquet from train.csv + transactions.csv + user_logs.csv + members.csv.

Pipeline:
  Stage 1 — base:           anchor expiry + per-user feature_cutoff_dt
  Stage 2 — txn_features:   transaction aggregates, filtered by cutoff per user
  Stage 3 — log_features:   user-log aggregates, filtered by cutoff per user
  Stage 4 — member_features: tenure_days + registered_via
  Stage 5 — final join:     LEFT JOIN all stages onto base
  Stage 6 — export:         Parquet to data/train_features.parquet

Notes:
  - feature_cutoff_dt = anchor_expiry - 14 days (fallback: 2017-02-15 for ~930K users
    whose March expiry is absent from transactions.csv)
  - LightGBM handles NaN natively; do NOT fill NaN with 0 (absence != zero activity)
  - Log features are NULL for ~98% of users (user_logs only covers March 1-31;
    most feature cutoffs fall before March 1)
  - has_anchor is metadata only — do not pass to LightGBM (always 1 at serving time)
"""

import json
import duckdb
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
OUT  = DATA / "train_features.parquet"

con = duckdb.connect()

# ── Stage 1: Base table ────────────────────────────────────────────────────────
# For each train.csv user, find their latest March 2017 expiry in transactions.csv.
# Exclude bad-expiry rows (membership_expire_date < transaction_date).
# ~40K users get a precise anchor; ~930K get the Feb 15 fallback.
print("Stage 1: building base table...")
con.execute(f"""
CREATE OR REPLACE TABLE base AS
WITH anchor AS (
    SELECT
        tr.msno,
        tr.is_churn,
        MAX(t.membership_expire_date) AS anchor_int   -- YYYYMMDD integer, or NULL
    FROM read_csv_auto('{DATA}/train.csv') tr
    LEFT JOIN read_csv_auto('{DATA}/transactions.csv') t
        ON  tr.msno = t.msno
        AND t.membership_expire_date BETWEEN 20170301 AND 20170331
        AND t.membership_expire_date >= t.transaction_date      -- drop bad-expiry rows
    GROUP BY tr.msno, tr.is_churn
)
SELECT
    msno,
    is_churn,
    CASE WHEN anchor_int IS NOT NULL THEN 1 ELSE 0 END AS has_anchor,
    CASE
        WHEN anchor_int IS NOT NULL
        THEN STRPTIME(CAST(anchor_int AS VARCHAR), '%Y%m%d') - INTERVAL '14 days'
        ELSE DATE '2017-02-15'
    END AS feature_cutoff_dt
FROM anchor
""")
n_base    = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
n_anchored = con.execute("SELECT SUM(has_anchor) FROM base").fetchone()[0]
print(f"  {n_base:,} users total | {n_anchored:,} anchored ({n_anchored/n_base:.1%}) | "
      f"{n_base-n_anchored:,} fallback ({(n_base-n_anchored)/n_base:.1%})")

# ── Stage 2: Transaction features ─────────────────────────────────────────────
# Filter each user's transactions to those on or before their feature_cutoff_dt.
# Exclude bad-expiry and legacy records from plan/price features (keep in counts).
# ROW_NUMBER gives us the latest transaction for "last_*" features.
print("Stage 2: building txn_features...")
con.execute(f"""
CREATE OR REPLACE TABLE txn_features AS
WITH txns AS (
    SELECT
        t.msno,
        STRPTIME(CAST(t.transaction_date AS VARCHAR), '%Y%m%d') AS txn_dt,
        t.is_auto_renew,
        t.is_cancel,
        t.actual_amount_paid,
        t.payment_plan_days,
        t.plan_list_price,
        t.plan_list_price - t.actual_amount_paid                   AS discount,
        CASE WHEN t.actual_amount_paid = 0 THEN 1 ELSE 0 END       AS is_promo,
        CASE
            WHEN t.payment_plan_days = 0
             AND t.plan_list_price   = 0
             AND t.actual_amount_paid > 0 THEN 1 ELSE 0
        END                                                         AS is_legacy
    FROM read_csv_auto('{DATA}/transactions.csv') t
    WHERE t.membership_expire_date >= t.transaction_date            -- drop bad-expiry rows
),
filtered AS (
    SELECT tx.*, b.feature_cutoff_dt
    FROM txns tx
    INNER JOIN base b ON tx.msno = b.msno
    WHERE tx.txn_dt <= b.feature_cutoff_dt
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY msno ORDER BY txn_dt DESC) AS rn
    FROM filtered
)
SELECT
    msno,
    COUNT(*)                                                                    AS n_txn,
    SUM(is_cancel)                                                              AS n_cancel,
    MAX(is_cancel)                                                              AS ever_cancelled,
    MAX(is_promo)                                                               AS ever_promo,
    -- latest-transaction features
    MAX(CASE WHEN rn = 1 THEN is_auto_renew            END)                     AS last_is_auto_renew,
    MAX(CASE WHEN rn = 1 THEN is_cancel                END)                     AS last_is_cancel,
    MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN actual_amount_paid  END)        AS last_actual_amount_paid,
    MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN payment_plan_days   END)        AS last_payment_plan_days,
    MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN plan_list_price     END)        AS last_plan_list_price,
    MAX(CASE WHEN rn = 1 AND is_legacy = 0 THEN discount            END)        AS last_discount,
    ROUND(SUM(is_cancel)::DOUBLE / NULLIF(COUNT(*), 0), 4)                      AS cancel_rate,
    DATE_DIFF('day', MAX(txn_dt), MAX(feature_cutoff_dt))                       AS days_since_last_txn
FROM ranked
GROUP BY msno
""")
print(f"  {con.execute('SELECT COUNT(*) FROM txn_features').fetchone()[0]:,} users with 1+ transaction before cutoff")

# ── Stage 3: User log features ─────────────────────────────────────────────────
# user_logs only covers March 1-31, 2017. Feature cutoffs mostly fall before March 1,
# so log features will be NULL for ~98% of users — this is expected, not an error.
# Winsorize: hard cap total_secs at 86,400 then at p99 of the training distribution.
print("Stage 3: computing log p99, then building log_features...")
p99_secs = con.execute(f"""
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LEAST(total_secs, 86400))
    FROM read_csv_auto('{DATA}/user_logs.csv')
""").fetchone()[0]
print(f"  total_secs p99 (post-86400 cap): {p99_secs:,.0f} s ({p99_secs/3600:.1f} hrs)")

# Persist threshold so serving pipeline uses the same value computed here,
# not a recomputed value from a potentially drifted distribution.
_config_path = BASE_DIR / "models" / "feature_config.json"
_config_path.parent.mkdir(exist_ok=True)
with open(_config_path, "w") as _f:
    json.dump({"p99_secs": float(p99_secs)}, _f, indent=2)
print(f"  Saved feature_config.json → {_config_path}")

con.execute(f"""
CREATE OR REPLACE TABLE log_features AS
WITH logs AS (
    SELECT
        l.msno,
        STRPTIME(CAST(l.date AS VARCHAR), '%Y%m%d')                     AS log_dt,
        LEAST(LEAST(l.total_secs, 86400), {p99_secs})                   AS total_secs_w,
        l.num_100,
        l.num_unq,
        l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100         AS total_songs
    FROM read_csv_auto('{DATA}/user_logs.csv') l
),
filtered AS (
    SELECT lg.*, b.feature_cutoff_dt
    FROM logs lg
    INNER JOIN base b ON lg.msno = b.msno
    WHERE lg.log_dt <= b.feature_cutoff_dt
)
SELECT
    msno,
    COUNT(*)                                                              AS log_days,
    ROUND(AVG(total_secs_w), 1)                                          AS avg_daily_secs,
    ROUND(SUM(total_secs_w), 0)                                          AS total_secs_sum,
    ROUND(AVG(num_100::DOUBLE / NULLIF(total_songs, 0)), 4)              AS avg_completion_ratio,
    ROUND(AVG(num_unq::DOUBLE), 2)                                       AS avg_daily_unq_songs,
    DATE_DIFF('day', MAX(log_dt), MAX(feature_cutoff_dt))                AS days_since_last_log
FROM filtered
GROUP BY msno
""")
n_log = con.execute("SELECT COUNT(*) FROM log_features").fetchone()[0]
print(f"  {n_log:,} users with 1+ log day before cutoff ({n_log/n_base:.1%} of train set)")

# ── Stage 4: Member features ───────────────────────────────────────────────────
# tenure_days + registered_via only (gender/bd dropped per EDA decisions).
# Exclude future reg dates (> 20170317) and suspiciously old dates (< 20000101).
print("Stage 4: building member_features...")
con.execute(f"""
CREATE OR REPLACE TABLE member_features AS
SELECT
    m.msno,
    m.registered_via,
    DATE_DIFF(
        'day',
        STRPTIME(CAST(m.registration_init_time AS VARCHAR), '%Y%m%d'),
        b.feature_cutoff_dt
    ) AS tenure_days
FROM read_csv_auto('{DATA}/members.csv') m
INNER JOIN base b ON m.msno = b.msno
WHERE m.registration_init_time BETWEEN 20000101 AND 20170317
""")
print(f"  {con.execute('SELECT COUNT(*) FROM member_features').fetchone()[0]:,} users with valid registration date")

# ── Stage 5: Final join ────────────────────────────────────────────────────────
# LEFT JOIN so users with no matching rows in any stage keep NaN (not dropped).
# has_anchor is kept as a column but excluded from model features — it's always 1
# at serving time (you always know a user's expiry date in production).
print("Stage 5: joining all stages...")
con.execute("""
CREATE OR REPLACE TABLE train_features AS
SELECT
    b.msno,
    b.is_churn,
    b.has_anchor,           -- metadata only, do NOT pass to LightGBM
    -- transaction features
    t.n_txn,
    t.n_cancel,
    t.ever_cancelled,
    t.ever_promo,
    t.last_is_auto_renew,
    t.last_is_cancel,
    t.last_actual_amount_paid,
    t.last_payment_plan_days,
    t.last_plan_list_price,
    t.last_discount,
    t.cancel_rate,
    t.days_since_last_txn,
    -- log features (NULL for ~98% of users — LightGBM handles NaN natively)
    l.log_days,
    l.avg_daily_secs,
    l.total_secs_sum,
    l.avg_completion_ratio,
    l.avg_daily_unq_songs,
    l.days_since_last_log,
    CASE WHEN l.msno IS NOT NULL THEN 1 ELSE 0 END AS has_log_data,
    -- member features
    m.registered_via,
    m.tenure_days
FROM base b
LEFT JOIN txn_features    t ON b.msno = t.msno
LEFT JOIN log_features    l ON b.msno = l.msno
LEFT JOIN member_features m ON b.msno = m.msno
""")

n_final = con.execute("SELECT COUNT(*) FROM train_features").fetchone()[0]
churn_pct = con.execute("SELECT ROUND(100.0*SUM(is_churn)/COUNT(*),4) FROM train_features").fetchone()[0]
n_null_txn = con.execute("SELECT COUNT(*) FROM train_features WHERE n_txn IS NULL").fetchone()[0]
n_log_data = con.execute("SELECT SUM(has_log_data) FROM train_features").fetchone()[0]
print(f"  {n_final:,} rows | churn rate = {churn_pct}% | "
      f"{n_null_txn:,} users with no transactions before cutoff | "
      f"{n_log_data:,} users with log data")

# ── Stage 6: Export ────────────────────────────────────────────────────────────
con.execute(f"COPY train_features TO '{OUT}' (FORMAT PARQUET)")
print(f"\nSaved to {OUT}")

# quick feature summary
print("\nColumn null rates:")
print(con.execute("""
SELECT col AS column_name, null_pct FROM (
    SELECT 'n_txn'                AS col, ROUND(100.0*COUNT(*) FILTER (WHERE n_txn                IS NULL)/COUNT(*),1) AS null_pct FROM train_features UNION ALL
    SELECT 'last_is_auto_renew',          ROUND(100.0*COUNT(*) FILTER (WHERE last_is_auto_renew   IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'last_plan_list_price',        ROUND(100.0*COUNT(*) FILTER (WHERE last_plan_list_price IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'last_actual_amount_paid',     ROUND(100.0*COUNT(*) FILTER (WHERE last_actual_amount_paid IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'log_days',                    ROUND(100.0*COUNT(*) FILTER (WHERE log_days              IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'avg_daily_secs',              ROUND(100.0*COUNT(*) FILTER (WHERE avg_daily_secs        IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'tenure_days',                 ROUND(100.0*COUNT(*) FILTER (WHERE tenure_days           IS NULL)/COUNT(*),1) FROM train_features UNION ALL
    SELECT 'registered_via',              ROUND(100.0*COUNT(*) FILTER (WHERE registered_via        IS NULL)/COUNT(*),1) FROM train_features
) ORDER BY null_pct DESC
""").df().to_string(index=False))
