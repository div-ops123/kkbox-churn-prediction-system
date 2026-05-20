"""
Build train_features.parquet from train.csv + transactions.csv + user_logs.csv + members.csv.

Pipeline:
  Stage 1 — base:     anchor expiry + per-user expire_dt (training-specific anchor logic)
  Stage 2 — p99_secs: compute winsorization threshold, freeze to models/feature_config.json
  Stage 3 — features: call build_features() from feature_module (shared with serving)
  Stage 4 — export:   join with labels/metadata, write data/train_features.parquet

Notes:
  - feature_cutoff_dt = expire_dt - 14 days, enforced inside build_features()
  - Fallback expire_dt = 2017-03-01 (cutoff = 2017-02-15) for ~930K users whose
    March expiry is absent from transactions.csv
  - LightGBM handles NaN natively; do NOT fill NaN with 0
  - has_anchor is metadata only — excluded from model features (always 1 at serving time)
"""

import json
import sys

import duckdb
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.feature_module import build_features

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA     = BASE_DIR / "data"
OUT      = DATA / "train_features.parquet"

# ── Stage 1: Base table ───────────────────────────────────────────────────────
# For each train.csv user, find their latest March 2017 expiry in transactions.csv.
# Exclude bad-expiry rows (membership_expire_date < transaction_date).
# ~40K users get a precise anchor; ~930K get the fallback expire_dt of 2017-03-01.
print("Stage 1: building base table...")
con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE TABLE base AS
WITH anchor AS (
    SELECT
        tr.msno,
        tr.is_churn,
        MAX(t.membership_expire_date) AS anchor_int
    FROM read_csv_auto('{DATA.as_posix()}/train.csv') tr
    LEFT JOIN read_csv_auto('{DATA.as_posix()}/transactions.csv') t
        ON  tr.msno = t.msno
        AND t.membership_expire_date BETWEEN 20170301 AND 20170331
        AND t.membership_expire_date >= t.transaction_date
    GROUP BY tr.msno, tr.is_churn
)
SELECT
    msno,
    is_churn,
    CASE WHEN anchor_int IS NOT NULL THEN 1 ELSE 0 END AS has_anchor,
    -- expire_dt is passed to build_features(); it computes cutoff = expire_dt - 14 days
    CASE
        WHEN anchor_int IS NOT NULL
        THEN STRPTIME(CAST(anchor_int AS VARCHAR), '%Y%m%d')
        ELSE DATE '2017-03-01'    -- cutoff = 2017-02-15 (= 2017-03-01 - 14 days)
    END AS expire_dt
FROM anchor
""")
n_base     = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
n_anchored = con.execute("SELECT SUM(has_anchor) FROM base").fetchone()[0]
print(f"  {n_base:,} users | {n_anchored:,} anchored ({n_anchored/n_base:.1%}) | "
      f"{n_base - n_anchored:,} fallback ({(n_base - n_anchored)/n_base:.1%})")

# ── Stage 2: Compute p99_secs once and freeze for serving ────────────────────
print("Stage 2: computing log p99_secs...")
p99_secs = con.execute(f"""
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LEAST(total_secs, 86400))
    FROM read_csv_auto('{DATA.as_posix()}/user_logs.csv')
""").fetchone()[0]
print(f"  total_secs p99 (post-86400 cap): {p99_secs:,.0f} s ({p99_secs/3600:.1f} hrs)")

_config_path = BASE_DIR / "models" / "feature_config.json"
_config_path.parent.mkdir(exist_ok=True)
with open(_config_path, "w") as _f:
    json.dump({"p99_secs": float(p99_secs)}, _f, indent=2)
print(f"  Saved feature_config.json -> {_config_path}")

# Extract base table before closing connection
base_df = con.execute("SELECT msno, is_churn, has_anchor, expire_dt FROM base").df()
con.close()

# ── Stage 3: Build features via shared module ─────────────────────────────────
print(f"Stage 3: building features for {len(base_df):,} users...")
features_df = build_features(
    msno_list    = base_df["msno"].tolist(),
    expire_dates = base_df["expire_dt"].tolist(),
    p99_secs     = p99_secs,
    data_source  = "csv",
    data_dir     = DATA,
)
print(f"  feature matrix: {features_df.shape[0]:,} rows x {features_df.shape[1]} cols")

# ── Stage 4: Join labels/metadata and export ─────────────────────────────────
print("Stage 4: joining labels and exporting...")
labels = base_df.set_index("msno")[["is_churn", "has_anchor"]]
final  = labels.join(features_df, how="left")
final["has_log_data"] = final["log_days"].notna().astype("int8")
final = final.reset_index()

n_final    = len(final)
churn_pct  = final["is_churn"].mean() * 100
n_null_txn = final["n_txn"].isna().sum()
n_log_data = final["has_log_data"].sum()
print(f"  {n_final:,} rows | churn rate = {churn_pct:.4f}% | "
      f"{n_null_txn:,} users with no transactions before cutoff | "
      f"{n_log_data:,} users with log data")

final.to_parquet(OUT, index=False)
print(f"\nSaved to {OUT}")

print("\nColumn null rates:")
null_rates = (
    final[features_df.columns.tolist()]
    .isna()
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"index": "column_name", 0: "null_pct"})
)
print(null_rates.to_string(index=False))
