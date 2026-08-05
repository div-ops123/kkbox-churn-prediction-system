"""
Build train_features.parquet from train.csv + transactions.csv + user_logs.csv + members.csv.

Pipeline:
  Stage 1 — base:     anchor expiry + per-user expire_dt (training-specific anchor logic)
  Stage 2 — p99_secs: compute winsorization threshold, freeze to models/feature_config.json
  Stage 3 — features: call build_features() from feature_module (shared with serving)
  Stage 4 — export:   join with labels/metadata, write data/train_features.parquet

Notes:
  - feature_cutoff_dt = expire_dt - 14 days, enforced inside build_features()
  - Training cohort is anchored-only: only the ~40K train.csv users with a verified
    March 2017 expiry in transactions.csv. The other ~930K are excluded rather than
    given a fallback cutoff — every user scored in production has a real, known
    expiry date (has_anchor is always 1 at serving time), so training exclusively
    on real anchors removes a train/serve skew the fallback used to introduce.
    This is a deliberate experiment, not an assumption that it outperforms the
    full-cohort+fallback approach — validate both on the same held-out set.
  - LightGBM handles NaN natively; do NOT fill NaN with 0
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
# JOIN (not LEFT JOIN) means users with no March-2017 expiry are dropped entirely —
# no fallback cutoff, no has_anchor flag. Every remaining user has a real,
# per-user anchor_expiry_dt, matching what production always has at serving time.
#
# The old join condition also required membership_expire_date >= transaction_date,
# to exclude rows that looked like backdated cancellations. That turned out to be
# a false alarm (explore/date_issues_check.py Cell 2): every one of those rows is
# a 1-3 day gap between expiry and a cancellation being processed shortly after —
# ordinary behavior, not corrupted data — so that exclusion has been dropped here too.
print("Stage 1: building base table (anchored users only)...")
con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT
    tr.msno,
    tr.is_churn,
    STRPTIME(CAST(MAX(t.membership_expire_date) AS VARCHAR), '%Y%m%d') AS expire_dt
FROM read_csv_auto('{DATA.as_posix()}/train.csv') tr
JOIN read_csv_auto('{DATA.as_posix()}/transactions.csv') t
    ON  tr.msno = t.msno
    AND t.membership_expire_date BETWEEN 20170301 AND 20170331
GROUP BY tr.msno, tr.is_churn
""")
n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
print(f"  {n_base:,} anchored users (train.csv users without a verified "
      f"March 2017 expiry are excluded, not fallback-filled)")

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
base_df = con.execute("SELECT msno, is_churn, expire_dt FROM base").df()
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
labels = base_df.set_index("msno")[["is_churn"]]
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
