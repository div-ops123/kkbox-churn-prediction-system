"""
LightGBM churn prediction — baseline model.

Input:  data/train_features.parquet   (produced by src/features.py)
Output: models/lgbm_baseline.pkl
        mlflow.db                      (MLflow experiment log, project root)

Run (from project root):
    python src/train.py
"""

# %%
import json
import os
import pickle
import sys
import mlflow
import mlflow.lightgbm
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_module import FEATURE_COLS, CAT_COLS

# %%
BASE_DIR      = Path(__file__).resolve().parent.parent
PARQUET       = BASE_DIR / "data" / "train_features.parquet"
MODELS        = BASE_DIR / "models"
FEATURE_CFG   = MODELS / "feature_config.json"   # written by features.py
TOP_K         = 100   # marketing budget constraint: contacts per campaign

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/mlflow.db")
)
mlflow.set_experiment("kkbox-churn-baseline")

# %%
# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
print(f"Loaded: {len(df):,} rows x {df.shape[1]} cols")
print(f"Churn rate: {df['is_churn'].mean():.4%}  "
      f"({df['is_churn'].sum():,} churned / {(df['is_churn']==0).sum():,} retained)")

with open(FEATURE_CFG) as _f:
    feature_config = json.load(_f)
print(f"Feature config loaded: p99_secs={feature_config['p99_secs']:,.0f}")
# %%

# ── 2. Null rate audit ────────────────────────────────────────────────────────
#
# WHY WE DO NOT IMPUTE NULLS
# --------------------------
# In a production setting, KKBox would supply full transaction history and
# rolling user logs, so the ~97-98% null rates seen here would not exist.
# These nulls are artifacts of the Kaggle dataset extract, not genuine
# missing values at inference time:
#
#   CAUSE A  extract gap — transaction features (~96.7% null)
#     The transactions.csv extract covers only ~40K of the 970K train users.
#     In production, every user has a transaction history.
#     => Imputing 0 would wrongly encode "zero activity" for users who actually
#        have full, rich histories. LightGBM's NaN routing learns that NULL here
#        signals "data not present in extract" — a dataset artifact, not a signal.
#
#   CAUSE B  log coverage gap — engagement features (~98.2% null)
#     user_logs.csv only covers March 1-31, 2017. Most feature cutoffs fall
#     before March 1, so no log data falls inside those windows.
#     In production, rolling 30/60/90-day logs would be available.
#     => Same reasoning: absence of data in this extract ≠ zero listening.
#
#   CAUSE C  legitimate data quality — member features (~11.4% null)
#     ~55K members.csv rows have registration_init_time > 20170317 (future
#     dates). These are excluded from tenure_days per EDA decisions.
#     => This IS a genuine data quality issue that could persist in production.
#        Still no imputation: LightGBM will learn the null pattern if it
#        correlates with churn; imputing the mean would destroy that signal.
#
# The goal of this project is to demonstrate model-building judgment and
# awareness of dataset limitations. The high null rates do not invalidate
# the model — they are fully expected given the extract, and LightGBM's
# native NaN handling is the correct approach for all three causes.

# %%
FEATURE_GROUPS = {
    "transaction  [CAUSE A — extract gap, ~96.7% null]": [
        "n_txn",
        "n_cancel",
        "ever_cancelled",
        "ever_promo",
        "last_is_auto_renew",
        "last_is_cancel",
        "last_actual_amount_paid",
        "last_payment_plan_days",
        "last_plan_list_price",
        "last_discount",
        "cancel_rate",
        "days_since_last_txn",
    ],
    "engagement-log  [CAUSE B — March-only coverage, ~98.2% null]": [
        "log_days",
        "avg_daily_secs",
        "total_secs_sum",
        "avg_completion_ratio",
        "avg_daily_unq_songs",
        "days_since_last_log",
    ],
    "member  [CAUSE C — invalid reg dates, ~11.4% null]": [
        "registered_via",
        "tenure_days",
    ],
}

SEP = "=" * 68
print(f"\n{SEP}")
print("NULL RATE AUDIT  (null pct | bar chart)")
print(SEP)
for group_label, cols in FEATURE_GROUPS.items():
    print(f"\n  [{group_label}]")
    for col in cols:
        pct = df[col].isna().mean() * 100
        bar = "#" * int(pct / 4)
        print(f"    {col:<30}  {pct:5.1f}%  {bar}")

print("\n  [metadata — excluded from model]")
n_fallback = (df["has_anchor"] == 0).sum()
n_no_log   = (df["has_log_data"] == 0).sum()
print(f"    has_anchor = 0 (fallback cutoff):    {n_fallback:,}  ({n_fallback/len(df):.1%})")
print(f"    has_log_data = 0 (no logs in window): {n_no_log:,}  ({n_no_log/len(df):.1%})")
print(SEP)
# %%

df.head()

# %%
# ── 3. Feature matrix ─────────────────────────────────────────────────────────
# has_anchor excluded: always 1 at serving time — spurious training artifact.
# has_log_data excluded: redundant with log feature nulls (LightGBM sees NaN).

X = df[FEATURE_COLS]
y = df["is_churn"]

# %%
# ── 4. Train / validation split ───────────────────────────────────────────────
# Random stratified split — all training rows are from the same cohort (March
# 2017), so there is no temporal ordering within the training set to preserve.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Val: {len(X_val):,}")
print(f"Train churn: {y_train.mean():.4%}  |  Val churn: {y_val.mean():.4%}")

# %%
# ── 5. LightGBM baseline ──────────────────────────────────────────────────────
scale_pos = (y == 0).sum() / (y == 1).sum()   # ≈ 10.1

params = {
    "objective":         "binary",
    "metric":            "average_precision",   # AUC-PR — primary metric
    "scale_pos_weight":  scale_pos,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "verbosity":         -1,
    "random_state":      42,
}

print(f"\nscale_pos_weight = {scale_pos:.2f}  (retained:churned ratio)")
print("Training LightGBM (early stopping on val AUC-PR)...\n")

ds_train = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_COLS, free_raw_data=False)
ds_val   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=CAT_COLS,
                        reference=ds_train,     free_raw_data=False)

with mlflow.start_run(run_name="lgbm_baseline"):

    mlflow.log_params({
        **{k: v for k, v in params.items() if k != "verbosity"},
        "features":     ",".join(FEATURE_COLS),
        "cat_features": ",".join(CAT_COLS),
        "n_features":   len(FEATURE_COLS),
        "train_rows":   len(X_train),
        "val_rows":     len(X_val),
        "top_k":        TOP_K,
        "p99_secs":     feature_config["p99_secs"],
    })
    mlflow.log_artifact(str(FEATURE_CFG), artifact_path="feature_config")

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    y_score = model.predict(X_val)

    auc_pr  = average_precision_score(y_val, y_score)
    auc_roc = roc_auc_score(y_val, y_score)

    # Precision@K and Recall@K — business-facing metrics.
    # Marketing can contact at most TOP_K users per campaign; we rank by
    # predicted churn probability and evaluate quality of that shortlist.
    y_score_s   = pd.Series(y_score, index=y_val.index)
    top_k_idx   = y_score_s.nlargest(TOP_K).index
    prec_at_k   = float(y_val.loc[top_k_idx].mean())
    recall_at_k = float(y_val.loc[top_k_idx].sum() / y_val.sum())

    mlflow.log_metrics({
        "val_auc_pr":                       auc_pr,
        "val_auc_roc":                      auc_roc,
        f"val_precision_at_{TOP_K}":        prec_at_k,
        f"val_recall_at_{TOP_K}":           recall_at_k,
    })

    print(f"\nVal AUC-PR  = {auc_pr:.4f}   (primary — use this for model comparison)")
    print(f"Val AUC-ROC = {auc_roc:.4f}  (reference only)")
    print(f"Precision@{TOP_K:<4} = {prec_at_k:.4f}  "
          f"({prec_at_k:.1%} of top-{TOP_K} flagged users are actual churners)")
    print(f"Recall@{TOP_K:<7} = {recall_at_k:.4f}  "
          f"(top-{TOP_K} contacts capture {recall_at_k:.2%} of all churners in val set)")

    # Feature importance — what the model actually learned from
    fi = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nFeature importance (gain):")
    print(fi.to_string(index=False))

    # Features with zero importance will be the null-heavy ones — confirms the
    # data-limitation analysis above.

    # ── 7. Save ───────────────────────────────────────────────────────────────
    MODELS.mkdir(exist_ok=True)
    out = MODELS / "lgbm_baseline.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {out}")

    model_info = mlflow.lightgbm.log_model(model, name="lgbm_baseline", input_example=X_train)
    print(f"MLflow run logged to mlflow.db  (run id: {mlflow.active_run().info.run_id})")

# %%
mv = mlflow.register_model(model_uri=model_info.model_uri, name="LightGBMChurnClassifier")
model_name = mv.name
model_version = mv.version

model_uri = f"models:/{model_name}/{model_version}"

print(f"Name: {model_name}")
print(f"Version: {model_version}")
print(model_uri)
# %%
