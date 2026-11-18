# Feature Schema — the 20 features the model trains on

Source of truth: `src/core/feature_module.py` (`FEATURE_COLS`, `CAT_COLS`, `_build_txn`,
`_build_log`, `_build_member`). Raw column meanings are in `docs/raw_data_descriptions.md`;
data problems and decisions are in `docs/dataset_overview.md`. This doc defines the
**engineered** features and explains why each can be missing.

---

## 1. Ground rules

- **One row per member**, indexed by `msno`. Columns are in `FEATURE_COLS` order:
  12 transaction → 6 engagement-log → 2 member. (`serve.py::log_pipeline_health_metrics`
  slices by position, so don't reorder.)
- **Feature cutoff**: every feature uses only data dated **on or before**
  `feature_cutoff_dt = expiry_date − 14 days`. The caller passes the expiry date; the cutoff
  is applied inside `build_features`, never by the caller.
- **Not features** (present in the stored parquet, never given to the model): `is_churn`
  (target) and `cohort_month` (used only for the time-based split).
- **Never used from the raw tables**: `members.city`, `members.bd`, `members.gender`,
  `transactions.payment_method_id`.
- **Transaction rows considered**: only rows with `membership_expire_date >= transaction_date`.
- **"Last" transaction** = the most recent one by `transaction_date` (`ROW_NUMBER() ... ORDER BY
  txn_dt DESC`, `rn = 1`). Two transactions on the same day are broken arbitrarily.
- **Legacy rows** (`payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0`,
  the Apr–May 2015 migration burst): they count in `n_txn`, `n_cancel`, `cancel_rate`, etc.,
  but if the *last* transaction is a legacy row the four `last_*` money/plan features are NULL.

---

## 2. Transaction features (12) — from `transactions`

| # | Feature | Type | Definition | Missing when |
|---|---|---|---|---|
| 1 | `n_txn` | int | Number of transactions up to the cutoff | No transaction on or before the cutoff (see §4) |
| 2 | `n_cancel` | int (stored float) | Number of those with `is_cancel = 1` | Same |
| 3 | `ever_cancelled` | **cat** 0/1 | 1 if any transaction was a cancellation | Same |
| 4 | `ever_promo` | **cat** 0/1 | 1 if any transaction had `actual_amount_paid = 0` (free/promo) | Same |
| 5 | `last_is_auto_renew` | **cat** 0/1 | Auto-renew flag on the most recent transaction | Same |
| 6 | `last_is_cancel` | **cat** 0/1 | Cancel flag on the most recent transaction | Same |
| 7 | `last_actual_amount_paid` | int (NTD) | Amount paid on the most recent transaction | Same, **or** last txn is a legacy row |
| 8 | `last_payment_plan_days` | int | Plan length (days) of the most recent transaction | Same, or legacy last txn |
| 9 | `last_plan_list_price` | int (NTD) | List price of the most recent transaction | Same, or legacy last txn |
| 10 | `last_discount` | int (NTD) | `plan_list_price − actual_amount_paid` of the most recent transaction (0 = no discount) | Same, or legacy last txn |
| 11 | `cancel_rate` | float | `n_cancel / n_txn`, rounded to 4 dp | Same |
| 12 | `days_since_last_txn` | int | Days from the most recent transaction to the cutoff | Same |

All 12 come out of one `GROUP BY msno` and are joined to the cohort with a `LEFT JOIN`, so a
member with no qualifying transaction is NaN in **all 12 at once** — including `n_txn` and
`n_cancel`, which are NaN, not 0.

---

## 3. Engagement-log features (6) — from `user_logs`

Per-day rows with `date <= cutoff`. Daily seconds are winsorized first:
`total_secs_w = LEAST(LEAST(total_secs, 86400), p99_secs)` (86400 = seconds in a day;
`p99_secs` is computed once at dataset-build time, stored in `feature_config.json`, and reused
at serving).

| # | Feature | Type | Definition | Missing when |
|---|---|---|---|---|
| 13 | `log_days` | int | Number of log rows (days with listening activity) up to the cutoff | No log row on or before the cutoff (see §4) |
| 14 | `avg_daily_secs` | float | Mean of `total_secs_w` over those days | Same |
| 15 | `total_secs_sum` | float | Sum of `total_secs_w` over those days | Same |
| 16 | `avg_completion_ratio` | float 0–1 | Mean over days of `num_100 / (num_25+num_50+num_75+num_985+num_100)`, i.e. share of songs played to the end | Same; also NULL if *every* log day had zero songs played (the per-day ratio is NULL when the denominator is 0) |
| 17 | `avg_daily_unq_songs` | float | Mean of `num_unq` (unique songs per day) | Same |
| 18 | `days_since_last_log` | int | Days from the most recent log day to the cutoff | Same |

---

## 4. Member features (2) — from `members`

| # | Feature | Type | Definition | Missing when |
|---|---|---|---|---|
| 19 | `registered_via` | **cat** int | Registration channel code (observed values in the 2017-03 training set: 3, 4, 7, 9, 13; KKBox doesn't document their meaning) | Member row dropped by the registration-date filter, or no `members` row |
| 20 | `tenure_days` | int | Days from `registration_init_time` to the cutoff | Same |

The filter in `_build_member` keeps a member only if their registration date is between
`2000-01-01` and **their own** `feature_cutoff_dt`. Outside that range both columns are NULL,
not a made-up value.

---

## 5. Categorical vs numeric

| Kind | Features |
|---|---|
| Categorical (5, `CAT_COLS`) | `last_is_auto_renew`, `last_is_cancel`, `ever_cancelled`, `ever_promo`, `registered_via` |
| Numeric (15) | the other 15 |

Categories are **frozen** from the training cohort (`freeze_cat_categories`), saved in
`feature_config.json`, and applied with `pd.Categorical(col, categories=frozen_list)` right
before the model sees the data (train, validate, serve). The stored parquet stays numeric
because drift (PSI) needs numeric input.

---

## 6. How missing values are handled

- **No imputation anywhere.** `build_features` returns NaN as-is ("absence != zero for this
  dataset"). XGBoost learns a default branch for missing values at every split.
- A categorical value that is null or **not in the frozen list** becomes NaN after the cast, so
  XGBoost treats it as missing as well.
- **Drift monitoring ignores NaNs** (`compute_psi` drops them), so a change in a feature's
  *null rate* is invisible to PSI. Null rates are only visible in `serve.py`'s health log
  (alerts only if overall null rate > 0.99).
- The only fallback value in the pipeline is `p99_secs = 86400.0` when a cohort has no log rows.

---

## 7. Why each group is missing

Three independent causes, each all-or-nothing within its group. **Receipts** column says where
the explanation is backed by repo code, and what is only inferred from the feature SQL.

| Group | Cause | Receipts |
|---|---|---|
| Log features (6) | `user_logs.csv` covers **2017-03-01 to 2017-03-31 only**. A member whose cutoff (expiry − 14 d) is before 03-01 has no log rows by construction. Remaining nulls are members with no listening rows. | `explore/date_issues_check.py` **Cell 6** (log date range + anchored members with/without log rows before their own cutoff; `dataset_overview.md` §7 reports 52.6%). Chart: `explore/figures/04_log_coverage_gap.png` (`visualize_findings.py`). |
| Transaction features (12) | The filter is `txn_dt <= cutoff AND membership_expire_date >= transaction_date`, inner-joined to the cohort. A member with no qualifying transaction before the cutoff gets nothing. The extract is also incomplete: for the original 970K `train.csv` set the null rate was ~96.7%. | Extract completeness: `explore/date_issues_check.py` **Cell 5** and `explore/cohort_check.py` (users with no transaction record / latest expiry after March); `dataset_overview.md` §3 Error 3. Legacy-row nulls of the `last_*` fields: **Cell 3**. **Inferred, not in `explore/`:** for the anchored cohort the exact reason a member has no pre-cutoff transaction (their expiry-setting transaction falling after the cutoff) comes from reading `_build_txn`; no explore script measures it. |
| Member features (2) | Registration date outside `[2000-01-01, cutoff]`. | Future registrations (> 2017-03-31, 55,094 rows): `explore/date_issues_check.py` **Cell 4**; pre-2000 registrations: `explore/eda.py` section 5 (`suspiciously_old`, `after_eval_period` counts); `dataset_overview.md` §3 Error 4. **Not established by `explore/`:** the observed 14.2% in the anchored training set is far above what those two counts give on their own. A member who registered *after their own cutoff but before 03-31* is also dropped (the filter uses the per-member cutoff), which is the likely missing piece, but no explore script measures it. `dataset_overview.md` says `members.csv` holds every user, while `raw_data_descriptions.md` says "not all users exist": unresolved, so a missing `members` row is not ruled out either. |

### Observed null rates (2017-03 dataset build, 35,518 rows, train + val + test)

| Group | Null rate |
|---|---|
| 12 transaction features | 61.1% |
| 6 log features | 51.0% |
| `registered_via`, `tenure_days` | 14.2% |

These were measured from the dataset-build parquet artifacts, **not** from an `explore/`
script. To reproduce, download `train/val/test.parquet` from the `dataset_build` MLflow run
and take `df[FEATURE_COLS].isna().mean()`. Within each group the nulls always co-occur (a
member is either missing all columns of the group or none).

---

## 8. References

| File | What it gives |
|---|---|
| `src/core/feature_module.py` | Definitions (this doc is derived from `_build_txn`, `_build_log`, `_build_member`) |
| `src/core/model_trainer.py` | `compute_p99_secs`, XGBoost params (`enable_categorical=True`) |
| `explore/date_issues_check.py` | Cells 3, 4, 5, 6 — the data-side causes of missingness |
| `explore/eda.py` | Missing-value counts per raw table, `registration_init_time` anomalies |
| `explore/cohort_check.py` | Train users with no transaction record |
| `docs/dataset_overview.md` | Summary of issues and decisions |
