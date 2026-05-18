# Dataset Limitations: Training Data vs Production Requirements

## What a production churn model needs

At serving time, for each user whose subscription is expiring in month M:

| Input | What you have in production |
|---|---|
| Subscription expiry date | Always known — it's in your subscription system |
| Transaction history up to cutoff | Full rolling history from account creation |
| Engagement logs up to cutoff | Rolling months of daily usage data |
| Member registration metadata | Live membership table |

The feature cutoff is `expiry_date - 14 days` — computed precisely per user from the expiry date you already know.

---

## What the training extract provides

| File | Rows | Coverage |
|---|---|---|
| transactions.csv | 1,431,009 | Ends March 31, 2017; only ~35K of 970K train users have a verified March expiry |
| user_logs.csv | 18,396,362 | **March 1–31, 2017 only** (one calendar month) |
| members.csv | 6,769,473 | Full table, but 55,094 rows have future registration dates (>20170317) |

---

## Gap 1 — Missing expiry anchors (affects 96.4% of training users)

- `train.csv` has 970,960 users. Only **35,433** have a March 2017 expiry in `transactions.csv`.
- The remaining **935,527** users' March subscription records are absent from the extract.
- Without an anchor, we cannot compute a precise per-user feature cutoff.
- **Workaround applied:** fallback `feature_cutoff_dt = 2017-02-15` (14 days before the earliest possible March expiry).
- `has_anchor = 0` flags these users in the dataset.

**Production impact:** None. The serving system always knows the exact expiry date. The fallback is a training-data-only approximation.

**Do not use `has_anchor` as a model feature.** It is always 1 at serving time — training on it would teach the model a pattern that doesn't exist in production.

---

## Gap 2 — Sparse transaction features (affects 96.7% of training users)

- Most transactions in our extract are dated in or around March 2017.
- With feature cutoffs mostly at Feb 15 (fallback), transactions dated after Feb 15 are outside the feature window and correctly excluded.
- Result: **96.7% of users have NULL for all transaction features** (`n_txn`, `last_is_auto_renew`, etc.).

**Production impact:** At serving time, the full transaction history up to `expiry - 14 days` is available. Nulls would be rare (only for brand-new accounts with no prior transactions).

**Model consequence:** Transaction features will have near-zero importance in the trained model because they are null for nearly all training rows. The trained model is not a reliable indicator of what transaction features would contribute with complete data.

---

## Gap 3 — Near-absent engagement log features (affects 98.2% of training users)

- `user_logs.csv` covers **March 1–31, 2017 only**.
- Feature cutoffs for 96.4% of users (fallback) fall on Feb 15 — before March 1.
- Even among the 35K anchored users, those expiring in the first half of March have cutoffs before March 1.
- Result: **98.2% of users have NULL for all log features** (`log_days`, `avg_daily_secs`, etc.).
- Only ~17,400 users (1.8%) have any log data in their feature window.

**Production impact:** At serving time, a properly maintained engagement log table would cover the full feature window. Log features like recency and listening intensity are strong churn predictors in literature — the training extract cannot demonstrate this.

---

## Gap 4 — Invalid member registration dates (affects ~11%)

- 55,094 rows in `members.csv` have `registration_init_time > 20170317` (registered after the eval period).
- These are excluded from `tenure_days` computation.
- This is a genuine data quality issue — it may persist in production if the membership table has integrity problems.
- `tenure_days` and `registered_via` are NULL for these users (~11.4%).

**Production impact:** Could persist. Recommend a data validation check upstream for any user with a future registration date.

---

## Summary: Null rate by feature group

| Feature group | Null rate in training | Root cause | Null rate in production |
|---|---|---|---|
| Transaction features (`n_txn`, `last_is_auto_renew`, etc.) | ~96.7% | Incomplete extract | Near 0% |
| Log features (`log_days`, `avg_daily_secs`, etc.) | ~98.2% | Log coverage: March only | Near 0% |
| Member features (`tenure_days`, `registered_via`) | ~11.4% | Invalid reg dates | ~11% (data quality issue) |

---

## What this means for the model

The LightGBM model trained on this data will:
1. Have **very low importance for transaction and log features** — they are null for 96–98% of rows.
2. Derive most signal from **`tenure_days`**, **`registered_via`**, and the binary flags (`ever_cancelled`, `ever_promo`, `has_log_data`).
3. **Not generalize well to a production system with complete features** — it has never seen what a user with full transaction history looks like.

This is a known limitation of the competition dataset. The model is a valid baseline for the competition but would require re-training on production data before deployment.

---

## What a production pipeline would need

1. **Transaction table:** rolling history per user, updated daily, covering from account creation to `NOW()`.
2. **Engagement log table:** rolling N-month window per user (at minimum 3 months recommended).
3. **Subscription table:** current expiry date per active user — always available in a real system.
4. **Members table:** deduplicated, validated registration records (no future dates).
