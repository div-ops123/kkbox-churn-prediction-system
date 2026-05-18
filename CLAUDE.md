# KKBox Churn Prediction

## Project Setup

Data is in data/:
- transactions.csv
- user_logs.csv
- members.csv
- train.csv (official labels — use as-is, do not regenerate)

Files are large → use DuckDB (do not load fully into memory).

Goal: Build a churn prediction model using LightGBM. EDA and cleaning are complete; next step is feature engineering.

## Churn Definition

A user whose subscription expired in month M is labeled is_churn = 1 if no new valid transaction (is_cancel = 0) appears within 30 days of their expiry date.

## Business Context

We predict churn 14 days before subscription expiry.
So for each user:
- features must only use data up to (membership_expire_date - 14 days)  ← feature_cutoff_date
- anything after that is leakage and must be excluded from training logic

Label cohort: March 2017 (users whose subscription expired in March 2017).
Feature cutoff per user: anchor_expiry_dt - 14 days → ranges Feb 15 to March 17.

---

## EDA Findings

### Dataset Shape

| File | Rows | Unique Users |
|---|---|---|
| members.csv | 6,769,473 | 6,769,473 |
| transactions.csv | 1,431,009 | 1,197,050 |
| user_logs.csv | 18,396,362 | 1,103,894 |
| train.csv | 970,960 | 970,960 |

Class imbalance: 8.99% churn (87,330 churned / 883,630 retained). Handle with scale_pos_weight ≈ 10.1 in LightGBM. Primary eval metric: AUC-PR (not AUC-ROC).

### Missing Values

| Column | Table | Missing % | Action |
|---|---|---|---|
| gender | members | 65.43% | Drop — too sparse and introduces demographic bias |
| bd (age) | members | 0% null, but 67.07% are <= 0 (invalid) | Treat as missing; only 32.83% of values (10–100) are usable |
| All transaction columns | transactions | 0% | No null handling needed |
| All user_log columns | user_logs | 0% | No null handling needed |

### Outliers

**members.csv — bd (age):**
- Range: -7,168 to 2,016 (clearly corrupted)
- 67.07% of values are <= 0 (sentinel for "not provided")
- 0.08% are > 100
- Only 32.83% fall in the plausible range 10–100 (mean=29.5, median=27, stddev=10.5)
- Decision: drop bd entirely; too noisy to salvage

**user_logs.csv — total_secs:**
- Max = 9,194,058 sec = ~2,554 hours in a single day (physically impossible)
- 4,200 rows exceed 86,400 sec (24 hours)
- p99 = 43,805 sec (~12 hours/day) — plausible for a power user
- Decision: hard cap at 86,400 first, then winsorize at p99 of training distribution

**user_logs.csv — num_100:**
- Max = 41,107 fully-played songs in a single day — bot/scraper artifact
- Decision: winsorize at p99.9 per-column or domain cap (e.g., 500 songs/day)

### Data Errors

**Error 1 — Legacy migration records (2,218 rows):**
- payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0
- All dated April–May 2015; membership_expire_date extends years into the future
- These are early KKBox system migration entries — plan metadata was not captured
- Action: flag as is_legacy_record = 1; exclude from plan-price and plan-days features but keep in transaction history counts

**Error 2 — membership_expire_date < transaction_date (5,106 rows):**
- Logically impossible: subscription expires before the transaction that created it
- Likely backdated cancellation records
- Action: exclude from anchor_expiry_dt calculation and feature_cutoff derivation; flag as is_bad_expiry = 1

**Error 3 — Cannot observe the full renewal window for the March 2017 cohort:**
- The 30-day renewal window for March expirations extends into April 2017
- transactions.csv only goes to March 31, 2017 — April data is absent
- This means we cannot independently verify labels for users expiring mid-to-late March
- Action: use train.csv labels as authoritative (KKBox generated them with full data); do not attempt to regenerate labels

**Error 4 — Incomplete transactions extract:**
- train.csv has 970,960 users; only ~40,227 have a verified March 2017 expiry in our transactions.csv
- The remaining ~930K users' March expiry records are missing from our extract
- For users without a verified March expiry in transactions.csv, use a global fallback feature_cutoff of 2017-02-15 and flag has_anchor = 0

**Error 5 — Future registration dates in members.csv (55,094 rows):**
- registration_init_time > 20170331 (registered after the eval period)
- Likely data integrity issue from a later system export
- Action: exclude these rows from tenure calculation; treat tenure_days as NULL for these users

### Other Anomalies

**user_logs.csv — date coverage:**
- Only covers March 1–31, 2017 (31 days)
- For users with feature_cutoff before March 1 (i.e., anchor_expiry before March 15), NO log features are available
- These users get has_log_data = 0; LightGBM handles NaN natively — do not impute 0 (absence of data != zero listening)

**transactions.csv — zero-price rows (21,448):**
- actual_amount_paid = 0 → promotional or trial users
- Flag as is_promo = 1; these users have different retention dynamics

**transactions.csv — overpaid rows (2,264):**
- actual_amount_paid > plan_list_price
- Almost entirely overlap with is_legacy_record rows (list_price = 0, actual > 0)
- Not a genuine payment anomaly; covered by the legacy record flag

**members.csv — registered_via = -1 (1 row):**
- Invalid channel value; treat as "unknown" or drop

**transactions.csv — far-future expire dates:**
- membership_expire_date up to 20361015 (year 2036)
- These are legacy migration users with effectively lifetime plans
- Not an error — legitimate long-term accounts; keep in history

### Key Distributional Facts (for feature engineering reference)

| Stat | Value |
|---|---|
| % of transactions with payment_plan_days = 30 | 85.11% — monthly plan dominates |
| Most common plan_list_price | 149 NTD (41.54%), then 99 NTD (28.49%) |
| is_auto_renew = 1 | 78.53% of transactions |
| is_cancel = 1 | 2.46% of transactions |
| Median user_log total_secs/day | 4,583 sec (~1.27 hrs) |
| Completion ratio (num_100/total songs) median | 0.79 — users mostly listen to full songs |
| Median active log days (March) | 18 out of 31 days |
| Transactions per user: median | 1, p99 = 3 — most users have sparse history |
