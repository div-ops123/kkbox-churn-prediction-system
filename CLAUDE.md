# KKBox Churn Prediction

> **Never add 'Co-Authored-By' to git commit messages**

## Python Coding Standards & SWE Principles

1. Comments & Documentation
• Use # for single-line comments.
• Use triple quotes for multi-line comments.
• Use docstrings (Google style) for all public modules, classes, and functions.
# This is a single-line comment
"""
This is a multi-line comment.
It can span several lines.
"""
def enroll_student(student, course):
"""Enroll a student in a course.
Args:
student (Student): The student to enroll.
course (Course): The course to enroll in.
Returns:
bool: True if successful, False otherwise.
"""
pass

2. Function & Method Design
• Use verbs for function names.
• One function, one job (single responsibility).
• Prefer short functions (<20 lines).
• Limit arguments (1-3 preferred).
• Avoid flag arguments; split into separate functions instead.

---

## Project Setup

Data is in data/:
- transactions.csv
- user_logs.csv
- members.csv
- train.csv (official labels — use as-is, do not regenerate during experimentation)

Files are large → use DuckDB (do not load fully into memory).

Goal: Build a production churn prediction system.

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

---

## Infrastructure (Step 1)

### Stack

- **PostgreSQL 16** in Docker (`kkbox_postgres`, container port 5432, host port **5433**)
- **MLflow 3.12.0** in Docker (`kkbox_mlflow`, port 5000), backed by Postgres for metadata and a named volume for artifacts
- Both defined in `docker-compose.yml`; brought up with `docker compose up -d`

### Why port 5433 (not 5432)

A local Windows PostgreSQL installation occupies port 5432 on this machine. Docker maps the container's internal port 5432 to host port 5433 to avoid the conflict. All connection strings and defaults use 5433 accordingly (`.env`, `db/seed.py`, `src/feature_module.py`).

### Database schema (`db/init.sql`)

Runs automatically on first container start via `/docker-entrypoint-initdb.d/`. Contains DDL for:
- **Source tables**: `transactions`, `user_logs`, `members`, `train_labels` — dates stored as INTEGER (YYYYMMDD) to match CSV format and stay compatible with DuckDB queries
- **Prediction store**: `predictions` — one row per user per scoring day; UNIQUE index on `(msno, scoring_date)` prevents duplicate batch runs
- **Label store**: `labels` — monthly churn outcomes; PRIMARY KEY `(msno, anchor_expiry_date)` enforces one label per user per cohort
- **Monitoring store**: `monitoring_metrics` — written by all three monitoring tracks (data drift, model performance, pipeline health)
- No indexes in `init.sql` — bulk-load performance: indexes built by `db/seed.py` AFTER `COPY` completes (3-5× faster than maintaining B-tree during load)

### Seeding (`db/seed.py`)

One-time CSV → Postgres loader. Key decisions:
- `conn.autocommit = True` — avoids holding a multi-GB transaction open; each COPY commits immediately
- `COPY FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')` — `NULL ''` treats empty strings as NULL (needed for the ~65%-null `gender` column in members.csv)
- Idempotent: checks `SELECT COUNT(*)` before each table; skips if rows already exist
- `SET maintenance_work_mem = '512MB'` before index creation — more sort RAM → faster index builds
- 12 indexes created after all tables are loaded

### MLflow tracking URI

`src/train.py` reads `MLFLOW_TRACKING_URI` from the environment, falling back to `sqlite:///mlflow.db` if not set. This means:
- Without Docker → falls back to local `mlflow.db` (dev workflow unchanged)
- With Docker running → `.env` sets `http://localhost:5000` → logs to the containerized MLflow server

---

## Feature Module Refactor (Step 2)

### The problem this solves

The original `src/features.py` was a flat script with inline DuckDB SQL. If the serving pipeline duplicated that SQL (even slightly differently), features computed at serving time could silently diverge from features computed at training time — the model would score on a different distribution than it was trained on.

### Solution: `src/feature_module.py`

A single callable `build_features()` that both training and serving import:

```python
build_features(
    msno_list,      # list of user IDs
    expire_dates,   # subscription expiry date per user
    p99_secs,       # winsorization threshold — load from feature_config.json
    data_source,    # "csv" (training) or "postgres" (serving)
    data_dir,       # CSV directory, default data/
    pg_conn_str,    # Postgres URI, required when data_source="postgres"
) -> pd.DataFrame   # indexed by msno, columns = FEATURE_COLS
```

Also exports `FEATURE_COLS` (20 features) and `CAT_COLS` (5 categoricals) — single source of truth imported by both `features.py` and `train.py`.

### Key design decisions

**feature_cutoff_date computed internally, not passed by caller.**
The caller passes `expire_date` (the subscription expiry). The function internally computes `feature_cutoff_dt = expire_date - 14 days`. This prevents the serving pipeline from accidentally passing today's date instead of the correct cutoff, which would silently include post-cutoff data (leakage).

**p99_secs injected as a parameter, never recomputed.**
`features.py` computes p99_secs once from the training distribution and saves it to `models/feature_config.json`. The serving pipeline loads that file and passes the frozen value to `build_features()`. Recomputing from the serving-time distribution would give a different threshold as listening patterns drift — silent feature skew.

**Data source abstraction via DuckDB views.**
`_register_sources()` creates views `v_transactions`, `v_user_logs`, `v_members` pointing at either CSV files or Postgres tables. All four feature-building functions (`_build_txn`, `_build_log`, `_build_member`, plus the final join) query these views — the SQL is identical for both data sources.

**Postgres mode uses DuckDB's postgres extension.**
`ATTACH '{pg_conn_str}' AS pg (TYPE POSTGRES, READ_ONLY)` — DuckDB runs the same analytical SQL against Postgres without pulling all rows into Python first. For the serving cohort (a few thousand users expiring in 13-15 days) this is efficient enough; for large backfills, consider batching.

**`tenure_days` filter is per-user, not hardcoded.**
Original: `WHERE registration_init_time <= 20170317` (hardcoded to training cohort max). Refactored: `WHERE ... <= c.feature_cutoff_dt` — correct for any future cohort without code changes.

### How `src/features.py` changed

Reduced to a thin training wrapper:
1. Stage 1: anchor logic (training-specific — builds user cohort from `train.csv` + `transactions.csv`)
2. Stage 2: compute p99_secs, save to `models/feature_config.json`
3. Stage 3: call `build_features()` with the cohort's expire dates
4. Stage 4: join `is_churn`/`has_anchor` metadata back in, export parquet

The inline Stages 2-5 (txn, log, member feature SQL + join) were removed — that logic now lives exclusively in `feature_module.py`.

### Verified output

Running `python src/features.py` after the refactor produces identical null rates to the pre-refactor run:
- Transaction features: 96.7% null (extract gap)
- Log features: 98.2% null (March-only coverage)
- Member features: 12.2% null (future reg dates)

---

## Serving Pipeline & Score API (Steps 3 & 4)

### Files added

| File | Purpose |
|---|---|
| `src/config.py` | Frozen dataclasses (`PostgresConfig`, `MLflowConfig`, `ServingConfig`, `ApiConfig`) loaded from env vars. `POSTGRES_PASSWORD` has no default — fails fast if absent. |
| `src/model_loader.py` | Factory + abstract loader. `make_model_loader(use_mlflow=True/False)` returns `MLflowModelLoader` or `PickleModelLoader`. Callers never instantiate directly. |
| `src/risk_tier.py` | Strategy pattern. `make_tier_strategy(tier_config).assign(scores)` → `list["HIGH"\|"MED"\|"LOW"]`. Add new strategies as subclasses without touching the pipeline. |
| `src/serve.py` | Prefect `@flow` with 8 `@task` functions. CLI: `python src/serve.py --date YYYY-MM-DD --no-mlflow`. |
| `src/api/main.py` | FastAPI app factory + lifespan (pool lifecycle). |
| `src/api/schemas.py` | Pydantic schemas: `PredictionRecord`, `CohortResponse`, `ErrorResponse`, `HealthResponse`. |
| `src/api/exceptions.py` | Custom exceptions + all exception handlers registered in `create_app()`. |
| `src/api/dependencies.py` | FastAPI `Depends()` providers for pool and repository. |
| `src/api/repositories/base.py` | `AbstractPredictionRepository` — 3-method interface. |
| `src/api/repositories/postgres.py` | `PostgresPredictionRepository` — owns all SQL; uses server-side cursor for cohort > 50K rows. |
| `src/api/routers/score.py` | `GET /score/{user_id}` |
| `src/api/routers/cohort.py` | `GET /cohort?date=YYYY-MM-DD` |
| `models/risk_tiers_config.json` | Versioned tier thresholds: HIGH ≥ 0.5, MED [0.2, 0.5), LOW < 0.2. |


### LightGBM Booster vs sklearn predict_proba

`lgb.train()` returns a `lgb.Booster` with `.predict()` (returns probabilities directly, 1-D), not `.predict_proba()`. `mlflow.lightgbm.load_model()` also returns a `Booster`. The `score_cohort` task dispatches correctly:
- If `hasattr(model, "predict_proba")` → sklearn wrapper, take `[:, 1]`
- Else → `Booster.predict()` already returns churn probabilities

### Serving pipeline design decisions

**One `psycopg2.connect()` per task, not a shared pool.**
Tasks run sequentially over up to 2 hours. A single long-lived connection would risk TCP timeout. Per-task connections (opened and closed in a `_pg_conn()` context manager) are safer and simpler.

**Upsert is the write strategy.**
`INSERT ... ON CONFLICT (msno, scoring_date) DO UPDATE SET ...` means re-running the pipeline on the same day is always safe. The unique index `idx_pred_msno_scoring` enforces this at the DB level.

**Parquet write and health metrics are non-fatal.**
The DB write (`write_predictions_postgres`) is the authoritative output. If Parquet write fails, the pipeline logs WARNING and continues — downstream marketing/CRM reads from Postgres, not the file. Health metrics failure sets `status="partial"` but never kills serving.

**Empty cohort is not an error.**
`len(cohort_df) == 0` triggers an early exit: one monitoring metric is written (`cohort_size=0, alert_triggered=True`), the flow returns `status="success"`. This prevents false failure alerts on days when no subscriptions expire in the 13–15-day window.

**Retry configs are per-task, not global.**
DB reads retry 3× (transient drops). MLflow loads retry 2× (network latency). Inference retries 0× (deterministic — a failure is a code bug, not transient). Parquet and metrics tasks retry but do not abort the flow on exhaustion.

### Score API design decisions

**Repository pattern isolates all SQL.**
`AbstractPredictionRepository` defines exactly 3 methods (`get_latest_for_user`, `get_cohort_for_date`, `health_check`). Routers depend on the abstract class. In tests, override via `app.dependency_overrides[get_prediction_repository] = lambda: InMemoryRepo()` — no DB needed.

**`ThreadedConnectionPool` created once in lifespan.**
`psycopg2.pool.ThreadedConnectionPool(minconn, maxconn)` is created in the FastAPI `@asynccontextmanager lifespan` and stored on `app.state.pool`. `get_connection_pool()` retrieves it; it never creates a new pool. Pool is closed on shutdown via `pool.closeall()`.

**`GET /cohort` returns 200 with empty list, never 404.**
A date with no predictions is valid (the pipeline may not have run yet, or the cohort was empty). `{"scoring_date": "...", "count": 0, "predictions": []}` is the correct response. Only a missing/malformed `date` param returns 4xx.

**Two-step date validation on `GET /cohort`.**
`Query(pattern=r"^\d{4}-\d{2}-\d{2}$")` catches format errors → 422 (FastAPI). `date.fromisoformat()` catches semantic errors (e.g. `2026-13-99`) → 400 via the `ValueError` handler. This gives callers a meaningful distinction between "wrong format" and "invalid calendar date".

**Uniform `ErrorResponse` envelope.**
Every 4xx/5xx returns `{"error": "...", "message": "...", "field": null, "retry_after_seconds": null}`. No raw FastAPI/Pydantic error objects are ever exposed. All exception handlers are registered in `register_exception_handlers(app)` called once from `create_app()`.

### Risk tier config design decisions

`models/risk_tiers_config.json` is the versioned source of truth for thresholds. `load_risk_tiers_config()` validates at startup that the intervals are contiguous and cover [0.0, 1.0] with no gap or overlap — a misconfigured file fails fast before any task runs. The `FixedThresholdStrategy` iterates tiers in config order and assigns the first matching interval, so tier order matters (HIGH is checked first). Adding a new tier (e.g. CRITICAL) requires only: editing the JSON + adding `"CRITICAL"` to the DB `CHECK` constraint + adding `Literal["CRITICAL"]` to Pydantic schemas — zero changes to `serve.py`.

### Start the API

```bash
# With Docker running (Postgres on port 5433):
uv run uvicorn src.api.main:app --reload --port 8000

# Smoke test:
curl http://localhost:8000/health
curl "http://localhost:8000/cohort?date=2017-03-01"
curl "http://localhost:8000/score/<msno>" # Zy4W5mkOlk8+qCMQD4K+MFH7LXuRi8tGeiaFBfCTu78=
curl "http://localhost:8000/score/nonexistent_user"
curl "http://localhost:8000/score/user with spaces"
```

OpenAPI docs auto-generated at `http://localhost:8000/docs`.

---

## Labeling Pipeline (Step 5)

### Files added / modified

| File | Change | Purpose |
|---|---|---|
| `src/config.py` | Modified | Added `LabelingConfig` dataclass + `load_labeling_config()` function |
| `src/label_module.py` | New | Pure labeling domain logic — no Prefect, no psycopg2, no side effects |
| `src/label.py` | New | Prefect `@flow` with 4 `@task` functions. CLI: `python src/label.py --cohort-month YYYY-MM` |

### Design decisions

**Two label sources, same cohort-build logic.**
`label_source="train_csv"` — joins the cohort against the official train.csv / `pg.train_labels` table; required for March 2017 because our transactions.csv lacks April data (Error 3). `label_source="transactions"` — derives labels from renewal patterns in the 30-day window; used for future cohorts where the full window is observable.

**`build_cohort()` in `label_module.py` is NOT `features.py` Stage 1.**
`features.py` Stage 1 starts with all 970K train.csv users and computes per-user anchors (including the `has_anchor=0` fallback). `build_cohort()` starts from transactions and only returns users with a confirmed expiry in the target month (~40K for March 2017). The `labels` table only stores users with a verified `anchor_expiry_date`; users with `has_anchor=0` appear only in `train_labels`.

**DuckDB source abstraction via `_register_txn_source()`.**
Creates a `v_transactions` view pointing at either CSV files or `pg.transactions`. When `data_source="postgres"`, also attaches the DB as `pg` — making `pg.train_labels` accessible without a second attach call.

**`renewal_window_days` injected as integer into f-string SQL, not as a bind parameter.**
DuckDB's INTERVAL syntax (`INTERVAL '30 days'`) does not support bind parameters. The value is validated as a positive integer in `load_labeling_config()` before reaching the SQL, preventing injection.

**NaN `is_churn` is possible only for `label_source="train_csv"`.**
Users in the cohort (confirmed March expiry in transactions) who are absent from `train_labels` get `is_churn = NULL`. `write_labels_postgres` drops these rows before upserting and logs a WARNING; the `labels` table schema enforces `NOT NULL`.

**Metrics alert thresholds:**
- `cohort_size < 100` → alert (unexpectedly small cohort)
- `churn_rate > 0.5` → alert (unexpectedly high churn rate)
- `missing_label_rate > 0.05` → alert (>5% of cohort missing labels in train_csv mode)

### Run the labeling pipeline

```bash
# March 2017 training cohort — use official KKBox labels from train_labels table:
uv run python src/label.py --cohort-month 2017-03 --label-source train_csv

# Future cohort — derive labels from transaction renewals:
uv run python src/label.py --cohort-month 2017-04 --label-source transactions

# Dev / no Postgres — read from CSV files (still needs Postgres for writing labels):
uv run python src/label.py --cohort-month 2017-03 --label-source train_csv --data-source csv
```
