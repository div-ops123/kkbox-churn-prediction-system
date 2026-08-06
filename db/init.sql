-- KKBox Churn Prediction System — Database DDL
--
-- Runs automatically inside the Postgres container on first `docker compose up`
-- via /docker-entrypoint-initdb.d/. Has no effect on subsequent starts
-- (volume already initialised).
--
-- Indexes are NOT defined here. seed.py creates them after bulk-loading the
-- source tables — building an index from scratch after COPY is 3-5× faster
-- than maintaining it row-by-row during the load.

-- ── Source tables ─────────────────────────────────────────────────────────────
-- Date columns are stored as INTEGER (YYYYMMDD) to match the CSV source format
-- and remain compatible with DuckDB SQL queries in the feature module.

CREATE TABLE IF NOT EXISTS transactions (
    msno                   VARCHAR(64) NOT NULL,
    payment_method_id      SMALLINT,
    payment_plan_days      SMALLINT,
    plan_list_price        INTEGER,
    actual_amount_paid     INTEGER,
    is_auto_renew          SMALLINT,
    transaction_date       INTEGER,            -- YYYYMMDD
    membership_expire_date INTEGER,            -- YYYYMMDD
    is_cancel              SMALLINT
);

CREATE TABLE IF NOT EXISTS user_logs (
    msno       VARCHAR(64) NOT NULL,
    date       INTEGER     NOT NULL,           -- YYYYMMDD
    num_25     INTEGER,
    num_50     INTEGER,
    num_75     INTEGER,
    num_985    INTEGER,
    num_100    INTEGER,
    num_unq    INTEGER,
    total_secs REAL
);

CREATE TABLE IF NOT EXISTS members (
    msno                   VARCHAR(64) NOT NULL,
    city                   SMALLINT,
    bd                     SMALLINT,
    gender                 VARCHAR(6),         -- nullable: ~65% missing in source
    registered_via         SMALLINT,
    registration_init_time INTEGER             -- YYYYMMDD
);

CREATE TABLE IF NOT EXISTS train_labels (
    msno     VARCHAR(64) NOT NULL,
    is_churn SMALLINT    NOT NULL
);

-- ── Label store ───────────────────────────────────────────────────────────────
-- Written monthly by the labeling pipeline after the 30-day renewal window
-- closes for the previous cohort.

CREATE TABLE IF NOT EXISTS labels (
    msno                VARCHAR(64) NOT NULL,
    anchor_expiry_date  DATE        NOT NULL,
    feature_cutoff_date DATE        NOT NULL,
    is_churn            SMALLINT    NOT NULL CHECK (is_churn IN (0, 1)),
    labeled_date        DATE        NOT NULL,
    PRIMARY KEY (msno, anchor_expiry_date)     -- one label per user per cohort
);

-- ── Prediction store ──────────────────────────────────────────────────────────
-- Written daily by the serving pipeline. Read by the Score API and monitoring.

CREATE TABLE IF NOT EXISTS predictions (
    id             BIGSERIAL   PRIMARY KEY,
    msno           VARCHAR(64) NOT NULL,
    score          REAL        NOT NULL,
    risk_tier      VARCHAR(4)  NOT NULL CHECK (risk_tier IN ('HIGH', 'MED', 'LOW')),
    scoring_date   DATE        NOT NULL,
    expiry_date    DATE        NOT NULL,
    model_version  VARCHAR(64) NOT NULL,       -- MLflow run_id of the production model
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prevents duplicate rows if the batch job retries on the same day.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_msno_scoring
    ON predictions (msno, scoring_date);

-- ── Monitoring metrics store ──────────────────────────────────────────────────
-- Written by all three monitoring tracks (data_drift, model_perf, pipeline_health).
-- Read by maybe_trigger_retraining() to decide whether to kick off retrain.py;
-- otherwise queried directly via SQL — no dashboard layer on top of it.

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id              BIGSERIAL   PRIMARY KEY,
    run_date        DATE        NOT NULL,
    pipeline_type   VARCHAR(20) NOT NULL,      -- 'data_drift'|'model_perf'|'pipeline_health'
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    REAL,
    alert_triggered BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
