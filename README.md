# KKBox Churn Prediction — Production ML System

A production-grade churn prediction system built on the [KKBox Music Streaming dataset](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). Predicts which subscribers will churn 14 days before their subscription expires, so marketing and CRM teams can intervene.

---

## Demo

<!-- Video walkthrough — replace this comment block with the uploaded video/GIF -->
<!-- [![Demo video](docs/demo-thumbnail.png)](docs/demo.mp4) -->

*(Video walkthrough coming soon.)*

---

## The Problem

KKBox loses revenue when subscribers cancel and the business finds out too late to act. Churn prediction needs to be:

- **Early enough** — scored 14 days before expiry, so retention campaigns can reach users while they're still active
- **Reliable** — a model that silently degrades after retraining destroys trust faster than no model at all
- **Operationalized** — predictions sitting in a notebook don't help CRM; they need to land in a queryable store with an API on top

The dataset: 970K users, 8.99% churn rate, 18M+ user log events, transactions through March 2017.

---

## The Solution

An end-to-end ML platform with seven decoupled pipelines that mirror real industry architecture — each independently invocable, chained together only for the common-case flow:

<!-- Architecture diagram: replace this comment block with your architecture image when ready -->
<!-- ![Architecture Diagram](docs/architecture.png) -->

```
DATA SOURCES (PostgreSQL)
transactions │ user_logs │ members
        │
        ▼
FEATURE ENGINEERING MODULE
core/feature_module.py — single shared library. Called separately by
the dataset-build pipeline, the serving pipeline, and (on the standalone
CLI backfill path only) the monitoring pipeline's drift track.
Input: msno + expiry_date. Output: 20-feature vector per user.
        │
        ├─────────────────────────────────┬───────────────────────────┐
        ▼                                 ▼                           │
RETRAIN PATH (monthly, or perf-triggered)  SERVING PATH (daily)        │
pipelines/retrain.py orchestrates:         pipelines/serve.py          │
                                            1. Users expiring in       │
1. build_dataset.py — pull labels,            exactly 14 days         │
   build features, rolling train/val/test   2. Build features         │
   split, log artifacts to MLflow           3. Load production model  │
2. train.py — fit LightGBM on train/val,    4. Score cohort           │
   register a new, unpromoted version       5. Apply risk tiers       │
3. validate.py — score candidate vs         6. Write predictions      │
   production on held-out test, alias          (Postgres + Parquet)   │
   'challenger' if the candidate wins       7. Chain drift check ─────┤
        │                                      (Track 1, in-memory    │
        ▼                                      feature matrix,       │
MODEL REGISTRY (MLflow)                        non-fatal)             │
versions + metrics +                                │                 │
production / challenger aliases                     ▼                 │
                                            PREDICTION STORE (PG)      │
                                                     │                 │
                                                     ▼                 │
                                            SCORE API — FastAPI        │
                                            GET /health                │
                                            GET /score/{user_id}       │
                                            GET /cohort?date=...       │
                                                                       │
LABELING PIPELINE (monthly)                                           │
pipelines/label.py — derive is_churn labels                           │
from transaction renewals within 30 days                              │
                                                                       │
MONITORING PIPELINE ───────────────────────────────────────────────────┘
pipelines/monitor.py
Track 1 — Data drift: chained onto serve.py's run above in the common
          case; standalone `monitor.py drift --date` rebuilds features
          for backfill or recovery after a failed serving run
Track 2 — Model performance (monthly): joins predictions + labels,
          triggers pipelines/retrain.py when AUC-PR < threshold
Track 3 — Pipeline health: computed inline by every pipeline above
All three tracks: computed and logged via Prefect run output — no
persistence layer, no dashboard (see Monitoring Strategy below).
```

### Tech Stack

| Component | Choice | Why |
|---|---|---|
| Source DB | PostgreSQL 16 on Docker | Handles 20M+ rows; indexed date-range queries |
| Feature engineering | DuckDB (inside Python) | Queries CSVs + Postgres without loading into RAM |
| Pipeline orchestration | Prefect | Python-native `@flow`/`@task`, simpler for solo local iteration than standing up Airflow's scheduler/webserver/metadata DB — trade-off below |
| Experiment tracking | MLflow | Self-hosted, open source; covers tracking + registry + promotion in one tool |
| Model | LightGBM | Native null handling, fast training, well-calibrated probabilities for risk tiers |
| Prediction store | PostgreSQL table | Same instance as source; low write volume, simple queries |
| Drift monitoring | Custom PSI (no Evidently) | Full control over bin-edge anchoring and alert logic |
| Score API | FastAPI + Uvicorn | Async, auto OpenAPI docs, sub-200ms reads from indexed table |
| Containerization | Docker Compose (infra only) | Single-command Postgres + MLflow; pipelines/API still run as bare Python processes — see Learnings below |

### Key Features

- **Leakage-safe feature engineering** — `feature_cutoff_dt = expire_date - 14 days` is enforced *inside* `build_features()`, not left to the caller, so training and serving can't accidentally see post-cutoff data.
- **Zero train/serve skew** — training, serving, and drift monitoring all call the same `core/feature_module.py` function; there is no second implementation to drift out of sync.
- **Automatic retraining trigger** — `monitor.py`'s performance track calls `retrain.py` as a direct sub-flow when monthly AUC-PR drops below threshold, no manual intervention required.
- **Drift-aware, not just accuracy-aware monitoring** — PSI per feature against a training-time baseline, anchored bin edges, distinguishes real model degradation from expected structural drift (see the March 2017 log-coverage example below).

---

## The Results

**Technical metrics — held-out test set** (`retrain.py` validation step, n=5,328, mixed expiry dates across the training cohort):

| Metric | Value |
|---|---|
| AUC-PR | **0.976** |
| AUC-ROC | 0.816 |
| Precision@K (K=100) | 1.0 |
| Recall@K (K=100) | 0.021 |
| Features engineered | 20 (transaction, log, member) |
| Feature training-serving skew | **Zero** — single shared module |

**Technical metrics — production cohort** (`monitor.py performance`, actual scored predictions joined against actual labels, cohort_month=2017-03, n=1,055, all expiring 2017-03-15):

| Metric | Value |
|---|---|
| AUC-PR | 0.870 |
| AUC-ROC | 0.490 |
| Precision@K (K=100) | 0.90 |
| Recall@K (K=100) | 0.098 |
| Churn rate in this cohort | 86.7% (915 of 1,055) |

This cohort is far more churn-heavy (86.7%) than the training population (90.5% among the 35,518 confirmed-anchor users, 8.99% in KKBox's raw dataset) because it's a single day's worth of expiring users, not a representative sample. At that base rate, a classifier outputting random scores would already land near 0.87 AUC-PR, and AUC-ROC on a cohort this lopsided is close to uninformative — 0.49 here reflects the cohort's composition, not the model's ranking ability. The held-out test set above (mixed expiry dates) is the more meaningful accuracy signal; this table exists to show what Track 2 actually reports in production, caveats included.

**Business-relevant metrics:**

| Metric | Value |
|---|---|
| Training cohort size | 35,518 verified-anchor users |
| Serving cohort size (single date, actionable by CRM) | 1,055 users |
| Pipelines with retry/alerting | All 6 (build_dataset, label, train, validate, serve, monitor) — retrain.py orchestrates them and inherits their retries |
| API endpoints | 3 (`/health`, `/score/{user_id}`, `/cohort`) |

**User testimonials:** None yet — this is a pre-launch portfolio project, not a deployed product with users. The honest equivalent at this stage is the monitoring output below: proof the system correctly distinguishes real problems from noise, which is what a CRM team would actually need to trust before acting on its output.

**Monitoring output (2017-03-01 drift run, chained from `serve.py`):**
- 7 of 20 features triggered PSI > 0.2 alerts: `n_txn`, `last_is_auto_renew`, `days_since_last_txn` (transaction features) and `log_days`, `total_secs_sum`, `avg_completion_ratio`, `days_since_last_log` (log features)
- Log features: the known structural gap — this cohort's feature_cutoff (March 1) sits right at the edge of `user_logs.csv`'s March 1-31 coverage, so near-zero log history is expected, not model degradation
- Transaction features: likely a cohort-composition effect — 1,055 users all expiring on the same single day vs. a training baseline spanning many expiry dates across the month, so anything tied to billing-cycle timing (transaction recency, auto-renew status right before expiry) can differ from the training-set average without any real drift happening
- System correctly flagged alerts and continued without a false retraining trigger

---

## Technical Deep Dive

### Data Pipeline

**Challenge:** The KKBox dataset has a critical coverage gap. `user_logs.csv` only covers March 1-31, 2017. Users expiring in early March have `feature_cutoff = expiry - 14 days` falling before March 1 — meaning no log data is available for them. The label dataset (970K users in `train.csv`) contains ~930K users without a verified March expiry record in the transactions extract.

**Decision:** Train only on the 35,518 users with a confirmed `anchor_expiry_date` in the transactions table. Users without a verified anchor (`has_anchor=0`) have feature_cutoff=2017-02-15 and no log features — training on them would produce a 97% null feature matrix and AUC-PR=0.39. The clean 35,518-user cohort yields AUC-PR=0.976 on the held-out test set (see The Results above).

**Feature cutoff enforcement:** `feature_cutoff_dt = expire_date - 14 days` is computed *inside* `build_features()`, not passed by the caller. This prevents the serving pipeline from accidentally passing today's date and silently including post-cutoff data (leakage).

### Model Approach

LightGBM with `scale_pos_weight=10.1` to handle the 8.99% churn class imbalance. Primary metric: AUC-PR (not AUC-ROC) because the positive class is rare and AUC-ROC is optimistic under class imbalance.

20 features across three categories:
- **Transaction features** (8): recency, frequency, plan type, payment patterns, cancellation history
- **Log/behavioral features** (7): listening days, total seconds, song completion rate, days since last activity
- **Member features** (5): registration channel, tenure, city

The `p99_secs` winsorization threshold (43,573.98 sec/day) is computed once from the training distribution by `build_dataset.py` and logged to MLflow as `feature_config.json`, part of the `dataset_build` run's artifacts (see Key Decision: Dataset Storage below). The serving and monitoring pipelines fetch this frozen value from whichever dataset trained the current production model — never a local file, never recomputed from the serving distribution, which would otherwise silently change the feature scale as listening patterns drift.

### Monitoring Strategy

Three independent tracks in `src/pipelines/monitor.py`, each with its own failure domain:

**Track 1 — Data Drift (daily):** PSI computed for all 20 features against `baseline_features.parquet` — an MLflow artifact from the `dataset_build` run that trained the current production model, fetched fresh each run, never a local file. Bin edges are anchored to the training distribution — never recomputed from serving data. Alert threshold: PSI > 0.2. Log-based features near month-start are expected to alert (structural coverage gap, not model degradation).

**Track 2 — Performance (monthly):** Join month-M predictions with month-M labels on `msno`. Compute AUC-PR, AUC-ROC, Precision@100, Recall@100. Alert threshold: AUC-PR < 0.45 triggers automatic retraining via sub-flow call.

**Track 3 — Pipeline health (inline):** Cohort size, null rates, and run status computed and logged by each pipeline at execution time. Alert threshold: cohort_size < 100.

All three tracks log to the Prefect run output only — nothing is persisted. `maybe_trigger_retraining` acts on the in-memory perf dict `compute_and_log_performance` returns within that same flow run, not a re-query of any store. A `monitoring_metrics` table used to sit here; it was removed since nothing — no dashboard, no automated trigger, no manual query — ever read it back. Logging what's already computed costs nothing extra; a table nothing queries is dead weight.

---

## Learnings & Trade-offs

**What worked:**
- The shared feature module (`core/feature_module.py`) eliminated train/serve skew outright — not "monitored for," actually structurally impossible, since there's only one implementation to diverge from.
- Chaining drift monitoring onto serving (passing the in-memory feature matrix instead of rebuilding it) removed a genuine double-compute with zero new infrastructure.
- The custom 40-line PSI implementation was faster to get exactly right — anchored bin edges — than configuring Evidently to do the same thing.

**What didn't / what's still missing:**
- The app itself isn't containerized — Docker Compose runs the infra (Postgres, MLflow), but the pipelines and API still run as bare `python`/`uvicorn` processes on the host. That's inconsistent: if infra is containerized for reproducibility, the app driving it should be too.
- Prefect pipelines are decorated (`@flow`/`@task`) but not actually deployed on a schedule — "daily serving" and "monthly training" are manual invocations today, not live automation. See the Prefect vs. Airflow trade-off below for why that gap exists.
- No live deployment. There's no URL to hit directly — reproducing this requires cloning the repo and standing up the stack locally, which is a real gap for anyone trying to evaluate it quickly.

**What I'd do differently:**
- Containerize the app (pipelines + API) alongside the infra services, so `docker compose up` is the entire reproduction story instead of "start infra, then remember to run five Python commands by hand."
- Deploy the Score API and a scheduled `serve.py` run to a low-cost host (Railway/Fly.io) so there's a real, live endpoint to hit — even a slow one — instead of asking a reviewer to take the metrics on faith.
- Stand up an actual Prefect worker/deployment now, since "zero infra for local dev" (see below) was never meant to substitute permanently for real scheduling.

### Key Decision: Feature Consistency Mechanism

**Options considered:**

| Option | Guarantee | Setup time | Operational overhead |
|---|---|---|---|
| Feature store (Feast/Tecton) | Strong — enforced at infrastructure level | 2-3 weeks | High — separate service, versioning ceremony |
| Shared Python module (`core/feature_module.py`) | Enforced by convention — single import | 1 day | Low — just a function |
| Duplicate code in serve.py and train.py | None — diverges silently | 0 | Zero until it breaks |

**Decision:** Shared module.

**Reasoning:** A feature store solves point-in-time correctness at scale, when dozens of models and pipelines share features and the risk of silent divergence is high. At 35K users/month with a single model and two pipeline files, a shared module enforces the same guarantee with a fraction of the operational overhead. The key design decision is that `feature_cutoff_dt = expire_date - 14 days` is computed *internally* by `build_features()` — the caller passes `expire_date` and never touches the cutoff. That one invariant prevents the most common leakage pattern.

**Outcome:** Zero training-serving skew confirmed. AUC-PR on the held-out test set: 0.976. Verified by running the same `build_features()` call with identical inputs from both training and monitoring pipelines and comparing outputs.

**What I'd do differently at scale:** Add a feature lineage test that runs in CI — serialize the feature vector from training and serving for the same user+date and assert they match byte-for-byte. Right now the invariant is enforced by code structure; at scale it should be enforced by a test.

---

### Key Decision: PSI Drift Detection Without Evidently

**Options considered:**

| Option | Flexibility | Dependencies | Interview signal |
|---|---|---|---|
| Evidently AI library | High — rich reports | External dependency | "I used a library" |
| Custom PSI in `core/drift_module.py` | Full control | None beyond numpy/pandas | Shows you understand what PSI is |

**Decision:** Custom PSI.

**Reasoning:** The HLD originally specified Evidently. During implementation, the specific requirement — anchoring bin edges to the training distribution so the baseline never shifts — was easier to implement directly than to configure correctly in Evidently. Total code: 40 lines. The tradeoff is that custom code needs to be maintained; Evidently would handle edge cases in the distribution comparison. At MVP scale with 20 known features, the custom implementation is auditable and fast.

---

### Key Decision: Dataset Storage — MLflow Artifacts, Not DVC

**Options considered:**

| Option | Fit for the job | Setup cost | Operational overhead |
|---|---|---|---|
| DVC | Purpose-built: content-addressable hashing, dataset diffing, `dvc.yaml` dependency graph | ~Half a day — remote storage config, `.dvc` files | A second tool alongside MLflow, a second remote to manage |
| MLflow artifacts (what I built) | Repurposed: not designed for datasets — no diffing, no dataset entity, no rollback workflow | Zero — same MLflow server already running | None — rides on the durable volume MLflow already needs |

**Decision:** MLflow artifacts. `pipelines/build_dataset.py` logs `train.parquet` / `val.parquet` / `test.parquet` / `baseline_features.parquet` / `feature_config.json` to a dedicated `dataset_build` MLflow run; the run_id becomes `dataset_version_id`, tagged onto whatever model version the training pipeline fits from it.

**Reasoning:** The actual requirement was narrow — a stable handle that two independent pipelines (training, validation) could agree on, with automatic lineage back to whichever model consumed it. MLflow's artifact store already satisfies that, and it's infrastructure this project needs regardless (model registry, experiment tracking). Standing up DVC would mean a second tool and a second remote for a requirement MLflow already covers at this scale — nothing here uses DVC's actual differentiators (content-hash diffing, checking out an old dataset).

**Tradeoff being accepted:** This is a genuine misuse of MLflow's mental model. MLflow expects a "run" to produce metrics and a model; a `dataset_build` run produces neither — just files. Anyone browsing the MLflow UI sees a run with nothing trained, which is confusing without knowing the pattern.

**What I'd do differently at scale:** Move dataset storage to DVC (or MLflow's own `mlflow.data` dataset-logging API, a smaller step closer to purpose-built) so datasets are versioned by content hash instead of "whatever run happened to log it," and so the `dataset_build` MLflow run doesn't have to double as a dataset registry it was never designed to be.

---

### Key Decision: Drift Monitoring Chained Onto Serving, Not Independent

**Options considered:**

| Option | Redundant compute | Independent scheduling | New moving parts |
|---|---|---|---|
| Keep fully independent (original design) | Yes — `monitor.py` rebuilds the same day's feature matrix `serve.py` already built | Yes — drift can run on its own schedule | None |
| Persist a shared artifact (widen `predictions` with feature columns) | No | Yes | A DB migration + a read-or-rebuild fallback branch in `monitor.py` |
| Chain as a direct subflow call (what I built) | No | No — drift now runs only when serving runs | None — reuses the existing subflow-call pattern `monitor.py` already uses for auto-retraining |

**Decision:** Chain it. `run_serving_pipeline` calls `run_drift_monitoring(current_df=feature_df)` right after writing predictions, passing the feature matrix it already built in memory. `run_drift_monitoring` gained one optional parameter; when it's omitted, the standalone `monitor.py drift --date ...` CLI path is untouched and still rebuilds from source, which is what backfills and post-failure recovery use.

**Reasoning:** Both pipelines were computing the identical `FEATURE_COLS` matrix for the identical cohort on the identical day — a genuine double-compute, not a deliberate independence tradeoff. The shared-artifact option would have fixed that too, but at the cost of a schema change and a permanent dual-path (read vs. rebuild) branch to maintain. The subflow call needed no new infrastructure and mirrors a pattern already established in this codebase (`monitor.py`'s `maybe_trigger_retraining` calling `run_retrain_and_validate_pipeline` the same way) — same tool for the same kind of problem.

**Tradeoff being accepted:** Drift monitoring is no longer independently schedulable in the common case — it only runs when serving runs. If `serve.py` fails before reaching the drift step, that day's drift check doesn't happen automatically; there's no self-healing. Recovery is a manual `monitor.py drift --date <date>` call, the same operator action the 14-day (formerly 13-15) serving window now relies on instead of a rolling retry window.

**What I'd do differently at scale:** Once there's an actual on-call/alerting story — which would need a real metrics sink wired to a pager, not the removed `monitoring_metrics` table, which nothing ever read — reconsider whether losing independent drift scheduling is still the right trade. A system with reliable alerting can afford tighter coupling like this; one without it benefits more from redundant, self-healing pipelines even at extra compute cost.

---

### Key Decision: Prefect Over Airflow

**Options considered:**

| Option | Local dev cost | Fit for backfill / date-partitioned runs | Ecosystem & recognizability |
|---|---|---|---|
| Airflow | Scheduler + webserver + metadata DB required just to run one DAG | Native — `execution_date`, catchup, and backfill are built-in scheduler concepts | Industry-standard; what most data platform teams run in production |
| Prefect (what I built) | A plain Python function with `@flow`/`@task` decorators, runnable directly (`python serve.py --date ...`) | Hand-rolled — `scoring_date` param + `--date` CLI flag reimplement a slice of what Airflow gives for free | Smaller ecosystem; the API has changed substantially across major versions (1 → 2 → 3) |

**Decision:** Prefect. Every pipeline (`label.py`, `build_dataset.py`, `train.py`, `validate.py`, `serve.py`, `monitor.py`, `retrain.py`) is a plain Python module decorated with `@flow`/`@task`, runnable directly with no scheduler daemon or metadata DB needed to execute it.

**Reasoning:** For solo local development, not fighting Airflow's DAG-file/metadata-DB coupling just to test one pipeline change was worth more than Airflow's maturity. That said, this is a dev-ergonomics tradeoff, not a clean architectural win — it holds up less cleanly than "zero infra" makes it sound.

**Tradeoff being accepted:**
- **The infra saving is temporary, not structural.** It applies to local dev only. The moment these pipelines need to actually run on a schedule — the entire point of `serve.py`/`monitor.py` — Prefect needs a server, a worker, and deployment configs running somewhere, the same category of infra Airflow needs. That cost hasn't been paid yet because the pipelines are decorated but not actually scheduled (see "What didn't / what's still missing" above).
- **Airflow's core idiom is a better fit for what this project reimplements by hand.** `scoring_date` and the `--date` backfill flag are a hand-rolled version of `execution_date`/catchup/backfill — semantics Airflow's scheduler provides natively and has battle-tested for years.
- **Ecosystem maturity and recognizability.** Airflow is what most data platform teams actually run, and its DAG API has stayed far more stable across major versions than Prefect's (Prefect 1 → 2 → 3 were near-total rewrites). For a project meant to be read by other engineers, that ubiquity carries real legibility value that Prefect's nicer local syntax doesn't offset.

**What I'd do differently at scale:** Once real recurring scheduling with backfill/catchup semantics is actually needed — not just decorated-but-manual pipelines — re-evaluate Airflow specifically for its native `execution_date`/backfill model, which this project currently reimplements by hand. If Prefect stays, the honest next step is standing up its server/worker/deployment configs now, instead of continuing to treat "zero infra" as a permanent property of the choice rather than a deferred one.

---

### Key Decision: DuckDB Over Spark for Feature Aggregation

**Options considered:**

| Option | Fit for this data's scale | Setup cost | Operational overhead |
|---|---|---|---|
| Spark (PySpark) | Built for data that doesn't fit on one machine — distributed shuffle/partitioning | Cluster or local session, JVM dependency, serialization boundary between JVM and pandas | A cluster (or at minimum a JVM) to run and monitor, even for local dev |
| DuckDB, embedded in-process (what I built) | Purpose-built for single-machine, columnar, vectorized OLAP on GBs, not TBs | Zero — `pip install duckdb`, `duckdb.connect()` inside the same Python process | None — lives and dies with the Python process; nothing separate to deploy or monitor |

**Decision:** DuckDB, embedded directly inside `core/feature_module.py`. `build_features()` opens an in-process DuckDB connection, points views at either CSVs (`read_csv_auto`) or Postgres (via DuckDB's `postgres` extension/`ATTACH`), and runs three SQL aggregation queries — `_build_txn`, `_build_log`, `_build_member` — that join back into one feature matrix.

**Reasoning:** The KKBox source tables are tens of millions of rows across `transactions`/`user_logs`/`members` — a few GB, comfortably single-machine. Spark's distributed shuffle/partition machinery solves a problem this dataset doesn't have; running it here would mean paying JVM startup and serialization overhead (or standing up a whole cluster) for aggregations DuckDB's vectorized columnar engine does faster in-process, with no serialization boundary between the query engine and pandas, and no separate service to deploy alongside Prefect and Postgres. The aggregation logic itself — window functions, `GROUP BY`, `LEFT JOIN`s — is dbt's philosophy (let SQL do the transform work) without dbt's tooling: it's called as a plain Python function, not compiled and materialized against a warehouse, because there's no warehouse here to model against — just CSVs and one Postgres instance DuckDB can query directly.

**Tradeoff being accepted:** No distributed scale-out path — if the source tables grew past what fits comfortably on one machine's memory/disk, DuckDB's single-node model needs re-architecting, not a config change. No dbt-style model versioning, testing, or lineage graph either; the SQL lives inside Python functions (`_build_txn` etc.), not as tracked, independently-run `.sql` model files.

**What I'd do differently at scale:** If raw event volume grew by 10-100x (billions of rows, multi-TB), move the aggregation to Spark or a warehouse-native transform layer (Snowflake/BigQuery + dbt), and let DuckDB's role shrink back to what it's actually built for — fast local/single-node analytics. At this dataset's actual size, making that move now would add operational cost for a scale problem that doesn't exist yet.

---

## Try It

No live deployment yet — see the Demo section at the top for a video walkthrough. To run it yourself:

### Local Setup

### Prerequisites

- Docker Desktop
- Python 3.11+
- [uv install](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) (`pip install uv`) OPTIONAL


```bash
git clone https://github.com/div-ops123/kkbox-churn-prediction-system.git
cd kkbox-churn-prediction-system
cp .env.example .env   # set POSTGRES_PASSWORD=kkbox

# Start infrastructure
docker compose --env-file .env up -d

# Install dependencies
# Option 1
uv sync
# Option 2
pip install -r requirements.txt

# Seed database (one-time, ~5 min)
uv run python db/seed.py

# Run full pipeline cycle (commands below assume Option 1 / uv sync —
# drop `uv run` if you installed via Option 2 / pip into an activated venv)
uv run python src/pipelines/label.py --cohort-month 2017-03

# build_dataset.py -> train.py -> validate.py, chained by retrain.py
# validate.py only ever sets the 'challenger' alias on a winning candidate —
# it never touches 'production'. On a first run there's no production model
# to beat, so the candidate wins by default and still only gets 'challenger'.
uv run python src/pipelines/retrain.py --cohort-months 2017-03

# Manual step: challenger -> production promotion is an explicit human
# decision, not automated (see validate.py docstring). Do this once so
# serve.py has a production model to load; re-run after any retrain
# you want to actually deploy.
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -c "
   import mlflow
   client = mlflow.tracking.MlflowClient()
   client.set_registered_model_alias(
       name='LightGBMChurnClassifier', alias='production', version='1',
   )
   mv = client.get_model_version_by_alias('LightGBMChurnClassifier', 'production')
   print(f'production alias -> version {mv.version}, run_id={mv.run_id}')
   "

uv run python src/pipelines/serve.py --date 2017-03-01   # also chains drift monitoring (Track 1)
uv run python src/pipelines/monitor.py performance --cohort-month 2017-03

# Optional: backfill/re-run drift monitoring for a specific date standalone
uv run python src/pipelines/monitor.py drift --date 2017-03-01

# Start Score API
uv run uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/cohort?date=2017-03-01"
curl "http://localhost:8000/score/62xvWFfwSygahQNtlmr4E0xuntCBrRqjG3Njqv9wi2Y="
```

OpenAPI docs: `http://localhost:8000/docs`

| Dashboard | URL | Credentials |
|---|---|---|
| MLflow experiments | http://localhost:5000 | — |

### Project Layout

```
src/
├── config.py               — frozen dataclasses for all pipeline configs
├── core/                   — pure domain logic (no Prefect, no psycopg2)
│   ├── feature_module.py   — build_features(), FEATURE_COLS, CAT_COLS
│   ├── label_module.py     — build_cohort(), compute_labels()
│   ├── drift_module.py     — compute_feature_drift(), evaluate_cohort_performance()
│   ├── model_trainer.py    — train_lgbm(), split_cohort_3way(), compute_p99_secs()
│   ├── model_loader.py     — make_model_loader() factory
│   └── risk_tier.py        — make_tier_strategy() factory
├── pipelines/              — Prefect @flow + @task, CLI entry points
│   ├── label.py            — monthly labeling pipeline
│   ├── build_dataset.py    — labeled cohort → features → train/val/test split → MLflow
│   ├── train.py            — fit LightGBM on an already-built dataset, register unpromoted
│   ├── validate.py         — score candidate vs production on held-out test, alias challenger
│   ├── retrain.py          — thin orchestrator: build_dataset → train → validate
│   ├── serve.py            — daily scoring pipeline, chains drift monitoring
│   └── monitor.py          — drift (Track 1) + performance (Track 2) monitoring
├── experiments/            — one-off baseline scripts (not the production path)
│   ├── build_training_set.py — cohort selection + label join, writes train_features.parquet
│   └── train_baseline.py     — train LightGBM baseline
└── api/                    — FastAPI Score API
    ├── main.py
    ├── schemas.py
    ├── routers/
    └── repositories/
```
