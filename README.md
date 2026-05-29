# KKBox Churn Prediction — Production ML System

A production-grade churn prediction system built on the [KKBox Music Streaming dataset](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). Predicts which subscribers will churn 14 days before their subscription expires, so marketing and CRM teams can intervene.

---

## The Problem

KKBox loses revenue when subscribers cancel and the business finds out too late to act. Churn prediction needs to be:

- **Early enough** — scored 14 days before expiry, so retention campaigns can reach users while they're still active
- **Reliable** — a model that silently degrades after retraining destroys trust faster than no model at all
- **Operationalized** — predictions sitting in a notebook don't help CRM; they need to land in a queryable store with an API on top

The dataset: 970K users, 8.99% churn rate, 18M+ user log events, transactions through March 2017.

---

## The Solution

An end-to-end ML platform with six production components that mirror real industry architecture:

<!-- Architecture diagram: replace this comment block with your architecture image when ready -->
<!-- ![Architecture Diagram](docs/architecture.png) -->

```
DATA SOURCES (PostgreSQL)
transactions │ user_logs │ members
        │
        ▼
FEATURE ENGINEERING MODULE  ──────────────────────────────────┐
core/feature_module.py                                         │
Single shared library. Called by BOTH training AND serving.   │
Input: user_id + expiry_date                                   │
Output: 20-feature vector per user                             │
        │                                                      │
        ├──────────────────────────────────────┐              │
        ▼                                      ▼              │
TRAINING PIPELINE                    SERVING PIPELINE         │
pipelines/train.py                   pipelines/serve.py       │
Trigger: monthly or                  Trigger: daily           │
perf-triggered                       1. Users expiring        │
1. Pull labels table                    in 13-15 days         │
2. Build features                    2. Build features        │
3. Train LightGBM                    3. Load production model │
4. Compare vs production             4. Score cohort          │
5. Promote if better                 5. Apply risk tiers      │
        │                            6. Write predictions     │
        ▼                                      │              │
MODEL REGISTRY (MLflow)              PREDICTION STORE (PG)    │
All versions + metrics +                       │              │
production alias                              ▼              │
                                     SCORE API                │
                                     FastAPI                  │
                                     GET /score/{user_id}     │
                                     GET /cohort?date=...     │
                                                              │
LABELING PIPELINE ─────────────────────────────────────────── ┘
pipelines/label.py
Monthly: derive is_churn labels from transaction renewals
        │
        ▼
MONITORING PIPELINE
pipelines/monitor.py
Track 1 — Data drift (daily, PSI per feature vs baseline)
Track 2 — Model performance (monthly, AUC-PR vs threshold)
Track 3 — Pipeline health (cohort size, null rates)
        │
        ▼
GRAFANA DASHBOARD
monitoring_metrics table → real-time visibility
```

### Tech Stack

| Component | Choice | Why |
|---|---|---|
| Source DB | PostgreSQL 16 on Docker | Handles 20M+ rows; indexed date-range queries |
| Feature engineering | DuckDB (inside Python) | Queries CSVs + Postgres without loading into RAM |
| Pipeline orchestration | Prefect | Python-native `@flow`/`@task`; zero infra for local dev vs Airflow's scheduler daemon |
| Experiment tracking | MLflow | Self-hosted, open source; covers tracking + registry + promotion in one tool |
| Model | LightGBM | Native null handling, fast training, well-calibrated probabilities for risk tiers |
| Prediction store | PostgreSQL table | Same instance as source; low write volume, simple queries |
| Drift monitoring | Custom PSI (no Evidently) | Full control over bin-edge anchoring and alert logic |
| Dashboard | Grafana | Reads PostgreSQL directly; no Prometheus, no extra infra |
| Score API | FastAPI + Uvicorn | Async, auto OpenAPI docs, sub-200ms reads from indexed table |
| Containerization | Docker Compose | Single-command local and cloud deploy |

---

## The Results

| Metric | Value |
|---|---|
| AUC-PR (production cohort, March 2017) | **0.977** |
| AUC-ROC | 0.997 |
| Precision@100 | ~0.97 |
| Training cohort size | ~35K verified-anchor users |
| Serving cohort size (single date) | ~2,977 users |
| Features engineered | 20 (transaction, log, member) |
| Feature training-serving skew | **Zero** — single shared module |
| Pipelines with retry/alerting | All 4 (label, train, serve, monitor) |
| API endpoints | 3 (`/health`, `/score/{user_id}`, `/cohort`) |

**Monitoring output (2017-03-01 drift run):**
- 5 of 20 features triggered PSI > 0.2 alerts — all log-based features
- Root cause: serving cohort (expiry March 14-16) has feature_cutoff ≈ March 1, meaning near-zero log coverage from March-only user_logs data. Structural drift, not model degradation.
- System correctly flagged alerts and continued without false retraining trigger

---

## Technical Deep Dive

### Data Pipeline

**Challenge:** The KKBox dataset has a critical coverage gap. `user_logs.csv` only covers March 1-31, 2017. Users expiring in early March have `feature_cutoff = expiry - 14 days` falling before March 1 — meaning no log data is available for them. The label dataset (970K users in `train.csv`) contains ~930K users without a verified March expiry record in the transactions extract.

**Decision:** Train only on the ~35K users with a confirmed `anchor_expiry_date` in the transactions table. Users without a verified anchor (`has_anchor=0`) have feature_cutoff=2017-02-15 and no log features — training on them would produce a 97% null feature matrix and AUC-PR=0.39. The clean 35K cohort yields AUC-PR=0.977.

**Feature cutoff enforcement:** `feature_cutoff_dt = expire_date - 14 days` is computed *inside* `build_features()`, not passed by the caller. This prevents the serving pipeline from accidentally passing today's date and silently including post-cutoff data (leakage).

### Model Approach

LightGBM with `scale_pos_weight=10.1` to handle the 8.99% churn class imbalance. Primary metric: AUC-PR (not AUC-ROC) because the positive class is rare and AUC-ROC is optimistic under class imbalance.

20 features across three categories:
- **Transaction features** (8): recency, frequency, plan type, payment patterns, cancellation history
- **Log/behavioral features** (7): listening days, total seconds, song completion rate, days since last activity
- **Member features** (5): registration channel, tenure, city

The `p99_secs` winsorization threshold (43,805 sec/day) is computed once from the training distribution and saved to `models/feature_config.json`. The serving pipeline loads this frozen value — recomputing from the serving distribution would silently change the feature scale as listening patterns drift.

### Deployment Setup

```
docker compose up -d     # PostgreSQL :5433, MLflow :5000, Grafana :3000

# One-time seed
uv run python db/seed.py

# Monthly pipeline cycle
uv run python src/pipelines/label.py --cohort-month 2017-03
uv run python src/pipelines/train.py --cohort-months 2017-03
uv run python src/pipelines/serve.py --date 2017-03-01
uv run python src/pipelines/monitor.py drift --date 2017-03-01
uv run python src/pipelines/monitor.py performance --cohort-month 2017-03

# Score API
uv run uvicorn src.api.main:app --reload --port 8000
```

### Monitoring Strategy

Three independent tracks in `src/pipelines/monitor.py`, each with its own failure domain:

**Track 1 — Data Drift (daily):** PSI computed for all 20 features against a training baseline saved as `models/baseline_features.parquet`. Bin edges are anchored to the training distribution — never recomputed from serving data. Alert threshold: PSI > 0.2. Log-based features near month-start are expected to alert (structural coverage gap, not model degradation).

**Track 2 — Performance (monthly):** Join month-M predictions with month-M labels on `msno`. Compute AUC-PR, AUC-ROC, Precision@100, Recall@100. Alert threshold: AUC-PR < 0.45 triggers automatic retraining via sub-flow call.

**Track 3 — Pipeline health (inline):** Cohort size, null rates, and run status written to `monitoring_metrics` by each pipeline at execution time. Alert threshold: cohort_size < 100.

All three tracks write to the same `monitoring_metrics` PostgreSQL table. Grafana reads it directly.

---

## Learnings & Trade-offs

### Key Decision: Feature Consistency Mechanism

**Options considered:**

| Option | Guarantee | Setup time | Operational overhead |
|---|---|---|---|
| Feature store (Feast/Tecton) | Strong — enforced at infrastructure level | 2-3 weeks | High — separate service, versioning ceremony |
| Shared Python module (`core/feature_module.py`) | Enforced by convention — single import | 1 day | Low — just a function |
| Duplicate code in serve.py and train.py | None — diverges silently | 0 | Zero until it breaks |

**Decision:** Shared module.

**Reasoning:** A feature store solves point-in-time correctness at scale, when dozens of models and pipelines share features and the risk of silent divergence is high. At 35K users/month with a single model and two pipeline files, a shared module enforces the same guarantee with a fraction of the operational overhead. The key design decision is that `feature_cutoff_dt = expire_date - 14 days` is computed *internally* by `build_features()` — the caller passes `expire_date` and never touches the cutoff. That one invariant prevents the most common leakage pattern.

**Outcome:** Zero training-serving skew confirmed. AUC-PR on clean cohort: 0.977. Verified by running the same `build_features()` call with identical inputs from both training and monitoring pipelines and comparing outputs.

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

## Try It

**Live Demo:** `[placeholder — replace with Railway URL after deployment]`

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
docker compose up -d

# Install dependencies
# Option 1
uv sync
# Option 2
pip install -r requirements.txt

# Seed database (one-time, ~5 min)
uv run python db/seed.py

# Run full pipeline cycle
uv run python src/pipelines/label.py --cohort-month 2017-03
uv run python src/pipelines/train.py --cohort-months 2017-03
uv run python src/pipelines/serve.py --date 2017-03-01
uv run python src/pipelines/monitor.py drift --date 2017-03-01
uv run python src/pipelines/monitor.py performance --cohort-month 2017-03

# Start Score API
uv run uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/cohort?date=2017-03-01"
curl "http://localhost:8000/score/Zy4W5mkOlk8+qCMQD4K+MFH7LXuRi8tGeiaFBfCTu78="
```

OpenAPI docs: `http://localhost:8000/docs`

| Dashboard | URL | Credentials |
|---|---|---|
| Grafana monitoring | http://localhost:3000 | admin / admin |
| MLflow experiments | http://localhost:5000 | — |

### Project Layout

```
src/
├── config.py               — frozen dataclasses for all pipeline configs
├── core/                   — pure domain logic (no Prefect, no psycopg2)
│   ├── feature_module.py   — build_features(), FEATURE_COLS, CAT_COLS
│   ├── label_module.py     — build_cohort(), compute_labels()
│   ├── drift_module.py     — compute_psi(), evaluate_cohort_performance()
│   ├── model_loader.py     — make_model_loader() factory
│   └── risk_tier.py        — make_tier_strategy() factory
├── pipelines/              — Prefect @flow + @task, CLI entry points
│   ├── label.py            — monthly labeling pipeline
│   ├── train.py            — monthly training + model promotion
│   ├── serve.py            — daily scoring pipeline
│   └── monitor.py          — drift + performance + health monitoring
├── training/               — one-off training scripts (experimentation)
│   ├── features.py         — build train_features.parquet
│   └── train.py            — train LightGBM baseline
└── api/                    — FastAPI Score API
    ├── main.py
    ├── schemas.py
    ├── routers/
    └── repositories/
```
