
# High-Level Design

## **High-Level Design**

---

### **Components and Responsibilities**

#### Data Sources

`transactions` table · `user_logs` table · `members` table (raw PostgreSQL tables).

#### Feature Engineering Module — `src/core/feature_module.py`

Shared, stateless library — a pure **Transform**, not a pipeline. Called separately, each with its own cohort, by three independent pipelines: the dataset-build pipeline, the serving pipeline, and the monitoring pipeline's drift track. There is no single shared dataset that all three read from — what's shared is the code, not the output. Single source of truth for all feature transformations, versioned in code.

Input: `msno`, `feature_cutoff_date` (always `expiry_date - 14 days`, enforced internally). Output: feature vector per user.

#### Dataset-Build Pipeline (ETL) — `src/pipelines/build_dataset.py`

Trigger: called by the retrain orchestrator, or run standalone.

1. Pull the labeled cohort from the label store.
2. Compute `p99_secs` for this cohort (the `total_secs` winsorization threshold).
3. Call the feature module.
4. Rolling 3-way split: the most recent labeled month becomes the **test** set (held out entirely — never used for fitting or early stopping), the second-most-recent becomes **val** (early stopping only), everything older becomes **train**.
5. Load train/val/test parquet + a drift baseline (train+val features only, never test) + `feature_config.json` to a dedicated MLflow run.

Output: `dataset_version_id` (the MLflow run_id) — the handle the training and validation pipelines consume instead of rebuilding features themselves.

#### Training Pipeline (ML — fit only) — `src/pipelines/train.py`

Trigger: called by the retrain orchestrator with a `dataset_version_id`, or run standalone.

1. Fetch train + val from that dataset (no raw data, no feature computation — that's the dataset-build pipeline's job).
2. Fit LightGBM, early-stopping on val.
3. Register a new model version, tagged with the `dataset_version_id` it was trained on.
4. Never sets an alias. The model exists, versioned and inspectable, but unpromoted.

#### Validation Pipeline (ML — evaluate & decide) — `src/pipelines/validate.py`

Trigger: called by the retrain orchestrator with a `dataset_version_id` + `candidate_version`, or run standalone.

1. Fetch the held-out test set from that dataset — the fold neither fitting nor early stopping ever touched.
2. Load the candidate model and the current production model.
3. Score both on the same test set.
4. If the candidate wins (or no production model exists yet): alias it `challenger`. Never touches `production` — promoting `challenger` → `production` is a deployment-strategy decision (canary, shadow mode, etc.), explicitly out of scope for this project.

#### Retrain Orchestrator (thin caller, not a pipeline) — `src/pipelines/retrain.py`

Wires Dataset-Build → Training → Validation together for the common case: monthly cron, or a monitoring-triggered retrain. Each of the three stays independently invocable — e.g. re-running validation alone against an older candidate never touches this file.

#### Model Registry — MLflow

Stores every model version with:

- version
- metrics
- feature list
- `dataset_version_id` tag — lineage back to the exact train/val/test snapshot this version was trained on
- aliases: `production` (live, read by serving and monitoring) and `challenger` (a candidate that beat production in validation, not yet live)

#### Serving Pipeline (ETL + inference) — `src/pipelines/serve.py`

Trigger: daily cron.

1. Query transactions for users expiring in 13–15 days.
2. Call the feature module per user at `expiry_date - 14 days`.
3. Load the production model and its `dataset_version_id` tag from the registry.
4. Fetch `p99_secs` from that dataset's MLflow artifacts — not a local file (see the durability note under Design Decisions Locked).
5. Score the cohort.
6. Apply tier thresholds from config.
7. Write to the prediction store with a timestamp.
8. Write to an output file (Parquet/CSV).

#### Prediction Store

Permanent log of every scoring run: `msno`, `score`, `risk_tier`, `scoring_date`, `expiry_date`, `model_version`.

#### Score API

Lightweight REST API. Reads from the prediction store only. Does **not** run the model.

`GET /score/{user_id}` · `GET /cohort/{date}`

#### Labeling Pipeline (ETL) — `src/pipelines/label.py`

Trigger: monthly, on the 1st (labels for month M-1 now ready).

1. Query transactions for users whose `anchor_expiry_date` fell in month M-1.
2. Apply the churn rule: no valid non-cancel transaction within 30 days of expiry.
3. Write to the label store: `msno`, `anchor_expiry_date`, `feature_cutoff_date`, `is_churn`.
4. Emit event → triggers the monitoring pipeline.

#### Monitoring Pipeline — `src/pipelines/monitor.py`

Two tracks, deliberately kept structurally different — see Design Decisions Locked.

**Track 1 — Data Drift** (ET-shaped: extracts a cohort and transforms it via the feature module, but loads only a metrics row, not a dataset). Runs daily. Resolves the `dataset_version_id` tagged on whichever model currently holds the `production` alias, fetches that dataset's baseline features + `p99_secs` from MLflow (never a local file), calls the feature module on today's serving cohort, computes PSI per feature. Alerts if PSI > threshold.

**Track 2 — Model Performance** (no transform stage at all — it joins already-scored data, it never calls the feature module). Runs monthly. Joins the prediction store (month M-1 predictions) with the label store (month M-1 actuals). Computes AUC-PR, AUC-ROC, Precision@K, Recall@K. Compares against a baseline threshold. If AUC-PR falls below threshold, triggers the **retrain orchestrator** — not the training pipeline directly (see Design Decisions Locked).

**Track 3 — Pipeline Health** (handled inline by every pipeline, not a separate flow). Cohort size anomaly detection, missing data checks, feature null rate checks. Each of `serve.py`, `label.py`, `build_dataset.py`, `train.py`, and `validate.py` computes and logs its own health metrics inline — console output only, no persistence.

Output: metrics log (every run), alert if drift or degradation threshold breached, retraining trigger if AUC-PR < defined threshold.

---

### **Data Flow Summary**

**Retrain path** (monthly cron, or monitoring-triggered):

```
RAW TABLES
  → LABELING PIPELINE → LABEL STORE
      → MONITORING PIPELINE (Track 2) ← PREDICTION STORE
          → RETRAIN ORCHESTRATOR
              → DATASET-BUILD PIPELINE ← LABEL STORE
                  → MLflow artifacts (dataset_version_id: train / val / test / baseline / feature_config)
                      → TRAINING PIPELINE → MODEL REGISTRY (new version, unpromoted)
                          → VALIDATION PIPELINE ← MLflow artifacts (test set) + MODEL REGISTRY (production model)
                              → MODEL REGISTRY (challenger alias, only if the candidate wins)
```

**Serving path** (daily cron, fully independent of the retrain path above):

```
RAW TABLES
  → FEATURE MODULE (shared code, called separately here — not a shared dataset)
      → SERVING PIPELINE ← MODEL REGISTRY (production alias + its dataset_version_id)
          → PREDICTION STORE → SCORE API
              → OUTPUT FILE (Parquet) → MARKETING / CRM
```

---

### **Key Interfaces Between Components**

| From | To | Interface | Format |
| ----- | ----- | ----- | ----- |
| Raw tables | Feature module | Direct query with `msno` + cutoff date | SQL / Pandas |
| Feature module | Dataset-build pipeline | Feature matrix | DataFrame |
| Feature module | Serving pipeline | Feature matrix | DataFrame |
| Feature module | Monitoring pipeline (drift track) | Feature matrix | DataFrame |
| Label store | Dataset-build pipeline | Labeled cohort | SQL |
| Label store | Monitoring pipeline | Actuals for evaluation | SQL |
| Dataset-build pipeline | MLflow artifacts | train/val/test + baseline features + feature_config, keyed by `dataset_version_id` | Parquet / JSON |
| MLflow artifacts (`dataset_version_id`) | Training pipeline | train + val datasets | Parquet |
| MLflow artifacts (`dataset_version_id`) | Validation pipeline | test dataset | Parquet |
| Training pipeline | Model registry | New model version, tagged `dataset_version_id` (unpromoted) | MLflow |
| Model registry (candidate + production) | Validation pipeline | Models to compare on the test set | MLflow |
| Validation pipeline | Model registry | `challenger` alias (only if the candidate wins) | MLflow |
| Model registry (`production` alias) | Serving pipeline | Production model + its `dataset_version_id` | MLflow |
| Model registry (`production` alias) + MLflow artifacts | Monitoring pipeline (drift track) | Baseline features + `p99_secs` | Parquet / JSON |
| Serving pipeline | Prediction store | Scored cohort with metadata | Parquet |
| Prediction store | Score API | Query by `msno` or date | REST / JSON |
| Prediction store | Monitoring pipeline | Historical predictions | SQL |
| Monitoring pipeline | Retrain orchestrator | Retraining trigger | Direct Prefect subflow call |
| Serving pipeline | Marketing/CRM | Daily scored list | Parquet / CSV |

---

### **Design Decisions Locked Before Low-Level Design**

**1. Monitoring's two tracks stay structurally separate.** Daily drift checks and monthly performance checks run on different cadences and have different shapes (Track 1 calls the feature module, Track 2 never does). They share a monitoring module but have separate entry points and separate alert channels. Do not merge them into one script that tries to do both — daily drift failures must not block monthly performance evaluation.

**2. Training never decides promotion.** The dataset-build pipeline holds out the most recent labeled month as a test set the training pipeline never sees — not for fitting, not for early stopping. Training always registers a new model version and stops there; it never compares against production and never sets an alias. Only the validation pipeline, scoring on that untouched test fold, decides whether a candidate becomes `challenger`. This closes a real gap in the original single-pipeline design: comparing production against a candidate on the same fold the candidate's early stopping had already used gave the candidate a structural advantage in the comparison. Promoting `challenger` to `production` is a further, deliberately out-of-scope decision — it belongs to a deployment strategy (canary, shadow mode, etc.) not designed here.

**3. Dataset artifacts live in MLflow, not local disk.** `feature_config.json` and the drift baseline used to be written to `models/` on local disk, read back by whichever pipeline needed them next. `docker-compose.yml` mounts durable volumes for Postgres and MLflow, but nothing for `models/` — a redeploy between training and the next drift check would silently wipe them. Worse, the file was overwritten on every training run regardless of promotion, so an unpromoted candidate could silently change what serving used. Both are fixed by tagging every model version with the `dataset_version_id` it was trained on and having callers fetch `p99_secs`/baseline from that dataset's MLflow artifacts — durability comes from MLflow's already-durable artifact store, and staleness is fixed because callers always resolve the artifact through *the currently-aliased model*, never "whatever was last written."

---


# Tech Stack With Justifications

---

### **Source Database — PostgreSQL on Docker (local or cloud VM)**

**What:** Load all 3 KKBox CSV tables into a PostgreSQL instance running in Docker.

**Why:** Your pipelines need a queryable, indexed data source — not raw CSVs. PostgreSQL handles tens of millions of rows without issue. It supports the date-range queries your labeling and serving pipelines need. It runs locally for experimentation and on a cheap VM in production. It is the simplest thing that works at this scale.

---

### **Experimentation Environment — Jupyter Notebook \+ DuckDB**

**What:** Jupyter notebook for notebooks. DuckDB as your in-notebook query engine.

**Why DuckDB specifically:** DuckDB runs inside your Python process, queries Parquet files and CSVs directly without loading them fully into RAM, and executes analytical SQL at speeds that make 18M row aggregations take seconds not minutes. You do not need a database server running for EDA. You run SQL directly on your files. This is the single best tool for exploratory data work at this scale on a laptop.

**For model training:** once you move from EDA to actual training, the dataset-build pipeline queries PostgreSQL and outputs feature matrices as Parquet artifacts in MLflow. Training and validation then read Parquet. This keeps RAM usage bounded.

---

### **Pipeline Orchestration — Prefect (not Airflow)**

**What:** Prefect Cloud free tier or Prefect self-hosted.

**Why not Airflow:** Airflow requires running a scheduler daemon, a metadata database, and a webserver as separate processes. For one engineer managing several pipelines, the operational overhead is disproportionate. When the scheduler crashes you get silent failures.

**Why Prefect:** Prefect is Python-native — your pipelines are just Python functions decorated with `@flow` and `@task`. It has a local runner that requires zero infrastructure, a free cloud dashboard for monitoring runs, built-in retry logic, and failure alerting. Calling one `@flow` from inside another creates a traceable child flow run, which is exactly the mechanism the retrain orchestrator and the monitoring-triggered retrain use — no separate event bus. You can graduate to Airflow later when the team grows and complexity justifies it.

**For interview framing:** you know Airflow exists, you chose Prefect for MVP because operational simplicity matters for a solo engineer, and you can migrate to Airflow when team scale demands it. That's a stronger answer than "I used Airflow because it's standard."

---

### **Experiment Tracking \+ Model Registry — MLflow**

**What:** MLflow tracking server \+ MLflow model registry.

**Why:** MLflow gives you the things you need in one tool: experiment tracking (log metrics per training run), artifact storage (save model files — and, now, dataset snapshots: each dataset-build run logs its train/val/test/baseline parquet and `feature_config.json` as artifacts, so training and validation always agree on exactly which snapshot they're using), model versioning (every trained model gets a version, tagged with the `dataset_version_id` it was trained on), and alias-based promotion (`production` for what's live, `challenger` for a validated candidate not yet live — promoting challenger to production is a separate, not-yet-built deployment step). It runs as a lightweight server on your VM with artifact storage pointing to local disk or S3.

**Specific justification over alternatives:** Weights & Biases requires cloud dependency. SageMaker is AWS lock-in. MLflow is open source, self-hostable, and integrates with scikit-learn and LightGBM in three lines of code.

---

### **Model — LightGBM**

**What:** LightGBM gradient boosting classifier.

**Why:** Handles tabular data with mixed feature types (numeric behavioral features, categorical demographic features) without extensive preprocessing. Trains fast on datasets with hundreds of thousands of rows. Natively handles missing values. Produces well-calibrated probability scores, which your risk tier thresholds depend on. Outperforms random forest and logistic regression on this class of problem consistently.

---

### **Prediction Store \+ Label Store — PostgreSQL (same instance)**

**What:** Two additional tables in your existing PostgreSQL instance.

predictions table:  
  user\_id, score, risk\_tier, scoring\_date,   
  expiry\_date, model\_version

labels table:  
  user\_id, anchor\_expiry\_date,   
  feature\_cutoff\_date, is\_churn, labeled\_date

**Why not a separate database:** you have two stores with low write volume (one daily batch, one monthly batch) and simple query patterns (join by user\_id and date). A separate database adds operational complexity with zero benefit at this scale.

---

### **Monitoring output — logged, not persisted**

**What:** Every pipeline computes its own metrics (drift PSI, performance AUC-PR/precision/recall, pipeline health cohort size/null rates) and logs them via the Prefect run logger. There is no metrics table and no dashboard — a `monitoring_metrics` PostgreSQL table and a Grafana dashboard on top of it were removed after review found nothing (no automated trigger, no dashboard, no manual query) ever read either back. The retraining trigger in `monitor.py` acts on the in-memory result of the same flow run, not a re-query of anything persisted.

**Drift detection library:** Evidently AI. Open source, Python-native, produces feature drift reports and data quality reports that you can parse programmatically and log.

---

### **Score API — FastAPI \+ Uvicorn**

**What:** FastAPI application with two endpoints, reading from PostgreSQL prediction store.

**Why FastAPI:** async support, automatic OpenAPI docs, fast enough for your latency requirement (sub-200ms reads from an indexed PostgreSQL table), minimal boilerplate.

GET /score/{user\_id}        → latest score for a user  
GET /cohort?date=2017-03-01 → all scores for a scoring date

The API does not touch the model. It reads pre-computed scores only.

---

### **Containerization — Docker \+ Docker Compose**

**What:** Each pipeline (serving, dataset-build, training, validation, retrain, labeling, monitoring) and the API runs in its own Docker container. Docker Compose orchestrates them locally and on the VM.

**Why:** Solves the "works on my machine" problem. Pins all library versions. Makes deployment to any cloud VM a single command. For CI/CD, your GitHub Actions pipeline builds and pushes the Docker image — the VM pulls and runs it.

**What you containerize:**

containers:  
  \- kkbox\_postgres          (database)  
  \- kkbox\_mlflow            (tracking server + artifact store — now also holds dataset snapshots)  
  \- kkbox\_api               (FastAPI score API)  
  \- kkbox\_prefect\_worker    (runs all pipeline flows)

---

### **Deployment — Railway or Render (free tier)**

**What:** Single VM on Railway or Render running your Docker Compose stack.

**Why Railway over GCP/AWS for MVP:** GCP free tier expires, requires billing setup, and has significant configuration overhead. Railway gives you a persistent VM, environment variable management, GitHub integration, and a public URL for your API — in under 30 minutes. When you need to scale, you migrate to GCP or AWS. For MVP demo and interview purposes, Railway is sufficient and honest.

**Compute estimate:** 2 vCPU, 4GB RAM VM is sufficient. Your heaviest job (dataset-build + training, back to back) runs monthly and can run overnight.

---

### **CI/CD — GitHub Actions**

**What:** On push to main, GitHub Actions runs: tests, builds Docker image, pushes to container registry, deploys to Railway.

**Why:** You already chose this. It's correct. Keep it.

---

### **Event Triggers — Prefect direct subflow calls**

**What:** Two trigger relationships, same mechanism: the monthly labeling flow has a downstream dependency on the monitoring flow, and monitoring's performance track (Track 2) calls the retrain orchestrator directly when AUC-PR falls below threshold — which itself calls the dataset-build, training, and validation flows as subflows in sequence. In Prefect 3.x, calling one `@flow` from inside another creates a traceable child flow run. No separate event bus.

**Why not Kafka or SQS:** these pipelines trigger each other at most once a day, and the retrain chain only when an alert fires. An event queue is massive overengineering for this cadence.

---

### **Full Stack Summary**

| Component | Tool | Justification |
| ----- | ----- | ----- |
| Source database | PostgreSQL on Docker | Queryable, indexed, handles 20M+ rows, simple |
| EDA / experimentation | JupyterLab \+ DuckDB | Queries large files without RAM explosion |
| Pipeline orchestration | Prefect | Python-native, low ops overhead vs Airflow |
| Experiment tracking | MLflow | Open source, self-hosted, versioning \+ registry \+ dataset lineage |
| Model | LightGBM | Best prior for tabular churn, fast, handles nulls |
| Prediction \+ label store | PostgreSQL tables | Same instance, low volume, simple queries |
| Monitoring output | Logged via Prefect run output | No persisted store — nothing ever read one back |
| Drift \+ perf monitoring | Evidently AI | Python-native, open source drift reports |
| Score API | FastAPI \+ Uvicorn | Async, fast, minimal boilerplate |
| Containerization | Docker \+ Compose | Environment consistency, single-command deploy |
| Deployment | Railway free tier | Fast setup, persistent VM, GitHub integration |
| CI/CD | GitHub Actions | Standard, integrates with Railway and Docker |
| Programming language | Python 3.10+ | ML standard, all tools have Python SDKs |
| Event triggering | Prefect direct subflow calls | No event bus needed at this cadence |

---
