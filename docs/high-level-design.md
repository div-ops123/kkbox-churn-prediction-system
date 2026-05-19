
# High-Level Design

## Failure 1 — Feature Store question: you are half right

Your 3 tables are your point-in-time source of truth during training because you control the feature cutoff per user explicitly in code. But at serving time you are computing features on the fly from raw tables every single day. That means your serving pipeline is re-running the same feature logic daily on raw data. This is acceptable for MVP. But you need to acknowledge that your feature transformation code must be identical between training and serving — one shared module, not two separate scripts. That is your point-in-time correctness mechanism for MVP. Not a feature store. A feature store solves this at scale. You solve it at MVP scale with disciplined code reuse.

## Failure 2 — Risk tier definition is floating

You output High/Medium/Low tiers but you haven't defined the thresholds. Who sets them? Are they fixed? Are they percentile-based? If your model's score distribution shifts after retraining, do your tiers shift with it? This needs to be a defined, versioned configuration — not a hardcoded if/else.

---

## **High-Level Design**

---

### **Components and Responsibilities**

┌─────────────────────────────────────────────────────────────────┐  
│                        DATA SOURCES                             │  
│         transactions table │ user_logs table │ members table    │  
└─────────────────┬───────────────────────────────────────────────┘  
                  │ raw data  
        ┌─────────▼──────────────────────────────────────────┐  
        │              FEATURE ENGINEERING MODULE             │  
        │  Shared library. Called by BOTH training pipeline   │  
        │  and serving pipeline. Single source of truth for   │  
        │  all feature transformations. Versioned in code.    │  
        │  Input: user\_id, feature\_cutoff\_date                │  
        │  Output: feature vector per user                    │  
        └─────────┬──────────────────────────┬───────────────┘  
                  │                          │  
      ┌───────────▼──────────┐   ┌───────────▼──────────────┐  
      │   TRAINING PIPELINE  │   │    SERVING PIPELINE       │  
      │                      │   │                           │  
      │  Trigger: monthly     │   │  Trigger: daily cron      │  
      │  or perf-triggered   │   │                           │  
      │                      │   │  1\. Query transactions    │  
      │  1\. Pull label table │   │     for users expiring    │  
      │  2\. Call feature     │   │     in 13-15 days         │  
      │     module per user  │   │  2\. Call feature module   │  
      │     at feature\_cutoff│   │     per user at           │  
      │  3\. Time-based split │   │     expiry \- 14 days      │  
      │  4\. Train candidate  │   │  3\. Load production model │  
      │     model            │   │     from model registry   │  
      │  5\. Compare against  │   │  4\. Score cohort          │  
      │     production model │   │  5\. Apply tier thresholds │  
      │     on validation set│   │     from config           │  
      │  6\. If better:       │   │  6\. Write to prediction   │  
      │     push to registry │   │     store with timestamp  │  
      │     tag as production│   │  7\. Write to output file  │  
      └───────────┬──────────┘   │     (Parquet/CSV)         │  
                  │              └───────────┬───────────────┘  
                  │                          │  
      ┌───────────▼──────────┐   ┌───────────▼───────────────┐  
      │    MODEL REGISTRY    │   │     PREDICTION STORE       │  
      │                      │   │                            │  
      │  Stores all model    │   │  Permanent log of every   │  
      │  artifacts with:     │   │  scoring run:             │  
      │  \- version           │   │  user\_id                  │  
      │  \- metrics           │   │  score                    │  
      │  \- feature list      │   │  risk\_tier                │  
      │  \- training date     │   │  scoring\_date             │  
      │  \- production tag    │   │  expiry\_date              │  
      │                      │   │  model\_version            │  
      └──────────────────────┘   └───────────┬───────────────┘  
                                             │  
                                 ┌───────────▼───────────────┐  
                                 │        SCORE API           │  
                                 │                            │  
                                 │  Lightweight REST API      │  
                                 │  Reads from prediction     │  
                                 │  store only.               │  
                                 │  Does NOT run the model.   │  
                                 │  Endpoints:                │  
                                 │  GET /score/{user\_id}      │  
                                 │  GET /cohort/{date}        │  
                                 └───────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐  
│                     LABELING PIPELINE                            │  
│                                                                  │  
│  Trigger: monthly, on the 1st (labels for month M-1 now ready)  │  
│                                                                  │  
│  1\. Query transactions for users whose anchor\_expiry\_date        │  
│     fell in month M-1                                           │  
│  2\. Apply churn rule: no valid non-cancel transaction            │  
│     within 30 days of expiry                                    │  
│  3\. Write to label store:                                        │  
│     user\_id, anchor\_expiry\_date, feature\_cutoff\_date, is\_churn  │  
│  4\. Emit event → triggers monitoring pipeline                    │  
└──────────────────────────┬───────────────────────────────────────┘  
                           │ triggers  
┌──────────────────────────▼───────────────────────────────────────┐  
│                    MONITORING PIPELINE                           │  
│                                                                  │  
│  Trigger: after labeling pipeline completes                      │  
│                                                                  │  
│  Track 1 — Data Drift (runs daily after serving)                │  
│    Compare today's feature distributions against                 │  
│    training baseline. Alert if PSI \> threshold.                  │  
│                                                                  │  
│  Track 2 — Model Performance (runs monthly after labeling)      │  
│    Join prediction store (month M-1 predictions)                 │  
│    with label store (month M-1 actuals)                         │  
│    Compute: AUC, Precision@K, Recall                            │  
│    Compare against baseline threshold                            │  
│                                                                  │  
│  Track 3 — Pipeline Health (runs after every pipeline)          │  
│    Cohort size anomaly detection                                 │  
│    Missing data checks                                           │  
│    Feature null rate checks                                      │  
│                                                                  │  
│  Output:                                                         │  
│    \- Metrics log (every run)                                     │  
│    \- Alert if drift or degradation threshold breached            │  
│    \- Retraining trigger if AUC \< defined threshold               │  
└──────────────────────────────────────────────────────────────────┘

---

### **Data Flow Summary**

RAW TABLES  
    │  
    ├──► LABELING PIPELINE ──► LABEL STORE  
    │         │  
    │         └──► MONITORING PIPELINE ◄── PREDICTION STORE  
    │                   │  
    │                   └──► RETRAINING TRIGGER  
    │                               │  
    ├──► FEATURE MODULE ◄───────────┘  
    │         │  
    │         ├──► TRAINING PIPELINE ──► MODEL REGISTRY  
    │         │  
    │         └──► SERVING PIPELINE ──► PREDICTION STORE ──► SCORE API  
    │                                        │  
    │                                        └──► OUTPUT FILE (Parquet)  
    │                                                │  
    │                                                └──► MARKETING / CRM

---

### **Key Interfaces Between Components**

| From | To | Interface | Format |
| ----- | ----- | ----- | ----- |
| Raw tables | Feature module | Direct query with user\_id \+ cutoff\_date | SQL / Pandas |
| Feature module | Training pipeline | Feature matrix | DataFrame |
| Feature module | Serving pipeline | Feature matrix | DataFrame |
| Label store | Training pipeline | Labeled cohort | Parquet |
| Label store | Monitoring pipeline | Actuals for evaluation | Parquet |
| Serving pipeline | Prediction store | Scored cohort with metadata | Parquet |
| Prediction store | Score API | Query by user\_id or date | REST / JSON |
| Prediction store | Monitoring pipeline | Historical predictions | Parquet |
| Model registry | Serving pipeline | Production model artifact | Pickle / ONNX |
| Monitoring pipeline | Training pipeline | Retraining trigger | Event / Flag |
| Serving pipeline | Marketing/CRM | Daily scored list | Parquet / CSV |

---

### **One Design Decision To Lock Before Low-Level Design**

Your monitoring pipeline has two distinct tracks running on different cadences — daily drift checks and monthly performance checks. These should share a monitoring module but have separate entry points and separate alert channels. Do not merge them into one script that tries to do both. Daily drift failures should not block monthly performance evaluation.


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

**For model training:** once you move from EDA to actual training, your feature engineering module queries PostgreSQL and outputs feature matrices as Parquet. Training then reads Parquet. This keeps RAM usage bounded.

---

### **Pipeline Orchestration — Prefect (not Airflow)**

**What:** Prefect Cloud free tier or Prefect self-hosted.

**Why not Airflow:** Airflow requires running a scheduler daemon, a metadata database, and a webserver as separate processes. For one engineer managing 4 pipelines, the operational overhead is disproportionate. When the scheduler crashes you get silent failures.

**Why Prefect:** Prefect is Python-native — your pipelines are just Python functions decorated with `@flow` and `@task`. It has a local runner that requires zero infrastructure, a free cloud dashboard for monitoring runs, built-in retry logic, and failure alerting. You can graduate to Airflow later when the team grows and complexity justifies it.

**For interview framing:** you know Airflow exists, you chose Prefect for MVP because operational simplicity matters for a solo engineer, and you can migrate to Airflow when team scale demands it. That's a stronger answer than "I used Airflow because it's standard."

---

### **Experiment Tracking \+ Model Registry — MLflow**

**What:** MLflow tracking server \+ MLflow model registry.

**Why:** MLflow gives you four things you need in one tool: experiment tracking (log metrics per training run), artifact storage (save model files), model versioning (every trained model gets a version), and production tagging (promote a model version to "production" so serving knows what to load). It runs as a lightweight server on your VM with artifact storage pointing to local disk or S3.

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

### **Monitoring Metrics Store — PostgreSQL table \+ Grafana**

**What:** A `monitoring_metrics` table in PostgreSQL. Grafana reads from it.

run\_date, pipeline\_type, metric\_name, metric\_value, alert\_triggered

Grafana connects to PostgreSQL directly via its built-in PostgreSQL data source plugin. No Prometheus needed. No separate time-series database. Simple queries, visible dashboards, free.

**Drift detection library:** Evidently AI. Open source, Python-native, produces feature drift reports and data quality reports that you can parse programmatically and log to your metrics table.

---

### **Score API — FastAPI \+ Uvicorn**

**What:** FastAPI application with two endpoints, reading from PostgreSQL prediction store.

**Why FastAPI:** async support, automatic OpenAPI docs, fast enough for your latency requirement (sub-200ms reads from an indexed PostgreSQL table), minimal boilerplate.

GET /score/{user\_id}        → latest score for a user  
GET /cohort?date=2017-03-01 → all scores for a scoring date

The API does not touch the model. It reads pre-computed scores only.

---

### **Containerization — Docker \+ Docker Compose**

**What:** Each pipeline (serving, training, labeling, monitoring) and the API runs in its own Docker container. Docker Compose orchestrates them locally and on the VM.

**Why:** Solves the "works on my machine" problem. Pins all library versions. Makes deployment to any cloud VM a single command. For CI/CD, your GitHub Actions pipeline builds and pushes the Docker image — the VM pulls and runs it.

**What you containerize:**

containers:  
  \- kkbox\_postgres          (database)  
  \- kkbox\_mlflow            (tracking server)  
  \- kkbox\_api               (FastAPI score API)  
  \- kkbox\_prefect\_worker    (runs all pipeline flows)  
  \- kkbox\_grafana           (dashboard)

---

### **Deployment — Railway or Render (free tier)**

**What:** Single VM on Railway or Render running your Docker Compose stack.

**Why Railway over GCP/AWS for MVP:** GCP free tier expires, requires billing setup, and has significant configuration overhead. Railway gives you a persistent VM, environment variable management, GitHub integration, and a public URL for your API — in under 30 minutes. When you need to scale, you migrate to GCP or AWS. For MVP demo and interview purposes, Railway is sufficient and honest.

**Compute estimate:** 2 vCPU, 4GB RAM VM is sufficient. Your heaviest job (training) runs monthly and can run overnight.

---

### **CI/CD — GitHub Actions**

**What:** On push to main, GitHub Actions runs: tests, builds Docker image, pushes to container registry, deploys to Railway.

**Why:** You already chose this. It's correct. Keep it.

---

### **Event Trigger (Labeling → Monitoring) — Prefect flow dependency**

**What:** In Prefect, your monthly labeling flow has a downstream dependency on the monitoring flow. When labeling completes successfully, Prefect automatically triggers the monitoring flow. No separate event bus needed.

**Why not Kafka or SQS:** you have two pipelines triggering each other once a month. An event queue is massive overengineering for this cadence.

---

### **Full Stack Summary**

| Component | Tool | Justification |
| ----- | ----- | ----- |
| Source database | PostgreSQL on Docker | Queryable, indexed, handles 20M+ rows, simple |
| EDA / experimentation | JupyterLab \+ DuckDB | Queries large files without RAM explosion |
| Pipeline orchestration | Prefect | Python-native, low ops overhead vs Airflow |
| Experiment tracking | MLflow | Open source, self-hosted, versioning \+ registry |
| Model | LightGBM | Best prior for tabular churn, fast, handles nulls |
| Prediction \+ label store | PostgreSQL tables | Same instance, low volume, simple queries |
| Monitoring metrics store | PostgreSQL \+ Evidently | Simple, Grafana-compatible, no extra infra |
| Drift \+ perf monitoring | Evidently AI | Python-native, open source drift reports |
| Dashboard | Grafana | Reads PostgreSQL directly, free |
| Score API | FastAPI \+ Uvicorn | Async, fast, minimal boilerplate |
| Containerization | Docker \+ Compose | Environment consistency, single-command deploy |
| Deployment | Railway free tier | Fast setup, persistent VM, GitHub integration |
| CI/CD | GitHub Actions | Standard, integrates with Railway and Docker |
| Programming language | Python 3.10+ | ML standard, all tools have Python SDKs |
| Event triggering | Prefect flow deps | No event bus needed at this cadence |

---

