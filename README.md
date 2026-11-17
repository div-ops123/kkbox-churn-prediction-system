# PyTorch ANN Experiment — Customer Churn Prediction

### 1. Experiment Context

This experiment extends the existing KKBOX customer churn prediction project.

The original project already includes:
- Data preparation and feature engineering.
- A traditional ML implementation using XGBoost.
- An established evaluation methodology and business-oriented metrics.

The objective of this experiment is to implement an Artificial Neural Network (ANN) using PyTorch and compare its performance against the existing XGBoost model.

**Scope:** Model development and experimentation, not another end-to-end ML project.

The focus is on developing practical competence in the PyTorch model-development lifecycle.

### 2. Problem Definition

**What are we predicting?**

Predict which subscribers will churn 14 days before their subscription expires.

This is a binary classification problem.

**What should the model output?**

The model should output a churn probability score for each subscriber, rather than a hard classification label.

This allows the business to rank subscribers according to their estimated churn risk.

### 3. Existing Benchmark

The existing XGBoost model serves as the traditional ML benchmark.

Its configuration includes:

- Objective: `binary:logistic`
- Evaluation metric: `aucpr`

The model produces raw probabilities using:

```python
proba = model.predict_proba(X)
scores = proba[:, 1] if proba.ndim == 2 else proba
```

The ANN should produce equivalent probability scores to enable a meaningful comparison.

### 4. Business Context and Motivation

The existing experiment evaluated 1,055 subscribers whose subscriptions were expiring on 2017-03-15.

Of these subscribers, 915 were identified as genuine churn cases.

The existing XGBoost model achieved:

- AUC-PR: 0.859
- Precision@100: 0.86

The current ML solution already provides a strong result.

A neural network is not being introduced because there is an established business need for one.

In this particular problem:
- The existing ML model performs well.
- XGBoost is relatively interpretable compared with a neural network.
- ANN training introduces additional computational and operational complexity.

The ANN experiment is therefore primarily educational.

The objective is to understand when neural networks are useful, how to implement them with PyTorch, and how their performance compares with traditional ML.

### 5. Baselines and Model Comparison

The experiment will use the following comparison:

| Approach | Purpose |
|---|---|
| Heuristic / rule-based | Simple business-rule baseline, where feasible |
| XGBoost | Existing traditional ML benchmark |
| PyTorch ANN | Neural network experiment |

The existing XGBoost model is the primary benchmark.

A heuristic baseline is not yet established. The KKBOX transaction and listening-log data have limitations, including missing listening-log coverage for many subscribers.

A business-informed heuristic could be explored separately if the available data supports a defensible rule.

### 6. Evaluation Strategy

**Primary metric: AUC-PR (Average Precision)**

AUC-PR is preferred over AUC-ROC because churn prediction involves class imbalance, and precision-recall performance is more informative about identifying positive cases.

**Secondary business metric: Precision@100**

Precision@100 measures the proportion of actual churners among the 100 highest-risk subscribers.

This reflects a practical business scenario where retention resources are limited and the business can only target a fixed number of subscribers.

The ANN will be evaluated using the same evaluation population and metric definitions as XGBoost.

### 7. Optimization Objective

The ANN will be trained as a binary classification model.

The initial training objective will be binary cross-entropy

The training objective and evaluation metric serve different purposes:

- Binary cross-entropy: Training loss optimized by gradient descent.
- AUC-PR: Primary evaluation metric.
- Precision@100: Business-oriented evaluation metric.

The initial ANN will use PyTorch's `BCEWithLogitsLoss`, with raw logits supplied directly to the loss function.

### 8. Experimental Fairness

To make the comparison meaningful, the experiment will preserve:

- Existing prediction target and label definition.
- Existing data preparation and feature engineering, subject to ANN input compatibility.
- Existing train/validation/test split.
- Existing evaluation population.
- Existing metric definitions.

Any ANN-specific preprocessing, such as feature scaling or categorical encoding, will be documented.

Preprocessing parameters must be fitted using training data only to avoid data leakage.

### 9. Learning Objectives

The primary learning objective is to become independently comfortable implementing the PyTorch model-development lifecycle.

Topics to practise:

1. Tensor creation, shapes, data types and operations.
2. Dataset and DataLoader.
3. Building neural network architectures using `nn.Module`.
4. Forward propagation.
5. Loss functions.
6. Optimizers and gradient descent.
7. Training and validation loops.
8. Model evaluation.
9. Device management (CPU/GPU).
10. Model checkpointing and training resumption.
11. Debugging tensor shapes, gradients, training instability and model behaviour.

The experiment will also develop an understanding of when different PyTorch abstractions are appropriate.

### 10. Experiment Philosophy

The goal is not to use a neural network simply because it is more sophisticated.

The goal is to understand how to develop neural networks correctly, evaluate their performance, identify their limitations and make informed engineering decisions about when they are appropriate.

The existing XGBoost model remains the benchmark against which the ANN will be evaluated.

---

## Try It

To run it yourself:

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
       name='ChurnClassifier', alias='production', version='1',
   )
   mv = client.get_model_version_by_alias('ChurnClassifier', 'production')
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
