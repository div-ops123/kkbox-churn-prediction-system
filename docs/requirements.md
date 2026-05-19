# Requirements

# **Requirement Gathering**

### **Business & Product Questions**

* What retention actions are available? (discount, push notification, email, in-app message, human outreach?)  
* What is the cost of each intervention? (a 50% discount coupon is expensive — do we cap it to top-K users only?)  
* What is the average monthly subscription revenue per user?  
* What is the average customer lifetime value?  
* What is our current observed churn rate? (baseline — if only 8% churn, your model needs to beat that trivially)  
* How far in advance do we need predictions before the subscription expires?  
* Is there a cap on how many users marketing can actually contact per month?

### **Data & Label Questions**

* Is the churn label already computed or do we derive it? (you actually asked this implicitly — good instinct, but you need to be explicit)  
* How reliable is historical transaction data? Any known gaps or quality issues?  
* How far back does behavioral data go? Can we use 6-month history or only 30-day?  
* Are there known seasonal patterns? (holiday spikes, promotional periods that distort behavior?)

### **System & Consumer Questions**

* Who consumes the predictions? (Growth team? CRM system? Marketing automation tool like Braze or Iterable?)  
* What format do they need? (CSV drop? REST API? Direct DB write? Webhook?)  
* What's their SLA? (Do they need scores by 9AM Monday? End of day? Real-time?)  
* Do they need a churn score (probability) or a binary label or a risk tier (High/Medium/Low)?  
* Do they need model explanations? (why is this user flagged?) — important for trust and campaign personalization

### **Operational Questions**

* Who owns this system after I build it? (me alone, or a team?)  
* What's the retraining cadence expectation?  
* What monitoring or alerting infrastructure already exists?  
* What does failure look like? If scores aren't ready Monday morning, what breaks?

---

## **Now: Actual Requirements**

### **Functional Requirements — What The System Must Do**

**F1 — Label Generation** The system must derive churn labels from raw transaction data. A user is churned if they have no valid subscription transaction within 30 days of their membership expiry date. This cannot be pulled from a pre-labeled column in production — you generate it from `transactions.csv` logic. You were right to question this. In production, this is a scheduled label computation job, not a manual step.

**F2 — Feature Engineering Pipeline** The system must compute features from three sources on a per-user basis:

* Behavioral features from `user_logs`.
* Subscription features from `transactions`.
* Demographic features from `members`.

Features must be computed consistently between training and inference. This is a hard requirement — feature skew between train and serve is one of the most common production failures.

**F3 — Model Training Pipeline** The system must train a binary classifier on the engineered features. Training must be reproducible: same data \+ same config \= same model. Experiment tracking must record: features used, hyperparameters, evaluation metrics, and the model artifact.

**F4 — Offline Evaluation** Before any model goes near production, it must be evaluated on a held-out time-based validation set. Metrics must include: AUC-ROC, precision at top-K (because marketing can only contact N users), recall, and a cost-sensitive business metric (expected revenue saved vs. campaign cost).

**F5 — Batch Inference Pipeline** Monthly (or weekly), the system must score all users whose subscriptions are expiring in the next N days. Output: a scored user list with churn probability, risk tier, and timestamp. This is batch, not real-time. The scores are consumed downstream by a growth or CRM team.

**F6 — Serving Layer / Output Delivery** Scores must be delivered to consumers in a usable format. For MVP: a clean output file (Parquet or CSV) written to a defined location, plus a lightweight REST API that allows a downstream system to query scores by user ID. The API does not run the model — it reads from a pre-computed score store.

**F7 — Monitoring** The system must detect and alert on: data drift in input features, model performance degradation over time, and pipeline failures. Monitoring runs after each batch scoring cycle.

**F8 — Retraining Trigger** The system must support scheduled retraining (monthly) and performance-triggered retraining (if AUC drops below a defined threshold on the monitoring check).

---

### **Non-Functional Requirements — How The System Must Perform**

**NF1 — Latency** This is a batch system. Scoring latency is not sub-second. The full batch scoring pipeline (feature computation \+ inference for \~300K users) must complete within 2 hours. API query latency for pre-computed scores must be under 200ms.

**NF2 — Throughput** Score \~200-300K users per monthly batch run. Feature computation is the bottleneck, not inference.

**NF3 — Availability** The score store and API must have 99.5% uptime. The batch pipeline can tolerate a retry window — if it fails at 2AM, it must alert and allow a re-run before business hours.

**NF4 — Consistency** Features computed at training time and features computed at inference time must use identical logic. This is enforced by sharing the same feature transformation code between both pipelines. No separate scripts. One codebase.

**NF5 — Reproducibility** Any training run must be reproducible from a given dataset snapshot \+ config file. Random seeds are fixed. Data versioning is tracked.

**NF6 — Maintainability** A single ML engineer must be able to understand, modify, and debug the system. This means: no unnecessary tooling, clear module boundaries, documented configs, and no "magic" steps that only work on one person's machine.

**NF7 — Cost** MVP runs on a single machine or small cloud instance. No distributed compute needed for this data scale. Pandas \+ Scikit-learn \+ LightGBM is sufficient. Kubernetes, Spark, and Flink are out of scope and would be overengineering.

**NF8 — Observability** Every pipeline run must produce a structured log: what data was read, how many users were scored, what the score distribution looked like, and whether any anomalies were detected. This log is the foundation of your monitoring layer.

---
