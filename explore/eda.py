"""
KKBox Churn EDA — DuckDB-based, no full in-memory loads.
"""


# %%
import duckdb

# %%
con = duckdb.connect()



# %%
DATA = "data"

SEP = "\n" + "=" * 70 + "\n"
SEP2 = "\n" + "-" * 50 + "\n"

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"{SEP}{title}{SEP}")

def sub(title):
    print(f"{SEP2}{title}{SEP2}")

def q(sql):
    return con.execute(sql).df()

def show(df, max_rows=40):
    with duckdb.connect() as _:
        pass
    print(df.to_string(index=False, max_rows=max_rows))
    print()

# %%

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — EDA per file
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────
# 1. MEMBERS
# ─────────────────────────────────────────
section("MEMBERS.CSV")

sub("1. Schema + row count")
members_schema = q(f"DESCRIBE SELECT * FROM read_csv_auto('{DATA}/members.csv')")
show(members_schema)
row_count = q(f"SELECT COUNT(*) AS row_count FROM read_csv_auto('{DATA}/members.csv')")
show(row_count)

# %%

sub("2. Missing values")
show(q(f"""
WITH t AS (SELECT * FROM read_csv_auto('{DATA}/members.csv'))
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) - COUNT(msno)              AS msno_null,
  COUNT(*) - COUNT(city)              AS city_null,
  COUNT(*) - COUNT(bd)                AS bd_null,
  COUNT(*) - COUNT(gender)            AS gender_null,
  COUNT(*) - COUNT(registered_via)    AS registered_via_null,
  COUNT(*) - COUNT(registration_init_time) AS reg_init_null
FROM t
"""))

show(q(f"""
SELECT
  ROUND(100.0*(COUNT(*) - COUNT(msno))/COUNT(*), 2)              AS msno_pct_null,
  ROUND(100.0*(COUNT(*) - COUNT(city))/COUNT(*), 2)              AS city_pct_null,
  ROUND(100.0*(COUNT(*) - COUNT(bd))/COUNT(*), 2)                AS bd_pct_null,
  ROUND(100.0*(COUNT(*) - COUNT(gender))/COUNT(*), 2)            AS gender_pct_null,
  ROUND(100.0*(COUNT(*) - COUNT(registered_via))/COUNT(*), 2)    AS registered_via_pct_null,
  ROUND(100.0*(COUNT(*) - COUNT(registration_init_time))/COUNT(*), 2) AS reg_init_pct_null
FROM read_csv_auto('{DATA}/members.csv')
"""))

sub("3. Numeric stats — bd")
show(q(f"""
SELECT
  COUNT(bd)                    AS bd_count,
  MIN(bd)                      AS bd_min,
  MAX(bd)                      AS bd_max,
  ROUND(AVG(bd), 2)            AS bd_mean,
  MEDIAN(bd)                   AS bd_median,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bd) AS bd_p25,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bd) AS bd_p75
FROM read_csv_auto('{DATA}/members.csv')
"""))

# %%

sub("3b. bd outlier buckets")
show(q(f"""
SELECT
  CASE
    WHEN bd IS NULL THEN 'NULL'
    WHEN bd <= 0    THEN '<=0 (invalid)'
    WHEN bd < 10    THEN '1–9'
    WHEN bd <= 100  THEN '10–100 (plausible)'
    ELSE            '>100 (outlier)'
  END AS bucket,
  COUNT(*) AS cnt,
  ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/members.csv')
GROUP BY 1 ORDER BY 2 DESC
"""))

# %%

sub("4. Categorical — city")
show(q(f"""
SELECT city, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/members.csv')
GROUP BY 1 ORDER BY 2 DESC LIMIT 15
"""))


sub("4. Categorical — gender")
show(q(f"""
SELECT gender, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/members.csv')
GROUP BY 1 ORDER BY 2 DESC
"""))

sub("4. Categorical — registered_via")
show(q(f"""
SELECT registered_via, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/members.csv')
GROUP BY 1 ORDER BY 2 DESC
"""))

# %%
sub("5. registration_init_time range + anomalies")
show(q(f"""
SELECT
  MIN(registration_init_time) AS earliest_reg_yyyymmdd,
  MAX(registration_init_time) AS latest_reg_yyyymmdd,
  COUNT(*) FILTER (WHERE registration_init_time IS NULL) AS null_dates,
  COUNT(*) FILTER (WHERE registration_init_time < 20000101) AS suspiciously_old,
  COUNT(*) FILTER (WHERE registration_init_time > 20170331) AS after_eval_period
FROM read_csv_auto('{DATA}/members.csv')
"""))

# %%

# ─────────────────────────────────────────
# 2. TRANSACTIONS
# ─────────────────────────────────────────
section("TRANSACTIONS.CSV")

sub("1. Schema + row count")
show(q(f"DESCRIBE SELECT * FROM read_csv_auto('{DATA}/transactions.csv')"))
show(q(f"SELECT COUNT(*) AS row_count FROM read_csv_auto('{DATA}/transactions.csv')"))

sub("2. Unique users")
show(q(f"SELECT COUNT(DISTINCT msno) AS unique_users FROM read_csv_auto('{DATA}/transactions.csv')"))

# %%

sub("3. Missing values")
show(q(f"""
SELECT
  COUNT(*) - COUNT(msno)                   AS msno_null,
  COUNT(*) - COUNT(payment_method_id)       AS payment_method_null,
  COUNT(*) - COUNT(payment_plan_days)       AS plan_days_null,
  COUNT(*) - COUNT(plan_list_price)         AS list_price_null,
  COUNT(*) - COUNT(actual_amount_paid)      AS actual_paid_null,
  COUNT(*) - COUNT(is_auto_renew)           AS auto_renew_null,
  COUNT(*) - COUNT(transaction_date)        AS txn_date_null,
  COUNT(*) - COUNT(membership_expire_date)  AS expire_date_null,
  COUNT(*) - COUNT(is_cancel)               AS is_cancel_null
FROM read_csv_auto('{DATA}/transactions.csv')
"""))
# %%

sub("4. Numeric stats")
show(q(f"""
SELECT
  MIN(payment_plan_days)  AS plan_days_min,
  MAX(payment_plan_days)  AS plan_days_max,
  ROUND(AVG(payment_plan_days), 2) AS plan_days_mean,
  MEDIAN(payment_plan_days)        AS plan_days_median
FROM read_csv_auto('{DATA}/transactions.csv')
"""))

# %%

show(q(f"""
SELECT
  MIN(plan_list_price)    AS list_price_min,
  MAX(plan_list_price)    AS list_price_max,
  ROUND(AVG(plan_list_price), 2) AS list_price_mean,
  MEDIAN(plan_list_price)        AS list_price_median,
  MIN(actual_amount_paid) AS actual_min,
  MAX(actual_amount_paid) AS actual_max,
  ROUND(AVG(actual_amount_paid), 2) AS actual_mean,
  MEDIAN(actual_amount_paid)         AS actual_median
FROM read_csv_auto('{DATA}/transactions.csv')
"""))

# %%

sub("5. Anomalies — negative / zero prices")
show(q(f"""
SELECT
  COUNT(*) FILTER (WHERE plan_list_price < 0)    AS negative_list_price,
  COUNT(*) FILTER (WHERE actual_amount_paid < 0) AS negative_actual_paid,
  COUNT(*) FILTER (WHERE actual_amount_paid = 0) AS zero_actual_paid,
  COUNT(*) FILTER (WHERE payment_plan_days <= 0) AS non_positive_plan_days
FROM read_csv_auto('{DATA}/transactions.csv')
"""))


sub("5b. Anomalies — actual_paid > list_price")
show(q(f"""
SELECT COUNT(*) AS overpaid_rows
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE actual_amount_paid > plan_list_price
"""))

# %%

sub("5c. Date range — transaction_date (YYYYMMDD integers)")
show(q(f"""
SELECT
  MIN(transaction_date)  AS txn_min,
  MAX(transaction_date)  AS txn_max,
  COUNT(*) FILTER (WHERE transaction_date < 20150101) AS before_2015,
  COUNT(*) FILTER (WHERE transaction_date > 20170331) AS after_mar2017
FROM read_csv_auto('{DATA}/transactions.csv')
"""))

sub("5d. Date range — membership_expire_date")
show(q(f"""
SELECT
  MIN(membership_expire_date) AS expire_min,
  MAX(membership_expire_date) AS expire_max,
  COUNT(*) FILTER (WHERE membership_expire_date < transaction_date) AS expire_before_txn,
  COUNT(*) FILTER (WHERE membership_expire_date < 20150101) AS expire_before_2015,
  COUNT(*) FILTER (WHERE membership_expire_date > 20991231) AS expire_far_future
FROM read_csv_auto('{DATA}/transactions.csv')
"""))

# %%
sub("4. Categorical — payment_method_id (top 10)")
show(q(f"""
SELECT payment_method_id, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC LIMIT 10
"""))

# %%
sub("4. Categorical — is_auto_renew")
show(q(f"""
SELECT is_auto_renew, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC
"""))

sub("4. Categorical — is_cancel")
show(q(f"""
SELECT is_cancel, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC
"""))

# %%
sub("4. Categorical — payment_plan_days (top 10)")
show(q(f"""
SELECT payment_plan_days, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC LIMIT 10
"""))

sub("6. Transactions per user distribution")
show(q(f"""
WITH per_user AS (
  SELECT msno, COUNT(*) AS txn_count
  FROM read_csv_auto('{DATA}/transactions.csv')
  GROUP BY msno
)
SELECT
  MIN(txn_count) AS min_txns,
  MAX(txn_count) AS max_txns,
  ROUND(AVG(txn_count), 2) AS mean_txns,
  MEDIAN(txn_count) AS median_txns,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY txn_count) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY txn_count) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY txn_count) AS p99
FROM per_user
"""))


# ─────────────────────────────────────────
# 3. USER LOGS
# ─────────────────────────────────────────
section("USER_LOGS.CSV")

sub("1. Schema + row count")
show(q(f"DESCRIBE SELECT * FROM read_csv_auto('{DATA}/user_logs.csv')"))
show(q(f"SELECT COUNT(*) AS row_count FROM read_csv_auto('{DATA}/user_logs.csv')"))

sub("2. Unique users")
show(q(f"SELECT COUNT(DISTINCT msno) AS unique_users FROM read_csv_auto('{DATA}/user_logs.csv')"))

sub("3. Missing values")
show(q(f"""
SELECT
  ROUND(100.0*(COUNT(*)-COUNT(msno))/COUNT(*), 2)       AS msno_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(date))/COUNT(*), 2)        AS date_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_25))/COUNT(*), 2)      AS num_25_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_50))/COUNT(*), 2)      AS num_50_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_75))/COUNT(*), 2)      AS num_75_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_985))/COUNT(*), 2)     AS num_985_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_100))/COUNT(*), 2)     AS num_100_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(num_unq))/COUNT(*), 2)     AS num_unq_pct_null,
  ROUND(100.0*(COUNT(*)-COUNT(total_secs))/COUNT(*), 2)  AS total_secs_pct_null
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

sub("4. Date range")
show(q(f"""
SELECT
  MIN(date) AS log_date_min,
  MAX(date) AS log_date_max,
  COUNT(DISTINCT date) AS distinct_dates
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

sub("4. Numeric stats — listening columns")
show(q(f"""
SELECT
  MIN(num_25)   AS n25_min,  MAX(num_25)  AS n25_max,  ROUND(AVG(num_25),2)  AS n25_mean,  MEDIAN(num_25)  AS n25_med,
  MIN(num_50)   AS n50_min,  MAX(num_50)  AS n50_max,  ROUND(AVG(num_50),2)  AS n50_mean,  MEDIAN(num_50)  AS n50_med,
  MIN(num_75)   AS n75_min,  MAX(num_75)  AS n75_max,  ROUND(AVG(num_75),2)  AS n75_mean,  MEDIAN(num_75)  AS n75_med,
  MIN(num_985)  AS n985_min, MAX(num_985) AS n985_max, ROUND(AVG(num_985),2) AS n985_mean, MEDIAN(num_985) AS n985_med,
  MIN(num_100)  AS n100_min, MAX(num_100) AS n100_max, ROUND(AVG(num_100),2) AS n100_mean, MEDIAN(num_100) AS n100_med
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

show(q(f"""
SELECT
  MIN(num_unq)     AS unq_min,    MAX(num_unq)     AS unq_max,    ROUND(AVG(num_unq),2)     AS unq_mean,    MEDIAN(num_unq)     AS unq_med,
  MIN(total_secs)  AS secs_min,   MAX(total_secs)  AS secs_max,   ROUND(AVG(total_secs),2)  AS secs_mean,   MEDIAN(total_secs)  AS secs_med
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

sub("5. Anomalies — negatives")
show(q(f"""
SELECT
  COUNT(*) FILTER (WHERE num_25 < 0)    AS neg_num25,
  COUNT(*) FILTER (WHERE num_50 < 0)    AS neg_num50,
  COUNT(*) FILTER (WHERE num_75 < 0)    AS neg_num75,
  COUNT(*) FILTER (WHERE num_985 < 0)   AS neg_num985,
  COUNT(*) FILTER (WHERE num_100 < 0)   AS neg_num100,
  COUNT(*) FILTER (WHERE total_secs < 0) AS neg_total_secs,
  COUNT(*) FILTER (WHERE num_unq < 0)   AS neg_num_unq
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

sub("5b. Zero-activity rows")
show(q(f"""
SELECT COUNT(*) AS zero_activity_rows
FROM read_csv_auto('{DATA}/user_logs.csv')
WHERE num_25=0 AND num_50=0 AND num_75=0 AND num_985=0 AND num_100=0 AND total_secs=0
"""))

sub("6. Logs per user distribution")
show(q(f"""
WITH per_user AS (
  SELECT msno, COUNT(*) AS log_days
  FROM read_csv_auto('{DATA}/user_logs.csv')
  GROUP BY msno
)
SELECT
  MIN(log_days)  AS min_days,
  MAX(log_days)  AS max_days,
  ROUND(AVG(log_days),2) AS mean_days,
  MEDIAN(log_days) AS median_days,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY log_days) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY log_days) AS p95
FROM per_user
"""))

sub("6b. total_secs percentiles (engagement spread)")
show(q(f"""
SELECT
  PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY total_secs) AS p10,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_secs) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_secs) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_secs) AS p75,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_secs) AS p90,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_secs) AS p99,
  MAX(total_secs) AS max_secs
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Univariate analysis (members, transactions, user_logs)
# ══════════════════════════════════════════════════════════════════════════════

section("TASK 2 — UNIVARIATE ANALYSIS")

# ── Members: bd distribution shape ──────────────────────────────────────────
sub("Members — bd (valid only: 10–100)")
show(q(f"""
WITH valid AS (
  SELECT bd FROM read_csv_auto('{DATA}/members.csv')
  WHERE bd >= 10 AND bd <= 100
)
SELECT
  ROUND(AVG(bd),2) AS mean,
  MEDIAN(bd)       AS median,
  MIN(bd)  AS min,
  MAX(bd)  AS max,
  ROUND(STDDEV(bd),2) AS stddev,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bd) AS p25,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bd) AS p75
FROM valid
"""))

sub("Members — registration_init_time (year distribution)")
show(q(f"""
SELECT
  SUBSTR(CAST(registration_init_time AS VARCHAR), 1, 4) AS reg_year,
  COUNT(*) AS cnt,
  ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/members.csv')
WHERE registration_init_time IS NOT NULL
GROUP BY 1 ORDER BY 1
"""))

# ── Transactions: plan_list_price distribution ───────────────────────────────
sub("Transactions — plan_list_price distribution (top values)")
show(q(f"""
SELECT plan_list_price, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC LIMIT 15
"""))

sub("Transactions — actual_amount_paid percentiles")
show(q(f"""
SELECT
  PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY actual_amount_paid) AS p10,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_amount_paid) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_amount_paid) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_amount_paid) AS p75,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY actual_amount_paid) AS p90,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY actual_amount_paid) AS p99,
  MAX(actual_amount_paid) AS max_paid
FROM read_csv_auto('{DATA}/transactions.csv')
"""))

sub("Transactions — discount (list - actual) percentiles")
show(q(f"""
WITH disc AS (
  SELECT plan_list_price - actual_amount_paid AS discount
  FROM read_csv_auto('{DATA}/transactions.csv')
)
SELECT
  MIN(discount)  AS min_discount,
  MAX(discount)  AS max_discount,
  ROUND(AVG(discount),2) AS mean_discount,
  MEDIAN(discount)       AS median_discount,
  COUNT(*) FILTER (WHERE discount > 0) AS discounted_rows,
  COUNT(*) FILTER (WHERE discount < 0) AS over_paid_rows,
  COUNT(*) FILTER (WHERE discount = 0) AS full_price_rows
FROM disc
"""))

sub("Transactions — payment_plan_days distribution")
show(q(f"""
SELECT payment_plan_days, COUNT(*) AS cnt,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_csv_auto('{DATA}/transactions.csv')
GROUP BY 1 ORDER BY 2 DESC LIMIT 15
"""))

# ── User logs: song play completeness ratio ──────────────────────────────────
sub("User logs — completion ratio (num_100 / total songs) percentiles")
show(q(f"""
WITH ratio AS (
  SELECT
    CASE WHEN (num_25+num_50+num_75+num_985+num_100) = 0 THEN NULL
         ELSE CAST(num_100 AS DOUBLE)/(num_25+num_50+num_75+num_985+num_100)
    END AS completion_ratio
  FROM read_csv_auto('{DATA}/user_logs.csv')
)
SELECT
  ROUND(AVG(completion_ratio),4)    AS mean_ratio,
  MEDIAN(completion_ratio)          AS median_ratio,
  PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY completion_ratio) AS p10,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY completion_ratio) AS p90
FROM ratio
"""))

sub("User logs — total_secs to hours per day")
show(q(f"""
SELECT
  ROUND(AVG(total_secs)/3600, 2) AS mean_hours_per_day,
  ROUND(MEDIAN(total_secs)/3600, 2) AS median_hours_per_day,
  COUNT(*) FILTER (WHERE total_secs/3600 > 24) AS impossible_over_24h
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Churn rate in train.csv
# ══════════════════════════════════════════════════════════════════════════════

section("TASK 3 — CHURN RATE (train.csv)")

show(q(f"""
SELECT
  COUNT(*)                            AS total_users,
  SUM(is_churn)                       AS churned,
  COUNT(*) - SUM(is_churn)            AS retained,
  ROUND(100.0*SUM(is_churn)/COUNT(*), 4) AS churn_rate_pct
FROM read_csv_auto('{DATA}/train.csv')
"""))

print("EDA complete.")
