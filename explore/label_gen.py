"""
Investigates anomalies + generates churn labels from transactions.
Compares derived labels against data/train.csv.
"""
# %%
import duckdb

con = duckdb.connect()
DATA = "data"

SEP  = "\n" + "=" * 70 + "\n"
SEP2 = "\n" + "-" * 50 + "\n"

def section(t): print(f"{SEP}{t}{SEP}")
def sub(t):     print(f"{SEP2}{t}{SEP2}")
def q(sql):     return con.execute(sql).df()
def show(df, max_rows=30): print(df.to_string(index=False, max_rows=max_rows)); print()


# ══════════════════════════════════════════════════════════════════════════════
# Q6  actual_amount_paid > plan_list_price — sample + context
# ══════════════════════════════════════════════════════════════════════════════
section("Q6 — actual_amount_paid > plan_list_price (2,264 rows)")

sub("Random sample of 20 rows")
show(q(f"""
SELECT
  payment_plan_days,
  plan_list_price,
  actual_amount_paid,
  actual_amount_paid - plan_list_price AS overage,
  is_auto_renew,
  is_cancel,
  transaction_date,
  membership_expire_date
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE actual_amount_paid > plan_list_price
ORDER BY RANDOM()
LIMIT 20
"""))

# %%
sub("Cross-tab: is these rows concentrated in specific plan_list_price values?")
show(q(f"""
SELECT
  plan_list_price,
  actual_amount_paid,
  actual_amount_paid - plan_list_price AS overage,
  COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE actual_amount_paid > plan_list_price
GROUP BY 1, 2, 3
ORDER BY cnt DESC
LIMIT 20
"""))

# %%
sub("Do these rows cluster on specific payment_plan_days combos?")
show(q(f"""
SELECT
  payment_plan_days,
  plan_list_price,
  actual_amount_paid,
  COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE actual_amount_paid > plan_list_price
GROUP BY 1, 2, 3
ORDER BY cnt DESC
LIMIT 20
"""))

# %%
# ══════════════════════════════════════════════════════════════════════════════
# Q8  payment_plan_days <= 0 — are prices populated?
# ══════════════════════════════════════════════════════════════════════════════
section("Q8 — payment_plan_days <= 0 (2,218 rows)")

sub("Distribution of payment_plan_days <= 0")
show(q(f"""
SELECT
  payment_plan_days,
  COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days <= 0
GROUP BY 1
ORDER BY 1
"""))

# %%
sub("Sample 20 rows — do plan_list_price / actual_amount_paid have values?")
show(q(f"""
SELECT
  payment_plan_days,
  plan_list_price,
  actual_amount_paid,
  is_auto_renew,
  is_cancel,
  transaction_date,
  membership_expire_date
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days <= 0
ORDER BY RANDOM()
LIMIT 20
"""))

# %%
sub("Price breakdown for payment_plan_days <= 0")
show(q(f"""
SELECT
  plan_list_price,
  actual_amount_paid,
  COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days <= 0
GROUP BY 1, 2
ORDER BY cnt DESC
LIMIT 20
"""))

# %%
# ══════════════════════════════════════════════════════════════════════════════
# Q9  Generate churn labels — February 2017 cohort
# ══════════════════════════════════════════════════════════════════════════════
section("Q9 — Churn label generation (February 2017 cohort)")

sub("Step 1: How many users have a valid expiry in February 2017?")
show(q(f"""
SELECT COUNT(DISTINCT msno) AS users_with_feb_expiry
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date BETWEEN 20170201 AND 20170228
"""))

# %%
sub("Step 2: Generate labels — anchor = latest expiry in Feb 2017")
gen_labels_sql = f"""
WITH txns AS (
  SELECT *,
    STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d')      AS txn_dt,
    STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
  FROM read_csv_auto('{DATA}/transactions.csv')
),
-- Cohort: users whose latest membership_expire_date falls in Feb 2017
-- (we do NOT filter is_cancel here — a user who cancelled still has a real expiry)
feb_cohort AS (
  SELECT msno, MAX(expire_dt) AS anchor_expiry_dt
  FROM txns
  WHERE expire_dt >= '2017-02-01' AND expire_dt <= '2017-02-28'
  GROUP BY msno
),
-- Renewal check: any non-cancel transaction strictly AFTER expiry, within 30 days
renewals AS (
  SELECT DISTINCT t.msno
  FROM txns t
  INNER JOIN feb_cohort fc ON t.msno = fc.msno
  WHERE t.txn_dt > fc.anchor_expiry_dt
    AND t.txn_dt <= fc.anchor_expiry_dt + INTERVAL '30 days'
    AND t.is_cancel = 0
)
SELECT
  fc.msno,
  fc.anchor_expiry_dt                          AS anchor_expiry_date,
  fc.anchor_expiry_dt - INTERVAL '14 days'     AS feature_cutoff_date,
  CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END   AS is_churn
FROM feb_cohort fc
LEFT JOIN renewals r ON fc.msno = r.msno
"""

derived = con.execute(gen_labels_sql).df()
print(f"Derived label rows: {len(derived):,}")
print(f"Derived churn rate: {derived['is_churn'].mean():.4%}")
print()

# %%
sub("Save derived labels")
derived.to_csv(f"{DATA}/derived_labels_feb2017.csv", index=False)
print("Saved to data/derived_labels_feb2017.csv")
print()

# %%
sub("Distribution of feature_cutoff_date (range check)")
show(q(f"""
WITH labels AS (
  SELECT
    MIN(anchor_expiry_dt - INTERVAL '14 days') AS cutoff_min,
    MAX(anchor_expiry_dt - INTERVAL '14 days') AS cutoff_max
  FROM (
    WITH txns AS (
      SELECT *,
        STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d')      AS txn_dt,
        STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
      FROM read_csv_auto('{DATA}/transactions.csv')
    )
    SELECT msno, MAX(expire_dt) AS anchor_expiry_dt
    FROM txns
    WHERE expire_dt >= '2017-02-01' AND expire_dt <= '2017-02-28'
    GROUP BY msno
  )
)
SELECT * FROM labels
"""))

# %%
# ══════════════════════════════════════════════════════════════════════════════
# Q9b  Compare derived labels vs train.csv
# ══════════════════════════════════════════════════════════════════════════════
section("Q9b — Derived labels vs train.csv")

sub("train.csv stats")
show(q(f"""
SELECT COUNT(*) AS total, SUM(is_churn) AS churned,
       ROUND(100.0*SUM(is_churn)/COUNT(*), 4) AS churn_pct
FROM read_csv_auto('{DATA}/train.csv')
"""))

sub("Users in train.csv but NOT in derived labels (cohort miss)")
show(q(f"""
WITH derived AS ({gen_labels_sql}),
     train   AS (SELECT * FROM read_csv_auto('{DATA}/train.csv'))
SELECT COUNT(*) AS in_train_not_derived
FROM train t
LEFT JOIN derived d ON t.msno = d.msno
WHERE d.msno IS NULL
"""))

sub("Users in derived labels but NOT in train.csv")
show(q(f"""
WITH derived AS ({gen_labels_sql}),
     train   AS (SELECT * FROM read_csv_auto('{DATA}/train.csv'))
SELECT COUNT(*) AS in_derived_not_train
FROM derived d
LEFT JOIN train t ON d.msno = t.msno
WHERE t.msno IS NULL
"""))

# %%
sub("Agreement on is_churn for matched users")
show(q(f"""
WITH derived AS ({gen_labels_sql}),
     train   AS (SELECT * FROM read_csv_auto('{DATA}/train.csv')),
     matched AS (
       SELECT d.msno, d.is_churn AS derived_churn, t.is_churn AS train_churn
       FROM derived d
       INNER JOIN train t ON d.msno = t.msno
     )
SELECT
  COUNT(*)                                          AS matched_users,
  SUM(CASE WHEN derived_churn = train_churn THEN 1 ELSE 0 END) AS agree,
  SUM(CASE WHEN derived_churn != train_churn THEN 1 ELSE 0 END) AS disagree,
  ROUND(100.0*SUM(CASE WHEN derived_churn = train_churn THEN 1 ELSE 0 END)/COUNT(*), 4) AS agree_pct,
  -- breakdown of disagreements
  SUM(CASE WHEN derived_churn=1 AND train_churn=0 THEN 1 ELSE 0 END) AS derived1_train0,
  SUM(CASE WHEN derived_churn=0 AND train_churn=1 THEN 1 ELSE 0 END) AS derived0_train1
FROM matched
"""))

# %%
sub("Churn rate comparison on matched users")
show(q(f"""
WITH derived AS ({gen_labels_sql}),
     train   AS (SELECT * FROM read_csv_auto('{DATA}/train.csv')),
     matched AS (
       SELECT d.msno, d.is_churn AS derived_churn, t.is_churn AS train_churn
       FROM derived d INNER JOIN train t ON d.msno = t.msno
     )
SELECT
  ROUND(100.0*AVG(derived_churn), 4) AS derived_churn_pct,
  ROUND(100.0*AVG(train_churn),   4) AS train_churn_pct
FROM matched
"""))

print("\nLabel generation complete.")
