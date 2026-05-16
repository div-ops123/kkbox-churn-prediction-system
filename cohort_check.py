# %%
import duckdb
con = duckdb.connect()
DATA = "data"

print("=== Latest membership_expire_date for each train.csv user ===")
print(con.execute(f"""
WITH train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv')),
     latest AS (
       SELECT t.msno, MAX(membership_expire_date) AS latest_expire
       FROM read_csv_auto('{DATA}/transactions.csv') t
       INNER JOIN train tr ON t.msno = tr.msno
       GROUP BY t.msno
     )
SELECT
  SUBSTR(CAST(latest_expire AS VARCHAR), 1, 6) AS expire_ym,
  COUNT(*) AS cnt,
  ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(), 2) AS pct
FROM latest
GROUP BY 1 ORDER BY cnt DESC LIMIT 20
""").df().to_string(index=False))

# %%
print()
print("=== Train users with NO transaction record ===")
print(con.execute(f"""
WITH train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv'))
SELECT COUNT(*) AS no_txn_match
FROM train t
LEFT JOIN read_csv_auto('{DATA}/transactions.csv') tx ON t.msno = tx.msno
WHERE tx.msno IS NULL
""").df().to_string(index=False))

# %%
print()
print("=== Try March 2017 cohort — how many users? ===")
print(con.execute(f"""
SELECT COUNT(DISTINCT msno) AS users_with_march_expiry
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date BETWEEN 20170301 AND 20170331
""").df().to_string(index=False))

# %%
# Try generating March 2017 cohort labels
print()
print("=== March 2017 cohort vs train.csv overlap ===")
print(con.execute(f"""
WITH txns AS (
  SELECT *,
    STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d')       AS txn_dt,
    STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')  AS expire_dt
  FROM read_csv_auto('{DATA}/transactions.csv')
),
mar_cohort AS (
  SELECT msno, MAX(expire_dt) AS anchor_expiry_dt
  FROM txns
  WHERE expire_dt >= '2017-03-01' AND expire_dt <= '2017-03-31'
  GROUP BY msno
),
-- renewal: non-cancel txn after expiry within 30 days
-- Note: we only have data to 2017-03-31 so this will be incomplete for late March
renewals AS (
  SELECT DISTINCT t.msno
  FROM txns t
  INNER JOIN mar_cohort fc ON t.msno = fc.msno
  WHERE t.txn_dt > fc.anchor_expiry_dt
    AND t.txn_dt <= fc.anchor_expiry_dt + INTERVAL '30 days'
    AND t.is_cancel = 0
),
derived AS (
  SELECT fc.msno,
    CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END AS is_churn
  FROM mar_cohort fc LEFT JOIN renewals r ON fc.msno = r.msno
),
train AS (SELECT * FROM read_csv_auto('{DATA}/train.csv'))
SELECT
  COUNT(DISTINCT d.msno)                                                   AS derived_users,
  COUNT(DISTINCT t.msno)                                                   AS train_users,
  SUM(CASE WHEN t.msno IS NOT NULL THEN 1 ELSE 0 END)                     AS matched,
  SUM(CASE WHEN t.msno IS NULL THEN 1 ELSE 0 END)                         AS derived_not_in_train
FROM derived d LEFT JOIN train t ON d.msno = t.msno
""").df().to_string(index=False))

# %%
# Check if it is ALL users active (expiry >= some date) rather than a specific month
print()
print("=== Users with expiry >= 2017-02-01 (all recent subscriptions) ===")
print(con.execute(f"""
SELECT COUNT(DISTINCT msno) AS users
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date >= 20170201
  AND is_cancel = 0
""").df().to_string(index=False))

print()
print("=== Latest expiry >= 2017-02-01 per user — overlap with train.csv ===")
print(con.execute(f"""
WITH latest AS (
  SELECT msno, MAX(membership_expire_date) AS latest_expire
  FROM read_csv_auto('{DATA}/transactions.csv')
  WHERE is_cancel = 0
  GROUP BY msno
  HAVING MAX(membership_expire_date) >= 20170201
),
train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv'))
SELECT
  COUNT(DISTINCT l.msno)                                       AS latest_users,
  COUNT(DISTINCT CASE WHEN t.msno IS NOT NULL THEN l.msno END) AS in_train,
  COUNT(DISTINCT CASE WHEN t.msno IS NULL THEN l.msno END)     AS not_in_train
FROM latest l LEFT JOIN train t ON l.msno = t.msno
""").df().to_string(index=False))
