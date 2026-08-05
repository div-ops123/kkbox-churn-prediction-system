"""
Tests the hypothesis: "930K train.csv users have no March 2017 expiry record
because their subscription plans are longer, not because data is missing."

Background: train.csv has 970,960 users. Only 40,227 (~4%) have a
transaction row whose membership_expire_date falls in March 2017 (the
"anchored" cohort). The other ~930K users are NOT missing data -- 93% of
them have a LATER known expiry (mostly April/May 2017), and only 3.93%
have zero transaction rows at all. This script tests WHY their expiry
lands later: is it because they bought longer plans (payment_plan_days),
or something else?

DuckDB over read_csv_auto -- data too large to load fully into memory.
"""

# %%
import duckdb

con = duckdb.connect()
DATA = "data"

SEP = "\n" + "=" * 70 + "\n"
SEP2 = "\n" + "-" * 50 + "\n"

def section(title): print(f"{SEP}{title}{SEP}")
def sub(title):     print(f"{SEP2}{title}{SEP2}")
def q(sql):          return con.execute(sql).df()
def show(df, max_rows=40):
    print(df.to_string(index=False, max_rows=max_rows))
    print()

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 -- Build each train.csv user's TRUE latest transaction row, and
#           bucket them by where their most recent membership_expire_date
#           actually lands.
#
# Important: this is NOT "does this user have a March expiry row somewhere
# in their history" -- it's "what does their SINGLE most recent transaction
# say". ROW_NUMBER() picks exactly one row per user (the latest by
# transaction_date), so plan length and expiry are read off the same row,
# not mixed from different transactions.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 1 -- BUCKET train.csv USERS BY THEIR TRUE LATEST TRANSACTION")

BUCKET_SQL = f"""
WITH train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv')),
     ranked AS (
       SELECT tx.*,
         ROW_NUMBER() OVER (
           PARTITION BY tx.msno
           ORDER BY tx.transaction_date DESC, tx.membership_expire_date DESC
         ) AS rn
       FROM train t
       JOIN read_csv_auto('{DATA}/transactions.csv') tx ON t.msno = tx.msno
     ),
     latest AS (SELECT * FROM ranked WHERE rn = 1)
SELECT *,
  CASE
    WHEN membership_expire_date BETWEEN 20170301 AND 20170331 THEN 'expires_march_2017 (anchored)'
    WHEN SUBSTR(CAST(membership_expire_date AS VARCHAR), 1, 6) = '201704' THEN 'expires_april_2017'
    WHEN SUBSTR(CAST(membership_expire_date AS VARCHAR), 1, 6) = '201705' THEN 'expires_may_2017'
    ELSE 'other'
  END AS bucket
FROM latest
"""

sub("Bucket sizes (users with a transaction record)")
show(q(f"""
SELECT bucket, COUNT(*) AS users
FROM ({BUCKET_SQL})
GROUP BY 1 ORDER BY 2 DESC
"""))

sub("Users with NO transaction record at all (can't be bucketed above)")
show(q(f"""
SELECT COUNT(*) AS no_txn_record
FROM read_csv_auto('{DATA}/train.csv') t
LEFT JOIN read_csv_auto('{DATA}/transactions.csv') tx ON t.msno = tx.msno
WHERE tx.msno IS NULL
"""))

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 -- The actual hypothesis test: does payment_plan_days differ
#           between the March-anchored group and the April/May groups?
#
# If longer plans were the explanation, we'd expect the April/May groups
# to show noticeably bigger payment_plan_days than the March group.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 2 -- HYPOTHESIS TEST: IS IT ABOUT LONGER PLANS?")

show(q(f"""
SELECT
  bucket,
  COUNT(*)                          AS users,
  MEDIAN(payment_plan_days)         AS median_plan_days,
  ROUND(AVG(payment_plan_days), 1)  AS avg_plan_days,
  MIN(transaction_date)             AS min_txn_date,
  MAX(transaction_date)             AS max_txn_date
FROM ({BUCKET_SQL})
WHERE bucket IN ('expires_march_2017 (anchored)', 'expires_april_2017', 'expires_may_2017')
GROUP BY bucket
ORDER BY bucket
"""))

sub("Full payment_plan_days distribution, March-anchored vs April-expiry")
show(q(f"""
SELECT bucket, payment_plan_days, COUNT(*) AS cnt
FROM ({BUCKET_SQL})
WHERE bucket IN ('expires_march_2017 (anchored)', 'expires_april_2017')
GROUP BY 1, 2
ORDER BY 1, cnt DESC
LIMIT 20
"""))

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 -- Verdict
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 3 -- VERDICT")
print("If median/avg payment_plan_days is essentially the same across")
print("buckets (~30 days), the hypothesis is REJECTED: it is not plan")
print("length that determines whether a user's latest expiry lands in")
print("March vs April vs May.")
print()
print("What actually determines it: membership_expire_date typically")
print("extends forward from wherever a user's EXISTING expiry already")
print("sits (early renewals stack on top of remaining time), not fresh")
print("from transaction_date + payment_plan_days each time. So where a")
print("user's expiry lands is a product of their whole personal renewal")
print("history as of the extract's cutoff (2017-03-31) -- not the length")
print("of any single plan.")
print()
print("Practical implication: the ~930K 'unanchored' train.csv users are")
print("not missing data by error -- their own subscription genuinely runs")
print("past March in this extract. This means train.csv's cohort is NOT")
print("literally 'everyone whose subscription expires in March 2017' --")
print("only ~4% of train.csv actually shows that. The other ~96% are in")
print("train.csv for reasons this extract alone can't reconstruct.")
