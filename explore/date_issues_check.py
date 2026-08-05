"""
Shows the actual date-related problems in the raw KKBox data, with real
example rows, so they can be seen directly instead of taken on faith from
docs/dataset_overview.md.

DuckDB over read_csv_auto - data too large to load fully into memory.
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
# CELL 1 -- What month are we actually predicting churn for, and why is it hard?
#
# The label cohort is "users whose subscription expires in March 2017"
# (CLAUDE.md > Churn Definition). A user is is_churn=1 if no new valid
# transaction (is_cancel=0) shows up within 30 days of their expiry date.
# That 30-day window is the whole problem: to confirm a user who expires
# on, say, March 31, DIDN'T renew, you need transaction data through
# April 30. transactions.csv stops at March 31, 2017 -- so for anyone
# expiring in the back half of March, we cannot independently verify
# their own label. This cell proves the data cutoff and quantifies how
# many March-cohort users fall in that unverifiable window.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 1 -- THE CHURN MONTH IS MARCH 2017 -- WHY IT'S HARD TO VERIFY")

sub("The raw data's hard stop")
show(q(f"""
SELECT MAX(transaction_date) AS last_date_we_have
FROM read_csv_auto('{DATA}/transactions.csv')
"""))
print("Every transaction record in the entire extract stops here. There is")
print("no April 2017 data at all, in this file or any other.")

sub("March-2017 expiring users, split by whether their 30-day renewal window")
sub("extends past that hard stop (2017-03-31)")
show(q(f"""
WITH txns AS (
  SELECT *, STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
  FROM read_csv_auto('{DATA}/transactions.csv')
),
mar_cohort AS (
  SELECT msno, MAX(expire_dt) AS anchor_expiry_dt
  FROM txns
  WHERE expire_dt BETWEEN '2017-03-01' AND '2017-03-31'
  GROUP BY msno
)
SELECT
  COUNT(*) AS total_march_cohort,
  COUNT(*) FILTER (WHERE anchor_expiry_dt + INTERVAL '30 days' > '2017-03-31')
    AS renewal_window_runs_past_data_end,
  ROUND(100.0 * COUNT(*) FILTER (WHERE anchor_expiry_dt + INTERVAL '30 days' > '2017-03-31')
        / COUNT(*), 2) AS pct_unverifiable
FROM mar_cohort
"""))
print("This is why CLAUDE.md says: use train.csv labels as authoritative --")
print("KKBox generated them with data we were never given (April 2017),")
print("so we cannot regenerate/verify them ourselves for most of the cohort.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 -- membership_expire_date < transaction_date -- NOT actually an error.
#
# First pass at this looked "impossible" (a transaction recorded after its
# own expiry). But that read the raw gap as YYYYMMDD-integer subtraction,
# which is wrong across month boundaries (e.g. 20170301 - 20170228 = 73,
# even though the real calendar gap is 1 day). Once parsed as actual dates
# with DATE_DIFF, every one of these 5,106 rows is a 1-3 day gap -- a
# completely ordinary short grace period between expiry and a cancellation
# being formally processed. Kept here as a worked example of the bug, not
# as a real data issue -- see CLAUDE.md, this is no longer listed as an error.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 2 -- EXPIRE DATE BEFORE TRANSACTION DATE (looked bad, isn't)")

show(q(f"""
SELECT COUNT(*) AS rows_with_expire_before_txn
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date < transaction_date
"""))

sub("WRONG way to measure the gap -- raw integer subtraction on YYYYMMDD columns")
show(q(f"""
SELECT
  msno, transaction_date, membership_expire_date,
  transaction_date - membership_expire_date AS bogus_gap_from_int_subtraction
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date < transaction_date
  AND transaction_date - membership_expire_date > 10
LIMIT 10
"""))
print("These look like 73-74 day gaps. They are not -- transaction_date=20170301")
print("and membership_expire_date=20170228 is 1 real day apart, not 73. Raw")
print("integer subtraction breaks across month boundaries.")

sub("RIGHT way -- parse as real dates, then DATE_DIFF")
show(q(f"""
WITH t AS (
  SELECT *,
    STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d')      AS txn_dt,
    STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
  FROM read_csv_auto('{DATA}/transactions.csv')
  WHERE membership_expire_date < transaction_date
)
SELECT DATE_DIFF('day', expire_dt, txn_dt) AS real_gap_days, COUNT(*) AS cnt
FROM t
GROUP BY 1 ORDER BY 1
"""))

sub("Is this just cancellations? (is_cancel breakdown)")
show(q(f"""
SELECT is_cancel, COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE membership_expire_date < transaction_date
GROUP BY 1
"""))
print("Every row's real gap is 1-3 days, and almost all are is_cancel=1:")
print("a cancellation formally processed within a few days of the plan")
print("lapsing. Ordinary short grace-period behavior, not corrupted data.")
print("No exclusion/flagging needed -- MAX(membership_expire_date) per user")
print("already handles this fine on its own.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 -- Issue: legacy migration rows (payment_plan_days=0, plan_list_price=0,
#           actual_amount_paid > 0)
#
# These rows charged the user real money but recorded zero plan metadata
# -- there's no way to compute "price per day" or "plan length" for them.
# The examples below show real money paid against a $0/0-day plan, all
# clustered in a narrow ~3-week window in 2015, which is the signature of
# a one-time system migration event, not everyday transaction noise.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 3 -- LEGACY ROWS: REAL PAYMENT, BUT ZERO PLAN METADATA")

sub("Real examples")
show(q(f"""
SELECT msno, transaction_date, payment_plan_days, plan_list_price,
       actual_amount_paid, membership_expire_date
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0
ORDER BY RANDOM()
LIMIT 15
"""))

sub("They cluster in one narrow window -- a migration event, not organic activity")
show(q(f"""
SELECT MIN(transaction_date) AS earliest, MAX(transaction_date) AS latest,
       COUNT(*) AS total_rows
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0
"""))
print("All ~2,200 rows fall between late April and mid-May 2015 -- a single")
print("~3-week burst, consistent with a one-time backfill/migration rather")
print("than everyday transaction traffic (which is spread across 2 years).")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 -- Issue: registration_init_time in the future relative to the eval period
#
# The training/eval period ends 2017-03-31. Any user whose registration
# date is AFTER that has no valid "time known as a member" -- tenure_days
# would be negative or nonsensical if computed naively. The examples show
# real registration dates sitting in April 2017, i.e. after the window
# we're supposed to be predicting inside of.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 4 -- MEMBER REGISTRATION DATES AFTER THE EVAL PERIOD ENDS")

sub("Real examples -- registered AFTER 2017-03-31")
show(q(f"""
SELECT msno, city, bd, registered_via, registration_init_time
FROM read_csv_auto('{DATA}/members.csv')
WHERE registration_init_time > 20170331
ORDER BY RANDOM()
LIMIT 15
"""))

sub("How many, and how far into the future do they go?")
show(q(f"""
SELECT
  COUNT(*) FILTER (WHERE registration_init_time > 20170331) AS future_reg_rows,
  MIN(registration_init_time) FILTER (WHERE registration_init_time > 20170331) AS earliest_future_reg,
  MAX(registration_init_time) AS latest_reg_overall
FROM read_csv_auto('{DATA}/members.csv')
"""))
print("55,094 members.csv rows register AFTER the eval period ends -- these")
print("are excluded from tenure_days (treated as NULL), per CLAUDE.md Error 5.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 -- Issue: most train.csv users have NO March-2017 expiry record at all
#
# This is the big one. train.csv has 970,960 labeled users, but only
# ~40K of them have a transaction row showing a March 2017 expiry --
# the very thing we need to compute their feature_cutoff_date. The
# breakdown below shows WHY -- and it's not what you'd guess: their
# latest known expiry is mostly LATER than March 2017 (April/May/beyond),
# not earlier. Meaning most of these users' subscriptions run past March
# entirely -- they were never "expiring in March" in the first place --
# yet train.csv still labels them as part of this cohort.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 5 -- MOST train.csv USERS HAVE NO MARCH-2017 EXPIRY RECORD")

show(q(f"""
WITH train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv')),
     anchored AS (
       SELECT DISTINCT msno FROM read_csv_auto('{DATA}/transactions.csv')
       WHERE membership_expire_date BETWEEN 20170301 AND 20170331
     )
SELECT
  (SELECT COUNT(*) FROM train) AS train_total_users,
  COUNT(*) AS users_with_march_anchor,
  (SELECT COUNT(*) FROM train) - COUNT(*) AS users_missing_march_anchor
FROM anchored a
"""))

sub("For train.csv users WITHOUT a March anchor: when IS their latest known expiry?")
show(q(f"""
WITH train AS (SELECT msno FROM read_csv_auto('{DATA}/train.csv')),
     latest AS (
       SELECT t.msno, MAX(tx.membership_expire_date) AS latest_expire
       FROM train t
       LEFT JOIN read_csv_auto('{DATA}/transactions.csv') tx ON t.msno = tx.msno
       GROUP BY t.msno
       HAVING MAX(tx.membership_expire_date) IS NULL
           OR MAX(tx.membership_expire_date) NOT BETWEEN 20170301 AND 20170331
     )
SELECT
  CASE WHEN latest_expire IS NULL THEN 'no transaction record at all'
       ELSE SUBSTR(CAST(latest_expire AS VARCHAR), 1, 6) END AS latest_known_expiry_yyyymm,
  COUNT(*) AS cnt,
  ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct
FROM latest
GROUP BY 1
ORDER BY cnt DESC
LIMIT 15
"""))
print("~93% of the unanchored users have a LATEST known expiry in April 2017")
print("or later (some as far as 2018) -- their subscription simply runs past")
print("March, so they never had a March expiry row to begin with. Only ~3.9%")
print("have no transaction record at all. So Gap 1 isn't really 'incomplete")
print("extract' for most of these users -- it's that train.csv's cohort is")
print("broader than 'expires in March 2017', and the has_anchor=0 fallback")
print("(feature_cutoff=2017-02-15) is applied to users who may be mid-plan,")
print("not actually approaching an expiry at all.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 -- Issue: user_logs.csv only covers March 2017, so most anchored
#           users still have no engagement data inside their feature window
#
# Even among the ~40K users we CAN anchor, the feature_cutoff_date (expiry
# - 14 days) for early-March expiries falls before March 1 -- before
# user_logs.csv even starts. This cell shows the log file's exact date
# range, then directly checks how many of the anchored cohort actually
# have at least one log row before their own cutoff.
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 6 -- USER_LOGS.CSV COVERS MARCH 2017 ONLY")

show(q(f"""
SELECT MIN(date) AS log_min, MAX(date) AS log_max, COUNT(DISTINCT date) AS distinct_days
FROM read_csv_auto('{DATA}/user_logs.csv')
"""))

sub("Of the anchored March-2017 cohort, how many have ANY log row before their own cutoff?")
show(q(f"""
WITH txns AS (
  SELECT *, STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
  FROM read_csv_auto('{DATA}/transactions.csv')
),
mar_cohort AS (
  SELECT msno, MAX(expire_dt) AS anchor_expiry_dt,
         MAX(expire_dt) - INTERVAL '14 days' AS feature_cutoff_dt
  FROM txns
  WHERE expire_dt BETWEEN '2017-03-01' AND '2017-03-31'
  GROUP BY msno
),
has_log AS (
  SELECT c.msno,
    EXISTS (
      SELECT 1 FROM read_csv_auto('{DATA}/user_logs.csv') l
      WHERE l.msno = c.msno
        AND STRPTIME(CAST(l.date AS VARCHAR), '%Y%m%d') <= c.feature_cutoff_dt
    ) AS has_log_before_cutoff
  FROM mar_cohort c
)
SELECT
  COUNT(*) AS anchored_users,
  SUM(CASE WHEN has_log_before_cutoff THEN 1 ELSE 0 END) AS with_log_data,
  SUM(CASE WHEN NOT has_log_before_cutoff THEN 1 ELSE 0 END) AS without_log_data
FROM has_log
"""))
print("Even for the 'good' anchored users, most still have no engagement")
print("signal because their cutoff falls before user_logs.csv begins.")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 -- Which cohort is train.csv actually? Feb-2017 expiry or March-2017
#           expiry? The official KKBox competition had TWO rounds with
#           different answers, and it's easy to quote the wrong one:
#             Round 1: train = users expiring Feb 2017 (churn observed ~March)
#             Round 2 (_v2 files): train = users expiring March 2017
#                                  (churn observed ~April)
#           transactions.csv here runs through 2017-03-31, and train.csv has
#           exactly 970,960 rows -- both point at round 2 -- but let's not
#           take that on faith. Two direct tests below:
#             (a) row count check
#             (b) which expiry-month cohort actually overlaps train.csv,
#                 and which cohort's KKBox-rule-derived is_churn actually
#                 agrees with train.csv's real labels
# ══════════════════════════════════════════════════════════════════════════════
section("CELL 7 -- IS train.csv THE FEBRUARY OR MARCH 2017 EXPIRY COHORT?")

sub("Row count -- known dataset variants: round-1 train.csv has 992,931 rows, round-2 train_v2.csv has 970,960")
show(q(f"SELECT COUNT(*) AS train_rows FROM read_csv_auto('{DATA}/train.csv')"))

def _cohort_agreement(month_start, month_end, label):
    return q(f"""
    WITH txns AS (
      SELECT *,
        STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d') AS txn_dt,
        STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS expire_dt
      FROM read_csv_auto('{DATA}/transactions.csv')
    ),
    cohort AS (
      SELECT msno, MAX(expire_dt) AS anchor_expiry_dt
      FROM txns
      WHERE expire_dt >= '{month_start}' AND expire_dt <= '{month_end}'
      GROUP BY msno
    ),
    renewals AS (
      SELECT DISTINCT t.msno
      FROM txns t JOIN cohort c ON t.msno = c.msno
      WHERE t.txn_dt > c.anchor_expiry_dt
        AND t.txn_dt <= c.anchor_expiry_dt + INTERVAL 30 DAY
        AND t.is_cancel = 0
    ),
    derived AS (
      SELECT c.msno, CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END AS derived_churn
      FROM cohort c LEFT JOIN renewals r ON c.msno = r.msno
    ),
    train AS (SELECT * FROM read_csv_auto('{DATA}/train.csv')),
    matched AS (
      SELECT d.msno, d.derived_churn, t.is_churn AS train_churn
      FROM derived d JOIN train t ON d.msno = t.msno
    )
    SELECT
      '{label}' AS cohort,
      (SELECT COUNT(*) FROM cohort) AS total_anchored_in_extract,
      COUNT(*) AS also_in_train_csv,
      ROUND(100.0*COUNT(*) / (SELECT COUNT(*) FROM cohort), 2) AS pct_of_cohort_in_train,
      ROUND(100.0*SUM(CASE WHEN derived_churn = train_churn THEN 1 ELSE 0 END)/COUNT(*), 2) AS is_churn_agree_pct
    FROM matched
    """)

sub("Feb-2017-expiry cohort vs train.csv")
show(_cohort_agreement('2017-02-01', '2017-02-28', 'Feb-2017 expiry'))

sub("March-2017-expiry cohort vs train.csv")
show(_cohort_agreement('2017-03-01', '2017-03-31', 'Mar-2017 expiry'))

print("March wins on overlap, decisively: ~99% of identifiable March-expiry")
print("users are in train.csv, vs a handful (350 total, 25 in train) for Feb.")
print("That confirms train.csv IS the March-2017 cohort (round 2 / _v2),")
print("matching CLAUDE.md -- NOT the round-1 'train=Feb, test=March' framing")
print("that's easy to find if you read the competition's original overview.")
print()
print("The March cohort's is_churn agreement with our own derived labels is")
print("only ~57% -- that is NOT evidence against March being the right cohort.")
print("It's the direct consequence of Cell 1: ~98% of the March cohort needs")
print("April transaction data to verify their 30-day renewal, and we don't")
print("have any April data, so our self-derived labels are biased toward")
print("false 'churn' for anyone who actually renewed in April.")

print("\nDate issues check complete.")
