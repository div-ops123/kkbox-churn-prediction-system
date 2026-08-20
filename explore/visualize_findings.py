"""
Visualizes five EDA findings already established (and printed to stdout) by
anchor_hypothesis_check.py and date_issues_check.py, using matplotlib +
seaborn. Re-derives each finding from the same DuckDB queries -- this file
adds a plotting layer on top, it does not compute anything new.

Run from repo root:
    uv run python explore/visualize_findings.py

DuckDB over read_csv_auto -- data too large to load fully into memory.
Note: the log-coverage chart (finding 4) runs a correlated EXISTS check
against user_logs.csv (1.43GB) -- expect it to take noticeably longer than
the other four queries.
"""

# %%
import textwrap
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

con = duckdb.connect()
DATA = "data"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def q(sql):
    return con.execute(sql).df()


# %%
# ── Palette + chart chrome (validated categorical order, dataviz skill reference palette) ──
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY_INK, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

sns.set_theme(style="white")
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SECONDARY_INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "font.family": "sans-serif",
    }
)


def style_ax(fig, ax, title, subtitle=None, wrap=58):
    # Figure-fraction placement (not axes-fraction) so the title block never
    # depends on axes size/pad interactions -- avoids title/subtitle overlap.
    n_lines = 0
    wrapped = None
    if subtitle:
        wrapped = textwrap.fill(subtitle, width=wrap)
        n_lines = wrapped.count("\n") + 1

    left = 0.1
    fig.subplots_adjust(top=0.83 - 0.055 * n_lines, left=left, right=0.96, bottom=0.16)
    fig.text(left, 0.965, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    if wrapped:
        fig.text(left, 0.90, wrapped, fontsize=9.5, color=MUTED, ha="left", va="top", linespacing=1.5)

    ax.grid(axis="y", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def label_bars(ax, bars, fmt="{:,.0f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=SECONDARY_INK,
        )


def savefig(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


# %%
# ══════════════════════════════════════════════════════════════════════════
# Finding 1 -- payment_plan_days hypothesis rejected
# Verbatim BUCKET_SQL from anchor_hypothesis_check.py:46-66
# ══════════════════════════════════════════════════════════════════════════
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

print("Finding 1: payment_plan_days by expiry bucket...")
plan_days = q(f"""
SELECT
  bucket,
  COUNT(*)                          AS users,
  MEDIAN(payment_plan_days)         AS median_plan_days,
  ROUND(AVG(payment_plan_days), 1)  AS avg_plan_days
FROM ({BUCKET_SQL})
WHERE bucket IN ('expires_march_2017 (anchored)', 'expires_april_2017', 'expires_may_2017')
GROUP BY bucket
ORDER BY bucket
""")

order = ["expires_march_2017 (anchored)", "expires_april_2017", "expires_may_2017"]
plan_days = plan_days.set_index("bucket").loc[order].reset_index()
labels = ["March\n(anchored)", "April", "May"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(labels, plan_days["median_plan_days"], color=[BLUE, ORANGE, AQUA], width=0.55)
label_bars(ax, bars, fmt="{:.0f} days")
ax.set_ylim(0, plan_days["median_plan_days"].max() * 1.3)
style_ax(
    fig,
    ax,
    "Median payment_plan_days is flat across expiry buckets",
    "Hypothesis rejected: longer plans don't explain why 96% of train.csv expires after March",
)
savefig(fig, "01_payment_plan_days_by_cohort.png")

# %%
# ══════════════════════════════════════════════════════════════════════════
# Finding 2 -- the "expire < transaction" date-arithmetic false alarm
# Verbatim query from date_issues_check.py:110-121
# ══════════════════════════════════════════════════════════════════════════
print("Finding 2: real day-gap distribution...")
gap_dist = q(f"""
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
""")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(gap_dist["real_gap_days"].astype(str) + " day", gap_dist["cnt"], color=BLUE, width=0.5)
label_bars(ax, bars)
style_ax(
    fig,
    ax,
    "The 5,106 'expire before transaction' rows are 1-3 day gaps",
    "Raw YYYYMMDD int subtraction made these look like 73-day anomalies -- real gap is a normal grace period",
)
savefig(fig, "02_date_gap_distribution.png")

# %%
# ══════════════════════════════════════════════════════════════════════════
# Finding 3 -- ~2,200-row legacy-migration anomaly cluster (Apr-May 2015)
# Same filter as date_issues_check.py:150-165, grouped by day for the chart
# ══════════════════════════════════════════════════════════════════════════
print("Finding 3: legacy-migration cluster by day...")
legacy = q(f"""
SELECT transaction_date, COUNT(*) AS cnt
FROM read_csv_auto('{DATA}/transactions.csv')
WHERE payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0
GROUP BY 1 ORDER BY 1
""")
legacy["txn_dt"] = pd.to_datetime(legacy["transaction_date"], format="%Y%m%d")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(legacy["txn_dt"], legacy["cnt"], color=BLUE, width=1.2)
fig.autofmt_xdate(rotation=30, ha="right")
style_ax(
    fig,
    ax,
    f"{int(legacy['cnt'].sum()):,} zero-plan-metadata rows cluster in one 3-week window",
    "Real payment against $0 / 0-day plans, isolated to Apr-May 2015 -- a one-time migration, not ongoing noise",
)
savefig(fig, "03_legacy_migration_cluster.png")

# %%
# ══════════════════════════════════════════════════════════════════════════
# Finding 4 -- 52.6% log-coverage gap among anchored March-2017 users
# Verbatim query from date_issues_check.py:279-305 (slow: correlated EXISTS
# over user_logs.csv, 1.43GB)
# ══════════════════════════════════════════════════════════════════════════
print("Finding 4: log-coverage gap (this one is slow -- scans user_logs.csv)...")
log_cov = q(f"""
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
""")

total = int(log_cov["anchored_users"].iloc[0])
with_log = int(log_cov["with_log_data"].iloc[0])
without_log = int(log_cov["without_log_data"].iloc[0])
pct_without = 100 * without_log / total

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(["Has log data\nbefore cutoff", "No log data\nbefore cutoff"], [with_log, without_log], color=[BLUE, ORANGE], width=0.55)
label_bars(ax, bars)
style_ax(
    fig,
    ax,
    f"{pct_without:.1f}% of anchored users have zero usage-log history",
    f"Of {total:,} anchored March-2017 users -- user_logs.csv only starts March 1, missing early-month cutoffs",
)
savefig(fig, "04_log_coverage_gap.png")

# %%
# ══════════════════════════════════════════════════════════════════════════
# Finding 5 -- March-2017 vs Feb-2017: which round is train.csv?
# Verbatim _cohort_agreement() from date_issues_check.py:330-367
# ══════════════════════════════════════════════════════════════════════════
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
      ROUND(100.0*COUNT(*) / (SELECT COUNT(*) FROM cohort), 2) AS pct_of_cohort_in_train
    FROM matched
    """)

print("Finding 5: Feb-2017 vs March-2017 cohort overlap with train.csv...")
overlap = pd.concat(
    [
        _cohort_agreement("2017-02-01", "2017-02-28", "Feb-2017\nexpiry"),
        _cohort_agreement("2017-03-01", "2017-03-31", "Mar-2017\nexpiry"),
    ],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(overlap["cohort"], overlap["pct_of_cohort_in_train"], color=[ORANGE, BLUE], width=0.5)
label_bars(ax, bars, fmt="{:.1f}%")
ax.set_ylim(0, 110)
style_ax(
    fig,
    ax,
    "March-2017 expiry cohort matches train.csv, not February",
    "% of each candidate cohort found in train.csv -- March overlaps 99.4%, February only 7.1%",
)
savefig(fig, "05_cohort_overlap_train_csv.png")

print("\nAll 5 figures written to explore/figures/")
