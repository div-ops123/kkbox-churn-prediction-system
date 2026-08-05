# KKBox Churn Dataset — Overview, Issues, and Decisions

This is the single source of truth for what this dataset is, what's wrong with it, what was
investigated, and what was decided as a result. Every claim below is backed by a runnable
query in `explore/` — treat this doc as a summary with receipts, not an assertion of authority.
If a number here ever looks off, re-run the referenced script rather than trust the prose.

For raw column-level schema, see `docs/raw_data_descriptions.md`. For the label derivation
pseudocode, see `docs/label_logic.md`. This doc covers the *why* and *what's broken*; those
two cover the *what*.

---

## 1. What this dataset is

Four source files, joined on `msno` (user ID):

| File | Rows | Unique users | Coverage |
|---|---|---|---|
| `members.csv` | 6,769,473 | 6,769,473 | Full table, static account metadata |
| `transactions.csv` | 1,431,009 | 1,197,050 | 2015-01-01 to 2017-03-31 |
| `user_logs.csv` | 18,396,362 | 1,103,894 | **2017-03-01 to 2017-03-31 only** |
| `train.csv` | 970,960 | 970,960 | Official labels — churn cohort for March 2017 |

Class imbalance: 8.99% churn (87,330 churned / 883,630 retained).

**Two distinct 30/14-day windows exist here — don't conflate them:**
- **KKBox's own churn definition** (used to compute the `is_churn` label): a user is churned if
  no new valid transaction (`is_cancel = 0`) appears within **30 days** of their subscription
  expiring. This defines ground truth, looking *forward* from expiry.
- **This project's serving framing**: predict churn **14 days before** expiry, so marketing/CRM
  can intervene. This defines *feature availability* — an earlier, separate cutoff
  (`feature_cutoff_dt = expire_dt - 14 days`), looking *backward* from expiry.

KKBox's definition also explicitly leans on the fact that most subscriptions are monthly: *"Since
the majority of KKBox's subscription length is 30 days, a lot of users re-subscribe every
month."* This matters later (see §6/§7) — the dataset's natural composition already skews toward
30-day-plan subscribers, independent of any choice made in this project.

---

## 2. Confirmed dataset identity: this is the round-2 (`_v2`) KKBox release

The KKBox churn competition had two rounds with different train cohorts:
- **Round 1**: train = users expiring **Feb 2017** (churn observed by ~March)
- **Round 2 / `_v2`**: train = users expiring **March 2017** (churn observed by ~April)

| Cohort test | Users with that expiry, whole dataset | Also in `train.csv` | Overlap |
|---|---|---|---|
| Feb-2017 expiry | 350 | 25 | 7% |
| March-2017 expiry | 40,227 | 39,972 | **99.4%** |

`train.csv`'s row count (970,960) also matches the known round-2 total, not round-1's 992,931.
**Conclusion: `train.csv` is the March-2017-expiry cohort.** → `explore/date_issues_check.py` Cell 7

---

## 3. Known data issues

### Error 1 — Legacy migration records (2,215 rows)
- `payment_plan_days = 0 AND plan_list_price = 0 AND actual_amount_paid > 0`
- All dated in a ~3-week burst, late April–mid May 2015 — a one-time system migration event, not
  organic transaction traffic
- **Handled**: `src/core/feature_module.py::_build_txn` flags these inline (`is_legacy`) and
  excludes them from plan-price/plan-days features, while keeping them in transaction history
  counts (`n_txn`, `cancel_rate`, etc.) → `explore/date_issues_check.py` Cell 3

### Error 2 — Cannot observe the full renewal window for the March cohort
- The 30-day renewal window for March expiries extends into April 2017; `transactions.csv` stops
  at 2017-03-31 — April data is simply absent
- Quantified: **98.27%** of the 40,227-user March-anchored cohort has a renewal window that runs
  past the data's end — almost none of this cohort's labels are independently verifiable from
  this extract
- **Action**: use `train.csv` labels as authoritative (KKBox computed them with data we don't
  have); never attempt to regenerate labels from this extract → `explore/date_issues_check.py` Cell 1

### Error 3 — Incomplete transactions extract (revised finding — see §5)
- Only ~40,227 of `train.csv`'s 970,960 users (4.1%) have a transaction row with
  `membership_expire_date` in March 2017
- The other ~930K are **not missing data by error**: 84.56% show a real, later expiry in April
  2017, 8.54% in May 2017; only 3.93% (37,382 users) have zero transaction rows at all
- **Open question, unresolved**: `train.csv` is not literally "everyone whose subscription
  expires in March 2017" — only ~4% of it empirically shows that. Why KKBox included the other
  ~96% in this cohort can't be reconstructed from this extract alone
  → `explore/date_issues_check.py` Cell 5, `explore/anchor_hypothesis_check.py`

### Error 4 — Future member registration dates (55,094 rows)
- `registration_init_time > 20170331` (registered after the eval period ends)
- **Handled**: `src/core/feature_module.py::_build_member` excludes these per-user (compares
  against each user's own `feature_cutoff_dt`, not a hardcoded date) — `tenure_days` and
  `registered_via` come back NULL for these users, not a fabricated value

### Related, orthogonal gap — transaction/log feature sparsity in training
Because most training rows fall back to a Feb-15 cutoff (see §6 for why this fallback was later
removed) or have a cutoff before `user_logs.csv` even starts, feature availability in the
original 970K-row training set was extremely sparse:

| Feature group | Null rate (full 970K set) | Root cause |
|---|---|---|
| Transaction features | ~96.7% | Incomplete extract (Error 3) |
| Log features | ~98.2% | `user_logs.csv` covers March only |
| Member features | ~11.4% | Invalid/future registration dates (Error 4) |

This is why transaction and log features had near-zero importance in early training runs — they
were null for nearly all rows, not because they lack predictive value.

---

## 4. Retracted: "Error 2, expire-before-transaction" was never an error

An earlier pass flagged 5,106 rows where `membership_expire_date < transaction_date` as
"logically impossible" — a transaction recorded after its own subscription had expired. It
looked worse than it was: measuring the gap as `transaction_date − membership_expire_date`
using raw `YYYYMMDD` integers produced values like 73–74 (e.g. `20170301 − 20170228 = 73`),
which reads as a ~2.5 month gap. That's an artifact of subtracting across a month boundary as
plain integers — the real calendar gap between Feb 28 and Mar 1 is **1 day**, not 73.

Once measured correctly (parse as real dates, then `DATE_DIFF`), **all 5,106 rows are a 1–3 day
gap**, and 5,104 of them are `is_cancel = 1`. This is completely ordinary: a subscription lapses,
and within a couple of days a cancellation gets formally processed. Not corrupted data, no
exclusion needed — `MAX(membership_expire_date)` per user already handles it fine on its own.

→ `explore/date_issues_check.py` Cell 2 shows the wrong measurement and the corrected one side by side.

**Lesson for future date-gap analysis in this project: never subtract `YYYYMMDD` integers
directly. Always parse to a real date type first.**

---

## 5. The anchor-hypothesis investigation

Given Error 3's finding, the obvious next question: why do ~930K users show a *later* expiry
instead of a March one? Hypothesis tested: maybe these users bought longer subscription plans.

**Rejected.** Median `payment_plan_days` is **30 days** for both the March-anchored group and
the April/May groups — plan length is not the driver. What actually determines it:
`membership_expire_date` typically extends forward from a user's *existing* expiry when they
renew (early renewals stack on top of remaining time), not fresh from
`transaction_date + payment_plan_days`. So where a user's expiry lands as of the 2017-03-31
snapshot is a product of their whole personal renewal history, not plan length or a data gap.

→ `explore/anchor_hypothesis_check.py`

---

## 6. The decision: train on the anchored cohort only

**Chosen approach**: restrict the training set to the ~40,227 users with a *real, verified*
March-2017 anchor. Drop the `has_anchor` flag and the global Feb-15 fallback cutoff entirely —
users without a verified anchor are excluded from training, not filled in with an approximation.

**Reasoning**: every user scored in production has a real, known expiry date pulled directly
from the subscription system — `has_anchor` is always 1 at serving time. Training on the
970K-row set with a fallback cutoff for 96% of rows meant the model partly learned from a
feature pattern (null transaction/log features, a synthetic fixed cutoff) that never occurs in
production. Training exclusively on real per-user anchors removes that train/serve skew instead
of managing around it.

This is treated as **an experiment to validate, not an assumed win** — compare against the
previous full-cohort-plus-fallback approach on the same held-out set before trusting it.

---

## 7. Tradeoffs accepted

Stated plainly, not buried in the code:

- **Sample size collapses**: 970,960 → 39,972 rows.
- **Churn is no longer a minority class in this cohort — verified by running the actual
  pipeline, not estimated**: churn rate jumps from 8.99% (full `train.csv`) to **48.46%**
  (19,367 of 39,972 anchored users). This is a materially bigger shift than "selection skew"
  implies on its own — conditioning on "has a verified March expiry" selects almost
  specifically for users at their renewal decision point. Users who keep renewing roll their
  `membership_expire_date` forward past March and drop out of this cohort entirely (they're
  exactly the ~930K "unanchored" users from §3/§5, 84.56% of whom show a later, April expiry).
  So the anchored cohort isn't just smaller — it's a fundamentally different population,
  weighted toward users already near a churn/renew decision, not a random sample of subscribers.
  This changes the interpretation of any metric trained on it: `scale_pos_weight` will be
  close to 1 instead of ~10, and performance here should not be assumed to generalize to
  scoring a full, unconditioned production cohort without accounting for this shift.
- **Does not fix the log-data gap.** Even within the anchored cohort, **52.6%** of users
  (21,141 of 40,227) still have zero `user_logs` rows before their own cutoff — because
  `user_logs.csv` only covers March, a completely separate root cause from anchor status.
  → `explore/date_issues_check.py` Cell 6
- **Selection skew toward 30-day/monthly-plan subscribers** in the anchored cohort — accepted,
  because KKBox's own churn definition is already built around the 30-day renewal case (see §1).
  Not a bias introduced by this project's choices; it reflects the underlying subscription base.
- **Both cohort choices (40K anchored vs. 970K+fallback) share a limitation neither fixes**:
  training on a single historical calendar month, with no seasonal/temporal variation observed.
- **Still open**: why KKBox included the other ~96% of `train.csv` users in the March cohort at
  all (§3, Error 3) remains unresolved — this extract can't reconstruct KKBox's real internal
  criteria.

---

## 8. References

**Explore scripts** (all runnable via `uv run python explore/<file>.py`, cell-by-cell via `# %%`):

| Script | What it proves |
|---|---|
| `explore/eda.py` | Baseline EDA — schema, missing values, outliers, per-file distributions |
| `explore/cohort_check.py` | Early exploration of March-2017 expiry cohort sizing vs. `train.csv` |
| `explore/label_gen.py` | Derives Feb-2017-cohort labels from transactions; compares to `train.csv` |
| `explore/date_issues_check.py` | The churn-month explanation, all current data issues with real example rows, the Feb-vs-March round confirmation, and the Error 2 retraction (buggy vs. correct gap measurement side by side) |
| `explore/anchor_hypothesis_check.py` | Tests and rejects the plan-length hypothesis for why most `train.csv` users lack a March anchor |

**Other docs**: `docs/raw_data_descriptions.md` (column-level schema), `docs/label_logic.md`
(label derivation pseudocode, includes the April-data-gap warning).
