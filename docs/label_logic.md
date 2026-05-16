=============================

## Churn Label Logic

- Cohort: users whose latest membership_expire_date falls in March 2017 (20170301–20170331)
- Anchor expiry = that March date
- Feature cutoff = anchor expiry − 14 days → ranges Feb 15 to March 17
- Renewal window: (anchor expiry, anchor expiry + 30 days] → extends into April 2017
- Label: no non-cancel renewal in that window → is_churn = 1

WARNING:
We cannot observe the full renewal window.
The transactions.csv ends at March 31. For users expiring March 15–31, the 30-day renewal window runs into April — which we don't have. This is why the labels had to come from KKBox's full dataset.


## Label Derivation end-to-end

input:  transactions table, target cohort month
output: user_id, anchor_expiry_date, feature_cutoff_date, is_churn

it must be:
- deterministic (same input = same output always)
- independently testable
- reusable for any historical month during training
- identical in logic to how you will generate labels in production monitoring


Step 1: For each user, find their latest membership_expire_date
        that falls within your target cohort month (e.g. February 2017)

Step 2: That date becomes their anchor expiry date

Step 3: Feature cutoff = anchor expiry date - 14 days
        Pull ALL user_logs and transaction features 
        strictly BEFORE this cutoff date

Step 4: Look forward 30 days from anchor expiry date
        Check transactions table for any record where:
        - transaction_date is within (expiry_date, expiry_date + 30 days]
        - is_cancel = 0

Step 5: If no such record exists → is_churn = 1
        If such a record exists → is_churn = 0


==================================================================

## Training Data
You are not limited to one day's worth of users.
You collect all users whose subscription expired within a given month. Each user has their own expiry date. User A expires Feb 3rd, User B expires Feb 17th, User C expires Feb 28th. You treat each user independently:
User A: feature cutoff = Feb 3 - 14 = Jan 20
User B: feature cutoff = Feb 17 - 14 = Feb 3  
User C: feature cutoff = Feb 28 - 14 = Feb 14
Every user in that month's cohort is a valid training example.
