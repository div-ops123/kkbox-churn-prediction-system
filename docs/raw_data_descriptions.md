# Churn Prediction Dataset Overview

This document describes the raw tables used in the churn prediction system. All datasets are centered around a unique user identifier: `msno`.

---

## 1. train.csv

**Time reference:**  
Churn status for **March 2017**

**Schema:**

- `msno`: Unique user ID
- `is_churn`: Target label  
  - `1` → user churned in March 2017  
  - `0` → user did not churn

---

## 2. transactions.csv

**Purpose:**  
Records all user payment and subscription transactions from `2015-01-01` to `2017-03-31`.

**Schema:**

- `msno`: Unique user ID
- `payment_method_id`: Payment method identifier
- `payment_plan_days`: Subscription length in days
- `plan_list_price`: Listed price (NTD)
- `actual_amount_paid`: Final amount paid (NTD)
- `is_auto_renew`: Whether auto-renew was enabled
- `transaction_date`: Date of transaction (`YYYYMMDD`)
- `membership_expire_date`: Subscription expiration date (`YYYYMMDD`)
- `is_cancel`: Whether the transaction involved a cancellation

---

## 3. user_logs.csv

**Purpose:**  
Daily user listening behavior logs (usage signals) all days between `March 1, 2017` to `March 31 2017`.

**Schema:**

- `msno`: Unique user ID
- `date`: Log date (`YYYYMMDD`)
- `num_25`: Songs played < 25% of duration
- `num_50`: Songs played 25%–50%
- `num_75`: Songs played 50%–75%
- `num_985`: Songs played 75%–98.5%
- `num_100`: Songs played > 98.5%
- `num_unq`: Number of unique songs played
- `total_secs`: Total listening time in seconds

---

## 4. members.csv

**Purpose:**  
Static user demographic and account metadata. Not all users exist in this table.

**Schema:**

- `msno`: Unique user ID
- `city`: User’s city
- `bd`: Age (contains noisy/outlier values; requires cleaning)
- `gender`: Male / Female / Unknown
- `registered_via`: Registration channel/source
- `registration_init_time`: Account creation date (`YYYYMMDD`)

---