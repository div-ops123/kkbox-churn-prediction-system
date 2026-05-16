Data is in data/:
- transactions.csv
- user_logs.csv
- members.csv

Files are large → use DuckDB (do not load fully into memory).

Goal:
Understand data quality + structure before feature engineering for a churn model.

For KKBox, churn is defined as:
A user whose subscription expired in month M is labeled is_churn = 1 if no new valid transaction appears within 30 days of their expiry date.


IMPORTANT BUSINESS CONTEXT:
We predict churn 14 days before subscription expiry.
So for each user:
- features must only use data up to (membership_expire_date - 14 days)
- anything after that is leakage and must be ignored in training logic
