"""
One-time seed: load CSVs from data/ into PostgreSQL and create indexes.

Prerequisites:
  1. docker compose up -d   (wait for both containers to be healthy)
  2. cp .env.example .env   (or export env vars manually)

Usage:
  python db/seed.py

Idempotent — re-running skips any table that already has rows.
Expected total time: 5-10 minutes (user_logs.csv is 1.4 GB).
"""

import os
import sys
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA     = BASE_DIR / "data"

CONN_PARAMS = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB",   "kkbox"),
    "user":     os.getenv("POSTGRES_USER", "kkbox"),
    "password": os.getenv("POSTGRES_PASSWORD", "kkbox"),
}

# Load order: smaller files first so a failure is cheaper to diagnose.
TABLES = [
    ("members",      "members.csv"),
    ("transactions", "transactions.csv"),
    ("user_logs",    "user_logs.csv"),
    ("train_labels", "train.csv"),
]

# Created AFTER all tables are loaded: building from scratch is 3-5× faster
# than maintaining indexes during COPY for large tables.
INDEXES = [
    # transactions — serving query filters on membership_expire_date
    ("idx_txn_msno",
     "CREATE INDEX IF NOT EXISTS idx_txn_msno ON transactions(msno)"),
    ("idx_txn_expire",
     "CREATE INDEX IF NOT EXISTS idx_txn_expire ON transactions(membership_expire_date)"),
    ("idx_txn_expire_msno",
     "CREATE INDEX IF NOT EXISTS idx_txn_expire_msno ON transactions(membership_expire_date, msno)"),
    # user_logs — feature module queries by msno + date range
    ("idx_logs_msno",
     "CREATE INDEX IF NOT EXISTS idx_logs_msno ON user_logs(msno)"),
    ("idx_logs_date",
     "CREATE INDEX IF NOT EXISTS idx_logs_date ON user_logs(date)"),
    # members — single-column lookup
    ("idx_members_msno",
     "CREATE INDEX IF NOT EXISTS idx_members_msno ON members(msno)"),
    # train_labels
    ("idx_train_msno",
     "CREATE INDEX IF NOT EXISTS idx_train_msno ON train_labels(msno)"),
    # prediction store — Score API queries by msno; monitoring queries by scoring_date
    ("idx_pred_msno",
     "CREATE INDEX IF NOT EXISTS idx_pred_msno ON predictions(msno)"),
    ("idx_pred_scoring_date",
     "CREATE INDEX IF NOT EXISTS idx_pred_scoring_date ON predictions(scoring_date)"),
    # label store
    ("idx_labels_labeled_date",
     "CREATE INDEX IF NOT EXISTS idx_labels_labeled_date ON labels(labeled_date)"),
    # monitoring metrics
    ("idx_mon_run_date",
     "CREATE INDEX IF NOT EXISTS idx_mon_run_date ON monitoring_metrics(run_date)"),
    ("idx_mon_pipeline",
     "CREATE INDEX IF NOT EXISTS idx_mon_pipeline ON monitoring_metrics(pipeline_type, run_date)"),
]

SEP = "─" * 60


def _copy_table(cur, table: str, csv_path: Path) -> int:
    size_mb = csv_path.stat().st_size / 1_048_576
    print(f"  COPY {table:<15} ← {csv_path.name}  ({size_mb:.0f} MB) ...",
          end=" ", flush=True)
    t0 = time.time()
    with open(csv_path, encoding="utf-8") as f:
        cur.copy_expert(
            f"COPY {table} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')",
            f,
        )
    print(f"{time.time() - t0:.1f}s")
    return cur.rowcount


def _create_indexes(cur) -> None:
    print(f"\n{SEP}")
    print("Creating indexes (this takes ~2-5 min for user_logs) ...")
    print(SEP)
    for name, sql in INDEXES:
        print(f"  {name} ...", end=" ", flush=True)
        t0 = time.time()
        cur.execute(sql)
        print(f"{time.time() - t0:.1f}s")


def main() -> None:
    print(f"\n{SEP}")
    print("KKBox seed — connecting to PostgreSQL ...")
    print(SEP)

    try:
        conn = psycopg2.connect(**CONN_PARAMS)
    except psycopg2.OperationalError as e:
        print(f"\nERROR: could not connect — {e}")
        print("Make sure `docker compose up -d` is running and healthy.")
        sys.exit(1)

    # autocommit=True: each COPY and CREATE INDEX commits immediately.
    # Avoids holding a multi-GB transaction open during the bulk load.
    conn.autocommit = True
    cur = conn.cursor()

    print(f"Connected → {CONN_PARAMS['host']}:{CONN_PARAMS['port']}"
          f"/{CONN_PARAMS['dbname']}\n")

    # Increase sort memory for index builds.
    cur.execute("SET maintenance_work_mem = '512MB'")

    print(f"{SEP}")
    print("Loading source tables ...")
    print(SEP)

    for table, csv_file in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        existing = cur.fetchone()[0]
        if existing > 0:
            print(f"  {table:<15} already has {existing:>12,} rows — skipping")
            continue

        csv_path = DATA / csv_file
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping {table}")
            continue

        rows = _copy_table(cur, table, csv_path)
        print(f"    → {rows:>12,} rows committed")

    _create_indexes(cur)

    print(f"\n{SEP}")
    print("Final row counts")
    print(SEP)
    for table, _ in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table:<15} {cur.fetchone()[0]:>12,}")

    cur.close()
    conn.close()
    print(f"\nSeed complete.\n")


if __name__ == "__main__":
    main()
