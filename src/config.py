"""
Centralized configuration for the serving pipeline, Score API, and labeling pipeline.

All values are read from environment variables. A .env file is supported
for local development (loaded via python-dotenv if present).

Usage
-----
    from config import load_serving_config, load_api_config, load_labeling_config
    cfg = load_serving_config()
    print(cfg.postgres.conn_str)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent (e.g. inside a container)

BASE_DIR = Path(__file__).resolve().parent.parent


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @property
    def conn_str(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    model_name: str
    model_alias: str


@dataclass(frozen=True)
class ServingConfig:
    postgres: PostgresConfig
    mlflow: MLflowConfig
    feature_config_path: Path
    risk_tiers_config_path: Path
    predictions_parquet_dir: Path
    log_level: str


@dataclass(frozen=True)
class ApiConfig:
    postgres: PostgresConfig
    pool_min_conn: int
    pool_max_conn: int
    log_level: str


@dataclass(frozen=True)
class LabelingConfig:
    postgres: PostgresConfig
    cohort_month: str               # "YYYY-MM"
    label_source: str               # "train_csv" | "transactions"
    data_source: str                # "csv" | "postgres"
    data_dir: Path
    train_csv_path: Path
    renewal_window_days: int
    feature_cutoff_lag_days: int
    labeled_date: date
    log_level: str


# ── Loaders ───────────────────────────────────────────────────────────────────

def _require_postgres() -> PostgresConfig:
    missing = []
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        missing.append("POSTGRES_PASSWORD")
    if missing:
        raise EnvironmentError(
            f"Required environment variables not set: {', '.join(missing)}"
        )
    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "kkbox"),
        user=os.getenv("POSTGRES_USER", "kkbox"),
        password=password,
    )


def load_serving_config() -> ServingConfig:
    """
    Read all environment variables and return a frozen ServingConfig.
    Raises EnvironmentError if POSTGRES_PASSWORD is not set.
    """
    pg = _require_postgres()
    mlflow_cfg = MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        model_name=os.getenv("MLFLOW_MODEL_NAME", "LightGBMChurnClassifier"),
        model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "production"),
    )
    return ServingConfig(
        postgres=pg,
        mlflow=mlflow_cfg,
        feature_config_path=BASE_DIR / os.getenv("FEATURE_CONFIG_PATH", "models/feature_config.json"),
        risk_tiers_config_path=BASE_DIR / os.getenv("RISK_TIERS_CONFIG_PATH", "models/risk_tiers_config.json"),
        predictions_parquet_dir=BASE_DIR / os.getenv("PREDICTIONS_PARQUET_DIR", "data"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_api_config() -> ApiConfig:
    """Read environment variables for the API process."""
    pg = _require_postgres()
    return ApiConfig(
        postgres=pg,
        pool_min_conn=int(os.getenv("API_POOL_MIN_CONN", "2")),
        pool_max_conn=int(os.getenv("API_POOL_MAX_CONN", "10")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_feature_config(path: Path) -> dict[str, float]:
    """
    Load models/feature_config.json.
    Returns {"p99_secs": 43805.516}

    Raises FileNotFoundError, KeyError, ValueError.
    """
    if not path.exists():
        raise FileNotFoundError(f"feature_config.json not found at {path}")
    with open(path) as f:
        data = json.load(f)
    if "p99_secs" not in data:
        raise KeyError("feature_config.json is missing the 'p99_secs' key")
    p99 = data["p99_secs"]
    if not isinstance(p99, (int, float)) or p99 <= 0:
        raise ValueError(f"p99_secs must be a positive number, got {p99!r}")
    return data


def load_labeling_config(
    cohort_month: str,
    label_source: str = "train_csv",
    data_source: str = "postgres",
    data_dir: str = "data/",
) -> LabelingConfig:
    """
    Read environment variables and return a frozen LabelingConfig.

    Parameters
    ----------
    cohort_month : str  — "YYYY-MM" target cohort month (e.g. "2017-03")
    label_source : str  — "train_csv" uses official train.csv / train_labels table;
                          "transactions" derives labels from renewal patterns.
    data_source  : str  — "postgres" reads from PostgreSQL via DuckDB extension;
                          "csv" reads from CSV files (dev / testing only).
    data_dir     : str  — relative path to the CSV directory.

    Raises EnvironmentError if POSTGRES_PASSWORD is not set.
    Raises ValueError for invalid cohort_month format or unknown source values.
    """
    if not re.match(r"^\d{4}-\d{2}$", cohort_month):
        raise ValueError(f"cohort_month must be 'YYYY-MM', got {cohort_month!r}")
    if label_source not in ("train_csv", "transactions"):
        raise ValueError(
            f"label_source must be 'train_csv' or 'transactions', got {label_source!r}"
        )
    if data_source not in ("csv", "postgres"):
        raise ValueError(
            f"data_source must be 'csv' or 'postgres', got {data_source!r}"
        )

    renewal_days = int(os.getenv("RENEWAL_WINDOW_DAYS", "30"))
    if renewal_days <= 0:
        raise ValueError(f"RENEWAL_WINDOW_DAYS must be positive, got {renewal_days}")
    cutoff_lag = int(os.getenv("FEATURE_CUTOFF_LAG_DAYS", "14"))
    if cutoff_lag <= 0:
        raise ValueError(f"FEATURE_CUTOFF_LAG_DAYS must be positive, got {cutoff_lag}")

    pg = _require_postgres()
    return LabelingConfig(
        postgres=pg,
        cohort_month=cohort_month,
        label_source=label_source,
        data_source=data_source,
        data_dir=BASE_DIR / data_dir,
        train_csv_path=BASE_DIR / os.getenv("TRAIN_CSV_PATH", "data/train.csv"),
        renewal_window_days=renewal_days,
        feature_cutoff_lag_days=cutoff_lag,
        labeled_date=date.today(),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_risk_tiers_config(path: Path) -> dict:
    """
    Load and validate models/risk_tiers_config.json.

    Validation rules:
      - "tiers" must be a list of exactly 3 dicts
      - Each dict has: name (str in HIGH/MED/LOW), score_min/score_max (float),
        inclusive_min/inclusive_max (bool)
      - All scores in [0.0, 1.0] with score_min < score_max
      - Union of intervals covers [0.0, 1.0] with no gap and no overlap

    Raises FileNotFoundError, json.JSONDecodeError, ValueError.
    """
    if not path.exists():
        raise FileNotFoundError(f"risk_tiers_config.json not found at {path}")
    with open(path) as f:
        data = json.load(f)

    tiers = data.get("tiers")
    if not isinstance(tiers, list) or len(tiers) != 3:
        raise ValueError("risk_tiers_config.json must contain exactly 3 tiers")

    valid_names = {"HIGH", "MED", "LOW"}
    seen_names = set()
    for t in tiers:
        name = t.get("name")
        if name not in valid_names:
            raise ValueError(f"Invalid tier name {name!r}. Must be one of {valid_names}")
        if name in seen_names:
            raise ValueError(f"Duplicate tier name: {name!r}")
        seen_names.add(name)

        lo, hi = t.get("score_min"), t.get("score_max")
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise ValueError(f"Tier {name}: score_min and score_max must be numbers")
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"Tier {name}: scores must be in [0.0, 1.0], got [{lo}, {hi}]")
        if lo >= hi:
            raise ValueError(f"Tier {name}: score_min ({lo}) must be < score_max ({hi})")

    # Check that intervals collectively cover [0.0, 1.0] without gap or overlap.
    # Build a sorted list of (min, max, inclusive_min, inclusive_max) and sweep.
    intervals = sorted(
        [(t["score_min"], t["score_max"], t["inclusive_min"], t["inclusive_max"]) for t in tiers],
        key=lambda x: x[0],
    )
    # Verify the lowest interval starts at 0 and the highest ends at 1.
    if intervals[0][0] != 0.0:
        raise ValueError("Tiers must start at score_min=0.0")
    if intervals[-1][1] != 1.0:
        raise ValueError("Tiers must end at score_max=1.0")
    # Each interval's max must equal the next interval's min (no gap, no overlap).
    for i in range(len(intervals) - 1):
        cur_max = intervals[i][1]
        nxt_min = intervals[i + 1][0]
        if cur_max != nxt_min:
            raise ValueError(
                f"Tier intervals do not form a contiguous cover of [0,1]. "
                f"Gap/overlap between {cur_max} and {nxt_min}"
            )

    return data
