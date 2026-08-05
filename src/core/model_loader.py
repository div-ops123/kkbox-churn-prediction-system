"""
Model loader — Factory pattern.

Decouples "load from MLflow registry" from "load from local pickle",
so the serving pipeline and tests can swap backends without branching logic.

Public API
----------
    make_model_loader(use_mlflow, ...) -> BaseModelLoader
    model = loader.load()
    version = loader.get_model_version()
    dataset_version_id = loader.get_dataset_version_id()
    get_model_version_dataset_id(mlflow_config, alias) -> str | None  — tag lookup without loading the model
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScoringModel(Protocol):
    """
    Structural protocol — any object that can produce churn scores.
    LightGBM Booster uses predict(); sklearn-style wrappers use predict_proba().
    Both are accepted; score_cohort() handles the dispatch.
    """

    def predict(self, X: Any) -> Any: ...


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseModelLoader(ABC):

    @abstractmethod
    def load(self) -> ScoringModel:
        """Load and return the scoring model. Raises ModelLoadError on failure."""
        ...

    @abstractmethod
    def get_model_version(self) -> str:
        """Return a version string to stamp on every prediction row."""
        ...

    @abstractmethod
    def get_dataset_version_id(self) -> str | None:
        """
        Return the dataset_version_id (dataset-build MLflow run_id) tagged on
        the loaded model version, or None if the loaded model has no such tag
        (e.g. a local dev pickle, or a model registered before this tag existed).
        """
        ...


# ── MLflow loader ─────────────────────────────────────────────────────────────

class MLflowModelLoader(BaseModelLoader):
    """
    Loads the model tagged with a given alias from the MLflow Model Registry.
    get_model_version() returns the MLflow run_id — the canonical identifier
    stamped on every row written to the predictions table.
    """

    def __init__(self, tracking_uri: str, model_name: str, alias: str) -> None:
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._alias = alias
        self._version: str | None = None
        self._dataset_version_id: str | None = None

    def load(self) -> ScoringModel:
        try:
            import mlflow
            import mlflow.lightgbm
            from mlflow.exceptions import MlflowException
        except ImportError as exc:
            raise ModelLoadError("mlflow is not installed") from exc

        mlflow.set_tracking_uri(self._tracking_uri)
        client = mlflow.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(self._model_name, self._alias)
        except Exception as exc:
            raise ModelNotFoundError(
                f"No model '{self._model_name}' with alias '{self._alias}' "
                f"in registry at {self._tracking_uri}"
            ) from exc

        self._version = mv.run_id
        self._dataset_version_id = mv.tags.get("dataset_version_id")
        model_uri = f"models:/{self._model_name}/{mv.version}"
        try:
            model = mlflow.lightgbm.load_model(model_uri)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from {model_uri}: {exc}"
            ) from exc

        if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
            raise ModelLoadError(
                f"Loaded object at {model_uri} has neither predict() nor predict_proba()"
            )
        return model

    def get_model_version(self) -> str:
        if self._version is None:
            raise RuntimeError("Call load() before get_model_version()")
        return self._version

    def get_dataset_version_id(self) -> str | None:
        if self._version is None:
            raise RuntimeError("Call load() before get_dataset_version_id()")
        return self._dataset_version_id


# ── Pickle loader ─────────────────────────────────────────────────────────────

class PickleModelLoader(BaseModelLoader):
    """
    Loads a model from a local pickle file.
    Used for local development and unit tests — no MLflow dependency.
    """

    def __init__(self, pkl_path: Path, version_label: str = "local") -> None:
        self._pkl_path = Path(pkl_path)
        self._version_label = version_label

    def load(self) -> ScoringModel:
        if not self._pkl_path.exists():
            raise ModelLoadError(f"Model file not found: {self._pkl_path}")
        try:
            with open(self._pkl_path, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to unpickle model at {self._pkl_path}: {exc}"
            ) from exc

        if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
            raise ModelLoadError(
                f"Object at {self._pkl_path} has neither predict() nor predict_proba()"
            )
        return model

    def get_model_version(self) -> str:
        return self._version_label

    def get_dataset_version_id(self) -> str | None:
        """Local dev pickles carry no dataset lineage."""
        return None


# ── Standalone lookups ────────────────────────────────────────────────────────

def get_model_version_dataset_id(mlflow_config: Any, alias: str) -> str | None:
    """
    Look up the dataset_version_id tag on whichever model version currently
    holds `alias`, without deserializing the model itself.

    Used by pipelines (e.g. monitor.py's drift check) that need to know which
    dataset produced the current production model but don't need the model
    object loaded into memory.

    Returns None if the tag is missing. Raises ModelNotFoundError if no model
    version currently holds `alias`.
    """
    import mlflow

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(mlflow_config.model_name, alias)
    except Exception as exc:
        raise ModelNotFoundError(
            f"No model '{mlflow_config.model_name}' with alias '{alias}' "
            f"in registry at {mlflow_config.tracking_uri}"
        ) from exc
    return mv.tags.get("dataset_version_id")


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model_loader(
    use_mlflow: bool,
    mlflow_config: Any | None = None,   # MLflowConfig from config.py
    pkl_path: Path | None = None,
    version_label: str = "local",
) -> BaseModelLoader:
    """
    Factory function — callers never instantiate loaders directly.

    Parameters
    ----------
    use_mlflow    : True → MLflowModelLoader; False → PickleModelLoader
    mlflow_config : MLflowConfig, required when use_mlflow=True
    pkl_path      : Path to .pkl file, required when use_mlflow=False
    version_label : Version label for PickleModelLoader
    """
    if use_mlflow:
        if mlflow_config is None:
            raise ValueError("mlflow_config is required when use_mlflow=True")
        return MLflowModelLoader(
            tracking_uri=mlflow_config.tracking_uri,
            model_name=mlflow_config.model_name,
            alias=mlflow_config.model_alias,
        )
    else:
        if pkl_path is None:
            raise ValueError("pkl_path is required when use_mlflow=False")
        return PickleModelLoader(pkl_path=pkl_path, version_label=version_label)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ModelLoadError(Exception):
    """Raised when the model cannot be loaded from the registry or disk."""


class ModelNotFoundError(ModelLoadError):
    """Raised when no model with the requested alias exists in the registry."""
