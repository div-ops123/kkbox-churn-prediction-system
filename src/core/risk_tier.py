"""
Risk tier classification — Strategy pattern.

The pipeline calls make_tier_strategy(tier_config).assign(scores).
Swapping from fixed thresholds to percentile-based tiers requires only
a new subclass — no changes to serve.py.

Public API
----------
    make_tier_strategy(tier_config: dict) -> BaseTierStrategy
    tiers = strategy.assign(scores)   # list[Literal["HIGH","MED","LOW"]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

RiskTier = Literal["HIGH", "MED", "LOW"]


# ── Abstract strategy ─────────────────────────────────────────────────────────

class BaseTierStrategy(ABC):

    @abstractmethod
    def assign(self, scores: np.ndarray) -> list[RiskTier]:
        """
        Map each score to a risk tier.

        Parameters
        ----------
        scores : 1-D float array with values in [0.0, 1.0]

        Returns
        -------
        list[RiskTier] of the same length, one tier per score.

        Raises ValueError  if any score is outside [0.0, 1.0].
        Raises TierAssignmentError  if any score cannot be mapped (config gap).
        """
        ...

    @abstractmethod
    def describe(self) -> dict:
        """Return a loggable dict summarising the strategy and thresholds."""
        ...


# ── Fixed-threshold strategy ──────────────────────────────────────────────────

class FixedThresholdStrategy(BaseTierStrategy):
    """
    Assigns tiers based on fixed score thresholds from risk_tiers_config.json.

    Tiers are evaluated in the order they appear in the config. The first
    interval that contains a score is assigned. A fully-validated config (via
    load_risk_tiers_config) guarantees every score in [0,1] maps to exactly one
    tier, so TierAssignmentError should never occur in practice.
    """

    def __init__(self, tier_config: dict) -> None:
        # Build sorted list of (lo, hi, inclusive_lo, inclusive_hi, name).
        self._tiers = [
            (
                float(t["score_min"]),
                float(t["score_max"]),
                bool(t["inclusive_min"]),
                bool(t["inclusive_max"]),
                str(t["name"]),
            )
            for t in tier_config["tiers"]
        ]
        self._strategy_name = tier_config.get("strategy", "fixed_threshold")
        self._version = tier_config.get("version", "unknown")

    def assign(self, scores: np.ndarray) -> list[RiskTier]:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            bad = scores[(scores < 0.0) | (scores > 1.0)]
            raise ValueError(f"All scores must be in [0.0, 1.0]. Found out-of-range: {bad[:5]}")

        result: list[RiskTier] = []
        for score in scores:
            assigned = None
            for lo, hi, inc_lo, inc_hi, name in self._tiers:
                lo_ok = (score >= lo) if inc_lo else (score > lo)
                hi_ok = (score <= hi) if inc_hi else (score < hi)
                if lo_ok and hi_ok:
                    assigned = name
                    break
            if assigned is None:
                raise TierAssignmentError(
                    f"Score {score} did not match any tier interval. "
                    "Check risk_tiers_config.json for a gap."
                )
            result.append(assigned)  # type: ignore[arg-type]
        return result

    def describe(self) -> dict:
        return {
            "strategy": self._strategy_name,
            "version": self._version,
            "tiers": [
                {"name": name, "score_min": lo, "score_max": hi}
                for lo, hi, _, _, name in self._tiers
            ],
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def make_tier_strategy(tier_config: dict) -> BaseTierStrategy:
    """
    Factory for tier strategies. Reads tier_config["strategy"] to dispatch.
    Currently supports: "fixed_threshold".
    Raises ValueError for unknown strategy names.
    """
    strategy = tier_config.get("strategy", "fixed_threshold")
    if strategy == "fixed_threshold":
        return FixedThresholdStrategy(tier_config)
    raise ValueError(
        f"Unknown tier strategy '{strategy}'. "
        "Supported strategies: 'fixed_threshold'"
    )


# ── Exception ─────────────────────────────────────────────────────────────────

class TierAssignmentError(Exception):
    """Raised when a score cannot be mapped to any tier (indicates a config gap)."""
