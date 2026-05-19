"""
Abstract repository interface — Interface Segregation Principle.

Routers depend on AbstractPredictionRepository, not on the concrete
PostgreSQL implementation. This makes it trivial to inject an
InMemoryPredictionRepository in tests via app.dependency_overrides.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..schemas import CohortResponse, PredictionRecord


class AbstractPredictionRepository(ABC):

    @abstractmethod
    def get_latest_for_user(self, user_id: str) -> PredictionRecord | None:
        """
        Return the most recent prediction for user_id.
        Returns None if no prediction exists (router maps this to 404).
        Raises RepositoryConnectionError on DB failure.
        """
        ...

    @abstractmethod
    def get_cohort_for_date(self, scoring_date: date) -> CohortResponse:
        """
        Return all predictions for scoring_date, ordered by score DESC.
        Returns CohortResponse with an empty list if the date has no rows.
        Raises RepositoryConnectionError on DB failure.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Execute SELECT 1 with a 2-second timeout.
        Returns True on success.
        Raises RepositoryConnectionError on failure.
        """
        ...
