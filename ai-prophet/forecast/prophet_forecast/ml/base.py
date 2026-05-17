"""Base class for all ML predictor heads."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMLPredictor(ABC):
    """Every category ML head must implement predict()."""

    @abstractmethod
    def predict(self, features: dict) -> float:
        """Return p_yes ∈ [0, 1]."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if a trained model artifact is loaded and ready."""
        ...
