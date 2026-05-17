"""Sports ML predictor — logistic regression on Elo features."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from .base import BaseMLPredictor

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_MODEL_PATH = os.path.join(_ARTIFACTS_DIR, "sports_model.pkl")


def elo_win_prob(elo_home: float, elo_away: float) -> float:
    """Theoretical Elo win probability (no features — pure formula)."""
    return 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400))


class SportsMLPredictor(BaseMLPredictor):
    """
    Logistic regression on Elo features.

    Feature vector (in order):
        [elo_diff/400, home_advantage, rest_diff_days, recent_form_diff]

    Falls back to the Elo formula if no trained model is available.
    """

    def __init__(self) -> None:
        self._model = None
        self._meta: dict = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(_MODEL_PATH):
            log.info("Sports model artifact not found at %s — using Elo formula fallback", _MODEL_PATH)
            return
        try:
            import joblib
            bundle = joblib.load(_MODEL_PATH)
            self._model = bundle["model"]
            self._meta  = bundle.get("meta", {})
            log.info(
                "Sports model loaded: train_acc=%.3f  n_train=%d",
                self._meta.get("train_acc", 0),
                self._meta.get("n_train", 0),
            )
        except Exception as e:
            log.warning("Failed to load sports model: %s — using Elo formula", e)

    def is_available(self) -> bool:
        return self._model is not None

    def predict(self, features: dict) -> float:
        """
        features keys (all optional — missing values filled with defaults):
            elo_home        float  Elo rating of home/favoured team
            elo_away        float  Elo rating of away/underdog team
            home_advantage  float  1.0 if home team, 0.0 neutral, -1.0 if away
            rest_diff_days  float  home_rest - away_rest (days since last game)
            form_diff       float  home_last5_winpct - away_last5_winpct
        """
        elo_home = float(features.get("elo_home", 1500))
        elo_away = float(features.get("elo_away", 1500))
        home_adv = float(features.get("home_advantage", 0.0))
        rest_d   = float(features.get("rest_diff_days", 0.0))
        form_d   = float(features.get("form_diff", 0.0))

        elo_diff_norm = (elo_home - elo_away) / 400.0

        if self._model is not None:
            X = np.array([[elo_diff_norm, home_adv, rest_d, form_d]])
            try:
                p = float(self._model.predict_proba(X)[0][1])
                return max(0.01, min(0.99, p))
            except Exception as e:
                log.warning("Model.predict_proba failed: %s — using Elo formula", e)

        # Pure Elo formula fallback (add ~65 Elo points for home advantage)
        effective_elo_home = elo_home + home_adv * 65
        p = elo_win_prob(effective_elo_home, elo_away)
        return max(0.01, min(0.99, p))
