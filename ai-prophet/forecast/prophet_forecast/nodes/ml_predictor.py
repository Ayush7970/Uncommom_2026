"""Layer 4: ML Predictor — loads category-specific trained model."""

from __future__ import annotations

import logging

from ..state import ForecastState

log = logging.getLogger(__name__)

# Module-level model cache
_sports_model = None


def _get_sports_model():
    global _sports_model
    if _sports_model is None:
        from ..ml.sports_model import SportsMLPredictor
        _sports_model = SportsMLPredictor()
    return _sports_model


def ml_predictor_node(state: ForecastState) -> dict:
    """Run the category-specific ML head and return p_ml."""
    category = state.get("category", "culture")
    p_market = state.get("p_market", 0.5)

    if category == "sports":
        model = _get_sports_model()
        # Extract ml_features from evidence (set by sports researcher)
        features = {}
        for ev in state.get("evidence", []):
            if ev.get("source") == "ml_features" and "ml_features" in ev:
                features = ev["ml_features"]
                break

        p_ml = model.predict(features) if features else p_market
        source = "sports_model" if model.is_available() else "elo_formula"
        log.info("ML predictor [%s]: market=%s  p_ml=%.3f", source, state["market_id"], p_ml)
        return {"p_ml": p_ml}

    # Fallback for other categories — use market price until Phase 3
    log.info("ML predictor (stub): market=%s  category=%s  p_ml=%.3f (=p_market)",
             state["market_id"], category, p_market)
    return {"p_ml": p_market}
