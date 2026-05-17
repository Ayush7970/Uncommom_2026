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

    if category == "finance":
        # Mean-reversion prior: extreme market prices tend to be overconfident
        # Markets priced >0.80 or <0.20 are pulled 8% toward 0.50
        if p_market > 0.80:
            p_ml = p_market - 0.08
        elif p_market < 0.20:
            p_ml = p_market + 0.08
        else:
            p_ml = p_market
        log.info("ML predictor [finance/mean-reversion]: market=%s  p_ml=%.3f", state["market_id"], p_ml)
        return {"p_ml": p_ml}

    if category == "politics":
        # Base rate prior: incumbents and status-quo outcomes win ~55% of the time
        # Blend market price 70% with a 0.45 base rate 30%
        p_ml = 0.70 * p_market + 0.30 * 0.45
        log.info("ML predictor [politics/base-rate]: market=%s  p_ml=%.3f", state["market_id"], p_ml)
        return {"p_ml": p_ml}

    # science_tech and culture — no prior, use market price
    log.info("ML predictor (no prior): market=%s  category=%s  p_ml=%.3f",
             state["market_id"], category, p_market)
    return {"p_ml": p_market}
