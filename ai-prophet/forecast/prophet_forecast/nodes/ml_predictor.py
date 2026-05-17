"""Layer 4: ML Predictor — delegates to get_p_ml for all categories."""

from __future__ import annotations

import logging

from ..ml.predict import get_p_ml
from ..state import ForecastState

log = logging.getLogger(__name__)


def ml_predictor_node(state: ForecastState) -> dict:
    """Run the category-specific ML head and return p_ml."""
    p_ml = get_p_ml(state)
    log.info("ML predictor: market=%s  category=%s  p_ml=%.3f",
             state.get("market_id"), state.get("category"), p_ml)
    return {"p_ml": p_ml}
