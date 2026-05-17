"""Layer 7: Calibration + Market Blend."""

from __future__ import annotations

import logging

import numpy as np

from ..ml.calibrate import calibrate
from ..state import ForecastState

log = logging.getLogger(__name__)


def calibrator_node(state: ForecastState) -> dict:
    """Apply calibration then blend with market price via w(t).

    p_final = w_t * p_market + (1 - w_t) * p_calibrated
    """
    p_calibrated = calibrate(state)
    w_t = state.get("w_t", 0.45)
    p_market = state.get("p_market", 0.5)
    p_final = float(np.clip(w_t * p_market + (1 - w_t) * p_calibrated, 0.01, 0.99))

    log.info("Calibrator: market=%s  p_calibrated=%.3f  w_t=%.2f  p_final=%.3f",
             state.get("market_id"), p_calibrated, w_t, p_final)

    return {"p_calibrated": p_calibrated, "p_final": p_final}
