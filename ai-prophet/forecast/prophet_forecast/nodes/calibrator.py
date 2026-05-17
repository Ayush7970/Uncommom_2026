"""Layer 7: Calibration + Market Blend — stub for Phase 1, real impl in Phase 4."""

from __future__ import annotations

import logging
import os

from ..state import ForecastState

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")


def _load_calibrator():
    """Load Platt calibrator if available, else return None (identity fallback)."""
    path = os.path.join(_ARTIFACTS_DIR, "calibrator.pkl")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        log.warning("Could not load calibrator: %s", e)
        return None


def calibrator_node(state: ForecastState) -> dict:
    """
    Apply Platt calibration then blend with market price via w(t).

    p_final = w_t * p_market + (1 - w_t) * p_calibrated

    Phase 1: no calibrator loaded → p_calibrated = p_llm_raw (identity).
    Phase 4: loads fitted Platt scaler from artifacts/calibrator.pkl.
    """
    p_llm_raw = state.get("p_llm_raw") or state.get("p_market", 0.5)
    p_market = state.get("p_market", 0.5)
    w_t = state.get("w_t", 0.45)

    calibrator = _load_calibrator()
    if calibrator is not None:
        try:
            import numpy as np
            p_calibrated = float(calibrator.predict_proba([[p_llm_raw]])[0][1])
        except Exception as e:
            log.warning("Calibrator failed: %s — using raw", e)
            p_calibrated = p_llm_raw
    else:
        p_calibrated = p_llm_raw  # identity in Phase 1

    p_final = w_t * p_market + (1.0 - w_t) * p_calibrated
    p_final = max(0.01, min(0.99, p_final))

    log.info("Calibrator: market=%s  p_llm_raw=%.3f  p_calibrated=%.3f  w_t=%.2f  p_final=%.3f",
             state["market_id"], p_llm_raw, p_calibrated, w_t, p_final)

    return {
        "p_calibrated": p_calibrated,
        "p_final": p_final,
    }
