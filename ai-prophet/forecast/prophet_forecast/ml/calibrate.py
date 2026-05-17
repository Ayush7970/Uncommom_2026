"""Layer 7 LangGraph node: blender + calibration → p_calibrated.

The graph calls calibrate(state) and writes state["p_calibrated"].
The w(t) market blend is applied by the graph node wrapper (not here):
    p_final = w_t * p_market + (1 - w_t) * p_calibrated

This function NEVER raises.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .feature_extraction import (
    build_blender_features,
    build_calibrator_features,
    hours_to_close,
    safe_logit,
    safe_sigmoid,
)

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

_blender = None
_calibrator = None
_models_loaded = False


def _load_models() -> None:
    global _blender, _calibrator, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    try:
        import joblib

        cal_path = os.path.join(ARTIFACTS_DIR, "market_calibrator.pkl")
        if os.path.exists(cal_path):
            _calibrator = joblib.load(cal_path)
            log.info("calibrate.py: loaded market_calibrator.pkl")

        blender_path = os.path.join(ARTIFACTS_DIR, "blender.pkl")
        if os.path.exists(blender_path):
            _blender = joblib.load(blender_path)
            log.info("calibrate.py: loaded blender.pkl")
    except Exception as exc:
        log.warning("calibrate.py model load error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def calibrate(state: dict) -> float:
    """Layer 7 node entry point. Returns p_calibrated in [0.01, 0.99]."""
    _load_models()
    try:
        return _blend(state)
    except Exception as exc:
        log.warning("calibrate() failed (%s), returning p_market", exc)
        return float(np.clip(state.get("p_market", 0.5), 0.01, 0.99))


# ---------------------------------------------------------------------------
# Internal blend logic
# ---------------------------------------------------------------------------

def _blend(state: dict) -> float:
    p_market = float(np.clip(state.get("p_market", 0.5), 0.01, 0.99))
    p_ml = float(np.clip(state.get("p_ml") or p_market, 0.01, 0.99))
    p_llm_raw = state.get("p_llm_raw")
    h = hours_to_close(state)
    category = (state.get("category") or "").lower()

    # Use trial variance if available (LLM uncertainty signal)
    iterations = state.get("p_iterations") or []
    if len(iterations) >= 2:
        trial_variance = float(np.var(iterations))
    else:
        trial_variance = 0.0

    # Path 1: Full blender (requires p_llm and trained blender.pkl)
    if _blender is not None and p_llm_raw is not None:
        p_llm = float(np.clip(p_llm_raw, 0.01, 0.99))
        X = [build_blender_features(p_market, p_ml, p_llm, h, category, trial_variance)]
        p = float(_blender.predict_proba(X)[0][1])
        log.debug("blend path=blender  p_final=%.3f", p)
        return float(np.clip(p, 0.01, 0.99))

    # Path 2: Market calibrator only (p_llm not available or blender not trained)
    if _calibrator is not None:
        X = [build_calibrator_features(p_market, h, category)]
        p = float(_calibrator.predict_proba(X)[0][1])
        log.debug("blend path=calibrator  p_final=%.3f", p)
        return float(np.clip(p, 0.01, 0.99))

    # Path 3: Logit-space average of whatever signals we have
    signals = [safe_logit(p_market), safe_logit(p_ml)]
    if p_llm_raw is not None:
        signals.append(safe_logit(float(np.clip(p_llm_raw, 0.01, 0.99))))
    logit_avg = sum(signals) / len(signals)
    p = safe_sigmoid(logit_avg)
    log.debug("blend path=logit_avg  p_final=%.3f", p)
    return float(np.clip(p, 0.01, 0.99))
