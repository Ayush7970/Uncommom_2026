"""
Layer 7: Calibration + Market Blend.

Logic:
  - Urgent markets (<3h): output p_market (model adds noise not signal)
  - Low confidence (<0.40): output p_market (don't make things worse)
  - Uninformative market (p_market ≈ 0.5, no real price): skip blend, use model directly
  - High confidence (≥0.65): skip Platt compression (preserve strong signal)
  - Otherwise: Platt calibration → anti-conservatism sharpening → w(t) blend
"""

from __future__ import annotations

import logging
import math
import os

from ..state import ForecastState

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")

CONFIDENCE_GATE          = 0.40  # default: below this → output p_market
CONFIDENCE_GATE_CULTURE  = 0.15  # culture: Claude+search has strong entertainment knowledge
CONFIDENCE_GATE_SPORTS   = 0.33  # sports: result search finds league/playoff winners
CONFIDENCE_GATE_POLITICS = 0.35  # politics: result search finds election winners
UNINFORMATIVE_THRESH   = 0.01   # |p_market - 0.5| < this → market is useless anchor
HIGH_CONFIDENCE        = 0.65   # above this → bypass Platt compression
SHARPEN_CONFIDENCE_MIN = 0.65
SHARPEN_EXTREME_LOW    = 0.20
SHARPEN_EXTREME_HIGH   = 0.80
SHARPEN_FACTOR         = 1.30
SHARPEN_CAP            = 0.97


def _load_calibrator():
    path = os.path.join(_ARTIFACTS_DIR, "calibrator.pkl")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        log.warning("Could not load calibrator: %s", e)
        return None


def _sharpen(p: float, confidence: float) -> float:
    """Anti-conservatism: push confident extremes further (§4.2.3)."""
    if confidence < SHARPEN_CONFIDENCE_MIN:
        return p
    if SHARPEN_EXTREME_LOW <= p <= SHARPEN_EXTREME_HIGH:
        return p
    logit_p = math.log(p / (1.0 - p))
    p_sharp = 1.0 / (1.0 + math.exp(-logit_p * SHARPEN_FACTOR))
    return max(1.0 - SHARPEN_CAP, min(SHARPEN_CAP, p_sharp))


def calibrator_node(state: ForecastState) -> dict:
    p_llm_raw  = state.get("p_llm_raw") or state.get("p_market", 0.5)
    p_market   = state.get("p_market", 0.5)
    w_t        = state.get("w_t", 0.30)
    confidence = state.get("confidence") or 0.0
    bucket     = state.get("time_bucket", "long")
    category   = state.get("category", "culture")

    # Category-specific confidence gates
    if category == "culture":
        gate = CONFIDENCE_GATE_CULTURE   # 0.15 — Claude+search knows entertainment well
    elif category == "politics":
        gate = CONFIDENCE_GATE_POLITICS  # 0.35 — result search finds election winners
    elif category == "sports":
        gate = CONFIDENCE_GATE_SPORTS    # 0.33 — result search finds league/playoff winners
    else:
        gate = CONFIDENCE_GATE           # 0.40 — default for finance/science_tech

    # --- Gate 1: urgent bucket or very low confidence → trust market ---
    if bucket == "urgent" or confidence < gate:
        reason = "urgent" if bucket == "urgent" else f"low_conf={confidence:.2f}(gate={gate:.2f}/{category})"
        log.info("Gate [%s]: market=%s → p_final=p_market=%.3f",
                 reason, state["market_id"], p_market)
        return {"p_calibrated": p_market, "p_final": p_market}

    # --- Gate 2: uninformative market price (p_market ≈ 0.5) ---
    # No real price signal — skip the blend and let the model speak at full strength
    uninformative = abs(p_market - 0.5) < UNINFORMATIVE_THRESH

    # --- Platt calibration ---
    # Skip when: (a) highly confident — preserve strong signal, OR
    #            (b) uninformative market — Platt was trained on markets with real prices;
    #                compressing p_llm here just moves toward 0.5 without justification
    if confidence >= HIGH_CONFIDENCE or uninformative:
        p_calibrated = p_llm_raw
    else:
        calibrator = _load_calibrator()
        if calibrator is not None:
            try:
                import numpy as np
                p_calibrated = float(calibrator.predict_proba([[p_llm_raw]])[0][1])
            except Exception as e:
                log.warning("Calibrator failed: %s", e)
                p_calibrated = p_llm_raw
        else:
            p_calibrated = p_llm_raw

    # --- Anti-conservatism sharpening ---
    p_calibrated = _sharpen(p_calibrated, confidence)

    # --- Market blend ---
    if uninformative:
        # p_market=0.5 has no information — blending in 0.5 just shrinks toward random
        p_final = p_calibrated
        log.info(
            "Calibrator [no-blend, uninformative market]: market=%s  "
            "p_llm_raw=%.3f  p_calibrated=%.3f  conf=%.2f  p_final=%.3f",
            state["market_id"], p_llm_raw, p_calibrated, confidence, p_final,
        )
    else:
        p_final = w_t * p_market + (1.0 - w_t) * p_calibrated
        log.info(
            "Calibrator [blend]: market=%s  p_llm_raw=%.3f  p_calibrated=%.3f  "
            "conf=%.2f  w_t=%.2f  p_final=%.3f",
            state["market_id"], p_llm_raw, p_calibrated, confidence, w_t, p_final,
        )

    p_final = max(0.01, min(0.99, p_final))
    return {"p_calibrated": p_calibrated, "p_final": p_final}
