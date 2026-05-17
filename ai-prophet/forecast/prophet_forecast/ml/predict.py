"""Layer 4 LangGraph node: domain-specific ML head → p_ml.

The LangGraph graph calls get_p_ml(state) and writes state["p_ml"].
This function NEVER raises — worst case it returns np.clip(p_market, 0.01, 0.99).
"""

from __future__ import annotations

import logging
import math
import os
import re

import numpy as np

from .feature_extraction import (
    build_calibrator_features,
    hours_to_close,
    safe_logit,
)

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Lazily loaded models — populated once on first call
_calibrator = None
_sports_model = None
_sports_elo_model = None   # SportsMLPredictor (4-feature Elo model)
_culture_model = None
_faiss_index = None
_faiss_meta = None  # {"questions": [...], "outcomes": [...]}
_models_loaded = False


def _load_models() -> None:
    global _calibrator, _sports_model, _sports_elo_model, _culture_model, _faiss_index, _faiss_meta, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    try:
        import joblib

        cal_path = os.path.join(ARTIFACTS_DIR, "market_calibrator.pkl")
        if os.path.exists(cal_path):
            _calibrator = joblib.load(cal_path)
            log.info("Loaded market_calibrator.pkl")
        else:
            log.warning("market_calibrator.pkl not found — using p_market as baseline")

        sports_path = os.path.join(ARTIFACTS_DIR, "sports_logistic.pkl")
        if os.path.exists(sports_path):
            _sports_model = joblib.load(sports_path)
            log.info("Loaded sports_logistic.pkl")

        culture_path = os.path.join(ARTIFACTS_DIR, "culture_logistic.pkl")
        if os.path.exists(culture_path):
            _culture_model = joblib.load(culture_path)
            log.info("Loaded culture_logistic.pkl")

        faiss_path = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
        meta_path = os.path.join(ARTIFACTS_DIR, "faiss_metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            import faiss  # type: ignore
            _faiss_index = faiss.read_index(faiss_path)
            _faiss_meta = joblib.load(meta_path)
            log.info("Loaded FAISS science_tech index (%d vectors)", _faiss_index.ntotal)

    except Exception as exc:
        log.warning("Model loading error (non-fatal): %s", exc)

    try:
        from .sports_model import SportsMLPredictor
        _sports_elo_model = SportsMLPredictor()
        if _sports_elo_model.is_available():
            log.info("Loaded SportsMLPredictor (sports_model.pkl, 4-feature Elo)")
        else:
            log.info("sports_model.pkl absent — SportsMLPredictor will use Elo formula fallback")
    except Exception as exc:
        log.warning("SportsMLPredictor init failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_p_ml(state: dict) -> float:
    """Layer 4 node entry point. Returns p_ml in [0.01, 0.99]."""
    _load_models()
    try:
        return _route(state)
    except Exception as exc:
        log.warning("get_p_ml failed (%s), returning p_market", exc)
        return float(np.clip(state.get("p_market", 0.5), 0.01, 0.99))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route(state: dict) -> float:
    category = (state.get("category") or "").lower()
    p_market = float(np.clip(state.get("p_market", 0.5), 0.01, 0.99))
    question = state.get("question", "")
    h = hours_to_close(state)

    if "sports" in category:
        return _sports(question, p_market, h, state)
    if "finance" in category or "economics" in category:
        return _finance(question, p_market, h)
    if "politic" in category:
        return _politics(question, p_market, h)
    if "science" in category or "tech" in category:
        return _science(question, p_market)
    # culture, entertainment, other → use calibrator
    return _apply_calibrator(p_market, h, category)


# ---------------------------------------------------------------------------
# Sports: logistic regression on Elo
# ---------------------------------------------------------------------------

def _sports(question: str, p_market: float, h: float, state: dict | None = None) -> float:
    # Layer 3→4 handoff: use Elo features pre-computed by research_sports if available
    if state is not None and _sports_elo_model is not None:
        for ev in (state.get("evidence") or []):
            if ev.get("source") == "ml_features" and ev.get("ml_features"):
                try:
                    return _sports_elo_model.predict(ev["ml_features"])
                except Exception as exc:
                    log.warning("SportsMLPredictor.predict failed: %s — falling back", exc)
                break  # only try the first ml_features entry

    from .external.elo import fetch_elo_diff

    elo_diff = fetch_elo_diff(question)  # returns float or None

    if _sports_model is not None:
        n_features = getattr(_sports_model, "n_features_in_", None)
        if n_features == 2:
            # Current sports_logistic.pkl is a market-correction model trained on
            # [logit(p_market), log1p(hours_to_close)]. Do not feed Elo-shaped
            # features into it; that turns successful Elo lookup into a fallback.
            X = [safe_logit(p_market), math.log1p(h)]
            p = float(_sports_model.predict_proba([X])[0][1])
            return float(np.clip(p, 0.01, 0.99))
        if n_features == 4 and elo_diff is not None:
            is_home = _detect_home(question)
            X = [
                elo_diff / 400.0,
                float(is_home),
                0.0,  # rest_diff placeholder — set to 0 if not available
                0.0,  # injury proxy placeholder
            ]
            p = float(_sports_model.predict_proba([X])[0][1])
            return float(np.clip(p, 0.01, 0.99))

    return _apply_calibrator(p_market, h, "sports")


def _detect_home(question: str) -> bool:
    """Heuristic: 'X at Y' → Y is home. Returns True if the outcome team is home."""
    m = re.search(r"\bat\s+([A-Z][a-zA-Z ]+)", question)
    return m is not None


# ---------------------------------------------------------------------------
# Finance: Black-Scholes analytical
# ---------------------------------------------------------------------------

def _finance(question: str, p_market: float, h: float) -> float:
    from .external.fred import fetch_threshold_data

    result = fetch_threshold_data(question)
    if result is None:
        return _apply_calibrator(p_market, h, "finance")

    current, threshold, volatility = result
    if current >= threshold and _is_barrier_threshold_question(question):
        return 0.99

    days = h / 24.0
    p = _black_scholes_p(current, threshold, volatility, days)
    if p is None:
        return _apply_calibrator(p_market, h, "finance")
    return float(np.clip(p, 0.01, 0.99))


def _is_barrier_threshold_question(question: str) -> bool:
    """Return True when reaching the threshold settles the event as yes."""
    return bool(
        re.search(
            r"\b(?:exceed(?:s|ed)?|surpass(?:es|ed)?|hit(?:s)?|reach(?:es|ed)?|"
            r"touch(?:es|ed)?|break(?:s)?|cross(?:es|ed)?)\b",
            question,
            re.I,
        )
    )


def _black_scholes_p(current: float, threshold: float,
                     volatility: float, days: float) -> float | None:
    """Log-normal Black-Scholes probability: P(S_T > K).

    Uses N(d2) from standard BS, which treats current/threshold as price levels
    and volatility as annual fractional vol (e.g. 0.80 for 80% per year).
    Correct for crypto, equities, and any price-threshold question.
    Falls back to calibrator if inputs are degenerate.
    """
    from scipy.stats import norm  # type: ignore
    T = days / 365.0
    if T <= 0 or volatility <= 0 or current <= 0 or threshold <= 0:
        return float(current > threshold)
    # d2 = (ln(S/K) - 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = (math.log(current / threshold) - 0.5 * volatility**2 * T) / (volatility * math.sqrt(T))
    return float(norm.cdf(d2))


def _parse_threshold(question: str) -> float | None:
    m = re.search(
        r"(?:above|exceed|over|greater than|more than)\s*\$?([\d,]+\.?\d*)",
        question, re.I,
    )
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"\$\s*([\d,]+\.?\d*)", question)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


# ---------------------------------------------------------------------------
# Politics: Beta-Binomial Bayesian
# ---------------------------------------------------------------------------

def _politics(question: str, p_market: float, h: float) -> float:
    is_incumbent = _detect_incumbent(question)
    days = h / 24.0
    alpha, beta_p = (7.0, 3.0) if is_incumbent else (3.0, 7.0)
    n = max(1.0, 30.0 - days / 10.0)
    alpha_post = alpha + n * p_market
    beta_post = beta_p + n * (1 - p_market)
    p = alpha_post / (alpha_post + beta_post)
    return float(np.clip(p, 0.01, 0.99))


def _detect_incumbent(question: str) -> bool:
    keywords = ("incumbent", "re-elect", "reelect", "current president",
                "sitting president", "current prime minister", "defend")
    q = question.lower()
    return any(kw in q for kw in keywords)


# ---------------------------------------------------------------------------
# Science/Tech: k-NN on embeddings
# ---------------------------------------------------------------------------

def _science(question: str, p_market: float) -> float:
    if _faiss_index is None or _faiss_meta is None:
        return _apply_calibrator(p_market, 24.0, "science_tech")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = _get_embedding_model()
        vec = _model.encode([question], normalize_embeddings=True).astype("float32")
        K = min(5, _faiss_index.ntotal)
        distances, indices = _faiss_index.search(vec, K)
        outcomes = [_faiss_meta["outcomes"][i] for i in indices[0] if i >= 0]
        if not outcomes:
            return _apply_calibrator(p_market, 24.0, "science_tech")
        p = sum(outcomes) / len(outcomes)
        return float(np.clip(p, 0.01, 0.99))
    except Exception as exc:
        log.warning("FAISS k-NN failed (%s), using calibrator", exc)
        return _apply_calibrator(p_market, 24.0, "science_tech")


_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Fallback: market calibrator
# ---------------------------------------------------------------------------

def _apply_calibrator(p_market: float, h: float, category: str) -> float:
    if _calibrator is None:
        return float(np.clip(p_market, 0.01, 0.99))
    X = [build_calibrator_features(p_market, h, category)]
    p = float(_calibrator.predict_proba(X)[0][1])
    return float(np.clip(p, 0.01, 0.99))
