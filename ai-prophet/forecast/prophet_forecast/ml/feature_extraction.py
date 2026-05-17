"""Shared feature helpers for all ML nodes."""

from __future__ import annotations

import math
from datetime import UTC, datetime


def safe_logit(p: float, eps: float = 1e-6) -> float:
    """logit(p) with clamping to avoid log(0)."""
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))


def safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def hours_to_close(state: dict) -> float:
    """Compute hours between snapshot_ts and resolves_at. Returns 24.0 on error."""
    try:
        snap = _parse_ts(state.get("snapshot_ts", ""))
        close = _parse_ts(state.get("resolves_at", ""))
        if snap and close:
            delta = (close - snap).total_seconds() / 3600.0
            return max(0.0, delta)
    except Exception:
        pass
    return 24.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            pass
    return None


def category_dummies(category: str) -> dict[str, float]:
    """One-hot encoding for the 5 categories. Culture is reference (all zeros)."""
    cats = ("sports", "finance", "politics", "science_tech")
    cat = (category or "").lower()
    return {f"cat_{c}": 1.0 if c in cat else 0.0 for c in cats}


def build_calibrator_features(p_market: float, h: float, category: str) -> list[float]:
    """Feature vector for market calibrator (Phase 1). Must match feature_schema.json."""
    dummies = category_dummies(category)
    return [
        safe_logit(p_market),
        h,
        math.log1p(h),
        dummies["cat_sports"],
        dummies["cat_finance"],
        dummies["cat_politics"],
        dummies["cat_science_tech"],
    ]


CALIBRATOR_FEATURE_NAMES = [
    "logit_p_market",
    "hours_to_close",
    "log1p_hours",
    "cat_sports",
    "cat_finance",
    "cat_politics",
    "cat_science_tech",
]


def build_blender_features(
    p_market: float,
    p_ml: float,
    p_llm: float,
    h: float,
    category: str,
    trial_variance: float = 0.0,
) -> list[float]:
    """Feature vector for Layer 7 blender (Phase 3)."""
    dummies = category_dummies(category)
    lm = safe_logit(p_market)
    ll = safe_logit(p_ml)
    lq = safe_logit(p_llm)
    return [
        lm,
        ll,
        lq,
        abs(ll - lm),
        abs(lq - lm),
        trial_variance,
        h,
        math.log1p(h),
        dummies["cat_sports"],
        dummies["cat_finance"],
        dummies["cat_politics"],
        dummies["cat_science_tech"],
    ]


BLENDER_FEATURE_NAMES = [
    "logit_p_market",
    "logit_p_ml",
    "logit_p_llm",
    "abs_ml_market_disagreement",
    "abs_llm_market_disagreement",
    "trial_variance",
    "hours_to_close",
    "log1p_hours",
    "cat_sports",
    "cat_finance",
    "cat_politics",
    "cat_science_tech",
]
