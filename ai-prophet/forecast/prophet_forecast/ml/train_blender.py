"""Step 10: Train the Layer 7 blender (requires offline LLM predictions).

Option A: full blender — requires artifacts/llm_predictions.csv
Option B: partial blender — uses only p_market + p_ml features (no p_llm required)

Run:
    # Option A (preferred):
    python -m prophet_forecast.ml.train_blender --with-llm

    # Option B (fallback, no LLM predictions needed):
    python -m prophet_forecast.ml.train_blender

Reads:  artifacts/training_data.csv
        artifacts/llm_predictions.csv (Option A only)
Writes: artifacts/blender.pkl
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAINING_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")
LLM_CSV = os.path.join(ARTIFACTS_DIR, "llm_predictions.csv")
BLENDER_PATH = os.path.join(ARTIFACTS_DIR, "blender.pkl")


def _safe_logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p)))
    return math.log(p / (1 - p))


def _load_training(with_llm: bool) -> tuple[list, list, list, list, list]:
    """Load feature matrix and metadata for the blender.

    Returns: X, y, p_markets, hours, cats
    hours and cats are passed directly to _calibrator_brier_on so it never
    needs to reverse-engineer them from the blender feature matrix.

    With --with-llm: only rows that have a matching LLM prediction are included
    (typically 150-200 rows).
    """
    rows = {}
    with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("title", ""), row.get("market_name", ""))
            rows[key] = row

    llm_preds: dict = {}
    if with_llm and os.path.exists(LLM_CSV):
        with open(LLM_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("title", ""), row.get("market_name", ""))
                llm_preds[key] = row
        log.info("Loaded %d LLM predictions from %s", len(llm_preds), LLM_CSV)

    X, y, p_markets, hours, cats = [], [], [], [], []
    matched = 0
    for key, row in rows.items():
        try:
            p_market = float(row["p_market"])
            h = float(row["hours_to_close"])
            o = int(row["outcome"])
            cat = row.get("category", "")
            if not (0.0 < p_market < 1.0):
                continue
        except (ValueError, KeyError):
            continue

        lm = _safe_logit(p_market)

        if with_llm:
            if key not in llm_preds:
                continue
            llm_row = llm_preds[key]
            p_llm = float(llm_row.get("p_llm", p_market))
            p_ml = float(llm_row.get("p_ml", p_market))
            trial_var = float(llm_row.get("trial_variance", 0.0))
            ll = _safe_logit(max(1e-6, min(1 - 1e-6, p_llm)))
            lml = _safe_logit(max(1e-6, min(1 - 1e-6, p_ml)))
            feats = [lm, lml, ll, abs(lml - lm), abs(ll - lm), trial_var,
                     h, math.log1p(h)]
            matched += 1
        else:
            feats = [lm, lm, h, math.log1p(h)]

        for kw in ("sports", "finance", "politics", "science"):
            feats.append(1.0 if kw in cat else 0.0)

        X.append(feats)
        y.append(o)
        p_markets.append(p_market)
        hours.append(h)
        cats.append(cat)

    if with_llm:
        log.info("Joined %d rows (LLM predictions matched to training data)", matched)

    return X, y, p_markets, hours, cats


def train(with_llm: bool = False) -> None:
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, train_test_split
    except ImportError:
        log.error("Missing deps: pip install scikit-learn joblib numpy")
        sys.exit(1)

    if not os.path.exists(TRAINING_CSV):
        log.error("Run build_training_data.py first")
        sys.exit(1)

    mode = "Option A (full blender)" if with_llm else "Option B (partial blender)"
    log.info("Training blender — %s", mode)

    X, y, p_markets, hours, cats = _load_training(with_llm)
    n = len(X)

    if n < 50:
        log.error("Only %d rows — not enough to train blender (need ≥50)", n)
        sys.exit(1)

    log.info("Blender training rows: %d", n)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    p_markets = np.array(p_markets, dtype=float)

    # Use stronger regularization for sparse LLM-labeled sets (N < 300)
    C = 0.1 if n < 300 else 0.3
    model = LogisticRegression(C=C, max_iter=500)

    # Use 5-fold CV when N < 300 (single 80/20 split gives ~35 val rows —
    # too few for a reliable gate decision)
    if n < 300:
        log.info("N=%d < 300 — using 5-fold CV for Brier estimate (C=%.1f)", n, C)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_brier_score")
        brier_blender_cv = float(-cv_scores.mean())
        brier_blender_std = float(cv_scores.std())
        log.info("Blender  Brier(5-fold CV)=%.5f ± %.5f", brier_blender_cv, brier_blender_std)

        brier_calibrator_local = _calibrator_brier_on(y, p_markets, hours, cats)
        log.info("Calibrator Brier on same %d rows=%.5f  (global baseline=0.10310)", n, brier_calibrator_local)

        if brier_blender_cv >= brier_calibrator_local:
            log.warning(
                "Blender CV Brier (%.5f) did NOT beat calibrator on these rows (%.5f). "
                "Not saving. Layer 7 will keep using market_calibrator.pkl.",
                brier_blender_cv, brier_calibrator_local,
            )
            return

        model.fit(X, y)
        log.info("Gate passed. Refit on all %d rows.", n)

    else:
        split = int(n * 0.8)
        idx = list(range(n))
        import random; random.seed(42); random.shuffle(idx)
        tr, val = idx[:split], idx[split:]

        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]
        h_val   = [hours[i]    for i in val]
        cat_val = [cats[i]     for i in val]
        pm_val  = p_markets[val]

        model.fit(X_tr, y_tr)
        p_blender_val = model.predict_proba(X_val)[:, 1]
        brier_blender_cv = float(((p_blender_val - y_val) ** 2).mean())

        brier_calibrator_local = _calibrator_brier_on(y_val, pm_val, h_val, cat_val)
        log.info("Blender  Brier=%.5f  Calibrator=%.5f  delta=%+.5f",
                 brier_blender_cv, brier_calibrator_local, brier_blender_cv - brier_calibrator_local)

        if brier_blender_cv >= brier_calibrator_local:
            log.warning("Blender did NOT beat calibrator. Not saving.")
            return

    joblib.dump(model, BLENDER_PATH)
    log.info("Saved blender → %s  (C=%.1f, n=%d, Brier=%.5f)", BLENDER_PATH, C, n, brier_blender_cv)


def _calibrator_brier_on(
    y: "np.ndarray",
    p_markets: "np.ndarray",
    hours: list,
    cats: list,
) -> float:
    """Compute the market calibrator's Brier score on the given rows.

    Uses hours and cats directly — no column-index arithmetic on the blender
    feature matrix, so it works for both 8-feature Option B and 12-feature
    Option A without any changes.

    Falls back to raw p_market Brier if market_calibrator.pkl is missing.
    """
    import numpy as np
    cal_path = os.path.join(ARTIFACTS_DIR, "market_calibrator.pkl")
    if os.path.exists(cal_path):
        try:
            import joblib
            from .feature_extraction import build_calibrator_features
            calibrator = joblib.load(cal_path)
            X_cal = np.array(
                [build_calibrator_features(float(p), float(h), c)
                 for p, h, c in zip(p_markets, hours, cats)],
                dtype=float,
            )
            p_cal = calibrator.predict_proba(X_cal)[:, 1]
            return float(((p_cal - y) ** 2).mean())
        except Exception as exc:
            log.warning("Could not run calibrator for local gate (%s) — using p_market Brier", exc)

    return float(((p_markets - y) ** 2).mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true",
                        help="Use artifacts/llm_predictions.csv (Option A)")
    args = parser.parse_args()
    train(with_llm=args.with_llm)
