"""Step 2: Train the Phase 1 market calibrator (universal logistic regression).

Run:
    python -m prophet_forecast.ml.train_market_calibrator

Reads:  artifacts/training_data.csv
Writes: artifacts/market_calibrator.pkl
        artifacts/feature_schema.json

Brier gate: if the trained model does NOT beat raw market on validation, it is
discarded and a warning is logged. The system falls back to p_market in that case.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAINING_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "market_calibrator.pkl")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_schema.json")


def brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def load_csv(path: str) -> tuple[list[list[float]], list[int]]:
    """Load training CSV into feature matrix and labels."""
    from .feature_extraction import build_calibrator_features

    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["p_market"])
                h = float(row["hours_to_close"])
                o = int(row["outcome"])
                cat = row.get("category", "")
                if not (0.0 < p < 1.0):
                    continue
                X.append(build_calibrator_features(p, h, cat))
                y.append(o)
            except (ValueError, KeyError):
                continue
    return X, y


def train(training_csv: str = TRAINING_CSV) -> None:
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.error("Missing deps: pip install scikit-learn joblib numpy")
        sys.exit(1)

    if not os.path.exists(training_csv):
        log.error("Training CSV not found: %s — run build_training_data.py first", training_csv)
        sys.exit(1)

    X, y = load_csv(training_csv)
    if len(X) < 50:
        log.error("Only %d usable rows — not enough to train. Check build_training_data.py output.", len(X))
        sys.exit(1)

    log.info("Loaded %d training rows", len(X))
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Try multiple configs; pick the one that beats market on val
    best_model, best_brier = None, float("inf")
    for C, cw in [(0.5, None), (1.0, None), (2.0, None), (0.5, "balanced")]:
        m = LogisticRegression(C=C, class_weight=cw, max_iter=500)
        m.fit(X_tr, y_tr)
        b = brier(m.predict_proba(X_val)[:, 1].tolist(), y_val.tolist())
        log.info("  C=%-4s  class_weight=%-10s  Brier=%.5f", C, str(cw), b)
        if b < best_brier:
            best_brier = b
            best_model = m
    model = best_model

    p_raw_val = X_val[:, 0]  # logit(p_market) is feature 0
    p_raw_probs = [1 / (1 + math.exp(-lp)) for lp in p_raw_val]

    p_cal_probs = model.predict_proba(X_val)[:, 1].tolist()

    brier_raw = brier(p_raw_probs, y_val.tolist())
    brier_cal = brier(p_cal_probs, y_val.tolist())

    log.info("Brier(market_raw)   = %.5f", brier_raw)
    log.info("Brier(calibrated)   = %.5f  (%+.5f vs raw)", brier_cal, brier_cal - brier_raw)

    if brier_cal >= brier_raw:
        log.warning(
            "Calibrator did NOT beat market baseline (%.5f >= %.5f). "
            "Model will NOT be saved. System will use p_market as-is.",
            brier_cal, brier_raw,
        )
        return

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("Saved calibrator → %s", MODEL_PATH)

    # Save feature schema so inference code can validate column order
    from .feature_extraction import CALIBRATOR_FEATURE_NAMES
    with open(SCHEMA_PATH, "w") as f:
        json.dump({"calibrator_features": CALIBRATOR_FEATURE_NAMES}, f, indent=2)
    log.info("Saved feature schema → %s", SCHEMA_PATH)

    # Print per-category breakdown
    _per_category_report(X_val, y_val, p_cal_probs, p_raw_probs)


def _per_category_report(X_val, y_val, p_cal, p_raw):
    """Log Brier score per category (columns 3-6 are category dummies)."""
    import numpy as np

    cat_names = ["sports", "finance", "politics", "science_tech", "culture"]
    # Indices 3-6 = cat_sports, cat_finance, cat_politics, cat_science_tech
    # culture = all zeros in those columns

    for i, cat in enumerate(cat_names[:-1]):
        mask = X_val[:, 3 + i].astype(bool)
        if mask.sum() < 5:
            continue
        bc = brier([p_cal[j] for j, m in enumerate(mask) if m],
                   y_val[mask].tolist())
        br = brier([p_raw[j] for j, m in enumerate(mask) if m],
                   y_val[mask].tolist())
        log.info("  %-15s  n=%-4d  Brier(cal)=%.4f  Brier(raw)=%.4f  delta=%+.4f",
                 cat, int(mask.sum()), bc, br, bc - br)

    # culture = rows where none of the 4 dummies are 1
    mask = (X_val[:, 3:7].sum(axis=1) == 0)
    if mask.sum() >= 5:
        bc = brier([p_cal[j] for j, m in enumerate(mask) if m],
                   y_val[mask].tolist())
        br = brier([p_raw[j] for j, m in enumerate(mask) if m],
                   y_val[mask].tolist())
        log.info("  %-15s  n=%-4d  Brier(cal)=%.4f  Brier(raw)=%.4f  delta=%+.4f",
                 "culture", int(mask.sum()), bc, br, bc - br)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
