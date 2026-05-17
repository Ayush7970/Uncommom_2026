"""
Fit a Platt (logistic regression) calibrator on historical forecast data.

Usage:
    python scripts/fit_calibrator.py               # uses synthetic data
    python scripts/fit_calibrator.py --db forecast_log.db  # uses real SQLite log

The calibrator corrects systematic LLM overconfidence:
  - When LLM says 0.85, true freq is ~0.72  → calibrator compresses toward centre
  - When LLM says 0.15, true freq is ~0.28  → calibrator pulls up toward centre

Saves artifact to prophet_forecast/ml/artifacts/calibrator.pkl.
Also saves before/after calibration plots to outputs/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("fit_calibrator")

_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "prophet_forecast", "ml", "artifacts"
)
_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(_OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_db(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (p_llm_raw, outcome) pairs from SQLite forecast log."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT p_llm_raw, outcome FROM forecasts WHERE p_llm_raw IS NOT NULL AND outcome IS NOT NULL"
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"No resolved forecasts found in {db_path}")
    arr = np.array(rows, dtype=float)
    log.info("Loaded %d resolved forecasts from %s", len(arr), db_path)
    return arr[:, 0].reshape(-1, 1), arr[:, 1]


def generate_synthetic_data(n: int = 3000, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate systematic LLM overconfidence.

    LLMs tend to be overconfident:
      - Say 0.85 when truth is ~0.72
      - Say 0.15 when truth is ~0.28
    We model this as: true_prob = sigmoid(0.7 * logit(llm_prob))
    """
    rng = np.random.default_rng(seed)

    # Sample LLM predictions from a realistic distribution
    # (concentrated near extremes due to overconfidence)
    p_llm = np.concatenate([
        rng.beta(2, 5, size=n // 3),     # over-confident lows
        rng.uniform(0.35, 0.65, n // 3), # uncertain midrange
        rng.beta(5, 2, size=n // 3),     # over-confident highs
    ])
    p_llm = np.clip(p_llm, 0.02, 0.98)

    # True probability is a compressed version (Platt model: sigmoid(a*logit + b))
    logit_llm  = np.log(p_llm / (1 - p_llm))
    true_logit = 0.70 * logit_llm - 0.05   # compress + small negative bias
    p_true     = 1 / (1 + np.exp(-true_logit))

    # Sample binary outcomes
    outcomes = rng.binomial(1, p_true)

    log.info("Generated %d synthetic forecasts (mean p_llm=%.3f, mean outcome=%.3f)",
             len(p_llm), p_llm.mean(), outcomes.mean())
    return p_llm.reshape(-1, 1), outcomes


# ---------------------------------------------------------------------------
# Plot calibration curves
# ---------------------------------------------------------------------------

def _plot_calibration(
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    y: np.ndarray,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping calibration comparison plot")
        return

    frac_raw, mean_raw = calibration_curve(y, p_raw, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y, p_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_raw, frac_raw, "o-", color="tomato",     label=f"Before (Brier={brier_score_loss(y, p_raw):.4f})")
    ax.plot(mean_cal, frac_cal, "s-", color="steelblue",  label=f"After  (Brier={brier_score_loss(y, p_cal):.4f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual frequency")
    ax.set_title("Calibration: Before vs After Platt Scaling")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    log.info("Calibration comparison plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Fit + save
# ---------------------------------------------------------------------------

def fit_calibrator(db_path: str | None = None) -> None:
    if db_path and os.path.exists(db_path):
        X, y = load_from_db(db_path)
    else:
        log.info("No DB path / no resolved data — using synthetic overconfidence data")
        X, y = generate_synthetic_data(n=3000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Platt scaling = logistic regression on raw scores
    calibrator = LogisticRegression(C=1.0, max_iter=200)
    calibrator.fit(X_train, y_train)

    p_raw_test = X_test.ravel()
    p_cal_test = calibrator.predict_proba(X_test)[:, 1]

    brier_before = brier_score_loss(y_test, p_raw_test)
    brier_after  = brier_score_loss(y_test, p_cal_test)

    log.info("Calibration results on held-out test set:")
    log.info("  Brier BEFORE: %.4f", brier_before)
    log.info("  Brier AFTER:  %.4f  (delta=%+.4f)", brier_after, brier_after - brier_before)
    log.info("  Platt coef: a=%.4f  b=%.4f",
             calibrator.coef_[0][0], calibrator.intercept_[0])

    # Save
    out_path = os.path.join(_ARTIFACTS_DIR, "calibrator.pkl")
    joblib.dump(calibrator, out_path)
    log.info("Calibrator saved to %s", out_path)

    # Plot before/after
    _plot_calibration(
        p_raw=p_raw_test,
        p_cal=p_cal_test,
        y=y_test,
        out_path=os.path.join(_OUTPUTS_DIR, "calibration_platt.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt calibrator for Prophet Forecast")
    parser.add_argument("--db", default=None,
                        help="Path to SQLite forecast log (default: use synthetic data)")
    args = parser.parse_args()
    fit_calibrator(db_path=args.db)


if __name__ == "__main__":
    main()
