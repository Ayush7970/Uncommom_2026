"""
Train ML heads and save artifacts.

Usage:
    python scripts/train_ml.py --sport nba
    python scripts/train_ml.py --all

Trains logistic regression on Elo features using synthetic historical data
seeded with realistic Elo distributions.  A real dataset from ESPN or
FiveThirtyEight can be dropped in later — same feature schema.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("train_ml")

_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "prophet_forecast", "ml", "artifacts"
)
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data generator for sports
# ---------------------------------------------------------------------------

def _elo_win_prob(elo_home: float, elo_away: float, home_bonus: float = 65) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_bonus)) / 400))


def generate_sports_data(n: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic NBA/NFL/NHL game data with realistic Elo distributions.

    Features (X columns):
        0: elo_diff / 400         (normalised Elo difference)
        1: home_advantage          (1.0 home, 0.0 neutral, -1.0 away)
        2: rest_diff_days          (−3 to +3)
        3: form_diff               (recent 5-game win% difference)

    Label (y):
        1 = home/team-1 wins, 0 = away/team-2 wins
    """
    rng = np.random.default_rng(seed)

    # Sample Elo from realistic range: elite ≈ 1650, league average ≈ 1500
    elo_home = rng.normal(loc=1500, scale=80, size=n).clip(1200, 1750)
    elo_away = rng.normal(loc=1500, scale=80, size=n).clip(1200, 1750)
    elo_diff_norm = (elo_home - elo_away) / 400.0

    # Home advantage: most games have a home team
    home_adv = rng.choice([1.0, 0.0, -1.0], size=n, p=[0.45, 0.1, 0.45])

    # Rest difference: -3 to +3 days (back-to-back fatigue, etc.)
    rest_diff = rng.integers(-3, 4, size=n).astype(float)

    # Recent form: difference in last-5 win %
    form_diff = rng.uniform(-0.6, 0.6, size=n)

    X = np.column_stack([elo_diff_norm, home_adv, rest_diff, form_diff])

    # Generate outcomes using the Elo formula + noise
    p_true = np.array([
        _elo_win_prob(elo_home[i], elo_away[i],
                      home_bonus=65 * home_adv[i])
        for i in range(n)
    ])
    # Add small random noise to simulate real-world variance
    p_noisy = np.clip(p_true + rng.normal(0, 0.04, size=n), 0.01, 0.99)
    y = rng.binomial(1, p_noisy)

    return X, y


# ---------------------------------------------------------------------------
# Train + save
# ---------------------------------------------------------------------------

def train_sports_model() -> None:
    log.info("Generating synthetic sports training data ...")
    X, y = generate_sports_data(n=6000)

    # Strict train / test split — NEVER mix test markets into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log.info("Training LogisticRegression on %d samples ...", len(X_train))
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    train_acc  = model.score(X_train, y_train)
    test_acc   = model.score(X_test, y_test)
    p_test     = model.predict_proba(X_test)[:, 1]
    test_brier = brier_score_loss(y_test, p_test)
    test_ll    = log_loss(y_test, p_test)

    log.info("Train acc=%.4f  Test acc=%.4f  Brier=%.4f  LogLoss=%.4f",
             train_acc, test_acc, test_brier, test_ll)
    log.info("Coefficients: elo_diff=%.3f  home_adv=%.3f  rest=%.3f  form=%.3f  intercept=%.3f",
             *model.coef_[0], model.intercept_[0])

    bundle = {
        "model": model,
        "meta": {
            "feature_names": ["elo_diff_norm", "home_advantage", "rest_diff_days", "form_diff"],
            "n_train":   len(X_train),
            "train_acc": train_acc,
            "test_acc":  test_acc,
            "brier":     test_brier,
        },
    }

    out_path = os.path.join(_ARTIFACTS_DIR, "sports_model.pkl")
    joblib.dump(bundle, out_path)
    log.info("Saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML heads for Prophet Forecast")
    parser.add_argument("--sport",  action="store_true", help="Train sports model")
    parser.add_argument("--all",    action="store_true", help="Train all available models")
    args = parser.parse_args()

    if args.all or args.sport:
        train_sports_model()
    else:
        parser.print_help()
        log.info("Hint: run with --sport or --all")


if __name__ == "__main__":
    main()
