"""
Train ML heads and save artifacts.

Usage:
    python scripts/train_ml.py --sport
    python scripts/train_ml.py --all

Trains logistic regression on Elo features.
Real NBA data is fetched from FiveThirtyEight's public dataset first;
falls back to synthetic data if the download fails.
"""

from __future__ import annotations

import argparse
import io
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
# Real NBA data from FiveThirtyEight
# ---------------------------------------------------------------------------

_FTE_NBA_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "nba-elo/nbaallelo.csv"
)


def _fetch_real_nba_data() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Download FiveThirtyEight NBA Elo dataset and convert to our feature schema.

    Features: [elo_diff_norm, home_advantage, rest_diff_days, form_diff]
    Label:    1 = team won, 0 = team lost
    """
    try:
        import requests
        import csv

        log.info("Fetching real NBA data from FiveThirtyEight ...")
        resp = requests.get(_FTE_NBA_URL, timeout=30)
        resp.raise_for_status()

        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        log.info("Downloaded %d game rows", len(rows))

        X_list, y_list = [], []
        for row in rows:
            try:
                elo_i     = float(row["elo_i"])
                opp_elo_i = float(row["opp_elo_i"])
                location  = row.get("game_location", "N")
                result    = row.get("game_result", "")

                if not result:
                    continue

                elo_diff_norm  = (elo_i - opp_elo_i) / 400.0
                home_adv       = 1.0 if location == "H" else (-1.0 if location == "A" else 0.0)
                rest_diff_days = 0.0   # not in this dataset
                form_diff      = 0.0   # not in this dataset
                outcome        = 1 if result == "W" else 0

                X_list.append([elo_diff_norm, home_adv, rest_diff_days, form_diff])
                y_list.append(outcome)
            except (ValueError, KeyError):
                continue

        if len(X_list) < 1000:
            log.warning("Too few valid rows (%d) — falling back to synthetic", len(X_list))
            return None

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)
        log.info("Real NBA data: %d samples, win_rate=%.3f", len(y), y.mean())
        return X, y

    except Exception as e:
        log.warning("Could not fetch real NBA data: %s — falling back to synthetic", e)
        return None


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _elo_win_prob(elo_home: float, elo_away: float, home_bonus: float = 65) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_bonus)) / 400))


def generate_sports_data(n: int = 6000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic fallback using realistic Elo distributions."""
    rng = np.random.default_rng(seed)
    elo_home = rng.normal(loc=1500, scale=80, size=n).clip(1200, 1750)
    elo_away = rng.normal(loc=1500, scale=80, size=n).clip(1200, 1750)
    elo_diff_norm = (elo_home - elo_away) / 400.0
    home_adv  = rng.choice([1.0, 0.0, -1.0], size=n, p=[0.45, 0.1, 0.45])
    rest_diff = rng.integers(-3, 4, size=n).astype(float)
    form_diff = rng.uniform(-0.6, 0.6, size=n)
    X = np.column_stack([elo_diff_norm, home_adv, rest_diff, form_diff])
    p_true  = np.array([_elo_win_prob(elo_home[i], elo_away[i], 65 * home_adv[i]) for i in range(n)])
    p_noisy = np.clip(p_true + rng.normal(0, 0.04, size=n), 0.01, 0.99)
    y = rng.binomial(1, p_noisy)
    return X, y


# ---------------------------------------------------------------------------
# Train + save
# ---------------------------------------------------------------------------

def train_sports_model() -> None:
    # Try real data first, fall back to synthetic
    data = _fetch_real_nba_data()
    if data is not None:
        X, y = data
        source = "real FiveThirtyEight NBA"
    else:
        log.info("Generating synthetic sports training data ...")
        X, y = generate_sports_data(n=6000)
        source = "synthetic"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log.info("Training LogisticRegression on %d samples [%s] ...", len(X_train), source)
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
            "data_source": source,
            "n_train":   len(X_train),
            "train_acc": train_acc,
            "test_acc":  test_acc,
            "brier":     test_brier,
        },
    }

    out_path = os.path.join(_ARTIFACTS_DIR, "sports_model.pkl")
    joblib.dump(bundle, out_path)
    log.info("Saved to %s  (source: %s)", out_path, source)


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
