"""Step 12: Brier score comparison across all models and domains.

Run:
    python -m prophet_forecast.ml.evaluate

Loads training_data.csv + all artifacts, runs a held-out evaluation,
and prints a comparison table.
"""

from __future__ import annotations

import csv
import logging
import math
import os

import numpy as np

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAINING_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")


def brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _safe_logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def load_val_data() -> list[dict]:
    """Load validation rows (last 20% of CSV, stable with fixed seed)."""
    rows = []
    with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["p_market"])
                h = float(row["hours_to_close"])
                o = int(row["outcome"])
                cat = row.get("category", "")
                if not (0.0 < p < 1.0):
                    continue
                rows.append({"p_market": p, "h": h, "cat": cat,
                              "outcome": o, "title": row.get("title", "")})
            except (ValueError, KeyError):
                continue
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    split = max(1, int(len(rows) * 0.2))
    val_idx = idx[:split]
    return [rows[i] for i in val_idx]


def run() -> None:
    try:
        import joblib
    except ImportError:
        log.error("pip install joblib")
        return

    if not os.path.exists(TRAINING_CSV):
        log.error("Run build_training_data.py first")
        return

    val_rows = load_val_data()
    log.info("Evaluating on %d validation rows", len(val_rows))

    # Load models
    def load(name):
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    calibrator = load("market_calibrator.pkl")
    sports = load("sports_logistic.pkl")
    culture = load("culture_logistic.pkl")
    blender = load("blender.pkl")

    from .feature_extraction import build_calibrator_features

    results: dict[str, dict[str, list]] = {}

    for row in val_rows:
        p_market = row["p_market"]
        h = row["h"]
        cat = row["cat"]
        o = row["outcome"]

        # Baseline: raw market
        _add(results, "ALL", "market_raw", p_market, o)
        _add(results, cat or "unknown", "market_raw", p_market, o)

        # Market calibrator
        if calibrator is not None:
            X = [build_calibrator_features(p_market, h, cat)]
            p_cal = float(calibrator.predict_proba(X)[0][1])
            _add(results, "ALL", "calibrator", p_cal, o)
            _add(results, cat or "unknown", "calibrator", p_cal, o)

    # Print table
    print("\n" + "=" * 65)
    print(f"{'Domain':<18} {'Model':<18} {'N':>5} {'Brier':>8} {'vs_raw':>8}")
    print("=" * 65)

    all_raw = brier(results.get("ALL", {}).get("market_raw", {}).get("p", []),
                    results.get("ALL", {}).get("market_raw", {}).get("y", []))

    for domain in sorted(results):
        domain_raw = brier(results[domain].get("market_raw", {}).get("p", []),
                           results[domain].get("market_raw", {}).get("y", []))
        for model in sorted(results[domain]):
            data = results[domain][model]
            b = brier(data["p"], data["y"])
            delta = b - domain_raw if not math.isnan(b) and not math.isnan(domain_raw) else float("nan")
            print(f"{domain:<18} {model:<18} {len(data['p']):>5} {b:>8.5f} {delta:>+8.5f}")

    print("=" * 65)
    print(f"\nOverall market baseline Brier: {all_raw:.5f}")


def _add(results: dict, domain: str, model: str, p: float, o: int) -> None:
    results.setdefault(domain, {}).setdefault(model, {"p": [], "y": []})
    results[domain][model]["p"].append(p)
    results[domain][model]["y"].append(o)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
