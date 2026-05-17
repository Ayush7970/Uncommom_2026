"""Brier score, log-loss, and calibration plot."""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np

log = logging.getLogger(__name__)


def brier_score(p_yes: list[float], outcomes: list[int]) -> float:
    """Mean Brier score: lower is better. Perfect = 0, random = 0.25."""
    if not p_yes:
        return float("nan")
    arr_p = np.array(p_yes, dtype=float)
    arr_o = np.array(outcomes, dtype=float)
    return float(np.mean((arr_p - arr_o) ** 2))


def log_loss(p_yes: list[float], outcomes: list[int], eps: float = 1e-7) -> float:
    """Mean log-loss (cross-entropy)."""
    if not p_yes:
        return float("nan")
    arr_p = np.clip(p_yes, eps, 1 - eps)
    arr_o = np.array(outcomes, dtype=float)
    return float(-np.mean(arr_o * np.log(arr_p) + (1 - arr_o) * np.log(1 - arr_p)))


def calibration_plot(
    p_yes: list[float],
    outcomes: list[int],
    n_bins: int = 10,
    output_path: str = "outputs/calibration.png",
    title: str = "Calibration",
) -> None:
    """Save a reliability diagram (predicted vs actual frequency)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping calibration plot")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    arr_p = np.array(p_yes, dtype=float)
    arr_o = np.array(outcomes, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_fracs, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (arr_p >= lo) & (arr_p < hi)
        if mask.sum() == 0:
            continue
        bin_means.append(arr_p[mask].mean())
        bin_fracs.append(arr_o[mask].mean())
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, "o-", label="Model", color="steelblue")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    log.info("Calibration plot saved to %s", output_path)


def summarise(
    records: list[dict],
    output_dir: str = "outputs",
) -> dict:
    """
    Compute Brier, log-loss, and per-bucket/category breakdowns.

    Each record must have: p_final, p_market, outcome, time_bucket, category.
    Returns a dict of summary stats and prints a table.
    """
    os.makedirs(output_dir, exist_ok=True)

    p_finals  = [r["p_final"]  for r in records]
    p_markets = [r["p_market"] for r in records]
    outcomes  = [r["outcome"]  for r in records]

    overall_brier  = brier_score(p_finals, outcomes)
    baseline_brier = brier_score(p_markets, outcomes)
    overall_ll     = log_loss(p_finals, outcomes)

    print("\n" + "=" * 60)
    print(f"  RESULTS  ({len(records)} forecasts)")
    print("=" * 60)
    print(f"  Brier score    : {overall_brier:.4f}")
    print(f"  Market baseline: {baseline_brier:.4f}")
    print(f"  Delta          : {overall_brier - baseline_brier:+.4f}  "
          f"({'better' if overall_brier < baseline_brier else 'worse'} than market)")
    print(f"  Log-loss       : {overall_ll:.4f}")

    # Per time-bucket
    by_bucket: dict[str, list] = defaultdict(list)
    for r in records:
        by_bucket[r.get("time_bucket", "unknown")].append(r)

    print("\n  --- By time bucket ---")
    for bucket, recs in sorted(by_bucket.items()):
        b = brier_score([r["p_final"] for r in recs], [r["outcome"] for r in recs])
        print(f"    {bucket:<10}  n={len(recs):>4}  Brier={b:.4f}")

    # Per category
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "unknown")].append(r)

    print("\n  --- By category ---")
    for cat, recs in sorted(by_cat.items()):
        b = brier_score([r["p_final"] for r in recs], [r["outcome"] for r in recs])
        print(f"    {cat:<15}  n={len(recs):>4}  Brier={b:.4f}")
    print("=" * 60 + "\n")

    calibration_plot(
        p_finals, outcomes,
        output_path=os.path.join(output_dir, "calibration.png"),
        title=f"Calibration (Brier={overall_brier:.4f} vs market {baseline_brier:.4f})",
    )

    return {
        "brier": overall_brier,
        "baseline_brier": baseline_brier,
        "log_loss": overall_ll,
        "n": len(records),
        "by_bucket": {k: brier_score([r["p_final"] for r in v], [r["outcome"] for r in v])
                      for k, v in by_bucket.items()},
        "by_category": {k: brier_score([r["p_final"] for r in v], [r["outcome"] for r in v])
                        for k, v in by_cat.items()},
    }
