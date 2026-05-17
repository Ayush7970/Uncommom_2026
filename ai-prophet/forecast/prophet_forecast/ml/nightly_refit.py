"""Step 11: Online learning refit script for the 2-week evaluation window.

Appends newly resolved events to training_data.csv, then retrains
market_calibrator + blender. Run nightly during the evaluation window.

Usage:
    python -m prophet_forecast.ml.nightly_refit \
        --new-events artifacts/resolved_today.csv

The resolved_today.csv should have the same schema as training_data.csv
(columns: title, market_name, category, p_market, hours_to_close, outcome).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from datetime import datetime

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAINING_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")


def append_new_events(new_csv: str) -> int:
    """Append rows from new_csv to training_data.csv. Returns count appended."""
    if not os.path.exists(new_csv):
        log.error("New events file not found: %s", new_csv)
        return 0

    # Read new rows
    new_rows = []
    with open(new_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_rows.append(row)

    if not new_rows:
        log.info("No new events to append")
        return 0

    # Backup existing training data
    backup = TRAINING_CSV + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(TRAINING_CSV, backup)
    log.info("Backed up training data → %s", backup)

    # Append new rows
    with open(TRAINING_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["title", "market_name", "category", "p_market", "hours_to_close", "outcome"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(new_rows)

    log.info("Appended %d new rows to %s", len(new_rows), TRAINING_CSV)
    return len(new_rows)


def refit() -> None:
    """Retrain market calibrator and blender on updated training data."""
    from .train_market_calibrator import train as train_calibrator
    from .train_blender import train as train_blender

    log.info("Refitting market calibrator...")
    train_calibrator()

    llm_csv = os.path.join(ARTIFACTS_DIR, "llm_predictions.csv")
    with_llm = os.path.exists(llm_csv)
    log.info("Refitting blender (with_llm=%s)...", with_llm)
    train_blender(with_llm=with_llm)

    # Also update FAISS index if science_tech rows added
    try:
        from .train_domain_heads import train_science_knn
        train_science_knn()
    except Exception as exc:
        log.warning("FAISS refit failed (non-fatal): %s", exc)

    log.info("Nightly refit complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-events", required=True,
                        help="CSV of newly resolved events to append")
    args = parser.parse_args()

    n = append_new_events(args.new_events)
    if n > 0:
        refit()
    else:
        log.info("No new data — skipping refit")
