"""Step 1: Load the HuggingFace dataset and expand rows into binary training examples.

Actual schema (prophetarena/Prophet-Arena-Subset-1200):
  market_data:    JSON str → {"Market Name": {"yes_ask": float, "yes_bid": float, ...}}
  market_outcome: JSON str → {"Market Name": 0 or 1}
  markets:        JSON str → ["Market Name", ...]
  category:       "Sports" | "Entertainment" | "Politics" | "Economics" |
                  "Companies" | "Mentions" | "Climate and Weather" | "Other"
  snapshot_time, close_time: ISO timestamp strings

Run:
    python -m prophet_forecast.ml.build_training_data

Output: prophet_forecast/ml/artifacts/training_data.csv
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
OUTPUT_CSV = os.path.join(ARTIFACTS_DIR, "training_data.csv")
HF_DATASET = "prophetarena/Prophet-Arena-Subset-1200"

# Map raw HuggingFace categories → our 5 domain buckets
CATEGORY_MAP = {
    "sports":              "sports",
    "entertainment":       "culture",
    "politics":            "politics",
    "economics":           "finance",
    "companies":           "finance",
    "climate and weather": "science_tech",
    "mentions":            "other",
    "other":               "other",
}


def _normalize_category(raw: str) -> str:
    return CATEGORY_MAP.get(raw.lower().strip(), "other")


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    ts = str(ts)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts[:26] + ts[26:].replace(":", ""), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def _hours_between(t_start, t_end) -> float:
    s = _parse_ts(t_start)
    e = _parse_ts(t_end)
    if s and e:
        return max(0.0, (e - s).total_seconds() / 3600.0)
    return 24.0  # default if timestamps missing


def _load_json_field(value) -> dict | list | None:
    """Parse a field that may already be a dict/list or a JSON string."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def expand_row(row: dict) -> list[dict]:
    """Expand one dataset row into one or more binary training examples."""
    category = _normalize_category(row.get("category") or "")
    title = str(row.get("augmented_title") or row.get("title") or "")
    hours = _hours_between(row.get("snapshot_time"), row.get("close_time"))

    market_data = _load_json_field(row.get("market_data"))
    market_outcome = _load_json_field(row.get("market_outcome"))
    markets = _load_json_field(row.get("markets"))

    if not isinstance(market_data, dict) or not isinstance(market_outcome, dict):
        return []

    if not isinstance(markets, list):
        markets = list(market_data.keys())

    examples = []
    for market_name in markets:
        md = market_data.get(market_name)
        outcome_val = market_outcome.get(market_name)

        if not isinstance(md, dict) or outcome_val is None:
            continue

        yes_ask = md.get("yes_ask")
        yes_bid = md.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            continue

        # Values are on 0-100 scale → convert to 0-1
        p_market = (float(yes_ask) + float(yes_bid)) / 200.0

        if not (0.0 < p_market < 1.0):
            continue

        try:
            outcome = int(outcome_val)
        except (ValueError, TypeError):
            continue

        examples.append({
            "title": title,
            "market_name": market_name,
            "category": category,
            "p_market": round(p_market, 6),
            "hours_to_close": round(hours, 2),
            "outcome": outcome,
        })

    return examples


def build(output_path: str = OUTPUT_CSV) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets package not installed: pip install datasets")
        sys.exit(1)

    log.info("Loading %s from HuggingFace...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")
    log.info("Loaded %d source rows", len(ds))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["title", "market_name", "category", "p_market", "hours_to_close", "outcome"]
    rows_written = 0
    rows_skipped = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for raw_row in ds:
            examples = expand_row(dict(raw_row))
            if examples:
                writer.writerows(examples)
                rows_written += len(examples)
            else:
                rows_skipped += 1

    log.info(
        "Wrote %d training rows from %d source rows (%d skipped) → %s",
        rows_written, len(ds), rows_skipped, output_path,
    )

    # Print category breakdown
    _log_category_breakdown(output_path)

    if rows_written < 100:
        log.error("Too few rows written — check dataset parsing above")
        sys.exit(1)

    return rows_written


def _log_category_breakdown(csv_path: str) -> None:
    from collections import Counter
    counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["category"]] += 1
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        log.info("  %-15s  %d rows", cat, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = build()
    print(f"\nDone. {count} rows written to {OUTPUT_CSV}")
