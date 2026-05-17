"""
Replay harness — runs the forecast graph over a dataset and prints Brier score.

Usage:
    python scripts/run_replay.py --dataset small
    python scripts/run_replay.py --dataset path/to/dataset.json
    python scripts/run_replay.py --dataset path/to/dataset.json --output-dir outputs/ --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime

# Allow running from the forecast/ directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from prophet_forecast.graph import forecast_graph
from prophet_forecast.eval.metrics import summarise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_replay")

# ---------------------------------------------------------------------------
# Built-in small stub dataset (used when --dataset small)
# ---------------------------------------------------------------------------
STUB_DATASET = [
    {
        "market_id": "stub:NBA-LAL-GSW-001",
        "question": "Will the Lakers beat the Warriors?",
        "category_hint": "sports",
        "resolves_at": "2026-06-01T03:00:00Z",
        "price_snapshots": [
            {"ts": "2026-05-27T03:00:00Z", "price_yes": 0.52},
            {"ts": "2026-05-30T03:00:00Z", "price_yes": 0.48},
        ],
        "outcome": 1,
    },
    {
        "market_id": "stub:SENATE-TX-2026",
        "question": "Will the Democratic candidate win the Texas Senate seat in 2026?",
        "category_hint": "politics",
        "resolves_at": "2026-11-10T06:00:00Z",
        "price_snapshots": [
            {"ts": "2026-05-16T00:00:00Z", "price_yes": 0.35},
        ],
        "outcome": 0,
    },
    {
        "market_id": "stub:BTC-100K",
        "question": "Will Bitcoin exceed $100,000 by end of June 2026?",
        "category_hint": "finance",
        "resolves_at": "2026-06-30T23:59:00Z",
        "price_snapshots": [
            {"ts": "2026-05-16T00:00:00Z", "price_yes": 0.62},
        ],
        "outcome": 0,
    },
    {
        "market_id": "stub:NHL-COL-CUP",
        "question": "Will the Colorado Avalanche win the 2026 Stanley Cup?",
        "category_hint": "sports",
        "resolves_at": "2026-06-20T03:00:00Z",
        "price_snapshots": [
            {"ts": "2026-05-16T00:00:00Z", "price_yes": 0.40},
            {"ts": "2026-06-01T00:00:00Z", "price_yes": 0.55},
        ],
        "outcome": 1,
    },
    {
        "market_id": "stub:FDA-DRUG-X",
        "question": "Will the FDA approve Drug X by Q3 2026?",
        "category_hint": "science_tech",
        "resolves_at": "2026-09-30T23:59:00Z",
        "price_snapshots": [
            {"ts": "2026-05-16T00:00:00Z", "price_yes": 0.70},
        ],
        "outcome": 1,
    },
]


def load_dataset(path: str) -> list[dict]:
    if path == "small":
        log.info("Using built-in stub dataset (%d markets)", len(STUB_DATASET))
        return STUB_DATASET

    with open(path) as f:
        data = json.load(f)

    # Support both a list of markets and a dict with a "markets" key
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("markets", data.get("events", [data]))
    raise ValueError(f"Unrecognised dataset format in {path}")


def run_market(market: dict) -> list[dict]:
    """Run the forecast graph for every snapshot of a market. Returns result rows."""
    rows = []
    outcome = market.get("outcome")
    if outcome is None:
        log.warning("Skipping %s — no outcome (not yet resolved)", market["market_id"])
        return rows

    for snap in market.get("price_snapshots", []):
        state_in = {
            "market_id": market["market_id"],
            "question": market["question"],
            "p_market": float(snap["price_yes"]),
            "snapshot_ts": snap["ts"],
            "resolves_at": market["resolves_at"],
            "category_hint": market.get("category_hint"),
            "evidence": [],
            "search_queries_used": [],
            "p_iterations": [],
            "new_queries": [],
            "iteration": 0,
            "trace_id": str(uuid.uuid4()),
        }

        try:
            result = forecast_graph.invoke(state_in)
            rows.append({
                "market_id": market["market_id"],
                "question": market["question"],
                "snapshot_ts": snap["ts"],
                "resolves_at": market["resolves_at"],
                "category": result.get("category", "unknown"),
                "time_bucket": result.get("time_bucket", "unknown"),
                "p_market": float(snap["price_yes"]),
                "p_ml": result.get("p_ml"),
                "p_llm_raw": result.get("p_llm_raw"),
                "p_calibrated": result.get("p_calibrated"),
                "p_final": result.get("p_final", float(snap["price_yes"])),
                "w_t": result.get("w_t"),
                "outcome": int(outcome),
                "rationale": result.get("rationale", ""),
                "error": result.get("error"),
            })
        except Exception as e:
            log.error("Graph failed for %s snapshot %s: %s", market["market_id"], snap["ts"], e)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet Forecast replay harness")
    parser.add_argument("--dataset", required=True,
                        help='Path to dataset JSON, or "small" for the built-in stub')
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for CSV output and calibration plot")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of markets to process (for quick tests)")
    args = parser.parse_args()

    markets = load_dataset(args.dataset)
    if args.limit:
        markets = markets[:args.limit]

    log.info("Running replay on %d markets ...", len(markets))

    all_rows: list[dict] = []
    for i, market in enumerate(markets, 1):
        log.info("[%d/%d] %s", i, len(markets), market["market_id"])
        all_rows.extend(run_market(market))

    if not all_rows:
        log.error("No results — check dataset format or that markets have outcomes.")
        sys.exit(1)

    # Write CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "forecasts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Forecasts written to %s", csv_path)

    # Print summary + save calibration plot
    summarise(all_rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
