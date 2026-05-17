"""
Replay harness — runs the forecast graph over a dataset and prints Brier score.

Usage:
    python scripts/run_replay.py --dataset small           # built-in 5-market stub
    python scripts/run_replay.py --dataset resolved        # 26 resolved markets (has Brier)
    python scripts/run_replay.py --dataset sports          # 16 sports markets (predictions only)
    python scripts/run_replay.py --dataset economics       # 13 economics markets
    python scripts/run_replay.py --dataset entertainment   # 13 entertainment markets
    python scripts/run_replay.py --dataset path/to/file.json  # custom file
    python scripts/run_replay.py --dataset resolved --limit 5  # quick test
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "snowflake"))
from snowflake.ingest import push_to_snowflake

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


_DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

# Named dataset shortcuts → folder name
_NAMED_DATASETS = {
    "sports":        "sample-sports",
    "economics":     "sample-economics",
    "entertainment": "sample-entertainment",
    "resolved":      "sample-resolved",
}


def _convert_task(task: dict) -> dict | None:
    """Convert ai-prophet-datasets tasks.jsonl format → our replay format."""
    from datetime import timedelta

    question = task.get("title", "")
    if not question:
        return None

    resolves_at = task.get("predict_by") or task.get("metadata", {}).get("source", {}).get("close_time")
    if not resolves_at:
        resolves_at = "2026-12-31T23:59:00Z"

    category_hint = task.get("metadata", {}).get("category", "").lower() or None

    # Simulate being 48 hours before the market closes.
    # Using datetime.now() would make all markets "urgent" (already closed).
    try:
        resolves_dt  = datetime.fromisoformat(resolves_at.replace("Z", "+00:00"))
        snapshot_dt  = resolves_dt - timedelta(hours=48)
        snapshot_ts  = snapshot_dt.isoformat()
    except Exception:
        snapshot_ts  = resolves_at  # fallback

    # Determine outcome from resolved_outcome if present
    outcome = None
    outcomes_list = task.get("outcomes", [])
    resolved = task.get("resolved_outcome")
    if resolved:
        resolved_values = resolved.get("value", [])
        if outcomes_list and resolved_values:
            outcome = 1 if resolved_values[0] == outcomes_list[0] else 0

    # Rewrite "Who won X vs Y?" to binary form so models know which side = YES.
    # The YES side is always outcomes_list[0] (the first listed outcome).
    binary_question = question
    if outcomes_list and len(outcomes_list) >= 2:
        yes_side = outcomes_list[0]
        no_side  = outcomes_list[1]
        q_lower  = question.lower()
        # "Who won X vs Y?" → "Did [yes_side] win against [no_side]?"
        if q_lower.startswith("who won") or q_lower.startswith("who will win"):
            binary_question = (
                f"Did {yes_side} win/prevail? "
                f"(YES = {yes_side} wins, NO = {no_side} wins or other outcome) "
                f"Original: {question}"
            )
        # "Who became X?" or "Which Y?" → keep but add framing
        elif q_lower.startswith("who ") or q_lower.startswith("which "):
            binary_question = (
                f"Did {yes_side} win/achieve the result? "
                f"(YES = {yes_side}, NO = anyone else) "
                f"Original: {question}"
            )
        # "How many?" numerical questions — keep as-is (too ambiguous to rewrite)

    return {
        "market_id":       task.get("task_id", task.get("source", "unknown")),
        "question":        binary_question,
        "category_hint":   category_hint,
        "resolves_at":     resolves_at,
        "price_snapshots": [{"ts": snapshot_ts, "price_yes": 0.5}],
        "outcome":         outcome,
    }


def _load_jsonl(path: str) -> list[dict]:
    """Load a .jsonl file (one JSON object per line)."""
    markets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            converted = _convert_task(task)
            if converted:
                markets.append(converted)
    return markets


def load_dataset(path: str) -> list[dict]:
    # Built-in stub
    if path == "small":
        log.info("Using built-in stub dataset (%d markets)", len(STUB_DATASET))
        return STUB_DATASET

    # Named shortcut (sports, economics, entertainment, resolved)
    if path in _NAMED_DATASETS:
        folder = _NAMED_DATASETS[path]
        jsonl_path = os.path.join(_DATASETS_DIR, folder, "tasks.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Dataset '{path}' not found at {jsonl_path}.\n"
                f"Run from forecast/ directory or download datasets first."
            )
        markets = _load_jsonl(jsonl_path)
        has_outcomes = sum(1 for m in markets if m["outcome"] is not None)
        log.info("Loaded '%s': %d markets (%d with outcomes)", path, len(markets), has_outcomes)
        if has_outcomes == 0:
            log.warning("No outcomes found — Brier score cannot be computed. Predictions will still run.")
        return markets

    # Path to .jsonl file
    if path.endswith(".jsonl"):
        markets = _load_jsonl(path)
        log.info("Loaded %d markets from %s", len(markets), path)
        return markets

    # Path to .json file
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("markets", data.get("events", [data]))
    raise ValueError(f"Unrecognised dataset format in {path}")


def run_market(market: dict) -> list[dict]:
    """Run the forecast graph for every snapshot of a market. Returns result rows."""
    rows = []
    outcome = market.get("outcome")

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
                "outcome": int(outcome) if outcome is not None else None,
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
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N markets (0-indexed)")
    args = parser.parse_args()

    markets = load_dataset(args.dataset)
    if args.offset:
        markets = markets[args.offset:]
    if args.limit:
        markets = markets[:args.limit]

    log.info("Running replay on %d markets ...", len(markets))

    all_rows: list[dict] = []
    for i, market in enumerate(markets, 1):
        log.info("[%d/%d] %s", i, len(markets), market["market_id"])
        all_rows.extend(run_market(market))

    if not all_rows:
        log.error("No results — all markets were skipped.")
        sys.exit(1)

    # Write CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "forecasts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Forecasts written to %s", csv_path)

    # Only compute Brier if outcomes are available
    rows_with_outcomes = [r for r in all_rows if r.get("outcome") is not None]
    if rows_with_outcomes:
        summarise(rows_with_outcomes, output_dir=args.output_dir)
    else:
        log.info("No outcomes available — Brier score skipped. Predictions saved to %s", csv_path)
        print(f"\n{'='*60}")
        print(f"  PREDICTIONS ONLY ({len(all_rows)} forecasts — no outcomes yet)")
        print(f"  Results saved to: {csv_path}")
        print(f"{'='*60}\n")
    
    db_path = os.environ.get("FORECAST_DB_PATH", "forecast_log.db")
    push_to_snowflake(db_path)


if __name__ == "__main__":
    main()
