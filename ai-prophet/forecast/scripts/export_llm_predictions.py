"""
Convert run_replay.py output → llm_predictions.csv for blender training.

run_replay.py writes outputs/forecasts.csv with columns:
  question, market_id, p_llm_raw, p_ml, outcome, ...

train_blender.py --with-llm expects artifacts/llm_predictions.csv with:
  title, market_name, p_llm, p_ml, trial_variance, outcome

Usage:
    python scripts/export_llm_predictions.py
    python scripts/export_llm_predictions.py --input outputs/forecasts.csv --output prophet_forecast/ml/artifacts/llm_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert forecasts.csv → llm_predictions.csv for blender training"
    )
    parser.add_argument(
        "--input", default="outputs/forecasts.csv",
        help="Path to run_replay.py output CSV (default: outputs/forecasts.csv)",
    )
    parser.add_argument(
        "--output", default="prophet_forecast/ml/artifacts/llm_predictions.csv",
        help="Destination path for llm_predictions.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(
            f"Input not found: {args.input}\n"
            "Run 'python scripts/run_replay.py --dataset <data>' first."
        )

    rows_out = []
    skipped = 0

    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p_llm = row.get("p_llm_raw", "").strip()
            p_ml  = row.get("p_ml", "").strip()
            outcome = row.get("outcome", "").strip()

            # Skip rows where LLM or ML prediction is missing (graph fell back)
            if not p_llm or not p_ml or not outcome:
                skipped += 1
                continue

            rows_out.append({
                "title":          row.get("question", ""),
                "market_name":    row.get("market_id", ""),
                "p_llm":          p_llm,
                "p_ml":           p_ml,
                # p_iterations variance is not stored in CSV; default to 0.0
                "trial_variance": "0.0",
                "outcome":        outcome,
            })

    if not rows_out:
        sys.exit(
            f"No usable rows found in {args.input}. "
            "Ensure the pipeline ran with an LLM key (ANTHROPIC_API_KEY) so p_llm_raw is populated."
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "market_name", "p_llm", "p_ml", "trial_variance", "outcome"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Exported {len(rows_out)} rows → {args.output}")
    if skipped:
        print(f"Skipped {skipped} rows with missing p_llm_raw or p_ml (fallback forecasts).")
    print()
    print("Next steps:")
    print("  python3 -m prophet_forecast.ml.train_blender --with-llm")


if __name__ == "__main__":
    main()
