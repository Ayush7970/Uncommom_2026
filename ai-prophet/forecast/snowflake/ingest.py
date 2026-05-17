"""
Snowflake ingest helper — pushes forecast_log.db rows to Snowflake.

Usage:
    python snowflake/ingest.py --db forecast_log.db

Falls back gracefully if credentials are missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)


def push_to_snowflake(db_path: str) -> None:
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    if not all(os.environ.get(k) for k in required):
        log.warning("Snowflake credentials not set — skipping ingest. "
                    "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.")
        return

    try:
        import snowflake.connector
    except ImportError:
        log.warning("snowflake-snowpark-python not installed. "
                    "Run: pip install snowflake-snowpark-python")
        return

    # Columns shared between SQLite and Snowflake (excludes 'id' and 'evidence_count')
    # Snowflake has EVIDENCE VARIANT (JSON) which SQLite doesn't store — left NULL on ingest
    COLS = [
        "logged_at", "market_id", "question", "category", "time_bucket",
        "snapshot_ts", "resolves_at", "p_market", "p_ml", "p_llm_raw",
        "p_calibrated", "p_final", "w_t", "outcome", "brier",
        "rationale", "confidence", "iteration", "error", "evidence",
    ]
    col_sql = ", ".join(COLS)

    # Load from SQLite
    conn_sqlite = sqlite3.connect(db_path)
    rows = conn_sqlite.execute(f"SELECT {col_sql} FROM forecasts").fetchall()
    conn_sqlite.close()

    if not rows:
        log.info("No forecasts to push.")
        return

    # Push to Snowflake
    conn_sf = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ.get("SNOWFLAKE_DATABASE", "PROPHET_FORECAST"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    )
    cur = conn_sf.cursor()


    # PARSE_JSON in a VALUES clause is rejected by the connector; use SELECT instead.
    select_exprs = ", ".join(
        "PARSE_JSON(%s)" if c == "evidence" else "%s" for c in COLS
    )
    insert_sql = f"INSERT INTO FORECASTS ({col_sql}) SELECT {select_exprs}"
    for row in rows:
        cur.execute(insert_sql, row)
    conn_sf.commit()
    cur.close()
    conn_sf.close()

    log.info("Pushed %d rows to Snowflake FORECASTS table.", len(rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Push forecast log to Snowflake")
    parser.add_argument("--db", default="forecast_log.db")
    args = parser.parse_args()
    push_to_snowflake(args.db)


if __name__ == "__main__":
    main()
