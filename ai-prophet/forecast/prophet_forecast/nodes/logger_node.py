"""Layer 8: Logging — writes forecast to SQLite (Snowflake in Phase 5)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime

from ..state import ForecastState

log = logging.getLogger(__name__)

_DB_PATH = os.environ.get("FORECAST_DB_PATH", "forecast_log.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT,
            market_id TEXT,
            question TEXT,
            category TEXT,
            time_bucket TEXT,
            snapshot_ts TEXT,
            resolves_at TEXT,
            p_market REAL,
            p_ml REAL,
            p_llm_raw REAL,
            p_calibrated REAL,
            p_final REAL,
            w_t REAL,
            outcome INTEGER,                                       
            brier REAL, 
            rationale TEXT,
            confidence REAL,
            iteration INTEGER,
            evidence_count INTEGER,
            error TEXT
        )
    """)
    conn.commit()


def logger_node(state: ForecastState) -> dict:
    """Write completed forecast state to SQLite log."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        _ensure_table(conn)
        conn.execute("""
            INSERT INTO forecasts (
                logged_at, market_id, question, category, time_bucket,
                snapshot_ts, resolves_at, p_market, p_ml, p_llm_raw,
                p_calibrated, p_final, w_t, rationale, confidence,
                iteration, evidence_count, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(UTC).isoformat(),
            state.get("market_id"),
            state.get("question"),
            state.get("category"),
            state.get("time_bucket"),
            state.get("snapshot_ts"),
            state.get("resolves_at"),
            state.get("p_market"),
            state.get("p_ml"),
            state.get("p_llm_raw"),
            state.get("p_calibrated"),
            state.get("p_final"),
            state.get("w_t"),
            state.get("rationale"),
            state.get("confidence"),
            state.get("iteration", 0),
            len(state.get("evidence", [])),
            state.get("error"),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("Logger failed (non-fatal): %s", e)

    return {}
