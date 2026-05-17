"""
Analogue retrieval — finds past resolved markets similar to the current question.

Primary:  Snowflake Cortex Search (if credentials available)
Fallback: FAISS + sentence-transformers (local, no credentials needed)

Used by research nodes to inject base-rate context from resolved history.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

import numpy as np

log = logging.getLogger(__name__)

_DB_PATH    = os.environ.get("FORECAST_DB_PATH", "forecast_log.db")
_INDEX_PATH = "prophet_forecast/ml/artifacts/faiss_index.bin"
_META_PATH  = "prophet_forecast/ml/artifacts/faiss_meta.json"

SNOWFLAKE_CONFIG = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.environ.get("SNOWFLAKE_DATABASE", "MY_DB"),
    "schema": os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
    "role": os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
}

# Module-level FAISS index cache
_index   = None
_meta    = []
_encoder = None


# ---------------------------------------------------------------------------
# Sentence encoder (lazy load)
# ---------------------------------------------------------------------------

def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Sentence encoder loaded")
        return _encoder
    except Exception as e:
        log.warning("SentenceTransformer unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# FAISS index (build from SQLite resolved forecasts)
# ---------------------------------------------------------------------------

def _load_or_build_index():
    global _index, _meta

    if _index is not None:
        return _index, _meta

    # Try to load pre-built
    if os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
        try:
            import faiss
            _index = faiss.read_index(_INDEX_PATH)
            with open(_META_PATH) as f:
                _meta = json.load(f)
            log.info("FAISS index loaded: %d entries", _index.ntotal)
            return _index, _meta
        except Exception as e:
            log.warning("Could not load FAISS index: %s — will rebuild", e)

    # Build from SQLite resolved forecasts
    if not os.path.exists(_DB_PATH):
        log.info("No DB found — analogue retrieval unavailable")
        return None, []

    try:
        conn = sqlite3.connect(_DB_PATH)
        rows = conn.execute(
            "SELECT market_id, question, category, p_final, outcome, rationale "
            "FROM forecasts WHERE outcome IS NOT NULL AND question IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("DB query failed: %s", e)
        return None, []

    if len(rows) < 5:
        log.info("Too few resolved forecasts (%d) for FAISS index", len(rows))
        return None, []

    encoder = _get_encoder()
    if encoder is None:
        return None, []

    try:
        import faiss
        questions  = [r[1] for r in rows]
        embeddings = encoder.encode(questions, show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim    = embeddings.shape[1]
        _index = faiss.IndexFlatIP(dim)
        _index.add(embeddings)

        _meta = [
            {"market_id": r[0], "question": r[1], "category": r[2],
             "p_final": r[3], "outcome": r[4], "rationale": r[5]}
            for r in rows
        ]

        os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
        faiss.write_index(_index, _INDEX_PATH)
        with open(_META_PATH, "w") as f:
            json.dump(_meta, f)

        log.info("FAISS index built and saved: %d entries", _index.ntotal)
        return _index, _meta
    except Exception as e:
        log.warning("FAISS index build failed: %s", e)
        return None, []


# ---------------------------------------------------------------------------
# Snowflake Cortex Search (optional)
# ---------------------------------------------------------------------------

def _snowflake_search(question: str, n: int) -> list[dict]:
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT market_id, question, category, p_final, outcome, rationale
            FROM TABLE(
                CORTEX_SEARCH('RATIONALE_SEARCH', %s, %s)
            )
            """,
            (question, n),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"market_id": r[0], "question": r[1], "category": r[2],
             "p_final": r[3], "outcome": r[4], "rationale": r[5]}
            for r in rows
        ]
    except Exception as e:
        log.debug("Snowflake Cortex Search unavailable: %s", e)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_analogues(question: str, n: int = 3) -> list[dict]:
    """
    Return up to n resolved markets similar to the given question.
    Each result: {market_id, question, category, p_final, outcome, rationale}
    """
    # Try Snowflake first
    if all(os.environ.get(k) for k in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]):
        log.info("Attempting analogue retrieval via Snowflake Cortex Search...")
        results = _snowflake_search(question, n)
        log.info("Snowflake Cortex Search returned %d results", len(results))
        if results:
            log.info("Analogue retrieval [Snowflake]: %d results", len(results))
            return results

    # Fallback: FAISS
    index, meta = _load_or_build_index()
    if index is None or not meta:
        return []

    encoder = _get_encoder()
    if encoder is None:
        return []

    try:
        import faiss
        q_vec = encoder.encode([question], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, indices = index.search(q_vec, min(n + 1, index.ntotal))
        results = [meta[i] for i in indices[0] if i < len(meta)]
        log.info("Analogue retrieval [FAISS]: %d results", len(results))
        return results[:n]
    except Exception as e:
        log.warning("FAISS search failed: %s", e)
        return []
