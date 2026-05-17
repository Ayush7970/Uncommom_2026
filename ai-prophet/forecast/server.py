"""
PRIMA — ProphetHacks Forecast Endpoint
FastAPI server that wraps the full PRIMA LangGraph pipeline.

Endpoints:
  GET  /health   → wake-up ping (Render free tier needs this)
  POST /predict  → full PRIMA forecast (router → research → ensemble → calibrator)

Deploy on Render:
  Build command : pip install -r forecast/requirements.txt
  Start command : cd forecast && uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from datetime import UTC, datetime

from dotenv import load_dotenv

# ── Load .env (works locally; on Render use env var dashboard) ───────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Ensure prophet_forecast package is importable ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("prima.server")

app = FastAPI(title="PRIMA Forecast Agent", version="1.0.0")


# ── Request / Response models (ProphetHacks contract) ────────────────────────

class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str


class PredictionResponse(BaseModel):
    p_yes: float
    rationale: str


# ── Lazy-load the LangGraph pipeline (avoids slow import on health checks) ───

_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        log.info("Loading PRIMA forecast graph...")
        from prophet_forecast.graph import forecast_graph
        _graph = forecast_graph
        log.info("PRIMA graph loaded.")
    return _graph


# ── Core forecast logic ──────────────────────────────────────────────────────

def _build_question(event: EventRequest) -> str:
    """Combine title + subtitle into a single clear question."""
    q = event.title.strip()
    if event.subtitle:
        q = f"{q} ({event.subtitle.strip()})"
    return q


def _run_prima(event: EventRequest) -> PredictionResponse:
    """
    Run the full PRIMA 8-layer pipeline for one event.
    Falls back to p_yes=0.50 if the pipeline raises.
    """
    question = _build_question(event)

    # Build snapshot: treat now as prediction time
    snapshot_ts = datetime.now(UTC).isoformat()

    state_in = {
        "market_id":          event.market_ticker,
        "question":           question,
        "p_market":           0.50,          # no market price from ProphetHacks
        "snapshot_ts":        snapshot_ts,
        "resolves_at":        event.close_time,
        "category_hint":      event.category.lower() if event.category else None,
        "evidence":           [],
        "search_queries_used": [],
        "p_iterations":       [],
        "new_queries":        [],
        "iteration":          0,
        "trace_id":           str(uuid.uuid4()),
    }

    try:
        graph  = _get_graph()
        result = graph.invoke(state_in)
        p_yes  = float(result.get("p_final", 0.50))
        p_yes  = max(0.01, min(0.99, p_yes))
        rationale = result.get("rationale") or "PRIMA ensemble forecast."
        log.info("Forecast OK: %s  p_yes=%.3f", event.market_ticker, p_yes)
        return PredictionResponse(p_yes=p_yes, rationale=rationale)

    except Exception as e:
        log.error("Pipeline error for %s: %s — returning 0.50 fallback", event.market_ticker, e)
        return PredictionResponse(p_yes=0.50, rationale=f"Pipeline error: {e}")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Wake-up ping. ProphetHacks GETs this before starting evaluation."""
    return {"status": "ok", "agent": "PRIMA", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(event: EventRequest) -> PredictionResponse:
    """
    Receive a market event, run PRIMA pipeline, return p_yes.
    Real evaluation allows up to 10 minutes — pipeline typically takes 2-4 min.
    """
    log.info("POST /predict  market=%s  title=%s", event.market_ticker, event.title[:60])

    # Run pipeline in a thread so we don't block the event loop
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_prima, event)
    return result


# ── Local dev entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
