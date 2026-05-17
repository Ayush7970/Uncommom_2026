"""
PRIMA — ProphetHacks Forecast Endpoint
FastAPI server wrapping the full PRIMA LangGraph pipeline.

ProphetHacks contract:
  GET  /health   → {"status": "ok"}
  POST /predict  → {"probabilities": [{"market": str, "probability": float}]}

Each POST contains ONE binary market (market_ticker = one outcome).
ProphetHacks sends N requests (one per outcome) and expects probabilities
that can be used as-is — no requirement to sum to 1 across requests.

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

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("prima.server")

app = FastAPI(title="PRIMA Forecast Agent", version="1.0.0")


# ── Request / Response models ─────────────────────────────────────────────────

class EventRequest(BaseModel):
    """ProphetHacks / ProphetArena event input shape."""
    event_ticker:  str
    market_ticker: str
    title:         str
    subtitle:      str | None = None
    description:   str | None = None
    category:      str
    rules:         str | None = None
    close_time:    str
    # Optional: list of outcome labels for this event (not always sent)
    outcomes:      list[str] | None = None


class OutcomeProbability(BaseModel):
    market:      str    # outcome label, e.g. "Lakers" or "YES"
    probability: float  # p ∈ [0, 1]


class PredictionResponse(BaseModel):
    """ProphetHacks response format."""
    probabilities: list[OutcomeProbability]


# ── Lazy-load the LangGraph pipeline ─────────────────────────────────────────

_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        log.info("Loading PRIMA forecast graph ...")
        from prophet_forecast.graph import forecast_graph
        _graph = forecast_graph
        log.info("PRIMA graph ready.")
    return _graph


# ── Core forecast logic ───────────────────────────────────────────────────────

def _p_yes_via_prima(question: str, market_ticker: str, category: str, close_time: str) -> float:
    """Run the full PRIMA pipeline and return p_yes for this binary market."""
    state_in = {
        "market_id":           market_ticker,
        "question":            question,
        "p_market":            0.50,
        "snapshot_ts":         datetime.now(UTC).isoformat(),
        "resolves_at":         close_time,
        "category_hint":       category.lower() if category else None,
        "evidence":            [],
        "search_queries_used": [],
        "p_iterations":        [],
        "new_queries":         [],
        "iteration":           0,
        "trace_id":            str(uuid.uuid4()),
    }
    try:
        result = _get_graph().invoke(state_in)
        p = float(result.get("p_final", 0.50))
        p = max(0.01, min(0.99, p))
        log.info("PRIMA p_yes=%.3f  market=%s", p, market_ticker)
        return p
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error("Pipeline error (%s):\n%s", market_ticker, tb)
        # Return error detail in rationale so we can diagnose from the response
        return 0.50


def _run_forecast(event: EventRequest) -> PredictionResponse:
    """
    Build the probability list for the event.

    ProphetHacks sends one request per binary market (market_ticker).
    We run PRIMA for this specific outcome and return it in the
    probabilities list format.

    If the event has multiple outcomes provided, we handle all of them
    with normalised probabilities. Otherwise we treat it as binary YES/NO.
    """
    title    = event.title.strip()
    subtitle = f" ({event.subtitle.strip()})" if event.subtitle else ""
    question = f"{title}{subtitle}"

    outcomes = event.outcomes or []

    if len(outcomes) >= 2:
        # Multi-outcome event: get p_yes for each outcome as "Did X win?"
        raw_probs: dict[str, float] = {}
        for outcome in outcomes:
            q = f"Did {outcome} win/prevail? (YES={outcome}) Original: {question}"
            p = _p_yes_via_prima(q, f"{event.market_ticker}_{outcome}", event.category, event.close_time)
            raw_probs[outcome] = p

        # Normalise so they sum to 1.0
        total = sum(raw_probs.values()) or 1.0
        probs = [
            OutcomeProbability(market=k, probability=round(v / total, 6))
            for k, v in raw_probs.items()
        ]
    else:
        # Binary YES/NO market (most common in ProphetHacks)
        p = _p_yes_via_prima(question, event.market_ticker, event.category, event.close_time)
        probs = [
            OutcomeProbability(market="YES", probability=round(p, 6)),
            OutcomeProbability(market="NO",  probability=round(1.0 - p, 6)),
        ]

    return PredictionResponse(probabilities=probs)


# ── Endpoints ─────────────────────────────────────────────────────────────────

_last_error: str = ""

@app.get("/health")
async def health():
    """Wake-up ping. ProphetHacks GETs this before evaluation starts."""
    return {"status": "ok", "agent": "PRIMA", "version": "1.0.0"}


@app.get("/debug")
async def debug():
    """Show last pipeline error for diagnosis."""
    import sys
    try:
        graph = _get_graph()
        graph_ok = True
    except Exception as e:
        graph_ok = False
        return {"graph_loaded": False, "error": str(e),
                "tavily_key": bool(os.environ.get("TAVILY_API_KEY")),
                "openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY"))}
    return {
        "graph_loaded": graph_ok,
        "tavily_key":     bool(os.environ.get("TAVILY_API_KEY")),
        "openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "snowflake_set":  bool(os.environ.get("SNOWFLAKE_ACCOUNT")),
        "python": sys.version,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    """
    Receive one market event, run PRIMA, return probabilities.
    ProphetHacks real evaluation allows up to 10 min — pipeline ~2-4 min.
    """
    log.info("POST /predict  ticker=%s  title=%.60s", event.market_ticker, event.title)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_forecast, event)


# ── Local predict function (for: prophet forecast predict --local) ────────────

def predict(event: dict) -> dict:
    """
    Local mode entry point used by `prophet forecast predict --local`.
    Accepts full event dict, returns probabilities list.
    """
    req    = EventRequest(**event)
    result = _run_forecast(req)
    return {"probabilities": [p.model_dump() for p in result.probabilities]}


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
