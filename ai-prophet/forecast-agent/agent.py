"""
Uprising Forecast Agent
-----------------------
A web-search-augmented forecasting agent for Prophet Arena.

Flow per event:
  1. Generate smart search queries from the event
  2. Search Brave API for real-world information
  3. Extract article text via trafilatura
  4. Ask Claude to produce a calibrated p_yes with the search context

Supports two modes:
  - Server mode:  python agent.py  (FastAPI on port 8000)
  - Local mode:   prophet forecast predict --local forecast_agent.agent

Usage:
  # Server mode
  python agent.py

  # Local mode (no server needed)
  prophet forecast predict --events events.json --local forecast_agent.agent --output predictions.json
"""

from __future__ import annotations

import json
import logging
import os
import time

import anthropic
import requests
import trafilatura
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("forecast-agent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
FORECAST_MODEL = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-6")

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_SEARCH_RESULTS = 3
MAX_ARTICLE_CHARS = 2000
MAX_QUERIES = 2

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _generate_queries(event: EventRequest) -> list[str]:
    """Generate targeted search queries based on the event category and title."""
    title = event.title
    category = event.category.lower()

    queries = [title]

    if "sport" in category or any(k in title.lower() for k in ["nba", "nfl", "nhl", "mlb", "soccer", "cup", "championship", "playoff"]):
        queries.append(f"{title} latest result score 2026")
    elif "politic" in category or any(k in title.lower() for k in ["election", "senate", "president", "congress", "poll"]):
        queries.append(f"{title} poll forecast 2026")
    elif "crypto" in category or any(k in title.lower() for k in ["bitcoin", "eth", "price"]):
        queries.append(f"{title} price prediction market")
    elif "econ" in category or any(k in title.lower() for k in ["fed", "rate", "gdp", "inflation"]):
        queries.append(f"{title} forecast economist")
    else:
        queries.append(f"{title} prediction odds probability")

    return queries[:MAX_QUERIES]


def _brave_search(query: str) -> list[dict]:
    """Call Brave Search API and return a list of results with url + snippet."""
    if not BRAVE_API_KEY:
        return []
    try:
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers={
                "X-Subscription-Token": BRAVE_API_KEY,
                "Accept": "application/json",
            },
            params={"q": query, "count": MAX_SEARCH_RESULTS, "country": "US", "search_lang": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("web", {}).get("results", []):
            if item.get("url"):
                results.append({"url": item["url"], "snippet": item.get("description", "")})
        return results
    except Exception as e:
        log.warning("Brave search failed for '%s': %s", query, e)
        return []


def _fetch_article(url: str) -> str:
    """Fetch and extract main text from a URL using trafilatura."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, no_fallback=True)
        if text:
            return text[:MAX_ARTICLE_CHARS]
    except Exception:
        pass
    return ""


def gather_context(event: EventRequest) -> str:
    """Run web searches and extract article text. Returns a combined context string."""
    if not BRAVE_API_KEY:
        log.info("No BRAVE_API_KEY — forecasting without web search")
        return ""

    queries = _generate_queries(event)
    context_parts: list[str] = []

    for query in queries:
        log.info("Searching: '%s'", query)
        results = _brave_search(query)
        for result in results:
            text = _fetch_article(result["url"])
            if text:
                context_parts.append(f"Source: {result['url']}\n{text}")
                break
            elif result["snippet"]:
                context_parts.append(f"Snippet: {result['snippet']}")

        time.sleep(0.3)

    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Claude forecasting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert superforecaster producing calibrated probability estimates.

Your ONLY task: estimate the probability this event resolves YES.

CALIBRATION RULES:
- Base rates matter: what usually happens in similar situations?
- Weight search evidence by recency and reliability.
- Don't be overconfident. Extremes (p < 0.08 or p > 0.92) require very strong evidence.
- If a market price is given, it reflects many traders' views — deviate only if your research gives specific new information.
- When uncertain, stay close to 0.5 or the market price.

COMMON MISTAKES TO AVOID:
- Overconfidence from generic information ("X is a good team" is NOT strong evidence)
- Ignoring base rates
- Treating recent news as more predictive than it is

Output ONLY valid JSON — no other text:
{"p_yes": <float 0.01-0.99>, "rationale": "<2-3 sentences explaining your key reasoning>"}"""


def _build_user_prompt(event: EventRequest, context: str) -> str:
    parts = [f"Event: {event.title}"]
    if event.subtitle:
        parts.append(f"Subtitle: {event.subtitle}")
    if event.description:
        parts.append(f"Description: {event.description[:500]}")
    if event.rules:
        parts.append(f"Resolution rules: {event.rules[:300]}")
    parts.append(f"Category: {event.category}")
    parts.append(f"Closes: {event.close_time}")

    if context:
        parts.append(f"\nWEB SEARCH FINDINGS:\n{context[:3000]}")
    else:
        parts.append("\n(No web search results available — use your training knowledge.)")

    parts.append("\nWhat is your calibrated probability this resolves YES? Respond with JSON only.")
    return "\n".join(parts)


_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment or .env")
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def forecast(event: EventRequest) -> PredictionResponse:
    """Full pipeline: search → context → Claude forecast."""
    log.info("Forecasting: %s", event.market_ticker)

    context = gather_context(event)

    client = _get_client()
    response = client.messages.create(
        model=FORECAST_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(event, context)}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)
    p = max(0.01, min(0.99, float(data["p_yes"])))
    rationale = data.get("rationale", "")
    log.info("  %s → p_yes=%.3f | %s", event.market_ticker, p, rationale[:80])
    return PredictionResponse(p_yes=p, rationale=rationale)


# ---------------------------------------------------------------------------
# Local predict function (used by: prophet forecast predict --local)
# ---------------------------------------------------------------------------

def predict(event: dict) -> dict:
    """Entry point for --local mode."""
    req = EventRequest(**event)
    resp = forecast(req)
    return {"p_yes": resp.p_yes, "rationale": resp.rationale}


# ---------------------------------------------------------------------------
# FastAPI server (used by: prophet forecast predict --agent-url)
# ---------------------------------------------------------------------------

app = FastAPI(title="Uprising Forecast Agent")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    return forecast(event)


@app.get("/health")
async def health():
    return {"status": "ok", "model": FORECAST_MODEL, "search": bool(BRAVE_API_KEY)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting Uprising Forecast Agent")
    log.info("Model: %s", FORECAST_MODEL)
    log.info("Web search: %s", "enabled" if BRAVE_API_KEY else "disabled (set BRAVE_API_KEY)")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
