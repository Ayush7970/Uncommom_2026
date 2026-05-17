"""Layer 2: Dual Router — LLM category classifier + deterministic time bucket."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import yaml
from pydantic import BaseModel

from ..state import ForecastState

log = logging.getLogger(__name__)

# Load config once at import time
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")

with open(os.path.join(_CONFIG_DIR, "categories.yaml")) as f:
    _CATEGORIES_CFG = yaml.safe_load(f)["categories"]

with open(os.path.join(_CONFIG_DIR, "time_buckets.yaml")) as f:
    _BUCKETS_CFG = yaml.safe_load(f)["time_buckets"]

_VALID_CATEGORIES = list(_CATEGORIES_CFG.keys())

# ---------------------------------------------------------------------------
# Structured output schema for LLM router
# ---------------------------------------------------------------------------

class CategoryClassification(BaseModel):
    category: str       # one of: sports | finance | politics | science_tech | culture
    confidence: float   # 0.0 – 1.0
    reasoning: str      # one sentence


# ---------------------------------------------------------------------------
# LLM classifier (Haiku — cheap + fast)
# ---------------------------------------------------------------------------

_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from langchain_anthropic import ChatAnthropic
        model = os.environ.get("ROUTER_MODEL", "claude-haiku-4-5-20251001")
        _llm_client = ChatAnthropic(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            max_tokens=200,
            temperature=0,
        ).with_structured_output(CategoryClassification)
        return _llm_client
    except Exception as e:
        log.warning("Could not initialise LLM router: %s — falling back to keywords", e)
        return None


_ROUTER_SYSTEM = """\
You are a question classifier for a prediction market forecasting system.
Classify the question into exactly ONE category:
- sports: games, matches, championships, athlete performance
- finance: prices, rates, economic indicators, crypto, stocks
- politics: elections, legislation, geopolitical events, government actions
- science_tech: FDA approvals, product launches, scientific milestones
- culture: movies, awards, entertainment, casting, social trends
"""


def _llm_classify(question: str) -> CategoryClassification | None:
    llm = _get_llm()
    if llm is None:
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        result = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=f"Classify this question: {question}"),
        ])
        if result.category not in _VALID_CATEGORIES:
            log.warning("LLM returned unknown category '%s' — falling back", result.category)
            return None
        return result
    except Exception as e:
        log.warning("LLM classification failed: %s — falling back to keywords", e)
        return None


# ---------------------------------------------------------------------------
# Keyword fallback classifier
# ---------------------------------------------------------------------------

def _keyword_classify(question: str, hint: str | None) -> str:
    if hint and hint in _CATEGORIES_CFG:
        return hint
    q = question.lower()
    for cat, cfg in _CATEGORIES_CFG.items():
        if any(kw in q for kw in cfg["keywords"]):
            return cat
    return "culture"


# ---------------------------------------------------------------------------
# Time bucket (deterministic)
# ---------------------------------------------------------------------------

def compute_time_bucket(snapshot_ts: str, resolves_at: str) -> tuple[str, float, float]:
    """Return (bucket_name, hours_to_resolution, w_t)."""
    snap   = datetime.fromisoformat(snapshot_ts.replace("Z", "+00:00"))
    resolv = datetime.fromisoformat(resolves_at.replace("Z", "+00:00"))
    hours  = max(0.0, (resolv - snap).total_seconds() / 3600)

    for bucket, cfg in _BUCKETS_CFG.items():
        min_h = cfg["min_hours"]
        max_h = cfg["max_hours"]
        if max_h is None:
            if hours >= min_h:
                return bucket, hours, cfg["w_t"]
        else:
            if min_h <= hours < max_h:
                return bucket, hours, cfg["w_t"]

    return "urgent", hours, _BUCKETS_CFG["urgent"]["w_t"]


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def router_node(state: ForecastState) -> dict:
    """Classify category via LLM (Haiku) + compute time bucket deterministically."""
    question = state["question"]
    hint     = state.get("category_hint")

    # Try LLM first, fall back to keywords
    classification = _llm_classify(question)
    if classification:
        category   = classification.category
        confidence = classification.confidence
        reasoning  = classification.reasoning
        source     = "llm"
    else:
        category   = _keyword_classify(question, hint)
        confidence = 0.7
        reasoning  = "keyword fallback"
        source     = "keywords"

    bucket, hours, w_t = compute_time_bucket(state["snapshot_ts"], state["resolves_at"])

    log.info(
        "Router [%s]: market=%s  category=%s (conf=%.2f)  bucket=%s  hours=%.1f  w_t=%.2f",
        source, state["market_id"], category, confidence, bucket, hours, w_t,
    )

    return {
        "category":            category,
        "time_bucket":         bucket,
        "hours_to_resolution": hours,
        "w_t":                 w_t,
    }
