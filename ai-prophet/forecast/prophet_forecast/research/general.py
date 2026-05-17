"""General research specialist — recursive Brave search for all non-sports categories."""

from __future__ import annotations

import logging
import os

from ..state import ForecastState
from ..tools.recursive_search import recursive_search, summarise_evidence

log = logging.getLogger(__name__)


def research_general(state: ForecastState) -> dict:
    """
    General research specialist using recursive Brave search.
    Used for: finance, politics, science_tech, culture.
    """
    question    = state["question"]
    category    = state.get("category", "culture")
    snapshot_ts = state["snapshot_ts"]
    api_key     = os.environ.get("BRAVE_API_KEY", "")

    log.info("General research: market=%s  category=%s", state["market_id"], category)

    raw_evidence = recursive_search(
        question=question,
        category=category,
        snapshot_ts=snapshot_ts,
        api_key=api_key,
        max_iterations=2,
        results_per_query=3,
    )

    # Convert raw search results into structured evidence items
    evidence = []
    queries_used = []

    for item in raw_evidence:
        body = item.get("text") or item.get("snippet") or ""
        if not body:
            continue
        evidence.append({
            "source":    item["url"],
            "title":     item.get("title", ""),
            "content":   body[:2000],
            "iteration": item.get("iteration", 0),
            "relevance": 0.8 if item.get("text") else 0.5,
        })
        q = item.get("query", "")
        if q and q not in queries_used:
            queries_used.append(q)

    log.info("General research complete: %d evidence items  %d queries",
             len(evidence), len(queries_used))

    return {
        "evidence":            evidence,
        "search_queries_used": queries_used,
    }
