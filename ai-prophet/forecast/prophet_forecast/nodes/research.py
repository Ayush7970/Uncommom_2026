"""Layer 3: Research dispatcher — routes to category specialists or uses critic's new_queries."""

from __future__ import annotations

import logging
import os

from ..state import ForecastState

log = logging.getLogger(__name__)


def research_node(state: ForecastState) -> dict:
    """
    Dispatch to the appropriate research specialist.

    On first call (iteration=0): use category specialist with auto-generated queries.
    On refinement calls (iteration>0): use critic's new_queries directly via
    targeted Brave search, then MERGE with existing evidence.
    """
    category  = state.get("category", "culture")
    iteration = state.get("iteration", 0)
    new_queries = state.get("new_queries", [])

    # --- Refinement path: critic gave us specific queries to run ---
    if iteration > 0 and new_queries:
        log.info("Research [refinement iter=%d]: market=%s  queries=%s",
                 iteration, state["market_id"], new_queries)
        return _targeted_search(state, new_queries)

    # --- First pass: category specialist ---
    if category == "sports":
        from ..research.sports import research_sports
        return research_sports(state)

    from ..research.general import research_general
    return research_general(state)


def _targeted_search(state: ForecastState, queries: list[str]) -> dict:
    """Run specific queries from the critic and merge with existing evidence."""
    from ..tools.recursive_search import recursive_search

    api_key  = os.environ.get("BRAVE_API_KEY", "")
    snapshot = state["snapshot_ts"]
    category = state.get("category", "culture")

    new_evidence = []
    for query in queries[:2]:  # max 2 follow-up queries
        results = recursive_search(
            question=query,
            category=category,
            snapshot_ts=snapshot,
            api_key=api_key,
            max_iterations=1,       # single pass — no recursion in refinement
            results_per_query=3,
        )
        for item in results:
            body = item.get("text") or item.get("snippet") or ""
            if body:
                new_evidence.append({
                    "source":    item["url"],
                    "title":     item.get("title", ""),
                    "content":   body[:2000],
                    "iteration": state.get("iteration", 1),
                    "relevance": 0.9,
                })

    # Merge with existing evidence (don't discard what we already found)
    existing   = state.get("evidence", [])
    merged     = existing + new_evidence
    used_queries = list(state.get("search_queries_used", [])) + queries

    log.info("Targeted search: +%d new evidence items (total=%d)", len(new_evidence), len(merged))
    return {
        "evidence":            merged,
        "search_queries_used": used_queries,
        "new_queries":         [],   # clear so critic can set fresh ones next round
    }
