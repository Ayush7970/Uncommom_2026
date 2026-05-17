"""General research specialist — recursive Brave search for all non-sports categories."""

from __future__ import annotations

import logging
import os
import re
import time

from ..state import ForecastState
from ..tools.recursive_search import recursive_search, _clean_for_search

log = logging.getLogger(__name__)


def _extract_candidates(binary_question: str) -> tuple[str | None, str | None]:
    """
    Pull YES and NO candidate names out of the binary-rewritten question.
    e.g. "Did Péter Magyar win? (YES = Péter Magyar, NO = Viktor Orbán)" →
         yes_name="Péter Magyar", no_name="Viktor Orbán"
    """
    yes_name = no_name = None
    m = re.search(r"YES\s*=\s*([^,\)\n]+)", binary_question)
    if m:
        yes_name = m.group(1).strip().rstrip("wins").strip()
    m = re.search(r"NO\s*=\s*([^,\)\n]+)", binary_question)
    if m:
        no_name = m.group(1).strip().rstrip("wins").strip()
    return yes_name, no_name


def _politics_result_searches(
    binary_question: str,
    clean_question: str,
    snap_year: str,
    api_key: str,
    snapshot_ts: str,
) -> list[dict]:
    """
    For politics markets: run targeted result searches using candidate/coalition names.
    Searches for "[candidate] won [year]" to find the actual election result,
    not pre-election polls or incumbency history.
    """
    yes_name, no_name = _extract_candidates(binary_question)
    snap_year = snap_year[:4]

    # Build name-specific result queries
    result_queries = []
    if yes_name and len(yes_name) > 3:
        result_queries.append(f'"{yes_name}" won elected winner {snap_year}')
        result_queries.append(f'"{yes_name}" election result {snap_year}')
    if no_name and len(no_name) > 3:
        result_queries.append(f'"{no_name}" won elected winner {snap_year}')
    # Fallback: clean question with "result winner"
    result_queries.append(f"{clean_question} official result winner {snap_year}")

    log.info("Politics result search: yes='%s'  no='%s'  queries=%d",
             yes_name, no_name, len(result_queries))

    all_evidence = []
    seen_urls: set[str] = set()

    for query in result_queries[:4]:
        results = recursive_search(
            question=query,
            category="politics",
            snapshot_ts=snapshot_ts,
            api_key=api_key,
            max_iterations=1,
            results_per_query=3,
        )
        for item in results:
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            body = item.get("text") or item.get("snippet") or ""
            if body:
                all_evidence.append({
                    "source":    url,
                    "title":     item.get("title", ""),
                    "content":   f"[RESULT SEARCH] {body[:2000]}",
                    "iteration": 0,
                    "relevance": 0.95,  # high relevance — direct result search
                    "query":     query,
                })
                if item.get("text"):
                    break  # one good article per query is enough
        time.sleep(0.2)

    log.info("Politics result search complete: %d items", len(all_evidence))
    return all_evidence


def research_general(state: ForecastState) -> dict:
    """
    General research specialist using recursive Brave search.
    Used for: finance, politics, science_tech, culture.
    """
    raw_question = state["question"]
    search_question = _clean_for_search(raw_question)

    category    = state.get("category", "culture")
    snapshot_ts = state["snapshot_ts"]
    snap_year   = snapshot_ts[:4]
    api_key     = os.environ.get("BRAVE_API_KEY", "")

    log.info("General research: market=%s  category=%s  query_base='%s'",
             state["market_id"], category, search_question[:60])

    # Standard broad search
    raw_evidence = recursive_search(
        question=search_question,
        category=category,
        snapshot_ts=snapshot_ts,
        api_key=api_key,
        max_iterations=2,
        results_per_query=5,
    )

    evidence = []
    queries_used = []
    seen_urls: set[str] = set()

    for item in raw_evidence:
        url = item.get("url", "")
        body = item.get("text") or item.get("snippet") or ""
        if not body or url in seen_urls:
            continue
        seen_urls.add(url)
        evidence.append({
            "source":    url,
            "title":     item.get("title", ""),
            "content":   body[:2000],
            "iteration": item.get("iteration", 0),
            "relevance": 0.8 if item.get("text") else 0.5,
        })
        q = item.get("query", "")
        if q and q not in queries_used:
            queries_used.append(q)

    # ── Culture: run an entertainment result search ─────────────────────────
    # TV show winners, Netflix announcements etc. are findable via web search.
    if category == "culture":
        yes_name, _ = _extract_candidates(raw_question)
        culture_queries = []
        snap_yr = snap_year

        ql = search_question.lower()
        if yes_name and len(yes_name) > 3:
            culture_queries.append(f'"{yes_name}" winner revealed {snap_yr}')
            culture_queries.append(f'"{yes_name}" wins {search_question[:50]} {snap_yr}')
        if any(k in ql for k in ["survivor", "masked singer", "big brother", "bachelor", "idol"]):
            culture_queries.append(f"{search_question} winner finale {snap_yr}")
        elif any(k in ql for k in ["netflix", "roast", "special"]):
            culture_queries.append(f"{search_question} announced confirmed {snap_yr}")
        elif any(k in ql for k in ["tournament", "competition", "championship"]):
            culture_queries.append(f"{search_question} winner champion {snap_yr}")
        else:
            culture_queries.append(f"{search_question} result winner {snap_yr}")

        seen_urls_culture: set[str] = set()
        culture_evidence = []
        for query in culture_queries[:3]:
            results = recursive_search(
                question=query, category="culture",
                snapshot_ts=snapshot_ts, api_key=api_key,
                max_iterations=1, results_per_query=3,
            )
            for item in results:
                url = item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls_culture.add(url)
                body = item.get("text") or item.get("snippet") or ""
                if body:
                    culture_evidence.append({
                        "source": url, "title": item.get("title", ""),
                        "content": f"[RESULT SEARCH] {body[:2000]}",
                        "iteration": 0, "relevance": 0.95,
                        "query": query,
                    })
                    if item.get("text"):
                        break
            import time; time.sleep(0.2)

        evidence = culture_evidence + evidence
        queries_used += [e.get("query", "") for e in culture_evidence if e.get("query")]
        log.info("Culture result search: +%d items", len(culture_evidence))

    # ── Politics: run an extra name-specific result search ──────────────────
    # The standard search often finds incumbency history / polls instead of
    # the actual 2026 result. We run dedicated "[candidate] won [year]" queries
    # to find the specific election outcome.
    if category == "politics":
        result_evidence = _politics_result_searches(
            binary_question=raw_question,
            clean_question=search_question,
            snap_year=snap_year,
            api_key=api_key,
            snapshot_ts=snapshot_ts,
        )
        # Prepend result evidence so synthesis sees it first (higher priority)
        evidence = result_evidence + evidence
        queries_used += [e.get("query", "") for e in result_evidence if e.get("query")]

    log.info("General research complete: %d evidence items  %d queries",
             len(evidence), len(queries_used))

    return {
        "evidence":            evidence,
        "search_queries_used": queries_used,
    }
