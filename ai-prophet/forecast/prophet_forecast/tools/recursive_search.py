"""
Recursive search tool — Brave API + article extraction + auto query generation.

Flow per call:
  1. Generate 2-3 targeted queries from question + category
  2. Search Brave for each query
  3. Extract article text (trafilatura)
  4. Optionally generate 1 follow-up query to fill evidence gaps
  5. Return structured evidence list

Temporal safety: all results are tagged with retrieval time.
Caller must pass snapshot_ts and treat any result as "info available as of snapshot_ts".
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime

import requests
import trafilatura

log = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_ARTICLE_CHARS = 2500
FETCH_TIMEOUT = 8
SEARCH_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Source quality deny-list (§D.3: forecast bots actively degrade Brier)
# ---------------------------------------------------------------------------
_DENY_DOMAINS = {
    "priceforecastbot.com", "walletinvestor.com", "cryptopredictions.com",
    "longforecast.com", "coinpriceforecast.com", "stockforecasttoday.com",
    "priceprediction.net", "digitalcoinprice.com", "telegaon.com",
    "ai-forex-trading.com", "fxleaders.com",
}
_DENY_PATTERNS = [
    "ai predicts", "chatgpt forecasts", "ai forecast", "bot predicts",
    "price prediction 2025", "price prediction 2026", "ai-generated",
]

def _is_low_quality(url: str, title: str, snippet: str) -> bool:
    """Return True if this source should be rejected (§D.3)."""
    url_lower = url.lower()
    for domain in _DENY_DOMAINS:
        if domain in url_lower:
            return True
    combined = (title + " " + snippet).lower()
    return any(pat in combined for pat in _DENY_PATTERNS)

# ---------------------------------------------------------------------------
# Query generation (category-aware)
# ---------------------------------------------------------------------------

def generate_queries(question: str, category: str, snapshot_ts: str) -> list[str]:
    """
    Generate 2-3 targeted search queries for a market question.
    Injects date context from snapshot_ts to bias toward temporally relevant results.
    """
    snap_date = snapshot_ts[:10]  # YYYY-MM-DD
    snap_year = snapshot_ts[:4]

    q = question.strip().rstrip("?")
    queries = [f"{q} {snap_year}"]  # primary query always has the year

    cat = category.lower()
    ql = question.lower()

    if cat == "sports":
        ql = question.lower()
        if "tennis" in ql or "wta" in ql or "atp" in ql or "itf" in ql:
            # For tennis: ranking + head-to-head are what matter
            queries.append(f"{q} ranking head-to-head form {snap_year}")
            queries.append(f"{q} WTA ATP ranking recent results")
        elif "cricket" in ql or "test match" in ql or "county" in ql:
            queries.append(f"{q} team ranking recent form {snap_year}")
            queries.append(f"{q} cricket match preview head-to-head")
        elif "who won" in ql or "who will win" in ql:
            # Generic "who won" — search for result or standings
            queries.append(f"{q} result winner {snap_year}")
            queries.append(f"{q} standings prediction {snap_year}")
        else:
            queries.append(f"{q} result score odds {snap_year}")
    elif cat == "finance":
        queries.append(f"{q} forecast analyst prediction {snap_date}")
        queries.append(f"{q} market outlook probability")
    elif cat == "politics":
        queries.append(f"{q} poll survey forecast {snap_year}")
        queries.append(f"{q} election prediction probability")
    elif cat == "science_tech":
        queries.append(f"{q} latest news update {snap_year}")
        queries.append(f"{q} probability estimate expert opinion")
    else:  # culture / general
        queries.append(f"{q} confirmed rumor news {snap_year}")
        queries.append(f"{q} odds probability")

    return queries[:3]


def generate_followup_query(question: str, category: str, evidence_so_far: list[dict]) -> str | None:
    """
    Given what we've found so far, generate one follow-up query to fill gaps.
    Simple heuristic — no LLM needed here.
    """
    if not evidence_so_far:
        return f"{question} analysis expert"

    # Check if we got meaningful content
    total_chars = sum(len(e.get("text", "")) for e in evidence_so_far)
    if total_chars < 500:
        return f"{question} detailed analysis"

    return None


# ---------------------------------------------------------------------------
# Brave search
# ---------------------------------------------------------------------------

def _brave_search(query: str, api_key: str, n_results: int = 3) -> list[dict]:
    if not api_key:
        return []
    try:
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": n_results, "country": "US", "search_lang": "en"},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("web", {}).get("results", []):
            url     = item.get("url", "")
            title   = item.get("title", "")
            snippet = item.get("description", "")
            if not url:
                continue
            if _is_low_quality(url, title, snippet):
                log.debug("Rejected low-quality source: %s", url[:60])
                continue
            results.append({"url": url, "title": title, "snippet": snippet})
        log.debug("Brave returned %d quality results for '%s'", len(results), query[:60])
        return results
    except Exception as e:
        log.warning("Brave search failed for '%s': %s", query[:60], e)
        return []


# ---------------------------------------------------------------------------
# Article extraction
# ---------------------------------------------------------------------------

def _fetch_article(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ProphetForecast/1.0)"},
        )
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, no_fallback=True)
        if text:
            return text[:MAX_ARTICLE_CHARS]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def recursive_search(
    question: str,
    category: str,
    snapshot_ts: str,
    api_key: str | None = None,
    max_iterations: int = 2,
    results_per_query: int = 5,
) -> list[dict]:
    """
    Run recursive web search for a market question.

    Returns a list of evidence dicts, each with:
        query       str   the search query used
        url         str   source URL
        title       str   page title
        snippet     str   search snippet
        text        str   extracted article text (may be empty)
        iteration   int   which search round this came from
    """
    key = api_key or os.environ.get("BRAVE_API_KEY", "")
    if not key:
        log.warning("No BRAVE_API_KEY — returning empty evidence")
        return []

    all_evidence: list[dict] = []
    queries = generate_queries(question, category, snapshot_ts)

    for iteration in range(max_iterations):
        if iteration == 0:
            batch = queries
        else:
            # Follow-up query based on what we found
            followup = generate_followup_query(question, category, all_evidence)
            if not followup:
                log.info("No follow-up query needed — stopping at iteration %d", iteration)
                break
            batch = [followup]

        log.info("Search iteration %d: %d quer%s", iteration + 1, len(batch),
                 "y" if len(batch) == 1 else "ies")

        for query in batch:
            log.info("  Searching: '%s'", query[:80])
            results = _brave_search(query, key, n_results=results_per_query)

            for r in results:
                text = _fetch_article(r["url"])
                if text or r["snippet"]:
                    all_evidence.append({
                        "query":     query,
                        "url":       r["url"],
                        "title":     r["title"],
                        "snippet":   r["snippet"],
                        "text":      text,
                        "iteration": iteration,
                    })
                    if text:
                        break  # One good article per query is enough

            time.sleep(0.3)  # be polite to Brave

    log.info("Search complete: %d evidence items across %d queries",
             len(all_evidence), len(queries))
    return all_evidence


def summarise_evidence(evidence: list[dict]) -> str:
    """Flatten evidence into a single string for LLM consumption."""
    if not evidence:
        return "No web search results available."

    parts = []
    for i, e in enumerate(evidence, 1):
        body = e.get("text") or e.get("snippet") or "(no content)"
        parts.append(
            f"[Source {i}] {e.get('title', 'Unknown')}\n"
            f"URL: {e.get('url', '')}\n"
            f"{body[:1000]}"
        )
    return "\n\n---\n\n".join(parts)
