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

BRAVE_ENDPOINT  = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
MAX_ARTICLE_CHARS = 2500
FETCH_TIMEOUT  = 8
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

def _clean_for_search(question: str) -> str:
    """
    Strip binary-rewriting preamble so Brave gets a clean, natural query.

    Rewrites like "Did Péter Magyar win? (YES=Magyar, NO=Orbán) Original: Who became PM?"
    are reduced to "Who became PM?" for search purposes.
    """
    # Extract everything after "Original:" if present
    if "Original:" in question:
        return question.split("Original:", 1)[1].strip()
    # Strip "(YES = ..., NO = ...)" parenthetical if present
    if "(YES" in question:
        idx = question.find("(YES")
        if idx > 5:
            return question[:idx].strip().rstrip(",").strip()
    return question


def generate_queries(question: str, category: str, snapshot_ts: str) -> list[str]:
    """
    Generate 2-3 targeted search queries for a market question.
    Always uses the CLEAN original question (strips binary preamble).
    """
    snap_year = snapshot_ts[:4]

    # Always clean the question before building search queries
    clean_q = _clean_for_search(question).strip().rstrip("?")
    queries  = [f"{clean_q} {snap_year}"]   # primary: clean question + year

    cat = category.lower()
    ql  = clean_q.lower()

    if cat == "sports":
        if any(k in ql for k in ["tennis", "wta", "atp", "itf", "challenger"]):
            queries.append(f"{clean_q} ranking head-to-head {snap_year}")
            queries.append(f"{clean_q} WTA ATP player ranking recent form")
        elif any(k in ql for k in ["cricket", "test match", "county", "odi", "t20"]):
            queries.append(f"{clean_q} result winner {snap_year}")
            queries.append(f"{clean_q} cricket match head-to-head ranking")
        elif any(k in ql for k in ["ligue 1", "la liga", "serie a", "bundesliga",
                                    "premier league", "eredivisie", "liga portugal"]):
            queries.append(f"{clean_q} final standings champion {snap_year}")
            queries.append(f"{clean_q} league winner title {snap_year}")
        else:
            queries.append(f"{clean_q} result winner score {snap_year}")
            queries.append(f"{clean_q} standings prediction {snap_year}")

    elif cat == "politics":
        # For elections: look for RESULTS, not predictions
        if any(k in ql for k in ["prime minister", "president", "chancellor", "pm "]):
            queries.append(f"{clean_q} election result winner {snap_year}")
            queries.append(f"{clean_q} election outcome {snap_year}")
        elif any(k in ql for k in ["primary", "senate", "congress", "district"]):
            queries.append(f"{clean_q} primary result winner {snap_year}")
            queries.append(f"{clean_q} election results {snap_year}")
        elif any(k in ql for k in ["vote", "senator", "supreme court"]):
            queries.append(f"{clean_q} vote count tally {snap_year}")
            queries.append(f"{clean_q} final vote result {snap_year}")
        else:
            queries.append(f"{clean_q} election result {snap_year}")
            queries.append(f"{clean_q} winner outcome {snap_year}")

    elif cat == "culture":
        # Entertainment: look for reveals, finales, winners
        if any(k in ql for k in ["survivor", "masked singer", "bachelor", "big brother",
                                   "amazing race", "drag race", "idol"]):
            queries.append(f"{clean_q} winner finale reveal {snap_year}")
            queries.append(f"{clean_q} episode result eliminated {snap_year}")
        elif any(k in ql for k in ["netflix", "roast", "special", "host"]):
            queries.append(f"{clean_q} announced confirmed {snap_year}")
            queries.append(f"{clean_q} news reveal {snap_year}")
        elif any(k in ql for k in ["emmy", "oscar", "grammy", "tony", "award"]):
            queries.append(f"{clean_q} winner {snap_year}")
            queries.append(f"{clean_q} award result {snap_year}")
        else:
            queries.append(f"{clean_q} winner result {snap_year}")
            queries.append(f"{clean_q} confirmed news {snap_year}")

    elif cat == "finance":
        queries.append(f"{clean_q} forecast analyst {snap_year}")
        queries.append(f"{clean_q} market outlook probability")

    elif cat == "science_tech":
        queries.append(f"{clean_q} latest news {snap_year}")
        queries.append(f"{clean_q} update result announcement {snap_year}")

    else:
        queries.append(f"{clean_q} result winner {snap_year}")
        queries.append(f"{clean_q} confirmed news {snap_year}")

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
# Tavily search (primary — free tier, built for LLM agents)
# ---------------------------------------------------------------------------

def _tavily_search(query: str, api_key: str, n_results: int = 5) -> list[dict]:
    """
    Tavily returns pre-extracted content — no need to fetch articles separately.
    Free tier: 1000 searches/month.
    """
    if not api_key:
        return []
    try:
        resp = requests.post(
            TAVILY_ENDPOINT,
            json={
                "api_key":              api_key,
                "query":                query,
                "max_results":          n_results,
                "search_depth":         "advanced",
                "include_answer":       False,
                "include_raw_content":  False,
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("results", []):
            url     = item.get("url", "")
            title   = item.get("title", "")
            snippet = item.get("content", "")   # Tavily gives full content here
            if not url:
                continue
            if _is_low_quality(url, title, snippet):
                log.debug("Rejected low-quality source: %s", url[:60])
                continue
            results.append({"url": url, "title": title, "snippet": snippet,
                             "text": snippet})   # use content as text directly
        log.debug("Tavily returned %d results for '%s'", len(results), query[:60])
        return results
    except Exception as e:
        log.warning("Tavily search failed for '%s': %s", query[:60], e)
        return []


# ---------------------------------------------------------------------------
# Brave search (fallback — used if BRAVE_API_KEY set and Tavily fails)
# ---------------------------------------------------------------------------

def _brave_search(query: str, api_key: str, n_results: int = 5) -> list[dict]:
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
        log.debug("Brave returned %d results for '%s'", len(results), query[:60])
        return results
    except Exception as e:
        log.warning("Brave search failed for '%s': %s", query[:60], e)
        return []


def _search(query: str, n_results: int = 5) -> list[dict]:
    """
    Unified search: Tavily first (primary), Brave as fallback.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    brave_key  = os.environ.get("BRAVE_API_KEY", "")

    if tavily_key:
        results = _tavily_search(query, tavily_key, n_results)
        if results:
            return results
        log.info("Tavily returned no results — trying Brave fallback")

    if brave_key:
        return _brave_search(query, brave_key, n_results)

    log.warning("No search API key set (TAVILY_API_KEY or BRAVE_API_KEY)")
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
    # Check at least one search key is available
    if not os.environ.get("TAVILY_API_KEY") and not os.environ.get("BRAVE_API_KEY") and not api_key:
        log.warning("No search API key (TAVILY_API_KEY or BRAVE_API_KEY) — returning empty evidence")
        return []

    all_evidence: list[dict] = []
    queries = generate_queries(question, category, snapshot_ts)

    for iteration in range(max_iterations):
        if iteration == 0:
            batch = queries
        else:
            followup = generate_followup_query(question, category, all_evidence)
            if not followup:
                log.info("No follow-up query needed — stopping at iteration %d", iteration)
                break
            batch = [followup]

        log.info("Search iteration %d: %d quer%s", iteration + 1, len(batch),
                 "y" if len(batch) == 1 else "ies")

        for query in batch:
            log.info("  Searching: '%s'", query[:80])
            results = _search(query, n_results=results_per_query)

            for r in results:
                # Tavily pre-extracts content in r["text"] — skip expensive article fetch
                text = r.get("text") or ""
                if not text:
                    text = _fetch_article(r["url"])  # Brave fallback: fetch article
                if text or r.get("snippet"):
                    all_evidence.append({
                        "query":     query,
                        "url":       r["url"],
                        "title":     r["title"],
                        "snippet":   r.get("snippet", ""),
                        "text":      text,
                        "iteration": iteration,
                    })
                    if text:
                        break  # one good article per query is enough

            time.sleep(0.2)

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
