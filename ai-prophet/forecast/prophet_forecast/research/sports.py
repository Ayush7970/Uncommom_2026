"""Sports research specialist — ESPN API + pre-seeded Elo ratings."""

from __future__ import annotations

import logging
import os
import re

import requests

from ..state import ForecastState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-seeded Elo ratings (2025-26 season estimates)
# Used as fallback when ESPN is unreachable or team not found
# ---------------------------------------------------------------------------

_NBA_ELO: dict[str, float] = {
    "celtics": 1650, "boston": 1650,
    "thunder": 1620, "oklahoma": 1620, "okc": 1620,
    "nuggets": 1580, "denver": 1580,
    "timberwolves": 1560, "minnesota": 1560, "wolves": 1560,
    "cavaliers": 1550, "cleveland": 1550, "cavs": 1550,
    "warriors": 1520, "golden state": 1520, "gsw": 1520,
    "lakers": 1510, "los angeles lakers": 1510, "lal": 1510,
    "suns": 1490, "phoenix": 1490,
    "bucks": 1480, "milwaukee": 1480,
    "76ers": 1470, "sixers": 1470, "philadelphia": 1470,
    "clippers": 1460, "la clippers": 1460,
    "heat": 1450, "miami": 1450,
    "knicks": 1440, "new york": 1440,
    "mavericks": 1430, "dallas": 1430, "mavs": 1430,
    "kings": 1420, "sacramento": 1420,
    "pacers": 1410, "indiana": 1410,
    "hawks": 1400, "atlanta": 1400,
    "bulls": 1390, "chicago": 1390,
    "magic": 1380, "orlando": 1380,
    "nets": 1370, "brooklyn": 1370,
    "raptors": 1360, "toronto": 1360,
    "grizzlies": 1350, "memphis": 1350,
    "pelicans": 1340, "new orleans": 1340,
    "jazz": 1330, "utah": 1330,
    "rockets": 1320, "houston": 1320,
    "spurs": 1310, "san antonio": 1310,
    "trail blazers": 1300, "portland": 1300, "blazers": 1300,
    "pistons": 1290, "detroit": 1290,
    "wizards": 1280, "washington": 1280,
    "hornets": 1270, "charlotte": 1270,
}

_NHL_ELO: dict[str, float] = {
    "avalanche": 1600, "colorado": 1600,
    "panthers": 1580, "florida": 1580,
    "rangers": 1560, "new york rangers": 1560,
    "bruins": 1540, "boston": 1540,
    "oilers": 1520, "edmonton": 1520,
    "golden knights": 1500, "vegas": 1500,
    "hurricanes": 1490, "carolina": 1490,
    "stars": 1480, "dallas": 1480,
    "kings": 1470, "los angeles": 1470,
    "maple leafs": 1460, "toronto": 1460,
    "jets": 1450, "winnipeg": 1450,
    "lightning": 1440, "tampa bay": 1440,
}

_NFL_ELO: dict[str, float] = {
    "chiefs": 1700, "kansas city": 1700,
    "ravens": 1650, "baltimore": 1650,
    "niners": 1640, "49ers": 1640, "san francisco": 1640,
    "bills": 1620, "buffalo": 1620,
    "eagles": 1600, "philadelphia": 1600,
    "lions": 1580, "detroit": 1580,
    "cowboys": 1560, "dallas": 1560,
    "packers": 1540, "green bay": 1540,
    "bengals": 1520, "cincinnati": 1520,
    "dolphins": 1500, "miami": 1500,
}

_ALL_ELOS = [_NBA_ELO, _NHL_ELO, _NFL_ELO]

DEFAULT_ELO = 1500


# ---------------------------------------------------------------------------
# Team extraction
# ---------------------------------------------------------------------------

def _extract_teams(question: str) -> tuple[str | None, str | None]:
    """
    Try to extract two team/player names from the question.
    Returns (team1, team2) — either may be None.
    """
    q = question.lower()

    # "Who won the X vs Y [sport] match" — most common in our datasets
    m = re.search(r"who won the (.+?)\s+vs\s+(.+?)\s+(?:tennis|cricket|football|basketball|hockey|baseball|match|game|series|playoff|test|wta|atp|itf)", q)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # "X vs Y match" without "Who won the" prefix
    m = re.search(r"^(?:the\s+)?(.+?)\s+vs\s+(.+?)\s+(?:match|game|series|test|playoff)", q)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # "Will X beat/defeat/win against Y"
    patterns = [
        r"will the (.+?)\s+(?:beat|defeat|win against|top)\s+(?:the\s+)?(.+?)[\?\s]",
        r"will (.+?)\s+(?:beat|defeat|win against|top)\s+(.+?)[\?\s]",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1).strip(), m.group(2).strip()

    # Single team pattern: "Will the X win the championship"
    m = re.search(r"will (?:the )?(.+?)\s+win", q)
    if m:
        return m.group(1).strip(), None

    return None, None


def _lookup_elo(team_name: str | None) -> float:
    if not team_name:
        return DEFAULT_ELO
    t = team_name.lower().strip()
    for elo_dict in _ALL_ELOS:
        # Exact match
        if t in elo_dict:
            return elo_dict[t]
        # Partial match
        for key, val in elo_dict.items():
            if key in t or t in key:
                return val
    return DEFAULT_ELO


# ---------------------------------------------------------------------------
# ESPN public API helpers (no API key required)
# ---------------------------------------------------------------------------

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_TIMEOUT = 8


def _fetch_espn_standings(sport: str, league: str) -> list[dict]:
    url = f"{_ESPN_BASE}/{sport}/{league}/standings"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        entries = []
        for group in data.get("children", []):
            for entry in group.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
                entries.append({
                    "name": team.get("displayName", ""),
                    "abbr": team.get("abbreviation", ""),
                    "wins": stats.get("wins", 0),
                    "losses": stats.get("losses", 0),
                    "win_pct": stats.get("winPercent", 0.5),
                })
        return entries
    except Exception as e:
        log.debug("ESPN standings fetch failed (%s/%s): %s", sport, league, e)
        return []


def _detect_sport(question: str) -> tuple[str, str]:
    q = question.lower()
    # Tennis keywords
    if any(k in q for k in ["tennis", "wta", "atp", "itf", "challenger", "grand slam",
                             "wimbledon", "us open", "australian open", "roland garros", "french open"]):
        return "tennis", "tennis"
    # Cricket keywords
    if any(k in q for k in ["cricket", "test match", "test cricket", "odi", "t20", "ipl",
                             "county championship", "ashes"]):
        return "cricket", "cricket"
    # North American leagues
    if any(k in q for k in ["nba", "basketball"]):  return "basketball", "nba"
    if any(k in q for k in ["nhl", "hockey"]):       return "hockey", "nhl"
    if any(k in q for k in ["nfl", "american football"]):  return "football", "nfl"
    if any(k in q for k in ["mlb", "baseball"]):     return "baseball", "mlb"
    # European football keywords
    if any(k in q for k in ["ligue 1", "la liga", "serie a", "bundesliga", "premier league",
                             "eredivisie", "liga portugal", "football", "soccer", "fc ", " fc",
                             "united", "city", "atletico", "juventus", "napoli", "arsenal",
                             "chelsea", "liverpool", "real madrid", "barcelona"]):
        return "soccer", "uefa"
    # Guess from team names
    if any(t in q for t in _NBA_ELO):  return "basketball", "nba"
    if any(t in q for t in _NHL_ELO):  return "hockey", "nhl"
    if any(t in q for t in _NFL_ELO):  return "football", "nfl"
    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Main research function
# ---------------------------------------------------------------------------

def _sports_result_search(
    binary_question: str,
    clean_question: str,
    sport: str,
    team1: str | None,
    team2: str | None,
    snap_year: str,
    snapshot_ts: str,
    api_key: str,
) -> list[dict]:
    """
    For championship/league/playoff markets: run targeted result searches.
    Finds the actual winner rather than pre-match analysis.
    """
    import re, time
    from ..tools.recursive_search import recursive_search

    # Extract YES/NO team names from binary question
    yes_name = no_name = None
    m = re.search(r"YES\s*=\s*([^,\)\n]+)", binary_question)
    if m:
        raw = m.group(1).strip()
        yes_name = raw.replace("wins", "").replace("win", "").strip()
    m = re.search(r"NO\s*=\s*([^,\)\n]+)", binary_question)
    if m:
        raw = m.group(1).strip()
        no_name = raw.replace("wins", "").replace("win", "").strip()

    result_queries = []
    if sport in ("tennis",):
        if team1 and team2:
            result_queries.append(f"{team1} vs {team2} tennis result winner {snap_year}")
            result_queries.append(f"{team1} {team2} match score {snap_year}")
        result_queries.append(f"{clean_question} result winner {snap_year}")
    elif sport in ("soccer", "uefa"):
        # League champion searches
        if yes_name:
            result_queries.append(f'"{yes_name}" wins {clean_question} champion {snap_year}')
        result_queries.append(f"{clean_question} final standings champion winner {snap_year}")
        result_queries.append(f"{clean_question} title winner {snap_year}")
    elif sport in ("basketball", "nba") and any(k in clean_question.lower() for k in ["playoff", "series", "round", "finals", "championship"]):
        if yes_name:
            result_queries.append(f'"{yes_name}" wins series {snap_year} result')
        result_queries.append(f"{clean_question} result winner series {snap_year}")
    elif sport in ("hockey", "nhl"):
        result_queries.append(f"{clean_question} winner {snap_year}")
        if yes_name:
            result_queries.append(f'"{yes_name}" wins {snap_year}')
    elif sport in ("cricket",):
        result_queries.append(f"{clean_question} result winner match {snap_year}")
        if team1:
            result_queries.append(f"{team1} wins cricket {snap_year} result")

    if not result_queries:
        return []

    all_evidence = []
    seen_urls: set[str] = set()
    for query in result_queries[:3]:
        results = recursive_search(
            question=query, category="sports",
            snapshot_ts=snapshot_ts, api_key=api_key,
            max_iterations=1, results_per_query=3,
        )
        for item in results:
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            body = item.get("text") or item.get("snippet") or ""
            if body:
                all_evidence.append({
                    "source": url, "title": item.get("title", ""),
                    "content": f"[RESULT SEARCH] {body[:2000]}",
                    "iteration": 0, "relevance": 0.95,
                })
                if item.get("text"):
                    break
        time.sleep(0.2)

    return all_evidence


def research_sports(state: ForecastState) -> dict:
    """
    Gather sports evidence for the given market snapshot.
    Returns evidence list and features dict for the ML model.
    """
    from ..tools.recursive_search import _clean_for_search
    question    = state["question"]
    clean_q     = _clean_for_search(question)
    snapshot_ts = state["snapshot_ts"]
    snap_year   = snapshot_ts[:4]
    api_key     = os.environ.get("BRAVE_API_KEY", "")

    team1_name, team2_name = _extract_teams(clean_q)
    sport, league = _detect_sport(clean_q)

    log.info("Sports research: teams=('%s', '%s')  sport=%s", team1_name, team2_name, league)

    evidence = []

    if sport in ("tennis", "cricket", "soccer", "unknown"):
        # No Elo dictionary for these sports — tell synthesis to rely on web search
        evidence.append({
            "source": "sport_context",
            "content": (
                f"Sport detected: {sport}. "
                f"Player/Team 1: {team1_name or 'see question'}. "
                f"Player/Team 2: {team2_name or 'see question'}. "
                f"No pre-seeded Elo ratings available for this sport — "
                f"please rely entirely on web search evidence for rankings, form, and head-to-head record."
            ),
            "relevance": 0.8,
        })
        ml_features = {
            "elo_home": 1500.0, "elo_away": 1500.0,
            "home_advantage": 0.0, "rest_diff_days": 0.0, "form_diff": 0.0,
        }
    else:
        # Elo lookup for NBA / NHL / NFL / MLB
        elo_home = _lookup_elo(team1_name)
        elo_away = _lookup_elo(team2_name)
        evidence.append({
            "source": "elo_ratings",
            "content": (
                f"Team 1 ({team1_name or 'unknown'}): Elo={elo_home:.0f}  "
                f"Team 2 ({team2_name or 'unknown'}): Elo={elo_away:.0f}  "
                f"Theoretical win prob for team 1: "
                f"{1/(1+10**((elo_away-elo_home)/400)):.3f}"
            ),
            "relevance": 0.9,
        })

        standings = _fetch_espn_standings(sport, league)
        if standings:
            top5 = sorted(standings, key=lambda x: -x.get("win_pct", 0))[:5]
            standings_text = " | ".join(
                f"{e['name']} {e['wins']}-{e['losses']}" for e in top5
            )
            evidence.append({
                "source": "espn_standings",
                "content": f"Current {league.upper()} top-5 standings: {standings_text}",
                "relevance": 0.7,
            })
            for team_name in [team1_name, team2_name]:
                if not team_name:
                    continue
                tn = team_name.lower()
                match_entry = next(
                    (e for e in standings
                     if tn in e["name"].lower() or tn in e["abbr"].lower()),
                    None,
                )
                if match_entry:
                    evidence.append({
                        "source": "espn_standings",
                        "content": (
                            f"{match_entry['name']} current record: "
                            f"{match_entry['wins']}-{match_entry['losses']} "
                            f"(win%={match_entry['win_pct']:.3f})"
                        ),
                        "relevance": 0.85,
                    })

        ml_features = {
            "elo_home":       elo_home,
            "elo_away":       elo_away,
            "home_advantage": 1.0,
            "rest_diff_days": 0.0,
            "form_diff":      0.0,
        }

    evidence.append({
        "source":      "ml_features",
        "content":     str(ml_features),
        "ml_features": ml_features,
        "relevance":   1.0,
    })

    # For knowable sports results (leagues, playoffs, championships) run a
    # targeted result search — same approach that fixed politics accuracy.
    # Skip for obscure individual matches (tennis/ITF) where no data exists.
    is_result_searchable = sport in ("tennis", "soccer", "basketball", "hockey", "cricket")
    if is_result_searchable:
        result_items = _sports_result_search(
            binary_question=question,
            clean_question=clean_q,
            sport=sport,
            team1=team1_name,
            team2=team2_name,
            snap_year=snap_year,
            snapshot_ts=snapshot_ts,
            api_key=api_key,
        )
        # Prepend so synthesis sees result evidence first
        evidence = result_items + evidence

    log.info("Sports research complete: %d evidence items  sport=%s", len(evidence), sport)
    return {
        "evidence":            evidence,
        "search_queries_used": [f"{team1_name} vs {team2_name} {sport}"],
    }
