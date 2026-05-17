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
    Try to extract two team names from the question.
    Returns (team1, team2) — either may be None.
    """
    q = question.lower()

    # Pattern: "Will X beat/defeat/win against Y"
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


_INDIVIDUAL_SPORTS_KEYWORDS = [
    "tennis", "atp", "wta", "itf", "challenger", "wimbledon", "us open",
    "french open", "australian open", "roland garros",
    "cricket", "test match", "odi", "ipl", "t20",
    "golf", "pga", "lpga", "masters",
    "boxing", "mma", "ufc",
    "cycling", "tour de france",
    "swimming", "athletics", "track",
]


def _detect_sport(question: str) -> tuple[str, str]:
    q = question.lower()
    # Individual sports: no team Elo applies
    if any(k in q for k in _INDIVIDUAL_SPORTS_KEYWORDS):
        return "individual", "none"
    if any(k in q for k in ["nba", "basketball", "nfl", "football", "nhl", "hockey", "mlb", "baseball"]):
        if "nba" in q or "basketball" in q:  return "basketball", "nba"
        if "nhl" in q or "hockey" in q:      return "hockey", "nhl"
        if "nfl" in q or "football" in q:    return "football", "nfl"
        if "mlb" in q or "baseball" in q:    return "baseball", "mlb"
    # Guess from team names
    q_lower = question.lower()
    if any(t in q_lower for t in _NBA_ELO):  return "basketball", "nba"
    if any(t in q_lower for t in _NHL_ELO):  return "hockey", "nhl"
    if any(t in q_lower for t in _NFL_ELO):  return "football", "nfl"
    return "basketball", "nba"


# ---------------------------------------------------------------------------
# Main research function
# ---------------------------------------------------------------------------

def research_sports(state: ForecastState) -> dict:
    """
    Gather sports evidence for the given market snapshot.
    Returns evidence list and features dict for the ML model.
    """
    question    = state["question"]
    snapshot_ts = state["snapshot_ts"]

    team1_name, team2_name = _extract_teams(question)
    sport, league = _detect_sport(question)

    log.info("Sports research: teams=('%s', '%s')  sport=%s", team1_name, team2_name, league)

    evidence = []
    individual_sport = (sport == "individual")

    if not individual_sport:
        # Elo lookup — only meaningful for team sports with known rosters
        elo_home = _lookup_elo(team1_name)
        elo_away = _lookup_elo(team2_name)

        if team1_name or team2_name:
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
    else:
        elo_home = elo_away = 1500.0  # unused for individual sports

    # ESPN standings (current-season context — skip for individual sports)
    standings = [] if individual_sport else _fetch_espn_standings(sport, league)
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

        # Look up specific team records
        for team_name in [team1_name, team2_name]:
            if not team_name:
                continue
            tn = team_name.lower()
            match = next(
                (e for e in standings
                 if tn in e["name"].lower() or tn in e["abbr"].lower()),
                None
            )
            if match:
                evidence.append({
                    "source": "espn_standings",
                    "content": (
                        f"{match['name']} current record: "
                        f"{match['wins']}-{match['losses']} "
                        f"(win%={match['win_pct']:.3f})"
                    ),
                    "relevance": 0.85,
                })

    # Build ML features — empty dict for individual sports (ml_predictor falls back to p_market)
    ml_features = {} if individual_sport else {
        "elo_home":        elo_home,
        "elo_away":        elo_away,
        "home_advantage":  1.0 if team1_name else 0.0,
        "rest_diff_days":  0.0,
        "form_diff":       0.0,
    }

    # Attach features to evidence for ml_predictor_node to consume
    evidence.append({
        "source":      "ml_features",
        "content":     str(ml_features),
        "ml_features": ml_features,
        "relevance":   1.0,
    })

    # Brave web search for injury / matchup news (no-op if no API key)
    brave_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_key:
        from ..tools.recursive_search import recursive_search
        web_results = recursive_search(
            question=question,
            category="sports",
            snapshot_ts=snapshot_ts,
            api_key=brave_key,
            max_iterations=1,
            results_per_query=2,
        )
        for item in web_results:
            body = item.get("text") or item.get("snippet") or ""
            if body:
                evidence.append({
                    "source":    item["url"],
                    "title":     item.get("title", ""),
                    "content":   body[:1500],
                    "relevance": 0.75,
                })

    log.info("Sports research complete: %d evidence items", len(evidence))
    return {
        "evidence":           evidence,
        "search_queries_used": [f"{team1_name} vs {team2_name} {league}"],
    }
