"""Elo rating fetcher for sports domain.

Sources (in order of priority):
1. artifacts/elo_cache.json — pre-fetched file cache (survives restarts)
2. ClubElo.com REST API (soccer, no auth)
3. ESPN unofficial API (NBA/NFL/MLB)

All HTTP calls have 2s timeout. Returns None on any failure → caller falls back.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
ELO_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "elo_cache.json")

_file_cache: dict[str, float] | None = None

_DATE_PREFIX_RE = re.compile(
    r"\s+(?:on|for|during)\s+"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?|\d{1,2}/\d{1,2})\b.*$",
    re.I,
)

# Deterministic seed cache for common hackathon sports markets. Live ClubElo
# values overwrite soccer rows when available; NBA/NFL entries keep the sports
# path useful when unauthenticated live sources are unavailable or rate-limited.
STATIC_ELO_RATINGS: dict[str, float] = {
    # NBA
    "celtics": 1665.0,
    "nuggets": 1635.0,
    "thunder": 1625.0,
    "timberwolves": 1605.0,
    "mavericks": 1595.0,
    "bucks": 1585.0,
    "knicks": 1575.0,
    "76ers": 1565.0,
    "suns": 1560.0,
    "warriors": 1555.0,
    "lakers": 1545.0,
    "clippers": 1540.0,
    "heat": 1535.0,
    "cavaliers": 1530.0,
    "pelicans": 1525.0,
    "pacers": 1520.0,
    "kings": 1515.0,
    "magic": 1510.0,
    "rockets": 1505.0,
    "bulls": 1495.0,
    "hawks": 1490.0,
    "nets": 1485.0,
    "raptors": 1480.0,
    "jazz": 1475.0,
    "grizzlies": 1470.0,
    "spurs": 1465.0,
    "trail blazers": 1460.0,
    # NFL
    "chiefs": 1660.0,
    "49ers": 1640.0,
    "ravens": 1630.0,
    "bills": 1615.0,
    "eagles": 1605.0,
    "lions": 1595.0,
    "cowboys": 1585.0,
    "packers": 1570.0,
    "bengals": 1565.0,
    "rams": 1555.0,
    "dolphins": 1550.0,
    "texans": 1545.0,
    "browns": 1540.0,
    "seahawks": 1530.0,
    "saints": 1515.0,
    "buccaneers": 1510.0,
    "broncos": 1500.0,
    "patriots": 1485.0,
    # Soccer seeds; refreshed from ClubElo when network works.
    "mancity": 2010.0,
    "arsenal": 1950.0,
    "liverpool": 1945.0,
    "chelsea": 1835.0,
    "tottenham": 1825.0,
    "manunited": 1815.0,
    "realmadrid": 1990.0,
    "barcelona": 1930.0,
    "bayern": 1960.0,
    "dortmund": 1855.0,
    "psg": 1905.0,
    "juventus": 1810.0,
}


def _load_file_cache() -> dict[str, float]:
    global _file_cache
    if _file_cache is not None:
        return _file_cache
    if os.path.exists(ELO_CACHE_PATH):
        try:
            with open(ELO_CACHE_PATH) as f:
                _file_cache = json.load(f)
            log.debug("Loaded %d Elo cache entries", len(_file_cache))
            return _file_cache
        except Exception as exc:
            log.warning("Failed to load elo_cache.json: %s", exc)
    _file_cache = {}
    return _file_cache


@lru_cache(maxsize=512)
def fetch_elo_diff(question: str) -> float | None:
    """Extract Elo difference (team_a - team_b) from the question.

    Returns positive when team_a has higher Elo (favored).
    Returns None if teams cannot be identified or Elo unavailable.
    """
    teams = _extract_teams(question)
    if not teams:
        return None
    team_a, team_b = teams

    elo_a = _get_elo(team_a)
    elo_b = _get_elo(team_b)
    if elo_a is None or elo_b is None:
        return None

    return elo_a - elo_b


def _extract_teams(question: str) -> tuple[str, str] | None:
    """Parse common binary sports matchup question patterns."""
    patterns = [
        r"Will (?:the )?(.+?)\s+(?:beat|defeat|win against)\s+(?:the )?(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        r"(.+?)\s+at\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.I)
        if m:
            a = _clean_team_name(m.group(1))
            b = _clean_team_name(m.group(2))
            if len(a) > 2 and len(b) > 2:
                return a, b
    return None


def _clean_team_name(raw: str) -> str:
    team = raw.strip().rstrip("?.! ")
    team = re.sub(r"^(?:the|a|an)\s+", "", team, flags=re.I)
    team = _DATE_PREFIX_RE.sub("", team)
    team = re.sub(r"\s+", " ", team).strip()
    return team


def _get_elo(team: str) -> float | None:
    """Look up Elo for a team name. Tries file cache, then live APIs."""
    cache = _load_file_cache()
    key = team.lower().strip()

    # Try exact match in file cache
    if key in cache:
        return cache[key]

    # Try partial match in file cache
    for cached_key, elo in cache.items():
        if key in cached_key or cached_key in key:
            return elo

    # Try live ClubElo (soccer)
    elo = _fetch_clubelo(team)
    if elo is not None:
        return elo

    # Try ESPN (NBA/NFL)
    elo = _fetch_espn_elo(team)
    return elo


def _fetch_clubelo(team: str) -> float | None:
    """Fetch Elo from ClubElo.com REST API."""
    try:
        import httpx
        team_slug = team.replace(" ", "")
        r = httpx.get(f"http://api.clubelo.com/{team_slug}", timeout=2.0)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None
        last = lines[-1].split(",")
        if len(last) >= 5:
            return float(last[4])
    except Exception:
        pass
    return None


def _fetch_espn_elo(team: str) -> float | None:
    """ESPN unofficial API doesn't expose Elo directly — stub for future extension."""
    return None


def prebuild_cache() -> int:
    """Fetch and save Elo ratings for common teams to elo_cache.json.

    Call this once before the 2-week evaluation window opens.
    Returns number of teams cached.
    """
    # Common NBA, NFL, soccer teams
    teams = [
        # NBA
        "Lakers", "Warriors", "Celtics", "Heat", "Bucks", "Nets", "Bulls",
        "Knicks", "Suns", "Nuggets", "Clippers", "Raptors", "76ers", "Mavericks",
        "Rockets", "Spurs", "Grizzlies", "Jazz", "Timberwolves", "Trail Blazers",
        # NFL
        "Chiefs", "Eagles", "Cowboys", "49ers", "Bills", "Ravens", "Patriots",
        "Packers", "Rams", "Bengals", "Broncos", "Seahawks", "Saints", "Buccaneers",
        # Soccer (use ClubElo names)
        "ManCity", "Arsenal", "Liverpool", "Chelsea", "Tottenham", "ManUnited",
        "RealMadrid", "Barcelona", "Bayern", "Dortmund", "PSG", "Juventus",
    ]

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cache = dict(STATIC_ELO_RATINGS)
    for team in teams:
        elo = _fetch_clubelo(team)
        if elo:
            cache[team.lower()] = elo
            log.info("Cached %s Elo=%.0f", team, elo)

    with open(ELO_CACHE_PATH, "w") as f:
        json.dump(dict(sorted(cache.items())), f, indent=2)

    global _file_cache
    _file_cache = cache
    log.info("Saved %d Elo entries to %s", len(cache), ELO_CACHE_PATH)
    return len(cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = prebuild_cache()
    print(f"Cached {n} teams")
