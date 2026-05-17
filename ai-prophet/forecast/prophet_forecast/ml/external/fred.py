"""FRED and CoinGecko data fetcher for finance/economics domain.

Returns (current_value, threshold, volatility) tuples for Black-Scholes.
Returns None on any failure → caller uses market calibrator fallback.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

# FRED series mappings
FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "inflation": "CPIAUCSL",
    "fed": "FEDFUNDS",
    "federal funds": "FEDFUNDS",
    "interest rate": "FEDFUNDS",
    "gdp": "A191RL1Q225SBEA",
    "unemployment": "UNRATE",
    "treasury": "DGS10",
    "yield": "DGS10",
}

# Rough annualized volatilities for common indicators (used when live vol unavailable)
DEFAULT_VOL = {
    "CPIAUCSL": 0.03,    # CPI ~3% annual vol
    "FEDFUNDS": 0.50,    # Fed rate: can move 0-5% range
    "A191RL1Q225SBEA": 0.02,
    "UNRATE": 0.01,
    "DGS10": 0.20,
    "BTC": 0.80,         # crypto ~80% annual vol
    "ETH": 1.00,
}


def fetch_threshold_data(question: str) -> tuple[float, float, float] | None:
    """Parse question, fetch current value, return (current, threshold, volatility).

    Returns None if question doesn't match a threshold pattern or data fetch fails.
    """
    threshold = _parse_threshold(question)
    if threshold is None:
        return None

    # Try crypto first
    if re.search(r"\b(bitcoin|btc)\b", question, re.I):
        return _fetch_crypto("bitcoin", threshold, "BTC")
    if re.search(r"\b(ethereum|eth)\b", question, re.I):
        return _fetch_crypto("ethereum", threshold, "ETH")

    # Try FRED macro
    series = _match_fred_series(question)
    if series:
        return _fetch_fred(series, threshold)

    return None


def _parse_threshold(question: str) -> float | None:
    m = re.search(
        r"(?:above|exceed|over|greater than|more than|higher than|surpass|hit|reach)"
        r"\s*\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB])?\b",
        question, re.I,
    )
    if m:
        return _parse_number_with_suffix(m.group(1), m.group(2))
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB])?\b", question)
    if m:
        return _parse_number_with_suffix(m.group(1), m.group(2))
    # Percentage threshold: "above 3.5%"
    m = re.search(r"(?:above|exceed|over)\s*([\d.]+)\s*%", question, re.I)
    if m:
        return float(m.group(1))
    return None


def _parse_number_with_suffix(value: str, suffix: str | None) -> float:
    number = float(value.replace(",", ""))
    multipliers = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}
    if suffix:
        number *= multipliers[suffix.lower()]
    return number


def _match_fred_series(question: str) -> str | None:
    q = question.lower()
    for keyword, series in FRED_SERIES.items():
        if keyword in q:
            return series
    return None


@lru_cache(maxsize=64)
def _fetch_fred(series: str, threshold: float) -> tuple[float, float, float] | None:
    """Fetch latest FRED series value via public API (no key required for most series)."""
    try:
        import httpx
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        )
        r = httpx.get(url, timeout=2.0)
        if r.status_code != 200:
            return None
        lines = [l for l in r.text.strip().split("\n") if l and not l.startswith("DATE")]
        if not lines:
            return None
        last_line = lines[-1].split(",")
        if len(last_line) < 2:
            return None
        current = float(last_line[1].strip())
        vol = DEFAULT_VOL.get(series, 0.10)
        return current, threshold, vol
    except Exception as exc:
        log.debug("FRED fetch failed for %s: %s", series, exc)
    return None


@lru_cache(maxsize=32)
def _fetch_crypto(coin: str, threshold: float, vol_key: str) -> tuple[float, float, float] | None:
    """Fetch current price from CoinGecko (no auth)."""
    try:
        import httpx
        r = httpx.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd",
            timeout=2.0,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        price = float(data[coin]["usd"])
        vol = DEFAULT_VOL.get(vol_key, 0.80)
        return price, threshold, vol
    except Exception as exc:
        log.debug("CoinGecko fetch failed: %s", exc)
    return None
