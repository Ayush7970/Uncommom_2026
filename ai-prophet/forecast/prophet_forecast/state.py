"""ForecastState — single source of truth flowing through the LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class ForecastState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Inputs (provided by replay harness or live runner)
    # ------------------------------------------------------------------
    market_id: str
    question: str
    p_market: float          # crowd price at snapshot_ts — given as input signal
    snapshot_ts: str         # ISO 8601 — treat as "now"; no data newer than this
    resolves_at: str         # ISO 8601 resolution time
    category_hint: str | None  # dataset hint, may be None

    # ------------------------------------------------------------------
    # Layer 2: Router outputs
    # ------------------------------------------------------------------
    category: str            # sports | finance | politics | science_tech | culture
    time_bucket: str         # long | medium | short | urgent
    hours_to_resolution: float
    w_t: float               # blend weight looked up from time_bucket

    # ------------------------------------------------------------------
    # Layer 3: Research evidence
    # ------------------------------------------------------------------
    evidence: list[dict[str, Any]]   # list of {source, content, relevance}
    search_queries_used: list[str]

    # ------------------------------------------------------------------
    # Layer 4: ML predictor
    # ------------------------------------------------------------------
    p_ml: float | None       # ML model probability estimate

    # ------------------------------------------------------------------
    # Layer 5: Synthesis
    # ------------------------------------------------------------------
    p_llm_raw: float | None  # raw LLM probability before calibration
    rationale: str | None

    # ------------------------------------------------------------------
    # Layer 6: Critic + refinement
    # ------------------------------------------------------------------
    critique: dict[str, Any] | None
    new_queries: list[str]   # queries the critic wants to run next
    p_iterations: list[float]  # track oscillation
    iteration: int           # 0, 1, or 2 max

    # ------------------------------------------------------------------
    # Layer 7: Calibration + blend
    # ------------------------------------------------------------------
    p_calibrated: float | None
    p_final: float           # final output — the number we submit

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    confidence: float | None  # 0–1 self-reported model confidence
    trace_id: str | None
    error: str | None         # set if any layer fails gracefully
