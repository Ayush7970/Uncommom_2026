"""
Layer 6: Critic + Recursive Refinement.

Implements all 5 rules:
  1. Only loop if critic returns non-empty new_queries
  2. Monotonicity gate (KL divergence threshold)
  3. Hard cap: max 2 iterations
  4. Variance check + fallback to p_market on oscillation
  5. Calibration applied after the loop (not here)
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from ..state import ForecastState
from ..tools.refine_controller import apply_monotonicity_or_fallback

log = logging.getLogger(__name__)

MAX_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class CritiqueOutput(BaseModel):
    assessment: str = Field(
        description="'accepted' if estimate is well-supported, 'needs_refinement' if more research needed"
    )
    issues: list[str] = Field(
        description="Specific gaps or weaknesses in the current estimate (empty if accepted)"
    )
    new_queries: list[str] = Field(
        description="1-2 concrete search queries to resolve the issues (empty if accepted)"
    )
    reasoning: str = Field(
        description="One sentence explaining the decision"
    )


# ---------------------------------------------------------------------------
# LLM client (Haiku — cheap)
# ---------------------------------------------------------------------------

_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from langchain_anthropic import ChatAnthropic
        model = os.environ.get("CRITIC_MODEL", "claude-haiku-4-5-20251001")
        _llm_client = ChatAnthropic(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            max_tokens=512,
            temperature=0,
        ).with_structured_output(CritiqueOutput)
        return _llm_client
    except Exception as e:
        log.warning("Could not init critic LLM: %s — skipping refinement", e)
        return None


_SYSTEM = """\
You are a calibration auditor for AI probability forecasts.
Decide: accept the current estimate, or request targeted web refinement?

══════════════════════════════════════════════════════
AUTO-ACCEPT — answer "accepted" immediately if ANY condition is true:
══════════════════════════════════════════════════════
• The estimate is NOT 0.50 AND the rationale mentions specific facts (rankings, scores, standings)
• The estimate is 0.45–0.55 AND confidence < 0.35 — model is genuinely uncertain; more search won't help
• The category is sports AND the question involves obscure low-tier athletes (ITF, local leagues, challengers with no web presence)
• The evidence already contains specific relevant data (player rankings, team standings, recent match scores)

══════════════════════════════════════════════════════
REQUEST REFINEMENT — only when ALL conditions are met:
══════════════════════════════════════════════════════
• Confidence < 0.50 (model is uncertain)
• The estimate is 0.45–0.55 AND you suspect useful data EXISTS online
• You can write 1-2 HIGHLY SPECIFIC queries (player name + date + tournament, not "more info about X")
• The rationale is purely generic ("player X is a good player") with no concrete facts

NEVER request refinement for:
• Obscure ITF/Challenger matches where rankings aren't publicly available
• Questions where the estimate already uses base rates or rankings
• Confident estimates (use the auto-accept rules above)

IMPORTANT: market price = 0.50 is a NEUTRAL PLACEHOLDER (no crowd data).
Never penalize deviation from 0.50. The model should deviate based on relative strength.

══════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
══════════════════════════════════════════════════════

[ACCEPT — confident estimate with reasoning]
Q: "Did OKC win vs Lakers?"  estimate=0.70  conf=0.68
Rationale: "OKC #1 seed, home court, 65+ wins this season"
→ ACCEPTED. Confident, concrete reasoning. No refinement needed.

[ACCEPT — genuinely uncertain, don't waste search]
Q: "Did Najzer win vs Ebster ITF W15 Klagenfurt R32?"  estimate=0.50  conf=0.15
Rationale: "Cannot find information about these players"
→ ACCEPTED. Obscure ITF match. No useful data exists online. Accept 0.50.

[ACCEPT — market placeholder, model used base rate correctly]
Q: "Did Watson win vs Okamura WTA Challenger?"  estimate=0.72  conf=0.62  market=0.50
→ ACCEPTED. Deviation from 0.50 is appropriate. Market is uninformative. Good base-rate estimate.

[REFINE — specific gap, searchable fact exists]
Q: "Did Bangladesh win vs Pakistan Test cricket?"  estimate=0.50  conf=0.25
Rationale: "Not sure about relative strength, no evidence found"
→ REFINE with: ["Pakistan Bangladesh Test cricket May 2026 result", "Pakistan Bangladesh ICC Test ranking 2026"]

[REFINE — league winner, specific standings available]
Q: "Did PSG win Ligue 1?"  estimate=0.50  conf=0.20
Rationale: "No web results found"
→ REFINE with: ["Ligue 1 2025-26 final standings winner champion", "PSG Ligue 1 2026 season result"]"""

_USER_TEMPLATE = """\
MARKET QUESTION: {question}
CATEGORY: {category}
MARKET PRICE (crowd prior): {p_market:.3f}{market_note}
CURRENT ESTIMATE: {p_llm_raw:.3f}
DEVIATION FROM MARKET: {deviation:+.3f} ({direction})

CURRENT RATIONALE:
{rationale}

EVIDENCE GATHERED ({n_evidence} sources, {n_queries} queries):
{evidence_preview}

ITERATION: {iteration} of {max_iter} allowed

Should this estimate be accepted, or does it need specific additional research?
If refinement is needed, provide 1-2 very specific search queries."""


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def critic_node(state: ForecastState) -> dict:
    """
    Critique the synthesis output and decide whether to refine or accept.
    Applies monotonicity gate + variance check before triggering refinement.
    """
    iteration    = state.get("iteration", 0)
    p_llm_raw    = state.get("p_llm_raw") or state.get("p_market", 0.5)
    p_market     = state.get("p_market", 0.5)
    p_iterations = list(state.get("p_iterations", [p_llm_raw]))

    confidence = state.get("confidence") or 0.0

    # Rule 3: hard cap on iterations
    if iteration >= MAX_ITERATIONS:
        log.info("Critic: max iterations (%d) reached — accepting", MAX_ITERATIONS)
        return {
            "critique":   {"accepted": True, "reason": "max iterations reached"},
            "new_queries": [],
        }

    # Fast-path: auto-accept without calling LLM (saves ~$0.002/market and ~8s)
    # Case A: model is already confident enough
    if confidence >= 0.60:
        log.info("Critic [auto-accept, conf=%.2f]: market=%s", confidence, state["market_id"])
        return {"critique": {"accepted": True, "reason": f"confident enough (conf={confidence:.2f})"}, "new_queries": []}

    # Case B: model is genuinely uncertain and estimate is near 0.5 — more search won't help
    if confidence < 0.35 and abs(p_llm_raw - 0.5) < 0.06:
        log.info("Critic [auto-accept, uncertain+neutral]: market=%s  p=%.3f  conf=%.2f",
                 state["market_id"], p_llm_raw, confidence)
        return {"critique": {"accepted": True, "reason": "genuinely uncertain, near 0.5 — refinement won't help"}, "new_queries": []}

    # Rule 4: variance check (before calling LLM — saves cost)
    if len(p_iterations) >= 2:
        _, fell_back = apply_monotonicity_or_fallback(
            p_new=p_llm_raw, p_old=p_iterations[-2],
            p_market=p_market, p_iterations=p_iterations,
        )
        if fell_back:
            log.info("Critic: oscillation detected — falling back to p_market=%.3f", p_market)
            return {
                "p_llm_raw":   p_market,
                "critique":    {"accepted": True, "reason": "oscillation fallback to market"},
                "new_queries": [],
                "confidence":  0.2,
            }

    llm = _get_llm()
    if llm is None:
        return {"critique": {"accepted": True, "reason": "no LLM"}, "new_queries": []}

    # Build evidence preview (first 800 chars of each item)
    evidence   = state.get("evidence", [])
    search_ev  = [e for e in evidence if e.get("source") != "ml_features"]
    n_queries  = len(state.get("search_queries_used", []))
    ev_preview = "\n\n".join(
        f"[{e.get('title', e.get('source', '')[:40])}]: {e.get('content', '')[:400]}"
        for e in search_ev[:3]
    ) or "No search evidence."

    deviation   = p_llm_raw - p_market
    direction   = "above market" if deviation > 0 else "below market"
    market_note = "  ← uninformative placeholder (no real crowd price)" if abs(p_market - 0.5) < 0.05 else ""

    user_msg = _USER_TEMPLATE.format(
        question     = state["question"],
        category     = state.get("category", "unknown"),
        p_market     = p_market,
        market_note  = market_note,
        p_llm_raw    = p_llm_raw,
        deviation    = deviation,
        direction    = direction,
        rationale    = state.get("rationale", "(none)"),
        n_evidence   = len(search_ev),
        n_queries    = n_queries,
        evidence_preview = ev_preview,
        iteration    = iteration + 1,
        max_iter     = MAX_ITERATIONS,
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        result: CritiqueOutput = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])

        accepted    = result.assessment == "accepted"
        new_queries = result.new_queries if not accepted else []

        # Rule 2: monotonicity gate — if no prior iteration, skip check
        if not accepted and len(p_iterations) >= 2:
            _, fell_back = apply_monotonicity_or_fallback(
                p_new=p_llm_raw, p_old=p_iterations[-2],
                p_market=p_market, p_iterations=p_iterations,
            )
            if fell_back:
                accepted    = True
                new_queries = []

        log.info(
            "Critic [llm]: market=%s  assessment=%s  new_queries=%d  reason='%s'",
            state["market_id"], result.assessment, len(new_queries), result.reasoning[:60],
        )

        return {
            "critique":   {
                "accepted":    accepted,
                "issues":      result.issues,
                "reasoning":   result.reasoning,
            },
            "new_queries": new_queries,
            # Increment iteration counter when we decide to refine
            "iteration":   iteration + (0 if accepted else 1),
        }

    except Exception as e:
        log.warning("Critic LLM failed: %s — accepting current estimate", e)
        return {
            "critique":   {"accepted": True, "reason": f"LLM error: {e}"},
            "new_queries": [],
        }


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def should_refine(state: ForecastState) -> str:
    """LangGraph conditional edge: 'refine' or 'calibrate'."""
    # Rule 1: only loop if new_queries is non-empty
    if state.get("new_queries"):
        return "refine"
    return "calibrate"
