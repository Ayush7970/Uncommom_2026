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
You are a critical reviewer of AI probability forecasts.
Your role: decide if the current estimate needs more web research to be reliable.

ACCEPT the estimate when:
- The evidence directly addresses the question with specific facts
- The deviation from market price is justified by concrete evidence
- The estimate is well-calibrated (not overconfident)

REQUEST REFINEMENT when:
- The estimate deviates >15% from market AND the market price is NOT near 0.50, and the evidence is thin or generic
- There are specific unanswered questions that a targeted web search could resolve
- The rationale is vague ("X is likely" without citing specific facts)

IMPORTANT: When the market price is near 0.50 (within ±0.05), the crowd has NO real price signal —
this means 0.50 is just a neutral placeholder, NOT evidence that the answer is 50/50.
In that case, DO NOT penalize the estimate for deviating from 0.50. The model's estimate
can and should deviate if it has any specific evidence (e.g., player rankings, team form).

Only request refinement if you can write specific, targeted search queries.
Generic queries like "more information about X" are useless. Be precise."""

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

    # Rule 3: hard cap on iterations
    if iteration >= MAX_ITERATIONS:
        log.info("Critic: max iterations (%d) reached — accepting", MAX_ITERATIONS)
        return {
            "critique":   {"accepted": True, "reason": "max iterations reached"},
            "new_queries": [],
        }

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
