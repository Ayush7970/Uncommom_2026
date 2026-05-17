"""
Layer 5: Synthesis — logit-mean ensemble across available LLM providers.

Each available model produces a p_yes. They are combined in logit space
(proper scoring rule for probabilities) rather than averaging raw probabilities,
which avoids squeezing estimates toward 0.5 artificially.

Models used (based on available API keys):
  - Claude Sonnet  (primary,  ANTHROPIC_API_KEY)
  - GPT-4o-mini    (optional, OPENAI_API_KEY)
  - Gemini Flash   (optional, GEMINI_API_KEY)
"""

from __future__ import annotations

import logging
import math
import os

from pydantic import BaseModel, Field

from ..state import ForecastState
from ..tools.recursive_search import summarise_evidence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    p_yes: float = Field(ge=0.01, le=0.99,
        description="Calibrated probability that the event resolves YES")
    rationale: str = Field(
        description="2-3 sentences: key evidence and how it moves the needle vs market price")
    key_factors: list[str] = Field(
        description="2-4 bullet-point factors that most influenced the estimate")
    confidence: float = Field(ge=0.0, le=1.0,
        description="Self-assessed confidence (0=very uncertain, 1=very confident)")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert superforecaster producing calibrated probability estimates for
prediction markets. Your output is scored by Brier score — lower is better.

CALIBRATION PRINCIPLES:
1. The market price is a useful reference — but your job is to find where it is WRONG.
2. Generic statements do NOT justify deviation. You need specific, concrete facts.
3. Extremes (p < 0.08 or p > 0.92) require overwhelming, unambiguous evidence.
4. Consider base rates for similar historical events — use analogues if provided.

NOTE: The time-bucket blend (w(t)) already anchors p_final toward the market structurally.
You do NOT need to do that yourself — focus on producing the most accurate raw estimate.

OUTPUT: Use the provided structured format."""

_USER_TEMPLATE = """\
MARKET QUESTION: {question}

CONTEXT:
- Category: {category}
- Time bucket: {time_bucket} ({hours:.0f} hours to resolution)
- Market price (crowd prior): {p_market:.3f}
- ML model estimate: {p_ml:.3f}

ANALOGOUS PAST MARKETS:
{analogues_text}

RESEARCH EVIDENCE ({n_evidence} sources):
{evidence_summary}

TASK:
Ask yourself: "What do I know that the market doesn't?"
If your evidence is strong and specific, deviate confidently — the w(t) blend will
anchor the final output to the market structurally, so do not self-censor here.
Produce your most accurate calibrated p_yes."""


# ---------------------------------------------------------------------------
# Logit-mean ensemble helper
# ---------------------------------------------------------------------------

def logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit_mean(probs: list[float]) -> float:
    """Combine probabilities in logit space — proper ensemble for Bernoulli outcomes."""
    if not probs:
        return 0.5
    mean_logit = sum(logit(p) for p in probs) / len(probs)
    return sigmoid(mean_logit)


# ---------------------------------------------------------------------------
# Per-provider LLM clients
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _build_anthropic_client() -> object | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.environ.get("SYNTHESIS_MODEL", "claude-sonnet-4-6"),
            api_key=key, max_tokens=1024, temperature=0.1,
        ).with_structured_output(SynthesisOutput)
    except Exception as e:
        log.warning("Anthropic client init failed: %s", e)
        return None


def _build_openai_client() -> object | None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini", api_key=key, max_tokens=1024, temperature=0.1,
        ).with_structured_output(SynthesisOutput)
    except Exception as e:
        log.warning("OpenAI client init failed: %s", e)
        return None


def _build_gemini_client() -> object | None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=key, temperature=0.1,
        ).with_structured_output(SynthesisOutput)
    except Exception as e:
        log.warning("Gemini client init failed: %s", e)
        return None


def _get_clients() -> dict[str, object]:
    global _clients
    if not _clients:
        for name, builder in [
            ("anthropic", _build_anthropic_client),
            ("openai",    _build_openai_client),
            ("gemini",    _build_gemini_client),
        ]:
            client = builder()
            if client is not None:
                _clients[name] = client
        log.info("Synthesis ensemble: %d provider(s) active: %s",
                 len(_clients), list(_clients.keys()))
    return _clients


# ---------------------------------------------------------------------------
# Single provider call
# ---------------------------------------------------------------------------

def _call_one(client, messages: list) -> SynthesisOutput | None:
    try:
        return client.invoke(messages)
    except Exception as e:
        log.warning("Provider call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def synthesis_node(state: ForecastState) -> dict:
    """Logit-mean ensemble synthesis across all available LLM providers."""
    p_market  = state.get("p_market", 0.5)
    p_ml      = state.get("p_ml") or p_market
    category  = state.get("category", "culture")
    bucket    = state.get("time_bucket", "long")
    hours     = state.get("hours_to_resolution", 96.0)
    evidence  = state.get("evidence", [])

    # Build evidence summary
    search_ev = [e for e in evidence if e.get("source") != "ml_features"]
    evidence_summary = summarise_evidence([
        {"title": e.get("title", e.get("source", "")),
         "url":   e.get("source", ""),
         "text":  e.get("content", e.get("text", "")),
         "snippet": ""}
        for e in search_ev
    ]) if search_ev else "No web search results available."

    # Inject analogue context
    analogues_text = _build_analogues_text(state["question"], category)

    user_msg_str = _USER_TEMPLATE.format(
        question         = state["question"],
        category         = category,
        time_bucket      = bucket,
        hours            = hours,
        p_market         = p_market,
        p_ml             = p_ml,
        n_evidence       = len(search_ev),
        evidence_summary = evidence_summary[:4000],
        analogues_text   = analogues_text,
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_msg_str),
    ]

    clients = _get_clients()
    if not clients:
        return _fallback(p_market, p_ml, state)

    # Call all available providers
    results: list[SynthesisOutput] = []
    for name, client in clients.items():
        result = _call_one(client, messages)
        if result is not None:
            p = max(0.01, min(0.99, result.p_yes))
            log.info("  [%s] p_yes=%.3f  conf=%.2f", name, p, result.confidence)
            results.append(result)

    if not results:
        return _fallback(p_market, p_ml, state)

    # Logit-mean ensemble
    p_ensemble = logit_mean([max(0.01, min(0.99, r.p_yes)) for r in results])
    p_ensemble = max(0.01, min(0.99, p_ensemble))

    # Use primary model's rationale; average confidence
    primary    = results[0]
    avg_conf   = sum(r.confidence for r in results) / len(results)

    log.info(
        "Synthesis [ensemble %d models]: market=%s  p_market=%.3f  p_ml=%.3f  "
        "p_llm_raw=%.3f  conf=%.2f",
        len(results), state["market_id"], p_market, p_ml, p_ensemble, avg_conf,
    )

    iterations = list(state.get("p_iterations", []))
    iterations.append(p_ensemble)

    return {
        "p_llm_raw":    p_ensemble,
        "rationale":    primary.rationale,
        "confidence":   avg_conf,
        "p_iterations": iterations,
        "iteration":    state.get("iteration", 0),
    }


def _build_analogues_text(question: str, category: str) -> str:
    try:
        from ..memory.analogue_retrieval import find_analogues
        analogues = find_analogues(question, n=3)
        if not analogues:
            return "No analogous resolved markets found."
        lines = []
        for a in analogues:
            outcome_str = f"→ resolved {'YES' if a['outcome'] == 1 else 'NO'}" if a.get("outcome") is not None else "→ unresolved"
            lines.append(f"• {a['question']} (p_final={a.get('p_final', '?'):.2f} {outcome_str})")
        return "\n".join(lines)
    except Exception:
        return "Analogue retrieval unavailable."


def _fallback(p_market: float, p_ml: float, state: ForecastState) -> dict:
    p_raw = logit_mean([p_market, p_ml])
    p_raw = max(0.01, min(0.99, p_raw))
    log.warning("Synthesis [fallback logit-mean]: market=%s  p_raw=%.3f", state["market_id"], p_raw)
    iterations = list(state.get("p_iterations", []))
    iterations.append(p_raw)
    return {
        "p_llm_raw":    p_raw,
        "rationale":    f"Fallback logit-mean(p_market={p_market:.3f}, p_ml={p_ml:.3f})",
        "confidence":   0.3,
        "p_iterations": iterations,
        "iteration":    state.get("iteration", 0),
    }
