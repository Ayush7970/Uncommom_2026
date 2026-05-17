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
You are an expert superforecaster. Estimates are scored by Brier score (lower = better; random = 0.25, perfect = 0.00). Every 0.01 improvement in Brier matters.

══════════════════════════════════════════════════════
CORE TASK
══════════════════════════════════════════════════════
Output p_yes ∈ [0.01, 0.99] = probability the YES outcome occurs.

YES is ALWAYS the first option named in the question. Examples:
  "Did Watson win? (YES=Watson, NO=Okamura)" → p_yes = P(Watson wins)
  "Did PSG win Ligue 1? (YES=PSG, NO=anyone else)" → p_yes = P(PSG won)

══════════════════════════════════════════════════════
STRICT GUARDRAILS
══════════════════════════════════════════════════════
G1. Market price = 0.50 is a NEUTRAL PLACEHOLDER with zero information content.
    It does NOT mean the answer is 50/50. Treat it as if no market price exists.

G2. NEVER output exactly 0.50 if you have ANY information about relative strength.
    If one side is measurably stronger, reflect that (even small advantages → 0.55–0.65).

G3. Extremes (p < 0.10 or p > 0.90) require definitive, specific evidence.
    A ranking gap alone justifies 0.65–0.80, NOT 0.90+.

G4. Confidence must match evidence quality:
    0.10–0.30 = unknown/no data (genuinely 50/50 territory)
    0.30–0.55 = some indirect evidence (base rates, rankings)
    0.55–0.75 = good specific evidence (standings, recent form, head-to-head)
    0.75–0.90 = definitive evidence (confirmed result, overwhelming data)

══════════════════════════════════════════════════════
DOMAIN BASE RATES (use when web evidence is absent)
══════════════════════════════════════════════════════
TENNIS:      Top-100 WTA/ATP vs Top-200: ~68%  |  Top-50 vs Top-300: ~78%
CRICKET:     Pakistan vs Bangladesh Test: ~65%  |  England vs Zimbabwe: ~80%
NBA:         #1 seed vs #4-5 seed Round 2: ~65–70%
SOCCER:      Dominant league champion (5+ consecutive titles): ~75–85%
POLITICS:    Incumbent vs challenger (no polling): ~58%  |  Two unknowns: ~50%
ENTERTAINMENT: Returning favourite / defending champion: ~60–70%
POLITICS-NUMERIC: Use expected value reasoning for count questions

══════════════════════════════════════════════════════
FEW-SHOT CALIBRATED EXAMPLES
══════════════════════════════════════════════════════

[SPORTS - Tennis ranking gap]
Q: "Did Heather Watson win? (YES=Watson WTA ~70, NO=Okamura WTA ~300)"
Research: No specific match result. Watson career titles: 8. Okamura: challenger level.
Reasoning: 230-place ranking gap → top-100 beats top-300 ~72% in practice.
→ p_yes=0.72, confidence=0.62, rationale="Watson ~230 ranks above Okamura; base rate ~72% for this gap"

[SPORTS - Dominant league winner]
Q: "Did PSG win/prevail? (YES=PSG wins Ligue 1, NO=Lille, Monaco, etc.)"
Research: PSG won 8 of last 10 Ligue 1 titles; led 2025-26 table most of season.
Reasoning: Strong favourite historically and in-season. Small risk from Lille/Monaco.
→ p_yes=0.82, confidence=0.78, rationale="PSG's historical dominance + in-season standing"

[CRICKET - Established ranking gap]
Q: "Did Pakistan win? (YES=Pakistan wins Test, NO=Bangladesh)"
Research: Pakistan ranked 3rd Test; Bangladesh ranked 8th. H2H: Pakistan wins ~65%.
Reasoning: Consistent historical gap between these nations.
→ p_yes=0.65, confidence=0.63, rationale="Pakistan ranked 3rd vs Bangladesh 8th; historical H2H ~65%"

[POLITICS - Unknown candidates, no polling]
Q: "Did Brit Aguirre win? (YES=Aguirre wins WV Dem primary)"
Research: No polling, no fundraising data, two low-profile candidates.
Reasoning: Truly no differentiating information available.
→ p_yes=0.50, confidence=0.18, rationale="No polling or public data for either candidate"

[NBA - Seeding advantage]
Q: "Did Oklahoma City win? (YES=OKC wins series vs Lakers)"
Research: OKC #1 seed Western Conference 2025-26; Lakers #5; OKC home court advantage.
Reasoning: Higher seed with home court in a 7-game series = significant advantage.
→ p_yes=0.70, confidence=0.68, rationale="OKC #1 seed + home court; #1 beats #5 ~68% historically"

[ENTERTAINMENT - Defending champion]
Q: "Did [Contestant X] win Tournament of Champions Season 7?"
Research: Contestant X won Season 6; considered fan favourite entering Season 7.
Reasoning: Defending champions have moderate advantage but competition is reset.
→ p_yes=0.35, confidence=0.45, rationale="Large field, defending champ wins repeat ~35% of time"

══════════════════════════════════════════════════════
THINKING STEPS (execute mentally before outputting)
══════════════════════════════════════════════════════
1. WHO is YES? Identify from question framing.
2. WEB EVIDENCE: Does the research contain specific, useful facts? If yes, use them.
3. BASE RATE: If no web evidence, apply the domain base rate above.
4. RELATIVE STRENGTH: Is there a measurable gap (ranking, seeding, form)? Quantify it.
5. p_yes: Combine evidence + base rate. Do NOT output 0.50 unless genuinely 50/50.
6. confidence: Match to evidence quality per G4 above."""

_USER_TEMPLATE = """\
MARKET QUESTION: {question}

CONTEXT:
- Category: {category}
- Time bucket: {time_bucket} ({hours:.0f} hours to resolution)
- Market price (crowd prior): {p_market:.3f}{market_note}
- ML model estimate (domain model): {p_ml:.3f}

ANALOGOUS PAST MARKETS:
{analogues_text}

RESEARCH EVIDENCE ({n_evidence} sources):
{evidence_summary}

YOUR TASK:
Execute the 6 thinking steps. Output your most accurate p_yes.
Remember: a calibrated 0.65 beats a lazy 0.50 when the answer is known to be ~0.70.
Do not self-censor — the w(t) blend structurally anchors p_final toward market price."""


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


def _build_openrouter_client(model: str, label: str) -> object | None:
    """Build a ChatOpenAI-compatible client pointed at OpenRouter."""
    key  = os.environ.get("OPENROUTER_API_KEY", "")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base,
            max_tokens=1024,
            temperature=0.1,
            default_headers={
                "HTTP-Referer": "https://uprising.ai",
                "X-Title": "PRIMA Forecast",
            },
        ).with_structured_output(SynthesisOutput, method="function_calling")
    except Exception as e:
        log.warning("OpenRouter %s client init failed: %s", label, e)
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
        log.warning("OpenAI direct client init failed: %s", e)
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
        log.warning("Gemini direct client init failed: %s", e)
        return None


def _get_clients() -> dict[str, object]:
    global _clients
    if not _clients:
        builders = [
            ("anthropic",      _build_anthropic_client),
            # OpenRouter routes to GPT-4o and Gemini 2.5 Flash
            ("gpt4o",          lambda: _build_openrouter_client("openai/gpt-4o-2024-11-20", "GPT-4o")),
            ("gemini_flash",   lambda: _build_openrouter_client("google/gemini-2.5-flash", "Gemini-2.5-Flash")),
        ]
        # Fallback: direct keys if OpenRouter isn't available
        if not os.environ.get("OPENROUTER_API_KEY"):
            builders += [
                ("openai_direct",  _build_openai_client),
                ("gemini_direct",  _build_gemini_client),
            ]
        for name, builder in builders:
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
        result = client.invoke(messages)
        # Fix: Claude sometimes returns key_factors as a newline-delimited string
        if isinstance(result.key_factors, str):
            result.key_factors = [f.strip() for f in result.key_factors.split("\n") if f.strip()]
        return result
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

    market_note = "  ← no real crowd price (neutral placeholder)" if abs(p_market - 0.5) < 0.05 else ""

    user_msg_str = _USER_TEMPLATE.format(
        question         = state["question"],
        category         = category,
        time_bucket      = bucket,
        hours            = hours,
        p_market         = p_market,
        market_note      = market_note,
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

    # Use primary model's rationale; use MAX confidence so a confident model
    # isn't buried by uncertain ones (e.g. Claude 0.55 + GPT 0.20 → use 0.55 not 0.28)
    primary  = results[0]
    avg_conf = max(r.confidence for r in results)

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
