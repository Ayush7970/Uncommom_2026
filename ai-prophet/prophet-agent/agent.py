"""
Prophet Arena Trading Agent
----------------------------
A clean, minimal agent that trades prediction markets using the
ai_prophet_core package for tick lifecycle and Anthropic Claude
for market analysis.

Usage:
    python agent.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime

import anthropic
from dotenv import load_dotenv

from ai_prophet_core import (
    DEFAULT_API_URL,
    ServerAPIClient,
    TradeIntentRequest,
)
from ai_prophet_core.arena import BenchmarkSession

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prophet-agent")

SERVER_URL   = DEFAULT_API_URL  # always use the canonical URL from the package
API_KEY      = os.environ["PA_SERVER_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

# Experiment settings
EXPERIMENT_SLUG       = "claude-agent-v1"
MODEL_NAME            = "claude-sonnet-4-6"
STARTING_CASH         = 10_000.0
N_TICKS               = 500       # how many ticks to run (experiment budget)
SHARES_PER_TRADE      = "10"      # fixed size per trade intent
MAX_MARKETS           = 15        # analyse at most N markets per tick
MAX_SHARES_PER_SIDE   = 100       # cap stacking: at most 100 shares on any (market, side)
SLEEP_ON_NO_TICK      = 15        # seconds to wait when no tick is available

# Network resilience
NETWORK_BACKOFF_BASE  = 30        # first retry delay (s); doubles each attempt
NETWORK_BACKOFF_MAX   = 300       # cap on retry delay (s)
NETWORK_BLACKOUT_ALARM = 300      # log ERROR once blackout exceeds this (s)


# ---------------------------------------------------------------------------
# LLM market analysis (structured output via tool_use)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert prediction market analyst making trades in a paper-trading
arena. For each market in the input, choose ONE action from its
`allowed_actions` list. The system will reject anything else.

Pricing rules:
- Markets resolve YES (1.0) or NO (0.0).
- best_bid / best_ask are the current market prices (0-1 scale).
- If your fair value of TRUE > best_ask, BUY_YES is profitable.
- If your fair value of TRUE < best_bid (i.e. NO is underpriced at 1-best_bid),
  BUY_NO is profitable.
- Otherwise HOLD.

Position discipline:
- `current_holdings` shows shares you already own on this market.
- You CANNOT open the opposite side of an existing position. The opposite
  action will not be in `allowed_actions` whenever you already hold the
  other side. Don't try.
- If you already hold a side, default to HOLD. Only add to a position when
  the market has moved materially in your favor since your average entry,
  or your prior thesis has strengthened. Stacking the same trade every
  tick is exactly the failure mode to avoid.

Output is a single tool call: `submit_decisions`. One decision per input
market. `reasoning` must be one concise sentence."""


DECISION_TOOL = {
    "name": "submit_decisions",
    "description": "Submit one trading decision per input market.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "market_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["BUY_YES", "BUY_NO", "HOLD"],
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["market_id", "action", "reasoning"],
                },
            },
        },
        "required": ["decisions"],
    },
}


def analyse_markets(client: anthropic.Anthropic, markets: list[dict]) -> list[dict]:
    """Ask Claude to evaluate a list of markets and return trading decisions.

    Uses Anthropic tool_use with a forced tool_choice, so the model is
    guaranteed to emit a parsed JSON object matching the DECISION_TOOL
    schema. No prose-vs-JSON retry loop.
    """
    user_msg = (
        "Analyse these prediction markets and return one decision per "
        f"market_id via `submit_decisions`:\n\n{json.dumps(markets, indent=2)}"
    )

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[DECISION_TOOL],
        tool_choice={"type": "tool", "name": "submit_decisions"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_decisions":
            decisions = block.input.get("decisions") if isinstance(block.input, dict) else None
            if isinstance(decisions, list):
                return decisions

    log.error("LLM produced no submit_decisions tool_use block; holding all markets")
    return [
        {"market_id": m["market_id"], "action": "HOLD", "reasoning": "LLM tool_use missing"}
        for m in markets
    ]


# ---------------------------------------------------------------------------
# Position-aware market context
# ---------------------------------------------------------------------------
def _shares_by_side(portfolio) -> dict[str, dict[str, float]]:
    """Index portfolio positions as {market_id: {YES: shares, NO: shares}}."""
    out: dict[str, dict[str, float]] = {}
    if portfolio is None:
        return out
    for pos in portfolio.positions:
        shares = float(pos.shares)
        if shares <= 0:
            continue
        out.setdefault(pos.market_id, {"YES": 0.0, "NO": 0.0})[pos.side] = shares
    return out


def _allowed_actions(held: dict[str, float] | None) -> list[str]:
    """Determine which actions are legal given current holdings on a market.

    - No holdings: BUY_YES, BUY_NO, HOLD.
    - Holding only YES under cap: BUY_YES, HOLD.
    - Holding only NO under cap: BUY_NO, HOLD.
    - Holding any side at or above cap: HOLD only.
    The server enforces "no opposite side"; we enforce the cap.
    """
    if not held:
        return ["BUY_YES", "BUY_NO", "HOLD"]
    yes = held.get("YES", 0.0)
    no = held.get("NO", 0.0)
    if yes > 0 and no > 0:
        return ["HOLD"]
    if yes > 0:
        return ["BUY_YES", "HOLD"] if yes < MAX_SHARES_PER_SIDE else ["HOLD"]
    if no > 0:
        return ["BUY_NO", "HOLD"] if no < MAX_SHARES_PER_SIDE else ["HOLD"]
    return ["BUY_YES", "BUY_NO", "HOLD"]


def build_market_list(candidates_markets, positions_by_market: dict) -> list[dict]:
    """Build the per-tick LLM input, skipping markets where only HOLD is legal."""
    out: list[dict] = []
    for m in candidates_markets[:MAX_MARKETS]:
        held = positions_by_market.get(m.market_id)
        actions = _allowed_actions(held)
        if actions == ["HOLD"]:
            continue
        out.append({
            "market_id": m.market_id,
            "question": m.question,
            "topic": m.topic or "unknown",
            "best_bid": m.quote.best_bid,
            "best_ask": m.quote.best_ask,
            "volume_24h": m.quote.volume_24h,
            "resolution_time": m.resolution_time.isoformat(),
            "current_holdings": [
                {"side": s, "shares": held[s]}
                for s in ("YES", "NO") if held and held.get(s, 0) > 0
            ],
            "allowed_actions": actions,
        })
    return out


# ---------------------------------------------------------------------------
# Trade intent builder (with belt-and-braces guard)
# ---------------------------------------------------------------------------
def decisions_to_intents(
    decisions: list[dict],
    positions_by_market: dict[str, dict[str, float]],
) -> list[TradeIntentRequest]:
    """Convert LLM decisions to trade intents, enforcing per-side cap and
    rejecting any opposite-side opens. The LLM should already respect
    `allowed_actions`, but we re-check here so a confused model can't
    bypass the rule."""
    intents: list[TradeIntentRequest] = []
    for d in decisions:
        action_str = d.get("action", "HOLD")
        if action_str == "HOLD":
            continue
        side = "YES" if action_str == "BUY_YES" else "NO"
        opposite = "NO" if side == "YES" else "YES"
        held = positions_by_market.get(d["market_id"], {})
        if held.get(opposite, 0) > 0:
            log.info("  SKIP  %s on %s: already hold %s", action_str, d["market_id"], opposite)
            continue
        if held.get(side, 0) >= MAX_SHARES_PER_SIDE:
            log.info("  SKIP  %s on %s: at cap (%d)", action_str, d["market_id"], MAX_SHARES_PER_SIDE)
            continue
        intents.append(
            TradeIntentRequest(
                market_id=d["market_id"],
                action="BUY",
                side=side,
                shares=SHARES_PER_TRADE,
                idempotency_key="",
            )
        )
    return intents


# ---------------------------------------------------------------------------
# Config hash helper (determines experiment identity)
# ---------------------------------------------------------------------------
def _config_hash(cfg: dict) -> str:
    raw = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------
def run():
    log.info("Starting Prophet Arena agent")
    log.info("Server: %s", SERVER_URL)

    llm = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    api = ServerAPIClient(base_url=SERVER_URL, api_key=API_KEY, timeout=60)

    config = {
        "model": MODEL_NAME,
        "shares_per_trade": SHARES_PER_TRADE,
        "max_markets": MAX_MARKETS,
        "version": "1.0",
    }

    with BenchmarkSession(api) as session:
        # --- Create / get experiment -----------------------------------------
        exp_resp = session.create_experiment(
            slug=EXPERIMENT_SLUG,
            config_hash=_config_hash(config),
            config_json=config,
            n_ticks=N_TICKS,
        )
        experiment_id = exp_resp.experiment_id
        log.info(
            "Experiment: %s  (created=%s)", experiment_id, exp_resp.created
        )

        # --- Register participant ---------------------------------------------
        part_resp = session.upsert_participant(
            model=MODEL_NAME, rep=0, starting_cash=STARTING_CASH
        )
        participant_idx = part_resp.participant_idx
        log.info(
            "Participant idx=%d  (created=%s)", participant_idx, part_resp.created
        )

        tick_count = 0
        net_fail_count = 0
        net_blackout_started: datetime | None = None

        # --- Tick loop --------------------------------------------------------
        while True:
            try:
                lease = session.claim_tick()
            except Exception as net_err:
                net_fail_count += 1
                if net_blackout_started is None:
                    net_blackout_started = datetime.now(UTC)
                blackout_s = (datetime.now(UTC) - net_blackout_started).total_seconds()
                delay = min(NETWORK_BACKOFF_BASE * (2 ** (net_fail_count - 1)), NETWORK_BACKOFF_MAX)
                if blackout_s >= NETWORK_BLACKOUT_ALARM:
                    log.error(
                        "claim_tick BLACKOUT %.1f min, retry #%d: %s; next attempt in %ds",
                        blackout_s / 60, net_fail_count, net_err, delay,
                    )
                else:
                    log.warning(
                        "claim_tick network error #%d: %s; retrying in %ds",
                        net_fail_count, net_err, delay,
                    )
                time.sleep(delay)
                continue

            if net_fail_count > 0 and net_blackout_started is not None:
                blackout_s = (datetime.now(UTC) - net_blackout_started).total_seconds()
                log.info(
                    "Network recovered after %d retries (%.1f min blackout)",
                    net_fail_count, blackout_s / 60,
                )
                net_fail_count = 0
                net_blackout_started = None

            if not lease.available:
                wait = lease.retry_after_sec or SLEEP_ON_NO_TICK
                log.info(
                    "No tick available (%s). Sleeping %ds ...",
                    lease.reason,
                    wait,
                )
                time.sleep(wait)
                continue

            tick_count += 1
            log.info("=== Tick #%d  id=%s ===", tick_count, lease.tick_id)

            try:
                # 1. Load candidate markets
                tick_data = session.load_candidates(lease)
                lease = tick_data.lease
                candidates = tick_data.candidates
                log.info("Markets available: %d", candidates.market_count)

                # 2. Peek at current portfolio (drives position-aware filtering)
                portfolio = session.get_portfolio(participant_idx)
                if portfolio:
                    log.info(
                        "Portfolio: cash=$%.2f  equity=$%.2f  pnl=$%.2f  positions=%d",
                        float(portfolio.cash),
                        float(portfolio.equity),
                        float(portfolio.total_pnl),
                        len(portfolio.positions),
                    )
                positions_by_market = _shares_by_side(portfolio)

                # 3. Prepare market summaries for LLM, skipping any market
                #    where we're already at cap or holding both sides.
                market_list = build_market_list(candidates.markets, positions_by_market)
                skipped = min(len(candidates.markets), MAX_MARKETS) - len(market_list)
                if skipped:
                    log.info("Skipping %d market(s) (locked or at cap)", skipped)

                if not market_list:
                    log.info("No tradeable markets this tick.")
                    decisions = []
                else:
                    # 4. Ask Claude for trading decisions
                    log.info("Analysing %d markets with Claude...", len(market_list))
                    decisions = analyse_markets(llm, market_list)
                    trades = [d for d in decisions if d.get("action") != "HOLD"]
                    log.info(
                        "Decisions: %d trade(s), %d hold(s)",
                        len(trades),
                        len(decisions) - len(trades),
                    )
                    for t in trades:
                        log.info("  %s  %s — %s", t["action"], t["market_id"], t["reasoning"])

                # 5. Persist plan
                session.put_plan(
                    lease,
                    participant_idx=participant_idx,
                    plan_json={"decisions": decisions},
                )

                # 6. Submit trade intents (with cap + opposite-side guard)
                intents = decisions_to_intents(decisions, positions_by_market)
                if intents:
                    result = session.submit_intents(
                        lease,
                        participant_idx=participant_idx,
                        intents=intents,
                    )
                    log.info(
                        "Submitted %d intent(s): %d filled, %d rejected",
                        len(intents),
                        result.accepted,
                        result.rejected,
                    )
                    for fill in result.fills:
                        log.info(
                            "  FILL  %s %s x%s @ %s",
                            fill.action,
                            fill.side,
                            fill.shares,
                            fill.price,
                        )
                    for rej in result.rejections:
                        log.info("  REJECTED  %s", rej.reason)
                else:
                    log.info("No trades this tick (all HOLD)")

                # 7. Finalize participant
                session.finalize(lease, participant_idx=participant_idx)

            except Exception as exc:
                log.exception("Error during tick %s: %s", lease.tick_id, exc)
                try:
                    session.finalize(
                        lease,
                        participant_idx=participant_idx,
                        status="FAILED",
                        error_code="AGENT_ERROR",
                        error_detail=str(exc)[:200],
                    )
                except Exception:
                    pass

            finally:
                # 8. Complete tick (releases lease)
                try:
                    session.complete_tick(lease)
                except Exception as e:
                    log.warning("complete_tick failed: %s", e)

    log.info("Agent finished after %d ticks.", tick_count)


if __name__ == "__main__":
    run()
