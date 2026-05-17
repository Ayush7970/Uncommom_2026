"""
Refinement controller — enforces the 5 rules for the recursive critic loop.

Rules (from CLAUDE.md):
  1. Critic must return concrete new_queries; empty list = no loop
  2. Monotonicity gate: only accept refinement if KL(p_new || p_old) > threshold
  3. Hard cap: max 2 iterations (enforced in should_refine)
  4. Variance check: if estimates oscillate, fall back to p_market
  5. Calibration applied AFTER the loop, not inside it
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

KL_THRESHOLD      = 0.005   # minimum KL divergence to accept a refinement
VARIANCE_THRESHOLD = 0.04   # std dev across iterations triggers oscillation fallback


def kl_divergence(p: float, q: float, eps: float = 1e-7) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) — always >= 0, = 0 when p == q."""
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def monotonicity_gate(p_new: float, p_old: float) -> bool:
    """
    Rule 2: Return True (accept refinement) only if the new estimate
    differs enough from the old one to be worth keeping.
    A tiny KL means the new search changed nothing — reject the refinement.
    """
    kl = kl_divergence(p_new, p_old)
    accept = kl > KL_THRESHOLD
    log.debug("Monotonicity gate: p_old=%.3f  p_new=%.3f  KL=%.5f  accept=%s",
              p_old, p_new, kl, accept)
    return accept


def variance_check(p_iterations: list[float]) -> bool:
    """
    Rule 4: Return True (oscillating = bad) if the std dev of probability
    estimates across iterations exceeds VARIANCE_THRESHOLD.
    High variance means the model is confused — fall back to market price.
    """
    if len(p_iterations) < 2:
        return False
    mean = sum(p_iterations) / len(p_iterations)
    var  = sum((p - mean) ** 2 for p in p_iterations) / len(p_iterations)
    std  = math.sqrt(var)
    oscillating = std > VARIANCE_THRESHOLD
    if oscillating:
        log.warning("Variance check: std=%.4f > threshold=%.4f — oscillating, will fall back",
                    std, VARIANCE_THRESHOLD)
    return oscillating


def apply_monotonicity_or_fallback(
    p_new: float,
    p_old: float,
    p_market: float,
    p_iterations: list[float],
) -> tuple[float, bool]:
    """
    Combined check for the refinement loop.

    Returns (p_accepted, fell_back_to_market).
      - If oscillating: return (p_market, True)
      - If KL too small: return (p_old, False)  — reject new estimate silently
      - Otherwise:       return (p_new, False)   — accept refinement
    """
    if variance_check(p_iterations):
        log.info("Falling back to p_market=%.3f due to oscillation", p_market)
        return p_market, True

    if not monotonicity_gate(p_new, p_old):
        log.info("Monotonicity gate rejected: p_new=%.3f too close to p_old=%.3f", p_new, p_old)
        return p_old, False

    return p_new, False
