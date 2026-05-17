"""LangGraph wiring — connects all 8 layers into a single runnable graph."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from .nodes.calibrator import calibrator_node
from .nodes.critic import critic_node, should_refine
from .nodes.logger_node import logger_node
from .nodes.ml_predictor import ml_predictor_node
from .nodes.research import research_node
from .nodes.router import router_node
from .nodes.synthesis import synthesis_node
from .state import ForecastState

log = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph forecast pipeline."""
    graph = StateGraph(ForecastState)

    # --- Register nodes (one per layer) ---
    graph.add_node("router",       router_node)        # Layer 2
    graph.add_node("research",     research_node)      # Layer 3
    graph.add_node("ml_predictor", ml_predictor_node)  # Layer 4
    graph.add_node("synthesis",    synthesis_node)     # Layer 5
    graph.add_node("critic",       critic_node)        # Layer 6
    graph.add_node("calibrator",   calibrator_node)    # Layer 7
    graph.add_node("logger",       logger_node)        # Layer 8

    # --- Wire edges ---
    graph.set_entry_point("router")
    graph.add_edge("router",       "research")
    graph.add_edge("research",     "ml_predictor")
    graph.add_edge("ml_predictor", "synthesis")
    graph.add_edge("synthesis",    "critic")

    # Conditional: critic decides whether to refine or move to calibration
    graph.add_conditional_edges(
        "critic",
        should_refine,
        {
            "refine":    "research",   # loop back for another iteration
            "calibrate": "calibrator",
        },
    )

    graph.add_edge("calibrator", "logger")
    graph.add_edge("logger",     END)

    return graph.compile()


# Module-level compiled graph (import and call .invoke())
forecast_graph = build_graph()
