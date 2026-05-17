"""
Prophet Forecast — Streamlit Dashboard

Shows:
  - Live forecast table (from SQLite log)
  - Calibration curve
  - Per-category and per-bucket Brier scores
  - Evidence panel per forecast

Run:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow imports from forecast/ root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH      = os.environ.get("FORECAST_DB_PATH", "forecast_log.db")
OUTPUTS_DIR  = Path(__file__).parent.parent / "outputs"
REFRESH_SECS = 30

st.set_page_config(
    page_title="Prophet Forecast",
    page_icon="🔮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_SECS)
def load_forecasts() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM forecasts ORDER BY logged_at DESC LIMIT 500", conn
    )
    conn.close()
    return df


def brier(df: pd.DataFrame) -> float | None:
    resolved = df.dropna(subset=["p_final", "outcome"])
    if resolved.empty:
        return None
    return float(((resolved["p_final"] - resolved["outcome"]) ** 2).mean())


def market_brier(df: pd.DataFrame) -> float | None:
    resolved = df.dropna(subset=["p_market", "outcome"])
    if resolved.empty:
        return None
    return float(((resolved["p_market"] - resolved["outcome"]) ** 2).mean())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔮 Prophet Forecast Dashboard")
st.caption("Sigma Lab AI Forecasting Track · Uprising Team")

col_refresh, col_db = st.columns([1, 4])
with col_refresh:
    if st.button("⟳ Refresh"):
        st.cache_data.clear()
with col_db:
    st.caption(f"DB: `{DB_PATH}` · auto-refresh every {REFRESH_SECS}s")

df = load_forecasts()

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

st.divider()
m1, m2, m3, m4, m5 = st.columns(5)

total    = len(df)
resolved = int(df["outcome"].notna().sum()) if (not df.empty and "outcome" in df.columns) else 0
our_b    = brier(df) if "outcome" in df.columns else None
mkt_b    = market_brier(df) if "outcome" in df.columns else None
delta    = (our_b - mkt_b) if (our_b is not None and mkt_b is not None) else None

m1.metric("Total Forecasts",    total)
m2.metric("Resolved",           resolved)
m3.metric("Our Brier",          f"{our_b:.4f}" if our_b is not None else "—")
m4.metric("Market Baseline",    f"{mkt_b:.4f}" if mkt_b is not None else "—")
m5.metric(
    "Delta vs Market",
    f"{delta:+.4f}" if delta is not None else "—",
    delta_color="inverse" if delta is not None else "off",
)

st.divider()

# ---------------------------------------------------------------------------
# Two-column layout: calibration + breakdowns
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration Curve")
    cal_path = OUTPUTS_DIR / "calibration.png"
    platt_path = OUTPUTS_DIR / "calibration_platt.png"
    if platt_path.exists():
        st.image(str(platt_path), caption="Before vs After Platt Scaling", use_container_width=True)
    elif cal_path.exists():
        st.image(str(cal_path), caption="Reliability Diagram", use_container_width=True)
    else:
        st.info("Run `python scripts/run_replay.py --dataset small` to generate calibration plot.")

with right:
    st.subheader("Brier Score by Category")
    if not df.empty and resolved > 0:
        resolved_df = df.dropna(subset=["outcome"])
        resolved_df = resolved_df.copy()
        resolved_df["brier"] = (resolved_df["p_final"] - resolved_df["outcome"]) ** 2
        resolved_df["mkt_brier"] = (resolved_df["p_market"] - resolved_df["outcome"]) ** 2

        by_cat = (
            resolved_df.groupby("category")
            .agg(n=("brier", "count"), our_brier=("brier", "mean"), mkt_brier=("mkt_brier", "mean"))
            .reset_index()
        )
        by_cat["delta"] = by_cat["our_brier"] - by_cat["mkt_brier"]
        by_cat = by_cat.sort_values("our_brier")

        st.dataframe(
            by_cat.style.format({"our_brier": "{:.4f}", "mkt_brier": "{:.4f}", "delta": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Brier Score by Time Bucket")
        by_bucket = (
            resolved_df.groupby("time_bucket")
            .agg(n=("brier", "count"), our_brier=("brier", "mean"), mkt_brier=("mkt_brier", "mean"))
            .reset_index()
        )
        by_bucket["delta"] = by_bucket["our_brier"] - by_bucket["mkt_brier"]
        st.dataframe(
            by_bucket.style.format({"our_brier": "{:.4f}", "mkt_brier": "{:.4f}", "delta": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No resolved forecasts yet. Outcomes are revealed when markets close.")

st.divider()

# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------

st.subheader("Recent Forecasts")

if df.empty:
    st.warning(
        f"No forecasts found in `{DB_PATH}`. "
        "Run `python scripts/run_replay.py --dataset small` first."
    )
else:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", cats)
        buckets = ["All"] + sorted(df["time_bucket"].dropna().unique().tolist())
        sel_bucket = st.selectbox("Time Bucket", buckets)

    view = df.copy()
    if sel_cat != "All":
        view = view[view["category"] == sel_cat]
    if sel_bucket != "All":
        view = view[view["time_bucket"] == sel_bucket]

    display_cols = [
        "market_id", "category", "time_bucket",
        "p_market", "p_ml", "p_llm_raw", "p_final",
        "outcome", "rationale", "logged_at",
    ]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols].head(50).style.format({
            "p_market":  "{:.3f}",
            "p_ml":      "{:.3f}",
            "p_llm_raw": "{:.3f}",
            "p_final":   "{:.3f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    # Evidence expander per forecast
    st.divider()
    st.subheader("Evidence Inspector")
    market_ids = view["market_id"].dropna().unique().tolist()
    if market_ids:
        sel_market = st.selectbox("Select market to inspect", market_ids)
        row = view[view["market_id"] == sel_market].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Market Price",  f"{row.get('p_market', '—'):.3f}" if pd.notna(row.get('p_market')) else "—")
            st.metric("ML Estimate",   f"{row.get('p_ml', '—'):.3f}"     if pd.notna(row.get('p_ml'))     else "—")
            st.metric("LLM Raw",       f"{row.get('p_llm_raw', '—'):.3f}" if pd.notna(row.get('p_llm_raw')) else "—")
        with col_b:
            st.metric("Final p_yes",   f"{row.get('p_final', '—'):.3f}"  if pd.notna(row.get('p_final'))  else "—")
            st.metric("w(t) blend",    f"{row.get('w_t', '—'):.2f}"      if pd.notna(row.get('w_t'))      else "—")
            outcome_val = row.get('outcome')
            st.metric("Outcome",       int(outcome_val) if pd.notna(outcome_val) else "Pending")

        st.markdown("**Rationale:**")
        st.info(row.get("rationale") or "No rationale recorded.")

# ---------------------------------------------------------------------------
# Architecture diagram (for demo)
# ---------------------------------------------------------------------------

with st.expander("🏗️ Architecture Overview"):
    st.markdown("""
    ```
    Market Question
         │
    ┌────▼────────────────────────────────────────────────────┐
    │  Layer 2: Router (Claude Haiku)                        │
    │  → category: sports | finance | politics | sci | cult  │
    │  → time_bucket: long | medium | short | urgent         │
    │  → w(t): 0.20 → 0.85 (trust market near resolution)   │
    └────┬────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────┐  ┌─────────────────────────────┐
    │  Layer 3: Research   │  │  Layer 4: ML Predictor      │
    │  Sports: ESPN + Elo  │  │  Sports: LogReg on Elo      │
    │  Others: Brave       │  │  Others: p_market fallback  │
    │  recursive search    │  └──────────────┬──────────────┘
    └──────────┬───────────┘                 │
               └──────────────┬─────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────┐
    │  Layer 5: Synthesis (Claude Sonnet + logit ensemble)    │
    │  → structured p_yes + rationale + confidence           │
    └─────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────┐
    │  Layer 6: Critic (Claude Haiku) — 5 rules              │
    │  ① Empty queries = no loop                             │
    │  ② Monotonicity gate (KL threshold)                    │
    │  ③ Hard cap: max 2 iterations                          │
    │  ④ Variance check → fallback to p_market               │
    │  ⑤ Calibration after the loop                          │
    └─────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────┐
    │  Layer 7: Calibrator + w(t) blend                      │
    │  p_final = w(t)·p_market + (1-w(t))·p_calibrated      │
    └─────────────────────────┬───────────────────────────────┘
                              │
                         p_yes ∈ [0,1]
    ```
    """)

# Auto-refresh
st.markdown(
    f"""<script>
    setTimeout(function(){{window.location.reload()}}, {REFRESH_SECS * 1000});
    </script>""",
    unsafe_allow_html=True,
)
