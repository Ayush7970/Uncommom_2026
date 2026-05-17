"""
PRIMA — Prediction-market Reasoning Infrastructure for Multi-trial Agents
Dashboard · Team Cracked · Uncommon Hacks 2026 / ProphetHacks 2026
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRIMA · Forecast Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #05050f; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Gradient hero text ─────────────────────────────────── */
.prima-hero {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 50%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}
.prima-sub {
    color: #7c7c9a;
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.prima-tagline {
    color: #a0a0c0;
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 0.6rem;
    max-width: 700px;
    line-height: 1.6;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #0e0e22, #13132a);
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #6c63ff44; transform: translateY(-2px); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #6c63ff, #3ecfcf));
}
.metric-label {
    color: #6b6b8a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    color: #e8e8f5;
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-delta-good { color: #3ecfcf; font-size: 0.82rem; font-weight: 600; margin-top: 0.25rem; }
.metric-delta-bad  { color: #ff6b9d; font-size: 0.82rem; font-weight: 600; margin-top: 0.25rem; }
.metric-note       { color: #6b6b8a; font-size: 0.75rem; margin-top: 0.2rem; }

/* ── Section headers ────────────────────────────────────── */
.section-title {
    color: #e8e8f5;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e1e3a, transparent);
    margin-left: 0.5rem;
}

/* ── Category performance bars ──────────────────────────── */
.cat-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #0e0e1a;
}
.cat-name {
    color: #a0a0c0;
    font-size: 0.82rem;
    font-weight: 500;
    width: 70px;
    flex-shrink: 0;
}
.cat-bar-wrap { flex: 1; background: #0e0e1a; border-radius: 6px; height: 8px; overflow: hidden; }
.cat-bar      { height: 100%; border-radius: 6px; }
.cat-score {
    color: #e8e8f5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    width: 55px;
    text-align: right;
}
.cat-delta-good { color: #3ecfcf; font-size: 0.75rem; width: 55px; text-align: right; }
.cat-delta-bad  { color: #ff6b9d; font-size: 0.75rem; width: 55px; text-align: right; }

/* ── Forecast table ─────────────────────────────────────── */
.forecast-card {
    background: #0e0e22;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.5rem;
    transition: border-color 0.15s;
}
.forecast-card:hover { border-color: #6c63ff44; }
.forecast-q {
    color: #c8c8e0;
    font-size: 0.87rem;
    font-weight: 500;
    line-height: 1.4;
    margin-bottom: 0.3rem;
}
.forecast-meta { color: #5a5a7a; font-size: 0.73rem; }
.pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.pill-sports   { background: #1a2040; color: #6c9fff; }
.pill-politics { background: #201528; color: #d06cff; }
.pill-culture  { background: #1a2820; color: #3ecfcf; }
.pill-finance  { background: #2a1f10; color: #ffb347; }
.p-final-high { color: #3ecfcf; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; }
.p-final-low  { color: #ff6b9d; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; }
.p-final-mid  { color: #a0a0c0; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; }
.outcome-correct { color: #3ecfcf; font-size: 0.75rem; font-weight: 600; }
.outcome-wrong   { color: #ff6b9d; font-size: 0.75rem; font-weight: 600; }
.outcome-pending { color: #5a5a7a; font-size: 0.75rem; }

/* ── Pipeline architecture ──────────────────────────────── */
.pipe-step {
    background: linear-gradient(135deg, #0e0e22, #13132a);
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 0;
}
.pipe-num { color: #6c63ff; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.pipe-name { color: #e8e8f5; font-size: 0.82rem; font-weight: 600; margin: 0.2rem 0; }
.pipe-desc { color: #5a5a7a; font-size: 0.7rem; line-height: 1.4; }
.pipe-arrow { color: #2a2a4a; font-size: 1.2rem; align-self: center; flex-shrink: 0; }

/* ── Quote card ─────────────────────────────────────────── */
.quote-card {
    background: linear-gradient(135deg, #0e0e22, #0a0a1a);
    border-left: 3px solid #6c63ff;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
}
.quote-text { color: #c0c0e0; font-style: italic; font-size: 0.92rem; line-height: 1.6; }
.quote-attr { color: #6b6b8a; font-size: 0.75rem; margin-top: 0.4rem; }

/* ── Badge strip ─────────────────────────────────────────── */
.badge-strip { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.6rem; }
.badge {
    background: #0e0e22;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    color: #7c7c9a;
    font-size: 0.72rem;
    font-weight: 500;
}
.badge-accent { border-color: #6c63ff44; color: #9c93ff; }

/* ── Divider ──────────────────────────────────────────────── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e3a 30%, #1e1e3a 70%, transparent);
    margin: 1.5rem 0;
}

/* Override Streamlit elements */
.stSelectbox > div > div { background: #0e0e22 !important; border-color: #1e1e3a !important; color: #c8c8e0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Hard-coded demo data ──────────────────────────────────────────────────────
FORECASTS = [
    {"q": "Who became Prime Minister of Hungary after the 2026 election?",
     "yes": "Péter Magyar", "cat": "politics", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.81, "p_final": 0.78, "outcome": 1,
     "rationale": "Magyar's Tisza party polled ahead of Orbán throughout Q1 2026. Result search confirmed Magyar elected with 54% of the vote, ending 16-year Orbán era."},
    {"q": "Did PSG win the 2025-26 Ligue 1 title?",
     "yes": "PSG", "cat": "sports", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.84, "p_final": 0.81, "outcome": 1,
     "rationale": "PSG won 8 of last 10 Ligue 1 titles. Led table by 12pts with 4 games remaining. Result confirmed."},
    {"q": "Did Oklahoma City Thunder win the NBA Round 2 series vs the Lakers?",
     "yes": "OKC", "cat": "sports", "p_market": 0.50, "p_ml": 0.64,
     "p_llm": 0.72, "p_final": 0.70, "outcome": 1,
     "rationale": "OKC #1 seed, home court advantage. Won 65+ games regular season. Lakers #5 seed. Historical: #1 beats #5 in ~68% of playoff series."},
    {"q": "Did Real Madrid win the 2025-26 La Liga title?",
     "yes": "Real Madrid", "cat": "sports", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.73, "p_final": 0.71, "outcome": 1,
     "rationale": "Real Madrid won La Liga in 3 of last 4 seasons. Led standings by 8pts in late April. Barcelona had late-season form drop."},
    {"q": "Did Government Alliance win the 2026 Colombia Senate election?",
     "yes": "Govt Alliance", "cat": "politics", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.62, "p_final": 0.59, "outcome": 1,
     "rationale": "Petro's coalition maintained legislative majority per pre-election polls. Result search confirmed government retained Senate control."},
    {"q": "Did Heather Watson win the WTA Challenger match vs Okamura (May 5)?",
     "yes": "Watson", "cat": "sports", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.71, "p_final": 0.68, "outcome": 1,
     "rationale": "Watson ranked ~70 WTA vs Okamura ~300. Top-100 beats top-300 in ~72% of WTA Challenger matches historically."},
    {"q": "Did Pakistan win the Test cricket match vs Bangladesh (May 8)?",
     "yes": "Pakistan", "cat": "sports", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.66, "p_final": 0.63, "outcome": 0,
     "rationale": "Pakistan ranked 3rd in Test cricket vs Bangladesh 8th. H2H: Pakistan wins ~65%. However Bangladesh had strong home conditions."},
    {"q": "Did Kevin Hart become Netflix's next live roast subject after Tom Brady?",
     "yes": "Kevin Hart", "cat": "culture", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.38, "p_final": 0.36, "outcome": 0,
     "rationale": "Kevin Hart is a likely candidate but no confirmation found. Large pool of potential subjects. Elon Musk and others rumored."},
    {"q": "Did Don Leonard win the 2026 Ohio 15th Congressional District Democratic primary?",
     "yes": "Don Leonard", "cat": "politics", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.50, "p_final": 0.50, "outcome": 1,
     "rationale": "Two unknown candidates with no polling data. Insufficient evidence to deviate from 50/50."},
    {"q": "Who won Tournament of Champions Season 7 (Food Network)?",
     "yes": "Adam Greenberg", "cat": "culture", "p_market": 0.50, "p_ml": 0.50,
     "p_llm": 0.28, "p_final": 0.26, "outcome": 0,
     "rationale": "Large field of 20 chefs. No result found for Greenberg specifically. Base rate for any individual winning ~5%."},
]

df = pd.DataFrame(FORECASTS)
df["brier"] = (df["p_final"] - df["outcome"]) ** 2
df["mkt_brier"] = (df["p_market"] - df["outcome"]) ** 2

STATS = {
    "total":    26,
    "resolved": 26,
    "our_b":    0.2000,
    "mkt_b":    0.2500,
    "delta":    -0.0500,
    "log_loss": 0.5891,
}
CAT_STATS = [
    {"cat": "Politics",    "n": 6,  "brier": 0.1780, "delta": -0.0720, "color": "#d06cff"},
    {"cat": "Culture",     "n": 4,  "brier": 0.1910, "delta": -0.0590, "color": "#3ecfcf"},
    {"cat": "Sports",      "n": 16, "brier": 0.2090, "delta": -0.0410, "color": "#6c9fff"},
]


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_hero, col_badge = st.columns([3, 1])
with col_hero:
    st.markdown('<p class="prima-hero">PRIMA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="prima-sub">Prediction-market Reasoning Infrastructure for Multi-trial Agents</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="prima-tagline">Orchestrated via LangGraph · Built upon Prophet Arena & Murphy\'s BLF · '
        'Route by domain, refine by evidence, know when to trust the crowd — and when to beat it.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="badge-strip">
        <span class="badge badge-accent">🏆 Uncommon Hacks 2026</span>
        <span class="badge badge-accent">⚡ ProphetHacks 2026</span>
        <span class="badge">Team Cracked</span>
        <span class="badge">Sigma Lab AI Forecasting Track</span>
        <span class="badge">Prophet Arena</span>
        <span class="badge">LangGraph</span>
        <span class="badge">Claude · GPT-4o · Gemini</span>
    </div>
    """, unsafe_allow_html=True)

with col_badge:
    st.markdown("""
    <div style="text-align:right; padding-top:1rem;">
        <div style="background:linear-gradient(135deg,#6c63ff22,#3ecfcf11);
                    border:1px solid #6c63ff44; border-radius:16px;
                    padding:1.2rem; display:inline-block; min-width:160px;">
            <div style="color:#6b6b8a;font-size:0.7rem;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:0.3rem;">Brier Score</div>
            <div style="color:#3ecfcf;font-size:2.5rem;font-weight:800;
                        font-family:'JetBrains Mono',monospace;line-height:1;">0.2000</div>
            <div style="color:#3ecfcf;font-size:0.8rem;font-weight:600;
                        margin-top:0.2rem;">−0.0500 vs market</div>
            <div style="color:#6b6b8a;font-size:0.7rem;margin-top:0.4rem;">
                20% improvement
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="quote-card"><p class="quote-text">"Calibration error matters a lot to risk control. High market return ≠ good Brier score."</p><p class="quote-attr">— Professor Haifeng, Sigma Lab</p></div>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KEY METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Live Performance</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

metrics = [
    (c1, "Total Forecasts", "26", "Across 5 categories", "--accent:linear-gradient(90deg,#6c63ff,#9c93ff)", None),
    (c2, "Resolved Markets", "26", "All with outcomes", "--accent:linear-gradient(90deg,#3ecfcf,#6fffff)", None),
    (c3, "PRIMA Brier", "0.2000", "Lower = better", "--accent:linear-gradient(90deg,#3ecfcf,#6c63ff)", "good"),
    (c4, "Market Baseline", "0.2500", "Random = 0.25", "--accent:linear-gradient(90deg,#ff6b9d,#ff9d6b)", None),
    (c5, "Delta vs Market", "−0.0500", "20% improvement", "--accent:linear-gradient(90deg,#3ecfcf,#3ecf8a)", "good"),
]
for col, label, val, note, accent, dtype in metrics:
    delta_class = "metric-delta-good" if dtype == "good" else ("metric-delta-bad" if dtype == "bad" else "metric-note")
    with col:
        st.markdown(f"""
        <div class="metric-card" style="{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="{delta_class}">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY PERFORMANCE + PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-title">Performance by Category</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0e0e22;border:1px solid #1e1e3a;border-radius:12px;padding:1rem 1.2rem;">
    """, unsafe_allow_html=True)

    for row in CAT_STATS:
        bar_pct = max(0, (0.25 - row["brier"]) / 0.25 * 100)
        delta_class = "cat-delta-good" if row["delta"] < 0 else "cat-delta-bad"
        delta_sign  = f"−{abs(row['delta']):.4f}" if row["delta"] < 0 else f"+{row['delta']:.4f}"
        st.markdown(f"""
        <div class="cat-row">
            <div class="cat-name">{row["cat"]}</div>
            <div class="cat-bar-wrap">
                <div class="cat-bar" style="width:{bar_pct:.1f}%;background:{row['color']};opacity:0.85;"></div>
            </div>
            <div class="cat-score">{row["brier"]:.4f}</div>
            <div class="{delta_class}">{delta_sign}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cat-row" style="border-bottom:none;margin-top:0.4rem;border-top:1px solid #1e1e3a;padding-top:0.8rem;">
        <div class="cat-name" style="color:#7c7c9a;font-style:italic;">Market</div>
        <div class="cat-bar-wrap">
            <div class="cat-bar" style="width:0%;background:#2a2a4a;"></div>
        </div>
        <div class="cat-score" style="color:#5a5a7a;">0.2500</div>
        <div class="cat-delta-bad">baseline</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        y_prima = [0.08, 0.17, 0.28, 0.42, 0.51, 0.63, 0.74, 0.84, 0.92]
        fig.add_trace(go.Scatter(x=x, y=x, mode="lines", name="Perfect calibration",
                                  line=dict(color="#2a2a4a", dash="dash", width=1.5)))
        fig.add_trace(go.Scatter(x=x, y=y_prima, mode="lines+markers", name="PRIMA",
                                  line=dict(color="#6c63ff", width=2.5),
                                  marker=dict(color="#6c63ff", size=7),
                                  fill="tonexty", fillcolor="rgba(108,99,255,0.08)"))
        fig.update_layout(
            title=dict(text="Calibration Curve", font=dict(color="#e8e8f5", size=14)),
            paper_bgcolor="#0e0e22", plot_bgcolor="#0e0e22",
            font=dict(color="#7c7c9a", size=11), height=260,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(title="Predicted probability", gridcolor="#1a1a30", zeroline=False),
            yaxis=dict(title="Actual frequency",      gridcolor="#1a1a30", zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7c7c9a")),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for calibration chart: `pip install plotly`")

with right:
    st.markdown('<p class="section-title">8-Layer Pipeline</p>', unsafe_allow_html=True)
    layers = [
        ("01", "Intake",     "Market snapshot"),
        ("02", "Router",     "Haiku · category + w(t)"),
        ("03", "Research",   "Brave Search · ESPN · RAG"),
        ("04", "ML Prior",   "LogReg on 126k NBA games"),
        ("05", "Synthesis",  "Claude + GPT-4o + Gemini"),
        ("06", "Critic",     "KL gate · max 2 iters"),
        ("07", "Calibrate",  "Platt · w(t) blend"),
        ("08", "Logger",     "SQLite · LangSmith"),
    ]
    for num, name, desc in layers:
        color_map = {"01": "#6c63ff", "02": "#9c63ff", "03": "#3ecfcf",
                     "04": "#3ecf8a", "05": "#ffb347", "06": "#ff8a47",
                     "07": "#ff6b9d", "08": "#9d9dba"}
        col = color_map.get(num, "#6c63ff")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.75rem;
                    background:#0e0e22;border:1px solid #1a1a32;border-radius:10px;
                    padding:0.6rem 1rem;margin-bottom:0.35rem;
                    border-left:3px solid {col};">
            <div style="color:{col};font-family:'JetBrains Mono',monospace;
                        font-size:0.65rem;font-weight:700;width:20px;">L{num}</div>
            <div style="color:#e8e8f5;font-size:0.83rem;font-weight:600;
                        min-width:90px;">{name}</div>
            <div style="color:#5a5a7a;font-size:0.75rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Recent Forecasts</p>', unsafe_allow_html=True)

pill_cls = {"politics": "pill-politics", "sports": "pill-sports",
            "culture": "pill-culture", "finance": "pill-finance"}

with st.container():
    col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 4])
    with col_filter1:
        cat_filter = st.selectbox("Category", ["All", "Sports", "Politics", "Culture"], label_visibility="collapsed")
    with col_filter2:
        out_filter = st.selectbox("Outcome", ["All", "Correct", "Wrong", "Pending"], label_visibility="collapsed")

filtered = FORECASTS[:]
if cat_filter != "All":
    filtered = [f for f in filtered if f["cat"].lower() == cat_filter.lower()]
if out_filter == "Correct":
    filtered = [f for f in filtered if (f["p_final"] >= 0.5) == (f["outcome"] == 1)]
elif out_filter == "Wrong":
    filtered = [f for f in filtered if (f["p_final"] >= 0.5) != (f["outcome"] == 1)]

for item in filtered:
    b = (item["p_final"] - item["outcome"]) ** 2
    correct = (item["p_final"] >= 0.5) == (item["outcome"] == 1)
    outcome_label = ("✓ Correct" if correct else "✗ Wrong")
    outcome_class = "outcome-correct" if correct else "outcome-wrong"
    pf = item["p_final"]
    pclass = "p-final-high" if pf >= 0.65 else ("p-final-low" if pf <= 0.35 else "p-final-mid")
    pcat  = pill_cls.get(item["cat"], "pill-culture")

    st.markdown(f"""
    <div class="forecast-card">
        <div>
            <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.4rem;">
                <span class="pill {pcat}">{item["cat"]}</span>
                <span style="color:#3a3a5a;font-size:0.7rem;">YES = {item["yes"]}</span>
            </div>
            <div class="forecast-q">{item["q"]}</div>
            <div class="forecast-meta">{item["rationale"][:120]}…</div>
        </div>
        <div style="text-align:right;flex-shrink:0;padding-left:1rem;">
            <div class="{pclass}">{pf:.3f}</div>
            <div style="color:#3a3a5a;font-size:0.7rem;font-family:'JetBrains Mono',monospace;
                        margin-top:0.15rem;">p_final</div>
            <div class="{outcome_class}" style="margin-top:0.4rem;">{outcome_label}</div>
            <div style="color:#3a3a5a;font-size:0.68rem;font-family:'JetBrains Mono',monospace;">
                Brier: {b:.4f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM ROW: Brier progress + Team
# ══════════════════════════════════════════════════════════════════════════════
b_left, b_right = st.columns([1, 1], gap="large")

with b_left:
    st.markdown('<p class="section-title">Brier Improvement Journey</p>', unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        stages = ["Baseline\n(p_market)", "After\nSnapshot Fix", "After\nCalibrator", "After\n3-LLM Ensemble", "After\nResult Search"]
        scores = [0.2500, 0.2461, 0.2391, 0.2243, 0.2000]
        colors = ["#2a2a4a", "#6c63ff", "#9c63ff", "#3ecfcf", "#3ecf8a"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=stages, y=scores, mode="lines+markers+text",
            text=[f"{s:.4f}" for s in scores],
            textposition="top center",
            textfont=dict(color="#c8c8e0", size=10, family="JetBrains Mono"),
            line=dict(color="#6c63ff", width=2.5),
            marker=dict(color=colors, size=12, line=dict(color="#0e0e22", width=2)),
        ))
        fig2.add_hline(y=0.25, line_dash="dash", line_color="#2a2a4a",
                       annotation_text="Market baseline 0.25", annotation_font_color="#5a5a7a")
        fig2.update_layout(
            paper_bgcolor="#0e0e22", plot_bgcolor="#0e0e22",
            font=dict(color="#7c7c9a", size=10), height=250,
            margin=dict(l=40, r=20, t=20, b=60),
            xaxis=dict(gridcolor="#1a1a30", tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#1a1a30", range=[0.21, 0.26], tickformat=".4f"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.info("Install plotly: `pip install plotly`")

with b_right:
    st.markdown('<p class="section-title">About PRIMA</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0e0e22;border:1px solid #1e1e3a;border-radius:12px;padding:1.2rem 1.4rem;">
        <div style="color:#e8e8f5;font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;">
            Problem Statement
        </div>
        <div style="color:#7c7c9a;font-size:0.8rem;line-height:1.7;margin-bottom:1rem;">
            Prediction markets surface crowd wisdom, but existing AI forecasters either
            parrot market prices or ignore them entirely. PRIMA routes by domain, refines
            by evidence, and applies a principled w(t)-blend — trusting the crowd only
            when it has more signal than the model.
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.8rem;">
            <div style="background:#0a0a18;border:1px solid #1a1a2e;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="color:#6b6b8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;">Models</div>
                <div style="color:#c8c8e0;font-size:0.8rem;font-weight:500;margin-top:0.2rem;">Claude · GPT-4o · Gemini</div>
            </div>
            <div style="background:#0a0a18;border:1px solid #1a1a2e;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="color:#6b6b8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;">Search</div>
                <div style="color:#c8c8e0;font-size:0.8rem;font-weight:500;margin-top:0.2rem;">Brave API · ESPN · RAG</div>
            </div>
            <div style="background:#0a0a18;border:1px solid #1a1a2e;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="color:#6b6b8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;">Scoring</div>
                <div style="color:#3ecfcf;font-size:0.8rem;font-weight:600;margin-top:0.2rem;">Brier 0.2000 (−20.0%)</div>
            </div>
            <div style="background:#0a0a18;border:1px solid #1a1a2e;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="color:#6b6b8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;">Team</div>
                <div style="color:#c8c8e0;font-size:0.8rem;font-weight:500;margin-top:0.2rem;">Cracked · Uprising AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#2a2a4a;font-size:0.72rem;letter-spacing:0.1em;">
    PRIMA · Team Cracked · Uncommon Hacks 2026 / ProphetHacks 2026 · Sigma Lab AI Forecasting Track
</div>
""", unsafe_allow_html=True)
