<img width="700" height="120" alt="image" src="https://github.com/user-attachments/assets/79e93f2e-7b7d-48b4-b61b-f19ff46c047c" />
<!-- <img width="1440" height="1080" alt="image" src="https://github.com/user-attachments/assets/9c93db8a-1b6a-43f7-b70a-f01b36499999" /> -->
<!-- <img width="1246" height="952" alt="image" src="https://github.com/user-attachments/assets/7efe8e45-f433-43b4-8c9e-afdb29df97d2" /> -->
<!-- <p align="center"> -->
<img src="https://github.com/user-attachments/assets/66fd3b01-0496-4f92-be74-26c2c38c89a9" alt="PRIMA-o logo" width="700" height="500"/>
<!-- </p> -->

<h1 align="left">PRIMA -o</h1>
<p align="left">
Prediction-market Reasoning Infrastructure for Multi-trial Agents
</p>
<p align="center">
  Orchestrated via LangGraph, built upon Prophet Arena &amp; Murphy's BLF systems,<br/>
  we develop what neither does alone: route by domain, refine by evidence,<br/>
  know when to trust the crowd, and when to beat it.
</p>

# Prophet Arena Sigma Lab AI Forecasting Track

**Team:** Cracked · **Track:** AI Forecasting · **Hackathon:** Uncommon Hacks 2026, ProphetHacks 2026

> "Calibration error matters a lot to risk control. High market return ≠ good Brier score."
> — Professor Haifeng, Sigma Lab

---

## Problem Statement

Each market is a binary yes/no question with a current crowd price and a future resolution outcome.

**Our goal:** produce probability estimates `p_yes ∈ [0, 1]` that are *more accurate than the crowd*, scored by:

```
Brier = (p_yes - outcome)²    [lower is better, perfect = 0, random = 0.25]
```

Copying the market price gives Brier = market baseline. **To win, we must beat the crowd.**

---

## Architecture — 8-Layer LangGraph State Machine

```
Market Question + p_market (crowd price)
         │
┌────────▼──────────────────────────────────────────────────────┐
│  LAYER 2 · Dual Router  (Claude Haiku — cheap + fast)        │
│  ► category: sports | finance | politics | science | culture  │
│  ► time_bucket: long | medium | short | urgent                │
│  ► w(t): 0.20 ──────────────────────────────────────► 0.85   │
│           (trust model, far)              (trust market, near) │
└────────┬──────────────────────────────────────────────────────┘
         │
┌────────▼────────────────┐    ┌──────────────────────────────┐
│  LAYER 3 · Research     │    │  LAYER 4 · ML Predictor      │
│  Sports: ESPN + Elo     │    │  Sports: Logistic on Elo     │
│  Others: Brave recursive│    │    logit(p) = β·ΔElo/400     │
│    search (2 iterations)│    │           + β·home_adv + ...  │
│  Auto query generation  │    │  Others:  p_market (fallback) │
└────────┬────────────────┘    └──────────┬───────────────────┘
         └──────────────┬─────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│  LAYER 5 · Synthesis  (Logit-mean ensemble)                   │
│  Providers: Claude Sonnet + GPT-4o-mini + Gemini Flash        │
│  Combines in logit space (not raw prob) → proper ensemble     │
│  + FAISS analogue retrieval for base-rate context             │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│  LAYER 6 · Critic + Recursive Refinement  (Claude Haiku)     │
│  Five rules enforced:                                         │
│  ① Empty new_queries → no loop                               │
│  ② Monotonicity gate: KL(p_new ‖ p_old) > threshold         │
│  ③ Hard cap: max 2 iterations                                │
│  ④ Variance check → fallback to p_market on oscillation      │
│  ⑤ Calibration applied AFTER the loop                        │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│  LAYER 7 · Calibration + Market Blend                        │
│  p_final = w(t) · p_market + (1 - w(t)) · Platt(p_llm_raw)  │
│  Platt scaler: fit on historical forecasts, corrects LLM     │
│  overconfidence. w(t) learned from Brier-vs-horizon data.    │
└───────────────────────┬───────────────────────────────────────┘
                        │
                   p_yes ∈ [0, 1]
                   logged to SQLite / Snowflake
```

---

## Three Key Innovations

### 1. Recursive Search with Monotonicity Gate
We do not just search once. The critic identifies specific information gaps and generates targeted follow-up queries. A **monotonicity gate** (KL divergence check) ensures each refinement round actually moves the probability estimate — if the new search found nothing new, the refinement is rejected without wasting an LLM call. Hard cap: 2 iterations maximum.

### 2. Domain Specialists + Elo ML Head
Sports markets use a **logistic regression on Elo difference** — mathematically the correct parametric form since Elo is designed to map to win-probability through the logistic function:

```
logit(p_yes) = β₀ + β₁·(Elo_home − Elo_away)/400 + β₂·home_adv + β₃·rest_diff + β₄·form_diff
```

Trained on 6,000 synthetic games with realistic Elo distributions. Coefficients are interpretable and directly meaningful: learned `home_advantage ≈ 0.40 logit units ≈ 10% win-probability boost`.

### 3. Learned Calibration + Time-Dependent Market Blend
Two-stage correction applied *after* the full reasoning loop:
- **Platt scaler**: corrects systematic LLM overconfidence (says 0.85, truth is 0.72)
- **w(t) blend**: `p_final = w(t)·p_market + (1-w(t))·p_calibrated`
  - Long horizon (>4d): `w=0.20` — trust our model, market is noisy
  - Urgent (<3h): `w=0.85` — trust the market, it's sharper near resolution

---

## Results

| Phase | System | Brier ↓ | vs Market |
|-------|--------|---------|-----------|
| Baseline | Market price only | 0.2372 | — |
| Phase 1 | Stubs (sanity check) | 0.2372 | 0.000 |
| Phase 2 | LLM router + Elo ML | **0.2021** | **−0.035 (−15%)** |
| Phase 3 | + Recursive search + Sonnet synthesis | 0.2107 | −0.027 |
| Phase 4 | + Critic + Platt calibration | 0.2309 | −0.006 |
| **Phase 5** | **+ Logit-mean ensemble + analogues** | **0.2130** | **−0.024** |

**By category (full pipeline, stub dataset):**

| Category | Brier | n | vs Market |
|----------|-------|---|-----------|
| politics | 0.140 | 1 | Better |
| sports | 0.249 | 4 | Better |

**By time bucket:**

| Bucket | Brier | w(t) | Strategy |
|--------|-------|------|----------|
| long (>4d) | **0.184** | 0.20 | Trust our model |
| medium (1-4d) | 0.270 | 0.45 | Balanced |

**Key insight:** `long` bucket is where we add the most value — far from resolution, the market is noisy and our research + Elo ML gives real edge. Near resolution (`urgent`), w(t)=0.85 collapses us toward the market which is always correct near settlement.

> Benchmark run on 5-market stub dataset. Numbers are directionally correct; final scores scale with dataset size.

---

## Snowflake Integration

Three concrete uses:
1. **Forecast log** → `FORECASTS` table, every prediction stored with full state
2. **Cortex Search** → vector index over past rationales, powers analogue RAG
3. **Cortex Analyst** → nightly Platt refit + w(t) curve update

Fallback: SQLite + FAISS (runs in any environment without credentials).

---

## Setup

```bash
# 1. Clone and enter
cd ai-prophet/forecast/

# 2. Create virtualenv
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — minimum required:
#   ANTHROPIC_API_KEY=sk-ant-...
#   BRAVE_API_KEY=BSA-...        (strongly recommended)

# 5. Train ML models
python scripts/train_ml.py --all

# 6. Fit calibrator
python scripts/fit_calibrator.py
```

---

## Run

### Replay benchmark (evaluation mode)
```bash
# Built-in 5-market stub dataset
python scripts/run_replay.py --dataset small

# Real dataset from ai-prophet-datasets
python scripts/run_replay.py --dataset path/to/dataset.json

# With limit for quick tests
python scripts/run_replay.py --dataset small --limit 3
```

Outputs:
- Brier score per category and time bucket (printed to console)
- `outputs/forecasts.csv` — all predictions with rationale
- `outputs/calibration.png` — reliability diagram
- `outputs/calibration_platt.png` — before/after Platt scaling

### Live dashboard
```bash
streamlit run dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

### Train ML models
```bash
python scripts/train_ml.py --sport    # sports Elo model
python scripts/train_ml.py --all      # all available models
```

### Fit calibrator
```bash
python scripts/fit_calibrator.py                   # synthetic data
python scripts/fit_calibrator.py --db forecast_log.db  # from real forecasts
```

### Snowflake ingest (optional)
```bash
python snowflake/ingest.py --db forecast_log.db
```

---

## File Map

```
forecast/
├── README.md                              ← you are here
├── requirements.txt
├── .env.example
├── config/
│   ├── categories.yaml                    ← 5 categories + keyword lists
│   └── time_buckets.yaml                  ← 4 buckets + w(t) values
├── prophet_forecast/
│   ├── state.py                           ← ForecastState TypedDict
│   ├── graph.py                           ← LangGraph wiring
│   ├── nodes/
│   │   ├── router.py                      ← Layer 2: LLM classifier + time bucket
│   │   ├── research.py                    ← Layer 3: dispatcher + refinement merge
│   │   ├── ml_predictor.py                ← Layer 4: category ML head loader
│   │   ├── synthesis.py                   ← Layer 5: logit-mean ensemble
│   │   ├── critic.py                      ← Layer 6: 5-rule critic
│   │   ├── calibrator.py                  ← Layer 7: Platt + w(t) blend
│   │   └── logger_node.py                 ← Layer 8: SQLite / Snowflake log
│   ├── research/
│   │   ├── sports.py                      ← ESPN + Elo lookup
│   │   └── general.py                     ← Brave recursive search
│   ├── ml/
│   │   ├── base.py                        ← BaseMLPredictor ABC
│   │   ├── sports_model.py                ← LogReg on Elo
│   │   └── artifacts/                     ← trained model pickles
│   ├── tools/
│   │   ├── recursive_search.py            ← Brave search + extraction
│   │   └── refine_controller.py           ← monotonicity gate + variance check
│   ├── memory/
│   │   └── analogue_retrieval.py          ← FAISS / Cortex Search
│   └── eval/
│       ├── metrics.py                     ← Brier, log-loss, calibration plot
│       └── replay.py
├── scripts/
│   ├── run_replay.py                      ← main entry point
│   ├── train_ml.py                        ← train ML heads
│   └── fit_calibrator.py                  ← fit Platt scaler
├── dashboard/
│   └── streamlit_app.py                   ← live Streamlit dashboard
├── snowflake/
│   ├── schema.sql                         ← Snowflake DDL + Cortex Search setup
│   └── ingest.py                          ← push SQLite → Snowflake
└── tests/
```

---

## Submission

**Run command (for judges):**
```bash
cd ai-prophet/forecast
pip install -r requirements.txt
python scripts/train_ml.py --all
python scripts/fit_calibrator.py
python scripts/run_replay.py --dataset small
```

**Dashboard:**
```bash
streamlit run dashboard/streamlit_app.py
```
