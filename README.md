# SpendLens — Weekly Marketing Mix Copilot

**Client (fictional):** Aurel & Co., a single-market D2C skincare brand.
**Problem:** a small growth team with no data-science headcount needs to know
"where do I put next week's budget" — same-day, not after a month of modeling
— from a model that retrains automatically, stays interpretable, and doesn't
silently change its mind about a channel without telling anyone.

## What's actually novel here (read this before the code)

1. **Adstock + Hill saturation are fit *inside* the same ElasticNet pipeline**,
   via a joint randomized search — not hand-tuned per channel before
   regression. See `src/models/train_model.py`.
2. **Expanding-window (walk-forward) time-series CV**, not a single train/test
   split — the model is never validated on data "before" data it was trained
   on. See `src/models/cv_utils.py`.
3. **The model is validated against known ground truth**, not just fit to
   noisy weekly numbers — the synthetic dataset bakes in true adstock decay
   and saturation curves, and `src/models/evaluate.py` proves the pipeline
   recovers them (saturation shape: ~0.98 correlation; adstock decay:
   harder to identify for slow-decay channels — an honest, real MMM
   limitation, not hidden).
4. **Automated drift detection** flags any *channel's* coefficient that moves
   >25% week-over-week — the trust mechanism that lets a marketer act on a
   weekly retrain without a data scientist reviewing every run.
5. **Budget reallocation is a real constrained optimization problem**
   (SLSQP, concave objective from Hill saturation), not a spreadsheet
   heuristic — see `src/optimization/budget_optimizer.py`.
6. **Non-media drivers are modeled as explicit controls**, not left for media
   coefficients to silently absorb: price, discount, stock-outs, delivery
   delays, macro/category demand, seasonality. This is what separates an
   MMM from "regress revenue on ad spend." See the dataset generator's
   docstring for the full reasoning.

## Project structure

```
spendlens-mmm/
├── data/
│   ├── raw/                 aurel_weekly_observed.csv (source of truth)
│   ├── processed/           modeling_dataset.csv (cleaned, calendar-checked)
│   └── external/            calendar_events.csv (festive/sale calendar)
├── notebooks/                EDA, feature engineering, validation (exploration only — logic lives in src/)
├── src/
│   ├── data/                 load_data.py, preprocess.py
│   ├── features/              adstock.py, saturation.py, build_features.py
│   ├── models/                train_model.py, evaluate.py, cv_utils.py
│   ├── optimization/          budget_optimizer.py
│   └── monitoring/            drift_check.py
├── configs/model_config.yaml  single source of truth for bounds/thresholds/paths
├── outputs/                   plots, JSON reports, model_artifacts (joblib)
├── scripts/weekly_retrain.py  the automation entrypoint (cron/GH Actions target)
├── dashboard/                 Streamlit app + chatbot
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline (first time)

```bash
python src/data/preprocess.py
python src/models/train_model.py       # joint hyperparameter search + refit
python src/models/evaluate.py          # recovered-vs-true validation
python src/optimization/budget_optimizer.py
```

## Simulate weekly automation

```bash
python scripts/weekly_retrain.py --weeks 1
```
This appends one new (simulated) week to `data/raw/`, retrains, runs the
drift check against the previous run, and re-optimizes next week's budget —
end to end, the way a scheduled job would. In production, step 1 (simulate)
is replaced by a real data pull; everything downstream is unchanged.

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Six sections + a chatbot:

| Tab | What it answers |
|---|---|
| 🟢 Overview | Revenue, spend, blended ROI, orders — the 10-second CMO view |
| 🔵 Channel Performance | Contribution %, ROI, spend-vs-ROI scatter |
| 🟡 Diminishing Returns | Steady-state spend→response curve per channel, saturation flag |
| 🔴 Budget Recommendation | Current vs. recommended spend, expected lift |
| 🟠 Trust & Stability | Coefficient trend across retrains, drift alerts, CV metrics |
| ⚪ External Factors | Seasonality and promo impact on demand |
| 💬 Ask SpendLens | NL chatbot, retrieval-grounded over the same numbers shown elsewhere |

### Chatbot / free LLM

The chatbot **retrieves structured facts first** (contribution, ROI, drift
flags, optimizer output), then either:
- phrases them via a free-tier **Groq-hosted Llama 3.1** call, if
  `GROQ_API_KEY` is set (get one free at console.groq.com), or
- falls back to a **plain-language template** with zero external calls.

The LLM is only ever asked to phrase numbers it's given — never to compute
or infer them. This keeps answers auditable and means the demo works fully
offline if no API key is configured:

```bash
export GROQ_API_KEY=your_key_here   # optional
streamlit run dashboard/app.py
```

## Known limitations (say these out loud in the interview — they're strengths, not weaknesses)

- **Adstock decay identifiability is weak for slow-decay channels** (TV/OOH,
  Influencer) with ~130 weeks of data — saturation *shape* recovers almost
  perfectly (0.98 avg correlation to ground truth), decay *speed* is
  genuinely harder to pin down. This is a real, known MMM limitation, not a
  bug.
- **`website_sessions` / `branded_search_index` are mediators**, not pure
  controls — they're generated as diagnostic signals but deliberately
  excluded from the regression to avoid post-treatment bias. See
  `generate_dataset.py`'s docstring.
- **The synthetic-week simulator in `weekly_retrain.py`** uses a simplified
  demand proxy (not the full ground-truth generator) to keep the automation
  demo self-contained — good enough to exercise drift/retrain mechanics, not
  a substitute for `generate_dataset.py`'s validated ground truth.
