"""
chatbot.py
==========
"Explainability copilot" for the dashboard: answers natural-language
questions about the model using RETRIEVAL over structured, already-computed
facts (contributions, ROI, drift flags, optimizer output) -- never lets an
LLM freely reason about numbers it wasn't given, which is what keeps this
auditable instead of hallucination-prone.

Two modes:
    1. LLM-phrased (if GROQ_API_KEY is set): retrieved facts are handed to
       a free-tier Groq-hosted model (Llama 3.1) purely to phrase them as a
       natural sentence. The model is NOT asked to compute or infer numbers
       -- only to read back what's already in the context.
    2. Template fallback (no API key / no network): the same retrieved
       facts are formatted with an f-string template. This means the demo
       works fully offline -- a hackathon judge on venue wifi with no
       Groq key configured still sees a working "chatbot."

To enable the LLM mode: `export GROQ_API_KEY=...` (free tier at
console.groq.com) before launching the dashboard.
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _match_channel(question: str, channel_names: list) -> Optional[str]:
    q = question.lower()
    for ch in channel_names:
        if ch.replace("_", " ") in q or ch in q:
            return ch
    aliases = {"fb": "meta", "facebook": "meta", "instagram": "meta", "insta": "meta",
               "search": "google", "sms": "email_sms", "email": "email_sms",
               "tv": "tv_ooh", "ooh": "tv_ooh", "influencers": "influencer"}
    for alias, ch in aliases.items():
        if alias in q and ch in channel_names:
            return ch
    return None


def retrieve_context(question: str, artifact: dict, roi_df: pd.DataFrame,
                      drift_report: dict, optimizer_report: dict, cv_report: dict) -> dict:
    """Rule-based intent routing -> pulls only the structured facts relevant to the question."""
    channel_names = [c.replace("spend_", "") for c in artifact["channel_cols"]]
    matched_channel = _match_channel(question, channel_names)
    q = question.lower()

    context = {"question": question, "matched_channel": matched_channel}

    if matched_channel:
        row = roi_df[roi_df["channel"] == matched_channel]
        if not row.empty:
            context["channel_stats"] = row.iloc[0].to_dict()
        spend_col = f"spend_{matched_channel}"
        context["adstock_lambda"] = artifact["best_params"]["lambdas"].get(spend_col)
        context["saturation_k"] = artifact["best_params"]["k"].get(spend_col)

    if any(w in q for w in ["drift", "chang", "stable", "trust", "swing"]):
        context["drift_report"] = drift_report

    if any(w in q for w in ["recommend", "budget", "allocat", "optimi", "next week", "spend more", "spend less"]):
        context["optimizer_report"] = optimizer_report

    if any(w in q for w in ["accura", "confiden", "how good", "mape", "r2", "r-squared", "error"]):
        context["cv_report"] = cv_report

    if any(w in q for w in ["best", "top", "highest roi", "which channel"]):
        context["top_channels"] = roi_df.sort_values("roi", ascending=False).head(3).to_dict(orient="records")

    if any(w in q for w in ["worst", "lowest", "cut", "underperform"]):
        context["bottom_channels"] = roi_df.sort_values("roi", ascending=True).head(3).to_dict(orient="records")

    return context


def template_answer(context: dict) -> str:
    """Offline fallback: formats retrieved facts directly, no LLM needed."""
    parts = []

    if "channel_stats" in context:
        s = context["channel_stats"]
        parts.append(
            f"**{s['channel'].title()}**: ROI of {s['roi']:.2f}x over the last 4 weeks "
            f"(₹{s['spend']:,.0f} spend → ₹{s['attributed_revenue']:,.0f} attributed revenue). "
            f"Adstock decay λ={context.get('adstock_lambda', 0):.2f} "
            f"({'slow, brand-building carryover' if context.get('adstock_lambda', 0) > 0.4 else 'fast decay, near-immediate response'})."
        )

    if "drift_report" in context:
        d = context["drift_report"]
        if d.get("status") == "insufficient_history":
            parts.append("Not enough retrain history yet to check coefficient drift.")
        else:
            flagged = [f for f in d["flags"] if f["flagged"]]
            if flagged:
                lines = "; ".join(f"{f['channel'].replace('spend_','')} {f['pct_change']:+.1f}%" for f in flagged)
                parts.append(f"⚠ {len(flagged)} channel(s) show coefficient drift beyond the {d['threshold_pct']:.0f}% threshold: {lines}.")
            else:
                parts.append(f"No channels breached the {d['threshold_pct']:.0f}% drift threshold this run — coefficients are stable.")

    if "optimizer_report" in context:
        o = context["optimizer_report"]
        top = sorted(o["optimized_spend"].items(), key=lambda x: -x[1])[:3]
        lines = ", ".join(f"{k.replace('spend_','')}: ₹{v:,.0f}" for k, v in top)
        parts.append(
            f"For a ₹{o['total_budget']:,.0f} weekly budget, the recommended split leads with {lines}. "
            f"Expected lift vs. an even split across channels: {o['expected_lift_vs_even_split_pct']:.1f}%."
        )

    if "cv_report" in context:
        c = context["cv_report"]
        parts.append(
            f"Cross-validated MAPE is {c['cv_metrics']['mean_mape']*100:.1f}% (expanding-window, walk-forward). "
            f"R² is {c['cv_metrics']['mean_r2']:.2f} — modest because total variance is dominated by the long-run "
            f"growth trend, which inflates R²'s denominator; MAPE is the more decision-relevant number here."
        )

    if "top_channels" in context:
        lines = ", ".join(f"{r['channel']} ({r['roi']:.1f}x)" for r in context["top_channels"])
        parts.append(f"Highest-ROI channels (last 4 weeks): {lines}.")

    if "bottom_channels" in context:
        lines = ", ".join(f"{r['channel']} ({r['roi']:.1f}x)" for r in context["bottom_channels"])
        parts.append(f"Lowest-ROI channels (last 4 weeks): {lines}. Worth checking saturation before cutting further.")

    if not parts:
        parts.append(
            "I can answer questions about channel ROI, saturation/diminishing returns, budget recommendations, "
            "coefficient drift, or model accuracy. Try asking e.g. \"is Meta saturated?\" or \"what should I spend on Google next week?\""
        )

    return "\n\n".join(parts)


def call_groq_llm(question: str, context: dict) -> Optional[str]:
    """Optional: phrase the SAME retrieved facts via a free-tier Groq LLM call. Returns None on any failure (falls back to template)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        import requests
        system_prompt = (
            "You are SpendLens Copilot, an MMM assistant for a marketing team. "
            "Answer the user's question using ONLY the JSON context provided below. "
            "Do not invent numbers. Keep the answer to 2-4 sentences, plain language, "
            "no jargon a non-technical marketer wouldn't know without a one-line explanation."
        )
        user_prompt = f"Question: {question}\n\nContext (JSON): {context}"
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def answer_question(question: str, artifact: dict, roi_df: pd.DataFrame,
                     drift_report: dict, optimizer_report: dict, cv_report: dict) -> dict:
    context = retrieve_context(question, artifact, roi_df, drift_report, optimizer_report, cv_report)
    # strip non-JSON-safe objects (DataFrames already converted via to_dict above)
    llm_answer = call_groq_llm(question, {k: v for k, v in context.items() if k not in ("question",)})
    if llm_answer:
        return {"answer": llm_answer, "source": "groq-llama-3.1-8b-instant", "context": context}
    return {"answer": template_answer(context), "source": "offline-template", "context": context}
