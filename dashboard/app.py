"""
app.py
======
SpendLens dashboard. Run with:
    streamlit run dashboard/app.py

Six sections (per product spec) + an explainability chatbot:
    1. Overview            - executive view (CMO 10-second read)
    2. Channel Performance  - core MMM output
    3. Diminishing Returns  - saturation curves per channel
    4. Budget Recommendation - the hero feature
    5. Trust & Stability    - drift monitoring
    6. External Factors     - seasonality/promo impact
    7. Ask SpendLens        - NL chatbot over the model's own outputs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_bridge import (  # noqa: E402
    load_everything, per_channel_weekly_contribution, channel_roi_summary, steady_state_response_curve,
)
from dashboard.chatbot import answer_question  # noqa: E402

st.set_page_config(page_title="SpendLens — MMM Copilot", layout="wide", page_icon="📊")


@st.cache_data(ttl=300)
def get_data():
    cfg, artifact, processed, reports, run_history = load_everything()
    contrib = per_channel_weekly_contribution(artifact, processed)
    return cfg, artifact, processed, reports, run_history, contrib


cfg, artifact, processed, reports, run_history, contrib = get_data()
channel_cols = artifact["channel_cols"]
channel_names = [c.replace("spend_", "") for c in channel_cols]

st.title("📊 SpendLens")
st.caption(f"Weekly Marketing Mix Copilot — {cfg['project']['client']}")

tabs = st.tabs([
    "🟢 Overview", "🔵 Channel Performance", "🟡 Diminishing Returns",
    "🔴 Budget Recommendation", "🟠 Trust & Stability", "⚪ External Factors",
    "💬 Ask SpendLens",
])

# ------------------------------------------------------------------
# 1. OVERVIEW
# ------------------------------------------------------------------
with tabs[0]:
    recent = processed.tail(4)
    total_revenue = recent["revenue"].sum()
    total_spend = recent[channel_cols].sum().sum()
    blended_roi = total_revenue / total_spend if total_spend > 0 else np.nan
    orders_trend = recent["orders"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (last 4 wks)", f"₹{total_revenue:,.0f}")
    c2.metric("Total Spend (last 4 wks)", f"₹{total_spend:,.0f}")
    c3.metric("Blended ROI", f"{blended_roi:.2f}x")
    c4.metric("Orders (last 4 wks)", f"{orders_trend:,.0f}")

    st.subheader("Revenue vs. Spend")
    plot_df = processed[["week_start", "revenue"]].copy()
    plot_df["total_spend"] = processed[channel_cols].sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["week_start"], y=plot_df["revenue"], name="Revenue", line=dict(color="#2E7D32")))
    fig.add_trace(go.Scatter(x=plot_df["week_start"], y=plot_df["total_spend"], name="Total Spend",
                              line=dict(color="#1565C0"), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Revenue (₹)"), yaxis2=dict(title="Spend (₹)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1), height=380, margin=dict(t=30),
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Orders Trend")
    fig2 = px.line(processed, x="week_start", y="orders", markers=False)
    fig2.update_traces(line=dict(color="#EF6C00"))
    fig2.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig2, width='stretch')

# ------------------------------------------------------------------
# 2. CHANNEL PERFORMANCE
# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Contribution by Channel")
    roi_df = channel_roi_summary(contrib, channel_cols, last_n_weeks=4)
    roi_df["contribution_pct"] = roi_df["attributed_revenue"] / roi_df["attributed_revenue"].sum() * 100

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.bar(roi_df, x="channel", y="contribution_pct", text_auto=".1f",
                       labels={"contribution_pct": "% of attributed revenue"}, color="channel")
        fig3.update_layout(height=380, showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig3, width='stretch')
    with col2:
        fig4 = px.scatter(roi_df, x="spend", y="roi", size="attributed_revenue", color="channel",
                           text="channel", labels={"spend": "Spend (₹, last 4 wks)", "roi": "ROI (x)"})
        fig4.update_traces(textposition="top center")
        fig4.update_layout(height=380, margin=dict(t=30))
        st.plotly_chart(fig4, width='stretch')

    st.dataframe(
        roi_df.rename(columns={"channel": "Channel", "spend": "Spend (₹)", "attributed_revenue": "Attributed Revenue (₹)",
                                "roi": "ROI", "contribution_pct": "Contribution %"})
        .style.format({"Spend (₹)": "₹{:,.0f}", "Attributed Revenue (₹)": "₹{:,.0f}", "ROI": "{:.2f}x", "Contribution %": "{:.1f}%"}),
        width='stretch', hide_index=True,
    )

# ------------------------------------------------------------------
# 3. DIMINISHING RETURNS
# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("Spend → Return Curves (steady-state)")
    st.caption("If a channel spent this amount every week indefinitely, this is the model's expected weekly contribution.")

    selected = st.selectbox("Channel", channel_names, key="dr_channel")
    spend_col = f"spend_{selected}"
    current_spend = processed[spend_col].iloc[-1]
    max_grid = max(current_spend * 2.5, artifact["best_params"]["k"][spend_col] * 2)
    grid = np.linspace(0, max_grid, 200)
    response = steady_state_response_curve(artifact, spend_col, grid)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=grid, y=response, mode="lines", name="Response curve", line=dict(color="#F9A825", width=3)))
    fig5.add_vline(x=current_spend, line_dash="dash", line_color="gray", annotation_text="Current spend")
    k_val = artifact["best_params"]["k"][spend_col]
    fig5.add_vline(x=k_val, line_dash="dot", line_color="red", annotation_text="Half-saturation point")
    fig5.update_layout(xaxis_title="Weekly spend (₹)", yaxis_title="Predicted incremental units/week",
                        height=420, margin=dict(t=30))
    st.plotly_chart(fig5, width='stretch')

    pct_of_k = current_spend / k_val * 100 if k_val > 0 else 0
    if pct_of_k > 130:
        st.warning(f"**{selected.title()} is saturated** — current spend (₹{current_spend:,.0f}) is {pct_of_k:.0f}% of its half-saturation point (₹{k_val:,.0f}). Additional spend here is hitting steep diminishing returns.")
    elif pct_of_k > 70:
        st.info(f"**{selected.title()} is approaching saturation** — currently at {pct_of_k:.0f}% of its half-saturation point (₹{k_val:,.0f}).")
    else:
        st.success(f"**{selected.title()} has room to grow** — currently at only {pct_of_k:.0f}% of its half-saturation point (₹{k_val:,.0f}).")

    st.caption(
        "Note: adstock decay is harder to identify than saturation shape with ~130 weeks of data "
        "(see model validation report) — treat the exact half-saturation ₹ figure as directional, "
        "especially for slow-decay channels like TV/OOH."
    )

# ------------------------------------------------------------------
# 4. BUDGET RECOMMENDATION
# ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Next Week's Recommended Allocation")
    opt = reports["optimizer"]
    if opt is None:
        st.warning("No optimizer report found — run `python src/optimization/budget_optimizer.py`.")
    else:
        budget_input = st.number_input("Total weekly budget (₹)", value=float(opt["total_budget"]), step=10000.0)
        if budget_input != opt["total_budget"]:
            import joblib
            from src.optimization.budget_optimizer import optimize_budget
            opt = optimize_budget(artifact, cfg, total_budget=budget_input)

        rows = []
        for c in channel_cols:
            short = c.replace("spend_", "")
            current = processed[c].iloc[-1]
            rec = opt["optimized_spend"][c]
            rows.append({
                "Channel": short, "Current Spend (₹)": current, "Recommended (₹)": rec,
                "Change (₹)": rec - current, "Change (%)": (rec - current) / current * 100 if current > 0 else np.nan,
            })
        rec_df = pd.DataFrame(rows)
        st.dataframe(
            rec_df.style.format({"Current Spend (₹)": "₹{:,.0f}", "Recommended (₹)": "₹{:,.0f}",
                                  "Change (₹)": "{:+,.0f}", "Change (%)": "{:+.1f}%"})
            .background_gradient(subset=["Change (%)"], cmap="RdYlGn", vmin=-50, vmax=50),
            width='stretch', hide_index=True,
        )

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=rec_df["Channel"], y=rec_df["Current Spend (₹)"], name="Current"))
        fig6.add_trace(go.Bar(x=rec_df["Channel"], y=rec_df["Recommended (₹)"], name="Recommended"))
        fig6.update_layout(barmode="group", height=380, margin=dict(t=30))
        st.plotly_chart(fig6, width='stretch')

        lift = opt["expected_lift_vs_even_split_pct"]
        st.metric("Expected lift vs. naive even-split baseline", f"{lift:+.1f}%")
        st.caption(
            "Optimization respects per-channel min/max spend bounds (configs/model_config.yaml) and solves "
            "a constrained concave-maximization problem via SLSQP — the same saturation curves shown in the "
            "Diminishing Returns tab are what make an even split suboptimal."
        )

# ------------------------------------------------------------------
# 5. TRUST & STABILITY
# ------------------------------------------------------------------
with tabs[4]:
    st.subheader("Coefficient Stability Over Retrains")
    if len(run_history) < 2:
        st.info("Need at least 2 retrain runs to show a stability trend. Run `python scripts/weekly_retrain.py` again.")
    else:
        hist_df = pd.DataFrame([
            {"run": i, "trained_at": r["trained_at"], **{ch.replace("spend_", ""): r["coefficients"].get(ch) for ch in channel_cols}}
            for i, r in enumerate(run_history)
        ])
        melted = hist_df.melt(id_vars=["run", "trained_at"], var_name="channel", value_name="coefficient")
        fig7 = px.line(melted, x="run", y="coefficient", color="channel", markers=True)
        fig7.update_layout(height=380, margin=dict(t=30), xaxis_title="Retrain run #")
        st.plotly_chart(fig7, width='stretch')

    st.subheader("Drift Alerts (latest retrain)")
    drift = reports["drift"]
    if drift is None or drift.get("status") == "insufficient_history":
        st.info("No drift comparison available yet.")
    else:
        any_flag = False
        for f in drift["flags"]:
            if f["flagged"]:
                any_flag = True
                icon = "🚨" if f["flag_level"] == "red" else "⚠️"
                st.warning(f"{icon} **{f['channel'].replace('spend_', '').title()}** ROI/coefficient moved "
                           f"{f['pct_change']:+.1f}% week-over-week (₹{f['prev_coefficient']:.1f} → ₹{f['curr_coefficient']:.1f}). Verify before acting on this channel's recommendation.")
        if not any_flag:
            st.success(f"✅ No channels breached the ±{drift['threshold_pct']:.0f}% drift threshold. Coefficients are stable — safe to trust this week's recommendation.")

    st.subheader("Model Validation")
    rec = reports["recovery"]
    if rec:
        c1, c2, c3 = st.columns(3)
        c1.metric("CV MAPE", f"{rec['cv_metrics']['mean_mape']*100:.1f}%")
        c2.metric("CV R²", f"{rec['cv_metrics']['mean_r2']:.2f}")
        c3.metric("Saturation shape match (vs. ground truth)", f"{rec['saturation_mean_shape_correlation']:.2f}")

# ------------------------------------------------------------------
# 6. EXTERNAL FACTORS
# ------------------------------------------------------------------
with tabs[5]:
    st.subheader("Seasonality Impact")
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=processed["week_start"], y=processed["category_search_index"],
                               name="Category demand index", line=dict(color="#6A1B9A")))
    fig8.add_trace(go.Bar(x=processed["week_start"], y=processed["festive_flag"] * processed["revenue"].max() * 0.15,
                           name="Festive week", marker_color="rgba(255,99,71,0.4)", yaxis="y"))
    fig8.update_layout(height=380, margin=dict(t=30), yaxis_title="Category demand index / festive marker")
    st.plotly_chart(fig8, width='stretch')

    st.subheader("Promotions Impact")
    promo_weeks = processed[processed["promo_flag"] == 1]
    non_promo_weeks = processed[processed["promo_flag"] == 0]
    c1, c2 = st.columns(2)
    c1.metric("Avg. weekly units — promo weeks", f"{promo_weeks['units_sold'].mean():,.0f}")
    c2.metric("Avg. weekly units — non-promo weeks", f"{non_promo_weeks['units_sold'].mean():,.0f}")

    fig9 = px.box(processed, x="promo_flag", y="units_sold", points="all",
                  labels={"promo_flag": "Promo week (0=No, 1=Yes)", "units_sold": "Units sold"})
    fig9.update_layout(height=350, margin=dict(t=30))
    st.plotly_chart(fig9, width='stretch')

    st.caption(
        "Promotions and discounts are modeled as explicit controls (discount_rate, promo_flag) rather than "
        "left for media coefficients to silently absorb — see the data generation notes for why that separation matters."
    )

# ------------------------------------------------------------------
# 7. ASK SPENDLENS (CHATBOT)
# ------------------------------------------------------------------
with tabs[6]:
    st.subheader("💬 Ask SpendLens")
    st.caption(
        "Answers are generated by RETRIEVING the exact numbers shown elsewhere in this dashboard — the model "
        "never invents figures. Set `GROQ_API_KEY` (free tier) for natural-language phrasing; without it, "
        "answers use a plain-language template so the demo works fully offline."
    )

    roi_df_full = channel_roi_summary(contrib, channel_cols, last_n_weeks=4)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    example_qs = ["Is Meta saturated?", "What should I spend on Google next week?",
                  "Which channels are stable?", "How accurate is the model?"]
    cols = st.columns(len(example_qs))
    for i, ex in enumerate(example_qs):
        if cols[i].button(ex, width='stretch'):
            st.session_state.pending_q = ex

    user_q = st.chat_input("Ask about channel performance, saturation, budget, or drift...")
    q_to_process = user_q or st.session_state.pop("pending_q", None)

    if q_to_process:
        result = answer_question(
            q_to_process, artifact, roi_df_full,
            reports["drift"] or {"status": "insufficient_history"},
            reports["optimizer"] or {}, reports["cv"] or {"cv_metrics": {"mean_mape": 0, "mean_r2": 0}},
        )
        st.session_state.chat_history.append(("user", q_to_process))
        st.session_state.chat_history.append(("assistant", result["answer"], result["source"]))

    for turn in st.session_state.chat_history:
        if turn[0] == "user":
            st.chat_message("user").write(turn[1])
        else:
            with st.chat_message("assistant"):
                st.write(turn[1])
                st.caption(f"source: {turn[2]}")
