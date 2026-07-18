const D = SPENDLENS_DATA;
const CHANNEL_COLS = D.model.channel_cols;
const CHANNEL_NAMES = D.model.channel_names;
const PALETTE = ["#5fc9c0", "#c9a15b", "#e0665c", "#8ea6c9", "#a888d1", "#e0b25c"];
const chanColor = (i) => PALETTE[i % PALETTE.length];

function fmtINR(n, decimals = 0) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}
function fmtPct(n, decimals = 1) { return (n >= 0 ? "+" : "") + n.toFixed(decimals) + "%"; }
function short(c) { return c.replace("spend_", ""); }

// ============================================================
// TABS
// ============================================================
function initTabs() {
  const btns = document.querySelectorAll(".tab-btn");
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.target).classList.add("active");
      history.replaceState(null, "", "#" + btn.dataset.target);
    });
  });
  const hash = location.hash.replace("#", "");
  if (hash) { const b = document.querySelector(`.tab-btn[data-target="${hash}"]`); if (b) b.click(); }
}

// ============================================================
// CLIENT-SIDE MODEL FUNCTIONS (mirrors src/optimization/budget_optimizer.py)
// ============================================================
function steadyStateContribution(channelCol, spend) {
  const lam = D.model.lambdas[channelCol];
  const k = D.model.k[channelCol];
  const s = D.model.s[channelCol];
  const coef = D.model.raw_coefs[channelCol];
  const steadyAdstock = spend / Math.max(1e-6, 1 - lam);
  const sat = Math.pow(steadyAdstock, s) / (Math.pow(steadyAdstock, s) + Math.pow(k, s));
  return coef * sat;
}

function marginalReturn(channelCol, spend, eps = 500) {
  return (steadyStateContribution(channelCol, spend + eps) - steadyStateContribution(channelCol, spend)) / eps;
}

// Greedy marginal-utility water-filling: solves the same separable concave
// budget-allocation problem as scipy's SLSQP in the Python pipeline, without
// needing a backend. Optimal in the continuous limit for separable concave
// objectives with one linear budget constraint.
function optimizeBudget(totalBudget) {
  const bounds = D.channel_bounds;
  let spend = {};
  CHANNEL_COLS.forEach(c => spend[c] = bounds[c].min);
  let allocated = CHANNEL_COLS.reduce((s, c) => s + spend[c], 0);
  let remaining = totalBudget - allocated;
  if (remaining < 0) {
    // budget too small to cover minimums -- scale mins down proportionally
    const scale = totalBudget / allocated;
    CHANNEL_COLS.forEach(c => spend[c] = spend[c] * scale);
    return spend;
  }
  const step = Math.max(500, totalBudget / 4000);
  let guard = 0;
  while (remaining > step && guard < 20000) {
    guard++;
    let best = null, bestMR = -Infinity;
    for (const c of CHANNEL_COLS) {
      if (spend[c] + step > bounds[c].max) continue;
      const mr = marginalReturn(c, spend[c]);
      if (mr > bestMR) { bestMR = mr; best = c; }
    }
    if (best === null) break;
    spend[best] += step;
    remaining -= step;
  }
  return spend;
}

// ============================================================
// 1. OVERVIEW
// ============================================================
function renderOverview() {
  const recent = D.weekly.slice(-4);
  const totalRevenue = recent.reduce((s, r) => s + r.revenue, 0);
  const totalSpend = recent.reduce((s, r) => s + CHANNEL_COLS.reduce((ss, c) => ss + r[c], 0), 0);
  const blendedROI = totalRevenue / totalSpend;
  const totalOrders = recent.reduce((s, r) => s + r.orders, 0);

  document.getElementById("ov-revenue").textContent = fmtINR(totalRevenue);
  document.getElementById("ov-spend").textContent = fmtINR(totalSpend);
  document.getElementById("ov-roi").textContent = blendedROI.toFixed(2) + "x";
  document.getElementById("ov-orders").textContent = totalOrders.toLocaleString("en-IN");

  const labels = D.weekly.map(r => r.week_start);
  const revSeries = D.weekly.map(r => r.revenue);
  const spendSeries = D.weekly.map(r => CHANNEL_COLS.reduce((s, c) => s + r[c], 0));

  new Chart(document.getElementById("chart-revenue-spend"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Revenue", data: revSeries, borderColor: "#6bbf8a", backgroundColor: "transparent", yAxisID: "y", tension: 0.25, pointRadius: 0, borderWidth: 2 },
        { label: "Total Spend", data: spendSeries, borderColor: "#5fc9c0", backgroundColor: "transparent", yAxisID: "y1", tension: 0.25, pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: chartBaseOptions({ dualAxis: true }),
  });

  new Chart(document.getElementById("chart-orders"), {
    type: "line",
    data: { labels, datasets: [{ label: "Orders", data: D.weekly.map(r => r.orders), borderColor: "#c9a15b", backgroundColor: "rgba(201,161,91,0.08)", fill: true, tension: 0.25, pointRadius: 0, borderWidth: 2 }] },
    options: chartBaseOptions({}),
  });
}

// ============================================================
// 2. CHANNEL PERFORMANCE
// ============================================================
function renderChannelPerformance() {
  const roi = [...D.roi_4wk].sort((a, b) => b.attributed_revenue - a.attributed_revenue);
  const totalAttrib = roi.reduce((s, r) => s + r.attributed_revenue, 0);

  new Chart(document.getElementById("chart-contribution"), {
    type: "bar",
    data: {
      labels: roi.map(r => r.channel),
      datasets: [{ data: roi.map(r => (r.attributed_revenue / totalAttrib * 100)), backgroundColor: roi.map((_, i) => chanColor(i)), borderRadius: 6 }],
    },
    options: chartBaseOptions({ legend: false, yLabel: "% of attributed revenue" }),
  });

  new Chart(document.getElementById("chart-spend-roi"), {
    type: "bubble",
    data: {
      datasets: roi.map((r, i) => ({
        label: r.channel,
        data: [{ x: r.spend, y: r.roi, r: Math.max(6, Math.sqrt(r.attributed_revenue) / 40) }],
        backgroundColor: chanColor(i) + "cc",
      })),
    },
    options: chartBaseOptions({ xLabel: "Spend (₹, last 4 wks)", yLabel: "ROI (x)" }),
  });

  const tbody = document.getElementById("channel-perf-table");
  tbody.innerHTML = roi.map((r, i) => `
    <tr>
      <td class="label-cell"><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${chanColor(i)};margin-right:8px;"></span>${cap(r.channel)}</td>
      <td>${fmtINR(r.spend)}</td>
      <td>${fmtINR(r.attributed_revenue)}</td>
      <td>${r.roi.toFixed(2)}x</td>
      <td>${(r.attributed_revenue / totalAttrib * 100).toFixed(1)}%</td>
    </tr>`).join("");
}

// ============================================================
// 3. DIMINISHING RETURNS
// ============================================================
let drChart = null;
function renderDiminishingReturns() {
  const sel = document.getElementById("dr-channel-select");
  sel.innerHTML = CHANNEL_COLS.map(c => `<option value="${c}">${cap(short(c))}</option>`).join("");
  sel.addEventListener("change", () => drawResponseCurve(sel.value));
  drawResponseCurve(CHANNEL_COLS[0]);
}

function drawResponseCurve(channelCol) {
  const curve = D.response_curves[channelCol];
  const k = D.model.k[channelCol];
  const currentSpend = curve.current_spend;

  if (drChart) drChart.destroy();
  drChart = new Chart(document.getElementById("chart-diminishing"), {
    type: "line",
    data: {
      labels: curve.x.map(v => Math.round(v)),
      datasets: [{ label: "Predicted incremental units/week", data: curve.y, borderColor: "#c9a15b", backgroundColor: "rgba(201,161,91,0.08)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2.5 }],
    },
    options: chartBaseOptions({ legend: false, xLabel: "Weekly spend (₹)", yLabel: "Predicted units/week" }),
  });

  const pctOfK = (currentSpend / k) * 100;
  const box = document.getElementById("dr-insight");
  let cls, text;
  if (pctOfK > 130) {
    cls = "alert-red";
    text = `<strong>${cap(short(channelCol))} is saturated</strong> — current spend (${fmtINR(currentSpend)}) is ${pctOfK.toFixed(0)}% of its half-saturation point (${fmtINR(k)}). Additional spend here hits steep diminishing returns.`;
  } else if (pctOfK > 70) {
    cls = "alert-amber";
    text = `<strong>${cap(short(channelCol))} is approaching saturation</strong> — currently at ${pctOfK.toFixed(0)}% of its half-saturation point (${fmtINR(k)}).`;
  } else {
    cls = "alert-green";
    text = `<strong>${cap(short(channelCol))} has room to grow</strong> — currently at only ${pctOfK.toFixed(0)}% of its half-saturation point (${fmtINR(k)}).`;
  }
  box.className = "alert " + cls + " insight-box";
  box.innerHTML = text;
}

// ============================================================
// 4. BUDGET RECOMMENDATION
// ============================================================
let budgetChart = null;
function renderBudgetRecommendation() {
  const input = document.getElementById("budget-input");
  input.value = D.optimizer_report.total_budget;
  input.addEventListener("input", debounce(() => drawBudgetTable(Number(input.value)), 250));
  drawBudgetTable(Number(input.value));
}

function drawBudgetTable(totalBudget) {
  const optimized = optimizeBudget(totalBudget);
  const evenSplit = totalBudget / CHANNEL_COLS.length;

  let totalOptContrib = 0, totalEvenContrib = 0;
  const rows = CHANNEL_COLS.map(c => {
    const current = D.weekly[D.weekly.length - 1][c];
    const rec = optimized[c];
    const change = rec - current;
    const changePct = current > 0 ? (change / current) * 100 : 0;
    totalOptContrib += steadyStateContribution(c, rec);
    totalEvenContrib += steadyStateContribution(c, Math.max(D.channel_bounds[c].min, Math.min(D.channel_bounds[c].max, evenSplit)));
    return { channel: short(c), current, rec, change, changePct };
  });

  const tbody = document.getElementById("budget-table");
  tbody.innerHTML = rows.map(r => {
    const barColor = r.changePct >= 0 ? "#6bbf8a" : "#e0665c";
    const barWidth = Math.min(100, Math.abs(r.changePct));
    return `<tr>
      <td class="label-cell">${cap(r.channel)}</td>
      <td>${fmtINR(r.current)}</td>
      <td>${fmtINR(r.rec)}</td>
      <td>
        <div class="diffbar-wrap">
          <span style="color:${barColor};min-width:70px;display:inline-block;">${fmtPct(r.changePct)}</span>
          <div class="diffbar-track"><div class="diffbar-fill" style="width:${barWidth}%;background:${barColor};${r.changePct >= 0 ? 'left:50%;' : 'right:50%;'}"></div></div>
        </div>
      </td>
    </tr>`;
  }).join("");

  if (budgetChart) budgetChart.destroy();
  budgetChart = new Chart(document.getElementById("chart-budget"), {
    type: "bar",
    data: {
      labels: rows.map(r => cap(r.channel)),
      datasets: [
        { label: "Current", data: rows.map(r => r.current), backgroundColor: "#8ea6c9" },
        { label: "Recommended", data: rows.map(r => r.rec), backgroundColor: "#5fc9c0" },
      ],
    },
    options: chartBaseOptions({}),
  });

  const lift = ((totalOptContrib - totalEvenContrib) / Math.abs(totalEvenContrib)) * 100;
  document.getElementById("budget-lift").textContent = fmtPct(lift);
}

// ============================================================
// 5. TRUST & STABILITY
// ============================================================
function renderTrustStability() {
  const history = D.run_history;
  if (history.length >= 2) {
    const labels = history.map((_, i) => "Run " + (i + 1));
    new Chart(document.getElementById("chart-stability"), {
      type: "line",
      data: {
        labels,
        datasets: CHANNEL_COLS.map((c, i) => ({
          label: cap(short(c)),
          data: history.map(r => r.coefficients[c]),
          borderColor: chanColor(i), backgroundColor: "transparent", tension: 0.2, borderWidth: 2, pointRadius: 4,
        })),
      },
      options: chartBaseOptions({}),
    });
  } else {
    document.getElementById("stability-empty").style.display = "block";
    document.getElementById("chart-stability").style.display = "none";
  }

  const drift = D.drift_report;
  const alertsBox = document.getElementById("drift-alerts");
  if (!drift || drift.status === "insufficient_history") {
    alertsBox.innerHTML = `<div class="alert alert-amber">No drift comparison available yet — need at least 2 retrain runs.</div>`;
  } else {
    const flagged = drift.flags.filter(f => f.flagged);
    if (flagged.length === 0) {
      alertsBox.innerHTML = `<div class="alert alert-green">✅ No channels breached the ±${drift.threshold_pct.toFixed(0)}% drift threshold. Coefficients are stable — safe to trust this week's recommendation.</div>`;
    } else {
      alertsBox.innerHTML = flagged.map(f => {
        const cls = f.flag_level === "red" ? "alert-red" : "alert-amber";
        const icon = f.flag_level === "red" ? "🚨" : "⚠️";
        return `<div class="alert ${cls}">${icon} <strong>${cap(short(f.channel))}</strong> coefficient moved ${fmtPct(f.pct_change)} week-over-week (${f.prev_coefficient.toFixed(1)} → ${f.curr_coefficient.toFixed(1)}). Verify before acting on this channel's recommendation.</div>`;
      }).join("");
    }
  }

  const rec = D.recovery_report;
  if (rec) {
    document.getElementById("cv-mape").textContent = (rec.cv_metrics.mean_mape * 100).toFixed(1) + "%";
    document.getElementById("cv-r2").textContent = rec.cv_metrics.mean_r2.toFixed(2);
    document.getElementById("cv-shape").textContent = rec.saturation_mean_shape_correlation.toFixed(2);
  }
}

// ============================================================
// 6. EXTERNAL FACTORS
// ============================================================
function renderExternalFactors() {
  const labels = D.weekly.map(r => r.week_start);
  new Chart(document.getElementById("chart-seasonality"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Category demand index", data: D.weekly.map(r => r.category_search_index), borderColor: "#a888d1", backgroundColor: "transparent", tension: 0.25, pointRadius: 0, borderWidth: 2 },
        { label: "Festive week", data: D.weekly.map(r => r.festive_flag ? 100 : null), type: "bar", backgroundColor: "rgba(224,102,92,0.35)", yAxisID: "y" },
      ],
    },
    options: chartBaseOptions({}),
  });

  const promo = D.weekly.filter(r => r.promo_flag);
  const nonPromo = D.weekly.filter(r => !r.promo_flag);
  const avg = (arr, key) => arr.reduce((s, r) => s + r[key], 0) / arr.length;
  document.getElementById("promo-avg").textContent = avg(promo, "units_sold").toLocaleString("en-IN", { maximumFractionDigits: 0 });
  document.getElementById("nonpromo-avg").textContent = avg(nonPromo, "units_sold").toLocaleString("en-IN", { maximumFractionDigits: 0 });

  new Chart(document.getElementById("chart-promo-box"), {
    type: "bar",
    data: {
      labels: ["Non-promo weeks", "Promo weeks"],
      datasets: [{ data: [avg(nonPromo, "units_sold"), avg(promo, "units_sold")], backgroundColor: ["#8ea6c9", "#c9a15b"], borderRadius: 6 }],
    },
    options: chartBaseOptions({ legend: false, yLabel: "Avg. weekly units sold" }),
  });
}

// ============================================================
// CHART DEFAULTS
// ============================================================
function chartBaseOptions({ dualAxis = false, legend = true, xLabel = null, yLabel = null } = {}) {
  const gridColor = "rgba(255,255,255,0.05)";
  const textColor = "#7c8896";
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: legend, labels: { color: "#b7c0cb", font: { family: "Inter", size: 11 }, boxWidth: 10, usePointStyle: true } },
      tooltip: { backgroundColor: "#121822", borderColor: "#2a3341", borderWidth: 1, titleColor: "#eef1f4", bodyColor: "#b7c0cb", padding: 10 },
    },
    scales: {
      x: { ticks: { color: textColor, font: { family: "JetBrains Mono", size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { color: gridColor }, title: xLabel ? { display: true, text: xLabel, color: textColor, font: { family: "JetBrains Mono", size: 10 } } : undefined },
      y: { ticks: { color: textColor, font: { family: "JetBrains Mono", size: 10 } }, grid: { color: gridColor }, title: yLabel ? { display: true, text: yLabel, color: textColor, font: { family: "JetBrains Mono", size: 10 } } : undefined },
    },
  };
  if (dualAxis) {
    opts.scales.y1 = { position: "right", ticks: { color: textColor, font: { family: "JetBrains Mono", size: 10 } }, grid: { display: false } };
  }
  return opts;
}

function cap(s) { return s.split("_").map(w => w[0].toUpperCase() + w.slice(1)).join(" "); }
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ============================================================
// INIT
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("nav-mini-aperture").innerHTML = buildMiniAperture({ size: 24 });
  if (!document.getElementById("tab-overview")) return; // not the console page — only shared helpers needed
  initTabs();
  renderOverview();
  renderChannelPerformance();
  renderDiminishingReturns();
  renderBudgetRecommendation();
  renderTrustStability();
  renderExternalFactors();
  initChatbot();
});
