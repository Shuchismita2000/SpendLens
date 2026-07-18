// ============================================================
// EDA dashboard — mirrors notebooks/01_eda.ipynb, computed live in-browser
// ============================================================

function edaFmt(n) { return Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 }); }

function pearson(a, b) {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / Math.sqrt(da * db);
}

function renderEDA() {
  renderEdaHeader();
  renderRevenueUnits();
  renderSpendComposition();
  renderCorrelationTable();
  renderDistributions();
  renderPromoStockoutChecks();
}

function renderEdaHeader() {
  document.getElementById("eda-nweeks").textContent = D.weekly.length;
  const first = D.weekly[0].week_start, last = D.weekly[D.weekly.length - 1].week_start;
  document.getElementById("eda-daterange").textContent = `${first} → ${last}`;
  document.getElementById("eda-nchannels").textContent = CHANNEL_COLS_EDA().length;
}
function CHANNEL_COLS_EDA() { return SPENDLENS_DATA.model.channel_cols; }

function renderRevenueUnits() {
  const labels = D.weekly.map(r => r.week_start);
  new Chart(document.getElementById("eda-chart-revenue-units"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Revenue (₹)", data: D.weekly.map(r => r.revenue), borderColor: "#6bbf8a", yAxisID: "y", pointRadius: 0, borderWidth: 2, tension: 0.2 },
        { label: "Units sold", data: D.weekly.map(r => r.units_sold), borderColor: "#5fc9c0", yAxisID: "y1", pointRadius: 0, borderWidth: 2, tension: 0.2 },
      ],
    },
    options: chartBaseOptions({ dualAxis: true }),
  });
}

function renderSpendComposition() {
  const labels = D.weekly.map(r => r.week_start);
  const cols = CHANNEL_COLS_EDA();
  new Chart(document.getElementById("eda-chart-spend-area"), {
    type: "line",
    data: {
      labels,
      datasets: cols.map((c, i) => ({
        label: cap(short(c)), data: D.weekly.map(r => r[c]), borderColor: chanColor(i),
        backgroundColor: chanColor(i) + "33", fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.15,
      })),
    },
    options: { ...chartBaseOptions({}), scales: { ...chartBaseOptions({}).scales, y: { ...chartBaseOptions({}).scales.y, stacked: true } }, elements: { line: { fill: true } } },
  });
}

function renderCorrelationTable() {
  const cols = CHANNEL_COLS_EDA();
  const rev = D.weekly.map(r => r.revenue);
  const units = D.weekly.map(r => r.units_sold);
  const rows = cols.map(c => {
    const spend = D.weekly.map(r => r[c]);
    return { channel: short(c), corrRevenue: pearson(spend, rev), corrUnits: pearson(spend, units) };
  });
  const tbody = document.getElementById("eda-corr-table");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="label-cell">${cap(r.channel)}</td>
      <td>${corrCell(r.corrRevenue)}</td>
      <td>${corrCell(r.corrUnits)}</td>
    </tr>`).join("");
}
function corrCell(v) {
  const color = v > 0.5 ? "#6bbf8a" : v > 0.2 ? "#e0b25c" : "#e0665c";
  const width = Math.min(100, Math.abs(v) * 100);
  return `<div style="display:flex;align-items:center;gap:8px;">
    <span style="min-width:44px;color:${color};">${v.toFixed(2)}</span>
    <div class="diffbar-track" style="max-width:120px;"><div class="diffbar-fill" style="width:${width}%;background:${color};left:0;"></div></div>
  </div>`;
}

function renderDistributions() {
  const container = document.getElementById("eda-distributions");
  const vars = [
    { key: "revenue", label: "Revenue" }, { key: "units_sold", label: "Units sold" },
    { key: "avg_price", label: "Avg. price" }, { key: "discount_rate", label: "Discount rate" },
  ];
  container.innerHTML = vars.map((v, i) => `<div class="chart-card" style="height:230px;"><div class="chart-title">${v.label} distribution</div><canvas id="eda-hist-${i}"></canvas></div>`).join("");
  vars.forEach((v, i) => {
    const values = D.weekly.map(r => r[v.key]);
    const { bins, counts } = histogram(values, 14);
    new Chart(document.getElementById(`eda-hist-${i}`), {
      type: "bar",
      data: { labels: bins.map(b => Math.round(b)), datasets: [{ data: counts, backgroundColor: chanColor(i), borderRadius: 3 }] },
      options: chartBaseOptions({ legend: false }),
    });
  });
}

function histogram(values, nBins) {
  const min = Math.min(...values), max = Math.max(...values);
  const width = (max - min) / nBins || 1;
  const counts = new Array(nBins).fill(0);
  values.forEach(v => { let idx = Math.floor((v - min) / width); if (idx >= nBins) idx = nBins - 1; if (idx < 0) idx = 0; counts[idx]++; });
  const bins = new Array(nBins).fill(0).map((_, i) => min + i * width);
  return { bins, counts };
}

function renderPromoStockoutChecks() {
  const avg = (arr, key) => arr.reduce((s, r) => s + r[key], 0) / arr.length;
  const promo = D.weekly.filter(r => r.promo_flag), nonPromo = D.weekly.filter(r => !r.promo_flag);
  const stockout = D.weekly.filter(r => r.stock_out_flag), normal = D.weekly.filter(r => !r.stock_out_flag);
  const delay = D.weekly.filter(r => r.delivery_delay_flag), noDelay = D.weekly.filter(r => !r.delivery_delay_flag);

  const rows = [
    ["Promo week", avg(promo, "units_sold"), avg(nonPromo, "units_sold"), "units sold"],
    ["Stock-out week", avg(stockout, "units_sold"), avg(normal, "units_sold"), "units sold"],
    ["Delivery-delay week", avg(delay, "returns_pct"), avg(noDelay, "returns_pct"), "returns %"],
  ];
  document.getElementById("eda-checks-table").innerHTML = rows.map(([label, withFlag, without, unit]) => {
    const delta = ((withFlag - without) / without) * 100;
    const color = label.includes("Delivery") || label.includes("Stock") ? (delta > 0 === label.includes("Delivery") ? "#e0665c" : "#e0665c") : (delta > 0 ? "#6bbf8a" : "#e0665c");
    return `<tr><td class="label-cell">${label}</td><td>${edaFmt(withFlag)} ${unit}</td><td>${edaFmt(without)} ${unit}</td><td style="color:${color};">${fmtPct(delta)}</td></tr>`;
  }).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  renderEDA();
  document.getElementById("nav-mini-aperture").innerHTML = buildMiniAperture({ size: 24 });
});
