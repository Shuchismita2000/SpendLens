document.addEventListener("DOMContentLoaded", () => {
  const D = SPENDLENS_DATA;
  const cv = D.cv_report.cv_metrics;
  const opt = D.optimizer_report;
  const rec = D.recovery_report;

  // ---- hero stats ----
  document.getElementById("stat-mape").textContent = (cv.mean_mape * 100).toFixed(1) + "%";
  document.getElementById("stat-lift").textContent = opt.expected_lift_vs_even_split_pct.toFixed(0) + "%";
  document.getElementById("stat-shape").textContent = rec.saturation_mean_shape_correlation.toFixed(2);
  document.getElementById("stat-channels").textContent = D.model.channel_names.length;

  // ---- hero aperture ----
  const stage = document.getElementById("hero-aperture");
  stage.innerHTML = buildApertureSVG({ size: 420, blades: 9, openness: 0.5 });
  animateAperture(stage);

  // ---- traps section: pull the real "before" numbers where possible ----
  const naiveCorr = computeNaiveCorrelations(D);
  document.getElementById("trap-leakage-fact").textContent =
    `Expanding-window CV, ${D.model.n_weeks_trained_on} weeks, walk-forward only — no fold ever trains on its own future.`;
  document.getElementById("trap-collinearity-fact").textContent =
    `ElasticNet (alpha=${D.model.alpha.toFixed(3)}, l1_ratio=${D.model.l1_ratio.toFixed(2)}) absorbs channel correlation automatically.`;
  document.getElementById("trap-trust-fact").textContent =
    `Coefficient drift threshold at ±25% — ${D.drift_report && D.drift_report.status !== "insufficient_history" ? D.drift_report.n_channels_flagged + " channel(s) flagged in the latest retrain." : "monitored on every retrain."}`;

  // ---- flow diagram: build node list dynamically ----
  const flowTrack = document.getElementById("flow-track");
  const nodes = [
    ["input", "Raw weekly spend"],
    ["transform", "Adstock (geometric decay)"],
    ["transform", "Hill saturation"],
    ["model", "ElasticNet"],
    ["output", "units_sold"],
    ["action", "SLSQP optimizer"],
    ["result", "Next week's budget"],
  ];
  flowTrack.innerHTML = nodes.map((n, i) => {
    const node = `<div class="flow-node"><div class="k">${n[0]}</div><div class="v">${n[1]}</div></div>`;
    const arrow = i < nodes.length - 1 ? `<div class="flow-arrow">&#10230;</div>` : "";
    return node + arrow;
  }).join("");
});

function computeNaiveCorrelations(D) {
  const w = D.weekly;
  const rev = w.map(r => r.revenue);
  const out = {};
  D.model.channel_cols.forEach(c => {
    const spend = w.map(r => r[c]);
    out[c] = pearson(spend, rev);
  });
  return out;
}

function pearson(a, b) {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / Math.sqrt(da * db);
}

function animateAperture(container) {
  const blades = container.querySelectorAll(".ap-blade");
  let t = 0;
  function tick() {
    t += 0.006;
    blades.forEach((b, i) => {
      const phase = t + i * 0.35;
      const op = 0.10 + 0.14 * (0.5 + 0.5 * Math.sin(phase));
      b.setAttribute("opacity", op.toFixed(3));
    });
    container.querySelector("svg").style.transform = `rotate(${(t * 6).toFixed(2)}deg)`;
    requestAnimationFrame(tick);
  }
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    requestAnimationFrame(tick);
  }
}
