// ============================================================
// Ask SpendLens — retrieval-grounded chatbot
// Mirrors src/dashboard/chatbot.py: retrieve structured facts first,
// then either phrase them via a free-tier Groq LLM call (if the person
// supplies their own API key, stored only in this browser's localStorage)
// or fall back to a plain-language template. The LLM is never asked to
// compute numbers, only to read back what it's given — keeps answers
// auditable instead of hallucination-prone.
// ============================================================

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = "llama-3.1-8b-instant";

function matchChannel(question) {
  const q = question.toLowerCase();
  const aliases = { fb: "meta", facebook: "meta", instagram: "meta", insta: "meta", search: "google", sms: "email_sms", email: "email_sms", tv: "tv_ooh", ooh: "tv_ooh", influencers: "influencer" };
  for (const name of CHANNEL_NAMES) { if (q.includes(name.replace("_", " ")) || q.includes(name)) return name; }
  for (const [alias, name] of Object.entries(aliases)) { if (q.includes(alias) && CHANNEL_NAMES.includes(name)) return name; }
  return null;
}

function retrieveContext(question) {
  const q = question.toLowerCase();
  const ctx = { question };
  const matched = matchChannel(question);

  if (matched) {
    const spendCol = "spend_" + matched;
    const roiRow = D.roi_4wk.find(r => r.channel === matched);
    ctx.channel_stats = roiRow || null;
    ctx.adstock_lambda = D.model.lambdas[spendCol];
    ctx.saturation_k = D.model.k[spendCol];
    const curve = D.response_curves[spendCol];
    ctx.pct_of_k = curve ? (curve.current_spend / D.model.k[spendCol]) * 100 : null;
  }
  if (/drift|chang|stable|trust|swing/.test(q)) ctx.drift_report = D.drift_report;
  if (/recommend|budget|allocat|optimi|next week|spend more|spend less/.test(q)) ctx.optimizer_report = D.optimizer_report;
  if (/accura|confiden|how good|mape|r2|r-squared|error/.test(q)) ctx.cv_report = D.cv_report;
  if (/best|top|highest roi|which channel/.test(q)) ctx.top_channels = [...D.roi_4wk].sort((a, b) => b.roi - a.roi).slice(0, 3);
  if (/worst|lowest|cut|underperform/.test(q)) ctx.bottom_channels = [...D.roi_4wk].sort((a, b) => a.roi - b.roi).slice(0, 3);
  if (/satura/.test(q) && matched) ctx.saturation_question = true;

  return ctx;
}

function templateAnswer(ctx) {
  const parts = [];

  if (ctx.channel_stats) {
    const s = ctx.channel_stats;
    const decayNote = ctx.adstock_lambda > 0.4 ? "slow, brand-building carryover" : "fast decay, near-immediate response";
    parts.push(`<strong>${cap(s.channel)}</strong>: ROI of ${s.roi.toFixed(2)}x over the last 4 weeks (${fmtINR(s.spend)} spend → ${fmtINR(s.attributed_revenue)} attributed revenue). Adstock decay λ=${ctx.adstock_lambda.toFixed(2)} (${decayNote}).`);
  }
  if (ctx.saturation_question && ctx.pct_of_k !== null) {
    if (ctx.pct_of_k > 130) parts.push(`It's <strong>saturated</strong> — current spend is ${ctx.pct_of_k.toFixed(0)}% of its half-saturation point (${fmtINR(ctx.saturation_k)}). More spend here hits steep diminishing returns.`);
    else if (ctx.pct_of_k > 70) parts.push(`It's <strong>approaching saturation</strong> — at ${ctx.pct_of_k.toFixed(0)}% of its half-saturation point (${fmtINR(ctx.saturation_k)}).`);
    else parts.push(`It <strong>has room to grow</strong> — only at ${ctx.pct_of_k.toFixed(0)}% of its half-saturation point (${fmtINR(ctx.saturation_k)}).`);
  }
  if (ctx.drift_report) {
    const d = ctx.drift_report;
    if (!d || d.status === "insufficient_history") parts.push("Not enough retrain history yet to check coefficient drift.");
    else {
      const flagged = d.flags.filter(f => f.flagged);
      if (flagged.length) parts.push(`⚠ ${flagged.length} channel(s) show drift beyond the ${d.threshold_pct.toFixed(0)}% threshold: ` + flagged.map(f => `${cap(short(f.channel))} ${fmtPct(f.pct_change)}`).join(", ") + ".");
      else parts.push(`No channels breached the ${d.threshold_pct.toFixed(0)}% drift threshold this run — coefficients are stable.`);
    }
  }
  if (ctx.optimizer_report && ctx.optimizer_report.optimized_spend) {
    const o = ctx.optimizer_report;
    const top = Object.entries(o.optimized_spend).sort((a, b) => b[1] - a[1]).slice(0, 3);
    parts.push(`For a ${fmtINR(o.total_budget)} weekly budget, the recommended split leads with ` + top.map(([k, v]) => `${cap(short(k))}: ${fmtINR(v)}`).join(", ") + `. Expected lift vs. an even split: ${fmtPct(o.expected_lift_vs_even_split_pct)}.`);
  }
  if (ctx.cv_report && ctx.cv_report.cv_metrics) {
    const c = ctx.cv_report.cv_metrics;
    parts.push(`Cross-validated MAPE is ${(c.mean_mape * 100).toFixed(1)}% (expanding-window, walk-forward). R² is ${c.mean_r2.toFixed(2)} — modest because total variance is dominated by the long-run growth trend; MAPE is the more decision-relevant number here.`);
  }
  if (ctx.top_channels) parts.push(`Highest-ROI channels (last 4 weeks): ` + ctx.top_channels.map(r => `${cap(r.channel)} (${r.roi.toFixed(1)}x)`).join(", ") + ".");
  if (ctx.bottom_channels) parts.push(`Lowest-ROI channels (last 4 weeks): ` + ctx.bottom_channels.map(r => `${cap(r.channel)} (${r.roi.toFixed(1)}x)`).join(", ") + ". Worth checking saturation before cutting further.");

  if (!parts.length) parts.push(`I can answer questions about channel ROI, saturation/diminishing returns, budget recommendations, coefficient drift, or model accuracy. Try "is Meta saturated?" or "what should I spend on Google next week?"`);

  return parts.join("<br><br>");
}

async function callGroq(question, ctx) {
  const key = localStorage.getItem("spendlens_groq_key");
  if (!key) return null;
  try {
    const resp = await fetch(GROQ_URL, {
      method: "POST",
      headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: "system", content: "You are SpendLens Copilot, an MMM assistant for a marketing team. Answer using ONLY the JSON context provided. Never invent numbers. 2-4 sentences, plain language." },
          { role: "user", content: `Question: ${question}\n\nContext (JSON): ${JSON.stringify(ctx)}` },
        ],
        temperature: 0.2, max_tokens: 300,
      }),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    return data.choices[0].message.content.trim();
  } catch (e) { return null; }
}

async function answerQuestion(question) {
  const ctx = retrieveContext(question);
  const llmAnswer = await callGroq(question, ctx);
  if (llmAnswer) return { answer: llmAnswer, source: "groq · llama-3.1-8b-instant" };
  return { answer: templateAnswer(ctx), source: "offline template" };
}

function initChatbot() {
  const log = document.getElementById("chat-log");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const keyInput = document.getElementById("groq-key-input");
  const keyStatus = document.getElementById("groq-key-status");

  const savedKey = localStorage.getItem("spendlens_groq_key");
  if (savedKey) { keyInput.value = savedKey; keyStatus.textContent = "LLM phrasing active"; keyStatus.className = "pill pill-green"; }

  keyInput.addEventListener("change", () => {
    if (keyInput.value.trim()) {
      localStorage.setItem("spendlens_groq_key", keyInput.value.trim());
      keyStatus.textContent = "LLM phrasing active"; keyStatus.className = "pill pill-green";
    } else {
      localStorage.removeItem("spendlens_groq_key");
      keyStatus.textContent = "Offline template mode"; keyStatus.className = "pill pill-amber";
    }
  });
  if (!savedKey) { keyStatus.textContent = "Offline template mode"; keyStatus.className = "pill pill-amber"; }

  function addMsg(role, html, src) {
    const div = document.createElement("div");
    div.className = "chat-msg " + role;
    div.innerHTML = html + (src ? `<span class="src">${src}</span>` : "");
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  async function send(text) {
    if (!text.trim()) return;
    addMsg("user", escapeHtml(text));
    input.value = "";
    const thinking = document.createElement("div");
    thinking.className = "chat-msg bot"; thinking.textContent = "…";
    log.appendChild(thinking); log.scrollTop = log.scrollHeight;

    const result = await answerQuestion(text);
    thinking.remove();
    addMsg("bot", result.answer, result.source);
  }

  sendBtn.addEventListener("click", () => send(input.value));
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(input.value); });
  document.querySelectorAll(".chat-suggest button").forEach(b => b.addEventListener("click", () => send(b.textContent)));

  addMsg("bot", "Ask me about channel ROI, saturation, budget recommendations, drift, or model accuracy — I only answer from the numbers already computed in this dashboard.", "offline template");
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
