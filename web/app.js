(() => {
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const C = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", yellow: "#c98500", magenta: "#d55181", violet: "#9085e9",
  good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b",
  ink: "#ffffff", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", surface: "#1a1a19" };
const SERIES = [C.blue, C.orange, C.aqua, C.yellow, C.magenta, C.violet];
const CHANNELS = {
  feed: "tick counts per bar, backfill, reconnects",
  bars: "closed bars: close, return, trades, volume",
  features: "standardized inputs at the latest bar",
  model: "forecast quantiles, P(up), regime gate, attention",
  policy: "how each forecast became BUY, SELL or HOLD",
  learn: "matured labels, training steps, loss",
  news: "headless-browser skim: headlines, tone, ideas",
  signals: "WSB retail chatter, guru 13F refreshes, and the whole-market scan",
  operator: "input you inject from the dashboard, and the nudges it applies",
  system: "checkpoints, settings, lifecycle",
};
const CONTROL_FIELDS = [
  ["score_threshold", "score threshold", 0.01], ["prob_margin", "P(up) margin", 0.01], ["cost_bps", "cost (bps)", 0.1],
  ["max_size", "max size", 0.05], ["lr", "learning rate", 0.0001], ["steps_per_label", "steps per label", 1],
  ["min_labels", "labels before trusted", 1], ["news_minutes", "news every (min)", 1],
];

const state = { config: null, status: {}, controls: {}, prices: {}, bars: {}, latest: {}, gate: [], outcomes: {},
  metrics: {}, history: { loss: [], hit: [], coverage: [], pnl: [] }, log: [], news: null, bars_fine: {},
  sources: [], news_sources: [], providers: {}, classes: {},
  signals: null, signal_providers: [], brief: null, burry: { enabled: true }, keys: [], muted: [], classes: {}, universe: [] };
const cards = {};
const consoles = {};
const lastPrices = {};
let ws, retry = 1000, drawQueued = false;
let audioCtx = null, soundOn = false;
let demoLoading = /[?&#]loading\b/i.test(location.href);
const prevAction = {};
try { soundOn = localStorage.getItem("flint.sound") === "1"; } catch (e) { /* ignore */ }

function beep(freq, dur, when = 0, type = "sine", gain = 0.14) {
  if (!soundOn || !audioCtx) return;
  const t = audioCtx.currentTime + when;
  const osc = audioCtx.createOscillator(), g = audioCtx.createGain();
  osc.type = type; osc.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g); g.connect(audioCtx.destination);
  osc.start(t); osc.stop(t + dur + 0.02);
}

function ringEvent(kind) {
  if (kind === "buy") { beep(660, 0.12); beep(990, 0.16, 0.12, "sine", 0.16); }
  else if (kind === "sell") { beep(440, 0.12); beep(300, 0.18, 0.12, "sine", 0.16); }
  else if (kind === "flip") { beep(520, 0.09); beep(400, 0.09, 0.1); beep(300, 0.14, 0.2); }
}

function checkAlerts() {
  if (!state.config) return;
  const mutedSet = new Set(state.muted || []);
  state.config.symbols.forEach(sym => {
    const L = state.latest[sym];
    if (!L || mutedSet.has(sym)) return;
    const now = L.trusted && L.side ? L.action : "HOLD";
    const was = prevAction[sym];
    if (was !== undefined && now !== was && now !== "HOLD") {
      const flipped = L.overlay && L.overlay.some(n => /contrarian flip/.test(n));
      ringEvent(flipped ? "flip" : now === "BUY" ? "buy" : "sell");
      flashCard(sym);
    }
    prevAction[sym] = now;
  });
}

function flashCard(sym) {
  const card = cards[sym];
  if (!card) return;
  card.el.classList.add("flash");
  setTimeout(() => card.el.classList.remove("flash"), 1400);
}

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const safeUrl = u => /^https?:\/\//i.test(u || "") ? u : "#";
const fmtPrice = v => v == null ? "·" : v >= 100 ? v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : v >= 1 ? v.toFixed(3) : v.toFixed(5);
const fmtBps = v => v == null ? "·" : (v >= 0 ? "+" : "") + v.toFixed(1);
const fmtPct = v => v == null ? "·" : (v * 100).toFixed(1) + "%";
const fmtTime = t => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
const fmtDur = s => s < 60 ? `${s | 0}s` : s < 3600 ? `${(s / 60) | 0}m ${(s % 60) | 0}s` : `${(s / 3600) | 0}h ${((s % 3600) / 60) | 0}m`;
const base = sym => sym.split("-")[0];

// Canvas helpers ------------------------------------------------------------------

function ctx2d(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(10, r.width | 0), h = Math.max(10, r.height | 0);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function ema(a, p) { const k = 2 / (p + 1); let e; return a.map((v, i) => (e = i ? v * k + e * (1 - k) : v)); }

function drawChart(card) {
  const sym = card.sym;
  const fine = card.res === "1m";
  const src = fine ? (state.bars_fine[sym] || []) : (state.bars[sym] || []);
  const cap = fine ? ((state.config && state.config.fine_bars) || 240) : ((state.config && state.config.chart_bars) || 160);
  const bars = src.slice(-cap);
  const { ctx, w, h } = ctx2d(card.canvas);
  ctx.font = "10px system-ui, sans-serif";
  if (bars.length < 2) { ctx.fillStyle = C.muted; ctx.fillText("waiting for bars", 10, 20); return; }
  const cfg = state.config, H = cfg.horizon, L = state.latest[sym];
  const padL = 8, padR = 8, padT = 10, padB = 16;
  const n = bars.length;
  const plotW = w - padL - padR;
  const fanW = fine ? 8 : Math.max(0.2 * plotW, 64);
  const dx = (plotW - fanW) / (n - 1);
  const xs = i => Math.min(padL + i * dx, padL + plotW);
  const closes = bars.map(b => b.c);
  const upC = "#26a269", downC = "#e0574b";

  // panes: price on top, MACD below, shared x axis
  const macdH = 46, gap = 8, axisY = h - padB;
  const mB = axisY, mT = mB - macdH, pT = padT, pB = mT - gap;

  const live = state.prices[sym] && state.prices[sym].price;
  let lo = Math.min(...bars.map(b => b.l)), hi = Math.max(...bars.map(b => b.h));
  if (live) { lo = Math.min(lo, live); hi = Math.max(hi, live); }
  let fan = null;
  if (!fine && L && L.q) { fan = L.q.map(q => L.price * Math.exp(q / 1e4)); }   // fan drawn but clipped; wide model bands must not crush the candles
  const span = (hi - lo) || closes[n - 1] * 1e-4;
  lo -= span * 0.06; hi += span * 0.06;
  const ys = v => pT + (hi - v) / (hi - lo) * (pB - pT);

  // price grid + labels
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  for (let k = 0; k < 3; k++) {
    const v = lo + (hi - lo) * (0.14 + 0.36 * k), y = Math.round(ys(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    const label = fmtPrice(v), tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(26,26,25,0.85)"; ctx.fillRect(padL, y - 13, tw + 4, 12);
    ctx.fillStyle = C.muted; ctx.fillText(label, padL + 2, y - 7);
  }

  // forecast fan
  const xLast = xs(n - 1), xEnd = padL + plotW;
  const anchor = L ? L.price : closes[n - 1];
  if (fan) {
    ctx.save();
    ctx.beginPath(); ctx.rect(padL, pT, w - padL - padR, pB - pT); ctx.clip();   // keep the fan inside the price pane
    const wedge = (a, b, color) => { ctx.beginPath(); ctx.moveTo(xLast, ys(anchor)); ctx.lineTo(xEnd, ys(a)); ctx.lineTo(xEnd, ys(b)); ctx.closePath(); ctx.fillStyle = color; ctx.fill(); };
    wedge(fan[0], fan[4], "rgba(57,135,229,0.16)");
    wedge(fan[1], fan[3], "rgba(57,135,229,0.30)");
    ctx.setLineDash([4, 3]); ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(xLast, ys(anchor)); ctx.lineTo(xEnd, ys(fan[2])); ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    const my = Math.max(pT + 8, Math.min(pB - 2, ys(fan[2]) - 9));   // keep the median label on-pane
    ctx.fillStyle = C.ink2; ctx.textAlign = "right"; ctx.fillText(`${fmtBps(L.q[2])} bps`, xEnd - 2, my); ctx.textAlign = "left";
  }

  // candlesticks -- but spot feeds (e.g. metals via GoldAPI) have no intrabar OHLC, so their
  // bars are flat (o=h=l=c) and would render as meaningless dashes; fall back to a close line there.
  const flat = bars.reduce((k, b) => k + (b.o === b.h && b.h === b.l && b.l === b.c ? 1 : 0), 0);
  if (flat < bars.length * 0.5) {
    const cw = Math.max(1, Math.min(dx * 0.68, 9));
    bars.forEach((b, i) => {
      const x = xs(i), col = b.c >= b.o ? upC : downC;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
      const wx = Math.round(x) + 0.5;
      ctx.beginPath(); ctx.moveTo(wx, ys(b.h)); ctx.lineTo(wx, ys(b.l)); ctx.stroke();
      const yo = ys(b.o), yc = ys(b.c);
      ctx.fillRect(x - cw / 2, Math.min(yo, yc), cw, Math.max(1, Math.abs(yc - yo)));
    });
  } else {
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath(); bars.forEach((b, i) => (i ? ctx.lineTo(xs(i), ys(b.c)) : ctx.moveTo(xs(i), ys(b.c)))); ctx.stroke();
  }
  if (live) {
    ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(xLast, ys(live), 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = C.surface; ctx.lineWidth = 2; ctx.stroke();
  }

  // MACD pane (12/26/9)
  const ef = ema(closes, 12), es = ema(closes, 26);
  const mline = closes.map((_, i) => ef[i] - es[i]);
  const sigl = ema(mline, 9);
  const hist = mline.map((v, i) => v - sigl[i]);
  let mmax = 1e-9;
  for (let i = 0; i < n; i++) mmax = Math.max(mmax, Math.abs(mline[i]), Math.abs(sigl[i]), Math.abs(hist[i]));
  mmax *= 1.15;
  const ym = v => (mT + mB) / 2 - (v / mmax) * (macdH / 2);
  const hw = Math.max(1, Math.min(dx * 0.6, 7));
  hist.forEach((v, i) => { const x = xs(i), y0 = ym(0), y1 = ym(v); ctx.fillStyle = v >= 0 ? "rgba(38,160,105,0.55)" : "rgba(224,87,75,0.55)"; ctx.fillRect(x - hw / 2, Math.min(y0, y1), hw, Math.max(1, Math.abs(y1 - y0))); });
  const zeroY = Math.round(ym(0)) + 0.5;
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(xLast, zeroY); ctx.stroke();
  const drawLine = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.lineJoin = "round"; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = xs(i), y = ym(arr[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); };
  drawLine(mline, "#3987e5");
  drawLine(sigl, "#e0913b");
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
  let lx = padL + 1;
  ctx.fillStyle = "#3987e5"; ctx.fillText("MACD", lx, mT + 9); lx += ctx.measureText("MACD").width + 6;
  ctx.fillStyle = "#e0913b"; ctx.fillText("signal", lx, mT + 9); lx += ctx.measureText("signal").width + 6;
  ctx.fillStyle = C.muted; ctx.fillText("12·26·9", lx, mT + 9);

  // x axis + now marker — show dates when the window spans more than a day, so a multi-day
  // chart doesn't read as a few minutes (times of day alone repeat every 24h)
  const spanH = (bars[n - 1].t - bars[0].t) / 3600;
  const fmtAxis = ts => { const d = new Date(ts * 1000), hm = d.toTimeString().slice(0, 5);
    return spanH > 20 ? `${d.getMonth() + 1}/${d.getDate()} ${hm}` : d.toTimeString().slice(0, 8); };
  const mid = (n - 1) >> 1;
  ctx.fillStyle = C.muted; ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left"; ctx.fillText(fmtAxis(bars[0].t), padL, h - 4);
  ctx.textAlign = "center"; ctx.fillText(fmtAxis(bars[mid].t), xs(mid), h - 4);
  ctx.textAlign = "right"; ctx.fillText(fmtAxis(bars[n - 1].t), xLast, h - 4);
  if (!fine) ctx.fillText(`+${Math.round(H * cfg.bar_seconds / 60)}m`, xEnd, h - 4); ctx.textAlign = "left";
  ctx.strokeStyle = C.axis; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(Math.round(xLast) + 0.5, pT); ctx.lineTo(Math.round(xLast) + 0.5, mB); ctx.stroke(); ctx.setLineDash([]);

  card.geom = { padL, dx, n, bars };
  if (card.hoverX != null) {
    const i = Math.max(0, Math.min(n - 1, Math.round((card.hoverX - padL) / dx)));
    const x = xs(i);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x, pT); ctx.lineTo(x, mB); ctx.stroke(); ctx.setLineDash([]);
    const b = bars[i], prev = i ? bars[i - 1].c : b.o;
    ctx.fillStyle = b.c >= b.o ? upC : downC; ctx.beginPath(); ctx.arc(x, ys(b.c), 3.5, 0, Math.PI * 2); ctx.fill();
    showTooltip(card.tipX, card.tipY, `${fmtTime(b.t)}\nO ${fmtPrice(b.o)}  H ${fmtPrice(b.h)}\nL ${fmtPrice(b.l)}  C ${fmtPrice(b.c)}\n${fmtBps(Math.log(b.c / prev) * 1e4)} bps  vol ${b.v.toPrecision(3)}\nMACD ${mline[i].toFixed(3)}  sig ${sigl[i].toFixed(3)}`);
  }
}

function drawSpark(sel, values, { color = C.blue, fmt = v => v.toFixed(2), ref = null } = {}) {
  const canvas = $(sel);
  const { ctx, w, h } = ctx2d(canvas);
  const vals = (values || []).filter(v => v != null);
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "alphabetic";
  if (vals.length < 2) { ctx.fillStyle = C.muted; ctx.fillText("collecting", 6, 16); return; }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (ref != null) { lo = Math.min(lo, ref); hi = Math.max(hi, ref); }
  if (hi === lo) { hi += 1e-6; lo -= 1e-6; }
  const padT = 14, padB = 6, padR = 52;
  const xs = i => 4 + i / (vals.length - 1) * (w - padR - 4);
  const ys = v => padT + (hi - v) / (hi - lo) * (h - padT - padB);
  if (ref != null) {
    ctx.strokeStyle = C.axis; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(4, Math.round(ys(ref)) + 0.5); ctx.lineTo(w - padR, Math.round(ys(ref)) + 0.5); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.beginPath(); vals.forEach((v, i) => i ? ctx.lineTo(xs(i), ys(v)) : ctx.moveTo(xs(i), ys(v))); ctx.stroke();
  const last = vals[vals.length - 1];
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(xs(vals.length - 1), ys(last), 3.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = C.ink; ctx.font = "12px ui-monospace, Menlo, monospace"; ctx.textBaseline = "middle";
  ctx.fillText(fmt(last), w - padR + 6, ys(last));
  ctx.fillStyle = C.muted; ctx.font = "10px system-ui, sans-serif"; ctx.textBaseline = "alphabetic";
  ctx.fillText(fmt(hi), 4, 10); ctx.fillText(fmt(lo), 4, h - 1);
}

const tooltip = $("#tooltip");
function showTooltip(x, y, text) {
  tooltip.textContent = text; tooltip.style.display = "block";
  const r = tooltip.getBoundingClientRect();
  tooltip.style.left = Math.min(x + 14, window.innerWidth - r.width - 8) + "px";
  tooltip.style.top = Math.max(4, y - r.height - 12) + "px";
}
function hideTooltip() { tooltip.style.display = "none"; }

// Dashboard -----------------------------------------------------------------------

function buildCards() {
  const root = $("#cards"); root.innerHTML = "";
  for (const k of Object.keys(cards)) delete cards[k];
  state.config.symbols.forEach(sym => {
    const el = document.createElement("section"); el.className = "card"; el.dataset.sym = sym;
    el.innerHTML = `<header><span class="sym">${esc(sym)}</span><span class="spread num"></span><span class="price num">·</span></header>
      <div class="via" title="active data source for this symbol"></div>
      <div class="signal"><span class="badge hold">■ HOLD</span><span class="size num"></span><span class="warm" hidden>warming up</span></div>
      <div class="chart-wrap"><canvas class="chart"></canvas><span class="res-toggle"><button data-res="5m" class="on">5m</button><button data-res="1m">1m</button></span></div>
      <div class="stats"><div><label>median</label><b class="q50">·</b></div><div><label>10-90 band</label><b class="band">·</b></div>
        <div><label>P(up)</label><b class="pup">·</b></div><div><label>score</label><b class="score">·</b></div></div>
      <div class="pmeter" title="P(up): fill right of centre is up, left is down"><i></i></div>
      <div class="outcomes" title="matured forecasts, newest right: green hit, red miss, outlined = acted on"></div>
      <div class="why"></div>
      <div class="strategy"></div>
      <div class="fundamentals"></div>`;
    root.appendChild(el);
    const card = cards[sym] = { sym, el, canvas: $("canvas", el), hoverX: null, res: "5m" };
    $$(".res-toggle button", el).forEach(btn => btn.onclick = () => {
      card.res = btn.dataset.res;
      $$(".res-toggle button", el).forEach(b => b.classList.toggle("on", b === btn));
      drawChart(card);
    });
    card.canvas.addEventListener("mousemove", e => { card.hoverX = e.offsetX; card.tipX = e.clientX; card.tipY = e.clientY; drawChart(card); });
    card.canvas.addEventListener("mouseleave", () => { card.hoverX = null; hideTooltip(); drawChart(card); });
  });
}

function updatePrice(sym) {
  const card = cards[sym]; if (!card) return;
  const p = state.prices[sym] || {};
  const el = $(".price", card.el), prev = lastPrices[sym];
  el.textContent = fmtPrice(p.price);
  if (prev != null && p.price !== prev) { el.classList.toggle("up", p.price > prev); el.classList.toggle("down", p.price < prev); }
  lastPrices[sym] = p.price;
  $(".spread", card.el).textContent = p.bid && p.ask ? `spread ${((p.ask - p.bid) / ((p.ask + p.bid) / 2) * 1e4).toFixed(1)} bps` : "";
}

function updateVia(sym) {
  const card = cards[sym]; if (!card) return;
  const via = $(".via", card.el);
  const pid = state.providers[sym];
  const src = state.sources.find(x => x.id === pid);
  const cls = state.classes[sym] || "";
  if (!pid) { via.className = "via none"; via.innerHTML = `${cls} · <b>no live source</b>`; }
  else { via.className = "via" + (pid === "sim" ? " sim" : ""); via.innerHTML = `${cls} · via <b>${esc(src ? src.name : pid)}</b>`; }
}

function updateCard(sym) {
  const card = cards[sym]; if (!card) return;
  updatePrice(sym); updateVia(sym);
  const L = state.latest[sym];
  if (L) {
    const badge = $(".badge", card.el);
    badge.className = "badge " + L.action.toLowerCase();
    badge.textContent = (L.action === "BUY" ? "▲ " : L.action === "SELL" ? "▼ " : "■ ") + L.action;
    $(".size", card.el).textContent = L.side ? `size ${L.size.toFixed(2)}` : "";
    $(".warm", card.el).hidden = !!L.trusted;
    $(".q50", card.el).textContent = `${fmtBps(L.q[2])} bps`;
    $(".band", card.el).textContent = `${fmtBps(L.q[0])} to ${fmtBps(L.q[4])}`;
    $(".pup", card.el).textContent = L.p_up.toFixed(2);
    $(".score", card.el).textContent = (L.score >= 0 ? "+" : "") + L.score.toFixed(2);
    const m = $(".pmeter i", card.el), pu = L.p_up;
    if (pu >= 0.5) { m.style.left = "50%"; m.style.width = `${(pu - 0.5) * 100}%`; m.style.background = C.good; }
    else { m.style.left = `${pu * 100}%`; m.style.width = `${(0.5 - pu) * 100}%`; m.style.background = C.critical; }
    $(".why", card.el).textContent = L.why;
    const strat = $(".strategy", card.el);
    if (L.crowding != null) {
      const crowd = L.crowding, hot = crowd > 0.15, cold = crowd < -0.15;
      const acted = L.base_action && L.base_action !== L.action;
      strat.classList.toggle("acted", !!(acted || (L.overlay && L.overlay.length)));
      let txt = `crowding <b class="crowd-chip">${crowd >= 0 ? "+" : ""}${crowd.toFixed(2)}</b>`;
      if (L.guru_tilt) txt += ` · guru <b class="crowd-chip">${L.guru_tilt >= 0 ? "+" : ""}${L.guru_tilt.toFixed(2)}</b>`;
      if (acted) txt += ` · overlay: ${esc(L.base_action)}→<b>${esc(L.action)}</b>`;
      if (L.overlay && L.overlay.length) txt += ` — ${esc(L.overlay[0])}`;
      strat.innerHTML = txt;
    } else strat.textContent = "";
  }
  const outs = (state.outcomes[sym] || []).slice(-40);
  $(".outcomes", card.el).innerHTML = outs.map(o =>
    `<i class="${o.hit === true ? "hit" : o.hit === false ? "miss" : ""}${o.side ? " acted" : ""}" title="${fmtTime(o.t)}: realized ${fmtBps(o.y)} bps vs median ${fmtBps(o.q50)}${o.pnl != null ? `, paper ${fmtBps(o.pnl)} bps` : ""}"></i>`).join("");
  drawChart(card);
}

function renderTiles() {
  const m = state.metrics || {}, cfg = state.config;
  const f2 = v => v != null ? v.toFixed(2) : "·";
  const tiles = [
    { label: "labels learned", value: m.labels ?? 0, small: m.trusted ? `${m.live_labels} live, trusted` : `${m.live_labels ?? 0} live, need ${cfg.min_labels}` },
    { label: "training steps", value: m.steps ?? 0 },
    { label: "train pinball", value: f2(m.pinball), small: "bps, in sample" },
    { label: "live pinball", value: f2(m.live_pinball), small: "bps, out of sample" },
    { label: "hit rate", value: fmtPct(m.hit_rate), small: m.decisions ? `n=${m.decisions}` : "", cls: m.hit_rate > 0.55 ? "good" : (m.hit_rate < 0.45 && m.decisions > 50) ? "bad" : "" },
    { label: "10-90 coverage", value: fmtPct(m.coverage), small: `raw ${fmtPct(m.coverage_raw)}` },
    { label: "band scale", value: m.band_scale != null ? m.band_scale.toFixed(2) + "x" : "·", small: "conformal" },
    { label: "P(up) temper", value: f2(m.p_scale), small: "1 = raw" },
    { label: "paper P&L", value: fmtBps(m.pnl_bps), small: m.suggestions ? `${m.suggestions} calls, ${fmtPct(m.win_rate)} won` : "no calls yet", cls: m.pnl_bps > 0 ? "good" : m.pnl_bps < 0 ? "bad" : "" },
  ];
  $("#tiles").innerHTML = tiles.map(t => `<div class="tile ${t.cls || ""}"><label>${t.label}</label><b class="num">${t.value ?? "·"}</b>${t.small ? `<small>${t.small}</small>` : ""}</div>`).join("");
}

function renderSparks() {
  const h = state.history;
  drawSpark("#spark-loss", h.loss, { color: C.orange, fmt: v => v.toFixed(2) });
  drawSpark("#spark-hit", h.hit, { color: C.aqua, ref: 0.5, fmt: v => (v * 100).toFixed(0) + "%" });
  drawSpark("#spark-coverage", h.coverage, { color: C.blue, ref: 0.8, fmt: v => (v * 100).toFixed(0) + "%" });
  drawSpark("#spark-pnl", h.pnl, { color: C.yellow, ref: 0, fmt: v => fmtBps(v) });
}

function renderGate() {
  const g = state.gate || [];
  $("#gate").innerHTML = g.length ? g.map((v, k) =>
    `<div class="g"><span>E${k + 1}</span><div class="bar"><i style="width:${(v * 100).toFixed(1)}%;background:${SERIES[k % SERIES.length]}"></i></div><span>${v.toFixed(2)}</span></div>`).join("")
    : `<span class="why">no forecast yet</span>`;
}

function attnColor(v) {
  const t = Math.max(0, Math.min(1, v));
  const mix = (a, b, k) => a.map((x, i) => Math.round(x + (b[i] - x) * k));
  const c = t < 0.5 ? mix([34, 34, 32], [37, 106, 191], t * 2) : mix([37, 106, 191], [158, 197, 244], (t - 0.5) * 2);
  return `rgb(${c.join(",")})`;
}

function renderAttn() {
  const syms = state.config.symbols, el = $("#attn");
  let cells = `<div class="c h"></div>` + syms.map(s => `<div class="c h">${esc(base(s))}</div>`).join("");
  syms.forEach(s => {
    const L = state.latest[s];
    cells += `<div class="c h">${esc(base(s))}</div>` + syms.map((_, j) => {
      const v = (L && L.attn && L.attn[j]) || 0;
      return `<div class="c" style="background:${attnColor(v)}">${v.toFixed(2)}</div>`;
    }).join("");
  });
  el.innerHTML = `<div class="attn-grid" style="grid-template-columns:34px repeat(${syms.length}, minmax(30px,1fr))">${cells}</div>`;
}

function renderArch() {
  const c = state.config, m = c.model;
  $("#arch").innerHTML = `FlintNet, <b>${m.params.toLocaleString()}</b> parameters<br>` +
    `${c.symbols.length} assets x <b>${c.window}</b> bars of <b>${c.bar_seconds}s</b>, ${c.features.length} features each<br>` +
    `causal conv dilations <b>${m.dilations.join(",")}</b>, receptive field ${m.receptive_field} bars<br>` +
    `cross-asset attention with <b>${m.heads}</b> heads, <b>${m.experts}</b> regime experts<br>` +
    `quantiles <b>${c.quantiles.join("/")}</b> of the <b>${c.horizon}</b>-bar (${c.horizon * c.bar_seconds}s) return<br>` +
    `feed <b>${esc(state.status.feed || "?")}</b>, news skim every <b>${state.controls.news_minutes}</b> min`;
}

function renderLog() {
  $("#log").innerHTML = state.log.slice().reverse().map(l =>
    `<li class="${esc(l.kind)}"><span class="t">${fmtTime(l.t)}</span><span>${esc(l.text)}</span></li>`).join("")
    || `<li><span>no resolved suggestions yet</span></li>`;
}

function renderStatus() {
  const s = state.status || {};
  const phase = $("#phase");
  phase.textContent = s.phase || "?";
  phase.className = "pill " + (s.phase === "live" ? "live" : s.phase === "error" ? "err" : "warn");
  $("#feed").textContent = "feed: " + (s.feed || "?");
  $("#bars").textContent = s.bar_index ?? 0;
  $("#pending").textContent = s.pending ?? 0;
  $("#learning").classList.toggle("off", s.learning === false);
  const mk = $("#market");
  if (mk) {
    let fresh = 0;
    for (const v of Object.values(state.prices || {})) if (v && v.ts) fresh = Math.max(fresh, v.ts);
    const age = fresh ? (Date.now() / 1000 - fresh) : Infinity;
    if (age < 90) { mk.textContent = "● market open"; mk.className = "pill market-nav live"; mk.title = "live trades arriving"; }
    else if (fresh) { mk.textContent = "● market closed"; mk.className = "pill market-nav closed"; mk.title = "no live trades — last at " + fmtTime(fresh); }
    else { mk.textContent = "● market —"; mk.className = "pill market-nav closed"; mk.title = "no data yet"; }
  }
  updateLoading();
}

function seedSparks() {
  const f = $("#spark-field");
  if (!f || f.childElementCount) return;
  const hues = ["#fab219", "#ec835a", "#fff2b0", "#ff8b3a", "#d98a2b"];
  for (let i = 0; i < 30; i++) {
    const e = document.createElement("i");
    e.className = "ember";
    const dur = 1.8 + Math.random() * 2.8, size = 1 + Math.random() * 2.6;
    e.style.left = (Math.random() * 100) + "%";
    e.style.width = e.style.height = size.toFixed(1) + "px";
    e.style.setProperty("--dur", dur.toFixed(2) + "s");
    e.style.setProperty("--delay", (-Math.random() * dur).toFixed(2) + "s");
    e.style.setProperty("--drift", (Math.random() * 60 - 30).toFixed(0) + "px");
    e.style.setProperty("--hue", hues[i % hues.length]);
    f.appendChild(e);
  }
}

function updateLoading(offline) {
  const el = $("#loading");
  if (!el) return;
  seedSparks();
  if (demoLoading) {
    el.hidden = false;
    $("#loading-title").textContent = "Starting a fire…";
    $("#loading-sub").textContent = "loading-screen preview — click to dismiss";
    return;
  }
  const phase = offline ? "offline" : ((state.status && state.status.phase) || "starting");
  const busy = phase !== "live" && phase !== "error";
  el.hidden = !busy;
  if (!busy) return;
  const title = $("#loading-title"), sub = $("#loading-sub");
  const feed = state.status && state.status.feed;
  if (phase === "offline") { title.textContent = "Rekindling…"; sub.textContent = "reconnecting to flint"; }
  else if (phase === "starting") { title.textContent = "Striking flint…"; sub.textContent = "waking up"; }
  else if (phase.indexOf("backfilling") === 0) { title.textContent = "Gathering tinder…"; sub.textContent = "backfilling market history" + (feed && feed !== "none" ? ` from ${feed}` : ""); }
  else if (phase === "training") { title.textContent = "Starting a fire…"; const m = state.metrics || {}; sub.textContent = `training on history — step ${m.steps || 0}`; }
  else { title.textContent = "Kindling the model…"; sub.textContent = phase; }
}

function renderAll() {
  if (!state.config) return;
  renderStatus();
  syncControls();
  if (document.body.dataset.view === "dashboard") {
    state.config.symbols.forEach(updateCard);
    renderTiles(); renderSparks(); renderGate(); renderAttn(); renderArch(); renderLog(); renderMarket(); renderWatch(); reorderCards(true);
  }
  renderNews();
  if (document.body.dataset.view === "consoles") { renderUniverse(); renderKeys(); renderSources(); renderSignals(); renderRadar(); renderBriefCtl(); }
  if (document.body.dataset.view === "brief") renderBrief();
}

function scheduleDraw() {
  if (drawQueued || document.body.dataset.view !== "dashboard") return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; Object.values(cards).forEach(drawChart); });
}

// Consoles + controls ----------------------------------------------------------------

function buildConsoles() {
  const grid = $("#console-grid"); grid.innerHTML = "";
  for (const k of Object.keys(consoles)) delete consoles[k];
  Object.entries(CHANNELS).forEach(([ch, desc]) => {
    const el = document.createElement("div");
    el.className = "console" + (ch === "model" || ch === "news" ? " tall" : "");
    el.innerHTML = `<header><span class="name">${ch}</span><span class="desc">${desc}</span><span class="rate">0/min</span>
      <button type="button" class="pause">pause</button><button type="button" class="clear">clear</button></header><div class="body"></div>`;
    grid.appendChild(el);
    const c = consoles[ch] = { el, body: $(".body", el), paused: false, times: [], rate: $(".rate", el) };
    $(".pause", el).onclick = e => {
      c.paused = !c.paused; el.classList.toggle("paused", c.paused);
      e.target.textContent = c.paused ? "resume" : "pause";
      if (!c.paused) c.body.scrollTop = c.body.scrollHeight;
    };
    $(".clear", el).onclick = () => { c.body.innerHTML = ""; };
  });
}

function appendTrace(ev, scroll = true) {
  const c = consoles[ev.ch]; if (!c) return;
  const atBottom = c.body.scrollHeight - c.body.scrollTop - c.body.clientHeight < 40;
  const line = document.createElement("div");
  line.className = "line " + (ev.lvl || "info");
  line.innerHTML = `<span class="t">${fmtTime(ev.t)}</span><span class="msg">${esc(ev.text)}</span>`;
  c.body.appendChild(line);
  while (c.body.childElementCount > 400) c.body.removeChild(c.body.firstChild);
  if (scroll) c.times.push(Date.now() / 1000);
  if (scroll && !c.paused && atBottom) c.body.scrollTop = c.body.scrollHeight;
}

// ---- unified console: a terminal-style firehose of every trace channel ----
const termFilters = new Set(Object.keys(CHANNELS));
let termFollow = true;
function appendTerm(ev) {
  const term = $("#term"); if (!term) return;
  const atBottom = term.scrollHeight - term.scrollTop - term.clientHeight < 60;
  const line = document.createElement("div");
  line.className = "tline " + (ev.lvl || "info");
  line.dataset.ch = ev.ch;
  line.innerHTML = `<span class="tt">${fmtTime(ev.t)}</span><span class="tch ch-${ev.ch}">${esc(ev.ch)}</span><span class="tmsg">${esc(ev.text)}</span>`;
  if (!termFilters.has(ev.ch)) line.style.display = "none";
  term.appendChild(line);
  while (term.childElementCount > 2500) term.removeChild(term.firstChild);
  if (termFollow && atBottom) term.scrollTop = term.scrollHeight;
}
function setTermFollow(on) {
  termFollow = on;
  const fb = $("#term-follow"); if (fb) { fb.classList.toggle("on", on); fb.textContent = on ? "following" : "paused"; }
  if (on) { const t = $("#term"); if (t) t.scrollTop = t.scrollHeight; }
}
(function initTerm() {
  const box = $("#term-filters"); if (!box) return;
  box.innerHTML = Object.keys(CHANNELS).map(ch =>
    `<button class="tchip ch-${ch} on" data-ch="${ch}" title="${esc(CHANNELS[ch])}">${ch}</button>`).join("");
  box.addEventListener("click", e => {
    const b = e.target.closest(".tchip"); if (!b) return;
    const ch = b.dataset.ch;
    termFilters.has(ch) ? termFilters.delete(ch) : termFilters.add(ch);
    b.classList.toggle("on", termFilters.has(ch));
    $$("#term .tline").forEach(l => { if (l.dataset.ch === ch) l.style.display = termFilters.has(ch) ? "" : "none"; });
  });
  $("#term-follow").addEventListener("click", () => setTermFollow(!termFollow));
  $("#term-clear").addEventListener("click", () => { $("#term").innerHTML = ""; });
  $("#term").addEventListener("scroll", () => {
    const t = $("#term"); const atBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 60;
    if (atBottom !== termFollow) setTermFollow(atBottom);
  });
})();

function buildControls() {
  const form = $("#controls");
  form.innerHTML = CONTROL_FIELDS.map(([k, label, step]) =>
    `<label>${label}<input name="${k}" type="number" step="${step}" value="${state.controls[k] ?? ""}"></label>`).join("") +
    `<button type="submit">Apply</button>`;
  form.onsubmit = async e => {
    e.preventDefault();
    const set = {};
    CONTROL_FIELDS.forEach(([k]) => { const v = form.elements[k].value; if (v !== "" && Number(v) !== state.controls[k]) set[k] = Number(v); });
    if (Object.keys(set).length) await control({ set });
  };
}

function syncControls() {
  const form = $("#controls");
  if (!form.elements.length) return;
  CONTROL_FIELDS.forEach(([k]) => { const inp = form.elements[k]; if (inp && document.activeElement !== inp) inp.value = state.controls[k] ?? ""; });
  const b = $("#btn-pause");
  b.textContent = state.controls.learning === false ? "Resume learning" : "Pause learning";
  b.dataset.action = state.controls.learning === false ? "resume" : "pause";
  $("#learning").classList.toggle("off", state.controls.learning === false);
}

async function control(payload) {
  const r = await fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const res = await r.json();
  if (res.controls) { state.controls = res.controls; syncControls(); }
  if (res.metrics) { state.metrics = res.metrics; if (document.body.dataset.view === "dashboard") renderTiles(); }
  return res;
}

// --- inject input: a compose box that pipes a human note into the model ---
const injectFab = $("#inject-fab"), injectPop = $("#inject-pop"), injectText = $("#inject-text"), injectAck = $("#inject-ack");
function injectToggle(show) {
  injectPop.hidden = show === undefined ? !injectPop.hidden : !show;
  if (!injectPop.hidden) { injectText.focus(); }
}
async function injectSend() {
  const text = injectText.value.trim();
  if (!text) return;
  const res = await control({ action: "inject", text });
  if (res && res.applied && res.applied.length) {
    injectAck.textContent = `nudged ${res.applied.join(", ")} (${res.sentiment >= 0 ? "+" : ""}${res.sentiment})`;
  } else {
    injectAck.textContent = "logged to the operator console";
  }
  injectText.value = "";
  clearTimeout(injectSend._t); injectSend._t = setTimeout(() => { injectAck.textContent = ""; injectToggle(false); }, 2200);
}
if (injectFab) {
  injectFab.onclick = () => injectToggle();
  $("#inject-send").onclick = injectSend;
  injectText.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); injectSend(); }
    else if (e.key === "Escape") { injectToggle(false); }
  });
  document.addEventListener("keydown", e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") { e.preventDefault(); injectToggle(); }
  });
  document.addEventListener("click", e => {
    if (!injectPop.hidden && !injectPop.contains(e.target) && e.target !== injectFab) injectToggle(false);
  });
}

$$(".actions button").forEach(b => b.onclick = () => {
  const a = b.dataset.action;
  if (a === "reset" && !confirm("Re-initialize the model weights and optimizer? The replay buffer is kept.")) return;
  control({ action: a });
});

function sourceToggle(kind, id, on, disabled, reason) {
  const wrap = document.createElement("label");
  wrap.className = "switch" + (disabled ? " disabled" : "");
  if (reason) wrap.title = reason;
  wrap.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} ${disabled ? "disabled" : ""}><span class="track"></span><span class="thumb"></span>`;
  wrap.querySelector("input").onchange = e => {
    control({ action: kind === "news" ? "toggle_news" : "toggle_source", id, on: e.target.checked });
  };
  return wrap;
}

function applyMuted() {
  if (!state.config) return;
  const muted = new Set(state.muted || []);
  state.config.symbols.forEach(sym => { if (cards[sym]) cards[sym].el.classList.toggle("muted", muted.has(sym)); });
  lastOrder = ""; reorderCards(true);
}

function renderUniverse() {
  const box = $("#uni-list");
  if (!box || !state.config) return;
  const active = new Set(state.config.symbols || []);
  const universe = (state.universe && state.universe.length) ? state.universe : state.config.symbols;
  box.innerHTML = universe.map(sym => {
    const cls = (state.classes || {})[sym] || "";
    const on = active.has(sym);
    return `<div class="uni ${on ? "" : "muted"}"><span><span class="un">${esc(base(sym))}</span> <span class="uc">${esc(cls)}</span></span>` +
      `<span class="right"><label class="switch"><input type="checkbox" data-mute="${esc(sym)}" ${on ? "checked" : ""}><span class="track"></span><span class="thumb"></span></label>` +
      `<button type="button" class="rm" data-remove="${esc(sym)}" title="remove from universe">×</button></span></div>`;
  }).join("");
  const meta = $("#uni-meta");
  if (meta) meta.textContent = `${(state.config.symbols || []).length} active of ${universe.length} (cap ${state.config.max_universe || 64}) — mute to pause, remove to drop`;
  // class-level disable/enable
  const ct = $("#class-toggles");
  if (ct) {
    const universe2 = (state.universe && state.universe.length) ? state.universe : state.config.symbols;
    const classes = [...new Set(universe2.map(s => (state.classes || {})[s]).filter(Boolean))];
    ct.innerHTML = classes.map(cls => {
      const syms = universe2.filter(s => (state.classes || {})[s] === cls);
      const allMuted = syms.every(s => !active.has(s));
      return `<button type="button" class="${allMuted ? "" : "danger"}" data-class="${esc(cls)}" data-on="${allMuted ? "0" : "1"}">${allMuted ? "Enable" : "Disable"} ${esc(cls)}</button>`;
    }).join("");
  }
}

// ---- first-launch setup walkthrough (every key is optional) ----
const ONBOARD_KEY = "flint.onboarded";
let onbStep = 0;
function onboardSteps() {
  return [{ type: "intro" }, ...(state.keys || []).map(k => ({ type: "key", id: k.id })), { type: "brief" }, { type: "done" }];
}
function onboardStart(force) {
  if (!force) { try { if (localStorage.getItem(ONBOARD_KEY)) return; } catch (e) { /* storage may be unavailable */ } }
  if (!state.keys || !state.keys.length) return;      // wait until the service list is known
  onbStep = 0; const el = $("#onboard"); if (el) { el.hidden = false; renderOnboard(); }
}
function onboardClose() {
  try { localStorage.setItem(ONBOARD_KEY, "1"); } catch (e) { /* storage may be unavailable */ }
  const el = $("#onboard"); if (el) el.hidden = true;
}
function onbNext() { onbStep++; if (onbStep >= onboardSteps().length) onboardClose(); else renderOnboard(); }
function onbBack() { if (onbStep > 0) { onbStep--; renderOnboard(); } }
function renderOnboard() {
  const steps = onboardSteps();
  onbStep = Math.max(0, Math.min(onbStep, steps.length - 1));
  const step = steps[onbStep], n = steps.length, nKeys = n - 3;
  $("#onb-progress").innerHTML = steps.map((_, i) => `<span class="onb-dot${i === onbStep ? " on" : i < onbStep ? " done" : ""}"></span>`).join("");
  const body = $("#onboard-body");
  if (step.type === "intro") {
    body.innerHTML = `<h2>Welcome to flint</h2>` +
      `<p>flint runs out of the box on free, no-key data sources. Adding an API key unlocks better or faster data for that provider \u2014 but <b>every key is optional</b>, and you can add them anytime from the Control panel.</p>` +
      `<p class="onb-sub">Let's walk through them. Skip any you don't have.</p>` +
      `<div class="onb-actions"><button class="onb-ghost" id="onb-skipall">Skip setup</button><button class="onb-primary" id="onb-go">Get started</button></div>`;
  } else if (step.type === "key") {
    const k = (state.keys || []).find(x => x.id === step.id);
    if (!k) { onbNext(); return; }
    const fields = k.fields.map(f =>
      `<div class="onb-field"><label>${esc(f.label)}${f.present ? ` <span class="cur">set: ${esc(f.masked)}</span>` : ""}</label>` +
      `<input type="text" data-svc="${esc(k.id)}" data-field="${esc(f.id)}" placeholder="${f.present ? "replace\u2026" : "paste key\u2026"}" autocomplete="off" autocapitalize="off" spellcheck="false"></div>`).join("");
    body.innerHTML = `<div class="onb-step">key ${onbStep} of ${nKeys}</div>` +
      `<h2>${esc(k.name)} ${k.present ? '<span class="onb-set">\u2713 set</span>' : '<span class="onb-opt">optional</span>'}</h2>` +
      `<p>${esc(k.note || "")}</p>` +
      `<a class="onb-link" href="${esc(safeUrl(k.url))}" target="_blank" rel="noopener noreferrer">Get a ${esc(k.name)} key \u2197</a>` +
      `<div class="onb-fields">${fields}</div>` +
      `<div class="onb-actions"><button class="onb-ghost" id="onb-back">Back</button><button class="onb-ghost" id="onb-skip">Skip</button><button class="onb-primary" id="onb-save">Save &amp; continue</button></div>`;
  } else if (step.type === "brief") {
    body.innerHTML = `<h2>Narrative brief <span class="onb-opt">local \u00b7 optional</span></h2>` +
      `<p>flint can write a plain-English market brief with a local LLM through <b>Ollama</b> \u2014 it runs entirely on your machine, no key and no cloud. If Ollama is installed and running, the brief works automatically.</p>` +
      `<a class="onb-link" href="https://ollama.com/download" target="_blank" rel="noopener noreferrer">Install Ollama \u2197</a>` +
      `<p class="onb-sub">You can switch the brief on and off anytime in the Control panel.</p>` +
      `<div class="onb-actions"><button class="onb-ghost" id="onb-back">Back</button><button class="onb-primary" id="onb-go">Continue</button></div>`;
  } else {
    body.innerHTML = `<h2>You're all set</h2>` +
      `<p>flint is already running on free sources and learning from live data. Add or change keys anytime from <b>Control panel \u2192 API keys</b>.</p>` +
      `<div class="onb-actions"><button class="onb-primary" id="onb-go">Enter flint</button></div>`;
  }
  const on = (id, fn) => { const b = $("#" + id); if (b) b.onclick = fn; };
  on("onb-go", onbNext); on("onb-skip", onbNext); on("onb-back", onbBack); on("onb-skipall", onboardClose);
  on("onb-save", async () => {
    const svc = step.id;
    if (svc && $$(`#onboard input[data-svc="${svc}"]`).some(i => i.value.trim())) { await saveKey(svc); }
    onbNext();
  });
}

function renderKeys() {
  const box = $("#key-list");
  if (!box) return;
  box.innerHTML = (state.keys || []).map(k => {
    const fields = k.fields.map(f =>
      `<div class="kfield"><label>${esc(f.label)}${f.present ? ` <span class="cur">set: ${esc(f.masked)}</span>` : ""}</label>` +
      `<input type="text" class="kmask" data-svc="${esc(k.id)}" data-field="${esc(f.id)}" placeholder="${f.present ? "replace…" : "paste key…"}" autocomplete="off" autocapitalize="off" spellcheck="false"></div>`).join("");
    return `<div class="keyrow"><div class="kh"><span class="dot ${k.present ? "on" : "off"}"></span>` +
      `<span class="kname">${esc(k.name)}</span>` +
      `<a class="klink" href="${esc(safeUrl(k.url))}" target="_blank" rel="noopener noreferrer">${k.present ? "manage ↗" : "get a key ↗"}</a></div>` +
      `<div class="knote">${esc(k.note || "")}</div>` +
      `<div class="kfields">${fields}<button type="button" class="ksave" data-svc="${esc(k.id)}">Save</button></div></div>`;
  }).join("");
}

async function saveKey(service) {
  const values = {};
  const scope = ($("#onboard") && !$("#onboard").hidden) ? "#onboard " : "";
  $$(`${scope}input[data-svc="${service}"]`).forEach(i => { if (i.value.trim()) values[i.dataset.field] = i.value.trim(); });
  if (!Object.keys(values).length) return;
  const r = await fetch("/api/keys", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ service, values }) });
  const res = await r.json();
  if (res.keys) { state.keys = res.keys; renderKeys(); }
}

document.addEventListener("click", e => {
  const b = e.target.closest && e.target.closest("button.ksave");
  if (b) saveKey(b.dataset.svc);
  const cb = e.target.closest && e.target.closest("button[data-class]");
  if (cb) control({ action: "mute_class", class: cb.dataset.class, on: cb.dataset.on === "1" });
  const rm = e.target.closest && e.target.closest("button[data-remove]");
  if (rm) control({ action: "remove_symbols", symbols: [rm.dataset.remove] });
  if (e.target.closest && e.target.closest("button[data-action=add_symbol]")) {
    const inp = $("#uni-add-input"); const v = (inp.value || "").trim().toUpperCase();
    if (v) { control({ action: "add_symbols", symbols: v.split(/[ ,]+/).filter(Boolean) }); inp.value = ""; }
  }
  if (e.target.closest && e.target.closest("button[data-action=add_movers]")) control({ action: "add_movers", n: 20 });
});
document.addEventListener("change", e => {
  if (e.target && e.target.dataset && e.target.dataset.mute)
    control({ action: "mute", symbols: [e.target.dataset.mute], on: !e.target.checked });
});

function renderSources() {
  const list = $("#source-list");
  if (!list) return;
  list.innerHTML = "";
  state.sources.forEach(src => {
    const owning = src.owned && src.owned.length;
    const el = document.createElement("div");
    el.className = "source" + (src.enabled ? "" : " off") + (owning ? " owning" : "");
    const isSim = src.id === "sim";
    const capList = (a, n) => a.slice(0, n).map(base).join(", ") + (a.length > n ? ` +${a.length - n}` : "");
    const ownsText = owning ? `serving ${capList(src.owned, 8)}`
      : src.enabled ? (src.supported && src.supported.length ? "standby / backup" : "idle — no matching symbols") : "off";
    const noteErr = /fail|stopped|error|invalid|no API/i.test(src.note || "");
    el.innerHTML = `<div></div><div>
        <div class="sname">${esc(src.name)}
          <span class="tag ${esc(src.kind)}">${esc(src.kind)}</span>
          <span class="tag ${esc(src.mechanism)}">${esc(src.mechanism)}</span>
          <span class="tag" title="priority (lower wins)">P${src.priority}</span></div>
        <div class="meta">supports ${src.supported && src.supported.length ? capList(src.supported, 8) : "none of the active symbols"} · ${src.ticks.toLocaleString()} ticks</div>
        <div class="owns ${owning ? "" : "backup"}">${esc(ownsText)}</div>
        ${src.note ? `<div class="snote ${noteErr ? "err" : ""}">${esc(src.note)}</div>` : ""}
      </div>`;
    const disabled = isSim || (!src.enabled && (!src.supported || !src.supported.length));
    el.firstChild.replaceWith(sourceToggle("source", src.id, src.enabled, disabled,
      isSim ? "the simulator is the fallback provider" : (!src.supported || !src.supported.length) ? "no active symbols in this asset class" : ""));
    list.appendChild(el);
  });
  const nl = $("#news-source-list");
  if (nl) {
    nl.innerHTML = "";
    state.news_sources.forEach(src => {
      const el = document.createElement("div");
      el.className = "source" + (src.enabled ? "" : " off");
      el.innerHTML = `<div></div><div><div class="sname">${esc(src.name)}</div></div>`;
      el.firstChild.replaceWith(sourceToggle("news", src.id, src.enabled, false, ""));
      nl.appendChild(el);
    });
  }
  const pm = $("#provider-map");
  if (pm) pm.innerHTML = (state.config ? state.config.symbols : []).map(sym => {
    const pid = state.providers[sym];
    const src = state.sources.find(x => x.id === pid);
    const cls = !pid ? "none" : pid === "sim" ? "sim" : "";
    return `<span class="pm ${cls}">${esc(base(sym))} <b>${esc(src ? src.name : pid || "—")}</b></span>`;
  }).join("");
}

const urgencyEMA = {};
function realizedVol(sym) {
  const b = (state.bars[sym] || []).slice(-30);
  if (b.length < 4) return 0;
  const r = [];
  for (let i = 1; i < b.length; i++) r.push(Math.log(b[i].c / b[i - 1].c) * 1e4);
  const mean = r.reduce((a, x) => a + x, 0) / r.length;
  return Math.sqrt(r.reduce((a, x) => a + (x - mean) ** 2, 0) / r.length);  // bps
}
function rawUrgency(sym) {
  const L = state.latest[sym] || {};
  const pa = state.signals && state.signals.per_asset && state.signals.per_asset[sym];
  const f = (pa && pa.feat) || {};
  const traded = (L.side && L.trusted) ? 1 : 0;
  // Primary signal is volatility — the model's expected 10-90 band plus recent realized vol,
  // both slow-moving — so the ranking reflects what is actually active, not tick noise.
  const band = L.q ? (L.q[4] - L.q[0]) : 0;
  const vol = Math.min(1, band / 50) * 0.6 + Math.min(1, realizedVol(sym) / 30) * 0.4;
  return 2.0 * vol + 1.2 * traded + 0.5 * Math.abs(L.crowding || 0) +
    0.35 * (f.wsb_attn || 0) + 0.3 * Math.abs(L.guru_tilt || 0);
}
function cardUrgency(sym) {
  const r = rawUrgency(sym);
  urgencyEMA[sym] = urgencyEMA[sym] == null ? r : 0.9 * urgencyEMA[sym] + 0.1 * r;
  return urgencyEMA[sym];
}

let lastOrder = "", lastReorder = 0;
function reorderCards(force) {
  if (!state.config || document.body.dataset.view !== "dashboard") return;
  const now = Date.now();
  if (!force && now - lastReorder < 1500) return;
  lastReorder = now;
  const HYST = 0.15;
  const u = {};
  state.config.symbols.forEach(s => { if (cards[s]) u[s] = cardUrgency(s); });
  // Start from the current on-screen order and only let a card overtake its neighbour
  // when it clearly leads (hysteresis) — so near-ties don't shuffle every bar.
  const mutedSet = new Set(state.muted || []);
  let order = [...document.querySelectorAll("#cards .card")].map(c => c.dataset.sym).filter(x => cards[x] && !mutedSet.has(x));
  state.config.symbols.forEach(s => { if (cards[s] && !mutedSet.has(s) && !order.includes(s)) order.push(s); });
  for (let pass = 0, changed = true; changed && pass < 40; pass++) {
    changed = false;
    for (let i = 0; i < order.length - 1; i++) {
      if (u[order[i + 1]] > u[order[i]] + HYST) { const t = order[i]; order[i] = order[i + 1]; order[i + 1] = t; changed = true; }
    }
  }
  const key = order.join(",");
  if (key === lastOrder) return;
  const firstRun = lastOrder === "";
  lastOrder = key;
  const root = $("#cards");
  const syms = state.config.symbols.filter(s => cards[s]);
  // FLIP: record current positions, reorder the DOM, then animate each card from
  // its old position to its new one so the swap slides instead of jumping.
  const first = {};
  syms.forEach(s => { first[s] = cards[s].el.getBoundingClientRect(); });
  order.forEach(sym => { if (cards[sym]) root.appendChild(cards[sym].el); });
  if (firstRun) return;
  syms.forEach(s => {
    const el = cards[s].el, a = first[s], b = el.getBoundingClientRect();
    const dx = a.left - b.left, dy = a.top - b.top;
    if (!dx && !dy) return;
    el.style.transition = "none";
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    el.style.zIndex = "2";
  });
  requestAnimationFrame(() => {
    syms.forEach(s => {
      const el = cards[s].el;
      if (!el.style.transform) return;
      el.style.transition = "transform 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)";
      el.style.transform = "";
      el.addEventListener("transitionend", () => { el.style.transition = ""; el.style.zIndex = ""; }, { once: true });
    });
  });
}

function renderWatch() {
  const box = $("#watch-grid");
  if (!box) return;
  const m = state.signals && state.signals.market;
  const rows = (m && m.radar) || [];
  const modeled = new Set((state.config ? state.config.symbols : []).map(base));
  const urg = r => Math.abs(r.chg || 0) + (r.wsb ? 2 : 0) + ((r.gurus || []).length ? 3 : 0) + (modeled.has(r.symbol) ? 6 : 0);
  const filter = ($("#watch-filter") && $("#watch-filter").value || "").trim().toUpperCase();
  let list = rows.slice().sort((a, b) => urg(b) - urg(a));
  if (filter) list = list.filter(r => (r.symbol || "").includes(filter));
  const meta = $("#mw-meta");
  if (meta) meta.textContent = rows.length ? `${list.length} of ${rows.length} names, most urgent first` : "waiting for market scan…";
  box.innerHTML = list.map(r => {
    const mdl = modeled.has(r.symbol);
    const flags = [];
    if (mdl) flags.push(`<span class="mdl">modeled</span>`);
    if (r.wsb) flags.push(`<span class="wsb">WSB ${r.wsb}</span>`);
    (r.gurus || []).forEach(g => flags.push(`<span class="guru">${esc(g)}</span>`));
    return `<div class="wtile ${mdl ? "modeled" : ""}"><div class="wt"><span class="sym">${esc(r.symbol)}</span>` +
      `<span class="chg ${r.chg >= 0 ? "up" : "down"}">${r.chg >= 0 ? "+" : ""}${(r.chg || 0).toFixed(1)}%</span></div>` +
      `<div class="px">${r.price != null ? fmtPrice(r.price) : "·"}</div><div class="wf">${flags.join("")}</div></div>`;
  }).join("");
}

function renderRadar() {
  const box = $("#radar-table");
  if (!box) return;
  const m = state.signals && state.signals.market;
  const rows = (m && m.radar) || [];
  const meta = $("#radar-meta");
  if (meta) meta.textContent = rows.length ? `${rows.length} movers, ranked by |% change|` : "waiting for scan…";
  const filter = ($("#radar-filter") && $("#radar-filter").value || "").trim().toUpperCase();
  const shown = filter ? rows.filter(r => (r.symbol || "").includes(filter)) : rows;
  const fvol = v => v == null ? "·" : v >= 1e9 ? (v / 1e9).toFixed(1) + "B" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "K" : v;
  const head = `<div class="rrow head"><span>symbol</span><span>% chg</span><span>price</span><span>volume</span><span>type</span><span>signals</span></div>`;
  box.innerHTML = head + shown.map(r => {
    const flags = [];
    if (r.wsb) flags.push(`<span class="wsb">WSB ${r.wsb}</span>`);
    (r.gurus || []).forEach(g => flags.push(`<span class="guru">${esc(g)}</span>`));
    return `<div class="rrow"><span class="sym">${esc(r.symbol)}</span>` +
      `<span class="chg ${r.chg >= 0 ? "up" : "down"}">${r.chg >= 0 ? "+" : ""}${(r.chg || 0).toFixed(1)}%</span>` +
      `<span>${r.price != null ? fmtPrice(r.price) : "·"}</span><span>${fvol(r.vol)}</span>` +
      `<span class="cat">${esc(r.cat || "")}</span><span class="flags">${flags.join("") || `<span class="nm">${esc(r.name || "")}</span>`}</span></div>`;
  }).join("");
}

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

function gaugeSVG(frac, big, label, color) {
  frac = clamp01(frac);
  const cx = 60, cy = 56, r = 46;
  const pol = d => [cx + r * Math.cos(d * Math.PI / 180), cy + r * Math.sin(d * Math.PI / 180)];
  const arc = (a, b) => { const [x0, y0] = pol(a), [x1, y1] = pol(b); const large = (b - a) > 180 ? 1 : 0;
    return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`; };
  return `<svg viewBox="0 0 120 74" class="gauge">` +
    `<path d="${arc(180, 360)}" fill="none" stroke="var(--grid)" stroke-width="9" stroke-linecap="round"/>` +
    `<path d="${arc(180, 180 + frac * 180)}" fill="none" stroke="${color}" stroke-width="9" stroke-linecap="round"/>` +
    `<text x="60" y="48" class="g-big">${big}</text><text x="60" y="68" class="g-lab">${esc(label)}</text></svg>`;
}

function rangeBar(q) {
  const scale = Math.max(12, Math.abs(q[0]), Math.abs(q[4])) * 1.15;
  const pos = v => (50 + (v / scale) * 50).toFixed(1);
  return `<div class="brange"><div class="zero"></div>` +
    `<div class="band" style="left:${pos(q[0])}%;right:${(100 - pos(q[4])).toFixed(1)}%"></div>` +
    `<div class="mid" style="left:${pos(q[2])}%"></div></div>`;
}

function briefText(sig, m) {
  const out = [];
  const fmtc = v => v == null ? "·" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  const breadth = m.breadth != null ? Math.round(m.breadth * 100) : null;
  let s1 = `Tone is <b>${esc(m.regime || "—")}</b>`;
  if (breadth != null) s1 += `, with <b>${breadth}%</b> of tracked ETFs higher`;
  if (m.sectors && m.sectors.length) { const a = m.sectors[0], z = m.sectors[m.sectors.length - 1];
    s1 += `; <b>${esc(a.name)}</b> leads (${fmtc(a.chg)}) and <b>${esc(z.name)}</b> lags (${fmtc(z.chg)})`; }
  out.push(s1 + (m.vix ? `. VIX ${m.vix.toFixed(1)}.` : "."));
  if (m.crypto) out.push(`Crypto is <b>$${(m.crypto.total_mcap / 1e12).toFixed(2)}T</b> (${fmtc(m.crypto.chg24)} 24h), BTC dominance <b>${m.crypto.btc_dom.toFixed(0)}%</b>.`);
  // retail + crowding extreme
  const cr = sig.crowding || {};
  const hot = Object.entries(cr).sort((a, b) => b[1] - a[1])[0];
  const cold = Object.entries(cr).sort((a, b) => a[1] - b[1])[0];
  let topWsb = null, topN = 0;
  Object.entries(sig.per_asset || {}).forEach(([s, pa]) => { if ((pa.mentions || 0) > topN) { topN = pa.mentions; topWsb = s; } });
  if (topWsb && topN) out.push(`Retail attention is on <b>${esc(base(topWsb))}</b> (${topN} WSB mentions)` +
    (hot && hot[1] > 0.2 ? `; most crowded name is <b>${esc(base(hot[0]))}</b> (${hot[1].toFixed(2)})` : "") + ".");
  // council lean
  const gt = sig.guru_tilt || {};
  const bear = Object.entries(gt).sort((a, b) => a[1] - b[1])[0];
  const bull = Object.entries(gt).sort((a, b) => b[1] - a[1])[0];
  if (bear && bear[1] < -0.2) out.push(`Smart money leans <b>bearish on ${esc(base(bear[0]))}</b> (${bear[1].toFixed(2)})` +
    (bull && bull[1] > 0.2 ? ` and <b>bullish on ${esc(base(bull[0]))}</b> (+${bull[1].toFixed(2)})` : "") + ".");
  // model
  const m2 = state.metrics || {};
  const calls = state.config.symbols.map(s => state.latest[s]).filter(L => L && L.side && L.trusted);
  if (calls.length) out.push(`The model has <b>${calls.length}</b> live call${calls.length > 1 ? "s" : ""}: ` +
    calls.slice(0, 4).map(L => `${esc(base(L.symbol))} ${L.action}`).join(", ") + ".");
  else out.push(`The model is <b>holding</b> across the board` + (m2.trusted ? "" : ` while it warms up (${m2.live_labels || 0}/${state.config.min_labels} labels)`) + ".");
  return out;
}

function briefColumn() {
  const b = state.brief;
  const busy = b && b.generating;
  const regen = `<button id="brief-regen"${busy ? " disabled" : ""}>${busy ? "writing…" : "\u21bb regenerate"}</button>`;
  const masthead = meta => `<div class="col-masthead"><span class="col-name">flint \u00b7 Markets</span><span class="col-meta">${esc(meta)}</span>${regen}</div>`;
  if (busy && !(b && b.text)) {
    return `<div class="brief-column">${masthead("the local desk is writing\u2026")}<div class="col-loading"><div class="spark-mini"></div>` +
      `<p>Fast models are digesting the tape, macro, positioning and smart money; a bigger model writes the column. This runs entirely on your machine and can take a minute.</p></div></div>`;
  }
  if (!b || (!b.text && !b.error)) return null;                 // fall back to the templated summary
  if (b.error && !b.text) {
    return `<div class="brief-column">${masthead("local brief unavailable")}<div class="col-error">${esc(b.error)}</div>` +
      `<p class="col-hint">The written brief runs on a local model via <b>Ollama</b> \u2014 nothing leaves this machine.</p></div>`;
  }
  const parts = (b.text || "").trim().split(/\n{2,}/);
  const headline = parts.shift() || "Market Brief";
  const body = parts.map(x => `<p>${esc(x).replace(/\n/g, "<br>")}</p>`).join("");
  const models = b.models || {};
  const when = b.t ? new Date(b.t * 1000).toLocaleString() : "";
  const takesArr = Object.entries(b.takes || {}).filter(([, v]) => v);
  const takes = takesArr.length
    ? `<details class="desk-notes"><summary>desk notes${models.small ? " \u00b7 " + esc(models.small) : ""}</summary>` +
      takesArr.map(([k, v]) => `<div class="dnote"><span class="dk">${esc(k.replace("_", " "))}</span><span class="dv">${esc(v)}</span></div>`).join("") +
      `</details>` : "";
  return `<div class="brief-column">${masthead(when + (models.big ? " \u00b7 written locally by " + models.big : ""))}` +
    `<article class="col-body"><h2 class="col-head">${esc(headline)}</h2>${body}</article>${takes}` +
    `${busy ? '<div class="col-refreshing">refreshing\u2026</div>' : ""}</div>`;
}

function renderBrief() {
  const box = $("#brief-body");
  if (!box || !state.config) return;
  const sig = state.signals || {}, m = sig.market || {};
  if (!m.t && !Object.keys(sig).length) { return; }
  const rc = m.regime === "risk-on" ? "on" : m.regime === "risk-off" ? "off" : "mixed";
  const templated = `<div class="brief-hero"><div class="htop"><span class="regime ${rc}">${esc((m.regime || "—").toUpperCase())}</span>` +
    `<span class="hts">market brief · ${m.t ? fmtTime(m.t) : "scanning…"}</span></div>` +
    briefText(sig, m).map(p => `<p>${p}</p>`).join("") + `</div>`;
  const hero = briefColumn() || templated;

  // gauges
  const fg = sig.fear_greed || {};
  const breadth = m.breadth;
  const gauges = [];
  if (breadth != null) gauges.push(gauge_card(gaugeSVG(breadth, Math.round(breadth * 100) + "%", "breadth", breadth > 0.55 ? C.good : breadth < 0.45 ? C.serious : C.warning), `${m.breadth_up || 0}/${m.breadth_n || 0} ETFs up`));
  if (fg.value != null) gauges.push(gauge_card(gaugeSVG(fg.value / 100, fg.value, "fear/greed", fg.value > 60 ? C.serious : fg.value < 40 ? C.blue : C.warning), esc(fg.class || "")));
  if (m.vix) gauges.push(gauge_card(gaugeSVG(clamp01((m.vix - 10) / 40), m.vix.toFixed(1), "VIX", m.vix > 25 ? C.serious : m.vix > 18 ? C.warning : C.good), m.vix > 25 ? "elevated" : m.vix > 18 ? "moderate" : "calm"));
  if (m.crypto) gauges.push(gauge_card(gaugeSVG(clamp01(m.crypto.btc_dom / 100), m.crypto.btc_dom.toFixed(0) + "%", "BTC dominance", C.yellow), `$${(m.crypto.total_mcap / 1e12).toFixed(2)}T total`));
  const gaugeRow = gauges.length ? `<div class="gauges">${gauges.join("")}</div>` : "";

  // movers
  const mv = m.movers || {};
  const moverPanel = (title, rows, up) => `<div class="bpanel"><h3>${title}</h3>` +
    (rows || []).slice(0, 6).map(r => { const w = Math.min(100, Math.abs(r.chg) * 4);
      return `<div class="bmover"><span class="sym">${esc(r.symbol)}</span>` +
      `<div class="bar" style="width:${w}%;background:${up ? C.good : C.serious}"></div>` +
      `<span class="v ${r.chg >= 0 ? "up" : "down"}">${r.chg >= 0 ? "+" : ""}${(r.chg || 0).toFixed(1)}%</span></div>`; }).join("") + `</div>`;

  // model calls (strongest by |score|)
  const calls = state.config.symbols.map(s => state.latest[s]).filter(L => L && L.q)
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 6);
  const callPanel = `<div class="bpanel"><h3>Model calls <small>60s forecast, 10–90 band</small></h3>` +
    (calls.length ? calls.map(L => `<div class="bcall"><span class="sym">${esc(base(L.symbol))}</span>` +
      `<span class="badge ${L.action.toLowerCase()}">${L.action}</span>${rangeBar(L.q)}</div>`).join("")
      : `<div class="why">warming up…</div>`) + `</div>`;

  // positioning & sentiment
  const cr = sig.crowding || {};
  const crowdRows = Object.entries(cr).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4)
    .map(([s, v]) => `<div class="bline"><span>${esc(base(s))} crowding</span><b class="${v >= 0 ? "down" : "up"}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</b></div>`).join("");
  const posPanel = `<div class="bpanel"><h3>Positioning &amp; sentiment</h3>` +
    (fg.value != null ? `<div class="bline"><span>Fear &amp; Greed</span><b>${fg.value} ${esc(fg.class || "")}</b></div>` : "") +
    crowdRows + `</div>`;

  // smart money
  const council = sig.council || {};
  const ethos = Object.entries(council).sort((a, b) => b[1] - a[1]).slice(0, 4)
    .map(([k, v]) => `<span>${esc(k)} <b>${v.toFixed(2)}</b></span>`).join("");
  const gt = sig.guru_tilt || {};
  const gtRows = Object.entries(gt).filter(([, v]) => Math.abs(v) > 0.05).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5)
    .map(([s, v]) => `<div class="bline"><span>${esc(base(s))}</span><b class="${v >= 0 ? "up" : "down"}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</b></div>`).join("");
  const smartPanel = `<div class="bpanel"><h3>Smart money <small>${(sig.gurus || []).filter(g => g.enabled).length} investors</small></h3>` +
    `<div class="ethos-mini">${ethos}</div>` + (gtRows || `<div class="why">loading 13Fs…</div>`) + `</div>`;

  // sectors
  const sectorPanel = m.sectors && m.sectors.length ? `<div class="bpanel"><h3>Sector rotation</h3><div class="sector-heat">` +
    m.sectors.map(sc => { const t = Math.max(-1, Math.min(1, sc.chg / 2));
      const bg = t >= 0 ? `rgba(12,163,12,${0.25 + 0.55 * t})` : `rgba(208,59,59,${0.25 + 0.55 * -t})`;
      return `<div class="sc" style="background:${bg}" title="${esc(sc.name)}"><span>${esc(sc.etf)}</span><b>${sc.chg >= 0 ? "+" : ""}${sc.chg.toFixed(1)}</b></div>`; }).join("") + `</div></div>` : "";

  box.innerHTML = hero + gaugeRow + `<div class="brief-grid">` +
    callPanel + moverPanel("Top gainers", mv.gainers, true) + moverPanel("Top losers", mv.losers, false) +
    sectorPanel + posPanel + smartPanel + `</div>`;
}

function gauge_card(svg, sub) { return `<div class="gcard">${svg}<div class="gsub">${esc(sub)}</div></div>`; }

function renderMarket() {
  const m = state.signals && state.signals.market;
  const meta = $("#market-meta");
  if (!meta) return;
  if (!m || !m.t) { meta.textContent = "scanning…"; return; }
  meta.textContent = "updated " + fmtTime(m.t);
  const pct = v => v == null ? "·" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  const cr = m.crypto || {};
  const stats = [
    `<div class="mstat regime ${m.regime === "risk-on" ? "on" : m.regime === "risk-off" ? "off" : "mixed"}">regime <b>${esc(m.regime || "·")}</b></div>`,
    `<div class="mstat">breadth <b>${m.breadth != null ? Math.round(m.breadth * 100) + "%" : "·"}</b> ${m.breadth_up || 0}/${m.breadth_n || 0} up</div>`,
    m.vix ? `<div class="mstat">VIX <b>${m.vix.toFixed(1)}</b></div>` : "",
    cr.total_mcap ? `<div class="mstat">crypto <b>$${(cr.total_mcap / 1e12).toFixed(2)}T</b> <span class="${cr.chg24 >= 0 ? "up" : "down"}">${pct(cr.chg24)}</span></div>` : "",
    cr.btc_dom ? `<div class="mstat">BTC dom <b>${cr.btc_dom.toFixed(1)}%</b></div>` : "",
  ];
  $("#market-stats").innerHTML = stats.join("");
  const heatColor = c => { const t = Math.max(-1, Math.min(1, c / 2)); return t >= 0 ? `rgba(12,163,12,${0.25 + 0.55 * t})` : `rgba(208,59,59,${0.25 + 0.55 * -t})`; };
  $("#sector-heat").innerHTML = (m.sectors || []).map(sc =>
    `<div class="sc" style="background:${heatColor(sc.chg)}" title="${esc(sc.name)}"><span>${esc(sc.etf)}</span><b>${sc.chg >= 0 ? "+" : ""}${sc.chg.toFixed(1)}</b></div>`).join("");
  const mv = m.movers || {};
  const col = (title, rows) => `<div class="mov"><h5>${title}</h5>` +
    (rows || []).slice(0, 6).map(r => `<div class="row"><span class="t">${esc(r.symbol || "")}</span><span class="c ${r.chg >= 0 ? "up" : "down"}">${r.chg >= 0 ? "+" : ""}${(r.chg || 0).toFixed(1)}%</span></div>`).join("") + "</div>";
  $("#market-movers").innerHTML = col("Most active", mv.actives) + col("Gainers", mv.gainers) + col("Losers", mv.losers);
}

function renderBriefCtl() {
  const wrap = $("#brief-toggle-wrap");
  if (wrap && !wrap.querySelector(".switch")) {
    const sw = document.createElement("label"); sw.className = "switch";
    sw.innerHTML = `<input type="checkbox"><span class="track"></span><span class="thumb"></span>`;
    sw.querySelector("input").onchange = e => control({ action: "toggle_brief", on: e.target.checked });
    wrap.appendChild(sw);
  }
  if (wrap) { const inp = wrap.querySelector("input"); if (inp) inp.checked = state.controls.brief_enabled !== false; }
  const el = $("#brief-status"); if (!el) return;
  const b = state.brief || {}, models = b.models || {};
  let status;
  if (b.generating) status = `<span class="bstat-run">writing\u2026</span> local analysts + writer running`;
  else if (b.error) status = `<span class="bstat-err">${esc(b.error)}</span>`;
  else if (b.text) status = `last written ${b.t ? fmtTime(b.t) : ""} \u00b7 <b>${esc(models.big || "?")}</b> from <b>${esc(models.small || "?")}</b> analysts`;
  else status = `no brief yet \u2014 click \u201cWrite now\u201d`;
  el.innerHTML = status;
}

function renderSignals() {
  const list = $("#signal-list");
  if (!list) return;
  list.innerHTML = "";
  state.signal_providers.forEach(src => {
    const el = document.createElement("div");
    el.className = "source" + (src.enabled ? "" : " off");
    el.innerHTML = `<div></div><div><div class="sname">${esc(src.name)}</div>` +
      (src.status ? `<div class="snote">${esc(src.status)}</div>` : "") + `</div>`;
    el.firstChild.replaceWith(sourceToggle("signal", src.id, src.enabled, false, ""));
    list.appendChild(el);
  });
  // Burry overlay master toggle
  const wrap = $("#burry-toggle-wrap");
  if (wrap && !wrap.querySelector(".switch")) {
    const sw = document.createElement("label");
    sw.className = "switch";
    sw.innerHTML = `<input type="checkbox"><span class="track"></span><span class="thumb"></span>`;
    sw.querySelector("input").onchange = e => control({ action: "burry", on: e.target.checked });
    wrap.appendChild(sw);
  }
  if (wrap) { const inp = wrap.querySelector("input"); if (inp) inp.checked = !!(state.burry && state.burry.enabled); }

  const sig = state.signals;
  const assets = $("#signal-assets");
  if (sig && sig.per_asset) {
    const mk = sig.market || {};
    assets.innerHTML = state.config.symbols.map(sym => {
      const pa = sig.per_asset[sym] || {}; const f = pa.feat || {};
      const crowd = (sig.crowding || {})[sym] || 0, guru = (sig.guru_tilt || {})[sym] || 0;
      const sgn = v => (v >= 0 ? "pos" : "neg");
      const fmt = v => v == null ? "·" : (v >= 0 ? "+" : "") + Number(v).toFixed(2);
      return `<div class="sig-asset">
        <div class="sh"><b>${esc(base(sym))}</b><span class="crowd ${crowd > 0.15 ? "hot" : crowd < -0.15 ? "cold" : ""}">crowding ${fmt(crowd)}</span></div>
        <div class="rows">
          <span>WSB <b>${pa.mentions || 0}</b> mentions</span><span>WSB tone <b class="${sgn(f.wsb_sent)}">${fmt(f.wsb_sent)}</b></span>
          <span>guru net <b class="${sgn(f.guru_net)}">${fmt(f.guru_net)}</b></span><span>ethos bias <b class="${sgn(f.ethos_bias)}">${fmt(f.ethos_bias)}</b></span>
          ${pa.fundamentals && pa.fundamentals.pe != null ? `<span>value <b class="${sgn(f.f_value)}">${fmt(f.f_value)}</b></span><span>quality <b class="${sgn(f.f_quality)}">${fmt(f.f_quality)}</b></span>` : ""}
        </div>${pa.fundamentals && pa.fundamentals.pe != null ? `<div class="frow">P/E ${pa.fundamentals.pe.toFixed(1)} · EPS ${pa.fundamentals.eps != null ? pa.fundamentals.eps.toFixed(2) : "·"} · ROE ${pa.fundamentals.roe != null ? pa.fundamentals.roe.toFixed(0) + "%" : "·"} · β ${pa.fundamentals.beta != null ? pa.fundamentals.beta.toFixed(2) : "·"}</div>` : ""}${guruHolders(pa)}</div>`;
    }).join("");
  } else if (assets) assets.innerHTML = `<div class="why">signals not gathered yet</div>`;

  // council ethos summary
  const ce = $("#council-ethos");
  if (ce) ce.innerHTML = Object.entries((sig && sig.council) || {}).map(([k, v]) =>
    `<span class="ed">${esc(k)} <b>${v >= 0 ? "+" : ""}${v.toFixed(2)}</b></span>`).join("");
  // dynamic guru panels
  const glist = $("#guru-list");
  if (glist) {
    const gurus = (sig && sig.gurus) || [];
    glist.innerHTML = gurus.length ? gurus.map(g => guruPanel(g)).join("") : `<div class="why">council 13Fs not loaded yet</div>`;
  }
}

function guruHolders(pa) {
  const g = pa && pa.gurus;
  if (!g || !Object.keys(g).length) return "";
  const names = {};
  ((state.signals && state.signals.gurus) || []).forEach(x => names[x.id] = x.name.split(" / ")[0]);
  const items = Object.entries(g).filter(([, v]) => Math.abs(v) > 0.02)
    .map(([id, v]) => `<span class="${v >= 0 ? "pos" : "neg"}">${esc(names[id] || id)} ${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`);
  return items.length ? `<div class="holders">${items.join(" ")}</div>` : "";
}

function guruPanel(g) {
  const max = Math.max(...(g.holdings || []).map(h => Math.abs(h.weight)), 0.01);
  const chips = Object.entries(g.ethos || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4)
    .map(([k, v]) => `<span class="${Math.abs(v) >= 0.7 ? "hi" : ""}">${esc(k)} ${v >= 0 ? "+" : ""}${v.toFixed(1)}</span>`).join("");
  const rows = (g.holdings || []).slice(0, 7).map(h => {
    const w = h.weight, put = h.put;
    return `<div class="hrow"><span class="w">${w < 0 ? "−" : ""}${(Math.abs(w) * 100).toFixed(0)}%</span>` +
      `<span class="${put ? "put" : "long"}">${put ? "PUT" : "LONG"}</span>` +
      `<span class="nm" title="${esc(h.issuer || "")}">${esc(h.ticker || h.issuer || "")}</span>` +
      `<div class="bar"><i style="width:${(Math.abs(w) / max * 100).toFixed(0)}%;background:${put ? "var(--critical)" : "var(--good)"}"></i></div></div>`;
  }).join("");
  return `<div class="guru ${g.enabled ? "" : "disabled"}">
    <h4>${esc(g.name)} <small>13F ${esc(g.asof || "—")}</small></h4>
    <div class="echips">${chips}</div>
    <div class="ebasis">${esc(g.basis || "")}</div>
    ${rows || '<div class="why">no holdings</div>'}</div>`;
}

function renderGuru(id, g, title) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!g || !g.holdings || !g.holdings.length) { el.innerHTML = `<h4>${esc(title)}</h4><div class="why">no 13F loaded</div>`; return; }
  const max = Math.max(...g.holdings.map(h => Math.abs(h.weight))) || 1;
  el.innerHTML = `<h4>${esc(title)} <small>13F as of ${esc(g.asof || "?")}</small></h4>` +
    g.holdings.slice(0, 8).map(h => {
      const w = h.weight, pct = (Math.abs(w) * 100).toFixed(0);
      const put = h.put;
      return `<div class="hrow">
        <span class="w">${w < 0 ? "−" : ""}${pct}%</span>
        <span class="${put ? "put" : "long"}">${put ? "PUT" : "LONG"}</span>
        <span class="nm" title="${esc(h.issuer || "")}">${esc(h.ticker || h.issuer || "")}</span>
        <div class="bar"><i style="width:${(Math.abs(w) / max * 100).toFixed(0)}%;background:${put ? "var(--critical)" : "var(--good)"}"></i></div>
      </div>`;
    }).join("");
}

function renderNews() {
  const n = state.news;
  if (!n || !n.t) { $("#news-meta").textContent = "not run yet"; return; }
  $("#news-meta").textContent = `${fmtTime(n.t)} via ${n.method}, ${n.headlines.length} relevant headlines, ${n.new} new`;
  $("#tone").innerHTML = Object.entries(n.per_asset || {}).map(([s, a]) => {
    const v = a.sentiment, w = Math.abs(v) * 50, left = v >= 0 ? 50 : 50 - w;
    return `<div class="a"><span>${esc(base(s))}</span><div class="bar" title="tone ${v.toFixed(2)}, attention ${a.attention.toFixed(2)}"><i style="left:${left}%;width:${w}%;background:${v >= 0 ? C.good : C.critical}"></i></div><span>${a.mentions} hl</span></div>`;
  }).join("") + `<div class="why">bar: tone from -1 to +1; hl: headlines naming the asset</div>` +
    (n.ideas || []).map(i => `<div class="why">${esc(i)}</div>`).join("");
  $("#headlines").innerHTML = n.headlines.slice(0, 80).map(h =>
    `<li class="${h.new ? "new" : ""}"><span class="s ${h.sentiment > 0 ? "pos" : h.sentiment < 0 ? "neg" : ""}">${h.sentiment >= 0 ? "+" : ""}${h.sentiment.toFixed(2)}</span>` +
    (h.assets.length ? h.assets : ["mkt"]).map(a => `<span class="tag">${esc(a)}</span>`).join("") +
    `<a href="${esc(safeUrl(h.url))}" target="_blank" rel="noopener noreferrer">${esc(h.title)}</a><span class="src">${esc(h.source)}</span></li>`).join("");
}

// Transport ------------------------------------------------------------------------

function handle(msg) {
  switch (msg.type) {
    case "snapshot":
      Object.assign(state, { config: msg.config, status: msg.status, controls: msg.controls, prices: msg.prices, bars: msg.bars,
        latest: msg.latest, gate: msg.gate, outcomes: msg.outcomes, metrics: msg.metrics, history: msg.history, log: msg.log, news: msg.news, bars_fine: msg.bars_fine || {},
        sources: msg.sources || [], news_sources: msg.news_sources || [], providers: (msg.status && msg.status.providers) || {}, classes: msg.classes || {},
        signals: msg.signals || null, signal_providers: msg.signal_providers || [], brief: msg.brief || null, burry: msg.burry || { enabled: true }, keys: msg.keys || [], muted: msg.muted || [], universe: msg.universe || (msg.config ? msg.config.symbols : []) });
      buildCards(); buildConsoles(); buildControls(); applyMuted();
      Object.values(msg.trace || {}).forEach(evs => evs.forEach(ev => appendTrace(ev, false)));
      Object.values(consoles).forEach(c => { c.body.scrollTop = c.body.scrollHeight; });
      Object.values(msg.trace || {}).flat().sort((a, b) => (a.seq || a.t) - (b.seq || b.t)).forEach(appendTerm);
      { const t = $("#term"); if (t) t.scrollTop = t.scrollHeight; }
      renderAll();
      onboardStart();
      break;
    case "tick":
      state.prices = msg.prices; state.status = msg.status; state.providers = msg.status.providers || state.providers; renderStatus();
      if (document.body.dataset.view === "dashboard") { state.config.symbols.forEach(updatePrice); scheduleDraw(); }
      break;
    case "bar":
      state.status = msg.status;
      Object.entries(msg.bars).forEach(([s, b]) => { (state.bars[s] ||= []).push(b); if (state.bars[s].length > 240) state.bars[s].shift(); });
      state.latest = msg.latest; state.gate = msg.gate; state.metrics = msg.metrics; state.outcomes = msg.outcomes; state.log = msg.log;
      state.providers = (msg.status && msg.status.providers) || state.providers;
      absorbHistory(msg.history);
      renderAll();
      checkAlerts();
      break;
    case "metrics":
      state.metrics = msg.metrics; absorbHistory(msg.history);
      updateLoading();
      if (document.body.dataset.view === "dashboard") { renderTiles(); renderSparks(); }
      break;
    case "sources": state.sources = msg.sources; if (msg.status) { state.status = msg.status; state.providers = msg.status.providers || state.providers; } renderSources(); renderStatus(); if (document.body.dataset.view === "dashboard") state.config.symbols.forEach(updateVia); break;
    case "news_sources": state.news_sources = msg.news_sources; renderSources(); break;
    case "signals": state.signals = msg.signals; if (msg.signal_providers) state.signal_providers = msg.signal_providers; renderSignals(); renderMarket(); renderRadar(); renderWatch(); renderBrief(); if (document.body.dataset.view === "dashboard") { state.config.symbols.forEach(updateCard); reorderCards(true); } break;
    case "signal_providers": state.signal_providers = msg.signal_providers; renderSignals(); break;
    case "keys": state.keys = msg.keys; renderKeys(); break;
    case "muted": state.muted = msg.muted; applyMuted(); renderUniverse(); break;
    case "trace": appendTrace(msg.ev); appendTerm(msg.ev); break;
    case "brief": state.brief = msg.brief; if (document.body.dataset.view === "brief") renderBrief(); if (document.body.dataset.view === "consoles") renderBriefCtl(); break;
    case "fine": {
      const cap = (state.config && state.config.fine_bars) || 240;
      Object.entries(msg.bars).forEach(([sy, b]) => { (state.bars_fine[sy] ||= []).push(b); while (state.bars_fine[sy].length > cap) state.bars_fine[sy].shift(); });
      if (document.body.dataset.view === "dashboard") Object.values(cards).forEach(c => { if (c.res === "1m") drawChart(c); });
      break;
    }
    case "news": state.news = msg.news; renderNews(); break;
    case "status": state.status = msg.status; state.providers = msg.status.providers || state.providers; renderStatus(); if (document.body.dataset.view === "dashboard") state.config.symbols.forEach(updateVia); break;
    case "controls": state.controls = msg.controls; state.metrics = msg.metrics; if (msg.sources) state.sources = msg.sources; if (msg.news_sources) state.news_sources = msg.news_sources; if (msg.signal_providers) state.signal_providers = msg.signal_providers; if (msg.controls && "burry" in msg.controls) state.burry.enabled = msg.controls.burry; syncControls(); renderSources(); renderSignals(); if (document.body.dataset.view === "dashboard") renderTiles(); break;
  }
}

function absorbHistory(h) {
  Object.entries(h || {}).forEach(([k, pts]) => {
    const arr = state.history[k] ||= [];
    pts.forEach(p => { arr.push(p); if (arr.length > 240) arr.shift(); });
  });
}

function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { retry = 1000; $("#conn").classList.add("ok"); };
  ws.onclose = () => { $("#conn").classList.remove("ok"); updateLoading(true); setTimeout(connect, retry); retry = Math.min(retry * 2, 15000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => handle(JSON.parse(e.data));
}

$$(".tabs button").forEach(b => b.onclick = () => {
  $$(".tabs button").forEach(x => x.classList.toggle("active", x === b));
  document.body.dataset.view = b.dataset.tab;
  try { localStorage.setItem("flint.view", b.dataset.tab); } catch (e) { /* storage may be unavailable */ }
  if (b.dataset.tab === "consoles") Object.values(consoles).forEach(c => { c.body.scrollTop = c.body.scrollHeight; });
  if (b.dataset.tab === "console") setTermFollow(true);
  renderAll();
  if (b.dataset.tab === "consoles") setTimeout(() => { const a = document.activeElement; if (a && a.tagName === "INPUT") a.blur(); }, 0);
});
try {
  const v = localStorage.getItem("flint.view");
  if (v === "consoles") $$(".tabs button").find(b => b.dataset.tab === v).click();
} catch (e) { /* ignore */ }

document.addEventListener("click", e => { if (e.target.closest && (e.target.closest("#brief-regen") || e.target.closest("#brief-now"))) control({ action: "brief" }); });
{ const c = $("#onb-close"); if (c) c.onclick = onboardClose; }
{ const r = $("#run-setup"); if (r) r.onclick = () => onboardStart(true); }
window.addEventListener("resize", () => renderAll());
setInterval(() => {
  if (state.status && state.status.started) $("#uptime").textContent = fmtDur(Date.now() / 1000 - state.status.started);
  const now = Date.now() / 1000;
  Object.values(consoles).forEach(c => { while (c.times.length && c.times[0] < now - 60) c.times.shift(); c.rate.textContent = `${c.times.length}/min`; });
}, 1000);
document.addEventListener("input", e => { if (e.target && e.target.id === "radar-filter") renderRadar(); if (e.target && e.target.id === "watch-filter") renderWatch(); });
updateLoading();
const soundBtn = $("#sound-toggle");
if (soundBtn) {
  const paint = () => { soundBtn.textContent = soundOn ? "🔔" : "🔕"; soundBtn.classList.toggle("live", soundOn); };
  paint();
  soundBtn.onclick = () => {
    soundOn = !soundOn;
    try { localStorage.setItem("flint.sound", soundOn ? "1" : "0"); } catch (e) { /* ignore */ }
    if (soundOn) { try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); if (audioCtx.state === "suspended") audioCtx.resume(); beep(880, 0.1); } catch (e) { /* ignore */ } }
    paint();
  };
}
connect();
})();
