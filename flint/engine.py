"""The engine: feed -> bars -> features -> model -> suggestions, plus everything the UI sees."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re

import httpx
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .bars import Bar, BarBuilder
from .features import FEATURES, N_FEATURES, FeatureBuilder
from .feed import Tick
from .learner import OnlineLearner, Prediction
from .news import NewsHub
from .schwab import SchwabAuth
from .signals import SignalHub
from .sources import AlphaVantage, SourceManager, asset_class
from .trace import Trace

log = logging.getLogger(__name__)

TUNABLE = {"score_threshold": float, "prob_margin": float, "cost_bps": float, "max_size": float,
           "lr": float, "steps_per_label": int, "min_labels": int, "news_minutes": float,
           "burry_aggr": float, "burry_fade_at": float, "burry_safety": float, "signals_minutes": float}


def clean(o):
    """Make a value JSON safe: numpy types to Python, NaN and inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, deque)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ema(prev: float | None, value: float, a: float) -> float:
    return value if prev is None else (1 - a) * prev + a * value


KEY_SERVICES = [
    {"id": "alphavantage", "name": "Alpha Vantage", "file": "keys", "kind": "token",
     "url": "https://www.alphavantage.co/support/#api-key",
     "note": "Free tier: news sentiment + delayed equity quotes (25 requests/day).",
     "fields": [{"id": "key", "label": "API key"}]},
    {"id": "finnhub", "name": "Finnhub", "file": "finnhub.json", "kind": "json",
     "url": "https://finnhub.io/register",
     "note": "Free real-time US equity quotes and websocket.",
     "fields": [{"id": "key", "label": "API key"}]},
    {"id": "eodhd", "name": "EODHD", "file": "eodhd.json", "kind": "json",
     "url": "https://eodhd.com/register",
     "note": "Delayed US equity quotes.",
     "fields": [{"id": "key", "label": "API token"}]},
    {"id": "fmp", "name": "Financial Modeling Prep", "file": "fmp.json", "kind": "json",
     "url": "https://site.financialmodelingprep.com/developer/docs",
     "note": "Primary equity feed: reliable 5-min history + quotes (Starter plan).",
     "fields": [{"id": "key", "label": "API key"}]},
    {"id": "schwab", "name": "Charles Schwab", "file": "schwab.json", "kind": "json_multi",
     "url": "https://developer.schwab.com",
     "note": "Market data via OAuth. After saving, run  flint schwab-auth  to log in.",
     "fields": [{"id": "app_key", "label": "App key"}, {"id": "app_secret", "label": "App secret"}]},
]


def _mask(v: str) -> str:
    v = v or ""
    return (v[:2] + "***" + v[-2:]) if len(v) >= 6 else ("***" if v else "")


@dataclass(slots=True)
class Pending:
    index: int
    ts: float
    x: np.ndarray
    closes: np.ndarray
    pred: Prediction | None
    suggestions: dict[str, dict]


@dataclass
class Metrics:
    labels: int = 0
    steps: int = 0
    loss: float | None = None
    pinball: float | None = None
    bce: float | None = None
    decisions: int = 0
    hits: int = 0
    hit_ema: float | None = None
    coverage_n: int = 0
    covered: int = 0
    coverage_ema: float | None = None
    sharpness_ema: float | None = None
    suggestions: int = 0
    suggestion_wins: int = 0
    pnl_bps: float = 0.0
    pnl_by_symbol: dict = field(default_factory=dict)
    trusted: bool = False
    live_labels: int = 0
    live_pinball: float | None = None
    coverage_raw_n: int = 0
    covered_raw: int = 0
    band_scale: float = 2.0   # conformal multiplier on the predicted band widths; starts wide, earns its way down
    p_scale: float = 0.3      # temperature on the direction logit (1 = raw); starts sceptical

    def as_dict(self) -> dict:
        return {
            "labels": self.labels, "steps": self.steps, "loss": self.loss, "pinball": self.pinball, "bce": self.bce,
            "decisions": self.decisions, "hit_rate": self.hits / self.decisions if self.decisions else None,
            "hit_ema": self.hit_ema, "coverage": self.covered / self.coverage_n if self.coverage_n else None,
            "coverage_ema": self.coverage_ema, "sharpness": self.sharpness_ema,
            "suggestions": self.suggestions, "win_rate": self.suggestion_wins / self.suggestions if self.suggestions else None,
            "pnl_bps": self.pnl_bps, "pnl_by_symbol": dict(self.pnl_by_symbol), "trusted": self.trusted,
            "live_labels": self.live_labels, "live_pinball": self.live_pinball,
            "coverage_raw": self.covered_raw / self.coverage_raw_n if self.coverage_raw_n else None,
            "band_scale": self.band_scale, "p_scale": self.p_scale,
        }


class Engine:
    def __init__(self, cfg):
        import os
        self.cfg = cfg
        self.all_symbols = [s.upper() for s in cfg.symbols]     # full configured universe
        self.muted = {x.strip().upper() for x in cfg.muted_symbols.split(",") if x.strip()} & set(self.all_symbols)
        self.subs: set[asyncio.Queue] = set()
        self.trace = Trace(self._publish)
        self.av = AlphaVantage(cfg.av_key, cfg.av_rate_seconds)
        ak, asec, acb = cfg.schwab_creds
        token_file = cfg.schwab_token_file or os.path.join(cfg.state_dir, "schwab_tokens.json")
        self.schwab_auth = SchwabAuth(ak, asec, acb, token_file)
        self.log: deque[dict] = deque(maxlen=cfg.log_size)
        self.operator: dict[str, dict] = {}                 # per-symbol human bias {sent, attn, t, text}
        self.operator_notes: deque[dict] = deque(maxlen=50)  # raw injected notes, for display
        self.status = "starting"
        self.started = time.time()
        self.learning_enabled = True
        self.burry_enabled = cfg.burry_enabled
        self.brief_enabled = cfg.brief_enabled
        self.news: dict = {"t": None, "method": None, "status": {}, "headlines": [], "per_asset": {}, "ideas": [], "new": 0}
        self.market_status: dict = {"isOpen": None, "session": None, "holiday": None, "t": None}  # from Finnhub, holiday-aware
        self.brief: dict = {"text": None, "takes": {}, "models": {}, "t": None, "generating": False, "error": None}
        self.autotune_info = None
        self._skimming = False
        self._gathering = False
        self._reconfiguring = False
        self._tasks: list[asyncio.Task] = []
        self._lifecycle: asyncio.Task | None = None
        self._build(self._active())

    def _active(self) -> list[str]:
        a = [s for s in self.all_symbols if s not in self.muted]
        return a or self.all_symbols[:1]

    def _build(self, symbols: list[str]) -> None:
        """(Re)build all per-symbol state for the active universe. Muted symbols are left
        out entirely, so they cost no model compute and no data/API calls."""
        cfg = self.cfg
        cfg.symbols = list(symbols)                            # active set drives model + feeds
        self.symbols = list(symbols)
        self.base = {s: s.split("-")[0] for s in self.symbols}
        if cfg.auto_size:
            from .autotune import autotune
            self.autotune_info = autotune(cfg, N_FEATURES, say=lambda m: print("  [tune]", m, flush=True))
            ai = self.autotune_info
            cfg.device, cfg.torch_threads = ai["device"], ai["threads"]
            cfg.d_model, cfg.dilations = ai["d_model"], tuple(ai["dilations"])
            cfg.n_experts, cfg.n_heads, cfg.window = ai["n_experts"], ai["n_heads"], ai["window"]
        else:
            from .autotune import pick_device
            cfg.device = pick_device(cfg.device)
        self.builder = BarBuilder(self.symbols, cfg.bar_seconds)
        self.features = FeatureBuilder(self.symbols, cfg.window)
        self.news_base = {s: (0.0, 0.0) for s in self.symbols}  # last pure news skim, before operator blend
        self.learner = OnlineLearner(cfg, N_FEATURES)
        self.sources = SourceManager(cfg, self.symbols, self.av, on_tick=self._on_live_tick,
                                     on_provider_change=self._on_provider_change, trace=self.trace,
                                     schwab=self.schwab_auth)
        self.news_hub = NewsHub(cfg, self.symbols, self.av)
        self.signals = SignalHub(cfg, self.symbols)
        self.prices = {s: {"price": None, "bid": None, "ask": None, "ts": None} for s in self.symbols}
        self.tick_counts = {s: 0 for s in self.symbols}
        self.bars: dict[str, deque[Bar]] = {s: deque(maxlen=cfg.chart_bars) for s in self.symbols}
        self.fine_builder = BarBuilder(self.symbols, cfg.fine_seconds)                       # display-only high-res
        self.fine_bars: dict[str, deque[Bar]] = {s: deque(maxlen=cfg.fine_bars) for s in self.symbols}
        self.bar_index = 0
        self.pending: deque[Pending] = deque()
        self.latest: dict[str, dict] = {}
        self.gate: list[float] = []
        self.outcomes = {s: deque(maxlen=40) for s in self.symbols}
        self.history = {k: deque(maxlen=240) for k in ("loss", "hit", "coverage", "pnl")}
        self._history_new: dict[str, list] = {k: [] for k in self.history}
        self.metrics = Metrics(pnl_by_symbol={s: 0.0 for s in self.symbols})
        self.signals_state = self.signals.state
        self.crowding: dict[str, float] = {}
        self.guru_tilt: dict[str, float] = {}
        self.ethos_bias: dict[str, float] = {}
        self.council: dict[str, float] = {}

    def start(self) -> None:
        self._lifecycle = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._lifecycle:
            self._lifecycle.cancel()
        for t in self._tasks:
            t.cancel()
        await self.sources.stop()
        await asyncio.to_thread(self._save)

    async def reconfigure(self) -> None:
        """Rebuild for the current active universe (all_symbols minus muted) and restart
        the lifecycle. Muted / removed symbols then cost no compute or API calls."""
        if self._reconfiguring:
            return
        self._reconfiguring = True
        try:
            self.muted &= set(self.all_symbols)
            for t in self._tasks:
                t.cancel()
            if self._lifecycle:
                self._lifecycle.cancel()
            try:
                await self.sources.stop()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.15)
            self._tasks = []
            self.status = "starting"
            await asyncio.to_thread(self._build, self._active())   # benchmark/build off the event loop
            self.trace.emit("system", f"universe rebuilt: {len(self.symbols)} active "
                                      f"({len(self.muted)} muted) — {', '.join(self.base[s] for s in self.symbols)}", "act")
            self._publish(self.snapshot())                         # UI rebuilds cards for the new set
            self._lifecycle = asyncio.create_task(self.run())
        finally:
            self._reconfiguring = False

    # Pub/sub ------------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.discard(q)

    def _publish(self, msg: dict) -> None:
        if not self.subs:
            return
        text = json.dumps(clean(msg), separators=(",", ":"))
        for q in list(self.subs):
            try:
                q.put_nowait(text)
            except asyncio.QueueFull:
                pass

    def _hpush(self, key: str, value) -> None:
        self.history[key].append(value)
        self._history_new[key].append(value)

    def _drain_history(self) -> dict[str, list]:
        out = {k: v for k, v in self._history_new.items() if v}
        self._history_new = {k: [] for k in self.history}
        return out

    def _log(self, text: str, kind: str = "info") -> None:
        self.log.append({"t": time.time(), "kind": kind, "text": text})

    _BULL = {"buy", "bull", "bullish", "long", "longs", "breakout", "moon", "squeeze", "rip", "rally",
             "pump", "strong", "beat", "beats", "upgrade", "surge", "calls", "bounce", "green", "add"}
    _BEAR = {"sell", "bear", "bearish", "short", "shorts", "crash", "dump", "puke", "fade", "tank",
             "weak", "miss", "misses", "downgrade", "drop", "puts", "risk", "red", "trim", "avoid"}

    def _op_effective(self, sym: str):
        """Current decayed operator bias for a symbol, or None once it has faded out."""
        op = self.operator.get(sym)
        if not op:
            return None
        decay = 0.5 ** ((time.time() - op["t"]) / max(60.0, self.cfg.operator_half_life))
        if decay < 0.05:
            self.operator.pop(sym, None)
            return None
        return op["sent"], op["attn"] * decay

    def _blend_news(self, sym: str, base_sent: float, base_attn: float):
        """Blend a human note over the news skim: attention takes the max, sentiment is
        pulled toward the note in proportion to its (decayed) attention."""
        eff = self._op_effective(sym)
        if not eff:
            return base_sent, base_attn
        osent, oattn = eff
        return base_sent * (1 - oattn) + osent * oattn, max(base_attn, oattn)

    def inject_note(self, text: str) -> dict:
        """Human input from the dashboard: log it to the operator console and, for any
        watched ticker it names, nudge that symbol's news sentiment/attention feature."""
        text = (text or "").strip()
        if not text:
            return {"ok": False, "reason": "empty"}
        self.operator_notes.append({"t": time.time(), "text": text})
        self.trace.emit("operator", text, "act")
        self._log("note: " + text, "operator")

        words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
        hits = [w for w in words if w in self._BULL or w in self._BEAR]
        score = sum(1 for w in words if w in self._BULL) - sum(1 for w in words if w in self._BEAR)
        sent = max(-1.0, min(1.0, score / 2.0)) if hits else 0.0

        bases = {self.base[s]: s for s in self.symbols}       # NVDA->NVDA, XAU->XAU-USD
        mentioned, seen = [], set()
        for dollar, tok in re.findall(r"(\$?)([A-Za-z][A-Za-z.\-]{0,6})", text):
            u = tok.upper().strip(".-")
            if u in bases and bases[u] not in seen and (dollar or tok.isupper() or len(tok) >= 3):
                mentioned.append(bases[u]); seen.add(bases[u])

        now, applied = time.time(), []
        for sym in mentioned:
            self.operator[sym] = {"sent": sent, "attn": 0.85, "t": now, "text": text}
            es, ea = self._blend_news(sym, *self.news_base.get(sym, (0.0, 0.0)))
            self.features.set_news(sym, es, ea)
            applied.append(sym)
        if applied:
            self.trace.emit("features", f"operator bias on {', '.join(applied)}: sentiment {sent:+.2f}, "
                            f"attention 85%" + (f" ({', '.join(sorted(set(hits)))})" if hits else " (attention only)"), "act")
        elif not mentioned:
            self.trace.emit("system", "note logged as context (named no watched ticker)", "info")
        self._publish({"type": "operator", "operator": self.operator_state()})
        return {"ok": True, "applied": applied, "sentiment": round(sent, 3)}

    def operator_state(self) -> dict:
        return {"notes": list(self.operator_notes)[-20:],
                "bias": {s: {"sent": o["sent"], "text": o["text"], "t": o["t"]} for s, o in self.operator.items()}}

    def status_dict(self) -> dict:
        providers = self.sources.provider_map()
        active = sorted({p for p in providers.values() if p})
        return {"phase": self.status, "feed": ", ".join(active) or "none", "providers": providers,
                "started": self.started, "now": time.time(), "bar_index": self.bar_index,
                "pending": len(self.pending), "learning": self.learning_enabled, "subscribers": len(self.subs),
                "market_status": self.market_status}

    def controls(self) -> dict:
        return {k: getattr(self.cfg, k) for k in TUNABLE} | {"learning": self.learning_enabled, "burry": self.burry_enabled, "brief_enabled": self.brief_enabled}

    def _key_values(self) -> dict:
        cfg = self.cfg
        ak, asec, acb = cfg.schwab_creds
        return {"alphavantage": {"key": cfg.av_key}, "finnhub": {"key": cfg.finnhub_key},
                "eodhd": {"key": cfg.eodhd_key}, "fmp": {"key": cfg.fmp_key}, "schwab": {"app_key": ak, "app_secret": asec}}

    def keys_status(self) -> list:
        vals = self._key_values()
        out = []
        for svc in KEY_SERVICES:
            v = vals.get(svc["id"], {})
            fields = [{"id": f["id"], "label": f["label"], "present": bool(v.get(f["id"])),
                       "masked": _mask(v.get(f["id"], ""))} for f in svc["fields"]]
            out.append({"id": svc["id"], "name": svc["name"], "url": svc["url"], "note": svc["note"],
                        "present": all(f["present"] for f in fields), "fields": fields})
        return out

    def set_key(self, service: str, values: dict) -> dict:
        import json as _json
        svc = next((x for x in KEY_SERVICES if x["id"] == service), None)
        if not svc:
            return {"error": "unknown service"}
        vals = {f["id"]: str(values.get(f["id"], "")).strip() for f in svc["fields"]}
        if not any(vals.values()):
            return {"error": "no value provided"}
        path = pathlib.Path(svc["file"])
        try:
            if svc["kind"] == "token":
                path.write_text(vals["key"] + "\n")
            elif svc["kind"] == "json":
                path.write_text(_json.dumps({"key": vals["key"]}))
            else:  # json_multi (schwab)
                existing = {}
                try:
                    existing = _json.loads(path.read_text())
                except (OSError, ValueError):
                    pass
                existing.update({k: v for k, v in vals.items() if v})
                existing.setdefault("callback", "https://127.0.0.1")
                path.write_text(_json.dumps(existing))
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except OSError as e:
            return {"error": f"could not write {svc['file']}: {e.__class__.__name__}"}
        self._apply_key(service, vals)
        self.trace.emit("system", f"{svc['name']} key saved to {svc['file']}", "act")
        self._publish({"type": "keys", "keys": self.keys_status()})
        return {"ok": True, "keys": self.keys_status()}

    def _apply_key(self, service: str, vals: dict) -> None:
        """Best-effort live reload so a saved key takes effect without a restart
        (backfill from the new source still needs a restart)."""
        cfg = self.cfg
        if service == "alphavantage":
            cfg.av_key = self.av.key = vals["key"]
            self.av.exhausted = False
        elif service in ("finnhub", "eodhd", "fmp"):
            setattr(cfg, f"{service}_key", vals["key"])
            src = self.sources.sources.get(service)
            if src:
                src.key = vals["key"]
                src.note = "key set; polling"
                self.sources.toggle(service, True)
        elif service == "schwab":
            ak = vals.get("app_key") or self.schwab_auth.app_key
            asec = vals.get("app_secret") or self.schwab_auth.app_secret
            cfg.schwab_creds = (ak, asec, self.schwab_auth.callback)
            self.schwab_auth.app_key, self.schwab_auth.app_secret = ak, asec

    def snapshot(self) -> dict:
        cfg = self.cfg
        return {
            "type": "snapshot",
            "config": {
                "symbols": self.symbols, "bar_seconds": cfg.bar_seconds, "horizon": cfg.horizon, "window": cfg.window, "max_universe": cfg.max_universe, "chart_bars": cfg.chart_bars,
                "quantiles": list(cfg.quantiles), "min_labels": cfg.min_labels, "features": FEATURES,
                "model": {"params": self.learner.n_params, "d_model": cfg.d_model, "experts": cfg.n_experts,
                          "dilations": list(cfg.dilations), "heads": cfg.n_heads, "window": cfg.window,
                          "receptive_field": self.learner.model.receptive_field,
                          "device": cfg.device, "ms_per_step": (self.autotune_info or {}).get("ms_per_step"),
                          "preset": (self.autotune_info or {}).get("preset")},
            },
            "status": self.status_dict(),
            "controls": self.controls(),
            "sources": self.sources.status(),
            "news_sources": self.news_hub.status(),
            "signals": self.signals_state,
            "signal_providers": self.signals.status(),
            "keys": self.keys_status(),
            "burry": {"enabled": self.burry_enabled, "aggr": self.cfg.burry_aggr, "fade_at": self.cfg.burry_fade_at,
                      "safety": self.cfg.burry_safety},
            "classes": {s: asset_class(s) for s in self.all_symbols},
            "muted": sorted(self.muted),
            "universe": self.all_symbols,
            "prices": self.prices,
            "bars": {s: [b.to_json() for b in self.bars[s]] for s in self.symbols},
            "bars_fine": {s: [b.to_json() for b in self.fine_bars[s]] for s in self.symbols},
            "latest": self.latest,
            "gate": self.gate,
            "outcomes": {s: list(self.outcomes[s]) for s in self.symbols},
            "metrics": self.metrics.as_dict(),
            "history": {k: list(v) for k, v in self.history.items()},
            "log": list(self.log),
            "trace": self.trace.recent(),
            "operator": self.operator_state(),
            "news": self.news,
            "brief": self.brief,
        }

    def snapshot_json(self) -> str:
        return json.dumps(clean(self.snapshot()), separators=(",", ":"))

    # Lifecycle ------------------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self._start()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("engine failed")
            self.status = "error"
            self.trace.emit("system", f"engine failed: {e}", "error")
            self._publish({"type": "status", "status": self.status_dict()})

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.sources.stop()
        await asyncio.to_thread(self._save)

    async def _start(self) -> None:
        cfg = self.cfg
        extra = self.learner.load(cfg.state_dir)
        if extra is not None:
            if "norm" in extra:
                self.features.norm.load(extra["norm"])
            self.metrics.labels = self.learner.labels
            self.metrics.steps = self.learner.steps
            self.trace.emit("system", f"restored checkpoint from {cfg.state_dir}: {self.learner.steps} steps, "
                                      f"{self.learner.labels} labels, {self.learner.size} windows in replay")
        ai = self.autotune_info
        dev = cfg.device + (f", {ai['preset']} preset, {ai['ms_per_step']} ms/step" if ai else "")
        self.trace.emit("system", f"FlintNet: {self.learner.n_params:,} parameters on {dev}; {cfg.n_assets} assets, "
                                  f"{cfg.window} x {cfg.bar_seconds:.0f}s window, {cfg.horizon}-bar horizon, "
                                  f"{cfg.n_experts} regime experts, {cfg.n_heads} heads")

        self.status = "backfilling"
        self._publish({"type": "status", "status": self.status_dict()})
        enabled = [sid for sid, on in self.sources.enabled.items() if on]
        self.trace.emit("feed", f"sources enabled: {', '.join(enabled)}; backfilling "
                                f"{cfg.backfill_seconds / 60:.0f} min per symbol from the best available")
        cutoff = time.time() - cfg.backfill_seconds
        ticks, chosen = await self.sources.backfill(cutoff)
        if not ticks:
            self.trace.emit("feed", "no source returned history; the simulator will provide a live fallback", "warn")
        for s_id in sorted(set(chosen.values())):
            covered = ", ".join(self.base[s] for s, c in chosen.items() if c == s_id)
            self.trace.emit("feed", f"history for {covered} from {self.sources.sources[s_id].name}")
        per_sym = {sym: 0 for sym in self.symbols}
        for t in ticks:
            per_sym[t.symbol] = per_sym.get(t.symbol, 0) + 1
        self.trace.emit("feed", f"{len(ticks)} ticks backfilled (" +
                        ", ".join(f"{self.base[s]} {per_sym[s]}" for s in self.symbols) + ")")
        self._ingest_history(ticks)
        await self._warmup()
        self._seed_forecast()
        self.status = "live"
        self._publish({"type": "status", "status": self.status_dict()})
        self.trace.emit("system", "live: streaming ticks, forecasting every bar, learning as labels mature")
        self.sources.start()
        self._tasks = [asyncio.create_task(c) for c in
                       (self._clock(), self._ticker(), self._checkpoints(), self._news_loop(), self._signals_loop(), self._brief_loop(), self._market_status_loop())]
        await asyncio.gather(*self._tasks)

    def _ingest_history(self, ticks: list[Tick]) -> None:
        for t in ticks:
            self.builder.add(t)
            self._on_tick(t, live=False)
        # Symbols with no deep history (e.g. equities on free feeds backfill one seed tick,
        # commodities a spot price) would otherwise block every aligned row and leave the
        # feature window empty until ~64 live bars pass. Seed each symbol's last close from
        # its earliest backfilled tick so historical rows emit, flat-filling the sparse ones.
        first: dict[str, float] = {}
        for t in ticks:
            first.setdefault(t.symbol, t.price)
        for sym, px in first.items():
            self.builder.last_close.setdefault(sym, px)
        rows = self.builder.roll(time.time())
        self.trace.muted = True
        try:
            for row in rows:
                self._history_row(row)
        finally:
            self.trace.muted = False
        self.trace.emit("bars", f"built {len(rows)} bars of {self.cfg.bar_seconds:.0f}s from history; "
                                f"{self.learner.size} labeled windows ready for training")

    def _history_row(self, row: dict[str, Bar]) -> None:
        self.bar_index += 1
        for s in self.symbols:
            self.bars[s].append(row[s])
        closes = np.array([row[s].close for s in self.symbols], dtype=np.float32)
        self._mature(closes)
        x = self.features.push(row)
        if x is not None:
            self.pending.append(Pending(self.bar_index, row[self.symbols[0]].ts, x, closes, None, {}))

    async def _warmup(self) -> None:
        cfg = self.cfg
        n = self.learner.size
        if n < 16 or cfg.warmup_steps <= 0:
            self.trace.emit("learn", f"only {n} labeled windows; skipping warmup, learning starts live")
            return
        self.status = "training"
        self._publish({"type": "status", "status": self.status_dict()})
        self.trace.emit("learn", f"warmup: {cfg.warmup_steps} steps of batch {min(cfg.batch_size, n)} over {n} windows")
        done = 0
        while done < cfg.warmup_steps:
            chunk = min(25, cfg.warmup_steps - done)
            res = await asyncio.to_thread(self.learner.train_steps, chunk)
            done += chunk
            if res:
                self._absorb_train(res)
                self.trace.emit("learn", f"warmup {done}/{cfg.warmup_steps}: loss {res['loss']:.3f} "
                                         f"(pinball {res['pinball']:.2f} bps, bce {res['bce']:.3f})")
            self._publish({"type": "metrics", "metrics": self.metrics.as_dict(), "history": self._drain_history()})

    # Live loops -------------------------------------------------------------------------

    def _on_live_tick(self, tick: Tick) -> None:
        """Called by the source manager with the active provider's tick for a symbol.
        Quote/heartbeat ticks update the live price but never build candles -- otherwise a
        market-closed feed echoing the last quote would fabricate flat bars from stale data."""
        if not tick.quote:
            self.builder.add(tick)
            self.fine_builder.add(tick)
        self._on_tick(tick, live=True)

    def _on_provider_change(self, sym: str, prev, active) -> None:
        pn = self.sources.sources[prev].name if prev else "none"
        an = self.sources.sources[active].name if active else "none"
        lvl = "warn" if active is None or active == "sim" else "act"
        self.trace.emit("feed", f"{self.base[sym]} provider: {pn} -> {an}", lvl)
        self._publish({"type": "status", "status": self.status_dict()})

    def _on_tick(self, t: Tick, live: bool) -> None:
        p = self.prices.get(t.symbol)
        if p is None:
            return
        p["price"] = t.price
        p["ts"] = t.ts
        if t.bid:
            p["bid"] = t.bid
        if t.ask:
            p["ask"] = t.ask
        if live:
            self.tick_counts[t.symbol] += 1

    async def _clock(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            now = time.time()
            for row in self.builder.roll(now):
                await self._process_row(row)
            upd = {}
            for row in self.fine_builder.roll(now):
                for s in self.symbols:
                    self.fine_bars[s].append(row[s])
                    upd[s] = row[s].to_json()
            if upd:
                self._publish({"type": "fine", "bars": upd})

    async def _ticker(self) -> None:
        period = 1.0 / max(self.cfg.tick_hz, 0.2)
        while True:
            await asyncio.sleep(period)
            self._publish({"type": "tick", "prices": self.prices, "status": self.status_dict()})

    async def _checkpoints(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.checkpoint_minutes * 60)
            await asyncio.to_thread(self._save)
            self.trace.emit("system", f"checkpoint saved to {self.cfg.state_dir} ({self.learner.steps} steps, "
                                      f"{self.learner.size} windows)")

    def _save(self) -> None:
        try:
            self.learner.save(self.cfg.state_dir, extra={"norm": self.features.norm.state()})
        except Exception as e:  # noqa: BLE001
            log.warning("checkpoint failed: %s", e)

    # Per-bar work ------------------------------------------------------------------------

    def _fmt_price(self, s: str, v: float) -> str:
        return f"{v:,.2f}" if v >= 100 else f"{v:.4f}"

    def _seed_forecast(self) -> None:
        """Populate forecasts from the backfilled window immediately at go-live, so cards
        aren't blank until the first live bar closes (important at long bar intervals)."""
        if self.pending:
            pp = self.pending[-1]
            x, closes, ts = pp.x, pp.closes, pp.ts
        else:                                        # not enough bars for a full window yet
            x = self.features.peek_window()
            if x is None or not all(self.bars[s] for s in self.symbols):
                return
            closes = np.array([self.bars[s][-1].close for s in self.symbols], dtype=np.float32)
            ts = self.bars[self.symbols[0]][-1].ts
        pred = self.learner.predict(x)
        sugg = {s: self._suggest(i, s, pred) for i, s in enumerate(self.symbols)}
        self.gate = [float(g) for g in pred.gate]
        self.latest = {}
        for i, s in enumerate(self.symbols):
            self.latest[s] = {**sugg[s], "price": float(closes[i]), "ts": ts, "attn": [float(a) for a in pred.attn[i]]}
        self._publish({"type": "bar", "index": self.bar_index,
                       "bars": {s: self.bars[s][-1].to_json() for s in self.symbols if self.bars[s]},
                       "latest": self.latest, "gate": self.gate, "metrics": self.metrics.as_dict(),
                       "history": {}, "outcomes": {s: list(self.outcomes[s]) for s in self.symbols},
                       "log": list(self.log), "status": self.status_dict()})

    async def _process_row(self, row: dict[str, Bar]) -> None:
        cfg = self.cfg
        self.bar_index += 1
        for s in self.symbols:
            self.bars[s].append(row[s])
        closes = np.array([row[s].close for s in self.symbols], dtype=np.float32)
        ts = row[self.symbols[0]].ts

        self.trace.emit("feed", f"{cfg.bar_seconds:.0f}s: " + " | ".join(
            f"{self.base[s]} {self.tick_counts[s]} ticks" for s in self.symbols))
        for s in self.symbols:
            self.tick_counts[s] = 0
        parts = []
        for s in self.symbols:
            b = row[s]
            r = math.log(b.close / b.open) * 1e4 if b.open > 0 else 0.0
            parts.append(f"{self.base[s]} {self._fmt_price(s, b.close)} ({r:+.1f}bps, {b.trades} trd, vol {b.volume:.3g})")
        self.trace.emit("bars", f"#{self.bar_index} " + " | ".join(parts))

        matured = self._mature(closes)
        x = self.features.push(row)

        if x is not None:
            z = x[:, -1, :]
            for i, s in enumerate(self.symbols):
                self.trace.emit("features", f"{self.base[s]} " + " ".join(f"{n} {z[i, j]:+.2f}" for j, n in enumerate(FEATURES)))

        if matured and self.learning_enabled:
            res = await asyncio.to_thread(self.learner.train_steps, min(8, cfg.steps_per_label * matured))
            if res:
                self._absorb_train(res)
                self.trace.emit("learn", f"step {res['steps']}: loss {res['loss']:.3f} (pinball {res['pinball']:.2f} bps, "
                                         f"bce {res['bce']:.3f}, gate balance {res.get('balance', 0):.3f}) on {matured} new label(s)")
        elif matured:
            self.trace.emit("learn", f"{matured} new label(s) stored; learning is paused")

        if x is not None:
            pred = self.learner.predict(x)
            sugg = {s: self._suggest(i, s, pred) for i, s in enumerate(self.symbols)}
            self.pending.append(Pending(self.bar_index, ts, x, closes, pred, sugg))
            self.gate = [float(g) for g in pred.gate]
            self.latest = {}
            for i, s in enumerate(self.symbols):
                self.latest[s] = {**sugg[s], "price": float(closes[i]), "ts": ts, "attn": [float(a) for a in pred.attn[i]]}
            self.trace.emit("model", f"bar #{self.bar_index}: regime gate " + " ".join(
                f"E{k + 1}={g:.2f}" for k, g in enumerate(self.gate)) +
                f" | band scale x{self.metrics.band_scale:.2f}, P(up) temper {self.metrics.p_scale:.2f}")
            for i, s in enumerate(self.symbols):
                q = pred.q[i]
                att = sorted(((float(pred.attn[i, j]), self.base[self.symbols[j]]) for j in range(len(self.symbols)) if j != i), reverse=True)
                sg = sugg[s]
                self.trace.emit("model", f"{self.base[s]} raw q10 {q[0]:+.1f} q25 {q[1]:+.1f} q50 {q[2]:+.1f} q75 {q[3]:+.1f} "
                                         f"q90 {q[4]:+.1f} bps, P(up) {pred.p_up[i]:.2f} | calibrated band {sg['q'][0]:+.1f} to "
                                         f"{sg['q'][4]:+.1f}, P(up) {sg['p_up']:.2f} | attends " +
                                " ".join(f"{n} {a:.2f}" for a, n in att[:3]))
                self.trace.emit("policy", f"{self.base[s]} {sg['action']}" + (f" size {sg['size']:.2f}" if sg['side'] else "") +
                                f": {sg['why']}" + ("" if sg["trusted"] else " [warming up, not counted]"),
                                "act" if sg["side"] else "info")

        self._publish({
            "type": "bar", "index": self.bar_index, "bars": {s: row[s].to_json() for s in self.symbols},
            "latest": self.latest, "gate": self.gate, "metrics": self.metrics.as_dict(),
            "history": self._drain_history(),
            "outcomes": {s: list(self.outcomes[s]) for s in self.symbols}, "log": list(self.log),
            "status": self.status_dict(),
        })

    def _mature(self, closes: np.ndarray) -> int:
        n = 0
        while self.pending and self.pending[0].index + self.cfg.horizon <= self.bar_index:
            pp = self.pending.popleft()
            y = (np.log(closes / pp.closes) * 1e4).astype(np.float32)
            self.learner.add(pp.x, y)
            if pp.pred is not None:
                self._score(pp, y)
            n += 1
        self.metrics.labels = self.learner.labels
        self.metrics.trusted = self.metrics.live_labels >= self.cfg.min_labels
        return n

    def _absorb_train(self, res: dict) -> None:
        m = self.metrics
        m.loss, m.pinball, m.bce, m.steps = res["loss"], res["pinball"], res["bce"], res["steps"]

    def _calibrate(self, q, p_raw: float) -> tuple[list[float], float]:
        """Apply the online conformal band scale and the P(up) temperature."""
        m = self.metrics
        q50 = float(q[2])
        qc = [q50 + m.band_scale * (float(v) - q50) for v in q]
        p_raw = min(max(p_raw, 1e-6), 1 - 1e-6)
        z = max(-6.0, min(6.0, math.log(p_raw / (1 - p_raw))))
        return qc, 1.0 / (1.0 + math.exp(-m.p_scale * z))

    def _suggest(self, i: int, s: str, pred: Prediction) -> dict:
        cfg = self.cfg
        m = self.metrics
        p_raw = float(pred.p_up[i])
        qc, p = self._calibrate(pred.q[i], p_raw)
        q10, q25, q50, q75, q90 = qc
        iqr = max(q75 - q25, 1e-3)
        score = q50 / iqr
        edge = abs(q50) - cfg.cost_bps
        conviction = abs(2 * p - 1)
        reasons = []
        if abs(score) < cfg.score_threshold:
            reasons.append("the expected move is small next to its uncertainty")
        if conviction < 2 * cfg.prob_margin:
            reasons.append(f"direction is near a coin flip ({p * 100:.0f}% up)")
        if edge <= 0:
            reasons.append("the expected move does not cover trading cost")
        if (score > 0) != (p > 0.5):
            reasons.append("its trend and direction signals disagree")
        side = 0 if reasons else (1 if score > 0 else -1)
        size = cfg.max_size * min(1.0, max(0.0, (conviction - 2 * cfg.prob_margin) / 0.5)) if side else 0.0
        why = "holding: " + "; ".join(reasons) if reasons else f"leaning in: {q50:+.0f} bps expected, {p * 100:.0f}% odds up"
        base_side, base_size = side, size
        muted = s in self.muted
        if muted:
            side, size, reasons = 0, 0.0, ["muted"]
            why = "muted — watching but not suggesting"
        overlay = self._council_overlay(s, side, size, score, qc, iqr) if (self.burry_enabled and not muted) else None
        if overlay:
            side, size = overlay["side"], overlay["size"]
            if overlay["notes"]:
                why = why + "  |  Burry overlay: " + "; ".join(overlay["notes"])
        return {"symbol": s, "action": "BUY" if side > 0 else "SELL" if side < 0 else "HOLD", "side": side,
                "size": round(size, 3), "score": score, "p_up": p, "p_raw": p_raw, "q": qc,
                "q_raw": [float(v) for v in pred.q[i]], "iqr": iqr, "band_scale": m.band_scale, "p_scale": m.p_scale,
                "trusted": m.trusted, "why": why,
                "crowding": round(self.crowding.get(s, 0.0), 3), "guru_tilt": round(self.guru_tilt.get(s, 0.0), 3),
                "base_action": "BUY" if base_side > 0 else "SELL" if base_side < 0 else "HOLD",
                "overlay": overlay["notes"] if overlay else [], "muted": muted}

    def _council_overlay(self, s: str, side: int, size: float, score: float, qc: list, iqr: float) -> dict:
        """Apply the investor council's combined trading ethos to the raw model call.

        The council aggregates each enabled investor's documented philosophy (contrarian,
        value, quality, momentum, macro-bear, activist, margin-of-safety). That shapes:
          - margin of safety: penalize asymmetric downside, scaled by the council's safety
          - fade the crowd: cut/hold/flip trades aligned with euphoria, scaled by contrarian
            (softened when the council leans momentum, e.g. Cohen)
          - counterpoint: the guru consensus + ethos bias push direction; disagreeing with
            a strong lean fades or flips the trade, agreeing boosts it
          - macro-bear drag: a bearish council (Schiff, Burry) trims longs across the board
        Each adjustment names the investors driving it.
        """
        cfg = self.cfg
        crowd = self.crowding.get(s, 0.0)
        guru = self.guru_tilt.get(s, 0.0)
        ethos = self.ethos_bias.get(s, 0.0)
        c = self.council or {}
        contrarian = c.get("contrarian", 0.6)
        safety = c.get("safety", 0.6)
        momentum = c.get("momentum", 0.0)
        macro_bear = c.get("macro_bear", 0.0)
        holders = ((self.signals_state.get("per_asset", {}) or {}).get(s, {}) or {}).get("gurus", {})
        names = {g["id"]: g["name"].split(" / ")[0] for g in self.signals_state.get("gurus", [])}
        longs = [names.get(g, g) for g, v in holders.items() if v > 0.05]
        shorts = [names.get(g, g) for g, v in holders.items() if v < -0.05]

        q10, q90 = qc[0], qc[4]
        notes = []
        o_side, o_size = side, size
        # margin of safety, weighted by the council's safety ethos
        if o_side != 0:
            down = max(0.0, -q10) if o_side > 0 else max(0.0, q90)
            up = max(0.0, q90) if o_side > 0 else max(0.0, -q10)
            if down - up > 0:
                pen = min(1.0, (cfg.burry_safety * (0.4 + safety)) * (down - up) / iqr)
                o_size *= (1 - 0.6 * pen)
                if pen > 0.15:
                    notes.append(f"margin of safety: downside {down:.0f} vs upside {up:.0f} bps, trim {60 * pen:.0f}%")
        # fade the crowd, weighted by contrarian ethos and softened by momentum lean
        if o_side != 0:
            align = o_side * crowd
            fade_at = cfg.burry_fade_at * (1.4 - 0.8 * contrarian)      # more contrarian council fades sooner
            if align > fade_at:
                mag = min(1.0, (align - fade_at) / max(1e-6, 1 - fade_at))
                strength = cfg.burry_aggr * contrarian * max(0.2, 1 - 0.6 * max(0.0, momentum))
                o_size *= max(0.0, 1 - strength * mag)
                notes.append(f"fade the crowd (crowding {crowd:+.2f}, council contrarian {contrarian:.2f}), trim {strength * mag * 100:.0f}%")
                if align > 0.8 and abs(score) < 2 * cfg.score_threshold and contrarian > 0.5:
                    o_side, o_size = -o_side, cfg.max_size * 0.3
                    notes.append("extreme crowding + thin edge: contrarian flip")
                elif o_size < 0.05:
                    o_side = 0
                    notes.append("faded to HOLD")
        # counterpoint from the guru consensus + ethos bias
        lean = 0.6 * guru + 0.4 * ethos
        if lean < -0.3 and o_side > 0:
            o_size *= max(0.1, 1 + lean)
            who = ("Burry" if "Burry" in shorts else (shorts[0] if shorts else "the council"))
            notes.append(f"{who} bearish here (lean {lean:+.2f}) — trimming long")
            if lean < -0.6:
                o_side = 0
                notes.append("council conviction against: BUY -> HOLD")
        elif lean > 0.3 and o_side < 0:
            o_size *= max(0.3, 1 - lean * 0.5)
            who = (longs[0] if longs else "the council")
            notes.append(f"{who} holds this long (lean {lean:+.2f}) — trimming short")
        elif lean > 0.3 and o_side > 0:
            o_size = min(cfg.max_size, o_size * (1 + 0.2 * lean))
            who = (longs[0] if longs else "council")
            notes.append(f"{who}-backed long (lean {lean:+.2f})")
        # macro-bearish council drags longs
        if macro_bear > 0.5 and o_side > 0:
            o_size *= (1 - 0.3 * (macro_bear - 0.5) * 2)
            notes.append(f"macro-bearish council ({macro_bear:.2f}) — trimming long")
        o_size = round(max(0.0, o_size), 3)
        if o_side != side or abs(o_size - size) > 1e-3:
            return {"side": o_side, "size": o_size, "notes": notes, "crowding": crowd, "guru": guru, "ethos": ethos}
        return {"side": side, "size": size, "notes": [], "crowding": crowd, "guru": guru, "ethos": ethos}

    def _score(self, pp: Pending, y: np.ndarray) -> None:
        m = self.metrics
        cfg = self.cfg
        taus = np.asarray(cfg.quantiles, dtype=np.float64)
        parts = []
        for i, s in enumerate(self.symbols):
            q = pp.pred.q[i]
            yi = float(y[i])
            q50 = float(q[2])
            sug = pp.suggestions.get(s, {"side": 0, "size": 0.0, "trusted": False, "action": "HOLD"})
            # Out-of-sample pinball loss of the raw forecast made before this label existed.
            diff = yi - q.astype(np.float64)
            pin = float(np.maximum(taus * diff, (taus - 1) * diff).mean())
            m.live_pinball = _ema(m.live_pinball, pin, 0.05)
            hit = None
            if yi != 0.0 and q50 != 0.0:
                hit = (q50 > 0) == (yi > 0)
                m.decisions += 1
                m.hits += int(hit)
                m.hit_ema = _ema(m.hit_ema, float(hit), 0.02)
            raw_inside = float(q[0]) <= yi <= float(q[-1])
            m.coverage_raw_n += 1
            m.covered_raw += int(raw_inside)
            qc = sug.get("q") or [float(v) for v in q]
            inside = qc[0] <= yi <= qc[-1]
            m.coverage_n += 1
            m.covered += int(inside)
            m.coverage_ema = _ema(m.coverage_ema, float(inside), 0.02)
            m.sharpness_ema = _ema(m.sharpness_ema, float(qc[-1] - qc[0]), 0.05)
            # Adaptive conformal update: the band scale settles where 20% of labels fall outside.
            m.band_scale = float(np.clip(m.band_scale * math.exp(cfg.band_gamma * ((0.0 if inside else 1.0) - 0.2)), 0.5, 25.0))
            # Online temperature on the direction head, one SGD step of the calibration cross-entropy.
            if yi != 0.0:
                p_raw = min(max(float(sug.get("p_raw", pp.pred.p_up[i])), 1e-6), 1 - 1e-6)
                z = max(-6.0, min(6.0, math.log(p_raw / (1 - p_raw))))
                pc = 1.0 / (1.0 + math.exp(-m.p_scale * z))
                m.p_scale = float(np.clip(m.p_scale - cfg.temper_lr * (pc - (1.0 if yi > 0 else 0.0)) * z, 0.02, 1.5))
            pnl = None
            if sug["side"] != 0 and sug["trusted"]:
                pnl = sug["side"] * sug["size"] * yi - cfg.cost_bps * sug["size"]
                m.pnl_bps += pnl
                m.pnl_by_symbol[s] = m.pnl_by_symbol.get(s, 0.0) + pnl
                m.suggestions += 1
                m.suggestion_wins += int(pnl > 0)
                self._log(f"{s} {sug['action']} x{sug['size']:.2f} resolved {yi:+.1f} bps, paper {pnl:+.2f} bps",
                          "win" if pnl > 0 else "loss")
            self.outcomes[s].append({"t": pp.ts, "y": yi, "q50": q50, "hit": hit, "inside": inside,
                                     "side": sug["side"], "pnl": pnl, "pin": pin})
            mark = "hit" if hit else "miss" if hit is False else "flat"
            parts.append(f"{self.base[s]} {yi:+.1f} vs q50 {q50:+.1f} {mark}" + ("" if inside else " (outside band)"))
        m.live_labels += 1
        m.trusted = m.live_labels >= cfg.min_labels
        self._hpush("loss", m.live_pinball)
        self._hpush("hit", m.hit_ema)
        self._hpush("coverage", m.coverage_ema)
        self._hpush("pnl", m.pnl_bps)
        self.trace.emit("learn", f"label for bar #{pp.index} matured: " + " | ".join(parts) +
                        f" | live pinball {m.live_pinball:.2f} bps, band scale x{m.band_scale:.2f}, P(up) temper {m.p_scale:.2f}")

    # Signals + Burry/Buffett overlay ----------------------------------------------------

    async def _signals_loop(self) -> None:
        await asyncio.sleep(6)
        while True:
            await self.gather_signals()
            await asyncio.sleep(max(1.0, self.cfg.signals_minutes) * 60)

    async def gather_signals(self) -> None:
        if self._gathering:
            return
        self._gathering = True
        say = lambda text, level="info": self.trace.emit("signals", text, level)  # noqa: E731
        try:
            active = [p["name"] for p in self.signals.status() if p["enabled"]]
            self.trace.emit("signals", "refreshing: " + (", ".join(active) or "no providers enabled"))
            st = await asyncio.wait_for(self.signals.gather(say), timeout=180)
            self.signals_state = st
            self.crowding = st.get("crowding", {})
            self.guru_tilt = st.get("guru_tilt", {})
            self.ethos_bias = st.get("ethos_bias", {})
            self.council = st.get("council", {})
            for sym in self.symbols:
                pa = st["per_asset"].get(sym)
                if pa:
                    self.features.set_exo(sym, pa["feat"])
            mk = st.get("market", {})
            if mk.get("fng_value") is not None:
                self.trace.emit("signals", f"Fear & Greed {mk['fng_value']} ({mk['fng_class']})")
            for sym in self.symbols:
                pa = st["per_asset"].get(sym, {})
                f = pa.get("feat", {})
                self.trace.emit("signals", f"{self.base[sym]} crowding {self.crowding.get(sym, 0):+.2f} | WSB {pa.get('mentions', 0)} "
                                f"mentions sent {f.get('wsb_sent', 0):+.2f} | funding {f.get('funding', 0):+.2f} "
                                f"L/S {f.get('longshort', 0):+.2f} | guru {f.get('guru_net', 0):+.2f} ethos {f.get('ethos_bias', 0):+.2f}")
            c = st.get("council", {})
            if c:
                self.trace.emit("signals", "council ethos: " + ", ".join(f"{k} {v:+.2f}" for k, v in c.items()), "act")
            for g in st.get("gurus", []):
                if g.get("holdings"):
                    top = ", ".join(f"{h['ticker'] or h['issuer'][:8]} {h['weight']*100:.0f}%{'(PUT)' if h['put'] else ''}" for h in g["holdings"][:5])
                    self.trace.emit("signals", f"{g['name']} 13F ({g['asof']}): {top}", "act")
            self._publish({"type": "signals", "signals": self.signals_state, "signal_providers": self.signals.status()})
        except Exception as e:  # noqa: BLE001
            self.trace.emit("signals", f"signal refresh failed: {type(e).__name__}: {str(e)[:120]}", "error")
        finally:
            self._gathering = False

    # News ------------------------------------------------------------------------------

    async def _news_loop(self) -> None:
        if not self.cfg.news_enabled:
            self.trace.emit("news", "news skimming disabled (FLINT_NEWS=0)")
            return
        await asyncio.sleep(3)
        while True:
            await self.skim_news()
            await asyncio.sleep(max(1.0, self.cfg.news_minutes) * 60)

    async def skim_news(self) -> None:
        if self._skimming:
            self.trace.emit("news", "skim already in progress", "warn")
            return
        self._skimming = True
        say = lambda text, level="info": self.trace.emit("news", text, level)  # noqa: E731
        try:
            active = [x["name"] for x in self.news_hub.status() if x["enabled"]]
            self.trace.emit("news", "skimming: " + (", ".join(active) or "no news sources enabled"))
            res = await asyncio.wait_for(self.news_hub.skim(say), timeout=300)
            for s in self.symbols:
                pa = res["per_asset"].get(s)
                base = (pa["sentiment"], pa["attention"]) if pa else (0.0, 0.0)
                self.news_base[s] = base
                self.features.set_news(s, *self._blend_news(s, *base))
            self.news = res
            fresh = [h for h in res["headlines"] if h["new"]]
            self.trace.emit("news", f"{len(res['headlines'])} relevant headlines via {res['method']}, {len(fresh)} new")
            for h in fresh[:25]:
                tags = ",".join(h["assets"]) or "market"
                self.trace.emit("news", f"[{tags}] {h['sentiment']:+.2f} {h['title']} ({h['source']})",
                                "act" if abs(h["sentiment"]) >= 0.5 else "info")
            for idea in res["ideas"]:
                self.trace.emit("news", "idea: " + idea, "act")
            self._publish({"type": "news", "news": self.news})
        except Exception as e:  # noqa: BLE001
            self.trace.emit("news", f"skim failed: {type(e).__name__}: {str(e)[:120]}", "error")
        finally:
            self._skimming = False

    # Controls ---------------------------------------------------------------------------

    def _brief_slices(self) -> dict:
        """Assemble flint's current state into compact per-topic data slices for the analysts."""
        sig = self.signals_state or {}
        m = sig.get("market", {}) or {}
        met = self.metrics
        b = self.base
        stance = "trusted" if met.trusted else f"warming up ({met.live_labels}/{self.cfg.min_labels} live labels)"

        def line(v):
            q = v.get("q") or [0, 0, 0, 0, 0]
            return (f"{b.get(v['symbol'], v['symbol'])} {v.get('action')} score {v.get('score', 0):+.2f} "
                    f"q50 {q[2]:+.0f}bps P(up) {v.get('p_up', 0):.2f} band[{q[0]:+.0f},{q[4]:+.0f}]")
        calls = sorted((v for v in self.latest.values() if v.get("q")), key=lambda v: -abs(v.get("score", 0)))
        tape = f"model stance: {stance}; regime {m.get('regime', '?')}\n" + "\n".join(line(v) for v in calls[:10])

        secs = m.get("sectors") or []
        lead = ", ".join(f"{s['name']} {s['chg']:+.1f}%" for s in secs[:2])
        lag = ", ".join(f"{s['name']} {s['chg']:+.1f}%" for s in secs[-2:])
        mv = m.get("movers", {}) or {}
        g = ", ".join(f"{x['symbol']} {x['chg']:+.1f}%" for x in (mv.get("gainers") or [])[:5])
        lo = ", ".join(f"{x['symbol']} {x['chg']:+.1f}%" for x in (mv.get("losers") or [])[:5])
        macro = (f"breadth {round((m.get('breadth') or 0) * 100)}% of {m.get('breadth_n', '?')} ETFs up; "
                 f"VIX {m.get('vix', '?')}; regime {m.get('regime', '?')}\n"
                 f"sector leaders: {lead}; laggards: {lag}\ntop gainers: {g}\ntop losers: {lo}")

        cr = sig.get("crowding", {}) or {}
        hot = ", ".join(f"{b.get(s, s)} {v:+.2f}" for s, v in sorted(cr.items(), key=lambda kv: -kv[1])[:3])
        wsb = ", ".join(f"{b.get(s, s)} {n} mentions" for n, s in
                        sorted(((pa.get("mentions", 0), s) for s, pa in (sig.get("per_asset") or {}).items()), reverse=True)[:3] if n)
        positioning = f"most crowded: {hot or 'n/a'}\ntop retail attention: {wsb or 'n/a'}"

        gt = sig.get("guru_tilt", {}) or {}
        bull = ", ".join(f"{b.get(s, s)} +{v:.2f}" for s, v in sorted(gt.items(), key=lambda kv: -kv[1])[:4] if v > 0.02)
        bear = ", ".join(f"{b.get(s, s)} {v:.2f}" for s, v in sorted(gt.items(), key=lambda kv: kv[1])[:4] if v < -0.02)
        council = sig.get("council", {}) or {}
        lean = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(council.items(), key=lambda kv: -abs(kv[1]))[:4])
        smart = f"bullish tilts: {bull or 'none'}\nbearish tilts: {bear or 'none'}\npanel bias: {lean or 'n/a'}"

        stats = (f"{len(self.symbols)} US equities tracked; model {stance}; regime {m.get('regime', '?')}; "
                 f"breadth {round((m.get('breadth') or 0) * 100)}%; VIX {m.get('vix', '?')}")
        return {"tape": tape, "macro": macro, "positioning": positioning, "smart_money": smart, "stats": stats}

    async def generate_brief(self) -> None:
        if self.brief.get("generating"):
            return
        self.brief = {**self.brief, "generating": True, "error": None}
        self._publish({"type": "brief", "brief": self.brief})
        say = lambda t: self.trace.emit("system", "brief: " + t)  # noqa: E731
        try:
            from .brief import write_brief
            res = await write_brief(self.cfg, self._brief_slices(), say)
            self.brief = {"text": None, "takes": {}, "models": {}, "generating": False, **res}
            if res.get("error"):
                self.trace.emit("system", "brief unavailable: " + res["error"], "warn")
            else:
                n = sum(1 for t in res.get("takes", {}).values() if t)
                self.trace.emit("system", f"brief written by {res['models']['big']} from {n} local desk notes", "act")
        except Exception as ex:  # noqa: BLE001
            self.brief = {**self.brief, "generating": False, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
            self.trace.emit("system", "brief failed: " + str(ex)[:120], "error")
        self._publish({"type": "brief", "brief": self.brief})

    async def _market_status_loop(self) -> None:
        """Poll Finnhub for the holiday-aware US market status (open/pre/post/closed)."""
        if not self.cfg.finnhub_key:
            return
        while True:
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get("https://finnhub.io/api/v1/stock/market-status",
                                    params={"exchange": "US", "token": self.cfg.finnhub_key})
                    if r.status_code == 200:
                        d = r.json()
                        ms = {"isOpen": bool(d.get("isOpen")), "session": d.get("session"),
                              "holiday": d.get("holiday"), "t": d.get("t")}
                        if ms != self.market_status:
                            self.market_status = ms
                            self._publish({"type": "status", "status": self.status_dict()})
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(60)

    async def _brief_loop(self) -> None:
        while True:
            # only write when the model is live: during backfill/warmup the local LLM and the
            # DNN contend for the GPU and both crawl to a halt
            if self.brief_enabled and self.status == "live" and (self.signals_state or {}).get("market"):
                try:
                    await self.generate_brief()
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(max(2.0, self.cfg.brief_minutes) * 60)
            else:
                await asyncio.sleep(15)

    def apply_control(self, payload: dict) -> dict:
        changed = {}
        for k, v in (payload.get("set") or {}).items():
            if k not in TUNABLE:
                continue
            try:
                val = TUNABLE[k](v)
            except (TypeError, ValueError):
                continue
            setattr(self.cfg, k, val)
            changed[k] = val
            if k == "lr":
                for g in self.learner.opt.param_groups:
                    g["lr"] = val
        if changed:
            self.trace.emit("system", "settings changed: " + ", ".join(f"{k}={v}" for k, v in changed.items()), "act")
            self.metrics.trusted = self.metrics.live_labels >= self.cfg.min_labels
        action = payload.get("action")
        if action == "pause":
            self.learning_enabled = False
            self.trace.emit("system", "learning paused; the model keeps forecasting with frozen weights", "act")
        elif action == "resume":
            self.learning_enabled = True
            self.trace.emit("system", "learning resumed", "act")
        elif action == "checkpoint":
            self._save()
            self.trace.emit("system", f"checkpoint saved to {self.cfg.state_dir}", "act")
        elif action == "reset":
            self.learner.reset(clear_replay=bool(payload.get("clear_replay")))
            self.metrics = Metrics(pnl_by_symbol={s: 0.0 for s in self.symbols})
            self.metrics.labels = self.learner.labels
            for v in self.history.values():
                v.clear()
            self._history_new = {k: [] for k in self.history}
            for v in self.outcomes.values():
                v.clear()
            self.trace.emit("system", "model weights and optimizer re-initialized" +
                            (", replay cleared" if payload.get("clear_replay") else ", replay kept"), "act")
        elif action == "skim":
            asyncio.get_running_loop().create_task(self.skim_news())
        elif action == "toggle_source":
            res = self.sources.toggle(payload.get("id", ""), bool(payload.get("on")))
            if res.get("error"):
                self.trace.emit("feed", f"could not toggle {payload.get('id')}: {res['error']}", "warn")
            self._publish({"type": "sources", "sources": self.sources.status(), "status": self.status_dict()})
        elif action == "toggle_news":
            self.news_hub.toggle(payload.get("id", ""), bool(payload.get("on")))
            self.trace.emit("news", f"{payload.get('id')} news turned {'on' if payload.get('on') else 'off'}", "act")
            self._publish({"type": "news_sources", "news_sources": self.news_hub.status()})
        elif action == "toggle_signal":
            self.signals.toggle(payload.get("id", ""), bool(payload.get("on")))
            self.trace.emit("signals", f"{payload.get('id')} signal turned {'on' if payload.get('on') else 'off'}", "act")
            asyncio.get_running_loop().create_task(self.gather_signals())
            self._publish({"type": "signal_providers", "signal_providers": self.signals.status()})
        elif action == "refresh_signals":
            asyncio.get_running_loop().create_task(self.gather_signals())
        elif action == "brief":
            asyncio.get_running_loop().create_task(self.generate_brief())
        elif action == "toggle_brief":
            self.brief_enabled = bool(payload.get("on"))
            self.trace.emit("system", f"local LLM brief {'enabled' if self.brief_enabled else 'disabled'}", "act")
            if self.brief_enabled and not self.brief.get("text"):
                asyncio.get_running_loop().create_task(self.generate_brief())
        elif action == "burry":
            self.burry_enabled = bool(payload.get("on"))
            self.trace.emit("policy", f"Burry/Buffett overlay {'enabled' if self.burry_enabled else 'disabled'}", "act")
        elif action == "mute":
            on = bool(payload.get("on"))  # on == muted
            for sym in payload.get("symbols", []):
                sym = str(sym).upper()
                self.muted.add(sym) if on else self.muted.discard(sym)
            asyncio.get_running_loop().create_task(self.reconfigure())
        elif action == "mute_class":
            cls, on = payload.get("class"), bool(payload.get("on"))
            for sym in self.all_symbols:
                if asset_class(sym) == cls:
                    self.muted.add(sym) if on else self.muted.discard(sym)
            asyncio.get_running_loop().create_task(self.reconfigure())
        elif action == "add_symbols":
            added = []
            for sym in payload.get("symbols", []):
                sym = str(sym).strip().upper()
                if sym and sym not in self.all_symbols and len(self.all_symbols) < self.cfg.max_universe:
                    self.all_symbols.append(sym); added.append(sym)
            if added:
                self.trace.emit("system", f"added to universe: {', '.join(added)}", "act")
                asyncio.get_running_loop().create_task(self.reconfigure())
        elif action == "remove_symbols":
            rem = {str(s).strip().upper() for s in payload.get("symbols", [])}
            self.all_symbols = [s for s in self.all_symbols if s not in rem]
            self.muted -= rem
            if rem:
                self.trace.emit("system", f"removed from universe: {', '.join(sorted(rem))}", "act")
                asyncio.get_running_loop().create_task(self.reconfigure())
        elif action == "add_movers":
            n = int(payload.get("n", 20))
            radar = (self.signals_state.get("market", {}) or {}).get("radar", [])
            added = []
            for r in radar:
                sym = str(r.get("symbol", "")).upper()
                if sym and "." not in sym and sym not in self.all_symbols and len(self.all_symbols) < self.cfg.max_universe:
                    self.all_symbols.append(sym); added.append(sym)
                if len(added) >= n:
                    break
            if added:
                self.trace.emit("system", f"added top {len(added)} movers: {', '.join(added)}", "act")
                asyncio.get_running_loop().create_task(self.reconfigure())
        elif action == "inject":
            return {"type": "inject", **self.inject_note(payload.get("text", ""))}
        state = {"type": "controls", "controls": self.controls(), "metrics": self.metrics.as_dict(),
                 "sources": self.sources.status(), "news_sources": self.news_hub.status(),
                 "signal_providers": self.signals.status()}
        self._publish(state)
        return state
