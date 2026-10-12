"""The engine: feed -> bars -> features -> model -> suggestions, plus everything the UI sees."""
from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging
import math
import os
import re

import httpx
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field

import numpy as np

from .autotune import PRESETS, free_memory_gb
from .backtest import combine_reports, run_backtest, variant_summary
from .meta import MetaLabeler, features as meta_features
from .newshist import NewsHistory, backfill_span
from .bars import Bar, BarBuilder
from .events import AHEAD_S, EarningsCalendar
from .features import BARS_D, FEATURE_SET, FEATURES, N_FEATURES, FeatureBuilder
from .feed import Tick
from .learner import OnlineLearner, Prediction
from .news import TICKER_STOPLIST, NewsHub, is_crypto
from .paper import PaperBook
from .schwab import SchwabAuth
from .etrade import ETradeAuth
from .signals import FundamentalsProvider, SignalHub
from .target import hedge_targets, index_columns
from .sources import AlphaVantage, SourceManager, asset_class
from .trace import Trace
from . import clock, policy

log = logging.getLogger(__name__)

TUNABLE = {"score_threshold": float, "prob_margin": float, "cost_bps": float, "max_size": float,
           "lr": float, "steps_per_label": int, "min_labels": int, "news_minutes": float,
           "burry_aggr": float, "burry_fade_at": float, "burry_safety": float, "signals_minutes": float,
           "kelly_fraction": float, "move_floor_bps": float, "min_hit_rate": float, "meta_min": float,
           "deep_backfill_days": float, "idle_train_epochs": float,
           "extended_hours": lambda v: str(v).lower() in ("1", "true", "yes", "on"),
           "train_in_session": lambda v: str(v).lower() in ("1", "true", "yes", "on"),
           "skill_min_n": int, "confirm_bars": int, "stock_min_hold_bars": int, "max_spread_bps": float,
           "min_price": float, "option_max_frac": float, "min_position_dollars": float,
           "impact_eta": float, "max_impact_bps": float, "theta_in_edge": lambda v: str(v).lower() in ("1", "true", "yes", "on"),
           "straddle_edge": float, "stop_loss_bps": float, "take_profit_bps": float, "max_hold_bars": int,
           "exit_cooldown_bars": int, "daily_loss_limit": float,
           "backtest_weekends": lambda v: str(v).lower() in ("1", "true", "yes", "on"),
           "backtest_hours": float, "backtest_days": int, "backtest_cost_bps": float, "backtest_night_epochs": float,
           "backtest_target": lambda v: str(v).lower() if str(v).lower() in ("absolute", "relative", "both") else "both",
           "hedge_index": lambda v: str(v).lower() in ("1", "true", "yes", "on"), "hedge_min": float,
           "overfit_patience": int, "memory_floor_gb": float,
           "universe_refresh": lambda v: str(v).lower() in ("1", "true", "yes", "on"), "interesting_count": int,
           "size_by_coverage": lambda v: str(v).lower() in ("1", "true", "yes", "on")}

WARM_S = 9 * 86400.0          # a week of sessions plus the window: what an offline chunk walks before its first emitted bar


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


def summarize_holds(suggestions: dict[str, dict]) -> list[dict]:
    """Summarize why Flint is holding for each symbol into a count-by-reason list.

    Takes per-symbol suggestion dicts (with "reasons" and "muted" keys) and returns
    a list of {"reason": label, "n": count} sorted by count descending. Parenthesised
    suffixes like "(3/8 calls)" are stripped from the label.

    A muted symbol (muted: True with reasons ["muted"]) counts under "muted".
    """
    counts: dict[str, int] = {}
    for sym, sug in suggestions.items():
        if not isinstance(sug, dict):
            continue
        if sug.get("muted") is True and sug.get("reasons") == ["muted"]:
            counts["muted"] = counts.get("muted", 0) + 1
            continue
        reasons = sug.get("reasons") or []
        for r in reasons:
            # Strip parenthesised suffix: "no track record (3/8 calls)" -> "no track record"
            label = re.sub(r"\s*\([^)]*\)\s*$", "", r).strip()
            counts[label] = counts.get(label, 0) + 1
    # Sort by count descending
    return [{"reason": r, "n": n} for r, n in sorted(counts.items(), key=lambda x: -x[1])]


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
    {"id": "alpaca", "name": "Alpaca", "file": "alpaca.json", "kind": "json_multi",
     "url": "https://alpaca.markets/",
     "note": "Free API keys: real-time IEX quotes + bars (full SIP on a paid plan). No brokerage account needed.",
     "fields": [{"id": "key_id", "label": "API key ID"}, {"id": "secret_key", "label": "Secret key"}]},
    {"id": "tradier", "name": "Tradier", "file": "tradier.json", "kind": "json_multi",
     "url": "https://documentation.tradier.com/",
     "note": "Brokerage access token: real-time quotes with bid/ask + 5-min history.",
     "fields": [{"id": "token", "label": "Access token"}]},
    {"id": "etrade", "name": "E*TRADE", "file": "etrade.json", "kind": "json_multi",
     "url": "https://developer.etrade.com/",
     "note": "Market data via OAuth. After saving, run  flint etrade-auth  to log in (tokens renew daily).",
     "fields": [{"id": "consumer_key", "label": "Consumer key"}, {"id": "consumer_secret", "label": "Consumer secret"}]},
]


# upper-case words in headlines that look like tickers and are not
NOT_TICKERS = TICKER_STOPLIST | {"CEO", "CFO", "COO", "CTO", "IPO", "ETF", "ETFS", "GDP", "CPI", "PPI", "PCE", "FED", "FOMC", "SEC", "USA",
                                 "NYSE", "EPS", "YTD", "PMI", "OPEC", "GOP", "DOJ", "FTC", "FDA", "NFL", "NBA", "MLB", "NATO", "LLC",
                                 "INC", "ESG", "AND", "THE", "FOR", "NEW", "TOP", "BUY", "SELL", "HOLD", "USD", "EUR", "GBP", "JPY",
                                 "ECB", "BOJ", "BOE", "IMF", "WHO", "WSJ", "CNBC", "ETH", "BTC", "XRP", "SOL", "DOGE", "AI", "EV", "UK",
                                 "US", "EU", "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "YOY", "QOQ", "MOM", "WOW", "ATH", "DCF"}
_TICKER_RX = re.compile(r"\b[A-Z]{3,5}\b")


def headline_tickers(headlines: list[dict], known: set[str]) -> dict[str, int]:
    """Tickers named in headline text, as upper-case words of three to five letters that are
    known symbols (on the radar, in a sector record or in a guru's book) and not ordinary
    acronyms. The organic way a new name shows up: the press is talking about it."""
    counts: dict[str, int] = {}
    for h in headlines or []:
        for tok in set(_TICKER_RX.findall(str(h.get("title") or ""))):
            if tok in known and tok not in NOT_TICKERS:
                counts[tok] = counts.get(tok, 0) + 1
    return counts


def rank_scores(ordered: list[str], span: int) -> dict[str, float]:
    """100 for the first name, falling linearly to 0 at position `span`: one scale for every
    candidate source, so a top mover, the most-mentioned name and a fund's biggest position
    rank as equally interesting."""
    span = max(1, span)
    return {s: max(0.0, 100.0 * (1.0 - i / span)) for i, s in enumerate(ordered)}


def pick_universe(core: list[str], held, current: list[str], candidates: dict[str, float], record: dict,
                  *, count: int, max_universe: int, today: str, grace_days: int = 14, min_tenure_days: int = 5):
    """The universe Flint should model: the configured core and every holding always, plus the
    `count` most interesting names in the room left under max_universe, with a promotion and
    a demotion path. A name is promoted when it ranks in the top `count` candidates. Once in,
    it stays while it keeps ranking inside the top 1.5 x count (`last_hit`), and is demoted
    only after `grace_days` without a hit and `min_tenure_days` of tenure, because every
    demotion forfeits that name's learned row and history. Returns (symbols, record, promoted,
    demoted); `record` maps interesting names to their since/last_hit dates."""
    fixed: list[str] = []
    for s in list(core) + sorted(set(held) - set(core)):
        if s not in fixed:
            fixed.append(s)
    fixed = fixed[:max_universe]
    room = max(0, min(int(count), max_universe - len(fixed)))
    record = {s: v for s, v in (record or {}).items() if s in current}   # names that never made it in carry no record
    ranked = [s for s, _ in sorted(candidates.items(), key=lambda kv: -kv[1]) if s not in fixed]
    band = set(ranked[:int(room * 1.5)])
    today_d = datetime.fromisoformat(today).date()
    rec = {s: dict(v) for s, v in (record or {}).items()}
    interesting = [s for s in current if s not in fixed]
    keep, demoted = [], []
    for s in interesting:
        r = rec.setdefault(s, {"since": today, "last_hit": today})
        if s in band:
            r["last_hit"] = today
        idle = (today_d - datetime.fromisoformat(r.get("last_hit", today)).date()).days
        tenure = (today_d - datetime.fromisoformat(r.get("since", today)).date()).days
        if idle > grace_days and tenure >= min_tenure_days:
            demoted.append(s)
        else:
            keep.append(s)
    if len(keep) > room:                                     # the room shrank (holdings grew): the weakest go first
        keep.sort(key=lambda s: -candidates.get(s, 0.0))
        demoted += keep[room:]
        keep = keep[:room]
    promoted = []
    for s in ranked:
        if len(keep) + len(promoted) >= room:
            break
        if s not in keep and s not in demoted:
            promoted.append(s)
            rec[s] = {"since": today, "last_hit": today}
    for s in demoted:
        rec.pop(s, None)
    for s in list(rec):
        if s in fixed:
            rec.pop(s, None)
    final_set = set(fixed) | set(keep) | set(promoted)
    out = [s for s in current if s in final_set]            # retained names keep their place
    out += [s for s in fixed + keep + promoted if s not in out]
    return out, rec, promoted, demoted


def memory_low(device: str, floor_gb: float, reader=free_memory_gb, settle: float = 3.0) -> float | None:
    """The free memory figure when it sits under `floor_gb` on two readings `settle` seconds
    apart, else None. One reading is not enough: a compile's autotuning or a cache release in
    flight can dip free memory for a moment, and a stop on that moment threw a weekend run away
    on 2026-10-10 two minutes in."""
    f = reader(device)
    if f is None or f >= floor_gb:
        return None
    time.sleep(settle)
    g = reader(device)
    if g is None or g >= floor_gb:
        return None
    return min(f, g)


def backtest_preset_for(free_gb: float | None, live_d_model: int, live_peak_gb: float, *, floor: float = 10.0, presets=PRESETS):
    """The widest preset, no wider than the live one, whose training peak fits in the memory free
    right now with `floor` GB left for everything else. A training peak scales about linearly
    with d_model (the 2026-10-10 ladder at 232 names: 768 -> 51 GB, 512 -> 34, 384 -> 25), so the
    live preset's measured peak is scaled by width. None when nothing fits. With no free-memory
    figure (a CPU box) the live preset is returned. Exists because two 5XL nets training in one
    121 GB box took the Spark down on 2026-10-10."""
    live = next((p for p in presets if p[1] == live_d_model), presets[-1])
    if free_gb is None or live_peak_gb <= 0:
        return live
    for p in reversed(presets):
        if p[1] > live_d_model:
            continue
        need = live_peak_gb * (p[1] / float(live_d_model)) * 1.25 + floor
        if need <= free_gb:
            return p
    return None


def _stalled(vals: list[float], patience: int) -> bool:
    """True when the last `patience` checks brought no improvement over the best before them."""
    if patience <= 0 or len(vals) <= patience:
        return False
    return min(vals[-patience:]) >= min(vals[:-patience])


def _mask(v: str) -> str:
    v = v or ""
    return (v[:2] + "***" + v[-2:]) if len(v) >= 6 else ("***" if v else "")


@dataclass(slots=True)
class Pending:
    index: int
    ts: float
    x: np.ndarray
    closes: np.ndarray
    real: np.ndarray
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
    holdout_pinball: float | None = None   # loss on the 5% of windows the trainer never samples
    holdout_bce: float | None = None
    band_scale: float = 2.0   # conformal multiplier on the predicted band widths; starts wide, earns its way down
    p_scale: float = 0.3      # temperature on the direction logit (1 = raw); starts sceptical

    def as_dict(self) -> dict:
        return {
            "labels": self.labels, "steps": self.steps, "loss": self.loss, "pinball": self.pinball, "bce": self.bce,
            "decisions": self.decisions, "hit_rate": self.hits / self.decisions if self.decisions else None,
            "hit_ema": self.hit_ema, "coverage": self.covered / self.coverage_n if self.coverage_n else None,
            "coverage_ema": self.coverage_ema, "sharpness": self.sharpness_ema,
            "holdout_pinball": self.holdout_pinball, "holdout_bce": self.holdout_bce,
            "suggestions": self.suggestions, "win_rate": self.suggestion_wins / self.suggestions if self.suggestions else None,
            "pnl_bps": self.pnl_bps, "pnl_by_symbol": dict(self.pnl_by_symbol), "trusted": self.trusted,
            "live_labels": self.live_labels, "live_pinball": self.live_pinball,
            "coverage_raw": self.covered_raw / self.coverage_raw_n if self.coverage_raw_n else None,
            "band_scale": self.band_scale, "p_scale": self.p_scale,
        }


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.all_symbols = [s.upper() for s in cfg.symbols]     # full configured universe
        self.core_symbols = list(self.all_symbols)             # the configured names: always modeled
        self._held: set[str] = set()                           # latest holdings (portfolio loop): always modeled
        self._universe_refreshed = ""                          # ET date of the last promotion/demotion pass
        self._interesting: dict = {}                           # name -> {since, last_hit} for promoted names
        self._built_universe: list[str] = []                   # the full list the current model was built for
        # Merge the persisted universe (state/universe.json): symbols added at runtime by the
        # portfolio/movers/news loops must survive a restart, or the rebuilt
        # engine starts at the config default, mismatches the checkpoint's symbol list, and
        # resets the trained model. Delete the file to intentionally shrink the universe.
        try:
            uni = json.load(open(os.path.join(cfg.state_dir, "universe.json")))
            saved = uni.get("symbols", [])
            extra_syms = [s for s in saved if s not in set(self.all_symbols)]
            room = cfg.max_universe - len(self.all_symbols)
            self.all_symbols.extend(extra_syms[:max(0, room)])
            self._universe_refreshed = str(uni.get("refreshed") or "")
            self._interesting = dict(uni.get("interesting") or {})
        except (OSError, ValueError):
            pass
        self.muted = {x.strip().upper() for x in cfg.muted_symbols.split(",") if x.strip()} & set(self.all_symbols)
        self.subs: set[asyncio.Queue] = set()
        self.trace = Trace(self._publish)
        self.av = AlphaVantage(cfg.av_key, cfg.av_rate_seconds)
        ak, asec, acb = cfg.schwab_creds
        token_file = cfg.schwab_token_file or os.path.join(cfg.state_dir, "schwab_tokens.json")
        self.schwab_auth = SchwabAuth(ak, asec, acb, token_file)
        et_ck, et_cs = cfg.etrade_creds
        self.etrade_auth = ETradeAuth(et_ck, et_cs, cfg.etrade_token_file or os.path.join(cfg.state_dir, "etrade_tokens.json"))
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
        self._status_http: int | None = None     # last market-status HTTP code, so a rejected key is reported once
        self.backtest: dict = {}                  # last walk-forward report (state/backtest.json)
        self.backtest_progress: dict = {"running": False}
        self._backtesting = False
        self._backtest_requested = False
        self._backtest_stop = False
        self._backtest_stop_reason = ""
        self._backtest_waiting = False
        self._lowmem_said = False
        self._ring_full_said = False
        try:
            with open(os.path.join(self.cfg.state_dir, "backtest.json")) as f:
                self.backtest = json.load(f)
        except (OSError, ValueError):
            pass
        self.meta = MetaLabeler()                   # the meta-labeler: per forecast, not per name, so it outlives a universe rebuild
        self.news_history = NewsHistory(os.path.join(cfg.state_dir, "news_history.json"))   # dated headlines, so paged history carries the news slots
        self.news_history.load()
        self.events = EarningsCalendar(os.path.join(cfg.state_dir, "events.json"))   # dated report dates, for the extended set's earnings flags
        self.events.load()
        self._news_task: asyncio.Task | None = None

        self._events_task: asyncio.Task | None = None
        self._startup_span: tuple[float, float] | None = None   # the startup backfill's days, rewritten once their news is in
        self.sweep: dict = {}                     # the last policy sweep over a backtest's stored forecasts (state/sweep.json)
        try:
            with open(os.path.join(self.cfg.state_dir, "sweep.json")) as f:
                self.sweep = json.load(f)
        except (OSError, ValueError):
            pass
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
        self.features = FeatureBuilder(self.symbols, cfg.window, index=self._market_symbol())
        self.features.events = self.events                 # the earnings flags read the dated calendar live and for history alike
        self.news_base = {s: (0.0, 0.0) for s in self.symbols}  # last pure news skim, before operator blend
        self.learner = OnlineLearner(cfg, N_FEATURES)
        self.sources = SourceManager(cfg, self.symbols, self.av, on_tick=self._on_live_tick,
                                     on_provider_change=self._on_provider_change, trace=self.trace,
                                     schwab=self.schwab_auth, on_quote=self._on_quote, etrade=self.etrade_auth)
        self.news_hub = NewsHub(cfg, self.symbols, self.av)
        self.signals = SignalHub(cfg, self.symbols)
        self.prices = {s: {"price": None, "bid": None, "ask": None, "ts": None} for s in self.symbols}
        self.tick_counts = {s: 0 for s in self.symbols}
        self.bars: dict[str, deque[Bar]] = {s: deque(maxlen=cfg.chart_bars) for s in self.symbols}
        self.option_iv: dict[str, float] = {}             # raw ATM implied vol per name, for the policy's decay and breakeven arithmetic
        self._cooldown: dict[str, int] = {}               # bar index of a name's last exit (stop, target, time limit)
        self._day_open_equity: tuple[str, float] | None = None   # (ET date, book equity at the day's first bar) for the daily loss limit
        self._book_equity = 0.0
        self._streak: dict[str, tuple[int, int]] = {}    # symbol -> (raw side, consecutive bars) for the confirmation gate
        self._held_since: dict[str, int] = {}             # symbol -> bar index a nonzero paper target began
        self._last_target: dict[str, float] = {}
        self.paper = PaperBook(cfg.paper_start, cfg.cost_bps, min_trade_frac=cfg.paper_min_trade_frac,
                               min_notional=float(getattr(cfg, "min_position_dollars", 0.0) or 0.0),
                               option_commission=cfg.option_commission, option_max_frac=cfg.option_max_frac,
                               put_min_hold_s=cfg.put_min_hold_bars * cfg.bar_seconds,
                               straddle_min_hold_s=cfg.straddle_hold_bars * cfg.bar_seconds)   # $100k paper book (shorts via puts)
        self.portfolio: dict = {}                               # Schwab account positions (read-only), for the Portfolio tab
        self._cc_last = 0.0                                     # last covered-call opportunity refresh
        self._cc_cache: list = []                               # cached covered-call opportunities (persist across 60s position refreshes)
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
        await asyncio.to_thread(self._save, True)

    async def reconfigure(self) -> None:
        """Rebuild for the current active universe (all_symbols minus muted) and restart
        the lifecycle. Muted / removed symbols then cost no compute or API calls."""
        if self._reconfiguring:
            return
        self._reconfiguring = True
        try:
            self.muted &= set(self.all_symbols)
            if self._backtesting:                                   # its net shares the GPU: a benchmark next to it reads nonsense peaks
                self._backtest_stop = True
                self.trace.emit("system", "universe change: stopping the backtest first", "act")
            for t in self._tasks:
                t.cancel()
            if self._lifecycle:
                self._lifecycle.cancel()
            try:
                await self.sources.stop()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.15)
            for _ in range(900):                                    # the worker stops at its next day boundary
                if not self._backtesting:
                    break
                await asyncio.sleep(1)
            self._tasks = []
            self.status = "starting"
            prev = (list(self.symbols), list(self._built_universe or self.all_symbols), dict(self.base), list(self.cfg.symbols))
            try:
                await asyncio.to_thread(self._save, True)           # the replay in memory is hours of paging; keep it
            except Exception:  # noqa: BLE001
                log.exception("save before rebuild failed")
            self.learner.release_replay()                            # the new learner allocates its own; load() migrates from disk
            self.learner.release_cache()
            try:
                await asyncio.to_thread(self._build, self._active())   # benchmark/build off the event loop
                self._built_universe = list(self.all_symbols)          # the full list the model now matches
                self.trace.emit("system", f"universe rebuilt: {len(self.symbols)} active "
                                          f"({len(self.muted)} muted) — {', '.join(self.base[s] for s in self.symbols)}", "act")
            except Exception as e:  # noqa: BLE001
                # A rebuild that fails (no memory to benchmark, a bad symbol) must not leave the
                # engine with every loop cancelled and no lifecycle, as it did on 2026-10-10.
                # Put the previous universe back and resume with the model we still have.
                log.exception("universe rebuild failed")
                self.symbols, self.all_symbols, self.base, self.cfg.symbols = prev
                self.muted &= set(self.all_symbols)
                self._interesting = {k: v for k, v in self._interesting.items() if k in self.all_symbols}
                self.trace.emit("system", f"universe rebuild failed: {type(e).__name__}: {str(e)[:160]}; "
                                          f"keeping the previous {len(self.symbols)} names", "error")
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

    def _live_prices(self) -> dict:
        return {s: self.prices[s]["price"] for s in self.symbols if self.prices.get(s) and self.prices[s].get("price")}

    def status_dict(self) -> dict:
        providers = self.sources.provider_map()
        active = sorted({p for p in providers.values() if p})
        return {"phase": self.status, "feed": ", ".join(active) or "none", "providers": providers,
                "started": self.started, "now": time.time(), "bar_index": self.bar_index,
                "pending": len(self.pending), "learning": self.learning_enabled, "subscribers": len(self.subs),
                "market_status": self.market_status,
                "stalled": bool(getattr(self, "_stalled", False)),
                "last_bar_age": round(time.time() - getattr(self, "_last_row_at", time.time())),
                "target": self.target_info()}

    def target_info(self) -> dict:
        """What the net forecasts: each name's return, or its return relative to the index."""
        L = self.learner
        return {"mode": L.target_mode, "configured": str(self.cfg.target), "index": [self.symbols[c] for c in L.cols],
                "configured_index": [s.strip().upper() for s in str(self.cfg.index_symbols).split(",") if s.strip()],
                "hedge": bool(L.cols and self.cfg.hedge_index)}

    def controls(self) -> dict:
        return {k: getattr(self.cfg, k) for k in TUNABLE} | {"learning": self.learning_enabled, "burry": self.burry_enabled, "brief_enabled": self.brief_enabled}

    def _key_values(self) -> dict:
        cfg = self.cfg
        ak, asec, acb = cfg.schwab_creds
        kid, sec = (tuple(cfg.alpaca_creds) + ("", ""))[:2]
        ck, cs = (tuple(cfg.etrade_creds) + ("", ""))[:2]
        # One entry per KEY_SERVICES id, or the panel shows a saved key as "not set" forever
        # (Alpaca, Tradier and E*TRADE were missing until 2026-10-10).
        return {"alphavantage": {"key": cfg.av_key}, "finnhub": {"key": cfg.finnhub_key},
                "eodhd": {"key": cfg.eodhd_key}, "fmp": {"key": cfg.fmp_key}, "schwab": {"app_key": ak, "app_secret": asec},
                "alpaca": {"key_id": kid, "secret_key": sec}, "tradier": {"token": cfg.tradier_token},
                "etrade": {"consumer_key": ck, "consumer_secret": cs}}

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
        path = Path(svc["file"])
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
                if service == "schwab":
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
            if service == "finnhub":
                # Everything else that copied the key at construction: the market radar (sector
                # fill, index and sector ETF quotes) and the fundamentals signal. The market-status
                # poll reads cfg on every pass, so it starts by itself.
                self.signals.market.finnhub_key = vals["key"]
                for p in self.signals.providers.values():
                    if isinstance(p, FundamentalsProvider):
                        p.key = vals["key"]
        elif service == "schwab":
            ak = vals.get("app_key") or self.schwab_auth.app_key
            asec = vals.get("app_secret") or self.schwab_auth.app_secret
            cfg.schwab_creds = (ak, asec, self.schwab_auth.callback)
            self.schwab_auth.app_key, self.schwab_auth.app_secret = ak, asec
        elif service == "alpaca":
            cfg.alpaca_creds = (vals.get("key_id", ""), vals.get("secret_key", ""))
            src = self.sources.sources.get("alpaca")
            if src:
                src.kid, src.sec = cfg.alpaca_creds
                if src.kid and src.sec:
                    src.note = "keys set; polling"
                    self.sources.toggle("alpaca", True)
        elif service == "tradier":
            cfg.tradier_token = vals.get("token", "")
            src = self.sources.sources.get("tradier")
            if src:
                src.token = cfg.tradier_token
                if src.token:
                    src.note = "token set; polling"
                    self.sources.toggle("tradier", True)
        elif service == "etrade":
            ck = vals.get("consumer_key") or self.etrade_auth.ck
            cs = vals.get("consumer_secret") or self.etrade_auth.cs
            cfg.etrade_creds = (ck, cs)
            self.etrade_auth.ck, self.etrade_auth.cs = ck, cs

    def snapshot(self) -> dict:
        cfg = self.cfg
        return {
            "type": "snapshot",
            "config": {
                "symbols": self.symbols, "bar_seconds": cfg.bar_seconds, "horizon": cfg.horizon, "window": cfg.window, "max_universe": cfg.max_universe, "chart_bars": cfg.chart_bars,
                "quantiles": list(cfg.quantiles), "min_labels": cfg.min_labels, "features": FEATURES,
                "model": {"params": self.learner.n_params, "members": self.learner.members, "d_model": cfg.d_model, "experts": cfg.n_experts,
                          "dilations": list(cfg.dilations), "heads": cfg.n_heads, "window": cfg.window,
                          "receptive_field": self.learner.model.receptive_field,
                          "device": cfg.device, "ms_per_step": (self.autotune_info or {}).get("ms_per_step"),
                          "preset": (self.autotune_info or {}).get("preset")},
            },
            "status": self.status_dict(),
            "backtest": self.backtest_summary(),         # a page opened mid-run sees the run, not "no run yet"
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
            "holds": summarize_holds(self.latest),
            "universe": self.all_symbols,
            "prices": self.prices,
            "bars": {s: [b.to_json() for b in self.bars.get(s, ())] for s in self.symbols},
            "latest": self.latest,
            "gate": self.gate,
            "outcomes": {s: list(self.outcomes.get(s, ())) for s in self.symbols},
            "metrics": self.metrics.as_dict(),
            "history": {k: list(v) for k, v in self.history.items()},
            "log": list(self.log),
            "trace": self.trace.recent(),
            "operator": self.operator_state(),
            "news": self.news,
            "brief": self.brief,
            "paper": self.paper.snapshot(self._live_prices()),
            "portfolio": self.portfolio,
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

    def _load_norm(self, st) -> None:
        """Restore the feature normalizer from a checkpoint, rows matched by name so a universe
        that changed since the save keeps every retained name's statistics; a state saved before
        it carried names (0.5.15) is named by the checkpoint's own symbol list. Whatever the state,
        the start goes on: before this a positional load died on a broadcast error after the
        universe loop's first promotion (found in Steel 2026-10-11)."""
        try:
            st = dict(st) if isinstance(st, dict) else {}
            if not st.get("symbols") and self.learner.loaded_symbols:
                st["symbols"] = list(self.learner.loaded_symbols)
            ok = self.features.norm.load(st, list(self.symbols)) if st else False
        except Exception as e:  # noqa: BLE001 -- normalization statistics must never kill the start
            log.warning("could not restore the normalization state (%s: %s); starting fresh", type(e).__name__, e)
            ok = False
        if not ok:
            self.trace.emit("system", "normalization state could not be restored; features re-standardize from fresh", "warn")

    async def _start(self) -> None:
        cfg = self.cfg
        extra = self.learner.load(cfg.state_dir)
        if extra is not None:
            if "norm" in extra:
                self._load_norm(extra["norm"])
            if extra.get("meta") is not None:
                try:
                    ok = self.meta.load(extra["meta"])
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    self.trace.emit("system", "meta-labeler state could not be restored; it relearns from live outcomes", "warn")
            self.metrics.labels = self.learner.labels
            self.metrics.steps = self.learner.steps
            md = extra.get("metrics") or {}
            m = self.metrics
            for k in self._PERSIST_METRICS:
                if md.get(k) is not None:
                    setattr(m, k, md[k])
            # Calibration state is versioned: values saved by an older calibration regime measured
            # the wrong thing (the pre-batch updater pinned the scales at their clips and poisoned
            # the EMAs), so a pre-v2 checkpoint re-enters with the fresh-model priors and lets live
            # labels re-earn everything. Bump CALIB_V whenever the calibration math changes shape.
            if int(md.get("calib_v") or 1) < self.CALIB_V:
                m.band_scale, m.p_scale = 2.0, 0.3
                m.hit_ema = m.coverage_ema = None
            for s, outs in (extra.get("outcomes") or {}).items():   # per-name track records survive a restart (the skill gate needs them)
                if s in self.outcomes:
                    self.outcomes[s] = deque(list(outs)[-40:], maxlen=40)
            for sym, v in (md.get("pnl_by_symbol") or {}).items():
                if sym in m.pnl_by_symbol:
                    m.pnl_by_symbol[sym] = v
            m.trusted = m.live_labels >= cfg.min_labels
            if extra.get("paper"):
                self.paper.load_state(extra["paper"], set(self.symbols))
            self.trace.emit("system", f"restored checkpoint from {cfg.state_dir}: {self.learner.steps} steps, "
                                      f"{self.learner.labels} labels, {m.live_labels}/{cfg.min_labels} live labels, "
                                      f"{self.learner.size} windows in replay")
        self._built_universe = list(self.all_symbols)
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
        self._deep_cutoff = cutoff                     # deep-backfill loop pages further back from here
        self._news_task = asyncio.create_task(self._news_history_job())   # dated news for the history the deep loop pages
        if self.features.wants_events:
            self._events_task = asyncio.create_task(self._events_job())    # and the earnings calendar, when the set reads one
        held = self.learner.rt[:self.learner.size]
        held = held[held > 0]
        if self.learner.migrated_new:                  # new names: walk the whole history again; fill() takes their columns
            self.trace.emit("system", f"universe changed: re-paging history to fill {len(self.learner.migrated_new)} new "
                                      f"name(s) into the replay ({', '.join(self.learner.migrated_new[:12])}"
                                      f"{'…' if len(self.learner.migrated_new) > 12 else ''})", "act")
        elif len(held):                                # resume where the last run's paging stopped, not from today
            self._deep_cutoff = min(self._deep_cutoff, float(held.min()))
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
        self.features.news_history = self.news_history   # the startup days read the dated store, as paged history does
        try:
            self._ingest_history(ticks)
        finally:
            self.features.news_history = None             # live bars read the skim
        self._startup_span = (cutoff, time.time())
        await self._warmup()
        self._seed_forecast()
        self.status = "live"
        self._publish(self.snapshot())   # full resync: clients that connected during backfill/warmup now get the built bars (no reload needed)
        self.trace.emit("system", "live: streaming ticks, forecasting every bar, learning as labels mature")
        self.sources.start()
        self._tasks = [asyncio.create_task(c) for c in
                       (self._clock(), self._ticker(), self._checkpoints(), self._news_loop(), self._signals_loop(), self._market_loop(), self._brief_loop(), self._market_status_loop(), self._portfolio_loop(), self._option_features_loop(), self._deep_backfill_loop(), self._idle_train_loop(), self._backtest_loop(), self._universe_loop())]
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
        blind = [s for s in self.symbols if s not in first]
        if blind:                                    # a symbol with zero history would block EVERY joint row -> no bars at all
            self.trace.emit("feed", f"no backfill history for {', '.join(self.base[s] for s in blind)}; "
                                    f"flat-seeding so they don't stall the rest of the universe", "warn")
            for s in blind:
                first[s] = 0.0                       # zero seed: flat bars read as all-zero neutral features, and the first real
                                                     # bar has prev==0 so its return is 0 (no fake spike when the symbol recovers)
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
            self.bars.setdefault(s, deque(maxlen=self.cfg.chart_bars)).append(row[s])
        closes = np.array([row[s].close for s in self.symbols], dtype=np.float32)
        real_now = np.array([row[s].volume > 0.0 for s in self.symbols])
        self._mature(closes, real_now)
        x = self.features.push(row)
        if x is not None:
            self.pending.append(Pending(self.bar_index, row[self.symbols[0]].ts, x, closes, real_now, None, {}))

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

    async def _deep_backfill_loop(self) -> None:
        """While the market is closed, page history back beyond the startup backfill (in
        ~10-day chunks, down to deep_backfill_days), add the labeled windows to the replay
        buffer, and train on them. Idle hours become training time. Runs nothing while the
        market is open, so it never competes with live forecasting for the GPU."""
        wait = 600.0
        while True:
            await asyncio.sleep(wait)
            wait = 600.0                                  # nothing to fetch: check back every 10 min
            cfg = self.cfg
            days = float(getattr(cfg, "deep_backfill_days", 0) or 0)
            target = time.time() - days * 86400.0
            if (days <= 0 or self.market_status.get("isOpen") is not False
                    or not self.learning_enabled or self._deep_cutoff <= target):
                continue
            if self._backtesting:                         # the backtest owns the GPU and the replay must not grow under it
                wait = 60.0
                continue
            cap = self.learner.rx.shape[0]
            if self.learner.size >= cap - 800:            # a chunk is ~700 windows; the ring must not wrap onto the newest
                if not self._ring_full_said:
                    self.trace.emit("feed", f"deep backfill stops: the replay is full ({self.learner.size} of {cap} windows)", "info")
                    self._ring_full_said = True
                continue
            free = free_memory_gb(str(self.learner.device))
            if free is not None and free < float(cfg.memory_floor_gb) + 4.0:
                if not self._lowmem_said:
                    self.trace.emit("feed", f"deep backfill paused: {free:.0f} GB free, floor {cfg.memory_floor_gb:g} GB", "warn")
                    self._lowmem_said = True
                wait = 300.0
                continue
            self._lowmem_said = False
            wait = 150.0                                  # more history to page: a 211-symbol burst every 150 s stays under Schwab's 120 req/min
            until = self._deep_cutoff
            cutoff = max(target, until - 10 * 86400.0)
            fetching = any(t is not None and not t.done() for t in (self._news_task, self._events_task))
            if fetching and not self._history_ready(cutoff - WARM_S, until):
                wait = 60.0                               # the chunk's news is still on its way: page it with the news, not without
                continue
            if self._startup_span and not fetching:
                lo, hi = self._startup_span
                self._startup_span = None
                try:
                    ticks = await self.sources.backfill_range(lo - WARM_S, hi)
                    n = await asyncio.to_thread(self._ingest_offline, ticks, lo, True) if ticks else 0
                    self.trace.emit("learn", f"re-paged the startup days with their news: {n} windows rewritten", "info")
                except Exception as e:  # noqa: BLE001
                    self.trace.emit("learn", f"re-paging the startup days failed: {type(e).__name__}: {str(e)[:80]}", "warn")
                continue
            try:
                ticks = await self.sources.backfill_range(cutoff - WARM_S, until)   # a warm prefix the builder walks but does not emit
                self._deep_cutoff = cutoff
                if not ticks:
                    self.trace.emit("feed", "deep backfill: no history returned for "
                                            f"{datetime.fromtimestamp(cutoff):%b %d} – {datetime.fromtimestamp(until):%b %d}", "warn")
                    continue
                self._deep_filled = 0
                added = await asyncio.to_thread(self._ingest_offline, ticks, cutoff)
                filled = self._deep_filled
                res = await asyncio.to_thread(self.learner.train_steps, 30) if (added or filled) else None
                if res:
                    self._absorb_train(res)
                self.learner.release_cache()
                self.trace.emit("learn", f"deep backfill {datetime.fromtimestamp(cutoff):%b %d} – "
                                         f"{datetime.fromtimestamp(until):%b %d}: {len(ticks)} ticks, "
                                         f"{added} windows added ({self.learner.size} in replay)" +
                                         (f", {filled} columns filled for new names" if filled else "") +
                                         (f", 30 training steps (loss {res['loss']:.3f})" if res else ""))
                self._publish({"type": "metrics", "metrics": self.metrics.as_dict(), "history": self._drain_history()})
            except Exception as e:  # noqa: BLE001
                self.trace.emit("learn", f"deep backfill failed: {type(e).__name__}: {str(e)[:80]}", "warn")

    def universe_candidates(self) -> dict[str, float]:
        """Interesting names and a score on one 0-100 scale, from three organic sources: the
        market radar (move weighted by dollar volume, liquid names only), tickers the press is
        naming (headline mentions, only for symbols known elsewhere) and the tracked funds'
        13F books (position weight). The best rank across sources wins, with a small bonus for
        showing up in more than one."""
        cfg = self.cfg
        radar = (self.signals_state.get("market", {}) or {}).get("radar", []) or []
        known = {str(r.get("symbol") or "").upper() for r in radar}
        try:
            known |= set(json.load(open(os.path.join(cfg.state_dir, "sectors.json"))).keys())
        except (OSError, ValueError):
            pass
        movers = []
        for r in radar:
            sym = str(r.get("symbol") or "").upper()
            px, vol, chg = float(r.get("price") or 0), float(r.get("vol") or 0), abs(float(r.get("chg") or 0))
            if not sym or "." in sym or not sym.isalnum() or len(sym) > 5 or px < cfg.min_price:
                continue
            dv = px * vol
            if dv < float(cfg.interesting_min_dollar_volume):
                continue
            movers.append((chg * math.log10(max(dv, 10.0)), sym))
        movers.sort(reverse=True)
        gurus = []
        for g in self.signals_state.get("gurus") or []:
            for h in g.get("holdings") or []:
                t = str(h.get("ticker") or "").upper()
                if t and t.isalnum() and len(t) <= 5:
                    gurus.append((float(h.get("weight") or 0), t))
                    known.add(t)
        gurus.sort(reverse=True)
        mentions = headline_tickers((self.news or {}).get("headlines") or [], known)
        named = sorted(((n, s) for s, n in mentions.items() if n >= 2), reverse=True)
        span = max(10, int(cfg.interesting_count) * 3)
        sources = [rank_scores([s for _, s in movers], span), rank_scores([s for _, s in named], span),
                   rank_scores([s for _, s in gurus], span)]
        out: dict[str, float] = {}
        for sc in sources:
            for s, v in sc.items():
                if v <= 0:
                    continue
                seen_before = s in out
                out[s] = max(out.get(s, 0.0), v) + (5.0 if seen_before else 0.0)
        return out

    async def _universe_loop(self) -> None:
        """Once a day after the close: the universe becomes the configured core, every holding,
        and the most interesting names (pick_universe: promotion by rank, demotion after a
        grace period). The rebuild keeps retained names' embedding rows and replay columns and
        re-pages history to fill the new ones."""
        while True:
            await asyncio.sleep(60)
            cfg = self.cfg
            if not getattr(cfg, "universe_refresh", True) or self._reconfiguring or self._backtesting:
                continue
            if self.market_status.get("isOpen") is not False:
                continue
            now = datetime.now(ZoneInfo("America/New_York"))
            today = now.date().isoformat()
            if self._universe_refreshed == today or (now.weekday() < 5 and now.hour < 16):
                continue
            radar = (self.signals_state.get("market", {}) or {}).get("radar", []) or []
            if len(radar) < 100 or not (self.news or {}).get("t") or not (self.signals_state.get("gurus") or []):
                continue                                  # the radar, the news skim and the 13F books must have reported
            cands = self.universe_candidates()
            if len(cands) < 20:
                continue
            new, rec, promoted, demoted = pick_universe(
                self.core_symbols, self._held, self.all_symbols, cands, self._interesting,
                count=int(cfg.interesting_count), max_universe=int(cfg.max_universe), today=today,
                grace_days=int(cfg.interesting_grace_days), min_tenure_days=int(cfg.interesting_min_tenure_days))
            self._universe_refreshed = today
            self._interesting = rec
            if not promoted and not demoted and new == self.all_symbols:
                self.trace.emit("system", f"universe refresh: no change ({len(new)} names; "
                                          f"{len(cands)} candidates scored)", "info")
                continue
            self.all_symbols = new
            self.muted -= set(demoted)
            self.trace.emit("system", f"universe refresh: promoted {len(promoted)} ({', '.join(promoted[:10])}"
                                      f"{'…' if len(promoted) > 10 else ''}), demoted {len(demoted)} "
                                      f"({', '.join(demoted[:10])}{'…' if len(demoted) > 10 else ''}); {len(new)} names: "
                                      f"{len(self.core_symbols)} core, {len(set(self._held) - set(self.core_symbols))} holdings, "
                                      f"{len(new) - len(set(self.core_symbols) | set(self._held))} interesting", "act")
            asyncio.get_running_loop().create_task(self.reconfigure())
            return                                        # the rebuild cancels this task; the new lifecycle starts another

    def backtest_summary(self) -> dict:
        """The report the dashboard shows: totals (live partials while a run is on), the day
        rows, the best and worst names, and the run's configuration."""
        b = self.backtest or {}
        pr = self.backtest_progress or {}
        syms = [(k, v) for k, v in (b.get("symbols") or {}).items() if v.get("signals")]
        syms.sort(key=lambda kv: kv[1]["net_bps"])
        return {"progress": pr, "ran_at": b.get("ran_at"), "elapsed_s": b.get("elapsed_s"), "stopped": b.get("stopped"),
                "error": b.get("error"), "config": b.get("config"),
                "totals": pr.get("totals") if pr.get("running") else b.get("totals"),
                "days": (pr.get("days") or []) if pr.get("running") else (b.get("days") or [])[-260:],
                "baseline": ((pr.get("done") or [None])[0] if pr.get("running") else b.get("baseline")),   # a finished variant shows under the one still running
                "best": [dict(sym=k, **v) for k, v in syms[-8:][::-1]], "worst": [dict(sym=k, **v) for k, v in syms[:8]],
                "sweep": self.sweep or None}

    def _backtest_due(self, now: datetime | None = None) -> bool:
        """Once per weekend (Saturday or Sunday ET with the market closed), or on request. A run
        that stopped before scoring a day (a memory squeeze at the start) does not count as the
        weekend's run, so the schedule tries again."""
        if self._backtest_requested:
            return True
        if not getattr(self.cfg, "backtest_weekends", True):
            return False
        now = now or datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() < 5 or self.market_status.get("isOpen") is not False:
            return False
        last = self.backtest or {}
        if last.get("stopped") and not (last.get("totals") or {}).get("days"):
            return True
        return now.timestamp() - float((self.backtest or {}).get("ran_at") or 0) > 4 * 86400

    async def _backtest_loop(self) -> None:
        """Weekend walk-forward backtest (flint/backtest.py) on the live preset, in process so
        the replay is shared and the idle trainer yields the GPU. Waits for the deep backfill
        to finish paging so the ring cannot move under the run. Never touches the live model."""
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(30)
            if self._backtesting or not self._backtest_due():
                continue
            cfg = self.cfg
            if self._paging_pending():
                if self._backtest_requested and not self._backtest_waiting:
                    self._backtest_waiting = True
                    self.trace.emit("learn", "backtest queued: waiting for the deep backfill to finish paging", "info")
                continue                                  # still paging history: the replay must not grow under a run
            self._backtest_waiting = False
            L = self.learner
            n = L.size
            if n < 64 or float(L.rt[:n].max()) <= 0:
                if self._backtest_requested:
                    self._backtest_requested = False
                    self.trace.emit("learn", "backtest: no timestamped replay windows yet", "warn")
                continue
            self._backtest_requested = False
            self._backtest_stop = False
            self._backtest_stop_reason = ""
            # Own the GPU before measuring it: from here the idle trainer and the deep loop start
            # nothing new, and the chunk one of them may have in flight finishes first, or the free
            # figure is read with its working set still up and the run is sized into a squeeze
            # (two such starts stopped at 1 GB free on 2026-10-10).
            self._backtesting = True
            await asyncio.to_thread(L.wait_idle)
            L.release_cache()
            dev = str(L.device)
            free = free_memory_gb(dev)
            live_peak = float((self.autotune_info or {}).get("peak_gb") or 0)
            choice = backtest_preset_for(free, int(cfg.d_model), live_peak, floor=float(cfg.memory_floor_gb))
            if choice is None:
                self.trace.emit("learn", f"backtest skipped: {free:.0f} GB free leaves no room for any net next to the live model "
                                         f"(floor {cfg.memory_floor_gb:g} GB)", "warn")
                self.backtest_progress = {"running": False}
                self._backtesting = False
                continue
            name, d_model, dil, n_exp, n_heads, _win = choice
            cfg_bt = copy.copy(cfg)
            cfg_bt.d_model, cfg_bt.dilations, cfg_bt.n_experts, cfg_bt.n_heads = int(d_model), tuple(dil), int(n_exp), int(n_heads)
            preset = str(name) + ("" if d_model == cfg.d_model else f" (sized to {free:.0f} GB free; live is "
                                                                     f"{(self.autotune_info or {}).get('preset') or cfg.d_model})")
            variants = {"absolute": ["absolute"], "relative": ["relative"],
                        "both": ["absolute", "relative"]}.get(str(cfg.backtest_target).lower(), ["absolute"])
            if "relative" in variants and not index_columns(self.symbols, cfg.index_symbols):
                self.trace.emit("learn", f"backtest: none of {cfg.index_symbols} is modeled, so the relative target cannot run; "
                                         f"absolute only", "warn")
                variants = ["absolute"]
            hours_each = float(cfg.backtest_hours) / len(variants) if float(cfg.backtest_hours) > 0 else 0.0
            started = time.time()
            self.backtest_progress = {"running": True, "day": 0, "days": 0, "started": started,
                                      "variant": variants[0], "variants": variants}
            self.trace.emit("learn", f"backtest: walking {n} replay windows day by day with a fresh {preset} "
                                     f"(budget {cfg.backtest_hours:g} h, {cfg.backtest_night_epochs:g} night epochs/day, "
                                     f"cost {cfg.backtest_cost_bps:g} bps; target{'s' if len(variants) > 1 else ''} "
                                     f"{' then '.join(variants)}); live training waits", "act")
            self._publish({"type": "backtest", "backtest": self.backtest_summary()})

            def should_stop() -> bool:
                if self._backtest_stop:
                    return True
                low = memory_low(dev, float(cfg.memory_floor_gb))
                if low is not None:
                    self._backtest_stop_reason = f"memory low ({low:.0f} GB free)"
                    self._backtest_stop = True
                    return True
                return False

            def prog(day: int, total: int, totals: dict, info: dict | None = None) -> None:
                self.backtest_progress = {**self.backtest_progress, "running": True, "day": day, "days": total, "started": started,
                                          "totals": totals, **(info or {})}
                loop.call_soon_threadsafe(self._publish, {"type": "backtest", "backtest": self.backtest_summary()})

            def job() -> dict:
                # Runs on the worker thread: the variants back to back on one net's worth of memory,
                # each on its share of the budget; the second report carries the first as its baseline.
                reports: list[dict] = []
                done: list[dict] = []                                  # finished variants, carried through the progress so the tab keeps them
                for vi, variant in enumerate(variants):
                    if self._backtest_stop:
                        break
                    self.backtest_progress = {"running": True, "day": 0, "days": 0, "started": started,
                                              "variant": variant, "variants": variants, "done": done}
                    if len(variants) > 1:
                        loop.call_soon_threadsafe(self.trace.emit, "learn",
                                                  f"backtest {vi + 1}/{len(variants)}: the {variant} target", "info")
                    reports.append(run_backtest(
                        cfg_bt, L.rx[:n], L.ry[:n], L.rmask[:n], L.rt[:n], device=dev,
                        symbols=list(self.symbols), progress=prog, should_stop=should_stop,
                        hours=hours_each, days=int(cfg.backtest_days), warm_days=int(cfg.backtest_warm_days),
                        cost_bps=float(cfg.backtest_cost_bps), hedge_cost_bps=float(cfg.hedge_cost_bps),
                        preset=preset, night_epochs=float(cfg.backtest_night_epochs),
                        target=variant, index_symbols=str(cfg.index_symbols),
                        beta_windows=int(cfg.beta_windows), beta_shrink=float(cfg.beta_shrink),
                        store=os.path.join(cfg.state_dir, "backtest_forecasts"),
                        members=int(getattr(cfg, "backtest_members", 1) or 1)))
                    done.append(variant_summary(reports[-1]))
                    loop.call_soon_threadsafe(self._publish, {"type": "backtest", "backtest": self.backtest_summary()})
                return combine_reports(reports)

            def finish(f: asyncio.Future) -> None:
                # Runs on the event loop when the worker thread exits, whether or not the loop task
                # that started it is still alive: a universe rebuild cancels this loop, but the
                # thread keeps going until its next day boundary and its report is still wanted.
                self._backtesting = False
                L.release_cache()
                self.backtest_progress = {"running": False}
                res = None
                try:
                    res = f.result()
                except Exception as e:  # noqa: BLE001
                    log.error("backtest failed", exc_info=e)
                    self.trace.emit("learn", f"backtest failed: {type(e).__name__}: {str(e)[:120]}", "error")
                if res:
                    self._finish_backtest(res)
                self._publish({"type": "backtest", "backtest": self.backtest_summary()})

            fut = loop.run_in_executor(None, job)
            fut.add_done_callback(finish)
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                self._backtest_stop = True                          # the worker checks at each day boundary; finish() runs when it exits
                raise
            except Exception:  # noqa: BLE001 -- reported by finish()
                pass

    def _paging_pending(self) -> bool:
        """Whether the deep backfill still has chunks to add: history left to page and room in the
        ring for a chunk. A full ring ends paging early (the deep loop stops within a chunk of
        full), so the backtest must not wait on the cutoff alone: before 0.5.14 it did, and a
        queued run waited forever once the replay filled before the configured depth."""
        days = float(getattr(self.cfg, "deep_backfill_days", 0) or 0)
        if days <= 0 or self._deep_cutoff <= time.time() - days * 86400.0:
            return False
        L = self.learner
        return L.size < L.rx.shape[0] - 800

    def _finish_backtest(self, res: dict) -> None:
        cfg = self.cfg
        self.backtest = res
        try:
            tmp = os.path.join(cfg.state_dir, "backtest.json.tmp")
            with open(tmp, "w") as f:
                json.dump(clean(res), f)
            os.replace(tmp, os.path.join(cfg.state_dir, "backtest.json"))
        except OSError as e:
            log.warning("could not save backtest.json: %s", e)
        t = res.get("totals") or {}
        _sharpe = lambda tt: "·" if (tt or {}).get("sharpe") is None else f"{tt['sharpe']:.1f}"  # noqa: E731
        if res.get("error"):
            self.trace.emit("learn", f"backtest: {res['error']}", "warn")
        else:
            how = ("stopped (" + self._backtest_stop_reason + ")" if res.get("stopped") and self._backtest_stop_reason
                   else "stopped" if res.get("stopped") else "done")
            self.trace.emit("learn", f"backtest {how} in {res.get('elapsed_s', 0) / 60:.0f} min: "
                                     f"{t.get('days', 0)} days, net {t.get('net_bps', 0):+.0f} bps "
                                     f"({t.get('net_bps_per_day', 0):+.1f}/day, Sharpe {_sharpe(t)}), "
                                     f"hit {((t.get('hit_rate') or 0) * 100):.0f}%, coverage {((t.get('coverage') or 0) * 100):.0f}%, "
                                     f"{t.get('signals', 0)} signals; always-long {t.get('long_bps', 0):+.0f} bps"
                                     + (f" [{(res.get('config') or {}).get('target')} target]" if (res.get('config') or {}).get('target') else "")
                                     + ((lambda b: f"; the {b.get('target')} target made net {b.get('totals', {}).get('net_bps', 0):+.0f} bps "
                                                   f"(Sharpe {_sharpe(b.get('totals'))})")
                                        (res["baseline"]) if res.get("baseline") else ""), "act")
        path = (res.get("config") or {}).get("forecasts")
        if path and not res.get("error"):
            # The policy sweep over the forecasts this run stored: seconds of arithmetic on the
            # CPU, so it follows every run; the result is a proposal on the dashboard, never applied.
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self._sweep_job, str(path)).add_done_callback(self._finish_sweep)
            except RuntimeError:                                    # no loop (a test calling this directly)
                self._finish_sweep_result(self._sweep_job(str(path)))

    def _sweep_job(self, path: str) -> dict:
        from .sweep import load_forecasts, run_sweep
        return run_sweep(load_forecasts(path), copy.copy(self.cfg), float(self.cfg.backtest_cost_bps))

    def _finish_sweep(self, f) -> None:
        try:
            res = f.result()
        except Exception as e:  # noqa: BLE001
            log.error("policy sweep failed", exc_info=e)
            self.trace.emit("learn", f"policy sweep failed: {type(e).__name__}: {str(e)[:120]}", "error")
            return
        self._finish_sweep_result(res)

    def _finish_sweep_result(self, res: dict) -> None:
        self.sweep = res
        try:
            tmp = os.path.join(self.cfg.state_dir, "sweep.json.tmp")
            with open(tmp, "w") as f:
                json.dump(clean(res), f)
            os.replace(tmp, os.path.join(self.cfg.state_dir, "sweep.json"))
        except OSError as e:
            log.warning("could not save sweep.json: %s", e)
        if res.get("error"):
            self.trace.emit("learn", f"policy sweep: {res['error']}", "warn")
        else:
            cur, prop = res.get("current") or {}, res.get("proposal")
            fmt = lambda r: f"{(r or {}).get('net_bps_per_day', 0):+.1f}/day"  # noqa: E731
            text = (f"policy sweep over {res.get('days')} days ({res.get('evaluations')} settings in {res.get('elapsed_s', 0):.0f} s): "
                    + (f"proposes {', '.join(f'{k} {v}' for k, v in prop['knobs'].items())}; tune {fmt(prop['tune'])} vs "
                       f"{fmt(cur.get('tune'))} now, confirm {fmt(prop['confirm'])} vs {fmt(cur.get('confirm'))}; "
                       + ("worth a look on the Backtest tab" if res.get("accepted") else "not confirmed, so not proposed")
                       if prop else "no setting beat the current gates"))
            self.trace.emit("learn", text, "act" if res.get("accepted") else "info")
        self._publish({"type": "backtest", "backtest": self.backtest_summary()})

    async def _idle_train_loop(self) -> None:
        """While the market is closed, train on the replay buffer instead of leaving the GPU
        idle between deep-backfill chunks. The budget is idle_train_epochs passes over the
        replay per closed session (recomputed as backfill grows the replay), so it is bounded
        by data, not by hours, and cannot grind the same windows all night. Small chunks keep
        the learner lock short so pre-market forecasts are not held up. Never runs while the
        market is open; the open resets the budget."""
        done = 0
        self._idle_was_open = None
        wait = 20.0
        held: list[float] = []                            # held-out pinball at each check this session
        while True:
            await asyncio.sleep(wait)
            wait = 20.0                                   # idle poll; drops to ~0 while there is budget to burn
            cfg = self.cfg
            is_open = self.market_status.get("isOpen")
            if is_open is None or (is_open and not getattr(cfg, "train_in_session", False)):
                continue                                  # unknown status, or open and session training is off
            if self._backtesting:
                continue                                  # the weekend backtest owns the GPU
            free = free_memory_gb(str(self.learner.device))
            if free is not None and free < float(cfg.memory_floor_gb) + 4.0:
                if not self._lowmem_said:
                    self.trace.emit("learn", f"idle training paused: {free:.0f} GB free, floor {cfg.memory_floor_gb:g} GB", "warn")
                    self._lowmem_said = True
                self.learner.release_cache()
                continue
            if is_open != self._idle_was_open:            # each session (open or closed) gets its own budget
                done, self._idle_was_open, held = 0, is_open, []
            epochs = float(getattr(cfg, "idle_train_epochs", 0) or 0)
            size = self.learner.size
            if epochs <= 0 or not self.learning_enabled or size < 64:
                continue
            budget = int(epochs * size / max(1, cfg.batch_size))
            if done >= budget:
                continue
            wait = 0.5                                    # back to back: keep the GPU busy, yield to the event loop
            chunk = min(4, budget - done)
            res = await asyncio.to_thread(self.learner.train_steps, chunk)
            if not res:
                continue
            done += chunk
            self._absorb_train(res)
            if done % 40 < chunk or done >= budget:
                val = await asyncio.to_thread(self.learner.validate)
                note = ""
                if val:
                    self.metrics.holdout_pinball, self.metrics.holdout_bce = val["pinball"], val["bce"]
                    held.append(val["pinball"])
                    note = f", held-out {val['pinball']:.2f} bps"
                    if _stalled(held, int(getattr(cfg, "overfit_patience", 0) or 0)):
                        # The in-sample loss keeps falling while this stops improving: the net is
                        # memorising the replay. End the session's budget here, not at the epoch count.
                        self.trace.emit("learn", f"overfit guard: held-out pinball {val['pinball']:.2f} bps has not improved "
                                                 f"over {cfg.overfit_patience} checks (best {min(held):.2f}); ending this session's "
                                                 f"idle training at {done}/{budget} steps", "warn")
                        done = budget
                self.trace.emit("learn", f"idle training {done}/{budget} steps this {'open' if is_open else 'closed'} session "
                                         f"({size} windows in replay): loss {res['loss']:.3f} "
                                         f"(pinball {res['pinball']:.2f} bps, bce {res['bce']:.3f}{note})")
                self._publish({"type": "metrics", "metrics": self.metrics.as_dict(), "history": self._drain_history()})
                self.learner.release_cache()              # every 40 steps: the caching allocator never accumulates

    def _ingest_offline(self, ticks: list[Tick], emit_from: float | None = None, replace: bool = False) -> int:
        """Turn a chunk of old history into labeled replay windows without touching the live
        pipeline: a throwaway bar builder and feature builder (seeded with the live
        normalization state, reading the dated news store) walk the chunk chronologically and
        feed the learner directly. Bars before `emit_from` only warm the builder's rolling
        state (a week of returns, the window itself) so the chunk's first windows carry the
        same features live bars would; `replace` rewrites windows already in the buffer."""
        cfg = self.cfg
        builder = BarBuilder(self.symbols, cfg.bar_seconds)
        feats = FeatureBuilder(self.symbols, cfg.window, index=self._market_symbol())
        feats.norm.load(self.features.norm.state())
        feats.news_history = self.news_history
        feats.events = self.events
        emit_from = float(emit_from or 0.0)
        first: dict[str, float] = {}
        for t in ticks:
            first.setdefault(t.symbol, t.price)
        for s in self.symbols:
            first.setdefault(s, 0.0)                   # flat-seed history-less symbols, same as startup backfill
        for sym, px in first.items():
            builder.last_close.setdefault(sym, px)
        for t in ticks:
            builder.add(t)
        rows = builder.roll(ticks[-1].ts + cfg.bar_seconds)
        added, filled, idx = 0, 0, 0
        pend: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = deque()
        for row in rows:
            idx += 1
            closes = np.array([row[s].close for s in self.symbols], dtype=np.float32)
            real_now = np.array([row[s].volume > 0.0 for s in self.symbols])
            while pend and pend[0][0] + cfg.horizon <= idx:
                _, x0, c0, r0, t0 = pend.popleft()
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = (np.log(closes / c0) * 1e4).astype(np.float32)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                mk = (r0 & real_now).astype(np.float32)
                if self.learner.add(x0, y, mk, ts=t0):
                    added += 1
                elif replace and self.learner.replace(t0, x0, y, mk):
                    added += 1
                else:
                    filled += self.learner.fill(t0, x0, y, mk)
            x = feats.push(row)
            ts_row = float(next(iter(row.values())).ts)
            if x is not None and ts_row >= emit_from:
                pend.append((idx, x, closes, real_now, ts_row))
        self.metrics.labels = self.learner.labels
        self._deep_filled = filled
        return added

    def _market_symbol(self) -> str | None:
        """The first modeled index name: the market the features and the relative target use."""
        cols = index_columns(self.symbols, self.cfg.index_symbols)
        return self.symbols[cols[0]] if cols else None

    def _news_names(self) -> list[str]:
        return [s for s in self.symbols if not is_crypto(s)]

    def _history_span(self) -> tuple[float, float]:
        """The [lo, hi] of history the deep loop can hold and so the dated stores must cover: the
        ring's capacity in sessions (about 1.45 calendar days each) plus the warm prefix, under
        deep_backfill_days, plus the startup backfill, with the skim's horizon in front."""
        cfg = self.cfg
        cap_days = float(self.learner.rx.shape[0]) / BARS_D * 1.45 + WARM_S / 86400.0
        days = min(float(getattr(cfg, "deep_backfill_days", 0) or 0), cap_days) + float(cfg.backfill_seconds) / 86400.0
        return backfill_span(time.time(), max(days, 3.0))

    def _history_ready(self, lo: float, hi: float) -> bool:
        """Whether the dated stores hold what a chunk of [lo, hi] needs to carry the features the
        live feed will: its news, and its earnings calendar when the feature set reads one."""
        if not self.news_history.covers(self._news_names(), lo, hi):
            return False
        return not self.features.wants_events or self.events.covers(lo, hi)

    async def _events_job(self) -> None:
        """Keep the dated earnings calendar current: at the start the span the deep backfill will
        page (the deep loop waits for it, as for the news), then the weeks ahead again every six
        hours, since dates get confirmed and moved. Public Nasdaq calendar, one request per day."""
        first = True
        while True:
            lo, _ = self._history_span()
            hi = time.time() + AHEAD_S
            refresh = None if first else (datetime.now(timezone.utc).date() - timedelta(days=1))
            try:
                res = await self.events.fetch(lo, hi, say=lambda t: self.trace.emit("news", t, "info"), refresh_from=refresh)
            except Exception as e:  # noqa: BLE001
                self.trace.emit("news", f"earnings calendar fetch failed: {type(e).__name__}: {str(e)[:100]}", "warn")
                res = {}
            if res.get("days"):
                self.trace.emit("news", f"earnings calendar: {res['done']} of {res['days']} days fetched, {res['names']} reports"
                                        + (f"; {res['failed']} days failed" if res.get("failed") else "") + f" ({self.events.status})",
                                "warn" if res.get("failed") else "info")
            first = False
            await asyncio.sleep(6 * 3600)

    async def _news_history_job(self) -> None:
        """Fill the dated news store for the span the deep backfill will page, in the
        background and paced under Finnhub's limit; the deep loop waits for a chunk's news
        before paging it, and the startup days are rewritten once theirs is in."""
        cfg = self.cfg
        key = cfg.finnhub_key
        names = self._news_names()
        if not key or not names:
            return
        lo, hi = self._history_span()
        days = (hi - lo) / 86400.0
        missing = self.news_history.missing(names, lo, hi)
        if not missing:
            return
        scorer = getattr(self.news_hub, "scorer", None)
        self.trace.emit("news", f"news history: fetching {len(missing)} names over {days:.0f} days from Finnhub, "
                                f"scored by {getattr(scorer, 'name', 'word lists')}", "info")
        old_threads = getattr(scorer, "threads", None)
        if old_threads is not None:
            scorer.threads = max(int(old_threads), min(8, os.cpu_count() or 2))
        try:
            res = await self.news_history.fetch(names, lo, hi, key, scorer=scorer, rate_s=1.5,
                                                say=lambda t: self.trace.emit("news", t, "info"))
        except Exception as e:  # noqa: BLE001
            self.trace.emit("news", f"news history fetch failed: {type(e).__name__}: {str(e)[:100]}", "warn")
            return
        finally:
            if old_threads is not None:
                scorer.threads = old_threads
        if res.get("error"):
            self.trace.emit("news", f"news history: {res['error']}", "warn")
            return
        self.trace.emit("news", f"news history: {res['headlines']} headlines in {res['requests']} requests over "
                                f"{res['elapsed_s'] / 60:.0f} min" + (f"; {len(res['failed'])} names failed" if res["failed"] else ""),
                        "warn" if res["failed"] else "info")

    def _on_live_tick(self, tick: Tick) -> None:
        """Called by the source manager with the active provider's tick for a symbol.
        Quote/heartbeat ticks update the live price but never build candles -- otherwise a
        market-closed feed echoing the last quote would fabricate flat bars from stale data.
        A non-quote tick with zero size and no OHLC is a stale price echo, not a real trade
        (a real trade has volume), so it updates the price but must not build a candle -- one
        such echo would mark the interval active and fabricate flat bars across every symbol."""
        if not tick.quote and (tick.size > 0 or tick.o is not None):
            self.builder.add(tick)
        self._on_tick(tick, live=True)

    def _on_provider_change(self, sym: str, prev, active) -> None:
        pn = self.sources.sources[prev].name if prev else "none"
        an = self.sources.sources[active].name if active else "none"
        lvl = "warn" if active is None or active == "sim" else "act"
        self.trace.emit("feed", f"{self.base[sym]} provider: {pn} -> {an}", lvl)
        self._publish({"type": "status", "status": self.status_dict()})

    def _on_quote(self, sym: str, bid: float, ask: float) -> None:
        """Bid/ask update from a non-active quote source (keeps the paper-book spread current)."""
        p = self.prices.get(sym)
        if p:
            p["bid"] = bid
            p["ask"] = ask

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

    STALL_BARS = 3   # no bar for this many intervals while the market is open = no source is delivering trades

    async def _clock(self) -> None:
        self._last_row_at = time.time()
        self._stalled = False
        while True:
            await asyncio.sleep(0.25)
            for row in self.builder.roll(time.time()):
                try:
                    await self._process_row(row)
                except Exception:  # noqa: BLE001 -- one bad bar must not kill the clock for the rest of the session
                    log.exception("bar #%d failed; the clock keeps running", self.bar_index)
                    self.trace.emit("system", f"bar #{self.bar_index} failed; see the log", "warn")
                self._last_row_at = time.time()
                if self._stalled:
                    self._stalled = False
                    self.trace.emit("system", "bars are forming again", "info")
            self._check_stall(time.time())

    def _check_stall(self, now: float) -> None:
        """Quotes can keep every price ticking while no candle forms (2026-09-04: the one trade
        websocket died and Flint sat blind all session). Say so, loudly, once per stall."""
        gap = now - self._last_row_at
        if self._stalled or gap < self.STALL_BARS * self.cfg.bar_seconds:
            return
        open_now = self.market_status.get("isOpen")
        if open_now is None:
            open_now = clock.regular_session()
        if not open_now:
            return
        self._stalled = True
        srcs = ", ".join(f"{p['id']} ({p.get('note') or 'ok'})" for p in self.sources.status() if p.get("enabled") and p.get("ticks"))
        msg = (f"no bar for {gap / 60:.0f} minutes while the market is open: prices tick but no source is "
               f"delivering trades, so nothing forecasts or trades. Sources: {srcs}")
        log.warning(msg)
        self.trace.emit("system", msg, "warn")

    async def _ticker(self) -> None:
        period = 1.0 / max(self.cfg.tick_hz, 0.2)
        while True:
            await asyncio.sleep(period)
            self._publish({"type": "tick", "prices": self.prices, "status": self.status_dict(),
                           "paper_eq": round(self.paper.equity(self._live_prices()), 2)})

    async def _checkpoints(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.checkpoint_minutes * 60)
            await asyncio.to_thread(self._save)
            self.trace.emit("system", f"checkpoint saved to {self.cfg.state_dir} ({self.learner.steps} steps, "
                                      f"{self.learner.size} windows)")

    CALIB_V = 2   # calibration-state schema: 2 = batch (once-per-bar) updates; pre-2 values are untrusted

    _PERSIST_METRICS = ("live_labels", "live_pinball", "band_scale", "p_scale", "decisions", "hits",
                        "hit_ema", "coverage_n", "covered", "coverage_ema", "sharpness_ema",
                        "coverage_raw_n", "covered_raw", "suggestions", "suggestion_wins", "pnl_bps")

    def _save(self, final: bool = False) -> None:
        """Checkpoint. `final` (shutdown) also forces the replay buffer to disk."""
        try:
            m = self.metrics
            metrics = {k: getattr(m, k) for k in self._PERSIST_METRICS}
            metrics["pnl_by_symbol"] = dict(m.pnl_by_symbol)
            metrics["calib_v"] = self.CALIB_V
            self.learner.save(self.cfg.state_dir, extra={"norm": self.features.norm.state(list(self.symbols)), "meta": self.meta.state(), "metrics": metrics,
                                                          "paper": self.paper.to_state(), "target": self.learner.target_mode,
                                                          "outcomes": {s: list(d) for s, d in self.outcomes.items() if d}},
                              force_replay=final)
            tmp = os.path.join(self.cfg.state_dir, "universe.json.tmp")
            with open(tmp, "w") as f:
                json.dump({"symbols": list(self.all_symbols), "refreshed": self._universe_refreshed,
                           "interesting": self._interesting}, f)
            os.replace(tmp, os.path.join(self.cfg.state_dir, "universe.json"))
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
            if x is None or not all(self.bars.get(s) for s in self.symbols):
                return
            closes = np.array([self.bars[s][-1].close for s in self.symbols], dtype=np.float32)
            ts = self.bars[self.symbols[0]][-1].ts
        pred = self.learner.predict(x)
        sugg = {s: self._suggest(i, s, pred) for i, s in enumerate(self.symbols)}
        self.gate = [float(g) for g in pred.gate]
        self.latest = {}
        for i, s in enumerate(self.symbols):
            self.latest[s] = {**sugg[s], "price": float(closes[i]), "ts": ts}
        self._publish({"type": "bar", "index": self.bar_index,
                       "bars": {s: self.bars.get(s)[-1].to_json() for s in self.symbols if self.bars.get(s)},
                       "latest": self.latest, "gate": self.gate, "metrics": self.metrics.as_dict(),
                       "history": {}, "outcomes": {s: list(self.outcomes.get(s, ())) for s in self.symbols},
                       "log": list(self.log), "status": self.status_dict()})

    async def _process_row(self, row: dict[str, Bar]) -> None:
        cfg = self.cfg
        self.bar_index += 1
        for s in self.symbols:
            self.bars.setdefault(s, deque(maxlen=self.cfg.chart_bars)).append(row[s])
        closes = np.array([row[s].close for s in self.symbols], dtype=np.float32)
        real_now = np.array([row[s].volume > 0.0 for s in self.symbols])   # a v=0 bar is a carry-forward, not a real trade
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

        matured = self._mature(closes, real_now)
        x = self.features.push(row)

        if x is not None:
            z = x[:, -1, :]
            for i, s in enumerate(self.symbols):
                self.trace.emit("features", f"{self.base[s]} " + " ".join(f"{n} {z[i, j]:+.2f}" for j, n in enumerate(FEATURES)))

        if matured and self.learning_enabled and not self._backtesting:
            res = await asyncio.to_thread(self.learner.train_steps, min(8, cfg.steps_per_label * matured))
            if res:
                self._absorb_train(res)
                self.trace.emit("learn", f"step {res['steps']}: loss {res['loss']:.3f} (pinball {res['pinball']:.2f} bps, "
                                         f"bce {res['bce']:.3f}, gate balance {res.get('balance', 0):.3f}) on {matured} new label(s)")
        elif matured:
            self.trace.emit("learn", f"{matured} new label(s) stored; "
                                     + ("training waits for the backtest" if self._backtesting else "learning is paused"))

        if x is not None:
            closes_d = {s: float(closes[i]) for i, s in enumerate(self.symbols)}
            self._book_equity = float(self.paper.equity(closes_d, ts))
            pred = self.learner.predict(x)
            sugg = {s: self._suggest(i, s, pred) for i, s in enumerate(self.symbols)}
            self.pending.append(Pending(self.bar_index, ts, x, closes, real_now, pred, sugg))
            self.gate = [float(g) for g in pred.gate]
            self.latest = {}
            for i, s in enumerate(self.symbols):
                self.latest[s] = {**sugg[s], "price": float(closes[i]), "ts": ts,
                                  **({"beta": {self.symbols[c]: round(float(self.learner.beta[i, j]), 2)
                                               for j, c in enumerate(self.learner.cols)}} if self.learner.cols else {})}
            # trade only once the model is trusted AND has shown real directional skill (not just tenure)
            tradeable = self.metrics.trusted and (self.metrics.hit_ema or 0.0) > cfg.min_hit_rate
            targets = {s: (self.latest[s]["side"] * self.latest[s]["size"] if tradeable else 0.0) for s in self.symbols}
            # minimum hold: a fading signal does not close a young position; a reversed one does
            for s in self.symbols:
                t, prev = targets[s], self._last_target.get(s, 0.0)
                if t == 0.0 and prev != 0.0:
                    held = self.bar_index - self._held_since.get(s, self.bar_index)
                    reversed_ = ((self.latest[s].get("score") or 0.0) > 0) != (prev > 0)
                    if held < cfg.stock_min_hold_bars and not reversed_ and tradeable:
                        targets[s] = t = prev
                if t != 0.0 and prev == 0.0:
                    self._held_since[s] = self.bar_index
                self._last_target[s] = t
            # Exits: a stop, a profit target and a time limit on stock positions, each overriding the
            # minimum hold; a name that exits waits out a cooldown before it may re-enter; and once
            # the book has lost `daily_loss_limit` of the day's opening equity, no new entries today.
            for s, why in policy.exit_decisions(self.paper.pos, self.paper.avg, closes_d, self._held_since, self.bar_index, cfg).items():
                if targets.get(s, 0.0) > 0:
                    targets[s] = self._last_target[s] = 0.0
                    self._cooldown[s] = self.bar_index
                    self.latest[s].update({"side": 0, "size": 0.0, "action": "HOLD", "why": "exit: " + why})
                    self._log(f"{s} exit: {why}", "act")
            cooldown = int(getattr(cfg, "exit_cooldown_bars", 0) or 0)
            day = datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).date().isoformat()
            if not self._day_open_equity or self._day_open_equity[0] != day:
                self._day_open_equity = (day, self._book_equity)
            limit = float(getattr(cfg, "daily_loss_limit", 0.0) or 0.0)
            down = 1.0 - self._book_equity / self._day_open_equity[1] if self._day_open_equity[1] > 0 else 0.0
            for s in self.symbols:
                if targets[s] == 0.0 or self.paper.pos.get(s) or s in self.paper.puts:
                    continue                                                     # only new entries are held back
                if cooldown and self.bar_index - self._cooldown.get(s, -10**9) < cooldown:
                    targets[s] = self._last_target[s] = 0.0
                    self.latest[s].update({"side": 0, "size": 0.0, "action": "HOLD",
                                           "why": f"cooling down after an exit ({self.bar_index - self._cooldown[s]}/{cooldown} bars)"})
                elif limit > 0 and down >= limit:
                    targets[s] = self._last_target[s] = 0.0
                    self.latest[s].update({"side": 0, "size": 0.0, "action": "HOLD",
                                           "why": f"daily loss limit: the book is down {down * 100:.1f}% today, no new entries"})
            # The relative target says "beats its indices", so the book holds, per index, the weight
            # that cancels the basket's loading on it: a put under a long basket, index stock over a
            # bearish one. An index name is a hedge instrument here, never a signal of its own.
            hedge_syms: set[str] = set()
            if self.learner.cols and cfg.hedge_index:
                hs = hedge_targets(targets, self.learner.beta, self.symbols, self.learner.cols) if tradeable else {}
                for c in self.learner.cols:
                    isym = self.symbols[c]
                    h = hs.get(isym, 0.0)
                    h = 0.0 if abs(h) < float(cfg.hedge_min) else h
                    targets[isym] = self._last_target[isym] = h
                    hedge_syms.add(isym)
                    self.latest[isym].update({"hedge": round(h, 4), "side": (h > 0) - (h < 0), "size": round(abs(h), 3),
                                              "action": "HEDGE" if h else "HOLD",
                                              "why": (f"index hedge: {h:+.0%} of the book cancels the basket's loading on {isym}, as "
                                                      f"{'index stock' if h > 0 else 'an index put'}") if h else "index hedge: nothing to cancel"})
            put_quotes = {}                                    # a short is opened as a long put (bounded loss), priced off a live chain
            need = [s for s in self.symbols if targets[s] < 0 and s not in self.paper.puts]
            if need and self.schwab_auth.authenticated:
                from .options import fetch_put
                res = await asyncio.gather(*[fetch_put(self.schwab_auth, s, self.cfg.option_dte) for s in need],
                                           return_exceptions=True)
                for s, pq in zip(need, res):
                    if isinstance(pq, dict):
                        put_quotes[s] = pq
            # Fat-tail forecasts trade as long straddles: they need band skill, not directional
            # skill, so they gate on calibrated coverage instead of hit rate.
            vol_tradeable = (self.metrics.trusted and cfg.straddle_enabled
                             and (self.metrics.coverage_ema or 0.0) >= cfg.straddle_min_coverage)
            straddle_targets: dict[str, float] = {}
            straddle_quotes: dict[str, dict] = {}
            if vol_tradeable:
                flagged = sorted((s for s in self.symbols if self.latest[s].get("straddle")),
                                 key=lambda s: -(self.latest[s].get("width") or 0.0))
                keep = [s for s in flagged if s in self.paper.straddles]
                room = max(0, cfg.straddle_max - len(self.paper.straddles))
                new = [s for s in flagged if s not in self.paper.straddles][:room]
                straddle_targets = {s: cfg.straddle_budget for s in (*keep, *new)}
                if new and self.schwab_auth.authenticated:
                    from .options import fetch_straddle
                    sres = await asyncio.gather(*[fetch_straddle(self.schwab_auth, s, self.cfg.option_dte) for s in new],
                                                return_exceptions=True)
                    for s, sq in zip(new, sres):
                        if isinstance(sq, dict):
                            straddle_quotes[s] = sq
            self.paper.rebalance(targets,
                                 {s: float(closes[i]) for i, s in enumerate(self.symbols)}, ts,
                                 quotes={s: {"bid": self.prices[s].get("bid"), "ask": self.prices[s].get("ask")}
                                         for s in self.symbols},
                                 put_quotes=put_quotes,
                                 straddle_targets=straddle_targets, straddle_quotes=straddle_quotes, uncapped=hedge_syms)
            self.trace.emit("model", f"bar #{self.bar_index}: regime gate " + " ".join(
                f"E{k + 1}={g:.2f}" for k, g in enumerate(self.gate)) +
                f" | band scale x{self.metrics.band_scale:.2f}, P(up) temper {self.metrics.p_scale:.2f}")
            for i, s in enumerate(self.symbols):
                q = pred.q[i]
                att = sorted(((float(pred.attn[i, j]), self.base[self.symbols[j]]) for j in range(len(self.symbols)) if j != i), reverse=True)
                sg = sugg[s]
                self.trace.emit("model", f"{self.base[s]} raw q10 {q[0]:+.1f} q25 {q[1]:+.1f} q50 {q[2]:+.1f} q75 {q[3]:+.1f} "
                                         f"q90 {q[4]:+.1f} bps, P(up) {pred.p_up[i]:.2f} | calibrated band {sg['q'][0]:+.1f} to "
                                         f"{sg['q'][4]:+.1f}, P(up) {sg['p_up']:.2f} P(down) {sg['p_down']:.2f} | attends " +
                                " ".join(f"{n} {a:.2f}" for a, n in att[:3]))
                self.trace.emit("policy", f"{self.base[s]} {sg['action']}" + (f" size {sg['size']:.2f}" if sg['side'] else "") +
                                f": {sg['why']}" + ("" if sg["trusted"] else " [warming up, not counted]"),
                                "act" if sg["side"] else "info")

        self._publish({
            "type": "bar", "index": self.bar_index, "bars": {s: row[s].to_json() for s in self.symbols},
            "latest": self.latest, "gate": self.gate, "metrics": self.metrics.as_dict(),
            "history": self._drain_history(),
            "outcomes": {s: list(self.outcomes.get(s, ())) for s in self.symbols}, "log": list(self.log),
            "status": self.status_dict(), "paper": self.paper.snapshot(self._live_prices()),
            "backtest": self.backtest_summary(),
        })

    def _mature(self, closes: np.ndarray, real_now: np.ndarray | None = None) -> int:
        n = 0
        while self.pending and self.pending[0].index + self.cfg.horizon <= self.bar_index:
            pp = self.pending.popleft()
            with np.errstate(divide="ignore", invalid="ignore"):
                y = (np.log(closes / pp.closes) * 1e4).astype(np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)   # a 0-seeded / history-less symbol yields no return, not a NaN that poisons the whole batch
            mask = (pp.real & real_now).astype(np.float32) if real_now is not None else None   # train only on symbols that really traded at both ends
            self.learner.add(pp.x, y, mask, ts=pp.ts)       # the replay keeps the absolute label
            if pp.pred is not None:
                self._score(pp, self.learner.target(y))     # the forecast was made on the configured target
            n += 1
        self.metrics.labels = self.learner.labels
        self.metrics.trusted = self.metrics.live_labels >= self.cfg.min_labels
        return n

    def _absorb_train(self, res: dict) -> None:
        m = self.metrics
        m.loss, m.pinball, m.bce, m.steps = res["loss"], res["pinball"], res["bce"], res["steps"]

    def _temper(self, p_raw: float) -> float:
        """Apply the online P(up)/P(down) temperature (p_scale)."""
        p_raw = min(max(p_raw, 1e-6), 1 - 1e-6)
        z = max(-6.0, min(6.0, math.log(p_raw / (1 - p_raw))))
        return 1.0 / (1.0 + math.exp(-self.metrics.p_scale * z))

    def _calibrate(self, q, p_raw: float) -> tuple[list[float], float]:
        """Apply the online conformal band scale and the P(up) temperature."""
        m = self.metrics
        q50 = float(q[2])
        qc = [q50 + m.band_scale * (float(v) - q50) for v in q]
        return qc, self._temper(p_raw)

    def _regular_session(self) -> bool:
        """Regular US equity session by the wall clock (9:30-16:00 ET, weekdays). Gates trading;
        a no-trade (volume 0) bar additionally covers holidays and early closes."""
        return clock.regular_session()

    def _extended_session(self) -> bool:
        """Schwab's extended sessions by the wall clock: 4:00-9:30 and 16:00-20:00 ET, weekdays.
        Stock only -- options do not trade here, so puts and straddles keep the regular gate."""
        return clock.extended_session()

    def _stock_session(self) -> bool:
        """When a long-stock entry or exit is allowed: the regular session, plus the extended
        sessions when `extended_hours` is on."""
        return clock.stock_session(self.cfg.extended_hours)

    def _suggest(self, i: int, s: str, pred: Prediction) -> dict:
        cfg = self.cfg
        m = self.metrics
        p_raw = float(pred.p_up[i])
        p_down_raw = float(pred.p_down[i])
        qc, p = self._calibrate(pred.q[i], p_raw)
        p_down = self._temper(p_down_raw)
        q10, q25, q50, q75, q90 = qc
        iqr = max(q75 - q25, 1e-3)
        score = q50 / iqr
        # the live spread is the real cost of a round trip; a flat cost_bps only sets the floor
        pr = self.prices.get(s) or {}
        bid, ask = pr.get("bid"), pr.get("ask")
        spread_bps = (ask - bid) / ((ask + bid) / 2) * 1e4 if bid and ask and ask > bid > 0 else None
        # Beyond the spread, an hour of this position costs the impact of the order it would place
        # and, for a bearish view bought as a put, the put's decay; the edge has to cover all of it.
        bars_s = self.bars.get(s) or ()
        eq = float(getattr(self, "_book_equity", 0.0) or 0.0) or 100_000.0
        size0 = cfg.max_size * min(1.0, cfg.kelly_fraction * abs(score))        # the size this signal would take
        floor_usd = float(getattr(cfg, "min_position_dollars", 0.0) or 0.0)
        if floor_usd > 0 and eq * size0 < floor_usd:
            size0 = min(cfg.max_size, floor_usd / eq)                            # a smaller ticket is not worth placing: the floor or nothing
        px0 = float(pr.get("price") or (bars_s[-1].close if bars_s else 0.0) or 0.0)
        adv = policy.dollar_volume(bars_s)
        impact = policy.impact_bps(eq * size0, adv, policy.sigma_daily(bars_s), float(getattr(cfg, "impact_eta", 0.1)))
        iv = (getattr(self, "option_iv", None) or {}).get(s)
        theta = (policy.theta_bps_per_hour(iv, cfg.option_dte)
                 if score < 0 and iv and getattr(cfg, "theta_in_edge", True) else 0.0)
        edge = abs(q50) - max(cfg.cost_bps, spread_bps or 0.0) - impact - theta
        # The fixed part of the ticket. Stock trades free; a bearish view is whole put contracts that
        # pay a commission each way, so a small position pays a large share of its notional before
        # the put has moved. The hourly edge has the hold to earn it back.
        hold_h = max(int(cfg.put_min_hold_bars if score < 0 else cfg.stock_min_hold_bars), int(cfg.horizon), 1) * float(cfg.bar_seconds) / 3600.0
        contracts, fixed_bps = 0, 0.0
        if score < 0 and px0 > 0:
            contracts = max(1, int(round(eq * size0 / (100.0 * px0 * 0.5))))     # delta-equivalent exposure in ATM contracts
            fixed_bps = 2.0 * float(cfg.option_commission) * contracts / max(eq * size0, 1.0) * 1e4
        kyle = policy.kyle_impact_bps(policy.amihud(bars_s), eq * size0)
        conviction = abs(2 * p - 1)
        reasons = []
        if abs(score) < cfg.score_threshold:
            reasons.append("the expected move is small next to its uncertainty")
        if conviction < 2 * cfg.prob_margin:
            reasons.append(f"direction is near a coin flip ({p * 100:.0f}% up)")
        if edge <= 0:
            parts = [f"{spread_bps:.0f} bps spread"] if spread_bps else []
            if impact >= 0.5:
                parts.append(f"{impact:.1f} bps of impact")
            if theta >= 0.5:
                parts.append(f"{theta:.1f} bps/h of put decay")
            reasons.append("the expected move does not cover trading cost" + (f" ({', '.join(parts)})" if parts else ""))
        elif fixed_bps > 0 and edge * hold_h <= fixed_bps:
            reasons.append(f"a {contracts}-contract ticket costs {fixed_bps:.0f} bps round trip against "
                           f"{edge * hold_h:.0f} bps expected over a {hold_h:.1f} h hold")
        if kyle > float(getattr(cfg, "max_impact_bps", 10.0)):
            reasons.append(f"too illiquid: a {size0 * 100:.0f}% position would move the price about {kyle:.0f} bps")
        if spread_bps is not None and spread_bps > cfg.max_spread_bps:
            reasons.append(f"spread {spread_bps:.0f} bps is wider than the {cfg.max_spread_bps:.0f} bps limit")
        # per-name skill: a name earns the right to trade with its own record, not the book's
        judged = [o for o in (self.outcomes.get(s) or []) if o.get("hit") is not None]
        if len(judged) < cfg.skill_min_n:
            reasons.append(f"no track record on this name yet ({len(judged)}/{cfg.skill_min_n} calls)")
        else:
            hr = sum(1 for o in judged if o["hit"]) / len(judged)
            if hr < cfg.min_hit_rate:
                reasons.append(f"its record on this name is {hr * 100:.0f}% over {len(judged)} calls")
        # the meta-labeler's second opinion: the odds this forecast pays after cost, learned from matured ones
        meta = getattr(self, "meta", None)
        meta_p = None
        if meta is not None and meta.fitted:
            meta_p = float(meta.predict(meta_features(pred.q[i], p_raw, p_down_raw)[None])[0])
            if meta_p < float(getattr(cfg, "meta_min", 0.0)):
                reasons.append(f"the meta-labeler gives it {meta_p * 100:.0f}% odds of paying after cost")
        if abs(q50) < cfg.move_floor_bps:
            reasons.append(f"expected move {abs(q50):.0f} bps is below the {cfg.move_floor_bps:.0f} bps floor")
        bar = self.bars[s][-1] if self.bars.get(s) else None
        if not (bar and bar.volume > 0):
            reasons.append("no live trade this bar (stale/flat price)")
        px = pr.get("price") or (self.bars[s][-1].close if self.bars.get(s) else None)
        if px and px > 0 and (bar and bar.volume > 0 or pr.get("price")) and px < cfg.min_price:   # a real quote or a traded bar, never a flat-seeded zero
            reasons.append(f"price {px:.2f} is below the {cfg.min_price:.0f} floor")
        if not self._stock_session():
            reasons.append("outside trading hours")
        elif score < 0 and not self._regular_session():
            reasons.append("shorts are puts, and options only trade in regular hours")
        if (score > 0) != (p > 0.5):
            reasons.append("its trend and direction signals disagree")
        # confirmation: the same side has to persist before the book acts on it
        raw_side = 0 if reasons else (1 if score > 0 else -1)
        prev = self._streak.get(s)
        run = prev[1] + 1 if prev and prev[0] == raw_side else 1
        self._streak[s] = (raw_side, run)
        if raw_side and run < cfg.confirm_bars:
            reasons.append(f"waiting for the signal to persist ({run}/{cfg.confirm_bars} bars)")
        side = 0 if reasons else raw_side
        # Compounding-oriented sizing: allocate by risk-adjusted edge (|q50|/IQR = |score|, a
        # Sharpe-like fractional-Kelly proxy), so capital concentrates where edge-per-unit-risk is
        # highest -- the direction/conviction checks above still gate entry. Fractional and hard-capped,
        # because edge estimates are noisy and drawdowns erode compounding faster than wins build it.
        size = cfg.max_size * min(1.0, cfg.kelly_fraction * abs(score)) if side else 0.0
        if side and cfg.size_by_coverage and m.coverage_ema is not None:
            size *= max(0.25, min(1.0, m.coverage_ema / 0.8))   # overconfident bands -> smaller positions
        if side and floor_usd > 0 and eq * size < floor_usd:
            size = min(cfg.max_size, floor_usd / eq)             # never a position too small to be worth its ticket
        why = "holding: " + "; ".join(reasons) if reasons else f"leaning in: {q50:+.0f} bps expected, {p * 100:.0f}% up / {p_down * 100:.0f}% down"
        fat_tails = p > 0.5 and p_down > 0.5
        width = q90 - q10
        # A fat-tail forecast is tradeable without direction: a long straddle wins on |move|,
        # loss capped at the premium. Requires a live bar and market hours like any entry.
        if iv:
            # Gamma breakeven: a straddle pays only when the move beats what its premium already
            # prices in (the implied one-sigma move over the hold) plus the decay over that hold.
            hold_h = float(getattr(cfg, "straddle_hold_bars", 12)) * float(getattr(cfg, "bar_seconds", 300)) / 3600.0
            implied = policy.implied_move_bps(iv, hours=hold_h)
            decay = policy.theta_bps_per_hour(iv, cfg.option_dte) * hold_h
            big = policy.expected_abs_move_bps(qc) > float(getattr(cfg, "straddle_edge", 1.0)) * implied + decay
        else:
            big = width >= cfg.straddle_band_bps
        straddle = (cfg.straddle_enabled and fat_tails and big
                    and bool(bar and bar.volume > 0) and self._regular_session())
        if fat_tails:
            why += "  |  both tails fat — expect a big move, direction unclear"
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
                "straddle": straddle and not muted, "width": round(width, 1),
                "size": round(size, 3), "score": score, "p_up": p, "p_raw": p_raw, "p_down": p_down, "p_down_raw": p_down_raw, "q": qc,
                "q_raw": [float(v) for v in pred.q[i]], "iqr": iqr, "band_scale": m.band_scale, "p_scale": m.p_scale,
                "trusted": m.trusted, "why": why, "reasons": reasons, "meta": meta_p, "ticket_bps": round(fixed_bps, 1),
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
        # Symbols move together, so a bar's 211 labels are one observation of calibration, not 211.
        # Per-symbol updates here compound into one enormous step (and with the horizon's delay,
        # band_scale bang-bangs between its clips). Accumulate the bar, apply each update once.
        pins, hits_b, widths, insides, p_grads = [], [], [], [], []
        mx, mt = [], []                                                     # the meta-labeler's lesson from this bar
        for i, s in enumerate(self.symbols):
            q = pp.pred.q[i]
            yi = float(y[i])
            q50 = float(q[2])
            sug = pp.suggestions.get(s, {"side": 0, "size": 0.0, "trusted": False, "action": "HOLD"})
            # Out-of-sample pinball loss of the raw forecast made before this label existed.
            diff = yi - q.astype(np.float64)
            pin = float(np.maximum(taus * diff, (taus - 1) * diff).mean())
            pins.append(pin)
            hit = None
            if yi != 0.0 and q50 != 0.0:
                hit = (q50 > 0) == (yi > 0)
                m.decisions += 1
                m.hits += int(hit)
                hits_b.append(float(hit))
                mx.append(meta_features(q, float(pp.pred.p_up[i]), float(pp.pred.p_down[i])))
                mt.append(float(yi * (1.0 if q50 > 0 else -1.0) > float(getattr(cfg, "meta_cost_bps", 5.0))))
            raw_inside = float(q[0]) <= yi <= float(q[-1])
            m.coverage_raw_n += 1
            m.covered_raw += int(raw_inside)
            qc = sug.get("q") or [float(v) for v in q]
            inside = qc[0] <= yi <= qc[-1]
            m.coverage_n += 1
            m.covered += int(inside)
            insides.append(float(inside))
            widths.append(float(qc[-1] - qc[0]))
            if yi != 0.0:
                p_raw = min(max(float(sug.get("p_raw", pp.pred.p_up[i])), 1e-6), 1 - 1e-6)
                z = max(-6.0, min(6.0, math.log(p_raw / (1 - p_raw))))
                pc = 1.0 / (1.0 + math.exp(-m.p_scale * z))
                p_grads.append((pc - (1.0 if yi > 0 else 0.0)) * z)
            pnl = None
            if sug["side"] != 0 and sug["trusted"]:
                pnl = sug["side"] * sug["size"] * yi - cfg.cost_bps * sug["size"]
                m.pnl_bps += pnl
                m.pnl_by_symbol[s] = m.pnl_by_symbol.get(s, 0.0) + pnl
                m.suggestions += 1
                m.suggestion_wins += int(pnl > 0)
                self._log(f"{s} {sug['action']} x{sug['size']:.2f} resolved {yi:+.1f} bps, paper {pnl:+.2f} bps",
                          "win" if pnl > 0 else "loss")
            self.outcomes.setdefault(s, deque(maxlen=40)).append({"t": pp.ts, "y": yi, "q50": q50, "hit": hit, "inside": inside,
                                     "side": sug["side"], "pnl": pnl, "pin": pin})
            mark = "hit" if hit else "miss" if hit is False else "flat"
            parts.append(f"{self.base[s]} {yi:+.1f} vs q50 {q50:+.1f} {mark}" + ("" if inside else " (outside band)"))
        if mx:
            self.meta.update(np.asarray(mx), np.asarray(mt))
        m.live_pinball = _ema(m.live_pinball, float(np.mean(pins)), 0.05)
        if hits_b:
            m.hit_ema = _ema(m.hit_ema, float(np.mean(hits_b)), 0.02)
        inside_frac = float(np.mean(insides))
        m.coverage_ema = _ema(m.coverage_ema, inside_frac, 0.02)
        m.sharpness_ema = _ema(m.sharpness_ema, float(np.mean(widths)), 0.05)
        # Adaptive conformal update: the band scale settles where 20% of labels fall outside.
        m.band_scale = float(np.clip(m.band_scale * math.exp(cfg.band_gamma * ((1.0 - inside_frac) - 0.2)), 0.5, 25.0))
        # Online temperature on the direction head, one SGD step of the calibration cross-entropy.
        if p_grads:
            m.p_scale = float(np.clip(m.p_scale - cfg.temper_lr * float(np.mean(p_grads)), 0.02, 1.5))
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

    async def _market_loop(self) -> None:
        """Refresh whole-market movers/sectors/breadth on a tight loop; the heavier radar stays on the signals cadence."""
        if not self.signals.enabled.get("market"):
            return
        await asyncio.sleep(12)
        while True:
            if not self._gathering:                        # don't collide with the full signals gather
                try:
                    await asyncio.wait_for(self.signals.market.scan(light=True), timeout=45)
                    if isinstance(self.signals_state, dict):
                        self.signals_state["market"] = self.signals.market.state
                        self._publish({"type": "signals", "signals": self.signals_state,
                                       "signal_providers": self.signals.status()})
                except Exception as e:  # noqa: BLE001
                    self.trace.emit("signals", f"market refresh failed: {type(e).__name__}", "warn")
            await asyncio.sleep(max(20.0, self.cfg.market_scan_seconds))

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

    async def _option_features_loop(self) -> None:
        """Feed forward-looking option signals (ATM implied vol, put/call skew) to the model."""
        if not self.schwab_auth.authenticated:
            return
        import math as _math
        from .options import fetch_iv_skew
        sem = asyncio.Semaphore(8)
        await asyncio.sleep(15)

        async def one(s):
            async with sem:
                try:
                    return s, await fetch_iv_skew(self.schwab_auth, s, self.cfg.option_dte)
                except Exception:  # noqa: BLE001
                    return s, None
        while True:
            try:
                res = await asyncio.gather(*[one(s) for s in self.symbols])
                n = 0
                for s, r in res:
                    if r:
                        self.option_iv[s] = float(r["iv"])
                        self.features.set_exo(s, {"opt_iv": max(-1, min(1, (r["iv"] - 0.30) * 2.5)),
                                                  "opt_skew": max(-1, min(1, r["skew"] * 15))})
                        n += 1
                if n:
                    self.trace.emit("signals", f"option signals refreshed: ATM IV + skew for {n} names", "act")
            except Exception as e:  # noqa: BLE001
                self.trace.emit("signals", f"option features: {type(e).__name__}", "warn")
            await asyncio.sleep(max(120.0, self.cfg.option_features_seconds))

    async def _covered_calls(self) -> None:
        """Attach covered-call opportunities to the portfolio snapshot (read-only analysis; never
        trades). Throttled to covered_call_seconds and cached, so it survives the 60s position poll."""
        now = time.time()
        if now - self._cc_last >= self.cfg.covered_call_seconds:
            self._cc_last = now
            try:
                self._cc_cache = await self._compute_covered_calls(now)
            except Exception as ex:  # noqa: BLE001
                self.trace.emit("system", f"covered calls: {type(ex).__name__}", "warn")
        if self._cc_cache:
            self.portfolio["covered_calls"] = self._cc_cache

    async def _compute_covered_calls(self, now: float) -> list:
        """For each holding of >=100 shares, price an OTM call and recommend writing it only when
        Flint is not bullish -- a covered call caps upside, so it fits holdings where Flint expects
        limited further gains. Purely informational; the app never places an order."""
        if not self.schwab_auth.authenticated:
            return []
        holdings: dict[str, float] = {}
        for a in self.portfolio.get("accounts", []):
            for p in a.get("positions", []):
                sym = p.get("symbol") or ""
                if (p.get("asset_type") in ("EQUITY", "ETF", "COLLECTIVE_INVESTMENT")
                        and sym.isalpha() and 1 <= len(sym) <= 5 and (p.get("qty") or 0) >= 100):
                    holdings[sym] = holdings.get(sym, 0.0) + float(p["qty"])
        if not holdings:
            return []
        from .options import fetch_call
        syms = sorted(holdings)
        quotes = await asyncio.gather(
            *[fetch_call(self.schwab_auth, s, self.cfg.option_dte, self.cfg.covered_call_otm,
                          self.cfg.covered_call_delta) for s in syms],
            return_exceptions=True)
        opps = []
        for s, cq in zip(syms, quotes):
            if not isinstance(cq, dict) or not cq.get("premium") or not cq.get("spot"):
                continue
            contracts = int(holdings[s] // 100)
            if contracts < 1:
                continue
            spot, strike, prem = cq["spot"], cq["strike"], cq["premium"]
            otm_pct = (strike - spot) / spot * 100.0
            if not (0.0 <= otm_pct <= 60.0):    # skip garbage strikes (bad chain/spot/delta data); a real covered call is modestly OTM
                continue
            dte = max(1.0, (cq["expiry_ts"] - now) / 86400.0)
            yld = prem / spot * 100.0
            lat = self.latest.get(s, {})
            p_up = lat.get("p_up")
            q = lat.get("q") or []
            q50 = q[2] if len(q) >= 3 else None
            bullish = (p_up is not None and p_up > 0.58) or (q50 is not None and q50 > self.cfg.move_floor_bps)
            delta = cq.get("delta")
            opps.append({
                "symbol": s, "shares": round(holdings[s]), "contracts": contracts,
                "strike": strike, "expiry": datetime.fromtimestamp(cq["expiry_ts"]).strftime("%b %d"),
                "dte": round(dte), "premium": round(prem, 2), "income": round(prem * 100.0 * contracts),
                "yield_pct": round(yld, 2), "annualized_pct": round(yld * 365.0 / dte, 1),
                "otm_pct": round(otm_pct, 1),
                "assign_prob": round(abs(delta) * 100.0) if delta is not None else None,
                "if_called_pct": round(otm_pct + yld, 2),
                "p_up": round(p_up, 2) if p_up is not None else None,
                "recommend": not bullish,
                "note": ("Flint sees limited upside \u2014 harvest premium" if not bullish
                         else "hold \u2014 Flint expects upside past the strike"),
            })
        opps.sort(key=lambda x: (x["recommend"], x["annualized_pct"]), reverse=True)
        return opps

    async def _portfolio_loop(self) -> None:
        """Poll Schwab account positions (read-only) for the Portfolio tab. Never trades."""
        if not self.schwab_auth.authenticated:
            return
        import httpx as _httpx
        from .portfolio import fetch_schwab_positions
        await asyncio.sleep(9)
        while True:
            try:
                self.portfolio = await fetch_schwab_positions(self.schwab_auth)
                await self._covered_calls()
                self._publish({"type": "portfolio", "portfolio": self.portfolio})
                held = {p["symbol"] for a in self.portfolio.get("accounts", []) for p in a.get("positions", [])
                        if p.get("asset_type") in ("EQUITY", "ETF", "COLLECTIVE_INVESTMENT")
                        and (p.get("symbol") or "").isalpha() and 1 <= len(p.get("symbol") or "") <= 5}
                self._held = set(held)
                room = self.cfg.max_universe - len(self.all_symbols)
                new = [s for s in sorted(held) if s not in self.all_symbols][:max(0, room)]
                if new:                                    # model your holdings too, so the Portfolio tab shows real calls
                    self.all_symbols.extend(new)
                    self.trace.emit("system", f"portfolio holdings added to the universe: {', '.join(new)}", "act")
                    asyncio.get_running_loop().create_task(self.reconfigure())
            except _httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    self.portfolio = {"error": "Schwab account access not enabled -- add the Accounts product and re-run flint schwab-auth"}
                    self._publish({"type": "portfolio", "portfolio": self.portfolio})
                    self.trace.emit("system", "Portfolio: Schwab account access not enabled", "warn")
                    return
                self.trace.emit("system", f"portfolio: HTTP {e.response.status_code}", "warn")
            except Exception as e:  # noqa: BLE001
                self.trace.emit("system", f"portfolio: {type(e).__name__}", "warn")
            await asyncio.sleep(max(20.0, self.cfg.portfolio_seconds))

    async def _market_status_loop(self) -> None:
        """Poll Finnhub for the holiday-aware US market status (open/pre/post/closed). The
        deep-backfill and idle-training loops wait on this status, so the poll checks for a
        key on every pass: one saved from the Control panel starts it without a restart."""
        while True:
            polled = await self._poll_market_status()
            await asyncio.sleep(60 if polled else 5)

    async def _poll_market_status(self) -> bool:
        """One market-status request. Returns False, without touching the network, while there
        is no Finnhub key. A rejected key is said once in the trace instead of leaving the
        status silently unknown (which keeps every closed-market learner off)."""
        key = self.cfg.finnhub_key
        if not key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://finnhub.io/api/v1/stock/market-status",
                                params={"exchange": "US", "token": key})
            if r.status_code == 200:
                d = r.json()
                ms = {"isOpen": bool(d.get("isOpen")), "session": d.get("session"),
                      "holiday": d.get("holiday"), "t": d.get("t")}
                self._status_http = 200
                if ms != self.market_status:
                    self.market_status = ms
                    self._publish({"type": "status", "status": self.status_dict()})
            elif r.status_code != self._status_http:
                self._status_http = r.status_code
                self.trace.emit("system", f"Finnhub market status returned HTTP {r.status_code}"
                                          + (" (check the key)" if r.status_code in (401, 403) else "")
                                          + "; backfill and idle training wait on this status", "warn")
        except Exception:  # noqa: BLE001
            pass
        return True

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
        elif action == "refresh_universe":
            self._universe_refreshed = ""                 # the loop runs the next pass the pool allows
            return {"ok": True, "universe": "refresh queued"}
        elif action == "backtest":
            self._backtest_requested = True
            return {"ok": True, "backtest": "running" if self._backtesting else "queued"}
        elif action == "backtest_stop":
            self._backtest_stop = True
            return {"ok": True, "backtest": "stopping" if self._backtesting else "idle"}
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
