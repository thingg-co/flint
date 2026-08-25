"""Runtime configuration. Every field can be overridden with a FLINT_<NAME> environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default, cast=None):
    raw = os.environ.get(f"FLINT_{name}")
    if raw is None:
        return default
    if cast is not None:
        return cast(raw)
    return type(default)(raw)


def _symbols() -> list[str]:
    return [s.strip().upper() for s in _env("SYMBOLS", "NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA,AVGO,AMD,QCOM,MU,ORCL,CRM,PLTR,MSTR,COIN,SMCI,NFLX,DIS,UBER,INTC,JPM,BAC,AXP,V,MA,KO,WMT,COST,LLY,CVX,XOM,SPY,QQQ,XAU-USD,XAG-USD").split(",") if s.strip()]


def _eodhd_key() -> str:
    key = _env("EODHD_KEY", "")
    if key:
        return key
    for path in ("eodhd.json", ".eodhd.json"):
        try:
            import json
            return json.loads(open(path).read()).get("key", "")
        except (OSError, ValueError):
            pass
    return ""


def _fmp_key() -> str:
    key = _env("FMP_KEY", "")
    if key:
        return key
    for path in ("fmp.json", ".fmp.json"):
        try:
            import json
            return json.loads(open(path).read()).get("key", "")
        except (OSError, ValueError):
            pass
    return ""


def _anthropic_key() -> str:
    key = _env("ANTHROPIC_KEY", "")
    if key:
        return key
    for path in ("anthropic.json", ".anthropic.json"):
        try:
            import json
            return json.loads(open(path).read()).get("key", "")
        except (OSError, ValueError):
            pass
    return ""


def _finnhub_key() -> str:
    key = _env("FINNHUB_KEY", "")
    if key:
        return key
    for path in ("finnhub.json", ".finnhub.json"):
        try:
            import json
            return json.loads(open(path).read()).get("key", "")
        except (OSError, ValueError):
            pass
    return ""


def _schwab() -> tuple[str, str, str]:
    key = _env("SCHWAB_APP_KEY", "")
    secret = _env("SCHWAB_APP_SECRET", "")
    callback = _env("SCHWAB_CALLBACK", "https://127.0.0.1")
    if not (key and secret):
        for path in ("schwab.json", ".schwab.json"):
            try:
                import json
                d = json.loads(open(path).read())
                key = key or d.get("app_key", "")
                secret = secret or d.get("app_secret", "")
                callback = d.get("callback", callback)
                break
            except (OSError, ValueError):
                pass
    return key, secret, callback


def _av_key() -> str:
    key = _env("AV_KEY", "")
    if key:
        return key
    for path in ("keys", ".keys"):
        try:
            with open(path) as f:
                toks = f.read().split()
                if toks:
                    return toks[0]
        except OSError:
            pass
    return ""


@dataclass
class Config:
    # Market data
    symbols: list[str] = field(default_factory=_symbols)
    feed: str = _env("FEED", "auto")                 # auto | or a single source id to force
    sources_on: str = _env("SOURCES_ON", "")         # comma list to force EXACTLY these on (plus sim); empty = defaults
    sources_off: str = _env("SOURCES_OFF", "coingecko,kraken,bitfinex,gemini")  # available but off until toggled
    av_key: str = field(default_factory=_av_key)
    av_quote_seconds: float = _env("AV_QUOTE_SECONDS", 900.0)  # min seconds between Alpha Vantage quote polls (free tier is 25/day)
    av_news_minutes: float = _env("AV_NEWS_MINUTES", 30.0)     # min minutes between Alpha Vantage news pulls
    schwab_creds: tuple = field(default_factory=_schwab)       # (app_key, app_secret, callback) from env or schwab.json
    schwab_token_file: str = _env("SCHWAB_TOKEN_FILE", "")     # defaults to <state_dir>/schwab_tokens.json
    schwab_seconds: float = _env("SCHWAB_SECONDS", 10.0)       # seconds between real-time quote polls
    finnhub_key: str = field(default_factory=_finnhub_key)     # Finnhub API key (env FLINT_FINNHUB_KEY or finnhub.json)
    finnhub_seconds: float = _env("FINNHUB_SECONDS", 15.0)     # seconds between Finnhub quote heartbeats
    eodhd_key: str = field(default_factory=_eodhd_key)         # EODHD API token (env FLINT_EODHD_KEY or eodhd.json)
    eodhd_seconds: float = _env("EODHD_SECONDS", 20.0)         # seconds between EODHD delayed-quote polls
    fmp_key: str = field(default_factory=_fmp_key)             # Financial Modeling Prep (reliable 5-min history)
    fmp_seconds: float = _env("FMP_SECONDS", 20.0)            # seconds between FMP quote polls
    anthropic_key: str = field(default_factory=_anthropic_key)  # optional Claude key for the narrative brief
    brief_model: str = _env("BRIEF_MODEL", "claude-haiku-4-5-20251001")
    brief_minutes: float = _env("BRIEF_MINUTES", 5.0)          # cache the narrative brief this long
    av_rate_seconds: float = _env("AV_RATE_SECONDS", 1.0)     # global floor between ANY two Alpha Vantage calls
    bar_seconds: float = _env("BAR_SECONDS", 300.0)   # 5-minute bars (FMP provides reliable 5-min history)
    backfill_seconds: float = _env("BACKFILL_SECONDS", 172800.0)  # ~2 trading days of 5-min history
    backfill_pages: int = _env("BACKFILL_PAGES", 40)  # max REST pages per symbol
    coinbase_ws: str = "wss://ws-feed.exchange.coinbase.com"
    coinbase_rest: str = "https://api.exchange.coinbase.com"

    # Problem definition
    window: int = _env("WINDOW", 64)      # bars of context fed to the model
    horizon: int = _env("HORIZON", 12)    # bars ahead (12 x 5min = 1h forecast)
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    # Compute + auto-sizing (benchmarks the machine on first run and picks the biggest
    # model that still trains in real time; cached to <state_dir>/machine.json).
    device: str = _env("DEVICE", "auto")               # auto | cpu | cuda | mps
    auto_size: bool = _env("AUTO_SIZE", "1") not in ("0", "false", "no")
    autotune_util: float = _env("AUTOTUNE_UTIL", 0.7)  # fraction of the bar interval training may use

    # Model (overridden by auto-sizing unless auto_size is off)
    d_model: int = _env("D_MODEL", 48)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_experts: int = _env("N_EXPERTS", 3)
    n_heads: int = _env("N_HEADS", 4)
    dropout: float = _env("DROPOUT", 0.15)

    # Online learning
    lr: float = _env("LR", 5e-4)
    weight_decay: float = _env("WEIGHT_DECAY", 1e-2)
    batch_size: int = _env("BATCH_SIZE", 32)
    steps_per_label: int = _env("STEPS_PER_LABEL", 2)
    replay_size: int = _env("REPLAY_SIZE", 4096)
    recent_frac: float = _env("RECENT_FRAC", 0.3)   # share of each batch drawn from the newest samples
    recent_n: int = _env("RECENT_N", 256)
    warmup_steps: int = _env("WARMUP_STEPS", 60)   # training steps on backfilled history before going live
    input_noise: float = _env("INPUT_NOISE", 0.1)  # gaussian noise on standardized inputs during training
    min_labels: int = _env("MIN_LABELS", 48)        # live (out-of-sample) labels before suggestions are counted
    band_gamma: float = _env("BAND_GAMMA", 0.05)     # adaptive conformal step for the band scale
    temper_lr: float = _env("TEMPER_LR", 0.01)       # online temperature step for P(up)
    torch_threads: int = _env("TORCH_THREADS", 2)

    # Suggestion policy
    score_threshold: float = _env("SCORE_THRESHOLD", 0.35)  # |q50| / IQR needed to act
    prob_margin: float = _env("PROB_MARGIN", 0.06)          # |P(up) - 0.5| needed to act
    cost_bps: float = _env("COST_BPS", 0.0)                 # round-trip cost charged to paper P&L
    max_size: float = _env("MAX_SIZE", 1.0)

    # Signals + Burry overlay
    muted_symbols: str = _env("MUTED", "")           # symbols to watch-but-not-suggest (comma list)
    signals_off: str = _env("SIGNALS_OFF", "")          # signal providers off by default: wsb, feargreed, derivatives, scion
    signals_minutes: float = _env("SIGNALS_MINUTES", 5.0)   # how often to refresh exogenous signals
    radar_top: int = _env("RADAR_TOP", 250)                 # how many market-wide movers to watch
    max_universe: int = _env("MAX_UNIVERSE", 64)            # cap on modeled symbols (cross-attention + data-rate limit)
    burry_enabled: bool = _env("BURRY", "1") not in ("0", "false", "no")
    burry_aggr: float = _env("BURRY_AGGR", 0.7)         # 0..1: how hard the contrarian overlay fades crowded trades
    burry_fade_at: float = _env("BURRY_FADE_AT", 0.45)  # crowding magnitude above which aligned trades get faded
    burry_safety: float = _env("BURRY_SAFETY", 0.5)     # margin-of-safety penalty on asymmetric downside

    # News skimmer
    news_enabled: bool = _env("NEWS", "1") not in ("0", "false", "no")
    news_minutes: float = _env("NEWS_MINUTES", 10.0)
    news_browser: bool = _env("NEWS_BROWSER", "1") not in ("0", "false", "no")
    news_off: str = _env("NEWS_OFF", "")           # news source ids off by default: browser, alphavantage
    news_sources: str = _env("NEWS_SOURCES", "")   # "Name|https://url,Name2|https://url2" to override the browser defaults

    # Server / UI
    host: str = _env("HOST", "127.0.0.1")
    port: int = _env("PORT", 8000)
    tick_hz: float = _env("TICK_HZ", 2.0)
    chart_bars: int = _env("CHART_BARS", 160)   # ~13h of trading at 5-min bars
    log_size: int = 60

    # Persistence
    state_dir: str = _env("STATE_DIR", "state")
    checkpoint_minutes: float = _env("CHECKPOINT_MINUTES", 5.0)

    def __post_init__(self):
        if not self.sources_on and self.feed and self.feed != "auto":
            self.sources_on = self.feed

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)
