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
    return [s.strip().upper() for s in _env("SYMBOLS", "NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA,AVGO,ORCL,NFLX,AMD,QCOM,MU,INTC,TXN,AMAT,ARM,SMCI,TSM,CSCO,CRM,ADBE,NOW,PANW,PLTR,UBER,IBM,JPM,BAC,WFC,GS,MS,AXP,V,MA,SCHW,LLY,UNH,JNJ,ABBV,MRK,PFE,TMO,ABT,WMT,COST,KO,PEP,PG,MCD,NKE,HD,DIS,CVX,XOM,COP,BA,CAT,HON,GE,MSTR,COIN,HOOD,F,SPY,QQQ,IWM,SMH,XLF,LRCX,KLAC,ADI,MRVL,MCHP,NXPI,ON,ANET,DELL,HPQ,HPE,WDC,GLW,TEL,APH,MPWR,INTU,ADSK,FTNT,CRWD,ZS,SNOW,DDOG,NET,MDB,TEAM,WDAY,SNPS,CDNS,SHOP,SPOT,ABNB,DASH,RBLX,PINS,SNAP,RDDT,EA,TTWO,ROKU,U,APP,DOCU,OKTA,HUBS,T,VZ,TMUS,CMCSA,CHTR,WBD,C,USB,PNC,TFC,COF,BLK,SPGI,CME,ICE,MCO,AON,PGR,TRV,ALL,MET,PRU,AIG,SYF,CB,BX,KKR,APO,AJG,DHR,BMY,AMGN,GILD,VRTX,REGN,ISRG,MDT,SYK,BSX,CI,CVS,HCA,ELV,ZTS,BDX,MCK,IDXX,DXCM,EW,SBUX,CMG,TGT,LOW,TJX,ROST,DG,DLTR,YUM,BKNG,MAR,HLT,RCL,CCL,GM,LULU,ULTA").split(",") if s.strip()]


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


def _alpaca() -> tuple[str, str]:
    kid = _env("ALPACA_KEY_ID", ""); sec = _env("ALPACA_SECRET_KEY", "")
    if not (kid and sec):
        for path in ("alpaca.json", ".alpaca.json"):
            try:
                import json
                d = json.loads(open(path).read())
                kid = kid or d.get("key_id", ""); sec = sec or d.get("secret_key", ""); break
            except (OSError, ValueError):
                pass
    return kid, sec


def _tradier() -> str:
    tok = _env("TRADIER_TOKEN", "")
    if tok:
        return tok
    for path in ("tradier.json", ".tradier.json"):
        try:
            import json
            return json.loads(open(path).read()).get("token", "")
        except (OSError, ValueError):
            pass
    return ""


def _etrade() -> tuple[str, str]:
    ck = _env("ETRADE_CONSUMER_KEY", ""); cs = _env("ETRADE_CONSUMER_SECRET", "")
    if not (ck and cs):
        for path in ("etrade.json", ".etrade.json"):
            try:
                import json
                d = json.loads(open(path).read())
                ck = ck or d.get("consumer_key", ""); cs = cs or d.get("consumer_secret", ""); break
            except (OSError, ValueError):
                pass
    return ck, cs


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
    schwab_seconds: float = _env("SCHWAB_SECONDS", 5.0)        # seconds between real-time quote polls
    alpaca_creds: tuple = field(default_factory=_alpaca)       # (key_id, secret_key) from env or alpaca.json
    alpaca_feed: str = _env("ALPACA_FEED", "iex")             # iex (free) or sip (paid)
    alpaca_seconds: float = _env("ALPACA_SECONDS", 6.0)
    tradier_token: str = field(default_factory=_tradier)       # brokerage access token from env or tradier.json
    tradier_seconds: float = _env("TRADIER_SECONDS", 6.0)
    etrade_creds: tuple = field(default_factory=_etrade)       # (consumer_key, consumer_secret) from env or etrade.json
    etrade_token_file: str = _env("ETRADE_TOKEN_FILE", "")     # defaults to <state_dir>/etrade_tokens.json
    etrade_seconds: float = _env("ETRADE_SECONDS", 6.0)
    finnhub_key: str = field(default_factory=_finnhub_key)     # Finnhub API key (env FLINT_FINNHUB_KEY or finnhub.json)
    finnhub_seconds: float = _env("FINNHUB_SECONDS", 15.0)     # seconds between Finnhub quote heartbeats
    eodhd_key: str = field(default_factory=_eodhd_key)         # EODHD API token (env FLINT_EODHD_KEY or eodhd.json)
    eodhd_seconds: float = _env("EODHD_SECONDS", 20.0)         # seconds between EODHD delayed-quote polls
    fmp_key: str = field(default_factory=_fmp_key)             # Financial Modeling Prep (reliable 5-min history)
    fmp_seconds: float = _env("FMP_SECONDS", 20.0)            # seconds between FMP quote polls
    ibkr_enabled: bool = _env("IBKR_ENABLED", False, cast=lambda v: v.lower() in ("1", "true", "yes", "on"))
    ibkr_host: str = _env("IBKR_HOST", "127.0.0.1")           # IB Gateway / TWS host
    ibkr_port: int = _env("IBKR_PORT", 4002)                  # Gateway paper 4002 / live 4001; TWS paper 7497 / live 7496
    ibkr_client_id: int = _env("IBKR_CLIENT_ID", 17)          # any unused API client id
    ibkr_market_data_type: int = _env("IBKR_MARKET_DATA_TYPE", 1)  # 1 live, 2 frozen, 3 delayed, 4 delayed-frozen (auto-falls back)
    anthropic_key: str = field(default_factory=_anthropic_key)  # optional Claude key for the narrative brief
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")   # local LLM runtime; the brief runs fully on-device
    brief_model: str = _env("BRIEF_MODEL", "qwen3.5:latest")          # local writer model (bump to qwen3.6:27b / deepseek-r1:32b for more quality, much slower)
    brief_small_model: str = _env("BRIEF_SMALL_MODEL", "qwen3.5:latest")  # fast local analysts that pre-digest each data slice
    brief_minutes: float = _env("BRIEF_MINUTES", 15.0)         # auto-refresh cadence for the narrative brief
    brief_timeout: float = _env("BRIEF_TIMEOUT", 1800.0)       # max seconds for the writer model (local, can be slow)
    brief_enabled: bool = _env("BRIEF_ENABLED", False, cast=lambda v: v.lower() in ("1", "true", "yes", "on"))  # local LLM brief on/off (off for now)
    brief_backend: str = _env("BRIEF_BACKEND", "ollama")     # local LLM backend: "ollama" (default) or "openai" (vLLM / any OpenAI-compatible local server)
    brief_openai_base: str = _env("BRIEF_OPENAI_BASE", "http://localhost:8001/v1")  # OpenAI-compatible base URL when brief_backend=openai (e.g. vLLM `vllm serve <model> --port 8001`). Local only, never cloud.
    av_rate_seconds: float = _env("AV_RATE_SECONDS", 1.0)     # global floor between ANY two Alpha Vantage calls
    bar_seconds: float = _env("BAR_SECONDS", 300.0)   # 5-minute bars (FMP provides reliable 5-min history)
    backfill_seconds: float = _env("BACKFILL_SECONDS", 432000.0)  # ~5 trading days: enough real bars to fill the model window + training
    backfill_pages: int = _env("BACKFILL_PAGES", 40)  # max REST pages per symbol
    deep_backfill_days: float = _env("DEEP_BACKFILL_DAYS", 30.0)  # while the market is closed, keep fetching history back this far and train on it (0 = off); the cap on market-data use
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
    max_warmup_seconds: float = _env("MAX_WARMUP_SECONDS", 180.0)  # go-live warmup ceiling; caps auto model size

    # Model (overridden by auto-sizing unless auto_size is off)
    d_model: int = _env("D_MODEL", 48)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_experts: int = _env("N_EXPERTS", 3)
    n_heads: int = _env("N_HEADS", 4)
    dropout: float = _env("DROPOUT", 0.15)

    # Online learning
    lr: float = _env("LR", 5e-4)
    weight_decay: float = _env("WEIGHT_DECAY", 1e-2)
    label_smoothing: float = _env("LABEL_SMOOTHING", 0.1)   # softens the P(up)/P(down) targets so the net cannot claim ~99% certainty
    direction_threshold_bps: float = _env("DIRECTION_THRESHOLD_BPS", 10.0)  # deadband: P(up)=P(ret>+t), P(down)=P(ret<-t); the gap is P(flat)
    batch_size: int = _env("BATCH_SIZE", 16)   # smaller batch: half the activation memory (this Mac is memory-tight at a big universe)
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
    option_commission: float = _env("OPTION_COMMISSION", 0.65)  # per-contract commission for paper option trades
    option_dte: int = _env("OPTION_DTE", 35)                # target days-to-expiry when opening a put for a short
    option_features_seconds: float = _env("OPTION_FEATURES_SECONDS", 600.0)  # cadence to refresh ATM IV / skew features
    covered_call_seconds: float = _env("COVERED_CALL_SECONDS", 300.0)  # cadence to refresh covered-call opportunities for the Portfolio tab
    covered_call_otm: float = _env("COVERED_CALL_OTM", 0.05)   # target out-of-the-money fraction for the suggested covered-call strike
    covered_call_delta: float = _env("COVERED_CALL_DELTA", 0.30)  # target call delta (~assignment probability) for covered-call selection
    max_size: float = _env("MAX_SIZE", 1.0)
    move_floor_bps: float = _env("MOVE_FLOOR_BPS", 8.0)     # min |expected move| (bps) to take a side; kills false confidence from flat/degenerate-IQR series
    min_hit_rate: float = _env("MIN_HIT_RATE", 0.5)         # only trade once directional skill (hit_ema) beats a coin flip
    paper_min_trade_frac: float = _env("PAPER_MIN_TRADE_FRAC", 0.015)  # skip paper rebalances smaller than this share of equity (anti-churn)
    put_min_hold_bars: int = _env("PUT_MIN_HOLD_BARS", 3)   # hold a paper put at least this many bars before closing (anti-churn)
    kelly_fraction: float = _env("KELLY_FRACTION", 0.15)   # fractional-Kelly sizing: weight = this * |q50|/IQR (risk-adjusted edge).
    #                                                      Calibrated to the per-name weight cap so tradeable scores (~0.35-1.0) map
    #                                                      into (0, max_weight] and differentiate; only the strongest edges hit the cap.

    # Signals + Burry overlay
    muted_symbols: str = _env("MUTED", "")           # symbols to watch-but-not-suggest (comma list)
    signals_off: str = _env("SIGNALS_OFF", "")          # signal providers off by default: wsb, feargreed, derivatives, scion
    signals_minutes: float = _env("SIGNALS_MINUTES", 5.0)   # how often to refresh exogenous signals (heavy: WSB/gurus/13F)
    market_scan_seconds: float = _env("MARKET_SCAN_SECONDS", 45.0)  # tight loop for movers/sectors/breadth (skips the heavy radar)
    portfolio_seconds: float = _env("PORTFOLIO_SECONDS", 60.0)      # how often to poll Schwab account positions (read-only)
    operator_half_life: float = _env("OPERATOR_HALF_LIFE", 1800.0)  # decay of an injected human note (seconds)
    radar_top: int = _env("RADAR_TOP", 250)                 # how many market-wide movers to watch
    max_universe: int = _env("MAX_UNIVERSE", 256)            # cap on modeled symbols (cross-attention + data-rate limit)
    radar_top: int = _env("RADAR_TOP", 750)                  # movers watchlist size (display + breadth); Yahoo supply ~1000
    radar_count: int = _env("RADAR_COUNT", 250)              # per-screener fetch (Yahoo hard cap is 250)
    burry_enabled: bool = _env("BURRY", "1") not in ("0", "false", "no")
    burry_aggr: float = _env("BURRY_AGGR", 0.7)         # 0..1: how hard the contrarian overlay fades crowded trades
    burry_fade_at: float = _env("BURRY_FADE_AT", 0.45)  # crowding magnitude above which aligned trades get faded
    burry_safety: float = _env("BURRY_SAFETY", 0.5)     # margin-of-safety penalty on asymmetric downside

    # News skimmer
    news_enabled: bool = _env("NEWS", "1") not in ("0", "false", "no")
    news_minutes: float = _env("NEWS_MINUTES", 3.0)
    news_browser: bool = _env("NEWS_BROWSER", "1") not in ("0", "false", "no")
    news_off: str = _env("NEWS_OFF", "")           # news source ids off by default: browser, alphavantage
    news_sources: str = _env("NEWS_SOURCES", "")   # "Name|https://url,Name2|https://url2" to override the browser defaults

    # Server / UI
    host: str = _env("HOST", "127.0.0.1")
    port: int = _env("PORT", 8000)
    tick_hz: float = _env("TICK_HZ", 2.0)
    chart_bars: int = _env("CHART_BARS", 160)   # ~13h of trading at 5-min bars
    paper_start: float = _env("PAPER_START", 100000.0)  # starting cash for the paper-trading book
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
