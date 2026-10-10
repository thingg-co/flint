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
    train_in_session: bool = _env("TRAIN_IN_SESSION", "0") not in ("0", "false", "no")   # let the idle-training loop run while the market is open too (a step can delay a forecast by one step, ~2 s on the Spark)
    idle_train_epochs: float = _env("IDLE_TRAIN_EPOCHS", 2.0)   # while the market is closed, keep training on the replay up to this many passes over it per closed session (0 = off); otherwise the GPU idles between backfill chunks
    overfit_patience: int = _env("OVERFIT_PATIENCE", 3)   # idle training stops for the session when the held-out pinball (5% of windows never trained on) fails to improve over this many 40-step checks (0 = off)
    deep_backfill_days: float = _env("DEEP_BACKFILL_DAYS", 30.0)  # while the market is closed, keep fetching history back this far and train on it (0 = off); the cap on market-data use
    coinbase_ws: str = "wss://ws-feed.exchange.coinbase.com"
    coinbase_rest: str = "https://api.exchange.coinbase.com"

    # Problem definition
    window: int = _env("WINDOW", 64)      # bars of context fed to the model
    horizon: int = _env("HORIZON", 12)    # bars ahead (12 x 5min = 1h forecast)
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    # The target: "absolute" trains on each name's log return; "relative" on the return minus beta
    # times the index's (the market-relative abnormal return), with the index hedged in the book.
    # The replay stores absolute labels either way, so this is a run-time switch, not a reset.
    target: str = _env("TARGET", "absolute")
    index_symbols: str = _env("INDEX_SYMBOLS", "SPY,QQQ,IWM,SMH,XLF")  # "the market" for the relative target: modeled index names, first = the prior; unmodeled ones are ignored
    hedge_index: bool = _env("HEDGE_INDEX", "1") not in ("0", "false", "no")  # relative target: hold, per index, the weight that cancels the book's loading on it
    hedge_min: float = _env("HEDGE_MIN", 0.01)          # an index hedge smaller than this share of the book is not worth its cost
    beta_windows: int = _env("BETA_WINDOWS", 6000)      # label rows the loadings are fit on (about 77 sessions)
    beta_shrink: float = _env("BETA_SHRINK", 400.0)     # pseudo-observations behind the prior (1 on the first index, 0 on the rest)

    # Compute + auto-sizing (benchmarks the machine on first run and picks the biggest
    # model that still trains in real time; cached to <state_dir>/machine.json).
    device: str = _env("DEVICE", "auto")               # auto | cpu | cuda | mps
    compile: bool = _env("COMPILE", "0") not in ("0", "false", "no")   # torch.compile the net (CUDA only): fuses the tap slicing/concat and expert heads; a few minutes of compile at start, falls back to eager on failure
    auto_size: bool = _env("AUTO_SIZE", "1") not in ("0", "false", "no")
    autotune_util: float = _env("AUTOTUNE_UTIL", 0.7)  # fraction of the bar interval training may use
    max_warmup_seconds: float = _env("MAX_WARMUP_SECONDS", 180.0)  # go-live warmup ceiling; caps auto model size (this, not the bar cadence, is the binding budget; raise it on a box that restarts rarely)

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
    label_cap_bps: float = _env("LABEL_CAP_BPS", 2000.0)    # cap on label magnitude in bps (0 = off); treats labels exceeding cap as carry-forward bars
    direction_threshold_bps: float = _env("DIRECTION_THRESHOLD_BPS", 10.0)  # deadband: P(up)=P(ret>+t), P(down)=P(ret<-t); the gap is P(flat)
    batch_size: int = _env("BATCH_SIZE", 16)   # step time grows ~linearly with batch (activation-bandwidth bound, even on a GPU), so a bigger batch only buys a smaller model under the step budget
    steps_per_label: int = _env("STEPS_PER_LABEL", 2)
    replay_size: int = _env("REPLAY_SIZE", 4096)
    replay_gb: float = _env("REPLAY_GB", 0.0)      # replay budget in GB; when set, the window count follows the universe (windows = GB / (names x window x features x 4 B)) and replay_size is ignored
    recent_frac: float = _env("RECENT_FRAC", 0.3)   # share of each batch drawn from the newest samples
    recent_n: int = _env("RECENT_N", 256)
    warmup_steps: int = _env("WARMUP_STEPS", 60)   # training steps on backfilled history before going live
    input_noise: float = _env("INPUT_NOISE", 0.1)  # gaussian noise on standardized inputs during training
    min_labels: int = _env("MIN_LABELS", 48)        # live (out-of-sample) labels before suggestions are counted
    band_gamma: float = _env("BAND_GAMMA", 0.05)     # adaptive conformal step for the band scale
    temper_lr: float = _env("TEMPER_LR", 0.01)       # online temperature step for P(up)
    torch_threads: int = _env("TORCH_THREADS", 2)   # overridden by autotune (up to half the cores on a GPU box)

    # Suggestion policy
    extended_hours: bool = _env("EXTENDED_HOURS", "0") not in ("0", "false", "no")  # allow long-stock entries/exits in Schwab's extended sessions (4:00-9:30, 16:00-20:00 ET); puts and straddles stay regular-hours (options do not trade then)
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
    move_floor_bps: float = _env("MOVE_FLOOR_BPS", 8.0)
    # Risk gates (policy only; the model is untouched). All live controls.
    skill_min_n: int = _env("SKILL_MIN_N", 8)                # matured direction calls a name needs before it may trade
    confirm_bars: int = _env("CONFIRM_BARS", 2)              # the same side must persist this many bars before entry
    stock_min_hold_bars: int = _env("STOCK_MIN_HOLD_BARS", 6)  # a stock position is kept at least this long unless the signal reverses
    # The cost edge beyond the quoted spread (flint/policy.py): square-root price impact of the order
    # the signal would place, the hourly decay of the put a bearish view buys, and an illiquidity gate
    # on Amihud's measure (the price move the order would buy) that works where no quote exists.
    impact_eta: float = _env("IMPACT_ETA", 0.1)              # square-root impact coefficient: cost = eta * sigma_daily * sqrt(Q / ADV)
    max_impact_bps: float = _env("MAX_IMPACT_BPS", 10.0)     # a name whose Amihud illiquidity says the order would move it more than this does not trade
    theta_in_edge: bool = _env("THETA_IN_EDGE", "1") not in ("0", "false", "no")  # charge a put's hourly decay against a bearish signal's edge
    straddle_edge: float = _env("STRADDLE_EDGE", 1.0)        # a straddle needs the model's expected |move| above this times the implied one-sigma move, plus the decay
    # Exits on stock positions (0 = off) and a daily loss limit that stops new entries for the day.
    stop_loss_bps: float = _env("STOP_LOSS_BPS", 200.0)      # close a stock position down this much from entry, min hold or not
    take_profit_bps: float = _env("TAKE_PROFIT_BPS", 0.0)    # close a stock position up this much from entry (0: the forecast decides)
    max_hold_bars: int = _env("MAX_HOLD_BARS", 0)            # close a stock position held this many bars (0: the forecast decides)
    exit_cooldown_bars: int = _env("EXIT_COOLDOWN_BARS", 12)  # after an exit the name waits this many bars before re-entering
    daily_loss_limit: float = _env("DAILY_LOSS_LIMIT", 0.02)  # fraction of the day's opening equity lost before new entries stop for the day (0 = off)
    max_spread_bps: float = _env("MAX_SPREAD_BPS", 25.0)     # names quoting wider than this do not trade; the live spread also replaces cost_bps when larger
    min_price: float = _env("MIN_PRICE", 5.0)                # names under this price do not trade (penny names: fake spreads, fake fills)
    size_by_coverage: bool = _env("SIZE_BY_COVERAGE", "1") not in ("0", "false", "no")  # scale size by band coverage / 0.8, so an overconfident model trades small
    option_max_frac: float = _env("OPTION_MAX_FRAC", 0.02)   # premium per option position, as a fraction of equity     # min |expected move| (bps) to take a side; kills false confidence from flat/degenerate-IQR series
    min_hit_rate: float = _env("MIN_HIT_RATE", 0.5)         # only trade once directional skill (hit_ema) beats a coin flip
    paper_min_trade_frac: float = _env("PAPER_MIN_TRADE_FRAC", 0.015)  # skip paper rebalances smaller than this share of equity (anti-churn)
    put_min_hold_bars: int = _env("PUT_MIN_HOLD_BARS", 3)   # hold a paper put at least this many bars before closing (anti-churn)
    kelly_fraction: float = _env("KELLY_FRACTION", 0.15)   # fractional-Kelly sizing: weight = this * |q50|/IQR (risk-adjusted edge).
    #                                                      Calibrated to the per-name weight cap so tradeable scores (~0.35-1.0) map
    #                                                      into (0, max_weight] and differentiate; only the strongest edges hit the cap.
    straddle_enabled: bool = _env("STRADDLE", True, cast=lambda v: v.lower() in ("1", "true", "yes", "on"))  # open paper straddles on fat-tail forecasts (big move, direction unclear)
    straddle_band_bps: float = _env("STRADDLE_BAND_BPS", 120.0)  # min calibrated q10-q90 width (bps) to consider a straddle
    straddle_budget: float = _env("STRADDLE_BUDGET", 0.01)  # premium spent per straddle, as a fraction of equity (also the max loss)
    straddle_max: int = _env("STRADDLE_MAX", 3)             # max concurrent paper straddles
    straddle_min_coverage: float = _env("STRADDLE_MIN_COVERAGE", 0.55)  # band coverage EMA required before trusting the width forecast
    straddle_hold_bars: int = _env("STRADDLE_HOLD_BARS", 12)  # hold a straddle at least this many bars (~the forecast horizon)

    # Signals + Burry overlay
    muted_symbols: str = _env("MUTED", "")           # symbols to watch-but-not-suggest (comma list)
    signals_off: str = _env("SIGNALS_OFF", "")          # signal providers off by default: wsb, feargreed, derivatives, scion
    signals_minutes: float = _env("SIGNALS_MINUTES", 5.0)   # how often to refresh exogenous signals (heavy: WSB/gurus/13F)
    market_scan_seconds: float = _env("MARKET_SCAN_SECONDS", 45.0)  # tight loop for movers/sectors/breadth (skips the heavy radar)
    portfolio_seconds: float = _env("PORTFOLIO_SECONDS", 60.0)      # how often to poll Schwab account positions (read-only)
    operator_half_life: float = _env("OPERATOR_HALF_LIFE", 1800.0)  # decay of an injected human note (seconds)
    max_universe: int = _env("MAX_UNIVERSE", 256)            # cap on modeled symbols (cross-attention + data-rate limit)
    universe_refresh: bool = _env("UNIVERSE_REFRESH", "1") not in ("0", "false", "no")  # once a day after the close: core + every holding + the most interesting names (a live control)
    interesting_count: int = _env("INTERESTING_COUNT", 40)        # how many "interesting" slots the universe keeps beyond core and holdings (a live control)
    interesting_min_dollar_volume: float = _env("INTERESTING_MIN_DOLLAR_VOLUME", 2e7)  # a radar name needs this much daily dollar volume to be a candidate
    interesting_grace_days: int = _env("INTERESTING_GRACE_DAYS", 14)   # a promoted name is demoted after this many days without ranking in the top 1.5 x count
    interesting_min_tenure_days: int = _env("INTERESTING_MIN_TENURE_DAYS", 5)  # ... and never before this many days in the universe
    radar_top: int = _env("RADAR_TOP", 750)                  # movers watchlist size (display + breadth); supply is ~8 screeners x radar_count, deduped
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
    replay_save_minutes: float = _env("REPLAY_SAVE_MINUTES", 30.0)  # the replay buffer (tens of GB when full) is written at most this often, and at shutdown; the model every checkpoint
    backtest_weekends: bool = _env("BACKTEST_WEEKENDS", "1") not in ("0", "false", "no")  # walk-forward backtest over the replay on Saturday/Sunday ET (a live control; action "backtest" runs one any time)
    backtest_hours: float = _env("BACKTEST_HOURS", 30.0)      # time budget for a run; the per-day training budget shrinks to fit it (0 = uncapped)
    backtest_night_epochs: float = _env("BACKTEST_NIGHT_EPOCHS", 2.0)  # passes over the history seen so far per simulated day, as the idle trainer does per closed session
    backtest_days: int = _env("BACKTEST_DAYS", 0)             # score only the last N days (0 = all timestamped history)
    backtest_cost_bps: float = _env("BACKTEST_COST_BPS", 5.0)  # round-trip cost per unit of position in the backtest (the live book charges the quoted spread)
    backtest_warm_days: int = _env("BACKTEST_WARM_DAYS", 10)   # training-only days before scoring starts, like warmup
    backtest_target: str = _env("BACKTEST_TARGET", "both")    # absolute | relative | both (two runs, each on half the budget, reported side by side)
    memory_floor_gb: float = _env("MEMORY_FLOOR_GB", 10.0)   # GB that must stay free: the backtest sizes its net to fit above it and stops below it; deep paging pauses near it (a live control)

    def __post_init__(self):
        if not self.sources_on and self.feed and self.feed != "auto":
            self.sources_on = self.feed

    def replay_windows(self, n_features: int) -> int:
        """The replay capacity in windows: replay_gb spread over the current universe when set
        (so growing the universe shrinks the window count instead of the free memory), else
        replay_size."""
        gb = float(getattr(self, "replay_gb", 0.0) or 0.0)
        if gb > 0:
            per = max(1, self.n_assets) * int(self.window) * int(n_features) * 4
            return max(256, int(gb * 1e9 // per))
        return int(self.replay_size)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)
