# flint

A small neural network that watches live markets, forecasts each stock's return
distribution over the next hour, keeps training on its own predictions as they resolve,
and layers the trading ethos of well-known investors on top. It runs continuously and
shows everything in a browser page: an at-a-glance brief, per-symbol forecast cards, a
whole-market overview with a 250-name movers radar, and a control panel for the universe,
data sources, keys, and signals.

It is a research toy. The forecasts are in basis points, the paper P&L ignores execution
costs unless you set them, and none of it is advice. Do not trade on it.

## Run

```
uv sync
uv run playwright install chromium   # only needed for the headless-browser news source
uv run flint
```

Open http://127.0.0.1:8000. The default universe is US equities plus gold and silver
(NVDA, AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, … and more). On start it benchmarks the
machine, picks a model size that trains in real time, backfills recent history, trains on
it, and switches to the live feeds. Crypto sources are available but off by default; add
crypto or any ticker from the control panel or with `--symbols`.

```
uv run flint --symbols AAPL,NVDA,BTC-USD,XAU-USD    # any mix of equities, crypto, metals
uv run flint --port 9000 --no-news
```

Every field in `flint/config.py` can also be set with a `FLINT_<NAME>` environment
variable, e.g. `FLINT_BAR_SECONDS=60 FLINT_MAX_UNIVERSE=48`.

## Compute — GPU and auto-sizing

On start flint detects the best device (CUDA, then Apple's Metal/MPS, then CPU) and, on a
new machine, benchmarks a ladder of model sizes to pick the largest that still trains
within the bar interval. The result is cached to `state/machine.json`, so shared to a
different machine it adapts on its own — a big model on a GPU box, a small one on a
laptop. On Apple Silicon it also picks the thread count that is actually fastest rather
than the most threads (over-threading spills onto the slow efficiency cores).

## The model

Ticks are aggregated into 5-minute bars, aligned across assets. Each bar becomes a set of
features per asset: price microstructure (return, range, volume, order-flow imbalance,
spread, trade count, close position, momentum, realized volatility) plus slow-moving
exogenous signals (news tone, WallStreetBets, Fear & Greed, perp funding/OI/long-short,
the investor-council consensus and ethos bias, fundamentals, market breadth, and VIX).

FlintNet takes the last N bars of every asset and predicts the 10/25/50/75/90 quantiles
of each asset's return over the next hour plus a probability the return is positive. It
combines a per-asset dilated causal convolution stack read out at several scales,
cross-asset attention, and a regime gate mixing several expert heads. Quantiles are
non-crossing by construction. Learning is online: every bar it forecasts, the horizon
later the realized return becomes a label, and a few optimizer steps run per label from a
replay buffer. Two calibrators fitted only on out-of-sample outcomes correct the raw
forecasts. Checkpoints save to `state/`. A suggestion comes from the calibrated forecast,
then the investor-council overlay adjusts it.

## Data sources

Market data is a registry of sources you toggle from the control panel; each symbol is
served by the highest-priority live source, with automatic failover. Equities: Charles
Schwab, Financial Modeling Prep, Finnhub, EODHD, Yahoo, Alpha Vantage. Crypto: Coinbase,
Binance, Bitfinex, Gemini, Kraken, CoinGecko. Commodities: gold-api (spot metals), Yahoo
(futures).

- **Financial Modeling Prep** is the primary equity feed — reliable 5-minute intraday
  history for backfill plus quotes (Starter plan or higher).
- **Finnhub** — real-time equity quotes and websocket; also fundamentals (P/E, EPS,
  margins, growth) fed to the model.
- **Charles Schwab** — real-time quotes + price history over OAuth, market data only. Run
  `flint schwab-auth` once after registering an app to enable it.
- **Alpha Vantage / EODHD** — news sentiment and delayed quotes; both free-tier limited.

Keys go in gitignored files (`fmp.json`, `finnhub.json`, `eodhd.json`, `keys`, …) or the
**API keys panel** in the control panel, and are masked everywhere in the UI, API, and
logs. The keys panel links to where to get each one.

## Universe

The control panel lets you curate what the model trades: add any ticker, remove one, mute
to pause it (muting frees its compute and API calls), or pull in the current top movers
from the radar. Changing the universe rebuilds the model for the new set and the
autotuner resizes it. There is a cap (`max_universe`, default 64) because cross-asset
attention is quadratic and the feeds are rate-limited; the radar watches all 250 movers
regardless.

## Signals and the investor council

Every few minutes flint refreshes exogenous signals: WallStreetBets attention and
sentiment, the Fear & Greed index, and Binance perp positioning. The **investor council**
tracks the disclosed 13F positions of Michael Burry, Warren Buffett, Charlie Munger, Bill
Ackman, Steven Cohen, Carl Icahn, and Peter Schiff (puts read as bearish), and each also
carries an ethos profile encoding their documented philosophy. The combined council
drives an overlay on every suggestion — margin of safety, fade-the-crowd, a holdings
counterpoint, and a macro-bearish drag — and each adjustment names the investor driving
it.

## The interface

- **Brief** — an instant market synthesis: regime, breadth/Fear-&-Greed/VIX gauges, sector
  rotation, top movers, and the model's strongest calls.
- **Dashboard** — a forecast card per symbol (the session's chart plus the forecast fan),
  ranked by volatility, with a whole-market panel and the 250-name movers grid below.
- **Control panel** — the universe, data sources, API keys, the investor council, the
  movers radar, runtime policy controls, and one live console per stage of the system.

Everything is on the websocket, also reachable over HTTP: `GET /api/state`, `/api/news`,
`/api/sources`, `/api/signals`, `/api/keys`, and `POST /api/control`.

## Layout

```
flint/config.py    settings and FLINT_* overrides
flint/autotune.py  device detection + first-run model auto-sizing
flint/feed.py      shared Tick type and the offline simulator
flint/sources.py   market-data source registry + manager
flint/schwab.py    Schwab OAuth token manager (market data only)
flint/bars.py      tick to bar aggregation, aligned across symbols
flint/features.py  feature engineering and running normalization
flint/model.py     FlintNet and the loss
flint/learner.py   replay buffer, optimizer loop (GPU/MPS/CPU), checkpoints
flint/signals.py   WSB, derivatives, fundamentals, and the investor council
flint/market.py    whole-market scanner and the movers radar
flint/news.py      news hub: headless-browser skimmer + Alpha Vantage sentiment
flint/engine.py    orchestration, policy + council overlay, universe, metrics, controls
flint/server.py    FastAPI app, websocket, control + keys endpoints
web/               dashboard (no build step, plain HTML/CSS/JS)
tests/             pytest suite: risk rules, gates, checkpoints, bars, features, sources, routes
```

`uv run pytest -q tests` runs the suite without a model or a feed. `uv run flint check` reports
what a start would do on this machine (device, free memory, the cached model size) and exits
non-zero when the model would not fit.

## License

PolyForm Noncommercial 1.0.0: free for personal, hobby, research and other noncommercial use.
Commercial use needs a separate license. See LICENSE and NOTICE.

© 2026 thingg LLC · [thingg.co](https://thingg.co)
