# flint

A small neural network that watches live markets, forecasts the next minute's return
distribution for a handful of assets, keeps training on its own predictions as they
resolve, and layers the trading ethos of well-known investors on top. Everything shows
up in a browser page that updates continuously: a whole-market overview, per-asset
forecast cards, a 250-name movers watch, an investor council, and a set of consoles that
stream what each stage of the system is doing.

It is a research toy. Forecasts are over 60-second horizons in basis points, the paper
P&L ignores real execution costs unless you set them, and none of it is advice. Do not
trade on it.

## Run

```
uv sync
uv run playwright install chromium   # only needed for the headless-browser news source
uv run flint
```

Open http://127.0.0.1:8000. With no arguments it tracks a mix of crypto (BTC, ETH, SOL)
and equities (NVDA, AAPL, TSLA, MSTR, PLTR). Crypto trades come from Coinbase; equities
come from Finnhub if a key is present, otherwise Yahoo. On start it backfills recent
history, trains on it for a few seconds, and switches to the live feeds.

```
uv run flint --symbols BTC-USD,ETH-USD,AAPL,NVDA,SPY   # any mix of crypto and US equities
uv run flint --feed sim                                 # offline simulator only
uv run flint --port 9000 --no-news
```

Every field in `flint/config.py` can also be set with a `FLINT_<NAME>` environment
variable, e.g. `FLINT_BAR_SECONDS=10 FLINT_SIGNALS_MINUTES=3`.

## The model

Ticks are aggregated into 5-second bars, aligned across assets. Each bar becomes 21
features per asset: nine market-microstructure features (return, range, volume, taker
imbalance, spread, trade count, close position, momentum, realized volatility) and a set
of slow-moving exogenous signals (news tone and attention, WallStreetBets sentiment and
attention, Fear & Greed, perp funding, open-interest change, long/short ratio, the
investor-council consensus and ethos bias, market breadth, and VIX). Features are
standardized with running statistics.

FlintNet takes the last 64 bars of every asset and predicts the 10/25/50/75/90 quantiles
of each asset's return over the next 12 bars plus a probability the return is positive.
It combines a per-asset dilated causal convolution stack read out at several scales,
cross-asset attention (one asset can condition on what the others just did), and a regime
gate that mixes three expert heads. Quantiles are non-crossing by construction. Learning
is online: every bar it forecasts, 12 bars later the realized return becomes a label, and
a couple of optimizer steps run per label from a replay buffer. Two calibrators fitted
only on out-of-sample outcomes correct the raw forecasts (a conformal band scale and a
temperature on the direction head). Checkpoints save to `state/` every five minutes.

A suggestion comes from the calibrated forecast: the median over the interquartile range
is the score, and the direction head has to agree with margin. BUY or SELL when both
clear their thresholds and the median covers cost, HOLD otherwise. Then the investor
council overlay adjusts it (below).

## Data sources

Market data is a registry of sources you can toggle from the control panel. Each symbol
is served by the highest-priority source that is enabled and live, with automatic
failover. Crypto: Coinbase and Binance (websocket trades), Kraken and CoinGecko (poll).
Equities: Charles Schwab, Finnhub, EODHD, Yahoo Finance, Alpha Vantage, in that priority
order. A simulator is the offline fallback.

- **Finnhub** and **EODHD** — real-time and delayed US equity quotes. Put the key in
  `finnhub.json` / `eodhd.json` (or `FLINT_FINNHUB_KEY` / `FLINT_EODHD_KEY`). Free tiers
  have no intraday history, so history comes from Yahoo.
- **Charles Schwab** — real-time quotes and price history over Schwab's OAuth API, market
  data only. Register an app at developer.schwab.com (callback `https://127.0.0.1`), put
  the App Key and Secret in `schwab.json`, then run `uv run flint schwab-auth` once to log
  in. It becomes the top-priority equity feed.
- **Alpha Vantage** — the key in `keys` is free-tier: intraday is premium and the cap is
  25 requests/day, so it is used for news sentiment and delayed quotes only.

All API keys are read from gitignored files and are masked everywhere in the UI, API and
logs.

## Signals and the investor council

Every few minutes flint refreshes a set of exogenous signals that feed the model and a
contrarian overlay: WallStreetBets attention and sentiment (via the ApeWisdom and
Tradestie aggregators, since Reddit's own API is closed), the crypto Fear & Greed index,
and Binance perp positioning (funding, open interest, long/short ratio).

The **investor council** tracks the disclosed positions of several investors from their
SEC 13F filings — Michael Burry (Scion), Warren Buffett (Berkshire), Bill Ackman
(Pershing Square), Steven Cohen (Point72), Carl Icahn, and Peter Schiff (Euro Pacific) —
reading puts as bearish. Each also carries an ethos profile encoding their documented
philosophy across a few dimensions (contrarian, value, momentum, quality, macro-bear,
activist, margin of safety). The combined council drives an overlay on every suggestion:
a margin-of-safety penalty on asymmetric downside, a fade-the-crowd rule that cuts or
flips trades aligned with euphoria, a counterpoint from the holdings consensus, and a
macro-bearish drag. Each adjustment is explained on the card and names the investor
driving it.

## Whole-market view

A scanner surveys the broader market: breadth and sector rotation from a basket of ETFs,
the VIX, crypto total market cap and BTC dominance, and the day's movers. It feeds two
market-wide features (breadth and a volatility signal) into the model and a whole-market
panel at the top of the dashboard. It also builds a 250-name **movers radar**, ranked by
how far they have moved and enriched with WallStreetBets mentions and which council
investors hold each name. The dashboard's asset cards reorder in real time by urgency,
and a market-watch grid below them shows everything the scanner can see.

## Control panel

The second tab has one console per stage (feed, bars, features, model, policy, learn,
news, signals, system), each streaming a line-by-line account of what came in and what
the model did with it. It also has the source and signal toggles, the investor council
with each member's ethos and holdings, the movers radar, and runtime controls for the
policy thresholds, learning rate, and the overlay. The same things are available over
HTTP: `GET /api/state`, `/api/news`, `/api/sources`, `/api/signals`, and
`POST /api/control`.

## Layout

```
flint/config.py    settings and FLINT_* overrides
flint/feed.py      shared Tick type and the offline simulator
flint/sources.py   market-data source registry + manager (Coinbase, Binance, Kraken,
                   Schwab, Finnhub, EODHD, Yahoo, CoinGecko, Alpha Vantage)
flint/schwab.py    Schwab OAuth token manager (market data only)
flint/bars.py      tick to bar aggregation, aligned across symbols
flint/features.py  feature engineering and running normalization
flint/model.py     FlintNet and the loss
flint/learner.py   replay buffer, optimizer loop, checkpoints
flint/signals.py   WSB, derivatives, Fear & Greed, and the investor council
flint/market.py    whole-market scanner and the movers radar
flint/news.py      news hub: headless-browser skimmer + Alpha Vantage sentiment
flint/engine.py    orchestration, suggestion policy + council overlay, metrics, controls
flint/server.py    FastAPI app, websocket, control endpoints
web/               dashboard (no build step, plain HTML/CSS/JS)
```
