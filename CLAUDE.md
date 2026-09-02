# Flint

Flint is a self-hosted, continuously-learning market model. A PyTorch net (FlintNet)
forecasts each stock's return distribution over the next hour on 5-minute bars, trains
online as its own forecasts resolve, and serves everything to a no-build browser
dashboard over FastAPI and a websocket. Everything runs on the local machine. It is a
research toy, not advice.

## Running and restarting

`uv run flint` starts the server on 127.0.0.1:8000. It refuses to start if that port is
already in use — a second instance would run the GPU autotune (thrashing the first)
before failing to bind.

To restart during development, kill the old process first:

```
pkill -9 -f "python -m flint"; pkill -9 -f "caffeinate -i .venv/bin/python -m flint"
```

Wait for :8000 to free, then relaunch. One-off benchmark scripts run as
`.venv/bin/python -`, which those patterns won't match — kill those by PID.

Run only one GPU benchmark at a time. Two autotune or benchmark processes on MPS thrash
memory into swap (this once reached 26 GB and looked like a leak; it was stray processes).
The autotune holds a file lock and the server owns the port, so don't run benchmark
scripts next to a live server.

Every restart re-runs warmup and resets progress toward "trusted" (48 live, out-of-sample
labels), and live labels only accrue while the market is open. Avoid restarting unless a
change actually requires it.

## Layout

- `flint/engine.py` — orchestrator: bar clock, forecasting, online training, the periodic
  loops (signals, market scan, news, checkpoints, market status), paper rebalancing,
  snapshot/publish, and the `KEY_SERVICES` registry.
- `flint/model.py` — FlintNet (dilated causal convs, cross-asset attention, regime-gated
  experts, non-crossing quantiles) and the loss.
- `flint/learner.py` — replay buffer and online optimizer; atomic, corruption-resilient
  checkpoints.
- `flint/autotune.py` — picks the model preset by benchmarking against a warmup-time
  budget; caches the choice in `state/machine.json`.
- `flint/features.py`, `flint/bars.py` — per-bar features with running normalization, and
  tick → 5-minute-bar aggregation.
- `flint/sources.py` — the data-source `REGISTRY` and `SourceManager` (priority-ordered
  failover per symbol).
- `flint/market.py`, `flint/signals.py` — whole-market scan (movers/sectors/breadth/VIX)
  and the exogenous signals (WSB, gurus, 13F, fear & greed).
- `flint/paper.py` — the paper-trading book: spread-aware, shorts as long puts (bounded loss), persisted in the checkpoint.
- `flint/options.py` — Black-Scholes marking + Schwab option-chain selection (put for a short; ATM IV / skew as model features).
- `flint/portfolio.py` — read-only Schwab account positions for the Portfolio tab (GET /trader/v1/accounts only, never orders).
- `flint/schwab.py`, `flint/etrade.py` — brokerage OAuth.
- `flint/server.py` — FastAPI routes and the `/ws` websocket.
- `flint/config.py` — every knob, each reading a `FLINT_*` env var or a json file, with a
  default.
- `web/` — the dashboard: plain HTML/CSS/JS, no build step, served static.
- `docs/` — the public product page (GitHub Pages).

## Invariants worth keeping

- The model runs on 5-minute bars. Backfill has to be deep enough that trading bars exceed
  window + horizon, or no windows form and it never forecasts.
- The checkpoint reloads only when the replay shape and the symbol list match the current
  config. On a mismatch it renames the old checkpoint to `model.pt.bak` and starts fresh,
  so a config change never silently destroys a trained model. Changing the symbol list,
  window, or feature set resets the model.
- Every modeled symbol needs a price for a bar row to emit, so a symbol with no history
  would block the whole universe. History-less symbols are flat-seeded at 0.0 (a neutral,
  zero-return series). Keep that — one dead ticker must not zero every bar.
- Candles build only from real trades (`Tick.quote is False`). Quote/heartbeat ticks update
  price and bid/ask but never build bars; otherwise a closed-market feed echoing a stale
  quote fabricates flat bars.
- Sources live in `REGISTRY` (sources.py); their keys live in `KEY_SERVICES` (engine.py).
  `KEY_SERVICES` drives onboarding and the Control panel, so adding a provider there makes
  it appear as a skippable setup step with no frontend code.
- The universe grown at runtime (portfolio holdings, movers, news adds) persists in
  `state/universe.json` and is merged back on start, so a restart rebuilds at the same
  symbol count and the checkpoint still matches. Delete the file to intentionally shrink
  the universe back to the config default.
- While the market is closed, `_deep_backfill_loop` pages history back in ~10-day chunks
  (one every 150 s while there is history left, to `deep_backfill_days`, a live control; 0 disables), adds labeled windows to the replay
  through an isolated bar/feature pipeline, and trains on them. It must never run while the
  market is open. `_idle_train_loop` fills the gaps between chunks: while closed it trains on
  the replay up to `idle_train_epochs` passes per closed session (a live control; 0 disables),
  a budget bounded by replay size rather than by hours, so a long night cannot overfit it. With
  `train_in_session` on (a live control, off by default) it also runs through the open session with
  its own budget; a step can delay a forecast by at most one step.
- On CUDA the train step runs under bf16 autocast with TF32 matmuls (loss in fp32); MPS and CPU
  stay fp32. `FLINT_COMPILE=1` wraps the training forward in `torch.compile` (2.1x on the GB10,
  same loss range; a two-minute compile on the first step after each start). Prediction and
  checkpoints use the uncompiled module, so the switch never changes what is saved. The autotune cache key includes the precision mode. Step time grows about linearly
  with batch size on a bandwidth-bound GPU, so a bigger batch only buys a smaller preset.
- The market radar keeps each name's market cap from the Yahoo screeners and fills a sector per
  symbol from Finnhub's company profile (industry folded into GICS-style sectors by `SECTOR_OF`),
  40 names per scan under the 60/min limit, persisted in `state/sectors.json`. The dashboard treemap
  groups by sector and sizes by market cap once those fields are present; until then it is one map
  sized by dollar volume.
- The narrative brief runs on a **local** LLM backend — Ollama (default) or any OpenAI-compatible
  server such as llama-server or vLLM (set `brief_backend=openai` + `brief_openai_base`). Never route it
  to a cloud API. On the Spark, start.sh points it at llama-server on 127.0.0.1:8080.
- User-facing text capitalizes "Flint" as a proper noun.
- FlintNet returns five tensors (quantiles, up-logit, down-logit, gate, attention). If you change the head, update `flint_loss` AND `autotune._bench` together -- an arity mismatch only surfaces on a re-benchmark (a feature/symbol-count change), not on a cached start.
- Changing the feature set or the symbol list reshapes the model, so it resets once (guarded; the old checkpoint is backed up to model.pt.bak).
- No position may carry unlimited risk, ever. Defined risk is the rule, not premium-capped: long
  options, debit and credit spreads, covered calls on held stock and cash-secured puts are all fine,
  because the worst case is a known finite number at entry. Naked short stock and naked written
  options are not. Today a bearish view is a long put and a volatility view is a long straddle; paper
  shorts are long puts -- never naked shorts; longs are stock. Paper only trades once the model is `trusted`. With `extended_hours` on (a live control), long stock may enter and exit in Schwab's 4:00-9:30 / 16:00-20:00 ET sessions; puts and straddles stay regular-hours because options do not trade then.
- The Portfolio tab and account access are strictly read-only; Flint never places, changes, or cancels an order on any provider.

## Working on it

- Inspect live state over the JSON API: `curl -s localhost:8000/api/state` (phase, metrics,
  bars, paper), plus `/api/sources`, `/api/signals`, `/api/keys`.
- Frontend edits are static — hard-reload the tab, no restart. Backend edits need a restart.
- Python ≥3.11, managed with uv. Brokerage logins: `uv run flint schwab-auth`,
  `uv run flint etrade-auth`.
