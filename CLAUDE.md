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
- `flint/paper.py` — the paper-trading book: spread-aware, persisted inside the checkpoint.
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
- The narrative brief runs on local Ollama. Never route it to a cloud API.
- User-facing text capitalizes "Flint" as a proper noun.

## Working on it

- Inspect live state over the JSON API: `curl -s localhost:8000/api/state` (phase, metrics,
  bars, paper), plus `/api/sources`, `/api/signals`, `/api/keys`.
- Frontend edits are static — hard-reload the tab, no restart. Backend edits need a restart.
- Python ≥3.11, managed with uv. Brokerage logins: `uv run flint schwab-auth`,
  `uv run flint etrade-auth`.
