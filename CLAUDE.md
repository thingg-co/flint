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

On the Spark (hostname `omega`, 192.168.1.128 since the 2026-10 reinstall), a systemd user timer
starts Flint each weekday at 19:45 local (08:45 ET, 45 minutes before the 09:30 ET market open) via
`deploy/spark/flint.service` and `flint.timer`, and does nothing when Flint is already listening on
:8000, so a running model is never restarted. The service runs `deploy/spark/flint-start`, which
`install.sh` copies to `~/.local/bin/flint-start`: it holds the Spark's `FLINT_*` environment
(`FLINT_HOST=0.0.0.0`, `FLINT_COMPILE=1`, the local llama-server brief backend) and launches
`.venv/bin/python -m flint` as a transient user unit `flint-run`, so the launcher is also the way
to start by hand: `flint-start`, then `journalctl --user -u flint-run -f` for the log, and
`systemctl --user stop flint-run` to stop; a stop returns in seconds and saves the checkpoint on
the way out. The `install.sh` script handles the one-time setup (it checks for `loginctl
enable-linger` and for the venv Python's dev headers, and prints the command for whichever is
missing). A start unloads any llama-server unit first, so a daytime cl session ends at 19:45
unless Flint is already up.

## Layout

- `flint/engine.py` — orchestrator: bar clock, forecasting, online training, the periodic
  loops (signals, market scan, news, checkpoints, market status), paper rebalancing,
  snapshot/publish, and the `KEY_SERVICES` registry.
- `flint/model.py` — FlintNet (dilated causal convs, cross-asset attention, regime-gated
  experts, non-crossing quantiles) and the loss.
- `flint/learner.py` — replay buffer and online optimizer; atomic, corruption-resilient
  checkpoints.
- `flint/backtest.py` — walk-forward backtest over the timestamped replay: a fresh net of the live
  preset predicts each day before training on it, `decide` applies the live policy's gates, and
  the realised returns score it. Run by the engine on weekends and by `flint backtest`.
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
  it appear as a skippable setup step with no frontend code. `set_key` writes the service's
  gitignored file, the one the config reads on the next start, and `_apply_key` pushes the key
  into everything that copied it at construction (the source; for Finnhub also the radar and the
  fundamentals signal), so a key saved in the panel works without a restart. `_key_values` must
  have an entry per service too, or the panel reports a saved key as "not set" forever (Alpaca,
  Tradier and E*TRADE did, 2026-10-10). A new consumer that copies a key at construction must be
  added to `_apply_key`; `tests/test_keys.py` walks KEY_SERVICES for the save, the panel status,
  the restart read-back through `Config()`, and the live apply against a real `SourceManager`.
- The `/ws` handler reads its socket as well as pushing to it. uvicorn's graceful shutdown closes
  every socket and then waits for each handler to return, cancelling nothing, so a push-only
  handler parked on its queue held SIGTERM until systemd's SIGKILL 90 s later and the final save
  never ran (the 2026-10-10 "stop hangs"). `timeout_graceful_shutdown=10` in `__main__` is a
  backstop, not the fix; `tests/test_server.py` drives the ASGI app directly to check the return.
- The universe grown at runtime (portfolio holdings, movers, news adds) persists in
  `state/universe.json` and is merged back on start, so a restart rebuilds at the same
  symbol count and the checkpoint still matches. Delete the file to intentionally shrink
  the universe back to the config default.
- While the market is closed, `_deep_backfill_loop` pages history back in ~10-day chunks
  (one every 150 s while there is history left, to `deep_backfill_days`, a live control; 0 disables), adds labeled windows to the replay
  through an isolated bar/feature pipeline, and trains on them. It must never run while the
  market is open. Ranged history comes from any source with `backfill_range`: FMP, and since
  0.5.10 Schwab, which serves 5-minute candles at least 250 days back by date range (probed
  2026-10-10), so no FMP key is needed. Each replay window is keyed by its bar timestamp, so the
  startup backfill after a restart, which walks the same ten days again, adds nothing (before
  0.5.10 every restart stacked another copy of recent days; a replay saved without timestamps
  is kept but not deduplicated). The replay is written uncompressed, at most every
  `replay_save_minutes` and at shutdown, because the save holds the training lock and a full
  buffer is tens of GB; the Spark launcher sizes it at 12,000 windows (~33 GB at 212 names,
  about seven months) with `deep_backfill_days` 200, which fills the buffer without wrapping it; a
  restart resumes paging from the oldest window held. `_idle_train_loop` fills the gaps between chunks: while closed it trains on
  the replay up to `idle_train_epochs` passes per closed session (a live control; 0 disables),
  a budget bounded by replay size rather than by hours, so a long night cannot overfit it. Every
  20th bar by timestamp (5%) is held out of training for good (`learner.validate`, derived from the
  timestamp so a reload keeps the same slice); the idle trainer checks it every 40 steps and ends
  the session's budget when it stops improving for `overfit_patience` checks (the overfit guard),
  with the value on the "held-out pinball" tile. The in-sample loss cannot tell fitting from
  overfitting: it rises whenever the deep backfill adds unseen history. With
  `train_in_session` on (a live control, off by default) it also runs through the open session with
  its own budget; a step can delay a forecast by at most one step. Both loops wait on the Finnhub
  market status (holiday-aware; unknown counts as neither open nor closed), so without a Finnhub key
  neither runs and the GPU idles all night. The status poll checks for a key on every pass, so one
  saved in the Control panel starts it without a restart; a rejected key is reported in the trace.
- On CUDA the train step runs under bf16 autocast with TF32 matmuls (loss in fp32); MPS and CPU
  stay fp32. `FLINT_COMPILE=1` wraps the training forward in `torch.compile` (2.1x on the GB10,
  same loss range; a two-minute compile on the first step after each start). Prediction and
  checkpoints use the uncompiled module, so the switch never changes what is saved. On that first
  step Triton builds a CUDA helper against `Python.h`, so the venv's Python needs its dev headers
  (`python3.12-dev`; the 2026-10 Spark reinstall dropped it and `install.sh` warns when it is
  missing). If the first compiled step fails for any reason the learner logs the cause once and
  trains eager for the rest of the run; the engine never dies over it. The autotune cache key includes the precision mode. Step time grows about linearly
  with batch size on a bandwidth-bound GPU, so a bigger batch only buys a smaller preset. The
  preset is chosen by step time, never by memory: the binding budget is `max_warmup_seconds` over
  `warmup_steps`, and both are in the tune cache key, so raising the ceiling re-benchmarks on the
  next start. The Spark launcher sets 300 s so the ladder reaches 5XL, the top rung, about 57 GB
  with the replay at 212 names; that cannot share the box with an 84 GB llama-server.
- The autotune records each rung's peak memory and refuses to start (`NotEnoughMemory`, exit 2)
  when the box cannot hold the tuned model plus the replay buffer, naming any `llama-server@*`
  unit sharing the memory. A ladder cut short by memory is never cached, so a transient squeeze
  cannot downgrade the model for good. On unified memory the kernel's MemAvailable is the truth;
  CUDA's own free figure counts page cache as used. Exists because the 2026-09-04 5XL benchmark
  next to an 84 GB llama-server hung the Spark.
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
- Risk gates in the policy (all live controls; the model is untouched): a name needs `skill_min_n`
  matured direction calls at or above `min_hit_rate` before it may trade; the same side must persist
  `confirm_bars` bars; a stock position is held `stock_min_hold_bars` unless the signal reverses; the
  live bid/ask spread replaces `cost_bps` when larger and `max_spread_bps` / `min_price` block
  illiquid and penny names; size scales with band coverage / 0.8; option premium per position is
  capped at `option_max_frac` of equity. These exist because of the 2026-09-02 session (churn and a
  $1 stock cost 7% in a night).
- The Portfolio tab and account access are strictly read-only; Flint never places, changes, or cancels an order on any provider.
- The backtest (`_backtest_loop`, Backtest tab, `/api/backtest`, `flint backtest`) never touches the
  live model. It runs in process, once per weekend (Saturday or Sunday ET, market closed) or on the
  `backtest` control action, only after the deep backfill has finished paging (so the ring cannot move
  under it), and the idle trainer yields the GPU while it runs. It is out of sample by construction:
  the fresh net forecasts day D before it trains on day D, signals wait for the live trust rule, and
  every calibration update lands one horizon late, as live. `backtest.decide` mirrors the gates of
  `_suggest` that history can answer; change them together. Each simulated day trains the live
  online budget plus `backtest_night_epochs` passes over the history so far (the idle trainer's
  regimen); `backtest_hours` (30 by default, a weekend) caps the run by shrinking that budget, so a
  run always finishes; the report says what it used. A universe rebuild stops a running backtest
  first and waits for it, because a benchmark next to another net reads nonsense peaks.
- A failed universe rebuild (no memory to benchmark, a bad symbol) puts the previous universe back
  and resumes the lifecycle on the model it still has; it saves first so the replay in memory is
  not lost. Before 2026-10-10 it left every loop cancelled and the status "starting" for good. The
  benchmark's peak is measured above what the process already holds, so a live model or a backtest
  in the same process cannot inflate it. Note that growing the universe still resets the model and
  the replay (shape mismatch), so "+ Top 20 movers" and holdings adds cost the deep-backfill history.
- News: the browser skimmer's feeds follow the universe's asset mix (`default_sources`): market desks
  for equities, the crypto desks only when crypto is modeled. Equity mentions come mainly from
  `FinnhubNews`, company news keyed by ticker (30 names per pass in rotation, three days kept), with
  general market news for the tone; headline text matches a ticker only as an upper-case word of
  three or more letters outside `TICKER_STOPLIST`. The Control panel lists only covered names.

## Working on it

- Inspect live state over the JSON API: `curl -s localhost:8000/api/state` (phase, metrics,
  bars, paper), plus `/api/sources`, `/api/signals`, `/api/keys`.
- Judge memory on the Spark from `free -g` system used, never from the process RSS or the unit's
  `MemoryCurrent`: the CUDA allocations live outside both, so a live 4XL run reads about 3 GB by
  RSS and 30+ GB by system used, and a dead engine (phase `error`, steps not moving) looks the
  same by RSS.
- Frontend edits are static — hard-reload the tab, no restart. Backend edits need a restart.
- `uv run flint check` prints what a start would do without loading a model or touching the port:
  device, free memory, any llama-server unit on the same memory, the cached preset and its peak,
  and whether the model plus replay fits. Exit 2 means a start would refuse. Run it with the same
  `FLINT_*` environment the launcher uses, or the replay size and universe will be the defaults.
- The Backtest tab shows the last walk-forward report and live progress; "Run now" queues one. The
  loop logs a one-line summary to the trace when it finishes; `state/backtest.json` holds the report.
- Tests: `uv run pytest -q tests`. They cover the memory guard, the paper book's defined-risk
  rules, the policy gates in `_suggest`, checkpoint mismatch handling, bar building, features and
  source failover, key saving and live application, the eager fallback when a compiled step
  fails, and the websocket handler returning on disconnect, all without loading a model or
  opening a feed. Add a test when you touch one of those.
- Python ≥3.11, managed with uv. Brokerage logins: `uv run flint schwab-auth`,
  `uv run flint etrade-auth`.
