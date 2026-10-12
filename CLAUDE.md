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
the way out. The launcher uses the unit whenever the user manager reports `running` or `degraded` (degraded is a running manager with some unrelated failed unit; before 0.5.15 it sent the start to the `nohup` fallback, a bare process in `state/flint.log` that no `systemctl stop` can reach). The `install.sh` script handles the one-time setup (it checks for `loginctl
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
- The checkpoint reloads when the replay shape and the symbol list match the current config, and
  migrates when only the symbol list changed (see the universe bullets). A change of window or
  feature set, or a universe with no name in common, renames the old checkpoint to `model.pt.bak`
  and the replay to `replay.bak` and starts fresh, so a config change never silently destroys a
  trained model.
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
- The universe is the configured core, every holding (the portfolio loop adds a new holding
  within a minute) and up to `interesting_count` promoted names, under `max_universe`. Once a day
  after the close `_universe_loop` scores candidates from three organic sources on one scale
  (`universe_candidates`: radar movers by move x dollar volume, tickers named in headlines that are
  known symbols, and the tracked funds' 13F positions) and `pick_universe` promotes the top of the
  ranking into the room left (never on a thin pool: the radar, the news skim and the 13F books must
  have reported and at least 20 candidates scored, else it waits; "Refresh universe now" on the
  Control panel queues a pass); a promoted name is demoted only after `interesting_grace_days`
  without ranking in the top 1.5 x count and `interesting_min_tenure_days` of tenure, because a
  demotion forfeits its learned row and history. `state/universe.json` persists the list, the
  refresh date and each promoted name's since/last_hit; delete it to shrink back to the config.
- A symbol-list change no longer resets the model: `learner.load` migrates the checkpoint (every
  parameter is shared across assets except the per-asset embedding table and its Adam moments,
  whose rows are copied for retained names) and the replay (retained columns copied, new names
  zero and masked), then the deep backfill re-pages the whole history and `learner.fill` writes the
  new names' columns into windows already held. Only a change of window or features, or a universe
  with no name in common, still backs up and starts fresh. The replay lives in `state/replay/` as
  one .npy per array plus meta.json, swapped in atomically and memory-mapped on load (`read_replay`
  also reads the pre-0.5.12 replay.npz); a rebuild frees the old buffer before the new learner
  allocates its own. The feature normalizer's rows travel by name too (`RunningNorm.state(symbols)` /
  `load(st, symbols)`, `engine._load_norm`): retained names keep their statistics and count, new names
  start fresh with their own count, a state saved without names is named by the checkpoint's own
  symbol list, and a state that cannot be mapped leaves the normalizer fresh with a trace warning
  rather than failing the start (found in Steel 2026-10-11: the positional copy died on a broadcast
  error after the first promotion, phase `error` until a restart with a matching universe).
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
  buffer is tens of GB. The Spark launcher budgets it in bytes (`replay_gb` 36: the window count
  follows the universe, 12,000 windows at 232 names and 7,000 at 400; `replay_size` applies only
  when the budget is unset) with `deep_backfill_days` 200; deep paging stops when the ring is
  within a chunk of full, since wrapping would overwrite the newest windows, and a smaller buffer
  on load keeps the newest windows. A restart resumes paging from the oldest window held. `_idle_train_loop` fills the gaps between chunks: while closed it trains on
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
  with batch size on a bandwidth-bound GPU, so a bigger batch only buys a smaller preset. The Spark
  launcher allows 400 names; at that count the ladder lands on 4XL (step time scales with the
  universe) and the memory guard needs the 36 GB replay budget to pass. The
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
- Changing the feature set reshapes the model, so it resets once (guarded; the old checkpoint is backed
  up to model.pt.bak); changing the symbol list migrates it. Since 0.5.18 (the reset of 2026-10-12)
  every model input is computable from the bar stream plus dated news, so paged history and the live
  feed produce the same numbers: per name from its own bars (return, range, volume, close position,
  the hour's momentum and vol, Garman-Klass and Yang-Zhang vol, day and week realised vol, the day's
  return, the same-slot return a day back and its five-day mean, Amihud, Roll, the overnight and
  intraday legs, the session clock), across names (breadth this bar and over the hour, the market's
  return, momentum and vol with the first index name as the market, each name's rolling beta, the
  market's first half hour) and the two news slots. The live-only signals (WSB, Fear & Greed,
  funding, 13F, ethos, fundamentals, IV, skew) and the three bar features history candles lack
  (bid/ask imbalance, quoted spread, trade count) are not inputs; `set_exo` keeps them for the policy
  and the dashboard. There is no day-of-week or date feature on purpose. The news slots for history
  come from `NewsHistory` (`flint/newshist.py`, `state/news_history.json`): Finnhub company news by
  date over the backfill span, six headlines per name per day, scored like the live skim, read at
  each bar as the skim would have reported it. `_news_history_job` fills it at every start, paced
  under Finnhub's limit; the deep loop waits for a chunk's news before paging it, walks a nine-day
  warm prefix (`WARM_S`) it does not emit so the week-long features are exact from a chunk's first
  window, and rewrites the startup backfill's days once their news is in (`learner.replace`).
- The feature set is a switch (`FLINT_FEATURE_SET`, `core` by default, read by `flint/features.py` at
  import because the net's shape follows it; `Config.feature_set` only documents it). `core` is the 28
  inputs above. `extended` (0.5.19, coded ahead of the reset that will judge it) adds four opinions
  computed from the week's ring alone (the beta-adjusted residual path's z-score and AR(1) half-life,
  the trend's t statistic, a GARCH(1,1) one-step vol at fixed persistence; `FeatureBuilder._opinions`,
  zero under 20 bars of ring, so a chunk's warm prefix makes them exact) and two earnings flags from a
  dated calendar (`flint/events.py`, `state/events.json`: Nasdaq's public calendar, one request per
  weekday over the history span plus 45 days ahead at every start, refreshed every six hours; the flags
  decay as exp(-days/3) to and from a name's report dates, both 1 on the day itself because Nasdaq
  rarely gives the hour). The deep loop waits for the calendar as it waits for the news
  (`_history_ready`), the live builder and the offline one read the same store, and a builder without
  one reads zeros. Flipping the switch in the Spark launcher reshapes the net, so it is a weekend
  change: the autotune re-benchmarks (the feature count is in its key) and the checkpoint guard backs
  the model up. Finnhub's free earnings calendar reaches only a few weeks back and its dividend
  endpoints are premium (probed 2026-10-12), which is why the calendar is Nasdaq's and there is no
  ex-dividend flag yet.
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
  $1 stock cost 7% in a night). Since 0.5.14 the edge also pays the square-root price impact of the
  order the signal would place (`impact_eta`, from the bars' dollar volume and volatility) and, for
  a bearish view bought as a put, the put's hourly decay (`theta_in_edge`, from the chain's ATM IV
  kept raw in `option_iv`); Amihud's illiquidity blocks a name whose order would move the price more
  than `max_impact_bps`, which works where no quote exists; a straddle needs the model's expected
  |move| (folded-normal mean of the quantiles) above `straddle_edge` times the implied one-sigma
  move plus the decay over the hold, falling back to the width rule when no IV is known. Stock
  positions exit on `stop_loss_bps` (200 by default), `take_profit_bps` and `max_hold_bars` (both
  off), overriding the minimum hold, then wait `exit_cooldown_bars`; once the book is down
  `daily_loss_limit` of the day's opening equity no new entry opens that day. All of it is
  `flint/policy.py` (pure functions) and `tests/test_policy.py`; the backtest's `decide` cannot
  mirror the impact, decay and illiquidity terms because the replay holds standardized features,
  not raw volume or IV, so the backtest's flat cost stays the conservative floor. Since 0.5.18 a position below `min_position_dollars` (1,000, a live
  control) is lifted to that floor when its edge still covers the costs at that size and held otherwise, and
  a put's whole-contract commission is amortised over the hold in the edge test (`ticket_bps` on the signal):
  costs were all proportional before, so Kelly over hundreds of names sized most positions under the paper
  book's own rebalance floor and they never filled; the paper book fills whole contracts, never past the
  premium cap, with the same floor, and the backtest, which works in weights, cannot mirror any of it. An
  index hedge leg pays `hedge_cost_bps` per unit
  (1 bps; SPY and the other index ETFs quote a cent or two wide) in the backtest, `replay_policy` and the
  sweep, recorded in the stored forecasts; at the names' 5 bps the hedge cost 1.5x the basket's cost and was
  28 of the 2026-10-11 relative run's 33 bps loss.
- The meta-labeler (`flint/meta.py`, item 5) is a second opinion on each forecast: an online logistic model over
  six numbers the forecast itself gives, taught by matured forecasts whether the move in the called direction
  beat `meta_cost_bps`, one horizon late as everything else in the calibration. It gates in four places that
  must change together: `Engine._suggest` (live, `meta_min`, a live control, silent until the model has
  learned from 2,000 forecasts), `backtest.decide` and the scoring loop (which store each forecast's odds in
  the forecasts file) and `sweep.replay_policy` (which reads them, so the sweep tunes `meta_min`). Features
  are per forecast, not per name, so the model outlives a universe rebuild (it lives in `Engine.__init__`,
  not `_build`) and its state rides in the checkpoint's `extra["meta"]`, kept through a migration like the
  book and the normalizer. The band-implied P(move clears the cost) is not a probability (item 6) and sizing
  by any such number is only exposure: the meta-labeler is a gate.
- The Portfolio tab and account access are strictly read-only; Flint never places, changes, or cancels an order on any provider.
- The target is a run-time switch (`FLINT_TARGET=absolute|relative`, `flint/target.py`). The replay
  always stores absolute labels (the hour-ahead log return in bps); with `relative` every consumer
  of labels (the training step, the held-out validation, the live scoring in `_mature`, the
  backtest) sees the residual against the modeled index names instead (`index_symbols`, default
  SPY,QQQ,IWM,SMH,XLF; unmodeled ones are ignored, none modeled means absolute). Loadings are a
  ridge regression per name on the last `beta_windows` label rows with `beta_shrink`
  pseudo-observations behind the prior (1 on the first index, 0 on the rest), refit once a session
  in `train_steps` and after a load; the backtest refits from the days already seen. Index names
  keep absolute labels (zero rows). The paper book then holds, per index, the weight that cancels
  the basket's loading (`hedge_targets`: an index put under a long basket, index stock over a
  bearish one), passed to `rebalance` as `uncapped` so the hedge skips the per-position weight cap,
  the leverage scale and the premium cap; below `hedge_min` it is not placed. Switching the target
  does not reset the model: the checkpoint records the target it was trained on and `load` warns
  when it differs. `backtest_target` (default `both`) runs the variants back to back on halves of
  the budget and the second report carries the first as `baseline`, which the Backtest tab plots
  as the amber curve. `tests/test_target.py` covers the fit, the transform, the hedge, both
  learner and backtest modes and the uncapped hedge weight.
- The backtest (`_backtest_loop`, Backtest tab, `/api/backtest`, `flint backtest`) never touches the
  live model. It runs in process, once per weekend (Saturday or Sunday ET, market closed) or on the
  `backtest` control action, only after the deep backfill has finished paging (so the ring cannot move
  under it; `_paging_pending` counts a ring within a chunk of full as finished, because the deep loop
  stops there and at 362 names the 30 GB budget holds 6,474 windows of the 128-bar preset, about 83
  days, well short of `deep_backfill_days`), and the idle trainer yields the GPU while it runs. It is out of sample by construction:
  the fresh net forecasts day D before it trains on day D, signals wait for the live trust rule, and
  every calibration update lands one horizon late, as live. `backtest.decide` mirrors the gates of
  `_suggest` that history can answer; change them together. Each simulated day trains the live
  online budget plus `backtest_night_epochs` passes over the history so far (the idle trainer's
  regimen); `backtest_hours` (30 by default, a weekend) caps the run by shrinking that budget, so a
  run always finishes; the report says what it used. A universe rebuild stops a running backtest
  first and waits for it, because a benchmark next to another net reads nonsense peaks.
- A backtest owns the GPU. While one runs, nothing else trains: matured labels are stored and
  trained on afterwards, the idle trainer skips, and the deep backfill pauses so the replay cannot
  grow under the run; a manual "Run now" also waits for paging to finish. Before it starts, the
  backtest sizes its net to the memory actually free (`backtest_preset_for`: the widest preset no
  wider than the live one whose peak, scaled from the live rung's measured peak by d_model, fits
  above `memory_floor_gb`), it stops itself at the next day boundary if free memory falls below the
  floor, and deep paging pauses near the floor too. Two 5XL nets training in one 121 GB box took
  the Spark down on 2026-10-10 (NVRM out of memory, reboot); the GPU allocations are outside any
  cgroup limit, so these in-process rules are the only guard. The caching allocator is the other
  way to lose the box: it keeps blocks per shape, so the training step, the held-out validation pass
  and per-bar prediction each held their own working set (about 90 GB outside any process at 232
  names). The idle trainer and the deep loop release the cache every 40 steps, validation runs at
  batch 8 and releases after, the idle trainer pauses near the floor like deep paging, and the
  launcher sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- A failed universe rebuild (no memory to benchmark, a bad symbol) puts the previous universe back
  and resumes the lifecycle on the model it still has; it saves first so the replay in memory is
  not lost. Before 2026-10-10 it left every loop cancelled and the status "starting" for good. The
  benchmark's peak is measured above what the process already holds, so a live model or a backtest
  in the same process cannot inflate it. Since 0.5.12 a universe change migrates the model and the
  replay instead of resetting them, so holdings adds and the daily refresh cost nothing but the
  re-page that fills the new names in.
- News: the browser skimmer's feeds follow the universe's asset mix (`default_sources`): market desks
  for equities, the crypto desks only when crypto is modeled. Equity mentions come mainly from
  `FinnhubNews`, company news keyed by ticker (30 names per pass in rotation, three days kept), with
  general market news for the tone; headline text matches a ticker only as an upper-case word of
  three or more letters outside `TICKER_STOPLIST`. The Control panel lists only covered names.
  Headline tone is graded (`flint/sentiment.py`): `news_scorer=finbert` (the default) runs
  ProsusAI/finbert locally on the CPU, tone = P(positive) − P(negative), fetched from the Hugging
  Face hub on first use (about 440 MB, cached under ~/.cache/huggingface) and cached per title so a
  pass costs inference only for new headlines (about 20 ms each on the Spark); `lexicon` is the
  graded word lists (one tone word ±0.5, two ±0.67, a hedge takes a third off), also the fallback
  when the model cannot load. Before 0.5.14 the word count gave every headline −1, 0 or +1. Per-name
  attention is relative to the pass's most-covered name (`attention_of`: the square root of the
  share, 0 for no coverage) and a name's tone is its own headlines with the market tone as one
  headline of prior (`name_tone`); before, half of the same general headlines entered every name's
  weight, so attention read 0.76 for everyone and tone was the market's. Both feed `news_sent` and
  `news_attn`; the feature count is unchanged, so no reset.

## Working on it

- Inspect live state over the JSON API: `curl -s localhost:8000/api/state` (phase, metrics,
  bars, paper), plus `/api/sources`, `/api/signals`, `/api/keys`.
- Judge memory on the Spark from `free -g` system used, never from the process RSS or the unit's
  `MemoryCurrent`: the CUDA allocations live outside both, so a live 4XL run reads about 3 GB by
  RSS and 30+ GB by system used, and a dead engine (phase `error`, steps not moving) looks the
  same by RSS.
- Frontend edits are static — reload the tab, no restart. Backend edits need a restart. The page and
  `/static` are served `Cache-Control: no-cache` and the page links its script and stylesheet by the
  files' mtimes (`index_html`), so a normal reload always gets the current files; before 0.5.15 a
  browser could keep app.js for hours under the heuristic rule and only a hard reload replaced it.
- `uv run flint check` prints what a start would do without loading a model or touching the port:
  device, free memory, any llama-server unit on the same memory, the cached preset and its peak,
  and whether the model plus replay fits. Exit 2 means a start would refuse. Run it with the same
  `FLINT_*` environment the launcher uses, or the replay size and universe will be the defaults.
- The Backtest tab shows the last walk-forward report and live progress; "Run now" queues one. The
  loop logs a one-line summary to the trace when it finishes; `state/backtest.json` holds the report.
  Each run also stores every raw forecast it made (`state/backtest_forecasts_<target>.npz`) and then
  runs the policy sweep (`flint/sweep.py`, `flint sweep` by hand): `replay_policy` is the backtest's
  scoring loop vectorized over names and reproduces the run's net to the bps (`tests/test_sweep.py`
  checks this on both targets, so change `backtest.decide`, the scoring loop and `replay_policy`
  together), and `run_sweep` is a coordinate descent over seven gates on the first half of the
  scored days, confirmed on the second. The result is a proposal on the Backtest tab and in
  `state/sweep.json`; nothing applies it, the operator changes the live controls or not.
- Tests: `uv run pytest -q tests`. They cover the memory guard, the paper book's defined-risk
  rules, the policy gates in `_suggest`, checkpoint mismatch handling, bar building, features and
  source failover, key saving and live application, the eager fallback when a compiled step
  fails, and the websocket handler returning on disconnect, all without loading a model or
  opening a feed. Add a test when you touch one of those.
- Python ≥3.11, managed with uv. Brokerage logins: `uv run flint schwab-auth`,
  `uv run flint etrade-auth`.
