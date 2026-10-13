"""Walk-forward backtest: out of sample by construction, the live policy's gates, the
weekend schedule. CPU only, a toy replay."""
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from flint.backtest import decide, run_backtest
from flint.config import Config
from flint.engine import Engine

NY = ZoneInfo("America/New_York")


def toy_cfg():
    cfg = Config()
    cfg.symbols = ["A", "B"]
    cfg.device = "cpu"
    cfg.torch_threads = 1
    cfg.d_model = 16
    cfg.dilations = (1, 2)
    cfg.n_experts = 2
    cfg.n_heads = 2
    cfg.window = 8
    cfg.horizon = 2
    cfg.batch_size = 4
    cfg.compile = False
    cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    cfg.score_threshold = 0.1
    cfg.prob_margin = 0.05
    cfg.move_floor_bps = 5.0
    cfg.skill_min_n = 0
    cfg.confirm_bars = 1
    cfg.kelly_fraction = 1.0
    cfg.max_size = 1.0
    cfg.size_by_coverage = False
    cfg.min_hit_rate = 0.0
    return cfg


def toy_replay(n_days=8, bars=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * bars
    rx = rng.random((n, 2, 8, 3), dtype=np.float32)
    ry = (rng.standard_normal((n, 2)) * 30).astype(np.float32)
    rmask = np.ones((n, 2), dtype=np.float32)
    rt = np.zeros(n)
    t0 = datetime(2026, 9, 1, 9, 35, tzinfo=NY)
    for d in range(n_days):
        for b in range(bars):
            rt[d * bars + b] = (t0 + timedelta(days=d, minutes=5 * b)).timestamp()
    return rx, ry, rmask, rt


def test_predicts_each_day_before_training_on_it():
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay()
    order = []
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=["A", "B"], warm_days=3, steps_per_day=2,
                       hooks={"on_predict": lambda d: order.append(("predict", d)), "on_train": lambda d: order.append(("train", d))})
    assert res["totals"]["days"] == 5 and len(res["days"]) == 5
    scored = [d for k, d in order if k == "predict"]
    assert scored == [3, 4, 5, 6, 7]
    for d in scored:                                          # the forecast of day d precedes its training step
        assert order.index(("predict", d)) < order.index(("train", d))
    assert all(r["steps"] == 2 and r["pinball"] is not None for r in res["days"])
    assert res["config"]["history_days"] == 8 and res["config"]["windows"] == 48
    assert set(res["symbols"]) == {"A", "B"}


def test_untimed_windows_are_ignored_and_too_little_history_is_an_error():
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay(n_days=4)
    rt[:12] = 0.0                                              # legacy rows carry no date
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", warm_days=3, steps_per_day=1)
    assert "error" in res and res["days"] == []


def test_stop_returns_a_partial_report():
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay()
    calls = []
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", warm_days=3, steps_per_day=1,
                       should_stop=lambda: len(calls) >= 5 or calls.append(1))
    assert res["stopped"] is True and 0 < res["totals"]["days"] < 5


def test_decide_mirrors_the_live_gates():
    cfg = toy_cfg()
    q = np.array([-10.0, 10.0, 30.0, 50.0, 70.0])              # q50 +30 bps, IQR 40 -> score 0.75
    side, size, qc, p, streak = decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, None, None, 5.0)
    assert side == 1 and abs(size - 0.75) < 1e-9 and qc[2] == 30.0
    assert decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, None, None, 40.0)[0] == 0          # cost eats the edge
    assert decide(cfg, q, 0.3, 0.6, 1.0, 1.0, None, None, None, 5.0)[0] == 0           # trend and direction disagree
    assert decide(cfg, -q[::-1], 0.2, 0.8, 1.0, 1.0, None, None, None, 5.0)[0] == -1   # a short
    cfg.skill_min_n = 8
    cfg.min_hit_rate = 0.5
    assert decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, [False] * 8, None, 5.0)[0] == 0    # a bad record blocks
    assert decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, [True] * 8, None, 5.0)[0] == 1
    cfg.confirm_bars = 2
    s1 = decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, [True] * 8, None, 5.0)
    assert s1[0] == 0 and s1[4] == (1, 1)                                             # first bar: wait
    assert decide(cfg, q, 0.8, 0.1, 1.0, 1.0, None, [True] * 8, s1[4], 5.0)[0] == 1    # second bar: go
    cfg.size_by_coverage = True
    assert decide(cfg, q, 0.8, 0.1, 1.0, 1.0, 0.4, [True] * 8, (1, 2), 5.0)[1] == 0.75 * 0.5   # 0.4/0.8 coverage halves it


def test_weekend_schedule():
    fake = SimpleNamespace(_backtest_requested=False, cfg=SimpleNamespace(backtest_weekends=True),
                           market_status={"isOpen": False}, backtest={})
    due = lambda **k: Engine._backtest_due(fake, **k)
    sat = datetime(2026, 10, 10, 12, 0, tzinfo=NY)
    assert due(now=sat)
    assert not due(now=datetime(2026, 10, 9, 12, 0, tzinfo=NY))                       # a Friday
    fake.backtest = {"ran_at": sat.timestamp() - 3600}
    assert not due(now=sat)                                                           # already ran this weekend
    fake.backtest = {"ran_at": sat.timestamp() - 3600, "stopped": True, "totals": {"days": 0}}
    assert due(now=sat)                                                               # a run that scored nothing does not count
    fake.backtest = {"ran_at": sat.timestamp() - 3600, "stopped": True, "totals": {"days": 12}}
    assert not due(now=sat)                                                           # a partial run does
    fake.backtest = {"ran_at": sat.timestamp() - 3600}
    assert due(now=datetime(2026, 10, 17, 12, 0, tzinfo=NY))                          # next Saturday
    fake.market_status = {"isOpen": None}
    assert not due(now=datetime(2026, 10, 17, 12, 0, tzinfo=NY))                      # unknown status: wait
    fake._backtest_requested = True
    assert due(now=datetime(2026, 10, 9, 12, 0, tzinfo=NY))                           # a request runs any day
    fake._backtest_requested = False
    fake.cfg.backtest_weekends = False
    fake.market_status = {"isOpen": False}
    assert not due(now=datetime(2026, 10, 24, 12, 0, tzinfo=NY))


def test_night_epochs_add_training_over_the_history_seen():
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay(n_days=6, bars=6)
    plain = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", warm_days=2)
    night = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", warm_days=2, night_epochs=1.0)
    # day k (0-based) has seen 6*(k+1) windows; one pass over them at batch 4 is 6*(k+1)/4 extra steps
    for p, q, k in zip(plain["days"], night["days"], range(2, 6)):
        assert p["steps"] == cfg.steps_per_label * 6
        assert q["steps"] == p["steps"] + int(6 * (k + 1) / 4)
    assert night["config"]["night_epochs"] == 1.0


def test_backtest_waits_for_paging_only_while_the_ring_has_room():
    """A full ring ends paging early; the queued backtest must then run, not wait for a cutoff
    the deep loop will never reach."""
    import time
    import types
    cfg = SimpleNamespace(deep_backfill_days=200.0)
    rx = np.zeros((6474, 1, 1, 1), dtype=np.float32)
    learner = SimpleNamespace(size=6058, rx=rx)
    fake = types.SimpleNamespace(cfg=cfg, learner=learner, _deep_cutoff=time.time() - 83 * 86400.0)
    assert Engine._paging_pending(fake) is False                       # within a chunk of full: paging is over
    fake.learner.size = 3000
    assert Engine._paging_pending(fake) is True                        # room left and history left: still paging
    fake._deep_cutoff = time.time() - 201 * 86400.0
    assert Engine._paging_pending(fake) is False                       # paged to the configured depth
    fake.cfg.deep_backfill_days = 0
    assert Engine._paging_pending(fake) is False                       # deep backfill off


def test_memory_low_needs_two_low_readings():
    from flint.engine import memory_low
    reads = iter([2.0, 30.0])
    assert memory_low("cuda", 10.0, reader=lambda d: next(reads), settle=0.0) is None     # a dip that passed
    reads = iter([2.0, 3.0])
    assert memory_low("cuda", 10.0, reader=lambda d: next(reads), settle=0.0) == 2.0      # still low: stop
    assert memory_low("cuda", 10.0, reader=lambda d: 40.0, settle=0.0) is None
    assert memory_low("cuda", 10.0, reader=lambda d: None, settle=0.0) is None            # no figure: never stop on it


def test_wait_idle_waits_out_a_training_step_in_flight():
    import threading
    import time
    from flint.learner import OnlineLearner
    cfg = toy_cfg()
    L = OnlineLearner(cfg, n_features=3)
    held = threading.Event()
    def step():
        with L.lock:                                                       # what train_steps holds while it runs
            held.set()
            time.sleep(0.3)
    t = threading.Thread(target=step); t.start(); held.wait()
    t0 = time.time(); L.wait_idle(); dt = time.time() - t0
    t.join()
    assert dt >= 0.25                                                      # it waited for the step, not past it
    assert time.time() - t0 < 2.0


def test_progress_reports_every_day_warm_ones_included():
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay()
    calls = []
    run_backtest(cfg, rx, ry, rmask, rt, device="cpu", warm_days=3, steps_per_day=1,
                 progress=lambda day, total, totals, info=None: calls.append((day, total, info)))
    assert len(calls) == 8                                                             # one per history day
    assert [c[2]["phase"] for c in calls] == ["warming"] * 3 + ["scoring"] * 5
    assert [c[2]["history_day"] for c in calls] == list(range(1, 9)) and calls[0][2]["history_days"] == 8
    assert calls[0][0] == 0 and calls[-1][0] == 5 and calls[-1][1] == 5 and calls[0][2]["warm_days"] == 3
    rows = calls[-1][2]["days"]
    assert len(rows) == 5 and rows[-1]["date"] and len(rows[-1]["gate"]) == cfg.n_experts       # day rows ride along with progress
    assert abs(sum(rows[-1]["gate"]) - 1.0) < 0.01                                               # the day's mean expert mix
    assert calls[2][2]["days"] == []                                                              # nothing scored yet in the warm days


def test_a_finished_variant_rides_along_as_the_baseline_while_the_next_one_runs():
    """The variants run back to back inside one job and the combined report is saved only at the
    end, so until 0.5.15 the first variant's result vanished from the tab for the whole second
    run (the 2026-10-11 weekend: 15 hours with nothing to look at). `variant_summary` is what the
    job appends to the progress, and the summary serves it as the baseline while running."""
    from types import SimpleNamespace
    from flint.backtest import variant_summary
    from flint.engine import Engine
    rep = {"config": {"target": "absolute"}, "totals": {"net_bps": -144.1, "sharpe": -3.07, "days": 74}, "stopped": False,
           "days": [{"date": "2026-06-01", "net_bps": 20.0, "cum_net_bps": 20.0, "hit": 0.5, "gate": [0.1] * 12},
                    {"date": "2026-06-02", "net_bps": -39.0, "cum_net_bps": -19.0, "hit": 0.49}]}
    vs = variant_summary(rep)
    assert vs["target"] == "absolute" and vs["totals"]["net_bps"] == -144.1 and vs["stopped"] is False
    assert vs["days"] == [{"date": "2026-06-01", "cum_net_bps": 20.0}, {"date": "2026-06-02", "cum_net_bps": -19.0}]   # the curve only
    old = {"ran_at": 1.0, "totals": {"net_bps": 1.0}, "days": [], "baseline": {"target": "stale"}, "config": {}}
    eng = SimpleNamespace(backtest=old, sweep=None,
                          backtest_progress={"running": True, "variant": "relative", "variants": ["absolute", "relative"],
                                             "day": 21, "totals": {"net_bps": -20.2}, "days": [], "done": [vs]})
    s = Engine.backtest_summary(eng)
    assert s["baseline"] is vs and s["totals"] == {"net_bps": -20.2}                 # the finished variant, not the old report's
    eng.backtest_progress = {"running": True, "variant": "absolute", "variants": ["absolute", "relative"], "day": 3, "done": []}
    assert Engine.backtest_summary(eng)["baseline"] is None                           # first variant still running: nothing finished yet
    eng.backtest_progress = {"running": False}
    assert Engine.backtest_summary(eng)["baseline"] == {"target": "stale"}            # idle: the saved report's own baseline
    # while running, the scored rows ride under "rows" and "days" stays the count the tab prints (before 0.5.22 the rows
    # overwrote the count and the tab read "day 5 of [object Object],...")
    rows = [{"date": "2026-07-14", "cum_net_bps": -5.4}, {"date": "2026-07-15", "cum_net_bps": -7.6}]
    eng.backtest_progress = {"running": True, "variant": "relative", "variants": ["relative"], "day": 2, "days": 74, "rows": rows,
                             "totals": {"days": 2}, "done": []}
    s = Engine.backtest_summary(eng)
    assert s["days"] == rows and s["progress"]["days"] == 74
