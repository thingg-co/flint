"""The policy sweep replays a backtest's stored forecasts to the same net, then tunes the
gates on the first half of the days and confirms on the second."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from flint.backtest import run_backtest
from flint.config import Config
from flint.sweep import load_forecasts, replay_policy, run_sweep

NY = ZoneInfo("America/New_York")


def _cfg():
    cfg = Config()
    cfg.symbols = ["A", "B", "SPY"]
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
    cfg.score_threshold = 0.0
    cfg.prob_margin = 0.0
    cfg.move_floor_bps = 0.0
    cfg.skill_min_n = 0
    cfg.confirm_bars = 1
    cfg.kelly_fraction = 1.0
    cfg.max_size = 1.0
    cfg.size_by_coverage = True
    cfg.min_hit_rate = 0.0
    return cfg


def _replay(n_days=16, bars=12, seed=1):
    rng = np.random.default_rng(seed)
    n = n_days * bars
    rx = rng.random((n, 3, 8, 3), dtype=np.float32)
    spy = rng.standard_normal(n) * 40
    ry = np.stack([1.2 * spy + rng.standard_normal(n) * 30, 0.6 * spy + rng.standard_normal(n) * 30, spy], axis=1).astype(np.float32)
    rmask = (rng.random((n, 3)) > 0.1).astype(np.float32)           # some names sit out some bars
    rmask[:, 2] = 1.0
    rt = np.zeros(n)
    t0 = datetime(2026, 9, 1, 9, 35, tzinfo=NY)
    for d in range(n_days):
        for b in range(bars):
            rt[d * bars + b] = (t0 + timedelta(days=d, minutes=5 * b)).timestamp()
    return rx, ry, rmask, rt


def test_replay_reproduces_the_backtest_on_both_targets(tmp_path):
    cfg = _cfg()
    rx, ry, rmask, rt = _replay()
    for target in ("absolute", "relative"):
        res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=cfg.symbols, warm_days=3, steps_per_day=2,
                           target=target, index_symbols="SPY", beta_shrink=0.0, cost_bps=0.0, store=str(tmp_path / "fc"))
        fc = load_forecasts(res["config"]["forecasts"])
        rep = replay_policy(fc, cfg, 0.0)
        t = res["totals"]
        assert rep["days"] == t["days"] and rep["signals"] == t["signals"] > 0, (target, rep["signals"], t["signals"])
        assert abs(rep["net_bps"] - t["net_bps"]) < 1e-6 * max(1.0, abs(t["net_bps"])), (target, rep["net_bps"], t["net_bps"])
        assert abs(rep["net_bps_per_day"] - t["net_bps_per_day"]) < 1e-9
        assert rep["hit_rate"] == t["hit_rate"] and abs(rep["coverage"] - t["coverage"]) < 1e-12
        assert abs(rep["pinball"] - t["pinball"]) < 1e-9 and abs(rep["max_drawdown_bps"] - t["max_drawdown_bps"]) < 1e-9
        per_day = [d["net_bps"] for d in res["days"]]
        assert np.allclose(rep["nets"], per_day)


def test_sweep_tunes_on_the_first_half_and_confirms_on_the_second(tmp_path):
    cfg = _cfg()
    rx, ry, rmask, rt = _replay(n_days=20)
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=cfg.symbols, warm_days=3, steps_per_day=2,
                       cost_bps=0.0, store=str(tmp_path / "fc"))
    fc = load_forecasts(res["config"]["forecasts"])
    grid = {"score_threshold": [0.0, 0.3, 0.6], "kelly_fraction": [0.5, 1.0], "confirm_bars": [1, 2]}
    rep = run_sweep(fc, cfg, 0.0, grid=grid)
    assert rep["days"] == 17 and rep["split_day"] == 8 and rep["evaluations"] >= 1
    assert rep["current"]["knobs"] == {"score_threshold": 0.0, "kelly_fraction": 1.0, "confirm_bars": 1}
    assert rep["current"]["tune"]["days"] == 8 and rep["current"]["confirm"]["days"] == 9
    if rep["proposal"]:
        assert set(rep["proposal"]["knobs"]) <= set(grid) and rep["proposal"]["tune"]["net_bps_per_day"] >= rep["current"]["tune"]["net_bps_per_day"]
        assert rep["accepted"] == (rep["proposal"]["confirm"]["net_bps_per_day"] > rep["current"]["confirm"]["net_bps_per_day"])
    else:
        assert rep["accepted"] is False
    assert "why" in rep and all(s["knob"] in grid for s in rep["steps"])
    short = dict(fc)
    short["day_of"] = np.minimum(fc["day_of"], 3)
    assert "error" in run_sweep(short, cfg, 0.0, grid=grid)
