"""The market-relative target: loadings on the modeled indices from the labels held, the label
transform, the per-index hedge, the learner and backtest on either target, and the paper
book's uncapped hedge weight. CPU only, toy sizes."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import torch

from flint.backtest import combine_reports, run_backtest
from flint.config import Config
from flint.learner import OnlineLearner
from flint.paper import PaperBook
from flint.target import estimate_beta, hedge_targets, index_columns, prior_beta, relative

NY = ZoneInfo("America/New_York")


def _labels(n=3000, betas=(0.5, 1.5, 2.0), seed=0):
    """Names loading `betas` on one index, the index last."""
    rng = np.random.default_rng(seed)
    idx = rng.standard_normal(n) * 50
    y = np.stack([b * idx + rng.standard_normal(n) * 10 for b in betas] + [idx], axis=1).astype(np.float32)
    return y


def _labels2(n=4000, seed=0):
    """Two correlated indices (SPY-like, QQQ-like) and two names: one broad, one tech-heavy."""
    rng = np.random.default_rng(seed)
    spy = rng.standard_normal(n) * 40
    qqq = 1.1 * spy + rng.standard_normal(n) * 25
    a = 1.0 * spy + 0.0 * qqq + rng.standard_normal(n) * 10
    b = 0.3 * spy + 1.2 * qqq + rng.standard_normal(n) * 10
    return np.stack([a, b, spy, qqq], axis=1).astype(np.float32)


def test_index_columns_keep_the_configured_order_and_skip_unmodeled_names():
    assert index_columns(["A", "SPY", "QQQ", "B"], "qqq, SPY,IWM,SPY") == [2, 1]
    assert index_columns(["A", "B"], "SPY,QQQ") == []
    assert index_columns(["A", "SPY"], ["spy"]) == [1]


def test_loadings_are_recovered_with_index_rows_at_zero():
    y = _labels()
    beta = estimate_beta(y, None, [3], shrink=0.0)
    assert beta.shape == (4, 1)
    assert np.allclose(beta[:3, 0], [0.5, 1.5, 2.0], atol=0.08)
    assert beta[3, 0] == 0.0 and beta.dtype == np.float32
    strong = estimate_beta(y, None, [3], shrink=1e9)
    assert np.allclose(strong[:3, 0], 1.0, atol=1e-3)               # the prior wins with no evidence against it
    assert (prior_beta(4, [3])[:, 0] == [1, 1, 1, 0]).all()


def test_two_correlated_indices_are_separated_by_the_ridge():
    y = _labels2()
    beta = estimate_beta(y, None, [2, 3], shrink=0.0)
    assert beta.shape == (4, 2)
    assert np.allclose(beta[0], [1.0, 0.0], atol=0.12)
    assert np.allclose(beta[1], [0.3, 1.2], atol=0.12)
    assert (beta[2] == 0).all() and (beta[3] == 0).all()
    strong = estimate_beta(y, None, [2, 3], shrink=1e9)
    assert np.allclose(strong[:2], [[1.0, 0.0], [1.0, 0.0]], atol=1e-3)   # the prior: the first index only


def test_loadings_ignore_rows_a_name_did_not_trade_and_default_to_the_prior():
    y = _labels()
    mask = np.ones_like(y)
    bad = np.arange(0, len(y), 7)
    y[bad, 0] = 1e4                                                   # garbage where the name did not trade
    mask[bad, 0] = 0.0
    beta = estimate_beta(y, mask, [3], shrink=0.0)
    assert abs(beta[0, 0] - 0.5) < 0.08
    assert estimate_beta(np.zeros((0, 4), np.float32), None, [3])[0, 0] == 1.0
    assert estimate_beta(y, mask, []).shape == (4, 0)                # absolute target: no loadings


def test_relative_transform_on_numpy_and_torch_and_identity_when_absolute():
    y = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    beta = np.array([[0.5], [0.0], [2.0]], dtype=np.float32)
    out = relative(y, beta, [1])
    assert np.allclose(out, [[0.0, 20.0, -10.0], [0.0, 2.0, -1.0]])
    assert torch.allclose(relative(torch.from_numpy(y), beta, [1]), torch.from_numpy(out))
    two = relative(y, np.array([[0.5, 1.0], [0.0, 0.0], [0.0, 0.0]], np.float32), [1, 2])
    assert np.allclose(two[:, 0], y[:, 0] - 0.5 * y[:, 1] - 1.0 * y[:, 2])
    assert relative(y, beta, []) is y and relative(y, None, [1]) is y


def test_hedges_cancel_the_basket_loading_per_index_with_its_sign():
    syms = ["A", "B", "SPY", "QQQ"]
    beta = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    h = hedge_targets({"A": 0.1, "B": 0.2, "SPY": 0.9, "QQQ": 0.9}, beta, syms, [2, 3])
    assert abs(h["SPY"] + 0.1) < 1e-9 and abs(h["QQQ"] + 0.4) < 1e-9       # long basket: index puts
    h = hedge_targets({"A": 0.1, "B": -0.2}, beta, syms, [2, 3])
    assert abs(h["SPY"] + 0.1) < 1e-9 and abs(h["QQQ"] - 0.4) < 1e-9       # net bearish on QQQ: QQQ stock
    assert hedge_targets({"A": 0.1}, beta, syms, []) == {}


def _cfg(target: str) -> Config:
    cfg = Config()
    cfg.symbols = ["A", "B", "SPY"]
    cfg.target = target
    cfg.index_symbols = "SPY,QQQ"
    cfg.device = "cpu"
    cfg.torch_threads = 1
    cfg.d_model = 16
    cfg.dilations = (1, 2)
    cfg.n_experts = 2
    cfg.n_heads = 2
    cfg.replay_size = 64
    cfg.window = 8
    cfg.batch_size = 4
    cfg.compile = False
    cfg.beta_shrink = 0.0
    return cfg


def test_learner_trains_on_relative_labels_and_refits_loadings(tmp_path):
    L = OnlineLearner(_cfg("relative"), n_features=5)
    assert L.cols == [2] and L.target_mode == "relative" and L.beta.shape == (3, 1) and L.beta[2, 0] == 0.0
    y = _labels(60, betas=(0.5, 2.0))
    x = np.zeros((3, 8, 5), dtype=np.float32)
    for k in range(60):
        L.add(x, y[k], None, ts=1e9 + 300 * k)
    beta = L.refresh_beta()
    assert np.allclose(beta[:2, 0], [0.5, 2.0], atol=0.15) and beta[2, 0] == 0.0
    assert np.allclose(L.target(y[0]), y[0] - beta[:, 0] * y[0, 2])
    assert L.train_steps(2)["steps"] == 2
    L.save(tmp_path, extra={"target": L.target_mode}, force_replay=True)
    L2 = OnlineLearner(_cfg("relative"), n_features=5)
    extra = L2.load(tmp_path)
    assert extra.get("target") == "relative" and np.allclose(L2.beta, beta, atol=1e-3)

    A = OnlineLearner(_cfg("absolute"), n_features=5)
    assert A.cols == [] and A.target_mode == "absolute"
    yy = y[0]
    assert A.target(yy) is yy
    no_index = _cfg("relative")
    no_index.symbols = ["A", "B"]
    assert OnlineLearner(no_index, n_features=5).cols == []            # no index modeled: absolute, quietly


def _toy(n_days=12, bars=10, seed=0, betas=(1.5,)):
    rng = np.random.default_rng(seed)
    n = n_days * bars
    rx = rng.random((n, 2, 8, 3), dtype=np.float32)
    ry = _labels(n, betas=betas, seed=seed)
    rmask = np.ones((n, 2), dtype=np.float32)
    rt = np.zeros(n)
    t0 = datetime(2026, 9, 1, 9, 35, tzinfo=NY)
    for d in range(n_days):
        for b in range(bars):
            rt[d * bars + b] = (t0 + timedelta(days=d, minutes=5 * b)).timestamp()
    return rx, ry, rmask, rt


def _bt_cfg():
    cfg = Config()
    cfg.symbols = ["A", "SPY"]
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


def test_backtest_runs_the_relative_target_hedged_and_reports_it(tmp_path):
    cfg = _bt_cfg()
    rx, ry, rmask, rt = _toy()
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=["A", "SPY"], warm_days=3, steps_per_day=2,
                       target="relative", index_symbols="SPY,QQQ", beta_shrink=0.0, store=str(tmp_path / "fc"))
    c = res["config"]
    assert c["target"] == "relative" and c["index"] == ["SPY"] and "SPY" not in c["beta"]
    z = np.load(c["forecasts"])                                       # every forecast the run made, for the policy sweep
    n_scored = (12 - 3) * 10
    assert c["forecasts"].endswith("fc_relative.npz") and z["q"].shape == (n_scored, 2, 5) and z["y"].shape == (n_scored, 2)
    assert z["up"].shape == z["dn"].shape == (n_scored, 2) and z["beta"].shape == (9, 2, 1) and str(z["target"]) == "relative"
    assert np.allclose(z["t"], rt[z["rows"]]) and list(z["symbols"]) == ["A", "SPY"] and int(z["day_of"].max()) == 8
    assert abs(c["beta"]["A"]["SPY"] - 1.5) < 0.3                     # fit from the days already seen
    assert res["days"] and all("unhedged_bps" in d for d in res["days"])
    assert np.isfinite(res["totals"]["net_bps"]) and "unhedged_bps" in res["totals"]
    absent = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=["A", "B"], warm_days=3, steps_per_day=1,
                          target="relative", index_symbols="SPY")
    assert absent["config"]["target"].startswith("absolute") and absent["config"]["index"] is None
    plain = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=["A", "SPY"], warm_days=3, steps_per_day=1)
    assert plain["config"]["target"] == "absolute" and plain["config"]["beta"] is None


def test_combine_reports_keeps_the_first_run_as_the_baseline():
    a = {"config": {"target": "absolute"}, "totals": {"net_bps": 1.0}, "days": [{"date": "d1", "cum_net_bps": 1.0, "net_bps": 1.0}]}
    b = {"config": {"target": "relative"}, "totals": {"net_bps": 2.0}, "days": []}
    out = combine_reports([a, b])
    assert out is b and out["baseline"]["target"] == "absolute" and out["baseline"]["days"][0]["cum_net_bps"] == 1.0
    assert combine_reports([a]) is a and "baseline" not in a
    assert combine_reports([a, {"error": "x"}]) is a
    assert combine_reports([{"error": "x"}])["error"] == "x"


def test_paper_book_takes_the_hedge_weight_uncapped_and_outside_the_leverage_scale():
    book = PaperBook(start=100_000.0, cost_bps=0.0, max_weight=0.15)
    prices = {"A": 100.0, "SPY": 500.0}
    pq = {"premium": 10.0, "delta": -0.5, "strike": 500.0, "expiry_ts": 2e9, "iv": 0.2}
    book.rebalance({"A": 0.15, "SPY": -0.6}, prices, 1.0, put_quotes={"SPY": pq}, uncapped={"SPY"})
    assert abs(book.pos["A"] * 100.0 - 15_000.0) < 1.0                 # the basket is not shrunk by the hedge's notional
    put = book.puts["SPY"]
    assert abs(put["contracts"] - 60_000.0 / (100.0 * 500.0 * 0.5)) < 1e-6   # delta-equivalent to 60% of equity
    capped = PaperBook(start=100_000.0, cost_bps=0.0, max_weight=0.15)
    capped.rebalance({"A": 0.15, "SPY": -0.6}, prices, 1.0, put_quotes={"SPY": pq})
    assert capped.puts["SPY"]["contracts"] < put["contracts"] / 3     # without the exemption the weight is clipped to 15% and the premium capped
