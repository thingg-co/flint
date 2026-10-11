"""The meta-labeler: features per forecast, an online logistic fit on matured outcomes, silent
until it has seen enough, its state through the checkpoint, and its gate in the backtest's
`decide` and the replay."""
import numpy as np

from flint.backtest import decide
from flint.config import Config
from flint.meta import K, MetaLabeler, features, label
from flint.sweep import replay_policy


def test_features_and_label_are_per_forecast():
    q = np.array([[-20, -10, 30, 50, 70], [-70, -50, -30, -10, 20]], float)   # a long call and a short one
    f = features(q, [0.7, 0.3], [0.2, 0.8])
    assert f.shape == (2, K)
    assert f[0, 0] == 30 / 60 and f[1, 0] == 30 / 40                           # |q50| / IQR
    assert abs(f[0, 1] - 0.4) < 1e-12 and abs(f[1, 1] - 0.4) < 1e-12            # |2 P(up) - 1|
    assert abs(f[0, 4] - 0.5) < 1e-12 and abs(f[1, 4] - 0.5) < 1e-12            # both heads agree with the median's sign
    assert list(label([12.0, -12.0, 3.0, 12.0], [30, -30, 30, -30], 5.0)) == [1, 1, 0, 0]
    assert features(q[0], 0.7, 0.2).shape == (K,)


def test_fit_learns_a_monotone_rule_and_is_silent_before_enough_rows():
    rng = np.random.default_rng(0)
    m = MetaLabeler(cap=5000, min_rows=200, refit_every=100)
    x = rng.random((3000, K))
    t = (x[:, 0] + 0.3 * rng.standard_normal(3000) > 0.5).astype(float)
    m.update(x[:100], t[:100])
    assert not m.fitted and m.predict(x[:3]) is None
    m.update(x[100:], t[100:])
    assert m.fitted and m.fits >= 1 and m.n == 3000
    p = m.predict(x)
    assert p.shape == (3000,) and np.all((p > 0) & (p < 1))
    assert p[x[:, 0] > 0.7].mean() > p[x[:, 0] < 0.3].mean() + 0.3
    assert abs(p.mean() - t.mean()) < 0.05                                       # calibrated on average


def test_ring_keeps_the_newest_rows_and_the_state_round_trips():
    m = MetaLabeler(cap=50, min_rows=20, refit_every=10)
    x = np.arange(80 * K, dtype=float).reshape(80, K) / 100
    t = (np.arange(80) % 2).astype(float)
    m.update(x, t)
    assert len(m.t) == 50 and m.n == 80 and np.allclose(m.x[-1], x[-1])
    fresh = MetaLabeler(cap=50)
    assert fresh.load(m.state()) and fresh.fitted and fresh.n == 80
    assert np.allclose(fresh.predict(x[:5]), m.predict(x[:5]))
    assert MetaLabeler().load({}) and not MetaLabeler().fitted                   # an empty state is a fresh model
    assert not MetaLabeler().load({"w": [1, 2, 3]}) and not MetaLabeler().load("junk")
    assert not MetaLabeler().load({"x": np.zeros((3, K)), "t": np.zeros(2)})
    bad = MetaLabeler()
    bad.update(np.full((5, K), np.nan), np.ones(5))
    assert len(bad.t) == 0                                                        # NaN rows are dropped, never fitted on


def _open_cfg():
    cfg = Config()
    for k, v in dict(score_threshold=0.0, prob_margin=0.0, move_floor_bps=0.0, skill_min_n=0, min_hit_rate=0.0,
                     confirm_bars=1, kelly_fraction=1.0, max_size=1.0, meta_min=0.5).items():
        setattr(cfg, k, v)
    return cfg


def test_decide_holds_on_low_odds_and_trades_without_an_opinion():
    cfg = _open_cfg()
    q = np.array([10, 20, 40, 60, 80], float)
    args = (cfg, q, 0.8, 0.1, 1.0, 1.0, None, None, None, 0.0)
    assert decide(*args)[0] == 1 and decide(*args, meta_p=None)[0] == 1 and decide(*args, meta_p=0.9)[0] == 1
    assert decide(*args, meta_p=0.3)[0] == 0
    cfg.meta_min = 0.0
    assert decide(*args, meta_p=0.3)[0] == 1                                      # 0 disables the gate


def _fc(n_days=4, bars=10):
    n = n_days * bars
    q50 = np.full((n, 3), 50.0, dtype=np.float32)
    q = np.stack([q50 - 20, q50 - 10, q50, q50 + 10, q50 + 20], axis=-1)
    return {"q": q, "up": np.full((n, 3), 0.9, np.float32), "dn": np.full((n, 3), 0.1, np.float32),
            "y": np.full((n, 3), 10.0, np.float32), "mask": np.ones((n, 3), np.uint8), "beta": np.zeros((n_days, 3, 0), np.float32),
            "day_of": np.repeat(np.arange(n_days), bars), "symbols": ["A", "B", "C"], "index": [], "target": "absolute"}


def test_the_replay_applies_the_stored_odds_where_the_model_had_an_opinion():
    cfg = _open_cfg()
    fc = _fc()
    base = replay_policy(fc, cfg, 0.0)["signals"]
    assert base > 0
    meta = np.full((40, 3), np.nan, np.float32)                                  # no opinion anywhere: nothing changes
    assert replay_policy(dict(fc, meta=meta), cfg, 0.0)["signals"] == base
    meta[:, 1] = 0.2                                                               # B never pays, in the model's view
    meta[:, 2] = 0.9
    assert replay_policy(dict(fc, meta=meta), cfg, 0.0)["signals"] == base * 2 // 3
    assert replay_policy(dict(fc, meta=meta), cfg, 0.0, {"meta_min": 0.0})["signals"] == base
    assert replay_policy(dict(fc, meta=meta[:10]), cfg, 0.0)["signals"] == base   # a stored array of the wrong shape is ignored
