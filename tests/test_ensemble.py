"""Item 4: the learner as an ensemble of members, in the live learner, the checkpoint, the backtest and the sweep."""
import numpy as np
import torch

from flint.learner import OnlineLearner
from tests.test_checkpoint import fill, make_cfg


def _learner(members, **kw):
    cfg = make_cfg(**kw)
    cfg.members = members
    return OnlineLearner(cfg, n_features=3)


def test_members_differ_take_turns_and_forecast_as_their_mean():
    L = _learner(2)
    assert L.members == 2 and len(L.models) == 2 and L.model is L.models[0] and L.opt is L.opts[0]
    a, b = L.models
    assert not torch.equal(a.asset_emb, b.asset_emb)                            # different draws
    assert L.n_params == 2 * sum(p.numel() for p in a.parameters())
    fill(L, 12)
    before = [m.asset_emb.detach().clone() for m in L.models]
    L.steps = 0
    L.train_steps(1)                                                              # step 0 is member 0's
    assert not torch.equal(L.models[0].asset_emb, before[0]) and torch.equal(L.models[1].asset_emb, before[1])
    L.train_steps(1)                                                              # step 1 is member 1's
    assert not torch.equal(L.models[1].asset_emb, before[1])
    x = L.rng.random((2, L.cfg.window, 3), dtype=np.float32)
    p = L.predict(x)
    with torch.no_grad():
        outs = [m.eval()(torch.from_numpy(x)[None]) for m in L.models]
    q = torch.stack([o[0][0] for o in outs]).mean(0).numpy()
    pu = torch.stack([torch.sigmoid(o[1][0]) for o in outs]).mean(0).numpy()
    assert np.allclose(p.q, q, atol=1e-5) and np.allclose(p.p_up, pu, atol=1e-5)
    assert (np.diff(p.q, axis=1) >= -1e-6).all()                                  # the mean of non-crossing quantiles does not cross
    v = L.validate()
    assert v is None or np.isfinite(v["pinball"])


def test_one_member_is_the_old_learner():
    L = _learner(1)
    fill(L, 12)
    x = L.rng.random((2, L.cfg.window, 3), dtype=np.float32)
    p = L.predict(x)
    with torch.no_grad():
        q, up, dn, gate, attn = L.model.eval()(torch.from_numpy(x)[None])
    assert np.allclose(p.q, q[0].numpy()) and np.allclose(p.p_up, torch.sigmoid(up[0]).numpy())


def test_checkpoint_carries_every_member_and_seeds_across_member_counts(tmp_path):
    two = _learner(2)
    fill(two, 10)
    two.train_steps(4)
    two.save(tmp_path, extra={"k": 1})
    again = _learner(2)
    assert again.load(tmp_path) == {"k": 1}
    for m_saved, m_loaded in zip(two.models, again.models):
        for k, v in m_saved.state_dict().items():
            assert torch.equal(v, m_loaded.state_dict()[k])
    one = _learner(1)                                                             # an ensemble's first member serves a one-member run
    one.load(tmp_path)
    for k, v in two.models[0].state_dict().items():
        assert torch.equal(v, one.model.state_dict()[k])
    solo = _learner(1)
    fill(solo, 10)
    solo.save(tmp_path / "one")
    three = _learner(3)                                                           # a one-member checkpoint seeds an ensemble
    three.load(tmp_path / "one")
    for k, v in solo.model.state_dict().items():
        assert torch.equal(v, three.models[0].state_dict()[k])
    assert not torch.equal(three.models[1].asset_emb, three.models[0].asset_emb)
    assert three.size == solo.size == 10


def test_universe_migration_moves_every_members_rows(tmp_path):
    A = _learner(2, symbols=("AAA", "BBB", "CCC"))
    fill(A, 10)
    A.save(tmp_path)
    B = _learner(2, symbols=("BBB", "CCC", "DDD"))
    B.load(tmp_path)
    for k in range(2):
        ea, eb = A.models[k].asset_emb.detach(), B.models[k].asset_emb.detach()
        assert torch.allclose(eb[0], ea[1]) and torch.allclose(eb[1], ea[2])
    assert B.migrated_new == ["DDD"]


def test_backtest_ensemble_stores_the_first_member_and_the_sweep_replays_either(tmp_path):
    from flint.backtest import run_backtest
    from flint.sweep import load_forecasts, replay_policy
    from tests.test_backtest import toy_cfg, toy_replay
    cfg = toy_cfg()
    rx, ry, rmask, rt = toy_replay(n_days=6)
    res = run_backtest(cfg, rx, ry, rmask, rt, device="cpu", symbols=["A", "B"], warm_days=3, steps_per_day=2,
                       store=str(tmp_path / "fc"), members=2)
    assert res["config"]["members"] == 2 and res["totals"]["days"] == 3
    fc = load_forecasts(tmp_path / "fc_absolute.npz")
    assert fc["members"] == 2 and fc["member"] is None and "q0" in fc
    one = load_forecasts(tmp_path / "fc_absolute.npz", member=0)
    assert one["member"] == 0 and one["meta"] is None and one["q"].shape == fc["q"].shape
    assert not np.allclose(one["q"], fc["q"])                                      # one net is not the mean of two
    both = replay_policy(fc, cfg, 5.0)
    single = replay_policy(one, cfg, 5.0)
    assert abs(both["net_bps"] - res["totals"]["net_bps"]) < 0.5                  # the replay still reproduces the run
    assert np.isfinite(single["net_bps"])
