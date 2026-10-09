"""A universe that changes without losing the model: migration of the checkpoint and the
replay across a symbol-list change, column fill for new names, and the promotion/demotion
rules that keep the universe at core + holdings + the most interesting names."""
import json
import shutil

import numpy as np
import torch

from flint.config import Config
from flint.engine import headline_tickers, pick_universe, rank_scores
from flint.learner import OnlineLearner, read_replay


def make(symbols, cap=64):
    cfg = Config()
    cfg.symbols = list(symbols)
    cfg.device = "cpu"
    cfg.torch_threads = 1
    cfg.d_model = 16
    cfg.dilations = (1, 2)
    cfg.n_experts = 2
    cfg.n_heads = 2
    cfg.window = 8
    cfg.replay_size = cap
    cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    cfg.batch_size = 4
    cfg.compile = False
    return OnlineLearner(cfg, n_features=3)


def fill_windows(L, n, seed=0, t0=1_700_000_000.0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        L.add(rng.random((len(L.cfg.symbols), L.cfg.window, 3), dtype=np.float32),
              rng.random((len(L.cfg.symbols),), dtype=np.float32), ts=t0 + 300 * i)


def test_migration_keeps_retained_names_and_adds_fresh_ones(tmp_path):
    A = make(["AAA", "BBB", "CCC"])
    fill_windows(A, 20)
    A.train_steps(3)
    A.save(tmp_path, force_replay=True)
    emb_a = A.model.asset_emb.detach().clone()
    B = make(["BBB", "CCC", "DDD"])                               # AAA dropped, DDD new, order changed
    fresh_d = B.model.asset_emb.detach()[2].clone()
    assert B.load(tmp_path) is not None
    assert B.migrated_new == ["DDD"] and B.steps == A.steps and B.size == 20
    assert torch.allclose(B.model.asset_emb.detach()[0], emb_a[1]) and torch.allclose(B.model.asset_emb.detach()[1], emb_a[2])
    assert torch.allclose(B.model.asset_emb.detach()[2], fresh_d)
    assert np.array_equal(B.rx[:20, 0], A.rx[:20, 1]) and np.array_equal(B.ry[:20, 1], A.ry[:20, 2])
    assert not B.rx[:20, 2].any() and not B.rmask[:20, 2].any()  # DDD: empty and masked until history fills it
    assert B.rmask[:20, :2].all()
    B.train_steps(2)                                              # the migrated optimizer state is usable
    # the re-walk fills DDD's column, and only that column
    x = np.full((3, 8, 3), 0.5, dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert B.add(x, y, ts=1_700_000_300.0) is False               # already held
    assert B.fill(1_700_000_300.0, x, y) == 1
    assert B.rmask[1, 2] == 1.0 and B.ry[1, 2] == 3.0 and B.ry[1, 0] != 1.0
    assert B.fill(1_700_000_300.0, x, y) == 0                    # nothing left to fill


def test_total_mismatch_still_resets(tmp_path):
    A = make(["AAA", "BBB"])
    fill_windows(A, 10)
    A.save(tmp_path, force_replay=True)
    B = make(["XXX", "YYY"])                                      # no name in common: a different universe
    assert B.load(tmp_path) is None and (tmp_path / "model.pt.bak").exists() and (tmp_path / "replay.bak").exists()


def test_release_then_load_reallocates(tmp_path):
    A = make(["AAA", "BBB"])
    fill_windows(A, 12)
    A.save(tmp_path, force_replay=True)
    A.release_replay()
    assert A.size == 0 and A.rx.shape[0] == 0
    A.load(tmp_path)
    assert A.size == 12 and A.rx.shape[0] == 64


def test_read_replay_handles_both_forms(tmp_path):
    A = make(["AAA", "BBB"])
    fill_windows(A, 5)
    A.save(tmp_path, force_replay=True)
    rep = read_replay(tmp_path)
    assert rep["n"] == 5 and rep["symbols"] == ["AAA", "BBB"] and rep["x"].shape == (5, 2, 8, 3)
    shutil.rmtree(tmp_path / "replay")
    np.savez(tmp_path / "replay.npz", x=A.rx[:5], y=A.ry[:5], mask=A.rmask[:5], t=A.rt[:5], ptr=5)
    rep = read_replay(tmp_path)
    assert rep["n"] == 5 and rep["symbols"] == [] and (np.asarray(rep["t"]) > 0).all()


def test_headline_tickers_only_known_symbols():
    hs = [{"title": "Final Trade: PBR, GTLB, XLE, GOOGL"}, {"title": "CEO says AI will lift GOOGL and NOW"}, {"title": "GOOGL hits ATH"}]
    counts = headline_tickers(hs, known={"PBR", "GTLB", "GOOGL", "NOW", "ATH", "CEO"})
    assert counts == {"PBR": 1, "GTLB": 1, "GOOGL": 3}            # NOW/ATH/CEO are words, XLE unknown


def test_rank_scores_one_scale():
    s = rank_scores(["A", "B", "C", "D"], span=4)
    assert s["A"] == 100.0 and s["C"] == 50.0 and s["D"] == 25.0


def test_promotion_and_demotion():
    core, held = ["AAA", "BBB"], {"HLD"}
    cands = {"M1": 95.0, "M2": 90.0, "M3": 80.0, "OLD": 10.0}
    # day 1: nothing interesting yet; room for 2
    u, rec, promoted, demoted = pick_universe(core, held, ["AAA", "BBB", "HLD"], cands, {}, count=2, max_universe=10, today="2026-10-10")
    assert u == ["AAA", "BBB", "HLD", "M1", "M2"] and promoted == ["M1", "M2"] and demoted == []
    assert rec["M1"]["since"] == "2026-10-10"
    # day 2: M1 falls out of the band but is young and within grace: stays; M3 cannot enter (no room)
    cands2 = {"M2": 95.0, "M3": 90.0, "X1": 85.0, "X2": 84.0, "M1": 1.0}
    u2, rec2, p2, d2 = pick_universe(core, held, u, cands2, rec, count=2, max_universe=10, today="2026-10-12")
    assert u2 == u and p2 == [] and d2 == [] and rec2["M1"]["last_hit"] == "2026-10-10"
    # three weeks later with M1 still cold: demoted, M3 promoted into its slot
    u3, rec3, p3, d3 = pick_universe(core, held, u2, cands2, rec2, count=2, max_universe=10, today="2026-11-02")
    assert d3 == ["M1"] and p3 == ["M3"] and u3 == ["AAA", "BBB", "HLD", "M2", "M3"] and "M1" not in rec3
    # a sold holding that is not interesting leaves after the grace period too; a new holding always enters
    u4, _, p4, d4 = pick_universe(core, {"NEWH"}, u3, {"M2": 95.0, "M3": 90.0}, rec3, count=2, max_universe=10, today="2026-11-20")
    assert "NEWH" in u4 and "HLD" not in u4 and "HLD" in d4 and "NEWH" not in p4   # a holding is fixed, not promoted
    # the room shrinks when holdings grow: the weakest interesting name goes regardless of grace
    u5, _, _, d5 = pick_universe(core, {"H1", "H2", "H3", "H4", "H5", "H6"}, u3, {"M2": 95.0, "M3": 90.0}, rec3, count=2, max_universe=9, today="2026-11-03")
    assert set(d5) == {"M3", "HLD"} and len(u5) == 9           # HLD was sold and is not interesting: it goes too


def test_fresh_model_keeps_the_replay_when_the_preset_changed(tmp_path):
    A = make(["AAA", "BBB", "CCC"])
    fill_windows(A, 20)
    A.train_steps(2)
    A.save(tmp_path, force_replay=True, extra={"norm": {"k": 1}, "paper": {"cash": 5}, "metrics": {"live_labels": 9}, "outcomes": {}})
    B = make(["BBB", "CCC", "DDD"])
    B.cfg.d_model = 32                                           # a different preset: the saved weights cannot load
    B.model = __import__("flint.model", fromlist=["FlintNet"]).FlintNet(3, 3, 5, 32, (1, 2), 2, 2, 0.1)
    B.opt = torch.optim.AdamW(B.model.parameters(), lr=1e-3)
    extra = B.load(tmp_path)
    assert extra == {"norm": {"k": 1}, "paper": {"cash": 5}}       # the book and the normalization survive; the old metrics do not
    assert B.steps == 0 and B.size == 20 and B.labels == 20          # fresh net, full replay, migrated columns
    assert np.array_equal(B.rx[:20, 0], A.rx[:20, 1]) and not B.rmask[:20, 2].any()
    assert B.migrated_new == ["DDD"] and B.model.asset_emb.shape == (3, 32)
    B.train_steps(2)


def test_replay_columns_alone_flag_new_names(tmp_path):
    """The model may load cleanly (same preset and list) while the replay on disk is from an
    older list: the re-page must still run for the names the replay lacks."""
    A = make(["AAA", "BBB"])
    fill_windows(A, 6)
    A.save(tmp_path, force_replay=True)
    B = make(["AAA", "BBB", "CCC"])
    B._replay_labels = B.labels                                      # nothing new in B's (empty) buffer: the model saves, the replay stays
    B.save(tmp_path)                                                 # a checkpoint for the new list over the older replay
    C = make(["AAA", "BBB", "CCC"])
    C.load(tmp_path)
    assert C.size == 6 and C.migrated_new == ["CCC"] and not C.rmask[:6, 2].any()
