"""One replay window per labeled bar, across restarts. Exists because every restart re-walked
the same ten days of startup history into the persisted replay: 526 windows held 180 distinct
label vectors on 2026-10-10, so recent days were trained on three times over."""
import numpy as np

from flint.config import Config
from flint.learner import OnlineLearner


def make_learner(cap=16):
    cfg = Config()
    cfg.symbols = ["A", "B"]
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


def window(learner, seed):
    r = np.random.default_rng(seed)
    return r.random((2, learner.cfg.window, 3), dtype=np.float32), r.random((2,), dtype=np.float32)


def test_same_bar_is_stored_once():
    L = make_learner()
    x, y = window(L, 1)
    assert L.add(x, y, ts=1_700_000_300.0) is True
    assert L.add(x, y, ts=1_700_000_300.0) is False          # the restart re-walk
    assert L.add(x, y, ts=1_700_000_600.0) is True
    assert (L.size, L.labels) == (2, 2)


def test_untimed_windows_still_add():
    L = make_learner()
    x, y = window(L, 2)
    assert L.add(x, y) and L.add(x, y)                       # legacy callers: no key, no dedupe
    assert L.size == 2


def test_dedupe_survives_save_and_load(tmp_path):
    L = make_learner()
    for i in range(5):
        L.add(*window(L, i), ts=1_700_000_000.0 + 300 * i)
    L.save(tmp_path)
    M = make_learner()
    M.load(tmp_path)
    assert M.size == 5
    assert M.add(*window(M, 1), ts=1_700_000_300.0) is False    # the bar a restart would re-add
    assert M.add(*window(M, 9), ts=1_700_009_000.0) is True


def test_legacy_replay_without_timestamps_loads(tmp_path):
    L = make_learner()
    for i in range(3):
        L.add(*window(L, i), ts=1_700_000_000.0 + 300 * i)
    L.save(tmp_path)
    z = dict(np.load(tmp_path / "replay.npz"))
    del z["t"]
    np.savez_compressed(tmp_path / "replay.npz", **z)
    M = make_learner()
    M.load(tmp_path)
    assert M.size == 3 and not M._seen                        # kept, not deduplicated


def test_ring_overwrite_frees_the_old_bar():
    L = make_learner(cap=2)
    for i in range(3):                                        # third add overwrites slot 0
        L.add(*window(L, i), ts=1_700_000_000.0 + 300 * i)
    assert L.size == 2
    assert L.add(*window(L, 0), ts=1_700_000_000.0) is True   # bar 0 left the buffer, so it may return
