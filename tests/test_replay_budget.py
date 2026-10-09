"""The replay budget in bytes, and a smaller buffer keeping the newest windows on load."""
import numpy as np

from flint.config import Config
from tests.test_universe import fill_windows, make


def test_replay_gb_follows_the_universe():
    cfg = Config()
    cfg.symbols = ["A"] * 232
    cfg.window = 128
    cfg.replay_gb = 36.0
    assert 12000 <= cfg.replay_windows(25) <= 12200
    cfg.symbols = ["A"] * 400
    assert 7000 <= cfg.replay_windows(25) <= 7100
    cfg.replay_gb = 0.0
    cfg.replay_size = 4096
    assert cfg.replay_windows(25) == 4096


def test_smaller_buffer_keeps_the_newest_windows(tmp_path):
    A = make(["AAA", "BBB"], cap=64)
    fill_windows(A, 30)
    A.save(tmp_path, force_replay=True)
    B = make(["AAA", "BBB"], cap=16)
    B.load(tmp_path)
    assert B.size == 16 and B.ptr == 0
    assert np.array_equal(B.rt[:16], A.rt[14:30])                    # the last 16, in order
    assert np.array_equal(B.ry[:16], A.ry[14:30])
    assert B.add(np.zeros((2, 8, 3), dtype=np.float32), np.zeros(2, dtype=np.float32), ts=float(A.rt[29])) is False   # still deduplicated
