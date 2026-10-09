"""The replay buffer is written only when it has new windows and the interval has passed, or
at shutdown; the model every call. A full buffer is tens of GB and the save holds the lock."""
import hashlib

import numpy as np

from tests.test_replay_dedupe import make_learner, window


def digest(path):
    return hashlib.blake2b(path.read_bytes(), digest_size=12).hexdigest()


def test_replay_is_written_sparingly(tmp_path):
    L = make_learner()
    L.add(*window(L, 0), ts=1_700_000_000.0)
    L.save(tmp_path)                                         # first save: nothing on disk yet, written
    r = tmp_path / "replay.npz"
    first = digest(r)
    L.add(*window(L, 1), ts=1_700_000_300.0)
    L.save(tmp_path)                                         # new window but inside the interval: model only
    assert digest(r) == first
    L.save(tmp_path, force_replay=True)                      # shutdown: written
    assert digest(r) != first and len(np.load(r)["y"]) == 2
    forced = digest(r)
    L.save(tmp_path, force_replay=True)                      # nothing new: not rewritten even when forced
    assert digest(r) == forced
    L.add(*window(L, 2), ts=1_700_000_600.0)
    L._replay_at = 0.0                                       # interval elapsed
    L.save(tmp_path)
    assert len(np.load(r)["y"]) == 3


def test_loaded_replay_counts_as_saved(tmp_path):
    L = make_learner()
    L.add(*window(L, 0), ts=1_700_000_000.0)
    L.save(tmp_path, force_replay=True)
    M = make_learner()
    M.load(tmp_path)
    before = digest(tmp_path / "replay.npz")
    M.save(tmp_path, force_replay=True)                      # same windows as on disk: no rewrite
    assert digest(tmp_path / "replay.npz") == before
