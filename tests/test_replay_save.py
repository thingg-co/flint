"""The replay buffer is written only when it has new windows and the interval has passed, or
at shutdown; the model every call. A full buffer is tens of GB and the save holds the lock."""
import json

from flint.learner import read_replay
from tests.test_replay_dedupe import make_learner, window


def stamp(d):
    return json.loads((d / "replay" / "meta.json").read_text())["saved_at"]


def test_replay_is_written_sparingly(tmp_path):
    L = make_learner()
    L.add(*window(L, 0), ts=1_700_000_000.0)
    L.save(tmp_path)                                         # first save: nothing on disk yet, written
    first = stamp(tmp_path)
    L.add(*window(L, 1), ts=1_700_000_300.0)
    L.save(tmp_path)                                         # new window but inside the interval: model only
    assert stamp(tmp_path) == first
    L.save(tmp_path, force_replay=True)                      # shutdown: written
    assert stamp(tmp_path) != first and read_replay(tmp_path)["n"] == 2
    forced = stamp(tmp_path)
    L.save(tmp_path, force_replay=True)                      # nothing new: not rewritten even when forced
    assert stamp(tmp_path) == forced
    L.add(*window(L, 2), ts=1_700_000_600.0)
    L._replay_at = 0.0                                       # interval elapsed
    L.save(tmp_path)
    assert read_replay(tmp_path)["n"] == 3


def test_loaded_replay_counts_as_saved(tmp_path):
    L = make_learner()
    L.add(*window(L, 0), ts=1_700_000_000.0)
    L.save(tmp_path, force_replay=True)
    M = make_learner()
    M.load(tmp_path)
    before = stamp(tmp_path)
    M.save(tmp_path, force_replay=True)                      # same windows as on disk: no rewrite
    assert stamp(tmp_path) == before
