"""Five percent of windows are held out of training for good, and the idle trainer stops a
session when the held-out loss stops improving."""
import numpy as np

from flint.engine import _stalled
from flint.learner import HOLDOUT_EVERY
from tests.test_replay_dedupe import make_learner, window


def add_bars(L, n, t0=1_700_000_000.0):
    for i in range(n):
        L.add(*window(L, i), ts=t0 + 300 * i)


def test_every_twentieth_bar_is_held_out_and_never_sampled():
    L = make_learner(cap=256)
    add_bars(L, 200)
    held = np.flatnonzero(L.rhold[:L.size])
    assert len(held) == 10 and all((int(L.rt[i]) // 300) % HOLDOUT_EVERY == 0 for i in held)
    picks = set()
    for _ in range(100):                                   # the sampler's picks, recovered from the labels it returns
        xs, ys, _m = L._sample(8)
        for row in ys.numpy():
            k = int(np.flatnonzero((L.ry[:L.size] == row).all(axis=1))[0])
            picks.add(k)
    assert not (picks & set(held.tolist()))


def test_untimed_windows_are_never_held_out():
    L = make_learner(cap=64)
    for i in range(40):
        L.add(*window(L, i))
    assert not L.rhold[:L.size].any() and L.validate() is None


def test_validate_scores_only_the_holdout_and_survives_a_reload(tmp_path):
    L = make_learner(cap=256)
    add_bars(L, 200)
    v = L.validate()
    assert v and v["n"] == 10 and v["pinball"] > 0 and v["bce"] > 0
    L.save(tmp_path, force_replay=True)
    M = make_learner(cap=256)
    M.load(tmp_path)
    assert np.array_equal(M.rhold[:M.size], L.rhold[:L.size])


def test_stalled():
    assert not _stalled([5.0, 4.0, 3.0, 2.9], 3)          # still improving
    assert _stalled([3.0, 2.8, 2.9, 2.85, 2.95], 3)        # best was before the last three checks
    assert not _stalled([3.0, 2.8], 3)                     # too few checks to say
    assert not _stalled([3.0, 2.8, 2.9, 2.85, 2.95], 0)    # guard off
