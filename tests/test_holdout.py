"""Five percent of sessions are held out of training for good, and the idle trainer stops a
session when the held-out loss stops improving."""
import numpy as np

from flint.engine import _stalled
from flint.learner import HOLDOUT_EVERY, HOLDOUT_SHIFT
from tests.test_replay_dedupe import make_learner, window


def add_days(L, days, bars=10, t0=1_700_000_000.0):
    for d in range(days):
        for b in range(bars):
            L.add(*window(L, d * bars + b), ts=t0 + 86400 * d + 300 * b)


def test_whole_sessions_are_held_out_and_never_sampled():
    L = make_learner(cap=1024)
    add_days(L, 56, bars=10)                               # 56 calendar days of 10 bars: two held-out days
    held = np.flatnonzero(L.rhold[:L.size])
    assert len(held) == 20
    days = {((int(L.rt[i]) - HOLDOUT_SHIFT) // 86400) for i in held}
    assert len(days) == 2 and all(d % HOLDOUT_EVERY == 0 for d in days)
    for d in days:                                         # a held-out day is held out whole
        rows = [i for i in range(L.size) if (int(L.rt[i]) - HOLDOUT_SHIFT) // 86400 == d]
        assert all(L.rhold[i] for i in rows) and len(rows) == 10
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
    L = make_learner(cap=1024)
    add_days(L, 56, bars=10)
    v = L.validate()
    assert v and v["n"] == 20 and v["pinball"] > 0 and v["bce"] > 0
    L.save(tmp_path, force_replay=True)
    M = make_learner(cap=1024)
    M.load(tmp_path)
    assert np.array_equal(M.rhold[:M.size], L.rhold[:L.size])


def test_stalled():
    assert not _stalled([5.0, 4.0, 3.0, 2.9], 3)          # still improving
    assert _stalled([3.0, 2.8, 2.9, 2.85, 2.95], 3)        # best was before the last three checks
    assert not _stalled([3.0, 2.8], 3)                     # too few checks to say
    assert not _stalled([3.0, 2.8, 2.9, 2.85, 2.95], 0)    # guard off
