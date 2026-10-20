"""The lookahead guards: labels resolve by trading-bar count through one gate (matching the
backtest's overnight semantics), the startup walk standardizes old bars by their own day's
statistics, and the replay refuses a window whose label is still in the future."""
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from flint.engine import Engine, Pending
from flint.learner import bar_resolved, label_resolved
from tests.test_replay_dedupe import make_learner, window


def test_startup_walk_pends_from_the_fresh_normalizer_builder():
    """The startup walk pushes rows through the live builder (warm-up) but the replay
    window's features come from the throwaway fresh-normalizer builder: old bars must be
    standardized by their own day's statistics, not the present's (9d, here at startup)."""
    pushed = []

    class Live:
        def push(self, row):
            pushed.append("live")
            return np.zeros((1, 4, 3), dtype=np.float32)   # the live builder's x

    class Fresh:
        def push(self, row):
            pushed.append("fresh")
            return np.ones((1, 4, 3), dtype=np.float32)    # the replay builder's x

    fake = SimpleNamespace(
        bar_index=0,
        bars={},
        cfg=SimpleNamespace(chart_bars=10),
        features=Live(),
        pending=deque(),
        _mature=lambda closes, real: 0,
        symbols=["A", "B"],
    )
    row = {s: SimpleNamespace(ts=1.0, close=1.0, volume=1.0, open=1.0, high=1.0, low=1.0) for s in ("A", "B")}
    Engine._history_row(fake, row, Fresh())
    assert pushed == ["live", "fresh"]                     # both builders see the row
    assert fake.pending[0].x[0, 0, 0] == 1.0               # the pending carries the fresh x
    # without a replay builder, the live builder's own x is used
    fake.pending.clear()
    Engine._history_row(fake, row, None)
    assert fake.pending[0].x[0, 0, 0] == 0.0


def test_label_resolved_waits_for_the_horizon():
    horizon, bar = 12, 300.0
    assert not label_resolved(1_000_000.0, 1_000_000.0 + 12 * bar - 1, horizon, bar)
    assert label_resolved(1_000_000.0, 1_000_000.0 + 12 * bar, horizon, bar)


def test_bar_resolved_counts_trading_bars_not_wall_time():
    """A window resolves 12 trading bars later even across a gap: Friday's close resolves at
    the next session's eleventh bar, exactly as the backtest labels it (9a)."""
    assert not bar_resolved(0, 11, 12)
    assert bar_resolved(0, 12, 12)
    assert bar_resolved(100, 100 + 12, 12)      # bar count, so a weekend gap is just two more bars


def test_add_refuses_a_window_whose_label_is_still_in_the_future():
    L = make_learner()
    x, y = window(L, 1)
    future = time.time() + 2 * L.cfg.horizon * L.cfg.bar_seconds
    assert L.add(x, y, ts=future) is False          # the tripwire: future bars never train
    assert L.size == 0 and L.labels == 0
    past = time.time() - 2 * L.cfg.horizon * L.cfg.bar_seconds
    assert L.add(x, y, ts=past) is True             # a resolved window still adds
    assert L.size == 1


def test_mature_resolves_horizon_bars_after_the_window():
    """The gate is the bar count, as the backtest's is: a window resolves exactly `horizon`
    bars later, never before, whatever the wall clock says."""
    horizon = 12
    added = []

    class Learner:
        labels = 0
        def add(self, x, y, mask, ts=None):
            added.append(ts)
            return True

    fake = SimpleNamespace(
        cfg=SimpleNamespace(horizon=horizon, bar_seconds=300.0, min_labels=0),
        pending=deque(),
        bar_index=11,                                 # one short of the horizon: no resolution
        learner=Learner(),
        metrics=SimpleNamespace(labels=0, live_labels=0, trusted=False),
    )
    closes = np.array([100.0], dtype=np.float32)
    real = np.array([True])
    fake.pending.append(Pending(0, 2_000_000.0, np.zeros((1, 4, 3), dtype=np.float32), closes, real, None, {}))
    assert Engine._mature(fake, closes, real) == 0
    assert len(fake.pending) == 1 and added == []
    fake.bar_index = 12                            # at the horizon: resolved
    assert Engine._mature(fake, closes, real) == 1
    assert len(fake.pending) == 0 and added == [2_000_000.0]


def test_mature_never_resolves_early_across_a_gap():
    """A feed gap means fewer rows, not faster resolution: bar_index still has to advance
    the full horizon, so a window cannot resolve against a close that is not yet known."""
    horizon = 12
    fake = SimpleNamespace(
        cfg=SimpleNamespace(horizon=horizon, bar_seconds=300.0, min_labels=0),
        pending=deque(),
        bar_index=12,                               # 12 rows processed, but the window is bar 1
        learner=SimpleNamespace(add=lambda x, y, mask, ts=None: True, labels=0),
        metrics=SimpleNamespace(labels=0, live_labels=0, trusted=False),
    )
    closes = np.array([10.0], dtype=np.float32)
    real = np.array([True])
    fake.pending.append(Pending(1, 2_000_000.0, np.zeros((1, 4, 3), dtype=np.float32), closes, real, None, {}))
    assert Engine._mature(fake, closes, real) == 0  # window at bar 1 needs bar_index >= 13
    fake.bar_index = 13
    assert Engine._mature(fake, closes, real) == 1
