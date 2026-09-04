"""Tests for OnlineLearner.save/load in flint/learner.py"""
import numpy as np
import pytest
import torch

from flint.learner import OnlineLearner
from flint.config import Config


def make_cfg(symbols=("A", "B"), window=8):
    cfg = Config()
    cfg.symbols = list(symbols)
    cfg.device = "cpu"
    cfg.torch_threads = 1
    cfg.d_model = 16
    cfg.dilations = (1, 2)
    cfg.n_experts = 2
    cfg.n_heads = 2
    cfg.window = window
    cfg.replay_size = 16
    cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)  # 5 quantiles
    cfg.batch_size = 4
    cfg.compile = False
    return cfg


def make_learner(**kw):
    return OnlineLearner(make_cfg(**kw), n_features=3)


def fill(learner, n):
    """Add n samples to learner."""
    n_assets = len(learner.cfg.symbols)
    for _ in range(n):
        x = learner.rng.random((n_assets, learner.cfg.window, 3), dtype=np.float32)
        y = learner.rng.random((n_assets,), dtype=np.float32)
        learner.add(x, y)
    learner.steps = 7
    learner.labels = n


class TestOnlineLearnerCheckpoint:
    def test_round_trip(self, tmp_path):
        """Checkpoint saves and loads correctly with matching config."""
        a = make_learner(symbols=("A", "B"), window=8)
        fill(a, 5)
        a.save(tmp_path, extra={"k": 1})

        # Check files exist and temp files don't
        assert (tmp_path / "model.pt").exists()
        assert (tmp_path / "replay.npz").exists()
        assert not (tmp_path / "model.pt.tmp").exists()
        assert not (tmp_path / "replay.tmp.npz").exists()

        # Load into new learner with same config
        b = make_learner(symbols=("A", "B"), window=8)
        extra = b.load(tmp_path)

        assert extra == {"k": 1}
        assert b.steps == 7
        assert b.labels == 5
        assert b.size == 5

        # Check model state matches
        for k, v in a.model.state_dict().items():
            assert torch.equal(v, b.model.state_dict()[k])

    def test_symbol_mismatch_backs_up(self, tmp_path):
        """Checkpoint with different symbols backs up and returns None."""
        a = make_learner(symbols=("A", "B"), window=8)
        fill(a, 5)
        a.save(tmp_path)

        # Load into learner with different symbols
        b = make_learner(symbols=("A", "C"), window=8)
        extra = b.load(tmp_path)

        assert extra is None
        assert (tmp_path / "model.pt.bak").exists()
        assert (tmp_path / "replay.npz.bak").exists()
        assert not (tmp_path / "model.pt").exists()
        assert not (tmp_path / "replay.npz").exists()

    def test_window_mismatch_backs_up(self, tmp_path):
        """Checkpoint with different window backs up and returns None."""
        a = make_learner(symbols=("A", "B"), window=8)
        fill(a, 5)
        a.save(tmp_path)

        # Load into learner with different window (same symbols)
        b = make_learner(symbols=("A", "B"), window=4)
        extra = b.load(tmp_path)

        assert extra is None
        assert (tmp_path / "model.pt.bak").exists()
        assert (tmp_path / "replay.npz.bak").exists()
        assert not (tmp_path / "model.pt").exists()
        assert not (tmp_path / "replay.npz").exists()

    def test_corrupt_checkpoint_starts_fresh(self, tmp_path):
        """Corrupt checkpoint returns None and does not back up."""
        a = make_learner(symbols=("A", "B"), window=8)
        fill(a, 5)
        a.save(tmp_path)

        # Corrupt the checkpoint
        (tmp_path / "model.pt").write_bytes(b"not a checkpoint")

        # Load into fresh learner
        b = make_learner(symbols=("A", "B"), window=8)
        extra = b.load(tmp_path)

        assert extra is None
        # Corrupt file should still exist, no backup created
        assert (tmp_path / "model.pt").exists()
        assert not (tmp_path / "model.pt.bak").exists()

    def test_unreadable_replay_keeps_model(self, tmp_path):
        """Unreadable replay buffer returns extra but keeps model state."""
        a = make_learner(symbols=("A", "B"), window=8)
        fill(a, 5)
        a.save(tmp_path, extra={"k": 1})

        # Corrupt replay but keep model
        (tmp_path / "replay.npz").write_bytes(b"garbage")

        # Load into fresh learner
        b = make_learner(symbols=("A", "B"), window=8)
        extra = b.load(tmp_path)

        assert extra == {"k": 1}
        assert b.steps == 7
        assert b.size == 0  # Replay was empty
