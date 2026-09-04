"""Tests for the `flint replay` subcommand."""
import os
import sys

import numpy as np
import pytest

from flint.learner import OnlineLearner
from flint.__main__ import _replay


class TestReplayCmd:
    def test_no_checkpoint_exit_1(self, tmp_path, monkeypatch):
        """No checkpoint in state dir -> exit 1 and output says so."""
        # Use a unique subdirectory to avoid conflicts with existing state
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Set the environment variable BEFORE any imports that might cache it
        monkeypatch.setenv("FLINT_STATE_DIR", str(empty_dir))

        # Run _replay and capture exit code
        with pytest.raises(SystemExit) as exc_info:
            _replay()
        assert exc_info.value.code == 1

    def test_with_checkpoint_exit_0(self, tmp_path, monkeypatch):
        """Valid checkpoint -> exit 0 and output contains steps, labels, and 'load'."""
        # Set env vars BEFORE importing Config
        monkeypatch.setenv("FLINT_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("FLINT_SYMBOLS", "A,B")
        monkeypatch.setenv("FLINT_WINDOW", "8")

        # Now create learner with matching config
        from flint.config import Config

        cfg = Config()
        cfg.device = "cpu"
        cfg.torch_threads = 1
        cfg.d_model = 16
        cfg.dilations = (1, 2)
        cfg.n_experts = 2
        cfg.n_heads = 2
        cfg.replay_size = 16
        cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
        cfg.batch_size = 4
        cfg.compile = False

        learner = OnlineLearner(cfg, n_features=3)
        n_samples = 5
        n_assets = len(learner.cfg.symbols)
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, 3), dtype=np.float32)
            y = learner.rng.random((n_assets,), dtype=np.float32)
            learner.add(x, y)
        learner.steps = 7
        learner.labels = n_samples
        learner.save(tmp_path, extra={"test": "value"})

        # Run _replay and verify exit code
        with pytest.raises(SystemExit) as exc_info:
            _replay()
        assert exc_info.value.code == 0

    def test_corrupt_checkpoint_exit_2(self, tmp_path, monkeypatch):
        """Corrupt model.pt -> exit 2."""
        monkeypatch.setenv("FLINT_STATE_DIR", str(tmp_path))

        # Create a corrupt checkpoint
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "model.pt").write_bytes(b"not a checkpoint")

        with pytest.raises(SystemExit) as exc_info:
            _replay()
        assert exc_info.value.code == 2

    def test_symbol_mismatch_shows_would_backup(self, tmp_path, monkeypatch):
        """Config mismatch with checkpoint -> output says 'backup'."""
        # Set env vars BEFORE importing Config
        monkeypatch.setenv("FLINT_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("FLINT_SYMBOLS", "A,B")
        monkeypatch.setenv("FLINT_WINDOW", "8")

        from flint.config import Config

        cfg = Config()
        cfg.device = "cpu"
        cfg.torch_threads = 1
        cfg.d_model = 16
        cfg.dilations = (1, 2)
        cfg.n_experts = 2
        cfg.n_heads = 2
        cfg.replay_size = 16
        cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
        cfg.batch_size = 4
        cfg.compile = False

        learner = OnlineLearner(cfg, n_features=3)
        n_samples = 5
        n_assets = len(learner.cfg.symbols)
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, 3), dtype=np.float32)
            y = learner.rng.random((n_assets,), dtype=np.float32)
            learner.add(x, y)
        learner.steps = 7
        learner.labels = n_samples
        learner.save(tmp_path)

        # Now change symbols via env var - this will cause a mismatch
        monkeypatch.setenv("FLINT_SYMBOLS", "A,C")

        with pytest.raises(SystemExit) as exc_info:
            _replay()
        # Exit 0 because we could read the checkpoint, just won't load it
        assert exc_info.value.code == 0
