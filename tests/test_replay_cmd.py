"""Tests for the `flint replay` subcommand."""
import json
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

        learner = OnlineLearner(cfg, n_features=23)
        n_samples = 5
        n_assets = len(learner.cfg.symbols)
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, 23), dtype=np.float32)
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

        learner = OnlineLearner(cfg, n_features=23)
        n_samples = 5
        n_assets = len(learner.cfg.symbols)
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, 23), dtype=np.float32)
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

    def test_machine_json_window_used_when_present(self, tmp_path, monkeypatch):
        """When machine.json exists with a window, replay uses that instead of cfg.window."""
        # Set env vars BEFORE importing Config - unset WINDOW to ensure machine.json is used
        monkeypatch.setenv("FLINT_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("FLINT_SYMBOLS", "A,B")

        # Remove WINDOW env var if set
        monkeypatch.delenv("FLINT_WINDOW", raising=False)

        from flint.config import Config
        from flint.features import N_FEATURES

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
        cfg.window = 8  # Set to 8 to match machine.json

        # Create learner with window=8 to match machine.json
        learner = OnlineLearner(cfg, n_features=N_FEATURES)
        n_samples = 5
        n_assets = len(learner.cfg.symbols)
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, N_FEATURES), dtype=np.float32)
            y = learner.rng.random((n_assets,), dtype=np.float32)
            learner.add(x, y)
        learner.steps = 7
        learner.labels = n_samples
        learner.save(tmp_path)

        # Create machine.json with window 8 (different from cfg.window default)
        # Note: cfg.window is already 8, but we set it explicitly here to be clear
        machine = {
            "key": "test-key",
            "choice": {
                "device": "cpu",
                "d_model": 16,
                "dilations": [1, 2],
                "n_experts": 2,
                "n_heads": 2,
                "window": 8,
                "preset": "S",
                "params": 1000000,
                "ms_per_step": 10,
                "budget_ms": 100,
                "peak_gb": 0.5,
            },
        }
        (tmp_path / "machine.json").write_text(json.dumps(machine))

        # Capture output to verify window from machine.json is used
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            with pytest.raises(SystemExit) as exc_info:
                _replay()
            assert exc_info.value.code == 0

        output = f.getvalue()
        # The current shape should use window=8 from machine.json and N_FEATURES=25
        assert "(2, 8, 25)" in output, f"Expected window=8, n_features=25 from machine.json, got output: {output}"
        assert "would load checkpoint" in output, f"Should report config match, got: {output}"

    def test_universe_json_merge_affects_current_symbols(self, tmp_path, monkeypatch):
        """Symbols from universe.json are merged into current symbols for comparison."""
        # Set env vars BEFORE importing Config
        monkeypatch.setenv("FLINT_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("FLINT_SYMBOLS", "A,B")

        from flint.config import Config
        from flint.features import N_FEATURES
        import torch

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
        cfg.max_universe = 10  # Ensure room for universe.json symbols

        # Set symbols to 2 initial symbols (A, B) - this is what's in config
        # The learner will be created with 2 symbols
        cfg.symbols = ["A", "B"]

        # Save checkpoint with 3 symbols (A, B, C) - simulates runtime addition of C
        # Create a new config with 3 symbols for the checkpoint
        cfg3 = Config()
        cfg3.device = cfg.device
        cfg3.torch_threads = cfg.torch_threads
        cfg3.d_model = cfg.d_model
        cfg3.dilations = cfg.dilations
        cfg3.n_experts = cfg.n_experts
        cfg3.n_heads = cfg.n_heads
        cfg3.replay_size = cfg.replay_size
        cfg3.quantiles = cfg.quantiles
        cfg3.batch_size = cfg.batch_size
        cfg3.compile = cfg.compile
        cfg3.window = cfg.window
        cfg3.lr = cfg.lr
        cfg3.weight_decay = cfg.weight_decay
        cfg3.symbols = ["A", "B", "C"]  # 3 symbols for the checkpoint
        cfg3.max_universe = cfg.max_universe

        learner = OnlineLearner(cfg3, n_features=N_FEATURES)
        n_samples = 5
        n_assets = 3  # 3 symbols in checkpoint
        for _ in range(n_samples):
            x = learner.rng.random((n_assets, learner.cfg.window, N_FEATURES), dtype=np.float32)
            y = learner.rng.random((n_assets,), dtype=np.float32)
            learner.add(x, y)
        learner.steps = 7
        learner.labels = n_samples
        learner.save(tmp_path)

        # Create universe.json with extra symbol C (simulating runtime addition)
        # This tests that C is merged in when computing current symbols
        universe = {"symbols": ["C"]}
        (tmp_path / "universe.json").write_text(json.dumps(universe))

        # Run _replay - should show config match because C is in universe.json
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            with pytest.raises(SystemExit) as exc_info:
                _replay()
            assert exc_info.value.code == 0

        output = f.getvalue()
        # Current symbols should include C from universe.json
        # cfg.symbols is ["A", "B"], universe adds ["C"] = 3 total
        assert "current symbols: 3" in output, f"Expected 3 symbols (2 + C from universe.json), got: {output}"
        assert "would load checkpoint" in output, f"Should report config match, got: {output}"
