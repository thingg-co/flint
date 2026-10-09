"""A torch.compile that fails on its first train step costs speed, not the engine."""
from types import SimpleNamespace

import numpy as np
import pytest

from flint.config import Config
from flint.learner import OnlineLearner


def make_learner():
    cfg = Config()
    cfg.symbols = ["A", "B"]
    cfg.device = "cpu"
    cfg.torch_threads = 1
    cfg.d_model = 16
    cfg.dilations = (1, 2)
    cfg.n_experts = 2
    cfg.n_heads = 2
    cfg.window = 8
    cfg.replay_size = 16
    cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    cfg.batch_size = 4
    cfg.compile = False
    learner = OnlineLearner(cfg, n_features=3)
    for _ in range(12):
        learner.add(learner.rng.random((2, cfg.window, 3), dtype=np.float32),
                    learner.rng.random((2,), dtype=np.float32))
    return learner


class Broken:
    """Stands in for torch.compile's OptimizedModule when the backend build fails on the
    first forward (the lazy compile is what raised on the Spark: Triton could not build its
    CUDA helper without Python.h)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        raise RuntimeError("cuda_utils.c:9:10: fatal error: Python.h: No such file or directory")


def test_first_compiled_step_failure_falls_back_to_eager(caplog):
    learner = make_learner()
    broken = Broken()
    learner.train_model = broken
    with caplog.at_level("WARNING", logger="flint.learner"):
        out = learner.train_steps(2)
    assert out is not None and out["steps"] == 2 and out["loss"] is not None   # both steps trained
    assert broken.calls == 1                                                   # tried once, then eager
    assert learner.train_model is learner.model
    assert learner.compile_broken
    assert "training eager" in caplog.text and "Python.h" in caplog.text        # the cause is in the log


def test_eager_failure_still_raises():
    learner = make_learner()
    assert learner.train_model is learner.model

    def boom(x):
        raise RuntimeError("real bug")

    learner.model.forward = boom
    with pytest.raises(RuntimeError, match="real bug"):
        learner.train_steps(1)


def test_reset_does_not_recompile_after_a_failure(monkeypatch):
    learner = make_learner()
    learner.cfg.compile = True
    learner.device = SimpleNamespace(type="cuda")          # the only case _compiled would wrap
    learner.compile_broken = True
    import torch
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("compile retried after a failure"))
    assert learner._compiled(learner.model) is learner.model
