"""Label cap tests: extreme labels are masked out, normal labels pass through."""
import numpy as np
import pytest

from flint.config import Config
from flint.learner import OnlineLearner


class TestLabelCap:
    """Tests for the label_cap_bps feature."""

    def _make_learner(self, n_features=10, label_cap_bps=2000.0, symbols="A,B,C"):
        """Helper to create a learner with test config."""
        cfg = Config()
        cfg.label_cap_bps = label_cap_bps
        cfg.device = "cpu"
        cfg.torch_threads = 1
        cfg.d_model = 16
        cfg.dilations = (1, 2)
        cfg.n_experts = 2
        cfg.n_heads = 2
        cfg.replay_size = 16
        cfg.window = 8
        cfg.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
        cfg.batch_size = 4
        cfg.compile = False
        # Override symbols to get exact count needed
        cfg.symbols = symbols.split(",")

        learner = OnlineLearner(cfg, n_features=n_features)
        return learner

    def test_normal_label_passes_through(self):
        """A normal label (within cap) is stored unchanged."""
        learner = self._make_learner(n_features=10, label_cap_bps=2000.0)

        n_assets = len(learner.cfg.symbols)
        x = np.zeros((n_assets, learner.cfg.window, 10), dtype=np.float32)
        # All within cap
        y = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        mask = np.ones(n_assets, dtype=np.float32)

        learner.add(x, y, mask)

        # Labels should be unchanged
        assert np.array_equal(learner.ry[0], y)
        # Mask should be all ones
        assert np.array_equal(learner.rmask[0], mask)

    def test_extreme_label_masked(self):
        """An extreme label (exceeding cap) is masked out."""
        learner = self._make_learner(n_features=10, label_cap_bps=2000.0)

        n_assets = len(learner.cfg.symbols)
        x = np.zeros((n_assets, learner.cfg.window, 10), dtype=np.float32)
        # One extreme label (3000 bps > 2000 cap)
        y = np.array([100.0, 3000.0, -500.0], dtype=np.float32)
        mask = np.ones(n_assets, dtype=np.float32)

        learner.add(x, y, mask)

        # Labels should be unchanged (cap only affects mask)
        assert np.array_equal(learner.ry[0], y)
        # Mask should be zero for the extreme label
        expected_mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        assert np.array_equal(learner.rmask[0], expected_mask)

    def test_extreme_label_negative_masked(self):
        """An extreme negative label is also masked out."""
        learner = self._make_learner(n_features=10, label_cap_bps=2000.0)

        n_assets = len(learner.cfg.symbols)
        x = np.zeros((n_assets, learner.cfg.window, 10), dtype=np.float32)
        # Extreme negative label (-2500 bps < -2000 cap)
        y = np.array([-2500.0, 100.0, 100.0], dtype=np.float32)
        mask = np.ones(n_assets, dtype=np.float32)

        learner.add(x, y, mask)

        expected_mask = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        assert np.array_equal(learner.rmask[0], expected_mask)

    def test_cap_zero_disables_rule(self):
        """cap=0 disables the rule, all labels pass through."""
        learner = self._make_learner(n_features=10, label_cap_bps=0.0)

        n_assets = len(learner.cfg.symbols)
        x = np.zeros((n_assets, learner.cfg.window, 10), dtype=np.float32)
        # Extreme labels that would be masked with a cap
        y = np.array([5000.0, -10000.0, 100.0], dtype=np.float32)
        mask = np.ones(n_assets, dtype=np.float32)

        learner.add(x, y, mask)

        # Labels and mask unchanged (cap=0 means no masking)
        assert np.array_equal(learner.ry[0], y)
        assert np.array_equal(learner.rmask[0], mask)

    def test_extreme_label_overwrites_existing_mask(self):
        """Label cap applies even when a mask already exists."""
        learner = self._make_learner(n_features=10, label_cap_bps=2000.0)

        n_assets = len(learner.cfg.symbols)
        x = np.zeros((n_assets, learner.cfg.window, 10), dtype=np.float32)
        y = np.array([100.0, 3000.0, 100.0], dtype=np.float32)
        # Existing mask has 0 for first asset, 1 for second
        mask = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        learner.add(x, y, mask)

        # Resulting mask should be AND of existing mask and label cap
        expected_mask = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert np.array_equal(learner.rmask[0], expected_mask)
