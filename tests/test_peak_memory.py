import pytest

import torch
from flint.autotune import PeakMemory


class TestPeakMemory:
    """Tests for PeakMemory class."""

    def test_mps_samples_and_tracks_peak(self, monkeypatch):
        """On MPS, sample() tracks the maximum allocation."""
        # Mock torch.mps.driver_allocated_memory to return specific values
        calls = []

        def mock_driver_allocated_memory():
            # Return 1e9, 5e9, 2e9 on successive calls
            values = [1e9, 5e9, 2e9]
            result = values[len(calls)]
            calls.append(result)
            return result

        monkeypatch.setattr(torch.mps, "driver_allocated_memory", mock_driver_allocated_memory)

        pk = PeakMemory("mps")

        # After init, peak is 0.0 (reset was called)
        assert pk.peak == 0.0

        # First sample: 1e9 bytes = 1 GB
        pk.sample()
        assert pk.peak == 1.0

        # Second sample: 5e9 bytes = 5 GB (new peak)
        pk.sample()
        assert pk.peak == 5.0

        # Third sample: 2e9 bytes = 2 GB (lower, peak unchanged)
        pk.sample()
        assert pk.peak == 5.0

        # gb() should return the peak
        assert pk.gb() == pytest.approx(5.0)

    def test_mps_reset_clears_peak(self, monkeypatch):
        """On MPS, reset() clears the peak back to 0."""
        # Mock torch.mps.driver_allocated_memory to return 5e9
        def mock_driver_allocated_memory():
            return 5e9

        monkeypatch.setattr(torch.mps, "driver_allocated_memory", mock_driver_allocated_memory)

        pk = PeakMemory("mps")

        # Set peak to 5.0
        pk.sample()
        assert pk.peak == 5.0

        # Reset should clear it
        pk.reset()
        assert pk.peak == 0.0

        # gb() should return 0.0 after reset
        assert pk.gb() == 0.0

    def test_cpu_is_harmless(self, monkeypatch):
        """On CPU, sample() is harmless and gb() returns 0.0."""
        # Make sure mps module is not available
        monkeypatch.setitem(torch.__dict__, "mps", None)

        pk = PeakMemory("cpu")

        # sample() should do nothing (no-op)
        pk.sample()
        assert pk.peak == 0.0

        # gb() should return 0.0
        assert pk.gb() == 0.0

        # reset() should also be no-op
        pk.reset()
        assert pk.peak == 0.0
