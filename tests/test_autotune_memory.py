import pytest
from unittest.mock import patch
from flint.autotune import free_memory_gb, _check_start_memory, NotEnoughMemory


def test_free_memory_gb_cuda_unified_memory(monkeypatch):
    """Test free_memory_gb on unified memory (CUDA) returns host MemAvailable."""
    # Mock /proc/meminfo read
    meminfo_content = """MemTotal:       127600588 kB
MemAvailable:    34412768 kB
"""
    def mock_open(*args, **kwargs):
        if args[0] == '/proc/meminfo':
            from io import StringIO
            return StringIO(meminfo_content)
        # For other files, use real open (though we don't expect any)
        return open(*args, **kwargs)

    monkeypatch.setattr('builtins.open', mock_open)

    # Mock torch.cuda.mem_get_info to return (0.94e9, 130.66e9)
    with patch('torch.cuda.mem_get_info', return_value=(0.94e9, 130.66e9)):
        # Mock _host_total_gb to return a value that makes unified detection true
        # We want abs(total/1e9 - _host_total_gb()) < 0.15 * total/1e9
        # Let's set _host_total_gb to return 127.6 (matching MemTotal)
        with patch('flint.autotune._host_total_gb', return_value=127.6):
            result = free_memory_gb("cuda")
            # Should return host MemAvailable in GB: 34412768 / 1e6 = 34.412768 GB
            assert result == pytest.approx(34.412768, rel=1e-5)


def test_free_memory_gb_cuda_discrete_gpu(monkeypatch):
    """Test free_memory_gb on discrete GPU returns CUDA free memory."""
    # Mock /proc/meminfo read (same as before)
    meminfo_content = """MemTotal:       127600588 kB
MemAvailable:    34412768 kB
"""
    def mock_open(*args, **kwargs):
        if args[0] == '/proc/meminfo':
            from io import StringIO
            return StringIO(meminfo_content)
        return open(*args, **kwargs)

    monkeypatch.setattr('builtins.open', mock_open)

    # Mock torch.cuda.mem_get_info to return (20e9, 24e9) for discrete GPU
    with patch('torch.cuda.mem_get_info', return_value=(20e9, 24e9)):
        # Mock _host_total_gb to return a value that makes unified detection false
        # We want abs(total/1e9 - _host_total_gb()) >= 0.15 * total/1e9
        # total is 24e9 -> 24 GB, 0.15*24 = 3.6
        # So if _host_total_gb returns something like 100, then |24-100|=76 > 3.6 -> not unified
        with patch('flint.autotune._host_total_gb', return_value=100.0):
            result = free_memory_gb("cuda")
            # Should return CUDA free memory: 20e9 / 1e9 = 20.0 GB
            assert result == pytest.approx(20.0, rel=1e-5)


def test_free_memory_gb_cpu_unreadable_meminfo(monkeypatch):
    """Test free_memory_gb returns None for CPU when /proc/meminfo is unreadable."""
    # Mock open to raise OSError for /proc/meminfo
    def mock_open(*args, **kwargs):
        if args[0] == '/proc/meminfo':
            raise OSError("Cannot read /proc/meminfo")
        return open(*args, **kwargs)

    monkeypatch.setattr('builtins.open', mock_open)

    # For CPU, we don't need to mock anything else
    result = free_memory_gb("cpu")
    assert result is None


def test_check_start_memory_raises_when_not_enough(monkeypatch):
    """Test _check_start_memory raises NotEnoughMemory when free memory is insufficient."""
    # Create a fake cfg object
    class Cfg:
        replay_size = 1000
        n_assets = 10
        state_dir = "/tmp"

    cfg = Cfg()
    n_features = 5

    # Choice dict with peak_gb, window, device, preset
    choice = {
        "peak_gb": 10.0,  # 10 GB peak
        "window": 64,
        "device": "cuda",
        "preset": "XL"
    }

    # Calculate what free memory would be needed:
    # replay = cfg.replay_size * cfg.n_assets * choice["window"] * n_features * 4 / 1e9
    #        = 1000 * 10 * 64 * 5 * 4 / 1e9 = 0.0128 GB
    # need = peak * 1.25 + replay + MEMORY_HEADROOM_GB
    #      = 10.0 * 1.25 + 0.0128 + 3.0 = 12.5 + 0.0128 + 3.0 = 15.5128 GB
    # So if free memory is less than 15.5128 GB, it should raise

    # Patch free_memory_gb to return a low value (e.g., 15.0 GB)
    with patch('flint.autotune.free_memory_gb', return_value=15.0):
        # Patch other_gpu_tenants to return empty list
        with patch('flint.autotune.other_gpu_tenants', return_value=[]):
            with pytest.raises(NotEnoughMemory) as exc_info:
                _check_start_memory(cfg, n_features, choice, say=lambda *a, **k: None)

            # Check that the message mentions the preset
            assert "start preset XL" in str(exc_info.value)


def test_check_start_memory_does_not_raise_when_enough(monkeypatch):
    """Test _check_start_memory does not raise when there is enough memory."""
    # Create a fake cfg object
    class Cfg:
        replay_size = 1000
        n_assets = 10
        state_dir = "/tmp"

    cfg = Cfg()
    n_features = 5

    choice = {
        "peak_gb": 10.0,
        "window": 64,
        "device": "cuda",
        "preset": "XL"
    }

    # With the same calculation as above, need is 15.5 GB
    # Patch free_memory_gb to return a high value (e.g., 30.0 GB)
    with patch('flint.autotune.free_memory_gb', return_value=30.0):
        # Patch other_gpu_tenants to return empty list
        with patch('flint.autotune.other_gpu_tenants', return_value=[]):
            # Should not raise
            _check_start_memory(cfg, n_features, choice, say=lambda *a, **k: None)


def test_check_start_memory_is_noop_when_no_peak_gb(monkeypatch):
    """Test _check_start_memory is a no-op when choice has no peak_gb."""
    # Create a fake cfg object
    class Cfg:
        replay_size = 1000
        n_assets = 10
        state_dir = "/tmp"

    cfg = Cfg()
    n_features = 5

    # Choice without peak_gb (or with 0 or None)
    choice = {
        "peak_gb": 0.0,  # This should cause early return
        "window": 64,
        "device": "cuda",
        "preset": "XL"
    }

    # Should not raise regardless of free memory
    with patch('flint.autotune.free_memory_gb', return_value=0.0):
        with patch('flint.autotune.other_gpu_tenants', return_value=[]):
            # Should not raise
            _check_start_memory(cfg, n_features, choice, say=lambda *a, **k: None)


def test_check_start_memory_message_names_tenant(monkeypatch):
    """Test NotEnoughMemory message names the tenant when other_gpu_tenants returns a tenant."""
    # Create a fake cfg object
    class Cfg:
        replay_size = 1000
        n_assets = 10
        state_dir = "/tmp"

    cfg = Cfg()
    n_features = 5

    choice = {
        "peak_gb": 10.0,
        "window": 64,
        "device": "cuda",
        "preset": "XL"
    }

    # Set free memory low enough to trigger the error
    with patch('flint.autotune.free_memory_gb', return_value=15.0):
        # Patch other_gpu_tenants to return a tenant
        with patch('flint.autotune.other_gpu_tenants', return_value=["llama-server@super.service"]):
            with pytest.raises(NotEnoughMemory) as exc_info:
                _check_start_memory(cfg, n_features, choice, say=lambda *a, **k: None)

            msg = str(exc_info.value)
            # Check that the message includes the tenant
            assert "llama-server@super.service" in msg
            assert "Running on the same memory:" in msg


def test_ladder_refuses_when_memory_short(monkeypatch, tmp_path):
    """Test that autotune ladder refuses before benchmarking when memory is short."""
    from flint.autotune import autotune, NotEnoughMemory

    # Patch free_memory_gb to return 2.0 GB (very low)
    monkeypatch.setattr("flint.autotune.free_memory_gb", lambda d: 2.0)

    # Patch _bench so the test fails if it's ever called
    def fail_bench(*args, **kwargs):
        raise RuntimeError("_bench should not be called when memory is insufficient")

    monkeypatch.setattr("flint.autotune._bench", fail_bench)

    # Patch _best_threads to return 1
    monkeypatch.setattr("flint.autotune._best_threads", lambda *args, **kwargs: 1)

    # Patch pick_device to return "cpu"
    monkeypatch.setattr("flint.autotune.pick_device", lambda *args: "cpu")

    # Patch other_gpu_tenants to return empty list
    monkeypatch.setattr("flint.autotune.other_gpu_tenants", lambda: [])

    # Build a cfg object with the attributes autotune() reads
    class Cfg:
        state_dir = str(tmp_path)
        n_assets = 10
        bar_seconds = 300
        batch_size = 16
        autotune_util = 0.5
        max_warmup_seconds = 300
        warmup_steps = 10
        steps_per_label = 1
        n_quantiles = 5
        device = "cpu"

    cfg = Cfg()

    # Assert autotune(cfg, 5) raises NotEnoughMemory
    with pytest.raises(NotEnoughMemory):
        autotune(cfg, 5)