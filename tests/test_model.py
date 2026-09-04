"""Contract tests for FlintNet and flint_loss."""
from __future__ import annotations

import torch
import pytest
from flint.model import FlintNet, flint_loss
from flint.autotune import _bench, PRESETS

torch.manual_seed(0)
torch.set_num_threads(1)


def make_net():
    """Create a small FlintNet for testing."""
    return FlintNet(
        n_features=3, n_assets=2, n_quantiles=5,
        d_model=16, dilations=(1, 2), n_experts=2, n_heads=2, dropout=0.0
    ).eval()


def make_x(batch=4):
    """Create input tensor."""
    return torch.randn(batch, 2, 8, 3)


QUANT = (1/6, 2/6, 3/6, 4/6, 5/6)


def test_forward_returns_five_tensors_with_expected_shapes():
    """Forward pass returns five tensors with expected shapes."""
    net = make_net()
    x = make_x(batch=4)
    out = net(x)

    assert len(out) == 5
    q, up, down, gate, attn = out

    assert q.shape == (4, 2, 5)
    assert up.shape == (4, 2)
    assert down.shape == (4, 2)
    # gate: (batch, n_experts) = (4, 2)
    assert gate.shape == (4, 2)
    # attn: (batch, assets, assets) = (4, 2, 2)
    assert attn.shape == (4, 2, 2)


def test_quantiles_never_cross():
    """Quantiles are monotonically non-decreasing across quantile dimension."""
    net = make_net()
    for seed in range(5):
        torch.manual_seed(seed)
        for scale in (1, 10, 100):
            x = make_x(batch=4) * scale
            q = net(x)[0]
            assert torch.all(q[..., 1:] >= q[..., :-1])


def test_gate_and_attention_are_distributions():
    """Gate and attention are valid probability distributions."""
    net = make_net()
    x = make_x(batch=4)
    out = net(x)
    gate, attn = out[3], out[4]

    # Gate: sum over experts should be 1, all non-negative
    assert torch.allclose(gate.sum(-1), torch.ones_like(gate.sum(-1)), atol=1e-5)
    assert torch.all(gate >= 0)

    # Attention: sum over last dimension should be 1, all non-negative
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    assert torch.all(attn >= 0)


def test_loss_is_finite_scalar_with_parts():
    """Loss is finite scalar with pinball, bce, and balance parts."""
    net = make_net()
    x = make_x(batch=4)
    q, up, down, gate, _ = net(x)
    y = torch.randn(4, 2) * 50

    loss, parts = flint_loss(q, up, down, gate, y, QUANT)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"pinball", "bce", "balance"}
    assert all(torch.isfinite(torch.tensor(v)) for v in parts.values())


def test_loss_mask_drops_an_asset():
    """Masked assets do not contribute to the loss."""
    net = make_net()
    x = make_x(batch=4)
    q, up, down, gate, _ = net(x)
    y = torch.randn(4, 2) * 50

    # Base loss with full mask
    mask = torch.tensor([[1.0, 1.0]] * 4)
    loss_full, _ = flint_loss(q, up, down, gate, y, QUANT, mask=mask)

    # Loss with first asset masked out
    mask = torch.tensor([[1.0, 0.0]] * 4)
    loss_masked, _ = flint_loss(q, up, down, gate, y, QUANT, mask=mask)

    # Modify y for masked asset - loss should be unchanged
    y_altered = y.clone()
    y_altered[:, 1] += 1e4
    loss_altered, _ = flint_loss(q, up, down, gate, y_altered, QUANT, mask=mask)

    assert torch.isfinite(loss_masked)
    # Masked asset should not affect loss
    assert torch.allclose(loss_masked, loss_altered, rtol=1e-6)


def test_gradients_reach_every_parameter():
    """Gradients flow to all parameters with requires_grad."""
    net = make_net()
    net.train()
    x = make_x(batch=4)
    q, up, down, gate, _ = net(x)
    y = torch.randn(4, 2) * 50
    loss, _ = flint_loss(q, up, down, gate, y, QUANT)

    loss.backward()

    for name, param in net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no grad"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


def test_pinball_prefers_the_right_quantile():
    """Pinball loss correctly prefers quantiles closer to the actual value."""
    # Single asset, y=0, compare q1 (centered around 0) vs q2 (far above)
    q1 = torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]])
    q2 = torch.tensor([[[5.0, 6.0, 7.0, 8.0, 9.0]]])
    y = torch.tensor([[0.0]])

    # Zero logits and uniform gate to isolate pinball - need (batch, n_assets) = (1, 1)
    zeros = torch.zeros(1, 1)
    uniform_gate = torch.ones(1, 1) / 2

    # Pinball for q1
    loss1, parts1 = flint_loss(q1, zeros, zeros, uniform_gate, y, QUANT)
    # Pinball for q2
    loss2, parts2 = flint_loss(q2, zeros, zeros, uniform_gate, y, QUANT)

    assert parts1["pinball"] < parts2["pinball"]


def test_bench_contract():
    """_bench returns ms, n_params, peak with expected types."""
    ms, n, peak = _bench("cpu", PRESETS[0], n_assets=2, n_features=3, batch=2, n_quant=5, iters=1)

    assert ms > 0
    assert n > 0
    assert peak == 0.0  # CPU has no peak tracking
