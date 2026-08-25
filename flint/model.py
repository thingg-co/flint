"""FlintNet: a small network for short-horizon return distributions across several assets.

Three ideas combined:

1. A multi-scale causal convolution stack per asset. Each dilated block widens
   the receptive field, and the last timestep of every block is read out, so the
   model sees the same history at several resolutions at once.
2. Cross-asset attention on those embeddings, so one asset can condition on what
   the others just did (lead-lag effects).
3. A regime gate: pooled market state selects a soft mixture over expert heads,
   each of which emits a monotone set of return quantiles plus a direction logit.
   Different regimes get different heads instead of one blended predictor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalBlock(nn.Module):
    """Gated residual block with a dilated causal convolution over the time axis.

    The convolution is written as a concatenation of time-shifted copies of the
    input followed by one linear layer. That is the same computation as
    nn.Conv1d with left padding, but it runs through BLAS instead of PyTorch's
    slow CPU paths for dilated 1-d convolutions (about 30x faster here).
    """

    def __init__(self, d: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.norm = nn.LayerNorm(d)
        self.taps = nn.Linear(d * kernel, 2 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, channels). Output at time t only depends on inputs at times <= t."""
        T = x.shape[1]
        pad = (self.kernel - 1) * self.dilation
        h = F.pad(self.norm(x), (0, 0, pad, 0))
        taps = [h[:, k * self.dilation:k * self.dilation + T, :] for k in range(self.kernel)]
        y = F.glu(self.taps(torch.cat(taps, dim=-1)), dim=-1)
        return x + self.drop(self.proj(y))


class FlintNet(nn.Module):
    def __init__(self, n_features: int, n_assets: int, n_quantiles: int, d_model: int = 48,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), n_experts: int = 3, n_heads: int = 4,
                 dropout: float = 0.1, kernel: int = 3):
        super().__init__()
        if n_quantiles % 2 != 1:
            raise ValueError("use an odd number of quantiles so the median is one of them")
        self.n_quantiles = n_quantiles
        self.inp = nn.Linear(n_features, d_model)
        self.asset_emb = nn.Parameter(torch.randn(n_assets, d_model) * 0.02)
        self.blocks = nn.ModuleList([CausalBlock(d_model, kernel, d, dropout) for d in dilations])
        self.scale_mix = nn.Linear(d_model * len(dilations), d_model)
        self.pre_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pre_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_quantiles + 1))
            for _ in range(n_experts)
        ])
        self.gate_noise = 0.5
        self.receptive_field = 1 + sum((kernel - 1) * d for d in dilations)

    def forward(self, x: torch.Tensor):
        """x: (batch, assets, time, features) -> quantiles (B,N,Q), logit (B,N), gate (B,K), attention (B,N,N)."""
        B, N, T, Fd = x.shape
        h = self.inp(x.reshape(B * N, T, Fd))
        h = h + self.asset_emb.unsqueeze(0).expand(B, N, -1).reshape(B * N, 1, -1)
        scales = []
        for blk in self.blocks:
            h = blk(h)
            scales.append(h[:, -1, :])
        z = self.scale_mix(torch.cat(scales, dim=-1)).view(B, N, -1)
        zn = self.pre_attn(z)
        a, w = self.attn(zn, zn, zn, need_weights=True, average_attn_weights=True)
        z = z + a
        z = z + self.ff(self.pre_ff(z))
        pooled = z.mean(1)
        spread = z.std(1) if N > 1 else torch.zeros_like(pooled)
        logits = self.gate(torch.cat([pooled, spread], dim=-1))
        if self.training:
            logits = logits + self.gate_noise * torch.randn_like(logits)  # noisy gating keeps every expert in play
        gate = torch.softmax(logits, dim=-1)
        outs = torch.stack([e(z) for e in self.experts], dim=1)          # (B, K, N, Q+1)
        out = (gate[:, :, None, None] * outs).sum(1)                      # (B, N, Q+1)
        q = self._monotone(out[..., :-1])
        return q, out[..., -1], gate, w

    def _monotone(self, raw: torch.Tensor) -> torch.Tensor:
        """Median plus positive increments outward, so quantiles never cross."""
        mid = raw.shape[-1] // 2
        m = raw[..., mid:mid + 1]
        up = m + torch.cumsum(F.softplus(raw[..., mid + 1:]), dim=-1)
        down = m - torch.flip(torch.cumsum(F.softplus(torch.flip(raw[..., :mid], [-1])), dim=-1), [-1])
        return torch.cat([down, m, up], dim=-1)


def flint_loss(q: torch.Tensor, logit: torch.Tensor, gate: torch.Tensor, y: torch.Tensor,
               quantiles: tuple[float, ...], balance: float = 0.3):
    taus = torch.tensor(quantiles, dtype=q.dtype, device=q.device)
    diff = y[..., None] - q
    pinball = torch.maximum(taus * diff, (taus - 1) * diff).mean()
    bce = F.binary_cross_entropy_with_logits(logit, (y > 0).to(q.dtype))
    usage = gate.mean(0)
    # KL(usage || uniform): keeps the gate from collapsing onto one expert early on.
    bal = (usage * torch.log(usage * gate.shape[1] + 1e-8)).sum()
    loss = pinball + bce + balance * bal
    return loss, {"pinball": pinball.item(), "bce": bce.item(), "balance": bal.item()}
