"""The training target: a name's hour-ahead log return, or its market-relative part.

Over one hour the index dominates every name's move, and nothing in a five-minute history
predicts the index, so a net trained on raw returns spends its capacity on noise it cannot
resolve. The market-relative abnormal return, the part of the return the indices do not
explain, is where a nonlinear cross-sectional model earns its keep (Gu, Kelly and Xiu 2020).

"The market" is a set of index names the universe already models (SPY, QQQ, IWM, SMH, XLF by
default; any that are not modeled are ignored): each name's label is the residual of a ridge
regression of its return on the indices' returns over the same hour, so a semiconductor name
loads on SMH and a small cap on IWM instead of everything being measured against SPY alone.
The prior is one on the first index and zero on the rest, with `shrink` pseudo-observations
behind it, so a name with little history is assumed to move with the broad market.

The replay keeps absolute labels. The transform is applied where labels are consumed, in the
training step, the held-out validation, the live scoring and the backtest, so the target is a
run-time switch (`FLINT_TARGET=relative|absolute`) and one replay serves both. Betas refit
once a session from the labels held; the backtest refits them from the days already seen, so
no day's own labels shape the betas that score it. The index names themselves keep absolute
labels (zero rows), so the net still forecasts the market and the policy can read it.

A relative forecast says "beats its indices", so a book built on it has to be hedged or it
carries index risk the model said nothing about: `hedge_targets` sizes, per index, the
position that cancels the basket's loading on it (an index put under a long basket, index
stock over a bearish one, both defined-risk).
"""
from __future__ import annotations

import numpy as np

BETA_LO, BETA_HI = -2.0, 4.0


def index_columns(symbols, index_symbols) -> list[int]:
    """The columns of the configured indices that are modeled, in the configured order and
    without repeats; empty when none is (the target then degrades to absolute)."""
    if isinstance(index_symbols, str):
        index_symbols = index_symbols.split(",")
    syms = [str(s).strip().upper() for s in symbols]
    out: list[int] = []
    for name in index_symbols:
        name = str(name).strip().upper()
        if name and name in syms and syms.index(name) not in out:
            out.append(syms.index(name))
    return out


def prior_beta(n_assets: int, cols: list[int]) -> np.ndarray:
    """One on the first index, zero elsewhere; index rows all zero."""
    beta = np.zeros((n_assets, len(cols)), dtype=np.float32)
    if cols:
        beta[:, 0] = 1.0
        beta[cols, :] = 0.0
    return beta


def estimate_beta(y: np.ndarray, mask: np.ndarray | None, cols: list[int], *, recent: int = 6000,
                  shrink: float = 400.0) -> np.ndarray:
    """Loadings per name on the index columns, (n_assets, k), from the last `recent` label
    rows: a ridge regression per name over the rows where it and every index traded, pulled
    toward the prior by `shrink` pseudo-observations of the first index's variance. Rows are
    in buffer order, oldest first. Index rows are zero so their labels stay absolute."""
    y = np.asarray(y, dtype=np.float64)
    n_assets = y.shape[-1] if y.ndim else 0
    cols = [int(c) for c in cols if 0 <= int(c) < n_assets]
    beta = prior_beta(n_assets, cols)
    if not cols or y.ndim != 2 or y.shape[0] == 0:
        return beta
    y = y[-int(recent):] if recent and recent > 0 else y
    m = np.ones_like(y) if mask is None else np.asarray(mask, dtype=np.float64)[-y.shape[0]:]
    w = m * np.prod(m[:, cols], axis=1, keepdims=True)         # the name and every index traded
    n = w.sum(axis=0)
    ok = n > len(cols) + 1
    if not ok.any():
        return beta
    x = y[:, cols]                                               # rows × k
    k = len(cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_x = (w[:, :, None] * x[:, None, :]).sum(axis=0) / np.maximum(n, 1)[:, None]      # n × k, per-name means of the indices
        mu_y = (w * y).sum(axis=0) / np.maximum(n, 1)
        dx = x[:, None, :] - mu_x[None, :, :]                   # rows × n × k
        dy = y - mu_y[None, :]                                   # rows × n
        xtx = np.einsum("rn,rnk,rnl->nkl", w, dx, dx)           # n × k × k
        xty = np.einsum("rn,rnk,rn->nk", w, dx, dy)             # n × k
        var1 = np.maximum(xtx[:, 0, 0] / np.maximum(n, 1), 1e-9)  # the first index's variance per name
        lam = float(shrink) * var1                               # the prior's weight in observations of that variance
        prior = prior_beta(n_assets, cols).astype(np.float64)
        a = xtx + lam[:, None, None] * np.eye(k)[None]
        b = xty + lam[:, None] * prior
        est = np.linalg.solve(a, b[:, :, None])[:, :, 0]
    est = np.clip(np.nan_to_num(est, nan=0.0), BETA_LO, BETA_HI)
    beta[ok] = est[ok].astype(np.float32)
    beta[cols, :] = 0.0
    return beta


def relative(y, beta: np.ndarray | None, cols: list[int]):
    """y minus the indices' labels weighted by each name's loadings, along the last axis;
    identity when the target is absolute (no columns or no beta). Works on numpy arrays and
    torch tensors alike."""
    if not cols or beta is None:
        return y
    cols = list(cols)
    if isinstance(y, np.ndarray):
        b = np.asarray(beta, dtype=y.dtype)
        return y - y[..., cols] @ b.T
    import torch  # noqa: PLC0415 -- torch is heavy; this module is imported by tests that never need it
    b = torch.as_tensor(np.asarray(beta), dtype=y.dtype, device=y.device)
    return y - y[..., cols] @ b.T


def hedge_targets(targets: dict, beta: np.ndarray, symbols, cols: list[int]) -> dict[str, float]:
    """Per index, the weight that cancels the book's loading on it: a long basket with net
    loading E on an index gets -E of it (a put, defined risk); a bearish one gets +|E| of
    index stock. A put target is already a negative weight, so it enters with its sign."""
    if not cols:
        return {}
    symbols = list(symbols)
    exposure = np.zeros(len(cols))
    for i, s in enumerate(symbols):
        if i in cols:
            continue
        w = float(targets.get(s, 0.0) or 0.0)
        if w:
            exposure += w * np.asarray(beta[i], dtype=np.float64)
    return {symbols[c]: float(-exposure[j]) for j, c in enumerate(cols)}
