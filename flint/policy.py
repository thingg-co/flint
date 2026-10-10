"""The policy's cost and exit arithmetic, as pure functions the engine's `_suggest` and the
rebalance step call: what an hour of a position costs beyond the quoted spread (the decay of a
put, the price impact of the order), when a name is too illiquid to touch, when a straddle's
expected move beats what its premium already prices in, and when an open position has to go
(a stop, a profit target, a time limit). Nothing here touches the model.

Sources: an at-the-money option's premium is about 0.4 σ √T of spot and decays by about
premium / (2 T) per unit time (the Black-Scholes ATM approximation); market impact follows the
square-root law, cost ≈ η σ √(Q / ADV) (Almgren-Chriss, Gatheral); Amihud's illiquidity,
|return| per dollar traded, is Kyle's lambda in practice, the price move an order of a given
size buys; and a straddle's hourly profit is about ½ Γ (ΔS)² − θ Δt, so it pays only when the
move beats the implied one net of the decay.
"""
from __future__ import annotations

import math

TRADING_HOURS = 6.5
TRADING_DAYS = 252
BARS_PER_DAY = 78                                   # 5-minute bars in a regular session


def theta_bps_per_hour(iv: float | None, dte_days: float, delta: float = 0.5) -> float:
    """What an at-the-money option about `dte_days` out costs in decay per trading hour, in
    bps of the exposure it carries: premium ≈ 0.4 σ √T of spot, exposure = spot × delta per
    unit of premium's spot, decay ≈ premium / (2 × dte × 6.5) per hour."""
    if not iv or iv <= 0 or dte_days <= 0:
        return 0.0
    prem_frac = 0.4 * float(iv) * math.sqrt(float(dte_days) / 365.0)          # premium as a fraction of spot
    per_exposure = prem_frac / max(0.05, abs(float(delta)))                      # per unit of delta-equivalent exposure
    return 1e4 * per_exposure / (2.0 * float(dte_days) * TRADING_HOURS)


def implied_move_bps(iv: float | None, hours: float = 1.0) -> float:
    """The one-sigma move the implied volatility prices in over `hours` of trading, in bps."""
    if not iv or iv <= 0 or hours <= 0:
        return 0.0
    return 1e4 * float(iv) * math.sqrt(float(hours) / (TRADING_DAYS * TRADING_HOURS))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_abs_move_bps(q) -> float:
    """E|y| from a five-quantile forecast, read as a normal with median q50 and sigma from the
    10-90 spread (the folded-normal mean); the size of the move a straddle is paid on."""
    if q is None or len(q) < 5:
        return 0.0
    mu = float(q[2])
    sigma = max(1e-6, (float(q[4]) - float(q[0])) / 2.563)
    z = mu / sigma
    return sigma * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z) + mu * (1.0 - 2.0 * _phi(-z))


def dollar_volume(bars) -> float:
    """A daily dollar-volume estimate from the bars held: the mean bar's dollar volume times
    the bars in a session. Zero when nothing traded."""
    vals = [float(b.volume) * float(b.close) for b in bars if getattr(b, "volume", 0.0) and getattr(b, "close", 0.0)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) * BARS_PER_DAY


def sigma_daily(bars) -> float:
    """Daily volatility from the bars' close-to-close log returns, scaled by √(bars per day)."""
    closes = [float(b.close) for b in bars if getattr(b, "close", 0.0) and getattr(b, "volume", 0.0)]
    if len(closes) < 3:
        return 0.0
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(var, 0.0) * BARS_PER_DAY)


def impact_bps(dollars: float, adv_dollars: float, sigma_d: float, eta: float = 0.1) -> float:
    """Square-root price impact of an order of `dollars` against a name trading `adv_dollars`
    a day with daily volatility `sigma_d`, in bps: η σ √(Q / ADV)."""
    if dollars <= 0 or adv_dollars <= 0 or sigma_d <= 0 or eta <= 0:
        return 0.0
    return 1e4 * float(eta) * float(sigma_d) * math.sqrt(float(dollars) / float(adv_dollars))


def amihud(bars) -> float:
    """Amihud's illiquidity over the bars held: mean |bar return| per dollar traded, in bps per
    dollar. Zero when there is not enough to say."""
    closes = [(float(b.close), float(b.volume) * float(b.close)) for b in bars
              if getattr(b, "close", 0.0) and getattr(b, "volume", 0.0)]
    if len(closes) < 3:
        return 0.0
    vals = [abs(math.log(c1 / c0)) * 1e4 / dv1 for (c0, _), (c1, dv1) in zip(closes, closes[1:]) if c0 > 0 and dv1 > 0]
    return sum(vals) / len(vals) if vals else 0.0


def kyle_impact_bps(amihud_bps_per_dollar: float, dollars: float) -> float:
    """The price move, in bps, that an order of `dollars` buys at that illiquidity."""
    if amihud_bps_per_dollar <= 0 or dollars <= 0:
        return 0.0
    return float(amihud_bps_per_dollar) * float(dollars)


def exit_decisions(pos: dict, avg: dict, prices: dict, held_since: dict, bar_index: int, cfg) -> dict[str, str]:
    """Which open stock positions have to close now, and why: a stop (`stop_loss_bps`), a
    profit target (`take_profit_bps`) or a time limit (`max_hold_bars`); each 0 means off."""
    stop = float(getattr(cfg, "stop_loss_bps", 0.0) or 0.0)
    take = float(getattr(cfg, "take_profit_bps", 0.0) or 0.0)
    limit = int(getattr(cfg, "max_hold_bars", 0) or 0)
    out: dict[str, str] = {}
    for s, sh in pos.items():
        if sh <= 0:
            continue
        px, a = prices.get(s), avg.get(s)
        if not px or not a or px <= 0 or a <= 0:
            continue
        upnl = (float(px) / float(a) - 1.0) * 1e4
        held = bar_index - int(held_since.get(s, bar_index))
        if stop > 0 and upnl <= -stop:
            out[s] = f"stop: down {-upnl:.0f} bps from entry (limit {stop:.0f})"
        elif take > 0 and upnl >= take:
            out[s] = f"profit target: up {upnl:.0f} bps from entry (target {take:.0f})"
        elif limit > 0 and held >= limit:
            out[s] = f"time limit: held {held} bars (limit {limit})"
    return out
