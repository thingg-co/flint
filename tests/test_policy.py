"""The policy's cost and exit arithmetic (flint/policy.py) and its gates in Engine._suggest:
put decay and price impact in the cost edge, the Amihud illiquidity gate, the straddle's
gamma breakeven, and the stop / target / time-limit exits."""
import collections
import math

import numpy as np

from flint import policy
from flint.bars import Bar
from flint.engine import Engine
from test_gates import make_fake, make_pred


def _bar(close, volume=1000.0, ts=1_800_000_000.0):
    return Bar(ts=ts, open=close, high=close * 1.001, low=close * 0.999, close=close, volume=volume,
               buy_volume=volume * 0.6, sell_volume=volume * 0.4, trades=15, spread_bps=5.0)


def _bars(n=80, volume=1000.0, step=0.0057, start=200.0):
    """Closes alternating up and down by `step` log units: a known daily volatility."""
    out = collections.deque()
    px = start
    for k in range(n):
        px = px * math.exp(step if k % 2 == 0 else -step)
        out.append(_bar(px, volume, ts=1_800_000_000.0 + 300 * k))
    return out


def test_decay_implied_move_and_expected_move():
    assert abs(policy.theta_bps_per_hour(0.3, 35) - 1.63) < 0.05        # about 1.6 bps of exposure per hour
    assert policy.theta_bps_per_hour(None, 35) == 0.0 and policy.theta_bps_per_hour(0.3, 0) == 0.0
    assert abs(policy.implied_move_bps(0.3, 1.0) - 74.1) < 0.5           # one sigma over one hour
    assert policy.implied_move_bps(0.0, 1.0) == 0.0
    sym = policy.expected_abs_move_bps([-150, -60, 0, 60, 150])
    assert abs(sym - 117.05 * math.sqrt(2 / math.pi)) < 0.5             # folded normal, zero median
    assert abs(policy.expected_abs_move_bps([99, 99.5, 100, 100.5, 101]) - 100.0) < 0.1   # a sure move is its size
    assert policy.expected_abs_move_bps(None) == 0.0


def test_volume_volatility_impact_and_illiquidity_from_bars():
    bars = _bars()
    adv = policy.dollar_volume(bars)
    assert abs(adv / (1000.0 * 200.0 * 78) - 1.0) < 0.02                # mean bar dollar volume times a session
    assert abs(policy.sigma_daily(bars) - 0.0057 * math.sqrt(78)) < 0.002
    assert policy.dollar_volume([]) == 0.0 and policy.sigma_daily(list(bars)[:2]) == 0.0
    assert abs(policy.impact_bps(10_000.0, 5_000_000.0, 0.04, 0.1) - 1.79) < 0.02   # eta sigma sqrt(Q/ADV)
    assert policy.impact_bps(10_000.0, 0.0, 0.04) == 0.0
    ami = policy.amihud(bars)
    assert abs(ami - 57.0 / (1000.0 * 200.0)) < 0.02 * ami              # ~57 bps per bar over ~$200k per bar
    assert abs(policy.kyle_impact_bps(ami, 20_000.0) - 5.7) < 0.3        # a $20k order moves it ~5.7 bps
    assert policy.amihud(list(bars)[:2]) == 0.0


def test_exit_decisions_stop_target_and_time_limit():
    cfg = type("C", (), {"stop_loss_bps": 200.0, "take_profit_bps": 300.0, "max_hold_bars": 24})()
    pos = {"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0, "P": 0.0}
    avg = {"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0}
    prices = {"A": 97.5, "B": 103.5, "C": 100.5, "D": 100.5}
    held = {"A": 90, "B": 90, "C": 70, "D": 95}
    out = policy.exit_decisions(pos, avg, prices, held, 100, cfg)
    assert out["A"].startswith("stop: down 250 bps") and out["B"].startswith("profit target: up 350 bps")
    assert out["C"].startswith("time limit: held 30 bars") and "D" not in out and "P" not in out
    off = type("C", (), {"stop_loss_bps": 0.0, "take_profit_bps": 0.0, "max_hold_bars": 0})()
    assert policy.exit_decisions(pos, avg, prices, held, 100, off) == {}


def test_impact_is_charged_against_the_edge():
    # a thin edge (12 bps over a 10 bps spread) survives on a liquid name and dies on an illiquid one
    liquid = make_fake(bars={"AAPL": _bars(volume=1_000_000.0)}, cfg={"confirm_bars": 1})
    r = Engine._suggest(liquid, 0, "AAPL", make_pred([-10, 0, 12, 24, 40], 0.7, 0.2))
    assert r["action"] == "BUY", r["why"]
    thin = make_fake(bars={"AAPL": _bars(volume=100.0)}, cfg={"confirm_bars": 1})
    r = Engine._suggest(thin, 0, "AAPL", make_pred([-10, 0, 12, 24, 40], 0.7, 0.2))
    assert r["action"] == "HOLD" and "bps of impact" in r["why"], r["why"]


def test_illiquid_names_do_not_trade():
    fake = make_fake(bars={"AAPL": _bars(volume=2.0)}, cfg={"confirm_bars": 1})
    r = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
    assert r["action"] == "HOLD" and "too illiquid" in r["why"], r["why"]


def test_put_decay_is_charged_against_a_bearish_edge_unless_switched_off():
    q = [-40, -24, -12, 0, 10]
    on = make_fake(bars={"AAPL": _bars(volume=1_000_000.0)}, option_iv={"AAPL": 0.5}, cfg={"confirm_bars": 1})
    r = Engine._suggest(on, 0, "AAPL", make_pred(q, 0.2, 0.7))
    assert r["action"] == "HOLD" and "put decay" in r["why"], r["why"]
    off = make_fake(bars={"AAPL": _bars(volume=1_000_000.0)}, option_iv={"AAPL": 0.5}, cfg={"confirm_bars": 1, "theta_in_edge": False})
    r = Engine._suggest(off, 0, "AAPL", make_pred(q, 0.2, 0.7))
    assert r["action"] == "SELL", r["why"]
    long_ = make_fake(bars={"AAPL": _bars(volume=1_000_000.0)}, option_iv={"AAPL": 0.5}, cfg={"confirm_bars": 1})
    r = Engine._suggest(long_, 0, "AAPL", make_pred([-10, 0, 12, 24, 40], 0.7, 0.2))
    assert r["action"] == "BUY", r["why"]                               # stock pays no decay


def test_straddle_needs_the_move_to_beat_the_implied_one_when_iv_is_known():
    wide, narrow = [-150, -60, 0, 60, 150], [-80, -30, 0, 30, 80]
    fake = make_fake(option_iv={"AAPL": 0.3})
    assert Engine._suggest(fake, 0, "AAPL", make_pred(wide, 0.6, 0.6))["straddle"] is True
    assert Engine._suggest(fake, 0, "AAPL", make_pred(narrow, 0.6, 0.6))["straddle"] is False   # 160 bps wide, but under the implied move
    blind = make_fake()                                                   # no chain: the width rule as before
    assert Engine._suggest(blind, 0, "AAPL", make_pred(narrow, 0.6, 0.6))["straddle"] is True
    assert Engine._suggest(blind, 0, "AAPL", make_pred([-40, -10, 0, 10, 40], 0.6, 0.6))["straddle"] is False
