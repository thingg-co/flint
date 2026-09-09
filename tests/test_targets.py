"""Tests for Engine._targets, _straddle_plan and _seed_targets: a closed market is not a signal."""
import types

from flint.config import Config
from flint.engine import Engine
from flint.paper import PaperBook


def make_engine(latest, last_target=None, held_since=None, session_bars=0, stock_open=True, options_open=True,
                hold_through_close=True, min_hold=6, straddles=None):
    cfg = Config()
    cfg.hold_through_close = hold_through_close
    cfg.stock_min_hold_bars = min_hold
    cfg.straddle_budget = 0.01
    cfg.straddle_max = 3
    book = PaperBook(start=100_000.0)
    for s in straddles or ():
        book.straddles[s] = {"opened": 0.0}
    return types.SimpleNamespace(
        cfg=cfg, symbols=list(latest), latest=latest, paper=book,
        _last_target=dict(last_target or {}), _held_since=dict(held_since or {}), _session_bars=session_bars,
        _stock_session=lambda: stock_open, _regular_session=lambda: options_open,
    )


def sug(side, size=0.5, score=None, straddle=False, width=0.0):
    return {"side": side, "size": size, "score": score if score is not None else float(side), "straddle": straddle,
            "width": width}


class TestTargets:
    def test_closed_market_carries_a_long(self):
        # 16:05 ET: the policy says HOLD (outside trading hours), the book holds AAPL
        e = make_engine({"AAPL": sug(0, 0.0, score=0.1)}, last_target={"AAPL": 0.5}, stock_open=False, options_open=False)
        targets, frozen = Engine._targets(e, tradeable=True)
        assert targets == {"AAPL": 0.5}
        assert frozen == {"AAPL"}

    def test_closed_market_carries_even_when_the_model_loses_trust(self):
        e = make_engine({"AAPL": sug(0, 0.0)}, last_target={"AAPL": 0.5}, stock_open=False, options_open=False)
        targets, frozen = Engine._targets(e, tradeable=False)
        assert targets == {"AAPL": 0.5} and frozen == {"AAPL"}

    def test_put_carries_in_extended_hours(self):
        # stock trades 16:00-20:00 with extended hours on; options do not, so a put is left alone
        e = make_engine({"AAPL": sug(0, 0.0, score=0.2), "MSFT": sug(0, 0.0, score=0.2)},
                        last_target={"AAPL": -0.3, "MSFT": 0.4}, session_bars=50, held_since={"AAPL": 0, "MSFT": 0},
                        stock_open=True, options_open=False)
        targets, frozen = Engine._targets(e, tradeable=True)
        assert targets["AAPL"] == -0.3 and "AAPL" in frozen
        assert targets["MSFT"] == 0.0 and "MSFT" not in frozen      # stock is live: a faded signal past min hold exits

    def test_hold_through_close_off_flattens(self):
        e = make_engine({"AAPL": sug(0, 0.0, score=0.1)}, last_target={"AAPL": 0.5}, stock_open=False, options_open=False,
                        hold_through_close=False, session_bars=50, held_since={"AAPL": 0})
        targets, frozen = Engine._targets(e, tradeable=True)
        assert targets == {"AAPL": 0.0} and frozen == set()

    def test_no_entries_while_closed(self):
        e = make_engine({"AAPL": sug(1, 0.5)}, stock_open=False, options_open=False)
        targets, frozen = Engine._targets(e, tradeable=True)
        # _suggest already says HOLD outside hours; even a stray BUY here opens nothing frozen
        assert frozen == set()
        assert e._session_bars == 0

    def test_min_hold_counts_session_bars(self):
        # entered at session bar 10; two session bars later the signal fades: still held
        e = make_engine({"AAPL": sug(0, 0.0, score=0.1)}, last_target={"AAPL": 0.5}, held_since={"AAPL": 10}, session_bars=11)
        targets, _ = Engine._targets(e, tradeable=True)
        assert e._session_bars == 12 and targets == {"AAPL": 0.5}
        # six session bars in, the fade closes it
        e._session_bars = 15
        targets, _ = Engine._targets(e, tradeable=True)
        assert targets == {"AAPL": 0.0}

    def test_reversal_closes_inside_min_hold(self):
        e = make_engine({"AAPL": sug(0, 0.0, score=-0.4)}, last_target={"AAPL": 0.5}, held_since={"AAPL": 10}, session_bars=10)
        targets, _ = Engine._targets(e, tradeable=True)
        assert targets == {"AAPL": 0.0}

    def test_entry_records_session_bar(self):
        e = make_engine({"AAPL": sug(1, 0.5)}, session_bars=7)
        targets, _ = Engine._targets(e, tradeable=True)
        assert targets == {"AAPL": 0.5} and e._held_since == {"AAPL": 8} and e._last_target == {"AAPL": 0.5}


class TestStraddlePlan:
    def test_open_straddles_keep_their_budget_after_hours(self):
        e = make_engine({"AAPL": sug(0, 0.0), "MSFT": sug(0, 0.0, straddle=True, width=200)},
                        straddles=["AAPL"], options_open=False)
        targets, new = Engine._straddle_plan(e, vol_tradeable=True)
        assert targets == {"AAPL": 0.01} and new == []

    def test_after_hours_with_hold_off_closes(self):
        e = make_engine({"AAPL": sug(0, 0.0)}, straddles=["AAPL"], options_open=False, hold_through_close=False)
        assert Engine._straddle_plan(e, vol_tradeable=True) == ({}, [])

    def test_in_session_plan_keeps_flagged_and_adds_room(self):
        e = make_engine({"AAPL": sug(0, 0.0, straddle=True, width=150), "MSFT": sug(0, 0.0, straddle=True, width=300),
                         "NVDA": sug(0, 0.0)}, straddles=["AAPL", "NVDA"])
        targets, new = Engine._straddle_plan(e, vol_tradeable=True)
        assert targets == {"AAPL": 0.01, "MSFT": 0.01} and new == ["MSFT"]      # NVDA lost its signal: no budget, so it closes

    def test_in_session_untrusted_bands_close_everything(self):
        e = make_engine({"AAPL": sug(0, 0.0, straddle=True, width=150)}, straddles=["AAPL"])
        assert Engine._straddle_plan(e, vol_tradeable=False) == ({}, [])


class TestSeedTargets:
    def test_restored_positions_get_their_current_weight(self):
        book = PaperBook(start=100_000.0)
        book.pos = {"AAPL": 50.0}
        book.last = {"AAPL": 200.0}          # $10k of $100k -> 0.10
        book.cash = 90_000.0
        book.puts = {"MSFT": {"contracts": 1.0, "strike": 400.0, "expiry_ts": 2_000_000_000.0, "entry_prem": 5.0,
                              "iv": 0.3, "opened": 0.0}}
        e = types.SimpleNamespace(paper=book, _last_target={})
        Engine._seed_targets(e)
        assert abs(e._last_target["AAPL"] - 0.10) < 0.01
        assert e._last_target["MSFT"] < 0

    def test_seeded_long_is_carried_on_the_first_closed_bar(self):
        book = PaperBook(start=100_000.0)
        book.pos = {"AAPL": 50.0}
        book.last = {"AAPL": 200.0}
        book.cash = 90_000.0
        e = make_engine({"AAPL": sug(0, 0.0)}, stock_open=False, options_open=False)
        e.paper = book
        Engine._seed_targets(e)
        targets, frozen = Engine._targets(e, tradeable=False)
        assert frozen == {"AAPL"} and targets["AAPL"] > 0
