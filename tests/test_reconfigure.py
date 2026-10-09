"""A universe rebuild that fails must not leave the engine dead. On 2026-10-10 "add top 20
movers" re-benchmarked next to a running backtest, the memory guard refused, and every loop
stayed cancelled with status "starting" until a restart."""
import asyncio
from types import SimpleNamespace

from flint.engine import Engine


def make(fail: bool):
    log = {"trace": [], "published": 0, "saved": [], "built": [], "runs": 0, "cache": 0}

    async def stop():
        pass

    async def run():
        log["runs"] += 1

    def build(symbols):
        log["built"].append(list(symbols))
        if fail:
            raise MemoryError("not enough memory to benchmark preset M")
        e.symbols = list(symbols)
        e.base = {s: s for s in symbols}

    e = SimpleNamespace(
        _reconfiguring=False, _backtesting=False, _backtest_stop=False, muted=set(),
        symbols=["A", "B"], all_symbols=["A", "B", "C"], base={"A": "A", "B": "B"}, _built_universe=["A", "B"], _interesting={"C": {"since": "2026-10-10", "last_hit": "2026-10-10"}},
        cfg=SimpleNamespace(symbols=["A", "B"]), _tasks=[], _lifecycle=None, status="live",
        sources=SimpleNamespace(stop=stop), learner=SimpleNamespace(release_cache=lambda: log.__setitem__("cache", log["cache"] + 1),
                                                                     release_replay=lambda: log.__setitem__("released", True)),
        trace=SimpleNamespace(emit=lambda ch, text, lvl="info": log["trace"].append((lvl, text))),
        _publish=lambda m: log.__setitem__("published", log["published"] + 1), snapshot=lambda: {},
        _save=lambda final=False: log["saved"].append(final), run=run, log=log,
    )
    e._active = lambda: [s for s in e.all_symbols if s not in e.muted]
    e._build = build
    e.reconfigure = Engine.reconfigure.__get__(e)
    return e


def test_failed_rebuild_keeps_previous_universe_and_resumes():
    e = make(fail=True)
    asyncio.run(e.reconfigure())
    assert e.symbols == ["A", "B"] and e.cfg.symbols == ["A", "B"]
    assert e.all_symbols == ["A", "B"] and e._interesting == {}      # the full list the model matches, not the one that failed
    assert e._lifecycle is not None                                   # the engine runs on
    assert any(lvl == "error" and "rebuild failed" in t for lvl, t in e.log["trace"])
    assert e.log["saved"] == [True] and e.log["cache"] == 1          # saved first, cache released for the benchmark
    assert not e._reconfiguring


def test_successful_rebuild_still_works():
    e = make(fail=False)
    asyncio.run(e.reconfigure())
    assert e.symbols == ["A", "B", "C"] and e.log["built"] == [["A", "B", "C"]] and e._built_universe == ["A", "B", "C"]
    assert any("universe rebuilt" in t for _, t in e.log["trace"]) and e._lifecycle is not None


def test_rebuild_stops_a_running_backtest_first():
    e = make(fail=False)
    e._backtesting = True

    async def go():
        async def release():
            await asyncio.sleep(1.2)
            e._backtesting = False
        asyncio.get_running_loop().create_task(release())
        await e.reconfigure()

    asyncio.run(go())
    assert e._backtest_stop is True and e.symbols == ["A", "B", "C"]
