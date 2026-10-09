"""A backtest sizes its net to the memory actually free and never runs two training peaks
in one box. Exists because the live 5XL and the backtest's 5XL took the Spark down on
2026-10-10 (NVRM out of memory, reboot)."""
from flint.autotune import PRESETS
from flint.engine import backtest_preset_for

LIVE_D, LIVE_PEAK = 768, 51.0          # the 5XL rung measured at 232 names


def name(p):
    return p[0] if p else None


def test_plenty_of_memory_keeps_the_live_preset():
    assert name(backtest_preset_for(110.0, LIVE_D, LIVE_PEAK)) == "5XL"
    assert name(backtest_preset_for(None, LIVE_D, LIVE_PEAK)) == "5XL"      # no figure (CPU box): the live preset


def test_short_memory_steps_down_the_ladder():
    assert name(backtest_preset_for(60.0, LIVE_D, LIVE_PEAK)) == "4XL"      # 51*1.25+10 = 74 > 60; 34*1.25+10 = 52 fits
    assert name(backtest_preset_for(20.0, LIVE_D, LIVE_PEAK)) == "M"
    assert backtest_preset_for(5.0, LIVE_D, LIVE_PEAK) is None


def test_never_wider_than_live():
    assert name(backtest_preset_for(200.0, 512, 34.0)) == "4XL"


def test_floor_is_respected():
    assert name(backtest_preset_for(60.0, LIVE_D, LIVE_PEAK, floor=30.0)) == "XXL"       # 25.5*1.25+30 = 62 > 60; 17*1.25+30 = 51 fits
    assert [p[0] for p in PRESETS][-1] == "5XL"
